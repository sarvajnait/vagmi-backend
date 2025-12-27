"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from app.schemas import *
from app.models import *
from langchain_core.messages import AIMessage
from langchain_core.callbacks import get_usage_metadata_callback
from app.core.agents.graph import EducationPlatform
from app.services.database import get_session
from app.services.llm_usage import get_user_daily_total, record_usage_metadata
from app.core.config import settings
from loguru import logger
from app.api.v1.auth import get_current_user

router = APIRouter()
platform = EducationPlatform()


async def get_hierarchy_names(
    class_level_id: int,
    board_id: int,
    medium_id: int,
    subject_id: int,
    chapter_id: int = None,
    session: AsyncSession = None,
) -> dict:
    """Helper function to get names from IDs for better logging and context"""
    names = {}

    if session:
        try:
            class_level = await session.get(ClassLevel, class_level_id)
            board = await session.get(Board, board_id)
            medium = await session.get(Medium, medium_id)
            subject = await session.get(Subject, subject_id)

            names["class_level"] = (
                class_level.name if class_level else str(class_level_id)
            )
            names["board"] = board.name if board else str(board_id)
            names["medium"] = medium.name if medium else str(medium_id)
            names["subject"] = subject.name if subject else str(subject_id)

            if chapter_id:
                chapter = await session.get(Chapter, chapter_id)
                names["chapter"] = chapter.name if chapter else str(chapter_id)

        except Exception as e:
            logger.warning(f"Could not fetch hierarchy names: {e}")

    return names


@router.post("/stream-chat")
async def stream_chat(
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Stream chat responses with hierarchical filtering and image support."""
    if chat_request.user_id is not None and chat_request.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match token")
    user_id = current_user.id

    async def generate_response():
        try:
            daily_total = await get_user_daily_total(session, user_id)
            if daily_total >= settings.DAILY_TOKEN_LIMIT:
                message = "Daily token limit reached. Please try again tomorrow."
                yield f"data: {json.dumps({'type': 'token', 'content': message})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"
                return

            # -----------------------------
            # Prepare filters and context
            # -----------------------------
            filters = {
                "class_level_id": str(chat_request.class_level_id),
                "board_id": str(chat_request.board_id),
                "medium_id": str(chat_request.medium_id),
                "subject_id": str(chat_request.subject_id),
            }

            if chat_request.chapter_id:
                filters["chapter_id"] = str(chat_request.chapter_id)

            names = await get_hierarchy_names(
                chat_request.class_level_id,
                chat_request.board_id,
                chat_request.medium_id,
                chat_request.subject_id,
                chat_request.chapter_id,
                session,
            )

            logger.info(f"Chat request - Filters: {filters}, Context: {names}")

            additional_notes_content = ""
            if chat_request.chapter_id:
                _result = await session.exec(
                    select(AdditionalNotes).where(
                        AdditionalNotes.chapter_id == chat_request.chapter_id
                    ))
                notes = _result.all()
                if notes:
                    additional_notes_content = (
                        "\n\n**Important Teacher Context (Must Integrate):**\n"
                    )
                    for note in notes:
                        additional_notes_content += f"- {note.note}\n"

            rag_graph = platform.create_educational_agent(
                filters,
                names,
                additional_notes_content=additional_notes_content,
            )

            filter_key = (
                f"{chat_request.class_level_id}_"
                f"{chat_request.board_id}_"
                f"{chat_request.medium_id}_"
                f"{chat_request.subject_id}"
            )
            if chat_request.chapter_id:
                filter_key += f"_{chat_request.chapter_id}"

            config = {"configurable": {"thread_id": f"session_{filter_key}"}}

            input_messages = [{"role": "user", "content": chat_request.message}]

            # -----------------------------
            # Image state (per request)
            # -----------------------------
            retrieved_images_metadata = []
            selected_images = []      # ✅ FIX: always defined
            images_streamed = False

            # -----------------------------
            # Stream events
            # -----------------------------
            with get_usage_metadata_callback() as usage_cb:
                config["callbacks"] = [usage_cb]
                async for event in rag_graph.astream_events(
                    {"messages": input_messages, "query": chat_request.message},
                    config=config,
                    version="v2",
                ):
                    if not isinstance(event, dict):
                        continue

                    event_type = event.get("event")
                    name = event.get("name", "")
                    data = event.get("data", {})

                    # -----------------------------
                    # IMAGE RETRIEVAL
                    # -----------------------------
                    if (
                        event_type == "on_tool_end"
                        and "retrieve_textbook_with_images" in name
                    ):
                        output = data.get("output")

                        if hasattr(output, "artifact") and output.artifact:
                            artifact = output.artifact
                            retrieved_images_metadata = artifact.get("image_metadata", [])
                            selected_images = []     # reset per retrieval
                            images_streamed = False

                    # -----------------------------
                    # IMAGE SELECTION
                    # -----------------------------
                    elif event_type == "on_tool_end" and "select_relevant_images" in name:
                        output = data.get("output")

                        if hasattr(output, "artifact") and output.artifact:
                            artifact = output.artifact

                            if artifact.get("action") == "select_images":
                                selected_ids = artifact.get("selected_image_ids", [])

                                if not retrieved_images_metadata:
                                    continue

                                selected_ids_str = {str(i) for i in selected_ids}

                                selected_images = [
                                    img
                                    for img in retrieved_images_metadata
                                    if str(img.get("image_id")) in selected_ids_str
                                ]

                                if selected_images and not images_streamed:
                                    for image in selected_images:
                                        yield (
                                            f"data: {json.dumps({'type': 'image', 'data': image})}\n\n"
                                        )

                                    images_streamed = True
                                    retrieved_images_metadata = []

                    # -----------------------------
                    # STREAM CLEAN TEXT TOKENS
                    # -----------------------------
                    elif event_type == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        if not chunk:
                            continue

                        # Skip tool calls
                        if getattr(chunk, "tool_calls", None):
                            continue

                        content = chunk.content

                        if isinstance(content, str):
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                        elif isinstance(content, list):
                            for part in content:
                                if (
                                    isinstance(part, dict)
                                    and part.get("type") == "text"
                                    and part.get("text")
                                ):
                                    yield f"data: {json.dumps({'type': 'token', 'content': part['text']})}\n\n"

            # -----------------------------
            # Update checkpoint with images
            # -----------------------------
            try:
                state_snapshot = rag_graph.get_state(config)
                messages = state_snapshot.values.get("messages", [])

                if selected_images:
                    for i in range(len(messages) - 1, -1, -1):
                        if isinstance(messages[i], AIMessage):
                            messages[i].additional_kwargs[
                                "selected_images"
                            ] = selected_images
                            break

                    rag_graph.update_state(config, {"messages": messages})

            except Exception as e:
                logger.error(f"Error updating checkpoint: {e}", exc_info=True)

            if usage_cb.usage_metadata:
                await record_usage_metadata(
                    session, user_id, usage_cb.usage_metadata
                )

            yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"

        except Exception as e:
            logger.exception("Error in stream chat")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/usage")
async def get_usage(
    user_id: int | None = Query(None, description="User ID to check usage for"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if user_id is not None and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match token")
    resolved_user_id = user_id or current_user.id
    total = await get_user_daily_total(session, resolved_user_id)
    limit = settings.DAILY_TOKEN_LIMIT
    return {
        "user_id": resolved_user_id,
        "daily_total_tokens": total,
        "daily_limit_tokens": limit,
        "remaining_tokens": max(limit - total, 0),
        "limit_reached": total >= limit,
    }
