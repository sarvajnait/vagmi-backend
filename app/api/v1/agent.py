"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from app.schemas import *
from app.models import *
from langchain_core.messages import AIMessage
from app.core.langgraph.graph import EducationPlatform
from app.services.database import get_session
from loguru import logger

router = APIRouter()
platform = EducationPlatform()
COLLECTION_NAME = "education_documents"


async def get_hierarchy_names(
    class_level_id: int,
    board_id: int,
    medium_id: int,
    subject_id: int,
    chapter_id: int = None,
    session: Session = None,
) -> dict:
    """Helper function to get names from IDs for better logging and context"""
    names = {}

    if session:
        try:
            class_level = session.get(ClassLevel, class_level_id)
            board = session.get(Board, board_id)
            medium = session.get(Medium, medium_id)
            subject = session.get(Subject, subject_id)

            names["class_level"] = (
                class_level.name if class_level else str(class_level_id)
            )
            names["board"] = board.name if board else str(board_id)
            names["medium"] = medium.name if medium else str(medium_id)
            names["subject"] = subject.name if subject else str(subject_id)

            if chapter_id:
                chapter = session.get(Chapter, chapter_id)
                names["chapter"] = chapter.name if chapter else str(chapter_id)
        except Exception as e:
            logger.warning(f"Could not fetch hierarchy names: {e}")

    return names


@router.post("/stream-chat")
async def stream_chat(
    chat_request: ChatRequest, session: Session = Depends(get_session)
):
    """Stream chat responses with ID-based hierarchical filtering"""

    async def generate_response():
        try:
            # Build filters using IDs (stored as strings in vector store metadata)
            filters = {
                "class_level_id": str(chat_request.class_level_id),
                "board_id": str(chat_request.board_id),
                "medium_id": str(chat_request.medium_id),
                "subject_id": str(chat_request.subject_id),
            }

            if chat_request.chapter_id:
                filters["chapter_id"] = str(chat_request.chapter_id)

            # Get names for better context (optional but recommended)
            names = await get_hierarchy_names(
                chat_request.class_level_id,
                chat_request.board_id,
                chat_request.medium_id,
                chat_request.subject_id,
                chat_request.chapter_id,
                session,
            )

            logger.info(f"Chat request - Filters: {filters}, Context: {names}")

            # Create RAG graph with ID-based filters
            rag_graph = platform.create_rag_graph(filters, names)

            # Generate thread config using IDs for consistency
            filter_key = f"{chat_request.class_level_id}_{chat_request.board_id}_{chat_request.medium_id}_{chat_request.subject_id}"
            if chat_request.chapter_id:
                filter_key += f"_{chat_request.chapter_id}"

            config = {"configurable": {"thread_id": f"session_{filter_key}"}}

            input_messages = [{"role": "user", "content": chat_request.message}]

            # Stream response
            async for event in rag_graph.astream(
                {"messages": input_messages, "query": chat_request.message},
                config=config,
                stream_mode="messages",
            ):
                message, metadata = event
                if isinstance(message, AIMessage) and message.content:
                    for chunk in message.content:
                        if chunk:
                            yield f"data: {json.dumps({'type': 'token', 'content': str(chunk)})}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )
