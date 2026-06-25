import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, AIMessage

from app.schemas.comp_chat import (
    CompChatRequest,
    SessionCreateRequest,
    SessionResponse,
    SessionHistoryResponse,
    MessageOut,
)
from app.models.competitive_hierarchy import CompSubject, CompChapter
from app.models.comp_llm_resources import CompAdditionalNote
from app.models.comp_chat import CompChatSession
from app.models.user import User
from app.core.agents.comp_graph import CompEducationPlatform
from app.services.database import get_session
from app.services.llm_usage import get_user_daily_total, record_usage_metadata
from app.core.config import settings
from loguru import logger
from app.api.v1.auth import get_current_user

router = APIRouter()
comp_platform = CompEducationPlatform()


def _get_checkpointer(request: Request):
    return request.app.state.comp_checkpointer


def _session_to_response(s: CompChatSession) -> SessionResponse:
    return SessionResponse(
        id=s.id,
        comp_subject_id=s.comp_subject_id,
        comp_chapter_id=s.comp_chapter_id,
        title=s.title,
        thread_id=s.thread_id,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session CRUD
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    subject = await db.get(CompSubject, body.comp_subject_id)
    if not subject:
        raise HTTPException(status_code=404, detail="Comp subject not found")

    if body.comp_chapter_id is not None:
        chapter = await db.get(CompChapter, body.comp_chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Comp chapter not found")
        if chapter.subject_id != subject.id:
            raise HTTPException(status_code=400, detail="Chapter does not belong to subject")

    thread_id = f"comp_u{current_user.id}_s{uuid.uuid4().hex}"

    session = CompChatSession(
        user_id=current_user.id,
        comp_subject_id=body.comp_subject_id,
        comp_chapter_id=body.comp_chapter_id,
        title="",
        thread_id=thread_id,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return _session_to_response(session)


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.exec(
        select(CompChatSession)
        .where(CompChatSession.user_id == current_user.id)
        .order_by(CompChatSession.updated_at.desc())
    )
    sessions = result.all()
    return [_session_to_response(s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    session = await db.get(CompChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    checkpointer = _get_checkpointer(request)
    config = {"configurable": {"thread_id": session.thread_id}}

    def _extract_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return ""

    messages: list[MessageOut] = []
    try:
        checkpoint = await checkpointer.aget(config)
        if checkpoint:
            raw_messages = checkpoint.get("channel_values", {}).get("messages", [])
            for msg in raw_messages:
                if isinstance(msg, HumanMessage):
                    content = _extract_content(msg.content)
                    messages.append(MessageOut(role="human", content=content))
                elif isinstance(msg, AIMessage):
                    content = _extract_content(msg.content)
                    if content:  # skip tool-call-only messages with no text
                        messages.append(MessageOut(role="ai", content=content))
    except Exception as e:
        logger.warning(f"Could not load checkpoint for session {session_id}: {e}")

    return SessionHistoryResponse(
        session=_session_to_response(session),
        messages=messages,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    session = await db.get(CompChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    await db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Streaming chat
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/stream-chat")
async def comp_stream_chat(
    chat_request: CompChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Stream competitive exam RAG chat responses backed by comp_llm_textbooks."""
    user_id = current_user.id

    session = await db.get(CompChatSession, chat_request.session_id)
    if not session or session.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    subject = await db.get(CompSubject, session.comp_subject_id)
    if not subject:
        raise HTTPException(status_code=404, detail="Comp subject not found")

    chapter = None
    if session.comp_chapter_id is not None:
        chapter = await db.get(CompChapter, session.comp_chapter_id)

    # Set session title from first message
    if not session.title and chat_request.message.strip():
        session.title = chat_request.message.strip()[:80]
        db.add(session)
        await db.commit()

    checkpointer = _get_checkpointer(request)

    async def generate_response():
        try:
            daily_total = await get_user_daily_total(db, user_id)
            if daily_total >= settings.DAILY_TOKEN_LIMIT:
                msg = "Daily token limit reached. Please try again tomorrow."
                yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"
                return

            filters = {}
            if chapter:
                filters["chapter_id"] = str(chapter.id)

            names = {"subject": subject.name}
            if chapter:
                names["chapter"] = chapter.name

            additional_notes_content = ""
            if chapter:
                _result = await db.exec(
                    select(CompAdditionalNote).where(
                        CompAdditionalNote.comp_chapter_id == chapter.id
                    )
                )
                notes = _result.all()
                if notes:
                    additional_notes_content = "\n\n**Important Teacher Context (Must Integrate):**\n"
                    for note in notes:
                        additional_notes_content += f"- {note.note}\n"

            logger.info(
                f"Comp chat - user={user_id} session={session.id} filters={filters} names={names}"
            )

            rag_graph = comp_platform.create_comp_agent(
                filters,
                names,
                additional_notes_content=additional_notes_content,
                checkpointer=checkpointer,
            )

            config = {"configurable": {"thread_id": session.thread_id}}
            input_messages = [{"role": "user", "content": chat_request.message}]

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
                    data = event.get("data", {})

                    if event_type == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        if not chunk:
                            continue
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

            if usage_cb.usage_metadata:
                await record_usage_metadata(db, user_id, usage_cb.usage_metadata)

            # Touch updated_at
            session.updated_at = datetime.now(timezone.utc)
            db.add(session)
            await db.commit()

            yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"

        except Exception as e:
            logger.exception("Error in comp stream chat")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
