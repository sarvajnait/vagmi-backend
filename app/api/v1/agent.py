"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.api.v1.session import get_session_by_id
from app.core.config import settings
from app.core.langgraph.graph import LangGraphAgent
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.session import Session
from app.schemas.agent import ChatResponse, ChatRequest, AgentContext
from app.utils.files import (
    get_or_create_session_dir,
    get_session_files,
    clear_tmp_directory,
)
from app.core.metrics import llm_stream_duration_seconds
from langchain_core.messages import HumanMessage
from app.utils.serialization import custom_default
from app.schemas.streams import *

router = APIRouter()
agent = LangGraphAgent()


@router.post("/run", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def run(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_session_by_id),
):
    """
    Handle a chat request using LangGraph.
    """
    session_path = get_or_create_session_dir(session.id)

    try:
        logger.info(
            "chat_request_received",
            session_id=session.id,
            message=chat_request.query,
        )

        files = get_session_files(session_path, True)

        message = HumanMessage(
            content=chat_request.query,
            additional_kwargs={"created_at": datetime.now().isoformat()},
        )

        context = AgentContext(
            files=files, llm_provider=chat_request.llm_provider, session_id=session.id, user_id=session.user_id
        )

        result = await agent.get_response(message, context)

        logger.info("chat_request_processed", session_id=session.id)
        return ChatResponse(messages=result)

    except Exception as e:
        logger.error("chat_request_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        clear_tmp_directory(session_path)


@router.post("/stream")
async def stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_session_by_id),
):
    session_path = get_or_create_session_dir(session.id)

    try:
        files = get_session_files(session_path, True)

        async def event_generator():
            message = HumanMessage(
                content=chat_request.query,
                additional_kwargs={"created_at": datetime.now().isoformat()},
            )

            context = AgentContext(
                files=files,
                llm_provider=chat_request.llm_provider,
                session_id=session.id,
                user_id=session.user_id,
            )

            async for event in agent.get_stream_response(message, context):
                try:
                    if isinstance(event, BaseModel):
                        payload = event.model_dump_json()
                    else:
                        payload = json.dumps(event, default=custom_default)

                    yield f"data: {payload}\n\n"

                except Exception as inner_e:
                    logger.error("event_serialization_failed", error=str(inner_e))
                    continue

            # Always end cleanly
            yield f"data: {EndOfStreamEvent().model_dump_json()}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error("stream_chat_request_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        clear_tmp_directory(session_path)


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_session_by_id),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(session.id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.error("get_messages_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_session_by_id),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(session.id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(
            "clear_chat_history_failed",
            session_id=session.id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
