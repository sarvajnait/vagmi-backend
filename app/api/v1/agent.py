"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas import *
from langchain_core.messages import AIMessage
from app.core.langgraph.graph import EducationPlatform
from loguru import logger

router = APIRouter()
platform = EducationPlatform()
COLLECTION_NAME = "education_documents"
router = APIRouter()


@router.post("/stream-chat")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses with hierarchical filtering"""

    async def generate_response():
        try:
            # Build filters
            filters = {}
            if chat_request.class_level:
                filters["class_level"] = chat_request.class_level
            if chat_request.board:
                filters["board"] = chat_request.board
            if chat_request.medium:
                filters["medium"] = chat_request.medium
            if chat_request.subject:
                filters["subject"] = chat_request.subject
            if chat_request.chapter:
                filters["chapter"] = chat_request.chapter

            # Create RAG graph with filters
            rag_graph = platform.create_rag_graph(filters)

            # Generate thread config
            filter_key = "_".join([f"{k}_{v}" for k, v in filters.items() if v])
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
