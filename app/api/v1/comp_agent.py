import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from langchain_core.callbacks import get_usage_metadata_callback
from app.schemas.chat import CompChatRequest
from app.models.competitive_hierarchy import CompSubject, CompChapter
from app.models.comp_llm_resources import CompAdditionalNote
from app.models.user import User
from app.core.agents.comp_graph import CompEducationPlatform
from app.services.database import get_session
from app.services.llm_usage import get_user_daily_total, record_usage_metadata
from app.core.config import settings
from loguru import logger
from app.api.v1.auth import get_current_user

router = APIRouter()
comp_platform = CompEducationPlatform()


@router.post("/stream-chat")
async def comp_stream_chat(
    chat_request: CompChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Stream competitive exam RAG chat responses backed by comp_llm_textbooks."""
    user_id = current_user.id

    subject = await session.get(CompSubject, chat_request.comp_subject_id)
    if not subject:
        raise HTTPException(status_code=404, detail="Comp subject not found")

    chapter = None
    if chat_request.comp_chapter_id is not None:
        chapter = await session.get(CompChapter, chat_request.comp_chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Comp chapter not found")
        if chapter.subject_id != subject.id:
            raise HTTPException(status_code=400, detail="Chapter does not belong to subject")

    async def generate_response():
        try:
            daily_total = await get_user_daily_total(session, user_id)
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
                _result = await session.exec(
                    select(CompAdditionalNote).where(
                        CompAdditionalNote.comp_chapter_id == chapter.id
                    )
                )
                notes = _result.all()
                if notes:
                    additional_notes_content = "\n\n**Important Teacher Context (Must Integrate):**\n"
                    for note in notes:
                        additional_notes_content += f"- {note.note}\n"

            logger.info(f"Comp chat - user={user_id} filters={filters} names={names}")

            rag_graph = comp_platform.create_comp_agent(
                filters,
                names,
                additional_notes_content=additional_notes_content,
            )

            thread_id = f"comp_session_{chapter.id if chapter else subject.id}"
            config = {"configurable": {"thread_id": thread_id}}
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
                await record_usage_metadata(session, user_id, usage_cb.usage_metadata)

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
