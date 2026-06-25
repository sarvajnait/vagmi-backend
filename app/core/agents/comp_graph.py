from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from loguru import logger

from app.core.config import settings

CONNECTION_STRING = settings.POSTGRES_URL
COMP_COLLECTION_TEXTBOOKS = "comp_llm_textbooks"

MAX_CONTEXT_MESSAGES = 20       # max non-tool messages kept per model call
TOOL_RESULT_MAX_CHARS = 15_000  # cap retrieval output sent to the model
FILLER_TOOL_MESSAGE = "EMPTY: tool call not completed"

comp_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", output_dimensionality=768
)

comp_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=False,
    temperature=0.3,
    streaming=True,
)

comp_vector_store_textbooks = PGVector(
    embeddings=comp_embeddings,
    collection_name=COMP_COLLECTION_TEXTBOOKS,
    connection=CONNECTION_STRING,
)

comp_memory_checkpointer = MemorySaver()


# ─────────────────────────────────────────────────────────────────────────────
# Context engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def drop_leading_non_human(messages: list[Any]) -> list[Any]:
    """Drop messages until the first HumanMessage (safe start for any model)."""
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            return messages[idx:]
    return []


def trim_messages_for_model(messages: list[Any], max_messages: int) -> list[Any]:
    """Keep the most recent `max_messages` non-tool messages.

    ToolMessages are pulled in alongside their paired AIMessage so no tool call
    is ever left without a result when the slice reaches the model.
    """
    if not messages or max_messages <= 0:
        return messages or []
    if len(messages) <= max_messages:
        return messages

    included: set[int] = set()
    tool_idxs_by_call_id: dict[str, list[int]] = {}
    count = 0

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]

        if isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None)
            if call_id:
                tool_idxs_by_call_id.setdefault(call_id, []).append(idx)
            continue

        if count >= max_messages:
            continue

        included.add(idx)
        count += 1

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id = tc.get("id")
                if call_id:
                    included.update(tool_idxs_by_call_id.get(call_id, []))

    trimmed = [messages[i] for i in sorted(included)]
    return drop_leading_non_human(trimmed)


def fix_orphaned_tool_calls(
    messages: list[Any],
) -> tuple[list[RemoveMessage], list[ToolMessage]]:
    """Find AIMessages whose tool_calls have no matching ToolMessage.

    Returns (to_remove, fillers) — callers add both to the state so the
    checkpointer no longer holds an invalid tool-call/result pair.
    """
    answered_ids: set[str] = {
        getattr(m, "tool_call_id", None)
        for m in messages
        if isinstance(m, ToolMessage)
    } - {None}

    to_remove: list[RemoveMessage] = []
    fillers: list[ToolMessage] = []
    ts = datetime.now(timezone.utc).isoformat()

    for msg in messages:
        if not (isinstance(msg, AIMessage) and msg.tool_calls):
            continue
        for tc in msg.tool_calls:
            tc_id = tc.get("id")
            if tc_id and tc_id not in answered_ids:
                to_remove.append(RemoveMessage(id=msg.id))
                fillers.append(
                    ToolMessage(
                        content=FILLER_TOOL_MESSAGE,
                        tool_call_id=tc_id,
                        name=tc.get("name", "unknown_tool"),
                        additional_kwargs={"created_at": ts, "isFiller": True},
                    )
                )
                break  # one RemoveMessage per AIMessage is enough

    return to_remove, fillers


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

class CompEducationPlatform:
    """Competitive exam RAG chat agent backed by comp_llm_textbooks."""

    def __init__(self):
        self.vector_store = comp_vector_store_textbooks

    def create_comp_agent(
        self,
        filters: Dict[str, str],
        names: Optional[Dict[str, str]] = None,
        additional_notes_content: str = "",
        checkpointer=None,
    ):
        names = names or {}

        context_parts = []
        if names.get("exam"):
            context_parts.append(names["exam"])
        if names.get("subject"):
            context_parts.append(names["subject"])
        if names.get("chapter"):
            context_parts.append(f"Chapter: {names['chapter']}")
        filter_desc = " → ".join(context_parts) if context_parts else "Competitive Exam"

        # ── Tool ─────────────────────────────────────────────────────────────

        @tool(response_format="content_and_artifact")
        def retrieve_textbook(query: str):
            """Retrieve relevant content from the competitive exam textbook.
            Use for explaining concepts, definitions, processes, and topic understanding."""
            metadata_filter = (
                {"chapter_id": str(filters["chapter_id"])}
                if filters.get("chapter_id")
                else {}
            )
            docs = self.vector_store.similarity_search(query, k=5, filter=metadata_filter)
            if not docs:
                return "No content found for this query.", {"documents": [], "source": "comp_textbook"}

            content = "\n\n".join(f"📚 **Content**: {d.page_content}" for d in docs)

            if len(content) > TOOL_RESULT_MAX_CHARS:
                content = (
                    content[:TOOL_RESULT_MAX_CHARS]
                    + f"\n\n[TRUNCATED: result exceeded {TOOL_RESULT_MAX_CHARS} chars]"
                )

            return content, {"documents": docs, "source": "comp_textbook"}

        # ── System prompt ─────────────────────────────────────────────────────

        system_prompt = f"""
You are VAGMI, an AI tutor for competitive exam preparation (RRB, SSC, UPSC, and similar exams).
You ONLY teach from the officially uploaded textbook content for the chosen exam, subject, and chapter.
You explain concepts clearly and accurately in exam-focused language.

════════════════════════════════════
🎯 CONTEXT
{filter_desc}
{additional_notes_content}
════════════════════════════════════

## ⛔ CRITICAL RULES (READ FIRST)
1. **NO OUTSIDE KNOWLEDGE:** You must rely exclusively on what the `retrieve_textbook` tool returns. Do not add facts from your training data about the specific topic.
2. **MANDATORY TOOL USE:** You are FORBIDDEN from answering any question about concepts, definitions, or topics without first calling `retrieve_textbook`.
3. **NO GUESSING:** If the tool returns no information, say so honestly: "I cannot find information on that specific topic in the uploaded material for this chapter."

## 🧠 HOW YOU SHOULD THINK

1. **Understand the student's intent**
   - Explanation or concept → use `retrieve_textbook`
   - Short answer for a topic → use `retrieve_textbook`
   - Practice question → use `retrieve_textbook` (may not have Q&A; inform if absent)

2. **Rewrite the query before retrieval**
   - Convert pronouns and vague references to clear, keyword-rich terms
   - Use exam-standard terminology
   - Example: "what about this topic" → "railway recruitment board general science electricity"

3. **Teach clearly**
   - Use headings and bullet points
   - Bold key terms and definitions
   - Provide exam-relevant context (e.g., "this is a common question type in RRB exams")

## 📚 TEXTBOOK TOOL

### Tool: `retrieve_textbook`

Use this for:
- Concept explanations
- Definitions and formulas
- Topic overviews
- Any factual question about the chapter content

Rules:
- Always rewrite the student's question into a precise search query first
- Base your answer on what the tool returns
- Do not fabricate facts not present in the retrieved content

## 🛡️ TRUTH & SAFETY RULES
- If retrieval returns nothing relevant, say so clearly
- Do not invent dates, statistics, formulas, or historical facts
- Do not answer from memory about exam-specific content

## 🗣️ TONE & STYLE
- Professional yet encouraging
- Exam-focused: highlight what is important for competitive exams
- Use structured formatting: headings, bullet points, bold key terms

Start by choosing the right query rewrite, then retrieve and teach.
"""

        # ── Graph nodes ───────────────────────────────────────────────────────

        tools_list = [retrieve_textbook]
        tools_by_name = {t.name: t for t in tools_list}
        bound_model = comp_llm.bind_tools(tools_list)

        def preprocess_node(state: MessagesState) -> dict:
            """Fix orphaned tool calls left in the checkpoint from interrupted turns."""
            messages = list(state["messages"])
            to_remove, fillers = fix_orphaned_tool_calls(messages)
            if not to_remove:
                return {}
            logger.warning(f"comp_agent: fixing {len(to_remove)} orphaned tool call(s)")
            return {"messages": to_remove + fillers}

        async def call_model_node(state: MessagesState) -> dict:
            """Trim history, prepend system prompt, call the model."""
            messages = list(state["messages"])
            trimmed = trim_messages_for_model(messages, MAX_CONTEXT_MESSAGES)
            model_messages = [SystemMessage(content=system_prompt), *trimmed]
            response = await bound_model.ainvoke(model_messages)
            return {"messages": [response]}

        def execute_tools_node(state: MessagesState) -> dict:
            """Execute all tool calls on the last AIMessage."""
            last_msg = state["messages"][-1]
            tool_messages: list[ToolMessage] = []

            for tc in last_msg.tool_calls:
                t = tools_by_name.get(tc["name"])
                if not t:
                    tool_messages.append(ToolMessage(
                        content=f"Unknown tool: {tc['name']}",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    ))
                    continue

                try:
                    result = t.invoke(tc["args"])
                    # response_format="content_and_artifact" returns (content, artifact)
                    content = str(result[0]) if isinstance(result, tuple) else str(result)
                    if len(content) > TOOL_RESULT_MAX_CHARS:
                        content = (
                            content[:TOOL_RESULT_MAX_CHARS]
                            + f"\n\n[TRUNCATED: exceeded {TOOL_RESULT_MAX_CHARS} chars]"
                        )
                    tool_messages.append(ToolMessage(
                        content=content,
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    ))
                except Exception as exc:
                    logger.warning(f"comp_agent: tool '{tc['name']}' error: {exc}")
                    tool_messages.append(ToolMessage(
                        content=f"Tool error: {exc}",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    ))

            return {"messages": tool_messages}

        def should_continue(state: MessagesState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "execute_tools"
            return END

        # ── Compile ───────────────────────────────────────────────────────────

        builder = StateGraph(MessagesState)
        builder.add_node("preprocess", preprocess_node)
        builder.add_node("call_model", call_model_node)
        builder.add_node("execute_tools", execute_tools_node)

        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "call_model")
        builder.add_conditional_edges("call_model", should_continue, ["execute_tools", END])
        builder.add_edge("execute_tools", "call_model")

        return builder.compile(checkpointer=checkpointer or comp_memory_checkpointer)
