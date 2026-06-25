from langchain.agents import create_agent
from langchain.tools import tool
from langchain_postgres import PGVector
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Dict, Optional
from loguru import logger
from langgraph.checkpoint.memory import MemorySaver

CONNECTION_STRING = settings.POSTGRES_URL
COMP_COLLECTION_TEXTBOOKS = "comp_llm_textbooks"

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


class CompEducationPlatform:
    """LangChain agent for competitive exam RAG chat, backed by comp_llm_textbooks."""

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
            return content, {"documents": docs, "source": "comp_textbook"}

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

        agent = create_agent(
            model=comp_llm,
            tools=[retrieve_textbook],
            system_prompt=system_prompt,
            checkpointer=checkpointer or comp_memory_checkpointer,
        )

        return agent
