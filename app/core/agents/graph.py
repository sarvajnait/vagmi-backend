from langchain.agents import create_agent
from langchain.tools import tool
from langchain_postgres import PGVector
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Dict, Optional
from loguru import logger
from langgraph.checkpoint.memory import MemorySaver

# Database configuration
CONNECTION_STRING = settings.POSTGRES_URL
COLLECTION_NAME_TEXTBOOKS = "llm_textbooks"
COLLECTION_NAME_NOTES = "llm_notes"
COLLECTION_NAME_QA = "qa_patterns"
COLLECTION_NAME_IMAGES = "llm_images"

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=False,
    temperature=0.3,
    streaming=True,
)

# Create separate vector stores
vector_store_textbooks = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME_TEXTBOOKS,
    connection=CONNECTION_STRING,
)

vector_store_notes = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME_NOTES,
    connection=CONNECTION_STRING,
)

vector_store_qa = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME_QA,
    connection=CONNECTION_STRING,
)

vector_store_images = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME_IMAGES,
    connection=CONNECTION_STRING,
)

memory_checkpointer = MemorySaver()


def merge_chunks_remove_overlap(chunks: list[str], overlap_chars: int = 200) -> str:
    if not chunks:
        return ""

    merged = chunks[0]

    for next_chunk in chunks[1:]:
        # Try to find overlap window
        overlap_window = merged[-overlap_chars:]

        if overlap_window and overlap_window in next_chunk:
            merged += next_chunk.split(overlap_window, 1)[-1]
        else:
            merged += "\n\n" + next_chunk

    return merged


# Image selection tool
@tool(response_format="content_and_artifact")
def select_relevant_images(image_ids: list[int]):
    """Select which images to show the student.

    Use this tool to pick the most relevant images from the ones you just retrieved.
    Simply provide the image IDs as a list of numbers.

    Args:
        image_ids: List of image ID numbers, e.g., [3, 5, 7]

    Example: select_relevant_images([3, 5])
    """
    try:
        if not image_ids:
            return "No images selected.", {"selected_images": []}

        logger.info(f"[IMAGE_SELECTION] LLM selected image IDs: {image_ids}")

        return (
            f"Selected {len(image_ids)} image(s) to show the student.",
            {"selected_image_ids": image_ids, "action": "select_images"},
        )

    except Exception as e:
        logger.error(f"[IMAGE_SELECTION] Error: {e}", exc_info=True)
        return f"Error selecting images: {str(e)}", {"selected_images": []}


class EducationPlatform:
    """Main class handling the education platform logic using LangChain v1 Agents"""

    def __init__(self):
        self.vector_store_textbooks = vector_store_textbooks
        self.vector_store_notes = vector_store_notes
        self.vector_store_qa = vector_store_qa
        self.vector_store_images = vector_store_images

    def create_educational_agent(
        self,
        filters: Dict[str, str],
        names: Optional[Dict[str, str]] = None,
        additional_notes_content: str = "",
    ):
        """
        Create a unified Agent with hierarchical context filters.

        This replaces the old manual graph construction with the v1 `create_agent`.
        """
        names = names or {}

        # 1. Prepare Context Strings
        if names:
            context_parts = []
            if names.get("class_level"):
                context_parts.append(f"Class {names['class_level']}")
            if names.get("board"):
                context_parts.append(names["board"])
            if names.get("medium"):
                context_parts.append(f"{names['medium']} Medium")
            if names.get("subject"):
                context_parts.append(names["subject"])
            if names.get("chapter"):
                context_parts.append(f"Chapter: {names['chapter']}")
            filter_desc = " → ".join(context_parts)
            subject_name = names.get("subject", "the subject")
            class_name = names.get("class_level", "")
        else:
            filter_desc = ", ".join([f"{k}: {v}" for k, v in filters.items() if v])
            subject_name = "the subject"
            class_name = ""
        # 2. Additional SQL Notes (pre-fetched by caller)
        additional_notes_content = additional_notes_content or ""

        @tool(response_format="content_and_artifact")
        def retrieve_textbook_with_images(query: str):
            """
            Retrieves core textbook content and related images.
            The agent may choose to ignore images if they are not useful.
            """
            metadata_filter = (
                {"chapter_id": str(filters["chapter_id"])}
                if filters.get("chapter_id")
                else {}
            )

            text_docs = self.vector_store_textbooks.similarity_search(
                query, k=5, filter=metadata_filter
            )

            image_docs = self.vector_store_images.similarity_search(
                query, k=5, filter=metadata_filter
            )

            textbook_content = (
                "\n\n".join(f"📚 **Textbook**: {d.page_content}" for d in text_docs)
                if text_docs
                else "No textbook content found."
            )

            image_metadata = []
            serialized_images = []

            for i, doc in enumerate(image_docs, 1):
                serialized_images.append(
                    f"🖼️ **Image {i}**: {doc.metadata.get('title')} "
                    f"(ID: {doc.metadata.get('image_id')})\n"
                    f"Desc: {doc.metadata.get('description')}"
                )
                image_metadata.append(
                    {
                        "image_id": doc.metadata.get("image_id"),
                        "file_url": doc.metadata.get("file_url"),
                        "title": doc.metadata.get("title"),
                        "description": doc.metadata.get("description"),
                        "tags": doc.metadata.get("tags", []),
                    }
                )

            content = textbook_content
            if serialized_images:
                content += "\n\n🖼️ **Available Visual Aids:**\n" + "\n\n".join(
                    serialized_images
                )

            return content, {
                "source": "core_with_images",
                "documents": text_docs + image_docs,
                "image_metadata": image_metadata,
            }

        @tool(response_format="content_and_artifact")
        def retrieve_full_chapter_textbook():
            """
            Retrieve ALL textbook chunks for the current chapter.
            Use ONLY when the user asks for whole-chapter outputs that require full context
            (e.g., chapter summary/overview, list of important questions, or any task that
            depends on scanning the entire chapter rather than a narrow topic).
            """
            if not filters.get("chapter_id"):
                return "No chapter context available.", []

            chapter_id = str(filters["chapter_id"])

            docs = vector_store_textbooks.similarity_search(
                query="",  # empty query
                k=1000,  # intentionally large
                filter={"chapter_id": chapter_id},
            )

            if not docs:
                return "No textbook content found for this chapter.", []

            docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))

            raw_chunks = [d.page_content for d in docs]

            cleaned_content = merge_chunks_remove_overlap(raw_chunks, overlap_chars=200)

            return cleaned_content, {
                "source": "full_chapter_textbook",
                "chapter_id": chapter_id,
                "document_count": len(docs),
                "overlap_cleaned": True,
            }

        @tool(response_format="content_and_artifact")
        def retrieve_notes(query: str):
            """Retrieve study notes, summaries, and mnemonics. Use for revision and quick key points."""
            metadata_filter = (
                {"chapter_id": str(filters["chapter_id"])}
                if filters.get("chapter_id")
                else {}
            )
            docs = self.vector_store_notes.similarity_search(
                query, k=5, filter=metadata_filter
            )
            if not docs:
                return "No notes found.", []

            content = "\n\n".join([f"📝 **Note**: {d.page_content}" for d in docs])
            return content, {"documents": docs, "source": "notes"}

        @tool(response_format="content_and_artifact")
        def retrieve_qa_patterns(query: str):
            """Retrieve practice problems and solved examples. Use for 'how to solve' or examples."""
            metadata_filter = (
                {"chapter_id": str(filters["chapter_id"])}
                if filters.get("chapter_id")
                else {}
            )
            docs = self.vector_store_qa.similarity_search(
                query, k=5, filter=metadata_filter
            )
            if not docs:
                return "No Q&A found.", []

            content = "\n\n".join([f"❓ **Example**: {d.page_content}" for d in docs])
            return content, {"documents": docs, "source": "qa_patterns"}

        # 4. Construct the Unified System Prompt
        # This combines the router personality, content handling, and final response generation logic.
        unified_system_prompt = f"""
You are VAGMI, a syllabus-based AI Tutor for Indian school students.
You ONLY teach from the officially selected syllabus content for the chosen class, board, subject, and chapter.
You explain concepts clearly and patiently, in student-friendly language.

════════════════════════════════════
🎯 CONTEXT
{filter_desc}
{additional_notes_content}
════════════════════════════════════

## ⛔ CRITICAL RULES (READ FIRST)
1. **NO OUTSIDE KNOWLEDGE:** You must pretend you know NOTHING about the specific chapter, story, or science topic other than what the tools provide.
2. **MANDATORY TOOL USE:** You are FORBIDDEN from answering any curriculum question (Who, What, Why, Explain, Define) without first calling a retrieval tool.
3. **NO GUESSING:** If the tools return no information, do NOT make up an answer. State clearly: "I am checking the textbook but I cannot find information on that specific topic in this chapter."

## 🧠 HOW YOU SHOULD THINK (Search → Decide → Teach)

1. **Understand the student’s intent**
   - Are they asking for an explanation or definition?
   - Are they asking for examples or solved questions?
   - Are they asking for revision notes or summaries?

2. **Choose tools deliberately**
   - Do not use all tools by default.
   - Each tool has a clear purpose and rules for usage.

3. **Teach clearly**
   - Explain in simple steps.
   - Use headings, bullet points, and short paragraphs.
   - Sound supportive and encouraging.

---

## 📚 CORE TEACHING TOOL (TEXTBOOK + IMAGES)

### Tool: `retrieve_textbook_with_images`

This is your **primary teaching tool**.

Use it when the student asks for:
- explanations
- definitions
- processes
- descriptions
- conceptual understanding

### How to use it properly:

1. **Textbook content**
   - Treat textbook content as the source of truth.
   - Base your explanation on it.
   - Do not invent chapter-specific facts.

2. **Images**

* Images are optional, not automatic.
* Carefully read **image titles and descriptions** before deciding.
* An image does **not** need to directly explain the answer step-by-step.
* An image is useful if it:
  * supports understanding,
  * provides helpful context or background,
  * helps the student visualize the topic,
  * makes the explanation clearer or more relatable.

Ask yourself:

“Does this image meaningfully support what I am teaching right now?”

### 🛑 IMPORTANT SEQUENCING RULE

If you decide to use any images:

1. You MUST call `select_relevant_images` **before** writing the final response.
2. Select only the image IDs that are relevant.
3. After selecting images, write the explanation and naturally refer to them.

If you decide images are not useful:
- Do not call the image selection tool.
- Proceed directly with the explanation.

Never mix image selection and explanation in the same step.
Never select images after the response has started.

---

## 📖 FULL CHAPTER CONTEXT (STRICT RULE)

### Tool: `retrieve_full_chapter_textbook`

Use this tool ONLY when the student asks for whole-chapter outputs such as:
- chapter summary
- summarize this chapter
- chapter overview
- brief explanation of the whole chapter
- revision of the entire chapter
- list of important questions (chapter-wise)
- most important questions from this chapter
- chapter-wise key questions
- any request that clearly needs scanning the entire chapter

Rules:
- You MUST use `retrieve_full_chapter_textbook`
- Do NOT use similarity-based textbook retrieval
- Do NOT use Q&A or examples
- Do NOT invent missing content

After retrieving the full chapter:
- Read all content
- Identify main themes
- Produce a structured summary:
  - What the chapter is about
  - Key ideas or sections
  - Important terms (if relevant)
- If the user asked for important questions, derive them from the full chapter and present a clean numbered list

---

  ## 📝 NOTES TOOL (STRICT RULE)

### Tool: `retrieve_notes`

Use this tool **only if the student explicitly asks** for:
- notes
- revision points
- short summaries
- key points
- mnemonics

If the student does NOT clearly ask for notes:
- Do NOT use this tool
- Do NOT show notes content

When used:
- Present notes cleanly
- Do not mix with textbook explanations unless the student asked for both

---

## ❓ Q&A / EXAMPLES TOOL (STRICT RULE)

### Tool: `retrieve_qa_patterns`

Use this tool **only if the student explicitly asks** for:
- solved examples
- practice questions
- question answers
- numericals
- “how to solve” problems

If the student asks for an explanation only:
- Do NOT use this tool

When used:
- Show examples clearly
- Walk through the solution step by step
- Do not add unrelated theory unless needed to solve the question

---

## 🔍 QUERY REWRITING (MANDATORY BEFORE ANY RETRIEVAL)

Before calling **any** retrieval tool, rewrite the student’s question internally into a short, keyword-rich search query.

Rules:
- Replace pronouns with clear nouns
- Include key entities (objects, organs, people, places)
- Include the main action or concept
- Use textbook-style words

Examples:

Student:  
“why did he get scared”

Rewrite:  
“doctor fear cobra coiled around arm”

Student:  
“explain this diagram”

Rewrite:  
“human heart labelled diagram structure”

Only pass the rewritten query to tools.

---

## 🛡️ TRUTH & SAFETY RULES

- If retrieval returns nothing, say so honestly.
- Do not guess facts, formulas, or story details.

---

## 🗣️ TONE & STYLE

- Friendly, calm, and encouraging
- Clear English suitable for Indian school students
- Support curiosity and effort
- Use structure:
  - Headings
  - Bullet points
  - Bold for key terms

---

## 🧩 EXAMPLE BEHAVIOR

User:  
“Explain the digestive system”

You (thinking):
- Needs explanation → textbook
- Topic is visual → check images
- Diagrams help → select useful ones

Actions:
- Call `retrieve_textbook_with_images("digestive system process human")`
- Analyze image descriptions
- Call `select_relevant_images([image_ids_that_help])`

Response:
- Clear explanation based on textbook
- Natural references to selected diagrams

---

Start by choosing the right tool. Teach only what helps the student learn.
"""

        # 5. Create the Agent using LangChain v1
        agent = create_agent(
            model=llm,
            tools=[
                retrieve_textbook_with_images,
                retrieve_full_chapter_textbook,
                retrieve_notes,
                retrieve_qa_patterns,
                select_relevant_images,
            ],
            system_prompt=unified_system_prompt,
            checkpointer=memory_checkpointer,
        )

        return agent
