from langchain.agents import create_agent
from langchain.tools import tool
from langchain_postgres import PGVector
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Dict, Optional
from loguru import logger
from langgraph.checkpoint.memory import MemorySaver
from sqlmodel import select
from app.services.database import get_session
from app.models import AdditionalNotes

# Database configuration
CONNECTION_STRING = settings.POSTGRES_URL
COLLECTION_NAME_TEXTBOOKS = "llm_textbooks"
COLLECTION_NAME_NOTES = "llm_notes"
COLLECTION_NAME_QA = "qa_patterns"
COLLECTION_NAME_IMAGES = "llm_images"

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
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

        # 2. Fetch Additional SQL Notes (Pre-fetch for system prompt)
        additional_notes_content = ""
        if filters.get("chapter_id"):
            session = next(get_session())
            try:
                chapter_id = int(filters["chapter_id"])
                notes = session.exec(
                    select(AdditionalNotes).where(
                        AdditionalNotes.chapter_id == chapter_id
                    )
                ).all()
                if notes:
                    additional_notes_content = (
                        "\n\n📌 **Important Teacher Context (Must Integrate):**\n"
                    )
                    for note in notes:
                        additional_notes_content += f"• {note.note}\n"
            except Exception as e:
                logger.error(f"Error fetching additional context: {e}")
            finally:
                session.close()

        # 3. Define Tools with Closures (Capturing filters)
        # We define them here so they have access to the specific 'filters' for this session

        @tool(response_format="content_and_artifact")
        def retrieve_textbooks(query: str):
            """Retrieve textbook content for foundational concepts. Use for 'what is', definitions, and core explanations."""
            metadata_filter = (
                {"chapter_id": str(filters["chapter_id"])}
                if filters.get("chapter_id")
                else {}
            )
            docs = self.vector_store_textbooks.similarity_search(
                query, k=5, filter=metadata_filter
            )
            print("---")
            print(docs)
            print("---")

            if not docs:
                return "No textbook content found.", []

            content = "\n\n".join([f"📚 **Textbook**: {d.page_content}" for d in docs])
            return content, {"documents": docs, "source": "textbooks"}

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

        @tool(response_format="content_and_artifact")
        def retrieve_images(query: str):
            """Retrieve diagrams and visual aids. Returns image metadata."""
            metadata_filter = (
                {"chapter_id": str(filters["chapter_id"])}
                if filters.get("chapter_id")
                else {}
            )
            docs = self.vector_store_images.similarity_search(
                query, k=5, filter=metadata_filter
            )
            if not docs:
                return "No images found.", {"image_metadata": []}

            # Format strictly for the LLM's logic
            serialized = []
            metadata_list = []
            for i, doc in enumerate(docs, 1):
                info = f"🖼️ **Image {i}**: {doc.metadata.get('title')} (ID: {doc.metadata.get('image_id')})\nDesc: {doc.metadata.get('description')}"
                serialized.append(info)
                metadata_list.append(
                    {
                        "image_id": doc.metadata.get("image_id"),
                        "file_url": doc.metadata.get("file_url"),
                        "title": doc.metadata.get("title"),
                        "description": doc.metadata.get("description"),
                        "tags": doc.metadata.get("tags", []),
                    }
                )

            return "\n\n".join(serialized), {
                "documents": docs,
                "source": "images",
                "image_metadata": metadata_list,
            }

        # 4. Construct the Unified System Prompt
        # This combines the router personality, content handling, and final response generation logic.
        unified_system_prompt = f"""
You are VAGMI, a careful, honest, syllabus-focused teacher for Indian school students.

You teach strictly according to the official syllabus and textbook.
You NEVER invent facts, names, examples, or answers.

════════════════════════════════════
🎯 CURRENT TEACHING CONTEXT
{filter_desc}
════════════════════════════════════
{additional_notes_content}

You behave like a real teacher who ALWAYS checks the textbook or notes
before answering factual questions.

────────────────────────────────────
🚨 NON-NEGOTIABLE RULE (MOST IMPORTANT)
────────────────────────────────────

You MUST ALWAYS use the available learning tools
before answering ANY academic question.

There are NO exceptions.

• Never answer from memory
• Never guess
• Never invent names, facts, or explanations
• Never rely on general knowledge
• Never assume you already know the answer

If you do not use a tool, your answer is INVALID.

────────────────────────────────────
🛠️ TOOL USAGE REQUIREMENTS
────────────────────────────────────

Before answering ANY question, follow this order:

1. Decide what kind of question it is
2. Call the correct tool
3. Read the returned material
4. Answer ONLY from that material

If the tool returns nothing useful:
• Say clearly and calmly:
  “This is not clearly stated in the syllabus material.”
• Then explain only what is commonly taught at this class level
• Do NOT add new facts, names, or details

────────────────────────────────────
📚 WHICH TOOL TO USE (MANDATORY)
────────────────────────────────────

You MUST choose at least ONE tool.

• Names of characters, places, stories, facts → retrieve_textbooks
• Meanings, definitions, explanations → retrieve_textbooks
• Revision, short points → retrieve_notes
• How to solve, examples → retrieve_qa_patterns
• Diagrams, objects, processes → retrieve_images (and then select images)

If the question is academic and you did not use a tool,
you have failed your task.

────────────────────────────────────
🛑 STRICT ANTI-HALLUCINATION RULES
────────────────────────────────────

You are NOT allowed to:

• Make up names (even if they sound reasonable)
• Change answers between turns
• “Correct yourself” without tool evidence
• Say “according to the story” unless verified
• Sound confident when unsure

If the learning material does NOT confirm something,
you MUST say it is not confirmed.

Silence or refusal is better than a wrong answer.

────────────────────────────────────
📖 SYLLABUS BOUNDARY
────────────────────────────────────

You teach ONLY what is part of the syllabus.

If the student asks:
• Off-syllabus content → answer briefly and bring it back to syllabus
• Exam shortcuts → refuse politely and explain the concept
• Casual chat → respond warmly, no tools needed

────────────────────────────────────
📷 IMAGE RULES
────────────────────────────────────

Use images whenever they support understanding.

If images are retrieved:
• Select relevant ones
• Speak naturally as if the picture is already visible
• Never mention IDs, tools, or storage

────────────────────────────────────
🎓 ANSWER STYLE
────────────────────────────────────

After using tools, explain like a patient teacher:

• Clear
• Step-by-step
• Simple language
• Calm tone

Short answers are fine.
Accuracy is more important than fluency.

────────────────────────────────────
🏁 FINAL CHECK (INTERNAL)
────────────────────────────────────

Before finalising your answer, silently verify:

• Did I use a tool?
• Is every fact supported by the material?
• Did I avoid guessing?

If any answer is “no”, STOP and use a tool.

Teach carefully.
Truth matters more than confidence.
"""

        # 5. Create the Agent using LangChain v1
        agent = create_agent(
            model=llm,
            tools=[
                retrieve_textbooks,
                retrieve_notes,
                retrieve_qa_patterns,
                retrieve_images,
                select_relevant_images,
            ],
            system_prompt=unified_system_prompt,
            checkpointer=memory_checkpointer,
        )

        return agent
