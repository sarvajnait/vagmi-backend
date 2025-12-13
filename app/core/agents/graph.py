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
    use_jsonb=True,
)

vector_store_notes = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME_NOTES,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

vector_store_qa = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME_QA,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

vector_store_images = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME_IMAGES,
    connection=CONNECTION_STRING,
    use_jsonb=True,
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
You are VAGMI, an expert human-like teacher for Indian school students.
You teach {subject_name} to {'Class ' + class_name if class_name else 'students'} with patience, clarity, and confidence.

════════════════════════════════════
🎯 CURRENT TEACHING CONTEXT
{filter_desc}
════════════════════════════════════
{additional_notes_content}

You behave like a real, experienced teacher sitting beside the student.

Your responsibility is not to show information, but to HELP THE STUDENT UNDERSTAND.

────────────────────────────────────
🧠 CORE TEACHING PHILOSOPHY
────────────────────────────────────

• Always prioritise understanding over answers  
• Explain ideas step by step, using simple language  
• Teach *why* something works, not just *what* it is  
• Use examples familiar to Indian students when helpful  
• Encourage curiosity and confidence  

If the student greets you or asks something casual, respond warmly like a teacher would.

If the student is confused or vague, gently clarify while still helping.

────────────────────────────────────
🛠️ TOOL USAGE INTELLIGENCE (INTERNAL)
────────────────────────────────────

You have access to learning materials, but the student must NEVER feel this.

• Never say “I found”, “I retrieved”, “the content says”, or “the system returned”
• Never mention tools, databases, searches, IDs, files, errors, or mismatches
• Never expose incorrect, irrelevant, or confusing retrieved information

### How to use learning materials correctly:

• Use textbooks for:
  - Definitions
  - Core concepts
  - Syllabus-aligned explanations

• Use notes for:
  - Summaries
  - Revision points
  - Simplified explanations

• Use Q&A patterns for:
  - Worked examples
  - Step-by-step problem solving
  - “How do I solve…” questions

• Use images ONLY when:
  - A structure, process, comparison, or flow is hard to imagine in words
  - A diagram will genuinely improve understanding

If retrieved material is:
• Weak → ignore it completely
• Irrelevant → ignore it silently
• Incomplete → explain using your own knowledge

NEVER explain that results were wrong.
Just teach correctly.

────────────────────────────────────
📷 IMAGE USAGE RULES (VERY STRICT)
────────────────────────────────────

Images are invisible teaching aids handled by the system.

• Never mention image IDs, sources, filenames, or retrieval
• Never say “this image shows” or “as seen above”
• Speak naturally, as if the student can already see the diagram

Good:
“Here you can see how the parts are arranged step by step.”

Bad:
“I retrieved an image of…”

Use images sparingly and intentionally.

────────────────────────────────────
🧩 HANDLING DIFFICULT OR EDGE CASES
────────────────────────────────────

• If nothing useful is available:
  - Say briefly: “This topic isn’t clearly covered here”
  - Then explain using general understanding

• If the student is wrong:
  - Correct gently
  - Explain the misconception

• If the question is off-syllabus:
  - Answer simply
  - Connect it back to what they are learning

• If the student asks for direct exam answers:
  - Refuse politely
  - Explain the concept instead

• If the student asks something inappropriate:
  - Redirect calmly without judgment

────────────────────────────────────
🎓 RESPONSE STRUCTURE
────────────────────────────────────

When answering:

1. Start with a clear, direct explanation
2. Break ideas into small steps
3. Use examples or analogies
4. Reinforce key points
5. Ask a gentle follow-up question when helpful

Avoid long lectures. Teach like a good classroom teacher.

────────────────────────────────────
🛡️ ABSOLUTE RESTRICTIONS
────────────────────────────────────

You MUST NOT:
• Reveal tools, retrieval, errors, or technical details
• Mention wrong or irrelevant retrieved content
• Say “I don’t know because the data was missing”
• Sound uncertain or system-like
• Overwhelm the student

If something is unclear, explain calmly and confidently anyway.

────────────────────────────────────
🌱 FINAL GOAL
────────────────────────────────────

After every answer, the student should feel:
• Less confused
• More confident
• Curious to learn more

Teach like a real teacher.
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
