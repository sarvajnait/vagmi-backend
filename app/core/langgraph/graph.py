from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_postgres import PGVector
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Dict, Optional
from loguru import logger
from langgraph.checkpoint.memory import MemorySaver
from sqlmodel import Session, select
from app.services.database import get_session
from app.models import LLMImage, AdditionalNotes

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
    temperature=0.5,
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


class EducationPlatform:
    """Main class handling the education platform logic"""

    def __init__(self):
        self.vector_store_textbooks = vector_store_textbooks
        self.vector_store_notes = vector_store_notes
        self.vector_store_qa = vector_store_qa
        self.vector_store_images = vector_store_images

    def create_textbook_retrieval_tool(
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
    ):
        """Create textbook retrieval tool with ID-based hierarchical filters"""
        names = names or {}

        @tool(response_format="content_and_artifact")
        def retrieve_textbooks(query: str):
            """Retrieve textbook content for foundational concepts and curriculum topics.

            This tool searches through official textbooks and curriculum materials to find
            relevant information for the student's question. Use this tool when:
            - Students need foundational explanations of concepts
            - Questions relate to core curriculum topics
            - You need official textbook definitions and examples
            - Students ask "what is..." or "explain..." questions

            The content is filtered by the student's class level, board, medium, subject, and chapter.
            """
            try:
                metadata_filter = {}
                if filters.get("class_level_id"):
                    metadata_filter["class_level_id"] = str(filters["class_level_id"])
                if filters.get("board_id"):
                    metadata_filter["board_id"] = str(filters["board_id"])
                if filters.get("medium_id"):
                    metadata_filter["medium_id"] = str(filters["medium_id"])
                if filters.get("subject_id"):
                    metadata_filter["subject_id"] = str(filters["subject_id"])
                if filters.get("chapter_id"):
                    metadata_filter["chapter_id"] = str(filters["chapter_id"])

                logger.info(f"[TEXTBOOKS] Query: {query}")
                logger.info(f"[TEXTBOOKS] Metadata filter: {metadata_filter}")

                retrieved_docs = self.vector_store_textbooks.similarity_search(
                    query, k=5, filter=metadata_filter
                )

                logger.info(f"[TEXTBOOKS] Found {len(retrieved_docs)} documents")

                if not retrieved_docs:
                    return "No textbook content found for this query.", []

                serialized_docs = []
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    content = doc.page_content.strip()
                    content_type = doc.metadata.get('content_type', 'textbook')

                    serialized_docs.append(
                        f"📚 **Textbook {i}** - Source: {source}\n"
                        f"Type: {content_type}\n\n"
                        f"{content}\n"
                        f"{'─' * 50}"
                    )

                serialized = "\n\n".join(serialized_docs)
                return serialized, {"documents": retrieved_docs, "source": "textbooks"}

            except Exception as e:
                logger.error(f"[TEXTBOOKS] Error in retrieval: {e}", exc_info=True)
                return f"Error retrieving textbook content: {str(e)}", []

        return retrieve_textbooks

    def create_notes_retrieval_tool(
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
    ):
        """Create notes retrieval tool for supplementary study materials"""
        names = names or {}

        @tool(response_format="content_and_artifact")
        def retrieve_notes(query: str):
            """Retrieve study notes and supplementary learning materials.

            This tool searches through curated study notes, summaries, and supplementary
            materials uploaded by educators. Use this tool when:
            - Students need quick summaries or key points
            - Looking for study tips or mnemonics
            - Need supplementary explanations beyond textbooks
            - Students are revising or preparing for exams

            The content is filtered by the student's class level, board, medium, subject, and chapter.
            """
            try:
                metadata_filter = {}
                if filters.get("class_level_id"):
                    metadata_filter["class_level_id"] = str(filters["class_level_id"])
                if filters.get("board_id"):
                    metadata_filter["board_id"] = str(filters["board_id"])
                if filters.get("medium_id"):
                    metadata_filter["medium_id"] = str(filters["medium_id"])
                if filters.get("subject_id"):
                    metadata_filter["subject_id"] = str(filters["subject_id"])
                if filters.get("chapter_id"):
                    metadata_filter["chapter_id"] = str(filters["chapter_id"])

                logger.info(f"[NOTES] Query: {query}")
                logger.info(f"[NOTES] Metadata filter: {metadata_filter}")

                retrieved_docs = self.vector_store_notes.similarity_search(
                    query, k=5, filter=metadata_filter
                )

                logger.info(f"[NOTES] Found {len(retrieved_docs)} documents")

                if not retrieved_docs:
                    return "No study notes found for this query.", []

                serialized_docs = []
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    content = doc.page_content.strip()
                    note_id = doc.metadata.get('note_id', 'N/A')

                    serialized_docs.append(
                        f"📝 **Study Note {i}** - Source: {source} (Note ID: {note_id})\n\n"
                        f"{content}\n"
                        f"{'─' * 50}"
                    )

                serialized = "\n\n".join(serialized_docs)
                return serialized, {"documents": retrieved_docs, "source": "notes"}

            except Exception as e:
                logger.error(f"[NOTES] Error in retrieval: {e}", exc_info=True)
                return f"Error retrieving notes: {str(e)}", []

        return retrieve_notes

    def create_qa_retrieval_tool(
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
    ):
        """Create Q&A patterns retrieval tool for practice problems and examples"""
        names = names or {}

        @tool(response_format="content_and_artifact")
        def retrieve_qa_patterns(query: str):
            """Retrieve Q&A patterns, practice problems, and solved examples.

            This tool searches through question-answer patterns, practice problems,
            and solved examples. Use this tool when:
            - Students need practice problems or examples
            - Looking for step-by-step solutions
            - Need to understand problem-solving approaches
            - Students ask "how to solve..." or "show me an example..."

            The content is filtered by the student's class level, board, medium, subject, and chapter.
            """
            try:
                metadata_filter = {}
                if filters.get("class_level_id"):
                    metadata_filter["class_level_id"] = str(filters["class_level_id"])
                if filters.get("board_id"):
                    metadata_filter["board_id"] = str(filters["board_id"])
                if filters.get("medium_id"):
                    metadata_filter["medium_id"] = str(filters["medium_id"])
                if filters.get("subject_id"):
                    metadata_filter["subject_id"] = str(filters["subject_id"])
                if filters.get("chapter_id"):
                    metadata_filter["chapter_id"] = str(filters["chapter_id"])

                logger.info(f"[QA] Query: {query}")
                logger.info(f"[QA] Metadata filter: {metadata_filter}")

                retrieved_docs = self.vector_store_qa.similarity_search(
                    query, k=5, filter=metadata_filter
                )

                logger.info(f"[QA] Found {len(retrieved_docs)} documents")

                if not retrieved_docs:
                    return "No Q&A patterns or practice problems found for this query.", []

                serialized_docs = []
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    content = doc.page_content.strip()
                    pattern_id = doc.metadata.get('pattern_id', 'N/A')

                    serialized_docs.append(
                        f"❓ **Q&A Pattern {i}** - Source: {source} (Pattern ID: {pattern_id})\n\n"
                        f"{content}\n"
                        f"{'─' * 50}"
                    )

                serialized = "\n\n".join(serialized_docs)
                return serialized, {"documents": retrieved_docs, "source": "qa_patterns"}

            except Exception as e:
                logger.error(f"[QA] Error in retrieval: {e}", exc_info=True)
                return f"Error retrieving Q&A patterns: {str(e)}", []

        return retrieve_qa_patterns

    def create_images_retrieval_tool(
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
    ):
        """Create images retrieval tool for visual learning aids"""
        names = names or {}

        @tool(response_format="content_and_artifact")
        def retrieve_images(query: str):
            """Retrieve relevant images, diagrams, and visual learning aids.

            This tool searches through educational images, diagrams, charts, and visual
            materials based on their descriptions and tags. Use this tool when:
            - Students need visual explanations or diagrams
            - Questions involve concepts better explained with images
            - Looking for charts, graphs, or illustrations
            - Students ask to "show me..." or "what does it look like..."

            The images are filtered by the student's class level, board, medium, subject, and chapter.
            Returns image URLs with descriptions for the frontend to display.
            """
            try:
                metadata_filter = {}
                if filters.get("class_level_id"):
                    metadata_filter["class_level_id"] = str(filters["class_level_id"])
                if filters.get("board_id"):
                    metadata_filter["board_id"] = str(filters["board_id"])
                if filters.get("medium_id"):
                    metadata_filter["medium_id"] = str(filters["medium_id"])
                if filters.get("subject_id"):
                    metadata_filter["subject_id"] = str(filters["subject_id"])
                if filters.get("chapter_id"):
                    metadata_filter["chapter_id"] = str(filters["chapter_id"])

                logger.info(f"[IMAGES] Query: {query}")
                logger.info(f"[IMAGES] Metadata filter: {metadata_filter}")

                retrieved_docs = self.vector_store_images.similarity_search(
                    query, k=5, filter=metadata_filter
                )

                logger.info(f"[IMAGES] Found {len(retrieved_docs)} images")

                if not retrieved_docs:
                    return "No relevant images found for this query.", []

                serialized_docs = []
                for i, doc in enumerate(retrieved_docs, 1):
                    title = doc.metadata.get('title', 'Untitled')
                    description = doc.metadata.get('description', '')
                    file_url = doc.metadata.get('file_url', '')
                    tags = doc.metadata.get('tags', [])
                    image_id = doc.metadata.get('image_id', 'N/A')

                    img_info = f"🖼️ **Image {i}: {title}**\n"
                    if description:
                        img_info += f"📝 Description: {description}\n"
                    if tags:
                        img_info += f"🏷️ Tags: {', '.join(tags)}\n"
                    img_info += f"🔗 URL: {file_url}\n"
                    img_info += f"ID: {image_id}\n"
                    img_info += f"{'─' * 50}"

                    serialized_docs.append(img_info)

                serialized = "\n\n".join(serialized_docs)
                return serialized, {"documents": retrieved_docs, "source": "images"}

            except Exception as e:
                logger.error(f"[IMAGES] Error in retrieval: {e}", exc_info=True)
                return f"Error retrieving images: {str(e)}", []

        return retrieve_images

    def create_rag_graph(
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
    ):
        """Create RAG graph with ID-based hierarchical filtering and multiple specialized tools

        Args:
            filters: Dictionary with ID-based filters (e.g., {"class_level_id": "5"})
            names: Optional dictionary with human-readable names for better prompts
        """
        # Create all four specialized retrieval tools
        textbook_tool = self.create_textbook_retrieval_tool(filters, names)
        notes_tool = self.create_notes_retrieval_tool(filters, names)
        qa_tool = self.create_qa_retrieval_tool(filters, names)
        images_tool = self.create_images_retrieval_tool(filters, names)

        # Fetch AdditionalNotes for this chapter (if chapter_id is available)
        additional_notes_content = ""
        if filters.get("chapter_id"):
            try:
                session = next(get_session())
                chapter_id = int(filters["chapter_id"])

                # Fetch AdditionalNotes
                notes_statement = select(AdditionalNotes).where(AdditionalNotes.chapter_id == chapter_id)
                additional_notes = session.exec(notes_statement).all()

                if additional_notes:
                    additional_notes_content = "\n\n📌 **Important Additional Context:**\n"
                    for note in additional_notes:
                        additional_notes_content += f"• {note.note}\n"
                    additional_notes_content += "\n"

                logger.info(f"Fetched {len(additional_notes)} additional notes for chapter {chapter_id}")

            except Exception as e:
                logger.error(f"Error fetching additional context: {e}")
            finally:
                session.close()

        # Use names for context description if available, otherwise use IDs
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

        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            system_message = SystemMessage(
                content=f"""You are VAGMI, an advanced AI educational assistant designed specifically for Indian students. You are an expert tutor specializing in {subject_name} for {'Class ' + class_name if class_name else 'students'}.

🎯 CURRENT LEARNING CONTEXT:
{filter_desc}

📚 YOUR ROLE & CAPABILITIES:
- Expert tutor with deep knowledge of Indian curriculum and educational standards
- Specialized in {subject_name} with understanding of {'Class ' + class_name if class_name else 'appropriate grade level'} concepts
- Patient, encouraging, and supportive learning companion
- Capable of explaining complex topics in simple, age-appropriate language

🎓 EDUCATIONAL APPROACH:
- Use clear, simple language appropriate for {'Class ' + class_name if class_name else 'the student level'}
- Break down complex concepts into digestible parts
- Provide step-by-step explanations with examples
- Use analogies and real-world examples relevant to Indian students
- Encourage critical thinking and problem-solving skills
- Adapt explanations based on student's apparent understanding level

🔍 RESPONSE STRATEGY & TOOL USAGE:
- For greetings, general questions, or non-academic queries: Respond directly with warmth and helpfulness
- For subject-specific questions, homework help, or academic queries: Use the appropriate retrieval tools
- **Tool Selection Guide:**
  • Use `retrieve_textbooks` for foundational concepts, definitions, and curriculum explanations
  • Use `retrieve_notes` for summaries, study tips, quick revision, and supplementary materials
  • Use `retrieve_qa_patterns` for practice problems, solved examples, and step-by-step solutions
  • Use `retrieve_images` for visual aids, diagrams, charts, and illustrations to explain concepts
  • You can use MULTIPLE tools in sequence if needed for comprehensive answers
- Always maintain an encouraging and supportive tone{additional_notes_content}

🛡️ SAFETY & GUIDELINES:
- Never provide direct answers to exam questions or homework without explanation
- Encourage learning and understanding over memorization
- Maintain academic integrity and ethical standards
- Be culturally sensitive and inclusive
- If asked about inappropriate content, politely redirect to educational topics

💡 TEACHING STYLE:
- Use interactive questioning to gauge understanding
- Provide multiple examples and practice problems when appropriate
- Connect new concepts to previously learned material
- Celebrate progress and effort, not just correct answers
- Use visual descriptions and analogies when helpful

Remember: Your goal is to foster genuine understanding and love for learning, not just to provide answers."""
            )

            messages = state.get("messages") or []
            messages = [system_message] + messages

            llm_with_tools = llm.bind_tools([textbook_tool, notes_tool, qa_tool, images_tool])
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def generate_final_response(state: MessagesState):
            """Generate final response using retrieved content from multiple tools."""
            # Get all retrieved content from tool calls
            tool_messages = [
                msg for msg in state["messages"]
                if hasattr(msg, 'type') and msg.type == "tool"
            ]

            if tool_messages:
                # Combine content from all tool calls
                context_parts = []
                for tool_msg in tool_messages:
                    if tool_msg.content:
                        context_parts.append(tool_msg.content)
                context = "\n\n".join(context_parts) if context_parts else "No specific educational content was retrieved for this query."
            else:
                context = "No specific educational content was retrieved for this query."

            # Get the original user question
            human_messages = [
                msg for msg in state["messages"]
                if hasattr(msg, 'type') and msg.type == "human"
            ]
            user_question = human_messages[-1].content if human_messages else ""

            system_message_content = f"""You are VAGMI, an expert AI tutor specializing in {subject_name} for {'Class ' + class_name if class_name else 'students'}. You have access to comprehensive educational content and must provide accurate, helpful responses.

📖 LEARNING CONTEXT:
{filter_desc}

❓ STUDENT'S QUESTION:
{user_question}
{additional_notes_content}
📚 RETRIEVED EDUCATIONAL CONTENT:
{context}

🎯 RESPONSE REQUIREMENTS:

1. **ACCURACY & COMPLETENESS:**
   - Use the retrieved content as your primary source of information
   - If content is insufficient, acknowledge limitations and provide what you can
   - Cross-reference information for accuracy
   - Cite specific concepts, formulas, or examples from the content when relevant

2. **EDUCATIONAL APPROACH:**
   - Explain concepts step-by-step in {'Class ' + class_name if class_name else 'age-appropriate'} language
   - Use analogies and real-world examples relevant to Indian students
   - Break down complex topics into digestible parts
   - Encourage understanding over memorization

3. **RESPONSE STRUCTURE:**
   - Start with a clear, direct answer to the question
   - Provide detailed explanation with examples
   - Use bullet points, numbered lists, or headings for clarity
   - Include practice problems or follow-up questions when appropriate
   - End with encouragement and next steps

4. **SAFETY & ETHICS:**
   - Never provide direct answers to exam questions without explanation
   - Encourage learning and critical thinking
   - Maintain academic integrity
   - Be culturally sensitive and inclusive

5. **TONE & STYLE:**
   - Be encouraging, patient, and supportive
   - Use positive reinforcement
   - Adapt complexity to student's apparent level
   - Celebrate effort and progress

6. **CONTENT HANDLING:**
   - If retrieved content is unclear, acknowledge limitations
   - If no specific content was found, use your general knowledge to help the student
   - Suggest additional resources or study methods when appropriate
   - Connect new concepts to previously learned material
   - Provide multiple examples to reinforce understanding
   - Always be honest about the source of your information
   - When images/diagrams are provided in the retrieved content, reference them in your explanation
   - Explain what each image shows and how it relates to the concept being discussed
   - Encourage students to view the images for better understanding

7. **FALLBACK STRATEGY:**
   - If no specific educational content is available, rely on your general knowledge
   - Clearly indicate when you're using general knowledge vs. specific content
   - Still provide helpful explanations and examples
   - Encourage the student to consult their textbooks or teachers for verification

Remember: Your goal is to foster genuine understanding and academic growth, not just provide answers."""

            # Get only the conversation messages (exclude tool messages)
            conversation_messages = [
                msg for msg in state["messages"]
                if hasattr(msg, 'type') and msg.type in ("human", "system")
                or (hasattr(msg, 'type') and msg.type == "ai" and not getattr(msg, "tool_calls", None))
            ]

            prompt_messages = [SystemMessage(system_message_content)] + conversation_messages
            response = llm.invoke(prompt_messages)
            return {"messages": [response]}

        # Build the graph
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", ToolNode([textbook_tool, notes_tool, qa_tool, images_tool]))
        graph_builder.add_node("generate", generate_final_response)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        return graph_builder.compile(checkpointer=memory_checkpointer)