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

# Database configuration
CONNECTION_STRING = settings.POSTGRES_URL
COLLECTION_NAME = "education_documents"

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=False,
    temperature=0.5,
    streaming=True,
)

# Create the vector store
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

memory_checkpointer = MemorySaver()


class EducationPlatform:
    """Main class handling the education platform logic"""

    def __init__(self):
        self.vector_store = vector_store

    def create_retrieval_tool(
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
    ):
        """Create retrieval tool with ID-based hierarchical filters

        Args:
            filters: Dictionary with ID-based filters (e.g., {"class_level_id": "5"})
            names: Optional dictionary with human-readable names for context
        """
        names = names or {}

        @tool(response_format="content_and_artifact")
        def retrieve_filtered_content(query: str):
            """Retrieve educational content based on hierarchical filters using IDs.
            
            This tool searches through the educational content database to find relevant
            information for the student's question. It filters content by:
            - Class Level (e.g., Class 10, Class 12)
            - Board (e.g., CBSE, ICSE, State Board)
            - Medium (e.g., English, Hindi, Regional languages)
            - Subject (e.g., Mathematics, Physics, Chemistry)
            - Chapter (specific chapter within the subject)
            
            Use this tool when students ask subject-specific questions, need help with
            homework, or require detailed explanations of academic concepts.
            """
            try:
                # Build metadata filter using IDs
                # IMPORTANT: All values must be strings as they're stored as strings in jsonb
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

                logger.info(f"Query: {query}")
                logger.info(f"Searching with metadata filter: {metadata_filter}")

                # Perform similarity search with metadata filter
                retrieved_docs = self.vector_store.similarity_search(
                    query, k=5, filter=metadata_filter
                )

                logger.info(f"Found {len(retrieved_docs)} documents")

                if not retrieved_docs:
                    # Use names for better user experience if available
                    if names:
                        filter_desc = " → ".join([v for v in names.values() if v])
                    else:
                        filter_desc = ", ".join(
                            [f"{k.replace('_id', '')}: {v}" for k, v in metadata_filter.items()]
                        )
                    
                    no_content_msg = (
                        f"📚 I don't have specific educational content available for {filter_desc} at the moment. "
                        f"This could be because:\n"
                        f"• Content hasn't been uploaded for this specific combination yet\n"
                        f"• The content might be under a different chapter or topic\n"
                        f"• Try rephrasing your question with different keywords\n\n"
                        f"💡 I can still help you with general concepts and explanations! "
                        f"Feel free to ask me about {names.get('subject', 'the subject')} concepts, "
                        f"and I'll do my best to assist you with my general knowledge."
                    )
                    return no_content_msg, []

                # Format retrieved content with better structure
                serialized_docs = []
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    content = doc.page_content.strip()
                    
                    # Add chapter context if available
                    chapter_info = ""
                    if doc.metadata.get('chapter_id'):
                        chapter_info = f" (Chapter ID: {doc.metadata['chapter_id']})"
                    
                    serialized_docs.append(
                        f"📄 **Document {i}** - Source: {source}{chapter_info}\n\n"
                        f"{content}\n"
                        f"{'─' * 50}"
                    )
                
                serialized = "\n\n".join(serialized_docs)

                return serialized, retrieved_docs

            except Exception as e:
                logger.error(f"Error in retrieval: {e}", exc_info=True)
                return (
                    f"🔧 I encountered a technical issue while searching for content. "
                    f"Don't worry! I can still help you with your question using my general knowledge. "
                    f"Please feel free to ask me about {names.get('subject', 'the subject')} concepts, "
                    f"and I'll do my best to provide helpful explanations."
                ), []

        return retrieve_filtered_content

    def create_rag_graph(
        self, filters: Dict[str, str], names: Optional[Dict[str, str]] = None
    ):
        """Create RAG graph with ID-based hierarchical filtering

        Args:
            filters: Dictionary with ID-based filters (e.g., {"class_level_id": "5"})
            names: Optional dictionary with human-readable names for better prompts
        """
        retrieval_tool = self.create_retrieval_tool(filters, names)

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

🔍 RESPONSE STRATEGY:
- For greetings, general questions, or non-academic queries: Respond directly with warmth and helpfulness
- For subject-specific questions, homework help, or academic queries: Use the retrieval tool to access relevant educational content
- Always maintain an encouraging and supportive tone
- If unsure about academic content, use the retrieval tool to ensure accuracy

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

            llm_with_tools = llm.bind_tools([retrieval_tool])
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def generate_final_response(state: MessagesState):
            """Generate final response using retrieved content."""
            # Get the retrieved content from the tool call
            tool_messages = [
                msg for msg in state["messages"] 
                if hasattr(msg, 'type') and msg.type == "tool"
            ]
            
            if tool_messages:
                context = tool_messages[-1].content
                # Check if the tool returned an error message (starts with emoji indicators)
                if context.startswith(("📚", "🔧")):
                    # This is an error or no-content message, handle gracefully
                    context = f"Note: {context}"
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
        graph_builder.add_node("tools", ToolNode([retrieval_tool]))
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