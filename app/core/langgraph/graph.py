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
            """Retrieve educational content based on hierarchical filters using IDs."""
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
                        f"I don't have any educational content available for {filter_desc}. "
                        f"Please make sure documents have been uploaded for this specific combination."
                    )
                    return no_content_msg, []

                # Format retrieved content with better structure
                serialized_docs = []
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    content = doc.page_content.strip()
                    serialized_docs.append(
                        f"[Document {i} - Source: {source}]\n{content}"
                    )
                
                serialized = "\n\n---\n\n".join(serialized_docs)

                return serialized, retrieved_docs

            except Exception as e:
                logger.error(f"Error in retrieval: {e}", exc_info=True)
                return f"I encountered an error while searching for content: {str(e)}", []

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
                content=f"""You are an educational assistant for Indian students studying {subject_name}.

Current Learning Context: {filter_desc}

Guidelines:
- Use simple, clear language appropriate for {'Class ' + class_name if class_name else 'the student level'}
- For greetings or general questions, respond directly without using tools
- For subject-specific questions, use the retrieval tool to access educational content
- Be encouraging and patient in your explanations
- Use examples relevant to Indian students when possible
"""
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
            else:
                context = "No additional context available."

            # Get the original user question
            human_messages = [
                msg for msg in state["messages"] 
                if hasattr(msg, 'type') and msg.type == "human"
            ]
            user_question = human_messages[-1].content if human_messages else ""

            system_message_content = f"""You are an educational assistant helping students with {subject_name}.

Learning Context: {filter_desc}

Student's Question: {user_question}

Retrieved Educational Content:
{context}

Instructions:
- Answer the question clearly and comprehensively using the retrieved content
- Use simple language appropriate for {'Class ' + class_name if class_name else 'the student'}
- Structure your answer with bullet points or numbered lists when helpful
- Include relevant examples or explanations
- If the content doesn't fully answer the question, acknowledge this and provide what you can
- Be encouraging and supportive in your tone
"""

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