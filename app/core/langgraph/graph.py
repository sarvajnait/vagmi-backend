from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_postgres import PGVector
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Dict
from loguru import logger
from langgraph.checkpoint.memory import MemorySaver
from app.models import *
from app.schemas import *

# Database configuration

# Initialize PGVector store
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

    def create_retrieval_tool(self, filters: Dict[str, str]):
        """Create retrieval tool with hierarchical filters"""

        @tool(response_format="content_and_artifact")
        def retrieve_filtered_content(query: str):
            """Retrieve educational content based on hierarchical filters."""
            try:
                # Build metadata filter
                metadata_filter = {}
                if filters.get("class_level"):
                    metadata_filter["class_level"] = filters["class_level"]
                if filters.get("board"):
                    metadata_filter["board"] = filters["board"]
                if filters.get("medium"):
                    metadata_filter["medium"] = filters["medium"]
                if filters.get("subject"):
                    metadata_filter["subject"] = filters["subject"]
                if filters.get("chapter"):
                    metadata_filter["chapter"] = filters["chapter"]

                # Perform similarity search with metadata filter
                retrieved_docs = self.vector_store.similarity_search(
                    query, k=3, filter=metadata_filter
                )

                if not retrieved_docs:
                    filter_desc = ", ".join(
                        [f"{k}: {v}" for k, v in metadata_filter.items()]
                    )
                    return f"No relevant content found for filters: {filter_desc}", []

                # Format retrieved content
                serialized = "\n\n".join(
                    [
                        f"Source: {doc.metadata.get('source_file', 'Unknown')}\n"
                        f"Chapter: {doc.metadata.get('chapter', 'N/A')}\n"
                        f"Content: {doc.page_content}"
                        for doc in retrieved_docs
                    ]
                )

                return serialized, retrieved_docs

            except Exception as e:
                logger.error(f"Error in retrieval: {e}")
                return f"Error retrieving content: {str(e)}", []

        return retrieve_filtered_content

    def create_rag_graph(self, filters: Dict[str, str]):
        """Create RAG graph with hierarchical filtering"""
        retrieval_tool = self.create_retrieval_tool(filters)

        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            filter_desc = ", ".join([f"{k}: {v}" for k, v in filters.items() if v])

            system_message = SystemMessage(
                content=f"""You are an educational assistant for Indian students.
                Current context: {filter_desc}
                Use simple, age-appropriate language suitable for the class level.
                If the question is a greeting, respond directly.
                Otherwise, use the retrieval tool to get relevant educational content.
                """
            )

            messages = state.get("messages") or []
            messages = [system_message] + messages

            llm_with_tools = llm.bind_tools([retrieval_tool])
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def generate_final_response(state: MessagesState):
            """Generate final response using retrieved content."""
            context = state["messages"][-1].content

            system_message_content = (
                f"You are an educational assistant. Use the following retrieved content "
                f"to answer the question. Current filters: {filters}\n\n"
                f"Question: {state.get('query', '')}\n\n"
                f"Context: {context}"
            )

            conversation_messages = [
                msg
                for msg in state["messages"]
                if msg.type in ("human", "system")
                or (msg.type == "ai" and not getattr(msg, "tool_calls", False))
            ]

            prompt_messages = [
                SystemMessage(system_message_content)
            ] + conversation_messages
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
