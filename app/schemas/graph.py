from typing import List, Dict
from langchain_core.documents import Document
from typing_extensions import TypedDict


# LangGraph State
class RAGState(TypedDict):
    """State schema for RAG (Retrieval Augmented Generation) operations in LangGraph.
    
    Attributes:
        messages: List of conversation messages (human, AI, system, tool messages)
        context: List of retrieved documents from the vector store
        query: The current user query being processed
        filters: Dictionary of hierarchical filters (class_level_id, board_id, etc.)
    """
    messages: List
    context: List[Document]
    query: str
    filters: Dict[str, str]
