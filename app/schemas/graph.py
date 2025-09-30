from typing import List, Dict
from langchain_core.documents import Document
from typing_extensions import TypedDict


# LangGraph State
class RAGState(TypedDict):
    messages: List
    context: List[Document]
    query: str
    filters: Dict[str, str]
