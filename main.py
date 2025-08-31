import json
import os
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import TypedDict

# Load environment variables from .env file
load_dotenv()

# Gemini API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
os.environ["GOOGLE_API_KEY"] = api_key

app = FastAPI()

# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories for each grade
if not os.path.exists("files"):
    os.mkdir("files")

# Create grade-specific directories
for grade in range(1, 8):  # Grades 1-7
    grade_dir = f"files/grade_{grade}"
    vectorstore_dir = f"vectorstore_db/grade_{grade}"

    if not os.path.exists(grade_dir):
        os.makedirs(grade_dir)
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)


# Pydantic model for request validation
class ChatRequest(BaseModel):
    message: str
    grade: int


# LangGraph State for RAG with grade support
class RAGState(TypedDict):
    """State for RAG application with grade-specific context"""

    grade: int
    messages: List
    context: List[Document]
    query: str


# Dictionary to store vector stores and graph instances for each grade
vector_stores = {}
rag_graphs = {}
memory_checkpointer = MemorySaver()

# Initialize embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=False,
    temperature=0.5,
    streaming=True,
)

# Initialize vector stores for each grade
for grade in range(1, 8):
    vector_stores[grade] = Chroma(
        persist_directory=f"vectorstore_db/grade_{grade}",
        embedding_function=embeddings,
        collection_name=f"grade_{grade}_collection",
    )


def validate_grade(grade: int) -> bool:
    """Validate if grade is between 1-7"""
    return 1 <= grade <= 7


def create_grade_specific_retrieval_tool(grade: int):
    """Create a retrieval tool for specific grade"""

    @tool(response_format="content_and_artifact")
    def retrieve_for_grade(query: str):
        """Retrieve educational content for Grade students in India related to the query."""
        try:
            vector_store = vector_stores[grade]
            retrieved_docs = vector_store.similarity_search(query, k=3)

            if not retrieved_docs:
                return f"No relevant content found for Grade {grade}", []

            # Format retrieved content with source information
            serialized = "\n\n".join(
                f"Source: {doc.metadata.get('source_file_path', 'Unknown')}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        except Exception as e:
            logger.error(f"Error in retrieval for grade {grade}: {e}")
            return f"Error retrieving content: {str(e)}", []

    return retrieve_for_grade


def create_rag_graph(grade: int):
    """Create a LangGraph RAG application for specific grade"""

    # Create grade-specific retrieval tool
    retrieval_tool = create_grade_specific_retrieval_tool(grade)

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond directly."""
        # Add grade-specific system message for educational context
        system_message = SystemMessage(
            content=f"""You are an educational assistant for Grade {grade} students in India. 
            Use simple, age-appropriate language suitable for Grade {grade}.
            If the question is a greeting or does not need specific educational content, respond directly.
            Otherwise, use the retrieval tool to get syllabus relevant content.
           """
        )

        # Get the messages and add system message if not present
        messages = state.get("messages") or []

        messages = [system_message] + messages

        llm_with_tools = llm.bind_tools([retrieval_tool])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def generate_final_response(state: MessagesState):
        """Generate final response using retrieved content."""
        # Get the most recent tool messages (retrieved content)
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format retrieved content
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        # Create grade-specific system message for final response
        system_message_content = (
            f"You are an educational assistant for Grade {grade} students in India. "
            "Use the following retrieved educational content to answer the question. "
            f"Explain concepts in simple terms appropriate for Grade {grade} level. "
            "If you don't know the answer, say that you don't know. "
            "Keep your answer clear, concise, and educational.\n\n"
            f"Retrieved Content:\n{docs_content}"
        )

        # Get conversation messages (exclude tool messages for cleaner context)
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not getattr(message, "tool_calls", False))
        ]

        # Create final prompt
        prompt_messages = [
            SystemMessage(system_message_content)
        ] + conversation_messages

        # Generate response
        response = llm.invoke(prompt_messages)
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(MessagesState)

    # Add nodes
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", ToolNode([retrieval_tool]))
    graph_builder.add_node("generate", generate_final_response)

    # Add edges
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # Compile with memory checkpointer
    return graph_builder.compile(checkpointer=memory_checkpointer)


# Initialize RAG graphs for each grade
for grade in range(1, 8):
    rag_graphs[grade] = create_rag_graph(grade)


def handle_get_documents(grade: int):
    """Get documents for a specific grade"""
    if not validate_grade(grade):
        return []

    vector_store = vector_stores[grade]
    try:
        coll = vector_store.get()
        if not coll or not coll.get("metadatas"):
            return []
        source_file_paths = [
            metadata["source_file_path"] for metadata in coll["metadatas"]
        ]
        return list(set(source_file_paths))
    except Exception as e:
        logger.error(f"Error getting documents for grade {grade}: {e}")
        return []


def upload_pdf_to_vectorstore_db(file_path: str, grade: int):
    """Upload PDF to grade-specific vector store"""
    if not validate_grade(grade):
        raise ValueError(f"Invalid grade: {grade}. Must be between 1-7")

    vector_store = vector_stores[grade]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=20, length_function=len
    )

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(text_splitter)

    for doc in docs:
        doc.metadata = {
            "source_file_path": file_path.split("/")[-1],
            "grade": grade,
            "full_path": file_path,
        }

    vector_store.add_documents(docs)
    vector_store.persist()  # Ensure persistence

    logger.info(
        f"Successfully uploaded {len(docs)} documents from {file_path} to Grade {grade}"
    )
    return len(docs)


def delete_document_from_grade(file_name: str, grade: int):
    """Delete a document from specific grade vector store"""
    if not validate_grade(grade):
        return False

    vector_store = vector_stores[grade]
    try:
        coll = vector_store.get()
        ids_to_del = [
            id
            for idx, id in enumerate(coll["ids"])
            if coll["metadatas"][idx]["source_file_path"] == file_name
        ]
        if ids_to_del:
            vector_store._collection.delete(ids_to_del)
            vector_store.persist()
            logger.info(f"Deleted document {file_name} from Grade {grade}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting document {file_name} from grade {grade}: {e}")
        return False


# LangGraph-powered SSE streaming chat endpoint
@app.post("/stream-chat")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses using LangGraph and SSE"""
    if not validate_grade(chat_request.grade):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid grade: {chat_request.grade}. Must be between 1-7",
        )

    async def generate_response():
        try:
            # Get grade-specific RAG graph
            rag_graph = rag_graphs[chat_request.grade]

            # Create thread config for this conversation
            config = {
                "configurable": {"thread_id": f"grade_{chat_request.grade}_session"}
            }

            # Prepare input message
            input_messages = [{"role": "user", "content": chat_request.message}]

            # Stream the graph execution
            full_response = ""
            sources_sent = False

            async for event in rag_graph.astream(
                {"messages": input_messages}, config=config, stream_mode="messages"
            ):
                message, metadata = event
                # Handle AI message content (streaming tokens)
                if isinstance(message, AIMessage) and message.content:
                    # Handle streamed content
                    for chunk in message.content:
                        if chunk:
                            full_response += str(chunk)
                            yield f"data: {json.dumps({'type': 'token', 'content': str(chunk)})}\n\n"

                # Handle tool messages (sources)
                elif message.type == "tool" and not sources_sent:
                    # Extract source information from tool message
                    try:
                        # Get the state to access retrieved documents
                        current_state = rag_graph.get_state(config)

                        # Try to get recent tool executions
                        tool_messages = [
                            msg
                            for msg in current_state.values["messages"]
                            if msg.type == "tool"
                        ]

                        if tool_messages:
                            # Extract source files from the last tool execution
                            last_tool_message = tool_messages[-1]
                            if (
                                hasattr(last_tool_message, "artifact")
                                and last_tool_message.artifact
                            ):
                                sources = []
                                for doc in last_tool_message.artifact:
                                    source_file = doc.metadata.get(
                                        "source_file_path", "Unknown"
                                    )
                                    if source_file != "Unknown":
                                        sources.append(f"**{source_file}**")

                                if sources:
                                    unique_sources = list(set(sources))
                                    sources_content = (
                                        f"\n\n**Source PDFs (Grade {chat_request.grade}):**\n\n"
                                        + "\n".join(unique_sources)
                                    )
                                    yield f"data: {json.dumps({'type': 'sources', 'content': sources_content})}\n\n"
                                    sources_sent = True
                    except Exception as e:
                        logger.error(f"Error extracting sources: {e}")

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Error in LangGraph stream_chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@app.get("/get_documents/{grade}")
def get_documents(grade: int):
    """Get all documents for a specific grade"""
    if not validate_grade(grade):
        return {"error": f"Invalid grade: {grade}. Must be between 1-7", "data": []}

    documents = handle_get_documents(grade)
    return {"grade": grade, "data": documents}


@app.get("/get_all_documents/")
def get_all_documents():
    """Get documents organized by grade"""
    all_documents = {}
    for grade in range(1, 8):
        all_documents[f"grade_{grade}"] = handle_get_documents(grade)
    return {"data": all_documents}


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), grade: int = Form(...)):
    """Upload PDF to specific grade and reinitialize RAG graph"""
    # Validate grade
    if not validate_grade(grade):
        return {"error": f"Invalid grade: {grade}. Must be between 1-7"}

    # Create grade-specific file path
    file_location = f"files/grade_{grade}/{file.filename}"

    logger.info(f"Uploading PDF: {file_location} for Grade {grade}")

    try:
        # Save file
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Upload to vector store
        doc_count = upload_pdf_to_vectorstore_db(file_location, grade)

        # Reinitialize the RAG graph for this grade to include new documents
        rag_graphs[grade] = create_rag_graph(grade)

        logger.info(f"Reinitialized RAG graph for Grade {grade} with new documents")

        return {
            "message": f"PDF uploaded and processed successfully for Grade {grade}",
            "grade": grade,
            "filename": file.filename,
            "documents_processed": doc_count,
        }
    except Exception as e:
        logger.error(f"Error uploading PDF for grade {grade}: {e}")
        return {"error": f"Failed to upload PDF: {str(e)}"}


@app.delete("/delete_document/{grade}/{filename}")
async def delete_document(grade: int, filename: str):
    """Delete a document from specific grade and reinitialize RAG graph"""
    if not validate_grade(grade):
        return {"error": f"Invalid grade: {grade}. Must be between 1-7"}

    success = delete_document_from_grade(filename, grade)
    if success:
        # Also delete physical file
        file_path = f"files/grade_{grade}/{filename}"
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting physical file {file_path}: {e}")

        # Reinitialize the RAG graph for this grade
        rag_graphs[grade] = create_rag_graph(grade)
        logger.info(
            f"Reinitialized RAG graph for Grade {grade} after document deletion"
        )

        return {"message": f"Document {filename} deleted from Grade {grade}"}
    else:
        return {"error": f"Document {filename} not found in Grade {grade}"}


@app.get("/chat_history/{grade}")
def get_chat_history(grade: int, limit: int = 10):
    """Get chat history for a specific grade (last N messages)"""
    if not validate_grade(grade):
        return {"error": f"Invalid grade: {grade}. Must be between 1-7", "data": []}

    try:
        config = {"configurable": {"thread_id": f"grade_{grade}_session"}}
        rag_graph = rag_graphs[grade]

        # Get current state
        state = rag_graph.get_state(config)
        if not state or not state.values.get("messages"):
            return {"grade": grade, "messages": [], "total_messages": 0}

        messages = state.values["messages"]
        # Get last N human and AI messages (exclude tool messages for cleaner history)
        chat_messages = [
            {
                "type": msg.type,
                "content": msg.content,
                "timestamp": getattr(msg, "timestamp", None),
            }
            for msg in messages[-limit * 2 :]  # Get more to account for tool messages
            if msg.type in ("human", "ai")
            and not (hasattr(msg, "tool_calls") and msg.tool_calls)
        ][
            -limit:
        ]  # Take last 'limit' messages

        return {
            "grade": grade,
            "messages": chat_messages,
            "total_messages": len(chat_messages),
        }

    except Exception as e:
        logger.error(f"Error getting chat history for grade {grade}: {e}")
        return {"error": f"Error retrieving chat history: {str(e)}"}


@app.delete("/clear_chat_history/{grade}")
def clear_chat_history(grade: int):
    """Clear chat history for a specific grade"""
    if not validate_grade(grade):
        return {"error": f"Invalid grade: {grade}. Must be between 1-7"}

    try:
        # Reinitialize the RAG graph to clear memory
        rag_graphs[grade] = create_rag_graph(grade)
        logger.info(f"Cleared chat history for Grade {grade}")

        return {"message": f"Chat history cleared for Grade {grade}"}

    except Exception as e:
        logger.error(f"Error clearing chat history for grade {grade}: {e}")
        return {"error": f"Error clearing chat history: {str(e)}"}


@app.get("/graph_info/{grade}")
def get_graph_info(grade: int):
    """Get information about the LangGraph structure for a specific grade"""
    if not validate_grade(grade):
        return {"error": f"Invalid grade: {grade}. Must be between 1-7"}

    try:
        rag_graph = rag_graphs[grade]

        # Get graph structure information
        graph_dict = rag_graph.get_graph().to_json()

        return {
            "grade": grade,
            "nodes": list(graph_dict.get("nodes", {}).keys()),
            "edges": len(graph_dict.get("edges", [])),
            "entry_point": "query_or_respond",
            "description": f"LangGraph RAG pipeline for Grade {grade} with retrieval, generation, and chat history",
        }

    except Exception as e:
        logger.error(f"Error getting graph info for grade {grade}: {e}")
        return {"error": f"Error retrieving graph info: {str(e)}"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "grades_supported": list(range(1, 8)),
        "langgraph_enabled": True,
        "total_graphs": len(rag_graphs),
    }


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Rural Student Learning API with LangGraph RAG",
        "description": "Grade-based PDF learning system for Indian students using LangGraph",
        "grades_supported": list(range(1, 8)),
        "langgraph_features": [
            "Conversational memory",
            "Tool-based retrieval",
            "Grade-specific educational responses",
            "Streaming support",
            "State persistence",
        ],
        "endpoints": {
            "upload_pdf": "POST /upload_pdf/ (with grade field)",
            "get_documents": "GET /get_documents/{grade}",
            "stream_chat": "POST /stream-chat (SSE with LangGraph)",
            "chat": "POST /chat (Non-streaming LangGraph)",
            "delete_document": "DELETE /delete_document/{grade}/{filename}",
            "chat_history": "GET /chat_history/{grade}",
            "clear_chat_history": "DELETE /clear_chat_history/{grade}",
            "graph_info": "GET /graph_info/{grade}",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
