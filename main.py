import json
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import uvicorn
from loguru import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Any, Dict, List
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings

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


template = """
Answer the question based on the context for Grade {grade} students in India. 
Provide age-appropriate explanations and use simple language suitable for the grade level.
Context: {context}
History: {history}
Question: {question}
Answer:
"""


class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            model_out = self.model(**tokens)
            # Use mean pooling (same as sentence-transformers default)
            embeddings = model_out.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings


# Initialize embedding model (shared across all grades)
embedding_model = MiniLMEmbeddings()

# Dictionary to store vector stores for each grade
vector_stores = {}
memories = {}

# Initialize vector stores and memories for each grade
for grade in range(1, 8):
    vector_stores[grade] = Chroma(
        persist_directory=f"vectorstore_db/grade_{grade}",
        embedding_function=embedding_model,
        collection_name=f"grade_{grade}_collection",
    )
    memories[grade] = ConversationBufferMemory(
        memory_key="history", return_messages=True, input_key="question"
    )


def validate_grade(grade: int) -> bool:
    """Validate if grade is between 1-7"""
    return 1 <= grade <= 7


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


# SSE streaming chat endpoint
@app.post("/stream-chat")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses using SSE"""
    if not validate_grade(chat_request.grade):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid grade: {chat_request.grade}. Must be between 1-7",
        )

    async def generate_response():
        try:
            # Get grade-specific components
            vector_store = vector_stores[chat_request.grade]
            memory = memories[chat_request.grade]

            # Initialize LLM without callbacks - we'll use astream
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                verbose=False,
                temperature=0.3,
                streaming=True,
            )

            # Create grade-specific prompt
            grade_prompt = PromptTemplate(
                input_variables=["history", "context", "question", "grade"],
                template=template,
                partial_variables={"grade": chat_request.grade},
            )

            # Create retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Get relevant documents
            docs = await retriever.aget_relevant_documents(chat_request.message)
            context = "\n".join([doc.page_content for doc in docs])

            # Get conversation history
            history = memory.chat_memory.messages
            history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history])

            # Format the prompt
            formatted_prompt = grade_prompt.format(
                context=context,
                history=history_text,
                question=chat_request.message,
                grade=chat_request.grade,
            )

            # Stream the response using astream
            full_response = ""
            async for chunk in llm.astream(formatted_prompt):
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    full_response += content
                    # Send SSE event
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            # Add to memory
            memory.chat_memory.add_user_message(chat_request.message)
            memory.chat_memory.add_ai_message(full_response)

            # Send source documents
            if docs:
                file_names = [
                    f"**{doc.metadata.get('source_file_path', '')}**" for doc in docs
                ]
                if file_names:
                    sources = "\n".join(list(set(file_names)))
                    sources_content = f"\n\n**Source PDFs (Grade {chat_request.grade}):**\n\n{sources}"
                    yield f"data: {json.dumps({'type': 'sources', 'content': sources_content})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
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
    """Upload PDF to specific grade"""
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
    """Delete a document from specific grade"""
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

        return {"message": f"Document {filename} deleted from Grade {grade}"}
    else:
        return {"error": f"Document {filename} not found in Grade {grade}"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "grades_supported": list(range(1, 8))}


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Rural Student Learning API",
        "description": "Grade-based PDF learning system for Indian students",
        "grades_supported": list(range(1, 8)),
        "endpoints": {
            "upload_pdf": "POST /upload_pdf/ (with grade field)",
            "get_documents": "GET /get_documents/{grade}",
            "stream_chat": "POST /stream-chat (SSE)",
            "delete_document": "DELETE /delete_document/{grade}/{filename}",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
