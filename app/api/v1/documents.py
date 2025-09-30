from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Form
from loguru import logger
from app.models import *
from app.schemas import *
import os
from app.core.langgraph.graph import EducationPlatform
from sqlmodel import Session, text
from app.services.database import get_session
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

router = APIRouter()
platform = EducationPlatform()
COLLECTION_NAME = "education_documents"


def process_document_upload(file_path: str, metadata: Dict[str, str]) -> int:
    """Upload document with hierarchical metadata"""
    try:
        loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20, length_function=len
        )
        documents = loader.load_and_split(text_splitter)

        # Add metadata to each document chunk
        for doc in documents:
            doc.metadata.update(
                {
                    "class_level": metadata["class_level"],
                    "board": metadata["board"],
                    "medium": metadata["medium"],
                    "subject": metadata["subject"],
                    "chapter": metadata["chapter"],
                    "source_file": os.path.basename(file_path),
                    "full_path": file_path,
                }
            )

        # Add documents to vector store
        platform.vector_store.add_documents(documents)

        logger.info(f"Successfully uploaded {len(documents)} document chunks")
        return len(documents)

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise


@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    class_level: str = Form(...),
    board: str = Form(...),
    medium: str = Form(...),
    subject: str = Form(...),
    chapter: str = Form(...),
):
    """Upload document with hierarchical classification"""
    try:
        # Create directory structure
        dir_path = f"files/{class_level}/{board}/{medium}/{subject}/{chapter}"
        os.makedirs(dir_path, exist_ok=True)

        file_path = f"{dir_path}/{file.filename}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Upload to vector store
        metadata = {
            "class_level": class_level,
            "board": board,
            "medium": medium,
            "subject": subject,
            "chapter": chapter,
        }

        doc_count = process_document_upload(file_path, metadata)

        return {
            "message": "Document uploaded successfully",
            "metadata": metadata,
            "documents_processed": doc_count,
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=Dict[str, List[DocumentResponse]])
async def get_documents(
    class_level: Optional[str] = Query(None),
    board: Optional[str] = Query(None),
    medium: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    chapter: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    """Get documents based on hierarchical filters"""
    try:
        # Build WHERE clause
        where_conditions = ["1=1"]
        params = {"collection_name": COLLECTION_NAME}

        if class_level:
            where_conditions.append("cmetadata->>'class_level' = :class_level")
            params["class_level"] = class_level
        if board:
            where_conditions.append("cmetadata->>'board' = :board")
            params["board"] = board
        if medium:
            where_conditions.append("cmetadata->>'medium' = :medium")
            params["medium"] = medium
        if subject:
            where_conditions.append("cmetadata->>'subject' = :subject")
            params["subject"] = subject
        if chapter:
            where_conditions.append("cmetadata->>'chapter' = :chapter")
            params["chapter"] = chapter

        where_clause = " AND ".join(where_conditions)

        query_text = f"""
        SELECT DISTINCT cmetadata->>'source_file' as source_file,
               cmetadata->>'class_level' as class_level,
               cmetadata->>'board' as board,
               cmetadata->>'medium' as medium,
               cmetadata->>'subject' as subject,
               cmetadata->>'chapter' as chapter
        FROM langchain_pg_embedding 
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection 
            WHERE name = :collection_name
        ) AND {where_clause}
        ORDER BY class_level, board, medium, subject, chapter, source_file
        """

        result = session.execute(text(query_text), params)
        documents = [
            DocumentResponse(
                filename=row[0],
                class_level=row[1],
                board=row[2],
                medium=row[3],
                subject=row[4],
                chapter=row[5],
            )
            for row in result.fetchall()
        ]

        return {"data": documents}

    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return {"data": []}


@router.delete("/")
async def delete_document(
    delete_request: DeleteRequest, session: Session = Depends(get_session)
):
    """Delete document from vector store"""
    try:
        # Build WHERE clause for finding documents
        where_conditions = [
            "collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name)"
        ]
        params = {"collection_name": COLLECTION_NAME}

        where_conditions.append("cmetadata->>'source_file' = :filename")
        params["filename"] = delete_request.filename

        if delete_request.class_level:
            where_conditions.append("cmetadata->>'class_level' = :class_level")
            params["class_level"] = delete_request.class_level
        if delete_request.board:
            where_conditions.append("cmetadata->>'board' = :board")
            params["board"] = delete_request.board
        if delete_request.medium:
            where_conditions.append("cmetadata->>'medium' = :medium")
            params["medium"] = delete_request.medium
        if delete_request.subject:
            where_conditions.append("cmetadata->>'subject' = :subject")
            params["subject"] = delete_request.subject
        if delete_request.chapter:
            where_conditions.append("cmetadata->>'chapter' = :chapter")
            params["chapter"] = delete_request.chapter

        where_clause = " AND ".join(where_conditions)

        # Delete from vector store
        delete_query = f"DELETE FROM langchain_pg_embedding WHERE {where_clause}"
        result = session.execute(text(delete_query), params)
        session.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "message": f"Document '{delete_request.filename}' deleted successfully",
            "deleted_chunks": result.rowcount,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
