"""Utility functions for cleaning up resources (files, embeddings) when deleting entities."""

from sqlmodel import Session, text, select
from loguru import logger
from app.utils.files import delete_from_do
from app.models import (
    LLMTextbook,
    LLMImage,
    LLMNote,
    QAPattern,
    AdditionalNotes,
    StudentTextbook,
    StudentNotes,
    StudentVideo,
    PreviousYearQuestionPaper,
    Chapter,
)
from app.core.agents.graph import (
    COLLECTION_NAME_TEXTBOOKS,
    COLLECTION_NAME_NOTES,
    COLLECTION_NAME_QA,
    COLLECTION_NAME_IMAGES,
)


def delete_embeddings_by_chapter_id(
    session: Session, chapter_id: int, collection_name: str, metadata_key: str
) -> int:
    """
    Delete embeddings from a collection by chapter_id.
    
    Args:
        session: Database session
        chapter_id: Chapter ID to filter by
        collection_name: Name of the collection
        metadata_key: Metadata key to filter by (e.g., 'textbook_id', 'note_id', 'image_id', 'pattern_id')
    
    Returns:
        Number of embeddings deleted
    """
    try:
        query = f"""
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
        )
        AND cmetadata->>'chapter_id' = :chapter_id
        """
        result = session.execute(
            text(query),
            {
                "collection_name": collection_name,
                "chapter_id": str(chapter_id),
            },
        )
        deleted_count = result.rowcount
        logger.info(
            f"Deleted {deleted_count} embeddings from {collection_name} for chapter_id={chapter_id}"
        )
        return deleted_count
    except Exception as e:
        logger.error(f"Error deleting embeddings from {collection_name}: {e}")
        return 0


def delete_embeddings_by_resource_id(
    session: Session,
    resource_id: int,
    collection_name: str,
    metadata_key: str,
) -> int:
    """
    Delete embeddings from a collection by resource ID (textbook_id, note_id, etc.).
    
    Args:
        session: Database session
        resource_id: Resource ID to filter by
        collection_name: Name of the collection
        metadata_key: Metadata key to filter by (e.g., 'textbook_id', 'note_id', 'image_id', 'pattern_id')
    
    Returns:
        Number of embeddings deleted
    """
    try:
        query = f"""
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
        )
        AND cmetadata->>'{metadata_key}' = :resource_id
        """
        result = session.execute(
            text(query),
            {
                "collection_name": collection_name,
                "resource_id": str(resource_id),
            },
        )
        deleted_count = result.rowcount
        logger.info(
            f"Deleted {deleted_count} embeddings from {collection_name} for {metadata_key}={resource_id}"
        )
        return deleted_count
    except Exception as e:
        logger.error(
            f"Error deleting embeddings from {collection_name} for {metadata_key}={resource_id}: {e}"
        )
        return 0


def cleanup_chapter_resources(session: Session, chapter_id: int) -> dict:
    """
    Clean up all resources (files and embeddings) for a chapter.
    
    This includes:
    - LLM Textbooks (files + embeddings)
    - LLM Images (files + embeddings)
    - LLM Notes (files + embeddings)
    - Q&A Patterns (files + embeddings)
    - Additional Notes (no files/embeddings, just DB records)
    - Student Textbooks (files only)
    - Student Notes (files only)
    - Student Videos (files only)
    
    Args:
        session: Database session
        chapter_id: Chapter ID to clean up
    
    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "files_deleted": 0,
        "embeddings_deleted": 0,
        "errors": [],
    }

    try:
        # Get all resources for this chapter
        llm_textbooks = session.exec(
            select(LLMTextbook).where(LLMTextbook.chapter_id == chapter_id)
        ).all()
        llm_images = session.exec(
            select(LLMImage).where(LLMImage.chapter_id == chapter_id)
        ).all()
        llm_notes = session.exec(
            select(LLMNote).where(LLMNote.chapter_id == chapter_id)
        ).all()
        qa_patterns = session.exec(
            select(QAPattern).where(QAPattern.chapter_id == chapter_id)
        ).all()
        student_textbooks = session.exec(
            select(StudentTextbook).where(StudentTextbook.chapter_id == chapter_id)
        ).all()
        student_notes = session.exec(
            select(StudentNotes).where(StudentNotes.chapter_id == chapter_id)
        ).all()
        student_videos = session.exec(
            select(StudentVideo).where(StudentVideo.chapter_id == chapter_id)
        ).all()

        # Delete LLM Textbook files and embeddings
        for textbook in llm_textbooks:
            if textbook.file_url:
                try:
                    delete_from_do(textbook.file_url)
                    stats["files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error deleting textbook file {textbook.file_url}: {e}")
            # Embeddings will be deleted by chapter_id below

        # Delete LLM Image files and embeddings
        for image in llm_images:
            if image.file_url:
                try:
                    delete_from_do(image.file_url)
                    stats["files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error deleting image file {image.file_url}: {e}")
            # Embeddings will be deleted by chapter_id below

        # Delete LLM Note files and embeddings
        for note in llm_notes:
            if note.file_url:
                try:
                    delete_from_do(note.file_url)
                    stats["files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error deleting note file {note.file_url}: {e}")
            # Embeddings will be deleted by chapter_id below

        # Delete Q&A Pattern files and embeddings
        for pattern in qa_patterns:
            if pattern.file_url:
                try:
                    delete_from_do(pattern.file_url)
                    stats["files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error deleting pattern file {pattern.file_url}: {e}")
            # Embeddings will be deleted by chapter_id below

        # Delete Student Textbook files
        for textbook in student_textbooks:
            if textbook.file_url:
                try:
                    delete_from_do(textbook.file_url)
                    stats["files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error deleting student textbook file {textbook.file_url}: {e}")

        # Delete Student Note files
        for note in student_notes:
            if note.file_url:
                try:
                    delete_from_do(note.file_url)
                    stats["files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error deleting student note file {note.file_url}: {e}")

        # Delete Student Video files
        for video in student_videos:
            if video.file_url:
                try:
                    delete_from_do(video.file_url)
                    stats["files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error deleting student video file {video.file_url}: {e}")

        # Delete all embeddings by chapter_id from all collections
        stats["embeddings_deleted"] += delete_embeddings_by_chapter_id(
            session, chapter_id, COLLECTION_NAME_TEXTBOOKS, "chapter_id"
        )
        stats["embeddings_deleted"] += delete_embeddings_by_chapter_id(
            session, chapter_id, COLLECTION_NAME_IMAGES, "chapter_id"
        )
        stats["embeddings_deleted"] += delete_embeddings_by_chapter_id(
            session, chapter_id, COLLECTION_NAME_NOTES, "chapter_id"
        )
        stats["embeddings_deleted"] += delete_embeddings_by_chapter_id(
            session, chapter_id, COLLECTION_NAME_QA, "chapter_id"
        )

        logger.info(
            f"Chapter cleanup completed for chapter_id={chapter_id}: "
            f"{stats['files_deleted']} files, {stats['embeddings_deleted']} embeddings deleted"
        )

    except Exception as e:
        logger.error(f"Error during chapter cleanup for chapter_id={chapter_id}: {e}")
        stats["errors"].append(f"Cleanup error: {e}")

    return stats


def cleanup_subject_resources(session: Session, subject_id: int) -> dict:
    """
    Clean up all resources for all chapters in a subject.
    
    Args:
        session: Database session
        subject_id: Subject ID to clean up
    
    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "chapters_cleaned": 0,
        "total_files_deleted": 0,
        "total_embeddings_deleted": 0,
        "errors": [],
    }

    try:
        # Clean up subject-level resources
        pyq_papers = session.exec(
            select(PreviousYearQuestionPaper).where(
                PreviousYearQuestionPaper.subject_id == subject_id
            )
        ).all()
        for paper in pyq_papers:
            if paper.file_url:
                try:
                    delete_from_do(paper.file_url)
                    stats["total_files_deleted"] += 1
                except Exception as e:
                    stats["errors"].append(
                        f"Error deleting previous year paper file {paper.file_url}: {e}"
                    )

        # Get all chapters for this subject
        chapters = session.exec(
            select(Chapter).where(Chapter.subject_id == subject_id)
        ).all()

        for chapter in chapters:
            chapter_stats = cleanup_chapter_resources(session, chapter.id)
            stats["chapters_cleaned"] += 1
            stats["total_files_deleted"] += chapter_stats["files_deleted"]
            stats["total_embeddings_deleted"] += chapter_stats["embeddings_deleted"]
            stats["errors"].extend(chapter_stats["errors"])

        logger.info(
            f"Subject cleanup completed for subject_id={subject_id}: "
            f"{stats['chapters_cleaned']} chapters, "
            f"{stats['total_files_deleted']} files, "
            f"{stats['total_embeddings_deleted']} embeddings deleted"
        )

    except Exception as e:
        logger.error(f"Error during subject cleanup for subject_id={subject_id}: {e}")
        stats["errors"].append(f"Cleanup error: {e}")

    return stats

