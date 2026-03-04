"""
Regenerate all chapter artifacts (chapter_summary, one_mark_questions, important_questions)
for every chapter that has embeddings in the DB.

Usage:
    python scripts/regenerate_artifacts.py [--chapter-id 123]

Options:
    --chapter-id    Only regenerate for a specific chapter (optional)
    --dry-run       Print chapters that would be processed without running LLM calls
"""

import argparse
import sys
import os

# Make sure app is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import psycopg
from loguru import logger
from app.core.config import settings
from app.services.activity_ai import (
    generate_chapter_summary,
    generate_one_mark_questions,
    generate_important_questions,
    get_full_chapter_text,
)

ARTIFACT_TYPES = ["chapter_summary", "one_mark_questions", "important_questions"]
GENERATORS = {
    "chapter_summary": generate_chapter_summary,
    "one_mark_questions": generate_one_mark_questions,
    "important_questions": generate_important_questions,
}


def get_postgres_url():
    return settings.POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")


def get_all_chapter_ids(conn) -> list[int]:
    """Return all chapter IDs that have at least one embedding."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT (cmetadata->>'chapter_id')::int
            FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = 'llm_textbooks'
            )
            AND cmetadata->>'chapter_id' IS NOT NULL
            ORDER BY 1
            """
        )
        return [row[0] for row in cur.fetchall()]


def get_medium_name(conn, chapter_id: int) -> str:
    """Resolve medium name for a chapter."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.name
            FROM chapters c
            JOIN subjects s ON s.id = c.subject_id
            JOIN mediums m ON m.id = s.medium_id
            WHERE c.id = %s
            """,
            (chapter_id,),
        )
        row = cur.fetchone()
        return row[0] if row else ""


def delete_existing_artifacts(conn, chapter_id: int):
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chapter_artifacts WHERE chapter_id = %s",
            (chapter_id,),
        )
    conn.commit()


def upsert_artifact(conn, chapter_id: int, artifact_type: str, content: str, status: str, error: str = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chapter_artifacts (chapter_id, artifact_type, status, content, error)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (chapter_id, artifact_type, status, content, error),
        )
    conn.commit()


def process_chapter(conn, chapter_id: int, medium_name: str):
    logger.info(f"[chapter={chapter_id}] Starting artifact regeneration (medium={medium_name!r})")

    delete_existing_artifacts(conn, chapter_id)
    logger.info(f"[chapter={chapter_id}] Deleted existing artifacts")

    for atype in ARTIFACT_TYPES:
        logger.info(f"[chapter={chapter_id}] Generating {atype}...")
        try:
            content = GENERATORS[atype](chapter_id, medium_name)
            upsert_artifact(conn, chapter_id, atype, content, "completed")
            logger.info(f"[chapter={chapter_id}] {atype} done (len={len(content)})")
        except Exception as e:
            logger.error(f"[chapter={chapter_id}] {atype} FAILED: {e}")
            upsert_artifact(conn, chapter_id, atype, "", "failed", str(e))


def main():
    parser = argparse.ArgumentParser(description="Regenerate chapter artifacts")
    parser.add_argument("--chapter-id", type=int, help="Only process this chapter")
    parser.add_argument("--dry-run", action="store_true", help="List chapters without running LLM")
    args = parser.parse_args()

    postgres_url = get_postgres_url()

    with psycopg.connect(postgres_url) as conn:
        if args.chapter_id:
            chapter_ids = [args.chapter_id]
        else:
            chapter_ids = get_all_chapter_ids(conn)

        logger.info(f"Found {len(chapter_ids)} chapters to process: {chapter_ids}")

        if args.dry_run:
            for cid in chapter_ids:
                medium = get_medium_name(conn, cid)
                text = get_full_chapter_text(cid)
                logger.info(f"[chapter={cid}] medium={medium!r} text_len={len(text)}")
            return

        for chapter_id in chapter_ids:
            medium_name = get_medium_name(conn, chapter_id)
            try:
                process_chapter(conn, chapter_id, medium_name)
            except Exception as e:
                logger.error(f"[chapter={chapter_id}] Unexpected error: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
