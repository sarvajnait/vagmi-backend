"""
Smoke test for the comp RAG chat agent.

Picks a real comp chapter with embeddings from the DB, runs the agent against it
with an actual question, and prints the streaming response with tool call details.

Usage:
    python scripts/smoke_test_comp_agent.py
    python scripts/smoke_test_comp_agent.py --chapter-id 47
    python scripts/smoke_test_comp_agent.py --chapter-id 47 --question "What is Newton's first law?"
    python scripts/smoke_test_comp_agent.py --list-chapters
"""

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 output so emoji from tool results don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import psycopg
from loguru import logger
from app.core.config import settings
from app.core.agents.comp_graph import CompEducationPlatform

DEFAULT_QUESTION = "Explain the most important concept in this chapter in detail"


def get_postgres_url() -> str:
    return settings.POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")


def list_embedded_chapters(conn) -> list[dict]:
    """Return all comp chapters that have embeddings, ordered by chunk count desc."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                cc.id            AS chapter_id,
                cc.name          AS chapter_name,
                cs.id            AS subject_id,
                cs.name          AS subject_name,
                e.name           AS exam_name,
                COUNT(lpe.id)    AS chunk_count
            FROM langchain_pg_embedding lpe
            JOIN langchain_pg_collection lpc ON lpc.uuid = lpe.collection_id
            JOIN comp_chapters cc ON cc.id = (lpe.cmetadata->>'chapter_id')::int
            JOIN comp_subjects cs ON cs.id = cc.subject_id
            JOIN comp_levels cl ON cl.id = cs.level_id
            JOIN comp_exam_mediums cem ON cem.id = cl.medium_id
            JOIN exams e ON e.id = cem.exam_id
            WHERE lpc.name = 'comp_llm_textbooks'
              AND lpe.cmetadata->>'chapter_id' IS NOT NULL
            GROUP BY cc.id, cc.name, cs.id, cs.name, e.name
            ORDER BY COUNT(lpe.id) DESC
        """)
        rows = cur.fetchall()
        return [
            {
                "chapter_id":   r[0],
                "chapter_name": r[1],
                "subject_id":   r[2],
                "subject_name": r[3],
                "exam_name":    r[4],
                "chunk_count":  r[5],
            }
            for r in rows
        ]


def find_chapter(conn, chapter_id: int | None) -> dict | None:
    """
    If chapter_id given, return that chapter (must have embeddings).
    Otherwise return the chapter with the most chunks.
    """
    chapters = list_embedded_chapters(conn)
    if not chapters:
        return None
    if chapter_id is None:
        return chapters[0]
    for ch in chapters:
        if ch["chapter_id"] == chapter_id:
            return ch
    return None


async def run_agent(chapter_info: dict, question: str) -> bool:
    """Run the agent and stream output. Returns True if answer was produced."""
    platform = CompEducationPlatform()

    filters = {"chapter_id": str(chapter_info["chapter_id"])}
    names = {
        "exam":    chapter_info["exam_name"],
        "subject": chapter_info["subject_name"],
        "chapter": chapter_info["chapter_name"],
    }

    print()
    print("=" * 65)
    print(f"  Exam    : {chapter_info['exam_name']}")
    print(f"  Subject : {chapter_info['subject_name']}")
    print(f"  Chapter : {chapter_info['chapter_name']}  (id={chapter_info['chapter_id']}, chunks={chapter_info['chunk_count']})")
    print(f"  Filter  : chapter_id = {filters['chapter_id']!r}")
    print(f"  Q       : {question!r}")
    print("=" * 65)
    print()

    agent = platform.create_comp_agent(filters, names)

    thread_id = f"smoke_test_comp_{chapter_info['chapter_id']}"
    config = {"configurable": {"thread_id": thread_id}}
    input_messages = [{"role": "user", "content": question}]

    response_chars = 0
    tool_calls = 0
    got_content = False

    async for event in agent.astream_events(
        {"messages": input_messages, "query": question},
        config=config,
        version="v2",
    ):
        if not isinstance(event, dict):
            continue

        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "on_tool_start":
            tool_calls += 1
            query = (data.get("input") or {}).get("query", "")
            print(f"\n[TOOL #{tool_calls}] retrieve_textbook({query!r})\n", flush=True)

        elif event_type == "on_tool_end":
            output = data.get("output")
            if output:
                text = str(output[0]) if isinstance(output, tuple) else str(output)
                preview = text[:300].replace("\n", " ")
                print(f"[RETRIEVED] {preview}{'...' if len(text) > 300 else ''}\n", flush=True)

        elif event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            if not chunk:
                continue
            if getattr(chunk, "tool_calls", None):
                continue

            content = chunk.content
            if isinstance(content, str) and content:
                print(content, end="", flush=True)
                response_chars += len(content)
                got_content = True
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        text = part["text"]
                        print(text, end="", flush=True)
                        response_chars += len(text)
                        got_content = True

    print()
    print()
    print("=" * 65)
    print(f"  Tool calls  : {tool_calls}")
    print(f"  Response    : {response_chars} chars")
    print(f"  Result      : {'PASS — agent produced a response' if got_content else 'FAIL — no content returned'}")
    print("=" * 65)
    print()

    return got_content


def main():
    parser = argparse.ArgumentParser(description="Smoke test: comp RAG chat agent against real DB embeddings")
    parser.add_argument("--chapter-id", type=int, help="Test a specific comp chapter ID")
    parser.add_argument("--question",   type=str, default=DEFAULT_QUESTION, help="Question to ask the agent")
    parser.add_argument("--list-chapters", action="store_true", help="List all chapters with embeddings and exit")
    args = parser.parse_args()

    postgres_url = get_postgres_url()

    with psycopg.connect(postgres_url) as conn:
        if args.list_chapters:
            chapters = list_embedded_chapters(conn)
            if not chapters:
                print("No comp chapters have embeddings in comp_llm_textbooks.")
                return
            print(f"\n{'ID':>6}  {'Chunks':>6}  {'Exam':<20}  {'Subject':<25}  Chapter")
            print("-" * 90)
            for ch in chapters:
                print(f"{ch['chapter_id']:>6}  {ch['chunk_count']:>6}  {ch['exam_name']:<20}  {ch['subject_name']:<25}  {ch['chapter_name']}")
            print()
            return

        chapter_info = find_chapter(conn, args.chapter_id)

    if not chapter_info:
        if args.chapter_id:
            logger.error(f"Chapter ID {args.chapter_id} not found in comp_llm_textbooks embeddings")
        else:
            logger.error("No comp chapters with embeddings found. Has the embedding job run?")
        sys.exit(1)

    success = asyncio.run(run_agent(chapter_info, args.question))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
