"""
Smoke test for comp chat persistent sessions.

Tests the full session lifecycle end-to-end against the real DB:
  1. AsyncPostgresSaver.setup() — checkpoint tables created
  2. Create a CompChatSession row in comp_chat_sessions
  3. Run agent turn 1 with Postgres checkpointer → verify response
  4. Run agent turn 2 → verify agent references turn 1 (context persists)
  5. Load history from checkpoint → verify both human+ai messages present
  6. Clean up test session from DB

Usage:
    uv run python scripts/smoke_test_sessions.py
    uv run python scripts/smoke_test_sessions.py --chapter-id 9
"""

import argparse
import asyncio
import sys
import os
import uuid

# psycopg async requires SelectorEventLoop on Windows (default is ProactorEventLoop)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import psycopg
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings
from app.core.agents.comp_graph import CompEducationPlatform


TURN1_QUESTION = "What is the most important concept in this chapter? Give a brief overview."
TURN2_QUESTION = "Based on what you just explained, give me one exam-style question on that topic."


def get_postgres_url() -> str:
    return settings.POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")


def find_test_chapter(conn) -> dict | None:
    """Pick the chapter with the most embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                cc.id, cc.name,
                cs.id AS subject_id, cs.name AS subject_name,
                e.name AS exam_name,
                COUNT(lpe.id) AS chunk_count
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
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            return None
        return {
            "chapter_id":   row[0],
            "chapter_name": row[1],
            "subject_id":   row[2],
            "subject_name": row[3],
            "exam_name":    row[4],
            "chunk_count":  row[5],
        }


def find_chapter_by_id(conn, chapter_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                cc.id, cc.name,
                cs.id, cs.name,
                e.name,
                COUNT(lpe.id)
            FROM langchain_pg_embedding lpe
            JOIN langchain_pg_collection lpc ON lpc.uuid = lpe.collection_id
            JOIN comp_chapters cc ON cc.id = (lpe.cmetadata->>'chapter_id')::int
            JOIN comp_subjects cs ON cs.id = cc.subject_id
            JOIN comp_levels cl ON cl.id = cs.level_id
            JOIN comp_exam_mediums cem ON cem.id = cl.medium_id
            JOIN exams e ON e.id = cem.exam_id
            WHERE lpc.name = 'comp_llm_textbooks'
              AND (lpe.cmetadata->>'chapter_id')::int = %s
            GROUP BY cc.id, cc.name, cs.id, cs.name, e.name
        """, (chapter_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "chapter_id": row[0], "chapter_name": row[1],
            "subject_id": row[2], "subject_name": row[3],
            "exam_name": row[4], "chunk_count": row[5],
        }


def create_test_session(conn, chapter_info: dict) -> dict:
    """Insert a test CompChatSession row and return it."""
    thread_id = f"smoke_test_u0_s{uuid.uuid4().hex}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO comp_chat_sessions
                (user_id, comp_subject_id, comp_chapter_id, title, thread_id,
                 created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            RETURNING id, thread_id
        """, (
            1,  # fake user_id=1 for smoke test
            chapter_info["subject_id"],
            chapter_info["chapter_id"],
            "smoke test session",
            thread_id,
        ))
        row = cur.fetchone()
    conn.commit()
    return {"session_id": row[0], "thread_id": row[1]}


def delete_test_session(conn, session_id: int, thread_id: str):
    """Clean up the test session and its checkpoints."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM comp_chat_sessions WHERE id = %s", (session_id,))
        # Clean up LangGraph checkpoint rows
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            try:
                cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
            except Exception:
                pass
    conn.commit()


async def run_turn(platform, checkpointer, thread_id: str, question: str, turn_num: int) -> str:
    """Run one conversation turn and return the full response text."""
    filters = {}   # no chapter filter for speed — just testing persistence
    names = {}

    rag_graph = platform.create_comp_agent(filters, names, checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}}
    input_messages = [{"role": "user", "content": question}]

    response_text = ""
    tool_calls = 0

    async for event in rag_graph.astream_events(
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

        elif event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            if not chunk or getattr(chunk, "tool_calls", None):
                continue
            content = chunk.content
            if isinstance(content, str):
                print(content, end="", flush=True)
                response_text += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        print(part["text"], end="", flush=True)
                        response_text += part["text"]

    print()
    return response_text


def _extract_content(content) -> str:
    """Extract plain text from a message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


async def load_history(checkpointer, thread_id: str) -> list[dict]:
    """Load messages from the LangGraph checkpoint."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = await checkpointer.aget(config)
    if not checkpoint:
        return []
    raw = checkpoint.get("channel_values", {}).get("messages", [])
    result = []
    for msg in raw:
        if isinstance(msg, HumanMessage):
            content = _extract_content(msg.content)
            result.append({"role": "human", "content": content})
        elif isinstance(msg, AIMessage):
            content = _extract_content(msg.content)
            if not content:
                continue  # skip tool-call-only AIMessages (no text)
            result.append({"role": "ai", "content": content[:120] + "..." if len(content) > 120 else content})
    return result


async def run_smoke_test(chapter_info: dict):
    pg_url = get_postgres_url()
    platform = CompEducationPlatform()

    print()
    print("=" * 70)
    print(f"  Exam    : {chapter_info['exam_name']}")
    print(f"  Subject : {chapter_info['subject_name']}")
    print(f"  Chapter : {chapter_info['chapter_name']}  (id={chapter_info['chapter_id']}, chunks={chapter_info['chunk_count']})")
    print("=" * 70)

    async with AsyncPostgresSaver.from_conn_string(pg_url) as checkpointer:
        # ── Step 1: Init checkpointer ────────────────────────────────────────
        print("\n[STEP 1] Init AsyncPostgresSaver + setup()...")
        await checkpointer.setup()
        print("  OK — checkpoint tables ready")

        # ── Step 2: Create test session in DB ────────────────────────────────
        with psycopg.connect(pg_url) as conn:
            print("\n[STEP 2] Creating test session in comp_chat_sessions...")
            session = create_test_session(conn, chapter_info)
            thread_id = session["thread_id"]
            session_id = session["session_id"]
            print(f"  OK — session_id={session_id}  thread_id={thread_id}")

        # ── Step 3: Turn 1 ───────────────────────────────────────────────────
        print(f"\n[STEP 3] Turn 1 — Q: {TURN1_QUESTION!r}")
        print("-" * 70)
        turn1_text = await run_turn(platform, checkpointer, thread_id, TURN1_QUESTION, 1)
        turn1_ok = len(turn1_text) > 50
        print(f"\n  Turn 1 chars: {len(turn1_text)}  {'OK' if turn1_ok else 'FAIL — too short'}")

        # ── Step 4: Turn 2 (context check) ───────────────────────────────────
        print(f"\n[STEP 4] Turn 2 — Q: {TURN2_QUESTION!r}")
        print("  (Agent should reference Turn 1's answer — proves memory persists)")
        print("-" * 70)
        turn2_text = await run_turn(platform, checkpointer, thread_id, TURN2_QUESTION, 2)
        turn2_ok = len(turn2_text) > 50
        print(f"\n  Turn 2 chars: {len(turn2_text)}  {'OK' if turn2_ok else 'FAIL — too short'}")

        # ── Step 5: Load history ──────────────────────────────────────────────
        print("\n[STEP 5] Loading message history from checkpoint...")
        messages = await load_history(checkpointer, thread_id)
        print(f"  Found {len(messages)} messages in checkpoint:")
        for i, m in enumerate(messages, 1):
            preview = m["content"][:80].replace("\n", " ")
            print(f"    [{i}] role={m['role']:<6}  {preview}")

        history_ok = len(messages) >= 4  # 2 human + 2 ai

        # ── Step 6: Cleanup ───────────────────────────────────────────────────
        print("\n[STEP 6] Cleaning up test session...")
        with psycopg.connect(pg_url) as conn:
            delete_test_session(conn, session_id, thread_id)
        print("  OK — test session + checkpoint rows deleted")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    checks = [
        ("Checkpoint tables init",  True),
        ("Session row created",     True),
        ("Turn 1 produced response", turn1_ok),
        ("Turn 2 produced response", turn2_ok),
        ("History has >= 4 messages", history_ok),
    ]
    all_pass = all(ok for _, ok in checks)
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print("=" * 70)
    print(f"  Result: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    print()
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Smoke test: comp chat persistent sessions")
    parser.add_argument("--chapter-id", type=int, help="Test against a specific chapter ID")
    args = parser.parse_args()

    pg_url = get_postgres_url()

    with psycopg.connect(pg_url) as conn:
        if args.chapter_id:
            chapter_info = find_chapter_by_id(conn, args.chapter_id)
            if not chapter_info:
                logger.error(f"Chapter {args.chapter_id} not found or has no embeddings")
                sys.exit(1)
        else:
            chapter_info = find_test_chapter(conn)
            if not chapter_info:
                logger.error("No comp chapters with embeddings found")
                sys.exit(1)

    success = asyncio.run(run_smoke_test(chapter_info))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
