# Comp RAG Chat Agent

## What's Live

**Endpoint:** `POST /api/v1/comp/agent/stream-chat`  
**Auth:** User JWT (`Authorization: Bearer <token>`)  
**Response:** SSE stream — same format as academic chat

**Request:**
```json
{
  "message": "Explain this topic",
  "comp_subject_id": 1,
  "comp_chapter_id": 5
}
```
`comp_chapter_id` is optional — omit for subject-level chat (broader retrieval, no chapter scoping).

**SSE Events:**
```
data: {"type": "token", "content": "..."}
data: {"type": "complete", "content": ""}
data: {"type": "error", "content": "..."}
```

---

## Architecture

```
Flutter  →  POST /api/v1/comp/agent/stream-chat
              │
              ├─ validates CompSubject + CompChapter (belongs-to check)
              ├─ checks daily token limit (settings.DAILY_TOKEN_LIMIT)
              ├─ fetches CompAdditionalNote for chapter (injected into system prompt)
              │
              └─ CompEducationPlatform.create_comp_agent()
                    │
                    └─ LangChain agent (Gemini 2.5 Flash)
                          tool: retrieve_textbook(query)
                            └─ PGVector similarity_search
                                 collection: comp_llm_textbooks
                                 filter:     {chapter_id: str(comp_chapter_id)}
                                 k=5 chunks
```

**pgvector metadata key:** `"chapter_id"` (string) — value is the `comp_chapter_id` or `sub_chapter_id` integer. Same key name as academic collection; different collection name.

**Conversation memory:** `MemorySaver` keyed by `comp_session_{chapter_id}` — persists within a server process lifetime (in-memory, resets on restart).

---

## Files

| File | Role |
|------|------|
| `app/core/agents/comp_graph.py` | `CompEducationPlatform` class + LangChain agent definition |
| `app/api/v1/comp_agent.py` | FastAPI router, SSE streaming, token tracking |
| `app/schemas/chat.py` | `CompChatRequest` schema |
| `app/api/v1/api.py` | Router registration at `/comp/agent` |

---

## Current State of Comp pgvector Collections

| Collection | Chunks | Status |
|------------|--------|--------|
| `comp_llm_textbooks` | 1329 | ✅ embedded, live in agent |
| `comp_llm_notes` | 0 | ⚠️ 106 DB rows, never embedded |
| `comp_qa_patterns` | 0 | ⚠️ 9 DB rows, never embedded |

---

## Adding More Resources (Future)

### 1. Wire `comp_llm_notes` embedding

Currently notes are uploaded (`POST /comp/llm/llm-note`) but the background job doesn't embed them. Steps:
- Add `_async_embed_comp_notes()` in `activity_jobs.py` (same pattern as `_async_embed_comp` for textbooks)
- Collection name: `"comp_llm_notes"` (create this as a new PGVector collection)
- Trigger from `upload_comp_llm_note` via a new job type `"comp_note_process"`
- Metadata key: `"chapter_id"` (consistent with textbooks)

### 2. Add notes tool to the agent

Once notes are embedded, add to `comp_graph.py`:
```python
comp_vector_store_notes = PGVector(
    embeddings=comp_embeddings,
    collection_name="comp_llm_notes",
    connection=CONNECTION_STRING,
)

@tool(response_format="content_and_artifact")
def retrieve_notes(query: str):
    """Retrieve revision notes and short summaries. Use when student asks for notes or key points."""
    ...
```
Add `retrieve_notes` to the `tools=[...]` list in `create_comp_agent`.

### 3. Wire `comp_qa_patterns` embedding

Same pattern. Collection name: `"comp_qa_patterns"` (already defined as `COMP_QA_COLLECTION` in `activity_ai.py`). Add job trigger in `upload_comp_qa_pattern`.

### 4. Add QA tool to the agent

```python
@tool(response_format="content_and_artifact")
def retrieve_qa_patterns(query: str):
    """Retrieve practice problems and solved examples."""
    ...
```

### 5. Embed existing unembedded rows (backfill)

Rows already in DB but not yet in pgvector need a one-time backfill script:
- Fetch all `CompLLMNote` rows with their `file_url`
- Run `_async_embed_comp_notes(file_url, ..., chapter_id=row.comp_chapter_id or row.sub_chapter_id)`
- Same for `CompQAPattern`

---

## Token Tracking

Uses the same `LLMUsage` table and `record_usage_metadata()` as academic chat. Daily limit enforced per-user via `settings.DAILY_TOKEN_LIMIT` (default 200,000 tokens).

---

## User Identity Note

There is currently no `user_type` flag on the `User` model. Flutter must know whether to call `/agent/stream-chat` (academic) or `/comp/agent/stream-chat` (comp) based on app context or the presence of a `UserCompProfile` record. `UserCompProfile` stores `exam_id`, `comp_medium_id`, `level_id` — set via `POST /comp/student/onboarding`. This is the recommended signal for routing.
