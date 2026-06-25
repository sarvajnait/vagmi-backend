# Comp Chat — Sessions API Reference

Full reference for the competitive exam AI chat feature. Covers session management, streaming, history loading, and all edge cases.

---

## Overview

Every conversation is tied to a **session**. A session binds a user to a specific subject (and optionally a chapter). The AI remembers the full conversation history within a session across app restarts — history is stored server-side.

**Typical flow:**
1. Check if user has a `comp` subscription
2. Fetch subject/chapter IDs for the exam level
3. Create (or resume) a session
4. Stream messages back and forth
5. On revisit — list sessions, pick one, load its history, continue

---

## Base URL

```
/api/v1/comp/agent
```

All endpoints require:
```
Authorization: Bearer <access_token>
```

---

## Who gets access

Check `plan_type` in the auth response subscription:

```json
{
  "user": {
    "is_premium": true,
    "subscription": {
      "plan_type": "comp",
      "level_id": 10,
      "is_active": true,
      "ends_at": "2026-12-31"
    }
  }
}
```

- `plan_type == "comp"` → show comp chat
- anything else or `subscription == null` → hide it

`level_id` is what you'll use to fetch subjects.

---

## Get subject and chapter IDs

```
GET /api/v1/comp/subjects?level_id=<level_id>
```

```json
{
  "data": [
    { "id": 12, "name": "General Science" },
    { "id": 13, "name": "Mathematics" }
  ]
}
```

```
GET /api/v1/comp/chapters?subject_id=12
```

```json
{
  "data": [
    { "id": 47, "name": "Physics — Laws of Motion" },
    { "id": 48, "name": "Chemistry — Periodic Table" }
  ]
}
```

---

## Session endpoints

### Create session

```
POST /api/v1/comp/agent/sessions
Content-Type: application/json
```

**Request body:**

```json
{
  "comp_subject_id": 12,
  "comp_chapter_id": 47
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `comp_subject_id` | int | yes | Must exist and belong to user's level |
| `comp_chapter_id` | int | no | If given, must belong to the subject. Omit for subject-level chat (broader answers, no chapter filter) |

**Response `201`:**

```json
{
  "id": 5,
  "comp_subject_id": 12,
  "comp_chapter_id": 47,
  "title": "",
  "thread_id": "comp_u42_s9a3f...",
  "created_at": "2026-06-25T10:00:00Z",
  "updated_at": "2026-06-25T10:00:00Z"
}
```

| Field | Notes |
|-------|-------|
| `id` | Your `session_id` — use this everywhere |
| `title` | Empty on creation. Auto-set to the first 80 chars of the first message sent |
| `thread_id` | Internal LangGraph thread ID — you don't need this, but it's there |

**Errors:**

| Status | Body | Meaning |
|--------|------|---------|
| `404` | `"Comp subject not found"` | Bad `comp_subject_id` |
| `404` | `"Comp chapter not found"` | Bad `comp_chapter_id` |
| `400` | `"Chapter does not belong to subject"` | Mismatched IDs |

---

### List sessions

```
GET /api/v1/comp/agent/sessions
```

Returns all sessions for the current user, sorted by `updated_at` descending (most recently active first).

**Response `200`:**

```json
[
  {
    "id": 5,
    "comp_subject_id": 12,
    "comp_chapter_id": 47,
    "title": "Explain Newton's first law",
    "thread_id": "comp_u42_s9a3f...",
    "created_at": "2026-06-25T10:00:00Z",
    "updated_at": "2026-06-25T10:15:00Z"
  },
  {
    "id": 3,
    "comp_subject_id": 13,
    "comp_chapter_id": null,
    "title": "What topics are in this subject?",
    "thread_id": "comp_u42_s2b1c...",
    "created_at": "2026-06-24T09:00:00Z",
    "updated_at": "2026-06-24T09:10:00Z"
  }
]
```

Returns `[]` if the user has no sessions.

`comp_chapter_id: null` means the session was created at subject level (no chapter filter).

---

### Get session + message history

```
GET /api/v1/comp/agent/sessions/<session_id>
```

**Response `200`:**

```json
{
  "session": {
    "id": 5,
    "comp_subject_id": 12,
    "comp_chapter_id": 47,
    "title": "Explain Newton's first law",
    "thread_id": "comp_u42_s9a3f...",
    "created_at": "2026-06-25T10:00:00Z",
    "updated_at": "2026-06-25T10:15:00Z"
  },
  "messages": [
    { "role": "human", "content": "Explain Newton's first law" },
    { "role": "ai",    "content": "Newton's first law states that an object at rest stays at rest..." },
    { "role": "human", "content": "Give me an exam question on this" },
    { "role": "ai",    "content": "Here is an MCQ based on Newton's first law:..." }
  ]
}
```

`messages` is ordered oldest first. `role` is always `"human"` or `"ai"`.

If the session exists but has no messages yet (just created, no chat sent), `messages` is `[]`.

**Error:**

| Status | Meaning |
|--------|---------|
| `404` | Session not found or belongs to a different user |

---

### Delete session

```
DELETE /api/v1/comp/agent/sessions/<session_id>
```

**Response `204 No Content`**

Removes the session record. The underlying conversation checkpoint is also cleaned up from the LangGraph tables.

**Error:**

| Status | Meaning |
|--------|---------|
| `404` | Session not found or belongs to a different user |

---

## Send a message (streaming)

```
POST /api/v1/comp/agent/stream-chat
Content-Type: application/json
```

**Request body:**

```json
{
  "session_id": 5,
  "message": "Explain Newton's first law in simple terms"
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `session_id` | int | yes | Must be an existing session owned by the current user |
| `message` | str | yes | The student's question |

**Response:** `text/event-stream` (SSE)

The connection stays open while the AI generates. Events arrive as:

```
data: {"type": "token",    "content": "Newton's first law"}
data: {"type": "token",    "content": " states that an object"}
data: {"type": "token",    "content": " at rest stays at rest..."}
data: {"type": "complete", "content": ""}
```

**Event types:**

| `type` | What to do |
|--------|-----------|
| `token` | Append `content` to the current chat bubble (streaming word by word) |
| `complete` | Stream finished — stop the loading spinner, finalize the bubble |
| `error` | Display `content` as an error message to the user |

**Side effects on the server:**
- If this is the first message in the session, `session.title` is set to the first 80 chars of `message`
- `session.updated_at` is updated on every successful stream

**Errors (before stream opens, plain HTTP):**

| Status | Body | Meaning |
|--------|------|---------|
| `401` | — | Token expired — re-authenticate |
| `404` | `"Session not found"` | Bad `session_id` or wrong user |

---

## Conversation memory

- The AI remembers the full conversation within a session
- No need to send history back — just send the new message
- History survives server restarts (stored in Postgres)
- The last 20 turns are used as active context; older turns are trimmed automatically

---

## Daily token limit

When a user hits the daily limit, the stream still opens and returns a plain message through the normal `token`/`complete` flow — no special event type:

```
data: {"type": "token",    "content": "Daily token limit reached. Please try again tomorrow."}
data: {"type": "complete", "content": ""}
```

No error state, just display it as a regular AI message.

---

## Typical UI patterns

### New chat button (from subject or chapter screen)

```
1. POST /comp/agent/sessions  { comp_subject_id, comp_chapter_id }
2. Store session.id
3. Open chat screen with empty messages
4. User sends first message → POST /comp/agent/stream-chat  { session_id, message }
5. Stream tokens into chat bubble
6. On complete, finalize bubble
```

### Resume screen (chat history list)

```
1. GET /comp/agent/sessions
2. Render list sorted by updated_at, show session.title as preview
3. User taps a session →
4. GET /comp/agent/sessions/<id>  → load session + messages
5. Render existing messages in chat UI
6. User continues → POST /comp/agent/stream-chat  { session_id, new_message }
```

### Delete a session

```
1. User swipes or taps delete on session card
2. DELETE /comp/agent/sessions/<id>
3. Remove from local list on 204
```

---

## Field reference

### SessionResponse

```json
{
  "id":               5,
  "comp_subject_id":  12,
  "comp_chapter_id":  47,
  "title":            "Explain Newton's first law",
  "thread_id":        "comp_u42_s9a3f...",
  "created_at":       "2026-06-25T10:00:00Z",
  "updated_at":       "2026-06-25T10:15:00Z"
}
```

### MessageOut

```json
{ "role": "human", "content": "Explain Newton's first law" }
{ "role": "ai",    "content": "Newton's first law states that..." }
```

### SSE event

```json
{ "type": "token",    "content": "<partial text>" }
{ "type": "complete", "content": "" }
{ "type": "error",    "content": "<error message>" }
```
