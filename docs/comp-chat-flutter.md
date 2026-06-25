# Comp AI Chat — Flutter Integration Guide

---

## For the Flutter Dev

### 1. Which users get comp chat?

Check `plan_type` on the subscription from any auth response:

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

- `plan_type == "comp"` → comp chat
- `plan_type == "academic"` → academic chat
- `subscription == null` → no chat

---

### 2. Get subject and chapter IDs

```
GET /api/v1/comp/subjects?level_id=<level_id>
Authorization: Bearer <access_token>
```

```json
{ "data": [{ "id": 12, "name": "General Science" }] }
```

```
GET /api/v1/comp/chapters?subject_id=12
Authorization: Bearer <access_token>
```

```json
{ "data": [{ "id": 47, "name": "Physics — Laws of Motion" }] }
```

`level_id` comes from `subscription.level_id` in the auth response.

---

### 3. Session lifecycle

Every conversation belongs to a **session**. Create one before sending any messages.

#### 3a. Create a session

```
POST /api/v1/comp/agent/sessions
Authorization: Bearer <access_token>
Content-Type: application/json
```

```json
{
  "comp_subject_id": 12,
  "comp_chapter_id": 47
}
```

`comp_chapter_id` is optional. Omit it for a subject-level session (broader answers).

Response (`201`):

```json
{
  "id": 5,
  "comp_subject_id": 12,
  "comp_chapter_id": 47,
  "title": "",
  "thread_id": "comp_u42_s3f1e2a...",
  "created_at": "2026-06-25T10:00:00Z",
  "updated_at": "2026-06-25T10:00:00Z"
}
```

Store `id` — this is your `session_id` for all subsequent calls.

#### 3b. List sessions (resume screen)

```
GET /api/v1/comp/agent/sessions
Authorization: Bearer <access_token>
```

Returns all sessions for the current user, newest first:

```json
[
  {
    "id": 5,
    "comp_subject_id": 12,
    "comp_chapter_id": 47,
    "title": "Explain Newton's first law",
    "thread_id": "comp_u42_s3f1e2a...",
    "created_at": "2026-06-25T10:00:00Z",
    "updated_at": "2026-06-25T10:05:00Z"
  }
]
```

`title` is auto-set from the first message the student sends.

#### 3c. Load message history

```
GET /api/v1/comp/agent/sessions/<session_id>
Authorization: Bearer <access_token>
```

```json
{
  "session": { "id": 5, "title": "Explain Newton's first law", ... },
  "messages": [
    { "role": "human", "content": "Explain Newton's first law" },
    { "role": "ai",    "content": "Newton's first law states that..." }
  ]
}
```

#### 3d. Delete a session

```
DELETE /api/v1/comp/agent/sessions/<session_id>
Authorization: Bearer <access_token>
```

Returns `204 No Content`.

---

### 4. Send a message

```
POST /api/v1/comp/agent/stream-chat
Authorization: Bearer <access_token>
Content-Type: application/json
```

```json
{
  "session_id": 5,
  "message": "Explain Newton's first law"
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `session_id` | yes | From step 3a |
| `message` | yes | Student's question |

---

### 5. Response — SSE stream

```
data: {"type": "token", "content": "Newton's first law"}
data: {"type": "token", "content": " states that..."}
data: {"type": "complete", "content": ""}
```

| Event type | Meaning |
|------------|---------|
| `token` | Append `content` to the chat bubble |
| `complete` | Stream done — stop loading indicator |
| `error` | Show `content` as error message |

---

### 6. Edge cases

**Daily limit hit** — comes through as a normal `token` + `complete`, no special handling:
```
data: {"type": "token", "content": "Daily token limit reached. Please try again tomorrow."}
data: {"type": "complete", "content": ""}
```

**HTTP errors (before stream opens):**

| Status | Meaning |
|--------|---------|
| `401` | Token expired |
| `404` | Session not found or doesn't belong to user |
| `400` | Chapter doesn't belong to that subject (on session create) |

**Conversation memory** — the AI remembers context across turns within the same session. Follow-up questions work without resending history. History is persisted in the database — survives server restarts.
