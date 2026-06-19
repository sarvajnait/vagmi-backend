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

### 3. Send a message

```
POST /api/v1/comp/agent/stream-chat
Authorization: Bearer <access_token>
Content-Type: application/json
```

```json
{
  "message": "Explain Newton's first law",
  "comp_subject_id": 12,
  "comp_chapter_id": 47
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `message` | yes | Student's question |
| `comp_subject_id` | yes | From step 2 |
| `comp_chapter_id` | no | Send when inside a chapter — gives more accurate answers |

---

### 4. Response — SSE stream

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

### 5. Edge cases

**Daily limit hit** — comes through as a normal `token` + `complete`, no special handling:
```
data: {"type": "token", "content": "Daily limit reached. Please try again tomorrow."}
data: {"type": "complete", "content": ""}
```

**HTTP errors (before stream opens):**

| Status | Meaning |
|--------|---------|
| `401` | Token expired |
| `404` | Invalid subject or chapter ID |
| `400` | Chapter doesn't belong to that subject |

**Conversation memory** — the AI remembers context within the same chapter session. Follow-up questions work without resending history. Resets when the student switches chapter.
