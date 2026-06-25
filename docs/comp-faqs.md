# Comp Chapter FAQs — API Reference

FAQs are a list of questions (no answers) attached to a comp chapter. Admins add them via the admin panel; the Flutter app fetches and displays them.

---

## Base URL

```
/api/v1/comp
```

No auth required for the GET endpoint. Mutations are admin-facing (called only from `griidai-admin`).

---

## Endpoints

### Get FAQs for a chapter

```
GET /api/v1/comp/faqs?comp_chapter_id=<id>
```

**Query param:**

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `comp_chapter_id` | int | yes | The chapter ID |

**Response `200`:**

```json
{
  "data": [
    {
      "id": 1,
      "question": "What is Newton's first law?",
      "comp_chapter_id": 47,
      "sort_order": 1,
      "created_at": "2026-06-26T10:00:00Z",
      "updated_at": "2026-06-26T10:00:00Z"
    },
    {
      "id": 2,
      "question": "Define inertia with an example.",
      "comp_chapter_id": 47,
      "sort_order": 2,
      "created_at": "2026-06-26T10:05:00Z",
      "updated_at": "2026-06-26T10:05:00Z"
    }
  ]
}
```

Returns `{ "data": [] }` if no FAQs exist for the chapter.

Results are ordered by `sort_order` (nulls last), then `created_at`.

---

### Create FAQ (admin)

```
POST /api/v1/comp/faqs
Content-Type: application/json
```

```json
{
  "question": "What is Newton's first law?",
  "comp_chapter_id": 47
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `question` | string | yes | The FAQ question text |
| `comp_chapter_id` | int | yes | Must be a valid chapter |
| `sort_order` | int | no | Auto-assigned if omitted (appended to end) |

**Response `200`:**
```json
{ "message": "FAQ created", "data": { "id": 1, ... } }
```

**Error:**
| Status | Body | Meaning |
|--------|------|---------|
| `404` | `"Comp chapter not found"` | Invalid `comp_chapter_id` |

---

### Update FAQ (admin)

```
PUT /api/v1/comp/faqs/<id>
Content-Type: application/json
```

```json
{
  "question": "Updated question text",
  "comp_chapter_id": 47
}
```

**Response `200`:**
```json
{ "message": "FAQ updated", "data": { ... } }
```

**Error:** `404` if FAQ not found.

---

### Delete FAQ (admin)

```
DELETE /api/v1/comp/faqs/<id>
```

**Response `200`:**
```json
{ "message": "FAQ deleted" }
```

**Error:** `404` if FAQ not found.

---

## When to fetch FAQs (Flutter)

Fetch FAQs when the student opens a chapter page or starts a comp chat session for a chapter. Display them as a list of suggested questions the student can tap to start a conversation.

```
GET /api/v1/comp/faqs?comp_chapter_id=<session.comp_chapter_id>
```

If `comp_chapter_id` is null (subject-level session), skip the FAQ fetch — there are no chapter-level FAQs.

---

## Adding FAQs (admin)

1. Open `griidai-admin` → navigate to a comp chapter
2. On the chapter resource types page, click **FAQs**
3. Click **Add FAQ**, enter the question, save
4. Repeat for each question
5. Edit or delete existing FAQs using the buttons on each card

---

## Field reference

```json
{
  "id":               1,
  "question":         "What is Newton's first law?",
  "comp_chapter_id":  47,
  "sort_order":       1,
  "created_at":       "2026-06-26T10:00:00Z",
  "updated_at":       "2026-06-26T10:00:00Z"
}
```
