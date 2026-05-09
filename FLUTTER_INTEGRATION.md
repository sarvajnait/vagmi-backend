# Flutter Integration Guide — Competitive Exam Student Features

All endpoints are prefixed with your base URL + `/api/v1`. Every request (except auth endpoints) requires the JWT bearer token in the `Authorization` header.

```
Authorization: Bearer <token>
```

---

## Table of Contents

0. [Auth & Onboarding](#0-auth--onboarding)
1. [Chapter Progress](#1-chapter-progress)
2. [MCQ Practice Sessions](#2-mcq-practice-sessions)
3. [Streak & Calendar](#3-streak--calendar)
4. [Study Time](#4-study-time)
5. [Wrong Answer Notebook](#5-wrong-answer-notebook)
6. [Performance Dashboard](#6-performance-dashboard)
7. [Profile Stats](#7-profile-stats)
8. [Notification Inbox](#8-notification-inbox)
9. [Typical UI Flows](#9-typical-ui-flows)

---

## 0. Auth & Onboarding

### Phone Check (does user exist?)
`POST /check-user`  *(no auth header required)*

Request:
```json
{ "phone": "9876543210" }
```

Response:
```json
{ "exists": true }
```

If `exists: false` → show OTP screen to register.
If `exists: true` → show password login screen.

---

### Send OTP
`POST /send-otp`  *(no auth header required)*

Request:
```json
{ "phone": "9876543210" }
```

Response:
```json
{ "otp_secret": "<base64-secret>" }
```

Store `otp_secret` — you need it for `/login-otp`.

---

### Verify OTP + Login (new user)
`POST /login-otp`  *(no auth header required)*

Request:
```json
{
  "phone": "9876543210",
  "otp": "123456",
  "otp_secret": "<base64-secret>"
}
```

Response: `AuthResponse` (see Register below for shape).

---

### Register (new user — after OTP verified)
`POST /register`  *(no auth header required)*

For competitive app users, **do not send** `board`, `medium`, or `grade` — those are for the academic app only.

Request:
```json
{
  "phone": "9876543210",
  "name": "Rajesh Kumar",
  "password": "SecurePass123"
}
```

Response:
```json
{
  "user": {
    "id": 42,
    "phone": "9876543210",
    "name": "Rajesh Kumar",
    "board_id": null,
    "medium_id": null,
    "class_level_id": null,
    "is_premium": false,
    "subscription": null
  },
  "tokens": {
    "access_token": "<jwt>",
    "refresh_token": "<jwt>",
    "token_type": "bearer",
    "expires_at": "2026-06-09T00:00:00Z",
    "refresh_expires_at": "2026-06-09T00:00:00Z"
  }
}
```

Store both tokens. Use `access_token` in the `Authorization` header for all subsequent requests.

---

### Password Login (returning user)
`POST /login`  *(no auth header required)*

Request:
```json
{ "username": "9876543210", "password": "SecurePass123" }
```

Response: same `AuthResponse` shape as Register.

---

### Refresh Token
`POST /refresh`

Use `refresh_token` as the Bearer token for this request only.

Response: new `AuthResponse` with fresh tokens.

---

### Onboarding Flow (after first login/register)

After login, call `GET /comp/student/onboarding`. If it returns `404`, the user hasn't completed onboarding — show the onboarding screens.

**Step 1 — Get exam categories**
`GET /comp/student/exam-categories`

Response:
```json
{ "data": [{ "id": 1, "name": "Teaching Entrance Exam" }, ...] }
```

**Step 2 — Get exams for selected category**
`GET /comp/student/exams?category_id=1`

Response:
```json
{ "data": [{ "id": 1, "name": "CTET", "exam_category_id": 1 }, { "id": 2, "name": "UPTET", "exam_category_id": 1 }] }
```

**Step 3 — Get mediums for selected exam**
`GET /comp/student/exam-mediums?exam_id=1`

Response:
```json
{ "data": [{ "id": 1, "name": "English", "exam_id": 1 }, { "id": 2, "name": "Hindi", "exam_id": 1 }] }
```

**Step 4 — Get papers/levels for selected medium**
`GET /comp/student/exam-levels?medium_id=1`

Response:
```json
{ "data": [{ "id": 1, "name": "Paper 1", "medium_id": 1 }, { "id": 2, "name": "Paper 2", "medium_id": 1 }] }
```

**Step 5 — Save onboarding (upsert)**
`POST /comp/student/onboarding`

All fields are optional — you can save in multiple steps or all at once.

Request:
```json
{
  "exam_id": 1,
  "comp_medium_id": 2,
  "level_id": 3,
  "exam_date": "2025-12-15",
  "daily_commitment_hours": 4
}
```

Response:
```json
{
  "data": {
    "exam_id": 1,
    "exam_name": "CTET",
    "comp_medium_id": 2,
    "medium_name": "Hindi",
    "level_id": 3,
    "level_name": "Paper 1",
    "exam_date": "2025-12-15",
    "daily_commitment_hours": 4,
    "days_until_exam": 220
  }
}
```

**Get onboarding profile (any time)**
`GET /comp/student/onboarding`

Returns same shape as POST response. Returns `404` if onboarding not done yet — use this to gate the home screen.

---

## 1. Chapter Progress

### List chapters for a subject
`GET /comp/student/subjects/{subject_id}/chapters`

Response:
```json
{
  "data": {
    "subject": { "id": 1, "name": "Quantitative Aptitude" },
    "chapters": [
      {
        "id": 10,
        "name": "Percentage",
        "sort_order": 1,
        "total_questions": 40,
        "answered": 20,
        "progress_pct": 50,
        "status": "in_progress",
        "last_active_at": "2026-05-03T14:00:00Z"
      }
    ],
    "last_active_chapter_id": 10,
    "total_questions": 120,
    "total_answered": 60,
    "overall_accuracy_pct": 72
  }
}
```

`status` values: `"not_started"` | `"in_progress"` | `"completed"`

### Chapter detail (activity groups + notes/videos)
`GET /comp/student/chapters/{chapter_id}`

Response:
```json
{
  "data": {
    "chapter": { "id": 10, "name": "Percentage" },
    "activity_groups": [
      {
        "id": 5,
        "name": "Basic Percentage",
        "sort_order": 1,
        "timer_seconds": 600,
        "total_questions": 10,
        "answered": 10,
        "correct": 8,
        "accuracy_pct": 80,
        "status": "completed"
      }
    ],
    "notes": [ { "id": 1, "title": "Notes PDF", "file_url": "..." } ],
    "videos": [ { "id": 2, "title": "Concept Video", "video_url": "..." } ]
  }
}
```

---

## 2. MCQ Practice Sessions

### Start a session

`POST /comp/sessions`

Two modes — pick one:

**Mode A — Activity Group (recommended for chapter detail screen)**
```json
{ "activity_group_id": 5 }
```

**Mode B — Full Chapter**
```json
{ "comp_chapter_id": 10 }
```

Response (both modes):
```json
{
  "data": {
    "session": {
      "id": 99,
      "status": "in_progress",
      "total_questions": 10,
      "correct_count": 0,
      "score": 0,
      "activity_group_id": 5,
      "comp_chapter_id": 10
    },
    "next_activity": {
      "id": 201,
      "type": "mcq",
      "question_text": "What is 20% of 150?",
      "options": ["25", "30", "35", "40"],
      "sort_order": 1
    }
  }
}
```

> `next_activity` is the first question. Show it immediately — no need for a separate fetch.

### Submit an answer

`POST /comp/sessions/{session_id}/answers`

```json
{
  "activity_id": 201,
  "selected_option_index": 2
}
```

`selected_option_index` is **1-based** (1, 2, 3, or 4).

Response:
```json
{
  "data": {
    "is_correct": true,
    "score": 1,
    "correct_answer": {
      "correct_option_index": 2,
      "correct_option_text": "30",
      "answer_description": "20% of 150 = 0.20 × 150 = 30"
    },
    "answer_image_url": null,
    "next_activity": {
      "id": 202,
      "type": "mcq",
      "question_text": "...",
      "options": ["...", "...", "...", "..."]
    },
    "completed": false,
    "session": {
      "id": 99,
      "status": "in_progress",
      "total_questions": 10,
      "correct_count": 1,
      "score": 1
    }
  }
}
```

- When `next_activity` is `null` AND `completed` is `true` → session is done, navigate to result screen.
- On session completion the backend automatically updates the wrong answer notebook and streak.

### Session report (result screen)

`GET /comp/sessions/{session_id}/report`

Response:
```json
{
  "data": {
    "session": {
      "id": 99,
      "status": "completed",
      "total_questions": 10,
      "correct_count": 8,
      "score": 8,
      "duration_seconds": 342,
      "skipped_count": 0,
      "started_at": "2026-05-03T14:00:00Z",
      "completed_at": "2026-05-03T14:05:42Z"
    },
    "answers": [
      {
        "activity_id": 201,
        "question_text": "What is 20% of 150?",
        "options": ["25", "30", "35", "40"],
        "submitted": { "selected_option_index": 2 },
        "correct": {
          "correct_option_index": 2,
          "correct_option_text": "30",
          "answer_description": "..."
        },
        "is_correct": true,
        "score": 1,
        "answer_image_url": null
      }
    ]
  }
}
```

`duration_seconds` = `completed_at - started_at` in seconds. `skipped_count` = questions never submitted.

### List activities for a chapter (browse mode, no session)

`GET /comp/activities?comp_chapter_id=10`

or for a sub-chapter: `GET /comp/activities?sub_chapter_id=20`

Each item includes `"completed": true/false` for the current user.

---

## 3. Streak & Calendar

### Record daily activity (call once per app session)

`POST /comp/student/streak/activity`

Call this when the user opens the app or completes any study activity for the day. Safe to call multiple times — idempotent within the same calendar day.

Response:
```json
{
  "data": {
    "current_streak": 5,
    "longest_streak": 12,
    "new_milestone": { "days": 7, "name": "Week Warrior" }
  }
}
```

`new_milestone` is `null` if no milestone was just unlocked.

### Get streak data (calendar screen)

`GET /comp/student/streak`

Response:
```json
{
  "data": {
    "current_streak": 5,
    "longest_streak": 12,
    "last_activity_date": "2026-05-05",
    "calendar": [
      { "date": "2026-03-07", "active": true },
      { "date": "2026-03-08", "active": false }
    ],
    "milestones": [
      { "days": 7,  "name": "Week Warrior",    "achieved": true,  "achieved_at": "2026-04-10T00:00:00Z" },
      { "days": 14, "name": "Fortnight Fighter","achieved": false, "achieved_at": null },
      { "days": 21, "name": "3-Week Champion",  "achieved": false, "achieved_at": null },
      { "days": 30, "name": "Monthly Master",   "achieved": false, "achieved_at": null },
      { "days": 60, "name": "60-Day Legend",    "achieved": false, "achieved_at": null },
      { "days": 100,"name": "Centurion",        "achieved": false, "achieved_at": null }
    ]
  }
}
```

`calendar` covers the last 60 days. Each entry has `date` (ISO) and `active` (bool).

---

## 4. Study Time

Flutter tracks screen time locally and POSTs it to the backend.

`POST /comp/student/study-time`

```json
{
  "duration_seconds": 1800,
  "date": "2026-05-05"
}
```

`date` is optional — defaults to today (server time). Sends partial time during the session, add more later; the backend accumulates it.

Response:
```json
{
  "data": {
    "total_today_seconds": 3600,
    "total_all_time_seconds": 86400
  }
}
```

---

## 5. Wrong Answer Notebook

### List wrong answers

`GET /comp/student/wrong-answers`

Optional filter: `?comp_chapter_id=10`

Response:
```json
{
  "data": [
    {
      "id": 1,
      "activity_id": 201,
      "comp_chapter_id": 10,
      "chapter_name": "Percentage",
      "activity_group_id": 5,
      "activity_group_name": "Basic Percentage",
      "times_attempted": 3,
      "last_wrong_at": "2026-05-03T14:00:00Z",
      "is_mastered": false,
      "question_text": "What is 20% of 150?",
      "options": ["25", "30", "35", "40"],
      "correct_option_index": 2,
      "answer_description": "..."
    }
  ]
}
```

Only returns entries where `is_mastered = false`. When the user answers correctly in a retry session, the entry is automatically marked as mastered.

### Start a retry session

`POST /comp/student/wrong-answers/retry-session`

Option A — retry all wrong answers in a chapter:
```json
{ "comp_chapter_id": 10 }
```

Option B — retry specific questions:
```json
{ "activity_ids": [201, 205, 210] }
```

Response:
```json
{
  "data": {
    "session_id": 102,
    "total_questions": 5,
    "activities": [
      {
        "id": 201,
        "question_text": "...",
        "options": ["...", "...", "...", "..."],
        "type": "mcq"
      }
    ]
  }
}
```

Use `session_id` with the normal `POST /comp/sessions/{session_id}/answers` flow to submit answers.

---

## 6. Performance Dashboard

`GET /comp/student/performance`

Optional filter: `?level_id=3`

Response:
```json
{
  "data": {
    "overall_accuracy_pct": 74,
    "accuracy_delta_pct": 3,
    "questions_done_total": 320,
    "questions_done_this_week": 45,
    "study_time_today_seconds": 3600,
    "study_time_week_seconds": 18000,
    "subjects": [
      {
        "id": 1,
        "name": "Quantitative Aptitude",
        "total_topics": 12,
        "topics_done": 7,
        "accuracy_pct": 68
      }
    ]
  }
}
```

`accuracy_delta_pct` = this week's accuracy minus last week's (positive = improving).

---

## 7. Profile Stats

`GET /comp/student/profile-stats`

Response:
```json
{
  "data": {
    "current_streak": 5,
    "questions_done": 320,
    "accuracy_pct": 74,
    "wrong_answers_pending_revision": 12
  }
}
```

---

## 8. Notification Inbox

### Get notifications (grouped)

`GET /comp/student/notifications?limit=50&offset=0`

Response:
```json
{
  "data": {
    "unread_count": 3,
    "groups": [
      {
        "label": "Today",
        "items": [
          {
            "id": 1,
            "title": "🏆 Week Warrior unlocked!",
            "body": "You've hit a 7-day streak. Keep it up!",
            "notif_type": "milestone",
            "icon_emoji": "🏆",
            "is_read": false,
            "created_at": "2026-05-05T10:00:00Z"
          }
        ]
      },
      {
        "label": "Yesterday",
        "items": [...]
      },
      {
        "label": "Earlier",
        "items": [...]
      }
    ]
  }
}
```

`notif_type` values: `"milestone"` | `"wrong_answer_reminder"` | `"chapter_complete"` | `"admin_broadcast"`

Empty groups are omitted from the response.

### Unread count (for bell badge)

`GET /comp/student/notifications/unread-count`

```json
{ "data": { "count": 3 } }
```

Poll or call on app resume to update the bell badge.

### Mark specific notifications read

`POST /comp/student/notifications/mark-read`

```json
{ "ids": [1, 2, 3] }
```

```json
{ "data": { "updated": 3 } }
```

### Mark all read

`POST /comp/student/notifications/mark-all-read`

```json
{ "data": { "updated": 5 } }
```

---

## 9. Typical UI Flows

### Chapter list screen
1. `GET /comp/student/subjects/{subject_id}/chapters` — render chapter cards with progress bar
2. `last_active_chapter_id` in response — scroll to / highlight the in-progress chapter

### Chapter detail screen
1. `GET /comp/student/chapters/{chapter_id}` — render activity groups with accuracy badges
2. Tap group → `POST /comp/sessions` with `{ "activity_group_id": <id> }` → start MCQ flow

### MCQ practice loop
```
POST /comp/sessions  →  show first question
  ↓ user picks option
POST /comp/sessions/{id}/answers  →  show result + correct answer
  if next_activity != null  →  show next question
  if completed == true      →  navigate to report screen
GET /comp/sessions/{id}/report
```

### Streak / calendar screen
1. On app open → `POST /comp/student/streak/activity` (once per day)
2. `GET /comp/student/streak` → render 60-day calendar grid + milestone badges

### Wrong answer notebook
1. `GET /comp/student/wrong-answers` (optional `?comp_chapter_id=`)
2. Tap "Retry All" → `POST /comp/student/wrong-answers/retry-session`
3. Use returned `session_id` with normal MCQ answer flow
4. Correctly answered questions auto-disappear from the notebook

### Notification bell
1. On app resume → `GET /comp/student/notifications/unread-count` → update badge
2. Tap bell → `GET /comp/student/notifications` → render grouped list
3. On open inbox → `POST /comp/student/notifications/mark-all-read`

---

## Error Responses

All errors follow:
```json
{ "detail": "Human-readable message" }
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (e.g. answer already submitted, invalid option index) |
| 401 | Missing or invalid JWT |
| 403 | Accessing another user's resource |
| 404 | Resource not found |
