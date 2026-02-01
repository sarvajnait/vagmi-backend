# Activity Session API Documentation

## Overview
The Activity Session API allows users to play through chapter activities one question at a time. Users create a session, answer questions sequentially, and receive immediate feedback.

## Workflow

### 1. Create Session
Start a new activity session for a chapter.

**Endpoint:** `POST /api/v1/activities/sessions`

**Request:**
```json
{
  "chapter_id": 1
}
```

**Response:**
```json
{
  "data": {
    "session": {
      "id": 123,
      "user_id": 1,
      "chapter_id": 1,
      "status": "in_progress",
      "total_questions": 10,
      "correct_count": 0,
      "score": 0,
      "started_at": "2024-01-01T10:00:00Z"
    },
    "next_activity": {
      "id": 1,
      "type": "mcq",
      "question_text": "What is 2+2?",
      "options": ["1", "2", "3", "4"],
      ...
    }
  }
}
```

### 2. Submit Answer
Submit an answer for the current question and get the next one.

**Endpoint:** `POST /api/v1/activities/sessions/{session_id}/answers`

**Request (MCQ):**
```json
{
  "activity_id": 1,
  "selected_option_index": 4
}
```

**Request (Descriptive):**
```json
{
  "activity_id": 2,
  "submitted_answer_text": "Your answer here"
}
```

**Response:**
```json
{
  "data": {
    "is_correct": true,
    "score": 1,
    "correct_answer": {
      "correct_option_index": 4,
      "correct_option_text": "4",
      "answer_description": "2+2 equals 4"
    },
    "answer_image_url": null,
    "next_activity": {
      "id": 2,
      "type": "descriptive",
      "question_text": "Explain photosynthesis",
      ...
    },
    "completed": false,
    "session": {
      "id": 123,
      "correct_count": 1,
      "score": 1,
      ...
    }
  }
}
```

**When session completes:**
- `completed` will be `true`
- `next_activity` will be `null`
- Session status changes to `"completed"`

### 3. Get Session Report
View all answers after completing the session.

**Endpoint:** `GET /api/v1/activities/sessions/{session_id}/report`

**Response:**
```json
{
  "data": {
    "session": {
      "id": 123,
      "status": "completed",
      "total_questions": 10,
      "correct_count": 8,
      "score": 85,
      "completed_at": "2024-01-01T10:15:00Z"
    },
    "answers": [
      {
        "activity_id": 1,
        "type": "mcq",
        "question_text": "What is 2+2?",
        "options": ["1", "2", "3", "4"],
        "submitted": {
          "selected_option_index": 4
        },
        "correct": {
          "correct_option_index": 4,
          "correct_option_text": "4",
          "answer_description": "2+2 equals 4"
        },
        "is_correct": true,
        "score": 1
      }
      // ... more answers
    ]
  }
}
```

## Key Features

- **One Question at a Time:** Each answer submission returns the next question
- **No Re-submission:** Cannot answer the same question twice in a session (unique constraint)
- **Immediate Feedback:** Get correctness and correct answer after each submission
- **Auto-completion:** Session automatically marked complete when all questions answered
- **AI Evaluation:** Descriptive answers evaluated by AI with feedback and scoring (0-100)
- **Progress Tracking:** Session tracks correct count and total score in real-time

## Activity Types

### MCQ
- Must provide `selected_option_index` (1-4)
- Score: 1 if correct, 0 if incorrect
- Returns correct option and description

### Descriptive
- Must provide `submitted_answer_text`
- AI evaluates answer against correct answer
- Score: 0-100 based on AI evaluation
- Considered correct if score >= 70
- Returns AI feedback with evaluation details

## Error Cases

- **404:** Session/activity not found
- **403:** Session doesn't belong to current user
- **400:** Session already completed
- **400:** Answer already submitted for this activity
- **400:** Invalid option index or missing answer text
