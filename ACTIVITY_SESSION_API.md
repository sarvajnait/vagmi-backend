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

**Response (MCQ):**
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

**Response (Descriptive with AI Feedback):**
```json
{
  "data": {
    "is_correct": true,
    "score": 85,
    "correct_answer": {
      "answer_text": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
      "answer_description": "Detailed explanation of photosynthesis"
    },
    "answer_image_url": "https://...",
    "ai_feedback": {
      "score": 85,
      "feedback": [
        "Good: Correctly explained the basic concept",
        "Good: Mentioned light energy conversion",
        "Improve: Could add more details about chlorophyll",
        "Improve: Missing information about carbon dioxide"
      ]
    },
    "next_activity": {
      "id": 3,
      "type": "mcq",
      ...
    },
    "completed": false,
    "session": {
      "id": 123,
      "correct_count": 2,
      "score": 86,
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
- **AI Evaluation for Descriptive Questions:** Descriptive answers are evaluated by AI immediately upon submission with detailed feedback (3-4 bullet points) and scoring (0-100)
- **Progress Tracking:** Session tracks correct count and total score in real-time
- **Language-Aware Feedback:** AI feedback is generated in the medium's language (e.g., Kannada, Hindi, Tamil, etc.)

## Activity Types

### MCQ
- Must provide `selected_option_index` (1-4)
- Score: 1 if correct, 0 if incorrect
- Returns correct option and description

### Descriptive
- Must provide `submitted_answer_text`
- AI evaluates answer immediately against correct answer
- Score: 0-100 based on AI evaluation
- Considered correct if score >= 70
- Returns `ai_feedback` object with:
  - `score`: Numerical score (0-100)
  - `feedback`: Array of 3-4 bullet points (each starting with "Good:" or "Improve:")
- Feedback is language-aware (matches the medium's language)

## Error Cases

- **404:** Session/activity not found
- **403:** Session doesn't belong to current user
- **400:** Session already completed
- **400:** Answer already submitted for this activity
- **400:** Invalid option index or missing answer text
