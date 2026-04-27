# Comp Notes — Flutter Integration Guide

This document covers the full content pipeline for **Competitive Exam Notes**: how notes are stored, what the markdown format looks like, every API endpoint, and how to implement the audio + word-sync highlighting feature in Flutter.

---

## 1. Content Pipeline Overview

```
Admin uploads DOCX/XLSX
        ↓
Backend extracts text + images
        ↓
Gemini AI converts to Extended Markdown
        ↓
Stored in DB (content field)
        ↓
Admin publishes note
        ↓
Flutter fetches via API
```

Notes are authored as Word documents (`.docx`) or Excel files (`.xlsx`). The backend converts them into a custom **Extended Markdown** format that Flutter must render. Images from the DOCX are extracted, optimised to WebP, uploaded to CDN, and embedded as standard markdown image tags.

---

## 2. Extended Markdown Format

The content field is standard Markdown with six custom block types layered on top.

### 2.1 Standard Markdown Elements

These render exactly as normal markdown:

| Element | Syntax | Notes |
|---|---|---|
| Headings | `# H1`, `## H2`, `### H3` | H1 = chapter, H2 = section, H3 = sub-section |
| Bold | `**text**` | Key terms |
| Italic | `*text*` | Emphasis |
| Bullet list | `- item` | |
| Numbered list | `1. item` | |
| Table | `\| col \| col \|` | GFM pipe tables |
| Image | `![alt](https://cdn-url)` | CDN-hosted WebP |
| Horizontal rule | `---` | Section separator |

### 2.2 Custom Block Types

Blocks follow this pattern — opening tag on its own line, content in the middle, closing `:::` on its own line:

```
:::block_type
content here (can be multi-line, can contain markdown)
:::
```

#### `:::formula`
Mathematical or scientific formulas, key equations.

```
:::formula
Work = Force × Displacement × cos θ

W = F · d · cos θ  (SI unit: Joule)
:::
```

**Render as:** Distinct card with purple left border, "Formula" label. Monospace or math font for the content.

---

#### `:::shortcut`
Memory tricks, mnemonics, quick tips for fast recall.

```
:::shortcut
Remember: Work is ZERO when force and displacement are perpendicular (cos 90° = 0).
Carrying a bag horizontally → no work done by gravity!
:::
```

**Render as:** Teal/green card with "⚡ Shortcut" label.

---

#### `:::pyq_alert`
Previous year question alerts — exam frequency and pattern info.

```
:::pyq_alert
This topic appeared in CTET 2019, 2021, 2023. Typically 2–3 MCQs per paper.
Focus on: calculating work done, identifying zero-work scenarios.
:::
```

**Render as:** Orange card with "🔥 PYQ Alert" label.

---

#### `:::mistake`
Common errors students make.

```
:::mistake
Do NOT confuse Work (scalar) with Force (vector). Work has no direction.
W = Fd only works when force and displacement are parallel. Use W = Fd·cosθ for the general case.
:::
```

**Render as:** Red card with "⚠️ Common Mistake" label.

---

#### `:::exam_insight`
How the topic is tested in exams — question patterns, difficulty level.

```
:::exam_insight
CTET Paper 2 tests this topic at application level — expect scenario-based MCQs
(e.g., "A boy pushes a wall and fails to move it. Work done is…").
:::
```

**Render as:** Purple card with "⭐ Exam Insight" label.

---

#### `:::solved_example`
Fully worked numerical or conceptual examples.

```
:::solved_example
A force of 10 N moves a box 5 m in the direction of force. Find work done.

W = F × d = 10 × 5 = **50 J**
:::
```

**Render as:** Green card with "✅ Solved Example" label.

---

#### `:::mcq`
Interactive multiple-choice question. Content uses key-value pairs.

```
:::mcq
question: A coolie carries luggage on his head and walks horizontally. Work done by him is?
options: ["Positive", "Negative", "Zero", "Cannot be determined"]
correct: 2
explanation: The force (upward, supporting luggage) is perpendicular to displacement (horizontal), so cos 90° = 0, W = 0.
:::
```

Fields:
| Field | Type | Description |
|---|---|---|
| `question` | string | The question text |
| `options` | JSON array | Answer choices (0-indexed) |
| `correct` | integer | Index of correct answer (0-based) |
| `explanation` | string | Shown after answering |

**Render as:** Interactive card — show options as tappable buttons. On tap: green = correct, red = wrong, reveal explanation.

---

## 3. API Endpoints

Base URL: `/api/v1/comp/student`

All endpoints return JSON. Auth requirements depend on your app's auth middleware (check with backend team).

---

### 3.1 List Published Notes

```
GET /api/v1/comp/student/notes/published
```

**Query params:**

| Param | Type | Required | Description |
|---|---|---|---|
| `comp_chapter_id` | int | one of these | Notes for a chapter |
| `sub_chapter_id` | int | one of these | Notes for a sub-chapter |
| `language` | string | optional | Filter by language code (`en`, `hi`, `kn`, `ta`, `te`) |

**Response** — lightweight list, no content field:

```json
{
  "data": [
    {
      "id": 42,
      "title": "Work, Energy and Machines",
      "description": "Complete notes for CTET Paper 2",
      "language": "en",
      "version": 1,
      "word_count": 6189,
      "read_time_min": 25,
      "updated_at": "2026-04-26T12:00:00Z"
    }
  ]
}
```

Use this for the notes list screen. Do **not** fetch full content here.

---

### 3.2 Get Full Note (with content + audio)

```
GET /api/v1/comp/student/notes/published/{note_id}
```

**Response:**

```json
{
  "data": {
    "id": 42,
    "title": "Work, Energy and Machines",
    "description": "Complete notes for CTET Paper 2",
    "language": "en",
    "version": 1,
    "word_count": 6189,
    "read_time_min": 25,
    "content": "# Work, Energy and Machines\n\n## 1. Work\n\n:::formula\nW = F × d\n:::\n\n...",
    "audio_url": "https://blr1.digitaloceanspaces.com/vagmi/comp/notes/audio/42/audio.wav",
    "audio_status": "completed",
    "updated_at": "2026-04-26T12:00:00Z"
  }
}
```

**Field reference:**

| Field | Type | Notes |
|---|---|---|
| `content` | string | Extended Markdown — render this |
| `audio_url` | string \| null | Direct CDN URL to audio file (.wav or .mp3) |
| `audio_status` | string | `"completed"`, `"processing"`, `"failed"`, or null |
| `language` | string | `en`, `hi`, `kn`, `ta`, `te` |
| `word_count` | int | Approximate word count of rendered content |
| `read_time_min` | int | Estimated reading time in minutes |

---

## 4. Audio Playback

When `audio_status == "completed"` and `audio_url` is present, show an audio player. Load `audio_url` directly into the player — it is a public CDN URL (.wav or .mp3).

### What the audio contains

The audio is a teacher-style narration of the note content. Not every block is read aloud:

| Block type | Audio behaviour |
|---|---|
| Regular text, headings, lists | Narrated naturally |
| `:::formula` | Spoken as "Formula. [content]" |
| `:::shortcut` | Spoken as "Quick tip. [content]" |
| `:::pyq_alert` | Spoken as "Previous year question alert. [content]" |
| `:::mistake` | Spoken as "Common mistake. [content]" |
| `:::exam_insight` | Spoken as "Exam insight. [content]" |
| `:::solved_example` | Spoken as "Solved example. [content]" |
| `:::mcq` | **Skipped** — interactive only, not narrated |
| GFM tables | Replaced with "Refer to the table in the notes." |
| Images | Omitted |

---

## 5. Language Handling

Notes are per-language — a chapter can have the same note in multiple languages as separate records. Use the `language` filter on the list endpoint:

```
GET /notes/published?comp_chapter_id=5&language=hi   → Hindi notes
GET /notes/published?comp_chapter_id=5&language=en   → English notes
```

Language codes: `en`, `hi`, `kn` (Kannada), `ta` (Tamil), `te` (Telugu).

---

## 6. Recommended Implementation Flow

```
1. Fetch note list         GET /notes/published?comp_chapter_id=X
2. User taps a note
3. Fetch full note         GET /notes/published/{id}
4. Render content          Parse Extended Markdown → custom widgets
5. If audio_status==completed && audio_url present:
     Show audio player
6. User plays audio:
     Standard audio player controls (play/pause/seek)
```

