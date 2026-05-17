# Flutter Integration Guide — Competitive Exam Student Features

All endpoints are prefixed with your base URL + `/api/v1`. Every request (except auth endpoints) requires the JWT bearer token in the `Authorization` header.

```
Authorization: Bearer <token>
```

---

## Table of Contents

0. [Auth & Onboarding](#0-auth--onboarding)
0.5. [Push Notifications — FCM Setup](#05-push-notifications--fcm-setup)
1. [Chapter Progress](#1-chapter-progress)
2. [MCQ Practice Sessions](#2-mcq-practice-sessions)
3. [Streak & Calendar](#3-streak--calendar)
4. [Study Time](#4-study-time)
5. [Wrong Answer Notebook](#5-wrong-answer-notebook)
6. [Performance Dashboard](#6-performance-dashboard)
7. [Profile Stats](#7-profile-stats)
8. [Notification Inbox](#8-notification-inbox)
9. [Typical UI Flows](#9-typical-ui-flows)
10. [Previous Year Papers](#10-previous-year-papers)
11. [Notes Reader](#11-notes-reader)

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

## 0.5. Push Notifications — FCM Setup

This covers the Flutter-side wiring. The backend already handles FCM delivery via Firebase Admin SDK — you only need to set up the receiving side.

### 1. Dependencies

```yaml
# pubspec.yaml
dependencies:
  firebase_core: ^3.x.x
  firebase_messaging: ^15.x.x
  flutter_local_notifications: ^18.x.x
```

Run `flutter pub get` after adding.

---

### 2. Firebase Project Setup

1. Go to [Firebase Console](https://console.firebase.google.com) → your project → Project Settings → "Your apps"
2. Download `google-services.json` (Android) → place in `android/app/`
3. Download `GoogleService-Info.plist` (iOS) → place in `ios/Runner/`
4. Follow the `firebase_core` FlutterFire setup steps if not already done (`flutterfire configure`)

---

### 3. Android — required config

In `android/app/build.gradle`:
```gradle
apply plugin: 'com.google.gms.google-services'
```

In `android/build.gradle` (project-level):
```gradle
dependencies {
    classpath 'com.google.gms:google-services:4.4.x'
}
```

For high-priority (heads-up) notifications on Android 13+, add to `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
```

---

### 4. iOS — required config

In `ios/Runner/AppDelegate.swift`:
```swift
import UIKit
import Flutter
import FirebaseCore

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    FirebaseApp.configure()
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
```

In Xcode → Runner → Signing & Capabilities → add **Push Notifications** and **Background Modes → Remote notifications**.

---

### 5. Background message handler

Must be a top-level function (not inside a class), annotated with `@pragma`:

```dart
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
  // FCM already shows the notification automatically when app is in background/killed.
  // Only use this for silent data-only messages or local side effects.
}
```

Register it in `main()` **before** `runApp`:

```dart
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
  runApp(const MyApp());
}
```

---

### 6. Request permission + get token

Call this once after the user logs in (e.g. in your auth state listener or home screen `initState`):

```dart
Future<void> initFCM() async {
  // 1. Request permission (required on iOS, Android 13+)
  final settings = await FirebaseMessaging.instance.requestPermission(
    alert: true,
    badge: true,
    sound: true,
  );

  if (settings.authorizationStatus != AuthorizationStatus.authorized) {
    return; // User denied — skip token registration
  }

  // 2. Get token and register with backend
  final token = await FirebaseMessaging.instance.getToken();
  if (token != null) {
    await registerFCMToken(token);
  }

  // 3. Re-register if token rotates
  FirebaseMessaging.instance.onTokenRefresh.listen(registerFCMToken);
}

Future<void> registerFCMToken(String token) async {
  await apiClient.post('/users/fcm-token', data: {'token': token});
}
```

`POST /users/fcm-token` — requires JWT auth header.

Request:
```json
{ "fcm_token": "<device-fcm-token>" }
```

Response:
```json
{ "message": "FCM token updated" }
```

---

### 7. Foreground messages (show banner while app is open)

FCM does **not** show a visual notification when the app is in the foreground. You need `flutter_local_notifications` for this.

```dart
// Set up local notifications channel (do this once in main/initState)
const AndroidNotificationChannel channel = AndroidNotificationChannel(
  'vagmi_high_importance', // must match channel ID used below
  'Vagmi Notifications',
  importance: Importance.high,
);

final flutterLocalNotificationsPlugin = FlutterLocalNotificationsPlugin();

await flutterLocalNotificationsPlugin
    .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
    ?.createNotificationChannel(channel);

// Listen for foreground messages
FirebaseMessaging.onMessage.listen((RemoteMessage message) {
  final notification = message.notification;
  if (notification == null) return;

  flutterLocalNotificationsPlugin.show(
    notification.hashCode,
    notification.title,
    notification.body,
    NotificationDetails(
      android: AndroidNotificationDetails(
        channel.id,
        channel.name,
        icon: '@mipmap/ic_launcher',
      ),
    ),
  );
});
```

---

### 8. Handle notification tap (open app from background/killed)

```dart
// App opened by tapping a notification while in background
FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
  _navigateToInbox();
});

// App was killed — check if launched from a notification tap
final initialMessage = await FirebaseMessaging.instance.getInitialMessage();
if (initialMessage != null) {
  _navigateToInbox();
}

void _navigateToInbox() {
  // Navigate to notification inbox screen
  router.push('/notifications');
}
```

---

### 9. Summary — what triggers a push notification

| Event | Who triggers | Backend call |
|-------|-------------|--------------|
| Admin broadcast | Admin panel → Notifications page | `POST /admin/notifications/send` |
| Milestone unlocked (7-day streak etc.) | Auto on session complete | Internal — no Flutter action needed |
| Wrong answer reminder (≥5 unmastered) | Auto on session complete | Internal — no Flutter action needed |
| Chapter complete | Auto on session complete | Internal — no Flutter action needed |

All events also write to the in-app inbox (`GET /comp/student/notifications`) — so even if the push is missed, the user sees it in the bell icon.

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
    "notes": [ { "id": 1, "title": "Notes PDF", "description": null, "language": "hi", "file_url": "...", "word_count": 3200, "read_time_min": 13, "version": 1 } ],
    "videos": [ { "id": 2, "title": "Concept Video", "file_url": "..." } ]
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

## 10. Previous Year Papers

Papers are scoped to a **Level** (Paper 1 / Paper 2 etc.) and serve as downloadable PDFs.

### Get papers for a level

`GET /comp/student/pyq?level_id=<id>`

Requires JWT. Returns only enabled papers, sorted by sort_order then year descending.

Response:
```json
{
  "data": [
    {
      "id": 1,
      "title": "CTET 2023 Paper 1",
      "year": 2023,
      "num_questions": 150,
      "num_pages": 28,
      "file_url": "https://cdn.example.com/comp/levels/3/previous-year-papers/file.pdf",
      "is_premium": false
    },
    {
      "id": 2,
      "title": "CTET 2022 Paper 1",
      "year": 2022,
      "num_questions": 150,
      "num_pages": 26,
      "file_url": "...",
      "is_premium": true
    }
  ]
}
```

- `year`, `num_questions`, `num_pages` may be `null` if not filled in by admin
- `is_premium: true` → gate behind subscription check before opening the PDF
- `file_url` is a direct DO Spaces URL — open in an in-app PDF viewer or browser

---

## 11. Notes Reader

Notes are Markdown documents (converted from .docx, .xlsx, or .pdf by the backend). Each note may optionally have an audio file for a read-along experience.

### Step 1 — Note list (already in chapter detail)

The chapter detail response (`GET /comp/student/chapters/{chapter_id}`) already includes the notes list — no separate API call needed:

```json
"notes": [
  {
    "id": 1,
    "title": "Child Development",
    "description": "CDP notes for Paper 1",
    "language": "hi",
    "file_url": "...",
    "word_count": 3200,
    "read_time_min": 13,
    "version": 2
  }
]
```

Use `read_time_min` to show "13 min read" on the list card. Render these as cards on the chapter detail screen. When the user taps one, proceed to Step 2.

---

### Step 2 — Open a note (full content)

`GET /comp/student/notes/published/{note_id}`

Requires JWT.

Response:
```json
{
  "data": {
    "id": 1,
    "title": "Child Development",
    "description": "CDP notes for Paper 1",
    "language": "hi",
    "content": "# Child Development\n\n## Introduction\n\nChild development refers to...",
    "content_status": "completed",
    "word_count": 3200,
    "read_time_min": 13,
    "version": 2,
    "file_url": "https://cdn.example.com/comp/chapters/5/student-content/notes/file.docx",
    "audio_url": "https://cdn.example.com/comp/notes/audio/1/audio.mp3",
    "audio_status": "completed",
    "audio_sync_json": null,
    "is_published": true,
    "updated_at": "2025-05-10T08:30:00Z"
  }
}
```

Key fields:

| Field | Notes |
|-------|-------|
| `content` | Full Markdown string — render with a Markdown widget |
| `audio_url` | MP3 for read-along — `null` if audio not generated yet |
| `audio_status` | `"completed"` \| `"processing"` \| `"failed"` \| `null` |
| `audio_sync_json` | Reserved for future word-level sync — always `null` currently |
| `language` | `en`, `hi`, `kn`, `ta`, `te` — use to set text direction / font |
| `version` | Increments each time the note is regenerated |

> **Returns 404** if the note does not exist or is not published yet.

---

### Rendering the Markdown

Use the [`flutter_markdown`](https://pub.dev/packages/flutter_markdown) package:

```yaml
# pubspec.yaml
dependencies:
  flutter_markdown: ^0.7.3
```

```dart
import 'package:flutter_markdown/flutter_markdown.dart';

Markdown(
  data: note.content,
  selectable: true,
)
```

---

### Audio read-along (optional)

If `audio_url` is present and `audio_status == "completed"`, you can play the audio alongside the text. The audio is a plain MP3 — use `just_audio` or `audioplayers` to play it.

The audio is a plain MP3 — play/pause controls are sufficient. `audio_sync_json` is reserved for future word-level highlighting and is `null` for now.

---

### Typical notes flow

```
GET /comp/student/chapters/{id}
  → notes[] in response (id + title + language)
    ↓ user taps a note
GET /comp/student/notes/published/{note_id}
  → render content (Markdown widget)
  → if audio_url present, show play button
```

---

## 9. Typical UI Flows

### Chapter list screen
1. `GET /comp/student/subjects/{subject_id}/chapters` — render chapter cards with progress bar
2. `last_active_chapter_id` in response — scroll to / highlight the in-progress chapter

### Chapter detail screen
1. `GET /comp/student/chapters/{chapter_id}` — render activity groups with accuracy badges, notes list, videos list
2. Tap activity group → `POST /comp/sessions` with `{ "activity_group_id": <id> }` → start MCQ flow
3. Tap note → `GET /comp/student/notes/published/{note_id}` → render Markdown reader (see Section 11)

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
