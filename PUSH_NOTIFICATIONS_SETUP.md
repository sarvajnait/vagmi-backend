# Push Notifications Setup Guide

## What's Done (Backend + Admin)

- `fcm_token` field added to `User` model
- `notifications` table for history
- `POST /api/v1/users/fcm-token` — Flutter registers device token
- `POST /api/v1/admin/notifications/send` — admin sends notification
- `GET /api/v1/admin/notifications/history` — admin views past sends
- Admin frontend page at `/admin/notifications`

---

## Remaining Backend Steps

### 1. Run Migration

```bash
alembic upgrade head
```

### 2. Create Firebase Project

1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Create project (or use existing)
3. Go to **Project Settings → Service Accounts**
4. Click **Generate new private key** → download JSON file

### 3. Set Environment Variable

Paste the entire service account JSON as a single line in `.env`:

```
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"your-project",...}
```

> Tip: `cat service-account.json | tr -d '\n'` to collapse to one line.

### 4. Install Dependencies

```bash
pip install firebase-admin>=6.5.0
```

Or just `pip install -r requirements.txt` after pulling.

---

## Flutter Developer Guide

### Step 1 — Add FCM to Flutter

In `pubspec.yaml`:
```yaml
dependencies:
  firebase_core: latest
  firebase_messaging: latest
```

Follow [FlutterFire setup](https://firebase.flutter.dev/docs/overview) to add `google-services.json` (Android) and `GoogleService-Info.plist` (iOS).

### Step 2 — Request Permission & Get Token

```dart
import 'package:firebase_messaging/firebase_messaging.dart';

Future<void> initPushNotifications() async {
  await FirebaseMessaging.instance.requestPermission();

  final token = await FirebaseMessaging.instance.getToken();
  if (token != null) {
    await registerFcmToken(token);
  }

  // Refresh token if it rotates
  FirebaseMessaging.instance.onTokenRefresh.listen(registerFcmToken);
}
```

### Step 3 — Register Token with Backend

Call this after the user logs in (has a valid JWT):

```
POST /api/v1/users/fcm-token
Authorization: Bearer <access_token>
Content-Type: application/json

{ "token": "<fcm_device_token>" }
```

Returns `204 No Content` on success.

### Step 4 — Handle Foreground Notifications

```dart
FirebaseMessaging.onMessage.listen((RemoteMessage message) {
  // Show in-app banner or use flutter_local_notifications
  print('Title: ${message.notification?.title}');
  print('Body: ${message.notification?.body}');
});
```

### Step 5 — Background / Terminated State

FCM handles background and killed-state notifications automatically on Android/iOS — no extra code needed for display.

For handling taps when app is in background:
```dart
FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
  // Navigate based on message.data if needed
});
```

### Android — High Priority

The backend already sends `priority: high` for Android so notifications arrive immediately even in Doze mode. No extra config needed.

### iOS — Additional Setup

In Xcode: enable **Push Notifications** capability and **Background Modes → Remote notifications**.

---

## Flow Summary

```
Flutter app starts
  → user logs in
  → get FCM token from FirebaseMessaging
  → POST /api/v1/users/fcm-token  ← store token in DB

Admin opens /admin/notifications
  → fills title + message
  → optionally filters by class / board / medium
  → clicks Send
  → backend queries users with matching fcm_token
  → sends FCM multicast
  → Flutter device shows push notification
```
