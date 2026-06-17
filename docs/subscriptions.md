# Subscription Integration Guide

Two subscription types exist: **academic** (scoped to class + board + medium) and **comp** (scoped to a competitive exam level — which encodes exam + language + paper). Everything flows through the same endpoints — the `plan_type` field tells you which is which.

---

## For the Flutter Dev

### 1. Subscription state on login / token refresh

Every auth response includes the user's active subscription (or `null`):

```json
// POST /auth/login  |  POST /auth/verify-otp  |  POST /auth/refresh
{
  "user": {
    "id": 42,
    "phone": "9999999999",
    "name": "Ravi",
    "is_premium": true,
    "subscription": {
      "id": 7,
      "plan_id": 3,
      "plan_name": "UPSC Hindi - Prelims",
      "plan_type": "comp",           // "academic" | "comp"
      "starts_at": "2026-01-01",
      "ends_at": "2026-06-30",
      "status": "active",
      "is_active": true,
      // Academic scope (null for comp plans)
      "class_level_id": null,
      "board_id": null,
      "medium_id": null,
      // Comp scope (null for academic plans)
      "level_id": 10
    }
  },
  "tokens": { ... }
}
```

`is_premium: false` and `subscription: null` means no active subscription.

### 2. Check subscription mid-session

```
GET /users/me
Authorization: Bearer <access_token>
```

Returns the same `UserResponse` as login. Call this any time you need a fresh subscription check without a full re-login.

### 3. Payment flow (user self-subscribes)

#### Step 1 — Create a Razorpay order

```
POST /payments/create-order
Authorization: Bearer <access_token>
Content-Type: application/json

{ "plan_id": 3 }
```

Response:
```json
{
  "data": {
    "razorpay_order_id": "order_AbCdEf123456",
    "amount": 14900,
    "currency": "INR",
    "key_id": "rzp_live_xxxx",
    "plan_name": "UPSC Hindi - Prelims",
    "duration_days": 180
  }
}
```

#### Step 2 — Open Razorpay checkout in Flutter

Use the `razorpay_flutter` package with the values above. On payment success the SDK gives you:
- `razorpay_payment_id`
- `razorpay_order_id`
- `razorpay_signature`

#### Step 3 — Verify payment (creates subscription automatically)

```
POST /payments/verify
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "razorpay_order_id": "order_AbCdEf123456",
  "razorpay_payment_id": "pay_XyZ789",
  "razorpay_signature": "<hmac_signature_from_sdk>"
}
```

Success response (`200`):
```json
{
  "message": "Payment successful",
  "subscription": {
    "id": 8,
    "plan_id": 3,
    "plan_name": "UPSC Hindi - Prelims",
    "starts_at": "2026-06-14",
    "ends_at": "2026-12-11",
    "status": "active"
  }
}
```

On success — store the subscription locally and update `is_premium`. No need to re-fetch `/users/me`; the response has everything.

### 4. Fetching available plans (paywall screen)

```
GET /subscriptions/plans
GET /subscriptions/plans?plan_type=comp
GET /subscriptions/plans?plan_type=academic
GET /subscriptions/plans?level_id=10
```

Response:
```json
{
  "data": [
    {
      "id": 3,
      "name": "UPSC Hindi - Prelims",
      "plan_type": "comp",
      "level_id": 10,
      "level_name": "Prelims",
      "comp_medium_name": "Hindi",
      "exam_name": "UPSC",
      "class_level_id": null,
      "board_id": null,
      "medium_id": null,
      "class_level_name": null,
      "board_name": null,
      "medium_name": null,
      "amount_inr": 149,
      "duration_days": 180,
      "fixed_end_date": null,
      "is_active": true,
      "description": "Full access to UPSC Prelims content in Hindi"
    }
  ]
}
```

Show only `is_active: true` plans. If `fixed_end_date` is set, all subscribers share that end date regardless of when they bought.

### 5. Gating content on the client

The server does not enforce content gating server-side right now — it's handled on the client. Logic:

```dart
bool canAccess(bool isContentPremium, SubscriptionSummary? subscription) {
  if (!isContentPremium) return true;
  if (subscription == null || !subscription.isActive) return false;

  if (subscription.planType == 'academic') {
    // Gate by class/board/medium match
    return user.classLevelId == content.classLevelId &&
           user.boardId == content.boardId &&
           user.mediumId == content.mediumId;
  }

  if (subscription.planType == 'comp') {
    // Gate by level match
    return subscription.levelId == content.levelId;
  }

  return false;
}
```

### 6. Subscription expiry

`ends_at` is a plain date. The `is_active` field in the summary is always `true` (it means "this row is the active one") — don't rely on it for expiry. Check `ends_at >= today` yourself.

---

## For the Admin (how to set up subscriptions)

Go to **Admin Panel → Subscriptions**.

### Creating a plan

1. Click **Create Plan**
2. Toggle **Academic** or **Competitive** at the top of the form
   - Academic: pick Class → Board → Medium
   - Competitive: pick Exam Category → Exam → Medium/Language → Level/Paper
3. Set a name, price (INR), and optionally a **Fixed End Date**
   - Fixed End Date: every subscriber gets access until that date, regardless of when they paid. Useful for academic year cutoffs.
   - Leave blank to use **Duration Days** (rolling access from payment date)
4. Save

One plan per scope combination is enforced (e.g. you can't have two plans for "UPSC Hindi Prelims").

### Assigning a subscription manually (without payment)

1. Go to **Subscriptions → Assign Subscriptions**
2. Search for the student by name or phone
3. Click **Assign** on their row
4. Pick a plan, set start/end dates, status = `active`
5. Save

### Bulk assign

1. Tick the checkboxes on the user rows you want
2. Click **Bulk Assign (N)**
3. Pick plan + dates → Assign

Duplicate active subscriptions for the same user+plan are skipped automatically.

### Managing a user's subscriptions

Click **Manage** on any user row to see all their subscriptions (past and present), edit dates/status, or delete.
