# Razorpay Payment Integration — Frontend Guide

## Overview

Payment flow is 3 steps:
1. Backend creates a Razorpay order → returns `order_id` + `key_id`
2. Frontend opens Razorpay checkout modal with those details
3. On success, frontend calls backend verify endpoint → subscription is activated

---

## Step 1 — Create Order

Call this when user clicks "Buy / Subscribe".

**`POST /api/v1/payments/create-order`**

Headers:
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

Request body:
```json
{
  "plan_id": 1
}
```

Response:
```json
{
  "data": {
    "razorpay_order_id": "order_ABC123",
    "amount": 9900,
    "currency": "INR",
    "key_id": "rzp_test_xxxxxxxx",
    "plan_name": "KAR ENG Class 10",
    "duration_days": 30
  }
}
```

> `amount` is in **paise** (99 INR = 9900 paise) — pass it directly to Razorpay, do not convert.

---

## Step 2 — Open Razorpay Checkout

Install the Razorpay JS SDK in your HTML:
```html
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
```

Or for React Native use the `react-native-razorpay` package.

### Web (React) example:
```js
const openRazorpay = (orderData) => {
  const options = {
    key: orderData.key_id,
    amount: orderData.amount,
    currency: orderData.currency,
    name: "Vagmi",
    description: orderData.plan_name,
    order_id: orderData.razorpay_order_id,
    handler: async (response) => {
      // Step 3 — verify payment
      await verifyPayment(response);
    },
    prefill: {
      name: user.name,
      contact: user.phone,
    },
    theme: {
      color: "#6366f1",
    },
  };

  const rzp = new window.Razorpay(options);
  rzp.open();
};
```

### React Native example:
```js
import RazorpayCheckout from 'react-native-razorpay';

const openRazorpay = async (orderData) => {
  const options = {
    key: orderData.key_id,
    amount: orderData.amount,
    currency: orderData.currency,
    name: 'Vagmi',
    description: orderData.plan_name,
    order_id: orderData.razorpay_order_id,
    prefill: {
      name: user.name,
      contact: user.phone,
    },
  };

  try {
    const response = await RazorpayCheckout.open(options);
    await verifyPayment(response);
  } catch (error) {
    // user cancelled or payment failed
    console.log('Payment cancelled or failed', error);
  }
};
```

---

## Step 3 — Verify Payment

After Razorpay calls the `handler` (or resolves the promise in RN), you get back three values. Send them to the backend immediately.

**`POST /api/v1/payments/verify`**

Headers:
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

Request body:
```json
{
  "razorpay_order_id": "order_ABC123",
  "razorpay_payment_id": "pay_XYZ789",
  "razorpay_signature": "abc123signature..."
}
```

Success response (`200`):
```json
{
  "message": "Payment successful",
  "subscription": {
    "id": 42,
    "plan_id": 1,
    "plan_name": "KAR ENG Class 10",
    "starts_at": "2026-02-22",
    "ends_at": "2026-03-24",
    "status": "active",
    "is_active": true
  }
}
```

After this call succeeds, refresh the user profile (`GET /api/v1/users/me`) — `is_premium` will be `true`.

Error responses:
| Status | Reason |
|--------|--------|
| `400` | Invalid signature — payment tampered or wrong IDs |
| `400` | Order already processed |
| `404` | Order not found |

---

## Full Flow Summary

```
User taps "Subscribe"
  → POST /api/v1/payments/create-order  { plan_id }
  → Get back order_id, amount, key_id

Open Razorpay modal/sheet
  → User completes payment
  → Razorpay returns { razorpay_order_id, razorpay_payment_id, razorpay_signature }

  → POST /api/v1/payments/verify  { razorpay_order_id, razorpay_payment_id, razorpay_signature }
  → On success → user is now premium
  → Refresh user profile to get updated is_premium: true
```

---

## Getting Available Plans

To show the user which plans are available (before initiating payment):

**`GET /api/v1/subscriptions/plans`**

Optional query params: `class_level_id`, `board_id`, `medium_id`

Response:
```json
{
  "data": [
    {
      "id": 1,
      "name": "KAR ENG Class 10",
      "amount_inr": 99,
      "duration_days": 30,
      "is_active": true,
      "class_level_name": "10",
      "board_name": "KARNATAKA BOARD",
      "medium_name": "ENGLISH",
      "description": "..."
    }
  ]
}
```

Use `amount_inr` for display (e.g. "₹99"). Use `id` as `plan_id` in the create-order call.

---

## Notes

- Always verify on the backend — never trust the frontend alone for payment confirmation
- If the user closes the app after paying but before `/verify` is called, the webhook handles it automatically — subscription will still get created on the backend
- Test with Razorpay test keys (`rzp_test_*`) during development. Use test card `4111 1111 1111 1111`, any future expiry, any CVV
