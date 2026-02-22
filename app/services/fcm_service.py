"""
Firebase Cloud Messaging (FCM) service.

Uses firebase-admin SDK with a service account JSON (stored as env var).
Supports sending multicast messages to a list of FCM tokens.
"""

import json
import os

from loguru import logger

_app = None


def _get_app():
    """Lazy-initialise the Firebase Admin app once."""
    global _app
    if _app is not None:
        return _app

    import firebase_admin
    from firebase_admin import credentials

    sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
    if not sa_json:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_JSON env var is not set. "
            "Paste the Firebase service account JSON as a single-line string."
        )

    sa_dict = json.loads(sa_json)
    cred = credentials.Certificate(sa_dict)
    _app = firebase_admin.initialize_app(cred)
    return _app


def send_multicast(tokens: list[str], title: str, body: str) -> int:
    """
    Send a notification to a list of FCM tokens.

    Returns the number of successful deliveries.
    Silently skips invalid/expired tokens.
    """
    if not tokens:
        return 0

    from firebase_admin import messaging

    _get_app()

    # FCM multicast supports up to 500 tokens per call
    BATCH = 500
    success_count = 0
    for i in range(0, len(tokens), BATCH):
        batch_tokens = tokens[i : i + BATCH]
        message = messaging.MulticastMessage(
            tokens=batch_tokens,
            notification=messaging.Notification(title=title, body=body),
            android=messaging.AndroidConfig(priority="high"),
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(sound="default")
                )
            ),
        )
        response = messaging.send_each_for_multicast(message)
        success_count += response.success_count
        if response.failure_count:
            logger.warning(
                f"FCM batch {i // BATCH + 1}: "
                f"{response.success_count} sent, {response.failure_count} failed"
            )

    return success_count
