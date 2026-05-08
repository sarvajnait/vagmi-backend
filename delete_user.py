"""
Script to delete a user and all related records by phone number.
Usage: python delete_user.py
"""
import os
import sys

# Load .env manually
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

import psycopg

PHONE = "7406832289"
POSTGRES_URL = os.environ["POSTGRES_URL"]

# psycopg uses postgresql:// not postgresql+psycopg://
conn_str = POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")


def main():
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Find user
            cur.execute("SELECT id, name, phone FROM \"user\" WHERE phone = %s", (PHONE,))
            row = cur.fetchone()
            if not row:
                print(f"No user found with phone {PHONE}")
                sys.exit(1)

            user_id, name, phone = row
            print(f"Found user: id={user_id}, name={name}, phone={phone}")
            print()

            # Show counts before deletion
            tables_with_user_id = [
                ("llmusage", "user_id"),
                ("user_subscriptions", "user_id"),
                ("razorpay_orders", "user_id"),
                ("activity_play_sessions", "user_id"),
                ("comp_activity_play_sessions", "user_id"),
                ("user_streaks", "user_id"),
                ("user_streak_days", "user_id"),
                ("user_milestones", "user_id"),
                ("wrong_answer_entries", "user_id"),
                ("study_time_logs", "user_id"),
                ("user_notifications", "user_id"),
            ]

            print("Records to be deleted:")
            for table, col in tables_with_user_id:
                cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {col} = %s', (user_id,))
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"  {table}: {count} rows")

            # activity_answers are cascade-deleted via activity_play_sessions
            cur.execute(
                "SELECT COUNT(*) FROM activity_answers aa "
                "JOIN activity_play_sessions aps ON aa.session_id = aps.id "
                "WHERE aps.user_id = %s", (user_id,)
            )
            aa_count = cur.fetchone()[0]
            if aa_count > 0:
                print(f"  activity_answers (via sessions): {aa_count} rows")

            cur.execute(
                "SELECT COUNT(*) FROM comp_activity_answers caa "
                "JOIN comp_activity_play_sessions caps ON caa.session_id = caps.id "
                "WHERE caps.user_id = %s", (user_id,)
            )
            caa_count = cur.fetchone()[0]
            if caa_count > 0:
                print(f"  comp_activity_answers (via sessions): {caa_count} rows")

            print()
            if "--yes" in sys.argv:
                print(f"Auto-confirming deletion (--yes flag provided).")
                confirm = "yes"
            else:
                confirm = input(f"Delete user id={user_id} ({name} / {phone}) and all above records? [yes/no]: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                sys.exit(0)

            # Delete non-CASCADE tables first
            cur.execute('DELETE FROM "llmusage" WHERE user_id = %s', (user_id,))
            print(f"  Deleted {cur.rowcount} rows from llmusage")

            cur.execute('DELETE FROM "user_subscriptions" WHERE user_id = %s', (user_id,))
            print(f"  Deleted {cur.rowcount} rows from user_subscriptions")

            cur.execute('DELETE FROM "razorpay_orders" WHERE user_id = %s', (user_id,))
            print(f"  Deleted {cur.rowcount} rows from razorpay_orders")

            # Delete user — CASCADE handles the rest
            cur.execute('DELETE FROM "user" WHERE id = %s', (user_id,))
            print(f"  Deleted user row (id={user_id})")

            conn.commit()
            print()
            print(f"Done. User {phone} and all related records have been deleted.")


if __name__ == "__main__":
    main()
