from app.core.config import settings
import psycopg


def main():
    conn_str = settings.POSTGRES_URL
    if conn_str.startswith("postgresql+psycopg://"):
        conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://", 1)
    print(f"Using POSTGRES_URL: {conn_str}")

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, uuid FROM langchain_pg_collection")
            collections = cur.fetchall()
            if not collections:
                print("No collections found in langchain_pg_collection.")
                return

            for name, uuid in collections:
                cur.execute(
                    "SELECT vector_dims(embedding) "
                    "FROM langchain_pg_embedding "
                    "WHERE collection_id = %s "
                    "LIMIT 1",
                    (uuid,),
                )
                row = cur.fetchone()
                dim = row[0] if row else None
                cur.execute(
                    "SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s",
                    (uuid,),
                )
                count = cur.fetchone()[0]
                print(f"{name}: dim={dim}, rows={count}")


if __name__ == "__main__":
    main()
