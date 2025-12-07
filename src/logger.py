from src.db import get_db_connection


def log_sentiment(input_text: str, sentiment: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO sentiment_logs (input_text, sentiment)
            VALUES (%s, %s)
            """,
            (input_text, sentiment),
        )

        conn.commit()
        cur.close()
        conn.close()

        print("Log saved to database.")

    except Exception as e:
        print(f"Failed to log sentiment: {e}")
