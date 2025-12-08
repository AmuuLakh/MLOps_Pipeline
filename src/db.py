import os

import psycopg2


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        database=os.getenv("DB_NAME", "sentiment_logs"),
        user=os.getenv("DB_USER", "mlops"),
        password=os.getenv("DB_PASSWORD", "mlops123"),
    )
