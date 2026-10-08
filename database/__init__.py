import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _dsn() -> dict:
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "dbname": os.environ.get("POSTGRES_DB", "myopic"),
        "user": os.environ.get("POSTGRES_USER", "myopic"),
        "password": os.environ.get("POSTGRES_PASSWORD", ""),
    }


@contextmanager
def get_connection():
    conn = psycopg2.connect(cursor_factory=psycopg2.extras.RealDictCursor, **_dsn())
    try:
        # fails on a brand-new DB before init_db() has created the `vector`
        # extension yet - harmless to skip just that one time, every
        # connection after the schema exists registers it fine.
        register_vector(conn)
    except Exception:
        conn.rollback()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
