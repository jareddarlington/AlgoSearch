import sqlite3
from contextlib import contextmanager


@contextmanager
def get_db_connection(db_path: str):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
