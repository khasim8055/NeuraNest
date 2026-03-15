# core/database.py
# ================================================================
# NeuraCare — Database layer
# Handles: connection, initialization, encryption, migrations
# ================================================================

import sqlite3
import os
from pathlib import Path
from cryptography.fernet import Fernet
import json
import tempfile

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
DB_FILE    = DATA_DIR / "neuranest.db"
KEY_FILE   = BASE_DIR / "secret.key"
SCHEMA_FILE= BASE_DIR / "schema.sql"

DATA_DIR.mkdir(exist_ok=True)


# ── Encryption helpers ───────────────────────────────────────────
def load_key() -> Fernet:
    """Load encryption key or raise clear error if missing."""
    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"secret.key not found at {KEY_FILE}\n"
            "Run generate_key.py first, or copy your backup key here."
        )
    with open(KEY_FILE, "rb") as f:
        return Fernet(f.read())


def encrypt_db(fernet: Fernet) -> None:
    """Encrypt the SQLite file after closing connection."""
    if not DB_FILE.exists():
        return
    with open(DB_FILE, "rb") as f:
        data = f.read()
    encrypted = fernet.encrypt(data)
    with open(str(DB_FILE) + ".enc", "wb") as f:
        f.write(encrypted)
    os.remove(DB_FILE)


def decrypt_db(fernet: Fernet) -> None:
    """Decrypt the encrypted DB file to a working SQLite file."""
    enc_file = str(DB_FILE) + ".enc"
    if not os.path.exists(enc_file):
        return  # first run — no encrypted file yet
    with open(enc_file, "rb") as f:
        encrypted = f.read()
    data = fernet.decrypt(encrypted)
    with open(DB_FILE, "wb") as f:
        f.write(data)


# ── Connection ───────────────────────────────────────────────────
class Database:
    """
    Context manager for NeuraCare database access.

    Usage:
        with Database() as db:
            db.conn.execute("SELECT * FROM patients")

    Automatically:
    - Decrypts DB on open
    - Enables foreign keys
    - Re-encrypts DB on close
    """

    def __init__(self):
        self.fernet = load_key()
        self.conn   = None

    def __enter__(self):
        decrypt_db(self.fernet)
        self.conn = sqlite3.connect(str(DB_FILE))
        self.conn.row_factory = sqlite3.Row   # access columns by name
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")  # safer writes
        self._initialize_schema()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
        encrypt_db(self.fernet)
        return False  # do not suppress exceptions

    def _initialize_schema(self):
        """Create tables if they do not exist yet (first run)."""
        if SCHEMA_FILE.exists():
            with open(SCHEMA_FILE) as f:
                sql = f.read()
            # Skip placeholder admin insert on first real run
            # Real admin is created by setup wizard
            try:
                self.conn.executescript(sql)
            except sqlite3.OperationalError:
                pass  # tables already exist

    def execute(self, sql: str, params: tuple = ()):
        """Shorthand for single query execution."""
        return self.conn.execute(sql, params)

    def fetchall(self, sql: str, params: tuple = ()):
        """Execute and return all rows as dicts."""
        cursor = self.conn.execute(sql, params)
        cols   = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def fetchone(self, sql: str, params: tuple = ()):
        """Execute and return one row as dict, or None."""
        cursor = self.conn.execute(sql, params)
        cols   = [d[0] for d in cursor.description]
        row    = cursor.fetchone()
        return dict(zip(cols, row)) if row else None


# ── Quick connection test ────────────────────────────────────────
if __name__ == "__main__":
    print("Testing database connection...")
    try:
        with Database() as db:
            tables = db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            print(f"✓ Connected successfully")
            print(f"✓ Tables: {[t['name'] for t in tables]}")
            config = db.fetchall("SELECT key, value FROM app_config")
            print(f"✓ Config: {config}")
    except FileNotFoundError as e:
        print(f"✗ Key error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
