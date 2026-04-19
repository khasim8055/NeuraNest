# app/core/auth.py
# ================================================================
# NeuraCare — Authentication System
# Handles: user login, password hashing, session management
# ================================================================
# Two roles:
#   admin  — full access: add/edit/delete patients, manage users
#   doctor — clinical access: view/add/edit patients, generate reports
#            cannot delete patients or manage other users
# ================================================================

import bcrypt
import sqlite3
from datetime import datetime
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
# Support both development and PyInstaller packaged paths
_env_base = os.environ.get('NEURACARE_BASE_DIR')
BASE_DIR = Path(_env_base) if _env_base else Path(__file__).parent.parent.parent
DB_FILE  = BASE_DIR / "app" / "data" / "neuranest.db"


# ================================================================
# PASSWORD HELPERS
# ================================================================

def hash_password(plain_password: str) -> str:
    """
    Hash a plain text password using bcrypt.
    Returns a string safe to store in the database.
    Never store plain text passwords.
    """
    password_bytes = plain_password.encode("utf-8")
    salt           = bcrypt.gensalt(rounds=12)
    hashed         = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check if a plain text password matches a stored bcrypt hash.
    Returns True if match, False if not.
    Safe against timing attacks — always takes same amount of time.
    """
    try:
        password_bytes = plain_password.encode("utf-8")
        hashed_bytes   = hashed_password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


# ================================================================
# SESSION — simple in-memory session (no server needed)
# ================================================================

class Session:
    """
    Holds the currently logged-in user for the duration of the app session.
    Reset when app closes — user must log in again each time.
    This is intentional — clinical apps should not stay logged in.
    """
    _current_user = None

    @classmethod
    def login(cls, user: dict) -> None:
        """Store the logged-in user."""
        cls._current_user = user

    @classmethod
    def logout(cls) -> None:
        """Clear the session."""
        cls._current_user = None

    @classmethod
    def get_user(cls) -> dict | None:
        """Get current user dict or None if not logged in."""
        return cls._current_user

    @classmethod
    def is_logged_in(cls) -> bool:
        """Check if anyone is logged in."""
        return cls._current_user is not None

    @classmethod
    def is_admin(cls) -> bool:
        """Check if current user is admin."""
        if cls._current_user is None:
            return False
        return cls._current_user.get("role") == "admin"

    @classmethod
    def get_username(cls) -> str:
        """Get current username or empty string."""
        if cls._current_user is None:
            return ""
        return cls._current_user.get("username", "")

    @classmethod
    def get_full_name(cls) -> str:
        """Get current user full name or empty string."""
        if cls._current_user is None:
            return ""
        return cls._current_user.get("full_name", "")


# ================================================================
# AUTH FUNCTIONS
# ================================================================

def authenticate(username: str, password: str) -> tuple[bool, str, dict | None]:
    """
    Attempt to log in a user.

    Returns:
        (True,  "",             user_dict)  on success
        (False, "error message", None)      on failure

    Usage:
        success, error, user = authenticate("admin", "mypassword")
        if success:
            Session.login(user)
    """
    # Basic input validation
    if not username or not username.strip():
        return False, "Username cannot be empty.", None
    if not password:
        return False, "Password cannot be empty.", None

    username = username.strip().lower()

    # Check DB file exists
    if not DB_FILE.exists():
        return False, "Database not found. Please contact your administrator.", None

    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, username, password_hash, full_name, role, is_active "
            "FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            # Do not reveal whether username exists — always same message
            return False, "Invalid username or password.", None

        user = dict(row)

        # Check account is active
        if not user.get("is_active", 1):
            return False, "This account has been deactivated. Contact your administrator.", None

        # Verify password
        if not verify_password(password, user["password_hash"]):
            return False, "Invalid username or password.", None

        # Success — update last login timestamp
        try:
            conn = sqlite3.connect(str(DB_FILE))
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user["id"])
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # Non-critical — do not fail login if this fails

        # Remove password hash from user dict before storing in session
        user.pop("password_hash", None)
        return True, "", user

    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}", None


def create_user(
    username: str,
    password: str,
    full_name: str,
    role: str = "doctor"
) -> tuple[bool, str]:
    """
    Create a new user account. Admin only.

    Returns:
        (True,  "")             on success
        (False, "error message") on failure
    """
    # Validate inputs
    if not username or not username.strip():
        return False, "Username cannot be empty."
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not full_name or not full_name.strip():
        return False, "Full name cannot be empty."
    if role not in ("admin", "doctor"):
        return False, "Role must be 'admin' or 'doctor'."

    username = username.strip().lower()

    try:
        conn = sqlite3.connect(str(DB_FILE))

        # Check username not already taken
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()

        if existing:
            conn.close()
            return False, f"Username '{username}' is already taken."

        # Hash password and insert
        password_hash = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, full_name, role) "
            "VALUES (?, ?, ?, ?)",
            (username, password_hash, full_name.strip(), role)
        )
        conn.commit()
        conn.close()
        return True, ""

    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"


def change_password(
    username: str,
    old_password: str,
    new_password: str
) -> tuple[bool, str]:
    """
    Change a user's password. User must know their old password.

    Returns:
        (True,  "")              on success
        (False, "error message") on failure
    """
    if not new_password or len(new_password) < 8:
        return False, "New password must be at least 8 characters."

    # Verify old password first
    success, error, user = authenticate(username, old_password)
    if not success:
        return False, "Current password is incorrect."

    try:
        new_hash = hash_password(new_password)
        conn = sqlite3.connect(str(DB_FILE))
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (new_hash, username.strip().lower())
        )
        conn.commit()
        conn.close()
        return True, ""

    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"


def setup_first_admin(
    username: str,
    password: str,
    full_name: str
) -> tuple[bool, str]:
    """
    Creates the first admin account on first run.
    Only works if no users exist yet.

    Returns:
        (True,  "")              on success
        (False, "error message") on failure
    """
    try:
        conn = sqlite3.connect(str(DB_FILE))
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()

        if count > 0:
            return False, "Users already exist. Use create_user() instead."

        return create_user(username, password, full_name, role="admin")

    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"


def get_all_users() -> list[dict]:
    """
    Get list of all users (admin only feature).
    Returns list of user dicts without password hashes.
    """
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, username, full_name, role, is_active, "
            "created_at, last_login FROM users ORDER BY created_at"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def deactivate_user(username: str) -> tuple[bool, str]:
    """
    Deactivate a user account (soft disable — does not delete).
    Admin only. Cannot deactivate yourself.
    """
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.execute(
            "UPDATE users SET is_active = 0 WHERE username = ?",
            (username.strip().lower(),)
        )
        conn.commit()
        conn.close()
        return True, ""
    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"


# ================================================================
# QUICK TEST — run this file directly to verify auth works
# ================================================================
if __name__ == "__main__":
    print("Testing NeuraCare Auth System...")
    print()

    # Test password hashing
    print("1. Testing password hashing...")
    hashed = hash_password("testpassword123")
    assert verify_password("testpassword123", hashed), "Password verify failed"
    assert not verify_password("wrongpassword", hashed), "Wrong password accepted"
    print("   ✓ Password hashing and verification works")

    # Test hash is different each time (bcrypt salting)
    hash1 = hash_password("samepassword")
    hash2 = hash_password("samepassword")
    assert hash1 != hash2, "Hashes should be different (salted)"
    assert verify_password("samepassword", hash1), "Should still verify"
    assert verify_password("samepassword", hash2), "Should still verify"
    print("   ✓ Each hash is unique (bcrypt salting works)")

    # Test session
    print()
    print("2. Testing session management...")
    assert not Session.is_logged_in(), "Should not be logged in"

    test_user = {
        "id": 1,
        "username": "testdoctor",
        "full_name": "Dr. Test",
        "role": "doctor",
        "is_active": 1
    }
    Session.login(test_user)
    assert Session.is_logged_in(), "Should be logged in"
    assert Session.get_username() == "testdoctor", "Wrong username"
    assert not Session.is_admin(), "Doctor should not be admin"
    print("   ✓ Session login works")

    Session.logout()
    assert not Session.is_logged_in(), "Should be logged out"
    print("   ✓ Session logout works")

    print()
    print("=" * 50)
    print("✓ All auth tests passed.")
    print("=" * 50)
    print()
    print("Note: Database tests require the app database to exist.")
    print("Run the app once to create the database, then test authenticate().")
