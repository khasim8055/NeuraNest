# app/tests/test_auth.py
# ================================================================
# NeuraCare — Tests for auth.py
# Run with: pytest app/tests/test_auth.py -v
# ================================================================

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.auth import (
    hash_password,
    verify_password,
    Session,
)


# ================================================================
# PASSWORD TESTS
# ================================================================

def test_hash_password_returns_string():
    """hash_password must return a non-empty string."""
    result = hash_password("mypassword123")
    assert isinstance(result, str)
    assert len(result) > 0


def test_verify_password_correct():
    """Correct password must verify successfully."""
    hashed = hash_password("correctpassword")
    assert verify_password("correctpassword", hashed) is True


def test_verify_password_wrong():
    """Wrong password must not verify."""
    hashed = hash_password("correctpassword")
    assert verify_password("wrongpassword", hashed) is False


def test_verify_password_empty():
    """Empty password must not verify."""
    hashed = hash_password("correctpassword")
    assert verify_password("", hashed) is False


def test_hash_is_unique():
    """Same password hashed twice must produce different hashes (bcrypt salting)."""
    hash1 = hash_password("samepassword")
    hash2 = hash_password("samepassword")
    assert hash1 != hash2


def test_both_hashes_verify():
    """Both unique hashes of same password must still verify correctly."""
    hash1 = hash_password("samepassword")
    hash2 = hash_password("samepassword")
    assert verify_password("samepassword", hash1) is True
    assert verify_password("samepassword", hash2) is True


def test_short_password():
    """Short passwords must still hash and verify correctly."""
    hashed = hash_password("ab")
    assert verify_password("ab", hashed) is True
    assert verify_password("ac", hashed) is False


def test_special_characters_in_password():
    """Passwords with special characters must work correctly."""
    password = "P@ssw0rd!#€£$%"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True
    assert verify_password("P@ssw0rd!#€£$%_wrong", hashed) is False


def test_german_characters_in_password():
    """German umlauts in passwords must work correctly."""
    password = "GutesPasswort123ü"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True


# ================================================================
# SESSION TESTS
# ================================================================

def test_session_starts_empty():
    """Session must start with no user logged in."""
    Session.logout()  # ensure clean state
    assert Session.is_logged_in() is False
    assert Session.get_user() is None


def test_session_login():
    """After login, session must reflect logged-in state."""
    Session.logout()
    test_user = {
        "id": 1,
        "username": "dr.mueller",
        "full_name": "Dr. Anna Mueller",
        "role": "doctor",
        "is_active": 1
    }
    Session.login(test_user)
    assert Session.is_logged_in() is True
    assert Session.get_username() == "dr.mueller"
    assert Session.get_full_name() == "Dr. Anna Mueller"


def test_session_logout():
    """After logout, session must be empty."""
    test_user = {"id": 1, "username": "test", "full_name": "Test", "role": "doctor"}
    Session.login(test_user)
    Session.logout()
    assert Session.is_logged_in() is False
    assert Session.get_user() is None
    assert Session.get_username() == ""


def test_session_doctor_is_not_admin():
    """Doctor role must not have admin privileges."""
    Session.login({"id": 2, "username": "doctor1", "full_name": "Dr. Smith", "role": "doctor"})
    assert Session.is_admin() is False
    Session.logout()


def test_session_admin_is_admin():
    """Admin role must have admin privileges."""
    Session.login({"id": 1, "username": "admin", "full_name": "Admin", "role": "admin"})
    assert Session.is_admin() is True
    Session.logout()


def test_session_get_username_when_not_logged_in():
    """get_username must return empty string when not logged in."""
    Session.logout()
    assert Session.get_username() == ""


def test_session_get_full_name_when_not_logged_in():
    """get_full_name must return empty string when not logged in."""
    Session.logout()
    assert Session.get_full_name() == ""


def test_session_is_not_admin_when_not_logged_in():
    """is_admin must return False when not logged in."""
    Session.logout()
    assert Session.is_admin() is False
