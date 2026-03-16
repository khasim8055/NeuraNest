# app/tests/test_audit.py
# ================================================================
# NeuraCare — Tests for audit.py
# Run with: pytest app/tests/test_audit.py -v
# ================================================================

import pytest
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.audit import (
    log, log_login, log_logout, log_patient_create, log_patient_view,
    log_patient_edit, log_patient_delete, log_patient_restore,
    log_summary_generate, log_pdf_export, log_csv_export,
    get_recent_logs, get_logs_for_patient, get_logs_for_user,
    get_audit_summary, export_audit_to_csv, VALID_ACTIONS,
)

@pytest.fixture
def test_db(tmp_path, monkeypatch):
    import app.core.audit as audit_module
    db_path = tmp_path / "test_neuranest.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, user_id INTEGER,
        username TEXT NOT NULL, action TEXT NOT NULL,
        patient_id INTEGER, patient_name TEXT, detail TEXT)""")
    conn.commit()
    conn.close()
    monkeypatch.setattr(audit_module, "DB_FILE", db_path)
    return db_path

def test_valid_actions_not_empty():
    assert len(VALID_ACTIONS) > 0

def test_valid_actions_contains_required():
    required = {"login","logout","patient_create","patient_view",
                "patient_edit","patient_delete","patient_restore",
                "summary_generate","pdf_export","csv_export"}
    assert required == VALID_ACTIONS

def test_log_valid_action(test_db):
    assert log("login", "dr.mueller", user_id=1) is True

def test_log_invalid_action(test_db):
    assert log("invalid_action", "dr.mueller") is False

def test_log_empty_username(test_db):
    assert log("login", "") is False

def test_log_writes_to_database(test_db):
    log("login", "dr.mueller", user_id=1)
    conn = sqlite3.connect(str(test_db))
    row = conn.execute("SELECT * FROM audit_log").fetchone()
    conn.close()
    assert row is not None
    assert row[3] == "dr.mueller"
    assert row[4] == "login"

def test_log_timestamp_format(test_db):
    log("login", "dr.mueller")
    conn = sqlite3.connect(str(test_db))
    row = conn.execute("SELECT timestamp FROM audit_log").fetchone()
    conn.close()
    datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")

def test_log_with_patient_info(test_db):
    log("patient_view", "dr.mueller", patient_id=42, patient_name="Maria Hoffmann")
    conn = sqlite3.connect(str(test_db))
    row = conn.execute("SELECT patient_id, patient_name FROM audit_log").fetchone()
    conn.close()
    assert row[0] == 42
    assert row[1] == "Maria Hoffmann"

def test_log_never_crashes_without_db(tmp_path, monkeypatch):
    import app.core.audit as audit_module
    monkeypatch.setattr(audit_module, "DB_FILE", tmp_path / "nonexistent.db")
    assert log("login", "dr.mueller") is False

def test_multiple_logs(test_db):
    log("login", "dr.mueller", user_id=1)
    log("patient_view", "dr.mueller", patient_id=1, patient_name="Test")
    log("patient_create", "dr.mueller", patient_id=2, patient_name="New")
    log("logout", "dr.mueller", user_id=1)
    conn = sqlite3.connect(str(test_db))
    count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    conn.close()
    assert count == 4

def test_log_login(test_db):
    assert log_login("dr.mueller", user_id=1) is True

def test_log_logout(test_db):
    assert log_logout("dr.mueller", user_id=1) is True

def test_log_patient_create(test_db):
    assert log_patient_create("dr.mueller", 5, "Klaus Berger", user_id=1) is True
    conn = sqlite3.connect(str(test_db))
    row = conn.execute("SELECT action, patient_name FROM audit_log").fetchone()
    conn.close()
    assert row[0] == "patient_create"
    assert row[1] == "Klaus Berger"

def test_log_patient_edit_with_fields(test_db):
    assert log_patient_edit("dr.mueller", 5, "Klaus Berger",
                            changed_fields=["diagnosis","notes"], user_id=1) is True
    conn = sqlite3.connect(str(test_db))
    row = conn.execute("SELECT detail FROM audit_log").fetchone()
    conn.close()
    assert "diagnosis" in row[0]
    assert "notes" in row[0]

def test_log_patient_delete(test_db):
    assert log_patient_delete("admin", 5, "Klaus Berger", user_id=2) is True

def test_log_patient_restore(test_db):
    assert log_patient_restore("admin", 5, "Klaus Berger", user_id=2) is True

def test_log_summary_generate_with_level(test_db):
    assert log_summary_generate("dr.mueller", 5, "Klaus Berger",
                                autonomy_level="AL1", user_id=1) is True
    conn = sqlite3.connect(str(test_db))
    row = conn.execute("SELECT detail FROM audit_log").fetchone()
    conn.close()
    assert "AL1" in row[0]

def test_log_pdf_export(test_db):
    assert log_pdf_export("dr.mueller", 5, "Klaus Berger", user_id=1) is True

def test_log_csv_export_anonymised(test_db):
    assert log_csv_export("dr.mueller", record_count=15,
                          anonymised=True, user_id=1) is True
    conn = sqlite3.connect(str(test_db))
    row = conn.execute("SELECT detail FROM audit_log").fetchone()
    conn.close()
    assert "Anonymised" in row[0]
    assert "15" in row[0]

def test_get_recent_logs_empty(test_db):
    assert get_recent_logs() == []

def test_get_recent_logs_returns_entries(test_db):
    log_login("dr.mueller", user_id=1)
    log_patient_view("dr.mueller", 1, "Maria Hoffmann", user_id=1)
    assert len(get_recent_logs()) == 2

def test_get_recent_logs_newest_first(test_db):
    """get_recent_logs must return newest entries first — check by id."""
    log_login("dr.mueller")
    log_logout("dr.mueller")
    result = get_recent_logs()
    # Higher id = inserted later = should appear first (ORDER BY timestamp DESC)
    assert result[0]["id"] > result[1]["id"]

def test_get_recent_logs_limit(test_db):
    for i in range(10):
        log_patient_view("dr.mueller", i, f"Patient {i}")
    assert len(get_recent_logs(limit=5)) == 5

def test_get_logs_for_patient(test_db):
    log_patient_view("dr.mueller", patient_id=1, patient_name="Maria")
    log_patient_view("dr.mueller", patient_id=2, patient_name="Klaus")
    log_patient_edit("dr.mueller", patient_id=1, patient_name="Maria")
    result = get_logs_for_patient(1)
    assert len(result) == 2
    for entry in result:
        assert entry["action"] in ("patient_view", "patient_edit")

def test_get_logs_for_user(test_db):
    log_login("dr.mueller")
    log_login("dr.schmidt")
    log_patient_view("dr.mueller", 1, "Maria")
    result = get_logs_for_user("dr.mueller")
    assert len(result) == 2
    for entry in result:
        assert entry["action"] in ("login", "patient_view")

def test_get_audit_summary(test_db):
    log_login("dr.mueller")
    log_login("dr.schmidt")
    log_patient_view("dr.mueller", 1, "Maria")
    summary = get_audit_summary()
    assert summary.get("login") == 2
    assert summary.get("patient_view") == 1

def test_export_audit_to_csv(test_db):
    log_login("dr.mueller")
    log_patient_view("dr.mueller", 1, "Maria Hoffmann")
    csv = export_audit_to_csv()
    assert len(csv) > 0
    lines = csv.strip().split("\n")
    assert "timestamp" in lines[0]
    assert "username" in lines[0]
    assert "action" in lines[0]
    assert len(lines) == 3

def test_export_audit_empty_db(test_db):
    csv = export_audit_to_csv()
    lines = csv.strip().split("\n")
    assert len(lines) == 1
    assert "timestamp" in lines[0]
