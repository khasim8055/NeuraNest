# app/core/audit.py
# ================================================================
# NeuraCare — Audit Logger
# GDPR Article 30 compliance — every action on patient data logged
# ================================================================
# Rules:
#   - Every entry is permanent — no deletes, no edits
#   - Logs who did what, to which patient, and when
#   - Works silently — never crashes the main app if logging fails
#   - Exportable as PDF for GDPR compliance reports
# ================================================================

import sqlite3
from datetime import datetime
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
# Support both development and PyInstaller packaged paths
_env_base = os.environ.get('NEURACARE_BASE_DIR')
BASE_DIR = Path(_env_base) if _env_base else Path(__file__).parent.parent.parent
DB_FILE  = BASE_DIR / "app" / "data" / "neuranest.db"

# ── Valid actions — must match schema.sql CHECK constraint ────────
VALID_ACTIONS = {
    "login",
    "logout",
    "patient_create",
    "patient_view",
    "patient_edit",
    "patient_delete",
    "patient_restore",
    "summary_generate",
    "pdf_export",
    "csv_export",
}


# ================================================================
# CORE LOG FUNCTION
# ================================================================

def log(
    action: str,
    username: str,
    user_id: int | None = None,
    patient_id: int | None = None,
    patient_name: str | None = None,
    detail: str | None = None,
) -> bool:
    """
    Write one entry to the audit_log table.

    This function NEVER raises an exception — if logging fails,
    it returns False silently. The main app must never crash
    because of a logging failure.

    Args:
        action:       One of VALID_ACTIONS (see above)
        username:     Username of the person doing the action
        user_id:      ID from users table (optional)
        patient_id:   ID from patients table (None for login/logout)
        patient_name: Patient name snapshot (None for login/logout)
        detail:       Any extra context (optional)

    Returns:
        True if logged successfully, False if failed
    """
    if action not in VALID_ACTIONS:
        return False

    if not username or not username.strip():
        return False

    if not DB_FILE.exists():
        return False

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(str(DB_FILE))
        conn.execute(
            """INSERT INTO audit_log
               (timestamp, user_id, username, action,
                patient_id, patient_name, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, user_id, username.strip(), action,
             patient_id, patient_name, detail)
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# ================================================================
# CONVENIENCE FUNCTIONS
# one function per action — clean call sites in the main app
# ================================================================

def log_login(username: str, user_id: int | None = None) -> bool:
    """Call when a user successfully logs in."""
    return log("login", username, user_id=user_id,
               detail="User logged in")


def log_logout(username: str, user_id: int | None = None) -> bool:
    """Call when a user logs out."""
    return log("logout", username, user_id=user_id,
               detail="User logged out")


def log_patient_create(
    username: str,
    patient_id: int,
    patient_name: str,
    user_id: int | None = None
) -> bool:
    """Call when a new patient record is created."""
    return log("patient_create", username, user_id=user_id,
               patient_id=patient_id, patient_name=patient_name,
               detail="New patient record created")


def log_patient_view(
    username: str,
    patient_id: int,
    patient_name: str,
    user_id: int | None = None
) -> bool:
    """Call when a patient record is opened/viewed."""
    return log("patient_view", username, user_id=user_id,
               patient_id=patient_id, patient_name=patient_name)


def log_patient_edit(
    username: str,
    patient_id: int,
    patient_name: str,
    changed_fields: list | None = None,
    user_id: int | None = None
) -> bool:
    """Call when a patient record is edited."""
    detail = None
    if changed_fields:
        detail = f"Fields changed: {', '.join(changed_fields)}"
    return log("patient_edit", username, user_id=user_id,
               patient_id=patient_id, patient_name=patient_name,
               detail=detail)


def log_patient_delete(
    username: str,
    patient_id: int,
    patient_name: str,
    user_id: int | None = None
) -> bool:
    """Call when a patient record is deleted."""
    return log("patient_delete", username, user_id=user_id,
               patient_id=patient_id, patient_name=patient_name,
               detail="Patient record deleted")


def log_patient_restore(
    username: str,
    patient_id: int,
    patient_name: str,
    user_id: int | None = None
) -> bool:
    """Call when a deleted patient record is restored."""
    return log("patient_restore", username, user_id=user_id,
               patient_id=patient_id, patient_name=patient_name,
               detail="Deleted patient record restored")


def log_summary_generate(
    username: str,
    patient_id: int,
    patient_name: str,
    autonomy_level: str | None = None,
    user_id: int | None = None
) -> bool:
    """Call when a discharge summary is generated."""
    detail = f"Level: {autonomy_level}" if autonomy_level else None
    return log("summary_generate", username, user_id=user_id,
               patient_id=patient_id, patient_name=patient_name,
               detail=detail)


def log_pdf_export(
    username: str,
    patient_id: int,
    patient_name: str,
    user_id: int | None = None
) -> bool:
    """Call when a PDF is downloaded."""
    return log("pdf_export", username, user_id=user_id,
               patient_id=patient_id, patient_name=patient_name,
               detail="PDF discharge summary exported")


def log_csv_export(
    username: str,
    record_count: int | None = None,
    anonymised: bool = False,
    user_id: int | None = None
) -> bool:
    """Call when CSV data is exported."""
    detail = f"{'Anonymised' if anonymised else 'Full'} export"
    if record_count is not None:
        detail += f" — {record_count} records"
    return log("csv_export", username, user_id=user_id, detail=detail)


# ================================================================
# QUERY FUNCTIONS — for audit report display
# ================================================================

def get_recent_logs(limit: int = 50) -> list[dict]:
    """
    Get most recent audit entries.
    Used to display audit log in the UI.
    Returns list of dicts, newest first.
    """
    if not DB_FILE.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, timestamp, username, action,
                      patient_name, detail
               FROM audit_log
               ORDER BY id DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


def get_logs_for_patient(patient_id: int) -> list[dict]:
    """
    Get all audit entries for one specific patient.
    Used in patient detail view to show history.
    """
    if not DB_FILE.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, timestamp, username, action, detail
               FROM audit_log
               WHERE patient_id = ?
               ORDER BY id DESC""",
            (patient_id,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


def get_logs_for_user(username: str, limit: int = 100) -> list[dict]:
    """
    Get all audit entries for one specific user.
    Admin use — see what a doctor has been doing.
    """
    if not DB_FILE.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, timestamp, action,
                      patient_name, detail
               FROM audit_log
               WHERE username = ?
               ORDER BY id DESC
               LIMIT ?""",
            (username.strip(), limit)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


def get_audit_summary() -> dict:
    """
    Get summary statistics for the audit log.
    Used in admin dashboard.
    Returns counts per action type.
    """
    if not DB_FILE.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB_FILE))
        rows = conn.execute(
            """SELECT action, COUNT(*) as count
               FROM audit_log
               GROUP BY action
               ORDER BY count DESC"""
        ).fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


def export_audit_to_csv() -> str:
    """
    Export entire audit log as CSV string.
    Used for GDPR compliance reports.
    Returns CSV string or empty string on failure.
    """
    if not DB_FILE.exists():
        return ""
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp, username, action,
                      patient_name, detail
               FROM audit_log
               ORDER BY timestamp ASC"""
        ).fetchall()
        conn.close()

        lines = ["timestamp,username,action,patient_name,detail"]
        for row in rows:
            # Escape commas in fields
            fields = []
            for val in row:
                val = str(val) if val is not None else ""
                if "," in val or '"' in val:
                    val = f'"{val.replace(chr(34), chr(34)+chr(34))}"'
                fields.append(val)
            lines.append(",".join(fields))

        return "\n".join(lines)
    except Exception:
        return ""
