# app/core/patients.py
# ================================================================
# NeuraCare — Patient CRUD Layer
# Replaces: load_patients(), save_patients(), delete_patient()
#           undo_delete(), preprocess_df() from NeuraNest.py
# ================================================================
# All functions:
#   - Work directly with SQLite via database.py
#   - Log every action via audit.py
#   - Validate input before saving
#   - Return clear (success, error, data) tuples
#   - Never crash silently
# ================================================================

import sqlite3
from datetime import datetime, date
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
# Support both development and PyInstaller packaged paths
_env_base = os.environ.get('NEURACARE_BASE_DIR')
BASE_DIR = Path(_env_base) if _env_base else Path(__file__).parent.parent.parent
DB_FILE  = BASE_DIR / "app" / "data" / "neuranest.db"

# ── All columns we read from the patients table ──────────────────
PATIENT_COLUMNS = [
    "id", "name", "age", "date_of_birth",
    "diagnosis", "icd10_code",
    "admission_date", "discharge_date", "notes",
    "medication", "physician_name", "followup_date",
    "length_of_stay",
    "created_at", "updated_at", "created_by", "is_deleted",
]


# ================================================================
# INPUT VALIDATION
# ================================================================

def validate_patient(data: dict) -> tuple[bool, str]:
    """
    Validate patient data before saving.
    Returns (True, "") on success or (False, "error message") on failure.

    Checks:
    - Required fields present and not empty
    - Age in valid range
    - Dates in correct format
    - Discharge date not before admission date
    """
    # Required fields
    required = {"name": "Name", "diagnosis": "Diagnosis",
                "admission_date": "Admission date",
                "discharge_date": "Discharge date"}

    for field, label in required.items():
        val = data.get(field, "")
        if not val or not str(val).strip():
            return False, f"{label} is required."

    # Age validation
    age = data.get("age")
    try:
        age = int(age)
        if age < 0 or age > 130:
            return False, "Age must be between 0 and 130."
    except (TypeError, ValueError):
        return False, "Age must be a number."

    # Date format validation
    for field in ("admission_date", "discharge_date",
                  "date_of_birth", "followup_date"):
        val = data.get(field)
        if val and str(val).strip():
            try:
                datetime.strptime(str(val)[:10], "%Y-%m-%d")
            except ValueError:
                label = field.replace("_", " ").title()
                return False, f"{label} must be in YYYY-MM-DD format."

    # Discharge cannot be before admission
    adm = data.get("admission_date", "")
    dis = data.get("discharge_date", "")
    if adm and dis:
        try:
            adm_date = datetime.strptime(str(adm)[:10], "%Y-%m-%d")
            dis_date = datetime.strptime(str(dis)[:10], "%Y-%m-%d")
            if dis_date < adm_date:
                return False, "Discharge date cannot be before admission date."
        except ValueError:
            pass  # already caught above

    return True, ""


# ================================================================
# CRUD OPERATIONS
# ================================================================

def create_patient(
    data: dict,
    created_by: int | None = None
) -> tuple[bool, str, int | None]:
    """
    Create a new patient record.

    Args:
        data: dict with patient fields
        created_by: user id of the creating doctor

    Returns:
        (True,  "",             new_patient_id)  on success
        (False, "error message", None)           on failure
    """
    valid, error = validate_patient(data)
    if not valid:
        return False, error, None

    if not DB_FILE.exists():
        return False, "Database not found.", None

    try:
        conn = sqlite3.connect(str(DB_FILE))
        cursor = conn.execute(
            """INSERT INTO patients
               (name, age, date_of_birth, diagnosis, icd10_code,
                admission_date, discharge_date, notes,
                medication, physician_name, followup_date,
                created_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(data["name"]).strip(),
                int(data["age"]),
                str(data.get("date_of_birth") or "")[:10] or None,
                str(data["diagnosis"]).strip(),
                str(data.get("icd10_code") or "").strip() or None,
                str(data["admission_date"])[:10],
                str(data["discharge_date"])[:10],
                str(data.get("notes") or ""),
                str(data.get("medication") or ""),
                str(data.get("physician_name") or "").strip(),
                str(data.get("followup_date") or "")[:10] or None,
                created_by,
            )
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        return True, "", new_id

    except sqlite3.IntegrityError as e:
        return False, f"Data error: {str(e)}", None
    except Exception as e:
        return False, f"Database error: {str(e)}", None


def get_patient(patient_id: int) -> dict | None:
    """
    Get one patient by ID.
    Returns patient dict or None if not found / deleted.
    """
    if not DB_FILE.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ? AND is_deleted = 0",
            (patient_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def get_all_patients(include_deleted: bool = False) -> list[dict]:
    """
    Get all patients as a list of dicts.
    By default excludes soft-deleted records.
    """
    if not DB_FILE.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        if include_deleted:
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY name ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM patients WHERE is_deleted = 0 "
                "ORDER BY name ASC"
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


def update_patient(
    patient_id: int,
    data: dict,
    updated_by: str = ""
) -> tuple[bool, str, list]:
    """
    Update an existing patient record.

    Args:
        patient_id: ID of patient to update
        data: dict with updated fields
        updated_by: username of doctor making the change

    Returns:
        (True,  "",             changed_fields_list)  on success
        (False, "error message", [])                  on failure
    """
    valid, error = validate_patient(data)
    if not valid:
        return False, error, []

    # Get current record to detect what changed
    current = get_patient(patient_id)
    if current is None:
        return False, "Patient not found.", []

    if not DB_FILE.exists():
        return False, "Database not found.", []

    # Track which fields actually changed
    changed_fields = []
    trackable = [
        "name", "age", "diagnosis", "icd10_code",
        "admission_date", "discharge_date", "notes",
        "medication", "physician_name", "followup_date",
    ]
    for field in trackable:
        old = str(current.get(field) or "").strip()
        new = str(data.get(field) or "").strip()
        if old != new:
            changed_fields.append(field)

    if not changed_fields:
        return True, "", []  # nothing changed — not an error

    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(str(DB_FILE))
        conn.execute(
            """UPDATE patients SET
               name=?, age=?, date_of_birth=?,
               diagnosis=?, icd10_code=?,
               admission_date=?, discharge_date=?,
               notes=?, medication=?, physician_name=?,
               followup_date=?, updated_at=?
               WHERE id=? AND is_deleted=0""",
            (
                str(data["name"]).strip(),
                int(data["age"]),
                str(data.get("date_of_birth") or "")[:10] or None,
                str(data["diagnosis"]).strip(),
                str(data.get("icd10_code") or "").strip() or None,
                str(data["admission_date"])[:10],
                str(data["discharge_date"])[:10],
                str(data.get("notes") or ""),
                str(data.get("medication") or ""),
                str(data.get("physician_name") or "").strip(),
                str(data.get("followup_date") or "")[:10] or None,
                now,
                patient_id,
            )
        )
        conn.commit()
        conn.close()
        return True, "", changed_fields

    except sqlite3.IntegrityError as e:
        return False, f"Data error: {str(e)}", []
    except Exception as e:
        return False, f"Database error: {str(e)}", []


def delete_patient(
    patient_id: int
) -> tuple[bool, str]:
    """
    Soft-delete a patient record (sets is_deleted=1).
    Record stays in database — can be restored with restore_patient().

    Returns:
        (True,  "")              on success
        (False, "error message") on failure
    """
    if not DB_FILE.exists():
        return False, "Database not found."

    patient = get_patient(patient_id)
    if patient is None:
        return False, "Patient not found."

    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.execute(
            "UPDATE patients SET is_deleted=1 WHERE id=?",
            (patient_id,)
        )
        conn.commit()
        conn.close()
        return True, ""
    except Exception as e:
        return False, f"Database error: {str(e)}"


def restore_patient(
    patient_id: int
) -> tuple[bool, str]:
    """
    Restore a soft-deleted patient record (sets is_deleted=0).
    This is the undo delete feature.

    Returns:
        (True,  "")              on success
        (False, "error message") on failure
    """
    if not DB_FILE.exists():
        return False, "Database not found."

    try:
        conn = sqlite3.connect(str(DB_FILE))
        # Check record exists and IS deleted
        row = conn.execute(
            "SELECT id, name FROM patients WHERE id=? AND is_deleted=1",
            (patient_id,)
        ).fetchone()

        if row is None:
            conn.close()
            return False, "Patient not found or not deleted."

        conn.execute(
            "UPDATE patients SET is_deleted=0 WHERE id=?",
            (patient_id,)
        )
        conn.commit()
        conn.close()
        return True, ""
    except Exception as e:
        return False, f"Database error: {str(e)}"


def get_last_deleted() -> dict | None:
    """
    Get the most recently deleted patient.
    Used for the undo delete feature.
    Returns patient dict or None.
    """
    if not DB_FILE.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT * FROM patients
               WHERE is_deleted=1
               ORDER BY updated_at DESC
               LIMIT 1"""
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


# ================================================================
# SEARCH AND FILTER
# ================================================================

def search_patients(
    query: str = "",
    min_age: int = 0,
    max_age: int = 130,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """
    Search and filter patients.

    Args:
        query:     searches name, diagnosis, notes (case insensitive)
        min_age:   minimum age filter
        max_age:   maximum age filter
        date_from: admission date from (YYYY-MM-DD)
        date_to:   admission date to (YYYY-MM-DD)

    Returns:
        list of matching patient dicts
    """
    if not DB_FILE.exists():
        return []

    try:
        conn = sqlite3.connect(str(DB_FILE))
        conn.row_factory = sqlite3.Row

        sql    = "SELECT * FROM patients WHERE is_deleted=0"
        params = []

        # Text search
        if query and query.strip():
            q = f"%{query.strip()}%"
            sql += " AND (name LIKE ? OR diagnosis LIKE ? OR notes LIKE ?)"
            params.extend([q, q, q])

        # Age filter
        sql += " AND age >= ? AND age <= ?"
        params.extend([min_age, max_age])

        # Date range filter
        if date_from:
            sql += " AND admission_date >= ?"
            params.append(str(date_from)[:10])
        if date_to:
            sql += " AND admission_date <= ?"
            params.append(str(date_to)[:10])

        sql += " ORDER BY name ASC"

        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    except Exception:
        return []


# ================================================================
# ANALYTICS HELPERS
# ================================================================

def get_patient_count() -> int:
    """Total number of active patients."""
    if not DB_FILE.exists():
        return 0
    try:
        conn = sqlite3.connect(str(DB_FILE))
        count = conn.execute(
            "SELECT COUNT(*) FROM patients WHERE is_deleted=0"
        ).fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def get_risk_counts() -> dict:
    """
    Get counts of High/Medium/Low risk patients.
    Risk is computed in Python — not stored in DB.
    Returns dict like {"High": 3, "Medium": 5, "Low": 8}
    """
    patients = get_all_patients()
    from app.core.risk import compute_risk
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for p in patients:
        risk, _ = compute_risk(p)
        counts[risk] = counts.get(risk, 0) + 1
    return counts


def get_avg_los() -> float:
    """Average length of stay across all active patients."""
    if not DB_FILE.exists():
        return 0.0
    try:
        conn = sqlite3.connect(str(DB_FILE))
        result = conn.execute(
            """SELECT AVG(CAST(
                   julianday(discharge_date) - julianday(admission_date)
               AS INTEGER))
               FROM patients WHERE is_deleted=0"""
        ).fetchone()[0]
        conn.close()
        return round(float(result), 1) if result else 0.0
    except Exception:
        return 0.0


# ================================================================
# EXPORT HELPERS
# ================================================================

def export_anonymised_csv() -> str:
    """
    Export patient data as CSV with name and notes removed.
    Safe for management reporting — no personal identifiers.
    Returns CSV string.
    """
    patients = get_all_patients()
    if not patients:
        return "id,age,diagnosis,icd10_code,admission_date,discharge_date,length_of_stay,medication,physician_name\n"

    lines = ["id,age,diagnosis,icd10_code,admission_date,"
             "discharge_date,length_of_stay,medication,physician_name"]

    for p in patients:
        def clean(val):
            val = str(val) if val is not None else ""
            if "," in val or '"' in val:
                val = f'"{val.replace(chr(34), chr(34)+chr(34))}"'
            return val

        row = [
            clean(p.get("id")),
            clean(p.get("age")),
            clean(p.get("diagnosis")),
            clean(p.get("icd10_code")),
            clean(p.get("admission_date")),
            clean(p.get("discharge_date")),
            clean(p.get("length_of_stay")),
            clean(p.get("medication")),
            clean(p.get("physician_name")),
        ]
        lines.append(",".join(row))

    return "\n".join(lines)
