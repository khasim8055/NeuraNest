# app/core/csv_import.py
# ================================================================
# NeuraCare — CSV Patient Import
# ================================================================
# Imports patients from a NeuraCare-format CSV template.
# Template: NeuraCare_Import_Template.csv
# ================================================================

import csv
import sqlite3
import os
from pathlib import Path
from datetime import datetime


def get_db_path() -> Path:
    base = Path(os.environ.get("NEURACARE_BASE_DIR", ""))
    if not base or not base.exists():
        base = Path(__file__).parent.parent.parent
    return base / "app" / "data" / "neuranest.db"


def validate_date(date_str: str) -> str:
    """Validate and normalise date to YYYY-MM-DD."""
    if not date_str or not date_str.strip():
        return ""
    for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def import_patients_from_csv(
    file_path: str,
    created_by: str = "import",
) -> tuple[bool, int, int, list[str]]:
    """
    Import patients from a NeuraCare CSV template.

    Returns:
        (success, imported_count, skipped_count, error_list)
    """
    errors = []
    imported = 0
    skipped = 0

    try:
        db = get_db_path()
        conn = sqlite3.connect(str(db))

        with open(file_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            # Validate headers
            required = ["name", "age", "diagnosis"]
            headers = [h.strip().lower() for h in (reader.fieldnames or [])]
            missing = [r for r in required if r not in headers]
            if missing:
                return False, 0, 0, [
                    f"Missing required columns: {', '.join(missing)}. "
                    f"Please use the NeuraCare CSV template."
                ]

            for row_num, row in enumerate(reader, start=2):
                # Normalise keys
                row = {k.strip().lower(): v.strip() for k, v in row.items() if k}

                name = row.get("name", "").strip()
                if not name:
                    skipped += 1
                    errors.append(f"Row {row_num}: skipped — name is empty")
                    continue

                # Check duplicate
                existing = conn.execute(
                    "SELECT id FROM patients WHERE name=? AND is_deleted=0",
                    (name,)
                ).fetchone()
                if existing:
                    skipped += 1
                    errors.append(f"Row {row_num}: skipped — {name} already exists")
                    continue

                # Parse fields
                age = 0
                try:
                    age = int(row.get("age", 0) or 0)
                except ValueError:
                    pass

                adm = validate_date(row.get("admission_date", ""))
                dis = validate_date(row.get("discharge_date", ""))
                fol = validate_date(row.get("followup_date", ""))

                conn.execute("""
                    INSERT INTO patients
                    (name, age, date_of_birth, diagnosis, icd10_code,
                     admission_date, discharge_date, notes, medication,
                     physician_name, followup_date, created_by, is_deleted)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)
                """, (
                    name,
                    age,
                    validate_date(row.get("date_of_birth", "")),
                    row.get("diagnosis", ""),
                    row.get("icd10_code", ""),
                    adm,
                    dis,
                    row.get("notes", ""),
                    row.get("medication", ""),
                    row.get("physician_name", ""),
                    fol,
                    created_by,
                ))
                imported += 1

        conn.commit()
        conn.close()
        return True, imported, skipped, errors

    except Exception as e:
        return False, 0, 0, [f"Import failed: {str(e)}"]


def get_template_path() -> Path:
    """Return path to the CSV template file."""
    base = Path(os.environ.get("NEURACARE_BASE_DIR", ""))
    if not base or not base.exists():
        base = Path(__file__).parent.parent.parent
    return base / "NeuraCare_Import_Template.csv"


def create_template(path: str = None) -> str:
    """Create a blank CSV template and return its path."""
    if path is None:
        path = str(Path.home() / "Desktop" / "NeuraCare_Import_Template.csv")

    headers = [
        "name", "age", "date_of_birth", "diagnosis", "icd10_code",
        "admission_date", "discharge_date", "physician_name",
        "notes", "medication", "followup_date"
    ]

    example = [
        "Hans Mueller", "72", "1952-03-14", "Herzinsuffizienz", "I50.0",
        "2026-01-10", "2026-01-24", "Dr. Schmidt",
        "Patient aufgenommen mit Dyspnoe. Unter Therapie Besserung.",
        "Furosemid 40mg 1-0-0, Bisoprolol 5mg 1-0-0",
        "2026-02-10"
    ]

    import csv as _csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(headers)
        writer.writerow(example)

    return path
