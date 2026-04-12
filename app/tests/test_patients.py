# app/tests/test_patients.py
# ================================================================
# NeuraCare — Tests for patients.py and risk.py
# Run with: pytest app/tests/test_patients.py -v
# ================================================================

import pytest
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import app.core.patients as patients_module
import app.core.risk     as risk_module

from app.core.patients import (
    validate_patient,
    create_patient,
    get_patient,
    get_all_patients,
    update_patient,
    delete_patient,
    restore_patient,
    get_last_deleted,
    search_patients,
    get_patient_count,
    get_avg_los,
    export_anonymised_csv,
)
from app.core.risk import compute_risk, get_risk_explanation


# ================================================================
# FIXTURES
# ================================================================

@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """
    Creates a full test database with patients table.
    Patches DB_FILE in patients and risk modules.
    """
    db_path = tmp_path / "test_neuranest.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE patients (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT NOT NULL,
            age            INTEGER NOT NULL,
            date_of_birth  TEXT,
            diagnosis      TEXT NOT NULL,
            icd10_code     TEXT,
            admission_date TEXT NOT NULL,
            discharge_date TEXT NOT NULL,
            notes          TEXT DEFAULT '',
            medication     TEXT DEFAULT '',
            physician_name TEXT DEFAULT '',
            followup_date  TEXT,
            length_of_stay INTEGER GENERATED ALWAYS AS (
                CAST((julianday(discharge_date) -
                      julianday(admission_date)) AS INTEGER)
            ) VIRTUAL,
            created_at     TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
            created_by     INTEGER,
            is_deleted     INTEGER NOT NULL DEFAULT 0,
            CHECK(discharge_date >= admission_date)
        )
    """)
    conn.commit()
    conn.close()

    monkeypatch.setattr(patients_module, "DB_FILE", db_path)
    monkeypatch.setattr(risk_module,     "CHRONIC_KEYWORDS", risk_module.CHRONIC_KEYWORDS)
    return db_path


@pytest.fixture
def sample_patient():
    """Valid patient data for reuse in tests."""
    return {
        "name":           "Maria Hoffmann",
        "age":            74,
        "diagnosis":      "Chronic Heart Failure",
        "admission_date": "2025-01-08",
        "discharge_date": "2025-01-22",
        "notes":          "Patient stable on discharge.",
        "medication":     "Furosemide 40mg",
        "physician_name": "Dr. Mueller",
        "icd10_code":     "I50.0",
        "followup_date":  "2025-02-05",
    }


# ================================================================
# VALIDATION TESTS
# ================================================================

def test_validate_valid_patient(sample_patient):
    """Valid patient data must pass validation."""
    ok, err = validate_patient(sample_patient)
    assert ok is True
    assert err == ""


def test_validate_missing_name(sample_patient):
    """Missing name must fail validation."""
    sample_patient["name"] = ""
    ok, err = validate_patient(sample_patient)
    assert ok is False
    assert "Name" in err


def test_validate_missing_diagnosis(sample_patient):
    """Missing diagnosis must fail validation."""
    sample_patient["diagnosis"] = ""
    ok, err = validate_patient(sample_patient)
    assert ok is False
    assert "Diagnosis" in err


def test_validate_missing_admission_date(sample_patient):
    """Missing admission date must fail validation."""
    sample_patient["admission_date"] = ""
    ok, err = validate_patient(sample_patient)
    assert ok is False


def test_validate_missing_discharge_date(sample_patient):
    """Missing discharge date must fail validation."""
    sample_patient["discharge_date"] = ""
    ok, err = validate_patient(sample_patient)
    assert ok is False


def test_validate_age_negative(sample_patient):
    """Negative age must fail validation."""
    sample_patient["age"] = -1
    ok, err = validate_patient(sample_patient)
    assert ok is False
    assert "Age" in err


def test_validate_age_too_high(sample_patient):
    """Age over 130 must fail validation."""
    sample_patient["age"] = 131
    ok, err = validate_patient(sample_patient)
    assert ok is False


def test_validate_age_not_number(sample_patient):
    """Non-numeric age must fail validation."""
    sample_patient["age"] = "not_a_number"
    ok, err = validate_patient(sample_patient)
    assert ok is False


def test_validate_discharge_before_admission(sample_patient):
    """Discharge before admission must fail validation."""
    sample_patient["admission_date"] = "2025-01-22"
    sample_patient["discharge_date"] = "2025-01-08"
    ok, err = validate_patient(sample_patient)
    assert ok is False
    assert "Discharge" in err or "discharge" in err


def test_validate_same_day_admission_discharge(sample_patient):
    """Same day admission and discharge must be valid."""
    sample_patient["admission_date"] = "2025-01-10"
    sample_patient["discharge_date"] = "2025-01-10"
    ok, err = validate_patient(sample_patient)
    assert ok is True


def test_validate_invalid_date_format(sample_patient):
    """Wrong date format must fail validation."""
    sample_patient["admission_date"] = "10/01/2025"
    ok, err = validate_patient(sample_patient)
    assert ok is False


# ================================================================
# CREATE PATIENT TESTS
# ================================================================

def test_create_patient_success(test_db, sample_patient):
    """Valid patient must be created successfully."""
    ok, err, patient_id = create_patient(sample_patient)
    assert ok is True
    assert err == ""
    assert patient_id is not None
    assert patient_id > 0


def test_create_patient_returns_id(test_db, sample_patient):
    """create_patient must return the new patient's ID."""
    _, _, id1 = create_patient(sample_patient)
    sample_patient["name"] = "Klaus Berger"
    _, _, id2 = create_patient(sample_patient)
    assert id2 > id1


def test_create_patient_invalid_data(test_db, sample_patient):
    """Invalid patient data must not be created."""
    sample_patient["name"] = ""
    ok, err, patient_id = create_patient(sample_patient)
    assert ok is False
    assert patient_id is None


def test_create_patient_discharge_before_admission(test_db, sample_patient):
    """Patient with discharge before admission must not be created."""
    sample_patient["admission_date"] = "2025-01-22"
    sample_patient["discharge_date"] = "2025-01-08"
    ok, err, patient_id = create_patient(sample_patient)
    assert ok is False
    assert patient_id is None


def test_create_patient_without_db(tmp_path, monkeypatch, sample_patient):
    """create_patient must fail gracefully without database."""
    monkeypatch.setattr(patients_module, "DB_FILE",
                        tmp_path / "nonexistent.db")
    ok, err, patient_id = create_patient(sample_patient)
    assert ok is False
    assert patient_id is None


# ================================================================
# GET PATIENT TESTS
# ================================================================

def test_get_patient_exists(test_db, sample_patient):
    """get_patient must return the correct patient."""
    _, _, pid = create_patient(sample_patient)
    p = get_patient(pid)
    assert p is not None
    assert p["name"] == "Maria Hoffmann"
    assert p["diagnosis"] == "Chronic Heart Failure"


def test_get_patient_not_found(test_db):
    """get_patient must return None for non-existent ID."""
    p = get_patient(99999)
    assert p is None


def test_get_patient_all_fields(test_db, sample_patient):
    """get_patient must return all fields including new ones."""
    _, _, pid = create_patient(sample_patient)
    p = get_patient(pid)
    assert p["medication"]     == "Furosemide 40mg"
    assert p["physician_name"] == "Dr. Mueller"
    assert p["icd10_code"]     == "I50.0"
    assert p["followup_date"]  == "2025-02-05"


def test_get_patient_los_computed(test_db, sample_patient):
    """length_of_stay must be computed automatically."""
    _, _, pid = create_patient(sample_patient)
    p = get_patient(pid)
    assert p["length_of_stay"] == 14  # Jan 8 to Jan 22 = 14 days


# ================================================================
# GET ALL PATIENTS TESTS
# ================================================================

def test_get_all_patients_empty(test_db):
    """get_all_patients must return empty list when no patients."""
    result = get_all_patients()
    assert result == []


def test_get_all_patients_returns_list(test_db, sample_patient):
    """get_all_patients must return all active patients."""
    create_patient(sample_patient)
    sample_patient["name"] = "Klaus Berger"
    create_patient(sample_patient)
    result = get_all_patients()
    assert len(result) == 2


def test_get_all_patients_excludes_deleted(test_db, sample_patient):
    """get_all_patients must not include deleted patients."""
    _, _, pid = create_patient(sample_patient)
    delete_patient(pid)
    result = get_all_patients()
    assert len(result) == 0


def test_get_all_patients_include_deleted(test_db, sample_patient):
    """get_all_patients with include_deleted=True must include all."""
    _, _, pid = create_patient(sample_patient)
    delete_patient(pid)
    result = get_all_patients(include_deleted=True)
    assert len(result) == 1


# ================================================================
# UPDATE PATIENT TESTS
# ================================================================

def test_update_patient_success(test_db, sample_patient):
    """Valid update must succeed."""
    _, _, pid = create_patient(sample_patient)
    sample_patient["diagnosis"] = "Acute Heart Failure"
    ok, err, changed = update_patient(pid, sample_patient)
    assert ok is True
    assert "diagnosis" in changed


def test_update_patient_reflects_in_get(test_db, sample_patient):
    """Updated fields must be reflected when fetching patient."""
    _, _, pid = create_patient(sample_patient)
    sample_patient["medication"] = "Bisoprolol 5mg"
    update_patient(pid, sample_patient)
    p = get_patient(pid)
    assert p["medication"] == "Bisoprolol 5mg"


def test_update_patient_no_change(test_db, sample_patient):
    """Update with same data must succeed with empty changed list."""
    _, _, pid = create_patient(sample_patient)
    ok, err, changed = update_patient(pid, sample_patient)
    assert ok is True
    assert changed == []


def test_update_patient_not_found(test_db, sample_patient):
    """Update on non-existent patient must fail."""
    ok, err, changed = update_patient(99999, sample_patient)
    assert ok is False


def test_update_patient_invalid_data(test_db, sample_patient):
    """Update with invalid data must fail."""
    _, _, pid = create_patient(sample_patient)
    sample_patient["age"] = -5
    ok, err, changed = update_patient(pid, sample_patient)
    assert ok is False


# ================================================================
# DELETE AND RESTORE TESTS
# ================================================================

def test_delete_patient_success(test_db, sample_patient):
    """delete_patient must succeed for existing patient."""
    _, _, pid = create_patient(sample_patient)
    ok, err = delete_patient(pid)
    assert ok is True
    assert err == ""


def test_delete_patient_hides_from_get(test_db, sample_patient):
    """Deleted patient must not be returned by get_patient."""
    _, _, pid = create_patient(sample_patient)
    delete_patient(pid)
    p = get_patient(pid)
    assert p is None


def test_delete_patient_not_found(test_db):
    """delete_patient must fail for non-existent patient."""
    ok, err = delete_patient(99999)
    assert ok is False


def test_restore_patient_success(test_db, sample_patient):
    """restore_patient must successfully restore a deleted patient."""
    _, _, pid = create_patient(sample_patient)
    delete_patient(pid)
    ok, err = restore_patient(pid)
    assert ok is True
    assert err == ""


def test_restore_patient_visible_after_restore(test_db, sample_patient):
    """Restored patient must be visible via get_patient."""
    _, _, pid = create_patient(sample_patient)
    delete_patient(pid)
    restore_patient(pid)
    p = get_patient(pid)
    assert p is not None
    assert p["name"] == "Maria Hoffmann"


def test_restore_patient_not_deleted(test_db, sample_patient):
    """restore_patient must fail for active (non-deleted) patient."""
    _, _, pid = create_patient(sample_patient)
    ok, err = restore_patient(pid)
    assert ok is False


def test_get_last_deleted(test_db, sample_patient):
    """get_last_deleted must return the most recently deleted patient."""
    _, _, pid = create_patient(sample_patient)
    delete_patient(pid)
    p = get_last_deleted()
    assert p is not None
    assert p["name"] == "Maria Hoffmann"


def test_get_last_deleted_none_when_empty(test_db):
    """get_last_deleted must return None when nothing is deleted."""
    p = get_last_deleted()
    assert p is None


# ================================================================
# SEARCH TESTS
# ================================================================

def test_search_by_name(test_db, sample_patient):
    """Search by name must return matching patients."""
    create_patient(sample_patient)
    sample_patient["name"] = "Klaus Berger"
    create_patient(sample_patient)
    result = search_patients(query="Maria")
    assert len(result) == 1
    assert result[0]["name"] == "Maria Hoffmann"


def test_search_by_diagnosis(test_db, sample_patient):
    """Search by diagnosis must return matching patients."""
    create_patient(sample_patient)
    result = search_patients(query="Heart Failure")
    assert len(result) == 1


def test_search_by_notes(test_db, sample_patient):
    """Search by notes content must return matching patients."""
    create_patient(sample_patient)
    result = search_patients(query="stable")
    assert len(result) == 1


def test_search_no_match(test_db, sample_patient):
    """Search with no match must return empty list."""
    create_patient(sample_patient)
    result = search_patients(query="zzznomatch")
    assert len(result) == 0


def test_search_age_filter(test_db, sample_patient):
    """Age filter must exclude patients outside range."""
    create_patient(sample_patient)  # age 74
    result = search_patients(min_age=80)
    assert len(result) == 0
    result = search_patients(max_age=70)
    assert len(result) == 0
    result = search_patients(min_age=70, max_age=80)
    assert len(result) == 1


# ================================================================
# ANALYTICS TESTS
# ================================================================

def test_get_patient_count_empty(test_db):
    """Patient count must be 0 when empty."""
    assert get_patient_count() == 0


def test_get_patient_count(test_db, sample_patient):
    """Patient count must reflect active patients only."""
    create_patient(sample_patient)
    sample_patient["name"] = "Klaus"
    _, _, pid2 = create_patient(sample_patient)
    assert get_patient_count() == 2
    delete_patient(pid2)
    assert get_patient_count() == 1


def test_get_avg_los(test_db, sample_patient):
    """Average LOS must be computed correctly."""
    create_patient(sample_patient)  # LOS = 14 days
    avg = get_avg_los()
    assert avg == 14.0


# ================================================================
# EXPORT TESTS
# ================================================================

def test_export_anonymised_csv_empty(test_db):
    """Export must return header row even when empty."""
    csv = export_anonymised_csv()
    assert "id" in csv
    assert "diagnosis" in csv
    assert "Maria Hoffmann" not in csv


def test_export_anonymised_csv_no_names(test_db, sample_patient):
    """Export must not include patient names."""
    create_patient(sample_patient)
    csv = export_anonymised_csv()
    assert "Maria Hoffmann" not in csv
    assert "Heart Failure" in csv


# ================================================================
# RISK SCORING TESTS
# ================================================================

def test_risk_low():
    """Young patient with no flags must be Low risk."""
    p = {"age": 30, "diagnosis": "Appendectomy",
         "notes": "", "length_of_stay": 2}
    risk, score = compute_risk(p)
    assert risk == "Low"
    assert score <= 1


def test_risk_high_age():
    """Patient over 80 gets +2 age score."""
    p = {"age": 82, "diagnosis": "Fracture",
         "notes": "", "length_of_stay": 3}
    risk, score = compute_risk(p)
    assert score >= 2


def test_risk_chronic_diagnosis():
    """Chronic diagnosis keyword adds +2."""
    p = {"age": 50, "diagnosis": "Chronic heart failure",
         "notes": "", "length_of_stay": 5}
    risk, score = compute_risk(p)
    assert score >= 2


def test_risk_long_stay():
    """LOS >= 10 adds +1."""
    p = {"age": 40, "diagnosis": "Recovery",
         "notes": "", "length_of_stay": 12}
    _, score = compute_risk(p)
    assert score >= 1


def test_risk_complication_flag():
    """Complication in notes adds +1."""
    p = {"age": 40, "diagnosis": "Surgery",
         "notes": "post-op complication noted", "length_of_stay": 3}
    _, score = compute_risk(p)
    assert score >= 1


def test_risk_high_combined():
    """Old patient + chronic + complication must be High."""
    p = {"age": 81, "diagnosis": "heart failure",
         "notes": "sepsis noted", "length_of_stay": 12}
    risk, score = compute_risk(p)
    assert risk == "High"
    assert score >= 4


def test_risk_explanation_not_empty():
    """get_risk_explanation must return at least one reason."""
    p = {"age": 75, "diagnosis": "Diabetes",
         "notes": "", "length_of_stay": 5}
    reasons = get_risk_explanation(p)
    assert len(reasons) >= 1


def test_risk_explanation_low_risk():
    """Low risk patient must get 'no significant factors' message."""
    p = {"age": 25, "diagnosis": "Appendectomy",
         "notes": "", "length_of_stay": 2}
    reasons = get_risk_explanation(p)
    assert any("No significant" in r for r in reasons)
