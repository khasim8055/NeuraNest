# app/core/csv_export.py
# ================================================================
# NeuraCare — CSV Export Module
# ================================================================
# Two export types:
#   1. Anonymised patient data — no names, no notes, safe to share
#   2. Audit log export       — full GDPR Article 30 compliance log
#
# Both save to desktop and return the file path.
# ================================================================

import os
import csv
from datetime import datetime
from pathlib import Path

from app.core.patients import get_all_patients
from app.core.audit    import export_audit_to_csv
from app.core.risk     import compute_risk


# ================================================================
# HELPERS
# ================================================================

def _default_save_dir() -> Path:
    """Default save location — user desktop."""
    desktop = Path.home() / "Desktop"
    return desktop if desktop.exists() else Path.home()


def _make_filename(prefix: str, ext: str = "csv") -> str:
    """Generate a timestamped filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"NeuraCare_{prefix}_{ts}.{ext}"


# ================================================================
# EXPORT 1 — ANONYMISED PATIENT DATA
# ================================================================

def export_anonymised_patients(
    save_dir: str | None = None
) -> tuple[bool, str, str]:
    """
    Export anonymised patient data as CSV.
    No names, no notes, no personal identifiers.
    Safe for management reporting and research.

    Returns:
        (True,  file_path, "")       on success
        (False, "",        error)    on failure
    """
    patients = get_all_patients()
    if not patients:
        return False, "", "No patients to export."

    if save_dir is None:
        save_dir = str(_default_save_dir())

    file_path = Path(save_dir) / _make_filename("Anonymised_Data")

    headers = [
        "patient_id", "age", "diagnosis", "icd10_code",
        "admission_date", "discharge_date", "length_of_stay",
        "physician_name", "risk_level", "risk_score",
        "medication_count", "has_followup",
    ]

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for p in patients:
                risk, score = compute_risk(p)
                # Count medication lines as proxy for complexity
                medication = p.get("medication", "") or ""
                med_count  = len([m for m in medication.split(",") if m.strip()])
                has_followup = 1 if p.get("followup_date") else 0

                writer.writerow({
                    "patient_id":      f"PT-{p.get('id', 0):04d}",
                    "age":             p.get("age", ""),
                    "diagnosis":       p.get("diagnosis", ""),
                    "icd10_code":      p.get("icd10_code", "") or "",
                    "admission_date":  p.get("admission_date", ""),
                    "discharge_date":  p.get("discharge_date", ""),
                    "length_of_stay":  p.get("length_of_stay", 0) or 0,
                    "physician_name":  p.get("physician_name", "") or "",
                    "risk_level":      risk,
                    "risk_score":      score,
                    "medication_count": med_count,
                    "has_followup":    has_followup,
                })

        return True, str(file_path), ""

    except Exception as e:
        return False, "", f"Export failed: {str(e)}"


# ================================================================
# EXPORT 2 — AUDIT LOG
# ================================================================

def export_audit_log(
    save_dir: str | None = None
) -> tuple[bool, str, str]:
    """
    Export the full audit log as CSV.
    Required for GDPR Article 30 compliance reports.

    Returns:
        (True,  file_path, "")       on success
        (False, "",        error)    on failure
    """
    csv_text = export_audit_to_csv()
    if not csv_text or csv_text.count("\n") < 1:
        return False, "", "No audit log entries to export."

    if save_dir is None:
        save_dir = str(_default_save_dir())

    file_path = Path(save_dir) / _make_filename("Audit_Log")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        return True, str(file_path), ""
    except Exception as e:
        return False, "", f"Export failed: {str(e)}"


# ================================================================
# EXPORT 3 — PRACTICE SUMMARY REPORT
# ================================================================

def export_practice_summary(
    save_dir: str | None = None
) -> tuple[bool, str, str]:
    """
    Export a practice summary report as CSV.
    Key statistics per diagnosis for management.

    Returns:
        (True,  file_path, "")       on success
        (False, "",        error)    on failure
    """
    patients = get_all_patients()
    if not patients:
        return False, "", "No patients to summarise."

    if save_dir is None:
        save_dir = str(_default_save_dir())

    file_path = Path(save_dir) / _make_filename("Practice_Summary")

    # Group by diagnosis
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for p in patients:
        diag = p.get("diagnosis", "Unknown").strip()
        groups[diag].append(p)

    headers = [
        "diagnosis", "icd10_code", "patient_count",
        "avg_age", "avg_los_days",
        "high_risk_count", "medium_risk_count", "low_risk_count",
    ]

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for diag, pts in sorted(groups.items(),
                                    key=lambda x: len(x[1]), reverse=True):
                ages    = [int(p.get("age", 0) or 0) for p in pts]
                los_vals = [int(p.get("length_of_stay", 0) or 0) for p in pts]
                risks   = [compute_risk(p)[0] for p in pts]

                icd = pts[0].get("icd10_code", "") or "" if pts else ""

                writer.writerow({
                    "diagnosis":        diag,
                    "icd10_code":       icd,
                    "patient_count":    len(pts),
                    "avg_age":          round(sum(ages)/len(ages), 1) if ages else 0,
                    "avg_los_days":     round(sum(los_vals)/len(los_vals), 1) if los_vals else 0,
                    "high_risk_count":  risks.count("High"),
                    "medium_risk_count": risks.count("Medium"),
                    "low_risk_count":   risks.count("Low"),
                })

        return True, str(file_path), ""

    except Exception as e:
        return False, "", f"Export failed: {str(e)}"
