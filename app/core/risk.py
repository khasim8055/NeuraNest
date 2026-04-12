# app/core/risk.py
# ================================================================
# NeuraCare — Readmission Risk Scoring
# Extracted from NeuraNest.py compute_readmission_risk()
# Rule-based, transparent, EU AI Act Article 13 compliant
# ================================================================
# Scoring rules (total score → risk level):
#   0–1  → Low
#   2–3  → Medium
#   4+   → High
#
# Score components:
#   +2  age >= 80
#   +1  age >= 65
#   +2  chronic diagnosis or notes keyword
#   +1  length of stay >= 10 days
#   +1  complication/infection flag in notes
# ================================================================

CHRONIC_KEYWORDS = [
    "diabetes", "herzinsuffizienz", "heart failure",
    "copd", "chronic", "renal", "dialysis", "stroke",
]

COMPLICATION_FLAGS = [
    "complication", "infection", "wound",
    "reoperation", "unstable", "sepsis",
]


def compute_risk(patient: dict) -> tuple[str, int]:
    """
    Compute readmission risk for one patient.

    Args:
        patient: dict with patient fields

    Returns:
        ("Low" | "Medium" | "High",  score_0_to_5)
    """
    score = 0

    age   = int(patient.get("age", 0) or 0)
    diag  = str(patient.get("diagnosis", "") or "").lower()
    notes = str(patient.get("notes", "") or "").lower()
    los   = int(patient.get("length_of_stay", 0) or 0)

    # Age component
    if age >= 80:
        score += 2
    elif age >= 65:
        score += 1

    # Chronic condition component
    if any(k in diag for k in CHRONIC_KEYWORDS) or \
       any(k in notes for k in CHRONIC_KEYWORDS):
        score += 2

    # Long stay component
    if los >= 10:
        score += 1

    # Complication flags component
    if any(k in notes for k in COMPLICATION_FLAGS):
        score += 1

    # Map score to label
    if score <= 1:
        return "Low",    score
    elif score <= 3:
        return "Medium", score
    else:
        return "High",   score


def get_risk_label_de(risk: str) -> str:
    """German translation of risk label."""
    return {"Low": "Niedrig", "Medium": "Mittel", "High": "Hoch"}.get(risk, risk)


def get_risk_explanation(patient: dict) -> list[str]:
    """
    Return plain-English list of reasons for the risk score.
    Used in UI to explain why a patient is High/Medium/Low risk.
    EU AI Act Article 13 — explainability requirement.
    """
    reasons = []

    age   = int(patient.get("age", 0) or 0)
    diag  = str(patient.get("diagnosis", "") or "").lower()
    notes = str(patient.get("notes", "") or "").lower()
    los   = int(patient.get("length_of_stay", 0) or 0)

    if age >= 80:
        reasons.append(f"Age {age} — patients over 80 have higher readmission rates")
    elif age >= 65:
        reasons.append(f"Age {age} — patients over 65 have elevated readmission risk")

    matched_chronic = [k for k in CHRONIC_KEYWORDS
                       if k in diag or k in notes]
    if matched_chronic:
        reasons.append(
            f"Chronic condition detected: {', '.join(matched_chronic[:2])}"
        )

    if los >= 10:
        reasons.append(
            f"Length of stay {los} days — stays over 10 days indicate higher complexity"
        )

    matched_flags = [k for k in COMPLICATION_FLAGS
                     if k in notes]
    if matched_flags:
        reasons.append(
            f"Clinical notes flag: {', '.join(matched_flags[:2])}"
        )

    if not reasons:
        reasons.append("No significant risk factors identified")

    return reasons
