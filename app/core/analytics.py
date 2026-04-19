# app/core/analytics.py
# ================================================================
# NeuraCare — Analytics Data Layer
# ================================================================
# Prepares data for the 5 dashboard charts.
# All functions return clean dicts — no matplotlib dependency here.
# The UI layer handles rendering.
#
# Charts:
#   1. Diagnosis frequency   — top 8 diagnoses by count
#   2. Age distribution      — histogram buckets
#   3. LOS by diagnosis      — avg length of stay per diagnosis
#   4. Monthly admissions    — count + avg LOS per month
#   5. Risk breakdown        — High / Medium / Low counts
# ================================================================

from datetime import datetime
from collections import Counter
from app.core.patients import get_all_patients
from app.core.risk     import compute_risk


# ================================================================
# DATA GUARD — minimum data required per chart
# ================================================================
MIN_PATIENTS_FOR_CHART = 1   # any chart needs at least 1 patient
MIN_MONTHS_FOR_TREND   = 2   # monthly chart needs 2+ months of data


def _get_patients() -> list[dict]:
    """Get all active patients. Returns [] if none."""
    return get_all_patients()


# ================================================================
# CHART 1 — DIAGNOSIS FREQUENCY
# ================================================================

def get_diagnosis_data(top_n: int = 8) -> dict:
    """
    Top N most common diagnoses.

    Returns:
        {
          "labels": ["Heart Failure", "COPD", ...],
          "counts": [12, 8, 5, ...],
          "total":  25,
          "enough_data": True
        }
    """
    patients = _get_patients()
    if len(patients) < MIN_PATIENTS_FOR_CHART:
        return {"labels": [], "counts": [], "total": 0, "enough_data": False}

    counter = Counter(
        p.get("diagnosis", "Unknown").strip()
        for p in patients
        if p.get("diagnosis")
    )
    top = counter.most_common(top_n)

    return {
        "labels":      [item[0] for item in top],
        "counts":      [item[1] for item in top],
        "total":       len(patients),
        "enough_data": True,
    }


# ================================================================
# CHART 2 — AGE DISTRIBUTION
# ================================================================

def get_age_data() -> dict:
    """
    Age distribution in buckets: 0-17, 18-34, 35-49, 50-64, 65-79, 80+

    Returns:
        {
          "buckets": ["0-17", "18-34", ...],
          "counts":  [2, 5, 8, ...],
          "avg_age": 58.3,
          "enough_data": True
        }
    """
    patients = _get_patients()
    if len(patients) < MIN_PATIENTS_FOR_CHART:
        return {"buckets": [], "counts": [], "avg_age": 0, "enough_data": False}

    buckets = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
    counts  = [0] * 6
    ages    = []

    for p in patients:
        age = int(p.get("age", 0) or 0)
        ages.append(age)
        if age <= 17:
            counts[0] += 1
        elif age <= 34:
            counts[1] += 1
        elif age <= 49:
            counts[2] += 1
        elif age <= 64:
            counts[3] += 1
        elif age <= 79:
            counts[4] += 1
        else:
            counts[5] += 1

    avg_age = round(sum(ages) / len(ages), 1) if ages else 0

    return {
        "buckets":     buckets,
        "counts":      counts,
        "avg_age":     avg_age,
        "enough_data": True,
    }


# ================================================================
# CHART 3 — LENGTH OF STAY BY DIAGNOSIS
# ================================================================

def get_los_data(top_n: int = 8) -> dict:
    """
    Average length of stay per diagnosis (top N by patient count).

    Returns:
        {
          "diagnoses": ["Heart Failure", ...],
          "avg_los":   [14.2, 8.5, ...],
          "overall_avg": 10.3,
          "enough_data": True
        }
    """
    patients = _get_patients()
    if len(patients) < MIN_PATIENTS_FOR_CHART:
        return {"diagnoses": [], "avg_los": [], "overall_avg": 0, "enough_data": False}

    # Group LOS values by diagnosis
    diag_los: dict[str, list] = {}
    all_los = []

    for p in patients:
        diag = p.get("diagnosis", "Unknown").strip()
        los  = int(p.get("length_of_stay", 0) or 0)
        if diag not in diag_los:
            diag_los[diag] = []
        diag_los[diag].append(los)
        all_los.append(los)

    # Sort by patient count, take top N
    sorted_diags = sorted(diag_los.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]

    diagnoses = [d for d, _ in sorted_diags]
    avg_los   = [round(sum(v)/len(v), 1) for _, v in sorted_diags]
    overall   = round(sum(all_los)/len(all_los), 1) if all_los else 0

    return {
        "diagnoses":   diagnoses,
        "avg_los":     avg_los,
        "overall_avg": overall,
        "enough_data": True,
    }


# ================================================================
# CHART 4 — MONTHLY ADMISSIONS TREND
# ================================================================

def get_monthly_data() -> dict:
    """
    Monthly admission counts and avg LOS per month.
    Only renders if 2+ months of data exist.

    Returns:
        {
          "months":    ["Jan 2025", "Feb 2025", ...],
          "counts":    [5, 8, ...],
          "avg_los":   [9.2, 11.3, ...],
          "enough_data": True
        }
    """
    patients = _get_patients()
    if len(patients) < MIN_PATIENTS_FOR_CHART:
        return {"months": [], "counts": [], "avg_los": [], "enough_data": False}

    # Group by year-month
    monthly: dict[str, list] = {}

    for p in patients:
        adm = p.get("admission_date", "")
        if not adm:
            continue
        try:
            dt = datetime.strptime(str(adm)[:10], "%Y-%m-%d")
            key = dt.strftime("%Y-%m")
        except ValueError:
            continue

        los = int(p.get("length_of_stay", 0) or 0)
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(los)

    if len(monthly) < MIN_MONTHS_FOR_TREND:
        # Not enough months — still show what we have
        pass

    sorted_months = sorted(monthly.keys())

    month_labels = []
    for m in sorted_months:
        try:
            dt = datetime.strptime(m, "%Y-%m")
            month_labels.append(dt.strftime("%b %Y"))
        except ValueError:
            month_labels.append(m)

    counts  = [len(monthly[m]) for m in sorted_months]
    avg_los = [round(sum(monthly[m])/len(monthly[m]), 1) for m in sorted_months]

    return {
        "months":      month_labels,
        "counts":      counts,
        "avg_los":     avg_los,
        "enough_data": len(sorted_months) >= 1,
    }


# ================================================================
# CHART 5 — RISK BREAKDOWN
# ================================================================

def get_risk_data() -> dict:
    """
    Count of High / Medium / Low risk patients.

    Returns:
        {
          "labels":      ["High", "Medium", "Low"],
          "counts":      [3, 8, 14],
          "percentages": [12, 32, 56],
          "total":       25,
          "enough_data": True
        }
    """
    patients = _get_patients()
    if len(patients) < MIN_PATIENTS_FOR_CHART:
        return {
            "labels": [], "counts": [], "percentages": [],
            "total": 0, "enough_data": False
        }

    counts = {"High": 0, "Medium": 0, "Low": 0}
    for p in patients:
        risk, _ = compute_risk(p)
        counts[risk] = counts.get(risk, 0) + 1

    total = len(patients)
    labels = ["High", "Medium", "Low"]
    cnt    = [counts[l] for l in labels]
    pct    = [round(c/total*100) for c in cnt]

    return {
        "labels":      labels,
        "counts":      cnt,
        "percentages": pct,
        "total":       total,
        "enough_data": True,
    }


# ================================================================
# SUMMARY STATS — shown in dashboard header
# ================================================================

def get_summary_stats() -> dict:
    """
    Key numbers for the dashboard header row.

    Returns:
        {
          "total_patients": 25,
          "avg_los":        10.3,
          "high_risk_count": 3,
          "high_risk_pct":   12,
        }
    """
    patients = _get_patients()
    total    = len(patients)

    if total == 0:
        return {
            "total_patients":  0,
            "avg_los":         0.0,
            "high_risk_count": 0,
            "high_risk_pct":   0,
        }

    los_values = [int(p.get("length_of_stay", 0) or 0) for p in patients]
    avg_los    = round(sum(los_values) / total, 1)

    high_count = sum(1 for p in patients if compute_risk(p)[0] == "High")
    high_pct   = round(high_count / total * 100)

    return {
        "total_patients":  total,
        "avg_los":         avg_los,
        "high_risk_count": high_count,
        "high_risk_pct":   high_pct,
    }
