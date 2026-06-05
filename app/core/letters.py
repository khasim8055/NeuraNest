# app/core/letters.py
# ================================================================
# NeuraCare — Discharge Letter Generation Engine
# ================================================================
# Autonomy Levels:
#   AL0 — bullet summary (fastest, structured)
#   AL1 — short paragraph (standard clinical letter)
#   AL2 — extended letter with full notes (detailed)
#   AL3 — AI-generated via Ollama (Day 9, not built yet)
#
# Languages:
#   Deutsch (default) — Arztbrief format
#   English           — standard discharge summary
#
# All templates include:
#   - Patient identity block
#   - Diagnosis + ICD-10
#   - Admission / discharge dates + LOS
#   - Responsible physician
#   - Readmission risk level
#   - Clinical notes (AL1+)
#   - Medication at discharge (AL1+)
#   - Follow-up recommendation (AL2+)
# ================================================================

from datetime import datetime
from app.core.risk import compute_risk, get_risk_label_de


# ================================================================
# HELPERS
# ================================================================

def _fmt_date(date_str: str, lang: str = "Deutsch") -> str:
    """Format ISO date string to readable format."""
    if not date_str:
        return "—"
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        if lang == "Deutsch":
            return d.strftime("%d.%m.%Y")
        return d.strftime("%d %B %Y")
    except ValueError:
        return str(date_str)[:10]


def _los_text(los: int, lang: str = "Deutsch") -> str:
    """Format length of stay."""
    if los is None or los == 0:
        return "—"
    if lang == "Deutsch":
        return f"{los} Tag{'e' if los != 1 else ''}"
    return f"{los} day{'s' if los != 1 else ''}"


def _risk_text(risk: str, lang: str = "Deutsch") -> str:
    """Localised risk label."""
    if lang == "Deutsch":
        return get_risk_label_de(risk)
    return risk


def _physician_line(physician: str, lang: str = "Deutsch") -> str:
    """Format physician attribution line."""
    if not physician or not physician.strip():
        return ""
    if lang == "Deutsch":
        return f"Behandelnder Arzt: {physician}"
    return f"Responsible physician: {physician}"


def _followup_line(followup_date: str, lang: str = "Deutsch") -> str:
    """Format follow-up recommendation line."""
    if not followup_date or not followup_date.strip():
        if lang == "Deutsch":
            return "Ambulante Nachsorge empfohlen."
        return "Outpatient follow-up recommended."
    date_fmt = _fmt_date(followup_date, lang)
    if lang == "Deutsch":
        return f"Ambulanter Nachsorgetermin: {date_fmt}"
    return f"Follow-up appointment: {date_fmt}"


def _divider(char: str = "─", width: int = 52) -> str:
    return char * width


# ================================================================
# AL0 — BULLET SUMMARY
# ================================================================

def generate_al0(patient: dict, lang: str = "Deutsch") -> str:
    """
    AL0 — Structured bullet summary.
    Fastest option. No prose. Key facts only.
    Good for: quick reference, handover notes.
    """
    name      = patient.get("name", "—")
    age       = patient.get("age", "—")
    diag      = patient.get("diagnosis", "—")
    icd10     = patient.get("icd10_code", "") or ""
    adm       = _fmt_date(patient.get("admission_date", ""), lang)
    dis       = _fmt_date(patient.get("discharge_date", ""), lang)
    los       = int(patient.get("length_of_stay", 0) or 0)
    physician = patient.get("physician_name", "") or ""
    icd_str   = f" [{icd10}]" if icd10 else ""

    risk, _ = compute_risk(patient)

    if lang == "Deutsch":
        lines = [
            f"ENTLASSUNGSZUSAMMENFASSUNG",
            _divider(),
            "",
            f"Patient:         {name}, {age} Jahre",
            f"Diagnose:        {diag}{icd_str}",
            f"Aufnahme:        {adm}",
            f"Entlassung:      {dis}",
            f"Verweildauer:    {_los_text(los, lang)}",
            f"Risiko:          {_risk_text(risk, lang)} (Wiederaufnahme)",
        ]
        if physician:
            lines.append(f"Behandl. Arzt:   {physician}")
        lines += [
            "",
            "Status bei Entlassung:",
            "• Zustand stabil, Entlassung möglich",
            f"• {_followup_line(patient.get('followup_date',''), lang)}",
        ]
    else:
        lines = [
            f"DISCHARGE SUMMARY",
            _divider(),
            "",
            f"Patient:         {name}, {age} years",
            f"Diagnosis:       {diag}{icd_str}",
            f"Admission:       {adm}",
            f"Discharge:       {dis}",
            f"Length of stay:  {_los_text(los, lang)}",
            f"Risk:            {risk} (readmission)",
        ]
        if physician:
            lines.append(f"Physician:       {physician}")
        lines += [
            "",
            "Status at discharge:",
            "• Condition stable, patient cleared for discharge",
            f"• {_followup_line(patient.get('followup_date',''), lang)}",
        ]

    return "\n".join(lines)


# ================================================================
# AL1 — STANDARD PARAGRAPH LETTER
# ================================================================

def generate_al1(patient: dict, lang: str = "Deutsch") -> str:
    """
    AL1 — Standard clinical letter with paragraph prose.
    Standard for most discharge letters.
    Good for: routine discharges, standard GP referrals.
    """
    name      = patient.get("name", "—")
    age       = int(patient.get("age", 0) or 0)
    diag      = patient.get("diagnosis", "—")
    icd10     = patient.get("icd10_code", "") or ""
    adm       = _fmt_date(patient.get("admission_date", ""), lang)
    dis       = _fmt_date(patient.get("discharge_date", ""), lang)
    los       = int(patient.get("length_of_stay", 0) or 0)
    notes     = patient.get("notes", "") or ""
    medication = patient.get("medication", "") or ""
    physician  = patient.get("physician_name", "") or ""
    followup   = patient.get("followup_date", "") or ""
    icd_str    = f" ({icd10})" if icd10 else ""

    # Truncate notes to 300 chars for AL1
    notes_short = notes[:300] + "…" if len(notes) > 300 else notes

    risk, _ = compute_risk(patient)

    if lang == "Deutsch":
        today = datetime.now().strftime("%d.%m.%Y")
        header = (
            f"ARZTBRIEF\n"
            f"{_divider()}\n"
            f"Datum: {today}\n"
        )
        if physician:
            header += f"Ausstellender Arzt: {physician}\n"
        header += "\n"

        body = (
            f"Sehr geehrte Damen und Herren,\n\n"
            f"wir berichten über unseren gemeinsamen Patienten / unsere gemeinsame Patientin:\n\n"
            f"Name:         {name}\n"
            f"Alter:        {age} Jahre\n"
            f"Diagnose:     {diag}{icd_str}\n"
            f"Aufnahme:     {adm}\n"
            f"Entlassung:   {dis}\n"
            f"Verweildauer: {_los_text(los, lang)}\n"
            f"\n"
            f"{_divider('─', 30)}\n"
            f"\n"
        )

        if notes_short:
            body += (
                f"Klinischer Verlauf:\n"
                f"{notes_short}\n\n"
            )

        if medication:
            body += (
                f"Medikation bei Entlassung:\n"
                f"{medication}\n\n"
            )

        body += (
            f"Wiederaufnahmerisiko: {_risk_text(risk, lang)}\n\n"
            f"{_followup_line(followup, lang)}\n\n"
            f"Mit freundlichen kollegialen Grüßen"
        )

        if physician:
            body += f"\n{physician}"

        return header + body

    else:
        today = datetime.now().strftime("%d %B %Y")
        header = (
            f"DISCHARGE LETTER\n"
            f"{_divider()}\n"
            f"Date: {today}\n"
        )
        if physician:
            header += f"Issuing physician: {physician}\n"
        header += "\n"

        body = (
            f"Dear Colleague,\n\n"
            f"We are writing regarding our shared patient:\n\n"
            f"Name:           {name}\n"
            f"Age:            {age} years\n"
            f"Diagnosis:      {diag}{icd_str}\n"
            f"Admission:      {adm}\n"
            f"Discharge:      {dis}\n"
            f"Length of stay: {_los_text(los, lang)}\n"
            f"\n"
            f"{_divider('─', 30)}\n"
            f"\n"
        )

        if notes_short:
            body += (
                f"Clinical course:\n"
                f"{notes_short}\n\n"
            )

        if medication:
            body += (
                f"Medication at discharge:\n"
                f"{medication}\n\n"
            )

        body += (
            f"Readmission risk: {risk}\n\n"
            f"{_followup_line(followup, lang)}\n\n"
            f"Yours sincerely"
        )

        if physician:
            body += f"\n{physician}"

        return header + body


# ================================================================
# AL2 — EXTENDED LETTER WITH FULL NOTES
# ================================================================

def generate_al2(patient: dict, lang: str = "Deutsch") -> str:
    """
    AL2 — Comprehensive extended letter.
    Full clinical notes included. Full follow-up detail.
    Good for: complex cases, specialist referrals, legal documentation.
    """
    name       = patient.get("name", "—")
    age        = int(patient.get("age", 0) or 0)
    dob        = _fmt_date(patient.get("date_of_birth", ""), lang)
    diag       = patient.get("diagnosis", "—")
    icd10      = patient.get("icd10_code", "") or ""
    adm        = _fmt_date(patient.get("admission_date", ""), lang)
    dis        = _fmt_date(patient.get("discharge_date", ""), lang)
    los        = int(patient.get("length_of_stay", 0) or 0)
    notes      = patient.get("notes", "") or ""
    medication = patient.get("medication", "") or ""
    physician  = patient.get("physician_name", "") or ""
    followup   = patient.get("followup_date", "") or ""
    icd_str    = f" ({icd10})" if icd10 else ""

    risk, score = compute_risk(patient)
    from app.core.risk import get_risk_explanation
    risk_reasons = get_risk_explanation(patient, lang)

    if lang == "Deutsch":
        today = datetime.now().strftime("%d.%m.%Y")
        letter = (
            f"AUSFÜHRLICHER ARZTBRIEF\n"
            f"{_divider('═', 52)}\n"
            f"Datum der Erstellung: {today}\n"
        )
        if physician:
            letter += f"Ausstellender Arzt:   {physician}\n"

        letter += (
            f"\n"
            f"{_divider()}\n"
            f"PATIENTENDATEN\n"
            f"{_divider()}\n"
            f"\n"
            f"Name:              {name}\n"
            f"Alter:             {age} Jahre\n"
        )
        if dob and dob != "—":
            letter += f"Geburtsdatum:      {dob}\n"

        letter += (
            f"Hauptdiagnose:     {diag}{icd_str}\n"
            f"Aufnahmedatum:     {adm}\n"
            f"Entlassungsdatum:  {dis}\n"
            f"Verweildauer:      {_los_text(los, lang)}\n"
            f"\n"
            f"{_divider()}\n"
            f"KLINISCHER VERLAUF\n"
            f"{_divider()}\n"
            f"\n"
            f"{notes if notes else 'Keine detaillierten Notizen vorhanden.'}\n"
            f"\n"
        )

        if medication:
            letter += (
                f"{_divider()}\n"
                f"MEDIKATION BEI ENTLASSUNG\n"
                f"{_divider()}\n"
                f"\n"
                f"{medication}\n"
                f"\n"
            )

        letter += (
            f"{_divider()}\n"
            f"RISIKOBEWERTUNG\n"
            f"{_divider()}\n"
            f"\n"
            f"Wiederaufnahmerisiko: {_risk_text(risk, lang)} (Score: {score}/6)\n"
            f"\n"
            f"Begründung:\n"
        )
        for reason in risk_reasons:
            letter += f"• {reason}\n"

        letter += (
            f"\n"
            f"{_divider()}\n"
            f"EMPFEHLUNGEN\n"
            f"{_divider()}\n"
            f"\n"
            f"{_followup_line(followup, lang)}\n"
            f"\n"
            f"Mit freundlichen kollegialen Grüßen"
        )
        if physician:
            letter += f"\n{physician}"

    else:
        today = datetime.now().strftime("%d %B %Y")
        letter = (
            f"COMPREHENSIVE DISCHARGE LETTER\n"
            f"{_divider('═', 52)}\n"
            f"Date of issue: {today}\n"
        )
        if physician:
            letter += f"Issuing physician: {physician}\n"

        letter += (
            f"\n"
            f"{_divider()}\n"
            f"PATIENT INFORMATION\n"
            f"{_divider()}\n"
            f"\n"
            f"Name:             {name}\n"
            f"Age:              {age} years\n"
        )
        if dob and dob != "—":
            letter += f"Date of birth:    {dob}\n"

        letter += (
            f"Primary diagnosis: {diag}{icd_str}\n"
            f"Admission date:   {adm}\n"
            f"Discharge date:   {dis}\n"
            f"Length of stay:   {_los_text(los, lang)}\n"
            f"\n"
            f"{_divider()}\n"
            f"CLINICAL COURSE\n"
            f"{_divider()}\n"
            f"\n"
            f"{notes if notes else 'No detailed notes available.'}\n"
            f"\n"
        )

        if medication:
            letter += (
                f"{_divider()}\n"
                f"MEDICATION AT DISCHARGE\n"
                f"{_divider()}\n"
                f"\n"
                f"{medication}\n"
                f"\n"
            )

        letter += (
            f"{_divider()}\n"
            f"RISK ASSESSMENT\n"
            f"{_divider()}\n"
            f"\n"
            f"Readmission risk: {risk} (Score: {score}/6)\n"
            f"\n"
            f"Risk factors:\n"
        )
        for reason in risk_reasons:
            letter += f"• {reason}\n"

        letter += (
            f"\n"
            f"{_divider()}\n"
            f"RECOMMENDATIONS\n"
            f"{_divider()}\n"
            f"\n"
            f"{_followup_line(followup, lang)}\n"
            f"\n"
            f"Yours sincerely"
        )
        if physician:
            letter += f"\n{physician}"

    return letter


# ================================================================
# MAIN GENERATE FUNCTION — single entry point
# ================================================================

def generate_letter(
    patient: dict,
    level: int = 1,
    lang: str = "Deutsch"
) -> tuple[bool, str, str]:
    """
    Generate a discharge letter for a patient.

    Args:
        patient: patient dict from get_patient()
        level:   0=AL0, 1=AL1, 2=AL2, 3=AL3(not yet)
        lang:    "Deutsch" or "English"

    Returns:
        (True,  letter_text, "")      on success
        (False, "",          error)   on failure
    """
    if not patient:
        return False, "", "No patient data provided."

    if lang not in ("Deutsch", "English"):
        lang = "Deutsch"

    try:
        if level == 0:
            text = generate_al0(patient, lang)
        elif level == 1:
            text = generate_al1(patient, lang)
        elif level == 2:
            text = generate_al2(patient, lang)
        elif level == 3:
            from app.core.ollama_client import generate_al3
            return generate_al3(patient, lang)
        else:
            return False, "", f"Unknown level: {level}"

        return True, text, ""

    except Exception as e:
        return False, "", f"Letter generation failed: {str(e)}"


# ================================================================
# LEVEL DESCRIPTIONS — for UI display
# ================================================================

LEVEL_INFO = {
    0: {
        "name":        "AL0 — Schnellzusammenfassung",
        "name_en":     "AL0 — Quick Summary",
        "description": "Strukturierte Stichpunkte. Ideal für schnelle Übergaben.",
        "description_en": "Structured bullet points. Best for quick handovers.",
        "time":        "< 1 Sekunde",
    },
    1: {
        "name":        "AL1 — Standardbrief",
        "name_en":     "AL1 — Standard Letter",
        "description": "Kurzer Fließtext-Arztbrief. Standard für Routineentlassungen.",
        "description_en": "Short prose letter. Standard for routine discharges.",
        "time":        "< 1 Sekunde",
    },
    2: {
        "name":        "AL2 — Ausführlicher Brief",
        "name_en":     "AL2 — Extended Letter",
        "description": "Vollständiger Arztbrief mit allen Notizen und Risikobewertung.",
        "description_en": "Full letter with all notes and risk assessment.",
        "time":        "< 1 Sekunde",
    },
    3: {
        "name":        "AL3 — KI-generiert (bald)",
        "name_en":     "AL3 — AI-generated (coming soon)",
        "description": "Natürlichsprachlicher Brief via Ollama + Mistral. In Entwicklung.",
        "description_en": "Natural language letter via Ollama + Mistral. Coming in Day 9.",
        "time":        "~15 Sekunden",
    },
}
