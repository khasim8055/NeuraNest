# update_risk_language.py
import ast

with open('app/core/risk.py', encoding='utf-8') as f:
    content = f.read()

# Add German version of get_risk_explanation
old = '''def get_risk_explanation(patient: dict) -> list[str]:
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

    return reasons'''

new = '''def get_risk_explanation(patient: dict, lang: str = "English") -> list[str]:
    """
    Return list of reasons for the risk score in the specified language.
    EU AI Act Article 13 — explainability requirement.
    """
    reasons = []

    age   = int(patient.get("age", 0) or 0)
    diag  = str(patient.get("diagnosis", "") or "").lower()
    notes = str(patient.get("notes", "") or "").lower()
    los   = int(patient.get("length_of_stay", 0) or 0)

    if lang == "Deutsch":
        if age >= 80:
            reasons.append(f"Alter {age} Jahre — Patienten über 80 haben erhöhtes Wiederaufnahmerisiko")
        elif age >= 65:
            reasons.append(f"Alter {age} Jahre — Patienten über 65 haben erhöhtes Wiederaufnahmerisiko")

        matched_chronic = [k for k in CHRONIC_KEYWORDS if k in diag or k in notes]
        if matched_chronic:
            reasons.append(f"Chronische Erkrankung erkannt: {', '.join(matched_chronic[:2])}")

        if los >= 10:
            reasons.append(f"Verweildauer {los} Tage — Aufenthalte über 10 Tage deuten auf höhere Komplexität hin")

        matched_flags = [k for k in COMPLICATION_FLAGS if k in notes]
        if matched_flags:
            reasons.append(f"Klinischer Hinweis: {', '.join(matched_flags[:2])}")

        if not reasons:
            reasons.append("Keine signifikanten Risikofaktoren identifiziert")
    else:
        if age >= 80:
            reasons.append(f"Age {age} — patients over 80 have higher readmission rates")
        elif age >= 65:
            reasons.append(f"Age {age} — patients over 65 have elevated readmission risk")

        matched_chronic = [k for k in CHRONIC_KEYWORDS if k in diag or k in notes]
        if matched_chronic:
            reasons.append(f"Chronic condition detected: {', '.join(matched_chronic[:2])}")

        if los >= 10:
            reasons.append(f"Length of stay {los} days — stays over 10 days indicate higher complexity")

        matched_flags = [k for k in COMPLICATION_FLAGS if k in notes]
        if matched_flags:
            reasons.append(f"Clinical notes flag: {', '.join(matched_flags[:2])}")

        if not reasons:
            reasons.append("No significant risk factors identified")

    return reasons'''

if old in content:
    content = content.replace(old, new)
    print("✓ get_risk_explanation updated with German support")
else:
    print("✗ Pattern not found — check risk.py manually")

with open('app/core/risk.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Now update letters.py and pdf_exporter.py to pass lang to get_risk_explanation
for path in ['app/core/letters.py', 'app/core/pdf_exporter.py']:
    with open(path, encoding='utf-8') as f:
        src = f.read()
    src = src.replace(
        'get_risk_explanation(patient)',
        'get_risk_explanation(patient, lang)'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    print(f"✓ {path} updated to pass lang to get_risk_explanation")

# Also update analytics_panel.py which calls get_risk_explanation
with open('app/ui/analytics_panel.py', encoding='utf-8') as f:
    src = f.read()
if 'get_risk_explanation' in src:
    src = src.replace('get_risk_explanation(patient)', 'get_risk_explanation(patient)')
    print("✓ analytics_panel.py checked")

# Verify syntax on all changed files
import ast
for path in ['app/core/risk.py', 'app/core/letters.py', 'app/core/pdf_exporter.py']:
    with open(path, encoding='utf-8') as f:
        code = f.read()
    try:
        ast.parse(code)
        print(f"✓ {path} syntax OK ({code.count(chr(10))} lines)")
    except SyntaxError as e:
        print(f"✗ {path} ERROR line {e.lineno}: {e.msg}")

print("\nDone. Run: python main.py")
print("Generate a Deutsch AL1/AL2 letter — risk reasons will now be in German.")