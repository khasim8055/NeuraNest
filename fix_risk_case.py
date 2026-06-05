# fix_risk_case.py
import ast

with open('app/core/risk.py', encoding='utf-8') as f:
    content = f.read()

# Fix lowercase in matched_chronic and matched_flags display
old1 = "        if matched_chronic:\n            reasons.append(f\"Chronische Erkrankung erkannt: {', '.join(matched_chronic[:2])}\")"
new1 = "        if matched_chronic:\n            reasons.append(f\"Chronische Erkrankung erkannt: {', '.join(k.capitalize() for k in matched_chronic[:2])}\")"
content = content.replace(old1, new1)

old2 = "        if matched_chronic:\n            reasons.append(f\"Chronic condition detected: {', '.join(matched_chronic[:2])}\")"
new2 = "        if matched_chronic:\n            reasons.append(f\"Chronic condition detected: {', '.join(k.capitalize() for k in matched_chronic[:2])}\")"
content = content.replace(old2, new2)

old3 = "        if matched_flags:\n            reasons.append(f\"Klinischer Hinweis: {', '.join(matched_flags[:2])}\")"
new3 = "        if matched_flags:\n            reasons.append(f\"Klinischer Hinweis: {', '.join(k.capitalize() for k in matched_flags[:2])}\")"
content = content.replace(old3, new3)

old4 = "        if matched_flags:\n            reasons.append(f\"Clinical notes flag: {', '.join(matched_flags[:2])}\")"
new4 = "        if matched_flags:\n            reasons.append(f\"Clinical notes flag: {', '.join(k.capitalize() for k in matched_flags[:2])}\")"
content = content.replace(old4, new4)

with open('app/core/risk.py', 'w', encoding='utf-8') as f:
    f.write(content)

ast.parse(content)
print("✓ Fixed — keywords now capitalised in risk reasons")
print("  e.g. 'herzinsuffizienz' → 'Herzinsuffizienz'")