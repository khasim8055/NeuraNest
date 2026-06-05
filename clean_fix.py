# clean_fix.py
with open('app/ui/letter_panel.py', encoding='utf-8') as f:
    c = f.read()

fixes = [
    # Fix 1 - title label default
    ('self.title_label = QLabel("Discharge Letter")',
     'self.title_label = QLabel("Arztbrief")'),

    # Fix 2 - English generate button text
    ('self.generate_btn.setText("Generate Letter")',
     'self.generate_btn.setText("Generate Discharge Letter")'),

    # Fix 3 - after generate always resets to German regardless of lang
    ('        self.generate_btn.setText("Arztbrief generieren")\n',
     '        self.generate_btn.setText("Arztbrief generieren" if self._lang == "Deutsch" else "Generate Discharge Letter")\n'),

    # Fix 4 - copy feedback respects language
    ('            self.copy_btn.setText("Copied!")',
     '            self.copy_btn.setText("Kopiert!" if self._lang == "Deutsch" else "Copied!")'),

    ('lambda: self.copy_btn.setText("Copy Text")',
     'lambda: self.copy_btn.setText("Kopieren" if self._lang == "Deutsch" else "Copy Text")'),

    # Fix 5 - desc_label static text
    ('        self.desc_label = QLabel(\n            LEVEL_INFO[1]["description"]\n        )',
     '        self.desc_label = QLabel(\n            "Vollst\u00e4ndiger Arztbrief mit allen Notizen und Risikobewertung."\n        )'),

    # Fix 6 - placeholder text in German
    ('    "Select a level and click Generate Letter to preview the discharge letter here."',
     '    "Klicken Sie auf Arztbrief generieren um den Brief zu erstellen."'),
]

for old, new in fixes:
    if old in c:
        c = c.replace(old, new)
        print(f"Fixed: {old[:50]}...")
    else:
        print(f"NOT FOUND: {old[:50]}...")

with open('app/ui/letter_panel.py', 'w', encoding='utf-8') as f:
    f.write(c)

import ast
ast.parse(c)
print(f"\nSyntax OK ({c.count(chr(10))} lines)")
print("Run: python main.py")