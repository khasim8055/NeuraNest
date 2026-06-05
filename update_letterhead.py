# update_letterhead.py — run once to update pdf_exporter and settings_panel
import re

# ── 1. Update pdf_exporter.py ─────────────────────────────────────
with open('app/core/pdf_exporter.py', encoding='utf-8') as f:
    pdf = f.read()

# Add get_clinic_settings function after imports
clinic_fn = '''

def get_clinic_settings() -> dict:
    """Read clinic letterhead from app_config."""
    defaults = {
        "clinic_name": "NeuraCare Klinik",
        "clinic_doctor": "",
        "clinic_street": "",
        "clinic_city": "",
        "clinic_phone": "",
        "clinic_email": "",
        "clinic_bsnr": "",
        "clinic_lanr": "",
    }
    try:
        import sqlite3, os
        from pathlib import Path as _Path
        base = _Path(os.environ.get("NEURACARE_BASE_DIR", ""))
        if not base or not base.exists():
            base = _Path(__file__).parent.parent.parent
        db = base / "app" / "data" / "neuranest.db"
        if not db.exists():
            return defaults
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT key,value FROM app_config WHERE key LIKE 'clinic_%'"
        ).fetchall()
        conn.close()
        for key, val in rows:
            if key in defaults and val:
                defaults[key] = val
    except Exception:
        pass
    return defaults

'''

if 'get_clinic_settings' not in pdf:
    pdf = pdf.replace('def _make_styles()', clinic_fn + 'def _make_styles()')
    print("✓ get_clinic_settings added to pdf_exporter.py")
else:
    print("  get_clinic_settings already exists")

# Update the header section to use clinic settings
old_header = '''        story.append(Paragraph(clinic_name, styles["clinic_name"]))

        if lang == "Deutsch":
            sub = "Vertraulich · Arztbrief · Datenschutzkonform"
        else:
            sub = "Confidential · Discharge Letter · GDPR Compliant"
        story.append(Paragraph(sub, styles["clinic_sub"]))'''

new_header = '''        # Load clinic settings from database
        cs = get_clinic_settings()
        if clinic_name == "NeuraCare Klinik":
            clinic_name = cs["clinic_name"]

        story.append(Paragraph(clinic_name, styles["clinic_name"]))

        # Build address line from settings
        addr_parts = []
        if cs.get("clinic_doctor"): addr_parts.append(cs["clinic_doctor"])
        if cs.get("clinic_street"): addr_parts.append(cs["clinic_street"])
        if cs.get("clinic_city"):   addr_parts.append(cs["clinic_city"])
        if addr_parts:
            story.append(Paragraph("  ·  ".join(addr_parts), styles["clinic_sub"]))

        contact_parts = []
        if cs.get("clinic_phone"): contact_parts.append("Tel: " + cs["clinic_phone"])
        if cs.get("clinic_email"): contact_parts.append(cs["clinic_email"])
        if cs.get("clinic_bsnr"):  contact_parts.append("BSNR: " + cs["clinic_bsnr"])
        if contact_parts:
            story.append(Paragraph("  ·  ".join(contact_parts), styles["clinic_sub"]))

        if lang == "Deutsch":
            sub = "Vertraulich · Arztbrief · Datenschutzkonform"
        else:
            sub = "Confidential · Discharge Letter · GDPR Compliant"
        story.append(Paragraph(sub, styles["clinic_sub"]))'''

if old_header in pdf:
    pdf = pdf.replace(old_header, new_header)
    print("✓ PDF header updated with clinic settings")
else:
    print("  PDF header pattern not found — check manually")

with open('app/core/pdf_exporter.py', 'w', encoding='utf-8') as f:
    f.write(pdf)

# ── 2. Add letterhead form to settings_panel.py ──────────────────
with open('app/ui/settings_panel.py', encoding='utf-8') as f:
    settings = f.read()

letterhead_class = '''

class LetterheadSection(QWidget):
    """Clinic letterhead settings — shown in settings panel."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(FIELD_STYLE)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(_make_section("Clinic Letterhead"))
        layout.addWidget(_make_divider())

        info = QLabel(
            "These details appear on every generated PDF letterhead."
        )
        info.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        fields = [
            ("clinic_name",   "Practice / Clinic Name *",  "e.g. Praxis Dr. Mueller"),
            ("clinic_doctor", "Lead Physician",             "e.g. Dr. Anna Mueller"),
            ("clinic_street", "Street Address",             "e.g. Hauptstraße 12"),
            ("clinic_city",   "Postcode and City",          "e.g. 69115 Heidelberg"),
            ("clinic_phone",  "Phone Number",               "e.g. +49 6221 123456"),
            ("clinic_email",  "Email Address",              "e.g. praxis@mueller.de"),
            ("clinic_bsnr",   "BSNR (Betriebsstättennr.)", "9-digit practice number"),
            ("clinic_lanr",   "LANR (Arztnummer)",          "9-digit doctor number"),
        ]

        self._inputs = {}
        for key, label, placeholder in fields:
            layout.addWidget(_make_label(label))
            inp = QLineEdit()
            inp.setPlaceholderText(placeholder)
            inp.setFixedHeight(36)
            self._inputs[key] = inp
            layout.addWidget(inp)

        self.msg_label = QLabel("")
        self.msg_label.setStyleSheet(
            f"color: {COLORS['accent_red']}; font-size: 11px;"
        )
        layout.addWidget(self.msg_label)

        save_btn = _btn("Save Letterhead", COLORS["accent_green"])
        save_btn.clicked.connect(self._on_save)
        layout.addWidget(save_btn)

        self._load_current()

    def _load_current(self):
        """Load existing values from database."""
        try:
            import sqlite3, os
            from pathlib import Path
            base = Path(os.environ.get("NEURACARE_BASE_DIR", ""))
            if not base or not base.exists():
                base = Path(__file__).parent.parent.parent
            db = base / "app" / "data" / "neuranest.db"
            if not db.exists():
                return
            conn = sqlite3.connect(str(db))
            rows = conn.execute(
                "SELECT key, value FROM app_config WHERE key LIKE 'clinic_%'"
            ).fetchall()
            conn.close()
            for key, val in rows:
                if key in self._inputs and val:
                    self._inputs[key].setText(val)
        except Exception:
            pass

    def _on_save(self):
        """Save letterhead settings to database."""
        self.msg_label.setStyleSheet(
            f"color: {COLORS['accent_red']}; font-size: 11px;"
        )

        if not self._inputs["clinic_name"].text().strip():
            self.msg_label.setText("Practice name is required.")
            return

        try:
            import sqlite3, os
            from pathlib import Path
            base = Path(os.environ.get("NEURACARE_BASE_DIR", ""))
            if not base or not base.exists():
                base = Path(__file__).parent.parent.parent
            db = base / "app" / "data" / "neuranest.db"
            conn = sqlite3.connect(str(db))
            for key, inp in self._inputs.items():
                conn.execute(
                    "UPDATE app_config SET value=? WHERE key=?",
                    (inp.text().strip(), key)
                )
            conn.commit()
            conn.close()
            self.msg_label.setStyleSheet(
                f"color: {COLORS['accent_green']}; font-size: 11px;"
            )
            self.msg_label.setText("Letterhead saved. Next PDF will use these details.")
        except Exception as e:
            self.msg_label.setText(f"Error: {str(e)}")

'''

if 'LetterheadSection' not in settings:
    settings = settings.replace(
        'class ChangePasswordSection',
        letterhead_class + 'class ChangePasswordSection'
    )
    print("✓ LetterheadSection added to settings_panel.py")
else:
    print("  LetterheadSection already exists")

# Add LetterheadSection to SettingsPanel.refresh()
old_refresh = '''        # Change password — available to all users
        self.body_layout.addWidget(ChangePasswordSection())'''
new_refresh = '''        # Letterhead settings — available to all users
        self.body_layout.addWidget(LetterheadSection())
        self.body_layout.addSpacing(8)

        # Change password — available to all users
        self.body_layout.addWidget(ChangePasswordSection())'''

if old_refresh in settings:
    settings = settings.replace(old_refresh, new_refresh)
    print("✓ LetterheadSection added to settings panel refresh")
else:
    print("  refresh pattern not found")

with open('app/ui/settings_panel.py', 'w', encoding='utf-8') as f:
    f.write(settings)

# ── 3. Verify syntax ─────────────────────────────────────────────
import ast
for path in ['app/core/pdf_exporter.py', 'app/ui/settings_panel.py']:
    with open(path, encoding='utf-8') as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"✓ {path} syntax OK ({src.count(chr(10))} lines)")
    except SyntaxError as e:
        print(f"✗ {path} ERROR line {e.lineno}: {e.msg}")

print("\nDone. Run: python main.py")
print("Go to Settings → you will see the Letterhead form at the top.")
print("Fill in your clinic details → Save → generate a PDF to see the result.")