# build_privacy_panel.py
import ast

# ── 1. Create privacy_panel.py ───────────────────────────────────
privacy_panel = '''# app/ui/privacy_panel.py
# ================================================================
# NeuraCare — Data & Privacy Panel
# ================================================================
# GDPR compliance tools:
#   - Export anonymised patient data (Article 15)
#   - Export audit log (Article 30)
#   - Export practice summary
#   - Permanent deletion of all data (Article 17)
# ================================================================

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QMessageBox,
)
from PyQt6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

COLORS = {
    "bg_dark":      "#1A1D2E",
    "bg_panel":     "#22253A",
    "bg_card":      "#2A2D42",
    "accent_blue":  "#4A90D9",
    "accent_green": "#52B788",
    "accent_amber": "#F4A50A",
    "accent_red":   "#D64545",
    "text_primary": "#E8EAF6",
    "text_muted":   "#8B90A8",
    "border":       "#35384F",
}


def _divider():
    f = QFrame()
    f.setStyleSheet(f"background-color: {COLORS[\'border\']};")
    f.setFixedHeight(1)
    return f


def _btn(text, color, danger=False):
    bg = COLORS["accent_red"] if danger else color
    btn = QPushButton(text)
    btn.setFixedHeight(38)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {bg};
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            padding: 0 16px;
        }}
        QPushButton:hover {{ background-color: {bg}CC; }}
    """)
    return btn


class PrivacyPanel(QWidget):
    """Data and Privacy panel — GDPR compliance tools."""

    def __init__(self, on_close=None):
        super().__init__()
        self.on_close = on_close
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(
            f"background-color: {COLORS[\'bg_panel\']}; "
            f"border-bottom: 1px solid {COLORS[\'border\']};"
        )
        header.setFixedHeight(56)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Data & Privacy")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(32)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS[\'bg_card\']};
                color: {COLORS[\'text_primary\']};
                border: 1px solid {COLORS[\'border\']};
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{ border-color: {COLORS[\'accent_blue\']}; }}
        """)
        close_btn.clicked.connect(self._on_close)

        hl.addWidget(title, stretch=1)
        hl.addWidget(close_btn)

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {COLORS[\'bg_dark\']}; }}"
        )

        body = QWidget()
        body.setStyleSheet(f"background-color: {COLORS[\'bg_dark\']};")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(32, 24, 32, 32)
        layout.setSpacing(20)

        # Status message
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet(
            f"color: {COLORS[\'accent_green\']}; font-size: 12px;"
        )

        # Section 1 — Export
        layout.addWidget(self._section_title("Export Patient Data"))
        layout.addWidget(_divider())

        exp_info = QLabel(
            "Export data to your Desktop for compliance reporting or "
            "patient access requests (GDPR Article 15)."
        )
        exp_info.setWordWrap(True)
        exp_info.setStyleSheet(
            f"color: {COLORS[\'text_muted\']}; font-size: 11px;"
        )
        layout.addWidget(exp_info)

        row1 = QHBoxLayout()
        anon_btn = _btn("Export Anonymised Data", COLORS["accent_blue"])
        anon_btn.clicked.connect(self._on_export_anon)
        row1.addWidget(anon_btn)

        audit_btn = _btn("Export Audit Log", COLORS["accent_amber"])
        audit_btn.clicked.connect(self._on_export_audit)
        row1.addWidget(audit_btn)

        summary_btn = _btn("Practice Summary", COLORS["accent_green"])
        summary_btn.clicked.connect(self._on_export_summary)
        row1.addWidget(summary_btn)

        layout.addLayout(row1)
        layout.addWidget(self.msg)

        # Section 2 — Danger Zone
        layout.addSpacing(16)
        layout.addWidget(self._section_title("Danger Zone", danger=True))
        layout.addWidget(_divider())

        danger_info = QLabel(
            "Permanently delete all patient records and audit log from "
            "this device. This cannot be undone. Use only when offboarding "
            "a clinic or responding to a GDPR Article 17 erasure request."
        )
        danger_info.setWordWrap(True)
        danger_info.setStyleSheet(
            f"color: {COLORS[\'text_muted\']}; font-size: 11px;"
        )
        layout.addWidget(danger_info)

        delete_btn = _btn(
            "Permanently Delete ALL Patient Data",
            COLORS["accent_red"],
            danger=True
        )
        delete_btn.clicked.connect(self._on_delete_all)
        layout.addWidget(delete_btn)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(header)
        outer.addWidget(scroll, stretch=1)

    def _section_title(self, text, danger=False):
        label = QLabel(text)
        color = COLORS["accent_red"] if danger else COLORS["accent_blue"]
        label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;"
        )
        return label

    def _set_msg(self, text, success=True):
        color = COLORS["accent_green"] if success else COLORS["accent_red"]
        self.msg.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.msg.setText(text)

    def _on_export_anon(self):
        from app.core.csv_export import export_anonymised_patients
        ok, path, err = export_anonymised_patients()
        if ok:
            self._set_msg(f"Saved to Desktop: {Path(path).name}")
        else:
            self._set_msg(err, success=False)

    def _on_export_audit(self):
        from app.core.csv_export import export_audit_log
        ok, path, err = export_audit_log()
        if ok:
            self._set_msg(f"Audit log saved: {Path(path).name}")
        else:
            self._set_msg(err, success=False)

    def _on_export_summary(self):
        from app.core.csv_export import export_practice_summary
        ok, path, err = export_practice_summary()
        if ok:
            self._set_msg(f"Summary saved: {Path(path).name}")
        else:
            self._set_msg(err, success=False)

    def _on_delete_all(self):
        reply = QMessageBox.warning(
            self,
            "Delete All Patient Data",
            "This permanently deletes ALL patient records and the "
            "audit log from this device.\\n\\nThis cannot be undone.\\n\\n"
            "Are you absolutely sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from app.core.csv_export import delete_all_patient_data
        ok, msg = delete_all_patient_data(confirm=True)
        self._set_msg(msg, success=ok)

    def _on_close(self):
        if self.on_close:
            self.on_close()
'''

with open('app/ui/privacy_panel.py', 'w', encoding='utf-8') as f:
    f.write(privacy_panel)
print("1. app/ui/privacy_panel.py created")

# ── 2. Wire into main_window.py ──────────────────────────────────
with open('app/ui/main_window.py', encoding='utf-8-sig') as f:
    mw = f.read()

# Add import
old_imp = "from app.ui.settings_panel  import SettingsPanel"
new_imp = "from app.ui.settings_panel  import SettingsPanel\nfrom app.ui.privacy_panel   import PrivacyPanel"
mw = mw.replace(old_imp, new_imp)
print("2. Import added")

# Add privacy panel to stack
old_stack = '''        # Settings panel (index 4)
        self.settings = SettingsPanel(
            on_close=self._handle_settings_close,
        )
        self.stack.addWidget(self.settings)'''

new_stack = '''        # Settings panel (index 4)
        self.settings = SettingsPanel(
            on_close=self._handle_settings_close,
        )
        self.stack.addWidget(self.settings)

        # Privacy panel (index 5)
        self.privacy = PrivacyPanel(
            on_close=self._handle_privacy_close,
        )
        self.stack.addWidget(self.privacy)'''

mw = mw.replace(old_stack, new_stack)
print("3. Privacy panel added to stack")

# Add show_privacy and handle_privacy_close methods
old_method = "    def _handle_settings_close(self):\n        \"\"\"Go back to welcome on settings close.\"\"\"\n        self.stack.setCurrentWidget(self.welcome)"
new_method = """    def _handle_settings_close(self):
        \"\"\"Go back to welcome on settings close.\"\"\"
        self.stack.setCurrentWidget(self.welcome)

    def show_privacy(self):
        \"\"\"Show privacy panel.\"\"\"
        self.stack.setCurrentWidget(self.privacy)

    def _handle_privacy_close(self):
        \"\"\"Go back to welcome on privacy close.\"\"\"
        self.stack.setCurrentWidget(self.welcome)"""

mw = mw.replace(old_method, new_method)
print("4. show_privacy method added")

# Add Data & Privacy button to RightPanel after Settings button
old_btn = '''        settings_btn.clicked.connect(lambda: self._settings_callback())
        layout.addWidget(settings_btn)
        self._settings_callback = lambda: None'''

new_btn = '''        settings_btn.clicked.connect(lambda: self._settings_callback())
        layout.addWidget(settings_btn)
        self._settings_callback = lambda: None

        privacy_btn = QPushButton("Data & Privacy")
        privacy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_red']}; color: {COLORS['text_primary']}; }}
        """)
        privacy_btn.setFixedHeight(38)
        privacy_btn.clicked.connect(lambda: self._privacy_callback())
        layout.addWidget(privacy_btn)
        self._privacy_callback = lambda: None'''

mw = mw.replace(old_btn, new_btn)
print("5. Data & Privacy button added to right panel")

# Wire callback in _build_main_ui
old_wire = "        self.right_panel._settings_callback  = self._on_settings"
new_wire = """        self.right_panel._settings_callback  = self._on_settings
        self.right_panel._privacy_callback   = self._on_privacy"""
mw = mw.replace(old_wire, new_wire)
print("6. Privacy callback wired")

# Add _on_privacy method
old_on = "    def _on_settings(self):"
new_on = """    def _on_privacy(self):
        \"\"\"Show data and privacy panel.\"\"\"
        self.center.show_privacy()
        self.right_panel.update_patient(None)
        self.status.showMessage("Data & Privacy  |  NeuraCare v1.0")

    def _on_settings(self):"""
mw = mw.replace(old_on, new_on)
print("7. _on_privacy method added")

with open('app/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(mw)

# Verify syntax
for path in ['app/ui/privacy_panel.py', 'app/ui/main_window.py']:
    with open(path, encoding='utf-8') as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"   Syntax OK: {path} ({src.count(chr(10))} lines)")
    except SyntaxError as e:
        print(f"   ERROR {path} line {e.lineno}: {e.msg}")

print("\nDone. Run: python main.py")
print("Right panel: Data & Privacy button below Settings")
print("Opens dedicated panel with export and delete tools")