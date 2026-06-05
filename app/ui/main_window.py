# app/ui/main_window.py
# ================================================================
# NeuraCare — Main Application Window
# PyQt6 desktop UI — 3-panel layout
# ================================================================

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFrame, QSplitter, QStackedWidget, QMessageBox,
    QStatusBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPalette

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.auth    import authenticate, Session
from app.core.audit   import log_login, log_logout
from app.core.patients import get_all_patients, get_patient_count
from app.ui.patient_form import PatientForm
from app.ui.letter_panel import LetterPanel
from app.ui.analytics_panel import AnalyticsPanel
from app.ui.settings_panel  import SettingsPanel
from app.ui.privacy_panel   import PrivacyPanel

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


def make_stylesheet() -> str:
    c = COLORS
    return f"""
        QMainWindow, QWidget {{
            background-color: {c['bg_dark']};
            color: {c['text_primary']};
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }}
        QLabel {{
            color: {c['text_primary']};
            background: transparent;
        }}
        QLabel#muted {{
            color: {c['text_muted']};
            font-size: 11px;
        }}
        QLabel#heading {{
            font-size: 18px;
            font-weight: bold;
            color: {c['text_primary']};
        }}
        QLabel#subheading {{
            font-size: 14px;
            font-weight: bold;
            color: {c['accent_blue']};
        }}
        QPushButton {{
            background-color: {c['accent_blue']};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 18px;
            font-size: 13px;
            font-weight: 500;
        }}
        QPushButton:hover {{ background-color: #5BA3E8; }}
        QPushButton:pressed {{ background-color: #3A7FCC; }}
        QPushButton#secondary {{
            background-color: {c['bg_card']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
        }}
        QPushButton#secondary:hover {{
            background-color: {c['bg_panel']};
            border-color: {c['accent_blue']};
        }}
        QPushButton#danger {{ background-color: {c['accent_red']}; }}
        QPushButton#danger:hover {{ background-color: #E05555; }}
        QPushButton#success {{ background-color: {c['accent_green']}; }}
        QLineEdit {{
            background-color: {c['bg_card']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
        }}
        QLineEdit:focus {{ border-color: {c['accent_blue']}; }}
        QLineEdit::placeholder {{ color: {c['text_muted']}; }}
        QFrame#panel {{
            background-color: {c['bg_panel']};
            border-right: 1px solid {c['border']};
        }}
        QFrame#card {{
            background-color: {c['bg_card']};
            border: 1px solid {c['border']};
            border-radius: 8px;
        }}
        QFrame#divider {{
            background-color: {c['border']};
            max-height: 1px;
        }}
        QStatusBar {{
            background-color: {c['bg_panel']};
            color: {c['text_muted']};
            border-top: 1px solid {c['border']};
            font-size: 11px;
        }}
        QSplitter::handle {{
            background-color: {c['border']};
            width: 1px;
        }}
    """


# ================================================================
# LOGIN SCREEN
# ================================================================

class LoginScreen(QWidget):
    def __init__(self, on_success):
        super().__init__()
        self.on_success = on_success
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("card")
        card.setFixedWidth(380)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(16)

        title = QLabel("NeuraCare")
        title.setObjectName("heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {COLORS['accent_blue']};")

        subtitle = QLabel("Clinical Documentation Assistant")
        subtitle.setObjectName("muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)

        user_label = QLabel("Username")
        user_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setFixedHeight(40)

        pass_label = QLabel("Password")
        pass_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setFixedHeight(40)
        self.password_input.returnPressed.connect(self._attempt_login)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet(f"color: {COLORS['accent_red']}; font-size: 11px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setWordWrap(True)

        self.login_btn = QPushButton("Sign In")
        self.login_btn.setFixedHeight(42)
        self.login_btn.clicked.connect(self._attempt_login)

        privacy = QLabel("🔒  100% offline — no data leaves this device")
        privacy.setObjectName("muted")
        privacy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        privacy.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")

        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(8)
        card_layout.addWidget(divider)
        card_layout.addSpacing(8)
        card_layout.addWidget(user_label)
        card_layout.addWidget(self.username_input)
        card_layout.addWidget(pass_label)
        card_layout.addWidget(self.password_input)
        card_layout.addWidget(self.error_label)
        card_layout.addSpacing(4)
        card_layout.addWidget(self.login_btn)
        card_layout.addSpacing(8)
        card_layout.addWidget(privacy)

        outer.addWidget(card)

    def _attempt_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        self.error_label.setText("")
        self.login_btn.setEnabled(False)
        self.login_btn.setText("Signing in...")
        success, error, user = authenticate(username, password)
        if success:
            Session.login(user)
            log_login(username, user_id=user.get("id"))
            self.on_success()
        else:
            self.error_label.setText(error or "Login failed. Please try again.")
            self.password_input.clear()
            self.password_input.setFocus()
        self.login_btn.setEnabled(True)
        self.login_btn.setText("Sign In")

    def clear(self):
        self.username_input.clear()
        self.password_input.clear()
        self.error_label.setText("")
        self.username_input.setFocus()


# ================================================================
# LEFT PANEL — Patient List
# ================================================================

class PatientListPanel(QFrame):
    def __init__(self, on_select, on_new, on_import=None, on_template=None):
        super().__init__()
        self.setObjectName("panel")
        self.setFixedWidth(250)
        self.on_select   = on_select
        self.on_new      = on_new
        self.on_import   = on_import
        self.on_template = on_template
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet(f"background-color: {COLORS['bg_panel']};")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 16, 14, 12)
        header_layout.setSpacing(8)

        title = QLabel("Patients")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search patients...")
        self.search_input.setFixedHeight(34)
        self.search_input.textChanged.connect(self._on_search)

        new_btn = QPushButton("+ New Patient")
        new_btn.setObjectName("success")
        new_btn.setFixedHeight(34)
        new_btn.clicked.connect(self.on_new)
        new_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_green']};
                color: white;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #63C899; }}
        """)

        # CSV buttons row
        csv_row = QHBoxLayout()
        csv_row.setSpacing(6)

        template_btn = QPushButton("CSV Template")
        template_btn.setFixedHeight(26)
        template_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                font-size: 10px;
                padding: 0 6px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_amber']}; color: {COLORS['text_primary']}; }}
        """)
        if self.on_template:
            template_btn.clicked.connect(self.on_template)

        import_btn = QPushButton("Import CSV")
        import_btn.setFixedHeight(26)
        import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                font-size: 10px;
                padding: 0 6px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_blue']}; color: {COLORS['text_primary']}; }}
        """)
        if self.on_import:
            import_btn.clicked.connect(self.on_import)

        csv_row.addWidget(template_btn)
        csv_row.addWidget(import_btn)

        header_layout.addWidget(title)
        header_layout.addWidget(self.search_input)
        header_layout.addWidget(new_btn)
        header_layout.addLayout(csv_row)

        self.count_label = QLabel("0 patients")
        self.count_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; padding: 4px 14px;"
        )

        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(8, 4, 8, 8)
        self.list_layout.setSpacing(2)
        self.list_layout.addStretch()

        layout.addWidget(header)
        layout.addWidget(self.count_label)
        layout.addWidget(self.list_widget)

    def _on_search(self, text):
        self.refresh(search=text)

    def refresh(self, search: str = ""):
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        patients = get_all_patients()

        if search:
            q = search.lower()
            patients = [p for p in patients
                       if q in p.get("name","").lower()
                       or q in p.get("diagnosis","").lower()]

        self.count_label.setText(
            f"{len(patients)} patient{'s' if len(patients) != 1 else ''}"
        )

        if not patients:
            empty = QLabel("No patients found")
            empty.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; padding: 20px;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.list_layout.insertWidget(0, empty)
            return

        for p in patients:
            btn = self._make_patient_row(p)
            self.list_layout.insertWidget(self.list_layout.count() - 1, btn)

    def _make_patient_row(self, patient: dict) -> QPushButton:
        from app.core.risk import compute_risk
        risk, _ = compute_risk(patient)
        risk_colors = {
            "High":   COLORS["accent_red"],
            "Medium": COLORS["accent_amber"],
            "Low":    COLORS["accent_green"],
        }
        risk_color = risk_colors.get(risk, COLORS["text_muted"])

        name = patient.get("name", "Unknown")
        diag = patient.get("diagnosis", "")
        if len(diag) > 22:
            diag = diag[:22] + "…"

        btn = QPushButton()
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-left: 3px solid {risk_color};
                border-radius: 6px;
                padding: 8px 10px;
                text-align: left;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_panel']};
                border-color: {COLORS['accent_blue']};
                border-left-color: {risk_color};
            }}
        """)

        inner = QVBoxLayout(btn)
        inner.setContentsMargins(4, 2, 4, 2)
        inner.setSpacing(1)

        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: 500; font-size: 12px; background: transparent;")
        diag_label = QLabel(diag)
        diag_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px; background: transparent;")

        inner.addWidget(name_label)
        inner.addWidget(diag_label)

        patient_id = patient.get("id")
        btn.clicked.connect(lambda _, pid=patient_id: self.on_select(pid))
        btn.setFixedHeight(52)
        return btn


# ================================================================
# CENTER PANEL
# ================================================================

class CenterPanel(QWidget):
    def __init__(self, on_form_save=None, on_form_cancel=None, on_letter_pdf=None):
        super().__init__()
        self.on_form_save   = on_form_save
        self.on_form_cancel = on_form_cancel
        self.on_letter_pdf  = on_letter_pdf
        self._build_ui()

    def _build_ui(self):
        self.stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

        self.welcome = self._make_welcome()
        self.stack.addWidget(self.welcome)
        self._detail_widget = None

        self.form = PatientForm(
            on_save=self._handle_form_save,
            on_cancel=self._handle_form_cancel,
        )
        self.stack.addWidget(self.form)

        self.letter_panel = LetterPanel(
            on_pdf=self._handle_letter_pdf,
            on_close=self._handle_letter_close,
        )
        self.stack.addWidget(self.letter_panel)

        self.analytics = AnalyticsPanel(
            on_close=self._handle_analytics_close,
        )
        self.stack.addWidget(self.analytics)

        self.settings = SettingsPanel(
            on_close=self._handle_settings_close,
        )
        self.stack.addWidget(self.settings)

        self.privacy = PrivacyPanel(
            on_close=self._handle_privacy_close,
        )
        self.stack.addWidget(self.privacy)

    def show_form_new(self):
        self.form.clear()
        self.stack.setCurrentWidget(self.form)

    def show_form_edit(self, patient: dict):
        self.form.load_patient(patient)
        self.stack.setCurrentWidget(self.form)

    def _handle_form_save(self, patient_id: int, is_new: bool):
        if self.on_form_save:
            self.on_form_save(patient_id, is_new)

    def _handle_form_cancel(self):
        self.stack.setCurrentWidget(self.welcome)
        if self.on_form_cancel:
            self.on_form_cancel()

    def show_letter(self, patient: dict):
        self.letter_panel.load_patient(patient)
        self.stack.setCurrentWidget(self.letter_panel)

    def _handle_letter_pdf(self, patient, letter_text):
        pass

    def _handle_letter_close(self):
        self.stack.setCurrentWidget(self.welcome)

    def show_analytics(self):
        self.analytics.refresh()
        self.stack.setCurrentWidget(self.analytics)

    def _handle_analytics_close(self):
        self.stack.setCurrentWidget(self.welcome)

    def show_settings(self):
        self.settings.refresh()
        self.stack.setCurrentWidget(self.settings)

    def _handle_settings_close(self):
        self.stack.setCurrentWidget(self.welcome)

    def show_privacy(self):
        self.stack.setCurrentWidget(self.privacy)

    def _handle_privacy_close(self):
        self.stack.setCurrentWidget(self.welcome)

    def _make_welcome(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("🏥")
        icon.setStyleSheet("font-size: 48px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        msg = QLabel("Select a patient from the list\nor create a new record")
        msg.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 14px;")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon)
        layout.addSpacing(12)
        layout.addWidget(msg)
        return w

    def show_welcome(self):
        self.stack.setCurrentWidget(self.welcome)

    def show_patient(self, patient: dict):
        if hasattr(self, '_detail_widget') and self._detail_widget is not None:
            try:
                self.stack.removeWidget(self._detail_widget)
                self._detail_widget.setParent(None)
            except RuntimeError:
                pass
            self._detail_widget = None

        detail = self._make_patient_detail(patient)
        self._detail_widget = detail
        self.stack.addWidget(detail)
        self.stack.setCurrentWidget(detail)

    def _make_patient_detail(self, patient: dict) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        name = QLabel(patient.get("name", "Unknown Patient"))
        name.setStyleSheet("font-size: 20px; font-weight: bold;")

        diag = QLabel(patient.get("diagnosis", ""))
        diag.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)

        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setSpacing(24)

        fields = [
            ("Age",        str(patient.get("age", ""))),
            ("Admitted",   str(patient.get("admission_date", ""))),
            ("Discharged", str(patient.get("discharge_date", ""))),
            ("LOS",        f"{patient.get('length_of_stay', 0)} days"),
            ("ICD-10",     str(patient.get("icd10_code", "") or "—")),
            ("Physician",  str(patient.get("physician_name", "") or "—")),
        ]

        for label, value in fields:
            field_widget = QWidget()
            field_layout = QVBoxLayout(field_widget)
            field_layout.setSpacing(2)
            field_layout.setContentsMargins(0, 0, 0, 0)

            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
            val = QLabel(value)
            val.setStyleSheet("font-size: 13px; font-weight: 500;")

            field_layout.addWidget(lbl)
            field_layout.addWidget(val)
            info_layout.addWidget(field_widget)

        info_layout.addStretch()

        notes_label = QLabel("Clinical Notes")
        notes_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        notes_card = QFrame()
        notes_card.setObjectName("card")
        notes_layout = QVBoxLayout(notes_card)
        notes_text = QLabel(patient.get("notes", "") or "No notes recorded.")
        notes_text.setWordWrap(True)
        notes_text.setStyleSheet("font-size: 13px; line-height: 1.5;")
        notes_layout.addWidget(notes_text)

        layout.addWidget(name)
        layout.addWidget(diag)
        layout.addWidget(divider)
        layout.addWidget(info_widget)
        layout.addWidget(notes_label)
        layout.addWidget(notes_card)

        if patient.get("medication"):
            med_label = QLabel("Medication at Discharge")
            med_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            med_card = QFrame()
            med_card.setObjectName("card")
            med_layout = QVBoxLayout(med_card)
            med_text = QLabel(patient.get("medication", ""))
            med_text.setWordWrap(True)
            med_layout.addWidget(med_text)
            layout.addWidget(med_label)
            layout.addWidget(med_card)

        layout.addStretch()
        return w


# ================================================================
# RIGHT PANEL
# ================================================================

class RightPanel(QFrame):
    def __init__(self, on_generate, on_pdf, on_edit, on_delete):
        super().__init__()
        self.setObjectName("panel")
        self.setFixedWidth(300)
        self.on_generate = on_generate
        self.on_pdf      = on_pdf
        self.on_edit     = on_edit
        self.on_delete   = on_delete
        self._current_patient = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.risk_card = QFrame()
        self.risk_card.setObjectName("card")
        risk_layout = QVBoxLayout(self.risk_card)
        risk_layout.setSpacing(6)

        risk_title = QLabel("Readmission Risk")
        risk_title.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        self.risk_label = QLabel("—")
        self.risk_label.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.risk_reason = QLabel("Select a patient to see risk score")
        self.risk_reason.setWordWrap(True)
        self.risk_reason.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        risk_layout.addWidget(risk_title)
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_reason)

        actions_label = QLabel("Quick Actions")
        actions_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        self.generate_btn = QPushButton("Generate Discharge Letter")
        self.generate_btn.setFixedHeight(38)
        self.generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #5BA3E8; }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.generate_btn.clicked.connect(
            lambda: self.on_generate(self._current_patient)
        )

        self.edit_btn = QPushButton("Edit Patient")
        self.edit_btn.setObjectName("secondary")
        self.edit_btn.setFixedHeight(38)
        self.edit_btn.clicked.connect(
            lambda: self.on_edit(self._current_patient)
        )

        self.delete_btn = QPushButton("Delete Patient")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.setFixedHeight(38)
        self.delete_btn.clicked.connect(
            lambda: self.on_delete(self._current_patient)
        )

        self._set_actions_enabled(False)

        layout.addWidget(self.risk_card)
        layout.addSpacing(4)
        layout.addWidget(actions_label)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.delete_btn)
        layout.addStretch()

        self.session_label = QLabel("")
        self.session_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

        for label, attr, hover_color in [
            ("Practice Analytics", "_analytics_callback", COLORS["accent_blue"]),
            ("Settings",           "_settings_callback",  COLORS["accent_blue"]),
            ("Data & Privacy",     "_privacy_callback",   COLORS["accent_red"]),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(38)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_card']};
                    color: {COLORS['text_muted']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 6px;
                    padding: 6px 16px;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    border-color: {hover_color};
                    color: {COLORS['text_primary']};
                }}
            """)
            setattr(self, attr, lambda: None)
            _attr = attr
            btn.clicked.connect(lambda checked=False, a=_attr: getattr(self, a)())
            layout.addWidget(btn)

        logout_btn = QPushButton("Log Out")
        logout_btn.setObjectName("secondary")
        logout_btn.setFixedHeight(32)
        logout_btn.clicked.connect(self._on_logout)
        layout.addWidget(logout_btn)

        self._logout_callback = None

    def set_logout_callback(self, callback):
        self._logout_callback = callback

    def _on_logout(self):
        if self._logout_callback:
            self._logout_callback()

    def _set_actions_enabled(self, enabled: bool):
        for btn in [self.generate_btn, self.edit_btn, self.delete_btn]:
            btn.setEnabled(enabled)

    def update_patient(self, patient: dict | None):
        self._current_patient = patient

        if patient is None:
            self.risk_label.setText("—")
            self.risk_label.setStyleSheet("font-size: 22px; font-weight: bold;")
            self.risk_reason.setText("Select a patient to see risk score")
            self._set_actions_enabled(False)
            return

        from app.core.risk import compute_risk, get_risk_explanation
        risk, score = compute_risk(patient)
        reasons = get_risk_explanation(patient)

        risk_colors = {
            "High":   COLORS["accent_red"],
            "Medium": COLORS["accent_amber"],
            "Low":    COLORS["accent_green"],
        }
        color = risk_colors.get(risk, COLORS["text_muted"])

        self.risk_label.setText(f"{risk}  (score: {score})")
        self.risk_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        self.risk_reason.setText("\n".join(f"• {r}" for r in reasons[:3]))
        self._set_actions_enabled(True)

    def update_session(self):
        user = Session.get_user()
        if user:
            self.session_label.setText(
                f"Logged in as:\n{user.get('full_name', '')} ({user.get('role', '')})"
            )


# ================================================================
# MAIN WINDOW
# ================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuraCare — Clinical Documentation")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)
        self.setStyleSheet(make_stylesheet())

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_screen = LoginScreen(on_success=self._on_login_success)
        self.main_ui      = self._build_main_ui()

        self.stack.addWidget(self.login_screen)
        self.stack.addWidget(self.main_ui)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("NeuraCare v1.0 — Ready")

        self.stack.setCurrentWidget(self.login_screen)

    def _build_main_ui(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.patient_list = PatientListPanel(
            on_select=self._on_patient_selected,
            on_new=self._on_new_patient,
            on_import=self._on_csv_import,
            on_template=self._on_csv_template,
        )

        self.center = CenterPanel(
            on_form_save=self._on_form_saved,
            on_form_cancel=self._on_form_cancelled,
        )

        self.right_panel = RightPanel(
            on_generate=self._on_generate,
            on_pdf=self._on_pdf,
            on_edit=self._on_edit,
            on_delete=self._on_delete,
        )
        self.right_panel.set_logout_callback(self._on_logout)
        self.right_panel._analytics_callback = self._on_analytics
        self.right_panel._settings_callback  = self._on_settings
        self.right_panel._privacy_callback   = self._on_privacy

        layout.addWidget(self.patient_list)
        layout.addWidget(self.center, stretch=1)
        layout.addWidget(self.right_panel)

        return container

    def _on_login_success(self):
        self.right_panel.update_session()
        self.patient_list.refresh()
        self.center.show_welcome()
        self.right_panel.update_patient(None)
        self.stack.setCurrentWidget(self.main_ui)
        user = Session.get_user()
        name = user.get("full_name", "") if user else ""
        self.status.showMessage(f"Welcome, {name}  |  NeuraCare v1.0")

    def _on_logout(self):
        reply = QMessageBox.question(
            self, "Log Out", "Are you sure you want to log out?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            username = Session.get_username()
            user_id  = Session.get_user().get("id") if Session.get_user() else None
            log_logout(username, user_id=user_id)
            Session.logout()
            self.login_screen.clear()
            self.stack.setCurrentWidget(self.login_screen)
            self.status.showMessage("Logged out  |  NeuraCare v1.0")

    def _on_patient_selected(self, patient_id: int):
        from app.core.patients import get_patient
        from app.core.audit    import log_patient_view
        patient = get_patient(patient_id)
        if patient:
            log_patient_view(
                Session.get_username(),
                patient_id=patient_id,
                patient_name=patient.get("name", ""),
                user_id=Session.get_user().get("id") if Session.get_user() else None,
            )
            self.center.show_patient(patient)
            self.right_panel.update_patient(patient)
            self.status.showMessage(f"Viewing: {patient.get('name', '')}  |  NeuraCare v1.0")

    def _on_privacy(self):
        self.center.show_privacy()
        self.right_panel.update_patient(None)
        self.status.showMessage("Data & Privacy  |  NeuraCare v1.0")

    def _on_settings(self):
        self.center.show_settings()
        self.right_panel.update_patient(None)
        self.status.showMessage("Settings  |  NeuraCare v1.0")

    def _on_analytics(self):
        self.center.show_analytics()
        self.right_panel.update_patient(None)
        self.status.showMessage("Practice Analytics  |  NeuraCare v1.0")

    def _on_csv_template(self):
        from PyQt6.QtWidgets import QMessageBox
        from app.core.csv_import import create_template
        path = create_template()
        QMessageBox.information(
            self, "Template Saved",
            f"CSV template saved to Desktop:\n{path}\n\n"
            "Fill it in and use Import CSV to add patients."
        )

    def _on_csv_import(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from app.core.csv_import import import_patients_from_csv
        ok_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if not ok_path:
            return
        ok, imported, skipped, errors = import_patients_from_csv(
            ok_path, created_by=Session.get_username() or "import"
        )
        if ok:
            msg = f"Imported {imported} patient(s)."
            if skipped:
                msg += f"\nSkipped {skipped} duplicates."
            QMessageBox.information(self, "Import Complete", msg)
            self.patient_list.refresh()
        else:
            QMessageBox.warning(self, "Import Failed", "\n".join(errors[:3]))

    def _on_new_patient(self):
        self.center.show_form_new()
        self.right_panel.update_patient(None)
        self.status.showMessage("New patient  |  NeuraCare v1.0")

    def _on_form_saved(self, patient_id: int, is_new: bool):
        from app.core.patients import get_patient
        patient = get_patient(patient_id)
        self.patient_list.refresh()
        if patient:
            self.center.show_patient(patient)
            self.right_panel.update_patient(patient)
        action = "Created" if is_new else "Updated"
        name = patient.get("name", "") if patient else ""
        self.status.showMessage(f"{action}: {name}  |  NeuraCare v1.0")

    def _on_form_cancelled(self):
        self.status.showMessage("Cancelled  |  NeuraCare v1.0")

    def _on_generate(self, patient):
        if patient:
            self.center.show_letter(patient)
            self.status.showMessage(
                f"Discharge letter — {patient.get('name','')}  |  NeuraCare v1.0"
            )

    def _on_pdf(self, patient, file_path: str = ""):
        if patient:
            if file_path:
                self.status.showMessage(f"PDF saved: {file_path}  |  NeuraCare v1.0")
            else:
                self.status.showMessage(f"PDF exported for {patient.get('name','')}  |  NeuraCare v1.0")

    def _on_edit(self, patient):
        if patient:
            self.center.show_form_edit(patient)
            self.status.showMessage(f"Editing: {patient.get('name','')}  |  NeuraCare v1.0")

    def _on_delete(self, patient):
        if not patient:
            return
        reply = QMessageBox.question(
            self, "Delete Patient",
            f"Delete {patient.get('name','')}?\n\nThis can be undone from the admin panel.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            from app.core.patients import delete_patient
            from app.core.audit    import log_patient_delete
            ok, err = delete_patient(patient["id"])
            if ok:
                log_patient_delete(
                    Session.get_username(),
                    patient_id=patient["id"],
                    patient_name=patient.get("name", ""),
                    user_id=Session.get_user().get("id") if Session.get_user() else None,
                )
                self.patient_list.refresh()
                self.center.show_welcome()
                self.right_panel.update_patient(None)
                self.status.showMessage(f"Deleted: {patient.get('name','')}  |  NeuraCare v1.0")
            else:
                QMessageBox.warning(self, "Error", f"Could not delete: {err}")


# ================================================================
# ENTRY POINT
# ================================================================

def run():
    app = QApplication(sys.argv)
    app.setApplicationName("NeuraCare")
    app.setApplicationVersion("1.0.0")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()