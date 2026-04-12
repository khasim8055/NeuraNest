# app/ui/main_window.py
# ================================================================
# NeuraCare — Main Application Window
# PyQt6 desktop UI — 3-panel layout
# ================================================================
# Layout:
#   Left panel  (250px) — patient list + search
#   Center panel (flex) — patient detail / form
#   Right panel (300px) — risk score + quick actions
#
# Screens:
#   1. Login screen   — shown on startup
#   2. Main window    — shown after successful login
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.auth    import authenticate, Session
from app.core.audit   import log_login, log_logout
from app.core.patients import get_all_patients, get_patient_count
from app.ui.patient_form import PatientForm


# ================================================================
# COLOURS — NeuraCare brand palette
# ================================================================
COLORS = {
    "bg_dark":      "#1A1D2E",   # main background
    "bg_panel":     "#22253A",   # panel background
    "bg_card":      "#2A2D42",   # card / input background
    "accent_blue":  "#4A90D9",   # primary action blue
    "accent_green": "#52B788",   # success / low risk
    "accent_amber": "#F4A50A",   # warning / medium risk
    "accent_red":   "#D64545",   # danger / high risk
    "text_primary": "#E8EAF6",   # main text
    "text_muted":   "#8B90A8",   # secondary text
    "border":       "#35384F",   # subtle border
}


# ================================================================
# STYLE HELPERS
# ================================================================

def make_stylesheet() -> str:
    """Central stylesheet for the entire application."""
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

        QPushButton:hover {{
            background-color: #5BA3E8;
        }}

        QPushButton:pressed {{
            background-color: #3A7FCC;
        }}

        QPushButton#secondary {{
            background-color: {c['bg_card']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
        }}

        QPushButton#secondary:hover {{
            background-color: {c['bg_panel']};
            border-color: {c['accent_blue']};
        }}

        QPushButton#danger {{
            background-color: {c['accent_red']};
        }}

        QPushButton#danger:hover {{
            background-color: #E05555;
        }}

        QPushButton#success {{
            background-color: {c['accent_green']};
        }}

        QLineEdit {{
            background-color: {c['bg_card']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
        }}

        QLineEdit:focus {{
            border-color: {c['accent_blue']};
        }}

        QLineEdit::placeholder {{
            color: {c['text_muted']};
        }}

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
    """
    Full-screen login form shown on startup.
    Emits login_success signal when credentials are verified.
    """

    def __init__(self, on_success):
        super().__init__()
        self.on_success = on_success
        self._build_ui()

    def _build_ui(self):
        # Outer layout — centres the card
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.setContentsMargins(0, 0, 0, 0)

        # Login card
        card = QFrame()
        card.setObjectName("card")
        card.setFixedWidth(380)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(16)

        # Logo / title
        title = QLabel("NeuraCare")
        title.setObjectName("heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; "
                           f"color: {COLORS['accent_blue']};")

        subtitle = QLabel("Clinical Documentation Assistant")
        subtitle.setObjectName("muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)

        # Username field
        user_label = QLabel("Username")
        user_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setFixedHeight(40)

        # Password field
        pass_label = QLabel("Password")
        pass_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setFixedHeight(40)
        self.password_input.returnPressed.connect(self._attempt_login)

        # Error label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet(f"color: {COLORS['accent_red']}; font-size: 11px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setWordWrap(True)

        # Login button
        self.login_btn = QPushButton("Sign In")
        self.login_btn.setFixedHeight(42)
        self.login_btn.clicked.connect(self._attempt_login)

        # Privacy note
        privacy = QLabel("🔒  100% offline — no data leaves this device")
        privacy.setObjectName("muted")
        privacy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        privacy.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")

        # Assemble card
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
        """Try to authenticate with entered credentials."""
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
        """Clear fields — called when logging out."""
        self.username_input.clear()
        self.password_input.clear()
        self.error_label.setText("")
        self.username_input.setFocus()


# ================================================================
# LEFT PANEL — Patient List
# ================================================================

class PatientListPanel(QFrame):
    """
    Left panel showing patient list and search.
    250px wide, scrollable list of patient names.
    """

    def __init__(self, on_select, on_new):
        super().__init__()
        self.setObjectName("panel")
        self.setFixedWidth(250)
        self.on_select = on_select
        self.on_new    = on_new
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"background-color: {COLORS['bg_panel']};")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 16, 14, 12)
        header_layout.setSpacing(10)

        title = QLabel("Patients")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search patients...")
        self.search_input.setFixedHeight(34)
        self.search_input.textChanged.connect(self._on_search)

        # New patient button
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

        header_layout.addWidget(title)
        header_layout.addWidget(self.search_input)
        header_layout.addWidget(new_btn)

        # Patient count label
        self.count_label = QLabel("0 patients")
        self.count_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; "
            f"padding: 4px 14px;"
        )

        # Patient list area
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(8, 4, 8, 8)
        self.list_layout.setSpacing(2)
        self.list_layout.addStretch()

        layout.addWidget(header)
        layout.addWidget(self.count_label)
        layout.addWidget(self.list_widget)

    def _on_search(self, text):
        """Filter patient list as user types."""
        self.refresh(search=text)

    def refresh(self, search: str = ""):
        """Reload patient list from database."""
        # Clear existing items
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        patients = get_all_patients()

        # Filter by search
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
            empty.setStyleSheet(f"color: {COLORS['text_muted']}; "
                               f"font-size: 11px; padding: 20px;")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.list_layout.insertWidget(0, empty)
            return

        for p in patients:
            btn = self._make_patient_row(p)
            self.list_layout.insertWidget(
                self.list_layout.count() - 1, btn
            )

    def _make_patient_row(self, patient: dict) -> QPushButton:
        """Create one patient row button."""
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
# CENTER PANEL — Patient Detail (placeholder)
# ================================================================

class CenterPanel(QWidget):
    """
    Center panel — shows patient detail, form, or welcome screen.
    """

    def __init__(self, on_form_save=None, on_form_cancel=None):
        super().__init__()
        self.on_form_save   = on_form_save
        self.on_form_cancel = on_form_cancel
        self._build_ui()

    def _build_ui(self):
        self.stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

        # Welcome screen (index 0)
        self.welcome = self._make_welcome()
        self.stack.addWidget(self.welcome)

        # Patient form (index 1) — reused for add and edit
        self.form = PatientForm(
            on_save=self._handle_form_save,
            on_cancel=self._handle_form_cancel,
        )
        self.stack.addWidget(self.form)

    def show_form_new(self):
        """Show empty form for adding a new patient."""
        self.form.clear()
        self.stack.setCurrentWidget(self.form)

    def show_form_edit(self, patient: dict):
        """Show form pre-filled for editing a patient."""
        self.form.load_patient(patient)
        self.stack.setCurrentWidget(self.form)

    def _handle_form_save(self, patient_id: int, is_new: bool):
        if self.on_form_save:
            self.on_form_save(patient_id, is_new)

    def _handle_form_cancel(self):
        self.stack.setCurrentWidget(self.welcome)
        if self.on_form_cancel:
            self.on_form_cancel()

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
        """Show patient detail — full form built in Day 6."""
        # Remove old detail widget if exists
        if self.stack.count() > 1:
            old = self.stack.widget(1)
            self.stack.removeWidget(old)
            old.deleteLater()

        detail = self._make_patient_detail(patient)
        self.stack.addWidget(detail)
        self.stack.setCurrentWidget(detail)

    def _make_patient_detail(self, patient: dict) -> QWidget:
        """Placeholder patient detail — full version in Day 6."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        name = QLabel(patient.get("name", "Unknown Patient"))
        name.setStyleSheet("font-size: 20px; font-weight: bold;")

        diag = QLabel(patient.get("diagnosis", ""))
        diag.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)

        # Quick info grid
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setSpacing(24)

        fields = [
            ("Age",            str(patient.get("age", ""))),
            ("Admitted",       str(patient.get("admission_date", ""))),
            ("Discharged",     str(patient.get("discharge_date", ""))),
            ("LOS",            f"{patient.get('length_of_stay', 0)} days"),
            ("ICD-10",         str(patient.get("icd10_code", "") or "—")),
            ("Physician",      str(patient.get("physician_name", "") or "—")),
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

        # Notes
        notes_label = QLabel("Clinical Notes")
        notes_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        notes_card = QFrame()
        notes_card.setObjectName("card")
        notes_layout = QVBoxLayout(notes_card)
        notes_text = QLabel(patient.get("notes", "") or "No notes recorded.")
        notes_text.setWordWrap(True)
        notes_text.setStyleSheet("font-size: 13px; line-height: 1.5;")
        notes_layout.addWidget(notes_text)

        # Medication
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

        layout.addWidget(name)
        layout.addWidget(diag)
        layout.addWidget(divider)
        layout.addWidget(info_widget)
        layout.addWidget(notes_label)
        layout.addWidget(notes_card)
        layout.addStretch()
        return w


# ================================================================
# RIGHT PANEL — Risk Score + Quick Actions
# ================================================================

class RightPanel(QFrame):
    """
    Right panel — risk score, quick actions, follow-up info.
    300px wide.
    """

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

        # Risk score card
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

        # Quick actions
        actions_label = QLabel("Quick Actions")
        actions_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")

        self.generate_btn = QPushButton("Generate Discharge Letter")
        self.generate_btn.setFixedHeight(38)
        self.generate_btn.clicked.connect(
            lambda: self.on_generate(self._current_patient)
        )

        self.pdf_btn = QPushButton("Export PDF")
        self.pdf_btn.setObjectName("secondary")
        self.pdf_btn.setFixedHeight(38)
        self.pdf_btn.clicked.connect(
            lambda: self.on_pdf(self._current_patient)
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

        # Disable all until patient selected
        self._set_actions_enabled(False)

        layout.addWidget(self.risk_card)
        layout.addSpacing(4)
        layout.addWidget(actions_label)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.pdf_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.delete_btn)
        layout.addStretch()

        # Session info at bottom
        self.session_label = QLabel("")
        self.session_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px;"
        )
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

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
        for btn in [self.generate_btn, self.pdf_btn,
                    self.edit_btn, self.delete_btn]:
            btn.setEnabled(enabled)

    def update_patient(self, patient: dict | None):
        """Update risk score and actions for selected patient."""
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
        self.risk_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color};"
        )
        self.risk_reason.setText("\n".join(f"• {r}" for r in reasons[:3]))
        self._set_actions_enabled(True)

    def update_session(self):
        """Show current user info."""
        user = Session.get_user()
        if user:
            self.session_label.setText(
                f"Logged in as:\n{user.get('full_name', '')} "
                f"({user.get('role', '')})"
            )


# ================================================================
# MAIN WINDOW
# ================================================================

class MainWindow(QMainWindow):
    """
    The main NeuraCare application window.
    Manages login screen ↔ main UI switching.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuraCare — Clinical Documentation")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # Apply stylesheet
        self.setStyleSheet(make_stylesheet())

        # Stack: login (0) and main UI (1)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Build screens
        self.login_screen = LoginScreen(on_success=self._on_login_success)
        self.main_ui      = self._build_main_ui()

        self.stack.addWidget(self.login_screen)
        self.stack.addWidget(self.main_ui)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("NeuraCare v1.0 — Ready")

        # Show login first
        self.stack.setCurrentWidget(self.login_screen)

    def _build_main_ui(self) -> QWidget:
        """Build the 3-panel main interface."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left panel
        self.patient_list = PatientListPanel(
            on_select=self._on_patient_selected,
            on_new=self._on_new_patient,
        )

        # Center panel
        self.center = CenterPanel(
            on_form_save=self._on_form_saved,
            on_form_cancel=self._on_form_cancelled,
        )

        # Right panel
        self.right_panel = RightPanel(
            on_generate=self._on_generate,
            on_pdf=self._on_pdf,
            on_edit=self._on_edit,
            on_delete=self._on_delete,
        )
        self.right_panel.set_logout_callback(self._on_logout)

        layout.addWidget(self.patient_list)
        layout.addWidget(self.center, stretch=1)
        layout.addWidget(self.right_panel)

        return container

    def _on_login_success(self):
        """Called when login succeeds — switch to main UI."""
        self.right_panel.update_session()
        self.patient_list.refresh()
        self.center.show_welcome()
        self.right_panel.update_patient(None)
        self.stack.setCurrentWidget(self.main_ui)

        user = Session.get_user()
        name = user.get("full_name", "") if user else ""
        self.status.showMessage(f"Welcome, {name}  |  NeuraCare v1.0")

    def _on_logout(self):
        """Log out — return to login screen."""
        reply = QMessageBox.question(
            self, "Log Out",
            "Are you sure you want to log out?",
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
        """Called when a patient row is clicked."""
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
            self.status.showMessage(
                f"Viewing: {patient.get('name', '')}  |  NeuraCare v1.0"
            )

    def _on_new_patient(self):
        """Called when New Patient button is clicked — show add form."""
        self.center.show_form_new()
        self.right_panel.update_patient(None)
        self.status.showMessage("New patient  |  NeuraCare v1.0")

    def _on_form_saved(self, patient_id: int, is_new: bool):
        """Called after form saves successfully."""
        from app.core.patients import get_patient
        patient = get_patient(patient_id)
        self.patient_list.refresh()
        if patient:
            self.center.show_patient(patient)
            self.right_panel.update_patient(patient)
        action = "Created" if is_new else "Updated"
        name = patient.get("name", "") if patient else ""
        self.status.showMessage(
            f"{action}: {name}  |  NeuraCare v1.0"
        )

    def _on_form_cancelled(self):
        """Called when form cancel is clicked."""
        self.status.showMessage("Cancelled  |  NeuraCare v1.0")

    def _on_generate(self, patient):
        if patient:
            QMessageBox.information(
                self, "Generate Letter",
                f"Discharge letter generation for {patient.get('name','')} "
                f"— coming in Day 8."
            )

    def _on_pdf(self, patient):
        if patient:
            QMessageBox.information(
                self, "Export PDF",
                f"PDF export for {patient.get('name','')} — coming in Day 8."
            )

    def _on_edit(self, patient):
        """Show edit form for selected patient."""
        if patient:
            self.center.show_form_edit(patient)
            self.status.showMessage(
                f"Editing: {patient.get('name','')}  |  NeuraCare v1.0"
            )

    def _on_delete(self, patient):
        if not patient:
            return
        reply = QMessageBox.question(
            self, "Delete Patient",
            f"Delete {patient.get('name','')}?\n\n"
            f"This can be undone from the admin panel.",
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
                self.status.showMessage(
                    f"Deleted: {patient.get('name','')}  |  NeuraCare v1.0"
                )
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
