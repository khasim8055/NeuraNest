# app/ui/patient_form.py
# ================================================================
# NeuraCare — Patient Add / Edit Form
# ================================================================
# Used for both adding new patients and editing existing ones.
# All fields from the patients table are included.
# Validates before saving and shows inline error messages.
# ================================================================

import sys
from pathlib import Path
from datetime import date

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFrame,
    QScrollArea, QMessageBox, QDateEdit, QSpinBox,
    QComboBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtGui import QFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.patients import create_patient, update_patient, validate_patient
from app.core.auth     import Session
from app.core.audit    import log_patient_create, log_patient_edit

# ── Colours (imported from main_window to stay consistent) ───────
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

FIELD_STYLE = f"""
    QLineEdit, QTextEdit, QSpinBox, QDateEdit, QComboBox {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
    }}
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus,
    QDateEdit:focus, QComboBox:focus {{
        border-color: {COLORS['accent_blue']};
    }}
    QSpinBox::up-button, QSpinBox::down-button {{
        background-color: {COLORS['bg_panel']};
        border: none;
        width: 20px;
    }}
    QDateEdit::drop-down {{
        background-color: {COLORS['bg_panel']};
        border: none;
        width: 24px;
    }}
    QComboBox::drop-down {{
        background-color: {COLORS['bg_panel']};
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent_blue']};
    }}
"""


# ================================================================
# FORM FIELD HELPERS
# ================================================================

def make_label(text: str, required: bool = False) -> QLabel:
    """Create a form field label."""
    suffix = " *" if required else ""
    label = QLabel(f"{text}{suffix}")
    label.setStyleSheet(
        f"color: {COLORS['text_muted']}; font-size: 11px; "
        f"font-weight: 500; margin-bottom: 2px;"
    )
    return label


def make_error_label() -> QLabel:
    """Create an inline error label (hidden by default)."""
    label = QLabel("")
    label.setStyleSheet(
        f"color: {COLORS['accent_red']}; font-size: 10px;"
    )
    label.hide()
    return label


def make_section_header(text: str) -> QLabel:
    """Create a section divider header."""
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {COLORS['accent_blue']}; font-size: 12px; "
        f"font-weight: bold; padding: 8px 0 4px 0;"
    )
    return label


def make_divider() -> QFrame:
    frame = QFrame()
    frame.setStyleSheet(f"background-color: {COLORS['border']};")
    frame.setFixedHeight(1)
    return frame


# ================================================================
# PATIENT FORM WIDGET
# ================================================================

class PatientForm(QWidget):
    """
    Complete add/edit patient form.

    Usage:
        form = PatientForm(on_save=callback, on_cancel=callback)
        form.load_patient(patient_dict)  # for editing
        form.clear()                     # for new patient
    """

    def __init__(self, on_save, on_cancel):
        super().__init__()
        self.on_save   = on_save
        self.on_cancel = on_cancel
        self._patient_id = None  # None = new, int = editing
        self.setStyleSheet(FIELD_STYLE)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        header.setFixedHeight(56)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        self.title_label = QLabel("New Patient")
        self.title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        self.error_banner = QLabel("")
        self.error_banner.setStyleSheet(
            f"color: {COLORS['accent_red']}; font-size: 12px;"
        )
        self.error_banner.hide()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                border-color: {COLORS['accent_blue']};
            }}
        """)
        cancel_btn.setFixedHeight(32)
        cancel_btn.clicked.connect(self.on_cancel)

        self.save_btn = QPushButton("Save Patient")
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_green']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 20px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #63C899; }}
            QPushButton:disabled {{ background-color: {COLORS['bg_card']}; color: {COLORS['text_muted']}; }}
        """)
        self.save_btn.setFixedHeight(32)
        self.save_btn.clicked.connect(self._on_save)

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.error_banner, stretch=1)
        header_layout.addWidget(cancel_btn)
        header_layout.addSpacing(8)
        header_layout.addWidget(self.save_btn)

        # ── Scrollable form body ──────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {COLORS['bg_dark']}; }}"
        )

        body = QWidget()
        body.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        form_layout = QVBoxLayout(body)
        form_layout.setContentsMargins(32, 24, 32, 32)
        form_layout.setSpacing(8)

        # ── Section 1: Patient Identity ───────────────────────────
        form_layout.addWidget(make_section_header("Patient Identity"))
        form_layout.addWidget(make_divider())

        # Name
        form_layout.addWidget(make_label("Full Name", required=True))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Maria Hoffmann")
        self.name_input.setFixedHeight(38)
        self.name_error = make_error_label()
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(self.name_error)

        # Age + DOB row
        age_dob_row = QHBoxLayout()
        age_dob_row.setSpacing(16)

        age_col = QVBoxLayout()
        age_col.setSpacing(4)
        age_col.addWidget(make_label("Age", required=True))
        self.age_input = QSpinBox()
        self.age_input.setRange(0, 130)
        self.age_input.setValue(0)
        self.age_input.setFixedHeight(38)
        self.age_error = make_error_label()
        age_col.addWidget(self.age_input)
        age_col.addWidget(self.age_error)

        dob_col = QVBoxLayout()
        dob_col.setSpacing(4)
        dob_col.addWidget(make_label("Date of Birth (optional)"))
        self.dob_input = QLineEdit()
        self.dob_input.setPlaceholderText("YYYY-MM-DD")
        self.dob_input.setFixedHeight(38)
        dob_col.addWidget(self.dob_input)
        dob_col.addWidget(make_error_label())

        age_dob_row.addLayout(age_col)
        age_dob_row.addLayout(dob_col)
        form_layout.addLayout(age_dob_row)

        # ── Section 2: Clinical Information ──────────────────────
        form_layout.addSpacing(8)
        form_layout.addWidget(make_section_header("Clinical Information"))
        form_layout.addWidget(make_divider())

        # Diagnosis
        form_layout.addWidget(make_label("Primary Diagnosis", required=True))
        self.diagnosis_input = QLineEdit()
        self.diagnosis_input.setPlaceholderText(
            "e.g. Chronic Heart Failure"
        )
        self.diagnosis_input.setFixedHeight(38)
        self.diagnosis_error = make_error_label()
        form_layout.addWidget(self.diagnosis_input)
        form_layout.addWidget(self.diagnosis_error)

        # ICD-10 + Physician row
        icd_phy_row = QHBoxLayout()
        icd_phy_row.setSpacing(16)

        icd_col = QVBoxLayout()
        icd_col.setSpacing(4)
        icd_col.addWidget(make_label("ICD-10 Code (optional)"))
        self.icd10_input = QLineEdit()
        self.icd10_input.setPlaceholderText("e.g. I50.0")
        self.icd10_input.setFixedHeight(38)
        icd_col.addWidget(self.icd10_input)

        phy_col = QVBoxLayout()
        phy_col.setSpacing(4)
        phy_col.addWidget(make_label("Responsible Physician"))
        self.physician_input = QLineEdit()
        self.physician_input.setPlaceholderText("e.g. Dr. Mueller")
        self.physician_input.setFixedHeight(38)
        phy_col.addWidget(self.physician_input)

        icd_phy_row.addLayout(icd_col)
        icd_phy_row.addLayout(phy_col)
        form_layout.addLayout(icd_phy_row)

        # Admission + Discharge row
        dates_row = QHBoxLayout()
        dates_row.setSpacing(16)

        adm_col = QVBoxLayout()
        adm_col.setSpacing(4)
        adm_col.addWidget(make_label("Admission Date", required=True))
        self.admission_input = QLineEdit()
        self.admission_input.setPlaceholderText("YYYY-MM-DD")
        self.admission_input.setFixedHeight(38)
        self.admission_error = make_error_label()
        adm_col.addWidget(self.admission_input)
        adm_col.addWidget(self.admission_error)

        dis_col = QVBoxLayout()
        dis_col.setSpacing(4)
        dis_col.addWidget(make_label("Discharge Date", required=True))
        self.discharge_input = QLineEdit()
        self.discharge_input.setPlaceholderText("YYYY-MM-DD")
        self.discharge_input.setFixedHeight(38)
        self.discharge_error = make_error_label()
        dis_col.addWidget(self.discharge_input)
        dis_col.addWidget(self.discharge_error)

        dates_row.addLayout(adm_col)
        dates_row.addLayout(dis_col)
        form_layout.addLayout(dates_row)

        # Follow-up date
        form_layout.addWidget(make_label("Follow-up Date (optional)"))
        self.followup_input = QLineEdit()
        self.followup_input.setPlaceholderText("YYYY-MM-DD")
        self.followup_input.setFixedHeight(38)
        form_layout.addWidget(self.followup_input)

        # ── Section 3: Clinical Notes ─────────────────────────────
        form_layout.addSpacing(8)
        form_layout.addWidget(make_section_header("Clinical Notes & Medication"))
        form_layout.addWidget(make_divider())

        # Notes
        form_layout.addWidget(make_label("Clinical Notes"))
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText(
            "Enter clinical observations, treatment summary, "
            "patient condition on discharge..."
        )
        self.notes_input.setFixedHeight(100)
        form_layout.addWidget(self.notes_input)

        # Medication
        form_layout.addWidget(make_label("Medication at Discharge"))
        self.medication_input = QTextEdit()
        self.medication_input.setPlaceholderText(
            "e.g. Furosemide 40mg once daily, Bisoprolol 5mg once daily..."
        )
        self.medication_input.setFixedHeight(80)
        form_layout.addWidget(self.medication_input)

        form_layout.addStretch()

        scroll.setWidget(body)
        outer.addWidget(header)
        outer.addWidget(scroll, stretch=1)

    # ================================================================
    # PUBLIC METHODS
    # ================================================================

    def clear(self):
        """Reset form for adding a new patient."""
        self._patient_id = None
        self.title_label.setText("New Patient")
        self.save_btn.setText("Save Patient")

        self.name_input.clear()
        self.age_input.setValue(0)
        self.dob_input.clear()
        self.diagnosis_input.clear()
        self.icd10_input.clear()
        self.physician_input.clear()
        self.admission_input.setText(date.today().strftime("%Y-%m-%d"))
        self.discharge_input.setText(date.today().strftime("%Y-%m-%d"))
        self.followup_input.clear()
        self.notes_input.clear()
        self.medication_input.clear()
        self._clear_errors()
        self.name_input.setFocus()

    def load_patient(self, patient: dict):
        """Load an existing patient into the form for editing."""
        self._patient_id = patient.get("id")
        name = patient.get("name", "Unknown")
        self.title_label.setText(f"Edit Patient — {name}")
        self.save_btn.setText("Save Changes")

        self.name_input.setText(str(patient.get("name", "")))
        self.age_input.setValue(int(patient.get("age", 0) or 0))
        self.dob_input.setText(str(patient.get("date_of_birth", "") or ""))
        self.diagnosis_input.setText(str(patient.get("diagnosis", "")))
        self.icd10_input.setText(str(patient.get("icd10_code", "") or ""))
        self.physician_input.setText(
            str(patient.get("physician_name", "") or "")
        )
        self.admission_input.setText(
            str(patient.get("admission_date", ""))
        )
        self.discharge_input.setText(
            str(patient.get("discharge_date", ""))
        )
        self.followup_input.setText(
            str(patient.get("followup_date", "") or "")
        )
        self.notes_input.setPlainText(str(patient.get("notes", "") or ""))
        self.medication_input.setPlainText(
            str(patient.get("medication", "") or "")
        )
        self._clear_errors()

    # ================================================================
    # PRIVATE METHODS
    # ================================================================

    def _collect_data(self) -> dict:
        """Collect all form fields into a dict."""
        return {
            "name":           self.name_input.text().strip(),
            "age":            self.age_input.value(),
            "date_of_birth":  self.dob_input.text().strip() or None,
            "diagnosis":      self.diagnosis_input.text().strip(),
            "icd10_code":     self.icd10_input.text().strip() or None,
            "physician_name": self.physician_input.text().strip(),
            "admission_date": self.admission_input.text().strip(),
            "discharge_date": self.discharge_input.text().strip(),
            "followup_date":  self.followup_input.text().strip() or None,
            "notes":          self.notes_input.toPlainText().strip(),
            "medication":     self.medication_input.toPlainText().strip(),
        }

    def _clear_errors(self):
        """Hide all error labels."""
        self.error_banner.hide()
        for err in [self.name_error, self.age_error,
                    self.diagnosis_error, self.admission_error,
                    self.discharge_error]:
            err.setText("")
            err.hide()

    def _show_field_error(self, error_msg: str):
        """Show error on the correct field based on message content."""
        self._clear_errors()
        msg_lower = error_msg.lower()

        if "name" in msg_lower:
            self.name_error.setText(error_msg)
            self.name_error.show()
            self.name_input.setFocus()
        elif "age" in msg_lower:
            self.age_error.setText(error_msg)
            self.age_error.show()
        elif "diagnosis" in msg_lower:
            self.diagnosis_error.setText(error_msg)
            self.diagnosis_error.show()
            self.diagnosis_input.setFocus()
        elif "admission" in msg_lower:
            self.admission_error.setText(error_msg)
            self.admission_error.show()
            self.admission_input.setFocus()
        elif "discharge" in msg_lower:
            self.discharge_error.setText(error_msg)
            self.discharge_error.show()
            self.discharge_input.setFocus()
        else:
            self.error_banner.setText(error_msg)
            self.error_banner.show()

    def _on_save(self):
        """Validate and save the patient record."""
        self._clear_errors()
        data = self._collect_data()

        # Validate first
        valid, error = validate_patient(data)
        if not valid:
            self._show_field_error(error)
            return

        self.save_btn.setEnabled(False)
        self.save_btn.setText("Saving...")

        user    = Session.get_user() or {}
        user_id = user.get("id")
        username = Session.get_username()

        if self._patient_id is None:
            # Creating new patient
            ok, err, new_id = create_patient(data, created_by=user_id)
            if ok:
                log_patient_create(
                    username,
                    patient_id=new_id,
                    patient_name=data["name"],
                    user_id=user_id,
                )
                self.on_save(new_id, is_new=True)
            else:
                self._show_field_error(err or "Failed to save patient.")
        else:
            # Updating existing patient
            ok, err, changed = update_patient(
                self._patient_id, data, updated_by=username
            )
            if ok:
                if changed:
                    log_patient_edit(
                        username,
                        patient_id=self._patient_id,
                        patient_name=data["name"],
                        changed_fields=changed,
                        user_id=user_id,
                    )
                self.on_save(self._patient_id, is_new=False)
            else:
                self._show_field_error(err or "Failed to update patient.")

        self.save_btn.setEnabled(True)
        self.save_btn.setText(
            "Save Patient" if self._patient_id is None else "Save Changes"
        )
