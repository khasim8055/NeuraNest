# app/ui/settings_panel.py
# ================================================================
# NeuraCare — Settings Panel
# ================================================================
# Features:
#   - Change password (any user)
#   - Create new user account (admin only)
#   - View all users (admin only)
#   - Deactivate user account (admin only)
# ================================================================

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QScrollArea,
    QMessageBox, QComboBox,
)
from PyQt6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.auth   import (
    change_password, create_user, get_all_users,
    deactivate_user, Session,
)
from app.core.audit  import log_login

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
    QLineEdit, QComboBox {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border-color: {COLORS['accent_blue']};
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent_blue']};
    }}
"""


def _make_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {COLORS['text_muted']}; font-size: 11px; font-weight: 500;"
    )
    return label


def _make_section(title: str) -> QLabel:
    label = QLabel(title)
    label.setStyleSheet(
        f"color: {COLORS['accent_blue']}; font-size: 13px; "
        f"font-weight: bold; padding: 4px 0;"
    )
    return label


def _make_divider() -> QFrame:
    f = QFrame()
    f.setStyleSheet(f"background-color: {COLORS['border']};")
    f.setFixedHeight(1)
    return f


def _btn(text: str, color: str = None, danger: bool = False) -> QPushButton:
    bg = color or (COLORS['accent_red'] if danger else COLORS['accent_blue'])
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
        QPushButton:disabled {{
            background-color: {COLORS['bg_card']};
            color: {COLORS['text_muted']};
        }}
    """)
    return btn


# ================================================================
# CHANGE PASSWORD SECTION
# ================================================================

class ChangePasswordSection(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(FIELD_STYLE)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(_make_section("Change Password"))
        layout.addWidget(_make_divider())

        # Current password
        layout.addWidget(_make_label("Current Password"))
        self.current_pw = QLineEdit()
        self.current_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.current_pw.setPlaceholderText("Enter current password")
        self.current_pw.setFixedHeight(38)
        layout.addWidget(self.current_pw)

        # New password
        layout.addWidget(_make_label("New Password (min. 8 characters)"))
        self.new_pw = QLineEdit()
        self.new_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.new_pw.setPlaceholderText("Enter new password")
        self.new_pw.setFixedHeight(38)
        layout.addWidget(self.new_pw)

        # Confirm password
        layout.addWidget(_make_label("Confirm New Password"))
        self.confirm_pw = QLineEdit()
        self.confirm_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_pw.setPlaceholderText("Repeat new password")
        self.confirm_pw.setFixedHeight(38)
        self.confirm_pw.returnPressed.connect(self._on_change)
        layout.addWidget(self.confirm_pw)

        # Error/success label
        self.msg_label = QLabel("")
        self.msg_label.setStyleSheet(
            f"color: {COLORS['accent_red']}; font-size: 11px;"
        )
        layout.addWidget(self.msg_label)

        # Button
        self.change_btn = _btn("Change Password", COLORS['accent_green'])
        self.change_btn.clicked.connect(self._on_change)
        layout.addWidget(self.change_btn)

    def _on_change(self):
        self.msg_label.setStyleSheet(
            f"color: {COLORS['accent_red']}; font-size: 11px;"
        )
        self.msg_label.setText("")

        current = self.current_pw.text()
        new     = self.new_pw.text()
        confirm = self.confirm_pw.text()

        if not current:
            self.msg_label.setText("Please enter your current password.")
            return
        if not new or len(new) < 8:
            self.msg_label.setText("New password must be at least 8 characters.")
            return
        if new != confirm:
            self.msg_label.setText("New passwords do not match.")
            return

        username = Session.get_username()
        ok, err = change_password(username, current, new)

        if ok:
            self.msg_label.setStyleSheet(
                f"color: {COLORS['accent_green']}; font-size: 11px;"
            )
            self.msg_label.setText("Password changed successfully.")
            self.current_pw.clear()
            self.new_pw.clear()
            self.confirm_pw.clear()
        else:
            self.msg_label.setText(err or "Failed to change password.")


# ================================================================
# CREATE USER SECTION (admin only)
# ================================================================

class CreateUserSection(QWidget):
    def __init__(self, on_user_created=None):
        super().__init__()
        self.on_user_created = on_user_created
        self.setStyleSheet(FIELD_STYLE)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(_make_section("Create New User Account"))
        layout.addWidget(_make_divider())

        # Username
        layout.addWidget(_make_label("Username"))
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("e.g. dr.mueller")
        self.username_input.setFixedHeight(38)
        layout.addWidget(self.username_input)

        # Full name
        layout.addWidget(_make_label("Full Name"))
        self.fullname_input = QLineEdit()
        self.fullname_input.setPlaceholderText("e.g. Dr. Anna Mueller")
        self.fullname_input.setFixedHeight(38)
        layout.addWidget(self.fullname_input)

        # Password
        layout.addWidget(_make_label("Temporary Password (min. 8 characters)"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Set a temporary password")
        self.password_input.setFixedHeight(38)
        layout.addWidget(self.password_input)

        # Role
        layout.addWidget(_make_label("Role"))
        self.role_combo = QComboBox()
        self.role_combo.addItems(["doctor", "admin"])
        self.role_combo.setFixedHeight(38)
        layout.addWidget(self.role_combo)

        # Message
        self.msg_label = QLabel("")
        self.msg_label.setStyleSheet(
            f"color: {COLORS['accent_red']}; font-size: 11px;"
        )
        layout.addWidget(self.msg_label)

        # Button
        create_btn = _btn("Create User Account", COLORS['accent_blue'])
        create_btn.clicked.connect(self._on_create)
        layout.addWidget(create_btn)

    def _on_create(self):
        self.msg_label.setStyleSheet(
            f"color: {COLORS['accent_red']}; font-size: 11px;"
        )
        self.msg_label.setText("")

        username  = self.username_input.text().strip()
        fullname  = self.fullname_input.text().strip()
        password  = self.password_input.text()
        role      = self.role_combo.currentText()

        ok, err = create_user(username, password, fullname, role)

        if ok:
            self.msg_label.setStyleSheet(
                f"color: {COLORS['accent_green']}; font-size: 11px;"
            )
            self.msg_label.setText(
                f"User '{username}' created. They can log in now."
            )
            self.username_input.clear()
            self.fullname_input.clear()
            self.password_input.clear()
            if self.on_user_created:
                self.on_user_created()
        else:
            self.msg_label.setText(err or "Failed to create user.")


# ================================================================
# USER LIST SECTION (admin only)
# ================================================================

class UserListSection(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(_make_section("User Accounts"))
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(28)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                font-size: 11px;
                padding: 0 10px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_blue']}; }}
        """)
        refresh_btn.clicked.connect(self.refresh)
        header.addStretch()
        header.addWidget(refresh_btn)

        layout.addLayout(header)
        layout.addWidget(_make_divider())

        self.user_list_layout = QVBoxLayout()
        self.user_list_layout.setSpacing(6)
        layout.addLayout(self.user_list_layout)

        self.refresh()

    def refresh(self):
        # Clear existing
        while self.user_list_layout.count():
            item = self.user_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        users = get_all_users()
        for user in users:
            row = self._make_user_row(user)
            self.user_list_layout.addWidget(row)

        if not users:
            label = QLabel("No users found.")
            label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
            self.user_list_layout.addWidget(label)

    def _make_user_row(self, user: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)

        is_active = user.get("is_active", 1)
        status_color = COLORS["accent_green"] if is_active else COLORS["accent_red"]

        name_label = QLabel(
            f"{user.get('full_name', '')} ({user.get('username', '')})"
        )
        name_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 12px; font-weight: 500; background: transparent;"
        )

        role_label = QLabel(user.get("role", "").upper())
        role_label.setStyleSheet(
            f"color: {COLORS['accent_blue']}; font-size: 10px; background: transparent;"
        )

        status_label = QLabel("Active" if is_active else "Inactive")
        status_label.setStyleSheet(
            f"color: {status_color}; font-size: 10px; background: transparent;"
        )

        layout.addWidget(name_label, stretch=1)
        layout.addWidget(role_label)
        layout.addSpacing(12)
        layout.addWidget(status_label)

        # Deactivate button — cannot deactivate yourself
        current_user = Session.get_username()
        if user.get("username") != current_user and is_active:
            deact_btn = QPushButton("Deactivate")
            deact_btn.setFixedHeight(26)
            deact_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {COLORS['accent_red']};
                    border: 1px solid {COLORS['accent_red']};
                    border-radius: 4px;
                    font-size: 10px;
                    padding: 0 8px;
                }}
                QPushButton:hover {{ background-color: {COLORS['accent_red']}22; }}
            """)
            username = user.get("username", "")
            deact_btn.clicked.connect(
                lambda _, u=username: self._on_deactivate(u)
            )
            layout.addSpacing(8)
            layout.addWidget(deact_btn)

        return card

    def _on_deactivate(self, username: str):
        reply = QMessageBox.question(
            self, "Deactivate User",
            f"Deactivate account '{username}'?\n\n"
            "They will no longer be able to log in.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            ok, err = deactivate_user(username)
            if ok:
                self.refresh()
            else:
                QMessageBox.warning(self, "Error", err)


# ================================================================
# MAIN SETTINGS PANEL
# ================================================================

class SettingsPanel(QWidget):
    """
    Full settings panel — shown when Settings is clicked.
    Shows different sections based on user role.
    """

    def __init__(self, on_close=None):
        super().__init__()
        self.on_close = on_close
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ───────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        header.setFixedHeight(56)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.user_label = QLabel("")
        self.user_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_blue']}; }}
        """)
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self._on_close)

        header_layout.addWidget(title)
        header_layout.addWidget(self.user_label, stretch=1)
        header_layout.addWidget(close_btn)

        # ── Scrollable body ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {COLORS['bg_dark']}; }}"
        )

        self.body = QWidget()
        self.body.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(32, 24, 32, 32)
        self.body_layout.setSpacing(24)

        scroll.setWidget(self.body)
        outer.addWidget(header)
        outer.addWidget(scroll, stretch=1)

    def refresh(self):
        """Rebuild settings for current user."""
        # Clear body
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        user = Session.get_user()
        if not user:
            return

        name = user.get("full_name", "")
        role = user.get("role", "")
        self.user_label.setText(f"Logged in as: {name} ({role})")

        # Change password — available to all users
        self.body_layout.addWidget(ChangePasswordSection())

        # Admin-only sections
        if Session.is_admin():
            self.body_layout.addSpacing(8)
            self.user_list = UserListSection()
            self.body_layout.addWidget(self.user_list)

            self.body_layout.addSpacing(8)
            self.body_layout.addWidget(
                CreateUserSection(
                    on_user_created=self.user_list.refresh
                )
            )

        self.body_layout.addStretch()

    def _on_close(self):
        if self.on_close:
            self.on_close()
