# main.py
# ================================================================
# NeuraCare — Application Entry Point
# ================================================================
# Run with:   python main.py          (development)
# Packaged:   NeuraCare.exe           (production)
# ================================================================

import sys
import os
from pathlib import Path


def get_base_dir() -> Path:
    """
    Get base directory — works in development and PyInstaller bundle.
    """
    if getattr(sys, "frozen", False):
        # Running as PyInstaller .exe — data lives next to the .exe
        return Path(sys.executable).parent
    else:
        # Development — data lives next to main.py
        return Path(__file__).parent


BASE_DIR = get_base_dir()
sys.path.insert(0, str(BASE_DIR))
os.environ["NEURACARE_BASE_DIR"] = str(BASE_DIR)

from app.core.database import init_db
from app.core.auth     import setup_first_admin
from app.ui.main_window import run


def bootstrap():
    """First-run setup: create DB and admin account."""
    data_dir = BASE_DIR / "app" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    success, msg = setup_first_admin(
        username="admin",
        password="NeuraCare2024!",
        full_name="Administrator",
    )
    if success:
        print("=" * 55)
        print("  NeuraCare — First Launch")
        print("  Username: admin  |  Password: NeuraCare2024!")
        print("  Please change your password after first login.")
        print("=" * 55)


if __name__ == "__main__":
    bootstrap()
    run()
