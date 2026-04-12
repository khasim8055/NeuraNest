# main.py
# ================================================================
# NeuraCare — Application Entry Point
# Run with: python main.py
# ================================================================
# This file:
#   1. Initialises the database if first run
#   2. Creates default admin account if no users exist
#   3. Launches the PyQt6 desktop window
# ================================================================

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import init_db
from app.core.auth     import setup_first_admin
from app.ui.main_window import run


def bootstrap():
    """
    Run once on first launch:
    - Create database and tables
    - Create default admin account
    """
    # Ensure data directory exists
    data_dir = Path(__file__).parent / "app" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Initialise database (creates tables from schema.sql)
    init_db()

    # Create default admin if no users exist
    success, msg = setup_first_admin(
        username="admin",
        password="NeuraCare2024!",
        full_name="Administrator",
    )
    if success:
        print("=" * 55)
        print("  First run — default admin account created.")
        print("  Username: admin")
        print("  Password: NeuraCare2024!")
        print("  CHANGE THIS PASSWORD after first login.")
        print("=" * 55)


if __name__ == "__main__":
    bootstrap()
    run()
