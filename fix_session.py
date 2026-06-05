# fix_session.py
import ast

with open('app/core/auth.py', encoding='utf-8-sig') as f:
    content = f.read()

# Find the Session class and add persistence
old = '''class Session:
    """
    In-memory session — stores the currently logged-in user.
    Reset when app closes — user must log in again each time.
    """
    _user: dict | None = None'''

new = '''class Session:
    """
    Persistent session — stores the currently logged-in user.
    Session persists for 12 hours (working day) across app restarts.
    """
    _user: dict | None = None
    _SESSION_FILE = None
    _SESSION_HOURS = 12

    @classmethod
    def _get_session_file(cls):
        import os
        from pathlib import Path
        base = Path(os.environ.get("NEURACARE_BASE_DIR", ""))
        if not base or not base.exists():
            base = Path(__file__).parent.parent.parent
        return base / "app" / "data" / ".session"

    @classmethod
    def save(cls):
        """Save session to disk."""
        if not cls._user:
            return
        try:
            import json
            from datetime import datetime
            data = {
                "user": cls._user,
                "saved_at": datetime.now().isoformat(),
            }
            sf = cls._get_session_file()
            sf.parent.mkdir(parents=True, exist_ok=True)
            sf.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def restore(cls) -> bool:
        """
        Restore session from disk if less than SESSION_HOURS old.
        Returns True if session was restored successfully.
        """
        try:
            import json
            from datetime import datetime
            sf = cls._get_session_file()
            if not sf.exists():
                return False
            data = json.loads(sf.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(data["saved_at"])
            age_hours = (datetime.now() - saved_at).total_seconds() / 3600
            if age_hours > cls._SESSION_HOURS:
                sf.unlink(missing_ok=True)
                return False
            cls._user = data["user"]
            return True
        except Exception:
            return False

    @classmethod
    def clear_saved(cls):
        """Delete saved session file."""
        try:
            sf = cls._get_session_file()
            if sf.exists():
                sf.unlink()
        except Exception:
            pass'''

if old in content:
    content = content.replace(old, new)
    print("1. Session persistence added")
else:
    print("1. Session class pattern not found")
    # Find it
    idx = content.find("class Session:")
    print(repr(content[idx:idx+200]))

# Update login() to save session
old2 = '''    @classmethod
    def login(cls, user: dict):
        """Store the logged-in user."""
        cls._user = user'''

new2 = '''    @classmethod
    def login(cls, user: dict):
        """Store the logged-in user and persist to disk."""
        cls._user = user
        cls.save()'''

if old2 in content:
    content = content.replace(old2, new2)
    print("2. login() saves session")
else:
    print("2. login() pattern not found")

# Update logout() to clear saved session
old3 = '''    @classmethod
    def logout(cls):
        """Clear the session."""
        cls._user = None'''

new3 = '''    @classmethod
    def logout(cls):
        """Clear the session and delete saved file."""
        cls._user = None
        cls.clear_saved()'''

if old3 in content:
    content = content.replace(old3, new3)
    print("3. logout() clears saved session")
else:
    print("3. logout() pattern not found")

with open('app/core/auth.py', 'w', encoding='utf-8') as f:
    f.write(content)

ast.parse(content)
print("4. auth.py syntax OK")

# Now update main.py to restore session on startup
with open('main.py', encoding='utf-8-sig') as f:
    main = f.read()

old_bootstrap = '''def bootstrap():
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
        print("  Default admin account created. Log in and change your password.")
        print("=" * 55)'''

new_bootstrap = '''def bootstrap():
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
        print("  Default admin account created. Log in and change your password.")
        print("=" * 55)

    # Restore session if still valid (within working day)
    from app.core.auth import Session
    Session.restore()'''

if old_bootstrap in main:
    main = main.replace(old_bootstrap, new_bootstrap)
    print("5. main.py restores session on startup")
else:
    print("5. bootstrap pattern not found")

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(main)

ast.parse(main)
print("6. main.py syntax OK")
print("\nDone. Run: python main.py")
print("Log in once — close the app — reopen — should skip login for 12 hours")