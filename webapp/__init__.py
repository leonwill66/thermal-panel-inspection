from pathlib import Path

from dotenv import load_dotenv

# Loaded before any submodule (db.py, storage.py, auth.py) reads env vars at
# import time - .env is optional and gitignored; without it, everything
# falls back to local SQLite + disk as before.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
