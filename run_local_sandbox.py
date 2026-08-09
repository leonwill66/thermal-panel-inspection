"""Local-only dev server launcher - see the project memory note on .env
pointing at production Supabase. Setting these to "" here (in Python, before
webapp.__init__'s load_dotenv runs) keeps the keys genuinely *present* in
os.environ, which is what makes python-dotenv's override=False skip them.
A Windows batch file's `set VAR=` instead *removes* the key entirely, which
silently let .env's real DATABASE_URL through - do not swap this back to a
.bat wrapper without accounting for that.
"""

import os

os.environ["DATABASE_URL"] = ""
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""

import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import webapp.db as _db  # noqa: E402

print(f"DEBUG engine url: {_db.engine.url}", flush=True)
assert str(_db.engine.url).startswith("sqlite:"), "refusing to start - not pointed at local SQLite"

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("webapp.server:app", host="127.0.0.1", port=8000, reload=True)
