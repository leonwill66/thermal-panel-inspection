"""Shared test fixtures.

Test isolation from production is enforced at IMPORT time, not just in
fixtures: DATABASE_URL/SUPABASE_*/GOOGLE_* are pinned to safe values at
module level, below, before any test module (or this file) imports
anything from webapp - see the project memory note on .env pointing at
production Supabase. `webapp/__init__.py`'s load_dotenv() only fills in
keys that aren't already present in os.environ (override=False), so
setting these here first is what keeps a developer's real .env from ever
reaching a running test process.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="thermal_inspector_tests_"))

# A real temp file, not sqlite:///:memory: - :memory: gives every new
# connection its own blank database, which breaks under FastAPI's
# TestClient (different requests can land on different connections).
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_DB_DIR / 'test.db').as_posix()}"
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""
os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = ""
os.environ["GOOGLE_DRIVE_FOLDER_ID"] = ""
os.environ["SESSION_SECRET_KEY"] = "test-session-secret-not-for-production"
os.environ.pop("SESSION_HTTPS_ONLY", None)  # must stay falsy - TestClient isn't served over https

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from webapp import storage
from webapp.auth import hash_password
from webapp.db import SessionLocal, engine
from webapp.models import Base, User
from webapp.server import app


@pytest.fixture(autouse=True)
def clean_db():
    """Fresh tables before every test - cheap enough at this DB size, and
    removes any need to reason about ordering/pollution between tests."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture(autouse=True)
def reset_login_rate_limit():
    """webapp.server's failed-login tracker is module-level in-memory state,
    keyed by client host - TestClient always presents the same host, so
    without this, failed-login tests would accumulate across the whole test
    run and eventually 429 unrelated tests."""
    import webapp.server as server_module

    server_module._failed_login_attempts.clear()
    yield


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Redirects local-disk image storage to a per-test temp dir instead of
    the real webapp/data/runs/ - keeps test runs from leaving files behind
    in the actual project checkout."""
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path)


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def make_user(db_session):
    """Factory fixture: make_user('alice', role='admin') -> User, already
    committed. Default password is 'password123' unless overridden."""

    def _make(username: str, role: str = "admin", password: str = "password123", active: bool = True) -> User:
        user = User(username=username, password_hash=hash_password(password), role=role, is_active=active)
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user

    return _make


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def logged_in_client(client, make_user):
    """A TestClient already logged in as an admin user named 'admin' /
    'password123'. Session cookie persists across requests on this client."""
    make_user("admin", role="admin")
    resp = client.post("/api/login", data={"username": "admin", "password": "password123"})
    assert resp.status_code == 200
    return client


def synthetic_temperature_c(height: int = 60, width: int = 80, ambient: float = 20.0) -> np.ndarray:
    """A flat 'ambient' plate with one clearly hot rectangular region baked
    in - big enough (default min_area_px=25) and hot enough (default
    min_delta_c=8.0) to be found by find_hotspots with default thresholds."""
    temp = np.full((height, width), ambient, dtype=np.float32)
    temp[10:20, 10:20] = ambient + 25.0  # 100px region, delta_t=25 -> "serious"
    return temp


@pytest.fixture
def mock_radiometric(monkeypatch):
    """Patches webapp.server.load_radiometric so /api/analyze can be
    exercised end-to-end (detection, annotation, DB writes, PNG encoding)
    without a real radiometric FLIR JPEG - the uploaded file's actual bytes
    are ignored and a synthetic Thermogram is returned instead. Returns the
    temperature array used, so tests can assert on expected hotspot values."""
    import webapp.server as server_module
    from thermal_inspector.core import Thermogram

    temp_c = synthetic_temperature_c()

    def _fake_load_radiometric(path):
        return Thermogram(source_path=Path(path), temperature_c=temp_c.copy(), visual=None)

    monkeypatch.setattr(server_module, "load_radiometric", _fake_load_radiometric)
    return temp_c
