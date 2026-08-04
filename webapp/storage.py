"""Stores/retrieves annotated run images.

Two modes, chosen automatically:
- Local disk under webapp/data/runs/ (default - no external credentials
  needed, good for local dev and for a single-instance Fly.io deployment
  with a mounted volume).
- Supabase Storage, when SUPABASE_URL and SUPABASE_SERVICE_KEY are set -
  used when Postgres/Supabase is the system of record instead of local
  SQLite+disk.

Only one mode is active per process; there's no migration between them here.
"""

from __future__ import annotations

import os

from .db import RUNS_DIR

_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "thermal-images")

_client = None
if _SUPABASE_URL and _SUPABASE_SERVICE_KEY:
    from supabase import create_client

    _client = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_KEY)


def using_supabase() -> bool:
    return _client is not None


def save_image(relative_path: str, data: bytes) -> None:
    """relative_path is e.g. '12/0000_FLIR0893.jpg.png' - the same value
    stored in AnalysisImage.annotated_image_path."""
    if _client is not None:
        _client.storage.from_(_BUCKET).upload(
            relative_path,
            data,
            file_options={"content-type": "image/png", "upsert": "true"},
        )
    else:
        dest = RUNS_DIR / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def load_image(relative_path: str) -> bytes | None:
    if _client is not None:
        try:
            return _client.storage.from_(_BUCKET).download(relative_path)
        except Exception:
            return None
    else:
        path = RUNS_DIR / relative_path
        return path.read_bytes() if path.exists() else None
