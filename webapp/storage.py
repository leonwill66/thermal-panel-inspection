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


def _clean_env(name: str) -> str | None:
    """Reads and validates an env var used as an HTTP header value later
    (Supabase's client puts these straight into request headers). Strips
    incidental whitespace and fails loudly and immediately if the value
    contains non-ASCII characters - e.g. from a mangled copy-paste into a
    hosting provider's dashboard - rather than letting it surface as an
    opaque UnicodeEncodeError deep inside httpx on the first real request."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{name} contains non-ASCII characters (likely corrupted during copy/paste "
            f"into your hosting provider's env var settings) - re-copy it fresh from "
            f"Supabase and re-set it. Original error: {exc}"
        ) from exc
    return value


_SUPABASE_URL = _clean_env("SUPABASE_URL")
_SUPABASE_SERVICE_KEY = _clean_env("SUPABASE_SERVICE_KEY")
_BUCKET = _clean_env("SUPABASE_STORAGE_BUCKET") or "thermal-images"

_client = None
if _SUPABASE_URL and _SUPABASE_SERVICE_KEY:
    from supabase import create_client

    _client = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_KEY)


def using_supabase() -> bool:
    return _client is not None


def save_image(relative_path: str, data: bytes, content_type: str = "image/png") -> None:
    """relative_path is e.g. '12/0000_FLIR0893.jpg.png' - the same value
    stored in AnalysisImage.annotated_image_path. content_type defaults to
    png (every caller originally only ever stored derived PNG renderings);
    pass e.g. 'image/jpeg' when storing the original radiometric file
    itself (see AnalysisImage.raw_image_path) - local disk storage ignores
    this, but Supabase Storage serves whatever content-type it was given."""
    if _client is not None:
        _client.storage.from_(_BUCKET).upload(
            relative_path,
            data,
            file_options={"content-type": content_type, "upsert": "true"},
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
