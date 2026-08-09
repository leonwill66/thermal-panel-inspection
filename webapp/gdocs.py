"""Auth wiring for Google Docs export (thermal_inspector.gdocs_report).

Uses a Workspace service account. Service accounts have no personal Drive
storage quota of their own, so GOOGLE_DRIVE_FOLDER_ID must point at a folder
inside a Shared Drive that the service account has been added to (Content
Manager or above) - a personal "My Drive" folder will not work here even if
shared with the service account's email.

GOOGLE_SERVICE_ACCOUNT_JSON leaving both unset disables the feature entirely
(configured() returns False) rather than failing at import time, so local
dev without Google credentials configured keeps working.
"""

from __future__ import annotations

import json
import os

SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


_SERVICE_ACCOUNT_RAW = _clean_env("GOOGLE_SERVICE_ACCOUNT_JSON")
_DRIVE_FOLDER_ID = _clean_env("GOOGLE_DRIVE_FOLDER_ID")

_docs_service = None
_drive_service = None

if _SERVICE_ACCOUNT_RAW:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    # Accept either the raw JSON key content (for hosting providers where a
    # file-based secret is awkward, e.g. Fly.io) or a path to the key file
    # (convenient for local dev) - same "value or path" flexibility other
    # deploy targets commonly need.
    if _SERVICE_ACCOUNT_RAW.lstrip().startswith("{"):
        _info = json.loads(_SERVICE_ACCOUNT_RAW)
        _credentials = service_account.Credentials.from_service_account_info(_info, scopes=SCOPES)
    else:
        _credentials = service_account.Credentials.from_service_account_file(_SERVICE_ACCOUNT_RAW, scopes=SCOPES)
    _docs_service = build("docs", "v1", credentials=_credentials, cache_discovery=False)
    _drive_service = build("drive", "v3", credentials=_credentials, cache_discovery=False)


def configured() -> bool:
    return _docs_service is not None and _DRIVE_FOLDER_ID is not None


def docs_service():
    return _docs_service


def drive_service():
    return _drive_service


def target_folder_id() -> str | None:
    return _DRIVE_FOLDER_ID
