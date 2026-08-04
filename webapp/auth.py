from __future__ import annotations

import os
import secrets

import bcrypt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import DATA_DIR, get_db
from .models import User

SESSION_SECRET_FILE = DATA_DIR / ".session_secret"


def get_session_secret() -> str:
    """Signing key for session cookies. Reads SESSION_SECRET_KEY if set (do
    this in production / whenever the server might restart with a different
    working directory); otherwise persists a generated key locally so
    sessions survive a plain restart, but not a fresh checkout/deploy."""
    env_secret = os.environ.get("SESSION_SECRET_KEY")
    if env_secret:
        return env_secret
    if SESSION_SECRET_FILE.exists():
        return SESSION_SECRET_FILE.read_text(encoding="utf-8").strip()
    secret = secrets.token_hex(32)
    SESSION_SECRET_FILE.write_text(secret, encoding="utf-8")
    return secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def get_user_from_session(request: Request, db: Session) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    user = db.get(User, user_id)
    if not user or not user.is_active:
        return None
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """API-route dependency: 401 JSON if not logged in (or account deactivated)."""
    user = get_user_from_session(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_role(*roles: str):
    """API-route dependency factory: 403 JSON if the logged-in user's role
    isn't one of `roles`. Use after get_current_user has confirmed login."""

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {' or '.join(roles)}")
        return user

    return checker
