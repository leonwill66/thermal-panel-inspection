from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "app.db"

# DATABASE_URL (e.g. a Supabase Postgres connection string) takes over when
# set; otherwise falls back to a local SQLite file - no external database
# needed for local dev.
_DATABASE_URL = os.environ.get("DATABASE_URL")
if _DATABASE_URL:
    engine = create_engine(_DATABASE_URL, pool_pre_ping=True)
else:
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# (table, column, SQL type) added to a model after its table already existed
# in deployed databases - create_all() only creates missing TABLES, it never
# alters an existing one, so a plain new mapped_column() silently does
# nothing on a database that predates it. Add an entry here whenever that
# happens; _migrate_add_missing_columns() applies whichever of these aren't
# already present, and is a no-op (skips the whole table) on a brand new
# database, where create_all() already defines the column correctly.
_COLUMN_MIGRATIONS = [
    ("analysis_images", "visual_note", "TEXT"),
    ("analysis_images", "visual_anomaly", "BOOLEAN NOT NULL DEFAULT FALSE"),
]


def _migrate_add_missing_columns() -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table, column, sql_type in _COLUMN_MIGRATIONS:
        if table not in existing_tables:
            continue
        existing_columns = {col["name"] for col in inspector.get_columns(table)}
        if column in existing_columns:
            continue
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate_add_missing_columns()


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
