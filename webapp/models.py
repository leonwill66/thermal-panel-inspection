from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

ROLES = ("admin", "inspector", "viewer")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)


class AnalysisRun(Base):
    """One /api/analyze (or /api/report) call — a batch of one or more images
    analyzed together with the same parameters."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_by_username: Mapped[str] = mapped_column(String(64))  # denormalized: survives user deletion

    ambient_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_delta_c: Mapped[float] = mapped_column(Float)
    min_area_px: Mapped[int] = mapped_column(Integer)
    roi_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    roi_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    roi_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    roi_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    load_percent: Mapped[float | None] = mapped_column(Float, nullable=True)

    summary_json: Mapped[str] = mapped_column(Text)
    skipped_json: Mapped[str] = mapped_column(Text, default="[]")

    images: Mapped[list["AnalysisImage"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="AnalysisImage.id"
    )


class AnalysisImage(Base):
    __tablename__ = "analysis_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"))
    filename: Mapped[str] = mapped_column(String(255))
    ambient_c: Mapped[float] = mapped_column(Float)
    hotspots_json: Mapped[str] = mapped_column(Text)  # list[dict], same shape as hotspots_to_rows()
    annotated_image_path: Mapped[str] = mapped_column(String(500))  # relative to DATA_DIR
    comparative_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # list[dict], comparative_to_rows()

    run: Mapped["AnalysisRun"] = relationship(back_populates="images")
