"""Local multi-user web UI for thermal_inspector: log in, upload one or more
radiometric FLIR JPEGs (or a whole folder), get back annotated images, a
severity-classified hotspot report, and a PDF export — with every analysis
saved to a local history any logged-in user can browse.

Roles: admin (manage users, full access), inspector (analyze + report),
viewer (browse/download past reports only).

First run: no accounts exist yet. Create an admin with:
  python -m webapp.manage create-user <username> --role admin

Run with:  uvicorn webapp.server:app --reload
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask
from starlette.middleware.sessions import SessionMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_inspector import (
    ImageReportEntry,
    ReportMetadata,
    annotate_image,
    classify_severity,
    find_hotspots,
    generate_audit_findings_report,
    generate_pdf_report,
    load_radiometric,
    load_radiometric_with_emissivity,
)
from thermal_inspector.annotate import compute_scale, draw_hotspot_rows
from thermal_inspector.core import find_comparative_anomalies
from thermal_inspector.gdocs_report import build_findings_doc
from thermal_inspector.report import comparative_to_rows, hotspots_to_rows, summarize

from . import gdocs, storage
from .auth import get_current_user, get_session_secret, get_user_from_session, hash_password, require_role, verify_password
from .db import get_db, init_db
from .models import ROLES, AnalysisImage, AnalysisRun, AuditLogEntry, User

init_db()

app = FastAPI(title="Thermal Panel Inspector")
app.add_middleware(
    SessionMiddleware,
    secret_key=get_session_secret(),
    same_site="lax",
    # Set SESSION_HTTPS_ONLY=true once deployed behind TLS (e.g. Fly.io
    # terminates TLS at the edge but the app itself still sees plain HTTP,
    # so this can't be inferred automatically - it's an explicit opt-in).
    https_only=os.environ.get("SESSION_HTTPS_ONLY", "false").lower() == "true",
)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Login rate limiting (in-memory - fine for a single-process deployment;
# resets on restart, which is an acceptable tradeoff for a small internal tool)
# ---------------------------------------------------------------------------

LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300
_failed_login_attempts: dict[str, list[float]] = defaultdict(list)


def _login_rate_limit_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_login_rate_limit(key: str) -> None:
    now = time.time()
    attempts = _failed_login_attempts[key]
    attempts[:] = [t for t in attempts if now - t < LOGIN_WINDOW_SECONDS]
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Try again in a few minutes.",
        )


def _record_failed_login(key: str) -> None:
    _failed_login_attempts[key].append(time.time())


def _clear_failed_logins(key: str) -> None:
    _failed_login_attempts.pop(key, None)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_roi(roi_x: int | None, roi_y: int | None, roi_w: int | None, roi_h: int | None):
    roi_fields = (roi_x, roi_y, roi_w, roi_h)
    if any(v is not None for v in roi_fields) and not all(v is not None for v in roi_fields):
        raise HTTPException(
            status_code=422, detail="roi_x, roi_y, roi_w, roi_h must all be set together, or all left blank"
        )
    return (roi_x, roi_y, roi_w, roi_h) if roi_x is not None else None


async def _save_upload(file: UploadFile, dest_dir: Path, index: int) -> Path:
    file_dir = dest_dir / str(index)
    file_dir.mkdir(parents=True, exist_ok=True)
    original_name = Path(file.filename or "upload.jpg").name
    dest_path = file_dir / original_name
    dest_path.write_bytes(await file.read())
    return dest_path


def _detect(upload_path: Path, ambient, min_delta, min_area, roi, load_percent=None):
    try:
        thermogram = load_radiometric(upload_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        hotspots, ambient_used = find_hotspots(
            thermogram.temperature_c,
            ambient_c=ambient,
            min_delta_c=min_delta,
            min_area_px=min_area,
            roi=roi,
            load_percent=load_percent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return thermogram, hotspots, ambient_used


def _parse_compare_regions(raw: str | None) -> list[tuple[tuple[int, int, int, int], str]] | None:
    """raw is a JSON string like [[[x,y,w,h],"label"], ...] from the frontend
    (or a CLI-style caller could build the same shape directly). Returns None
    if raw is empty/absent - comparative analysis is then simply skipped."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"compare_regions is not valid JSON: {exc}")

    if not isinstance(parsed, list) or len(parsed) < 2:
        raise HTTPException(status_code=422, detail="compare_regions needs at least 2 regions")

    regions = []
    for item in parsed:
        try:
            (x, y, w, h), label = item
            regions.append(((int(x), int(y), int(w), int(h)), str(label)))
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422, detail=f"compare_regions entry must be [[x,y,w,h], label], got {item!r}"
            )
    return regions


def _user_summary(u: User) -> dict:
    return {
        "id": u.id,
        "username": u.username,
        "role": u.role,
        "is_active": u.is_active,
        "created_at": u.created_at.isoformat(),
    }


def _excluded_indices(img: AnalysisImage) -> set[int]:
    return set(json.loads(img.excluded_hotspot_indices)) if img.excluded_hotspot_indices else set()


def _effective_hotspot_rows(img: AnalysisImage) -> list[dict]:
    """Hotspot rows for img with reviewer-dismissed false positives (e.g. a
    tool in frame, bare reflective metal) removed - what should actually be
    treated as a finding."""
    excluded = _excluded_indices(img)
    rows = json.loads(img.hotspots_json)
    return [row for i, row in enumerate(rows) if i not in excluded]


def _recompute_run_summary(run: AnalysisRun) -> None:
    """Recomputes run.summary_json from all images' effective (non-excluded)
    hotspot rows. Call after changing an image's excluded set, before commit."""
    all_rows = [row for img in run.images for row in _effective_hotspot_rows(img)]
    run.summary_json = json.dumps(summarize(all_rows))


def _log_audit(db: Session, *, run_id: int, image_id: int | None, user: User, action: str, detail: dict) -> None:
    """Appends an audit-log entry. Doesn't commit - callers already commit
    their own change in the same transaction; add this call before that
    commit so both land atomically."""
    db.add(
        AuditLogEntry(
            run_id=run_id,
            image_id=image_id,
            user_id=user.id,
            username=user.username,
            action=action,
            detail_json=json.dumps(detail),
        )
    )


def _placeholder_image_bytes(message: str) -> bytes:
    """A small placeholder PNG for when a stored image is missing (e.g. lost
    from object storage) - shown instead of a broken image icon."""
    canvas = np.full((360, 480, 3), 40, dtype=np.uint8)
    words = message.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 28:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    y = 170 - (len(lines) - 1) * 12
    for text_line in lines:
        cv2.putText(canvas, text_line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
        y += 24
    ok, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _rendered_annotated_bytes(img: AnalysisImage) -> bytes | None:
    """The annotated (boxed) image to actually show/report for img, reflecting
    any reviewer exclusions. Images stored before has_unannotated_base existed
    already have hotspot boxes baked in permanently and are returned as-is -
    exclusions on those are only reflected in tables/summaries, not the
    picture itself."""
    base_bytes = storage.load_image(img.annotated_image_path)
    if base_bytes is None or not img.has_unannotated_base:
        return base_bytes
    base_arr = cv2.imdecode(np.frombuffer(base_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    rendered = draw_hotspot_rows(base_arr, _effective_hotspot_rows(img), scale=img.annotate_scale)
    ok, buf = cv2.imencode(".png", rendered)
    return buf.tobytes() if ok else base_bytes


def _run_summary(run: AnalysisRun) -> dict:
    return {
        "id": run.id,
        "created_at": run.created_at.isoformat(),
        "created_by": run.created_by_username,
        "image_count": len(run.images),
        "summary": json.loads(run.summary_json),
    }


# ---------------------------------------------------------------------------
# Pages (session-cookie gated; redirect to /login rather than 401 JSON)
# ---------------------------------------------------------------------------


@app.get("/login")
def login_page():
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/")
def index(request: Request, db: Session = Depends(get_db)):
    if get_user_from_session(request, db) is None:
        return RedirectResponse("/login")
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/history")
def history_page(request: Request, db: Session = Depends(get_db)):
    if get_user_from_session(request, db) is None:
        return RedirectResponse("/login")
    return FileResponse(STATIC_DIR / "history.html")


@app.get("/admin")
def admin_page(request: Request, db: Session = Depends(get_db)):
    user = get_user_from_session(request, db)
    if user is None:
        return RedirectResponse("/login")
    if user.role != "admin":
        return RedirectResponse("/")
    return FileResponse(STATIC_DIR / "admin.html")


# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------


@app.post("/api/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    rate_limit_key = _login_rate_limit_key(request)
    _check_login_rate_limit(rate_limit_key)

    user = db.query(User).filter_by(username=username).first()
    if not user or not user.is_active or not verify_password(password, user.password_hash):
        _record_failed_login(rate_limit_key)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    _clear_failed_logins(rate_limit_key)
    request.session.clear()
    request.session["user_id"] = user.id
    return {"username": user.username, "role": user.role}


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}


# ---------------------------------------------------------------------------
# Analysis + reporting (admin, inspector)
# ---------------------------------------------------------------------------


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    ambient: float | None = Form(None),
    min_delta: float = Form(8.0),
    min_area: int = Form(25),
    roi_x: int | None = Form(None),
    roi_y: int | None = Form(None),
    roi_w: int | None = Form(None),
    roi_h: int | None = Form(None),
    load_percent: float | None = Form(None),
    compare_regions: str | None = Form(None),
    user: User = Depends(require_role("admin", "inspector")),
    db: Session = Depends(get_db),
):
    """Analyze one or more images with shared parameters, and save the run to
    history. roi and compare_regions, if given, apply to every file — only
    sensible when they share the same framing/resolution. load_percent
    severity-classifies each hotspot by its load-corrected delta-T instead
    of the raw observed one (see core.load_adjusted_delta_t). compare_regions
    is a JSON string [[[x,y,w,h],"label"], ...] with at least 2 entries."""
    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded")
    roi = _resolve_roi(roi_x, roi_y, roi_w, roi_h)
    regions = _parse_compare_regions(compare_regions)

    tmp_dir = Path(tempfile.mkdtemp(prefix="thermal_analyze_"))
    try:
        run = AnalysisRun(
            created_by_id=user.id,
            created_by_username=user.username,
            ambient_c=ambient,
            min_delta_c=min_delta,
            min_area_px=min_area,
            roi_x=roi_x,
            roi_y=roi_y,
            roi_w=roi_w,
            roi_h=roi_h,
            load_percent=load_percent,
            summary_json="{}",
            skipped_json="[]",
        )
        db.add(run)
        db.flush()  # assigns run.id, doesn't commit yet

        results = []
        skipped = []
        all_rows: list[dict] = []
        for i, file in enumerate(files):
            image_name = file.filename or f"upload_{i}.jpg"
            upload_path = await _save_upload(file, tmp_dir, i)
            try:
                thermogram, hotspots, ambient_used = _detect(
                    upload_path, ambient, min_delta, min_area, roi, load_percent=load_percent
                )
            except HTTPException as exc:
                skipped.append({"filename": image_name, "reason": str(exc.detail)})
                continue

            # Store the UNANNOTATED base (no hotspot boxes) so a reviewer's
            # later exclusions can be reflected by redrawing just the
            # currently-active hotspots (see draw_hotspot_rows), rather than
            # a fixed image baked in at analyze time that would still show a
            # box around something a reviewer has dismissed as a false
            # positive (e.g. a tool left in frame, bare reflective metal).
            base_image = annotate_image(thermogram, [], roi=roi)
            scale = compute_scale(thermogram.temperature_c.shape[1])
            ok, base_buf = cv2.imencode(".png", base_image)
            if not ok:
                skipped.append({"filename": image_name, "reason": "Failed to encode annotated image"})
                continue

            rows = hotspots_to_rows(image_name, hotspots)
            all_rows.extend(rows)
            height, width = thermogram.temperature_c.shape[:2]

            preview = draw_hotspot_rows(base_image, rows, scale=scale)
            ok_p, buf = cv2.imencode(".png", preview)
            if not ok_p:
                skipped.append({"filename": image_name, "reason": "Failed to encode annotated image"})
                continue

            comparative_rows = None
            if regions:
                comparative = find_comparative_anomalies(thermogram.temperature_c, regions)
                comparative_rows = comparative_to_rows(image_name, comparative)

            visual_bytes = None
            if thermogram.visual is not None:
                ok_v, vbuf = cv2.imencode(".png", thermogram.visual)
                visual_bytes = vbuf.tobytes() if ok_v else None

            stored_name = f"{i:04d}_{Path(image_name).name}"
            storage.save_image(f"{run.id}/{stored_name}", base_buf.tobytes())
            visual_path = None
            if visual_bytes is not None:
                visual_path = f"{run.id}/{stored_name}.photo.png"
                storage.save_image(visual_path, visual_bytes)
            # The original radiometric file itself - not just a rendering of
            # it - so a reviewer can later recompute temperature at a
            # different emissivity (thermal_inspector.core.load_radiometric
            # needs the raw FLIR bytes, not the derived PNG, for that).
            # upload_path may already be gone by the time this runs if the
            # file failed to encode above, but reaching this point means it
            # was read successfully, so it's still on disk in tmp_dir.
            raw_path = f"{run.id}/{stored_name}.raw{Path(image_name).suffix or '.jpg'}"
            storage.save_image(raw_path, upload_path.read_bytes(), content_type="image/jpeg")
            image_row = AnalysisImage(
                run_id=run.id,
                filename=image_name,
                ambient_c=ambient_used,
                hotspots_json=json.dumps(rows),
                annotated_image_path=f"{run.id}/{stored_name}",
                comparative_json=json.dumps(comparative_rows) if comparative_rows is not None else None,
                visual_image_path=visual_path,
                annotate_scale=scale,
                has_unannotated_base=True,
                raw_image_path=raw_path,
            )
            db.add(image_row)
            db.flush()  # assigns image_row.id

            results.append(
                {
                    "image_id": image_row.id,
                    "filename": image_name,
                    "ambient_c": round(ambient_used, 2),
                    "image_width": width,
                    "image_height": height,
                    "hotspots": rows,
                    "comparative": comparative_rows,
                    "annotated_image_png_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "visual_image_png_base64": base64.b64encode(visual_bytes).decode("ascii") if visual_bytes else None,
                }
            )

        summary = summarize(all_rows)
        run.summary_json = json.dumps(summary)
        run.skipped_json = json.dumps(skipped)
        db.commit()

        return {"run_id": run.id, "results": results, "skipped": skipped, "summary": summary}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/history/{run_id}/images/{image_id}/exclude")
def set_excluded_hotspots(
    run_id: int,
    image_id: int,
    hotspot_indices: list[int] = Body(..., embed=True),
    user: User = Depends(require_role("admin", "inspector")),
    db: Session = Depends(get_db),
):
    """Marks specific hotspot findings on one image as reviewer-dismissed
    false positives (e.g. a tool left in frame, bare reflective metal, the
    back of the panel enclosure) - excluded from reports and summary counts,
    but the underlying detection data isn't deleted so this can be undone.
    hotspot_indices replaces the full excluded set for this image (send every
    index that should currently be excluded, not just newly-added ones)."""
    img = db.get(AnalysisImage, image_id)
    if not img or img.run_id != run_id:
        raise HTTPException(status_code=404, detail="Image not found")
    total = len(json.loads(img.hotspots_json))
    if any(i < 0 or i >= total for i in hotspot_indices):
        raise HTTPException(status_code=422, detail=f"hotspot_indices must be in range [0, {total})")

    before = sorted(_excluded_indices(img))
    after = sorted(set(hotspot_indices))
    img.excluded_hotspot_indices = json.dumps(after)
    _recompute_run_summary(img.run)
    if before != after:
        _log_audit(
            db,
            run_id=run_id,
            image_id=image_id,
            user=user,
            action="exclude_hotspots",
            detail={"before": before, "after": after},
        )
    db.commit()

    excluded = _excluded_indices(img)
    rows = json.loads(img.hotspots_json)
    return {
        "hotspots": [{**row, "excluded": i in excluded} for i, row in enumerate(rows)],
        "run_summary": json.loads(img.run.summary_json),
    }


@app.post("/api/history/{run_id}/images/{image_id}/note")
def set_visual_note(
    run_id: int,
    image_id: int,
    note: str | None = Body(None, embed=True),
    anomaly: bool = Body(False, embed=True),
    user: User = Depends(require_role("admin", "inspector")),
    db: Session = Depends(get_db),
):
    """Records an inspector's explicit flag for a visually-observed issue
    that thermal detection wouldn't catch - physical damage, corrosion, a
    cracked enclosure, etc - plus an optional free-text description. anomaly
    is the flag that actually drives report inclusion (from
    generate_audit_findings_report onward, enough on its own to keep the
    image from being treated as 'clean' and dropped from the client-facing
    audit report); note is just supplementary text and doesn't affect
    inclusion by itself. Unlike hotspot findings this isn't detected
    automatically, so there's nothing to recompute here - just persisted."""
    img = db.get(AnalysisImage, image_id)
    if not img or img.run_id != run_id:
        raise HTTPException(status_code=404, detail="Image not found")

    new_note = note.strip() if note and note.strip() else None
    if (img.visual_note, img.visual_anomaly) != (new_note, anomaly):
        _log_audit(
            db,
            run_id=run_id,
            image_id=image_id,
            user=user,
            action="set_visual_anomaly",
            detail={
                "before": {"visual_note": img.visual_note, "visual_anomaly": img.visual_anomaly},
                "after": {"visual_note": new_note, "visual_anomaly": anomaly},
            },
        )
    img.visual_note = new_note
    img.visual_anomaly = anomaly
    db.commit()
    return {"visual_note": img.visual_note, "visual_anomaly": img.visual_anomaly}


@app.post("/api/history/{run_id}/images/{image_id}/recompute-emissivity")
def recompute_emissivity(
    run_id: int,
    image_id: int,
    hotspot_index: int = Body(..., embed=True),
    emissivity: float = Body(..., embed=True),
    reflected_apparent_temperature: float | None = Body(None, embed=True),
    user: User = Depends(require_role("admin", "inspector")),
    db: Session = Depends(get_db),
):
    """Re-derives one hotspot's max/mean temperature at a different
    emissivity than the camera used at capture time - see
    thermal_inspector.core.load_radiometric_with_emissivity for why this
    matters (bare/tarnished metal reads artificially cool at a painted-
    surface emissivity). Needs the original radiometric file, which is only
    available for images analyzed after raw_image_path started being
    stored - older runs 422 with a message explaining that instead of a
    generic 404/500.

    Deliberately does NOT recompute ambient_c: the override emissivity is
    specific to this component's material, and reapplying it across the
    whole frame would incorrectly recolor the (differently-emissive)
    surrounding enclosure too. The existing ambient_c on the row - computed
    at the camera's original setting, appropriate for the general enclosure
    surface - stays as the reference point; only the flagged region's own
    max/mean move to reflect the corrected material.

    Overwrites the hotspot row in place, same as excluding a finding does -
    everything downstream (reports, summaries) picks up the corrected
    values automatically, no separate code path needed."""
    img = db.get(AnalysisImage, image_id)
    if not img or img.run_id != run_id:
        raise HTTPException(status_code=404, detail="Image not found")
    if not img.raw_image_path:
        raise HTTPException(
            status_code=422,
            detail="This image predates raw-file storage and can't be recomputed - re-upload it to enable emissivity override.",
        )

    rows = json.loads(img.hotspots_json)
    if hotspot_index < 0 or hotspot_index >= len(rows):
        raise HTTPException(status_code=422, detail=f"hotspot_index must be in range [0, {len(rows)})")

    raw_bytes = storage.load_image(img.raw_image_path)
    if raw_bytes is None:
        raise HTTPException(status_code=422, detail="Original radiometric file is unavailable (lost from storage)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="thermal_recompute_"))
    try:
        tmp_path = tmp_dir / "raw.jpg"
        tmp_path.write_bytes(raw_bytes)
        try:
            thermogram = load_radiometric_with_emissivity(tmp_path, emissivity, reflected_apparent_temperature)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    row = rows[hotspot_index]
    x, y, w, h = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
    height, width = thermogram.temperature_c.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise HTTPException(status_code=422, detail="Hotspot bounding box is out of range for this image")
    region = thermogram.temperature_c[y : y + h, x : x + w]
    new_max = float(region.max())
    new_mean = float(region.mean())
    new_delta = new_max - row["ambient_c"]

    old_row = dict(row)
    row["max_temp_c"] = round(new_max, 2)
    row["mean_temp_c"] = round(new_mean, 2)
    row["delta_t_c"] = round(new_delta, 2)
    row["severity"] = classify_severity(new_delta)
    row["emissivity_override"] = emissivity

    rows[hotspot_index] = row
    img.hotspots_json = json.dumps(rows)
    _recompute_run_summary(img.run)
    _log_audit(
        db,
        run_id=run_id,
        image_id=image_id,
        user=user,
        action="emissivity_override",
        detail={"hotspot_index": hotspot_index, "before": old_row, "after": row},
    )
    db.commit()

    excluded = _excluded_indices(img)
    return {
        "hotspots": [{**r, "excluded": i in excluded} for i, r in enumerate(rows)],
        "run_summary": json.loads(img.run.summary_json),
    }


def _load_report_entries(run_id: int, db: Session) -> tuple[list[ImageReportEntry], list[tuple[str, str]]]:
    """Shared by the PDF and Google Docs report endpoints: loads an
    already-analyzed run's images into ImageReportEntry objects plus its
    skipped-files list, or raises the same 404/422 either way."""
    run = db.get(AnalysisRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    entries = []
    for img in run.images:
        # A missing stored image (e.g. lost from object storage) doesn't
        # abort the whole report - ImageReportEntry.annotated_image=None
        # renders as a "picture unavailable" note, findings still included.
        image_bytes = _rendered_annotated_bytes(img)
        visual_bytes = storage.load_image(img.visual_image_path) if img.visual_image_path else None
        entries.append(
            ImageReportEntry(
                image_name=img.filename,
                annotated_image=image_bytes,
                hotspot_rows=_effective_hotspot_rows(img),
                ambient_c=img.ambient_c,
                comparative_rows=json.loads(img.comparative_json) if img.comparative_json else None,
                visual_image=visual_bytes,
                note=img.visual_note,
                visual_anomaly=img.visual_anomaly,
            )
        )
    if not entries:
        raise HTTPException(status_code=422, detail="This run has no analyzable images to report on")

    skipped_pairs = [(s["filename"], s["reason"]) for s in json.loads(run.skipped_json)]
    return entries, skipped_pairs


@app.post("/api/history/{run_id}/report")
def report_from_history(
    run_id: int,
    report_style: str = Form("full"),
    report_title: str | None = Form(None),
    client_name: str | None = Form(None),
    site_location: str | None = Form(None),
    audit_date: str | None = Form(None),
    inspector_name: str | None = Form(None),
    report_id: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Build a PDF from an already-analyzed (and saved) run — no re-upload
    needed. Available to any logged-in role, including viewer."""
    if report_style not in ("full", "audit"):
        raise HTTPException(status_code=422, detail="report_style must be 'full' or 'audit'")

    entries, skipped_pairs = _load_report_entries(run_id, db)
    metadata = ReportMetadata(
        client_name=client_name,
        site_location=site_location,
        audit_date=audit_date,
        inspector_name=inspector_name,
        report_id=report_id,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="thermal_report_"))
    try:
        pdf_path = tmp_dir / "report.pdf"
        default_title = "Thermal Inspection Findings" if report_style == "audit" else "Thermal Inspection Report"
        title = report_title or default_title
        if report_style == "audit":
            generate_audit_findings_report(entries, pdf_path, title=title, excluded=skipped_pairs, metadata=metadata)
        else:
            generate_pdf_report(entries, pdf_path, title=title, skipped=skipped_pairs, metadata=metadata)

        download_name = (
            "thermal_report.pdf" if len(entries) > 1 else f"{Path(entries[0].image_name).stem}_report.pdf"
        )
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=download_name,
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


@app.post("/api/history/{run_id}/report/gdoc")
def gdoc_report_from_history(
    run_id: int,
    report_style: str = Form("full"),
    report_title: str | None = Form(None),
    client_name: str | None = Form(None),
    site_location: str | None = Form(None),
    audit_date: str | None = Form(None),
    inspector_name: str | None = Form(None),
    report_id: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Same as report_from_history, but exports to a Google Doc in the
    configured Shared Drive folder instead of a downloaded PDF, returning
    {"doc_url": ...}. 501s if Google credentials aren't configured
    (GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_DRIVE_FOLDER_ID unset) so local dev
    without them keeps working."""
    if not gdocs.configured():
        raise HTTPException(
            status_code=501,
            detail="Google Docs export isn't configured on this server (missing GOOGLE_SERVICE_ACCOUNT_JSON/GOOGLE_DRIVE_FOLDER_ID)",
        )
    if report_style not in ("full", "audit"):
        raise HTTPException(status_code=422, detail="report_style must be 'full' or 'audit'")

    entries, skipped_pairs = _load_report_entries(run_id, db)
    metadata = ReportMetadata(
        client_name=client_name,
        site_location=site_location,
        audit_date=audit_date,
        inspector_name=inspector_name,
        report_id=report_id,
    )
    default_title = "Thermal Inspection Findings" if report_style == "audit" else "Thermal Inspection Report"

    doc_url = build_findings_doc(
        gdocs.docs_service(),
        gdocs.drive_service(),
        entries,
        gdocs.target_folder_id(),
        style=report_style,
        title=report_title or default_title,
        excluded=skipped_pairs,
        metadata=metadata,
    )
    return {"doc_url": doc_url}


# ---------------------------------------------------------------------------
# History (all roles, read-only)
# ---------------------------------------------------------------------------


@app.get("/api/history")
def list_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    runs = db.query(AnalysisRun).order_by(AnalysisRun.id.desc()).limit(200).all()
    return {"runs": [_run_summary(r) for r in runs]}


@app.get("/api/history/{run_id}")
def get_history_run(run_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.get(AnalysisRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": run.id,
        "created_at": run.created_at.isoformat(),
        "created_by": run.created_by_username,
        "ambient_c": run.ambient_c,
        "min_delta_c": run.min_delta_c,
        "min_area_px": run.min_area_px,
        "roi": [run.roi_x, run.roi_y, run.roi_w, run.roi_h] if run.roi_x is not None else None,
        "load_percent": run.load_percent,
        "summary": json.loads(run.summary_json),
        "skipped": json.loads(run.skipped_json),
        "images": [
            {
                "id": img.id,
                "filename": img.filename,
                "ambient_c": round(img.ambient_c, 2),
                "hotspots": [
                    {**row, "excluded": i in _excluded_indices(img)}
                    for i, row in enumerate(json.loads(img.hotspots_json))
                ],
                "comparative": json.loads(img.comparative_json) if img.comparative_json else None,
                "image_url": f"/api/history/{run.id}/image/{img.id}",
                "photo_url": f"/api/history/{run.id}/photo/{img.id}" if img.visual_image_path else None,
                "image_updates_on_exclude": img.has_unannotated_base,
                "visual_note": img.visual_note,
                "visual_anomaly": img.visual_anomaly,
                "can_recompute_emissivity": img.raw_image_path is not None,
            }
            for img in run.images
        ],
    }


@app.get("/api/history/{run_id}/audit-log")
def get_audit_log(run_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Who changed what on this run, and when - available to any logged-in
    role, same as browsing the run itself. Newest first."""
    if not db.get(AnalysisRun, run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    entries = (
        db.query(AuditLogEntry)
        .filter(AuditLogEntry.run_id == run_id)
        .order_by(AuditLogEntry.created_at.desc())
        .all()
    )
    return {
        "entries": [
            {
                "id": e.id,
                "image_id": e.image_id,
                "username": e.username,
                "action": e.action,
                "detail": json.loads(e.detail_json),
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ]
    }


@app.get("/api/history/{run_id}/image/{image_id}")
def get_history_image(run_id: int, image_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    img = db.get(AnalysisImage, image_id)
    if not img or img.run_id != run_id:
        raise HTTPException(status_code=404, detail="Image not found")
    image_bytes = _rendered_annotated_bytes(img) or _placeholder_image_bytes("Image unavailable - lost from storage")
    return Response(content=image_bytes, media_type="image/png")


@app.get("/api/history/{run_id}/photo/{image_id}")
def get_history_photo(run_id: int, image_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    img = db.get(AnalysisImage, image_id)
    if not img or img.run_id != run_id or not img.visual_image_path:
        raise HTTPException(status_code=404, detail="Photo not found")
    photo_bytes = storage.load_image(img.visual_image_path) or _placeholder_image_bytes("Photo unavailable - lost from storage")
    return Response(content=photo_bytes, media_type="image/png")


# ---------------------------------------------------------------------------
# User management (admin only)
# ---------------------------------------------------------------------------


@app.get("/api/users")
def list_users(admin: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    return {"users": [_user_summary(u) for u in db.query(User).order_by(User.id).all()]}


@app.post("/api/users")
def create_user(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    username = username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="Username required")
    if role not in ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {ROLES}")
    if len(password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")
    if db.query(User).filter_by(username=username).first():
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(username=username, password_hash=hash_password(password), role=role, is_active=True)
    db.add(user)
    db.commit()
    return _user_summary(user)


@app.post("/api/users/{user_id}/role")
def set_user_role(
    user_id: int,
    role: str = Form(...),
    admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    if role not in ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {ROLES}")
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == admin.id and role != "admin":
        raise HTTPException(status_code=400, detail="Cannot remove your own admin role")
    target.role = role
    db.commit()
    return _user_summary(target)


@app.post("/api/users/{user_id}/active")
def set_user_active(
    user_id: int,
    active: bool = Form(...),
    admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == admin.id and not active:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    target.is_active = active
    db.commit()
    return _user_summary(target)
