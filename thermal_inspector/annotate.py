from __future__ import annotations

import numpy as np
import cv2

from .core import Hotspot, Thermogram

SEVERITY_COLOR_BGR = {
    "minor": (0, 220, 220),        # yellow
    "serious": (0, 140, 255),      # orange
    "critical": (0, 0, 255),       # red
    "critical_immediate": (255, 0, 255),  # magenta
}


def _colorize_temperature(temp_c: np.ndarray) -> np.ndarray:
    t_min, t_max = float(temp_c.min()), float(temp_c.max())
    span = max(t_max - t_min, 1e-6)
    normalized = np.clip((temp_c - t_min) / span * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def compute_scale(width: int, min_display_width: int = 640) -> int:
    """Nearest-neighbor upscale factor needed to bring an image `width` px
    wide up to at least min_display_width. Radiometric FLIR frames are often
    tiny (e.g. 160x120), too small for fixed-size label text to fit without
    overlapping or running off the edge; pass 0 for min_display_width to
    disable (factor 1)."""
    return max(1, -(-min_display_width // width)) if min_display_width else 1


def _draw_box(base: np.ndarray, x: int, y: int, w: int, h: int, severity: str, label: str, scale: int) -> None:
    x, y, w, h = (int(round(v * scale)) for v in (x, y, w, h))
    color = SEVERITY_COLOR_BGR.get(severity, (255, 255, 255))
    cv2.rectangle(base, (x, y), (x + w, y + h), color, 2)
    label_y = y - 8 if y - 8 > 14 else y + h + 20
    cv2.putText(base, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_hotspot_rows(base: np.ndarray, rows: list[dict], scale: int = 1) -> np.ndarray:
    """Draws hotspot boxes onto a copy of an already-colorized/upscaled base
    image, from row dicts shaped like thermal_inspector.report.hotspots_to_rows
    output rather than Hotspot dataclasses + a live Thermogram. Lets a caller
    redraw a stored base image with a possibly-filtered set of hotspots (e.g.
    after a reviewer excludes false positives - a tool left in frame, bare
    reflective metal) without re-running detection on the original file.
    scale must match whatever compute_scale() returned when the base was
    upscaled, so boxes land in the right place."""
    base = base.copy()
    for row in rows:
        label = f"{row['severity']} dT={row['delta_t_c']:.1f}C max={row['max_temp_c']:.1f}C"
        _draw_box(base, row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"], row["severity"], label, scale)
    return base


def annotate_image(
    thermogram: Thermogram,
    hotspots: list[Hotspot],
    use_visual: bool = False,
    roi: tuple[int, int, int, int] | None = None,
    min_display_width: int = 640,
) -> np.ndarray:
    """Return a BGR image with hotspots outlined and labeled by severity.

    By default draws over a colorized version of the temperature data itself
    (always available). Pass use_visual=True to draw over the camera's
    embedded optical photo instead, when present. If roi was used for
    detection, pass it here too so the searched area is outlined for context.

    Radiometric FLIR frames are often tiny (e.g. 160x120), too small for
    fixed-size label text to fit without overlapping or running off the edge.
    The base image is upscaled with nearest-neighbor interpolation (preserving
    the native thermal pixel blockiness rather than blurring it) until it's at
    least min_display_width wide before anything is drawn; pass 0 to disable.
    """
    if use_visual and thermogram.visual is not None:
        base = thermogram.visual.copy()
        if base.shape[:2] != thermogram.temperature_c.shape[:2]:
            base = cv2.resize(base, (thermogram.temperature_c.shape[1], thermogram.temperature_c.shape[0]))
    else:
        base = _colorize_temperature(thermogram.temperature_c)

    scale = compute_scale(base.shape[1], min_display_width)
    if scale > 1:
        base = cv2.resize(
            base, (base.shape[1] * scale, base.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )

    if roi is not None:
        rx, ry, rw, rh = (int(round(v * scale)) for v in roi)
        cv2.rectangle(base, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1, cv2.LINE_AA)

    for h in hotspots:
        x, y, w, h_px = h.bbox
        label = f"{h.severity} dT={h.delta_t_c:.1f}C max={h.max_temp_c:.1f}C"
        _draw_box(base, x, y, w, h_px, h.severity, label, scale)

    return base
