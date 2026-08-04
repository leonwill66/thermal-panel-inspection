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

    scale = max(1, -(-min_display_width // base.shape[1])) if min_display_width else 1
    if scale > 1:
        base = cv2.resize(
            base, (base.shape[1] * scale, base.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )

    def scaled(*vals: int) -> tuple[int, ...]:
        return tuple(int(round(v * scale)) for v in vals)

    if roi is not None:
        rx, ry, rw, rh = scaled(*roi)
        cv2.rectangle(base, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1, cv2.LINE_AA)

    for h in hotspots:
        x, y, w, h_px = scaled(*h.bbox)
        color = SEVERITY_COLOR_BGR.get(h.severity, (255, 255, 255))
        cv2.rectangle(base, (x, y), (x + w, y + h_px), color, 2)

        label = f"{h.severity} dT={h.delta_t_c:.1f}C max={h.max_temp_c:.1f}C"
        label_y = y - 8 if y - 8 > 14 else y + h_px + 20
        cv2.putText(base, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return base
