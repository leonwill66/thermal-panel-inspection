"""Radiometric loading and hotspot detection for electrical-panel thermography.

Severity thresholds follow the common IR-inspection convention used in NETA/NFPA 70B
style guidance: classify anomalies by temperature rise (delta-T) above a reference
("ambient") temperature rather than by absolute temperature, since panels run hot
under normal load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

try:
    import flyr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'flyr' package is required to read radiometric FLIR images. "
        "Install it with: pip install flyr"
    ) from exc


# (delta_t_cutoff_celsius, severity_label) — first matching cutoff (highest first) wins.
DEFAULT_THRESHOLDS = [
    (40.0, "critical_immediate"),  # imminent failure risk, repair now
    (20.0, "critical"),            # major discrepancy, repair as soon as possible
    (10.0, "serious"),             # probable deficiency, schedule repair
    (0.0, "minor"),                # possible deficiency, monitor
]


@dataclass
class Hotspot:
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixel coords
    centroid: tuple[float, float]
    max_temp_c: float
    mean_temp_c: float
    ambient_c: float
    delta_t_c: float
    area_px: int
    severity: str


@dataclass
class Thermogram:
    source_path: Path
    temperature_c: np.ndarray  # HxW float array
    visual: Optional[np.ndarray] = field(default=None)  # embedded optical photo, BGR, if present


def load_radiometric(path: str | Path) -> Thermogram:
    """Extract the per-pixel temperature array from a radiometric FLIR JPEG.

    Raises FileNotFoundError if the path doesn't exist, and ValueError if the
    file has no embedded radiometric data (e.g. it's a plain photo, or the
    camera's raw format needs exiftool on PATH and extraction failed).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    try:
        thermogram = flyr.unpack(str(path))
    except KeyError:
        # FLIR cameras often export a paired non-radiometric visual-only photo
        # alongside each thermal capture (same APP1 FLIR block, but missing the
        # raw-data record) — this is the most common cause of this failure.
        raise ValueError(
            f"{path.name} has FLIR metadata but no embedded radiometric data. "
            "This usually means it's the camera's paired visual-only photo "
            "rather than the thermal capture - check for a companion file."
        )
    except Exception as exc:
        raise ValueError(
            f"Could not extract radiometric data from {path.name}: {exc}. "
            "Confirm this is a radiometric FLIR JPEG (not a plain photo), and "
            "that 'exiftool' is on PATH if your camera model requires it."
        ) from exc

    temp_c = thermogram.celsius.astype(np.float32)

    visual = None
    try:
        optical = thermogram.optical
        if optical is not None:
            visual = cv2.cvtColor(np.array(optical), cv2.COLOR_RGB2BGR)
    except Exception:
        visual = None

    return Thermogram(source_path=path, temperature_c=temp_c, visual=visual)


def classify_severity(delta_t_c: float, thresholds=DEFAULT_THRESHOLDS) -> str:
    for cutoff, label in thresholds:
        if delta_t_c >= cutoff:
            return label
    return thresholds[-1][1]


def find_hotspots(
    temp_c: np.ndarray,
    ambient_c: Optional[float] = None,
    min_delta_c: float = 8.0,
    min_area_px: int = 25,
    thresholds=DEFAULT_THRESHOLDS,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> tuple[list[Hotspot], float]:
    """Find anomalously hot regions in a temperature array.

    If ambient_c isn't supplied, it's estimated as the 25th percentile of the
    search area — a robust stand-in for "normal" background temperature, since
    most of a panel image is typically enclosure/backplane rather than an
    active hotspot. Pass an explicit ambient_c (e.g. measured off a known-good
    reference component) for more accurate comparative-method results.

    roi restricts both the ambient estimate and the hotspot search to a
    sub-rectangle (x, y, w, h) in pixel coordinates, so that background scenery
    outside the panel enclosure (a warm wall, an open door, another panel)
    isn't mistaken for a hotspot or skewed into the ambient baseline. Returned
    hotspot bboxes are still in the full frame's coordinate space. Omit roi to
    search the whole frame, as before.
    """
    height, width = temp_c.shape[:2]
    if roi is None:
        rx, ry, rw, rh = 0, 0, width, height
    else:
        rx, ry, rw, rh = roi
        if rw <= 0 or rh <= 0:
            raise ValueError(f"roi width/height must be positive, got {roi}")
        if rx < 0 or ry < 0 or rx + rw > width or ry + rh > height:
            raise ValueError(
                f"roi {roi} falls outside the {width}x{height} frame"
            )

    search_area = temp_c[ry : ry + rh, rx : rx + rw]

    if ambient_c is None:
        ambient_c = float(np.percentile(search_area, 25))

    delta = search_area - ambient_c
    mask = (delta >= min_delta_c).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    hotspots: list[Hotspot] = []
    for i in range(1, num_labels):  # label 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT]) + rx
        y = int(stats[i, cv2.CC_STAT_TOP]) + ry
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        region_temps = search_area[labels == i]
        max_t = float(region_temps.max())
        mean_t = float(region_temps.mean())
        delta_t = max_t - ambient_c

        hotspots.append(
            Hotspot(
                bbox=(x, y, w, h),
                centroid=(float(centroids[i][0]) + rx, float(centroids[i][1]) + ry),
                max_temp_c=max_t,
                mean_temp_c=mean_t,
                ambient_c=ambient_c,
                delta_t_c=delta_t,
                area_px=area,
                severity=classify_severity(delta_t, thresholds),
            )
        )

    hotspots.sort(key=lambda h: h.delta_t_c, reverse=True)
    return hotspots, ambient_c
