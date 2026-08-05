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

# Below this load %, the I^2R correction (see load_adjusted_delta_t) is
# considered unreliable and callers should warn rather than silently trust
# the corrected number - convective/radiative heat loss doesn't scale
# cleanly enough at very light load for the square-law approximation to hold.
MIN_RELIABLE_LOAD_PERCENT = 40.0


@dataclass
class Hotspot:
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixel coords
    centroid: tuple[float, float]
    max_temp_c: float
    mean_temp_c: float
    ambient_c: float
    delta_t_c: float  # as observed, at whatever load was present during inspection
    area_px: int
    severity: str
    load_percent: Optional[float] = None  # load at inspection time, if supplied
    delta_t_corrected_c: Optional[float] = None  # delta_t_c projected to 100% load


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


def load_adjusted_delta_t(delta_t_observed_c: float, load_percent: float) -> float:
    """Projects an observed temperature rise to what it would likely be at
    100% rated load, using the standard I^2R approximation that resistive
    heating - and so temperature rise above ambient - scales with the square
    of load current: delta_T_corrected = delta_T_observed * (100 / load%)^2.

    This is standard practice for evaluating a finding captured while
    equipment wasn't at full load (a real fault can look deceptively mild
    when lightly loaded, and vice versa) - see NETA/IR-inspection guidance
    on load correction. It is only a reasonable approximation - convective
    and radiative losses don't scale as cleanly at light load, so treat
    corrected values below MIN_RELIABLE_LOAD_PERCENT load with caution
    rather than as a precise prediction.
    """
    if load_percent <= 0:
        raise ValueError(f"load_percent must be > 0, got {load_percent}")
    return delta_t_observed_c * (100.0 / load_percent) ** 2


def find_hotspots(
    temp_c: np.ndarray,
    ambient_c: Optional[float] = None,
    min_delta_c: float = 8.0,
    min_area_px: int = 25,
    thresholds=DEFAULT_THRESHOLDS,
    roi: Optional[tuple[int, int, int, int]] = None,
    load_percent: Optional[float] = None,
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

    load_percent, if given, is the equipment's load at inspection time as a
    percent of rated (e.g. 60.0 for 60% load). Detection itself still uses
    the raw observed delta-T (min_delta_c) - load correction only affects how
    each found hotspot is severity-classified, via load_adjusted_delta_t().
    Every Hotspot keeps its raw delta_t_c regardless; delta_t_corrected_c is
    only populated when load_percent is supplied.
    """
    if load_percent is not None and load_percent <= 0:
        raise ValueError(f"load_percent must be > 0, got {load_percent}")

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

        delta_t_corrected = load_adjusted_delta_t(delta_t, load_percent) if load_percent else None
        severity_delta = delta_t_corrected if delta_t_corrected is not None else delta_t

        hotspots.append(
            Hotspot(
                bbox=(x, y, w, h),
                centroid=(float(centroids[i][0]) + rx, float(centroids[i][1]) + ry),
                max_temp_c=max_t,
                mean_temp_c=mean_t,
                ambient_c=ambient_c,
                delta_t_c=delta_t,
                area_px=area,
                severity=classify_severity(severity_delta, thresholds),
                load_percent=load_percent,
                delta_t_corrected_c=delta_t_corrected,
            )
        )

    hotspots.sort(key=lambda h: h.delta_t_corrected_c if h.delta_t_corrected_c is not None else h.delta_t_c, reverse=True)
    return hotspots, ambient_c


# (delta_t_cutoff_celsius, severity_label) for the comparative method - a
# different scale than DEFAULT_THRESHOLDS, since components being compared
# against each other under similar load should normally read very close to
# identical, so even a few degrees of difference is meaningful (unlike
# ambient-referenced ΔT, where enclosure heating alone commonly accounts for
# a much larger gap). Commonly cited IR-inspection comparative-method bands.
COMPARATIVE_THRESHOLDS = [
    (15.0, "comparative_major"),     # major discrepancy, repair immediately
    (4.0, "comparative_probable"),   # probable deficiency, repair as time permits
    (1.0, "comparative_possible"),   # possible deficiency, warrants investigation
]


@dataclass
class ComparativeAnomaly:
    label: str
    bbox: tuple[int, int, int, int]
    max_temp_c: float
    delta_t_c: float  # vs. the coolest region in the group
    severity: Optional[str]  # None if within normal range of its peers


def classify_comparative_severity(delta_t_c: float, thresholds=COMPARATIVE_THRESHOLDS) -> Optional[str]:
    for cutoff, label in thresholds:
        if delta_t_c >= cutoff:
            return label
    return None  # below the lowest threshold - no comparative concern


def find_comparative_anomalies(
    temp_c: np.ndarray,
    regions: list[tuple[tuple[int, int, int, int], str]],
    thresholds=COMPARATIVE_THRESHOLDS,
) -> list[ComparativeAnomaly]:
    """Compares corresponding components (e.g. three-phase breakers, or any
    set of components expected to run at similar temperature under similar
    load) against each other, rather than each in isolation against an
    estimated ambient. This is the NETA-preferred "comparative method" -
    it catches subtler faults that a single component's absolute deviation
    from ambient can miss, and it's far less prone to the background/
    reflection false positives the ambient-referenced method is vulnerable
    to, since every region compared is one you've deliberately marked as an
    equivalent component, not "whatever happens to be warm in frame."

    regions is a list of (bbox, label) pairs - e.g.
    [((10,10,20,20), "Phase A"), ((40,10,20,20), "Phase B"), ...] - and needs
    at least 2. The coolest region's peak temperature is used as the
    reference point (the presumed-healthy member of the group, assuming
    roughly balanced load); every region's delta_t_c is measured against
    that reference, not a global ambient estimate.
    """
    if len(regions) < 2:
        raise ValueError("find_comparative_anomalies needs at least 2 regions to compare")

    height, width = temp_c.shape[:2]
    max_temps = []
    for bbox, label in regions:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"region {label!r} width/height must be positive, got {bbox}")
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"region {label!r} {bbox} falls outside the {width}x{height} frame")
        max_temps.append(float(temp_c[y : y + h, x : x + w].max()))

    baseline = min(max_temps)

    anomalies = [
        ComparativeAnomaly(
            label=label,
            bbox=bbox,
            max_temp_c=max_t,
            delta_t_c=max_t - baseline,
            severity=classify_comparative_severity(max_t - baseline, thresholds),
        )
        for (bbox, label), max_t in zip(regions, max_temps)
    ]
    anomalies.sort(key=lambda a: a.delta_t_c, reverse=True)
    return anomalies
