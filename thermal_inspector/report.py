from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import Hotspot

FIELDNAMES = [
    "image",
    "severity",
    "delta_t_c",
    "max_temp_c",
    "mean_temp_c",
    "ambient_c",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area_px",
]


def hotspots_to_rows(image_name: str, hotspots: list[Hotspot]) -> list[dict]:
    rows = []
    for h in hotspots:
        x, y, w, hh = h.bbox
        rows.append(
            {
                "image": image_name,
                "severity": h.severity,
                "delta_t_c": round(h.delta_t_c, 2),
                "max_temp_c": round(h.max_temp_c, 2),
                "mean_temp_c": round(h.mean_temp_c, 2),
                "ambient_c": round(h.ambient_c, 2),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": hh,
                "area_px": h.area_px,
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def summarize(rows: list[dict]) -> dict:
    by_severity: dict[str, int] = {}
    for row in rows:
        by_severity[row["severity"]] = by_severity.get(row["severity"], 0) + 1

    worst = max(rows, key=lambda r: r["delta_t_c"], default=None)
    images_with_hotspots = len({row["image"] for row in rows})

    return {
        "total_hotspots": len(rows),
        "images_with_hotspots": images_with_hotspots,
        "counts_by_severity": by_severity,
        "worst_hotspot": worst,
    }
