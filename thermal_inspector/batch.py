from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from .core import find_hotspots, find_comparative_anomalies, load_radiometric
from .report import comparative_to_rows, hotspots_to_rows, summarize, write_csv, write_json
from .annotate import annotate_image
from .pdf_report import (
    ImageReportEntry,
    ReportMetadata,
    generate_pdf_report,
    generate_audit_findings_report,
)

DEFAULT_EXTENSIONS = (".jpg", ".jpeg")


def iter_images(input_path: Path, extensions=DEFAULT_EXTENSIONS) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in extensions)


def process_images(
    input_path: str | Path,
    outdir: str | Path,
    ambient_c: float | None = None,
    min_delta_c: float = 8.0,
    min_area_px: int = 25,
    use_visual: bool = False,
    roi: tuple[int, int, int, int] | None = None,
    pdf: bool = False,
    notes: dict[str, str] | None = None,
    pdf_title: str = "Thermal Inspection Report",
    report_style: str = "full",
    metadata: ReportMetadata | None = None,
    load_percent: float | None = None,
    compare_regions: list[tuple[tuple[int, int, int, int], str]] | None = None,
) -> dict:
    """Run hotspot detection over a single image or a folder, writing annotated
    images plus a combined CSV/JSON report to outdir. Returns the summary dict.

    Per-file failures (unreadable/non-radiometric images) are logged to stderr
    and skipped rather than aborting the whole batch. roi, if given, is applied
    to every image in the batch in the same pixel coordinates — only sensible
    when all images share the same framing/resolution (e.g. a fixed camera
    position across a series). compare_regions carries the same caveat - it's
    a list of (bbox, label) pairs applied to every image, so only meaningful
    when they share identical framing (e.g. a fixed camera position, or a
    single image passed as input_path).

    load_percent, if given, is passed straight to find_hotspots() to
    severity-classify each hotspot by its load-corrected ΔT rather than the
    raw observed one (see core.load_adjusted_delta_t).

    If pdf is True, also writes outdir/report.pdf, in one of two styles:
    - report_style="full" (default): every image, ambient reference, full
      hotspot detail, and any per-image notes — an investigation record.
    - report_style="audit": only images with findings get a section (clean
      images are listed as a one-line compliant note), no ambient/methodology
      text, and per-image notes are omitted. This reports exactly what the
      detector found for the given thresholds/ROI; it can't know that a
      specific flagged region is a camera artifact rather than a real issue
      (that's a human judgment call) — exclude such regions by calling
      generate_audit_findings_report directly with filtered hotspot_rows.

    notes optionally maps an image filename to a freeform annotation (e.g.
    "likely reflection off bare metal, verify from another angle") appended
    under that image's table — only used by report_style="full".
    """
    if report_style not in ("full", "audit"):
        raise ValueError(f"report_style must be 'full' or 'audit', got {report_style!r}")
    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    notes = notes or {}

    images = iter_images(input_path)
    if not images:
        raise FileNotFoundError(f"No radiometric images found under {input_path}")

    all_rows: list[dict] = []
    pdf_entries: list[ImageReportEntry] = []
    skipped: list[tuple[str, str]] = []
    for img_path in images:
        try:
            thermogram = load_radiometric(img_path)
            hotspots, ambient_used = find_hotspots(
                thermogram.temperature_c,
                ambient_c=ambient_c,
                min_delta_c=min_delta_c,
                min_area_px=min_area_px,
                roi=roi,
                load_percent=load_percent,
            )
        except Exception as exc:
            print(f"[skip] {img_path.name}: {exc}", file=sys.stderr)
            skipped.append((img_path.name, str(exc)))
            continue

        rows = hotspots_to_rows(img_path.name, hotspots)
        all_rows.extend(rows)

        comparative_rows = None
        if compare_regions:
            comparative = find_comparative_anomalies(thermogram.temperature_c, compare_regions)
            comparative_rows = comparative_to_rows(img_path.name, comparative)

        annotated = annotate_image(thermogram, hotspots, use_visual=use_visual, roi=roi)
        annotated_path = outdir / f"{img_path.stem}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated)

        pdf_entries.append(
            ImageReportEntry(
                image_name=img_path.name,
                annotated_image=annotated_path,
                hotspot_rows=rows,
                ambient_c=ambient_used,
                note=notes.get(img_path.name),
                comparative_rows=comparative_rows,
            )
        )

    write_csv(all_rows, outdir / "report.csv")
    write_json(all_rows, outdir / "report.json")

    summary = summarize(all_rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if pdf:
        if report_style == "audit":
            generate_audit_findings_report(
                pdf_entries, outdir / "report.pdf", title=pdf_title, excluded=skipped, metadata=metadata
            )
        else:
            generate_pdf_report(
                pdf_entries, outdir / "report.pdf", title=pdf_title, skipped=skipped, metadata=metadata
            )

    return summary
