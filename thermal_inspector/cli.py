from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import process_images
from .pdf_report import ReportMetadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect and classify thermal anomalies in electrical panel/system images."
    )
    parser.add_argument("input", help="Radiometric FLIR JPEG, or a folder of them")
    parser.add_argument("-o", "--outdir", default="thermal_report", help="Output directory")
    parser.add_argument(
        "--ambient",
        type=float,
        default=None,
        help="Reference/ambient temperature in Celsius. If omitted, estimated per-image "
        "as the 25th percentile of that image's temperatures.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=8.0,
        help="Minimum temperature rise above ambient (C) to flag as a hotspot (default: 8.0)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="Minimum hotspot region size in pixels, filters out sensor noise (default: 25)",
    )
    parser.add_argument(
        "--use-visual",
        action="store_true",
        help="Draw annotations over the camera's embedded optical photo instead of the "
        "colorized temperature map, when the file has one",
    )
    parser.add_argument(
        "--roi",
        default=None,
        metavar="X,Y,W,H",
        help="Restrict ambient estimation and hotspot search to a pixel rectangle "
        "(e.g. the panel enclosure), excluding background like walls or open doors. "
        "Applied to every image if --input is a folder, so only use it when all "
        "images share the same framing.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a client-facing PDF report (outdir/report.pdf) with the "
        "annotated image and hotspot table for each analyzed file.",
    )
    parser.add_argument(
        "--pdf-title",
        default="Thermal Inspection Report",
        help="Title shown at the top of the PDF report",
    )
    parser.add_argument(
        "--report-style",
        choices=["full", "audit"],
        default="full",
        help="'full' (default) includes every image, ambient reference, and notes - an "
        "investigation record. 'audit' lists only findings (clean images get a one-line "
        "compliant note), with no methodology/notes text - for audit deliverables. Note: "
        "'audit' reports whatever the detector flagged for the given thresholds/ROI as-is; "
        "it can't exclude a specific region later judged to be a camera artifact rather "
        "than a real issue - do that via generate_audit_findings_report() directly.",
    )
    parser.add_argument(
        "--notes-file",
        default=None,
        metavar="PATH",
        help="Path to a JSON file mapping image filename to a freeform note "
        "(e.g. field judgment that a flagged region is a reflection, not a "
        "fault). Appended under that image's table in the PDF report.",
    )
    parser.add_argument("--client", default=None, help="Client name, shown in the PDF report header")
    parser.add_argument("--site", default=None, help="Site/location, shown in the PDF report header")
    parser.add_argument(
        "--audit-date",
        default=None,
        help="Date the audit/inspection was performed (e.g. 2026-08-03). "
        "Defaults to today if omitted. Distinct from the report generation timestamp.",
    )
    parser.add_argument("--inspector", default=None, help="Inspector/auditor name, shown in the PDF report header")
    parser.add_argument("--report-id", default=None, help="Report ID/reference number, shown in the PDF report header")

    args = parser.parse_args(argv)

    roi = None
    if args.roi is not None:
        try:
            parts = [int(p.strip()) for p in args.roi.split(",")]
            if len(parts) != 4:
                raise ValueError
            roi = tuple(parts)
        except ValueError:
            print(f"error: --roi must be four integers as X,Y,W,H, got {args.roi!r}", file=sys.stderr)
            return 1

    notes = None
    if args.notes_file is not None:
        try:
            notes = json.loads(Path(args.notes_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: could not read --notes-file {args.notes_file!r}: {exc}", file=sys.stderr)
            return 1

    metadata = ReportMetadata(
        client_name=args.client,
        site_location=args.site,
        audit_date=args.audit_date,
        inspector_name=args.inspector,
        report_id=args.report_id,
    )

    try:
        summary = process_images(
            args.input,
            args.outdir,
            ambient_c=args.ambient,
            min_delta_c=args.min_delta,
            min_area_px=args.min_area,
            use_visual=args.use_visual,
            roi=roi,
            pdf=args.pdf,
            notes=notes,
            pdf_title=args.pdf_title,
            report_style=args.report_style,
            metadata=metadata,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    print(f"\nAnnotated images and reports written to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
