"""Renders a batch analysis run into a client-facing PDF inspection report."""

from __future__ import annotations

import datetime
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SEVERITY_ORDER = ["critical_immediate", "critical", "serious", "minor"]
SEVERITY_LABELS = {
    "critical_immediate": "Critical - repair immediately",
    "critical": "Critical - repair ASAP",
    "serious": "Serious - schedule repair",
    "minor": "Minor - monitor",
}
SEVERITY_ROW_COLORS = {
    "minor": colors.HexColor("#FFF6DA"),
    "serious": colors.HexColor("#FFE8D1"),
    "critical": colors.HexColor("#FFDADA"),
    "critical_immediate": colors.HexColor("#FADAFF"),
}
SEVERITY_CONSEQUENCE = {
    "critical_immediate": (
        "an imminent risk of electrical component failure - potentially arcing or fire - "
        "and a safety hazard"
    ),
    "critical": "a high risk of accelerated electrical damage and unplanned equipment failure",
    "serious": (
        "a developing electrical fault - commonly a loose, corroded, or overloaded connection "
        "- that will likely worsen without intervention"
    ),
    "minor": "an early-stage electrical deviation worth tracking before it develops into a larger fault",
}
SEVERITY_ACTION = {
    "critical_immediate": (
        "we recommend the affected component be de-energized and repaired without delay"
    ),
    "critical": "we recommend corrective action as soon as possible, ideally within days",
    "serious": "we recommend corrective action be scheduled in the near term",
    "minor": "we recommend continued monitoring, with corrective action planned if the condition persists or worsens",
}

COMPARATIVE_LABELS = {
    "comparative_major": "Major discrepancy vs. peers - repair immediately",
    "comparative_probable": "Probable deficiency vs. peers - repair as time permits",
    "comparative_possible": "Possible deficiency vs. peers - investigate",
}
COMPARATIVE_ROW_COLORS = {
    "comparative_possible": colors.HexColor("#FFF6DA"),
    "comparative_probable": colors.HexColor("#FFE8D1"),
    "comparative_major": colors.HexColor("#FFDADA"),
}
COMPARATIVE_ACTION = {
    "comparative_major": "we recommend immediate repair - this component is running well outside the range of its peers",
    "comparative_probable": "we recommend repair as time permits",
    "comparative_possible": "we recommend investigating this component at the next opportunity",
}


@dataclass
class ImageReportEntry:
    image_name: str
    annotated_image: Path | bytes  # local file path, or raw PNG bytes (e.g. loaded from object storage)
    hotspot_rows: list[dict]  # as produced by thermal_inspector.report.hotspots_to_rows
    ambient_c: float
    note: str | None = None
    comparative_rows: list[dict] | None = None  # as produced by thermal_inspector.report.comparative_to_rows
    visual_image: Path | bytes | None = None  # camera's embedded optical photo, unannotated, if present


@dataclass
class ReportMetadata:
    client_name: str | None = None
    site_location: str | None = None
    audit_date: str | None = None  # e.g. "2026-08-03"; defaults to today if unset
    inspector_name: str | None = None
    report_id: str | None = None


def _metadata_block(metadata: ReportMetadata | None, styles) -> list:
    metadata = metadata or ReportMetadata()
    fields = [
        ("Client", metadata.client_name),
        ("Site / Location", metadata.site_location),
        ("Audit Date", metadata.audit_date or datetime.date.today().isoformat()),
        ("Inspector", metadata.inspector_name),
        ("Report ID", metadata.report_id),
    ]
    story = []
    for label, value in fields:
        if value:
            story.append(Paragraph(f"<b>{label}:</b> {value}", styles["Normal"]))
    return story


def _hotspot_table(rows: list[dict], columns: list[tuple[str, str]], row_colors_map=None) -> Table:
    """columns is a list of (row-dict key, header label); float values are
    formatted to 1 decimal place, None displays as "-", everything else is
    stringified as-is. row_colors_map defaults to the ambient-referenced
    SEVERITY_ROW_COLORS; pass COMPARATIVE_ROW_COLORS for comparative-method rows."""
    row_colors_map = row_colors_map if row_colors_map is not None else SEVERITY_ROW_COLORS
    header = [label for _, label in columns]
    data = [header]
    row_colors = [colors.white]
    for row in rows:
        data_row = []
        for key, _ in columns:
            value = row[key]
            if value is None:
                data_row.append("-")
            elif isinstance(value, float):
                data_row.append(f"{value:.1f}")
            else:
                data_row.append(str(value))
        data.append(data_row)
        row_colors.append(row_colors_map.get(row["severity"], colors.white))

    table = Table(data, hAlign="LEFT")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2A2F37")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, c in enumerate(row_colors):
        if i == 0:
            continue
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), c))
    table.setStyle(TableStyle(style_cmds))
    return table


def _narrative_summary(all_rows: list[dict]) -> str:
    """A short plain-English paragraph summarizing the findings, for readers
    who want the takeaway without parsing the technical tables. all_rows is
    every hotspot row across every image (each already carries an "image"
    key, per thermal_inspector.report.hotspots_to_rows)."""
    if not all_rows:
        return (
            "No thermal anomalies were identified in this inspection. All panels "
            "reviewed appear to be operating within normal thermal parameters."
        )

    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["severity"]] = counts.get(row["severity"], 0) + 1
    total = len(all_rows)

    breakdown = ", ".join(
        f"{counts[sev]} {sev.replace('_', ' ')}" for sev in SEVERITY_ORDER if counts.get(sev)
    )

    def _rank(row: dict) -> float:
        corrected = row.get("delta_t_corrected_c")
        return corrected if corrected is not None else row["delta_t_c"]

    worst = max(all_rows, key=_rank)
    worst_consequence = SEVERITY_CONSEQUENCE[worst["severity"]]
    worst_action = SEVERITY_ACTION[worst["severity"]]

    if worst.get("delta_t_corrected_c") is not None:
        finding_sentence = (
            f"The most significant finding, on {worst['image']}, shows a temperature rise of "
            f"{worst['delta_t_corrected_c']:.1f}°C above ambient when corrected to full load "
            f"(observed {worst['delta_t_c']:.1f}°C at {worst['load_percent']:.0f}% load), consistent "
            f"with {worst_consequence}. {worst_action[0].upper()}{worst_action[1:]}."
        )
    else:
        finding_sentence = (
            f"The most significant finding, on {worst['image']}, shows a {worst['delta_t_c']:.1f}°C rise "
            f"above ambient (peak {worst['max_temp_c']:.1f}°C), consistent with {worst_consequence}. "
            f"{worst_action[0].upper()}{worst_action[1:]}."
        )

    sentences = [
        f"This inspection identified {total} thermal anomal{'y' if total == 1 else 'ies'} ({breakdown}). "
        "Elevated temperatures at electrical connections and components are a leading indicator of "
        "developing faults - loose or corroded connections, overloaded circuits, or degrading hardware "
        "- and typically worsen over time if left unaddressed.",
        finding_sentence,
    ]
    if total > 1:
        sentences.append(
            "Additional findings are detailed below, each with its own severity and recommended action."
        )
    return " ".join(sentences)


def _comparative_narrative(all_comparative_rows: list[dict]) -> str | None:
    """Short paragraph summarizing comparative-method findings specifically -
    returns None if there's nothing to report (no rows, or none with a
    severity, meaning every compared component was within normal range of
    its peers)."""
    flagged = [row for row in all_comparative_rows if row.get("severity")]
    if not flagged:
        return None

    worst = max(flagged, key=lambda r: r["delta_t_c"])
    worst_action = COMPARATIVE_ACTION[worst["severity"]]
    label = worst.get("label") or "one component"

    return (
        f"Comparing corresponding components against each other (rather than against an "
        f"estimated ambient) surfaced {len(flagged)} additional finding{'s' if len(flagged) != 1 else ''}: "
        f"{label} on {worst['image']} runs {worst['delta_t_c']:.1f}°C hotter than its peers under "
        f"similar load, which the ambient-referenced method alone may not flag as severe. "
        f"{worst_action[0].upper()}{worst_action[1:]}."
    )


def _has_findings(entry: ImageReportEntry) -> bool:
    if entry.hotspot_rows:
        return True
    return any(row.get("severity") for row in (entry.comparative_rows or []))


def _report_image(annotated_image: Path | bytes, display_width_in: float = 4.0) -> RLImage:
    source = io.BytesIO(annotated_image) if isinstance(annotated_image, bytes) else annotated_image
    with PILImage.open(source) as im:
        w_px, h_px = im.size
    if isinstance(source, io.BytesIO):
        source.seek(0)
    display_w = display_width_in * inch
    display_h = display_w * (h_px / w_px)
    return RLImage(source if isinstance(source, io.BytesIO) else str(source), width=display_w, height=display_h)


def _thermal_photo_block(
    thermal_image: Path | bytes, visual_image: Path | bytes | None, styles
) -> Table | RLImage:
    """Thermal (annotated) image, with the camera's matching visual photo
    alongside it for context when available. The visual photo is shown
    unannotated - its framing/lens isn't pixel-aligned with the thermal
    sensor, so hotspot boxes drawn from thermal coordinates would land in
    the wrong place on it. Falls back to the thermal image alone (same as
    before this existed) when no visual photo was captured."""
    if visual_image is None:
        return _report_image(thermal_image, display_width_in=4.0)

    pair_width_in = 3.15
    thermal_flow = _report_image(thermal_image, display_width_in=pair_width_in)
    visual_flow = _report_image(visual_image, display_width_in=pair_width_in)
    caption_style = styles["Normal"]
    table = Table(
        [
            [Paragraph("<b>Thermal</b>", caption_style), Paragraph("<b>Visual photo</b>", caption_style)],
            [thermal_flow, visual_flow],
        ],
        colWidths=[(pair_width_in + 0.15) * inch, (pair_width_in + 0.15) * inch],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def generate_audit_findings_report(
    entries: list[ImageReportEntry],
    out_path: str | Path,
    title: str = "Thermal Inspection Findings",
    excluded: list[tuple[str, str]] | None = None,
    metadata: ReportMetadata | None = None,
) -> Path:
    """Write a clean, client-facing audit PDF listing only thermal findings -
    no methodology/interpretation narrative, no per-image list of clean
    results, no appendix of files excluded from review. Just the summary
    counts, a persuasive plain-English narrative, and the findings themselves.

    Each entry's hotspot_rows should already reflect reviewed judgment —
    e.g. regions assessed as camera artifacts (reflections, objects in frame)
    should be excluded from hotspot_rows before calling this, not included
    with a caveat. entries.note is ignored by this report; use
    generate_pdf_report if you need the fuller investigation-record version
    (every image individually, clean or not, plus a skipped-files appendix).

    excluded is accepted for API symmetry with generate_pdf_report but not
    rendered here - clean images and skipped files are omitted entirely
    rather than listed, to keep this report focused for a client audience.
    metadata, if given, renders a client/site/date/inspector header block
    under the title.
    """
    out_path = Path(out_path)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch
    )
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(
        Paragraph(datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"), styles["Normal"])
    )
    story.append(Spacer(1, 0.1 * inch))
    story.extend(_metadata_block(metadata, styles))
    story.append(Spacer(1, 0.2 * inch))

    findings_entries = [e for e in entries if _has_findings(e)]

    counts: dict[str, int] = {}
    for e in findings_entries:
        for row in e.hotspot_rows:
            counts[row["severity"]] = counts.get(row["severity"], 0) + 1

    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(f"Images reviewed: {len(entries)}", styles["Normal"]))
    story.append(Paragraph(f"Images with findings: {len(findings_entries)}", styles["Normal"]))
    story.append(
        Paragraph(f"Total findings: {sum(len(e.hotspot_rows) for e in findings_entries)}", styles["Normal"])
    )
    for sev in SEVERITY_ORDER:
        if counts.get(sev):
            story.append(Paragraph(f"&nbsp;&nbsp;{SEVERITY_LABELS[sev]}: {counts[sev]}", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    all_rows = [row for e in findings_entries for row in e.hotspot_rows]
    story.append(Paragraph(_narrative_summary(all_rows), styles["Normal"]))

    all_comparative_rows = [row for e in entries for row in (e.comparative_rows or [])]
    comparative_text = _comparative_narrative(all_comparative_rows)
    if comparative_text:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(comparative_text, styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    columns = [
        ("severity", "Severity"),
        ("delta_t_c", "ΔT (°C)"),
        ("max_temp_c", "Max Temp (°C)"),
        ("bbox_location", "Location (px)"),
    ]
    comparative_columns = [
        ("label", "Component"),
        ("severity", "Severity"),
        ("delta_t_c", "ΔT vs. Peers (°C)"),
        ("max_temp_c", "Max Temp (°C)"),
        ("bbox_location", "Location (px)"),
    ]

    if findings_entries:
        story.append(Paragraph("Findings", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))
        for entry in findings_entries:
            story.append(Paragraph(entry.image_name, styles["Heading3"]))
            story.append(_thermal_photo_block(entry.annotated_image, entry.visual_image, styles))
            story.append(Spacer(1, 0.1 * inch))

            if entry.hotspot_rows:
                story.append(Paragraph("<b>Electrical thermal anomaly detected</b> at the location(s) below.", styles["Normal"]))
                story.append(Spacer(1, 0.05 * inch))
                rows_with_location = [
                    {**row, "bbox_location": f"{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}"}
                    for row in entry.hotspot_rows
                ]
                story.append(_hotspot_table(rows_with_location, columns))

            flagged_comparative = [row for row in (entry.comparative_rows or []) if row.get("severity")]
            if flagged_comparative:
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph("Comparative findings", styles["Heading4"] if "Heading4" in styles else styles["Heading3"]))
                comp_rows_with_location = [
                    {**row, "bbox_location": f"{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}"}
                    for row in flagged_comparative
                ]
                story.append(_hotspot_table(comp_rows_with_location, comparative_columns, row_colors_map=COMPARATIVE_ROW_COLORS))

            story.append(Spacer(1, 0.35 * inch))

    doc.build(story)
    return out_path


def generate_pdf_report(
    entries: list[ImageReportEntry],
    out_path: str | Path,
    title: str = "Thermal Inspection Report",
    skipped: list[tuple[str, str]] | None = None,
    metadata: ReportMetadata | None = None,
) -> Path:
    """Write a PDF summarizing a batch of analyzed thermal images.

    entries should be ordered worst-first (or however the caller wants them
    presented); this function doesn't re-sort. skipped is an optional list of
    (filename, reason) pairs for files that failed extraction, listed at the end.
    metadata, if given, renders a client/site/date/inspector header block
    under the title.
    """
    out_path = Path(out_path)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch
    )
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(
        Paragraph(
            datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.extend(_metadata_block(metadata, styles))
    story.append(Spacer(1, 0.2 * inch))

    total_hotspots = sum(len(e.hotspot_rows) for e in entries)
    counts: dict[str, int] = {}
    for e in entries:
        for row in e.hotspot_rows:
            counts[row["severity"]] = counts.get(row["severity"], 0) + 1

    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(f"Images analyzed: {len(entries)}", styles["Normal"]))
    story.append(Paragraph(f"Total anomalies flagged: {total_hotspots}", styles["Normal"]))
    for sev in SEVERITY_ORDER:
        if counts.get(sev):
            story.append(Paragraph(f"&nbsp;&nbsp;{SEVERITY_LABELS[sev]}: {counts[sev]}", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    all_rows = [row for e in entries for row in e.hotspot_rows]
    story.append(Paragraph(_narrative_summary(all_rows), styles["Normal"]))

    all_comparative_rows = [row for e in entries for row in (e.comparative_rows or [])]
    comparative_text = _comparative_narrative(all_comparative_rows)
    if comparative_text:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(comparative_text, styles["Normal"]))
    story.append(Spacer(1, 0.3 * inch))

    columns = [
        ("severity", "Severity"),
        ("delta_t_c", "ΔT (°C)"),
        ("max_temp_c", "Max (°C)"),
        ("mean_temp_c", "Mean (°C)"),
        ("area_px", "Area (px)"),
        ("bbox_location", "Location (x,y,w,h)"),
    ]
    comparative_columns = [
        ("label", "Component"),
        ("severity", "Severity"),
        ("delta_t_c", "ΔT vs. Peers (°C)"),
        ("max_temp_c", "Max Temp (°C)"),
        ("bbox_location", "Location (px)"),
    ]

    for entry in entries:
        story.append(Paragraph(entry.image_name, styles["Heading2"]))
        story.append(Paragraph(f"Ambient reference: {entry.ambient_c:.1f}&deg;C", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_thermal_photo_block(entry.annotated_image, entry.visual_image, styles))
        story.append(Spacer(1, 0.1 * inch))

        if entry.hotspot_rows:
            story.append(Paragraph("<b>Electrical thermal anomaly detected</b> at the location(s) below.", styles["Normal"]))
            story.append(Spacer(1, 0.05 * inch))
            rows_with_location = [
                {**row, "bbox_location": f"{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}"}
                for row in entry.hotspot_rows
            ]
            story.append(_hotspot_table(rows_with_location, columns))
        else:
            story.append(Paragraph("No electrical thermal anomalies detected.", styles["Normal"]))

        if entry.comparative_rows:
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph("Comparative findings", styles["Heading3"]))
            comp_rows_with_location = [
                {**row, "bbox_location": f"{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}"}
                for row in entry.comparative_rows
            ]
            story.append(_hotspot_table(comp_rows_with_location, comparative_columns, row_colors_map=COMPARATIVE_ROW_COLORS))

        if entry.note:
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph(f"<b>Note:</b> {entry.note}", styles["Normal"]))

        story.append(Spacer(1, 0.35 * inch))

    if skipped:
        story.append(Paragraph("Files skipped", styles["Heading2"]))
        for name, reason in skipped:
            story.append(Paragraph(f"<b>{name}:</b> {reason}", styles["Normal"]))

    doc.build(story)
    return out_path
