"""Renders a batch analysis run into a Google Doc, as an alternative to the
PDF export in pdf_report.py. Content/structure mirrors pdf_report.py's two
styles (generate_audit_findings_report / generate_pdf_report) closely enough
that narrative text and severity labels are imported straight from there
rather than duplicated.

Unlike the PDF renderer - which builds one static document from a fully
in-memory "story" list - this talks to the live Docs/Drive APIs as it goes.
Google Docs indices are UTF-16 code-unit offsets into the document body, and
inserting a table or an inline image shifts subsequent indices by amounts
that aren't practical to hand-compute correctly. So tables and images are
treated as sync points: insert, then re-fetch the document to learn the real
resulting index before queuing more content. Plain text paragraphs between
sync points are cheap and safely batched together.

Requires a Docs API service and a Drive API service (see webapp/gdocs.py for
how the webapp constructs these from a service account); this module itself
has no auth or webapp dependency, so it could be reused from the CLI later.
"""

from __future__ import annotations

import datetime
import io
from pathlib import Path

from PIL import Image as PILImage
from googleapiclient.http import MediaIoBaseUpload
from reportlab.lib import colors

from .pdf_report import (
    COMPARATIVE_ROW_COLORS,
    ImageReportEntry,
    ReportMetadata,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
    SEVERITY_ROW_COLORS,
    _comparative_narrative,
    _has_findings,
    _narrative_summary,
)

_IMAGE_DISPLAY_WIDTH_PT = 4.0 * 72  # matches pdf_report's 4.0in single-image width


def _hex_to_rgb(color: colors.Color) -> dict:
    return {"red": color.red, "green": color.green, "blue": color.blue}


def _format_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _as_bytes(image: Path | bytes) -> bytes:
    return image if isinstance(image, bytes) else Path(image).read_bytes()


def _image_dims(image: Path | bytes) -> tuple[int, int]:
    source = io.BytesIO(image) if isinstance(image, bytes) else image
    with PILImage.open(source) as im:
        return im.size


def _upload_temp_image(drive_service, image_bytes: bytes, folder_id: str) -> str:
    media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype="image/png", resumable=False)
    file = (
        drive_service.files()
        .create(
            body={"name": "report_image.png", "parents": [folder_id]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    return file["id"]


def _find_table(doc: dict, start_index: int) -> tuple[int, dict]:
    """Returns (actual_start_index, table_object). The actual start index
    isn't guaranteed to equal the index we requested insertTable at - it's
    also needed later for updateTableCellStyle's tableStartLocation, which
    404s/400s if given a stale or merely-assumed index rather than the one
    Docs actually assigned."""
    for el in doc["body"]["content"]:
        if "table" in el and el.get("startIndex") == start_index:
            return el["startIndex"], el["table"]
    # Fall back to the most recently inserted table if the exact start
    # index isn't found (e.g. Docs coalesced an adjacent structural element).
    tables = [(el["startIndex"], el["table"]) for el in doc["body"]["content"] if "table" in el]
    if tables:
        return tables[-1]
    raise RuntimeError("Inserted table not found when re-fetching the document")


class _DocBuilder:
    def __init__(self, docs_service, document_id: str):
        self._docs = docs_service
        self._document_id = document_id
        self._pending: list[dict] = []
        self.index = 1  # body content starts at index 1

    def _flush(self) -> None:
        if self._pending:
            self._docs.documents().batchUpdate(
                documentId=self._document_id, body={"requests": self._pending}
            ).execute()
            self._pending = []

    def _resync_index(self) -> None:
        doc = self._docs.documents().get(documentId=self._document_id).execute()
        self.index = doc["body"]["content"][-1]["endIndex"] - 1

    def paragraph(self, text: str, *, heading: str | None = None, bold: bool = False, italic: bool = False) -> None:
        text = text + "\n"
        start = self.index
        self._pending.append({"insertText": {"location": {"index": start}, "text": text}})
        text_end = start + len(text) - 1  # exclude the paragraph-break newline from character styling
        style = {}
        if bold:
            style["bold"] = True
        if italic:
            style["italic"] = True
        if style:
            self._pending.append(
                {
                    "updateTextStyle": {
                        "range": {"startIndex": start, "endIndex": text_end},
                        "textStyle": style,
                        "fields": ",".join(style.keys()),
                    }
                }
            )
        if heading:
            self._pending.append(
                {
                    "updateParagraphStyle": {
                        "range": {"startIndex": start, "endIndex": start + len(text)},
                        "paragraphStyle": {"namedStyleType": heading},
                        "fields": "namedStyleType",
                    }
                }
            )
        self.index = start + len(text)

    def rich_paragraph(self, runs: list[tuple[str, bool]]) -> None:
        """runs is (text, bold) pairs on one paragraph, e.g. a bold label
        followed by a plain value: [("Client: ", True), ("Acme Corp", False)]."""
        start = self.index
        full_text = "".join(text for text, _ in runs) + "\n"
        self._pending.append({"insertText": {"location": {"index": start}, "text": full_text}})
        cursor = start
        for text, bold in runs:
            if bold and text:
                self._pending.append(
                    {
                        "updateTextStyle": {
                            "range": {"startIndex": cursor, "endIndex": cursor + len(text)},
                            "textStyle": {"bold": True},
                            "fields": "bold",
                        }
                    }
                )
            cursor += len(text)
        self.index = start + len(full_text)

    def image(self, drive_service, folder_id: str, image: Path | bytes, display_width_pt: float = _IMAGE_DISPLAY_WIDTH_PT) -> None:
        """Inserts image as its own paragraph at the current cursor. The
        source file is uploaded to Drive only long enough for Docs to fetch
        and embed a copy of it - Docs keeps its own copy once inserted, so
        the temp Drive file is trashed immediately after (not permanently
        deleted: a Shared Drive Content Manager can trash files but not
        hard-delete them - files.delete() 404s instead of the 403 you'd
        expect, since Drive API surfaces "no delete capability" on a Shared
        Drive item as "not found" rather than "forbidden")."""
        self._flush()
        w_px, h_px = _image_dims(image)
        display_height_pt = display_width_pt * (h_px / w_px)
        file_id = _upload_temp_image(drive_service, _as_bytes(image), folder_id)
        try:
            drive_service.permissions().create(
                fileId=file_id, body={"role": "reader", "type": "anyone"}, fields="id", supportsAllDrives=True
            ).execute()
            uri = f"https://drive.google.com/uc?id={file_id}"
            self._docs.documents().batchUpdate(
                documentId=self._document_id,
                body={
                    "requests": [
                        {
                            "insertInlineImage": {
                                "location": {"index": self.index},
                                "uri": uri,
                                "objectSize": {
                                    "width": {"magnitude": display_width_pt, "unit": "PT"},
                                    "height": {"magnitude": display_height_pt, "unit": "PT"},
                                },
                            }
                        }
                    ]
                },
            ).execute()
        finally:
            # Best-effort cleanup - if the try block already failed, let
            # that original exception propagate rather than a cleanup
            # failure masking it.
            try:
                drive_service.files().update(fileId=file_id, body={"trashed": True}, supportsAllDrives=True).execute()
            except Exception:
                pass
        self._resync_index()

    def table(self, columns: list[tuple[str, str]], rows: list[dict], color_map: dict) -> None:
        """columns is (row-dict key, header label) pairs, same shape as
        pdf_report._hotspot_table. Data rows are colored by row['severity']
        via color_map (SEVERITY_ROW_COLORS or COMPARATIVE_ROW_COLORS)."""
        self._flush()
        header = [label for _, label in columns]
        data_rows = [[_format_cell(row[key]) for key, _ in columns] for row in rows]
        row_colors = [color_map.get(row.get("severity")) for row in rows]
        all_rows = [header] + data_rows
        n_rows, n_cols = len(all_rows), len(header)

        table_start = self.index
        self._docs.documents().batchUpdate(
            documentId=self._document_id,
            body={"requests": [{"insertTable": {"rows": n_rows, "columns": n_cols, "location": {"index": table_start}}}]},
        ).execute()

        doc = self._docs.documents().get(documentId=self._document_id).execute()
        table_start, table_el = _find_table(doc, table_start)

        # A freshly inserted table cell holds exactly one empty paragraph;
        # its startIndex is where that cell's text goes. Capture every
        # cell's index from this single snapshot, then fill text starting
        # from the highest index and working down - each insertText only
        # shifts indices *after* it, which are cells we've already filled,
        # so earlier (lower-index, not-yet-filled) cells stay valid.
        cells = [
            (r, c, cell["content"][0]["startIndex"])
            for r, row in enumerate(table_el["tableRows"])
            for c, cell in enumerate(row["tableCells"])
        ]
        fill_requests = []
        for r, c, cell_start in sorted(cells, key=lambda t: -t[2]):
            text = all_rows[r][c]
            if not text:
                continue
            fill_requests.append({"insertText": {"location": {"index": cell_start}, "text": text}})
            if r == 0:
                fill_requests.append(
                    {
                        "updateTextStyle": {
                            "range": {"startIndex": cell_start, "endIndex": cell_start + len(text)},
                            "textStyle": {"bold": True, "foregroundColor": {"color": {"rgbColor": {"red": 1, "green": 1, "blue": 1}}}},
                            "fields": "bold,foregroundColor",
                        }
                    }
                )
        if fill_requests:
            self._docs.documents().batchUpdate(documentId=self._document_id, body={"requests": fill_requests}).execute()

        # Cell background colors address cells by (row, column) position
        # rather than text index, so they're independent of the fills above
        # and can be computed straight from the original table structure.
        header_bg = _hex_to_rgb(colors.HexColor("#2A2F37"))
        style_requests = []
        for r in range(n_rows):
            bg = header_bg if r == 0 else (_hex_to_rgb(row_colors[r - 1]) if row_colors[r - 1] is not None else None)
            if bg is None:
                continue
            for c in range(n_cols):
                style_requests.append(
                    {
                        "updateTableCellStyle": {
                            # A single cell is addressed as a 1x1 tableRange -
                            # updateTableCellStyle has no bare "target this
                            # one cell" field, only tableRange (a span of
                            # cells) or tableStartLocation (the whole table).
                            "tableRange": {
                                "tableCellLocation": {
                                    "tableStartLocation": {"index": table_start},
                                    "rowIndex": r,
                                    "columnIndex": c,
                                },
                                "rowSpan": 1,
                                "columnSpan": 1,
                            },
                            "tableCellStyle": {"backgroundColor": {"color": {"rgbColor": bg}}},
                            "fields": "backgroundColor",
                        }
                    }
                )
        if style_requests:
            self._docs.documents().batchUpdate(documentId=self._document_id, body={"requests": style_requests}).execute()

        self._resync_index()


def _metadata_runs(metadata: ReportMetadata | None) -> list[list[tuple[str, bool]]]:
    metadata = metadata or ReportMetadata()
    fields = [
        ("Client", metadata.client_name),
        ("Site / Location", metadata.site_location),
        ("Audit Date", metadata.audit_date or datetime.date.today().isoformat()),
        ("Inspector", metadata.inspector_name),
        ("Report ID", metadata.report_id),
    ]
    return [[(f"{label}: ", True), (str(value), False)] for label, value in fields if value]


def _insert_entry_images(builder: _DocBuilder, drive_service, folder_id: str, entry: ImageReportEntry) -> None:
    """Simplified relative to pdf_report's thermal_photo_block: thermal and
    visual photos are inserted as sequential images rather than side by side
    in a table cell - table-cell image insertion adds another layer of
    index bookkeeping that isn't worth the risk for a supporting visual."""
    if entry.annotated_image is None:
        builder.paragraph(
            "Image unavailable (lost from storage) - the findings below are still from the original analysis.",
            italic=True,
        )
        return
    builder.image(drive_service, folder_id, entry.annotated_image)
    if entry.visual_image is not None:
        builder.paragraph("Visual photo:", bold=True)
        builder.image(drive_service, folder_id, entry.visual_image)


_FINDINGS_COLUMNS = [
    ("severity", "Severity"),
    ("delta_t_c", "ΔT (°C)"),
    ("max_temp_c", "Max Temp (°C)"),
    ("bbox_location", "Location (px)"),
]
_FULL_COLUMNS = [
    ("severity", "Severity"),
    ("delta_t_c", "ΔT (°C)"),
    ("max_temp_c", "Max (°C)"),
    ("mean_temp_c", "Mean (°C)"),
    ("area_px", "Area (px)"),
    ("bbox_location", "Location (x,y,w,h)"),
]
_COMPARATIVE_COLUMNS = [
    ("label", "Component"),
    ("severity", "Severity"),
    ("delta_t_c", "ΔT vs. Peers (°C)"),
    ("max_temp_c", "Max Temp (°C)"),
    ("bbox_location", "Location (px)"),
]


def _with_bbox_location(rows: list[dict]) -> list[dict]:
    return [{**row, "bbox_location": f"{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}"} for row in rows]


def build_findings_doc(
    docs_service,
    drive_service,
    entries: list[ImageReportEntry],
    folder_id: str,
    *,
    style: str = "audit",
    title: str | None = None,
    excluded: list[tuple[str, str]] | None = None,
    metadata: ReportMetadata | None = None,
) -> str:
    """Builds a Google Doc in folder_id (must be a Shared Drive folder the
    service account has write access to) equivalent to
    generate_audit_findings_report (style="audit") or generate_pdf_report
    (style="full"), and returns its webViewLink.

    excluded is accepted for API symmetry with the PDF functions but, like
    them, only rendered in "full" style (as a skipped-files appendix) - it's
    omitted from "audit" style to keep that report focused for a client
    audience.
    """
    if style not in ("audit", "full"):
        raise ValueError("style must be 'audit' or 'full'")

    doc_title = title or ("Thermal Inspection Findings" if style == "audit" else "Thermal Inspection Report")

    file = (
        drive_service.files()
        .create(
            body={"name": doc_title, "mimeType": "application/vnd.google-apps.document", "parents": [folder_id]},
            fields="id, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )
    document_id = file["id"]
    builder = _DocBuilder(docs_service, document_id)

    builder.paragraph(doc_title, heading="TITLE")
    builder.paragraph(datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"), italic=True)
    for runs in _metadata_runs(metadata):
        builder.rich_paragraph(runs)

    if style == "audit":
        findings_entries = [e for e in entries if _has_findings(e)]
        counts: dict[str, int] = {}
        for e in findings_entries:
            for row in e.hotspot_rows:
                counts[row["severity"]] = counts.get(row["severity"], 0) + 1

        builder.paragraph("Summary", heading="HEADING_2")
        builder.paragraph(f"Images reviewed: {len(entries)}")
        builder.paragraph(f"Images with findings: {len(findings_entries)}")
        builder.paragraph(f"Total findings: {sum(len(e.hotspot_rows) for e in findings_entries)}")
        for sev in SEVERITY_ORDER:
            if counts.get(sev):
                builder.paragraph(f"    {SEVERITY_LABELS[sev]}: {counts[sev]}")

        all_rows = [row for e in findings_entries for row in e.hotspot_rows]
        builder.paragraph(_narrative_summary(all_rows))

        all_comparative_rows = [row for e in entries for row in (e.comparative_rows or [])]
        comparative_text = _comparative_narrative(all_comparative_rows)
        if comparative_text:
            builder.paragraph(comparative_text)

        if findings_entries:
            builder.paragraph("Findings", heading="HEADING_2")
            for entry in findings_entries:
                builder.paragraph(entry.image_name, heading="HEADING_3")
                _insert_entry_images(builder, drive_service, folder_id, entry)

                if entry.hotspot_rows:
                    builder.paragraph("Electrical thermal anomaly detected at the location(s) below.", bold=True)
                    builder.table(_FINDINGS_COLUMNS, _with_bbox_location(entry.hotspot_rows), SEVERITY_ROW_COLORS)

                flagged_comparative = [row for row in (entry.comparative_rows or []) if row.get("severity")]
                if flagged_comparative:
                    builder.paragraph("Comparative findings", heading="HEADING_3")
                    builder.table(_COMPARATIVE_COLUMNS, _with_bbox_location(flagged_comparative), COMPARATIVE_ROW_COLORS)

                if entry.note:
                    builder.rich_paragraph([("Visual issue noted: ", True), (entry.note, False)])

    else:  # style == "full"
        total_hotspots = sum(len(e.hotspot_rows) for e in entries)
        counts = {}
        for e in entries:
            for row in e.hotspot_rows:
                counts[row["severity"]] = counts.get(row["severity"], 0) + 1

        builder.paragraph("Summary", heading="HEADING_2")
        builder.paragraph(f"Images analyzed: {len(entries)}")
        builder.paragraph(f"Total anomalies flagged: {total_hotspots}")
        for sev in SEVERITY_ORDER:
            if counts.get(sev):
                builder.paragraph(f"    {SEVERITY_LABELS[sev]}: {counts[sev]}")

        all_rows = [row for e in entries for row in e.hotspot_rows]
        builder.paragraph(_narrative_summary(all_rows))

        all_comparative_rows = [row for e in entries for row in (e.comparative_rows or [])]
        comparative_text = _comparative_narrative(all_comparative_rows)
        if comparative_text:
            builder.paragraph(comparative_text)

        for entry in entries:
            builder.paragraph(entry.image_name, heading="HEADING_2")
            builder.paragraph(f"Ambient reference: {entry.ambient_c:.1f}°C")
            _insert_entry_images(builder, drive_service, folder_id, entry)

            if entry.hotspot_rows:
                builder.paragraph("Electrical thermal anomaly detected at the location(s) below.", bold=True)
                builder.table(_FULL_COLUMNS, _with_bbox_location(entry.hotspot_rows), SEVERITY_ROW_COLORS)
            else:
                builder.paragraph("No electrical thermal anomalies detected.")

            if entry.comparative_rows:
                builder.paragraph("Comparative findings", heading="HEADING_3")
                builder.table(_COMPARATIVE_COLUMNS, _with_bbox_location(entry.comparative_rows), COMPARATIVE_ROW_COLORS)

            if entry.note:
                builder.rich_paragraph([("Note: ", True), (entry.note, False)])

        if excluded:
            builder.paragraph("Files skipped", heading="HEADING_2")
            for name, reason in excluded:
                builder.rich_paragraph([(f"{name}: ", True), (reason, False)])

    builder._flush()
    return file["webViewLink"]
