"""Tests thermal_inspector.gdocs_report against fake Docs/Drive services -
never touches the real Google APIs, so these run with no credentials and no
network. The fakes deliberately return a table start index DIFFERENT from
the one _DocBuilder.table() naively requested (mirroring what real Docs
actually does), so these tests would have caught both bugs fixed after
testing against live production on 2026-08-09:
  1. updateTableCellStyle needs cell location wrapped in "tableRange", not
     a bare "tableCellLocation" (400: unknown field).
  2. that wrapped location must use the table's *actual* resolved start
     index, not the index originally requested for insertTable (400:
     "table start location is invalid").
"""

from __future__ import annotations

import pytest

from thermal_inspector.gdocs_report import (
    _DocBuilder,
    _find_table,
    build_findings_doc,
)
from thermal_inspector.pdf_report import ImageReportEntry, ReportMetadata


class _Executable:
    def __init__(self, value):
        self._value = value

    def execute(self):
        return self._value


class _FakeDocsService:
    """Simulates just enough of the Docs API for _DocBuilder to run against:
    batchUpdate records every request it's given, and get() reports a table
    structure whose actual start index is requested_index + 7 - deliberately
    different from what insertTable was asked for, since real Docs doesn't
    guarantee they match either."""

    TABLE_START_OFFSET = 7

    def __init__(self):
        self.batch_calls: list[list[dict]] = []
        self._last_table_start: int | None = None
        self._last_table: dict | None = None
        self._body_end = 50

    def documents(self):
        return self

    def batchUpdate(self, documentId, body):
        requests = body["requests"]
        self.batch_calls.append(requests)
        for req in requests:
            if "insertTable" in req:
                self._insert_table(req["insertTable"])
            elif "insertText" in req:
                self._body_end += len(req["insertText"]["text"])
            elif "insertInlineImage" in req:
                self._body_end += 1
        return _Executable({})

    def get(self, documentId, fields=None):
        content = []
        if self._last_table is not None:
            content.append(
                {"startIndex": self._last_table_start, "endIndex": self._last_table_start + 200, "table": self._last_table}
            )
        content.append({"startIndex": self._body_end, "endIndex": self._body_end + 1})
        return _Executable({"body": {"content": content}})

    def _insert_table(self, spec):
        rows, cols = spec["rows"], spec["columns"]
        requested_index = spec["location"]["index"]
        actual_start = requested_index + self.TABLE_START_OFFSET
        cursor = actual_start + 4
        table_rows = []
        for _ in range(rows):
            cells = []
            for _ in range(cols):
                cell_start = cursor
                cells.append({"startIndex": cell_start, "content": [{"startIndex": cell_start}]})
                cursor += 10
            table_rows.append({"tableCells": cells})
        self._last_table_start = actual_start
        self._last_table = {"tableRows": table_rows}
        self._body_end = cursor + 5


class _FakeFilesResource:
    def __init__(self, parent):
        self._parent = parent

    def create(self, body, media_body=None, fields=None, supportsAllDrives=None):
        file_id = self._parent._next_file_id()
        self._parent.created_files.append(body)
        if body.get("mimeType") == "application/vnd.google-apps.document":
            return _Executable({"id": file_id, "webViewLink": f"https://docs.google.com/document/d/{file_id}/edit"})
        return _Executable({"id": file_id})

    def update(self, fileId, body, supportsAllDrives=None):
        self._parent.trashed.append((fileId, body))
        return _Executable({})

    def delete(self, fileId, supportsAllDrives=None):
        raise AssertionError(
            "files().delete() should never be called on a Shared Drive item - "
            "a Content Manager can only trash (files().update trashed=True), "
            "and delete() 404s instead of the 403 you'd expect for that."
        )


class _FakePermissionsResource:
    def __init__(self, parent):
        self._parent = parent

    def create(self, fileId, body, fields=None, supportsAllDrives=None):
        self._parent.permissions_created.append((fileId, body))
        return _Executable({})


class _FakeDriveService:
    def __init__(self):
        self.created_files: list[dict] = []
        self.trashed: list[tuple] = []
        self.permissions_created: list[tuple] = []
        self._counter = 0

    def _next_file_id(self) -> str:
        self._counter += 1
        return f"fake-file-{self._counter}"

    def files(self):
        return _FakeFilesResource(self)

    def permissions(self):
        return _FakePermissionsResource(self)


@pytest.fixture
def fake_docs():
    return _FakeDocsService()


@pytest.fixture
def fake_drive():
    return _FakeDriveService()


class TestFindTable:
    def test_resolves_actual_start_index_not_requested_one(self, fake_docs):
        # Simulate the state right after an insertTable request at index 10.
        fake_docs._insert_table({"rows": 2, "columns": 3, "location": {"index": 10}})
        doc = fake_docs.get(documentId="doc1").execute()
        resolved_start, table = _find_table(doc, start_index=10)
        assert resolved_start == 10 + _FakeDocsService.TABLE_START_OFFSET
        assert resolved_start != 10  # the whole point of the fake


class TestDocBuilderTable:
    def test_cell_style_uses_table_range_wrapper_not_bare_location(self, fake_docs):
        """Regression test for bug #1: a bare tableCellLocation directly
        under updateTableCellStyle 400s - Docs requires it nested inside a
        1x1 tableRange."""
        builder = _DocBuilder(fake_docs, "doc1")
        columns = [("severity", "Severity"), ("delta_t_c", "ΔT (°C)")]
        rows = [{"severity": "serious", "delta_t_c": 12.3}]
        color_map = {"serious": _fake_color(1.0, 0.9, 0.8)}

        builder.table(columns, rows, color_map)

        style_requests = [
            req
            for call in fake_docs.batch_calls
            for req in call
            if "updateTableCellStyle" in req
        ]
        assert style_requests, "expected at least one updateTableCellStyle request"
        for req in style_requests:
            body = req["updateTableCellStyle"]
            assert "tableRange" in body, "cell location must be wrapped in tableRange"
            assert "tableCellLocation" not in body, "must not be a bare top-level field"
            assert body["tableRange"]["rowSpan"] == 1
            assert body["tableRange"]["columnSpan"] == 1

    def test_cell_style_uses_resolved_index_not_requested_index(self, fake_docs):
        """Regression test for bug #2: using the originally-requested
        insertTable index (rather than the index Docs actually assigned)
        400s with 'table start location is invalid'."""
        builder = _DocBuilder(fake_docs, "doc1")
        requested_index = builder.index  # captured before table() runs
        columns = [("severity", "Severity")]
        rows = [{"severity": "serious"}]
        color_map = {"serious": _fake_color(1.0, 0.9, 0.8)}

        builder.table(columns, rows, color_map)

        style_requests = [
            req["updateTableCellStyle"]
            for call in fake_docs.batch_calls
            for req in call
            if "updateTableCellStyle" in req
        ]
        used_indices = {r["tableRange"]["tableCellLocation"]["tableStartLocation"]["index"] for r in style_requests}
        assert used_indices == {requested_index + _FakeDocsService.TABLE_START_OFFSET}

    def test_text_fill_processes_cells_from_highest_index_down(self, fake_docs):
        """Cells must be filled starting from the highest index and working
        down, or earlier (lower-index) cells' captured positions get
        invalidated by shifts from filling later ones first."""
        builder = _DocBuilder(fake_docs, "doc1")
        columns = [("a", "A"), ("b", "B")]
        rows = [{"a": "row1a", "b": "row1b"}]
        builder.table(columns, rows, {})

        fill_call = next(call for call in fake_docs.batch_calls if any("insertText" in r for r in call))
        indices = [r["insertText"]["location"]["index"] for r in fill_call if "insertText" in r]
        assert indices == sorted(indices, reverse=True)

    def test_header_row_is_bold_white(self, fake_docs):
        builder = _DocBuilder(fake_docs, "doc1")
        columns = [("a", "Header A")]
        builder.table(columns, [{"a": "value"}], {})

        style_text_requests = [
            req["updateTextStyle"]
            for call in fake_docs.batch_calls
            for req in call
            if "updateTextStyle" in req
        ]
        assert any(r["textStyle"].get("bold") is True for r in style_text_requests)


class TestBuildFindingsDocNoFindings:
    def test_clean_run_produces_no_tables_or_images(self, fake_docs, fake_drive):
        """The simplest real path through build_findings_doc - no findings,
        no images - exercised end to end against the fakes."""
        entry = ImageReportEntry(
            image_name="FLIR0001.jpg",
            annotated_image=None,
            hotspot_rows=[],
            ambient_c=20.0,
        )
        url = build_findings_doc(
            fake_docs,
            fake_drive,
            [entry],
            folder_id="fake-folder",
            style="audit",
            metadata=ReportMetadata(client_name="Acme Corp"),
        )
        assert url.startswith("https://docs.google.com/document/d/")
        assert fake_drive.created_files[0]["mimeType"] == "application/vnd.google-apps.document"
        assert fake_drive.created_files[0]["parents"] == ["fake-folder"]

        all_requests = [req for call in fake_docs.batch_calls for req in call]
        assert not any("insertTable" in req for req in all_requests)
        assert not any("insertInlineImage" in req for req in all_requests)

    def test_invalid_style_rejected(self, fake_docs, fake_drive):
        with pytest.raises(ValueError):
            build_findings_doc(fake_docs, fake_drive, [], folder_id="fake-folder", style="bogus")


def _fake_color(r: float, g: float, b: float):
    """A minimal stand-in with the .red/.green/.blue attributes
    gdocs_report._hex_to_rgb reads - avoids depending on reportlab's
    HexColor parsing just to build a test color."""

    class _Color:
        red, green, blue = r, g, b

    return _Color()
