"""Tests the visual-note feature: an inspector can flag a visible (non-
thermal) issue on an image, and it's enough on its own to keep that image
from being dropped as 'clean' by the client-facing audit report - both the
PDF and Google Docs versions."""

from __future__ import annotations

from pypdf import PdfReader

from thermal_inspector.gdocs_report import build_findings_doc
from thermal_inspector.pdf_report import ImageReportEntry, ReportMetadata, _has_findings, generate_audit_findings_report

# Matches how pytest itself imports test files under this project's default
# "prepend" import mode (no tests/__init__.py, so tests/ is added straight to
# sys.path and modules are top-level) - importing as `tests.test_gdocs_report`
# instead would load a second, distinct copy of that module.
from test_gdocs_report import _FakeDocsService, _FakeDriveService


def _upload_and_get_ids(client):
    resp = client.post(
        "/api/analyze",
        files=[("files", ("FLIR0001.jpg", b"fake-radiometric-bytes", "image/jpeg"))],
    )
    data = resp.json()
    return data["run_id"], data["results"][0]["image_id"]


class TestSetVisualNoteEndpoint:
    def test_set_and_read_back(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/note", json={"note": "Cracked enclosure door"}
        )
        assert resp.status_code == 200
        assert resp.json()["visual_note"] == "Cracked enclosure door"

        detail = logged_in_client.get(f"/api/history/{run_id}").json()
        assert detail["images"][0]["visual_note"] == "Cracked enclosure door"

    def test_blank_note_clears_it(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"note": "Something"})
        resp = logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"note": "   "})
        assert resp.status_code == 200
        assert resp.json()["visual_note"] is None

    def test_omitted_note_clears_it(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"note": "Something"})
        resp = logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={})
        assert resp.status_code == 200
        assert resp.json()["visual_note"] is None

    def test_wrong_run_id_404s(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id + 999}/images/{image_id}/note", json={"note": "x"}
        )
        assert resp.status_code == 404

    def test_viewer_cannot_set_note(self, client, make_user, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        make_user("frank", role="viewer", password="password123")
        client.post("/api/login", data={"username": "frank", "password": "password123"})
        resp = client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"note": "x"})
        assert resp.status_code == 403


class TestHasFindingsIncludesNotes:
    def _entry(self, **overrides):
        defaults = dict(
            image_name="FLIR0001.jpg", annotated_image=None, hotspot_rows=[], ambient_c=20.0, comparative_rows=None, note=None
        )
        defaults.update(overrides)
        return ImageReportEntry(**defaults)

    def test_no_findings_no_note_is_excluded(self):
        assert _has_findings(self._entry()) is False

    def test_note_alone_counts_as_a_finding(self):
        assert _has_findings(self._entry(note="Cracked enclosure door")) is True

    def test_empty_string_note_does_not_count(self):
        # Matches the endpoint's own behavior of normalizing blank -> None.
        assert _has_findings(self._entry(note="")) is False


class TestAuditReportIncludesNoteOnlyImages:
    def test_pdf_audit_report_shows_the_note_text(self, tmp_path):
        entries = [
            ImageReportEntry(
                image_name="FLIR0001.jpg",
                annotated_image=None,
                hotspot_rows=[],  # no thermal findings at all
                ambient_c=20.0,
                note="Visible corrosion on the bus bar connection",
            )
        ]
        out_path = tmp_path / "report.pdf"
        generate_audit_findings_report(entries, out_path, metadata=ReportMetadata(client_name="Acme Corp"))

        reader = PdfReader(str(out_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert "FLIR0001.jpg" in text
        assert "Visible corrosion on the bus bar connection" in text

    def test_pdf_audit_report_omits_images_with_neither(self, tmp_path):
        entries = [
            ImageReportEntry(image_name="FLIR0002.jpg", annotated_image=None, hotspot_rows=[], ambient_c=20.0, note=None)
        ]
        out_path = tmp_path / "report.pdf"
        generate_audit_findings_report(entries, out_path)

        reader = PdfReader(str(out_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert "FLIR0002.jpg" not in text
        assert "Images with findings: 0" in text

    def test_gdocs_audit_report_includes_the_note(self):
        fake_docs = _FakeDocsService()
        fake_drive = _FakeDriveService()
        entries = [
            ImageReportEntry(
                image_name="FLIR0001.jpg",
                annotated_image=None,
                hotspot_rows=[],
                ambient_c=20.0,
                note="Visible corrosion on the bus bar connection",
            )
        ]
        build_findings_doc(fake_docs, fake_drive, entries, folder_id="fake-folder", style="audit")

        all_text_inserts = [
            req["insertText"]["text"]
            for call in fake_docs.batch_calls
            for req in call
            if "insertText" in req
        ]
        joined = "".join(all_text_inserts)
        assert "FLIR0001.jpg" in joined
        assert "Visible corrosion on the bus bar connection" in joined
