"""Tests the visual-anomaly feature: an inspector can explicitly flag a
visible (non-thermal) issue on an image, with an optional free-text
description. The flag alone - not the text - is what keeps that image from
being dropped as 'clean' by the client-facing audit report, in both the PDF
and Google Docs versions."""

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
    def test_set_flag_and_note_together(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/note",
            json={"note": "Cracked enclosure door", "anomaly": True},
        )
        assert resp.status_code == 200
        assert resp.json() == {"visual_note": "Cracked enclosure door", "visual_anomaly": True, "asset_label": None}

        detail = logged_in_client.get(f"/api/history/{run_id}").json()
        assert detail["images"][0]["visual_note"] == "Cracked enclosure door"
        assert detail["images"][0]["visual_anomaly"] is True

    def test_note_without_flag_is_not_an_anomaly(self, logged_in_client, mock_radiometric):
        # Text alone, with the checkbox unchecked, shouldn't flip the flag.
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/note",
            json={"note": "Something worth mentioning but not an issue", "anomaly": False},
        )
        assert resp.json()["visual_anomaly"] is False

    def test_flag_without_note_is_still_an_anomaly(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/note", json={"anomaly": True}
        )
        assert resp.status_code == 200
        assert resp.json() == {"visual_note": None, "visual_anomaly": True, "asset_label": None}

    def test_omitting_anomaly_defaults_to_false(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={})
        assert resp.json()["visual_anomaly"] is False

    def test_blank_note_is_stored_as_none(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/note", json={"note": "   ", "anomaly": True}
        )
        assert resp.json() == {"visual_note": None, "visual_anomaly": True, "asset_label": None}

    def test_wrong_run_id_404s(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id + 999}/images/{image_id}/note", json={"anomaly": True}
        )
        assert resp.status_code == 404

    def test_viewer_cannot_set_note(self, client, make_user, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        make_user("frank", role="viewer", password="password123")
        client.post("/api/login", data={"username": "frank", "password": "password123"})
        resp = client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"anomaly": True})
        assert resp.status_code == 403


class TestHasFindingsUsesTheFlagNotTheText:
    def _entry(self, **overrides):
        defaults = dict(
            image_name="FLIR0001.jpg",
            annotated_image=None,
            hotspot_rows=[],
            ambient_c=20.0,
            comparative_rows=None,
            note=None,
            visual_anomaly=False,
        )
        defaults.update(overrides)
        return ImageReportEntry(**defaults)

    def test_nothing_flagged_is_excluded(self):
        assert _has_findings(self._entry()) is False

    def test_flag_true_counts_as_a_finding_even_with_no_text(self):
        assert _has_findings(self._entry(visual_anomaly=True)) is True

    def test_text_alone_without_the_flag_does_not_count(self):
        # This is the exact behavior the user rejected: a description with
        # no explicit anomaly flag must NOT pull the image into the report.
        assert _has_findings(self._entry(note="Something worth mentioning", visual_anomaly=False)) is False


class TestAuditReportRespectsTheFlag:
    def test_pdf_report_includes_flagged_image_with_its_note(self, tmp_path):
        entries = [
            ImageReportEntry(
                image_name="FLIR0001.jpg",
                annotated_image=None,
                hotspot_rows=[],
                ambient_c=20.0,
                note="Visible corrosion on the bus bar connection",
                visual_anomaly=True,
            )
        ]
        out_path = tmp_path / "report.pdf"
        generate_audit_findings_report(entries, out_path, metadata=ReportMetadata(client_name="Acme Corp"))

        reader = PdfReader(str(out_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert "FLIR0001.jpg" in text
        assert "Visible corrosion on the bus bar connection" in text
        assert "Images with findings: 1" in text

    def test_pdf_report_flags_without_text_still_included(self, tmp_path):
        entries = [
            ImageReportEntry(
                image_name="FLIR0003.jpg", annotated_image=None, hotspot_rows=[], ambient_c=20.0, visual_anomaly=True
            )
        ]
        out_path = tmp_path / "report.pdf"
        generate_audit_findings_report(entries, out_path)

        reader = PdfReader(str(out_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert "FLIR0003.jpg" in text
        assert "no description provided" in text.lower()

    def test_pdf_report_omits_unflagged_image_even_with_a_note(self, tmp_path):
        entries = [
            ImageReportEntry(
                image_name="FLIR0002.jpg",
                annotated_image=None,
                hotspot_rows=[],
                ambient_c=20.0,
                note="A description with no flag attached",
                visual_anomaly=False,
            )
        ]
        out_path = tmp_path / "report.pdf"
        generate_audit_findings_report(entries, out_path)

        reader = PdfReader(str(out_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        assert "FLIR0002.jpg" not in text
        assert "Images with findings: 0" in text

    def test_gdocs_report_includes_flagged_image_with_its_note(self):
        fake_docs = _FakeDocsService()
        fake_drive = _FakeDriveService()
        entries = [
            ImageReportEntry(
                image_name="FLIR0001.jpg",
                annotated_image=None,
                hotspot_rows=[],
                ambient_c=20.0,
                note="Visible corrosion on the bus bar connection",
                visual_anomaly=True,
            )
        ]
        build_findings_doc(fake_docs, fake_drive, entries, folder_id="fake-folder", style="audit")

        all_text_inserts = [
            req["insertText"]["text"] for call in fake_docs.batch_calls for req in call if "insertText" in req
        ]
        joined = "".join(all_text_inserts)
        assert "FLIR0001.jpg" in joined
        assert "Visible corrosion on the bus bar connection" in joined

    def test_gdocs_report_omits_unflagged_image_even_with_a_note(self):
        fake_docs = _FakeDocsService()
        fake_drive = _FakeDriveService()
        entries = [
            ImageReportEntry(
                image_name="FLIR0002.jpg",
                annotated_image=None,
                hotspot_rows=[],
                ambient_c=20.0,
                note="A description with no flag attached",
                visual_anomaly=False,
            )
        ]
        build_findings_doc(fake_docs, fake_drive, entries, folder_id="fake-folder", style="audit")

        all_text_inserts = [
            req["insertText"]["text"] for call in fake_docs.batch_calls for req in call if "insertText" in req
        ]
        joined = "".join(all_text_inserts)
        assert "FLIR0002.jpg" not in joined
