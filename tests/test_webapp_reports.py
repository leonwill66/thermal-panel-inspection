from __future__ import annotations


def _upload_and_get_run_id(client):
    resp = client.post(
        "/api/analyze",
        files=[("files", ("FLIR0001.jpg", b"fake-radiometric-bytes", "image/jpeg"))],
    )
    return resp.json()["run_id"]


class TestPdfReport:
    def test_pdf_report_full_style(self, logged_in_client, mock_radiometric):
        run_id = _upload_and_get_run_id(logged_in_client)
        resp = logged_in_client.post(f"/api/history/{run_id}/report", data={"report_style": "full"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content[:4] == b"%PDF"

    def test_pdf_report_audit_style(self, logged_in_client, mock_radiometric):
        run_id = _upload_and_get_run_id(logged_in_client)
        resp = logged_in_client.post(f"/api/history/{run_id}/report", data={"report_style": "audit"})
        assert resp.status_code == 200
        assert resp.content[:4] == b"%PDF"

    def test_invalid_report_style_rejected(self, logged_in_client, mock_radiometric):
        run_id = _upload_and_get_run_id(logged_in_client)
        resp = logged_in_client.post(f"/api/history/{run_id}/report", data={"report_style": "bogus"})
        assert resp.status_code == 422

    def test_nonexistent_run_404s(self, logged_in_client):
        resp = logged_in_client.post("/api/history/999999/report", data={"report_style": "full"})
        assert resp.status_code == 404

    def test_metadata_fields_accepted(self, logged_in_client, mock_radiometric):
        run_id = _upload_and_get_run_id(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/report",
            data={
                "report_style": "audit",
                "client_name": "Acme Corp",
                "site_location": "Building 4",
                "audit_date": "2026-08-09",
                "inspector_name": "Jane Doe",
                "report_id": "AUDIT-001",
            },
        )
        assert resp.status_code == 200

    def test_viewer_can_generate_report(self, client, make_user, logged_in_client, mock_radiometric):
        # report_from_history is explicitly documented as available to any
        # logged-in role, including viewer.
        run_id = _upload_and_get_run_id(logged_in_client)
        make_user("eve", role="viewer", password="password123")
        client.post("/api/login", data={"username": "eve", "password": "password123"})
        resp = client.post(f"/api/history/{run_id}/report", data={"report_style": "full"})
        assert resp.status_code == 200


class TestGoogleDocsReportUnconfigured:
    """conftest.py deliberately leaves GOOGLE_SERVICE_ACCOUNT_JSON/
    GOOGLE_DRIVE_FOLDER_ID unset, so this endpoint should always 501 rather
    than attempting a real (and untestable-without-live-credentials) Docs
    API call. See test_gdocs_report.py for the actual Docs API request
    logic, tested against a fake service instead."""

    def test_gdoc_export_501s_when_unconfigured(self, logged_in_client, mock_radiometric):
        run_id = _upload_and_get_run_id(logged_in_client)
        resp = logged_in_client.post(f"/api/history/{run_id}/report/gdoc", data={"report_style": "audit"})
        assert resp.status_code == 501
        assert "not configured" in resp.json()["detail"].lower() or "isn't configured" in resp.json()["detail"]

    def test_gdoc_export_501s_before_touching_the_db(self, logged_in_client):
        # The configured() check happens before the run lookup, so even a
        # nonexistent run_id should 501, not 404 - confirms no DB/Docs work
        # is attempted when credentials aren't set up.
        resp = logged_in_client.post("/api/history/999999/report/gdoc", data={"report_style": "audit"})
        assert resp.status_code == 501
