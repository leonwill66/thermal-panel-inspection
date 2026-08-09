"""Tests the audit trail: reviewer actions (excluding a hotspot, flagging a
visual anomaly) get an append-only log entry recording who did it and when,
retrievable via /api/history/{run_id}/audit-log."""

from __future__ import annotations


def _upload_and_get_ids(client):
    resp = client.post(
        "/api/analyze",
        files=[("files", ("FLIR0001.jpg", b"fake-radiometric-bytes", "image/jpeg"))],
    )
    data = resp.json()
    return data["run_id"], data["results"][0]["image_id"]


class TestAuditLogForExclusions:
    def test_excluding_a_hotspot_logs_an_entry(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": [0]})

        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert len(log) == 1
        assert log[0]["action"] == "exclude_hotspots"
        assert log[0]["username"] == "admin"
        assert log[0]["image_id"] == image_id
        assert log[0]["detail"] == {"before": [], "after": [0]}

    def test_re_saving_the_same_state_does_not_log_again(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": [0]})
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": [0]})

        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert len(log) == 1

    def test_no_op_exclude_call_does_not_log(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": []})

        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert log == []


class TestAuditLogForVisualAnomaly:
    def test_flagging_an_anomaly_logs_an_entry(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/note", json={"note": "Cracked door", "anomaly": True}
        )

        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert len(log) == 1
        assert log[0]["action"] == "set_visual_anomaly"
        assert log[0]["detail"]["after"] == {"visual_note": "Cracked door", "visual_anomaly": True}
        assert log[0]["detail"]["before"] == {"visual_note": None, "visual_anomaly": False}

    def test_unchanged_note_does_not_log(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={})  # note=None, anomaly=False - matches defaults

        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert log == []

    def test_entries_are_newest_first(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"anomaly": True})
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": [0]})

        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert [e["action"] for e in log] == ["exclude_hotspots", "set_visual_anomaly"]


class TestAuditLogAccess:
    def test_wrong_run_404s(self, logged_in_client):
        resp = logged_in_client.get("/api/history/999999/audit-log")
        assert resp.status_code == 404

    def test_requires_login(self, client):
        resp = client.get("/api/history/1/audit-log")
        assert resp.status_code == 401

    def test_viewer_can_read_the_log(self, client, make_user, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"anomaly": True})

        make_user("gina", role="viewer", password="password123")
        client.post("/api/login", data={"username": "gina", "password": "password123"})
        resp = client.get(f"/api/history/{run_id}/audit-log")
        assert resp.status_code == 200
        assert len(resp.json()["entries"]) == 1
