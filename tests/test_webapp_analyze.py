from __future__ import annotations


def _upload(client, filename="FLIR0001.jpg", **form):
    return client.post(
        "/api/analyze",
        files=[("files", (filename, b"fake-radiometric-bytes", "image/jpeg"))],
        data=form,
    )


class TestAnalyze:
    def test_analyze_creates_a_run_with_expected_hotspot(self, logged_in_client, mock_radiometric):
        resp = _upload(logged_in_client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] is not None
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["filename"] == "FLIR0001.jpg"
        assert len(result["hotspots"]) == 1
        assert result["hotspots"][0]["severity"] == "critical"
        assert data["summary"]["total_hotspots"] == 1

    def test_no_files_rejected(self, logged_in_client):
        resp = logged_in_client.post("/api/analyze", files=[])
        assert resp.status_code == 422

    def test_partial_roi_rejected(self, logged_in_client, mock_radiometric):
        resp = _upload(logged_in_client, roi_x=0, roi_y=0)  # missing roi_w/roi_h
        assert resp.status_code == 422

    def test_higher_min_delta_finds_nothing(self, logged_in_client, mock_radiometric):
        # The synthetic hotspot is +25C over ambient; a 30C threshold should
        # find nothing even though the same region would normally trigger.
        resp = _upload(logged_in_client, min_delta=30.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["hotspots"] == []
        assert data["summary"]["total_hotspots"] == 0

    def test_run_appears_in_history(self, logged_in_client, mock_radiometric):
        resp = _upload(logged_in_client)
        run_id = resp.json()["run_id"]
        hist = logged_in_client.get("/api/history")
        assert hist.status_code == 200
        run_ids = [r["id"] for r in hist.json()["runs"]]
        assert run_id in run_ids

    def test_run_detail_matches_analyze_result(self, logged_in_client, mock_radiometric):
        run_id = _upload(logged_in_client).json()["run_id"]
        detail = logged_in_client.get(f"/api/history/{run_id}").json()
        assert detail["images"][0]["filename"] == "FLIR0001.jpg"
        assert len(detail["images"][0]["hotspots"]) == 1

    def test_stored_image_is_retrievable(self, logged_in_client, mock_radiometric):
        run_id = _upload(logged_in_client).json()["run_id"]
        image_id = logged_in_client.get(f"/api/history/{run_id}").json()["images"][0]["id"]
        resp = logged_in_client.get(f"/api/history/{run_id}/image/{image_id}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 0


class TestExcludeHotspot:
    def test_excluding_a_finding_updates_summary(self, logged_in_client, mock_radiometric):
        run_id = _upload(logged_in_client).json()["run_id"]
        image_id = logged_in_client.get(f"/api/history/{run_id}").json()["images"][0]["id"]

        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": [0]}
        )
        assert resp.status_code == 200
        assert resp.json()["hotspots"][0]["excluded"] is True

        detail = logged_in_client.get(f"/api/history/{run_id}").json()
        assert detail["summary"]["total_hotspots"] == 0  # excluded, so no longer counted

    def test_out_of_range_index_rejected(self, logged_in_client, mock_radiometric):
        run_id = _upload(logged_in_client).json()["run_id"]
        image_id = logged_in_client.get(f"/api/history/{run_id}").json()["images"][0]["id"]
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": [99]}
        )
        assert resp.status_code == 422

    def test_wrong_run_id_404s(self, logged_in_client, mock_radiometric):
        run_id = _upload(logged_in_client).json()["run_id"]
        image_id = logged_in_client.get(f"/api/history/{run_id}").json()["images"][0]["id"]
        resp = logged_in_client.post(
            f"/api/history/{run_id + 999}/images/{image_id}/exclude", json={"hotspot_indices": []}
        )
        assert resp.status_code == 404


class TestViewerReadOnlyAccess:
    def test_viewer_can_browse_history(self, client, make_user, logged_in_client, mock_radiometric):
        run_id = _upload(logged_in_client).json()["run_id"]
        make_user("dave", role="viewer", password="password123")
        client.post("/api/login", data={"username": "dave", "password": "password123"})
        resp = client.get(f"/api/history/{run_id}")
        assert resp.status_code == 200

    def test_viewer_cannot_exclude_findings(self, client, make_user, logged_in_client, mock_radiometric):
        run_id = _upload(logged_in_client).json()["run_id"]
        image_id = logged_in_client.get(f"/api/history/{run_id}").json()["images"][0]["id"]
        make_user("dave", role="viewer", password="password123")
        client.post("/api/login", data={"username": "dave", "password": "password123"})
        resp = client.post(f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": []})
        assert resp.status_code == 403
