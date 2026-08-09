"""Tests cross-run trending: tagging images with an asset_label via the
/note endpoint, then correlating them across separate runs via /api/trend."""

from __future__ import annotations


def _upload_and_get_ids(client, filename="FLIR0001.jpg"):
    resp = client.post(
        "/api/analyze",
        files=[("files", (filename, b"fake-radiometric-bytes", "image/jpeg"))],
    )
    data = resp.json()
    return data["run_id"], data["results"][0]["image_id"]


class TestSettingAssetLabel:
    def test_label_is_saved_and_readable(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/note", json={"asset_label": "Main Panel - Breaker 3"}
        )
        assert resp.json()["asset_label"] == "Main Panel - Breaker 3"

        detail = logged_in_client.get(f"/api/history/{run_id}").json()
        assert detail["images"][0]["asset_label"] == "Main Panel - Breaker 3"

    def test_blank_label_is_stored_as_none(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"asset_label": "Breaker 3"})
        resp = logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"asset_label": "   "})
        assert resp.json()["asset_label"] is None

    def test_label_change_is_audit_logged(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"asset_label": "Breaker 3"})
        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert len(log) == 1
        assert log[0]["detail"]["after"]["asset_label"] == "Breaker 3"


class TestAssetLabelList:
    def test_lists_distinct_labels_across_runs(self, logged_in_client, mock_radiometric):
        run1, img1 = _upload_and_get_ids(logged_in_client, "a.jpg")
        run2, img2 = _upload_and_get_ids(logged_in_client, "b.jpg")
        logged_in_client.post(f"/api/history/{run1}/images/{img1}/note", json={"asset_label": "Breaker 3"})
        logged_in_client.post(f"/api/history/{run2}/images/{img2}/note", json={"asset_label": "Breaker 5"})

        labels = logged_in_client.get("/api/trend/labels").json()["labels"]
        assert labels == ["Breaker 3", "Breaker 5"]

    def test_untagged_images_are_excluded(self, logged_in_client, mock_radiometric):
        _upload_and_get_ids(logged_in_client)  # never tagged
        labels = logged_in_client.get("/api/trend/labels").json()["labels"]
        assert labels == []


class TestTrendLookup:
    def test_correlates_the_same_label_across_separate_runs(self, logged_in_client, mock_radiometric):
        run1, img1 = _upload_and_get_ids(logged_in_client, "visit1.jpg")
        run2, img2 = _upload_and_get_ids(logged_in_client, "visit2.jpg")
        logged_in_client.post(f"/api/history/{run1}/images/{img1}/note", json={"asset_label": "Breaker 3"})
        logged_in_client.post(f"/api/history/{run2}/images/{img2}/note", json={"asset_label": "Breaker 3"})

        trend = logged_in_client.get("/api/trend", params={"label": "Breaker 3"}).json()
        assert trend["label"] == "Breaker 3"
        assert len(trend["points"]) == 2
        assert {p["filename"] for p in trend["points"]} == {"visit1.jpg", "visit2.jpg"}
        # mock_radiometric's synthetic hotspot is always the same severity/delta.
        assert all(p["worst_severity"] == "critical" for p in trend["points"])

    def test_differently_labeled_images_are_not_mixed_in(self, logged_in_client, mock_radiometric):
        run1, img1 = _upload_and_get_ids(logged_in_client, "a.jpg")
        run2, img2 = _upload_and_get_ids(logged_in_client, "b.jpg")
        logged_in_client.post(f"/api/history/{run1}/images/{img1}/note", json={"asset_label": "Breaker 3"})
        logged_in_client.post(f"/api/history/{run2}/images/{img2}/note", json={"asset_label": "Breaker 5"})

        trend = logged_in_client.get("/api/trend", params={"label": "Breaker 3"}).json()
        assert len(trend["points"]) == 1
        assert trend["points"][0]["filename"] == "a.jpg"

    def test_unknown_label_returns_empty_points(self, logged_in_client):
        trend = logged_in_client.get("/api/trend", params={"label": "Nonexistent Component"}).json()
        assert trend["points"] == []

    def test_excluded_hotspot_is_not_counted_as_worst(self, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"asset_label": "Breaker 3"})
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/exclude", json={"hotspot_indices": [0]})

        trend = logged_in_client.get("/api/trend", params={"label": "Breaker 3"}).json()
        assert trend["points"][0]["worst_severity"] is None

    def test_requires_login(self, client):
        resp = client.get("/api/trend", params={"label": "x"})
        assert resp.status_code == 401
        resp2 = client.get("/api/trend/labels")
        assert resp2.status_code == 401

    def test_viewer_can_read_trends(self, client, make_user, logged_in_client, mock_radiometric):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(f"/api/history/{run_id}/images/{image_id}/note", json={"asset_label": "Breaker 3"})

        make_user("ivan", role="viewer", password="password123")
        client.post("/api/login", data={"username": "ivan", "password": "password123"})
        resp = client.get("/api/trend", params={"label": "Breaker 3"})
        assert resp.status_code == 200
        assert len(resp.json()["points"]) == 1
