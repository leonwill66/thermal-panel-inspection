"""Tests the emissivity-override recompute endpoint. Doesn't touch a real
FLIR file - load_radiometric_with_emissivity is monkeypatched to return a
controllable synthetic temperature array, the same technique conftest.py's
mock_radiometric fixture uses for /api/analyze itself."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_inspector.core import Thermogram


def _upload_and_get_ids(client):
    resp = client.post(
        "/api/analyze",
        files=[("files", ("FLIR0001.jpg", b"fake-radiometric-bytes", "image/jpeg"))],
    )
    data = resp.json()
    return data["run_id"], data["results"][0]["image_id"]


@pytest.fixture
def mock_recompute(monkeypatch):
    """Patches webapp.server.load_radiometric_with_emissivity so the
    recompute endpoint can be exercised without a real FLIR file. Returns a
    dict the test can mutate (`region_temp_c`) to control what the "recomputed"
    hotspot region reads as - the same bbox (10,10,10,10) mock_radiometric's
    synthetic upload puts its hotspot at, ambient elsewhere stays 20.0."""
    import webapp.server as server_module

    state = {"region_temp_c": 70.0}

    def _fake_load_radiometric_with_emissivity(path, emissivity, reflected_apparent_temperature=None):
        # Mirrors the real function's validation so endpoint-level tests of
        # the error path stay meaningful, even though the temperature math
        # itself is faked.
        if not (0.0 < emissivity <= 1.0):
            raise ValueError(f"emissivity must be in (0, 1], got {emissivity}")
        temp = np.full((60, 80), 20.0, dtype=np.float32)
        temp[10:20, 10:20] = state["region_temp_c"]
        return Thermogram(source_path=Path(path), temperature_c=temp, visual=None)

    monkeypatch.setattr(server_module, "load_radiometric_with_emissivity", _fake_load_radiometric_with_emissivity)
    return state


class TestRecomputeHappyPath:
    def test_recompute_updates_the_hotspot_row(self, logged_in_client, mock_radiometric, mock_recompute):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        mock_recompute["region_temp_c"] = 70.0  # ambient(20) + 50 delta -> critical_immediate

        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 0.3},
        )
        assert resp.status_code == 200
        row = resp.json()["hotspots"][0]
        assert row["max_temp_c"] == 70.0
        assert row["delta_t_c"] == 50.0  # unchanged ambient_c (20.0) is reused, not recomputed
        assert row["severity"] == "critical_immediate"
        assert row["emissivity_override"] == 0.3

    def test_persisted_in_history(self, logged_in_client, mock_radiometric, mock_recompute):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 0.3},
        )
        detail = logged_in_client.get(f"/api/history/{run_id}").json()
        assert detail["images"][0]["hotspots"][0]["emissivity_override"] == 0.3

    def test_logs_an_audit_entry(self, logged_in_client, mock_radiometric, mock_recompute):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 0.3},
        )
        log = logged_in_client.get(f"/api/history/{run_id}/audit-log").json()["entries"]
        assert len(log) == 1
        assert log[0]["action"] == "emissivity_override"
        assert log[0]["detail"]["after"]["emissivity_override"] == 0.3

    def test_ambient_is_not_recomputed(self, logged_in_client, mock_radiometric, mock_recompute):
        # Even though the fake "recomputed" frame's non-hotspot pixels are
        # still 20.0, changing them wouldn't matter - ambient_c must come
        # from the stored row, not a fresh percentile estimate.
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 0.3},
        )
        row = resp.json()["hotspots"][0]
        assert row["ambient_c"] == 20.0


class TestRecomputeErrors:
    def test_missing_raw_image_path_422s(self, logged_in_client, mock_radiometric, mock_recompute, db_session):
        from webapp.models import AnalysisImage

        run_id, image_id = _upload_and_get_ids(logged_in_client)
        img = db_session.get(AnalysisImage, image_id)
        img.raw_image_path = None
        db_session.commit()

        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 0.3},
        )
        assert resp.status_code == 422
        assert "predates raw-file storage" in resp.json()["detail"]

    def test_out_of_range_hotspot_index_422s(self, logged_in_client, mock_radiometric, mock_recompute):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 99, "emissivity": 0.3},
        )
        assert resp.status_code == 422

    def test_invalid_emissivity_422s(self, logged_in_client, mock_radiometric, mock_recompute):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 1.5},
        )
        assert resp.status_code == 422

    def test_wrong_run_id_404s(self, logged_in_client, mock_radiometric, mock_recompute):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        resp = logged_in_client.post(
            f"/api/history/{run_id + 999}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 0.3},
        )
        assert resp.status_code == 404

    def test_viewer_cannot_recompute(self, client, make_user, logged_in_client, mock_radiometric, mock_recompute):
        run_id, image_id = _upload_and_get_ids(logged_in_client)
        make_user("henry", role="viewer", password="password123")
        client.post("/api/login", data={"username": "henry", "password": "password123"})
        resp = client.post(
            f"/api/history/{run_id}/images/{image_id}/recompute-emissivity",
            json={"hotspot_index": 0, "emissivity": 0.3},
        )
        assert resp.status_code == 403
