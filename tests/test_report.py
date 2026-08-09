from __future__ import annotations

from thermal_inspector.core import ComparativeAnomaly, Hotspot
from thermal_inspector.report import comparative_to_rows, hotspots_to_rows, summarize


def make_hotspot(**overrides) -> Hotspot:
    defaults = dict(
        bbox=(10, 10, 5, 5),
        centroid=(12.5, 12.5),
        max_temp_c=45.123,
        mean_temp_c=40.456,
        ambient_c=20.0,
        delta_t_c=25.123,
        area_px=25,
        severity="critical",
        load_percent=None,
        delta_t_corrected_c=None,
    )
    defaults.update(overrides)
    return Hotspot(**defaults)


class TestHotspotsToRows:
    def test_fields_and_rounding(self):
        h = make_hotspot()
        rows = hotspots_to_rows("FLIR0001.jpg", [h])
        assert len(rows) == 1
        row = rows[0]
        assert row["image"] == "FLIR0001.jpg"
        assert row["severity"] == "critical"
        assert row["max_temp_c"] == 45.12
        assert row["delta_t_c"] == 25.12
        assert row["bbox_x"] == 10 and row["bbox_y"] == 10
        assert row["bbox_w"] == 5 and row["bbox_h"] == 5
        assert row["delta_t_corrected_c"] is None

    def test_corrected_delta_rounds_when_present(self):
        h = make_hotspot(delta_t_corrected_c=33.4567)
        rows = hotspots_to_rows("FLIR0001.jpg", [h])
        assert rows[0]["delta_t_corrected_c"] == 33.46

    def test_empty_list(self):
        assert hotspots_to_rows("FLIR0001.jpg", []) == []


class TestComparativeToRows:
    def test_fields(self):
        a = ComparativeAnomaly(label="Phase A", bbox=(1, 2, 3, 4), max_temp_c=50.555, delta_t_c=12.345, severity="comparative_major")
        rows = comparative_to_rows("FLIR0002.jpg", [a])
        assert rows == [
            {
                "image": "FLIR0002.jpg",
                "label": "Phase A",
                "severity": "comparative_major",
                "delta_t_c": 12.35,
                "max_temp_c": 50.55,
                "bbox_x": 1,
                "bbox_y": 2,
                "bbox_w": 3,
                "bbox_h": 4,
            }
        ]


class TestSummarize:
    def test_empty(self):
        summary = summarize([])
        assert summary["total_hotspots"] == 0
        assert summary["images_with_hotspots"] == 0
        assert summary["counts_by_severity"] == {}
        assert summary["worst_hotspot"] is None

    def test_counts_and_worst_by_raw_delta(self):
        rows = [
            {**hotspots_to_rows("a.jpg", [make_hotspot(severity="minor", delta_t_c=5.0)])[0]},
            {**hotspots_to_rows("a.jpg", [make_hotspot(severity="serious", delta_t_c=15.0)])[0]},
            {**hotspots_to_rows("b.jpg", [make_hotspot(severity="critical_immediate", delta_t_c=41.0)])[0]},
        ]
        summary = summarize(rows)
        assert summary["total_hotspots"] == 3
        assert summary["images_with_hotspots"] == 2  # a.jpg + b.jpg, not 3
        assert summary["counts_by_severity"] == {"minor": 1, "serious": 1, "critical_immediate": 1}
        assert summary["worst_hotspot"]["delta_t_c"] == 41.0

    def test_worst_prefers_load_corrected_delta_when_present(self):
        # Raw delta_t=10 but corrected to 80 should outrank a raw delta_t=50
        # row that has no correction - summarize should rank like find_hotspots does.
        low_raw_high_corrected = hotspots_to_rows(
            "a.jpg", [make_hotspot(delta_t_c=10.0, delta_t_corrected_c=80.0)]
        )[0]
        high_raw_no_correction = hotspots_to_rows("b.jpg", [make_hotspot(delta_t_c=50.0)])[0]
        summary = summarize([low_raw_high_corrected, high_raw_no_correction])
        assert summary["worst_hotspot"]["image"] == "a.jpg"
