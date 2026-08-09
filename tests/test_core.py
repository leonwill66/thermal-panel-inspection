from __future__ import annotations

import numpy as np
import pytest

from thermal_inspector.core import (
    DEFAULT_THRESHOLDS,
    classify_comparative_severity,
    classify_severity,
    find_comparative_anomalies,
    find_hotspots,
    load_adjusted_delta_t,
)


def flat_plate_with_hotspot(ambient=20.0, delta=25.0, region=(10, 10, 10, 10), height=60, width=80):
    temp = np.full((height, width), ambient, dtype=np.float32)
    x, y, w, h = region
    temp[y : y + h, x : x + w] = ambient + delta
    return temp


class TestClassifySeverity:
    @pytest.mark.parametrize(
        "delta_t, expected",
        [
            (45.0, "critical_immediate"),
            (40.0, "critical_immediate"),  # boundary is inclusive
            (39.9, "critical"),
            (20.0, "critical"),
            (19.9, "serious"),
            (10.0, "serious"),
            (9.9, "minor"),
            (0.0, "minor"),
            (-5.0, "minor"),  # below every cutoff still falls to the lowest band, not an error
        ],
    )
    def test_thresholds(self, delta_t, expected):
        assert classify_severity(delta_t, DEFAULT_THRESHOLDS) == expected


class TestFindHotspots:
    def test_finds_the_hot_region(self):
        temp = flat_plate_with_hotspot(ambient=20.0, delta=25.0, region=(10, 10, 10, 10))
        hotspots, ambient_used = find_hotspots(temp, min_delta_c=8.0, min_area_px=25)
        assert len(hotspots) == 1
        h = hotspots[0]
        assert h.severity == "critical"  # delta_t ~25 falls in [20, 40)
        assert h.bbox == (10, 10, 10, 10)
        assert h.max_temp_c == pytest.approx(45.0)
        assert h.delta_t_c == pytest.approx(45.0 - ambient_used, abs=0.5)

    def test_no_hotspot_on_uniform_plate(self):
        temp = np.full((60, 80), 20.0, dtype=np.float32)
        hotspots, _ = find_hotspots(temp, min_delta_c=8.0, min_area_px=25)
        assert hotspots == []

    def test_region_smaller_than_min_area_is_ignored(self):
        temp = flat_plate_with_hotspot(ambient=20.0, delta=25.0, region=(10, 10, 3, 3))  # 9px < min_area_px
        hotspots, _ = find_hotspots(temp, min_delta_c=8.0, min_area_px=25)
        assert hotspots == []

    def test_explicit_ambient_overrides_estimate(self):
        temp = flat_plate_with_hotspot(ambient=20.0, delta=25.0, region=(10, 10, 10, 10))
        hotspots, ambient_used = find_hotspots(temp, ambient_c=15.0, min_delta_c=8.0, min_area_px=25)
        assert ambient_used == 15.0
        assert hotspots[0].delta_t_c == pytest.approx(45.0 - 15.0)

    def test_roi_restricts_search_area(self):
        temp = flat_plate_with_hotspot(ambient=20.0, delta=25.0, region=(50, 10, 10, 10))
        # ROI that excludes the hotspot entirely
        hotspots, _ = find_hotspots(temp, min_delta_c=8.0, min_area_px=25, roi=(0, 0, 30, 30))
        assert hotspots == []
        # ROI that includes it - bbox still reported in full-frame coordinates
        hotspots, _ = find_hotspots(temp, min_delta_c=8.0, min_area_px=25, roi=(40, 0, 40, 30))
        assert len(hotspots) == 1
        assert hotspots[0].bbox == (50, 10, 10, 10)

    def test_roi_outside_frame_raises(self):
        temp = np.full((60, 80), 20.0, dtype=np.float32)
        with pytest.raises(ValueError):
            find_hotspots(temp, roi=(0, 0, 1000, 1000))

    def test_load_percent_corrects_severity(self):
        # Raw delta_t=15 ("serious"), but at 50% load the corrected value
        # (15 * (100/50)^2 = 60) pushes it to "critical_immediate".
        temp = flat_plate_with_hotspot(ambient=20.0, delta=15.0, region=(10, 10, 10, 10))
        hotspots, _ = find_hotspots(temp, min_delta_c=8.0, min_area_px=25, load_percent=50.0)
        assert len(hotspots) == 1
        h = hotspots[0]
        assert h.delta_t_c == pytest.approx(15.0, abs=0.5)
        assert h.delta_t_corrected_c == pytest.approx(60.0, abs=2.0)
        assert h.severity == "critical_immediate"

    def test_results_sorted_worst_first(self):
        temp = np.full((60, 80), 20.0, dtype=np.float32)
        temp[10:20, 10:20] = 35.0  # delta 15
        temp[40:50, 40:50] = 60.0  # delta 40
        hotspots, _ = find_hotspots(temp, min_delta_c=8.0, min_area_px=25)
        assert len(hotspots) == 2
        assert hotspots[0].delta_t_c > hotspots[1].delta_t_c


class TestLoadAdjustedDeltaT:
    def test_full_load_is_unchanged(self):
        assert load_adjusted_delta_t(10.0, 100.0) == pytest.approx(10.0)

    def test_half_load_quadruples(self):
        assert load_adjusted_delta_t(10.0, 50.0) == pytest.approx(40.0)

    def test_zero_or_negative_load_raises(self):
        with pytest.raises(ValueError):
            load_adjusted_delta_t(10.0, 0.0)
        with pytest.raises(ValueError):
            load_adjusted_delta_t(10.0, -5.0)


class TestComparativeAnomalies:
    def test_needs_at_least_two_regions(self):
        temp = np.full((60, 80), 20.0, dtype=np.float32)
        with pytest.raises(ValueError):
            find_comparative_anomalies(temp, [((0, 0, 5, 5), "only one")])

    def test_coolest_region_is_the_baseline(self):
        temp = np.full((60, 80), 20.0, dtype=np.float32)
        temp[0:5, 0:5] = 20.0  # Phase A - coolest
        temp[0:5, 10:15] = 25.0  # Phase B - +5
        temp[0:5, 20:25] = 40.0  # Phase C - +20, way outside normal
        regions = [
            ((0, 0, 5, 5), "Phase A"),
            ((10, 0, 5, 5), "Phase B"),
            ((20, 0, 5, 5), "Phase C"),
        ]
        anomalies = find_comparative_anomalies(temp, regions)
        by_label = {a.label: a for a in anomalies}
        assert by_label["Phase A"].delta_t_c == pytest.approx(0.0)
        assert by_label["Phase A"].severity is None
        assert by_label["Phase B"].delta_t_c == pytest.approx(5.0)
        assert by_label["Phase C"].delta_t_c == pytest.approx(20.0)
        assert by_label["Phase C"].severity == "comparative_major"
        # sorted worst-first
        assert anomalies[0].label == "Phase C"

    def test_region_outside_frame_raises(self):
        temp = np.full((60, 80), 20.0, dtype=np.float32)
        with pytest.raises(ValueError):
            find_comparative_anomalies(temp, [((0, 0, 5, 5), "A"), ((0, 0, 1000, 1000), "B")])


class TestClassifyComparativeSeverity:
    def test_below_lowest_threshold_is_none(self):
        assert classify_comparative_severity(0.5) is None

    def test_bands(self):
        assert classify_comparative_severity(1.0) == "comparative_possible"
        assert classify_comparative_severity(4.0) == "comparative_probable"
        assert classify_comparative_severity(15.0) == "comparative_major"
