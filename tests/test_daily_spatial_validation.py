import unittest

import numpy as np
import pandas as pd

import daily_spatial_validation as validation


class DailySpatialValidationTests(unittest.TestCase):
    def test_local_date_uses_europe_rome_boundary(self):
        values = pd.Series(["2022-05-01T21:59:00Z", "2022-05-01T22:01:00Z"])
        result = validation._local_date(values)
        self.assertEqual(str(result.iloc[0].date()), "2022-05-01")
        self.assertEqual(str(result.iloc[1].date()), "2022-05-02")

    def test_protocol_produces_three_disjoint_test_origins(self):
        dates = pd.date_range("2022-05-01", periods=90, freq="D")
        self.assertEqual(validation._split_origins(dates, validation.Protocol()), [35, 49, 63])

    def test_fold_local_arpa_features_exclude_target_station(self):
        date = pd.Timestamp("2022-05-01")
        rows = pd.DataFrame(
            [
                {"local_date": date, "pollutant": "pm10", "source_id": "A", "lat": 45.0, "lon": 9.0, "target": 10.0},
                {"local_date": date, "pollutant": "pm10", "source_id": "B", "lat": 45.0, "lon": 9.1, "target": 20.0},
            ]
        )
        result = validation._add_arpa_features(rows.iloc[[0]], rows)
        self.assertEqual(result.iloc[0]["arpa_source_count"], 1.0)
        self.assertEqual(result.iloc[0]["arpa_nearest"], 20.0)

    def test_idw_exponent_sensitivity_uses_same_sources(self):
        target = pd.Series({"lat": 45.0, "lon": 9.0})
        sources = pd.DataFrame(
            [
                {"lat": 45.0, "lon": 9.01, "spatial_value": 10.0},
                {"lat": 45.0, "lon": 9.10, "spatial_value": 30.0},
            ]
        )
        result = validation._spatial_summary(target, sources, "arpa")
        self.assertLess(result["arpa_idw_p2"], result["arpa_idw"])
        self.assertLess(result["arpa_idw_p3"], result["arpa_idw_p2"])
        self.assertEqual(result["arpa_source_count"], 2.0)

    def test_moving_block_bootstrap_preserves_pooled_record_weighting(self):
        estimates = validation._moving_block_pooled_estimates(
            daily_sums=np.array([0.0, 9.0]),
            daily_counts=np.array([2.0, 1.0]),
            iterations=20,
            block_length=2,
            seed=42,
        )
        np.testing.assert_allclose(estimates, np.full(20, 3.0))


if __name__ == "__main__":
    unittest.main()
