"""Check inference boundaries, metadata integrity, and failure detection."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import salerno_campus_case_study as campus
from scripts import verify_inputs, verify_results


class ReproducibilityTests(unittest.TestCase):
    def test_public_claims_match_manuscript(self):
        self.assertEqual(verify_results.check_claims(verify_results.ROOT / "results")["checked_claim_rows"], 34)

    def test_artifact_hashes_detect_changes(self):
        self.assertGreater(verify_results.check_integrity(), 20)

    def test_missing_original_inputs_fail_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(verify_inputs.verify("milan", Path(directory))), 3)

    def test_regulatory_folds_have_no_heldout_or_temporal_leakage(self):
        base = verify_results.ROOT / "results"
        milan = verify_results.rows(base / "milan/split_assignments.csv")
        self.assertEqual(len(milan), 21)
        for row in milan:
            self.assertEqual(row["heldout_in_train"], "False")
            self.assertEqual(row["heldout_in_validation"], "False")
            self.assertEqual(row["test_time_overlap"], "False")
            self.assertGreater(int(row["test_rows"]), 0)
        campania = verify_results.rows(base / "campania/leakage_audit.csv")
        self.assertEqual(len(campania), 36)
        self.assertTrue(all(row["passed"] == "True" for row in campania))

    def test_campus_exact_signflip_resolution(self):
        p_value, permutations = campus._exact_signflip_pvalue(np.ones(4))
        self.assertEqual(permutations, 16)
        self.assertEqual(p_value, 0.125)

    def test_campus_exact_signflip_mixed_effects(self):
        p_value, _ = campus._exact_signflip_pvalue(np.array([-1.0, 1.0]))
        self.assertEqual(p_value, 1.0)

    def test_campus_idw_is_distinct_from_descriptive_map_rule(self):
        protocol = campus.Protocol()
        self.assertEqual(protocol.idw_power, 2)
        self.assertEqual(protocol.distance_floor_m, 10)
        peers = pd.DataFrame([{"lat": 40.0, "lon": 14.01, "pm10": 10.0},
                              {"lat": 40.0, "lon": 14.1, "pm10": 30.0}])
        target = pd.Series({"lat": 40.0, "lon": 14.0, "date": "2026-07-24"})
        values = campus._spatial_features(target, peers, "pm10", protocol)
        self.assertLess(values["idw"], values["network_mean"])

    def test_campus_evaluation_start_preserved(self):
        self.assertEqual(campus.TEST_START.isoformat(), "2026-07-24")

    def test_campus_interval_sign_conversion(self):
        directory = verify_results.export_tables()
        row = verify_results.select(directory / "replication_paired_effects.csv", experiment="campus", pollutant="pm10")
        self.assertGreater(float(row["difference"]), 0)
        self.assertLess(float(row["ci_low"]), 0)
        self.assertGreater(float(row["ci_high"]), 0)

    def test_verified_output_comparison_rejects_corruption(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            for path in (verify_results.ROOT / "results/campus").glob("*.csv"):
                shutil.copyfile(path, Path(directory) / path.name)
            path = Path(directory) / "metrics.csv"
            content = path.read_text()
            path.write_text(content.replace("1.7887558084040815", "9.7887558084040815"))
            with self.assertRaises(ValueError):
                verify_results.check_recomputed(Path(directory), "campus")


if __name__ == "__main__":
    unittest.main()
