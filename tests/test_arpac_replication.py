import unittest

import pandas as pd

import arpac_it1508_replication as replication


class ArpacReplicationTests(unittest.TestCase):
    def test_idw_exponent_sensitivity_uses_same_sources(self):
        target = pd.Series({"lat": 40.0, "lon": 14.0})
        sources = pd.DataFrame(
            [
                {"station_id": "near", "lat": 40.0, "lon": 14.01, "target": 10.0},
                {"station_id": "far", "lat": 40.0, "lon": 14.10, "target": 30.0},
            ]
        )
        result = replication._spatial_features(target, sources)
        self.assertLess(result["spatial_idw_p2"], result["spatial_idw"])
        self.assertLess(result["spatial_idw_p3"], result["spatial_idw_p2"])
        self.assertEqual(result["spatial_source_count"], 2.0)
        self.assertEqual(result["spatial_source_ids"], "far|near")


if __name__ == "__main__":
    unittest.main()
