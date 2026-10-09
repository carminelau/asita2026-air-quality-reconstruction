"""Check the public result files against the manuscript's numerical claims.

This checks archived aggregate results; it does not refit models or recreate
observations. Use reproduce.py and the original inputs for that second step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select(path: Path, **filters: str) -> dict[str, str]:
    selected = [row for row in rows(path) if all(row.get(k) == v for k, v in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one row in {path.name} for {filters}; found {len(selected)}")
    return selected[0]


def check_claims(results: Path) -> dict:
    claims = json.loads((ROOT / "config/claims.json").read_text())
    for claim in claims:
        row = select(results / claim["file"], **claim["select"])
        for field, expected in claim["expected"].items():
            actual = float(row[field])
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=claim.get("tolerance", 0.00051)):
                raise ValueError(f"{claim['file']}: {field}={actual}, expected {expected}")
    return {"checked_claim_rows": len(claims), "status": "passed", "mode": "aggregate_result_verification"}


def check_integrity() -> int:
    manifest = json.loads((ROOT / "config/checksums.json").read_text())
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Repository artifact hash mismatch: {name}")
    return len(manifest)


def check_recomputed(directory: Path, experiment: str) -> int:
    files = {
        "milan": ["metrics.csv", "bootstrap_ci.csv", "paired_bootstrap_ci.csv"],
        "campania": ["metrics.csv", "paired_comparisons.csv", "sensitivity_metrics.csv", "sensitivity_paired_comparisons.csv"],
        "campus": ["metrics.csv", "bootstrap_ci.csv", "paired_bootstrap_ci.csv", "device_signflip_test.csv"],
    }[experiment]
    # Row order is part of the frozen pipeline output. Float serialization can
    # differ between operating systems; statistics must agree within 1e-8.
    for name in files:
        expected = rows(ROOT / "results" / experiment / name)
        actual = rows(directory / name)
        if len(actual) != len(expected):
            raise ValueError(f"Recomputed {name}: row count differs")
        for left, right in zip(expected, actual):
            if left.keys() != right.keys():
                raise ValueError(f"Recomputed {name}: columns differ")
            for key, value in left.items():
                other = right[key]
                try:
                    a, b = float(value), float(other)
                except (ValueError, TypeError):
                    equal = value == other
                else:
                    equal = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-10)
                if not equal:
                    raise ValueError(f"Recomputed {name}: {key} differs ({value} vs {other})")
    return len(files)


def export_tables() -> Path:
    destination = ROOT / "generated/tables"
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results"
    milan = [row for row in rows(source / "milan/metrics.csv") if row["scope"] == "overall"]
    fields = ["model", "n", "mae", "rmse", "bias", "r2"]
    with (destination / "milan_models.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in milan)
    paired = select(source / "campania/paired_comparisons.csv", scope="overall", model="rf", pollutant="all")
    campus = [row for row in rows(source / "campus/paired_bootstrap_ci.csv") if row["model"] == "network_median"]
    # Campus archives store median minus IDW. The paper uses competing IDW
    # minus reference median: negate the estimate and reverse interval bounds.
    output = [{"experiment": "campania", "pollutant": "all", "difference": float(paired["mean_mae_difference"]),
               "ci_low": float(paired["difference_ci_low"]), "ci_high": float(paired["difference_ci_high"])}]
    output += [{"experiment": "campus", "pollutant": row["pollutant"], "difference": -float(row["estimate"]),
                "ci_low": -float(row["ci_upper"]), "ci_high": -float(row["ci_lower"])} for row in campus]
    with (destination / "replication_paired_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-tables", action="store_true")
    parser.add_argument("--recomputed", type=Path)
    parser.add_argument("--experiment", choices=["milan", "campania", "campus"])
    args = parser.parse_args()
    try:
        checked = check_integrity()
        result = check_claims(ROOT / "results")
        result["verified_file_hashes"] = checked
        if args.recomputed:
            if not args.experiment:
                parser.error("--experiment is required with --recomputed")
            result["recomputed_files_verified"] = check_recomputed(args.recomputed, args.experiment)
        if args.export_tables:
            result["table_directory"] = str(export_tables().relative_to(ROOT))
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Verification failed: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
