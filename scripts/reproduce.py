"""Refit the preserved analyses using verified, authorised original inputs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_inputs import verify
from scripts.verify_results import ROOT, check_recomputed


def run(experiment: str, data_root: Path) -> dict:
    problems = verify(experiment, data_root)
    if experiment == "campus":
        problems += verify("campania", data_root)
    if problems:
        raise ValueError(f"Input verification failed: {len(problems)} problem(s). " + "; ".join(problems[:3]))
    folders = {"milan": "daily_spatial_validation", "campania": "arpac_it1508", "campus": "salerno_campus"}
    output = ROOT / "reports" / folders[experiment]
    environment = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        environment[name] = "1"
    if experiment == "milan":
        command = [sys.executable, str(ROOT / "daily_spatial_validation.py"),
                   "--observations", str(data_root / "data/processed/good_dataset/milano_city_good_dataset.csv"),
                   "--cams", str(data_root / "data/processed/cams_milano_city_hourly.csv"),
                   "--weather", str(data_root / "data/processed/openmeteo_milano_city_hourly.csv"),
                   "--output-dir", str(output)]
    elif experiment == "campania":
        command = [sys.executable, str(ROOT / "arpac_it1508_replication.py"),
                   "--data-dir", str(data_root / "data/raw/arpac_salerno"), "--output-dir", str(output)]
    else:
        # The archived campus acquisition code rewrites its cache manifest.
        # Copy the cache into a temporary ignored directory so the author's
        # original archive remains unchanged; --offline makes no API calls.
        (ROOT / "generated").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="campus-cache-", dir=ROOT / "generated") as temporary:
            shutil.copytree(data_root / "data/raw/salerno_campus", temporary, dirs_exist_ok=True)
            command = [sys.executable, str(ROOT / "salerno_campus_case_study.py"),
                       "--raw-dir", temporary, "--arpac-dir", str(data_root / "data/raw/arpac_salerno"),
                       "--report-dir", str(output), "--offline", "--skip-figures"]
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
        return {"experiment": experiment, "verified_output_files": check_recomputed(output, experiment)}
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return {"experiment": experiment, "verified_output_files": check_recomputed(output, experiment)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["milan", "campania", "campus", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    args = parser.parse_args()
    names = ["milan", "campania", "campus"] if args.experiment == "all" else [args.experiment]
    try:
        results = [run(name, args.data_root.resolve()) for name in names]
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"Reproduction failed: {error}")
        return 1
    destination = ROOT / "generated/reproduction_checks.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
