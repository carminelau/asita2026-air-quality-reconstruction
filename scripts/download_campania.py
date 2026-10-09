"""Download ARPA Campania inputs from their original public provider URLs.

Provider resources can change. A changed resource is saved with .changed
suffix, never substituted silently for a frozen research input.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_inputs import ROOT, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/raw/arpac_salerno")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    files = json.loads((ROOT / "config/arpac_downloads.json").read_text())["files"]
    problems = 0
    for item in files:
        path = args.destination / item["path"]
        if path.is_file() and sha256(path) == item["sha256"]:
            print(f"Verified cached input: {path.name}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            request = urllib.request.Request(item["source_url"], headers={"User-Agent": "ASITA2026-reproducibility/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if sha256(temporary) != item["sha256"]:
                temporary.replace(path.with_suffix(path.suffix + ".changed"))
                print(f"Provider snapshot differs: {path.name}. Request the archived copy; see docs/DATA_ACCESS.md.")
                problems += 1
            else:
                temporary.replace(path)
                print(f"Downloaded and verified: {path.name}")
        except (OSError, urllib.error.URLError) as error:
            print(f"Download failed for {path.name}: {error}")
            problems += 1
    return int(problems > 0)


if __name__ == "__main__":
    raise SystemExit(main())
