"""Verify the frozen research inputs without exposing restricted device IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(experiment: str, data_root: Path) -> list[str]:
    entries = json.loads((ROOT / "config/inputs.json").read_text())[experiment]
    problems = []
    cache_index = {}
    if experiment == "campus":
        directory = data_root / "data/raw/salerno_campus"
        for path in directory.rglob("*.json"):
            relative = path.relative_to(directory).as_posix()
            cache_index[hashlib.sha256(relative.encode()).hexdigest()] = path
    for ordinal, entry in enumerate(entries, 1):
        path = (data_root / entry["path"] if "path" in entry
                else cache_index.get(entry["relative_path_sha256"]))
        label = entry.get("path", f"restricted cache artifact {ordinal}")
        if path is None or not path.is_file():
            problems.append(f"MISSING: {label}")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"HASH MISMATCH: {label}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["milan", "campania", "campus", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    args = parser.parse_args()
    names = ["milan", "campania", "campus"] if args.experiment == "all" else [args.experiment]
    problems = [problem for name in names for problem in verify(name, args.data_root)]
    if problems:
        print("\n".join(problems[:10]))
        print(f"{len(problems)} input problem(s). See docs/DATA_ACCESS.md.")
        return 1
    print("All selected inputs match the archived SHA-256 hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
