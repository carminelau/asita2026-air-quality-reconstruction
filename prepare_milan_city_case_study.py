"""Download and prepare municipality-correct Milan case-study inputs.

The script uses public Square geography plus authenticated ``placeView`` daily
responses.  API-derived daily square averages are retained as descriptive
provenance only because the response does not identify the contributing source
for each value.  Validation inputs are filtered independently to the Comune di
Milano by upstream municipality metadata and ``Milano-Milano`` square IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


SQUAREJSON_URL = "https://square.sensesquare.eu/api/v1/squarejson"
PLACEVIEW_URL = "https://square.sensesquare.eu:5001/placeView"
APIKEY_ENVS = ("SENSE_SQUARE_APIKEY", "SENSESQUARE_APIKEY")
COUNTRY = "Italia"
REGION = "Lombardia"
PROVINCE = "Milano"
MUNICIPALITY = "Milano"
SQUARE_PREFIX = "Milano-Milano "

RAW_DIR = Path("data/raw/milano_city")
DAILY_DIR = RAW_DIR / "placeview_daily"
SQUARES_PATH = RAW_DIR / "milano_city_squares.geojson"
MANIFEST_PATH = RAW_DIR / "manifest.json"
API_DAILY_PATH = Path("data/processed/milano_city_square_daily.csv")
OBSERVATIONS_SOURCE = Path("data/processed/good_dataset/milano_good_dataset.csv")
CAMS_SOURCE = Path("data/processed/cams_milano_basiglio_hourly.csv")
WEATHER_SOURCE = Path("data/processed/openmeteo_milano_basiglio_hourly.csv")
OBSERVATIONS_OUTPUT = Path("data/processed/good_dataset/milano_city_good_dataset.csv")
CAMS_OUTPUT = Path("data/processed/cams_milano_city_hourly.csv")
WEATHER_OUTPUT = Path("data/processed/openmeteo_milano_city_hourly.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def api_key() -> str:
    for name in APIKEY_ENVS:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(f"Missing API key. Set one of: {', '.join(APIKEY_ENVS)}")


def post_json(session: requests.Session, url: str, data: dict[str, str], retries: int = 5) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.post(url, data=data, timeout=120)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(8.0, 0.75 * (2**attempt)))
    raise RuntimeError(f"POST failed after {retries} attempts: {url}") from last_error


def download_squares(session: requests.Session) -> dict[str, Any]:
    result = post_json(
        session,
        SQUAREJSON_URL,
        {"provincia": PROVINCE, "comune": MUNICIPALITY},
    )
    features = result.get("features", []) if isinstance(result, dict) else []
    names = [str(item.get("properties", {}).get("name", "")) for item in features]
    if result.get("type") != "FeatureCollection" or not features:
        raise RuntimeError("squarejson did not return a non-empty FeatureCollection")
    if len(names) != len(set(names)) or any(not name.startswith(SQUARE_PREFIX) for name in names):
        raise RuntimeError("squarejson returned duplicate or non-Milan square IDs")
    atomic_json(SQUARES_PATH, result)
    return result


def download_daily(
    session: requests.Session,
    key: str,
    start: date,
    end: date,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    requests_log: list[dict[str, Any]] = []
    for index, day in enumerate(date_range(start, end), start=1):
        raw_path = DAILY_DIR / f"{day.isoformat()}.json"
        if raw_path.exists() and not force:
            response = json.loads(raw_path.read_text(encoding="utf-8"))
            status = "cached"
        else:
            response = post_json(
                session,
                PLACEVIEW_URL,
                {
                    "apikey": key,
                    "zoom": "4",
                    "tempo": "giorno",
                    "date": day.isoformat(),
                    "nazione": COUNTRY,
                    "regione": REGION,
                    "provincia": PROVINCE,
                    "comune": MUNICIPALITY,
                },
            )
            atomic_json(raw_path, response)
            status = "downloaded"
            time.sleep(0.22)
        code = response.get("response_code") if isinstance(response, dict) else None
        result = response.get("result", {}) if isinstance(response, dict) else {}
        if code != 200 or not isinstance(result, dict):
            raise RuntimeError(f"placeView failed for {day}: response_code={code}")
        day_records = 0
        for square_id, payload in result.items():
            if not str(square_id).startswith(SQUARE_PREFIX) or not isinstance(payload, dict):
                continue
            row = dict(payload)
            row["squareID"] = str(square_id)
            row["request_date"] = day.isoformat()
            records.append(row)
            day_records += 1
        requests_log.append(
            {
                "date": day.isoformat(),
                "status": status,
                "response_code": code,
                "records": day_records,
                "raw_path": str(raw_path),
                "sha256": sha256_file(raw_path),
            }
        )
        if index % 15 == 0:
            print(f"placeView {index}/{(end - start).days + 1} days")
    return records, requests_log


def normalise_api_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    preferred = [
        "request_date",
        "timestamp",
        "squareID",
        "pm10",
        "pm2_5",
        "no2",
        "o3",
        "so2",
        "co",
        "aqi",
        "temperatura",
        "umidita",
        "nazione",
        "regione",
        "media",
    ]
    for column in preferred:
        if column not in frame:
            frame[column] = pd.NA
    numeric = ["pm10", "pm2_5", "no2", "o3", "so2", "co", "aqi", "temperatura", "umidita"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[preferred + sorted(set(frame.columns) - set(preferred))]
    return frame.sort_values(["request_date", "squareID"]).reset_index(drop=True)


def filter_city_inputs(square_names: set[str]) -> dict[str, Any]:
    observations = pd.read_csv(OBSERVATIONS_SOURCE, low_memory=False)
    source_type = observations["source_type"].fillna("").astype(str)
    grid_id = observations["grid_id"].fillna("").astype(str)
    municipality = observations["municipality"].fillna("").astype(str)
    arpa_city = source_type.eq("arpa") & municipality.eq(MUNICIPALITY)
    low_cost_city = source_type.str.startswith("sensesquare") & grid_id.isin(square_names)
    city_observations = observations[arpa_city | low_cost_city].copy()
    atomic_csv(OBSERVATIONS_OUTPUT, city_observations)

    hourly_stats: dict[str, Any] = {}
    for source_path, output_path, label in (
        (CAMS_SOURCE, CAMS_OUTPUT, "cams"),
        (WEATHER_SOURCE, WEATHER_OUTPUT, "weather"),
    ):
        frame = pd.read_csv(source_path, low_memory=False)
        city = frame[frame["grid_id"].astype(str).isin(square_names)].copy()
        if city["grid_id"].nunique() != len(square_names):
            raise RuntimeError(
                f"{label} grid mismatch: {city['grid_id'].nunique()} found, {len(square_names)} expected"
            )
        atomic_csv(output_path, city)
        hourly_stats[label] = {
            "rows": int(len(city)),
            "grid_cells": int(city["grid_id"].nunique()),
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        }

    inventory = city_observations[
        ["source_id", "source_type", "grid_id", "municipality", "lat", "lon"]
    ].drop_duplicates("source_id")
    return {
        "observations": {
            "rows": int(len(city_observations)),
            "sources": int(city_observations["source_id"].nunique()),
            "arpa_sources": int(inventory["source_type"].eq("arpa").sum()),
            "low_cost_sources": int(inventory["source_type"].astype(str).str.startswith("sensesquare").sum()),
            "path": str(OBSERVATIONS_OUTPUT),
            "sha256": sha256_file(OBSERVATIONS_OUTPUT),
        },
        **hourly_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 5, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2022, 7, 29))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must not precede --start")

    key = api_key()
    session = requests.Session()
    session.headers.update({"User-Agent": "AI-Spatial-Expansion-ASITA2026/1.0"})
    squares = download_squares(session)
    square_names = {
        str(feature["properties"]["name"])
        for feature in squares["features"]
    }
    raw_records, requests_log = download_daily(session, key, args.start, args.end, args.force)
    api_daily = normalise_api_records(raw_records)
    atomic_csv(API_DAILY_PATH, api_daily)
    city_inputs = filter_city_inputs(square_names)

    pollutant_counts = {
        column: int(api_daily[column].notna().sum())
        for column in ("pm10", "pm2_5")
    }
    manifest = {
        "scope": {
            "country": COUNTRY,
            "region": REGION,
            "province": PROVINCE,
            "municipality": MUNICIPALITY,
            "square_prefix": SQUARE_PREFIX,
        },
        "period": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "api": {
            "squarejson": SQUAREJSON_URL,
            "placeView": PLACEVIEW_URL,
            "authentication": "API key read from environment; never persisted",
            "route_selection": {
                "selected": "placeView",
                "rejected": "download",
                "reason": "authenticated preflight returned HTTP 500 on 2026-08-27",
            },
        },
        "geography": {
            "square_count": len(square_names),
            "path": str(SQUARES_PATH),
            "sha256": sha256_file(SQUARES_PATH),
        },
        "api_daily": {
            "records": int(len(api_daily)),
            "dates": int(api_daily["request_date"].nunique()),
            "squares_with_any_response": int(api_daily["squareID"].nunique()),
            "nonnull_pollutant_values": pollutant_counts,
            "path": str(API_DAILY_PATH),
            "sha256": sha256_file(API_DAILY_PATH),
            "role": "descriptive API evidence only; excluded from validation features and targets because per-record source attribution is absent",
        },
        "requests": requests_log,
        "validation_inputs": city_inputs,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "requests": requests.__version__,
        },
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "squares": len(square_names),
                "api_records": len(api_daily),
                "api_dates": api_daily["request_date"].nunique(),
                "validation_inputs": city_inputs,
                "manifest": str(MANIFEST_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
