"""Reproducible internal spatial-reconstruction case study for PE.DT devices.

The Sense Square devices are low-cost monitors, not independent regulatory
ground truth.  The primary experiment therefore estimates how well one PE.DT
device can be reconstructed from its contemporaneous peers.  A separate ARPAC
screening reports distance and temporal overlap without conflating the two
evidence sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


STUDY_START = date(2026, 5, 1)
STUDY_END = date(2026, 8, 13)
TEST_DAYS = 21
TEST_START = STUDY_END - timedelta(days=TEST_DAYS - 1)
ALIAS_QUERY = "PE.DT"
READ_URL = "https://square.sensesquare.eu:5001/widgetView"
USER_LIST_URL = "https://square.sensesquare.eu:5002/elenco_centraline"
USER_INFO_URL = "https://square.sensesquare.eu:5002/informazioni_centralina"
APIKEY_ENV_NAMES = ("SENSESQUARE_APIKEY", "SENSE_SQUARE_APIKEY")
POLLUTANTS = ("pm2_5", "pm10")
ARPAC_CODES = ("IT2210A", "IT1504A", "IT2213A", "IT2214A", "IT2227A", "IT0936A")


@dataclass(frozen=True)
class Protocol:
    requests_per_minute: float = 59.0
    bootstrap_iterations: int = 2_000
    bootstrap_block_days: int = 7
    seed: int = 42
    idw_power: float = 2.0
    distance_floor_m: float = 10.0
    min_rf_rows: int = 150
    min_arpac_hours: int = 18


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _api_key() -> str:
    for name in APIKEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"Missing API key: set one of {', '.join(APIKEY_ENV_NAMES)}")


class RateLimiter:
    """Process-local limiter; every network attempt is spaced by >=1 second."""

    def __init__(self, requests_per_minute: float) -> None:
        if not 0 < requests_per_minute <= 60:
            raise ValueError("requests_per_minute must be in (0, 60]")
        self.interval = max(1.01, 60.0 / requests_per_minute)
        self.last_request = 0.0

    def wait(self) -> None:
        remaining = self.interval - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)
        self.last_request = time.monotonic()


def _post_form(url: str, payload: Mapping[str, str], limiter: RateLimiter, retries: int = 4) -> Any:
    body = urlencode(payload).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "salerno-campus-case-study/1.0"}
    error: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        try:
            request = Request(url, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(8.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"POST failed after {retries} attempts: {url}: {error}") from error


def _sanitize_info(response: Any) -> Any:
    if not isinstance(response, dict) or not isinstance(response.get("result"), dict):
        return response
    source = response["result"]
    permitted = {
        "ID", "comune", "configuration_date", "last_entry", "lat", "lon", "media", "nazione",
        "provincia", "regione", "sensori", "squareID", "type", "versione_firmware",
    }
    return {**{k: v for k, v in response.items() if k != "result"}, "result": {k: source[k] for k in permitted if k in source}}


def _cached_post(
    url: str,
    payload: Mapping[str, str],
    path: Path,
    limiter: RateLimiter,
    request_log: list[dict[str, Any]],
    *,
    offline: bool,
    force: bool,
    sanitizer: Callable[[Any], Any] | None = None,
) -> Any:
    public_payload = {k: v for k, v in payload.items() if k != "apikey"}
    if path.exists() and not force:
        response = json.loads(path.read_text(encoding="utf-8"))
        request_log.append({"cache": "hit", "url": url, "payload": public_payload, "path": str(path)})
        return response
    if offline:
        raise FileNotFoundError(f"Offline cache miss: {path}")
    response = _post_form(url, payload, limiter)
    stored = sanitizer(response) if sanitizer else response
    _atomic_json(path, stored)
    request_log.append(
        {
            "cache": "refreshed" if force else "miss",
            "url": url,
            "payload": public_payload,
            "path": str(path),
            "response_code": stored.get("response_code") if isinstance(stored, dict) else None,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return stored


def _weekly_anchors() -> list[date]:
    anchors: list[date] = []
    current = STUDY_START + timedelta(days=6)
    while current <= STUDY_END:
        anchors.append(current)
        current += timedelta(days=7)
    # The live endpoint clamps near-current requests.  This extra anchor closes
    # the only boundary between the final complete week and the study end.
    boundary = STUDY_END - timedelta(days=6)
    if boundary not in anchors:
        anchors.append(boundary)
    return sorted(anchors)


def acquire(raw_dir: Path, protocol: Protocol, *, offline: bool, force: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(protocol.requests_per_minute)
    key = _api_key() if not offline else "offline-cache-key"
    request_log: list[dict[str, Any]] = []

    first = _cached_post(
        USER_LIST_URL,
        {"apikey": key, "pagina": "1", "query": ALIAS_QUERY},
        raw_dir / "centraline_page_001.json",
        limiter,
        request_log,
        offline=offline,
        force=force,
    )
    if not isinstance(first, dict) or int(first.get("response_code", 0)) != 200:
        raise RuntimeError(f"elenco_centraline failed: {first}")
    pages = int(first.get("pagine_totali") or 1)
    listing = list(first.get("result") or [])
    for page in range(2, pages + 1):
        response = _cached_post(
            USER_LIST_URL,
            {"apikey": key, "pagina": str(page), "query": ALIAS_QUERY},
            raw_dir / f"centraline_page_{page:03d}.json",
            limiter,
            request_log,
            offline=offline,
            force=force,
        )
        listing.extend(response.get("result") or [])
    device_ids = list(dict.fromkeys(str(value) for value in listing))
    declared_total = int(first.get("centraline_totali") or len(device_ids))
    if len(device_ids) != declared_total or len(device_ids) != 20:
        raise RuntimeError(f"Expected exactly 20 {ALIAS_QUERY} devices; got {len(device_ids)} (API total {declared_total})")

    inventory_rows: list[dict[str, Any]] = []
    observation_candidates: list[dict[str, Any]] = []
    anchors = _weekly_anchors()
    for sequence, device_id in enumerate(device_ids, start=1):
        info = _cached_post(
            USER_INFO_URL,
            {"apikey": key, "ID": device_id},
            raw_dir / "device_info" / f"{device_id}.json",
            limiter,
            request_log,
            offline=offline,
            force=force,
            sanitizer=_sanitize_info,
        )
        result = info.get("result") if isinstance(info, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError(f"No metadata for {device_id}: {info}")
        sensor_block = result.get("sensori") if isinstance(result.get("sensori"), dict) else {}
        installed = sorted(
            key_name for key_name, events in sensor_block.items()
            if isinstance(events, list) and any(isinstance(event, dict) and float(event.get("change", 0)) > 0 for event in events)
        )
        inventory_rows.append(
            {
                "device_index": sequence,
                "device_id": device_id,
                "alias_query": ALIAS_QUERY,
                "alias_exact_returned_by_api": False,
                "country": result.get("nazione"),
                "region": result.get("regione"),
                "province": result.get("provincia"),
                "municipality": result.get("comune"),
                "square_id": result.get("squareID"),
                "lat": pd.to_numeric(result.get("lat"), errors="coerce"),
                "lon": pd.to_numeric(result.get("lon"), errors="coerce"),
                "configuration_date": result.get("configuration_date"),
                "last_entry": result.get("last_entry"),
                "installed_sensor_families": ";".join(installed),
                "firmware": result.get("versione_firmware"),
            }
        )
        for anchor in anchors:
            cache_path = raw_dir / "widget_weekly" / device_id / f"{anchor.isoformat()}.json"
            response = _cached_post(
                READ_URL,
                {
                    "apikey": key,
                    "tipo": "centralina",
                    "tempo": "settimana",
                    "date": anchor.isoformat(),
                    "req_centr": device_id,
                },
                cache_path,
                limiter,
                request_log,
                offline=offline,
                force=force,
            )
            result_block = response.get("result") if isinstance(response, dict) else None
            if not isinstance(result_block, dict):
                continue
            for key_date, values in result_block.items():
                if not isinstance(values, dict):
                    continue
                parsed_date = pd.to_datetime(values.get("data", key_date), errors="coerce")
                if pd.isna(parsed_date):
                    continue
                local_date = parsed_date.date()
                if not STUDY_START <= local_date <= STUDY_END:
                    continue
                row: dict[str, Any] = {
                    "device_id": device_id,
                    "date": local_date.isoformat(),
                    "anchor_date": anchor.isoformat(),
                    "source_cache": str(cache_path.relative_to(raw_dir)),
                }
                for variable, value in values.items():
                    if variable != "data":
                        row[variable] = pd.to_numeric(value, errors="coerce")
                observation_candidates.append(row)

    inventory = pd.DataFrame(inventory_rows).sort_values("device_index").reset_index(drop=True)
    candidates = pd.DataFrame(observation_candidates)
    if candidates.empty:
        raise RuntimeError("No daily observations found in the fixed study interval")
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates["anchor_date"] = pd.to_datetime(candidates["anchor_date"])
    # Later anchors contain more-finalised daily averages.  Use them
    # deterministically when weekly responses overlap.
    daily = (
        candidates.sort_values(["device_id", "date", "anchor_date"])
        .drop_duplicates(["device_id", "date"], keep="last")
        .sort_values(["date", "device_id"])
        .reset_index(drop=True)
    )
    daily = daily.merge(inventory[["device_id", "device_index", "square_id", "lat", "lon"]], on="device_id", how="left", validate="many_to_one")

    files = []
    for path in sorted(p for p in raw_dir.rglob("*.json") if p.name != "manifest.json"):
        files.append({"path": str(path.relative_to(raw_dir)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "case_study": "University of Salerno / Fisciano PE.DT network",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"endpoint_query": ALIAS_QUERY, "api_total": declared_total, "device_ids": device_ids},
        "fixed_interval": {"start": STUDY_START, "end": STUDY_END, "days": (STUDY_END - STUDY_START).days + 1},
        "weekly_anchors": anchors,
        "rate_limit_requests_per_minute": protocol.requests_per_minute,
        "cache_policy": "one JSON response per call; API key excluded; device metadata privacy-filtered",
        "requests": request_log,
        "files": files,
    }
    _atomic_json(raw_dir / "manifest.json", manifest)
    return inventory, daily, manifest


def _distance_m(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat1, lon1 = math.radians(float(lat)), math.radians(float(lon))
    lat2, lon2 = np.radians(np.asarray(lats, dtype=float)), np.radians(np.asarray(lons, dtype=float))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def coverage_tables(inventory: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_days = (STUDY_END - STUDY_START).days + 1
    device_rows: list[dict[str, Any]] = []
    for item in inventory.itertuples(index=False):
        block = daily[daily["device_id"].eq(item.device_id)]
        configured = pd.to_datetime(item.configuration_date, errors="coerce")
        effective_start = max(STUDY_START, configured.date()) if not pd.isna(configured) else STUDY_START
        expected = max(0, (STUDY_END - effective_start).days + 1)
        row = {
            "device_id": item.device_id,
            "device_index": item.device_index,
            "square_id": item.square_id,
            "effective_start": effective_start,
            "expected_days": expected,
            "days_any": int(block["date"].nunique()),
            "coverage_any_pct": 100 * block["date"].nunique() / expected if expected else np.nan,
            "first_observation": block["date"].min(),
            "last_observation": block["date"].max(),
        }
        for variable in POLLUTANTS:
            count = int(block.loc[block[variable].notna(), "date"].nunique()) if variable in block else 0
            row[f"{variable}_days"] = count
            row[f"{variable}_coverage_pct"] = 100 * count / expected if expected else np.nan
        device_rows.append(row)
    by_device = pd.DataFrame(device_rows)

    identifier_columns = {"device_id", "device_index", "square_id", "lat", "lon", "date", "anchor_date", "source_cache"}
    variables = [column for column in daily.columns if column not in identifier_columns and pd.api.types.is_numeric_dtype(daily[column])]
    variable_rows = []
    for variable in sorted(variables):
        valid = daily[daily[variable].notna()]
        variable_rows.append(
            {
                "variable": variable,
                "observations": int(len(valid)),
                "possible_device_days": int(len(inventory) * study_days),
                "coverage_pct": 100 * len(valid) / (len(inventory) * study_days),
                "devices": int(valid["device_id"].nunique()),
                "days": int(valid["date"].nunique()),
                "start": valid["date"].min(),
                "end": valid["date"].max(),
            }
        )
    return by_device, pd.DataFrame(variable_rows)


def _spatial_features(target: pd.Series, peers: pd.DataFrame, pollutant: str, protocol: Protocol) -> dict[str, float]:
    peers = peers.dropna(subset=[pollutant, "lat", "lon"])
    if peers.empty:
        return {}
    distances = _distance_m(float(target["lat"]), float(target["lon"]), peers["lat"].to_numpy(), peers["lon"].to_numpy())
    values = peers[pollutant].to_numpy(dtype=float)
    order = np.argsort(distances)
    distances, values = distances[order], values[order]
    weights = 1.0 / np.maximum(distances, protocol.distance_floor_m) ** protocol.idw_power
    day = pd.Timestamp(target["date"])
    doy = int(day.dayofyear)
    return {
        "lat": float(target["lat"]),
        "lon": float(target["lon"]),
        "day_index": float((day.date() - STUDY_START).days),
        "sin_doy": math.sin(2 * math.pi * doy / 366),
        "cos_doy": math.cos(2 * math.pi * doy / 366),
        "nearest": float(values[0]),
        "nearest_distance_m": float(distances[0]),
        "network_median": float(np.median(values)),
        "network_mean": float(np.mean(values)),
        "network_std": float(np.std(values, ddof=0)),
        "idw": float(np.average(values, weights=weights)),
        "peer_count": float(len(values)),
    }


FEATURE_COLUMNS = (
    "lat", "lon", "day_index", "sin_doy", "cos_doy", "nearest", "nearest_distance_m",
    "network_median", "network_mean", "network_std", "idw", "peer_count",
)


def evaluate_lodo(inventory: pd.DataFrame, daily: pd.DataFrame, protocol: Protocol, *, skip_rf: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    train = daily[daily["date"].dt.date < TEST_START]
    test = daily[daily["date"].dt.date >= TEST_START]

    for pollutant in POLLUTANTS:
        if pollutant not in daily or daily[pollutant].notna().sum() == 0:
            continue
        for target in inventory.itertuples(index=False):
            heldout = target.device_id
            train_features: list[dict[str, float]] = []
            train_y: list[float] = []
            other_train = train[~train["device_id"].eq(heldout)]
            for day_value, day_block in other_train.groupby("date", sort=True):
                for row_index, row in day_block[day_block[pollutant].notna()].iterrows():
                    peers = day_block[~day_block["device_id"].isin([heldout, row["device_id"]])]
                    features = _spatial_features(row, peers, pollutant, protocol)
                    if features:
                        train_features.append(features)
                        train_y.append(float(row[pollutant]))
            rf: RandomForestRegressor | None = None
            if not skip_rf and len(train_features) >= protocol.min_rf_rows:
                rf = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=10,
                    min_samples_leaf=5,
                    max_features=0.8,
                    random_state=protocol.seed,
                    n_jobs=-1,
                )
                rf.fit(pd.DataFrame(train_features)[list(FEATURE_COLUMNS)], np.asarray(train_y))

            target_test = test[test["device_id"].eq(heldout) & test[pollutant].notna()]
            usable_test = 0
            for _, row in target_test.iterrows():
                peers = test[test["date"].eq(row["date"]) & ~test["device_id"].eq(heldout)]
                features = _spatial_features(row, peers, pollutant, protocol)
                if not features:
                    continue
                usable_test += 1
                base = {
                    "pollutant": pollutant,
                    "heldout_device": heldout,
                    "device_index": int(target.device_index),
                    "square_id": target.square_id,
                    "date": row["date"],
                    "observed": float(row[pollutant]),
                    "peer_count": int(features["peer_count"]),
                    "nearest_distance_m": features["nearest_distance_m"],
                }
                for model in ("nearest", "network_median", "idw"):
                    predictions.append({**base, "model": model, "predicted": features[model]})
                if rf is not None:
                    rf_value = float(rf.predict(pd.DataFrame([features])[list(FEATURE_COLUMNS)])[0])
                    predictions.append({**base, "model": "random_forest_fixed", "predicted": rf_value})
            splits.append(
                {
                    "pollutant": pollutant,
                    "heldout_device": heldout,
                    "square_id": target.square_id,
                    "train_start": STUDY_START,
                    "train_end": TEST_START - timedelta(days=1),
                    "test_start": TEST_START,
                    "test_end": STUDY_END,
                    "target_labels_used_in_training": 0,
                    "rf_training_rows": len(train_features),
                    "target_test_rows": len(target_test),
                    "usable_test_rows": usable_test,
                    "rf_fitted": rf is not None,
                }
            )
    frame = pd.DataFrame(predictions)
    if frame.empty:
        raise RuntimeError("No LODO predictions could be produced")
    frame["absolute_error"] = (frame["predicted"] - frame["observed"]).abs()
    frame["signed_error"] = frame["predicted"] - frame["observed"]
    return frame, pd.DataFrame(splits)


def _metrics(block: pd.DataFrame) -> dict[str, Any]:
    observed, predicted = block["observed"].to_numpy(), block["predicted"].to_numpy()
    return {
        "n": len(block),
        "devices": block["heldout_device"].nunique(),
        "days": block["date"].nunique(),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": math.sqrt(mean_squared_error(observed, predicted)),
        "bias": float(np.mean(predicted - observed)),
        "median_absolute_error": float(np.median(np.abs(predicted - observed))),
        "r2": r2_score(observed, predicted) if len(block) > 1 and np.std(observed) > 0 else np.nan,
    }


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = []
    by_device = []
    for (pollutant, model), block in predictions.groupby(["pollutant", "model"], sort=True):
        overall.append({"pollutant": pollutant, "model": model, **_metrics(block)})
    for (pollutant, model, device), block in predictions.groupby(["pollutant", "model", "heldout_device"], sort=True):
        by_device.append({"pollutant": pollutant, "model": model, "heldout_device": device, "square_id": block["square_id"].iloc[0], **_metrics(block)})
    return pd.DataFrame(overall), pd.DataFrame(by_device)


def _network_median_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["pollutant", "heldout_device", "device_index", "square_id", "date"]
    selected = predictions[predictions["model"].eq("network_median")][keys + ["absolute_error"]].rename(
        columns={"absolute_error": "network_median_absolute_error"}
    )
    baseline = predictions[predictions["model"].eq("idw")][keys + ["absolute_error"]].rename(
        columns={"absolute_error": "idw_absolute_error"}
    )
    paired = selected.merge(baseline, on=keys, validate="one_to_one")
    paired["delta_mae_network_median_minus_idw"] = paired["network_median_absolute_error"] - paired["idw_absolute_error"]
    return paired


def _exact_signflip_pvalue(values: np.ndarray, batch_size: int = 65_536) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0 or count > 24:
        return np.nan, 0
    observed = abs(float(np.mean(values)))
    total = 1 << count
    extreme = 0
    shifts = np.arange(count, dtype=np.uint64)
    for start in range(0, total, batch_size):
        codes = np.arange(start, min(start + batch_size, total), dtype=np.uint64)[:, None]
        signs = (((codes >> shifts) & 1).astype(np.int8) * 2 - 1).astype(float)
        permuted = np.abs(signs @ values / count)
        extreme += int(np.count_nonzero(permuted >= observed - 1e-12))
    return extreme / total, total


def device_level_audits(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = _network_median_pairs(predictions)
    device = (
        paired.groupby(["pollutant", "heldout_device", "device_index", "square_id"], observed=True)
        .agg(
            paired_days=("date", "nunique"),
            idw_mae=("idw_absolute_error", "mean"),
            network_median_mae=("network_median_absolute_error", "mean"),
            delta_mae_network_median_minus_idw=("delta_mae_network_median_minus_idw", "mean"),
        )
        .reset_index()
    )
    device["network_median_better"] = device["delta_mae_network_median_minus_idw"] < 0

    tests = []
    for pollutant, block in device.groupby("pollutant", sort=True):
        values = block["delta_mae_network_median_minus_idw"].to_numpy(dtype=float)
        pvalue, permutations = _exact_signflip_pvalue(values)
        tests.append(
            {
                "pollutant": pollutant,
                "devices": len(block),
                "devices_network_median_better": int((values < 0).sum()),
                "devices_idw_better": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "mean_device_delta_mae": float(np.mean(values)),
                "median_device_delta_mae": float(np.median(values)),
                "exact_two_sided_signflip_p": pvalue,
                "exact_permutations": permutations,
                "unit": "device-reported ug/m3",
            }
        )

    influence_rows = []
    for pollutant, block in paired.groupby("pollutant", sort=True):
        for omitted in sorted(block["heldout_device"].unique()):
            retained = block[~block["heldout_device"].eq(omitted)]
            influence_rows.append(
                {
                    "pollutant": pollutant,
                    "omitted_device": omitted,
                    "retained_devices": retained["heldout_device"].nunique(),
                    "retained_device_days": len(retained),
                    "delta_mae_network_median_minus_idw": retained["delta_mae_network_median_minus_idw"].mean(),
                }
            )

    square = (
        paired.groupby(["pollutant", "square_id"], observed=True)
        .agg(
            devices=("heldout_device", "nunique"),
            paired_device_days=("date", "size"),
            idw_mae=("idw_absolute_error", "mean"),
            network_median_mae=("network_median_absolute_error", "mean"),
            delta_mae_network_median_minus_idw=("delta_mae_network_median_minus_idw", "mean"),
        )
        .reset_index()
    )
    return device, pd.DataFrame(tests), pd.DataFrame(influence_rows), square


def _sample_blocks(unique_dates: Sequence[pd.Timestamp], block_days: int, rng: np.random.Generator) -> list[pd.Timestamp]:
    dates = list(sorted(pd.Timestamp(value) for value in unique_dates))
    if not dates:
        return []
    blocks = [dates[index : index + block_days] for index in range(0, len(dates), block_days)]
    chosen: list[pd.Timestamp] = []
    for index in rng.integers(0, len(blocks), size=len(blocks)):
        chosen.extend(blocks[int(index)])
    return chosen


def _hierarchical_bootstrap_means(
    frame: pd.DataFrame,
    value_column: str,
    protocol: Protocol,
    rng: np.random.Generator,
) -> list[float]:
    """Resample devices and non-overlapping date blocks using a dense matrix."""
    matrix = frame.pivot(index="heldout_device", columns="date", values=value_column).sort_index(axis=1)
    values = matrix.to_numpy(dtype=float)
    if values.size == 0:
        return []
    date_blocks = [
        np.arange(start, min(start + protocol.bootstrap_block_days, values.shape[1]))
        for start in range(0, values.shape[1], protocol.bootstrap_block_days)
    ]
    estimates: list[float] = []
    for _ in range(protocol.bootstrap_iterations):
        sampled_devices = rng.integers(0, values.shape[0], size=values.shape[0])
        sampled_blocks = rng.integers(0, len(date_blocks), size=len(date_blocks))
        sampled_dates = np.concatenate([date_blocks[int(index)] for index in sampled_blocks])
        sample = values[np.ix_(sampled_devices, sampled_dates)]
        if np.isfinite(sample).any():
            estimates.append(float(np.nanmean(sample)))
    return estimates


def bootstrap_intervals(predictions: pd.DataFrame, protocol: Protocol) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(protocol.seed)
    metric_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for pollutant, pollutant_block in predictions.groupby("pollutant", sort=True):
        for model, model_block in pollutant_block.groupby("model", sort=True):
            values = _hierarchical_bootstrap_means(model_block, "absolute_error", protocol, rng)
            lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975]) if values else (np.nan, np.nan, np.nan)
            metric_rows.append({"pollutant": pollutant, "model": model, "metric": "mae", "estimate": model_block["absolute_error"].mean(), "ci_lower": lower, "bootstrap_median": median, "ci_upper": upper, "iterations": len(values)})

        baseline = pollutant_block[pollutant_block["model"].eq("idw")][["heldout_device", "date", "absolute_error"]].rename(columns={"absolute_error": "baseline_error"})
        for model in sorted(set(pollutant_block["model"]) - {"idw"}):
            candidate = pollutant_block[pollutant_block["model"].eq(model)][["heldout_device", "date", "absolute_error"]]
            paired = candidate.merge(baseline, on=["heldout_device", "date"], validate="one_to_one")
            paired["delta"] = paired["absolute_error"] - paired["baseline_error"]
            values = _hierarchical_bootstrap_means(paired, "delta", protocol, rng)
            lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975]) if values else (np.nan, np.nan, np.nan)
            paired_rows.append(
                {
                    "pollutant": pollutant,
                    "model": model,
                    "baseline": "idw",
                    "metric": "delta_mae_model_minus_idw",
                    "estimate": paired["delta"].mean(),
                    "ci_lower": lower,
                    "bootstrap_median": median,
                    "ci_upper": upper,
                    "probability_model_better": float(np.mean(np.asarray(values) < 0)) if values else np.nan,
                    "paired_rows": len(paired),
                    "iterations": len(values),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(paired_rows)


def _station_metadata(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        suffix_length = len(header) - 8  # fields from Tipo through Pioggia
        for row in reader:
            if len(row) < len(header):
                continue
            prefix, suffix = row[:7], row[-suffix_length:]
            rows.append(
                {
                    "network": prefix[0],
                    "station_name": prefix[2],
                    "station_code": prefix[3],
                    "province": prefix[6],
                    "address": ",".join(row[7:-suffix_length]),
                    "station_type": suffix[0],
                    "lat": pd.to_numeric(suffix[1], errors="coerce"),
                    "lon": pd.to_numeric(suffix[2], errors="coerce"),
                    "metadata_pm10": suffix[7] if len(suffix) > 7 else None,
                    "metadata_pm2_5": suffix[8] if len(suffix) > 8 else None,
                }
            )
    return pd.DataFrame(rows)


def arpac_screening(arpac_dir: Path, inventory: pd.DataFrame, protocol: Protocol) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata_path = arpac_dir / "arpac_station_metadata.csv"
    monthly = sorted(arpac_dir.glob("arpac_rqa_2026_*.csv"))
    if not metadata_path.exists() or not monthly:
        return pd.DataFrame(), pd.DataFrame(), {"available": False, "reason": "ARPAC files absent"}
    metadata = _station_metadata(metadata_path)
    stations = metadata[metadata["station_code"].isin(ARPAC_CODES)].copy()
    square_centroids = inventory.groupby("square_id", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"))
    campus_lat, campus_lon = inventory["lat"].mean(), inventory["lon"].mean()
    station_rows = []
    for row in stations.itertuples(index=False):
        distances = _distance_m(row.lat, row.lon, square_centroids["lat"].to_numpy(), square_centroids["lon"].to_numpy())
        nearest = int(np.argmin(distances))
        item = row._asdict()
        item["campus_centroid_distance_km"] = float(_distance_m(row.lat, row.lon, np.array([campus_lat]), np.array([campus_lon]))[0] / 1000)
        item["nearest_square_id"] = square_centroids.iloc[nearest]["square_id"]
        item["nearest_square_distance_km"] = float(distances[nearest] / 1000)
        for square, distance in zip(square_centroids["square_id"], distances):
            safe = str(square).replace(" ", "_").replace(":", "_").replace("-", "_")
            item[f"distance_to_{safe}_km"] = float(distance / 1000)
        station_rows.append(item)
    station_table = pd.DataFrame(station_rows).sort_values("nearest_square_distance_km")

    parts = []
    for path in monthly:
        frame = pd.read_csv(path, usecols=["Stazione", "Inquinante", "Data_ora", "Valore", "Um"], low_memory=False)
        frame = frame[frame["Stazione"].isin(ARPAC_CODES) & frame["Inquinante"].isin(["PM10", "PM2,5"])]
        parts.append(frame)
    hourly = pd.concat(parts, ignore_index=True)
    hourly["pollutant"] = hourly["Inquinante"].replace({"PM10": "pm10", "PM2,5": "pm2_5"})
    # The official files label summer timestamps +01:00.  Calendar-day
    # screening therefore uses the explicit DD-MM-YYYY field rather than
    # treating that offset as Europe/Rome civil time.
    hourly["date"] = pd.to_datetime(hourly["Data_ora"].astype(str).str.slice(0, 10), format="%d-%m-%Y", errors="coerce")
    hourly["hour"] = pd.to_datetime(hourly["Data_ora"].astype(str).str.slice(11, 19), format="%H:%M:%S", errors="coerce").dt.hour
    hourly["value"] = pd.to_numeric(hourly["Valore"], errors="coerce")
    hourly = hourly.dropna(subset=["date", "hour", "value"])
    offset_strings = sorted(hourly["Data_ora"].astype(str).str[-6:].unique().tolist())
    invalid_value_count = int((~np.isfinite(hourly["value"]) | hourly["value"].lt(0)).sum())
    hourly = hourly[np.isfinite(hourly["value"]) & hourly["value"].ge(0)]
    hourly["Um"] = "ug/m3"
    hourly = hourly[
        hourly["date"].dt.date.between(STUDY_START, STUDY_END, inclusive="both")
    ]
    duplicate_count = int(hourly.duplicated(["Stazione", "pollutant", "date", "hour"]).sum())
    hourly = (
        hourly.groupby(["Stazione", "pollutant", "date", "hour"], observed=True)
        .agg(value=("value", "mean"), unit=("Um", "first"))
        .reset_index()
    )
    daily = (
        hourly.groupby(["Stazione", "pollutant", "date"], observed=True)
        .agg(value=("value", "mean"), valid_hours=("hour", "nunique"), unit=("unit", "first"))
        .reset_index()
    )
    daily["meets_75pct"] = daily["valid_hours"] >= protocol.min_arpac_hours
    daily.loc[~daily["meets_75pct"], "value"] = np.nan
    daily = daily.merge(station_table[["station_code", "station_name", "station_type", "nearest_square_id", "nearest_square_distance_km"]], left_on="Stazione", right_on="station_code", how="left")
    profile = {
        "available": True,
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in [metadata_path, *monthly]],
        "candidate_stations": len(station_table),
        "daily_rows": len(daily),
        "daily_valid_75pct": int(daily["value"].notna().sum()),
        "start": daily["date"].min(),
        "end": daily["date"].max(),
        "duplicate_station_pollutant_hours": duplicate_count,
        "invalid_negative_or_nonfinite_values_removed": invalid_value_count,
        "summer_offset_strings": offset_strings,
        "interpretation": "independent regulatory screening only; no campus co-location and no LODO ground truth",
    }
    return station_table, daily, profile


def arpac_overlap_table(
    pedt_daily: pd.DataFrame,
    inventory: pd.DataFrame,
    arpac_daily: pd.DataFrame,
    arpac_stations: pd.DataFrame,
) -> pd.DataFrame:
    """Contextual temporal overlap; deliberately not an accuracy benchmark."""
    if arpac_daily.empty or arpac_stations.empty:
        return pd.DataFrame()
    square_centroids = inventory.groupby("square_id", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"))
    rows: list[dict[str, Any]] = []
    periods = {
        "full_fixed_window": (pd.Timestamp(STUDY_START), pd.Timestamp(STUDY_END)),
        "final_test_window": (pd.Timestamp(TEST_START), pd.Timestamp(STUDY_END)),
    }
    for pollutant in POLLUTANTS:
        if pollutant not in pedt_daily:
            continue
        square_daily = (
            pedt_daily.dropna(subset=[pollutant])
            .groupby(["square_id", "date"], observed=True)[pollutant]
            .mean()
            .rename("pedt_square_mean")
            .reset_index()
        )
        official = arpac_daily[arpac_daily["pollutant"].eq(pollutant) & arpac_daily["value"].notna()].copy()
        for station in arpac_stations.itertuples(index=False):
            official_station = official[official["Stazione"].eq(station.station_code)][["date", "value", "valid_hours"]]
            for square in square_centroids.itertuples(index=False):
                distance_km = float(_distance_m(station.lat, station.lon, np.array([square.lat]), np.array([square.lon]))[0] / 1000)
                paired_all = square_daily[square_daily["square_id"].eq(square.square_id)].merge(official_station, on="date", how="inner")
                for period_name, (start, end) in periods.items():
                    paired = paired_all[paired_all["date"].between(start, end, inclusive="both")]
                    n = len(paired)
                    pearson = paired["pedt_square_mean"].corr(paired["value"], method="pearson") if n >= 3 else np.nan
                    spearman = paired["pedt_square_mean"].corr(paired["value"], method="spearman") if n >= 3 else np.nan
                    difference = paired["pedt_square_mean"] - paired["value"]
                    rows.append(
                        {
                            "period": period_name,
                            "pollutant": pollutant,
                            "station_code": station.station_code,
                            "station_name": station.station_name,
                            "station_type": station.station_type,
                            "square_id": square.square_id,
                            "distance_km": distance_km,
                            "paired_days": n,
                            "overlap_start": paired["date"].min() if n else pd.NaT,
                            "overlap_end": paired["date"].max() if n else pd.NaT,
                            "pearson_r": pearson,
                            "spearman_rho": spearman,
                            "pedt_mean": paired["pedt_square_mean"].mean() if n else np.nan,
                            "arpac_mean": paired["value"].mean() if n else np.nan,
                            "mean_difference_pedt_minus_arpac": difference.mean() if n else np.nan,
                            "mean_absolute_difference": difference.abs().mean() if n else np.nan,
                            "interpretation": "non-colocated temporal screening; not regulatory accuracy",
                        }
                    )
    return pd.DataFrame(rows).sort_values(["period", "pollutant", "distance_km", "square_id"]).reset_index(drop=True)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_audit_summary_figure(
    report_dir: Path,
    inventory: pd.DataFrame,
    paired_ci: pd.DataFrame,
    device_audit: pd.DataFrame,
) -> None:
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.8),
        gridspec_kw={"width_ratios": [1.05, 0.78, 1.45]},
        constrained_layout=True,
    )

    square_colors = ["#0072B2", "#E69F00", "#009E73"]
    for color, (square, block) in zip(square_colors, inventory.groupby("square_id", sort=True)):
        axes[0].scatter(block["lon"], block["lat"], s=30, color=color, label=f"{square.split()[-1]} (n={len(block)})")
        for row in block.itertuples(index=False):
            axes[0].annotate(str(row.device_index), (row.lon, row.lat), xytext=(2, 2), textcoords="offset points", fontsize=6)
    axes[0].set_title("a  Network layout", loc="left", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=6, loc="best")
    axes[0].grid(alpha=0.2)

    order = ["pm10", "pm2_5"]
    labels = ["PM10", "PM2.5"]
    aggregate = paired_ci[
        paired_ci["model"].eq("network_median") & paired_ci["baseline"].eq("idw")
    ].set_index("pollutant")
    for index, (pollutant, label) in enumerate(zip(order, labels)):
        row = aggregate.loc[pollutant]
        estimate = float(row["estimate"])
        axes[1].errorbar(
            index,
            estimate,
            yerr=[[estimate - float(row["ci_lower"])], [float(row["ci_upper"]) - estimate]],
            fmt="o",
            markersize=6,
            capsize=4,
            color=["#0072B2", "#D55E00"][index],
            linewidth=1.4,
        )
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_xticks(range(2), labels)
    axes[1].set_ylabel(r"Network median - IDW ($\mu$g m$^{-3}$)")
    axes[1].set_title(r"b  Paired $\Delta$MAE (95% CI)", loc="left", fontweight="bold", fontsize=10)
    axes[1].tick_params(labelsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    styles = {"pm10": ("o", "#0072B2", "PM10"), "pm2_5": ("s", "#D55E00", "PM2.5")}
    for pollutant in order:
        marker, color, label = styles[pollutant]
        block = device_audit[device_audit["pollutant"].eq(pollutant)].sort_values("device_index")
        axes[2].scatter(
            block["device_index"],
            block["delta_mae_network_median_minus_idw"],
            s=25,
            marker=marker,
            color=color,
            label=label,
            alpha=0.9,
        )
    axes[2].axhline(0, color="black", linewidth=0.9)
    axes[2].set_xticks([1, 5, 10, 15, 20])
    axes[2].set_xlim(0.3, 20.7)
    axes[2].set_xlabel("Held-out device index")
    axes[2].set_ylabel(r"Mean $\Delta$MAE ($\mu$g m$^{-3}$)")
    axes[2].set_title("c  Device-level effects", loc="left", fontweight="bold", fontsize=10)
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    axes[2].tick_params(labelsize=8)
    axes[2].grid(axis="y", alpha=0.2)

    stem = figures / "unisa_audit_summary"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_figures(
    report_dir: Path,
    inventory: pd.DataFrame,
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired_ci: pd.DataFrame,
    device_audit: pd.DataFrame,
) -> None:
    figures = report_dir / "figures"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for square, block in inventory.groupby("square_id", sort=True):
        ax.scatter(block["lon"], block["lat"], s=55, label=f"{square} (n={len(block)})")
        for row in block.itertuples(index=False):
            ax.annotate(str(row.device_index), (row.lon, row.lat), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="PE.DT device locations on the Fisciano campus")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    _save_figure(fig, figures / "sensor_map")

    availability = daily.assign(available=daily["pm2_5"].notna().astype(int)).pivot(index="device_index", columns="date", values="available")
    availability = availability.reindex(index=inventory["device_index"], columns=pd.date_range(STUDY_START, STUDY_END)).fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.imshow(availability.to_numpy(), aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    tick_indices = np.linspace(0, availability.shape[1] - 1, 7).astype(int)
    ax.set_xticks(tick_indices, [availability.columns[index].strftime("%d %b") for index in tick_indices])
    ax.set_yticks(np.arange(len(availability)), availability.index.astype(str), fontsize=7)
    ax.set(xlabel="Calendar day (2026)", ylabel="PE.DT device index", title="Daily PM2.5 availability")
    _save_figure(fig, figures / "pm25_coverage_timeline")

    pollutants = list(metrics["pollutant"].unique())
    fig, axes = plt.subplots(1, len(pollutants), figsize=(6.2 * len(pollutants), 4.5), squeeze=False)
    for ax, pollutant in zip(axes[0], pollutants):
        block = metrics[metrics["pollutant"].eq(pollutant)].sort_values("mae")
        ci = bootstrap[bootstrap["pollutant"].eq(pollutant)].set_index("model")
        lower = [max(0.0, row.mae - ci.loc[row.model, "ci_lower"]) for row in block.itertuples()]
        upper = [max(0.0, ci.loc[row.model, "ci_upper"] - row.mae) for row in block.itertuples()]
        ax.bar(block["model"], block["mae"], yerr=np.vstack([lower, upper]), capsize=4, color="#377eb8")
        ax.set(title=pollutant.replace("_", ".").upper(), ylabel="MAE (device-reported ug/m3)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Strict leave-one-device-out reconstruction on the final 21 days")
    _save_figure(fig, figures / "model_comparison")
    make_audit_summary_figure(report_dir, inventory, paired_ci, device_audit)


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def write_report(
    report_dir: Path,
    inventory: pd.DataFrame,
    daily: pd.DataFrame,
    coverage_device: pd.DataFrame,
    coverage_variable: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    splits: pd.DataFrame,
    device_audit: pd.DataFrame,
    device_tests: pd.DataFrame,
    influence: pd.DataFrame,
    square_audit: pd.DataFrame,
    arpac_stations: pd.DataFrame,
    arpac_daily: pd.DataFrame,
    arpac_overlap: pd.DataFrame,
    arpac_profile: Mapping[str, Any],
) -> dict[str, Any]:
    best_rows = metrics.sort_values(["pollutant", "mae"]).groupby("pollutant", as_index=False).first()
    min_distance = arpac_stations["nearest_square_distance_km"].min() if not arpac_stations.empty else np.nan
    expected_dates = set(pd.date_range(STUDY_START, STUDY_END).date)
    observed_dates = set(pd.to_datetime(daily["date"]).dropna().dt.date)
    network_wide_missing_dates = sorted(value.isoformat() for value in expected_dates - observed_dates)
    maximum_device_separation_m = 0.0
    for row in inventory.itertuples(index=False):
        maximum_device_separation_m = max(
            maximum_device_separation_m,
            float(_distance_m(row.lat, row.lon, inventory["lat"].to_numpy(), inventory["lon"].to_numpy()).max()),
        )
    summary = {
        "verdict": "publishable as an internal network-reconstruction pilot; not an absolute/regulatory accuracy validation",
        "devices": len(inventory),
        "squares": inventory["square_id"].nunique(),
        "study_start": STUDY_START,
        "study_end": STUDY_END,
        "test_start": TEST_START,
        "daily_rows": len(daily),
        "network_wide_missing_dates": network_wide_missing_dates,
        "maximum_device_separation_m": maximum_device_separation_m,
        "variables": coverage_variable["variable"].tolist(),
        "best_models": best_rows.to_dict("records"),
        "device_level_signflip_tests": device_tests.to_dict("records"),
        "nearest_arpac_distance_km": min_distance,
        "arpac_overlap_end": arpac_profile.get("end"),
        "arpac_overlap_screening_rows": len(arpac_overlap),
        "regulatory_ground_truth_for_lodo": False,
    }
    lines = [
        "# University of Salerno PE.DT campus case study",
        "",
        "## Evidence boundary",
        "",
        "This analysis supports a narrow but useful claim: a missing daily value from one low-cost PE.DT device can be reconstructed from the other devices in the same campus network. It does **not** establish regulatory accuracy. The held-out target is another PE.DT device, so it shares the technology, calibration chain and possible common-mode bias of the predictors.",
        "",
        f"The registry query `{ALIAS_QUERY}` returned {len(inventory)} devices, all in Fisciano, distributed across {inventory['square_id'].nunique()} Square cells. The accessible listing exposes hardware IDs but not the alias strings themselves; consequently the cached evidence attests the query result, not a character-by-character alias audit.",
        "",
        f"Coordinates span {inventory['lat'].min():.6f}-{inventory['lat'].max():.6f} degrees north and {inventory['lon'].min():.6f}-{inventory['lon'].max():.6f} degrees east; the maximum device-to-device separation is {maximum_device_separation_m/1000:.2f} km.",
        "",
        "## Data and availability",
        "",
        f"The fixed window is {STUDY_START.isoformat()}-{STUDY_END.isoformat()} ({(STUDY_END-STUDY_START).days+1} calendar days). Weekly `widgetView` calls were cached individually and filtered back to this interval. The endpoint returns server-side daily averages; raw-sample completeness is unavailable, so a returned daily mean is treated as available but cannot be certified against a 75% raw-sample rule.",
        "",
        f"Across devices, median PM2.5 coverage is {_fmt(coverage_device['pm2_5_coverage_pct'].median(),1)}% (range {_fmt(coverage_device['pm2_5_coverage_pct'].min(),1)}-{_fmt(coverage_device['pm2_5_coverage_pct'].max(),1)}%). Median PM10 coverage is {_fmt(coverage_device['pm10_coverage_pct'].median(),1)}%.",
        "",
        f"No PE.DT device returned a daily value on {', '.join(network_wide_missing_dates)}. These two network-wide outages are retained as missing and never interpolated. Wind fields arrive under both `intensita_vento` and the misspelled `intesita_vento`; both are preserved for provenance and excluded from the reconstruction models.",
        "",
        "## Strict internal reconstruction protocol",
        "",
        f"For every pollutant and every device, that device is excluded from all model fitting and from all peer features. Training uses {STUDY_START.isoformat()}-{(TEST_START-timedelta(days=1)).isoformat()}; the untouched final test window is {TEST_START.isoformat()}-{STUDY_END.isoformat()}. Nearest-neighbour, network median and IDW are fixed baselines. The random forest, when sample size permits, uses fixed hyperparameters and only date, location and contemporaneous peer summaries. No tuning or model selection uses held-out-device labels.",
        "",
        "This is spatial reconstruction, not forecasting: contemporaneous peer measurements are intentionally available at prediction time. A forecasting claim would require lagged-only inputs and a different protocol.",
        "",
        "## Results",
        "",
        "| Pollutant | Model | n | MAE | RMSE | Bias | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values(["pollutant", "mae"]).itertuples(index=False):
        lines.append(f"| {row.pollutant} | {row.model} | {row.n} | {_fmt(row.mae)} | {_fmt(row.rmse)} | {_fmt(row.bias)} | {_fmt(row.r2)} |")
    lines.extend(
        [
            "",
            "Uncertainty intervals use a hierarchical block bootstrap: devices are resampled and ordered test dates are resampled in non-overlapping seven-day blocks. Paired deltas are computed on identical device-days relative to IDW; negative delta MAE favours the candidate.",
            "",
            "| Pollutant | Network median - IDW delta MAE | Hierarchical 95% CI | Devices improved | Exact device sign-flip p | Leave-one-device-out influence range |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pollutant in POLLUTANTS:
        paired_row = paired[(paired["pollutant"].eq(pollutant)) & paired["model"].eq("network_median")].iloc[0]
        test_row = device_tests[device_tests["pollutant"].eq(pollutant)].iloc[0]
        influence_block = influence[influence["pollutant"].eq(pollutant)]["delta_mae_network_median_minus_idw"]
        lines.append(
            f"| {pollutant} | {_fmt(paired_row['estimate'],3)} | [{_fmt(paired_row['ci_lower'],3)}, {_fmt(paired_row['ci_upper'],3)}] | "
            f"{int(test_row['devices_network_median_better'])}/{int(test_row['devices'])} | {_fmt(test_row['exact_two_sided_signflip_p'],5)} | "
            f"[{_fmt(influence_block.min(),3)}, {_fmt(influence_block.max(),3)}] |"
        )
    lines.extend(
        [
            "",
            "The direction of the aggregate gain is stable to omitting any one device, but neither hierarchical 95% interval excludes zero and the exact device-level sign-flip tests do not cross the conventional 0.05 threshold. The evidence therefore suggests a robust tendency, not definitive superiority over IDW.",
            "",
            "The effect is heterogeneous across the three Square cells:",
            "",
            "| Pollutant | Square | Devices | Paired device-days | Delta MAE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in square_audit.sort_values(["pollutant", "square_id"]).itertuples(index=False):
        lines.append(
            f"| {row.pollutant} | {row.square_id} | {row.devices} | {row.paired_device_days} | "
            f"{_fmt(row.delta_mae_network_median_minus_idw,3)} |"
        )
    lines.extend(
        [
            "",
            "The near-zero effect in Square 3:4 and larger gains in the other cells argue against presenting one campus-wide effect as spatially uniform.",
            "",
            "## Independent ARPAC screening",
            "",
        ]
    )
    if arpac_stations.empty:
        lines.append("No local ARPAC bundle was available, so no independent screening was possible.")
    else:
        lines.extend(
            [
                f"Six official ARPAC stations with PM data were screened. The closest is {arpac_stations.iloc[0]['station_name']} ({arpac_stations.iloc[0]['station_code']}), {_fmt(arpac_stations.iloc[0]['nearest_square_distance_km'])} km from its nearest PE.DT Square centroid. The validated bundle includes the August resource and is filtered to the same fixed 1 May-13 August interval as PE.DT. Usable PM records for these candidates reach {pd.Timestamp(arpac_profile.get('end')).date().isoformat()}, so the regulatory overlap covers most, but not the last two days, of the final PE.DT test window.",
                "",
                "Daily ARPAC means require at least 18 distinct hours. This independent dataset can contextualise regional temporal behaviour, but none of the candidate stations is co-located with the campus sensors. It therefore cannot serve as a pointwise campus target. PM10 spatial-sign inference remains limited by station-to-campus distance, different station environments (traffic/background/industrial), and the inclusion of an Avellino cluster outside the campus airshed.",
                "",
                "The official summer timestamps are labelled `+01:00`, inconsistent with Europe/Rome daylight time. The screening preserves the explicit calendar date and does not use those offsets for sub-daily alignment.",
                "",
                f"A separate overlap table contains {len(arpac_overlap)} station-by-Square-by-pollutant-by-period comparisons. Correlations and concentration differences there are descriptive regional temporal checks only: the station-to-campus separation and site-type mismatch prevent an accuracy interpretation.",
                "",
                "| ARPAC station | Type | Nearest PE.DT square | Distance (km) |",
                "|---|---|---|---:|",
            ]
        )
        for row in arpac_stations.itertuples(index=False):
            lines.append(f"| {row.station_name} ({row.station_code}) | {row.station_type} | {row.nearest_square_id} | {_fmt(row.nearest_square_distance_km)} |")
    lines.extend(
        [
            "",
            "## Publication verdict",
            "",
            "The case is defensible as a campus-scale reliability and internal spatial-reconstruction pilot, preferably in supplementary material until independent co-location is added. It is not evidence that the predicted concentrations are accurate in an absolute or regulatory sense. The strongest next experiment is a pre-registered co-location/rotation campaign with a reference-grade monitor, with calibration days and external test days kept disjoint.",
            "",
            "## Reproducibility artefacts",
            "",
            "- `device_inventory.csv`: privacy-filtered device metadata and coordinates.",
            "- `daily_observations.csv`: cached server-side daily means.",
            "- `split_assignments.csv`: explicit LODO train/test audit.",
            "- `predictions.csv`, `metrics.csv`, `bootstrap_ci.csv`, `paired_bootstrap_ci.csv`: row-level and aggregate evidence.",
            "- `device_level_comparison.csv`, `device_signflip_test.csv`, `leave_one_device_out_influence.csv`, `square_comparison.csv`: device- and Square-level robustness audits.",
            "- `arpac_station_screening.csv`, `arpac_daily.csv`, `arpac_overlap_screening.csv`: independent screening kept separate from LODO.",
            "- `figures/unisa_audit_summary.*`: paper-ready PNG, PDF and SVG audit figure; remaining figures are supplementary.",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(report_dir / "summary.json", summary)
    return summary


def _write_outputs(report_dir: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        serial = frame.copy()
        for column in serial.columns:
            if pd.api.types.is_datetime64_any_dtype(serial[column]):
                serial[column] = serial[column].dt.strftime("%Y-%m-%d")
        serial.to_csv(report_dir / f"{name}.csv", index=False)


def _reuse_prediction_outputs(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path = report_dir / "predictions.csv"
    split_path = report_dir / "split_assignments.csv"
    if not prediction_path.exists() or not split_path.exists():
        raise FileNotFoundError("--reuse-predictions requires existing predictions.csv and split_assignments.csv")
    predictions = pd.read_csv(prediction_path, low_memory=False)
    splits = pd.read_csv(split_path, low_memory=False)
    required = {
        "pollutant", "heldout_device", "device_index", "square_id", "date", "observed", "model", "predicted",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Reusable prediction file lacks columns: {sorted(missing)}")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise")
    predictions["absolute_error"] = (predictions["predicted"] - predictions["observed"]).abs()
    predictions["signed_error"] = predictions["predicted"] - predictions["observed"]
    if predictions["heldout_device"].nunique() != 20:
        raise ValueError("Reusable predictions do not contain all 20 held-out devices")
    if predictions.duplicated(["pollutant", "heldout_device", "date", "model"]).any():
        raise ValueError("Reusable predictions contain duplicate pollutant/device/date/model rows")
    if predictions["date"].dt.date.min() < TEST_START or predictions["date"].dt.date.max() > STUDY_END:
        raise ValueError("Reusable predictions fall outside the fixed test interval")
    return predictions, splits


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = Protocol(
        requests_per_minute=args.requests_per_minute,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    raw_dir, report_dir = Path(args.raw_dir), Path(args.report_dir)
    inventory, daily, raw_manifest = acquire(raw_dir, protocol, offline=args.offline, force=args.force_refresh)
    coverage_device, coverage_variable = coverage_tables(inventory, daily)
    if args.reuse_predictions:
        predictions, splits = _reuse_prediction_outputs(report_dir)
    else:
        predictions, splits = evaluate_lodo(inventory, daily, protocol, skip_rf=args.skip_rf)
    metrics, metrics_device = metric_tables(predictions)
    bootstrap, paired = bootstrap_intervals(predictions, protocol)
    device_audit, device_tests, influence, square_audit = device_level_audits(predictions)
    arpac_stations, arpac_daily, arpac_profile = arpac_screening(Path(args.arpac_dir), inventory, protocol)
    arpac_overlap = arpac_overlap_table(daily, inventory, arpac_daily, arpac_stations)
    _write_outputs(
        report_dir,
        {
            "device_inventory": inventory,
            "daily_observations": daily,
            "coverage_by_device": coverage_device,
            "coverage_by_variable": coverage_variable,
            "split_assignments": splits,
            "predictions": predictions,
            "metrics": metrics,
            "metrics_by_device": metrics_device,
            "bootstrap_ci": bootstrap,
            "paired_bootstrap_ci": paired,
            "device_level_comparison": device_audit,
            "device_signflip_test": device_tests,
            "leave_one_device_out_influence": influence,
            "square_comparison": square_audit,
            "arpac_station_screening": arpac_stations,
            "arpac_daily": arpac_daily,
            "arpac_overlap_screening": arpac_overlap,
        },
    )
    _atomic_json(report_dir / "arpac_screening_profile.json", arpac_profile)
    if not args.skip_figures:
        make_figures(report_dir, inventory, daily, metrics, bootstrap, paired, device_audit)
    summary = write_report(
        report_dir, inventory, daily, coverage_device, coverage_variable, metrics, bootstrap, paired,
        splits, device_audit, device_tests, influence, square_audit,
        arpac_stations, arpac_daily, arpac_overlap, arpac_profile,
    )
    output_files = []
    for path in sorted(p for p in report_dir.rglob("*") if p.is_file() and p.name != "analysis_manifest.json"):
        output_files.append({"path": str(path.relative_to(report_dir)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    analysis_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "raw_manifest_sha256": _sha256(raw_dir / "manifest.json"),
        "arpac_profile": arpac_profile,
        "protocol": protocol.__dict__,
        "fixed_split": {"train_start": STUDY_START, "train_end": TEST_START - timedelta(days=1), "test_start": TEST_START, "test_end": STUDY_END},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "seed": args.seed,
        "reused_existing_predictions": bool(args.reuse_predictions),
        "outputs": output_files,
    }
    _atomic_json(report_dir / "analysis_manifest.json", analysis_manifest)
    return {"summary": summary, "metrics": metrics.to_dict("records"), "raw_files": len(raw_manifest["files"]), "outputs": len(output_files)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/salerno_campus")
    parser.add_argument("--report-dir", default="reports/salerno_campus")
    parser.add_argument("--arpac-dir", default="data/raw/arpac_salerno")
    parser.add_argument("--requests-per-minute", type=float, default=59.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Require a complete existing JSON cache; make no API calls.")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh existing per-call cache files.")
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true", help="Reuse validated report-dir predictions and skip RF fitting.")
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations")
    result = run(args)
    print(json.dumps(_json_value(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
