"""Strict leave-one-station-out replication on the ARPAC IT1508 network.

The analysis intentionally uses the *printed local calendar* in the ARPAC
``Data_ora`` field.  The offset printed in the source export is therefore not
used to shift observations between dates.  Models and hyperparameters are
fixed before testing and every fold removes the held-out station from fitting,
validation, and contemporaneous spatial features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "arpac-it1508-seed-42"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/raw/arpac_salerno")
OUTPUT_DIR = Path("reports/arpac_it1508")
METADATA_FILE = "arpac_station_metadata.csv"
START_DATE = pd.Timestamp("2026-05-01")
END_DATE = pd.Timestamp("2026-07-29")
POLLUTANTS = ("PM10", "PM2.5")
IDW_POWERS = (1, 2, 3)
MAIN_PANEL = (
    "IT2210A",  # Solofra Zona Industriale
    "IT2213A",  # Cava dei Tirreni Stadio
    "IT1504A",  # Salerno SA22
    "IT2214A",  # Nocera Inferiore Scuola Solimena
    "IT2227A",  # Avellino Scuola Alighieri
    "IT0936A",  # Avellino AV41
)
BATTIPAGLIA = "IT2268A"
AVELLINO_PAIR = frozenset(("IT2227A", "IT0936A"))


@dataclass(frozen=True)
class Protocol:
    min_train_days: int = 35
    validation_days: int = 7
    test_days: int = 14
    step_days: int = 14
    min_valid_hours: int = 18
    bootstrap_iterations: int = 2_000
    bootstrap_block_days: int = 7
    seed: int = 42


FEATURES = (
    "spatial_idw",
    "spatial_nearest",
    "spatial_nearest_distance_m",
    "spatial_source_count",
    "spatial_source_mean",
    "spatial_source_std",
    "spatial_source_min",
    "spatial_source_max",
    "lat",
    "lon",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True), encoding="utf-8")


def _normalise_pollutant(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    return cleaned.replace({"PM2,5": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10"})


def _read_metadata(path: Path, requested: Iterable[str]) -> pd.DataFrame:
    """Read station metadata while repairing unquoted commas in addresses.

    The official file has 35 header fields.  At least one address contains an
    unquoted comma, so a conventional dataframe parser shifts every later
    column.  The first seven fields are structurally fixed and the 27 fields
    from ``Tipo`` through ``Pioggia`` are right-aligned; only the address field
    can therefore span multiple CSV tokens.
    """

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [item.strip() for item in next(reader)]
        if len(header) != 35:
            raise ValueError(f"Unexpected metadata schema: {len(header)} columns, expected 35")
        rows: list[dict[str, str]] = []
        suffix_length = len(header) - 8
        for line_number, raw in enumerate(reader, start=2):
            if not raw or len(raw) < len(header):
                continue
            prefix = raw[:7]
            suffix = raw[-suffix_length:]
            address_tokens = raw[7 : len(raw) - suffix_length]
            repaired = [*prefix, ",".join(address_tokens), *suffix]
            if len(repaired) != len(header):
                raise ValueError(f"Could not repair metadata line {line_number}")
            rows.append(dict(zip(header, repaired)))

    metadata = pd.DataFrame(rows)
    metadata["station_id"] = metadata["Codice Europeo"].astype(str).str.strip()
    metadata["station_name"] = metadata["Nome Stazione"].astype(str).str.strip()
    metadata["zone"] = metadata["Zona"].astype(str).str.strip()
    metadata["lat"] = pd.to_numeric(metadata["Latitudine"], errors="coerce")
    metadata["lon"] = pd.to_numeric(metadata["Longitudine"], errors="coerce")
    requested_set = {str(value) for value in requested}
    selected = metadata[metadata["station_id"].isin(requested_set)].copy()
    selected = selected.drop_duplicates("station_id", keep="first")
    missing = sorted(requested_set - set(selected["station_id"]))
    if missing:
        raise ValueError(f"Stations missing from metadata: {missing}")
    if selected[["lat", "lon"]].isna().any().any():
        raise ValueError("Selected station coordinates are not numeric")
    if not selected["lat"].between(35.0, 45.0).all() or not selected["lon"].between(5.0, 20.0).all():
        raise ValueError("Selected station coordinates fall outside Italy")
    return selected[["station_id", "station_name", "zone", "lat", "lon", "Provincia", "Tipo"]].reset_index(drop=True)


def _read_daily_observations(
    data_dir: Path,
    metadata: pd.DataFrame,
    protocol: Protocol,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Read hourly exports and aggregate valid observations to printed dates."""

    source_files = sorted(data_dir.glob("arpac_rqa_2026_*.csv"))
    if not source_files:
        raise FileNotFoundError(f"No ARPAC monthly CSV files found in {data_dir}")
    station_ids = set(metadata["station_id"])
    chunks: list[pd.DataFrame] = []
    for path in source_files:
        frame = pd.read_csv(
            path,
            usecols=["Stazione", "Descrizione", "Inquinante", "Data_ora", "Valore"],
            dtype=str,
            encoding="utf-8-sig",
            on_bad_lines="skip",
            low_memory=False,
        )
        frame.columns = frame.columns.str.strip()
        frame["station_id"] = frame["Stazione"].astype(str).str.strip()
        frame = frame[frame["station_id"].isin(station_ids)].copy()
        frame["pollutant"] = _normalise_pollutant(frame["Inquinante"])
        frame = frame[frame["pollutant"].isin(POLLUTANTS)].copy()
        # Keep the wall-clock part exactly as printed.  Do not apply the source
        # offset, which would move values across the requested local date.
        printed = frame["Data_ora"].astype(str).str.extract(
            r"(?P<printed>\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})",
            expand=False,
        )
        frame["local_hour"] = pd.to_datetime(
            printed,
            format="%d-%m-%Y %H:%M:%S",
            errors="coerce",
        ).dt.floor("h")
        numeric_text = frame["Valore"].astype(str).str.strip().str.replace(",", ".", regex=False)
        frame["value"] = pd.to_numeric(numeric_text, errors="coerce")
        frame = frame[
            frame["local_hour"].notna()
            & np.isfinite(frame["value"])
            & frame["value"].ge(0.0)
            & frame["value"].ne(-9999.0)
        ].copy()
        frame["local_date"] = frame["local_hour"].dt.normalize()
        frame = frame[frame["local_date"].between(START_DATE, END_DATE)].copy()
        chunks.append(frame[["station_id", "pollutant", "local_hour", "local_date", "value"]])

    raw = pd.concat(chunks, ignore_index=True)
    # Duplicate records for the same printed hour must not receive extra weight.
    hourly = (
        raw.groupby(["station_id", "pollutant", "local_hour", "local_date"], observed=True)
        .agg(value=("value", "mean"), duplicate_records=("value", "size"))
        .reset_index()
    )
    daily_all = (
        hourly.groupby(["station_id", "pollutant", "local_date"], observed=True)
        .agg(
            target=("value", "mean"),
            valid_hours=("local_hour", "nunique"),
            hourly_records=("value", "size"),
            duplicate_records=("duplicate_records", "sum"),
        )
        .reset_index()
    )
    daily_all["eligible"] = daily_all["valid_hours"].ge(protocol.min_valid_hours)
    quality = (
        daily_all.groupby(["station_id", "pollutant"], observed=True)
        .agg(
            observed_days=("local_date", "nunique"),
            eligible_days=("eligible", "sum"),
            median_valid_hours=("valid_hours", "median"),
            min_valid_hours=("valid_hours", "min"),
            max_valid_hours=("valid_hours", "max"),
            duplicate_hourly_rows=("duplicate_records", lambda values: int(np.maximum(values - 1, 0).sum())),
        )
        .reset_index()
    )
    daily = daily_all[daily_all["eligible"]].copy()
    daily = daily.merge(metadata, on="station_id", how="left", validate="many_to_one")
    local_date = pd.to_datetime(daily["local_date"])
    day_of_year = local_date.dt.dayofyear.astype(float)
    day_of_week = local_date.dt.dayofweek.astype(float)
    daily["day_of_year_sin"] = np.sin(2.0 * np.pi * (day_of_year - 1.0) / 365.25)
    daily["day_of_year_cos"] = np.cos(2.0 * np.pi * (day_of_year - 1.0) / 365.25)
    daily["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    daily["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    daily["record_id"] = (
        daily["pollutant"].str.replace(".", "_", regex=False)
        + "|"
        + daily["station_id"]
        + "|"
        + daily["local_date"].dt.strftime("%Y-%m-%d")
    )
    daily = daily.sort_values(["pollutant", "station_id", "local_date"]).reset_index(drop=True)
    return daily, quality, source_files


def _distance_m(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat1 = math.radians(float(lat))
    lon1 = math.radians(float(lon))
    lat2 = np.radians(np.asarray(lats, dtype=float))
    lon2 = np.radians(np.asarray(lons, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6_371_000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _spatial_features(target: pd.Series, sources: pd.DataFrame) -> dict[str, Any]:
    if sources.empty:
        return {
            "spatial_idw": np.nan,
            "spatial_idw_p2": np.nan,
            "spatial_idw_p3": np.nan,
            "spatial_nearest": np.nan,
            "spatial_nearest_distance_m": np.nan,
            "spatial_source_count": 0.0,
            "spatial_source_mean": np.nan,
            "spatial_source_std": np.nan,
            "spatial_source_min": np.nan,
            "spatial_source_max": np.nan,
            "spatial_source_ids": "",
            "spatial_nearest_source_id": "",
        }
    distances = _distance_m(
        float(target["lat"]),
        float(target["lon"]),
        sources["lat"].to_numpy(dtype=float),
        sources["lon"].to_numpy(dtype=float),
    )
    values = sources["target"].to_numpy(dtype=float)
    order = np.argsort(distances)
    distances = distances[order]
    values = values[order]
    ordered_sources = sources.iloc[order]
    clipped_distances = np.maximum(distances, 100.0)
    idw = {
        power: float(np.average(values, weights=1.0 / clipped_distances**power))
        for power in IDW_POWERS
    }
    return {
        "spatial_idw": idw[1],
        "spatial_idw_p2": idw[2],
        "spatial_idw_p3": idw[3],
        "spatial_nearest": float(values[0]),
        "spatial_nearest_distance_m": float(distances[0]),
        "spatial_source_count": float(len(values)),
        "spatial_source_mean": float(np.mean(values)),
        "spatial_source_std": float(np.std(values, ddof=0)),
        "spatial_source_min": float(np.min(values)),
        "spatial_source_max": float(np.max(values)),
        "spatial_source_ids": "|".join(sorted(ordered_sources["station_id"].astype(str))),
        "spatial_nearest_source_id": str(ordered_sources.iloc[0]["station_id"]),
    }


def _add_spatial_features(rows: pd.DataFrame, source_pool: pd.DataFrame) -> pd.DataFrame:
    groups = {
        (pd.Timestamp(date), str(pollutant)): group
        for (date, pollutant), group in source_pool.groupby(["local_date", "pollutant"], observed=True)
    }
    feature_rows: list[dict[str, Any]] = []
    for _, target in rows.iterrows():
        sources = groups.get((pd.Timestamp(target["local_date"]), str(target["pollutant"])), pd.DataFrame()).copy()
        if not sources.empty:
            sources = sources[sources["station_id"].astype(str).ne(str(target["station_id"]))]
        feature_rows.append(_spatial_features(target, sources))
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)


def _models(protocol: Protocol) -> dict[str, Any]:
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=protocol.seed,
                # Serial prediction accumulation avoids last-bit variation
                # caused by non-deterministic thread completion order.
                n_jobs=1,
            ),
        ),
    }


def _split_origins(calendar: Sequence[pd.Timestamp], protocol: Protocol) -> list[int]:
    last_origin = len(calendar) - protocol.validation_days - protocol.test_days
    return list(range(protocol.min_train_days, last_origin + 1, protocol.step_days))


def _contains_station(values: pd.Series, station_id: str) -> bool:
    return bool(
        values.fillna("")
        .astype(str)
        .str.split("|")
        .map(lambda items: station_id in items)
        .any()
    )


def run_validation(
    daily: pd.DataFrame,
    panel: Sequence[str],
    protocol: Protocol,
    analysis_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed-model expanding-window leave-one-station-out validation."""

    calendar = list(pd.date_range(START_DATE, END_DATE, freq="D"))
    origins = _split_origins(calendar, protocol)
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    daily = daily[daily["station_id"].isin(panel)].copy()

    for pollutant in POLLUTANTS:
        pollutant_data = daily[daily["pollutant"].eq(pollutant)].copy()
        for heldout in panel:
            source_pool = pollutant_data[pollutant_data["station_id"].ne(heldout)].copy()
            featured = _add_spatial_features(pollutant_data, source_pool)
            for origin_position, origin in enumerate(origins):
                train_dates = calendar[:origin]
                validation_dates = calendar[origin : origin + protocol.validation_days]
                test_dates = calendar[
                    origin + protocol.validation_days : origin + protocol.validation_days + protocol.test_days
                ]
                fold_id = f"{analysis_label}|{pollutant}|{heldout}|{origin_position:02d}"
                train = featured[
                    featured["local_date"].isin(train_dates) & featured["station_id"].ne(heldout)
                ].copy()
                validation = featured[
                    featured["local_date"].isin(validation_dates) & featured["station_id"].ne(heldout)
                ].copy()
                test_before_source_filter = featured[
                    featured["local_date"].isin(test_dates) & featured["station_id"].eq(heldout)
                ].copy()
                test = test_before_source_filter[
                    np.isfinite(test_before_source_filter["spatial_idw"])
                    & np.isfinite(test_before_source_filter["spatial_nearest"])
                ].copy()
                fit_base = pd.concat([train, validation], ignore_index=True)

                feature_scope = pd.concat(
                    [
                        train[["spatial_source_ids"]],
                        validation[["spatial_source_ids"]],
                        test[["spatial_source_ids"]],
                    ],
                    ignore_index=True,
                )
                heldout_in_features = _contains_station(feature_scope["spatial_source_ids"], heldout)
                train_test_overlap = bool(set(train["local_date"]) & set(test["local_date"]))
                validation_test_overlap = bool(set(validation["local_date"]) & set(test["local_date"]))
                audit = {
                    "analysis": analysis_label,
                    "fold_id": fold_id,
                    "pollutant": pollutant,
                    "heldout_station": heldout,
                    "train_start": min(train_dates),
                    "train_end": max(train_dates),
                    "validation_start": min(validation_dates),
                    "validation_end": max(validation_dates),
                    "test_start": min(test_dates),
                    "test_end": max(test_dates),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "fit_rows": len(fit_base),
                    "test_target_rows": len(test_before_source_filter),
                    "test_rows": len(test),
                    "test_rows_without_contemporaneous_sources": len(test_before_source_filter) - len(test),
                    "heldout_in_train": bool(train["station_id"].eq(heldout).any()),
                    "heldout_in_validation": bool(validation["station_id"].eq(heldout).any()),
                    "heldout_in_fit": bool(fit_base["station_id"].eq(heldout).any()),
                    "heldout_in_spatial_features": heldout_in_features,
                    "train_test_date_overlap": train_test_overlap,
                    "validation_test_date_overlap": validation_test_overlap,
                    "model_selection_performed": False,
                    "test_labels_used_for_tuning": False,
                }
                audit["passed"] = bool(
                    len(train) > 0
                    and len(validation) > 0
                    and len(test) > 0
                    and not audit["heldout_in_train"]
                    and not audit["heldout_in_validation"]
                    and not audit["heldout_in_fit"]
                    and not audit["heldout_in_spatial_features"]
                    and not train_test_overlap
                    and not validation_test_overlap
                )
                audits.append(audit)
                if not audit["passed"]:
                    continue

                fixed_predictions: dict[str, np.ndarray] = {
                    "climatology": np.full(len(test), float(fit_base["target"].mean())),
                    "nearest": test["spatial_nearest"].to_numpy(dtype=float),
                    "idw": test["spatial_idw"].to_numpy(dtype=float),
                    "idw_p2": test["spatial_idw_p2"].to_numpy(dtype=float),
                    "idw_p3": test["spatial_idw_p3"].to_numpy(dtype=float),
                }
                for model_name, model in _models(protocol).items():
                    model.fit(fit_base[list(FEATURES)], fit_base["target"])
                    fixed_predictions[model_name] = model.predict(test[list(FEATURES)])

                for model_name, values in fixed_predictions.items():
                    part = test[
                        [
                            "record_id",
                            "local_date",
                            "station_id",
                            "station_name",
                            "pollutant",
                            "target",
                            "lat",
                            "lon",
                            "valid_hours",
                            "spatial_source_count",
                            "spatial_source_ids",
                            "spatial_nearest_source_id",
                        ]
                    ].copy()
                    part["analysis"] = analysis_label
                    part["fold_id"] = fold_id
                    part["model"] = model_name
                    part["prediction"] = np.asarray(values, dtype=float)
                    part["error"] = part["prediction"] - part["target"]
                    part["absolute_error"] = part["error"].abs()
                    part["squared_error"] = part["error"] ** 2
                    predictions.append(part)

    if not predictions:
        raise RuntimeError(f"No valid predictions produced for {analysis_label}")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audits)


def _metric_row(frame: pd.DataFrame, scope: str, model: str, pollutant: str = "all") -> dict[str, Any]:
    actual = frame["target"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    return {
        "scope": scope,
        "pollutant": pollutant,
        "model": model,
        "n": len(frame),
        "dates": frame["local_date"].nunique(),
        "stations": frame["station_id"].nunique(),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(frame["absolute_error"].median()),
        "p90_absolute_error": float(frame["absolute_error"].quantile(0.90)),
        "bias": float(np.mean(predicted - actual)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
    }


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, frame in predictions.groupby("model", observed=True):
        rows.append(_metric_row(frame, "overall", str(model)))
        for pollutant, part in frame.groupby("pollutant", observed=True):
            rows.append(_metric_row(part, "pollutant", str(model), str(pollutant)))
    return pd.DataFrame(rows).sort_values(["scope", "pollutant", "mae", "model"]).reset_index(drop=True)


def calculate_per_station(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (station_id, station_name, pollutant, model), frame in predictions.groupby(
        ["station_id", "station_name", "pollutant", "model"], observed=True
    ):
        row = _metric_row(frame, "station_pollutant", str(model), str(pollutant))
        row.update({"station_id": station_id, "station_name": station_name})
        rows.append(row)
    columns = [
        "station_id",
        "station_name",
        "pollutant",
        "model",
        "n",
        "dates",
        "mae",
        "rmse",
        "median_absolute_error",
        "p90_absolute_error",
        "bias",
        "r2",
        "scope",
        "stations",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["pollutant", "station_id", "mae"]).reset_index(drop=True)


def _moving_date_block_pooled_estimates(
    frame: pd.DataFrame,
    value_column: str,
    iterations: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap dates in moving blocks while retaining the pooled estimand.

    Each sampled date contributes all of its record-level values.  If a date is
    selected more than once, its sum and count are repeated accordingly.  The
    replicate statistic is therefore a pooled record mean, matching the point
    estimate even when the number of eligible station records varies by date.
    """

    date_stats = (
        frame[["local_date", value_column]]
        .dropna()
        .groupby("local_date", observed=True)[value_column]
        .agg(value_sum="sum", value_count="count")
        .sort_index()
    )
    date_sums = date_stats["value_sum"].to_numpy(dtype=float)
    date_counts = date_stats["value_count"].to_numpy(dtype=float)
    n = len(date_stats)
    if n == 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    starts = np.arange(n)
    blocks_needed = int(math.ceil(n / block_length))
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled: list[int] = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            sampled.extend((int(start) + offset) % n for offset in range(block_length))
        sampled_indices = np.asarray(sampled[:n], dtype=int)
        estimates[iteration] = float(
            date_sums[sampled_indices].sum() / date_counts[sampled_indices].sum()
        )
    return estimates


def _exact_sign_flip(effects: np.ndarray) -> float:
    effects = np.asarray(effects, dtype=float)
    if len(effects) == 0:
        return np.nan
    observed = abs(float(np.mean(effects)))
    null = np.asarray(
        [
            float(np.mean(effects * np.asarray(signs, dtype=float)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(effects))
        ]
    )
    return float(np.mean(np.abs(null) >= observed - np.finfo(float).eps * 10.0))


def _geo_cluster(station_id: str) -> str:
    return "AVELLINO_MERGED" if station_id in AVELLINO_PAIR else station_id


def paired_comparisons(
    predictions: pd.DataFrame,
    protocol: Protocol,
    baseline: str = "idw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["record_id", "fold_id", "local_date", "station_id", "station_name", "pollutant"]
    base = predictions[predictions["model"].eq(baseline)][[*keys, "absolute_error"]].rename(
        columns={"absolute_error": "baseline_absolute_error"}
    )
    rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    candidate_set = set(predictions["model"]) - {baseline}
    sensitivity_candidates = sorted(model for model in candidate_set if model.startswith("idw_p"))
    candidates = sorted(candidate_set - set(sensitivity_candidates)) + sensitivity_candidates
    for candidate_position, model in enumerate(candidates):
        candidate = predictions[predictions["model"].eq(model)][[*keys, "absolute_error"]]
        paired = candidate.merge(base, on=keys, how="inner", validate="one_to_one")
        paired["difference"] = paired["absolute_error"] - paired["baseline_absolute_error"]
        paired["geo_cluster"] = paired["station_id"].map(_geo_cluster)
        for (station_id, station_name, pollutant), part in paired.groupby(
            ["station_id", "station_name", "pollutant"], observed=True
        ):
            station_rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "station_id": station_id,
                    "station_name": station_name,
                    "pollutant": pollutant,
                    "n": len(part),
                    "mean_mae_difference": float(part["difference"].mean()),
                }
            )
        scopes = [("overall", "all", paired)] + [
            ("pollutant", pollutant, paired[paired["pollutant"].eq(pollutant)])
            for pollutant in POLLUTANTS
        ]
        for scope_position, (scope, pollutant, frame) in enumerate(scopes):
            if frame.empty:
                continue
            daily_effects = frame.groupby("local_date", observed=True)["difference"].mean().sort_index()
            station_effects = frame.groupby("station_id", observed=True)["difference"].mean()
            geo_effects = frame.groupby("geo_cluster", observed=True)["difference"].mean()
            estimates = _moving_date_block_pooled_estimates(
                frame,
                "difference",
                protocol.bootstrap_iterations,
                protocol.bootstrap_block_days,
                protocol.seed + 100 * candidate_position + 17 * scope_position,
            )
            rows.append(
                {
                    "scope": scope,
                    "pollutant": pollutant,
                    "model": model,
                    "baseline": baseline,
                    "n": len(frame),
                    "date_clusters": len(daily_effects),
                    "station_clusters": len(station_effects),
                    "mean_mae_difference": float(frame["difference"].mean()),
                    "equal_date_mean_difference": float(daily_effects.mean()),
                    "bootstrap_estimand": "pooled_record_mean",
                    "difference_ci_low": float(np.quantile(estimates, 0.025)),
                    "difference_ci_high": float(np.quantile(estimates, 0.975)),
                    "station_mean_difference": float(station_effects.mean()),
                    "station_difference_min": float(station_effects.min()),
                    "station_difference_max": float(station_effects.max()),
                    "station_sign_flip_p": _exact_sign_flip(station_effects.to_numpy(dtype=float)),
                    "geo_clusters": len(geo_effects),
                    "geo_cluster_mean_difference": float(geo_effects.mean()),
                    "geo_cluster_sign_flip_p": _exact_sign_flip(geo_effects.to_numpy(dtype=float)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(station_rows)


def _plot_paired_comparison(
    comparisons: pd.DataFrame,
    station_differences: pd.DataFrame,
    metadata: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    models = ("ridge", "rf")
    model_labels = {"ridge": "Ridge", "rf": "Random Forest"}
    colours = {"ridge": "#2F6B9A", "rf": "#D06C3B"}
    scope_order = [("overall", "all", "Overall"), ("pollutant", "PM10", "PM10"), ("pollutant", "PM2.5", "PM2.5")]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    y_labels: list[str] = []
    y_positions: list[float] = []
    position = 0
    for scope, pollutant, scope_label in scope_order:
        for model in models:
            row = comparisons[
                comparisons["scope"].eq(scope)
                & comparisons["pollutant"].eq(pollutant)
                & comparisons["model"].eq(model)
            ]
            if row.empty:
                continue
            item = row.iloc[0]
            y = float(position)
            ax.errorbar(
                float(item["mean_mae_difference"]),
                y,
                xerr=np.asarray(
                    [
                        [float(item["mean_mae_difference"] - item["difference_ci_low"])],
                        [float(item["difference_ci_high"] - item["mean_mae_difference"])],
                    ]
                ),
                fmt="o",
                color=colours[model],
                capsize=3,
                markersize=5,
            )
            y_labels.append(f"{scope_label} — {model_labels[model]}")
            y_positions.append(y)
            position += 1
        position += 0.35
    ax.axvline(0.0, color="0.25", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions, y_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean paired $\Delta$MAE vs IDW ($\mu$g m$^{-3}$)")
    ax.set_title("(a) Date-block bootstrap, 95% CI", loc="left", fontweight="bold")
    ax.grid(axis="x", color="0.9", linewidth=0.8)

    ax = axes[1]
    station_order = list(MAIN_PANEL)
    y = np.arange(len(station_order), dtype=float)
    offsets = {"ridge": -0.16, "rf": 0.16}
    for model in models:
        frame = station_differences[
            station_differences["model"].eq(model)
        ].copy()
        # Panel (b) pools the two pollutants within station with equal record
        # weighting; the main comparison table retains pollutant-specific rows.
        values = frame.groupby("station_id", observed=True).apply(
            lambda part: np.average(part["mean_mae_difference"], weights=part["n"]),
            include_groups=False,
        )
        x = [float(values.get(station, np.nan)) for station in station_order]
        ax.scatter(x, y + offsets[model], label=model_labels[model], color=colours[model], s=30)
    ax.axvline(0.0, color="0.25", linewidth=1.0, linestyle="--")
    names = metadata.set_index("station_id")["station_name"].to_dict()
    ax.set_yticks(y, [f"{station}\n{names.get(station, '')}" for station in station_order])
    ax.invert_yaxis()
    ax.set_xlabel(r"Station-level paired $\Delta$MAE vs IDW ($\mu$g m$^{-3}$)")
    ax.set_title("(b) Held-out station effects", loc="left", fontweight="bold")
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.suptitle("ARPAC IT1508: fixed models against contemporaneous regulatory IDW", fontsize=12)
    fig.tight_layout()
    for extension in ("pdf", "svg", "png"):
        if extension == "pdf":
            kwargs = {"metadata": {"CreationDate": None, "ModDate": None}}
        elif extension == "svg":
            kwargs = {"metadata": {"Date": None}}
        else:
            kwargs = {"dpi": 240}
        fig.savefig(figures / f"paired_comparison.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_value(record) for record in frame.to_dict(orient="records")]


def _overall_metrics(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    return _records(metrics[metrics["scope"].eq("overall")].sort_values("mae"))


def _comparison_records(comparisons: pd.DataFrame) -> list[dict[str, Any]]:
    return _records(
        comparisons[
            comparisons["model"].isin(["ridge", "rf"])
        ].sort_values(["scope", "pollutant", "model"])
    )


def _format_metrics_table(metrics: pd.DataFrame) -> str:
    overall = metrics[metrics["scope"].eq("overall")].sort_values("mae")
    lines = ["| Model | N | MAE | RMSE | Bias | R2 |", "|---|---:|---:|---:|---:|---:|"]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.n} | {row.mae:.3f} | {row.rmse:.3f} | {row.bias:.3f} | {row.r2:.3f} |"
        )
    return "\n".join(lines)


def _format_comparison_table(comparisons: pd.DataFrame) -> str:
    selected = comparisons[
        comparisons["model"].isin(["ridge", "rf"])
    ].sort_values(["scope", "pollutant", "model"])
    lines = [
        "| Scope | Model - IDW delta MAE | 95% date-block CI | Site p | Avellino-merged p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        label = "Overall" if row.scope == "overall" else row.pollutant
        lines.append(
            f"| {label}, {row.model} | {row.mean_mae_difference:.3f} | "
            f"[{row.difference_ci_low:.3f}, {row.difference_ci_high:.3f}] | "
            f"{row.station_sign_flip_p:.4f} | {row.geo_cluster_sign_flip_p:.4f} |"
        )
    return "\n".join(lines)


def _make_report(
    path: Path,
    quality: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    sensitivity_metrics: pd.DataFrame,
    sensitivity_comparisons: pd.DataFrame,
    audits: pd.DataFrame,
    sensitivity_audits: pd.DataFrame,
    avellino_distance_m: float,
    protocol: Protocol,
) -> None:
    primary_models = ["climatology", "nearest", "idw", "ridge", "rf"]
    primary_metrics = metrics[metrics["model"].isin(primary_models)]
    primary_comparisons = comparisons[
        comparisons["model"].isin(["climatology", "nearest", "ridge", "rf"])
    ]
    best = primary_metrics[primary_metrics["scope"].eq("overall")].sort_values("mae").iloc[0]
    sensitivity_primary_metrics = sensitivity_metrics[sensitivity_metrics["model"].isin(primary_models)]
    sensitivity_best = sensitivity_primary_metrics[
        sensitivity_primary_metrics["scope"].eq("overall")
    ].sort_values("mae").iloc[0]
    idw_sensitivity = metrics[
        metrics["scope"].eq("overall") & metrics["model"].isin(["idw", "idw_p2", "idw_p3"])
    ].copy()
    quality_text = quality[quality["station_id"].isin(MAIN_PANEL)].copy()
    quality_lines = ["| Station | Pollutant | Eligible days | Median valid hours |", "|---|---|---:|---:|"]
    for row in quality_text.sort_values(["station_id", "pollutant"]).itertuples(index=False):
        quality_lines.append(
            f"| {row.station_id} | {row.pollutant} | {int(row.eligible_days)} | {row.median_valid_hours:.1f} |"
        )
    text = f"""# ARPAC IT1508 strict spatial replication

## Design

The prespecified main panel contains exactly {len(MAIN_PANEL)} stations: {', '.join(MAIN_PANEL)}. Hourly PM10 and PM2.5 records were assigned to the local calendar date printed by ARPAC, without applying the printed UTC offset. Negative values, `-9999`, non-numeric values, and station-days with fewer than {protocol.min_valid_hours} distinct valid hours were excluded. Duplicate records within a printed station-hour were averaged before daily aggregation.

The calendar was fixed in advance to {START_DATE.date()}--{END_DATE.date()} ({(END_DATE - START_DATE).days + 1} days). Expanding folds used {protocol.min_train_days} initial training days, {protocol.validation_days} validation days, {protocol.test_days} test days, and a {protocol.step_days}-day step. The validation block was retained to reproduce the Milan schedule; because the model set and hyperparameters were fixed a priori, non-held-out validation rows were appended to the fitting set. No validation-based model selection was performed.

Each test target was one regulatory station-day. Climatology, nearest-station, inverse-distance weighting (IDW), fixed Ridge, and fixed random-forest predictions used only the contemporaneous values of the other regulatory stations. Ridge and random forest received spatial summaries, target coordinates, and deterministic calendar terms. The held-out station was absent from training, validation, fitting, and every spatial feature.

## Data availability

{chr(10).join(quality_lines)}

## Main results

Negative paired differences below mean that a model improved on IDW.

{_format_metrics_table(primary_metrics)}

{_format_comparison_table(primary_comparisons)}

The lowest overall MAE was {best.mae:.3f} micrograms per cubic metre for `{best.model}`. This ranking is descriptive; the paired tables retain date dependence through a seven-day moving-block bootstrap and report exact sign-flip inference across held-out stations.

## IDW exponent sensitivity

The primary inverse-distance exponent was fixed at p=1. The same held-out targets were also reconstructed with p=2 and p=3, without changing source eligibility or test rows.

{_format_metrics_table(idw_sensitivity)}

## Dependence and sensitivity

The two Avellino stations are {avellino_distance_m / 1_000.0:.3f} km apart. The `Avellino-merged p` column treats their mean effect as one geographical cluster before the exact sign-flip test. With five independent clusters, a two-sided exact p-value cannot be smaller than 0.0625.

Battipaglia Parco Fiume (`{BATTIPAGLIA}`) had sufficient PM10 and PM2.5 coverage, so a prespecified sensitivity analysis reran the complete protocol with that station added to the source/target panel. Its best overall model was `{sensitivity_best.model}` (MAE {sensitivity_best.mae:.3f}). Full sensitivity results are in `sensitivity_metrics.csv` and `sensitivity_paired_comparisons.csv`.

{_format_metrics_table(sensitivity_metrics)}

## Leakage audit

Main folds passed: {int(audits['passed'].sum())}/{len(audits)}. Battipaglia-sensitivity folds passed: {int(sensitivity_audits['passed'].sum())}/{len(sensitivity_audits)}. The audit checks held-out membership in training, validation, final fitting, and feature-source IDs, plus all train/validation versus test date overlaps. Test labels were never used for tuning; tuning and fold-wise model selection were not performed.

## Interpretation boundary

This is a regulatory-network interpolation experiment in zone IT1508, not a calibration or accuracy test of the PE.DT campus devices. It tests how well a withheld ARPAC station can be reconstructed from simultaneous neighbouring ARPAC observations over one warm-season window. The short period, spatial heterogeneity, and proximity of the two Avellino sites limit generalisation. The dependence-aware and Battipaglia analyses are therefore part of the result, not optional robustness decoration.
"""
    path.write_text(text, encoding="utf-8")


def run(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR, protocol: Protocol = Protocol()) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    requested = (*MAIN_PANEL, BATTIPAGLIA)
    metadata_path = data_dir / METADATA_FILE
    metadata = _read_metadata(metadata_path, requested)
    if not set(metadata.loc[metadata["station_id"].isin(MAIN_PANEL), "zone"]).issubset({"IT1508"}):
        raise ValueError("The prespecified main panel is not entirely in zone IT1508")
    daily, quality, source_files = _read_daily_observations(data_dir, metadata, protocol)

    main_daily = daily[daily["station_id"].isin(MAIN_PANEL)].copy()
    main_predictions, main_audits = run_validation(main_daily, MAIN_PANEL, protocol, "main")
    main_metrics = calculate_metrics(main_predictions)
    main_per_station = calculate_per_station(main_predictions)
    main_comparisons, main_station_differences = paired_comparisons(main_predictions, protocol)

    sensitivity_panel = (*MAIN_PANEL, BATTIPAGLIA)
    sensitivity_daily = daily[daily["station_id"].isin(sensitivity_panel)].copy()
    battipaglia_coverage = quality[
        quality["station_id"].eq(BATTIPAGLIA) & quality["pollutant"].isin(POLLUTANTS)
    ]
    sensitivity_feasible = bool(
        len(battipaglia_coverage) == len(POLLUTANTS)
        and battipaglia_coverage["eligible_days"].ge(protocol.min_train_days + protocol.validation_days + protocol.test_days).all()
    )
    if not sensitivity_feasible:
        raise RuntimeError("Battipaglia sensitivity was requested but lacks sufficient eligible daily coverage")
    sensitivity_predictions, sensitivity_audits = run_validation(
        sensitivity_daily,
        sensitivity_panel,
        protocol,
        "add_battipaglia",
    )
    sensitivity_metrics = calculate_metrics(sensitivity_predictions)
    sensitivity_per_station = calculate_per_station(sensitivity_predictions)
    sensitivity_comparisons, sensitivity_station_differences = paired_comparisons(
        sensitivity_predictions,
        protocol,
    )

    metadata_index = metadata.set_index("station_id")
    av_a = metadata_index.loc["IT2227A"]
    av_b = metadata_index.loc["IT0936A"]
    avellino_distance_m = float(
        _distance_m(float(av_a["lat"]), float(av_a["lon"]), np.asarray([av_b["lat"]]), np.asarray([av_b["lon"]]))[0]
    )

    main_predictions.to_csv(output_dir / "predictions.csv", index=False)
    main_metrics.to_csv(output_dir / "metrics.csv", index=False)
    main_per_station.to_csv(output_dir / "per_station.csv", index=False)
    main_audits.to_csv(output_dir / "leakage_audit.csv", index=False)
    main_comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    main_station_differences.to_csv(output_dir / "paired_station_differences.csv", index=False)
    quality.to_csv(output_dir / "data_quality.csv", index=False)
    metadata.to_csv(output_dir / "station_inventory.csv", index=False)

    sensitivity_predictions.to_csv(output_dir / "sensitivity_predictions.csv", index=False)
    sensitivity_metrics.to_csv(output_dir / "sensitivity_metrics.csv", index=False)
    sensitivity_per_station.to_csv(output_dir / "sensitivity_per_station.csv", index=False)
    sensitivity_audits.to_csv(output_dir / "sensitivity_leakage_audit.csv", index=False)
    sensitivity_comparisons.to_csv(output_dir / "sensitivity_paired_comparisons.csv", index=False)
    sensitivity_station_differences.to_csv(output_dir / "sensitivity_paired_station_differences.csv", index=False)

    _plot_paired_comparison(main_comparisons, main_station_differences, metadata, figures)
    _make_report(
        output_dir / "report.md",
        quality,
        main_metrics,
        main_comparisons,
        sensitivity_metrics,
        sensitivity_comparisons,
        main_audits,
        sensitivity_audits,
        avellino_distance_m,
        protocol,
    )

    input_files = [metadata_path, *source_files]
    script_path = Path(__file__).resolve()
    output_files = [
        output_dir / "predictions.csv",
        output_dir / "metrics.csv",
        output_dir / "per_station.csv",
        output_dir / "leakage_audit.csv",
        output_dir / "paired_comparisons.csv",
        output_dir / "paired_station_differences.csv",
        output_dir / "data_quality.csv",
        output_dir / "station_inventory.csv",
        output_dir / "sensitivity_predictions.csv",
        output_dir / "sensitivity_metrics.csv",
        output_dir / "sensitivity_per_station.csv",
        output_dir / "sensitivity_leakage_audit.csv",
        output_dir / "sensitivity_paired_comparisons.csv",
        output_dir / "sensitivity_paired_station_differences.csv",
        output_dir / "report.md",
        figures / "paired_comparison.pdf",
        figures / "paired_comparison.svg",
        figures / "paired_comparison.png",
    ]
    missing_outputs = [str(path) for path in output_files if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(f"Final outputs missing before provenance capture: {missing_outputs}")
    summary = {
        "analysis": "ARPAC IT1508 strict leave-one-station-out replication",
        "date_assignment": "local calendar date printed in Data_ora; no offset conversion",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "calendar_days": int((END_DATE - START_DATE).days + 1),
        "main_panel": list(MAIN_PANEL),
        "pollutants": list(POLLUTANTS),
        "protocol": asdict(protocol),
        "models": {
            "baselines": ["climatology", "nearest", "idw"],
            "idw_sensitivity": {
                "primary_power": 1,
                "alternative_powers": [2, 3],
                "minimum_distance_m": 100.0,
            },
            "ridge": {"alpha": 10.0, "preprocessing": "training-median imputation and standardisation"},
            "rf": {
                "n_estimators": 500,
                "min_samples_leaf": 5,
                "max_features": 0.8,
                "random_state": protocol.seed,
            },
            "features": list(FEATURES),
            "selection": "none; all models and hyperparameters fixed before test",
        },
        "daily_rows": int(len(main_daily)),
        "main_test_records": int(main_predictions["record_id"].nunique()),
        "main_folds_expected": int(len(POLLUTANTS) * len(MAIN_PANEL) * len(_split_origins(list(pd.date_range(START_DATE, END_DATE)), protocol))),
        "main_folds_audited": int(len(main_audits)),
        "main_folds_passed": int(main_audits["passed"].sum()),
        "main_metrics_overall": _overall_metrics(main_metrics),
        "main_paired_comparisons": _comparison_records(main_comparisons),
        "dependence_sensitivity": {
            "avellino_station_ids": sorted(AVELLINO_PAIR),
            "distance_m": avellino_distance_m,
            "method": "average station effects into one geographical cluster before exact sign flips",
        },
        "battipaglia_sensitivity": {
            "feasible": sensitivity_feasible,
            "added_station": BATTIPAGLIA,
            "panel_size": len(sensitivity_panel),
            "test_records": int(sensitivity_predictions["record_id"].nunique()),
            "folds_audited": int(len(sensitivity_audits)),
            "folds_passed": int(sensitivity_audits["passed"].sum()),
            "metrics_overall": _overall_metrics(sensitivity_metrics),
            "paired_comparisons": _comparison_records(sensitivity_comparisons),
        },
        "leakage": {
            "heldout_labels_in_train_validation_or_fit": bool(
                main_audits[["heldout_in_train", "heldout_in_validation", "heldout_in_fit"]].any().any()
            ),
            "heldout_station_in_spatial_features": bool(main_audits["heldout_in_spatial_features"].any()),
            "test_date_overlap": bool(
                main_audits[["train_test_date_overlap", "validation_test_date_overlap"]].any().any()
            ),
            "test_labels_used_for_tuning": bool(main_audits["test_labels_used_for_tuning"].any()),
        },
        "input_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "used_rows_restricted_to_prespecified_window": True,
            }
            for path in input_files
        ],
        "script": {
            "path": str(script_path),
            "bytes": script_path.stat().st_size,
            "sha256": _sha256(script_path),
        },
        "output_files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in output_files
        ],
        "summary_hash_policy": (
            "summary.json is intentionally excluded from output_files to avoid a recursive self-hash; "
            "it is regenerated from the hashed script, inputs, and final non-summary outputs"
        ),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation_boundary": "Regulatory-network interpolation; not PE.DT device accuracy or calibration.",
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=Protocol.bootstrap_iterations)
    args = parser.parse_args(argv)
    protocol = Protocol(bootstrap_iterations=args.bootstrap_iterations)
    summary = run(args.data_dir, args.output_dir, protocol)
    best = summary["main_metrics_overall"][0]
    print(
        f"Completed {summary['main_folds_passed']}/{summary['main_folds_expected']} main folds; "
        f"best overall model={best['model']}, MAE={best['mae']:.3f}. "
        f"Outputs: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
