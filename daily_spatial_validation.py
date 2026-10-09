"""Availability-aware daily spatial validation for Comune di Milano.

The analysis harmonises daily ARPA targets, hourly CAMS/weather covariates, and
hourly low-cost observations on Europe/Rome calendar days.  Each experiment
holds one ARPA station out from model fitting, validation-based model selection,
and fold-local spatial features.  Test windows are chronological and disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "daily-spatial-validation"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OBSERVATIONS = Path("data/processed/good_dataset/milano_city_good_dataset.csv")
CAMS_HOURLY = Path("data/processed/cams_milano_city_hourly.csv")
WEATHER_HOURLY = Path("data/processed/openmeteo_milano_city_hourly.csv")
OUTPUT_DIR = Path("reports/daily_spatial_validation")
TIMEZONE = "Europe/Rome"
POLLUTANTS = ("pm10", "pm2_5")
IDW_POWERS = (1, 2, 3)


@dataclass(frozen=True)
class Protocol:
    min_train_days: int = 35
    validation_days: int = 7
    test_days: int = 14
    step_days: int = 14
    max_grid_distance_m: float = 5_000.0
    min_low_cost_hours: int = 18
    bootstrap_iterations: int = 2_000
    bootstrap_block_days: int = 7
    seed: int = 42


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
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True), encoding="utf-8")


def _distance_m(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat1 = math.radians(float(lat))
    lon1 = math.radians(float(lon))
    lat2 = np.radians(np.asarray(lats, dtype=float))
    lon2 = np.radians(np.asarray(lons, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6_371_000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _nearest_grid(points: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid[["grid_id", "lat", "lon"]].drop_duplicates("grid_id").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for point in points[["source_id", "lat", "lon"]].drop_duplicates("source_id").itertuples(index=False):
        distances = _distance_m(point.lat, point.lon, grid["lat"].to_numpy(), grid["lon"].to_numpy())
        nearest = int(np.argmin(distances))
        rows.append(
            {
                "source_id": point.source_id,
                "model_grid_id": str(grid.iloc[nearest]["grid_id"]),
                "model_grid_distance_m": float(distances[nearest]),
            }
        )
    return pd.DataFrame(rows)


def _local_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()


def _spatial_summary(target: pd.Series, sources: pd.DataFrame, prefix: str) -> dict[str, float]:
    if sources.empty:
        return {
            f"{prefix}_idw": np.nan,
            f"{prefix}_idw_p2": np.nan,
            f"{prefix}_idw_p3": np.nan,
            f"{prefix}_nearest": np.nan,
            f"{prefix}_nearest_distance_m": np.nan,
            f"{prefix}_source_count": 0.0,
            f"{prefix}_source_mean": np.nan,
            f"{prefix}_source_std": np.nan,
            f"{prefix}_source_min": np.nan,
            f"{prefix}_source_max": np.nan,
        }
    distances = _distance_m(
        float(target["lat"]),
        float(target["lon"]),
        sources["lat"].to_numpy(dtype=float),
        sources["lon"].to_numpy(dtype=float),
    )
    values = sources["spatial_value"].to_numpy(dtype=float)
    order = np.argsort(distances)
    distances = distances[order]
    values = values[order]
    nearest_count = min(8, len(values))
    clipped_distances = np.maximum(distances[:nearest_count], 100.0)
    idw = {
        power: float(np.average(values[:nearest_count], weights=1.0 / clipped_distances**power))
        for power in IDW_POWERS
    }
    return {
        f"{prefix}_idw": idw[1],
        f"{prefix}_idw_p2": idw[2],
        f"{prefix}_idw_p3": idw[3],
        f"{prefix}_nearest": float(values[0]),
        f"{prefix}_nearest_distance_m": float(distances[0]),
        f"{prefix}_source_count": float(len(values)),
        f"{prefix}_source_mean": float(np.mean(values)),
        f"{prefix}_source_std": float(np.std(values, ddof=0)),
        f"{prefix}_source_min": float(np.min(values)),
        f"{prefix}_source_max": float(np.max(values)),
    }


def _read_hourly_covariates(
    cams_path: Path,
    weather_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    cams = pd.read_csv(
        cams_path,
        usecols=["timestamp_utc", "grid_id", "lat", "lon", "model_lat", "model_lon", "pm10", "pm2_5"],
        low_memory=False,
    )
    cams["local_date"] = _local_date(cams["timestamp_utc"])
    grid = cams[["grid_id", "lat", "lon", "model_lat", "model_lon"]].drop_duplicates("grid_id")
    cams_native_supports = int(grid[["model_lat", "model_lon"]].drop_duplicates().shape[0])
    cams_daily = (
        cams.groupby(["local_date", "grid_id"], observed=True)
        .agg(cams_pm10=("pm10", "mean"), cams_pm2_5=("pm2_5", "mean"), cams_hours=("timestamp_utc", "nunique"))
        .reset_index()
    )
    del cams

    weather_columns = [
        "timestamp_utc",
        "grid_id",
        "model_lat",
        "model_lon",
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "surface_pressure_hpa",
        "pressure_msl_hpa",
        "wind_speed_m_s",
        "wind_direction",
        "boundary_layer_height_m",
    ]
    weather = pd.read_csv(weather_path, usecols=weather_columns, low_memory=False)
    weather_native_supports = int(weather[["model_lat", "model_lon"]].drop_duplicates().shape[0])
    weather["local_date"] = _local_date(weather["timestamp_utc"])
    radians = np.radians(pd.to_numeric(weather["wind_direction"], errors="coerce"))
    weather["wind_direction_sin"] = np.sin(radians)
    weather["wind_direction_cos"] = np.cos(radians)
    weather_daily = (
        weather.groupby(["local_date", "grid_id"], observed=True)
        .agg(
            temperature_c=("temperature_c", "mean"),
            relative_humidity_pct=("relative_humidity_pct", "mean"),
            precipitation_mm=("precipitation_mm", "sum"),
            surface_pressure_hpa=("surface_pressure_hpa", "mean"),
            pressure_msl_hpa=("pressure_msl_hpa", "mean"),
            wind_speed_m_s=("wind_speed_m_s", "mean"),
            wind_direction_sin=("wind_direction_sin", "mean"),
            wind_direction_cos=("wind_direction_cos", "mean"),
            boundary_layer_height_m=("boundary_layer_height_m", "mean"),
            weather_hours=("timestamp_utc", "nunique"),
        )
        .reset_index()
    )
    support = {
        "application_grid_cells": int(grid["grid_id"].nunique()),
        "cams_native_supports": cams_native_supports,
        "weather_native_supports": weather_native_supports,
    }
    return cams_daily, weather_daily, grid, support


def prepare_daily_dataset(
    observations_path: Path = OBSERVATIONS,
    cams_path: Path = CAMS_HOURLY,
    weather_path: Path = WEATHER_HOURLY,
    protocol: Protocol = Protocol(),
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(observations_path, low_memory=False)
    observations = observations[observations["quality_flag"].fillna("").astype(str).str.lower().eq("ok")].copy()
    observations["timestamp_utc"] = pd.to_datetime(observations["timestamp_utc"], utc=True, errors="coerce")

    arpa = observations[observations["source_type"].eq("arpa")].copy()
    arpa["period_start_utc"] = pd.to_datetime(arpa["period_start_utc"], utc=True, errors="coerce")
    arpa["period_end_utc"] = pd.to_datetime(arpa["period_end_utc"], utc=True, errors="coerce")
    arpa["available_at_utc"] = pd.to_datetime(arpa["available_at_utc"], utc=True, errors="coerce")
    arpa["local_date"] = _local_date(arpa["period_start_utc"])
    arpa = arpa.rename(columns={"value_ug_m3": "target"})

    cams_daily, weather_daily, grid, support = _read_hourly_covariates(cams_path, weather_path)
    station_grid = _nearest_grid(arpa, grid)
    arpa = arpa.merge(station_grid, on="source_id", how="left", validate="many_to_one")
    arpa["within_model_domain"] = arpa["model_grid_distance_m"].le(protocol.max_grid_distance_m)
    station_inventory = (
        arpa.groupby(["source_id", "station_name", "lat", "lon", "model_grid_id", "model_grid_distance_m", "within_model_domain"], observed=True)
        .agg(
            rows=("target", "size"),
            valid_days=("local_date", "nunique"),
            pollutants=("pollutant", lambda values: ";".join(sorted(set(values.astype(str))))),
            start=("local_date", "min"),
            end=("local_date", "max"),
        )
        .reset_index()
    )
    arpa = arpa[arpa["within_model_domain"]].copy()

    arpa = arpa.merge(cams_daily, left_on=["local_date", "model_grid_id"], right_on=["local_date", "grid_id"], how="left")
    arpa = arpa.drop(columns=["grid_id_y"], errors="ignore").rename(columns={"grid_id_x": "native_grid_id"})
    arpa = arpa.merge(weather_daily, left_on=["local_date", "model_grid_id"], right_on=["local_date", "grid_id"], how="left")
    arpa = arpa.drop(columns=["grid_id"], errors="ignore")
    arpa["cams_value"] = np.where(arpa["pollutant"].eq("pm10"), arpa["cams_pm10"], arpa["cams_pm2_5"])

    low_cost = observations[observations["source_type"].astype(str).str.startswith("sensesquare")].copy()
    low_cost["local_date"] = _local_date(low_cost["timestamp_utc"])
    low_cost_daily = (
        low_cost.groupby(["local_date", "source_id", "source_type", "pollutant", "lat", "lon"], observed=True)
        .agg(spatial_value=("value_ug_m3", "median"), hourly_mean=("value_ug_m3", "mean"), hours=("timestamp_utc", "nunique"))
        .reset_index()
    )
    low_cost_daily["eligible"] = low_cost_daily["hours"].ge(protocol.min_low_cost_hours)
    low_cost_daily["eligible_date"] = low_cost_daily["local_date"].where(low_cost_daily["eligible"])
    eligible_low_cost = low_cost_daily[low_cost_daily["eligible"]].copy()
    low_cost_groups = {
        (date, pollutant): group
        for (date, pollutant), group in eligible_low_cost.groupby(["local_date", "pollutant"], observed=True)
    }
    low_cost_features: list[dict[str, float]] = []
    for _, target in arpa.iterrows():
        sources = low_cost_groups.get((target["local_date"], target["pollutant"]), pd.DataFrame())
        low_cost_features.append(_spatial_summary(target, sources, "lcs"))
    arpa = pd.concat([arpa.reset_index(drop=True), pd.DataFrame(low_cost_features)], axis=1)

    low_cost_inventory = (
        low_cost_daily.groupby(["source_id", "source_type", "lat", "lon"], observed=True)
        .agg(
            rows=("spatial_value", "size"),
            eligible_source_days=("eligible_date", "nunique"),
            observed_days=("local_date", "nunique"),
            start=("local_date", "min"),
            end=("local_date", "max"),
            median_hours=("hours", "median"),
        )
        .reset_index()
    )
    low_cost_inventory["coverage_class"] = np.where(low_cost_inventory["eligible_source_days"].ge(60), "long", "short")
    nearest_lcs_rows: list[dict[str, Any]] = []
    for station in station_inventory.itertuples(index=False):
        distances = _distance_m(
            station.lat,
            station.lon,
            low_cost_inventory["lat"].to_numpy(dtype=float),
            low_cost_inventory["lon"].to_numpy(dtype=float),
        )
        nearest = int(np.argmin(distances))
        nearest_lcs_rows.append(
            {
                "source_id": station.source_id,
                "nearest_low_cost_source_id": low_cost_inventory.iloc[nearest]["source_id"],
                "nearest_low_cost_distance_m": float(distances[nearest]),
            }
        )
    station_inventory = station_inventory.merge(pd.DataFrame(nearest_lcs_rows), on="source_id", how="left", validate="one_to_one")

    arpa["record_id"] = arpa["source_id"].astype(str) + "|" + arpa["pollutant"].astype(str) + "|" + arpa["local_date"].astype(str)
    arpa = arpa.sort_values(["local_date", "pollutant", "source_id"]).reset_index(drop=True)
    complete_hours = arpa[["cams_hours", "weather_hours"]].ge(23).all(axis=1)
    profile = {
        "analysis_grain": "one ARPA station-pollutant-local-day observation",
        "rows": int(len(arpa)),
        "dates": int(arpa["local_date"].nunique()),
        "start": arpa["local_date"].min(),
        "end": arpa["local_date"].max(),
        "stations": int(arpa["source_id"].nunique()),
        "station_pollutant_pairs": int(arpa[["source_id", "pollutant"]].drop_duplicates().shape[0]),
        "rows_by_pollutant": arpa["pollutant"].value_counts().to_dict(),
        "stations_by_pollutant": arpa.groupby("pollutant", observed=True)["source_id"].nunique().to_dict(),
        "excluded_stations_outside_domain": int((~station_inventory["within_model_domain"]).sum()),
        "complete_hourly_covariate_fraction": float(complete_hours.mean()),
        **support,
        "low_cost_sources": int(low_cost_inventory["source_id"].nunique()),
        "low_cost_sources_with_60_eligible_days": int(
            low_cost_inventory.loc[low_cost_inventory["eligible_source_days"].ge(60), "source_id"].nunique()
        ),
        "duplicate_keys": int(arpa.duplicated(["source_id", "pollutant", "local_date"]).sum()),
        "target_missing": int(arpa["target"].isna().sum()),
    }
    return arpa, profile, station_inventory, low_cost_inventory


def _split_origins(dates: Sequence[pd.Timestamp], protocol: Protocol) -> list[int]:
    last_origin = len(dates) - protocol.validation_days - protocol.test_days
    return list(range(protocol.min_train_days, last_origin + 1, protocol.step_days))


def _add_arpa_features(rows: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    groups = {
        (date, pollutant): group
        for (date, pollutant), group in pool.groupby(["local_date", "pollutant"], observed=True)
    }
    features: list[dict[str, float]] = []
    for _, target in rows.iterrows():
        sources = groups.get((target["local_date"], target["pollutant"]), pd.DataFrame()).copy()
        if not sources.empty:
            sources = sources[sources["source_id"].astype(str).ne(str(target["source_id"]))]
            sources = sources.rename(columns={"target": "spatial_value"})
        features.append(_spatial_summary(target, sources, "arpa"))
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(features)], axis=1)


METEO_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "surface_pressure_hpa",
    "pressure_msl_hpa",
    "wind_speed_m_s",
    "wind_direction_sin",
    "wind_direction_cos",
    "boundary_layer_height_m",
]
ARPA_FEATURES = [
    "arpa_idw",
    "arpa_nearest",
    "arpa_nearest_distance_m",
    "arpa_source_count",
    "arpa_source_mean",
    "arpa_source_std",
    "arpa_source_min",
    "arpa_source_max",
]
LCS_FEATURES = [
    "lcs_idw",
    "lcs_nearest",
    "lcs_nearest_distance_m",
    "lcs_source_count",
    "lcs_source_mean",
    "lcs_source_std",
    "lcs_source_min",
    "lcs_source_max",
]
REGIONAL_FEATURES = ["cams_value", "model_grid_distance_m", *METEO_FEATURES, "lat", "lon"]
FEATURE_VARIANTS = {
    "regional": REGIONAL_FEATURES,
    "arpa_spatial": [*ARPA_FEATURES, "lat", "lon"],
    "regional_arpa": [*REGIONAL_FEATURES, *ARPA_FEATURES],
    "full": [*REGIONAL_FEATURES, *ARPA_FEATURES, *LCS_FEATURES],
}


def _prepare_features(train: pd.DataFrame, frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    prepared_train = train.copy()
    prepared_frames = [frame.copy() for frame in frames]
    effective = list(columns)
    for column in columns:
        median = pd.to_numeric(prepared_train[column], errors="coerce").median()
        if pd.isna(median):
            median = 0.0
        has_missing = prepared_train[column].isna().any() or any(frame[column].isna().any() for frame in prepared_frames)
        if has_missing:
            indicator = f"{column}__missing"
            prepared_train[indicator] = prepared_train[column].isna().astype(float)
            for frame in prepared_frames:
                frame[indicator] = frame[column].isna().astype(float)
            effective.append(indicator)
        prepared_train[column] = pd.to_numeric(prepared_train[column], errors="coerce").fillna(float(median))
        for frame in prepared_frames:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(float(median))
    return prepared_train, prepared_frames, effective


def _models(seed: int) -> dict[str, Any]:
    return {
        "rf": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=seed,
            n_jobs=1,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def _baseline_predictions(frame: pd.DataFrame, climatology: float) -> dict[str, np.ndarray]:
    return {
        "climatology": np.full(len(frame), float(climatology)),
        "cams_raw": frame["cams_value"].to_numpy(dtype=float),
        "arpa_nearest": frame["arpa_nearest"].to_numpy(dtype=float),
        "arpa_idw": frame["arpa_idw"].to_numpy(dtype=float),
        "arpa_idw_p2": frame["arpa_idw_p2"].to_numpy(dtype=float),
        "arpa_idw_p3": frame["arpa_idw_p3"].to_numpy(dtype=float),
        "lcs_idw": frame["lcs_idw"].to_numpy(dtype=float),
    }


def _mae_for_selection(actual: pd.Series, predicted: np.ndarray) -> float:
    mask = np.isfinite(actual.to_numpy(dtype=float)) & np.isfinite(predicted)
    return float(mean_absolute_error(actual.to_numpy(dtype=float)[mask], predicted[mask])) if mask.any() else math.inf


def run_validation(daily: pd.DataFrame, protocol: Protocol = Protocol()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.Timestamp(value) for value in daily["local_date"].dropna().unique())
    origins = _split_origins(dates, protocol)
    predictions: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for pollutant in POLLUTANTS:
        pollutant_data = daily[daily["pollutant"].eq(pollutant)].copy()
        heldout_stations = sorted(pollutant_data["source_id"].unique())
        for heldout_position, heldout in enumerate(heldout_stations):
            source_pool = pollutant_data[pollutant_data["source_id"].ne(heldout)].copy()
            featured = _add_arpa_features(pollutant_data, source_pool)
            for origin_position, origin in enumerate(origins):
                train_dates = dates[:origin]
                validation_dates = dates[origin : origin + protocol.validation_days]
                test_dates = dates[origin + protocol.validation_days : origin + protocol.validation_days + protocol.test_days]
                fold_id = f"{pollutant}|{heldout}|{origin_position:02d}"
                train = featured[featured["local_date"].isin(train_dates) & featured["source_id"].ne(heldout)].copy()
                validation = featured[featured["local_date"].isin(validation_dates) & featured["source_id"].ne(heldout)].copy()
                test = featured[featured["local_date"].isin(test_dates) & featured["source_id"].eq(heldout)].copy()
                split_rows.append(
                    {
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
                        "test_rows": len(test),
                        "heldout_in_train": bool(train["source_id"].eq(heldout).any()),
                        "heldout_in_validation": bool(validation["source_id"].eq(heldout).any()),
                        "test_time_overlap": bool(set(train["local_date"]) & set(test["local_date"]) or set(validation["local_date"]) & set(test["local_date"])),
                    }
                )
                if train.empty or validation.empty or test.empty:
                    continue

                climatology = float(train["target"].mean())
                validation_baselines = _baseline_predictions(validation, climatology)
                eligible_baselines = {
                    name: _mae_for_selection(validation["target"], values)
                    for name, values in validation_baselines.items()
                    if np.isfinite(values).mean() >= 0.95
                }
                selected_baseline = min(eligible_baselines, key=eligible_baselines.get)

                validation_ml: dict[str, np.ndarray] = {}
                for algorithm in ("rf", "ridge"):
                    model = _models(protocol.seed + heldout_position * 100 + origin_position)[algorithm]
                    fit_train, (fit_validation,), effective = _prepare_features(train, [validation], FEATURE_VARIANTS["full"])
                    model.fit(fit_train[effective], fit_train["target"])
                    validation_ml[f"{algorithm}_full"] = model.predict(fit_validation[effective])
                selected_ml = min(
                    validation_ml,
                    key=lambda name: _mae_for_selection(validation["target"], validation_ml[name]),
                )

                fit_base = pd.concat([train, validation], ignore_index=True)
                test_predictions = _baseline_predictions(test, float(fit_base["target"].mean()))
                for variant, features in FEATURE_VARIANTS.items():
                    prepared_base, (prepared_test,), effective = _prepare_features(fit_base, [test], features)
                    for algorithm, model in _models(protocol.seed + heldout_position * 100 + origin_position).items():
                        model.fit(prepared_base[effective], prepared_base["target"])
                        test_predictions[f"{algorithm}_{variant}"] = model.predict(prepared_test[effective])
                test_predictions["selected_baseline"] = test_predictions[selected_baseline]
                test_predictions["selected_ml"] = test_predictions[selected_ml]

                selections.append(
                    {
                        "fold_id": fold_id,
                        "selected_baseline": selected_baseline,
                        "selected_baseline_validation_mae": eligible_baselines[selected_baseline],
                        "selected_ml": selected_ml,
                        "selected_ml_validation_mae": _mae_for_selection(validation["target"], validation_ml[selected_ml]),
                        "validation_baseline_mae": json.dumps(eligible_baselines, sort_keys=True),
                        "validation_ml_mae": json.dumps(
                            {name: _mae_for_selection(validation["target"], values) for name, values in validation_ml.items()},
                            sort_keys=True,
                        ),
                    }
                )
                for model_name, values in test_predictions.items():
                    part = test[
                        ["record_id", "local_date", "source_id", "station_name", "pollutant", "target", "lat", "lon", "arpa_source_count", "lcs_source_count"]
                    ].copy()
                    part["fold_id"] = fold_id
                    part["model"] = model_name
                    part["prediction"] = values
                    part["error"] = part["prediction"] - part["target"]
                    part["absolute_error"] = part["error"].abs()
                    part["squared_error"] = part["error"] ** 2
                    predictions.append(part)

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(selections), pd.DataFrame(split_rows)


def _metric_row(frame: pd.DataFrame, scope: str, model: str, pollutant: str = "all") -> dict[str, Any]:
    actual = frame["target"].to_numpy(dtype=float)
    predicted = frame["prediction"].to_numpy(dtype=float)
    return {
        "scope": scope,
        "pollutant": pollutant,
        "model": model,
        "n": len(frame),
        "dates": frame["local_date"].nunique(),
        "stations": frame["source_id"].nunique(),
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
    return pd.DataFrame(rows).sort_values(["scope", "pollutant", "mae"]).reset_index(drop=True)


def source_concordance(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarise uncalibrated low-cost/CAMS agreement with daily ARPA network means."""
    dates = sorted(daily["local_date"].unique())
    test_dates = set(dates[-42:])
    grouped = (
        daily.groupby(["local_date", "pollutant"], observed=True)
        .agg(
            arpa_network_mean=("target", "mean"),
            low_cost_network_mean=("lcs_source_mean", "first"),
            cams_network_mean=("cams_value", "mean"),
        )
        .reset_index()
    )
    grouped["period"] = np.where(grouped["local_date"].isin(test_dates), "test_42_days", "pre_test")
    rows: list[dict[str, Any]] = []
    for pollutant, frame in grouped.groupby("pollutant", observed=True):
        for source_column, source_label in (
            ("low_cost_network_mean", "low_cost"),
            ("cams_network_mean", "cams"),
        ):
            valid = frame.dropna(subset=[source_column, "arpa_network_mean"])
            rows.append(
                {
                    "pollutant": pollutant,
                    "source": source_label,
                    "days": len(valid),
                    "correlation": float(valid[[source_column, "arpa_network_mean"]].corr().iloc[0, 1]),
                    "mean_ratio_to_arpa": float(valid[source_column].mean() / valid["arpa_network_mean"].mean()),
                    "raw_mae_to_arpa_network_mean": float((valid[source_column] - valid["arpa_network_mean"]).abs().mean()),
                    "pre_test_source_mean": float(valid.loc[valid["period"].eq("pre_test"), source_column].mean()),
                    "test_source_mean": float(valid.loc[valid["period"].eq("test_42_days"), source_column].mean()),
                    "pre_test_arpa_mean": float(valid.loc[valid["period"].eq("pre_test"), "arpa_network_mean"].mean()),
                    "test_arpa_mean": float(valid.loc[valid["period"].eq("test_42_days"), "arpa_network_mean"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _moving_block_pooled_estimates(
    daily_sums: np.ndarray,
    daily_counts: np.ndarray,
    iterations: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap ordered date clusters while retaining record-level weighting."""
    daily_sums = np.asarray(daily_sums, dtype=float)
    daily_counts = np.asarray(daily_counts, dtype=float)
    if daily_sums.shape != daily_counts.shape:
        raise ValueError("daily_sums and daily_counts must have the same shape")
    n = len(daily_sums)
    if n == 0 or np.any(daily_counts <= 0):
        raise ValueError("moving-block bootstrap requires non-empty positive date counts")
    rng = np.random.default_rng(seed)
    starts = np.arange(n)
    estimates = np.empty(iterations, dtype=float)
    blocks_needed = int(math.ceil(n / block_length))
    for iteration in range(iterations):
        sampled: list[int] = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            sampled.extend((start + offset) % n for offset in range(block_length))
        indices = np.asarray(sampled[:n], dtype=int)
        estimates[iteration] = float(daily_sums[indices].sum() / daily_counts[indices].sum())
    return estimates


def bootstrap_metrics(predictions: pd.DataFrame, protocol: Protocol) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, scoped in [("overall", predictions), *[("pollutant", predictions[predictions["pollutant"].eq(p)]) for p in POLLUTANTS]]:
        pollutant = "all" if scope == "overall" else str(scoped["pollutant"].iloc[0])
        for model, frame in scoped.groupby("model", observed=True):
            daily = frame.groupby("local_date", observed=True)["absolute_error"].agg(["sum", "count"]).sort_index()
            estimates = _moving_block_pooled_estimates(
                daily["sum"].to_numpy(),
                daily["count"].to_numpy(),
                protocol.bootstrap_iterations,
                protocol.bootstrap_block_days,
                protocol.seed,
            )
            rows.append(
                {
                    "scope": scope,
                    "pollutant": pollutant,
                    "model": model,
                    "date_clusters": len(daily),
                    "mae": float(frame["absolute_error"].mean()),
                    "mae_ci_low": float(np.quantile(estimates, 0.025)),
                    "mae_ci_high": float(np.quantile(estimates, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: pd.DataFrame, protocol: Protocol, baseline: str = "selected_baseline") -> pd.DataFrame:
    base = predictions[predictions["model"].eq(baseline)][["record_id", "local_date", "source_id", "pollutant", "absolute_error"]].rename(
        columns={"absolute_error": "baseline_absolute_error"}
    )
    rows: list[dict[str, Any]] = []
    for model in sorted(set(predictions["model"]) - {baseline}):
        candidate = predictions[predictions["model"].eq(model)][["record_id", "local_date", "source_id", "pollutant", "absolute_error"]]
        paired = candidate.merge(base, on=["record_id", "local_date", "source_id", "pollutant"], how="inner", validate="one_to_one")
        paired["difference"] = paired["absolute_error"] - paired["baseline_absolute_error"]
        for scope, frame in [("overall", paired), *[("pollutant", paired[paired["pollutant"].eq(p)]) for p in POLLUTANTS]]:
            pollutant = "all" if scope == "overall" else str(frame["pollutant"].iloc[0])
            daily = frame.groupby("local_date", observed=True)["difference"].agg(["sum", "count"]).sort_index()
            site_effects = frame.groupby("source_id", observed=True)["difference"].mean().to_numpy(dtype=float)
            sign_flip_estimates = np.asarray(
                [
                    float(np.mean(site_effects * np.asarray(signs, dtype=float)))
                    for signs in itertools.product((-1.0, 1.0), repeat=len(site_effects))
                ]
            )
            observed_site_mean = float(np.mean(site_effects))
            sign_flip_p = float(
                np.mean(np.abs(sign_flip_estimates) >= abs(observed_site_mean) - np.finfo(float).eps * 10)
            )
            estimates = _moving_block_pooled_estimates(
                daily["sum"].to_numpy(),
                daily["count"].to_numpy(),
                protocol.bootstrap_iterations,
                protocol.bootstrap_block_days,
                protocol.seed + 11,
            )
            rows.append(
                {
                    "scope": scope,
                    "pollutant": pollutant,
                    "model": model,
                    "baseline": baseline,
                    "n": len(frame),
                    "date_clusters": len(daily),
                    "site_clusters": len(site_effects),
                    "mean_mae_difference": float(frame["difference"].mean()),
                    "difference_ci_low": float(np.quantile(estimates, 0.025)),
                    "difference_ci_high": float(np.quantile(estimates, 0.975)),
                    "site_mean_difference": observed_site_mean,
                    "site_difference_min": float(np.min(site_effects)),
                    "site_difference_max": float(np.max(site_effects)),
                    "site_sign_flip_p": sign_flip_p,
                }
            )
    return pd.DataFrame(rows)


def _claim_assessment(profile: Mapping[str, Any], metrics: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
    primary_candidate = "rf_full"
    primary_baseline = "arpa_idw"
    result = {
        "strong_ml_result": False,
        "primary_candidate": primary_candidate,
        "primary_baseline": primary_baseline,
        "required_relative_mae_improvement": 0.05,
        "reasons": [],
    }
    overall = metrics[(metrics["scope"].eq("overall")) & (metrics["pollutant"].eq("all"))].set_index("model")
    if profile["dates"] < 84 or min(profile["stations_by_pollutant"].values()) < 4:
        result["reasons"].append("minimum_daily_coverage_not_met")
    if primary_candidate not in overall.index or primary_baseline not in overall.index:
        result["reasons"].append("required_models_unavailable")
        return result
    skill = 1.0 - float(overall.loc[primary_candidate, "mae"]) / float(overall.loc[primary_baseline, "mae"])
    result["relative_mae_skill"] = skill
    if skill < 0.05:
        result["reasons"].append("mae_improvement_below_5_percent")
    comparison = paired[
        (paired["scope"].eq("overall"))
        & (paired["model"].eq(primary_candidate))
        & (paired["baseline"].eq(primary_baseline))
    ]
    if comparison.empty or float(comparison.iloc[0]["difference_ci_high"]) >= 0.0:
        result["reasons"].append("paired_interval_not_entirely_favourable")
    if comparison.empty or float(comparison.iloc[0]["site_sign_flip_p"]) > 0.05:
        result["reasons"].append("site_level_evidence_not_conclusive")
    pollutant_pairs = paired[
        (paired["scope"].eq("pollutant"))
        & (paired["model"].eq(primary_candidate))
        & (paired["baseline"].eq(primary_baseline))
    ]
    if len(pollutant_pairs) != len(POLLUTANTS) or (pollutant_pairs["mean_mae_difference"] >= 0.0).any():
        result["reasons"].append("improvement_not_present_for_each_pollutant")
    result["strong_ml_result"] = not result["reasons"]
    return result


def _save_publication_figure(fig: plt.Figure, path: Path) -> None:
    """Write figures without volatile creation timestamps."""
    metadata: dict[str, Any] | None = None
    if path.suffix.lower() == ".pdf":
        metadata = {"CreationDate": None, "ModDate": None}
    elif path.suffix.lower() == ".svg":
        metadata = {"Date": None}
    fig.savefig(path, dpi=220, bbox_inches="tight", metadata=metadata)


def _plot_study_area(
    daily: pd.DataFrame,
    station_inventory: pd.DataFrame,
    low_cost_inventory: pd.DataFrame,
    grid: pd.DataFrame,
    figures: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.scatter(grid["lon"], grid["lat"], s=6, c="#d9dee5", label="model grid cells", zorder=1)
    included = station_inventory[station_inventory["within_model_domain"]]
    excluded = station_inventory[~station_inventory["within_model_domain"]]
    axis.scatter(included["lon"], included["lat"], marker="^", s=85, c="#b5282f", edgecolor="white", label="ARPA targets", zorder=4)
    if not excluded.empty:
        axis.scatter(excluded["lon"], excluded["lat"], marker="x", s=55, c="#777777", label="ARPA outside domain", zorder=3)
    long = low_cost_inventory[low_cost_inventory["coverage_class"].eq("long")]
    short = low_cost_inventory[low_cost_inventory["coverage_class"].eq("short")]
    axis.scatter(long["lon"], long["lat"], marker="o", s=55, c="#176b8f", edgecolor="white", label="low-cost, >=60 days", zorder=4)
    axis.scatter(short["lon"], short["lat"], marker="o", s=28, facecolors="none", edgecolors="#176b8f", label="low-cost, <60 days", zorder=3)
    for row in included.itertuples(index=False):
        axis.annotate(str(row.source_id).split(":")[-1], (row.lon, row.lat), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude (degrees north)")
    axis.set_title("Comune di Milano model domain and observation sites")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    for extension in ("pdf", "svg", "png"):
        _save_publication_figure(fig, figures / f"study_area.{extension}")
    plt.close(fig)


def _plot_model_comparison(metrics: pd.DataFrame, bootstrap: pd.DataFrame, figures: Path) -> None:
    display = ["arpa_idw", "cams_raw", "lcs_idw", "ridge_regional", "rf_arpa_spatial", "rf_full"]
    overall = metrics[(metrics["scope"].eq("overall")) & metrics["model"].isin(display)].merge(
        bootstrap[(bootstrap["scope"].eq("overall")) & bootstrap["model"].isin(display)][["model", "mae_ci_low", "mae_ci_high"]],
        on="model",
        how="left",
    )
    overall["order"] = overall["model"].map({name: index for index, name in enumerate(display)})
    overall = overall.sort_values("order", ascending=False)
    pretty_labels = {
        "arpa_idw": "ARPA IDW",
        "cams_raw": "Raw CAMS",
        "lcs_idw": "Low-cost IDW",
        "ridge_regional": "Ridge, regional",
        "rf_arpa_spatial": "RF, ARPA spatial",
        "rf_full": "RF, full",
    }
    labels = [pretty_labels.get(name, name.replace("_", " ")) for name in overall["model"]]
    values = overall["mae"].to_numpy()
    errors = np.vstack([values - overall["mae_ci_low"].to_numpy(), overall["mae_ci_high"].to_numpy() - values])
    colors = ["#176b8f" if name == "arpa_idw" else "#b5282f" if name == "rf_full" else "#75879a" for name in overall["model"]]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(overall))
    axis.barh(y, values, color=colors, alpha=0.9)
    axis.errorbar(values, y, xerr=errors, fmt="none", ecolor="#1f2630", capsize=3, linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel(r"MAE ($\mu$g m$^{-3}$)")
    axis.set_title("Strict unseen-station daily test performance")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for extension in ("pdf", "svg", "png"):
        _save_publication_figure(fig, figures / f"model_comparison.{extension}")
    plt.close(fig)


def _plot_paired_difference_summary(
    predictions: pd.DataFrame,
    paired: pd.DataFrame,
    figures: Path,
) -> None:
    """Separate temporal uncertainty from the effective spatial replicates."""

    comparison = paired[
        paired["model"].eq("rf_full") & paired["baseline"].eq("arpa_idw")
    ].copy()
    requested = pd.concat(
        [
            comparison[(comparison["scope"].eq("overall")) & comparison["pollutant"].eq("all")],
            comparison[(comparison["scope"].eq("pollutant")) & comparison["pollutant"].isin(POLLUTANTS)],
        ],
        ignore_index=True,
    )
    requested["display"] = requested["pollutant"].map(
        {"all": "Overall", "pm10": r"PM$_{10}$", "pm2_5": r"PM$_{2.5}$"}
    )
    requested["order"] = requested["pollutant"].map({"all": 0, "pm10": 1, "pm2_5": 2})
    requested = requested.sort_values("order")

    wide = predictions[predictions["model"].isin(["rf_full", "arpa_idw"])].pivot_table(
        index=["record_id", "source_id"],
        columns="model",
        values="absolute_error",
        aggfunc="first",
    )
    wide = wide.dropna(subset=["rf_full", "arpa_idw"]).reset_index()
    wide["difference"] = wide["rf_full"] - wide["arpa_idw"]
    sites = wide.groupby("source_id", as_index=False)["difference"].mean().sort_values("difference")
    sites["display"] = sites["source_id"].astype(str).str.split(":").str[-1]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True)
    x_min = min(-0.25, float(requested["difference_ci_low"].min()) - 0.05)
    x_max = max(2.10, float(sites["difference"].max()) + 0.10)

    y_left = np.arange(len(requested))
    means = requested["mean_mae_difference"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            means - requested["difference_ci_low"].to_numpy(dtype=float),
            requested["difference_ci_high"].to_numpy(dtype=float) - means,
        ]
    )
    axes[0].errorbar(
        means,
        y_left,
        xerr=errors,
        fmt="D",
        color="#176b8f",
        ecolor="#273746",
        markersize=5,
        capsize=3,
        linewidth=1.1,
    )
    axes[0].set_yticks(y_left, requested["display"])
    axes[0].invert_yaxis()
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")

    y_right = np.arange(len(sites))
    axes[1].scatter(sites["difference"], y_right, s=34, color="#596b7d", edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(y_right, sites["display"])
    axes[1].invert_yaxis()
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")

    for axis in axes:
        axis.axvline(0.0, color="#222222", linewidth=0.9)
        axis.set_xlim(x_min, x_max)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_xlabel(r"$\Delta$MAE ($\mu$g m$^{-3}$)")
    fig.tight_layout(w_pad=1.0)
    for extension in ("pdf", "svg", "png"):
        _save_publication_figure(fig, figures / f"paired_difference_summary.{extension}")
    plt.close(fig)


def make_report(
    output_dir: Path,
    profile: Mapping[str, Any],
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    claim: Mapping[str, Any],
) -> Path:
    overall = metrics[metrics["scope"].eq("overall")].sort_values("mae")
    lines = [
        "# Daily spatial validation: Comune di Milano",
        "",
        f"Status: **{'strong_result' if claim['strong_ml_result'] else 'claim_limited'}**",
        "",
        f"Data: {profile['rows']} ARPA station-pollutant-days, {profile['dates']} local days, "
        f"{profile['stations']} stations in the model domain, and {profile['station_pollutant_pairs']} station-pollutant pairs.",
        "",
        "Protocol: Europe/Rome daily harmonisation; one ARPA station excluded from model fitting, validation selection, and fold-local spatial features; "
        "35-day initial training, 7-day validation, and three disjoint 14-day tests.",
        "",
        "## Overall test metrics",
        "",
        "| Model | n | MAE | RMSE | Bias | R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(f"| {row.model} | {row.n} | {row.mae:.3f} | {row.rmse:.3f} | {row.bias:.3f} | {row.r2:.3f} |")
    comparison = paired[
        (paired["scope"].eq("overall"))
        & paired["model"].eq("rf_full")
        & paired["baseline"].eq("arpa_idw")
    ]
    if not comparison.empty:
        row = comparison.iloc[0]
        lines.extend(
            [
                "",
                "## Primary paired comparison",
                "",
                f"Full RF minus ARPA IDW MAE: {row['mean_mae_difference']:.3f} ug/m3 "
                f"(95% moving-block interval {row['difference_ci_low']:.3f} to {row['difference_ci_high']:.3f}; "
                f"{int(row['date_clusters'])} date clusters; seven-day blocks). The exact site-level sign-flip p-value is "
                f"{row['site_sign_flip_p']:.4f} across {int(row['site_clusters'])} held-out stations. Negative values favour RF.",
            ]
        )
    sensitivity_models = ["arpa_idw", "arpa_idw_p2", "arpa_idw_p3"]
    sensitivity = overall[overall["model"].isin(sensitivity_models)].set_index("model")
    if set(sensitivity_models).issubset(sensitivity.index):
        lines.extend(
            [
                "",
                "## IDW exponent sensitivity",
                "",
                "The primary IDW exponent was fixed at p=1. Recomputing the same held-out targets with p=2 and p=3 changes only the distance decay, not the eligible sources or test rows.",
                "",
                "| Exponent | n | MAE | RMSE | Bias | R2 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model, power in (("arpa_idw", 1), ("arpa_idw_p2", 2), ("arpa_idw_p3", 3)):
            sensitivity_row = sensitivity.loc[model]
            lines.append(
                f"| {power} | {int(sensitivity_row['n'])} | {sensitivity_row['mae']:.3f} | "
                f"{sensitivity_row['rmse']:.3f} | {sensitivity_row['bias']:.3f} | {sensitivity_row['r2']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"Strong ML result: **{bool(claim['strong_ml_result'])}**.",
            "Reasons: " + (", ".join(claim["reasons"]) or "none"),
            "",
            "This is a retrospective daily spatial reconstruction test, not a one-hour-ahead forecast. CAMS reanalysis, same-day weather, "
            "and contemporaneous non-target observations are evaluated only after harmonisation to the ARPA daily support.",
            "",
        ]
    )
    path = output_dir / "validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(
    observations_path: Path = OBSERVATIONS,
    cams_path: Path = CAMS_HOURLY,
    weather_path: Path = WEATHER_HOURLY,
    output_dir: Path = OUTPUT_DIR,
    protocol: Protocol = Protocol(),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    daily, profile, station_inventory, low_cost_inventory = prepare_daily_dataset(
        observations_path, cams_path, weather_path, protocol
    )
    predictions, selections, splits = run_validation(daily, protocol)
    metrics = calculate_metrics(predictions)
    concordance = source_concordance(daily)
    bootstrap = bootstrap_metrics(predictions, protocol)
    paired = pd.concat(
        [
            paired_bootstrap(predictions, protocol, baseline="selected_baseline"),
            paired_bootstrap(predictions, protocol, baseline="arpa_idw"),
            paired_bootstrap(predictions, protocol, baseline="rf_regional_arpa"),
        ],
        ignore_index=True,
    )
    claim = _claim_assessment(profile, metrics, paired)

    daily.to_csv(output_dir / "daily_analysis_table.csv", index=False)
    station_inventory.to_csv(output_dir / "arpa_station_inventory.csv", index=False)
    low_cost_inventory.to_csv(output_dir / "low_cost_source_inventory.csv", index=False)
    splits.to_csv(output_dir / "split_assignments.csv", index=False)
    selections.to_csv(output_dir / "validation_selections.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    concordance.to_csv(output_dir / "source_concordance.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    paired.to_csv(output_dir / "paired_bootstrap_ci.csv", index=False)

    grid = pd.read_csv(cams_path, usecols=["grid_id", "lat", "lon"]).drop_duplicates("grid_id")
    _plot_study_area(daily, station_inventory, low_cost_inventory, grid, figures)
    _plot_model_comparison(metrics, bootstrap, figures)
    _plot_paired_difference_summary(predictions, paired, figures)
    manifest = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC"),
        "protocol": protocol.__dict__,
        "idw_sensitivity": {
            "primary_power": 1,
            "alternative_powers": [2, 3],
            "minimum_distance_m": 100.0,
            "maximum_sources": 8,
        },
        "profile": profile,
        "source_concordance": concordance.to_dict(orient="records"),
        "claim_assessment": claim,
        "leakage_checks": {
            "heldout_in_train_folds": int(splits["heldout_in_train"].sum()),
            "heldout_in_validation_folds": int(splits["heldout_in_validation"].sum()),
            "test_time_overlap_folds": int(splits["test_time_overlap"].sum()),
            "empty_test_folds": int(splits["test_rows"].eq(0).sum()),
        },
        "inputs": {
            "observations": {"path": observations_path, "sha256": _sha256(observations_path)},
            "cams_hourly": {"path": cams_path, "sha256": _sha256(cams_path)},
            "weather_hourly": {"path": weather_path, "sha256": _sha256(weather_path)},
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    _write_json(output_dir / "validation_manifest.json", manifest)
    make_report(output_dir, profile, metrics, bootstrap, paired, claim)
    return {"profile": profile, "metrics": metrics, "paired": paired, "claim": claim, "manifest": manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=OBSERVATIONS)
    parser.add_argument("--cams", type=Path, default=CAMS_HOURLY)
    parser.add_argument("--weather", type=Path, default=WEATHER_HOURLY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run(args.observations, args.cams, args.weather, args.output_dir)
    print(json.dumps({"profile": result["profile"], "claim": result["claim"]}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
