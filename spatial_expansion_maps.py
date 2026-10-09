"""Create evidence-aligned spatial-expansion maps for the ASITA paper.

The figure is descriptive. It applies the fixed primary IDW rule (p=1) to all
same-day inputs after model evaluation; it does not create new validation rows.
Dates are selected mechanically among held-out test dates: retain dates with
maximum source coverage, then choose the daily network median closest to the
median across those eligible dates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "spatial_expansion_maps"
FIGURES = OUT / "figures"

SOURCES = {
    "milan_observations": ROOT / "reports" / "daily_spatial_validation" / "daily_analysis_table.csv",
    "milan_test_predictions": ROOT / "reports" / "daily_spatial_validation" / "test_predictions.csv",
    "milan_cells": ROOT / "data" / "raw" / "milano_city" / "milano_city_squares.geojson",
    "unisa_observations": ROOT / "reports" / "salerno_campus" / "daily_observations.csv",
    "unisa_inventory": ROOT / "reports" / "salerno_campus" / "device_inventory.csv",
    "unisa_cells": ROOT / "data" / "raw" / "fisciano_squares.geojson",
}

INK = "#20262e"
GRID = "#d8dde3"
CMAP = mpl.colormaps["cividis"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_polygons(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: list[dict] = []
    for feature in data["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            groups = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            groups = geometry["coordinates"]
        else:
            continue
        for group in groups:
            exterior = np.asarray(group[0], dtype=float)
            result.append(
                {
                    "name": feature.get("properties", {}).get("name", ""),
                    "xy": exterior,
                }
            )
    return result


def polygon_centroid(xy: np.ndarray) -> tuple[float, float]:
    ring = xy
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    cross = ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]
    area2 = cross.sum()
    if abs(area2) < 1e-14:
        return float(ring[:-1, 0].mean()), float(ring[:-1, 1].mean())
    cx = ((ring[:-1, 0] + ring[1:, 0]) * cross).sum() / (3.0 * area2)
    cy = ((ring[:-1, 1] + ring[1:, 1]) * cross).sum() / (3.0 * area2)
    return float(cx), float(cy)


def distance_m(query_xy: np.ndarray, source_xy: np.ndarray) -> np.ndarray:
    """Local equirectangular distance; adequate for each plotted domain."""
    mean_lat = np.deg2rad((query_xy[:, None, 1] + source_xy[None, :, 1]) / 2.0)
    dx = (query_xy[:, None, 0] - source_xy[None, :, 0]) * 111_320.0 * np.cos(mean_lat)
    dy = (query_xy[:, None, 1] - source_xy[None, :, 1]) * 110_540.0
    return np.hypot(dx, dy)


def idw(query_xy: np.ndarray, source_xy: np.ndarray, values: np.ndarray, max_neighbors: int = 8) -> np.ndarray:
    distances = distance_m(query_xy, source_xy)
    order = np.argsort(distances, axis=1)[:, : min(max_neighbors, source_xy.shape[0])]
    selected_d = np.take_along_axis(distances, order, axis=1)
    selected_v = values[order]
    weights = 1.0 / np.maximum(selected_d, 100.0)
    return np.sum(weights * selected_v, axis=1) / np.sum(weights, axis=1)


def choose_date(data: pd.DataFrame, date_col: str, value_col: str, id_col: str) -> tuple[str, pd.DataFrame]:
    clean = data.dropna(subset=[date_col, value_col, "lat", "lon"]).drop_duplicates([date_col, id_col])
    summary = (
        clean.groupby(date_col)
        .agg(n=(id_col, "nunique"), daily_median=(value_col, "median"))
        .reset_index()
    )
    eligible = summary.loc[summary["n"].eq(summary["n"].max())].copy()
    reference = float(eligible["daily_median"].median())
    eligible["distance_to_reference"] = (eligible["daily_median"] - reference).abs()
    selected = str(eligible.sort_values(["distance_to_reference", date_col]).iloc[0][date_col])
    return selected, clean.loc[clean[date_col].astype(str).eq(selected)].copy()


def convex_path(points: np.ndarray) -> tuple[MplPath, np.ndarray]:
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    closed = np.vstack([vertices, vertices[0]])
    return MplPath(closed), closed


def padded_limits(points: np.ndarray, fraction: float = 0.06) -> tuple[float, float, float, float]:
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    padx = max((xmax - xmin) * fraction, 0.0004)
    pady = max((ymax - ymin) * fraction, 0.0004)
    return xmin - padx, xmax + padx, ymin - pady, ymax + pady


def style_axis(ax: plt.Axes, mean_lat: float, panel: str, title: str) -> None:
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    ax.set_title(title, fontsize=8.2, fontweight="bold", color=INK, pad=4)
    ax.text(0.015, 0.985, panel, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, fontweight="bold", color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
    ax.tick_params(labelsize=6.6, colors=INK, length=2.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#7a838d")
        spine.set_linewidth(0.6)


def add_colorbar(fig: plt.Figure, ax: plt.Axes, norm: Normalize, label: str) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    bar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.055, pad=0.105, aspect=28)
    bar.set_label(label, fontsize=6.7, labelpad=1.5)
    bar.ax.tick_params(labelsize=6.2, length=2)
    bar.outline.set_linewidth(0.5)


def load_panels() -> tuple[dict, dict]:
    milan_test = pd.read_csv(SOURCES["milan_test_predictions"])
    milan_dates = set(
        milan_test.loc[
            milan_test["pollutant"].eq("pm10") & milan_test["model"].eq("arpa_idw"),
            "local_date",
        ].astype(str)
    )
    milan = pd.read_csv(SOURCES["milan_observations"])
    milan = milan.loc[
        milan["pollutant"].eq("pm10")
        & milan["within_model_domain"].eq(True)
        & milan["observation_valid"].eq(True)
        & milan["local_date"].astype(str).isin(milan_dates)
    ]
    milan_date, milan_day = choose_date(milan, "local_date", "target", "source_id")

    unisa = pd.read_csv(SOURCES["unisa_observations"])
    unisa["date"] = unisa["date"].astype(str)
    unisa = unisa.loc[unisa["date"].between("2026-07-24", "2026-08-13")]
    unisa_date, unisa_day = choose_date(unisa, "date", "pm10", "device_id")

    return (
        {"date": milan_date, "data": milan_day, "value": "target", "id": "source_id"},
        {"date": unisa_date, "data": unisa_day, "value": "pm10", "id": "device_id"},
    )


def prepare_milan(panel: dict) -> dict:
    cells = feature_polygons(SOURCES["milan_cells"])
    centroids = np.asarray([polygon_centroid(cell["xy"]) for cell in cells])
    day = panel["data"]
    sources = day[["lon", "lat"]].to_numpy(float)
    values = day[panel["value"]].to_numpy(float)
    predictions = idw(centroids, sources, values)
    hull_path, hull_xy = convex_path(sources)
    inside = hull_path.contains_points(centroids, radius=1e-12)
    norm = Normalize(vmin=float(min(values.min(), predictions.min())), vmax=float(max(values.max(), predictions.max())))

    overlap = centroids[inside]
    zoom_limits = padded_limits(np.vstack([overlap, sources]), 0.12)
    domain_xy = np.vstack([cell["xy"] for cell in cells])
    domain_limits = padded_limits(np.vstack([domain_xy, sources]), 0.04)
    return {
        "cells": cells,
        "centroids": centroids,
        "sources": sources,
        "values": values,
        "predictions": predictions,
        "hull_xy": hull_xy,
        "inside": inside,
        "norm": norm,
        "zoom_limits": zoom_limits,
        "domain_limits": domain_limits,
    }


def draw_milan_cells(ax: plt.Axes, prepared: dict, zoom: bool) -> None:
    xmin, xmax, ymin, ymax = prepared["zoom_limits"] if zoom else prepared["domain_limits"]
    for cell, centroid, value, is_inside in zip(
        prepared["cells"], prepared["centroids"], prepared["predictions"], prepared["inside"]
    ):
        if zoom and not (xmin <= centroid[0] <= xmax and ymin <= centroid[1] <= ymax):
            continue
        ax.add_patch(
            MplPolygon(
                cell["xy"], closed=True, facecolor=CMAP(prepared["norm"](value)),
                edgecolor="#d5d9dd", linewidth=0.22,
                alpha=0.96 if is_inside else 0.48,
                hatch=None if is_inside else "////",
            )
        )
    ax.plot(prepared["hull_xy"][:, 0], prepared["hull_xy"][:, 1], color=INK,
            linewidth=0.8, linestyle="--", zorder=4)
    ax.scatter(
        prepared["sources"][:, 0], prepared["sources"][:, 1],
        c=prepared["values"], cmap=CMAP, norm=prepared["norm"],
        s=30 if zoom else 22, edgecolors="white", linewidths=0.75, zorder=5,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def draw_milan_overview(ax: plt.Axes, panel: dict, prepared: dict) -> None:
    draw_milan_cells(ax, prepared, zoom=False)
    xmin, xmax, ymin, ymax = prepared["zoom_limits"]
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                           edgecolor=INK, linewidth=1.0, linestyle=(0, (4, 2)), zorder=7))
    cell_count = len({cell["name"] for cell in prepared["cells"]})
    style_axis(
        ax,
        prepared["sources"][:, 1].mean(),
        "(a)",
        f"Milan municipality | complete {cell_count}-cell domain",
    )
    ax.set_ylabel("Latitude (degrees N)", fontsize=6.8)
    ax.set_xlabel("Longitude (degrees E)", fontsize=6.8)
    ax.legend(
        handles=[
            mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="#6f7b83",
                             markeredgecolor="white", markersize=5, label="same-day ARPA input"),
            Patch(facecolor="#a9afb4", edgecolor="#666", hatch="////", alpha=0.5,
                  label="outside station hull"),
            Patch(facecolor="none", edgecolor=INK, linestyle="--", label="zoom extent"),
        ],
        loc="lower left", fontsize=5.8, frameon=True, framealpha=0.92,
        borderpad=0.35, handlelength=1.5,
    )


def draw_milan_zoom(fig: plt.Figure, ax: plt.Axes, panel: dict, prepared: dict) -> None:
    draw_milan_cells(ax, prepared, zoom=True)
    style_axis(
        ax, prepared["sources"][:, 1].mean(), "(b)",
        f"Milan city interpolation zoom | {panel['date']} | {len(panel['data'])} ARPA",
    )
    ax.set_ylabel("Latitude (degrees N)", fontsize=6.8)
    ax.set_xlabel("Longitude (degrees E)", fontsize=6.8)
    add_colorbar(fig, ax, prepared["norm"], r"IDW PM$_{10}$ ($\mu$g m$^{-3}$)")


def prepare_fisciano(panel: dict) -> dict:
    day = panel["data"]
    sources = day[["lon", "lat"]].to_numpy(float)
    values = day[panel["value"]].to_numpy(float)
    occupied = set(pd.read_csv(SOURCES["unisa_inventory"])["square_id"].dropna().astype(str))
    all_cells = feature_polygons(SOURCES["unisa_cells"])
    campus_cells = [cell for cell in all_cells if cell["name"] in occupied]
    all_xy = np.vstack([cell["xy"] for cell in all_cells])
    campus_xy = np.vstack([cell["xy"] for cell in campus_cells])
    domain_limits = padded_limits(all_xy, 0.025)
    zoom_limits = padded_limits(campus_xy, 0.025)
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    campus_centroids = np.asarray([polygon_centroid(cell["xy"]) for cell in campus_cells])
    campus_values = idw(campus_centroids, sources, values)
    return {
        "all_cells": all_cells,
        "campus_cells": campus_cells,
        "sources": sources,
        "values": values,
        "norm": norm,
        "campus_values": campus_values,
        "domain_limits": domain_limits,
        "zoom_limits": zoom_limits,
    }


def draw_fisciano_overview(ax: plt.Axes, prepared: dict) -> None:
    campus_lookup = {cell["name"]: value for cell, value in zip(prepared["campus_cells"], prepared["campus_values"])}
    for cell in prepared["all_cells"]:
        occupied = cell["name"] in campus_lookup
        ax.add_patch(
            MplPolygon(
                cell["xy"], closed=True,
                facecolor=CMAP(prepared["norm"](campus_lookup[cell["name"]])) if occupied else "#edf0f2",
                edgecolor=INK if occupied else "#b9c0c7",
                linewidth=0.65 if occupied else 0.28,
                alpha=0.98,
            )
        )
    ax.scatter(prepared["sources"][:, 0], prepared["sources"][:, 1], s=10,
               facecolors=INK, edgecolors="white", linewidths=0.35, zorder=5)
    xmin, xmax, ymin, ymax = prepared["zoom_limits"]
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                           edgecolor=INK, linewidth=1.0, linestyle=(0, (4, 2)), zorder=7))
    dxmin, dxmax, dymin, dymax = prepared["domain_limits"]
    ax.set_xlim(dxmin, dxmax)
    ax.set_ylim(dymin, dymax)
    style_axis(ax, prepared["sources"][:, 1].mean(), "(c)", "Fisciano | complete 55-cell reporting grid")
    ax.set_ylabel("Latitude (degrees N)", fontsize=6.8)
    ax.set_xlabel("Longitude (degrees E)", fontsize=6.8)
    ax.legend(
        handles=[
            Patch(facecolor="#c5b66f", edgecolor=INK, label="three occupied campus cells"),
            Patch(facecolor="none", edgecolor=INK, linestyle="--", label="zoom extent"),
        ],
        loc="lower left", fontsize=5.8, frameon=True, framealpha=0.92,
        borderpad=0.35, handlelength=1.5,
    )


def draw_fisciano_zoom(fig: plt.Figure, ax: plt.Axes, panel: dict, prepared: dict) -> dict:
    xmin, xmax, ymin, ymax = prepared["zoom_limits"]
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 180), np.linspace(ymin, ymax, 180))
    query = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(query), dtype=bool)
    for cell in prepared["campus_cells"]:
        inside |= MplPath(cell["xy"]).contains_points(query, radius=1e-12)
    predicted = idw(query, prepared["sources"], prepared["values"]).reshape(gx.shape)
    predicted = np.ma.array(predicted, mask=~inside.reshape(gx.shape))
    ax.pcolormesh(gx, gy, predicted, cmap=CMAP, norm=prepared["norm"], shading="auto", rasterized=True)
    for cell in prepared["campus_cells"]:
        ax.plot(cell["xy"][:, 0], cell["xy"][:, 1], color=INK, linewidth=0.75, zorder=4)
    ax.scatter(
        prepared["sources"][:, 0], prepared["sources"][:, 1],
        c=prepared["values"], cmap=CMAP, norm=prepared["norm"],
        s=23, edgecolors="white", linewidths=0.7, zorder=5,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    style_axis(
        ax, prepared["sources"][:, 1].mean(), "(d)",
        f"UNISA campus zoom | {panel['date']} | {len(panel['data'])} PE.DT",
    )
    ax.set_xlabel("Longitude (degrees E)", fontsize=6.8)
    ax.set_ylabel("Latitude (degrees N)", fontsize=6.8)
    add_colorbar(fig, ax, prepared["norm"], r"IDW reported PM$_{10}$")
    return {"predicted_min": float(predicted.min()), "predicted_max": float(predicted.max())}


def figure_summaries(milan_prepared: dict, fisciano_prepared: dict, fisciano_zoom: dict) -> dict:
    return {
        "milan": {
            "predicted_min": float(milan_prepared["predictions"].min()),
            "predicted_max": float(milan_prepared["predictions"].max()),
            "outside_hull_polygon_parts": int((~milan_prepared["inside"]).sum()),
            "application_cell_count": len({cell["name"] for cell in milan_prepared["cells"]}),
            "polygon_part_count": len(milan_prepared["cells"]),
            "zoom_limits": [float(v) for v in milan_prepared["zoom_limits"]],
        },
        "fisciano": {
            "reporting_cell_count": len({cell["name"] for cell in fisciano_prepared["all_cells"]}),
            "occupied_cell_count": len({cell["name"] for cell in fisciano_prepared["campus_cells"]}),
            "zoom_limits": [float(v) for v in fisciano_prepared["zoom_limits"]],
            **fisciano_zoom,
        },
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.hashsalt": "asita-spatial-expansion-v2",
        }
    )
    milan, unisa = load_panels()
    milan_prepared = prepare_milan(milan)
    fisciano_prepared = prepare_fisciano(unisa)

    fig = plt.figure(figsize=(7.35, 4.90), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1, 1])
    ax_milan_overview = fig.add_subplot(grid[0, 0])
    ax_milan_zoom = fig.add_subplot(grid[0, 1])
    ax_fisciano_overview = fig.add_subplot(grid[1, 0])
    ax_fisciano_zoom = fig.add_subplot(grid[1, 1])

    draw_milan_overview(ax_milan_overview, milan, milan_prepared)
    draw_milan_zoom(fig, ax_milan_zoom, milan, milan_prepared)
    draw_fisciano_overview(ax_fisciano_overview, fisciano_prepared)
    fisciano_zoom = draw_fisciano_zoom(fig, ax_fisciano_zoom, unisa, fisciano_prepared)
    summaries = figure_summaries(milan_prepared, fisciano_prepared, fisciano_zoom)

    pdf = FIGURES / "spatial_expansion_maps.pdf"
    png = FIGURES / "spatial_expansion_maps.png"
    svg = FIGURES / "spatial_expansion_maps.svg"
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "spatial_expansion_maps.py", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight", metadata={"Creator": "spatial_expansion_maps.py", "Date": None})
    plt.close(fig)

    plotted = []
    for domain, panel in (("milan", milan), ("fisciano", unisa)):
        frame = panel["data"][[panel["id"], "lat", "lon", panel["value"]]].copy()
        frame.columns = ["source_id", "lat", "lon", "pm10"]
        frame.insert(0, "domain", domain)
        frame.insert(1, "date", panel["date"])
        plotted.append(frame)
    pd.concat(plotted, ignore_index=True).to_csv(OUT / "plotted_observations.csv", index=False, float_format="%.9f")

    manifest = {
        "purpose": "descriptive post-evaluation IDW spatial expansion; not additional validation",
        "date_selection": "maximum source coverage, then daily network median closest to eligible-date median",
        "idw": {"exponent": 1, "distance_floor_m": 100, "max_neighbors": 8},
        "panels": {
            "milan": {"date": milan["date"], "source_count": len(milan["data"]), **summaries["milan"]},
            "fisciano": {"date": unisa["date"], "source_count": len(unisa["data"]), **summaries["fisciano"]},
        },
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in SOURCES.items()},
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (pdf, png, svg, OUT / "plotted_observations.csv")
        },
        "script": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["panels"], indent=2))


if __name__ == "__main__":
    main()
