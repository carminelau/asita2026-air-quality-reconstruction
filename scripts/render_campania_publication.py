"""Render the Campania comparison at its final LNCS width, without reanalysis."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fitz

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "results" / "campania"
OUT = PAPER / "generated" / "figures" / "arpac_it1508_paired_comparison.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
AUDIT = PAPER / "generated" / "figure_audit"
AUDIT.mkdir(parents=True, exist_ok=True)
inputs = [DATA / "paired_comparisons.csv", DATA / "paired_station_differences.csv"]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
comparisons = pd.read_csv(inputs[0])
station_rows = pd.read_csv(inputs[1])
models = ("ridge", "rf")
colours = {"ridge": "#2F6B9A", "rf": "#D06C3B"}
markers = {"ridge": "o", "rf": "s"}
labels = {"ridge": "Ridge", "rf": "Random Forest"}
offsets = {"ridge": -0.13, "rf": 0.13}
stations = ["IT2210A", "IT2213A", "IT1504A", "IT2214A", "IT2227A", "IT0936A"]
names = ["Solofra", "Cava", "Salerno", "Nocera", "Avellino: Alighieri", "Avellino: AV41"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42})
fig = plt.figure(figsize=(122 / 25.4, 2.30), facecolor="white")
a = fig.add_axes([0.105, 0.23, 0.275, 0.60])
b = fig.add_axes([0.685, 0.23, 0.285, 0.60])
plotted = []
for y, (scope, pollutant) in enumerate([("overall", "all"), ("pollutant", "PM10"), ("pollutant", "PM2.5")]):
    for model in models:
        row = comparisons.loc[(comparisons.scope == scope) &
                              (comparisons.pollutant == pollutant) &
                              (comparisons.model == model)].iloc[0]
        value = float(row.mean_mae_difference)
        lo, hi = float(row.difference_ci_low), float(row.difference_ci_high)
        a.errorbar(value, y + offsets[model],
                   xerr=np.array([[value - lo], [hi - value]]),
                   fmt=markers[model], color=colours[model],
                   markersize=3.5, capsize=2, elinewidth=0.8)
        plotted.append({"panel": "a", "model": model, "scope": scope,
                        "pollutant": pollutant, "mean": value, "low": lo, "high": hi})
a.set_yticks([0, 1, 2], ["All", "PM10", "PM2.5"])
a.set_ylim(2.5, -0.5)
a.set_xticks([0, 2, 4])
a.set_title("(a) Date-block intervals", loc="left", pad=5)
for model in models:
    part = station_rows.loc[station_rows.model == model]
    values = []
    for station in stations:
        rows = part.loc[part.station_id == station]
        value = float(np.average(rows.mean_mae_difference, weights=rows.n))
        values.append(value)
        plotted.append({"panel": "b", "model": model, "station_id": station,
                        "mean": value, "n": int(rows.n.sum())})
    b.scatter(values, np.arange(6) + offsets[model], marker=markers[model],
              color=colours[model], s=13, label=labels[model])
b.set_yticks(np.arange(6), names)
b.set_ylim(5.5, -0.5)
b.set_xticks([-2, 0, 2, 4])
b.set_title("(b) Station means", loc="left", pad=5)
for ax in (a, b):
    ax.axvline(0, color="0.25", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("ΔMAE (µg/m³)", labelpad=3)
    ax.tick_params(length=2, pad=2)
handles, legend_labels = b.get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.51, 1.01), handletextpad=0.3)
fig.text(0.51, 0.025, "ARPA Campania IT1508 · Positive values favour IDW",
         ha="center", fontsize=7)
fig.savefig(OUT, metadata={"Creator": "render_campania_publication.py",
                          "CreationDate": None, "ModDate": None})
plt.close(fig)
doc = fitz.open(OUT)
p = doc[0]
spans = [s for block in p.get_text("dict")["blocks"] if "lines" in block
         for line in block["lines"] for s in line["spans"]]
scale = (122 / 25.4 * 72) / p.rect.width
minimum = min(s["size"] * scale for s in spans)
assert minimum >= 6, minimum
assert all(s["bbox"][0] >= -0.5 and s["bbox"][1] >= -0.5 and
           s["bbox"][2] <= p.rect.width + 0.5 and
           s["bbox"][3] <= p.rect.height + 0.5 for s in spans)
assert not p.get_images(), "The comparison figure must remain vector graphics."
p.get_pixmap(matrix=fitz.Matrix(3, 3)).save(AUDIT / "campania_preview.png")
p.get_pixmap(matrix=fitz.Matrix(3, 3), colorspace=fitz.csGRAY).save(AUDIT / "campania_gray.png")
assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
(AUDIT / "figure_verification.json").write_text(json.dumps({
    "input_sha256": hashes, "output_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    "width_mm": 122, "minimum_final_font_pt": minimum,
    "plotted_values": plotted, "no_reanalysis": True,
}, indent=2), encoding="utf-8")
print(f"Figure 4: vector PDF, final minimum font {minimum:.2f} pt; original data unchanged.")
