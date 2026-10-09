"""Regenerate the unchanged study-area maps as two publication figures."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import pandas as pd
import fitz

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER
OUT = PAPER / 'generated' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('source_maps', ROOT / 'spatial_expansion_maps.py')
maps = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = maps
spec.loader.exec_module(maps)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'pdf.fonttype': 42})
milan, campus = maps.load_panels()
mp, fp = maps.prepare_milan(milan), maps.prepare_fisciano(campus)

selected = []
for domain, panel in (('milan', milan), ('fisciano', campus)):
    frame = panel['data'][[panel['id'], 'lat', 'lon', panel['value']]].copy()
    frame.columns = ['source_id', 'lat', 'lon', 'pm10']
    frame.insert(0, 'domain', domain)
    frame.insert(1, 'date', panel['date'])
    selected.append(frame)
selected = pd.concat(selected, ignore_index=True)
previous = pd.read_csv(ROOT / 'reports/spatial_expansion_maps/plotted_observations.csv')
cols = ['domain', 'date', 'source_id', 'lat', 'lon', 'pm10']
pd.testing.assert_frame_equal(
    selected[cols].sort_values(['domain', 'source_id']).reset_index(drop=True),
    previous[cols].sort_values(['domain', 'source_id']).reset_index(drop=True),
    check_dtype=False, atol=1e-10, rtol=1e-10,
)
assert len({cell['name'] for cell in mp['cells']}) == 263
assert len({cell['name'] for cell in fp['all_cells']}) == 55
assert len(milan['data']) == 4

outputs = []
for name in ('milan', 'campus'):
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.05), layout='constrained')
    if name == 'milan':
        maps.draw_milan_overview(axes[0], milan, mp)
        maps.draw_milan_zoom(fig, axes[1], milan, mp)
        titles = ['(a) Milan: 263 cells', f'(b) Milan: {milan["date"]}']
        color_label = 'IDW PM10 (µg/m³)'
        for ax in axes:
            for patch in ax.patches:
                patch.set_hatch(None)
    else:
        maps.draw_fisciano_overview(axes[0], fp)
        maps.draw_fisciano_zoom(fig, axes[1], campus, fp)
        titles = ['(a) Fisciano: 55 cells', f'(b) Campus: {campus["date"]}']
        color_label = 'IDW PM10 (device units)'
        axes[1].set_xticks([14.785, 14.795])
    for ax, title in zip(axes, titles):
        if ax.get_legend():
            ax.get_legend().remove()
        for text in list(ax.texts):
            text.remove()
        ax.set_title(title, fontsize=7, pad=5)
        ax.set_xlabel('Longitude (°E)', fontsize=7)
        ax.set_ylabel('Latitude (°N)', fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.tick_params(labelsize=7)
    if name == 'campus':
        axes[1].set_xticks([14.785, 14.795])
    fig.axes[2].set_xlabel(color_label, fontsize=7)
    fig.axes[2].tick_params(labelsize=7)
    for text in fig.findobj(Text):
        text.set_fontsize(7)
    pdf = OUT / f'study_area_{name}.pdf'
    fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with fitz.open(pdf) as document:
        assert len(document) == 1
        document[0].get_pixmap(matrix=fitz.Matrix(3, 3)).save(
            PAPER / 'generated' / f'study_area_{name}.png')
    outputs.append({'path': str(pdf), 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()})

manifest = {
    'purpose': 'Separate Milan and Fisciano/campus figures; preserve scientific content',
    'observations_and_dates_unchanged': True,
    'milan_date': milan['date'], 'campus_date': campus['date'],
    'milan_sources': len(milan['data']), 'campus_sources_on_map_date': len(campus['data']),
    'milan_cells': len({cell['name'] for cell in mp['cells']}),
    'fisciano_cells': len({cell['name'] for cell in fp['all_cells']}),
    'outputs': outputs,
    'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in maps.SOURCES.values()],
}
(PAPER / 'generated/study_area_split_manifest.json').write_text(
    json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in manifest.items() if k not in ('inputs', 'outputs')}))
