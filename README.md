# ASITA 2026: spatial air-quality reconstruction

Reproducibility companion for **Multisource Data Integration for Air Quality
Monitoring and Spatial Prediction**, by Carmine Laudato, Monica Sebillo,
Pietro Battistoni, and Daniele Sofia.

The repository preserves the analysis code and aggregate evidence for three
experiments: daily reconstruction at unseen Milan regulatory stations,
independent ARPA Campania IT1508 replication, and internal reconstruction in
the University of Salerno low-cost network. Comparisons include inverse-distance
weighting (IDW) and Random Forest (RF). The study distinguishes spatial
generalisation from internal network consistency. The results do not establish
universal algorithm superiority or regulatory accuracy of low-cost devices.

## What can be reproduced

| Task | Public checkout alone | Original inputs required |
| --- | --- | --- |
| Verify published numerical claims and artifact hashes | Yes; Python standard library | No |
| Export comparison tables from archived aggregate results | Yes | No |
| Run methodological tests on synthetic observations | Yes; install requirements | No |
| Regenerate the Campania comparison figure | Yes; install figure requirements | No |
| Refit Milan and Campania and compare output statistics | No | Yes; verified research snapshots |
| Run the campus refit and diagnose differences from the archive | No | Yes; full numerical agreement remains unresolved |
| Regenerate Milan/campus descriptive maps | No | Yes; observations and grid geometries |
| Reproduce the additional 12-configuration RF tuning sensitivity | Not included | Original supplement must be recovered |

**Provider observations, authenticated responses, individual device identifiers,
and per-record predictions are not distributed.** Aggregates, protocols,
input hashes, original provider download URLs, and source code are included.
See [data access](docs/DATA_ACCESS.md) and [known limitations](docs/KNOWN_LIMITATIONS.md).

**Validation status:** Milan and Campania refits match all seven checked
statistics files. Campus baseline metrics and exact sign-flip tests match;
campus Random Forest metrics and bootstrap intervals differ from the archive.
The strict campus comparison therefore fails. See the
[verification report](docs/VERIFICATION.md) for measured differences.

## Quick verification: no credentials or installation

Run these commands from the repository root, using Python 3.12:

```sh
python scripts/verify_results.py --export-tables
```

This checks 34 numerical claim rows and the hashes of archived artifacts.
Exported CSV tables appear in `generated/tables/`. It checks frozen aggregate
results; it does not claim to recreate measurements or refit models.

## Install the research environment

The main analysis packages are pinned to the versions recorded by the original
research manifests. The archived Python reference version is 3.12.13. The local
runtime used for the October 2026 refits reports Python 3.12.14; the campus
discrepancy has not been attributed to a specific cause.

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Optional acquisition scripts require `requirements-acquisition.txt`; publication
figure checks require `requirements-figures.txt`. The core tests use synthetic
observations and require no API key. GitHub Actions runs public verification
and the tests on Ubuntu (Python 3.12.13) and Windows (Python 3.12.10).
GitHub's Windows runtime catalog does not provide 3.12.13; this CI check uses
the available 3.12 patch release. The original research runtime remains 3.12.13.

## Refit the main experiments

Obtain the original input snapshots following [DATA_ACCESS.md](docs/DATA_ACCESS.md).
Keep their relative paths under a directory supplied through `--data-root`.

```sh
python scripts/verify_inputs.py --experiment all --data-root /path/to/archive
python scripts/reproduce.py --experiment all --data-root /path/to/archive
```

On Windows, replace `/path/to/archive` with a quoted Windows path. An experiment
can also be run individually with `--experiment milan`, `campania`, or `campus`.
Input hashes are checked before fitting. The original analysis modules are
preserved byte-for-byte. Seeds, chronological splits, feature definitions,
bootstrap settings, and spatial exclusion rules are preserved. Outputs go to
the ignored `reports/` directory; input archives are not overwritten.

The runner compares the newly computed metrics and paired statistics with
the public results, allowing numerical differences of `1e-8` plus relative
tolerance `1e-10`. A successful run records its verification in
`generated/reproduction_checks.json`. This checks numerical reproducibility,
not byte identity of timestamps or platform-specific figure metadata.

## Main results

MAE denotes mean absolute error.

| Experiment | Reference MAE | Random Forest MAE | Held-out observations |
| --- | ---: | ---: | ---: |
| Milan: regulatory IDW vs full RF | 4.082 | 4.669 | 244 |
| Campania: regulatory IDW vs regulatory-only RF | 3.848 | 4.938 | 485 |
| Campus PM10: network median vs RF | 1.583 | 2.143 | 371 |
| Campus PM2.5: network median vs RF | 1.429 | 1.873 | 370 |

Regulatory errors are in µg/m³; campus errors remain in device-reported units.
The campus median's paired intervals against IDW include zero. The site-level
Milan and Campania tests do not establish general superiority.

The campus reconstruction benchmark uses IDW exponent 2 and a 10 m distance
floor in the archived code. Milan/Campania benchmarks use exponent 1 and
a 100 m floor; exponents 2 and 3 are sensitivity analyses. Descriptive maps
use exponent 1 after evaluation and add no validation targets. See
[METHODS.md](docs/METHODS.md) for experiment-specific settings.

## Figures

The three external publication figures are preserved in `figures/`. The
workflow diagram is drawn in the manuscript's LaTeX source and is not an
additional experimental result.

```sh
python -m pip install -r requirements-figures.txt
python scripts/render_campania_publication.py
```

The regenerated figure and checks are written to `generated/`. Milan/campus map
regeneration requires the private analysis outputs and the two grid geometries;
see [DATA_ACCESS.md](docs/DATA_ACCESS.md). Figure adapters change file paths
and output locations, not plotting values or model results.

## Citation and access

Use the metadata in [CITATION.cff](CITATION.cff). A GitHub release is not a DOI.
No DOI or accepted-paper status is claimed here. An archival DOI can be added
after the authors register an archive release.

Code is licensed under MIT. Research-result files and provider inputs do not
inherit that code license; see [DATA_POLICY.md](docs/DATA_POLICY.md). Contact
the corresponding author, Carmine Laudato, at `claudato@unisa.it` for the
author-held input snapshots, subject to provider permission.
