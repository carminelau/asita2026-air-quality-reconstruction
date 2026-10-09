# Access to the frozen inputs

The public repository does not contain provider observations or authenticated
responses. Model retraining requires authorised copies of the original
snapshots. New downloads may contain revisions or a different station/device
inventory, so matching calendar dates alone does not guarantee reproduction.

## Milan

Request the three frozen tables from the corresponding author:

```text
data/processed/good_dataset/milano_city_good_dataset.csv
data/processed/cams_milano_city_hourly.csv
data/processed/openmeteo_milano_city_hourly.csv
```

Their SHA-256 hashes are in `config/inputs.json`, and column names are in
`config/input_schemas.json`. The observation table includes both regulatory
and Sense Square sources. This combined research table is not a publicly
downloadable ARPA dataset. It must not be described as reconstructible from
open sources alone.

The original `prepare_milan_city_case_study.py` documents municipal filtering
and authenticated Square acquisition. It expects upstream prepared research
tables; it does not independently download or recreate those tables. The
downloaded Square daily payloads are provenance evidence, not model targets.

Source portals:

- ARPA Lombardia: https://www.dati.lombardia.it/d/nicp-bhqi
- CAMS Europe regional reanalysis: https://ads.atmosphere.copernicus.eu/
- Open-Meteo historical weather: https://open-meteo.com/en/docs/historical-weather-api
- Sense Square Sensy: https://sensesquare.eu/prodotti/sensy/

## ARPA Campania

The original public resource URLs, retrieval date, and hashes are preserved in
`config/arpac_downloads.json`.

```sh
python scripts/download_campania.py
python scripts/verify_inputs.py --experiment campania
python scripts/reproduce.py --experiment campania
```

Required files:

```text
data/raw/arpac_salerno/arpac_station_metadata.csv
data/raw/arpac_salerno/arpac_rqa_2026_05.csv
data/raw/arpac_salerno/arpac_rqa_2026_06.csv
data/raw/arpac_salerno/arpac_rqa_2026_07.csv
data/raw/arpac_salerno/arpac_rqa_2026_08.csv
```

The August file is the original 14 August month-to-date snapshot. It supports
campus contextual screening; regulatory replication uses 1 May–29 July.
If a provider resource has changed, the downloader retains it as `.changed`
and fails verification instead of replacing the frozen input silently. Request
the archived snapshot from the author when this occurs. Source metadata may
contain an unquoted address comma; the analysis preserves its original repair.

## Campus

Request the archived `data/raw/salerno_campus/` JSON cache and the ARPA Campania
files above. The cache comprises 341 hashed acquisition artifacts. Public
input metadata hashes the relative cache paths to avoid disclosing individual
device IDs. The verifier matches paths and file contents against those hashes.

The reproduction runner copies this cache into a temporary ignored directory
before offline evaluation; the original cache and manifest remain untouched.
The campus module's optional acquisition mode can read either
`SENSE_SQUARE_APIKEY` or `SENSESQUARE_APIKEY` from the environment. Keys must
never be stored in repository files. A fresh acquisition is not a substitute
for the original cache, since inventory and historical API responses may change.

## Descriptive maps

After the analyses, the map code expects:

```text
reports/daily_spatial_validation/daily_analysis_table.csv
reports/daily_spatial_validation/test_predictions.csv
reports/salerno_campus/daily_observations.csv
reports/salerno_campus/device_inventory.csv
data/raw/milano_city/milano_city_squares.geojson
data/raw/fisciano_squares.geojson
reports/spatial_expansion_maps/plotted_observations.csv
```

The first four files are private outputs produced during retraining. The
original geometry and plotted-observation snapshots must be obtained with
appropriate permissions. Run `python spatial_expansion_maps.py` to produce
the descriptive map outputs, including `plotted_observations.csv`, then
`python scripts/render_study_area_split.py` for the publication layout.
The split renderer verifies that its selected dates and values match the
descriptive map output before drawing.
