# Experiment settings and interpretation

## Milan

- Municipality of Milan, 1 May–29 July 2022; four reference stations,
  including three measuring PM2.5.
- Regulatory daily means define targets; hourly covariates use Europe/Rome
  days. Low-cost daily medians require 18 distinct hourly timestamps.
- Rolling origin: 35 initial training days, seven validation days, 14 test
  days, advancing 14 days. The final test windows cover 42 unique days.
- Target stations are excluded from fitting, validation-based selection,
  imputation, and neighbour-derived features.
- Fixed RF: 500 trees, minimum leaf size five, 80% features per split,
  seed 42. Ridge alpha is 10. Feature groups are regional, ARPA spatial,
  regional plus ARPA, and full including low-cost summaries.
- Fixed regulatory IDW: exponent 1, 100 m floor, at most eight sources;
  available reference neighbours are fewer in held-out municipal folds.
- Bootstrap: 2,000 replications, seven-day moving blocks, pooled record
  weighting. Exact paired sign flips use station-level mean effects.

The preserved code also emits optional validation-selected model/baseline rows.
These rows are not the manuscript's primary fixed-model comparison. Their
presence does not represent the separate 12-configuration RF tuning supplement.

## Campania

- IT1508 main panel: IT2210A, IT2213A, IT1504A, IT2214A, IT2227A, IT0936A;
  1 May–29 July 2026. Both pollutants use six stations.
- Printed local calendar dates are retained; the inconsistent +01:00 summer
  offset is not used to shift records. Daily means require 18 distinct hours.
- Same rolling schedule, 36 folds, 485 held-out station–pollutant–days.
- Fixed learned models use regulatory spatial summaries, coordinates, and
  calendar terms, without Milan's low-cost, CAMS, or weather features.
- IDW uses exponent 1 and a 100 m floor; exponents 2 and 3 are sensitivities.
- Avellino dependence sensitivity merges the two nearby station effects.
  A separate seven-station sensitivity adds Battipaglia IT2268A.

## University of Salerno campus

- Twenty fixed devices; platform daily values from 1 May–13 August 2026.
- Training: 1 May–23 July. Evaluation: 24 July–13 August, 20 observed test
  dates and 741 device–pollutant cases.
- Each device is excluded from fitting, tuning, selection, and peer features.
- Fixed RF has 150 trees, as defined by the campus code; the 500-tree Milan
  configuration is not applied to this experiment.
- Campus IDW uses exponent 2 and a 10 m floor. Network median, nearest device,
  and IDW use contemporaneous peers. This is spatial gap filling, not forecasting.
- Hierarchical bootstrap resamples devices and seven-day date blocks.
  Exact sign flips use mean effects per device.
- Shared calibration can produce common-mode bias. No regulatory monitor is
  co-located, so internal consistency does not establish regulatory accuracy.

## Differences and intervals

Regulatory paired differences are competing model minus IDW absolute errors.
Campus archives store network median minus IDW; the manuscript's comparison
table reverses this sign, so its positive values favour the median. Negating
an interval also reverses its lower and upper bounds; the export script
performs this transformation explicitly.

Few independent sites limit inference even with many daily records. Milan's
four sites have minimum exact two-sided p = 0.125; Campania's six sites can
attain 0.03125, but its observed RF–IDW p is 0.1875. The extra data families
and model complexity do not outperform regulatory IDW under these tested
conditions. Upstream CAMS assimilation was not audited; exclusion is guaranteed
within the local analysis pipeline only.
