# Verification record

Verification date: 9 October 2026. Machine-readable refit findings are in
[`config/verification.json`](../config/verification.json).

## Public checks

- The public aggregate verifier checks 34 manuscript claim rows and SHA-256
  hashes of the repository artifacts. This verifies the supplied archive;
  it does not independently refit models.
- All 16 methodological and integrity tests pass in the research environment
  and in a separate environment installed from the pinned requirements.
- Dependency consistency checks pass. GitHub Actions checks public artifacts
  and synthetic tests on Ubuntu (Python 3.12.13) and Windows (Python 3.12.10).
  Continuous integration does not have the restricted observations.
- The Campania figure regenerated from public aggregates is byte-identical
  to the supplied publication PDF in the local environment. Vector text and
  minimum font-size checks pass.

## Refitting original observations

The preserved analysis modules are byte-identical to the supplied research
code. Before fitting, the runner verifies all input hashes. No API request is
made for the campus offline reconstruction. The author-held input archive
is unchanged; the campus cache is copied to a temporary working directory.

| Experiment | Finding |
| --- | --- |
| Milan | Metrics, bootstrap intervals, and paired intervals match: three files |
| Campania | Metrics, paired comparisons, and both sensitivity files match: four files |
| Campus | Six baseline metric rows and both exact sign-flip rows match; Random Forest metrics and bootstrap intervals differ |

Comparisons use absolute tolerance `1e-8` and relative tolerance `1e-10`.
The campus discrepancy exceeds these tolerances. A full `--experiment all`
run returns a failure, as required by the strict checker.

Campus observations, device inventory, and split assignments match the
archived tables exactly. The differences include:

| Campus statistic | Archived | Refit |
| --- | ---: | ---: |
| Random Forest PM10 MAE | 2.1433984520 | 2.1512769064 |
| Random Forest PM2.5 MAE | 1.8725866690 | 1.8743286254 |
| Median minus IDW PM10 interval | -0.53688 to 0.05553 | -0.50098 to 0.06846 |

The archived Python version is 3.12.13; the local runtime now reports
3.12.14. Recorded versions of NumPy, pandas, scikit-learn, and Matplotlib
agree. The cause of the discrepancy is not established. The refit still
favours the network median and its paired intervals include zero, but this
does not count as exact reproduction of the archived estimates.

## Remaining boundaries

Milan and campus observations require authorised access. The additional
12-configuration Random Forest tuning sensitivity has not been located
in the supplied analysis project or its archives. It is not independently
reproduced by this release. No archival DOI has been registered.
