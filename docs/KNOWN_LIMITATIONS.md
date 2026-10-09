# Reproducibility boundaries

1. **Restricted inputs.** Public aggregates can be checked without credentials.
   Refitting Milan and campus models requires author-held, authorised snapshots.
   This repository therefore does not claim complete open-data reproducibility.
2. **RF tuning supplement.** The current manuscript additionally reports a
   12-configuration tuning sensitivity (MAE 4.678; paired difference against
   IDW 0.596, interval 0.222–0.975). Its original script, grid values, and output
   archive have not been located in the supplied AI_Spatial_Expansion project.
   These numbers are not represented as independently reproduced by this
   release. The missing supplement should be deposited after recovery; a new
   guessed grid would not reproduce the original experiment.
3. **Historical downloads.** Current provider resources may differ from the
   recorded August 2026 snapshots. SHA-256 checks reject changes. Byte-identical
   inputs must be obtained from the archived research package when necessary.
4. **Figures.** Publication figures are supplied as frozen artifacts. The
   Campania plot can be regenerated from public station-level aggregates.
   Milan/campus maps require restricted observations and original geometries.
5. **Platform effects.** Numerical comparisons allow floating-point tolerance;
   exact figure bytes can depend on fonts and rendering libraries. Hash
   verification of published files is distinct from numerical reproduction.
6. **Archive identifier.** No Zenodo DOI has been minted by this repository.
   A public GitHub URL is available, but an archival DOI requires a separate
   deposition step by the authors.
