# Public UrbanNav reconstruction - NOT a strict reproduction

These tables were rebuilt from the official public UrbanNav u-blox F9P NMEA
stream and raw ground truth by `build_urbannav_features.m`.

## Why these are not drop-in strict inputs

`soft_fault_prob` is absent. The prior presentation does not define its
residual-to-probability estimator, so it cannot be reconstructed. The column
`soft_fault_prob_surrogate` is a documented logistic function of `PR_RMS`
provided only so the training path can be exercised. Renaming it to
`soft_fault_prob` would make `validate_data_contract` accept fabricated
supervision, which is exactly what the strict gate exists to prevent.

`run_all_real` reads `data/real/` and will keep refusing to start until a
genuine soft label is available.

## Choices not specified by the prior study

| Item | Value used | Note |
|---|---|---|
| feature preset | `literal` | see below |
| numSV source | `gns` | GGA truncates the multi-GNSS count |
| low_elev_ratio threshold | 15 deg | not stated in the slides |
| Fault_SVID_count residual threshold | 10 m | RAIM threshold not stated |
| CNO_gap definition | `maxmin` | not stated in the slides |
| PR_RMS definition | `rms` | not stated in the slides |
| Fault_SVID_count mode | `count` | a raw count scales with tracked SV number |
| multi-signal residual reduction | `primary` | one residual per satellite |
| hard-label error threshold | 3 m | stated in the slides |
| GT/NMEA max time gap | 0.6 s | 1 Hz alignment tolerance |

Changing any of these changes the dataset. Restate them in any write-up.

## Feature presets

`featurePreset="literal"` (default) reads each feature name the most direct way.
`featurePreset="robust"` replaces the outlier- and scale-sensitive definitions:
`CNO_gap` p90-p10 instead of max-min, `PR_RMS` log-RMS, `Fault_SVID_count` as a
ratio rather than a raw count, low-elevation threshold 10 deg, residual threshold
30 m, numSV from GGA.
