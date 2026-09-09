# Processed UrbanNav-HK Data Contract

Required columns per synchronized epoch:

- `numSV`
- `hDOP`
- `vDOP`
- `hAcc`
- `vAcc`
- `gSpeed`
- `CNO_mean`
- `CNO_std`
- `CNO_gap`
- `low_elev_ratio`
- `PR_RMS`
- `Fault_SVID_count`
- `hard_label` (0/1), or `pos_error_2d` from which `>=3 m` is generated
- `soft_fault_prob` in `[0,1]`

Recommended provenance columns:

- `timestamp`
- `pos_error_2d`
- `residual`
- `scenario`
- receiver/source identifiers

Strict mode refuses to create a replacement `soft_fault_prob` because the prior presentation does not provide the exact residual-to-probability mapping.

## Rebuilding from raw public UrbanNav data

`build_urbannav_features("all")` reconstructs the twelve features and
`hard_label` from the official public UrbanNav u-blox F9P NMEA stream plus the
raw ground-truth trajectory. Output goes to `data/urbannav_public/`.

That covers thirteen of the fourteen required columns. It does not produce
`soft_fault_prob`, so those tables are not strict inputs. The surrogate it writes is
deliberately named `soft_fault_prob_surrogate`; renaming it would defeat the gate
above. `run_all_real("strict")` refuses them outright; a bare `run_all_real` instead
switches to the explicitly labelled non-strict run described in README_KO.md, writing
to `outputs/NONSTRICT_SURROGATE_<stamp>/` alongside a `NOT_A_REPRODUCTION.md`.

Definitions the prior slides leave unspecified (low-elevation threshold, RAIM
residual threshold, `CNO_gap` and `PR_RMS` reductions, multi-signal residual
handling) are explicit options recorded in
`data/urbannav_public/reconstruction_manifest.json` and
`PUBLIC_RECONSTRUCTION_NOTICE.md`. Restate them in any write-up.
