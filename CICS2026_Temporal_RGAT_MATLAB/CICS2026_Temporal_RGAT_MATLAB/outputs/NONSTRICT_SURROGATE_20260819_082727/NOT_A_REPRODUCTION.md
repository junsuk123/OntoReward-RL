# Not a reproduction of the prior study

Every metric in this directory was produced with `cics_config("nonstrict")`.

## What is real

- The 12 features and `hard_label`, rebuilt from the official public UrbanNav
  u-blox F9P NMEA stream and raw ground truth by `build_urbannav_features`.
- The train Medium+Harsh / test Deep Urban split, window length and 3 m hard
  fault threshold.
- The architectures, ablations, forecast horizons and statistical tests.

## What is not

The weak supervision. `soft_fault_prob` does not exist in these tables, so the
Dual-EDL weak head was trained on `soft_fault_prob_surrogate`, a logistic
function of `PR_RMS`. The prior presentation never defines its
residual-to-probability mapping, so the real one could not be used.

## Consequences

- Do NOT compare any number here against the reported F1=0.950 baseline.
- `baselineReproductionPass` in the manifest is not interpretable in this mode.
- Comparisons BETWEEN models in this directory are meaningful: every model saw
  identical data, split and supervision. That makes it valid for architecture
  and ablation conclusions, and invalid for reproduction claims.

For a paper-comparable run, supply the prior study's processed tables with a
genuine `soft_fault_prob` in `data/real/` and use `run_all_real("strict")`.
