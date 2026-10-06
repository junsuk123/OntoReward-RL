# Stopped 2026-10-06 10:30 KST, at the user's instruction

The bare-`run.sh` pipeline that owned this root (driver pid 477819, started
09:28) was stopped during its first cell, `baseline`, at iterations 90-127 of
200. Its six runs had produced 0 SUCCESS in ~7,800 training episodes, which
is the already-recorded failure of the shipped settings
(results/spatial_finetune_20261006) and needed no further budget.

The remaining cells (`masking`, `anneal`, `state_sigma`, `combined`) were not
started. All of them predate `--enforce-target-kl`; see AGENTS.md ("Pinning
sigma low is necessary and NOT sufficient") and
results/full_pipeline_20261006_lowsigma_kl for the cells that land.

`baseline/runs/*/progress.json` are partial and remain readable. No
summary.json was written, so a resumed driver on this root would re-run the
cell from scratch. `clone_plain` and `clone_state` are complete and were
copied into the lowsigma roots as their starting points.
