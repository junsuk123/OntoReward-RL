# Experiment status

## Primary

`two_axis_reference_v28_active.yaml` is the current theory-alignment profile.
It extends `two_axis_reference_v28.yaml` (first port) and
the preserved `two_axis_context_rgat_comparison.yaml` (historical v1) through
the executable config loader. Run `./run.sh reference-smoke` from repository root.
This is an in-process environment, NOT an Isaac backend. Consult
`docs/refactor/REFERENCE_V28_AUDIT_KO.md` for remaining parity gaps.

## Legacy and reproduction

The other YAML files declare separate experiments. They
remain versioned for provenance, figures, live-stack reproduction, or explicit
ablation. In particular, these mechanisms are not part of the primary method:

- a learned tilt/pitch action (the old planar action had three channels);
- additive FOV-risk reward or ontology-derived reward weights;
- historical learned/ontology PBRS; v2.8's explicitly versioned common fixed potential is allowed;
- frozen-base selective R-GAT and behavior-cloning warm starts;
- the Shin five-term/active-perception reward comparisons.

Legacy checkpoints are incompatible with `two-axis-reference-v28-active-port2` and
its v3-v28 smooth-normalization registry. Even unchanged feature counts do not
make a checkpoint compatible. Graph pretraining/adaptation are declared extra
comparison factors, not a representation-only claim.

`../control/spatial_acceleration_v1.yaml` records the user's true spatial ENU control
choice and remains a historical control-boundary fixture. The legacy loader
rejects it instead of silently running a planar actor. The separate
`./run.sh spatial` route now supplies a causal 35-field, 3-action backend;
`--training-backend isaac` explicitly collects actual PX4 training/validation
rollouts, while its default is local spatial training. Both are integration
profiles, not evidence of 2D numerical/performance parity. The wire route is SITL-only.
Explicit `--contract-version 8` instead uses the 43-field ABG/direct-acceleration
candidate and retains Isaac disturbances in both actual flight and the local
reduced-order training plant. Matched seeded draws do not imply an identical
PX4/visual plant. It remains an unaccepted candidate; do not relabel a v5 policy.

Opt-in version 9 uses `../spatial-isaac-system-v9.yaml`: the same disturbances,
47-field capture-aligned packet, two signed 9x12 reference context projections
with a shared typed encoder, and one seeded analytic CV-CA-CV scenario shared
by local/actual backends. Its local sensor has a declared 75 ms delivery delay.
The old `segmented_cruise_slow` trajectory remains legacy; it is NOT the new
CV-CA-CV scenario. Old checkpoints are rejected by exact signature checks.
This candidate is under retraining/flight validation, not an accepted default.

Version 10 reuses the v9 Isaac profile, NOT its checkpoint signature. It adds
episode-owned reference uncertainty trust, abort latch, delay/force-aware
braking and a bounded terminal corridor. Persistent disturbances use a causal
own-position abort hold overriding all axes. The measured optical bearing
feeds view reward; touchdown limits and disturbances are retained. Acceptance
`/3` separately checks confirmed stops and terminal abort-hold snapshots.

The active reference's graph-only repair adds at most 25 × 2048 train steps
when a selected nominal checkpoint is inactive. A validation performance and
residual guard may reject all scales; activity is not forced with injected
weights. Extra costs and selection factors are recorded in the run summary.
