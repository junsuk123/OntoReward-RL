# Repository execution contract

The primary research configuration is
`Ontology_RGAT_UAV_RL_ISAAC_PX4/config/experiments/two_axis_reference_v28_active.yaml`.
The v1 config remains an explicit historical experiment; Isaac/PX4 remains
`isaac-legacy`, not an implementation of the reference causal contract.

- The dimension-generic contract lives in `python/ontology_rgat/landing/`, and
  both routes instantiate it. `terminal.py` is the ONE terminal table,
  discount horizon and curriculum ramp; `plane_graph.py` is the ONE nine-node,
  twelve-channel plane, built once per horizontal axis; `packet.py` is the
  field spec; `observation.py` is the one normalize/decode pair. Do not restate
  a terminal magnitude, a packet scale or a context row anywhere else -- four
  copies of the first and two of the second are what let 2D and 3D drift apart.
  Cross-dimension invariants belong in `tests/test_dimension_parity.py`, which
  loads both routes: a per-route test cannot see the drift by construction.
- 2D and 3D must optimize the SAME objective, or the dimension study measures
  nothing. Both now carry upstream's `SUCCESS 25 / SAFE_ABORT -15 /
  TASK_TIMEOUT -12 / UNSAFE -40` with `discount_tau 70` and curriculum start
  -20. The 2026-10-05 spatial revision to -30/-40/-50 with tau 350 is RETIRED:
  measured over 60 seeds with only the table swapped, an attempting controller
  returned +20.29 against a holding arm's -24.45 under the new table and +19.14
  against -4.26 under this one, so attempting wins under both. The
  "landing break-even p > 53.9 %" derivation that justified the change compared
  discounted terminal values alone and omitted the dense readiness/potential
  terms. Test the reward by measuring arms, never by deriving a break-even.
- The spatial task is NOT infeasible. The recorded 23.3 % "oracle ceiling" was
  one weak hand-tuned controller (P-only gain 0.65, no derivative term, descent
  gated at 0.12 m, horizon 45). On `spatial-causal-rgat/5` over 200 nominal
  seeds a PD on the estimator reaches 86.0 % and a truth-fed one 98.0 %. An
  oracle figure is a ceiling only if its gains were actually swept, and it must
  always be quoted WITH its contract: the same controller scores 86.7 % on v5
  and 33.3 % on v10.
- The active spatial contract is `spatial-reference/1` (`REFERENCE_SCHEMA`),
  derived from the 2D reference rather than chosen from the ladder: 41 fields
  (12 per-axis x 2 + 14 shared + 3 declared transport-delay extras), two
  reference planes, and v10's capability set. It reproduces v10 exactly on 60
  seeds. `spatial-causal-rgat/3..10` are frozen historical contracts; v5 in
  particular is NOT a 3D version of the reference -- its graph fills the nine
  rows positionally, it lacks predicted bearing/margin and acceleration
  uncertainty, and its action is a velocity setpoint carrying a hidden
  integrated velocity reference instead of applied net acceleration.
- Record a behavioural snapshot before and after any contract-touching change:
  `python tools/contract_snapshot.py --out before.json`, then `--compare`. The
  2D reference port must come back byte-identical.
- Preserve the two-axis reference oracle: longitudinal and vertical net
  acceleration. The user explicitly selected true spatial `(ax, ay, az)` for
  the new Isaac integration on 2026-10-04. That separate contract is world ENU,
  with derived roll/pitch and held yaw, never an independently learned tilt.
  The separate `run.sh spatial` route now has a causal spatial learner,
  local PPO training and opt-in actual Isaac/PX4 training/evaluation. It remains an integration
  profile: learned landing performance and reference parity are not validated.
  The current spatial schema is `spatial-causal-rgat/5` (decaying target
  acceleration and measured-own-velocity XY abort braking). Versions 3/4
  remain explicit historical contracts. v4 weight transfer is opt-in PPO
  initialization only; it never transfers nominal checkpoint eligibility.
  Opt-in candidates 6/7/8 are not accepted replacements: v6 uses direct PX4
  acceleration-only OFFBOARD, v7 adds the reference ABG tracker/uncertainty,
  and v8 additionally samples Isaac's seeded external forces/torques and
  handover velocity/rate perturbations in the local reduced-order plant.
  The user explicitly chose to KEEP Isaac disturbances and match local
  training on 2026-10-04; do not silently disable them to obtain a landing.
  Opt-in v9 adds capture-time-aligned causal optical updates and restores
  reference channel meanings using two signed ENU projections, EACH 9x12.
  The typed encoder is shared between planes; four-group readouts are
  concatenated. Both planes use joint 3D descent evidence. Do not flatten
  away the sign of lateral errors or silently load older checkpoints.
  Opt-in v10 additionally restores uncertainty trust, a recovery latch and
  response-delay stopping/terminal-corridor logic. Its persistent-disturbance
  abort is an explicitly different own-EKF position hold overriding ALL axes;
  v9's upward actor action must not leak into that backup. Keep the 20-degree
  command/21-degree hard attitude limit, unchanged touchdown limits, and the
  captured-measurement view reward. Use the episode-owned supervisor, never
  the historical stateless v3-v9 functions. No v9 weights/eligibility aliasing.
  The older `config/control/spatial_acceleration_v1.yaml` is boundary-test-only.
- Keep `physical_state`, measurements, causal estimator memory, decision
  context, safety status, and episode status separate.
- All three primary arms must use the same causal packet registry hash,
  environment, reward, supervisor, timing, scenarios, and sensor events.
- Simulator truth is allowed only in the isolated training reward and
  evaluator. It must not enter an actor, critic, graph, estimator, or safety
  supervisor input.
- Keep legacy three-channel, PBRS, FOV-reward, adaptive-reward, behavior
  cloning, and frozen-selective experiments under their existing IDs. Never
  alias their checkpoints to `two-axis-context-rgat-v1`.
- Do not run long PPO training, real-vehicle commands, or push changes without
  explicit user approval. Bounded local tests and smoke rollouts are allowed.
- Keep the V2.5 environment invariants, each guarded by a test: an authorized
  touchdown must be reachable (the terminal-descent corridor replaces the
  abort-hold margin below it, never deletes it); terminal rewards must order
  `SUCCESS > TASK_TIMEOUT > SAFE_ABORT > unsafe`; `ax_max` must exceed the
  sampler's largest pad acceleration; the hard envelope is measured against the
  pad, not the world origin; a prolonged visual loss starts a bounded recovery,
  not a terminal.
- Curriculum difficulty is training-only and `difficulty = 1.0` must stay
  bit-for-bit identical to the nominal contract. Validation and test always run
  at 1.0. Only checkpoints reaching difficulty 1.0 are selectable.
  Optional loss-timeout curriculum (reference 12 -> 3 s) is local-training
  only; it must not change nominal/test/Isaac safety thresholds. Complete-
  episode batches report actual steps, not their unused decision-budget flag.
- The PN comparator in `two_axis.comparator` verifies physical feasibility
  only. It must never supervise PPO, seed a replay buffer, or enter a reward.
- Preserve the external three-field PX4 gateway only through
  `two_axis.adapter.px4_gateway_command`; its third field is derived from the
  two-axis request and is not learned. This applies to the LEGACY planar route.
  The spatial route uses `SpatialCommand` and `spatial_velocity_action`, not
  that adapter; do not reinterpret planar three-field actions as spatial.
  Schemas 6-10 instead use distinct `spatial_acceleration_action` with NaN
  position/velocity PX4 setpoints and no hidden integrated velocity target.
- That adapter produces normalized legacy-controller input, not gateway wire
  bytes. Require the actual controller envelope and reject unreachable commands.
- v2.8 allows its explicitly versioned common potential reward with terminal
  potential zero. Never conflate it with historical learned/PBRS reward arms.
- Preserve raw semantic bypass, 9x12 graph, 4-group residual readout, causal
  masked pretraining, staged adaptation, and actual relational-output audits.
  Do not claim a representation-only comparison when these factors differ.
- Upstream 83f10d6 has an active, small relational residual; do not repeat the
  outdated final-Wg=0 claim. Guarded activation must preserve raw actor/value
  and log_std, use real train-only PPO plus validation-only scale selection,
  record extra steps, and retain the anchor if no scale passes. Nonzero weights
  or residuals alone are not evidence of superiority.
- Bare `run.sh` is help. Status/smoke must never stop a live stack. Takeover is
  opt-in; collateral peer restarts must consume the retry budget.
- Entry geometry alone must never authorize policy handover: disarmed setup
  support can hold the same pose. Require armed PX4 and spatial OFFBOARD before
  collecting a transition. Only a fresh same-episode terminal contact can
  explain a disarmed terminal; setup/refusal is infrastructure failure, not RL.
- Spatial pre-policy EntryResetError may consume an explicit run-wide owned
  restart budget. Preserve attempt logs and retry the same seed; never restart
  adopted stacks or use this to discard policy-time failures. Setup support
  may span the first arm-to-OFFBOARD gap, never a post-policy AUTO.LAND.
- Direct-acceleration cleanup must replace the last acceleration with an own-EKF
  braking position hold BEFORE requesting LAND. Release the temporary hold
  when PX4 leaves OFFBOARD so it cannot overwrite AUTO.LAND's trajectory.
  This is unscored post-policy cleanup, never teacher control or a successful
  task landing. Preserve terminal verdict and failed cleanup as separate events.
- Spatial SITL continuous setpoints must be paced on the received physics clock,
  including entry and cleanup. A wall-time 50 Hz stream flooded slow lockstep
  at roughly 296 trajectory messages per simulated second in an actual probe.
  Keep wall-time maintenance/deadman checks, hardware and legacy behavior;
  never invent clock progress or replay missed heartbeat ticks. Mode prestream
  counts must count emitted setpoints, not skipped wall-timer callbacks.
- No MATLAB or Isaac performance-parity claim from Python unit tests, parameter
  counts, PN feasibility or synthetic reward audits. Record remaining gaps in
  `docs/refactor/REFERENCE_V28_AUDIT_KO.md`.
  Actual integration acceptance also requires a proposed-policy landing and
  measured nonzero relation output, not just a baseline landing. A small
  integration pass is still not evidence of robustness or superiority.
  Acceptance /3 additionally requires confirmed cleanup for every evaluated
  episode and a separately verified terminal hold snapshot for SAFE_ABORT.
  The timeout label alone is not evidence of a stable hold; the snapshot
  itself is not a sustained-stability proof.
