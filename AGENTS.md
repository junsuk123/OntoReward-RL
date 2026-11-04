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
  always be quoted WITH its contract. Swept properly, BOTH contracts are
  solvable: v5 reaches 86 %, and a 96-cell sweep on the active contract -- with
  disturbances, the 75 ms transport delay and the reference supervisor all
  enabled -- reaches **95.8 %** (23/24, zero unsafe or unauthorized contacts,
  four distinct gain sets on the plateau, median 58.3 % across cells). The
  earlier 33.3 % for that contract was a v5-tuned controller run on the
  direct-acceleration plant. Before quoting any oracle number, check whether
  the best cell sits on the grid BOUNDARY; each time it did here the real
  ceiling was far higher (23 -> 86 on v5, 33 -> 96 on the active contract).
- A ceiling measured with a tuned controller says the TASK is solvable. It says
  nothing about whether the policy's OWN action distribution can ever reach a
  landing, which is the quantity PPO depends on, and on 2026-10-05 the two came
  apart completely: 838 training episodes at the EASIEST curriculum rung
  produced zero SUCCESS on a rung where the constant action `[0, 0, -0.12]`
  lands 40/40 for a mean return of +25.3 and the swept PD lands 39/40, while
  the sampled policy returns -24.2. Run
  `python tools/audit_exploration_reachability.py` before blaming the task, the
  reward table or the budget again.
- Both supervisors rewrite ONLY negative vertical commands while descent is
  inhibited (`spatial/safety.py`, `two_axis/safety.py:180`): a descending
  vehicle is braked with up to full upward authority, an ascending one passes
  through untouched, and neither direct-acceleration plant has a restoring
  force on altitude, so zero-mean exploration integrates upward. Measured on a
  fresh 3D policy at difficulty 0.0, requested -0.065 m/s^2 comes out as
  applied +0.082, the median climb is +19.71 m and 27/40 episodes die on the
  20 m ceiling; a symmetric vertical hold with nothing else changed gives
  +0.70 m and 0/40. On the PRIMARY 2D contract the same law removes 92 % of the
  commanded descent (requested -0.169 -> applied -0.013). The swept oracle
  never meets it -- it sits inside the terminal-descent corridor on 64 % of its
  steps and has 2.8 % of its actions overridden -- so no ceiling measurement
  can see this. Changed on the 3D side 2026-10-05 by `spatial-reference/2`; the
  planar law at `two_axis/safety.py:180` is DELIBERATELY unchanged, because the
  2D route is a port of upstream MATLAB and `tools/contract_snapshot.py` must
  return it byte-identical. Quantify it there before touching it.
  It is a SAFETY fix, not a learning fix, and the distinction was measured after
  it shipped: on its own it reaches held-out return -20.2 against the baseline's
  -22.4, and added to complete-episode batches it COSTS performance (-10.5 ->
  -16.1, TASK_TIMEOUT 0.833 -> 0.250). Do not present it as the reason PPO
  improves; that is the batch size, above.
- **Check `relational_health` in the training summary before reading any
  three-arm result.** `ppo_ontology_rgat` differs from `ppo_semantic_flat` by
  exactly one term, `relational_delta`, whose readout is zero-initialised and
  which `set_adaptation` freezes for the first
  `ontology.adaptation_warmup_fraction` of the PLANNED budget. While frozen the
  two arms are the same function bit for bit. Comparing per-iteration
  `actor_loss` and `entropy` to twelve decimals across the shipped runs on
  2026-10-05: 8607/8607 iterations identical in `spatial_long_nominal` (it
  declared 20000 iterations, so warmup landed at 18000 and the run ended at
  8607 -- the relational path was never trainable for one update), and
  1800/2000 in both 2000-iteration runs. **Every three-arm spatial result
  recorded before 2026-10-05 measures two arms and a copy, and says nothing
  about the proposed representation.** The warmup is a fraction of the planned
  budget, not of what the run completes, so declaring a large `--iterations`
  and stopping early silences the arm entirely. Sweep it with
  `--adaptation-warmup-fraction`; guard is
  `tests/test_proposed_arm_is_distinguishable.py`.
- The spatial runs batch ONE complete episode per PPO update
  (`--episodes-per-iteration`, ~110-130 decisions) with rollout-level advantage
  normalisation, so every advantage is centred inside its own episode and no
  cross-episode comparison survives. The 2D route uses 2048 decisions. Nine
  controlled 2400-episode cells ranked the three candidate repairs and **this is
  the one that moves PPO**: every cell batching 12 complete episodes converges
  to TASK_TIMEOUT (0.833 / 0.542 / 0.250), i.e. the policy finally follows the
  terminal ordering, while every one-episode cell stays trapped in SAFE_ABORT at
  0.75-0.96. Best cell: batches alone, held-out unsafe 0.000, return -10.5
  against the baseline's -22.4, and the only cell to produce SUCCESS during
  training. Budget is NOT the constraint -- a 7200-episode cell with one episode
  per update produced exactly one TASK_TIMEOUT in 7200 episodes. Use
  `--episodes-per-iteration 12` or larger for any spatial training that is meant
  to learn.
- **The sampled policy, not the reward or the architecture, is why nothing
  lands.** Measured 2026-10-06 on behaviour-cloned checkpoints at difficulty
  0.0, the rung training opens on: evaluated deterministically they land
  79-96 %, and sampled at the shipped `initial_log_std=-1.1` (sigma 0.333)
  they land **0.0 %**. PPO rollouts are sampled, so no batch contains a
  success and nothing in the gradient points at a landing; fine-tuning from
  those clones for 200 iterations drove all six runs to 0.0 % held-out with 0
  SUCCESS in 14 400 episodes. Landing recovers monotonically as sigma falls
  (0.082 -> 33.3 %, 0.040 -> 66.7 %, 0.005 -> 100 %, which beats the
  deterministic policy), so `minimum_log_std=-2.5` (sigma 0.082) is itself a
  floor worth crossing. The mechanism is physical, not statistical, and
  `tools/audit_sigma_ladder.py` measures it: at sigma 0.333 the requested
  lateral acceleration jumps 1.1 m/s^2 per 0.1 s decision (0.16 for the
  mean), the derived attitude follows, the body rate sits inside its 10 deg/s
  limit on 11 % of steps (85 % for the mean), and the terminal-descent
  corridor -- which arms only while `settled` -- opens on 0.7 % of steps
  (62 %). At the -2.5 floor it is 45 % / 21 % and still 0/12 landings; at
  0.041 the mean's 50 % is back. A batch that cannot physically contain a
  landing teaches nothing about the reward, whatever the table says. Before
  blaming reward, batch or representation again,
  evaluate BOTH deterministically and sampled. Controls:
  `--final-log-std`, `--log-std-anneal-fraction`, `--state-dependent-log-std`.
  Reading `log_std.exp()` back is NOT reading the policy -- `forward` clamps
  to `minimum_log_std`, which silently invalidated a first pass at this.
- **Pinning sigma low is necessary and NOT sufficient: the shipped PPO never
  enforced `target_kl`.** `PPOTrainer.update` computes `approx_kl` from the
  ratio BEFORE each minibatch's own Adam step, so the early stop fires one
  minibatch late and the first actor step of every iteration is unbounded.
  That step's size in action space is set by `actor_lr` (Adam's first steps
  are ~lr per weight, coherently, and the optimiser is reset at fine-tune
  start), not by sigma, so KL per step grows as (dmu/sigma)^2. Measured
  2026-10-06 in `results/full_pipeline_20261006_lowsigma_unconstrained`
  (clones, `--final-log-std -3.5 --log-std-anneal-fraction 0.02`, sigma 0.030
  from iteration 5): per-iteration KL 0.5-3.4 against the 0.02 target,
  `minibatches = 2` every iteration, and both graph arms went from a 62.5 %
  landing clone to 12/12 SAFE_ABORT at ~118 steps by iteration 4-5;
  `ppo_vector_canonical`, with a smaller input and lever, survived and landed
  14 % of its training episodes. At the shipped sigma 0.333 the same step is
  KL ~0.05 and looks harmless, which is why no earlier run showed it.
  `--enforce-target-kl` (`PPOHyperparameters.enforce_target_kl`) re-measures
  the KL on the minibatch AFTER the step, reverts a step above
  1.5 * target_kl (weights and Adam moments), halves `actor_lr` only when the
  iteration accepted nothing, and grows it 1.5x (never above the configured
  value) after a clean iteration under target_kl / 2. Opt-in, because every
  recorded run predates it; guard is `tests/test_target_kl_is_enforced.py`.
  Pipeline cells: `low_sigma` (sigma alone, the control), `low_sigma_kl`,
  `low_sigma_kl_masked`; `combined` carries it. Read `rejected_steps`,
  `accepted_steps` and `actor_lr` in the history before trusting any
  small-sigma run. **Measured outcome, `low_sigma_kl`
  (results/full_pipeline_20261006_lowsigma_kl, 200 iterations x 12
  episodes from the clones, 48 held-out nominal seeds 4100-4147,
  deterministic, `tools/compare_spatial_arms.py`):** every arm lands after
  PPO for the first time. BC start -> after PPO, mean over the two training
  seeds: `ppo_ontology_rgat` 63.5 -> 75.0 % (seed 828: 72.9 -> 87.5 %),
  `ppo_semantic_flat` 54.2 -> 84.4 % (52.1 -> 93.8 %), `ppo_vector_canonical`
  11.5 -> 44.8 % (seed 829: 6.2 -> 77.1 %; seed 828 stayed at 12.5 % and
  converged to SAFE_ABORT 62.5 %, the terminal ordering's correct answer for
  a policy that crashed 36 % of its early rollouts). Training SUCCESS was
  1404-1845 of 2400 against the shipped settings' 0 of ~7800 at the same
  point. The relational path was ACTIVE in both proposed-arm runs. This is
  two training seeds with sd 9-32 points, and `ppo_semantic_flat` scored
  ABOVE the proposed arm; it is evidence that the training recipe works,
  not that the representation wins. Held-out unsafe after PPO is 6-31 %
  (rgat seed 829 rose 16.7 -> 31.2 %), so the safety side is not settled.
- PPO regresses on `raw_command`, the action the policy PROPOSED.
  `collect_rollout` already records `safety_flags["intervened"]` and no update
  reads it, so 38-44 % of 3D steps and 54 % of 2D steps are evidence about an
  action the supervisor replaced. Restricting the actor objective to executed
  steps cut unsafe outcomes from 0.29 to 0.04 over 2400 episodes and still
  landed nothing. A BINARY mask is the wrong shape once the supervisor touches
  one axis routinely: combined with a symmetric vertical hold it discards 92 %
  of the steps and the actor stops moving. Per-axis is the right shape.
- The arms CAN represent the solution, so do not attribute the training failure
  to the observation, the graph or the architecture. Behaviour cloning of the
  swept controller (120 episodes, loss in ACTION space) lands 38-71 % on
  held-out seeds for all three arms across two seeds, against 0 % for
  end-to-end PPO after 8600 episodes. DAgger rounds past the first COLLAPSE
  here and the training error rises 5-10x, because the teacher carries an
  integral term and its label on the learner's own trajectory is therefore
  history-dependent; use a memoryless teacher if you aggregate.
- Before citing any 2D number, read `algorithm_version` in its summary. Every
  substantive run in `results/two_axis_*` is `two-axis-context-rgat-v1`, the
  RETIRED v1 contract; the only reference-v28 run,
  `results/refactor_v28_smoke_20261004`, is a 2-iteration smoke with ZERO
  episodes. The 2D half of the dimension comparison has not been run.
  `two_axis/config.py: DEFAULT_CONFIG_PATH` also still points at the retired v1
  config, so a bare `load_config()` quietly returns -40/-30/-50 with tau 350
  instead of the reference table. Always pass the config path explicitly.
- The COMMON OBSERVATION O_t = {P_t, D_t, H_t} lives in
  `python/ontology_rgat/landing/common_observation.py` (2026-10-07,
  dimension-generic, instantiated for 3D). P_t: every registered board marker
  in a FIXED id slot (profile order, 17 on v11/v12), its four image corners in
  the detector's own order normalized by image size, a per-marker detection
  mask, the frame's capture stamp and `frame_valid` (frame <= 0.5 s old).
  D_t: own fused navigation state, 3D = ENU position, velocity, roll/pitch/yaw
  sin/cos, body rate (15). H_t: previous K=2 (P, D) records plus the last
  record with any detection; records are (P, D) only, never a previous O.
  No estimate, tracker std, predicted bearing, descent permission or abort
  flag enters it. `vector()` exposes ages, not stamps (687 values at M=17).
  Sources: the local backend projects the same board through the same mount
  and delay (corner noise 0.5 px from a SEPARATE RNG, an assumption, not a
  measurement; the existing stream is untouched and every contract snapshot
  is byte-identical), and the Isaac detector publishes
  `perception/marker_corners` (JSON per frame, also when empty), forwarded
  untouched by the gateway as `extra.marker_corners` into
  `Measurement.marker_corners_px`. Guard `tests/test_common_observation.py`
  (the Isaac detector's own PnP recovers the true pose from local corners
  with noise off). It is attached as `Observation.common` on rungs with a
  deployed board and is NOT yet read by any arm: switching the three arms to
  it needs the graph mapping, which the design leaves to the ontology stage.
  The Isaac publish path is compile-checked only, not yet flown.
- The MINIMAL-OBSERVATION contract (2026-10-07, user's redesign; design and all
  numbers in `docs/MINIMAL_OBSERVATION_ROS_PIPELINE_KO.md` section 11) is a
  SEPARATE contract, not a spatial-reference rung: observation
  `minimal-landing-obs/2` = ArUco pad position (ENU, held on a miss with its
  age) + EKF own position AND velocity (13 values), ontology
  `minimal-landing-ontology/4` (13 nodes, 6 relations, schema hash pinned in
  `tests/test_minimal_observation_pipeline.py`), pure code in
  `python/ontology_rgat/minimal/`, ROS nodes in `ros2_ws/src/ontology_rgat_landing`
  (interfaces `ontology_rgat_interfaces`). The supervisor reads the
  observation only and calls the SAME `pad_loss` classifier as the ontology
  (terminal occlusion vs lost; LOST is absorbing until re-detection; no
  TERMINAL without a known pad velocity). `TrackingBias` (integral of the
  track error) is REQUIRED: without it the integrating teacher's clones land
  0-4 % for every arm, with it 71-98 % held out; the teacher without
  integrals lands 0/24. Current ontology is `/5` (section 12 of the doc): the
  pad VELOCITY is estimated in an own-displacement frame (a relative-velocity
  slope lagged a manoeuvring vehicle by 0.4 m/s and caused every remaining
  TERMINAL-off-pad call), sink is capped at 0.24 m/s below the 0.40 m terminal
  entry, the hold permits a bounded view-recovery climb, the supervisor
  subtracts a causal disturbance estimate. Teacher 192/192 local; aggressive-
  gain stress grid 0 unsafe / 0 missed-pad in 1152; clones rgat 100 %, flat
  100 %, vector 56-60 % (own position removed from its input: locally own_z
  equalled height above the pad, a shortcut that does not exist in Isaac).
  Isaac 19/19 SUCCESS across /4 and /5, including 3/3 with the ROS chain as
  the actual control path (`tools/minimal_isaac_flight.py --control ros`).
  Episodes are few: an integration pass, not a superiority claim; no PPO yet.
  A recoverable PX4 link failsafe during the spatial entry hover is now an
  EntryResetError (owned restart, same seed) -- it once ended a whole run.
  Full pipeline: `tools/minimal_full_pipeline.py` (teacher -> BC -> PPO
  `minimal/ppo.py` -> 48 held-out -> opt-in Isaac). First run
  `results/minimal_pipeline_20261007`: graph arms are already at ceiling after
  BC (~100 %) and PPO moved them 0 to -6 points; flat degraded DURING training
  (best checkpoints at iterations 20-30) while R-GAT stayed stable; PPO DID
  improve the vector arm (sampled 48-58 -> 63-88 %). Isaac 8/8 incl. ROS-chain
  control; 27/27 for this contract. Three seeds: no superiority claim.
  Round 2 (5 seeds, 2026-10-08, doc section 14) adds evaluation-only STRESS
  scenarios (`minimal/stress.py`; never used to train, validate or select):
  under strong wind / poor vision / combined the FLAT arm beats R-GAT by
  13-17 points, beyond the seed spread -- the proposed relational arm is the
  LESS robust reader of the same graph here -- and PPO (nominal-only) lowers
  every graph arm's stress robustness; the vector arm is worst on fast_pad
  (14 %) and best on poor_vision (71 %, above the teacher's 58 %), cause
  unmeasured. Isaac 42/43 for this contract (one vector-arm timeout).
  Diagnosed (doc section 15, `tools/minimal_stress_diagnostics.py`): under
  stress the decision/loss nodes leave the nominal band (LandingInhibit /
  TargetLost ~20 % under poor vision) and R-GAT is 2.5-3.5x more sensitive
  than flat to exactly those nodes, so message passing AMPLIFIES the shift;
  the vector arm lacks pad velocity (fails fast pads) and is calmer than the
  teacher under poor vision, where the teacher itself is not optimal.
  Domain-randomized training (`--train-scenario dr`, validation/selection
  still nominal) confirms it: R-GAT gains +12 to +23 points under wind/poor
  vision and the gap to flat mostly closes; the vector arm loses 14-33 points;
  `combined` stays 10-14 % for all arms with 5-7.5 % unsafe. Isaac 50/51.
- The active spatial contract is `spatial-reference/12` (`REFERENCE_SCHEMA`,
  2026-10-07, profile v12, no legs). It is `/11` plus ONE law, the HANDOVER
  THRUST FLOOR: applied vertical acceleration >= 0 for the 2 s handover
  window (`handover_vertical_floor_m_s2`). /11's lateral cap did NOT stop the
  violations -- its flight (`results/full_pipeline_20261007_v11_5seeds`:
  held-out rgat 93.3 / flat 93.3 / vector 91.7 %, Isaac 19/30, rgat 9/10,
  flat 9/10, vector 1/10) had two, rgat 830 at 26.7 deg and flat 829 at
  21.7 deg, both inside the 2.0 cap with the tilt climbing 2 deg per
  decision under a -1.8 m/s^2 downward command. Over all 209 Isaac episodes
  on record the early (first 1.3 s) mean applied az separates them: < -0.6
  -> 5/5 violated, -0.6..0 -> 4/151, >= 0 -> 0/53, i.e. reduced collective
  thrust is reduced attitude authority against the seeded disturbance torque.
  What-if on the 15 /11 checkpoints: /11 92.8 % landing, abort 4.7 %, peak
  first-2-s tilt 19.7 deg; with the floor 95.7 %, abort 1.2 %, peak 12.1.
  The floor does NOT apply inside the terminal corridor: applied
  unconditionally it broke the curriculum's easy rung (hand-over at 0.4-0.7 m,
  fixed-descent probe 11/12 -> 2/12 SUCCESS, caught by
  `tests/test_exploration_reachability_tool.py`). With that exemption the
  real law on the /11 checkpoints gives 95.7 % / unsafe 2.9 % / abort 1.2 %.
  Guard in `tests/test_handover_limit_rung.py`. Run
  `results/full_pipeline_20261007_v12_5seeds`.
- `spatial-reference/11` (2026-10-07, profile v12, no legs) is `/10` plus ONE
  law, the HANDOVER LIMIT, on the user's instruction to fix the handover tilt that alone failed
  /10's acceptance. All seven Isaac SAFETY_ENVELOPE_VIOLATIONs on record
  (/2-/10) came 0.9-1.3 s after handover, each starting with a near-full
  1.5-2.3 m/s^2 lateral command issued while the track was not yet trusted;
  five then reversed it in one decision (-2.2 -> +2.3) and PX4 overshot past
  21 deg, two held it under the seeded disturbance torque. Locally the
  median first-2-s peak tilt is 15.8 deg and the max 21.0, no margin. For
  the first 2 s after handover the supervisor caps the lateral command at
  2.0 m/s^2 and its change at 1.5 m/s^2 per decision
  (`handover_lateral_cap_m_s2`, `handover_window_s`,
  `handover_lateral_slew_m_s2`). What-if on the 15 /10 checkpoints: none
  93.5 % / peak max 21.0 deg; cap 2.0 91.4 % / 15.0; cap + slew 1.5 90.8 % /
  14.9; braking until trusted, slew alone or a measured-tilt guard all left
  the max at 20.5-20.7. The slew limit is kept for the Isaac reversal
  mechanism the local plant cannot show. 20/21 deg limits unchanged. Guard
  `tests/test_handover_limit_rung.py` (replays the /10 violation's commands).
  Run `results/full_pipeline_20261007_v11_5seeds`.
- `spatial-reference/10` (2026-10-07, profile v12, no legs). It is `/9` plus ONE law, the
  terminal-corridor GATE without legs, on the user's choice to address the
  blind terminal phase with the legs kept off. The corridor's lateral gate is
  the camera footprint and shrinks to zero near the pad; without legs, at
  0.20-0.25 m body height it is 0.03-0.07 m. In the /9 Isaac flight the
  packet-only arm sat 0.10-0.17 m off-centre there with a FRESH track, the
  vertical hold kept it up, and it aborted (6/10) or hovered to the limit
  (4/10). `terminal_gate_width_floor_m` 0.2 (the commit still demands
  lateral + 2 std inside the pad) and `terminal_commit_window_s` 3.0 (was
  1.5). What-if on the 15 /9 checkpoints, 48 held-out seeds, nothing
  retrained: (0, 1.5) 90.0 % / 4.3 % unsafe / 3.6 % timeout; (0.2, 3.0) 92.9 /
  4.7 / 1.8, the proposed arm's timeouts 10.4 -> 5.4 %. The local optical
  model does NOT reproduce the Isaac low-altitude stall for the vector arm, so
  the Isaac flight is this law's real test. Guard
  `tests/test_corridor_brake_rung.py` (replays the stalled state).
  **Measured, `/10` five seeds (`results/full_pipeline_20261007_v10_5seeds`,
  2026-10-07 10:48):** held-out after PPO `ppo_ontology_rgat` 95.8 +- 1.3 %
  (unsafe 3.3, timeout 0.0; /9 85.0 +- 17.5), `ppo_semantic_flat` 97.1 +- 2.1
  (2.9), `ppo_vector_canonical` 87.5 +- 7.8 (8.3). **Isaac, 30 episodes: 20
  SUCCESS, the graph arms 10/10 EACH, zero UNSAFE_CONTACT.** The packet-only
  arm is 0/10 (7 TASK_TIMEOUT, 2 SAFE_ABORT, 1 SAFETY_ENVELOPE_VIOLATION at
  1.0 s, tilt 21.1 deg at 2 m right after handover). Its timeouts are no
  longer the gate stall: it now enters the corridor and descends at the 0.24
  brake target, but too late, or hovers at 0.6-0.8 m off-centre, or loses the
  track at 0.14-0.27 m and holds. Acceptance fails on `no_unsafe_outcomes`
  ALONE, from that one baseline handover tilt; every other check passes,
  including `proposed_landing_observed`. The legless terminal phase is now a
  property of the BASELINE's input, not of the task: both graph arms land it
  every time in Isaac. Still not a superiority claim on five seeds and ten
  Isaac episodes per arm, and the vector arm's held-out 87.5 % against its
  Isaac 0 % is itself a sim-to-sim gap worth measuring.
- `spatial-reference/9` (2026-10-07 ~01:30 KST, deployment profile
  `config/spatial-isaac-system-v12.yaml`). It is `/8` plus exactly ONE law,
  the terminal-corridor BRAKE, chosen on the user's instruction to close the
  gap between the descent the supervisor admits and the descent the verdict
  accepts: `SpatialConfig.terminal_descent_speed_factor` 0.8 (was 1.5, the
  2D reference's approach margin, which `two_axis/config.py` still carries as
  `terminal_descent_speed_margin`) and
  `terminal_descent_brake_one_step` (the excess is removed within one 0.1 s
  decision step, bounded by the vertical authority, instead of over the
  ~0.35 s response delay). It was MEASURED before it was written: on the SAME
  15 `/8` checkpoints over 48 held-out seeds with nothing retrained, the
  shipped brake gives 76.5 % landing / 14.2 % unsafe (96 of 101 unsafe
  terminals are corridor contacts at a median 0.34 m/s); 0.8x with the
  delay-paced gain 81.0 / 10.3; 1.0x with the one-step gain 79.6 / 11.5;
  **0.8x with the one-step gain 87.9 / 2.8**, timeouts and aborts unchanged,
  the 18 remaining contacts all at 0.31 m/s. Neither constant alone works.
  0.8x is the sink speed the reward's readiness term already targets
  (`desired_z = -0.8 * touchdown_z_speed`), so the supervisor now enforces
  what the reward asks for. Guard `tests/test_corridor_brake_rung.py`; /8's
  `config_sha256` is pinned there. **Measured, `/9` five seeds
  (`results/full_pipeline_20261007_v9_5seeds`, same recipe, 48 held-out
  seeds, deterministic):** after PPO `ppo_semantic_flat` 97.5 +- 2.0 %
  (unsafe 1.7), `ppo_vector_canonical` 87.5 +- 10.9 (unsafe 10.8),
  `ppo_ontology_rgat` 85.0 +- 17.5 (unsafe 0.4; one seed low, timeouts 10 %).
  Training unsafe fell 7.5 -> 1.7 % against /8 on the same runs. **Isaac, 30
  episodes: 18 SUCCESS and ZERO UNSAFE_CONTACT** -- the first flight with no
  impact-speed failure. rgat 9/10, flat 9/10, vector 0/10. The one unsafe
  outcome is a SAFETY_ENVELOPE_VIOLATION 1.1 s after handover (rgat 829,
  tilt 21.9 deg at 2 m), the handover transient seen on /6-/8. Every vector
  failure is the legless blind terminal phase: the track is lost at body
  0.20-0.25 m, the vehicle holds or aborts while the pad drives 6-8 m away
  (6 SAFE_ABORT), or hovers at 0.3-0.4 m to the 70 s limit (4 TIMEOUT); one
  flat episode did the same. Acceptance fails on `no_unsafe_outcomes` alone
  (that single handover tilt). Remaining work, both measured: the handover
  tilt transient, and the blind last 0.2 m without legs that the packet-only
  arm cannot fly.
- `spatial-reference/8` (2026-10-06 ~22:30 KST, deployment profile
  `config/spatial-isaac-system-v12.yaml`) is `/7` MINUS one law: the /4
  landing gear is removed on the user's instruction ("remove drone legs
  again"). `landing_gear_extension_m` is 0 on `/8`, so `touchdown_height` is
  the stock 0.12 m again, the supervisor's gate/corridor/stopping margin and
  the curriculum's low start band follow it down, and v12 = v11 with
  `vehicle.landing_gear.extension_m: 0` so `landing_world.py` attaches no leg
  colliders. Board, optical realism, the /6 servo and the /7 verdict are
  inherited; /7's `config_sha256` is unchanged and pinned. This gives back the
  geometry /4 was built to avoid -- the camera is 0.04 m BELOW the pad plane
  at contact and the last centimetres are flown on the estimator's memory --
  and whether the v11 centre cluster plus the /5 optical model carry that
  interval was never measured with the legs off; the /8 run measures it.
  Measured before launching it: the swept teacher lands 21/24 nominal on `/8`
  (22/24 on `/7`) and 12/12 at difficulties 0.5 and 0.0, so the task is
  solvable without legs. What broke was the CURRICULUM's easy end: the band
  `start_height_range_m` (0.2-0.5 m) is a body height, and without the legs
  its low end puts the camera 0.04 m above the deck where no tag fits the
  frame -- the estimator never initialises, `prolonged_visual_loss` latches
  at step 1 and the supervisor climbs; the fixed-descent reachability probe
  aborted 5 of 12 easy seeds (`/7`: 0 of 12). `SpatialConfig.easy_start_lift_m`
  therefore lifts the easy band by 0.18 m on /4-/8 (the gear on /4-/7, kept on
  /8 without the gear), which is byte-identical for /4-/7 and restores the
  probe to 11-12 of 12 on /8. Training-only; difficulty 1.0 never reads it.
  **Measured, `/8` five seeds (`results/full_pipeline_20261006_v8_5seeds`,
  2026-10-07 00:59, same recipe, 48 held-out seeds, deterministic):** clones
  rgat 72.9 / flat 75.0 / vector 47.1 % (vector timeouts 41 %); after PPO
  flat 77.9 +- 6.1, vector 77.9 +- 5.5, rgat 73.8 +- 15.6 % (seed 829 at
  52 %, seed 832 at 92 %), unsafe 12-16 %. Against /7 on the same seeds that
  is -10 to -15 points of landing and +5 of unsafe for every arm; over the
  same 12 training runs SUCCESS fell 87.5 -> 75.2 % and TASK_TIMEOUT rose
  6.6 -> 14.5 %. Isaac, 30 episodes, no restart needed: **16 SUCCESS**, 5
  UNSAFE_CONTACT (vz -0.30 to -0.32), 2 SAFETY_ENVELOPE_VIOLATION at
  0.9-1.3 s (tilt 21.3-21.5 deg, handover), 5 SAFE_ABORT, 2 TASK_TIMEOUT.
  Per arm: `ppo_ontology_rgat` 8/10, `ppo_semantic_flat` 7/10,
  `ppo_vector_canonical` 1/10 -- its five aborts and two timeouts all lost
  the track at 0.10-0.25 m body height and hovered or drifted while the
  pad drove off. EVERY one of the 16 landings touched down with
  `optical_detected` False and a track age of 0.6-3.0 s: without the legs
  the last 0.1-0.2 m is flown blind and only the corridor's 1.5 s commit
  memory carries it, which the graph arms survive and the packet-only arm
  does not. Acceptance fails on `no_unsafe_outcomes` and on
  `proposed_landing_observed`, which requires every proposed-arm run to
  land (seed 831 landed 0/2). Read this as a GEOMETRY measurement, not a
  representation claim: the legs were worth ~7 of 30 Isaac landings and the
  whole of the baseline's Isaac performance. Whether to restore them is the
  user's call.
  The driver now passes `--isaac-reset-recoveries` (default 2) to the Isaac stage
  because the /7 flight died at run 7 of 15 on a PX4 "Preflight Fail: High
  Accelerometer Bias" arming refusal with no restart budget (infrastructure,
  not RL; 18 episodes unflown). First run: `results/full_pipeline_20261006_v8_5seeds`.
- `spatial-reference/7` (2026-10-06 late evening, deployment profile
  `config/spatial-isaac-system-v11.yaml`) is `/6` plus exactly ONE law,
  the CONTACT VERDICT (`SpatialConfig.contact_verdict_by_speed`), and it is a
  USER DECISION, not a measurement: of the six non-landing contacts in the
  five-seed `/6` flight, four were vehicles sinking onto their legs from a low
  hover under the supervisor's vertical hold (`vertical_stopping_margin`) at
  |vz| 0.01-0.07 m/s -- two scored UNAUTHORIZED_CONTACT inside every touchdown
  limit, two scored UNSAFE_CONTACT just outside the attitude band (body rate
  10.47 deg/s; tilt 5.27 deg at 12.51 deg/s) -- and two touched inside the
  terminal-descent corridor at vz -0.62 and -0.38 m/s against the 0.30 m/s
  limit. The user ruled the four landings and the two not. The one rule that
  draws that line is the impact speed: on `/7` a pad contact is UNSAFE_CONTACT
  when |v_xy| > 0.35 or |vz| > 0.30 m/s, SAFETY_ENVELOPE_VIOLATION when the
  thrust axis is past 21 deg at contact, UNAUTHORIZED_CONTACT only while the
  abort hold owns the vehicle, and SUCCESS otherwise, whatever the inhibit
  flag, tilt or body rate. Tilt and rate remain the corridor's `settled`
  commit condition and the readiness term's price; the limit VALUES are
  unchanged, and so are plant, packet, graph, ontology, supervisor, terminal
  table and profile (guard `tests/test_contact_verdict_rung.py`, which
  replays the six recorded contacts under both rungs). The earlier
  description of two of the four as "boundary" cases (rate "exactly 10.0")
  was wrong -- read `truth_relative_velocity` / `truth_roll_pitch` /
  `truth_angular_rate` at the contact step of the per-run `isaac_*.jsonl`
  before characterising a contact. Re-scoring the recorded `/6` flight under
  `/7` gives 27 SUCCESS / 2 UNSAFE_CONTACT / 1 SAFE_ABORT, so acceptance would
  still fail on the two corridor impacts; the open design question (the
  corridor admits 1.5 x 0.30 m/s, landing requires 0.30) now concerns those
  two only. `--contract-version` lists `spatial-reference/3..7`, so every
  earlier rung's checkpoints stay loadable. **Measured, `/7` five seeds
  (`results/full_pipeline_20261006_v7_5seeds`, same recipe as /6, 48
  held-out seeds, deterministic, scored under /7):** clones 79-85 % landing
  with 0 % unsafe (their non-landings are 15-20 % TASK_TIMEOUT); after PPO
  `ppo_vector_canonical` 92.5 +- 4.7 %, `ppo_ontology_rgat` 90.4 +- 8.4 %,
  `ppo_semantic_flat` 84.6 +- 8.5 %, unsafe 7.1 / 9.2 / 14.6 %, timeouts
  ~0. PPO converts the clones' timeouts into contacts and a tenth of those
  are too fast: of 720 held-out terminals, 642 SUCCESS, 74 UNSAFE_CONTACT,
  and ALL 74 are vertical-speed exceedances (none lateral), 49 of them in
  the 0.30-0.35 m/s band and 24 in 0.35-0.45; SUCCESS contacts touch at a
  median 0.18 m/s (p90 0.27). 62 of the 642 landings were settle contacts
  under the supervisor's hold, i.e. the reclassified kind. Training SUCCESS
  rose from 864-1748 to 2056-2200 of 2400 per run (part of that is the
  reclassification itself). The Isaac flight reached 12 of 30 episodes
  before PX4 refused to arm (see the /8 bullet): 9 SUCCESS, 3
  UNSAFE_CONTACT at vz -0.31 / -0.33 / -0.3x against 0.30, all inside the
  corridor, with the SUCCESS cases touching at -0.29. Under /7 the only way
  left to be unsafe is impact speed, and the policies descend at the limit;
  the corridor admits 1.5 x 0.30 m/s while landing requires 0.30, so the
  remaining design question is that 0.15 m/s gap. No summary.json was
  written for the /7 run (the Isaac stage exited 1, not 2).
- `spatial-reference/6` (2026-10-06 night, deployment profile
  `config/spatial-isaac-system-v11.yaml`) is `/5` plus exactly ONE law, the
  attitude servo's damping:
  `SpatialConfig.attitude_omega / attitude_damping` = 10 rad/s at zeta 0.7
  (`REFERENCE_ATTITUDE_SERVO_6`), identified from the tilt traces of all four
  Isaac flights (46 episodes, 1360 steps) by simulating the local servo on the
  recorded commanded attitude: RMS tilt error 2.43 deg on a plateau at omega
  9-10 / zeta 0.65-0.7, against 2.91 deg for the /4-/5 servo (14, 1) and
  2.73 deg for /3's (10, 1). The velocity reversal estimator (225 events)
  ranks (10, 0.7) a close second to (14, 1), error 0.080 vs 0.035, so both
  estimators admit it; steady-state tilt gain is 0.90-1.02 (no gain error).
  Only zeta < 1 reproduces the mechanism behind the fourth flight's envelope
  violation: a roll reversal from +8.5 to -15 deg that PX4 overshot to
  -21.2 deg at 107 deg/s; a critically damped local servo never overshoots,
  so no policy trained on it could learn that margin. A 13 deg step peaks at
  13.47 deg on /6 and at 13.00 on /5 (guarded in
  `tests/test_actuation_delay_rung.py`). Teacher gains unchanged (20/24).
  First run: `results/full_pipeline_20261006_v6_5seeds`, FIVE training seeds
  (828-832), which is also the first arm comparison with more than two.
  **Result (48 held-out seeds per run, mean +- sd over the 5 training
  seeds, deterministic): `ppo_ontology_rgat` 81.2 +- 6.6 % landing, unsafe
  14.2 +- 4.0 %, return +19.3; `ppo_vector_canonical` 75.0 +- 12.1 %, unsafe
  21.7; `ppo_semantic_flat` 60.8 +- 22.3 %, unsafe 28.7 (its seed 828
  collapsed to 18.8 % with 56 % unsafe; its clones averaged 67.9 %, so PPO
  moved that arm DOWN).** Isaac, 30 episodes (15 runs x 2): 23 SUCCESS, 4
  UNSAFE_CONTACT, 2 UNAUTHORIZED_CONTACT, 1 SAFE_ABORT; the proposed arm
  8/10 SUCCESS. Acceptance still fails on `no_unsafe_outcomes` alone (6 of
  30). Read this as: the recipe lands every arm, the proposed arm is the
  most consistent of the three on this rung, and 5 seeds with sd 7-22 points
  is still not a superiority claim (on /5 with 2 seeds the flat arm was
  ahead). The relational path was ACTIVE in all five proposed-arm runs.
- `spatial-reference/5` (2026-10-06 evening) is `/4` plus exactly ONE law, optical realism of the LOCAL sensor
  (`SpatialConfig.optical_realism`, `LocalBackend._optical_quality`): the pad
  is detected when at least one tag of the deployed board (read from the
  deployment profile) projects fully inside the frame at >= 12 px, and
  `detectionConfidence` is the detector's own metric (largest seen tag's
  pixel side / 120 px, times a sharpness drawn from N(0.65, 0.08), times
  0.85 when a single tag braces the solve), with a 5 % frame-miss rate. The
  estimator never reads the confidence, so only the policy's input changes.
  **Why it exists: after the third Isaac flight (0/12, detection 100 %, hover
  under the stopping-margin inhibit), evaluation traces were given the
  normalized packet (`trace(..., packet=...)`) and the SAME checkpoints were
  re-flown; of 44 channels, `detectionConfidence` was the one that differed
  beyond its own spread -- a constant 0.98 locally against 0.22-0.58 in
  Isaac, falling with camera depth exactly as the tag-size formula says
  (0.22 at 1.8-2.6 m, 0.45 at 0.8-1.1 m, 0.58 at 0.2-0.5 m). Forcing the
  realistic generator into the /4 plant turned the /4 checkpoints from
  SUCCESS in 5-12 s (corridor armed 53-72 % of steps) into 70 s timeouts with
  the corridor armed 2-11 % and 1.5-2x larger, 2x jerkier commands -- the
  Isaac behaviour, reproduced locally with one channel.** A channel that is
  constant in training is an untrained direction of the policy; check every
  packet channel's training-time variance before the next flight
  (`tools/compare` script in results/full_pipeline_20261006_v4/low_sigma_kl_diag).
- `spatial-reference/4` (2026-10-06, deployment profile `config/spatial-isaac-system-v11.yaml`)
  is `/3` plus three MEASURED laws, each with its own tool and number, after
  the first two learned-policy Isaac flights landed 0 of 22 episodes while
  the same checkpoints landed locally: (a) attitude bandwidth 14 rad/s
  (`SpatialConfig.attitude_omega`; reversal-triggered response over 146
  Isaac command changes: Isaac [0.34, 1.09, 1.65, 1.80], 10 rad/s err 0.286,
  14 rad/s err 0.084, 20+ overshoots -- the plain FIR estimator is
  ill-conditioned on bang-bang commands, read the reversal estimator);
  (b) 0.18 m landing gear (`landing_gear_extension_m`, `touchdown_height`
  = 0.30 m; four leg colliders under `/body` in `landing_world.py`), because
  the camera sits 0.16 m below the body and the stock Iris touched down at
  0.12 m, i.e. the camera was 0.04 m BELOW the pad at contact and the tags
  left the frame before touchdown -- with the gear it stays 0.14 m above the
  pad; (c) a 17-tag ArUco board at three scales (0.26 m corners, 0.14 m
  mid-ring, 0.08/0.05 m centre cluster) in place of the five-tag board, so
  a tag is decodable from 2.5 m down to the last 0.3 m, including at the
  FOV edge where the /3 flight's track flickered and was distrusted. The
  supervisor's gate, corridor and stopping margin are heights above the
  stock-gear touchdown and shift with the gear; the camera footprint does
  not. Packet, graph, ontology, reward and capabilities are identical to
  /3; /1-/3 checkpoints load under their own schema. Teacher gains are per
  rung (`TEACHER_GAINS_BY_SCHEMA`); the /3 set lands 21/24 on /4.
- `spatial-reference/3` (2026-10-06) is `/2` plus exactly ONE law: the local plant applies a
  command only after `REFERENCE_ACTUATION_DELAY_S` = 0.10 s
  (`SpatialConfig.actuation_delay_s`, queue in `LocalBackend.advance`), and
  the supervisor's stopping margin counts the same latency. The value is
  measured, not chosen: the joint impulse-response fit of the first Isaac
  flight (see the Isaac bullet below) has no same-step response, and 0.10 s
  minimises the error against it (0.128 vs 0.229 for /2; 0.12 s gives 0.126).
  Packet, graph, ontology, reward, every capability and the deployment profile
  are byte-identical to /2 -- `tools/contract_snapshot.py` returns both shared
  contracts identical and the /2 `config_sha256` is pinned in
  `tests/test_actuation_delay_rung.py` -- so /1 and /2 checkpoints load under
  their own schema (`--contract-version spatial-reference/2`). The swept
  teacher of /2 (kp 0.4, kd 0.8, ki 0.4) lands 0 of 24 on /3, the same
  failure the Isaac flight showed, so /3 carries its own swept gains; a
  derivative gain above 0.6 lands nothing on the delayed plant.
  `/2` was derived from the 2D reference rather than chosen from the ladder: 44 fields
  (12 per-axis x 2 + 14 shared + the declared transport-delay and disturbance
  extras), two reference planes, and v10's capability set. `/2` differs from
  `/1` in exactly ONE law -- while descent is inhibited the supervisor holds
  the vertical rate instead of braking a descent one-sidedly
  (`SpatialConfig.vertical_inhibit_holds`, measured in
  `tests/test_spatial_vertical_inhibit.py`). Packet, graph, ontology, reward
  and every capability are byte-identical between the two, so `/1` stays
  loadable and the rungs stay directly comparable; only `config_sha256`
  differs, which is what stops a `/1` checkpoint loading under `/2`. Select the
  frozen rung with `--contract-version spatial-reference/1`.
  `spatial-causal-rgat/3..10` are frozen historical contracts; v5 in
  particular is NOT a 3D version of the reference -- its graph fills the nine
  rows positionally, it lacks predicted bearing/margin and acceleration
  uncertainty, and its action is a velocity setpoint carrying a hidden
  integrated velocity reference instead of applied net acceleration.
- Replicating the planar nine nodes per axis is a correct dimensional
  extension but NOT a complete one. `ppo_semantic_flat` and
  `ppo_ontology_rgat` read the graph alone; only `ppo_vector_canonical` reads
  the packet, so a packet field absent from the context rows is information the
  BASELINE holds exclusively. Measured by perturbation: planar 2/26 blind
  (upstream's policy-memory fields), spatial 6/44 before the extension -- the
  three extra being exactly the quantities describing what 3D adds. The spatial
  ontology therefore declares two extension nodes in
  `python/ontology_rgat/landing/ontology.py`, each justified by a measurement:
  `DisturbanceEstimate` (3D samples a constant per-episode external force up to
  0.75 N per axis, 0.5 m/s^2 against a 2.5 m/s^2 authority; the causal observer
  in `Estimator` recovers it from own state with mean error 0.030 m/s^2 and
  per-axis correlation 0.998-0.999) and `MeasurementLatency` (75 ms optical
  transport delay). Yaw was considered and REJECTED on measurement: 0.03 deg
  mean drift, 0.29 deg maximum. Extensions append, so the planar nine keep
  their indices, `DescentEligibility` stays at 7, the four grouped readouts are
  preserved and the planar schema hash is byte-identical. After adding any
  phenomenon to the 3D plant, check
  `test_3d_gives_the_baseline_no_information_the_graph_arms_cannot_see`.
- The 2D reference is NOT a noise-free experiment and must not be described as
  one: it carries `bearingNoiseStd 0.15 deg`, `relativePositionNoiseStd 0.02 m`,
  25 % short dropouts, 3.5-5 s sustained dropouts and a pitch-disturbance
  channel. What it lacks is a physics engine, 6-DOF rigid-body motion and 3D
  external forces.
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
- Do not run real-vehicle commands or push changes without explicit user
  approval. Bounded local tests and smoke rollouts are allowed. Long PPO
  training is what bare `run.sh` is FOR, so invoking it is itself the approval;
  starting one any other way still needs asking.
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
- **`spatial_current_semantics_v1` (2026-10-28, `./run.sh current-semantics-3d`)**
  is a SEPARATE current-observation 3D study built from the user's guide
  `docs/current_semantics/CURRENT_SEMANTICS_3D_REFACTOR_GUIDE_KO.md`; code in
  `python/ontology_rgat/current_semantics/`, status in
  `docs/current_semantics/G0_G4_STATUS_KO.md`, audit in `SOURCE_AUDIT_KO.md`.
  Policy input is the current 21D vector only (no history, integral, teacher,
  prior or supervisor); four budget-matched arms (raw 4022 / phi44 MLP 4008 /
  untyped GNN 4020 / ontology R-GAT 4016 parameters) with one common raw21
  critic. USER DECISION 2026-10-28: the environment is Isaac-linked -- train on
  the `spatial-reference/12` LocalBackend (Isaac-calibrated) and run G3 and
  every selected-checkpoint evaluation on Isaac/PX4; the direct_policy local
  toy is refused (its attitude is axis-swapped and pitch-flipped vs Isaac).
  Gates: G0/G1 blocked (the 2D v1 evidence bundle and commit 5853781 are not
  on this machine), G2 partial, G3 PASSED (local half on 8 pilot seeds; Isaac
  half `--stage coupling --backend isaac --allow-isaac`, flown 2026-10-28 on an
  owned stack after user approval: 6 episodes, all stops confirmed, axis corr
  0.89/-0.84, Isaac contact body height 0.068-0.070 m vs the calibrated 0.0672),
  G4 FAILED. Later the same day the user waived G0/G1 and the two G2 items
  (recorded as `waived`, never as passed) and ran the registered G4 pilot (Raw PPO,
  2 model seeds x 300 updates x 12 episodes, criteria registered first): 0 % on
  48 test seeds deterministic AND sampled, all TASK_TIMEOUT; 0 SUCCESS in 7,200
  training episodes. Cause, measured: the random-init mean action carries a small
  constant bias that integrates (net acceleration) into the 20 m / 60 m envelope
  within ~15 s, so the time-based curriculum's easy rungs were spent removing that
  drift; by then difficulty was >= 0.6 and the policy settled in the hover-to-
  timeout optimum (-12 beats a crashed attempt's -40). Landing IS reachable under
  the pilot's own exploration: a pad-tracking descent mean lands 15/24 at latent
  sigma 0.35 on the easy rung (24/24 noiseless). No arm comparison may be read
  from this (guide 13). The development and
  confirmation stages always exit 3. Second pass the same day (user: "edit all
  as you recommend"): task contract = the spatial-reference/12 Evaluator
  unchanged on the actual transition time, no supervisor (`task_hash`, bound
  into checkpoints); local contact height 0.0672 m (median of 78 legless Isaac
  first contacts); local handover waits for the first accepted frame; W_r init
  N(0, 1/d) (g8 RMS 0.004 -> 0.05-0.07). Thinning local frames to the
  decision-level "~10.6 Hz" was MEASURED AND REJECTED: the reference ABG track
  locked out on 11/16 local episodes vs 0/16 at 20 Hz and 1/161 in Isaac, so
  `local_plant.optical_cadence` stays identity. The Isaac half then measured
  ~1 frame per decision at the observer (spacing 0.092-0.096 s, i.e. ~10.4 Hz
  is real), yet the 21D flag/age statistics already match local 20 Hz (updated
  in view 0.988 vs 0.977, age at update 0.100 s both); why the same track locks
  out locally at 10 Hz but not in Isaac is unexplained.
  **G4 pilot 2 (`g4-pilot/2`, user "all")** -- actor output init x0.01,
  success-gated curriculum, 600 updates -- FAILED too: neither seed ever left
  difficulty 0, so no checkpoint was testable; 3 SUCCESS in 14,400 training
  episodes. Removing the init drift exposed the deeper cause, measured on
  diagnostic seeds 996001-996024 (`results/.../g4_pilot2_diagnostics/`):
  zero-mean exploration on the net-acceleration double integrator lands
  0-2/24 at d=0 for EVERY sigma 0.05-0.37, pad static or moving (disturbances
  are already x0 at d=0) -- sigma only rescales time, so landing is set by
  geometry (descend 0.4-0.7 m without drifting 0.5 m). With no success in the
  batch PPO drifted to a SATURATED flee policy (seed 0 mean [+2.47, +2.00,
  +1.93] m/s^2, 60 m lateral exit at 8.5 s) that is WORSE on its own
  discounted objective (-40.6 vs -34.0 from s0); a one-step bandit confirms
  the update direction is correct. Every spatial PPO policy in this repo that
  lands held out started from a BC clone; from-scratch PPO under the guide's
  no-teacher rule had never produced one.
  **G4 PASSED with pilot 3 (`g4-pilot/3`, 2026-10-28, user: reshape the common
  reward/observation so PPO lands without BC).** Same PPO, budget and criteria
  as pilot 2; only two common changes, both still truth-free in the policy
  path: task v2 = the /12 Evaluator's VERDICT and terminal table unchanged, but
  the dense part replaced by `current_semantics/reward.py`
  (`spatial-current-reward/2`): undiscounted Phi(s') - Phi(s) of
  Phi = -(2*dist + 4*|v_rel - v_des|) on an approach velocity field (align,
  then sink to 0.2 m/s), Phi kept at the terminal, time 0.3/s + control cost;
  observation v2 = same 21 channels with landing-regime scales (position 1 m,
  height 2 m, velocities 1 m/s; `OBSERVATION_SCALE_OVERRIDES`, `from_source`).
  Measured BEFORE training: under the Evaluator's own dense terms an action's
  projection on the approach direction correlates -0.05..0.08 with the next
  1-10 rewards (position potential = double integration away; readiness flat
  beyond ~1 m), so the advantage carried no direction; under v2 0.10-0.46.
  Result, 48 held-out test seeds at d = 1: seed 0 det 48/48, sampled 46/48;
  seed 1 det 46/48, sampled 44/48; every failure UNSAFE_CONTACT; first
  training SUCCESS at update 4-9, d = 1.0 by update 69-71 (pilot 2: 3 SUCCESS
  in 14,400). The policy beats the memoryless field-follower it was shaped
  toward (20/24 nominal). LOCAL plant only, Raw arm only, reward and
  observation changed together (contributions not separated). Pilot 2
  archived in `g4_pilot_v2/`.
  **Isaac evaluation of the two selected checkpoints (`isaac-eval/1`,
  `--stage isaac-eval --allow-isaac`, owned stack, flown 2026-10-28 on the
  user's approval): 4/10 and 3/10 deterministic, criterion 50 % NOT met**
  (same seeds 994001-994010 locally: 20/20). The 7 landings are clean (0.15-
  0.21 m off centre, 0.12-0.22 m/s). Failures: 3 tilt > 21 deg within 1.1 s of
  handover (21.5-23.4 deg; local max 17.7 deg even with a 1.5x kick), 10 a
  missed first approach with no recovery (seed 0 hovers at 0.5 m while the
  pad drives 60 m away; seed 1 descends below the pad plane 4-11 m off).
  Outcome is set by the SEED, not the checkpoint (9/10 seeds agree): handover
  lateral velocity kick 0.08-0.55 m/s -> both land, 0.62-0.98 -> both fail.
  Locally the policies keep 70-80 % at a 1.5x velocity kick, so the margin is
  small AND Isaac's post-handover attitude response is harsher than the local
  plant's -- the gap /11-/12 closed with SUPERVISOR laws, which this path does
  not have. **Traced diagnostic re-flight (`--stage isaac-trace --allow-isaac
  --pairs ...`, user chose it; status doc 10.7; `results/.../isaac_trace/`):**
  6 pairs reproduce the registered outcomes; every local counterpart lands.
  Mechanism, measured: right after handover Isaac's EKF own velocity jumps
  +-0.6 m/s between decisions (first-step "measured accel" +5.97 / -5.84
  m/s^2), so the estimated relative velocity takes the WRONG SIGN for
  0.2-0.4 s (+0.64 vs truth -0.78; 0/6 locally) and the height estimate lags
  0.43 m in a fast descent (local 0.12); the policy answers with +-2 m/s^2 and
  reverses 0.3-0.5 s later, and PX4 overshoots: realized/commanded tilt
  1.50-1.68 on the failing pairs vs 0.80-0.94 locally (994006: local's
  command change was LARGER, 3.09 vs 2.21, yet tilt 11.6 vs 23.6 deg). Tilt
  past 21 deg ends it (B); a -2 m/s^2 descent instead lands 0.8 m off the pad
  plane, the camera loses the pad, the estimate freezes and the policy never
  climbs to re-acquire (A). Not the verdict, reward ordering or missing
  information. Proposed (user decides): training-only DR of a decaying
  own-state error after handover (~0.6 m/s, ~0.3 s) and of servo overshoot,
  plus a command-change/tilt cost in the common reward (the /11 slew limit as
  a reward, no supervisor), then re-evaluate on NEW Isaac seeds (994001-994010
  are now diagnostic).
  **Replication `isaac-eval/2` (user "c": same checkpoints, fresh seeds
  994011-994020, traces on; `/1` archived in `isaac_eval_v1/`): 2/10 and 3/10,
  criterion not met; 12/40 over both.** One PX4 failsafe (gcs/battery) cut a
  flight with no verdict; `isaac-eval` now RESUMES an interrupted protocol
  (keeps every verdict, re-flies only the unjudged pair, logs
  `interruptions.jsonl`) and refuses to re-fly a complete one. Outcomes
  pre-registered from each seed's handover lateral kick were right on 15/16
  definite calls (the miss had a +0.91 m/s VERTICAL kick). New, measured: in
  8 of 14 missed-approach failures the reference ABG track LOCKS OUT 1.6-2.0 s
  after handover -- detections rejected for > 1 s while the pad sits well
  inside the camera footprint -- because the transient pushed the estimate
  > 0.5 m off; it never re-acquires (994017: 70 s). Same lockout G3 saw at
  local 10 Hz. Proposed (user decides): re-acquire the track after repeated
  rejected detections (common observation, truth-free), handover-transient
  DR, command-change/tilt cost, then new seeds.
  **observation/3 + pilot 4 (user "a", 2026-10-29; status doc 11).** Cause in
  code: `ReferenceSpatialTrack` initialises the PAD velocity to the OWN
  velocity and corrects it by 0.02/gap per frame, so after a handover kick,
  or at Isaac's ~10 Hz frame spacing even without one, its prediction leaves
  the innovation gate and it never accepts again (synthetic: locks out at
  0.4-0.8 s). The shared track is untouched; `CurrentObserver` now restarts
  it after 3 consecutive rejected fresh detections from their least-squares
  position/velocity (`TRACK_REACQUIRE_RULE`, in the observation manifest);
  bit-identical while frames are accepted. Pilot-3 policies locally: 20 Hz
  48/48, 47/48 either way; thinned to 10 Hz 31/48, 30/48 without the rule
  (Isaac-like failures) vs 48/48, 47/48 with it. Pilot 4 (pilot 3's protocol,
  only observation/3) PASSED: test det 48/48, 47/48, sampled 47/48, 47/48;
  10 Hz local 47/48, 48/48, 0 lockouts. G3 local re-run under the new
  observer: passed. **`isaac-eval/3` (fresh seeds 994021-994030, user "go"):
  PASSED, 6/10 and 7/10 (13/20 vs 12/40 for the pilot-3 checkpoints), every
  cleanup confirmed, no interruption, ZERO lockouts; both policies landed a
  1.04 m/s-kick seed that the old pipeline always lost.** 5 of the 7 failures
  are tilt > 21 deg within 1 s of handover (21.9-23.2 deg) -- the EKF
  transient / PX4 overshoot path, untouched by this fix; 2 descend below the
  pad plane 2-7 m off without lockout. Small sample, criterion barely passed:
  not robustness, not an arm comparison. A 10 Hz training cadence is now
  viable (it was rejected only for the lockout) but was not adopted.
  **Pilot 5 (user, 2026-10-29: training-only handover own-state error + reward
  /3 command-change and tilt costs; status doc 12).** The error (own velocity
  e0 exp(-t/tau), |e0_xy| <= 1.5 m/s, tau 0.15-0.5 s, position its integral,
  training episodes only) reproduces Isaac's wrong-sign transient locally
  (15-18/48 episodes, ~0.27 s). DESIGN FLAW, measured: the command-change
  cost charges the SAMPLED action, so at the initial sigma exploration noise
  alone costs ~-0.75 per decision and the policy dived into the pad (12/12
  UNSAFE in ~1 s) for ~340 updates until sigma shrank -- the design checks had
  used deterministic policies only. I stopped the first run prematurely at
  update 414 (it had already recovered; archived `g4_pilot5_stopped/`) and
  re-ran it unchanged (deterministic: identical path). PASSED with only ~190
  updates at d=1: test det 40/48, 46/48. Behaviour changed as intended: first
  1.5 s command jump 2.4-2.6 -> 1.2-1.3 m/s^2, max tilt p90 19-20 -> 14-15 deg.
  **`isaac-eval/4` (994031-994040, harsher kicks): PASSED 6/10, 5/10.** Tilt
  failures within 1.5 s fell 5/20 -> 2/20 (command-jump median 2.88 -> 1.19,
  tilt p90 22.1 -> 19.6 deg) but missed approaches rose 2 -> 7: after a big
  kick or a ~1.1 m/s estimate error the smoother policy drifts 1.6-2.5 m,
  loses the pad at 0.8-1.2 m and keeps DESCENDING on the frozen estimate,
  ending below the pad plane 3-4 m off; it never climbs to re-acquire.
  **Paired re-flight (user "a"; doc 12.5): pilot 4 on the same 994031-994040,
  12/20 vs pilot 5's 11/20 -- no difference** (both land 11, pilot 4 only 1,
  pilot 5 only 0, both fail 8). The /3 -> /4 drop was the HARDER SEED SET,
  not the policy. Pilot 5 survived the handover on 3 of pilot 4's 5 tilt
  failures and then lost the pad later, so reward /3 moved failures from
  handover tilt to failed recovery without adding landings. Diagnostic flights
  of archived checkpoints: `--stage isaac-trace --checkpoint-root <archived
  pilot> --pairs all` (loads a checkpoint trained on another TASK only when
  every other hash matches -- `load_checkpoint(allow_foreign_task=True)`,
  never in isaac-eval -- and checks the stored update against that pilot's
  record, because an archived g4_pilot.json's checkpoint path may point at a
  LATER pilot's file). Bottleneck now: no re-acquisition after losing the pad;
  remaining options: a vision-age-aware climb in the field, and removing the
  command-change cost's noise tax.
  **Pilot 6 (user "go" on both, 2026-10-29; status doc 13; measurements in
  `results/.../pilot6_design_checks/`, diagnostic seeds 995001-995048).**
  Reward /4 (task v4): (1) `VelocityField.lost_view` -- past 0.5 s of the
  policy's OWN vision age (full at 1.0 s) v_des_z turns from the sink into a
  climb toward 2.5 m above contact, except below 0.5 m AND within 0.75 m of
  the centre (every local landing goes blind below 0.43 m within 0.56 m).
  Replayed on the 60 recorded Isaac episodes it fires in 0/36 landings and
  10/10 late envelope violations, 4.3-15.9 s before each ended. (2) The
  command-change cost is REMOVED: any cost on the SAMPLED action's change
  taxes exploration noise, and no deadzone separates them (the jumps worth
  pricing, 0.5-1.2 normalised, are the size of the noise, Delta-u sd 0.47);
  a difficulty ramp fails too (log sigma is still -1.15..-1.3 when pilot 4
  reached d=1, tax 0.42/decision); a smoothness LOSS (CAPS) is excluded by the
  guide ("추가 head/loss"). The tilt cost stays: the plant filters the noise
  (0.000/decision untrained, 0.05 rough pilot 4, 0.007 smooth pilot 5); body
  rate does not (noise p99 73 deg/s vs pilot-4 handover median 76). (3)
  Training-only `vision_blackout` (half the episodes, no frames for 1-3 s
  starting 0.5-4 s after handover): trained policies lost the pad at altitude
  in only 0-5/48 local episodes, too rare to learn a reward from. With a
  forced blackout the pilot-4/-5 policies descend on the stale track and land
  30-37/48; the same policies with only "climb when age > 0.5 s above 0.5 m"
  overlaid land 46-48/48 and score higher under /4 (+23..+27 vs -4..+13).
  Pilot 5 archived in `g4_pilot_v5/` (+ reconstructed config snapshot),
  isaac-eval/4 in `isaac_eval_v4/`. Registered: `g4-pilot/6`, `isaac-eval/5`
  on 994041-994050, plus a paired pilot-6 re-flight on 994031-994040
  (pilot 4 12/20, pilot 5 11/20 there). Three changes at once: their effects
  are not separable.
  **Result: G4 PASSED** (test det 46/48, 48/48; sampled 43/48, 47/48; d=1.0
  at update 77/70 -- pilot 5's 405/411 delay is gone, no dive). The climb
  was NOT learned: 0 % of lost-view decisions at 0.5-3 m climb, locally or in
  Isaac. Cause is structural: the dense term is an UNDISCOUNTED potential
  difference and PPO discounts with tau 70 s (gamma 0.9986/decision), so the
  field moves the optimum only by (1-gamma)*Phi ~ -0.004/decision for a
  0.7 m/s mismatch, 1/8 of the time cost; it steers credit assignment, not
  the optimum. What PPO learned instead: slow the descent (-0.30 -> -0.20 m/s)
  and stay over the estimate (decisions > 1 m off: 0-6 vs 123-201), blackout
  landings 44-45/48 vs 33-40. **`isaac-eval/5` (994041-994050) PASSED 6/10,
  6/10**, every seed agreeing between the two policies. **Paired re-flight on
  994031-994040: 15/20 vs pilot 4 12/20 and pilot 5 11/20** (discordant 4:1
  and 5:1, exact p 0.38 / 0.22 -- a direction, not proof). All four newly
  landed pairs lost the pad at 1.0-1.3 m, 1.2-1.5 m off, HELD altitude
  (>= 0.64 m) and re-acquired, where pilots 4/5 kept descending. Pilot 6 in
  Isaac: 27/40. Remaining failures: handover tilt 6/40, and losses > 2 m off
  that need a real climb (994034). Over 100 traced Isaac episodes, mean
  applied az over the first 1.3 s < -0.6 -> 14/46 handover-tilt failures,
  -0.6..0 -> 4/46, >= 0 -> 0/8 (the /12 thrust-floor split). Diagnostic
  re-flights may now use already-diagnostic seeds (`DIAGNOSTIC_SEED_RANGES`)
  and are checked BEFORE a stack starts (`isaac_eval.check_pairs`); the first
  paired attempt booted Isaac and was refused with zero episodes.
  **Development stage (user "c" + "확장", 2026-10-29; status doc 14).** Task v5
  (`scenario.py`): pad bearing A and travel heading B over the full circle in 4
  sectors, held-out combos [[0,0],[1,2],[2,3],[3,1]] (balanced, not a cyclic
  shift) for the composition test -- the /12 draw (heading +-11.5 deg) gave the
  Raw policy 96/96 nominal and 0/96 at 90/180 deg, so composition had nothing to
  measure. `ScenarioLocalBackend` re-draws offset/heading after the /12 reset
  without touching shared files; Isaac cannot fly it yet (needs an Isaac-side
  scenario before guide 11.6). G4 pilot 7 (task v5, Raw) PASSED (45/48, 41/48).
  OOD levels fixed on Raw only by a pre-declared rule (mildest level with Raw in
  20-80 %): a2 0.6-0.9, dropout 3-4 s, optical noise x5. Raw is 0.91 on
  composition-OOD vs 0.94 nominal: little headroom on the primary metric.
  Registered `development/1` (plan d6be7db0, `config/current_semantics/
  development_v1.yaml`): 4 arms x seeds 10-14, 600 x 12 each, 10 cells x 100
  per seed, guide 11.5 gate; `--stage develop` (resumable, refuses any change)
  starts a live dashboard (`dashboard.py`, http://127.0.0.1:8773/;
  `--stage dashboard` read-only).
  **Result (2026-10-30, 8.3 h, local, status doc 14.4): the registered gate
  PASSED** (all 20 validated; composition-OOD R-GAT - MLP +1.8, - Untyped
  +2.4, - Raw +4.0 pp; ID +4.4, four-OOD +0.95, OOD unsafe -0.5 pp vs Raw).
  Read it as "superiority NOT established": per-seed R-GAT - MLP is
  [+9, -3, 0, +4, -1] pp (t 0.84), - Untyped [+6, -4, -3, +6, +7] (t 0.99);
  the held-out combinations are NOT harder (every arm scores composition-OOD
  >= its nominal ID, +0.6..+2.7 pp), so the composition margin equals the
  general ID margin (R-GAT - MLP +1.8 both) and does not isolate compositional
  generalization; at the dev mean/sd a 20-seed confirmation clears each
  97.5 % CI with p ~0.25 / 0.36. In the OOD cells R-GAT has the most SUCCESS
  and the fewest envelope/unsafe/timeout but the MOST MISSED_PAD_CONTACT
  (639 vs Raw 470) -- the no-re-acquisition failure. Combined-OOD is 6-8 %
  for every arm. Confirmation stage is not implemented; it and the Isaac
  11.6 evaluation each need the user's approval.
  **Isaac 11.6 evaluation `isaac-development/1` (user "3", flown 2026-10-30,
  status doc 15).** Task v5 now flies on Isaac: the LEARNER draws (A, B, r,
  a2) and sends them in the scenario name `spatial_current_omni/1:...`
  (`spatial/scenarios.py` encode/parse, gateway passes the prefix,
  `pad_motion.py`/`landing_world.py` apply it; entry offset -r(cos A, sin A),
  entry yaw world 0 +-5 deg because local yaw is always 0), and every reset
  ack is checked against the draw (`current_semantics/scenario.check_isaac_reset`).
  Found on the way: every earlier Isaac entry was pulled to ~centre by
  `constrain_camera_visible_entry` (median 0.07 m over 132 episodes vs the
  local +-0.4 m), so isaac-eval/1-/5 flew an easier entry than training;
  the omni path skips that pull. Plan `config/current_semantics/
  isaac_development_v1.yaml`: the 20 development best.pt x 30 cases
  (composition_id 994101-110, composition_ood 994111-120, combined_id
  994121-130) = 600 deterministic, environment-major, paired with a local
  reference on the identical cases; no pass criterion (guide 11.6.5).
  **Result: Isaac 324/600 (54 %) vs local 560/600 (93 %); per arm raw 82,
  mlp 81, untyped 78, rgat 83 of 150 -- indistinguishable** (R-GAT minus
  raw/mlp/untyped per training seed +0.7/+1.3/+3.3 pp, t <= 0.82). The
  ENVIRONMENT decides: 69 % of outcome variance is between the 30
  environments, 0.1 % between arms, 1.3 % between the 20 models; 14/30
  environments give all 20 models the same verdict. The Isaac-local gap
  grows with the seeded DR handover kick (U(-1, 1) m/s per axis, applied on
  both sides): |v_xy| < 0.6 Isaac 111/120 vs local 112/120, 0.6-0.9 133/260
  vs 237/260, >= 0.9 80/220 vs 211/220 -- no single threshold (1.22 m/s ->
  19/20, 0.87 -> 0/20). Failures: 100 handover tilt within 1.5 s, 87
  lost-pad descents below the pad plane ~4 m off, the same two paths as
  isaac-eval/1-/5; no representation changed either. Conditions cannot be
  compared on Isaac (environment-nested; composition_ood > id is two
  environments). Infrastructure: 3 SITL link failsafes + 1 mid-episode
  flight-clock stall; the owned-restart budget is RUN-WIDE (persisted in
  `isaac_stack/reset_recovery.jsonl`), so the default 2 was spent in the
  first run -- resume with `--reset-recoveries 5`; judged episodes are never
  re-flown. That gap turned out to be a SIMULATOR BUG, not plant or
  observation: see the next bullet.
- **ROOT CAUSE of the Isaac-local gap (2026-10-31, status doc 16): the
  handover kick reached PX4 clipped and, on most seeds, with the WRONG SIGN.**
  `landing_world._on_flight_state` wrote the DR handover kick as ONE 4 ms
  velocity step; `vn100_imu.py` differentiates velocity and clipped at
  +-16 g (0.63 m/s per axis per tick); PX4 SITL packs accel into an int16
  FIFO at g/2048 per count (`SimulatorMavlink.cpp`), so a value pinned at
  +16 g in FRD is +32768 counts and wraps to -16 g. That model predicts PX4's
  first-decision EKF velocity change on all 600 isaac-development episodes to
  a median 0.032 m/s per axis (p90 0.086) with 100 % of signs right; an ideal
  IMU misses by p90 1.49 m/s. Over the 700 current-semantics Isaac episodes:
  kick measurable 104/110 landed (95 %, 1 handover tilt), clipped only
  207/232 (89 %, 2), sign-inverted 61/358 (17 %, 112 of the 115 handover
  tilts); the local plant lands 90-94 % of each group. On the spatial route
  10 of the 11 handover tilts (/3-/11) were seed 12000, whose FRD y specific
  force is 158.1 > 156.9 m/s^2 (wraps); 12001 is measurable. Clip and kick
  both exist since 2026-09-11/12, so EVERY DR Isaac flight carries it. Read
  the /11 handover cap, the /12 thrust floor, pilot 5's
  `training_randomization.handover_own_state_error` (still in
  `spatial_view_contact_v1.yaml`, so in every development model) and the
  10.7 "EKF transient" as responses to this bug, not to a physical
  phenomenon. Fix (Isaac code only; local plant, contracts, checkpoints and
  profile hashes unchanged, `runtime_source_hash` changes as intended):
  `isaac_sim/handover_perturbation.py` spreads the kick over the fewest
  physics ticks that keep each inside half the accelerometer range (<= 4
  ticks, 16 ms, applied in the `_apply_wind` physics callback, which runs
  after Pegasus' state/IMU callbacks), and `sensor_profiles.saturate_16bit`
  makes the VN-100 saturate at +32767 counts in the FRD frame PX4 receives.
  Guard `tests/test_handover_perturbation_is_measurable.py`. **VERIFIED in
  flight 2026-10-31 (`--stage isaac-handover-fix --allow-isaac`, protocol
  `isaac-handover-fix/1`, status doc 16.5):** the identical isaac-development/1
  cases -- same seeds, conditions, registered checkpoints, deterministic -- on
  the first 10 wrapped environments by seed order (rule recomputed by the
  loader, guard `tests/test_isaac_handover_fix_protocol.py`), 40 episodes.
  Registered criteria, all passed: PX4 first-decision velocity sign right
  0/40 -> 40/40, per-axis error median 1.22 -> 0.028 m/s, handover tilts
  17 -> 0, landed 7 -> 38/40 (local reference 36/40; paired: 31 newly landed,
  0 lost). The two failures are touchdown-phase (vz -0.32, local identical;
  0.57 m off-centre). `isaac-handover-fix/2` (2026-11-01, status doc 16.6,
  `config/current_semantics/isaac_handover_fix_v2.yaml`, pass `--plan` as an
  ABSOLUTE path -- a relative one resolves against the launch directory) flew
  the other 20 environments, 80 episodes, all criteria passed: sign 80/80,
  error median 0.026 m/s, 0 handover tilts, 74/80 landed (pre-fix 57, local
  75). Over all 30 environments, 120 cases: Isaac 112 vs local 111 (pre-fix
  64) -- the isaac-development/1 Isaac-local gap is gone. Arms remain
  indistinguishable on every axis measured (final, composition-OOD, Isaac,
  and learning speed: d = 1.0 at a mean 98-109 updates per arm, training
  success 0.67-0.72), so fixing the measurement will not by itself produce a
  representation result.
- **Task v6 / representation redesign (2026-11-01, user "1 -> delete", "2. go";
  status doc 17): STOPPED by its own pre-declared rule.** It lives in
  `config/current_semantics/spatial_view_contact_v2.yaml` and the run root
  `results/current_semantics_3d/spatial_current_semantics_v1_task6`. It differs
  from v1 in three places only. Every v1, development/1 and Isaac-registration
  hash is byte-identical, guarded by `tests/current_semantics/test_cs3d_task6.py`.
  The CLI picks the root from the config's task and refuses the other task's
  root (`experiment.run_root_for`).
  - Change 1: `handover_own_state_error` is DELETED. A pilot-8/9 config that
    names it is refused.
  - Change 2: scenario/2 holds out ONE whole relative-geometry class B - A (a
    cyclic shift). Scenario/1 forbade this.
  - Change 3: the pilot protocol is g4-pilot/8, then /9.
  - Why: the v5 held-out orientations were interpolation, because the task is
    near rotation-equivariant.
  - Rule, declared before training: the held-out class must cost the Raw
    baseline >= 10 pp versus nominal.
  - Pilot 8 (fleeing B = A): G4 passed 46/48, 46/48. Raw scored 0.922 unseen
    vs 0.969 nominal, a gap of 4.7 pp: FAILED.
  - Pilot 9 (approaching, the one declared fallback): G4 passed 45/48, 46/48.
    Raw scored 0.984 unseen vs 0.938 nominal, i.e. EASIER: FAILED.
  - So development/2 was NOT registered, and no representation arm was trained
    on task v6.
  - Read it as: the common baseline already generalizes to unseen scenario
    combinations on this plant, so the composition axis cannot separate the
    arms. "Superiority not established", not "equivalent".
  - Removing the handover error left learnability unchanged.
  - In development/1, R-GAT - MLP has |t| < 1 in every cell. R-GAT - Raw is
    +4..+6 pp on the ID cells, which MLP - Raw shares, so that is the phi44
    features, not the graph.
- **Task v7 / coupling redesign (2026-11-02, user "a"; status doc 18): STOPPED
  by its own pre-declared rule.** `config/current_semantics/spatial_view_contact_v3.yaml`
  (scenario/3, `CouplingScenario`), root `..._task7`, guard
  `tests/current_semantics/test_cs3d_task7.py`. It is v2 plus one change: the
  pad starts 1.0-1.5 m off at 1.0-1.5 m/s, so tilt pushes the narrow image
  axis off the pad. The held-out cells are narrow-axis fleeing and wide-axis
  approaching. Isaac refuses scenario/3 (pad speed is not in the scenario name).
  - The rule (`development_design/calibration.py`, written before training)
    compares against a Raw trained on ALL 16 combinations on the same cells,
    not against nominal, because cells differ in difficulty.
  - G4 pilot 10 passed (35/48, 29/48 deterministic), a much harder task than
    v5 (nominal Raw 0.71 vs 0.94).
  - Held-out cells: reference 0.542, pilot 10 0.615, novelty gap -0.073
    against the required +0.10: FAILED. Development/3 was not registered.
  - Training-seed spread (reference 0.69 vs 0.40) exceeds any novelty effect.
  - Pilot 10's nominal-vs-composition gap of 13 pp is cell difficulty: the
    trained reference scores the same 0.448 there. The task-v6 rule would have
    misread it as a pass.
  - Across v5, v6 and v7, Raw never paid a measurable price for an unseen
    combination. Do not attempt a fourth composition axis without the user.
- **`relation-shuffle/1` (2026-11-02, user "3"; status doc 19):** guide 8.4's
  control. `current_semantics/relation_control.py` and
  `config/current_semantics/relation_shuffle_v1.yaml`, guard
  `tests/current_semantics/test_cs3d_relation_shuffle.py`.
  - What a shuffle is: the typed R-GAT with its 14 semantic edge labels permuted.
    It keeps the same pairs, counts, parameters and bit-identical initialisation;
    no typed edge set survives.
  - Ten graphs: 2 per training seed, shuffle seeds 7100 + 10s + k.
  - It is a paired fifth arm of development/1: same seeds and cells, compared by
    training seed with the STORED typed and untyped models. The current code
    reproduces their evaluation 24/24.
  - Arm names carry the graph hash (`ontology_rgat_shuffled_ppo:<seed>:<hash>`),
    so a shuffled checkpoint never loads as typed. `models.build_model` is the
    one factory and `pilot.py` uses it.
  - Rule: typed minus shuffled on mean success over all ten cells, 95 %
    t-interval over 5 seeds.
  - **Result 2026-11-02 (status doc 19.1): the rule says WORSE.** Typed
    minus shuffle-mean is -1.21 pp, per seed [-2.25, -1.5, -1.8, +0.1, -0.6],
    t(4) -2.85, CI [-2.39, -0.03]. All 10 shuffles reached d = 1.0 and their
    relational path is active. The evidence is marginal: the two shuffles of
    one seed differ by 3.5 pp on average (12 pp on seed 14), and a model-level
    SE gives z -0.55..-1.75. Quote it as "the semantic relation placement does
    not help the registered R-GAT arm", NOT as "semantics hurt". Shuffled
    minus untyped is +4.2 pp (t 1.89), not established.
- **ORL, ontology reinforcement learning (2026-11-02, user; status doc 20):**
  the ontology as the policy's inductive bias. `current_semantics/orl.py`, arm
  `orl_ppo:<ORL_HASH[:16]>`, guard `tests/current_semantics/test_cs3d_orl.py`.
  - Why it exists: the registered graph arms append an 8-D readout to raw21, so
    the head can ignore the graph. That is no bias at all, consistent with the
    noise-level arm differences.
  - ORL has four biases, each declared by the ontology:
    1. No raw bypass; typed node encoders over phi44 slots.
    2. The x and y planes share encoders, relation weights and the lateral
       readout, with y aligned to x (-eta, -sin roll, -roll rate).
    3. Exact Z2xZ2 mirror equivariance, by a group average over declared
       parities. These match `build_features` on reflected worlds exactly,
       except the half-pixel principal point (4.2e-3 on view entries).
    4. A learned descent gate: hold + sigmoid(DescentEligibility) * descend.
       Eligibility reads only Q, Contact_x/y and View_x/y.
  - Actor 2,747 parameters = the R-GAT arm's; total 4,016; common raw21 critic;
    no dead parameter.
  - It is NOT a registered v1 arm, and every v1 hash is unchanged.
  - `--stage orl-learnability` runs pilot 7's G4 protocol into `<root>/orl/`
    under its own lock. PASSED 2026-11-02 (status doc 20.2): test det 46/48,
    47/48, sampled 43/48, 43/48 (Raw pilot 7: 45, 41 / 43, 37); d = 1.0 at
    update 79/67 (Raw 99/96). Its `g4_hash` is the protocol's hash WITH the ORL
    arm (`run_pilot` swaps `g4.arm`), which `orl_development.pilot_blocker`
    must compare against; the first registration was refused on that.
  - `register-orl-development` / `orl-develop` (`orl_development.py`,
    `orl_development_v1.yaml`) pair ORL with the development/1 models.
    - Primary: ORL minus semantic_mlp_ppo, i.e. the same phi44 input and budget
      without the structure.
    - Registration is refused until the ORL pilot passes under the same ORL_HASH.
  - **Result 2026-11-02 (status doc 20.3): primary NOT ESTABLISHED.** ORL minus
    semantic_mlp_ppo on mean success over the ten cells is +2.2 pp, per seed
    [+8.6, +1.1, +9.4, -4.8, -3.2], t(4) 0.76, CI [-5.9, +10.4]. All five
    reached d = 1.0. Secondaries (never decisive): vs raw +3.9 (CI
    [-0.0, +7.7]), vs untyped +3.8 ([+0.2, +7.5]), vs rgat +0.9.
  - Exploratory, cells split after the fact, a hypothesis and not a result:
    - Where the pad stays visible (composition id/ood, motion_id, noise id/ood),
      ORL is the best of five arms, beating MLP on 24 of 25 seed-cell pairs
      (the 25th is 0).
    - Where it is lost (dropout, motion_ood, combined), ORL is equal or worse.
      Seeds 13/14 lose 13-27 pp there, which is what cancels the mean.
    - ORL has the fewest OOD unsafe contacts (26 vs 76-150), and its
      MISSED_PAD_CONTACT count is 627, against MLP 584 and R-GAT 639.
    - ORL reaches d = 1.0 at update 70 on average, against 98-109 for every
      development/1 arm. It is faster than all four arms on 4 of 5 seeds. This
      is the first consistent arm difference in the study, and the metric is
      not registered.
    - The losses in the lost-pad cells were then diagnosed (next bullet).
  - **Diagnosed 2026-11-03 (status doc 20.4, `orl/diagnostics_lost_pad/`;
    1600/1600 verdicts reproduced): the learned descent gate is INERT in every
    trained ORL model.**
    - Its value is constant per model: sd 0.004-0.027, flat across vision age.
    - hold(Z) and descend(Z) are collinear (cos 1.000), so the vertical output is
      a constant times one readout.
    - E does encode vision age (R^2 0.91-0.97). Its variance along the gate
      weight is 2-14 % of the total.
    - Cause: descend starts at x0.01, so the gate's gradient (proportional to
      descend(z)) is ~0 early, while hold and descend receive proportional
      gradients and align.
    - So orl-development/1 tested three biases, not four, and a "remove the gate"
      ablation would be a no-op.
    - dropout_ood loss mechanism: ORL descends faster while seeing the pad
      (0.39-0.60 vs MLP 0.29-0.36 m/s at 0.5-2 m). Seeds 13/14 keep sinking
      at 0.22-0.30 m/s blind and touch down 0.5-9 m off.
    - A truth-free what-if "brake vz when vision age > 0.5 s above 0.5 m":
      ORL 67.5 -> 85.8 %, MLP 73.8 -> 81.0 %.
    - motion_ood loss mechanism is different: an accelerating pad is lost
      0.6-0.9 m off at 0.4-0.6 m, and the vertical brake gives only +2-4 pp.
    - Next: ORL/2 with a gate that provably receives gradient and cannot
      collapse onto hold, checked before training.
  - **ORL/2 (user "yes", 2026-11-03; status doc 20.5).** It is
    `orl.ARMS["orl/2"]` = `orl_ppo:0d34c49904e549b3`; orl/1 stays buildable,
    hash-identical and loadable.
    - Vertical = `hold(Z) - sigmoid(gate(E)) * softplus(sink(Z))`.
    - VerticalDecision reads VerticalState ALONE: 2 edges removed, 59 edges
      left, same 4,016 parameters. Quality, contact and view therefore reach
      the vertical command only through the gate, and an open gate can only add
      descent.
    - Init: output weights x0.01 including the gate's; hold bias ln(2)/2
      cancels sigmoid(0)*softplus(0). The initial mean is ~0, and
      d(vertical)/d(gate bias) is -0.173 in every state (orl/1: +-0.0005,
      mixed sign). Guarded in `test_cs3d_orl.py`.
    - Each version owns a protocol and a directory
      (`orl_development.PROTOCOLS/OUTS`): orl/1 is orl-development/1 in
      `orl/`, orl/2 is orl-development/2 in `orl2/`.
    - The default plan is now `orl_development_v2.yaml`. To reach orl/1, pass
      `--plan <abs>/orl_development_v1.yaml`, which also applies to
      `orl-learnability`. The orl/1 registration hash is unchanged.
    - GATE PROBE (`orl_development.GATE_PROBE`, seeds 999101-999124, forced
      training blackout): a gate is active if mean gate on fresh decisions
      minus blind decisions is >= 0.2, both classes judged by the policy's
      OWN estimate above 0.5 m.
      - The orl/1 pilot checkpoints read as inert under it (drop -0.01 /
        -0.08).
      - orl-development/2 registers only after a G4 pass AND an active gate
        on both pilot checkpoints.
      - orl-develop probes all five development models; any inert gate means
        the verdict is not a test of the gate.
    - **Pilot result 2026-11-03 (status doc 20.6): G4 PASSED, gate probe
      FAILED, so orl-development/2 was NOT registered** (`pilot_blocker`
      refuses).
      - G4: det 47/48, 46/48; sampled 43/48, 43/48.
      - Gate drop: seed 0 0.756 (active), seed 1 0.174 (inert). Seed 1's
        best.pt is update 135; validation was already 1.00 there and the gate
        separated later (final.pt drop ~0.46). Checkpoint selection never sees
        the gate. Do not swap in final.pt after the fact.
      - Exploratory (`orl2/diagnostics_gate/`): the gradient fix works, but a
        CLOSED GATE DOES NOT STOP DESCENT. hold(Z) is unconstrained, and its
        learned equilibrium is a ~0.2 m/s sink: when blind, the total vertical
        command is ~0 net acceleration, and own vz stays -0.2 to -0.4 m/s.
      - Forced-blackout landings: 18-21/24, against orl/1's 20-21/24.
      - "Gate closed = hold altitude" must be built into the structure, e.g.
        hold = -softplus(k)*vz with its equilibrium at vz = 0. That is the
        proposed orl/3; the user decides.
  - **ORL/3 (user "go", 2026-11-03; status doc 20.7).** It is
    `orl.ARMS["orl/3"]` = `orl_ppo:80a8e23a39e9b0f4`, protocol
    orl-development/3, directory `orl3/`, plan `orl_development_v3.yaml`.
    orl/1 and orl/2 hashes, registration hashes and the orl/2 probe hash are
    pinned in the tests.
    - Vertical = `-softplus(gain(Z)) * (drone_vz + sigmoid(gate(E)) *
      softplus(sink(Z)))`. drone_vz is VerticalState's own phi44 slot. A
      closed gate is therefore a damper whose equilibrium is vz = 0 in EVERY
      state, whatever the weights, and the gate gradient is negative in every
      state.
    - Cost: the initial mean is a 0.091 m/s sink, not 0, because the gate
      starts half open with a full-open sink of 0.2 m/s. The sink is bounded;
      it does not drift.
    - `HOLD_GAIN_INIT = 3` was measured, untrained, with the gate closed, at
      d = 1 under the full seeded force (up to 0.5 m/s^2 vertical). Residual
      |vz| p90 was 0.39 / 0.14 / 0.11 / 0.092 / 0.089 m/s at gain 1-4, and
      jitter rose 7x at gain 4. A memoryless damper cannot cancel a constant
      force, so gain 1 would have let a closed gate sink at 0.35 m/s.
    - Gate probe 2 (`GATE_PROBES["orl/3"]`): probe 1 plus the policy's own
      mean vz on blind decisions >= -0.10 m/s. orl/2 seed 0 would fail it.
      Each seed's final.pt is reported but is never decisive.
    - **Pilot result 2026-11-03 (status doc 20.8): G4 PASSED, probe 2
      FAILED, orl-development/3 NOT registered.** Test det 47/48, 46/48,
      sampled 43/48, 40/48; d = 1.0 at update 65/66. The gate drop passed
      (0.35, 0.21), but the blind sink did not: vz -0.172 and -0.223 m/s
      against the -0.10 limit (final.pt -0.173 / -0.193).
    - Diagnosed (`orl3/diagnostics_gate/`): the STRUCTURE works. Forcing the
      gate shut on the probe's blind class (truth-free) lands 19/19/18/20 ->
      22/20/23/23 of 24 under forced blackout, and 44/43/42/45 -> 47/42/48/48
      of 48 deterministic in the training distribution.
    - PPO instead learns a HALF-open blind gate: 0.36-0.52 vs 0.69-0.79 fresh,
      with the gain rising 3 -> ~5 and the full sink 0.2 -> ~0.6 m/s. That
      gives a 0.15-0.23 m/s blind sink, about the reward field's 0.2.
    - Why PPO does not close it: under the distribution PPO optimises
      (training, SAMPLED), closing is worth only +1.9 discounted per episode
      (-1.2..+3.2, SE ~3). Most episodes lose a little time; rare ones save a
      miss (+65). Same lesson as pilot 6: a structure can make a behaviour
      expressible, but not optimal.
    - Options left to the user: stop the gate line; fix the gate as a
      non-learned ontology rule, paired with an MLP + same-rule control; or
      add a non-potential blind-descent cost to the COMMON reward and retrain
      every arm.
  - **ORL/4 + rule control (user "keep going" = option (b), 2026-11-03;
    status doc 20.9).**
    - The view rule (`view_rule.py`, `view-rule/1`) is reward /4's
      `lost_view_weight` evaluated on the policy's OWN estimate. It reads phi44
      raw slots only, is mirror-invariant, and equals the reward function to
      5e-7. Its weight w closes over vision age 0.5-1.0 s, but not below
      0.5 m within 0.75 m of the centre; w = 1 with no estimate. No constant
      was chosen: `check_config` refuses a config whose lost-view constants or
      contact height differ.
    - orl/4 (`orl_ppo:af6f4fb0994a5042`) is orl/3 with the gate times
      (1 - w): same parameters, and orl/3 weights load into it.
    - The control (`rule_control.py`, `semantic_mlp_rule_ppo:46967b8a9be973e4`)
      is the registered semantic MLP, init bit-identical, plus one learned
      gain: vertical = (1 - w) * mlp_z + w * (-softplus(g) * drone_vz).
    - orl-development/4 (`orl_development_v4.yaml`, plan 528727af, probe
      b4cc1976): the PRIMARY is orl/4 minus the CONTROL. Both arms need their
      own G4 pilot (`--orl-part control`, in `orl4/control/`) and probe 2
      before registration. `orl-develop --orl-part orl|control` runs one arm
      per process, each under its own lock.
    - Pre-check (`orl4/design_checks/`, rule laid onto existing weights): the
      rule stops blind descent in both arms (vz to ~0). Long blackout: orl/3
      17/15 -> 21/21; MLP seeds +0..+5 (one -1). Training distribution,
      sampled: orl/3 41/38 -> 43/43, discounted +1.4/+4.6.
    - **Pilots (status doc 20.10): both PASSED, and orl-development/4 was
      REGISTERED 2026-11-03 07:54 UTC.**

      | arm | det | sampled | d = 1 at | probe (blind vz) |
      |---|---|---|---|---|
      | orl/4 | 46, 45 | 44, 44 | 58/59 | active, -0.022/-0.024 |
      | control | 46, 46 | 44, 45 | 82/83 | active, -0.085/-0.024 |

      Both arms then train side by side. Whichever `orl-develop --orl-part`
      call finds every summary present writes the verdict; a call with no part
      is only needed if neither did.
    - **Result 2026-11-03 21:16 KST (status doc 20.11,
      `orl4/orl_development.json`): primary NOT ESTABLISHED.**
      - orl/4 minus the rule control, mean success over the ten cells:
        +3.94 pp, per seed [+6.9, +3.0, +5.3, +7.1, -2.6], t(4) 2.20,
        CI [-1.04, +8.92]. All ten models reached d = 1.0.
      - Gate precondition 9/10, so `reads_as_test_of_the_gate` is False. The
        exception is orl/4 seed 12: its drop passed, but its blind vz is
        -0.111 against the -0.10 limit.
      - Secondaries, never decisive and confounded: the stored arms have no
        rule. orl/4 minus MLP +6.5 [-1.1, +14.0]; minus R-GAT +5.1
        [+1.9, +8.4]; minus raw/untyped +8.1. The rule's own worth is control
        minus MLP, +2.5 [-1.4, +6.5].
      - Exploratory:
        - orl/4 has the highest ten-cell mean of any arm so far, 0.837 against
          the next best 0.798.
        - The rule fixed orl/1's lost-pad weakness: dropout_ood 0.650 -> 0.846,
          visible cells unchanged.
        - Seed 14 lost again on motion_ood (0.48 vs 0.64), the lateral loss of
          an accelerating pad, which no vertical rule touches.
        - orl/4 has the fewest OOD missed-pad contacts (369 vs control 448,
          MLP 584).
        - It reaches d = 1.0 at a mean update 66, against the control's 83.
      - A fresh-seed confirmation at the dev mean/sd has power 0.79 at 10
        seeds and 0.94 at 15; at half the effect, 0.29 and 0.43. The dev
        estimate comes from a design picked after four versions.
      - Next step is the user's: confirmation, a motion_ood design, or stop.
  - **ORL/5 (user "b" = the motion_ood design, 2026-11-04; status doc 20.12;
    `orl4/diagnostics_motion/`).** Built, tested (496 pass), NOT yet trained.
    - Diagnosed first, on 3000 re-flown episodes (verdicts reproduced 3000/3000). The
      motion_ood failure is common to all four arms, and it is not the lateral law:
      lateral gains are unchanged at OOD pad speeds and nothing saturates.
      1. While the pad accelerates, the shared track's pad-velocity estimate lags by
         0.37 m/s.
      2. The drone sits 0.85 m behind at ~1 m, sinking 0.4 m/s, and the pad leaves the
         narrow camera axis.
      3. view-rule/1's blind-touchdown exemption (below 0.5 m, estimated d < 0.75 m)
         then reopens descent on an estimate 0.6 m off, and the drone lands behind the
         pad. orl/4 9/122 re-acquired.
    - The field's alignment cone alone (r = 0.25 + 0.5 h) is too strong as the field's
      proportional scaling (ID -10..-30 pp, timeouts) and too weak as an edge only
      (+2..+5 pp).
    - What works is the field's own lost-view CLIMB (search height 2.5 m at 0.5 m/s),
      which view-rule/1 dropped. Laid onto the trained orl/4 and control weights it
      raises motion_ood and dropout_ood +6..+12 pp in both arms, with no ID loss.
    - Variant chosen by a rule written BEFORE its run (`selection_rule.txt`), on
      diagnostic seeds 995001-995048, NOT the verdict's environments: climb + cone 0.874
      vs climb 0.861, so `view_rule.ALIGNMENT_EDGE = True`.
    - view-rule/2 (`view_rule.rule_terms`):
      - W = max(w, clip((d - r)/0.25, 0, 1)).
      - The closed law tracks vz_ref = w * min(0.5, 1.0 * (2.5 - h)), and 0 with no
        estimate.
      - Every constant is the field's; `check_config_v2` refuses a mismatch.
    - orl/5 is `orl_ppo:c9e4f46f36a05004`: damper on B(vz - vz_ref), gate * (1 - W). It
      equals orl/4 where the rule is open, and orl/4 weights load into it.
    - The control is semantic-mlp-rule/2, `semantic_mlp_rule_ppo:3a696fc93f810588`.
    - orl-development/5: `orl_development_v5.yaml`, `<root>/orl5` (control in
      `orl5/control`), primary orl/5 - control v2, probe 2.
    - Every earlier hash is pinned in `test_cs3d_orl5.py`.
    - The rule is in both arms, so its gain is not the primary's answer.
    - User "go" 2026-11-04. Both pilots PASSED (status doc 20.13): orl/5 det 48/48,
      48/48, sampled 47/48, 45/48, d = 1 at update 55/56; the v2 control det 47/48,
      39/48, sampled 43/48, 35/48, d = 1 at 66/75. Probe 2 is active for every
      checkpoint, and blind vz is POSITIVE (+0.03..+0.19), i.e. the closed rule climbs.
      orl-development/5 was REGISTERED 2026-11-03 18:00 UTC (plan 87f71b5c, probe
      4f064274), and both arms are training in parallel.
  - The three biases went in together, so their contributions are not separated.
- **Study monitor** (user, 2026-11-02): `./run.sh current-semantics-3d --stage
  monitor` serves a read-only page at http://127.0.0.1:8780/
  (`current_semantics/monitor.py`). It shows every live study process, with its
  spawned workers folded in. It shows every training directory under the three
  task roots: running, stalled or done, with progress, ETA and charts. It also
  shows per-task G4, calibration, development and Isaac verdicts, the logs
  written in the last hour, and CPU/memory/GPU. It never writes, locks or
  starts anything; guard `tests/current_semantics/test_cs3d_monitor.py`. The
  development dashboard (port 8773) still follows one registered four-arm run.
- **Bare `./run.sh` (no arguments, or options only) now runs the WHOLE
  minimal-observation system** (user decision 2026-10-08,
  `scripts/run_minimal_system.sh`, doc section 16.3): build the two ROS
  packages in the ASCII workspace `~/.local/share/ontology_rgat_uav_rl/minimal_ws`,
  then `tools/minimal_full_pipeline.py` teacher -> BC -> PPO -> 48 held-out ->
  stress -> Isaac/PX4 (in-process and ROS-chain control), DR training and five
  seeds by default, run root `results/minimal_system_<ontology schema>`
  (resumes; a contract change starts fresh). It flies only when Isaac, ROS and
  the px4_msgs workspace are present and NO Isaac is already running; it never
  takes over. `--no-isaac` and `--dry-run` never touch a stack. Guard
  `tests/test_run_sh_system_route.py`. The paragraph below now describes
  `./run.sh all`, which is unchanged.
  Contract is `minimal-landing-ontology/6` (section 16): the terminal commit's
  lateral is flown by the supervisor (velocity match + dead reckoning +
  disturbance cancellation; the 0.5 m/s^2 policy cap lost to crosswinds), the
  pad-velocity window is 0.7 s with a 0.2 s span, and the teacher is kp 0.6 /
  kd 1.1 with the velocity boost only on fresh detections (a boost on a stale
  estimate was why a calmer clone out-landed it under poor vision). Teacher on
  seeds 5000-5095: combined 15.6 -> 51.0 %, poor vision 62.5 -> 84.4 %.
  DAgger (`./run.sh --dagger-rounds 2`, `results/minimal_system_ontology-6_dagger`,
  doc 16.5): the VECTOR arm after PPO goes 62.5 +- 10.3 -> 89.2 +- 4.6 % nominal
  (worst DAgger seed above best non-DAgger seed), so most of its gap to the graph
  arms was imitation distribution shift, not missing pad velocity; fast_pad still
  54.6 vs 89-92 %. Graph arms gain nothing (already at ceiling). Isaac 8/8, 66/67
  for this contract. The driver's BC stage is done only when the clone tool's
  `summary.json` exists: DAgger overwrites .pt files per round, and a run killed
  mid-round once left all 15 present and half stale.
- `./run.sh all` RUNS the spatial-reference local pipeline (`tools/run_full_pipeline.py`):
  behaviour-clone the teacher, train one cell per exploration/objective control
  from those same clones, evaluate, and write `summary.json`. It is
  long-running and writes under `results/`. It is resumable -- a stage whose
  summaries exist is skipped -- and `./run.sh all --dry-run` prints the plan
  without running it. Isaac/PX4 is opt-in behind `./run.sh all --isaac`,
  which flies only the cells that produced an eligible checkpoint and names
  the ones it skipped. The Isaac stage exits 2 when every episode flew but
  acceptance failed (one unsafe contact is enough: acceptance forbids any);
  the driver records that verdict from `isaac_acceptance.json` and still
  writes `summary.json`, instead of treating a measured result as a crash.
  The driver's final table reads every selected checkpoint, clones included,
  on 48 held-out seeds (`tools/compare_spatial_arms.py`); the in-run
  validation that SELECTS checkpoints is two seeds and cannot order arms. It is opt-in because `--stage isaac` refuses an
  ineligible checkpoint and a cell that lands 0 % never makes one, so flying
  by default would fail the run at its last step every time; and because it
  STARTS an Isaac Sim + PX4 SITL stack. That is not a takeover -- `live_stack`
  already refuses an Isaac that is already running unless `--adopt-stack`, and
  `--takeover` stays explicit on `isaac-legacy`. Help (`--help`), `status` and smoke still never acquire or stop a
  live stack.
  Because the bare route now works, NEVER execute `run.sh` just to inspect it:
  use `--help` or `all --dry-run`. A 5 s unit test invoked it bare once and
  started a behaviour-cloning job that outlived its own SIGKILL and ran to
  completion in the background; the guard is now
  `tests/test_run_sh_never_reaches_a_flight_stack_without_being_asked` plus
  `tests/test_full_pipeline_driver.py`. The run root holds a single-writer
  `pipeline.lock` naming the owning pid, so a second invocation refuses
  instead of interleaving artifacts, and a stale lock from a dead pid clears
  itself.
- **First learned-policy Isaac flight, 2026-10-06
  (`results/full_pipeline_20261006_lowsigma_kl/low_sigma_kl/isaac_*`, via
  `./run.sh all --cells low_sigma_kl --isaac --isaac-episodes 2`): 12
  episodes, 0 landings, 10 SAFE_ABORT at 12-14 s and 2 envelope violations
  at 0.9 s.** Acceptance fails on `no_unsafe_outcomes` and
  `proposed_landing_observed`; the stack itself booted and ran cleanly
  (relational path active, every stop confirmed). The SAME checkpoints land
  on the SAME seeds 12000/12001 in the local plant (rgat 828: SUCCESS at
  8.9 s and 10.8 s, mean |a_xy| 0.13-0.15). The gap is the plant, measured:
  fitting realized lateral acceleration against the commanded one over the
  first 1.5 s of every Isaac episode gives gain 0.97 at one 0.1 s step of
  lag (314 samples, corr 0.90); the local plant gives 0.66 at the same lag.
  `spatial/dynamics.py` (ATTITUDE_OMEGA 10, critically damped) reaches a
  commanded tilt in ~0.4 s where PX4 in Isaac does so within a decision
  step, so a policy tuned locally over-commands by ~1.5x in Isaac: the first
  lateral burst (2.1 m/s^2, normal locally) moves the vehicle 1 m in 1.2 s,
  the command reverses, tilt reaches 18-21 deg, and the pad leaves the FOV
  within 1.1 s; 3 s of loss then latches the abort. Do NOT read this as a
  policy or representation failure, and do not retune gains on the Isaac
  side: calibrate the local attitude/thrust response to Isaac's measured
  step response (or randomise it during training), snapshot the contract
  before and after, re-clone and re-fine-tune, then fly again. All 69
  Isaac terminals ever recorded under results/ contain 2 SUCCESS, both
  pre-policy teacher flights. **Second flight, on `/3`
  (results/full_pipeline_20261006_v3): 12 episodes, 0 landings, 0 unsafe**
  (9 SAFE_ABORT, 3 TASK_TIMEOUT). The envelope violations are gone; the
  vehicle now descends to ~1 m with the pad 0.9 m off-axis, the pad flickers
  at the FOV edge, the track is distrusted and the lateral loop limit-cycles
  at |a| ~0.8 with +-13 deg tilt until 3 s of loss latches the abort. Held
  out locally the same cell lands 60-83 % (graph arms). That measurement is
  what `/4` answers (bandwidth, gear, board). **Third flight, on `/4`
  (results/full_pipeline_20261006_v4): 12 episodes, 0 landings, 0 unsafe,
  9 TASK_TIMEOUT.** The board and gear did their job: detection 100 % for
  the whole 70 s, estimate error 5-13 cm, no visual-loss aborts. What is
  left is a closed-loop difference: the supervisor's `vertical_stopping_margin`
  rewrote the requested descent on 94-100 % of steps because the terminal
  corridor never armed, and it never armed because the policy held
  |a_xy| 1.5-1.8 m/s^2 with 1.0-1.2 m/s^2 step-to-step changes (tilt <= 5 deg
  on 6-23 % of steps) where the SAME checkpoint on the SAME seeds locally
  holds 0.45-0.69 with 0.11-0.21 changes, arms the corridor on 53-84 % of
  steps and lands in 6-12 s. Same scenario, different output, so a different
  input: evaluation traces now carry the normalized packet
  (`trace(..., packet=...)`), and the next step is a channel-by-channel
  Isaac-vs-local comparison before anything is retrained. Suspects, in
  order: EKF own-velocity smoothing (FIR [0.32, 0.43, 0.22] against the
  truth's [-0.02, 0.74, 0.24]), body-rate noise, and an optical transport
  age that varied 100-400 ms between episodes against the 100 ms trained.
  RESOLVED the same evening: the channel was `detectionConfidence` (see the
  `/5` bullet); own-velocity smoothing and body-rate noise were consequences
  of the oscillation, not causes. **Fourth flight, on `/5`
  (results/full_pipeline_20261006_v5, `./run.sh all --cells low_sigma_kl
  --isaac --isaac-episodes 2`): 8 of 12 episodes SUCCESS -- the first
  learned-policy landings in Isaac in this project.** `ppo_ontology_rgat`
  seed 828 landed 2/2 (11 s), seed 829 1/2; `ppo_semantic_flat` 2/2 and 1/2;
  `ppo_vector_canonical` 2/2 and 0/2. Acceptance still FAILS, on
  `no_unsafe_outcomes` alone: one SAFETY_ENVELOPE_VIOLATION at 0.9 s
  (`ppo_vector_canonical` 828, the weakest arm, tilt past 21 deg right after
  handover) and one UNSAFE_CONTACT at 9 s (`ppo_ontology_rgat` 829).
  `landing_observed`, `proposed_landing_observed`, `proposed_relation_active`
  and every stop confirmation pass. Held out locally on 48 seeds the cell
  lands rgat 83.3 / 89.6 %, flat 91.7 / 97.9 %, vector 72.9 / 95.8 %, with
  unsafe 0-27 %. Twelve Isaac episodes with two unsafe contacts is a small
  integration pass, not robustness and not a representation claim.
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
