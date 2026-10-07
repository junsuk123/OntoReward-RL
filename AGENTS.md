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
- The active spatial contract is `spatial-reference/11` (`REFERENCE_SCHEMA`,
  2026-10-07, profile v12, no legs). It is `/10` plus ONE law, the HANDOVER
  LIMIT, on the user's instruction to fix the handover tilt that alone failed
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
- Bare `run.sh` RUNS the whole local pipeline (`tools/run_full_pipeline.py`):
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
