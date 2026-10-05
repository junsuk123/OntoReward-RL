# Two-axis landing contract — V2.5 environment corrections

> Historical report, not current status. The 2026-10-04 gate-release training
> completed, and the Isaac pilot is no longer running. Later sections and older
> reward constants below reflect different experiment revisions. Current evidence,
> v2.8 corrections, bounded spatial Isaac integration and unverified learned
> landing performance are in
> [REFERENCE_V28_AUDIT_KO.md](REFERENCE_V28_AUDIT_KO.md).

Date: 2026-10-03
Primary experiment: `two_axis_context_rgat_v1`
Algorithm version: `two-axis-context-rgat-v1`
Contract revision: `planar-visibility-ppo-v2.5` (ported from the reduced 2-D
MATLAB study, [ugv_landing_2d_workspace](https://github.com/junsuk123/ugv_landing_2d_workspace))

**`performance_verified = false`.** This report covers environment and reward
corrections, their regression tests, and a non-learned feasibility reference.
No PPO training run has been executed under the corrected contract, so no
learned-performance claim is made for any of the three arms.

## Headline

| Metric | PN (non-learned) | Baseline PPO | Semantic-flat PPO | Ontology R-GAT PPO |
|---|---:|---:|---:|---:|
| Landing rate | **0.75** | pending | pending | pending |
| SAFE_ABORT | 0.10 | pending | pending | pending |
| TASK_TIMEOUT | 0.10 | pending | pending | pending |
| Unsafe outcome | 0.05 | pending | pending | pending |
| FOV capture | 0.86 | pending | pending | pending |
| Mean landing time | 16.2 s | pending | pending | pending |
| Inference time | not measured | pending | pending | pending |
| Training time | N/A | pending | pending | pending |

PN: 20 held-out validation seeds (2000–2019) at difficulty 1.0, the nominal
4–8 m distribution, with sensor perturbations on. Its gains were selected on
train seeds 1000–1023 only. Before these corrections the same controller, the
same seeds and the same envelope produced a **0.00** landing rate.

## 1. Observed behaviour

Three symptoms, measured over 30 seeds with a causal proportional tracker on
the shared estimator (`scratchpad/diag_baseline.py`, pre-change):

- no episode ever reached `SUCCESS`;
- 22 of 30 contacts were mechanically inside every touchdown limit — median
  longitudinal error 2.4 cm, median vertical speed 0.21 m/s — and were still
  scored `UNAUTHORIZED_CONTACT` at −40;
- 8 of 30 sampled scenarios drew a pad acceleration the vehicle could not
  match.

## 2. Direct cause supported by data

`landing_inhibited` was true at **30 of 30** contacts.

The supervisor inhibits landing whenever
`stopping_margin = z − stopping_distance < minimum_abort_hold_height_m`. With
no descent rate, `stopping_distance = 0` and `stopping_margin = z`. Every state
below 0.5 m therefore inhibited landing regardless of tracking quality, and
contact happens at 0 m. Authorization was unreachable by construction, so the
`SUCCESS` terminal (+10 at the time) was never paid to any policy and PPO had
no variation in the terminal term to learn from.

Two reward defects compounded it, both reproduced from the shipped audit
fixtures rather than inferred:

- `SAFE_ABORT` (−3) outranked `TASK_TIMEOUT` (−12). In the shipped
  `two_axis_reward_audit_v1.json`, "unnecessary induced loss followed by
  abort" returned **−2.74** while "stable near-pad hovering until deadline"
  returned **−4.43**. Inducing a visual loss was the cheapest way out of a hard
  episode, and the regression test asserted that ordering.
- The goal cost merged both axes into one saturation,
  `zeta = (e_x/3)² + (h/4)²`, `goal = zeta/(1+zeta)`. At 8 m altitude the
  altitude term dominates and a metre of longitudinal drift is nearly free, so
  descent is learned before tracking.

## 3. Root cause

The envelope and the outcome table were specified independently of the
geometry they had to admit.

1. **Touchdown authorization.** The abort-hold margin is the right rule for an
   approach and the wrong rule for a flare; nothing replaced it below 0.5 m.
2. **Contact plane.** Contact required the vehicle reference point to cross
   exactly 0 m, with no landing-gear plane, so the last centimetres were a
   region the supervisor braked and the classifier did not recognise.
3. **Actuation authority.** `ax_max = 1.2 m/s²` was set below the sampler's own
   `a2` maximum of 1.5 m/s², making some scenarios untrackable by any
   controller.
4. **Hard envelope frame.** `|x| > 60 m` bounded the vehicle in the *world*
   frame while the pad travels up to ~385 m downrange, terminating well-flown
   long episodes. Seven of 24 PN episodes died this way at 7–14 cm altitude
   while tracking the pad to within 2 cm.
5. **Abort semantics.** `abort_latched` was never cleared and
   `maximum_backup_duration_s` was computed but never used, so a 3 s loss was
   an irreversible terminal rather than a bounded recovery.

## 4. Modified files and changes

| File | Change |
|---|---|
| `config/experiments/two_axis_context_rgat_comparison.yaml` | `ax_max` 1.2→2.5; terminal table SUCCESS 10→25, SAFE_ABORT −3→−15; `horizontal_priority: 0.65`; `readiness: 0.5`; gear plane, terminal-descent corridor and commit window; curriculum block |
| `two_axis/config.py` | `CurriculumConfig`; `touchdown_height_m`, `terminal_descent_height_m/_speed_margin/_commit_s`; `horizontal_priority`, `readiness_weight`, `unsafe_reasons`; validators `_validate_actuation_authority`, `_validate_terminal_ordering`, `_validate_touchdown_reachability` |
| `two_axis/reward.py` | Per-axis saturation mixed by `horizontal_priority`; `landing_readiness` paid as a difference; `terminal_ordering_report` |
| `two_axis/safety.py` | `touchdown_gate_open` (footprint-tight gate), bounded flare commit, abort recovery/expiry, pad-relative hard envelope, gear-plane `interpolate_contact` |
| `two_axis/environment.py` | Difficulty-aware `reset`; authorization follows the supervisor alone; SAFE_ABORT on window expiry; readiness tracking; richer `info` |
| `two_axis/curriculum.py` | **new** — difficulty schedule, replay roles, promotion on active-difficulty outcomes, checkpoint eligibility, split landing rates |
| `two_axis/comparator.py` | **new** — PN feasibility reference with the height-staged approach gate |
| `two_axis/diagnostics.py` | **new** — episode recorder, altitude-plateau detector with upstream attribution, outcome summariser |
| `run_two_axis_experiment.py` | `--pn-reference`, `--curriculum-preview` |
| `tests/test_two_axis_context_experiment.py` | 17 → 32 tests |

### Corrections that were *not* made

The reduced 2-D study found its estimator was over-gained and amplified
position noise into velocity and acceleration oscillation. **That defect does
not reproduce here** and the gains were left alone. A sweep over
`process_acceleration_std ∈ [0.08, 0.8]` and `measurement_position_std ∈
{0.03, 0.05}`, replayed on recorded measurement streams
(`scratchpad/diag_est_sweep.py`), showed the shipped `q = 0.8` is at or near
the optimum on every axis; lowering it monotonically worsened position,
velocity and acceleration error. On fresh detections the filter holds 0.023 m
position RMSE against 0.03 m sensor noise. The residual 0.33 m/s² acceleration
RMSE is structural: a constant-acceleration model cannot track the
instantaneous `a2` step at `T1` and `T1+T2`. Changing the gains would have
traded a real improvement for a copied one.

## 5. Regression-test results

`pytest tests/test_two_axis_context_experiment.py` — **32 passed**. New tests
cover, one correction each: authorized-touchdown reachability, the
footprint-tight gate, the bounded flare commit, abort recovery and expiry, the
gear contact plane with its 0 m fallback, the pad-relative envelope,
longitudinal authority over every sampled `a2`, terminal ordering, the
horizontal gradient at altitude, readiness-as-a-difference,
difficulty-1.0-equals-nominal, promotion excluding replays, cross-arm episode
identity, PN feasibility and non-teaching, and plateau attribution.

Full-repository suite: **1070 passed, 2 skipped** (1059 passed before these
changes; the 11 new tests are the difference). No pre-existing test was
weakened. Two were corrected because they asserted the defects:
`test_primary_configuration...` pinned `ax_max == 1.2`, and
`test_reward_is_bounded...` asserted `SAFE_ABORT` ranks above `TASK_TIMEOUT`.

### Ontology and R-GAT structural audit

`python python/run_two_axis_experiment.py --graph-audit`, 1,430 graph samples
from eight nominal-difficulty rollouts:

- **No dead node.** All nine semantic nodes carry 5–7 active feature channels
  with nonzero variance. All twelve behaviour-case scores fire; the three
  rarest (`C03`, `C05`, `C11`) are the visual-loss cases and appear in ~5 % of
  steps, which is how often a loss happens.
- **One duplicate row,** `PolicyNode == ValueNode`. Both carry bias and
  identity only; they differ in the identity one-hot and in which role's
  encoder reads them. This is the intended design, not a redundancy.
- **Twelve channels are constant 1.0** in these rollouts: the hardcoded
  `DroneTranslation.valid/confidence` and `DroneAttitude.valid/confidence`,
  plus the `valid` channels that follow `trackInitialized`, which stays 1 once
  the pad has been seen. The four hardcoded ones can never vary and act as
  per-node biases.

**No node or relation was removed.** Dropping the four constant channels would
change `GRAPH_SCHEMA_HASH`, invalidating the checkpoint signature and the
capacity-matched control's input width, in exchange for removing four inputs
that behave as biases. There is no measurement showing that helps. The finding
is recorded here so the decision is explicit rather than implied by silence.

Attention sparsity — whether a typed relation's attention collapses to zero —
is **not** inferable from features and needs a trained encoder. It stays open.

## 6. Bounded diagnostic results

```
PN, 20 held-out validation seeds, difficulty 1.0
landing 0.75   safe_abort 0.10   timeout 0.10   unsafe 0.05
fov_capture 0.86   supervisor_intervention 0.085
plateau 0.05 (1 episode, attributed to horizontal_tracking)
mean landing time 16.2 s   mean terminal |e_x| 0.039 m
```

The one remaining plateau is attributed upstream to horizontal tracking, which
is the intended behaviour of the detector: it names the link to repair rather
than the supervisor that reacted to it.

## 7. The learning pipeline, and four blockers found by running it

A full PPO pipeline now exists for this contract:
`python/run_two_axis_pipeline.py`, with stages `tune`, `train`, `evaluate`,
`aggregate`. It trains every arm from a random initialisation, applies the
shared curriculum, selects checkpoints, scores them on two held-out splits and
re-asserts the fairness claim from what each run recorded. There is no
behaviour cloning, no demonstration collection, and no PN supervision: the
collect/train design of the legacy Isaac/PX4 driver is incompatible with the
no-imitation constraint and is not used.

Two defects in the bounded `ppo_minibatch_update` helper made it unusable for
a real run and are fixed in the new `PPOTrainer`: it built fresh `Adam`
optimisers on every call, discarding the moment estimates each iteration, and
it took a single full-batch step, so the sampling ratio was exactly 1 on the
only step taken and the PPO clip never engaged.

Running it surfaced four blockers, each identified by measurement and each
fixed in a way that leaves the nominal contract untouched.

**7.1 The easy curriculum began with the pad outside the frame.** The start
offset was a fixed 2 m. At the nominal 4-8 m that is 14-27 degrees from nadir,
inside the 35 degree half-FOV. At the 1.2-2.5 m curriculum start it is 47-59
degrees, outside it: the pad was geometrically invisible at t=0 in **0 of 30**
episodes, the tracker never initialised and the supervisor latched an abort
immediately. The offset is now `min(2.0, 0.5 * h)`, which is exactly 2.0 for
every height at or above 4 m, so the nominal case is bit-identical.

**7.2 The curriculum's easiness axis was inverted.** Measured with the PN
reference, 24 seeds per cell:

| training distribution | PN landing rate |
|---|---:|
| nominal | 0.42 |
| low altitude only (1.2-2.5 m) | **0.04** (abort 0.92) |
| slow pad only (v1 0.3-0.8, a2 0.1-0.4) | **0.71** |
| slow pad + low altitude | 0.12 |

The camera footprint is `0.7 * h`, so a lower approach shrinks the frame and
makes tracking harder, not easier. The ported prescription from the 2-D study
("begin at a low altitude") is wrong for this camera geometry. The curriculum
now holds altitude at nominal and interpolates the pad kinematics instead,
which produces a monotone ladder: PN 0.62 at difficulty 0 against 0.42 at 1.0.

**7.3 The terminal table created an exploration deadlock.** With SUCCESS +25,
TASK_TIMEOUT -12 and unsafe -40, attempting a landing beats hovering to the
deadline only once the success probability already exceeds

    p* = (|TIMEOUT| + |unsafe|) / (SUCCESS + |unsafe|) = 28 / 65 = 0.43

A policy starting at zero can therefore never discover the landing, and PPO
duly resolved it the other way: over 360 iterations it drove SAFE_ABORT from
0.67 down to 0.005 and settled at **97% TASK_TIMEOUT with zero landings** --
a correct solution to the problem as posed. The curriculum's unsafe penalty
now starts at -12, equal to TASK_TIMEOUT, where `p* = 0`, and ramps to the
nominal -40. SAFE_ABORT moved to -22 to keep enough gap for the dense terms.

**7.4 The readiness signal was zero over the whole approach.** It was a
product of clipped gates, one of which clipped on
`terminal_descent_height_m = 1.0`, so readiness was exactly 0 above 1 m and
the approach carried no shaping at all. The gates are now smooth
(`1/(1+(x/tol)^2)`), giving a monotone profile from 0.016 at 8 m to 1.0 at the
gear plane while still collapsing under misalignment. The dense weights rose
to `goal 4.0, view 0.5, readiness 3.0`; their total of 7.52 stays below the
smallest adjacent terminal gap of 10, which the ordering validator enforces.

**Not changed, because the data did not support it:** the initial exploration
scale and the injected sensor dropout. A sweep of `initial_log_std` over
{-0.7, -1.2, -1.8, -2.5} left landings at ~0 and made aborts worse, and
removing the 3.2 s sustained dropout entirely still gave zero landings. Both
were plausible causes and neither was the cause.

### First learned results

After these fixes, 250-iteration tuning runs produced the first nonzero
learned landing rates at nominal difficulty on held-out tuning seeds:
`ppo_semantic_flat` 0.167, `ppo_vector_canonical` 0.125, with several trials
still at 0.000. That is a short run at one eighth of the planned budget and
high variance; it is evidence that the pipeline learns, not a comparison
between arms.

The full study -- 3 arms x 3 seeds x 2,000 iterations, then both held-out
splits -- is the next step and its results are not in this document.

## 7b. Switching to the Isaac/PX4 stack (2026-10-03)

The study was moved onto the live Isaac Sim / PX4 SITL stack. The V2.5
contract was **not** ported wholesale: each correction was first checked
against what the Isaac configuration actually does, and most of them turned
out not to apply.

| V2.5 correction | Isaac status |
|---|---|
| terminal ordering SUCCESS > TIMEOUT > failure | already correct: +50 / -5 / -10 |
| exploration deadlock | absent: break-even attempt probability is 5/60 = **0.083**, against 0.43 on the planar contract |
| dense approach shaping | already present: `lateral_progress` is a progress difference with weight 5.0 |
| terminal replaces shaping | already correct |
| actuation authority over the deck | adequate: 1.6 m/s longitudinal against a deck capped at 1.0 m/s (`benchmark_speed_scale: 0.125`) |
| authorized touchdown reachable | reachable: the privileged teacher lands about one flight in three |
| curriculum eases **platform motion** | already the right axis -- `curriculum_motion_scale` scales deck motion, which is what the planar measurement independently identified as the axis that makes the task easier |

### The defect that was there: the ladder cannot be climbed

The inherited curriculum is Shin et al. Fig. 3, 80 performance-gated levels.
Under the gate a level costs at least `minimum_episodes_at_level = 20`
episodes, so the ladder costs at least `79 x 20 = 1,580` episodes. The
experiment's budget is `training.episodes_full = 1,000`.

    levels needed to reach c = 1 .... 80
    episodes that requires ......... 1,580
    training budget ................ 1,000
    best reachable level ........... 51 of 80  ->  c = 0.633, deck motion 76%
    realistic level (gate at 20%) .. near 1    ->  c = 0.000, deck motion 35%
    evaluation flies ............... c = 1.000, deck motion 100%

Both arms were therefore scored on a deck neither had trained against, and
nothing in a run reported it: the curriculum level is recorded, but no check
compares it to the budget or to the evaluation difficulty. With behaviour
cloning now disabled the gap widens, because the warm start that used to carry
the early episodes past the 20% success gate is gone.

**Fix.** The planar experiment declares `curriculum.levels: 40`, which needs
`39 x 20 = 780` episodes and leaves ~220 episodes of training at full platform
motion rather than merely touching it on the last episode. The gate itself is
unchanged and identical for both arms, and the paper's 80-level ladder stays in
the base config it reproduces.

**Guard.** `assert_curriculum_reaches_nominal` refuses any combination of
levels, per-level minimum and budget that cannot climb its own ladder, and
names the largest level count that fits. Four regression tests cover it,
including the shipped 80-level setting as the failure it was.

### Behaviour cloning is off

The inherited pipeline warm-starts PPO from
`privileged_relative_state_velocity_pd_v4`, a PD controller reading the
simulator's true relative state, and its own comments note that "PN guidance is
the teacher as well as the control condition". Imitation learning, a PN
teacher and privileged simulator state in a policy are all excluded by this
experiment's constraints, so `behavior_cloning.enabled: false` is declared in
both the smoke and pilot configs and honoured at `run_three_pipeline.py:1915`.

PPO from scratch has never been run on this stack, which is why a pilot
precedes the multi-day study.

### Bring-up verified

A bounded smoke with behaviour cloning off ran end to end: stack boot, keypoint
encoder fine-tune (**PCK@20 6.8% -> 89.2%**, landmark recall 0% -> 100%),
training flights on both arms, checkpoint selection, evaluation, artifacts.
Measured throughput was 20 episodes per hour per pair, matching the figure the
repository documents.

It also explains part of the 2026-10-02 failure in which the stack ran 20 h
without writing an episode: after the stack reports ready there is a ~4 minute
window with no output at all while the keypoint encoder is fine-tuned on CPU,
which is indistinguishable from a hang from outside. That does not account for
20 hours, so the full cause of that run remains unexplained; what the smoke
establishes is that the stack is not fundamentally broken.

## 8. Remaining unverified claims

- `performance_verified = false` for all three learned arms.
- No claim that the ontology graph state helps; no learned run exists.
- Unused nodes, permanently-zero edges and attention sparsity in the context
  graph have **not** been audited; that audit needs rollouts from a trained
  policy.
- Inference time, parameter-count-matched comparison at convergence, and
  Monte-Carlo mean/1-sigma trajectories are not measured.
- PN gains were tuned on 24 train seeds; 20 validation seeds is a small sample
  and the interval around 0.75 is wide.
- The curriculum is implemented and tested but has never driven a training run,
  so its promotion threshold and window are untested against real learning
  dynamics.

## 9. Reproduction commands

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
export PYTHONPATH=python

# contract + regressions
pytest -q tests/test_two_axis_context_experiment.py

# non-learned feasibility reference on held-out validation seeds
python python/run_two_axis_experiment.py --pn-reference --pn-seeds 2000 2020

# difficulty schedule, including the difficulty-1.0 == nominal identity
python python/run_two_axis_experiment.py --curriculum-preview

# reward-hacking fixtures (regenerates the shipped audit)
python python/run_two_axis_experiment.py \
    --reward-audit config/audits/two_axis_reward_audit_v1.json

# bounded integration smoke across all three representations
python python/run_two_axis_experiment.py --smoke --steps 32 --ppo-minibatch

# structural ontology audit over real rollouts
python python/run_two_axis_experiment.py --graph-audit
```

A running training process keeps the configuration it loaded at startup. Any
process started before this revision must be stopped and restarted; its results
must not be mixed with V2.5 results.
