# Two-axis context R-GAT refactor handoff

> Historical v1 handoff. Superseded 2026-10-04 by
> [REFERENCE_V28_AUDIT_KO.md](REFERENCE_V28_AUDIT_KO.md).
> This file does not describe current reference defaults or Isaac parity.

Date: 2026-10-02  
Primary experiment: `two_axis_context_rgat_v1`  
Algorithm version: `two-axis-context-rgat-v1`

This document reports an implementation and bounded integration validation. It
does not report PPO convergence, an ontology advantage, or real-flight safety.

> **Superseded in part on 2026-10-03.** The environment contract was revised to
> `planar-visibility-ppo-v2.5`. The supervisor, terminal table, goal cost,
> action limit, hard envelope and contact plane described below have all
> changed, and a training-only curriculum was added. Read
> [`FINAL_REPORT.md`](FINAL_REPORT.md) first; the sections here describe the
> structure that survived, not the constants.

## System contract

```text
seeded CV-CA-CV pad + planar drone state
        |
        v
body-fixed camera measurement + synchronized own state
        |
        v
shared timestamped causal pad tracker + task/safety memory
        |
        +----------------------+-----------------------+
        |                      |                       |
 canonical packet MLP   semantic-node flat MLP   typed context graph R-GAT
        |                      |                       |
        +----------------------+-----------------------+
                               |
               separate policy and value encoders
                               |
                raw Gaussian command u in R^2
                               |
                tanh -> requested [ax, az]
                               |
              common safety supervisor/limits
                               |
           pitch/thrust inner loop -> planar physics
```

The PX4 compatibility adapter may produce the live gateway's historical third
field, but that pitch setpoint is derived from `[ax, az]`; the policy never
samples or stores it as an action.

## Environment and scenario

World `x` is forward and `z` is upward. `y`, `vy`, roll, and yaw are exactly
zero. The state is `[x, z, vx, vz, theta, pitchRate, collectiveThrust]`.
Positive pitch tilts thrust toward `+x`.

The target is continuous CV-CA-CV motion:

- phase 1: independently sampled `v1, T1`;
- phase 2: independently sampled positive `a2, T2`;
- phase 3: independently sampled `T3`, with `v3 = v1 + a2*T2`;
- no independently sampled `v3`, velocity jump, or hidden fourth phase.

Duration and spatial lengths with units are persisted in disjoint seeded
train/validation/test/stress manifests. The stress manifest uses an initial
height outside the training interval while retaining duration and final-speed
feasibility checks.

The default dynamics use 0.01 s physics steps and 0.1 s held policy actions.
`theta_sp = atan2(ax, g+az)` and `T_sp = m*hypot(ax, g+az)` feed a damped
second-order pitch response and first-order thrust response. Actual pitch and
thrust drive translation and sensing. Gravity is subtracted once.

## Observation and estimation

`config/observation/causal_packet_v2.json` is the single ordered registry for
all arms. Its 26 scalar channels are grouped as follows:

| Group | Fields |
|---|---|
| own motion | `h, vx, vz, sinTheta, cosTheta, pitchRate` |
| pad track | `exEstimate, relativeVxEstimate, padVxEstimate, padAxEstimate`, three uncertainties, initialization mask |
| visibility | detection, measured bearing + mask, confidence, age, predicted bearing and FOV margin |
| task memory | remaining mission time, previous 2-D action, landing inhibit, abort request |

The constant-acceleration Kalman tracker receives timestamped measured relative
position and own position only. Missing measurements propagate the prior and
grow uncertainty. Reacquisition is innovation-gated. Updates are idempotent per
timestamp, and finite differences use actual measurement gaps. Phase IDs,
switch times, target commands, hidden current pad state, future events, reward,
and outcomes are not packet inputs.

## Action, safety, and termination

The actor samples `u_raw ~ Normal(mu, diag(sigma^2))`. PPO stores and evaluates
the Gaussian probability of that same raw sample. `tanh(u_raw)` maps to
normalized two-axis action, then to limits `ax=1.2 m/s^2` and `az=2.0 m/s^2`.
The supervisor logs policy-requested and applied accelerations separately.

All methods use one causal supervisor. It clips only at declared physical
limits, inhibits descent when track confidence or vertical stopping margin is
insufficient, latches abort after 3 s of visual loss, and applies bounded
braking/hold. An abort becomes `SAFE_ABORT` only after a feasible non-contact
hold; contact and hard violations retain their actual outcomes.

First pad-plane contact is interpolated within the physics step. Mechanical
contact uses pre-impact position, relative longitudinal/vertical speed, actual
pitch, and pitch rate. Authorization is separate from mechanical safety.
Terminal reasons are `SUCCESS`, `SAFE_ABORT`, `UNSAFE_CONTACT`,
`UNAUTHORIZED_CONTACT`, `MISSED_PAD_CONTACT`,
`SAFETY_ENVELOPE_VIOLATION`, and `TASK_TIMEOUT`. All are task terminations with
zero bootstrap. Only an external collection boundary is truncation and uses a
final-value bootstrap.

## Reward

The common reward is

```text
r_t = terminalBonus(event)
      - (dt_actual / 70) * (1.0*c_goal + 0.1*c_view + 0.02*c_control)
```

`c_goal = zeta/(1+zeta)`, where
`zeta=(ex_true/3)^2+(h_true/4)^2`. Truth is isolated to simulator
training/evaluation reward code and never reaches a deployable input.
`c_view` is squared measured bearing normalized by half-FOV, or one if the
measurement is invalid. `c_control = 0.5*sum(a_norm_policy^2)` regularizes the
requested policy action rather than claiming an energy measurement.

Terminal values are +10 success, -3 safe abort, -12 timeout, and -40 unsafe,
unauthorized, missed-pad, or envelope-violation outcomes. The terminal is paid
once. There is no remaining-horizon absorption multiplier, learned reward
weight, PBRS potential, pitch-magnitude penalty, or reward for ontology nodes.
The structure adapts the separation between action scaling, dense approach
reward, and terminal outcomes in the user-provided Choi et al. 2025 paper; it
does not copy that paper's DDPG architecture, thrust equation, weights, or
single-axis task.

`config/audits/two_axis_reward_audit_v1.json` contains exact discounted returns
for ten deterministic design fixtures. These are reward-hacking checks, not
policy results.

## Context graph and policy inputs

The graph has nine semantic nodes plus `PolicyNode` and `ValueNode`. Query
nodes contain identity/bias constants before message passing.

| Node group | Nodes |
|---|---|
| measurement/estimate/state | `PadVisibility, PadMotion, DroneTranslation, DroneAttitude, RelativeTracking` |
| bounded engineering context | `TrackingCorrection, ViewRecovery, DescentEligibility, LandingInhibit` |
| readout | `PolicyNode, ValueNode` |

Relations are `informs`, `affects_visibility`, `supports`, `inhibits`,
`contributes`, and `self`. The graph uses 17 declared semantic edges, 18
semantic-to-query contribution edges, and 11 self-loops. Inverse edges are not
added. Every semantic node reaches both queries. Supports/inhibits and the C01
to C12 scores are bounded engineering priors, not logical proof, action labels,
future predictions, or supervised attention targets.

The three arms are information-controlled:

- `ppo_vector_canonical`: shared low-level packet;
- `ppo_semantic_flat`: exactly the graph node tensor flattened, with no edges;
- `ppo_ontology_rgat`: exactly the same node tensor with two relation-aware
  layers, residual connection, and separate policy/value query readouts.

Policy and value R-GAT encoders are independent and trained end-to-end by the
clipped PPO policy objective and GAE-return value regression respectively.
Stored graphs are re-encoded during updates.

## Migration table

| Previous contract | Current primary contract |
|---|---|
| three learned channels `[a_fwd, a_z, tilt]` | two learned channels `[ax, az]`; pitch derived by inner loop |
| stepped or ramped target-speed profiles | exact continuous CV-CA-CV with derived `v3` |
| image/keypoint packet v1 and assorted policy inputs | named causal packet v2 shared by all arms |
| 9-node reward-era or selective schemas | 11-node compact decision-context/query graph |
| Shin five-term, two-term capture/distance, FOV-risk, PBRS, adaptive weights | three bounded running costs plus one terminal outcome |
| frozen selective base and behavior cloning | independent actor/value encoders trained by PPO, no BC by default |
| tracking mode could affect landing outcome | physical contact and mission authorization evaluated separately |
| fixed-step gamma/done mask | physical-time discount; terminated vs truncated bootstrap contract |
| legacy checkpoints accepted by matching dimensions | strict algorithm/registry/action/graph/config signature rejection |

Historical configs and results were not deleted. They are explicitly
reproduction-only in `config/experiments/README.md`.

## Task-by-task implementation record

| Task | Implemented in | Result |
|---|---|---|
| T00 audit/preserve | this handoff | clean `main` at `fbbdcd5`; one worktree; reviewed MATLAB commit absent locally; no work discarded or process stopped |
| T01 contracts/config | `two_axis/config.py`, `contracts.py`, packet v2 JSON | typed config, provenance registry, strict signature |
| T02 trajectory | `two_axis/scenario.py`, manifests | continuous seeded CV-CA-CV and distance helper |
| T03 dynamics/camera | `dynamics.py`, `sensing.py` | lagged pitch/thrust, actual-attitude projection |
| T04 causal memory | `estimation.py`, `contracts.py` | timestamped tracker, uncertainty/masks, no truth input handle |
| T05 safety/contact | `safety.py`, `environment.py` | physical contact, authorization, abort, earliest substep event checks |
| T06 reward | `reward.py`, `reward_audit.py`, audit JSON | bounded pure terms, terminal once, fixture returns |
| T07 ontology | `ontology.py` | typed deterministic graph, context scores, matched flat features |
| T08 PPO | `models.py`, `learning.py` | independent encoders, raw likelihood, time-aware GAE, finite update |
| T09 validation | `test_two_axis_context_experiment.py`, smoke CLI | bounded regression and all-arm smoke only |
| T10 docs/migration | READMEs, `AGENTS.md`, this handoff | primary and legacy paths separated |

## Commands actually run

```bash
pytest -q tests/test_two_axis_context_experiment.py
PYTHONPATH=python python python/run_two_axis_experiment.py --smoke --steps 4 --seed 7
PYTHONPATH=python python python/run_two_axis_experiment.py --smoke --steps 8 --seed 7 --ppo-minibatch
PYTHONPATH=python python python/run_two_axis_experiment.py \
  --manifest-dir config/manifests/two_axis_context_v1 \
  --reward-audit config/audits/two_axis_reward_audit_v1.json
```

The dedicated suite passed 17 tests. The full repository suite passed 1052
tests with 2 skips and 15 pre-existing dependency/Matplotlib warnings. The smoke run produced finite rewards,
finite graph embeddings, finite PPO losses/gradients, and no normalized action
saturation in its short sample. These checks establish integration only.

## Remaining scientific and deployment risks

- No multi-seed training or held-out evaluation has been run. Sample
  efficiency, landing rate, visibility recovery, and graph benefit are unknown.
- Dynamics, camera noise, tracker covariance, touchdown tolerances, and safety
  braking are provisional simulation values, not identified vehicle parameters
  or a safety certificate.
- The context thresholds and support functions need validation-only sensitivity
  analysis before they are frozen for a final study.
- Capacity differs among the three initial encoders. The full study must report
  parameter counts and include the configured capacity-matched MLP check.
- The new environment is in-process. Live Isaac/PX4 execution requires using
  the compatibility adapter and validating timing/measurement provenance at
  that boundary before any hardware consideration.
- Stress fixtures currently cover out-of-range initial height. Additional
  named kinematic and sensor-model stress strata should be added without using
  proposed-policy performance as a rejection criterion.
