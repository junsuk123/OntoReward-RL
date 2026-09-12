# Learning remediation audit (items 1, 2, 4, 5, 8, 9)

This audit records the sequential changes requested after the September 2026
live run. It distinguishes executable evidence from the Shin et al. paper:
the paper defines the algorithm; the figures below are from this repository's
Isaac/Pegasus/PX4 implementation and are not paper results.

## Baseline observation

The stopped checkpoint was preserved. Comparing policy episodes 9–28 with
49–68 showed position RMSE increasing from 2.89 m to 3.37 m, FOV loss from
58.1% to 68.6%, final lateral error from 6.31 m to 9.23 m, and zero landings.
Every sampled active-perception term was clipped at `-0.1`. These measurements
justify repair rather than continuing the same checkpoint.

## Sequential evidence

| Item | Defect and change | Verification/effect |
|---:|---|---|
| 1 | A global `tanh` capped physical relative state at ±1. The first six latent channels are now unbounded normalized coordinates, decoded to metres and m/s; loss uses `[3,3,8,3,3,2]` scales. | A deterministic test produces 6 m and 12 m estimates that the old model could not represent. One-scale error on every axis yields balanced loss 1.0. Checkpoint format v3 rejects old semantics. |
| 2 | Raw six-state MSE saturated the active reward and removed its causal gradient. Live reward now consumes the same normalized loss as the auxiliary estimator. | A representative error changes from clipped `-0.1000` to responsive `-0.0135`. Mean normalized loss, saturation fraction, and signal standard deviation are persisted per episode. |
| 4 | A deadline override forced 80 curriculum levels into 264 episodes (roughly one increase every three episodes), independent of performance. | Forty consecutive failing fixtures now remain at level 1; a level advances only after rolling success, FOV, and the same truth-side relative-position RMSE gate for every arm pass. |
| 5 | The frozen encoder used only synthetic board projections. The launcher now collects real unannotated Isaac frames, labels six pad landmarks through detected board-plane homography, reserves a held-out split, fine-tunes, and retains only the best validation state. | Initial live audit: 23.2 px RMSE and 29.2% PCK@20. Calibrated settings reached 16.8 px and 79.2% PCK@20 on the same held-out split (27.6% RMSE reduction). Full mode recollects 48 frames under the current config. |
| 8 | Table-II samples existed only in a unit test. Each episode now applies the seed to PX4-relative controller gain spread, Isaac force/torque, one handover velocity/rate perturbation, and the actual actor camera's texture/scale/brightness/RGB/light transform. | Range, deterministic serialization, and PX4 mapping tests pass. The inherited fixed urban wind is disabled in this benchmark so it is not double-counted. Applied values are returned in reset acknowledgement and recorded in episode CSV. |
| 9 | The old health gate detected only the conjunction of no success and high FOV loss. | Independent gates now detect battery-terminal rate after 20 episodes and, after a 40-episode grace, no success, high FOV, stalled position RMSE, and stalled active-reward saturation. Fixtures identify all intended causes. |

## Live integration check

A fresh one-episode quick run was executed in
`results/remediation/stage8_9` and stopped after the first atomic checkpoint.
It is calibration evidence, not benchmark evidence:

- held-out Isaac keypoint RMSE: 24.46 px before, 18.41 px after;
- PCK@20: 20.8% before, 75.0% after;
- normalized six-state loss mean: 0.593;
- active-reward saturation: 0%, with standard deviation 0.0216 (the stopped
  baseline was 100% saturated with no active-reward variation);
- unsuccessful warm-up retained curriculum level 1 (`advanced=0`);
- reset acknowledgement reported domain randomization enabled, 0.945 N force,
  0.00460 N·m torque, texture 6, and brightness 0.759.

One warm-up episode cannot establish landing-rate or RMSE convergence. Those
claims require the complete run and remain guarded by the new health checks.

## Interpretation

These checks prove that the former structural blockers are removed; they do
not pre-claim a landing success rate. The next valid evidence is a fresh v3
checkpoint trained by `./run.sh --mode full`. The dashboard shows normalized
estimation loss and active-reward saturation; a health-gate stop is a useful
failed experiment, not a reason to reuse an incompatible checkpoint.
