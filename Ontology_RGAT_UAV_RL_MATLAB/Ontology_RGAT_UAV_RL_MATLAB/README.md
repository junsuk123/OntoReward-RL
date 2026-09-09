# Ontology-RGAT Reward Shaping for 6-DOF UAV Landing

MATLAB research prototype for the follow-up direction of **“외란 환경에서의 온톨로지 기반 인공지능 드론 자율 착륙 의사결정 프레임워크”**.

The project extends the paper's `Drone State / Wind State / Landing Pad Observation -> semantic relations -> landing decision` idea into:

`distributed physics -> ontology graph -> R-GAT potential -> PBRS reward -> PPO landing control`.

## What is implemented

- 6-DOF rigid-body translation/rotation with quaternion attitude.
- Four first-order rotor/motor states and physical wrench allocation.
- Fuselage + arm **surface-panel aerodynamic model**.
- Every panel samples a different local 3-D wind vector.
- Atmospheric density/viscosity and Reynolds-dependent skin-friction approximation.
- Spatial wind shear, Fourier turbulence, localized gust packets, and a finite-core vortex.
- Paper-aligned semantic features: wind risk, visual stability, alignment, attitude stability, touchdown safety.
- Explicit ontology/KG schema with typed relations.
- Two-layer custom **relation-aware graph attention (R-GAT)** potential model.
- Potential target learned from signed discounted future safe/unsafe terminal outcomes.
- Potential-Based Reward Shaping (PBRS): `r_sparse + lambda*(gamma*Phi_next - Phi_now)`.
- Custom continuous-action PPO written with Deep Learning Toolbox only; Reinforcement Learning Toolbox is not required.
- Hand-weighted dense reward PPO baseline.
- Common-random-number paired Monte Carlo evaluation.
- Real-time training monitor and trajectory/disturbance monitor.
- Publication plots, CSV metrics, paired approximate 95% CIs, and per-panel force visualization.
- Deterministic physics sanity tests.

## Requirements

Recommended: MATLAB R2025a/R2025b or newer with **Deep Learning Toolbox**.

The code intentionally avoids Simulink and Reinforcement Learning Toolbox so the algorithmic pipeline can be inspected file-by-file by a coding agent.

## First run

```matlab
cd('Ontology_RGAT_UAV_RL_MATLAB')
setup_path
run_quick_smoke_test
```

Then run the integrated pipeline:

```matlab
run_all
```

`run_all.m` defaults to `defaultConfig('quick')`. For a paper-scale run change it to:

```matlab
cfg = defaultConfig('full');
```

Full mode intentionally performs many Monte Carlo episodes and can take hours depending on CPU/GPU and panel resolution.

## Real-time simulation after training

```matlab
run_realtime_demo
```

The live monitor displays 3-D trajectory, XY error, local wind components, tilt, reward and distributed aerodynamic resultant.

## Main outputs

Generated under `results/`:

- `models/rgat_model.mat`
- `models/ppo_manual.mat`
- `models/ppo_rgats_pbrs.mat`
- `comparison_results.mat`
- `summary_metrics.csv`
- `episode_metrics.csv`
- `paired_difference_ci.csv`
- `wind_generalization_sweep.csv`
- `figures/training_curves.png`
- `figures/evaluation_summary.png`
- `figures/representative_trajectory_3d.png`
- `figures/proposed_disturbance_response.png`
- `figures/rgat_relation_attention.png`
- `figures/panel_surface_loads_peak.png`
- `figures/wind_generalization_success.png`

## Baseline vs proposed

### Baseline
PPO with a manually tuned dense reward:

`position + velocity + tilt + angular rate + wind risk + action effort + terminal reward`.

The arbitrary coefficients are all exposed in `cfg.reward.manual` so the reward-design problem is visible rather than hidden.

### Proposed
The task reward is deliberately sparse. The ontology constrains allowable semantic relations and the R-GAT predicts a goal potential `Phi(G_t)`. PPO receives potential-based shaping from the change in that learned potential.

The R-GAT training labels are derived from **future terminal safe/unsafe outcomes**, not from the baseline manual reward.

## Fidelity statement

This is a high-detail **distributed panel + rigid-body model**, not CFD. It resolves different aerodynamic forces/moments over many body panels and local wind variation, but does not solve Navier-Stokes flow around the rotating propellers and body.

For publication-level physical validation, replace the clearly marked approximate parameters with CAD/URDF/SDF and experimental identification values. See `docs/MODEL_ASSUMPTIONS.md`.

## Agent-oriented navigation

- `config/`: all tunable experiment/physics parameters.
- `src/+dynamics/`: 6-DOF equations and RK4.
- `src/+aero/`: surface panels and aerodynamic loads.
- `src/+wind/`: spatial/temporal 3-D wind field.
- `src/+semantic/`: paper-aligned semantic features and ontology graph.
- `src/+rgat/`: relation-aware attention potential model and explanation.
- `src/+reward/`: baseline/sparse/PBRS rewards.
- `src/+training/`: dataset generation, R-GAT training and custom PPO.
- `src/+evaluation/`: paired evaluation, CSV outputs and figures.
- `src/+viz/`: live and post-hoc visualization.
- `validation/`, `tests/`: deterministic model checks.
- `docs/`: equations, assumptions, experiment protocol.

Read `PROJECT_TREE.md` for the complete tree and `docs/THEORY.md` for the equation-to-file mapping.
