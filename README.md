# Ontology–R-GAT–RL Autonomous Landing

Dynamic-wind, moving-UGV landing research stack for NVIDIA Isaac Sim,
Pegasus, PX4 SITL, ROS 2, ontology-based R-GAT reward design, and PPO.

![Ontology–R-GAT–RL autonomous landing system](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/system_architecture-metasejong-v3.png)

The repository contains one active implementation under
[`Ontology_RGAT_UAV_RL_ISAAC_PX4/`](Ontology_RGAT_UAV_RL_ISAAC_PX4/). The
subproject README is the complete installation, configuration, experiment, and
troubleshooting guide:

**[Open the full project README](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md)** ·
[System overview](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SYSTEM_OVERVIEW.md) ·
[Operations](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md) ·
[Architecture](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ARCHITECTURE.md)

## Research contract

The system models the two dynamic conditions central to the experiment: a
seeded turbulent wind field measured by an onboard anemometer, and a road-going
AGILEX RANGER MINI 3.0 carrying a moving landing pad. The UAV sensor suite is
profiled as ZED-F9P RTK GNSS, VectorNav VN-100 IMU, and one ZED 2i mono eye.
Policy observations contain only measurable
sensor/estimator values; simulator truth is isolated to reset, terminal reward,
and evaluation scoring.

R-GAT learns context-dependent safe-landing structure from a 14-node ontology.
Its counterfactual attributions are distilled once per execution into eight
bounded reward coefficients that sum to one. Those coefficients are saved and
frozen before PPO, making the proposed reward auditable and identical throughout
training and evaluation.

An experiment passes only when both independent criteria pass:

- **Reward effectiveness:** nominal landing success is at least 60%.
- **R-GAT consistency:** cross-condition success standard deviation is at most
  15 percentage points, worst-case success is at least 35%, and R-GAT validation
  MSE is at most 0.35.

## One-command run

From the repository root:

```bash
./run.sh
```

The root entry point defaults to the publication-scale Shin/OntoReward run. It
starts DDS, Isaac Sim, Pegasus, PX4, the ROS gateway and MATLAB-style dashboard;
prepares the frozen controlled R-GAT reward; trains the recurrent PPO arms;
runs paired scenario evaluation; and exports checkpoints, CSV tables,
confidence intervals and publication figures. It is independent of the current
working directory and forwards all benchmark options:

```bash
./run.sh --mode quick --headless
./run.sh --methods shin2026 sparse manual_no_active ontoreward ontoreward_plus_active --headless
./run.sh --help
```

The second command runs every ablation arm. Use
`Ontology_RGAT_UAV_RL_ISAAC_PX4/scripts/run_metasejong_pipeline.sh` directly for
the separate legacy 23-channel cooperative urban experiment.

## System figures

| System | R-GAT and fixed weights | Reward contract | PPO observation/state |
|---|---|---|---|
| [Architecture](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/system_architecture-metasejong-v3.png) | [R-GAT](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/rgat_network-v4.png) | [Reward](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/reward_function-v3.png) | [Observation](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/rl_observation_state-v3.png) |
