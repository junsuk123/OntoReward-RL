# Ontology–R-GAT–RL autonomous landing

Vision-only recurrent PPO for landing a PX4 multicopter on a road-following
UGV in NVIDIA Isaac Sim, with an estimator-free ontology/R-GAT reward variant.

![Live Isaac Sim flight over the Meta-Sejong S5 road](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/isaac_sim_s5_live.png)

![Live MATLAB-style three-pipeline dashboard](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/live_dashboard_status.png)

> Both screenshots are real runtime captures from the full pipeline on
> 2026-09-12. They demonstrate the simulator and monitoring path, not a
> completed benchmark result.

## Run the complete pipeline

From this repository root, the final entry point is:

```bash
./run.sh
```

The bare command runs `--mode full` with the deadline/seminar budget:

- 8 estimator warm-up flights for `shin_se`;
- 264 PPO flights for each of `shin_se`, `no_se`, and `onto_no_se`;
- 40 real estimator-free reward-design flights initially, automatically
  extended to at most 120 if both terminal classes are not yet present;
- 5 paired evaluation seeds for each of 7 scenarios and each pipeline
  (105 evaluation flights).

This is exactly 800 training flights before the separate reward-design and
evaluation flights. It is a preview-scale experiment, not publication-scale
evidence. Useful alternatives are:

```bash
./run.sh --mode quick --headless
./run.sh --mode full --pipelines shin_se no_se onto_no_se
./run.sh --mode full --training-replicate 1 \
  --train-episodes 40960 --rgat-data-episodes 400
./run.sh --help
```

Only one launcher may own the flight stack. A second `run.sh` exits before it
can reset the vehicle or modify results. Compatible checkpoints and completed
CSV rows resume automatically.

## What is compared

| Pipeline | State-estimation supervision | Active-perception reward | Ontology reward |
|---|---:|---:|---:|
| `shin_se` | yes, six-state auxiliary MSE | yes | no |
| `no_se` | no | no | no |
| `onto_no_se` | no | no | frozen direct R-GAT PBRS |

All three share the same 512×320 mono camera, frozen six-keypoint encoder,
512-unit LSTM, 256-D latent, `y[6:256]` actor slice, 7-D UAV proprioception,
4-D velocity/yaw-rate action, PX4 controller, PPO settings, curriculum, and
paired seeds. Simulator truth is isolated to the asymmetric critic, reset,
terminal labels, and physical evaluation.

The primary ontology is **13 nodes, 25 directed edges, 4 relation types, and
19 features per node**. It consumes keypoint/heatmap semantics, UAV motion and
attitude, and onboard battery reserve. It does not accept relative-state
estimates, UGV state, GNSS, or simulator truth. The trained R-GAT output is
frozen and used directly as `Phi(G)`:

```text
r_t = r_sparse + lambda * (gamma * Phi(G_t+1) - Phi(G_t))
```

The older 14-node/38-edge, 23-channel cooperative urban experiment and its
distilled fixed reward weights remain available only as a labeled legacy path.

## Runtime and monitoring

`run.sh` starts or adopts DDS (UDP 8888), Isaac Sim/Pegasus/PX4, the ROS 2
gateway (UDP 14650), RViz 2, and the dashboard at
<http://127.0.0.1:8770/>. The dashboard reports committed episodes separately
from the active episode and per-step telemetry, so a long rendered flight does
not look frozen.

Recoverable SITL transport, simulated-clock, and pure Offboard-heartbeat
interruptions discard only the partial trajectory, restart the stack owned by
the launcher, and retry the same seed. Geometry, perception, estimator, and
policy failures remain hard failures and are not hidden by retry.

## Meta-Sejong S5 environment

The default benchmark uses the Gwanggaeto/S5 campus asset and a closed 37-point
road route. The route is 99.70 m long; the offline mesh audit measured 1.00 m
of conservative clearance after the 1.5×1.5 m deck footprint and a maximum
waypoint elevation error of 0.001 m.

![Audited Meta-Sejong S5 UGV route](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/metasejong_gwanggaeto_ugv_route.png)

## Documentation

The implementation lives under
[`Ontology_RGAT_UAV_RL_ISAAC_PX4/`](Ontology_RGAT_UAV_RL_ISAAC_PX4/).

- [Complete project guide](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md)
- [Documentation index](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/README.md)
- [System overview](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SYSTEM_OVERVIEW.md)
- [Controlled comparison](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_PIPELINE_COMPARISON.md)
- [Operations and fault diagnosis](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md)
- [Architecture and interfaces](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ARCHITECTURE.md)
- [Paper-to-code baseline](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md)
- [Hardware safety gate](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/HARDWARE_SAFETY.md)

Run repository checks from the active project directory:

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/check_workspace.sh
```
