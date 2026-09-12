# Documentation map

The default experiment is the controlled three-pipeline comparison launched by
the repository-root `./run.sh`. Documents explicitly labeled **legacy** describe
retained secondary experiments and must not be used to infer the primary actor,
ontology, reward, budget, or output schema.

## Start here

| Document | Use it for |
|---|---|
| [Project README](../README.md) | installation, one-command run, outputs, and concise technical contract |
| [System overview](SYSTEM_OVERVIEW.md) | component map and end-to-end execution order |
| [Three-pipeline comparison](THREE_PIPELINE_COMPARISON.md) | scientific controls, equations, metrics, and budgets |
| [Operations](OPERATIONS.md) | monitoring, restart semantics, logs, and fault diagnosis |
| [Architecture](ARCHITECTURE.md) | detailed simulator, PX4, ROS, coordinate, timing, and migration boundaries |
| [Shin-2026 baseline](SHIN2026_BASELINE.md) | paper-to-code correspondence and intentional adaptations |
| [Data-flow audit](DEPENDENCY_DATAFLOW_AUDIT.md) | privileged-information exclusions and executable guards |
| [Hardware safety](HARDWARE_SAFETY.md) | mandatory real-vehicle constraints and opt-ins |
| [References](REFERENCES.md) | primary integration and algorithm references |

## Primary facts at a glance

| Item | Current primary value |
|---|---|
| Pipelines | `shin_se`, `no_se`, `onto_no_se` |
| Actor input | 512×320 mono image + body velocity 3 + quaternion 4 |
| Shared temporal model | frozen six-keypoint encoder, 512-unit LSTM, 256-D latent |
| Actor latent input | `y[6:256]` + 7-D proprioception |
| Action | heading-frame `vx, vy, vz, yaw_rate` |
| Primary ontology | 18 nodes, 35 directed edges, 4 relations, 24 features/node |
| R-GAT | two 24-wide relation-attention layers, direct frozen `Phi(G)` |
| Default scene | Meta-Sejong S5 / `gwanggaeto` |
| UGV route/speed | 37-point, 99.70 m closed road loop; 0.25–0.60 m/s draw |
| Default budget | 8 warm-up + 264 PPO × 3 = 800 training flights |
| Reward-design data | 40 real flights minimum; both outcomes and successful visual recovery required; 120 hard cap |
| Evaluation | 7 scenarios × 5 seeds × 3 pipelines |
| Dashboard | `http://127.0.0.1:8770/` |
| Runtime logs | `/tmp/ontology_rgat_stack/` |
| Results | `results/three_pipeline/<mode>/` |

## Current visuals

### Isaac Sim flight

![Live UAV and landing UGV in the Meta-Sejong S5 scene](images/isaac_sim_s5_live.png)

This is an actual Isaac Sim viewport from the current full run. The blue debug
line/trail is operator telemetry; it is not an input to the policy.

### Live monitor

![Live MATLAB-style dashboard](images/live_dashboard_status.png)

This is an actual in-progress full run captured on 2026-09-12. It is included
to document observability only; it is not a final benchmark figure.

### Audited S5 route

![Meta-Sejong S5 road and UGV waypoints](images/metasejong_gwanggaeto_ugv_route.png)

The plot comes from `tools/check_metasejong_route.py` reading the current YAML
and the licensed USD mesh. It is therefore evidence for the configured route,
not an illustrative campus sketch.

## Legacy material

The following retained files or figures describe the older cooperative urban
pipeline:

- `scripts/run_metasejong_pipeline.sh`;
- `python/run_pipeline.py` and `python/run_shin2026_pipeline.py` when invoked
  through legacy reward-arm CLI;
- 23-channel actor observations;
- 14-node/38-edge ontology graphs;
- eight R-GAT-distilled fixed reward coefficients;
- `system_architecture-metasejong-v3.png`, `rgat_network-v4.png`,
  `reward_function-v3.png`, and `rl_observation_state-v3.png`.

They remain for reproducibility but are not images of the primary
`onto_no_se` direct-R-GAT implementation.

The [retired MATLAB tree](../legacy_matlab/README.md) is reference-only. The
workspace checker fails if active Python, simulator, ROS, or launcher code
begins to depend on it.

## Documentation maintenance rule

Configuration values are authoritative in `config/`; behavior is authoritative
in `python/ontology_rgat/`, `isaac_sim/`, the ROS gateway, and `scripts/`.
Generated result manifests are authoritative for an individual run. When prose
and executable code differ, update the prose and record the run's configuration
hash rather than treating an old screenshot or legacy diagram as truth.
