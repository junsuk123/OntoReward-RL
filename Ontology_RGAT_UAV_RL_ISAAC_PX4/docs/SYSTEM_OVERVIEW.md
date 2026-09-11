# System overview

This document is the compact source map for the Meta-Sejong autonomous landing
experiment. Exact values remain authoritative in `config/`, and executable
behavior remains authoritative in `python/ontology_rgat/`, `isaac_sim/`, and
the scripts under `scripts/`.

![End-to-end system architecture](images/system_architecture-metasejong-v3.png)

## Objective and experimental boundary

The UAV must land on a moving UGV pad in a dynamic environment that combines
wind, visual-marker intermittency, GNSS degradation, vehicle motion, and limited
battery reserve. Isaac Sim/Pegasus owns physics, contacts, scenery, cameras,
wind, and synthetic sensors. PX4 owns the flight stack. ROS 2 and the versioned
gateway expose measurable state to the Python environment.

The policy may observe estimator output, camera-derived pad pose, measured wind,
UGV broadcast motion, battery state, and receiver self-assessment. Simulator
truth is a separate scoring path used only for reset acknowledgement, terminal
reward, and evaluation. It is never appended to the PPO observation or ontology
input.

## Runtime and learning flow

| Stage | Operation | Principal output |
|---:|---|---|
| 0 | Validate configuration, dependencies, assets, and reproducibility contract | preflight report |
| 1 | Start/adopt Isaac Sim, PX4, DDS, gateway, dashboard, and RViz | ready runtime |
| 2 | Fly seeded episodes and append ontology graphs/outcomes | R-GAT dataset |
| 3 | Resume and train the 14-node relational graph-attention model | R-GAT checkpoint/history |
| 4 | Measure counterfactual attribution and project it to a bounded simplex | frozen reward design |
| 5 | Resume/train PPO with the hand-designed baseline reward | manual PPO checkpoint |
| 6 | Resume/train PPO with the fixed R-GAT-distilled PBRS reward | proposed PPO checkpoint |
| 7 | Run paired nominal evaluation and wind/pad/GNSS/energy strata | metrics and dual-gate decision |
| 8 | Export tables, plots, dashboards, and run provenance | run summary and figures |

One command runs the complete sequence:

```bash
./scripts/run_metasejong_pipeline.sh --mode full
```

The script must be invoked from the active project directory shown above. From
the Git repository root, first run `cd Ontology_RGAT_UAV_RL_ISAAC_PX4`.

## Ontology and R-GAT

![Ontology R-GAT and fixed reward-weight pipeline](images/rgat_network-v4.png)

Every state becomes a graph with 14 semantic nodes, 38 directed edges, four
relation types, and 18 features per node. Two 24-wide relational attention
layers learn a bounded safe-landing outcome potential. Attention and output are
context dependent during R-GAT training; they are diagnostic evidence, not a
claim of physical causality.

After R-GAT training, stage 4 neutralizes one physical term at a time and
measures the mean absolute output change over the accumulated dataset. The eight
sensitivities are projected onto the configured bounded unit simplex. The result
is one fixed, interpretable coefficient for each term:

`position_error`, `vertical_speed`, `tilt`, `angular_rate`, `wind_risk`,
`pad_tracking`, `energy_risk`, and `navigation_risk`.

The artifact records the weights, normalization ranges, raw and signed
sensitivities, dataset size, R-GAT validation error, training epochs, and a
deterministic design ID. The weights sum to one and stay frozen throughout PPO
training and evaluation.

## Reward and success contract

![R-GAT-optimized fixed reward design](images/reward_function-v3.png)

The proposed reward combines fixed sparse task semantics with potential-based
shaping:

```text
Phi_w(s)   = -sum_i w_i * normalized_cost_i(s)
r_proposed = r_sparse + 2.0 * (0.999 * Phi_w(s') - Phi_w(s))
```

The sparse constants are `+10` for confirmed successful touchdown, `-10` for a
failed terminal event, timeout, or battery depletion, and `-0.15` per control
step. At an absorbing terminal state `Phi_w(s') = 0`. PPO never mutates or
re-queries R-GAT to change the coefficients online.

Evaluation deliberately separates two claims:

| Gate | Meaning | Default acceptance |
|---|---|---|
| Reward effectiveness | The fixed optimized reward produces successful nominal landings | success rate `>= 0.60` |
| R-GAT consistency | The distilled design remains stable across dynamic conditions and the R-GAT fit is adequate | success std `<= 0.15`, worst case `>= 0.35`, validation MSE `<= 0.35` |

`overall_pass` is the logical AND of the two gates. A uniformly failing policy
cannot pass by being consistent, and high nominal success cannot hide unstable
cross-condition behavior.

## PPO observation and environment state

![PPO observation and ontology-state representation](images/rl_observation_state-v3.png)

The actor and critic receive 23 normalized measurable channels. They include
pad-relative kinematics, landing context, UGV motion, wind/motion/energy, and
GNSS integrity features. The ontology is a parallel semantic representation of
the same observable experiment state. Ground-truth position, true NLOS counts,
and exact applied wind are excluded from both.

The dynamic-wind contribution uses a simulated three-axis onboard anemometer
with seeded bias, white noise, and first-order response. The moving-pad
contribution uses the road-following UGV state plus camera/GNSS-relative
navigation. These signals affect the ontology, the distilled reward terms, and
the policy observation without leaking simulator-only truth.

## Observability and outputs

The one-command launcher serves the dashboard at `http://127.0.0.1:8770/` and
starts RViz 2 when the graphical ROS environment is available. The dashboard
shows live vehicle/reward traces, the R-GAT graph, all frozen weights and ranges,
the shaping surface, and the two acceptance gates. RViz shows the vehicle,
moving pad, routes, camera detection, and terminal outcome.

Important persistent outputs are:

- `results/data/rgat_dataset_external.npz`: cumulative ontology dataset;
- `results/models/rgat_model_external.pt`: cumulative R-GAT checkpoint;
- `results/models/rgat_fixed_reward_external.json`: frozen design and provenance;
- `results/models/ppo_manual_external.pt` and
  `results/models/ppo_rgats_pbrs_external.pt`: cumulative PPO checkpoints;
- `results/optimization_acceptance.json`: effectiveness, consistency, and
  combined decision;
- `results/run_summary.json`, CSV metrics, sweeps, live plots, and figures.

See [Operations](OPERATIONS.md) for installation, launch, monitoring, and fault
diagnosis, and [Architecture](ARCHITECTURE.md) for the simulator, sensing,
vehicle, learning, and safety boundaries.
