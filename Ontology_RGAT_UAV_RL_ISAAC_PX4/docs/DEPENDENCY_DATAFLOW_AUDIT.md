# Dependency and data-flow audit

[Documentation map](README.md) · [Controlled comparison](THREE_PIPELINE_COMPARISON.md) ·
[Architecture](ARCHITECTURE.md)

Audit refreshed: 2026-09-12. Scope: current `main` working tree and the primary
`shin_se / no_se / onto_no_se` runner. This is a code-boundary audit, not a
claim about unfinished flight performance.

## Primary actor contract

`ActorObservation` exposes exactly a 512×320 grayscale image and seven onboard
UAV values: three body-frame velocity components and four attitude-quaternion
components. The frozen six-keypoint encoder, temporal LSTM, and actor are the
same in all three pipelines.

| Data | Actor | `shin_se` auxiliary/critic | Primary ontology | Terminal/evaluation |
|---|---:|---:|---:|---:|
| raw mono image | yes | yes | through keypoints/heatmaps | diagnostics |
| UAV body velocity/quaternion | yes | yes | bounded motion/attitude semantics | yes |
| six-keypoint output/heatmaps | through encoder | yes | yes | diagnostics |
| onboard battery reserve | no | no | yes | yes |
| relative-state truth | no | auxiliary target + critic | no | yes |
| predicted relative state | no | `shin_se` only | no | diagnostics only |
| marker pose solve | no | no | no | setup/visualization only |
| UGV pose/velocity or wheel odometry | no | no | no | setup/scoring only |
| platform GNSS/V2V | no | no | no | legacy profile only |
| simulator contact/truth | no | critic where declared | no | yes |

The actor always consumes `y[6:256]`; even `shin_se` does not append the six
predicted state values. The critic is separately called with 13 values only
during training and has no deployment wrapper input.

## Primary ontology boundary

`semantic_observation_from_payload` accepts only:

```text
keypoints, heatmaps, proprioception, battery_reserve
```

It recursively rejects aliases containing estimate, relative state, platform,
pad/deck motion, simulator/ground truth, critic, GNSS platform, or privileged
provenance. Unknown top-level fields also fail. The output is eight normalized
semantic observations and the fixed 13-node/25-edge graph.

This prevents the proposed method from secretly reconstructing the explicit
metric estimator it is intended to replace. Simulator truth may label an
episode `+1/-1`, but it cannot become an R-GAT feature.

## Reward-design dataset

The semantic dataset format is `ontology_rgat.semantic_rollouts/1`. Its
manifest records graph schema, config hash, source-policy checkpoint digest,
behavior mixture, seeds, flight/sample/contact counts, environment steps,
class counts, and forbidden-input declaration.

The trained `no_se` policy is the behavior source. Image-plane servo
corrections, bounded Gaussian noise, and bounded random exploration improve
coverage without metric pose. Synthetic success/failure insertion is forbidden.
Collection continues past its minimum only until both classes are observed or
the hard cap is reached. Validation splits by episode to prevent adjacent
frames from the same flight leaking across train/validation.

## Configuration and checkpoint guards

- Immutable `PipelineSpec` objects define estimator, auxiliary loss, active
  reward, ontology input mode, and direct-potential use.
- Startup verifies the YAML declarations exactly match those specs.
- R-GAT/PBRS/PPO gamma equality is checked.
- Recurrent checkpoints store format, pipeline spec, config hash, reward hash,
  optimizer state, and curriculum state.
- Incompatible artifacts are archived and retrained rather than shape-loaded.
- Direct R-GAT artifacts verify model digest, graph schema, and frozen state.
- JSON serialization rejects NaN; trainers stop on non-finite losses/gradients.

Relevant automated guards are in `tests/test_shin2026_integrity.py`,
`tests/test_three_pipeline.py`, and the protocol/config test modules.

## Runtime failure boundary

Infrastructure recovery is deliberately narrow. Gateway timeouts, genuine
simulated-clock stalls, and gateway-classified pure Offboard-heartbeat losses
may discard a partial trajectory and retry the same seed after restarting an
owned stack. Entry geometry, marker visibility, estimator validity, policy
health, other PX4 failsafes, and terminal outcomes are not relabeled or hidden
by that mechanism.

The root `run.sh` also serializes access to the single flight-control resource
with an OS lock. This prevents two learners from racing resets or receiving
each other's UDP replies.

## Retained legacy boundary

The 23-value cooperative observation in `semantic.make_observation`,
`/landing_pad/state/odom`, urban GNSS/V2V data, the 14-node graph, and distilled
eight-weight reward belong to `scripts/run_metasejong_pipeline.sh`. They are
valid for that extended experiment but privileged relative to the primary
non-cooperative actor contract. Primary and legacy artifacts are intentionally
incompatible.
