# Shin-2026 dependency and data-flow audit

Audit date: 2026-09-11. Source baseline: commit `42f0a2e` on `main`.

The retained urban OntoReward experiment builds a 23-value policy observation
in `semantic.make_observation`. It includes measured pad-relative pose and
velocity, a separately broadcast deck velocity, pad motion, battery state, and
GNSS integrity. The ROS gateway subscribes to `/landing_pad/state/odom` and
uses the deck broadcast to form this state. That is appropriate only for the
extended cooperative urban experiment; it is privileged relative to Shin et
al.'s non-cooperative actor contract.

| Data | Existing urban use | Shin-compatible classification | Benchmark action |
| --- | --- | --- | --- |
| `/landing_pad/state/odom` and deck velocity | observation/control fallback | forbidden actor input | excluded from typed actor schema |
| Deck GNSS quality/covariance | semantic observation and reward | forbidden actor input | benchmark GNSS disabled |
| Simulator pad/UAV odometry truth | reset and terminal scoring | reset/evaluation/critic/auxiliary target only | isolated in `CriticObservation` |
| Marker pose and marker quality | policy-relative pose and observation | actor may receive pixels, not solved pose | `pose_source_for_policy: false` |
| Raw onboard camera | operator/detector path did not expose raw mono | actor input | new unannotated `mono8` topic |
| UAV world velocity and attitude | PX4 telemetry | actor input | world velocity is rotated into body frame |
| Wind, energy, GNSS integrity | extended ontology observation/reward | excluded from primary comparison | retained only in original profile |

The old direct `[collective, roll, pitch, yaw-rate]` command remains available
for old checkpoints. A separate versioned `velocity_action` message drives a
PX4 velocity/yaw-rate setpoint for all new benchmark methods.

Automated guards live in `tests/test_shin2026_integrity.py`. They reject
privileged field names recursively before flattening, verify critic separation,
pair seeds, reset recurrent state, and enforce PBRS invariants.
