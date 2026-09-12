# Operations

[Documentation map](README.md) · [System overview](SYSTEM_OVERVIEW.md) ·
[Architecture](ARCHITECTURE.md) · [Hardware safety](HARDWARE_SAFETY.md)

This runbook covers the primary controlled three-pipeline experiment. Legacy
cooperative-urban commands and output paths are listed separately at the end.

## Canonical launch

From the Git repository root:

```bash
./run.sh
```

The root wrapper is the authoritative launcher. With no arguments it adds:

```text
--mode full
--total-train-episodes 800
--eval-episodes 5
--rgat-data-episodes 40
```

After the 8-flight `shin_se` estimator warm-up, the remaining 792 PPO flights
divide evenly into 264 per pipeline. The semantic-data collector may extend
from 40 to 120 real flights if its minimum set contains only one terminal
class. Evaluation is 105 flights: three pipelines, seven scenarios, and five
paired seeds.

```bash
./run.sh --mode quick --headless
./run.sh --mode full --pipelines shin_se no_se onto_no_se
./run.sh --mode full --training-replicate 2 \
  --train-episodes 40960 --rgat-data-episodes 400
```

Do not start a second launcher. The root script holds
`/tmp/ontology_rgat_flight_pipeline.lock`; a competing process exits with code
73 before touching the runtime or results. The PID and command in the error are
the existing owner, not a stale guess. A crashed owner releases the lock
automatically.

## Read-only health check

Use these while training is running; none resets the vehicle:

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/stack_status.sh
curl -fsS http://127.0.0.1:8770/api/state
ps -eo pid,etime,state,%cpu,%mem,cmd | \
  grep -E '[r]un_three_pipeline|[l]anding_world|[r]os2_gateway|[M]icroXRCEAgent'
```

Healthy primary-runtime signals are:

- DDS bound on UDP 8888;
- gateway bound on UDP 14650;
- dashboard bound on TCP 8770 unless explicitly disabled;
- `landing_world.py`, PX4, gateway, and learner processes alive;
- `/fmu/out/vehicle_odometry` rate changing;
- dashboard `active episode` or `live episode step` changing during flight;
- the pipeline checkpoint/history timestamp advancing after each committed
  episode.

The committed episode count changes only after a complete episode, optimizer
update, checkpoint write, and history write. A 30 s simulated episode may take
longer than 30 s of wall time in the rendered S5 scene. During that interval,
use `active episode` and `live episode step`; an unchanged committed count alone
is not evidence of a stall.

![MATLAB-style live status](images/live_dashboard_status.png)

The image is actual in-progress telemetry captured on 2026-09-12, not a final
success-rate result.

## Startup ownership

The runner starts in this order:

| Component | Readiness | Default endpoint |
|---|---|---|
| Micro XRCE-DDS Agent | UDP socket bound | 8888/UDP |
| Isaac Sim + Pegasus + PX4 | PX4 log contains `Ready for takeoff` | Pegasus owns PX4 link |
| ROS 2 gateway | UDP socket bound plus DDS discovery grace | 14650/UDP |
| dashboard | local HTTP bind | 8770/TCP |
| RViz 2 | process launch in graphical mode | `/landing_rl` topics |

An already compatible process is adopted and is not stopped on teardown. A
process started by the current runner is owned and is stopped when the run
finishes unless `--keep-stack` is supplied. Recovery can restart only an owned
stack; the code will not kill a hand-started simulator.

Runtime logs are:

```text
/tmp/ontology_rgat_stack/agent.log
/tmp/ontology_rgat_stack/isaac.log
/tmp/ontology_rgat_stack/gateway.log
```

## Dashboard and RViz

Open <http://127.0.0.1:8770/>. The primary view shows:

- stage and phase;
- current pipeline and state-estimation status;
- committed training total, active episode, and live step;
- paired-evaluation progress and scenario;
- target visibility, UAV/UGV speed, and battery energy/reserve;
- optimizer phase, effective learning rate, curriculum, UGV motion scale, and
  UAV action envelope;
- reward-design flight/sample/class counts and R-GAT diagnostics when those
  stages begin;
- moving landing success as the primary learning curve;
- episode return explicitly labeled diagnostic.

Per-pipeline chips report committed counts without inventing equal targets:
the run-wide denominator contains a `shin_se`-only warm-up and therefore cannot
be divided honestly by three.

RViz starts automatically unless `--headless` or `--no-rviz` is used. It shows
the vehicle, UGV/deck, road route, trails, terminal marker, and annotated
landing camera. An empty camera display means no image topic; `PAD NOT
DETECTED` means the image is present but the board solve failed.

If 8770 is already occupied, the runner continues with metric collection but
prints that the live dashboard is disabled. Either stop the old owner or use:

```bash
./run.sh --dashboard-port 8771
```

## Reset and entry hover

Every measured episode starts only after a physical PX4-controlled handover:

1. Isaac reseeds platform motion, camera/environment state, and battery.
2. A grounded UAV is re-seated on the deck; an airborne UAV is not teleported.
3. The UGV remains parked during estimator startup and initial climb.
4. The gateway continuously transforms the pad-relative entry target to a
   world-frame PX4 position setpoint.
5. PX4 arms and flies to the camera-centred hover.
6. The client requires entry-position tolerance, speed at most 0.40 m/s, and a
   marker detection seen within the last 2.0 s for a continuous 1.0 s hold.
7. The first policy action changes the control source from `goto` to action
   setpoints, releases UGV motion, and starts the measured battery budget.

Teleporting the airborne UAV is intentionally forbidden: PX4's EKF integrates
through the discontinuity and would make subsequent observations physically
meaningless. A failed entry gate is setup failure, not a training sample.

Between PPO episodes, the gateway holds an unfinished airborne vehicle at a
bounded position while optimization runs. This prevents an action deadman from
dropping PX4 into a landing mode before the next reset.

## Checkpoint and retry behavior

Each complete recurrent episode atomically commits:

- model weights;
- Adam state;
- curriculum state;
- completed episode number;
- configuration hash and pipeline contract;
- reward-design hash where applicable;
- training history CSV.

Compatible state resumes. An incompatible checkpoint is renamed
`*.incompatible-<old-hash>.pt` with its history and training starts under the
new configuration. It is never silently transferred across an information
boundary.

`collect_episode_resilient` retries only recoverable infrastructure failures:

- learner/gateway timeout;
- simulator clock that genuinely stops advancing;
- gateway-classified pure PX4 Offboard heartbeat loss.

The partial trajectory is discarded, an owned stack is restarted, and the same
seed is retried within the configured recovery count. This avoids a paired-seed
bias. Other PX4 failsafes and estimator, marker, geometry, policy-health, or
terminal failures are not treated as infrastructure recovery.

## Fault diagnosis

| Symptom | Meaning and action |
|---|---|
| `another ... pipeline is already active` | The lock is protecting a live owner. Inspect the reported PID; do not launch a competitor. |
| Completed count appears frozen | Check `active episode`, `live episode step`, process state, and odometry. A rendered flight commits only at episode end. |
| Dashboard says `debug`/return only | Refresh the browser after current code is running. The primary chart is training success; return is diagnostic. An adopted older process cannot load edited HTML until restarted. |
| `cannot bind 127.0.0.1:8770` | Another dashboard owns the port. Find its PID with `ss -ltnp` or use `--dashboard-port`. |
| `PX4 did not hold the entry pose` | Inspect reported offset, speed, and marker quality; then gateway/PX4 logs. The S5 profile admits the measured hover limit cycle up to 0.40 m/s and remembers a recent detection for 2 s. |
| `PX4 estimator state is not valid yet` | PX4 local position/velocity validity or freshness is false. Do not bypass it; inspect startup/odometry rate and PX4 preflight messages. |
| `PX4 simulated time advanced only ...` | A small non-negative advance is a real simulator stall. The current bridge also re-anchors an XRCE time-domain jump that older code misreported as a huge negative stall. |
| `OFFBOARD_HEARTBEAT_LOSS` | The gateway classifies a pure setpoint-link interruption as recoverable in SITL. Other simultaneous failsafe reasons remain hard failures. |
| UGV or UAV does not move during `R-GAT training` | R-GAT optimization is offline; no flight is expected. Flight resumes for `onto_no_se` PPO/evaluation. |
| Target repeatedly not visible | Check the annotated camera topic, board textures, camera mount, and entry pose. The board contains far/mid/micro tags specifically for the full descent. |
| `no /fmu/out/*` | Confirm DDS UDP 8888, PX4 uXRCE client, matching `px4_msgs`, Fast DDS RMW, and no `CYCLONEDDS_URI`. |
| Gateway state lacks current fields | The ASCII ROS workspace contains stale copied source. Stop the stack, run `./scripts/sync_gateway.sh`, then restart. |
| `Preflight Fail: Battery unhealthy` | Distinguish PX4 SITL's internal battery from the experiment pack. Current SITL clamps only PX4's unrelated internal pack; the 3S 3500 mAh experiment model still discharges and feeds R-GAT. |
| No success although the deck was reached | Inspect pad-contact topic/source and touchdown physical limits. Height alone is never counted as success. |
| Training health gate stops | The configured window detected no success, excessive FOV/RMSE/battery failure, poor reacquisition, unsafe blind descent, or a saturated active reward. The message names each failing gate; this protects the remaining budget from blind rollouts. |

Useful direct probes:

```bash
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 topic hz /fmu/out/vehicle_odometry
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 topic hz \
  /landing_uav0/perception/landing_camera/annotated
python3 tools/protocol_probe.py state
tail -n 100 /tmp/ontology_rgat_stack/gateway.log
tail -n 100 /tmp/ontology_rgat_stack/isaac.log
```

## Primary output paths

For the default run, use `results/three_pipeline/full/`:

| Path | When it appears |
|---|---|
| `manifest.json` | before the stack starts; updated through completion |
| `evaluation/paired_plan.csv` | before training |
| `models/shared/keypoint_encoder.pt` | after encoder preparation |
| `models/<pipeline>/<pipeline>.pt` | after every committed training episode |
| `models/<pipeline>/<pipeline>_training.csv` | after every committed training episode |
| `training/<pipeline>.csv` | after that pipeline's requested training completes |
| `rgat/semantic_rollouts.npz` | after each completed reward-design flight |
| `rgat/semantic_rollout_episodes.csv` | reward-design per-flight outcomes |
| `rgat/rgat_model.pt` | after direct R-GAT fitting/freezing |
| `evaluation/per_episode.csv` | after each completed evaluation pair |
| generated CSV/Markdown/figures | after report generation |

Replicates 1 and 2 write under `full/replicate_1/` and
`full/replicate_2/`. The report generator discovers these directories and
writes the hierarchical aggregate under `full/combined/`:

```bash
python3 python/generate_three_pipeline_report.py \
  --results-dir results/three_pipeline/full
```

Archive the manifest, config files, Git commit, checkpoint hashes, software
versions, GPU, wall-clock time, and raw per-episode records with any reported
result.

## Manual startup and maintenance

The one-command runner is preferred. For component diagnosis only:

```bash
./scripts/run_dds_agent.sh
ISAACSIM_PATH=/absolute/path/to/isaacsim \
  ./scripts/run_isaac.sh config/shin2026-system.yaml
./scripts/run_gateway.sh --config config/shin2026-system.yaml \
  --target sitl --allow-arm
./scripts/run_rviz.sh
```

If ROS gateway source changed, the Korean repository path means the live ROS 2
package may still be the copied ASCII workspace. Synchronize only while the
stack is stopped:

```bash
./scripts/sync_gateway.sh --check
./scripts/sync_gateway.sh
```

Run the offline code contracts with:

```bash
./scripts/check_workspace.sh
./scripts/check_learner_protocol.sh
./scripts/check_ros2_loopback.sh   # after ROS/PX4 bootstrap
```

## Legacy cooperative experiment

`scripts/run_metasejong_pipeline.sh` is a separate retained experiment. Its
23-channel policy, 14-node/38-edge ontology, eight distilled fixed reward
weights, output layout under legacy `results/` paths, and old figures do not
describe `onto_no_se`. Root invocations containing `--methods` or `--reward`
are routed to the legacy reward-arm runner for compatibility.

Do not copy checkpoints, result tables, or success claims between primary and
legacy experiments.
