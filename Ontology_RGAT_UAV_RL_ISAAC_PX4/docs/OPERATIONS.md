# Operations

## Upgrading a running stack to the urban environment

Nothing in a live stack picks these changes up on its own. A session started
before them keeps running the old code and fails in ways that look like bugs
elsewhere, so bring all four pieces forward together. This applies to the move
into the city as much as it did to the moving pad: the buildings, the road route
and both GNSS receivers are all created at Isaac startup.

```bash
./scripts/bootstrap_px4_ros2.sh     # re-mirrors the gateway, rebuilds PX4
# then stop and restart the agent, Isaac and the gateway
```

When only the gateway has changed, `./scripts/sync_gateway.sh` re-mirrors and
rebuilds that one package in a second, which is the whole of the bootstrap that
a Python-only edit needs.

| Piece | Why it has to be redone |
|---|---|
| `ros2_ws/src/ontology_rgat_px4` | The gateway runs from the ASCII mirror, which is a *copy*. Until the bootstrap re-mirrors it, the old gateway still reports world-frame position and no `pad`/`battery`, and `bridge.PX4Bridge` refuses it by design. |
| PX4 SITL | `battery_status` is new in `patches/px4-v1.14-publish-land-detected.patch`, and the client's topic table is generated from `dds_topics.yaml` at build time. The bootstrap now also forces a rebuild whenever that yaml is newer than the binary, so a hand-applied patch cannot leave a stale build behind. |
| Isaac / `landing_world.py` | The city prims, the deck prim, its collider, `/landing_pad/state/odom`, `/landing_pad/state/odom_truth` and `/landing_uav0/gnss/status` are all created at startup. A running instance has no city and no receivers, so the policy would be told it has open sky over an empty plane. |
| Trained checkpoints | `results/models/*.pt` from before the city have 20-element observations and a 13-node ontology. `load_agent`/`load_potential` refuse them on their dimensions; retrain, never transfer. |

Two failures are the signature of a half-upgraded stack:

- `Gateway state is missing field pad` from the learner: the mirror was not
  rebuilt.
- `no /landing_pad/state/odom has arrived` in the gateway log while
  `pad.motion` is not `static`: Isaac was not restarted. The gateway then
  reports `estimator_valid` false rather than pretending the deck is at the
  origin, so episodes refuse to start instead of quietly training against the
  wrong target.
- `no /landing_uav0/gnss/status has arrived` while `gnss.enabled` is true: the
  same cause. Here the gateway cannot refuse to run — a missing fix is not a
  link fault — so it logs the error once and reports open sky. An urban run
  whose GNSS integrity sits at 1.00 for the whole episode is this, not a lucky
  street.

Trained artefacts do not carry over either. The observation is 23 elements and
the ontology has 14 nodes, so a policy or potential from the fixed-pad or
open-field runs is not loadable; the entry points assert the dimensions instead
of letting a mismatched model run.

## A black Isaac window with only the overlay line in it

The city is collidable and the streets are 19 m wide, so the GUI chase camera
can end up *inside* a facade. Nothing renders from in there, but
`isaac_sim/live_overlay.py` draws through the debug-draw extension, which
ignores depth and lighting — so the drone-to-pad vector stays visible and the
window looks like a simulator that has failed to load the world. It has not.

`isaac.viewport_follow.frame: street` expresses the offset as
`[along the road, across, up]` in the lorry's own heading frame, so the camera
sits back down the carriageway, which is the one direction a canyon leaves
open. `UrbanLayout.clear_viewpoint` then pulls the eye in along its own ray
until it is clear, which covers the outside of a corner, where the
instantaneous tangent points into the corner building.

If you configure `frame: world`, check the offset against the street: a
horizontal reach longer than `urban.road_half_width_m + urban.sidewalk_m`
(9.5 m as shipped) will bury the camera on a straight.

## The ROS 2 workspace is built somewhere else

`scripts/bootstrap_px4_ros2.sh` does **not** build `ros2_ws/` in place. Humble's
`rosidl` dependency parser corrupts non-ASCII source and build paths, and this
workspace lives under a path with Korean characters, so the bootstrap mirrors
`ros2_ws/src/px4_msgs` and `ros2_ws/src/ontology_rgat_px4` into an ASCII-only
runtime workspace and builds there:

```text
$ASCII_ROS2_WS  (default /home/$USER/.local/share/ontology_rgat_uav_rl/ros2_ws)
```

`scripts/run_gateway.sh`, `scripts/check_ros2_loopback.sh` and
`scripts/stack_status.sh` source `$ASCII_ROS2_WS/install/local_setup.bash`. Two
consequences:

- Editing `ros2_ws/src/ontology_rgat_px4/**` changes nothing until the mirror is
  refreshed, because it is a copy. `colcon build --symlink-install` inside the
  mirror only symlinks within the mirror.
- `ROS workspace is not built` from `run_gateway.sh` usually means the mirror is
  missing or was built under a different `$USER`/`$ASCII_ROS2_WS`, not that
  `ros2_ws/` is empty.

`scripts/sync_gateway.sh` owns both halves of that:

```bash
./scripts/sync_gateway.sh            # re-mirror and rebuild the gateway package
./scripts/sync_gateway.sh --check    # report drift, change nothing
```

`run_gateway.sh` and `check_ros2_loopback.sh` run `--check` before they start a
gateway and refuse if the mirror has fallen behind, printing what differs and
the command that fixes it. That guard exists because the failure it prevents is
so far from its cause: a stale gateway boots cleanly, and the mismatch only
surfaces as a `BridgeError` several minutes into a run, after Isaac has come up
and the stack has retried the reset twice. The check compares the repository
against the package Python actually imports, so it also catches a mirror that
was refreshed but never rebuilt, and it stays quiet on an ASCII path where the
workspace is built in place and there is no copy to fall behind.

Every consumer of `/fmu/*` must use Fast DDS, because that is what the agent
speaks. The run scripts export `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and unset
`CYCLONEDDS_URI`; an interactive shell with a different default sees the topics
but reads nothing from them.

## Startup order

1. Start QGroundControl or ensure PX4's RC/GCS arming checks are intentionally
   configured for the test.
2. Start the Micro XRCE-DDS Agent on UDP 8888.
3. Start Isaac/Pegasus. Pegasus auto-launches PX4 SITL and owns TCP 4560.
   `scripts/run_isaac.sh` opens a window unless `HEADLESS=1`. The world renders
   once per `isaac.rendering_dt` rather than once per physics step; rendering
   every step drives the frame rate to the physics rate and, because PX4 is
   lockstepped to the simulator, slows the flight stack itself. With
   `vision.mode: aruco` the world still renders in a headless run, because the
   pad camera only produces an image on a rendered frame.
4. Confirm `/fmu/out/vehicle_odometry`, `/landing_uav0/state/pose`, and
   `/landing_pad/state/odom` are live.
5. Start the gateway.
6. Run `python3 python/run_episode.py --policy expert`, then one deterministic
   evaluation episode.

`ontology_rgat.stack.ExternalStack` automates exactly this order and waits on one readiness
signal per stage. Knowing which signal it waits for is what makes a startup
timeout diagnosable:

| Stage | Readiness signal | Default budget |
|---|---|---|
| DDS agent | UDP 8888 bound (`ss -lnu`) | 30 s |
| Isaac + PX4 SITL | `Ready for takeoff` in `isaac.log` | 600 s |
| Gateway | UDP 14650 bound | 60 s |

Each stage is skipped and adopted if it is already up — a UDP port already bound,
or a running `landing_world.py` — and an adopted process is never stopped on
teardown: a session started by hand belongs to whoever started it. Logs, pid
files and the generated launchers are written to `/tmp/ontology_rgat_stack/`.

One launch detail matters if a process is started any other way: each child is
launched in its own session (`start_new_session`), so teardown can signal the
whole process group and Pegasus' PX4 child goes down with Isaac instead of
holding TCP 4560 against the next run.

The MATLAB version of this class additionally had to scrub `LD_LIBRARY_PATH`,
`LD_PRELOAD`, `QT_PLUGIN_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH` and `GTK_PATH` out
of every child's environment, because MATLAB prepends its own runtime to the
loader path and that breaks ROS 2 and Isaac. A plain Python process does not, so
children now inherit the environment they would get from a terminal.

For hardware, replace step 2 with
`scripts/run_dds_agent.sh serial --dev /dev/ttyUSB0 -b 921600`, configure the
flight controller's `uxrce_dds_client` for the same serial link, omit Isaac, and
start the gateway with `--target hardware`.

## Reset semantics

`reset` is accepted only for `target=sitl`. The gateway publishes a seeded reset
request on `/landing_sim/reset` and waits for Isaac's matching acknowledgement on
`/landing_sim/reset_ack`. What it does with the vehicle first depends on where the
vehicle is:

- landed: disarm.
- airborne: command `NAV_LAND`, and log that it did. Cutting power to an airborne
  vehicle used to be harmless because Isaac teleported it anyway; it no longer
  does, so a mid-flight reset has to be a commanded landing. The climb that
  follows simply takes control back.

The learner does not treat a raw height threshold as a successful landing. It
requires an airborne transition, PX4's land-detector state and a valid
sensor-derived pad stream. Ground contact without pad confirmation is logged
as `ground_mislanding`; excessive tilt/closing speed is `unsafe_touchdown`,
and energy exhaustion is `battery_depleted`. For every terminal outcome the
gateway disables offboard control, requests landing/disarm, and the learner
waits up to `cfg.external.outcome_settle_timeout` seconds for landed plus
disarmed before issuing the next reset. An unconfirmed stop is marked
`unconfirmed_*` rather than being silently reset.

The learner's `sensor` state is the only input to control, semantics, rewards,
R-GAT and PPO. `ground_truth` state is kept for validation and comparison
outputs only; it must not be added to observations or training datasets.

Isaac reseeds the wind, deck, battery and both GNSS receivers together.
`wind_scale` multiplies the wind field; `pad_scale` multiplies the speed selected
for the seeded deck trajectory (zero parks the lorry, the static control);
`gnss_scale` multiplies the canyon's error mechanisms and moves no building, so
zero is open sky in the same city.

`pad.route_start` decides where on the lap an episode begins. `continue` (the
default) leaves the lorry where it is and reseeds only how it drives from
there, so the deck pose is continuous across the reset to within a millimetre
and the vehicle parked on it stays parked. `seeded` draws a fresh point on the
route instead: that makes the deck's absolute position a function of the seed
too, at the cost of teleporting it a mean of ~55 m away from the vehicle, which
then has to chase it across the city before the episode can start. Under
`continue` the lorry's position along the lap is the one part of the initial
condition the seed does not fix -- it is inherited from the previous episode --
so the canyon geometry an episode meets varies with history rather than with
the seed. Everything that defines the task (speed, traffic, lane, entry offset,
wind, energy, constellation) stays seeded either way. The entry is a **pad-relative** offset:
lateral offsets from 1.8 m and 1.4 m Gaussians, altitude uniform in [4.0, 5.8] m
above the lorry's roof, and roll/pitch/yaw from 4°/4°/12° Gaussians. The
acknowledgement also carries the seeded 9–55 hover-second starting reserve and
the episode's opening fixes. Isaac does **not** teleport the UAV.

The entry pose is then flown by PX4: the client sends a pad-frame `goto` with the
seeded offset and yaw, and the gateway recomputes the world-frame
`TrajectorySetpoint` from the live deck every control tick. It arms — retrying
every `cfg.external.armRetry` seconds, because PX4
rejects arming in transient pre-flight states — and the client hands over to the
policy only once PX4 holds that point inside
`entryTolerance`/`entrySpeedTolerance` for `entrySettle` seconds. The first
`action` message switches the gateway back to attitude offboard control, ends
the climb, and starts the seeded energy budget. `cfg.external.entryTimeout` is
90 s because it has to outlast PX4's
roughly 40 s post-boot arm refusal as well as the climb.

Teleporting is not an option. Pegasus' `PX4MavlinkBackend.reset()` is a
documented no-op, so PX4's EKF integrates straight through any pose jump: a
4.6 m teleport was measured to leave the estimator reporting −1.5 m and still
6 m short of truth several seconds later, while the disarmed vehicle free-fell
back to the pad. Because the controller is contractually fed PX4 estimator data,
that made every episode meaningless. An **airborne** vehicle is therefore left
exactly where the previous episode ended, always.

A **grounded** one is re-seated on the lorry's roof, and the acknowledgement
reports it as `reseated_on_deck`. Two things make that necessary and one makes
it cheap. The deck is redrawn with a new cruise speed at every reset, and a
kinematic body whose speed steps in a single physics tick shears whatever is
resting on it, so a vehicle that landed successfully can be left sliding off
the back. It may also have ended the last episode beside the lorry rather than
on it, or tipped past `landing.crash_tilt_deg`, which no climb recovers from.
What makes it cheap is `pad.route_start: continue`: the deck no longer
teleports between episodes, so the correction is sub-metre and the estimator
step is far smaller than the GNSS error this environment models in any case.

Three consequences to keep in mind when comparing runs:

- Each reset costs a real-time climb of several seconds, so external `full`
  sweeps are much slower than the in-process simulator.
- Entry *position* and *yaw* are reproduced from the seed. Entry roll, pitch and
  velocity are whatever PX4 settles at, not the seeded small perturbations of
  the original reset: `entry_rpy_deg` is reported in the acknowledgement but only
  its yaw component is flown.
- The episode clock starts at handover, not at the reset, and runs on PX4's
  simulated clock. SITL battery integration uses the same clock and deliberately
  does not charge the pre-episode climb to either policy.

Never compare two policies unless both received successful reset
acknowledgements and reached the entry pose.

## Long sweeps

PX4 SITL degrades over hours of lockstep: `battery_status` goes stale and
arming is refused with `Preflight Fail: Battery unhealthy`. That surfaces as a
reset that never reaches the entry pose. When `run_pipeline` owns the simulator
it registers it via `stack.current`, and `sim.resetState` then cycles the
simulator and retries up to `cfg.external.resetRecoveries` (2) times rather than
throwing away the run. A stack started by hand is never restarted from under the
user — the retry is skipped and the error propagates.

`pipeline.runAll` writes each stage's artefact before the next stage begins, so a
failure late in a long sweep does not discard the hours before it.

## R-GAT CPU/GPU selection

The local relation layer evaluates all nodes, edges, and batch entries in one
dense batched pass over the whole minibatch of graphs. The 13-node graph is too
small to fill a GPU on its own, so the win comes from batching graphs and from
not stalling the launch queue between batches: the trainer uploads the dataset
once, keeps the shuffle and the loss accumulation on the device, and enables
TF32. An RTX 4060 laptop measured about 0.9x/1.5x/6.6x/13.4x/20.4x GPU speed at
batches 32/64/128/1024/4096, so automatic placement uses a threshold of 64.
Override with `cfg.device.rgat` (`'auto' | 'cuda' | 'cpu'`, or `--rgat-device`),
and benchmark the current hardware with:

```bash
python3 python/run_benchmark.py --batches 32 64 128 1024 4096
```

Changing `cfg.rgat.batch_size` changes the number of optimizer updates. Record it
as a hyperparameter and retrain both compared conditions consistently -- which is
why `'auto'` declines below the threshold rather than rebatching to reach it.
The PPO actor/critic update was measured at about 2 s per 2048 collected steps;
collecting those steps takes about 41 s of lockstepped flight. Its 64-sample
networks therefore remain on CPU because GPU placement would not materially
change the end-to-end runtime.

## The city the episode is flown in

`urban.source` decides where the skyline comes from.

- `synthetic` generates the block from `config/system.yaml`. Its geometry is
  exactly known, which is why the occlusion tests use it, and it stays the
  control condition.
- `osm` builds it from a cached OpenStreetMap extract under `assets/city/`.
  Fetch one with

  ```bash
  scripts/fetch_city.py --name seoul-myeongdong \
      --latitude 37.5636 --longitude 126.9850 --radius 300
  ```

  then point `urban.extract` at the name and `urban.origin` at the same
  latitude/longitude. The origin is not decoration: PX4's home, every lat/lon
  on the wire and the constellation's elevations all come from it.

The simulator only ever reads the cached file, so a run reproduces from the
extract rather than from whatever Overpass returned that morning. Extracts are
small enough to commit next to the results they produced. OpenStreetMap data is
© OpenStreetMap contributors, ODbL 1.0, and the extract carries that
attribution.

Two things a real extract needs watching for:

- **Heights.** Most OSM buildings outside the dense cores carry no height tag
  at all -- 284 of 349 in the first Seoul extract. Those are drawn from
  `urban.height_range_m`, seeded, because one default flattens the skyline into
  a wall of equal blocks and the canyon stops varying along the lap. Mapped
  heights are always used when present.
- **The carriageway.** The lap is a rectangle imposed on the map, not a road
  traced from it, so buildings within `urban.route_clearance_m` of the route
  centreline are dropped. Check `sky_view_fraction` around the lap after
  changing site or `map_heading_deg`: if it never varies there is no canyon to
  measure, and if it reads 0.0 the route is inside a building.

## Failure diagnosis

- UAV motionless while the run prints `R-GAT epoch`: this is the offline
  potential-training stage. Isaac/PX4 stays online and the disarmed vehicle
  waits on the pad until PPO flight starts. Run `scripts/stack_status.sh`; a
  changing odometry stream confirms the simulation loop is healthy.
- `no /fmu/out/vehicle_land_detected received`: PX4 was built without
  `patches/px4-v1.14-publish-land-detected.patch`. The gateway reports
  `extra.land_detector: "missing"` and holds `landed` at false rather than
  guessing, so touchdown is never detected and every episode times out. Re-run
  `scripts/bootstrap_px4_ros2.sh` and rebuild PX4.
- `extra.land_detector: "stale"`: the topic exists but has gone quiet within
  `system.state_timeout_s`. Usually the simulator is starved, not the detector.
- `pad.motion is ... but no /landing_pad/state/odom has arrived`: do not run a
  moving-target policy. Restart Isaac with the current `landing_world.py`; on
  hardware, start the cooperative UGV localization publisher. The gateway marks
  the state invalid instead of subtracting a stale target.
- State lacks `position_frame`, `pad`, `battery` or `gnss`: the gateway or fake
  predates the contract this learner speaks. For the ASCII mirror the fix is
  `./scripts/sync_gateway.sh` with the stack stopped; editing the source tree
  alone does not update the build. The launchers now refuse to start a gateway
  that has drifted, so this should only be reachable from a fake gateway or a
  hand-started `ros2 run`.
- No `/fmu/out/*`: check the DDS agent and that PX4 `uxrce_dds_client` targets
  UDP port 8888. If the topics exist but read as empty, the client is almost
  certainly on a different RMW: the agent speaks Fast DDS, so every consumer
  needs `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and no `CYCLONEDDS_URI`. The run
  scripts set this; an interactive shell may not.
- `Compass needs calibration - Land now!` / `Failsafe activated` mid-climb,
  usually behind `Preflight Fail: horizontal velocity unstable` and
  `velocity estimate error`: PX4 levelled its estimator while the lorry carried
  it off. The lap is therefore held until the drone is more than a metre above
  the roof (`LandingDeck.hold`/`release_when_clear`), and a reset that re-seats
  the vehicle parks it again. If this returns, check that `_advance_deck` is
  still releasing on vehicle altitude rather than on the clock: a deck that
  drives while PX4 boots poisons the EKF, and the failsafe that follows takes
  the vehicle out of offboard in mid-climb -- it flies up and then falls away
  from the pad, which reads like a wind problem and is not one.
- `goto hold expired without handover; commanding a landing`: the entry climb
  ran out its `external.entry_timeout` before the learner sent an action. The
  gateway now puts the vehicle down rather than going quiet under it, so the
  reset retry starts from a vehicle on the deck. The cause is upstream --
  arming refused, offboard never accepted, or an entry pose PX4 cannot reach.
- `PX4 did not hold the entry pose`: the climb never converged. Check that the
  gateway logged no `goto hold expired`, that PX4 reached `OFFBOARD`
  (`nav_state` 14, or `extra.offboard_active`), and that `estimator_valid` is
  true — the gateway reports false whenever PX4's own `vehicle_local_position`
  validity flags are down, which is the honest answer during an EKF transient.
- Offboard rejected: the gateway already re-requests the mode every
  `control_hz/2` ticks while streaming setpoints, so a persistent refusal is
  PX4's, not a missed request. Check QGC/RC and `extra.last_command`, which
  carries the command id and PX4 result code from `vehicle_command_ack`; the
  rejection is also logged with its reason.
- `PX4 simulated time advanced only … ms`: the simulator stalled. This is the
  bridge refusing to fake progress; look at Isaac's log and its frame rate
  before looking at the learner.
- Vehicle dives: verify NED/ENU conversion and negative Z body thrust; do not
  compensate by flipping gains. If the expert controller touches down hard,
  suspect `px4.hover_thrust` first and re-run `tools/calibrate_hover_thrust.py`.
- Immediate `battery_depleted`: distinguish the experiment's seeded SITL energy
  model from PX4's preflight battery health. Inspect state `battery.source`,
  `hover_seconds_remaining`, and `extra.px4_battery`; the modeled episode budget
  starts only on the first action.
- Isaac waits for heartbeat: TCP 4560 is occupied or PX4 model/instance does not
  match Pegasus.
- `Landing camera is not looking down` / `produced no frame`: the camera stage
  setup failed. Dump frames with
  `ONTOLOGY_RGAT_VISION_DEBUG_DIR=/tmp/frames` — it is the only way to tell "no
  image" from "no marker in it".
- `RViz 2 publishing disabled: ROS 2 is not importable`: the learner was started
  without `/opt/ros/humble/setup.bash` sourced. Training continues without the
  live view; source it and restart to get one. The same applies to
  `cannot bind 127.0.0.1:8770` for the dashboard, where the usual cause is a
  previous run that has not exited.
- Gateway timeouts: keep gateway and learner hosts/ports consistent and allow UDP
  only on the loopback interface for a single-machine run. The gateway binds
  14650 without `SO_REUSEADDR` on purpose, so a second gateway fails to start
  instead of silently stealing the control link.

## Where the output goes

Everything lands under `results/` (`cfg.paths.*`), which is git-ignored:

| Path | Written by |
|---|---|
| `results/data/rgat_dataset_external.npz` (+ `.graph.pkl`) | stage 2, dataset generation |
| `results/models/rgat_model_external.pt` | stage 3, R-GAT potential (CPU float32, with the schema it was trained against) |
| `results/models/ppo_manual_external.pt` | stage 4, baseline PPO |
| `results/models/ppo_rgats_pbrs_external.pt` | stage 5, proposed PPO |
| `results/comparison_results_external.pkl`, `summary_metrics_external.csv`, `episode_metrics_external.csv` | stage 6, paired evaluation |
| `results/wind_generalization_sweep.csv` | stage 6, Isaac wind sweep |
| `results/pad_speed_sweep.csv` | stage 6, paired moving-deck speed sweep |
| `results/battery_reserve_bins.csv` | stage 6, paired outcomes binned by starting reserve |
| `results/paired_difference_ci.csv` | proposed-minus-manual paired 95% intervals, including relative speed, depletion and energy |
| `results/figures/*.png` | stage 7, publication figures |
| `results/run_summary.json` | stage 7, elapsed time, final R-GAT validation loss, the device it trained on |
| `results/live/*.png`, `results/live/*.csv` | live snapshots during dataset generation, R-GAT training and both PPO runs |

`results/live/` is the one to watch during an unattended run with nothing else
attached: the monitors export a PNG and a history CSV every `cfg.viz.live_every`
episodes or epochs. While the run is up, the dashboard at
`http://127.0.0.1:8770/` and RViz 2 (`./scripts/run_rviz.sh`) show the same data
live.

The dashboard's 3D ontology panel is published on the same throttle,
`cfg.viz.graph3d.every` control steps during an episode and epochs during R-GAT
training (`cfg.viz.graph3d.enabled: false` turns it off). Its snapshot travels
in the `/api/state` payload alongside the series, so it needs no extra port and
survives the SSH forward the rest of the dashboard uses. Before stage 3 finishes
there is no trained model to ask, and the panel says so: it draws the schema
with uniform edges and labels itself *schema only*.

## Reproducibility

Archive `config/system.yaml`, PX4 ULog, gateway JSONL log, Isaac version, Pegasus
commit, PX4 commit, `px4_msgs` commit, Micro-XRCE-DDS-Agent version, the applied
patches under `patches/`, `px4.hover_thrust` and the calibration that produced
it, `pad.motion`/speed range, pack parameters and reserve range,
`cfg.rgat.batch_size`, `cfg.device.rgat`, the R-GAT attention mode and head
count, the checkpoint hashes under `results/models/`, `results/run_summary.json`,
and the episode seed.
