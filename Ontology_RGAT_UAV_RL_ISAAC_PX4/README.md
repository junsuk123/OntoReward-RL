# Ontology-RGAT UAV landing: Isaac Sim + PX4

This workspace replaces the in-process MATLAB rigid-body simulator in
`../Ontology_RGAT_UAV_RL_MATLAB/Ontology_RGAT_UAV_RL_MATLAB` with an external,
flight-stack-in-the-loop system. The original directory is not modified.

## Data path

```text
MATLAB ontology/R-GAT/PPO
  <--- versioned UDP/JSON --->  ROS 2 gateway
                                  |  PX4 uXRCE-DDS (/fmu/in, /fmu/out)
                                  v
                              PX4 SITL or PX4 hardware
                                  |  Simulator MAVLink (SITL only)
                                  v
                         Isaac Sim 5.1 + Pegasus 5.1
                         physics, sensors, contact, wind
```

The controller sees PX4 estimator data and, where the pad is in view, the pose
its own camera recovers from the markers. Isaac ground truth is used only for
experiment telemetry and reset acknowledgement. ENU/FLU is the workspace
convention; conversion to PX4 NED/FRD happens only in the gateway.

## Marker-based landing

The pad carries printed ArUco tags, a downward camera on the vehicle sees them,
and the pose is solved with OpenCV. While the pad is visible the policy flies on
that pad-relative pose (`vision.pose_source_for_policy`); when it is not, the
gateway falls back to the PX4 estimate and reports `marker_quality` 0, which is
what lets the ontology reason about degraded perception instead of being handed
a function of ground truth.

The pad uses two marker scales because one cannot cover a landing: a tag big
enough to resolve from the 4.6 m entry altitude overflows the frame below about
0.4 m, and a tag small enough to survive touchdown is a few pixels from
altitude. `config/system.yaml` therefore places four 0.45 m tags around one
0.10 m tag, and the solver uses whichever are visible.

```bash
# Dump annotated camera frames while the simulator runs.
ONTOLOGY_RGAT_VISION_DEBUG_DIR=/tmp/frames ./scripts/run_isaac.sh
```

Set `vision.mode: pose_proxy` to switch the camera off and fall back to the
analytic marker-quality stand-in; rendering the camera costs simulation speed
even headless, because PX4 is lockstepped to the simulator.

Each episode starts in the air. Isaac draws the entry pose from the same
distribution as the original simulator but does **not** teleport the vehicle:
PX4 flies there under its own position controller, and the policy takes over
once that point is held. `docs/OPERATIONS.md` explains why teleporting is
unusable here.

## Pinned integration baseline

- Ubuntu 22.04, ROS 2 Humble
- NVIDIA Isaac Sim 5.1.0
- Pegasus Simulator v5.1.0
- PX4-Autopilot v1.14.3 (the version tested by Pegasus 5.1)
- `px4_msgs` branch `release/1.14`, matching the PX4 firmware definitions

Newer PX4 releases can be used, but `PX4_VERSION` and the `px4_msgs` branch must
match exactly. Do not mix generated message definitions between releases.

## Install external dependencies

Isaac Sim is intentionally not downloaded automatically. It is a large licensed
runtime. Install Isaac Sim 5.1, then install Pegasus and the PX4/ROS dependencies:

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
ISAACSIM_PATH=/absolute/path/to/isaacsim ./scripts/bootstrap_pegasus.sh
./scripts/bootstrap_px4_ros2.sh
```

The script clones dependencies into `external/` and builds `ros2_ws/`. Override
paths/versions with environment variables shown by `--help`. It also applies
`patches/px4-v1.14-publish-land-detected.patch`, which is required: stock PX4
v1.14 does not publish `vehicle_land_detected`, `vehicle_command_ack` or
`vehicle_thrust_setpoint` over uXRCE-DDS, and without the land detector the
gateway cannot tell a hovering vehicle from a landed one, so touchdown is never
detected.

Every consumer of `/fmu/*` must use Fast DDS, because that is what the agent
speaks. The run scripts export `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and clear
`CYCLONEDDS_URI`; an interactive shell with a different default sees the topics
but reads nothing from them.

## Run everything with one command

On this PC the verified Isaac Sim release is
`/home/j/isaacsim/_build/linux-x86_64/release`. Start MATLAB from the desktop
session and run:

```matlab
cd('/home/j/SynologyDrive/junsuk/학술대회/CICS2026/codes/Ontology_RGAT_UAV_RL_ISAAC_PX4/matlab')
setenv('ISAACSIM_PATH','/home/j/isaacsim/_build/linux-x86_64/release')
clear classes
out = run_pipeline();                        % quick mode
out = run_pipeline('Mode','full');           % the paper-scale sweep
```

`run_pipeline` starts the DDS agent, Isaac/Pegasus/PX4 and the gateway in the
order below, waits for each readiness signal, flies one expert episode as a
pre-flight check, runs dataset generation, R-GAT training, both PPO runs, the
paired evaluation and the plots, then stops whatever it started. Processes that
were already running are adopted and left running.

```matlab
run_pipeline('SmokeTestOnly',true)     % bring the stack up, fly one episode, stop
run_pipeline('UseRunningStack',true)   % attach to a stack you started yourself
run_pipeline('KeepStack',true)         % leave the simulator up afterwards
run_pipeline('Headless',true)          % no window, for unattended runs
```

The Isaac Sim window opens by default whenever `DISPLAY` is set, and the run
falls back to headless on a machine without one. MATLAB itself must be started
from the graphical session, because the window is opened on the `DISPLAY` that
MATLAB sees. Rendering costs simulation speed even though it is throttled to
`isaac.rendering_dt`: PX4 runs in lockstep, so a slower frame rate slows the
flight stack too. Raise `isaac.rendering_dt` in `config/system.yaml` for a
cheaper picture, or run long sweeps headless.

After pulling changes, run `clear classes` before `run_pipeline` if a MATLAB
session is already open. `bridge.PX4Bridge` and `stack.ExternalStack` are
classdef files, and MATLAB keeps running the copy it loaded first: a session
that predates an update will fly the old control loop and report failures that
are already fixed on disk.

PX4 runs in real time: quick mode flies roughly 400 episodes and full mode
roughly 7000, so budget hours and days respectively. If PX4 stops accepting arm
commands mid-sweep, `sim.resetState` cycles the simulator and retries
(`cfg.external.resetRecoveries`). Process logs are under
`/tmp/ontology_rgat_stack/`.

After dataset collection, stage 3 trains R-GAT offline. During that stage the
Isaac/PX4 loop remains live but receives no flight commands, so the disarmed UAV
stays motionless on the pad. This is not a stopped simulation; flight resumes
at stage 4 (PPO). Check the live odometry rate and gateway state from a terminal:

```bash
cd '/home/j/SynologyDrive/junsuk/학술대회/CICS2026/codes/Ontology_RGAT_UAV_RL_ISAAC_PX4'
./scripts/stack_status.sh
```

## Run SITL by hand

Open three terminals:

```bash
# 1. DDS agent (PX4 <-> ROS 2)
./scripts/run_dds_agent.sh

# 2. Isaac Sim + Pegasus + PX4 SITL
ISAACSIM_PATH=/absolute/path/to/isaacsim ./scripts/run_isaac.sh

# 3. PX4 gateway
./scripts/run_gateway.sh --target sitl --allow-arm
```

Calibrate the collective mapping once per airframe or Isaac version, with the
stack running:

```bash
python3 tools/calibrate_hover_thrust.py
```

It reports the thrust PX4 needs to hover and the `px4.hover_thrust` to put in
`config/system.yaml`. This matters: the policy normalises its collective as
`hover*(1 + span*a)`, so a wrong hover thrust spends the policy's authority
just holding altitude. With the shipped Pegasus Iris the measured value is
0.580, and using the previous 0.500 made the expert controller touch down at
1.9 m/s instead of 0.3 m/s. `px4.collective_span` must equal
`cfg.rl.collectiveSpan` in the original workspace; the adapter refuses to run
if the gateway reports a different mapping.

Then in MATLAB:

```matlab
cd('Ontology_RGAT_UAV_RL_ISAAC_PX4/matlab')
setup_external_path
cfg = defaultExternalConfig('quick');
run_external_smoke_test
```

`run_external_all.m` runs the same stages as `run_pipeline` against a stack you
manage yourself.

`run_external_episode.m` loads an existing policy/R-GAT model from the original
workspace and runs it against PX4/Isaac. Training can use the same external
`sim.*` API, but flight-stack-in-the-loop training is real-time and reset-heavy;
start with evaluation and data collection.

Both entry points perform dataset generation, R-GAT training, both PPO runs,
paired evaluation, dynamic Isaac wind scaling, and plot generation without
calling the MATLAB rigid-body simulator.

## Real vehicle

Run a serial XRCE-DDS agent and the gateway; Isaac and the reset service are absent:

```bash
./scripts/run_dds_agent.sh serial --dev /dev/ttyUSB0 -b 921600
ONTOLOGY_RGAT_HARDWARE_OFFBOARD=I_ACCEPT_FLIGHT_CONTROL \
  ./scripts/run_gateway.sh --target hardware --allow-offboard
```

Hardware mode never auto-arms, never resets, and never flies the entry pose
itself: the pilot does. It refuses arm commands
unless both `--allow-arm` and `ONTOLOGY_RGAT_HARDWARE_ARM=I_ACCEPT_PROPELLER_RISK`
are present. Keep RC/QGroundControl takeover and PX4 offboard-loss failsafes
configured. First tests must be performed without propellers.

For a MAVLink-only companion link, use:

```bash
./scripts/run_mavlink_gateway.sh --target hardware --device /dev/ttyUSB0
```

This alternative sends `SET_ATTITUDE_TARGET`; the MATLAB protocol remains the
same.

Before enabling a hardware policy, publish the UAV pose in the landing-pad ENU
frame on `/landing_uav0/perception/uav_pose_in_pad` and marker confidence on
`/landing_uav0/perception/marker_quality`. See `docs/HARDWARE_SAFETY.md`.
After disarmed/propeller-off checks, arm through the pilot's normal path and run
`matlab/run_hardware_policy.m`. That script never sends an arm command and yields
to the configured PX4 offboard-loss behavior on exit.

## Verification

Tests not requiring Isaac/PX4/ROS:

```bash
python3 -m pytest -q tests
./scripts/check_workspace.sh
./scripts/check_matlab_protocol.sh
# after bootstrap/build
./scripts/check_ros2_loopback.sh
```

Live integration checks:

```bash
./scripts/stack_status.sh
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 topic hz /fmu/out/vehicle_odometry
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 topic echo --once /landing_uav0/state/pose
python3 tools/protocol_probe.py state
python3 tools/calibrate_hover_thrust.py
```

## Known limitations

- Seeds are not comparable across backends. Isaac draws the entry pose with
  NumPy's generator and the original simulator uses MATLAB's Mersenne Twister,
  so the same seed gives the same *distribution* but not the same episode.
  Compare distributions, not paired seeds.
- `metrics.energyJ` is `NaN`. PX4 reports no shaft power, and the in-process
  rotor model that produced it is gone.
- PX4 SITL stops accepting arm commands after long unattended sessions
  (`Preflight Fail: Battery unhealthy`, from `battery_status` going stale under
  lockstep). The gateway logs the rejection with its PX4 result code; restart
  Isaac to clear it. Budget a restart between long sweeps.
- Upward collective saturates near `a0 = 0.65`, where the gateway's 0.90 thrust
  ceiling is reached. A vehicle hovering at 58% throttle cannot deliver the
  full modelled +85% span in any case.

See `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md`, and
`docs/HARDWARE_SAFETY.md` before flight. Protocol/API sources are collected in
`docs/REFERENCES.md`.
