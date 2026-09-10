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
                    physics, sensors, contact, city, wind, GNSS
```

The controller sees PX4 estimator data and, where the pad is in view, the pose
its own camera recovers from the markers. Isaac ground truth is used only for
experiment telemetry and reset acknowledgement. ENU/FLU is the workspace
convention; conversion to PX4 NED/FRD happens only in the gateway.

## The environment: a lorry on a city street

The pad is painted on the roof of a box lorry driving a lap of a city block, in
lane, through stop-and-go traffic at 2-8 m/s. The block is built into the
Isaac stage procedurally (`isaac_sim/urban_scene.py`) rather than loaded as a
canned environment, because the *same* geometry has to serve two consumers: the
camera that renders it and the GNSS model that occludes satellites with it. A
skyline drawn from one set of boxes and a satellite mask computed from another
would give an urban-canyon experiment whose outages have nothing to do with the
visible city.

The route (`isaac_sim/pad_motion.py`, mode `road`) is a rounded rectangle
parameterised by arc length, so position, tangent and curvature are closed form
all the way round and the deck twist the policy feeds forward is differentiated
analytically rather than sampled. Traffic is a Gaussian dip in the speed at each
light -- deep enough to be a full stop when the draw says so -- whose integral is
an error function, so the distance covered is closed form too. `pad.motion:
static` is unchanged and remains the fixed-pad control condition.

Driving the lap takes the deck through four street canyons at two orientations
and four open intersections, which is what makes satellite visibility, wind
direction and marker visibility all change *during* an episode instead of being
one number per run.

## GNSS in a street canyon

`isaac_sim/gnss.py` models what the facades do to the fix:

- a satellite whose line of sight crosses a building is not received directly;
- most blocked satellites are still tracked, through a reflection off the
  facade opposite, whose path is longer -- so the pseudorange carries a strictly
  positive excess delay of about `2 d cos(el)`. Positive-only bias is what makes
  urban GNSS error a *bias* rather than noise, and why it points across the
  street;
- the survivors are strung out along the street, so the geometry degrades as
  well as the count;
- the position is then solved by weighted least squares from those per-satellite
  errors, exactly as a receiver solves it, and its own covariance is inflated by
  the post-fit residuals.

Typical synthetic mid-block numbers over the deterministic test seeds with the
shipped configuration are about 19 tracked signals (12 LOS and 7 NLOS), 1.7 m
reported horizontal sigma and 0.8 m horizontal error.  Open sky is about 1.0 m
reported sigma and 0.6 m error.  More deeply shadowed parts of the loaded OSM
city still cross the low-integrity threshold and exercise the DR path.

Both the drone and the lorry carry a multi-constellation receiver and share the
same satellite sky. Each uses the 3-D building map and C/N0 to suppress NLOS
ranges; their differential fix consequently stays metre-scale in the nominal
canyon instead of inheriting a tens-of-metres common bias. The markers remain
the precise landing anchor on a roof only 2.45 m wide.

What the policy is shown is only what a receiver publishes: satellite count,
DOP, its own inflated covariance, mean C/N0 and the fraction of signals its
C/N0 test flags as probably reflected. The true error, the true NLOS count and
the true sky view are the simulator's and never cross into the learner --
`tests/test_urban_gnss.py` enforces that boundary. The C/N0 detector matters:
when every satellite is reflected off a facade the same distance away their
biases agree with one another, so no consistency check sees anything wrong, and
signal strength is the only evidence left. It is not an oracle either, because
a few reflections come in nearly as strong as the direct path.

The receiver first downweights reflections detected by C/N0 or the loaded 3-D
building shadow map in its range solution,
then replaces Pegasus' generic GPS on MAVLink `HIL_GPS`, including its reported
EPH/EPV. PX4 EKF2 therefore performs the real GNSS/IMU fusion: uncertain but
valid fixes become weak drift-bounding observations, a true loss of fix falls
back to inertial dead reckoning, and consistently good fixes regain full weight.
The gateway observes that estimate and does not add a second synthetic offset.
`docs/ARCHITECTURE.md`, "GNSS", details the path and its diagnostic topics.

## Marker-based landing

The pad carries printed ArUco tags, a downward camera on the vehicle sees them,
and the pose is solved with OpenCV. While the pad is visible it anchors the
pad-relative estimate. When it is not, the gateway propagates that anchor with
PX4/deck relative velocity and applies the two GNSS positions only as a
covariance-weighted drift correction. It also reports `marker_quality` 0, which
lets the ontology reason about degraded perception without receiving truth.

The pad uses two marker scales because one cannot cover a landing: a tag big
enough to resolve from the entry altitude overflows the frame near touchdown,
and a tag small enough to survive touchdown is a few pixels from altitude.
`config/system.yaml` therefore spreads four 0.62 m tags along the lorry's roof
around one 0.18 m tag, and the solver uses whichever are visible.

In the canyon this is not a redundancy but the primary absolute anchor. During
a marker outage the inertial propagation stays continuous, while GNSS integrity
controls how quickly the drift correction is trusted. The `GnssIntegrity`
ontology node tells the policy which regime it is in; the pose alone cannot.

```bash
# Dump annotated camera frames while the simulator runs.
ONTOLOGY_RGAT_VISION_DEBUG_DIR=/tmp/frames ./scripts/run_isaac.sh
```

Set `vision.mode: pose_proxy` to switch the camera off and fall back to the
analytic marker-quality stand-in; rendering the camera costs simulation speed
even headless, because PX4 is lockstepped to the simulator.

## Sensor and ground-truth boundary

The policy, ontology, rewards, R-GAT dataset and PPO training consume the
gateway's sensor/estimator contract: PX4 odometry, camera marker quality, the
lorry's own V2V broadcast, the receiver's own report of its fix, and pad
telemetry. The learner never reads the `ground_truth` namespace. This keeps a
simulated experiment from silently turning privileged information into a
control feature.

The episode outcome is the one exception, and it has to be: under a canyon fix
the pad-relative pose the policy flies on can be metres from the truth, so
grading the landing on it would score the receiver's mistake instead of the
landing. The gateway therefore publishes a `truth` block -- the simulator's own
pad-relative geometry -- and `env.truth_state` uses it for the terminal test and
the touchdown metrics, and for nothing else. A link that carries no `truth`
block falls back to the sensor, which is what the fixed-pad experiment always
did.

An episode is successful only after the vehicle has been airborne, PX4's land
detector confirms touchdown, and the pad sensor stream is valid. Ground contact
without those checks is recorded as an unsuccessful touchdown; battery
depletion, attitude/flight-limit violations and timeouts are separate terminal
outcomes. The vehicle is commanded to stop and its landed/disarmed state is
confirmed before the next reset is accepted.

`run_pipeline.py` starts the live dashboard, the Isaac in-window overlay and,
when a graphical ROS 2 session is available, RViz 2. Training progress is
also exported to `results/live/*.csv` and `results/live/*.png`; the dashboard
is served at `http://127.0.0.1:8770/`.

The dashboard's first panel is a rotatable 3D view of the ontology graph the
R-GAT is learning over: nodes placed by their distance from the raw semantic
channels to `SafeLanding`, sized and coloured by their current activation, and
edges weighted by the second layer's attention. It follows the R-GAT through
training epoch by epoch and then follows the live episode step by step, so the
relations the potential leans on -- markers against GNSS as the pad leaves the
frame, for one -- can be read off while the run is still going. Attention is
learned importance, not causal proof.

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
./scripts/check_learner_protocol.sh
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
- Results are not comparable with the fixed-pad runs, and not only because the
  deck moves: the success radius, the episode length, the arena limits and the
  ontology schema all changed with the environment. A potential trained against
  the 13-node schema will not load, by design.
- Satellites are frozen for the duration of an episode. Over fifteen seconds a
  MEO satellite moves well under a degree, so this is not a meaningful
  approximation at episode scale -- but it does mean an outage never clears by
  itself, only by the vehicle moving.
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
