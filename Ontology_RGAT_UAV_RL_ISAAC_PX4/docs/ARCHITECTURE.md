# Architecture and migration boundary

[Documentation map](README.md) · [System overview](SYSTEM_OVERVIEW.md) · [Operations](OPERATIONS.md) ·
[Hardware safety](HARDWARE_SAFETY.md) · [References](REFERENCES.md)

## Scope of this document

The repository contains two intentionally separate experiment families:

| Family | Launcher | Actor/ontology/reward |
|---|---|---|
| **Primary controlled comparison** | repository-root `./run.sh` | image + 7-D UAV proprioception actor; 13-node/25-edge estimator-free graph; frozen direct R-GAT `Phi(G)` |
| **Legacy cooperative urban profile** | `scripts/run_metasejong_pipeline.sh` | 23-channel cooperative actor; 14-node/38-edge graph; eight distilled fixed reward weights |

The simulator/PX4/ROS migration and interfaces below are shared unless a
section says **legacy cooperative**. The city/GNSS/reward-distillation sections
describe that retained legacy profile; the current primary learning path is
specified at the end and in
[THREE_PIPELINE_COMPARISON.md](THREE_PIPELINE_COMPARISON.md).

## What was replaced

Two migrations happened, and they are separate. The first moved the *simulator*
out of process; the second moved the *learner* out of MATLAB.

### The simulator: MATLAB rigid body -> Isaac Sim + PX4

The original `+dynamics`, `+aero`, `+prop`, `+wind` and `+sensor` path, and the
numerical integration it drove, are gone. The episode contract survives in
`python/ontology_rgat/env.py`:

| Old behavior | External replacement |
|---|---|
| `resetState` creates an in-process state | reset transaction to Isaac, then a PX4-flown climb to the entry pose |
| `getCurrent` synthesizes sensors | latest PX4 estimator sample |
| `step` runs RK4 and a rotor model | sends a PX4 Offboard setpoint and waits one control period of *simulated* time; the primary comparison uses velocity/yaw-rate commands |
| panel wind/aero diagnostics | Isaac physics-callback drag force and ROS 2 environment telemetry |
| analytic marker-visibility proxy | ArUco tags on the pad seen by a downward camera |
| ground clamp/terminal check | PX4 `vehicle_land_detected` plus shared landing criteria |
| synthetic sensor noise (`cfg.sensor.*`) | dropped; PX4's EKF already fuses noisy simulated sensors |

`terminal_status` takes a `has_been_airborne` flag. With a real flight stack the
vehicle is genuinely on the pad when control is handed over, and the old ground
test would have called that a touchdown on step one.

The behaviour-policy expert dropped the analytical ground-effect feed-forward
term: PX4 closes the attitude loop and Isaac supplies the actual thrust and
contact response, so a hand-rolled correction on top would fight both.

### The learner: MATLAB -> Python

The MATLAB workspace and its read-only parent are no longer on any execution
path. `python/ontology_rgat/` is the whole experiment, and
[`legacy_matlab/README.md`](../legacy_matlab/README.md) carries the file-by-file
map. Four things are worth stating as design decisions rather than transcription:

- **R-GAT is PyTorch, batched over graphs.** The layer follows Busbridge et al.
  2019 as implemented by `babylonhealth/rgat`, which is TensorFlow 1.x and
  cannot be installed on this baseline; see `NOTICE`. The port is a strict
  generalisation of the MATLAB layer -- ARGAT and WIRGAT, additive and
  multiplicative attention, multi-head aggregation, basis-decomposed kernels --
  with the MATLAB configuration as its default, so numbers stay comparable.
- **The graph template is captured from the first real rollout.** The original
  dataset helper reset a second environment just to obtain one; here that would
  create a second armed environment and collide on the UDP endpoint.
- **Nothing reconstructs panel aerodynamics.** The evaluation plots Isaac's
  logged resultant force. The retired monitors used to spread that force over
  legacy panel slots to keep a drawing contract alive; the contract is gone with
  the drawing.
- **The views are ROS 2 and Python.** RViz 2 for the live 3D view, an Isaac
  in-window overlay, a self-contained web dashboard for unattended progress, and
  matplotlib for the publication figures. See `python/ontology_rgat/viz/`.
- **The ontology is drawn twice, on purpose.** RViz's overlay hangs the graph
  beside the vehicle from a hand-written flat layout that stays readable from
  one viewpoint; `viz/graph3d.py` computes a depth-and-ring layout from the edge
  list for the dashboard's rotatable view, which nothing else in the frame has
  to share space with. A schema that gains a node changes the second
  automatically and the first by hand.

### Legacy cooperative city

`isaac_sim/urban_scene.py` owns one `UrbanLayout`: a block encircled by four
streets, the block itself plus the facades on the far side of each street, cut
by cross streets at the corners and mid-block. `UrbanScene.spawn` builds those
boxes into the stage (collidable, so a policy that flies into a facade hits it)
and `blocked_batch` tests lines of sight against the same boxes for the GNSS
model. One layout, two consumers, by construction: an outage always has a
building in the viewport to blame it on.

The boxes are axis-aligned, which is what makes the occlusion test exact rather
than sampled — within the horizontal span where a climbing ray crosses a box,
its lowest point is at the entry, so one test settles it. A whole constellation
against the whole city is ~30 µs, so it runs at the publication rate.

`Flat Plane` is the Pegasus environment: it supplies the ground plane and the
lighting, and nothing else. No shipped environment comes with a machine-readable
skyline, and one that did would still not be the one the GNSS model masks with.

### Legacy cooperative moving deck

`LandingDeck` is a kinematic rigid body with a box collider — the roof of a
6.2 × 2.45 m box lorry, 3.2 m above the road. `PadTrajectory` provides analytic
position/velocity for `static`, bounded straight-line, `circular`, `lissajous`
and `road` motion. `road` is what the urban experiment runs:

- the route is a rounded rectangle **parameterised by arc length**, so point,
  unit tangent and curvature are closed form at every `s`, including through the
  corners. A curve offset by a constant lane offset `e` advances at `(1 - κe)`
  times the centreline rate, which is the whole of the velocity expression;
- traffic is a Gaussian dip in the speed at each light, of a drawn depth, where
  1.0 is a full stop. Its integral is an error function, so the distance covered
  is closed form too and the deck is never numerically integrated;
- the driver keeps lane with a slow wander and at most one `tanh` lane change,
  both differentiable, so the lateral velocity they add is a bump and not an
  impulse.

The 2–8 m/s speed is drawn from the episode seed, then multiplied by
`pad_scale`; scale zero parks the lorry — including its lane wander — and is the
static control condition with the same seed. The route rectangle defaults to the
`urban.block_size_m` the city was built around, so the lorry cannot drive
through a building, and the lanes it may use are counted out from the
carriageway width — a lane change moves between them and never into oncoming
traffic.

Two placement rules follow from the city being solid, and each was a defect
first. The deck's position at construction comes from `PadTrajectory.pose(0)`
and not from `pad.start_position_enu_m`: for the road profile those are
different points, because the route is centred on the block and its origin is
therefore the middle of a building — which is where the deck, and the vehicle
that spawns on its roof, used to be put. And `pad.route_start: continue` leaves
the lorry where it is across a reset, reseeding only how it drives from there,
so the deck pose is continuous to within a millimetre; `seeded` draws a fresh
point on the lap instead and teleports the deck a mean of 55 m out from under
whatever is parked on it. See `docs/OPERATIONS.md` for what `continue` costs in
reproducibility. The deck pose is written at the 250 Hz physics rate so PhysX
sees a moving collider rather than a teleported static surface. It is
deliberately a trajectory source, not a lorry drivetrain model.

The marker quads are children of `/World/landing_rover`, so pose, heading, and
collision geometry move together.

### Legacy cooperative GNSS

`isaac_sim/gnss.py`. A `Constellation` of twelve satellites is drawn per episode
uniformly in `sin(el)` above a 7° mask — the distribution that is uniform over
the hemisphere; drawing elevation uniformly would over-populate the zenith and
make every canyon look better than it is. It is frozen for the episode, because
a MEO satellite moves well under a degree in fifteen seconds.

Per receiver, per update:

1. `UrbanLayout.blocked_batch` decides which lines of sight cross a facade.
2. A blocked satellite is still tracked with probability
   `nlos_tracking_probability`, through a reflection. Which ones survive is a
   property of the facades, so the draw lives on the constellation and is
   **shared** between the two receivers.
3. A tracked reflection carries a strictly positive excess delay `2 d cos(el)`,
   capped; a direct signal carries elevation-weighted diffuse multipath (a
   first-order Gauss–Markov process, so it wanders rather than flickers) and
   thermal noise.
4. Weighted least squares over `[-u, 1]` rows gives the position and clock
   error, the classical DOP comes from the unweighted normal matrix, and the
   post-fit residuals give an a-posteriori variance factor that inflates the
   receiver's own reported covariance.
5. Carrier-to-noise ratio is computed per satellite: it falls toward the horizon
   and a reflection costs an exponentially-drawn number of dB on top. A signal
   more than `cn0_detection_margin_db` below its elevation's expectation is
   flagged suspect. This is the receiver's only handle on NLOS when every
   satellite is reflected off facades the same distance away — their biases then
   agree with each other and no consistency check sees anything wrong.

The reported integrity is built from satellite count, DOP, the inflated
covariance and the suspect fraction, all observables. Fewer than four satellites
is an outage: the estimate coasts and drifts, `valid` goes false and integrity
goes to zero.

`gnss_scale` on the reset request multiplies the error mechanisms and moves no
building, so scale 0 is open sky **in the same city** — the facades still hide
the markers and still channel the wind. That is what isolates the fix from the
geometry in `evaluation.sweeps.gnss_sweep`.

**Where the error is applied.** The generic clean Pegasus GPS is replaced by
`UrbanGnssSensor`, which converts the urban receiver solution and reported
accuracy to MAVLink `HIL_GPS`. Before that conversion, reflections detected by
C/N0 or the OSM/3-D building shadow mask have their pseudorange variance
inflated so LOS ranges dominate. PX4 EKF2 therefore owns the navigation
estimate: large EPH/EPV makes a valid fix a weak, drift-bounding observation, a
true loss of fix makes the filter propagate on IMU dead reckoning, and a stable
recovery is fused back through the EKF instead of being added as a position step
in the ROS gateway. The status
topic still carries the receiver observables for the ontology, plus a simulator-
only `truth` subobject; `injected_into_px4` prevents the gateway applying that
error twice. `/fmu/out/estimator_status_flags`, `estimator_gps_status` and the
raw GPS topic expose the actual fusion decision.

**Two receivers.** The lorry has one too. What it broadcasts on
`/landing_pad/state/odom` is its own fix — errors and all — with its reported
accuracy in the pose covariance; the simulator's truth goes to
`/landing_pad/state/odom_truth`, which only the scoring path reads. The receivers
share a constellation but independently apply map/C/N0 NLOS mitigation, keeping
the nominal differential fix metre-scale while preserving severe low-integrity
regions for the DR path. A 2.45 m roof still requires the optical final anchor.

A moving deck with a stale broadcast makes `estimator_valid=false`; a degraded
one does not, because a bad fix is a state to reason about and not a link fault.
For the pad-relative navigation state, a live marker directly anchors position.
When it disappears, the gateway predicts from PX4 velocity and the cooperative
vehicle's wheel-odometry velocity, then
corrects toward differential GNSS with a time constant proportional to the
combined receiver variance. Thus a good fix recentres quickly, while an urban
20 m fix cannot create a position step or overpower short-term DR.

### Legacy cooperative energy and learning contract

SITL uses `BatteryModel`: momentum-theory induced power plus avionics draw,
integrated on PX4 simulated time from `/fmu/out/vehicle_thrust_setpoint`. The
model starts at policy handover with the 9–55 hover-second reserve Isaac drew
from the episode seed, so the climb is not charged to the policy. Hardware can
adopt PX4 `battery_status`. Every state carries finite energy fields, and an
empty modeled pack ends the episode as `battery_depleted`.

The learning contract is now 23 observations and 14 ontology nodes. `WindRisk`
is derived from the UAV anemometer's measured speed and temporal changes, not
the exact field applied by physics. `PadMotion`
degrades alignment, visual stability, touchdown safety and `SafeLanding`;
`BatteryReserve` supports touchdown safety and contributes to `SafeLanding`;
`GnssIntegrity` supports exactly what `MarkerQuality` supports — the alignment
solved from the pad-relative pose and the touchdown flown on it — and
contributes to `SafeLanding`. It is the *substitutability* of those two that the
relation weights have to learn: with the markers in frame the fix hardly
matters, and the moment they leave it the fix is all there is. `TouchdownSafety`
is therefore built on the noisy-OR of the two rather than on visual stability
alone.

Pad velocity, relative closing speed, normalized reserve, descent-energy margin,
GNSS integrity, the suspect-signal fraction and the reported horizontal
1-sigma are observable. The true error, the true NLOS count and the true sky
view are not, and `tests/test_urban_gnss.py` enforces it.

**Fixed reward design.** R-GAT remains context dependent while fitting the
discounted safe-landing outcome. After training, the learner creates one
counterfactual dataset per physical term by replacing that term with its neutral
value, measures the absolute change in R-GAT output, and projects the eight
sensitivities onto a bounded unit simplex. The resulting position, vertical
speed, tilt, body-rate, wind, pad-tracking, energy and navigation coefficients
are frozen for PPO. They define `Phi_w=-sum(w_i c_i)` inside PBRS; PPO never
reads the changing R-GAT attention as a changing reward. The JSON artifact
records coefficients, physical ranges, sensitivity, validation loss, dataset
size and a deterministic design ID.

**Optimization contract.** The proposed policy is accepted only if two gates
pass: nominal paired-evaluation success exceeds `eval.acceptance.min_success_rate`,
and its success-rate dispersion across wind, pad-motion, GNSS and energy strata
stays below `max_success_std` while worst-case success and R-GAT validation fit
also meet their limits. This prevents a uniformly failing policy from appearing
"consistent" and keeps reward effectiveness separate from R-GAT robustness.

**Scoring.** The fused pad-relative pose can still drift away from truth during
a long outage, so the terminal test and the touchdown metrics run on
`env.truth_state` — the gateway's `truth` block built from the simulator-only
`/landing_uav0/state/odom_truth` and deck truth streams — and nothing else does.
PX4 odometry cannot be reused as truth now that it fuses the degraded HIL_GPS.
A link with no `truth` block
falls back to the sensor, as the fixed-pad experiment always did. R-GAT forward/gradient computation is batched and
vectorized; GPU selection is explicit through `cfg.gpu.*`, with a conservative
RTX 4060 crossover of batch 1024. Models are gathered back to CPU double before
saving or real-time inference.

## Interfaces

### Simulator link

Pegasus uses PX4's Simulator MAVLink API: simulated IMU/GPS/ground truth flow to
PX4 and `HIL_ACTUATOR_CONTROLS` flows back to Isaac rotor dynamics. It is not the
same socket as the companion/offboard link. `isaac.lockstep` is on, so PX4 and
Isaac advance together and simulation speed is flight-stack speed.

### Wind and drag

Pegasus' own still-air `LinearDrag` is replaced with zeros and a wind-relative
quadratic drag is applied in an Isaac physics callback instead
(`WindField.force` in `isaac_sim/landing_world.py`). The field is a seeded mean
plus a six-mode turbulence sum plus configured Gaussian gusts, all scaled by the
per-episode `wind_scale` the reset request carries. The force is applied in the
body frame and republished in ENU for validation plots only. A separate UAV
anemometer adds seeded bias, white noise and a first-order response. Only that
measurement reaches `WindRisk`, the ontology, PPO and rewards, preventing
simulator truth from leaking into the policy.

`wind.canyon` adds the one thing a street does to wind that open ground does
not: the facades channel the mean flow along the carriageway and block most of
the cross-street component. The street axis is the deck's own heading and the
blend back to the gradient wind is the deck's sky view, so the channeling turns
when the lorry turns a corner and relaxes at the intersections — which is
exactly where the GNSS recovers. Only the mean is channeled; the turbulence is
what is left after the facades have finished with it.

### Companion link

The preferred gateway uses PX4 uXRCE-DDS and matching `px4_msgs`. It publishes:

- `/fmu/in/offboard_control_mode`
- `/fmu/in/vehicle_attitude_setpoint` (legacy collective/attitude actions)
- `/fmu/in/trajectory_setpoint` (pre-episode entry hover and primary
  velocity/yaw-rate actions)
- `/fmu/in/vehicle_command`

and consumes:

- `/fmu/out/vehicle_odometry`
- `/fmu/out/vehicle_local_position` (EKF validity flags)
- `/fmu/out/vehicle_status`
- `/fmu/out/battery_status` (hardware state of charge)
- `/fmu/out/vehicle_land_detected`
- `/fmu/out/vehicle_command_ack` (rejected commands are logged, not swallowed)
- `/fmu/out/vehicle_thrust_setpoint` (hover-thrust calibration)

The last four are the ones `patches/px4-v1.14-publish-land-detected.patch` adds;
stock PX4 v1.14 keeps them off the uXRCE-DDS bridge. `vehicle_odometry`,
`vehicle_local_position` and `vehicle_status` are already in stock
`dds_topics.yaml`.

Exactly one control source drives `_control_tick` at a time. A `goto` streams
position setpoints until the first policy command arrives. Legacy `action`
switches to attitude control; primary `velocity_action` switches to
position-backed velocity/yaw control. `state.extra.control_source` reports
which of `goto`, `action`, `velocity_action` or `idle` is live.

Requesting `OFFBOARD` is retried, not latched: PX4 accepts the mode switch only
after it has seen a steady setpoint stream and rejects it outright in some
pre-arm states, so `VEHICLE_CMD_DO_SET_MODE` is re-sent every `control_hz/2`
ticks until `vehicle_status.nav_state` actually reads `OFFBOARD` (14).

`estimator_valid` is PX4's answer, not an inference from finite numbers: it
requires `vehicle_local_position`'s `xy_valid`, `z_valid`, `v_xy_valid` and
  `v_z_valid` to all be set and fresh within `system.state_timeout_s`.
`heading_good_for_control` is deliberately excluded — it is normally false on a
stationary disarmed vehicle, which is the state every episode starts from — and
is reported in `extra` instead. On `target=hardware`, `estimator_valid` also
requires a fresh visual pad pose; every non-static target additionally requires
fresh `/landing_pad/state/odom` so its relative velocity is defined.

### Episode reset link

Reset is a two-topic transaction between the gateway and Isaac, carried as JSON
in `std_msgs/String`:

- gateway → Isaac on `/landing_sim/reset`:
  `{v, seq, seed, wind_scale, pad_scale, gnss_scale}`
- Isaac → gateway on `/landing_sim/reset_ack`: the request echoed plus
  `entry_offset_pad_m`, the instantaneous `entry_position_enu_m`,
  `entry_rpy_deg`, `entry_yaw_enu_rad`, `battery_hover_seconds`, deck state, the
  episode's opening GNSS fixes, and `reseated_on_deck`

The gateway forwards the acknowledgement to the learner as the `detail` of a
`reset_complete` ack, and `bridge.PX4Bridge` sends `entry_offset_pad_m` with a
pad-frame `goto`. The gateway recomputes the world target from the live deck on
every control tick. A reset whose ack carries no offset is an error, not a
default: it means Isaac is running an older `landing_world.py`.

### Profile-specific environment and perception telemetry

The legacy cooperative SITL sensor suite is hardware-profiled rather than ideal: ZED-F9P-05B
multi-constellation RTK at 5 Hz, a VN-100 IMU whose 800 Hz device capability is
sampled at the 250 Hz physics limit, and one ZED 2i eye at 1280 x 720/60 Hz.
The Meta-Sejong carrier references the official AGILEX Ranger Mini V3 mesh;
trajectory ownership remains with the kinematic pad so all seeds are exactly
repeatable. Full values and source links are in `config/system.yaml` and
`docs/REFERENCES.md`.

The primary profile overrides this with the Shin-compatible 512×320 mono
camera at 30 Hz and disables GNSS as an actor/ontology input. It retains the
same PX4/Isaac transport and physical contact/scoring topics.

Isaac publishes, under `/landing_uav0` (`isaac.namespace` + `vehicle_id`):

- `/sensors/wind` (`geometry_msgs/Vector3Stamped`, ENU): the UAV anemometer
  measurement with seeded bias, noise and first-order response; this is the
  only wind value forwarded to ontology/R-GAT/PPO
- `/environment/wind`, `/environment/aero_force` (`geometry_msgs/Vector3Stamped`, ENU):
  simulator truth for physics and validation only
- `/perception/marker_quality` (`std_msgs/Float32`, `[0,1]`)
- `/perception/uav_pose_in_pad` (`geometry_msgs/PoseStamped`, pad-frame ENU)
- `/perception/pad_contact` (`std_msgs/Bool`): physical UAV contact with the
  deck, filtered by the quadrotor body and the yaw-corrected roof footprint
- `/perception/pad_contact_force` (`std_msgs/Float32`, N): net contact force for
  operator diagnostics
- `/perception/landing_camera/annotated` (`sensor_msgs/Image`, `rgb8`): the
  downward camera with detected board outlines/IDs and the solve's confidence,
  reprojection error, pixel scale and pad-relative UAV position overlaid. A miss
  is also published and labelled, so loss of recognition is visually distinct
  from loss of the camera stream.
- `/gnss/status` (`std_msgs/String`, JSON): both receivers' fixes. Observables at
  the top level, the simulator's truth under `truth` — the gateway forwards the
  first and keeps the second. A custom message would be tidier and would need a
  message package; the reset link already works this way.
- `/state/*` and TF, from Pegasus' `ROS2Backend` (`pub_state`, `pub_tf`)

The lorry publishes independently of the UAV namespace:
`/landing_pad/state/odom` is what it broadcasts about itself, with its own GNSS
error in the pose and its reported accuracy in the covariance;
`/landing_pad/state/odom_truth` is the simulator's, and only the gateway's
scoring path subscribes to it.

Ground truth on `/state/*` is used for experiment telemetry and readiness checks
only. It never reaches the policy.

### Profile-specific marker vision

`isaac_sim/marker_vision.py` holds the pad geometry and the pose solve, and
imports no Isaac, so the frame conventions are testable without a simulator
(`tests/test_marker_vision.py` renders a synthetic pad view and round-trips the
pose across the whole approach). Three details are load-bearing and each was a
real defect first:

- Coplanar points are two-fold ambiguous and a level downward camera over a flat
  pad sits on that degeneracy, so candidate poses are scored here and the branch
  that puts the camera underground is rejected.
- The camera's optical frame is measured from the stage rather than assumed;
  Isaac's `camera_axes` conventions differ between `set_local_pose` and
  `get_world_pose`, and the mismatch aims the camera sideways while every
  readback still looks correct. The intrinsics are likewise read back with
  `get_intrinsics_matrix()` after the aperture is set, so a lens setting that
  did not take cannot silently bias every pose the policy flies on.
- `OmniPBR` enables world-space UV projection in its constructor, which ignores
  the quad's own UVs and crops away the marker's black border and quiet zone.
  A tag without them is not detectable.

`marker_quality` is the detector's own confidence — sharpness from reprojection
error, scale from the largest tag's pixel side, and a small penalty for a
single-tag pose — so the ontology consumes perception health rather than a
function of ground truth. A miss publishes `0.0` and no pose, which is what
makes the gateway fall back to the PX4 estimate.

In the primary comparison the solved metric marker pose is used for setup and
operator visualization, not as an actor or semantic-graph input. The actor sees
the raw mono image; `onto_no_se` uses only keypoints/heatmaps derived from that
image. The camera-centred entry gate may require a recent nonzero marker quality
before the measured episode begins.

The same detector invocation produces the annotated operator image on
`/landing_uav0/perception/landing_camera/annotated`; visualization does not run
the detector a second time and the overlay contains no simulator truth.

Accepted camera poses are fused around the IMU-DR prediction instead of
replacing it. Inter-frame pose changes, post-outage reacquisition error and the
maximum correction applied in one update are bounded by `vision.pose_max_step_m`,
`vision.pose_reacquire_error_m` and `vision.fusion_max_correction_m`. This keeps
a planar-PnP branch change from moving the controller by metres while preserving
the camera as the authoritative local correction.

The optional MAVLink gateway consumes `LOCAL_POSITION_NED`,
`ATTITUDE_QUATERNION`, `HIGHRES_IMU`, `HEARTBEAT` and `EXTENDED_SYS_STATE`, and
sends `SET_ATTITUDE_TARGET`. It has no reset and no perception input.

### Learner link

The learner and the gateway exchange one JSON object per UDP datagram. Each object has
`v`, `type`, `seq`, and `time_ns`. Commands are idempotent by sequence number: a
`seq` that is not newer than the last one is answered `duplicate` and otherwise
ignored. Packets with the wrong version, non-finite values, stale timestamps, or
out-of-range actions are rejected with an `error` reply.

Commands are `hello`, `state`, `action`, `goto`, `arm`, `disarm`, `reset`,
`enable_offboard`, `disable_offboard`. `hello` opens a session and restarts the
command sequence at one. Replies are `state`, `ack` or `error`. The gateway
replies to the address the datagram came from, so `network.matlab_host` and
`network.matlab_port` are recorded for documentation and are not what the reply
is addressed to. Those two keys keep their names because the wire schema does;
the client is `python/ontology_rgat/bridge.py`.

A `state` reply carries `sample_time_ns`, `px4_time_us`, `frame` (always
`ENU_FLU`), `position_frame` (always `pad`), pad-relative `position`/`velocity`,
world telemetry under `world`, deck pose/twist and reported accuracy under
`pad`, the full finite energy record under `battery`, the receiver's own report
under `gnss`, the simulator's pad-relative geometry under `truth` (scoring
only), `quaternion_wxyz`, `angular_velocity`,
`acceleration`, `wind`, `aero_force`, `marker_quality`, `armed`, `nav_state`,
`landed`, `estimator_valid`, `source`, the acknowledged command sequence, and an
`extra` object:

| `extra` key | Meaning |
|---|---|
| `position_source` | `uav_pose_in_pad` or `px4_local_minus_deck_gnss` — which pose the policy is flying |
| `control_source` | `action`, `goto` or `idle` |
| `offboard_active` | PX4 is actually in `OFFBOARD`, not merely asked |
| `control_mapping` | `hover_thrust`, `collective_span`, `max_roll_pitch_rad`, `max_yaw_rate_rad_s` |
| `land_detector` | `live`, `stale` or `missing` (PX4 built without the patch) |
| `pad_contact_raw` | current, non-latched roof contact sample |
| `pad_contact` | touchdown contact latched after the armed UAV first clears the roof |
| `px4_landed` | unmodified PX4 land-detector result |
| `touchdown_source` | `pad_contact`, `px4_land_detector` or `none` |
| `px4_thrust` | PX4's own normalised body thrust, for hover calibration |
| `px4_battery` | latest `[remaining_fraction, voltage]` received from PX4 |
| `last_command` | `[command, result]` from the most recent `vehicle_command_ack` |
| `heading_good_for_control` | PX4's flag, reported but not part of `estimator_valid` |

`bridge.PX4Bridge` compares `control_mapping` against `cfg.rl.collectiveSpan`,
`cfg.rl.maxRollPitch` and `cfg.rl.maxYawRate` during `hello` and refuses to run
against a gateway that scales actions differently. A silently rescaled action is
an invalid experiment, not a degraded one.

`goto` is guard-railed in `protocol.py` independently of the client: a 140 m
world-frame or 10 m pad-offset radius, 25 m ceiling, positive altitude, and a
hold of at most 120 s. The world-frame radius has to reach the far side of the
block the lorry laps. A pad-frame request also requires a fresh deck stream, and
the gateway clamps in the frame the request was made in: the *offset* against
the pad-relative arena, and only then the sum against the city radius it derives
from `urban.block_size_m`. Clamping the sum against the world origin instead —
which is what the fixed-pad gateway did — would drag the vehicle back to the
middle of the block every time the lorry drove away from it.

## Coordinates

Workspace state uses ENU position/velocity and FLU body rates. PX4 uses NED and
FRD. The exact mappings are:

```text
p_enu = [p_ned.y, p_ned.x, -p_ned.z]
v_enu = [v_ned.y, v_ned.x, -v_ned.z]
w_flu = [w_frd.x, -w_frd.y, -w_frd.z]
```

Quaternion conversion is implemented using rotation matrices and covered by
round-trip tests; no Euler-angle sign shortcuts are used. Body-frame odometry
velocity is rotated to NED before conversion, because PX4 may publish either
frame and says which in `velocity_frame`.

The learning frame is **translated pad-ENU**, not a yaw-rotating body frame:

```text
p_policy = p_uav_world - p_deck_world
v_policy = v_uav_world - v_deck_world
```

Its axes stay aligned with world ENU even while the lorry yaws. This makes the
camera solution and PX4 fallback identical without introducing rotating-frame
Coriolis terms. The deck yaw/yaw rate are retained as telemetry.

## Timing

The gateway owns the 50 Hz offboard stream and monotonic timestamps. Isaac runs
physics at 250 Hz (`isaac.physics_dt` 0.004) and renders once per
`isaac.rendering_dt` (0.02); rendering every physics step would drop the frame
rate to the physics rate and, because PX4 is lockstepped, slow the flight stack
itself. Environment telemetry is published on render boundaries.

The episode clock is PX4's simulated clock, never wall time. The gateway answers
a `state` or `action` request as soon as PX4 publishes odometry, which is several
times faster than the control rate, so `bridge.PX4Bridge.paceToControlPeriod`
re-polls until `px4_time_us` has advanced one `cfg.sim.dt`, and measures the next
period from the previous deadline so sampling jitter cannot accumulate. `sim.step`
then advances `env.t` by the simulated time that actually elapsed rather than by
the nominal period. Without this the policy ran far faster than `cfg.sim.dt`
while the clock still charged `cfg.sim.dt` per step, and episodes ran out of
steps before they could land. XRCE-DDS may temporarily remove and reacquire its
Unix-epoch offset when its timesync filter resets. The gateway converts those
raw timestamp domain switches into a continuous logical PX4 clock while
preserving ordinary simulated-time deltas; the bridge also defensively
re-anchors if it is connected to an older gateway. If simulated time fails to
advance within `cfg.external.timeout` of wall time, the bridge reports a
stalled simulator instead of hanging.

The learner may pause briefly without malformed setpoints being repeated forever:
the gateway stops publishing setpoints when the action deadman expires, so PX4's
configured offboard-loss failsafe takes control. Hardware uses
`system.action_timeout_s` (250 ms) of wall time. SITL uses
`system.sitl_action_timeout_s` (1 s) and the smaller of wall and PX4 simulated
action age. Slow lockstep rendering inflates wall age, while DDS backlog can
jump a newly delivered PX4 timestamp; neither alone is a missing controller,
whereas a real pause advances both clocks past the limit. A pending `goto` is
not cancelled by that deadman — the climb precedes the first action — but does
expire at its own `hold_s`.

## Configuration that is deliberately inert

`config/system.yaml` is shared by three processes and not every key is read by
all of them. These are recorded for provenance and changing them has no effect:

- `landing.success_*` and `landing.ground_z_m`: episode criteria are enforced
  from the learner's `cfg.sim.*`/`cfg.criteria.*`. `max_time_s` sets the modeled
  battery reserve normalization, arena/altitude limits guard pad-frame `goto`,
  and `crash_tilt_deg` controls tip-over recovery. `landing.success_xy_m` is
  additionally read by the Isaac overlay, to draw the tolerance ring.
  `tests/test_learner_contract.py` asserts that the duplicated criteria still
  agree with `cfg.criteria`, so "inert" does not drift into "contradictory".
- `px4.estimator_warmup_s`: the warmup that is actually applied is
  `cfg.external.estimator_warmup` in `python/ontology_rgat/config.py`.
- `network.matlab_host`, `network.matlab_port`: the gateway replies to the
  datagram's source address.
- `vision.max_range_m`, `vision.tilt_scale_deg`, `vision.xy_scale_m`: only the
  `pose_proxy` stand-in uses these.
## Controlled Shin-2026 benchmark path

The primary non-cooperative benchmark is documented in
[`SHIN2026_BASELINE.md`](SHIN2026_BASELINE.md). It consumes a raw 512×320
grayscale frame plus UAV body velocity and attitude only.
`/landing_pad/state/odom`, deck GNSS, wheel odometry, V2V velocity, marker pose,
and simulator pad truth do not enter the actor.

All three primary pipelines use the same frozen, live-Isaac-validated
six-keypoint encoder, 512-unit
LSTM, 256-D latent, `y[6:256] + proprioception` actor features, four
velocity/yaw-rate actions, and an asymmetric training critic. Only `shin_se`
constructs a six-state head on `y[0:6]` and uses its auxiliary MSE and
active-perception reward.

`onto_no_se` builds eight bounded, non-metric observations from keypoints,
heatmaps, UAV proprioception, and battery reserve. These become 13 nodes,
12 semantic edges, 13 self-loops, four relation types, and 19 features per
node. Two 24-wide R-GAT layers read `SafeLanding`; their direct output is
frozen for PBRS. No distilled linear coefficient vector is used in the primary
method.

The default world is the Meta-Sejong S5/Gwanggaeto asset rather than the
synthetic city described above. A RANGER MINI follows the audited 37-point,
99.70 m closed road loop at 0.25–0.60 m/s. The 1.5×1.5 m deck carries 45 ArUco
tags across three physical scales. The offline mesh audit is shown below.

![Primary S5 road and UGV route](images/metasejong_gwanggaeto_ugv_route.png)

### Primary reset and recovery path

The UAV is staged airborne and PX4 flies to a camera-centred pad-relative
entry hover. Handover requires bounded position error, speed no greater than
0.40 m/s, and a marker seen within the preceding 2.0 s for a 1.0 s continuous
settle. The UGV stays parked during estimator initialization/climb and moves
only after policy handover.

During a measured episode the gateway owns the Offboard stream. A missed
learner action deadline is converted to a position hold in SITL so optimization
does not turn an unfinished episode into an unrelated Offboard-loss landing.
The gateway still reports PX4 failsafe reasons. A pure Offboard-heartbeat loss,
gateway timeout, or genuine simulated-clock stall is recoverable: the partial
trajectory is discarded, a runner-owned stack is restarted, and the same seed
is retried. Mixed or vehicle-safety failsafes remain hard failures.
