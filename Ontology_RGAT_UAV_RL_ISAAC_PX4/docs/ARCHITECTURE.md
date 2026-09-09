# Architecture and migration boundary

## What was replaced

The original `+dynamics`, `+aero`, `+prop`, `+wind`, `+sensor`, and numerical
integration path is no longer called by the external workspace. Its public
episode contract is retained through replacement `matlab/src/+sim` functions:

| Old behavior | External replacement |
|---|---|
| `resetState` creates a MATLAB state | reset transaction to Isaac, then a PX4-flown climb to the entry pose |
| `getCurrent` synthesizes sensors | latest PX4 estimator sample |
| `step` runs RK4 and rotor model | sends PX4 offboard attitude/thrust setpoint and waits one sample |
| panel wind/aero diagnostics | Isaac force callback and ROS 2 environment telemetry |
| analytic marker-visibility proxy | ArUco tags on the pad seen by a downward camera |
| ground clamp/terminal check | PX4 `vehicle_land_detected` plus shared landing criteria |

The ontology graph, R-GAT inference, rewards, policy networks, evaluation tables,
and plotting functions remain sourced read-only from the original project.
`training.generateRGATDataset` is overlaid only to capture its graph template
from the active rollout; the original helper reset would otherwise create a
second armed external environment and collide on the UDP endpoint.
`evaluation.windSweep` is overlaid so each reset forwards the requested wind
scale to Isaac instead of changing the now-unused MATLAB wind structure.
`evaluation.makePlots` uses logged Isaac forces directly; it never reconstructs
panel loads with the retired MATLAB aerodynamic model.
The behavior-policy expert is also overlaid to remove the old analytical
ground-effect feed-forward term; PX4 closes the attitude loop and Isaac supplies
the actual thrust/contact response.

## Interfaces

### Simulator link

Pegasus uses PX4's Simulator MAVLink API: simulated IMU/GPS/ground truth flow to
PX4 and `HIL_ACTUATOR_CONTROLS` flows back to Isaac rotor dynamics. It is not the
same socket as the companion/offboard link.

### Companion link

The preferred gateway uses PX4 uXRCE-DDS and matching `px4_msgs`. It publishes:

- `/fmu/in/offboard_control_mode`
- `/fmu/in/vehicle_attitude_setpoint` (policy actions)
- `/fmu/in/trajectory_setpoint` (pre-episode climb to the entry pose)
- `/fmu/in/vehicle_command`

and consumes:

- `/fmu/out/vehicle_odometry`
- `/fmu/out/vehicle_local_position` (EKF validity flags)
- `/fmu/out/vehicle_status`
- `/fmu/out/vehicle_land_detected`
- `/fmu/out/vehicle_command_ack` (rejected commands are logged, not swallowed)
- `/fmu/out/vehicle_thrust_setpoint` (hover-thrust calibration)

The last four require `patches/px4-v1.14-publish-land-detected.patch`; stock PX4
v1.14 keeps them off the uXRCE-DDS bridge.

Exactly one control source drives `_control_tick` at a time. A `goto` streams
position setpoints until the first `action` arrives, which switches the gateway
to attitude control for the rest of the episode.

### Marker vision

`isaac_sim/marker_vision.py` holds the pad geometry and the pose solve, and
imports no Isaac, so the frame conventions are testable without a simulator.
Three details are load-bearing and each was a real defect first:

- Coplanar points are two-fold ambiguous and a level downward camera over a flat
  pad sits on that degeneracy, so candidate poses are scored here and the branch
  that puts the camera underground is rejected.
- The camera's optical frame is measured from the stage rather than assumed;
  Isaac's `camera_axes` conventions differ between `set_local_pose` and
  `get_world_pose`, and the mismatch aims the camera sideways while every
  readback still looks correct.
- `OmniPBR` enables world-space UV projection in its constructor, which ignores
  the quad's own UVs and crops away the marker's black border and quiet zone.
  A tag without them is not detectable.

The optional MAVLink gateway consumes `LOCAL_POSITION_NED`,
`ATTITUDE_QUATERNION`, and `HIGHRES_IMU`, and sends `SET_ATTITUDE_TARGET`.

### MATLAB link

MATLAB and the gateway exchange one JSON object per UDP datagram. Each object has
`v`, `type`, `seq`, and `time_ns`. Commands are idempotent by sequence number.
State replies include the acknowledged command sequence. Packets with the wrong
version, non-finite values, stale timestamps, or out-of-range actions are
rejected.

## Coordinates

Workspace state uses ENU position/velocity and FLU body rates. PX4 uses NED and
FRD. The exact mappings are:

```text
p_enu = [p_ned.y, p_ned.x, -p_ned.z]
v_enu = [v_ned.y, v_ned.x, -v_ned.z]
w_flu = [w_frd.x, -w_frd.y, -w_frd.z]
```

Quaternion conversion is implemented using rotation matrices and covered by
round-trip tests; no Euler-angle sign shortcuts are used.

## Timing

The gateway owns the 50 Hz offboard stream and monotonic timestamps. MATLAB may
pause briefly without malformed setpoints being repeated forever: after the
action timeout the gateway stops publishing setpoints so PX4's configured
offboard-loss failsafe takes control. Isaac runs physics at 250 Hz and rendering
at 50 Hz.
