# Hardware safety gate

[System overview](SYSTEM_OVERVIEW.md) · [Architecture](ARCHITECTURE.md) ·
[Operations](OPERATIONS.md) · [References](REFERENCES.md)

This code can command thrust. Hardware use requires an independent pilot and a
tested manual takeover path.

- Remove propellers for all first communication, frame, estimator, and mode tests.
- Set PX4 geofence, altitude limit, RC loss, data-link loss, and offboard-loss
  actions before installing propellers.
- Verify the vehicle's mass and hover thrust; the simulation default
  (`px4.hover_thrust`, measured against Pegasus' Iris in Isaac) is not a flight
  calibration, and `tools/calibrate_hover_thrust.py` is a SITL tool — it arms and
  flies an autonomous climb, so it must never be pointed at a real vehicle.
- Confirm quaternion/frame signs with a hand-tilt test while disarmed.
- For a moving target, independently verify the UGV pose/twist alignment and
  timestamp freshness before any propeller-on test. A correct UAV marker pose
  with an incorrect deck velocity still produces a dangerous closing command.
- Use a restrained thrust stand before free flight.
- Do not expose the UDP command port beyond localhost or a protected companion
  network.
- Hardware mode disables reset and auto-arm. Arming requires two independent
  opt-ins and should still be performed from the pilot's normal control path.
- Changing into Offboard requires `--allow-offboard` and
  `ONTOLOGY_RGAT_HARDWARE_OFFBOARD=I_ACCEPT_FLIGHT_CONTROL`. Action packets may
  pre-stream setpoints but cannot change flight mode without both.

## What the gateway refuses on hardware

These are enforced in `ontology_rgat_px4/safety.py` and covered by
`tests/test_config_safety.py`, not left to the operator to remember:

| Command | `target=sitl` | `target=hardware` |
|---|---|---|
| `reset` | allowed | **always refused** — there is nothing to reset |
| `goto` (autonomous climb) | allowed | **always refused**, even with both offboard opt-ins present: the pilot flies the entry pose |
| `arm` | needs `--allow-arm` | needs `--allow-arm` **and** `ONTOLOGY_RGAT_HARDWARE_ARM=I_ACCEPT_PROPELLER_RISK` |
| `enable_offboard` | allowed | needs `--allow-offboard` **and** `ONTOLOGY_RGAT_HARDWARE_OFFBOARD=I_ACCEPT_FLIGHT_CONTROL` |

`disarm` is never gated. In flight it is not sent as a disarm at all — PX4
refuses to disarm airborne, and rightly — so the gateway commands `NAV_LAND` and
acknowledges `landing_requested`, rather than leaving the vehicle to the
offboard-loss failsafe.

## Required perception input

Hardware state is considered valid only while a localization node publishes
`geometry_msgs/PoseStamped` on
`/landing_uav0/perception/uav_pose_in_pad`. Its position must be the UAV origin
expressed in an ENU landing-pad frame. Marker confidence in `[0,1]` must be
published on `/landing_uav0/perception/marker_quality`. A missing or stale pad
pose blocks policy execution instead of treating the PX4 EKF origin as the
landing pad: on `target=hardware` the gateway folds pad-pose freshness into
`estimator_valid` itself, and `ontology_rgat.bridge.PX4Bridge` rejects a state
that is not valid. Freshness is judged against `system.state_timeout_s`.

Unlike SITL, hardware is not a choice: the pad-relative pose always drives the
policy when it is fresh, regardless of `vision.pose_source_for_policy`.

For `pad.motion` other than `static`, a cooperative vehicle/localization node
must also publish `nav_msgs/Odometry` on `/landing_pad/state/odom`. Its pose and
twist must use the same world ENU axes as PX4 local odometry; the frame origin
must be the marker/deck surface. Populate `pose.covariance[0]`, `[7]` and `[14]`
with the vehicle's own reported accuracy: on hardware that is the only statement
of how good its fix is, and the ontology reads it. The gateway subtracts deck
world velocity from UAV world velocity and refuses a stale deck stream through
`estimator_valid=false`. Deck yaw is telemetry — the learning frame remains
translated world ENU and does not rotate with the vehicle.

If `gnss.enabled` is true, a receiver node must publish the drone's own fix as
JSON on `/landing_uav0/gnss/status`. Only the observable fields are read
(`valid`, `fix_type`, `satellites_tracked`, `hdop`, `vdop`, `residual_rms_m`,
`sigma_xy_m`, `cn0_mean_db`, `nlos_detected_fraction`, `quality`), and the
`truth` key the simulator uses to inject an error must be absent — on hardware
there is nothing to inject and PX4's estimate is already the real one. A missing
or stale topic does **not** block the policy, because a lost fix is a state to
reason about rather than a link fault; the gateway logs it once and reports open
sky, so verify the topic is live before trusting an integrity figure of 1.00.

There is no `/landing_pad/state/odom_truth` on hardware and there must not be:
without it the learner grades on the sensor, which is all a real flight has.

The ROS 2 gateway subscribes to PX4 `/fmu/out/battery_status` and prefers it on
hardware. Verify that `battery.source` is `px4`, voltage and state of charge are
credible, and the topic is fresh before relying on energy-aware behavior. The
seeded near-empty battery model is a SITL experiment mechanism, not a substitute
for a hardware battery monitor or PX4 low-battery failsafes.

The MAVLink-only fallback has no ROS perception, deck, or energy input; it
refuses `reset` and `goto`, and refuses to start while `pad.motion` is non-static.
It is therefore suitable for transport/attitude tests or systems that
deliberately align PX4 local origin with a fixed landing pad; use the ROS 2
gateway for vision-relative or moving-target landing.

## Running a policy

`python/run_hardware_policy.py` is the only hardware entry point.
`python/run_pipeline.py` refuses `--target hardware` outright, because it arms
and flies unattended.

```bash
python3 python/run_hardware_policy.py
```

It loads `results/models/ppo_rgats_pbrs_external.pt` and refuses a checkpoint
whose observation width is not this configuration's 20: the old fixed-pad policy
must never be transferred to a real vehicle.

The hardware script never sends an arm command. It reads state first, waits for
`battery.source=px4`, and errors out unless the vehicle is *already* armed
through the pilot's path. It also dimension-checks the policy so an old
fixed-pad model cannot be transferred. It warns if `marker_quality` is zero,
enables offboard, and on exit — including on error or Ctrl-C — disables offboard
so the configured PX4 offboard-loss behavior takes over. It stops the moment
`armed` goes false, so a pilot disarm ends the run.

The gateway deadman stops offboard setpoints after `system.action_timeout_s`
(250 ms) of wall time without a fresh action on hardware. (SITL uses the common
age of PX4 simulated and wall time so slow lockstep rendering or a DDS timestamp
jump does not create a false timeout.) PX4
must be configured to react safely to offboard loss; the gateway cannot
substitute for autopilot failsafes. The simulator's roof-contact topic is not a
hardware safety input: unless a real, independently validated pad switch is
integrated, hardware touchdown continues to depend on PX4's land detector.
