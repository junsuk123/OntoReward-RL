# Hardware safety gate

This code can command thrust. Hardware use requires an independent pilot and a
tested manual takeover path.

- Remove propellers for all first communication, frame, estimator, and mode tests.
- Set PX4 geofence, altitude limit, RC loss, data-link loss, and offboard-loss
  actions before installing propellers.
- Verify the vehicle's mass and hover thrust; the simulation default is not a
  flight calibration.
- Confirm quaternion/frame signs with a hand-tilt test while disarmed.
- Use a restrained thrust stand before free flight.
- Do not expose the UDP command port beyond localhost or a protected companion
  network.
- Hardware mode disables reset and auto-arm. Arming requires two independent
  opt-ins and should still be performed from the pilot's normal control path.
- Changing into Offboard requires `--allow-offboard` and
  `ONTOLOGY_RGAT_HARDWARE_OFFBOARD=I_ACCEPT_FLIGHT_CONTROL`. Action packets may
  pre-stream setpoints but cannot change flight mode without both.

Hardware state is considered valid only while a localization node publishes
`geometry_msgs/PoseStamped` on
`/landing_uav0/perception/uav_pose_in_pad`. Its position must be the UAV origin
expressed in an ENU landing-pad frame. Marker confidence in `[0,1]` must be
published on `/landing_uav0/perception/marker_quality`. A missing or stale pad
pose blocks MATLAB policy execution instead of treating the PX4 EKF origin as
the landing pad.

The MAVLink-only fallback has no ROS perception input. It is therefore suitable
for transport/attitude tests or systems that deliberately align PX4 local origin
with the landing pad; use the ROS 2 gateway for vision-relative landing.

The gateway deadman stops offboard setpoints after 250 ms without a fresh action.
PX4 must be configured to react safely to offboard loss; the gateway cannot
substitute for autopilot failsafes.
