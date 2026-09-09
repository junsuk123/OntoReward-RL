# Operations

## Startup order

1. Start QGroundControl or ensure PX4's RC/GCS arming checks are intentionally
   configured for the test.
2. Start the Micro XRCE-DDS Agent on UDP 8888.
3. Start Isaac/Pegasus. Pegasus auto-launches PX4 SITL and owns TCP 4560.
   `scripts/run_isaac.sh` opens a window unless `HEADLESS=1`. The world renders
   once per `isaac.rendering_dt` rather than once per physics step; rendering
   every step drives the frame rate to the physics rate and, because PX4 is
   lockstepped to the simulator, slows the flight stack itself.
4. Confirm `/fmu/out/vehicle_odometry` and `/landing_uav0/state/pose` are live.
5. Start the gateway.
6. Run MATLAB smoke test, then one deterministic evaluation episode.

For hardware, replace step 2 with
`scripts/run_dds_agent.sh serial --dev /dev/ttyUSB0 -b 921600`, configure the
flight controller's `uxrce_dds_client` for the same serial link, omit Isaac, and
start the gateway with `--target hardware`.

## Reset semantics

`reset` is accepted only for `target=sitl`. The gateway disarms, publishes a
seeded reset request, and waits for Isaac's matching acknowledgement. Isaac
reseeds the wind field and draws the episode entry pose from the same
distribution as the original in-process simulator, returning it in the
acknowledgement. It does **not** teleport the vehicle.

The entry pose is then flown by PX4: the client sends `goto`, the gateway
streams a `TrajectorySetpoint` position offboard setpoint, arms, and the client
hands over to the policy only once PX4 holds that point inside
`entryTolerance`/`entrySpeedTolerance` for `entrySettle` seconds. The first
`action` message switches the gateway back to attitude offboard control and ends
the climb.

Teleporting is not an option. Pegasus' `PX4MavlinkBackend.reset()` is a
documented no-op, so PX4's EKF integrates straight through any pose jump: a
4.6 m teleport was measured to leave the estimator reporting −1.5 m and still
6 m short of truth several seconds later, while the disarmed vehicle free-fell
back to the pad. Because the controller is contractually fed PX4 estimator data,
that made every episode meaningless. The only case where Isaac still moves the
vehicle is a tip-over, which no climb can recover from; the acknowledgement
reports this as `recovered_from_tipover`.

Two consequences to keep in mind when comparing runs:

- Each reset costs a real-time climb of several seconds, so external `full`
  sweeps are much slower than the in-process simulator.
- The entry attitude and velocity are whatever PX4 settles at, not the seeded
  small perturbations of the original reset. Only the entry *position*
  distribution is reproduced exactly.

Never compare two policies unless both received successful reset
acknowledgements and reached the entry pose.

## Failure diagnosis

- UAV motionless while MATLAB prints `R-GAT epoch`: this is the offline
  potential-training stage. Isaac/PX4 stays online and the disarmed vehicle
  waits on the pad until PPO flight starts. Run `scripts/stack_status.sh`; a
  changing odometry stream confirms the simulation loop is healthy.

- No `/fmu/out/*`: check the DDS agent and that PX4 `uxrce_dds_client` targets
  UDP port 8888. If the topics exist but read as empty, the client is almost
  certainly on a different RMW: the agent speaks Fast DDS, so every consumer
  needs `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and no `CYCLONEDDS_URI`. The run
  scripts set this; an interactive shell may not.
- `PX4 did not hold the entry pose`: the climb never converged. Check that the
  gateway logged no `goto hold expired`, that PX4 reached `OFFBOARD`
  (`nav_state` 14), and that `estimator_valid` is true — the gateway now
  reports false whenever PX4's own `vehicle_local_position` validity flags are
  down, which is the honest answer during an EKF transient.
- Isaac waits for heartbeat: TCP 4560 is occupied or PX4 model/instance does not
  match Pegasus.
- Offboard rejected: stream setpoints before requesting offboard, check QGC/RC,
  and inspect `vehicle_command_ack` in PX4 logs.
- Vehicle dives: verify NED/ENU conversion and negative Z body thrust; do not
  compensate by flipping gains.
- MATLAB timeouts: keep gateway and MATLAB hosts/ports consistent and allow UDP
  only on the loopback interface for a single-machine run.

## Reproducibility

Archive `config/system.yaml`, PX4 ULog, gateway JSONL log, Isaac version, Pegasus
commit, PX4 commit, `px4_msgs` commit, policy MAT file hash, and episode seed.
