# Integration references

## PX4

- [PX4 Simulator MAVLink API](https://docs.px4.io/main/en/simulation/): sensor,
  ground-truth, and `HIL_ACTUATOR_CONTROLS` message directions used by Pegasus,
  and the lockstep contract.
- [PX4 uXRCE-DDS bridge](https://docs.px4.io/main/en/middleware/uxrce_dds):
  release-matched `px4_msgs`, client/agent architecture, transport options, and
  the `dds_topics.yaml` that `patches/px4-v1.14-publish-land-detected.patch`
  extends.
- [PX4 ROS 2 Offboard example](https://docs.px4.io/main/en/ros2/offboard_control):
  pre-stream, mode/arm commands, setpoint topics, and NED convention.
- [PX4 `BatteryStatus` uORB message](https://docs.px4.io/main/en/msg_docs/BatteryStatus):
  remaining state of charge and voltage used as the hardware energy source.
- [px4_msgs](https://github.com/PX4/px4_msgs): the generated message
  definitions; the branch must match the firmware release exactly.
- [Micro XRCE-DDS Agent](https://github.com/eProsima/Micro-XRCE-DDS-Agent):
  the agent binary, its UDP/serial transports, and its Fast DDS dependency.

## Isaac Sim and Pegasus

- [Isaac Sim 5.1 ROS 2 standalone workflow](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_python.html):
  standalone stepping and ROS 2 bridge operation.
- [Isaac Sim ROS 2 reference architecture](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/ros2_reference_architecture.html):
  custom Python nodes and simulator/ROS process boundaries.
- [Isaac Sim camera sensor](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html):
  focal length/aperture instead of field of view, `camera_axes` conventions, and
  `get_intrinsics_matrix`.
- [Pegasus PX4 integration](https://pegasussimulator.github.io/PegasusSimulator/source/features/px4_integration.html):
  `PX4MavlinkBackend` configuration and PX4 auto-launch.
- [USD rigid bodies](https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html):
  kinematic rigid-body schema used for the reproducible moving deck.

## Marker vision

- [OpenCV ArUco detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html):
  dictionaries, the required black border and white quiet zone, and corner
  refinement.
- [OpenCV `solvePnPGeneric` and planar ambiguity](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html):
  `SOLVEPNP_IPPE_SQUARE`, the two-fold coplanar-pose ambiguity, and why
  candidates must be scored rather than trusted.

## GNSS in urban canyons

The sources behind `isaac_sim/gnss.py`. The model is a deliberately compact one
-- pseudorange-domain errors, a weighted least-squares solution, and a C/N0
test -- so these are cited for the mechanisms and the orders of magnitude, not
for a reproduction of any particular experiment.

- Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation
  Systems*, 2nd ed., Artech House, 2013: the pseudorange error budget, the
  elevation-dependent weighting, and DOP as the cofactor matrix of the
  least-squares solution.
- Kaplan and Hegarty, *Understanding GPS/GNSS: Principles and Applications*,
  3rd ed., Artech House, 2017: carrier-to-noise density, its elevation
  dependence, and receiver-reported accuracy.
- [Groves, "Shadow Matching: A New GNSS Positioning Technique for Urban
  Canyons", *Journal of Navigation* 64(3), 2011](https://doi.org/10.1017/S0373463311000087):
  predicting satellite visibility from a 3D building model, which is what
  `UrbanLayout.blocked_batch` does.
- [Groves and Jiang, "Height Aiding, C/N0 Weighting and Consistency Checking for
  GNSS NLOS and Multipath Mitigation in Urban Areas", *Journal of Navigation*
  66(5), 2013](https://doi.org/10.1017/S0373463313000350): why signal strength
  and consistency checking are the receiver's handles on NLOS, and why height
  aiding matters -- the reason `gnss.vertical_blend` is small.
- [Hsu, "Analysis and modeling GPS NLOS effect in highly urbanized area", *GPS
  Solutions* 22:7, 2018](https://doi.org/10.1007/s10291-017-0667-9): measured
  NLOS excess delays and positioning errors in a dense urban canyon, the source
  for the magnitudes the shipped configuration produces.
- [Parkinson and Axelrad, "Autonomous GPS Integrity Monitoring Using the
  Pseudorange Residual", *NAVIGATION* 35(2), 1988](https://doi.org/10.1002/j.2161-4296.1988.tb00955.x):
  post-fit residuals as the integrity observable, which is what inflates the
  reported covariance here.

## Urban wind

- [Oke, "Street design and urban canopy layer climate", *Energy and Buildings*
  11(1-3), 1988](https://doi.org/10.1016/0378-7788(88)90026-6): street-canyon
  flow regimes as a function of height-to-width ratio, and the sky view factor
  -- the same geometric quantity the GNSS model computes. The basis for
  `wind.canyon` channeling the mean flow along the street.

## ROS 2

- [ROS 2 Humble RMW implementations](https://docs.ros.org/en/humble/Installation/DDS-Implementations.html):
  why every `/fmu/*` consumer needs `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` to
  match the agent.

## Relational graph attention

- [Busbridge, Sherburn, Cavallo and Hammerla, *Relational Graph Attention
  Networks* (2019)](https://openreview.net/forum?id=Bklzkh0qFm): the model
  `python/ontology_rgat/rgat/layers.py` implements -- ARGAT and WIRGAT attention
  normalisation, additive and multiplicative attention styles, multi-head
  aggregation, and basis decomposition of the relational kernels.
- [babylonhealth/rgat](https://github.com/babylonhealth/rgat) (Apache-2.0): the
  authors' reference release. It is TensorFlow 1.x and cannot be installed on
  this Python 3.10 / ROS 2 Humble / Isaac Sim 5.1 baseline, so the layer here is
  a PyTorch reimplementation rather than a dependency. See `NOTICE`.
- [Velickovic et al., *Graph Attention Networks*
  (2018)](https://arxiv.org/abs/1710.10903): the additive attention score and
  multi-head aggregation the above generalises to relations.
- [Ng, Harada and Russell, *Policy Invariance Under Reward Transformations*
  (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf):
  why `cfg.reward.pbrs.gamma` must equal `cfg.ppo.gamma`.

## GPU acceleration

- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html):
  the asynchronous launch model the R-GAT trainer avoids synchronising against.
- [PyTorch TF32 on Ampere and later](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices):
  `cfg.device.allow_tf32`.

## Visualization

- [RViz 2 display types](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)
  and [`visualization_msgs/Marker`](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-Custom-Display/RViz-Custom-Display.html):
  what `python/ontology_rgat/viz/rviz.py` publishes.
- [Isaac Sim debug drawing](https://docs.isaacsim.omniverse.nvidia.com/latest/utilities/utilities_debug_drawing.html):
  the extension `isaac_sim/live_overlay.py` uses, and why the overlay has no
  text (the extension draws lines and points only).
