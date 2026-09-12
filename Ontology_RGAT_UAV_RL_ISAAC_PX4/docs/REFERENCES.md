# 통합 참고문헌

[문서 안내](README.md) · [시스템 개요](SYSTEM_OVERVIEW.md) · [아키텍처](ARCHITECTURE.md) ·
[운영](OPERATIONS.md) · [실제 기체 안전](HARDWARE_SAFETY.md)

## 연구 baseline

- Shin, Kim, Park, Bae, Kim, and Oh, *Vision-Based Autonomous Drone Landing on
  Moving Platforms With Uncertain Motion via Deep Reinforcement Learning*,
  IEEE Robotics and Automation Letters 11(5), 2026,
  [DOI 10.1109/LRA.2026.3674011](https://doi.org/10.1109/LRA.2026.3674011).
  논문 개념과 실행 코드의 대응 및 의도적인 변경은
  [SHIN2026_BASELINE.md](SHIN2026_BASELINE.md)에 기록했다.

논문은 과학적 참고자료이지 명령어 출처가 아니다. 이 구현이 실제로 실행하는 동작은
저장소 설정과 코드가 결정한다.

## PX4

- [PX4 Simulator MAVLink API](https://docs.px4.io/main/en/simulation/): Pegasus가
  사용하는 sensor, ground-truth, `HIL_ACTUATOR_CONTROLS` message 방향과 lockstep 계약
- [PX4 uXRCE-DDS bridge](https://docs.px4.io/main/en/middleware/uxrce_dds): release와
  일치하는 `px4_msgs`, client/agent 구조, transport option 및
  `patches/px4-v1.14-publish-land-detected.patch`가 확장하는 `dds_topics.yaml`
- [PX4 ROS 2 Offboard example](https://docs.px4.io/main/en/ros2/offboard_control):
  pre-stream, mode/arm command, setpoint topic과 NED 규약
- [PX4 `BatteryStatus` uORB message](https://docs.px4.io/main/en/msg_docs/BatteryStatus):
  hardware energy source로 쓰는 state of charge와 voltage
- [px4_msgs](https://github.com/PX4/px4_msgs): 생성 message 정의. Branch가 firmware
  release와 정확히 일치해야 한다.
- [Micro XRCE-DDS Agent](https://github.com/eProsima/Micro-XRCE-DDS-Agent): agent
  binary, UDP/serial transport와 Fast DDS 의존성

## Isaac Sim 및 Pegasus

- [Isaac Sim 5.1 ROS 2 standalone workflow](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_python.html):
  standalone stepping과 ROS 2 bridge 동작
- [Isaac Sim ROS 2 reference architecture](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/ros2_reference_architecture.html):
  custom Python node와 simulator/ROS process 경계
- [Isaac Sim camera sensor](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html):
  FOV 대신 focal length/aperture를 쓰는 방법, `camera_axes` 규약과
  `get_intrinsics_matrix`
- [Pegasus PX4 integration](https://pegasussimulator.github.io/PegasusSimulator/source/features/px4_integration.html):
  `PX4MavlinkBackend` 설정과 PX4 자동 시작
- [USD rigid bodies](https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html):
  재현 가능한 moving deck에 쓰는 kinematic rigid-body schema

## 실제 기체 구성

- [u-blox ZED-F9P-05B data sheet](https://content.u-blox.com/sites/default/files/documents/ZED-F9P-05B_DataSheet_UBXDOC-963802114-12824.pdf):
  5 Hz multi-constellation navigation rate, 10초 미만 RTK convergence,
  `0.01 m + 1 ppm` RTK horizontal/vertical accuracy
- [VectorNav VN-100 specifications](https://www.vectornav.com/products/detail/vn-100):
  800 Hz IMU, 400 Hz attitude output, ±2000 deg/s gyro, ±16 g accelerometer,
  bias stability와 noise density
- [Stereolabs ZED 2i data sheet](https://support.stereolabs.com/hc/en-us/article_attachments/27901419901463):
  Eye당 1280×720/60 Hz, 2.1 mm lens의 110×70° FOV
- [AGILEX RANGER MINI 3.0](https://global.agilex.ai/products/ranger-mini):
  최대 속도 2 m/s, payload 120 kg, open SDK
- [AGILEX `ugv_gazebo_sim`](https://github.com/agilexrobotics/ugv_gazebo_sim):
  `scripts/import_ranger_mini_v3.sh`가 import하는 고정 BSD `ranger_mini_v3`
  URDF와 visual mesh

## Marker 영상 인식

- [OpenCV ArUco detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html):
  dictionary, 필수 black border/white quiet zone과 corner refinement
- [OpenCV `solvePnPGeneric` 및 planar ambiguity](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html):
  `SOLVEPNP_IPPE_SQUARE`, 두 가지 coplanar-pose ambiguity와 candidate를 그대로
  믿지 않고 score해야 하는 이유

## 도심 협곡의 GNSS

다음은 `isaac_sim/gnss.py`의 근거다. Model은 pseudorange-domain error, weighted
least-squares solution과 C/N0 test로 의도적으로 간결하게 만들었다. 특정 실험을
재현한다는 의미가 아니라 mechanism과 magnitude order의 출처로 인용한다.

- Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation
  Systems*, 2nd ed., Artech House, 2013: pseudorange error budget,
  elevation-dependent weighting, least-squares solution cofactor matrix인 DOP
- Kaplan and Hegarty, *Understanding GPS/GNSS: Principles and Applications*,
  3rd ed., Artech House, 2017: carrier-to-noise density, elevation dependence와
  receiver-reported accuracy
- [Groves, "Shadow Matching: A New GNSS Positioning Technique for Urban
  Canyons", *Journal of Navigation* 64(3), 2011](https://doi.org/10.1017/S0373463311000087):
  `UrbanLayout.blocked_batch`가 수행하는 3D building model 기반 satellite visibility 예측
- [Groves and Jiang, "Height Aiding, C/N0 Weighting and Consistency Checking for
  GNSS NLOS and Multipath Mitigation in Urban Areas", *Journal of Navigation*
  66(5), 2013](https://doi.org/10.1017/S0373463313000350): signal strength,
  consistency check, height aiding의 필요성과 `gnss.vertical_blend`가 작은 이유
- [Hsu, "Analysis and modeling GPS NLOS effect in highly urbanized area", *GPS
  Solutions* 22:7, 2018](https://doi.org/10.1007/s10291-017-0667-9): 조밀한 urban
  canyon에서 측정한 NLOS excess delay와 position error, 기본 설정 magnitude의 근거
- [Parkinson and Axelrad, "Autonomous GPS Integrity Monitoring Using the
  Pseudorange Residual", *NAVIGATION* 35(2), 1988](https://doi.org/10.1002/j.2161-4296.1988.tb00955.x):
  보고 covariance를 확장하는 integrity observable로서 post-fit residual

## 도심 바람

- [Oke, "Street design and urban canopy layer climate", *Energy and Buildings*
  11(1-3), 1988](https://doi.org/10.1016/0378-7788(88)90026-6): height-to-width
  ratio에 따른 street-canyon flow regime와 GNSS model도 계산하는 sky-view factor.
  `wind.canyon`이 평균 flow를 도로 방향으로 channeling하는 근거

## ROS 2

- [ROS 2 Humble RMW 구현](https://docs.ros.org/en/humble/Installation/DDS-Implementations.html):
  모든 `/fmu/*` consumer가 agent와 맞도록
  `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`를 사용해야 하는 이유

## 관계형 graph attention

- [Busbridge, Sherburn, Cavallo and Hammerla, *Relational Graph Attention
  Networks* (2019)](https://openreview.net/forum?id=Bklzkh0qFm):
  `python/ontology_rgat/rgat/layers.py`가 구현한 model. ARGAT/WIRGAT attention
  normalization, additive/multiplicative style, multi-head aggregation과
  relational kernel basis decomposition
- [babylonhealth/rgat](https://github.com/babylonhealth/rgat) (Apache-2.0): 저자의
  reference release. TensorFlow 1.x라 Python 3.10/ROS 2 Humble/Isaac Sim 5.1
  baseline에 설치할 수 없어 dependency 대신 PyTorch로 재구현했다. `NOTICE` 참조
- [Velickovic et al., *Graph Attention Networks*
  (2018)](https://arxiv.org/abs/1710.10903): relation으로 일반화한 additive
  attention score와 multi-head aggregation
- [Ng, Harada and Russell, *Policy Invariance Under Reward Transformations*
  (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf):
  `cfg.reward.pbrs.gamma`와 `cfg.ppo.gamma`가 같아야 하는 이유

기본 비교는 2-layer R-GAT output을 동결 potential로 직접 쓴다. 이전 저장소 그림
일부의 counterfactual-to-eight-coefficient distillation은 legacy cooperative profile에만
속한다.

## GPU 가속

- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html): R-GAT
  trainer가 불필요하게 synchronize하지 않는 asynchronous launch model
- [Ampere 이후 PyTorch TF32](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices):
  `cfg.device.allow_tf32`

## 시각화

- [RViz 2 display type](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)과
  [`visualization_msgs/Marker`](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-Custom-Display/RViz-Custom-Display.html):
  `python/ontology_rgat/viz/rviz.py`가 발행하는 내용
- [Isaac Sim debug drawing](https://docs.isaacsim.omniverse.nvidia.com/latest/utilities/utilities_debug_drawing.html):
  `isaac_sim/live_overlay.py`가 사용하는 extension. 이 extension은 line/point만
  그리므로 overlay에 text가 없는 이유

현재 route figure는 `tools/check_metasejong_route.py`가 설정된 Meta-Sejong S5 USD
road mesh에서 생성한다. Live dashboard screenshot은 결과가 아니라 진행 중 runtime
capture임을 명시한다.
