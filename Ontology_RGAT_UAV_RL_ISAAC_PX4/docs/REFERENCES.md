# 참고문헌과 구현 근거

[문서 안내](README.md) · [제안 알고리즘](ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md) ·
[아키텍처](ARCHITECTURE.md)

## 연구 baseline

- W. Shin, J. Kim, J. Park, J. Bae, S. Kim, and H. Oh, “Vision-Based Autonomous
  Drone Landing on Moving Platforms With Uncertain Motion via Deep Reinforcement
  Learning,” *IEEE Robotics and Automation Letters*, vol. 11, no. 5, 2026,
  DOI [10.1109/LRA.2026.3674011](https://doi.org/10.1109/LRA.2026.3674011).
  논문-코드 대응은 [SHIN2026_BASELINE.md](SHIN2026_BASELINE.md)에 정리한다.
- J. Schulman et al., “Proximal Policy Optimization Algorithms,” 2017,
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
- A. Y. Ng, D. Harada, and S. Russell, “Policy Invariance Under Reward
  Transformations,” ICML, 1999. PBRS에서 PPO와 같은 $\gamma$를 사용하는 근거다.

## 관계형 graph attention

- D. Busbridge, D. Sherburn, P. Cavallo, and N. Y. Hammerla, “Relational Graph
  Attention Networks,” 2019, [OpenReview](https://openreview.net/forum?id=Bklzkh0qFm).
- [babylonhealth/rgat](https://github.com/babylonhealth/rgat), Apache-2.0 reference
  implementation. 현재 저장소는 같은 개념을 PyTorch로 구현하며 `NOTICE`에 출처를
  기록한다.
- P. Veličković et al., “Graph Attention Networks,” ICLR, 2018,
  [arXiv:1710.10903](https://arxiv.org/abs/1710.10903).

## PX4와 ROS 2

- [PX4 simulation](https://docs.px4.io/main/en/simulation/): SITL과 simulator 연결
- [PX4 ROS 2 Offboard control](https://docs.px4.io/main/en/ros2/offboard_control):
  pre-stream, mode/arm, setpoint와 NED 규약
- [PX4 uXRCE-DDS bridge](https://docs.px4.io/main/en/middleware/uxrce_dds):
  client/agent transport와 `px4_msgs` release 일치
- [px4_msgs](https://github.com/PX4/px4_msgs): PX4 ROS 2 message 정의
- [Micro XRCE-DDS Agent](https://github.com/eProsima/Micro-XRCE-DDS-Agent):
  UDP DDS agent
- [ROS 2 Humble DDS implementations](https://docs.ros.org/en/humble/Installation/DDS-Implementations.html):
  Fast DDS RMW 설정

## Isaac Sim과 Pegasus

- [Isaac Sim 5.1 ROS 2 standalone workflow](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_python.html)
- [Isaac Sim camera sensor](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html)
- [Isaac Sim ROS 2 reference architecture](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/ros2_reference_architecture.html)
- [Pegasus PX4 integration](https://pegasussimulator.github.io/PegasusSimulator/source/features/px4_integration.html)
- [OpenUSD rigid-body API](https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html)

## Landing marker와 vehicle

- [OpenCV ArUco detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html):
  dictionary, black border, quiet zone와 corner refinement
- [OpenCV planar pose methods](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html):
  planar ambiguity와 IPPE square
- [AGILEX RANGER MINI](https://global.agilex.ai/products/ranger-mini): UGV platform
- [AGILEX `ugv_gazebo_sim`](https://github.com/agilexrobotics/ugv_gazebo_sim):
  RANGER MINI URDF/visual mesh 원본

## 시각화와 계산

- [RViz 2 User Guide](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)
- [ROS 2 visualization markers](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-Custom-Display/RViz-Custom-Display.html)
- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)

저장소의 다이어그램은 코드와 설정에서 확정된 interface를 설명한다. 실제 경로 PNG는
현재 waypoint 설정에서 생성하고, runtime screenshot은 Isaac/RViz/dashboard의 실제
실행 화면을 사용한다.
