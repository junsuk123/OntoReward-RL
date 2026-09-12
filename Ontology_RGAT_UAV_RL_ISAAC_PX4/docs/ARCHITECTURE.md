# 아키텍처 및 migration 경계

[문서 안내](README.md) · [시스템 개요](SYSTEM_OVERVIEW.md) · [운영](OPERATIONS.md) ·
[실제 기체 안전](HARDWARE_SAFETY.md) · [참고문헌](REFERENCES.md)

## 문서 범위

저장소에는 의도적으로 분리한 실험 family 두 개가 있다.

| Family | Launcher | Actor/ontology/reward |
|---|---|---|
| **기본 통제 비교** | 저장소 루트 `./run.sh` | 영상 + 7-D UAV proprioception actor, 18-node/35-edge history-aware estimator-free graph, 동결 direct R-GAT `Phi(G)` |
| **Legacy cooperative urban profile** | `scripts/run_metasejong_pipeline.sh` | 23-channel cooperative actor, 14-node/38-edge graph, 증류한 고정 reward weight 8개 |

Simulator/PX4/ROS migration과 interface는 **legacy cooperative**라고 표시하지 않은 한
공통이다. City/GNSS/reward-distillation 절은 보존된 legacy profile을 설명한다. 현재
기본 learning path는 마지막 절과
[THREE_PIPELINE_COMPARISON.md](THREE_PIPELINE_COMPARISON.md)에 정의한다.

## 대체한 구성요소

Migration은 두 단계로 분리된다. 첫 단계는 simulator를 process 밖으로 옮겼고,
두 번째 단계는 learner를 MATLAB에서 Python으로 옮겼다.

### 시뮬레이터: MATLAB rigid body → Isaac Sim + PX4

이전 `+dynamics`, `+aero`, `+prop`, `+wind`, `+sensor` path와 numerical integration은
제거했다. Episode 계약은 `python/ontology_rgat/env.py`에 남아 있다.

| 이전 동작 | 외부 system의 대체 동작 |
|---|---|
| `resetState`가 process 내부 state 생성 | Isaac reset transaction 후 PX4가 entry pose까지 상승 |
| `getCurrent`가 sensor 합성 | 최신 PX4 estimator sample |
| `step`이 RK4와 rotor model 실행 | PX4 Offboard setpoint를 보내고 *simulated time* control period를 기다림. 기본 비교는 velocity/yaw-rate command 사용 |
| Panel wind/aero 진단 | Isaac physics-callback drag force와 ROS 2 environment telemetry |
| 분석식 marker-visibility proxy | Downward camera가 보는 pad의 ArUco tag |
| Ground clamp/terminal check | PX4 `vehicle_land_detected`와 공통 landing criteria |
| 합성 sensor noise(`cfg.sensor.*`) | 제거. PX4 EKF가 noisy simulated sensor를 이미 fusion함 |

`terminal_status`는 `has_been_airborne` flag를 받는다. 실제 flight stack에서는 control
handover 시 vehicle이 pad 위에 실제로 있으므로 이전 ground test를 그대로 쓰면 첫
step을 touchdown으로 오인한다.

Behavior-policy expert의 분석식 ground-effect feed-forward도 제거했다. PX4가 attitude
loop를 닫고 Isaac이 실제 thrust/contact response를 제공하므로 추가 보정은 둘과 충돌한다.

### 학습기: MATLAB → Python

MATLAB workspace와 read-only parent는 실행 경로에서 제외했다.
`python/ontology_rgat/`이 전체 실험이며 파일 mapping은
[`legacy_matlab/README.md`](../legacy_matlab/README.md)에 있다. 주요 설계 선택은 다음과 같다.

- **R-GAT은 graph batch를 처리하는 PyTorch 구현이다.** Busbridge et al. (2019)의
  `babylonhealth/rgat`을 따른다. Reference는 TensorFlow 1.x라 현재 baseline에 설치할
  수 없다(`NOTICE` 참조). ARGAT/WIRGAT, additive/multiplicative attention,
  multi-head aggregation, basis-decomposed kernel을 지원한다.
- **Graph template은 첫 실제 rollout에서 얻는다.** Template만 얻으려고 두 번째
  environment를 reset하면 armed environment가 둘이 되어 UDP endpoint와 충돌한다.
- **Panel aerodynamics를 복원하지 않는다.** Evaluation은 Isaac이 기록한 resultant
  force를 plot하며, 사용 종료 monitor용 panel slot 계약은 제거했다.
- **시각화는 ROS 2와 Python이다.** RViz 2 live 3D view, Isaac in-window overlay,
  독립 web dashboard, publication figure용 matplotlib를 쓴다.
- **Ontology를 두 번 그리는 것은 의도적이다.** RViz는 수동 flat layout,
  `viz/graph3d.py`는 edge list 기반 dashboard depth/ring layout을 쓴다.

## 이전 cooperative 구성

### 도심 환경

`isaac_sim/urban_scene.py`의 단일 `UrbanLayout`은 네 도로가 둘러싼 block, block과
도로 반대편 facade, corner/mid-block cross street를 정의한다. `UrbanScene.spawn`은
collision box를 stage에 만들고 `blocked_batch`는 같은 box로 GNSS line of sight를
검사한다. 따라서 outage 원인이 되는 building이 실제 viewport에도 보인다.

Box가 axis-aligned라 climbing ray가 box를 지나는 horizontal span의 entry point만으로
exact occlusion을 판정할 수 있다. 전체 constellation-city 검사는 약 30 µs다.
Pegasus `Flat Plane`은 ground와 light만 제공하며 machine-readable skyline은 없다.

### 이동 deck

`LandingDeck`은 box collider를 가진 kinematic rigid body로, 6.2×2.45 m box lorry의
도로 위 3.2 m roof다. `PadTrajectory`는 `static`, bounded straight-line, `circular`,
`lissajous`, `road`의 analytic position/velocity를 제공한다. Urban 실험의 `road`는:

- Arc length로 parameterize한 rounded rectangle이라 corner를 포함한 모든 `s`에서
  point, unit tangent, curvature가 closed form이다. Lane offset `e`는 centreline
  rate의 `(1-κe)`배로 진행한다.
- Traffic light마다 추출 depth의 Gaussian speed dip을 쓴다. 1.0이면 완전 정지다.
  Integral이 error function이라 이동 거리도 closed form이다.
- 느린 lane wander와 최대 한 번의 `tanh` lane change가 미분 가능하므로 lateral
  velocity가 impulse가 아니라 bump다.

Episode seed에서 2–8 m/s를 뽑아 `pad_scale`을 곱한다. Scale 0은 lane wander까지
멈춘 같은 seed의 static control이다. Route rectangle은 `urban.block_size_m`를
기본으로 하며 building과 oncoming lane을 침범하지 않는다.

Deck 초기 위치는 `pad.start_position_enu_m`가 아니라 `PadTrajectory.pose(0)`에서
얻는다. Road profile 원점은 building이 있는 block 중심이기 때문이다.
`pad.route_start: continue`는 reset 사이 lorry 위치를 1 mm 이내로 이어가고 운전
양상만 reseed한다. `seeded`는 lap의 새 위치를 뽑아 평균 55 m teleport하므로 위의
UAV와 분리될 수 있다. Deck pose는 physics 250 Hz로 써 PhysX가 moving collider로
본다. Drivetrain model이 아니라 trajectory source다. Marker quad는
`/World/landing_rover` child라 pose, heading, collision geometry와 함께 움직인다.

### GNSS

`isaac_sim/gnss.py`는 episode마다 7° mask 위 satellite 12개를 `sin(el)`에서
균일하게 뽑아 hemisphere에서 균일한 constellation을 만든다. 15초 동안 MEO
satellite 변화가 1°보다 작으므로 episode 안에서는 고정한다.

Receiver update 과정:

1. `UrbanLayout.blocked_batch`가 facade를 지나는 line of sight를 판정한다.
2. Blocked satellite는 `nlos_tracking_probability` 확률로 reflection을 통해 계속
   tracking한다. 생존 draw는 constellation에 있어 두 receiver가 공유한다.
3. Reflection에는 upper bound가 있는 양의 excess delay `2 d cos(el)`을, direct
   signal에는 elevation-weighted diffuse multipath와 thermal noise를 더한다.
4. `[-u,1]` row weighted least squares가 position/clock error를 계산한다. DOP는
   unweighted normal matrix, post-fit residual은 covariance를 늘리는 variance factor다.
5. Satellite별 C/N0는 horizon 쪽에서 낮아지고 reflection에는 지수 분포 dB loss가
   추가된다. 예상보다 `cn0_detection_margin_db` 이상 낮으면 suspect다.

Integrity는 satellite count, DOP, inflated covariance, suspect fraction만으로 만든다.
Satellite 4개 미만이면 outage로 보고 estimate가 coast/drift하며 `valid=false`,
integrity=0이 된다. Reset의 `gnss_scale`은 error mechanism만 곱하고 building은
움직이지 않는다. Scale 0도 **같은 city**의 open-sky 조건이라 geometry와 fix를 분리한다.

`UrbanGnssSensor`가 urban receiver solution을 MAVLink `HIL_GPS`로 변환한다. C/N0나
3-D building shadow mask가 찾은 reflection은 pseudorange variance를 키워 LOS range가
우선되게 한다. PX4 EKF2가 navigation estimate를 소유한다. 큰 EPH/EPV는 fix를 약한
drift-bounding observation으로 만들고, fix loss에서는 IMU dead reckoning, 안정 회복에서는
EKF fusion을 쓴다. Status의 simulator-only `truth`와 `injected_into_px4`가 gateway의
중복 error 적용을 막는다.

Lorry receiver는 `/landing_pad/state/odom`에 자체 error/covariance를 포함한 fix를,
scoring 전용 `/landing_pad/state/odom_truth`에 실제값을 발행한다. 두 receiver는
constellation을 공유하지만 map/C/N0 mitigation은 독립이다. Stale deck broadcast는
`estimator_valid=false`지만 낮은 품질은 추론 대상 state이므로 link fault가 아니다.
Marker가 보이면 pad-relative position을 직접 anchor하고, 사라지면 PX4 및 wheel
odometry velocity로 예측한 뒤 combined receiver variance에 비례한 time constant로
differential GNSS 쪽을 보정한다.

### 에너지 및 학습 계약

SITL `BatteryModel`은 momentum-theory induced power와 avionics draw를
`/fmu/out/vehicle_thrust_setpoint`에서 받아 PX4 simulated time으로 적분한다. Policy
handover 때 episode seed의 9–55 hover-second reserve로 시작해 initial climb 비용은
policy에 부과하지 않는다. Hardware는 PX4 `battery_status`를 사용할 수 있다. Empty
modeled pack은 `battery_depleted` terminal이다.

Legacy learning 계약은 observation 23개, ontology node 14개다. `WindRisk`는 physics
실제값이 아니라 UAV anemometer 측정에서 나온다. `PadMotion`은 alignment, visual
stability, touchdown safety, `SafeLanding`을 낮추고 `BatteryReserve`는 touchdown
safety를 지지한다. `GnssIntegrity`와 `MarkerQuality`는 서로 대체 가능한 alignment
source다. Marker가 보이면 fix 영향이 작고 사라지면 중요하므로 `TouchdownSafety`는
두 신호의 noisy-OR 위에 만든다.

관측 가능한 값은 pad velocity, relative closing speed, normalized reserve,
descent-energy margin, GNSS integrity, suspect fraction, reported horizontal 1-sigma다.
실제 error/NLOS count/sky view는 입력이 아니며 `tests/test_urban_gnss.py`가 검사한다.

**고정 reward design:** R-GAT이 discounted safe-landing outcome을 fitting한 뒤 각
physical term을 neutral value로 바꾼 counterfactual dataset에서 output 절대 변화를
측정한다. Sensitivity 8개를 bounded unit simplex로 projection해 position, vertical
speed, tilt, body rate, wind, pad tracking, energy, navigation coefficient로 동결한다.
PBRS의 `Phi_w=-sum(w_i c_i)`를 정의하며 PPO가 변화하는 attention을 reward로 쓰지
않는다. JSON artifact는 coefficient, range, sensitivity, validation loss, dataset size,
deterministic design ID를 기록한다.

**Optimization 계약:** Nominal paired success가 `eval.acceptance.min_success_rate`
이상이고 wind/pad-motion/GNSS/energy strata success dispersion이 `max_success_std`
이하이며 worst-case success와 R-GAT fit도 limit을 만족해야 proposed policy를 accept한다.

**Scoring:** 긴 outage에서 fused pose가 drift할 수 있어 terminal test와 touchdown
metric만 simulator 전용 UAV/deck stream으로 만든 `env.truth_state`를 쓴다. Degraded
`HIL_GPS`를 fusion한 PX4 odometry는 truth로 재사용하지 않는다. R-GAT은 batched 및
vectorized하며 `cfg.gpu.*`가 device를 정한다. RTX 4060의 보수적 crossover는 batch
1024이고 저장/real-time inference 전에 CPU double로 가져온다.

## 인터페이스

### 시뮬레이터 연결

Pegasus는 PX4 Simulator MAVLink API를 쓴다. Simulated IMU/GPS/ground truth는 PX4로,
`HIL_ACTUATOR_CONTROLS`는 Isaac rotor dynamics로 흐른다. Companion/offboard socket과
다르다. `isaac.lockstep`이 켜져 PX4와 Isaac이 함께 진행한다.

### 바람 및 항력

Pegasus still-air `LinearDrag`를 0으로 바꾸고 Isaac physics callback에서 wind-relative
quadratic drag를 적용한다(`landing_world.py`의 `WindField.force`). Field는 seeded mean,
six-mode turbulence sum, Gaussian gust에 episode별 `wind_scale`을 곱한다. Force는
body frame에 적용하고 validation 전용으로 ENU에 발행한다. 별도 UAV anemometer의
bias/noise/first-order response 측정값만 `WindRisk`, ontology, PPO, reward에 도달한다.

`wind.canyon`은 facade가 평균 flow를 도로 방향으로 모으고 cross-street component를
막는 현상을 구현한다. Street axis는 deck heading이고 gradient wind blend는 sky view다.
Lorry가 corner를 돌면 channel 방향도 돌고 intersection에서 완화된다.

### Companion 연결

기본 gateway는 PX4 uXRCE-DDS와 release가 일치하는 `px4_msgs`를 쓴다.

발행:

- `/fmu/in/offboard_control_mode`
- `/fmu/in/vehicle_attitude_setpoint` — legacy attitude action
- `/fmu/in/trajectory_setpoint` — entry hover와 기본 velocity/yaw-rate action
- `/fmu/in/vehicle_command`

구독:

- `/fmu/out/vehicle_odometry`
- `/fmu/out/vehicle_local_position` — EKF validity flag
- `/fmu/out/vehicle_status`
- `/fmu/out/battery_status`
- `/fmu/out/vehicle_land_detected`
- `/fmu/out/vehicle_command_ack`
- `/fmu/out/vehicle_thrust_setpoint`

마지막 네 topic은 `patches/px4-v1.14-publish-land-detected.patch`가 추가한다. 한 번에
control source 하나만 `_control_tick`을 구동한다. 첫 policy command 전에는 `goto`,
legacy `action`은 attitude control, 기본 `velocity_action`은 position-backed velocity/yaw
control이다. `state.extra.control_source`가 실제 source를 보고한다.

PX4는 충분한 setpoint stream 뒤에만 mode switch를 받아들이므로 `OFFBOARD` 요청은
latch하지 않고 실제 nav state가 `OFFBOARD`(14)가 될 때까지 재전송한다.

`estimator_valid`는 finite number로 추정하지 않고 PX4의 최신 `xy_valid`, `z_valid`,
`v_xy_valid`, `v_z_valid`를 모두 요구한다. 정지 disarm에서 보통 false인
`heading_good_for_control`은 제외해 `extra`에만 보고한다. Hardware에서는 최신 visual
pad pose, non-static target에서는 최신 `/landing_pad/state/odom`도 요구한다.

### Episode 초기화 연결

Reset은 `std_msgs/String` JSON을 쓰는 gateway–Isaac two-topic transaction이다.

- Gateway → Isaac `/landing_sim/reset`:
  `{v, seq, seed, wind_scale, pad_scale, gnss_scale}`
- Isaac → gateway `/landing_sim/reset_ack`: request echo,
  `entry_offset_pad_m`, `entry_position_enu_m`, `entry_rpy_deg`,
  `entry_yaw_enu_rad`, `battery_hover_seconds`, deck state, initial GNSS fix,
  `reseated_on_deck`

Gateway는 이를 `reset_complete` ack의 `detail`로 learner에 전달한다.
`PX4Bridge`가 pad-frame `goto`를 보내고 gateway는 live deck에서 world target을 매 tick
다시 계산한다. Ack에 offset이 없으면 이전 `landing_world.py`로 보고 실패한다.

### 환경 및 인식 telemetry

Legacy cooperative profile은 ZED-F9P-05B RTK 5 Hz, physics limit 250 Hz의 VN-100
IMU, ZED 2i eye 1280×720/60 Hz를 profile한다. 기본 profile은 Shin 호환 512×320
mono camera 30 Hz로 override하고 GNSS를 actor/ontology에서 제거한다.

Isaac의 `/landing_uav0` 주요 topic:

- `/sensors/wind`: UAV anemometer 측정, ontology에 전달되는 유일한 wind
- `/environment/wind`, `/environment/aero_force`: physics/validation 전용 truth
- `/perception/marker_quality`: detector confidence `[0,1]`
- `/perception/uav_pose_in_pad`: pad-frame ENU pose
- `/perception/pad_contact`, `/perception/pad_contact_force`: filtered contact/진단 force
- `/perception/landing_camera/annotated`: board outline/ID, confidence, reprojection error,
  pixel scale와 pad-relative UAV overlay. Miss도 발행한다.
- `/gnss/status`: 관측 가능한 receiver fix와 simulator-only `truth` JSON
- `/state/*`와 TF: Pegasus `ROS2Backend`

Lorry는 `/landing_pad/state/odom`에 broadcast fix를, scoring 전용
`/landing_pad/state/odom_truth`에 실제값을 발행한다. `/state/*` truth는 telemetry와
readiness에만 쓰고 policy에 넣지 않는다.

### Marker 영상 인식

`isaac_sim/marker_vision.py`는 Isaac을 import하지 않아 simulator 없이 frame convention을
시험할 수 있다. 다음 사항이 중요하다.

- Coplanar pose ambiguity candidate를 score하고 camera를 지하에 두는 branch를 거부한다.
- Optical frame은 stage에서 측정하고 aperture 설정 뒤 intrinsic을 다시 읽는다.
- Marker black border/quiet zone을 자르는 `OmniPBR` world-space UV projection은 피한다.

`marker_quality`는 reprojection sharpness, 최대 tag pixel scale, single-tag penalty로
만든 detector 자체 confidence다. Miss는 0.0과 no pose를 보낸다. 기본 비교에서 metric
marker pose는 setup/operator visualization 전용이다. Actor는 raw mono image,
`onto_no_se`는 그 image의 keypoint/heatmap만 쓴다. Annotated image도 같은 detector
호출에서 만들며 simulator truth를 overlay하지 않는다.

Camera pose는 IMU-DR prediction을 대체하지 않고 제한된 correction으로 fusion한다.
`vision.pose_max_step_m`, `vision.pose_reacquire_error_m`,
`vision.fusion_max_correction_m`이 planar-PnP branch jump를 제한한다.

### 학습기 연결

Learner와 gateway는 UDP datagram마다 `v`, `type`, `seq`, `time_ns`가 있는 JSON 하나를
주고받는다. Sequence 이하 중복 command는 `duplicate`로 답하고 무시한다. Version,
finite range, timestamp, action bound를 검사해 오류는 `error` reply로 거부한다.

Command: `hello`, `state`, `action`, `goto`, `arm`, `disarm`, `reset`,
`enable_offboard`, `disable_offboard`. Reply: `state`, `ack`, `error`.

`state`에는 pad-relative position/velocity, world telemetry, pad pose/twist/accuracy,
battery, GNSS, scoring-only truth, quaternion, motion, perception, vehicle status와
다음 `extra`가 있다.

| `extra` key | 의미 |
|---|---|
| `position_source` | `uav_pose_in_pad` 또는 `px4_local_minus_deck_gnss` |
| `control_source` | `action`, `goto`, `velocity_action`, `idle` |
| `offboard_active` | 실제 PX4 `OFFBOARD` 여부 |
| `control_mapping` | `hover_thrust`, `collective_span`, `max_roll_pitch_rad`, `max_yaw_rate_rad_s` |
| `land_detector` | `live`, `stale`, `missing` |
| `pad_contact_raw` | 현재 non-latched contact |
| `pad_contact` | Takeoff 뒤 latch한 touchdown contact |
| `px4_landed` | 원본 PX4 land-detector 결과 |
| `touchdown_source` | `pad_contact`, `px4_land_detector`, `none` |
| `px4_thrust` | Hover calibration용 normalized body thrust |
| `px4_battery` | 최신 `[remaining_fraction, voltage]` |
| `last_command` | 최신 ack의 `[command, result]` |
| `heading_good_for_control` | `estimator_valid`와 별도로 보고하는 PX4 flag |

`PX4Bridge`는 `hello`에서 `control_mapping`과 learner limit을 비교해 다른 action scale을
거부한다. `goto`는 world radius 140 m 또는 pad offset 10 m, ceiling 25 m, positive
altitude, hold 120 s로 제한한다. Pad-frame request는 최신 deck stream도 요구하며
offset을 pad arena에서 먼저 clamp한 뒤 city radius를 검사한다.

## 좌표계

Workspace는 ENU position/velocity와 FLU body rate, PX4는 NED/FRD를 쓴다.

```text
p_enu = [p_ned.y, p_ned.x, -p_ned.z]
v_enu = [v_ned.y, v_ned.x, -v_ned.z]
w_flu = [w_frd.x, -w_frd.y, -w_frd.z]
```

Quaternion은 rotation matrix로 변환하고 round-trip test한다. Learning frame은 yaw와
함께 도는 body frame이 아니라 **평행 이동한 pad-ENU**다.

```text
p_policy = p_uav_world - p_deck_world
v_policy = v_uav_world - v_deck_world
```

Lorry가 yaw해도 축은 world ENU와 나란하며 rotating-frame Coriolis term이 없다.

## 시간 동기화

Gateway가 50 Hz offboard stream과 monotonic timestamp를 소유한다. Isaac physics는
250 Hz(`physics_dt=0.004`), render는 0.02 s마다 한다. Episode clock은 wall time이
아니라 PX4 simulated clock이다. `PX4Bridge.paceToControlPeriod`가 `cfg.sim.dt`만큼
진행할 때까지 poll하고 이전 deadline 기준으로 다음 period를 계산해 jitter 누적을
막는다. XRCE-DDS timesync domain switch는 연속 logical clock으로 재고정한다.
`cfg.external.timeout` 안에 simulated time이 진행하지 않으면 stall을 보고한다.

Action deadman 만료 시 gateway가 setpoint 발행을 멈춘다. Hardware는 wall time
`system.action_timeout_s` 250 ms, SITL은 `system.sitl_action_timeout_s` 1 s와
wall/PX4 simulated age 중 작은 값을 쓴다. Pending `goto`는 자체 `hold_s`에 만료된다.

## 의도적으로 직접 효과가 없는 설정

`config/system.yaml`을 process 셋이 공유하므로 모든 key를 모두가 읽지는 않는다.

- `landing.success_*`, `landing.ground_z_m`: learner의 `cfg.sim.*`/`cfg.criteria.*`가
  episode 기준을 강제한다. 중복 criteria 일치는 test한다.
- `px4.estimator_warmup_s`: 실제 적용값은 `cfg.external.estimator_warmup`이다.
- `network.matlab_host`, `network.matlab_port`: reply는 datagram source로 보낸다.
- `vision.max_range_m`, `vision.tilt_scale_deg`, `vision.xy_scale_m`: `pose_proxy`만 쓴다.

## 기본 Shin-2026 benchmark path

기본 non-cooperative benchmark는 raw 512×320 grayscale frame과 UAV body
velocity/attitude만 actor에 넣는다. Deck odometry/GNSS, wheel odometry, V2V, marker
pose, simulator pad truth는 actor 입력이 아니다.

세 pipeline은 동결·live-Isaac 검증 6-keypoint encoder, 512-unit LSTM, 256-D latent,
`y[6:256]+proprioception`, velocity/yaw-rate action 4개와 asymmetric critic을 공유한다.
`shin_se`만 `y[0:6]` six-state head, auxiliary MSE와 active reward를 쓴다.

`onto_no_se`는 non-metric bounded observation 12개로 18-node/35-edge graph를 만든다.
Low-confidence heatmap은 geometry를 hallucinate하지 않고 invalid 처리한다. 폭 24 R-GAT
layer 2개의 direct output을 PBRS용으로 동결하며 linear coefficient로 증류하지 않는다.

기본 world는 Meta-Sejong S5/Gwanggaeto다. RANGER MINI가 37-point, 99.70 m route를
0.25–0.60 m/s로 달리고 1.5×1.5 m deck에 크기 3종 ArUco tag 45개가 있다.

![기본 S5 도로와 UGV route](images/metasejong_gwanggaeto_ugv_route.png)

### 초기화 및 복구

UAV는 공중 staging 뒤 PX4가 camera-visible pad-relative entry hover로 비행한다.
모든 curriculum, reward-design rollout, evaluation episode는 pad가 처음 보이는 조건이다.
Pitched-camera footprint 밖 horizontal offset만 줄이고 altitude/yaw는 유지한다. Handover는
position tolerance, 최대 0.40 m/s, 최근 2.0 s marker detection을 1.0 s 연속 요구한다.
UGV는 estimator/climb 동안 정지하고 policy handover 뒤 움직인다.

측정 중 gateway가 Offboard stream을 소유한다. SITL에서 learner action deadline을 놓치면
optimization 때문에 unrelated Offboard-loss landing이 생기지 않도록 position hold로
바꾼다. 순수 Offboard heartbeat loss, gateway timeout, 실제 simulated-clock stall은
불완전 trajectory를 버리고 owned stack을 재시작해 같은 seed를 재시도한다. Mixed/vehicle
safety failsafe는 hard failure다.

FOV loss 자체는 terminal이 아니다. Episode마다 loss/reacquisition event, conditional
rate/time, blind 상태 climb command, low-confidence descent command, 그리고
loss→reacquisition 뒤 landing 여부를 기록한다.
