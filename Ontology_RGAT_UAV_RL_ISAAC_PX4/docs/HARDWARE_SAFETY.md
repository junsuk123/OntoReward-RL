# 실제 기체 안전 Gate

[문서 안내](README.md) · [시스템 개요](SYSTEM_OVERVIEW.md) · [아키텍처](ARCHITECTURE.md) ·
[운영](OPERATIONS.md) · [참고문헌](REFERENCES.md)

이 코드는 thrust를 명령할 수 있다. 실제 기체를 사용할 때는 독립된 조종자와
검증된 수동 제어권 회수 경로가 반드시 필요하다.

> **현재 범위:** 기본 recurrent `shin_se / no_se / onto_no_se` 실행기는
> SITL에서만 검증되었다. 현재 `python/run_hardware_policy.py`는 별도의 legacy
> cooperative policy `results/models/ppo_rgats_pbrs_external.pt`를 불러오며,
> 기본 3-pipeline recurrent checkpoint를 불러오지 않는다. 이 경계를 우회하려고
> 기본 checkpoint의 이름을 바꾸거나 옮기지 말 것. Image/LSTM actor용 실제 기체
> 배포 adapter, camera preprocessing, 실제 비행 검증 계획은 별도로 구현하고
> 검토해야 한다.

- 최초 통신, frame, estimator와 mode 시험에서는 모두 propeller를 제거한다.
- Propeller 장착 전에 PX4 geofence, 고도 제한, RC loss, data-link loss와
  offboard-loss 동작을 설정한다.
- 기체 질량과 hover thrust를 검증한다. Simulation 기본값 `px4.hover_thrust`는
  Isaac의 Pegasus Iris에 맞춘 값이지 실제 비행 calibration이 아니다.
  `tools/calibrate_hover_thrust.py`도 arm 후 자동 상승하는 SITL 도구이므로 실제
  기체를 대상으로 실행하면 안 된다.
- Disarm 상태에서 손으로 기울여 quaternion/frame 부호를 확인한다.
- 이동 target에서는 propeller를 켜기 전에 UGV pose/twist 축 정렬과 timestamp
  freshness를 독립적으로 확인한다. UAV marker pose가 맞아도 deck velocity가
  틀리면 위험한 closing command가 생긴다.
- 자유 비행 전에 기체를 고정한 thrust stand를 사용한다.
- UDP command port를 localhost 또는 보호된 companion network 밖에 노출하지 않는다.
- Hardware mode는 reset과 auto-arm을 비활성화한다. Arm에는 독립된 opt-in 두 개가
  필요하며, 그래도 조종자의 정상 control path에서 수행해야 한다.
- Offboard 전환에는 `--allow-offboard`와
  `ONTOLOGY_RGAT_HARDWARE_OFFBOARD=I_ACCEPT_FLIGHT_CONTROL`이 모두 필요하다.
  Action packet은 setpoint를 미리 보낼 수 있지만 두 조건 없이는 mode를 바꾸지 못한다.

## 실제 기체에서 gateway가 거부하는 명령

다음 제한은 조종자가 기억하기를 기대하는 규칙이 아니라
`ontology_rgat_px4/safety.py`에서 강제하고 `tests/test_config_safety.py`로
검사하는 규칙이다.

| 명령 | `target=sitl` | `target=hardware` |
|---|---|---|
| `reset` | 허용 | **항상 거부** — 실제 기체에는 reset할 simulator가 없다. |
| `goto` (자동 상승) | 허용 | 두 offboard opt-in이 있어도 **항상 거부** — 조종자가 entry pose로 비행한다. |
| `arm` | `--allow-arm` 필요 | `--allow-arm`과 **`ONTOLOGY_RGAT_HARDWARE_ARM=I_ACCEPT_PROPELLER_RISK` 모두 필요** |
| `enable_offboard` | 허용 | `--allow-offboard`와 **`ONTOLOGY_RGAT_HARDWARE_OFFBOARD=I_ACCEPT_FLIGHT_CONTROL` 모두 필요** |

`disarm`은 gate로 막지 않는다. 비행 중에는 PX4가 공중 disarm을 거부하는 것이
정상이므로 gateway도 disarm 대신 `NAV_LAND`를 명령하고 `landing_requested`로
응답한다. 기체를 offboard-loss failsafe에 그대로 맡기지 않기 위한 동작이다.

## 필수 perception 입력

Localization node가 `/landing_uav0/perception/uav_pose_in_pad`에
`geometry_msgs/PoseStamped`를 발행하는 동안만 hardware state를 유효하게 본다.
Position은 ENU landing-pad frame에서 표현한 UAV 원점이어야 한다. `[0,1]` 범위의
marker confidence는 `/landing_uav0/perception/marker_quality`에 발행해야 한다.
Pad pose가 없거나 오래되면 PX4 EKF 원점을 landing pad로 간주하지 않고 policy
실행을 차단한다. `target=hardware`에서 gateway는 pad-pose freshness를
`estimator_valid`에 포함하고, `ontology_rgat.bridge.PX4Bridge`는 유효하지 않은
state를 거부한다. Freshness 기준은 `system.state_timeout_s`다.

SITL과 달리 hardware에서는 선택 사항이 아니다. `vision.pose_source_for_policy`와
관계없이 최신 pad-relative pose가 항상 policy를 구동한다.

`pad.motion`이 `static`이 아니면 cooperative vehicle/localization node가
`/landing_pad/state/odom`에 `nav_msgs/Odometry`도 발행해야 한다. Pose와 twist는
PX4 local odometry와 같은 world ENU 축을 사용하고 frame 원점은 marker/deck
표면이어야 한다. `pose.covariance[0]`, `[7]`, `[14]`에는 차량이 보고한 정확도를
기록한다. 실제 기체에서는 이것이 fix 품질을 나타내는 유일한 정보이며 ontology가
읽는다. Gateway는 UAV world velocity에서 deck world velocity를 빼고, deck stream이
오래되면 `estimator_valid=false`로 거부한다. Deck yaw는 telemetry일 뿐이며,
학습 frame은 차량과 함께 회전하지 않는 평행 이동된 world ENU다.

`gnss.enabled`가 true면 receiver node가 drone 자체 fix를 JSON으로
`/landing_uav0/gnss/status`에 발행해야 한다. 읽는 필드는 관측 가능한 `valid`,
`fix_type`, `satellites_tracked`, `hdop`, `vdop`, `residual_rms_m`, `sigma_xy_m`,
`cn0_mean_db`, `nlos_detected_fraction`, `quality`뿐이다. Simulator가 오차 주입에
사용하는 `truth` key는 없어야 한다. Topic 누락이나 stale 상태가 policy를 막지는
않는다. Fix loss는 link fault가 아니라 추론 대상 상태이기 때문이다. Gateway는
한 번 log를 남기고 open sky로 보고하므로 integrity 1.00을 믿기 전에 topic이
실제로 동작하는지 확인한다.

Hardware에는 `/landing_pad/state/odom_truth`가 없으며 존재해서도 안 된다.
그래야 learner가 실제 비행에서 얻을 수 있는 sensor만으로 평가된다.

ROS 2 gateway는 PX4 `/fmu/out/battery_status`를 구독하고 hardware에서 우선한다.
Energy-aware 동작을 신뢰하기 전에 `battery.source`가 `px4`인지, 전압과 state of
charge가 타당한지, topic이 최신인지 확인한다. Seeded near-empty battery model은
SITL 실험 장치일 뿐 실제 battery monitor나 PX4 low-battery failsafe를 대체하지 않는다.

MAVLink-only fallback에는 ROS perception, deck, energy 입력이 없다. `reset`과
`goto`를 거부하며 `pad.motion`이 static이 아니면 시작도 거부한다. 따라서
transport/attitude 시험 또는 PX4 local 원점을 고정 landing pad에 의도적으로 맞춘
시스템에만 적합하다. Vision-relative 또는 moving-target 착륙에는 ROS 2 gateway를 쓴다.

## 정책 실행

실제 기체 entry point는 `python/run_hardware_policy.py`뿐이다.
`python/run_pipeline.py`는 무인 arm/비행을 수행하므로 `--target hardware`를 거부한다.

```bash
python3 python/run_hardware_policy.py
```

이 스크립트는 legacy cooperative
`results/models/ppo_rgats_pbrs_external.pt`를 불러오며 현재 observation width는
23이다. `load_agent`가 weight 적용 전에 schema를 확인하므로 이전 fixed-pad model이나
기본 recurrent model을 조용히 전용할 수 없다.

Hardware script는 arm 명령을 보내지 않는다. 먼저 state를 읽고
`battery.source=px4`를 기다리며, 조종자 경로로 기체가 **이미** arm되지 않았다면
오류로 종료한다. Policy 차원도 확인한다. `marker_quality`가 0이면 경고한 뒤
offboard를 활성화하고, 오류나 Ctrl-C를 포함한 모든 종료에서 offboard를 꺼 설정된
PX4 offboard-loss 동작으로 제어를 넘긴다. `armed`가 false가 되는 즉시 멈추므로
조종자 disarm으로 실행을 끝낼 수 있다.

Gateway deadman은 hardware에서 새 action이 wall time 기준
`system.action_timeout_s`(250 ms) 동안 없으면 offboard setpoint를 중단한다.
SITL은 느린 lockstep rendering이나 DDS timestamp jump를 false timeout으로 보지 않도록
PX4 simulated time과 wall time의 공통 age를 쓴다. PX4는 offboard loss에 안전하게
반응하도록 별도 설정해야 하며 gateway가 autopilot failsafe를 대체할 수 없다.
Simulator roof-contact topic은 hardware safety 입력이 아니다. 독립 검증한 실제 pad
switch가 없다면 hardware touchdown 판정은 PX4 land detector에 계속 의존한다.

SITL의 episode 자동 복구도 hardware 기능이 아니다. Hardware gateway timeout,
Offboard loss, estimator fault와 battery warning은 설정된 PX4/조종자 안전 경로에
맡겨야 하며, 실제 비행을 software loop가 자동 재시작하거나 재시도해서는 안 된다.
