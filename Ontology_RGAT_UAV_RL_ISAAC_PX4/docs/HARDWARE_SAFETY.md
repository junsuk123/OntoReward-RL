# 실제 기체 안전 Gate

[문서 안내](README.md) · [아키텍처](ARCHITECTURE.md) · [운영](OPERATIONS.md)

현재 핵심 3-arm recurrent checkpoint는 Isaac Sim/Pegasus/PX4 SITL용이다. 실제 기체에
배포하려면 camera preprocessing, recurrent policy adapter, timing, frame, battery와
failsafe를 실제 장비에서 별도로 검증해야 한다.

## 배포 전 필수 조건

- 독립된 안전 조종자와 즉시 수동 제어권 회수
- PX4 geofence, 고도·속도 제한, RC/data-link loss action
- Propeller 제거 상태의 통신·frame·mode·setpoint 검사
- Tether/안전망 환경의 저고도 hover와 정지 패드 시험
- Moving UGV를 사용하기 전 정지 패드의 반복 성공
- SITL checkpoint와 실제 camera calibration의 명시적 provenance

## 제어 경계

실제 adapter가 허용할 명령은

$$
a_t=[v_x,v_y,v_z,\omega_z]
$$

뿐이어야 한다. Policy가 actuator 또는 motor thrust를 직접 출력하지 않도록 한다.
다음 제한은 policy와 독립된 safety layer에서 적용한다.

- velocity/yaw-rate saturation
- acceleration/yaw-acceleration limit
- altitude/geofence limit
- stale image/odometry timeout
- low battery return/land
- excessive tilt/rate abort
- marker 장기 상실 시 hover/climb/abort

## 실제 기체용 observation 검사

| 입력 | 확인 항목 |
|---|---|
| Camera | 해상도, grayscale 순서, lens distortion, exposure, timestamp |
| Body velocity | PX4 frame과 actor frame 변환, 단위, 지연 |
| Quaternion | $wxyz$ 순서, ENU/NED 변환, 정규화 |
| Battery | voltage/current calibration, usable capacity, reserve threshold |
| Keypoint | marker 종류별 검출률, FOV edge와 근접 touchdown 가시성 |

Actor observation에 GNSS platform truth, motion-capture relative pose 또는 simulator 전용
field를 추가하면 학습 정보경계와 달라지므로 동일 모델로 간주할 수 없다.

## 단계적 비행 시험

1. Software-in-the-loop 반복 평가
2. Hardware-in-the-loop 또는 bench PX4 연결
3. Propeller 제거 OFFBOARD setpoint 검사
4. Tether hover와 정지 marker tracking
5. 정지 패드 수직 접근·중단·재접근
6. 매우 저속 UGV 추종
7. 제한된 착륙 시도
8. 속도·외란 envelope의 점진적 확대

각 단계에서 emergency stop, RC override, battery/failsafe, log 회수가 검증되어야 다음
단계로 진행한다.

## 착륙 판정과 실제 안전

![엄격한 착륙 gate](images/landing_success_gate.svg)

Simulation의 성공 gate는 연구 평가 기준이며 실제 비행의 독립 safety monitor를
대체하지 않는다. 실제 기체에서는 deck contact detection, motor shutdown 조건,
미끄러짐, rotor/UGV clearance와 사람 접근 통제를 추가해야 한다.

## 배포 경계

`python/run_hardware_policy.py`의 hardware policy와 핵심 3-arm recurrent checkpoint는
서로 다른 observation/model contract다. 핵심 모델을 실제 기체에 배포하려면 전용
recurrent adapter를 구현·검증하고, camera·frame·timing·제어 제한을 deployment
manifest에 기록해야 한다.
