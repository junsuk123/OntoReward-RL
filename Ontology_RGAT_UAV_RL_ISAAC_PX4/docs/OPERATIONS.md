# 실행과 운영

[문서 안내](README.md) · [시스템 개요](SYSTEM_OVERVIEW.md) ·
[아키텍처](ARCHITECTURE.md) · [실제 기체 안전](HARDWARE_SAFETY.md)

## 표준 명령

저장소 루트에서 전체 파이프라인을 실행한다.

```bash
./run.sh
```

이 기본값은 핵심 3-arm, 3-pair, `--stay-open`, robust adaptive R-GAT
품질 프로필을 모두 포함한다. 병렬 구성을 명시하는 개발용 명령은 다음과 같다.

```bash
./run.sh --seminar-fast --parallel-pairs 3 --stay-open
```

`--stay-open`은 학습과 평가가 끝난 뒤에도 Isaac/PX4, RViz와 dashboard를 유지한다.
종료할 때는 실행 terminal에서 `Ctrl-C`를 사용한다.

## 실행 모드

| 명령 | 용도 |
|---|---|
| `./run.sh` | 핵심 3-arm, UAV/UGV 3쌍, 한 Isaac stage, RViz/dashboard, 완료 후 유지 |
| `./run.sh --mode full` | full mode 명시 |
| `./run.sh --mode quick` | 구성·연결·짧은 경로 검사 |
| `./run.sh --seminar-fast` | 핵심 3-arm 축소 실제 비교 |
| `./run.sh --seminar-fast --stay-open` | 완료 후 시각화 stack 유지 |
| `./run.sh --seminar-fast --parallel-pairs 3` | 한 Isaac stage에서 핵심 3-arm 병렬 학습·평가 |

동시에 하나의 flight pipeline만 실행한다. 실행 lock이 중복 `run.sh`를 차단한다.

## 세 쌍 병렬 모드

```bash
./run.sh --seminar-fast --parallel-pairs 3 --stay-open
```

| Pair | PX4 DDS namespace | Gateway/learner UDP | Simulator topic root |
|---:|---|---|---|
| 0 | `/fmu` | `14650` / `14651` | `/landing_pair_0` |
| 1 | `/px4_1/fmu` | `14652` / `14653` | `/landing_pair_1` |
| 2 | `/px4_2/fmu` | `14654` / `14655` | `/landing_pair_2` |

각 pair는 독립 reset, 카메라, contact, trajectory와 PPO optimizer를 갖는다. DDS Agent와
Isaac physics stage만 공유한다. R-GAT artifact가 없으면 pair 0에서 reward-design 단계를
먼저 완료하며, 동결 이후 세 PPO가 함께 시작된다. 실행 중 공통 simulator가 멈추면 같은
장애를 본 worker 중 하나만 stack을 재시작하고 모든 partial trajectory는 폐기한다.
세 UGV는 `parallel.route_phase_fractions`의 0/8/16% 지점부터 동일 campus waypoint를
따라가며, 도로 밖으로 평행 이동한 복제 경로를 사용하지 않는다. 공유 장애로 연결만
끊긴 worker는 자신의 복구 횟수를 소모하지 않고 새 stack generation에 다시 연결한다.
학습 방법↔pair 대응은 replicate별로 순환하고, 최종 평가는 같은
scenario/seed에서 각 방법을 세 물리 pair에 교차 배정한다.

## 단계

1. Workspace와 dependency 검사
2. Config resolve와 hash/manifest 생성
3. DDS Agent, Isaac Sim/Pegasus, PX4 SITL, gateway 시작 또는 호환 process 인수
4. RViz와 dashboard 시작
5. Keypoint encoder 준비·검증
6. 공통 BC warm start
7. `shin_se_fixed`, `no_se_fixed` PPO
8. 실제 adaptive reward dataset 수집
9. Hybrid R-GAT 학습·validation·동결
10. `onto_rgat_adaptive_weight_no_se` PPO
11. best/latest checkpoint의 held-out 결정론 비교와 selected checkpoint 저장
12. Pair-crossover paired evaluation과 report 생성

## 재개

같은 config와 결과 경로로 같은 명령을 실행하면 다음을 재사용한다.

- 검증된 keypoint encoder
- 완료된 teacher demonstration
- config·pipeline·reward artifact·`ppo_training_contract_id`가 모두 호환되는
  최신 PPO checkpoint/history
- 완료된 R-GAT dataset episode
- 품질 gate를 통과한 동결 R-GAT artifact
- 완료된 paired evaluation row

Config hash, pipeline contract, encoder hash, graph schema 또는 reward artifact가 다르면
기존 checkpoint를 현재 run에 혼합하지 않고 별도 incompatible artifact로 보존한다.
Robust PPO 알고리즘 버전이 달라진 경우도 동일하다. 단, 호환되는 이전 policy를
R-GAT 실제 reward-design trajectory의 출발 행동으로 쓰는 것은 허용하며 이 모델을
최종 비교 checkpoint로 등록하지 않는다.

## 모니터링

Dashboard:

```text
http://127.0.0.1:8770/
```

RViz topic namespace:

```text
/landing_rl/pair_0
/landing_rl/pair_1
/landing_rl/pair_2
```

상태 확인:

```bash
./Ontology_RGAT_UAV_RL_ISAAC_PX4/scripts/stack_status.sh
curl -fsS http://127.0.0.1:8770/api/state
```

Dashboard 항목:

- 공용 stage와 세 pair 각각의 독립 activity
- 물리 pair별 `현재 정책`과 `학습 배정` 정책의 분리 표시
- 세 pair별 committed episode와 live step
- pair별 전체/task/PBRS/active-perception reward
- pair별 FOV/keypoint, 상대 XYZ, UAV/UGV 속도와 battery reserve
- pair별 SE 오차, estimator-free 시각 신호 또는 R-GAT potential
- pair별 PX4 namespace, UDP endpoint, camera/RViz topic
- pair별 landing gate 6항목 통과 여부
- 3열 MATLAB 색상 학습·평가 비교와 R-GAT relation attention

웹 dashboard는 단일 pipeline용 레거시 view로 전환하지 않는다. 좁은 모바일 화면을
제외하면 pair 카드, pair별 실시간 plot과 전체 비교 plot 모두 세 열을 유지한다.
학습 단계의 `학습 배정`은 해당 물리 pair의 기본 방법이다. 최종 crossover
평가에서는 정책이 seed마다 세 물리 pair를 순환하므로, 카드 제목과 색은
`current_method`/“현재 정책”을 따른다. 성공률과 `evaluation/per_episode.csv` 행도
물리 pair의 학습 배정이 아니라 실제로 실행된 `method`에 귀속된다.

RViz는 세 annotated landing camera, 세 독립 TF(`landing_pad_0..2`, `uav_body_0..2`),
pair별 UAV/UGV trail과 landing gate를 동시에 표시한다. Isaac GUI는 1280×720에서 세
pair의 중심과 실제 점유 폭으로 camera 거리·높이를 조절한다.

Committed episode는 checkpoint/history 기록이 끝난 episode다. 현재 비행의 step과 구분한다.

## 로그

Runtime process log:

```text
/tmp/ontology_rgat_stack/
```

주요 파일:

| 파일 | 내용 |
|---|---|
| `isaac.log` | world, PX4 backend, marker/contact, render loop |
| `gateway.log` | ROS/PX4 state, command, reset/stop protocol |
| `rviz.log` | RViz startup와 display |
| `dashboard.log` | dashboard server |

Result directory의 `manifest.json`, training CSV, reward JSONL이 연구 결과의 기준 기록이다.

## 정상 stage에서 움직임이 없는 경우

R-GAT optimization은 저장된 graph dataset을 사용하는 offline stage이므로 UAV/UGV가
움직이지 않는 것이 정상이다. Dashboard stage가 `R-GAT training`인지 확인한다. PPO,
teacher flight, reward-data collection, evaluation stage에서는 실제 vehicle이 움직인다.

## 진단 표

| 증상 | 확인 | 조치 |
|---|---|---|
| Episode가 증가하지 않음 | dashboard의 live step/stage, training CSV | 현재 episode가 끝날 때 commit되므로 live step이 변하면 대기 |
| Dashboard port 사용 중 | `curl /api/state`, process command | 같은 run의 dashboard면 재사용; 다른 server만 정확한 PID로 종료 |
| Entry hover 실패 | gateway/Isaac log의 offset, speed, marker quality | stale Isaac/PX4 process 여부를 확인하고 소유한 stack만 재시작 |
| PX4 estimator invalid | odometry timestamp와 estimator validity | simulator/PX4 cold start 후 readiness 대기 |
| Simulated clock stall | Isaac render/physics log와 timestamp | 부분 trajectory 폐기 후 runner-owned stack 재시작 |
| Offboard heartbeat loss | gateway failsafe reason | control path 복구 후 같은 seed 재시도 |
| FOV loss 증가 | marker quality, keypoint visibility, camera view | marker/camera가 아니라 policy 문제인지 semantic telemetry로 분리 |
| 접촉했지만 실패 | landing gate columns | lateral, pre-contact vertical/relative speed, tilt, rate를 각각 확인 |
| R-GAT artifact 거부 | quality gate와 hash | 기본 실행은 실제 rollout을 24→36→48회로 자동 보강·재학습; 하드캡 실패는 즉시 중단 |

인프라 오류가 난 episode의 부분 trajectory는 PPO와 reward-design dataset에 기록하지 않는다.
Policy/geometry/착륙 실패는 실제 학습 결과이므로 실패 class로 유지한다.

## 결과 확인

세미나 profile:

```text
Ontology_RGAT_UAV_RL_ISAAC_PX4/results/seminar_fast/core3_hybrid_v3/
```

```bash
find Ontology_RGAT_UAV_RL_ISAAC_PX4/results/seminar_fast/core3_hybrid_v3 \
  -maxdepth 3 -type f | sort
```

핵심 결과:

- `manifest.json`
- `models/<pipeline>/<pipeline>.best.pt`
- `models/<pipeline>/<pipeline>.selected.pt`
- `models/<pipeline>/<pipeline>_training.csv`
- `rgat/adaptive_reward_weights.pt`
- `evaluation/checkpoint_selection.csv`
- `evaluation/crossover_plan.csv`
- `evaluation/per_episode.csv`
- `tables/*.csv`
- `figures/*`

## 검증

다음 정적 검사는 실행 중인 simulator의 상태를 변경하지 않는다.

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/check_workspace.sh
```

Fake gateway가 필요한 learner protocol 검사와 gateway loopback 검사는 flight pipeline이
정지한 상태에서만 실행한다.

```bash
./scripts/check_learner_protocol.sh
./scripts/sync_gateway.sh
./scripts/check_ros2_loopback.sh
```

## 안전한 종료

Runner가 소유한 process는 실행 terminal의 `Ctrl-C`가 다음 순서로 정리한다.

1. 현재 episode 수집 중단
2. gateway command 종료
3. RViz/dashboard 종료
4. runner가 시작한 Isaac/PX4 종료
5. checkpoint와 완료 row 보존

다른 세션이 소유한 호환 DDS Agent 등 인수한 process는 임의로 종료하지 않는다.
