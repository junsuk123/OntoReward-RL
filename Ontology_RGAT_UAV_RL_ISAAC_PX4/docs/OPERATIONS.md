# 실행과 운영

[문서 안내](README.md) · [시스템 개요](SYSTEM_OVERVIEW.md) ·
[아키텍처](ARCHITECTURE.md) · [실제 기체 안전](HARDWARE_SAFETY.md)

## 표준 명령

저장소 루트에서 전체 파이프라인을 실행한다.

```bash
./run.sh --mode full
```

짧은 세미나 비교:

```bash
./run.sh --seminar-fast --stay-open
```

`--stay-open`은 학습과 평가가 끝난 뒤에도 Isaac/PX4, RViz와 dashboard를 유지한다.
종료할 때는 실행 terminal에서 `Ctrl-C`를 사용한다.

## 실행 모드

| 명령 | 용도 |
|---|---|
| `./run.sh` | 기본 full pipeline |
| `./run.sh --mode full` | full mode 명시 |
| `./run.sh --mode quick` | 구성·연결·짧은 경로 검사 |
| `./run.sh --seminar-fast` | 핵심 3-arm 축소 실제 비교 |
| `./run.sh --seminar-fast --stay-open` | 완료 후 시각화 stack 유지 |

동시에 하나의 flight pipeline만 실행한다. 실행 lock이 중복 `run.sh`를 차단한다.

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
11. Paired evaluation과 report 생성

## 재개

같은 config와 결과 경로로 같은 명령을 실행하면 다음을 재사용한다.

- 검증된 keypoint encoder
- 완료된 teacher demonstration
- 호환되는 최신 PPO checkpoint/history
- 완료된 R-GAT dataset episode
- 품질 gate를 통과한 동결 R-GAT artifact
- 완료된 paired evaluation row

Config hash, pipeline contract, encoder hash, graph schema 또는 reward artifact가 다르면
기존 checkpoint를 현재 run에 혼합하지 않고 별도 incompatible artifact로 보존한다.

## 모니터링

Dashboard:

```text
http://127.0.0.1:8770/
```

RViz topic namespace:

```text
/landing_rl
```

상태 확인:

```bash
./Ontology_RGAT_UAV_RL_ISAAC_PX4/scripts/stack_status.sh
curl -fsS http://127.0.0.1:8770/api/state
```

Dashboard 항목:

- 현재 stage와 pipeline
- committed episode와 live step
- episode return과 엄격 성공률
- FOV/keypoint/semantic 상태
- UAV/UGV 속도와 command
- battery reserve
- adaptive weight, potential, relation attention
- landing gate별 통과 여부

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
| R-GAT artifact 거부 | quality gate와 hash | dataset class/split 또는 R-GAT 학습 품질을 개선한 뒤 재생성 |

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
- `models/<pipeline>/<pipeline>_training.csv`
- `rgat/adaptive_reward_weights.pt`
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
