# 구현 안내: 온톨로지 R-GAT 착륙 정책

이 디렉터리는 [상위 README](../README.md)에서 설명한 제안 방법의 실행 코드이다.
방법, 기여, 결과는 상위 README에 있다. 이 문서는 다음 세 가지를 정리한다.

- 알고리즘의 각 부분이 어느 모듈에 있는가
- 실험을 어떤 순서로 실행하고, 각 단계가 무엇을 남기는가
- 결과를 믿기 위해 코드가 강제하는 규칙

## 1. 알고리즘 ↔ 코드

경로는 모두 `python/ontology_rgat/` 기준이다.

| 구성 요소 | 모듈 | 내용 |
|---|---|---|
| 현재 관측 `O_t` | `current_semantics/observation.py` | 21채널 관측의 정규화와 복원 |
| 의미 특징 φ와 노드 입력 | `current_semantics/features.py`, `geometry.py` | 44개 의미 특징, 유효성 규칙, 8 × 10 노드 행렬, 카메라 좌표계 패드 기하 |
| 착륙 온톨로지 | `current_semantics/graph.py` | 개념 8개, 관계 유형 5개, 의미 간선 14개 + 자기 루프 |
| R-GAT 인코더와 정책 | `current_semantics/models.py` | 관계 어텐션 층(2홉 공유), LandingContext 판독, actor-critic, 비교 정책 세 가지 |
| 보상 | `current_semantics/reward.py` | 접근 속도장 퍼텐셜, 시간·제어·기울기 비용 |
| PPO | `current_semantics/ppo.py`, `pilot.py` | 완결 에피소드 배치, 목표 KL, 성공률 기반 커리큘럼, 체크포인트 선택 |
| 시나리오 | `current_semantics/scenario.py` | 패드 방위 A × 진행 방향 B, 학습 제외 조합, 패드 가속 |
| 항법 열화 | `current_semantics/navigation.py`, `spatial/wmon.py`, `spatial/kinematic_kf.py` | GNSS 결측 창과 자기 항법 추정기. `wmon` = FFT-assisted kinematic WLS 추정기(저자의 별도 연구, 실험 환경 구성 요소), `kalman` = CA-KF. 표시 이름은 `navigation.ESTIMATOR_DISPLAY_NAMES` |
| 플랜트와 판정 | `current_semantics/backend.py`, `spatial/dynamics.py`, `spatial/environment.py` | Isaac 응답에 보정한 로컬 플랜트, Isaac/PX4 백엔드, 착륙 판정기 |
| 실험 절차 | `current_semantics/development.py`, `confirmation.py`, `isaac_development.py` | 개발 단계, 확인 실험, Isaac 폐루프 평가 |
| 계약과 해시 | `current_semantics/contracts.py`, `experiment.py` | 관측·특징·그래프·과제·학습 설정의 해시, 실행 폴더, 단일 실행 잠금 |
| Isaac 월드 | `../isaac_sim/landing_world.py`, `pad_motion.py`, `vn100_imu.py`, `handover_perturbation.py` | 착륙 월드, 패드 운동, IMU, 인계 교란 |

설정 파일은 `config/current_semantics/`에 있다.

| 파일 | 역할 |
|---|---|
| `spatial_view_contact_v1.yaml` | 기준 항법 설정 (과제, 관측, 모델, 학습) |
| `spatial_view_contact_v1_nav4_wmon.yaml` | 같은 설정 + Kinematic WLS 자기 항법, GNSS 결측 (파일명의 `wmon`은 내부 식별자) |
| `spatial_view_contact_v1_nav4_kalman.yaml` | 같은 설정 + CA-KF 자기 항법, GNSS 결측 |
| `development_v1.yaml` | 개발 단계 계획 (학습 시드 5개, 평가 셀 10개, 판정 기준) |
| `confirmation_mini_v1.yaml` | 축소 확인 실험 계획 (새 학습 시드 6개, 모델당 300 업데이트, 사전 등록한 판정 규칙, Isaac 케이스). `confirm` 단계의 기본값 |
| `confirmation_v1.yaml` | 전체 규모 확인 실험 계획 (새 학습 시드 20개, 600 업데이트, Isaac 2,400회). `--plan`으로 지정 |
| `isaac_development_v1.yaml`, `isaac_development_v2.yaml` | 개발 모델의 Isaac 평가 케이스 |

## 2. 실험 절차

모든 명령은 저장소 루트에서 `./run.sh current-semantics-3d --config <설정> --stage <단계>`로 실행한다.
각 단계의 산출물은 설정마다 정해진 결과 폴더(`results/current_semantics_3d/<과제>[_nav4_<해시>]/`)에 쌓인다.

| 순서 | 단계 | 하는 일 | 산출물 |
|---|---|---|---|
| 1 | `audit`, `fixtures` | 관측·특징·그래프 계약 점검 | `audit.json`, `fixtures.json` |
| 2 | `coupling --backend local` (`isaac`) | 축 결합, 시야 이탈, 접지 높이 확인 | `coupling_local.json` |
| 3 | `learnability` | 공통 PPO로 착륙이 학습되는지 확인 | `g4_pilot.json` |
| 4 | `register-development` → `develop` | 4개 정책 × 학습 시드 5개 학습, 로컬 평가 10개 셀 | `development.json` |
| 5 | `register-isaac-development` → `isaac-development` | 개발 모델 전부를 Isaac/PX4에서 비행 | `isaac_development*.json` |
| 6 | `register-confirmation` → `confirm` | 새 학습 시드로 같은 절차 반복 | `confirmation_mini/` (전체 규모는 `confirmation/`) |
| 7 | `isaac-confirm` | 확인 모델 전부를 새 Isaac 케이스로 비행 | `confirmation_mini/isaac/` |
| 8 | `confirm-report` | 사전 등록 규칙으로 판정 | `confirmation_mini/confirmation.json` |

- 등록 단계는 학습 전에 계획 파일의 해시를 계약 해시와 묶는다. 이후 단계는 계획이 하나라도 바뀌면 거부한다.
- 확인 실험은 개발 판정을 통과한 결과 폴더에서만 열린다.
- `confirm --shard K/N`은 학습 시드를 N개 프로세스로 나눈다. 각 프로세스는 따로 잠금을 잡는다.
  모든 에피소드에 시드가 정해져 있으므로 `--workers`(롤아웃 프로세스 수)는 결과를 바꾸지 않는다.
- `isaac-confirm --wait-for-models`는 학습이 끝난 시드부터 비행한다. PX4 failsafe나 시뮬레이터 시계 정지 같은
  인프라 중단이 나면 스택을 다시 띄워 이어서 비행하고, 판정이 난 에피소드는 다시 비행하지 않는다.
- `--dry-run`은 어느 단계에서든 아무것도 만들지 않고 해석된 계획만 출력한다.

예시 (Kinematic WLS 항법 축소 확인 실험, 4개 샤드가 정책 하나씩 학습):

```bash
C=$PWD/Ontology_RGAT_UAV_RL_ISAAC_PX4/config/current_semantics/spatial_view_contact_v1_nav4_wmon.yaml
./run.sh current-semantics-3d --config $C --stage register-confirmation
for k in 0 1 2 3; do
  ./run.sh current-semantics-3d --config $C --stage confirm --shard $k/4 --workers 6 &
done
./run.sh current-semantics-3d --config $C --stage isaac-confirm --allow-isaac --headless --wait-for-models
./run.sh current-semantics-3d --config $C --stage confirm-report
./run.sh current-semantics-3d --stage monitor        # 읽기 전용 진행 현황 (http://127.0.0.1:8780/)
Ontology_RGAT_UAV_RL_ISAAC_PX4/scripts/monitor_tailscale.sh start   # 같은 모니터를 Tailscale 주소로 (tailnet 안에서만 접속)
```

모니터(`current_semantics/monitor.py`)는 읽기 전용이며 다음을 5초마다 갱신한다.

- 확인 실험: 시드 × 정책별 학습·평가 진행, Isaac 비행 진행, 인프라 중단, 판정
- 학습 곡선: 모델별 학습 성공률, 난이도, 위험 비율, 검증 점수
- **R-GAT 인코딩** (`current_semantics/encoding_view.py`)
  - 그래프 정책(R-GAT, Untyped GNN)의 최신 체크포인트로 프로브 에피소드 하나를 결정론적으로 비행한다.
  - 결정 시점마다 기록하는 값은 다음과 같다.
    - 간선별 어텐션 α (홉별)
    - 노드 상태 크기 ‖h‖
    - 8차원 LandingContext 판독 g_t
    - 정책이 본 패드 추정
  - 이 값들을 그래프, 막대, 시간 히트맵으로 재생한다. 학습 중 검증에서 체크포인트가 바뀌면 다시 계산한다.
  - 프로브 시드(999901)는 등록된 어떤 시드와도 겹치지 않는다.
  - 이 에피소드는 아무것도 학습하거나 쓰지 않으며, 비교 판정의 근거가 아니다.

Isaac Sim + PX4 스택을 띄우는 단계는 `--allow-isaac`를 줄 때만 실행된다. 이미 떠 있는 Isaac은 넘겨받지 않고 거부한다.

## 3. 결과를 믿기 위한 규칙

코드가 강제하는 규칙은 다음과 같다.

- **참값 격리:** 시뮬레이터 참값은 보상과 착륙 판정에만 들어간다. actor, critic, 그래프, 관측 경로에는 들어가지 않는다.
- **같은 조건:** 네 정책은 보상, 판정, PPO 설정, 커리큘럼, 학습 예산, 학습·평가 시드, 가치함수 설계가 같다.
  다른 것은 actor 입력의 표현뿐이다.
- **사전 등록:** 판정 규칙과 시드는 학습 전에 등록한다. 시드는 이미 쓴 모든 범위와 겹치지 않는지 로더가 확인한다.
- **분석 단위:** 분석 단위는 학습 시드이다. 같은 시드의 네 정책은 같은 평가 환경을 비행하고, 시드마다 짝지은 차이를 계산한다.
  한 셀의 환경 100개는 독립 반복 100개로 세지 않는다.
- **결정론적 평가:** 평가는 결정론적 정책으로 한다. 학습 가능성 점검에서는 확률적 정책도 함께 평가한다.
- **실패 처리:** 체크포인트를 고르지 못한 모델도 빼지 않는다. 그 모델의 성공은 0으로 센다.
- **인프라 실패와 정책 실패의 분리:** 인프라 실패(스택 시작 실패, PX4 failsafe, 시계 정지)는 판정에 넣지 않고 같은 케이스를 다시 비행한다.
  정책이 비행한 뒤 나온 판정은 버리지 않는다.

## 4. 테스트

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
python3 -m pytest -q tests/current_semantics
```

계약 해시, 시드 감사, 판정 통계(짝지은 t 구간, Holm), 단계 거부 규칙을 확인한다.
학습이나 비행은 하지 않는다.

## 5. 다른 실행 경로

저장소에는 재현을 위해 보존한 다른 실행 경로도 들어 있다. 2축 기준 이식, 이전 3축 경로, 최소 관측 ROS 파이프라인이 그것이다.
목록은 `./run.sh --help`로 볼 수 있다. 이 경로들은 위 제안 방법의 결과를 만드는 데 쓰이지 않는다.
