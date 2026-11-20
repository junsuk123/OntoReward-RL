# 구현 안내: ORL 착륙 정책

이 디렉터리는 [상위 README](../README.md)에서 설명한 제안 정책 ORL의 실행 코드이다.
방법, 기여, 결과는 상위 README에 있다. 이 문서는 다음 세 가지를 정리한다.

- 알고리즘의 각 부분이 어느 모듈에 있는가
- 실험을 어떤 순서로 실행하고, 각 단계가 무엇을 남기는가
- 결과를 믿기 위해 코드가 강제하는 규칙

2026-11-20부터 저장소에는 ORL과 그 비교 정책의 경로만 남아 있다. 나머지 경로는 [5절](#5-보관된-경로)의 보관본에 있다.

## 1. 알고리즘 ↔ 코드

경로는 모두 `python/ontology_rgat/` 기준이다.

| 구성 요소 | 모듈 | 내용 |
|---|---|---|
| 현재 관측 `O_t` | `current_semantics/observation.py` | 21채널 관측의 정규화와 복원 |
| 의미 특징 φ | `current_semantics/features.py`, `geometry.py` | 44개 의미 특징, 유효성 규칙, 카메라 좌표계 패드 기하 |
| **ORL** | `current_semantics/orl.py` | 개념·평면·결정 노드 19개, 관계 7종, 거울 부호, 개념별 인코더, 관계 어텐션 3홉, 결정 판독, 네 거울 반사의 평균. 버전 `orl/1`–`orl/5`, 각자 해시 |
| ORL의 시야 규칙과 규칙 대조군 | `current_semantics/view_rule.py`, `rule_control.py` | `orl/4`·`orl/5`의 고정 시야 규칙과, 같은 규칙을 단 Semantic MLP 대조군 |
| 비교 정책 | `current_semantics/graph.py`, `models.py` | PPO(raw), Untyped GNN, 기존 R-GAT(원 관측 우회 있음), Semantic MLP. `models.build_model`이 모든 정책의 단일 생성기 |
| 보상 | `current_semantics/reward.py` | 접근 속도장 퍼텐셜, 시간·제어·기울기 비용 |
| PPO와 학습 | `current_semantics/ppo.py`, `pilot.py` | 완결 에피소드 배치, 목표 KL, 성공률 기반 커리큘럼, 체크포인트 선택, 5 업데이트마다 쓰는 `live.pt` |
| 시나리오 | `current_semantics/scenario.py` | 패드 방위 A × 진행 방향 B, 학습 제외 조합, 패드 가속 |
| 플랜트와 판정 | `current_semantics/backend.py`, `spatial/` | Isaac 응답에 보정한 로컬 플랜트, Isaac/PX4 백엔드, 착륙 판정기 |
| 실험 절차 | `current_semantics/speed.py`, `orl_development.py`, `development.py` | 학습 속도 확인 실험, ORL 개발 비교, 비교 정책의 개발 단계(평가 셀 10개) |
| Isaac 평가 | `current_semantics/isaac_eval.py`, `isaac_development.py`, `live_stack.py` | 소유한 Isaac Sim + PX4 스택에서의 비행과 평가 |
| 계약과 해시 | `current_semantics/contracts.py`, `experiment.py` | 관측·특징·그래프·과제·학습 설정의 해시, 실행 폴더, 단일 실행 잠금 |
| 모니터 | `current_semantics/monitor.py`, `encoding_view.py` | 읽기 전용 진행 현황, 학습 중 가중치로 그리는 실시간 인코딩 보기 |
| Isaac 월드 | `../isaac_sim/` | 착륙 월드, 패드 운동, 센서(IMU, GNSS, 카메라), 인계 교란 |
| 문서 그림과 시연 영상 | `../tools/make_doc_figures.py`, `../tools/isaac_demo_capture.py`, `../tools/demo_tf_relay.py` | 구조·결과 그림, Isaac Sim + RViz2 시연 비행의 녹화와 GIF·MP4·사진 |

설정 파일은 `config/current_semantics/`에 있다.

| 파일 | 역할 |
|---|---|
| `spatial_view_contact_v1.yaml` | 기준 항법 설정 (과제, 관측, 모델, 학습). ORL과 비교 정책 모두 이 계약에서 학습한다 |
| `learning_speed_v1.yaml` | `learning-speed/1`: ORL vs PPO·Untyped GNN, 학습 시드 40–49, 사전 등록 판정 규칙 |
| `orl_development_v1.yaml` – `v5.yaml` | `orl-development/1`–`/5`: 각 ORL 버전을 개발 단계 모델과 짝지은 비교 |
| `development_v1.yaml` | 비교 정책의 개발 단계 계획 (학습 시드 10–14, 평가 셀 10개) |
| `isaac_development_v1.yaml`, `v2.yaml` | 개발 모델의 Isaac 평가 케이스 (Isaac 평가 단계의 형식) |

## 2. 실험 절차

모든 명령은 저장소 루트에서 `./run.sh current-semantics-3d --config <설정> --stage <단계>`로 실행한다.
산출물은 `results/current_semantics_3d/spatial_current_semantics_v1/`에 쌓인다.

| 순서 | 단계 | 하는 일 | 산출물 |
|---|---|---|---|
| 1 | `audit`, `fixtures` | 관측·특징·그래프 계약 점검 | `audit.json`, `fixtures.json` |
| 2 | `coupling --backend local` (`isaac`) | 축 결합, 시야 이탈, 접지 높이 확인 | `coupling_*.json` |
| 3 | `learnability` | 공통 PPO(PPO raw)로 착륙이 학습되는지 확인 | `g4_pilot.json` |
| 4 | `orl-learnability` | 같은 절차로 ORL의 학습 가능성 확인 | `orl*/orl_learnability.json` |
| 5 | `register-speed` → `speed` → `speed-report` | 학습 속도 확인 실험 | `learning_speed/` |
| 6 | `register-orl-development` → `orl-develop` | ORL을 개발 단계 모델과 짝지어 평가 셀 10개에서 비교 | `orl*/orl_development.json` |
| 7 | `isaac-eval`, `isaac-development` | 선택한 모델을 Isaac Sim + PX4에서 비행 (`--allow-isaac`) | `isaac_*.json` |

- 등록 단계는 학습 전에 계획 파일의 해시를 계약 해시와 묶는다. 이후 단계는 계획이 하나라도 바뀌면 거부한다.
- `speed`는 여러 프로세스가 모델을 나눠 맡는다. 모든 에피소드에 시드가 정해져 있으므로 `--workers`는 결과를 바꾸지 않는다.
  결과는 `OMP_NUM_THREADS=1`(run.sh가 설정)일 때 비트 단위로 재현된다.
- `--dry-run`은 어느 단계에서든 아무것도 만들지 않고 해석된 계획만 출력한다.

예시 (학습 속도 확인 실험, 4개 프로세스):

```bash
C=$PWD/Ontology_RGAT_UAV_RL_ISAAC_PX4/config/current_semantics
./run.sh current-semantics-3d --config $C/spatial_view_contact_v1.yaml --stage register-speed --plan $C/learning_speed_v1.yaml
for k in 1 2 3 4; do
  ./run.sh current-semantics-3d --config $C/spatial_view_contact_v1.yaml --stage speed --plan $C/learning_speed_v1.yaml --workers 3 &
done
./run.sh current-semantics-3d --config $C/spatial_view_contact_v1.yaml --stage speed-report --plan $C/learning_speed_v1.yaml
./run.sh current-semantics-3d --stage monitor        # 읽기 전용 진행 현황 (http://127.0.0.1:8780/)
Ontology_RGAT_UAV_RL_ISAAC_PX4/scripts/monitor_tailscale.sh start   # 같은 모니터를 Tailscale 주소로
```

모니터(`current_semantics/monitor.py`)는 읽기 전용이며 다음을 5초마다 갱신한다.

- **학습 속도 실험:** 시드 × 정책별 진행과 난이도 1.0 첫 도달 업데이트, 도달 분포, 정책별 요약, 잠정 비율, 등록 규칙의 판정
- **학습 곡선:** 모델별 학습 성공률, 난이도, 위험 비율, 검증 점수
- **온톨로지 그래프 인코딩** (`current_semantics/encoding_view.py`)
  - 학습 중인 모델은 `live.pt`(5 업데이트마다), 끝난 모델은 `best.pt`·`final.pt`로 프로브 에피소드 하나를 결정론적으로 비행한다.
  - ORL은 19개 노드 그래프의 간선별 어텐션 α(홉별), 노드 상태 크기, 결정 출력(μ_x, μ_y, μ_z, gate, hold, gate·descend)을 보여 준다.
    그래프는 네 거울 복사본 중 원래 관측의 것이고, 출력은 정책이 쓰는 네 복사본의 평균이다.
  - 간선은 다른 노드를 피해 가도록 곡선으로 그린다.
  - 프로브 시드(999901)는 등록된 어떤 시드와도 겹치지 않는다. 이 에피소드는 비교 판정의 근거가 아니다.

Isaac Sim + PX4 스택을 띄우는 단계는 `--allow-isaac`를 줄 때만 실행된다. 이미 떠 있는 Isaac은 넘겨받지 않고 거부한다.

`learning-speed/1`의 판정(2026-11-20)은 "ORL이 Untyped GNN·PPO보다 빠름"이다(진행 기록 21.3).

### 실제 운용 환경 시연 영상

`tools/isaac_demo_capture.py`가 README의 시연 GIF·MP4·사진을 만든다(진행 기록 23절).

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
python3 tools/isaac_demo_capture.py capture OUT <checkpoint>/final.pt --allow-isaac   # 비행 + 화면 녹화
python3 tools/isaac_demo_capture.py media OUT ../docs/figures                         # 잘라서 GIF·MP4·PNG
```

- `capture`는 소유한 스택을 GUI로 띄운다. 같은 화면에 ROS 2 `image_view`(ArUco 주석 카메라)와 RViz2(`tools/demo_tf_relay.py`의 TF로 기체 추종)를 배치하고, 화면을 녹화한다.
  - 시드는 시연 전용(기본 999902–999904)이다. 결과는 평가가 아니다.
  - VS Code(snap) 셸의 GTK·로케일 변수는 rviz2를 죽인다. 도구는 이 변수들을 지우고 ROS 도구를 띄운다.
- `media`는 녹화에서 세 창 영역만 잘라낸다. 다른 창이 찍혔을 수 있는 전체 화면 녹화는 올리지 않는다.

## 3. 결과를 믿기 위한 규칙

코드가 강제하는 규칙은 다음과 같다.

- **참값 격리:** 시뮬레이터 참값은 보상과 착륙 판정에만 들어간다. actor, critic, 그래프, 관측 경로에는 들어가지 않는다.
- **같은 조건:** 모든 정책은 보상, 판정, PPO 설정, 커리큘럼, 학습 예산, 학습·평가 시드, 가치함수 설계가 같다.
  다른 것은 actor의 구조와 입력 표현뿐이다. 파라미터 수는 4,008–4,022개로 맞춘다.
- **사전 등록:** 판정 규칙과 시드는 학습 전에 등록한다. 시드는 이미 쓴 모든 범위와 겹치지 않는지 로더가 확인한다.
- **분석 단위:** 분석 단위는 학습 시드이다. 같은 시드의 정책들은 같은 환경을 비행하고, 시드마다 짝지은 차이를 계산한다.
- **결정론적 평가:** 평가는 결정론적 정책으로 한다. 학습 가능성 점검에서는 확률적 정책도 함께 평가한다.
- **실패 처리:** 체크포인트를 고르지 못한 모델도 빼지 않는다. 학습 속도 실험에서 상한까지 도달하지 못한 모델은 상한값으로 센다.
- **인프라 실패와 정책 실패의 분리:** 인프라 실패(스택 시작 실패, PX4 failsafe, 시계 정지)는 판정에 넣지 않고 같은 케이스를 다시 비행한다.

## 4. 테스트

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
python3 -m pytest -q tests                    # 전체, 약 8분
python3 -m pytest -q tests/current_semantics  # ORL과 실험 절차만
```

계약 해시, ORL의 등변성·평면 공유·국소성, 시드 감사, 판정 통계, 단계 거부 규칙, Isaac 월드와 스택 코드를 확인한다.
학습이나 비행은 하지 않는다.

## 5. 보관된 경로

2026-11-20 사용자 결정으로, ORL이 쓰지 않는 경로를 모두 `archive/non_orl_20261120.tar.zst`(git에서 제외)로 옮겼다.

- 최소 관측 ROS 파이프라인, 3축 기준 사다리(spatial-reference), 2축 기준 이식, MATLAB 이식, 이전 Isaac 비교 경로
- 항법 열화 실험(Kinematic WLS = 내부 명칭 WMON, CA-KF)과 그 결과, 확인·재현 실험, extension, bond/1, 관계 섞기 대조군
- 위 경로의 도구, 스크립트, 설정, 테스트, 문서

보관본 안의 `MANIFEST.md`에 목록과 복원 방법이 있다. 코드는 git 기록에서도 복원할 수 있다.
