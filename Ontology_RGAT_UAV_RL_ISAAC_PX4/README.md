# 구현 개요

이 디렉터리는 [상위 README](../README.md)가 정의한 3-arm 비교의 실행 코드다.
아래는 알고리즘의 각 구성요소가 어느 모듈에 대응하는지, 그리고 어떤 artifact가
어느 단계에서 확정되는지를 정리한다.

## 1. 세 arm

| ID | 종류 | 구성 |
|---|---|---|
| `pn_guidance_v1` | **비학습 대조군** | 사가탈 평면 비례항법 유도. 학습·보상·checkpoint 없음 |
| `shin_se_fixed` | 학습 | Shin et al. 전체 baseline |
| `shin_se_onto_rgat_state` | 학습 | 동일 baseline + 관측에 온톨로지 상황 그래프 `G_t` |

대조군은 `PipelineSpec`을 갖지 않는다. 따라서 보상 dispatch에 넘길 수 없고, 러너가
**참조 arm의 method로 비행**시킨 뒤 기록만 대조군 이름으로 남긴다. 공정성 측면에서도
옳다 — 대조군이 학습 arm과 같은 encoder로 덱을 보고, 같은 3채널 액션을 낸다.

`pipelines`는 학습 arm 목록이고, 전체 arm 목록은 `manifest.json`의 `arms` 키다.
`learned` 플래그와 `ontology_role`을 싣는다.

두 학습 agent는 카메라, Isaac/Pegasus/PX4 환경, 초기조건, UGV 궤적, 6-keypoint
encoder와 checkpoint, LSTM과 6-D 상대상태 추정기, 보조손실, actor/critic,
observation의 이미지·proprio 부분, action, PPO hyperparameter, curriculum,
**보상 전체**(고정 5개 shaping 항과 가중치, active-perception, 종료 보상),
학습·평가 seed와 episode 예산을 공유한다. 같은 seed에서 **공유 파라미터가 비트
단위로 동일**하다(`tests/test_two_pipeline_fov.py`가 텐서 단위로 확인).

소거 실험 arm은 `shin_se_onto_gat_state`(관계 유형 통합)와
`shin_se_node_pool_state`(메시지 전달 없음)이며, 같은 부호기 클래스와 같은 PPO를
쓴다. 은퇴한 보상항 arm `shin_se_onto_rgat_recovery`는 자기 id로 남아 있고 지금도
실행할 수 있다. 두 방법을 한 실행에 섞는 것은
`validate_pipeline_configuration`이 거부한다 — 요인이 둘이 되기 때문이다.

`PipelineSpec`은 frozen dataclass이고 모델 구성, 목적함수, 보상 dispatch, dataset
준비가 모두 같은 spec을 읽는다. 따라서 추정기 없는 actor에 추정 의존 보상을
붙이거나, 그래프를 상태와 보상 양쪽에 넣는 조합은 구성 시점에 예외가 된다.

## 2. 알고리즘 ↔ 코드

### 2.1 인지와 정책

| 구성요소 | 모듈 | 형태 |
|---|---|---|
| 6-keypoint encoder | `perception/keypoint_encoder.py` | conv(1→32→64→128→128, stride 2) → 6채널 heatmap → soft-argmax → keypoint별 descriptor pooling → `Linear(768, 512)` + tanh |
| Encoder 사전학습 | `perception/keypoint_pretrain.py` | 패드 landmark 투영 label + 실제 Isaac 프레임 fine-tuning, 검증 개선 시에만 채택 후 동결 |
| 시간 backbone | `ppo/temporal_backbone.py` | 이미지 embedding 512 + proprio 7 → LSTM 512 → latent 256 |
| 상대상태 보조 head | `ppo/recurrent.py: RelativeStateAuxiliaryHead` | `latent[0:6]` → `[Δp^b, Δv^b]`, 정규화 scale `[3,3,8,3,3,2]` |
| Actor | `ppo/recurrent.py: PipelineActorCritic.actor` | `concat(latent[6:], proprio 7, g_t)` → 256 → 256 → **3**, tanh squash, `log σ` 학습 |
| Asymmetric critic | 같은 파일 `critic` | `[u_t(7), s^rel(6), g_t]` → 256 → 256 → 1, 학습 전용 |
| PPO | `ppo/recurrent_train.py`, `ppo/gae.py` | γ 0.99, GAE λ 0.95, clip 0.2, lr 5e-5, epochs 3, seq len 32, entropy 0.003, value 0.5, grad clip 5, 보조계수 0.05, target KL 0.03 |

`g_t`는 그래프 상태 arm에만 존재한다. 기준 arm에서 actor 입력은 `latent[6:]`와
proprio뿐이고, `graph_dim`이 0이라 두 arm의 유일한 구조적 차이는 각 head의
**입력층 폭**이다. 그 입력층은 생성 순서상 마지막에 만들어지므로, 나머지 모든
파라미터는 전역 난수열의 같은 지점에서 나온다.

Actor 관측은 grayscale 이미지, 7-D UAV proprioception(body velocity 3 +
quaternion 4), 그리고 제안 arm에서는 `G_t`다. 플랫폼 world pose, 상대 truth,
접촉력, 보상 label, critic value는 actor에 들어가지 않는다.

### 2.2 보상 (두 학습 arm 공통)

| 구성요소 | 모듈 |
|---|---|
| 5개 shaping 항 + active perception + 종료 보상 | `reward_modes/shin2026.py: ShinReward` |
| Pipeline별 보상 dispatch | `ppo/recurrent_train.py: _reward` |

`ShinReward`는 종료 전이에서 shaping 항 전체를 0으로 만들고 `±10`만 남긴다.
**온톨로지는 보상에 전혀 관여하지 않는다.** 다섯 번째 항만 선언된 이탈이다:
원문의 `-2|ω_z|`가 같은 가중치로 종방향 틸트 채널에 옮겨졌다
(`docs/PAPER_FIDELITY.md` §3의 9번).

### 2.3 온톨로지와 R-GAT

| 구성요소 | 모듈 |
|---|---|
| 시각 semantics 추출 | `perception/semantic_observation.py` |
| 9-node 상황 그래프 생성 | `rgat/state_graph.py` |
| R-GAT 부호기 + 그래프 수준 읽기 | `ppo/graph_state_encoder.py` |
| 관계 attention kernel | `rgat/layers.py`, `rgat/topology.py` |

`build_state_graph`의 유일한 생성자 인자는 `SemanticObservation`이며, 그 필드는
keypoint/이미지 이력과 기체 자신의 proprioception에서만 나온다. Simulator truth,
6-D 추정, critic state, 기하 패드중심 FOV 라벨이 들어갈 인자 자체가 존재하지 않고,
`tests/test_ontology_graph_state.py`가 서명과 구문 트리 양쪽으로 확인한다.

부호기는 **PPO가 함께 학습한다**. 오프라인 단계도, 동결 산출물도, checksum
게이트도 없다. 설정으로 동결하거나 사전학습 파일을 가리키려는 시도는
`validate_pipeline_configuration`이 거부한다.

### 2.4 제어

| 구성요소 | 모듈 |
|---|---|
| 평면 3채널 엔벨로프 | `controllers/planar_controller.py` |
| PN 유도 대조군 | `controllers/pn_guidance.py` |
| 속도 법칙 → 평면 액션 변환 | `run_three_pipeline._planar_from_velocity_action` |
| 틸트/yaw 유지 전송 | `bridge.py: step_velocity` |
| 틸트 → 가속도 피드포워드, yaw 절대 유지 | `ros2_ws/.../ros2_gateway.py` |
| 세 덱의 속도 프로파일 | `isaac_sim/pad_motion.py: segmented_cruise_speeds` |
| 평면 진입 자세 | `isaac_sim/landing_world.py` (`planar_entry`) |

교사도, 대조군도, 정책도 같은 적분기와 같은 한계를 통과해 PX4에 도달한다.
상세는 [평면 엔벨로프](docs/PLANAR_ENVELOPE.md).

### 2.5 실행과 평가

| 구성요소 | 모듈 |
|---|---|
| 주 실험 진입점 | `python/run_two_pipeline.py` (→ `run_three_pipeline.main(primary_only=True)`) |
| 실행 계약 spec | `ontology_rgat/pipelines/spec.py` |
| 실제 환경 adapter | `ontology_rgat/benchmarks/live_env.py`, `px4_adapter.py` |
| Isaac stage | `isaac_sim/landing_world.py`, `isaac_sim/keypoint_geometry.py` |
| 지표 집계·통계·표 | `ontology_rgat/evaluation/two_pipeline.py`, `three_pipeline.py` |
| 대시보드 | `ontology_rgat/viz/live.py`, `tools/watch_run.py` |
| RViz | `ontology_rgat/viz/rviz.py`, `rviz/*.rviz` |

주 실행 설정은 `config/experiments/planar_three_arm_comparison.yaml`, 시뮬레이터
프로파일은 `config/shin2026-planar-system.yaml`이다.

## 3. 실행 단계

```
0  stack startup            DDS · Isaac Sim · Pegasus · PX4 SITL
1  keypoint validation      실제 Isaac 카메라 · held-out label · 이후 동결
2  behavior cloning         공통 PN 유도 시연 → arm별 warm start
3  parallel PPO             모든 arm 동시 · 동일 예산 · 동일 seed · 동일 보상
4  checkpoint selection     held-out 결정론 multi-seed
5  crossover evaluation     방법 × 시나리오 × seed × 물리 pair
6  tables / figures
```

**오프라인 보상 설계 단계가 없다.** 은퇴한 보상항 경로를 선택했을 때만
FOV-risk 데이터 수집과 R-GAT 동결 단계가 켜진다.

중단 후 같은 명령을 다시 실행하면 호환되는 checkpoint와 이미 끝난 평가 행을
재사용한다. Episode 지표와 checkpoint를 성공적으로 기록한 뒤에만 committed episode가
증가하므로 부분 trajectory는 PPO에 들어가지 않는다.

## 4. 실행

```bash
# 상위 디렉터리에서
./run.sh                 # 기본: 평면 3-arm 비교
./run.sh --mode quick    # 전 구간 배관 검증 (수십 분, full 전에 권장)
```

각 pair는 PX4 instance, ROS namespace, UDP gateway/learner port, reset/controller
state, PPO buffer, optimizer와 log를 독립적으로 소유한다. GPU update만 lock으로
직렬화한다. 학습 페어는 **학습 arm에만** 나뉘고, 최종 평가는 동일 scenario/seed를
세 arm 전체에 교차 배정한다.

시뮬레이터 없이 수행하는 정적·단위 검증:

```bash
./scripts/check_workspace.sh
python -m pytest -q tests/test_ontology_graph_state.py tests/test_planar_envelope.py
python -m pytest -q tests/test_two_pipeline_fov.py tests/test_three_pipeline.py
```

## 5. Artifact와 경계

| Artifact | 확정 시점 | PPO 중 |
|---|---|---|
| Keypoint encoder | 사전학습/보정 단계 | 동결 |
| Actor / LSTM / critic / **그래프 부호기** | PPO 단계 | 학습 |
| Latest checkpoint | episode commit | 재개용 갱신 |
| Best checkpoint | 안전성 score 개선 시 | pre-update snapshot |
| Selected checkpoint | held-out 검증 후 | 최종 평가·배포용 동결 |

모든 artifact는 config/dataset/encoder hash와 architecture version을 저장하고,
불일치하면 재사용하지 않는다. 시연 집합은 v4 형식으로 전이별 `G_t`를 함께
저장한다 — 사후에 다시 계산하려면 비행이 가졌던 시각 recurrence가 필요한데
기록 파일에는 없다.

## 6. 결과 디렉터리

```
manifest.json                       resolved config · arms(learned·ontology_role) · spec · seed/budget · provenance
models/<pipeline>/                  독립 PPO latest/best/selected checkpoint
evaluation/per_episode.csv          paired · crossover 원자료
evaluation/paired_summary.csv       시나리오별 계층 bootstrap 요약
tables/primary_comparison.*         3-arm 주 비교 (대조군 포함)
```

## 7. Legacy

estimator-free, adaptive-weight, semantic PBRS 실험, 은퇴한 이진 분류 readout
(`shin_se_onto_rgat_fov`), 그리고 FOV-risk 보상항 방법
(`shin_se_onto_rgat_recovery`)은 기존 결과의 귀속을 위해 원래 ID로 보존된다.
기본 설정, 기본 runner, 대시보드 기본 화면, 주 결과 집계에는 나타나지 않는다.
대시보드 패널은 `ontology_role`을 선언하고 그 역할이 없는 실행에서는 숨는다 —
그래프 상태 실행에서 FOV-보상 패널이 빈 채로 그려지면 측정이 고장난 것처럼
읽히기 때문이다.
