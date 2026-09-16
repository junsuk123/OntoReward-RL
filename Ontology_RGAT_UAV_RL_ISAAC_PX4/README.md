# 구현 개요

이 디렉터리는 [상위 README](../README.md)가 정의한 두 pipeline 비교의 실행 코드다.
아래는 알고리즘의 각 구성요소가 어느 모듈에 대응하는지, 그리고 어떤 artifact가
어느 단계에서 확정되는지를 정리한다.

## 1. 두 agent

| ID | 구성 |
|---|---|
| `shin_se_fixed` | Shin et al. 전체 baseline |
| `shin_se_onto_rgat_recovery` | 동일 baseline + 시각 온톨로지 + 동결 R-GAT 미래 FOV 비가용성 + `-λ_fov q_θ(G_t)` |

두 agent는 카메라, Isaac/Pegasus/PX4 환경, 초기조건, UGV 궤적, 6-keypoint encoder와
checkpoint, LSTM과 6-D 상대상태 추정기, 보조손실, actor/critic, observation/action,
PPO hyperparameter, curriculum, 고정 5개 shaping 항과 가중치, active-perception과
terminal 보상, 학습·평가 seed와 episode 예산을 공유한다.

`PipelineSpec`은 frozen dataclass이고 모델 구성, 목적함수, 보상 dispatch, dataset
준비가 모두 같은 spec을 읽는다. 따라서 추정기 없는 actor에 추정 의존 보상을
붙이는 조합은 구성 시점에 예외가 된다.

## 2. 알고리즘 ↔ 코드

### 2.1 인지와 정책

| 구성요소 | 모듈 | 형태 |
|---|---|---|
| 6-keypoint encoder | `perception/keypoint_encoder.py` | conv(1→32→64→128→128, stride 2) → 6채널 heatmap → soft-argmax → keypoint별 descriptor pooling → `Linear(768, 512)` + tanh |
| Encoder 사전학습 | `perception/keypoint_pretrain.py` | 패드 landmark 투영 label + 실제 Isaac 프레임 fine-tuning, 검증 개선 시에만 채택 후 동결 |
| 시간 backbone | `ppo/temporal_backbone.py` | 이미지 embedding 512 + proprio 7 → LSTM 512 → latent 256 |
| 상대상태 보조 head | `ppo/recurrent.py: RelativeStateAuxiliaryHead` | `latent[0:6]` → `[Δp^b, Δv^b]`, 정규화 scale `[3,3,8,3,3,2]` |
| Actor | `ppo/recurrent.py: PipelineActorCritic.actor` | `concat(latent[6:], proprio 7)` → 256 → 256 → 4, tanh squash, `log σ` 학습 |
| Asymmetric critic | 같은 파일 `critic` | `[u_t(7), s^rel(6)]` 13-D → 256 → 256 → 1, 학습 전용 |
| PPO | `ppo/recurrent_train.py`, `ppo/gae.py` | γ 0.99, GAE λ 0.95, clip 0.2, lr 5e-5, epochs 3, seq len 32, entropy 0.003, value 0.5, grad clip 5, 보조계수 0.05, target KL 0.03 |

Actor 관측은 grayscale 이미지와 7-D UAV proprioception(body velocity 3 + quaternion 4)
뿐이다. 플랫폼 world pose, 상대 truth, 접촉력, 보상 label, critic value는 actor에
들어가지 않는다.

### 2.2 보상

| 구성요소 | 모듈 |
|---|---|
| 5개 shaping 항 + active perception + 종료 보상 | `reward_modes/shin2026.py: ShinReward` |
| 제안 가산항 `-λ q` | `reward_modes/fov_risk.py: ontology_fov_reward` |
| Pipeline별 보상 dispatch | `ppo/recurrent_train.py: _reward` |

`ShinReward`는 종료 전이에서 shaping 항 전체를 0으로 만들고 `±10`만 남긴다.
제안 arm의 가산항은 dispatch 뒤에 더해지므로 모든 transition에 적용된다.

### 2.3 온톨로지와 R-GAT

| 구성요소 | 모듈 |
|---|---|
| 시각 semantics 추출 | `perception/semantic_observation.py` |
| 10특징 투영과 15-node graph 생성 | `rgat/fov_graph.py` |
| 미래 FOV 비가용 label과 episode split | `rgat/fov_risk_dataset.py` |
| R-GAT 모델과 동결 loader | `rgat/fov_risk_model.py` |
| Huber + 규약 `R-04` 오프라인 학습 | `rgat/fov_risk_train.py` |
| 관계 attention kernel | `rgat/layers.py`, `rgat/topology.py` |

`fov_graph.py`의 유일한 생성자 인자는 `FOVSemanticObservation`이며, 그 필드는
keypoint/이미지 이력에서만 나온다. Simulator truth, 6-D 추정, critic state가 들어갈
인자 자체가 존재하지 않는다. `semantic_observation.py`의 payload validator는
중첩 mapping 안에서도 `truth`, `relative_position`, `relative_velocity`,
`platform_pose`, `critic`, `privileged`, `in_fov` 계열 이름을 거부한다.

### 2.4 실행과 평가

| 구성요소 | 모듈 |
|---|---|
| 주 실험 진입점 | `python/run_two_pipeline.py` (→ `run_three_pipeline.main(primary_only=True)`) |
| 실행 계약 spec | `ontology_rgat/pipelines/spec.py` |
| 실제 환경 adapter | `ontology_rgat/benchmarks/live_env.py`, `px4_adapter.py` |
| Isaac stage | `isaac_sim/landing_world.py`, `isaac_sim/keypoint_geometry.py` |
| 지표 집계·통계·표 | `ontology_rgat/evaluation/two_pipeline.py`, `three_pipeline.py` |
| 대시보드 | `ontology_rgat/viz/live.py`, `tools/watch_run.py` |

주 실행 설정은 `config/experiments/two_pipeline_comparison.yaml`, 세미나 단축
설정은 `config/experiments/seminar_10h_two_pipeline.yaml`이다. 두 설정 모두
`fov_risk.lambda_fov=0.1`, `prediction_horizon_seconds=1.0`,
`visibility_criterion=geometric_pad_center_in_fov`, `freeze_during_ppo=true`를
선언하거나 상속한다.

## 3. 실행 단계

```
0  stack startup            DDS · Isaac Sim · Pegasus · PX4 SITL
1  keypoint validation      실제 Isaac 카메라 · held-out label · 이후 동결
2  (선택) behavior cloning   공통 teacher warm start
3  baseline PPO             shin_se_fixed
4  FOV-risk data            시각 전용 graph + 기하 가시성 timeline
5  R-GAT training           Huber + R-04 → validation-best → 동결
6  proposed PPO             shin_se_onto_rgat_recovery (동일 예산·seed)
7  checkpoint selection     held-out 결정론 multi-seed
8  crossover evaluation     방법 × 시나리오 × seed × 물리 pair
9  tables / figures
```

중단 후 같은 명령을 다시 실행하면 호환되는 checkpoint와 이미 끝난 평가 행을
재사용한다. Episode 지표와 checkpoint를 성공적으로 기록한 뒤에만 committed episode가
증가하므로 부분 trajectory는 PPO에 들어가지 않는다.

## 4. 실행

```bash
# 상위 디렉터리에서
./run.sh --pipelines shin_se_fixed shin_se_onto_rgat_recovery --parallel-pairs 2
```

각 pair는 PX4 instance, ROS namespace, UDP gateway/learner port, reset/controller
state, PPO buffer, optimizer와 log를 독립적으로 소유한다. GPU update만 lock으로
직렬화한다. 최종 평가는 동일 scenario/seed를 쓰고 두 물리 pair에 교차 배정한다.

시뮬레이터 없이 수행하는 정적·단위 검증:

```bash
./scripts/check_workspace.sh
python -m pytest -q tests/test_two_pipeline_fov.py
```

## 5. Artifact와 경계

| Artifact | 확정 시점 | PPO 중 |
|---|---|---|
| Keypoint encoder | 사전학습/보정 단계 | 동결 |
| R-GAT 보상 모델 | 보상 설계 단계 | 동결 (checksum 검증) |
| Actor / LSTM / critic | PPO 단계 | 학습 |
| Latest checkpoint | episode commit | 재개용 갱신 |
| Best checkpoint | 안전성 score 개선 시 | pre-update snapshot |
| Selected checkpoint | held-out 검증 후 | 최종 평가·배포용 동결 |

모든 artifact는 config/dataset/encoder hash와 architecture version을 저장하고,
불일치하면 재사용하지 않는다.

`rgat/fov_risk_rollouts.npz`는 동일 simulator domain trajectory에서 만든다.
한 episode의 sample은 train과 validation에 나뉘지 않는다.
`rgat/fov_risk_model.pt`에는 dataset version/hash, seed, graph version, horizon,
train/validation episode ID, MAE, RMSE, bias, R², 상수 예측기 RMSE 기준선,
규약 위반량, model checksum이 저장된다.

## 6. 결과 디렉터리

```
manifest.json                       resolved config · 두 spec · seed/budget · provenance
rgat/fov_risk_rollouts.npz          episode ID를 포함한 미래 FOV 비가용 dataset
rgat/fov_risk_model.pt              동결 validation-best R-GAT과 checksum
rgat/fov_risk_training_history.csv  epoch별 train/validation huber·contract
models/<pipeline>/                  독립 PPO latest/best/selected checkpoint
evaluation/per_episode.csv          paired · crossover 원자료
evaluation/paired_summary.csv       시나리오별 계층 bootstrap 요약
tables/primary_comparison.*         두 pipeline 주 비교
```

## 7. Legacy

estimator-free, adaptive-weight, semantic PBRS 실험과 은퇴한 이진 분류 readout
(`shin_se_onto_rgat_fov`)은 기존 결과의 귀속을 위해 원래 ID로 보존된다. 기본 설정,
기본 runner, 대시보드, 주 결과 집계에는 나타나지 않으며
`validate_pipeline_configuration`이 은퇴한 readout의 실행을 거부한다.
필요할 때만 `--legacy-multi-pipeline`으로 명시적으로 실행한다.
