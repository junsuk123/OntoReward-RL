# 두 pipeline 구현

활성 연구 경로는 다음 두 agent뿐이다.

- `shin_se_fixed`: Shin et al. 기반 전체 baseline.
- `shin_se_onto_rgat_fov`: 동일 baseline + visual ontology + frozen R-GAT 미래 FOV-loss 예측 + `-lambda_fov * p_fov_loss`.

두 agent는 카메라, Isaac/Pegasus/PX4 환경, 초기조건, UGV 궤적, 6-keypoint encoder와
checkpoint, LSTM과 6-D 상대상태 추정기, auxiliary loss, actor/critic, observation/action,
PPO hyperparameter, curriculum, 고정 5개 shaping 항과 가중치, active-perception/terminal
보상, 학습·평가 seed와 episode 예산을 공유한다.

## 실행

저장소 상위 디렉터리에서 실행한다.

```bash
./run.sh --pipelines shin_se_fixed shin_se_onto_rgat_fov --parallel-pairs 2
```

각 pair는 PX4 instance, ROS namespace, UDP gateway/learner port, reset/controller state,
PPO buffer, optimizer와 log를 독립적으로 소유한다. 최종 평가는 동일 scenario/seed를
사용하고 두 물리 pair를 교차 배정한다.

## 주요 코드

| 경로 | 역할 |
|---|---|
| `python/run_two_pipeline.py` | 기본 두-agent 실험 진입점 |
| `python/ontology_rgat/pipelines/spec.py` | 실행 시 baseline 동일성을 검사하는 immutable spec |
| `python/ontology_rgat/perception/semantic_observation.py` | keypoint와 시각 이력 추출 |
| `python/ontology_rgat/rgat/fov_graph.py` | 엄격한 8-feature, 13-node ontology |
| `python/ontology_rgat/rgat/fov_risk_dataset.py` | 1초 horizon 미래 FOV-loss label과 episode split |
| `python/ontology_rgat/rgat/fov_risk_model.py` | sigmoid R-GAT classifier와 frozen loader |
| `python/ontology_rgat/rgat/fov_risk_train.py` | BCE-only offline 학습과 validation-best 저장 |
| `python/ontology_rgat/reward_modes/fov_risk.py` | 비양수 가산 보상 |
| `python/ontology_rgat/ppo/recurrent_train.py` | 공통 PPO rollout과 additive reward dispatch |
| `python/ontology_rgat/evaluation/two_pipeline.py` | 주 비교 metric 집계 |

## 데이터와 artifact

`rgat/fov_risk_rollouts.npz`는 동일 simulator domain trajectory에서 생성한다.
한 episode의 sample은 train과 validation에 나뉘지 않는다. Label은 현재 시점 이후
설정 horizon 안에 target이 unavailable/FOV criterion 미달이 되면 1이다.

`rgat/fov_risk_model.pt`에는 dataset version/hash, seed, graph version, horizon,
train/validation episode ID, AUROC, F1, precision, recall, confusion matrix와 model checksum이
저장된다. R-GAT은 PPO optimizer에 포함되지 않으며 매 episode 뒤 동결 상태와 checksum을
검사한다.

## 주요 평가 지표

- Landing Success Rate
- FOV Loss Episode Rate와 FOV Retention Ratio
- Mean/Maximum Continuous FOV Loss Duration
- R-GAT AUROC, F1, Precision, Recall, Confusion Matrix
- 공통 진단값: 위치/속도 추정 오차와 loss, return, landing error, touchdown vertical velocity

## 검증

```bash
./scripts/check_workspace.sh
```

이전 adaptive-weight, semantic PBRS와 estimator-free 코드는 재현용 legacy/ablation으로
보존되지만 기본 설정, 기본 runner, dashboard와 주 결과 집계에는 나타나지 않는다.
