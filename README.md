# Ontology-R-GAT future-FOV-risk reward for UAV landing

이 저장소는 NVIDIA Isaac Sim, Pegasus, PX4 SITL 환경에서 이동 UGV에 착륙하는
두 개의 recurrent PPO agent를 엄격하게 비교한다.

| Pipeline | 공통 Shin 구성 | 유일한 차이 |
|---|---|---|
| `shin_se_fixed` | 6-keypoint encoder, LSTM, 6-D 상대상태 추정과 auxiliary loss, PPO actor, asymmetric critic, 고정 5성분 보상, active-perception 보상 | 없음 |
| `shin_se_onto_rgat_fov` | 위 구성을 모두 동일하게 유지 | 시각 ontology와 동결 R-GAT의 미래 FOV-loss 확률에 따른 추가 보상 |

제안법은 Shin et al.의 상태추정기나 active-perception 보상을 대체하지 않는다.
고정 가중치도 변경하지 않는다.

$$
r_{base}=r_{task}+\sum_i w_i^0\rho_i+r_{active},\qquad
w^0=[1.0,1.0,0.5,1.0,2.0],
$$

$$
r_{proposed}=r_{base}+r_{onto\_fov},\qquad
r_{onto\_fov}=-\lambda_{fov}p_{fov\_loss},\quad \lambda_{fov}=0.1.
$$

R-GAT 입력은 `KeypointConfidence`, `VisibleKeypointFraction`, `FOVMargin`,
`ApparentTargetScale`, `ImagePlaneMotion`, `ScaleRate`, `VisibilityMemory`,
`ReacquisitionTrend`의 8개 시각 특징뿐이다. Simulator truth, 실제/추정 상대 위치·속도,
critic privileged state는 입력할 수 없다.

## 현재 실행 경로

현재 기본 연구 경로는 `config/experiments/two_pipeline_comparison.yaml`의 두
pipeline 비교다. `shin_se_fixed`와 `shin_se_onto_rgat_fov`는 Shin baseline을
공유하고, 후자만 동결된 visual-only R-GAT의 1초 미래 FOV-loss 보상을 추가한다.
Estimator-free, adaptive-weight, PBRS 실험은 legacy/ablation 경로다.

## 실행

기본 실행은 두 개의 독립 UAV/UGV·PX4·namespace·port·learner·buffer·optimizer를 쓴다.

```bash
./run.sh --pipelines shin_se_fixed shin_se_onto_rgat_fov --parallel-pairs 2
```

인자 없는 `./run.sh`는 `seminar_10h_two_pipeline.yaml`을 자동 선택하는 재현 가능한
세미나 프로파일이다. 두 pair, pipeline당 144 training episodes, 5 evaluation
episodes, 40 FOV-risk data episodes를 사용하고 결과를
`results/seminar_10h/two_pipe_parallel_144/`에 저장한다. 이 프로파일은
`publication_claim_allowed: false`이므로 논문 결과로 보고하지 않는다.

더 큰 명시적 실행은 위 명령처럼 실행한다. 기본 full 경로는 선택한 pipeline 전체에
800 training episodes, 5 evaluation episodes, 40 FOV-risk data episodes를 적용한다.
필요하면 `--total-train-episodes`, `--eval-episodes`, `--rgat-data-episodes`,
`--rgat-epochs`, `--results-dir`로 덮어쓸 수 있다. 직접 runner를 호출할 때는
`Ontology_RGAT_UAV_RL_ISAAC_PX4/`에서 `python python/run_two_pipeline.py --help`로
primary-only 옵션을 확인한다.

예전 estimator-free/adaptive-weight/PBRS 실험은 기본 경로와 결과 집계에서 제외되어 있고,
필요할 때만 `--legacy-multi-pipeline`으로 명시적으로 실행한다.

## 검증

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/check_workspace.sh
```

검사는 baseline 보상 보존, `lambda_fov=0` 동일성, 공통 actor/critic/estimator/PPO 계약,
정보 누출 차단, graph 결정성, 확률·보상 범위, R-GAT 동결, episode-level split,
단독 및 2-pair 실행 계약을 포함한다.

상세 내용은 [구현 README](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md),
[운영 절차](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md),
[두 pipeline 비교 설계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/TWO_PIPELINE_COMPARISON.md)를 참고한다.
