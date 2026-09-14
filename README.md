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

## 실행

기본 실행은 두 개의 독립 UAV/UGV·PX4·namespace·port·learner·buffer·optimizer를 쓴다.

```bash
./run.sh --pipelines shin_se_fixed shin_se_onto_rgat_fov --parallel-pairs 2
```

인자 없는 `./run.sh`도 같은 두 pipeline과 두 pair를 선택하며 세미나용 예산을 적용한다.
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

상세 내용은 [구현 README](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md)와
[두 pipeline 비교 설계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/TWO_PIPELINE_COMPARISON.md)를 참고한다.
