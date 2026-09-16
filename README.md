# Ontology-R-GAT Future-FOV-Risk Reward for Vision-Based UAV Landing on a Moving Platform

이 저장소는 **시야(FOV) 상실을 사후에 벌하는 대신 사전에 예측해 벌하는 보상항**을
제안하고, 그 항 하나만을 실험 요인으로 갖는 통제 비교를 Isaac Sim / Pegasus /
PX4 SITL 위에서 수행한다.

## 1. 문제 정의

이동하는 지상 플랫폼(UGV)에 단안 하향 카메라만으로 착륙하는 UAV는 접근 후반부에
구조적인 딜레마를 만난다. 패드에 가까워질수록 표적의 상대 각속도가 커져 패드가
프레임을 벗어나기 쉽고, 한 번 벗어나면 관측이 끊긴 상태에서 재획득해야 한다.
기존 보상은 *이미 벌어진* 추정오차나 *현재* 가시성만 본다. 즉 패드가 화면 중앙에
있는 순간에도 1초 뒤 프레임 밖으로 나갈 궤적이라면 그 궤적은 벌점을 받지 않는다.

그러므로 이 연구의 질문은 다음 하나다.

> 완전한 Shin et al. baseline을 어떤 항도 바꾸지 않은 채, **가까운 미래의 기하
> FOV 비가용성을 예측하는 가산 보상항 하나**를 더하면 표적 가시성 유지와 이동
> 플랫폼 착륙 성능이 향상되는가?

## 2. 기여

1. **미래 FOV 비가용 시간 비율**이라는 회귀 표적의 정의. 이진 "H 안에 놓치는가"가
   아니라 향후 H step 중 패드 중심이 절두체 밖인 *시간 비율* `y_t ∈ [0,1]`이다.
2. **시각 전용 온톨로지 graph**. 10개의 유계 시각 특징과 4개의 파생 개념 node,
   1개의 출력 node로 이루어진 결정적 15-node 관계 graph. 시뮬레이터 truth,
   6-D 상대상태 추정, critic state는 구조적으로 입력될 수 없다.
3. **분리된 readout이 없는 R-GAT**. 두 번째 관계 layer가 `units=1`이고 그 layer의
   `FutureFOVUnavailability` node 자체가 출력이다. 별도 MLP head나 linear head가
   없으므로 "graph가 보상 신호를 만든다"는 주장이 구조로 강제된다.
4. **결측과 여유의 분리**. `fov_margin`은 *마지막으로 신뢰 가능했던* centroid의
   여유이므로 측정이 끊긴 직후 가장 위험한 순간에 1.0에 가깝게 읽힌다. 유효성과
   경과시간 node를 함께 공개하고 경계 안전 seed를 유효성으로 gate한다.
5. **단일 요인 통제 비교**. 두 agent는 환경, 카메라, encoder, LSTM, 추정기,
   보조손실, actor/critic, PPO 하이퍼파라미터, curriculum, 5개 고정 shaping 항과
   가중치, active-perception 보상, seed와 episode 예산까지 모두 공유한다.

## 3. 방법

### 3.1 Baseline 보상 (변경 없음)

상대상태를 `s^rel = [Δp^b, Δv^b] ∈ R^6` (플랫폼 − UAV, UAV body frame), 행동을
`a_t = [v_x, v_y, v_z, ω_z]`라 하면 baseline 보상은 구간별로 정의된다.

$$
r_{\text{Shin}}(t)=
\begin{cases}
+10 & \text{엄격 안전착륙}\\
-10 & \text{충돌 · 과도이탈 · 배터리 고갈}\\
\sum_{i=1}^{5} w_i^0\rho_i(t)+r_{\text{active}}(t) & \text{그 외}
\end{cases}
$$

$$
w^0=[1.0,\;1.0,\;0.5,\;1.0,\;2.0],\qquad
r_{\text{active}}(t)=-0.1\,\mathrm{clip}\!\left(L_{\text{est}}(t{+}1)-0.01,\;0,\;1\right).
$$

종료 보상은 shaping을 **대체**한다. 다섯 항의 정확한 식은
[Shin baseline 대응](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md)에 있다.

### 3.2 제안 가산항

$$
r_{\text{proposed}}(t)=r_{\text{Shin}}(t)-\lambda_{\text{fov}}\;q_\theta\!\left(G_t\right),
\qquad \lambda_{\text{fov}}=0.1 .
$$

`G_t`는 시각 이력만으로 만든 온톨로지 graph, `q_θ`는 PPO 전에 학습하고 동결한
R-GAT의 스칼라 출력이다. 가산항은 비양수이고 `λ_fov = 0`이면 두 보상은 같은
trajectory에서 수치적으로 동일하다. 이것은 일반적인 shaped reward이며
potential-based shaping이 아니므로 최적정책 불변 주장은 따라오지 않는다.

### 3.3 예측 표적

제어 주파수를 `f`, horizon을 1.0초라 하면 `H = round(1.0 f)`이다(10 Hz → 10 step).

$$
y_t=\frac{1}{H}\sum_{k=1}^{H}\bigl(1-V_{\text{centre}}(t+k)\bigr),
\qquad
q_\theta(G_t)\;\approx\;\mathbb{E}\!\left[y_t \mid G_t\right].
$$

`V_centre`는 `isaac_sim/keypoint_geometry.geometric_pad_center_in_fov`, 즉 패드
중심이 양의 depth로 정규화 image 좌표 `[-1,1]^2`에 투영되는가 하나뿐이다.
검출 성공률도, 학습된 keypoint 신뢰도도 아니다. `H` step 미래가 관측되지 않은
episode 꼬리는 `valid=False`, target `NaN`으로 mask한다.

### 3.4 온톨로지 graph와 readout

| 항목 | 값 |
|---|---|
| 입력 특징 | 10개 (`KeypointConfidence` … `MeasurementAge`), 모두 `[0,1]` |
| Node | 15개 = 입력 10 + 파생 4 + 출력 1 |
| 관계 | `indicates`, `supports`, `constrains`, `self` |
| Edge | 선언 16개 + node별 self-loop 15개 |
| Node 특징 | `X ∈ R^{19×15}` = 값, 여값, 입력/파생 지시자, 15-D one-hot |
| 모델 | R-GAT(19→24) → tanh → R-GAT(24→1) → sigmoid |
| 출력 | 두 번째 layer의 `FutureFOVUnavailability` node |

계약 원본은 `config/ontology/fov_recovery.yaml`이며 규칙 `R-01`~`R-06` 중
`enforced`로 표시된 것만 손실이나 test로 실제 강제된다.

### 3.5 오프라인 학습과 동결

$$
\mathcal{L}=\underbrace{\mathrm{Huber}_{\delta=0.1}\!\left(q_\theta,\,y\right)}_{\text{회귀}}
+0.1\cdot\underbrace{\Bigl[q_\theta(G)-q_\theta(G^{+\Delta\text{age}})\Bigr]_+}_{\text{규칙 }R\text{-}04}
$$

Dataset은 episode ID 기준 80/20으로 나누어 시간 누출을 막고, validation loss
최소 checkpoint를 저장한다. PPO 동안에는 모든 parameter가 `requires_grad=False`,
eval mode이며 state-dict SHA-256이 매 episode 뒤 검증된다.

## 4. 비교하는 두 pipeline

| Pipeline | 구성 | 차이 |
|---|---|---|
| `shin_se_fixed` | 6-keypoint encoder, LSTM, 6-D 상대상태 추정과 보조손실, PPO actor, asymmetric critic, 고정 5성분 보상, active-perception 보상 | 없음 (baseline) |
| `shin_se_onto_rgat_recovery` | 위 구성을 전부 동일하게 유지 | `-λ_fov q_θ(G_t)` 한 항 추가 |

`assert_primary_baseline_equivalence()`와 YAML/spec 교차검증이 실행 시점에
baseline 계약 flag의 차이를 즉시 실패로 만든다. 두 model은 같은 seed에서 동일한
state-dict key, shape, 초기값을 갖는다.

## 5. 시스템 파이프라인

```
Isaac Sim / Pegasus  →  카메라 · 접촉 · UGV 주행 · 배터리 · 외란
        │
        ├── PX4 SITL (OFFBOARD)  ←  [v_x, v_y, v_z, ω_z] 속도 명령
        │
        └── ROS 2 gateway  →  learner
                              ├── 동결 6-keypoint encoder
                              ├── LSTM(512) → latent(256)
                              │     ├── [0:6]  → 6-D 상대상태 보조 회귀
                              │     └── [6:]   → actor MLP → 행동
                              ├── asymmetric critic([u_t, s^rel] 13-D, 학습 전용)
                              └── (제안 arm) 시각 10특징 → 15-node graph
                                                        → 동결 R-GAT → q_θ
```

RL actor는 모터나 추력을 직접 명령하지 않는다. 한 transition은 0.1초이고 episode
horizon은 300 step이다. 실행 단계는 다음 순서로 고정되어 있다.

1. 공통 keypoint encoder 사전학습 · 실기 카메라 검증 후 동결
2. baseline PPO 학습과 FOV-risk dataset 수집
3. R-GAT 회귀 학습 → validation-best 선택 → 동결
4. proposed PPO 학습 (동일 예산 · 동일 seed)
5. held-out 결정론 seed로 checkpoint 선택
6. paired · crossover 평가
7. 표와 그림 저장

## 6. 실험 설계

* **통제**: 환경, 초기조건 분포, UGV 궤적, seed, episode 예산, PPO 설정, curriculum.
* **요인**: FOV-risk 가산항의 유무 하나.
* **병렬 실행**: 한 Isaac stage 안에서 2개의 독립 UAV/UGV·PX4·namespace·port·
  learner·buffer·optimizer 쌍을 사용한다. 방법과 물리 쌍의 대응은
  `training_replicate`마다 cyclic Latin square로 바꾸어 경로 위상 편향을 분리한다.
* **평가 시나리오**: `training_random_walk`, `straight_8mps`,
  `linear_acceleration_wave`, `circle`, `zigzag`, `u_turn`, `vertical_heave_boat`.
* **통계**: 동일 replicate·시나리오·seed 짝의 차이를 모으고, replicate → episode
  순의 계층 bootstrap으로 95% 신뢰구간을 낸다.

### 주 지표

| 분류 | 지표 |
|---|---|
| 착륙 | 엄격 Landing Success Rate, touchdown lateral error, 수직/상대 수평 속도, tilt, 각속도 |
| 기하 FOV | Loss Episode Rate, Retention Ratio, 평균/최대 연속 손실 시간, 재획득율·재획득 시간 |
| 인지 품질 | `keypoint_confidence_mean`, `visible_keypoint_fraction_mean`, `low_keypoint_visibility_fraction` |
| R-GAT | MAE, RMSE, bias, R², **상수 예측기 RMSE 기준선**, 규약 위반량 |
| 공통 진단 | 위치/속도 추정 RMSE와 loss, return, touchdown 시간 |

엄격 성공 판정은 접촉 ∧ 수평오차 ≤ 0.35 m ∧ |v_z| ≤ 0.55 m/s ∧ 상대 수평속도
≤ 0.45 m/s ∧ tilt ≤ 10° ∧ 각속도 ≤ 45°/s다. 논문 기준의 접촉 성공률은
`paper_success`로 병기하되 엄격 기준을 대체하지 않는다.

**기하 FOV와 인지 품질은 서로 다른 변수다.** 패드가 프레임 안에 있는데 encoder가
놓치면 인지 저하이지 FOV 손실이 아니다. 지표도 `geometric_fov_*` 계열과
`*keypoint*` 계열로 분리해 보고한다.

## 7. 실행

```bash
./run.sh --pipelines shin_se_fixed shin_se_onto_rgat_recovery --parallel-pairs 2
```

인자 없는 `./run.sh`는 `seminar_10h_two_pipeline.yaml` 세미나 프로파일을 선택한다
(2 pair, pipeline당 144 training episodes, 5 evaluation episodes, 40 FOV-risk data
episodes). 이 프로파일은 `publication_claim_allowed: false`이므로 논문 결과로
보고하지 않는다. 명시적 full 경로는 선택한 pipeline 전체에 800 training episodes를
적용하며 `--total-train-episodes`, `--eval-episodes`, `--rgat-data-episodes`,
`--rgat-epochs`, `--results-dir`로 덮어쓸 수 있다.

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/check_workspace.sh      # 시뮬레이터 없이 계약 전체 검증
```

검증은 baseline 보상 보존, `λ_fov=0` 동일성, 공통 actor/critic/estimator/PPO 계약,
정보 누출 차단, graph 결정성과 도달성, 확률·보상 범위, R-GAT 동결, episode 단위
split, 단독 및 2-pair 실행 계약을 포함한다.

## 8. 주장하지 않는 것

* PBRS 최적정책 불변성 — 제안항은 potential-based shaping이 아니다.
* 운용 안전 보장 — 성공 gate는 연구 평가 기준이지 안전 monitor가 아니다.
* Attention 가중치의 인과 해석, 관계 이름의 단조성.
* 온톨로지 *단독* 기여 — 현재 설계는 제안 보상 모듈 전체의 효과만 비교한다.
* PACMAN 재현 — encoder와 표적은 같은 인터페이스를 갖는 문서화된 근사다.

## 9. 문서

| 문서 | 내용 |
|---|---|
| [구현 개요](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md) | 알고리즘과 코드의 대응, artifact |
| [아키텍처와 정보경계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ARCHITECTURE.md) | 관측·보상·critic 경계, 좌표계, 타이밍 |
| [제안 알고리즘](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_FOV_RISK.md) | graph, 표적, 목적함수, readout, 동결 |
| [실험 설계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/TWO_PIPELINE_COMPARISON.md) | 통제변수, 실험 요인, 지표, 통계 |
| [Shin baseline](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md) | 원문 구성과 구현 대응 |
| [논문 대조](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/PAPER_FIDELITY.md) | 확인된 값, 미기재 항목, 선언된 이탈 |
| [운영 절차](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md) | 실행 프로파일, 산출물, 상태 확인 |
