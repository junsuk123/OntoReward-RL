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
5. **단일 요인 통제 비교와 비학습 기준선**. 두 학습 agent는 환경, 카메라, encoder, LSTM, 추정기,
   보조손실, actor/critic, PPO 하이퍼파라미터, curriculum, 5개 고정 shaping 항과
   가중치, active-perception 보상, seed와 episode 예산까지 모두 공유한다. 여기에
   학습하지 않는 visual servo 대조군이 같은 덱·seed·착륙 기준으로 함께 평가되어,
   "제안 항이 baseline을 이기는가"와 "학습이 제어기를 이기는가"를 분리한다.

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

## 4. 비교하는 세 arm

| arm | 종류 | 구성 | 차이 |
|---|---|---|---|
| `image_based_visual_servo_v1` | **비학습 대조군** | 영상 기반 visual servo | 학습·보상·체크포인트 없음 |
| `shin_se_fixed` | 학습 | 6-keypoint encoder, LSTM, 6-D 상대상태 추정과 보조손실, PPO actor, asymmetric critic, 고정 5성분 보상, active-perception 보상 | 없음 (baseline) |
| `shin_se_onto_rgat_recovery` | 학습 | 위 구성을 전부 동일하게 유지 | `-λ_fov q_θ(G_t)` 한 항 추가 |

두 학습 arm의 단일 요인 비교는 "온톨로지 항이 PPO 정책을 개선하는가"에 답한다.
**대조군은 "이 중 무엇이든 직접 짤 수 있는 제어기보다 나은가"에 답한다** — 학습형
착륙 제어기에 대해 심사자가 먼저 묻는 질문이고, 같은 덱·같은 seed·같은 착륙 기준
으로만 답할 수 있다. 대조군은 학습 arm과 동일한 encoder로 덱을 보므로 차이는 관측이
아니라 제어에서 온다.

`assert_primary_baseline_equivalence()`와 YAML/spec 교차검증이 실행 시점에
baseline 계약 flag의 차이를 즉시 실패로 만든다. 두 학습 model은 같은 seed에서
동일한 state-dict key, shape, 초기값을 갖는다.

> `pipelines`는 **학습 arm 목록**이다(학습이 순회하고 checkpoint가 키로 삼는다).
> 전체 arm 목록은 `manifest.json`의 `arms` 키이며 `learned` 플래그를 싣는다.
> 하위 소비자는 `pipelines`에서 arm 집합을 추론하면 안 된다 —
> [3-arm 비교](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_ARM_BURST_COMPARISON.md) §3.1, §6.

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
horizon은 300 step이다. 실행은 **데이터 수집 단계와 학습 단계로 분리**되어 있고,
그 순서는 고정이다.

**1단계 — 데이터 수집** (`--stage collect`). 비행해서 재사용 가능한 데이터셋을
만드는 일만 한다. PPO는 한 에피소드도 돌지 않으므로 **모든 물리 페어가 수집에
쓰인다.**

기본 덱은 `straight_escape_burst_track` 하나다. 패드가 **닫힌 오벌 트랙**(20 m 직선 두 개 +
6 m 반경 등속 반원 두 개)을 0.500 m/s로 돌고, 드론이 추종에 들어선 순간 1.000 m/s로
급가속해 **카메라 프레임을 벗어난다.** 시험 대상 사건이 모든 에피소드에 들어 있어야 모든
에피소드가 증거가 되고, readout이 학습할 양성 표적도 생긴다.

트랙을 닫은 이유는 공간이다. 열린 직선 주행은 에피소드 안에서만 직선이고, 에피소드 열은
원점에서 멀어지는 랜덤워크가 되어 heading 끌어당김과 경계 clamp로 교정해야 했다. 닫힌
루프는 그 교정이 필요 없다 — 300 에피소드 동안 32×15 m 안에 머문다. 급가속은 경로를
벗어나는 게 아니라 **트랙 위에서 속도만 두 배**가 되므로, 직선 구간의 이탈 사건은 그대로
일어나면서 덱은 절대 표류하지 않는다.

프로파일이 덱을 선언하지 않으면 모든 수집(교사 시연·FOV-risk·semantic·adaptive)과 PPO가
이 덱을 쓴다. 나머지 덱은 삭제되지 않았고 `two_pipeline_comparison.yaml`이 여섯 개를,
열린 직선 주행(`straight_escape_burst`)도 그대로 남아 있다.

1. 공통 keypoint encoder 사전학습 · 실기 카메라 측량 검증 후 동결
2. 공통 teacher demonstration 비행 → 행동 복제(BC) 워밍업
3. FOV-risk rollout 수집 (동결 BC 정책 또는 디스크에 있는 호환 checkpoint,
   모든 페어 동시) → `results/.../rgat/` + 누적 datastore

**2단계 — 학습** (`--stage train`). 수집을 위해서는 비행하지 않는다. 필요한
데이터셋이 없으면 몇 시간짜리 비행을 조용히 시작하는 대신 `--stage collect`를
지목하는 오류로 멈춘다.

4. R-GAT 회귀 학습 → validation-best 선택 → 동결
5. **모든 arm의 PPO를 동시에 시작** (동일 예산 · 동일 seed · 완성된 같은 보상 설계)
6. held-out 결정론 seed로 checkpoint 선택
7. paired · crossover 평가
8. 표와 그림 저장

예전에는 이 둘이 뒤섞여 있었다. 고정 보상 arm의 PPO를 먼저 띄우고 남는 페어에서
FOV 데이터를 모았기 때문에, 4-페어 무대에서도 수집은 사실상 1페어였고, 페어 배정
산술이 맞지 않으면 수집기와 PPO replica가 같은 페어(같은 로컬 UDP 포트)를 잡았다.
무엇보다 제안 arm의 동결 readout이 비교 arm이 이미 수백 에피소드를 학습한 뒤에야
만들어졌다. 지금은 두 arm 모두 완성된 같은 보상 설계를 상대로 1 에피소드부터
출발한다.

## 6. 실험 설계

* **통제**: 환경, 초기조건 분포, UGV 궤적, seed, episode 예산, PPO 설정, curriculum.
* **요인**: 두 학습 arm 사이에서는 FOV-risk 가산항의 유무 하나. 대조군은 요인이
  아니라 기준선이다.
* **평가 시나리오**: `straight_escape_burst` 하나. 패드가 0.500 m/s로 등속 직진하다
  드론이 추종에 들어선 순간 1.000 m/s로 급가속해 **카메라 정후방으로 프레임을
  벗어난다.** 모든 에피소드가 시험 대상 사건을 포함하므로 모든 에피소드가 증거가
  된다. 나머지 여섯 덱은 `evaluation`에서 명시적으로 0이며 삭제되지 않았다.
* **병렬 실행**: 한 Isaac stage 안에서 독립 UAV/UGV·PX4·namespace·port·learner·
  buffer·optimizer 쌍을 사용한다(기계가 재는 만큼, 현재 4). 학습은 **학습 arm에만**
  페어를 나눈다(4 ÷ 2 = 각 2 replica). 대조군은 학습 중 페어를 갖지 않고 평가에서만
  참여한다. 방법과 물리 쌍의 대응은 `training_replicate`마다 cyclic Latin square로
  바꾸어 경로 위상 편향을 분리한다.
* **통계**: 동일 replicate·시나리오·seed 짝의 차이를 모으고, replicate → episode
  순의 계층 bootstrap으로 95% 신뢰구간을 낸다. 세 arm이 같은 seed를 날므로 seed
  자체의 난이도가 상쇄된다.

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
./run.sh                 # 기본: 3-arm 급가속 이탈 비교, full 예산 (수집 → 학습)
./run.sh --mode quick    # 전 구간 배관 검증 (수십 분)
```

**두 단계를 따로 돌릴 수 있다.** 인자 없는 `./run.sh`는 한 프로세스에서 수집을
끝낸 뒤 학습으로 넘어간다. 나누고 싶으면 `--stage`를 준다.

```bash
./run.sh --stage collect   # 모든 페어에서 데이터셋만 모으고 종료
./run.sh --stage train     # 모은 데이터로 readout 학습 + 전 arm PPO + 평가
```

두 명령에 **같은 `--config`와 `--system-config`를 준다.** 수집 산출물은
`results/<실험>/<mode>/`와 누적 datastore(`results/datastore/collected.sqlite3`)에
남으므로, 학습 단계를 여러 번 돌려도 비행은 반복되지 않는다. `--stage train`이
필요한 데이터셋을 찾지 못하면 비행하지 않고 오류로 멈춘다 — 학습을 돌린다고
생각한 실행이 조용히 수집으로 되돌아가는 일은 없다.

인자 없는 `./run.sh`는 `config/experiments/three_arm_burst_comparison.yaml`을 그
파일이 선언한 예산으로, 기계가 재는 만큼의 페어에서 실행한다. 대체된 6-덱 2-arm
설계도 그대로 남아 있다.

```bash
./run.sh --config Ontology_RGAT_UAV_RL_ISAAC_PX4/config/experiments/two_pipeline_comparison.yaml
./run.sh --seminar-fast          # publication_claim_allowed: false — 논문 결과로 보고하지 않는다
```

**full 전에 `--mode quick`을 먼저 돌리기를 권한다.** full 예산은 학습 2 arm × 1000
에피소드, 평가 3 arm × 1200으로 약 1.5–2일이다. 2026-09-22의 quick 패스는 full
이었다면 하루치 연산 뒤에야 드러났을 결함 세 개를 수십 분에 잡았다.
`--total-train-episodes`, `--eval-episodes`, `--rgat-data-episodes`,
`--rgat-epochs`, `--results-dir`로 예산을 덮어쓸 수 있다.

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/check_workspace.sh      # 시뮬레이터 없이 계약 전체 검증
```

검증은 baseline 보상 보존, `λ_fov=0` 동일성, 공통 actor/critic/estimator/PPO 계약,
정보 누출 차단, graph 결정성과 도달성, 확률·보상 범위, R-GAT 동결, episode 단위
split, 단독 및 2-pair 실행 계약을 포함한다.

## 8. 현재 상태와 알려진 제약

### ⚠ 제어 대역폭이 강제되지 않는다 — 모든 수치를 읽기 전에

학습기는 Isaac과 lockstep이 **아니다**. `isaac.lockstep: true`는 Isaac↔PX4
전용이고, Isaac의 월드 루프는 렌더·물리가 허용하는 한 빠르게 돌며, 학습기는
게이트웨이를 비동기로 샘플링한다. **어느 쪽도 상대를 기다리지 않는다.** 따라서
제어 스텝당 시뮬 시간은 보장된 값이 아니라 루프가 그때그때 해낸 값이다.

| 상태 | px4 s/step | 실효 제어율 |
|---|---|---|
| 정상 | 0.104 | **9.6 Hz** |
| 열화 | 0.69–0.95 | **1.05–1.45 Hz** |

즉 공칭 10 Hz 제어율은 Isaac이 우연히 실시간의 1/9로 돌았기 때문에 얻어진 값이며,
동시 렌더 부하에 따라 **8배까지 흔들린다**. 결과적으로:

* `steps × cfg.sim.dt`로 계산되는 모든 시간 지표가 **7–9배 과소** 기록된다 —
  `touchdown_time_s`와 네 개의 FOV 손실 지속 시간 열이 모두 여기 해당하며, 이들은
  비교의 **종속변수**다. 페이싱 수정 후에는 사후 보정 없이 그대로 맞아진다.
* image servo의 rate 항이 같은 배수로 과대하다. privileged PD는 `dt`를 쓰지 않아
  영향이 없다.

**기전은 독립적인 두 데이터셋으로 확립**되었으나 **착륙 결과와의 인과는 어느
쪽으로도 입증되지 않았다**(67개 비행에서 착륙 1.56 Hz, 실패 1.59 Hz로 분리 없음).
제어율 임계값을 근거로 튜닝하지 말 것.

완화책 `isaac.max_sim_speed_ratio`를 구현했으나 **켜면 안 된다** — Isaac의
`world.current_time`을 제어하는데 학습기가 적분하는 것은 PX4의 `px4_time_us`이고,
두 시계는 8.6배 어긋난다. 게다가 Isaac을 재우면 PX4가 CPU를 더 얻어 제어율이 오히려
떨어진다. 실제 해법은 학습기가 월드를 구동하는 스텝 요청 핸드셰이크다. 상세는
[3-arm 비교](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_ARM_BURST_COMPARISON.md) §5.4–5.5.

### 배관은 검증됨, 결과는 아직

파이프라인은 keypoint → teacher → 행동복제 → PPO → FOV-risk R-GAT → checkpoint
선택 → 3-arm 평가 → 리포트까지 완주한다(2026-09-22 quick 패스). 그 패스는 arm당
학습 10 에피소드, 실효 제어율 1.2 Hz에서 세 arm 모두 착륙 0건이었다. 확인된 것은
**구조**이지 성능이 아니다.

## 9. 주장하지 않는 것

* PBRS 최적정책 불변성 — 제안항은 potential-based shaping이 아니다.
* 운용 안전 보장 — 성공 gate는 연구 평가 기준이지 안전 monitor가 아니다.
* Attention 가중치의 인과 해석, 관계 이름의 단조성.
* 온톨로지 *단독* 기여 — 현재 설계는 제안 보상 모듈 전체의 효과만 비교한다.
* PACMAN 재현 — encoder와 표적은 같은 인터페이스를 갖는 문서화된 근사다.

## 10. 문서

| 문서 | 내용 |
|---|---|
| **[3-arm 급가속 이탈 비교](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_ARM_BURST_COMPARISON.md)** | **현재 주 비교** — 덱, 세 arm, 제어 포락선, 제어 대역폭 제약 |
| [구현 개요](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md) | 알고리즘과 코드의 대응, artifact |
| [아키텍처와 정보경계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ARCHITECTURE.md) | 관측·보상·critic 경계, 좌표계, 타이밍 |
| [제안 알고리즘](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_FOV_RISK.md) | graph, 표적, 목적함수, readout, 동결 |
| [실험 설계 (6-덱 2-arm)](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/TWO_PIPELINE_COMPARISON.md) | 통제변수, 실험 요인, 지표, 통계, 실험 이력 |
| [Shin baseline](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md) | 원문 구성과 구현 대응 |
| [논문 대조](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/PAPER_FIDELITY.md) | 확인된 값, 미기재 항목, 선언된 이탈 |
| [운영 절차](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md) | 실행 프로파일, 산출물, 상태 확인 |
