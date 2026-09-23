# Ontology-Graph State Representation for Vision-Based UAV Landing on a Moving Platform

이 저장소는 **온톨로지 상황 그래프를 강화학습 정책의 상태 표현으로 사용하는 방법**을
제안하고, 그 표현 하나만을 실험 요인으로 갖는 통제 비교를 Isaac Sim / Pegasus /
PX4 SITL 위에서 수행한다.

## 1. 문제 정의

이동하는 지상 플랫폼(UGV)에 단안 하향 카메라만으로 착륙하는 UAV는 접근 후반부에
구조적인 딜레마를 만난다. 패드에 가까워질수록 표적의 상대 각속도가 커져 패드가
프레임을 벗어나기 쉽고, 한 번 벗어나면 관측이 끊긴 상태에서 재획득해야 한다.
정책이 이 상황을 다루려면 "지금 무엇이 위험한가"를 상태에서 읽을 수 있어야 하는데,
원시 이미지와 자기 자세만으로 이루어진 관측 벡터는 그것을 **명시하지 않는다**.

> 완전한 Shin et al. baseline을 어떤 항도 바꾸지 않은 채, **관측에 온톨로지 상황
> 그래프 하나를 더하면** 표적 가시성 유지와 이동 플랫폼 착륙 성능이 향상되는가?

## 2. 이전 설계에서 무엇이 바뀌었고 왜 바뀌었는가 (2026-09-23)

이전 설계는 온톨로지를 **보상**에 넣었다. 동결된 R-GAT이 가까운 미래의 FOV
비가용 비율을 예측하고 그 스칼라를 `-λ_fov q(G_t)`로 뺐다. 두 가지 문제가 있었다.

1. **온톨로지의 기여를 그것이 곱해진 가중치와 분리할 수 없다.** 이미 작동하는
   5항 shaping 합에 대해 λ=0.1짜리 항은 작은 섭동이며, 결과는 귀속하기에 너무
   작은 차이이거나 손으로 고른 상수도 만들어냈을 차이다.
2. **정책은 온톨로지를 본 적이 없다.** 그래프가 담은 구조는 상황 전체를 하나로
   평균한 스칼라 벌점으로만 actor에 닿았다.

지금은 **그래프가 정책의 상태**다. 두 학습 arm의 보상은 완전히 같고, 유일한 실험
요인은 actor와 critic이 `G_t`를 보는지 여부다.

동시에 **제어 자유도를 줄였다.** 기존 `[v_x, v_y, v_z, ω_z]` 4자유도에서는 정책이
횡방향 추종·기수 유지·착륙을 동시에 풀어야 했고, 그 아래에서 온톨로지가 무엇을
바꿨는지 보이지 않았다. 지금은 사가탈 평면 3채널이며 **세 arm 전부 동일**하다.

두 변경 모두 축소 2-D 연구
([ugv_landing_2d_workspace](https://github.com/junsuk123/ugv_landing_2d_workspace))가
먼저 확립한 것이고, 이 저장소는 그 방법론을 실제 스택에 올린 것이다.

## 3. 기여

1. **정책 상태로서의 온톨로지 그래프.** 9 node · 4 관계 · 선언 간선 12개 + node별
   self-loop로 이루어진 결정적 상황 그래프를 R-GAT으로 부호화하고, 그래프 수준
   읽기 `g_t`를 actor와 critic 입력에 이어붙인다. 시뮬레이터 truth, 6-D 상대상태
   추정, critic state는 구조적으로 입력될 수 없다.
2. **PPO가 함께 학습하는 부호기.** 오프라인 단계도, 동결 산출물도, checksum
   게이트도 없다. 기울기는 PPO 목적함수에서 관계형 커널까지 이어진다. 그래프는
   고정 특징이 아니라 학습되는 표현이다.
3. **상태 표현으로 쓰기 위한 두 적응.** 두 자리 수 범위를 한 상수로 덮지 못하는
   잘라내기 대신 부드러운 포화 `x/(x+scale)`, 그리고 크기만 남는 온톨로지 값에
   방향을 되살리는 `[-1,1]` 부호 채널. node·간선·관계는 바꾸지 않았다.
4. **축소된 평면 제어 엔벨로프.** 전후진 가속도, 고도 가속도, 종방향 틸트의 3채널.
   횡방향 위치와 기수는 실험의 제약이며 제어가 아니다. 실행 중 덱 참값을 읽는
   코드는 없다.
5. **단일 요인 통제 비교와 비학습 기준선.** 두 학습 agent는 환경·카메라·encoder·
   LSTM·추정기·보조손실·actor/critic·PPO 하이퍼파라미터·curriculum·**보상 전체**·
   seed·episode 예산까지 모두 공유하며, 같은 seed에서 공유 파라미터가 비트 단위로
   동일하다. 여기에 PN 유도 대조군이 같은 덱·같은 seed·같은 착륙 기준·**같은 3채널
   액션**으로 함께 평가된다.

## 4. 방법

### 4.1 Baseline 보상 (두 arm 공통, 변경 없음)

상대상태를 `s^rel = [Δp^b, Δv^b] ∈ R^6` (플랫폼 − UAV, UAV body frame), 행동을
`a_t = [a_fwd, a_z, tilt]`라 하면 보상은 구간별로 정의된다.

$$
r(t)=
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

**이 보상은 두 학습 arm에 완전히 동일하다.** 다섯 번째 항만 선언된 이탈이다:
원문의 `-2|ω_z|`는 요레이트 채널에 대한 벌점인데 그 채널이 없어졌으므로, 같은
가중치로 그 자리를 대신한 종방향 틸트 채널에 옮겼다(`-2|tilt|`). 종료 보상은
shaping을 **대체**한다. 다섯 항의 정확한 식은
[Shin baseline 대응](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md),
이탈의 근거는
[논문 대조](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/PAPER_FIDELITY.md) §3에 있다.

### 4.2 제안하는 상태 표현

```
기준 arm :  이미지 + proprio(7) → encoder → LSTM → latent
            actor  ← latent[6:] ‖ proprio
            critic ← proprio ‖ s^rel

제안 arm :  위와 완전히 동일  +  같은 시각 의미채널 → G_t → R-GAT → g_t
            actor  ← latent[6:] ‖ proprio ‖ g_t
            critic ← proprio ‖ s^rel ‖ g_t
```

```
H_t = tanh( H_1 + RGAT_2(H_1) ),  H_1 = tanh( RGAT_1(X_t) )
g_t = tanh( W [ mean(H_t) ; max(H_t) ] + b )
```

읽기는 **모든 node를 본다**. 특정 node를 고르지 않는 것이 보상 readout과의 차이다.
actor와 critic은 부호기를 하나씩 가진다 — 두 망은 이미 학습률과 Adam 상태가
분리되어 있고, 공유 부호기는 서로 다르게 스케일된 두 기울기를 한 파라미터에
합쳐야 한다.

### 4.3 온톨로지 스키마 (최소 핵심)

| 항목 | 값 |
|---|---|
| Node | 9개 = 위험 6 + 지원 1 + 중간 1 + 목표 1 |
| 관계 | `degrades`, `supports`, `contributes`, `self` |
| Edge | 선언 12개 + node별 self-loop 9개 |
| Node 특징 | `X ∈ R^{14×9}` = 값, 여값, 위험 표시, 편향, **방향 부호**, 9-D one-hot |
| 부호기 | R-GAT(14→32) → tanh → R-GAT(32→32) + 잔차 → mean+max 읽기 → 32 |
| 목표 node | `SafeLanding`, 값은 항상 0 (정답 누출 방지) |

더 줄이지 않는 이유는 셋이며 모두 축소 연구가 측정한 것이다: 2단 구조가 없으면
전달할 메시지가 없고, 관계 유형을 합치면 "관계형" 주의가 성립하지 않으며, 목표에
도달하지 못하는 입력 node는 스키마가 조용히 버린다. 상세는
[제안 알고리즘](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_STATE.md).

### 4.4 축소된 평면 제어 엔벨로프 (세 arm 공통)

```
a_t = [a_fwd, a_z, tilt_long]          각 채널 [-1, 1]
  a_fwd      전후진 가속도  →  적분되어 속도 설정점
  a_z        고도 가속도    →  적분되어 속도 설정점
  tilt_long  종방향 틸트    →  a = g·tan(θ) 가속도 피드포워드로 PX4에 전달

제약 (제어가 아님):  횡방향 속도 ≡ 0,  요레이트 ≡ 0,  기수 = 덱 진행 방향
```

세 채널은 정상상태에서 독립이 아니다 — 쿼드로터는 사가탈 평면에서 부족구동이고,
그런 척하는 것은 허구다. **그 결합이 FOV 유지 문제 자체다**: 프레임을 지키려고
기울이면 접근이 교란된다. 제약이 오라클이 아닌 이유(덱이 일정 기수 직선 주행이고
진입이 그 선 위에 놓이므로 구성적으로 성립)는
[평면 엔벨로프](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/PLANAR_ENVELOPE.md).

## 5. 비교하는 세 arm

| arm | 종류 | 구성 | 차이 |
|---|---|---|---|
| `pn_guidance_v1` | **비학습 대조군** | 사가탈 평면 비례항법 유도. 동결 encoder 출력만 읽음 | 학습·보상·체크포인트 없음 |
| `shin_se_fixed` | 학습 | 6-keypoint encoder, LSTM, 6-D 상대상태 추정과 보조손실, PPO actor, 비대칭 critic, 고정 5성분 보상, active-perception 보상 | 없음 (baseline) |
| `shin_se_onto_rgat_state` | 학습 | 위 구성을 **보상까지 포함해 전부 동일하게 유지** | 관측에 `G_t` 추가 |

소거 실험은 같은 부호기 클래스와 같은 PPO로 실행한다:
`shin_se_onto_gat_state`(관계 유형 통합), `shin_se_node_pool_state`(메시지 전달 없음).
은퇴한 보상항 arm `shin_se_onto_rgat_recovery`는 자기 id로 남아 있고 지금도
실행할 수 있다. 두 방법을 한 실행에 섞는 것은 검증 단계에서 거부된다.

`assert_primary_baseline_equivalence()`와 YAML/spec 교차검증이 실행 시점에 계약
flag의 차이를 즉시 실패로 만든다.

## 6. 시스템 파이프라인

```
Isaac Sim / Pegasus  →  카메라 · 접촉 · UGV 주행 · 배터리 · 외란
        │
        ├── PX4 SITL (OFFBOARD)  ←  [v_fwd, 0, v_z] + yaw 유지 + 틸트 피드포워드
        │
        └── ROS 2 gateway  →  learner
                              ├── 동결 6-keypoint encoder
                              ├── LSTM(512) → latent(256)
                              │     ├── [0:6]  → 6-D 상대상태 보조 회귀
                              │     └── [6:]   → actor MLP → 3채널 행동
                              ├── asymmetric critic([u_t, s^rel](+g_t), 학습 전용)
                              └── (제안 arm) 시각 의미채널 → 9-node 그래프
                                                        → R-GAT → g_t → actor·critic
```

한 transition은 0.1초이고 episode horizon은 300 step이다. 실행은 **데이터 수집
단계와 학습 단계로 분리**되어 있다.

**1단계 — 데이터 수집** (`--stage collect`)

1. 공통 keypoint encoder 사전학습 · 실기 카메라 측량 검증 후 동결
2. 공통 PN 유도 시연 비행 → 행동 복제(BC) 워밍업용 데이터셋

**2단계 — 학습** (`--stage train`)

3. **모든 arm의 PPO를 동시에 시작** (동일 예산 · 동일 seed · 동일 보상)
4. held-out 결정론 seed로 checkpoint 선택
5. paired · crossover 평가, 표와 그림 저장

**오프라인 보상 설계 단계가 없다.** 그래프 부호기는 PPO가 학습하므로 동결할
산출물이 없고, FOV-risk 데이터 수집과 R-GAT 동결은 은퇴한 보상항 경로에서만
켜진다.

두 학습 arm은 **같은 시연 집합**에서 각자의 모델로 BC 워밍업한다. 하나의
checkpoint를 공유할 수는 없다 — 제안 arm에는 기준 arm에 없는 부호기 파라미터가
있다. 공유되는 것은 시연, epoch 수, seed다.

## 7. 실험 설계

* **통제**: 환경, 초기조건 분포, UGV 궤적, seed, episode 예산, PPO 설정,
  curriculum, **보상 전체**, 액션 공간, 제어 엔벨로프.
* **요인**: 두 학습 arm 사이에서는 관측에 `G_t`가 있는지 하나. 대조군은 요인이
  아니라 기준선이다.
* **평가 시나리오**: 세 덱. 각각 일정한 기수의 직선 주행이며 속도가 3구간으로
  나뉜다. 축소 연구의 속도 행렬을 반송차 상한(1.0 m/s)에 맞춰 ×1/7로 사상했고
  구간 전환 시각(3 s, 15 s)은 그대로다.

  | 시나리오 | 1구간 | 2구간 | 3구간 |
  |---|---|---|---|
  | `segmented_cruise_slow` | 0.143 | 0.571 | 0.214 m/s |
  | `segmented_cruise_medium` | 0.214 | 0.786 | 0.286 m/s |
  | `segmented_cruise_fast` | 0.286 | **1.000** | 0.357 m/s |

  첫 전환은 어떤 제어기도 그 전에 착륙해 교란을 건너뛸 수 없을 만큼 이르다.
  구간 전환은 속도 계단이 아니라 유계 가속도 램프다.

  **세 덱 모두 같은 차선을 같은 방향으로 달린다.** heading은 에피소드마다 뽑는
  값이 아니라 프로파일 상수(+X)이고, 병렬 페어는 그 차선을 가로질러 80 m 간격의
  **한 줄로** 스폰되어 나란히 출발한다. 고정 heading은 임의 heading 덱이 쓰던
  경계 회전 보정을 쓸 수 없으므로, 차선은 셔틀이다 — 덱이 150 m를 넘어가면
  *다음* 에피소드가 반대 방향으로 시작한다. 반전은 리셋에서만 일어나므로 어떤
  에피소드도 반전을 포함하지 않는다.
  [평면 엔벨로프](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/PLANAR_ENVELOPE.md) §4.
* **병렬 실행**: 한 Isaac stage 안에서 독립 UAV/UGV·PX4·namespace·port·learner·
  buffer·optimizer 쌍을 사용한다(현재 4). 학습은 학습 arm에만 페어를 나눈다.
  방법과 물리 쌍의 대응은 `training_replicate`마다 cyclic Latin square로 바꾼다.
* **통계**: 동일 replicate·시나리오·seed 짝의 차이를 모으고, replicate → episode
  순의 계층 bootstrap으로 95% 신뢰구간을 낸다.

### 주 지표

| 분류 | 지표 |
|---|---|
| 착륙 | 엄격 Landing Success Rate, touchdown lateral error, 수직/상대 수평 속도, tilt, 각속도 |
| 기하 FOV | Loss Episode Rate, Retention Ratio, 평균/최대 연속 손실 시간, 재획득율·재획득 시간 |
| 인지 품질 | `keypoint_confidence_mean`, `visible_keypoint_fraction_mean`, `low_keypoint_visibility_fraction` |
| 공통 진단 | 위치/속도 추정 RMSE와 loss, return, touchdown 시간 |

엄격 성공 판정은 접촉 ∧ 수평오차 ≤ 0.35 m ∧ |v_z| ≤ 0.55 m/s ∧ 상대 수평속도
≤ 0.45 m/s ∧ tilt ≤ 10° ∧ 각속도 ≤ 45°/s다. 논문 기준의 접촉 성공률은
`paper_success`로 병기하되 엄격 기준을 대체하지 않는다.

**기하 FOV와 인지 품질은 서로 다른 변수다.** 패드가 프레임 안에 있는데 encoder가
놓치면 인지 저하이지 FOV 손실이 아니다.

## 8. 실행

```bash
./run.sh                 # 기본: 평면 3-arm 비교, full 예산 (수집 → 학습)
./run.sh --mode quick    # 전 구간 배관 검증 (수십 분, full 전에 권장)
```

```bash
./run.sh --stage collect   # 모든 페어에서 데이터셋만 모으고 종료
./run.sh --stage train     # 모은 데이터로 전 arm PPO + 평가
```

두 명령에 **같은 `--config`와 `--system-config`를 준다.** 수집 산출물은
`results/<실험>/<mode>/`와 누적 datastore에 남으므로 학습 단계를 여러 번 돌려도
비행은 반복되지 않는다.

교체된 설계도 그대로 남아 있고 실행할 수 있다.

```bash
# 은퇴한 보상항 방법 (3-arm 급가속 이탈 비교)
./run.sh --config Ontology_RGAT_UAV_RL_ISAAC_PX4/config/experiments/three_arm_burst_comparison.yaml
# 6-덱 2-arm 설계
./run.sh --config Ontology_RGAT_UAV_RL_ISAAC_PX4/config/experiments/two_pipeline_comparison.yaml
# 소거 실험
./run.sh --pipelines shin_se_fixed shin_se_onto_gat_state
```

시뮬레이터 없이 수행하는 정적·단위 검증:

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/check_workspace.sh
python -m pytest -q tests/test_ontology_graph_state.py tests/test_planar_envelope.py
```

## 9. 현재 상태와 알려진 제약

### ⚠ 제어 대역폭이 강제되지 않는다 — 모든 수치를 읽기 전에

학습기는 Isaac과 lockstep이 **아니다**. `isaac.lockstep: true`는 Isaac↔PX4
전용이고, Isaac의 월드 루프는 렌더·물리가 허용하는 한 빠르게 돌며, 학습기는
게이트웨이를 비동기로 샘플링한다. **어느 쪽도 상대를 기다리지 않는다.**

| 상태 | px4 s/step | 실효 제어율 |
|---|---|---|
| 정상 | 0.104 | **9.6 Hz** |
| 열화 | 0.69–0.95 | **1.05–1.45 Hz** |

즉 공칭 10 Hz 제어율은 Isaac이 우연히 실시간의 1/9로 돌았기 때문에 얻어진 값이며,
동시 렌더 부하에 따라 **8배까지 흔들린다**. 결과적으로 `steps × cfg.sim.dt`로
계산되는 모든 시간 지표가 **7–9배 과소** 기록된다 — `touchdown_time_s`와 네 개의
FOV 손실 지속 시간 열이 모두 여기 해당하며, 이들은 비교의 **종속변수**다.

**기전은 독립적인 두 데이터셋으로 확립**되었으나 **착륙 결과와의 인과는 어느
쪽으로도 입증되지 않았다**(67개 비행에서 착륙 1.56 Hz, 실패 1.59 Hz로 분리 없음).
제어율 임계값을 근거로 튜닝하지 말 것. 완화책 `isaac.max_sim_speed_ratio`를
구현했으나 **켜면 안 된다** — 상세는
[3-arm 비교](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_ARM_BURST_COMPARISON.md) §5.4–5.5.

이 제약은 평면 개편과 무관하게 그대로다. 액션이 가속도 명령이 되면서 **제어
주기의 불확실성이 속도 적분에도 들어간다**는 점이 새로 추가된다: 컨트롤러는
`cfg.sim.dt`로 적분하는데 실제 경과 시간은 그보다 길 수 있다. 세 arm에 동일하게
적용되므로 비교의 교란요인은 아니지만, 절대 가속도 값을 물리량으로 인용하지 말 것.

### 배관은 검증됨, 결과는 아직

평면 개편(2026-09-23)은 정적·단위 검증을 통과했다(937 tests). 실기 전 구간
`--mode quick` 패스는 아직 수행하지 않았다. **full 전에 반드시 먼저 돌릴 것.**

## 10. 주장하지 않는 것

* 온톨로지 *단독* 기여 — 비교하는 것은 "R-GAT으로 부호화한 9채널 의미 요약을
  상태에 더하면 나아지는가"이지, 온톨로지라는 형식 자체의 가치가 아니다.
* 9개 의미 채널로 상황을 요약한다는 것 자체의 상한이 없다는 것 — 축소 연구는
  교사 모방 손실로 이 상한을 측정했다(그래프 0.0667 대 관측 벡터 0.0071).
* 운용 안전 보장 — 성공 gate는 연구 평가 기준이지 안전 monitor가 아니다.
* Attention 가중치의 인과 해석, 관계 이름의 단조성.
* Table III를 변경 없이 재현했다는 것 — 다섯 번째 항은 선언된 이탈이다.
* PACMAN 재현 — encoder와 표적은 같은 인터페이스를 갖는 문서화된 근사다.

## 11. 문서

| 문서 | 내용 |
|---|---|
| **[제안 알고리즘 (상태 표현)](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_STATE.md)** | **현재 주 방법** — 스키마, 부호기, 정보경계, 소거 실험 |
| **[평면 제어 엔벨로프](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/PLANAR_ENVELOPE.md)** | **현재 제어 계약** — 3채널 액션, 두 제약, 오라클이 아닌 이유 |
| [구현 개요](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md) | 알고리즘과 코드의 대응, artifact |
| [아키텍처와 정보경계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ARCHITECTURE.md) | 관측·보상·critic 경계, 좌표계, 타이밍 |
| [Shin baseline](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md) | 원문 구성과 구현 대응 |
| [논문 대조](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/PAPER_FIDELITY.md) | 확인된 값, 미기재 항목, 선언된 이탈 9건 |
| [운영 절차](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md) | 실행 프로파일, 산출물, 상태 확인 |
| [은퇴: FOV-risk 보상항](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_FOV_RISK.md) | 이전 제안 방법. arm은 지금도 실행 가능 |
| [은퇴: 3-arm 급가속 이탈 비교](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_ARM_BURST_COMPARISON.md) | 이전 주 비교. 제어 대역폭 제약의 원 기술 |
| [은퇴: 6-덱 2-arm 설계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/TWO_PIPELINE_COMPARISON.md) | 통제변수, 지표, 통계, 실험 이력 |
