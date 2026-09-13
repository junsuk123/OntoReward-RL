# Ontology–R-GAT–RL 이동식 드론 착륙

NVIDIA Isaac Sim·Pegasus·PX4 SITL에서 도로를 주행하는 UGV의 패드에 드론을
착륙시키는 vision-based recurrent PPO 시스템이다. 핵심 제안은 명시적 상대상태
추정기 없이 영상 의미를 ontology graph로 구성하고, 동결된 R-GAT이 상태별 보상
가중치와 관측성 potential을 함께 생성하는 것이다.

![Isaac Sim의 Meta-Sejong S5 이동식 착륙 환경](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/isaac_sim_s5_live.png)

## 1. 강화학습 문제 정의

비교군과 제안 모델은 아래 강화학습 계약을 공유한다.

| 요소 | 이 프로젝트에서의 정의 |
|---|---|
| **Environment** | Isaac Sim/Pegasus의 multicopter·UGV·접촉 물리, PX4 SITL 저수준 비행제어, 카메라·배터리·외란 모델 |
| **Agent** | 동결 6-keypoint encoder, LSTM memory, PPO actor와 asymmetric critic |
| **State** $s_t$ | 환경 내부의 UAV·패드 상대 동역학, 접촉, 배터리 등을 포함한 Markov state. 전체 state는 actor에 직접 제공하지 않는다. |
| **Observation** $o_t$ | Actor 입력 $o_t=[I_t,u_t]$. $I_t$는 $512\times320$ 흑백 영상, $u_t\in\mathbb{R}^7$은 body-frame 속도 3축과 자세 quaternion 4개다. |
| **Action** $a_t$ | $a_t=[v_x,v_y,v_z,\omega_z]$: body/heading frame 속도 3축과 yaw rate. PX4가 자세와 motor thrust를 제어한다. |
| **Reward** $r_t$ | 안전 착륙/실패 terminal reward와 접근·하강·yaw shaping. 제안법은 ontology R-GAT으로 shaping weight와 semantic potential을 적응시킨다. |

Actor는 현재 영상 한 장만 쓰지 않고 LSTM hidden state를 통해 관측 이력
$o_{0:t}$를 사용한다.

$$
a_t \sim \pi_\theta(a_t\mid o_{0:t}),
\qquad
\theta^*=\arg\max_\theta
\mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T}\gamma^t r_t\right].
$$

![강화학습의 state, observation, action, reward, agent, environment 관계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/rl_contract.svg)

## 2. 최종 비교군과 제안 모델

최종 핵심 비교는 세 파이프라인이다. 카메라, keypoint encoder, LSTM/actor 크기,
critic, controller, PPO 설정, 초기조건과 seed는 동일하다.

| 파이프라인 | 상태추정 보조손실 | Active-perception reward | Ontology/R-GAT | 보상 |
|---|---:|---:|---:|---|
| `shin_se_fixed` | 있음 | 있음 | 없음 | Shin et al. 기반 고정 5성분 + 추정오차 penalty |
| `no_se_fixed` | 없음 | 없음 | 없음 | 동일한 고정 5성분 |
| **`onto_rgat_adaptive_weight_no_se`** | **없음** | **없음** | **23-node hybrid R-GAT** | **상태 적응 5성분 + semantic PBRS** |

![동일한 actor와 PPO 위에서 보상 학습 신호만 비교하는 세 파이프라인](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/pipeline_comparison.svg)

### A. `shin_se_fixed`

Shin et al. (2026)의 구조를 Isaac/PX4 환경에 맞춘 비교군이다. LSTM latent의 처음
6개 성분을 body-frame 패드 상대 위치·속도로 보조학습한다.

$$
\tilde{s}^{\mathrm{rel}}_t=
[\Delta\tilde{x}^{b}_t,\Delta\tilde{v}^{b}_t]\in\mathbb{R}^{6},
\qquad
L^{\mathrm{est}}_t=\frac{1}{6}
\left\|s^{\mathrm{rel}}_t-\tilde{s}^{\mathrm{rel}}_t\right\|_2^2.
$$

다음 시점의 추정오차가 커지는 행동에는 active-perception penalty를 준다.

$$
r^{\mathrm{active}}_t
=-\alpha\,\mathrm{clip}
\!\left(\beta(L^{\mathrm{est}}_{t+1}-\tau),0,1\right),
\quad (\alpha,\beta,\tau)=(0.1,1,0.01).
$$

### B. `no_se_fixed`

명시적 상태추정 head, $L^{\mathrm{est}}$, active-perception reward를 모두 제거한
estimator-free 비교군이다. LSTM actor와 고정 Table-III shaping만 남기므로,
ontology가 제공하는 효과를 분리해서 측정하는 기준점이다.

### C. `onto_rgat_adaptive_weight_no_se` — 제안 방법

Actor observation은 `no_se_fixed`와 동일하다. keypoint 출력, UAV proprioception,
배터리에서 12개의 bounded semantic feature를 만들고, 이를 18개 의미/목표 노드와
5개 보상개념 노드로 구성한다.

- 영상·시간 의미: confidence, visible fraction, alignment, apparent scale,
  image/scale motion safety, visibility memory, reacquisition, visual-loss risk
- 기체·에너지 의미: vertical-motion safety, attitude stability, battery risk
- 관계: `indicates`, `supports`, `constrains`, `self`
- 출력: 상태 적응 보상 가중치 $w(G_t)\in\mathbb{R}^5$와 scalar potential
  $\Phi(G_t)$

![23-node ontology와 두 R-GAT head를 사용하는 제안 보상](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/adaptive_rgat_hybrid.svg)

R-GAT은 실제 Isaac/PX4 trajectory로 PPO 전에 학습한다. 데이터는 성공, 일반 실패,
위험 접촉 실패를 episode/scenario 단위로 층화하고, validation-best model만 품질
gate를 통과한 뒤 PPO 동안 완전히 동결한다.

## 3. 보상함수

### 공통 terminal reward

$$
r^{\mathrm{task}}_t=
\begin{cases}
+10, & \text{안전 착륙},\\
-10, & \text{충돌, 과도한 이탈, 배터리 고갈 또는 timeout},\\
0, & \text{그 외}.
\end{cases}
$$

### 공통 5개 shaping 성분

$d^{xy}_t=\|\Delta x^{b}_{t,xy}\|_2$라 두고, 코드와 데이터 설계가 함께 사용하는
무가중 성분 $\rho_t\in\mathbb{R}^5$는 다음과 같다.

$$
\begin{aligned}
\rho_{1,t}&=\mathrm{clip}(d^{xy}_t-d^{xy}_{t+1},-1,1),\\
\rho_{2,t}&=\frac{\mathrm{clip}(|\Delta z_t|-|\Delta z_{t+1}|,-1,1)}
{\max(d^{xy}_{t+1},1)},\\
\rho_{3,t}&=-\max(v^{\mathrm{uav}}_{z,t+1}+0.5,0),\\
\rho_{4,t}&=-\max(\Delta z_{t+1},0),\\
\rho_{5,t}&=-|\omega^{\mathrm{cmd}}_{z,t}|.
\end{aligned}
$$

서로 다른 단위를 직접 합산하지 않도록

$$
\bar\rho_{i,t}=\mathrm{clip}\!\left(\frac{\rho_{i,t}}{c_i},-1,1\right),
\qquad
c=\left(1,1,1.5,3,\frac{\pi}{2}\right)
$$

로 정규화한다. 고정 비교군의 weight는
$w^0=(1,1,0.5,1,2)$이고 $\sum_i w_i^0=5.5$다.

### 제안 모델의 상태 적응 weight

R-GAT weight head의 logit을 $z_i(G_t)$라 하고 $p_i^0=w_i^0/5.5$라 두면,

$$
\tilde p_i(G_t)=
\frac{p_i^0\exp\{\kappa\tanh z_i(G_t)\}}
{\sum_j p_j^0\exp\{\kappa\tanh z_j(G_t)\}},
$$

$$
p_i(G_t)=\varepsilon p_i^0+(1-\varepsilon)\tilde p_i(G_t),
\qquad
w_i(G_t)=5.5\,p_i(G_t),
$$

이며 $\kappa=\ln2$, $\varepsilon=0.2$다. 따라서 모든 weight는 양수이고 합은 항상
5.5이며, 초기 수동 설계에서 무제한으로 이탈하지 않는다.

5개 성분에 직접 존재하지 않는 FOV 유지·재포착 신호는 같은 R-GAT encoder의
potential head로 보완한다.

$$
r^{\mathrm{sem}}_t
=\lambda_\Phi\left[\gamma\Phi(G_{t+1})-\Phi(G_t)\right],
\qquad
(\lambda_\Phi,\gamma)=(0.75,0.99).
$$

Terminal 다음 상태는 absorbing state로 두어 $\Phi(G_{t+1})=0$으로 처리한다. 최종
제안 보상은

$$
r^{\mathrm{proposed}}_t=
\begin{cases}
r^{\mathrm{task}}_t+r^{\mathrm{sem}}_t,
& \text{terminal},\\[2mm]
\displaystyle\sum_{i=1}^{5}w_i(G_t)\bar\rho_{i,t}+r^{\mathrm{sem}}_t,
& \text{그 외}
\end{cases}
$$

이다.

## 4. 안전 착륙 판정

단순 접촉은 성공이 아니다. 아래 여섯 조건을 모두 만족해야 `paper_success=1`이다.

![접촉, 위치, 수직속도, 상대수평속도, 자세와 각속도를 모두 평가하는 착륙 성공 gate](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/landing_success_gate.svg)

접촉 위치는 실제 접촉 순간 값을 쓰고, 충격 반동이 섞이지 않도록 속도·자세·각속도는
직전 비접촉 샘플을 사용한다.

## 5. 전체 파이프라인 실행

저장소 루트에서 최종 실행 명령은 하나다.

```bash
./run.sh
```

인자 없는 기본 실행은 세미나용 핵심 3-arm 프로필을 선택하고, 한 Isaac stage의
UAV/UGV 3쌍에서 세 방법을 병렬 학습·평가하며, 완료 후 시각화 stack을 유지한다.
즉 다음 명시적 명령과 같다.

```bash
./run.sh --seminar-fast --parallel-pairs 3 --stay-open
```

병렬 모드는 UAV/UGV/PX4/Gateway/ROS topic/reset/trajectory/optimizer를 쌍별로 분리한다.
공통 keypoint·BC와 실제 trajectory 기반 R-GAT 학습·동결을 먼저 끝낸 다음 세 PPO를
동시에 시작한다. 세 UGV는 조사된 동일 폐곡선 waypoint의 0%, 8%, 16% 지점에
열차처럼 간격을 두고 배치되어 모두 도로 위를 같은 방향으로 주행한다. seed·상대
초기조건·외란 계약은 동일하지만 카메라 배경은 각 route 위치의 실제 campus 장면이다.
카메라 렌더링과 PX4 부하 때문에 속도 향상은 정확히 3배가 아니며, 결과 manifest에
병렬 실행 계약을 기록한다.

실행 순서:

1. 환경·의존성·설정 provenance 검사
2. Isaac Sim, Pegasus, PX4 SITL, DDS/ROS gateway, RViz, dashboard 기동
3. 공통 keypoint encoder 준비 및 검증
4. 실제 trajectory 수집과 hybrid R-GAT 학습·검증·동결
5. 비교군과 제안 모델 PPO 학습(병렬 모드에서는 세 쌍 동시 실행)
6. 안전성 기준 best checkpoint 저장
7. 동일 scenario/seed의 paired evaluation
8. 표·그래프·manifest 생성

중단된 실행은 같은 명령으로 checkpoint와 완료 row부터 재개한다. 동시에 두 개의
flight pipeline을 시작하지 않도록 실행 lock을 사용한다.

## 6. 모니터링과 결과

- Dashboard: `http://127.0.0.1:8770/`
- ROS/RViz namespace: `/landing_rl/pair_0`, `/landing_rl/pair_1`, `/landing_rl/pair_2`
- 세미나 결과: `Ontology_RGAT_UAV_RL_ISAAC_PX4/results/seminar_fast/core3_hybrid_v3/`
- 모델: `models/<pipeline>/<pipeline>.pt`, `<pipeline>.best.pt`
- 학습 이력: `training/*.csv`, `models/*/*_training.csv`
- 평가: `evaluation/`
- 표·그림: `tables/`, `figures/`
- 재현 정보: `manifest.json`

Dashboard는 MATLAB 기본 색상 순서로 세 pair의 에피소드/step, marker, 상대 위치,
UAV·UGV 속도, 배터리와 독립 PX4/UDP/topic 상태를 동시에 표시한다. RViz는 세 landing
camera dock과 `landing_pad_0..2` 독립 TF/trajectory를 사용하며 Isaac GUI는 전체 편대를
자동으로 한 화면에 맞춘다.

![MATLAB 스타일 실시간 대시보드](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/live_dashboard_status.png)

## 7. 문서

- [시스템 개요](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SYSTEM_OVERVIEW.md)
- [제안 Hybrid R-GAT 알고리즘](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md)
- [3개 파이프라인 비교 설계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_PIPELINE_COMPARISON.md)
- [Shin et al. 논문 대응](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md)
- [아키텍처와 데이터 경계](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ARCHITECTURE.md)
- [실행·재개·진단](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md)
- [실제 기체 안전](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/HARDWARE_SAFETY.md)
- [참고문헌](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/REFERENCES.md)

구현 본체와 설치 정보는 [프로젝트 디렉터리 README](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md)에 있다.
