# LEGACY / ABLATION ONLY: 3개 파이프라인 통제 비교

> 이 문서는 이전 실험 재현용이다. 현재 기본 실행과 과학적 주장은
> [두 pipeline 비교](TWO_PIPELINE_COMPARISON.md)가 정의한다.

[문서 안내](README.md) · [시스템 개요](SYSTEM_OVERVIEW.md) ·
[현재 제안 알고리즘](ONTOLOGY_RGAT_FOV_RISK.md) ·
[Shin baseline](SHIN2026_BASELINE.md)

## 연구 질문

최종 실험의 질문은 다음과 같다.

1. 명시적 상대상태 추정과 추정오차 reward가 이동식 착륙에 주는 효과는 무엇인가?
2. 상태추정을 제거한 동일 actor에서 고정 reward만으로 어느 수준까지 학습되는가?
3. 명시적 상태추정 없이 ontology/R-GAT reward가 관측성 유지와 안전 착륙을 대신
   학습시킬 수 있는가?

![최종 3개 arm](images/pipeline_comparison.svg)

## 공통 강화학습 요소

| 요소 | 모든 arm에서 동일한 값 |
|---|---|
| Environment | Isaac Sim/Pegasus + PX4 SITL + 동일 UGV route·pad motion |
| State distribution | 동일 초기조건, curriculum, domain randomization과 paired seed |
| Actor observation | $o_t=[I_t,u_t]$: grayscale image + UAV body velocity/quaternion |
| Action | $[v_x,v_y,v_z,\omega_z]$ |
| Perception | 동일한 동결 6-keypoint encoder |
| Memory/actor | 동일 LSTM, latent slice, MLP와 Gaussian policy |
| Critic | 동일 privileged relative-state critic |
| Optimization | 동일 PPO/GAE, gradient/KL/variance bound와 checkpoint 규칙 |
| Success | 동일한 6조건 안전 착륙 gate |

## 유일한 비교 요인

| ID | 보조 상태추정 | Active reward | Reward weight | Semantic potential |
|---|---:|---:|---|---:|
| `shin_se_fixed` | 6-D MSE | 있음 | 고정 | 없음 |
| `no_se_fixed` | 없음 | 없음 | 고정 | 없음 |
| `onto_rgat_adaptive_weight_no_se` | 없음 | 없음 | R-GAT 상태 적응 | R-GAT PBRS |

### `shin_se_fixed`

$$
r_t=
\begin{cases}
r_t^{\mathrm{task}}, & \text{terminal},\\
\sum_{i=1}^{5}w_i^0\bar\rho_{i,t}+r_t^{\mathrm{active}}, & \text{그 외}.
\end{cases}
$$

상대상태 보조 loss와 active reward만 이 arm에 존재한다.

### `no_se_fixed`

$$
r_t=
\begin{cases}
r_t^{\mathrm{task}}, & \text{terminal},\\
\sum_{i=1}^{5}w_i^0\bar\rho_{i,t}, & \text{그 외}.
\end{cases}
$$

제안법과 동일하게 state-estimation head를 사용하지 않으므로 ontology/R-GAT의 순효과를
측정하는 핵심 비교군이다.

### `onto_rgat_adaptive_weight_no_se`

$$
r_t=
\begin{cases}
r_t^{\mathrm{task}}+\lambda_\Phi[\gamma\Phi(G_{t+1})-\Phi(G_t)],
& \text{terminal},\\
\sum_{i=1}^{5}w_i(G_t)\bar\rho_{i,t}
+\lambda_\Phi[\gamma\Phi(G_{t+1})-\Phi(G_t)], & \text{그 외}.
\end{cases}
$$

$w(G_t)$와 $\Phi(G_t)$는 동일한 동결 R-GAT encoder의 두 head에서 나온다.
Terminal에서는 $\Phi(G_{t+1})=0$이다.

## 공통 초기학습

세미나 profile은 짧은 실제 비행 budget에서 sparse success를 확보하기 위해 성공한
training-only teacher trajectory를 공통 BC 자료로 사용한다. 저장되는 actor 입력은
영상 embedding과 UAV proprioception뿐이다. Simulator 상대상태는 teacher action label과
critic/평가에만 사용한다.

모든 arm에 동일하게 적용하는 항목:

- 성공 시연 4 episode BC warm start
- 초기 Gaussian 표준편차 $\sigma\approx0.082$
- PPO 1–16 episode의 감쇠형 1-epoch imitation anchor
- 이후 순수 PPO update

이 상호작용 수는 PPO sample efficiency와 별도로 기록한다.

## R-GAT reward-design 데이터

제안법 전용 reward artifact를 만들기 위한 비행은 제안 PPO episode에 포함하지 않는다.
Dataset은 실제 simulator transition만 포함하며 다음 class가 모두 필요하다.

- 엄격한 안전 착륙
- 접촉 없는 timeout/이탈 등 일반 실패
- unsafe pad contact 또는 low-visibility descent를 포함한 위험 실패

Split은 frame이 아니라 `(episode, scenario)` 단위다. 동일 trajectory의 인접 frame이
train과 validation에 동시에 들어가지 않는다.

## 학습 공정성

다음 항목을 manifest와 테스트로 고정한다.

- 동일 model seed와 environment seed
- 동일한 actor parameter 수와 초기화
- actor에 critic truth가 연결되지 않음
- 제안 graph에 metric relative position/velocity가 연결되지 않음
- R-GAT parameter가 PPO optimizer에 포함되지 않음
- R-GAT model hash가 PPO 전후 동일함
- reward component 계산 함수가 dataset과 live PPO에서 동일함

## 평가

학습 중 단일 stochastic episode로 선택한 best는 결정론 policy에서 재현되지
않을 수 있다. 따라서 각 arm의 training-best와 training-latest를 학습·최종
평가 seed와 분리된 3개 scenario에서 결정론으로 비교한 뒤 selected checkpoint를
고정한다. 동일 `(scenario, seed)` 쌍은 세 arm이 모두 실행하며, 경로
phase와 물리 pair의 편향을 줄이기 위해 각 arm을 pair 0·1·2에 교차 배정한다.

### 1차 지표

- 엄격한 안전 착륙률
- Wilson confidence interval
- paired success difference
- touchdown lateral error
- touchdown vertical speed와 relative horizontal speed
- unsafe contact와 crash rate

### 관측성과 안정성 지표

- FOV loss fraction과 최대 연속 loss 시간
- reacquisition rate/time
- low-visibility descent fraction
- action saturation과 oscillation
- 배터리 reserve/energy 사용량

### 학습비용

$$
N_{\mathrm{total}}
=N_{\mathrm{BC}}+N_{\mathrm{reward\ design}}+N_{\mathrm{PPO}}+N_{\mathrm{eval}}.
$$

보고서에서는 각 항을 분리한다. 제안법의 reward-design 비행을 숨긴 채 PPO episode만
비교하지 않는다.

## 구조 ablation

R-GAT 자체의 관계 표현 효과는 같은 graph, dataset, loss를 사용하고 encoder만 바꾸어
검사한다.

- `mlp_adaptive_weight_no_se`: 관계 없는 node-wise MLP
- `gat_adaptive_weight_no_se`: relation type을 하나로 합친 GAT
- `rgat_adaptive_weight_no_se`: relation-aware R-GAT

이 구조 ablation은 핵심 3-arm 결과와 별도 표로 보고한다.
