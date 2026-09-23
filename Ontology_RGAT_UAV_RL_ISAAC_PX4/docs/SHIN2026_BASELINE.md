# Shin et al. (2026) baseline 대응

[문서 안내](README.md) · [실험 설계](TWO_PIPELINE_COMPARISON.md) ·
[논문 대조](PAPER_FIDELITY.md) · [참고문헌](REFERENCES.md)

기준 논문은 W. Shin et al., *Vision-Based Autonomous Drone Landing on Moving
Platforms With Uncertain Motion via Deep Reinforcement Learning*, IEEE Robotics
and Automation Letters 11(5), 2026, DOI `10.1109/LRA.2026.3674011`이다.

`shin_se_fixed`는 논문의 방법론적 interface를 Isaac Sim/Pegasus/PX4에 구현한
비교군이다. 원문 값의 단일 전사본은 `config/paper/shin2026_reference.yaml`이고
`tests/test_shin2026_paper_fidelity.py`가 resolved config·보상 구현과 대조한다.
원문이 말하지 않는 항목과 백엔드의 선언된 이탈은 [논문 대조](PAPER_FIDELITY.md)에 있다.

## 1. 강화학습 구성

| 요소 | 논문 기반 구현 |
|---|---|
| Environment | 움직임이 불확실한 이동식 착륙 플랫폼과 제한된 camera FOV |
| Agent | keypoint encoder, LSTM 상태추정 layer, PPO actor |
| Observation | grayscale landing image + UAV body velocity/quaternion |
| Action | 원문 `[v_x, v_y, v_z, ω_z]` 속도 명령. **이 저장소는 평면 3채널 `[a_fwd, a_z, tilt]`로 축소**했다 ([PAPER_FIDELITY](PAPER_FIDELITY.md) §3.8, [PLANAR_ENVELOPE](PLANAR_ENVELOPE.md)) |
| Critic | UAV proprioception + 실제 상대상태를 쓰는 asymmetric critic |
| Reward | 5개 touchdown shaping + active-perception penalty + terminal ±10 |
| Curriculum | 플랫폼 운동 난이도를 단계적으로 증가 |

## 2. Keypoint perception

6개 keypoint의 location/heatmap descriptor를 사용한다. Classical pose estimator처럼
충분한 점이 보일 때만 pose를 계산하지 않고, 부분 가시 상태도 LSTM에 직접 입력한다.
Keypoint encoder는 PPO 전에 학습·검증하고 모든 arm에서 동일하게 동결한다.

현재 저장소가 사용하는 encoder는 논문 저자의 비공개 weight를 복제한 것이 아니라,
동일한 6-keypoint interface를 이 프로젝트의 표적과 Isaac camera에 맞게 학습한
모델이다.

## 3. LSTM 상대상태 보조학습

LSTM hidden, image embedding, UAV state에서 256-D latent `y_t`를 만든다. 처음 6개
성분은 패드의 body-frame 상대 위치·속도를 회귀한다.

$$
\tilde s_t^{\mathrm{rel}}=[\Delta\tilde x_t^b,\;\Delta\tilde v_t^b]\in\mathbb R^6,
\qquad
L_t^{\mathrm{est}}=\frac{1}{6}\sum_{i=1}^{6}\bigl(s_{t,i}^{\mathrm{rel}}-\tilde s_{t,i}^{\mathrm{rel}}\bigr)^2 .
$$

Actor는 논문 Fig. 4와 같이 `y_{t,6:256}`과 UAV state를 사용한다. 추정한 6개 값은
보조학습과 active reward 계산에 쓰이고 actor에 직접 concatenate하지 않는다.

## 4. 보상의 정확한 형태

상대상태 `s^rel = [Δx, Δy, Δz, Δv_x, Δv_y, Δv_z]`(플랫폼 − UAV, UAV body frame),
행동 `a`(원문 `[v_x, v_y, v_z, ω_z]`, 이 저장소 `[a_fwd, a_z, tilt]`),
UAV body 수직속도 `v_z^{UAV}`에 대해

| i | 항 | 식 | `w_i^0` |
|---|---|---|---|
| 1 | lateral progress | `clip(‖Δp_xy(t)‖ − ‖Δp_xy(t+1)‖, −1, 1)` | 1.0 |
| 2 | vertical progress | `clip(|Δz(t)| − |Δz(t+1)|, −1, 1) / max(‖Δp_xy(t+1)‖, 1)` | 1.0 |
| 3 | vertical-speed penalty | `−max(v_z^{UAV}(t+1) + 0.5, 0)` | 0.5 |
| 4 | undershoot penalty | `−Δz(t+1)` if `Δz(t+1) > 0` else `0` | 1.0 |
| 5 | attitude penalty | 원문 `−|ω_z^{cmd}|`. 이 저장소는 요 채널이 없어 같은 가중치로 `−|tilt^{cmd}|` ([PAPER_FIDELITY](PAPER_FIDELITY.md) §3.9) | 2.0 |

$$
r_t^{\mathrm{active}}=-0.1\,\mathrm{clip}\bigl(1.0\cdot(L_{t+1}^{\mathrm{est}}-0.01),\,0,\,1\bigr)
$$

현재 행동 뒤 추정오차가 증가하면 penalty를 주므로 policy가 패드를 계속 관측하기
좋은 궤적을 학습하도록 유도한다. 전체 baseline 보상은 구간별이며 **종료 보상이
shaping을 대체**한다.

$$
r_t=
\begin{cases}
+10, & \text{엄격 안전착륙}\\
-10, & \text{충돌 · 과도이탈 · 배터리 고갈}\\
\sum_i w_i^0\rho_i(t)+r_t^{\mathrm{active}}, & \text{그 외}
\end{cases}
$$

구현은 `reward_modes/shin2026.py`이고, Table III에서 독립적으로 다시 쓴 oracle과
항별로 일치하는지 test가 확인한다.

## 5. 저장소의 공통 simulator adaptation

| 항목 | 구현 |
|---|---|
| Dynamics | Pegasus multicopter + PX4 SITL control loop |
| Low-level control | PPO velocity/yaw-rate → PX4 OFFBOARD → PX4 attitude/rate control |
| Map | Meta-Sejong S5 road environment |
| Platform | 도로를 따라 주행하는 RANGER MINI landing deck |
| Marker | 원거리·전이·touchdown용 다중 크기 6-keypoint layout |
| Reset | UAV hover, UGV parked, entry position/speed/visibility gate 후 시작 |
| Battery | 3S 3500 mAh 에너지 적분과 reserve semantic |
| Success | 접촉 + 위치 + 수직/상대속도 + tilt + 각속도 |

이 adaptation은 두 arm에 동일하게 적용되므로 비교의 교란요인이 아니지만, 원문
재현 주장에는 제약이다. 자세한 목록은 [논문 대조](PAPER_FIDELITY.md) §3에 있다.

## 6. 제안법과의 관계

`shin_se_onto_rgat_state`(현재 제안법)는 위의 보조 상태추정, active-perception
reward, 다섯 shaping 항과 가중치를 **보상 쪽에서 아무것도 바꾸지 않고** 그대로
사용한다. 유일한 추가는 **관측**이다: 영상 관측성 cue의 9-node 온톨로지 상황
그래프를 R-GAT으로 부호화해 actor와 critic 입력에 이어붙인다.

$$
r_{\text{proposed}}(t)=r_{\text{Shin}}(t),\qquad
o_{\text{proposed}}(t)=o_{\text{Shin}}(t)\;\Vert\;g_t .
$$

은퇴한 `shin_se_onto_rgat_recovery`는 대신 보상에 항을 더했다. 그 arm은 자기
id로 남아 있고 지금도 실행할 수 있다.

$$
r_{\text{retired}}(t)=r_{\text{Shin}}(t)-\lambda_{\text{fov}}\,q_\theta(G_t).
$$
