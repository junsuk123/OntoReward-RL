# Shin et al. (2026) baseline 대응

[문서 안내](README.md) · [비교 설계](THREE_PIPELINE_COMPARISON.md) ·
[참고문헌](REFERENCES.md)

기준 논문은 W. Shin et al., *Vision-Based Autonomous Drone Landing on Moving
Platforms With Uncertain Motion via Deep Reinforcement Learning*, IEEE Robotics
and Automation Letters 11(5), 2026, DOI `10.1109/LRA.2026.3674011`이다.

이 저장소의 `shin_se_fixed`는 논문의 방법론적 interface를 Isaac Sim/Pegasus/PX4에
구현한 비교군이다.

## 강화학습 구성

| 요소 | 논문 기반 구현 |
|---|---|
| Environment | 움직임이 불확실한 이동식 착륙 플랫폼과 제한된 camera FOV |
| Agent | keypoint encoder, LSTM state-estimation layer, PPO actor |
| Observation | grayscale landing image + UAV body velocity/quaternion |
| Action | $[v_x,v_y,v_z,\omega_z]$ velocity command |
| Critic | UAV proprioception + 실제 상대상태를 쓰는 asymmetric critic |
| Reward | 5개 touchdown shaping + active-perception penalty + terminal $\pm10$ |
| Curriculum | 플랫폼 운동 난이도를 단계적으로 증가 |

## Keypoint perception

6개 keypoint의 location/heatmap descriptor를 사용한다. Classical pose estimator처럼
충분한 점이 보일 때만 pose를 계산하지 않고, 부분 가시 상태도 LSTM에 직접 입력한다.
Keypoint encoder는 PPO 전에 학습·검증하고 모든 arm에서 동일하게 동결한다.

현재 저장소가 사용하는 encoder는 논문 저자의 비공개 weight를 복제한 것이 아니라,
동일한 6-keypoint interface를 프로젝트의 marker와 Isaac camera에 맞게 학습한 모델이다.

## LSTM 상대상태 보조학습

LSTM hidden, image embedding, UAV state에서 256-D latent $y_t$를 만든다. 처음 6개
성분은 패드의 body-frame 상대 위치·속도를 회귀한다.

$$
\tilde s_t^{\mathrm{rel}}
=[\Delta\tilde x_t^b,\Delta\tilde v_t^b]\in\mathbb R^6,
$$

$$
L_t^{\mathrm{est}}
=\frac{1}{6}\sum_{i=1}^{6}
\left(s_{t,i}^{\mathrm{rel}}-\tilde s_{t,i}^{\mathrm{rel}}\right)^2.
$$

Actor는 논문 Fig. 4와 같이 $y_{t,6:256}$과 UAV state를 사용한다. 추정한 6개 값은
보조학습과 active reward 계산에 사용되고 actor에 직접 concatenate하지 않는다.

## Active-perception reward

$$
r_t^{\mathrm{active}}
=-0.1\,\mathrm{clip}
\left(L_{t+1}^{\mathrm{est}}-0.01,0,1\right).
$$

현재 행동 뒤 추정오차가 증가하면 penalty를 주므로, policy가 패드를 계속 관측하기 좋은
궤적을 학습하도록 유도한다.

## Touchdown shaping

고정 weight는

$$
w^0=(1,1,0.5,1,2)
$$

이며 lateral progress, vertical progress, vertical-speed penalty, undershoot penalty,
yaw-rate penalty에 각각 적용한다. 전체 baseline reward는

$$
r_t=
\begin{cases}
+10, & \text{안전 착륙},\\
-10, & \text{충돌·이탈·고갈·timeout},\\
(w^0)^\top\bar\rho_t+r_t^{\mathrm{active}}, & \text{그 외}.
\end{cases}
$$

이다.

## 저장소의 공통 simulator adaptation

| 항목 | 구현 |
|---|---|
| Dynamics | Pegasus multicopter + PX4 SITL control loop |
| Low-level control | PPO velocity/yaw-rate → PX4 OFFBOARD → PX4 attitude/rate control |
| Map | Meta-Sejong S5 road environment |
| Platform | Road-following RANGER MINI landing deck |
| Marker | 원거리·전이·touchdown용 다중 크기 6-keypoint layout |
| Reset | UAV hover, UGV parked, entry position/speed/visibility gate 후 시작 |
| Battery | 3S 3500 mAh energy integration과 reserve semantic |
| Success | 접촉 + 위치 + 수직/상대속도 + tilt + 각속도 |

## 제안법과의 관계

`shin_se_onto_rgat_fov`는 위의 보조 상태추정, active-perception reward,
다섯 shaping 항과 가중치를 모두 그대로 사용한다. 유일한 추가는 영상 관측성
cue의 ontology graph와 동결 R-GAT 미래 FOV-loss 확률 보상이다.

$$
r_{proposed}=r_{Shin}-\lambda_{fov}p_{fov__loss}.
$$

이다.
