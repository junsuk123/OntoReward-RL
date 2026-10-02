# Selective R-GAT v1

## 연구 질문

동일한 causal observation, 동일한 2항 reward, 동일한 action/environment/PPO budget에서
고정된 ontology R-GAT 위의 상황별 관계 선택만이 vector-state PPO보다 제어 성능과
강건성을 개선하는지 검증한다.

## 세 arm

1. `pn_guidance_v1`: 비학습 control condition
2. `ppo_vector_state`: canonical `O_t`를 직접 MLP actor/critic state로 사용
3. `ppo_ontology_selective_rgat`: 같은 `O_t`로 `G_t`를 만들고 고정 R-GAT과 선택 gate의
   `g_t`를 actor/critic state로 사용

두 PPO arm의 reward/action/environment/termination/curriculum/PPO hyperparameter/seed/
budget/evaluation cadence는 같아야 한다. behavior cloning은 기본 경로에서 끈다.

## 관계와 gate

Adaptive relation은 정확히 다음 네 개다.

- `Tracking_evidence`
- `VisualRetention_evidence`
- `ContactKinematics_evidence`
- `StateReliability_evidence`

`concept_to_landing_goal`과 `self`는 invariant이며 gate가 항상 1이다.

`alpha_t^r = 1 + 0.35 tanh(W_r normalize(O_t) + b_r)`

`W_r=0`, `b_r=0`으로 초기화하므로 첫 forward는 frozen R-GAT과 정확히 같다. adaptive
gate 범위는 `[0.65, 1.35]`이다. PPO optimizer가 소유할 수 있는 encoder parameter는
`gate.linear.weight`, `gate.linear.bias`뿐이다. R-GAT `W/a/E`, topology, normalization,
readout에는 gradient와 optimizer state가 생기지 않는다.

## 오프라인 artifact

PN 및 vector-PPO rollout을 episode 단위 train/validation/test로 분리한다. normalization은
train episode만으로 계산한다. target은 현재 visibility risk, observable motion trend,
capture/distance signal의 현재 부호, measurement reliability이며 future label은 쓰지 않는다.
artifact는 schema/topology/observation/normalization/dataset/split/seed/parameter hash와
train/validation trajectory, effective rank, control-sufficiency 판정을 포함해야 한다.
control-sufficiency를 통과하지 못한 artifact는 PPO에 로드할 수 없다.

## reward와 terminal

`r_t = 0.20 r_capture(t) + 0.71 r_distance(t)`

capture는 optical-axis에서 +1, FOV boundary에서 -0.5, FOV 밖에서 clipping 없이 -1에
점근한다. distance는 30% approach-rate와 70% proximity의 혼합이다. safe terminal은
두 신호 모두 +1, failed contact는 모두 -1이다. terminal 이후 전이는 저장하지 않으며
필요한 finite-horizon absorbing tail은 마지막 valid transition에만 반영한다.

## checkpoint와 평가

checkpoint signature는 algorithm/environment/observation/reward/action/graph/pretrained/
relation-partition/normalization/seed/budget hash 전부가 맞아야 한다. 선택은 held-out
deterministic boundary cases에서만 수행하고 PPO gradient와 분리한다. `robust_checkpoint_score`
는 robust/worst/nominal landing, capture, unsafe contact와 trajectory-quality를 사용하고
return은 작은 tie-breaker로만 사용한다.

평가는 paired seed, 동일 scenario/init/UGV route-speed/disturbance/camera/horizon을 쓴다.
보고값은 strict success, unsafe contact, touchdown, FOV retention/loss/reacquisition,
peak altitude, loss 이후 climb, longitudinal RMS, command total variation, return/capture,
latency, parameter/FLOP/wall-clock 및 paired hierarchical-bootstrap 95% CI다.
