# 학습 개선 감사 기록(항목 1, 2, 4, 5, 8, 9)

이 문서는 2026년 9월 실제 실행 이후 순차적으로 요청된 변경을 기록한다.
실행 가능한 근거와 Shin et al. 논문을 구분한다. 논문은 알고리즘을 정의하며,
아래 수치는 이 저장소의 Isaac/Pegasus/PX4 구현에서 얻은 값이지 논문 결과가 아니다.

## 개선 전 관측

중단된 checkpoint는 보존했다. Policy episode 9–28과 49–68을 비교하면 position
RMSE는 2.89 m에서 3.37 m, FOV loss는 58.1%에서 68.6%, 최종 lateral error는
6.31 m에서 9.23 m로 증가했고 착륙은 0회였다. 표본의 모든 active-perception
항은 `-0.1`로 clipping됐다. 따라서 같은 checkpoint를 계속 쓰기보다 구조를
고치는 것이 타당했다.

## 순차 개선 근거

| 항목 | 결함과 변경 | 검증/효과 |
|---:|---|---|
| 1 | 전역 `tanh`가 물리 상대 상태를 ±1로 제한했다. 첫 latent channel 6개를 제한 없는 정규화 좌표로 바꾸고 m 및 m/s로 decode하며, loss scale은 `[3,3,8,3,3,2]`를 사용한다. | 결정론적 test에서 이전 model이 표현하지 못한 6 m와 12 m를 출력한다. 모든 축에서 scale 하나만큼의 오차는 균형 loss 1.0이다. Checkpoint format v3가 이전 의미 체계를 거부한다. |
| 2 | 원본 6-state MSE가 active reward를 포화시켜 인과적 gradient를 없앴다. 실제 reward는 auxiliary estimator와 같은 정규화 loss를 사용한다. | 대표 오차의 보상이 clipping된 `-0.1000`에서 반응 가능한 `-0.0135`로 바뀐다. Episode마다 평균 normalized loss, saturation 비율과 signal 표준편차를 저장한다. |
| 4 | 마감용 override가 성능과 무관하게 264 episode 동안 80 curriculum level을 약 3 episode마다 올렸다. | 40회 연속 실패 fixture는 level 1에 머문다. Rolling success, FOV, 그리고 모든 arm에 동일한 truth-side relative-position RMSE gate가 통과해야 상승한다. |
| 5 | 동결 encoder가 합성 board projection만 사용했다. Launcher가 annotation 없는 실제 Isaac frame을 수집하고, 검출 board-plane homography로 6개 pad landmark label을 만든 뒤 held-out split을 남겨 fine-tuning하고 최상 validation state만 보존한다. | 최초 실제 감사에서 23.2 px RMSE와 PCK@20 29.2%였고 calibration 후 같은 held-out split에서 16.8 px와 79.2%를 얻었다(RMSE 27.6% 감소). Full mode는 현 설정으로 48 frame을 다시 수집한다. |
| 8 | Table-II sample이 unit test에만 있었다. 이제 episode seed가 PX4-relative controller gain 분산, Isaac force/torque, handover 시 1회 velocity/rate perturbation, 실제 actor camera의 texture/scale/brightness/RGB/light 변환에 적용된다. | 범위, 결정론적 직렬화와 PX4 mapping test를 통과한다. 중복 적용을 막기 위해 상속된 고정 urban wind는 이 benchmark에서 끈다. 적용값은 reset acknowledgement와 episode CSV에 기록된다. |
| 9 | 이전 health gate는 success가 없고 FOV loss도 높은 두 조건의 conjunction만 감지했다. | 독립 gate가 20 episode 이후 battery terminal 비율을, 40 episode grace 이후 no success, 높은 FOV, 정체된 position RMSE와 active-reward saturation을 감지한다. Fixture가 각 원인을 구별한다. |

## 실제 통합 확인

`results/remediation/stage8_9`에서 새로운 1-episode quick run을 실행하고 첫 atomic
checkpoint 뒤 중단했다. Benchmark 근거가 아니라 calibration 근거다.

- held-out Isaac keypoint RMSE: 24.46 px → 18.41 px
- PCK@20: 20.8% → 75.0%
- normalized 6-state loss 평균: 0.593
- active-reward saturation: 0%, 표준편차 0.0216(중단된 baseline은 변동 없이 100% 포화)
- 실패한 warm-up은 curriculum level 1 유지(`advanced=0`)
- reset acknowledgement: domain randomization 활성, force 0.945 N,
  torque 0.00460 N·m, texture 6, brightness 0.759

Warm-up 1회만으로 landing rate나 RMSE convergence를 입증할 수 없다. 해당 주장은
완료된 실행이 필요하며 새 health check가 계속 보호한다.

## 전체 실행 기동 확인

Commit `3e1f4ff`를 `main`에 push한 뒤 `./run.sh --mode full`을 시작했다. Launcher는
호환되지 않는 v2 checkpoint를 보관하고 dashboard와 RViz를 열어 새 v3 run을
시작했다. 48-frame empirical calibration에서 held-out keypoint RMSE가 76.66 px에서
9.83 px, PCK@20이 0.0%에서 98.6%로 개선됐다. 첫 atomic full-run episode는
normalized loss 0.308, active-reward saturation 0%, 0이 아닌 active signal 변동
0.00947과 Table-II randomization을 기록했고, 실패했으므로 curriculum은 오르지 않았다.

## 해석

이 검사는 이전 구조적 blocker가 제거되었음을 보이지만 landing success rate를 미리
주장하지 않는다. 유효한 다음 근거는 `./run.sh --mode full`로 새 v3 checkpoint를
학습한 결과다. Dashboard는 normalized estimation loss와 active-reward saturation을
표시한다. Health-gate 중단은 호환되지 않는 checkpoint를 재사용할 이유가 아니라
유용한 실패 실험 결과다.

## 관측성/재획득 후속 개선

첫 개선 full run도 PPO 38 episode에서 착륙 0회였고, 마지막 20 episode 평균 FOV
loss 59.0%, 실제 relative-position RMSE 3.69 m였다. 실행 계약을 바꾸기 전에 run을
중단하고, 같은 checkpoint를 계속 쓰는 대신 독립적으로 시험 가능한 장치를 추가했다.

- `c=1` reward-design/evaluation 비행을 포함한 모든 curriculum에서 handover 시
  platform이 검출되어야 한다.
- Keypoint network에 명시적으로 supervised visibility head를 추가하고
  target-absent negative frame으로 초기화한다.
- Low-confidence 좌표는 alignment, scale 또는 motion에 기여할 수 없다.
- Estimator-free visibility memory, reacquisition trend와 loss-duration risk로
  semantic graph가 유한 history를 반영한다.
- Reward-design 수집은 두 terminal class뿐 아니라 성공한
  loss→reacquisition→landing trajectory를 요구한다.
- Perception, recovery, battery를 불리하게 만든 counterfactual이 예상 방향의
  potential을 만드는지 학습·감사한 뒤에만 R-GAT을 동결한다.
- Episode/evaluation record에 reacquisition rate/time, recovery climb,
  low-confidence unsafe descent, recovery landing과 loss/reacquisition 전환 시
  potential 변화를 기록한다.

Graph format은 `ontology_rgat.semantic_graph/2-history-aware`이므로 이전 semantic
dataset과 potential은 의도적으로 호환되지 않는다. 동일 gamma, terminal-zero
potential과 동결 R-GAT은 MDP PBRS 계약을 지킨다. 유한 history를 완전한 Bayesian
belief state라고 제시하지 않으므로 정확한 POMDP policy invariance를 주장하지 않으며,
recovery는 경험적으로 평가한다.

새 shared-descriptor visibility head의 독립 합성 128-frame smoke set에서는 visibility
accuracy 94.5%, visible landmark true positive 97.1%, target-absent 21 frame의 false
positive 0%를 측정했다. 이는 absent-target decision boundary만 검사한다. 실제 renderer
domain acceptance test는 `run.sh`에서 수행하는 필수 held-out Isaac calibration이다.

첫 history-aware full launch는 필수 Isaac frame 48개를 수집했다. 12-frame held-out
split에서 fine-tuning 후 coordinate RMSE는 76.46 px에서 13.58 px, PCK@20은 0%에서
97.2%로 바뀌었고 visibility accuracy 100%, 합성 target-absent false-positive 0%였다.
Episode 1은 모든 curriculum에 적용되는 initial-FOV gate를 통과하고 atomic 완료됐으며,
visual-loss event 41회 중 40회 재획득(97.6%), low-confidence unsafe descent 15.9%,
active-reward saturation 3.3%를 기록했다.

이 run은 최종적으로 episode 47까지 checkpoint를 저장했지만 최근 policy episode
20회에서 착륙 성공이 0회여서 training health gate가 중단했다. 따라서 800-episode
benchmark 완료 결과가 아니라 재현 가능한 실패 근거이며, 성공률 주장에 사용하지 않는다.
