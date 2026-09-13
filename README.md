# Ontology–R-GAT–RL 자율 착륙

NVIDIA Isaac Sim에서 도로를 따라 움직이는 UGV 위에 PX4 multicopter를 착륙시키는
vision-only recurrent PPO 시스템이다. 명시적 estimator를 사용하지 않는
ontology/R-GAT reward 방식을 포함한다.

![Meta-Sejong S5 도로 위 실제 Isaac Sim 비행](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/isaac_sim_s5_live.png)

![MATLAB 스타일 3개 파이프라인 실시간 dashboard](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/live_dashboard_status.png)

> 두 screenshot은 2026-09-12 full pipeline의 실제 runtime capture다. Simulator와
> monitoring 경로를 보여 주며 완료된 benchmark 결과는 아니다.

## 전체 pipeline 실행

저장소 루트에서 사용하는 최종 entry point는 다음 하나다.

```bash
./run.sh
```

세미나용 결과를 몇 시간 안에 우선 확보해야 하면 별도 preview를 사용한다.

```bash
./run.sh --seminar-fast
```

완료된 결과와 checkpoint는 그대로 두고 Isaac/PX4, RViz와 대시보드를 계속 띄워
상태를 확인하려면 다음처럼 실행한다. 종료는 `Ctrl-C`로 안전하게 수행한다.

```bash
./run.sh --seminar-fast --stay-open
```

이 명령은 기존 full/이전 예비 결과를 건드리지 않고
`results/seminar_fast/core3_hybrid_v3`에 저장한다. 짧은 실험에서는 모든 arm에 동일한
저분산 탐색과 감쇠형 행동복제 앵커를 적용해, PPO가 첫 성공을 보기 전에 성공 시연을
잊는 현상을 줄인다.
실제 Isaac/PX4에서 성공한 training-only PD 교사 착륙 4회를 먼저 수집하고, 압축된
공통 camera embedding/action으로 세 actor를 behavior-cloning 초기화한다. 교사는
움직이는 UGV의 속도를 feed-forward하며 시연 action label 생성에만 simulator 상대
상태를 쓴다. 학습·평가 actor 입력은 계속 camera와 UAV proprioception뿐이다. 이후
`shin_se_fixed`, `no_se_fixed`, `onto_rgat_adaptive_weight_no_se`를 각각 PPO 32회
학습한다. Adaptive 설계 자료는 성공/실패/위험 실패를 층화해 최소 12회, 부족하면
24회까지 수집하고 R-GAT을 40 epoch 학습한다. 쉬운 scenario 3종을 paired seed
3개씩 평가한다. 실제 배터리 방전 모델은
유지하지만 시작 잔량을 35--55 hover-second로 제한한다. 이는 쉬운 조건의 예비 비교이며
논문 재현 또는 통계적으로 충분한 성능 주장이 아니다.

인수 없는 명령은 마감/seminar budget의 `--mode full`을 실행한다.

- `shin_se` estimator warm-up 8회
- `shin_se`, `no_se`, `onto_no_se` 각각 PPO 비행 264회
- Estimator-free reward-design 실제 비행 최소 40회. 두 terminal class와 성공한
  loss→reacquisition→landing example이 부족하면 최대 120회까지 자동 연장
- Scenario 7종과 pipeline별 paired evaluation seed 5개, 총 평가 비행 105회

별도 reward-design/evaluation 비행 전 학습 비행은 정확히 800회다. Preview-scale
실험이며 publication-scale 근거가 아니다. 다른 실행 예시는 다음과 같다.

```bash
./run.sh --mode quick --headless
./run.sh --mode full --pipelines shin_se no_se onto_no_se
./run.sh --mode quick \
  --config Ontology_RGAT_UAV_RL_ISAAC_PX4/config/experiments/adaptive_reward_weight_comparison.yaml
./run.sh --mode full --training-replicate 1 \
  --train-episodes 40960 --rgat-data-episodes 400
./run.sh --help
```

Flight stack은 launcher 하나만 소유할 수 있다. 두 번째 `run.sh`는 vehicle을 reset하거나
result를 수정하기 전에 종료된다. 호환 checkpoint와 완료 CSV row는 자동 재개한다.
기본 실행은 보고서 생성까지 끝나면 정상 종료하며, `--stay-open`은 완료 후에도 전체
시각화 stack을 유지하고 죽은 owned process가 있으면 자동 재시작한다.

## 비교 대상

| Pipeline | State-estimation supervision | Active-perception reward | Ontology reward |
|---|---:|---:|---:|
| `shin_se` | 있음, six-state auxiliary MSE | 있음 | 없음 |
| `no_se` | 없음 | 없음 | 없음 |
| `onto_no_se` | 없음 | 없음 | 동결 direct R-GAT PBRS |

새 제안 실험은 기존 3개 arm을 삭제하지 않고 별도 설정으로 실행한다.

| 명시적 모드 | 보상 함수 | Active perception |
|---|---|---:|
| `shin_se_fixed` | 고정 Shin 5성분 | 있음 |
| `shin_se_rgat_weight` | 동결 R-GAT 상태 적응 5성분 + semantic PBRS | 있음 |
| `no_se_fixed` | 고정 Shin 5성분 | 없음 |
| `onto_rgat_adaptive_weight_no_se` | 동결 R-GAT 상태 적응 5성분 + semantic PBRS | 없음 |
| `onto_rgat_potential_pbrs_no_se` | 보존된 scalar `Phi(G)` PBRS | 없음 |

세 pipeline은 512×320 mono camera, 동결 6-keypoint encoder, 512-unit LSTM,
256-D latent, `y[6:256]` actor slice, 7-D UAV proprioception, 4-D velocity/yaw-rate
action, PX4 controller, PPO 설정, curriculum과 paired seed를 공유한다. Simulator
truth는 asymmetric critic, reset, terminal label과 physical evaluation에만 허용한다.

### 보상함수

| Reward 항 | `shin_se` baseline | `no_se` 대조군 | `onto_no_se` 제안 방식 |
|---|---|---|---|
| Terminal | 안전 착륙 `+10`, 충돌/이탈/배터리 실패 `-10`; shaping을 대체 | `shin_se`와 동일 | Sparse terminal `+10/-10`, next potential 0 |
| Physical shaping | Lateral/vertical progress, vertical-speed/undershoot/yaw-rate penalty | Active term을 제외하고 동일 | 명시적 physical shaping 없음 |
| Active perception | `-0.1 clip(L_est,t+1-0.01,0,1)` | 없음 | 없음 |
| Ontology shaping | 없음 | 없음 | `lambda [gamma Phi(G_t+1)-Phi(G_t)]` |
| 상수 | `alpha=0.1`, `beta=1`, `tau=0.01` | 해당 없음 | `lambda=1`, `gamma=0.99` |
| Reward 정보 경계 | 실제 상대 상태와 privileged estimator loss | 실제 상대 상태 | Estimator-free semantic graph와 terminal event |

적응 가중치 arm의 무가중 성분은 동일한 Table-III 순수 함수를 공유한다. Baseline
가중치 `[1,1,0.5,1,2]`와 합 5.5를 보존하면서 `w(G_t)`가 상태에 따라 바뀐다.
기존 5성분에는 시야상실/재포착 항이 없으므로 같은 동결 R-GAT encoder의 두 번째
head가 `Phi(G)`를 출력하며 `0.75[gamma Phi(G_t+1)-Phi(G_t)]`를 더한다. Terminal에서는
5성분 shaping만 대체하고 absorbing-state potential correction은 유지한다. 제안 no-SE
arm에는 estimator loss나 active-perception 항이 없다. 자세한 수식과 누출/동결 계약은
[상태 적응형 보상 가중치 문서](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md)를 참고한다.

착륙 성공은 접촉 신호 하나로 판정하지 않는다. 패드 접촉과 함께 중심 수평 오차
`<= 0.35 m`, 수직속도 `<= 0.55 m/s`, 패드 상대 수평속도 `<= 0.45 m/s`, roll/pitch
합성 tilt `<= 10 deg`, 기체 각속도 `<= 45 deg/s`를 모두 만족해야 한다. 하나라도
위반한 접촉은 `unsafe_pad_contact` 실패이고 terminal reward는 `-10`이다.

자세한 항별 수식은
[프로젝트 guide의 보상함수 비교표](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md)를 참고한다.

기본 ontology는 **18 node, directed edge 35개, relation 4종, node당 feature 24개**다.
Keypoint/heatmap semantic, UAV motion/attitude와 onboard battery reserve를 사용한다.
Relative-state estimate, UGV state, GNSS, simulator truth를 허용하지 않는다. 학습한
R-GAT output을 `Phi(G)`로 직접 동결한다.

```text
r_t = r_sparse + lambda * (gamma * Phi(G_t+1) - Phi(G_t))
```

이전 14-node/38-edge, 23-channel cooperative urban 실험과 증류한 고정 reward
weight는 명시된 legacy path로만 제공한다.

## 실행 및 모니터링

`run.sh`는 DDS(UDP 8888), Isaac Sim/Pegasus/PX4, ROS 2 gateway(UDP 14650),
RViz 2와 <http://127.0.0.1:8770/> dashboard를 시작하거나 인수한다. Dashboard는
committed episode와 active episode/per-step telemetry를 분리해 rendered flight가
오래 걸려도 정지처럼 보이지 않게 한다.

Recoverable SITL transport, simulated-clock, 순수 Offboard-heartbeat 중단은 partial
trajectory만 버리고 launcher 소유 stack을 재시작해 같은 seed를 재시도한다. Geometry,
perception, estimator, policy failure는 hard failure로 남기고 retry로 숨기지 않는다.

## Meta-Sejong S5 환경

기본 benchmark는 Gwanggaeto/S5 campus asset과 37-point 폐곡선 도로 route를 쓴다.
길이는 99.70 m다. Offline mesh audit에서 1.5×1.5 m deck footprint를 제외한 보수적
clearance 1.00 m, waypoint 최대 elevation error 0.001 m를 측정했다.

![감사된 Meta-Sejong S5 UGV route](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/images/metasejong_gwanggaeto_ugv_route.png)

## 문서

구현은 [`Ontology_RGAT_UAV_RL_ISAAC_PX4/`](Ontology_RGAT_UAV_RL_ISAAC_PX4/)에 있다.

- [전체 프로젝트 guide](Ontology_RGAT_UAV_RL_ISAAC_PX4/README.md)
- [문서 안내](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/README.md)
- [시스템 개요](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SYSTEM_OVERVIEW.md)
- [3개 파이프라인 통제 비교](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/THREE_PIPELINE_COMPARISON.md)
- [상태 적응형 보상 가중치](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md)
- [운영 및 fault 진단](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md)
- [아키텍처와 interface](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/ARCHITECTURE.md)
- [논문-코드 baseline](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/SHIN2026_BASELINE.md)
- [실제 기체 안전 gate](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/HARDWARE_SAFETY.md)

활성 프로젝트 디렉터리에서 저장소 검사를 실행한다.

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
./scripts/check_workspace.sh
```
