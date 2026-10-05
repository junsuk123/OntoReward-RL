# Ontology-RGAT UAV landing workspace

현재 상태: **2D v2.8 핵심 알고리즘 이식·검증 중, Isaac 동등성/학습 성능 미검증**.
기준 저장소는 [ugv_landing_2d_workspace](https://github.com/junsuk123/ugv_landing_2d_workspace/tree/83f10d6c5d90fc9350d67af1f729f83fc49009f4)
알고리즘 대조 커밋은 `83f10d6c5d90fc9350d67af1f729f83fc49009f4`다.
원격 HEAD `1bc69db`(2026-10-04 재확인)는 온톨로지 시각화/문서만 추가했으며
그 사이 PPO·보상·시뮬레이션·설정 소스 변경은 없다.
새 `spatial` 경로는 실제 공간 3축 가속도, 인과적 관측·정책, 로컬/Isaac PPO 학습,
체크포인트 재로딩과 실제 Isaac/PX4 평가까지 연결됐다. 이전 외란 정합 v8 후보의
실제 학습→재평가 실행은 완료했고 15회 비행 모두 종료·disarm을 확인했다.
새 실제 시드의 세 모델은 모두 SAFE_ABORT, 학습 착륙 0회로 **미통과**다.
앞선 실제 6회 평가의 unsafe 접촉 1회도 별도 기록으로 보존한다.
시간 종료 비행·안전 종료를 학습 착륙 성공으로 보고하지 않는다.
v8 완료 전체 회귀: **1,272 passed / 2 skipped** (360.31초). 종료 hold의 자동 해제,
진단 전용 체크포인트 및 고정 정책 평가 시드 옵션을 포함한다. v5 실제 70초 3-arm 평가는 모두 TASK_TIMEOUT이며
`isaac_acceptance.json`은 **passes=false**다. 외란을 유지하는 v8 후보의 실제
분리 진단은 SUCCESS / UNAUTHORIZED_CONTACT였으며, PPO 착륙 성능이 아니다.

2026-10-05 추가 승인 구간에서 opt-in v9에 x/y 각각 reference
9×12 문맥, 공동 3D 하강 evidence, 촬영 시각과 own EKF의 정합, 로컬 영상 지연 및
동일 시드 CV–CA–CV 궤적을 반영했다. v9 부분 학습 결과는 보존하고, 외란을 유지한
v10의 2개 독립 학습 seed × 세 arm의 로컬 학습을 완료했다. 기본 v5를 통과 모델로 대체하지 않았다.
별도 v10은 uncertainty trust/abort latch/응답지연 제동/terminal corridor 및 외란 중
자체 위치 backup을 추가했다. measured optical view reward를 사용하며 v9와 별개로
재학습했다. 최신 전체 회귀 **1,358 passed / 2 skipped**; actual causal 진단 seed 14007은
SUCCESS와 confirmed cleanup이었지만 **PPO 학습 착륙 증거는 아니다**.
후속 진단에서 느린 lockstep에 대한 wall-time 명령 과다 발행을 찾아 simulator-time
최대 50Hz로 수정했다. 외란/감독기를 바꾸지 않은 두 optical-loss 진단은 terminal hold와
cleanup을 확인했다. 최신 전체 회귀에 이 변경이 포함된다. 수정 후 비학습 nominal 진단도
25.148초에 착륙하고 8.177초에 cleanup을 확인했다. v10 고정 로컬 시험 40회/arm의
착륙 vector/flat/graph는 **1/0/0**, unsafe는 **2/3/1**이다. 두 graph의 관계 출력은
비영이지만 성능 통과가 아니다. actual PPO 학습·저장 6개와 로컬 재로딩 12회도
완료했다(착륙0/unsafe0). 첫 실제 실행은 21회 정리 확인 뒤 pre-policy 식별 정보 거절로
중단됐고, 완료 5개를 그대로 보존하여 미완료 graph829만 명시적으로 재실행했다.
현재 실제 고정 시험을 진행 중이며 whole-system acceptance를 통과한 것은 아니다.

## 실행 경로

저장소 루트에서 실행한다. 인자 없는 실행은 도움말만 출력한다.

```bash
./run.sh                                      # 도움말, 프로세스 변경 없음
./run.sh status                               # 읽기 전용 프로세스·산출물 점검
./run.sh reference-smoke --steps 32 --ppo-minibatch
```

| 경로 | 설정 / 역할 | 비교 시 주의 |
|---|---|---|
| `reference` | `two_axis_reference_v28_active.yaml`, Python 경량 물리 | 관계 전용 PPO·validation 성능 가드 포함, 아래 잔여 차이 확인 |
| `spatial` | `spatial-isaac-system.yaml`, 별도 35-field/3-action 계약 | 기본 로컬 학습, `--training-backend isaac` 명시 시 실제 PX4 학습; reference 성능 동등성 아님 |
| 기존 `two_axis` v1 | `two_axis_context_rgat_comparison.yaml` 명시 | Claude v2.5 수정·실험 재검토용, 새 체크포인트와 호환되지 않음 |
| `isaac-legacy` | `planar_three_arm_comparison.yaml` 등 기존 설정 | Isaac/PX4 통합 경로, reference와 다른 알고리즘 계약 |

장시간 학습은 별도 예산 승인이 필요하다. 2026-10-04에는 사용자가 2시간을 승인하여
추가 학습·실제 Isaac 검증을 수행했다. 착륙 acceptance는 충족하지 못했다.
후속 4시간 예산 안에서 수정 후 정상 70초 비행 2회와 종료를 재검증했다.
추가 direct-acceleration/ABG/외란·문맥·감독기 정합 후보 v6–v10은 검증 중이며,
반복 reset 전반의 안정성 및 학습 착륙 acceptance는 아직 미충족이다.
현재 공간 기본 계약은 `spatial-causal-rgat/5`이며
이전 버전의 체크포인트를 그대로 재라벨링하지 않는다.

```bash
# 예: 사용자가 학습 예산을 승인한 후, 새 출력 디렉터리로 실행
./run.sh reference --stage train --seeds 1 2 3 --iterations 2000 \
  --workers 3 --threads 1 --output /path/to/new-run
./run.sh reference --stage evaluate --seeds 1 2 3 --output /path/to/new-run
./run.sh reference --stage aggregate --seeds 1 2 3 --output /path/to/new-run

# 기존 Isaac 배관의 명시적 실행. reference 정책 이식으로 간주하면 안 됨.
./run.sh isaac-legacy --mode quick \
  --config config/experiments/planar_isaac_smoke_no_bc.yaml
```

기존 비행 스택은 기본적으로 종료하지 않는다. 종료·재시작이 필요한 경우에만
`isaac-legacy --takeover`를 명시한다. `reference`는 Isaac/PX4를 시작하지 않는다.

## 알고리즘과 검증 범위

세 PPO arm은 두 개의 가속도 명령, 같은 인과적 26-field packet, 환경·보상·감독기를 사용한다.
pitch는 유도되는 물리 상태다. 새 `SpatialAccelerationController`는 월드 ENU
`[ax,ay,az]`를 사용하며 roll/pitch는 유도하고 yaw는 고정한다. 기존 Isaac의
`[a_fwd,a_z,tilt]`와 메시지·계약을 분리했다. 공간 backend는 35-field 관측과
9×12 그래프를 공유하며 actor/critic 모두 같은 인과적 정보만 받는다.
opt-in v6은 39-field, v7/v8은 43-field로 버전·서명을 구분한다.
v9/v10은 47-field와 축별 두 9×12 문맥을 사용한다.
v8은 사용자의 선택대로 Isaac의 외력·토크·인계 충격을 로컬에도 적용한다.

- 2D Vector / Semantic-flat / R-GAT 파라미터 수: **7,445 / 15,317 / 17,001**.
- v2.8 그래프: 9 nodes × 12 features, 5 relations, 26 edges, 4 grouped readouts.
- Raw semantic MLP + relational residual, 동일 초기 raw 경로, 조건부 하강 보정.
- Train-only masked reconstruction, staged graph adaptation, 보수적 체크포인트 선택.
- 비활성 selected graph는 raw 정책을 고정한 관계 전용 PPO 후 validation 가드로
  scale을 선택한다. 모두 탈락하면 원본 유지; test는 선택에 사용하지 않는다.
- 100Hz 관측/추정, 10Hz 정책, 시간별 공통 센서 잡음, 0.5초 인과 예측.
- 접촉 보간, terminal potential=0, dt-aware discount/GAE, nominal 평가 강제.

이는 **단일 표현 요인만 바꾼 실험이 아니다**. 그래프 사전학습, 후반부 적응, 선택 margin도
변경 요인으로 기록한다. 관계형 출력이 0이면 attention이 존재해도 관계 사용의 증거가 아니다.
최신 2D 저장소는 **비영 관계 경로가 활성화**됐다. 선택 scale `0.001`, test 평균
절대 residual `[1.27e-5,3.86e-5]`; 성공률 72% 유지, 평균 return 17.930이다.
관계 활성화는 확인됐지만 우월성은 미입증이다. 이전 `Wg=0` 최종 판정은 정정한다.
Python에서도 가드 구현을 이식했지만, 학습된 최종 성능 재현을 주장하지 않는다.

## 문서 / 보존 정책

- [검토 결과·누락·잔여 차이·검증 계획](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/refactor/REFERENCE_V28_AUDIT_KO.md)
- [운영 안내](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/OPERATIONS.md)
- [기존 README 보존본](Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/refactor/HISTORY_PRE_V28_README.md)

실행 코드는 `Ontology_RGAT_UAV_RL_ISAAC_PX4/python`, Isaac 장면은 `isaac_sim`,
ROS gateway는 `ros2_ws`, 프로파일은 `config/experiments`, 테스트는 `tests`에 있다.
과거 결과·체크포인트·`past/`·`past.zip`은 삭제하거나 덮어쓰지 않는다.
기존 설명서의 `./run.sh --...` 명령은 이제 `./run.sh isaac-legacy --...`로 실행한다.

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python -m pytest -q tests/
```
