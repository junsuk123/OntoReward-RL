# 리팩토링 동반 문서 (REFRACTOR_GUIDE_KO)

Spec: `ontoreward.paper_fidelity_and_fov_recovery/2`
기준 커밋: `7ae74b4aad3e1abda8854b70fac59fa32bce1f04`
원문: Shin et al., IEEE RA-L 11(5), May 2026, pp. 5542-5549,
DOI `10.1109/LRA.2026.3674011`. 전사본은
`config/paper/shin2026_reference.yaml` 하나뿐이며 코드·설정은 그 파일과 대조된다.

이 문서는 실행된 것과 실행되지 않은 것을 구분해서 적는다. 성능 수치는 실제 로그가
있을 때만 들어간다. 현재 성능 수치는 없다.

## 1. 두 파이프라인

| 역할 | ID | 상태 |
| --- | --- | --- |
| baseline | `shin_se_fixed` | 코드 존재, **성능 재현 미실행** |
| proposed | `shin_se_onto_rgat_recovery` | 코드 존재, **학습 미실행** |

`shin_se_onto_rgat_fov`는 은퇴한 ID다. goal embedding 위의 `nn.Linear` head로
"H 안에 FOV를 잃는가"라는 이진 라벨을 학습하던 이전 방법이며, 기존 결과의 귀속을
위해 `LEGACY_PIPELINES`에 원래 이름으로 남아 있다. 현재 방법으로 alias되지 않고,
`validate_pipeline_configuration`이 실행을 거부한다.

## 2. 이번 리팩토링에서 실제로 바뀐 것 (P3 + 코드 지적사항)

전부 `pytest tests/` 493 pass로 확인했다. 학습·평가는 돌리지 않았다.

1. **보상 target의 정의 변경.** 이진 "H 안에 FOV 상실"에서
   `y_t = sum(1 - V_centre[t+k] for k=1..H) / H`, 즉 향후 H step 중 패드 중심이 FOV
   밖인 *시간 비율*로 바꿨다. 출력은 `E[y_t | G]`이며 binary loss 확률이 아니다.
   (`rgat/fov_risk_dataset.py`)
2. **출력 head 제거.** 두 번째 relational layer를 `units=1`로 두고 그 layer의
   `FutureFOVUnavailability` node를 곧 출력으로 쓴다(sigmoid). `FOVRiskModel`은
   `nn.Linear` module을 하나도 만들지 않으며 24→1 residual도 없다. PPO actor/critic
   MLP와 R-GAT 내부 관계 변환은 건드리지 않았다. (`rgat/fov_risk_model.py`)
3. **손실 변경.** BCE → Huber regression + contract rule `R-04`(measurement age
   단조성 hinge, 기본 가중치 0.1). 지표도 분류 지표에서 MAE/RMSE/bias/R²로 바꾸고
   **상수 예측기 RMSE 기준선**을 함께 기록하게 했다. (`rgat/fov_risk_train.py`)
4. **미래창 tail 처리.** H step 미래가 관측되지 않은 sample은 `valid=False`,
   target `NaN`으로 mask한다. 0으로 채우거나 episode를 이어 붙이지 않는다. 검증
   함수가 zero-filled tail을 거부한다. (`rgat/fov_risk_dataset.py`)
5. **`ApparentTargetScale` 도달성.** 이전 graph에서 이 node는 self-loop 외에 어떤
   edge도 없어 출력에 도달하지 못했다. `ApparentTargetScale → BoundarySafety`를
   추가했고, `unreachable_input_nodes()`가 비어 있음을 test로 강제한다.
6. **missing과 높은 margin의 구분.** `fov_margin`은 *마지막으로 신뢰 가능했던*
   centroid의 margin이라 측정이 끊긴 직후 1.0에 가깝게 읽힌다. `MeasurementValidity`
   와 `MeasurementAge` node를 추가하고 `BoundarySafety` seed를
   `fov_margin * measurement_validity`로 gate했다. 두 feature 모두 같은 keypoint
   이력에서 나오므로 privileged 입력이 아니다.
7. **온톨로지 contract 파일 신설.** `config/ontology/fov_recovery.yaml`. 규칙마다
   `rule_id`/`condition`/`rationale`/`input_provenance`/`exceptions`/
   `loss_or_validation`을 갖는다. `loss_or_validation`이 `enforced`로 시작하는 규칙만
   실제로 손실이나 test로 강제되며, 나머지는 `declared only`로 표시했다. 관계
   이름이 강제되는 단조성 제약이 아니라는 점도 구조화된 필드로 적었다.

Graph version은 `ontology_rgat.future_fov_unavailability/2`, dataset format은
`ontology_rgat.future_fov_unavailability/2`, model format은
`ontology_rgat.future_fov_unavailability_readout/2`로 올렸다. 이전 artifact는
로드되지 않는다. 재수집·재학습이 필요하다.

## 3. 단계별 상태

| Phase | 산출물 | 상태 |
| --- | --- | --- |
| P0 | 원문 근거/기준 커밋/resolved config/미확인 항목 | **완료** — `config/paper/shin2026_reference.yaml`, §4, §4-1, §4-2 |
| P1 | 원문 환경과 baseline 수식/인터페이스 test | **완료** — `tests/test_shin2026_paper_fidelity.py` 18건 |
| P2 | baseline 학습 및 원문 성능 재현 보고 | not_run |
| P3 | direct scalar 온톨로지-R-GAT reward와 데이터/규약 검증 | **구현 완료, 학습 미실행** |
| P4 | 동일 초기화·동일 budget 두 정책 학습 | not_run |
| P5 | locked test에서 A/B 평가 suite | not_run |
| P6 | paired 통계·실패 사례·policy-only bundle | not_run |

성능 수치는 아직 없다. 이 저장소의 어떤 문서도 재현율·성공률·회복 시간을 자신의
측정값으로 보고하지 않는다.

## 4. 원문에서 확인된 것

전부 `config/paper/shin2026_reference.yaml`에 출처(표/절/그림)와 함께 전사했고,
`tests/test_shin2026_paper_fidelity.py`가 resolved config와 대조한다. 값이 어긋나면
test가 깨진다(고의로 세 값을 틀리게 만들어 검출되는지도 test한다).

- 관측/행동 (§II-A): grayscale image, 전방축 기준 60도 하향 카메라, proprio =
  body velocity `R^3` + quaternion `R^4`, action = heading frame `[vx,vy,vz,wz]`
- 시뮬레이션 (§IV-A): AerialGym + geometric controller, 512x320, 수평 FOV 90도,
  body velocity noise 0.05 m/s, attitude noise 0.5도, 패드 1.5x1.5 m, dt 0.1 s,
  horizon 300 step, 성공 = 패드 윗면 접촉, 종료 = 접촉/workspace 이탈/horizon,
  RTX 4090에서 약 3.5시간 수렴
- 네트워크 (§III-B, Fig. 4): `l_t` 512, `h_t` 512, `y_t` 256, `s_rel` 6,
  actor는 `y_{6:N}`+`u_t`, critic은 privileged `[u_t, s_rel]`, keypoint encoder는
  Park et al. [17]을 사전학습 후 동결, 육각 6-keypoint(고전 측위에는 4개 필요)
- Table I 초기조건: 고도 `U(2,8)` m, 측방 `U(-3,3)` m, 플랫폼 yaw 오정렬
  `U(-60,60)`도, 초기 속도 `U(0,8)` m/s, 초기 yaw rate 0, 속도 섭동 `U(-0.5,0.5)`
  m/s, yaw rate 섭동 `U(-3,3)` deg/s
- Table II 도메인 랜덤화: 13개 항목 전부 전사
- Table III 보상: 5개 항의 식과 가중치 (1.0 / 1.0 / 0.5 / 1.0 / 2.0)
- §III-C active perception: `r = -alpha[beta(L_est_{t+1}-tau)]_0^1`,
  `alpha=0.1, beta=1.0, tau=0.01`, Eq.(1) `L_est`는 6성분 MSE
- §III-D-4 종료 보상: 성공 +10, 충돌/과도 이탈 -10이며 **shaping을 대체**한다
- §III-D-2 curriculum: `c in [0,1]`, 512 episode마다 level 갱신 (Fig. 3: Level 10→80)

대조 결과 현재 repo의 원문-주장 상수는 **전부 일치**했다. `ShinReward` 구현은
Table III에서 독립적으로 다시 쓴 oracle과 항별로 일치한다.

### Table IV — 재현 목표 (이 repo의 결과가 아님)

10,000 episode, 학습 시나리오. 표는 정수 % 반올림이고 본문이 일부 행에 소수점을 준다.

| Approach | Variant | Success (%) | Pos. RMSE (m) | Vel. RMSE (m/s) |
| --- | --- | --- | --- | --- |
| Proposed | – | 97 | 0.474 | 0.589 |
| Proposed | w/o keypoint encoder | 91 | 0.953 | 1.063 |
| Proposed | w/o active-perception reward | 91 | 1.048 | 0.695 |
| Proposed | w/o state estimation | 73 (본문 73.99) | – | – |
| RL [16] | – | 3 (본문 3.82) | – | – |
| RL [16] | proposed perception backbone | 69 (본문 69.89) | – | – |
| FF-PBVS [27] | w/o GPS error | 67 (본문 67.81) | 1.678 | 0.883 |
| FF-PBVS [27] | w/ GPS error | 23 (본문 23.66) | 2.892 | 0.846 |
| Hybrid | EKF + RL | 59 (본문 59.77) | 1.331 | 1.501 |

Table V(시나리오별 각 1,000 episode, 단일 정책): 8 m/s 98.8, Lin. Accel. 79.1,
Circle 79.6, Zigzag 69.7, U-turn 55.0, Boat 99.5 (%).

이 숫자들이 runtime 코드에 상수로 박히지 않는지도 test가 검사한다.

## 4-1. 원문이 말하지 않는 것 (13건)

`shin2026_reference.yaml: unreported`에 항목과 *이 repo가 어디서 정하는지*를 함께
적었다. 이 값들은 재현되었다고 말할 수 없다. 핵심만:

- PPO hyperparameter 일체 (clip, lr, epoch, minibatch, gamma, GAE lambda, entropy)
- state-estimation MLP와 decision MLP의 층 폭/깊이 (입출력 차원만 주어짐)
- action scaling과 명령 속도/yaw rate 한계
- "workspace 이탈"과 "excessive drift"의 임계값
- 병렬 환경 수와 총 학습 step/episode (Fig. 5 가로축은 120,000 episode까지)
- curriculum level → `c` 대응과 level 일정
- 카메라 extrinsic 평행이동, 해상도/FOV 외 intrinsic
- PACMAN 사전학습 데이터·절차 (가중치 비공개)
- AerialGym/Isaac Gym 버전, geometric controller gain
- Fig. 5의 1-표준편차 밴드를 만든 학습 seed 수
- Table IV RMSE의 누적 구간과 pooled/per-seed 여부

인쇄값 해석 1건도 기록했다: Table II의 external torque가 `-4e3, 4e3 [N.m]`로
인쇄되어 있으나 인접 항이 `±0.75 N`이고 4000 N·m는 섭동이 아니므로
`±4e-3 N·m`로 읽었다. 판단이며 원문 값이 아니다.

## 4-2. px4_transfer 백엔드의 선언된 이탈 (7건)

`benchmarks/paper_reference.BACKEND_DEVIATIONS`에 원문 서술·repo 동작·이유·근거
설정 경로를 등록했고, 등록이 사라지거나 설정이 등록과 어긋나면 test가 깨진다.
7건 모두 **두 arm에 동일**하므로 비교의 교란요인은 아니지만, 원문 재현 주장에는
제약이다.

1. **simulator** — AerialGym + geometric controller → Isaac Sim + PX4 SITL.
   Table II gain은 절대값이 아니라 중앙값 대비 비율로 PX4 gain에 사상.
2. **platform_motion_model** — 평면 random walk → 캠퍼스 도로 waypoint 주행.
   step 섭동 범위는 원문과 동일.
3. **platform_speed** — 원문 `U(0,8)` m/s → 반송차 0.25–0.60 m/s, 상한 1.0 m/s.
   **가장 큰 단일 재현 격차**이며 한 자릿수 이상 차이다.
4. **command_envelope** — 명령 속도 상한 `[2.0, 2.0, 1.0]` m/s. 원문 미기재이지만
   8 m/s 플랫폼을 추종할 수 없으므로 Table V 속도는 이 백엔드에서 도달 불가다.
5. **battery_termination** — 원문 종료 집합에 없는 배터리 고갈 실패 추가.
6. **entry_handover** — 원문은 Table-I 표본 자세에서 episode가 그냥 시작한다.
   실제 비행 스택은 순간이동이 불가하므로 PX4가 그 자세로 날아가 pad-relative
   offset·속도·기하 가시성을 settle window 동안 유지해야 policy로 넘긴다. 설정
   전용이며 policy/보상/로그는 이 gate를 보지 않는다. 속도 한계는 Table II의 축당
   초기 속도 랜덤화(`U(-1,1)` m/s) 이상으로 두어, 원문 자신의 초기조건보다 엄격해지지
   않게 했다.
7. **keypoint_encoder** — PACMAN 가중치 비공개이므로 동등 경로를 자체 사전학습 후 동결.

`sample_initial_condition`/`sample_domain_randomization`은 Table I·II를 그대로
구현하고 있다(test로 확인). 위 3·4번은 그 위에서 **시스템 설정이 좁히는** 것이므로,
paper_reproduction 백엔드를 따로 세우면 sampler 교체 없이 해소된다.

## 5. 보고 시 금지 사항

- 구현 검증률을 성능 재현율로 바꿔 부르는 것
- 미확인 항목을 감춘 "100% 재현" 주장
- PBRS 최적정책 불변, 운용 안전 보장, attention 자체의 인과 해석,
  관계 중요도가 직접 학습 target이라는 주장
- 온톨로지 단독 기여 입증 — 현재 설계는 제안 보상 모듈 *전체*의 효과만 비교한다
- FOV-loss 성적이 나쁜 baseline trial을 health failure로 삭제하는 것
- Table IV·V의 숫자를 이 저장소의 측정값처럼 제시하는 것. 두 표는 재현 *목표*다.
- px4_transfer 백엔드의 결과를 원문 재현으로 부르는 것. §4-2의 6건, 특히 플랫폼
  속도 격차 때문에 이 백엔드는 Table IV·V 조건을 만족하지 않는다.
