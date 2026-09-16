# Shin baseline 대 baseline + Ontology-R-GAT FOV 비교

## 연구 질문

완전한 Shin et al. baseline을 변경하지 않고 미래 FOV-loss 위험 보상 하나를 추가하면
표적 가시성 유지와 이동 플랫폼 착륙 성능이 향상되는가?

## Pipeline

| 계약 | `shin_se_fixed` | `shin_se_onto_rgat_recovery` |
|---|---:|---:|
| 6-keypoint fiducial 착륙 표적 | 동일 | 동일 |
| 6-keypoint encoder | 동일 | 동일 |
| LSTM, 6-D relative-state estimator | 사용 | 사용 |
| position/velocity auxiliary loss | 사용 | 사용 |
| PPO actor, asymmetric critic | 동일 | 동일 |
| actor/critic observation, action | 동일 | 동일 |
| Shin 5개 shaping 항과 `[1,1,0.5,1,2]` | 동일 | 동일 |
| active-perception reward | 사용 | 사용 |
| terminal reward, curriculum, environment | 동일 | 동일 |
| Ontology-R-GAT future-FOV-loss branch | 없음 | 추가 |

실행 시 `assert_primary_baseline_equivalence()`와 YAML/spec validation이 baseline flag가
달라지면 즉시 실패한다. 두 agent model은 같은 seed에서 동일 state-dict key, shape와
초기값을 가져야 한다.

## 보상

$$r_{base}(t)=r_{task}(t)+\sum_iw_i^0\rho_i(t)+r_{active}(t)$$

$$r_{active}(t)=-0.1\,clip(L_{est}(t+1)-0.01,0,1)$$

$$r_{proposed}(t)=r_{base}(t)-\lambda_{fov}p_{fov\_loss}(t)$$

`lambda_fov`의 기본값은 0.1이며 설정 가능하다. 0이면 같은 trajectory의 두 보상은
수치적으로 동일하다. 다섯 baseline 가중치는 설정 대상이 아니다.

FOV-risk label의 기본 prediction horizon은 1.0초이며, 실제 control frequency에
맞춰 step 수로 변환한다(10 Hz면 10 step). `visibility_criterion`은
`geometric_pad_center_in_fov`이고, 제안 branch는 `freeze_during_ppo: true`로
설정된다. R-GAT은 Huber regression + contract rule `R-04` offline 학습의 validation-best checkpoint로 고정되며
PPO optimizer에 들어가지 않는다.

## 인지(perception)와 기하 FOV의 분리

배포되는 정책의 인지 경로는 학습된 6-keypoint encoder이므로, 착륙 표적도 bit-coded
tag board가 아니라 6개의 고정 landmark를 가진 fiducial이다. ArUco dictionary, marker
ID, `cv2.aruco` detector는 주 실험의 학습·평가·보상·label·reset 어디에도 없다
(legacy detector 프로파일은 `config/system.yaml`에만 남는다).

기하 FOV와 신경망 인지 품질은 서로 다른 변수이며 절대 섞지 않는다.

| 양 | 정의 | 출처 | 용도 |
|---|---|---|---|
| `geometric_pad_center_in_fov` | 패드 중심이 양의 depth로 정규화 image 좌표 `[-1,1]^2` 안에 투영 | 시뮬레이터 기하 (`isaac_sim/keypoint_geometry.py`) | reset gate, 지표, R-GAT label |
| `keypoint_confidence`, `visible_keypoint_fraction` | 동결된 encoder 출력 | 카메라 이미지 | 온라인 R-GAT 입력, 지표 |

패드가 프레임 안에 있는데 encoder가 놓치는 경우는 *인지 저하*이지 FOV 손실이 아니다.
반대로 keypoint 추정이 아직 확신에 차 있어도 패드 중심이 frustum을 벗어나면 기하
FOV 손실이다. 지표도 `geometric_fov_*` 계열과 `*keypoint*` 계열로 분리해 보고한다.

시뮬레이터 기하는 (1) keypoint 지도학습 label, (2) episode 초기 가시성 gate,
(3) 비대칭 critic과 relative-state 보조 지도학습, (4) 평가 지표와 offline
future-FOV-loss label 로만 쓰인다. PPO actor 관측과 온라인 R-GAT graph 입력에는
들어가지 않으며, 두 경계 모두 이름 기반 거부로 실행 시 강제된다.

PACMAN의 자산·landmark 표·가중치는 공개되어 있지 않다. 여기의 표적과 encoder는
같은 *인터페이스*를 갖는 문서화된 호환 근사이며 PACMAN 재현이 아니다.

## 설정과 실행 프로파일

주 설정은 `config/experiments/two_pipeline_comparison.yaml`이다. quick profile은
8 training episodes와 8 FOV-risk design episodes를 사용한다. 명시적 full 실행의
설정 예산은 40,000 training episodes와 40,000 FOV-risk design episodes이며, 실행
스크립트는 deadline을 위해 CLI에서 더 작은 예산으로 덮어쓸 수 있다.

인자 없는 상위 `./run.sh`는 별도 `seminar_10h_two_pipeline.yaml`을 선택해 pipeline당
144 training episodes, 40 data episodes, 80 epochs, 5 evaluation episodes를 사용한다.
이 설정은 `publication_claim_allowed: false`인 예비 비교이므로 full 결과와 섞어
집계하지 않는다.

## 평가

주 지표는 Landing Success Rate(엄격 안전착륙 기준), Geometric FOV Loss Episode
Rate, Geometric FOV Retention Ratio, Mean/Maximum Continuous Geometric FOV Loss
Duration이다. 참고로 baseline 논문의 접촉 기준 성공률은 `paper_contact_success`로
별도 병기하며 엄격 기준을 대체하지 않는다. 인지 품질은
`low_keypoint_visibility_fraction`, `keypoint_confidence_mean`,
`visible_keypoint_fraction_mean`로 따로 보고한다. 이 지표들은 비교의 종속변수이므로
학습 health gate가 이를 이유로 실행을 중단하지 않는다. R-GAT은 MAE, RMSE, bias, R², 상수 예측기 RMSE 기준선,
confusion matrix를 보고한다. 상태추정 오차와 loss는 양쪽 공통 진단값이며 제안 기여로
해석하지 않는다.

Attention/message-passing coefficient는 relation importance나 인과 근거로 보고하지 않는다.

YAML에 선언된 평가 시나리오는 `training_random_walk`, `straight_8mps`,
`linear_acceleration_wave`, `circle`, `zigzag`, `u_turn`, `vertical_heave_boat`다.
세미나 프로파일은 이 중 `training_random_walk`, `circle`, `zigzag`만 사용한다.

