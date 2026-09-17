# 실험 설계: Shin baseline 대 baseline + Ontology-R-GAT FOV

[문서 안내](README.md) · [제안 알고리즘](ONTOLOGY_RGAT_FOV_RISK.md) ·
[Shin baseline](SHIN2026_BASELINE.md) · [아키텍처](ARCHITECTURE.md)

## 1. 연구 질문

완전한 Shin et al. baseline을 어떤 항도 바꾸지 않은 채 미래 FOV-loss 위험 보상
하나를 더하면 표적 가시성 유지와 이동 플랫폼 착륙 성능이 향상되는가?

귀무가설은 "가산항은 기하 FOV 유지 지표와 엄격 착륙 성공률을 바꾸지 않는다"이며,
종속변수는 §4의 지표들이다.

## 2. 통제변수와 유일한 실험 요인

| 계약 | `shin_se_fixed` | `shin_se_onto_rgat_recovery` |
|---|:---:|:---:|
| 6-keypoint fiducial 착륙 표적 | 동일 | 동일 |
| 6-keypoint encoder와 checkpoint | 동일 | 동일 |
| LSTM, 6-D relative-state estimator | 사용 | 사용 |
| position/velocity 보조손실 | 사용 | 사용 |
| PPO actor, asymmetric critic | 동일 | 동일 |
| actor/critic observation, action | 동일 | 동일 |
| Shin 5개 shaping 항과 `[1, 1, 0.5, 1, 2]` | 동일 | 동일 |
| active-perception reward | 사용 | 사용 |
| terminal reward, curriculum, environment | 동일 | 동일 |
| 학습·평가 seed와 episode 예산 | 동일 | 동일 |
| **Ontology-R-GAT future-FOV branch** | **없음** | **추가** |

`assert_primary_baseline_equivalence()`와 YAML/spec 교차검증이 실행 시점에
baseline 계약 flag의 불일치를 즉시 실패로 만든다. 두 model은 같은 seed에서 동일한
state-dict key, shape, 초기값을 가져야 한다.

## 3. 보상 대비

$$r_{\text{base}}(t)=r_{\text{task}}(t)+\sum_{i=1}^{5} w_i^0\rho_i(t)+r_{\text{active}}(t)$$

$$r_{\text{active}}(t)=-0.1\,\mathrm{clip}\bigl(L_{\text{est}}(t{+}1)-0.01,\,0,\,1\bigr)$$

$$r_{\text{proposed}}(t)=r_{\text{base}}(t)-\lambda_{\text{fov}}\,q_\theta(G_t)$$

`λ_fov`의 기본값은 0.1이며 설정 가능하다. 0이면 같은 trajectory의 두 보상은 수치적
으로 동일하다. 다섯 baseline 가중치는 설정 대상이 아니다.

FOV-risk label의 기본 prediction horizon은 1.0초이고 실제 control frequency에 맞춰
step 수로 변환된다(10 Hz → 10 step). `visibility_criterion`은
`geometric_pad_center_in_fov`, 제안 branch는 `freeze_during_ppo: true`다.
R-GAT은 Huber 회귀 + 규약 `R-04`의 validation-best checkpoint로 고정되며 PPO
optimizer에 들어가지 않는다.

## 4. 종속변수

### 4.1 착륙

엄격 성공 판정은 다음을 모두 만족해야 한다.

| 조건 | 임계값 |
|---|---|
| deck 접촉 | contact sensor |
| 수평 오차 | ≤ 0.35 m |
| 접촉 수직속도 | ≤ 0.55 m/s |
| deck 대비 상대 수평속도 | ≤ 0.45 m/s |
| tilt | ≤ 10° |
| 각속도 | ≤ 45°/s |

접촉 위치는 접촉 sample에서, 속도·tilt·각속도는 직전 비접촉 sample에서 계산해
post-impact bounce를 배제한다. 논문 기준의 접촉 성공률은 `paper_success`로 병기
하지만 엄격 기준을 대체하지 않는다.

### 4.2 기하 FOV

`geometric_fov_loss_episode_rate`, `geometric_fov_retention_ratio`,
`mean_/maximum_continuous_geometric_fov_loss_duration_s`,
`geometric_fov_reacquisition_rate`, `mean_geometric_fov_reacquisition_time_s`,
`climb_during_geometric_fov_loss_fraction`.

### 4.3 인지 품질 (별도 계열)

`keypoint_confidence_mean`, `visible_keypoint_fraction_mean`,
`low_keypoint_visibility_fraction`, `descent_during_low_keypoint_visibility_fraction`.

### 4.4 R-GAT

MAE, RMSE, bias, R², 상수 예측기 RMSE 기준선, validation 규약 위반량, 추론 지연.

### 4.5 공통 진단

상대 위치/속도 추정 RMSE와 loss, return, touchdown 시간. 양쪽에 공통이므로 제안
기여로 해석하지 않는다.

이 지표들은 비교의 종속변수이므로 학습 health gate가 이를 이유로 실행을 중단하지
않는다. FOV 성적이 나쁜 baseline trial을 health failure로 삭제하면 제안 보상이
개선하려는 바로 그 약점을 지우게 된다.

## 5. 인지와 기하 FOV의 분리

배포되는 정책의 인지 경로는 학습된 6-keypoint encoder이므로 착륙 표적도 bit-coded
tag board가 아니라 6개의 고정 landmark를 가진 fiducial이다. ArUco dictionary,
marker ID, `cv2.aruco` detector는 주 실험의 학습·평가·보상·label·reset 어디에도
없다(legacy detector 프로파일은 `config/system.yaml`에만 남는다).

기하 FOV와 신경망 인지 품질은 서로 다른 변수이며 절대 섞지 않는다.

| 양 | 정의 | 출처 | 용도 |
|---|---|---|---|
| `geometric_pad_center_in_fov` | 패드 중심이 양의 depth로 정규화 image 좌표 `[-1,1]^2` 안에 투영 | 시뮬레이터 기하 (`isaac_sim/keypoint_geometry.py`) | reset gate, 지표, R-GAT label |
| `keypoint_confidence`, `visible_keypoint_fraction` | 동결 encoder 출력 | 카메라 이미지 | 온라인 R-GAT 입력, 지표 |

패드가 프레임 안에 있는데 encoder가 놓치는 경우는 *인지 저하*이지 FOV 손실이 아니다.
반대로 keypoint 추정이 아직 확신에 차 있어도 패드 중심이 frustum을 벗어나면 기하
FOV 손실이다.

시뮬레이터 기하는 (1) keypoint 지도학습 label, (2) episode 초기 가시성 gate,
(3) 비대칭 critic과 상대상태 보조 지도학습, (4) 평가 지표와 오프라인 future-FOV
label 로만 쓰인다. PPO actor 관측과 온라인 R-GAT graph 입력에는 들어가지 않으며,
두 경계 모두 이름 기반 거부로 실행 시 강제된다.

PACMAN의 자산·landmark 표·가중치는 공개되어 있지 않다. 여기의 표적과 encoder는
같은 *인터페이스*를 갖는 문서화된 호환 근사이며 PACMAN 재현이 아니다.

## 6. 평가 프로토콜

* **시나리오**: `training_random_walk`, `straight_8mps`,
  `linear_acceleration_wave`, `circle`, `zigzag`, `u_turn`, `vertical_heave_boat`.
  세미나 프로파일은 이 중 `training_random_walk`, `circle`, `zigzag`만 사용한다.
* **Paired seeds**: 두 pipeline이 동일 시나리오·동일 seed 집합을 비행한다.
* **Checkpoint 선택**: 시나리오별 held-out 결정론 seed로 latest/best 후보를
  검증해 선택하고(`held_out_deterministic_multi_seed_v1`) 그 결과를 동결한다.
* **Crossover**: 평가 episode를 방법 × episode index로 물리 pair에 순환 배정해
  각 방법이 모든 pair(= 경로 위상)를 고루 경험하게 한다.
* **학습 배정 counterbalance**: `training_replicate`마다 cyclic Latin square로
  방법-pair 대응을 바꾼다.
* **통계**: 동일 replicate·시나리오·seed 짝의 차이를 모으고, replicate를 먼저
  resampling한 뒤 그 안에서 episode를 resampling하는 계층 bootstrap(10,000 draw)
  으로 평균 차이의 95% 신뢰구간을 낸다. replicate가 하나뿐이면 단위가 `episode`로
  기록되어 결과에 드러난다.

Attention/message-passing coefficient는 relation importance나 인과 근거로 보고하지
않는다.

## 7. 설정과 실행 프로파일

주 설정은 `config/experiments/two_pipeline_comparison.yaml`이다. quick profile은
8 training episodes와 8 FOV-risk design episodes를 사용한다. full 실행의 설정
예산은 reference 예산인 pipeline당 40,000 training episodes, 40,000 FOV-risk
design episodes, pipeline당 16,000 evaluation episodes(random-walk 10,000 +
6개 궤적 시나리오 각 1,000)이다.

`./run.sh --mode full`은 이제 이 설정 예산을 그대로 실행한다. 이전에는 launcher가
seminar deadline을 위해 800 total / 5 evaluation / 40 design flight로 줄여
덮어썼으나, 그 preview 예산은 아래 `--seminar-fast` 프로파일로 분리했다. 더 짧은
임시 실행은 `--train-episodes`, `--total-train-episodes`, `--eval-episodes`,
`--rgat-data-episodes`로 여전히 덮어쓸 수 있다.

zero-success health gate의 상한 `ppo.health_no_landing_max_grace_episodes`는
reference 예산의 10%인 4,000으로 둔다. 코드 기본값 120은 preview 예산에 맞춘
고정 상한이라 40,000 예산에서는 전체의 0.3%에 불과했다. `no_landing_abort_episode`는
`min(상한, 0.75 x 계획 episode)`를 쓰므로 144-episode seminar 프로파일의 실효
상한(108)은 바뀌지 않는다.

인자 없는 상위 `./run.sh`는 별도 `seminar_10h_two_pipeline.yaml`을 선택해 pipeline당
144 training episodes, 40 data episodes, 80 epochs, 5 evaluation episodes를 쓴다.
이 설정은 `publication_claim_allowed: false`인 예비 비교이므로 full 결과와 섞어
집계하지 않는다.

## 8. 산출물

`evaluation/per_episode.csv`(원자료), `evaluation/paired_summary.csv`(시나리오별
계층 bootstrap 요약), `tables/primary_comparison.*`(주 비교), `manifest.json`
(resolved config, 두 spec, seed/budget, artifact provenance).
