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

### 5.1 Keypoint label 규약과 encoder v4 (2026-09-20)

2026-09-20까지의 동결 encoder는 여섯 landmark를 모두 패드 중심 한 점으로 예측했다
(48장의 기하 label Isaac 프레임에서 예측 산포 12.9 px, 실제 44.5 px; PCK@20 15 %).
원인은 두 가지였다.

* **Landmark identity는 관측 불가능하다.** 육각형 배치는 60° 회전 대칭이고 identity
  pip(1~6개 점)은 6 m에서 0.8 px, 4 m에서 1.2 px, 1.6 m에서 3 px다. 패드 좌표계의
  landmark 번호를 그대로 지도학습 target으로 쓰면 여섯 가지 순환 배정이 모두 동등하게
  가능하므로 index별 MSE·heatmap CE 손실의 최적해는 그 평균, 즉 패드 중심이다.
* **수용 영역 31 px.** 이전 trunk(stride-2 conv 4개, 정규화 없음)의 stride-16 셀은
  31×31 px만 보므로 1.6 m에서 300 px인 육각형 전체를 볼 수 없었다.

현재 규약(`keypoint_pretrain.LABEL_CONVENTION`): label은 **image plane에서 정규화**된다.
index 0은 투영된 패드 중심을 기준으로 image +x 축에서 반시계 방향 각도가 가장 작은
꼭짓점이고, 패드의 순환 순서는 유지된다(위에서 본 평면 볼록 육각형의 순환 방향은
카메라 자세와 무관하다). 따라서 target은 영상만의 결정적 함수가 되며, encoder의 k번
채널은 "패드 좌표계 landmark k"가 아니라 "+x에서 반시계 k번째 꼭짓점"을 뜻한다.
하류 소비자(centroid, apparent scale, 여섯 descriptor의 pooling, 가시성)는 모두
identity 비의존이므로 계약이 바뀌지 않는다. 합성·실측 label과 held-out 지표는 같은
규약을 쓴다.

Encoder v4(`ShinKeypointEncoder.implementation` = `isaac-canonical-six-keypoint-unet-v4-stride8-window-visibility`)는
프레임별 표준화 입력, GroupNorm을 가진 stride-2 단계 5개(10×16, 수용 영역 > 프레임),
bottleneck의 global-average context, skip이 있는 decoder 2단계로 stride-8(40×64)
heatmap을 낸다. 합성 렌더러 v2는 Isaac 실측처럼 어두운 배경(건물 모서리, 기둥,
그림자, 로버 본체·마스트) 위의 회색 표적을 그리고, 실측 보정은 label을 정확히 변환한
similarity warp·재노출 증강과 합성 재생(replay)을 섞는다. Held-out viewpoint에서
recall 게이트에 더해 **spread ratio 게이트**(예측 산포 / label 산포 ≥ 0.5)가 중심
수축 encoder를 거부한다.

48장 Isaac 프레임, viewpoint 단위 4-fold 교차검증(프레임 1회씩 검증) 측정치:

| 레시피 | CV RMSE px | CV PCK@20 | recall | 예측/실제 산포 px | 합성 전용 PCK@20 |
|---|---|---|---|---|---|
| v5 encoder + 기존 합성 512×6 + 무증강 미세조정(대조군) | 63.7 | 16.0 % | 81 % | 22.7 / 44.5 | 7 % |
| **참조**: v4 + 정규 label + 렌더러 v2 4096×15 + 증강·재생 800 step | 18.2 | 93.1 % | 97 % | 45.8 / 44.5 | 64 % |
| 참조 + 강한 증강(scale 0.45–2.0, ±60°) + 1600 step **(채택)** | 21.9 | 96.6 % | 98 % | 46.1 / 44.5 | 64 % |
| 참조 + 윈도우 soft-argmax(r=4 cell) **(채택)** | 25.9 | 94.7 % | 97 % | 47.5 / 44.5 | 78 % |
| **최종 채택 조합**: 참조 + 윈도우 soft-argmax + 강한 증강 1600 step | 40.9 | 97.3 % | 99 % | 50.4 / 44.5 | 78 % |
| 참조 + 실측 배경 합성 패드 합성(composite) | 16.4 | 92.4 % | 95 % | 45.5 / 44.5 | 64 % |
| 참조, decoder만 미세조정 | 26.1 | 85.9 % | 96 % | 46.7 / 44.5 | 64 % |
| 참조, 무증강·무재생 미세조정 60 epoch | 31.3 | 95.0 % | 98 % | 47.1 / 44.5 | 64 % |
| 소거: v3(구) 아키텍처 + 정규 label + 렌더러 v2 + 증강 | 57.8 | 17.9 % | 97 % | 24.2 / 44.5 | 8 % |
| 소거: v4 + 정규 label + **기존(v5) 합성 생성기** + 증강 | 43.8 | 93.5 % | 98 % | 48.6 / 44.5 | 58 % |
| 소거: v4 + **identity label**(패드 좌표계 번호) + 렌더러 v2 + 증강 | 32.4 | 98.1 % | 99 % | 47.5 / 44.5 | 38 % |

(측정 2026-09-20, 스크래치 하네스 `kp/harness.py`; RMSE는 한 폴드(cv1)의 소수 이상치에
민감하므로 PCK@20·중앙값·산포를 함께 본다.) 아키텍처 소거는 결정적이다: 구 trunk는 정규
label·새 렌더러·증강을 모두 주어도 산포 24 px로 다시 중심에 수축한다. 렌더러 v2의 기여는
그보다 작다: 기존 합성 생성기로도 미세조정 후 93.5 %에 이르며, v2는 합성 전용 전이(58 → 64 %)와
고정 분할 정확도(91.9 → 95.2 %)를 올린다. identity label 소거는
해석에 주의가 필요하다. 새 아키텍처에서는 identity label도 실측 held-out에서 가장 높은
PCK@20을 내는데, 이는 실측 프레임의 yaw가 {0°, ±35°}로 제한되고 로버 마스트가 항상
같은 landmark 옆에 있어 identity를 추론할 단서가 실측에만 존재하기 때문이다. yaw가
임의인 합성 데이터에서는 identity가 학습되지 않아 합성 전용 전이가 38 %(정규 label
64~78 %)에 그치고, cv1에서 순서 뒤바뀜 이상치(RMSE 66 px)가 난다. 비행 중 상대 yaw는
임의이므로 정규 label을 채택한다. 하류 소비자(centroid, scale, 가시성)는 어느 규약에도
무관하다.

채택 조합의 고정 held-out 분할 성적은 RMSE 6.3 px(중앙값 3.2 px), PCK@20 100 %, PCK@10
84 %다(참조 14.9 px / 95 % / 73 %; 단일 요인 변형은 강한 증강 6.4 px / 100 % / 82 %,
윈도우 soft-argmax 6.9 px / 100 % / 86 %). 파이프라인 자체의 보정 보고(같은 held-out
분할)도 일치한다: 2026-09-20 실행에서 108.8~112.7 px → 6.7~7.4 px, PCK@20 64~74 % →
97~100 %, recall 100 %, spread ratio 0.94.
같은 v4 합성 가중치는 실측 프레임을 한 장도 보기 전에 PCK@20 64~78 %를 낸다(v5 합성
전용: 4~7 %). 저장된 보정 viewpoint는 영상과 자세만 담으므로 datastore 지문에서 encoder
architecture 키를 제거했고, v5 지문으로 저장된 24개 viewpoint는 재비행 없이 다시
label된다. 시연(behavior-cloning) 파일은 v3부터 PNG 프레임을 보존하고 실험 hash가 아닌
교사 비행 지문(교사·게인·curriculum·제어 envelope·deck/battery 프로파일)으로 키잉되므로,
encoder나 PPO 예산이 바뀌어도 교사 비행을 반복하지 않고 재임베딩한다.

### 5.2 정직한 인지가 드러낸 교사 규칙 결함 (2026-09-20)

v4 encoder로 처음 재시작한 실행에서 privileged PD 교사는 40회 비행 중 0회 착륙했다(v5
뒤에서는 22회 중 4회). 원인은 교사의 시야 상실 상승 규칙이었다. 규칙은 encoder가 보고한
`visible_keypoint_fraction < 0.5`이면 고도 0.8 m 이상에서 상승했는데, 60° 하향 카메라의
수직 반시야각은 32°라 PD가 수렴하는 지점(패드 바로 위)에서는 0.4~1.0 m 고도에서 6개 중
2개 landmark만 프레임에 남는다(1.2~1.5 m에서 4개, 2 m부터 6개). v5 encoder는 프레임
밖 landmark의 77 %를 "보인다"고 보고해 이 규칙을 가렸고, v4 encoder(15 %)는 규칙을
정확히 최종 접근 구간에서 발동시켰다.

첫 수정은 규칙을 *지속적 맹목*(landmark 2개 미만이 0.5 s 이상,
`visual_loss_climb_risk`)으로 좁히는 것이었으나 두 번째 실행도 40회 중 0회였다. 어제
착륙했던 seed 90025는 측방 12 cm로 정렬된 채 300 step 동안 접지하지 못했다: 패드 바로
위에서는 landmark 2개만 프레임에 남으므로 정직한 encoder가 그중 하나를 한 프레임만 놓쳐도
관측이 "blind"가 되어 상승이 걸린다(패드 중심은 frustum 안이라
`climb_during_geometric_fov_loss` 지표에는 잡히지 않는다). 교사는 이미 시뮬레이터 진실로
비행하는 특권 교사이므로, 회복 신호도 파이프라인의 유일한 FOV 손실 정의인 기하 패드 중심
frustum 이탈로 바꿨다(`visual_loss_climb_source: geometric`). 교사의 행동은 어느 경우든
training-only label이며 학생에게 전달되는 관측은 바뀌지 않는다. 시연 비행마다 step 단위
트레이스(`training/teacher_trace_<지문>.jsonl`: 고도, 측방 오차, 목표 vz, 가시성,
기하 FOV, PX4 시각)를 남겨 다음 실패는 비행 중 상태로 진단할 수 있다.

세 번째 실행(기하 신호)의 트레이스는 마지막 구속 조건을 드러냈다: 상승은 사라졌지만
측방 정렬이 random-walk deck에 대해 간헐적이고, v4 하강 사다리(1.1 m 위 0.35, 0.6~1.1 m
0.14, 0.6 m 아래 0.04 m/s)는 5 m 진입에서 접지까지 약 30 s의 *연속* 정렬을 요구해 30 s
episode 안에 끝날 수 없었다(seed 90004: step 54에 정렬, step 300에 고도 1.9 m). 네 번째
실행은 사다리를 0.50 / 0.25 / 0.10 m/s(접지 기준 0.55 m/s의 1/5)로, 2.5 m 위·1.5 m 이내
정렬 중에는 0.20 m/s 접근 하강을 허용하고, 시연 비행의 지평선을 450 step으로 늘렸다
(`behavior_cloning.pd_descent_*`, `pd_approach_*`, `horizon_steps`). 결과: 비행한 4회 전부
착륙(seed 90001·90003·90005·90007, 82~195 step, 측방 3.6~10.9 cm; 0.56 팩 seed 90001
포함). 역대 이 교사는 seed 90000/90002/90004/90025에서만 착륙했었다.

### 5.3 시연 시나리오: 패드 급가속 이탈과 회복 (2026-09-21)

training_random_walk 위에서만 비행한 시연은 패드를 한 번도 놓치지 않는다. 5 m 진입에서 PD
교사는 step 15에 정렬, step 30에 1.5 m, step 80에 착지한다(seed 90000 트레이스). 학생은
접근만 복제하고, 실험이 측정하려는 시야 상실·회복은 시연에 없었다.

`behavior_cloning.scenario: training_random_walk_escape_burst`는 같은 seed에서 같은
random walk를 그대로 두고(대시 전까지 위치가 동일) 한 번의 직선 대시를 덧씌운다
(`isaac_sim/pad_motion.py`). 대시는 Isaac이 **UAV가 덱을 실제로 추종하는 순간** 발동한다
(`pad.escape_burst_trigger: following`): 정책 인수 후 2 s 이상 지났고 UAV가 패드 중심에서
측방 0.9 m 이내, 패드 위 0.5–2.5 m에 있을 때(`escape_burst_due`), 인수 25 s 후에는 무조건.
방향은 UAV 카메라의 정면 반대(뒤쪽)다. 카메라는 앞·아래를 보므로 뒤쪽 프레임 경계는
연직점에서 끝나고, 뒤로 달리는 패드는 상대 이동 1 s 남짓에 프레임을 벗어난다. 덱은 1 s 동안
캐리어의 시나리오 최고 속도(이 프로파일에서 8 m/s × 0.125 = 1.0 m/s, `straight_8mps`와 같은
값)까지 가속해 5 m를 달린 뒤 1 s에 걸쳐 원래 walk 속도로 돌아온다. 트랙은 현재 샘플 다음부터
다시 쓰이므로 발동 순간 위치·속도에 점프가 없고, 덱 yaw는 여느 기동처럼 yaw-rate 한계 안에서
새 속도 방향으로 돈다. 대시는 curriculum 스케일을 무시한다: 진입 curriculum과 함께 줄어드는
대시는 시야를 벗어나지 못한다.

첫 버전(2026-09-21 오전)은 seed로 추첨한 4–7 s에 대시를 시작했는데, 그 시점의 PD 교사는
접근 하강 0.2 m/s로 아직 3.6–4.4 m 고도에 있어 2.4 m 떨어진 패드도 프레임 안에 남았다
(3/3 비행 착륙, 시야 손실 0). 시간이 아니라 상태로 발동해야 하는 이유다. `timed`는 그
폐형식 변형으로 테스트용으로 남겼다.

대시 후 기하 FOV 손실 규칙(`visual_loss_climb_source: geometric`)이 0.22 m/s 상승을
명령하는 동안 수평 추종은 계속되고, 대시가 끝나 덱이 0.6 m/s 아래로 돌아오면 교사가
따라잡아 패드가 다시 프레임(연직점 부근)에 들어오고 정렬 후 하강 사다리가 재시작된다. 회복
비행은 접근보다 약 25 s 길어 `horizon_steps`를 450 → 600으로 늘렸다(PPO 에피소드는 300 step
그대로).

시연 단계는 대시보드에서 실시간으로 볼 수 있다(2026-09-21). "교사 시연 비행" 패널은 채택
/필요 착륙 수, 비행/최대 시도, 인프라 스킵, 최근 비행 표(seed·pair·결과·step·착지 측방오차
·FOV 손실 비율)와 **수집 중인 pair마다** 현재 비행의 교사 진단(고도, 측방 오차, 목표 vz,
기하 FOV, 시야 이탈 횟수, 이탈 후 최고 고도, 모드: 추종/이탈·상승/재포착/하강)을 step마다
갱신한다(`BenchmarkMonitor.teacher_step`, `demonstration_progress`; 5.5절). "실시간 궤적" 패널은 pair마다 현재 episode의
UAV(실선)와 착륙 패드(점선) world ENU top-down 궤적을 그린다. 좌표는 매 step 점에
`uav_x/y/z`, `pad_x/y/z`로 함께 발행되므로 시연·FOV 수집·PPO·평가 어느 단계에서나 같은
패널이 동작한다.

시나리오 이름은 게이트웨이(`ros2_ws/.../protocol.py`)와 Isaac(`pad_motion.py`) 양쪽 어휘에
있어야 하며 `scripts/sync_gateway.sh --check`가 두 복사본의 일치를 강제한다. `scenario`는
시연 지문(`_DEMONSTRATION_FLIGHT_KEYS`)에 포함되므로 다른 덱에서 비행한 저장 시연은
재사용되지 않는다. 반대로 순수 전송 예산(`gateway_timeout_s`, `setup_timeout_s`,
`reset_recoveries`, `entry_timeout_s`)은 같은 날 지문에서 제외했다: 완료된 비행의 의미를
바꾸지 않는 값이 바뀔 때마다 교사 비행 4회를 반복하고 있었다.

### 5.4 시연 수집을 스폰된 모든 pair에서 (2026-09-21)

시연 단계는 PPO worker가 하나도 제출되기 전에 실행된다(당시 순서:
교사 시연 → behavior cloning → 학습 스레드 제출 → reward-design 수집. 2026-09-22의
단계 분리 이후에는 학습 스레드 제출이 수집 **뒤로** 옮겨졌다. 5.11 참조). 즉 이
시점에는 스폰된 네 UAV/UGV pair가 전부 놀고 있는데도 수집은 pair 0 한 대에서만 순차로
돌았고, 나머지 세 대는
`max_attempts`(현재 40회) 전체 구간 동안 비어 있었다. 교사 비행도 Isaac/PX4 비행이므로
같은 공유 stage 위에서 병렬로 날 수 있다: 측정된 총 시뮬레이션 처리량은 pair 1·2·3·4대에서
1.00x, 2.60x, 3.28x, 4.96x다(`automatic_parallel_pairs`).

`_prepare_fast_demonstrations`는 이제 `parallel_contexts`(pair별 cfg·카메라·monitor)를 받아
pair 수만큼 seed를 한 배치로 동시에 날린다. 저장 규약은 순차 실행과 동일하게 유지된다.

* 배치가 **합류한 뒤** 메인 스레드에서 seed 순서대로만 기록한다. episode id·첨부 순서·
  attempts CSV는 어느 pair가 먼저 끝났는지와 무관하므로, 같은 seed 집합은 pair 수와 상관없이
  같은 시연 집합을 만든다. 행동 RNG도 seed에서 파생된다(`collect_episode`).
* 시연 지문(`demonstration_fingerprint`)에 pair 수는 들어가지 않는다. 4-pair로 모은 집합을
  1-pair 실행이 그대로 재사용하고 그 반대도 성립한다.
* 배치 폭은 남은 시도 예산(`max_attempts - flight_attempts`)으로 잘라, 병렬이 예산을
  초과해 비행하지 않는다.
* pair별로 교사 인스턴스를 따로 묶는다(서보의 적분·직전 centroid, PD의 트레이스 파일은 모두
  비행 단위 상태다). 병렬 실행에서 PD 트레이스는
  `training/teacher_trace_<지문>_pair<N>.jsonl`로 분리된다.
* 인프라 실패(`BridgeError`)는 worker에서 잡아 두고 **배치 합류 후** 메인 스레드에서만
  복구한다. 공유 stack을 형제 worker가 비행하는 도중에 재건하면 그 episode까지 같이 죽기
  때문이다(2026-09-21의 26회 재건 사건과 같은 종류의 실패). 같은 stack에서 함께 실패한
  pair들은 `recover_infrastructure`의 generation 중복 제거로 재시작 1회를 공유한다.

pair를 덜 쓰고 싶으면 `behavior_cloning.parallel_pairs`로 상한을 줄 수 있고, 기본값은
넘겨받은 pair 전부다.

### 5.5 네 pair 시연을 대시보드와 RViz 2에서 (2026-09-21)

병렬 수집은 관측 경로도 같이 바꿔야 한다. 네 대가 동시에 날면 "현재 비행" 하나로는 세 대가
보이지 않고, RViz 2의 네 panel은 서로 구분되지 않는다.

**대시보드.** "교사 시연 비행" 패널은 이제 왼쪽에 단계 전체의 진척(채택/필요, 비행/최대 시도,
인프라 스킵, 최근 비행 표)을, 오른쪽에 **수집 중인 pair마다 하나씩** 현재 비행 타일(seed,
step, 모드, 고도, 측방 오차, 목표 vz, 기하 FOV, 시야 이탈 횟수, 이탈 후 최고 고도)을 그린다.
진척은 pair 수와 무관한 하나의 값이므로 한 번만 발행하고(`collection_pairs`가 어느 물리
pair가 날고 있는지 말한다), 비행별 진단은 각 pair가 자기 panel로 발행한다
(`teacher_step`, pair_index 주입은 `_LockedMonitor`). 최근 비행 표에는 `pair` 열이 생겨
어느 기체가 그 seed를 날았는지 남는다(`physical_pair_index`, attempts CSV에도 기록).

**RViz 2.** 두 가지를 고쳤다.

* `RvizPublisher.create`는 publisher를 **물리 pair마다** 만든다. 이전에는 method마다
  만들어(`pair_methods=args.pipelines`) 4-pair 실행에서 두 개뿐이었고, pair 2·3의 trail과
  marker와 `map -> landing_pad` TF가 pair 0·1의 namespace로 들어갔다. 생성된 4-pair
  레이아웃(`scripts/make_rviz_layout.py`)의 pair 3·4 panel은 그동안 비어 있었다. 이제
  `pair_methods`는 pair별 method 목록(`_balanced_training_pair_assignment`)이고 이름이
  반복돼도 각 pair가 자기 `/landing_rl/pair_N`, `landing_pad_N`, `uav_body_N`을 갖는다.
* scene HUD의 첫 줄에 **단계와 물리 pair**를 붙였다(`TEACHER DEMONSTRATION · PAIR 3 ·
  BASELINE · Shin SE fixed`). 시연 단계에서는 네 pair가 모두 같은 warm-start 파이프라인을
  날기 때문에, 단계 표시가 없으면 네 panel이 전부 "그 panel 제목의 arm이 PPO episode를
  날고 있다"로 읽혔다. PPO 학습 episode는 단계 접두사 없이 pair와 arm만 쓴다.
  `BenchmarkMonitor.step`이 그 pair의 현재 phase를 함께 넘긴다.

에피소드 초기화 때 지우는 trail도 해석된 pair index로 지우도록 고쳤다. 한 method가 여러
물리 pair를 나는 병렬 학습에서, method만으로는 그 method의 첫 pair 것이 지워졌다.

### 5.6 시연 교사를 image_based_visual_servo_v1로 (2026-09-21)

`behavior_cloning.teacher`를 특권 PD에서 추정기 없는 영상 서보로 바꿨다(`position_gain`
0.35 → 0.55; PD 전용 값은 그대로 남겨 한 줄로 되돌릴 수 있다). 서보는 고정된 keypoint
encoder의 centroid와 겉보기 크기만 읽으므로, 복제되는 사상(寫像)이 학생이 시뮬레이터 없이
자기 관측만으로 재현할 수 있는 것이다 — PD는 시뮬레이터 상대 상태를 읽는다(어느 쪽이든
label은 training-only이고 학생 관측에는 들어가지 않는다).

기록상 착륙은 PD 쪽에 있다(4/4, 4/4, 4/4, 3/3, 4/8 대 서보 0/16). 다만 그 16회는 5.x의
서보 튜닝(적분 0.45x0.60 → 0.20x3.00, 댐핑 0.12 → 0.35, reference_scale 0.25 → 0.06,
베어링 하강 게이트와 flare) *이전* 기록이다. 그리고 재비행 비용이 5.4로 약 1/5이 됐다.

**리스크(명시)**: `_prepare_fast_demonstrations`는 `max_attempts` 안에
`successful_episodes`를 못 채우면 RuntimeError를 내고 실험이 거기서 멈춘다. 서보가 하나도
착륙시키지 못하면 `teacher:`를 `privileged_relative_state_velocity_pd_v4`,
`position_gain`을 0.35로 되돌리면 된다 — PD 시연은 자기 지문으로 저장돼 있어 재비행 없이
재사용된다. 교사와 gain은 지문에 포함되므로(`_DEMONSTRATION_FLIGHT_KEYS`) 이번 전환은
새 시연 집합을 뜻한다(지문 6b29351d93b6 → 6fdd99e3926f).

### 5.7 서보가 불안정했던 이유: 자기 기체의 기울기를 미분했다 (2026-09-21)

5.6으로 전환한 서보는 실기에서 곧바로 재현됐다: 4-pair 병렬로 13회 비행, 착륙 0회
(`accepted=0/4`). 기록된 0/16과 같은 실패다.

원인은 게인이 아니라 **미분항**이었다. 감쇠항은 centroid의 1-step 차분을 dt=0.1 s로
나눈 값을 쓰는데, 그 영상 움직임의 대부분은 덱이 아니라 **기체 자신**이다. 멀티로터는
속도 명령을 기울기로 낸다(약 `atan(a/g)`, 0.6 m/s를 0.35 s에 내면 약 10°, 반시야각
45°에 대해 정규화 column 0.2 — 정렬 cone의 여러 배). 카메라는 기체에 고정돼 있으므로
보정할 때마다 영상이 그만큼 흔들리고, 필터 없는 감쇠항은 그것을 덱의 운동으로 읽어
반대로 명령하며, 그 명령이 다시 기체를 더 기울인다. **기체를 경유한 양의 되먹임**이고,
이것이 기록된 hunting(±0.25 m)과 "3.94 m까지 올라가 이탈"의 정체다.

측정은 `tools/visual_servo_offline.py`로 했다. 투영(`pad_image_position`)과 제어
법칙(`_visual_servo_teacher_action`)은 저장소 코드 그대로이고, 차량(1차 지연 0.35 s),
인코더(centroid 잡음, 1/range 겉보기 크기, 문서화된 가시 landmark 수), 덱(순항
0.366 m/s + 1.0 m/s 대시)만 모델이다. **기체 틸트를 넣기 전에는 같은 게인이 92 %
착륙해** 실기 기록을 설명하지 못했다 — 틸트를 넣자 1 %가 되며 기록과 맞았다. 그것이
이 도구가 원인을 짚었다고 보는 근거다.

변형당 160비행(조건 5 × seed 32):

| 변형 | 착륙 | 후반 측방 진폭 | 적분기 clamp 점유 |
|---|---|---|---|
| shipped (필터 없음) | 1 % | 2.44 m | 72–78 % |
| + `rate_filter_s: 0.30` | 94 % | 0.89 m | 대시 구간 100 % |
| + `anti_windup: true` | 92 % | 1.09 m | 0 % |
| + 수평 권한 1.20 m/s | 74 % | 1.30 m | 81–94 % |

* **필터가 수정이다.** 0.30 s는 기체 틸트의 시정수이고, 0.15 s는 89 %, 0.45 s는 이득 없음.
* **anti-windup은 여유(margin)다.** 착륙률을 올리지는 않지만 적분기가 clamp에 눌러앉는
  것을 없앤다(100 % → 0 %). 방식은 conditional integration(포화 상태에서 더 밀어넣는
  오차만 적분 중단)이다. back-calculation을 먼저 시도했는데 여기서는 틀렸다: 덱을
  상쇄하느라 정당하게 포화된 정상상태까지 되감아 같은 권한에서 착륙률 20점을 잃었다.
* **권한 상향은 기각.** "덱이 clamp만큼 빠르다"는 읽기에서 나오는 당연한 처방이지만
  1.00–1.60 m/s 모두 더 나빴다. 이 루프는 적분 작용이 있고 감쇠가 적어, 권한이 커지면
  기울기도 커진다 — 2026-09-19 비행이 "1.20으로 열었더니 루프를 잃었다"고 기록한 것과
  같은 현상이다.
* **적분기 leak은 기각.** 12–25 s는 무 leak 대비 2–3점(변형당 320비행, 표본 산포 안)
  나았지만 6 s에서 루프가 2.5 %로 붕괴한다. 유효값 바로 옆이 절벽인 손잡이를 2점 때문에
  켜지 않는다. 구현과 `integral_leak_s`는 남겨 두었다.

`rate_filter_s`·`anti_windup`·`integral_leak_s`는 시연 지문
(`_DEMONSTRATION_FLIGHT_KEYS`)에 들어간다. 필터 없이 비행한 시연은 다른 집합이다
(지문 6fdd99e3926f → 01644dd62f49).

### 5.8 그래도 착륙하지 않아 PD로 되돌림 (2026-09-21)

5.7의 수정은 모델이 본 결함을 실제로 고쳤지만, 실기에서는 여전히 착륙이 없다. 같은 날
같은 스택에서 4-pair 병렬로 측정한 시연 원장:

| 교사 | 비행 | 인프라 스킵 | 착륙 |
|---|---|---|---|
| 서보, 필터 없음 (`5a94ac14be08`) | 26 | 10 | 0 |
| 서보, 필터 + anti-windup (`ff291b273219`) | 4 | 0 | 0 |
| privileged PD (`1dbfa17af276`) | 9 | 5 | 3 |

필터판 4비행의 종료 상태는 충돌 1, 배터리 소진 1, 타임아웃 2이고 **최종 측방 오차가
9.0–23.6 m**다. 5.7이 고친 hunting(0.2–0.5 m 규모의 진동)이 아니라 덱을 통째로
놓치고 재포착하지 못한 것이다. 남은 용의자는 루프가 아니라 인코더 쪽이다:
`raw_scale`은 keypoint 확산의 RMS이므로 landmark가 프레임을 벗어나면 붕괴하고,
range gain(`reference_scale / raw_scale`)이 그만큼 급변한다. 오프라인 하네스는 이
모드를 모델링하지 않는다(프레임 밖은 `visual_loss_risk`로만 표현) — 그래서 필터판을
94 %로 예측했고 실기는 0/4였다. 하네스의 한계가 어디인지도 이 숫자로 드러난 셈이다.

따라서 교사는 privileged PD로 되돌렸다(`teacher`, `position_gain: 0.35`). 서보의
안정화 값(`rate_filter_s` 0.30, `anti_windup` true, `integral_leak_s` 0.0)은
`_visual_servo_teacher_action`의 코드 기본값으로 남고 **실험 config에는 쓰지 않는다**:
세 값 모두 `_DEMONSTRATION_FLIGHT_KEYS`에 있으므로, PD가 교사인 동안 이를 기록하면
서보 손잡이가 영향을 줄 수 없는 PD 시연의 지문까지 바꿔 저장된 집합(3/4 성공)을
재비행하게 된다 — 전송 예산을 지문에서 뺀 것과 같은 이유다. 서보로 다시 전환할 때
config에 함께 적어야 서보 시연이 그것을 날린 게인으로 keying된다.

다음에 서보를 다시 볼 때의 출발점은 게인이 아니라 인코더다: 프레임 이탈 구간의
`raw_scale` 거동을 먼저 측정하고, range gain에 슬루 제한을 두거나 scale을 가시
landmark 수로 정규화하는 쪽이다.

### 5.9 시연 덱과 학습 덱의 불일치 (2026-09-21)

PD 교사로 warm start(4/4)를 확보하고 처음으로 PPO까지 간 실행은 episode 48에서 health
gate에 걸려 멈췄다:

```
training health gate stopped shin_se_fixed: position RMSE stalled at 11.20 m
(previous 7.26, limit 3.00); active reward saturation stalled at 87.0%
```

원장을 정상 런과 비교하면 원인이 보인다:

| | 정상 런 (`training_random_walk`) | 멈춘 런 (6개 경로 덱) |
|---|---|---|
| 추정기 위치 RMSE | 1.86 m | 8.73 m |
| 물리 추종 오차 | 3.80 m | 9.33 m |
| 기하 FOV 유지 | 0.83 | 0.47 |
| keypoint 신뢰도 / 가시 비율 | 0.51 / 0.76 | 0.28 / 0.36 |

정상 런에서는 추정기 오차가 추종 오차의 절반이었다 — 추정기가 일을 하고 있었다. 멈춘
런에서는 둘이 거의 같다(8.73 vs 9.33). 추정기가 사실상 0을 내놓는다는 뜻이고, 패드가
프레임 안에 있는 step이 47 %뿐이고 그중에도 landmark를 36 %만 보는 상태에서는 당연하다.

경로 자체가 첫 번째 원인이다: random walk는 원점 근처를 맴돌지만 여섯 덱은 에피소드당
약 10 m를 주행한다. (이 절은 원래 "덱 속도는 범인이 아니다"라고 적었는데, 그 근거로
든 산술이 틀렸다 — 5.10 참조. 덱 속도도 원인의 일부였다.) 그런데
warm start 시연은 `training_random_walk_escape_burst`로 비행하고 있었다 — 복제된 정책은
**주행하는 덱을 쫓아본 적이 없다**. 뒤처지고, 패드가 프레임을 벗어나고, 인지가 굶고,
추정기가 무너진다.

세 가지를 바꿨다.

* `behavior_cloning.scenarios`가 `training.scenarios`와 **정확히 같은 여섯 덱**을
  seed 단위로 순환한다(덱 = `(seed - seed0) % len`, 배치나 재개와 무관하게 결정적).
  테스트가 두 목록의 일치를 고정한다.
* `pd_horizontal_speed_limit_m_s: 0.60 → 0.90`. 서보는 0.60에서만 안정하다고 측정돼
  있으므로 공유 키를 올리지 않고 PD 전용 키로 분리했다(`pd_` 접두사 규약은 하강 키와
  같다). **이 값의 근거로 든 "덱 0.48 m/s" 계산은 틀렸고 0.90 역시 부족했다 —
  5.10을 보라.**
* health gate 재보정: `health_max_position_rmse_m` 3.0 → 12.0,
  `health_grace_episodes` 40 → 200. 기존 값은 학습도 random walk를 쓰던 시절
  기본값이다. 10 m를 주행하는 덱에서는 아직 배우는 중인 정책이 수 미터 뒤처지는 것이
  정상이고, 3.0 m 한계는 따라가는 법을 배우기도 전에 arm을 멈춘다 — 실제로 첫 판정
  기회에서 그렇게 됐다. 무착륙 게이트(계획 episode의 0.75배)는 건드리지 않았다.

시연 지문은 1dbfa17af276 → 718d19ead199로 바뀌므로 교사 집합을 다시 난다
(`max_attempts` 40 → 60: 경로 덱은 random walk보다 어렵고, 4-pair에서 60회는 15배치다).

### 5.10 정정: curriculum은 명명 덱의 속도를 바꾸지 않는다 (2026-09-22)

5.9가 "덱 속도는 범인이 아니다"라고 쓴 근거는 틀렸다. 그 계산은 명명 시나리오 덱의
속도에 curriculum 모션 스케일(`curriculum_min_pad_motion_scale` 0.35 → 1.0)을 곱했는데,
`_prepare_benchmark_motion`은 그 인자를 명명 분기에 적용하지 않는다.
`isaac_sim/pad_motion.py:804`의 `speed_scale = cfg.benchmark_speed_scale`가 지역 이름을
재바인딩하고, 아래의 모든 명명 분기는 그 값을 쓴다. curriculum `scale`은 그 위의
random walk / escape-burst 분기에서만 읽힌다.

`PadTrajectory.reset()`에 넘기는 스케일을 쓸어 측정한 30 s 최대 지면속도:

| 덱 | c=0.00 | c=0.20 | c=0.50 | c=1.00 |
|---|---|---|---|---|
| straight_8mps | 1.000 | 1.000 | 1.000 | 1.000 |
| linear_acceleration_wave | 1.000 | 1.000 | 1.000 | 1.000 |
| circle / zigzag / u_turn | 0.750 | 0.750 | 0.750 | 0.750 |
| vertical_heave_boat | 0.500 | 0.500 | 0.500 | 0.500 |
| training_random_walk | 0.000 | 0.120 | 0.300 | 0.600 |

이것은 프로파일의 의도와도 일치한다 — `shin2026-minimal-system.yaml`은
`benchmark_speed_scale: 0.125`가 "가장 빠른 시나리오를 캐리어 상한에 정확히 맞춘다
(8.0 → 1.00 m/s)"고 적고 있다. 덱은 **의도적으로 고정**이고, 잘못 맞춰진 것은 덱이
아니라 명령 엔벨로프였다.

따라서 5.9의 교사 clamp 0.90 m/s는 `straight_8mps`·`linear_acceleration_wave`(둘 다
1.000 m/s)에 **산술적으로 도달할 수 없었다**. 같은 날 교사 트레이스가 그것을 그대로
보여준다: straight_8mps에서 수평 명령이 100 % step 동안 상한에 포화된 채 상대속도가
0.11–0.18에 머물고 측방오차가 좁혀지지 않는다. 포화 비율(straight 100 % > 원·지그재그
·U턴 93–100 % > 가속파·수직동요 65 %)이 착륙 결과 순서와 정확히 일치한다 — 5.9의 세
수정으로 7비행 4착륙까지 갔지만, 남은 두 실패가 모두 `straight_8mps`였던 것은 우연이
아니었다.

또한 교사 행동은 마지막에 ±0.90으로 클립되므로 실제 전달 가능한 수평속도는
`0.9 × (max_velocity × action_scale)`이다. clamp만 올리면 무효이고, `action_scale`의
바닥(`curriculum_min_action_scale`)이 함께 올라가야 한다.

엔벨로프 수정(교사 clamp 1.80, `curriculum_min_action_scale` 0.50 → 0.90, 실효
1.62–1.66 m/s로 가장 빠른 덱 대비 1.62배)과 그 검증은 병행 세션에서 진행 중이며, 그
결과는 해당 변경과 함께 기록된다.

남은 관련 사항 하나: 하강 분기의 두 절대속도 리터럴(`relative_speed > 0.45`,
`< 0.70`)은 0.29–0.48 m/s 덱에 맞춘 값이다. 상대속도는 벡터 차이이므로 덱이 방향을
바꾸는 지그재그·U턴·원에서는 `명령 + 덱`까지 커진다. 0.90 clamp에서 이미 원 16 %,
지그재그 17 %의 step이 0.70을 넘었고(p90 0.84·0.93), 그 step에서는 하강이 0으로
잠긴다. 엔벨로프를 키우면 이 과도 구간은 더 커지므로, 새 트레이스에서 덱별
"하강 0 유지 비율"을 함께 봐야 한다.

### 5.11 수집 단계와 학습 단계의 분리 (2026-09-22)

한 실행이 수집과 학습을 동시에 하고 있었다. 고정 보상 arm의 PPO를 먼저 제출하고,
FOV-risk 수집기에는 학습 배정이 남긴 pair를 넘겼다. 결과는 세 가지다.

* **수집이 사실상 1페어였다.** 4페어 무대에서 두 학습 arm이 각 2 replica를 날면
  남는 pair가 없다. `_reward_design_pair_indices`는 그때 예외를 던지거나, arm 하나만
  먼저 띄운 경우 남은 2페어 중 **한 대**만 수집에 썼다. 측정된 총 처리량은 1·2·3·4페어에서
  1.00x, 2.60x, 3.28x, 4.96x(`automatic_parallel_pairs`)이므로, 수집 구간 내내 무대의
  대부분이 놀고 있었다.
* **페어 이중 점유가 가능했다.** 2026-09-21에 수집기와 baseline PPO replica가 pair 1을
  함께 잡았고, 두 bridge가 같은 로컬 UDP 포트에 bind(`SO_REUSEADDR`)되어 커널이 게이트웨이
  응답을 한쪽에만 전달했다. 다른 쪽은 hello/state timeout만 보았고 timeout마다 공유
  시뮬레이터를 전체 재건했다 — 4시간에 26회.
* **두 arm의 출발 조건이 달랐다.** 제안 arm의 동결 readout은 비교 arm이 이미 수백
  에피소드를 학습한 뒤에야 만들어졌다. 예산은 같아도 "같은 보상 설계를 상대로 1
  에피소드부터"는 성립하지 않았다.

이제 실행은 두 단계이고 순서는 고정이다. `--stage`가 어느 쪽인지를 말한다.

```bash
./run.sh                   # 수집 → 학습 (한 프로세스)
./run.sh --stage collect   # 데이터셋만 모으고 종료
./run.sh --stage train     # 모은 데이터로 학습·평가, 수집 비행 없음
```

**수집 단계**는 비행해서 재사용 가능한 데이터셋을 만드는 일만 한다: keypoint 측량,
교사 시연, FOV-risk / semantic / adaptive rollout. PPO는 한 에피소드도 돌지 않으므로
`_reward_design_pair_indices`가 반환하는 free pair는 **전부**이고, 수집은 무대 전체를
쓴다. `_collect_fov_risk_data`도 `parallel_contexts`를 받아 시연·adaptive 수집과 같은
배치 규약으로 난다.

* seed·덱·행동 variant는 배치가 **날기 전에** episode id 순서로 배정된다. 어느 pair가
  먼저 착륙했는지는 저장 내용에 영향을 주지 않는다.
* 배치가 합류한 뒤 메인 스레드에서 episode id 순서로만 기록한다. 중단된 수집은 완전한
  prefix에서 재개된다.
* datastore의 중복 제거 identity(seed · source policy · checkpoint sha · variant)와
  seed 고정 train/validation split은 그대로다. 4페어로 모은 집합을 1페어 실행이
  그대로 쓴다.
* 배치 폭은 남은 episode 예산으로 잘라, 하드 캡을 넘겨 비행하지 않는다.

**학습 단계**는 수집을 위해 비행하지 않는다. 각 수집기는 비행을 시작하기 직전에
`_require_flight`를 통과해야 하고, `--stage train`에서는 이것이
`CollectionUnavailable`을 던지며 `--stage collect` 명령을 지목한다. keypoint encoder도
같다: 저장된 artifact가 아직 실기 측량을 통과하지 못했으면 재측량하지 않고 멈춘다.
readout은 그 encoder의 좌표로 라벨된 데이터에 맞춰지므로, 학습 단계가 몰래 다른 계보의
프레임으로 측량을 다시 하면 안 된다.

예외는 한 곳뿐이고 코드에 명시돼 있다. adaptive artifact의 quality gate는 **적합을
시도해야** 데이터가 더 필요한지 알 수 있으므로, `--stage all`에서는 데이터셋을 늘려
재적합하고 `--stage train`에서는 거부한다.

### 5.12 수집 기본 덱을 급가속 이탈 하나로 (2026-09-22)

수집의 기본 덱을 `straight_escape_burst` 하나로 고정했다(`COLLECTION_BASE_SCENARIO`).
패드가 0.500 m/s로 등속 직진하다 드론이 추종에 들어선 순간 1.000 m/s로 급가속해
카메라 프레임을 벗어나는 덱이다. 제안 방법이 주장하는 것이 "패드가 시야를 벗어났을 때
착륙을 회복한다"이므로, 그 사건을 **안정적으로 만드는** 덱이 필요하지 부수적으로 가끔
만드는 덱 여섯 개가 필요한 게 아니다. readout 입장에서도 프레임을 거의 벗어나지 않는
덱은 학습할 양성 표적이 거의 없는 데이터다.

바꾼 곳은 코드의 **기본값**이다. 프로파일이 덱을 선언하면 그쪽이 이긴다.

| 키 | 적용 대상 |
|---|---|
| `training.scenarios` | PPO + 모든 수집의 기본 |
| `behavior_cloning.scenarios` | 교사 시연 |
| `fov_risk_design.scenarios` | FOV-risk rollout |
| `rgat_design.scenarios` | semantic R-GAT rollout |
| `adaptive_reward_design.scenarios` | adaptive reward rollout |

`three_arm_burst_comparison.yaml`은 이미 `training`·`behavior_cloning`·`evaluation`
모두 이 덱 하나였으므로 기본 실행의 동작은 바뀌지 않는다.
`two_pipeline_comparison.yaml`의 여섯 덱도 그대로 남아 있다.

두 가지를 함께 고쳤다.

* **semantic R-GAT 수집의 덱이 하드코딩돼 있었다.** `_collect_semantic_data`가
  `scenario="training_random_walk"`를 박아 쓰고 있어서, 정책이 실제로 나는 덱과 무관한
  궤적으로 semantic outcome 모델이 적합됐다. 이제 다른 수집과 같이 episode index로
  회전하는 덱 목록을 받고, `source_behavior_policy.scenario_cycle`에 기록된다.
* **덱이 FOV datastore 지문에 없었다.** `fov_risk_data_fingerprint`의 docstring은
  "the trajectory distribution the episodes are drawn from"을 포함한다고 말하면서
  실제로는 system 프로파일의 `pad.motion` 계열만 담고 있었다. 덱 이름은 없었다.
  2026-09-22 확인 시점의 활성 누적은 76 episode였고 그중 약 59개가 09-20~09-21의
  6-덱 실행에서 온 것이었다 — burst 전용 readout이 그 59개 위에 그대로 적합됐을
  것이고, 그랬다는 기록조차 남지 않았을 것이다. 이제 `scenarios`가 지문에 들어가고,
  각 episode의 provenance에도 덱 이름이 남는다.

  **대가:** 지문이 바뀌므로 기존 76 episode(약 17k step)는 새 수집에 보이지 않는다.
  DB에는 감사용으로 남는다. 09-18~09-19의 283 episode는 그 이전 지문이라 이미
  보이지 않는 상태였다.

### 5.13 덱이 실제로 어디 있었는지: 보간 앵커 버그 (2026-09-22)

대시보드 궤적 패널에서 `straight_escape_burst` 덱이 직선이 아니라 접힌 띠로 그려졌다.
덱 자체는 직선으로 달리고 있었고, **보고되는 위치가 틀렸다.**

`PadTrajectory.pose()`의 random_walk 보간:

```python
offset = (track[index] + random_walk_origin).copy()   # 앵커가 이미 더해짐
offset += fraction * (track[index + 1] - offset)      # 보간 대상에서 앵커가 빠짐
```

둘째 줄의 보간 목표가 `track[index+1] - origin`이 되어, `fraction`이 0→1로 가는 동안
보고 위치가 앵커 벡터 전체만큼 뒤로 밀렸다가 다음 샘플에서 되돌아온다. 톱니파다.

`route_start: continue`가 앵커를 0이 아니게 만드는 **두 번째 에피소드부터** 나타나므로
첫 에피소드만 보는 검사는 전부 통과했다. minimal 프로파일 실측:

| | 0.025 s 샘플당 이동 |
|---|---|
| 기대 (0.5 m/s) | 0.0125 m |
| 수정 전 | 3.76 – 11.24 m |
| 수정 후 | 0.0125 m (일정) |

연속 30 에피소드에서 앵커는 최대 38.2 m까지 자란다. 즉 1.5 m 덱이 0.1초마다 최대 38 m를
순간이동했고, 이건 그림만의 문제가 아니라 `LandingPad.advance()`가 그 값을 USD 프림에
그대로 쓰기 때문에 **카메라·기하 FOV 라벨·상대상태 보조 회귀·보상이 전부 그 덱을 봤다.**
이 버그 아래에서 수집된 FOV-risk episode는 무효다.

함께 고친 것: 한 arm의 모든 replica가 그 arm의 primary pair monitor로 보고하고 있어서,
150 m 떨어진 두 대의 패드가 한 패널에 한 폴리라인으로 그려졌다(`benchmark_step_pair_0`에
42행 / 21 step). `train_live(env_monitors=...)`로 replica마다 자기 pair에 보고한다.
비행 자체는 언제나 각자의 pair에서 이뤄졌으므로 텔레메트리 경로만의 문제다.

### 5.14 수집 기본 덱을 닫힌 트랙으로 (2026-09-22)

`straight_escape_burst`는 **에피소드 안에서만** 직선이다. `route_start: continue`가 덱을
끝난 자리에 두고 다음 에피소드가 새 heading을 뽑으므로, 에피소드 열 전체는 원점에서
멀어지는 2-D 랜덤워크다. 그래서 `reset`은 반경 절반을 넘으면 heading을 원점 쪽으로
돌려야 하고, `pose`는 경계에서 clamp해야 한다. 둘 다 "가면 안 되는 곳으로 가고 있는 덱"에
가하는 교정이다.

`straight_escape_burst_track`은 그 이유 자체를 없앤다. 20 m 직선 두 개를 6 m 반경의
등속 반원 두 개로 이은 닫힌 오벌이고, 덱은 그 위를 영원히 돈다.

* **직선은 직선이다.** 직선 구간의 급가속은 heading 분산 0.000000°, straightness 1.000000.
* **등속이다.** 코너에서 감속하지 않는다 — 회전은 `yaw_rate = v / R`의 등속 원운동이다.
* **유계다.** 300 에피소드(2.5시간), 매 에피소드 급가속 포함해서 32 x 15 m 안에 머문다.
  최대 반경 26.9 m로, inward steering이 시작되는 60 m에도 닿지 않는다.
* **에피소드를 위상으로 잇는다.** 위치 오프셋과 새 heading이 아니라 주행 거리(arc length)를
  넘긴다. 다음 에피소드가 같은 랩을 이어서 돌 뿐, 루프가 회전하지 않는다.

급가속을 **경로 이탈이 아니라 트랙 위의 속도 사건**으로 정의한 것이 핵심이다. 처음에는
열린 시나리오처럼 "카메라 정후방"으로 직진시켰는데, 그러면 매번 루프가 5 m씩 평행이동해
30 에피소드 만에 arena 경계에 닿았다(측정). 닫힌 루프는 평균 변위가 0이라 heading 회전으로
되돌릴 수도 없다. 속도만 두 배로 올리면 FOV 이탈 사건은 그대로 일어나면서 — 직선 구간에서의
2배속은 원래 시나리오 그 자체다 — 경로는 움직이지 않는다.

기하는 `pad.benchmark_track_straight_m`(20.0)과 `pad.benchmark_track_radius_m`(6.0)로
조정한다. `benchmark_speed_scale`에 스케일되지 않는다: 덱을 느리게 하는 것과 달리는 땅을
줄이는 것은 다른 일이다. 반경은 덱 대각선 이상이어야 하며, 아니면 설정 오류로 거부된다.

보상 설계 rollout을 나는 정책도 단일 규칙이 됐다. 디스크에 호환 checkpoint가 있으면
그것, 없으면 BC로 데워진 동결 초기화다. 이전에는 단일 페어 실행만 source arm의 PPO를
`train_count` 에피소드만큼 따로 돌렸고 그 비용이 `N_reward_design`에 들어갔다.
병렬 실행은 이미 동결 source를 쓰고 있었으므로, 이제 1페어와 4페어가 같은 행동 분포에서
수집하고 어느 쪽도 source 정책 PPO를 보상 설계 예산에 달지 않는다.

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

### 7.1 진입 예산 두 가지: 벽시계 guard와 게이트웨이 hold (2026-09-21)

`benchmark.entry_timeout_s`는 **멈춘 시뮬레이터**를 잡기 위한 벽시계 guard다. 4개
카메라를 렌더하는 stage에서는 길어야 해서 현재 1200 s이고, `_pair_live_config`가
pair 수에 따라 한 번 더 키운다(4-pair에서 1320 s).

게이트웨이가 진입 setpoint를 자율 유지하는 시간은 별개의 예산이며 프로토콜이
`GOTO_MAX_HOLD_S = 900 s`로 제한한다(`ros2_ws/.../protocol.py`). 학습기가 guard를
그대로 `hold_s`로 보내면 게이트웨이는 요청 자체를 거부한다:

```
ontology_rgat.bridge.GatewayRejected: PX4 gateway rejected request: hold_s must be in (0, 900.0]
```

이 거부는 시연 이후 **모든 단계의 첫 reset**을 실패시킨다. 오늘까지 보이지 않았던
이유는 시연 단계가 자기 비행을 위해 guard를 120 s로 줄여 쓰기 때문이고, 그래서
warm start를 처음으로 끝낸 실행(교사 4/4 확보 직후 FOV-risk 수집)에서 드러났다.

`bridge.entry_hold_seconds`가 요청을 프로토콜 상한으로 자른다(초과 시 1회 경고).
guard 자체는 건드리지 않는다 — 두 예산은 서로 다른 것을 지키고, 실제 진입은 수십 초
안에 끝나므로 900 s hold는 충분하다. 학습기 쪽 상수는 테스트가 게이트웨이 프로토콜
값과 같은지 고정한다(`test_the_entry_hold_never_exceeds_what_the_gateway_accepts`).

### 7.2 세 번째 arm이 드러낸 가정: `pipelines`는 arm 목록이 아니다 (2026-09-22)

> 이 문서는 6-덱 2-arm 설계의 기록이다. 3-arm 비교 자체는
> [THREE_ARM_BURST_COMPARISON.md](THREE_ARM_BURST_COMPARISON.md)가 다루고, 그 §6이
> 같은 결함 목록을 요약한다. 아래는 각 결함이 무엇을 어떻게 틀리게 했는지의
> 상세 기록이다.

비교가 2-arm에서 3-arm이 됐다 — 비학습 통제조건(visual servo, 안 되면 privileged PD),
논문 기준 PPO(`shin_se_fixed`), 제안 모델(`shin_se_onto_rgat_recovery`). 학습하는 것은
둘뿐이고 세 번째는 체크포인트도 학습곡선도 없다.

이 한 가지 구조 변경이 하루에 같은 부류의 결함 **다섯 개**를 드러냈다. 전부
"모든 arm은 학습한다"를 가정한 코드 경로다.

| 위치 | 증상 |
|---|---|
| 대시보드 | arm 목록 폴백이 비학습 arm을 학습형으로 되살림 |
| 러너 | 체크포인트 없는 arm의 평가 행이 superseded로 아카이브됨 |
| 러너 | pipeline spec이 없어 `_reward`가 마지막 분기로 떨어져 평가 중 예외 |
| 보고 | `METHODS` 하드코딩으로 통제 arm이 **모든 그림**에서 누락 |
| 보고 | 완결성 검사가 통제 arm에도 체크포인트를 요구 → 완료된 런이 **최종 평가 대신 학습 예비 데이터**를 그림 |

마지막 것이 가장 위험했다. 그림은 그림처럼 보이고, 실제로는 PPO 10 episode가 결과로
실려 있었다. 표에는 세 arm이 들어 있는데 그림에는 둘만 있는 상태도 눈에 띄지 않는다.

**근본 원인**은 `pipelines`가 *학습하는 목록*인데 *arm 목록*처럼 읽힌다는 것이다. 2-arm
시절에는 둘이 같았으므로 구분할 이유가 없었다.

**지속적 수정**은 manifest.json이 arm 목록을 직접 싣는 것이다:

```json
"arms": [
  {"method": "shin_se_fixed", "label": "...", "learned": true},
  {"method": "image_based_visual_servo_v1", "label": "Visual servo (non-learned)",
   "learned": false}
]
```

`BenchmarkMonitor.configure(arms=...)`도 같은 모양을 받아 store에 `benchmark_arms`로
발행하고, 대시보드는 여기서 arm 집합을 만든다(학습곡선은 `learned` arm만, 비학습 arm은
빈 차트 대신 아예 없음). 보고 모듈은 선언된 arm을 우선하고, 없으면 `pipelines`를 학습형
목록으로 보고 실제로 비행한 arm을 평가 행에서 주워 담는다. RViz HUD와 pair 제목도 같은
목록에서 나온다.

라벨은 config에서 온다. 단, 선언된 라벨이 method id와 같으면(= config에 라벨이 없어 런이
id를 적은 경우) 모듈의 이름이 이긴다 — id는 마지막 수단이다.

**남는 지침**: 이제 `pipelines`에서 arm 집합을 추론하는 코드는 *선택해서* 그러는 것이다.
새 소비자를 쓸 때는 `arms`를 읽고, `learned`로 학습 전용 경로를 거르라.

## 8. 산출물

`evaluation/per_episode.csv`(원자료), `evaluation/paired_summary.csv`(시나리오별
계층 bootstrap 요약), `tables/primary_comparison.*`(주 비교), `manifest.json`
(resolved config, 두 spec, seed/budget, artifact provenance).
