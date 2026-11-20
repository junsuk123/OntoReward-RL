# 자기 항법 열화(관측 /4)와 WMON·칼만 비교

> **명칭 안내:** 이 문서의 WMON(Wave-to-Matter Ontological Network)은 저자가 별도 연구에서 개발한 자기 항법 추정기의 내부 명칭이다. 기술적으로는 인과적 FFT 전처리, 주파수 기반 측정 신뢰도, 운동학적 일관성 잔차를 쓰는 가중 최소제곱 상태 추정기이며, 계수는 SPSA + Adam으로 보정한다. README와 그림에서는 *FFT-Assisted Kinematic Weighted Least Squares (WLS) Estimator*(약칭 Kinematic WLS)로 표기한다. 착륙 정책 연구에서는 실험 환경의 구성 요소로만 쓰며, 본 연구의 기여로 주장하지 않는다. 이 문서는 실험 기록이므로 내부 명칭과 수치를 그대로 둔다.

작성일: 2026-11-10 (KST)  
상태: **옵트인 관측 프로필 `spatial-current-observation/4`. 등록 관측 /3과 모든 등록 해시는 HEAD와 동일.**
2026-11-10 학습 경로에 연결(§6): /4는 설정 파일의 `navigation:` 섹션으로 선언하고, 그 설정으로 실행하는
모든 단계가 /4로 학습·평가·해시되며 자기 결과 폴더를 쓴다.

## 6. 학습 경로 연결과 등록 (2026-11-10)

| 변경 | 내용 |
|---|---|
| `contracts.load_config` | 최상위 선택 키 `navigation:` 하나만 추가로 허용(`NavigationProfile.from_mapping`, 내부 키는 모두 필수). 없으면 `config.navigation = None` — 기존 설정은 그대로 |
| `contracts.contract_manifest(config)` | 설정의 프로필로 관측 /4 매니페스트를 만든다. 다른 프로필을 명시하면 거부. `versions.observation`도 실제 버전 |
| `backend.make_local_backend / make_isaac_backend` | 설정에 프로필이 있으면 백엔드마다 새 `LiveNavigation`. 다른 프로필을 명시하면 거부 |
| `experiment.config_run_root / run_root_for` | /4 설정은 `<과제 폴더>_nav4_<프로필 해시 8자>`만 쓴다. 등록된 /3 폴더나 다른 /4 폴더는 거부 |
| `monitor.snapshot` | /4 폴더도 `<과제> obs/4 <해시>`로 표시 |
| `config/current_semantics/spatial_view_contact_v1_nav4_{wmon,kalman}.yaml` | v1 + `navigation:`. WMON은 §2의 SPSA 재보정 θ, 칼만은 q=31.6 |

해시(관측 / 알고리즘 / 폴더):

| 설정 | observation_hash | algorithm_hash | 결과 폴더 |
|---|---|---|---|
| v1 (등록 /3) | `d099a08e…` | `70ebb7e0…` | `spatial_current_semantics_v1` |
| v1_nav4_wmon | `f249247e…` | `511e320a…` | `spatial_current_semantics_v1_nav4_5aa4091b` |
| v1_nav4_kalman | `4d6bb0b0…` | `13f1f426…` | `spatial_current_semantics_v1_nav4_489abc2d` |

task/g3/g4/calibration/action 해시는 세 설정이 같다. 그래서 개발 계획 `development_v1.yaml`(같은 학습·평가
시드)을 그대로 등록하면 /3 development/1과 에피소드 단위로 짝지은 비교가 된다.

등록 체인(각 /4 폴더, `./run.sh current-semantics-3d --config <설정> --stage …`):
audit → fixtures(G2, 7/7) → coupling --backend local(G3 로컬, /3과 같은 검사 전부 통과) →
G3 Isaac 절반은 task 6/7과 같은 관례로 과제 5의 `coupling_isaac.json`을 복사(같은 g3_hash, 같은
2026-10-28 비행; 이유는 각 폴더 `PROVENANCE.json`) → learnability(G4 파일럿) → register-development → develop.

### G4 학습 가능성 파일럿 (g4-pilot/7, raw_ppo, 모델 시드 0·1, 600 업데이트, 기준 결정론 ≥ 0.5 · 확률 ≥ 0.25)

| 관측 | 판정 | 시험 결정론 (0 / 1) | 시험 확률 (0 / 1) | 검증 최고 (결정론, 확률) | 소요 |
|---|---|---|---|---|---|
| /3 (기존, 10-29) | 통과 | 0.94 / 0.85 | 0.90 / 0.77 | 1.00, 1.00 / 1.00, 0.92 | 850 s / 1,562 s |
| **/4 칼만** | **통과** | **0.94 / 0.92** | 0.90 / 0.75 | 1.00, 1.00 / 1.00, 1.00 | 1,620 s / 1,063 s |
| /4 WMON | 통과(근소) | 0.54 / 0.54 | 0.58 / 0.50 | 0.71, 0.67 / 0.71, 0.50 | 3,802 s / 3,374 s |

- 칼만 항법은 학습 가능성을 /3 수준 그대로 유지한다. WMON 항법은 기준을 겨우 넘는다.
- WMON 파일럿이 2.3배 오래 걸린 것은 계산 비용 때문이 아니다(샘플당 79 µs vs 칼만 56 µs). 착륙을 덜 해서
  에피소드가 길어진 탓이다.

### 개발 등록 (development/1, 2026-11-10 12:28 UTC)

| 폴더 | plan_hash | 학습 시드 | 상태 |
|---|---|---|---|
| `…_nav4_489abc2d` (칼만) | `8cb5dc75…` | 10–14 | REGISTERED, develop 실행 중(먼저) |
| `…_nav4_5aa4091b` (WMON) | `58bd9206…` | 10–14 | REGISTERED, 칼만 다음 순서 |

개발 학습은 세션과 분리(`setsid nohup`)해 칼만 → WMON 순으로 돈다. 로그는 각 폴더의
`development.log`, 결과는 `development.json`, 실시간 대시보드는 http://127.0.0.1:8773/
(/3 development/1은 8.3시간. 칼만은 비슷, WMON은 에피소드가 길어 더 걸릴 것으로 예상).
### 개발 결과: /4 칼만 (2026-11-11 10:12 KST 완료, 12.7 h)

판정: **development gate 실패** (`learnability_passed` false, `development_gate_passed` false).

- **학습 가능성:** `semantic_mlp_ppo` 시드 10·13이 600 업데이트 내내 난이도 0.0에서 성공 0.00이었다.
  커리큘럼이 진행되지 않아 체크포인트가 없다(탐색 실패). 나머지 18개 모델은 검증 결정론 0.79–1.00.
  /3에서는 20개 모두 학습했다.
- **비교 마진:** MLP 대비 +0.342(통과, 실패한 MLP 2개 때문에 커짐), Untyped 대비 +0.064(통과),
  Raw 대비 −0.008(**실패**). 가드: ID·OOD 성공은 통과, OOD unsafe vs raw +0.020(**실패**).

arm별 평균 성공(10개 셀) / ID / OOD / unsafe, /3 → /4 칼만:

| arm | /3 | /4 칼만 |
|---|---|---|
| raw_ppo | 0.756 / 0.888 / 0.624 / 0.206 | 0.730 / 0.884 / 0.577 / 0.231 |
| semantic_mlp_ppo | 0.772 / 0.914 / 0.631 / 0.195 | 0.455 / 0.545 / 0.365 / 0.139 (시드 2개 미학습) |
| untyped_gnn_ppo | 0.756 / 0.886 / 0.626 / 0.185 | 0.708 / 0.857 / 0.558 / 0.235 |
| ontology_rgat_ppo | 0.786 / 0.932 / 0.640 / 0.188 | 0.732 / 0.884 / 0.580 / 0.247 |

- GNSS 결측이 들어가면 학습된 정책 모두 성공이 3–6 pp 떨어지고 unsafe가 늘었다. 가장 크게 떨어진 셀은
  noise_ood(예: rgat 0.74 → 0.54)와 dropout·combined_id다.
- /3에서 가장 좋던 ontology_rgat_ppo(0.786)는 /4에서 raw와 같은 수준(0.732 vs 0.730)이 됐다.
  ObservationQuality 노드에 실시간 항법 품질이 들어와도 그래프 arm의 이점이 생기지 않았다.

### 개발 결과: /4 WMON (2026-11-12 03:19 KST 완료, 17.1 h)와 세 관측 비교

판정: **development gate 통과**, 20개 모델 전부 학습(검증 결정론 0.50–1.00). confirmation_opened(로컬 플랜트).
Isaac 부분(guide 11.6)은 사용자 승인이 필요해 아직 하지 않았다.

| 판정 | /3 | /4 칼만 | /4 WMON |
|---|---|---|---|
| 학습 가능성 | 20/20 | 18/20 (MLP 10·13 미학습) | 20/20 |
| 마진 vs MLP / Untyped / Raw | +0.018 / +0.024 / +0.040 | +0.342 / +0.064 / **−0.008** | +0.148 / +0.238 / +0.108 |
| 가드 OOD unsafe vs raw | −0.005 | **+0.020** | −0.003 |
| development gate | 통과 | **실패** | 통과 |

평균 성공(10개 셀), 학습된 모델만:

| arm | /3 | /4 칼만 | /4 WMON |
|---|---:|---:|---:|
| ontology_rgat_ppo | 0.786 | 0.732 | 0.696 |
| raw_ppo | 0.756 | 0.730 | 0.625 |
| semantic_mlp_ppo | 0.772 | 0.759 (n=3) | 0.591 |
| untyped_gnn_ppo | 0.756 | 0.708 | 0.503 |

- 절대 성능은 모든 arm에서 /3 > /4 칼만 > /4 WMON이다. 짝지은 (시드, arm) 20쌍에서 WMON − 칼만 평균
  −0.052, WMON이 나은 것은 5쌍. 그중 2쌍은 칼만에서 학습에 실패한 MLP다.
- WMON에서 게이트가 통과한 것은 rgat가 좋아져서가 아니다. 다른 arm이 더 크게 떨어졌다(rgat −0.036,
  raw −0.105, untyped −0.205 vs 칼만). 항법이 나빠질수록 온톨로지 arm의 상대적 강건성이 드러난다는 뜻으로
  읽을 수 있지만, 시드 5개 평균 마진 기준이고 확인(confirmation) 실험 전이다.
- 칼만에서 게이트가 실패한 주원인은 MLP 시드 2개의 탐색 실패와 raw 대비 마진(−0.008)이다.

### Isaac 개발 평가: /4 WMON (isaac-development/1, 2026-11-12 10:24 등록, 22:55 완료)

/3과 같은 계획·케이스·시드(composition_id, composition_ood, combined_id × 환경 시드 10 × 모델 20 = 600,
결정론). 사용자 승인 후 자체 Isaac/PX4 스택(헤드리스)에서 비행. 비행 7.8 h. 241번째 에피소드 뒤 PX4 인프라
failsafe(battery_unhealthy 등) 1회로 판정 없이 멈췄고, /3 때와 같은 절차로 판정된 241개를 유지한 채 새
스택에서 재개(`isaac_development_part1.log`, `interruptions.jsonl`). 이 단계에는 통과 기준이 없다
(guide 11.6.5, 통합 측정).

세 조건 평균 성공(unsafe):

| arm | /4 WMON Isaac | /4 WMON 로컬 (같은 케이스) | /3 로컬 (같은 케이스) | /3 Isaac (10-30, 핸드오버 수정 전) |
|---|---:|---:|---:|---:|
| ontology_rgat_ppo | **0.960 (0.033)** | 0.833 | 0.947 | 0.553 |
| semantic_mlp_ppo | 0.847 (0.120) | 0.740 | 0.947 | 0.540 |
| raw_ppo | 0.827 (0.153) | 0.753 | 0.880 | 0.547 |
| untyped_gnn_ppo | 0.707 (0.207) | 0.640 | 0.960 | 0.520 |

- 전체 Isaac: SUCCESS 501/600, UNSAFE_CONTACT 60, TIMEOUT 22, MISSED 12, ENVELOPE 5.
  /3 수정 전 비행은 ENVELOPE 203(핸드오버 IMU 랩)이라 직접 비교하지 않는다.
- 로컬과 짝: 둘 다 성공 406, Isaac만 95, 로컬만 39, 둘 다 실패 60. /4에서는 Isaac이 로컬보다 높다.
  원인은 확인하지 않았다(로컬 플랜트의 광학 잡음·cadence 모델이 더 가혹할 가능성).
- /3 수정 후 재비행 80개(isaac_handover_fix_2)와 같은 케이스: /4 WMON 66/80, /3 74/80.
  arm별 rgat 20/20 vs 18/20, raw 15 vs 18, mlp 17 vs 18, untyped 14 vs 20.
- **해석:** GNSS 결측과 WMON 항법 아래에서 ontology_rgat는 Isaac에서도 /3 수준을 유지했다(0.96, unsafe
  3 %). 다른 arm은 떨어졌다. 로컬 개발 게이트에서 본 "항법이 나빠질수록 rgat의 상대적 강건성"이 Isaac
  비행에서도 같은 방향으로 나타났다. 시드 5개, 조건당 환경 10개 규모의 측정이다.

### isaac-development/2: /3 수정 후 재비행과 /4 칼만 비행 (2026-11-13 등록, 11-15 13:45 완료)

- 새 프로토콜 `isaac-development/2`(`config/current_semantics/isaac_development_v2.yaml`,
  `isaac_development.py: PROTOCOLS`). 케이스·시드는 /1과 같아야 하고(로더가 확인), 산출물은
  `isaac_development_2_registration.json`, `isaac_development_2.json`, `isaac_development_2/`로 따로 쓴다.
  /1 기록은 건드리지 않는다. 게이트 통과 대신 개발 절차 완료를 요구하고, best.pt가 없는 모델은 등록 때
  `excluded_models`로 남긴다.
- CLI: `--stage register-isaac-development|isaac-development --plan …/isaac_development_v2.yaml`.
- 등록: /3 (`ed494edd…`, 모델 20, 600 에피소드, 게이트 통과) · /4 칼만 (`4487a469…`, 모델 18, 540
  에피소드, 게이트 실패, 제외 10/·13/semantic_mlp_ppo).
- 비행: /3 → /4 칼만 순서로 세션과 분리해 실행. PX4 인프라 중단(failsafe, 게이트웨이 시간 초과,
  비행 시계 정지)일 때만 같은 절차로 재개. 진행 기록은 `results/current_semantics_3d/isaac_development_2_chain.log`.
  bond 학습과 같은 PC에서 돌 때 Isaac이 CPU를 빼앗겨 시계 정지·게이트웨이 시간 초과가 났고, 11-14 17:46부터
  Isaac 체인은 CPU 0–5, 학습은 CPU 6–19로 완전히 분리했다(결정 주기 9.87 Hz 정상).
- **/3 결과 (11-15 완료, 600/600, 비행 6.1 h, 인프라 중단 6회 후 재개):** SUCCESS **592/600**. 수정 전
  비행(10-30)은 같은 케이스에서 324/600이었다. arm별 성공(unsafe): raw 0.987(0.000), mlp 0.987(0.013),
  untyped 0.980(0.013), rgat 0.993(0.007). 같은 케이스 로컬과 짝: 둘 다 성공 557, Isaac만 35, 로컬만 3.
- **같은 600개 케이스 짝 비교:** /4 WMON 501 vs /3 592. arm별 rgat 144 vs 149, raw 124 vs 148,
  mlp 127 vs 148, untyped 106 vs 147. WMON 항법의 손실은 rgat에서 가장 작다.
- **/4 칼만 결과 (11-15 13:45 완료, 540/540, 인프라 중단 3회 후 재개: 비행 시계 정지 2, PX4 failsafe 1):**
  SUCCESS **513/540**. ENVELOPE 17, UNSAFE_CONTACT 6, MISSED 3, TIMEOUT 1. 같은 케이스 로컬과 짝: 둘 다 성공
  456, Isaac만 57, 로컬만 11, 둘 다 실패 16.

#### 세 항법의 Isaac 비교 (같은 540개 케이스)

칼만 폴더에 없는 MLP 시드 10·13을 뺀 540개 케이스(조건 3 × 환경 시드 10 × 모델 18)에서 짝지어 비교했다.
/4 WMON은 isaac-development/1로 11-12에 비행했다. 스택(핸드오버 수정 후)과 케이스·시드는 같고, 산출물
이름만 다르다. 스크립트: `python tools/isaac_navigation_compare.py`(episodes.jsonl 세 개를 읽는다).

성공 수(unsafe 수):

| arm (케이스) | /3 참값 항법 | /4 칼만 | /4 WMON |
|---|---:|---:|---:|
| ontology_rgat_ppo (150) | 149 (0) | 145 (3) | 144 (2) |
| raw_ppo (150) | 148 (0) | 142 (7) | 124 (19) |
| semantic_mlp_ppo (90) | 88 (2) | 86 (4) | 72 (14) |
| untyped_gnn_ppo (150) | 147 (1) | 140 (9) | 106 (26) |
| **합계 (540)** | **532 (3)** | **513 (23)** | **446 (61)** |
| composition_id (180) | 180 | 179 | 150 |
| composition_ood (180) | 175 | 172 | 143 |
| combined_id (180) | 177 | 162 | 153 |

짝지은 불일치(앞쪽만 성공 / 뒤쪽만 성공, 정확 이항 p):

| 비교 | 전체 | rgat | raw | mlp | untyped |
|---|---|---|---|---|---|
| 칼만 vs /3 | 6 / 25, p = 0.0009 | 1 / 5 | 1 / 7 | 1 / 3 | 3 / 10 |
| WMON vs /3 | 4 / 90, p < 10⁻²¹ | 0 / 5 | 2 / 26 | 1 / 17 | 1 / 42 |
| 칼만 vs WMON | 84 / 17, p < 10⁻¹¹ | 5 / 4 | 24 / 6 | 15 / 1 | 40 / 6 |

- Isaac 폐루프에서도 순서는 로컬과 같다: **/3 > /4 칼만 > /4 WMON.** 칼만 항법의 손실(−19)은 작지만
  통계적으로 유의하다. 손실은 주로 combined_id(블랙아웃·패드 가속·GNSS 결측이 겹치는 조건)에 몰려 있다.
  WMON 항법의 손실(−86)은 그 4.5배이고, 세 조건 모두에서 난다.
- **ontology_rgat_ppo는 세 항법에서 149 / 145 / 144로 거의 변하지 않은 유일한 arm이다.** 칼만과 WMON의
  차이가 rgat에서는 사라진다(5 / 4). 다른 arm은 WMON 항법에서 크게 떨어진다(untyped 140 → 106).
  항법이 나빠질수록 rgat의 상대적 강건성이 커지는 것은 로컬 개발 게이트와 같은 방향이다. 시드 5개,
  조건당 환경 10개 규모의 측정이다.
- WMON에 대한 결론은 바뀌지 않는다. Isaac 폐루프에서도 같은 센서의 칼만이 WMON보다 낫다.

중단: `pkill -f "stage develop --config .*nav4"`. 같은 명령으로 다시 실행하면 끝난 모델(시드×arm)은 건너뛰고
나머지부터 이어간다(`run_development`: resumable per model).

## 0. 결론

1. **21D의 항법 채널을 살렸다.** /4에서 `drone_navigationValid`·`drone_navigationAge`는 GNSS 결측에 따라
   변하고, 21D의 자기 위치·속도와 패드 추적기의 촬영 시각 자기 위치는 항법 추정기의 출력이다.
   채널 수·순서·인코딩은 /3과 같다. /4 매니페스트는 별도 `observation_hash`/`algorithm_hash`를 낸다.
2. **같은 센서(GNSS 위치 + 속도 + IMU 가속도)에서는 일반 등가속도 칼만 필터가 WMON보다 정확하다.**
   자기 위치 0.106 vs 0.158 m(재보정 WMON), 폐루프 착륙은 칼만이 참값 항법과 같은 6/12, WMON 5/12.
   이 프로젝트의 자기 항법, WMON 자체 벤치마크(MATLAB `simulateEpisode`), OmniDrones 설정 모두에서
   그렇고, 잡음 버스트·이상치·IMU 바이어스를 넣어도 바뀌지 않았다(§3–§4).
   학습된 정책을 Isaac Sim + PX4에서 비행해도 같다(§6, 같은 540개 케이스): 참값 항법 532, 칼만 513,
   WMON 446 성공.
3. WMON은 결측을 단순 유지(hold)하는 것보다는 훨씬 낫다. 연구 주장 중 "결합이 결측을 메운다"는 재현되지만,
   "기존 상태추정보다 낫다"는 지금 데이터로는 지지되지 않는다.

## 1. 무엇을 바꿨나

| 파일 | 내용 |
|---|---|
| `current_semantics/navigation.py` | `NavigationDegradation`(GNSS 주기·잡음·결측 창, 속도·IMU 잡음), `NavigationEstimator`(hold/kalman/wmon), `LiveNavigation`(에피소드 시드로 결측 창 결정, 추정기 이력으로 촬영 시각 자기 위치 보간, 항법 채널 상태) |
| `current_semantics/backend.py` | `CurrentObserver(navigation=)`, `CurrentSpatialBackend/make_local_backend/make_isaac_backend(navigation=)`. `_ingest`에서 블랙아웃·핸드오버 오차 다음에 항법을 적용. `scientific=True`는 거부 |
| `current_semantics/contracts.py` | `observation_manifest(navigation=None)`, `contract_manifest(config, *, navigation=None)` — None이면 기존과 바이트 단위 동일 |
| `spatial/kinematic_kf.py` | 등가속도 칼만(백색 저크), 위치 결측 시 해당 행 생략 |
| `spatial/wmon.py` | `WMONStream` — `infer_episode`와 같은 산술을 한 샘플씩, 가변 dt |
| `tools/live_navigation_eval.py` | 비행 기록 → 추정기별 재생(같은 GNSS 잡음·결측) → (선택) WMON SPSA 재보정 → 폐루프 판정 |
| `tools/wmon_vs_kalman_benchmark.py`, `tools/wmon_episodes.m` | WMON 자체 벤치마크에서 WMON vs 칼만 |
| `tests/current_semantics/test_live_navigation.py` | 11개: 스트림=MATLAB, 칼만 정확성, 매니페스트 분리, 결측·채널, 거부 규칙, 로컬 비행 |

센서 모델(기본값): GNSS 10 Hz, σ = (0.3, 0.3, 0.5) m, 에피소드마다 시작 2–6 s · 길이 1–4 s 결측 1회;
속도 σ 0.05 m/s(광류/VIO 대용); IMU = 기준 속도 차분 + σ 0.2 m/s²(`Measurement`에 비력 필드가 없음).
기준 자기 상태는 로컬 플랜트의 참값(Isaac에서는 PX4 EKF 출력)이다.

## 2. 자기 항법 결과 (로컬 /12, `in_distribution`)

실행: `python tools/live_navigation_eval.py --episodes 36 --tune-epochs 40 --closed-loop`
(시드 920000–920035: 학습 18 / 검증 6 / **시험 12**), 산출물 `results/live_navigation_20261110/` (15.6 min).
모든 추정기는 같은 비행 기록과 같은 GNSS 잡음·결측(에피소드 시드)으로 재생했다. 칼만 q는 학습+검증
격자(31.6), WMON은 MATLAB θ와 같은 데이터로 SPSA 40 epoch 재보정한 θ.

시험 12회 RMSE (자기 위치 [m] / 자기 수평 속도 [m/s] / 21D 상대 위치 [m]):

| 항법 | 전체 | GNSS 결측 중 | 결측 후 1.5 s |
|---|---|---|---|
| 참값 (등록 /3) | 0 / 0 / 0.219 | 0 / 0 / 0.016 | 0 / 0 / 0.016 |
| hold (마지막 GNSS) | 0.800 / 0.071 / 45.7 | 1.733 / 0.068 / 0.360 | 0.662 / 0.070 / 1.958 |
| **칼만** | **0.106 / 0.015 / 0.262** | **0.115 / 0.013 / 0.017** | **0.117 / 0.014 / 0.025** |
| WMON (MATLAB θ) | 0.174 / 0.022 / 1.045 | 0.179 / 0.021 / 0.027 | 0.167 / 0.023 / 0.088 |
| WMON (SPSA 재보정) | 0.158 / 0.026 / 0.700 | 0.141 / 0.025 / 0.021 | 0.136 / 0.026 / 0.053 |

폐루프 (같은 시험 12시드, 스크립트 PD가 각 항법으로 비행):

| 항법 | SUCCESS | 나머지 |
|---|---:|---|
| 참값 (등록 /3) | 6 | TIMEOUT 1, ENVELOPE 3, UNSAFE 2 |
| **칼만** | **6** | 참값과 판정 분포가 같다 |
| WMON (SPSA 재보정) | 5 | TIMEOUT 2, ENVELOPE 3, UNSAFE 2 |
| WMON (MATLAB θ) | 3 | TIMEOUT 2, ENVELOPE 3, UNSAFE 2, MISSED 2 |
| hold | 0 | MISSED 4, ENVELOPE 8 |

- 21D 상대 위치 "전체"는 시야 상실 구간이 지배한다(참값 항법도 0.219 m). hold는 10 Hz GNSS 계단이
  패드 추적기의 혁신 게이트를 계속 넘겨 추정이 발산한다(45.7 m).
- GNSS 결측 중 상대 위치 오차가 모두 작은 것은 결측 창(이륙 후 2–6 s 시작)에 패드가 보여 상대 위치를
  카메라가 직접 주기 때문이다. 항법 오차는 자기 위치·속도와 시야 상실 외삽에 영향을 준다.
- `observation_hash`: /3 `d099a08e…`, /4(이 설정) `c68b0778…`.

## 3. WMON 자체 벤치마크에서 WMON vs 칼만

`python tools/wmon_vs_kalman_benchmark.py --retrain-epochs 300` — MATLAB R2024b가 `simulateEpisode`로 만든
69개 에피소드(학습 1–12, 검증 501–503, 시험 504–507, 확장 시험 508–557). 두 추정기 모두 같은
학습+검증으로 조정: 칼만은 q와 R 배율 2개, WMON은 7계수(MATLAB 결과 + 여기서 300 epoch·재시작 3회 재학습).
파이썬 WMON은 69개 전부에서 MATLAB과 1.4e-14 이내로 일치.

| 확장 시험 50개 | 손실 | 위치 RMSE [m] | 속도 RMSE [m/s] | 결측 중 위치 [m] |
|---|---:|---:|---:|---:|
| WMON (MATLAB θ) | 0.0142 | 0.108 | 0.068 | 0.110 |
| WMON (재학습) | 0.0134 | 0.105 | 0.066 | 0.109 |
| **칼만 (등가속도)** | **0.0098** | **0.089** | **0.059** | **0.092** |
| raw / hold | 1.257 | 1.031 | 0.184 | 1.750 |

에피소드별: 칼만이 50개 중 41개에서 위치 RMSE가 낮다. MATLAB 시험 4개만 보면 0.081 vs 0.100.

열화 주입(확장 시험 50개, 칼만은 깨끗한 데이터로 조정한 그대로):

| 조건 | WMON | 칼만 | WMON이 나은 에피소드 |
|---|---:|---:|---:|
| 깨끗함 | 0.108 | 0.089 | 9/50 |
| 위치 잡음 약 3배 버스트 (3 s) | 0.154 | 0.119 | 5/50 |
| 위치 이상치 5 % (±2 m) | 0.154 | 0.129 | 8/50 |
| 속도 잡음 약 3배 버스트 (3 s) | 0.141 | 0.136 | 21/50 |
| IMU 바이어스 0.5 m/s² | 0.359 | 0.236 | 0/50 |

## 4. 해석

- WMON의 L4 결합 에너지 최소화는 한 스텝짜리 가중 최소제곱이다. 칼만은 같은 운동학 모델에 공분산을 시간에
  걸쳐 전파하므로 같은 측정에서 더 많은 정보를 쓴다. FFT 신뢰도 가중은 잡음 수준이 바뀌는 조건에서도
  이 차이를 메우지 못했다.
- 연구를 이어가려면 칼만(또는 적응 R 칼만)을 기준선으로 두고, WMON이 이기는 조건을 먼저 찾는 것이
  순서다. 후보: 센서 잡음 모델을 모르는 상황(칼만 R 오지정), 입자 생성/소멸이 필요한 가변 차원 문제.

## 5. 사용법

```python
from ontology_rgat.current_semantics.backend import make_local_backend
from ontology_rgat.current_semantics.contracts import contract_manifest, load_config
from ontology_rgat.current_semantics.navigation import (LiveNavigation, NavigationDegradation,
                                                        NavigationEstimator)
from ontology_rgat.spatial.kinematic_kf import KalmanNoise

config = load_config()
nav = LiveNavigation(NavigationDegradation(),
                     NavigationEstimator("kalman", kalman=KalmanNoise(0.37, 0.05, 0.2, 31.6)))
backend = make_local_backend(config, scenario="in_distribution", navigation=nav)
manifest = contract_manifest(config, navigation=nav.manifest_part())   # /4 해시
```

```bash
python tools/live_navigation_eval.py --episodes 36 --tune-epochs 40 --closed-loop
python tools/wmon_vs_kalman_benchmark.py --retrain-epochs 300
```
