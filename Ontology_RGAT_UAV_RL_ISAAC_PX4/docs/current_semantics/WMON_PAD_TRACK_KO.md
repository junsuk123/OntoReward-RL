# WMON 패드 추적기 (옵트인 연구 옵션)

> **명칭 안내:** 이 문서의 WMON(Wave-to-Matter Ontological Network)은 저자가 별도 연구에서 개발한 자기 항법 추정기의 내부 명칭이다. 기술적으로는 인과적 FFT 전처리, 주파수 기반 측정 신뢰도, 운동학적 일관성 잔차를 쓰는 가중 최소제곱 상태 추정기이며, 계수는 SPSA + Adam으로 보정한다. README와 그림에서는 *FFT-Assisted Kinematic Weighted Least Squares (WLS) Estimator*(약칭 Kinematic WLS)로 표기한다. 착륙 정책 연구에서는 실험 환경의 구성 요소로만 쓰며, 본 연구의 기여로 주장하지 않는다. 이 문서는 실험 기록이므로 내부 명칭과 수치를 그대로 둔다.

작성일: 2026-11-10 (KST)  
원본 연구: `~/Downloads/wmon /run_WMON_demo.m` (WMON v0.1, Wave-to-Matter Ontological Network)  
상태: **옵트인 비교 옵션. 등록 관측 계약(`spatial-current-observation/3`, ABG 추적기)은 바뀌지 않았다.**
계약 해시(observation/feature/graph/algorithm 등)는 HEAD와 동일함을 확인했다.

## 0. 결론

- WMON 핵심(FFT 파동 → 입자 신뢰도 → 위치·속도·가속도 원자 → 결합 에너지 최소화 → SPSA+Adam 7계수)을
  파이썬으로 이식했고 MATLAB R2024b 출력과 **최대 7e-15** 차이로 일치한다.
- 그 원자를 `Measurement` 스트림의 **패드(UGV) 추적기**로 옮긴 `WMONSpatialTrack`을
  `CurrentObserver(track_factory=...)`로 끼울 수 있다. 기본값은 여전히 ABG 추적기다.
- 로컬 /12 비행 재생 비교에서 **등록 ABG(+재획득) 추적기가 WMON보다 같거나 낫다.** SPSA 보정도 이를
  바꾸지 못했다(아래 §3). 원인은 구조적이다: WMON은 **측정된** 속도·가속도 입자를 결합으로 적분해
  위치 결측을 메우는데, 패드에는 속도·가속도 센서가 없다.
- WMON이 맞는 자리는 **기체 자신의 항법(GNSS/EKF 위치 결측 + IMU 가속도 + 속도)**이다. 이 경로의
  21D 관측은 `drone_navigationValid=1, age=0`으로 고정(backend.NAVIGATION_NOTE)이라, 이를 쓰려면
  21D 스키마 개정이 필요하다 — 사용자 결정 사항.

## 1. 파일

| 파일 | 내용 |
|---|---|
| `python/ontology_rgat/spatial/wmon.py` | 차원 일반 numpy 코어: `WMONParams`(MATLAB theta 인코딩), `wave_field`, `solve_atoms`, `infer_episode`(MATLAB `inferEpisode`), `episode_loss`, `spsa_adam` |
| `python/ontology_rgat/spatial/wmon_track.py` | `WMONSpatialTrack`, `WMONTrackSettings`(JSON 저장/로드), `wmon_track_factory` |
| `python/ontology_rgat/current_semantics/backend.py` | `CurrentObserver`, `CurrentSpatialBackend`, `make_local_backend`, `make_isaac_backend`에 `track_factory=None` 인자. 교체 시 provenance에 `pad_estimator`·`pad_estimator_diagnostics` 기록, `scientific=True`면 거부 |
| `tools/wmon_pad_track_eval.py` | 비행 기록 → ABG/WMON 재생 비교 → (선택) SPSA 보정 → (선택) 폐루프 판정 |
| `tests/current_semantics/test_wmon_pad_track.py` | MATLAB 대조, 잠김 없음, 블랙아웃 통과, freshness 규칙, 계약 불변, scientific 거부 (22개) |
| `tests/fixtures/wmon_matlab_reference.npz`, `make_wmon_fixture.m` | MATLAB 기준 데이터와 재생성 스크립트 |

## 2. MATLAB → 착륙 문제 대응

| WMON 층 | MATLAB 프로토타입 | `WMONSpatialTrack` |
|---|---|---|
| 위치 입자 | GNSS 위치(간헐 결측) | 신선한 광학 해: 패드 = 촬영 시각 자기 위치 − (자기−패드) 벡터. FOV 상실·cadence·블랙아웃이 결측 |
| 속도 입자 | 속도 센서 | 창 안 최근 `velocity_span`개 위치 입자의 최소제곱 기울기(파생, 2프레임 전엔 없음) |
| 가속도 원자 | 가속도 센서 | 센서 없음 → `a_prev·exp(-gap/τ)` 지속 사전(τ=1.5 s, ABG와 동일), 신뢰도 1 |
| L0 파동장 | 최근 12샘플 FFT | 최근 12**프레임** FFT; 0.5 s 넘는 공백 뒤 창 해체(dissolution) |
| L4 결합 | p−v, v−a (dt=0.1) | 같은 두 결합, 가중치 `k·(0.1/gap)^exponent` (기본 exponent 1) |
| 게이트 | 없음 | 없음 — 어긋난 프레임은 고주파 비율이 올라 스스로 가중치가 낮아진다 |

출력은 마지막 수용 프레임의 촬영 시각에 있는 원자를 결정 시각까지 감쇠 가속도 닫힌식으로 전파한다
(capture-aligned, ABG와 같은 시각 의미).

## 3. 측정 결과 (로컬 /12, `in_distribution`, 평가 블랙아웃 확률 1 · 시작 3–6 s · 길이 0.5–2 s)

실행: `python tools/wmon_pad_track_eval.py --episodes 36 --calibrate --epochs 60 --closed-loop`
(시드 910000–910035: 학습 18 / 검증 6 / **시험 12**), 산출물 `results/wmon_pad_track_20261110/`.
ABG 재생은 온라인 관측기를 차이 0으로 재현했다(도구가 매번 확인).

상대 위치 RMSE [m] / 수평 상대 속도 RMSE [m/s], 시험 12회:

| 구간 | 결정 수 | ABG (등록) | WMON (MATLAB 계수) | WMON (SPSA 보정) |
|---|---:|---|---|---|
| 패드 보임 (tracking) | 3162 | **0.025** / 0.087 | 0.038 / 0.083 | 0.040 / 0.083 |
| 블랙아웃 중 | 118 | **0.043** / 0.035 | 0.058 / 0.044 | 0.057 / 0.044 |
| 블랙아웃 후 1.5 s | 135 | 0.022 / 0.036 | 0.019 / 0.049 | **0.018** / 0.046 |
| 시야 상실 (프레임 나이 > 0.5 s) | 1482 | **0.397** / 0.018 | 1.355 / 0.059 | 1.281 / 0.056 |
| 전체 | 4897 | **0.219** / 0.071 | 0.746 / 0.075 | 0.706 / 0.074 |

- 폐루프(같은 시험 12시드, 같은 PD): SUCCESS 3 / 3 / 2. 나머지 판정 분포도 거의 같다.
- SPSA 보정: 학습 손실은 0.077 → 0.045로 내려갔지만 검증 손실은 0.027 → 0.045로 올라 **1 epoch가
  선택**됐다. 이 센서 계약에서 7계수 조정으로 얻을 것이 없다는 뜻이다.
- 별도 24회(시드 910000–910023) 에피소드별 쌍대 비교(exponent 1): ABG가 더 나은 에피소드 수는 패드
  보임 21/24, 블랙아웃 17/22, 시야 상실 12/19.
- 결합 dt 지수: exponent 2(단위 정규화)는 패드 보임 0.073 m로 exponent 1(0.040 m)보다 나빠 기본값을 1로 둔다.
- WMON이 앞서는 곳은 블랙아웃 직후 재수렴(게이트가 없어 잠김이 없음)과 패드 보임 중 속도뿐이며 차이는 작다.
- 시야 상실 오차는 상실 시점 속도 오차 × 경과 시간으로 커진다. WMON의 속도 입자(4프레임 기울기)가
  FOV 경계에서 흔들려 상실 시점 속도가 ABG보다 부정확하다.

## 4. 사용법

```python
from ontology_rgat.current_semantics.backend import make_local_backend
from ontology_rgat.current_semantics.contracts import load_config
from ontology_rgat.spatial.wmon_track import WMONTrackSettings, wmon_track_factory

config = load_config()
backend = make_local_backend(config, scenario="in_distribution",
                             track_factory=wmon_track_factory(WMONTrackSettings()))
# Isaac: make_isaac_backend(config, allow_isaac=True, track_factory=...) — scientific=False 경로만
```

```bash
# 재생 비교만 (약 15 s)
python tools/wmon_pad_track_eval.py --episodes 12
# 보정 + 폐루프 (약 4 min)
python tools/wmon_pad_track_eval.py --episodes 36 --calibrate --closed-loop
# 보정된 계수로 다시 평가
python tools/wmon_pad_track_eval.py --settings results/<run>/wmon_track_settings_calibrated.json
```

## 5. 주의와 남은 일

- 재생 비교는 **개방 루프**다: 비행은 등록 ABG 관측을 읽는 스크립트 PD가 조종했고, 같은 센서 스트림을
  두 추정기에 넣어 추정 차이만 본다. 폐루프 판정은 같은 PD를 각 추정기로 다시 비행한 결과다.
- PD는 등록 교사가 아니다. 착륙 성공률 수치를 정책 성능으로 인용하지 않는다.
- 시야 상실(lost view) 오차는 몇 개의 장시간 상실 에피소드가 지배해 시드 집합마다 순위가 뒤집힌다.
  에피소드별 쌍대 비교로 읽는다.
- WMON 학습 체크포인트를 만들려면 관측 계약 개정과 등록 절차가 먼저다(`scientific=True`는 거부됨).
- 기체 항법 열화(GNSS 결측 + IMU)에 WMON을 적용한 결과와 칼만 비교는 `WMON_LIVE_NAVIGATION_KO.md`
  (옵트인 관측 /4, 2026-11-10).
