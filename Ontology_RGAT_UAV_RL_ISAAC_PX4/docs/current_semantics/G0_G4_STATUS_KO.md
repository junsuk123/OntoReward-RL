# spatial_current_semantics_v1 — 구현 상태와 G0–G4 결과

기록일: 2026-10-28 (KST), 같은 날 4차 갱신(공통 보상·관측 재구성과 G4 pilot 3 후)  
기준: HEAD `cab3dd2` + 같은 날 작업 트리(커밋되지 않은 사용자 변경 포함)  
근거 문서: `CURRENT_SEMANTICS_3D_REFACTOR_GUIDE_KO.md`(가이드), `SOURCE_AUDIT_KO.md`(감사)  
MATLAB은 실행하지 않았다. 학습은 local 보정 plant에서 G4 pilot 세 번(Raw arm만)만 했다. Isaac/PX4는 사용자 승인 후 owned stack에서 두 번 비행했다. G3 검사 6 episode와, pilot 3 선택 checkpoint 평가 20 episode다(§10.6, 기준 미달 4/10·3/10). arm 비교는 하지 않았다.

## 1. 사용자 결정

| 날짜 | 결정 | 반영 |
| --- | --- | --- |
| 2026-10-28 | 환경은 Isaac Sim 연계로 한다. **Isaac 기준 + 보정 local 학습**을 택한다. | 학습 plant는 `spatial-reference/12`의 `LocalBackend`(보정판)다. G3 검사와 모든 선택 checkpoint 평가는 Isaac/PX4에서 한다. direct_policy local toy는 거부한다. |
| 2026-10-28 | "제안대로 모두 수정" | §2의 다섯 항목. 광학 주기 보정은 측정 결과로 기각했다. |
| 2026-10-28 | "go": Isaac 쪽 G3 실행 승인 | owned stack에서 6 episode를 비행했다. 모두 정지가 확인됐고 stack은 정상 종료됐다. 결과는 §5. **G3 passed.** |
| 2026-10-28 | G0·G1과 G2의 두 항목(v1 2D fixture, nav 채널 상수)은 "불필요"하므로 G4 pilot을 진행한다. | gate 표에 **waived**(사용자 면제)로 기록하고, 남은 근거를 `unresolved`로 보존한다. 통과로 표기하지 않는다. local 학습 plant를 `g3_passed`로 승격했다. G4 pilot 프로토콜과 판정 기준을 실행 **전에** 등록했다(`g4_hash`). 결과는 §9. |
| 2026-10-28 | "PPO가 초기 BC 없이도 잘 착륙할 수 있게 공통 보상이나 공통 관측 입력을 이상적으로 재구성" | 판정(verdict)과 terminal 표는 그대로 두고, 공통 dense 보상(task v2)과 관측 scale(observation v2)을 바꿨다. 먼저 측정으로 설계를 확인하고, pilot 3을 등록·실행했다. §10 |

## 2. 이번에 바꾼 것

| 항목 | 변경 | 근거 |
| --- | --- | --- |
| 보상·종료 계약(가이드 §9.3) | `contracts.TaskContract` 등록: spatial-reference/12 `Evaluator`를 그대로 쓴다. terminal 표는 `landing/terminal.py`(SUCCESS 25 / TIMEOUT −12 / unsafe −40, τ 70)다. 접촉 판정은 충격 속도 기준이며, envelope는 21° / 20 m / 60 m다. dt는 실제 transition 시간이고, terminal potential은 0이다. 감독기가 없으므로 SAFE_ABORT는 일어나지 않는다. 해시는 `task_hash`로 algorithm_hash와 분리하고 checkpoint에 묶는다. | Isaac에서 실제로 비행한 계약이며 2D와 같은 terminal 표다. 새 의미 보상은 없다. |
| 강제 envelope | 위 계약에 포함된다. 기울기 21° 초과나 고도 20 m 초과면 episode가 끝난다. | 이전 backend에는 강제 한계가 없었다. |
| local 접촉 높이 | 0.12 m → **0.0672 m** | Isaac 다리 없는 첫 접촉 78건의 중앙값(0.0636–0.0732 m) |
| handover | local reset은 첫 영상이 수용될 때까지 hover로 기다린다(최대 1 s, 정책 시간 아님). | Isaac은 광학 해가 있어야 handover한다. |
| 그래프 초기화 | W_r ~ N(0, 1/d)(LeCun). 이전은 N(0, 0.1²). | 실제 결정 90개에서 g8 RMS 0.004–0.006 → **0.05–0.07**(raw21은 0.50). 규칙으로 정했고 목표 크기에 맞춰 조정하지 않았다. |
| Isaac 쪽 G3 | `coupling.run_isaac_checks`와 CLI `coupling --backend isaac --allow-isaac`를 구현했다(owned stack, 실행 중인 Isaac은 넘겨받지 않음). 승인 후 실행했고 통과했다. | §5 |
| 광학 주기 보정 | **기각**: 설정은 항등(매 프레임 유지, 추가 누락 0)이다. 도구 `LocalOpticalCadence`는 남긴다. | §4 |

## 3. Gate 상태

`./run.sh current-semantics-3d --stage report`의 실제 출력이다(run root `results/current_semantics_3d/spatial_current_semantics_v1/`).

| Gate | 상태 | 근거 | 막는 것 |
| --- | --- | --- | --- |
| G0 근거 동결 | **waived** (사용자) | 3D 기준 `cab3dd2` 고정, 3D 감사 완료 | 미해결로 보존: 2D v1 증거 묶음과 커밋 `5853781`·`d438a7d` 부재 |
| G1 2D 실패 진단 | **waived** (사용자) | — | 미해결로 보존: v1 checkpoint 부재 |
| G2 정합 | **passed** (면제 2건) | fixture 7/7, 촬영 시각 정렬 관측기, task 계약 등록, Isaac 21D 품질 채널 정합 | 면제: v1 2D fixture, nav 채널 상수 |
| G3 3D 결합 | **passed** | local 6종 통과(seed 8개), Isaac 7종 통과(seed 2개, 6 episode) | 남은 보정 차이 6건은 기록만 했다(§6). Isaac 표본이 작다. |
| G4 학습 가능성 | **passed** (pilot 3) | pilot 3(`g4-pilot/3`, task v2 + observation v2)을 등록 예산대로 완주했다. Raw PPO, BC 없음, 모델 seed 2개 × 600 update × 12 episode. 시험 seed 48개에서 seed 0은 deterministic 100 %, sampled 95.8 %이고, seed 1은 deterministic 95.8 %, sampled 91.7 %다. | 이전 계약의 pilot 1·2는 failed로 보관했다(§9, §9.2). G4는 local plant에서 판정한다. 선택 checkpoint의 Isaac 평가는 **기준 미달**이다(4/10, 3/10, §10.6). 네 arm 비교는 하지 않았다. §10.5 |

G3는 local 결과(`coupling_local.json`)와 Isaac 결과(`coupling_isaac.json`)가 모두 현재 `g3_hash`로 통과해야 `passed`가 된다. 보고 상태(가이드 §12)는 따로 두며, True인 항목은 없다.

## 4. 광학 주기 보정을 기각한 측정

감사에서 Isaac의 수용 영상 간격이 0.092/0.096 s(약 10.6 Hz)로 나왔다. 그래서 local의 20 Hz 영상을 10 Hz로 솎고 7.9 %를 더 누락시켜 맞췄다. 그 결과 G3 FOV 검사가 깨졌다. 원인을 추적하니, 패드 속도가 빠를 때 기준 ABG 추적기의 innovation이 0.25 m gate를 넘고 이후 모든 영상을 거부하는 **고착**이었다. 추정기와 새 광학 측정의 수평 간격을 같은 지표로 비교했다.

| 조건 | 간격 중앙값 / p99 | 0.3 m 초과 5프레임 연속 |
| --- | --- | --- |
| local 20 Hz | 0.027 / 0.12 m | 0 / 16 episode |
| local 10 Hz 솎음 | 0.142 / 3.6 m | 11 / 16 |
| Isaac 기존 비행(같은 추적기) | 0.013 / 0.20 m | 1 / 161 |

솎음은 local을 Isaac에서 더 멀어지게 했다. 이어서 Isaac 쪽 G3에서 관측기 수준의 주기를 직접 쟀다.

| 측정 | Isaac (관측기) | local 20 Hz |
| --- | --- | --- |
| 결정당 수용 프레임 | 0: 18 %, 1: 71 %, 2: 11 % (평균 0.93) | 대부분 2 |
| 수용 간격 | 0.092–0.096 s | 0.05 s |
| 시야 안 updated 비율 (21D) | 0.988 | 0.977 |
| 갱신 시 age 중앙값 (21D) | 0.100 s | 0.100 s |

Isaac은 실제로 약 10.4 Hz다. 결정 시점 값이 맞았고, 앞서 "결정 시점 기록으로는 알 수 없다"고 한 것은 지나친 유보였다. 그러나 정책이 읽는 21D 채널은 이미 같고, 솎으면 local 추적기만 고착된다. 그래서 `local_plant.optical_cadence`는 항등으로 둔다. 남은 차이는 추적기가 받는 갱신 수(결정당 2 vs 1)다. 같은 추적기가 local 10 Hz에서는 고착되고 Isaac 10.4 Hz에서는 그렇지 않은 이유는 밝히지 못했다(후보: 수신−촬영 지연 0.056 s vs 0.08 s, 패드 출발 방식).

## 5. Isaac 쪽 G3 결과 (2026-10-28 실행)

```bash
./run.sh current-semantics-3d --stage coupling --backend isaac --allow-isaac --dry-run   # 계획만
./run.sh current-semantics-3d --stage coupling --backend isaac --allow-isaac             # owned Isaac/PX4 시작
```

- **비행:** seed 990001·990002, seed마다 FOV 이탈 x·y와 하강 1회로 총 6 episode다. 기존 `live_stack`을 쓴다(비행 lock, 실행 중 Isaac 거부, 종료 확인).
- **검사:**
  - 축 부호: 실제 비행은 반복할 수 없어 짝 비교 대신 명령 가속도와 다음 결정 자세의 상관을 쓴다. 하한은 0.5다(감사 실측 0.686/−0.966).
  - FOV 이탈, 저고도 사각(Isaac 실제 접촉 높이), 품질 채널, 관측 순수성
  - 영상 주기: 보고 전용, §4의 판정용이다.
  - 종료와 cleanup을 분리해 기록한다.
- 이 검사 로직은 local plant에 꽂아 시험했다(테스트 포함). 축 상관은 0.84 / −0.80이었다.

**결과: `isaac_checks_passed = True`** (profile sha `af605e2b…`, `coupling_isaac.json`)

| 검사 | 결과 | Isaac 수치 |
| --- | --- | --- |
| 축 부호 | 통과 | 316 결정에서 corr(sin pitch, +a_x) 0.89, corr(sin roll, +a_y) −0.84 |
| FOV 이탈 | 통과 (4/4) | 마지막 검출이 패드 중심 경계 통과 후 0.18–0.33 m. pinhole 예측과의 차이 −0.15~+0.002 m(local −0.18~−0.07, local이 조금 이르게 멈춤) |
| 저고도 사각 | 보고 전용 | 마지막 영상 수용 시 body 0.261 / 0.334 m, **접촉 0.0697 / 0.068 m**(보정값 0.0672와 일치), 카메라는 패드 면 아래 0.090–0.092 m, 사각 0.80 / 1.12 s, 두 하강 모두 SUCCESS(vz −0.23 m/s) |
| 품질 채널 | 통과 | 322 결정에서 flag·stamp·age 불일치 0, nav 상수 |
| 관측 순수성 | 통과 | 322 결정을 다시 계산해도 바이트 단위로 같다. |
| 영상 주기 | 보고 전용 | §4 |
| 종료·cleanup | 통과 | 6 episode 모두 정지 확인. FOV 이탈 4회는 검사 종료 후 다음 reset에서 정리됐다. |

## 6. local G3 결과 (2차, 보정 후)

seed 990001–990008, difficulty 1.0, seeded 외란 포함, `spatial_local_v12`.

| 검사 | 결과 | 핵심 수치 |
| --- | --- | --- |
| 축 결합 | 통과 | +ax → Δsin(pitch) 중앙 +0.104, +ay → Δsin(roll) −0.104. 대각 명령의 중첩 잔차 0.004 |
| FOV 이탈 | 통과(16/16) | 센서의 마지막 검출이 패드 중심 경계 통과 후 0.15–0.26 m. board pinhole 예측보다 0.07–0.18 m 이르다. |
| 보정 동일성 | 통과 | local 카메라 두 개 = 등록 보정(1e-12 이내) |
| 저고도 사각 | 보고 전용 | 마지막 영상 수용 시 body 0.245–0.313 m, 접촉 0.0672 m, 사각 **0.70–0.98 s** |
| 품질 채널 | 통과 | 미추정 상태의 결정 0건. 시야 안에서 updated = 1인 비율 97.7 % |
| 관측 순수성 | 통과 | 재계산 결과가 바이트 단위로 같다. |

**남은 보정 차이(G3 미해결):** 광학 주기(Isaac 관측기 수준 측정 대기), 진입 yaw(Isaac은 heading ± 5°, local은 0°), board yaw(최대 0.2 rad), 자기 항법(local은 truth, Isaac은 PX4 EKF), 광학 위치 잡음 모델, 경계 검출 규칙. 보정을 마친 것은 접촉 높이와 handover 검출이다.

## 7. 학습 전에 남은 일

1. **G0/G1:** v1 실행 폴더와 git 묶음(커밋 `5853781`)을 이 컴퓨터로 가져와야 한다.
2. **nav valid/age 상수:** 바꾸려면 네 군 모두 21D schema를 개정해야 한다.
3. **G4 학습 가능성 pilot:** G3는 통과했다. G2가 닫히면, 등록된 task로 Raw PPO의 deterministic·sampled 착륙을 모두 확인한다. 학습은 별도 승인을 받는다.
4. **보정 local 학습 plant 등록:** `backend.BACKENDS["spatial_local_v12"]`는 아직 `pending_g3` 상태다. G3 통과를 근거로 학습 plant로 승격할지는 G4 pilot 등록 때 함께 정한다.

## 8. 구성과 명령

| 파일 (`P/python/ontology_rgat/current_semantics/`) | 내용 |
| --- | --- |
| `contracts.py` | ID·버전, 21D 채널, v12 카메라, 접촉 정의, 행동, 모델 설계, G3(local·Isaac), **task 계약, local plant 보정**. 해시는 observation/calibration/action/feature/graph/algorithm/g3/**task**로 나뉜다. |
| `observation.py`, `geometry.py`, `features.py`, `graph.py` | 21D 복원, 카메라 투영, Φ44, 8노드·22간선 |
| `models.py` | 네 군(4022/4008/4020/4016)과 공통 raw Critic. checkpoint는 task_hash까지 묶는다. |
| `backend.py` | 촬영 시각 정렬 관측기, 보정 local plant, `LocalOpticalCadence`(현재 항등), 등록 task의 보상·판정, handover 대기, 행동 추적 |
| `coupling.py` | G3 local 6종과 Isaac 쪽 검사 |
| `experiment.py` | fixture, gate 표(G3는 두 절반), 보고 상태, smoke, lock |

```bash
./run.sh current-semantics-3d --stage audit|fixtures|smoke|report [--dry-run]
./run.sh current-semantics-3d --stage coupling --backend local
# register-development / develop / register-confirmation / confirm -> 항상 거부(종료 코드 3)
```

테스트: `tests/current_semantics`(215개)와 matlab_port·run.sh·spatial 인접 테스트를 합쳐 **346개가 통과**했다.


## 9. G4 학습 가능성 pilot

**등록(실행 전, `g4_hash`):**
- 대상과 학습 방식: Raw PPO를 보정 local plant에서 학습한다. 모델 seed 0·1, 각 300 update × update당 완결 episode 12개.
- curriculum: 학습에만 쓴다. 난이도 0→1을 앞 50 % 구간에서 올린다.
- 검증: 난이도 1.0에서 10 update마다, seed 24개로 deterministic과 sampled를 모두 잰다. 난이도 1.0에 도달한 checkpoint만 선택할 수 있다.
- PPO 설정: epoch 8, minibatch 256, clip 0.2, λ 0.95, policy lr 3e-4, value lr 1e-3, entropy 0, 단계 후 KL 측정(1.5 × 0.02를 넘으면 되돌림).
- 판정: 선택에 쓰지 않은 시험 seed 48개에서, 모델 seed마다 **deterministic ≥ 50 %이고 sampled ≥ 25 %**여야 한다.

**결과 (`g4_pilot.json`, 등록 예산 완주, 합계 48분):** **G4 failed.** 가이드 §13에 따라 표현(arm) 비교 결론은 내리지 않는다.

| 모델 seed | 선택 update | 시험 deterministic | 시험 sampled | 학습 7,200 episode 중 |
| --- | --- | --- | --- | --- |
| 0 | 269 | 0/48 (모두 TIMEOUT, return −12.2) | 0/48 (모두 TIMEOUT) | SUCCESS 0, TIMEOUT 2,596, envelope 947, 접촉 57 |
| 1 | 249 | 0/48 (모두 TIMEOUT, return −12.4) | 0/48 (모두 TIMEOUT) | SUCCESS 0, TIMEOUT 1,886, envelope 1,566, 접촉 148 |

**진단 (측정):**
1. **초기 이탈.** 무작위 초기화의 평균 행동에 작은 상수 편향이 있다. 순가속도 명령이라 이 편향이 적분되어, 학습 전 정책은 약 15 s 안에 고도 20 m(7/12)나 수평 60 m(5/12)의 envelope에 걸린다. 쉬운 난이도로 배정된 학습 앞 구간을 이 이탈을 없애는 데 다 썼다(envelope 947/1,566회).
2. **떠 있기 국소 최적.** 이탈이 잡힌 update 80–100에는 난이도가 이미 0.6 이상이었다. 이후 TIMEOUT(−12)이 시도 후 충돌(−40)보다 나아서, 정책은 고도 2–4.6 m, 패드에서 수평 1.5–3 m 떨어진 자리에서 70 s를 버틴다. log σ는 약 −1.1~−1.4로, 탐색이 꺼지지는 않았다.
3. **탐색으로 착륙에 닿을 수 있다.** 패드를 따라 내려가는 단순한 평균 행동은 d = 0에서 잡음 없이 24/24, σ 0.08에서 24/24, **σ 0.35(pilot의 탐색 폭)에서 15/24** 착륙한다. 일정한 하강은 5/24, 4/24, 1/24다. 따라서 "표본 행동은 원리적으로 착륙할 수 없다"는 문제가 아니라, 학습 경로가 그 평균 행동 근처에 가지 못한 문제다. d = 1.0에서는 같은 단순 추종도 4–6/24에 그친다.

**pilot 1 기록 보관:** `results/.../g4_pilot_v1/`에 결과, 학습 기록, checkpoint, 당시 설정 사본(`config_snapshot.yaml`), `ARCHIVE.json`이 있다.

**다음 pilot 후보(당시 제안):** 기준은 그대로 두고 탐색과 curriculum 계약만 바꾼다. 가이드 §10 G4가 허용하는 범위다.
(a) 정책 출력층을 작게 초기화(예: ×0.01)해 평균 행동이 0에서 시작하게 한다. 사전 지식(prior)이 아니라 초기 이탈만 없애는 조치다.
(b) 시간 기반 ramp 대신, 학습 성공률이 일정 수준에 도달해야 난이도를 올리는 curriculum.
(c) 필요하면 예산을 늘린다.
teacher, BC, 적분 prior는 쓰지 않는다.



### 9.2 Pilot 2 (`g4-pilot/2`, 2026-10-28 사용자 "all"로 승인)

**등록(실행 전):** pilot 1에서 세 가지만 바꿨고, 판정 기준은 그대로다.
1. **출력층 초기화:** actor 출력층을 ×0.01로 줄이고 bias를 0으로 둔다(`MODEL_ARCHITECTURE.actor_output_init_scale`, 네 군 공통). 학습 전 평균 latent는 약 0.0005다.
2. **성공률 기반 curriculum:** 같은 난이도에서 최근 5 update(60 episode)의 sampled 학습 성공률이 50 % 이상이면 난이도를 0.1 올린다. 내리지는 않는다. 검증은 난이도 1.0에 도달한 뒤 10 update마다 한다.
3. **예산:** 600 update × 12 episode(모델 seed 0·1).

**실행 전 측정(정직하게 기록):** 평균이 0이어도 탐색 잡음 σ 0.37이 속도와 위치에 쌓인다. 그래서 학습 전 sampled 정책은 d = 0에서도 0/48, d = 0.1에서 1/48만 착륙하고 대부분 envelope로 끝난다. 착륙 신호는 처음에는 보상의 dense 항(고도·거리·readiness)에서 와야 한다. 이 pilot은 그 경로가 실제로 작동하는지를 시험한다.

**결과 (`g4_pilot.json`, 등록 예산 완주, 합계 13분):** **G4 failed (2회째).** 두 모델 seed 모두 난이도 0에서 한 번도 승급하지 못했다. 그래서 난이도 1.0에 도달한 checkpoint가 없고, 시험 seed 48개로 잴 정책도 없다. 가이드 §13에 따라 arm 비교 결론은 내리지 않는다.

| 모델 seed | 최종 학습 난이도 | 승급 | 시험 | 학습 7,200 episode 중 |
| --- | --- | --- | --- | --- |
| 0 | 0.0 | 0회 | 시험 불가(선택 가능한 checkpoint 없음) | SUCCESS 0, envelope 7,002, 패드 밖 접촉 195, UNSAFE 2, TIMEOUT 1 |
| 1 | 0.0 | 0회 | 시험 불가(선택 가능한 checkpoint 없음) | SUCCESS 3, envelope 6,791, 패드 밖 접촉 384, UNSAFE 6, TIMEOUT 16 |

학습이 진행될수록 episode가 짧아졌다. 평균 decision 수는 seed 0에서 240 → 85, seed 1에서 → 70이다. log σ는 −1.0에서 −1.1~−1.6으로 내려갔다.

**진단 (사후 측정, 등록 seed와 겹치지 않는 진단 seed 996001–996024, d = 0).** 스크립트와 수치는 `results/.../g4_pilot2_diagnostics/`(`summary.json`)에 있다.
1. **평균 0인 탐색은 가장 쉬운 난이도에서도 착륙하지 못한다.** 평균 latent를 0으로 고정하고 σ를 바꿔 쟀다. SUCCESS는 σ 0.05에서 0/24, 0.1에서 1/24, 0.2에서 1/24, 0.37(학습 전 모델)에서 0/24였다. 패드를 멈춰 세워도 2/24, 1/24, 0/24로 거의 같다. d = 0에서는 외란이 0배로 꺼져 있으므로 외란 탓도 아니다. 실패는 두 가지다. σ 0.37에서는 고도 20 m 천장(17/24)에 닿는다. 기울기는 10° 이하로 21°와 거리가 멀다. σ가 작으면 패드 밖 접촉이 대부분이다. 순가속도 명령은 이중적분이라, 평균 0 탐색은 σ가 시간 척도만 바꿀 뿐 궤적의 모양은 그대로다. 따라서 착륙 확률은 기하가 정한다. 시작 높이 0.4–0.7 m를 내려가는 동안 수평으로 0.5 m(패드 반폭) 넘게 흐르지 않아야 한다. 무작위로는 이 조건을 거의 만족하지 못한다. pilot 1 진단 3의 결과(패드를 따라가는 평균 행동은 σ 0.35에서도 15/24 착륙)와 맞물린다. 착륙에는 상대 위치에 대한 되먹임 평균이 필요하다. 그런데 그 평균으로 가는 길을 알려줄 성공 표본이 학습 batch에 거의 없다.
2. **성공이 없으면 PPO는 자기 목적에서 더 나쁜 정책으로 표류한다.** seed 0의 `final.pt`는 t0의 평균 행동이 [+2.47, +2.00, +1.93] m/s²로 한계 [2.5, 2.5, 2.0]에 붙어 있다. deterministic과 sampled 모두 24/24가 8.5 s에 수평 60 m envelope로 끝난다. s0 기준 할인 return은 학습 전 −34.0, 학습 후 −40.6이다. 같은 −40 종료라도 늦을수록 할인되어 낫기 때문에, 보상이 이탈을 선호한 것은 아니다. 성공 신호 없이 쌓인 갱신 편향으로 보인다. 정확한 기제는 재지 않았다.
3. **PPO 갱신 방향은 맞다.** 보상이 −|a − target|²인 1-step 문제에서, 평균은 40 update 뒤 0에서 [−1.27, 0.68, −0.87]로 target [−1.0, 0.5, −0.8] 쪽으로 이동했다.
4. **관측은 부족하지 않다.** 패드가 화면 밖에 있어도 21D 상대 위치 채널은 거리를 담는다(57 m에서 relative_x −0.95). critic의 V(s0) −40.5는 실측 −40.6과 맞는다.

**해석:** pilot 2의 세 가지 변경(초기 이탈 제거, 성공률 기반 curriculum, 두 배 예산)은 pilot 1의 첫 원인인 초기 이탈은 없앴다. 그러나 더 근본적인 문제가 남았다. 가이드 제약(teacher·BC·prior 없음, 순가속도 행동, task 계약 그대로) 아래에서는 가장 쉬운 난이도조차 무작위 탐색의 착륙률이 0–8 %다. 이 저장소에서 held-out에서 착륙한 공간 PPO 정책은 모두 BC로 초기화한 정책에서 출발했다(AGENTS.md, `low_sigma_kl` 이후). 처음부터 학습한 PPO가 착륙 정책을 만든 기록은 없다. 학습 중 SUCCESS가 몇 번 나온 cell은 있었지만, held-out 착륙 정책으로 이어지지는 않았다. 이번 결과도 그 기록과 같다.

**다음 선택지(사용자 결정, 실행하지 않음):**
(a) **가이드 안에서 pilot 3:** 초기 log σ를 약 −2.3(σ 0.1)으로 낮춰 d = 0의 무작위 착륙률을 0이 아니게 만든다. GAE λ를 0.99로 올려 +25의 신용이 약 10 s 거슬러 가게 한다. 학습 전에 진단 seed 48개로 무작위 착륙률(≥ 5 %)을 먼저 확인하는 조건을 함께 등록한다. 다만 σ 0.1에서는 TIMEOUT(−12)이 쉽게 나오므로 pilot 1의 떠 있기 국소 최적에 다시 빠질 위험이 크다. 성공 가능성은 낮게 본다.
(b) **가이드 완화:** 네 arm 모두에 같은 교사 데이터로 BC 초기화를 하고 같은 예산으로 PPO를 돌린다. 이 저장소에서 착륙이 확인된 방식이다. 다만 연구 질문이 "처음부터 학습"에서 "복제 정책의 미세조정"으로 바뀌고, 가이드의 teacher/BC 금지와 충돌한다.
(c) **여기서 G4를 닫는다:** "이 계약과 제약에서 처음부터 학습한 Raw PPO는 학습되지 않는다"를 결과로 기록한다.


## 10. 공통 보상·관측 재구성 (task v2, observation v2)과 G4 pilot 3

**사용자 요청(2026-10-28):** "PPO가 초기 BC 없이도 잘 착륙할 수 있게 공통 보상이나 공통 관측 입력을 이상적으로 재구성." 네 arm 모두에 같은 변경이 적용되므로, arm 비교의 공정성은 유지된다.

### 10.1 무엇을 바꾸지 않았나

- **판정:** spatial-reference/12 `Evaluator.evaluate`가 그대로 상태를 정한다. 충격 속도 기준 접촉 판정, 패드 반폭 0.5 m, envelope 21° / 20 m / 60 m가 그대로다. 착륙 성공의 정의가 바뀌지 않았으므로, 성공률은 이전 pilot들과 같은 척도다.
- **종료 보상:** Evaluator의 표를 그대로 읽는다(SUCCESS +25, TIMEOUT −12, unsafe −40). `reward.terminal_reward`가 Evaluator의 상수를 읽으며, 값을 다시 적지 않는다.
- **정책 경로:** 현재 21D만 입력하고, history·teacher·BC·prior·감독기는 없다. 시뮬레이터 truth는 보상에만 들어간다. 정책 경로 모듈이 `reward.py`를 import하지 않음을 테스트로 고정했다.
- **PPO와 프로토콜:** pilot 2와 같다(출력층 ×0.01, 성공률 기반 curriculum, 600 update × 12 episode, KL guard, 판정 기준 det ≥ 50 % / sampled ≥ 25 %).

### 10.2 공통 보상 v2 (`spatial-current-reward/2`, `reward.py`)

Φ(s) = −(w_d·√(|e|² + h²) + w_v·|v − v_des(e, h)|)
- e: 기체 − 패드의 수평 오차, h: 접촉 높이 위의 높이, v: 기체 − 패드의 상대 속도(모두 truth)
- v_des_xy = −e·min(1.0, 1.5/|e|) m/s
- v_des_z = −align·min(0.8, 0.2 + 0.5h) m/s, align = clip(1 − |e|/(0.25 + 0.5h), 0, 1). 정렬된 경우에만 하강을 요구하고, 접지 직전에는 0.2 m/s로 내려온다. 0.2 m/s는 판정 한계 0.30 m/s 안이다.
- 매 step 보상 = Φ(s') − Φ(s)(할인하지 않은 차이) − (0.3 + 0.5·mean(u²))·dt + 종료 보상. Φ는 종료 상태에서도 0으로 지우지 않는다. 그래서 빠른 접촉이나 멀리 빗나간 종료는 Φ만큼 더 깎인다.
- w_d = 2 /m, w_v = 4 /(m/s). 설정 검증은 접지 sink가 0.30 m/s 미만인지, 정렬 반경이 패드 안인지 확인한다.

**왜 이 모양인가 (실행 전 측정, 진단 seed 996001–996024, `harness.py`):**

| 측정 | 기존 Evaluator dense 항 | 새 보상 v2 |
| --- | --- | --- |
| 행동이 목표 방향을 따르는 정도와 다음 1/5/10/20 step 보상 합의 상관(초기 정책 σ 0.37, 약한 추종 α 0.25–0.5, d = 0과 1) | −0.05 ~ 0.08 (사실상 0) | **0.06 ~ 0.46** (1–10 step 0.10–0.46; 초기 정책 공칭 0.10–0.17, 약한 추종 0.24–0.46) |
| 공칭(d = 1) 평균 return: 속도장 추종 제어기 / 떠 있기 / 급강하 / 무작위 | +18.2 / **+0.7** / −39.9 / −40.0 | +17.1 / −28.2 / −38.5 / −113 |
| 0 → 속도장 추종으로 가는 α 경로(σ 0.2) | 단조 증가 | 단조 증가 |

- 기존 dense 항은 위치 potential이 중심이다. 가속도 행동이 위치를 바꾸려면 이중적분을 거쳐야 한다. readiness 항은 exp(−risk/2)라서 패드에서 약 1 m만 벗어나도 평평하다. 그래서 행동의 좋고 나쁨이 2 s 안의 보상에 드러나지 않았다. PPO의 advantage에 방향 정보가 없었다는 뜻이다.
- 속도 오차 potential은 행동이 0.1 s 지연과 자세 지연을 거친 뒤 바로 반영된다.
- 떠 있기는 시간 비용과 Φ 때문에 더 이상 이득이 없다.

**속도장은 관측만으로 실현 가능하다.** 정책이 볼 수 있는 21D 추정치만 쓰는 메모리 없는 제어기 a = k·(v_des − v̂)로 쟀다.
- d = 0에서 24/24 착륙했다.
- 공칭에서 k = 4 /s일 때 **20/24**다. k = 1.5, 2.5, 6, 8은 각각 5, 10, 3, 0/24다. 최고점이 grid 경계가 아니라 내부에 있다.
- sampled(σ 0.37)에서도 d = 0 14/24, 공칭 11/24가 착륙한다. 즉 이 보상이 가리키는 행동은 탐색 잡음을 견딘다.
- 패키지 보상은 harness 공식과 6 seed에서 소수점 6자리까지 같다.

### 10.3 공통 관측 v2 (`spatial-current-observation/2`)

21개 채널, 순서, 변환 규칙은 그대로다. 부호 있는 위치·속도 채널의 scale만 착륙 구간에 맞췄다(`contracts.OBSERVATION_SCALE_OVERRIDES`).

| 채널 | 이전 scale | 새 scale | 0.3 m/s 또는 0.2 m일 때 입력값 |
| --- | --- | --- | --- |
| relative_x / relative_y | 3 m | 1 m | 0.06 → 0.17 |
| relative_height | 8 m | 2 m | 0.024 → 0.09 |
| relative_vx / vy, ugv_vx / vy | 10 m/s | 1 m/s | 0.029 → 0.23 |
| drone_vz | 1.5 m/s | 1 m/s | 0.17 → 0.23 |

- Φ44 arm의 파생 접근 feature는 이미 1 m, 1 m/s를 쓰고 있었다. raw arm과 **모든 arm이 공유하는 raw21 critic**만 착륙 구간의 차이를 0.03 수준으로 보고 있었다.
- direct_policy 인코더의 출력을 `observation.from_source`로 정확히 다시 scale한다(decode 후 encode). 그래서 local과 Isaac은 같은 관측 경로를 쓴다.
- 이 변경의 학습 효과는 따로 재지 않았다. 보상 변경과 함께 pilot 3에 들어가므로, 둘의 기여를 나눌 수 없다.

### 10.4 hash와 gate

- observation_hash, feature_hash, algorithm_hash, task_hash, g4_hash가 바뀌었다. 그래서 pilot 1·2의 checkpoint는 새 계약에서 로드되지 않는다.
- g3_hash는 바뀌지 않았으므로 **G3 passed**가 유지된다.
- pilot 2는 `results/.../g4_pilot_v2/`에 보관했다(`ARCHIVE.json`, 설정 사본은 이번 두 편집을 되돌려 복원).

### 10.5 G4 pilot 3 (`g4-pilot/3`) 결과

**결과 (`g4_pilot.json`, 등록 예산 완주, 합계 21분): G4 passed.** BC·teacher·prior 없이 처음부터 학습한 Raw PPO가 두 모델 seed 모두에서 등록 기준(det ≥ 50 %, sampled ≥ 25 %)을 넘었다. 시험은 선택에 쓰지 않은 seed 993001–993048, 난이도 1.0(공칭, 외란 포함)이다.

| 모델 seed | 첫 SUCCESS | 난이도 1.0 도달 | 선택 update | 시험 deterministic | 시험 sampled | 학습 7,200 episode 중 SUCCESS |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | update 9 | update 71 | 540 | **48/48 (100 %)** | **46/48 (95.8 %)**, UNSAFE_CONTACT 2 | 6,178 (마지막 100 update: 1,136/1,200) |
| 1 | update 4 | update 69 | 599 | **46/48 (95.8 %)**, UNSAFE_CONTACT 2 | **44/48 (91.7 %)**, UNSAFE_CONTACT 4 | 6,227 (마지막 100 update: 1,145/1,200) |

- 같은 PPO와 같은 예산으로 pilot 2는 14,400 episode 중 SUCCESS 3번이었고 난이도 0을 벗어나지 못했다. 바뀐 것은 공통 보상과 관측 scale뿐이다.
- 시험에서 나온 실패는 모두 UNSAFE_CONTACT(충격 속도 초과)다. 떠 있기, envelope, 패드 밖 접촉은 없었다.
- 학습된 정책은 보상의 속도장을 그대로 따르는 메모리 없는 제어기(공칭 20/24, 83 %)보다 높다. 보상이 가리키는 방향을 출발점으로 삼았을 뿐, 그 제어기를 흉내 낸 것이 아니라는 뜻이다.
- 최종 log σ는 −1.5 ~ −1.7(σ 0.18–0.22)이다. 탐색 잡음이 남아 있어도 sampled 착륙률이 92–96 %다.

**한계(그대로 적는다):**
1. **local 보정 plant 결과다.** 사용자 결정(§1)에 따라 선택 checkpoint를 Isaac/PX4에서 평가했고, 기준에 미달했다(§10.6).
2. **Raw arm 하나, 모델 seed 2개다.** 네 arm 비교(가이드 §10–11의 development 단계)는 하지 않았고, CLI는 여전히 그 단계를 거부한다.
3. **보상과 관측을 함께 바꿨다.** 두 변경의 기여는 나누지 않았다. 나누려면 관측 v1 scale에 보상 v2만 적용한 pilot이 필요하다.
4. 보상은 truth로 계산한 속도장을 담고 있다. 정책 입력에는 truth가 없지만, 학습 신호가 특정 접근 방식(정렬 후 하강, 접지 0.2 m/s)을 선호하도록 설계됐다. 판정과 성공 정의는 바뀌지 않았다.

### 10.6 선택 checkpoint의 Isaac 평가 (`isaac-eval/1`)

**사용자 결정(2026-10-28):** "선택 checkpoint 2개를 Isaac에서 평가". §1의 결정(모든 선택 checkpoint 평가는 Isaac/PX4)을 이행한다.

**등록(비행 전, `isaac_eval_hash`):**
- 대상: pilot 3의 두 모델 seed의 `best.pt`(update 540, 599)다. checkpoint는 현재 계약 hash와 일치할 때만 로드된다.
- 정책: deterministic.
- seed: 994001–994010(10개). 두 checkpoint가 같은 seed를 비행하고, seed 순서로 번갈아 비행한다. 그래서 중간에 멈춰도 두 checkpoint의 표본이 비슷하게 남는다. 이 seed들은 등록·감사·진단 범위 어디와도 겹치지 않는다.
- 비행 방식: G3 Isaac 검사와 같은 owned stack(`--stage isaac-eval --allow-isaac`)이다. 실행 중인 Isaac은 넘겨받지 않는다. 진입 실패에는 같은 seed 재시도로 owned restart 예산 2를 쓴다.
- 판정: checkpoint마다 deterministic 착륙률 ≥ 50 %(G4와 같은 기준)이고, 모든 episode의 정지 확인이 있어야 한다. 비행 중 예외가 나면 그 자리에서 멈추고 "미완료"로 기록한다.
- 해석: 20 episode는 작은 통합 표본이다. 강건성이나 표현 우월성 주장이 아니다.

**결과 (`isaac_eval.json`, 20 episode 모두 비행, 모든 정지 확인, 약 23분): 기준 미달(isaac_eval_passed = False).**

| 모델 seed | Isaac deterministic | 상태 | 같은 seed의 local(참고) |
| --- | --- | --- | --- |
| 0 (update 540) | **4/10 (40 %)** | SUCCESS 4, envelope 4, TIMEOUT 2 | 10/10 |
| 1 (update 599) | **3/10 (30 %)** | SUCCESS 3, envelope 7 | 10/10 |

- **성공 7건**은 5.7–6.7 s에 패드 중심 0.15–0.21 m 안에 수직 0.12–0.22 m/s로 내려앉았다. 보상의 속도장(0.2 m/s 접지)과 같은 모양이다. BC 없이 학습한 정책이 Isaac에서 착륙한 첫 기록이다.
- **실패 13건은 두 유형이다.**
  - **(B) handover 직후 기울기 초과, 3건:** 0.9–1.1 s, 21.5–23.4°, 높이 1.6–1.75 m.
  - **(A) 첫 접근에서 패드를 놓침, 10건:** seed 0 정책은 놓친 뒤 0.5 m 높이에 머물고, 그동안 패드가 떠나 48–62 s에 수평 60 m envelope에 걸리거나 70 s timeout이 된다. seed 1 정책은 패드에서 4–11 m 떨어진 곳에서 패드 면 아래 0.2 m로 내려가 envelope에 걸린다(7.7–13.3 s). 첫 접근이 실패한 뒤 회복하는 행동은 학습되지 않았다. local에서는 거의 항상 5 s 안에 착륙하므로 회복을 배울 기회가 없었다.
- **결과는 seed가 정한다.** 10개 seed 중 9개에서 두 정책의 결과가 같았다. handover 초기 수평 속도 교란(`sample_domain_randomization`)으로 깔끔하게 갈린다. 0.08 / 0.11 / 0.55 m/s는 두 정책 모두 성공, 0.70은 갈림, 0.62–0.98은 두 정책 모두 실패다.

**사후 진단 (local, 비행 없음):** 같은 checkpoint와 seed를 local에서 다시 비행하며 handover 교란만 키웠다(`results/.../isaac_eval/diagnostics/kick.py`).

| 교란 배율 | seed 0 정책 | seed 1 정책 | local 최대 기울기 |
| --- | --- | --- | --- |
| ×1 (그대로) | 10/10 | 10/10 | 15.8° / 17.3° |
| 속도 ×1.25 | 10/10 | 9/10 | 15.9° / 17.5° |
| 속도 ×1.5 | 8/10 | 7/10 | 16.4° / 17.7° |
| 각속도 ×2 | 10/10 | 10/10 | 15.8° / 17.4° |

- 정책은 handover 속도 교란에 대한 여유가 작다(×1.5에서 20–30 % 감소).
- 그러나 local은 교란을 키워도 기울기가 18°를 넘지 않는다. Isaac은 같은 seed에서 1 s 안에 21.5–23.4°까지 기운다. 즉 Isaac의 handover 직후 자세 반응이 local plant보다 거세다. reference 경로가 `/11`(handover 측면 제한)과 `/12`(handover 추력 하한)에서 **감독기 규칙**으로 막았던 차이와 같은 현상이다. 이 경로에는 감독기가 없다.
- 이번 비행은 종료 상태만 기록했다. 실패의 step별 기제(추정치가 언제 패드를 잃는지, Isaac에서 명령 대비 실제 가속이 얼마인지)는 아직 측정하지 않았다. 다음 비행을 위해 결정별 궤적 기록(`traces/`)을 추가했다.

**다음 선택지(사용자 결정):**
(1) **진단 비행:** 궤적을 기록하며 실패 seed 몇 개를 짧게 비행한다(약 5분). Isaac의 handover 반응을 local과 같은 지표로 잰다. 비행 승인이 필요하다.
(2) **강건화 재학습(local):** 학습에서 handover 교란 범위를 넓히고(domain randomization, 학습 전용), 기울기·각속도에 비용을 준다. 그런 뒤 새 seed로 Isaac 재평가를 등록한다. teacher·prior·감독기는 쓰지 않는다.
(3) **감독기 도입:** reference `/12`의 handover 규칙을 정책 밖 안전층으로 넣는다. 가이드의 "감독기 없음" 원칙과 충돌한다.


### 10.7 궤적 진단 비행 (`isaac-trace`, 2026-10-28, 사용자 선택 "1")

**목적:** §10.6 실패의 기제를 결정 단위로 잰다. 판정이 아닌 진단이며, `isaac_eval.json`은 바꾸지 않는다.

**방법:** 등록 seed 중 6쌍을 owned stack에서 결정별 궤적과 함께 다시 비행했다(`--stage isaac-trace --allow-isaac --pairs ...`, 결과 `results/.../isaac_trace/20261028T115703Z/`). 같은 쌍을 local에서도 같은 기록으로 비행했다(`isaac_trace/local_counterpart/`). 비교는 `isaac_trace/compare_summary.py`이고, 표는 `compare_summary.txt`다.

**재현성:** Isaac 결과는 등록 비행과 같았다. seed 1 정책은 994001·994003·994006·994008에서 실패하고 994002에서 성공했다. seed 0 정책은 994008에서 성공했다. local에서는 6쌍이 모두 착륙했다(3.7–6.0 s).

**측정된 실패 경로:**
1. **handover 직후 추정치 과도 오차(Isaac에만 있다).**
   - Isaac에서 자기 속도(EKF)는 첫 결정 사이에 ±0.6 m/s씩 튄다. 첫 step의 측정 가속이 994003에서 +5.97 m/s², 994006에서 −5.84 m/s²였다(같은 seed의 local은 kick 방향과 일치).
   - 그 결과 상대 속도 추정이 0.2–0.4 s 동안 **실제와 반대 부호**가 된다. 994003에서는 y +0.64 m/s(실제 −0.78), 994001에서는 +0.64(실제 −0.57)였다. local에서는 6쌍 중 0건이다.
   - 급하강 중의 높이 추정 지연도 Isaac이 크다. 994001에서 Isaac 0.43 m, local 0.12 m였다.
2. **정책의 반응:** 잘못된 부호를 지우려는 ±2 m/s² 명령이 나오고, 0.3–0.5 s 뒤 반대로 급반전한다(명령 변화량 최대: Isaac 2.2–3.0, 같은 쌍의 local 1.0–1.6 m/s²).
3. **PX4 자세 overshoot:** 같은 크기의 명령에 Isaac은 더 크게 기운다. 명령 가속이 요구하는 기울기 대비 실제 최대 기울기는 실패 쌍 994003·994006에서 Isaac 1.50·1.68배, local 0.94·0.80배였다. 다른 쌍은 0.95–1.14배로 양쪽이 비슷했다. 994006은 local의 명령 변화가 더 컸는데도(3.09 대 2.21) 기울기는 local 11.6°, Isaac 23.6°였다. `/10`–`/11`에서 기록된 "명령 반전 → PX4 overshoot" 기제와 같다.
4. **결말 두 갈래:** 반전이 기울기 21°를 넘기면 B(1 s 안에 envelope)다. 수직 명령도 −2 m/s²로 공격적이면 급하강해 패드에서 0.8 m 비낀 채 패드 면 높이에 닿는다(994001, 1.7 s). 그 높이에서 카메라 시야가 좁아 패드를 잃고, 추정치는 멈춘 값을 유지하며, 정책은 다시 올라가 찾지 않는다. 이것이 A다.

**결론:** 실패의 시작은 Isaac handover 직후 1 s의 **관측 과도 오차와 자세 overshoot**다. local plant는 둘 다 재현하지 않으며, 정책은 이를 견딜 만큼 부드럽지 않다. 판정 정의, 보상 순서, 관측 정보 부족이 원인은 아니다.

**고치는 방향(측정에 근거한 제안, 사용자 결정):**
- (a) **학습 전용 domain randomization:** handover 뒤 짧은 시간 동안 자기 속도·위치 측정에 감쇠하는 무작위 오차를 넣는다(크기 ~0.6 m/s, 시상수 ~0.3 s, Isaac 궤적에서 잰 값). 자세 servo의 overshoot도 무작위로 넓힌다. 평가(d = 1.0 판정)는 그대로 둔다.
- (b) **공통 보상에 명령 변화·기울기 비용 추가:** 급반전을 비싸게 만든다(`/11`의 감독기 slew 제한 1.5 m/s²/결정을 보상으로 옮긴 것). 감독기는 쓰지 않는다.
- (c) 새 Isaac 평가 seed를 등록하고 재평가한다. 이번 seed 994001–994010은 진단에 썼으므로 재사용하지 않는다.

### 10.8 같은 checkpoint의 새 seed 재평가 (`isaac-eval/2`)

**사용자 선택(2026-10-28): "c".** 재학습(a·b) 없이 pilot 3의 선택 checkpoint 2개(update 540, 599)를 새 seed로 다시 비행한다. 수정 효과를 시험하는 것이 아니라, §10.6 결과(4/10, 3/10)가 새 seed에서도 재현되는지 보는 **독립 반복 측정**이다.

**등록(비행 전, `isaac_eval_hash` 변경):**
- seed: 994011–994020(10개). 두 checkpoint가 같은 seed를 비행하고, seed 순서로 번갈아 비행한다.
- seed 관리: 994001–994010은 `isaac-eval/1`과 진단 비행에 썼으므로 재사용 금지 범위(`contracts.DIAGNOSTIC_SEED_RANGES`)에 넣었다.
- 정책·기준·방식은 `/1`과 같다. deterministic, checkpoint마다 ≥ 50 %, 모든 정지 확인, owned stack, 재시작 예산 2.
- 달라진 점: 이번에는 결정별 궤적(`isaac_eval/traces/`)을 함께 남긴다.
- 보관: `/1`의 결과는 `results/.../isaac_eval_v1/`(ARCHIVE.json 포함)로 옮겼다.
- 사전 예상(기록만 한다): `/1`에서 성공은 handover 수평 교란이 작은 seed에 몰렸다. 그래서 결과는 새 seed의 교란 분포에 크게 좌우될 것이다.
- **비행 전 고정한 seed별 예측(진단 가설의 검증):** `/1`에서 관찰한 경계(|v_xy0| ≤ 0.55 m/s 착륙, ≥ 0.62 실패)를 새 seed의 교란(`sample_domain_randomization`, 비행 전에 계산)에 그대로 적용한다. 두 checkpoint 모두에 같은 예측이다.

| seed | handover 수평 교란 \|v_xy0\| (m/s) | 수직 교란 (m/s) | 외란 \|F_xy\| (N) | 예측 |
| --- | --- | --- | --- | --- |
| 994011 | 0.83 | +0.97 | 0.41 | fail |
| 994012 | 0.78 | +0.43 | 0.91 | fail |
| 994013 | 0.79 | +0.35 | 0.30 | fail |
| 994014 | 0.24 | -0.02 | 0.77 | land |
| 994015 | 0.99 | -0.36 | 0.84 | fail |
| 994016 | 0.66 | +0.53 | 0.36 | fail |
| 994017 | 0.36 | +0.91 | 0.51 | land |
| 994018 | 0.62 | +0.92 | 0.12 | uncertain |
| 994019 | 0.57 | +0.19 | 0.81 | uncertain |
| 994020 | 0.74 | +0.68 | 0.48 | fail |

**결과 (`isaac_eval.json`, 20 episode 모두 판정, 모든 정지 확인): 기준 미달(isaac_eval_passed = False). `/1`의 결과가 재현됐다.**

| 모델 seed | `/2` Isaac deterministic | 상태 | `/1`(참고) |
| --- | --- | --- | --- |
| 0 (update 540) | **2/10 (20 %)** | SUCCESS 2, TIMEOUT 5, envelope 3 | 4/10 |
| 1 (update 599) | **3/10 (30 %)** | SUCCESS 3, envelope 6, 패드 밖 접촉 1 | 3/10 |

- 두 평가를 합치면 40 episode 중 12건(30 %)이 착륙했다.
- **인프라 중단 1건:** 8번째 episode(model seed 1, seed 994014) 비행 중 PX4 failsafe(`gcs_connection_lost`, `battery_unhealthy` 등)가 났다. 판정 없이 중단됐다. 판정이 난 7 episode는 그대로 두고, 중단된 쌍부터 새 owned stack에서 이어 비행했다(`resume`). 이 쌍은 다시 비행해 착륙했다. 중단 기록은 `isaac_eval/interruptions.jsonl`과 보고서의 `interruptions`에 남겼다. 판정이 난 결과는 하나도 다시 비행하지 않았다.
- **사전 예측의 적중:** 확정 예측(착륙/실패) 16 episode 중 15건이 맞았다.
  - 틀린 1건: model seed 0 × 994017. 이 seed는 수평 교란이 0.36 m/s로 작았지만 수직 교란이 +0.91 m/s로 컸고, 예측은 수평 교란만 썼다.
  - 불확실로 둔 경계 seed: 994018(0.62 m/s)은 두 정책 모두 실패했고, 994019(0.57 m/s)는 둘 다 착륙했다.
  - 즉 `/1`에서 세운 "handover 교란이 결과를 정한다"는 진단이 새 seed에서 확인됐다.
- **새로 측정된 요소: 추적기 lockout.** 이번에는 20 episode 모두 결정별 궤적이 있다. 첫 접근 실패(A) 14건 중 **8건**에서 같은 일이 일어났다.
  - handover 1.6–2.0 s 뒤, 실제 패드가 카메라 시야 안쪽(수평 거리 < 0.6 × 깊이 × tan(VFOV/2), 깊이 > 0.3 m)에 있는데도 영상 갱신이 1 s 넘게 끊긴다.
  - 예: 994017 model 0은 2.15 s에 높이 1.9 m, 패드와 수평 0.41 m 차이였는데 갱신이 70 s까지 돌아오지 않았다.
  - handover 과도 오차로 추정치가 실제와 0.5 m 이상 어긋나면, 참조 ABG 추적기가 이후의 실제 검출을 이상치로 거부하고 다시 잡지 못한다. 추정치는 멈춘 값을 유지하고, 정책은 그 값을 따라 엉뚱한 곳에 머문다.
  - G3에서 local 10 Hz로 영상을 솎았을 때 보았던 lockout과 같은 현상이다. 학습된 공격적 정책과 Isaac 과도 오차가 겹치자 Isaac에서도 나타났다.
  - 나머지 A 6건은 패드가 실제로 시야 밖(저고도 또는 큰 오프셋)이었다. 기울기 유형(B)은 1건이었다.

**업데이트된 원인 정리:** handover 직후 Isaac 관측 과도 오차(§10.7)가 출발점이다. 그 뒤 세 갈래로 실패한다. (1) 명령 반전에 따른 PX4 기울기 초과. (2) 추정치 이탈에 따른 추적기 lockout. (3) 급하강·오프셋에 따른 시야 상실. (2)와 (3) 모두에서 정책은 다시 올라가 패드를 찾지 않는다.

**제안(사용자 결정, 측정 근거):**
- (a) **공통 관측 파이프라인의 재포착:** 실제 검출이 연속으로 거부되면 추적기를 새 검출로 다시 초기화한다. truth는 쓰지 않는다. 관측 버전이 바뀐다.
- (b) **학습 전용 handover 과도 오차 DR:** 자기 상태 측정에 감쇠하는 오차를 넣는다(§10.7).
- (c) **공통 보상의 명령 급변·기울기 비용.**
- (a)–(c) 뒤 새 seed로 Isaac 재평가를 등록한다.


## 11. 추적기 재포착 (observation/3)과 G4 pilot 4

**사용자 선택(2026-10-29): "a".** §10.8의 lockout을 공통 관측 파이프라인에서 고친다.

### 11.1 원인 (코드와 측정)

- 참조 ABG 추적기(`spatial/estimation.py`, reference 경로와 공유하므로 수정하지 않음)는 첫 검출에서 **패드 속도를 기체 자신의 속도로 초기화**한다(`velocity[:] = own_velocity`). 그 뒤 프레임마다 `0.02/gap × innovation`만큼만 속도를 고친다.
- handover 교란 직후라면 출발 속도가 교란만큼(0.6–1.0 m/s) 틀린다. 교란 크기가 성패를 예측했던 이유다(§10.8 예측 15/16).
- Isaac의 영상 간격 ~0.1 s(10 Hz)에서는 프레임당 보정이 local 20 Hz의 절반이고 프레임 수도 절반이라, 수렴이 약 4배 느리다.
- 그사이 예측이 innovation 문턱(max(4σ, 0.25 m))을 넘으면, 이후 검출을 모두 거부하고 σ가 줄 기회도 없어 영구히 잠긴다.
- 합성 시험(`tests/current_semantics/test_cs3d_reacquire.py`, 패드 0.8 m/s):
  - 20 Hz + 교란 −0.8 m/s: 0.4 s 뒤 lockout.
  - 10 Hz + 교란 없음: 0.8 s 뒤 lockout.
  - 10 Hz + 교란: 0.5 s 뒤 lockout.
  - 세 경우 모두 4 s 뒤 위치 오차가 1 m를 넘는다.

### 11.2 변경 (`spatial-current-observation/3`)

- `CurrentObserver`에 재포착을 넣었다. 추적기의 freshness 규칙으로 최신인 검출이 innovation 문턱에서 **연속 3번 거부**되면, 그 3개 검출로 추적기를 재초기화한다.
  - 위치·속도: 포획 시각에 대한 최소제곱 직선. 수직 속도는 0(평지).
  - 위치는 현재 시각까지 외삽한다. 가속도는 0, σ는 새 추적기의 초기값으로 둔다.
  - 재초기화는 수용된 프레임으로 센다.
- 추적기가 프레임을 받아들이는 동안에는 아무것도 바뀌지 않는다(테스트로 비트 단위 동일을 확인).
- 재포착 횟수는 provenance에만 남고 21D에는 들어가지 않는다. truth는 쓰지 않는다.
- 규칙은 관측 manifest(`pad_track`)에 들어가므로, observation_hash와 algorithm_hash가 바뀐다. 이전 checkpoint는 로드되지 않는다.
- G3 local 검사를 새 관측기로 다시 실행했고 통과했다(`coupling_local.json`, g3_hash는 그대로).

### 11.3 실행 전 측정 (local, pilot-3 정책, seed 997001–997048, `observation_v3_checks/`)

| 영상 주기 | 재포착 없음 | 재포착 있음 |
| --- | --- | --- |
| 20 Hz(학습 plant) | 48/48, 47/48 | 48/48, 47/48. 재포착 0회로 결과 동일 |
| 10 Hz(Isaac 간격) | 31/48, 30/48. lockout 9건, 패드 밖 접촉·envelope·timeout | **48/48, 47/48**. lockout 0건, 재포착은 episode의 35 % |

- local을 Isaac의 영상 간격으로 솎으면 Isaac과 같은 종류의 실패가 재현되고, 재포착이 이를 없앤다.
- G3에서 기각했던 10 Hz 보정(당시 lockout 11/16)이 이제 가능해졌다. 다만 이번 pilot은 관측 하나만 바꾸기 위해 학습 plant를 20 Hz로 둔다.
- 남은 Isaac 차이(§10.7의 EKF 과도 오차, PX4 기울기 overshoot)는 이 변경으로 고쳐지지 않는다.

### 11.4 G4 pilot 4 (`g4-pilot/4`)

**등록(실행 전):** pilot 3의 프로토콜을 그대로 쓰고 관측만 observation/3으로 바꾼다. Raw PPO, 모델 seed 0·1, 600 update × 12 episode, 같은 curriculum·PPO·판정 기준(시험 seed 48개, det ≥ 50 %, sampled ≥ 25 %). pilot 3과 Isaac 평가 두 번은 `g4_pilot_v3/`, `isaac_eval_v1/`, `isaac_eval_v2/`에 보관했다.

**결과 (`g4_pilot.json`, 등록 예산 완주, 합계 21분): G4 passed.**

| 모델 seed | 시험 deterministic | 시험 sampled | (pilot 3) |
| --- | --- | --- | --- |
| 0 (update 582) | 48/48 | 47/48 (UNSAFE 1) | 48/48, 46/48 |
| 1 (update 599) | 47/48 (UNSAFE 1) | 47/48 (UNSAFE 1) | 46/48, 44/48 |

- 공칭 난이도에는 update 62와 70에 도달했다(pilot 3: 71, 69).
- 학습 기록은 update 3부터 pilot 3과 달라졌다. 학습 초반의 거친 탐색에서 재포착이 실제로 일어났다는 뜻이다.
- local에서 Isaac 간격(10 Hz)으로 솎아 평가하면 47/48, 48/48이고 lockout은 0건이다(`observation_v3_checks/pilot4_summary.txt`). 20 Hz에서는 48/48, 48/48이다.

### 11.5 Isaac 재평가 등록 (`isaac-eval/3`, 비행은 사용자 승인 후)

- 대상: pilot 4의 `best.pt` 2개(update 582, 599). deterministic, owned stack, 궤적 기록, 중단 시 이어서 비행.
- seed: 994021–994030. `/1`(994001–994010)과 `/2`(994011–994020)의 seed는 재사용하지 않는다. `/2`의 궤적이 observation/3 설계의 근거였으므로, 같은 seed로 평가하면 낙관적으로 편향된다.
- 판정: checkpoint마다 deterministic ≥ 50 %, 모든 정지 확인.
- 이 변경이 고치지 않는 것: §10.7의 EKF handover 과도 오차와 PX4 기울기 overshoot는 그대로다. 따라서 `/1`·`/2`의 실패 중 기울기 유형(B)과 lockout 없는 시야 상실(A 6/14)은 남을 수 있다.
- 비행 전에 고정한 seed별 handover 교란(재포착 전후 실패 양상을 비교하기 위한 기록이다. `/2`의 경계 0.55/0.62 m/s는 재포착이 없을 때의 경계다):

| seed | 수평 교란 \|v_xy0\| (m/s) | 수직 교란 (m/s) | 외란 \|F_xy\| (N) |
| --- | --- | --- | --- |
| 994021 | 0.74 | +0.14 | 0.38 |
| 994022 | 0.43 | -0.34 | 0.66 |
| 994023 | 0.81 | -0.95 | 0.54 |
| 994024 | 1.04 | +0.20 | 0.16 |
| 994025 | 0.53 | -0.41 | 0.20 |
| 994026 | 0.49 | +0.07 | 0.44 |
| 994027 | 0.58 | -0.11 | 0.70 |
| 994028 | 1.12 | -0.39 | 0.71 |
| 994029 | 0.45 | +0.94 | 0.52 |
| 994030 | 1.02 | -0.78 | 0.81 |

**비행 승인(2026-10-29, 사용자 "go").** 결과는 아래 §11.6에 적는다.

### 11.6 `isaac-eval/3` 결과

**결과 (`isaac_eval.json`, 20 episode 모두 판정, 모든 정지 확인, 중단 없음, 약 13분): 통과(isaac_eval_passed = True).** BC 없이 학습한 정책이 이 경로에서 등록 기준으로 Isaac 평가를 통과한 첫 결과다.

| 모델 seed | Isaac deterministic | 상태 |
| --- | --- | --- |
| 0 (update 582) | **6/10 (60 %)** | SUCCESS 6, envelope 4 |
| 1 (update 599) | **7/10 (70 %)** | SUCCESS 7, envelope 3 |

- **이전과 비교:** pilot 3 checkpoint는 `/1` 7/20, `/2` 5/20, 합계 30 %였다. pilot 4 checkpoint는 13/20(65 %)이다. 두 checkpoint의 차이는 관측의 재포착 규칙 하나다(학습 결과는 그 때문에 달라졌다).
- **lockout 0건:** 실패 7건 모두에서 "패드가 시야 안인데 1 s 넘게 갱신 없음"이 없었다. `/2`에서는 첫 접근 실패 14건 중 8건이 lockout이었다.
- **큰 교란에서도 착륙:** `/2`에서는 수평 교란 0.62 m/s 이상인 seed가 모두 실패했다. 이번에는 994024(1.04 m/s)에서 두 정책 모두 착륙했고, 994021(0.74)에서도 seed 1이 착륙했다.
- **남은 실패:**
  - 5건은 handover 직후 1 s 안에 기울기가 21.9–23.2°에 닿았다(994021·994023·994028·994030). §10.7의 EKF 과도 오차와 PX4 overshoot 경로이며, 재포착으로 고치지 않은 부분이다.
  - 2건(seed 0의 994028 10.1 s, 994030 15.4 s)은 최대 기울기가 약 21°였고, 패드에서 2.3–7.3 m 떨어진 곳에서 패드 면 아래로 내려갔다. lockout은 없었다.
- **해석:** 20 episode, 두 checkpoint의 작은 표본이며, 판정 기준을 겨우 넘긴 수준이다. 강건성이나 arm 비교의 근거가 아니다. 남은 실패의 대부분은 handover 기울기다. 다음 개선 후보는 §10.7의 (a) handover 과도 오차 DR과 (b) 명령 급변·기울기 비용이다.


## 12. handover 과도 오차 대응: 학습 전용 측정 오차 + 보상 v3 (G4 pilot 5)

**사용자 결정(2026-10-29):** "handover 직후 측정 오차를 학습 때 무작위로 넣기 / 공통 보상에 명령 급변·기울기 비용 추가 / go".

### 12.1 변경 (두 가지만)

1. **학습 전용 handover 자기 상태 오차**(`g4_pilot.training_randomization`, `backend._ingest`):
   - 학습 episode에서만, reset 뒤 관측기로 들어가는 자기 속도에 e₀·exp(−t/τ)를 더한다. 자기 위치와 영상에 짝지어진 자기 위치에는 그 적분 e₀·τ·(1−exp(−t/τ))를 더한다.
   - e₀는 수평 방향이 균일하고 크기는 U(0, 1.5) m/s, 수직은 U(−0.4, 0.4) m/s다. τ는 U(0.15, 0.5) s다. 모두 episode seed로 뽑는다.
   - plant, truth, 판정, 보상은 그대로다. 검증·시험·Isaac은 오차를 넣지 않는다.
   - 근거: §10.7에서 Isaac EKF가 결정 사이에 ±0.6 m/s 튀었고, 상대 속도 추정 오차가 1.2–1.4 m/s, 부호 반전이 0.2–0.4 s였다.
2. **보상 v3**(`spatial-current-reward/3`, task v3):
   - 명령 변화 비용 3.0 × mean((u − u_prev)²)를 결정마다 준다. u는 정규화 행동이고, handover 시점의 u_prev는 hover(0)다.
   - 기울기 비용 20/s × max(0, (tilt − 10°)/11°)²를 준다. tilt는 truth의 추력축 기울기다.
   - 판정, 종료 표, 속도장 potential은 그대로다.

### 12.2 크기를 정한 측정 (pilot-4 정책, local, seed 998001–998048, `pilot5_design_checks/`)

- **1차 값(명령 변화 1.0, 기울기 5/s from 12°, 오차 0.8 m/s)은 너무 약했다.**
  - 비용은 episode당 −2 / −0.4였다.
  - 부호 반전은 48 중 3–5 episode에서 약 0.03 s뿐이었다.
- **등록 값에서:**
  - 명령 변화 비용은 거친 지금 정책이 −6~−7, 출력을 부드럽게 한 정책(α 0.5)이 약 −2다. 착륙률은 같다(47/48).
  - 기울기 비용은 −2~−4다. 착륙 episode의 return은 여전히 약 +20이다.
  - 오차를 켜면 부호 반전이 48 중 15–18 episode에서 나타난다(해당 episode 평균 약 0.27 s, Isaac 0.2–0.4 s).
- **한계:**
  - 지금 정책은 local에서도 handover 직후 한 결정에 2.4–2.8 m/s²씩 명령을 바꾸고, 최대 기울기 p90이 19–20°다. Isaac의 자세 overshoot(1.5–1.7배)에서 여유가 없다는 뜻이다.
  - local plant는 그 overshoot를 재현하지 않으므로, 오차를 켜도 local 착륙률은 45–47/48로 거의 그대로다.

### 12.3 G4 pilot 5 (`g4-pilot/5`)

**등록(실행 전):** pilot 4의 프로토콜(observation/3, PPO, 예산, curriculum, 판정 기준)에 위 두 변경만 더했다. 이어서 새 seed 994031–994040으로 `isaac-eval/4`를 등록했다. 사용자 "go"로 Isaac 평가까지 진행한다.

**첫 실행을 운영자가 너무 일찍 중단했다(기록, `g4_pilot5_stopped/`).**
- seed 0은 update 14부터 약 340까지 난이도 0에서 **12/12가 1.2 s 만에 패드로 떨어지는 급강하**(UNSAFE_CONTACT)에 머물렀다. 이를 붕괴로 판단해 update 414에서 중단했다.
- 급강하의 원인은 명령 변화 비용이다. 이 비용은 **표본 행동**의 변화에 부과된다. 초기 탐색 σ(latent 0.37)에서는 잡음만으로 결정당 약 −0.75가 붙어, episode를 빨리 끝내는 것(급강하 −40)이 조심스러운 시도보다 나았다. §12.2의 측정은 deterministic 정책으로만 했기 때문에 이 효과를 보지 못했다. 설계 측정의 한계다.
- 그러나 log σ가 약 −1.84까지 줄자 잡음 비용이 작아졌다. 정책은 update 342 무렵 급강하에서 빠져나와 update 404에 난이도 1.0에 도달했다(첫 검증 det 0.71, sampled 0.46). 즉 중단은 성급했다.
- 같은 등록을 처음부터 다시 실행한다. seed가 고정이라 학습은 결정론적이다.
- 이 학습 경로에서는 600 update 가운데 약 340을 급강하 국면에 쓴다. 공칭 난이도에서 학습하는 시간이 pilot 4(약 530 update)보다 훨씬 짧다.

**결과 (재실행, `g4_pilot.json`, 등록 예산 완주): G4 passed. 다만 pilot 4보다 약하다.**

| 모델 seed | 시험 deterministic | 시험 sampled | 공칭 난이도 학습 update | (pilot 4) |
| --- | --- | --- | --- | --- |
| 0 (update 474) | 40/48 (83 %) | 36/48 (75 %) | 195 | 48/48, 47/48 |
| 1 (update 580) | 46/48 (96 %) | 43/48 (90 %) | 189 | 47/48, 47/48 |

- 재실행은 첫 실행과 같은 경로를 밟았다(update 300의 log σ, update 404 승급까지 같음).
- **의도한 행동 변화는 생겼다**(local, seed 998001–998048, `pilot5_design_checks/pilot5_policies.txt`):
  - handover 직후 1.5 s 안의 최대 명령 급변이 2.40–2.59 → **1.21–1.30 m/s²**로 줄었다.
  - 최대 기울기 평균이 14.7–15.4° → **10.2–10.6°**, p90이 19.1–20.0° → **14.1–15.4°**로 줄었다.
  - 대가로 model 0의 local 착륙률이 40/48로 낮다. 측정 오차를 켜면 38/48, 43/48이다.

### 12.4 `isaac-eval/4` (사용자 "go")

- 대상: pilot 5의 `best.pt` 2개(update 474, 580). seed 994031–994040, deterministic, 기준 checkpoint마다 ≥ 50 %.
- 비행 전에 고정한 seed별 handover 교란:

| seed | 수평 교란 \|v_xy0\| (m/s) | 수직 교란 (m/s) | 외란 \|F_xy\| (N) |
| --- | --- | --- | --- |
| 994031 | 1.15 | -0.13 | 1.04 |
| 994032 | 1.08 | -0.06 | 0.25 |
| 994033 | 0.80 | +0.27 | 1.05 |
| 994034 | 0.94 | +0.22 | 0.81 |
| 994035 | 0.30 | -0.14 | 0.18 |
| 994036 | 0.41 | +0.07 | 0.47 |
| 994037 | 0.97 | -0.95 | 0.59 |
| 994038 | 0.89 | +0.66 | 0.48 |
| 994039 | 0.42 | +0.75 | 0.68 |
| 994040 | 1.21 | +0.41 | 0.65 |

**결과 (`isaac_eval.json`, 20 episode 모두 판정, 모든 정지 확인, 중단 없음, 약 13분): 통과(isaac_eval_passed = True).**

| 모델 seed | Isaac deterministic | 상태 |
| --- | --- | --- |
| 0 (update 474) | **6/10** | SUCCESS 6, envelope 4 |
| 1 (update 580) | **5/10** | SUCCESS 5, envelope 4, 패드 밖 접촉 1 |

**`/3`(pilot 4, 13/20)과 비교** (seed 집합이 달라 직접 비교는 아니다. `/4` seed는 10개 중 7개의 수평 교란이 0.8 m/s 이상이고, `/3`은 0.74 m/s 이상이 5개였다):

| 지표 (Isaac, 처음 1.5 s) | `/3` pilot 4 | `/4` pilot 5 |
| --- | --- | --- |
| 최대 명령 급변 중앙값 / p90 | 2.88 / 4.01 m/s² | **1.19 / 1.59 m/s²** |
| 최대 기울기 중앙값 / p90 | 16.1° / 22.1° | **12.1° / 19.6°** |
| 1.5 s 안의 기울기 실패 | 5/20 | **2/20** |
| 그 뒤의 실패(패드를 놓친 뒤 패드 면 아래로 하강, 패드 밖 접촉) | 2/20 | **7/20** |
| 추적기 lockout | 0 | 0 |

- **의도한 효과는 Isaac에서도 나타났다.** 명령 급변이 절반 이하로 줄었고, 기울기와 handover 기울기 실패도 줄었다.
- **대신 늘어난 실패**(궤적 확인, model1 994034, model0 994040):
  1. 큰 handover 교란 뒤, 또는 약 1 s 시점의 상대 속도 추정 오차(+1.16 m/s, 실제 0.04) 뒤에 수평으로 1.6–2.5 m 벗어난다.
  2. 높이 0.8–1.2 m에서 패드가 시야를 벗어난다(lockout이 아니라 실제 시야 밖이다).
  3. 정책은 멈춘 추정치를 믿고 수평 오차를 거의 줄이지 못한 채 계속 하강한다.
  4. 패드에서 3–4 m 떨어진 곳에서 패드 면 아래 0.2 m에 닿아 envelope 위반이 된다.
  - 정책이 부드러워진 만큼 큰 오차를 회복하는 속도도 느려졌다. 시야를 잃은 뒤 다시 올라가 찾는 행동은 여전히 학습되지 않았다.
- **학습 시간의 차이:** pilot 5는 명령 변화 비용의 잡음 부담 때문에 공칭 난이도 학습이 약 190 update뿐이었다(pilot 4는 약 530). local 착륙률도 model 0이 83 %로 낮다. 따라서 이 결과는 보상 v3의 효과와 학습 부족이 섞여 있다.

**다음 후보(사용자 결정):**
- (a) **짝 비교 비행:** pilot-4 checkpoint를 `/4`의 seed 994031–994040으로 비행한다. 이 seed는 pilot 4 설계에 쓰이지 않았으므로 편향이 없다. seed 난이도와 정책 효과를 분리할 수 있다. 20 episode, 약 14분이다.
- (b) **잡음 부담 제거(pilot 6):** 명령 변화 비용이 탐색 잡음에 부과되지 않게 한다(예: 학습 난이도에 따라 가중치를 0에서 올리거나, 잡음 수준 이하의 변화는 무시). 공칭 난이도 학습 시간을 되찾는다.
- (c) **시야 상실 시 회복을 보상에 넣기:** 속도장이 관측의 영상 나이를 써서, 오래 패드를 보지 못하면 하강 대신 상승(시야 확보)을 요구하게 한다. truth가 아닌 측정 상태를 쓰므로 정책 입력과 일관된다.

### 12.5 짝 비교 비행: pilot 4를 `/4`의 seed로 (사용자 선택 "a")

- **목적:** `/4`에서 pilot 5가 11/20을 기록했다. 그 seed가 어려웠던 탓인지 pilot 5 정책 탓인지를 가린다.
- **방법:** pilot 4의 `best.pt` 2개(update 582, 599, `g4_pilot_v4`)를 `/4`와 같은 seed 994031–994040에서 같은 순서와 조건(deterministic, owned stack, 궤적 기록)으로 비행한다. 이 seed는 pilot 4의 설계·선택 어디에도 쓰이지 않았으므로, pilot 4에 대해 편향이 없다.
- **판정이 아닌 진단이다:** `--stage isaac-trace --checkpoint-root g4_pilot_v4 --pairs all`로 실행하고, `isaac_eval.json`은 바꾸지 않는다.
- **로드 조건:** pilot 4 checkpoint는 보상 v2(task v2)로 학습됐다. 진단 비행에서만, algorithm·feature·graph·calibration hash가 모두 같고 task_hash만 다를 때 로드한다(`allow_foreign_task`). checkpoint의 task_hash는 결과에 기록한다.
- **안전장치:** 보관된 기록의 checkpoint 경로가 지금 pilot 5의 파일을 가리키고 있었다. 그래서 지정한 root의 `g4_pilot/`에서만 찾고, 저장된 update 번호가 그 pilot의 기록과 다르면 거부한다.
- **해석:** seed별로 두 정책의 결과를 짝지어 비교한다(성공/실패 2×2 표와 실패 유형).

**결과 (`isaac_trace/20261029T020617Z/`, 20 episode, 모든 정지 확인, 중단 없음): pilot 4 12/20, pilot 5 11/20. 차이 없음.**

| 같은 seed 20쌍 | pilot 5 성공 | pilot 5 실패 |
| --- | --- | --- |
| pilot 4 성공 | 11 | 1 (1:994031, 패드 밖 접촉) |
| pilot 4 실패 | 0 | 8 |

| 지표 (Isaac, 같은 seed) | pilot 4 | pilot 5 |
| --- | --- | --- |
| 착륙 | 12/20 | 11/20 |
| handover 기울기 실패(1.5 s 안) | 5 | 2 |
| 그 뒤 실패 | 3 | 7 |
| 처음 1.5 s 최대 기울기 중앙값 / p90 | 16.9° / 23.1° | 12.2° / 19.6° |
| 처음 1.5 s 명령 급변 중앙값 | 2.95 m/s² | 1.20 m/s² |

- **seed 난이도:** 두 정책이 모두 실패한 쌍이 8개다. `/4`(11/20)가 `/3`(13/20)보다 낮았던 이유는 seed가 더 가혹했기 때문이고, pilot 5가 나빠져서가 아니다. pilot 4의 Isaac 기록은 `/3`과 이번 비행을 합쳐 25/40이다.
- **pilot 5의 효과:** pilot 4가 handover 기울기로 떨어진 5쌍 중 3쌍(0:994034, 0:994037, 1:994034)에서 pilot 5는 handover를 **넘겼다**. 그러나 그 뒤 패드를 놓쳐 8–19 s에 실패했다. 기울기·명령 급변 비용과 측정 오차 학습은 실패를 "handover 기울기"에서 "회복 실패"로 옮겼을 뿐, 착륙 수는 늘리지 못했다.
- **현재의 병목:** 큰 교란이나 추정 오차 뒤에 시야를 잃으면, 정책이 다시 올라가 패드를 찾지 못한다. 다음 개선은 이 회복 행동이다(§12.4의 (c)). 명령 급변 비용의 잡음 부담을 없애 공칭 난이도 학습 시간을 되찾는 것(§12.4의 (b))도 함께 필요하다.

## 13. 시야 상실 회복: 보상 /4와 학습 전용 영상 끊김 (G4 pilot 6)

**사용자 결정(2026-10-29):** "시야를 잃었을 때 올라가게 하는 보상 / 명령 급변 비용의 잡음 부담 제거 / go". §12.4의 (c)와 (b)를 함께 진행한다. 측정 자료는 `results/.../pilot6_design_checks/`에 있고, seed 995001–995048은 진단 범위로 막았다.

### 13.1 변경

1. **보상 /4의 시야 상실 속도장**(`spatial-current-reward/4`, task v4, `VelocityField.lost_view`):
   - 정책 관측의 영상 나이(마지막으로 받아들인 촬영 시각부터의 시간)가 0.5 s를 넘으면 하강 요구가 상승 요구로 바뀐다. 1.0 s에서 완전히 바뀐다.
   - 상승 요구는 min(0.5 m/s, 1.0/s × (2.5 m − h))다. 접촉 높이 기준 2.5 m(카메라 시야가 다시 넓어지는 높이)까지 오르고, 그 위에서는 고도를 유지한다. 수평 요구는 그대로다.
   - **착륙의 마지막 무시야 구간은 예외다.** 높이 0.5 m 아래이면서 패드 중심에서 0.75 m 안이면 하강을 계속 요구한다. 두 경계는 0.25 m에 걸쳐 완만하게 바뀐다.
   - 영상 나이는 truth가 아니라 정책이 보는 값이다. 따라서 요구가 정책 입력의 함수다. 처음 프레임을 받기 전에는 나이를 무한대로 본다.
   - 판정, 종료 표, 나머지 potential, 기울기 비용은 그대로다.
2. **명령 급변 비용 제거**(보상 /3의 3.0 × mean((u − u_prev)²)):
   - 이 비용은 표본 행동에 부과됐다. 초기 σ에서는 탐색 잡음만으로 결정당 0.65를 냈다. 시간 비용의 22배다.
   - 데드존(|Δu| > δ만 부과)으로는 해결되지 않는다. 막고 싶은 급변(정규화 0.5–1.2)과 잡음(Δu 표준편차 0.47)의 크기가 같다. δ를 키우면 잡음과 신호가 함께 줄어든다(표 13.2-3).
   - 난이도에 따라 가중치를 올리는 방법도 안 된다. 난이도 1.0에 도달할 때 log σ는 −1.15 ~ −1.3(pilot 4)이다. 그때도 잡음 부담이 0.42/결정이다.
   - 정책 손실에 평활 항을 더하는 방법(CAPS)은 가이드가 금지한다("추가 head/loss"). 따라서 쓰지 않는다.
   - **기울기 비용은 남긴다.** plant의 자세 응답이 탐색 잡음을 걸러 주므로 부드러움을 유도하는 항으로 남는다(초기 정책 0.000/결정, pilot 4 0.05, pilot 5 0.007).
3. **학습 전용 영상 끊김**(`g4_pilot.training_randomization.vision_blackout`):
   - 학습 episode의 절반에서, handover 뒤 U(0.5, 4.0) s부터 U(1.0, 3.0) s 동안 관측기에 영상 프레임을 주지 않는다. episode seed로 뽑는다.
   - plant, truth, 판정은 그대로다. 검증·시험·Isaac은 이를 넣지 않는다. pilot 5의 handover 자기 상태 오차는 유지한다.
   - **필요한 이유:** 학습된 정책은 local에서 고도 중에 시야를 잃는 일이 거의 없었다(48 episode 중 0–5, 결정의 2 % 이하). 이 상태에 보상을 줘도 학습 중에 그 상태를 거의 만나지 않으면 배울 것이 없다.

### 13.2 설계 측정 (local, seed 995001–995048; Isaac은 기존 궤적)

1. **착륙의 마지막 무시야 구간**(pilot 4/5, deterministic, 183회 착륙):
   - 영상 나이가 0.5 s를 넘은 채 끝나는 구간은 높이 ≤ 0.43 m(p90 0.15 m), 중심에서 ≤ 0.56 m(p90 0.27 m)에서 시작한다.
   - 그래서 예외 구간을 0.5 m / 0.75 m로 정했다.
2. **기록된 Isaac 60 episode에 새 속도장을 적용**(`isaac_field.txt`):
   - 착륙 36개에서는 상승 요구(가중치 ≥ 0.5)가 한 번도 없었다.
   - 늦은 envelope 위반 10개(패드를 놓친 실패)에서는 모두 2.6–4.4 s에 상승을 요구했다. 각 episode가 끝나기 4.3–15.9 s 전이다.
   - 요구하지 않은 늦은 실패는 둘이다. 6.6 s의 패드 밖 접촉과 7.3 s의 과속 접촉으로, 시야 상실이 원인이 아니다.
3. **명령 급변 비용의 잡음 부담**(결정당, 3 × mean(max(0, |Δu| − δ)²)):

| δ | 0 | 0.3 | 0.5 | 0.8 |
| --- | --- | --- | --- | --- |
| 초기 정책, 표본, d=0 | 0.649 | 0.199 | 0.075 | 0.011 |
| 초기 정책, log σ −1.25, d=1 | 0.420 | 0.094 | 0.026 | 0.002 |
| 신호: 처음 1.5 s 합, pilot 4 − pilot 5 | 2.5 | 1.0 | 0.5 | 0.15 |

   - 기체 각속도도 구별하지 못한다. 초기 잡음의 p99가 73°/s이고, pilot 4의 handover 최대 각속도 중앙값은 76°/s다.
   - 기울기 비용(10° 초과)만 잡음 0, 거친 정책 0.05, 부드러운 정책 0.007로 갈린다.
4. **보상 /4로 본 영상 끊김**(`reward4_check.txt`, 매 episode에 1–3 s 끊김, d=1):

| 정책 | 끊김 없음 | 끊김, 정책 그대로 | 끊김, 나이 > 0.5 s이고 추정 높이 > 0.5 m일 때만 상승 |
| --- | --- | --- | --- |
| pilot 4 seed 0 | 48/48, +28.0 | 31/48, −2.5 | **47/48, +23.9** |
| pilot 4 seed 1 | 48/48, +27.6 | 30/48, −4.3 | **47/48, +23.0** |
| pilot 5 seed 0 | 43/48, +21.5 | 32/48, +0.9 | **48/48, +26.7** |
| pilot 5 seed 1 | 44/48, +23.8 | 37/48, +12.8 | **46/48, +25.1** |

   - local에서도 멈춘 추정치로 하강하면 패드 밖 접촉이 6–16회 생긴다. 이 상황은 실제로 위험하다.
   - 상승만 덧씌우면 착륙과 보상 /4의 return이 모두 오른다. 보상이 옳은 방향을 가리킨다.
   - 끊김이 없을 때 새 속도장은 거의 작동하지 않는다(결정의 0–1.7 %, 모두 pilot 5의 실패 episode).
5. **초기 정책, 표본, d=0, 전체 학습 무작위화:** 결정당 비용은 시간·제어 −0.035, 기울기 0.000이다. 잡음 부담이 없다(pilot 5는 −0.65였다).

### 13.3 등록 (실행 전)

- **G4 pilot 6(`g4-pilot/6`):** pilot 5의 프로토콜(관측 /3, PPO, 예산 600 update × 12, curriculum, 판정 기준, handover 자기 상태 오차)에 위 세 변경만 더했다.
- **`isaac-eval/5`:** pilot 6의 `best.pt` 2개를 새 seed 994041–994050에서 평가한다. deterministic, checkpoint마다 ≥ 50 %다.
- **짝 비교(진단, 판정 아님):** 같은 pilot-6 checkpoint를 seed 994031–994040에서도 비행한다(`isaac-trace --pairs`). 같은 seed에서 pilot 4는 12/20, pilot 5는 11/20이었다.
- **보관:** pilot 5는 `g4_pilot_v5/`(설정 스냅샷 포함), `isaac-eval/4`는 `isaac_eval_v4/`에 보관했다.
- **비행 전에 고정한 seed별 handover 교란**(`sample_domain_randomization`):

| seed | 수평 \|v_xy0\| (m/s) | 수직 (m/s) | 외란 \|F_xy\| (N) |
| --- | --- | --- | --- |
| 994041 | 0.47 | −0.70 | 0.74 |
| 994042 | 0.94 | +0.60 | 0.67 |
| 994043 | 0.56 | −0.36 | 0.71 |
| 994044 | 0.52 | −0.35 | 0.91 |
| 994045 | 0.58 | +0.99 | 0.68 |
| 994046 | 1.07 | +0.86 | 0.76 |
| 994047 | 0.47 | +0.14 | 0.81 |
| 994048 | 1.09 | −0.61 | 0.65 |
| 994049 | 0.88 | −0.78 | 0.57 |
| 994050 | 0.62 | +0.25 | 0.35 |

- **한계:** 보상 /4, 급변 비용 제거, 영상 끊김을 함께 바꿨으므로 효과를 분리할 수 없다. 영상 끊김 동안 실제로는 패드가 시야 안에 있다. 반면 Isaac의 실제 상실에서는 추정도 틀려 있다. 상승 행동이 그 차이를 넘어 일반화하는지는 Isaac 비행으로만 확인할 수 있다.

### 13.4 G4 pilot 6 결과 (`g4_pilot.json`, 등록 예산 완주): **G4 passed**

| 모델 seed | 시험 deterministic | 시험 sampled | 난이도 1.0 도달 update | (pilot 4 / pilot 5) |
| --- | --- | --- | --- | --- |
| 0 (update 536) | 46/48 (96 %) | 43/48 (90 %) | 77 | 48/48 · 47/48 / 40/48 · 36/48 |
| 1 (update 389) | 48/48 (100 %) | 47/48 (98 %) | 70 | 47/48 · 47/48 / 46/48 · 43/48 |

- **잡음 부담 제거가 효과를 냈다.** 난이도 1.0 도달이 pilot 5의 update 405–411에서 77·70으로 돌아왔다(pilot 4는 63·71). 초기에는 제자리 비행 timeout이 많았으나(update 12에 9/12), update 15부터 착륙이 나왔다. 급강하 국면은 없었다.
- **local 비교**(`pilot6_design_checks/pilot6_policies.txt`, seed 995001–995048, d=1, deterministic):

| 조건 | pilot 4 | pilot 5 | pilot 6 |
| --- | --- | --- | --- |
| 무작위화 없음 | 48, 48 | 43, 44 | **48, 48** |
| handover 자기 상태 오차 | 48, 48 | 43, 45 | **48, 48** |
| 1–3 s 영상 끊김(매 episode) | 35, 33 | 35, 40 | **44, 45** |
| 처음 1.5 s 최대 명령 급변 p50 (m/s²) | 2.30–2.45 | 1.37 | 1.86–1.88 |
| 처음 1.5 s 최대 기울기 p50 / p90 | 15.5–16.2° / 19.8–20.4° | 10.2–10.6° / 14.2–15.9° | 11.7–12.0° / 16.1–17.3° |

- **의도한 상승은 학습되지 않았다**(`lost_view_behaviour.txt`):
  - 상승 가중치 ≥ 0.5인 결정 중, 높이 0.5–3 m에서 실제로 상승한 비율은 pilot 4·5·6 모두 **0 %**다.
  - pilot 6은 하강을 늦추고(−0.30 → −0.20 m/s) 패드 위 수평 위치를 지킨다. 1 m 넘게 벗어난 결정이 0–6개로, pilot 4·5의 123–201개보다 훨씬 적다. 그래서 끊김에 강해졌다.
- **원인(구조적):**
  - 보상 v2부터 dense 항은 할인하지 않은 potential 차이다. PPO 할인은 τ = 70 s(결정당 γ = 0.9986)다. 따라서 속도장이 최적 정책에 미치는 힘은 결정당 (1−γ)·Φ ≈ 0.0014·Φ뿐이다.
  - 시야 상실 시 속도 오차 0.7 m/s의 Φ 차이는 −2.8이다. 결정당으로는 −0.004로, 시간 비용(−0.03)의 1/8이다.
  - 마지막 무시야 구간은 예외라서 착지 순간의 Φ도 상승을 요구하지 않는다.
  - 즉 속도장은 학습 경로(GAE의 신용 배분)만 바꾸고 최적 정책은 거의 바꾸지 않는다. 정책은 착륙 결과와 시간으로 최적화된다.
  - local에서는 천천히 내려가 패드 위에 머무는 것이 상승과 착륙률이 비슷하고(44–45 vs 상승 덧씌움 46–48/48) 시간은 덜 든다. 그래서 상승을 고를 이유가 없었다.
- **함의:** 상승이 실제로 필요한 상황은 추정이 틀린 채 패드가 시야 밖으로 나간 경우다. local plant는 이를 거의 만들지 않는다. 1.5 m 크기의 느린 자기 위치 오차를 넣어도 pilot 6은 48/48 착륙했고, 시야를 잃지 않았다. 상승을 정책에 강제하려면 속도 오차에 potential이 아닌 직접 비용을 주거나, Isaac처럼 추정이 틀어져 패드를 놓치는 상황을 학습에 넣어야 한다. 어느 쪽이 필요한지는 Isaac 결과로 판단한다.

### 13.5 `isaac-eval/5` (사용자 "go")

**결과 (`isaac_eval.json`, 20 episode 모두 판정, 모든 정지 확인, 중단 없음, 13.5분): 통과(isaac_eval_passed = True).**

| 모델 seed | Isaac deterministic | 상태 |
| --- | --- | --- |
| 0 (update 536) | **6/10** | SUCCESS 6, envelope 3, 패드 밖 접촉 1 |
| 1 (update 389) | **6/10** | SUCCESS 6, envelope 4 |

- **결과는 seed가 정했다.** 두 정책의 결과가 10개 seed 모두에서 같았다.
- **실패의 종류**(`pilot6_design_checks/isaac_eval5_traces.txt`):
  - handover 기울기 실패 4건(994046, 994049 × 두 정책)은 1 s 안에 23–25.5°까지 기울었다.
  - 늦은 실패는 4건이다. 994048 × 두 정책은 3.2–3.7 s에 높이 0.57–0.69 m, 1.4–1.6 m 벗어난 곳에서 패드를 놓쳤다. 이후 −0.30 / −0.14 m/s로 계속 내려가 패드 면 아래에 닿았다. 994041의 model 1은 4.2 s에 시야를 잃은 뒤 −0.04 m/s로 거의 멈춘 채 21 s에 패드 면 아래로 내려갔다. model 0은 5.1 s에 패드 밖에 접촉했다.
  - **Isaac에서도 시야를 잃은 뒤 상승한 episode는 없다.** §13.4의 local 측정과 같다.
- **handover 추력과 기울기 실패**(기록된 추적 Isaac 100 episode, `isaac-eval/3`–`/5`와 두 짝 비교 비행, `pilot6_design_checks/early_thrust.txt`):

| 처음 1.3 s 평균 실제 수직 가속 | handover 기울기 실패 |
| --- | --- |
| < −0.6 m/s² | 14/46 (30 %) |
| −0.6 ~ 0 | 4/46 (9 %) |
| ≥ 0 | 0/8 |

  - 초기에 추력을 줄여(하강 명령) 자세 제어 여유가 줄어드는 것이 위험 인자다. reference `/12`가 handover 2 s 동안 추력 하한으로 막은 기전과 같은 방향이다(그쪽 기록은 < −0.6에서 5/5). 이 경로에는 감독기가 없으므로, 막으려면 보상이나 학습 조건으로 다뤄야 한다.

### 13.6 짝 비교 비행: pilot 6을 `/4`의 seed로 (등록 §13.3)

- 같은 pilot-6 `best.pt` 2개를 seed 994031–994040에서 같은 순서와 조건으로 비행했다(`isaac_trace/20261029T044813Z/`, 20 episode, 모든 정지 확인, 중단 없음).
- 첫 시도는 한 episode도 날지 못했다. 진단 pair가 "등록된 isaac_eval seed"만 허용했고, 그 검사가 스택을 띄운 뒤에야 실행됐다.
- 고친 점:
  - 이미 진단에 쓴 seed(`DIAGNOSTIC_SEED_RANGES`)도 진단 재비행에 허용한다. 새 seed는 여전히 거부한다.
  - pair 검사를 스택을 띄우기 전, dry-run 단계로 옮겼다(`isaac_eval.check_pairs`).

**결과: pilot 6 15/20 (model 0 7/10, model 1 8/10). 같은 seed에서 pilot 4는 12/20, pilot 5는 11/20이었다.**

| 같은 seed 20쌍 | pilot 6 성공 | pilot 6 실패 |
| --- | --- | --- |
| pilot 4 성공 | 11 | 1 (0:994038, 패드 밖 접촉) |
| pilot 4 실패 | **4** (994033 × 2, 994040 × 2) | 4 (994034 × 2, 994037 × 2) |
| pilot 5 성공 | 10 | 1 (0:994038) |
| pilot 5 실패 | **5** | 4 |

- **새로 착륙한 4쌍은 모두 '시야 상실 → 고도 유지 → 재포착'이었다**(`pilot6_design_checks/isaac_paired_p6_traces.txt`):
  - 4.4–4.7 s에 높이 1.0–1.3 m, 1.2–1.5 m 벗어난 곳에서 패드를 놓쳤다.
  - 그 뒤 하강을 거의 멈췄다(평균 −0.03 ~ −0.19 m/s, 시야를 잃은 동안 최저 0.64–1.23 m).
  - 패드를 다시 잡아 9.3–12.2 s에 착륙했다. pilot 4와 5는 같은 상황에서 계속 내려가 패드 면 아래에 닿았다.
  - local에서 본 '하강을 늦추고 패드 위에 머물기'(§13.4)가 Isaac에서 회복으로 이어진 것이다. 상승은 아니다.
- **남은 실패:**
  - 994034 × 2: 3.1–3.5 s에 2 m 넘게 벗어나 패드를 놓친 뒤 −0.46 / −0.17 m/s로 내려갔다. 이 거리에서는 실제로 올라가야 다시 보인다.
  - 994037 × 2: handover 기울기 실패다(21.5° / 23.7°, 1 s).
  - 0:994038: 4.6 s의 패드 밖 접촉으로, 시야 상실은 없었다.
- **통계적 한계:** 엇갈린 쌍은 pilot 4 대비 4 대 1(정확 이항 양측 p = 0.38), pilot 5 대비 5 대 1(p = 0.22)이다. 방향은 일관되지만 20쌍으로는 우연과 구분되지 않는다.
- **pilot 6의 Isaac 합계는 27/40(68 %)이다**(`/5` 12/20 + 짝 비교 15/20). pilot 4는 25/40, pilot 5는 11/20이었다. 실패 13건은 handover 기울기 6건, 늦은 실패 7건이다.

**다음 후보(사용자 결정):**
- (a) **handover 기울기(6/40):** 초기에 추력을 줄이는 하강이 위험 인자다(표 13.5, < −0.6에서 30 %).
  - 학습 전용으로 자세 servo의 overshoot를 무작위화한다. Isaac의 handover 직후 실현/명령 기울기 비가 1.5–1.7배였다. 이렇게 하면 정책이 그 여유를 배운다.
  - 감독기 없이 `/12`의 추력 하한 효과를 내는 방법이다. 정책 입력에 handover 경과 시간은 넣지 않는다.
- (b) **멀리 벗어난 시야 상실(994034):** 실제 상승이 필요하다.
  - 속도장 오차에 potential이 아닌 직접 비용을 시야 상실 상태에서만 준다. 할인 τ = 70 s에서는 potential이 최적 정책을 거의 바꾸지 못하기 때문이다(§13.4).
  - 또는 학습에 추정이 틀어져 패드를 놓치는 상황을 넣는다.
- (c) **네 비교군 비교(본 실험):** 지금 공통 계약으로 development 단계를 등록한다. Isaac 성능이 60–70 %대에서 안정되었으므로, 공통 계약을 더 바꾸기 전에 표현 비교를 시작할 수도 있다.

## 14. 네 비교군 development (가이드 11) — 과제 v5와 등록

**사용자 결정(2026-10-29):**
- "c"(네 비교군 비교로 넘어가기)를 골랐다.
- Composition 정의로 "과제를 전 방향으로 확장"을 골랐다.
- "make live visualization for 4 comparison … visualized when the training beginning"을 요청했다.

### 14.1 과제 v5: 전 방향 시나리오 (`scenario.py`, `spatial-current-scenario/1`)

- **확장한 이유**(`development_design/factor_probe.txt`, pilot-6 Raw, seed 995001–995048):
  - `/12` plant는 패드 이동 방향을 ±11.5°, 초기 위치 오차를 0.4 m 정사각형에서 뽑았다.
  - 그 정책은 이 분포에서 96/96 착륙했지만, 패드 방향이 90°나 180°이면 0/96이었다.
  - 패드가 0.8 m 떨어진 상태로만 바꾸면 어느 방위에서든 45–48/48이었다.
  - 즉 이 분포 위에서는 Composition 시험이 측정할 것이 없었다.
- **새 분포:**
  - A(시작 시 드론에서 본 패드 방위)와 B(패드 이동 방향)를 각각 4구간(0°, 90°, 180°, 270° 중심)으로 전 방향에서 뽑는다. 초기 거리는 U(0.3, 0.9) m다.
  - 학습, 검증, 분포 내 시험은 12개 조합을 쓴다. Composition 시험은 보류한 `[[0,0],[1,2],[2,3],[3,1]]`만 쓴다.
  - 보류 조합은 각 A·B 구간을 정확히 한 번씩 뺀다. 순환 이동이 아니므로 상대각(B−A)의 네 부류는 모두 학습에 남는다. 이 조건은 config 로더가 강제한다.
- **구현:**
  - `ScenarioLocalBackend`가 `/12` plant의 reset 뒤에 위치와 방향만 바꾸고, 첫 프레임을 새 기하로 다시 만든다.
  - 쉬운 난이도에서의 위치 축소와 카메라 시야 제한은 `/12`와 같게 적용한다.
  - 공유 파일(`spatial/environment.py`)은 바꾸지 않았다.
- **Isaac:** Isaac은 아직 이 시나리오를 날 수 없다. 진입 위치와 패드 방향이 Isaac 쪽 설정과 `sample_spatial_scenario`에서 정해진다. Isaac 평가(가이드 11.6) 전에 Isaac 쪽 시나리오를 추가해야 한다.
- **G4 pilot 7(`g4-pilot/7`, 과제 v5, Raw): 통과.**
  - 시험 deterministic 45/48·41/48, sampled 43/48·37/48이다.
  - 난이도 1.0에 update 100·97에 도달했다.
  - pilot 6은 `g4_pilot_v6/`, `isaac-eval/5`는 `isaac_eval_v5/`에 보관했다.

### 14.2 OOD 강도 (표현 비교군을 학습하기 전에 Raw로만 확정)

- **규칙(측정 전에 선언):** pilot-7 Raw의 두 checkpoint 평균(deterministic, d=1, seed 995001–995048)으로, 성공률이 20–80 % 안에 드는 가장 약한 강도를 고른다(`development_design/ood_calibration.txt`).

| 조건 | 후보 (Raw 성공률) | 선택 |
| --- | --- | --- |
| Motion (패드 가속) | 0.4–0.6: 0.92 / 0.6–0.9: 0.71 / 0.9–1.2: 0.53 | **0.6–0.9 m/s²** |
| Dropout (영상 끊김) | 학습 범위 1–3 s: 0.89 / 3–4 s: 0.67 / 3–5 s: 0.59 / 4–6 s: 0.45 | **3–4 s** |
| Noise (광학 잡음 배율) | ×5: 0.73 / ×10: 0.02 / ×20: 0.00 | **×5** |
| Combined | 위 세 가지를 동시에 | — |

- **Composition은 조정하지 않는다.** 과제가 정한 보류 조합 그대로다. Raw는 Composition OOD에서 0.91로, 분포 내 0.94와 거의 같다. 주 지표에서 비교군 간 차이가 날 여지가 작을 수 있다는 뜻이다. 결과를 해석할 때 이 점을 함께 본다.

### 14.3 등록 (`development_registration.json`, plan hash `d6be7db0…`, 2026-10-29 07:09 UTC)

- **계획**(`config/current_semantics/development_v1.yaml`):
  - 네 비교군 × 학습 seed 10–14 = 모델 20개다.
  - 예산·curriculum·PPO·학습 무작위화는 G4 pilot 7과 같다. 모델당 600 update × 12 episode, 합계 144,000 학습 episode다.
  - checkpoint는 검증 seed 992101–992124(d=1, deterministic·sampled)로 고른다.
  - 평가는 10셀 × 100 episode × 학습 seed다. 학습 seed마다 고유하고 네 비교군이 공유한다. deterministic이며 합계 20,000 episode다.
- **판정(가이드 11.5):**
  - 모든 모델이 검증 기준(det ≥ 0.5, sampled ≥ 0.25)을 넘어야 한다.
  - Composition OOD에서 R-GAT − MLP, R-GAT − Untyped, R-GAT − Raw의 평균이 모두 0보다 커야 한다.
  - ID 다섯 셀 평균이 Raw보다 2 %p 넘게 낮으면 안 된다. 기존 4 OOD 평균도 2 %p 넘게 낮으면 안 된다.
  - OOD 다섯 셀의 unsafe가 Raw보다 1 %p 넘게 늘면 안 된다.
  - 실패한 seed를 제외하거나, 사후에 기준을 조정하거나, 같은 후보를 다시 학습하지 않는다.
- **seed 감사:** 쓰는 seed가 감사·진단·G4 pilot·등록 seed 범위와 겹치지 않음을 로더가 확인한다.
- **실행:** `--stage develop`은 등록 hash와 다르면 거부한다. 모델 단위로 이어서 실행할 수 있다(`development/seed*/<arm>/`). local plant만 쓰며 Isaac은 쓰지 않는다.
- **실시간 대시보드(`dashboard.py`):** develop이 시작되면 `http://127.0.0.1:8773/`이 열린다.
  - 보여주는 것: 진행률, 모델 20개 상태, 비교군별 학습·검증 곡선, 10셀 히트맵, 완료된 seed만으로 계산한 판정 미리보기다.
  - 따로 볼 때는 `--stage dashboard`를 쓴다(읽기 전용). 실행이 끝나면 `development/dashboard.html`에 정적 사본을 남긴다.

### 14.4 결과 (`development.json`, 2026-10-30 00:29, 8.3 h, local plant)

- **등록 판정: 통과(`development_gate_passed`, `confirmation_opened`).**
  - 학습 가능성: 모델 20개 모두 검증 기준을 넘었다(det 0.92–1.00, sampled 0.83–1.00).
  - Composition OOD 평균 차이: R-GAT − MLP +1.8 %p, R-GAT − Untyped +2.4 %p, R-GAT − Raw +4.0 %p. 모두 0보다 크다.
  - guard: ID는 Raw 대비 +4.4 %p, 기존 4 OOD 평균은 +0.95 %p, OOD unsafe는 −0.5 %p다. 모두 통과했다.
- **비교군별 평균 성공률(5 seed × 셀당 100 episode):**

| 셀 | Raw | MLP | Untyped | R-GAT |
| --- | --- | --- | --- | --- |
| Composition ID / OOD | 0.886 / 0.918 | 0.922 / 0.940 | 0.910 / 0.934 | 0.944 / 0.958 |
| Motion ID / OOD | 0.910 / 0.762 | 0.940 / 0.724 | 0.908 / 0.740 | 0.956 / 0.754 |
| Dropout ID / OOD | 0.852 / 0.702 | 0.890 / 0.708 | 0.834 / 0.714 | 0.896 / 0.692 |
| Noise ID / OOD | 0.904 / 0.658 | 0.940 / 0.714 | 0.904 / 0.676 | 0.956 / 0.738 |
| Combined ID / OOD | 0.886 / 0.082 | 0.878 / 0.068 | 0.876 / 0.066 | 0.906 / 0.058 |

- **해석(사후 기술 통계, 판정을 바꾸지 않음; `development_design/development_analysis.txt`):**
  - **차이는 잡음 수준이다.** 학습 seed별 R-GAT − MLP는 [+9, −3, 0, +4, −1] %p(t 0.84), R-GAT − Untyped는 [+6, −4, −3, +6, +7] %p(t 0.99)다. 5 seed의 97.5 % CI 하한은 −5.6, −6.1 %p다. 가이드 13에 따라 "우위를 확인하지 못했다"로 기록한다.
  - **보류 조합이 더 어렵지 않다.** 네 비교군 모두 Composition OOD 성공률이 분포 내 nominal 세 셀(300 episode)과 같거나 높다(+0.6 ~ +2.7 %p). 그래서 Composition 차이는 조합 일반화가 아니라 전반적인 성능 차이를 측정한다. 근거로 R-GAT − MLP의 Composition 차이(+1.8 %p)가 nominal ID 차이(+1.8 %p)와 같다. R-GAT 자신의 OOD−ID 차이(+0.6 %p)는 가장 작은 편이다.
  - **확인 단계의 검정력이 낮다.** 개발의 평균과 표준편차가 그대로라면 시드 20개에서 97.5 % CI 하한 > 0을 얻을 확률은 MLP 대비 약 0.25, Untyped 대비 약 0.36이다. 둘 다 통과해야 하므로 확인 통과 확률은 더 낮다. 확인 단계 비용은 모델 80개, local 약 33 h다.
  - **OOD 실패 방식이 다르다.** 다섯 OOD 셀 합계에서 R-GAT는 SUCCESS가 1600으로 가장 많다. SAFETY_ENVELOPE_VIOLATION(89), UNSAFE_CONTACT(76), TASK_TIMEOUT(96)은 가장 적다. 반면 MISSED_PAD_CONTACT는 639로 가장 많다(Raw 470). 패드를 잃은 뒤 버티지 않고 패드 밖으로 내려앉는 것으로, 13.6의 "재획득 없음" 병목과 같은 실패다.
  - **Combined OOD는 모든 비교군이 6–8 %**이고 unsafe가 77–88 %다. 이 강도에서는 어떤 표현도 착륙하지 못한다.
- **다음 단계(사용자 결정):**
  - 확인 단계(가이드 11.4–11.5)는 아직 구현하지 않았다. 새 seed 20개 감사가 필요하고, 실행 전 승인이 필요하다.
  - 가이드 11.6 Isaac 평가는 Isaac 쪽 전 방향 시나리오 구현과 승인(약 600 episode)이 필요하다.
  - Composition을 더 어렵게 다시 정의하는 것은 개발 결과를 본 뒤의 변경이다. 새 후보와 새 등록으로만 할 수 있고, 사후 변경임을 공개해야 한다.

## 15. 개발 모델의 Isaac 평가 (가이드 11.6, 사용자 "3", 2026-10-30)

### 15.1 Isaac의 과제 v5 지원

- **draw는 한 곳에서만 한다.** 학습기(`current_semantics/scenario.py`)가 episode seed로 방위 A·진행 방향 B·반경 r을 뽑는다. Motion 셀이면 CA 가속 a2도 뽑는다. 이 값을 시나리오 이름 `spatial_current_omni/1:heading:bearing:radius:a2`에 담아 보낸다.
  - 인코딩·파싱은 `spatial/scenarios.py`의 `encode_current_omni`/`parse_current_omni`/`current_omni_motion` 하나다.
  - 게이트웨이(`ontology_rgat_px4/protocol.py`)는 접두어만 보고 그대로 넘긴다.
  - Isaac의 `pad_motion.py`는 같은 함수로 이름을 파싱한다. seed의 CV-CA-CV(t1, t2, v0, a2) 위에 B 방향(과 a2)을 그대로 적용한다. 데크는 진행 방향을 향한다.
- **진입(`landing_world.py`):** 오프셋은 −r(cos A, sin A)다. 고도와 yaw 오차는 seed가 그대로 뽑는다. 수평 draw 두 개는 소비하고 버리므로 이후 난수 순서가 바뀌지 않는다.
  - 진입 yaw는 월드 0° + 오차(±5°)다. 로컬은 모든 진행 방향에서 yaw 0으로 난다. 데크 방향을 따라 yaw를 돌리면 `drone_sinPsi/cosPsi`가 학습에서 한 번도 보지 못한 값이 된다.
- **새로 발견한 기존 차이:** 지금까지 Isaac 진입은 `constrain_camera_visible_entry`가 데크 전체(반경 1.06 m)를 시야의 65 % 안에 넣으려고 오프셋을 거의 0까지 당겼다. 기록된 Isaac episode 132개의 첫 결정 시 수평 오프셋은 중앙값 0.07 m(p90 0.14 m)다. 로컬은 ±0.4 m 정사각형에서 뽑는다.
  - 즉 isaac-eval/1–5는 로컬보다 쉬운 중앙 진입에서 날았다.
  - 전 방향 시나리오에서는 이 당김을 적용하지 않는다(r 0.3–0.9 m). 패드 중심이 화면 안에 있는지는 bridge의 진입 시야 검사(margin 0.85)가 그대로 확인한다. r = 0.9 m, 고도 2.0 m에서 정규화 위치는 ≤ 0.65다.
- **응답 대조(`check_isaac_reset`):** reset 응답의 데크 시나리오·heading·a2, 진입 오프셋(1e-9 m), 진입 yaw(≤ 5°)가 보낸 draw와 다르면 episode를 거부하고 평가를 멈춘다. 옛 코드로 도는 월드가 /12 데크를 날리는 경우도 여기서 잡힌다.
- **Combined ID의 영상 끊김**은 로컬 셀과 같은 Python 쪽 관측 차단(`CurrentSpatialBackend._ingest`, stream 0xD0E7)이다. 그래서 Isaac에서도 허용했다. 광학 잡음 배율은 로컬 plant에서만 적용되므로 Isaac에서는 거부한다.
- 체크포인트 해시는 배포 YAML의 과학 설정만 묶는다(소스는 아님). 그래서 기존 체크포인트와 calibration hash는 바뀌지 않는다.

### 15.2 등록 (`isaac_development_registration.json`, plan hash `b64f8aaa…`, 2026-10-30)

- **계획**(`config/current_semantics/isaac_development_v1.yaml`, 프로토콜 `isaac-development/1`):
  - development 모델 20개(학습 seed 10–14 × 네 비교군)의 `best.pt`를 쓴다. 각 파일의 sha256을 등록했다.
  - 세 조건 × 환경 10개 = 30개 사례를 20개 모델이 모두 난다. 합계 600 episode, deterministic이다.
  - 조건: Composition ID(994101–994110), Composition OOD(994111–994120), Combined ID(994121–994130, 학습 범위 1–3 s 영상 끊김)다.
- **Combined OOD를 빼고 Combined ID를 고른 이유:** 비행 전에 정했다. 로컬에서 모든 비교군이 Combined OOD를 6–8 %만 착륙하고 unsafe가 77–88 %다. Isaac에서는 바닥값만 잴 것이고, 그 광학 잡음 요인은 Isaac에 없다. Combined ID는 지금까지 Isaac의 병목이던 "패드를 잃은 뒤 회복"을 학습 범위 안에서 잰다.
- **순서와 재개:** 조건별로 환경 우선 순서로 난다. 한 환경을 20개 모델이 모두 날고 나서 다음 환경으로 간다. 그래서 중단되어도 모델 간 범위가 같다. 같은 등록이면 재개한다.
- **로컬 짝 비교:** 같은 600개 사례를 로컬 plant(development 셀과 같은 코드 경로)에서도 날아 `local_reference.jsonl`에 둔다.
- **판정 기준은 없다.** 가이드 11.6.5대로 현재 배포의 통합 측정이며, Isaac에서의 우월성 검정이 아니다.
