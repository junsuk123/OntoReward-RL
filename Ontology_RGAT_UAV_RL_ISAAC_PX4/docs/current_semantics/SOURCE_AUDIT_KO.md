# 3D 리팩토링 가이드 §4 출처·계약·실행 환경 감사표

감사일: 2026-10-28 (KST)  
기준: HEAD `cab3dd2123c7eee3968a8c00753f2ba29d55fe71` + 같은 날 작업 트리(커밋되지 않은 사용자 변경 포함)  
방법: 영역별 독립 감사 6건과 반박 검증 6건. 모든 판정은 파일·행을 직접 열어 확인했다.  
범위: 코드·설정·기록의 정적 확인과 순수 함수 호출만 했다. 학습·Isaac·PX4·ROS·MATLAB은 실행하지 않았다.

판정 표기: **확인**(코드가 주장 그대로), **부분**(맞지만 조건·기전이 다름), **반박**, **확인 불가**.  
경로의 `P/`는 `Ontology_RGAT_UAV_RL_ISAAC_PX4/`이다.

## 1. 가이드 §4 행별 판정

| 가이드 행 | 판정 | 근거 (파일:행) | 정밀화 |
| --- | --- | --- | --- |
| 루트 `run.sh` bare 실행이 minimal teacher→BC→PPO→Isaac | 확인 | `run.sh:10-12` → `P/scripts/run_minimal_system.sh`; `P/tools/minimal_full_pipeline.py:326-334` | 실제 순서는 ROS colcon build → teacher → BC → PPO → held-out → stress → Isaac이다. 이 컴퓨터는 Isaac 사전 조건(`/opt/ros/humble`, ASCII ws, `ISAACSIM_PATH`)을 모두 만족하므로 bare 실행은 실제 비행까지 간다. `--help`는 **첫 인자일 때만** 안전하다. `./run.sh system --help`나 `./run.sh --no-isaac --help`는 colcon build를 먼저 실행한다. |
| 루트 `README.md`·`AGENTS.md` 시대 혼재, bare 설명 불일치 | 확인 | `README.md:110,113`("인자 없는 실행은 도움말만") ↔ `run.sh:2,10-12` | README 경로표(`README.md:118-123`)에 system/all/matlab-port* 경로가 없다. README의 활성 계약은 `spatial-reference/2`·`spatial-causal-rgat/5`이고 AGENTS의 활성 계약은 `/12`다. `P/README.md:172-173`, `P/docs/OPERATIONS.md:19,133-134`, `P/tools/run_full_pipeline.py:4`, `P/tests/conftest.py:12-26`의 설명도 낡았다. 문서에 남은 `./run.sh --mode quick`은 이제 minimal 시스템으로 들어간다. |
| `direct_policy/contracts.py` 옛 MATLAB `6082258`, source21D/stable24D 혼재 | 확인 | `contracts.py:22-24` SHA·버전 고정, `:291-313` 21D, `:361-367` stable에 `tracking_bias_x/y`·`handover_progress` 추가 | 21D는 MATLAB 원본이 아니라 포트의 자체 3D 확장이다(고정된 MATLAB 계약은 평면 12D, `runner.py:551-552`). stable 묶음도 `source_sha 6082258`과 같은 `execution_hash`를 쓴다. `ObservationField.scale`은 어디서도 읽지 않는 서술용 값이다. 실제 척도는 `observation.py`에 하드코딩되어 있다. |
| `runner.py` stable 적분·handover 입력, seed/CLI↔manifest 불일치 | 확인 | 작업 트리 `runner.py:84-102`(HEAD `:81-96`), 그래프 전 노드 broadcast `:41-45` | 적분은 고정 `policy_dt_s`(0.1 s), handover 진행도는 실제 `transition_s`를 써서 시간 기준이 서로 다르다. 계획에 seed·graph_seed·updates 등이 빠져 있다. `ScenarioManifest`의 seed 구간과 4–8 m 높이는 해시되지만 읽히지 않는다(실제 seed는 100000+, 1000001+, 12000+). `TrainingSpec`의 예산은 `algorithm_hash`에 들어가지만 CLI가 덮어쓴다. 초기화·행동·minibatch RNG seed는 기록되지 않는다. |
| `models.py` analytic prior·상황별 action transform·graph Critic | 부분 | prior `models.py:184-185,198-282`, 변환 `:284-324`, critic `:143-156` | **stable profile에서만** 활성이다(runner 기본값 `source`, pipeline 기본값 `stable`). 하강 허가는 prior 안의 연한 latent bias다. 강제 변환은 세 가지다: 수평 0.75 m/s²(시야 경과 8 s 초과 시 0.3) cap, handover climb floor, stale 하강 floor. 따라서 실행 수평 권한은 ActionSpec 2.5의 30 %이며 어떤 hash에도 남지 않는다. graph Critic은 graph 군에만 있다. stable의 `zero_last_layer_initialization` 때문에 초기 encoder 기울기가 정확히 0이다. |
| local direct backend 항상 영상 갱신, y운동/FOV 손실 미반영 | 부분 | `backends.py:106,195`(`updated=True`), `:192`(시야 항 상수), `:98,172-176`(pad x만) | 외생 pad·외란은 x축만 움직이지만 자기 기체는 지령 ay를 적분한다. **추가 결함:** 자세가 `roll=-ax/g, pitch=-ay/g`(`:180`)로, Isaac(`sin(pitch)~+ax`, `sin(roll)~-ay`, r = 0.686 / −0.966)과 축이 바뀌고 pitch 부호도 반대다. 시작 높이는 0.45–1.0 m다. |
| `isaac_adapter.py` capture 보존, KF·visionAge는 receive 기준 | 확인 | `isaac_adapter.py:215` `filter.update(measurement.time_s, …)`, `:223-224` age, `:208-210` capture 저장만 | 패드 위치는 촬영 시점 자기 위치로 기하 정렬하지만 수신 poll 시각에 적용한다. KF 상태 자체는 결정 시각까지 예측된다. stamp만 capture로 바꾸면 `pad estimator time moved backwards` 예외가 난다. 별도 필터 시계가 필요하며, 참고 구현은 `spatial/estimation.py:70-112`의 `_update_capture_aligned`다. live 경로에서는 `from_wire(capture_aligned=True)`가 capture 필드를 강제하므로 adapter의 fallback은 죽은 코드다. |
| view reward world xy/z·양축 90°, reward dt 고정 | 확인 | `backends.py:322-333`, `reward.py:135-142,155` | bearing은 body 원점의 ENU 동·북/|dz|로 계산하며 자세·0.16 m 장착·광축을 무시한다. 시야는 원형 45° 원뿔이다. `reward_v5`는 항상 dt = 0.1을 받는다. 실제 `transition_s`는 0.100–0.116 s(중앙 0.104, 접촉 step 0.040–0.052)이고 GAE 할인은 실제 `transition_s`를 써서 PBRS와 할인 기준이 어긋난다. |
| `graph.py` 정규화 ex/h 차에 고정 −30° slope | 확인 | `graph.py:78,84-88` `cross_x = rx − tan(π/6)·h`(정규화값), `:111` [−1,1] clip | 척도가 다른 정규화값(3 m, 8 m)을 섞는다. 같은 slope가 `relative_vx − 0.577·vz`에도 쓰인다. y에는 적용하지 않아 x/y가 비대칭이다. world ENU x 기준이라 yaw를 무시한다. `drone_cosPhi`는 그래프에 전혀 들어가지 않는다(raw 군만 봄). |
| `models.py` 1층·전체 노드 readout·raw skip 없음, Raw/Typed/Shuffle | 확인 | `models.py:67-105,131,157-158`; `contracts.py:348` | 노드별 trainable local skip이 있다. readout `Linear(112→32)`가 encoder 파라미터의 82 %다. 파라미터 수는 3D source에서 raw 7,015, graph 16,935다. Shuffle 군은 opt-in이며 critic도 함께 shuffle된다. |
| `ppo.py` source/stable 학습 경로 상이 | 부분 | `ppo.py:87-178` 단일 `update` | 하나의 함수 안에 flag 분기 세 개가 있다: KL 강제·revert, LR 적응, `log_std` 상한. 두 profile의 실질적 차이는 대부분 models·runner·spec에 있다. |
| `run_matlab_port_pipeline.py` 기본 stable+local, final 선행 검사 생략 | 확인 | `run_matlab_port_pipeline.py:37-39`(stable, 12 episodes), `:122-131,159-169,186-188`(final 생략) | 검증 경로의 단계도 실제 gate가 아니다. runner `audit`는 검사 없이 PASSED를 반환한다. parity는 2D 전용이며 고정 입력만 본다. `report`는 아무것도 읽지 않는다. |
| `spatial/scenarios.py`·`isaac_sim/pad_motion.py` Isaac CV–CA–CV, local toy와 다름 | 부분 | `pad_motion.py:914-924`가 `spatial/scenarios.py:35-41`을 직접 import | Isaac과 spatial LocalBackend는 **같은 생성기**를 쓴다(seed 4100–4147에서 불일치 0). 다른 것은 direct_policy toy(x만, v0 0.5–2.5)다. 남은 backend 차이는 네 가지다: deck이 heading을 따라 yaw함, Isaac 진입 yaw = heading + U(±5°)(local 0°), deck 상면 0.42 m, 진입 중 deck 정지. |
| `config/spatial-isaac-system-v12.yaml` nadir camera·mount·board/gear | 확인 | 상속 사슬 v12→…→`system.yaml`; `isaac_sim/marker_vision.py:36-40` | `R_CB = diag(1,−1,−1)`: 영상 오른쪽 = 기체 전방, 영상 아래 = 기체 오른쪽. 명목 K는 fx = fy = 320, cx = 320, cy = 240이고 왜곡은 0이다. **해시된 profile에 낡은 `vertical_fov_deg: 70`이 남아 있다**(읽는 코드 없음). 실제 VFOV는 73.74°다. Kit 로그 31회(v12, 10-10~13)에 실제 fx 320, 회전 diag(1,−1,−1), 장착 −0.1598 m가 기록되어 있다. 단 영상면 회전을 코드로 assert하지는 않는다. |
| `docs/matlab_port/MIGRATION_ACCEPTANCE.json` overall FAILED | 확인 | `MIGRATION_ACCEPTANCE.json:3-6` | 고정 대상 `3fc2a9f`는 HEAD보다 4커밋 앞이며 StableTrainingSpec 이전이다. 따라서 그 파일의 full_suite PASSED(1530)는 HEAD에 대해 아무것도 말하지 않는다. |

## 2. 관측·시각·접촉 계약 (가이드 §6–§7)

| 항목 | 판정 | 근거 | 결과 |
| --- | --- | --- | --- |
| nav valid/age 고정값 | 확인 | `isaac_adapter.py:236,249`; `backends.py:138,156` | 모든 backend에서 1.0과 0으로 상수다. `Measurement`에는 항법 유효성 필드가 없다(EKF가 무효면 예외). 학습 중 상수인 채널이 Isaac 전이를 깬 전례가 있다(`detectionConfidence`, spatial-reference/5). |
| 미초기화 표현 | 확인 | `observation.py:117-118` | 앞 7채널이 0이 되고 vision age가 inf→1.0이 된다. 21D 안에서 유효성을 나타내는 신호는 `ugv_visionAge < 1`뿐이다. |
| 높이 채널 | 확인 | `observation.py:104` | 3D h = body 원점 − board 중심이고 deck offset은 없다. 2D도 0.6을 빼고 다시 더하므로 결과적으로 같은 정의다(감사 중 정정). |
| 자세 규약 | 확인 | `isaac_adapter.py:182-185` scipy `as_euler("xyz")` | R_WB = Rz(ψ)Ry(θ)Rx(φ), ENU/FLU, 각속도는 body FLU다. |
| capture 정렬 사슬 | 확인 | `ros2_gateway.py:2086-2176` → `optical_timing.py:12-44` → `core.py:799-831` | 가이드가 적은 사슬 사이에 PnP–IMU 12° gate, 기하 gate, innovation gate가 있다. EKF 이력은 EKF stamp가 아니라 gateway 수신 sim-clock으로 찍힌다. |
| 결정 시점 촬영 간격 | 새 사실, **2차 정정** | v5–v11 비행 trace | 결정마다 기록된 최신 촬영의 간격은 0.092/0.096 s(누락 시 0.188 s, 약 10.6 Hz)다. 결정 시점의 전송 지연 중앙값은 0.096–0.100 s다. **보완(같은 날 Isaac G3):** 관측기에서 직접 재도 결정당 약 1프레임(평균 0.93, 간격 0.092–0.096 s)으로, 결정 시점 값과 같았다. 그러나 정책 입력인 21D의 updated 플래그와 age는 local 20 Hz와 이미 같다(시야 안 updated 0.988 vs 0.977, 갱신 시 age 중앙값 둘 다 0.100 s). local을 10 Hz로 솎으면 같은 추적기가 16 episode 중 11에서 고착되는데 Isaac은 그렇지 않다(1/161, 원인 미규명). 그래서 local은 20 Hz를 유지한다. |
| 접촉 높이 (§7.2) | 부분 | 설정 `core.py:399` 0.12 m; Isaac 다리 없는 첫 접촉 78건 | 실제 body 높이는 0.0636–0.0732 m다. 따라서 접촉 시 카메라는 패드 면 아래 **0.087–0.096 m**로, 가이드의 0.04 m보다 깊다. local은 Isaac보다 약 0.05 m 높은 곳에서 접촉을 선언한다. |
| direct 접촉 판정 | 확인 | `reward.py:176-183`; `backends.py:262-265,317-320` | `norm(roll,pitch) ≤ 5°`, `norm(p,q) ≤ 10°/s`, footprint는 world 축 정렬 \|x\|,\|y\| ≤ 0.5다. 강제 envelope는 25°로 spatial-reference의 21°와 다르다. Isaac deck은 1.5×1.5 m라서 중심에서 0.5–0.75 m 떨어진 deck 위 접촉도 MISSED로 판정된다. |

## 3. 2D 근거 (가이드 §1, G0/G1)

| 항목 | 판정 | 결과 |
| --- | --- | --- |
| v1 증거 묶음이 이 컴퓨터에 있는가 | **반박** | `registration.json`·checkpoint·CSV가 없다. 커밋 `5853781`과 `d438a7d`는 이 컴퓨터의 어느 git 저장소(약 90개)에도 없다. 동기화된 2D 폴더는 git이 아니며 마지막 파일이 2026-10-27 12:27 KST(`relational_residual_v7` 학습 중)다. 가이드 §1.1의 83.7/80.6/76.9/72.6 %를 뒷받침하는 로컬 파일은 없다. |
| `branchRollouts.m` | 확인 | `src/orchestration/+landing2d/+bias_diagnostics/branchRollouts.m`이다. a_x에만 ±0.25 섭동을 주며, 분기 시점은 FOV 경계가 아니라 UGV 가속 시각에 묶여 있다. struct 복사 snapshot은 perception mode에서만 정확하다(`env.sensorStream`은 handle). |
| v1 10D/Φ19/7노드 설계 | 부분 | 10D 필드는 로컬 `vectorSchema.m`에 있다(v1에서 바뀌었는지는 확인 불가). Φ19는 어디에도 없다. 로컬에 있는 7노드 설계 두 가지(2D 보관본 tosm7_v5/rgat_v2, 3D 포트의 direct_ppo_v1 12D)는 **모두 v1이 아니므로 fixture로 대체하면 안 된다.** |

로컬에서 사용 이력이 확인된 seed(가이드 §11.4 seed 감사용): 2D train 291–465, 2D 환경 9001–9300; 3D train 828–832, held-out 4100–4147, teacher stress 5000–5095, Isaac 12000+, matlab-port 3001–3200·100000+·1000001+.

## 4. 실제 실행 환경 (가이드 §4.1)

| 구성 | bootstrap 기본값 | 이 컴퓨터의 실제 checkout |
| --- | --- | --- |
| PX4-Autopilot | v1.14.3 | `1dacb4cd` = v1.14.3. 저장소 patch 2건과 중첩 submodule 수정(jMAVSim, jsbsim_bridge, flightgear_bridge, NuttX apps)이 있다. |
| px4_msgs | release/1.14(이동 branch) | `ffb6e80e` = tag v1.14.0. gitlink와 일치한다. |
| Micro XRCE-DDS Agent | v2.4.2 | `57d08621` = v2.4.2, Fast-DDS CMake patch가 있다. |
| Pegasus | v5.1.0 | `644da37e` = v5.1.0. gitlink와 일치한다. |
| `.gitmodules` | — | **없다.** gitlink 2개에 URL이 없어 `git submodule`로 복원할 수 없다. |
| Isaac Sim | — | `/home/j/isaacsim`, VERSION `5.1.0-rc.19+main.0.47d886f2.local`(describe v5.1.0). 내장 Python은 3.11이다. |
| ROS 2 | Humble | `/opt/ros/humble`만 있다. |
| Python 등 | — | 3.10.12, torch 2.5.1+cu121, numpy 1.26.4, scipy 1.13.1, OpenCV 4.11.0, PyYAML 6.0.3, pytest 8.4.2 |
| OS/GPU | — | Ubuntu 22.04.5, RTX 4060 Laptop, driver 570.172.08 |

## 5. 새 경로 설계에 반영한 결론

1. 새 경로는 `direct_policy`의 `make_contract`·`ContractBundle`·`SOURCE_SHA`를 쓰지 않는다. 21개 채널 이름·척도(`SPATIAL_FIELDS`)와 순수 함수 `digest`, `signed_scale`, `age_scale`만 재사용한다.
2. VFOV는 profile 값을 쓰지 않고 K에서 유도한다. `ξ = 2u/(W−1) − 1`(가이드 식)과 cx = W/2 때문에 광축은 ξ = 1/(W−1)에 놓인다. 이것은 등록된 성질로 기록한다.
3. 접촉 tilt·rate의 정의는 하나만 등록한다: thrust 축 각도 `acos(cosφ cosθ)`와 `‖(p,q)‖`, 한계 5°와 10°/s(spatial `settled` 정의). direct 판정의 `norm(roll,pitch)`과 섞지 않는다. G2에서 확정할 잠정값이다.
4. direct_policy local toy는 smoke 전용이다. 과학적 비교에서는 거부한다. 축이 뒤바뀐 자세 규약 때문에 실행 결과를 Isaac 성능 근거로도 쓸 수 없다.
5. `CausalIsaacObservation`의 receive 시각 기준 KF·visionAge, 상수 nav 채널, local과 Isaac의 추적기 갱신 수 차이(결정당 2 vs 1), 진입 yaw 차이는 G2/G3 미완 항목으로 남기고 성능 캠페인을 막는다.
