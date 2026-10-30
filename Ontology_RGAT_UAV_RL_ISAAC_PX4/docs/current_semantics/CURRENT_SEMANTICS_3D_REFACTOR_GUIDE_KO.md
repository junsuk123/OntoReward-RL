# 현재 관측 기반 이동 플랫폼 착륙 연구의 3차원 리팩토링 가이드

작성일: 2026-10-28 (KST)  
대상 저장소: [junsuk123/OntoReward-RL](https://github.com/junsuk123/OntoReward-RL)  
검토 기준: `cab3dd2123c7eee3968a8c00753f2ba29d55fe71`  
상태: **연구 판단 및 구현 지침. 새 알고리즘 구현·학습·검증·실험 등록은 수행하지 않았다.**

> 저장소 반입 메모 (2026-10-28): 사용자가 전달한 원문을 그대로 보존한다. 이 문서를 기준으로 수행한
> 구현·감사·gate 결과는 같은 폴더의 `G0_G4_STATUS_KO.md`와 `SOURCE_AUDIT_KO.md`에 따로 기록한다.

## 0. 결정

**횡방향 운동–기체 기울기–카메라의 두 영상 축–수평 패드 정렬의 결합을 검증하는 제한된 3차원 확장은 유의미하다.** 이 가이드는 그 목적의 새 연구 경로를 정의한다.

단순히 y 좌표를 추가하거나 현재 R-GAT를 Isaac에 옮기는 것만으로는 연구적 이점이 생기지 않는다. 2차원의 `current_semantics_v1` 개발 실패는 보존하며, 3차원 연구를 그 실험의 확인 단계로 취급하지 않는다. 지금 승인된 작업은 이 가이드 작성이며 아래 본 학습을 실행했다는 뜻이 아니다.

현재 3차원 코드에는 다음 문제가 실제로 존재한다.

- 기존 실행 경로에 교사 정책·행동 복제·적분 상태·해석적 제어 보정이 섞여 있다.
- local direct backend는 항상 영상 갱신을 제공하고 실제 시야 손실을 재현하지 않는다.
- 실제 하향 카메라와 그래프의 고정 전방 접근선이 다르다.
- 촬영·수신·결정 시각, 시야 보상, 실제 transition 시간의 의미가 일치하지 않는다.
- 현재 2차원의 네 비교군·공통 raw Critic·현재 입력만 사용하는 정책과 동일한 비교가 아니다.

따라서 **관측·좌표·실행·실험 계약을 먼저 정리한 뒤 한 개의 후보를 제한된 개발 실험으로 판정**한다. 선행 단계가 실패하면 본 학습을 열지 않는다.

---

## 1. 근거와 출처의 상태

| 근거 | 확인한 식별자/상태 | 해석 범위 |
| --- | --- | --- |
| 2D GitHub main | `d438a7d4096d5b186c731c894306754aa5ab628d` | C3(v8) 기록까지 확인. 원격에 current_semantics_v1 없음 |
| 별도 2D 전달 구현 | 로컬 commit `5853781a21e82d6e3ada1f36aa16690ec3a17ad9` | current_semantics_v1 설계·구현. 원격 commit이 아님 |
| 사용자가 실행한 v1 결과 | 2026-10-27 23:19 등록 → 10-28 01:15 종료, 20모델/16,000평가 | 사용자 제공 실행 보고. 이 가이드 작성 환경에서는 원본 결과·체크포인트를 재검산하지 않음 |
| 3D GitHub main | `cab3dd2123c7eee3968a8c00753f2ba29d55fe71` | 2026-10-10 커밋. 파일을 해당 SHA로 고정해 감사 |
| 3D migration acceptance | 2026-10-09 기록, overall FAILED | 예전 source/target pin의 검사 기록. HEAD 전체의 실행 인증이 아님 |
| Notion 13번 | 현재 관측 기반 재구성 문서 | 구현 전달 당시 기록. 이번 개발 실패 보고가 더 최신 |
| 이전 3D 연구 설계서 | 26/44-field, 9×12 graph, pretraining/보조 경로 등의 이력 | 현재 방법의 명세로 재사용하지 않음 |

### 1.1 이번 2D 결과에서 이어받는 사실

사용자 제공 평균 OOD 성공률은 Raw PPO 83.7%, 동일 특징 MLP 80.6%, Untyped 76.9%, R-GAT 72.6%다. R-GAT의 motion/combined는 66.0/38.2%, PPO는 81.4/64.6%다. 모든 학습 시드는 ID 학습 기준을 통과했으나 개발 비교와 위험 증가 기준은 실패했다.

- 실패를 설명하기 위해 “2D가 너무 쉬웠다”거나 “실행 불가능한 환경이었다”고 단정하지 않는다.
- FOV_LOSS는 timeout 종료 분류이며 최초 원인 검출기가 아니다.
- 큰 행동 교호작용은 좋은 제어의 증거가 아니다.
- Typed와 Untyped의 비슷한 교호작용은 관계 유형 없이도 해당 반응이 나타남을 뜻한다.
- 5개 학습 시드의 넓은 구간을 이유로 열등성·동등성을 확정하지 않는다.
- 새 결과로 기존 실패를 소급해 성공 처리하지 않는다.

### 1.2 착수 전에 확보할 동결 묶음

실제 사용자가 실행한 v1의 `registration.json`, resolved 설정, source manifest, 선택 checkpoint hash, validation history, gate 결과, episode CSV와 실행 환경을 연결한다. 로컬 전달 commit과 실제 실행판의 일치를 가정하지 않는다. 결과를 복사할 때 원본 manifest의 상대경로·해시·시각을 보존한다.

---

## 2. 연구 질문과 주장 범위

### 연구 질문 (Research Question)

**현재 상대 운동과 기체 자세로 계산한 두 축의 시야·정렬 조건을 온톨로지 관계로 구성하면, 동일한 현재 특징을 사용하는 MLP 및 관계 유형을 통합한 GNN보다 미학습 시나리오 조합에서 안전한 착륙 완료율을 높이는가?**

### 방법 (Methods)

현재 센서·추정 출력 → 공통 원 관측 → 결정론적 물리 특징 → 온톨로지 그래프 → R-GAT → 원 관측과 결합한 Actor → 3축 순가속도. Critic은 모든 군에서 같은 원 관측을 사용한다. R-GAT는 PPO Actor 목적의 기울기로 함께 학습한다.

### 기존 관찰 (Main findings)

현재 2D v1은 개발 기준을 통과하지 못했다. 3D 이점은 아직 측정하지 않았다. 3D 저장소에서 확인한 것은 구현 불일치와 새로 검증 가능한 물리 결합이다.

### 가정 (Assumptions)

- 수평 패드, 일정한 패드 높이, episode 안에서 고정된 플랫폼 진행 방향.
- 현재의 자체 자세·각속도와 플랫폼 위치·수평속도 추정이 사용 가능하다.
- 카메라 내·외부 보정과 기체/패드 치수는 공개된 고정 설정이다.
- yaw는 기존 PX4 경로에서 초기 heading을 유지한다. yaw를 정책 행동으로 추가하지 않는다.
- 기존 항법·추정기 내부 처리는 유지한다. 정책에 별도 시간 정보를 누적하지 않는다.

### 한계 (Limitations)

- 능동 yaw 제어, 회전·경사·상하 운동 패드, 패드 상대 자세 추정, 실기체 전이는 이번 범위 밖이다.
- 일반 R-GAT에는 물리 회전에 대한 등변성 보장이 없다. [R4]
- 현재 시야 여유는 표적 중심의 기하 지표이며 실제 마커 검출 확률이나 미래 가시성 보장이 아니다.
- 현재 입력만으로 숨은 가속도·관측 소실 원인·과거 행동 순서를 복원할 수 없다.
- 같은 입력의 결정론적 변환은 새로운 정보를 만드는 것이 아니다. 표현의 학습 효율과 일반화가 검증 대상이다.

### 기여 후보 (Contributions)

1. 현재 위치·속도·자세·시야·접촉 조건의 의존 관계를 정책 입력 구성에 사용하는 기법.
2. 같은 수치 특징을 공유하는 대조군으로 특징 가공과 그래프·관계 유형의 추가 효과를 구분하는 검증.
3. 2D 제한 조건–3D 축소 물리–Isaac/PX4의 단계별 구현 정합 및 성능 검증.

### 연구적 의미 (Implications)

확인 전에는 “기법과 검증 체계를 제안했다”고 쓴다. 성능 우위가 나오더라도 예산을 맞추면서 head 폭이 달라지는 한계와 평가 분포를 함께 밝힌다. “관계 중요도 학습”, “의미 이해”, “안전 보장”, “최적 관계 발견”은 직접 학습 목표 또는 입증된 결과가 아니므로 사용하지 않는다.

### 2.1 3D가 추가하는 것과 추가하지 않는 것

| 항목 | 판단 |
| --- | --- |
| 측방 이동과 roll, 종방향 이동과 pitch가 두 영상 축에 함께 미치는 영향 | 이번 3D의 핵심 검증 대상 |
| 가로·세로 정렬, 수직 하강, 접촉 자세를 함께 고려하는 제어 | 이번 범위 |
| pitch와 FOV의 결합 | 이미 2D에도 존재. 3D에서 처음 생겼다고 표현하지 않음 |
| episode 내부 선회·회전 패드 | 현재 기본 시나리오가 아님 |
| 3D이므로 GNN이 반드시 우수함 | 근거 없음 |
| 보지 못한 시나리오 인자의 조합 | 새 평가에서 사전 정의할 수 있음 |
| 모든 미학습 상태·모든 3D 회전에 대한 일반화 | 이번 실험으로 주장할 수 없음 |

PAMPC는 기동에 필요한 기체 회전과 가시성 목표가 충돌함을 다룬다. 이는 물리적 연구 문제의 근거다. 반대로 Goldschmid와 Ahmad는 축 분리 가정하 두 1D RL 제어기로 종·횡 이동을 처리했다. 따라서 차원이 아니라 실제 결합의 존재를 검증해야 한다. [R1][R2]

---

## 3. 고정할 방법론 제약

### 유지

- 기존 센서의 현재 추정 상태와 공개된 고정 상수.
- 온톨로지, feedforward R-GAT, PPO.
- 원 관측 경로 보존, 모든 비교군에 같은 정보 제공.
- 공통 raw Critic, 같은 보상·물리·센서·종료·PPO 계약.
- 현재 속도·각속도로 계산한 순간 운동학적 변화율.
- 기존 KF/EKF 내부 상태, 기존 촬영 시각 정합 처리, 표준 PPO rollout.
- 고정 한계와 기존 PX4 내부 자세 안정화.

### 새 정책 경로에서 제외

- GRU/LSTM, 관측·프레임 stack, 이전 행동, 시간 간선.
- 오차 적분, tracking_bias, handover_progress, 임무 단계 기억, 조건 지속시간.
- teacher/BC/DAgger, 별도 사전학습, auxiliary prediction head/loss.
- 해석적 추종 명령에 학습 보정을 더하는 정책, 학습 행동의 상황별 하강 허가·재포착 보정.
- simulator truth, 미래 플랫폼 운동 일정, 접촉 결과, 평가 라벨의 입력 사용.
- R-GAT만 받는 좌표 변환·시야 특징·추정 품질.

기존 PX4의 내부 제어와 항법 추정기의 상태를 제거하라는 뜻이 아니다. 새로 학습하는 정책이 현재 공통 관측 이외의 실행 이력을 읽지 않게 한다. 기존 센서 시각 정합용 이력을 정책 입력으로 노출하지 않는다.

---

## 4. 현 저장소에서 확인한 변경 지점

아래 `P/`는 `Ontology_RGAT_UAV_RL_ISAAC_PX4/`를 뜻한다.

| 우선순위 | 기존 파일 | 확인한 현상 | 리팩토링 지침 |
| --- | --- | --- | --- |
| P0 | 루트 `run.sh` | bare 실행이 minimal teacher→BC→PPO→Isaac 경로 | 새 명시 경로만 추가. 검토 목적으로 bare 실행 금지. 과거 route 유지 |
| P0 | 루트 `README.md`·`AGENTS.md` | 시대가 다른 명세 혼재, README의 bare 도움말 설명과 코드 불일치 | 활성 연구 명세와 역사적 route 표를 상단에 정리 |
| P0 | `P/python/ontology_rgat/direct_policy/contracts.py` | 옛 MATLAB `6082258`, source21D/stable24D 혼재 | 새 ID·실행 설정·현재 2D 출처와 변경 범위를 별도 등록 |
| P0 | `.../direct_policy/runner.py` | stable 적분과 handover 입력, 실행 seed/CLI가 manifest와 불완전 대응 | 새 runner에서 상태 누적 제거, 실제 resolved 설정과 모든 seed 등록 |
| P0 | `.../direct_policy/models.py` | analytic prior와 상황별 action transform, graph Critic | 새 모델은 직접 Actor 출력과 공통 raw Critic. 과거 클래스 동작 보존 |
| P0 | `.../direct_policy/backends.py` | local toy가 항상 영상 갱신, y운동/FOV 손실 미반영 | 해당 backend는 smoke 전용. 과학적 성능 비교에서 거부 |
| P0 | `.../direct_policy/isaac_adapter.py` | capture를 보존하지만 KF update/visionAge는 receive 기준 | 기존 촬영 시각 정합 경로와 필터 시각 의미를 일치시킴 |
| P0 | `.../direct_policy/backends.py` | view reward가 world xy/z·양축90°, reward dt는 고정값 | 실제 보정·센서 경로·transition 시간에 맞는 공통 3D 보상 명세 |
| P1 | `.../direct_policy/graph.py` | 정규화 ex/h의 차에 고정 −30° slope 적용 | 물리 단위 복원 후 실제 카메라 좌표에서 계산 |
| P1 | `.../direct_policy/models.py` | 1층·전체 노드 readout·raw skip 없음, Raw/Typed/Shuffle | 새 네 비교군, 공유 2홉, mission readout, raw bypass |
| P1 | `.../direct_policy/ppo.py` | source/stable 학습 경로가 다름 | v1 목적·GAE·latent action·optimizer fixture로 이식 범위 검증 |
| P1 | `P/python/run_matlab_port_pipeline.py` | 기본 stable+local 학습, final은 선행 검사 생략 | 새 gate 기반 실행기를 분리. 옛 final을 새 연구 진입점으로 쓰지 않음 |
| P1 | `P/python/ontology_rgat/spatial/scenarios.py`·`P/isaac_sim/pad_motion.py` | 실제 Isaac은 작은 heading의 CV–CA–CV, local toy와 다름 | 하나의 외생 시나리오 manifest를 두 backend에 적용 |
| P1 | `P/config/spatial-isaac-system-v12.yaml` 및 상속 설정 | nadir camera·mount·실제 board/gear 정의 | resolved 배포 설정과 camera calibration hash 고정 |
| P2 | `P/docs/matlab_port/MIGRATION_ACCEPTANCE.json` | 2026-10-09 overall FAILED | 보존. 새 범위의 acceptance 파일 작성 |

`.../`로 줄인 경로의 공통 접두사는 `P/python/ontology_rgat/`다. 위 파일들은 모두 검토한 HEAD에 존재한다. 아래 절의 `current_semantics/` 파일과 CLI는 **새로 구현할 항목**이다.

### 4.1 실제 현재 실행 환경의 확인 상태

bootstrap 기본값은 PX4 v1.14.3, px4_msgs release/1.14, Micro XRCE-DDS Agent v2.4.2, ROS 2 Humble, Pegasus v5.1.0이다. 환경변수와 기존 checkout으로 달라질 수 있으므로 실제 설치 버전으로 간주하지 않는다.

원격 gitlink는 Pegasus `644da37e9d5268e5f9a34e78bdcfd57a8bab82b4`, px4_msgs `ffb6e80e1c17e5714395611a020c282a87af8fa4`다. 실제 Isaac build, PX4 실행 SHA/patch, ROS 패키지, GPU/driver와 Python·PyTorch 버전은 **실행 컴퓨터에서 확인할 미완 항목**이다. 최신 패키지로 자동 업그레이드하지 않는다.

---

## 5. 새 코드 구성

과거 `minimal/`, `spatial/`, `landing/`, `direct_policy/`의 재현 경로를 보존한다. 새 연구 경로는 다음처럼 분리한다. 기존 센서·배포·구동기 코드는 검증된 경계에서 재사용하며 복사본을 여러 개 만들지 않는다.

```text
P/python/ontology_rgat/current_semantics/
    contracts.py          # schema·단위·카메라·행동·과제·실험 식별자
    observation.py        # source21D의 현재 관측 계약
    geometry.py           # 물리 역변환·카메라 투영·순간 변화율
    features.py           # 공통 phi44·node 배치
    graph.py              # 8 nodes / 22 edges 후보
    models.py             # 네 Actor / common raw Critic
    backend.py            # 관측·명령·평가 경계; 기존 자원 재사용
    experiment.py         # resolved config·seed·registration·gate
    evaluate.py           # paired episode 결과·seed 집계·진단
P/python/run_current_semantics_3d.py
P/config/current_semantics/spatial_view_contact_v1.yaml
P/docs/current_semantics/
P/tests/current_semantics/
```

제안 식별자: `spatial_current_semantics_v1`. 옛 `current_semantics_v1` checkpoint나 `matlab-direct-policy/1`, `stable-v14`의 hash/eligibility를 공유하지 않는다.

실행 entrypoint에는 별도 package 경로와 명시된 profile을 요구한다. 경로가 없을 때 local toy 또는 stable 경로로 자동 대체하지 않는다.

---

## 6. 현재 관측 계약: 21D 유지, 24D 이력 입력 제거

| 성분 | 차원 | 물리 의미/출처 |
| --- | ---: | --- |
| `e_x,e_y,h` | 3 | 추정 pad−own 수평 위치, own−pad 높이 |
| `v_rx,v_ry` | 2 | 추정 pad−own 수평 속도 |
| `v_Gx,v_Gy` | 2 | 현재 플랫폼 수평 속도 추정 |
| `v_Dz` | 1 | 현재 자체 수직 속도 |
| roll/pitch/yaw sin/cos | 6 | 현재 자체 항법 자세 |
| body p/q/r | 3 | 현재 자체 각속도, frame 명시 |
| vision update/age | 2 | 이번 결정의 측정 갱신·사용 측정의 촬영 경과시간 |
| navigation valid/age | 2 | 실제 항법 메시지의 가용성·시각 정보 |
| 합계 | **21** | 모든 비교군 공통 |

Source adapter의 navValid=True/navAge=0 고정값은 실제 품질 관측으로 주장할 수 없다. 실제 항법 메시지의 유효성·시각을 연결할 수 있는지 먼저 확인한다. 불가능하면 그 상태를 명세에 표시하고 정책 품질 추론의 근거로 쓰지 않는다. 필드를 바꾸려면 네 군의 schema를 함께 바꾸고 학습 전에 다시 고정한다.

2D는 현재 10D이며 옛 direct 포트의 planar12D와 다르다. 3D를 반드시 10D로 줄이지 않는다. MATLAB fixture용 투영은 공통 10개 성분을 **이름으로** 선택하며 추가 nav 두 값을 슬쩍 포함하지 않는다.

### 6.1 정규화와 가용성

- 부호 있는 물리 특징에는 공통 함수 `B(x;s)=x/(|x|+s)`를 사용하고 단위·scale을 registry에 명시한다. 원 관측의 sin/cos·flag·age는 각 채널의 별도 규약을 유지한다.
- `B(ex;3)−m B(h;8)`를 물리 접근 오차로 해석하지 않는다.
- `su/(1−|u|)` 역변환과 `|u|<1` 조건은 `signed_scale`을 적용한 위치·속도·각속도 채널에만 사용한다. sin/cos의 ±1과 boolean의 1은 정상 값이며, `age_scale=1`은 미초기화/무한 경과시간 센티널로 별도 처리한다. 각 역변환 규칙을 채널 registry에 명시한다.
- 미초기화·카메라 뒤쪽 상태를 수치 0만으로 정상 정렬과 혼동하지 않게 품질/기하 validity를 같은 특징 벡터에 포함한다.
- 현재 21D로 얻을 수 없는 pad 자세·각속도·공분산을 simulator에서 채우지 않는다.
- 새로 저장하는 정책 관측 이력은 없다. 특징 함수는 현재 관측과 고정 설정의 순수 함수다.

### 6.2 촬영·수신·결정 시각

세 시각 `t_capture≤t_receive≤t_decision`을 구분한다. vision age는 마지막으로 **수용한** 측정의 capture로부터 decision까지의 경과시간이다. 새 프레임 여부와 매 제어 tick의 재사용을 구분한다.

현재 실제 경로의 `ros2_gateway._on_pad_pose → own_optical_history.at(capture) → optical_board_pose_to_enu → Measurement.from_wire(capture_aligned=True)`를 재사용한다. 새 정책 이력을 추가하는 작업이 아니다.

`CausalIsaacObservation.ingest`가 측정을 어느 시각의 위치로 받아 필터를 갱신하는지 명확히 한다. receive stamp를 capture stamp로 바꾸는 한 줄 수정만으로 지연 보상이 끝났다고 판단하지 않는다. 기존 추정기의 시각 정합으로 처리할 수 없는 부분은 미완으로 남기고 성능 캠페인을 차단한다.

---

## 7. 3차원 기하: 새 정보 없이 계산할 수 있는 범위

세계 W는 ENU, 기체 B는 FLU, 영상 C는 optical frame(+X right,+Y down,+Z forward)로 명시한다. `R_BW`는 W 성분을 B 성분으로 변환한다. `R_CB`는 B 성분을 C 성분으로 변환한다. Isaac USD와 ROS camera, PX4 NED/FRD의 변환은 입출력 경계에서 한 번만 한다. [R5][R6]

수평·정고도 패드 범위에서는:

$$
r_W=[e_x,e_y,-h]^T,\qquad
v_{\mathrm{rel},W}=[v_{rx},v_{ry},-v_{Dz}]^T.
$$

기체 원점에서 카메라 중심까지의 고정 위치를 `t_BC`라 하면:

$$
b=R_{BW}r_W,\qquad
p_C=R_{CB}(b-t_{BC}),
$$
$$
\dot p_C=
R_{CB}\left(R_{BW}v_{\mathrm{rel},W}-\omega_B\times b\right).
$$

회전항에서 `b` 대신 보정되지 않은 `p_C`를 사용하면 장착 위치의 효과를 누락할 수 있다. 이 변화율은 현재 추정 상태의 순간 운동학이다. 추정기 측정 갱신의 jump까지 설명하는 실제 시계열 미분이나 미래 예측값이 아니다.

검토한 배포의 nominal camera는 아래쪽 0.16 m, nadir 장착이다. 이때 optical 변환은 `R_CB=diag(1,−1,−1)`다. 실제 구현에서는 이 값을 별도로 하드코딩하지 않고 resolved 보정에서 읽는다.

### 7.1 영상 좌표·여유·변화율

`p_C=[X,Y,Z]`이고 전방·정상 투영 조건을 만족할 때:

$$
u=f_xX/Z+c_x,\quad v=f_yY/Z+c_y,
$$
$$
\dot u=f_x(\dot XZ-X\dot Z)/Z^2,\quad
\dot v=f_y(\dot YZ-Y\dot Z)/Z^2.
$$

영상 좌표를 $\xi=2u/(W-1)-1,\ \eta=2v/(H-1)-1$로 정규화한다. 네 경계 여유는 $[1+\xi,1-\xi,1+\eta,1-\eta]$다. 음수는 해당 경계를 벗어났음을 뜻한다.

- raw camera 보정은 640×480, HFOV90°, VFOV 약73.74°인 현재 profile을 출발점으로 한다.
- 실제 K·왜곡 모델·유효 영상 경계를 사용한다. 두 축 모두90°로 놓지 않는다.
- `Z≤z_near`에서는 front-valid=0, 투영 관련 수치는 등록한 중립값으로 둔다. 분모에 epsilon만 넣어 정상 전방으로 처리하지 않는다.
- center margin은 전체 ArUco quad 가시성·최소 픽셀 크기·조명·검출 성공을 보장하지 않는다.
- pad 자세를 입력받지 않으므로 정책 특징에서 전체 board corners의 정확한 미래 투영을 만들지 않는다. 실제 검출 여부는 센서에서 받고, 전체 tag의 기하 판정은 평가용 truth 경로에서 별도로 측정한다.

### 7.2 종말 구간의 관측 소실

검토한 gear 없는 설정은 body 접촉 높이 약0.12 m, camera mount −0.16 m다. nominal 접촉 시 카메라는 패드 면보다 약0.04 m 아래에 올 수 있다. 이 기하는 실제 profile에서 다시 확인한다.

그러므로 “시야를 끝까지 유지” 자체를 성공 목표로 삼지 않는다. **접근 중 예방 가능한 시야 이탈**과 **접촉 직전 기하상 불가피한 tag 소실**을 구분한다. 후자를 자동 상승·재포착 규칙으로 바꾸거나 성공 기준을 완화하지 않는다.

---

## 8. 한 개의 제안 그래프 후보

다음은 구현할 후보의 정확한 초안이다. 아직 성능이 검증된 그래프가 아니다. 특징·보정 감사 후 학습 전에 최종 schema와 숫자를 동결한다. 결과를 보고 노드·간선을 계속 추가하지 않는다.

### 8.1 노드와 공통 특징

현재 배포의 nadir·수평 패드에서는 접근 정렬을 `e_a=[ex,ey]`, `v_a=[vrx,vry]`로 둔다. $\eta_a=e_a^Tv_a$는 수평 위치 오차 제곱의 순간 변화 방향을 나타낸다. 앞선 2D 전방 카메라의 `ex−mh`를 이 배포에 그대로 복사하지 않는다.

| 기호/노드 | 변화하는 수치 특징 | 개수 |
| --- | --- | ---: |
| Q ObservationQuality | raw vision update/age, nav valid/age | 4 |
| G RelativeGeometry | raw ex/ey/height | 3 |
| M MotionState | raw vrx/vry, pad vx/vy, own vz + 현재 차로 계산한 own vx/vy | 7 |
| A AttitudeState | raw roll/pitch/yaw sin/cos 및 p/q/r | 9 |
| P ApproachSituation | B(eax),B(eay),B(vax),B(vay),B(ea·va) | 5 |
| O ViewGeometry | B(ξ),B(η),B(ξdot),B(ηdot), 네 경계 여유의 B값, front-valid | 9 |
| C ContactCondition | ex/ey,vrx/vry,vz의 접촉 척도 비율 5개 + 현재 tilt/rate 기준 비율 2개 | 7 |
| L LandingContext | 초기 수치 특징 없음 | 0 |
| 합계 | **공통 flatPhi** | **44** |

`B`의 scale과 유효한 단위는 별도 registry에 등록한다. 시간 변화율의 scale을 위치 scale과 혼용하지 않는다. 현재 tilt/rate 특징은 **공통 접촉 평가가 실제 사용하는 정의**와 일치시킨다. 기존 direct 판정의 `norm([roll,pitch])`·`norm([p,q])`와 기하학적 tilt 각도·전체 각속도 norm은 서로 다르다. 임의로 섞지 않는다.

C는 현재 추정 기반 조건 특징이다. 물리 접촉 성공 승인, 회전한 패드 footprint 전체의 안전성 확인 또는 위험 확률이 아니다. 실제 pad yaw·collider·접촉점에 따른 판정은 격리된 평가기에서 정확히 수행한다. 향후 그 조건을 정책 입력에 직접 표현하려면 pad 자세 관측이 필요하므로 이번 범위를 넘는다.

- node 입력은 **8노드 × (수치9+고정 node ID1)=8×10**이다. 사용하지 않는 채널은 0 padding.
- `flatPhi44`에는 모든 변화하는 값과 validity가 들어간다. padding·상수 ID는 제외한다.
- raw21은 Q/G/M/A의 첫 raw 슬롯에서 이름별로 그대로 복원된다.
- MLP가 그래프와 같은 파생 특징을 모두 받는다.

### 8.2 간선과 홉 수

| 관계 유형 | 간선 | 개수 |
| --- | --- | ---: |
| qualifiesEstimate | Q→P, Q→O, Q→C | 3 |
| informsApproach | G→P, M→P | 2 |
| affectsVisibility | G→O, M→O, A→O | 3 |
| conditionsContact | G→C, M→C, A→C | 3 |
| informsLanding | P→L, O→L, C→L | 3 |
| self | 각 노드→자기 자신 | 8 |
| 합계 | 의미14 + self8 | **22** |

모든 primitive node가 두 홉 안에 L에 도달한다. 공유 가중치의 R-GAT 메시지 전달 2회, node embedding8, L readout8을 초기 후보로 유지한다. 별도 O→C 간선을 추가하지 않는다. 추가하면 경로 길이와 의미가 달라지므로 2홉 도달을 다시 검증해야 한다.

관계별 고정 곱셈 gate·행동 차단은 없다. 관계 이름은 자연어 명령으로 실행되지 않으며 관계별 가중치 공유와 메시지 경로를 구분한다.

### 8.3 PPO 연결과 직접 학습 목표

$$
\phi_t=f(o_t),\quad
g_t=\mathrm{RGAT}_{\theta_G}(X_t,E),\quad
\mu_t=\mathrm{MLP}_{\theta_A}([o_t;g_t]),
$$
$$
z_t\sim\mathcal N(\mu_t,\operatorname{diag}(\sigma^2)),\qquad
a_t=a_{\max}\odot\tanh z_t,\qquad
V_t=\mathrm{MLP}_{\theta_V}(o_t).
$$

Actor loss는 PPO clipped surrogate와 공통 entropy 항으로 구성한다. 현재의 전역 logσ 및 latent Gaussian entropy 정의에서 Actor 평균과 R-GAT의 $\theta_A,\theta_G$는 PPO surrogate의 기울기로 갱신되고, logσ는 surrogate와 entropy의 기울기를 받는다. Critic loss는 별도 공통 raw Critic을 갱신한다. 별도 관계 정답·관계 중요도 loss·예측 loss는 없다. 두 홉은 한 결정 시각의 계산이며 시간 관측을 저장하지 않는다.

PPO가 기록하는 latent action·log probability와 환경에 전달한 가속도를 모두 추적한다. 상황별 prior/cap/climb를 모델 내부에서 숨겨 적용하지 않는다.

### 8.4 네 비교군과 용량 초안

공통 Critic은 21→55→1, logσ는3개다. graph encoder는 위 10채널 입력·8차원 embedding·공유2홉, Typed 관계6종(self포함), Untyped 관계2종이다.

| 군 | Actor | 인코더 | Actor head | Critic+logσ | 합계 |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw PPO | 21→110→3 | 0 | 2,753 | 1,269 | **4,022** |
| Semantic MLP PPO | 44→57→3 | 0 | 2,739 | 1,269 | **4,008** |
| Untyped GNN PPO | [21+8]→74→3 | 306 | 2,445 | 1,269 | **4,020** |
| Ontology R-GAT PPO | [21+8]→66→3 | 566 | 2,181 | 1,269 | **4,016** |

이는 **설계 산식**이다. 구현 후 실제 trainable parameter 수와 optimizer 포함 여부를 검사해야 한다. 최대 차이는14개, 최소값 대비 약0.35%다. 사용하지 않는 parameter로 숫자를 맞추지 않는다.

이 비교는 같은 예산의 설계 비교다. Untyped/Typed의 head 폭이 달라 관계 유형만의 순수한 인과 효과는 분리되지 않는다. Raw 우회도 좁은 head가 Raw110의 모든 함수를 표현한다고 보장하지 않는다.

**관계 이름의 의미적 배치가 임의 배치보다 우수하다는 강한 주장이 필요하면, 동일 architecture/parameter/head의 relation-shuffle 대조를 별도 사전 계획으로 추가해야 한다.** 이번 네 군만으로 그 주장을 하지 않는다. 결과를 본 뒤 유리한 shuffle만 고르지 않는다.

---

## 9. 물리·센서·보상·행동의 공통 경계

### 9.1 backend

- 기존 `LocalReferenceBackend`는 연결 smoke와 배열 shape 확인에만 쓴다.
- 성능 학습용 3D 축소 backend에는 측방 운동, roll/pitch와 추력 응답, 실제 카메라 보정, 실제 크기의 marker/board, 검출 누락, 공통 추정 경로가 필요하다.
- 이미 있는 `spatial/`의 동역학·광학·capture alignment를 재사용할 수 있지만, supervisor·integral·임무 latch를 새 정책 경로로 함께 가져오지 않는다.
- Isaac backend와 하나의 scenario manifest, camera/asset hash, 시간 정의를 공유한다.
- 축소 환경과 Isaac은 다른 실행 backend임을 기록한다. 하나의 성공률로 합치지 않는다.

### 9.2 action

순가속도는 world ENU `[ax,ay,az]`다. PX4 경계에서 NED로 한 번만 변환한다. body FLU/FRD 각속도·자세도 별도로 검증한다.

기존 공통 물리 한계와 PX4 내부 안정화는 유지한다. 등록된 상수 한계와 물리 actuator 응답을 상황별 착륙 감독기의 명령 수정과 구분한다. 실제 배포가 한계를 적용하면 어떤 단계에서 얼마나 바뀌었는지 기록한다.

단계별 필드는 `latent_z → policy_accel → adapter_limited_accel → PX4_setpoint → measured_accel`이다. 마지막 두 값은 같다고 가정하지 않는다. 중간 보정이 있다면 변경 사유를 기록하며 “direct Actor 성능”으로 감추지 않는다.

정책 시작 전 entry/arming/offboard 준비는 임무 평가와 분리한다. 정책 시작 이후의 실패를 setup 실패로 바꾸거나 좋은 초기 상태가 나올 때까지 다시 뽑지 않는다. 종료 후 LAND/cleanup으로 내려앉은 것을 정책 착륙 성공으로 세지 않는다.

### 9.3 reward와 termination

모든 비교군에서 같은 함수를 사용한다. 실제 3D 카메라와 맞지 않는 옛 시야 보상을 고치는 것은 **과제 계약 변경**으로 기록하고, 네 군 학습 전에 한 번에 고정한다.

- 수평 오차와 접촉 속도는 두 축을 동일 단위로 확장한다.
- 실제 camera frame·HFOV/VFOV·센서 갱신에 맞는 시야 항을 정의한다.
- reward 적분, discount, GAE는 실제 transition dt를 일관되게 사용한다.
- terminal potential은 실제 terminal에서 0. 임의 rollout truncation을 물리 terminal로 처리하지 않는다.
- 접촉 판정은 접촉 전 위치·속도·자세·각속도와 실제 pad/collider에 기반한다.
- FOV 특징값이나 attention을 성공·실패 정답으로 재사용하지 않는다.
- 새 의미 보상, 추가 관계 정답, 실패를 지우는 새 종료 규칙을 넣지 않는다.

---

## 10. 학습 전에 통과할 단계

### G0 — 근거 동결과 실행 경계

**산출물:** 2D 실행 evidence manifest, 3D baseline SHA, 새 schema/실행경로 설계, 상태표.

**통과:** 원본 결과 보존, 현재/역사적 경로 구분, 실제 실행 설정 추적 가능. v1 확인은 닫힌 상태로 유지.

### G1 — 동결 2D 실패의 제한 진단

새 학습 없이 기존 개발 체크포인트를 사용한다. 첫 기하학적 FOV 경계 접근 직전의 같은 환경 snapshot에서 다음 네 행동 조합을 비교한다.

| 분기 | 수평 명령 | 수직 명령 |
| --- | --- | --- |
| 원본 | R-GAT | R-GAT |
| 수직 치환 | R-GAT | PPO |
| 수평 치환 | PPO | R-GAT |
| 두 축 치환 | PPO | PPO |

사례 선정 규칙, 분기 길이, 이후 continuation 정책을 먼저 고정한다. 원본 분기 재현이 우선이다. 모든 시드의 공통 규칙으로 사례를 모으고 유리한 성공 장면만 뽑지 않는다.

기존 `landing2d.bias_diagnostics/branchRollouts.m`은 ax perturbation용이므로 위 2축 치환이 이미 구현됐다고 가정하지 않는다. 추정기·난수·물리 상태를 함께 복원한다.

**판정:** 실제 오차·정규화·카메라 부호 오류가 있으면 먼저 그 오류를 고친다. 결과가 불명확하면 미확정으로 기록한다. 3D 필요성을 만들기 위해 하강 지연 원인을 강제로 채택하지 않는다.

### G2 — 수치·좌표·시간·정보 정합

검사할 항목:

- 실제 v1 10D/Φ19/7노드/2홉의 고정 입력·가중치·Actor·Critic·PPO loss/gradient/optimizer-step fixture.
- 새 공간21D→현재 특징의 물리 단위, 부호, raw21 복원.
- 추정 pose와 camera calibration의 축·quaternion 순서·라디안/도 단위.
- camera mount lever arm, 두 영상축, 앞/뒤/경계에서 투영.
- 동일 현재 관측은 호출 순서·과거 입력에 무관하게 같은 출력.
- source/fixture/model/operator hash가 checkpoint에 묶임.
- 두 비교 그래프의 모든 primitive→L 경로, 실제 encoder gradient.
- captured frame 재사용·지연·순서 역전의 timestamp 의미.
- latent action/PPO 평가/실제 적용 경로 일치.

평면 제한 시험은 횡운동·roll/yaw 변화가 없는 **같은 보정·같은 물리 법칙의 fixture**에서 x-z 결과가 환원되는지 확인한다. 실제 Isaac의 nadir camera와 2D의 전방 camera가 다른데 두 궤적이 같아야 한다고 요구하지 않는다. geometry를 같은 값으로 맞춘 fixture와 실제 배포 차이를 별도 보고한다.

**통과:** 구현 오류를 허용 오차 내에서 배제. 허용 오차는 float64 fixture와 실제 float32 runtime을 구분해 정한다. 단위 검사를 성능 검증으로 쓰지 않는다.

### G3 — 3D 물리 결합과 관측의 실제 작동

본 학습 전 고정된 짧은 물리 사례에서 다음을 확인한다.

- x만, y만, xy동시 명령이 실제 roll/pitch·두 영상축에 반영됨.
- 영상 경계 밖에서는 실제 검출 갱신이 달라지고 추정 상태/age에 연결됨.
- local과 Isaac에서 같은 배포 보정을 사용함.
- 태그 중심과 전체 tag quad 가시성의 차이, 낮은 고도의 사각 구간을 기록.
- 같은 가속도 크기에서 방향만 바꾼 사례를 비교. 두 축에 각 최대값을 넣어 √2배 난도를 만든 것을 방향 효과로 해석하지 않음.
- nav/vision 품질 채널이 실제 메시지 상태와 일치.
- 종료와 cleanup의 구분.

예비 사례는 구현 검사용으로 고정하며 main 개발·확인 시드와 분리한다. 관측·동역학이 사실상 평면이거나 항상 visible이면 본 연구용 backend로 채택하지 않는다.

### G4 — 공통 PPO 학습 가능성

물리·센서 계약 통과 후에만 공통 학습 설정을 제한된 train/validation pilot로 검토한다. 필요하면 Raw PPO만으로 탐색 분산·step budget의 작동 가능성을 점검하고 네 군 비교 전에 고정한다.

deterministic과 sampled 정책을 모두 확인한다. 평균 행동만 착륙하고 실제 rollout 표본은 전혀 착륙하지 못한다면 표현 비교의 전제가 충족되지 않은 것이다. 이 문제를 teacher·적분 prior로 숨기지 않는다.

네 군의 같은 설정·budget으로 학습 가능한지 개발 gate에서 다시 판정한다. 실제 runtime 속도가 없으므로 “2D와 같은 두 시간 안에 끝난다”는 계획은 세우지 않는다.

---

## 11. 새 실험 계획: 조합 일반화를 직접 검증

이 절의 수·기준은 **후속 구현을 위한 제안**이며 현재 등록된 실험이 아니다. G0–G4와 실제 비용 확인 후, 새 학습 전에 하나의 계획으로 고정한다. 본 작업에서는 학습을 시작하지 않는다.

### 11.1 조합 OOD의 정의

두 인자를 출발점으로 한다.

- A: 초기 표적 상대 방위 구간.
- B: 플랫폼의 고정 수평 이동 방위 구간.

각 인자의 값 범위는 학습에서 모두 경험시키되 일부 A×B 교차 조합을 학습·validation에서 제외한다. ID는 학습에 포함한 조합의 새 episode, 조합 OOD는 제외한 조합의 새 episode다. 속도·고도·카메라·잡음 등 다른 인자의 범위는 맞춘다.

가능하면 각 인자의 주변분포를 같게 하는 균형 교차표를 사용한다. 조합 목록·확률·연속값 샘플러·초기 물리 타당성을 manifest에 기록한다. 해당 교차 조합의 train 출현 건수가 0인지 검증한다.

roll/pitch는 정책의 결과이므로 “그 자세 조합이 학습 중 한 번도 없었다”고 주장하지 않는다. 평가 대상은 **보류한 시나리오 생성 인자의 조합**이다. 기존의 가속·누락·잡음 동시 증가와 구분한다. [R3]

### 11.2 조건표

| 평가 | ID 참조 | OOD/보류 조건 | 역할 |
| --- | --- | --- | --- |
| Motion | 등록 학습 가속 범위 | 등록 범위 밖 가속 크기 | 기존 일반화 추적 |
| Dropout | 등록 누락 범위 | 더 긴 외생 누락 | 기존 일반화 추적 |
| Noise | 공통 nominal noise | 공통 noise 배율 | 기존 일반화 추적 |
| Combined | 세 요인 모두 ID | 세 요인 모두 OOD | 기존 결합 스트레스 |
| Composition | 학습 교차 조합의 새 episode | 보류한 A×B 조합 | **새 주 평가** |

총10셀, 셀당 환경100개를 초기 예산안으로 둔다. 2D의 가속1.5–2.0/누락2.1–2.9/잡음×3을 Isaac에 무조건 복사하지 않는다. 3D의 실제 가속 한계·yaw 유지·광학·단계시간에 맞는 강도를 **표현 모델의 OOD 결과를 보기 전에** 정한다. 선택 근거는 운용 범위와 공통 baseline의 구분 가능성으로 기록한다.

외생 일정·초기조건·센서 noise table은 method 간 짝을 맞춘다. 정책에 따라 실제 검출·궤적·종료시각은 달라질 수 있다. 동결이 필요한 것은 시간축의 외생 난수와 일정이지 policy-step마다 RNG를 똑같이 한 번씩 호출한다는 조건이 아니다.

각 학습 시드에는 독립된 평가 환경 묶음을 배정하고, 그 시드의 네 방법에서는 같은 셀별100개 환경을 사용한다. 이 규칙도 seed manifest에 고정한다. 모든 학습 시드가 하나의 고정100개 환경을 공유하도록 변경한다면, 구간은 그 고정 평가 집합에 조건부임을 명시해야 한다.

### 11.3 주 지표와 비교

**주 지표:** 학습 시드별 Composition OOD의 고정 임무 시간 내 안전 착륙 성공률.

**두 주 비교:** Typed−Semantic MLP, Typed−Untyped.

**실용 비교:** 앞선 두 비교 통과 뒤 Typed−Raw PPO.

기존 네 OOD 평균은 별도 보조 지표와 성능 유지 guard로 둔다. 조합 성능이 낮은데 다른 셀 이득으로 평균을 높여 조합 일반화 우위를 주장하지 않는다. 새 주 지표는 2D의 네 OOD 평균과 다르므로 83.7% 같은 이전 수치와 직접 증감 비교하지 않는다.

보조 지표:

- unsafe contact, safety envelope failure, timeout.
- 접근 구간 기하 시야 이탈률과 첫 이탈 시점; 분모의 접근 구간 길이 공개.
- 전체 tag 가시성과 실제 검출 갱신, 외생 dropout 분리.
- 동일 episode에서 둘 다 성공한 착륙 시간, 모든 episode의 종료시간.
- 접촉 전 위치/접선·수직 속도/tilt/rate.
- 공통 현재 입력에서 행동·시야 변화 방향 진단.
- 실제 parameter 수, 학습 step, 추론 지연.

Attention와 교호작용 크기는 보조 진단이다. 성능 지표를 대체하지 않는다.

### 11.4 개발·확인·예산안

| 단계 | 독립 학습 시드 | 네 군 모델 수 | 10셀×100 평가 | 모델당 학습 상한안 |
| --- | ---: | ---: | ---: | --- |
| 개발 | 5 | 20 | 20,000 episode | 750 update×6 episode |
| 확인 | 20 | 80 | 80,000 episode | 동일 고정 규칙 |

숫자는 3D 축소 물리 backend 캠페인의 제안이다. 실제 시뮬레이션 step·episode 길이·validation/조기종료 규칙·wall time도 등록한다. episode 수만 같다고 실제 표본 수가 같다고 표현하지 않는다.

개발은 후보1개, 확인은 최대1회다. seed 번호는 원격·로컬의 모든 접근 가능한 예약/사용 기록을 감사한 뒤 정한다. 사용 여부를 확인하지 않은 숫자를 새 시드로 발급하지 않는다. 2D의 미사용 확인 시드도 재사용하지 않는다.

### 11.5 gate와 통계

개발:

- 네 군의 모든 학습 시드가 등록 validation 기준을 충족.
- Composition에서 Typed−MLP, Typed−Untyped, Typed−Raw 평균이 모두 양수.
- ID 성공률의 Raw 대비 감소 ≤2%p.
- 기존 네 OOD 평균의 Raw 대비 감소 ≤2%p.
- 평가 OOD 5셀 등가중 unsafe 증가 ≤1%p.
- 기준 실패 시 확인 미개방. 실패 시드 제외·같은 후보 재학습·사후 임계값 조정 없음.

확인:

- 독립 학습 시드·평가 환경 묶음별 네 방법의 짝 차이가 분석 단위. 환경100개를 독립 학습 시드100개로 세지 않는다.
- 두 주 비교는 각각 97.5% 양측 paired-t CI 하한>0. 두 주 비교 family에 Bonferroni 적용.
- 둘 다 통과한 뒤 Raw 대비 Composition 95% CI 하한>0.
- ID와 기존 네 OOD 성공률 guard는 각각 차이의95% CI 하한≥−2%p.
- OOD unsafe guard는 차이의95% CI 상한≤+1%p.
- guard는 사전에 선택한 운영 기준이며 입증된 물리 안전 한계가 아님.
- 시드20개가 작은 효과를 검출할 충분한 검정력을 보장하지 않음. 구간이 넓으면 그대로 보고하며 사후 시드 추가 없음.
- 전체 gate의 결합 주장과 각 조건의 개별 우월성 주장을 구분. secondary 조건별 우월성에는 별도 다중 비교 계획 필요.

### 11.6 Isaac/PX4 확인의 역할과 비용

단위 검사나 local3D 성공을 실제 Isaac 착륙 증거로 사용하지 않는다.

1. G3에서는 소수의 고정된 물리 사례로 센서·구동기·종료 경계를 검증한다.
2. 개발 통과 후에는 **개발의 모든 시드·네 군**에서 ID validation으로 선택된 checkpoint를 사용해 동일 Isaac 사례를 짝 평가한다. 최고 seed만 가져가지 않는다.
3. 예를 들어 20모델×3조건(ID/Composition/Combined)×10환경=600회는 integration 예산안이다. deterministic 기본과 sampled 점검을 분리하고 실제 실시간 배율을 측정한 후 확정한다.
4. 600회가 모두 최대70초라면 시뮬레이션 시간만 약11.7시간이다. 시작·reset·실행 속도는 별도이므로 벽시계 시간을 미리 약속하지 않는다.
5. 작은 Isaac integration 평가는 현 배포에서의 작동과 backend 차이를 보여준다. Isaac 자체의 평균 우월성을 주장하려면 그 backend에서도 독립 학습 시드/짝 환경/통계 예산을 별도로 충족해야 한다.
6. Isaac 격차를 줄이려고 제안 모델에만 보정 controller를 붙이지 않는다.

---

## 12. 구현 순서와 완료 기준

| 작업 | 구체 변경 | 완료 기준 |
| --- | --- | --- |
| A. 상태 정리 | v1 실패 기록 연결, 과거 route 표, 새 문서/ID | 사용자 보고와 검증한 source를 구분 |
| B. 실행 분리 | 새 명시 CLI, stable/minimal 유입 차단 | dry-run이 학습·stack 시작 없이 실제 resolved 설정 출력 |
| C. 센서 계약 | 21D·timestamp·nav provenance·카메라 보정 | 현재값만으로 재현, truth 입력 차단 |
| D. 물리 정합 | local toy 배제, 공통 scenario/camera/actuator | 평면 fixture와 3D 축 결합 검사 통과 |
| E. 특징/모델 | Φ44·8노드·22간선·4군·raw Critic | raw 복원, gradient, 저장/복원, 실측 parameter 확인 |
| F. PPO 이식 | 목적·GAE·latent action·시간·optimizer | 실제 v1 fixture 및 네 군 1 update |
| G. 등록/평가 | 모든 resolved CLI/seed/hash/checkpoint 선택/gate | 가짜 완결 자료와 누락/중복/변조에 대해 종료 기준 검증 |
| H. 제한 개발 | 하나의 등록 후보 실행 | 모든 원자료·실패 유지, 개발 gate 산출 |
| I. 독립 확인 | 개발 통과 시만 별도 등록 | 주 비교·실용 비교·guard 충족 여부 보고 |

현재 구현되지 않은 파일·명령을 기존처럼 표시하지 않는다. 다음은 **구현할 CLI 명세**다.

```bash
./run.sh current-semantics-3d --stage audit --dry-run
./run.sh current-semantics-3d --stage fixtures
./run.sh current-semantics-3d --stage smoke
./run.sh current-semantics-3d --stage register-development
./run.sh current-semantics-3d --stage develop
./run.sh current-semantics-3d --stage register-confirmation
./run.sh current-semantics-3d --stage confirm
./run.sh current-semantics-3d --stage report
```

`develop/confirm`은 등록된 실행 설정의 hash를 확인해야 하며 CLI override로 예산·backend·seed를 바꾸면 거부한다. `--dry-run`은 audit 외 어떤 stage에서도 mutation/학습/stack 시작을 하지 않는 의미로 일관되게 구현한다.

보고 상태는 다음을 분리한다.

- `procedure_complete`: 실행 절차가 끝남.
- `implementation_acceptance`: 계약·물리·센서 검증.
- `learnability_passed`: 네 군의 학습 기준.
- `development_gate_passed` / `confirmation_opened`.
- `scientific_goal_complete`: 사전 주 비교·실용 비교·guard까지 충족.
- `isaac_integration_passed`: 해당 배포/episode 범위의 작동 검증.

단순 `PASSED` 하나로 모든 상태를 대체하지 않는다.

---

## 13. 중단과 해석 규칙

- 시야 결합이 실제 backend에 없으면 3D 본 학습을 시작하지 않는다.
- 공통 PPO가 학습되지 않으면 센서·물리·탐색 계약 문제로 분류하고 관계 모델의 성능 결론을 내리지 않는다.
- Typed가 같은 Φ의 MLP보다 개발에서 낮으면 현재 후보를 종료한다.
- 시야 유지가 늘어도 착륙률이 낮거나 timeout/unsafe가 증가하면 핵심 목표 달성으로 처리하지 않는다.
- 통계적 차이가 없으면 “동등하다”가 아니라 “우위를 확인하지 못했다”고 기록한다.
- 3D가 불가능하면 이 가이드를 성공으로 간주하지 않고 2D 동결 진단과 현재 시야 기하 검토까지만 진행한다.
- 진단 결과가 뒷받침할 때만 2D의 실제 pitch·camera margin을 동일 특징 MLP와 그래프에 함께 넣는 단일 후속 후보를 검토한다. 자동으로 새 학습을 반복하지 않는다.
- 좌표·시야 기하를 수정한 MLP가 이미 같은 개선을 얻으면 그 개선은 특징 구성의 기여이며 R-GAT 관계 유형의 독립 효과라고 주장하지 않는다.

---

## 14. 후속 구현 담당자에게 전달할 작업 지시

```text
목표:
현재 관측만 사용하는 온톨로지–R-GAT–PPO를 제한된 3D 이동 플랫폼 착륙에
적용할 수 있도록 새 연구 경로를 구성한다. 이 문서의 G0–G4를 먼저 처리한다.

기준:
3D main cab3dd2123c7eee3968a8c00753f2ba29d55fe71.
2D 원격 d438a7d와 별도 current_semantics_v1 전달 commit5853781을 구분한다.
실제 사용자가 실행한 v1 등록 묶음을 확인하기 전 동일 실행판이라고 가정하지 않는다.

유지:
현재 센서·추정 입력, 기존 항법과 PX4 내부 안정화, 직접 3축 순가속도,
공통 PPO·보상·종료, 동일 특징 대조, 공통 raw Critic.

제외:
GRU/이력/이전행동/적분tracking_bias/handover_progress,
BC/teacher/DAgger, 추가 head/loss, analytic tracking prior와 상황별 action rewrite.

이번 구현의 첫 결과물:
1) 실제 source/contract/runtime/provenance 감사표
2) 기존 실행 경로를 보존하는 명시적 새 CLI와 dry-run
3) current-only observation/feature/geometry 정의와 검증 fixture
4) 구체적인 미완 항목과 gate 결과

학습:
가이드 작성 요청을 학습 실행 승인으로 확대하지 않는다.
새 본 학습은 구현·검증 완료와 별도 고정 실행 계획이 준비된 이후의 작업이다.
학습 중 등록 source를 바꾸지 않고, 결과를 보고 확인 조건을 완화하지 않는다.
```

---

## 15. 참고 자료와 직접 확인한 코드

### 원 저장소

- [3D 검토 기준 commit](https://github.com/junsuk123/OntoReward-RL/commit/cab3dd2123c7eee3968a8c00753f2ba29d55fe71)
- [3D run.sh](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/run.sh)
- [3D direct 계약](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/python/ontology_rgat/direct_policy/contracts.py)
- [3D 관측](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/python/ontology_rgat/direct_policy/observation.py)
- [3D 모델](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/python/ontology_rgat/direct_policy/models.py)
- [3D graph](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/python/ontology_rgat/direct_policy/graph.py)
- [3D backend](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/python/ontology_rgat/direct_policy/backends.py)
- [3D adapter](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/python/ontology_rgat/direct_policy/isaac_adapter.py)
- [3D 실험 runner](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/python/ontology_rgat/direct_policy/runner.py)
- [공개 migration acceptance](https://github.com/junsuk123/OntoReward-RL/blob/cab3dd2123c7eee3968a8c00753f2ba29d55fe71/Ontology_RGAT_UAV_RL_ISAAC_PX4/docs/matlab_port/MIGRATION_ACCEPTANCE.json)
- [2D 원격 검토 기준](https://github.com/junsuk123/ugv_landing_2d_workspace/commit/d438a7d4096d5b186c731c894306754aa5ab628d)
- [현재 관측 기반 2D 전달 문서](https://app.notion.com/p/406959378b2281d6b926cc752934b501)

### 1차 문헌·공식 명세

- [R1] Falanga et al., “PAMPC: Perception-Aware Model Predictive Control for Quadrotors,” IROS 2018. [원문](https://rpg.ifi.uzh.ch/docs/IROS18_Falanga.pdf). 기동·자세·가시성 결합의 근거이며 MPC를 새 방법으로 도입하자는 제안이 아니다.
- [R2] Goldschmid & Ahmad, “Reinforcement learning based autonomous multi-rotor landing on moving platforms,” Autonomous Robots, 2024. [원문](https://link.springer.com/article/10.1007/s10514-024-10162-8). 축 분리 가정하 1D 정책의 확장 사례.
- [R3] Zambaldi et al., “Relational Deep Reinforcement Learning,” 2018. [원문](https://arxiv.org/abs/1806.01830). 관계 가설에 맞는 미학습 조합 검증의 참고.
- [R4] Satorras et al., “E(n) Equivariant Graph Neural Networks,” ICML 2021. [원문](https://proceedings.mlr.press/v139/satorras21a.html). 회전/이동 등변성이 별도 설계 성질임을 구분하는 근거. EGNN 추가는 이번 범위 밖.
- [R5] NVIDIA, Isaac Sim Conventions. [공식 문서](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/reference_conventions.html). 단위·camera/world/ROS 축. 문서 버전과 실제 설치 버전을 동일시하지 않음.
- [R6] MAVLink Common Message Set. [공식 명세](https://mavlink.io/en/messages/common.html). ENU/NED/FRD 경계.

### 문서 자체의 완료 범위

- 완료: 최신 원격 source 읽기, 기존 제약과 대조, 3D 유의미성 판단, 파일별 리팩토링·수식·후보·검증·중단 지침.
- 미실행: 새 코드 구현, MATLAB/Isaac/PX4 실행, v1 원본 결과 재검산, 새 개발/확인 학습, 3D 성능 향상 확인.
