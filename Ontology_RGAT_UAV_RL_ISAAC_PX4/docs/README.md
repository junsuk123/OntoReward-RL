# 문서 안내

2026-10-05 최신 진입점은 [reference v2.8 대조 보고서](refactor/REFERENCE_V28_AUDIT_KO.md)와
저장소 루트 README다. **아래 문서 목록/연구 설명은 기존 Isaac 실험의 문서**다.
새 경량 reference 프로파일의 설명이나 현재 성능으로 읽지 않는다.

새 `spatial` 경로의 실제 Isaac PPO·재평가·측면 뷰·접촉 검증은
[운영 안내](OPERATIONS.md)와 위 대조 보고서의 2026-10-04 후속 절에 있다.
현재 기본은 `spatial-causal-rgat/5`다. 70초 실제 3-arm 실행은 완료됐지만
학습 정책의 평가 착륙은 관측되지 않아 acceptance는 false다.
기능 검증과 착륙 성능 검증을 구분한다.
외란 정합 후보 v8은 43-field packet과 실제 공간 `(ax,ay,az)`를 사용하며
로컬에도 동일 seed의 외력·토크·인계 충격을 적용한다. 아직 default v5를 대체한
통과 모델이 아니다. v8 완료 전체 회귀는 1,272 passed/2 skipped이며, 실제 학습 착륙
acceptance는 여전히 미충족이다. 상세 원시 결과는 대조 보고서와 `refactor/progress.json`에 있다.

2026-10-05 추가 v9는 47-field packet, x/y 각각 9×12 reference 문맥/shared R-GAT,
실제 capture-time 및 own EKF history 정합, 로컬 75ms 영상 지연과 공유 CV–CA–CV를
사용한다. 기존 서명과 호환되지 않으며 부분 학습 결과를 보존하고 v10 검증을 우선했다.
v10은 공용 trust/latch/제동거리/corridor와 외란 backup, measured-view reward를
별도 서명으로 추가했다. 명령의 물리 시간 50Hz 정합과 bounded LAND 재전송을 포함한
최신 전체 회귀는 소유 자식 회수·초기 식별 정보 준비 대기를 포함해 1,358 passed/2 skipped다. 독립 actual 진단 착륙 및 두 optical-loss
hold/cleanup을 관측했지만 PPO 착륙/2D 성능 동등성은 미검증이다. v10의 두 학습 seed
× 세 arm 로컬 학습과 120회 고정 test는 완료했다. 착륙 vector/flat/graph **1/0/0**,
unsafe **2/3/1**이며 actual PPO·저장·재검증은 진행 중이다.

아래 역사적 연구는 **평면 3-arm 비교**다(2026-09-23 개편) — 학습하지 않는 PN 유도
대조군, 레퍼런스 `shin_se_fixed`, 제안 `shin_se_onto_rgat_state`가 속도가
구간별로 바뀌는 세 직선 주행 덱에서 같은 seed로 겨룬다. 세 arm 모두 같은
평면 3채널 액션 `[a_fwd, a_z, tilt]`으로 비행하고, 두 학습 arm은 **보상까지
완전히 같다**. 단일 요인은 관측에 온톨로지 상황 그래프가 들어가는지 하나다.

이전 설계(온톨로지가 보상에 `-λ_fov q(G_t)`를 더하던 방법)는 은퇴했지만 arm과
문서가 그대로 남아 있고 지금도 실행할 수 있다.

아래 순서로 읽으면 문제 → 알고리즘 → 시스템 → 실험 → 운영으로 이어진다.

> **먼저 읽을 것**: [3-arm 급가속 이탈 비교](THREE_ARM_BURST_COMPARISON.md) §5는
> 현재 저장소의 모든 수치를 어떻게 읽어야 하는지를 정한다. 제어 대역폭이
> 강제되지 않아 `steps × dt`로 계산되는 시간 지표가 7–9배 과소 기록된다. 그
> legacy 경로에 대한 주의다. 새 spatial 경로는 별도 Isaac physics clock을 사용하므로
> 해당 7–9배 오차 주장을 그대로 적용하지 않는다. 최신 측정은 대조 보고서에 있다.

| 문서 | 내용 |
|---|---|
| **[기존 제안 알고리즘 (상태 표현)](ONTOLOGY_RGAT_STATE.md)** | **Isaac legacy 방법** — 9-node 상황 그래프, R-GAT 부호기, 정보경계, 소거 실험 |
| **[기존 평면 제어 엔벨로프](PLANAR_ENVELOPE.md)** | **Isaac legacy 제어 계약** — 3채널 액션, 두 제약, 오라클이 아닌 이유 |
| [시스템 개요](SYSTEM_OVERVIEW.md) | 한 장으로 보는 데이터 흐름과 정보경계 |
| [Shin et al. baseline](SHIN2026_BASELINE.md) | baseline 구성 요소와 구현 대응, 정확한 보상식 |
| [아키텍처와 정보경계](ARCHITECTURE.md) | actor/critic/온톨로지 경계, 좌표계, 타이밍, 종료 판정 |
| [은퇴: FOV-risk 보상항](ONTOLOGY_RGAT_FOV_RISK.md) | 이전 제안 방법. 10특징 15-node graph, 동결 readout. arm은 지금도 실행 가능 |
| [은퇴: 3-arm 급가속 이탈 비교](THREE_ARM_BURST_COMPARISON.md) | 이전 주 비교. **제어 대역폭 제약의 원 기술** |
| [은퇴: 6-덱 2-arm 설계](TWO_PIPELINE_COMPARISON.md) | 통제변수, 실험 요인, 지표 정의, 통계 절차, 실험 이력 |
| [논문 대조](PAPER_FIDELITY.md) | 원문에서 확인된 값, 미기재 항목, 선언된 백엔드 이탈, 보고 금지 사항 |
| [운영](OPERATIONS.md) | 실행 프로파일, 산출물, 상태 확인, 장애 대응 |
| [실제 기체 안전](HARDWARE_SAFETY.md) | SITL 결과와 실제 배포 사이의 안전 gate |
| [참고문헌](REFERENCES.md) | 연구·시뮬레이터·flight stack 자료 |

아래 legacy 계약의 원본은 문서가 아니라 코드와 설정이다. 설명이 어긋나면
`config/experiments/planar_three_arm_comparison.yaml`,
`config/shin2026-planar-system.yaml`,
`python/ontology_rgat/pipelines/spec.py`,
`python/ontology_rgat/rgat/state_graph.py`,
`python/ontology_rgat/controllers/planar_controller.py`,
`tests/test_ontology_graph_state.py`, `tests/test_planar_envelope.py`,
`tests/test_two_pipeline_fov.py`, `tests/test_three_pipeline.py`가 기준이다.

`THREE_PIPELINE_COMPARISON.md`와 `ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md`는
이전 실험 재현용 legacy 문서이며 현재 주장이나 기본 실행을 정의하지 않는다.
