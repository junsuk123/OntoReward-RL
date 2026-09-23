# 문서 안내

현재 연구는 **평면 3-arm 비교**다(2026-09-23 개편) — 학습하지 않는 PN 유도
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
> 문서는 은퇴한 비교의 것이지만 **이 제약은 개편과 무관하게 그대로다.**

| 문서 | 내용 |
|---|---|
| **[제안 알고리즘 (상태 표현)](ONTOLOGY_RGAT_STATE.md)** | **현재 주 방법** — 9-node 상황 그래프, R-GAT 부호기, 정보경계, 소거 실험 |
| **[평면 제어 엔벨로프](PLANAR_ENVELOPE.md)** | **현재 제어 계약** — 3채널 액션, 두 제약, 오라클이 아닌 이유 |
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

계약의 원본은 문서가 아니라 코드와 설정이다. 설명이 어긋나면
`config/experiments/planar_three_arm_comparison.yaml`,
`config/shin2026-planar-system.yaml`,
`python/ontology_rgat/pipelines/spec.py`,
`python/ontology_rgat/rgat/state_graph.py`,
`python/ontology_rgat/controllers/planar_controller.py`,
`tests/test_ontology_graph_state.py`, `tests/test_planar_envelope.py`,
`tests/test_two_pipeline_fov.py`, `tests/test_three_pipeline.py`가 기준이다.

`THREE_PIPELINE_COMPARISON.md`와 `ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md`는
이전 실험 재현용 legacy 문서이며 현재 주장이나 기본 실행을 정의하지 않는다.
