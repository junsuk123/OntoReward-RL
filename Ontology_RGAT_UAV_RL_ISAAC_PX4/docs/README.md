# 문서 안내

현재 연구는 **3-arm 급가속 이탈 비교**다 — 학습하지 않는 visual servo 대조군,
레퍼런스 `shin_se_fixed`, 제안 `shin_se_onto_rgat_recovery`가 패드가 카메라를
벗어나는 단일 덱에서 같은 seed로 겨룬다. 두 학습 arm 사이의 단일 요인은 여전히
FOV-risk 가산항 하나다.

아래 순서로 읽으면 문제 → 알고리즘 → 시스템 → 실험 → 운영으로 이어진다.

> **먼저 읽을 것**: [3-arm 비교](THREE_ARM_BURST_COMPARISON.md) §5는 현재 저장소의
> 모든 수치를 어떻게 읽어야 하는지를 정한다. 제어 대역폭이 강제되지 않아
> `steps × dt`로 계산되는 시간 지표가 7–9배 과소 기록된다.

| 문서 | 내용 |
|---|---|
| **[3-arm 급가속 이탈 비교](THREE_ARM_BURST_COMPARISON.md)** | **현재 주 비교** — 덱, 세 arm, 제어 포락선, 제어 대역폭 제약 |
| [시스템 개요](SYSTEM_OVERVIEW.md) | 한 장으로 보는 데이터 흐름과 정보경계 |
| [제안 알고리즘](ONTOLOGY_RGAT_FOV_RISK.md) | 10특징 15-node graph, 미래 FOV 비가용 시간 비율, 목적함수, readout, 동결 |
| [Shin et al. baseline](SHIN2026_BASELINE.md) | baseline 구성 요소와 구현 대응, 정확한 보상식 |
| [아키텍처와 정보경계](ARCHITECTURE.md) | actor/critic/온톨로지 경계, 좌표계, 타이밍, 종료 판정 |
| [실험 설계 (6-덱 2-arm)](TWO_PIPELINE_COMPARISON.md) | 통제변수, 실험 요인, 지표 정의, 통계 절차, 실험 이력 |
| [논문 대조](PAPER_FIDELITY.md) | 원문에서 확인된 값, 미기재 항목, 선언된 백엔드 이탈, 보고 금지 사항 |
| [운영](OPERATIONS.md) | 실행 프로파일, 산출물, 상태 확인, 장애 대응 |
| [실제 기체 안전](HARDWARE_SAFETY.md) | SITL 결과와 실제 배포 사이의 안전 gate |
| [참고문헌](REFERENCES.md) | 연구·시뮬레이터·flight stack 자료 |

계약의 원본은 문서가 아니라 코드와 설정이다. 설명이 어긋나면
`config/experiments/three_arm_burst_comparison.yaml`,
`config/ontology/fov_recovery.yaml`, `python/ontology_rgat/pipelines/spec.py`,
`tests/test_two_pipeline_fov.py`, `tests/test_three_pipeline.py`가 기준이다.

`THREE_PIPELINE_COMPARISON.md`와 `ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md`는
이전 실험 재현용 legacy 문서이며 현재 주장이나 기본 실행을 정의하지 않는다.
