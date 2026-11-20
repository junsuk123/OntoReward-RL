# 문서 안내

## 제안 방법 문서

제안 방법은 온톨로지 기반 R-GAT 상태 표현을 쓰는 PPO 착륙 정책이다.

| 문서 | 내용 |
|---|---|
| [저장소 README](../../README.md) | 문제 설정, 제안 방법, 검증 결과, 기여 |
| [그림 모음](../../docs/figures/README.md) | 시스템, 모델, 온톨로지 그래프, 결과 그림과 그 설명 |
| [구현 안내](../README.md) | 알고리즘 ↔ 코드, 실험 절차, 결과를 믿기 위한 규칙 |
| [설계 지침](current_semantics/CURRENT_SEMANTICS_3D_REFACTOR_GUIDE_KO.md) | 현재 관측만 쓰는 3축 착륙 문제의 설계 원칙, 개발·확인 실험의 통계 규칙 |
| [항법 열화 실험](current_semantics/WMON_LIVE_NAVIGATION_KO.md) | GNSS 결측, 자기 항법 추정기(Kinematic WLS = 내부 명칭 WMON, CA-KF), 정책들의 개발·확인 실험 기록 |
| [진행 기록](current_semantics/G0_G4_STATUS_KO.md) | 학습 가능성 점검과 단계별 판정의 원 기록 |
| [코드 감사](current_semantics/SOURCE_AUDIT_KO.md) | 관측·플랜트·판정 코드의 출처 감사 |
| [비행 시연](WMON_DEMO_KO.md) | 학습된 정책을 로컬 또는 Isaac에서 비행하며 RViz2로 보기 |
| [실제 기체 안전](HARDWARE_SAFETY.md) | SITL 결과와 실제 기체 배포 사이의 안전 조건 |
| [참고문헌](REFERENCES.md) | 연구·시뮬레이터·비행 제어 스택 자료 |

## 보관 문서

아래 문서는 같은 저장소에 보존한 다른 실행 경로의 설명이다.
제안 방법의 설계나 결과를 정의하지 않는다. 경로마다 계약의 원본은 문서가 아니라 해당 설정과 코드이다.

| 문서 | 다루는 경로 |
|---|---|
| [refactor/](refactor/) | 2축(종·수직) 기준 이식과 그 감사 |
| [matlab_port/](matlab_port/) | MATLAB 기준 구현의 이식 감사 |
| [MINIMAL_OBSERVATION_ROS_PIPELINE_KO.md](MINIMAL_OBSERVATION_ROS_PIPELINE_KO.md) | 최소 관측 ROS 2 파이프라인 |
| [OPERATIONS.md](OPERATIONS.md) | 이전 3축 경로의 스택 운영과 장애 대응 |
| [ONTOLOGY_RGAT_STATE.md](ONTOLOGY_RGAT_STATE.md), [PLANAR_ENVELOPE.md](PLANAR_ENVELOPE.md), [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md), [ARCHITECTURE.md](ARCHITECTURE.md), [SHIN2026_BASELINE.md](SHIN2026_BASELINE.md), [PAPER_FIDELITY.md](PAPER_FIDELITY.md) | 평면 3채널 행동을 쓰는 이전 Isaac 비교 경로 |
| [ONTOLOGY_RGAT_FOV_RISK.md](ONTOLOGY_RGAT_FOV_RISK.md), [ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md](ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md), [THREE_ARM_BURST_COMPARISON.md](THREE_ARM_BURST_COMPARISON.md), [TWO_PIPELINE_COMPARISON.md](TWO_PIPELINE_COMPARISON.md), [THREE_PIPELINE_COMPARISON.md](THREE_PIPELINE_COMPARISON.md) | 온톨로지를 보상에 쓰던 이전 비교 실험 |
