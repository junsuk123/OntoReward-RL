# 문서 안내

기본 연구 대상은 `shin_se_fixed`, `no_se_fixed`,
`onto_rgat_adaptive_weight_no_se`의 통제 비교다. 모든 문서는 현재 활성
`run.sh` 파이프라인을 기준으로 한다.

| 문서 | 내용 |
|---|---|
| [프로젝트 README](../README.md) | 강화학습 정의, 설치, 실행, 코드와 결과 구조 |
| [저장소 README](../../README.md) | 연구 질문, 비교군, 핵심 수식과 전체 실행 명령 |
| [시스템 개요](SYSTEM_OVERVIEW.md) | end-to-end component와 데이터 흐름 |
| [제안 알고리즘](ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md) | 23-node ontology, hybrid R-GAT, 학습 loss와 동결 계약 |
| [3개 파이프라인 비교](THREE_PIPELINE_COMPARISON.md) | 공통 조건, 유일한 실험 요인과 평가 지표 |
| [Shin et al. baseline](SHIN2026_BASELINE.md) | 논문 구성과 저장소 구현의 대응 |
| [아키텍처](ARCHITECTURE.md) | 모듈, 좌표계, actor/critic 정보경계, simulator/ROS 연결 |
| [운영](OPERATIONS.md) | 실행, 재개, monitoring과 장애 진단 |
| [실제 기체 안전](HARDWARE_SAFETY.md) | SITL checkpoint와 실제 기체 배포 사이의 안전 gate |
| [참고문헌](REFERENCES.md) | 논문, PX4, Isaac Sim, ROS와 hardware 자료 |

## 다이어그램

| 이미지 | 설명 |
|---|---|
| [rl_contract.svg](images/rl_contract.svg) | RL의 state, observation, action, reward, agent, environment |
| [pipeline_comparison.svg](images/pipeline_comparison.svg) | 최종 3개 비교 arm |
| [adaptive_rgat_hybrid.svg](images/adaptive_rgat_hybrid.svg) | 제안 hybrid Ontology R-GAT reward |
| [landing_success_gate.svg](images/landing_success_gate.svg) | 엄격한 착륙 성공 조건 |
| [isaac_sim_s5_live.png](images/isaac_sim_s5_live.png) | 실제 Isaac Sim runtime |
| [live_dashboard_status.png](images/live_dashboard_status.png) | MATLAB 스타일 dashboard |
| [metasejong_gwanggaeto_ugv_route.png](images/metasejong_gwanggaeto_ugv_route.png) | 활성 UGV 도로 경로 |

알고리즘 다이어그램은 편집·검토 가능한 SVG, 실제 환경 화면은 원본 PNG로 관리한다.
