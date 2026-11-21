# 문서 안내

제안 방법은 착륙 온톨로지를 정책의 구조로 쓰는 R-GAT 강화학습 정책 ORL(`current_semantics/orl.py`)이다.

| 문서 | 내용 |
|---|---|
| [저장소 README](../../README.md) | 문제 설정, 제안 방법, 검증 결과, 기여 |
| [그림 모음](../../docs/figures/README.md) | 전체 구조, 시스템 파이프라인, 온톨로지 그래프, 시나리오와 접근 속도장, 학습 속도 결과, 같은 예산·Isaac 결과, Isaac Sim + RViz2 시연 영상 |
| [구현 안내](../README.md) | 알고리즘 ↔ 코드, 실험 절차, 모니터, 결과를 믿기 위한 규칙 |
| [연구 기록](current_semantics/G0_G4_STATUS_KO.md) | 제안 모델 ORL의 정의(20절), 학습 속도 실험(21절), 시연 영상(23절), 같은 예산 실험과 Isaac 평가(24절)의 원 기록 |
| [코드 감사](current_semantics/SOURCE_AUDIT_KO.md) | 관측·플랜트·판정 코드의 출처 감사 (보정 상수의 근거로 코드가 인용한다) |
| [참고문헌](REFERENCES.md) | 연구·시뮬레이터·비행 제어 스택 자료 |

## 보관된 문서

- 2026-11-21에 중복된 과거 결과와 이점이 확인되지 않은 결과를 모든 문서에서 지웠다.
  - 원래 문서와 3D 설계 지침(`CURRENT_SEMANTICS_3D_REFACTOR_GUIDE_KO.md`)은 `archive/superseded_docs_20261121.tar.zst`(git 제외)로 옮겼다. git 기록(커밋 `a6fcd98`)에도 있다.

ORL이 쓰지 않는 경로의 문서는 2026-11-20에 `archive/non_orl_20261120.tar.zst`로 옮겼다(git에서 제외).
최소 관측 ROS 파이프라인, 2축 기준 이식, MATLAB 이식, 이전 Isaac 비교 경로, 온톨로지를 보상에 쓰던 비교 실험,
항법 열화 실험(Kinematic WLS = 내부 명칭 WMON, CA-KF), bond/1, 비행 시연, 실제 기체 안전 문서가 여기에 들어 있다.
목록과 복원 방법은 보관본 안의 `MANIFEST.md`에 있다.

---

Copyright © 2026 Jun Suk Kim (김준석). 관리자: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
