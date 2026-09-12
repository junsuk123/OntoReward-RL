# 문서 안내

기본 실험은 저장소 루트의 `./run.sh`로 실행하는 3개 파이프라인 통제 비교다.
**legacy**라고 명시된 문서는 보존된 부가 실험을 설명하므로 기본 actor, ontology,
reward, budget 또는 output schema를 추론하는 데 사용하면 안 된다.

## 먼저 읽을 문서

| 문서 | 용도 |
|---|---|
| [프로젝트 README](../README.md) | 설치, 단일 명령 실행, output과 간결한 기술 계약 |
| [시스템 개요](SYSTEM_OVERVIEW.md) | component 구성과 end-to-end 실행 순서 |
| [3개 파이프라인 통제 비교](THREE_PIPELINE_COMPARISON.md) | 과학적 통제, 수식, metric과 budget |
| [상태 적응형 보상 가중치](ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md) | 5성분 R-GAT 가중치, 데이터, loss, 동결 및 평가 계약 |
| [운영](OPERATIONS.md) | monitoring, 재시작 의미, log와 fault 진단 |
| [아키텍처](ARCHITECTURE.md) | simulator, PX4, ROS, 좌표계, timing과 migration 경계 상세 |
| [Shin-2026 baseline](SHIN2026_BASELINE.md) | 논문-코드 대응과 의도적 변경 |
| [데이터 흐름 감사](DEPENDENCY_DATAFLOW_AUDIT.md) | privileged-information 제외와 실행 가능한 guard |
| [실제 기체 안전](HARDWARE_SAFETY.md) | 필수 실제 기체 제한과 opt-in |
| [참고문헌](REFERENCES.md) | 주요 integration 및 algorithm 출처 |

## 기본 설정 요약

| 항목 | 현재 기본값 |
|---|---|
| Pipeline | `shin_se`, `no_se`, `onto_no_se` |
| Actor 입력 | 512×320 흑백 영상 + body velocity 3 + quaternion 4 |
| 공통 temporal model | 동결 6-keypoint encoder, 512-unit LSTM, 256-D latent |
| Actor latent 입력 | `y[6:256]` + 7-D proprioception |
| Action | heading-frame `vx, vy, vz, yaw_rate` |
| 기본 ontology | 18 node, directed edge 35개, relation 4종, node당 feature 24개 |
| R-GAT | 폭 24 relation-attention layer 2개, direct frozen `Phi(G)` |
| 기본 scene | Meta-Sejong S5 / `gwanggaeto` |
| UGV route/속도 | 37-point, 99.70 m 폐곡선 도로; 0.25–0.60 m/s 추출 |
| 기본 budget | warm-up 8 + PPO 264 × 3 = 학습 비행 800회 |
| Reward-design 데이터 | 실제 비행 최소 40회, 두 결과 class와 성공한 visual recovery 필수, hard cap 120회 |
| 평가 | scenario 7종 × seed 5개 × pipeline 3개 |
| Dashboard | `http://127.0.0.1:8770/` |
| Runtime log | `/tmp/ontology_rgat_stack/` |
| 결과 | `results/three_pipeline/<mode>/` |

## 현재 시각 자료

### Isaac Sim 비행

![Meta-Sejong S5 scene의 실제 UAV와 착륙 UGV](images/isaac_sim_s5_live.png)

현재 full run의 실제 Isaac Sim viewport다. 파란 debug line/trail은 운용자용
telemetry이며 policy 입력이 아니다.

### 실시간 monitor

![MATLAB 스타일 실시간 dashboard](images/live_dashboard_status.png)

2026-09-12 진행 중이던 실제 full run 화면이다. 관측 가능성을 문서화하기 위한
자료이며 최종 benchmark figure가 아니다.

### 감사된 S5 route

![Meta-Sejong S5 도로와 UGV waypoint](images/metasejong_gwanggaeto_ugv_route.png)

이 plot은 `tools/check_metasejong_route.py`가 현재 YAML과 licensed USD mesh를 읽어
만든다. 따라서 설명용 campus sketch가 아니라 설정된 route에 관한 근거다.

## Legacy 자료

다음 파일과 그림은 이전 cooperative urban pipeline을 설명한다.

- `scripts/run_metasejong_pipeline.sh`
- legacy reward-arm CLI를 통해 실행하는 `python/run_pipeline.py`와
  `python/run_shin2026_pipeline.py`
- 23-channel actor observation
- 14-node/38-edge ontology graph
- R-GAT으로 증류한 고정 reward coefficient 8개
- `system_architecture-metasejong-v3.png`, `rgat_network-v4.png`,
  `reward_function-v3.png`, `rl_observation_state-v3.png`

재현성을 위해 보존하지만 기본 `onto_no_se` direct-R-GAT 구현의 그림은 아니다.

[사용 종료된 MATLAB tree](../legacy_matlab/README.md)는 참고 전용이다. 활성 Python,
simulator, ROS 또는 launcher 코드가 여기에 의존하면 workspace checker가 실패한다.

## 문서 유지관리 규칙

설정값의 기준은 `config/`, 동작의 기준은 `python/ontology_rgat/`, `isaac_sim/`,
ROS gateway와 `scripts/`다. 개별 run의 기준은 생성된 result manifest다. 설명과
실행 코드가 다르면 오래된 screenshot이나 legacy diagram을 사실로 취급하지 말고
설명을 갱신하고 해당 run의 configuration hash를 기록한다.
