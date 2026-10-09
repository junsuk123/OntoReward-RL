# MATLAB direct-policy 이식 감사

감사일은 2026-10-09(KST)이다. 대상 저장소의 작업 시작 HEAD는
`5a12d937333996a47a6e605a413eb037519a69aa`이고 작업 시작 시 추적 파일 변경은
없었다. 원본은 GitHub에서 별도 임시 checkout하여
`ce2a3e5e5d8e9a95158e599a8730fe0307287ece`를 확인했다. 인접한
`codes/ugv_landing_2d_workspace`는 `.git`이 없는 복사본이며 manifest의 파일 배치와
달랐으므로 기준 자료로 사용하지 않았다.

실제 루트와 진입점은 다음과 같다.

- 저장소 루트: 이 프로젝트의 상위 디렉터리, 진입점 `run.sh`
- Python 프로젝트 루트: `Ontology_RGAT_UAV_RL_ISAAC_PX4/`
- 기존 무인자 경로: `scripts/run_minimal_system.sh`; 변경하지 않음
- 기존 reference 경로: `run_two_axis_experiment.py` 및
  `two_axis_reference_v28_active.yaml`; MATLAB direct-policy와 다른 9노드 계약
- 기존 spatial 경로: `run_spatial_pipeline.py`; reference supervisor가 있는 별도 계약
- 기존 Isaac 경로: `isaac-legacy`; 새 MATLAB 계약 구현으로 간주하지 않음
- 새 경로: `./run.sh matlab-port ...`; 무인자 실행이나 기존 routing과 분리

원본의 실제 활성 분기는 `primaryConfig.m`의 `trainingRegime='direct_ppo_v1'`이다.
정책 관측은 README의 과거 24D/26D 설명이 아니라 `vectorSchema.m`과 `toVector.m`의
12D이다. `inputNormalization.enabled=false`, 7개 노드, 4개 관계, 9개 의미 간선과
7개 self 간선, 단일 R-GAT 층의 local skip, 노드별 identity grouped readout,
Actor/Critic 별도 encoder가 활성이다. `pretrain=false`,
`freezeStaticBackbone=false`, `graphAdaptationWarmupFraction=0`이며 raw semantic
bypass와 relation action residual은 활성 기본 경로에 없다.

보상은 `reward_v5`이다. pseudo-Huber running goal cost, 카메라 접근축,
상대속도/수직속도 potential, readiness 차분, terminal table을 함께 사용한다. 정책은
latent Gaussian `u`를 표본화하고 환경에서 `a=a_max*tanh(u)`를 적용한다. PPO
log-probability는 `u` 좌표에서 계산한다.

설치 확인값은 Python 3.10.12, NumPy 1.26.4, PyTorch 2.5.1+cu121, SciPy 1.13.1,
OpenCV 4.11.0, PyYAML 6.0.3이다. 감사 시 Isaac, PX4, XRCE-DDS, ROS2 실행 프로세스는
없었다. 실제 Isaac 실행이나 MATLAB 실행은 감사 단계에서 수행하지 않았다.

## 불일치와 처리

- target의 13D minimal 관측, reference-v28 packet, spatial supervisor를 source 호환
  구현으로 재사용하지 않았다. 새 `direct_policy` 패키지로 격리했다.
- 원본의 MATLAB 내부 optional 3D는 과거 packet/reward-v2 경로다. 새 3D는 고정 21D
  스키마와 reward-v5 역할을 보존하는 `spatial_direct_v1`으로 별도 버전화했다.
- 원본 commit의 PnP는 자체 Gauss-Newton이다. Python은 동일 corner/ID/캘리브레이션
  경계와 capture stamp를 유지하되 OpenCV iterative PnP를 사용한다. 따라서 실제
  MATLAB golden fixture 비교 전에는 PnP 수치 정합을 PASSED로 표시하지 않는다.
- 초기 감사 시점에는 Isaac을 NOT_RUN으로 남겼다. 후속 구현에서
  `run.sh matlab-port-all`의 명시적 opt-in 경로로 2D/3D 실제 비행과
  command-owner/timestamp/cleanup을 검증했다.
