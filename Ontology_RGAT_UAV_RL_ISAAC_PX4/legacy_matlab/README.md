# MATLAB 참고 코드

이 디렉터리는 수치 비교를 위해 보존한 참고 코드이며 현재 실행 파이프라인에 포함되지
않는다. 최종 구현은 [`../python/`](../python/)과 저장소 루트의 `run.sh`다.

- 현재 PPO/R-GAT 학습은 PyTorch로 실행한다.
- 현재 simulator는 Isaac Sim/Pegasus/PX4다.
- `run.sh`, Python module 또는 테스트가 이 디렉터리를 import하지 않는다.
- 활성 알고리즘과 보상 수식은 [프로젝트 README](../README.md)와
  [제안 알고리즘](../docs/ONTOLOGY_RGAT_ADAPTIVE_REWARD_WEIGHTING.md)을 따른다.

MATLAB 파일은 현재 결과 생성이나 checkpoint 재개에 사용하지 않는다.
