# 두 pair 실행과 검증

## 기본 실행

```bash
./run.sh --pipelines shin_se_fixed shin_se_onto_rgat_fov --parallel-pairs 2
```

인자 없는 실행도 두 pipeline과 두 pair를 선택한다. Pair 0/1은 별도 PX4 instance,
ROS namespace, gateway/learner UDP port, controller/reset state를 사용한다. 각 worker가
별도 model, rollout buffer와 optimizer를 소유하고, GPU update만 lock으로 직렬화한다.

실행 순서는 공통 keypoint encoder 준비, baseline PPO와 FOV-risk dataset 수집,
BCE R-GAT validation-best 학습/동결, proposed PPO, held-out checkpoint 선택,
paired/crossover 평가, 표·그림 저장이다. 중단 후 같은 명령을 실행하면 compatible
checkpoint와 완료된 평가 행을 재사용한다.

## 짧은 개발 실행

```bash
./run.sh --mode quick --train-episodes 1 --eval-episodes 1 \
  --rgat-data-episodes 2 --rgat-epochs 1 --headless --no-dashboard --no-rviz
```

실제 simulator가 필요 없는 전체 정적·단위 검사는 다음과 같다.

```bash
./scripts/check_workspace.sh
```

실제 stack smoke test 전에는 UDP port 충돌을 피하도록 이전 실행을 종료한다.
`run.sh`는 flight lock으로 두 실행이 같은 stack을 동시에 채택하는 것을 막는다.

## 결과

- `manifest.json`: resolved config, 두 spec, seed/budget, 독립 pair와 artifact provenance
- `rgat/fov_risk_rollouts.npz`: episode ID를 포함한 미래 FOV-loss dataset
- `rgat/fov_risk_model.pt`: frozen validation-best R-GAT과 checksum
- `models/<pipeline>/`: 독립 PPO latest/best/selected checkpoint
- `evaluation/per_episode.csv`: paired/crossover 결과
- `tables/primary_comparison.*`: 두 pipeline 주 비교

Dashboard는 두 agent를 나란히 표시하며 공통 `active_perception`과 proposed-only
`ontology_fov_reward`를 별도 series로 표시한다.
