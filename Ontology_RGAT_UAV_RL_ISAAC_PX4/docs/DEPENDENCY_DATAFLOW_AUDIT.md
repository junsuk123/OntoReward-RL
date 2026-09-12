# 의존성 및 데이터 흐름 감사

[문서 안내](README.md) · [3개 파이프라인 통제 비교](THREE_PIPELINE_COMPARISON.md) ·
[아키텍처](ARCHITECTURE.md)

감사 갱신일: 2026-09-12. 범위는 현재 `main` 작업 트리와 기본
`shin_se / no_se / onto_no_se` 실행기다. 이 문서는 코드 경계를 감사한
결과이며, 아직 완료되지 않은 비행 성능에 관한 주장이 아니다.

## 기본 Actor 계약

`ActorObservation`은 정확히 512×320 흑백 영상과 UAV 탑재 센서 값 7개만
노출한다. 탑재 센서 값은 body frame 속도 3개와 자세 quaternion 4개다.
동결된 6-keypoint encoder, temporal LSTM, actor는 세 파이프라인에서 같다.

| 데이터 | Actor | `shin_se` 보조학습/critic | 기본 ontology | Terminal/평가 |
|---|---:|---:|---:|---:|
| 원본 흑백 영상 | 예 | 예 | keypoint/heatmap 경유 | 진단 |
| UAV body 속도/quaternion | 예 | 예 | 범위가 제한된 운동/자세 semantic | 예 |
| 6-keypoint 출력/heatmap | encoder 경유 | 예 | 예 | 진단 |
| 탑재 battery reserve | 아니요 | 아니요 | 예 | 예 |
| 실제 상대 상태 | 아니요 | 보조학습 target + critic | 아니요 | 예 |
| 추정 상대 상태 | 아니요 | `shin_se`만 | 아니요 | 진단 전용 |
| marker pose 계산값 | 아니요 | 아니요 | 아니요 | 설정/시각화 전용 |
| UGV pose/속도 또는 wheel odometry | 아니요 | 아니요 | 아니요 | 설정/scoring 전용 |
| 플랫폼 GNSS/V2V | 아니요 | 아니요 | 아니요 | legacy profile 전용 |
| simulator 접촉/실제값 | 아니요 | 선언된 critic에서만 | 아니요 | 예 |

Actor는 항상 `y[6:256]`을 사용한다. `shin_se`조차 추정 상태 6개를 actor
입력에 덧붙이지 않는다. Critic은 학습 중에만 별도로 13개 값을 받으며,
배포 wrapper 입력에는 존재하지 않는다.

## 기본 ontology 경계

`semantic_observation_from_payload`가 허용하는 입력은 다음뿐이다.

```text
keypoints, heatmaps, keypoint_visibility, proprioception, battery_reserve
```

estimate, relative state, platform, pad/deck motion, simulator/ground truth,
critic, platform GNSS 또는 privileged provenance를 뜻하는 별칭은 중첩된
경우까지 거부한다. 알 수 없는 최상위 필드도 실패 처리한다. 출력은 정규화된
semantic/history 관측 12개와 고정 18-node/35-edge graph다. Heatmap entropy가
영상 기하 사용 여부를 제어하므로 정보가 없는 frame의 soft-argmax 좌표가
가짜 alignment나 scale을 만들 수 없다.

이 경계는 제안 방식이 대체하려는 명시적 metric estimator를 몰래 복원하지
못하게 한다. Simulator 실제값으로 episode에 `+1/-1` label을 붙일 수는 있지만
R-GAT feature로 넣을 수는 없다.

## 보상 설계 데이터셋

Semantic dataset 형식은 `ontology_rgat.semantic_rollouts/2-recovery-aware`다.
Manifest에는 graph schema, config hash, source-policy checkpoint digest,
behavior mixture, seed, 비행/sample/contact 수, environment step, class 수,
성공한 시각 재획득 수와 금지 입력 선언을 기록한다.

학습된 `no_se` policy가 behavior source다. Metric pose를 쓰지 않는 image-plane
servo 보정, 제한된 Gaussian noise, loss-recovery climb과 제한된 무작위 탐색으로
coverage를 높인다. 합성 success/failure 삽입은 금지한다. 최소 비행 수, 두
terminal class, 그리고 loss→reacquisition→landing 성공 궤적 수를 모두 충족할
때까지만 hard cap 범위에서 추가 수집한다. 같은 비행의 인접 frame이
train/validation에 섞이지 않도록 episode 단위로 분할한다.

## 설정 및 checkpoint 보호 장치

- 불변 `PipelineSpec`이 estimator, auxiliary loss, active reward, ontology input
  mode와 direct-potential 사용 여부를 정의한다.
- 시작 시 YAML 선언이 해당 spec과 정확히 일치하는지 검사한다.
- R-GAT, PBRS, PPO의 gamma가 같은지 검사한다.
- Recurrent checkpoint는 format, pipeline spec, config hash, reward hash,
  optimizer state와 curriculum state를 저장한다.
- 호환되지 않는 artifact는 억지로 shape-load하지 않고 보관한 뒤 다시 학습한다.
- Direct R-GAT artifact는 model digest, graph schema와 동결 상태를 검증한다.
- JSON 직렬화는 NaN을 거부하고 trainer는 유한하지 않은 loss/gradient에서 멈춘다.

관련 자동 검사는 `tests/test_shin2026_integrity.py`,
`tests/test_three_pipeline.py`와 protocol/config test module에 있다.

## 실행 오류 경계

Infrastructure 자동 복구 범위는 의도적으로 좁다. Gateway timeout, 실제
simulated-clock 정지, gateway가 순수 Offboard heartbeat loss로 분류한 오류만
불완전 trajectory를 버리고 소유 중인 stack을 재시작한 뒤 같은 seed를 다시
시도할 수 있다. Entry geometry, marker visibility, estimator validity, policy
health, 그 밖의 PX4 failsafe와 terminal outcome은 이 과정에서 숨기거나 label을
바꾸지 않는다.

루트 `run.sh`는 OS lock으로 하나뿐인 flight-control resource에 대한 접근도
직렬화한다. 두 learner가 reset 경쟁을 벌이거나 상대의 UDP reply를 받는 일을
막는다.

## 유지되는 legacy 경계

`semantic.make_observation`의 23-value cooperative observation,
`/landing_pad/state/odom`, urban GNSS/V2V 데이터, 14-node graph와 증류된 8개
가중치 reward는 `scripts/run_metasejong_pipeline.sh`에 속한다. 확장 실험에는
유효하지만 기본 non-cooperative actor 계약과 비교하면 privileged 정보다.
기본 artifact와 legacy artifact는 의도적으로 호환되지 않는다.
