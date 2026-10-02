# 현행 파이프라인 감사 — 2026-09-29 KST

## 결론

현행 기본 실험은 요청된 선택적 관계 게이팅 방법과 동일하지 않다. 현재 기본 arm은
`shin_se_fixed`와 `shin_se_onto_rgat_state`이며, 후자는 9-node situation graph의
R-GAT 전체와 graph readout을 PPO로 end-to-end 갱신한다. 보상은 Shin Table-III 기반
5개 shaping 성분과 active-perception 성분이고, 기본 설정은 behavior cloning을 켠다.
따라서 기존 결과와 체크포인트는 `selective-rgat-v1` 결과로 재명명하거나 재사용할 수 없다.

## 보존되는 시뮬레이터 경로

다음 경로는 이번 리팩터링에서 변경하지 않는다.

- Isaac Sim/Pegasus scene, UAV/UGV asset, contact truth와 reset
- PX4 SITL/OFFBOARD 시작 및 MAVLink/ROS 2 연결
- 카메라 해상도/FOV/장착 자세, 6-keypoint encoder 입력
- ENU/FLU/body frame 변환과 planar `[a_fwd, a_z, tilt]` 물리 제한
- UGV scenario/route/speed, disturbance/battery, episode horizon
- strict landing, unsafe contact, crash, drift, battery/timeout 종료 판정

새 방법은 위 계층이 생산한 동일한 causal sensor packet만 소비한다. 시뮬레이터 참값은
critic, reward, 평가에만 남고 actor packet에는 들어가지 않는다.

## 현재 데이터 흐름

`camera → keypoint encoder → temporal backbone/LSTM → actor`

critic은 `[UAV proprioception, true relative state]`를 받는 asymmetric critic이다.
`shin_se_*`는 6차원 relative-state auxiliary head/loss를 사용한다. 현행 proposed arm은
같은 keypoint semantics에서 `state_graph.py`의 그래프를 만들고
`GraphStateEncoder`의 mean/max pooling 결과를 기존 actor/critic 입력에 붙인다.
R-GAT kernel, attention vector, relation embedding, readout이 모두 PPO optimizer에 포함된다.

## 현행 보상과 종료

`reward_modes/shin2026.py`는 lateral/vertical progress, vertical-speed,
undershoot, attitude 및 active-perception 항을 쓴다. terminal outcome은 shaping을 대체한다.
이 reward는 레거시 ablation으로 유지하되 새 기본 비교에는 섞지 않는다.

## 현행 실행·체크포인트

- 기본 runner: `python/run_two_pipeline.py` → `python/run_three_pipeline.py`
- 기본 YAML: `config/experiments/planar_three_arm_comparison.yaml`
- 현행 stage: `collect`, `train`, `all`
- 체크포인트: method/config/training-contract/pipeline/reward-design을 검사하지만 새 계약이
  요구하는 observation/reward/action/schema/pretrained/relation/normalization/budget hash를
  하나의 완전한 signature로 저장하지 않는다.
- 선택 점수는 단일/집계된 physical safety score이며 새 robust/worst-scenario 공식과 다르다.

## 문서·코드 불일치와 조치

상위 2-D 저장소의 최신 graph-state 구현도 end-to-end PPO R-GAT을 사용한다. 이번
patched prompt는 이를 그대로 이식하라는 요청이 아니라, 그 관측/2항 보상/평가 원칙을
가져오되 R-GAT base는 오프라인 사전학습 후 고정하고 네 관계 gate만 PPO로 학습하라는
새 계약이다. 따라서 기존 graph-state 코드는 삭제하지 않고 `legacy_end_to_end_rgat`
ablation으로 존치하며 새 계약은 별도 schema/artifact/signature로 분리한다.

## 새 경계

- 관측 authority: `config/observation/causal_packet_v1.json`
- graph authority: `rgat/selective_state.py`
- encoder authority: `ppo/selective_graph_encoder.py`
- reward authority: `reward_modes/two_term.py`
- checkpoint authority: `contracts/signature.py`
- selection authority: `evaluation/selection.py`

이 경계가 완전히 연결되기 전까지 기존 장기 학습을 새 방법의 학습으로 간주하지 않는다.
기존 checkpoint는 자동 변환하지 않고 signature mismatch로 거절한다.
