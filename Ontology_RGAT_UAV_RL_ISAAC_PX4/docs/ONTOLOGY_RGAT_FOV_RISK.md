# Ontology-R-GAT 미래 FOV 비가용성 보상

Contract 원본은 `config/ontology/fov_recovery.yaml`이며, 아래 설명과 코드가
어긋나면 contract와 `tests/test_two_pipeline_fov.py`가 기준이다.

## 입력 경계

Graph 입력은 `KeypointConfidence`, `VisibleKeypointFraction`, `FOVMargin`,
`ApparentTargetScale`, `ImagePlaneMotion`, `ScaleRate`, `VisibilityMemory`,
`ReacquisitionTrend`, `MeasurementValidity`, `MeasurementAge`의 10개 bounded
visual feature뿐이다. 뒤의 두 개는 같은 keypoint 이력에서 나온 missingness와
age이며 새로운 privileged 입력이 아니다.

정규화 centroid `(cx, cy)`에 대해
`FOVMargin = clip(1 - max(abs(cx), abs(cy)), 0, 1)`이다. 이 centroid는 *마지막으로
신뢰 가능했던* 측정값이므로, 측정이 없는 동안에는 값이 정지한다. 따라서 margin은
항상 `MeasurementValidity`와 함께 공개되고 `BoundarySafety` seed는
`fov_margin * measurement_validity`로 gate된다(contract rule `R-03`). 패드를 화면
중앙에서 놓친 직후 raw margin이 1.0에 가깝게 읽히는 상황을 "안전"으로 넘기지
않기 위한 것이다.

Simulator ground truth, true/estimated relative position·velocity, 6-D
relative-state estimate와 critic-only state는 함수 인터페이스와 dataset schema에서
거부된다. 기존 estimator는 별도의 baseline 구성으로 정상 동작한다.

## Graph

15개 node는 10개 입력과 `PerceptionQuality`, `BoundarySafety`, `TargetMotion`,
`VisualObservability`, `FutureFOVUnavailability`다. 관계는 `indicates`,
`supports`, `constrains`, `self`이며 `FutureFOVUnavailability`가 출력 node다.
Edge와 self-loop 순서는 고정되어 동일 입력은 byte-identical graph를 만든다.

관계 이름은 relational kernel이 쓰는 구조적 edge type이고 강제되는 단조성 제약이
아니다. `rgat.fov_graph.unreachable_input_nodes()`는 출력까지 non-self 경로가 없는
입력 node를 반환하며, contract rule `R-01`에 따라 항상 비어 있어야 한다.

## Target과 학습

실제 control frequency를 `f`라 하면 1.0초 horizon은 `H = round(1.0*f)` step이다.
Target은 이진 라벨이 아니라 시간 비율이다.

```
y_t = sum(1 - V_centre[t+k] for k in 1..H) / H
```

`V_centre`는 `isaac_sim/keypoint_geometry.geometric_pad_center_in_fov`의 기하학적
패드 중심 가시성 하나뿐이다. 모델 출력은 현재까지 관측의 graph에 대한
`E[y_t | G]`이며 binary loss 확률이 아니다.

`H` step 미래가 관측되지 않은 episode 꼬리 sample은 `valid=False`, target `NaN`으로
mask한다(rule `R-05`). 0으로 채우거나 episode를 이어 붙이지 않는다.

학습 목적은 두 항이다.

* Huber regression (기본 `huber_delta=0.1`), 지도 가능한 sample만 사용.
* Contract rule `R-04`: `MeasurementAge` node 값만 올렸을 때 예측 비가용성이
  감소하면 벌하는 one-sided hinge. 기본 가중치 0.1이고 perturbation은 node-local
  이므로 이 규칙 이상을 주장하지 않는다.

Dataset은 episode ID로 train과 validation을 나누어 temporal leakage를 막고,
validation loss가 가장 작은 checkpoint를 저장한다. PPO 학습·평가에서는 모든
parameter의 `requires_grad=False`, eval mode와 state-dict checksum을 검사한다.

## 출력 head

두 번째 relational layer가 `units=1`이고 그 layer의 `FutureFOVUnavailability`
node가 곧 출력이다(sigmoid). 별도 readout MLP, 별도 linear head, 24차원 1차원
간 residual은 없으며 `FOVRiskModel`은 `nn.Linear` module을 하나도 만들지 않는다
(rule `R-02`). 이 제약은 보상 readout에만 적용된다. PPO actor/critic의 MLP와 R-GAT
내부 관계 변환은 그대로다.

Artifact metadata는 dataset/graph version, seed, horizon, split episode ID, MAE,
RMSE, bias, R², 상수 예측기 RMSE 기준선, validation contract 위반량과 model
checksum을 기록한다. 상수 예측기 기준선은 "평균만 내는 모델"보다 나은지를 보고서가
숨기지 못하게 하려는 것이다.

## 보상

비종료 step은 `r_paper(t) - lambda * q_theta(G_{t+1})`이고 종료 보상은 원문 그대로다.
`lambda_fov=0`이면 공통 보상항·종료·합산 결과가 baseline과 동일하다. 이것은 일반적인
shaped reward이며 potential-based shaping이 아니므로 최적정책 불변 주장은 따라오지
않는다.

## 실행 경로

기본 artifact 경로는 결과 디렉터리 아래의 `rgat/fov_risk_rollouts.npz`와
`rgat/fov_risk_model.pt`다. 기본 설정의 `lambda_fov`는 0.1이며, 학습·평가 중
`freeze_during_ppo=true` 계약에 따라 model checksum이 변하지 않아야 한다. Target이
한쪽 regime만 담고 있으면(전부 0이거나 전부 양수) runner는
`--rgat-max-data-episodes`까지 실제 rollout을 추가한다. 빠른 세미나 실행은 40
episode에서 시작해 최대 120 episode로 제한한다.

## 은퇴한 readout

`shin_se_onto_rgat_fov`는 goal embedding 위에 `nn.Linear` head를 두고 "H 안에 FOV를
잃는가"의 이진 라벨을 학습하던 이전 방법이다. 해당 ID는 기존 결과의 귀속을 위해
`LEGACY_PIPELINES`에 남아 있고, 현재 방법으로 alias되지 않으며 primary 실험에서
실행할 수 없다.
