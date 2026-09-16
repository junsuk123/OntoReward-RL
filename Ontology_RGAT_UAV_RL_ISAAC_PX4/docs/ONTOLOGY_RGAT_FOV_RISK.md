# 제안 알고리즘: Ontology-R-GAT 미래 FOV 비가용성 보상

[문서 안내](README.md) · [시스템 개요](SYSTEM_OVERVIEW.md) ·
[실험 설계](TWO_PIPELINE_COMPARISON.md)

계약 원본은 `config/ontology/fov_recovery.yaml`이다. 이 문서와 코드가 어긋나면
계약 파일과 `tests/test_two_pipeline_fov.py`가 기준이다.

## 1. 착안

이동 플랫폼 착륙에서 시야 상실은 사후 복구가 어렵다. 기존 shaping은 *이미 발생한*
추정오차(`r_active`)나 *현재* 상태만 본다. 그래서 패드가 화면 중앙에 있는 순간의
행동이 1초 뒤 프레임 이탈을 만들어도 그 순간에는 벌점이 없다.

제안항은 이 시간 간극을 직접 메운다. 시각 이력만으로 구성한 graph에서 **가까운
미래의 기하 FOV 비가용 시간 비율**을 예측하고, 그 예측값에 비례하는 벌점을
행동이 일어난 step에 부과한다.

## 2. 입력 경계

Graph 입력은 아래 10개의 유계 시각 특징뿐이다.

| 특징 / Node | 출처 |
|---|---|
| `KeypointConfidence` | 현재 프레임의 heatmap 엔트로피와 점별 존재도 |
| `VisibleKeypointFraction` | 가시성 임계값을 넘는 keypoint 수의 비율 |
| `FOVMargin` | `clip(1 - max(|c_x|, |c_y|), 0, 1)`, 마지막으로 신뢰 가능했던 정규화 centroid |
| `ApparentTargetScale` | 신뢰도 가중 keypoint 분산의 RMS |
| `ImagePlaneMotion` | `1 - image_plane_motion_safety` (프레임 간 centroid 변화율) |
| `ScaleRate` | `1 - scale_rate_safety` (프레임 간 겉보기 크기 변화율) |
| `VisibilityMemory` | 측정 신뢰도의 지수감쇠 이력 |
| `ReacquisitionTrend` | 직전 프레임 대비 keypoint 신뢰도 변화 |
| `MeasurementValidity` | 현재 프레임이 사용 가능한 패드 기하를 냈으면 1, 아니면 0 |
| `MeasurementAge` | 마지막 유효 측정 이후의 유계 경과시간 |

뒤의 두 개는 같은 keypoint 이력에서 나오는 missingness와 age이며 새로운 privileged
입력이 아니다. Simulator ground truth, 실제/추정 상대 위치·속도, 6-D 상대상태 추정,
critic-only state는 함수 인터페이스와 dataset schema에서 거부된다. **기하 패드 중심
가시성 자체도 차단된다** — 그것이 학습 label이므로 입력이 되면 추론이 조회가 된다.

### 결측은 여유가 아니다 (규칙 `R-03`)

`FOVMargin`은 *마지막으로 신뢰 가능했던* centroid의 여유이므로 측정이 없는 동안
값이 정지한다. 패드를 화면 중앙에서 놓친 직후 raw margin은 1.0에 가깝게 읽히고,
이는 가시성이 가장 나쁜 순간이다. 따라서 margin은 항상 `MeasurementValidity`와
함께 공개되고, 파생 node `BoundarySafety`의 seed는

$$
\text{BoundarySafety}_{\text{seed}} = \text{fov\_margin} \times \text{measurement\_validity}
$$

로 gate된다. 동일한 stale margin이라도 validity가 다르면 graph도 예측도 달라져야
한다는 것을 test가 강제한다.

## 3. Graph

15개 node는 위의 입력 10개와 파생 4개(`PerceptionQuality`, `BoundarySafety`,
`TargetMotion`, `VisualObservability`), 출력 1개(`FutureFOVUnavailability`)다.

```
KeypointConfidence      --indicates-->  PerceptionQuality
VisibleKeypointFraction --indicates-->  PerceptionQuality
MeasurementValidity     --indicates-->  PerceptionQuality
FOVMargin               --indicates-->  BoundarySafety
ApparentTargetScale     --constrains--> BoundarySafety
MeasurementValidity     --constrains--> BoundarySafety
MeasurementAge          --constrains--> BoundarySafety
ImagePlaneMotion        --indicates-->  TargetMotion
ScaleRate               --indicates-->  TargetMotion
VisibilityMemory        --supports-->   VisualObservability
ReacquisitionTrend      --supports-->   VisualObservability
MeasurementAge          --constrains--> VisualObservability
PerceptionQuality       --supports-->   FutureFOVUnavailability
BoundarySafety          --supports-->   FutureFOVUnavailability
VisualObservability     --supports-->   FutureFOVUnavailability
TargetMotion            --constrains--> FutureFOVUnavailability
```

선언 edge 16개에 node별 self-loop 15개가 더해진다. Edge와 self-loop의 순서가
고정되어 있으므로 동일 입력은 byte-identical graph를 만든다.

Node 특징 행렬은 `X ∈ R^{19×15}`이며 행 구성은 값, 여값 `1-v`, 입력 node 지시자,
파생/출력 node 지시자, 그리고 15-D one-hot 항등행렬이다.

관계 이름은 relational kernel이 쓰는 **구조적 edge type**이고 강제되는 단조성
제약이 아니다. 어떤 부호도 이름에서 읽어서는 안 된다.
`rgat.fov_graph.unreachable_input_nodes()`는 출력까지 non-self 경로가 없는 입력
node를 반환하며 규칙 `R-01`에 따라 항상 비어 있어야 한다. 도달할 수 없는 입력은
schema가 무시하면서 선언만 하는 장식이기 때문이다.

## 4. 예측 표적

실제 제어 주파수를 `f`라 하면 1.0초 horizon은 `H = round(1.0 f)` step이다
(10 Hz → 10 step). 표적은 이진 라벨이 아니라 **시간 비율**이다.

$$
y_t=\frac{1}{H}\sum_{k=1}^{H}\bigl(1-V_{\text{centre}}(t+k)\bigr)\in[0,1]
$$

`V_centre`는 `isaac_sim/keypoint_geometry.geometric_pad_center_in_fov` 하나,
즉 패드 중심이 양의 depth로 정규화 image 좌표 `[-1,1]^2`에 투영되는가다.
검출 성공률이나 학습된 keypoint 신뢰도를 표적으로 삼으면 모델은 *검출기* 고장을
예측하게 되며, 이는 제안 보상이 억제하려는 양과 다른 양이다.

모델 출력은 `E[y_t | G_t]`이며 binary loss 확률이 아니다.

`H` step 미래가 관측되지 않은 episode 꼬리 sample은 `valid=False`, target `NaN`으로
mask한다(규칙 `R-05`). 0으로 채우면 "모든 episode의 끝은 안전하다"를 학습시키게
되고, episode를 이어 붙이면 존재하지 않는 미래를 만들어낸다.

## 5. 모델과 readout

$$
h = \tanh\bigl(\text{RGAT}_1(X)\bigr)\in\mathbb R^{15\times 24},\qquad
q_\theta = \sigma\Bigl(\text{RGAT}_2(h)\bigl[\text{FutureFOVUnavailability}\bigr]\Bigr)
$$

두 layer 모두 `wirgat` dot-product attention, relation 차원 6, head 1, head
aggregation `mean`, attention units 8이다. 두 번째 layer는 `units=1`이고 그 layer의
`FutureFOVUnavailability` node가 곧 출력이다.

별도 readout MLP, 별도 linear head, 24차원에서 1차원으로 가는 residual은 없으며
`FOVRiskModel`은 `nn.Linear` module을 하나도 만들지 않는다(규칙 `R-02`). 이것이
"graph 자체가 보상 신호를 만든다"는 주장을 구조로 강제한다. 이 제약은 보상 readout
에만 적용되며 PPO actor/critic의 MLP와 R-GAT 내부 관계 변환은 그대로다.

## 6. 오프라인 학습

목적함수는 두 항이다.

$$
\mathcal L=\mathrm{Huber}_{\delta=0.1}\bigl(q_\theta(X),\,y\bigr)
+w_c\Bigl[q_\theta(X)-q_\theta\bigl(X^{+\Delta}\bigr)\Bigr]_{+}
$$

* **회귀항**: 지도 가능한(unmasked) sample만 사용.
* **규약항 (`R-04`)**: `MeasurementAge` node 값만 `Δ = 0.25`만큼 올렸을 때 예측
  비가용성이 *감소*하면 벌하는 one-sided hinge. 기본 가중치 `w_c = 0.1`.
  더 오래된 마지막 관측은 패드가 아직 보인다는 더 약한 증거이므로, 경과시간을
  안심 신호로 쓰는 모델은 신호를 뒤집는 것이다. 섭동은 node-local이며 파생 node
  seed를 다시 계산하지 않으므로 이 규칙 이상을 주장하지 않는다.

| 설정 | 기본값 |
|---|---|
| optimizer / lr | Adam / 5e-4 |
| epochs / batch | 80 / 64 |
| validation 비율 | 0.2 (episode ID 기준) |
| hidden / relation / heads | 24 / 6 / 1 |
| grad clip | 5.0 |
| checkpoint 선택 | validation loss 최소 |

Dataset을 episode ID로 나누어 시간 누출을 막고, 학습 후 train/validation episode
집합의 교집합이 비어 있는지 다시 확인한다.

보고 지표는 MAE, RMSE, bias, R²와 **상수 예측기 RMSE 기준선**이다. 기준선을 함께
기록하는 이유는 "평균만 내는 모델"보다 나은지를 보고서가 숨기지 못하게 하기
위해서다. validation 규약 위반량도 함께 저장한다.

## 7. 동결

PPO 학습과 평가 동안 모든 parameter는 `requires_grad=False`, 모델은 eval mode이고,
state-dict SHA-256 checksum이 매 호출마다 검증된다. R-GAT은 PPO optimizer에
포함되지 않는다. Artifact metadata에는 dataset/graph version, seed, horizon(초·step),
split episode ID, MAE/RMSE/bias/R², 상수 예측기 기준선, validation 규약 위반량,
model checksum이 기록되고 loader가 graph version·node/relation schema·architecture
선언·checksum을 모두 대조한다.

## 8. 보상

$$
r_{\text{proposed}}(t)=r_{\text{Shin}}(t)-\lambda_{\text{fov}}\,q_\theta\!\left(G_t\right),
\qquad \lambda_{\text{fov}}=0.1
$$

`G_t`는 행동이 선택된 상태 `t`의 graph이고, 그 예측 대상 `y_t`는 구간
`t+1 … t+H`를 덮는다. 즉 벌점은 *그 행동이 만들 미래*에 부과된다.

구현상 가산항은 dispatch 뒤에 더해지므로 종료 전이를 포함한 모든 transition에
적용된다. `λ_fov = 0`이면 공통 보상항, 종료 보상, 합산 결과가 baseline과 수치적으로
동일하며 이를 test가 강제한다.

이것은 일반적인 shaped reward이며 potential-based shaping이 아니다. 최적정책
불변성은 주장하지 않는다.

## 9. 실행 경로와 artifact

기본 artifact 경로는 결과 디렉터리 아래다.

| 파일 | 내용 |
|---|---|
| `rgat/fov_risk_rollouts.npz` | graph `X`, 표적 `y`, valid mask, episode/seed/time metadata |
| `rgat/fov_risk_model.pt` | 동결 validation-best R-GAT, metadata, checksum |
| `rgat/fov_risk_training_history.csv` | epoch별 train/validation huber·contract |

데이터는 baseline(`fov_risk_design.source_pipeline: shin_se_fixed`)의 실제 rollout
에서 수집한다. 표적이 한쪽 regime만 담고 있으면(전부 0이거나 전부 양수) runner는
`--rgat-max-data-episodes`까지 실제 rollout을 추가한다. 세미나 프로파일은 40
episode에서 시작해 최대 120 episode로 제한한다.

## 10. 은퇴한 readout

`shin_se_onto_rgat_fov`는 goal embedding 위에 `nn.Linear` head를 두고 "H 안에 FOV를
잃는가"라는 이진 라벨을 학습하던 이전 방법이다. 해당 ID는 기존 결과의 귀속을 위해
`LEGACY_PIPELINES`에 원래 이름으로 남아 있고, 현재 방법으로 alias되지 않으며
`validate_pipeline_configuration`이 주 실험에서의 실행을 거부한다.

Graph version은 `ontology_rgat.future_fov_unavailability/2`, dataset format과
model format도 `/2` 계열이다. 이전 artifact는 로드되지 않는다.

## 11. 주장하지 않는 것

계약 파일의 `not_claimed` 항목을 그대로 옮긴다.

* potential-based shaping 또는 최적정책 불변성
* 운용 안전 보장
* attention 가중치의 인과적 증거성
* 관계 중요도가 직접 지도된 학습 표적이라는 주장
* 방향성 주장 (규칙 `R-06`) — graph는 부호 없는 image-plane 운동 크기만 담으므로
  "경계 쪽으로 흐르는 중"과 "안쪽으로 흐르는 중"을 구분할 수 없다.
