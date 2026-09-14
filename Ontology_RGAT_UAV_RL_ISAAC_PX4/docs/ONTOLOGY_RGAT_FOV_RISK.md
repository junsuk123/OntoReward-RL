# Ontology-R-GAT 미래 FOV-loss 보상

## 입력 경계

Graph 입력은 `KeypointConfidence`, `VisibleKeypointFraction`, `FOVMargin`,
`ApparentTargetScale`, `ImagePlaneMotion`, `ScaleRate`, `VisibilityMemory`,
`ReacquisitionTrend`의 8개 bounded visual feature뿐이다.

정규화 centroid `(cx, cy)`에 대해
`FOVMargin = clip(1 - max(abs(cx), abs(cy)), 0, 1)`이다. Simulator ground truth,
true/estimated relative position·velocity, 6-D relative-state estimate와 critic-only state는
함수 인터페이스와 dataset schema에서 거부된다. 기존 estimator는 별도의 baseline
구성으로 정상 동작한다.

## Graph

13개 node는 8개 입력과 `PerceptionQuality`, `BoundarySafety`, `TargetMotion`,
`VisualObservability`, `FOVRetention`이다. 관계는 `indicates`, `supports`, `constrains`,
`self`이며 `FOVRetention`이 classifier goal node다. Edge와 self-loop 순서는 고정되어
동일 입력은 byte-identical graph를 만든다.

## Label과 학습

실제 control frequency를 `f`라 하면 1.0초 horizon은 `round(1.0*f)` step으로 변환한다.
현재 이후 horizon 안에서 target이 unavailable 또는 visibility criterion 미달이면
`y_t=1`, 아니면 0이다.

R-GAT의 유일한 학습 목적은 binary cross entropy다. Dataset은 episode ID로 train과
validation을 나누어 temporal leakage를 막고, validation BCE가 가장 작은 checkpoint를
저장한다. PPO 학습·평가에서는 모든 parameter의 `requires_grad=False`, eval mode와
state-dict checksum을 검사한다.

Artifact metadata는 dataset/graph version, seed, horizon, split episode ID, AUROC, F1,
precision, recall, confusion matrix와 model checksum을 기록한다.

