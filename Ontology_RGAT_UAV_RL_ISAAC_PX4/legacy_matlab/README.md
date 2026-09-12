# legacy_matlab — 참고 전용, 실행 pipeline 아님

[기본 프로젝트 README](../README.md) · [문서 안내](../docs/README.md)

이 디렉터리는 [`../python/`](../python/)의 Python learner가 대체한 MATLAB
구현이다. Port한 수치의 출처 코드를 대조하기 위한 용도로만 보존한다.

- **이 안의 코드는 실행되지 않는다.** `run_pipeline.m`과 `run_external_all.m`은
  여전히 `sim.*`, `training.*`, `evaluation.*`을 호출하고
  `setup_external_path.m`을 통해 read-only 원본 workspace가 path에 있다고
  가정한다. 이 환경은 유지관리하지 않는다. 실제 실행 코드가 이 디렉터리를
  참조하면 `scripts/check_workspace.sh`가 build를 실패시킨다.
- **실험 구현은 Python이다.** 저장소 루트 `../../run.sh`는 기본
  `shin_se / no_se / onto_no_se` 비교에 `python/run_three_pipeline.py`를 선택한다.
  `python/run_pipeline.py`에는 이전 cooperative two-arm 실험이 남아 있다.
  어느 쪽도 이 디렉터리를 import하지 않는다.

## 파일별 이전 위치

| MATLAB | Python |
|---|---|
| `defaultExternalConfig.m`, `config/defaultConfig.m` | `ontology_rgat/config.py` |
| `src/+bridge/PX4Bridge.m` | `ontology_rgat/bridge.py` |
| `src/+sim/*.m` | `ontology_rgat/env.py` |
| `src/+semantic/*.m` | `ontology_rgat/semantic.py` |
| `src/+reward/*.m` | `ontology_rgat/rewards.py` |
| `src/+control/expertController.m` | `ontology_rgat/expert.py` |
| `src/+rgat/*.m`, `src/+training/{trainRGAT,rgatGradients,generateRGATDataset}.m` | `ontology_rgat/rgat/` |
| `src/+training/{initPPO,trainPPO,computeGAE,...}.m` | `ontology_rgat/ppo/` |
| `src/+evaluation/*.m` | `ontology_rgat/evaluation/` |
| `src/+viz/*.m` | `ontology_rgat/viz/` + RViz 2 + Isaac overlay |
| `src/+stack/ExternalStack.m` | `ontology_rgat/stack.py` |
| `src/+pipeline/runAll.m` | legacy cooperative `ontology_rgat/pipeline.py`; 기본 orchestration은 `python/run_three_pipeline.py` |
| `tests/test_rgat_equivalence.m` | `../tests/test_rgat_equivalence.py` |
| `tools/benchmark_rgat.m` | `../python/run_benchmark.py` |

## 의도적으로 이식하지 않은 항목

- **`+rgat/relationLayer.m`의 `dlarray` edge loop:**
  `ontology_rgat/rgat/layers.py`의 PyTorch layer는 이 구현과 같은 방식으로 평범한
  per-edge reference 구현과 비교 검증하며, 이를 엄격하게 일반화한다(NOTICE 참조).
- **Legacy panel-aero 시각화 계약:** `sim.stateToModel`은 사용 종료된 MATLAB panel
  slot에 Isaac의 resultant aerodynamic force를 균등 분배해
  `viz.RealtimeMonitor`가 화살표를 그리게 했다. 이 값을 physics로 읽은 코드는
  없었고 현재는 그리지 않는다.
- **`src/+rgat/{device,toGPU,toCPU}.m`:** device 선택은
  `ontology_rgat/rgat/train.py:select_device`가 담당하며 기준점은
  `python/run_benchmark.py`로 다시 측정한다.

기본 direct semantic R-GAT은 MATLAB 14-node/fixed-weight design을 옮긴 것이 아니다.
별도의 18-node/35-edge history-aware estimator-free 실험이며
[`docs/THREE_PIPELINE_COMPARISON.md`](../docs/THREE_PIPELINE_COMPARISON.md)에 설명한다.
