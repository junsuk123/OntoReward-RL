# legacy_matlab — reference only, not part of the pipeline

This is the MATLAB implementation the Python learner under [`../python/`](../python/)
replaced. It is kept so the ported numbers can be checked against the code they
came from, and for nothing else:

- **Nothing here runs.** `run_pipeline.m` and `run_external_all.m` still call
  `sim.*`, `training.*` and `evaluation.*`, and they still expect the read-only
  original workspace on the path via `setup_external_path.m`. None of that is
  maintained, and `scripts/check_workspace.sh` fails the build if anything that
  does run starts referencing this directory.
- **The Python port is the experiment.** `python/run_pipeline.py` runs the same
  seven stages: connectivity, dataset, R-GAT, both PPO arms, paired evaluation
  and figures.

## Where each file went

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
| `src/+pipeline/runAll.m` | `ontology_rgat/pipeline.py` |
| `tests/test_rgat_equivalence.m` | `../tests/test_rgat_equivalence.py` |
| `tools/benchmark_rgat.m` | `../python/run_benchmark.py` |

## What deliberately did not survive

- **`+rgat/relationLayer.m`'s `dlarray` edge loop.** The PyTorch layer in
  `ontology_rgat/rgat/layers.py` is checked against a plain per-edge reference
  implementation, the same way this one was, and is a strict generalisation of
  it (see NOTICE).
- **The legacy panel-aero visualisation contract.** `sim.stateToModel` used to
  spread Isaac's resultant aerodynamic force uniformly over the retired MATLAB
  panel slots purely so `viz.RealtimeMonitor` could draw arrows at them. Nothing
  read those numbers as physics, and nothing draws them now.
- **`src/+rgat/{device,toGPU,toCPU}.m`.** The device decision is
  `ontology_rgat/rgat/train.py:select_device`, and the crossover it uses is
  re-measured by `python/run_benchmark.py`.
