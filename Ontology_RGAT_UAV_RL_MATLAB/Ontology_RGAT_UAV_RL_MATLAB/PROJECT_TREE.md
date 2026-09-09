# Project tree

```text
Ontology_RGAT_UAV_RL_MATLAB/
├── config/
│   └── defaultConfig.m
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── MODEL_ASSUMPTIONS.md
│   ├── RESEARCH_MAPPING.md
│   └── THEORY.md
├── results/
├── src/
│   ├── +aero/
│   │   ├── airProperties.m
│   │   ├── distributedAero.m
│   │   └── makeSurfacePanels.m
│   ├── +control/
│   │   ├── actionToRotorCmd.m
│   │   └── expertController.m
│   ├── +dynamics/
│   │   ├── diagnostics.m
│   │   ├── eom.m
│   │   └── rk4Step.m
│   ├── +evaluation/
│   │   ├── comparePolicies.m
│   │   ├── evaluatePolicy.m
│   │   ├── makePlots.m
│   │   └── windSweep.m
│   ├── +mathx/
│   │   ├── eulerToQuat.m
│   │   ├── quatNormalize.m
│   │   ├── quatToEulerZYX.m
│   │   ├── quatToRotm.m
│   │   ├── skew.m
│   │   └── wrapPi.m
│   ├── +prop/
│   │   ├── mixWrenchToOmega.m
│   │   └── rotorForces.m
│   ├── +reward/
│   │   ├── manualDense.m
│   │   ├── proposedPBRS.m
│   │   └── sparseTask.m
│   ├── +rgat/
│   │   ├── explain.m
│   │   ├── forward.m
│   │   ├── initModel.m
│   │   ├── predict.m
│   │   └── relationLayer.m
│   ├── +semantic/
│   │   ├── buildOntologyGraph.m
│   │   └── computeFeatures.m
│   ├── +sensor/
│   │   └── observe.m
│   ├── +sim/
│   │   ├── getCurrent.m
│   │   ├── makeObservation.m
│   │   ├── resetState.m
│   │   ├── runEpisode.m
│   │   ├── step.m
│   │   └── terminalStatus.m
│   ├── +training/
│   │   ├── actorForward.m
│   │   ├── adamStep.m
│   │   ├── computeGAE.m
│   │   ├── criticForward.m
│   │   ├── generateRGATDataset.m
│   │   ├── initPPO.m
│   │   ├── policyAction.m
│   │   ├── ppoActorGradients.m
│   │   ├── ppoCriticGradients.m
│   │   ├── rgatGradients.m
│   │   ├── samplePolicy.m
│   │   ├── trainPPO.m
│   │   └── trainRGAT.m
│   ├── +viz/
│   │   ├── plotSurfaceLoads.m
│   │   ├── RealtimeMonitor.m
│   │   └── TrainingMonitor.m
│   └── +wind/
│       └── sampleField.m
├── tests/
│   └── test_basic.m
├── validation/
│   └── validatePhysics.m
├── PROJECT_TREE.md
├── README.md
├── run_all.m
├── run_quick_smoke_test.m
├── run_realtime_demo.m
├── setup_path.m
└── STATIC_VALIDATION.md
```

> `results/` is generated at runtime and intentionally starts empty.