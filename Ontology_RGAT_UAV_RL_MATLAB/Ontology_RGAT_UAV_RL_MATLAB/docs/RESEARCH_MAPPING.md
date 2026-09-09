# Mapping from the uploaded landing paper to this follow-up simulator

## What comes directly from the paper

The uploaded paper frames autonomous landing under disturbance with three semantic domains:

1. Drone state: position, attitude, vertical motion.
2. Wind state/risk: drag-related effect plus time-varying wind acceleration and direction change.
3. Landing-pad observation: marker detection, center error and detection stability.

It then derives semantic quantities such as wind risk, alignment and visual stability, combines them with sensor features, and uses a lightweight classifier to decide Land/Hold. The conclusion explicitly identifies reinforcement-learning state/reward design as a future extension.

## What this project changes

### Paper
`Sensors -> semantic ontology features -> Gaussian Naive Bayes -> Land/Hold`

### Follow-up code
`6-DOF distributed physics -> ontology graph -> R-GAT potential -> PBRS -> PPO -> continuous landing control`

The ontology is therefore no longer only a feature-engineering layer. It becomes a structural prior over allowable semantic relations for reward learning.

## Equation/code correspondence

- Wind drag / distributed load: `src/+aero/distributedAero.m`
- Wind acceleration and direction change: `src/+semantic/computeFeatures.m`
- Semantic risk encoding: `src/+semantic/computeFeatures.m`
- Ontology entities/relations: `src/+semantic/buildOntologyGraph.m`
- Learned relation importance: `src/+rgat/relationLayer.m`
- Goal potential: `src/+rgat/forward.m`
- Potential-based reward shaping: `src/+reward/proposedPBRS.m`
- RL control policy: `src/+training/trainPPO.m`

## Important scientific distinction

The paper does not provide enough data to identify a full physical 6-DOF aerodynamic plant, motor coefficients, surface geometry, or turbulence spectrum. Those parts in this repository are an explicit engineering extension and are parameterized for later identification. They must not be presented as values reported by the paper.
