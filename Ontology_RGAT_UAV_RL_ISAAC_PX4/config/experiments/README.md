# Experiment status

## Primary

`two_axis_context_rgat_comparison.yaml` is the only current primary scientific
contract. It compares canonical-vector PPO, information-matched semantic-flat
PPO, and typed ontology R-GAT PPO with exactly two learned acceleration
commands.

## Legacy and reproduction

Every other YAML file in this directory predates the two-axis contract. They
remain versioned for provenance, figures, live-stack reproduction, or explicit
ablation. In particular, these mechanisms are not part of the primary method:

- a learned tilt/pitch action (the old planar action had three channels);
- additive FOV-risk reward or ontology-derived reward weights;
- PBRS/state-potential shaping;
- frozen-base selective R-GAT and behavior-cloning warm starts;
- the Shin five-term/active-perception reward comparisons.

Legacy checkpoints are intentionally incompatible with algorithm version
`two-axis-context-rgat-v1`, causal packet v2, and the two-channel action hash.
