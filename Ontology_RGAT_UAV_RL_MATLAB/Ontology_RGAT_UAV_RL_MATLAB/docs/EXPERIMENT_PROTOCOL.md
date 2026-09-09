# Experiment protocol

## Baseline
`Manual`: PPO trained with a fixed dense reward whose weights are explicitly stored in `cfg.reward.manual`.

## Proposed
`Ontology-RGAT`: R-GAT potential pretrained from terminal safe/unsafe outcomes, then PPO trained with sparse task reward + potential-based shaping.

## Fair evaluation
- identical initial-condition and wind seeds (common random numbers),
- deterministic policy mean during evaluation,
- reward-independent landing criteria,
- paired differences and approximate 95% confidence intervals,
- save per-episode CSV, summary CSV, MAT model files and publication plots.

## Recommended paper ablations
1. sparse task reward only,
2. manual dense reward,
3. hand-crafted fixed potential PBRS,
4. GAT without relation types,
5. R-GAT without ontology edge constraints (fully connected),
6. proposed ontology-constrained R-GAT PBRS,
7. relation masking: remove WindRisk->TouchdownSafety / Alignment->SafeLanding, etc.

## Generalization tests
- wind intensity outside training range,
- unseen gust locations/timing,
- mass/inertia perturbation,
- sensor-noise increase / marker dropout,
- changed ground-effect coefficient,
- different initial lateral offsets and yaw.
