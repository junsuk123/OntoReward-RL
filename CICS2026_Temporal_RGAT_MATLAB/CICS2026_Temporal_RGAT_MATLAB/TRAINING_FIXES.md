# Why nothing was learning, and what changed

Five independent defects. Each one alone was enough to flatten the results; the
first two produced the symptom that every Dual-EDL curve sat at probability 0.5.

---

## 1. The EDL regulariser penalised correct evidence

`model_gradients.m`

The Dirichlet KL was applied to the full `alpha`:

```matlab
a = softplus_stable(raw)+1;
kl = dirichlet_kl_uniform(a);         % penalises ALL evidence
loss = ce + cfg.edl.klHardWeight*kl;  % weight 1.0, no annealing on the hard head
```

Standard EDL (Sensoy et al. 2018) applies the KL to the *misleading* evidence
only, `alpha_tilde = y + (1-y).*alpha`, so evidence for the correct class is
free. Without the mask the regulariser fights the data term and wins:

| evidence e | p_fault | CE | KL (unmasked) | CE+KL |
|---|---|---|---|---|
| 1 | 0.667 | 0.406 | 0.193 | **0.599** |
| 2 | 0.750 | 0.288 | 0.432 | 0.720 |
| 5 | 0.857 | 0.154 | 0.958 | 1.113 |
| 20 | 0.955 | 0.047 | 2.092 | 2.139 |

`CE+KL` is minimised at *low* evidence, so the optimum is `alpha -> 1`,
`p -> 0.5`, `u -> 1`. That is exactly what the old figures showed: every
Dual-EDL trace pinned inside 0.45-0.60 while the loss curve looked like it was
converging.

With the mask, KL is 0 whenever all evidence sits on the correct class, and
learning proceeds. The data term is now the proper EDL expected cross-entropy
`sum_j T_j (psi(S) - psi(alpha_j))` rather than CE on the mean probability, and
the anneal `lambda = min(1, epoch/annealEpochs)` is applied to the hard head too
(previously only the weak head was annealed).

## 2. The Temporal R-GAT lost 46x of its signal to un-normalised pooling

`build_model.m` (`rgat_head`), `build_transformer_network.m`

`TemporalRGATLayer` ends in a mean over the 12 feature nodes;
`globalAveragePooling1dLayer` then takes a mean over the 30 time steps. Two
un-normalised means in series crushed the input-dependent part of the
representation. Measured at initialisation:

| | pooled feature sd across windows |
|---|---|
| Transformer (LayerNorm in every encoder block) | 0.53 |
| R-GAT, as written | **0.012** |
| R-GAT + LayerNorm | 0.558 |

Every window produced almost identical logits - the hard-head probability spanned
only `[0.4984, 0.5023]` - so the model emitted a single class for its entire run
(val F1 frozen at the degenerate 0.4974, val AUROC 0.33-0.43, *below* chance).
The Transformer never hit this because each encoder block ends in a
`layerNormalizationLayer` that restores unit scale.

Adding one `layerNormalizationLayer` after the graph trunk: val AUROC over 10
epochs went **0.36 -> 0.84**.

## 3. No validation set and no model selection

`train_model.m`, `make_splits.m`

Training reported whatever the last epoch happened to be. With stride-1
overlapping windows the plain Transformer drove its training loss to 0.01 while
sitting at chance on held-out data. There is now a validation split, per-epoch
validation AUROC, best-epoch selection and early stopping
(`cfg.train.patience`). Selection uses AUROC rather than validation loss so the
choice is independent of the EDL anneal, which is still ramping while the model
is being selected.

Observed peaks are early - typically epoch 3-10 - so `cfg.train.epochs` dropped
from 50 to 30 as a wall-clock cap; early stopping is what actually ends a run.

## 4. Overlapping windows leaked across every split boundary

`make_splits.m`

Windows are built with stride 1 over a 30-sample history, so two windows whose
end epochs are less than 30 apart share raw samples. Any chronological cut at a
single index leaks up to 29 shared samples. `cfg.split.purgeWindows` (default
30) now drops windows on each side of every boundary, so no evaluation window
shares a single epoch with a training window.

## 5. A fixed 0.5 threshold under a shifting class prior

`select_threshold.m`

Faults cluster in time, so the chronological blocks do not share a class prior:
train 0.589, val 0.331, test 0.421. A model with AUROC 0.84 was scoring F1 0.497
- the exact all-positive degenerate value - because every probability landed
above 0.5. The threshold is now chosen on **validation only** by maximising F1
and applied unchanged to test. `paired_bootstrap_compare` and `mcnemar_exact`
accept `[thrBase thrNew]` so each model is compared at its own operating point.

---

## What this does not fix

Two problems are in the data, not the code, and no training change touches them.
Both are now reported at the top of every run instead of being silently absorbed
into the results table.

- **The evaluation protocol.** Train-on-medium+harsh, test-on-deep is below
  chance in all three leave-one-scenario-out folds. See `SPLIT_PROTOCOL.md`.
- **The weak label.** The `PR_RMS` surrogate scores AUROC 0.471 as a direct
  predictor of `hard_label`, so the weak EDL head can only fit noise and the
  dual fusion underperforms the hard-only head. `report_label_quality.m` prints
  this and warns. See `weak_label_surrogate.m` for the two available sources and
  why neither supports a reproduction claim.
