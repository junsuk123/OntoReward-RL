# Evaluation protocol and why it changed

## Summary

The benchmark previously trained on medium+harsh and tested on deep. Under that
protocol no model in the package could score above chance, because the
feature-to-fault relationship does not transfer between these three UrbanNav
environments. The default is now a pooled chronological split. The old protocol
is still available and still worth running - as a generalisation experiment with
a negative result, not as a way to compare architectures.

## Evidence

Logistic regression on window-mean + window-std features, 12 features, no deep
learning involved, so this measures the data rather than any model:

| protocol | train AUROC | test AUROC |
|---|---|---|
| train medium+harsh -> test deep | 0.852 | **0.457** |
| within-deep, chronological 50/50 | 0.890 | 0.859 |
| pooled 3 scenarios, chronological 70/30 | 0.797 | 0.747 |

Leave-one-scenario-out, all three folds:

| held-out scenario | test AUROC | base rate |
|---|---|---|
| medium | 0.334 | 0.282 |
| harsh | 0.439 | 0.671 |
| deep | 0.457 | 0.345 |

All three are at or below chance. This is not a shortfall in capacity, it is a
sign flip: the correlation between a feature and the fault label reverses
between environments.

| feature | medium | harsh | deep |
|---|---|---|---|
| gSpeed | +0.326 | +0.100 | **-0.150** |
| CNO_std | +0.117 | -0.015 | **-0.203** |
| low_elev_ratio | -0.051 | +0.060 | **-0.253** |
| vDOP | -0.025 | -0.142 | **+0.118** |

A model that fits medium+harsh correctly is therefore *required* to be wrong on
deep. Reporting an architecture comparison on top of that measures nothing.

## The protocol now used

`cfg.split.mode = "pooled_chronological"`

Each scenario is cut chronologically into train (60%), validation (15%) and
test (25%). Every environment appears in all three sets, and no future sample
informs a past one.

### Purge

Windows are built with stride 1 over a 30-sample history, so two windows whose
end epochs are less than 30 apart share raw samples. Cutting at a single index
would leak up to 29 shared samples across the boundary. `cfg.split.purgeWindows`
(default `cfg.window` = 30) windows are dropped on each side of every boundary,
so no evaluation window shares a single epoch with a training window.

### Class prior shift and the decision threshold

Faults cluster in time, so the chronological blocks do not share a class prior:
train 0.589, validation 0.331, test 0.421. At a fixed 0.5 threshold a model with
AUROC 0.84 was scoring F1 0.497 - the exact all-positive degenerate value -
because every probability landed above 0.5. The threshold is now selected on the
validation split by maximising F1 and applied unchanged to test
(`select_threshold.m`). Significance tests compare each model at its own
validation-selected threshold.

## Running the old protocol

```matlab
cfg = cics_config("nonstrict");
cfg.split.mode = "cross_scenario";
```

Report it as what it is: a cross-environment transfer test that fails.
