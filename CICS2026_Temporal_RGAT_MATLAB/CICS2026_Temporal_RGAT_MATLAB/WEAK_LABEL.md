# What the `pos_error` weak label is, and what it is not

`cfg.weakLabel.source = "pos_error"` (current default for non-strict runs)

```
soft_fault_prob = sigmoid( (pos_error_2d - 3.0 m) / tau ),   tau = 1.0 m
hard_label      = pos_error_2d >= 3.0 m
```

## What it legitimately adds

The hard label throws away everything except which side of 3 m the error fell
on. A 3.1 m error and a 30 m error are the same label. The soft label keeps the
**magnitude**, so the weak head is trained on how bad the error was, not just
whether it crossed the line. That is a real and useful auxiliary signal, and it
is almost certainly what the prior study did: their SoftLabel baseline - which
uses the soft label directly as the prediction, with no model at all - was
reported at F1 0.959. A label that alone scores 0.959 against the hard label is
derived from the same ground-truth error.

Describe it as **auxiliary regression on ground-truth error magnitude**, or as
**label smoothing over the error threshold**.

## What it is not

It is **not** independent weak supervision. Both labels come from the same
ground-truth trajectory. Consequences that must not be misreported:

1. **The `SoftLabel` baseline now scores F1 1.000 by construction.** At
   `tau = 1.0` the sign of `soft - 0.5` is identical to `hard_label`, for every
   epoch. That row is a consistency check, not a competitive baseline. Do not
   put it in a results table as if a model had to beat it.

2. **`report_label_quality` prints AUROC 1.000 and raises
   `cics:circularWeakLabelQuality`.** That warning is correct and must stay in
   the log. It is the reader's signal that the weak head's supervision is not
   independent evidence.

3. **No Dual-EDL result under this setting supports the claim that the weak head
   adds information from a cheap or independent source.** It supports the
   narrower claim that supervising on error magnitude alongside the binary label
   helps, which is a label-design result, not an architecture result.

4. **The reproduction gate is still meaningless.** It compares against a
   baseline trained on the prior study's own estimator, which this package does
   not have. `cfg.mode="nonstrict"` already warns about this.

## The alternative, and why it was rejected

`cfg.weakLabel.source = "pr_rms"` uses the surrogate shipped with the public
reconstruction, `sigmoid(PR_RMS/10 - 1)`. It is independent of the ground truth,
but it is uninformative: as a direct predictor of `hard_label` it scores
**AUROC 0.471**, below chance. Measured effect on the pooled chronological
split - the weak head degrades the fused decision:

| model | test AUROC |
|---|---|
| TransEDL-Hard (hard head only) | 0.760 |
| Transformer-DualEDL (hard + weak fusion) | 0.661 |

Rescaling cannot rescue it: AUROC is invariant to monotone transforms, and
`PR_RMS` itself is rank-uninformative about the fault label (per-scenario
correlation +0.06 / -0.04 / +0.07).

So the two options fail in opposite directions - one is independent but carries
no signal, the other carries signal but is not independent. Neither supports a
reproduction claim against the reported F1 = 0.950. Recovering that claim needs
the prior study's actual residual-to-probability estimator.
