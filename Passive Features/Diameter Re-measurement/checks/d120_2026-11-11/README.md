# Checks of 2026-11-11 (D-120.3, Block 19.8)

Standalone checks by computation (not pipeline code), run in the sandbox of the diameter calibration chat.
Run them from the workstream folder `Passive Features/Diameter Re-measurement`.

| Script | What it checks | Output |
|---|---|---|
| `outlier_rule_check.py` | Block 19.8's rule (`cell_outliers.assess_set` at the defaults: Normal, Bonferroni 0.05 / C, from 10 cells) and a variant (the variance of the reference mean added to the denominator, Student's t on C - 2 df) on simulated cells (between-cell SD 0.01 um, within 0.02 um; 400 datasets per case; seed 20261111): with no cell shifted, how often any cell is removed (10, 13, 51 cells; 2, 5, 30 nodes per cell); with 1, 2, 3 or 5 cells shifted by 0.2 um, how often exactly those are removed (13 and 51 cells, 5 nodes) | `outlier_rule_check.out` |

A stand-in, not the real data: the node values are Gaussian with a known split of variance, where the real ones are
mean residuals from the pooled law, whose spread between and within cells is unknown until the davinci pool runs.

```
python3 checks/d120_2026-11-11/outlier_rule_check.py      # about 4 min
```
