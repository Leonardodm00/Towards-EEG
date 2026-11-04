# Checks of 2026-11-04 (D-118.3, D-119)

Standalone checks by computation (not pipeline code), run in the sandbox of the diameter calibration chat.
Run them from the workstream folder `Passive Features/Diameter Re-measurement`.

| Script | What it checks | Output |
|---|---|---|
| `cluster_bootstrap_check.py` | the node, cell (one level), two-level and "larger of node and cell" bootstraps of an average over nodes nested in cells (30 nodes per cell; 2 to 51 cells; ICC 0, 0.05, 0.2; 400 datasets, 300 resamples; seed 20261104): SE estimate / true SE and the coverage of +-1.96 SE. The evidence put to the user for D-118.3 | `cluster_bootstrap_check.out` |
| `sparse_growth_check.py` | Block 10's growth fit with the sparse Jacobian (LSMR) against the dense fit on the same noisy synthetic scans (7 planes, 10 % NaN; N = 10 to 300; symmetric cubic and linear, min_origin), and the time per fit at N = 300, 1000, 1530 (D-119) | `sparse_growth_check.out` |

Both are a stand-in, not the real data: the first replaces the kernel fit by a grand mean (the kernel's own ICC is
unknown); the second uses a quadratic growth and Gaussian noise.

```
python3 checks/d118_2026-11-04/cluster_bootstrap_check.py      # about 15 s
python3 checks/d118_2026-11-04/sparse_growth_check.py          # about 50 s
```
