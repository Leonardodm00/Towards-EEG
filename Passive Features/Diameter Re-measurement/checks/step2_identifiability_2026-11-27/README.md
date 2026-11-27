# Step 2 (NS1): identifiability probe and line-renderer timing, 2026-11-27

Standalone checks of the step-2 implementation chat, not pipeline code. Run from
`Passive Features/Diameter Re-measurement/` with `PYTHONPATH=src`.

| File | What it does |
|---|---|
| `identifiability_probe.py` | Finite-difference Jacobian of the near core widths (and the dip's area) of 12 synthetic twins with respect to the shared table (ln sigma_r(0), ln theta_1..3) and each twin's axis depth and ln mu; correlations, condition numbers, SE ratios; first-order bias for twins 10 % too wide. Configuration in `CFG` (seed 20261127). |
| `identifiability_probe_output.txt` | Its output at `aadac6a`. |
| `line_twin_timing.py` | Time of one twin through the fast line renderer and `dip_statistics` (run 2026-11-24 at `78806af`). |

Limits: the forward model is the fast line renderer (Block 35, hybrid, 8 x 16 directions), noise-free, flat
tubes, the table read linearly between knots. It is a proxy for the full stack that step 2 uses (D-270), and
the numbers hold for the configuration run. Used in D-271 and D-273 of the decision log.
