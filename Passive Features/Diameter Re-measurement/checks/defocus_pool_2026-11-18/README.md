# Checks for the pooled defocus table on davinci (2026-11-18, D-170, D-175)

Standalone checks of the defocus-table chat, run in the sandbox at `152b988` (`sci/diameter-pipeline`; the code
is that of `1fb63c4`). Run each from the workstream folder (`Passive Features/Diameter Re-measurement`). Pure
ASCII. They change no pipeline code.

## 1. How long the pool job's calibration takes at 158 cells (`pool_timing.py`, `split_timing.py`)

Question: does `scripts/pbs/survey_pool.pbs` (1 CPU, 8 GB, walltime 2 h in its header) finish the pooled
calibration (`survey_cells.pooled_fits`: Block 10's near fit and its anchor bootstrap, Block 15's far fit, its
bootstrap and anchor sensitivity, Block 19.8's cell test) when the pool holds all of Allen's 158 cells?

Set-up: synthetic pooled scans made with the smoke suite's generators (`tests/smoke/test_smoke_s3c.py`): per cell
30 near scans (`near_cells`: 7 planes, the cell's own growth scale ~ N(1, 0.1^2), noise SD 0.003 um^2) and 2 far
scans (`far_scan`: exact, 33 planes, the cell's slope ~ N(0.40, 0.05^2), sigma0 = the near law's anchor); the pool
job's own path (`calibration_pool.pool_scans`, then `pooled_fits` with figures off) at two numbers B of bootstrap
resamples, the time at B = 1000 extrapolated (the cost is affine in B).

    python3 checks/defocus_pool_2026-11-18/pool_timing.py --cells 51 --near 30 --far 2 --b 5 15
    python3 checks/defocus_pool_2026-11-18/pool_timing.py --cells 158 --near 30 --far 2 --b 3 8
    python3 checks/defocus_pool_2026-11-18/split_timing.py

Results: `pool_timing_51.out`, `pool_timing_158.out`, `split_timing.out`. (At 51 cells Block 19.8 removed one
synthetic cell from the far set -- the exact far scans have no within-cell scatter -- so that time includes the
second pass; at 158 cells it removed none, so that time is one pass.) The far fit dominates: its dense
Jacobian (Block 15.4; D-119 gave the near fit a sparse one and left the far fit dense) makes one far fit grow from
about 1 s at 102 far nodes to about 18 s at 316, so 1000 cell resamples take about 5 h per pass at 158 cells, and
a cell removed by Block 19.8 runs both fits again (about 10 h). The pool job therefore needs a longer walltime than
its header's 2 h (D-175.3: 48 h). Limits: synthetic scans, 2 far nodes per cell (the pilot's yield), the sandbox's
CPU; davinci's time per fit is not known.

## 2. Block 10's near fit on phantoms of three thicknesses (`thick_near_check.py`)

Question (the davinci handoff, section 6): does Eq. 4's growth fit recover the kernel's growth G(delta) when the
near-set nodes are as thick as the pilot's clear "thin" nodes were fitted (0.77-1.09 um), rather than 0.15-0.25 um?

Set-up: the renderer's kernel set to the pilot's table (sigma_r = 0.080, 0.1536, 0.3924, 0.5620 um at 0, 0.28,
0.56, 0.84 um, slope 0.79): that is the truth. Per range of true diameters, 16 flat phantoms drawn by
`calibrate_kernel.py synthetic --set near` (log d uniform on the range, alpha 0.15-0.4, tilt < 2 deg, the default
camera chain and absorption `partition_vertical`), measured, scanned on k* +- 3 and fitted by `calibrate_kernel.py
fit`. Two departures from production, to save time or to let the phantoms in: the background patch is +-8 um
(`calibration.patch_half_um`, 25 um in production; one isolated tube per patch), and for the thick range the near
set's Allen-diameter bound is raised to 1.0 um (`k_wide.json`), since a phantom's Allen diameter is its true d.

    python3 checks/defocus_pool_2026-11-18/thick_near_check.py --n 16 --workers 1

Result (`thick_near_check.out`; the knot at 0.84 um is the line `anchor`):

| true d (um) | sigma_r(0.28) | sigma_r(0.56) | sigma_r(0.84), its SE |
|---|---|---|---|
| truth | 0.1536 | 0.3924 | 0.5620 |
| 0.15-0.25 | 0.1296 (-15.6 %) | 0.3754 (-4.3 %) | 0.5025 (-10.6 %), 0.0294 |
| 0.45-0.55 | 0.1063 (-30.8 %) | 0.3371 (-14.1 %) | 0.5359 (-4.6 %), 0.0052 |
| 0.75-0.95 | 0.1717 (+11.8 %) | 0.2978 (-24.1 %) | 0.5181 (-7.8 %), 0.0121 |

Reading [reasoning]: the first stage's table is biased under the pipeline's own forward model, by 5-30 % in
sigma_r at the knots, and the bias changes with the nodes' thickness; SPEC section 8 records the mechanism (the
core width of a mixture of defocus values) on thin phantoms under the default table. Pooling cells reduces the
table's noise, not this bias; the procedure's step 2 (NS1, not built) is what removes it. Limits: one seed, 16
phantoms per range (the thin range's anchor is 2 SE from the truth), flat tubes, one absorption model, the smaller
patch; the per-node constants c_i come out negative on some thin phantoms (symmetric convention), so they give no
bound on sigma_r(0).
