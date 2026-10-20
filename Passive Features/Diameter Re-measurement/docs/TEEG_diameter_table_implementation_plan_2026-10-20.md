# Implementation plan: the diameter correction table under D-052 to D-067 (2026-10-20)

| Date | Change |
|---|---|
| 2026-10-20 | Created at the user's request after the decisions of 2026-10-20 (D-061 to D-065, D-067; D-066 withdrawn). Replaces the staged plan of the status report of 2026-10-19 and the block plan of `docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` §6 where they differ (the estimator, the coordinates, the design size, the order of operations). Code facts at `84ea91f` **[src]**; the suites and hashes of 2026-10-19 **[run]**; decisions read from the log after the D-067 write **[KB]**. To be committed under `Passive Features/Diameter Re-measurement/docs/` with Block 13 (D-020). |

Basis tags: **[src]** read in the repository at `84ea91f`; **[run]** executed in the assistant's sandbox (date given);
**[KB]** the decision log or SPEC; **[user]** the user's decision; **[open]** not decided; **[reasoning]** inference.

## 0. What is being built, in one paragraph

A measurement of every dendrite node's diameter on Allen's 63x stacks (Block 13: the pilot, the Allen-guided plane
iteration, the skewed generalized-$t$ fit; D-052, D-061, D-064), a Monte Carlo of rendered cylinders measured by
exactly that chain ($G \approx 10^5$ draws x $n_{\rm real}$ = 10 realizations, D-059, D-062), a table of the chain's
bias factor $\hat b(x \mid \mathcal C, \mathcal S)$ over $x = (\ln d, \varphi, \ln(\hat\mu\hat d))$ estimated by local linear
kernel regression on a grid with its SE, scatter, failure rate and support (D-063, D-067.3), and the inversion that
corrects each real node at its own $(\varphi_i, \hat\mu_i\hat d_i)$ with its flags and fill (Block 7, D5, D-030). The
renderer stays Gaussian with a calibrated $\sigma_{\rm r}(\delta)$ table (D-067.2); the calibration and the real-cell
surveys run in Colab, the production on davinci (D-061.5, D-067.1).

## 1. State at `84ea91f` (2026-10-19) **[src; run]**

Blocks 0-12 built; 24 smoke suites, 151 pass / 0 fail / 21 skip; estimator hash `1bb3585b4a4f1373`, simulation
hash `e44ac0bd220f5214`. Block 12 (the SGT fit) is coded but not wired into `measure_node`; the plane iteration lives
in `scripts/plane_differences.py`; the table code is the old two-axis one-realization design; gate 2 last passed
2026-10-06 under the old hash; no davinci job; D-052's fit has never run on a rendered phantom or on Allen's data.

## 2. Stages

Dependencies first: S1 before everything; S3 needs S1 pushed and run in Colab; S4's logic needs S3's values only
for the configuration (its code can be written in parallel with S3); S5 after S4; S6 may start on a provisional
configuration after S5; S7 waits for the final configuration (every S3 output sits inside the simulation hash,
D-067.1).

| Stage | What | Decisions | Verification / gate | Output |
|---|---|---|---|---|
| **S0** done | branch, suites, hashes verified | -- | 24/24, hashes as recorded (2026-10-19) **[run]** | this plan |
| **S1 Block 13** -- the measurement | 13.1 move `iterate_planes`, `check_iteration`, `gradient_iteration`, `refit_at_plane`, `line_profiles`, `block_background`, `far_from_dendrites`, `plane_background` into `analysis/plane_iteration.py`; `scripts/plane_differences.py` imports them. 13.2 the square builder from `measure_node`'s branch, provider and pilot, and the scene input (segment table + soma discs; the builder moved from `survey.node_planes` into `analysis/cell.py`; `run_replicate` builds a phantom's). 13.3 `measure_node` = pilot function + iteration from $d^{(0)} = 2 r_{\rm br}$ over $K_j = k_{\rm SWC} \pm 6$ (lines $\pm 1.5 d$, $\le 5$ rounds, one provider call for the block and $K_j$, one per restack) + `fit_sgt.fit_plane(...).final` at $k^{\rm it}$; flags of $\mathcal S$ on $k^{\rm it}$; `NodeResult` gains `k_iter, iter_status, n_rounds, n_restacks, w_hat_um, beta_hat, xi_hat, zeta_hat, irls_ok, at_bound, d_pilot_um, alpha_pilot`, the pilot's status and flags; `MeasureConfig` gains the iteration's fields with `# source:` tags, all in the estimator hash. 13.4 suites: the Cell 4j-4m suites compare pilot with pilot; new `test_smoke_block13.py`. 13.5 the per-realization wall time on ~10 phantoms over $d$ and $\varphi$ | D-052.3, D-057 (iii), D-061.1-3, D-064 | oracles: the moved iteration with the old fit reproduces the script byte for byte on `fixtures_cell.py` (needs the scene); phantoms and real nodes share one function object, no phantom-only switch; `scene=None` equals the own branch alone; the one-call-plus-restacks rule on a phantom whose line outgrows the square; the flags are those of $k^{\rm it}$; the hash differs from `1bb3585b4a4f1373` and `535847535e83cf7d` and old tables are refused; a rendered phantom end to end with $\hat d/d$; defaults asserted; all 25 suites | `block 13` commits; the new estimator hash; the timing that sizes S6-S7 |
| **S2 small commits** | Block 10's selection by Allen's diameter (`node_allen_diameter_max_um` 0.6, `far_field_allen_diameter_um` (0.6, 0.8), the other criteria from the pilot's columns); the D-057 (ii) removals (`sigma_fit_study_um`, `with_sigma_fit`, `d_tilde_sigma_spread_um`, Cells 8/10c, the suites' uses) | D-061.4, D-067.4, D-057 (ii) | `test_smoke_calibration`, `config`, `cell`, `invert`, `table`; SPEC Block 10 and §2.4 | commits before S3 so that Colab runs the selection |
| **S3 Colab programme** (after S1-S2 are pushed) | (a) Block 13's chain on every cell (`run_cell.py` without a table): the distributions of $\hat\alpha_i$ and $\hat\mu_i\hat d_i$ per bin of $\hat d_i$ against the phantoms' band ($\ln\mu_{\min} + \ln d$, $\ln\mu_{\max} + \ln d$), the share inside; the real-data check: $\hat w$, $\hat\kappa$ of thick nodes in focus against rendered phantoms of the same $\hat d$ (the phantoms from the sandbox). (b) cell 6 over all cells: Allen's radii -> $d_{\max}$ and the line/block lengths. (c) cell 3b over all cells: $s^*$, $\Delta z^*$ -> `jitter_xy_um`, `jitter_z_um`. (d) Block 10's calibration of $\sigma_{\rm r}(\delta)$ on the S2 selection -> the kernel table. Then `config_production.json` | D-065, D-067.1-2, D-026/D-027, D-061.5 | each survey's figures and CSV under `MyDrive/diameters/survey/`; the pass rule of (a) fixed before the run (proposal: $\ge$ 95 % of real nodes inside the band in every bin with $\ge$ 20 nodes) | four decisions to log: the $\mu$ range (confirmed or widened), $d_{\max}$ with `profile_half_um` / `block_half_um`, the jitter SDs, the kernel table; the final configuration |
| **S4 Block 14** -- the table | 14.1 $\mathcal S$: `dark` out of `SELECTION_FLAGS`; `failure_rate_ignore = ()`. 14.2 design: `n_draws`, `n_realizations`, keys `[seed, g]`, `[seed, g, r]`, `[seed, g, r, 1]`; rows with `draw_index`, `realization_index`, `radius_um`; the radius $d/2 + \epsilon$, $\epsilon \sim \mathcal N(0, 0.2^2)$ redrawn until positive, per realization, in all four places. 14.3 coordinates `table_inputs(d, phi, mu_hat, d_hat)` -> $(\ln d, \varphi, \ln(\hat\mu\hat d))$; bandwidths (0.15, 5°, 0.3) raw, `validate` requires three; the failure count at the true $\ln(\mu_g d_g)$. 14.4 the estimator: local linear kernel regression on a regular grid (binned or kd-tree kernel sums, compact support PROVISIONAL), the by-draw sandwich SE, $\hat\tau$, $N^{\rm loc}_{\rm eff}$, $\hat p_{\rm fail}$, the leave-one-out masses and their 5th percentile; `BiasTable` holds the grid and a tensor cubic interpolant; `table_io` writes them; `tps_spline` kept for small tables only. 14.5 inversion `invert_node(d_hat, phi, mu_hat, table, cfg)`, `off_support`, `alpha_out_of_domain`; `correct_nodes` passes `mu_hat_per_um`; the per-node CSV gains `mu_d_hat`. 14.6 `build_table.py` / `build_table.pbs` chunked by draw, the merge grouped by draw and refusing other `sim_hash`; the per-draw table; `configs/gate2_reduced.json` in the new schema | D-030, D-057 (i), D-058, D-059, D-060 (target and per-draw table), D-062, D-063, D-067.3 | oracles: the 3-D fixture $b = 1 + 0.06/d + 0.15\sin^2\varphi + 0.05\ln(\mu d)$ under the $G \times n_{\rm real}$ design recovered within 4 SE at every grid point of the band, the sandwich SE within 25 % of the across-table SD (60 tables); the local linear weights sum to one and annihilate a linear $b$ (1e-10); the interpolant reproduces the grid; `off_support` fires on the C5 node and not in the band's middle; every row in the failure count; a node with $\hat\alpha > 1$ corrected, not filled, `dark` set; the radius law by KS against the truncated normal and the branch carrying it; draws reproducible per $(seed, g)$ and realizations per $(seed, g, r)$; the spline-era `check_b_target.py` toy: the table recovers $\mathbb E[\hat D/d \mid \hat\mu\hat d]$, not the draw-mean relation; folds by draw where CV is used | `block 14` commits; the CV multiplier proposal settled or defaulted to 1 |
| **S5 gate 2** (sandbox) | `build_table.py run --start 0 --stop <G> --workers 2 ... configs/gate2_reduced.json`, `merge`, `end_to_end.py phantoms --d 0.5,1,2,3 --reps 8 --seed 777` (no `--alpha-range`), `evaluate --tol 0.10` | procedure §3.10 | every mean $\tilde d/d$ within 0.10 of 1; same simulator for table and test | the recorded gate under the new hash |
| **S6 davinci pilot** (`hpc-python-compat`, `hpc-git-delivery`) | byte checks; `qsub -v DIAM_DRYRUN=1`; the two-replicate probe; a pilot of ~200 realizations (20 draws x 10) on a provisional configuration | D-022, D-054 (iii), D-061.5, D-067.1 | each `.o` shows the env, the Pillow version, `task done`; the per-realization time on davinci; the failure-rate curve by region | the array sizing (draws per task, walltime); the decision on `max_failure_rate` |
| **S7 production** (final configuration only) | the array over $G \approx 10^5$ draws x 10; the merge; the table's tests on davinci: procedure §3.10 (blur consistency, $\sigma_{\rm r}(0)$, $\mu$ percentiles, heading bins, end to end with an alternative generator), the per-draw table, the coverage of the real survey against the final band | D-062.2, D-061.5 | the tests' records committed under `checks/` | `bias_table_<hash>.npz/.json` |
| **S8 apply** | `run_cell.py` with the table on every cell -> `nodes_<sp>.csv`, the corrected SWCs (D-013) -> the I_h fitter's `--swc-dir` | D-013, D-030, D5 | the share of flagged and filled nodes per cell reported; `off_support` and `high_failure` counts against the survey | the deliverable |

## 3. Cost envelope and sizing **[reasoning from KB numbers; measured in S1 and S6]**

Per realization: rendering 10-140 s under today's chain plus 1-5 SGT fits of 1.5-9 s per round, say 60-200 s;
$10^6$ realizations: $1.7$-$5.6 \times 10^4$ CPU-h, 3.5-12 days on 200 CPUs. The PBS array is cut by draws per task
from S6's measured time (12 h walltime: 200-700 realizations per task at those rates, so 1400-5000 tasks or a
longer walltime -- davinci's limits for the array size are **[open]**). Rows: about 1 GB. The table build is
$O(N)$ per grid point with the kernel's compact support; a 48 x 19 x 36 grid at $N = 10^6$ is minutes on the login
node, to be timed in S4. Storage of the table: the grid arrays, a few MB.

## 4. Decisions still to be taken, and when

| When | Decision | Proposal |
|---|---|---|
| S3 (a) | the $\mu$ range: confirm [0.3, 3] /um or widen | by the survey's share inside the band per bin |
| S3 (b) | $d_{\max}$; `profile_half_um`, `block_half_um` | $d_{\max}$ at a high percentile of Allen's radii across cells; the line holding the thickest edges plus the kernel's tails |
| S3 (c) | `jitter_xy_um`, `jitter_z_um` | the SDs of $s^*$ and $\Delta z^*$ across cells |
| S3 (d) | the kernel table $\sigma_{\rm r}(\delta)$ and its continuation | Block 10's output on the D-067.4 nodes (D-026, D-027) |
| S4 | the bandwidth multiplier $c$ on (0.15, 5°, 0.3) | CV with folds by draw over a short grid, default 1 |
| S4 | grid sizes; the kernel's truncation; the `off_support` percentile | 48 x 19 x 36; 4 bandwidths; 5 % -- PROVISIONAL |
| S6 | `max_failure_rate` (D-054 (iii)) | from the pilot's failure-rate curve by region |
| S6 | array sizing | from the measured time |
| after S4 | remove `tps_spline` or keep it for small tables | keep until Block 14's suites no longer use it |

Closed on 2026-10-20: Q7 (no comparison), item 13 (no per-point smoothing), item 14 (D-054 (iii) stands), item 15
(no sensitivity check), the P-spline comparison (not built), the renderer's kernel (Gaussian, D-066 withdrawn).

## 5. Risks and what measures them

- The real blur is not Gaussian: accepted by D-067.2; the real-data check of S3 (a) reports the residual; the table
  does not contain it.
- Thin nodes: D-052's fit collapses at $\mu$'s bound on a share of thin tubes (16-40 % at 0.6 um in the sandbox
  Monte Carlo); they fail $\mathcal S$, enter the failure rate and are filled by D5; the thin-end bias the table
  reports is the one of the survivors.
- The truncated-normal start at the thin end (mean start 0.40 um for a 0.2 um phantom): the table is built on it; a
  real thin node starts from Allen's radius instead; the survey's $\hat d$ against $2 r_{\rm A}$ shows whether the
  starts are alike.
- Block 10's circularity (the calibration nodes' own $\hat d$ read wide): selection by Allen's diameter removes the
  estimator from the selection (D-061.4); the width-diameter trade-off in the calibration remains (procedure §3.4
  step 2, not built).
- davinci: nothing has run there; the env, Pillow, the queue, the array limits are **[open]** until S6.
- Project knowledge is at its size limit (2026-10-20): the decision log's next writes may be refused; the user
  decides what to remove.

## 6. Working rules

The block protocol for every sub-block (SPEC entry first, code, verify twice, commit); `scientific-coding`, and
`hpc-python-compat` / `hpc-git-delivery` for anything that runs on davinci; the log read before and written at
every decision; commits and pushes only on the user's agreement, never to `main`, never force; `.py` pure ASCII,
`.sh`/`.pbs` LF; documents of the workstream in `docs/` on `sci/diameter-pipeline` (D-020); equations inline in the
chat.
