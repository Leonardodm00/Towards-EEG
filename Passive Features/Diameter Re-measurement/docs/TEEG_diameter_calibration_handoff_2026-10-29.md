# Handoff: the kernel calibration on the pilot after D-112 -- reading Cell 7's Colab results, the open points (diameter implementation chat -> next chat)

| Date | Change |
|---|---|
| 2026-10-29 | Created at the close of the diameter implementation chat (2026-10-28/29: D-101, D-105, D-108, D-109, D-112, D-113), for the chat that receives the Colab results of Cell 7 as of `2c5fef4`. Branch `sci/diameter-pipeline` at `7d407a2` (pushed). Its facts re-checked against the branch, the decision log v31, the tester's report and the sandbox's run files before writing **[run, sandbox]**. Lives only in the repository, beside the code: `Passive Features/Diameter Re-measurement/docs/` on `sci/diameter-pipeline` (D-097.2: handoffs go there). Project knowledge refused it on 2026-10-29 ("over its maximum size"; full since 2026-10-28, D-097), so it is not at `claude/` there. |

Basis tags: **[src]** read in the named file at `7d407a2`; **[run, sandbox]** executed in the assistant's sandbox
(Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1; one core per run); **[user]** said or decided by the user; **[KB]**
project knowledge (`TEEG_decisions_and_ideas_log.md`, v31); **[reasoning]** the assistant's inference, not checked;
**[open]** neither decided nor verified.

Plainly: Cell 7 calibrates the renderer's defocus kernel on the pilot cell 529878215 -- a near-field growth table
from thin dendrites and a far-field slope gamma from slightly thicker ones. Since 2026-10-29 it no longer takes its
calibration nodes from the pilot table: before downloading anything it judges every dendrite node of the cell from
Allen's trace (flatness, and a clean background "stripe" in the planes each fit uses), keeps the best ones 5 um apart,
and then measures and scans them. The user is about to run that version in Colab and bring the output to the next
chat. Nothing of D-108, D-109 or D-112 has run on Allen's images yet: the sandbox cannot reach api.brain-map.org, so
every number below comes from the pilot's trace, not from its images.

## 1. First actions in the next chat

1. Read, in this order: this file; in the decision log (project knowledge) D-112, D-113, D-109, D-108, D-105, D-101,
   then D-088 (10-30 nodes per set), D-077, D-078 and D-081 (the far fit) for context; in `specs/SPEC.md`, Block 10's
   entries 10.S (the stripe) and 10.C (the candidate search), Block 15's 15.1-15.7 and Block 19's 19.1-19.3; the
   runbook's section "Cell 7" (v38).
2. Get the branch (in an existing clone `git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only`):
   ```
   git clone --branch sci/diameter-pipeline https://github.com/Leonardodm00/Towards-EEG.git && cd Towards-EEG && git log --oneline -1
   ```
   Expect `7d407a2` or later. Then `python3 specs/roadmap/roadmap.py status --yaml specs/roadmap/diameter-pipeline.yaml`
   before answering "where are we" (the roadmap rule).
3. With the user's upload, run the checklist of section 4 in order. If something is missing, ask for it, or fetch it
   from Drive with the Google Drive connector: Cell 7 writes everything under `MyDrive/diameters/calibration/`
   (section 3).
4. Re-read the decision log before giving any new decision a number. The noise-fit chat writes to the same log in
   parallel and took D-106, D-107, D-110 and D-111 while this chat was numbering, twice in one day. The next free IDs
   at hand-off are **D-114, I-005** **[KB]**.

## 2. Where things are

| What | Where | State at hand-off |
|---|---|---|
| Code | `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline`, workstream `Passive Features/Diameter Re-measurement/` (two spaces in the path) | `7d407a2`, pushed **[src]** |
| Spec | `specs/SPEC.md` (repository root): 10.S the stripe, 10.C the candidate search, 15.x the far field, 19.x the cap and S3c | statuses current to `c393834` **[src]** |
| Decision log | `TEEG_decisions_and_ideas_log.md` (project root; copy in `docs/project/`, D-097) | v31 **[KB]** |
| Path map | `claude/TEEG_diameter_HPC_paths_reference.md` (copy in `docs/project/claude/`) | v16: section 3's Cell 7 and Cell 15 rows, section 4's open checks **[KB]** |
| Colab runbook | `<workstream>/docs/TEEG_diameter_phase2_colab_runbook_2026-10-06.md`, section "Cell 7" | v38: Cell 7's code equals the notebook's, the expected lines **[src]** |
| Notebooks | `<workstream>/notebooks/s3_run_2026-10-27.ipynb` (the user's run file; `.py` mirror) and `phase2_colab.ipynb` (`.py` mirror) | Cell 7 and the pre-flight changed at `2c5fef4`; the run file's `.ipynb` code cells equal its `.py` **[src; run, sandbox]** |
| The search's code | `<workstream>/src/allen_diameter/analysis/candidate_search.py` (`far_search.py` until `c393834`); `scripts/calibrate_kernel.py` (`cell_candidates`, `candidate_sources`, `search_name`, `not_in_set`, `knobs_line`); `scripts/survey_cells.py` `step_calibration` | **[src]** |
| Tests | `<workstream>/tests/smoke/test_smoke_candidate_search.py` (and `_stripe`, `_isolation_view`, `_s3c`); the tester's `tests/spec/` (committed `67b2262`) | 41 smoke suites 254 pass 39 skip; `tests/spec` 69 pass **[run, sandbox]** |
| Tester report | `specs/test_reports/2026-10-28_1955_incremental.md` (on `e32d03b`) | F1, F5 fixed; F2 confirmed and F3 accepted (D-113); F4 kept; S4 open **[src]** |
| Roadmap | `specs/roadmap/diameter-pipeline.yaml` and `.md`; the page "Subproject Roadmaps" | Blocks 1, 10, 15, 16, 19 "pushed": no tester report covers `c393834` **[run, sandbox: `roadmap.py status`]** |
| Pilot inputs | Drive: `MyDrive/allen_cache/H16.06.010.01.03.05.03_668702901_m.swc`, `MyDrive/diameters/pilot/pilot_529878215.csv` (Cell 4b, 2026-10-07, 393 rows) | not in the repository; the next chat's sandbox must fetch them through the Drive connector to rerun anything **[KB: path map section 1]** |
| Pilot checks of the stripe | `<workstream>/checks/d105_2026-10-28/` (README: the tau sweep, the residual on the line) | **[src]** |
| This handoff | repository only: `<workstream>/docs/TEEG_diameter_calibration_handoff_2026-10-29.md` (project knowledge full) | the earlier diameter handoffs and the runbook are in the same `docs/` |

## 3. What Cell 7 does now

`calibrate_kernel.py real`, then `fit`, `far-fit` and `figures`, every step given
`OUT/calibration/calibration_config.json` (the cell's `CAL_KNOBS`). A failed `real` stops the cell. A failed `fit`
(fewer than two near nodes with at least 3 finite widths: `fit_growth`'s ValueError) skips `far-fit` and `figures`. A
failed `far-fit` still runs `figures`, without the far fit. Every step is called with `check=False`, so the notebook
goes on **[src: Cell 7 of `s3_run_2026-10-27.py`; `calibration.fit_growth`]**.

| Stage | Rule | Source |
|---|---|---|
| Candidate search (Block 10.C) | per set with `<set>_candidates = "cell"`: every dendrite node in the set's range of Allen's diameter (near 2r <= 0.6 um, far 0.6-0.8 um), judged from the trace -- tilt over +-2 um <= 10 deg (near) / 15 deg (far); the stripe at the trace's plane round(z / 0.28 um) usable (>= 500 clear px, >= 3 um each side); near: >= 3 usable of k_tr +- 3 (pairs +-1, +-2, +-3 in turn, early stop); far: >= 6 usable of the 24 planes 5 <= \|k - k_tr\| <= 16; the pilot table's medians in S as B_bar and mu_hat; kept 5 um apart (near: most pixels first; far: most usable planes first) | D-109, D-112 **[src]** |
| Scan order and cap (19.1) | seeded order (`calib_order_seed` with the specimen); a set stops at 30 scanned nodes (`calib_max_nodes_per_set`); the rest `set_full`, counted in one line | D-088.3 **[src]** |
| Each candidate (Blocks 5, 10.I, 10.S) | the pilot measured (`measure_pilot`, D-061.2), then S (flags, `faint` not a criterion), flatness by the pilot's phi_hat, the stripe at the pilot's k* (`not_isolated:too_few` / `:side_short`); straightness and optical depth not tested | D-093, D-094, D-096, D-101 **[src]** |
| Scans (10.P, 15.1, 15.2) | a +-25 um patch per plane, the background a plane fitted on each plane's stripe; near: planes k* +- 3, in each a Gaussian fitted to the dip by least squares (its width omega); far: planes k* +- 16, the dip's core fitted iteratively (at most 10 passes; `not_converged` left out) | D-077, D-078, D-079, D-101 **[src]** |
| `fit` (Block 10's growth fit, 10.A) | near growth G(delta) (needs >= 3 finite widths per node, >= 2 nodes), the table cut at 0.84 um, the anchor sigma_0_hat and its bootstrap SE; INSUFFICIENT below 10 nodes | D-080.1, D-081.7, D-088.3 **[src]** |
| `far-fit` (15.3-15.6) | planes with \|k - k*\| >= 5 passing the predicted-dip SNR rule (3 x RMS), nodes with >= 6 such planes; IRLS fit of gamma with the anchor fixed; bootstrap SE; per-sign and quadratic comparisons; a production kernel only with >= 10 nodes in both sets | D-078.3, D-081, D-088.3 **[src]** |

The knobs (`CAL_KNOBS`, all at the configuration's defaults) **[src: the notebooks at `2c5fef4`]**:

| Knob | Value | Decision |
|---|---|---|
| `background_support` | "stripe" | D-101 |
| `isolation_depth_gl` (tau) | 0.3 gl, PROVISIONAL | D-108 |
| `stripe_half_widths_um` | 1, 2, 4, 8, 12 | D-101.3 (PROVISIONAL) |
| `stripe_min_pixels`, `stripe_min_side_um` | 500, 3.0 um | D-101.4 (PROVISIONAL) |
| `patch_half_um` | 25 um | D-079 |
| `node_phi_max_deg`, `node_phi_max_deg_far` | 10, 15 deg | D-101.1 |
| `near_candidates`, `near_search_spacing_um` | "cell", 5 um (PROVISIONAL) | D-112 |
| `far_candidates`, `far_search_spacing_um` | "cell", 5 um (PROVISIONAL) | D-109 |

Outputs under `OUT/calibration/` (`OUT` = `MyDrive/diameters`) **[src; KB: path map section 3]**:
`calibration_config.json`; `near_search_529878215.csv` and `far_search_529878215.csv` (one row per node of the set's
range: `set`, `node_id`, `allen_diameter_um`, `stretch`, `s_um`, `x_um`, `y_um`, `z_um`, `k_trace`, `theta_rad`,
`phi_deg`, `reason`, `n_pixels_k`, `n_fit_usable`, `n_fit`, `chosen` -- `reason` empty for a node that passed, else
`not_flat`, `not_isolated:too_few` or `not_isolated:side_short` (at the trace's plane) or `few_planes`; `chosen`
`True`/`False`); `scans_529878215.json` (every candidate's reasons in `meta.nodes`); `patches_529878215.npz`;
`calibration_529878215.json`; `far_fit_529878215.json`; `figures/` -- from `figures`: `width_by_plane_near.png`,
`width_by_plane_far.png`, `example_near.png`, `example_far.png`, `kernel_law.png`, `gaussians/<set>_node_<id>.png`;
from `real`: `isolation/<set>_node_<id>.png` (panels (a)-(c); (c) every plane's stripe) and
`isolation/isolation_overview_<n>.png` (36 per page).

## 4. Reading the uploaded results: the checklist

| # | Check | Pass condition | If not |
|---|---|---|---|
| 1 | The code Colab ran | Cell 1 printed `[diameter] commit <hash> on sci/diameter-pipeline; python ...` with `<hash>` at or after `2c5fef4` (`git merge-base --is-ancestor 2c5fef4 <hash> && echo ok` in a clone) | the clone was not updated, or the notebook's Cell 7 is an older copy: the run tells nothing about D-112 |
| 2 | The knobs | `calibration knobs: background_support stripe, tau 0.3 gl, W 1, 2, 4, 8, 12 um, at least 500 px and 3 um per side, patch +-25 um; flatness phi <= 10 deg near, 15 deg far; near set from the whole cell, at least 5 um apart; far set from the whole cell, at least 5 um apart` | the knob block differs from the delivered cell |
| 3 | The search's inputs | `candidate search: the cell's typical background 159.0 gl and absorption 0.258 /um (the pilot table's medians in S)` | another pilot table was read (check `PILOT_CSV`) |
| 4 | The near search | after `near search: n of 7500 near-range nodes judged (t min)` every 500 nodes: `near search over the cell: 7500 near-range nodes; 3856 flat by the trace (phi <= 10 deg); 1960 with a usable stripe at the trace's plane; 1947 with at least 3 usable planes of the trace's plane +- 3; 380 chosen, at least 5 um apart (t min) -> .../calibration/near_search_529878215.csv` | the search is deterministic given the SWC, the table, the transform (pilot: no shift, no flip, z0 0) and the configuration: a different count means one of them differs; the CSV can be compared row by row with the sandbox's (section 7) |
| 5 | The far search | `far search over the cell: 723 far-range nodes; 502 flat by the trace (phi <= 15 deg); 90 with a usable stripe at the trace's plane; 74 with at least 6 usable far-fit planes; 19 chosen, at least 5 um apart (t min) -> .../calibration/far_search_529878215.csv`, then `399 candidate nodes` | as for 4 |
| 6 | The scan | `n of 399 candidates scanned: a near-field, b far-field (at most 30 each, D-088.3); reasons {...}`, and `not measured, their set already holding 30 scans (set_full): ...` | read the reasons: S flags, `not_flat` (the pilot's phi_hat), `not_isolated:*` at the pilot's k* -- the pilot's k* and fitted centre differ from the trace's |
| 6b | The isolation lines | at the end of `real`: `isolation figures: N candidates (k whose stripe at k* is not usable: fewer than 500 px or a side under 3.0 um) -> ...; overview ...` (N = the candidates measured; `set_full` ones get no figure) and `far-set candidates with a usable stripe in at least 6 of the far fit's planes (\|k - k*\| >= 5): m of n` (at most 19) | many unusable stripes at the pilot's k*: compare the pilot's k* and centre with the search CSV's `k_trace`, `x_um`, `y_um`; judge tau on panel (c) |
| 7 | Sufficiency | near a >= 10 and far b >= 10 for a production kernel; below, `fit` prints `INSUFFICIENT` and `far-fit` `no production kernel: insufficient ...` | the fits are diagnostics; options in section 6 |
| 8 | The near fit | `growth fit (...): N nodes`, the knot table, `residual RMS by \|k - k*\|`, `anchor sigma_0_hat = ... um, SE ... um` | fewer than two near nodes stop `fit` |
| 9 | The far fit | `far planes: J nodes kept ([planes per node]), d dropped ...; P planes; ... not converged, ... below the SNR rule`, `far fit: gamma_hat = ... +- ...`, the t interval, per-sign and quadratic comparisons, `production kernel: ... slope ...` or the reason none | the outermost planes are the weak point: see 10 |
| 10 | The figures | `width_by_plane_near/far.png` (every node grey, the median, the fitted law), `kernel_law.png`, the examples, the isolation overview pages | judge tau = 0.3 on panel (c); look for planes at \|k - k*\| near 16 bending away from the straight line |

Time in Colab **[reasoning]**: the two searches took 20.5 min in the sandbox before the first download; then each
measured near candidate costs its pilot's z-block (about 13 crops) and 7 patch crops, each far one about 46 crops;
Cell 11b downloaded 579 crops, with everything else it does, in 7.0 min in Colab **[user's log of 2026-10-28, read;
path map section 3]**, so expect roughly an hour in all -- more if many near candidates fail their pilot before
30 pass, since every one measured costs its crops.

What the far fit is likely to show **[run, sandbox: the model's shadows on the pilot's trace, not Allen's grey
levels]**: for the 19 far candidates, the neighbours' summed shadow left on the +-10 um line after the plane is fitted
on the stripe is small near focus and comparable to the dip at the outermost planes:

| \|k - k*\| | Residual, median | Own dip, median | Ratio, median |
|---|---|---|---|
| 5 (1.4 um) | 0.94 gl | 4.99 gl | 0.21 (15 of 18 below half the dip) |
| 16 (4.48 um) | 1.89 gl | 1.60 gl | 1.25 (7 of 18 below the dip) |

(Per node, the plane at k_tr + o if its stripe is usable, else at k_tr - o; at each offset 18 of the 19 have one, and
the medians are over those 18. The residual is the largest error on the line; the trace's plane k_tr stands in for
k* **[run, sandbox: `farsearch/pipe/chosen_residuals.py`, not kept in the repository; D-109's entry]**.)

The pilot's median background noise SD is 5.9 gl **[KB: D-101]**. Block 15.3's SNR rule uses the predicted dip, not
the residual, so the outer planes can pass it and still bias the width: compare `width_by_plane_far.png` at large
\|k - k*\| with the fitted line, and the per-sign fit.

## 5. Decisions in force for Cell 7

| ID | Rule | Status |
|---|---|---|
| D-061.4, D-067.4 | the calibration sets by Allen's diameter: near 2r <= 0.6 um, far 0.6-0.8 um (PROVISIONAL thresholds) | active |
| D-077, D-078, D-081 | the far field: k* +- 16, the iterative core fit, the planes' SNR rule (D-078.3), the sequential IRLS fit of gamma with the anchor fixed, the bootstrap SE (D-081.7: 1000 resamples) | active |
| D-088.3 | 10 to 30 scanned nodes per set, a seeded order; below 10 the fits are diagnostics | active [user] |
| D-093, D-094, D-096 | no straightness test, no optical-depth or `faint` criterion; S without `faint` | active [user] |
| D-101 | flatness per set (10 / 15 deg); the background on the stripe; the node's verdict at k* | active; its W grid, 500 px, 3 um, own radius rule PROVISIONAL |
| D-105 | panel (c), `CAL_KNOBS` in Cell 7, the recheck (F1 fixed) | active |
| D-108 | tau = 0.3 gl, PROVISIONAL, to judge on panel (c) | active [user: "As you recommend"] |
| D-109 | the far set's candidates from the whole cell's trace | active [user] |
| D-112 | both sets from the whole cell, in Cell 7, in Cell 15 and on davinci; the near rule and spacing PROVISIONAL | active [user: "the whole cell's nodes must be evaluated for both near and far field calibration"] |
| D-113 | the own cylinder = the union of the trace's and the fitted one (F2); F3's tilt error accepted, no change | active [user: "I confirm the union and the F3 without change"] |

The noise-fit chat's IDs interleave with these (D-095, D-097, D-100, D-102 to D-104, D-106, D-107, D-110, D-111).
They bind its own workstream, not this one, except D-097 (repository copies of project documents) **[KB]**.

## 6. Open points

For the user:
- **Cell 15 repeats Cell 7's search.** S3's calibration step searches the pilot again before its first download
  (another 20 min in the sandbox). The assistant asked on 2026-10-29 whether Cell 15 should reuse Cell 7's
  `<set>_search_529878215.csv`: **unanswered**. No code reuses them today **[src]**.
- **The kernel slope.** Cell 8's `renderer.kernel_continuation_slope` (in `phase2_colab.ipynb`) is the user's to set
  from `far fit: gamma_hat` once the far fit is sufficient; the configured 0.79 stays until then (D-024 (i), D-070)
  **[src: Cell 8's text; KB]**.
- **A set below 10 scanned nodes.** Options include: more cells through S3c on davinci (the per-cell quota is 5 scans
  per set, D-089); or rethink a rule. tau and the stripe's minima are knobs, the near/far ranges PROVISIONAL. The
  choice is the user's.
- **tau = 0.3 gl** (D-108): to judge on panel (c) and on the near fit's residuals.
- **S4** (the tester's): the far set's minimum side run, 3 um or the far line's 10 um.

The assistant's PROVISIONAL readings, to confirm on the results (D-112, D-109, D-101):
- the near walk's early stop once 3 planes are usable; the near ranking by the stripe's pixels at the trace's plane;
- the 5 um spacing of both sets;
- the pilot table's medians as every node's B_bar and mu_hat in the search;
- the trace's plane in place of k*;
- the trace's tilt as a pre-screen at the set's bound;
- S3's step searching every cell on davinci.

For implementation, not started:
- an integration-tester run on `c393834` and later (the roadmap shows the blocks "pushed", untested since `e32d03b`).
  The user starts it ("run the night test").
- Cell 15's reuse of Cell 7's search, if the user wants it.

## 7. Evidence produced by this chat

| Claim | Number | Basis |
|---|---|---|
| The pilot table's own candidates | 138 near, 7 far (the far ones on two stretches; none had a usable stripe in any far plane at 0.03 gl, none 6 usable planes at 0.3 gl) | **[run, sandbox: `checks/d105_2026-10-28/`]** |
| tau 0.03 vs 0.3 gl, the table's near candidates usable at k* | 21 vs 92 of 138 | **[run, sandbox: idem]** |
| The far search on the pilot | 723 / 502 / 90 / 74 / 19 chosen on 7 stretches; their path distance from the soma median 294 um (34-361) against 24 um for the table's far rows (16 of 31 within 50 um) | **[run, sandbox]** |
| The near search on the pilot | 7500 / 3856 / 1960 / 1947 / 380 chosen on 48 stretches; the walk stopped after 3 planes for 1879, 5 for 60, 7 for 8; of the table's 138 near candidates 71 pass, 12 chosen | **[run, sandbox]** |
| The searches' time | near 15.4 min, far 5.1 min (8.2 before the pre-cull), 20.5 min in all, one core | **[run, sandbox]** |
| The pre-cull in `shadow_masks` is exact | 60 of 60 random cases identical (1, 3, 33 planes; tau 0.03, 0.3, 3 gl; predicted and geometric margins), 3.2 -> 2.2 s per case; the far search's 723 records identical node for node | **[run, sandbox]** |
| The tests | 41 smoke suites 254 pass 39 skip; `tests/spec` 69 pass; 13 mutants of D-112's code and 11 of D-109's each killed | **[run, sandbox]** |
| The hashes | estimator `59678d4e2aedf2b6`, simulation `52aa73e479eb77cf` (no calibration field in a signature) | **[run, sandbox: a fresh clone of `2c5fef4`]** |
| F2 and F3 (the tester's phantoms) | F2: the trace 0.4 um off -> omega_hat -0.9 %, 0.6 um -> `not_isolated:side_short`; F3: tilt read up to 4.5 deg off on 0.2 um nodes; the along-branch depth smear 0.458 tan(phi) stays under half a plane up to 17.0 deg | **[run, sandbox: the tester's report; the arithmetic]** |

To reproduce the two searches on the pilot (about 20 min; the SWC and the CSV fetched from Drive first, into a
directory of their own; PATH and OUT_DIR to fill in):
```
cd "Passive Features/Diameter Re-measurement" && python3 -I -c "
import sys; sys.path[:0] = ['src', 'scripts']
import calibrate_kernel as CK
from allen_diameter.config import default_config
from allen_diameter.loading import swc_io, table_io
cfg = default_config(); print(CK.knobs_line(cfg))
c = CK.cell_candidates(swc_io.read_swc('PATH/H16.06.010.01.03.05.03_668702901_m.swc'), cfg,
                       dict(shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0),
                       table_io.read_rows(['PATH/pilot_529878215.csv']), 'OUT_DIR', 529878215, log=print)
print(len(c), 'candidate nodes')"
```

## 8. Not checked

- Nothing of D-108, D-109, D-112 has run on Allen's images (Colab is next). Open: how many of the 380 near and 19 far
  candidates pass their pilot, whether each set reaches 10, the far fit at large defocus **[open]**.
- The search's and the scans' time in Colab; Cell 15's repeated search **[open]**.
- davinci: the survey jobs (section 2b of the path map) have run only in the sandbox, against stubs; the array's
  calibration step now searches each cell (the pilot's 20.5 min in the sandbox), its time on davinci **[open]**.
- The residuals of section 4 are the model's predicted shadows (each neighbour as a tube with the cell's median
  B_bar and mu_hat, the renderer's blur law), not grey levels **[run, sandbox; the model's limits: reasoning]**.

## 9. Working rules learned in this chat

- **Decision IDs**: re-read the project log right before writing it. Number a decision only at that moment, and merge
  any change, never overwrite. Collisions happened twice on 2026-10-29: D-106/107 became D-108/109, D-110/111 became
  D-112/113, and unpushed commits were rewritten to carry the right IDs.
- **Document upkeep**: every project document written goes to project knowledge and, byte for byte, to `docs/project/`
  on `sci/diameter-pipeline` with a row in `docs/project/BRANCH_COPIES.md` (D-097, open point (d)); a handoff goes
  to `<workstream>/docs/` instead (D-097.2). Project knowledge refuses new documents (full: 2026-10-28, and this
  handoff on 2026-10-29); updates of its existing documents went through on 2026-10-29 (the log v31, the path map
  v16). A document it refuses lives only in the repository, and its changelog says so. Each reply ends with the line
  naming the documents written.
- **Pushing**: push verified work to `sci/diameter-pipeline` without asking (D-097.1, the user, 2026-10-28); never
  force-push, never to `main`; commit messages end with the session's attribution lines.
- **Notebooks**: edit the `.py` mirror and the `.ipynb` together. The run file's `.ipynb` code cells must equal its
  `.py`, and the runbook's Cell 7 code block must equal `phase2_colab.ipynb`'s. `test_smoke_isolation_view.py` loads
  the notebooks' knob block and requires the configuration's defaults.
- **Configuration**: `test_smoke_config.py` requires every field to be read by name somewhere; a field read only
  through `getattr(cal, name)` fails it (hence `calibrate_kernel.candidate_sources`).
- **Scripted edits**: LaTeX in a non-raw Python string turns `\t`, `\b`, `\r` into control bytes. A `\"` inside a raw
  string stays a backslash and a quote. After every scripted edit of a `.md`, scan the bytes for CR, BS, TAB and `\"`.
  `.py`, `.sh`, `.pbs` files stay pure ASCII with LF line endings.
- **Long runs**: the sandbox's commands stop at 2 min. Run suites and searches in the background with `nohup` and a
  log, polling with `sleep` of at most about 95 s. Suite times on two cores: `far_field` 10 min, `variant_iteration`
  5.5, `block13` 6, `calibration` 3, `candidate_search` 2-3; all 41 in about 35 min with two runners; `tests/spec`
  (pytest) 5 min.
- **The sandbox's reach**: api.brain-map.org is blocked (403). Image steps run only in Colab; the sandbox can run
  trace-based analyses on the pilot's SWC and table fetched from Drive.
