# Handoff: the defocus kernel table built on davinci from all the eligible cells (D-170) (clearance chat -> next chat)

| Date | Change |
|---|---|
| 2026-11-18 | Created at the user's request (17:38, D-170) for the chat that starts the construction of the defocus table on davinci. Branch `sci/diameter-pipeline` at `1fb63c4` (code) and the commit that adds this file; decision log on `main` at `30ed363` (next free IDs D-171, I-006). This chat did not run anything on davinci and did not change the calibration code. |

Basis tags: **[user]** said or decided by the user; **[log]** the decision log on `main`
(`docs/project/TEEG_decisions_and_ideas_log.md`); **[src]** read today in the named file at `1fb63c4`;
**[run, sandbox]** executed in the handing-over chat's sandbox; **[data]** from files the user uploaded;
**[reasoning]** the assistant's inference; **[open]** neither decided nor verified; **[not read]** exists, not opened
by this chat.

## 1. First actions in the next chat

1. Read this file; in the decision log D-085, D-088, D-099, D-117 to D-121, D-165, D-170 and the changelog rows of
   2026-11-18 (read the last rows again before taking an ID: two other chats log there).
2. Find out whether the survey has already run on davinci (section 4): ask the user for
   `<survey>/status.csv` or `results_<RUN_TAG>.tgz`. This chat does not know.
3. Settle section 7 with the user before changing code or submitting jobs.
4. For anything shipped to the cluster apply the skills `hpc-python-compat` and `hpc-git-delivery`; job scripts
   of roadmap stages keep their marker lines (the `subproject-roadmap` skill).

## 2. What the user asked (D-170)

"FInally write an handoff for another chat that will start the construction of the defocusing table on the Da Vinci
with all the eligible cells. So that we avoid using a single noisy table" **[user, 2026-11-18 17:38]**.

The "single table" in use: the pilot cell 529878215's own calibration (notebook Cell 7), kernel widths 0.080,
0.1536, 0.3924, 0.5620 um at defocus 0, 0.28, 0.56, 0.84 um, continued beyond with the configured slope 0.79 (a
diagnostic far fit on two nodes of the pilot gave 0.397) **[log; the pipeline theory document of 2026-11-17,
section 3.2]**. Every fit of the cleaning and clearance chats used it.

## 3. What the table is, and where it is used

| Quantity | Config field | How it is obtained today **[src: `analysis/calibration.py`, `scripts/calibrate_kernel.py`]** |
|---|---|---|
| Growth of the kernel's variance with defocus, $G(\delta)$ (um$^2$) at knots $\delta_m = m \cdot 0.28$ um | -- | `fit` (the near field, SPEC Block 10, Eq. 4): over the near-set nodes, minimise $\sum_{i,k}[c_i + G(z_{{\rm ax},i} - z_k) - \omega_{i,k}]^2$, $\omega_{i,k}$ the squared width (um$^2$) of a Gaussian fitted to the core of node $i$'s dip in plane $k$, $c_i$ a constant per node, $z_{{\rm ax},i}$ the node's axis depth. |
| Table $\sigma_{\rm r}(\delta_m) = [\sigma_{\rm r}(0)^2 + G(\delta_m)]^{1/2}$ (um), cut at 0.84 um | `renderer.kernel_table_delta_um`, `kernel_table_sigma_um` | `calibration.kernel_from_growth`. |
| In-focus width $\sigma_{\rm r}(0)$ (um) | `renderer.sigma_r0_um` = 0.080 | **Configured.** `config.py`: "never identified by the stacks"; `kernel_from_growth`: "it is not identified by the scan". |
| Slope beyond the table (um of width per um of defocus) | `renderer.kernel_continuation_slope` | `far-fit` (Block 15) on the far-set nodes; written as a production kernel only when both sets hold at least `calib_min_nodes_per_set` = 10 nodes. |

Consumers: the renderers (`model/kernel.py::sigma_r`), hence every fit. The Colab notebook's `cleaning_config`
(Cell 16) reads `OUT/calibration/calibration_<S>.json` (key `kernel`: the three fields above) and
`far_fit_<S>.json` (`kernel.kernel_continuation_slope`) **[src: `notebooks/cleaning_pilot_2026-11-14.py`]**. Since
`f6504b0`, `kernel.with_in_focus_width` and `clean_cell.py --sigma-r0-um` re-anchor a table at another in-focus
width, keeping $G$.

## 4. What exists for davinci (S3 across cells; SPEC Blocks 16 to 19; D-099)

Three jobs, submitted from the workstream directory
`/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement` **[src: the headers of the
three job scripts]**:

```
qsub -v RUN_TAG=v1 scripts/pbs/survey_prepare.pbs            # the pool, the acquisition check, the selection; prints the -J to use
qsub -J 0-50 -v RUN_TAG=v1 scripts/pbs/survey_cells.pbs      # one array task per selected cell: align,nodes,radius,registration,camera,calibration
qsub -v RUN_TAG=v1 scripts/pbs/survey_pool.pbs               # the pooled analyses; DIAM_POOL=calibration for the kernel alone
```

Each accepts `DIAM_DRYRUN=1` (prints the commands only). Defaults: conda env `spine_env`; survey directory
`/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_survey/<RUN_TAG>`; a module `proxy` opens the network for
the jobs that fetch images. The array is resumable: a step with a marker in `<survey>/<cell>/done/` is skipped.

| Step | Code | Writes |
|---|---|---|
| Pool, acquisition check, selection | `scripts/cell_pool.py` (`pool`, `check`, `select`, `pick`) | `pool.csv`, `acquisition.csv`, `selection.csv` |
| Per cell: alignment, node sample, calibration scans | `scripts/survey_cells.py run --task-index i --steps ...` | `<cell>/align/`, `<cell>/nodes/pilot_<sid>.csv`, `<cell>/calibration/scans_<sid>.json` |
| Pooled kernel | `scripts/survey_cells.py pool --what calibration`, then `calibrate_kernel.py fit` and `far-fit` on the pooled scans | under `<survey>/s3c/`; the archive `<survey>/results_<RUN_TAG>.tgz` |

Status as far as this chat can see:
- The roadmap (`specs/roadmap/diameter-pipeline.md`, rendered at `1fb63c4`) shows S3c "pushed", not running or done,
  and I19 (the pooled kernel) "pushed" with no tester report covering the current commit; the repository has no
  `hpc-status` branch **[src]**. So no cluster run of the pooled calibration is recorded. Whether one happened
  without a marker: **[open]**, ask the user.
- The Colab run of 2026-10-28 covered the pool, the acquisition check and a selection of 51 cells **[log, row
  2026-10-28 (later, 4)]**.
- `docs/project/claude/TEEG_diameter_HPC_paths_reference.md` (on `sci/diameter-pipeline`) and
  `docs/TEEG_diameter_calibration_handoff_2026-10-29.md` exist **[not read]** by this chat beyond the latter's
  headings.

## 5. Decisions in force for the pooled calibration **[log]**

| ID | Content | State |
|---|---|---|
| D-085 | The cell programme: 5 cells per class and layer, all when fewer; plane spacing and pixel size checked first. The pool: 158 human cells with a reconstruction, all 158 at 0.28 um and 0.1144 um per pixel; 51 selected. | Implemented; D-170 asks for all the eligible cells instead (section 7, P1). |
| D-088, D-118 | Up to 30 scans per cell and set (near, far); the pool takes every cell's, no cap in all; at least 10 nodes per set for a production kernel. | Implemented. |
| D-118.3 | The pooled standard errors resample cells from 10 cells; with 2 to 9 cells the larger of the node and cell SEs. | Implemented. |
| D-119 | The growth fit with a sparse Jacobian. | Implemented. |
| D-120.2 | The pooled fits weight nodes so that every cell carries the same total weight. | Implemented. |
| D-120.3 | A cell whose mean residual is too rare under the between-cell Gaussian leaves the pool. | Implemented; open finding: the rule removes a clean cell in 0.08 to 0.33 of simulated datasets instead of 0.05 (SPEC 19.8). |
| D-121.1 | A depth check: the near law fitted on the shallow and the deep half of the pooled nodes, flagged when a knot differs by more than 2 SE family-wise. | **Not implemented.** |
| D-121.2 | The calibration loop over the pool: each round reruns the cells' calibration step with the previous round's law, all rounds submitted at once, stopped below one SE or after 5 rounds. | **Not implemented.** |
| D-117 | Cell 7's loop on one cell (at most 3 rounds). | Implemented (Colab). |

## 6. What the next chat must know before trusting a pooled table

- **The in-focus width is not measured by this calibration.** Only $G$ is. A node's fitted diameter is confounded
  with $\sigma_{\rm r}(0)$: on synthetic tubes fitted with 0.080 um, a true in-focus width of 0.22 / 0.30 um makes a
  0.5 um tube come out at 0.81 / 1.03 um with a misfit under 1 gl, and the inflation falls with size (x 1.61, 1.26,
  1.10, 1.04 at 0.5, 0.8, 1.2, 2.0 um under 0.22 um) **[run, sandbox: `checks/kernel_2026-11-18/`]**. A pooled
  table over many cells averages the noise of $G$; it does not settle $\sigma_{\rm r}(0)$ (D-165 left that open).
- **How much the table's other errors matter:** a growth of the variance 0.5 to 2 times the table's moves a clear
  node's fitted diameter by -5 to +5 %; a halo of 10 to 30 % of the kernel at 3 sigma by +1 to +15 %
  **[run, sandbox: `kernel_sensitivity_check.py`]**. Synthetic, one flat tube.
- **The near set is chosen by Allen's diameter** (at most `node_allen_diameter_max_um` = 0.6 um, "thin") **[src:
  SPEC Block 10]**. On the pilot, clear nodes with Allen diameters of 0.45 to 0.54 um were fitted at 0.77 to
  1.09 um, and a node of Allen 1.2 um shows a dip about 2 um wide **[data: the user's Colab uploads of
  2026-11-18; figures read by eye]**. If the "thin" nodes are that thick, their depth extent is comparable to the
  plane spacing, and Eq. 4's reading of the width's growth as the kernel's alone is strained **[reasoning, not
  tested]**. SPEC section 8 already records a bias of the first stage near focus of about 0.01 um$^2$ in $G$ on
  thin phantoms and a second calibration step that is not built.
- **Pollution.** Three of five nodes that the trace calls clear carried untraced dark structures on their line
  **[data, 17:25]**. The calibration's own isolation test reads the trace and the predicted shadows of traced
  neighbours **[src: `calibrate_kernel.py` docstring, 10.I]**; whether it rejects such nodes was not checked.

## 7. Points to settle with the user before running

| # | Point | State and suggestion |
|---|---|---|
| P1 | What "all the eligible cells" means. | The pool has 158 cells, all at the configured spacing and pixel size; D-085 took 51. Candidates for "eligible": all 158; those whose automatic alignment is not set aside; those that reach `calib_min_nodes_per_set`. Suggestion: every cell of the pool whose alignment passes, the count reported before the array is submitted. Changing the selection is `cell_pool.py select` with a larger `survey.cells_per_group`, or a selection file listing every pool cell **[src]**. |
| P2 | Whether all cell classes are pooled into one kernel. | Today one kernel for all; the law by cell is plotted (19.5) and outlier cells leave (19.8). Confirm. |
| P3 | Whether D-121's depth check and loop are built before the run. | Suggestion: run the existing one-pass pool first on all cells, read it, then decide. |
| P4 | The in-focus width. | Not obtained by this run (section 6). Options put to the user on 2026-11-18, not decided: the narrowest in-focus dips of a cell as an upper bound; a fit on thick dendrites with the width free. Decide whether it belongs to this chat. |
| P5 | Which steps run on the new cells. | The calibration step needs `align` and `nodes` first. Suggestion: `DIAM_STEPS=align,nodes,calibration`. |
| P6 | Cost. | The calibration step took 964 s on the pilot in Colab **[log, D-121]**; 158 array tasks in parallel. Walltime and cores to confirm against davinci's limits. |
| P7 | What is delivered. | Suggestion: the pooled `calibration.json` and `far_fit.json`, the law-by-cell figures, the table's values with their SEs, and a config JSON the cleaning runs can take with `--config-json`. |

## 8. Not checked

- Nothing was run on davinci by this chat; the job scripts were read, not executed.
- The suites of Blocks 10, 15, 16 and 19 were not rerun on 2026-11-18; the last tester report covering Block 10 is
  of 2026-11-13.
- Whether the pool's `fit` and `far-fit` have ever run on more than the pilot's scans.
- The literature on the bright-field kernel's shape: three PubMed queries on 2026-11-18 returned nothing usable.
