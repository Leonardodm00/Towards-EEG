# Handoff: the defocus kernel table built on davinci from all the eligible cells (D-170) (clearance chat -> next chat)

| Date | Change |
|---|---|
| 2026-11-18 | Created at the user's request (17:38, D-170) for the chat that starts the construction of the defocus table on davinci. Branch `sci/diameter-pipeline` at `1fb63c4` (code) and the commit that adds this file; decision log on `main` at `30ed363` (next free IDs D-171, I-006). This chat did not run anything on davinci and did not change the calibration code. |
| 2026-11-18 (later) | The defocus-table chat (18:40, after the user's answers): section 7 settled by **D-174** and **D-175** (log on `main`, `a703319`); section 5's D-117 row corrected (the loop is not built); section 6 gains the pool job's cost at 158 cells and the near fit's bias on thick phantoms (`checks/defocus_pool_2026-11-18/`); section 8's suites rerun; new section 9, the run and its commands. No code changed. |
| 2026-11-18 (later, 2) | The defocus-table chat (21:40): D-174's held diameter is implemented at `513ad9e` (D-178, main's log at `44b285f`), so section 9's row 6 -- the `nodes` and `calibration` steps -- no longer waits for it and follows the alignment array (row 5); section 7's P5 notes the commit. Evidence: `git log` of `sci/diameter-pipeline` (`513ad9e`, "the first stage holds the diameter at Allen's for every user"). |
| 2026-11-19 | P4 (section 7): the clearance chat's kernel check on node 5353 noted in the row (in-focus width against heavier tails); nothing decided. |
| 2026-11-23 | The defocus-table chat (09:01): **D-226** (user) changes the plan -- the calibration's nodes are chosen by the user from per-node images (focus and $k^* \pm 3$ for the near set, $k^* \pm 1$ to $\pm 16$ for the far set) of the cells of the passive + I_h fit only, with diameters set by eye in a Colab notebook. A status note added under the title; sections 2, 5, 7 and 9 are kept as the record of the 158-cell plan, superseded for the cell set and the choice of nodes. Log on `main` at `cfcd1ae`. |
| 2026-11-23 (later) | The status note gains D-227 (user, 09:05): beside each plane's image, the intensity profile along the node's measuring line. Log on `main`. |
| 2026-11-27 | The defocus-table chat (10:00): **D-261** (user) -- the image at focus also shows Allen's diameter; the user chooses the nodes on its agreement with the image and on clean backgrounds across planes; no diameter is set by eye (D-226.4 and its Colab notebook withdrawn); the chosen nodes go to steps 1 and 2, step 2's twins carrying Allen's diameter as the true one. Status note updated. Log on `main` at `69fc18e`. |
| 2026-11-27 (night) | The step-2 implementation chat: **D-278** (user) -- the screening export of D-226, D-227 and D-261 is built (SPEC Blocks 61 and 62; the step `screening` of `scripts/survey_cells.py`). New section 10: the screening run on the 18 L3_exc cells, command by command. Not run on real images yet. |
| 2026-11-28 | The step-2 implementation chat (13:05): **D-281** (user) -- the screening run covers all the cells of the archive (every `specimen_<id>` under `L<n>_exc` and `L<n>_inh`, 68 in the user's listing of 2026-11-28), not the 18 L3_exc cells. Section 10: the opening paragraph marked corrected; rows 4 to 8 rewritten (the pick from all the groups, `selection_all.csv`; the probe on the pilot 529878215 by `DIAM_CELLS`; the array `-J 0-67`); the new pick and the job script's dry runs checked on a stand-in of the archive tree (68 picked, 69 lines; task 67 accepted, task 68 refused with exit 4). The marks of Allen's diameter are segments since `066e603` (D-280). |

**Status [updated 2026-11-23, D-226]:** the run below -- every eligible cell of the 158 (D-175.1), the nodes chosen by Block 10's rules -- is superseded for the cell set and the choice of nodes. The screening runs on davinci only on the cells of the passive + I_h fit (in the assistant's reading the 18 L3_exc cells, to be confirmed) and exports, for every apt node, its image at focus with $k^* \pm 3$ (near set) or $k^* \pm 1$ to $k^* \pm 16$ (far set, `far_offsets_planes`, D-074.1), each plane's image beside its intensity profile along the node's measuring line (D-227); the user picks the clearest nodes and sets their diameters by eye in a Colab notebook [corrected 2026-11-27, D-261: no diameter is set by eye and there is no such notebook; the image at focus also shows Allen's diameter, and the user picks the nodes whose Allen's diameter agrees with that image and whose background is clean across planes]; the calibration then runs on the picked nodes (steps 1 and 2, step 2's twins at Allen's diameter, D-261). None of this is built yet; sections 2, 5, 7 and 9 stay as the record of the 158-cell plan.

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
| D-117 | Cell 7's loop on one cell (at most 3 rounds). | ~~Implemented (Colab).~~ [corrected 2026-11-18 (later): **not implemented** -- no SPEC 10.L, no loop in Cell 7 (`notebooks/phase2_colab.py`), and D-117's status still reads "D-117.2 next" **[src, log]**; the pilot's table is one pass under the configured law] |

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

- [added 2026-11-18 (later)] **The pool job's cost at 158 cells.** On synthetic pooled scans of 158 cells x (30
  near, 2 far), the pool job's calibration (`survey_cells.pooled_fits`) takes about 5 h per pass at 1000 cell
  resamples: the far fit keeps a dense Jacobian (D-119 made only the near fit sparse) and one far fit takes 18 s at
  316 far nodes; a cell removed by Block 19.8 runs both fits again. The header's 2 h is not enough; D-175.3 sets 48 h
  **[run, sandbox: `checks/defocus_pool_2026-11-18/pool_timing.py`, `split_timing.py`]**.
- [added 2026-11-18 (later)] **The near fit is biased under the pipeline's own forward model, and the bias depends
  on thickness.** Flat phantoms rendered with the pilot's table as the truth (16 per range), fitted by Block 10:
  sigma_r at 0.28 / 0.56 / 0.84 um off by -15.6 / -4.3 / -10.6 % (true d 0.15-0.25 um), -30.8 / -14.1 / -4.6 %
  (0.45-0.55) and +11.8 / -24.1 / -7.8 % (0.75-0.95). A pooled table removes the noise, not this; step 2 (NS1) would
  **[run, sandbox: `checks/defocus_pool_2026-11-18/thick_near_check.py`]**. Whether step 2 belongs to this chat:
  asked (D-175.4).

## 7. Points to settle with the user before running

| # | Point | State and suggestion |
|---|---|---|
| P1 | What "all the eligible cells" means. | The pool has 158 cells, all at the configured spacing and pixel size; D-085 took 51. Candidates for "eligible": all 158; those whose automatic alignment is not set aside; those that reach `calib_min_nodes_per_set`. Suggestion: every cell of the pool whose alignment passes, the count reported before the array is submitted. Changing the selection is `cell_pool.py select` with a larger `survey.cells_per_group`, or a selection file listing every pool cell **[src]**. |
| P2 | Whether all cell classes are pooled into one kernel. | Today one kernel for all; the law by cell is plotted (19.5) and outlier cells leave (19.8). Confirm. |
| P3 | Whether D-121's depth check and loop are built before the run. | Suggestion: run the existing one-pass pool first on all cells, read it, then decide. |
| P4 | The in-focus width. | Not obtained by this run (section 6). Options put to the user on 2026-11-18, not decided: the narrowest in-focus dips of a cell as an upper bound; a fit on thick dendrites with the width free. Decide whether it belongs to this chat. [2026-11-19, the clearance chat: on the pilot's clear node 5353 the fit's $\chi^2/N$ changes by at most 0.6 between in-focus SDs of 0.080 and 0.25 um while $\hat d$ moves from 0.77 to 0.45 um, and heavier-tailed kernels at 0.080 um fit worse; its narrowest dip (0.58 um at half depth) caps the in-focus SD near 0.25 um (`checks/kernel_tails_2026-11-19/`; `TEEG_clear_node_fit_handoff_2026-11-19.md` section 7) **[run, sandbox]**.] |
| P5 | Which steps run on the new cells. | The calibration step needs `align` and `nodes` first. Suggestion: `DIAM_STEPS=align,nodes,calibration`. |
| P6 | Cost. | The calibration step took 964 s on the pilot in Colab **[log, D-121]**; 158 array tasks in parallel. Walltime and cores to confirm against davinci's limits. |
| P7 | What is delivered. | Suggestion: the pooled `calibration.json` and `far_fit.json`, the law-by-cell figures, the table's values with their SEs, and a config JSON the cleaning runs can take with `--config-json`. |

**Settled 2026-11-18, about 18:40 [user; log on `main`, `a703319`]:** P1 -- every pool cell with a class, alignment-
gated, the few-node cells decided at the pool step (D-175.1). P2 -- one kernel for all cells stands (D-088.1); the
residuals by cell are read by class and layer afterwards. P3 -- round 1 under the configured law, round 2 by hand
under the pooled law to D-121.2's stop rule; D-121.1 on the returned scans (D-175.2). P4 -- the in-focus width stays
0.080 um (D-172; D-175.5). P5 -- `align,nodes,calibration` (D-175.4); `nodes` and `calibration` wait for the first
stage with the diameter held at Allen's, the default once built (D-174; built at `513ad9e`, D-178). P6 -- at most 100 one-core tasks at once,
48 h walltime (D-175.3). P7 -- as suggested, `DIAM_POOL=calibration` only; this chat on the kernel only (D-175.4).

## 8. Not checked

- Nothing was run on davinci by this chat; the job scripts were read, not executed.
- [corrected 2026-11-18 (later): rerun at `2de4bf3` by the defocus-table chat -- the 14 suites of Blocks 10, 15, 16, 17, 19 and the survey jobs pass, design skips only **[run, sandbox]**] ~~The suites of Blocks 10, 15, 16 and 19 were not rerun on 2026-11-18;~~ the last tester report covering Block 10 is
  of 2026-11-13.
- Whether the pool's `fit` and `far-fit` have ever run on more than the pilot's scans.
- The literature on the bright-field kernel's shape: three PubMed queries on 2026-11-18 returned nothing usable.

## 9. The run as settled (D-174, D-175) [added 2026-11-18 (later)]

Run tag `allcells1`; one configuration for every job, `configs/survey_allcells1.json` (`{"survey":
{"cells_per_group": 1000}}`: every cell of each class and layer), copied to a path without spaces. A value with
commas goes to `qsub -v` by name, exported, never inline (PBS splits `-v` at commas). Submit from the workstream
directory. Expected lines are those of section 4's scripts.

| # | When | Command (login node) | Expect |
|---|---|---|---|
| 1 | now | `cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement" && git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only && git log -1 --oneline && bash -n scripts/pbs/survey_common.sh scripts/pbs/survey_prepare.pbs scripts/pbs/survey_cells.pbs scripts/pbs/survey_pool.pbs && echo "OK: bash -n"` | the commit pushed with this section, `OK: bash -n` |
| 2 | now | `mkdir -p ~/diam && cp configs/survey_allcells1.json ~/diam/ && cat ~/diam/survey_allcells1.json` | `{"survey": {"cells_per_group": 1000}}` |
| 3 | now | `qsub -v DIAM_DRYRUN=1,RUN_TAG=allcells1,DIAM_CONFIG_JSON=$HOME/diam/survey_allcells1.json scripts/pbs/survey_prepare.pbs`, then the same without `DIAM_DRYRUN=1,` | dry run: four `DRYRUN` lines with `--config-json`; real: `api.brain-map.org reachable`, `pool: ...`, `select: ... cells selected (seed 20261026, 1000 per group)`, `next: qsub -J 0-<N-1> ...` |
| 4 | after 3 | `qsub -v RUN_TAG=allcells1,DIAM_STEPS=align,DIAM_CONFIG_JSON=$HOME/diam/survey_allcells1.json,ROADMAP_DISABLE=1 scripts/pbs/survey_cells.pbs` (the probe: row 0 only) | `api.brain-map.org reachable`, one alignment line, `task 0 ... done` |
| 5 | after 4 | `qsub -J 0-<N-1>%100 -v RUN_TAG=allcells1,DIAM_STEPS=align,DIAM_CONFIG_JSON=$HOME/diam/survey_allcells1.json,ROADMAP_DISABLE=1 scripts/pbs/survey_cells.pbs` (alignment of every cell; at most 100 at once) | if qsub refuses `%100`: `-J 0-99` now, `-J 100-<N-1>` after it; then `status.csv` and the overlays of the set-aside cells to judge (D-098.5) |
| 6 | after 5; D-174's held diameter is in the code since `513ad9e` (D-178) [updated 2026-11-18, was: "after D-174 is in the code"] -- `git pull --ff-only` first, `git log -1` at `513ad9e` or later | `DIAM_STEPS=align,nodes,calibration qsub -v RUN_TAG=allcells1,DIAM_STEPS,DIAM_DRYRUN=1,DIAM_CONFIG_JSON=$HOME/diam/survey_allcells1.json scripts/pbs/survey_cells.pbs`, then `DIAM_STEPS=align,nodes,calibration qsub -J 0-<N-1>%100 -l walltime=48:00:00 -v RUN_TAG=allcells1,DIAM_STEPS,DIAM_CONFIG_JSON=$HOME/diam/survey_allcells1.json scripts/pbs/survey_cells.pbs` | dry run: `steps align,nodes,calibration`; the array: `align` skipped (done), the two steps per cell |
| 7 | after 6 | `cd "/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_survey/allcells1" && python3 -c 'import json,glob; r=[json.load(open(f)) for f in sorted(glob.glob("*/done/calibration.json"))]; [print(x["specimen_id"], x["summary"].get("n_near"), x["summary"].get("n_far"), int(x["seconds"])) for x in r]; print(len(r), "cells with the calibration step done")'` | one line per cell: near and far scans, seconds -- the counts on which D-175.1's minimum is chosen |
| 8 | after 7 | `qsub -l walltime=48:00:00 -v RUN_TAG=allcells1,DIAM_POOL=calibration,DIAM_CONFIG_JSON=$HOME/diam/survey_allcells1.json scripts/pbs/survey_pool.pbs` (with `DIAM_SELECTION=<a filtered selection, path without spaces>` if a minimum is set) | `pooled near field: ...`, `growth fit ...`, the table, the anchor and its cell SE, the far fit, the cell test, then `results_allcells1.tgz` |

Round 2 (D-175.2), after reading round 1: keep `s3c/` as `s3c_round1/`; a second configuration adds round 1's pooled
law (`renderer.kernel_table_delta_um`, `kernel_table_sigma_um`, `sigma_r0_um`, `kernel_continuation_slope`); the
array again with `DIAM_FORCE=calibration` (only the calibration step reruns), then step 8; stop when the anchor and
the far slope each move by less than one SE, at most 5 rounds.

## 10. The screening run (D-226, D-227, D-261, D-278) [added 2026-11-27]

**[corrected 2026-11-28, D-281: the run covers every cell of the archive -- each `specimen_<id>` under `L<n>_exc` and `L<n>_inh` of `/davinci-1/home/ldellamea/Human Neurons Fitting`, 68 cells in the user's listing of 2026-11-28 (L2 exc 10, inh 1; L3 exc 18, inh 11; L4 exc 6, inh 2; L5 exc 8, inh 5; L6 exc 5, inh 2) -- not the 18 L3_exc cells this paragraph names; one task per cell, 1 core each.]**

What it does: on the 18 L3_exc cells of the passive + I_h fit (D-124.2: the cells under
`/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc/specimen_*`), per cell: the alignment, the node sample, the
calibration scans of every apt node (the cap of 30 scans per cell and set lifted by `configs/survey_screen1.json`),
then the step `screening`, which writes under `<survey>/<sid>/screening/` one sheet per apt node
(`near_<node>.png`, `far_<node>.png`: each plane's image beside its profile along the measuring line, Allen's
diameter marked on the plane at focus, a far node's sheet opening with that plane zoomed), `screening_<sid>.npz`,
`screening_index_<sid>.csv` and `candidates_<sid>.csv`. Run tag `screen1`; survey directory
`/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_survey/screen1`. A value with commas goes to `qsub -v` by
name, exported; paths given to `qsub -v` have no spaces (section 9).

Checked in the sandbox **[run]**: `test_smoke_s3b.py` runs the four steps on a synthetic cell (one sheet, the files,
the candidates read back by step 2's reader); `test_smoke_survey_pbs.py` (the job script against stubs);
`py_compile`, `bash -n`, no non-ASCII byte and no CR in the files touched. Not checked: a run on real images (it
needs Allen's server), the time one cell takes with the cap lifted **[open]** -- the probe of step 5 measures it.

| # | Command (login node, one line each) | Expect |
|---|---|---|
| 1 | `cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement" && git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only && git log -1 --oneline && bash -n scripts/pbs/survey_common.sh scripts/pbs/survey_prepare.pbs scripts/pbs/survey_cells.pbs && python3 -m py_compile scripts/survey_cells.py src/allen_diameter/analysis/screening.py src/allen_diameter/loading/screening_io.py src/allen_diameter/plotting/screening_figures.py && python3 -c "import sys; b=[p for p in sys.argv[1:] if any(c>127 or c==13 for c in open(p,'rb').read())]; print('OK: bytes' if not b else 'BAD: %s' % b)" scripts/survey_cells.py scripts/pbs/survey_cells.pbs scripts/pbs/survey_common.sh src/allen_diameter/analysis/screening.py src/allen_diameter/loading/screening_io.py src/allen_diameter/plotting/screening_figures.py && grep -c "def step_screening" scripts/survey_cells.py` | the commit that carries this section or a later one; `OK: bytes`; `1` |
| 2 | `mkdir -p ~/diam && cp configs/survey_screen1.json ~/diam/ && cat ~/diam/survey_screen1.json` | `{"survey": {"calib_scans_per_cell": 100000}}` |
| 3 | `qsub -v RUN_TAG=screen1,DIAM_CONFIG_JSON=$HOME/diam/survey_screen1.json scripts/pbs/survey_prepare.pbs` | in its log: `api.brain-map.org reachable`, `pool: ...`, `select: ...` (the selection it writes is replaced in step 4) |
| 4 | after 3, with the environment of the jobs active (`conda activate spine_env`); every cell of the archive (D-281): `A="/davinci-1/home/ldellamea/Human Neurons Fitting" && SV="$A/diameter_survey/screen1" && IDS=$(ls -d "$A"/L?_exc/specimen_* "$A"/L?_inh/specimen_* \| sed 's/.*specimen_//' \| sort -un \| paste -sd, -) && echo "$IDS" \| tr ',' '\n' \| wc -l && python3 scripts/cell_pool.py pick --pool "$SV/pool.csv" --cells "$IDS" --out "$HOME/diam/selection_all.csv" && wc -l < "$HOME/diam/selection_all.csv" && grep -c "^529878215," "$HOME/diam/selection_all.csv"` | `68`; `pick: 68 cells -> ...` and the cells by group; `69` (the header and 68 rows); `1` (the pilot is in the selection). A cell outside the pool stops the pick with `pick_cells: specimen <id> is not in the pool`: report it |
| 5 | the dry run, then the probe on the pilot 529878215 (a cell whose earlier steps are known, so a failure is the new step's): `export DIAM_STEPS=align,nodes,calibration,screening && qsub -v RUN_TAG=screen1,DIAM_STEPS,DIAM_DRYRUN=1,DIAM_CELLS=529878215,DIAM_SELECTION=$HOME/diam/selection_all.csv,DIAM_CONFIG_JSON=$HOME/diam/survey_screen1.json scripts/pbs/survey_cells.pbs` and, after reading its log, `export DIAM_STEPS=align,nodes,calibration,screening && qsub -l walltime=48:00:00 -v RUN_TAG=screen1,DIAM_STEPS,ROADMAP_DISABLE=1,DIAM_CELLS=529878215,DIAM_SELECTION=$HOME/diam/selection_all.csv,DIAM_CONFIG_JSON=$HOME/diam/survey_screen1.json scripts/pbs/survey_cells.pbs` | dry run: `cells 529878215` and a `DRYRUN: python3 scripts/survey_cells.py run` line with `--steps align,nodes,calibration,screening`, `--cells 529878215` and `--config-json`; probe: a line with `near-field, ... far-field scans` (the calibration) and one with `sheets:` (the screening), then `cells 529878215 done`; note the walltime and the memory it used (`qstat -xf <jobid>`) |
| 6 | after the probe: `SV="/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_survey/screen1" && S=529878215 && ls "$SV/$S/screening" \| head -5 && ls "$SV/$S/screening" \| grep -c png && wc -l < "$SV/$S/screening/candidates_$S.csv" && du -sh "$SV/$S/screening"` | sheet names `near_<node>.png` / `far_<node>.png`; the number of sheets; that number plus one; the size of one cell's sheets (times 68 for the run) |
| 7 | after 6 looks right: `export DIAM_STEPS=align,nodes,calibration,screening && qsub -J 0-67 -l walltime=48:00:00 -v RUN_TAG=screen1,DIAM_STEPS,DIAM_SELECTION=$HOME/diam/selection_all.csv,DIAM_CONFIG_JSON=$HOME/diam/survey_screen1.json scripts/pbs/survey_cells.pbs` | 68 tasks of 1 core each, each ending with `task <i> ... done` (the pilot's task finds its steps done and skips them); in `status.csv` a cell whose step failed shows `pending` and is rerun by the same command; a cell set aside at the alignment shows `skipped:set_aside` and has no sheets (D-085.2) -- list them, the user decides whether they run with `DIAM_INCLUDE_SET_ASIDE=1` (D-098.5) |
| 8 | after 7: `cd "/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_survey/screen1" && ls -d */screening \| wc -l && grep -c "skipped:set_aside" status.csv; tar -czf screening_screen1.tgz */screening && ls -lh screening_screen1.tgz` | the cells with sheets (68 less the cells set aside or failed); the count of set-aside cells; the archive to download (binary transfer) |

What comes back from the user (D-261.2, D-275.3): one file `chosen_nodes.csv` with the header `specimen_id,node_id`
once and the chosen nodes' rows, made from the `candidates_<sid>.csv` files by deleting the rows not wanted. Step 2
reads it with `step2_io.read_chosen_nodes`; the cells' scans files (`<sid>/calibration/scans_<sid>.json`, now with
the full configuration in `meta`) are step 2's other input.
