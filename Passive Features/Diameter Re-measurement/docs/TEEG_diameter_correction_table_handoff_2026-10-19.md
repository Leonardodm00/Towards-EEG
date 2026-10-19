# Handoff: the bias (correction) table after Block 12 -- Blocks 13-14, the open points, the gates (table chat -> next table chat)

| Date | Change |
|---|---|
| 2026-10-19 (later) | Pushed as `6a8dbf4` on `sci/diameter-pipeline` with `checks/table_2026-10-18/` and `docs/TEEG_thin_plate_spline_in_3D/` at the user's agreement; a fresh clone of the pushed branch has the tree of the tested working copy, pure-ASCII `.py`, LF shell and PBS files, a clean `compileall`, the two hashes of §1 and `test_smoke_config` 6 pass 1 skip **[run, sandbox]**. The clone URL of §1 corrected to the repository's current name, `Towards-EEG` (the push reported that the lowercase name moved); §1's head and §2's repository row updated. |
| 2026-10-19 | Created at the close of the table chat (2026-10-16 to 2026-10-19), for the chat that implements Blocks 13-14. As the statement of the current state it supersedes §1, §2, §5, §7, §8, §10 and §11 of `TEEG_diameter_correction_table_handoff_2026-10-16.md`; that file's §3 (the estimator in full notation, (H1)-(H6)), §4 (the evidence behind D-052), §6 (where the node's radius enters) and §9 (the files of `checks/estimator_2026-10-15/`) stay the reference, and the parts of its superseded sections cited below (the gate commands of its §8.4, the real-data check at the end of its §7, the references of its §10) stay valid as cited. Code facts read on `sci/diameter-pipeline` at `d643aaf` **[src]**; the 24 smoke suites run there on 2026-10-19 **[run, sandbox]**; the numbers of §5 and §8 re-read from the logs of the table chat's checks; seven of the checks re-run from their copies in `checks/table_2026-10-18/` the same day: three reproduced their logs byte for byte, three agree with the numbers recorded before (the two scaling checks with D-054's Why, the quadrature check with SPEC Block 12's status), and `check_inverse_vs_classical.py` had no earlier record **[run, sandbox]**; decisions read from the log the same day, unchanged since the table chat's last write (2026-10-18, row "(later, 6)") **[KB]**. Fact-checked the same day by an independent agent against the code, SPEC, the log and the check logs; its corrections are in (the SPEC §8 bullets to read, the range-setting draw of $\ln\hat\alpha$, the plane counts of today's provider call, the diagnostics and flags Block 13 changes (§5.1 (c), (d)), the PROVISIONAL iteration settings, the full list of the $\sigma_{\rm fit}$ study's code, the B/B2 tolerances, the classical inverse's scope, a `bbar_few` row's fit, Block 10's dark nodes, D-020's `main`, the tags of SPEC claims, the notation clashes). |

Basis tags: **[src]** read in the named file at `d643aaf`; **[run, sandbox]**
executed in the assistant's sandbox (Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1);
**[user]** said or decided by the user; **[KB]** project knowledge (the
decision log); **[doc]** a document of this folder, section named;
**[reasoning]** the assistant's inference, not checked; **[open]** neither
decided nor verified.

Plainly: the new per-node fit is coded (Block 12) but does not yet run inside
the measurement, because the plane it must be applied at (the Allen-guided
iteration) still lives in a diagnostic script. In the table chat the user
settled most of the table's design: what one element of the table is, what the
spline is fitted through, the third axis and the scaling, the failure count, and
the radius phantoms start from. A short list of values and rules is still open
(§5). The next chat moves the iteration into the measurement (Block 13),
rebuilds the table code for those decisions (Block 14), and takes the result
through gate 2 and onto davinci.

## 1. First actions in the next table chat

1. Read, in this order:
   - this file;
   - the 2026-10-16 handoff's §3 ((H1)-(H6)), §4 and §6;
   - in `TEEG_decisions_and_ideas_log.md` (project knowledge): D-052, D-053, D-054, D-057,
     D-058, D-059, D-060, then D-024, D-030, D-031 and D-035 for context;
   - `specs/SPEC.md` §2.4, §4, Blocks 5-7 and 12, and the first four bullets of §8 (l. 611-620, raised by the table chat);
   - for the spline in three dimensions: `docs/TEEG_thin_plate_spline_in_3D/TEEG_thin_plate_spline_in_3D.md` §3.10-3.13
     (if it was pushed with this file; otherwise the zip the user received on 2026-10-18).

   **ID collision (unchanged):** the repository's code and the SPEC text written before
   2026-10-18 call the gradient-energy focus rule "D-030"; in the log it is **D-035**, and
   the log's D-030 is "every real node gets its own estimate whatever its optical depth"
   **[KB; src: SPEC §4, ID note]**.
2. Get the branch (in an existing clone, `git pull --ff-only` instead):

   ```
   cd /home/claude && git clone https://github.com/Leonardodm00/Towards-EEG.git towards-eeg && cd towards-eeg && git checkout sci/diameter-pipeline && git log --oneline -3
   ```

   Head: `6a8dbf4` (the commit that added this file, `checks/table_2026-10-18/` and the thin-plate document on top of
   `d643aaf`) or the later one that corrected this file, or newer.
3. Run every suite once before changing anything, and print the hashes:

   ```
   cd "Passive Features/Diameter Re-measurement" && for f in tests/smoke/test_smoke_*.py; do python3 "$f" > /dev/null 2>&1 && echo "pass $f" || echo "FAIL $f"; done
   PYTHONPATH=src python3 -c "from allen_diameter.config import DiameterConfig as D; c = D(); print(c.signature_hash('estimator'), c.signature_hash('simulation'))"
   ```

   At `d643aaf`: 24 suites, all pass (§2); hashes `1bb3585b4a4f1373 e44ac0bd220f5214` **[run, sandbox, 2026-10-19]**.
4. Block 13 needs only the four implementation points of §5.1, which the assistant proposes and the user confirms.
   Block 14's logic waits for §5.2 (scientific-coding §2: settle the science before writing logic), asked in
   batches of multiple-choice questions.
5. Blocks 13 and 14 (§6), each through the block protocol (SPEC entry first, code, verify twice, commit), then
   the gates (§7).

## 2. Where things are

| What | Where | Note |
|---|---|---|
| Repository | `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline` (the lowercase URL `.../towards-eeg` redirects: GitHub reports the repository moved) | code at `d643aaf`, equal to `origin` before this handoff's commit `6a8dbf4` (`git ls-remote`, 2026-10-19), which changed no code **[run]** |
| Workstream | `Passive Features/Diameter Re-measurement/` | `src/allen_diameter/`, `scripts/`, `tests/smoke/`, `configs/gate2_reduced.json`, `docs/`, `checks/`, `notebooks/` |
| Spec | `specs/SPEC.md` (repository root) | §2.4 table and correction; §4 the binding decisions to D-060 and the ID note; Block 6 = phantoms and table, Block 7 = inversion, flags, fill, Block 12 = the SGT fit; §8 open questions, the first four (l. 611-620) raised by the table chat |
| Decision log | project knowledge, `TEEG_decisions_and_ideas_log.md` | D-001 to D-060; next free D-061, I-003 (2026-10-19) **[KB]** |
| Documents | `docs/` of the workstream (D-020: the canonical home, not project knowledge) | the 2026-10-16 handoff (estimator, evidence, radius), this file, the procedure and mathematics (2026-10-04), `TEEG_thin_plate_spline_in_3D/` (2026-10-18, if pushed). D-020's Statement names the folder "on `main`"; with pushes never to `main` (§11) the workstream's documents live on `sci/diameter-pipeline` until it is merged, as the 2026-10-16 handoff did **[KB; src]** |
| Evidence | `checks/estimator_2026-10-15/` (D-052), `checks/table_2026-10-18/` (this chat's checks, §8) | each with its results and what a correct run prints: the first as JSON/NPZ and its `README.md`, the second as logs and its `README.md` |
| Block 12 code | `model/blur_kernels.py` (the SGT family, not in `FAMILIES`), `analysis/fit_sgt.py` (`fit_plane` -> `PlaneSGT(final, unweighted, gauss_fitted, gauss_fixed)`), the `sgt_*` fields of `MeasureConfig`, `tests/smoke/test_smoke_sgt_fit.py` | commit `bee1e0b` **[src]** |
| Per-node chain | `analysis/node_pipeline.py`: `measure_node` (l. 206); its final fit is still `fit_profile(v, I, phi, last.B_bar, m)` (l. 257), Block 5's Gaussian at $\sigma_{\rm fit}$, at the pilot's $k^*$; `_plane_range` (l. 122) requests $\pm$`planes_half` (3) planes widened for tilt in one provider call | **[src]** |
| Iteration code today | `scripts/plane_differences.py`: `ITER_LINE_MULT` 3.0 (l. 114), `node_stack` (148), `far_from_dendrites` (219), `plane_background` (238), `line_profiles` (252), `block_background` (270), `ITER_STATUSES` (427), `check_iteration` (430), `iterate_planes` (447), `refit_at_plane` (497), `gradient_iteration` (547) | **[src]** |
| Table code | `analysis/phantoms.py` (`SELECTION_FLAGS` l. 29, `_draw` 45, `draw_replicate` 66, `phantom_branch` 79, `reject_reasons` 91, `run_replicate` 103), `analysis/table.py` (`table_inputs` 26, `choose_smoothing` 38, `BiasTable` 63, `fit_table` 147), `analysis/invert.py` (`invert_node` 43, `correct_nodes` 74), `analysis/fill.py`, `loading/table_io.py`, `scripts/build_table.py`, `scripts/end_to_end.py`, `scripts/pbs/build_table.pbs` | **[src]** |
| Configuration | `config.py`: `MeasureConfig` l. 78 (`planes_half` 3 l. 88, `profile_half_um` 3.0 l. 103, `sigma_fit_study_um` l. 112, `sgt_max_nfev` 3000 l. 134, `alpha_dark_flag` 1.0 l. 139); `CorrectionConfig` l. 149 (`spline_smoothing_grid` l. 155, `tau_kernel_bandwidth` two values l. 156, `max_failure_rate` 0.2 l. 159, `failure_rate_ignore` `("dark",)` l. 160); `PhantomConfig` l. 210 (`n_replicates` 2000, `d_range_um` (0.2, 4.0) TEMPORARY, `mu_range_per_um` (0.3, 3.0), `jitter_z_um` 0); `validate` l. 266 (requires exactly two bandwidths, l. 369-370) | **[src]** |
| Hashes at the defaults | estimator `1bb3585b4a4f1373`, simulation `e44ac0bd220f5214` | **[run, sandbox, 2026-10-19]**; gate 2 passed at `535847535e83cf7d` (2026-10-06, dip-depth focus rule) **[src: SPEC Block 7]** |
| Cluster | davinci-1, clone `/davinci-1/home/ldellamea/TEEG/Towards-EEG`, tables under `/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_tables/<RUN_TAG>` | commands, knobs, exit codes, logs: project doc `claude/TEEG_diameter_HPC_paths_reference.md` §2 **[KB]**; no diameter job has run there **[KB]** |
| Real-data runs | Colab, `MyDrive/diameters/pilot/...` (Cells 4j-4m) | D-052's fit has never run on Allen's data **[KB]** |

Suites at `d643aaf` on 2026-10-19, all exit 0, 151 checks pass, 0 fail, 21 skip **[run, sandbox]**:
`test_smoke_calibration`, `camera_tables`, `cell`, `config`, `fit`, `fit_variants`, `focus`, `geometry`,
`invert`, `iteration_fits`, `node_pipeline`, `pbs`, `phase2`, `plane_diff`, `plane_entropy`,
`plane_gradient`, `plane_iteration`, `ray_world`, `render`, `sgt_fit`, `stretches`, `swc_io`, `table`,
`variant_iteration` (24; 0-218 s each, 21 min in all, on the sandbox's 2 cores with other checks running;
the slowest `variant_iteration` 218 s, `table` 195 s, `plane_iteration` 187 s, `iteration_fits` 117 s,
`sgt_fit` 115 s).

## 3. Decisions in force for the table, and what is coded

Symbols used below: $d$ the true diameter (um), $\varphi$ the tilt (rad), $\theta$ the heading, $\mu$ the
absorption coefficient (1/um), $\hat d$, $\hat\mu$ the fitted values, $\hat\alpha = \hat\mu\hat d/\cos\varphi$
the fitted optical depth; $\mathcal C$ the configuration, $\mathcal S$ the selection event;
$x = (\ln(d/\mathrm{um}),\ \varphi/\mathrm{rad},\ \ln\hat\alpha)$ the table's input; $\hat b(x \mid \mathcal C, \mathcal S)$
the fitted bias factor and $\hat m(d, \varphi, \hat\alpha \mid \mathcal C, \mathcal S) = d\,\hat b(x \mid \mathcal C, \mathcal S)$
the fitted mean response; $G$ the number of cylinder draws, $g = 1, \dots, G$; $n_{\rm real}$ the realizations
per draw, $r = 1, \dots, n_{\rm real}$; $\lambda_s$ the spline's smoothing (`RBFInterpolator`'s `smoothing`);
$r_{\rm br}$ the radius a node's branch carries, $r_{\rm A}$ Allen's radius; $\sigma_{\rm br}$ the SD of D-058's
radius noise; $\varpi_n(x)$ the Gaussian kernel weight of row $n$ at $x$ in the table's local statistics
(SPEC §2.4) and $\ell_n(x)$ the spline's own weight of row $n$ at $x$, $\hat b(x \mid \mathcal C, \mathcal S) = \sum_n \ell_n(x)\,y_n$
at a fixed $\lambda_s$.

**Notation clashes in the project's documents, flagged.** D-059 writes the realizations per draw as $k$, which
SPEC and the 2026-10-16 handoff use for the plane index: this file writes $n_{\rm real}$ (config field
`n_realizations`, D-059's proposal). D-058 writes the phantom's radius $r_n$ while D-059/D-060 use $r$ as the
realization index: this file writes $r_{\rm br}$ for the radius and $r_{\rm A}$ for Allen's. D-058 writes its
SD $\sigma_r$, which SPEC uses for the renderer's kernel width $\sigma_{\rm r}(\delta)$: this file writes
$\sigma_{\rm br}$ (config field `radius_noise_sd_um`, D-058's proposal). SPEC §2.4 and D-059 write the
nuisances as $\xi$, which Block 12 uses for the SGT's tail parameter: this file names the nuisances in words
(sub-pixel offset, axis depth, node jitter, camera noise). D-054 writes the spline's smoothing as $\lambda$,
which D-052 uses for the IRLS weight length (0.5 um): this file writes $\lambda_s$ for the former. The
procedure and mathematics documents write the bias's log-slope $\partial\ln b/\partial\ln d$ as $\beta$, which
Block 12 uses for the SGT's shape: this file writes $\beta_b$.

| ID | What it fixes for the table | Coded at `d643aaf`? |
|---|---|---|
| D-052 | the measurement: the Allen-guided plane (start $2r_{\rm br}$, lines $\pm 1.5 d$, $k_{\rm SWC} \pm 6$, at most 5 rounds) and at it the SGT fit, $(d, \mu, v_0, w, \beta, \xi, \zeta)$ all fitted, IRLS at $\lambda$ = 0.5 um; real nodes and phantoms alike | the kernel and the fit, yes (Block 12, `bee1e0b`); the plane rule inside `measure_node`, no (Block 13) **[src]** |
| D-053 | (i) the radius by a sensitivity check -- **superseded by D-058**; (ii) $\mathcal S$ fails on $d$, $\mu$, $v_0$, $w$ at a bound or an unconverged IRLS, never on the shapes; (iii) the third axis is $\ln\hat\alpha$ | (ii) yes, as Block 12's status rule; (iii) no (Block 14) |
| D-054 | (i) each input divided by its design range, CV on $\lambda_s$ alone; (ii) bandwidths $(0.15, 5°, 0.3)$ in $(\ln d, \varphi, \ln\hat\alpha)$, the third PROVISIONAL; (iii) `max_failure_rate` 0.2 PROVISIONAL until the davinci pilot's failure-rate curve; (iv) `sgt_max_nfev` 3000 | (iv) yes; (i)-(ii) no (Block 14); (iii) is a deferral |
| D-057 | (i) `failure_rate_ignore = ()`: every row of the table enters the failure rate, no screen removing any; (ii) the $\sigma_{\rm fit}$ study dropped, one table at 0.099 um; (iii) **[user]** the phantoms go through exactly the real-data pipeline, plane rule and profile extent included | no, no, Block 13's oracle |
| D-058 | each phantom's branch radius $r_{\rm br} = d/2 + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma_{\rm br}^2)$; real nodes keep Allen's radius $r_{\rm A}$ | no (Block 14); $\sigma_{\rm br}$ and three rules open (§5.2) |
| D-059 | $G$ independent draws $(d_g, \varphi_g, \theta_g, \mu_g)$ over the D-024 laws, each realized $n_{\rm real}$ times with fresh nuisances; per draw $\bar y_g = n_{\mathcal S,g}^{-1}\sum_{r:\,(g,r)\in\mathcal S} \hat d_{g,r}/d_g$ and the failure share $1 - n_{\mathcal S,g}/n_{\rm real}$, $n_{\mathcal S,g}$ the realizations of draw $g$ in $\mathcal S$; the phantoms' $\hat\alpha$ checked against the real cells' | no (Block 14); $G$, $n_{\rm real}$, the check's form open |
| D-060 | the spline through every realization in $\mathcal S$ at its own $x_{g,r} = (\ln d_g, \varphi_g, \ln\hat\alpha_{g,r})$ with $y_{g,r} = \hat d_{g,r}/d_g$, so that $\hat b(x \mid \mathcal C, \mathcal S)$ estimates $\mathbb E[\hat D/d \mid d, \varphi, \hat A = a, \mathcal C, \mathcal S]$ and a real node is inverted at its own $\ln\hat\alpha_i$; the per-draw means serve the scatter and the checks | no (Block 14); the CV folds open |
| standing | D-024 (v) the smoothing thin-plate spline; D-030 `dark` a label only, never a reason to leave $\mathcal S$; D-031 the third axis; D-035 the pilot's $k^*$ by gradient energy; D-020 documents in `docs/`; D-021 open choices as configuration fields; D-022 layout and branch | D-030: no -- `dark` is still in `SELECTION_FLAGS` (`phantoms.py` l. 29) **[src]** |

What the 2026-10-16 handoff's questions (its §7) became:

| Q | Now |
|---|---|
| Q1 the phantoms' radius | D-053 (i), superseded by D-058 ($d/2$ plus zero-mean noise) **[user, 2026-10-18 15:44]**; values and rules open, §5.2 items 1-4 |
| Q2 which bound hits fail $\mathcal S$ | D-053 (ii), coded in Block 12 |
| Q3 `failure_rate_ignore` | D-057 (i), `()`; not coded |
| Q4 the $\hat\alpha$ axis | coordinate $\ln\hat\alpha$ D-053 (iii); range scaling D-054 (i); third bandwidth 0.3 D-054 (ii); the budget is Q6; the coverage check's form open (§5.2 item 9) |
| Q5 the $\sigma_{\rm fit}$ axis | D-057 (ii), dropped; not coded |
| Q6 budget and target SE | open: $G$ and $n_{\rm real}$ after Block 13's per-replicate time is measured; the target $\epsilon_{\rm MC}$ and the davinci CPU-hours never asked (§5.2 item 5) |
| Q7 thin nodes | open **[KB: D-057]** |
| Q8 profile length and $d_{\max}$ | open; $d_{\max}$ still the TEMPORARY 4 um **[KB: D-057; src]** |
| Q9 plane range and depth jitter | open; `jitter_z_um` 0 **[KB: D-057; src]** |
| Q10 the renderer's kernel shape | open, I-002 **[KB: D-057]** |
| the failure-rate threshold | D-054 (iii): 0.2 until the pilot's curve is read |
| the per-solve budget | D-054 (iv): 3000, coded |

## 4. Today's code against the decisions

| Component | Today at `d643aaf` **[src]** | Required | By | Block |
|---|---|---|---|---|
| Plane of the final fit | the pilot's $k^*$ | the Allen-guided iteration ((H6) of the 2026-10-16 handoff) after the pilot | D-052.3 | 13 |
| Final fit | `fit.fit_profile`, Gaussian at $\sigma_{\rm fit}$, $(d, \mu, v_0)$ | `fit_sgt.fit_plane(...).final` at $k^{\rm it}$ | D-052.2 | 13 |
| Iteration code | in `scripts/plane_differences.py`; `node_stack` built on `survey.node_planes` (the real-data SWC machinery) | in `src/allen_diameter/analysis/`, fed by `measure_node`'s branch, provider and pilot; the scripts import it | D-052, D-057 (iii) | 13 |
| Plane range requested | $\pm 3$ planes widened for tilt, one provider call per node | $K_j = k_{{\rm SWC},j} \pm 6$ and squares up to $\pm 1.5\,\hat d + 2$ px | D-052.3 | 13 (§5.1 (a)) |
| `NodeResult` | Block 5's columns | plus the iteration's and the SGT fit's (§6.1) | D-052 | 13 |
| Estimator hash | `1bb3585b4a4f1373` | changes again with the iteration's fields; every older table refused | procedure §3.11 | 13 |
| $\mathcal S$, dark | `dark` in `SELECTION_FLAGS` | a label only, for phantoms and real nodes | D-030 | 14 |
| Failure count | `failure_rate_ignore = ("dark",)` (config l. 160; also in `configs/gate2_reduced.json`) | `()` | D-057 (i) | 14 |
| Table inputs | `table_inputs(d, phi)` -> $(\ln d, \varphi)$, unscaled | `table_inputs(d, phi, alpha_hat)` -> $(\ln d, \varphi, \ln\hat\alpha)$; each divided by its range, the three ranges stored in the table file and applied to every query | D-053 (iii), D-054 (i) | 14 |
| Bandwidths | two values; `validate` refuses any other count | `(0.15, 5.0, 0.3)` in raw coordinates; `validate` requires three | D-054 (ii) | 14 |
| Inversion | `invert_node(d_hat, phi, table, cfg)` | `invert_node(d_hat, phi, alpha_hat, table, cfg)`; `correct_nodes` passes `res.alpha_hat` | D-053 (iii) | 14 |
| Phantom radius | `Branch.from_points(xyz, 0.5 * draw.d_um)` (`phantom_branch`) | $r_{\rm br} = d/2 + \epsilon$ | D-058 | 14 |
| Design | one realization per replicate index $n$, `default_rng([seed, n])` | $G$ draws x $n_{\rm real}$ realizations; D-059 proposes `default_rng([seed, g])` for the draw and `default_rng([seed, g, r])` for its realizations; rows carry `draw_index`, `realization_index` | D-059 | 14 |
| Spline rows | the retained replicates at their true $(\ln d, \varphi)$ | every realization in $\mathcal S$ at $x_{g,r}$; `fit_table` also returns the per-draw table ($\bar y_g$, within-draw SD, $n_{\mathcal S,g}$, failure share) | D-060 | 14 |
| CV folds | a random permutation of the rows (`choose_smoothing`, folds seeded by `phantom.seed`) | open, folds by draw proposed (§5.2 item 6) | D-060 | 14 |
| $\sigma_{\rm fit}$ study | `MeasureConfig.sigma_fit_study_um` (l. 112), its membership check in `validate` (l. 329-331), its exclusion from `estimator_signature` (l. 385) and `config.with_sigma_fit` (l. 508, which refuses a value outside the set); the column `d_tilde_sigma_spread_um` in `analysis/cell.py` (l. 107-117); notebook Cell 8's loop that writes `config_sigma*.json` (`phase2_colab.py` l. 990-992, and `phase2_colab.ipynb`), the davinci notes for the study tables (l. 1004, 1026-1027) and Cell 10c, which runs `run_cell.py` on them (from l. 1057), and the runbook; in the suites `test_smoke_config.py` l. 69, 128, 156, 177-179 and the refusal at l. 264-269, `test_smoke_fit.py` l. 220, and `test_smoke_invert.py` l. 50, 168 and `test_smoke_table.py` l. 58, 185, which use `with_sigma_fit(cfg, 0.125)` only to make a configuration of another estimator hash and need another estimator field for that | removed from code, SPEC, notebook and runbook; `sigma_fit_um` = 0.099 um stays | D-057 (ii) | 14, or a small commit before it |
| Chunking | `build_table.py run --start --stop` over replicate indices; `build_table.pbs` splits `DIAM_N_TOTAL` (2000) over `-J 0-99` | by draw; the merge groups by draw | D-059 | 14 |
| Gate 2 configuration | `configs/gate2_reduced.json`: two bandwidths, `failure_rate_ignore` `["dark"]`, no `sgt_*` fields, `n_replicates`; loaded as `build_table.load_config` loads it (`config_from_dict`), the missing fields take their defaults and the estimator hash is the default's, `1bb3585b4a4f1373` **[run, sandbox]** | the new schema | D-054, D-057, D-059 | 14 |
| Test phantoms of gate 2 | `end_to_end.py phantoms --alpha-range` keeps them below the dark flag | not needed once `dark` is a label (D-030); one realization per test phantom, as a real node | D-030 | 14 |
| Gate 2 | passed 2026-10-06 at `535847535e83cf7d` | re-run under the new hash | D-052 | gates |
| Production job | `build_table.pbs`: `-J 0-99`, 1 core, 4 GB, 12 h; never run on davinci | resized after Q6 | D-022 | gates |

Three discrepancies between documents and code, to know about:
- D-057 names `scripts/build_table.py --sigma-fit` and `scripts/run_cell.py --sigma-fit` in its Binds and in its
  implications (ii), and "the `--sigma-fit` loops" in (D-057.2); neither flag exists at `d643aaf`: the study runs
  through Cell 8's configuration files and Cell 10c **[src; marked "[corrected 2026-10-19]" in the log's D-057]**.
- SPEC §2.2 (l. 118-121) says the $\bar B$ mask covers "any SWC segment in the block"; SPEC Block 5 (l. 408) and
  `node_pipeline._bbar` mask only the node's own Branch (on real data the stretch and its parent node), so SPEC
  contradicts itself -- the deviation the 2026-10-16 handoff's §6 found **[src]**.
- SPEC §8 (l. 614) says `profile_nan` and `bbar_few` "leave no fit"; a `bbar_few` node is fitted all the same
  (`measure_node` fits whenever the profile is finite and $\bar B > 0$, l. 254-258, and sets `bbar_few` apart,
  l. 261-262), so its row carries an $\hat\alpha$ **[src]**.

## 5. Open points

### 5.1 For Block 13 (implementation; the assistant proposes, the user confirms)

**(a) The provider call and the squares.** `measure_node` makes one provider call per node, sized for the pilot's
$\pm 3$ planes (widened for tilt) and the measurement's block (`block_half_um` 5 um) (SPEC Block 5) **[src]**.
The iteration needs $K_j = k_{{\rm SWC},j} \pm 6$ planes and squares up to $\pm 1.5\,\hat d + 2$ px, made larger
during the run when a later line outgrows them (`restack` in `gradient_iteration`; on real data cut from the
cached block while it lies inside it, fetched beyond) **[src]**. Options: (A) one call covering $\pm 6$ planes and
the block, plus one more call per restack; (B) one call sized for the largest line the box allows
($1.5 \times 6$ um + 2 px), which renders large squares for every phantom. Recommendation: (A) **[reasoning]**:
it renders what the real data would fetch, and the extra calls are the cost the real nodes pay too (D-057 (iii)).
Today's one call already requests 7 planes for a flat phantom, 15 at $\varphi$ = 10 deg, 27 at 30 deg, 43 at 60 deg
and 47 at 80 deg (the pilot's window of 7 nodes, each $\pm 3$ planes widened for tilt) **[run, sandbox:
`node_pipeline._plane_range` at the defaults]**, so $K_j$'s 13 planes add planes only to near-flat phantoms; the
cost grows mainly with the larger squares of thick nodes and with one SGT fit per round (`fit_plane` took 1.5-9 s
per profile on one core, **[src: SPEC Block 12's status]**), which is what Q6 must measure **[reasoning]**.

**(b) Block 10's selection.** `analysis/calibration.py`'s `scan_node` calls `measure_node` (l. 222);
`calibration_reasons` keeps a node when it passes Block 6's $\mathcal S$ (`phantoms.reject_reasons`) and is thin
($\hat d \le$ `node_dhat_max_um`), flat and faint ($\hat\alpha \le$ `node_alpha_max`), and the scan then reads the
planes $k^* \pm P$ **[src]**. After Block 13, $\hat d$, $\hat\alpha$ and `fit_status` are the SGT fit's at
$k^{\rm it}$, so the selection moves. (Block 14's change to $\mathcal S$ does not move it at the defaults: every
`dark` node, $\hat\alpha > 1.0$, already fails `node_alpha_max` = 0.5, config l. 139, 244 **[src]**.) Options: (A)
keep the pilot's values in `NodeResult` (`k_star`, `d_pilot_um`, `alpha_pilot`, the pilot's status) and point Block 10
at them, so its selection does not move; (B) let Block 10 select on the new values and re-run its suite against
the new numbers. Recommendation: (A) **[reasoning]**: Block 10 calibrates the renderer's defocus kernel on plane
scans of thin, flat, faint nodes, a choice of nodes that the per-node estimator should not change unannounced.

**(c) The diagnostics that read `NodeResult` as the pilot.** `scripts/plane_differences.node_stack` (l. 167) calls
`survey.node_planes`, which re-measures the node with `measure_node` (`survey.py` l. 197) and keeps its first
provider call (l. 198); the suites then assert that the pilot's fit equals the `NodeResult`'s
(`test_smoke_plane_iteration.py` l. 217-219: `refit_at_plane` at `r.k_star` against `r.d_hat_um`, `r.mu_hat_per_um`,
`r.v0_hat_um`, `r.B_bar`, `r.fit_status`; l. 281) or size their lines from `result.d_hat_um`
(`test_smoke_plane_gradient.py` l. 434, 451, 474) **[src]**. Once `measure_node` runs the iteration and the SGT fit,
those values are the new estimator's, and `node_stack` would run the iteration once more inside its own
re-measurement. Options: (A) split `measure_node` into the pilot (a function of its own) and the iteration with the
final fit, `measure_node` calling both; `survey.node_planes` and the diagnostics call the pilot only, so the suites
keep comparing pilot with pilot; (B) keep one function and let the diagnostics read the pilot's columns
(`d_pilot_um` and the rest), updating the suites. Recommendation: (A) **[reasoning]**: one code path for phantoms
and real nodes (D-057 (iii)) without a switch, and no second iteration inside the diagnostics.

**(d) Which plane the flags and $\mathcal S$ describe.** Today `faint` and `crossing` are computed on the profile at
$k^*$ (`_profile_flags`, `node_pipeline.py` l. 258), `stack_edge` from $k^*$'s place in its range (l. 259-260) and
`bbar_few` from the pilot's $\bar B$ (l. 261-262) **[src]**. After Block 13 the deliverable fit is at $k^{\rm it}$, and
the iteration can end on the first or last plane of $K_j$ (node 2 ended on plane 113, "the first of the 13 planes",
D-043 (b)) **[KB]**. Options: (A) the flags of $\mathcal S$ computed on the profile and plane actually fitted,
$k^{\rm it}$ (with `stack_edge` when $k^{\rm it}$ is an end plane of $K_j$), the pilot's kept as labelled columns;
(B) the pilot's, as today. Recommendation: (A) **[reasoning]**: $\mathcal S$ has to describe the fit that enters the
table and the inversion.

Not questions: $m$ = 3 (D-044), at most 5 rounds (D-042) and $K_j = k_{\rm SWC} \pm 6$ (D-052.3) are decided; the
background margin of 4.0 um (D-036, the assistant's PROVISIONAL value, its margin left open there) and D-042's
PROVISIONAL readings (a repeat of any earlier plane ends the iteration; round 1 counts among the 5) stay the
defaults, open per D-052 (h) **[KB]**; the pilot's path pass keeps Block 5's Gaussian (D-052 (e), open, the
default unchanged since D-035).

### 5.2 For Block 14 (the user's to decide)

| # | Point | Options, or the assistant's proposal | Evidence |
|---|---|---|---|
| 1 | $\sigma_{\rm br}$ of D-058 (um) | a value; one way to choose it: the spread of Allen's $2r_{\rm A}$ about the pilot's $\hat d$ on the 393 pilot nodes (`MyDrive/diameters/pilot/pilot_529878215.csv` **[KB: HPC paths reference §1]**) **[reasoning]** | Allen's $2r_{\rm A}$ is about 0.5 um in every bin of $\hat d$, a systematic, not zero-mean, error -- the reservation recorded in D-058's Why **[KB]** |
| 2 | where $r_{\rm br}$ enters | (A) all four places of the 2026-10-16 handoff's §6 (the iteration's start, the pilot's focus window, the $\bar B$ mask, `block_background`'s mask); (B) the iteration's start only. Recommendation (A): D-057 (iii) lets a phantom differ from a real node only by its inputs, and a real node's one radius reaches all four **[reasoning]** | -- |
| 3 | a draw with $r_{\rm br} \le 0$ | (A) redraw $\epsilon$ (a truncated normal; its mean is no longer $d/2$); (B) a floor (an atom at the floor) | $P(r_{\rm br} \le 0)$ at $d = 0.2$ um: 0.0228, 0.159, 0.309 for $\sigma_{\rm br}$ = 0.05, 0.1, 0.2 um; at $d = 0.4$ um: 3e-5, 0.0228, 0.159; 23 % of the log-uniform design lies below 0.4 um **[run, sandbox: `scipy.stats.norm`]** |
| 4 | $\epsilon$ per draw or per realization | per realization: the radius error is a nuisance, and D-059 asks each element of the table to have the nuisances marginalized away **[reasoning]**; per draw: every realization of a cylinder starts from the same wrong radius, as a real node's tracing would | -- |
| 5 | $G$, $n_{\rm real}$ (Q6) | after Block 13's per-replicate wall time is measured on about 10 phantoms; also the target $\epsilon_{\rm MC}$ and the davinci CPU-hours (never asked) | at an equal number of measured replicates, 300 draws x 5 gave a table SD 1.062 times that of 1500 independent draws without a draw-level effect and 1.335 times with one ($c$ = 0.5; the design effect predicts 1.342) **[run, sandbox: `check_grouped_design.py`]** |
| 6 | the CV folds | folds by draw (every realization of a draw in one fold); plain folds score the spline on near-copies of its training rows **[KB: D-060]** | on 300 draws x 5 with a draw-level offset: plain 5-fold CV chose $\lambda_s$ 0.01-0.1 in all 6 seeds, folds by draw 10-100 in 5 of 6; mean RMS error of the table 0.0752 against 0.0295 **[run, sandbox: `check_tps_deep.py` T7]** |
| 7 | the smoothing grid's upper end | extend `spline_smoothing_grid` beyond 100 (e.g. to $10^4$) and flag a choice at either end **[reasoning]** | under folds by draw the choice was 100, the grid's largest value, in 2 of 6 seeds (T7), and in 1 of 20 one-realization tables (`check_table_se_coverage.py`, hetero) **[run, sandbox]** |
| 8 | the failure rate's $\ln\hat\alpha$ for realizations without a usable fit: no fit at all (`profile_nan`, $\bar B \le 0$), a failed fit, or a collapse whose $\hat\alpha$ is set by the $\mu$ bound (a `bbar_few` row is fitted, §4) | the assistant's proposal: the true $\alpha_g = \mu_g d_g/\cos\varphi_g$ for every realization, evaluated at a real node's $\hat\alpha_i$ **[src: SPEC §8]** | -- |
| 9 | the form of D-059's coverage check, and what follows if it fails | proposal: fitted against fitted, per bin of $\hat d$ (the real $\hat\alpha_i$ against the phantoms'); if it fails, a wider $\mu$ range or a support flag at the real node **[KB: D-059; src: SPEC §8]** | a node can lie inside the phantoms' overall $\hat\alpha$ range and outside it at its own size (a 0.5 um node at 4 /um: $\hat\alpha$ 2.03, outside the band at 0.5 um) **[run, sandbox: `check_table_se_coverage.py` C5]** |
| 10 | which SE the table reports | (A) keep $\widehat{\rm SE}(x) = \hat\tau(x)/\sqrt{N^{\rm loc}_{\rm eff}(x)}$ of SPEC §2.4, $N^{\rm loc}_{\rm eff}(x) = (\sum_n \varpi_n(x))^2/\sum_n \varpi_n(x)^2$; (B) the spline's own $\sqrt{\sum_n \ell_n(x)^2\,\hat\tau^2(x_n)}$; (C) a bootstrap over draws with the CV re-run. Under D-060 (A) also counts a draw's realizations as independent, so it misses the design effect $1 + (n_{\rm real} - 1)\rho$, $\rho$ the within-draw correlation of the rows' errors **[reasoning]** | at a fixed smoothing (A) is 1.2-2.8 times the spline's own SD **[run, sandbox: `check_table_se_coverage.py` C2]**, but the CV choice leaves a smoothing bias at the thick end larger than (A): -0.0033 against 0.0023 at $d$ = 3 um, $\varphi$ = 45 deg, $\hat\alpha$ = 4, over the 14 tables that chose $\lambda_s$ = 10 (-0.0029 over all 20) **[run, sandbox: `check_table_steps45_extra.py` A; `check_table_se_coverage.py` C3]** |
| 11 | an `off_support` flag | proposal: on the kernel mass $\sum_n \varpi_n$ at the root, below the replicates' own leave-one-out 5th percentile **[src: SPEC §8 (b)]** | the dark thin node above was inverted with no flag at mass 2.05 against a 5th percentile of 3.06 (root 0.492 um for a true 0.5 um); at fixed $(\varphi, \hat\alpha)$ the phantoms cover at most 77 % of the bracket in $\ln d$ **[run, sandbox: `check_table_se_coverage.py` C4, C5]** |
| 12 | the $\ln\hat\alpha$ range of D-054 (i)'s normalisation | D-054.1 takes the retained rows' min and max (the assistant's PROVISIONAL reading); proposal: a percentile or a nominal range **[src: SPEC §8 (c)]** | the range is set by near-vertical draws: on one table the draw with the largest $\ln\alpha$ has $\varphi$ = 89.68 deg, and the 1st-99th percentile span is 69 % of the full one **[run, sandbox: `check_table_steps45_extra.py` D]**; over 10 tables that share is 50-77 %, 86-98 % of the rows lie in the lower half of the normalised axis, and the largest tilts drawn lie within 0.14 deg of 90 deg **[run, sandbox: `check09_lnalpha_span.py`]** |
| 13 | one $\lambda_s$ for a heteroscedastic table | proposal, untested: a per-point smoothing proportional to $\hat\tau^2(x_n)$ (`RBFInterpolator` accepts an array) **[src: SPEC §8 (d)]** | with the thin end's noise, CV chose 10 in 14 of 20 tables, against 1 in 18 of 20 with a constant $\tau$ **[run, sandbox: `check_table_se_coverage.py`]** |
| 14 | `high_failure` at 0.2 | not now: D-054 (iii) sets it on the pilot's curve; the pilot must read it knowing the noise | at $N^{\rm loc}_{\rm eff}$ 15-28, the strict flag at 0.2 fires for 45-47 % of nodes whose true failure probability is 0.2 and for 5-11 % at 0.1 **[run, sandbox: `check_table_steps45_extra.py` F]** |
| 15 | D-053 (i)'s sensitivity check, with its in-focus images | still run as a diagnostic, or drop (D-058's open point) | -- |
| 16 | still open from the 2026-10-16 handoff | Q7 (thin nodes: the default only, or also the pinned-kernel comparison), Q8 ($d_{\max}$, the profile's length), Q9 (`jitter_z_um` from $\Delta z^*$), Q10 (the renderer's kernel shape); the real-data check before production (that handoff's §7, last paragraph); the SGT quadrature (SPEC §8, first bullet, standby) | -- |

### 5.3 Settled findings that need no decision

- **The kernel.** `RBFInterpolator(kernel="thin_plate_spline")` evaluates $\varrho^2\ln\varrho$ in any dimension. With
  two inputs (the table at `d643aaf`) it fits the thin plate; with three (Block 14) the same call fits a spline of
  order 5/2, not the three-dimensional bending-energy spline, whose kernel is $\varrho$ (`kernel="linear"` with
  `degree=1`) **[doc: TEEG_thin_plate_spline_in_3D §3.7, §3.13 (a); run, sandbox: its `check02`]**. On the
  one-realization fixture `thin_plate_spline` had the lower RMS error in 9 of 10 tables (median 8 %) and the lower
  slope error in 9 of 10 (median 10 %; sign test $p$ = 0.021 each), and it gives a $C^1$ table where `linear` puts a
  cone at every phantom **[run, sandbox: `check07_kernel_comparison.py`, `check07b_paired.py`, `check05_smoothness.py`]**.
  The comparison has not been run under D-059/D-060. D-024 (v) stands.
- **$\lambda_s$ values do not carry over** from the two-input table: in a new dimension and normalisation the same
  number multiplies a different penalty; CV chooses afresh **[doc: §3.11; reasoning]**.
- **The smoothing matters more than the kernel:** within one table the CV choice cost a factor 3.45 in RMS error
  (0.0447 at $\lambda_s$ = 0.1 against 0.0130 at 10), against a median 8 % between kernels **[run, sandbox: `check07b_paired.py`]**.
- **The cost of a fit is dense:** 0.27, 2.85 and 24.8 s at $N$ = 1500, 3000 and 6000 rows (exponent 3.1), dense
  matrix 0.02, 0.07, 0.29 GB, in the sandbox (2 cores, BLAS threads not pinned) **[run, sandbox: `check_tps_deep.py`
  T6]**. A 5-fold CV over the 9-value grid is 45 fits at 0.8 $N$ rows: about 1 min at $N$ = 3000 and 9 min at 6000 on
  the same machine **[reasoning, from T6]**; on a one-core davinci task, re-time it. Under D-060 the spline's $N$ is the
  number of realizations in $\mathcal S$, $\sum_g n_{\mathcal S,g} \le G\,n_{\rm real}$.
- **The inversion is classical calibration**, correctly: its bias at a given $d$ (the Jensen term of mathematics
  §3.6: at $d$ = 1.5 um with a ratio SD of 0.2 the mean correction was -0.0021 relative, the term predicts -0.0019
  **[run, sandbox: `check_table_inversion.py` C1]**) does
  not depend on the population of real diameters, while regressing $d$ on $\hat d$ over the phantoms returns the
  design's posterior mean, which does: with a linear $m$ (no Jensen gap) the classical inverse was unbiased on two
  populations unlike the design (-0.00035 $\pm$ 0.00046 on the thin one) and the inverse estimator was off by -0.0397
  on the thin one and +0.00008 on the thick one **[run, sandbox: `check_inverse_vs_classical.py`]**; that the
  classical inverse's bias is population-free for any $m$ is **[reasoning]**.
- **The table's error propagates through its own slope:** a noise-free node inverted with an estimated table has
  $(\tilde d - d)/d = -(\hat b(x \mid \mathcal C, \mathcal S) - b(x \mid \mathcal C, \mathcal S))/\hat m'(d)$ to first order,
  $\hat m'(d) = \partial\hat m(d, \varphi, \hat\alpha \mid \mathcal C, \mathcal S)/\partial d$ at fixed $(\varphi, \hat\alpha)$,
  $b(x \mid \mathcal C, \mathcal S)$ the fixture's true bias factor (B2: 35 of 35 within the 5 % tolerance fixed
  before the run). With the fixture's true slope $b(1 + \beta_b)$, $\beta_b = \partial\ln b/\partial\ln d$, in place of
  $\hat m'(d)$, 29 of 35 fall within the same 5 % and the misses, up to 13 %, are at the two thinnest query points
  (B's own tolerance, $2(\hat b - b)^2 + 10^{-6}$, passed 16 of 35) **[run, sandbox: `check_table_steps45_extra.py` B,
  B2; the 29 of 35 recomputed from B's log]**.
- **What D-060's table estimates:** on a toy of the D-059/D-060 design with a closed form (within-draw correlation
  -0.9 of the $\hat d$ and $\hat\alpha$ errors), the rows at $\ln\hat\alpha = l$ average to $\mathbb E[y \mid \ln\hat\alpha = l]$
  (T1, PASS), which differs from the draws' own means averaged at $l$ by -0.129 at the top edge of the $\ln\alpha$
  range and by 0.000 inside it **[run, sandbox: `check_b_target.py`]**.

## 6. Block plan

Block numbers continue SPEC.md (Blocks 0-12 exist).

### 6.1 Block 13: the Allen-guided iteration inside Block 5 (D-052.3, D-057 (iii))

- New module (e.g. `analysis/plane_iteration.py`): `iterate_planes`, `check_iteration`, `ITER_STATUSES`,
  `gradient_iteration`, `refit_at_plane` (its fit Block 12's `fit_sgt.fit_plane`), `line_profiles`,
  `block_background` with `far_from_dendrites` and `plane_background`, and a square builder fed by `measure_node`'s
  branch, provider and pilot instead of `survey.node_planes`. `scripts/plane_differences.py` imports them, so
  there is one implementation; the suites of Cells 4j-4m (`plane_iteration`, `iteration_fits`, `fit_variants`,
  `variant_iteration`) and `plane_gradient` must still pass, comparing pilot with pilot (§5.1 (c)).
- `measure_node`: the pilot unchanged (D-035's $k^*$, the centres, the TLS direction, $\varphi_i$); then the
  iteration for node $i$ from $d^{(0)}_i = 2\,r_{{\rm br},i}$ (`branch.radius_um[i]`: Allen's on a real node, the
  phantom's under D-058) over $K_i = k_{{\rm SWC},i} \pm 6$; then `fit_sgt.fit_plane(v, I, phi, B_bar, cfg.measure)`
  at $k^{\rm it}_i$, $\bar B_i(k^{\rm it}_i)$ the D-018.1 masked median of that plane (`node_background`, as
  `refit_at_plane` does). The deliverable is `.final` (IRLS); `.unweighted`, `.gauss_fitted`, `.gauss_fixed` and the
  pilot's Gaussian at $k^*$ are labelled comparisons (D-052). The split of §5.1 (c) and the flags' plane of
  §5.1 (d) as confirmed.
- `NodeResult` gains `k_iter`, `iter_status`, `n_rounds`, `n_restacks`, `w_hat_um`, `beta_hat`, `xi_hat`, `zeta_hat`,
  `irls_ok`, `at_bound`, `d_pilot_um` (and per §5.1 (b) `alpha_pilot`); `d_hat_um`, `mu_hat_per_um`, `v0_hat_um`,
  `alpha_hat` and `fit_status` come from `.final` (its status is D-053 (ii)'s); `run_replicate`'s row and the
  per-node CSV of `analysis/cell.py` carry them.
- `MeasureConfig`: the iteration's fields with `# source:` tags (line multiplier 3.0 D-044, rounds 5 D-042, plane
  half-range 6 D-052.3, background margin 4.0 um, the start rule), all in the estimator hash.
- Oracles: on `tests/smoke/fixtures_cell.py` the moved iteration with the old fit reproduces the script's
  `gradient_iteration` exactly; phantoms and real nodes share one function object, and `measure_node` has no
  argument or configuration flag that a phantom sets differently from a real node beyond its inputs (D-057 (iii));
  the estimator hash differs from `1bb3585b4a4f1373` and from `535847535e83cf7d`, and a table of either is refused; a
  rendered phantom goes through pilot, iteration and SGT fit end to end with $\hat d/d$ reported; the defaults
  asserted; the per-replicate wall time measured on about 10 phantoms over $d$ and $\varphi$ (feeds §5.2 item 5).
- SPEC: Block 5's entry (the plane rule, the provider's plane range), Block 8's columns, Block 10's data flow,
  §2.2 (the plane of the final fit), and the $\bar B$ mask deviation of §4 recorded.

### 6.2 Block 14: Blocks 6-7 under D-030, D-031, D-053 (iii), D-054, D-057 to D-060

- $\mathcal S$: `dark` leaves `SELECTION_FLAGS` (D-030); `failure_rate_ignore = ()` (D-057 (i)).
- Design (D-059): `PhantomConfig` gains `n_draws` ($G$) and `n_realizations` ($n_{\rm real}$); the draw's features
  and each realization's nuisances from their own generators (D-059's proposal: `default_rng([seed, g])`,
  `default_rng([seed, g, r])`); rows carry `draw_index`, `realization_index`; `build_table.py` and `build_table.pbs`
  chunk by draw; the merge groups by draw and still refuses rows of another `sim_hash`.
- Radius (D-058): `PhantomConfig.radius_noise_sd_um` ($\sigma_{\rm br}$), $\epsilon$ from a stream of its own
  (D-058's proposal) whose key cannot equal a realization's: D-058's example `default_rng([seed, n, 1])`, with $n$
  become $g$, is D-059's key of realization $r$ = 1, `[seed, g, 1]`; e.g. `[seed, g, r, 1]` per realization
  **[reasoning]**; the scope, the $r_{\rm br} \le 0$ rule and per draw or per realization as §5.2 items 1-4 decide;
  rows carry `radius_um`.
- Table (D-053 (iii), D-054, D-060): `table_inputs(d, phi, alpha_hat)`; the three ranges stored with the table and
  applied to every query; three bandwidths in raw coordinates; the spline through every realization in
  $\mathcal S$ at $x_{g,r}$; the per-draw table; the folds and the grid as §5.2 items 6-7 decide.
- Inversion: `invert_node(d_hat, phi, alpha_hat, table, cfg)`; an $\hat\alpha_i$ outside the table's range flagged
  like `phi_out_of_domain` **[reasoning; proposal]**; `off_support` if §5.2 item 11 is accepted.
- D-057 (ii)'s removals (§4), if not done before.
- `configs/gate2_reduced.json` in the new schema.
- Oracles: the fixture of `test_smoke_table.py` extended to $b = 1 + 0.06/d + 0.15\sin^2\varphi + 0.05\ln\alpha$,
  recovered within 4 SE; the scaling's file round trip reproduces $\hat b$ exactly, and the inputs scaled by $c$ at
  smoothing $\lambda_s$ reproduce the unscaled fit at $\lambda_s/c^2$ (D-054's C1 identity, checked to 2.3e-13 in
  `check_tps_scaling.py`); three bandwidths required; every row counted in the
  failure rate; a node with $\hat\alpha > 1$ corrected, not filled, with `dark` set (D-030); the radius draw's law by
  Kolmogorov-Smirnov (mean 0, SD $\sigma_{\rm br}$) and the branch carrying it (D-058); draws reproducible per
  $(\text{seed}, g)$ and realizations per $(\text{seed}, g, r)$, a draw's realizations sharing its features
  (D-059); the spline sees $\sum_g n_{\mathcal S,g}$ rows and, on a fixture with a known within-draw $\hat d$-$\hat\alpha$
  correlation, recovers $\mathbb E[\hat D/d \mid \hat A = a]$, not the draw-mean relation (D-060; `check_b_target.py`
  has the closed form); folds by draw, if decided, beat plain folds on T7's fixture.

### 6.3 Dependencies

Block 13 before Block 14 (Block 14's rows need the iteration's $\hat\alpha$); Block 13's wall time before §5.2
item 5; §5.2 items 1-4 and 6-13 before Block 14's logic; the davinci pilot before D-054 (iii)'s threshold.

## 7. Gates

1. Every suite, then a clean clone of the pushed branch (as for `d643aaf`: byte checks, `compileall`, the hashes,
   `test_smoke_config`, the changed suites).
2. **Gate 2**, reduced, sandbox, new hash (`configs/gate2_reduced.json` in the new schema; no `--alpha-range` once
   `dark` is a label). The commands of the 2026-10-16 handoff's §8.4, with the run chunked by draw (the arguments as
   Block 14 defines them):

   ```
   python scripts/build_table.py run --start 0 --stop <G> --workers 2 --out-dir T --config-json configs/gate2_reduced.json
   python scripts/build_table.py merge --out-dir T --config-json configs/gate2_reduced.json
   python scripts/end_to_end.py phantoms --d 0.5,1,2,3 --reps 8 --seed 777 --out T/test.csv --config-json configs/gate2_reduced.json
   python scripts/end_to_end.py evaluate --table T/bias_table_<hash> --rows T/test.csv --config-json configs/gate2_reduced.json
   ```

   Pass: every mean $\tilde d/d$ within 0.10 of 1 (`--tol` 0.10). Same simulator for table and test: this tests the
   Monte Carlo, the estimator and the inversion, not the simulator (procedure §3.10).
3. **davinci** (`hpc-git-delivery`, `hpc-python-compat`): the byte checks, `qsub -v DIAM_DRYRUN=1`, the two-replicate
   probe, a pilot of about 200 replicates read for $\hat\tau$ and $\hat p_{\rm fail}(x \mid \mathcal C)$ by region
   (§5.2 items 5 and 14, D-054 (iii)), then production and the merge (commands: HPC paths reference §2.1 **[KB]**).
   `build_table.pbs`'s knobs change with the chunking by draw.
4. Procedure §3.10 on the production table: blur consistency, $\sigma_{\rm r}(0)$, kernel shape (Q10), $\mu$
   percentiles, heading bins, end-to-end with an alternative generator; D-059's $\hat\alpha$ coverage check against
   the real cells.
5. Before production, the real-data check of the 2026-10-16 handoff's §7: D-052's fit on Allen's thick nodes in
   Colab, $\hat w$ and $\hat\kappa$ in focus against phantoms of the same $\hat d$.

## 8. Evidence produced by the table chat

All in `checks/table_2026-10-18/` (each script with the log a correct run prints; how to run them in its
`README.md`), except the thin-plate document's own scripts (last row). Fixtures are synthetic: an analytic $b$,
Gaussian scatter, and $\hat\alpha$ set to the true $\alpha$ unless stated, so none of this tests the renderer or the
SGT fit on rendered stacks.

| Script | Claim tested | Result **[run, sandbox]** | Feeds |
|---|---|---|---|
| `check_tps_scaling.py`, `check_tps_scaling_seeds.py` (2026-10-16) | how the inputs' scaling changes the spline | an isotropic rescale equals a change of $\lambda_s$; one axis x 10 at fixed $\lambda_s$ moved $\hat b$ by up to 0.055; range-normalised RMS 0.0073-0.0100 against 0.0092-0.0106 raw, CV-scaled 0.0062-0.0087 at 25 times the fits (5 seeds) | D-054 (i) |
| `check_table_inversion.py` | the inversion's mean correction (Jensen term), the corrected scatter, the shortcut's error (mathematics Eq. 21) | 10 of 10 PASS | Block 7 |
| `check_inverse_vs_classical.py` | inverting $\hat m$ against regressing $d$ on $\hat d$ | classical unbiased on both populations; the inverse estimator -0.0397 on the thin one | §5.3 |
| `check_table_se_coverage.py` (hetero, homo) | C1 the spline is a linear smoother; C2-C3 the reported SE against the spline's own SD and the across-table SD; C4 the support along the inversion line; C5 a dark thin node | C1 to 7e-15; the SE 1.2-2.8 times the spline's SD; CV chose 10 in 14 of 20 tables (hetero) against 1 in 18 of 20 (homo); the support at most 77 % of the bracket; the dark node's root 0.492 um at mass 2.05 | SPEC §8 (a), (b); §5.2 items 9-11, 13 |
| `check_table_steps45_extra.py` (A-F, B2) | A the CV choice's share of the across-table SD; B, B2 the table term of the propagated error; C the reach of the spline's weights; D the $\ln\alpha$ range; E the median of 3; F the failure rate's noise | B 16 of 35 at its own tolerance (with the true slope), B2 35 of 35; C: 92 % of $\sum_n \lvert\ell_n\rvert$ beyond two bandwidths against 23 % of the kernel's, negative mass 0.104 (medians); D 69 %; E 0.670; F 21 of 21 within 10 % of the binomial SD | SPEC §8 (a), (c), (e); §5.2 items 12, 14 |
| `check_grouped_design.py` | the design effect of realizations per draw | SD ratio 1.062 (predicted 1.000) and 1.335 (predicted 1.342), PASS | D-059; §5.2 item 5 |
| `check_tps_deep.py` (T1-T7) | the spline's facts in three dimensions; T6 the cost; T7 the CV folds under D-059/D-060 | T2, T3, T5 PASS; df 13.4 at the CV choice; T6 and T7 as §5 | §5.2 items 6-7; §5.3 |
| `check_b_target.py` | what D-060's table estimates | T1, T2 PASS at the interior and the top edge; the gap to the draws' means -0.129 at the edge | D-060; Block 14's oracle |
| `check_quadrature_bias.py` | the D-052.2 quadrature's bias on the IRLS $\hat d$ (Block 12) | at most 0.071 grey levels of model error and -7.5e-4 um of $\hat d$ bias in the fitted regime | SPEC §8, first bullet |
| `docs/TEEG_thin_plate_spline_in_3D/scripts/`: `check07_kernel_comparison.py`, `check07b_paired.py`, `check09_lnalpha_span.py`, `check05_smoothness.py`, `check02_fourier_seminorm.py` | the kernel comparison on the one-realization fixture; the $\ln\alpha$ span over 10 tables; the $C^1$ table and the cones of `linear`; which kernel is which order on $\mathbb R^3$ | §5.3 and §5.2 item 12 | §5.3 |

## 9. Files delivered with this handoff

| File | Content |
|---|---|
| `docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` | this file |
| `checks/table_2026-10-18/` | the table chat's checks of §8 with their logs and the two result files that `check_table_steps45_extra.py` A reads; `README.md` with the commands. The `.py` files pure ASCII and compiled **[run, sandbox]** |
| `specs/SPEC.md` | §8's sentence on `bbar_few` rows marked "[corrected 2026-10-19]" (§4) |
| `docs/TEEG_thin_plate_spline_in_3D/` (if the user agreed) | the study document of 2026-10-18 with its `figures/` and `scripts/` (eleven checks with logs) |
| `README.md` of the workstream | rows for the new files, and a changelog row (Block 12 had none) |

## 10. Not checked

- Block 13 is not written: D-052's plane rule has never run inside `measure_node`, and D-052's fit has never run on a
  rendered phantom or on Allen's data **[open]**.
- The new chain's per-replicate wall time is unknown, so are $G$ and $n_{\rm real}$ **[open]**.
- No diameter job has run on davinci; the HPC paths reference's §4 lists what is untested there **[KB]**.
- Every table check above ran on synthetic fixtures (§8); the kernel comparison only on the one-realization design,
  not under D-059/D-060 **[run, sandbox]**.
- The real image noise (D-025) and the camera chain's black level and gain are not measured; the renderer's 3.0 gl
  and 210 gl are PROVISIONAL **[src]**.
- Block 10 and the Phase II diagnostics depend on `measure_node` (§5.1 (b), (c)) **[src]**.
- Exists, not checked for this change: `scripts/run_cell.py`, `analysis/cell.py`, `scripts/run_node.py`
  (`survey.measure_nodes`), the notebook's Cells 8-10 and the runbook (they read `NodeResult`, the table and
  `invert_node`'s arguments, which Blocks 13-14 change).
- The SGT family's standard references are named from memory in the 2026-10-16 handoff's §10 and remain unverified
  there; the kernel itself is checked numerically (Block 12's suite) **[src]**.

## 11. Working rules

- Load `scientific-coding` before writing or reviewing code; for anything that runs on davinci also
  `hpc-python-compat` and `hpc-git-delivery`.
- Read the decision log before acting on anything it constrains (`project-decision-log`); cite IDs; log a decision
  the moment the user makes it. Next free IDs on 2026-10-19: D-061, I-003.
- Commit and push only when the user agrees; never to `main`; never force-push.
- `.py` sources pure ASCII, shell and PBS files LF only.
- In the chat, displayed equations go out inline as LaTeX: the user's request of 2026-10-18 in the table chat,
  which replaced the 2026-10-16 handoff's PNG rule there **[user; KB: the log's 2026-10-18 (later, 2) row]**.
- Documents of this workstream live in `docs/` (D-020; on `sci/diameter-pipeline` until it is merged, since D-020's
  "on `main`" and the rule above cannot both hold -- the user's to settle when the branch is merged); project docs
  (the HPC paths reference, the decision log) follow DOCUMENT UPKEEP (re-read, merge, dated changelog rows,
  "[corrected <date>]" marks); end every reply with one line naming every doc written.
