# Diameter pipeline, Phase II: Colab runbook

| Date | Change |
|---|---|
| 2026-10-06 | v1. Written with Block 11 (branch `sci/diameter-pipeline`). Every script below was run in the sandbox on a synthetic cell served through `allen_image_io.fetch_zblock`; none has run on Allen's data (the sandbox gets 403 from api.brain-map.org). |
| 2026-10-07 | v2. The cells now exist as a notebook, `notebooks/phase2_colab.ipynb`, with the same numbers; the code below is the notebook's. Corrections: `CACHE_DIR` is not the cache of cells 1-12 (Drive folder listing); Cell 5 suggests `renderer.jpeg_qtables`, the tables themselves, because the old `jpeg_qtables_file` was read by nothing (SPEC Blocks 1, 4, 11); Cell 6 writes its per-node CSV to Drive (it went into the clone). Added: Cell 3a, the node list (`scripts/list_stretches.py`); Cell 4 split into a one-stretch timing run and the full pilot; a streaming `run()` that stops "Run all" on a failure; Cell 8 through `config.with_overrides`, writing one configuration per in-focus blur of D-023; Cell 9's Pillow check and the merge with the same configuration, which the merge now enforces (every replicate row carries `sim_hash`); Cell 10's $\sigma_{\rm fit}$ runs. Evidence: the notebook's offline cells ran in the sandbox against a stub `google.colab`, and every command it builds parsed with its script's own parser **[run, sandbox]**; the cells that reach api.brain-map.org have still not run anywhere. |
| 2026-10-07 (later) | v3. The focus rule is the gradient energy (D-030): Cell 2 runs `test_smoke_focus` too; Cells 4b/4c describe and show both focus curves, the summary's `k_star_vs_dip_depth` and the nodes where the two rules differ; the default estimator hash is now 6f0397236447fe6f. Evidence: `test_smoke_focus.py` 7 pass and every suite of the workstream re-run in the sandbox **[run, sandbox]**; the cells that reach api.brain-map.org have still not run here. |
| 2026-10-07 (later, 2) | v4. Added Cell 4d: plane montages of chosen nodes with Allen's reconstruction drawn on them (`scripts/node_planes.py`, SPEC Block 11); each node is measured again as in Cell 4b, so its crops come from the cache. Evidence: `test_smoke_phase2.py` 4 pass, 3 skip with the montage checks (the re-measurement equals the pilot's and makes the same request; the image extent agrees with the profile sampler to 1e-6 grey levels; the drawn trace lies on the rendered tube within 0.05 um), nine mutants of the new code killed, every suite re-run **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-08 | v5. Added Cell 4e: consecutive-plane differences around chosen nodes, the user's proposal of 2026-10-08, as a diagnostic (`scripts/plane_differences.py`, SPEC Block 11). Evidence: `test_smoke_plane_diff.py` 7 pass, 1 skip; seven mutants of the new code killed; every suite re-run; the cell ran on the synthetic cell with the fixture in place of Allen's server **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-08 (later) | v6. Cell 4e evaluates, by default, the area under the profile along the measuring line, plane by plane, and its change between consecutive planes (the user's evaluation, D-036), with the same areas background-normalised and each plane's background level beside them; the pixel-difference version stays as `--evaluation image`. Evidence: `test_smoke_plane_diff.py` 7 pass, 1 skip with the profile checks; nine mutants of the new code killed; every suite re-run **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-08 (evening) | v7. Added Cell 4f: the entropy of the grey-level histogram, plane by plane, of the samples along the measuring line and of the pixels of a strip around it (the user's proposal of 17:07; `scripts/plane_differences.py --evaluation entropy`, SPEC Block 11), with the dip of each curve framed. Evidence: `test_smoke_plane_entropy.py` 8 pass; fifteen mutants of the new code killed; every suite re-run; the cell ran on the synthetic cell with the fixture in place of Allen's server **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-08 (evening, 2) | v8. Cells 4e and 4f run over `LINE_HALF_UMS` = [3, 5], the half-length of the measuring line (the user's proposal to enlarge it; the pipeline's line stays +-3 um), each length into its own folder; the square around the node grows with the line and is fetched when it outgrows the pilot's block. Cell 4f picks each curve's global minimum (`--entropy-pick min`): on nodes 2 and 3 (run of 2026-10-08, 17:56 UTC) the curves had no W and the dip rule skipped node 2's lowest plane. The JSONs carry every plane's profile. Evidence: `test_smoke_plane_entropy.py` 8 pass and `test_smoke_plane_diff.py` 7 pass, 1 skip with the new checks; mutants of the new code killed; the cells ran on the synthetic cell **[run, sandbox]**; nodes 2 and 3 at +-3 um ran on Allen's data **[user]**. |
| 2026-10-09 | v9. Added Cell 4g: the entropy along a +-5 um line on thin dendrites, the user's request after the trunk runs (D-039): `--nodes thin` picks up to 12 pilot nodes (in S, Allen 2r <= 0.6 um, fitted d <= 1.0 um, k* equal to the dip depth's plane, flat; up to 3 per stretch), and the script writes a summary CSV and a summary figure of the picks against k*. Evidence: `test_smoke_plane_entropy.py` 8 pass with the selection worked out by hand on a seven-stretch SWC; the cell ran on the synthetic cell with the fixture in place of Allen's server **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-09 (afternoon) | v10. Cell 4g shows each node's figure after the summary, as Cell 4f does, and draws its planes without the measuring line and the strip's outline (`--hide-line`, new in `scripts/plane_differences.py`; the user's request of 14:53, to judge the focus by eye). The records do not change. Evidence: `test_smoke_plane_entropy.py` 8 pass and `test_smoke_plane_diff.py` 7 pass, 1 skip, with the lines on the plane panels counted and the flag followed from the command line to the figure; thirteen mutants of the new code each failed a check; the cell ran on the synthetic cell with the fixture in place of Allen's server **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-09 (evening) | v11. Added Cells 4h and 4i, the user's request of 16:19 (D-040). Cell 4h: the gradient energy over the whole line on three lines, +-3 um, +-5 um and the d-line +-m d_hat / 2 sized from the pilot's fitted diameter (m = `LINE_MULT` = 2), with one background per plane from the block, on nodes spread over bins of d_hat (`--nodes bydiameter`) plus the trunks 2 and 3. Cell 4i, on the same nodes and the d-line: G alone, the strip's entropy alone, and their min-max blend J = w g + (1 - w) eta with w(d_hat) = 1 / (1 + exp((d_hat - 1.5 um) / 0.3 um)). Evidence: `test_smoke_plane_gradient.py` 8 pass; 45 of 47 mutants of the new code each failed a check, and the other two are equivalent (a guard that `focus.gradient_energy` repeats, and a mask that the NaN profile rows of the missing planes already apply); every suite re-run (19, no failure); both cells ran on two synthetic cells (tubes of 0.8 and 3.0 um, Allen radius 0.3 um) with pilot tables made from those cells, the fixture in place of Allen's server **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-10 | v12. Cell 4i changed (D-041, the user's choice after the first run of Cells 4h and 4i on Allen's data, 13 nodes, written 2026-10-09 15:38-15:40 UTC in `pilot/planegrad/bydiameter/` and `pilot/planeblend/bydiameter/` **[Drive, data read: both summary CSVs]**) [corrected 2026-10-10: the v11 row's 'not yet run on Allen's data' no longer holds]: the entropy comes from the +-5 um strip (`--entropy-half-um`, `ENTROPY_HALF_UM`) instead of the d-line, on which its minimum fell on or next to an end plane on all 11 non-trunk nodes; G stays on the d-line; the blend is written J = w g - (1 - w) h, with h the min-max rescaled entropy (0 at its lowest), the same maximum as before; the figures draw the entropy as it is; the outputs go to `pilot/planeblend/bydiameter_strip5um/`, keeping the first run's. Evidence: `test_smoke_plane_gradient.py` 8 pass; 17 of 18 mutants of the change each failed a check, and the 18th is equivalent (a positivity guard that the refusal of an empty strip repeats); every suite re-run (19, no failure); the cell ran on two synthetic cells (tubes of 0.8 and 3.0 um) with pilot tables made from those cells, the fixture in place of Allen's server **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-10 (later) | v13. Added Cell 4j, the user's request of 15:30 (D-042, the option "Allen first, then refine"): G over the whole line, first on the line +-m r_Allen sized from Allen's diameter, then on the line +-m d / 2 sized from Block 5's fit at the plane the previous line picked, until a plane repeats or after 5 rounds (`scripts/plane_differences.py --evaluation iterate --max-rounds 5`), on the nodes of Cell 4h, into `pilot/planeiter/bydiameter/`. Evidence: `test_smoke_plane_iteration.py` 7 pass; 36 of 36 mutants of the new code each failed a check; every suite re-run (20, no failure); the cell ran on two synthetic cells (tubes of 0.8 and 3.0 um, Allen radius 0.3 um) with pilot tables made from those cells, the fixture in place of Allen's server, all crops from the cache **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-12 | v14. Cell 4j at m = 3 (D-044, the user's choice of 15:03 after the run at m = 2 on Allen's data, D-043): `LINE_MULT = 3.0`, every line +-1.5 d (round 1: +-1.5 times Allen's 2r); the output folder carries m (`pilot/planeiter/bydiameter_m3/`), so the run at m = 2 stays in `pilot/planeiter/bydiameter/`. The comparison line Gd is sized from the pilot's d_hat at the same m, as before (at m = 2 it was Cell 4h's d-line; the labels now say so). `--line-mult` now defaults to 3 for `--evaluation iterate` (`ITER_LINE_MULT`) and stays 2 for gradient and blend (`D_LINE_MULT`, D-040). Evidence: `test_smoke_plane_iteration.py` 7 pass (at m = 2 and m = 3) and `test_smoke_plane_gradient.py` 8 pass; 9 of 9 mutants of the defaults each failed a check; every suite re-run (20, no failure); Cell 4j at m = 3 ran on two synthetic cells (tubes of 0.8 and 3.0 um, Allen radius 0.3 um) into `pilot/planeiter/bydiameter_m3/`, ending on the planes and fits of m = 2 with lines +-1.5 d **[run, sandbox]**; not yet run on Allen's data at m = 3. |
| 2026-10-12 (later) | v15. Added Cell 4k, the user's request of 17:58 (D-045): for each node of Cell 4j's run, the plane the iteration chose, with the line that chose it (the last round's, through the SWC node) and the diameter fitted there as a caliper between its edges, beside the profile and the fitted model (`scripts/iteration_fits.py`, reading `planeiter_<sp>.json` of `ITER_DIR`, by default the run at m = 3), into `ITER_DIR/fits/`. Evidence: `test_smoke_iteration_fits.py` 6 pass, 1 skip; 16 of 16 mutants of the new code each failed a check; every suite re-run (21, no failure); Cells 4j and 4k ran from the notebook's code on two synthetic cells (tubes of 0.8 and 3.0 um): every node drawn, the fit redone equal to Cell 4j's d_iter, 0 downloads **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-13 | v16. Added Cell 4l, the user's request of 14:50 (D-046): at each node's plane of Cell 4j's run, Block 5's profile fitted 100 ways, 10 blur kernels (scipy families: Gaussian, Lorentzian, Voigt, Student, generalized normal, skew-normal; shapes fixed or fitted) x the width fixed at Block 5's or fitted x unweighted or with centre weights exp(-|v - v0| / lambda) re-centred on the fit's own v0, lambda 0.25-2 um (`scripts/fit_variants.py`, into `ITER_DIR/fitvar/`). Evidence: `test_smoke_fit_variants.py` 8 pass; 30 of 30 mutants of the new code each failed a check (26 at the first run; the re-weighting's stopping rule, the warm starts, the re-weighting's start and a skipped record's reason then got checks of their own); every suite re-run (22, no failure); Cells 4j, 4k and 4l ran from the notebook's own code on two synthetic cells (tubes of 0.8 and 3.0 um): 3 of 3 nodes fitted per cell, the baseline equal to Cell 4j's d_iter on 3 of 3, 0 downloads, 25-72 s per node **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-14 | v17. Added Cell 4m, the user's request of 15:35 (D-047): each of the 50 fit variants at the fitted width (10 kernels x unweighted and 4 centre weightings) runs its own Allen-guided iteration with Cell 4j's LINE_MULT and MAX_ROUNDS read from its records, Cell 4j's own iteration redone first as the reference; per node a 50-row stack of the edges v0 +- d/2 with the planes picked, and each variant's plane with its line, the fitted dendrite's edges and d (`scripts/variant_iteration.py`, into `ITER_DIR/variter/`). Evidence: `test_smoke_variant_iteration.py` 6 pass, 1 skip (`test_convergence`: nothing new iterates); 19 of 20 mutants of the new code each failed a check (the 20th, a record's planes with a gap accepted, is refused anyway by the square's own plane check); every suite re-run (23, no failure); Cells 4j to 4m ran from the notebook's own code on two synthetic cells (tubes of 0.8 and 3.0 um): Cell 4j's iteration redone identically on 3 of 3 nodes per cell, every variant ending on Cell 4j's plane, 0 downloads, 18-21 s per node on the 0.8 um tube and 2.4-3.8 min on the 3.0 um tube (other jobs running) **[run, sandbox]**; not yet run on Allen's data. |
| 2026-10-21 | v18. Cell 7's calibration nodes are chosen by Allen's diameter (S2.1 of the 2026-10-20 plan; D-061.4, D-067.4): the near-field set (2r ≤ 0.6 µm) is scanned, the far-field set (0.6-0.8 µm) is counted, not scanned; in S, flat and faint read from the pilot's columns; the cell's text and expected lines updated, the code unchanged. Evidence: `test_smoke_calibration.py` 7 pass with the S2.1 checks **[run, sandbox]**; not run on Allen's data. |
| 2026-10-21 (later) | v19. The σ_fit study is removed (S2.2 of the 2026-10-20 plan; D-057 (ii), the user, 2026-10-18): Cell 8 writes `config_production.json` only, Cell 9 builds one table, Cell 10c is gone, and the per-node CSV loses `d_tilde_sigma_spread_um`; the superseded text is struck through and kept. Evidence: the suites `test_smoke_config`, `invert`, `table`, `fit`, `cell` re-run, the estimator and simulation hashes unchanged by the removal **[run, sandbox]**; not run on Allen's data. |
| 2026-10-21 (later, 2) | v20. Cell 9's job chunks by draw (Block 14.2; D-059, D-062): the knobs `DIAM_N_DRAWS`, `DIAM_N_TASKS`, `DIAM_DRAW_OFFSET` replace `DIAM_N_TOTAL`; `phantom.n_replicates` is retired (`n_draws`, `n_realizations`). Evidence: `test_smoke_pbs.py` 4 pass, 3 skip **[run, sandbox]**; not run on davinci. |
| 2026-10-21 (later, 3) | v21. Cell 9's merge (Blocks 14.4-14.6): the table is the local linear fit on a three-axis grid; the merge reads only the columns it needs, refuses a (draw, realization) seen twice, notes the draws short of `n_realizations`, writes the per-draw table `draws_<hash>.csv` beside the table, and at $10^6$ rows takes of order 20 min and a few hundred MB, so it runs in a job, not on the login node; the expected lines updated. A `config_production.json` written before Block 14.3 (two bandwidths) is refused: make it again with Cell 8. Evidence: `test_smoke_table.py` 7 pass, the merge's refusal and notes included **[run, sandbox]**; not run on davinci. |
| 2026-10-23 | v22. Cell 7 runs the whole kernel calibration (Blocks 10 and 15; D-074 to D-081): `real` scans both sets on ±25 µm patches with a planar far-pixel background, the iterative core fit, the straightness and the 0.03 grey-level isolation rule (φ ≤ 5° for both sets), and keeps the example nodes' patch images; `fit` cuts the table at 0.84 µm and bootstraps the anchor's SE; the new `far-fit` fits the far slope γ̂ with its t interval, the per-sign and the quadratic comparison, and writes the production kernel's continuation slope; the new `figures` writes the width-by-plane, example-node and kernel-law figures. Evidence: `test_smoke_calibration.py` 8 pass, `test_smoke_calibration_scene.py` 6 pass 1 skip, `test_smoke_far_field.py` 6 pass 1 skip, `test_smoke_calibration_figures.py` 6 pass 1 skip; on rendered far phantoms γ̂ within 1 % noise-free and 0.83 ± 0.02 against 0.79 with camera noise (4 nodes) **[run, sandbox]**; not run on Allen's data. |
| 2026-10-26 | v23. Added Cell 11, S3's step 0 (Block 16.1; D-065, D-085): `scripts/cell_pool.py pool` lists the human cells with a reconstruction (Allen's `ApiCellTypesSpecimenDetail`, filtered as the Allen SDK filters it), `check` reads each cell's plane spacing, pixel size and usable planes over the whole pool and compares them with the configuration (0.28 µm, 0.1144 µm/px), `select` draws 5 cells per neuron class and layer (all of a smaller group) by a seeded permutation, among the matching cells only when `REQUIRE_MATCH`. Evidence: `test_smoke_cell_pool.py` 6 pass, 1 skip, on canned API rows and a substituted `api_query`; 17 of 17 mutants of the new code each failed a check **[run, sandbox]**; the raw field names read from the Allen SDK's source (allensdk 2.16.2, `cell_types_api.py`), not from the live API; not run against Allen's server. |
| 2026-10-26 (later) | v24. Cell 11: sparsely spiny cells are a third class, `sparsely_spiny` (D-087, the user's answer of 17:59), selected 5 per layer like `exc` and `inh`; the `--config-json` that raises the page size is written by hand (`{"survey": {...}}`). Evidence: `test_smoke_cell_pool.py` 6 pass, 1 skip with the changed oracles **[run, sandbox]**; not run against Allen's server. |
| 2026-10-26 (later, 2) | v25. Added Cell 12, S3's step 1 (Block 16.2; D-085.2, D-085.12): `scripts/survey_cells.py run --steps align` aligns every selected cell -- Allen's minimum-intensity xy projection at level 4 and `align_translation` with both flips (the user's method), the score ratio against block-shuffled images, a fine stage at level 1 on the cell's box (the assistant's addition) -- one folder per cell, resumable, a cell set aside below the ratio threshold (2, PROVISIONAL) for the user to look at, a manual alignment file to fix it. Evidence: `test_smoke_cell_alignment.py` 6 pass, 1 skip on painted synthetic cells (the shift within 2.5 full-res px, R about 7; an empty field R 1.08, set aside); 20 of 20 mutants each failed a check **[run, sandbox]**; not run against Allen's server. |
| 2026-10-26 (later, 3) | v26. Added Cell 13, S3a (Block 17; D-065.1, D-085.4-5): `survey_cells.py run --steps align,nodes` measures each aligned cell's node sample with Block 13's chain -- stratified, up to 20 nodes per bin of Allen's 2r round-robin over the stretches (the default, PROVISIONAL), or every dendrite node -- and `pool --what coverage` draws the coverage of the phantoms' band per bin of d_hat with the pass rule of D-085.5 and the diameters by layer and class. Evidence: `test_smoke_s3a.py` 6 pass, 1 skip (the step on the synthetic cell through `fetch_zblock`, the pooled coverage over it); 16 of 16 mutants each failed a check **[run, sandbox]**; not run on Allen's cells. |
| 2026-10-26 (later, 4) | v27. Added Cell 14, S3b (Block 18; D-065.2-3, D-085.8-9, D-086): the steps `radius`, `registration` and `camera` per cell and `pool --what inputs` -- d_max the largest Allen 2r over the cells, the jitter the pooled robust SDs of the ON nodes' offsets, one camera setting (the most common table set, the medians of B and the noise SD) -- as suggestions for Cell 8. Evidence: `test_smoke_s3b.py` 6 pass, 1 skip (the three steps on the synthetic cell, the pool over it and over hand-written folders); 13 of 13 mutants each failed a check **[run, sandbox]**; not run on Allen's cells. |
| 2026-10-26 (later, 5) | v28. Cell 7 (D-088.3, Block 19): at most 30 nodes per calibration set, in a seeded order, at least 10 for a production kernel (below, the fits are diagnostics and `far-fit` withholds the kernel). Added Cell 15, S3c (Block 19; D-088): the step `calibration` per cell (at most 5 scans per set) and `pool --what calibration` -- the cells' scans taken in turn across the cells up to 30 per set, then Cell 7's fit, far-fit and figures on them: one kernel. Evidence: `test_smoke_s3c.py` 6 pass, 1 skip (the cap and order of `scan_cell` with a counting provider, far-fit's kernel only with both sets sufficient on analytic far scans, the steps and the pool on the synthetic cell); 11 of 11 mutants each failed a check; `test_smoke_far_field.py` 6 pass, 1 skip with its CLI check's minimum lowered to 2 **[run, sandbox]**; not run on Allen's data. |
| 2026-10-27 | v29. Added Cell 11b (Block 16.3; D-090.1, the user: "understand this by yourself. If not possible write a colab function and cell"): `scripts/projection_check.py` lists every selected cell's SubImages that are not Primary planes, with every field Allen returns, the xy candidates and the one Cell 12's darkest rule takes, and tests the pilot's xy projections against the pixelwise minimum, mean, median and maximum over its planes on a 59 µm window (the best offset, r*, the verdict). Cell 15 (Block 19.5; D-090.2, the user: "plot the near and far field points together but with a cell base color code"): with two cells or more, the kernel law's points by cell, their residuals from the pooled law, `kernel_law_by_cell.csv` and two figures. The defaults of 2026-10-26 confirmed by the user (D-089): the set-aside ratio 2, the darkest-projection rule, the stratified default, the class name `sparsely_spiny`, the z0 flag at 1 µm, 5 scans per set per cell, "per type" as per calibration set, 529878215 the pilot. Evidence: `test_smoke_projection_check.py` 6 pass, 1 skip (16 of 16 mutants killed), `test_smoke_calibration_by_cell.py` 6 pass, 1 skip **[run, sandbox]**; neither run against Allen's server. |
| 2026-10-27 (later) | v30. Cell 7's calibration nodes are no longer screened for straightness (D-093, the user: "The straightness can be removed because we will evaluate only the measuring line"): flat ($\varphi \le 5°$), faint, in S and isolated remain; a configuration can restore the test at 2 radii (`calibration.node_straight_check`, `node_straight_tol_radii`). Re-run Cells 0, 1 and 7 to pick it up. The run file of 2026-10-27 also exists as `notebooks/s3_run_2026-10-27.ipynb`. Evidence: `test_smoke_calibration.py` with the switch **[run, sandbox]**; not yet re-run on Allen's data. |
| 2026-10-28 | v31. Cell 7's calibration nodes are no longer screened by their optical depth ($\hat\alpha \le 0.5$) nor by the flag `faint` (a dip under 6 grey levels) (D-094, the user: "Remove both faint"): flat ($\varphi \le 5°$), in S but for that flag, and isolated remain; S itself keeps the flag. A configuration restores either test (`calibration.node_alpha_check`, `calibration.node_faint_flag_check`). Same cells: re-run Cells 0, 1 and 7 (Cell 15's per-cell candidates follow the same rule). Evidence: `test_smoke_calibration.py` with both switches **[run, sandbox]**; the effect of dark nodes on the near growth in `checks/d094_2026-10-28/` **[run, sandbox]**; not yet re-run on Allen's data. |
| 2026-10-28 (later) | v32. The flag `faint` is no criterion of S any more (D-096, the user: "we have to correct ALL the nodes regardless the optical depth and the dip"): a faint node is fitted, kept in S and corrected through the table like any other, and keeps `faint` as a label, as `dark` since D-030; the bias table is built on phantoms of every dip depth. The flags of S enter the estimator signature, so the estimator and simulation hashes change (estimator `59678d4e2aedf2b6`, simulation `52aa73e479eb77cf`) and a table or row from before is refused. The pilot CSVs written before keep their old `in_S` column; Cell 7 recomputes its candidates from the flags. [corrects v31's "S itself keeps the flag"] Evidence: the smoke suites **[run, sandbox]**. |
| 2026-10-28 (later, 2) | v33. D-098 (the user, 15:01: "Let's do this", to the proposals made after the Colab run of Cells 11, 11b and 12): `fetch_swc` never takes Allen's marker file (D-098.1; four cells had failed Cell 12 on it); Cell 12 takes the xy projection of type 'MinimumIntensityProjection - xy', the darkest only when none has that type (D-098.2; on Allen's data the 'MaximumIntensityProjection - xy' is an inverted image, the darker, and misaligned 508305938); Cell 13's coverage recomputes S from each row's columns (D-098.3); Cell 7 draws, as it evaluates each candidate, the patch its background is fitted on with the traced neighbours whose predicted shadows decide `not_isolated` (D-098.4). The run file `notebooks/s3_run_2026-10-27.py` gains Cell 12b (508305938 aligned again) and Cell 13b (the set-aside cells whose overlays look aligned, D-098.5). Measured on 2026-10-28: Cell 11's check 6.6 min for 158 cells, Cell 11b 7.0 min, Cell 12 29.7 min for 51 cells; Cell 13 estimated at one to two days (not measured). The hashes are unchanged. |
| 2026-10-28 (later, 3) | v34. D-099 (the user, 17:07: "Blocks 13 and 14 are too slow. Let's change approach. Everything now on colab will run on a single pilot cell while the real runs will be on the DaVinci HPC"): the survey over the selected cells runs on davinci as three PBS jobs (`scripts/pbs/survey_prepare.pbs`, the array `survey_cells.pbs` with one task per cell, `survey_pool.pbs`; SPEC Block 16.4; the section "S3 on davinci" below), and every Colab cell runs on the pilot 529878215 only: Cells 11-15 in `survey_pilot/` (`survey/` keeps the run of 2026-10-28), Cell 11 with `cell_pool.py pick` in place of `select`, Cell 13 with 3 nodes per bin (PROVISIONAL). New in the code: the runner's `--task-index` (one row of the selection, a PBS array task's) and `--cache-per-cell` (a crop cache per cell); the run file's Cells 12b and 13b are knobs of the array job (`DIAM_CELLS`, `DIAM_FORCE`, `DIAM_INCLUDE_SET_ASIDE`). Evidence: `test_smoke_survey_pbs.py` (the jobs against stubs), `test_smoke_cell_pool.py`, `test_smoke_cell_alignment.py` **[run, sandbox]**; nothing yet run on davinci (the proxy's reach to api.brain-map.org from a compute node is the first thing the jobs test) **[open]**. |
| 2026-10-28 (later, 4) | v35. D-101 (the user, 18:17, after Cell 7's isolation overview on the pilot -- 70 candidates, 2 far-field, both shaded by the soma or a junction -- and the dialog's answers): (1) flatness per set, far $\varphi \le 15°$, near $\le 10°$ (`node_phi_max_deg_far`, `node_phi_max_deg`; was 5° for both, D-081.3); (2)-(4) the **stripe** (SPEC Block 10.S): the background of each scanned plane is fitted on the band along the measuring line that is clear of every traced neighbour's predicted shadow above 0.03 gl, the node's own fitted cylinder removed, walked outward on each side to the first shaded pixel column, the half-width (1-12 µm along the dendrite) with most pixels; a plane is usable with ≥ 500 clear pixels and ≥ 3 µm of clear run on each side; the node's verdict is the stripe at $k^*$ (`not_isolated:too_few`, `not_isolated:side_short`) in place of the square's any-shadow rule (kept as `calibration.background_support = "square"`). Why: the pilot table read off Drive shows S removing 3 of its 31 far-set nodes and flatness 26 (`bbar_few` never set), then isolation the last 2 **[data: `pilot_529878215.csv`]**. Same cells: Cells 0, 1 and 7 again. Evidence: `test_smoke_stripe.py` and the calibration suites **[run, sandbox]**; the stripe on Allen's data **[open]** (Cell 7's next run). No hash changes. |
| 2026-10-28 (later, 5) | v36. The recheck of D-101 and the stripe's knobs (the user, 19:39: "given the far field nodes highlight the stripe used, expose also the parameters to shape the stripe's features ... Also recheck the previous implementation"). Cell 7 starts with `CAL_KNOBS` -- the stripe's settings (`background_support`, $\tau$ = `isolation_depth_gl`, the $W$ grid, the minimum pixels and side run, the patch's half-side) and the two flatness bounds, at the defaults -- written to `OUT/calibration/calibration_config.json` and passed to `real`, `fit`, `far-fit` and `figures`; `real` prints `calibration knobs: ...`. The isolation figure has a panel (c), every plane's stripe (the far fit's planes lie 5-16 planes from $k^*$); the overview's labels end with $n/N$; `real` adds the far set's per-plane count. Cell 15 merges Cell 7's knobs (`REDO` re-runs its calibration step); davinci reads the same file as `DIAM_CONFIG_JSON`. Fixed after the integration tester's report of 2026-10-28 (`specs/test_reports/2026-10-28_1955_incremental.md`): the verdict now shades the soma at the stage's blur as the scan does (F1), the own cylinder includes the fitted line (F2, the reading of D-101.4 to confirm), the margins are vectorized (a near node's masks took 30-50 s per plane on the pilot's scene). **Found on the pilot's own reconstruction [run, sandbox: its SWC and the pilot table]: as built, the stripe keeps few nodes** -- 21 of the 138 near candidates usable at $k^*$, no clear pixel in any plane of the 7 far candidates -- because out-of-focus dendrites' predicted shadows exceed 0.03 gl over the node itself; the rule for them is the user's decision (Cell 7's text, "Before you run it"). Same cells: Cells 0, 1, 7 (and 15). Evidence: the smoke suites **[run, sandbox]**. No hash changes. |
| 2026-10-29 | v37. Two decisions of the user on v36's open point. D-108 ("As you recommend", to the assistant's recommendation (a) for D-101 (d)): $\tau$ = `isolation_depth_gl` is **0.3 gl** (was 0.03 gl), PROVISIONAL, to be judged on the isolation figures' panel (c). D-109 ("Just let the colab code for the pilot cell find the nodes for the far field calibration"): `real` finds the far set's candidates over the whole cell from its trace (Block 10.C) instead of taking the pilot table's -- the knobs `calibration.far_candidates` ("cell"; "table" for the table's) and `far_search_spacing_um` (5 µm) join `CAL_KNOBS`; `real` prints three `far search` lines and writes `OUT/calibration/far_search_529878215.csv`, and `calibration knobs:` ends with where the far set comes from. On the pilot's own reconstruction, in the sandbox: 723 far-range nodes, 19 chosen, 8.2 min before the first fetch. Same cells: Cells 0 and 1, then Cell 7 **opened again from GitHub** (its code changed: `CAL_KNOBS`), or paste Cell 7 below. Evidence: the smoke suites and 11 mutants **[run, sandbox]**. No hash changes. |
| 2026-10-29 (later) | v38. D-112 (the user: "Remember that the whole cell's nodes must be evaluated for both near and far field calibration"): the near set is searched over the whole cell too (Block 10.C for both sets; `calibration.near_candidates`, `near_search_spacing_um` join `CAL_KNOBS`), by Cell 7 and by S3's calibration step (Cell 15 on the pilot, the davinci array on every cell); `real` prints `candidate search: ...` once, then each set's progress and summary lines, and writes `near_search_529878215.csv` beside `far_search_529878215.csv`; the set_full nodes are counted in one line. On the pilot's reconstruction in the sandbox: 7500 near-range nodes, 3856 flat by the trace, 1960 usable at the trace's plane, 1947 with at least 3 usable planes, 380 chosen on 48 stretches, in 15.4 min; both searches 20.5 min. D-113 ("I confirm the union and the F3 without change"): no change. The pre-flight prints the search's typical background. Same cells: Cells 0 and 1, then the pre-flight and Cell 7 again from GitHub (both changed); Cell 15 unchanged in code. Evidence: the smoke suites and the mutants **[run, sandbox]**. No hash changes. |
| 2026-10-30 | v39. Three of D-115's answers (the user, 14:12, on the report of Cell 7's Colab run of the morning). D-115.4 ("do It"): `fit`'s anchor SE -- each bootstrap resample's anchor is $(\sigma_{\rm r}(0)^2 + \hat G^*(\delta_0))^{1/2}$ from the 0.84 µm knot alone, so no resample is dropped for a negative value at another knot; on the Colab scans of 2026-10-30 the SE is 0.0438 µm with 1000 of 1000 resamples (0.041 µm with 870 before), the anchor unchanged. D-115.5 ("Add the logging change"): `scans_<id>.json`'s `meta.nodes` holds each measured candidate's `pilot` -- the values its verdict read ($k^*$, the sub-plane depth, the centre, $\hat\theta$, $\hat\varphi$, $\bar B$, $\hat\mu$, $\hat d$, $\hat\alpha$, the fit status, the flags, the registration verdict). D-115.6 ("Cell 15 reuses Cell 7's search. Yes"): each search CSV gets a record beside it (`<set>_search_<id>.json`), and Cell 15 passes `--reuse-search OUT/calibration`, taking Cell 7's search when it was made from the same trace under the same settings (Cell 7's $\bar B$, $\hat\mu$ and transform go with it: the assistant's reading, PROVISIONAL). Same cells: Cells 0 and 1, then Cell 7 (it writes the records; its fit's SE changes) and Cell 15 (with `REDO = ["--force", "calibration"]` if its step has run). Evidence: `test_smoke_calibration.py` 8 pass, `test_smoke_s3c.py` 6 pass, 1 skip, `test_smoke_candidate_search.py` 6 pass, 1 skip; mutants: 1 for D-115.4, 5 for D-115.5 and 15 for D-115.6, each failing a check **[run, sandbox]**; the far search under two transforms in `checks/d115_2026-10-30/` **[run, sandbox]**; not run on Allen's data. No hash changes. |
| 2026-10-30 (later) | v40. D-115.3 (the user: "Add a goodness of fit please", then "Residual vs noise"): every scanned plane of either set whose Gaussian core fit leaves a residual RMS above $\kappa_{\rm g}$ = 1.2 times its background's noise SD on the stripe (1.4826 MAD) is dropped as `poor_fit`; its node keeps its other planes. $\kappa_{\rm g}$ (PROVISIONAL) is the 99th percentile of that ratio over 210 planes of 30 clean thin phantoms; at it, 31 % of the planes of 22 phantoms with an untraced neighbour 0.5-1.5 µm away are dropped (43 of 105 within 0.8 µm), but so would be about half the planes of clean dips 1-2.5 µm wide, and on this morning's pilot scans about 11 of 206 near planes, no far plane **[run, sandbox: `checks/d115_2026-10-30/`]**. `real` prints `goodness of fit (D-115.3: ...): planes dropped as poor_fit -- near p of n planes, on m nodes (k of them now below 3 finite widths); far ...`; `calibration.gof_check: false` in `CAL_KNOBS` would switch it off (not a knob of the cell). Same cells: Cell 7 (and 15). Evidence: `test_smoke_calibration.py` 9 pass, `test_smoke_far_field.py` 7 pass, 1 skip, 10 mutants killed **[run, sandbox]**; not run on Allen's data. No hash changes. |

This runbook covers the real-data steps, in order. Each cell names the line that must **appear** in its output. A missing line means the cell did not get as far as the code that prints it. Each step lists the decisions it feeds; those stay yours. Nothing in these scripts changes the configuration on its own.

**The notebook** [added 2026-10-07]: `Passive Features/Diameter Re-measurement/notebooks/phase2_colab.ipynb` on branch `sci/diameter-pipeline`. In Colab: File > Open notebook > GitHub, repository `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline`; or upload the file. Its code is the code below; if the two ever differ, the notebook is the one that was checked. In a new session run Cells 0, 1 and 3a first: they define the paths, `run()` and the node list `NODES`.

Paths: `REPO` is the clone in `/content`. Drive holds data only: the HttpFetcher cache (`CACHE_DIR`, ~~the one cells 1-12 use~~) and the outputs (`OUT`). [corrected 2026-10-07: cells 1-12 cached their crops in `MyDrive/Colab Notebooks/Allen Slices/cache` and the SWC in `Allen Slices/SWC` (Drive listing, 2026-10-07). Phase II keeps its own cache, `MyDrive/allen_cache`, which a run on 2026-10-06 already created (it holds the SWC `H16.06.010.01.03.05.03_668702901_m.swc`). The two are kept apart because Cell 5 reads the JPEG tables of every crop in the cache, and the montage crops of cells 1-12 were fetched at other downsample levels.] Specimen 529878215 has no shift and no flip (design handoff, "Coordinate transform"), and its plane index is z / 0.28 um (`--z0 0`).

## Cell 0: Drive and constants

```python
from google.colab import drive
drive.mount("/content/drive")

import os
SPECIMEN = 529878215
# Phase II keeps its own image cache. Cells 1-12 of the viewer notebook cached their crops in
# "MyDrive/Colab Notebooks/Allen Slices/cache" (montage crops at several downsample levels), and
# Cell 5 reads the JPEG tables of every crop in the cache, so the two caches are kept apart.
CACHE_DIR = "/content/drive/MyDrive/allen_cache"
OUT = "/content/drive/MyDrive/diameters"
# global alignment of 529878215 (cells 1-12, design handoff): no shift, no flip; plane index = z / 0.28 um
SHIFT_X, SHIFT_Y, FLIP_H, Z0 = 0.0, 0.0, None, 0.0
REG_JSON = "%s/registration/registration_%d.json" % (OUT, SPECIMEN)
for d in (CACHE_DIR, OUT):
    os.makedirs(d, exist_ok=True)
print("[cell 0] specimen %d | cache %s | outputs %s" % (SPECIMEN, CACHE_DIR, OUT))
```

Expect: `[cell 0] specimen 529878215 | cache ... | outputs ...`.

## Cell 1: bootstrap (clone or update, `sys.path`) and `run()`

```python
import importlib, os, shlex, subprocess, sys, time
REPO, BRANCH = "/content/Towards-EEG", "sci/diameter-pipeline"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "50", "-b", BRANCH,
                    "https://github.com/Leonardodm00/Towards-EEG.git", REPO], check=True)
else:      # a clone made earlier in this runtime: update it before importing anything from it
    for step in (["fetch", "--depth", "50", "origin", BRANCH], ["checkout", BRANCH], ["merge", "--ff-only", "FETCH_HEAD"]):
        subprocess.run(["git", "-C", REPO] + step, check=True)
sys.path.insert(0, os.path.join(REPO, "Passive Features", "Diameter Re-measurement", "scripts"))
import colab_bootstrap
colab_bootstrap = importlib.reload(colab_bootstrap)   # this pull's version, if an older one was imported
ENV = colab_bootstrap.bootstrap(REPO, BRANCH)      # fetch + fast-forward, sys.path, versions, commit
WS = ENV["workstream"]


def run(script, *args, check=True, every=200):
    """Run a workstream script in a fresh Python, stream its output, raise if it fails (check=True)."""
    cmd = [sys.executable, "-u", script] + [str(a) for a in args]
    print("$ python " + " ".join(shlex.quote(c) for c in cmd[2:]), flush=True)
    t0, fetched = time.time(), 0
    p = subprocess.Popen(cmd, cwd=WS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            if line.startswith("  fetched id="):        # allen_image_io.HttpFetcher: one line per crop
                fetched += 1
                if fetched % every == 0:
                    print("  ... %d crops downloaded" % fetched, flush=True)
                continue
            print(line, end="", flush=True)
        rc = p.wait()
    except KeyboardInterrupt:                           # stopping the cell stops the script too
        p.terminate()
        p.wait()
        raise
    print("[run] exit %d after %.1f min; %d crops downloaded" % (rc, (time.time() - t0) / 60, fetched), flush=True)
    if rc != 0 and check:
        raise RuntimeError("%s exited with %d: the traceback is above" % (script, rc))
    return rc


ALIGN = ["--shift-x", SHIFT_X, "--shift-y", SHIFT_Y, "--z0", Z0] + ([] if FLIP_H is None else ["--flip-h", FLIP_H])
```

Expect: `[diameter] commit <hash> on sci/diameter-pipeline`. The hash should match the branch head on GitHub.

[corrected 2026-10-07: v1's `run()` captured the output and printed stdout, or stderr only when stdout was empty, so a traceback after any printed line was hidden, and a failing script did not stop the notebook. The new `run()` streams both, raises on a non-zero exit, and counts the fetcher's one-line-per-crop messages instead of printing them. v1 also imported `colab_bootstrap` from an existing clone before pulling, so a clone older than the bootstrap failed at the import (seen in the sandbox on a clone at `bd8b2b6`); an existing clone is now fast-forwarded first.]

## Cell 2: smoke tests in Colab (about 1 min)

```python
for t in ("test_smoke_config", "test_smoke_geometry", "test_smoke_fit", "test_smoke_camera_tables",
          "test_smoke_stretches", "test_smoke_focus"):
    run("tests/smoke/%s.py" % t)
```

Expect `-- ... 0 fail, 0 error, 0 todo` six times. `test_smoke_camera_tables` checks that Allen's JPEG tables reach the renderer and that this Pillow orders them as the configuration does (8.3 or newer). `test_smoke_focus` [added 2026-10-07] checks the focus rule of D-030 on closed forms and on rendered planes.

## Cell 3: registration on node 4505 and 5-10 stretches ("cell 13")

### Cell 3a: the node list [added 2026-10-07]

```python
import json
run("scripts/list_stretches.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", OUT + "/stretches",
    "--include", 4505, "--per-type", 3, "--min-nodes", 10)
with open("%s/stretches/stretches_%d.json" % (OUT, SPECIMEN)) as f:
    NODES = json.load(f)["nodes_arg"]
# NODES = "4505,1234,5678"        # your own choice, one node id per stretch
print("NODES =", NODES)
```

The script lists every unbranched dendrite stretch: type, nodes, length, path distance from the soma, branch order and Allen's median diameter. It suggests node 4505 first, then, for basal and for apical, 3 stretches of at least 10 nodes spread over path distance, each by its middle node. Spreading over path distance reaches the trunk (proximal) and the tuft (distal); it is a starting point, not a classification. Expect the counts per type, the chosen stretches, `the pilot measures every node of these stretches: N nodes` and `NODES = "4505,..."`. The table is `stretches/stretches_<specimen>.csv` (the notebook displays it). The pilot (Cell 4b) measures every node of every stretch in `NODES`, so N sets its cost.

### Cell 3b: registration

```python
run("scripts/registration_survey.py", "--specimen", SPECIMEN, "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/registration", "--figures", *ALIGN)
```

Expect, for each node: `node 4505: ON THE PROCESS (lateral offset ...), s* ... um, dz* ... um, p ..., coverage ...` (or ALONGSIDE / FAR / NOT ON). Then a summary with `"verdicts"`. Figures go in `{OUT}/registration/figures/registration_<node>.png`.

This step feeds:
- the snap radius (design handoff §Method, step 2);
- the jitter amplitudes `jitter_xy_um` and `jitter_z_um`, which D-024 (vii) keeps at 0 until now. The summary's `suggested_jitter_um` is a suggestion, not a decision.

The file `registration_<specimen>.json` is what `--registration-json` takes below (`REG_JSON`). A node whose category is not ON is dropped from the selection S [corrected 2026-10-06: before Block 11, every registered node would have been dropped].

## Cell 4: pilot measurement, no table

### Cell 4a: node 4505's stretch, to time one stretch [added 2026-10-07]

```python
run("scripts/run_node.py", "--specimen", SPECIMEN, "--nodes", 4505, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/pilot", "--registration-json", REG_JSON, "--figures", "--background", *ALIGN)
```

Each node fetches one crop per plane: 7 planes for a flat node (3 on each side of its own), up to 23 for a steep one (`planes_half` and `planes_widen_for_tilt` in the configuration), plus one each for the figure and the background. The time per crop from Colab has not been measured: multiply the minutes printed here by the N of Cell 3a to budget Cell 4b.

### Cell 4b: every stretch of `NODES`

```python
run("scripts/run_node.py", "--specimen", SPECIMEN, "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/pilot", "--registration-json", REG_JSON, "--figures", "--background", *ALIGN)
```

Every node of the stretches through `NODES` is measured with the same chain that builds the table; node 4505's stretch comes back from the cache. Expect `stretch of N nodes from node ... measured`, then the summary JSON with `"n_in_S"`. Outputs (overwriting Cell 4a's):
- `pilot_<specimen>.csv`;
- `pilot_summary_<specimen>.json`;
- `figures/node_<id>.png`, showing the focus curves and the profile with the fitted model.

The focus rule [changed 2026-10-07, D-030]: the sharpest plane $k^*$ is the plane of maximum gradient energy of the profile, $G = B^{-2}\int_{|v| \le r + 0.5\,\mu\mathrm{m}} (d\tilde I/dv)^2\,dv$, with $r$ Allen's radius of the node; the dip depth of the design handoff's Eq. 1 is computed on the same planes and recorded as `k_star_depth`. Each figure draws both curves against the plane index, with $k^*$ (red), the dip depth's choice (grey) and the SWC's own plane (dotted).

The notebook's Cell 4c prints the summary's key fields, among them `focus_rule gradient_energy`, `k_star_vs_dip_depth` (how many nodes the two rules put in different planes) and `estimator_hash 6f0397236447fe6f`; then the table of the nodes where the rules differ, and the figures, those nodes first (or the ids in `SHOW_NODES`). A pilot written before D-030 has no `k_star_depth`: re-run Cell 4b.

This step feeds:
- the phantom μ range: `suggested_phantom_mu_range_per_um`, the 10th-90th percentile of μ̂ over the nodes in S (procedure §3.10);
- the dark-flag threshold: `alpha_hat` percentiles and `dark_share_of_converged`;
- the number of calibration nodes, `n_calibration_nodes` (Cell 7; since 2026-10-23 the per-set counts and the `reasons {...}` line of `real`).

### Cell 4d: the planes of a node, with Allen's reconstruction drawn on them [added 2026-10-07]

```python
# Cell 4d: the planes of a node with Allen's reconstruction drawn on them (run Cell 4c first)
import os
from IPython.display import Image, display
PLANE_NODES = (list(SHOW_NODES) + [int(n) for n in moved["node_id"] if int(n) not in SHOW_NODES])[:6]
if not PLANE_NODES:       # the two focus rules agree on every node: the first measured nodes instead
    PLANE_NODES = [int(n) for n in rows.loc[rows["z_sub_um"].notna(), "node_id"][:3]]
run("scripts/node_planes.py", "--specimen", SPECIMEN, "--nodes", ",".join(str(n) for n in PLANE_NODES),
    "--cache-dir", CACHE_DIR, "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN),
    "--out-dir", OUT + "/pilot/planes", *ALIGN)
for n in PLANE_NODES:
    p = "%s/pilot/planes/planes_%d.png" % (OUT, n)
    if os.path.exists(p):
        print(os.path.basename(p))
        display(Image(filename=p))
```

Run after Cell 4c: it takes the nodes of `SHOW_NODES`, then those of Cell 4c's table, at most 6 (if the two focus rules agree everywhere, the first three measured nodes). One figure per node, one panel per plane the focus rule scored (the SWC's own plane $\pm 3$, widened for tilt), all on one grey scale. On each panel: Allen's traced centre lines with their $\pm r$ outline (the SWC frustum), cyan for the measured stretch and amber for the other dendrites in the block, fainter the farther a segment's depth is from the plane's; the SWC node and the fit's profile line (black); in plane $k^*$ (red frame) the fitted edges (red ticks); the dip depth's plane in a grey dashed frame. Each panel title gives the plane, its depth minus the node's, and the two focus scores $G$ and $F$.

Each node is measured again exactly as in Cell 4b, so its crops come from the cache. Expect one `[planes] node N: ... | same as the pilot` line per node, then `[planes] wrote N montages (0 skipped) ...; crops: ... from the cache, 0 downloaded`. `DIFFERS from the pilot` means the pilot was measured with another configuration or code version: re-run Cell 4b.

Use it to judge, node by node, three things the numbers of Cell 4c cannot show: whether Allen's trace sits on the dendrite in the image (a lateral offset is a registration matter, Cell 3b); whether $k^*$ is the plane where the dendrite's edges are sharpest; and whether another neurite crosses the profile line near $k^*$ (D-030 open point on overlaps).

### Cell 4e: plane-to-plane evaluation around a node [added 2026-10-08; profile evaluation the default since 2026-10-08, 16:26] [line lengths compared since 2026-10-08, evening]

```python
# Cell 4e: plane-to-plane evaluation around a node (needs Cells 0 and 1 only)
import os
from IPython.display import Image, display
DIFF_NODES = [2, 3]       # node ids; 2 and 3 are trunk nodes of the pilot
LINE_HALF_UMS = [3, 5]    # half-lengths of the measuring line to compare, um (the pipeline's line is +-3 um)
for h in LINE_HALF_UMS:
    out = "%s/pilot/planediff/line_%gum" % (OUT, h)
    run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", ",".join(str(n) for n in DIFF_NODES),
        "--planes-half", 6, "--profile-half-um", h, "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
    for n in DIFF_NODES:
        p = "%s/planediff_%d.png" % (out, n)
        if os.path.exists(p):
            print("line +-%g um: %s" % (h, os.path.basename(p)))
            display(Image(filename=p))
```

The evaluation proposed on 2026-10-08, as a diagnostic that changes no measurement. For each node, the planes $k_{\rm SWC} - 6$ to $k_{\rm SWC} + 6$ of its block. In every plane, the profile $I_k(v)$ along the node's measuring line (the line the focus scores use: through the SWC node, across its fitted heading, $|v| \le h$, bilinear), the area under it $A_k = \int I_k\,dv$, and the difference $A_{k+1} - A_k$ between consecutive planes. Beside it, the same areas with each plane divided by its own background level $B_k$ (the interquartile mean of the block's pixels more than Allen's radius + 4 um from every traced dendrite and the soma) and multiplied by the planes' mean, so that a change of a whole plane's brightness drops out. The figure shows the planes with the measuring line (frames: smallest area blue, smallest normalised area violet, $k^*$ red, the dip depth's plane grey dashed, the SWC plane dotted), the profiles, the two area curves, their differences, and $B_k$.

`LINE_HALF_UMS` runs the evaluation once per half-length $h$ of the line, each into its own folder (`pilot/planediff/line_3um`, `line_5um`) [added 2026-10-08, evening: the user's proposal to enlarge the line]. The pipeline's line is $\pm 3$ um. The square around the node grows to hold a longer line; at $\pm 5$ um it reaches past the pilot's block, so its planes are fetched again (13 new crops per node). The JSON also holds every plane's profile (`v_um`, `profiles`).

Expect, for each $h$, one `[planediff] node N: planes ... (0 missing); smallest area at k ..., normalised k ...; background ... gl; k* ..., dip-depth plane ..., SWC plane ...` line per node, then `[planediff] wrote N figures (0 skipped) ...`. Add `"--evaluation", "image"` to the command for the pixel-difference version (the positive part of $I_{k+1} - I_k$ over the 10 x 10 um square).

What to expect (synthetic tubes, SPEC Block 11): blur moves light but does not remove it, so the area of a profile does not change while the dip stays inside the line; for a 0.5 um or a 1.5 um tube the smallest area falls at random. For a tube wider than the line, defocus pulls background light into the line, so the area is smallest in focus: for a 5 um tube and a $\pm 3$ um line, within one plane of the centre in 6 of 8 noise draws. A line long enough to hold the blurred tube keeps all its light, and that signal fades: on the 5 um tube the area's range shrinks from 9.0 to 5.0 to 2.5 grey levels x um at $\pm 3$, $\pm 4$ and $\pm 5$ um. A step in $B_k$ (a whole plane brighter or darker) moves the smallest raw area onto that plane; the normalised curve does not follow it.

### Cell 4f: entropy of the grey levels around a node [added 2026-10-08, evening] [global-minimum pick and line lengths since 2026-10-08, evening]

```python
# Cell 4f: entropy of the grey levels around a node (needs Cells 0 and 1 only)
import os
from IPython.display import Image, display
ENTROPY_NODES = [2, 3]    # node ids; 2 and 3 are trunk nodes of the pilot
LINE_HALF_UMS = [3, 5]    # half-lengths of the measuring line (and of the strip across the dendrite), um
for h in LINE_HALF_UMS:
    out = "%s/pilot/planeentropy/line_%gum" % (OUT, h)
    run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", ",".join(str(n) for n in ENTROPY_NODES),
        "--evaluation", "entropy", "--planes-half", 6, "--profile-half-um", h, "--stripe-half-um", 1.0,
        "--entropy-pick", "min", "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
    for n in ENTROPY_NODES:
        p = "%s/planeentropy_%d.png" % (out, n)
        if os.path.exists(p):
            print("line +-%g um: %s" % (h, os.path.basename(p)))
            display(Image(filename=p))
```

The user's proposal of 2026-10-08 (17:07), as a diagnostic that changes no measurement. For each node, the planes $k_{\rm SWC} - 6$ to $k_{\rm SWC} + 6$ of its block, and in every plane the Shannon entropy $H$ (bits) of the grey-level histogram, one bin per grey level, of two sets of values: the bilinear samples along the node's measuring line (the line of Cell 4e, $|v| \le h$: 53 samples at $h$ = 3 um, 89 at 5 um), and the pixels of a strip centred on that line, as long as the line across the dendrite and 2 um along it (`--stripe-half-um` sets the half-width along it). The figure shows the planes with the line and the strip's outline (frames: the plane the line's entropy picks blue, the strip's amber, $k^*$ red, the dip depth's plane grey dashed, the SWC plane dotted), the histograms of both sets coloured by plane, and the two entropy curves.

Each curve picks its global minimum (`--entropy-pick min`) [changed 2026-10-08, evening: on nodes 2 and 3 the curves have no W shape, and the earlier rule, the minimum between the two largest maxima (`--entropy-pick dip`), skipped node 2's lowest plane]. `LINE_HALF_UMS` runs the evaluation once per half-length, as in Cell 4e, into `pilot/planeentropy/line_3um` and `line_5um`.

Expect, for each $h$, one `[planediff] node N: planes ... (0 missing); lowest entropy at k ... along the line (53 samples), k ... in the strip (... pixels); k* ..., dip-depth plane ..., SWC plane ...` line per node (89 samples and about 1500 pixels at 5 um), then `[planediff] wrote N figures (0 skipped) ...`. The files are `planeentropy_<id>.png` and `planeentropy_529878215.json`.

What to expect (synthetic tubes, SPEC Block 11): the entropy is lowest in focus, where most of the neighbourhood sits at the ground level and the tube covers few pixels; it rises on both sides as the blur spreads the tube's darkness over more pixels and more grey levels, and falls again far out as a thin tube fades into the noise. With a $\pm 3$ um line that last fall can put the strip's lowest entropy on an end plane (0.5 um tube, 7 of 8 noise draws); with $\pm 4$ or $\pm 5$ um the global minima of both curves fall within one plane of the centre in 8 of 8 draws for the 0.5 um tube, and for the 1.5 um tube in 8 of 8 (strip) and 7 or 8 of 8 (line). For a faint 5 um tube both curves stay flat at every length, as every other score does. A whole plane brighter or darker by whole grey levels keeps its entropy (the area of Cell 4e moves with it). With the earlier dip rule and a $\pm 3$ um line (the default until 2026-10-08, evening), the strip's pick fell within one plane of the centre in 8 of 8 noise draws for the 0.5 um and the 1.5 um tube, and the line's in 7 and 8 of 8. The line's 53 samples give a lower and noisier entropy than the strip's pixels: on flat ground with 3 grey levels of noise, 0.25 bits low with a spread of 0.13 bits, against 0.02 and 0.03.

### Cell 4g: the entropy on thin dendrites, line +-5 um [added 2026-10-09; updated 2026-10-09, afternoon]

```python
# Cell 4g: the entropy on thin dendrites, line +-5 um (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
THIN = dict(max_nodes=12, per_stretch=3, max_2r_um=0.6, max_dhat_um=1.0)   # the choice of thin reference nodes
out = OUT + "/pilot/planeentropy/thin_line_5um"
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "thin",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--max-nodes", THIN["max_nodes"],
    "--per-stretch", THIN["per_stretch"], "--thin-max-2r-um", THIN["max_2r_um"],
    "--thin-max-dhat-um", THIN["max_dhat_um"], "--evaluation", "entropy", "--planes-half", 6,
    "--profile-half-um", 5, "--stripe-half-um", 1.0, "--entropy-pick", "min", "--hide-line",
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planeentropy_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    print(pd.read_csv(summ).to_string(index=False))
p = "%s/planeentropy_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planeentropy_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, as in Cell 4f; its planes drawn without the line (--hide-line)
        p = "%s/planeentropy_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))
```

The test asked for on 2026-10-09 (14:30): the entropy along a $\pm 5$ um line, which picked planes 114 and 115 on the trunk nodes 2 and 3 where $k^*$ picked 123, is run on thin dendrites, where the gradient energy is thought to work. The nodes come from the pilot's table (`--nodes thin`): in S, Allen's $2r \le 0.6$ um, a fitted $d \le 1.0$ um, $k^*$ equal to the dip depth's plane (the pipeline's two focus rules agree, so $k^*$ is a credible reference), not steep; up to 3 per unbranched stretch, spread along it, and 12 in all, spread over the stretches. The thresholds are the assistant's provisional choice; `THIN` sets them.

Output in `pilot/planeentropy/thin_line_5um/`: one figure per node, as in Cell 4f; `planeentropy_summary_529878215.png`, one panel per node with both entropy curves rescaled 0-1 against $k - k_{\rm SWC}$, $k^*$, the dip depth's plane, the SWC plane and the two picks; and `planeentropy_summary_529878215.csv`, per node the type, Allen's $2r$, the fitted $d$, the planes and each pick minus $k^*$, printed by the cell. [added 2026-10-09, afternoon: the user's request] The cell then shows each node's figure, in the summary's order, with the planes drawn without the measuring line and the strip's outline (`--hide-line`), so that the focus can be judged by eye; the node is at the centre of every panel and the line runs across the dendrite through it. Cells 4e and 4f still draw the line; `--hide-line` works there too.

Expect `[planediff] thin nodes (...): <ids>`, one `[planediff] node N: ...` line per node, then `[planediff] summary over N nodes: k_h_line within 1 plane of k* in a of N (median |diff| ... planes); k_h_strip within 1 plane of k* in b of N (...)` and `[planediff] wrote N figures (0 skipped) ...`; about 13 new crops per node (the $\pm 5$ um square).

What to expect (synthetic tubes, SPEC Block 11): with a $\pm 5$ um line the global minimum of both curves fell within one plane of the centre in 8 of 8 noise draws for a 0.5 um tube. If the thin nodes agree with $k^*$ too, the entropy works on both kinds of node; where it does not, the panels show how it parts from $k^*$ (a neighbouring neurite inside the line or the strip is the first thing to look for).

### Cell 4h: the gradient energy over the whole line, on three lines, across diameters [added 2026-10-09, evening]

```python
# Cell 4h: the gradient energy over the whole line, on three lines, across diameters (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
BYDIAM = dict(dhat_bins="0.8,1.0,1.5,2.0,3.0", per_bin=2, add_nodes="2,3")   # the choice of nodes by d_hat
GRAD_LINES_UM = "3,5"   # half-lengths of the fixed lines, um
LINE_MULT = 2.0         # the d-line's full width in units of d_hat (2: the line [-d_hat, d_hat])
out = OUT + "/pilot/planegrad/bydiameter"
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "bydiameter",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--dhat-bins", BYDIAM["dhat_bins"],
    "--per-bin", BYDIAM["per_bin"], "--add-nodes", BYDIAM["add_nodes"], "--evaluation", "gradient",
    "--grad-lines-um", GRAD_LINES_UM, "--line-mult", LINE_MULT, "--planes-half", 6,
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planegrad_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    print(pd.read_csv(summ).to_string(index=False))
p = "%s/planegrad_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planegrad_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, in order of d_hat; its planes drawn without any line
        p = "%s/planegrad_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))
```

The comparison asked for on 2026-10-09 (16:19), a diagnostic that changes no measurement (D-040). On the trunk nodes 2 and 3 the pipeline's focus rule, $k^*$, integrates $(\partial_v \tilde I_k)^2$ only over $|v| \le r_{\rm Allen} + 0.5$ um, about 1.6 um, while the fitted diameter $\hat d$ there is about 5 um: the window ends inside the dark core and sees neither edge. Here the gradient energy of plane $k$, $G_k = B_k^{-2} \int (\partial_v \tilde I_k)^2 \, dv$, is integrated over the whole of each of three lines through the node, across the dendrite: $\pm 3$ um, $\pm 5$ um, and the d-line $\pm m \hat d / 2$, sized from the pilot's fitted diameter $\hat d$ of the node (full width $m \hat d$; `LINE_MULT` sets $m$, and 2 gives the line $[-\hat d, \hat d]$). $\tilde I_k$ is the profile along the line, smoothed by one sample; $B_k$ is the plane's background, the interquartile mean of the block's pixels far from every traced dendrite (as in Cell 4e), the same for the three lines. Each line picks the plane of its largest $G_k$.

The nodes come from the pilot's table (`--nodes bydiameter`): the rows with a converged fit, a finite $z_{\rm sub}$, a finite positive $\hat d$, not steep and without the `crossing` flag, split by $\hat d$ into the bins $(0, 0.8]$, $(0.8, 1.0]$, $(1.0, 1.5]$, $(1.5, 2.0]$, $(2.0, 3.0]$ and $(3.0, \infty)$ um; up to 2 per bin, spread along the stretches; then the trunk nodes 2 and 3, whatever their row says; all ordered by $\hat d$. The bins and the counts are the assistant's provisional choice; `BYDIAM` sets them. Each node is measured again on its cached crops, so the $\hat d$ of its d-line equals the pilot's for an unchanged configuration.

Output in `pilot/planegrad/bydiameter/`: one figure per node, `planegrad_<id>.png`: the planes on one grey scale without any line, framed at each line's pick (blue G3, violet G5, aqua Gd) and at $k^*$ (red), the dip depth's plane (grey, dashed) and the SWC plane (black, dotted); below, every plane's profile along the longest line with each line's extent dashed, each line's $G_k$ relative to its own maximum with its pick, and $B_k$. `planegrad_summary_529878215.png`: one panel per node, the three curves rescaled 0-1 against $k - k_{\rm SWC}$, with the picks and each pick minus $k^*$ in the panel's title; `planegrad_summary_529878215.csv`: per node the type, Allen's $2r$, $\hat d$, the d-line's half-length, the planes and each pick minus $k^*$, printed by the cell. The cell then shows each node's figure, in order of $\hat d$.

Expect `[planediff] nodes by diameter (...): <id> (<d_hat> um), ...`, one `[planediff] node N: planes ... (0 missing); d_hat ... um; G picks: +-3 um k ..., +-5 um k ..., +-... um (d-line) k ...; k* ..., dip-depth plane ..., SWC plane ...` line per node, then `[planediff] summary over N nodes: k_G3 within 1 plane of k* in a of N (median |diff| ... planes); k_G5 ...; k_Gd ...` and `[planediff] wrote N figures (0 skipped) ...`. A node whose d-line would hold fewer than 3 samples keeps the two fixed lines, and its line says `no d-line: ...`.

What to expect (synthetic tubes, SPEC Block 11): on a 0.8 um tube (Allen radius 0.3 um) the three lines and $k^*$ pick the in-focus plane on every node; on a 3.0 um tube with the same Allen radius, so that $k^*$'s window ends 0.8 um from the axis, $k^*$ falls four planes from the tube's axis (flagged `stack_edge`), while the three lines agree with each other on every node and pick a plane one or two planes (0.28-0.56 um) from the axis, where $G_k$ stays within 3 % of its maximum over two planes; $\hat d$ there is about 2.7 um, fitted at $k^*$. On Allen's data, if the d-line agrees with $\pm 3$ um on the thin nodes and with $\pm 5$ um on the thick ones, one rule sized from $\hat d$ covers both kinds of node with no threshold. A wrong $\hat d$ (a neighbouring neurite inside the fit) gives a wrong d-line; the profile panel shows its extent against the dendrite.

### Cell 4i: G on the d-line, the entropy of the +-5 um strip and their blend, across diameters [added 2026-10-09, evening; changed 2026-10-10, D-041: the entropy on the +-5 um strip, J = w g - (1 - w) h]

```python
# Cell 4i: G on the d-line, the entropy of the +-5 um strip and their blend, across diameters (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
BYDIAM = dict(dhat_bins="0.8,1.0,1.5,2.0,3.0", per_bin=2, add_nodes="2,3")   # the same nodes as Cell 4h
LINE_MULT = 2.0                        # G's line: full width LINE_MULT x d_hat, as in Cell 4h
ENTROPY_HALF_UM = 5.0                  # the entropy's strip: +-5 um across the dendrite (D-041) ...
STRIPE_HALF_UM = 1.0                   # ... and +-1 um along it (as in Cells 4f and 4g)
SIGMOID = dict(d0_um=1.5, s_um=0.3)    # the weight of G: w(d_hat) = 1 / (1 + exp((d_hat - d0) / s))
out = OUT + "/pilot/planeblend/bydiameter_strip%gum" % ENTROPY_HALF_UM
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "bydiameter",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--dhat-bins", BYDIAM["dhat_bins"],
    "--per-bin", BYDIAM["per_bin"], "--add-nodes", BYDIAM["add_nodes"], "--evaluation", "blend",
    "--line-mult", LINE_MULT, "--entropy-half-um", ENTROPY_HALF_UM, "--stripe-half-um", STRIPE_HALF_UM,
    "--sigmoid-d0-um", SIGMOID["d0_um"], "--sigmoid-s-um", SIGMOID["s_um"], "--planes-half", 6,
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planeblend_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    print(pd.read_csv(summ).to_string(index=False))
p = "%s/planeblend_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planeblend_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, in order of d_hat; its planes drawn without any line
        p = "%s/planeblend_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))
```

The second comparison of 2026-10-09 (16:19, D-040), changed on 2026-10-10 (D-041), on the nodes of Cell 4h: three ways to pick the plane. (i) $G_k$ alone, over the whole d-line $\pm m \hat d / 2$ (`LINE_MULT` = $m$ = 2), as in Cell 4h. (ii) The strip's entropy alone, $H_k$: the Shannon entropy of the grey levels of the pixels within $|v| \le$ `ENTROPY_HALF_UM` (5 um) across the dendrite and $|u| \le$ `STRIPE_HALF_UM` (1 um) along it, the plane of its lowest value (as in Cells 4f and 4g). (iii) Their blend, in the user's form (D-041): over the planes where both curves are finite, each is min-max rescaled, $g_k = (G_k - \min G) / (\max G - \min G)$ (1 at the largest $G$) and $h_k = (H_k - \min H) / (\max H - \min H)$ (0 at the lowest $H$), and $J_k = w \, g_k - (1 - w) \, h_k$, largest where $G$ is high and $H$ low, with the weight of $G$ falling with the diameter, $w(\hat d) = 1 / (1 + e^{(\hat d - d_0) / s})$, $d_0$ = 1.5 um and $s$ = 0.3 um (`SIGMOID`): $w$ is 1/2 at $\hat d = d_0$, 0.97 at 0.5 um and 0.03 at 2.5 um. The blend picks the plane of its largest $J_k$; a flat curve scores 0 on every plane and leaves the choice to the other.

Why the entropy's strip is $\pm 5$ um and not the d-line (D-041): in the first run (2026-10-09, the entropy on the d-line, outputs kept in `pilot/planeblend/bydiameter/`) the entropy was highest in focus, and its minimum fell on the last plane or the one next to it on all 11 non-trunk nodes, so wherever $w < 1/2$ the blend followed it away from focus; on the trunks, where the d-line is itself about $\pm 5$ um, it agreed with the planes judged right. The form $J_k = w \, g_k - (1 - w) \, h_k$ replaces $w \, g_k + (1 - w)(1 - h_k)$, which differs from it by $1 - w$, the same on every plane of a node, so that form picked the same planes.

Output in `pilot/planeblend/bydiameter_strip5um/`: one figure per node, `planeblend_<id>.png`: the planes without any line, framed at the three picks (aqua Gd, green J, yellow Hs) and at the pipeline's planes; below, every plane's profile along the $\pm 5$ um line with the d-line's extent dashed, $g_k$ and $h_k$ with their picks ($h$ drawn as it is: its pick is its lowest point), and $J_k$ with its pick and $w$, on the axis $-(1 - w) .. w$ that $J$ spans. `planeblend_summary_529878215.png`: one panel per node with the three curves rescaled 0-1 and drawn as they are (the entropy's pick at the bottom), $w$ in each title; `planeblend_summary_529878215.csv`: per node $\hat d$, the d-line's half-length, $w$, the planes and each pick minus $k^*$ (`k_Gd`, `k_blend`, `k_Hs`), printed by the cell. The cell then shows each node's figure, in order of $\hat d$.

Expect the node list of Cell 4h, one `[planediff] node N: planes ... (0 missing); d_hat ... um; G on the d-line +-... um (... samples), entropy on the strip +-5.0 um (... pixels); w ...; G k ..., blend k ..., strip entropy k ...; k* ..., dip-depth plane ..., SWC plane ...` line per node, then `[planediff] summary over N nodes: k_Gd within 1 plane of k* in a of N (...); k_blend ...; k_Hs ...` and `[planediff] wrote N figures (0 skipped) ...`; about 13 new crops per node (the $\pm 5$ um square, as in Cell 4g). A node whose d-line would hold fewer than 3 samples is skipped, with the reason.

What to expect (synthetic tubes, SPEC Block 11): on a 0.8 um tube $w$ = 0.90, and $G_k$, the $\pm 5$ um strip's entropy and the blend all pick the in-focus plane on every node; on a 3.0 um tube whose Allen radius is 0.3 um, so that $k^*$ falls four planes from the axis, $w$ = 0.02: the $\pm 5$ um entropy picks the plane next to the axis on every node and the blend follows it, while $G_k$ alone picks one or two planes from the axis. On Allen's data the $\pm 5$ um strip's entropy fell within one plane of $k^*$ on 1 of 12 thin nodes in Cell 4g, but there $w$ is about 0.9 and $G_k$ decides; what this run adds is the nodes of 1.5-3 um, where $w < 1/2$ and the $\pm 5$ um entropy decides.

### Cell 4j: G on lines sized from Allen's diameter, then from the fit at the plane each picks [added 2026-10-10] [changed 2026-10-12: m = 3, D-044]

```python
# Cell 4j: G on lines sized from Allen's diameter, then from the fit at the plane each picks (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
BYDIAM = dict(dhat_bins="0.8,1.0,1.5,2.0,3.0", per_bin=2, add_nodes="2,3")   # the same nodes as Cells 4h and 4i
LINE_MULT = 3.0     # each line's full width in units of the diameter sizing it (3: +-1.5 d, D-044; round 1: Allen's 2r)
MAX_ROUNDS = 5      # at most this many rounds, round 1 included; the iteration stops earlier when a plane repeats
out = OUT + "/pilot/planeiter/bydiameter_m%g" % LINE_MULT    # bydiameter_m3; the run at m = 2 stays in bydiameter
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "bydiameter",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--dhat-bins", BYDIAM["dhat_bins"],
    "--per-bin", BYDIAM["per_bin"], "--add-nodes", BYDIAM["add_nodes"], "--evaluation", "iterate",
    "--line-mult", LINE_MULT, "--max-rounds", MAX_ROUNDS, "--planes-half", 6,
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planeiter_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    tab = pd.read_csv(summ)
    print(tab.drop(columns=["trajectory"]).to_string(index=False))
    for n, t in zip(tab["node_id"], tab["trajectory"]):    # per node: each round's line, its plane, the fit there
        print("node %d: %s" % (n, t))
p = "%s/planeiter_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planeiter_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, in order of d_hat; its planes drawn without any line
        p = "%s/planeiter_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))
```

The user's request of 2026-10-10 (15:30), after the runs of Cells 4h and 4i on Allen's data (Cell 4i's re-run of 13:20 UTC included): $G_k$ over the whole d-line was judged the best of the plane rules, but its line is sized from the pilot's $\hat d$, which is itself fitted at $k^*$, the plane under test. Here the first line comes from Allen's diameter instead, and the diameter is then refined (D-042, the option "Allen first, then refine"; the method carried forward, D-043). Round 1: the line $\pm m r_{\rm Allen}$ through the node, across the dendrite (full width $m$ times Allen's $2r$; `LINE_MULT` = $m$ = 3, the user's choice of 2026-10-12 after the run at $m$ = 2, D-044), and the plane $k_1$ of its largest $G_k = B_k^{-2} \int (\partial_v \tilde I_k)^2 \, dv$, with the background $B_k$ of Cell 4h. Block 5's fit at plane $k_1$ (the pipeline's own fit with only the plane changed: the pilot's centre, heading, tilt and offsets, and $\bar B$ of that plane; at $k^*$ it returns the pilot's $\hat d$ to the last digit) gives $d_1$. Round 2 takes the line $\pm m d_1 / 2$ and its plane $k_2$, the fit there gives $d_2$, and so on. The iteration stops when a plane comes up again (`converged` when it is the previous round's plane, `cycle` when it is an earlier one; the fit made there is reused), after `MAX_ROUNDS` = 5 rounds (`max_rounds`), or when a fit fails (`fit_failed`). That "the plane repeats" covers any earlier plane, and what the fit holds fixed, are the assistant's reading of the option, PROVISIONAL.

The nodes are Cell 4h's (`--nodes bydiameter`, `BYDIAM`), so the three cells compare node by node. The square around each node holds round 1's line and the line sized from the pilot's $\hat d$ at the same $m$, and is made again, larger, when a later round's line does not fit; up to about $\pm 4.7$ um it is cut from the pilot's cached block, beyond that it is fetched as new crops. From the fits of the run at $m$ = 2, only the trunks' lines pass that limit at $m$ = 3 (nodes 2 and 3, about $\pm 8$-9 um); theirs passed it at $m$ = 2 too.

Output in `pilot/planeiter/bydiameter_m3/` (the folder carries $m$, so the run at $m$ = 2 stays in `pilot/planeiter/bydiameter/`): one figure per node, `planeiter_<id>.png`: the planes without any line, framed at the iteration's last plane (blue, it), round 1's plane (yellow, dashed, A), the pick of the line sized from the pilot's $\hat d$ at the same $m$ (aqua, Gd; at $m$ = 2 it is Cell 4h's d-line) and the pipeline's planes; below, every plane's profile along the longest line used, with round 1's, the last round's and the pilot d-line's extents dashed; each round's $G_k$ relative to its own maximum with its pick (round 1 yellow, the later rounds blue, darker = later; the pilot d-line aqua); and the trajectory: the diameter sizing each line (Allen's $2r$, then the fit at each round's plane, labelled with the plane; hollow: a plane picked again) against the pilot's $\hat d$ (red, dashed). `planeiter_summary_529878215.png`: one panel per node with round 1's curve, the last round's and the pilot d-line's, rescaled 0-1, each pick minus $k^*$ and the iteration's ending in the title; `planeiter_summary_529878215.csv`: per node the type, Allen's $2r$, $\hat d$, $k^*$, the three planes (`k_allen`, `k_iter`, `k_Gd`) and each minus $k^*$, `d_iter_um` (the fit at the last plane), `iter_status`, `n_rounds`, `n_restacks` and the trajectory, which the cell prints one line per node. The cell then shows each node's figure, in order of $\hat d$.

Expect the node list of Cell 4h, one `[planediff] node N: planes ... (0 missing); Allen 2r ... um; start d ... um | r1 +-... um k ... d ... um | r2 ... | converged; the pilot's d-line +-... um k ...; k* ..., dip-depth plane ..., SWC plane ...` line per node, then `[planediff] summary over N nodes: k_allen within 1 plane of k* in a of N (...); k_iter ...; k_Gd ...`, `[planediff] iteration endings: converged ...` and `[planediff] wrote N figures (0 skipped) ...`. A node without Allen's radius is skipped, with the reason.

What to expect (synthetic tubes, SPEC Block 11, at $m$ = 3): on a 0.8 um tube with Allen's radius 0.3 um, round 1's line ($\pm 0.9$ um) already reaches the tube's edges: every node picks the in-focus plane in round 1, the fit there is the pilot's $\hat d$ (0.84-0.86 um), and round 2 picks the same plane (converged in 2 rounds). On a 3.0 um tube with the same Allen radius, round 1's line ($\pm 0.9$ um) lies inside the tube and picks the end plane of the range (-6), where the blurred profile fits 2.7-2.8 um; round 2's line ($\pm 4.1$-4.2 um) picks a plane one or two from the axis (-1 or -2), the fit there gives 3.0-3.2 um, and round 3 picks it again (converged in 3 rounds on every node, the square made again twice): the iteration recovers from an Allen diameter five times too small and ends on the plane the pilot's d-line picks, while $k^*$ stays four planes from the axis. On both tubes $m$ = 3 ends on the plane of $m$ = 2 with the same diameter; only the lines are longer. On Allen's data (the run at $m$ = 2) Allen's $2r$ is 0.45-0.76 um on the 11 nodes of the bins and 2.0-2.2 um on the trunks, so round 1 tests a start that knows little of the fit: on the thick nodes its line lies inside the dendrite, as on the 3.0 um tube. What to look at, node by node: whether `k_iter` and `d_iter_um` move from the run at $m$ = 2 (`pilot/planeiter/bydiameter/`: all 13 nodes converged, `k_iter` equal to `k_Gd` on all 13 and within one plane of $k^*$ on 10 of the 11 nodes of the bins); whether `k_iter` still agrees with `k_Gd`, now both at $m$ = 3; and the trunks (nodes 2 and 3): at $m$ = 2 they ended on 113 and 115, beside the planes 114 and 115 where the $\pm 5$ um entropy was judged right (D-039), node 2 on the range's end plane (113 of 113..125) with $d$ at the fit's 6 um bound; the longer lines may now reach neighbouring neurites.

### Cell 4k: the plane each node's iteration chose, with the line that chose it and the diameter fitted there [added 2026-10-12]

```python
# Cell 4k: the plane each node's iteration chose, with the line that chose it and the diameter fitted there (needs Cells 0 and 1 and Cell 4j's output)
import json
import os
from IPython.display import Image, display
ITER_DIR = OUT + "/pilot/planeiter/bydiameter_m3"   # Cell 4j's run at m = 3 (D-044); ".../bydiameter" for its run at m = 2
out = ITER_DIR + "/fits"
run("scripts/iteration_fits.py", "--specimen", SPECIMEN, "--iter-json", "%s/planeiter_%d.json" % (ITER_DIR, SPECIMEN),
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
p = "%s/planefit_overview_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planefit_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        drawn = [r for r in json.load(f) if "skipped" not in r]
    for r in drawn:       # each node's figure, in Cell 4j's order (the pilot's d_hat)
        print("node %d: k_iter %d (k* %d), d %.2f um, %s%s" % (r["node_id"], r["k_iter"], r["k_star"], r["d_refit_um"],
                                                             r["fit_status"], "".join("; " + n for n in r["notes"])))
        display(Image(filename=r["png"]))
```

The user's request of 2026-10-12 (17:58): for each node, the plane the Allen-guided iteration chose ($k^{\rm it}$, Cell 4j), drawn with the line that chose it and the diameter fitted there (D-045). The line is the one of the last round that picked a plane -- at convergence the repeat round, sized from $d^{\rm it}$ itself: $\pm m\, d^{\rm it} / 2$ through the SWC node, across the pilot's heading ($m$ = 3, D-044). The diameter is Block 5's fit redone at $k^{\rm it}$; its two edges, $\hat v_0 \pm d^{\rm it} / 2$ from the pilot's fitted centre, are drawn on that line as a caliper (blue). The fit's own line passes through the fitted centre, parallel to the drawn one; how far apart the two lie along the branch is printed per node (`u_um`).

Nothing is measured anew: the cell reads Cell 4j's `planeiter_529878215.json` in `ITER_DIR`, rebuilds each node's last square from the cached crops, and redoes the fit at $k^{\rm it}$ only to draw it. The fit must give Cell 4j's `d_iter_um` to the last digit and the square Cell 4j's half-width; a difference is printed in the node's notes, not hidden. Set `ITER_DIR` to `OUT + "/pilot/planeiter/bydiameter"` to draw the run at $m$ = 2.

Output in `ITER_DIR/fits/`: `planefit_overview_529878215.png`, every node's plane with the line (black) and the caliper (blue), in Cell 4j's order; one figure per node, `planefit_<id>.png`: on the left the plane $k^{\rm it}$ on its own grey scale with the line, the caliper, the SWC node (+) and a scale bar; on the right, against the position $v$ along the line (0 at the SWC node), the plane's profile along the line (grey), the fit's profile (black dots: 9 parallel lines averaged along the branch through the fitted centre, over the fit's $\pm 3$ um window, shaded), the fitted model (blue), $\bar B$ (dashed) and the edges (blue verticals, $d$ between them); `planefit_529878215.json`: per node $k^{\rm it}$, $k^*$, the fit (`d_refit_um`, `fit_status`, `v0_hat_um`), `s_um` and `u_um`, the caliper's ends (`edges_v_um` along the line, `edges_xy_um` in the image), the line (`line_xy_um`, `sel_round`, `sel_half_um`) and the notes. The cell prints one line per node and shows the overview, then each node's figure.

Expect one `[planefit] node N: k_iter K (k* K*); line +-H um (round R); d D um, edges at v -E and +E um from the SWC node` line per node, then `[planefit] 13 nodes drawn, 0 skipped; the fit redone at k_iter equals Cell 4j's d_iter on 13 of 13` and `[planefit] files in ...; crops: ... from the cache, 0 downloaded`: the crops are those Cell 4j fetched.

What to expect (synthetic tubes, SPEC Block 11): on a 0.8 um tube the caliper lies on the tube, its middle within 0.004 um of the SWC node (the tube's axis passes through it), with $d$ 0.84-0.86 um; on a 3.0 um tube $d$ is 3.03-3.24 um and the profile panel shows $\bar B$ 4 to 12 grey levels under the profile's plateau, so the model cannot reach it: $\bar B$ comes from the block's pixels outside Allen's radius + 1 um (0.3 + 1 um), which on this tube still holds its edges. What to look at, node by node: whether the caliper spans the dendrite as you see it in the plane; on the trunks (nodes 2 and 3), whether the edges lie inside the fit's $\pm 3$ um window (at $d$ = 6 um they reach its ends) and where $\bar B$ lies against the plateau.

### Cell 4l: fit variants at the plane each node's iteration chose: other blur kernels and centre weights [added 2026-10-13]

```python
# Cell 4l: fit variants at the plane each node's iteration chose: other blur kernels and centre weights (needs Cells 0 and 1 and Cell 4j's output; 25-75 s per node)
import json
import os
import pandas as pd
from IPython.display import Image, display
ITER_DIR = OUT + "/pilot/planeiter/bydiameter_m3"   # Cell 4j's run at m = 3 (D-044)
out = ITER_DIR + "/fitvar"
run("scripts/fit_variants.py", "--specimen", SPECIMEN, "--iter-json", "%s/planeiter_%d.json" % (ITER_DIR, SPECIMEN),
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
p = "%s/fitvar_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
p = "%s/fitvar_%d.csv" % (out, SPECIMEN)
if os.path.exists(p):           # d (um) of a few variants per node, in Cell 4j's order; the CSV holds all 100
    t = pd.read_csv(p)
    keys = ["gauss|w fixed|none", "gauss|w fixed|lam=0.25", "gauss|w fixed|lam=0.5", "gauss|w fixed|lam=1",
            "gauss|w fixed|lam=2", "gauss|w fitted|none", "lorentz|w fixed|none", "gennorm beta=1|w fixed|none"]
    keys = [k for k in keys if k in set(t["variant"])]
    tab = t[t["variant"].isin(keys)].pivot(index="node_id", columns="variant", values="d_um")
    display(tab.loc[list(dict.fromkeys(t["node_id"])), keys].round(2))
p = "%s/fitvar_%d.json" % (out, SPECIMEN)
if os.path.exists(p):
    with open(p) as f:
        done = [r for r in json.load(f) if "skipped" not in r]
    for r in done:       # each node's figure, in Cell 4j's order (the pilot's d_hat)
        print("node %d: k_iter %d (k* %d), Block 5's d %.2f um%s" % (r["node_id"], r["k_iter"], r["k_star"],
                                                                   r["d_base_um"], "".join("; " + n for n in r["notes"])))
        display(Image(filename=r["png"]))
```

The user's request of 2026-10-13 (14:50), on seeing Cell 4k: the thin dendrites look hugely overestimated, so try a fit whose weights fall exponentially from the dip's centre, and other families of the blur kernel, with and without the weights (D-046). A diagnostic: nothing in the measurement changes, Block 5's fit keeps its Gaussian (D-026). At each node's plane $k^{\rm it}$ (Cell 4j, not re-chosen) the profile Block 5 fitted there is fitted again 100 ways: 10 kernels (Gaussian, Lorentzian, Voigt $r$ = 1, Student $\nu$ = 3, generalized normal $\beta$ = 1 and 4, and the Voigt, Student and generalized normal with their shape fitted, and a skew-normal with its skew fitted), each at the width of Block 5's Gaussian (FWHM 0.233 um) and with the width fitted (0.1-2 um), each unweighted and with the weights $\exp(-|v - v_0| / \lambda)$, $\lambda$ = 0.25, 0.5, 1, 2 um. The weights follow the fit's own centre $v_0$: fit, re-centre the weights on the new $v_0$, fit again, until $v_0$ and $d$ move by less than 1e-4 um (at most 30 rounds). Every kernel has unit area and its peak at $v_0$; its width is its FWHM, so a kernel and the Gaussian compared at the same width have the same half-maximum width.

Nothing is fetched: the cell reads Cell 4j's `planeiter_529878215.json` in `ITER_DIR` and rebuilds each node's square from the cached crops. The Gaussian at the fixed width, unweighted, is Block 5's fit itself and must give Cell 4j's `d_iter_um`; a difference is printed in the node's notes. To try fewer variants, add for instance `"--families", "lorentz,gennorm"`, `"--lambdas", "0.5"` or `"--width-modes", "fixed"` to the `run(...)` call (the Gaussian is always kept); `"--nodes", "8497,3856"` for some nodes only.

Output in `ITER_DIR/fitvar/`: `fitvar_summary_529878215.png`, for every kernel and width mode $d / d_{\rm base}$ against $d_{\rm base}$ (Block 5's $d$), one series per weighting, hollow where a fit is flagged; one figure per node, `fitvar_<id>.png`: three columns of the profile at $k^{\rm it}$ against $v$ (um from the fitted centre) with the fits: every kernel unweighted at the fixed width, every kernel unweighted at a fitted width, the Gaussian at the fixed width unweighted and for each $\lambda$ (with the weights drawn below); above each column the fits' edges $v_0 \pm d/2$ as stacked calipers with their $d$; below, $d$ of all 100 variants as a table coloured by $d / d_{\rm base}$ (blue: smaller than Block 5's), flagged * at a bound, x failed, ~ weights not settled in 30 rounds; `fitvar_529878215.csv`, one row per node and variant (`d_um`, `fwhm_um`, `shape`, `v0_um`, `status`, `ratio`, the residual sums with and without the weights, ...) and `fitvar_529878215.json` with the profiles. The cell shows the summary, a table of eight variants' $d$ per node, then each node's figure.

Expect one `[fitvar] node N: k_iter K (k* K*); Block 5 d D um (converged); Gaussian centre-weighted d ... (lambda 0.25/0.5/1/2 um); unweighted kernels: fixed width d A-B, fitted width d C-E, the Gaussian's fitted FWHM W um; smallest d ...; 100 variants, F flagged; T s` line per node, then `[fitvar] 13 nodes fitted, 0 skipped; 100 variants per node; the baseline equals Cell 4j's d_iter on 13 of 13` and `[fitvar] files in ...; crops: ... from the cache, 0 downloaded`. It takes 25-75 s per node on one CPU of the sandbox (the longer on thick dendrites), some 10-15 minutes for the 13 nodes.

What to expect (synthetic profiles, SPEC Block 11; one noise draw each, so indicative): when the true blur is a Gaussian of FWHM 0.5 um, twice Block 5's, Block 5 gives 0.82 um for a 0.6 um tube and 1.47 um for a 1.4 um one; the Gaussian of fitted width gives 0.70 and 1.39 um (its FWHM 0.43 and 0.50 um), the heavy-tailed kernels at the fixed width (Lorentzian, generalized normal $\beta$ = 1) 0.67-0.69 um on the thin tube, and the weights change $d$ by a few hundredths on a clean single dip; the 3.0 um tube is near 3.0 um for every variant. Shapes often run to a bound (a Student $\nu$ of 100 or a Voigt $r$ of 0.01 is the Gaussian again; a skew of 10 at the fixed width widens the kernel on one side): the flag says so. Where a darker neighbour merges with the dip, the unweighted fit takes both as one tube and the re-centred weights at $\lambda$ = 0.25 um can move onto the neighbour: check $v_0$ in the calipers. What to look at: on the thin nodes, which variants bring $d$ down and whether their curves still follow the profile; whether the fitted widths agree across nodes (a common width would point at the blur rather than at the dendrites); on the trunks, whether the variants agree.

### Cell 4m: each fit variant's own Allen-guided iteration, its edges and its plane [added 2026-10-14]

```python
# Cell 4m: each fit variant's own Allen-guided iteration, its edges and its plane (needs Cells 0 and 1 and Cell 4j's output)
import json
import os
import pandas as pd
from IPython.display import Image, display
ITER_DIR = OUT + "/pilot/planeiter/bydiameter_m3"   # Cell 4j's run at m = 3: its LINE_MULT and MAX_ROUNDS are inherited from its records
out = ITER_DIR + "/variter"
run("scripts/variant_iteration.py", "--specimen", SPECIMEN, "--iter-json", "%s/planeiter_%d.json" % (ITER_DIR, SPECIMEN),
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
p = "%s/variter_%d.csv" % (out, SPECIMEN)
if os.path.exists(p):           # d_iter (um) per node: Cell 4j's own (Block 5's fit) and each kernel unweighted
    t = pd.read_csv(p)
    u = t[t["weight"] == "none"]
    tab = u.pivot(index="node_id", columns="kernel", values="d_iter_um")
    display(tab.loc[list(dict.fromkeys(t["node_id"])), list(dict.fromkeys(u["kernel"]))].round(2))
p = "%s/variter_%d.json" % (out, SPECIMEN)
if os.path.exists(p):
    with open(p) as f:
        done = [r for r in json.load(f) if "skipped" not in r]
    for r in done:       # each node's edge stack and planes, in Cell 4j's order (the pilot's d_hat)
        d = r["d_iter_cell4j_um"]
        print("node %d: Cell 4j's k_iter %s, d_iter %s um%s" % (r["node_id"], r["k_iter_cell4j"],
                                                              "n/a" if d is None else "%.2f" % d,
                                                              "".join("; " + n for n in r["notes"])))
        display(Image(filename=r["png_stack"]))
        display(Image(filename=r["png_planes"]))
```

The user's request of 2026-10-14 (15:35), on the figures of Cell 4l, and the answers to three questions (D-047): one table per node of the variants at the fitted width -- every kernel, unweighted and centre-weighted, as their edges $v_0 \pm d/2$ (a 50-row stack); each of the 50 runs its own Allen-guided iteration, the way Cell 4j searches the plane and sizes the measuring line, with Cell 4j's `LINE_MULT` and `MAX_ROUNDS` inherited (read from its records, so the two never disagree); and the figure of the fitted dendrite at each variant's own plane with its $d$. For a variant $V$: round 1's line is $\pm m\, r_A$ (from Allen's $2 r_A$), as in Cell 4j; each later line is $\pm m\, d / 2$ with $d$ $V$'s own fit at the plane the previous line picked (Block 5's inputs there: the profile through the pilot's fitted centre, the tilt, that plane's $\bar B$); the plane is the one of the largest gradient energy over the line; the iteration stops when a plane repeats or after `MAX_ROUNDS`. Cell 4j's own iteration (Block 5's fit) is redone first with the same machinery, as the reference.

Nothing is measured anew: the cell reads Cell 4j's `planeiter_529878215.json` in `ITER_DIR` and rebuilds each node's squares as Cell 4j made them, from the cached crops. A variant whose $d$ is larger than Block 5's can need a longer line than Cell 4j ever drew, and so a larger square: on the trunks some new crops may be fetched. The reference must give Cell 4j's planes and `d_iter_um` to the last digit; a difference is printed in the node's notes. To try fewer variants, add for instance `"--families", "lorentz,gennorm"` or `"--lambdas", "0.5"` to the `run(...)` call (the Gaussian is always kept); `"--nodes", "8497,3856"` for some nodes only; `"--width-modes", "fixed,fitted"` adds the fixed-width variants (100 then).

Output in `ITER_DIR/variter/`: per node `variter_stack_<id>.png`, one row per variant (Cell 4j's own first, in ink): its label and $d$, its edges $v_0 \pm d/2$ (um from the pilot's fitted centre) at the plane its own iteration chose, and the planes its rounds picked (grey) ending on that plane (filled), against $k^*$ (red) and the SWC plane (dotted); and `variter_planes_<id>.png`, each variant's chosen plane with the line that chose it (black, through the SWC node +) and the fitted dendrite's two edges drawn along the branch (the kernel's colour), titled with the plane and $d$: Cell 4j's own on top, then one row per kernel and one column per weighting, every panel on the same view; `variter_529878215.csv` (one row per node and variant: `k_iter`, `d_iter_um`, `iter_status`, `planes`, the fit's `v0_um`, `fwhm_um`, `shape`, `ratio_ref` = $d$ / the reference's, ...) and `variter_529878215.json` (with each variant's trajectory, edges and line). The cell shows a table of $d$ per node for Cell 4j's own and each kernel unweighted, then each node's two figures.

Expect one `[variter] node N: Cell 4j redone k_iter K d D um (converged); 50 variants: k_iter K1 xn1, ...; d_iter A-B um; unweighted ...; P planes fitted, S squares; T s` line per node, then `[variter] 13 nodes, 0 skipped; 50 variants per node, each with its own iteration; Cell 4j's iteration redone identically on 13 of 13` and `[variter] files in ...; crops: ...`. The fits at a plane are shared by every variant that visits it, so the time grows with the number of planes visited: about 20 s per plane on a thin node and 1 minute on a thick one (one CPU of the sandbox); expect some 10-40 minutes for the 13 nodes.

What to expect (synthetic tubes, SPEC Block 11): there, every variant's iteration ends on Cell 4j's plane (0.8 um tube: $d$ 0.81-0.86 um; 3.0 um tube: $d$ 3.20-3.35 um), and the reference redoes Cell 4j exactly. On Allen's data the variants whose fitted width takes up most of the dip (Cell 4l: nodes 8481, 8482 and 3851 gave $d$ of 0.2-0.6 um with widths of 1-2 um) size short lines, which may pick other planes. What to look at: in the stack, the rows whose filled plane is not Cell 4j's; in the images, whether each variant's edges follow the dendrite at the plane it chose.

## Cell 5: camera-chain inputs

```python
run("scripts/camera_calibration.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", OUT + "/camera",
    "--pilot-summary", "%s/pilot/pilot_summary_%d.json" % (OUT, SPECIMEN))
```

Expect `"renderer": {...}` with three fields:
- ~~`jpeg_qtables_file`~~ `jpeg_qtables`: Allen's tables, read from the cached crops. `table_sets` should show a single set, and `pillow_version` 8.3 or newer. [corrected 2026-10-07: the old field held a file path that the renderer never read, so a table built on it would have used quality 85; the tables themselves now go into the configuration, and `jpeg_qtables_<specimen>.json` stays as a record.]
- `background_B_gl`: the median of the masked block medians.
- `noise_sd_gl`: the injected SD whose post-JPEG flat-field SD matches the real background's. JPEG removes much of white noise, so the background SD read on the images is smaller than the injected SD. This value is an upper bound, because tissue texture adds to the real background SD.

Black level and gain cannot be identified from these crops. They stay as configured. If `n_tables` is 2, the crops are colour JPEGs: the synthetic planes are grey and use the first (luminance) table only (SPEC §8).

## Cell 6: Allen's radius distribution (sets d_max)

```python
os.makedirs(OUT + "/radius", exist_ok=True)
run("scripts/allen_radius_distribution.py", "--fetch", SPECIMEN, "--cache-dir", CACHE_DIR,
    "--out", "%s/radius/radius_summary_%d.csv" % (OUT, SPECIMEN),
    "--out-nodes", "%s/radius/radius_nodes_%d.csv" % (OUT, SPECIMEN))
```

Expect `pooled dendrite diameter (um): p0=..., ..., p100=...`, a text histogram and `wrote ...`. [corrected 2026-10-07: v1 passed only `--out`, so the per-node CSV went to the default `allen_radius_nodes.csv` inside the clone, lost with the runtime.]

This step feeds `phantom.d_range_um`, which is temporary until now (D-024 (iii), SPEC §8).

## Cell 7: kernel calibration -- near field (Block 10), far field (Block 15), figures

[rewritten 2026-10-23, v22; the v18 text is kept in the changelog row]

```python
# The stripe's knobs (D-101, D-108) and the candidates' search over the whole cell (D-109, D-112): the configuration's
# calibration.* fields, here at their defaults. Change one and run the cell again: the file below goes to real, fit,
# far-fit and figures, and real prints `calibration knobs: ...` with the values it ran with. Cell 15 and the davinci
# array read the same file when it is passed (runbook, "the stripe's knobs").
CAL_KNOBS = {
    "calibration.background_support": "stripe",   # "stripe" (D-101) or "square" (before 2026-10-28: the 3-sigma far mask)
    "calibration.isolation_depth_gl": 0.3,        # tau (gl): a pixel is shaded where a traced segment's predicted shadow exceeds it (D-108)
    "calibration.stripe_half_widths_um": [1.0, 2.0, 4.0, 8.0, 12.0],   # the band's half-widths W tried (um); most clear pixels wins
    "calibration.stripe_min_pixels": 500,         # a plane's stripe is usable with at least this many clear pixels ...
    "calibration.stripe_min_side_um": 3.0,        # ... and at least this clear run (um) on each side of the own cylinder
    "calibration.patch_half_um": 25.0,            # the patch's half-side (um): the runs end at its edge (larger: bigger crops)
    "calibration.node_phi_max_deg": 10.0,         # flatness, near set (deg)
    "calibration.node_phi_max_deg_far": 15.0,     # flatness, far set (deg)
    "calibration.near_candidates": "cell",        # near set: "cell" found over the whole cell from the trace (D-112), "table"
    "calibration.near_search_spacing_um": 5.0,    # the near candidates kept at least this far apart along their stretch (um)
    "calibration.far_candidates": "cell",         # far set: "cell" found over the whole cell from the trace (D-109), "table"
    "calibration.far_search_spacing_um": 5.0,     # the far candidates kept at least this far apart along their stretch (um)
}
import json
assert all(k.startswith("calibration.") for k in CAL_KNOBS), "CAL_KNOBS: calibration.* fields only"
os.makedirs(OUT + "/calibration/figures", exist_ok=True)
CFG_CAL = OUT + "/calibration/calibration_config.json"
with open(CFG_CAL, "w") as f:
    json.dump({"calibration": {k.split(".", 1)[1]: v for k, v in CAL_KNOBS.items()}}, f, indent=1)
CAL_CFG = ["--config-json", CFG_CAL]
SCANS = "%s/calibration/scans_%d.json" % (OUT, SPECIMEN)
CALIB = "%s/calibration/calibration_%d.json" % (OUT, SPECIMEN)
FARFIT = "%s/calibration/far_fit_%d.json" % (OUT, SPECIMEN)
PATCHES = "%s/calibration/patches_%d.npz" % (OUT, SPECIMEN)
rc = run("scripts/calibrate_kernel.py", "real", "--specimen", SPECIMEN, "--nodes-csv",
         "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--cache-dir", CACHE_DIR, "--out", SCANS,
         "--registration-json", REG_JSON, *ALIGN, *CAL_CFG, check=False)
if rc == 0:
    rc = run("scripts/calibrate_kernel.py", "fit", "--scans", SCANS, "--out", CALIB, *CAL_CFG, check=False)
if rc == 0:
    run("scripts/calibrate_kernel.py", "far-fit", "--scans", SCANS, "--calibration", CALIB, "--out", FARFIT, *CAL_CFG,
        check=False)
    fig_args = ["--scans", SCANS, "--calibration", CALIB, "--out-dir", OUT + "/calibration/figures"] + CAL_CFG
    if os.path.exists(FARFIT):
        fig_args += ["--far-fit", FARFIT]
    if os.path.exists(PATCHES):
        fig_args += ["--patches", PATCHES]
    run("scripts/calibrate_kernel.py", "figures", *fig_args, check=False)
import glob
from IPython.display import Image, display
for path in sorted(glob.glob(OUT + "/calibration/figures/isolation/isolation_overview_*.png")):   # D-098.4
    display(Image(filename=path))
```

[added 2026-10-28, v33; D-098.4] **The isolation figures.** As `real` evaluates each candidate it draws the patch its background is fitted on -- the ±25 µm square about the pilot's centre at the pilot's sharpest plane $k^*$ -- in two panels: (a) the traced segments over it, the node's own branch in blue, each neighbour in orange when its predicted shadow in the patch exceeds 0.03 gl in some plane of $k^* \pm 3$ (near) or $\pm 16$ (far), the number that makes `not_isolated`, and in grey below it; (b) the far pixels (unshaded) the planar background is fitted on, with the fitted plane. The title gives the verdict and the deepest predicted shadow against 0.03 gl; the footer lists the neighbours above it. One PNG per candidate in `OUT/calibration/figures/isolation/<set>_node_<id>.png`, overview pages `isolation_overview_<n>.png` (36 thumbnails each, shown by the cell), and `real` ends with `isolation figures: N candidates (k with a neighbour's predicted shadow above 0.03 gl) -> ...`. It costs one crop per candidate that is not scanned (a scanned one is in the cache). A segment joining a dendrite to the soma or to another stretch counts as a neighbour: a candidate near its dendrite's root or a branch point is shaded by it, and the figure shows which. [2026-10-28, v35; D-101] **The stripe.** Panel (b) now shows the stripe the plane's background is fitted on -- the band along the measuring line ($W$ = 1, 2, 4, 8 or 12 µm along the dendrite, the one with most pixels) from the edge of the node's own fitted cylinder outward on each side to the first pixel column a traced neighbour's predicted shadow reaches above 0.03 gl -- with its half-width, the two clear runs and its pixel count in the title; an orange neighbour bounds the stripe and no longer rejects the node by itself. The node's verdict is the stripe at $k^*$: `not_isolated:too_few` (under 500 pixels) or `not_isolated:side_short` (a run under 3 µm). The overview's thumbnails are orange when the stripe is not usable, their label its pixel count; `real` ends with `isolation figures: N candidates (k whose stripe at k* is not usable: fewer than 500 px or a side under 3.0 um) -> ...`. Within the scan each plane has its own stripe (the far planes' shadows are wider), a plane whose stripe fails has status `bg_few` and the scan's `stripe` record (in `scans_<specimen>.json`) says why. Plainly: instead of asking whether the whole 50 µm neighbourhood is empty, each plane asks how far one can walk along the measuring line, on each side, before stepping on anyone's shadow, and fits the background on that walk. The square rule of D-078.1 / D-081.1 is back with `{"calibration": {"background_support": "square"}}` as `--config-json`.

[added 2026-10-28, v36] **The stripe's knobs, and every plane's stripe.** The cell's first lines, `CAL_KNOBS`, are the stripe's settings at the configuration's defaults -- `calibration.background_support` ("stripe", or "square" for the rule before D-101), `isolation_depth_gl` ($\tau$, 0.03 gl [0.3 gl since v37, D-108]: a pixel is shaded where a traced segment's predicted shadow exceeds it), `stripe_half_widths_um` (the $W$ grid), `stripe_min_pixels` (500), `stripe_min_side_um` (3 µm), `patch_half_um` (25 µm: the runs end at the patch's edge; larger means bigger crops) -- and the flatness bounds `node_phi_max_deg` (10°) and `node_phi_max_deg_far` (15°). The cell writes them to `OUT/calibration/calibration_config.json` and passes that file to all four steps; `real` prints `calibration knobs: background_support stripe, tau 0.03 gl, W 1, 2, 4, 8, 12 um, at least 500 px and 3 um per side, patch +-25 um; flatness phi <= 10 deg near, 15 deg far` [corrected 2026-10-29, v37: `tau 0.3 gl`, and the line ends `; far set from the whole cell, at least 5 um apart` (D-109)] -- check that it shows what you set. Each isolation figure has a third panel, (c), the stripe of every plane of the stage: above, its clear pixels against the 500 (log scale); below, its runs along the measuring line beyond the node's own cylinder (light blue), one bar per side, as wide as that plane's $W$, dark grey where the plane's stripe is usable and orange where not, a tick where a side reaches 3 µm and the measuring line's ends dashed; the planes the set's fit does not use are faint (the far fit uses $|k - k^*| \ge 5$, the dip's area $|k - k^*|$ = 4-6 half-faint). Panel (b) marks the $+v$ end of the line. The overview's labels end with $n/N$, the fit's planes whose stripe is usable (orange still means: not usable at $k^*$), and after the figures `real` prints `far-set candidates with a usable stripe in at least 6 of the far fit's planes (|k - k*| >= 5): m of n` -- the far fit drops the others (`few_planes`).

[added 2026-10-29, v37; D-108, D-109] **$\tau$ = 0.3 gl, and the far set from the whole cell.** `isolation_depth_gl` is 0.3 gl (D-108, PROVISIONAL: judge it on panel (c)); every 0.03 gl in the paragraphs above is the knob's value, 0.3 gl since this version. On the pilot's own reconstruction the stripe at 0.3 gl keeps 92 of the 138 near candidates at $k^*$ (21 at 0.03 gl) **[run, sandbox: `checks/d105_2026-10-28/`]**. The far set no longer comes from the pilot table, whose far-range rows (31, from Cell 4b's stretches) lie on two stretches, and none of the 7 among them in S and flat had a usable stripe in any far-fit plane (D-105's check). With `calibration.far_candidates = "cell"` (the default; "table" brings the table's back) `real`, before it fetches anything, judges every dendrite node of the cell whose Allen diameter lies in (0.6, 0.8] µm from the trace alone -- the trace's tilt over ±2 µm ($\le 15°$), the stripe at the trace's plane round($z$ / 0.28 µm), and at least 6 usable stripes among the far fit's 24 planes ($5 \le |k - k_{\rm tr}| \le 16$) -- with the pilot table's median $\bar B$ and $\hat\mu$ over its rows in S (no pilot fit exists yet for these nodes), keeps the passing nodes at least `far_search_spacing_um` = 5 µm apart along their stretch (the most usable planes first), and hands them to the scan with the table's near candidates; each is then measured (its pilot) and judged again like any candidate. Expect, after the `calibration knobs:` line: `far search: the cell's typical background B gl and absorption m /um (the pilot table's medians in S)`, `far search: n far-range nodes judged` every 100 nodes [corrected 2026-10-29, v38: `candidate search: the cell's typical background ...` once for both searches, and `far search: n of N far-range nodes judged (t min)` every 500 nodes], then `far search over the cell: N far-range nodes; a flat by the trace (phi <= 15 deg); b with a usable stripe at the trace's plane; c with at least 6 usable far-fit planes; m chosen, at least 5 um apart (t min) -> .../calibration/far_search_529878215.csv` (one row per far-range node: its trace's position, plane, heading and tilt, `reason` -- empty when it passed, else `not_flat`, `not_isolated:too_few`, `not_isolated:side_short` or `few_planes` --, its stripe's pixels at the trace's plane, its usable far-fit planes, `chosen`), then `N candidate nodes` as before. In the sandbox the pilot gave N = 723, a = 502, b = 90, c = 74, m = 19 in 8.2 min [5.1 min since v38: the masks skip the segments that cannot reach the patch, the result unchanged], with B = 159.0 gl and m = 0.258 /um **[run, sandbox: the pilot's SWC, no image, no registration offset]**; Colab's search should take about as long, and then each of the 19 costs its pilot's z-block and the 33 far planes, about 46 crops. For those 19 the model predicts that the neighbours' shadows leave on the ±10 µm line, once the plane is fitted on the stripe, a median error of 0.94 gl at $|k - k^*|$ = 5 against a dip of 5.0 gl, and 1.9 gl at 16 against a dip of 1.6 gl **[run, sandbox: the model's shadows, not Allen's grey levels]**: the outermost planes are where the far fit's SNR rule (Block 15.3) and the width-by-plane figure will show whether they hold. Send back: the `calibration knobs:` line, the three `far search` lines and `far_search_529878215.csv`, the `real` counts and reasons line and its far-set line, the `fit` and `far-fit` reports, the figures and the isolation overview pages. Cell 15 and the davinci array do not search (Block 19.3 takes each cell's far candidates from its node table) [corrected 2026-10-29, v38: they do since D-112, below].

[added 2026-10-29, v38; D-112] **The near set from the whole cell too.** The user: "Remember that the whole cell's nodes must be evaluated for both near and far field calibration". With `calibration.near_candidates = "cell"` (the default; "table" brings the table's near candidates back) `real` judges, before the far search and before any download, every dendrite node of the cell with Allen's $2r \le 0.6$ µm from the trace alone -- the tilt ($\le 10°$, the near bound), the stripe at the trace's plane, and at least 3 usable stripes among the trace's plane $\pm 3$ (the near fit's own minimum of finite widths per node; the pairs $\pm 1$, $\pm 2$, $\pm 3$ evaluated in turn, stopping once 3 are usable or can no longer be) -- keeps the passing nodes at least `near_search_spacing_um` = 5 µm apart along their stretch (the most clear pixels at the trace's plane first), and hands them to the scan with the far set's; each is then measured and judged like any candidate, and each set stops at 30 scanned nodes (the others are counted in one line, `not measured, their set already holding 30 scans (set_full): n near, n far`). Expect, after `candidate search: ...`: `near search: n of N near-range nodes judged (t min)` every 500 nodes, `near search over the cell: N near-range nodes; a flat by the trace (phi <= 10 deg); b with a usable stripe at the trace's plane; c with at least 3 usable planes of the trace's plane +- 3; m chosen, at least 5 um apart (t min) -> .../calibration/near_search_529878215.csv`, then the far search's lines. On the pilot's own reconstruction, in the sandbox: 7500 near-range nodes, 3856 flat by the trace, 1960 usable at the trace's plane, 1947 with at least 3 usable planes, 380 chosen on 48 stretches, in 15.4 min; with the far search (now 5.1 min, its records unchanged) Cell 7 spends about 20.5 min before its first download **[run, sandbox]**. The pre-flight now prints the typical background and absorption the search uses. S3's calibration step (Cell 15 on the pilot, the davinci array on every cell) searches the same way, its records in `<cell>/calibration/`, so Cell 15 repeats Cell 7's search time before its first download. [corrected 2026-10-30, v39: not since D-115.6 -- Cell 15 reuses Cell 7's search when both were made from the same trace under the same settings; see Cell 15] The own cylinder (F2: the union of the trace's and the fitted one) is confirmed, and F3 (the pilot's tilt error on thin nodes) accepted without change (D-113).

[added 2026-10-30, v39; D-115.4-D-115.6] **What Cell 7 writes since D-115.** (1) `fit`'s anchor SE: the bootstrap reads each resample's anchor from $\hat G^*(\delta_0)$ alone (D-115.4); a resample is dropped only when its fit fails or $\sigma_{\rm r}(0)^2 + \hat G^*(0.84\ \mu{\rm m}) \le 0$, and the report's `n_boot_ok` counts the others. (2) `scans_529878215.json`'s `meta.nodes`: each measured candidate's `pilot`, the values its verdict read (D-115.5); a `set_full` node has `pilot: null`. (3) Beside each search CSV its record, `near_search_529878215.json` and `far_search_529878215.json`: the trace's SHA-256 (`key.swc_sha256`), the settings the search reads (`key.config`: the acquisition, calibration, measure and renderer sections), the CSV's SHA-256, and the search's $\bar B$, $\hat\mu$ and transform (D-115.6). Cell 15 reuses a search through its record: a search CSV edited by hand is searched again. [added 2026-10-30, v40; D-115.3] (4) After the scan's counts, `goodness of fit (D-115.3: core-fit RMS > 1.20 x the background's noise SD on its support): planes dropped as poor_fit -- near p of n planes, on m nodes (k of them now below 3 finite widths); far p' of n' planes, on m' nodes`. A dropped plane shows `poor_fit` in the every-Gaussian figure and as a dropped plane in the width-by-plane figure. Expect p around 5 % of n (the estimate on this morning's scans); far more points to wide or deep dips the Gaussian core misfits (the check's caveat), and the statistic needs your look before the kernel is taken. Send back that line with the others.

**Before you run it (2026-10-28, later):** [superseded 2026-10-29, v37: $\tau$ is 0.3 gl (D-108) and the far set is searched over the cell (D-109), above; this paragraph is the evidence at 0.03 gl] on the pilot's own reconstruction, the stripe as built keeps few nodes **[run, sandbox: the pilot's SWC as the scene, the pilot table's centres, $k^*$, $\bar B$, $\hat\mu$; no image; `checks/d105_2026-10-28/`]**: of the 138 near candidates 21 are usable at $k^*$ (91 `too_few`, 26 `side_short`), and no plane of the 7 far candidates has a single clear pixel. With $\tau$ = 0.3 gl the near set keeps 92 and with 1 gl 117, the far set 0 and 4 (16-24 of their 24 fit planes); the predicted error the neighbours' summed shadow leaves on the measuring line after the plane is fitted is then, over the usable near nodes, a median 0.24 gl at 0.3 gl and 0.35 gl at 1 gl (90th percentiles 0.8 and 2.1 gl), over the far nodes' planes 3.6-9.4 gl per node at 1 gl -- against the pilot's median background noise SD of 5.9 gl (D-101's data). The cause is the out-of-focus dendrites: a dendrite of radius 0.3 µm lying 9 µm above or below the plane has, with $\bar B$ = 200 gl and $\hat\mu$ = 1 /µm, a predicted shadow spread over $\sigma_{\rm r} \approx$ 7 µm that stays above 0.03 gl out to 21 µm from its axis, so at most nodes some such shadow covers the node itself. What to do about it is the open decision of this version (`isolation_depth_gl` is one of the knobs); with the defaults the near set may reach 10 nodes, the far set will not.

The calibration nodes are chosen by Allen's diameter $D_{\rm A} = 2 r_{\rm A}$ (D-061.4, D-067.4; thresholds PROVISIONAL): the near-field set, $D_{\rm A} \le 0.6$ µm, is scanned over $k^* \pm 3$ (offsets to 0.84 µm); the far-field set, $0.6 < D_{\rm A} \le 0.8$ µm, over $k^* \pm 16$ (offsets to 4.5 µm; D-026, D-078). Both sets must be in S, flat (~~$\varphi \le 5°$ for both, `node_phi_max_deg`, D-081.3~~ [corrected 2026-10-28, v35, D-101.1: $\varphi \le 10°$ for the near set, `node_phi_max_deg`, and $\le 15°$ for the far set, `node_phi_max_deg_far`]) ~~and faint ($\hat\alpha \le 0.5$)~~ by the pilot's columns [corrected 2026-10-28, D-094 (the user: "Remove both faint"): no optical-depth criterion, and the flag `faint` of S (a dip under 6 grey levels) no longer fails a calibration node; ~~S itself, for the table and the per-node diameters, keeps it.~~ [corrected 2026-10-28, D-096: S drops the flag too, for the table and the per-node diameters; `faint` is a label] A config JSON `{"calibration": {"node_alpha_check": true}}` passed to `calibrate_kernel.py real` as `--config-json` restores $\hat\alpha \le 0.5$, and `"node_faint_flag_check": true` the flag], and then on the trace (D-079, D-081.1): isolated (no other traced segment or the soma casts a predicted shadow deeper than `isolation_depth_gl` = 0.03 grey levels inside the ±25 µm patch in any scanned plane; the node's own branch is exempt). ~~straight (within one radius of the trace's own TLS line over ±10 µm)~~ [corrected 2026-10-27, D-093: no straightness test -- the calibration evaluates only the measuring line and flatness stays strict; a config JSON `{"calibration": {"node_straight_check": true}}` passed to `calibrate_kernel.py real` as `--config-json` restores it at `node_straight_tol_radii` = 2 radii (add `"node_straight_tol_radii": 4.0` for 4). On the pilot none of Cell 7's 68 candidates was a calibration node: the 57 lines the user pasted had all failed it at one radius, 46 of them isolation too (the user's run of 2026-10-27)]. Each scan reads a ±25 µm patch per plane, fits a planar background $\hat B_k(x, y)$ to the pixels farther than $r + 3\sigma_{\rm r}(\delta)$ from every traced segment (D-078.1) and fits the dip's Gaussian core iteratively, up to `far_core_max_passes` = 10 passes, stopping when the width moves by less than `far_core_stop_tol` = 1 SE (D-077, D-078.2).

Expect from `real`: `N candidate nodes`, one line per node (`node i: scanned (near field)`, `scanned (far field)` or `not a calibration node (...)` with the reasons: the selection flags of Block 6 (`status`, `crossing`, `stack_edge`, `dark`, ...), `allen_diameter_missing` / `allen_diameter_out`, `not_flat`, `not_faint`, `not_straight`, `not_isolated`; since D-093 and D-094, `not_straight`, `not_faint` and the flag `faint` appear only when a configuration switches their test on; since D-101 (v35) the isolation reason is the stripe's, `not_isolated:too_few` or `not_isolated:side_short`, and the plain `not_isolated` only under `background_support = "square"`), then `n of N candidates scanned: a near-field, b far-field; reasons {...}; in ... min -> ...`, `example near-field node i: patch images kept` (and the far one) and `patches -> .../patches_<specimen>.npz`. From `fit`: `growth fit (gaussian_core_width2, symmetric, cubic): N nodes`, the knot table, `residual RMS by |k - k*|`, `table (cut at 0.84 um, D-080.1)` with its `sigma` row (`increasing` or `NOT increasing`) and `anchor sigma_0_hat = ... um, SE ... um (node bootstrap, n of 1000 resamples, seed 0)` (D-081.7). From `far-fit`: `far planes: J nodes kept ([planes per node]), d dropped {...}; P planes; ... not converged, ... below the SNR rule`, `far fit: gamma_hat = ... +- ... (bootstrap ... over n resamples, anchor term ... x ...), t_{J-1} interval [lo, hi]; range ... um; 3 IRLS rounds, success`, the per-node $c_j$ and $s_j$, the residual RMS by |k − k*|, `per-sign: gamma_+ ..., gamma_- ...; quadratic comparison: gamma ..., gamma_2 ... per um (labelled comparison, D-081.6)` and `production kernel: table to 0.84 um, continuation linear with slope ...`. From `figures`: `wrote .../width_by_plane_near.png`, `width_by_plane_far.png`, `example_near.png`, `example_far.png`, `kernel_law.png` and one `gaussians/<set>_node_<id>.png` per scanned node.

What to read, in order (D-080.2, D-081.5): the width-by-plane figures -- every node's $\hat\omega_{j,k} - \hat c_j$ in light grey, the median in colour, the fitted law (near: $\hat G$; far: $(\hat\sigma_0 + \hat\gamma x)^2$) -- say whether the near table and the far straight line hold; the example figures show the node nearest the median with its plane images, the far-pixel mask and the fitted window; the quadratic comparison's $\gamma_2$ is printed as a comparison only. The slope for the production kernel, `renderer.kernel_continuation_slope` in Cell 8, is yours to set from `far fit: gamma_hat`; the default 0.79 stays until you do.

Fewer than two near-field nodes stop `fit` (`fit: need at least two nodes ...`); fewer than two far-field nodes with `far_min_planes_per_node` = 6 kept planes each stop `far-fit`. Measure more stretches in Cell 4b if so [2026-10-29, v37: for the near set; the far set is searched over the whole cell (D-109)]. With the 0.03 grey-level isolation rule [0.3 gl since v37, D-108] the sets may be small on a dense cell: the `reasons {...}` counts say which rule removed the nodes; relaxing `isolation_depth_gl` is a decision, not a default. A failure here does not stop the notebook (`check=False`).

The first-stage table is **not** a calibrated kernel on its own: on rendered thin phantoms its growth is about 0.01 µm² low within two planes of focus (SPEC Block 10). Read it next to the default table; the renderer's law is the near table to 0.84 µm continued linearly with the far slope (D-080.1).

[added 2026-10-26, v28; D-088, Block 19] **10 to 30 nodes per set.** `real` takes the candidates in a seeded order (`calib_order_seed` with the specimen) and stops each set at 30 scanned nodes (`calib_max_nodes_per_set`, D-088.3): the rest are `not a calibration node (set_full)`, never measured. The scan line reads `n of N candidates scanned: a near-field, b far-field (at most 30 each, D-088.3); reasons {...}`. Below 10 nodes in a set (`calib_min_nodes_per_set`) the set is insufficient: `fit` prints `INSUFFICIENT: J near-field nodes < 10 (D-088.3): a diagnostic, not the production kernel's table`, `far-fit` prints `no production kernel: insufficient: ...`, and both files record `meta.sufficiency`. One kernel for every cell (D-088.1): this cell, on the pilot cell, is the test for now (D-088.2); Cell 15 pools the nodes of the survey's cells.

## Cell 8: the production configuration (your decisions)

Copy the default configuration and change only the fields you have decided:
- `d_range_um` (from Cell 6);
- `mu_range_per_um` (from Cell 4);
- the camera fields (from Cell 5);
- `jitter_xy_um` and `jitter_z_um` (from Cell 3).

Record each choice as a decision before building the table on it.

```python
import importlib, json
import pandas as pd
import allen_diameter.config as C
C = importlib.reload(C)                      # the code of this session's pull


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


REG = load("%s/registration/registration_summary_%d.json" % (OUT, SPECIMEN))
PILOT = load("%s/pilot/pilot_summary_%d.json" % (OUT, SPECIMEN))
CAM = load("%s/camera/camera_%d.json" % (OUT, SPECIMEN))
try:
    RAD = pd.read_csv("%s/radius/radius_summary_%d.csv" % (OUT, SPECIMEN)).iloc[0].to_dict()
except FileNotFoundError:
    RAD = {}
base = C.default_config()
cam = CAM.get("renderer", {})
print("suggested by the runs                                      [current default]")
print("  d range      Allen dendrite diameter p99 %s um, max %s um   [%s]"
      % (RAD.get("dend_p99"), RAD.get("dend_p100"), base.phantom.d_range_um))
print("  mu range     %s   [%s]" % (PILOT.get("suggested_phantom_mu_range_per_um"), base.phantom.mu_range_per_um))
print("  jitter       %s   [%s, %s]" % (REG.get("suggested_jitter_um"), base.phantom.jitter_xy_um, base.phantom.jitter_z_um))
print("  background   %s   [%s]" % (cam.get("background_B_gl"), base.renderer.background_B_gl))
print("  noise SD     %s   [%s]" % (cam.get("noise_sd_gl"), base.renderer.noise_sd_gl))
print("  JPEG tables  %s table set(s) in the cache, Pillow %s   [none: quality %d]"
      % (len(CAM.get("table_sets", [])), CAM.get("pillow_version"), base.renderer.jpeg_quality))

DECIDED = {
    # "phantom.d_range_um": (0.2, 4.0),                                         # Cell 6 sets d_max (D-024 iii)
    # "phantom.mu_range_per_um": PILOT["suggested_phantom_mu_range_per_um"],   # Cell 4 (procedure s.3.10)
    # "phantom.jitter_xy_um": REG["suggested_jitter_um"]["jitter_xy_um"],      # Cell 3 (D-024 vii)
    # "phantom.jitter_z_um": REG["suggested_jitter_um"]["jitter_z_um"],
    # "renderer.background_B_gl": cam["background_B_gl"],                      # Cell 5
    # "renderer.noise_sd_gl": cam["noise_sd_gl"],                              # Cell 5 (an upper bound)
    # "renderer.jpeg_qtables": cam["jpeg_qtables"],                            # Cell 5
}
cfg = C.with_overrides(base, DECIDED)
PENDING = ("phantom.d_range_um", "phantom.mu_range_per_um", "phantom.jitter_xy_um", "phantom.jitter_z_um",
           "renderer.background_B_gl", "renderer.noise_sd_gl", "renderer.jpeg_qtables")
left = [k for k in PENDING if k not in DECIDED]
print("\nstill at the provisional default: %s" % (", ".join(left) if left else "none"))
with open("%s/config_production.json" % OUT, "w") as f:      # one configuration, sigma_fit 0.099 um (D-057 ii)
    f.write(cfg.to_json())
print("config_production.json   sigma_fit %.3f um -> table bias_table_%s"
      % (cfg.measure.sigma_fit_um, cfg.signature_hash("estimator")))
```

`config.with_overrides` [added 2026-10-07] checks every name, validates the result, and puts each value in the field's declared form: a JSON list becomes the tuple the configuration holds, and an integer given for a float field becomes a float, so the same value always gives the same hash. ~~The cell writes one configuration per in-focus blur of the D-023 study: `config_production.json` (the deliverable, σ_fit = 0.099 µm), `config_sigma0.080.json` and `config_sigma0.125.json`. Expect three lines `... -> table bias_table_<hash>`.~~ [corrected 2026-10-21, S2.2: D-057 (ii) drops the σ_fit study; the cell writes one configuration, `config_production.json` (σ_fit = 0.099 µm). Expect one line `config_production.json ... -> table bias_table_<hash>`.]

The estimator hash names the table. A real fit refuses a table with a different hash (procedure §3.11).

## Cell 9: the production table (davinci, not Colab)

The table is built with `scripts/pbs/build_table.pbs`: ~~100 tasks of 20 replicates~~ [corrected 2026-10-21, Block 14.2: an array of tasks, each rendering a chunk of draws with every realization, $G$ draws x `phantom.n_realizations`, sized from the davinci pilot], one core and 4 GB each, then the merge ~~on the login node~~ [corrected 2026-10-21, Block 14.6: in a job or an interactive session]. Copy `config_production.json` to the cluster, to a path **without spaces** (for example `$HOME/diam/config_production.json`; `qsub -v` takes a comma-separated list, and a value with a space is fragile there), and pass it as `DIAM_CONFIG_JSON`. The commands are in SPEC §6 and in the paths reference (project knowledge, `claude/TEEG_diameter_HPC_paths_reference.md`).

Update the code and check Pillow first [added 2026-10-07: 8.3 or newer, the order of the JPEG tables]:

```
cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement" && git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only && conda activate spine_env && python -c "import PIL; print('Pillow', PIL.__version__)" && python tests/smoke/test_smoke_config.py && python tests/smoke/test_smoke_camera_tables.py
```

Probe first:

```
qsub -J 0-1 -v DIAM_N_DRAWS=2,DIAM_N_TASKS=2,RUN_TAG=probe,DIAM_CONFIG_JSON=<path> scripts/pbs/build_table.pbs
```

Each replicate takes 10-140 s (sandbox timing). ~~`DIAM_N_TOTAL` (default 2000) sets the number of replicates; `phantom.n_replicates` in the configuration is a record only.~~ [corrected 2026-10-21, Block 14.2: `DIAM_N_DRAWS` (default 100000) sets the draws of a submission and `DIAM_DRAW_OFFSET` where it starts; every draw is rendered `phantom.n_realizations` times; `phantom.n_draws` is a record only. One realization of Block 13's chain took 5-132 s, mean 48 s, in the sandbox.]

Merge with the **same** configuration [added 2026-10-07] [2026-10-21, Block 14.6: in a job or an interactive session, not on the login node -- at $10^6$ rows the fit takes of order 20 min and a few hundred MB; the merge also refuses a (draw, realization) seen twice, notes the draws with fewer than `n_realizations` rows and writes `draws_<hash>.csv`, the per-draw table, beside the table]. Without `--config-json` the merge assumes the default configuration. When the estimator hashes agree, the row files are then found, and before 2026-10-07 the default ranges and signature would have been written into the table. Since 2026-10-07 every row records the configuration it was rendered under (`sim_hash`), and the merge stops with `merge refused` when it is given another one (SPEC Block 6):

```
python scripts/build_table.py merge --out-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_tables/v1" --config-json $HOME/diam/config_production.json
```

Copy `bias_table_<hash>.npz` and `bias_table_<hash>.json` [2026-10-21: and `draws_<hash>.csv`] to `MyDrive/diameters/tables/`. ~~For the σ_fit study, repeat with `RUN_TAG=v1_s0080` and `config_sigma0.080.json`, then `RUN_TAG=v1_s0125` and `config_sigma0.125.json`.~~ [corrected 2026-10-21, S2.2: the study is dropped, D-057 (ii); one table.]

## Cell 10: apply to the cell

```python
import importlib, json
import allen_diameter.config as C
C = importlib.reload(C)
CFG = OUT + "/config_production.json"
with open(CFG) as f:
    H = C.config_from_dict(json.load(f)).signature_hash("estimator")
TABLE = "%s/tables/bias_table_%s" % (OUT, H)
missing = [TABLE + e for e in (".npz", ".json") if not os.path.exists(TABLE + e)]
if missing:
    raise FileNotFoundError("copy these from davinci first (Cell 9, step 5): %s" % missing)
run("scripts/run_cell.py", "--specimen", SPECIMEN, "--table", TABLE, "--out-dir", OUT + "/cell",
    "--cache-dir", CACHE_DIR, "--config-json", CFG, "--registration-json", REG_JSON, "--nodes", NODES, *ALIGN)
```

Outputs:
- `nodes_<specimen>.csv`;
- `specimen_<specimen>/reconstruction.swc`, with radius = d_final / 2 on dendrite nodes and every other byte unchanged (D-013);
- `summary_<specimen>.json`, with the membrane-area ratio against Allen's radii.

Run it on a few stretches first, with `--nodes` (above); the notebook's Cell 10b then runs every dendrite node (the same command without `--nodes`), ~~and Cell 10c runs each σ_fit study table that is on Drive into `cell_sigma<value>/`. No script yet joins the three runs into the `d_tilde_sigma_spread_um` column (SPEC §8): compare the three `nodes_<specimen>.csv` by `node_id`.~~ [corrected 2026-10-21, S2.2: Cell 10c and the column are removed with the σ_fit study, D-057 (ii).]

## S3 on davinci: the survey over the selected cells (Block 16.4)

[added 2026-10-28, v34; D-099. Colab's Cells 11-15 below run on the pilot only since the same decision; the tables under each cell describe the steps, which the jobs run unchanged over the 51 selected cells.]

Three PBS jobs in `scripts/pbs/`, submitted from the workstream directory of the cluster's clone, share `survey_common.sh`. Every knob is a `qsub -v NAME=value,...` variable with a default; a set `CODE`, `ROOT` or `ENV_NAME` (the login shell's) is reported and ignored. The defaults put the survey under the archive root of D-012: `/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_survey/<RUN_TAG>/` (`DIAM_SURVEY`) and the crops under `.../Human Neurons Fitting/allen_cache/<specimen>/` (`DIAM_CACHE`; one folder per cell, so the 51 tasks never write the same file) -- the layout is the assistant's, as the table job's, PROVISIONAL.

| Job | `-N` | What | Walltime | Needs the network |
|---|---|---|---|---|
| `survey_prepare.pbs` | `diam_survey_prep` | Cell 11 over the whole pool (`pool`, `check`, `select --require-match`) into `<DIAM_SURVEY>/`, then Cell 11b (`projection_check.py --check <DIAM_PILOT>`, the pilot 529878215 by default, `none` lists only) into `<DIAM_SURVEY>/projections/`; ends with `selection: N cells` and the `qsub -J 0-<N-1>` line to submit next | 2 h (Colab: 0.1 + 6.6 + 0.0 min and 7.0 min) | yes |
| `survey_cells.pbs` | `diam_survey`, array `-J 0-50` | task $i$ runs row $i$ of `selection.csv` (0-based; `--task-index`) through the steps `DIAM_STEPS` (default `align,nodes,radius,registration,camera,calibration`: Cells 12, 13, 14 and 15's per-cell parts) with its own crop cache (`--cache-per-cell`) [2026-10-29, v38: the calibration step first searches the cell's candidates over its whole trace, both sets (D-112; the pilot's 20.5 min in the sandbox)]; resumable: a step with a marker in `<survey>/<cell>/done/` is skipped, so a killed or re-submitted array finishes what is left | 12 h per task, 1 CPU, 4 GB (Colab: about an hour per cell of 120 nodes) | yes |
| `survey_pool.pbs` | `diam_survey_pool` | `status`, then the pooled analyses (`DIAM_POOL`, default `coverage,inputs,calibration`: Cells 13, 14 and 15's pools; `DIAM_PHANTOM_ROWS` for the phantoms' band), then the archive `<DIAM_SURVEY>/results_<RUN_TAG>.tgz` of `status.csv`, `s3a/`, `s3b/`, `s3c/` and every cell's `align/` and `nodes/` files -- what is read off the cluster | 2 h, 8 GB | no |

Network: the compute nodes reach the outside only through `module load proxy` (`DIAM_PROXY_MODULE`; the H01 reference, section 9 of the paths reference), after which the job runs one RMA query to `api.brain-map.org` (specimen 529878215) before any work and exits 6 if it fails -- a blocked network costs a minute, not a queue wait and a walltime. The first probe job settles whether the proxy reaches Allen's server **[open until run]**.

Exit codes: 2 no scripts under `DIAM_CODE` (submit from the workstream directory); 3 the selection or `DIAM_CONFIG_JSON` missing; 4 a bad `RUN_TAG`, `DIAM_CELLS`, `DIAM_POOL`, `DIAM_PILOT`, a task beyond the selection, or `-J` and `DIAM_N_TASKS` disagreeing; 5 the conda env not active (`python3` not under `envs/spine_env/bin`); 6 `api.brain-map.org` unreachable; 1 the runner or an analysis failed (the pool job runs the other analyses and still writes the archive). The logs land in `$HOME` (`#PBS -k eo`): `diam_survey.o<jobid>.<task>` per task.

The roadmap: the array marks stage S3a and the pool job S3c of `specs/roadmap/diameter-pipeline.yaml` (running, then done or failed) through `specs/roadmap/hpc_status.sh`, which pushes the marker to the branch `hpc-status`; the cluster must be able to push to origin once -- check with `git push --dry-run origin HEAD:refs/heads/hpc-status-probe` from the clone; `ROADMAP_DISABLE=1` in `-v` turns the marker off for one submission, and a failing marker never fails the job.

Procedure, in this order (hpc-git-delivery: verify, dry run, one probe, the array, the pool):

1. Update the clone and verify the bytes on the login node (no allocation needed):

```
cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement" && git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only && git log --oneline -1
```

```
python3 - <<'EOF'
import os, sys
bad = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
    for f in files:
        p = os.path.join(root, f)
        if f.endswith((".sh", ".pbs")) and open(p, "rb").read().count(b"\r"):
            bad.append(("CRLF", p))
        if f.endswith(".py") and any(c > 127 for c in open(p, "rb").read()):
            bad.append(("NON-ASCII", p))
print(bad or "OK: every .sh/.pbs is LF-only and every .py pure ASCII")
sys.exit(1 if bad else 0)
EOF
```

```
bash -n scripts/pbs/survey_common.sh scripts/pbs/survey_prepare.pbs scripts/pbs/survey_cells.pbs scripts/pbs/survey_pool.pbs && echo "OK: bash -n" && python3 -m py_compile scripts/survey_cells.py scripts/cell_pool.py scripts/projection_check.py && echo "OK: compiles"
```

A `CRLF` here means the file was rewritten in transit: `sed -i 's/\r$//' <file>` and check again.

2. Dry runs (they print every command and activate nothing; seconds in the queue):

```
qsub -v DIAM_DRYRUN=1,RUN_TAG=v1 scripts/pbs/survey_prepare.pbs
```

Expect in `$HOME/diam_survey_prep.o<jobid>`: the header lines `[diam-survey-prep] code/survey/cache/env/config/pilot` and four `[diam-survey-prep] DRYRUN: python3 scripts/cell_pool.py pool ...`, `... check ...`, `... select ... --require-match ...`, `... projection_check.py ... --check 529878215 ...` lines.

3. The prepare job:

```
qsub -v RUN_TAG=v1 scripts/pbs/survey_prepare.pbs
```

Expect `[diam-survey-prep] python /davinci-1/.../envs/spine_env/bin/python3`, `... module proxy loaded (http_proxy http://...)`, **`[diam-survey-prep] api.brain-map.org reachable: specimen 529878215`** (the line that must appear; without it the job exits 6 and names the proxy), then Cell 11's `pool:`, `check:` and `select:` lines and Cell 11b's `[proj]` lines as the runbook gives them, `[diam-survey-prep] selection: 51 cells in .../selection.csv` and `[diam-survey-prep] next: qsub -J 0-50 -v RUN_TAG=v1 scripts/pbs/survey_cells.pbs`. Resumable: a second submission skips the cells already in `acquisition.csv`.

4. One probe task, the alignment of one cell:

```
qsub -v RUN_TAG=v1,DIAM_STEPS=align scripts/pbs/survey_cells.pbs
```

(No `-J`: a single job is task 0, row 0 of the selection; PBS Pro may refuse a one-element range such as `-J 0-0`, so the probe does not use one.) Expect in `$HOME/diam_survey.o<jobid>`: `[diam-survey] task 0 of 51 (row 0 of 51)`, the header lines, the python and versions, `[diam-survey] commit <hash>` (the branch head), `[diam-survey] module proxy loaded`, **`[diam-survey] api.brain-map.org reachable: specimen 529878215`**, then the runner's `survey: 1 cells, steps align -> ...`, `[1/1] <id> align: shift (x, y) px, ..., ratio R` and **`[diam-survey] task 0 of 51 (row 0 of 51) done`**. Read the whole `.o` file before going on.

5. The array, with the `-J` the prepare job printed (the default steps; `DIAM_CONFIG_JSON=<path without spaces>` for another node mode or quota, or for the stripe's knobs [added 2026-10-28, v36]: copy Cell 7's `calibration_config.json` from Drive to the cluster -- e.g. `$HOME/diam/calibration_config.json` -- and pass it as `DIAM_CONFIG_JSON` to the array **and** to the pool job; to change the node mode or quota as well, one JSON with both sections, `{"survey": {...}, "calibration": {...}}`; a cell whose calibration step is done runs it again only with `DIAM_FORCE=calibration`):

```
qsub -J 0-50 -v RUN_TAG=v1 scripts/pbs/survey_cells.pbs
```

Each task ends with its `... done` line; a task of a set-aside cell prints the runner's `skipped:set_aside` for the later steps and still ends `done`. A failed task (exit 1) is re-run alone by its cell, `qsub -v RUN_TAG=v1,DIAM_CELLS=<id> scripts/pbs/survey_cells.pbs` (no `-J`; the same steps, resumed), or the whole array is re-submitted: it only runs what is left. The run file's Cells 12b and 13b are knobs here: `qsub -v RUN_TAG=v1,DIAM_CELLS=508305938,DIAM_FORCE=align scripts/pbs/survey_cells.pbs` and `qsub -v RUN_TAG=v1,DIAM_CELLS=576084692,596887423,611598198,DIAM_INCLUDE_SET_ASIDE=1 scripts/pbs/survey_cells.pbs` (no `-J`: one job on the named cells; a cell aligned by hand gets its `alignment_manual.json` under `<DIAM_SURVEY>/<id>/align/` first, as Cell 12 describes).

6. The pool, when `qstat -t` shows no task of the array left:

```
qsub -v RUN_TAG=v1 scripts/pbs/survey_pool.pbs
```

Expect `[diam-survey-pool] pool coverage`, Cell 13's `coverage:` lines, `pool inputs` and Cell 14's `inputs:` lines, `pool calibration` and Cell 15's pooled lines, then **`[diam-survey-pool] archive .../results_v1.tgz (<size>)`** and `[diam-survey-pool] done`. A `pool <what> FAILED (exit n)` line means that analysis failed; the others ran, the archive is written, the job exits 1. `DIAM_PHANTOM_ROWS="<csv> <csv>"` passes the davinci table's replicate CSVs for the phantoms' own band (paths without spaces).

7. Read the archive off the cluster (MobaXterm's file browser or `scp`, binary mode) and unpack it beside `MyDrive/diameters/survey/` or anywhere else: it holds what Colab's Cells 12-15 printed and wrote, and Cell 8's decisions are made on it. The crops (`allen_cache/<specimen>/`) and the calibration scans stay on the cluster.

What is not settled by the sandbox and the first run settles: the proxy's reach to Allen's server from a compute node, the time per cell and per task on davinci (the 12 h walltime holds Colab's hour per cell with margin), whether the shared archive root tolerates 51 tasks fetching at once (each writes its own folder; Allen's server is the shared resource).

## Cell 11: S3 across cells, step 0 -- the cell pool, the acquisition check, the selection (Block 16.1)

[added 2026-10-26, v23; D-065, D-085. Needs Cells 0 and 1 only.]

[2026-10-28, v34; D-099: in Colab this cell runs on the pilot 529878215 only, in `survey_pilot/` -- the code below; the steps described after it run unchanged over the selected cells in the davinci jobs (the section above). The notebook's cell of 2026-10-26 ran the whole selection; its last run is `survey/` of 2026-10-28.]

```python
SURVEY = OUT + "/survey_pilot"     # the pilot's own folder (D-099); survey/ keeps the run of 2026-10-28 over 51 cells
os.makedirs(SURVEY, exist_ok=True)
run("scripts/cell_pool.py", "pool", "--out", SURVEY + "/pool.csv")
run("scripts/cell_pool.py", "pick", "--pool", SURVEY + "/pool.csv", "--cells", SPECIMEN, "--out", SURVEY + "/selection.csv")
run("scripts/cell_pool.py", "check", "--pool", SURVEY + "/selection.csv", "--out", SURVEY + "/acquisition.csv")
```

S3 runs the real-data surveys (S3a-S3d) over many Allen cells instead of 529878215 alone. This cell picks the cells; nothing here measures a node.

| Step | What it does | Writes |
|---|---|---|
| `pool` | every row of Allen's `ApiCellTypesSpecimenDetail`, kept when the species is Homo Sapiens and a reconstruction exists (the Allen SDK's own filter, `CellTypesApi.filter_cells_api`); one record per specimen | `survey/pool.csv` |
| `check` | per pool cell: the plane spacing (`NeuronReconstruction.scale_factor_z`), the pixel size (the Primary planes' `resolution`, µm/px at downsample 0), the usable and failed planes; compared with `acquisition.dz_um` = 0.28 µm and `res0_um` = 0.1144 µm/px at a relative tolerance of 1e-3 (PROVISIONAL) | `survey/acquisition.csv`, `survey/acquisition.json` |
| `pick` [v34] | the selection records of the named cells (`--cells 529878215`), in that order, each with its class and group `class|layer` (Block 16.1 (e)); Colab's pilot-only Cell 11 uses it, `check` then runs on that one-row file (`check: 1 cells, 0 failed; ...; 1 match the configuration`) | `survey_pilot/selection.csv`, `survey_pilot/selection.json` |
| `select` | per (class, layer): the cells in increasing specimen id, permuted by `default_rng([20261026, crc32("class|layer")])`, the first 5, or all of a smaller group (D-085.1); with `--require-match`, only the cells whose plane spacing and pixel size match | `survey/selection.csv`, `survey/selection.json` |

Class (D-085.1, D-087): spiny = `exc`, aspiny = `inh` (the mapping PROVISIONAL), sparsely spiny = `sparsely_spiny`, a third class [user, D-087; the name confirmed by the user, D-089]. Cells without a dendrite type have no class: they are listed by `pool` and never selected. Layer: Allen's label as given. [amended 2026-10-26, later, v24: sparsely spiny cells had no class in v23]

Expect from `pool`: `pool: N Homo Sapiens cells with a reconstruction (M rows read, api:model::ApiCellTypesSpecimenDetail) -> .../survey/pool.csv`, then `by class and layer: exc|2 n, ...` and `dendrite types without a class (left out of the selection): {...}`. A `ValueError` naming a missing field prints the keys Allen's rows hold: send them back (the field names come from the SDK's source, not from the live API). A `RuntimeWarning` ending `raise survey.pool_page_rows above ...` means the query needed more than one page, and pages without an order may skip rows: set `survey.pool_page_rows` above the row count in a `--config-json` written by hand -- `{"survey": {"pool_page_rows": 50000}}` (the other sections keep their defaults; `to_json` does not write the survey section) -- and run again.

Expect from `check`: `check: N cells in the pool, 0 already checked, N to check`, one line per cell (`<id>: dz <value> um, pixel <value> um, <n> planes`, with `; ERROR ...` when a query failed), then `check: N cells, k failed; dz (um): {...}; pixel (um): {...}; m match the configuration (dz 0.2800 um, pixel 0.1144 um, relative tolerance 0.001) -> ...`. It writes the table every 10 cells and skips the cells already in it, so after a disconnection run the cell again (`--fresh` checks every cell again). The time per cell against Allen's server is not measured: about four queries per cell.

Expect from `select`: `select: G groups, S cells selected (seed 20261026, 5 per group); left out {...} -> ...`, then one line per group, `exc|3   5 of  n: <ids>`. `selection.json` holds the seed, the rule, each group's counts and every cell left out with its reason (`no_class`, `no_layer`, `acquisition_missing`, `acquisition_mismatch:dz`, `:pixel`, `:dz;pixel`, `:error`, `:no_planes`).

Decisions it feeds: whether the cells whose plane spacing or pixel size differ stay out (D-085.3: `REQUIRE_MATCH`, read `acquisition.json` first; `True` is the cell's default, not a decision); the selected cells are the input of S3a-S3d (Blocks 16.2-20). Send back the `pool:` lines, the `check:` summary line with `acquisition.json`, and the `select:` lines.

## Cell 11b: what Allen's projections are (Block 16.3)

[added 2026-10-27, v29; D-090.1. Needs Cells 0 and 1, and Cell 11's `selection.csv` to list every selected cell (without it, the pilot only).]

[2026-10-28, v34; D-099: in Colab this cell runs on the pilot 529878215 only, in `survey_pilot/` -- the code below; the steps described after it run unchanged over the selected cells in the davinci jobs (the section above). The notebook's cell of 2026-10-26 ran the whole selection; its last run is `survey/` of 2026-10-28.]

```python
SURVEY = OUT + "/survey_pilot"
SEL = SURVEY + "/selection.csv"
run("scripts/projection_check.py", *(["--selection", SEL] if os.path.exists(SEL) else ["--specimens", SPECIMEN]),
    "--check", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", SURVEY + "/projections")
```

Why: Cell 12 takes, as each cell's minimum-intensity projection, the darkest of the SubImages that are not Primary planes and have the planes' size (Block 16.2 (a)). Neither the `image_type` strings Allen gives them nor what each projection computes was ever checked: the design chat recorded, from live API queries, four projections for 529878215 -- named there min-xy (546843114, 7607 × 9435 px), min-yz (546843119, 1333 × 9435), max-xy (550303812) and max-yz (550303817) -- and planes of 7605 × 9433 px (`docs/handoff_diameter_remeasurement.md`), not their types. The sandbox cannot reach Allen, so this cell settles it.

| Part | What | Who chose it |
|---|---|---|
| the listing | per cell: every non-Primary SubImage with every field, the planes' count and size, the xy candidates and the one the darkest rule takes (each fetched whole at level 4) | D-090.1 (the user: a Colab function and cell) |
| the test | on a 512 × 512 px window (59 µm) around the pilot's soma: each xy projection against the planes' pixelwise minimum, mean, median and maximum over integer offsets up to ±3 px -- the correlation r*, the line and its residual | the method the assistant's |
| the verdict | `matches <op>` when r* ≥ 0.9 and the next operator's line leaves at least 4 times the best's unexplained variance (twice its residual RMS): two operators' images can correlate at 0.95-0.97 while the wrong one's line leaves 4 times the residual, so the lead in r does not decide | PROVISIONAL |

Expect per cell `[proj] <id>: N non-Primary SubImages: <id> '<image_type>' WxH; ...` and `[proj] <id>: planes n usable (k failed), WxH px; xy candidates [...]; darkest at level 4: <id>`; for the pilot, one line per xy projection, `[proj] 529878215: <id> '<type>': matches min -- r* min 0.99.. at (+a, +b); mean 0.9..; median ...; max ...`, and `[proj] S specimens listed, 1 checked -> .../survey/projections`. The pilot's test downloads one 512 × 512 px crop per plane (about 520; minutes) and one per xy projection, all into `CACHE_DIR`. Files in `survey/projections/`: `projections_<id>.json` per cell (the rows verbatim; for the pilot the window, every operator's r*, offset, line and RMS, and the verdict), `projection_types.csv` (one row per non-Primary SubImage of every cell listed), `projection_check_529878215.png`.

What to read: (1) `projection_types.csv` -- whether every selected cell has the same projection types and sizes, and which one the darkest rule takes; (2) the pilot's verdicts: the darkest projection should `match min`; its offset (a, b) is the projection's pixel minus the planes' pixel holding the same content (Allen's projections are 2 px larger than its planes), so an offset other than (0, 0) means Cell 12's shifts, found on the projection, are that many pixels off on the planes (0.11 µm per pixel); (3) the figure: each projection beside the planes' minimum, mean, median and maximum, on one grey scale per row.

Decisions it feeds: whether Cell 12's rule stands, and what to do with a non-zero offset -- yours, on the report (D-090.1). Send back `survey/projections/projection_types.csv`, `projections_529878215.json` and `projection_check_529878215.png`.

## Cell 12: S3 across cells, step 1 -- every selected cell's global alignment (Block 16.2)

[added 2026-10-26, v25; D-085.2, D-085.12. Needs Cells 0, 1 and 11.]

[2026-10-28, v34; D-099: in Colab this cell runs on the pilot 529878215 only, in `survey_pilot/` -- the code below; the steps described after it run unchanged over the selected cells in the davinci jobs (the section above). The notebook's cell of 2026-10-26 ran the whole selection; its last run is `survey/` of 2026-10-28.]

```python
SURVEY = OUT + "/survey_pilot"
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align")
run("scripts/survey_cells.py", "status", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv")
```

(The notebook's cell then shows the overlay of every set-aside cell.)

What it does, per selected cell (`survey/<specimen>/align/`):

| Stage | What | Who chose it |
|---|---|---|
| projection | [corrected 2026-10-28, v33, D-098.2: the SubImage of type 'MinimumIntensityProjection - xy' among the xy candidates; the darkest only when none has that type -- on Allen's data the 'MaximumIntensityProjection - xy' is an inverted image and the darker, which misaligned 508305938] Allen's xy projections (SubImages other than Primary with the planes' size within 2 %), each fetched whole at level 4; the darkest (lowest median grey) is the minimum-intensity projection | the rule the assistant's, confirmed by the user, D-089; what the projections are: Cell 11b |
| coarse | `align_translation` on it: FFT cross-correlation of the darkness map with the skeleton (SWC types 3, 4), both y-flips, a ±250 µm window, ±2 px refinement; 1.83 µm/px | the user (D-085.2) |
| score ratio | the same search's best score, against its best score on 50 copies of the image with its 30 µm tiles shuffled: R = observed / null median | PROVISIONAL |
| fine | the skeleton's box plus 20 µm at level 1 (0.23 µm/px; level 2 if too large), a direct search over ±5 µm on the ridge darkness | the assistant's addition, PROVISIONAL |
| verdict | set aside when R < 2, or a search ends on its window's edge | the threshold the assistant's, confirmed by the user, D-089 |

Writes `alignment_<id>.json` (the shifts in full-res px, the flip height, `z0_um` = 0 as for 529878215, every stage's numbers, the projection chosen and its candidates, `set_aside` and the reasons) and `overlay_<id>.png` (the projection with the skeleton; the fine crop with the coarse skeleton in grey and the final in orange). Each finished step leaves a marker in `survey/<id>/done/`; a failure is written to `survey/<id>/errors.log` and the next cell starts. `survey/status.csv` holds every cell's state.

Expect `survey: S cells, steps align -> .../survey`, one line per cell, `[i/S] <id> align: shift (x, y) px, no flip, ratio R, auto (t s)` or with `; SET ASIDE (low_score_ratio)`, or `FAILED align: <error>`, then `survey: align done n, ... -> .../status.csv`. `status` prints one line per cell and `status: S cells -> .../status.csv`. Resumable: run the cell again after a disconnection; an aligned cell is not fetched again.

A set-aside cell: read its overlay. If the skeleton sits on the cell after all, or you can see the shift, write `survey/<id>/align/alignment_manual.json`:

```json
{"shift_x_px": 0.0, "shift_y_px": 0.0, "flip_h_px": null, "z0_um": 0.0}
```

(full-res pixels; `flip_h_px` the full-res height when y is flipped) and run `survey_cells.py run ... --cells <id> --force align`: the manual alignment is used as given and the cell is no longer set aside. `--force` also clears the markers of the steps after the forced one. Later steps skip set-aside cells unless `--include-set-aside`.

The synthetic check (sandbox): on painted cells the final shift came within 2.5 full-res px (0.29 µm) of the truth and R was about 7; an empty field with dark specks gave R = 1.08 and was set aside. On Allen's cells R is not known yet: the threshold 2 is to be read again on the first run's `status.csv` and overlays.

Decisions it feeds: which set-aside cells to fix or drop (the threshold, 2, confirmed by the user, D-089). Send back the `survey:` summary line, `status.csv` and the overlays of the set-aside cells.

## Cell 13: S3a -- the node survey across cells and the coverage of the phantoms' band (Block 17)

[added 2026-10-26, v26; D-059.1, D-065.1, D-085.4-6, D-087. Needs Cells 0, 1, 11 and 12.]

[2026-10-28, v34; D-099: in Colab this cell runs on the pilot 529878215 only, in `survey_pilot/` -- the code below; the steps described after it run unchanged over the selected cells in the davinci jobs (the section above). The notebook's cell of 2026-10-26 ran the whole selection; its last run is `survey/` of 2026-10-28.]

```python
import json
SURVEY = OUT + "/survey_pilot"
NODES_MODE = "stratified"      # or "all" (D-085.4)
NODES_PER_BIN = 3              # the Colab pilot's size (D-099, PROVISIONAL): <= 3 x 6 bins = 18 nodes; davinci keeps 20
PHANTOM_ROWS = []              # e.g. [OUT + "/tables/rows_<hash>_0_1000.csv"]
CFG_S3 = SURVEY + "/survey_config.json"
with open(CFG_S3, "w") as f:
    json.dump({"survey": {"nodes_mode": NODES_MODE, "nodes_per_bin": NODES_PER_BIN}}, f)
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes", "--config-json", CFG_S3)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "coverage", "--config-json", CFG_S3, *(["--phantom-rows"] + PHANTOM_ROWS if PHANTOM_ROWS else []))
```

(The notebook's cell then shows the four figures.) [v34] `NODES_PER_BIN = 3` is the Colab pilot's size (PROVISIONAL; at most 18 nodes over the six bins of Allen's 2r, about 10 min at the 30 s per node of 2026-10-28): every coverage bin is then `too_few`, and the cell checks the chain and the pool end to end; the davinci array keeps the default 20 per bin, and its `s3a/` is what decides the mu range.

| Part | What | Who chose it |
|---|---|---|
| sample, `stratified` | per bin of Allen's 2r -- [0, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.2), [1.2, 2), [2, ∞) µm -- up to 20 nodes, taken in turn from the cell's stretches (each stretch's first node, then each one's second, ...), in seeded orders | the user (D-085.4); the round-robin PROVISIONAL; the default mode confirmed by the user, D-089 |
| sample, `all` | every dendrite node: thousands per cell, each with 7-23 crops | the user (D-085.4) |
| measurement | Block 13's chain at each node (the pilot, the Allen-guided iteration, the SGT fit at its plane), the cell's alignment, no table; the background statistics for S3b's camera | D-065.1 |
| coverage | per bin of d_hat (<0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.2, 1.2-2, 2-4, ≥4 µm), the nodes in S: the share with mu_hat in the phantoms' range [0.3, 3] /µm -- the analytic band ln(mu d) of D-065.1 at the node's own d_hat -- and the shares below and above; the quantiles of ln(mu_hat d_hat) and ln alpha_hat; with `PHANTOM_ROWS`, the phantoms' own fitted band (their 0.5-99.5 % in the bin) | the band D-065.1; the bins PROVISIONAL |
| pass rule | every bin with ≥ 20 nodes has ≥ 95 % inside; fewer than 20: `too_few`, not failing | the user (D-085.5) |
| diameters | d_hat (in S; fitted, not corrected) and Allen's 2r, per (class, layer), per class, per layer | the user (D-085.5, D-087) |

No thick-node check (D-085.6: the renderer is not the final one).

Expect, per cell, `[i/S] <id> nodes: stratified: n sampled, n measured, k in S (t s)` (a set-aside cell: `skipped:set_aside` in `status.csv`), then `coverage: C cells, N nodes, M in S and usable`, one line per bin (`d_hat 0.4-0.6  n ... inside 0.97 ... pass`) and `coverage: PASS -- every bin of d_hat with >= 20 nodes in S has >= 95 % of them inside the band (D-085.5) -> .../survey/s3a` or `coverage: FAIL in <bins> -- ...`. Files in `survey/<id>/nodes/` (`pilot_<id>.csv`, `pilot_summary_<id>.json`) and `survey/s3a/` (`coverage.csv`, `coverage.json`, `diameters.csv`, `coverage_mud.png`, `coverage_share.png`, `coverage_alpha.png`, `diameters.png`).

[added 2026-10-28, v33; D-098.3] The pool reads S from each row's `fit_status`, `reg_verdict` and `flags` under the current rule, not from the stored `in_S`, and logs `coverage: S recomputed ... k now in S and j now out of S against their stored in_S`: a run whose CSVs were written before D-096 (`estimator_hash` `0508e7ff12f48728` in `pilot_summary_<id>.json`) counts its faint nodes in S without measuring again. Time: at least 13 plane crops and Block 13's fits per node, so about an hour per cell of 120 nodes and one to two days for the 42 cells of 2026-10-28 (an estimate; each cell's line gives its time). After a disconnection run Cells 0, 1 and 13 again: the finished cells are skipped, the interrupted one restarts with its crops in the cache. The run file's Cell 13b runs the set-aside cells you accept (`--include-set-aside --cells ...`, D-098.5).

Switching `NODES_MODE` after a run: add `"--force", "nodes"` to the `run` call (the marker does not record the mode). A failing bin is reported, not acted on: widening the mu range is your decision (D-065.1).

Decisions it feeds: the mu range (D-065.1), with the figures and the failing bins. Send back the coverage lines, `survey/s3a/coverage.json`, `diameters.csv` and the four figures.

## Cell 14: S3b -- the simulator's inputs across cells: d_max, the node jitter, the camera (Block 18)

[added 2026-10-26, v27; D-065.2-3, D-085.8-9, D-086. Needs Cells 0, 1, 11-13.]

[2026-10-28, v34; D-099: in Colab this cell runs on the pilot 529878215 only, in `survey_pilot/` -- the code below; the steps described after it run unchanged over the selected cells in the davinci jobs (the section above). The notebook's cell of 2026-10-26 ran the whole selection; its last run is `survey/` of 2026-10-28.]

```python
import json
SURVEY = OUT + "/survey_pilot"
CFG_S3 = SURVEY + "/survey_config.json"     # Cell 13's (node mode, the pilot's nodes per bin); written if Cell 13 did not run
if not os.path.exists(CFG_S3):
    with open(CFG_S3, "w") as f:
        json.dump({"survey": {"nodes_mode": "stratified", "nodes_per_bin": 3}}, f)
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes,radius,registration,camera", "--config-json", CFG_S3)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "inputs", "--config-json", CFG_S3)
```

(The notebook's cell then prints the suggestions and shows `inputs.png`.)

| Input | Per cell | Pooled suggestion | Who chose the rule |
|---|---|---|---|
| `phantom.d_range_um`'s upper end, d_max | `radius`: Allen's dendrite 2r (SWC types 3, 4), its maximum and the node | the largest over the cells, with the cell and node; p99 and p99.5 beside it as comparisons | the user (D-086) |
| `phantom.jitter_xy_um`, `jitter_z_um` | `registration`: the registration survey (Cell 3b's check) on list_stretches' suggestion (3 stretches of ≥ 10 nodes per type): s* and dz* per node | 1.4826 × MAD of the ON nodes' s* and dz*, pooled across the cells; per cell the same and the medians | the user (D-085.8) |
| the camera fields | `camera`: Cell 5's computation on the cell's nodes' background (from `nodes`) and the cache's JPEG tables | the most common table set; the medians of B and of the noise SD over the cells (the noise SD an upper bound) | the user (D-085.9) |

Expect per cell `radius: n nodes, max 2r ... um (node ...), p99 ... um`, `registration: k nodes, verdicts {...}, robust SD xy ..., z ...`, `camera: B ... gl, noise SD ... gl, tables yes`; then `inputs: d_max ... um (cell ..., node ...; p99 ..., p99.5 ... um); jitter xy ... um, z ... um[ (z0 flagged: [...])]; background ... gl, noise SD ... gl, n table set(s), the most common in m cells` and `inputs: suggestions only (Cell 8 decides) -> .../survey/s3b`. Files in `survey/<id>/radius/`, `registration/`, `camera/`, and `survey/s3b/` (`inputs.json` with `suggested`, `inputs_per_cell.csv`, `inputs.png`).

A cell flagged `z0` has |median dz*| above 1 µm (the assistant's, confirmed by the user, D-089): the plane origin `z0_um` = 0 of its alignment (Cell 12) may be off; its overlay and registration figures say more. One thick node -- a trunk near the soma, or a tracing error -- sets d_max for every cell (D-086): read the cell and node it names before you write it into `DECIDED`.

Decisions it feeds: Cell 8's `phantom.d_range_um`, `jitter_xy_um`, `jitter_z_um`, `renderer.background_B_gl`, `noise_sd_gl`, `jpeg_qtables` -- each a value you write, never taken automatically (D-065). Send back the `inputs:` lines, `survey/s3b/inputs.json`, `inputs_per_cell.csv` and `inputs.png`.

## Cell 15: S3c -- one defocus kernel calibrated on nodes pooled across cells (Block 19)

[added 2026-10-26, v28; D-088 (supersedes D-085.10). Needs Cells 0, 1, 11-13.]

[2026-10-28, v34; D-099: in Colab this cell runs on the pilot 529878215 only, in `survey_pilot/` -- the code below; the steps described after it run unchanged over the selected cells in the davinci jobs (the section above). The notebook's cell of 2026-10-26 ran the whole selection; its last run is `survey/` of 2026-10-28.]

```python
import json
SURVEY = OUT + "/survey_pilot"
CFG_S3 = SURVEY + "/survey_config.json"     # Cell 13's (node mode, the pilot's nodes per bin); written if Cell 13 did not run
if not os.path.exists(CFG_S3):
    with open(CFG_S3, "w") as f:
        json.dump({"survey": {"nodes_mode": "stratified", "nodes_per_bin": 3}}, f)
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes,calibration", "--config-json", CFG_S3)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "calibration", "--config-json", CFG_S3)
```

For now the calibration is tested on the pilot cell with Cell 7 (D-088.2); this cell is for when the survey has run.

| Part | What | Who chose it |
|---|---|---|
| one kernel | the near table to 0.84 µm and the far slope, fitted once on nodes of several cells; one production configuration, one table | the user (D-088.1) |
| per cell (`calibration`) | the candidates of the cell's node sample (Cell 13's CSV: in S, flat, Allen's 2r in a set; no optical-depth cut since D-094, and `faint` no criterion of S since D-096), scanned in a seeded order, at most 5 per set; `survey/<id>/calibration/scans_<id>.json` | the order PROVISIONAL; the per-cell quota the assistant's, confirmed by the user, D-089 |
| the pool | per set, the cells in increasing specimen id, each cell's first scan, then each one's second, ... up to 30; node ids qualified as specimen × 10⁶ + node (two cells can share a node id) | the cap the user's (D-088.3); the round-robin PROVISIONAL |
| sufficiency | at least 10 nodes per set; below, the fits run as diagnostics and no production kernel is written | the minimum the user's (D-088.3); the rule below it PROVISIONAL |
| by cell (v29) | with two cells or more: every plane's kernel SD of both sets against \|δ\| with its cell, its residual from the pooled law (the near table to 0.84 µm, then the far line), and per cell and set the median and quartiles of the residuals over the planes the fits used | the user (D-090.2: near and far together, colour by cell); the residual and the layout the assistant's: up to three cells one panel coloured by cell (the most a scatter keeps apart for colour-blind readers), beyond three one panel per cell |

Expect per cell `[i/S] <id> calibration: n candidates: a near-field, b far-field scans; reasons {...}`; then `pooled near field: k of n scans taken (at most 30) from c cells {...} -- sufficient` (or `INSUFFICIENT (< 10, D-088.3)`), the same for the far field, Cell 7's `fit` and `far-fit` report on the pooled scans, `wrote ...` for the figures, and `pool calibration -> .../survey/s3c`. Files in `survey/s3c/`: `scans_pooled.json` (its `meta` holds the counts per set and cell, the sufficiency and the id map), `calibration_pooled.json`, `far_fit_pooled.json`, `figures/`. [v29] With two cells or more, also `by cell: specimen <id> near field: n nodes, m planes used; median residual +x um (IQR +a to +b um)` per cell and set, `kernel_law_by_cell.csv` (those rows), `figures/kernel_law_by_cell.png` and `figures/kernel_law_residuals_by_cell.png`.

What to read: as Cell 7 -- the width-by-plane figures first, then the far fit -- and the counts per cell: a kernel resting mostly on one cell is a pooled kernel in name only. [v29] Then the by-cell figures (D-090.2): a cell whose median residual sits away from zero, in either set, is a cell whose blur differs from the pool's; the near residuals also carry, for every cell alike, the difference between the near fit's growth and the table's linear interpolation between knots 0.28 µm apart (up to 0.075 µm on a synthetic growth), so compare cells with each other rather than with zero there. Too few nodes in a set: raise `calib_scans_per_cell` (in `CFG_S3`: `{"survey": {..., "calib_scans_per_cell": 10}}`) and run again with `--force calibration`.

Decisions it feeds: Cell 8's `renderer.kernel_table_*` and `kernel_continuation_slope` -- yours to write once the pooled sets are sufficient; whether a cell that stands out is kept in the pool. Send back the pooled lines, the fit and far-fit lines, the `by cell:` lines and the figures.

[added 2026-10-28, v36] Cell 15 merges Cell 7's knobs: when `OUT/calibration/calibration_config.json` exists, its `calibration` section joins Cell 13's settings in `survey_pilot/survey_config_s3c.json`, which the cell prints (`[cell 15] calibration knobs: ...`) and passes to `run` and `pool`. A cell's calibration step runs once: after a change of the knobs set `REDO = ["--force", "calibration"]`.

[added 2026-10-30, v39; D-115.6] **Cell 15 reuses Cell 7's search.** The user: "Cell 15 reuses Cell 7's search. Yes". The cell passes `REUSE = ["--reuse-search", OUT + "/calibration"]`: for each set the calibration step first reads Cell 7's `calibration/<set>_search_529878215.csv` and its record (`.json`) and takes that search -- its chosen nodes, made with Cell 7's $\bar B$ and $\hat\mu$ (the pilot table's medians) and Cell 7's transform -- when the record's trace digest and settings equal the step's own and the CSV is the one the record describes; both files are copied to `survey_pilot/529878215/calibration/`, the record with `reused_from`. Expect `529878215 near search reused from .../calibration/near_search_529878215.csv: m chosen of n records, made with B_bar 159.0 gl, mu_hat 0.258 /um, transform {...}; this step's own: B_bar ... gl, mu_hat ... /um, transform {...}`, the far set's line, then the scans with no search time. Otherwise `near search not reused (<why>): searching` and the search as before: `no record ...` for the files Cell 7 wrote on 2026-10-30 (run Cell 7 again first, or let Cell 15 search once), `other settings: calibration` after a knob change, `another trace`, `the CSV changed since its record`. Why $\bar B$, $\hat\mu$ and the transform are not compared (the assistant's reading, PROVISIONAL): the step's own come from Cell 13's at most 18 nodes and Cell 12's automatic alignment, so they never equal Cell 7's; and the search's geometry is the trace's, moved rigidly by the transform -- in the sandbox the far search under Cell 7's transform and under a shift of (2.3, -1.7) px with $z_0$ = 0.1 µm chose 19 nodes each, 15 the same and the other 4 the adjacent node of the same stretch **[run, sandbox: `checks/d115_2026-10-30/`]**. `REUSE = []` searches anyway; the davinci array does not reuse (D-115.6).
