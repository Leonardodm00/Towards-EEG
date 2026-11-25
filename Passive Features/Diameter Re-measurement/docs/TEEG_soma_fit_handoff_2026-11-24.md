# TEEG diameter re-measurement -- the soma fit and the node fit's kernel: handoff for the next chat (2026-11-24)

| Date | Change |
|---|---|
| 2026-11-24 | Created at the end of the gallery chat (the user, 17:32: "Let's also continue the pipeline in another chat. Write an handoff with also the instruction to start with the explanation of the soma fit"). State after D-233 to D-243. |
| 2026-11-24 (later) | Section 5's SGT scope checked file by file and split by route (the first list mixed in the table route's Block 12-13 files and `measure.sgt_lambda_um`'s users); section 4's two attributions checked against handoff 14.1 and 14.6; D-243 named in sections 1-2; the roadmap's I49 noted. Evidence: `grep -rni sgt` on `src`, `scripts`, `tests`, `notebooks` **[run, sandbox]**. |
| 2026-11-25 | Section 1 done: the soma fit explained to the user (the chat of 2026-11-25). Section 4: a recovery check with the D-241 region now run on a simulated pilot-like soma -- recovered (a, b, c within 0.06 %, S +0.07 %). Evidence: `checks/soma_explain_2026-11-25/recovery_d241.py` **[run, sandbox]**. `fit_soma.py --synthetic` rerun under today's defaults: recovered (axes within 0.005 um, chi2/dof 9.03). |

Repository `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline` (workstream `Passive Features/Diameter
Re-measurement/`, called W below); SPEC at the repository root `specs/SPEC.md`; the decision log on `main`
(`docs/project/TEEG_decisions_and_ideas_log.md`). The detailed record of the soma work is
`W/docs/TEEG_clear_node_fit_handoff_2026-11-19.md` sections 14.4-14.7 and 15; this file is the entry point.

## 1. First task of the next chat: explain the soma fit to the user

The user asked that the next chat **start by explaining the soma fit** -- as it stands after D-240, D-241 and D-243,
step by step, with images. Do it before any other work:

- Load the `explanation-modes` skill (Mode A, comprehensive: CONTEXT, NOTATION, GLOSSARY, BODY in two parts) and follow
  it; the project's source rules apply (SPEC, this file, the handoff above, the log; PubMed where a claim needs it).
- Cover, in the pipeline's order (section 3 has the settings): the start plane (Cell 24 (b), SOMA_K); the planes (35,
  why); the region R (Allen's radius, the branches' pixels out, the processes' shadows kept); the forward model (the
  ellipsoid's column, partition absorption, the Gaussian blur law with the far slope held); the fit (variable projection
  for each plane's brightness, trf, the box, the SEs and what they do not measure); the area (Legendre's form; S rises
  almost linearly with c); the corrected SWC (the sphere of equal area, the stems moved); the bound flag (D-236).
- Then what is known not to work yet (section 4): on the pilot the free-slope fit ran c and z0 to their bounds; why
  (the c-z0-slope trade-off; the region was mostly background); what changed since (slope held, D-240; region, D-241).
- Images: regenerate with `W/checks/soma_diagnosis_2026-11-24/` (`fig_region.py` for the region, `explain_figures.py`
  for the area and the depth trade-off; they need the user's run files, section 6) and draw new ones where they earn
  their place. Keep the answer pedagogical; the user is a PhD student who wants the plain account first.
- The user's habits seen in this chat: "be brief" means Mode B; push-back is precise and usually right (check levels:
  a quantity computed from data against the same-named model quantity); they want images; they decide the science
  (log their decisions at once).

## 2. Where things are

| What | Where | State |
|---|---|---|
| Soma fit entry point | `W/scripts/fit_soma.py` (Blocks 45-48) | far slope held (D-240), region D-241, crop D-243, bound flag D-236 |
| Soma fit core | `W/src/allen_diameter/analysis/soma_fit.py` (Block 46), `model/soma_render.py` (45), `analysis/soma_morphology.py` (47: the corrected SWC), `plotting/soma_fit_figures.py` (48) | tested (section 7) |
| Soma plane viewer | `W/src/allen_diameter/plotting/plane_viewer.py`, `loading/soma_io.py`; `scripts/show_soma.py --z-margin-um` (Block 44 amendment 4, D-239) | smoke-tested; not tried in a browser |
| Notebook | `W/notebooks/cleaning_pilot_2026-11-14.py`: Cell 24 (static montage with Allen's trace), Cell 24 (b) (slider, no overlay, sets SOMA_K), Cell 25 (the soma fit) | Cells 26 and 26 (b) removed (D-242) |
| Node fit | `W/scripts/infer_nodes.py`, `analysis/node_inference.py` (Blocks 30, 36) | Gaussian kernel only from now on (D-242); the SGT code still present, to be removed (section 5) |
| Configuration | `W/src/allen_diameter/config.py`: `soma_fit` (outside every signature hash), `inference` (outside), `renderer` | |
| Roadmap | `specs/roadmap/diameter-pipeline.yaml`, `.md` (render with `python3 specs/roadmap/roadmap.py render --yaml specs/roadmap/diameter-pipeline.yaml`) | rendered 2026-11-24 |
| Checks of this chat | `W/checks/soma_diagnosis_2026-11-24/`, `W/checks/sgt_width_2026-11-24/` (each with a README) | standalone, not pipeline code |

## 3. The soma fit as it stands (pilot cell 529878215)

| Step | Setting | Source |
|---|---|---|
| Start depth | `SOMA_FIT_K` in Cell 25; None: the `SOMA_K` set in Cell 24 or with Cell 24 (b)'s button; else Allen's traced depth (32.76 um, k 117) | D-211, D-239 |
| Planes | every second recorded plane within +-10 um of the start: 35 planes (k 83..151 from the traced depth) | D-219.4, PROVISIONAL (`plane_half_um` 10, `plane_step` 2) |
| Region R | the disc of Allen's traced soma radius (5.97 um on the pilot) about Allen's centre | D-241 (`region_radius_um` None; a number overrides) |
| Mask | the traced branches at their traced radius (no widening), every depth; the soma-rooted segments not masked; the processes' shadows stay in the data (unmodelled, a slight bias accepted) | D-241 (`mask_widen_um` 0) |
| Crop (the model's support) | half-width max(14 um, region radius + 1 um) about the traced centre: the renderer draws the ellipsoid on the crop alone, so the crop must hold it; the record's `crop.ellipsoid_inside` flags a fitted footprint beyond it (logged as a warning) | D-243 [orchestrator] (`render_half_um` 14.0, PROVISIONAL) |
| Forward model | ellipsoid (x0, y0, z0, a, b, c, phi, mu), slabs 0.14 um, partition absorption, Gaussian blur of SD sigma_r(delta) | Block 45, D-216, D-219 |
| Blur law | near table: Cell 16's kernel (KERNEL_WIDTH "fixed": sigma_r(0) 0.080 um, ideal-Debye core widths, NOT fitted to the stacks; "calibrated": Cell 23B's sigma_0 0.258 um, node 7 only); beyond 0.84 um linear with slope 0.79 (optics model D-024), held | D-172, D-221, D-240 |
| Fit | per-plane brightness by variable projection; `least_squares` trf; centre +-4 um (x, y), +-6 um (z) of the start; a, b, c in [1, 15] um; mu in [0.001, 3] /um | Block 46, D-219.4 |
| Bound flag | a parameter within 1e-3 of its bound interval's width from a bound is "at a bound" (red in the summary figure) | D-236 |
| Area | Legendre's closed form with incomplete elliptic integrals (sorted semi-axes); r_eq = sqrt(S / 4 pi) | Block 46 |
| Corrected SWC | the soma node at the fitted centre with r_eq; each stem's first node moved along the stem onto the fitted surface; nodes inside removed | Block 47, D-216.3, D-222.5 |

Not decided: fitting the in-focus blur in the soma fit (the user: "the only thing we might fit is the in-focus blur",
D-240.2); cleaning the processes from the soma's planes with the existing cleaning (D-241.3).

## 4. What the pilot showed so far

| Run | Result | Reading |
|---|---|---|
| Cell 25, free far slope, start at the traced depth, 13 um region with + 2 um mask (the user's upload of 14:15) | c 15.00 um (upper bound), z0 26.76 um (lower bound, 6 um shallower than traced), slope 0.367, S 1174 +- 5.6 um^2 (Allen 448); residual RMS 3.5-6.3 gl per plane, largest on the top planes (handoff 14.6) | not usable: the fit pushed absorption as far toward small z as the box allowed **[data]** |
| Cell 25, far slope held at 0.79 (the user's planes figure of 16:30) | z0 30.96 um, inside its box; c and S not in the figure | the summary figure or the JSON would show c **[data]** |
| Not run yet | start at the user's SOMA_K (Cell 24 (b)), region D-241 | the next real test |

Diagnosis of the free-slope run (handoff 14.6, checks of `W/checks/soma_diagnosis_2026-11-24/`) **[run, sandbox]**:
c, z0 and the slope trade off -- a compact soma (c 7.26 um) with z0 28.39 um and slope 0.513 imitated the fitted one
within 2.4 gl RMS, half the fit's residual; the old region was 82 % background more than 1 um outside the soma's
outline (94 % of the outline masked); a bright background wedge or a tilt moved c by at most 0.1 um per gl, so not
the cause; what pulled the fit toward small z is not identified (candidates: the soma's real depth -- against the
user's D-211 reading, the soma seen from k = 107 toward larger k; an unmasked dark structure in the soma's column --
the first Cell 24 run found a basal dendrite in sharp focus inside the traced disc at k = 93, handoff 14.1; a far blur
that is not the Gaussian law).
With the D-241 region (7442 px per plane; the old region rebuilt by the same rule 7486, the run counted 7644) the
depth signal is somewhat larger **[run, sandbox: `region_new_check.py`]**: a compact soma (c 7.26 um, 2 mu c kept, z0
and slope as fitted) differs from the fitted one by 2.3 gl RMS per plane (median, each plane's brightness re-solved)
against 1.4 gl on the old region; the old fit's imitator (c 7.26, z0 28.39, slope 0.513, re-fitted on the old region)
by 1.9 against 1.4 gl -- the trade-off is reduced, not removed (and with the slope now held, D-240, one of its three
legs is gone). A recovery check on simulated stacks with the D-241 region has not been run. [corrected 2026-11-25: run -- a simulated pilot-like soma (a 7.0, b 5.6, c 5.0 um, z0 1.8 um above the traced depth, noise 3 gl, no branches, the model true) is recovered with the far slope held and the D-241 region: converged, nothing at a bound, a, b, c within 0.06 %, S +0.07 %, chi2/dof 9.07 (`W/checks/soma_explain_2026-11-25/`) **[run, sandbox]**. So the fit's settings identify c when the model holds; what happens on the pilot's stack is still step 2 of section 5.]

Withdrawn in this chat: holding mu at the dendrites' 0.18 /um (it forces 2c = 0.87 / 0.18 = 4.8 um).

## 5. Next steps, in order

1. **Explain the soma fit** (section 1). [done 2026-11-25]
2. **The user picks SOMA_K** with Cell 24 (b) (slider; contrast "fixed" or "per plane"; the button sets SOMA_K), then
   runs Cell 25 (Cell 1 pulls the branch first) and sends back `soma_fit_529878215.json`, `_planes.png`,
   `_summary.png`, `_morphology.png`. Check: the D-236 flags (c, z0 at a bound?), `crop.ellipsoid_inside` (D-243),
   the residual by plane. The crop is the pre-D-241 size again (D-243), so the time is as before: 48 min on Colab
   for the run of 14:15 (Cell 25's note: 8-21 min in the sandbox).
3. **Remove the SGT code** (D-242: "totally drop the SGT"; the notebook already has no SGT cell). Through the
   `scientific-coding` skill (SPEC first: the dropped marks are in place on Block 49, Block 30 amendment 3, Block 36
   amendment 23 and section 6's `infer_nodes.py` row). Scope, from `grep -rni sgt` read file by file
   **[run, sandbox]**:
   - **In** (the node fit's SGT: D-221, D-222, D-224, D-225; wave 7): `src/allen_diameter/model/line_sgt.py`
     (Block 49); `model/line_fast.py` (its `SGTKernel` branch); `analysis/node_inputs.py` (`kernel_family` "sgt");
     `analysis/node_inference.py` (the SGT stage, `SHAPE_NAMES`); `scripts/infer_nodes.py` (`--kernel sgt`,
     `--sgt-skew`; keep `--cov-fine-factor`); `config.py`: `INFERENCE_KERNELS`, `SGT_STRETCH_ALLOWANCE`,
     `inference.kernel_family`, `inference.sgt_*` and their validation (`inference.cov_steps` keeps its first four
     entries' meaning; its three shape entries fall out of use); tests: `tests/smoke/test_smoke_line_sgt.py`, the SGT
     parts of `test_smoke_node_inference.py` and `test_smoke_config.py`; `tests/spec/test_spec_block49_line_sgt.py`,
     `wave7_refs.py`, the SGT parts of `test_spec_block36_am22_23.py`, `test_spec_block30_node_inference.py` and
     `test_spec_block01_config_cleaning.py`.
   - **Out** (the table route's SGT: D-052, Blocks 12-13; the user did not discuss it -- ask before touching):
     `analysis/fit_sgt.py`, `model/blur_kernels.py`, `analysis/plane_iteration.py` (`sgt_at_plane`),
     `analysis/node_pipeline.py`, `analysis/cell.py` and `analysis/phantoms.py` (Block 13's columns), `measure.sgt_*`
     in `config.py` -- including `measure.sgt_lambda_um`, the centre-weight length that `analysis/contamination.py`,
     `analysis/coarse_fit.py` and their tests use (a weight, not a kernel); `tests/smoke/test_smoke_sgt_fit.py`,
     `test_smoke_block13.py`; the ignored artifacts under `tests/smoke/artifacts/`.
   - Run the suites; an independent Opus check after (coding directive 9). The roadmap's stage I49 (Block 49) stays
     until its code is gone; whether to drop the stage then is the user's call (the roadmap is not restructured
     unasked).
4. **Open questions to raise when they become relevant** (not decided): fit the in-focus blur in the soma fit
   (D-240.2); clean the processes from the soma's planes (D-241.3); the far slope's calibration on Allen's data (the
   roadmap's far-field stage, D-218.2); whether the real blur's shape biases the Gaussian fit's d on thin nodes (a
   cell-level shape from many thin nodes, with its width matched through the calibration's own Gaussian-core
   estimator, would answer it; the FWHM rule of D-222.2 does not reproduce that estimator except for a Gaussian --
   `W/checks/sgt_width_2026-11-24/core_fit_proxy.py`).

## 6. Data the checks need (not in the repository)

The user's uploads of 2026-11-24 (Colab `OUT`; this chat's sandbox copies, `/tmp/up24/`, are not reachable from
another chat):
`soma_fit_529878215.json`, `reconstruction_soma_529878215.swc`, `soma_fit_529878215_planes.png`, `_summary.png`,
`_morphology.png` (Cell 25, free slope, 14:15); `inference_529878215.csv` / `.json` (Cell 26 on node 5353, 14:15);
the held-slope planes figure (16:30). Ask the user to upload them again if a check has to be rerun.

## 7. Tests run at the end of this chat [run, sandbox]

| Suite | Result |
|---|---|
| `tests/smoke/test_smoke_soma_fit.py` | 9 pass, 1 skip (after D-240, D-241, D-243) |
| `tests/smoke/test_smoke_config.py` | 6 pass, 1 skip (after D-243) |
| `tests/smoke/test_smoke_plane_viewer.py` | 7 pass, 1 skip (ipywidgets 8.1.9) |
| `tests/smoke/test_smoke_soma_view.py` | 7 pass, 1 skip |
| spec: Blocks 01 (config), 45, 46 (with `_0027`), the 44-46 boundary, the soma fit end to end | after D-241: 226 pass, 1 fail -- the boundary test used `region_radius_um` as a number; fixed (it now pins the explicit-radius path, 13 um + 2 um, because under the defaults the synthetic cell has no branch inside the cut and the mask is empty); the boundary file rerun: 3 pass |
| spec after D-243: Blocks 01, 46, the 44-46 boundary, the soma fit end to end | 165 pass (Blocks 45 and 46 `_0027` not rerun: D-243 touches neither) |
| `tests/smoke/test_smoke_soma_fit_figures.py`, `test_smoke_soma_view.py` | 13 pass, 2 skip (pytest, the two files together) |
| Not run after D-241 / D-242 | the full spec and smoke suites (the next night test covers them); the notebook in Colab (Cells 26 and 26 (b) are gone: a Colab copy opened before the pull still has them) |

## 8. Decisions of this chat

| ID | Decision | By |
|---|---|---|
| D-233 | wave 7's final check settled | orchestrator |
| D-236 | the bound flag: within 1e-3 of the bound interval's width, or the solver's mark | user + orchestrator |
| D-239 | the soma's planes in a slider viewer, nothing drawn over them; the pick sets SOMA_K | user |
| D-240 | the soma fit does not fit the far slope (held at 0.79); only the in-focus blur might be fitted, not decided | user |
| D-241 | the soma fit's region: a disc of Allen's soma radius, only the branches' pixels left out | user |
| D-242 | the SGT dropped: the node fit renders and fits with the Gaussian only | user |
| D-243 | the soma fit's crop is the renderer's support: at least 14 um half-width, whatever the region; the record flags a fitted ellipsoid beyond it (corrects D-241.1's crop sentence, the orchestrator's) | orchestrator |

D-237 and D-238 are another chat's (the twins).
