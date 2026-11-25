# TEEG diameter re-measurement -- the soma fit and the node fit's kernel: handoff for the next chat (2026-11-24)

| Date | Change |
|---|---|
| 2026-11-24 | Created at the end of the gallery chat (the user, 17:32: "Let's also continue the pipeline in another chat. Write an handoff with also the instruction to start with the explanation of the soma fit"). State after D-233 to D-243. |
| 2026-11-24 (later) | Section 5's SGT scope checked file by file and split by route (the first list mixed in the table route's Block 12-13 files and `measure.sgt_lambda_um`'s users); section 4's two attributions checked against handoff 14.1 and 14.6; D-243 named in sections 1-2; the roadmap's I49 noted. Evidence: `grep -rni sgt` on `src`, `scripts`, `tests`, `notebooks` **[run, sandbox]**. |
| 2026-11-25 | Section 1 done: the soma fit explained to the user (the chat of 2026-11-25). Section 4: a recovery check with the D-241 region now run on a simulated pilot-like soma -- recovered (a, b, c within 0.06 %, S +0.07 %). Evidence: `checks/soma_explain_2026-11-25/recovery_d241.py` **[run, sandbox]**. `fit_soma.py --synthetic` rerun under today's defaults: recovered (axes within 0.005 um, chi2/dof 9.03). |
| 2026-11-25 (later) | Section 9 added: the independent review of the approach (the user's upload of 13:03) discussed with the user; its soma claims tested on simulated stacks (`checks/soma_review_2026-11-25/sensitivity.py` **[run, sandbox]**); a revised order of work proposed, **not decided**. Sections 3 and 5 are unchanged until the user decides. |
| 2026-11-25 (14:45) | Section 9: two statements of the proposal corrected and one finding added from a profile of the fit over a held c on simulated stacks (`checks/soma_review_2026-11-25/profile_c.py`, stopped early at the user's call) **[run, sandbox]**. Plan still not decided. |
| 2026-11-25 (15:40) | Section 9: D-248 (user, 15:02, c ranged from the literature on human pyramidal somata) and the literature search it asked for; the decision's log entry is held here because the push to `main` was refused in this session (to be copied into the log). |
| 2026-11-25 (15:35) | Section 9: three decisions of the user (15:27) held here until they can be logged on `main` (the push to `main` is refused in this session): the passive model gets the fitted soma's equal-area sphere; the held-c grid extends toward the living range; production cleans the close branches' shadows before the fit. Timing of the held-c fits measured. |

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

## 9. The independent review of 2026-11-25 and what was tested (plan revision proposed, not decided) [added 2026-11-25]

The user had the approach reviewed in an independent chat (brief `TEEG_soma_and_dendrite_estimation_review_brief_2026-11-25.md`,
sent from the gallery chat; the review's handoff `TEEG_soma_dendrite_approach_review_handoff_2026-11-25.md` is the
user's upload, not in the repository) and asked this chat to discuss its soma suggestions and revise the plan.

| Review's soma point | What this chat found | Evidence |
|---|---|---|
| c is fixed by the held far slope, not by the data | supported: a slope 20-24 % off moves c by -59 to +90 % and S by -18 to +28 % (flat somas worst); noise SEs are under 0.4 % | `checks/soma_review_2026-11-25/` **[run, sandbox; linearised]** |
| hold B_k at the value measured outside the soma | not supported: no gain in SE worth having, the slope's bias about doubles on the D-241 region | same |
| the D-241 region holds no soma edge | the premise (a 8.08, b 6.52 um) is the unusable fit's; the silhouette's equivalent radius is 5.95-6.55 um (handoff 14.2 **[data inspected]**); with the edge outside, a and b are still identified in simulation | same; handoff 14.2 |
| a, b from the in-focus outline | a thresholded outline's area depends on c (0.89-0.97 of pi a b); the fit's a, b move at most 3.9 % under the four mis-specifications tried | same |
| c by a prior, the range carried (the review's action 7) | this is option (2) of handoff 14.2; the stems leave the soma over 6.9 um of depth (14.2) | handoff 14.2 |
| the soma's area may not matter | partly: the soma is 1.45 % of the membrane the fit charges; S x 0.75 moves the fitted Cm by +7 to +17 % under the cell-matched parameters, under 3 % under Eyal's | handoff 14.2 **[run, earlier chat]** |
| a mounted soma with c > b is implausible (z-shrinkage) | the stack's depth is stage units; Allen's z-shrinkage factor is not measured (rough bound 2.4, unverified: `docs/handoff_diameter_remeasurement.md`); which c the passive model should get is open | project documents |
| the blur table's last segment (0.61) is below the held 0.79 | the arithmetic holds for the pilot's table (D-170); if the true slope is nearer 0.6, the held-slope fit over-estimates c | **[run]** |

Proposed to the user (awaiting the answer): keep the fit for the centre and the in-plane axes; hold c at the ends and
the middle of a range set from independent constraints and carry S as a range (a hold-c option for Block 46); settle
first what c means under z-shrinkage; run the variants in the sandbox on an exported crop block instead of repeated
Colab runs; the far slope's calibration (D-218.2) only if the range's effect on Cm is too large. Not adopted: holding
B_k; fitting the in-focus blur in the soma fit (D-240.2's open question: under 4 % on c in simulation).

[added 2026-11-25, 14:45] Profile over a held c (`W/checks/soma_review_2026-11-25/`, simulated a 7.0, b 5.6, c 4.5 um,
noise-free, gamma held at 0.79; stopped early) **[run, sandbox]**:
- [corrected 2026-11-25] "Q(c) shows whether the stack prefers a c" (the chat's explanation of 14:12) is withdrawn: with
  gamma held, Q's minimum is the c-free fit's c_hat by construction; in a world of slope 0.60 it lies above 5.0 um (truth 4.5).
- [corrected 2026-11-25] "keep the fit for the centre and the in-plane axes" holds only per held c: a_hat and b_hat
  compensate the held c (+11.7 and +13.9 % at c 2.5 um, -3.9 and -4.1 % at 5.5 um); the 3.9 % of the table above is for
  kernel mis-specifications with c free, not for a held c.
- Finding: because of that compensation S(a_hat(c), b_hat(c), c) stays within -6.2 to +6.3 % of the truth over c 2.5-5.5 um
  (world gamma 0.79) and 407-433 um^2 over c 2.5-5.0 um (world 0.60), against 315 to about 455 um^2 with the axes fixed: in
  simulation the stack fixes S far better than c. The living cell's S, if c is to be scaled for z-shrinkage, needs c itself.

[added 2026-11-25, 14:50] D-240.2 (fit the in-focus blur in the soma fit?), discussed with the user, not decided: in a
model-true simulation sigma_r(0) would be well determined by the soma fit (SE 0.005-0.007 um, correlations <= 0.4, SE(c)
unchanged: `W/checks/soma_review_2026-11-25/sigma0_identifiability.py` **[run, sandbox]**), but it moves c by at most 4 %
and S by at most 1.3 % (0.080 against 0.258 um, `sensitivity.py`). Proposed: keep it out of the default; run Cell 25 at
KERNEL_WIDTH "fixed" and "calibrated" for the systematic; free it once only as a second, independent estimate of
sigma_r(0) (an edge test on the soma's rim), to compare with node 7's 0.258 +- 0.022 um.

[added 2026-11-25, 15:40] **D-248 (user, 15:02)**: "I would say it is better to search across the literature for human
cells for this" -- the range of c is set from published measurements of human pyramidal somata, read in full text or in
the data, each value with its preparation. NOT YET IN THE LOG: the push to `main` was refused by this session's safety
check; copy the entry into `docs/project/TEEG_decisions_and_ideas_log.md` on `main` as D-248 (the next free ID at 15:10;
check before taking it) when the user allows it or does it.
Literature search (PubMed with full texts, the project's files, bioRxiv's published-version lookup, Allen's technical
documentation) and its arithmetic (`W/checks/soma_review_2026-11-25/literature_c_range.py` **[run, sandbox]**):
- Benavides-Piccione et al. 2024, Cereb Cortex, doi 10.1093/cercor/bhae180 **[full text]**: human L3a temporal somata,
  area of the maximum perimeter 244 +- 5.1 (BA20, n 90) and 253 +- 7.5 um^2 (BA21, n 50); human CA1 5041 +- 273.1 um^3 and
  350 +- 9.7 um^2 -> c/r = 1.02 (ratio of means, ellipsoid assumed); mouse CA1 1.07. Lucifer Yellow in fixed 300 um
  sections, confocal 63x, Neurolucida 360 soma contours, ProLong Gold or 50 % glycerol; no z correction stated. Their
  neocortical volumes are only in Fig. 3A / Suppl. Table 1A (not read).
- Mohan et al. 2015, Cereb Cortex, doi 10.1093/cercor/bhv188 **[full text]**: human 350 um slices, DAB, mowiol; 140.7 um
  after mounting (median), z-shrinkage 63 +- 10 %.
- Allen's technical white paper (Cell Morphology and Histology, v.7, 2018) **[documentation, read through a fetch]**:
  350 um slices, PFA/glutaraldehyde, DAB, glycerol-based Mowiol; no shrinkage measurement or correction stated; the soma
  radius from an outline drawn on one section.
- Derived: living c ~ 5.1-7.0 um (c/r 0.86-1.07 times the pilot's in-plane radius 5.94-6.56 um); stack c ~ 1.4-3.3 um
  (times 0.27-0.47, Mohan's mounted/original), central 1.9-2.8 um, within the stems' bound c <= 3.45 um; living S ~ 403-565
  um^2 (Allen's sphere 448). Assumptions: the soma shrinks in depth like the slice; x-y shrinkage negligible; the 0.84
  confocal factor of the same lab (Eyal et al. 2018) sets the low end of c/r.
- Open: Suppl. Table 1A of Benavides-Piccione 2024 (the user can upload it); whether that lab corrected z for somata;
  Allen's z-shrinkage for this specimen (the stack's tissue thickness against 350 um would measure it); H01's per-cell
  soma meshes (EM, human temporal cortex), the most direct data, not reachable from this session.

[added 2026-11-25, 15:35] **Decisions of the user, 15:27, NOT YET IN THE LOG** (copy them to `main`, taking the next free
IDs there):
1. "the fitted model will have the soma (rounded) that we find here, we will find the best fitting c, calculate the S and
   determine the radius of the sphere that has the same superficial area" -- the passive model gets the soma of the stack
   as fitted: the held c with the smallest misfit, S(a_hat, b_hat, c), and the equal-area sphere r_eq = sqrt(S / 4 pi)
   (Block 47's rule). Closes D-248's open frame question: the stack's, not the living cell's. Told to the user: with the
   far slope held, the best-fitting c is the c-free fit's c_hat (a profile's minimum is the joint minimum), so it rests
   on gamma = 0.79; and if the soma flattened with the slice, the stack's S is about 0.6-0.7 of a round living soma's
   (pilot radius 5.94-6.56 um: 249-374 against 403-566 um^2) **[run, sandbox]**.
2. "I would do more C fits also towards the lower end of the living cells" -- the held-c grid runs from the stack's range
   up to about the low end of the living range; proposed 1.5-5.5 um in 0.5 um steps (9 fits), not yet confirmed.
3. "we will first clear the close branches' shadow and then doing the fit in the HPC; now for the pilot we keep this" --
   production (HPC): the close branches' shadows are cleaned before the soma fit (settles the when of D-241.3); the pilot
   keeps D-241's region with the shadows in.
Timing **[run, sandbox: `checks/soma_review_2026-11-25/profile_g0*.out`]**: 15 held-c fits on simulated noise-free
stacks with warm starts took 30-320 s (median 169 s), 9-113 renders (median 60), about 3 s per render at c 2.5-6.5 um on
2 shared CPUs; the render's cost scales with the slab count (22 slabs at c 1.5 um, 72 at 5.0 um).
