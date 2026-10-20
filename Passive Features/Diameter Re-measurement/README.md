# Diameter Re-measurement (Allen 63x brightfield stacks)

Re-measuring dendrite diameters from the Allen Cell Types 63x brightfield image
stacks, so that the SWC radii can be corrected before the passive fit (Cm) is
redone. First specimen: 529878215 (human). Part of the passive-parameter
estimation from Allen data (`Passive Features/`).

**Status (2026-10-04): design and documentation only. Nothing in the
measurement pipeline is coded yet.** The image-access code (`allen_image_io.py`,
`allen_image_align.py`, `allen_image_plot.py`, `smoke_allen_image.py`,
`robustness_registration.py`) lives in the user's Google Drive
(`MyDrive/Colab Notebooks/Allen Slices/Codes`) and is not in this folder yet.

**Status (2026-10-06, implementation chat):** coding started on branch
`sci/diameter-pipeline` (decisions D-021 to D-024; the focus rule since 2026-10-07: D-030). Done: Block 0 (the six
2026-09-23 image modules imported byte-identical into `src/` and `tests/smoke/`,
suites 20/20 and 40/40 re-run), Block 1 (`src/allen_diameter/config.py`, every
parameter with its source), the SWC loader/writer of Block 8
(`src/allen_diameter/loading/swc_io.py`) and `scripts/allen_radius_distribution.py`.
The spec is the root-level `specs/SPEC.md` (Coverage = this folder). Run the
smoke tests from this folder: `python tests/smoke/test_smoke_config.py`,
`python tests/smoke/test_smoke_swc_io.py`; the imported suites with
`cd tests/smoke && PYTHONPATH=../../src python smoke_allen_image.py`.

**Status (2026-10-06, earlier):** still nothing coded. By D-021 [user, 2026-10-06 14:30]
the whole pipeline is now implemented in a separate chat, in parallel with the
theory study. Unsettled method choices enter the code as named configuration
parameters with labelled provisional defaults, confirmed in one batch. Start
from `docs/TEEG_diameter_implementation_handoff_2026-10-06.md`. The code is
proposed to live in this folder (`src/`, `specs/`, `tests/`, `scripts/`) on
branch `sci/diameter-pipeline`. **[corrected 2026-10-06]** Drive holds only the
2026-09-22 versions of the image modules, and no `robustness_registration.py`.
The only known copies of the 2026-09-23 versions are the files delivered in the
design chat, so the user has to supply them.

## Method in one paragraph

For each SWC dendrite node: find the sharpest plane, fit a straight line through
local centres to get the branch direction and tilt, take the intensity profile
perpendicular to the branch in the sharpest plane, and fit a blurred Beer-Lambert
tube to it (handoff Eq. 11, with the background fixed from a focal-plane median,
decision D-018, and the stain absorption coefficient mu fitted with the tilt from
the line fit, decision D-019). The fitted diameter is then corrected for the
optical bias of the whole chain with a table b(d, phi | C) built from synthetic
stacks of known geometry, and flagged nodes are filled from neighbours.

## Where the documents live (decision of 2026-10-04)

The documents of this workstream are kept **here**, in `docs/`, on `main`:
new documents and updates are committed and pushed to this folder, and the
assistant reads and cites them from the repository [user, 2026-10-04 12:47;
recorded as D-020]. Older copies of the procedure and mathematics documents and
of the optics notes also sit in the claude.ai project knowledge under
`claude/`; they are not the reference. Cross-references inside the documents
that say `claude/<name>` mean the same file name in this folder.

## Contents

### `docs/`

| File | What it is |
|---|---|
| `TEEG_diameter_table_implementation_plan_2026-10-20.md` | **The implementation plan of the correction table after the decisions of 2026-10-20** (D-061 to D-065, D-067; D-066 withdrawn): what is being built, the state at `84ea91f`, the stages S0-S8 (Block 13, Block 10's selection, the Colab surveys and calibration, Block 14, gate 2, the davinci pilot, production, the application to the cells) with their gates, the cost envelope, the decisions still to take and when. Replaces the 2026-10-19 handoff's §6 where they differ. |
| `TEEG_diameter_correction_table_handoff_2026-10-19.md` | **Start here to continue the bias (correction) table** (2026-10-19, end of the table chat). The state after Block 12: the decisions of the table chat (D-053 to D-060) and what of them is coded; today's code against them; the open points, those Block 13 needs and those for the user before Block 14; the plan of Blocks 13-14 and the gates; the table chat's checks and what they found. Supersedes the 2026-10-16 handoff's §1, §2, §5, §7, §8, §10 and §11; that file's §3, §4 and §6 stay the reference for the estimator and its evidence. |
| `TEEG_thin_plate_spline_in_3D/` | **What the table's spline is in three dimensions** (2026-10-18, a study document with its `figures/` and `scripts/`): from the bending plate to SciPy's $\varrho^2\ln\varrho$, the order-5/2 penalty it minimises on $\mathbb R^3$, the equivalent kernel and its negative weights, and the kernel comparison on a fixture of the table's design. |
| `TEEG_diameter_correction_table_handoff_2026-10-16.md` | [updated 2026-10-19: for the current state read the 2026-10-19 handoff; §3, §4 and §6 of this one stay the reference] **Start here to build the bias (correction) table** (2026-10-16). The settled per-node estimator (D-052: the Allen-guided plane, the skewed generalized-t kernel with its three shape parameters and its width fitted, centre weights at lambda = 0.5 um) in full notation; the simulation evidence behind it; today's table code against D-052, D-030 and D-031; where the node's radius enters; ten open questions with options; the block plan (Blocks 12-14) and the gates. |
| `TEEG_diameter_implementation_handoff_2026-10-06.md` | **Start here to implement the pipeline (D-021).** Contents: <br>- how the theory chat and the implementation chat split the work; <br>- decisions in force; <br>- the pipeline flow (real-data path, synthetic path, kernel calibration, cell level); <br>- the proposed code layout and Colab bootstrap; <br>- the block plan with smoke oracles; <br>- equations (S1)-(S10), which are written nowhere else; <br>- every open choice as a configuration parameter with a provisional default; <br>- the theory chat's findings (partition artefacts for dark tubes, kernel far field, phantom length, heading); <br>- validation, known gaps, first actions. |
| `TEEG_diameter_remeasurement_handoff_2026-10-05.md` | **Start here in a new chat.** (For the method and documents. For code, start from the 2026-10-06 implementation handoff above.) Handoff after the documentation chat: status, document map, decisions in force, corrections, open choices, next actions, admin left open (project-knowledge limit, D-020 not yet in the project log). Read after the design handoff below. |
| `handoff_diameter_remeasurement.md` | The design handoff (2026-09-30): data access, Eqs. 1-13, local decisions D1-D7, next actions. The source of the equation numbers "handoff Eq. n". Copied unchanged. |
| `TEEG_diameter_bias_table_procedure_2026-10-04.md` | **How the bias table b(d, phi \| C) is tabulated** (v1.1): configuration, defocus-kernel calibration, phantom rendering (with a walkthrough of one output plane on figure 4), measurement, estimation, inversion, flags, falsification checks. |
| `TEEG_diameter_bias_table_mathematics_2026-10-04.md` | **The mathematics behind it** (v1.1): forward model, Beer-Lambert law, LSF, slabs and slab rendering with the absorbed-light partition, absorbance vs absorbed fraction, squared-width additivity, defocused LSF, calibration identifiability, bias statistics and inversion. |
| `TEEG_microscope_optics_oil_immersion_2026-10-04.md` | **The physics of the microscope** (v1.2, with pointers to figures 1-3): light path, refraction, oil immersion and NA, diffraction and the in-focus blur, forming vs resolving, defocus and why the blur stays flat near focus, condenser and coherence, mounting-medium index mismatch. |
| `TEEG_interactive_figures_spec_2026-10-04.md` | **Instructions for Claude Design to reproduce the four interactive figures**: shared design rules, and per figure the claim, layout, controls, model equations, acceptance values and a paste-ready prompt; known gaps. |
| `TEEG_diameter_optics_notes.md` | Running notes (v2, 2026-10-04) summarising the chat explanations, with corrections marked "[corrected <date>]". |
| `TEEG_diameter_phase2_colab_runbook_2026-10-06.md` | **The real-data steps in Colab** (v17, 2026-10-14) [corrected 2026-10-08: the heading said v5 after v6]: Cells 0-10 in order (bootstrap, node list, registration, pilot, the plane montages of chosen nodes with Allen's trace drawn on them (Cell 4d), the plane-to-plane evaluations around chosen nodes (Cells 4e-4g), the focus scores compared across diameters (Cells 4h and 4i; Cell 4j, the line sized from Allen's diameter and then from the fit [added 2026-10-10, later]), camera inputs, radius distribution, kernel calibration, production configuration, the davinci table, apply to the cell), each with the line that must appear and the decisions it feeds. [added 2026-10-07: the runbook was not listed here] [updated 2026-10-07: v3, the focus rule of D-030; v4, Cell 4d] [updated 2026-10-08: v5, Cell 4e; v6, Cell 4e's profile-area evaluation; v7, Cell 4f, the entropy evaluation; v8, Cells 4e and 4f at two line lengths] [updated 2026-10-09: v9, Cell 4g, the entropy on thin dendrites; v10, Cell 4g shows each node's figure, its planes drawn without the measuring line; v11, Cells 4h and 4i, the gradient energy over the whole line on three lines and its sigmoid-weighted blend with the strip entropy, across diameters (D-040)] [updated 2026-10-10: v12, Cell 4i takes the entropy from the +-5 um strip and writes the blend J = w g - (1 - w) h (D-041); v13, Cell 4j, the gradient energy on lines sized from Allen's diameter and then from the fit at the plane each picks, until the plane repeats (D-042)] [updated 2026-10-12: v14, Cell 4j at m = 3, lines +-1.5 d (D-044), into `pilot/planeiter/bydiameter_m3/`; v15, Cell 4k, the planes the iteration chose with the line that chose them and the diameter fitted there (D-045)] [updated 2026-10-13: v16, Cell 4l, fit variants at those planes, other blur kernels and centre weights (D-046)] [updated 2026-10-14: v17, Cell 4m, each fit variant's own Allen-guided iteration, its edges and its plane (D-047)] |

### `notebooks/`

| File | What it is |
|---|---|
| `phase2_colab.ipynb` | The runbook's cells as a Colab notebook (2026-10-07), same numbers, same code. Open it in Colab from GitHub (File > Open notebook > GitHub, branch `sci/diameter-pipeline`). |
| `phase2_colab.py` | The same cells as plain Python (2026-10-07), one block per Colab cell between `# ===== CELL` banners, labelled with the runbook's numbers. |

The three 2026-10-04 documents were independently reviewed and revised the same
day; each ends with a revision record. Decisions D-018 and D-019 are recorded in
the project's decision log (`TEEG_decisions_and_ideas_log.md`, project knowledge).
That is the root-level copy; `claude/TEEG_decisions_and_ideas_log.md` is an older
copy with D-001 and I-001 only. D-020 (above) and D-021 (implementation handoff) are not yet
in the log, because project knowledge is at its size limit.

### `figures/`

Standalone interactive HTML pages (open in any browser; no build step, no
external data). Their behaviour and acceptance values are specified in
`docs/TEEG_interactive_figures_spec_2026-10-04.md`. The blur widths and
kernels in them are illustrative ideal-objective values, not the calibrated
kernel.

| File | Shows | Used in |
|---|---|---|
| `fig1_defocus_cone_disc_lsf.html` | cone of light, spot at a movable defocus, spot summed across a branch (uniformly bright spot: upper-bound RMS) | optics §3.5 |
| `fig2_diffraction_focal_spot_vs_na.html` | plane waves from a cone interfering into a spot; spot width vs NA (2-D scalar model; axis drawn horizontal) | optics §3.4 |
| `fig3_tilted_plane_wave_lateral_frequency.html` | one tilted plane wave and its lateral period lambda/(n sin theta) | optics §3.4 |
| `fig4_slab_rendering_one_output_plane.html` | a tube cut into slabs, absorbed-light partition, per-slab blur and the summed profile of one output plane, vs the linear sum | mathematics §3.2, procedure §3.6 |

### `checks/`

Small numerical checks cited in the documents as **[run, script]**. They are
exploratory scripts, not part of the pipeline, and have no smoke tests of their
own. Requirements: `numpy`, `scipy`, `Pillow` (for `blur_chain_check.py`).
Run them **from inside `checks/`**, because several of them load
`psf_check.py` by relative path:

```bash
cd "Passive Features/Diameter Re-measurement/checks"
python psf_check.py          # Airy / Debye PSF and LSF widths, defocus (slow, minutes)
python doc_checks.py         # C1-C6 of the 2026-10-04 documents (C5 is slow)
python review/rev_geo.py     # reviewer's checks also run from here
python stack_geometry_check.py                       # (S1)-(S4), heading, grid, U   (~3 min)
python optics_points_check.py checks A BC matched    # kernel, partition, ray world (~4 min)
python followup_checks.py; python focus_scan.py      # import optics_points_check (seconds)
```

| Script | Verifies |
|---|---|
| `psf_check.py` | Airy and scalar Debye PSF at NA 1.4: FWHM, first zero, LSF core width; LSF at small defocus |
| `blur_chain_check.py` | Variance added by pixel integration, bilinear interpolation and JPEG to a thin line |
| `defocus_forms.py` (+ `defocus_forms.out`) | Debye LSF core width vs defocus; extrapolation of three analytic growth laws |
| `doc_checks.py` | C1 partition conservation, C2 single-slab reduction, C3 partition vs linear sum, C4 squared-width additivity, C5 windowed LSF widths, C6 acceptance angle vs mounting index |
| `axial_check.py` | Axial FWHM of the Debye PSF; pixel pitch 4.54/(63 x 0.63); critical angle |
| `fwhm_mix.py` | How half-maximum width mixes diameter, blur and darkness |
| `mu_tie.py` | D-019: per-node mu vs alpha parameterisation; shared mu under a sigma error |
| `var_cancel.py` | Cancellation of the tube's own width in plane-to-plane width differences |
| `review/rev_c14.py` | Independent re-run of C1-C4 |
| `review/rev_geo.py` | Debye LSF far from focus vs the geometric law |
| `review/rev_tail.py`, `review/rev_tail2.py` | The \|v\|^-2 tail of the LSF in focus and out of focus |
| `review/ax2.py` | Axial first minimum; stage-unit defocus growth for low-index mountants |
| `stack_geometry_check.py` (+ `.out`) | 2026-10-05/06, source of the implementation handoff's (S1)-(S4) numbers: <br>- vertical-ray interval vs 3-D membership; <br>- exact slab chords; <br>- rotation equivariance on the fine grid and its loss after pixel integration; <br>- node-plane dip and flank haze vs phantom length $U$. <br>Uses the kernel continued in proportion beyond 0.84 µm. |
| `optics_points_check.py` (+ `.out`) | 2026-10-06, the three renderer assumptions: <br>- kernel far field and the three continuation rules (A); <br>- absorbed-light partition and vertical rays vs a geometric-optics ray world (BC); <br>- the matched comparison after $\mu$ matching (`matched`). |
| `followup_checks.py` (+ `.out`) | Grid convergence of A2; partition centre dip vs stain darkness in the ray world and with the Debye kernel; 3-point focus vertex (meaningless when not peaked) |
| `focus_scan.py` (+ `.out`) | Focus curve over ±6 planes for dark and faint flat tubes; opaque-limit leak of the partition, (S9) |
| `block13_2026-10-20/` | Block 13's oracles that need the repository's history: `check_13_1_move.py` (+ `.log`) runs the iteration of `scripts/plane_differences.py` at `84ea91f` (read with `git show`) and the moved one of `analysis/plane_iteration.py` on the synthetic cell, and asserts equal records (16 runs); `check_13_2_square.py` (+ `.log`) runs `survey.node_planes` and `node_stack` of `84ea91f` against the shared scene and square rules, key by key, squares inside the block and fetched, and the iteration on each pair (24 comparisons) |
| `table_2026-10-18/` (its own `README.md`) | 2026-10-16 to 2026-10-18, the table chat's checks behind D-054, D-059, D-060 and the Steps 4-5 findings of `specs/SPEC.md` §8: the inputs' scaling, the inversion (classical calibration, its Jensen term, the shortcut), the reported SE against the spline's own error, the phantoms' support along the inversion line, the failure rate's noise, the design effect of realizations per draw, cross-validation folds by draw, what D-060's table estimates, and the bias of Block 12's quadrature. Run from inside that folder. |
| `estimator_2026-10-15/` (its own `README.md`) | 2026-10-15, the evidence behind D-052: whether one profile identifies d when the blur's width and shape are free (impostor searches with Student and skewed generalized-t kernels), Monte Carlo of the fits, the kernel pinned from a thick node, an independent quadrature of the model; `sgt_kernel.py` is the reference implementation of the kernel. Import the workstream by `--repo`; run from inside that folder. |

All numbers in these scripts describe an ideal, index-matched objective at one
wavelength (0.55 um); they are orders of magnitude, not inputs to the bias table.

## Next steps (from the handoff, updated)

**[2026-10-06]** Steps 2-3 are now done the D-021 way. The open choices become
configuration parameters, confirmed in one batch at the start of the
implementation chat. The code is built block by block (blocks 0-11 of the
implementation handoff). Step 1 is Phase II of that plan.

1. Run cell 13 (local registration) on node 4505 and on 5-10 stretches. **[2026-10-07]** Cells 3a-3b of `notebooks/phase2_colab.ipynb`.
2. Settle the open choices: B_bar region (D-018 (a)), mu per node or shared
   (D-019 (a)), sigma_fit, kernel family, grid and N, selection rule.
3. Write `allen_image_diameter.py` with its smoke suite. **[corrected 2026-10-06]**
   Proposed as the package `src/allen_diameter/`, with one smoke test per block.
4. Calibrate the defocus kernel, build the bias table, validate (procedure doc
   sections 3.4-3.10).
5. Apply to 529878215, write the corrected SWC, compare membrane area with
   Allen's, refit Cm.

## Changelog

| Date | Change |
|---|---|
| 2026-10-20 (later) | Block 13.2: the scene and the square, shared by the diagnostics and the measurement (D-064). `plane_iteration.Scene` (the traced segments with Allen's radii at both ends and the soma discs; `Scene.from_branch` gives a phantom's own tube, what a missing scene means), `cell.cell_scene` (a cell's scene, the rule `survey.node_planes` used inline), `plane_iteration.scene_segments` (the segment rows of the block), `cut_square` and `iteration_state` (`node_stack`'s square and state dict); `survey.node_planes` and `scripts/plane_differences.node_stack` now call them, and `node_planes` also returns the scene. No behaviour change. Added `tests/smoke/test_smoke_block13.py` and `checks/block13_2026-10-20/check_13_2_square.py`. Evidence: the new suite 5 pass, 2 skip; 13 of 13 mutants of the new code killed; `check_13_2_square.py` PASS (24 comparisons equal bit for bit against `84ea91f`); the 25 suites pass (156 pass, 0 fail, 23 skip); hashes unchanged **[run, sandbox]**. Spec: `specs/SPEC.md` Block 13, Block 11, section 6. |
| 2026-10-20 | Block 13.1: the Allen-guided iteration (`iterate_planes`, `gradient_iteration`, `refit_at_plane`, `block_background`, `line_profiles` and their helpers) moved from `scripts/plane_differences.py` into the package, `src/allen_diameter/analysis/plane_iteration.py`, with its bodies unchanged; the script imports them, so the diagnostics and the measurement (Block 13.3) share one implementation. Added `docs/TEEG_diameter_table_implementation_plan_2026-10-20.md` (the plan after D-061 to D-067) and `checks/block13_2026-10-20/`. Evidence: the moved definitions identical by syntax tree; `check_13_1_move.py` PASS (16 runs equal bit for bit); the 24 suites pass (151 pass, 0 fail, 21 skip, as at `84ea91f`); hashes unchanged **[run, sandbox]**. Spec: `specs/SPEC.md` Block 13 (entered), Block 11, section 4. |
| 2026-10-19 | End of the table chat. Added `docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` (the next chat implements Blocks 13-14), `checks/table_2026-10-18/` (the table chat's checks with their logs) and `docs/TEEG_thin_plate_spline_in_3D/` (the study document of 2026-10-18). Recorded here late: Block 12, the skewed generalized-t kernel and the seven-parameter fit at a plane (`bee1e0b`, 2026-10-18; D-052.1-2, D-053 (ii), D-054 (iv); `tests/smoke/test_smoke_sgt_fit.py`), and `d643aaf` (SPEC records D-058 to D-060, no code change), neither of which had a row. No code changed. Evidence: the 24 suites at `d643aaf` and seven of the checks re-run from this folder's copies on 2026-10-19 **[run, sandbox]**; the decisions in the project's log. |
| 2026-10-16 | The per-node estimator is settled (D-052, the user's decision of 10:43 with three answers): the Allen-guided plane of D-042/D-044 with each round's line sized from the new fit, and as the fit the skewed generalized-t kernel with its three shape parameters and its width fitted, centre-weighted at lambda = 0.5 um and re-centred by IRLS (D-046). No code changed. Added `docs/TEEG_diameter_correction_table_handoff_2026-10-16.md` (the next chat builds the table) and `checks/estimator_2026-10-15/` (the checks behind D-052, re-run in part from the repository copy, the same output). Evidence: the sandbox runs of 2026-10-15 and their result files; the handoff fact-checked against the code at `9e3296c` and the result files by an independent agent. |
| 2026-10-07 (later, 2) | Plane montages: for chosen nodes, every plane the focus rule scored with Allen's reconstruction drawn on it (traced centre lines and their +-r outline, the stretch in cyan and the other dendrites in amber, faded by depth; the SWC node, the fit's profile line, the fitted edges in plane k*). `survey.node_planes` re-measures a node exactly as the pilot (same request, so from the cache), `plotting.figures.plane_montage` draws it, `scripts/node_planes.py` runs it (ids or `differ`, checked against the pilot CSV). Notebook Cell 4d, runbook v4; checks in `tests/smoke/test_smoke_phase2.py`. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-08 | Consecutive-plane differences (the user's proposal, a diagnostic): around chosen nodes, the positive part of I_(k+1) - I_k between consecutive planes and its mean over the pixels, S+, with S- (the same from the other end of the stack) and the dip of S+ between its two largest maxima. `focus.plane_differences` and `focus.difference_dip` compute it, `plotting.figures.plane_difference_figure` draws it, `scripts/plane_differences.py` runs it on the pilot's blocks (from the cache, planes beyond them fetched). Notebook Cell 4e, runbook v5; checks in `tests/smoke/test_smoke_plane_diff.py`. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-08 (later) | Profile-area evaluation, the default of `scripts/plane_differences.py` (D-036, the user's evaluation): in every plane, the profile along the node's measuring line (bilinear), the area under it and its change between consecutive planes, beside the same areas divided by each plane's background level; `focus.profile_areas`, `plotting.figures.profile_area_figure`. The pixel differences stay as `--evaluation image`. Notebook Cell 4e, runbook v6; checks in `tests/smoke/test_smoke_plane_diff.py`. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-08 (evening) | Entropy evaluation, `scripts/plane_differences.py --evaluation entropy` (D-037, the user's proposal): in every plane, the Shannon entropy of the grey-level histogram (one bin per grey level) of the bilinear samples along the node's measuring line and of the pixels of a strip around it (6 um across the dendrite, 2 um along it), with the dip of each curve framed; `focus.grey_histogram`, `focus.histogram_entropy`, `focus.plane_entropies`, `profiles.stripe_mask`, `plotting.figures.entropy_figure`; writes `planeentropy_*`. Notebook Cell 4f, runbook v7; checks in `tests/smoke/test_smoke_plane_entropy.py`. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-08 (evening, 2) | Line length and entropy pick (D-038 and D-037's annotation): Cells 4e and 4f run over `LINE_HALF_UMS` = [3, 5], the half-length of the measuring line (`--profile-half-um`), each into its own folder; the square around the node grows with the line and is fetched when it outgrows the pilot's block; the entropy frames each curve's global minimum (`--entropy-pick min`; `dip` keeps the earlier rule); the JSON records carry every plane's profile (`v_um`, `profiles`). Runbook v8; checks in `tests/smoke/test_smoke_plane_entropy.py` and `test_smoke_plane_diff.py`. Spec: `specs/SPEC.md` Block 11, section 8. |
| 2026-10-09 | The entropy at +-5 um tested on thin dendrites (D-039): `scripts/plane_differences.py --nodes thin --pilot-csv` picks pilot nodes that are thin (Allen 2r <= 0.6 um, fitted d <= 1.0 um), in S, flat, and where k* and the dip depth agree, up to 3 per stretch and 12 in all; a run over several nodes writes a summary CSV (each pick minus k*) and, for the entropy, a summary figure of small multiples. Notebook Cell 4g, runbook v9; checks in `tests/smoke/test_smoke_plane_entropy.py`. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-09 (afternoon) | `scripts/plane_differences.py --hide-line` (the user's request): the profile and entropy figures draw their planes without the measuring line and the strip's outline, so that the focus can be judged by eye; the frames stay and nothing measured changes. Notebook Cell 4g uses it and shows each node's figure after the summary, runbook v10; checks in `tests/smoke/test_smoke_plane_entropy.py` and `tests/smoke/test_smoke_plane_diff.py`. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-09 (evening) | The gradient energy over the whole line on three lines, and its blend with the strip entropy, across diameters (D-040, the user's request of 16:19): `scripts/plane_differences.py --evaluation gradient` integrates G = B^-2 * integral of (dI/dv)^2 over the whole of each line, +-3 um, +-5 um and the d-line +-m d_hat / 2 sized from the pilot's fitted diameter (`--line-mult`, m = 2), with one background per plane from the block; `--evaluation blend` compares on the d-line G alone, the strip's entropy alone and their min-max blend J = w g + (1 - w) eta, w = 1 / (1 + exp((d_hat - 1.5 um) / 0.3 um)); `--nodes bydiameter --pilot-csv` takes up to 2 pilot nodes per bin of d_hat (edges 0.8, 1.0, 1.5, 2.0, 3.0 um; converged, flat, not crossing) plus the trunks 2 and 3. Per-node figures with bare planes, a summary CSV and a summary figure. Notebook Cells 4h and 4i, runbook v11; Evidence: `test_smoke_plane_gradient.py` 8 pass; 45 of 47 mutants of the new code each failed a check, and the other two are equivalent (a guard that `focus.gradient_energy` repeats, and a mask that the NaN profile rows of the missing planes already apply); every suite re-run (19, no failure); both cells ran on two synthetic cells (tubes of 0.8 and 3.0 um, Allen radius 0.3 um) with pilot tables made from those cells, the fixture in place of Allen's server **[run, sandbox]**; not yet run on Allen's data. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-10 | The blend's entropy on the +-5 um strip (D-041, the user's choice after the first run of Cells 4h and 4i on Allen's data, where the d-line entropy's minimum fell on or next to an end plane on all 11 non-trunk nodes): `scripts/plane_differences.py --evaluation blend` keeps G on the d-line and takes the entropy from the strip +-`--entropy-half-um` (5 um) across, +-1 um along; the blend is written J = w g - (1 - w) h, h the min-max rescaled entropy (0 at its lowest), which has the same maximum as the earlier form; the records carry `h_norm` and `k_Hs` (were `eta`, `k_Hd`); the figures draw the entropy as it is, and the per-node figures wrap their labels. Notebook Cell 4i, runbook v12 (outputs in `pilot/planeblend/bydiameter_strip5um/`); checks in `tests/smoke/test_smoke_plane_gradient.py` (8 pass; 17 of 18 mutants killed, 1 equivalent; 19 suites pass). Spec: `specs/SPEC.md` Block 11. |
| 2026-10-10 (later) | The Allen-guided iteration of the gradient energy (D-042, the user's request of 15:30): `scripts/plane_differences.py --evaluation iterate` integrates G over the whole line, first on the line +-m r_Allen sized from Allen's diameter, then on the line +-m d / 2 sized from Block 5's fit at the plane the previous line picked (the pipeline's own fit with only the plane changed, so that at k* it returns the pilot's d_hat), until a plane repeats or after `--max-rounds` (5) rounds, and reports every round, the last plane (`k_iter`) and the fit there (`d_iter_um`) beside Cell 4h's d-line pick; `plane_differences.iterate_planes`, `refit_at_plane`, `gradient_iteration`, `plotting.figures.iteration_figure`; writes `planeiter_*`. Notebook Cell 4j, runbook v13 (outputs in `pilot/planeiter/bydiameter/`); checks in `tests/smoke/test_smoke_plane_iteration.py` (7 pass; 36 of 36 mutants killed; 20 suites pass); the cell ran on two synthetic cells **[run, sandbox]**; not yet run on Allen's data. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-12 | The Allen-guided iteration at m = 3 (D-044, the user's choice after the run at m = 2 on Allen's data, where all 13 nodes converged and the last plane equalled the Gd pick on every node, D-043): `plane_differences.ITER_LINE_MULT` = 3 is the default of `gradient_iteration` and of `--evaluation iterate`, `D_LINE_MULT` = 2 that of the gradient and blend evaluations (D-040); `run(line_mult=None)` takes each evaluation's own; the comparison line Gd is labelled "same multiplier". Notebook Cell 4j and runbook v14 (outputs in `pilot/planeiter/bydiameter_m3/`). Evidence: `test_smoke_plane_iteration.py` 7 pass (at m = 2 and m = 3) and `test_smoke_plane_gradient.py` 8 pass; 9 of 9 mutants of the defaults each failed a check; every suite re-run (20, no failure); Cell 4j at m = 3 ran on two synthetic cells (tubes of 0.8 and 3.0 um, Allen radius 0.3 um) into `pilot/planeiter/bydiameter_m3/`, ending on the planes and fits of m = 2 with lines +-1.5 d **[run, sandbox]**; not yet run on Allen's data at m = 3. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-12 (later) | The planes the Allen-guided iteration chose, drawn (D-045, the user's request of 17:58): `scripts/iteration_fits.py` reads Cell 4j's `planeiter_<sp>.json`, rebuilds each node's last square and redoes Block 5's fit at `k_iter` (`plane_differences.refit_at_plane(..., with_profile=True)`), and draws the plane with the line that chose it and the fitted edges as a caliper, beside the profiles and the model (`figures.iteration_fit_figure`, `figures.iteration_fits_overview`); the fit must reproduce `d_iter_um` and the square Cell 4j's, or the node's notes say so. Notebook Cell 4k, runbook v15. Evidence: `test_smoke_iteration_fits.py` 6 pass, 1 skip; 16 of 16 mutants of the new code each failed a check; every suite re-run (21, no failure); Cells 4j and 4k ran from the notebook's code on two synthetic cells (tubes of 0.8 and 3.0 um): every node drawn, the fit redone equal to Cell 4j's d_iter, 0 downloads **[run, sandbox]**; not yet run on Allen's data. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-13 | Fit variants at the iteration's plane (D-046, the user's request of 14:50): `model/blur_kernels.py` (scipy kernel families of unit area, FWHM w and peak at 0), `tube_model.model_profile_kernel` ((B3.1) with any kernel), `analysis/fit_variants.py` (the 100 variants: 10 kernels x width fixed or fitted x unweighted or centre-weighted by IRLS re-centred on v0, lambda 0.25-2 um), `scripts/fit_variants.py` (Cell 4l: Block 5's inputs at Cell 4j's `k_iter`, one figure per node, a summary, CSV and JSON), `figures.fit_variants_node_figure`, `figures.fit_variants_summary_figure`; a diagnostic, Block 5 unchanged. Notebook Cell 4l, runbook v16. Evidence: `test_smoke_fit_variants.py` 8 pass; 30 of 30 mutants of the new code each failed a check (26 at the first run; the re-weighting's stopping rule, the warm starts, the re-weighting's start and a skipped record's reason then got checks of their own); every suite re-run (22, no failure); Cells 4j, 4k and 4l ran from the notebook's own code on two synthetic cells (tubes of 0.8 and 3.0 um): 3 of 3 nodes fitted per cell, the baseline equal to Cell 4j's d_iter on 3 of 3, 0 downloads, 25-72 s per node **[run, sandbox]**; not yet run on Allen's data. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-14 | Each fit variant's own Allen-guided iteration (D-047, the user's request of 15:35): `scripts/variant_iteration.py` (Cell 4m) runs Cell 4j's rule once per variant at the fitted width (50), with Cell 4j's m and round limit from its records and its squares, each later line sized from the variant's own fit at the previous plane (`plane_differences.gradient_iteration(..., fit_at=...)`, `fit_variants.PlaneFits` sharing the fits at a plane); Cell 4j's own iteration is redone as the reference and must match its record; figures `figures.variant_stack_figure` (the 50-row edge stack with the planes picked) and `figures.variant_planes_figure` (each variant's plane with the line, the fitted edges and d). Notebook Cell 4m, runbook v17. Evidence: `test_smoke_variant_iteration.py` 6 pass, 1 skip (`test_convergence`: nothing new iterates); 19 of 20 mutants of the new code each failed a check (the 20th, a record's planes with a gap accepted, is refused anyway by the square's own plane check); every suite re-run (23, no failure); Cells 4j to 4m ran from the notebook's own code on two synthetic cells (tubes of 0.8 and 3.0 um): Cell 4j's iteration redone identically on 3 of 3 nodes per cell, every variant ending on Cell 4j's plane, 0 downloads, 18-21 s per node on the 0.8 um tube and 2.4-3.8 min on the 3.0 um tube (other jobs running) **[run, sandbox]**; not yet run on Allen's data. Spec: `specs/SPEC.md` Block 11. |
| 2026-10-07 (later) | The focus rule is the gradient energy of the profile (D-030): `src/allen_diameter/analysis/focus.py`, new suite `tests/smoke/test_smoke_focus.py`; the dip depth of the design handoff's Eq. 1 stays as the labelled comparison `dip_depth`. Runbook v3 and notebook Cells 2, 4b, 4c updated. Spec: `specs/SPEC.md` sections 2.2, 4, 8 and Blocks 1, 5, 11. |
| 2026-10-07 | Listed the Phase II runbook (v2) and the new `notebooks/phase2_colab.ipynb`; pointed next step 1 at its Cells 3a-3b. Added `notebooks/phase2_colab.py`, the same cells as plain Python. |
| 2026-10-06 (later) | Implementation started (branch `sci/diameter-pipeline`): `src/`, `scripts/`, `tests/smoke/` added; root `specs/SPEC.md`; the 2026-09-23 modules imported byte-identical. Decisions D-022 to D-024 in the project log. |
| 2026-10-06 | Added the implementation handoff (D-021) to the contents and status. Corrected the Drive statement: only the 2026-09-22 module versions are there (Drive listing, 2026-10-06). Noted which decision-log copy is live (project_read of both copies, 2026-10-06). Added four check scripts with reference outputs (`*.out`, re-run 2026-10-06, identical to the original runs). Marked next steps 2-3 as handled by D-021. |
