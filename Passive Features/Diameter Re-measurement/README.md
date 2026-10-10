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
| `TEEG_diameter_implementation_handoff_2026-10-06.md` | **Start here to implement the pipeline (D-021).** Contents: <br>- how the theory chat and the implementation chat split the work; <br>- decisions in force; <br>- the pipeline flow (real-data path, synthetic path, kernel calibration, cell level); <br>- the proposed code layout and Colab bootstrap; <br>- the block plan with smoke oracles; <br>- equations (S1)-(S10), which are written nowhere else; <br>- every open choice as a configuration parameter with a provisional default; <br>- the theory chat's findings (partition artefacts for dark tubes, kernel far field, phantom length, heading); <br>- validation, known gaps, first actions. |
| `TEEG_diameter_remeasurement_handoff_2026-10-05.md` | **Start here in a new chat.** (For the method and documents. For code, start from the 2026-10-06 implementation handoff above.) Handoff after the documentation chat: status, document map, decisions in force, corrections, open choices, next actions, admin left open (project-knowledge limit, D-020 not yet in the project log). Read after the design handoff below. |
| `handoff_diameter_remeasurement.md` | The design handoff (2026-09-30): data access, Eqs. 1-13, local decisions D1-D7, next actions. The source of the equation numbers "handoff Eq. n". Copied unchanged. |
| `TEEG_diameter_bias_table_procedure_2026-10-04.md` | **How the bias table b(d, phi \| C) is tabulated** (v1.1): configuration, defocus-kernel calibration, phantom rendering (with a walkthrough of one output plane on figure 4), measurement, estimation, inversion, flags, falsification checks. |
| `TEEG_diameter_bias_table_mathematics_2026-10-04.md` | **The mathematics behind it** (v1.1): forward model, Beer-Lambert law, LSF, slabs and slab rendering with the absorbed-light partition, absorbance vs absorbed fraction, squared-width additivity, defocused LSF, calibration identifiability, bias statistics and inversion. |
| `TEEG_microscope_optics_oil_immersion_2026-10-04.md` | **The physics of the microscope** (v1.2, with pointers to figures 1-3): light path, refraction, oil immersion and NA, diffraction and the in-focus blur, forming vs resolving, defocus and why the blur stays flat near focus, condenser and coherence, mounting-medium index mismatch. |
| `TEEG_interactive_figures_spec_2026-10-04.md` | **Instructions for Claude Design to reproduce the four interactive figures**: shared design rules, and per figure the claim, layout, controls, model equations, acceptance values and a paste-ready prompt; known gaps. |
| `TEEG_diameter_optics_notes.md` | Running notes (v2, 2026-10-04) summarising the chat explanations, with corrections marked "[corrected <date>]". |
| `TEEG_diameter_phase2_colab_runbook_2026-10-06.md` | **The real-data steps in Colab** (v13, 2026-10-10) [corrected 2026-10-08: the heading said v5 after v6]: Cells 0-10 in order (bootstrap, node list, registration, pilot, the plane montages of chosen nodes with Allen's trace drawn on them (Cell 4d), the plane-to-plane evaluations around chosen nodes (Cells 4e-4g), the focus scores compared across diameters (Cells 4h and 4i; Cell 4j, the line sized from Allen's diameter and then from the fit [added 2026-10-10, later]), camera inputs, radius distribution, kernel calibration, production configuration, the davinci table, apply to the cell), each with the line that must appear and the decisions it feeds. [added 2026-10-07: the runbook was not listed here] [updated 2026-10-07: v3, the focus rule of D-030; v4, Cell 4d] [updated 2026-10-08: v5, Cell 4e; v6, Cell 4e's profile-area evaluation; v7, Cell 4f, the entropy evaluation; v8, Cells 4e and 4f at two line lengths] [updated 2026-10-09: v9, Cell 4g, the entropy on thin dendrites; v10, Cell 4g shows each node's figure, its planes drawn without the measuring line; v11, Cells 4h and 4i, the gradient energy over the whole line on three lines and its sigmoid-weighted blend with the strip entropy, across diameters (D-040)] [updated 2026-10-10: v12, Cell 4i takes the entropy from the +-5 um strip and writes the blend J = w g - (1 - w) h (D-041); v13, Cell 4j, the gradient energy on lines sized from Allen's diameter and then from the fit at the plane each picks, until the plane repeats (D-042)] |

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
| 2026-10-07 (later) | The focus rule is the gradient energy of the profile (D-030): `src/allen_diameter/analysis/focus.py`, new suite `tests/smoke/test_smoke_focus.py`; the dip depth of the design handoff's Eq. 1 stays as the labelled comparison `dip_depth`. Runbook v3 and notebook Cells 2, 4b, 4c updated. Spec: `specs/SPEC.md` sections 2.2, 4, 8 and Blocks 1, 5, 11. |
| 2026-10-07 | Listed the Phase II runbook (v2) and the new `notebooks/phase2_colab.ipynb`; pointed next step 1 at its Cells 3a-3b. Added `notebooks/phase2_colab.py`, the same cells as plain Python. |
| 2026-10-06 (later) | Implementation started (branch `sci/diameter-pipeline`): `src/`, `scripts/`, `tests/smoke/` added; root `specs/SPEC.md`; the 2026-09-23 modules imported byte-identical. Decisions D-022 to D-024 in the project log. |
| 2026-10-06 | Added the implementation handoff (D-021) to the contents and status. Corrected the Drive statement: only the 2026-09-22 module versions are there (Drive listing, 2026-10-06). Noted which decision-log copy is live (project_read of both copies, 2026-10-06). Added four check scripts with reference outputs (`*.out`, re-run 2026-10-06, identical to the original runs). Marked next steps 2-3 as handled by D-021. |
