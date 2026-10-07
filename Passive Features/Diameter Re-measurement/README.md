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
`sci/diameter-pipeline` (decisions D-021 to D-024). Done: Block 0 (the six
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
| `TEEG_diameter_phase2_colab_runbook_2026-10-06.md` | **The real-data steps in Colab** (v2, 2026-10-07): Cells 0-10 in order (bootstrap, node list, registration, pilot, camera inputs, radius distribution, kernel calibration, production configuration, the davinci table, apply to the cell), each with the line that must appear and the decisions it feeds. [added 2026-10-07: the runbook was not listed here] |

### `notebooks/`

| File | What it is |
|---|---|
| `phase2_colab.ipynb` | The runbook's cells as a Colab notebook (2026-10-07), same numbers, same code. Open it in Colab from GitHub (File > Open notebook > GitHub, branch `sci/diameter-pipeline`). |

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
| 2026-10-07 | Listed the Phase II runbook (v2) and the new `notebooks/phase2_colab.ipynb`; pointed next step 1 at its Cells 3a-3b. |
| 2026-10-06 (later) | Implementation started (branch `sci/diameter-pipeline`): `src/`, `scripts/`, `tests/smoke/` added; root `specs/SPEC.md`; the 2026-09-23 modules imported byte-identical. Decisions D-022 to D-024 in the project log. |
| 2026-10-06 | Added the implementation handoff (D-021) to the contents and status. Corrected the Drive statement: only the 2026-09-22 module versions are there (Drive listing, 2026-10-06). Noted which decision-log copy is live (project_read of both copies, 2026-10-06). Added four check scripts with reference outputs (`*.out`, re-run 2026-10-06, identical to the original runs). Marked next steps 2-3 as handled by D-021. |
