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
| `TEEG_diameter_remeasurement_handoff_2026-10-05.md` | **Start here in a new chat.** Handoff after the documentation chat: status, document map, decisions in force, corrections, open choices, next actions, admin left open (project-knowledge limit, D-020 not yet in the project log). Read after the design handoff below. |
| `handoff_diameter_remeasurement.md` | The design handoff (2026-09-30): data access, Eqs. 1-13, local decisions D1-D7, next actions. The source of the equation numbers "handoff Eq. n". Copied unchanged. |
| `TEEG_diameter_bias_table_procedure_2026-10-04.md` | **How the bias table b(d, phi \| C) is tabulated** (v1.1): configuration, defocus-kernel calibration, phantom rendering (with a walkthrough of one output plane on figure 4), measurement, estimation, inversion, flags, falsification checks. |
| `TEEG_diameter_bias_table_mathematics_2026-10-04.md` | **The mathematics behind it** (v1.1): forward model, Beer-Lambert law, LSF, slabs and slab rendering with the absorbed-light partition, absorbance vs absorbed fraction, squared-width additivity, defocused LSF, calibration identifiability, bias statistics and inversion. |
| `TEEG_microscope_optics_oil_immersion_2026-10-04.md` | **The physics of the microscope** (v1.2, with pointers to figures 1-3): light path, refraction, oil immersion and NA, diffraction and the in-focus blur, forming vs resolving, defocus and why the blur stays flat near focus, condenser and coherence, mounting-medium index mismatch. |
| `TEEG_interactive_figures_spec_2026-10-04.md` | **Instructions for Claude Design to reproduce the four interactive figures**: shared design rules, and per figure the claim, layout, controls, model equations, acceptance values and a paste-ready prompt; known gaps. |
| `TEEG_diameter_optics_notes.md` | Running notes (v2, 2026-10-04) summarising the chat explanations, with corrections marked "[corrected <date>]". |

The three 2026-10-04 documents were independently reviewed and revised the same
day; each ends with a revision record. Decisions D-018 and D-019 are recorded in
the project's decision log (`TEEG_decisions_and_ideas_log.md`, project knowledge).

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

All numbers in these scripts describe an ideal, index-matched objective at one
wavelength (0.55 um); they are orders of magnitude, not inputs to the bias table.

## Next steps (from the handoff, updated)

1. Run cell 13 (local registration) on node 4505 and on 5-10 stretches.
2. Settle the open choices: B_bar region (D-018 (a)), mu per node or shared
   (D-019 (a)), sigma_fit, kernel family, grid and N, selection rule.
3. Write `allen_image_diameter.py` with its smoke suite.
4. Calibrate the defocus kernel, build the bias table, validate (procedure doc
   sections 3.4-3.10).
5. Apply to 529878215, write the corrected SWC, compare membrane area with
   Allen's, refit Cm.
