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

**Status (2026-10-06):** still nothing coded. By D-021 [user, 2026-10-06 14:30]
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
| `TEEG_diameter_theory_handoff_2026-10-07.md` | **Start here for a new theory chat** (written 2026-10-07 for 2026-10-08): the start prompt, the split of the two chats, decisions D-020 to D-032 in one line each, the findings the theory rests on, the open work in order, admin notes (how to merge into the large project log, GitHub push errors). |
| `TEEG_diameter_study_notes_2026-10-07.md` | Study notes the user asked for: the explanation of D-030 and of what happens near optical depth 1 (partition vs ray world, weak-object approximation), and the answers on D-031, D-032 and $\hat\tau$; with the figure `TEEG_ray_world_vs_partition_2026-10-07.svg`. |
| `TEEG_ray_world_vs_partition_2026-10-07.svg` | Figure: one oblique ray, one slice; the ray world reduces the light by the stain along the ray, the partition by the stain straight below the crossing point. |
| `TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md` | **The two renderers and the hybrid** (2026-10-09, at the user's request; independently reviewed before commit): how the partition and the ray world differ (one formula, two substitutions), what each gets right and wrong for thick and dark dendrites, why a comparison must share the blur, and the user's hybrid (ray-world absorption, kernel blur): its formula, why it must average over directions, how well it does, what is untested. |
| `TEEG_diameter_implementation_handoff_2026-10-06.md` | **Start here to implement the pipeline (D-021).** Contents: <br>- how the theory chat and the implementation chat split the work; <br>- decisions in force; <br>- the pipeline flow (real-data path, synthetic path, kernel calibration, cell level); <br>- the proposed code layout and Colab bootstrap; <br>- the block plan with smoke oracles; <br>- equations (S1)-(S10), which are written nowhere else; <br>- every open choice as a configuration parameter with a provisional default; <br>- the theory chat's findings (partition artefacts for dark tubes, kernel far field, phantom length, heading); <br>- validation, known gaps, first actions. |
| `TEEG_decision_log_additions_2026-10-06.md` | **Pending decision-log entries** D-025 (the stacks' noise measured on clean background patches), D-026 (the kernel beyond ±0.84 µm calibrated on dendrites just under 0.8 µm, $\gamma$ measured, Gaussian kept) idea I-002 (a non-Gaussian kernel family) and D-027 (calibration first on flat branches; tilted branches later), with the anchors where they go in the project log. They wait here because the project log could not be written (project knowledge full, 2026-10-06). **[corrected 2026-10-07]** Merged into the project log on 2026-10-07; this file keeps the text as written. |
| `TEEG_diameter_remeasurement_handoff_2026-10-05.md` | **Start here in a new chat.** (For the method and documents. For code, start from the 2026-10-06 implementation handoff above.) **[corrected 2026-10-07]** For a new theory chat, start from the 2026-10-07 theory handoff above; this file stays the map of the documentation chat. Handoff after the documentation chat: status, document map, decisions in force, corrections, open choices, next actions, admin left open (project-knowledge limit, D-020 not yet in the project log). Read after the design handoff below. |
| `handoff_diameter_remeasurement.md` | The design handoff (2026-09-30): data access, Eqs. 1-13, local decisions D1-D7, next actions. The source of the equation numbers "handoff Eq. n". Copied unchanged. |
| `TEEG_diameter_bias_table_procedure_2026-10-04.md` | **How the bias table b(d, phi \| C) is tabulated** (v1.1): configuration, defocus-kernel calibration, phantom rendering (with a walkthrough of one output plane on figure 4), measurement, estimation, inversion, flags, falsification checks. |
| `TEEG_diameter_bias_table_mathematics_2026-10-04.md` | **The mathematics behind it** (v1.1): forward model, Beer-Lambert law, LSF, slabs and slab rendering with the absorbed-light partition, absorbance vs absorbed fraction, squared-width additivity, defocused LSF, calibration identifiability, bias statistics and inversion. |
| `TEEG_microscope_optics_oil_immersion_2026-10-04.md` | **The physics of the microscope** (v1.2, with pointers to figures 1-3): light path, refraction, oil immersion and NA, diffraction and the in-focus blur, forming vs resolving, defocus and why the blur stays flat near focus, condenser and coherence, mounting-medium index mismatch. |
| `TEEG_interactive_figures_spec_2026-10-04.md` | **Instructions for Claude Design to reproduce the four interactive figures**: shared design rules, and per figure the claim, layout, controls, model equations, acceptance values and a paste-ready prompt; known gaps. |
| `TEEG_diameter_optics_notes.md` | Running notes (v2, 2026-10-04) summarising the chat explanations, with corrections marked "[corrected <date>]". |

The three 2026-10-04 documents were independently reviewed and revised the same
day; each ends with a revision record. Decisions D-018 and D-019 are recorded in
the project's decision log (`TEEG_decisions_and_ideas_log.md`, project knowledge).
That is the root-level copy; `claude/TEEG_decisions_and_ideas_log.md` is an older
copy with D-001 and I-001 only. D-020 (above) and D-021 (implementation handoff) are not yet
in the log, because project knowledge is at its size limit.
**[corrected 2026-10-06]** D-020 to D-024 are now in the root log, and the
`claude/` copy was deleted with the user's OK. D-025, D-026 and I-002 wait in
`docs/TEEG_decision_log_additions_2026-10-06.md`: the project log could not be
written again (project knowledge full). **[corrected 2026-10-07]** D-025 to
D-027 and I-002 are now in the root log too (merged 2026-10-07).

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
python calib_reach_check.py                          # imports optics_points_check (seconds)
python calib_width_check.py                          # imports optics_points_check (seconds)
python coherence_check.py                            # standalone (~4 min)
python kernel_confound_check.py                      # imports optics_points_check (~1 min)
python hybrid_absorption_check.py                    # imports optics_points_check (~30 s)
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
| `calib_reach_check.py` (+ `.out`) | 2026-10-06, D-026 comments (b), (c), (h): <br>- how far a faint 0.8 µm (and 0.5 µm) node keeps a centre dip as deep as a 0.3 µm node's 3 planes out (far slopes 0.79 and 1.21); <br>- the tilt range a calibrated depth range covers at $U$ = 10 µm, by (S5); <br>- the depth-extent constant $\gamma^2 r^2/4$ and the axis-error/anchor confound, checked numerically. |
| `calib_width_check.py` (+ `.out`) | 2026-10-06, D-026 comment (a) and D-027: <br>- why a windowed second moment depends on the window ($\lvert v\rvert^{-2}$ tails; a background offset weighted by $v^2$); <br>- how tilt changes the second moment and the core width of a thin node's profile with a depth-only kernel. |
| `coherence_check.py` (+ `.out`) | 2026-10-08, theory handoff Open work 2 (weak-object approximation): a straight branch in focus, thin object, scalar optics, imaged coherently (one plane wave), by the sum over condenser directions at $S=1$, and by $B\,(T*h_0)$; smoke checks for the empty field, the weak-object limit and the sign $I_{S=1}\le I_{\rm model}\le B$; reports dip depth, bright fringe and FWHM for $d$ = 0.5 and 1 µm at $\mu d$ = 0.1, 1, 3. |
| `kernel_confound_check.py` (+ `.out`) | 2026-10-09, theory handoff Open work 1 (the D-024 (i) study): the partition with the production-like Gaussian Debye-table kernel vs the ray world, same fit, same true $\mu$, no matching. The gap in $\hat d/d$ is set by the blur, about −0.11 for $d=1$ µm flat at every $\mu d$ from 0.05 to 3 and −0.25 at $d=0.5$ µm, 20°, so an unmatched comparison cannot see the absorption treatment (compare `optics_points_check.py matched`). Extended the same day with thick flat tubes: −0.058 / −0.048 at $d=2$ µm and −0.050 / −0.040 at $d=3$ µm ($\mu d$ = 0.5 / 3), the size of what leaving out diffraction costs a ray-world table for thick nodes. |
| `hybrid_absorption_check.py` (+ `.out`) | 2026-10-09, the user's hybrid proposal (theory handoff, Open work 1): absorbed light booked per slice point with the history along the rays (NA 1.4, $n_{\rm oil}$ 1.515), averaged over the directions arriving there, then spread by a kernel. It is tested in the ray world with the cone's own $W^*$-weighted kernel, where the answer is exact. Centre dip of a flat 1 µm tube against the ray world: 0.057 / 0.057, 0.450 / 0.441, 0.845 / 0.820, 0.961 / 0.965 at $\mu d$ = 0.05 / 0.5 / 1.5 / 3, but 0.811 / 1.000 and 0.732 / 1.000 at $\mu d$ = 10 / 50: it leaks near the opaque limit, as the partition does (0.736, 0.699). Extended later the same day for the renderers document (§3.8–3.9): the hybrid books the light where each direction enters the tube (10.3 % of it above the axis depth at $\mu d=50$, 26.8 % at 3), and its total equals the ray world's total shadow (1.114 / 1.114 µm per unit length at $\mu d=3$, 1.240 / 1.238 at 50), so the leak moves light without losing it; the opaque leak (S9) at the centre is 0.691 under $W^*$ and 0.737 under $W$; the cone's per-axis RMS slope is 0.7915 ($W$) / 0.8991 ($W^*$), and its characteristic function has zeros, so a Gaussian kernel cannot be the cone plus a per-ray blur. The same-$\mu$ partition and true-history columns are labelled P\*s and Gv\*s (earlier rows unchanged). |

All numbers in these scripts describe an ideal, index-matched objective at one
wavelength (0.55 um); they are orders of magnitude, not inputs to the bias table.

## Next steps (from the handoff, updated)

**[2026-10-06]** Steps 2-3 are now done the D-021 way. The open choices become
configuration parameters, confirmed in one batch at the start of the
implementation chat. The code is built block by block (blocks 0-11 of the
implementation handoff). Step 1 is Phase II of that plan.

1. Run cell 13 (local registration) on node 4505 and on 5-10 stretches.
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
| 2026-10-06 | Added the implementation handoff (D-021) to the contents and status. Corrected the Drive statement: only the 2026-09-22 module versions are there (Drive listing, 2026-10-06). Noted which decision-log copy is live (project_read of both copies, 2026-10-06). Added four check scripts with reference outputs (`*.out`, re-run 2026-10-06, identical to the original runs). Marked next steps 2-3 as handled by D-021. |
| 2026-10-06 (later, 2) | Theory chat: added `docs/TEEG_decision_log_additions_2026-10-06.md` (D-025, D-026, I-002, pending because the project log could not be written) and `checks/calib_reach_check.py` with its reference output (run 2026-10-06); corrected the decision-log paragraph under "Contents" (project_read of the log, 2026-10-06). The implementation handoff is now v1.2. |
| 2026-10-06 (later, 3) | Theory chat: D-027 added to `docs/TEEG_decision_log_additions_2026-10-06.md` (still pending for the project log); new `checks/calib_width_check.py` with its reference output; `calib_reach_check.py` extended to 0.5 µm nodes and re-run. The implementation handoff is now v1.3. |
| 2026-10-07 | Theory chat: D-025 to D-027 and I-002 merged into the project log; the statements calling them pending, here, in the additions file and in the implementation handoff (now v1.4), marked [corrected 2026-10-07]. |
| 2026-10-07 (later) | Theory chat: D-030 (every real node gets its own corrected diameter whatever its optical depth; the dark flag becomes a label) is in the project log; the implementation handoff is now v1.5. |
| 2026-10-07 (later, 2) | Theory chat: D-031 (the bias table gains the estimated optical depth $\hat\alpha$ as a third axis) and D-032 (tilt-specific tables dropped, as kernel and as test) are in the project log; the implementation handoff is now v1.6. |
| 2026-10-07 (later, 3) | Theory chat, at the user's request: new `docs/TEEG_diameter_theory_handoff_2026-10-07.md` (start of a new theory chat on 2026-10-08, with its start prompt), `docs/TEEG_diameter_study_notes_2026-10-07.md` and the figure `docs/TEEG_ray_world_vs_partition_2026-10-07.svg`; contents rows added, the 2026-10-05 handoff's "start here" marked. |
| 2026-10-08 | New theory chat, after explaining the weak-object approximation: the optics document is v1.3 (§3.6 gains a full-text source, Zuo et al. 2017, and the exact $S=1$ difference between the incoherent form and the true image; its fringe check is marked [corrected 2026-10-08]: it tests $S<1$, not the dark-node error at $S=1$); the theory handoff (Open work 2, next free IDs D-038 and I-003) and the study notes carry dated notes. |
| 2026-10-08 (later) | Previous theory chat's session, answering the user on coherent vs incoherent illumination: new `checks/coherence_check.py` with its reference output (run 2026-10-08); the theory handoff is v1.2 (a dated note under Open work 2 with its numbers). |
| 2026-10-09 | Theory chat, after the user asked what the ray world is used for. New `checks/kernel_confound_check.py` with its reference output (run 2026-10-09). The theory handoff is v1.3: Open work 1 now states the conditions of a like-for-like comparison with the ray world, and the next free IDs are D-039, I-003. The implementation handoff is v1.7, with a note on its end-to-end validation row. The study notes carry one bracketed note. |
| 2026-10-09 (later) | Theory chat, answering whether the ray world could build the table for thick or dark dendrites: `checks/kernel_confound_check.py` gains thick flat tubes ($d$ = 2, 3 µm; earlier rows unchanged, re-run); the theory handoff is v1.4 (one sentence under Open work 1). |
| 2026-10-09 (later, 2) | Theory chat, on the user's hybrid proposal: new `checks/hybrid_absorption_check.py` with its reference output (run 2026-10-09); the theory handoff is v1.5 (a dated note under Open work 1 qualifying the v1.4 sentence). |
| 2026-10-09 (later, 3) | Theory chat, at the user's request: new `docs/TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md` (the two renderers and the hybrid), checked before commit by an independent reviewer whose corrections are built in; contents row added. `checks/hybrid_absorption_check.py` extended and re-run (booking location, conservation, the leak under both weightings, the kernel split; same-$\mu$ columns relabelled; earlier rows unchanged). The theory handoff is v1.6: study material; [corrected 2026-10-09] marks on the kernel–cone width range and on the hybrid's booking; next free IDs D-041, I-003. |
