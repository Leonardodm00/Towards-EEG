# Handoff: diameter re-measurement, theory chat (2026-10-07 -> new chat on 2026-10-08)

| Date | Change |
|---|---|
| 2026-10-07 | v1. Written at the user's request (18:17 Europe/Rome) so that a new theory chat can take over on 2026-10-08 from the theory chat of 2026-10-05 to 2026-10-07 (claude.ai session `session_01VvzYj2wa7J3oP8EEw71KEg`). State after D-032. Evidence: the project log re-read and written on 2026-10-07 (D-025 to D-027 merged; D-030, D-031, D-032 added); `main` at `ff1028f` before this file; numbers below re-read today from the committed `checks/*.out` and the implementation handoff's "Findings". |
| 2026-10-08 | v1.1, by the new theory chat (session `session_017oQJ14njZ9i5nSBhHUDoMh`), after its first answer (the weak-object approximation). Open work 2 gains a dated note: the literature search widened (one full-text source), the exact $S=1$ difference between the incoherent form and the true image, and a correction of what the fringe check tests. "Next free IDs" marked [corrected 2026-10-08]. Evidence: the project log read 2026-10-08 (`project_read`, D-019 to D-037, I-002); Zuo et al. 2017, PubMed full text (PMC5550517); derivation in the chat **[reasoning, not run]**. |
| 2026-10-08 (later) | v1.2, by the previous theory chat's session (`session_01VvzYj2wa7J3oP8EEw71KEg`), answering the user on coherent vs incoherent illumination. Open work 2 gains a dated note with the numbers of the new `checks/coherence_check.py`: the $S=1$ image is never brighter than $B\,(T*h_0)$ (the sign derived in v1.1, confirmed numerically) and the in-focus size of the difference for $d$ = 0.5 and 1 µm at $\mu d$ = 0.1, 1, 3. Evidence: the script and its reference output, run 2026-10-08 **[run]**. |
| 2026-10-09 | v1.3, by the theory chat of `session_017oQJ14njZ9i5nSBhHUDoMh`, after the user asked what the ray world is used for. Open work 1 gains a dated note: the three conditions of a like-for-like comparison, which the branch cannot meet yet (no cone kernel), the size of the blur confound without them, and the evidential strength of "no $\mu_{\rm ph}$ reaches $\hat\mu$". "Next free IDs" marked [corrected 2026-10-09]. Evidence: `checks/optics_points_check.py` and `.out` read; the new `checks/kernel_confound_check.py` (+ `.out`) **[run]**; `sci/diameter-pipeline` at `3084991` read (`model/ray_world.py`, `config.py`, `scripts/end_to_end.py`, `specs/SPEC.md` Block 9 and open questions) **[src]**; the project log read 2026-10-09 (D-038 added by the implementation chat). |
| 2026-10-09 (later) | v1.4, same chat, after the user asked whether the ray world could build the table for thick or heavily stained dendrites. One sentence added to the Open work 1 note: what leaving out diffraction costs a ray-world table for thick flat tubes. Evidence: `checks/kernel_confound_check.py` extended to $d$ = 2, 3 µm **[run]**. |
| 2026-10-09 (later, 2) | v1.5, same chat. The user proposed the hybrid explicitly: ray-world absorption (NA 1.4, $n_{\rm oil}$ 1.515), each image plane rendered with the kernel. A dated note under Open work 1 records how far it goes, and qualifies the v1.4 sentence "supports the hybrid". Evidence: new `checks/hybrid_absorption_check.py` (+ `.out`) **[run]**. |
| 2026-10-09 (later, 3) | v1.6, same chat, after writing the renderers document at the user's request. "Study material" lists it. Two statements of the v1.5 note are marked [corrected 2026-10-09]: the production kernel is narrower than the cone beyond about 0.1 µm of defocus, not "near focus"; and the hybrid books each direction's light where that direction enters the tube, so its near-opaque factor is like (S9)'s but not (S9)'s, and its leak moves light without losing it. "Next free IDs" and the implementation handoff's version are marked [corrected 2026-10-09]. Evidence: `docs/TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md` §3.6, §3.8–3.9, independently reviewed; `checks/hybrid_absorption_check.py` extended and re-run (+ `.out`) **[run]**; the project log read 2026-10-09 (D-039, D-040 added by the diameter chat). |

Paths are relative to `Passive Features/Diameter Re-measurement/` in the repo
`Leonardodm00/Towards-EEG` (canonical URL
`https://github.com/Leonardodm00/Towards-EEG.git`; the lowercase URL
redirects). "The log" = the project-knowledge file
`TEEG_decisions_and_ideas_log.md` (root). "Implementation handoff" =
`docs/TEEG_diameter_implementation_handoff_2026-10-06.md` (v1.6). **[corrected
2026-10-09]** v1.7 since 2026-10-09.

## Start prompt for the new chat

Paste this as the first message of the new chat (the project "Towards EEG"
must be attached):

```
New theory chat for the diameter re-measurement workstream. Clone
https://github.com/Leonardodm00/Towards-EEG.git and read, on main,
"Passive Features/Diameter Re-measurement/docs/TEEG_diameter_theory_handoff_2026-10-07.md".
It hands over from the previous theory chat (claude.ai session
session_01VvzYj2wa7J3oP8EEw71KEg). Then read the project log
TEEG_decisions_and_ideas_log.md (project knowledge) from D-019 to D-032 and
I-002, and the implementation handoff v1.6 in the same docs folder.
You are the theory chat of D-021: you study, explain and record decisions;
the code is written in the implementation chat on branch sci/diameter-pipeline.
Follow the project instructions (explanation modes v8, source priority,
document upkeep); diameter docs live in the repo docs folder on main (D-020).
Start by telling me in three lines where we stand and listing the items of
the handoff's "Open work" section, then wait for my first question.
```

## How the work is split (D-021)

| Chat | Does | Where its output goes |
|---|---|---|
| theory (this one, continued) | studies the method, explains it, records the user's decisions | the log (decisions); `docs/` on `main` (D-020) |
| implementation | writes the pipeline code, block by block | branch `sci/diameter-pipeline` (last seen at `e908969`, 2026-10-06; not re-checked since) |

The implementation chat takes its instructions from the implementation
handoff, which the theory chat keeps current (v1.6 carries D-030 to D-032).

## Decisions of this workstream, D-020 to D-032

Full text in the log. One line each; the open points are the ones still
waiting for the user.

| ID | One line | Open |
|---|---|---|
| D-020 | diameter docs live in `docs/` on `main`, updated by commit and push | — |
| D-021 | two chats (theory, implementation); open choices are provisional config defaults, confirmed in batches | — |
| D-022 | code in this folder on `sci/diameter-pipeline`; root `specs/SPEC.md`; production table on davinci (PBS) | smoke-test layout; PBS array layout |
| D-023 | $\mu$ per node; $\bar B_i$ over the block with the traced path masked; $\sigma_{\rm fit}$ a study axis {0.080, 0.099, 0.125} µm, deliverable 0.099 | the study set |
| D-024 | partition renderer and linear kernel continuation $\gamma=0.79$ "for now"; random phantom design over ranges, $U=10$ µm, $\varphi\in[0°,90°)$; every node estimated whatever its tilt; thin-plate spline; D5, D7 confirmed; jitter 0 until cell 13 | (i) the study of the partition's failure; $d_{\max}$, $\mu$ range, replicates, jitter |
| D-025 | the stacks' noise is measured on clean background patches (every plane, several positions, several stacks): second moment first, then its distribution | comments (a)-(f) |
| D-026 | beyond ±0.84 µm the kernel is calibrated on dendrites just under 0.8 µm (Allen diameter), read beyond 0.84 µm; $\gamma$ fitted; Gaussian kept for now | comments (a)-(h) |
| D-027 | the calibration starts with flat branches only, near and far from focus | tilt tolerance; second half withdrawn by D-032 (reading to confirm) |
| D-028, D-029 | Ih Fit workstream (sample autocorrelation), not this one | — |
| D-030 | every real node gets its own corrected diameter whatever its optical depth; $\hat\alpha\le1.0$ leaves $\mathcal S$; `dark` is only a label | comments (a), (c) |
| D-031 | the bias table also accounts for the estimated optical depth: $\hat m(d,\varphi,\hat\alpha\mid\mathcal C)$, inverted at each node's own $\hat\alpha_i$ | axis scaling; $\hat\alpha$ or $\log\hat\alpha$; replicate budget; nodes outside the phantoms' $\hat\alpha$ range |
| D-032 | tilt-specific tables are dropped, as kernel and as test | does it also withdraw D-027's tilt-binned study? (the theory chat's reading: yes; not yet confirmed) |
| I-002 | idea: a non-Gaussian kernel family with the Gaussian as a special case | to discuss |

Next free IDs in the log: D-033, I-003. **[corrected 2026-10-08]** The log
now also holds D-033 and D-034 (Stage 7 noise, another workstream) and D-035
to D-037 (this workstream's implementation chat, logged 2026-10-08: D-035 the
focus plane by gradient energy, which the repository cites as "D-030"; D-036
and D-037 plane-to-plane diagnostics). Next free IDs: D-038, I-003.
**[corrected 2026-10-09]** D-038 (the measuring line enlarged beyond
±3 µm, to ±4 or ±5 µm) was logged by the implementation chat on 2026-10-08.
Next free IDs: D-039, I-003.
**[corrected 2026-10-09, later]** D-039 (14:30, the entropy at ±5 µm carried
forward and tested on thin dendrites) and D-040 (16:19, the gradient energy on
three line lengths and its blend with the strip entropy) were logged by the
diameter chat on 2026-10-09. Next free IDs: D-041, I-003.

## Findings the theory rests on

All **[run]** with the illustrative ideal-Debye kernel or the ray world; no
Allen data. Sources: implementation handoff "Findings"; `checks/*.out`.

| Topic | Result | Script |
|---|---|---|
| Partition fails for dark thick tubes | flat $d=1$ µm, Debye kernel: in-focus centre dip 0.831 / 0.788 / 0.723 at $\mu d$ = 3 / 10 / 50 (lighter as darker); focus peaks at plane −1 ($d=1$ µm, $\mu d=1.5$) and −3 ($d=2$ µm, $\mu d=2$), at 0 for faint stain; ray world symmetric, opaque centre black (1.000 at $\mu d=50$; partition 0.699 vs 0.691 from (S9)); no phantom $\mu$ reaches the reference's $\hat\mu$ at $\mu d$ = 1.5 or 3, one does at $\mu d\le0.5$ | `followup_checks.py`, `focus_scan.py`, `optics_points_check.py` |
| Why | the partition computes the history $T_{<j}$ along the vertical line through the crossing point instead of along the ray, (S7)/(S8); figure `docs/TEEG_ray_world_vs_partition_2026-10-07.svg` | — |
| Faint stain is fine | $\hat d/d$ partition vs ray world: −0.002 at $\mu d=0.05$, −0.005 at 0.5; vertical path elements only rescale $\mu$ ($\mu_{\rm ph}/\mu$ = 1.450 vs 1.447 from (S10)); with the true history but vertical path elements, +0.009 / +0.021 / +0.050 at $\mu d$ = 0.5 / 1.5 / 3 | `optics_points_check.py` |
| Beer–Lambert near $\alpha\approx1$ | central ray removes $1-e^{-\alpha}$: 0.39 at 0.5 (21 % below linear), 0.63 at 1; far side receives $e^{-\alpha}$: 0.61, 0.37 | arithmetic, 2026-10-07 |
| Width statistic | the full second moment of a real LSF is infinite ($\lvert v\rvert^{-2}$ tails); Gaussian-core width is the statistic (procedure §3.4) | `psf_check.py`, `doc_checks.py`, `calib_width_check.py` part 1 |
| Tilt (depth-only kernel, thin node) | $\sqrt{m_2}$ over ±5$\sigma_{\rm r}$: +4–5 % at 10°, +17–27 % at 20°, +32–59 % at 30°; core width within 1 / 3 / 6 %; one profile, two windows: $\sqrt{m_2}$ ratio 1.32 vs 1.78, core 1.054 | `calib_width_check.py` parts 2-3 |
| Reach of faint thick nodes | 0.8 µm, $\mu=0.6$ µm⁻¹: to plane 16 (4.48 µm) with $\gamma=0.79$, plane 11 (3.08 µm) with 1.21; $\mu=0.3$: planes 9, 7 | `calib_reach_check.py` |

## Open work for the theory chat

In the order the theory chat would take it; the user decides.

1. **D-024 (i), now on the critical path** (D-030 puts dark nodes in the
   deliverable). Render the same phantom draws with the partition and with
   `ray_world`, fit both, and report $\Delta\hat d/d$ against $\hat\alpha$
   (differences only: the ray world has no diffraction). Decide whether dark
   nodes need another renderer, e.g. the history computed along the rays.
   **[2026-10-09, theory chat; the user asked what the ray world is used
   for]** "Differences only" is not enough. A like-for-like comparison needs
   three conditions, all met by `checks/optics_points_check.py matched` (its
   P*): (a) the partition is rendered with the kernel of the ray world's own
   rays, (b) weighted by $1/\cos\vartheta$, which is what a faint-node
   calibration would deliver in a ray-world microscope, and (c) $\mu$ matched
   through $\hat\mu$ (procedure §3.5). Then the two renderers agree to first
   order in the absorbance for any geometry, and what differs is the
   bookkeeping alone (history along the vertical; vertical path element)
   **[reasoning; run: −0.002 / −0.005 at $\mu d$ = 0.05 / 0.5]**. The branch
   cannot meet (a)-(b) yet: its ray world (Block 9) has no kernel, the
   partition has only the Gaussian table, and `specs/SPEC.md` lists the cone
   kernel as not built (Block 9) **[src, `3084991`]**. Without the
   conditions the comparison measures blur: with the Gaussian Debye-table
   kernel, the partition's $\hat d/d$ minus the ray world's is −0.108 /
   −0.114 / −0.112 / −0.106 at $\mu d$ = 0.05 / 0.5 / 1.5 / 3 ($d=1$ µm,
   flat) and −0.254 at $d=0.5$ µm, 20°, $\mu d=0.5$ (new
   `checks/kernel_confound_check.py`, **[run]**). That gap is set by the
   geometry and moves by less than 0.01 with darkness, so it would hide the
   failure the matched comparison shows. The study therefore runs either at
   profile level with the `checks/` functions over the design's draws, or on
   the branch after Block 4 gains the cone kernel family. It should also
   report which $\hat\mu$ (and $\hat\alpha$) the partition can reach at all
   (D-031 coverage): the statement "no $\mu_{\rm ph}\in[0.5\mu,8\mu]$ reaches
   the ray world's $\hat\mu$ at $\mu d$ = 1.5 or 3" rests on the bracket's
   end points only. At the four values probed for the $d=1$ µm flat tube
   ($\mu_{\rm ph}$ = 0.75, 1.5, 12, 24 µm⁻¹) the partition's $\hat\mu$ is
   0.47-1.03 µm⁻¹, against the ray world's 1.333 and 2.327 µm⁻¹ **[run;
   arithmetic from the bracket values in `optics_points_check.out`]**. A scan
   over $\mu_{\rm ph}$ would turn that into a statement. For the end-to-end
   acceptance (procedure §3.10, last row), `specs/SPEC.md` proposes the
   empirical kernel, not the ray world, as the alternative generator, and
   keeps the ray world for absorption checks (open questions, raised
   2026-10-06).
   **[2026-10-09, later]** Could the ray world itself build the table for
   thick or dark nodes? Its missing diffraction costs less as $d$ grows but
   not as the stain darkens: the gap above is −0.058 / −0.048 at $d=2$ µm and
   −0.050 / −0.040 at $d=3$ µm ($\mu d$ = 0.5 / 3; flat, in focus,
   ideal-Debye kernel as the stand-in for the real blur) **[run]**. A
   ray-world table would then return thick faint nodes about 4-5 % too small
   ($\hat b$ 1.126 / 1.183 at 2 µm, 1.127 / 1.177 at 3 µm), against a whole
   correction of about 13 %. It would also bypass the kernel calibration
   (D-026, D-027), since the ray world has no kernel. A round tube's edges
   lie at its axis depth, where the chord, and so the stain along it, goes to
   zero. They are drawn by the near-focus blur, which is diffraction, while
   the bookkeeping error needs dark paths and sits in the interior. This
   supports the hybrid above (history along the rays, measured kernel), not
   the ray world as it is **[reasoning]**.
   **[qualified 2026-10-09, later; the user's proposal]** The hybrid has three parts:
   - absorption computed with the ray world's geometry (NA 1.4,
     $n_{\rm oil}$ 1.515, evenly filled cone);
   - every image plane rendered with the kernel, slice by slice (Eq. 6
     unchanged; only Eq. 5's $\Delta A_j$ is replaced);
   - the ray world's own cone spread dropped (blurring its finished planes
     would blur twice).

   A kernel does not know a ray's direction, so each slice point's absorbed
   light is averaged over the directions arriving there before it is spread:
   $\Delta A_j(\mathbf x)=\delta\zeta\,\big\langle(\mu/\cos\vartheta)\,
   e^{-\mu\ell_{\rm back}(\mathbf x,\zeta_j;\hat s)}\big\rangle_W$ inside
   the tube, where $\ell_{\rm back}$ is the path back along the ray to the
   tube's entry.

   It was tested where the answer is exact: in the ray world, with the
   cone's own $W^*$-weighted kernel (new `checks/hybrid_absorption_check.py`,
   **[run]**). For a flat 1 µm tube at the node plane, the hybrid's centre
   dip against the ray world's is 0.057 / 0.057, 0.450 / 0.441,
   0.845 / 0.820 and 0.961 / 0.965 at $\mu d$ = 0.05 / 0.5 / 1.5 / 3. It is
   first-order exact, and much closer than the partition. At $\mu d$ = 10
   and 50, however, it gives 0.811 and 0.732 against 1.000, a leak like the
   partition's (0.736, 0.699).

   Booking light per point and then spreading it without the ray's direction
   is the partition's own structure, and near the opaque limit it gives the
   skin-crossing factor of (S9) **[reasoning]**. **[corrected 2026-10-09,
   later 3]** It gives a factor like (S9)'s, not (S9)'s: the hybrid books each
   direction's light where that direction enters the tube (10.3 % of it above
   the axis depth at $\mu d=50$), not at the lower skin, and its total booking
   equals the ray world's total shadow (1.240 against 1.238 µm per unit length
   at $\mu d=50$, equal within the grid error), so the centre's leak is light
   moved, not lost
   (renderers document §3.9; `checks/hybrid_absorption_check.out`) **[run]**.
   Keeping each ray's history
   together with its own landing point is what makes the ray world black. A
   kernel narrower than the cone near focus cannot keep that pairing.
   **[corrected 2026-10-09, later 3]** The production kernel is narrower than
   the cone beyond about 0.1 µm of defocus (0.093 µm under $W^*$, 0.107 µm
   under $W$) and wider within it. Since variances add under convolution, a
   kernel narrower than the cone cannot be the cone plus a per-ray blur, so
   the direction average is forced for every slice farther than about 0.1 µm
   from the plane; within 0.1 µm the variance argument does not decide
   (renderers document §3.8, Eq. 15) **[reasoning; arithmetic]**.

   So the hybrid fixes the dark range up to about $\mu d$ = 3 and not
   near-opaque nodes. The real $\hat\mu_i$ / $\hat\alpha_i$ distribution
   (D-024 open point) decides whether that range is enough. Not yet tested:
   its fitted $\hat d$ (only centre dips so far), tilted tubes, and its cost.
2. **Coherence (weak-object approximation), effect 3 of the study notes.**
   Unmeasured. Data check: bright fringes beside dark branches in real
   profiles. No bright-field source found (PubMed, 5 queries, 2026-10-07);
   the literature search could be widened.
   **[2026-10-08, explained in the new theory chat]** Literature widened:
   PubMed, 15 queries; one full text, Zuo et al. 2017 (*Sci Rep* 7:7654,
   PMC5550517): under Köhler illumination the image is an incoherent sum of
   the coherent images from the source points; at $S\ge1$ the weak-object
   transfer function is the pupil autocorrelation (absorption imaged as by an
   incoherent microscope, phase not imaged), while full incoherence needs
   $S\to\infty$. No source on dark absorbers. Derived **[reasoning, not
   run]**, for a thin object, scalar optics, $S=1$ with the aperture evenly
   filled, $\mathbf q$ the transverse frequency of one illuminating plane wave
   $e_{\mathbf q}$ and $\Omega_{\rm obj}$ the objective's aperture:
   $I_{\rm model}(\mathbf x)-I_{\rm true}(\mathbf x)=\frac{B}{\int|h_a|^2}\int_{\mathbb R^2\setminus\Omega_{\rm obj}}\big|\big((w\,e_{\mathbf q})*h_a\big)(\mathbf x)\big|^2\,d^2\mathbf q\ \ge0$.
   The model is too bright, everywhere, by the stain's dark-field image from
   the directions the condenser does not supply; the difference has no
   constant or linear term in $w$ (hence the first-order agreement). Integrated
   over the image it equals $B\int|\tilde W(\mathbf m)|^2\,[1-\Lambda(\mathbf m)]\,d^2\mathbf m$
   (unapodised pupil; $\Lambda$ the normalised overlap of two aperture discs),
   whatever the focus: the light the stain diffracts out of the aperture,
   which the model keeps. **[corrected 2026-10-08]** The data check above does
   not test this term: at $S=1$ a thin passive object gives
   $I_{\rm true}\le I_{\rm model}\le B$, so a fringe above background signals
   that the premise failed ($S<1$, or an unevenly filled aperture) or
   refraction in thick nodes, and its absence does not bound effect 3. The effect on $\hat d$ is still
   uncomputed; a direct check would render thin-object phantoms by the sum
   over condenser directions and by $B\,(T*h_0)$, fit both and compare
   $\hat d$.
   **[2026-10-08, run in the previous theory chat's session, answering the
   user]** `checks/coherence_check.py` (+ `.out`) renders a straight branch in
   focus as a thin object, scalar optics, $\lambda=0.55$ µm, NA 1.4, three ways:
   coherent (one plane wave), the sum over condenser directions at $S=1$, and
   $B\,(T*h_0)$. It confirms the sign above numerically (smoke check:
   $I_{S=1}\le I_{\rm model}\le B$ everywhere, with the aperture sampled at the
   FFT grid spacing). The model is brighter by at most 0.013 $B$ ($\mu d=1$)
   and 0.060 $B$ ($\mu d=3$) for $d=0.5$ µm, and 0.009 / 0.045 $B$ for
   $d=1$ µm; the dip FWHM at $S=1$ exceeds the model's by 2.3 % / 4.0 %
   ($d=0.5$ µm) and 1.1 % / 2.1 % ($d=1$ µm) at $\mu d$ = 1 / 3, and the two
   agree at $\mu d=0.1$. Coherent illumination gives a bright fringe of
   +0.04 $B$ ($\mu d=1$) to +0.11 $B$ ($\mu d=3$) and, for $d=0.5$ µm, a
   narrower dip (FWHM 0.405 against 0.454 µm at $\mu d=1$). Widths of the dip,
   not fitted $\hat d$; no defocus, no depth extent, no camera **[run]**.
3. **D-032 confirmation.** Does dropping tilt-specific tables also withdraw
   D-027's tilt-binned study? Ask the user; correct the log if the reading is
   wrong.
4. **D-031 details:** axis scaling, $\hat\alpha$ or $\log\hat\alpha$, replicate
   budget, correction outside the phantoms' $\hat\alpha$ range.
5. **Comments awaiting the user:** D-025 (a)-(f), D-026 (a)-(h), D-027
   (a)-(d), D-030 (a), (c).
6. **Pending OK (asked 2026-10-06, not answered):** fold the partition and
   vertical-ray findings and D-025 to D-032 into procedure §3.4, §3.6, §3.9
   and mathematics §3.2, §3.5. Those two documents are stale on these points;
   the implementation handoff is current.
7. **I-002** (non-Gaussian kernel): to discuss.

## Study material for the user

`docs/TEEG_diameter_study_notes_2026-10-07.md` copies the two answers the user
asked to study: the explanation of D-030 and of what happens near
$\alpha\approx1$ (17:02), and the answers to questions 1-5 of 18:01 (D-031,
D-032, $\hat\tau$ as uncertainty, the weak-object approximation, ray world vs
partition with the figure).

**[added 2026-10-09, v1.6]**
`docs/TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md`, written
at the user's request and independently reviewed: how the partition and the ray
world differ (one formula, two substitutions), what each gets right and wrong
for thick and dark dendrites, why a comparison must share the blur, and the
user's hybrid (its formula, why it must average over directions, how well it
does, and what is untested).

## Admin notes

- **The log is large** (about 255,000 characters after D-032). A full
  `project_read` in the main context costs about 65,000 tokens. What worked on
  2026-10-07: a subagent does the `project_read` and reports only a
  fingerprint (the changelog rows of the day, the last 120 characters); the
  main chat extracts the full text from the subagent's transcript
  (`~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`, the
  `tool_result` whose JSON has `"path": "TEEG_decisions_and_ideas_log.md"`),
  diffs it against the last version written, merges the new entries with a
  script that asserts every anchor occurs exactly once, and writes with
  `project_write` using `local_path` inside the working directory (then
  removes the temporary file).
- **Merging into the log:** never edit a decision in place; add a dated note
  or annotate the index status; one write per turn; re-read before writing.
- **GitHub:** on 2026-10-07 five pushes to `main` failed with "Internal
  Server Error" while githubstatus.com reported no incident; a retry some
  minutes later went through. Retry rather than force.
- **Commits:** the user's name and email from the session context; the
  attribution lines the session asks for.
- **A local branch `diameter-remeasurement-docs`** was used in the old chat,
  tracking `origin/main`; nothing else depends on it.
