# Decision-log additions pending for the project log: D-025 to D-027, I-002 (2026-10-06)

| Date | Change |
|---|---|
| 2026-10-06 | Created by the theory chat. The project log `TEEG_decisions_and_ideas_log.md` (claude.ai project knowledge, root copy) could not be written: `project_write` refused the merged log (55,308 tokens) because project knowledge stood at 1,954,553 of 2,000,000 (`project_info`, 2026-10-06). The refusal implies that any rewrite of the log, even unchanged, exceeds the free space **[inferred from those two numbers]**, so freeing space needs the user's OK. Until then this file holds the text of the three entries, written after re-reading the log (D-001 to D-024, I-001; no D-025 or I-002 there). **[corrected 2026-10-07]** The size inference was not borne out: the log was rewritten on 2026-10-07 by the Ih Fit chat (D-028, D-029) and again, larger, by the merge of this file; whether space was freed in between was not checked, so why the 2026-10-06 write was refused is not established. |
| 2026-10-06 (later) | Adds **D-027** (user, 18:12): calibration first on flat branches, near and far; tilted branches later. Adds its index and log-changelog rows and annotates D-026's status. Comments rest on `checks/calib_width_check.py` (run 2026-10-06). The project log is still not written. |
| 2026-10-07 | **Merged** into the project log `TEEG_decisions_and_ideas_log.md` (root) at 13:48 UTC, after re-reading it at 13:46 UTC; it then held D-028 and D-029 from the Ih Fit chat and a row reserving D-025 to D-027 and I-002. Entries and index rows verbatim; anchors adapted (note under the anchor table). Status set to merged; the size inference of the first row marked [corrected 2026-10-07]. The project log is now the reference; this file keeps the text as written on 2026-10-06. Evidence: `project_read` and `project_write` (`replaced: true`), 2026-10-07. |

**Status: merged 2026-10-07** into the project log (its changelog row
2026-10-07 (later, 3)); the project log is now the reference.
~~**Status: pending.**~~ **[corrected 2026-10-07]** When the project log is
written, append these pieces verbatim at the anchors below, re-read the log
first (merge, never overwrite), then mark this file "merged <date>" here; do
not delete it without the user's OK.

## Where each piece goes in the project log

| Piece | Anchor in the project log |
|---|---|
| changelog row | a new line after the row starting `\| 2026-10-06 (later) \| Logs **D-022**` |
| index rows D-025, D-026, D-027 | new lines after the row starting `\| D-024 \| 2026-10-06 \|` |
| index row I-002 | a new line after the row starting `\| I-001 \| 2026-09-19 \|` |
| D-024 index status | replace `jitter knob at 0 until cell 13 \| active \|` by `jitter knob at 0 until cell 13 \| active; **(i)'s configured $\gamma$ to be measured -- D-026** \|` |
| D-024 status note | a new paragraph after D-024's status line ending `survives as a reported diagnostic column, \`steep\`).` |
| entries D-025, D-026, D-027 | before the heading `## Ideas for future parts` |
| entry I-002 | at the end of the log, after I-001 |

**[corrected 2026-10-07] Anchors used in the merge.** The log had changed
(D-028 and D-029 after D-024's entry; changelog rows 2026-10-07 to
2026-10-07 (later, 2)), so: the two changelog rows below became one log row,
2026-10-07 (later, 3), after the row `| 2026-10-07 (later, 2) |`; the entries
D-025 to D-027 went before `## D-028 --`, not before `## Ideas for future
parts`, to keep numeric order; the D-024 index status and the D-024 note also
name D-027. Index rows and entries verbatim.

## Changelog rows

| 2026-10-06 (later, 2) | Logs **D-025** (user, 17:01): the image noise of the real stacks is measured on clean background patches -- in every plane, at several positions per plane and in several stacks -- through its second moment, and its distribution is examined. Logs **D-026** (user, 17:01): the defocus-kernel calibration is extended beyond the ±0.84 µm that thin nodes reach, using dendrites whose Allen diameter is just under 0.8 µm, read on their planes beyond 0.84 µm; the near-focus law is continued there and adjusted, and the kernel stays Gaussian for now -- D-024 (i)'s configured slope $\gamma$ becomes a measured one. Logs idea **I-002** (user, 17:01, "to discuss further"): a non-Gaussian kernel family with the Gaussian as a special case. The assistant's comments on each are recorded in the entries as open points, not yet answered; the numbers behind D-026's comments come from `checks/calib_reach_check.py` (repo, `main`). Written by the theory chat after re-reading this log, whose D-022 to D-024 the implementation chat had added the same afternoon. |
| 2026-10-06 (later, 3) | Logs **D-027** (user, 18:12): the defocus calibration starts with flat branches only, for the near (thin) and far (thick) planes; tilted branches come later, to see whether the profile widths change. Refines D-026, whose status is annotated. The assistant's comments -- tilt changes the second moment much more than the core width, so tilted data are compared with the renderer's prediction -- rest on `checks/calib_width_check.py`. |

## Index rows

| ID | Date | Binds | One line | Status |
|---|---|---|---|---|
| D-025 | 2026-10-06 | the noise step of the renderer's camera chain (procedure §3.6 step 6), the nuisance "noise" (procedure §3.5), `RendererConfig.noise_sd_gl`, the configuration $\mathcal C$ of every table | the real stacks' noise is measured on clean background patches, in every plane, at several positions and in several stacks: its second moment (D-025.1) first, then its distribution | active; the assistant's comments (a)-(f) await the user |
| D-026 | 2026-10-06 | the kernel calibration (procedure §3.4, Eq. 4; block 10), `kernel_continuation` and its slope $\gamma$ (D-024 (i)), the choice of calibration nodes | beyond ±0.84 µm the kernel is calibrated on dendrites just under 0.8 µm (Allen diameter), read on their planes beyond 0.84 µm; the near law is continued (D-026.1) with $\gamma$ fitted, other terms only if the residuals need them; Gaussian for now | active; replaces D-024 (i)'s configured $\gamma$ once run; the assistant's comments (a)-(h) await the user; **refined by D-027 (flat nodes first)** |
| D-027 | 2026-10-06 | the choice of calibration nodes (procedure §3.4 thin set; D-026 thick set); block 10; a later tilt study | the calibration starts with flat branches only ($\hat\varphi_i \approx 0$), near and far from focus; tilted branches come later and are compared with the renderer's prediction | active; refines D-026; the assistant's comments (a)-(d) await the user |
| I-002 | 2026-10-06 | the kernel family of the renderer and of the calibration (procedure §3.4) | a flexible kernel family with the Gaussian as a special case, its parameters fitted in the calibration | idea, to discuss further [user]; not scheduled |

## Note for D-024's status

**Note added 2026-10-06 (later, 2); no change to D-024's text.** (i)'s
configured slope $\gamma$ is to be replaced by a value measured on dendrites
just under 0.8 µm -- **D-026**; the noise of the camera chain is to be
measured on clean background patches -- **D-025**; a non-Gaussian kernel is
idea **I-002**.

---

## D-025 -- The image noise of the stacks is measured on clean background patches, in every plane, at several positions and in several stacks

**Date:** 2026-10-06, 17:01. **Binds:** the noise step of the renderer's
camera chain (procedure §3.6 step 6) and the nuisance draw "noise" (procedure
§3.5); the noise field of `config.py` on `sci/diameter-pipeline`
(`RendererConfig.noise_sd_gl`, 3.0 grey levels, provisional); the
configuration $\mathcal C$ of every bias table, which lists the noise
(procedure §3.3) and must be rebuilt when it changes (§3.11); a Phase-II
measurement script (Colab, real stacks).

**Decision [user].** "We will measure the in slice noise, taking a neat patch
(without any branch crossing the patch across the z stack from the condenser
up to the current focus plane) on the focus plane and calculating the noise.
This will be done across all the real images planes, across images and
withint the same plane across several position. We calculate the second
moment of such noise and try to understand whether there are specific
ditributions used to describe such noise."

**Statement.** For a real stack $s$, a plane $k$ of it and a patch $P$ (a set
of pixels of plane $k$; size open), let $I_{s,k}(x, y)$ (grey levels) be the
recorded image, $m_{s,k,P}$ its mean over $P$, and
$$v_{s,k,P} = \frac{1}{\lvert P\rvert - 1}\sum_{(x,y) \in P}\bigl(I_{s,k}(x,y) - m_{s,k,P}\bigr)^2 \quad (\text{grey levels}^2)
\tag{D-025.1}$$
its second central moment. $v_{s,k,P}$ is a number computed from the data
(computed level); it estimates the variance of the noise of the camera chain
at the grey level $m_{s,k,P}$ (analytic level) only if the noise is
stationary within $P$ and nothing else varies there (comment (b)). It is
computed in every plane $k$, at several patches per plane and in several
stacks $s$; its spread over $(s, k, P)$ tests whether the noise is stationary
across depth, field position and specimen, and the pixel distribution within
the patches is compared with candidate families. A patch qualifies only if no
stained structure crosses its column (the user's criterion; comment (a)
extends it).

**Why.** The renderer's injected noise must reproduce the real background
statistics after the whole camera chain (procedure §3.6 step 6)
**[KB-repo]**. $b$ is an average, over nuisance draws that include the
noise, of a non-linear fit's output conditional on the selection
$\mathcal S$ (procedure Eq. 2), so the noise level can move $b$ itself, not
only its scatter **[KB-repo; reasoning]**. The real noise level is not
known, and with it the usable calibration range in planes (implementation
handoff, "Known gaps") **[KB-repo]**.

**The assistant's comments, recorded with the decision; not yet answered
[reasoning unless tagged].**

- (a) **Both sides of the focal plane.** Plane $k$ records the blurred shadow
  of every stained slab of the section, below and above it: procedure
  Eq. (6) sums over all slabs $j$ **[KB-repo]**. A clean patch therefore
  needs no stained structure at any depth within a lateral margin of about
  $3\sigma_{\rm r}(\delta)$ around it, $\delta$ being that structure's depth
  distance from plane $k$, on the objective side as well as on the condenser
  side.
- (b) **Detrend first.** Illumination shading and the haze of distant
  stained structure vary across a patch; a plane (or low-order polynomial)
  fitted per patch and removed before (D-025.1) keeps them out of $v$.
- (c) **Exposure noise and static structure.** For the same patch in
  neighbouring planes, $\tfrac12\operatorname{Var}_P(I_{s,k} - I_{s,k+1})$
  estimates the part of the noise that is independent between exposures;
  parts common to both planes (sensor fixed pattern, dust in the optics,
  slowly varying tissue texture, possibly JPEG errors on nearly identical
  blocks) cancel in it. The single-plane $v$ is what the fit sees; the
  difference between the two is the static part.
- (d) **Level dependence and candidate laws.** For raw camera data the
  standard description is signal-dependent: a Poisson part from photon
  counting plus a Gaussian part for the remaining stationary disturbances,
  the noise SD a function of the expected pixel value, with clipping at
  under- and over-exposure (Foi et al. 2008, **PubMed, abstract only; full
  text not accessible**); the same mixed Poisson-Gaussian model is used for
  light-sheet fluorescence images (Julia et al. 2024, **PubMed full text**).
  The Allen images have passed a grey-level mapping, 8-bit quantisation and
  JPEG, so that law is not transplanted: $v$ is plotted against $m$ and the
  relation is read off the data, which needs patches darker than the
  background (e.g. uniform interiors of broad out-of-focus shadows, after
  detrending). Uniform 8-bit quantisation alone adds about 1/12 grey
  level$^2$ when the noise SD exceeds about one grey level (textbook, from
  memory).
- (e) **Spatial correlation.** JPEG codes 8 x 8 pixel blocks (textbook, from
  memory), and the profile of handoff Eq. 9 is bilinearly interpolated and
  optionally averaged over ±0.5 µm along the branch **[KB-repo]**, so
  correlated noise does not average down as white noise does. The
  autocovariance at lags of 1-8 px in $x$ and $y$ is measured beside $v$.
- (f) **How it enters the renderer.** Either a parametric noise injected
  before the synthetic 8-bit and JPEG steps and tuned until the background
  statistics after the chain match the measured ones (procedure §3.6 step 6)
  **[KB-repo]**, or real detrended background patches added after the
  synthetic chain (distribution and correlation kept without choosing a
  family; exact only at the background level, and the synthetic JPEG then
  acts on a noise-free image).

**What it implies for code and tables.** A Phase-II script writes one row per
patch: stack, plane, position, size, $m$, $v$ (detrended), the lag-1 to
lag-8 autocovariances in $x$ and $y$, the plane-difference variance of (c),
and the clean-column criterion used. `config.py` gains a noise-model selector
whose parameters take that table as their source and which replaces the
provisional `noise_sd_gl` (names, e.g. `"gaussian_after_chain"` and
`"empirical_patches"`, to be fixed in `specs/SPEC.md` by the implementation
chat).

**What it does not decide [open].** The patch size and the clean-column
margin; whether the noise model depends on the grey level; the injection
route of (f); whether the noise differs between specimens (then it is an
element of $\mathcal C$ per specimen, procedure §3.11).

**Sources.** **[user]**, 2026-10-06 17:01. Procedure Eq. 2, §3.3, §3.5, §3.6,
§3.11; handoff Eq. 9; implementation handoff "Known gaps" (repo, D-020). Julia
A. et al. 2024, *Sensors* 24(7):2053 (PMID 38610265, PMC11014158), PubMed full
text, [DOI](https://doi.org/10.3390/s24072053). Foi A. et al. 2008, *IEEE
Trans Image Process* 17(10):1737-54 (PMID 18784024), PubMed abstract only,
[DOI](https://doi.org/10.1109/TIP.2008.2001399). Searches 2026-10-06: PubMed,
11 queries (Poisson-Gaussian noise in microscopy and in camera raw data;
photon transfer; sCMOS pixel noise; shot and read noise; the generalized
Anscombe transform; JPEG and lossy compression in quantitative microscopy;
brightfield noise models; brightfield defocus PSF and deconvolution of
biocytin-filled dendrites), 0-4 records each; bioRxiv, bioengineering, last 30
days (30 preprints): nothing relevant. JPEG block size and the quantisation
variance: textbook, from memory.

**Status:** active. Comments (a)-(f) await the user.

---

## D-026 -- Beyond ±0.84 µm the defocus kernel is calibrated on dendrites just under 0.8 µm, continuing the near-focus law; the kernel stays Gaussian for now

**Date:** 2026-10-06, 17:01. **Binds:** the kernel calibration (procedure
§3.4, Eq. 4; block 10 of the implementation handoff; `CalibrationConfig` on
`sci/diameter-pipeline`); `RendererConfig.kernel_continuation` and
`kernel_continuation_slope` ($\gamma$; D-024 (i): `linear`, 0.79, "for
now"); the choice of calibration nodes; the renderer's $\sigma_{\rm
r}(\delta)$ for $\lvert\delta\rvert > 0.84$ µm.

**Decision [user].** "I do not actually understand much the concern with
thick branches. It harden the estimation of the calibration table but these
are valuable for the calibration table values past the 0.8 um limit set by the
thin branches.. For the calibartion table we take a node, center it, find the
measuring direction, calculate the second moment and then run toward smaller
values of z across the slabs (with the same second moment calculation). A thin
branch (that has a diameter less than the z distance between slabs) casts a
shadow that has a good SNR till the third slab. Afterward the shadow is too
faint. However we have a great prior information to add when we estimate the
tables' values for far (>0.8 ) slabs from the thick dendrites. What we can do
is to find dendrites whose Allen diameter  is estimated to be slighlty less
than 0.80 so that we can do the analysis from the slabs that are further in
depth than 0.8um. We extrapolate the value from the law under 0.8 and then try
to adjust it (with other terms or whatever) Keeping for now the gaussian
kernel assumption." In this log's vocabulary the user's "slabs" are the
recorded planes ($\Delta z = 0.28$ µm), not the renderer's slabs (procedure
§3.6).

**Statement.** Two calibration sets enter procedure Eq. (4). The thin set
(procedure §3.4: $\hat d \lesssim 0.3$ µm, faint, $\varphi \lesssim 10°$,
isolated) gives $\Delta\sigma^2(\delta)$ for $\lvert\delta\rvert \le 0.84$
µm. The thick set -- nodes whose Allen diameter is just below 0.8 µm -- is
read on its planes with $\lvert z_{{\rm ax},i} - z_k\rvert > 0.84$ µm. Beyond
0.84 µm the kernel width follows, for each fixed $\delta$ with
$\lvert\delta\rvert > 0.84$ µm,
$$\sigma_{\rm r}(\delta) = \sigma_{\rm r}(0.84\ \mu{\rm m}) + \gamma\,\bigl(\lvert\delta\rvert - 0.84\ \mu{\rm m}\bigr),
\tag{D-026.1}$$
the continuation of D-024 (i), with $\gamma$ fitted on the thick set instead
of configured, and further terms added only if the residuals require them.
$K_\delta$ stays a circular Gaussian of width $\sigma_{\rm r}(\delta)$; the
alternative is I-002. Levels (R8): $\sigma_{\rm r}(\delta)$ and $\gamma$ are
the model's quantities; the calibration returns estimates
$\hat\sigma_{\rm r}(\delta)$ and $\hat\gamma$, computed from the real stacks.

**Why.** A thin node's shadow fades within about three planes (**[user]**;
implementation handoff "Findings", "Thin calibration nodes fade fast"
**[run]**). A thicker faint node removes more light: in the faint limit the
dip keeps its area from plane to plane (mathematics Eq. 13a), and that area is
$\bar B\mu\pi d^2/(4\cos\varphi)$ (D-019 (a)), while its depth falls as
$1/\sigma_{\rm r}$ (Eq. 13b); so it stays measurable further from focus
**[KB-repo; reasoning; run, comment (c)]**. D-024 (i) left $\gamma$
configured "for now".

**The assistant's comments, recorded with the decision; not yet answered
[reasoning unless tagged].**

- (a) **The statistic.** Procedure §3.4 [corrected 2026-10-04] calibrates with
  the squared Gaussian-core width of the dip, fitted within about one FWHM,
  and keeps the windowed second moment only as a cross-check: every real
  line-spread function has $\lvert v\rvert^{-2}$ tails at every defocus
  (mathematics §3.3), so its full second moment is infinite and a windowed one
  is dominated by the tails -- in focus, ideal Debye, a core width of 0.080 µm
  against $\sqrt{V_W}$ = 0.240 µm at $W$ = 1.5 µm and 0.348 µm at $W$ = 3 µm
  **[run, `psf_check.py`, `doc_checks.py` C5, as cited in procedure §3.4]**.
  "Second moment" in the decision is read as that core width; the reading is
  not confirmed by the user.
- (b) **Eq. (4) holds for the thick set as written; what the far planes
  identify.** Its per-node constant $c_i$ absorbs the node's own width. In
  the faint limit, and where $\sigma_{\rm r}$ is linear in $\delta$ across the
  node's depth, the depth extent of a flat round node ($r = d/2$) adds only
  the constant $\gamma^2 r^2/4$ to the dip's variance (its absorbance along
  depth has a semicircle profile, of variance $r^2/4$), which $c_i$ also
  absorbs -- exact for the second moment, approximate for the core width.
  Under (D-026.1), with $x_{i,k} = \lvert \hat z_{{\rm ax},i} - z_k\rvert -
  0.84$ µm on one side of the axis and $s_i$ the error of the node's estimated
  axis depth $\hat z_{{\rm ax},i}$, $\omega_{i,k}$ is, up to a per-node
  constant, $2\gamma\,(\sigma_{\rm r}(0.84\ \mu{\rm m}) + \gamma s_i)\,x_{i,k}
  + \gamma^2 x_{i,k}^2$: an axis-depth error acts exactly like a change of the
  anchor $\sigma_{\rm r}(0.84\ \mu{\rm m})$. With $z_{{\rm ax},i}$ free per
  node, only the curvature $\gamma^2$ identifies $\gamma$; with
  $z_{{\rm ax},i}$ taken from the node's own focus search (handoff Eqs. 1-2),
  or with planes on both sides of the axis (there $s_i$ enters with opposite
  signs and the two linear coefficients average to $2\gamma\sigma_{\rm
  r}(0.84\ \mu{\rm m})$, if $K_\delta = K_{-\delta}$), the linear term
  identifies it too. **[reasoning; both algebraic statements checked
  numerically, `checks/calib_reach_check.py` part 3]**
- (c) **How far it reaches** **[run, `checks/calib_reach_check.py`;
  illustrative ideal-Debye kernel, partition renderer, flat tube, planes on
  the light side, noise-free; reference computed with slope 0.79]**: a 0.8 µm
  node at $\mu = 0.6$ µm$^{-1}$ ($\mu d = 0.48$) keeps a centre dip at least
  as deep as a 0.3 µm node's at 3 planes out to 16 planes (4.48 µm) if the far
  slope is 0.79 and 11 planes (3.08 µm) if it is 1.21; at $\mu = 0.3$
  µm$^{-1}$, 9 and 7 planes. The run assumes a far law itself, so the numbers
  show the gain, not the range on real data.
- (d) **Windows and isolation grow with it.** At those distances
  $\sigma_{\rm r} \approx$ 2-3.5 µm (same run): the profile (±3 µm, handoff
  Eq. 9) and the block (about 10 µm, implementation handoff `block_half_um`)
  must widen to about $\pm 3\sigma_{\rm r}$, and the clearance from other
  stained structure with them. The background needs a trend term (D-025
  comment (b)); the masked block median of D-023 is biased low once the
  shadow is wider than the mask.
- (e) **Faint thick nodes only** ($\hat\alpha \lesssim 0.5$). Procedure §3.4
  step 2 tunes $\sigma_{\rm r}(\delta)$ by matching rendered phantoms to the
  real profiles, and the default renderer (partition along vertical rays,
  D-024 (i)) fails qualitatively for dark thick tubes: the in-focus centre
  lightens as the stain darkens, the synthetic focus curve peaks toward the
  light, and no phantom $\mu$ reaches the truth's $\hat\mu$ at $\mu d$ = 1.5
  or 3, while one does at $\mu d \le 0.5$ (implementation handoff,
  "Findings" (1), (2), (4) **[run]**). Dark nodes would also break the faint
  limit of (b).
- (f) **Both sides of focus.** Stepping only toward smaller $z$ measures one
  sign of $\delta$. $K_\delta = K_{-\delta}$ holds for an index-matched system
  free of spherical aberration (optics §3.5) and fails under index mismatch
  (optics §3.7); procedure §3.10 ("Above/below") fits Eq. (4) per sign to
  test it. Both sides also pin the axis depth, (b).
- (g) **A ladder of diameters** (about 0.3, 0.5 and 0.8 µm) gives overlapping
  $\delta$ ranges, so each class's curve is checked against the next where
  they overlap, not only joined at 0.84 µm. A node's near planes, where
  $\sigma_{\rm r}$ is not linear across its depth, do not satisfy the constant
  of (b).
- (h) **What it covers** **[run, same script]**. With D-024's phantoms
  ($U = 10$ µm, $\varphi \in [0°, 90°)$), the farthest slab from the plane
  through the node's axis is at $U\tan\varphi + r/\cos\varphi$ ((S5) of the
  implementation handoff). For $d = 0.5$ µm the thin set's 0.84 µm covers
  tilts up to 3.4°, and a calibrated range of 3.08-4.48 µm covers up to
  15.7-22.8°. Steeper phantoms, and the planes of the focus search, which
  reach $\lvert z_k - c_z\rvert$ further, keep relying on (D-026.1) beyond
  the measured range.

**What it implies for code and tables.** Block 10 takes a calibration-node
table with a diameter class per node and a per-class plane window, fits
Eq. (4) with per-node $c_i$ and, for the thick set, $z_{{\rm ax},i}$ from the
node's own focus search (comment (b)), and writes the tabulated
$\Delta\sigma^2$, its measured range and $\hat\gamma$.
`kernel_continuation_slope` changes status from configured to measured, and
the measured range enters $\mathcal C$; the renderer reads $\sigma_{\rm
r}(\delta)$ from the calibrated table where it exists and from (D-026.1)
beyond. A Phase-I oracle for block 10: synthetic thick-node scans rendered
with a known $\gamma$ return it.

**What it does not decide [open].** The thick set's selection
(Allen-diameter band, faintness threshold, isolation radius); its profile
window and block size; the form of the "other terms"; whether the two sets
are fitted jointly or the far law after the near one; the anchor of
(D-026.1) if the residual check of procedure §3.4 trims the thin set's range
to ±2 planes.

**Sources.** **[user]**, 2026-10-06 17:01. Procedure §3.4, §3.6, §3.10;
mathematics §3.3, Eqs. 13a-13b, Eq. 17; optics §3.5, §3.7; handoff Eqs. 1-2,
9; implementation handoff "Findings", (S5) and `block_half_um` (repo, D-020);
D-019 (a), D-023, D-024 (this log). Comments (c), (h):
`checks/calib_reach_check.py` with `calib_reach_check.out`, run 2026-10-06
**[run]**. Comment (b): **[reasoning]**, derivation as written, checked by
part 3 of the same script.

**Status:** active; replaces D-024 (i)'s configured $\gamma$ once the
calibration has run. Comments (a)-(h) await the user. **Refined 2026-10-06
18:12 by D-027:** both calibration sets start with flat nodes; tilted nodes
come later, as a test against the renderer.
---

## D-027 -- The defocus calibration starts with flat branches only, near and far from focus; tilted branches come later, as a test

**Date:** 2026-10-06, 18:12. **Binds:** the choice of calibration nodes in
procedure §3.4 (thin set, "$\varphi \lesssim 10°$") and in D-026 (thick set,
no tilt criterion); block 10 of the implementation handoff
(`CalibrationConfig` on `sci/diameter-pipeline`); a later tilt study.

**Decision [user].** "For the calibration of the defocus second moments I
would use initially only branches that have zero tilt angle for near and far
off focus layers. The if possible construct the same but finding braches with
several different tilt angles and studying if the second moments change in
the profile"

**Statement.** The first calibration of $\Delta\sigma^2(\delta)$ -- the thin
set for $\lvert\delta\rvert \le 0.84$ µm and the thick set of D-026 beyond --
uses only nodes whose tilt $\hat\varphi_i$ from the local line fit (handoff
Eq. 5) is zero within a tolerance $\varphi_{\rm tol}$ (open). Afterwards,
where enough nodes exist, the same plane scans are repeated on nodes binned by
tilt, and their width curves are compared.

**Why.** At $\varphi = 0$ a node's profile in plane $k$ holds a single
defocus, up to the node's own depth extent, which is what Eq. (4) and D-026
comment (b) assume **[KB-repo; reasoning]**.

**The assistant's comments, recorded with the decision; not yet answered
[reasoning unless tagged].**

- (a) **What a tilt study can show.** A tilted branch's profile at a node also
  collects light from the neighbouring stretches of the branch, which lie at
  other depths, so the profile changes with $\varphi$ even when the kernel
  depends on depth only. In the model **[run, `checks/calib_width_check.py`
  part 2: thin node, depth-only Gaussian kernel, planes 1-9]** the square root
  of the second moment over $\pm 5\sigma_{\rm r}$ grows by 4-5 % at 10°,
  17-27 % at 20° and 32-59 % at 30°, while the Gaussian-core width changes by
  at most 1 %, 3 % and 6 %. The comparison is therefore between the tilted
  measurements and the renderer's prediction from the flat calibration
  (procedure §3.10, "Real vs synthetic profiles"), not between tilted and flat
  numbers; a mismatch then means that the kernel is not depth-only or that the
  renderer misses something (oblique illumination, the partition).
- (b) **The statistic.** The second moment's sensitivity in (a) is the
  mechanism of D-026 comment (a): broad, faint contributions dominate it, so it
  also moves with the window. For the same profile (plane 1, 30°) its ratio is
  1.32 over $\pm 0.61$ µm and 1.78 over $\pm 1.0$ µm, while the core ratio is
  1.054 over both **[run, part 3]**. The core width is the statistic for the
  tilt study as well.
- (c) **"Zero" needs a tolerance.** At 5° the core width changes by at most
  0.3 % (the second moment by 0.9-1.2 %) in the same run, so a tolerance of
  about 5° costs little with the core width; it must also exceed the error of
  $\hat\varphi_i$ from the line fit, which is not measured.
- (d) **How many nodes.** Flat thick dendrites (Allen diameter just under
  0.8 µm, faint, isolated, $\hat\varphi_i \approx 0$) may be few in one cell;
  their count decides whether more specimens are needed **[open]**.

**What it implies for code and tables.** `CalibrationConfig` gains a tilt
tolerance applied to both sets (e.g. `phi_tol_deg`, provisional 5°) and a
tilt-binned mode for the later study, whose output is labelled as a comparison
with the renderer's prediction.

**What it does not decide [open].** $\varphi_{\rm tol}$; the tilt bins; the
number of nodes per bin.

**Sources.** **[user]**, 2026-10-06 18:12. Procedure §3.4, §3.10; handoff
Eq. 5; D-026 (this file). Comments (a)-(c): `checks/calib_width_check.py` with
`calib_width_check.out`, run 2026-10-06 **[run]**. Searches 2026-10-06:
PubMed, 3 queries (the second moment of a PSF with heavy tails and windowing;
tilted-fibre defocus PSF calibration in brightfield; orientation bias of
neurite diameters), 0 records each.

**Status:** active. Refines D-026 (adds a tilt criterion to both calibration
sets) and procedure §3.4's thin-set criterion ($\varphi \lesssim 10°$ becomes
$\hat\varphi_i \approx 0$). Comments (a)-(d) await the user.

---

## I-002 -- A non-Gaussian defocus kernel, with the Gaussian as a special case

**Date:** 2026-10-06, 17:01. **Concerns:** the kernel family of the renderer
and of the calibration (procedure §3.4, "Two kernel families"); D-026 keeps
the Gaussian for now.

**Idea [user].** "For now I see that we have assumed the defocus kernel as
being Gaussian which might not be the case. So in estimating the kernel and
the calibration table  we can also account for a more flexible family
distribution (that has as a specific case the Gaussian distribution but can
also be shaped into skewed ones) and fit also its parameters given the
estimated second moment. It's pretty difficult so maybe we might need to
discuss this further. Take note of this for the things to discuss furhter."

**Why it might matter.** The real line-spread function differs from a
Gaussian in two known ways: $\lvert v\rvert^{-2}$ tails at every defocus,
which dominate its second moment near focus (mathematics §3.3), and, away
from focus, a shape that broadens, flattens on top and grows a rim
(mathematics §3.4) **[KB-repo]**. A Gaussian matched to one statistic misses
the rest (procedure §3.4).

**First comments of the assistant, for the discussion [reasoning unless
tagged].**

- (a) **Symmetric across the branch, asymmetric in depth.** For a
  rotationally symmetric PSF the profile across a straight branch is even in
  $v$, so a kernel skewed in $v$ would need off-axis aberrations or oblique
  illumination -- testable as the skewness of averaged thin-node profiles.
  What can be asymmetric is the dependence on $\delta$: the defocused PSF is
  the same above and below focus in an index-matched system free of spherical
  aberration (optics §3.5) and differs under index mismatch (optics §3.7;
  procedure §3.10, "Above/below") **[KB-repo]**. The natural generalisation is
  then a family symmetric in $v$, with a shape parameter for the tails and the
  flatness of the top, allowed to differ for $\delta > 0$ and $\delta < 0$.
- (b) **Identifiability.** One second moment fixes one parameter; a shape
  parameter needs the profile's shape -- procedure §3.4 step 2 already fits
  whole per-plane profiles -- or further statistics (two window widths, a
  fourth moment). A heavy-tailed member has no finite second moment, so the
  window becomes part of the definition.
- (c) **Two alternatives to weigh against it.** The empirical kernel already
  in procedure §3.4 (averaged real profiles, after four corrections), and a
  physical PSF model whose few parameters (mounting index, depth, condenser
  aperture) generate the shape. For transmitted-light brightfield stacks of
  rat cortical tissue in Mowiol imaged with an oil objective, Oberlaender et
  al. 2009 measured the aberration function of microscope and tissue with a
  Shack-Hartmann sensor, report the tissue's index as homogeneous and the
  spherical aberration as relatively low, and deconvolve with a model PSF
  built from refraction between embedding and immersion medium and 3-D
  diffraction at the objective's pupil, for reconstructing biocytin-stained
  dendrites and axons (**PubMed, abstract only; full text not accessible**,
  [DOI](https://doi.org/10.1111/j.1365-2818.2009.03118.x)). The Allen
  mounting medium and its index are not known (implementation handoff,
  "Known gaps"), so that result is not transplanted.
- (d) **A literature lead, not yet usable.** 3-D localisation microscopy
  reads a single fluorophore's axial position from the shape of its
  astigmatic PSF (Huang et al. 2008, **PubMed, abstract only**: the PMC
  record returned no body text, [DOI](https://doi.org/10.1126/science.1153529)).
  Doing so needs a calibrated relation between PSF shape and defocus
  **[reasoning]**; whether that paper's calibration curve carries asymmetric
  terms usable as D-026's "other terms" is **from memory, not verified**.

**What would be needed.** Averaged thin-node profiles from block 10 (symmetry
and tail shape); the full texts of Oberlaender 2009 and Huang 2008 (with its
supplementary material), by upload; a literature search on PSF models and
kernel families when the discussion opens.

**Sources.** **[user]**, 2026-10-06 17:01. Mathematics §3.3, §3.4; procedure
§3.4, §3.10; optics §3.5, §3.7; implementation handoff "Known gaps" (repo,
D-020). Oberlaender M. et al. 2009, *J Microsc* (PMID 19220694) and Huang B.
et al. 2008, *Science* (PMID 18174397, PMC2633023): PubMed, abstract only.
Searches 2026-10-06: PubMed, 3 queries (3-D localisation by astigmatism; PSF
width versus defocus calibration; defocus blur from edge or line-spread
width), returning 2, 0 and 0 records; plus the brightfield PSF query of D-025.

**Status:** idea, to discuss further **[user]**; not scheduled.
