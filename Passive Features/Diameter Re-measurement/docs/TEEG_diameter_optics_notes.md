# Diameter re-measurement: optics and defocus-model notes

| Date | Change |
|---|---|
| 2026-10-03 | v1. Created at the user's request ("keep in the notes this kind of explanation"). Collects the explanations settled in chat 2026-09-30 to 10-03 on the blur in the Allen 63x stacks, the squared-width measure V, why the defocused line-spread function changes shape rather than gaining an added blur, and how the defocus model for the bias table b(d, phi) is calibrated and applied. Nothing here is coded. |
| 2026-10-04 | v2. Corrections marked "[corrected 2026-10-04]" in place, on the evidence of three topic documents written and independently reviewed the same day (`claude/TEEG_diameter_bias_table_procedure_2026-10-04.md`, `claude/TEEG_diameter_bias_table_mathematics_2026-10-04.md`, `claude/TEEG_microscope_optics_oil_immersion_2026-10-04.md`): LSF tails make every second moment window-dependent at every defocus; the rendering kernel is matched through per-plane profiles / core widths, not second moments; the geometric defocus coefficient depends on the angular weighting (0.79-1.21, not 1.2); in stage units the wider tissue cone cancels against the focal shift; the handoff's axial 0.85 um is a first-zero distance; Airy FWHM constant 0.514. New section 9 summarises them; Allen mounting facts added (Gouwens 2019, mouse). Earlier statements kept. |

**Scope.** Companion notes to the user's handoff `handoff_diameter_remeasurement.md`
(2026-09-30, the user's upload, not in project knowledge) and to decisions
**D-018** (background B fixed from a median of the focal plane) and **D-019**
(darkness parameter is mu; tilt phi_i from the local line fit) in
`TEEG_decisions_and_ideas_log.md`. Equation numbers "Eq. n" refer to that handoff.
The full, reviewed treatment is in the three 2026-10-04 documents named above;
these notes stay the running summary.

Tags: **[run]** checked by a script in the assistant's sandbox (scripts named);
**[reasoning]** derivation, hypotheses stated; **[textbook]** from memory;
**[open]** not settled.

---

## 1. Where this sits

The real fit (Eq. 11, amended by D-018/D-019) uses only the **in-focus** blur
sigma. The **defocus model** sigma(dz) is used only to render the synthetic
stacks from which the bias table b(d, phi) is built (Eq. 13): fake 3-D tubes,
each z-slab blurred by its own distance from the focal plane, run through the
real pipeline; b = E[d_hat | d, phi] / d. It exists because a real branch is 3-D:
its parts above and below the focal plane add out-of-focus haze that Eq. 11 does
not model. A simulator built from Eq. 11's own flat model would give b = 1
everywhere and correct nothing. [corrected 2026-10-04: true only for the
**defocus** part of b and only under the handoff's check conditions (correct
sigma, noise-free, no camera chain); b also absorbs every other systematic error
the simulator reproduces -- a sigma_fit mismatch, the B_bar median, pixels, JPEG,
tilt error (mathematics doc section 3.7).]

## 2. The in-focus blur budget (corrects the handoff's Eq. 12 optics line)

- 0.24 um at NA 1.4, lambda = 0.55 um is the Airy **first dark ring**
  (0.61 lambda/NA), not the FWHM; the FWHM is 0.20 um (0.51 lambda/NA)
  **[run, `psf_check.py`]**. [corrected 2026-10-04: the constant is 0.514,
  FWHM 0.202 um.]
- Across a straight branch the acting kernel is the **line-spread function**
  (LSF = PSF summed along the branch). Gaussian-core sigma of the LSF:
  0.083 um (paraxial Airy), 0.080 um (scalar high-NA Debye) **[run]**;
  scales with lambda (0.066-0.095 um for 0.45-0.65 um). [corrected 2026-10-04:
  with the same least-squares core fit the paraxial value is 0.082 um; 0.083/0.084
  was the FWHM-matched value.]
- Sampling chain adds, in variance: pixel integration px^2/12 + bilinear
  interpolation (Eq. 9) px^2/6 = px^2/4 = 0.0033 um^2 at px = 0.1144 um
  (simulated 0.0034); JPEG q75-95 at most 0.0002 um^2 on a synthetic line;
  0.14 um defocus about 0.001 um^2 **[run, `blur_chain_check.py`]**.
- Effective in-focus sigma(0) about 0.099 um at 550 nm (0.088-0.111 over
  450-650 nm). The handoff's "sigma about 0.1 um" is right as an **effective**
  value, for a different reason than stated there. [corrected 2026-10-04: this
  adds variances to a Gaussian-core width, which is not a second moment; it is
  a heuristic budget, not a variance sum.]
- Camera assumptions behind Eq. 11: grey level proportional to light with zero
  offset. A gamma is absorbed into mu; a black-level offset is not **[reasoning]**.
- **Allen acquisition (added 2026-10-04).** Berg 2021 (full text, human):
  AxioImager Z2, Axiocam 506, 0.63x Optivar, oil condenser NA 1.4,
  Plan-Apochromat 63x/1.4 oil at 0.28 um steps; mounting "as described
  previously". Gouwens 2019 (full text, **mouse**): glycerol-based Mowiol or
  Aqua-Poly/Mount, 4.54 um camera pixel, "Tl VIS-LED" lamp; 4.54/(63 x 0.63) =
  0.1144 um. Mounting medium of 529878215 not verified; Allen API specimen and
  treatment records have no mounting fields (data inspected).

## 3. The squared width V of a dip

For a profile I_k(v) across the branch in plane k, with background B_bar:

    w_k(v) = B_bar - I_k(v) >= 0                      (the dip: darkness at v)
    A_k    = integral w_k dv                          (total darkness)
    vbar_k = integral v w_k dv / A_k                  (centre)
    V_k    = integral (v - vbar_k)^2 w_k dv / A_k     (um^2)

- V is a **darkness-weighted mean squared distance from the centre** -- the
  "variance" of the dip treated as a distribution over position. No randomness
  is involved; the word is borrowed because the formula and its addition rule
  are those of a variance. sqrt(V) is an RMS half-width.
- **Why squared distance:** signed distance averages to zero (that is the
  centre); |v| measures spread but does not add under blur; v^2 does.
- **Why divide by A:** (i) V then measures shape, not amount -- B_bar, mu, stain
  concentration and lamp brightness drop out (doubling every w leaves V
  unchanged); (ii) only the normalised moments add: for raw second moments
  M2(w*g) = A_g M2(w) + A_w M2(g) and A(w*g) = A_w A_g, so dividing gives
  V(w*g) = V(w) + V(g). [corrected 2026-10-04: B_bar and lamp brightness drop
  out exactly; mu and stain concentration only in the faint limit, because they
  change the shape of 1 - T (d^2/16 to d^2/12). The raw-moment formula above
  also omits the cross term 2 M1(w) M1(g), which vanishes for centred profiles.]
- **Branch's own V (no blur), R = d/2:** faint (alpha << 1): dip proportional to
  the chord 2 sqrt(R^2 - v^2) (Beer-Lambert linearised: 1 - e^{-mu l} about
  mu l), a semicircle, V = R^2/4 = **d^2/16**; dark (alpha >> 1): dip saturates to
  a box of width d, V = R^2/3 = **d^2/12**; in between, smoothly (d = 0.3 um:
  0.00566 um^2 at alpha 0.1, 0.00703 at alpha 5) **[textbook; run,
  `var_cancel.py`]**.
- l(v) = 2 sqrt(R^2 - v^2) / cos(phi) is the stained path length of the vertical
  ray at offset v (Eq. 10); mu l(v) is the absorbance there.

## 4. What adds, and what does not  <-- the point the user asked to keep

**Each plane sees one kernel, not "in-focus blur + defocus blur".** The dip in
plane k is the tube's own darkness shape convolved with the line-spread function
for that plane's defocus:

    w_k(v) = B_bar * [ (1 - T) * LSF_{dz_k} ](v)

(exact in the single-depth incoherent model, because the kernel integrates to 1:
1 - T*g = (1 - T)*g). Moving to the next plane does **not** put a second blur on
top of the in-focus one: **the whole LSF changes shape** (out of focus it widens,
flattens on top and develops rings and long tails).

**Only the squared widths add:**

    V_k = V_tube + sigma^2(dz_k)
        = V_tube + sigma^2(0) + [ sigma^2(dz_k) - sigma^2(0) ]

- sigma^2(dz_k) is the **total** blur of plane k as a squared spread (um^2) --
  the V of LSF_{dz_k}, whatever its shape -- for a point at distance dz_k from
  the focal plane, measured across the branch only. It includes the in-focus
  part sigma^2(0) (optics + pixels + interpolation).
- "Tube + in-focus blur + extra from defocus" is exact **as a statement about
  squared widths**. It is **not** true of the profiles: LSF_{dz} is the in-focus
  LSF convolved with an extra kernel **only when both are Gaussian**.
- Why V adds: convolution moves each bit of darkness by an independent random
  shift drawn from the kernel; for independent X, Y,
  Var(X+Y) = Var X + Var Y (covariance zero). Holds for any shapes **[textbook]**.
- Plain widths do not add: a faint 0.4 um branch under sigma = 0.10 um has
  V = 0.010 + 0.010 = 0.020 um^2, sqrt = 0.141 um, not 0.1 + 0.1.
- For a round Gaussian PSF the LSF has the same sigma; for the real Airy PSF they
  differ slightly (LSF core 0.080 um vs FWHM-matched PSF 0.086 um, Debye, 550 nm)
  **[run]**.

**When the addition holds on real images:** the branch effectively at one depth
(thin) and imaging linear; the same kernel across the whole profile; second
moments finite and fully captured (real Airy tails make the LSF's second moment
diverge, so on data V depends on the window and on B_bar; a neighbour in the
window adds darkness that is not the branch).

[corrected 2026-10-04: the divergence is not specific to focus. A circular pupil
gives an LSF tail ~ |v|^-2 **at every defocus**, so V[LSF_dz] is infinite for
every dz and "sigma^2(dz)" as a full second moment does not exist. Everything
above holds for **windowed** moments V_W, approximately; the growth is defined as
the limit of windowed differences V_W[LSF_dz] - V_W[LSF_0], finite if the tail
coefficient does not depend on dz (reasoning plus reviewer's runs). In the Debye
model in focus, core sigma 0.080 um vs sqrt(V_W) 0.240 um (W = 1.5 um) and
0.348 um (W = 3 um) **[run, `doc_checks.py` C5]**. The theorem itself
(V(f*g) = V(f) + V(g) for finite moments) stands; mathematics doc section 3.3.]

## 5. Calibrating the defocus model from the stacks (the user's proposal, 2026-10-01)

Idea [user]: at a node, keep (x, y) fixed, step through the neighbouring planes,
watch the dip widen. Because V_tube and sigma^2(0) are the same in every plane,
taking differences leaves only the growth of the blur.

1. **Calibration nodes:** thin (d_hat <~ 0.3 um), faint, flat (phi <~ 10 deg),
   isolated, away from the stack edge, spread over depth in the slice.
2. **Scan:** fixed (x, y) and measuring line; for each plane within about
   +-3 planes, profile I_k(v), dip with B_bar_k (D-018 rule), V_k and A_k, same
   window in every plane.
3. **Fit, shared growth curve, per-node constants:**

        V_{i,k} = c_i + Dsigma2(z_k - z_ax_i),   Dsigma2(0) = 0

   c_i = V_tube + sigma^2(0) absorbs the tube (d never needed); z_ax_i the node's
   axis depth, fitted (better than plugging z_hat from Eq. 2); Dsigma2 shared by
   all nodes, flexible (per-dz bins or kappa^2 dz^2 + lambda dz^4).
   [corrected 2026-10-04: (i) write the argument as z_ax_i - z_k (object minus
   plane), the sign convention of the slab rendering; (ii) the curve is identified
   only up to a common shift of all z_ax_i, pinned by a convention (symmetry, or
   origin at the minimum); (iii) analytic forms extrapolate badly -- tabulate per
   plane offset; (iv) for **matching the rendering kernel**, use the squared
   Gaussian-core width per plane, or the per-plane profiles themselves, not V:
   a Gaussian matched to the windowed-V growth is 1.5-2x too wide in the core
   within one plane of focus (procedure doc section 3.4).]
4. **Checks:** A_k constant across planes (blur moves darkness, does not create
   it -- a change means contamination or a dark tube); above vs below focus
   (asymmetry = index mismatch); shallow vs deep nodes (aberration with depth);
   residuals before trusting beyond +-2 planes. [corrected 2026-10-04: A is
   constant whatever the darkness in the single-depth and partition models, so
   "a dark tube" is not a cause; on data A_W falls once the defocused spot
   outgrows the window. Asymmetry can also come from a phase (refractive-index)
   contrast between tissue and mountant, odd in dz (mathematics doc section 3.5).]
5. **Add the in-focus part:** sigma^2(dz) = sigma^2(0) + Dsigma2(dz), with
   sigma(0) from the Eq. 12 bracket. **The scan cannot give sigma(0):** it
   cancels together with V_tube. [corrected 2026-10-02: an earlier chat answer
   called sigma(0) a "bonus" of this scan.] [corrected 2026-10-04: the bracket
   must be in the same sense of width as the statistic -- the Eq. 12 bracket is a
   core width; for windowed V the optical value is 0.24-0.35 um, not 0.08.]

**Measuring V_k -- a practical choice [run, `var_cancel.py`]:** with a Gaussian
blur, the raw second moment cancels the tube exactly (ratio 0.999 for d 0.3-1 um,
alpha 0.1-2, sigma 0.15-0.4 um); a fitted Gaussian width^2 is robust to tails but
cancels only approximately: 0.96-0.99 at d = 0.3 um, 0.85-0.98 at 0.5 um,
0.61-0.89 at 1 um. Hence the thin-node restriction.

**The quadratic law is not reliable even within one plane:** Debye LSF core
sigma = 0.080, 0.086, 0.122 um at dz = 0, 0.14, 0.28 um; a parabola through the
ends predicts 0.092 at 0.14 **[run]**. Tabulate or use a flexible curve.

**Why each restriction:** thin -- the Gaussian-fit cancellation and the
single-depth assumption; isolated -- neighbours' haze grows with dz and does not
cancel; flat -- along a tilted branch nearby stretches sit at other depths (the
tilt halo, bias (b)); faint -- see section 6.

[corrected 2026-10-02: an earlier answer said the widths add "for a faint
process". In the single-depth model they add for **any** darkness; darkness only
moves V_tube between d^2/16 and d^2/12, and V_tube cancels.]

## 6. Applying the model to tubes that span several planes

The single-depth assumption binds the **calibration** only. In the simulator a
thick or steep tube is cut into z-slabs and each slab's shadow is blurred with
the kernel for its own distance from the focal plane, sigma(z_slab - z_focus);
that spread of blur across depth is exactly the halo b corrects.

- Faint tube: slab images add linearly (weak-object approximation); the thickness
  term kappa^2 <zeta^2> (zeta = height above the axis, <zeta^2> = d^2/16) then
  cancels in the calibration differences too, but only if sigma^2 grows exactly
  quadratically **[reasoning, not simulated]**.
- Dark thick tube: light absorbed in one slab never reaches the next, so adding
  slab darkness over-counts. Workaround: sum the blurred absorbance (mu x
  thickness) of the slabs, then apply Beer-Lambert -- also an approximation,
  **not checked [open]**. Part of the handoff's open choice of defocus model.
  [superseded 2026-10-04: the proposal now is the absorbed-light partition,
  Delta A_j = T_<j (1 - e^{-a_j}), each slab's Delta A_j blurred by its own kernel
  and summed (procedure doc Eqs. 5-6). It conserves Beer-Lambert, is exact for
  one slab and to first order when faint, and is a heuristic between; with nine
  slabs at centre-line absorbance 1.5 the linear sum overstates the dip by 86 %
  **[run, `doc_checks.py` C3]**. Recorded planes are never blurred and added.]
- Far from focus a Gaussian is a poor kernel (Zhang, Zerubia & Olivo-Marin 2007:
  no accurate Gaussian approximation exists for the 3-D widefield PSF --
  **PubMed, abstract only**, [DOI](https://doi.org/10.1364/ao.46.001819)); an
  empirical kernel (averaged measured profiles per dz) avoids the assumption.
  [added 2026-10-04: empirical kernels must be made 2-D (inverse Abel, assuming
  rotational symmetry), stripped of the sampling chain, corrected for V_tube,
  and registered to dz from z_ax (procedure doc section 3.4).]

## 7. Related points settled in the same chat

- **mu and lamp brightness (D-019):** brightness cancels through I/B_bar = T;
  mu = absorption per um of the stain under this illumination -- depends on stain
  amount and on the spectrum (DAB is coloured; broadband light deviates from a
  single exponential, matters if mu is shared) **[reasoning]**. Fit mu per node;
  share only where mu_hat is uniform (D-019 open point (a)).
- **Half-maximum width** mixes d with blur and darkness: in Eq. 11's own model
  FWHM/d ranges 0.83-1.07 at sigma = 0.10 um over d 0.3-2 um, alpha 0.1-3
  **[run, `fwhm_mix.py`]**; hence the full-profile fit.
- **b multiplicative vs additive:** because b is tabulated over d, any bias is
  representable; a constant offset c gives b = 1 + c/d, quadrature blur
  b = sqrt(1 + c^2/d^2), a gain a constant b. Choose the form that makes the
  correction flattest across d **[textbook]**.
- **Phantom draws:** sub-pixel lateral position (recovered by v0_hat, Eq. 11) and
  axis depth between planes, uniform within +-dz/2 = +-0.14 um (recovered by the
  Eq. 2 parabola, which feeds only the tilt).

## 8. Open

Region of the B_bar median (D-018 (a)); mu per node vs shared (D-019 (a));
defocus model form and slab combination for dark tubes (section 6); sigma(0)
value inside the Eq. 12 bracket; Allen's JPEG tables (readable from any crop with
PIL's `Image.open(f).quantization`) and camera offset/gamma; nothing coded.
Added 2026-10-04: mounting medium and its index for 529878215; the condenser
aperture-diaphragm setting; whether dendrites are round or z-flattened in the
mounted slice (round phantoms assumed); the selection rule S for fits counted in
b; the unidentified in-focus core sigma_r(0) and effective wavelength.

## 9. Corrections and additions of 2026-10-04 (summary)

Evidence: the three 2026-10-04 documents and their independent review.

- **Geometric defocus coefficient.** [corrected 2026-10-04] Chat answers used
  "sigma ~ 1.2 |dz| far from focus" (= tan(theta_obj)/2, a uniformly bright
  disc). The disc is not uniformly bright: a bundle of steep rays lands on an
  area growing as 1/cos^3; the RMS coefficient depends on the angular weighting:
  1.21 (uniform disc), 0.90 (isotropic), 0.79 (Debye's sqrt(cos) weighting,
  irradiance ~ cos^4). The Debye LSF at dz = 1-3 um gives 0.79-0.86 |dz|
  (reviewer's run). Only sigma ~ |dz| far from focus is robust.
- **Which cone.** In transmitted brightfield an absorber's shadow cone is set by
  condenser **and** objective; with the condenser diaphragm partly closed it is
  narrower than theta_obj = 67.5 deg (optics doc section 3.5).
- **Index mismatch in stage units.** [corrected 2026-10-04] A lower-index
  mountant widens the cone **in the tissue**, but per micrometre of **stage**
  travel the geometric growth is set by the angles in oil: the wider cone and
  the focal shift cancel. What remains is depth-dependent spherical aberration
  and a focal shift of absolute depths (Kner 2010, McGorty 2014, full text,
  fluorescence). If n_m < 1.4 the growth per stage um is slower (0.92 vs 1.21 at
  n_m = 1.33, uniform-disc convention).
- **Axial extent.** [corrected 2026-10-04] The handoff's "axial ~0.85 um" is the
  paraxial first-zero distance 2 n lambda / NA^2; the ideal high-NA first minimum
  is ~0.60 um; axial FWHM 0.53 um (Debye) vs 0.75 um (paraxial) **[run,
  `axial_check.py`]**.
- **Incoherent form.** Even at condenser NA = objective NA, I = B (T * h) holds
  only to first order in the absorbance (weak object); dark DAB dendrites exceed
  it (optics doc section 3.6, textbook from memory).
- **Bias-table statistics.** b is conditional on the configuration C and on the
  selection S of fits counted; correct by solving m_hat(d, phi_i | C) = d_hat_i;
  the handoff's shortcut d_hat/b(d_hat) is off by about -beta (b - 1), beta =
  dln b / dln d; per-node scatter after correction ~ tau / [b (1 + beta)]
  (mathematics doc section 3.6, procedure doc section 3.8).

## Sources

The user's handoff (Eqs. 1-13); D-018, D-019. Sandbox scripts (2026-09-30 to
10-02, not persisted to the project): `psf_check.py`, `blur_chain_check.py`,
`fwhm_mix.py`, `mu_tie.py`, `var_cancel.py` **[run]**. Textbook, from memory:
Airy pattern, px^2/12, FWHM = 2.355 sigma, moments under convolution,
Beer-Lambert. PubMed searches 2026-09-30 to 10-01 (PSF-convolved cylinder fits,
3-D brightfield image formation, DAB and Beer-Lambert): no full text used;
Streekstra & van Pelt 2002 (PMID 12222820) and Zhang et al. 2007 abstract only.
Added 2026-10-04: scripts `doc_checks.py`, `defocus_forms.py`, `axial_check.py`
**[run]**; reviewer's `rev_tail.py`, `rev_tail2.py`, `rev_geo.py`, `ax2.py`;
PubMed full text: Berg 2021 (PMC8494638), Gouwens 2019 (PMC8078853), Kner 2010
(PMC2897157), McGorty 2014 (PMC4030053); Allen API specimen 529878215 and
treatment 680087734 records (data inspected).
