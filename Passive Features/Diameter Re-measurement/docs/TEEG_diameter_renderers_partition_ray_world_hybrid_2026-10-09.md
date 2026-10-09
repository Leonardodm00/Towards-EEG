# The partition and the ray world: two renderers of a stained tube, and how to merge them

**Date:** 2026-10-09. **Project:** Towards EEG, diameter re-measurement from the
Allen 63× brightfield stacks (specimen 529878215). **Written by:** the theory chat
(claude.ai session `session_017oQJ14njZ9i5nSBhHUDoMh`), at the user's request.
**Status:** a theory document; it records no decision. The bias table's renderer is
still the partition (D-024 (i), "for now"). The hybrid of §3.8 is the user's
proposal of 2026-10-09; it is not yet logged as a decision or an idea.

| Date | Change |
|---|---|
| 2026-10-09 | v1. Written at the user's request. Before the first commit two independent review passes checked it against the committed check outputs, the companion documents, the branch code and the project log, and their corrections are built in. Three of them correct statements this chat made earlier the same day: the dark label is D-030's comment (c), not (a) (§3.5); the production kernel is narrower than the cone beyond about 0.1 µm of defocus, not "near focus" (§3.8); and the diffraction-limited spot does not grow more slowly than the cone's between 0.15 and 0.5 µm of defocus: it is narrower there, but from about 0.28 µm it grows faster and nearly catches up at 0.56 µm (§3.6). `checks/hybrid_absorption_check.py` gained the sections that §3.8–3.9 cite: where the hybrid books the light, its total against the ray world's, the opaque leak under both weightings, and the kernel split. |

**Companion documents** (this folder):
- design handoff `handoff_diameter_remeasurement.md` (2026-09-30, the user's upload;
  "handoff Eq. n");
- procedure `TEEG_diameter_bias_table_procedure_2026-10-04.md` ("procedure Eq. n");
- mathematics `TEEG_diameter_bias_table_mathematics_2026-10-04.md` ("mathematics
  Eq. n");
- optics `TEEG_microscope_optics_oil_immersion_2026-10-04.md` ("optics §n");
- implementation handoff `TEEG_diameter_implementation_handoff_2026-10-06.md`
  ("(S n)");
- theory handoff `TEEG_diameter_theory_handoff_2026-10-07.md` ("Open work n");
- study notes `TEEG_diameter_study_notes_2026-10-07.md`.

Check scripts are in `../checks/`.

**Abstract.** Every node's fitted diameter is corrected with a bias table. The table
is built by rendering synthetic stacks of tubes of known diameter and fitting them
with the real pipeline (procedure §3.1–3.8), so it is only as good as its renderer.
The project has two renderers: the **partition** (procedure Eqs. 5–6), which builds
the table, and the **ray world** ((S6)), a geometric-optics reference that never
does. This document answers three questions:
- How exactly do the two renderers differ?
- What does each get right and wrong, and what does that do to the diameters of
  thick and dark dendrites?
- Can they be merged into a **hybrid** that keeps the ray world's treatment of
  absorption and the real microscope's measured blur?

The findings, in order:
- Given the same rays, the two renderers are one formula with two substitutions
  (§3.4).
- The partition's substitutions are harmless in faint stain once two things are
  matched, and fail in dark stain (§3.5).
- The ray world has no diffraction, so it blurs too much near focus: from 0.14 µm of
  defocus on, its blur is wider than the diffraction-limited kernel, by a factor of
  about 2 near 0.28 µm. That rules it out as the table's renderer: even for 2–3 µm
  faint dendrites ($\mu d=0.5$) its table would leave the diameters 4–5 % too small
  (§3.6).
- Comparing the two renderers measures absorption only if they share the blur
  (§3.7).
- The hybrid's bookkeeping is exact to first order: with the kernel that faint nodes
  would calibrate in a ray-world microscope, it equals the ray world at first order.
  At the centre of a flat 1 µm tube in the node plane, its dip stays within 0.03 of
  the ray world's up to an optical depth of about 3. Near the opaque limit its centre
  is too light, as the partition's is. It books each direction's light where that
  direction enters the tube and then spreads it without the direction, which moves
  light without losing any. Keeping the direction would need a kernel at least as
  wide as the cone, and the production kernel is narrower than the cone beyond about
  0.1 µm of defocus (§3.8–3.9).

**Excluded:**
- coherence and the weak-object approximation (pointer in §3.10; Open work 2);
- the camera chain;
- the kernel calibration itself (D-026, D-027);
- the code of the hybrid, which is named here but not specified.

All numbers describe ideal, index-matched optics at 550 nm, noise-free, without the
camera chain. They show mechanisms and sizes, not values of the bias.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $\mathbf x=(x,y)$ | lateral position in the specimen plane | $\mathbb R^2$ | µm | 3.1 |
| $z$; $\zeta$ | depth in stage units, increasing in the direction the light travels; $\zeta$ a depth inside the object | $\mathbb R$ | µm | 3.1; 3.8 |
| $k$, $z_k$ | index of an output (image) plane; its depth | $k\in\mathbb Z$; $z_k\in\mathbb R$ | —; µm | 3.1 |
| $d$, $r$ | tube diameter and radius $r=d/2$; model level (the phantom's true values) | $\mathbb R_{>0}$ | µm | 3.1 |
| $\varphi$, $\theta$ | tilt and heading of the tube axis | $\varphi\in[0,\pi/2)$; $\theta\in[0,\pi)$ | rad | 3.1 |
| $\mu$ | absorption coefficient of the stain; model level | $\mathbb R_{\ge0}$ | µm⁻¹ | 3.1 |
| $B$ | background intensity | $\mathbb R_{>0}$ | grey levels | 3.1 |
| $I_k$; $I^{\rm P}_k$, $I^{\rm ray}_k$, $I^{\rm hyb}_k$ | intensity of plane $k$: the real one (physics level, recorded by the stacks); rendered by the partition, the ray world, the hybrid (model level) | functions $\mathbb R^2\to[0,B]$ | grey levels | 3.1 |
| $I^{\rm P}_k[K]$ | the partition's plane $k$ with the kernel family $K$ named in the symbol, where two kernels are compared; elsewhere $I^{\rm P}_k$ carries the kernel stated in the text | function $\mathbb R^2\to[0,B]$ | grey levels | 3.7 |
| $i$ | index of a real node | $i\in\mathbb N$ | — | 3.1 |
| $\hat d_i$, $\hat\mu_i$, $\varphi_i$, $\hat\alpha_i$ | real node $i$'s fitted diameter, its fitted absorption coefficient, its tilt from the line fit, its fitted optical depth $\hat\mu_i\hat d_i/\cos\varphi_i$; computed level | $\mathbb R_{>0}$; $\mathbb R_{\ge0}$; $[0,\pi/2)$; $\mathbb R_{\ge0}$ | µm; µm⁻¹; rad; dimensionless | 3.1 |
| $\hat m(d,\varphi,\hat\alpha\mid\mathcal C)$ | the bias table: mean fitted diameter of phantoms of true $d$ and $\varphi$ at fitted optical depth $\hat\alpha$, for each fixed configuration $\mathcal C$ (D-031); computed level | function into $\mathbb R_{>0}$ | µm | 3.1 |
| $\mathcal C$ | configuration of estimator and renderer (procedure §3.3) | set of settings | — | 3.1 |
| $n$, $\hat d_n$ | index of a phantom replicate; its fitted diameter; computed level | $n\in\mathbb N$; $\hat d_n\in\mathbb R_{>0}$ | —; µm | 3.1 |
| $\hat\tau$ | local spread of $\hat d_n/d$ among phantom replicates (procedure Eq. 7); computed level | $\mathbb R_{\ge0}$ | dimensionless | 3.1 |
| $j$, $j'$, $\zeta_j$, $\delta\zeta$, $J$ | index of a slice of the object, numbered in the direction the light travels; an index over the slices before $j$; slice $j$'s centre depth; the slice thickness; the number of slices | $j,j'\in\{1,\dots,J\}$; $\zeta_j\in\mathbb R$; $\delta\zeta>0$; $J\in\mathbb N$ | —; —; µm; µm; — | 3.2 |
| $\delta$ | defocus of a slice relative to a plane, $\delta=\zeta_j-z_k$ | $\mathbb R$ | µm | 3.2 |
| $a_j(\mathbf x)$ | absorbance of slice $j$ along the vertical through $\mathbf x$: the coefficient times the length of that vertical line inside both the slice and the tube ((S4)) | $\mathbb R_{\ge0}$ | dimensionless | 3.2 |
| $T_{<j}(\mathbf x)$ | vertical history: the fraction of light left on the vertical line through $\mathbf x$ before slice $j$ | $(0,1]$ | dimensionless | 3.2 |
| $\Delta A_j(\mathbf x)$ | absorbed fraction booked to slice $j$ at $\mathbf x$ by the partition | $[0,1)$ | dimensionless | 3.2 |
| $K_\delta$ | defocus kernel at defocus $\delta$; in production a circular Gaussian of per-axis standard deviation $\sigma_{\rm r}(\delta)$ | unit-area function $\mathbb R^2\to\mathbb R_{\ge0}$ | µm⁻² | 3.2 |
| $\sigma_{\rm r}(\delta)$ | the production kernel's width: the ideal-Debye core width (optics §3.5), continued linearly beyond 0.84 µm | $\mathbb R_{>0}$ | µm | 3.2 |
| $*$ | two-dimensional convolution in $\mathbf x$ (one-dimensional for profiles) | operator | — | 3.2 |
| $T$, $g$ | the specimen's transmittance along vertical lines, a function of lateral position (in the fit, its profile across the branch), and the fit's Gaussian blur (handoff Eq. 11, D-019.1) | $T$ into $[0,1]$; $g$ unit-area, into $\mathbb R_{\ge0}$ | dimensionless; µm⁻¹ (profile) | 3.2 |
| $c=(c_x,c_y,c_z)$ | the node point on the tube axis; $c_z$ is the axis depth there | $\mathbb R^3$ | µm | 3.3 |
| $\hat s=(s_x,s_y,s_z)$ | unit direction of an illuminating ray, $s_z>0$ | upper unit hemisphere | dimensionless | 3.3 |
| $\rho$ | radial position of a direction in the aperture, $\rho=\sqrt{s_x^2+s_y^2}$ | $[0,s_m]$ | dimensionless | 3.3 |
| $\vartheta$ | polar angle of $\hat s$, $\cos\vartheta=s_z$ | $[0,\vartheta_{\max}]$ | rad | 3.3 |
| $\mathrm{NA}$, $n_{\rm oil}$, $s_m$ | numerical aperture of the condenser cone (1.4); immersion-oil index (1.515); $s_m=\mathrm{NA}/n_{\rm oil}=0.924$ | $\mathbb R_{>0}$ | dimensionless | 3.3 |
| $\vartheta_{\max}$ | largest ray angle, $\sin\vartheta_{\max}=s_m$, so $\vartheta_{\max}=67.5°$ | $(0,\pi/2)$ | rad | 3.3 |
| $W$ | illumination measure: directions with $(s_x,s_y)$ uniform on the disc of radius $s_m$ (evenly filled aperture, sine condition) | probability measure on the cone | — | 3.3 |
| $\langle f\rangle_W$, $\langle f\rangle_{W^*}$ | average of $f$ over directions under $W$, under $W^*$ | — | units of $f$ | 3.3 |
| $L(\mathbf x,z_k,\hat s)$ | length inside the tube of the line through $(\mathbf x,z_k)$ with direction $\hat s$ | $\mathbb R_{\ge0}$ | µm | 3.3 |
| $\mathbf t_{\hat s}$ | lateral slope of a ray, $(s_x,s_y)/s_z$ | $\mathbb R^2$ | dimensionless | 3.4 |
| $\mathbf x_j(\mathbf x,\hat s)$ | where the ray through $(\mathbf x,z_k)$ with direction $\hat s$ crosses slice $j$: $\mathbf x+(\zeta_j-z_k)\,\mathbf t_{\hat s}$; for each fixed plane $k$, which the symbol suppresses | $\mathbb R^2$ | µm | 3.4 |
| $T^{\rm ray}_{<j}(\mathbf x,\hat s)$ | ray history: the fraction of light left along that same ray before slice $j$; for each fixed plane $k$, which the symbol suppresses (shorthand) | $(0,1]$ | dimensionless | 3.4 |
| $K^{\rm cone}_\delta$, $K^{{\rm cone}*}_\delta$ | "kernel of the same rays", or cone kernel: the law of $\delta\,\mathbf t_{\hat s}$ under $W$, and under $W^*$ | probability measures on $\mathbb R^2$ | — | 3.4 |
| $t_j$ | generic factor in the telescoping identity; here a slice's transmission along a ray | $(0,1]$ | dimensionless | 3.4 |
| $f$ | any function of the direction $\hat s$ | function on the cone | any | 3.5 |
| $\alpha$ | optical depth on the vertical ray through the axis, $\mu d/\cos\varphi$; model level ($\hat\alpha$ is its estimate) | $\mathbb R_{\ge0}$ | dimensionless | 3.5 |
| $s_v$ | component of $\hat s$ across the branch | $[-s_m,s_m]$ | dimensionless | 3.5 |
| $W^*$ | obliquity-weighted measure, $dW^*=\dfrac{(1/\cos\vartheta)\,dW}{\langle1/\cos\vartheta\rangle_W}$ | probability measure on the cone | — | 3.5 |
| $\mu d$ | optical depth on the vertical through the axis of a flat tube ($\alpha$ at $\varphi=0$) | $\mathbb R_{\ge0}$ | dimensionless | 3.5 |
| $\mu^{(1)}_{\rm ph}$ | first-order phantom coefficient for a vertical-path renderer, $\langle1/\cos\vartheta\rangle_W\,\mu=1.447\,\mu$; analytic | $\mathbb R_{\ge0}$ | µm⁻¹ | 3.5 |
| $\mu_{\rm ph}$ | phantom coefficient at which a renderer's fitted $\hat\mu$ equals the reference's, found by root search ("$\mu$ matched through $\hat\mu$", procedure §3.5); a renderer input whose value is computed from fits | $\mathbb R_{\ge0}$ | µm⁻¹ | 3.5 |
| $I^{\rm P*}_k$ | the partition's plane $k$ with the kernel $K^{{\rm cone}*}$ and the coefficient $\mu^{(1)}_{\rm ph}$; model level | function $\mathbb R^2\to[0,B]$ | grey levels | 3.5 |
| $\omega(\vartheta)$ | angular weighting of the rays per unit solid angle (mathematics Eq. 17): $W$ is $\omega\propto\cos\vartheta$ (aplanatic), $W^*$ is $\omega$ constant (isotropic) | function $[0,\vartheta_{\max}]\to\mathbb R_{\ge0}$ | — | 3.5 |
| $\gamma_W$, $\gamma_{W^*}$ | per-axis RMS lateral slope of the rays under $W$ (0.7915) and under $W^*$ (0.8991), so that the cone kernel's per-axis RMS spread is $\gamma_W\lvert\delta\rvert$ or $\gamma_{W^*}\lvert\delta\rvert$; mathematics Eq. 17's $\gamma_\omega$. Always written with its subscript here, because D-024 and `config.py` use a bare $\gamma$ for the kernel table's continuation slope (0.79), a configured parameter | $\mathbb R_{>0}$ | dimensionless | 3.5 |
| $\hat d$, $\hat\mu$ | diameter and absorption coefficient returned by the fit (D-019.1); computed level | $\mathbb R_{>0}$; $\mathbb R_{\ge0}$ | µm; µm⁻¹ | 3.5 |
| $\Delta\hat d/d$ | a renderer's $\hat d/d$ minus the reference's, same phantom, same fit | $\mathbb R$ | dimensionless | 3.5 |
| G, P, P\*, Gv\*, P\*s, Gv\*s, H | variants in the check scripts; table below | labels | — | 3.5 |
| $D_c$ | centre dip of a rendered plane: $1-I^{\bullet}_k(\mathbf 0)/B$ at the centre of a flat tube, in the plane through its axis, for the renderer $\bullet$ named with it; model level | $[0,1]$ | dimensionless | 3.5 |
| $v$, $z_{\rm lo}(v)$, $z'_{\rm lo}(v)$, $h$ | across-branch coordinate; depth of the tube's lower skin at $v$; its slope ${\rm d}z_{\rm lo}/{\rm d}v$; height of a point above the skin of the column it is in | $\mathbb R$ | µm; µm; dimensionless; µm | 3.5 |
| $v_e$ | across-branch position at which a ray enters through the lower skin | $(-r,r)$ | µm | 3.5 |
| $R_{\rm disc}(\delta)$ | radius at which the cone's outermost rays land: the edge of the geometric defocus spot | $\mathbb R_{\ge0}$ | µm | 3.6 |
| $u$ | coordinate along the tube's horizontal projection ((S1)) | $\mathbb R$ | µm | 3.6 |
| $\ell(v)$ | length of the vertical chord through the tube at across-position $v$ ((S3)) | $\mathbb R_{\ge0}$ | µm | 3.6 |
| $\sigma_{\rm fit}$, $\bar B$ | the fit's fixed blur width; its background estimate (D-018, D-023) | $\mathbb R_{>0}$ | µm; grey levels | 3.6 |
| $\lambda$ | vacuum wavelength | $\mathbb R_{>0}$ | µm | 3.6 |
| $\Delta\phi_{\max}(\delta)$ | largest defocus phase error across the pupil, for either sign of $\delta$ | $\mathbb R_{\ge0}$ | rad | 3.6 |
| $\hat b$; $\hat b_{\rm diff}$, $\hat b_{\rm ray}$ | a renderer's $\hat d/d$ for phantoms of true $d$, the fit's bias in that renderer's world (in the table, the mean over replicates; in the noise-free runs here, a single fit); for the diffraction-limited partition and for the ray world; computed level | $\mathbb R_{>0}$ | dimensionless | 3.6 |
| $\tilde d$ | corrected diameter (procedure Eq. 1) | $\mathbb R_{>0}$ | µm | 3.6 |
| $\ell_{\rm back}(\mathbf x,\zeta;\hat s)$ | path length inside the tube from the point $(\mathbf x,\zeta)$ back along $-\hat s$ to where that ray entered the tube | $\mathbb R_{\ge0}$ | µm | 3.8 |
| $\Delta A^{\rm hyb}_j(\mathbf x)$ | absorbed fraction booked to slice $j$ at $\mathbf x$ by the hybrid | $[0,1)$ | dimensionless | 3.8 |
| $\mathbf q$, $p$, $\mathbf e$, $\ell^{\rm 2D}_{\rm back}$ | for a flat tube: a point's position in the across-branch plane relative to the axis; the length of the projection of $\hat s$ onto that plane, $p=\sqrt{s_v^2+s_z^2}$; the backward direction projected onto that plane, as a unit vector; the in-plane distance back to the tube's boundary | $\mathbb R^2$; $(0,1]$; unit vector in $\mathbb R^2$; $\mathbb R_{\ge0}$ | µm; dimensionless; —; µm | 3.8 |
| $\kappa$, $\kappa_{\hat s}$ | residual blur of a hypothetical ray-resolved kernel: one for all rays, or one per ray $\hat s$, centred on that ray's landing point | non-negative unit-area functions $\mathbb R^2\to\mathbb R_{\ge0}$ | µm⁻² | 3.8 |
| $\sigma^2_{K_\delta}$, $\sigma^2_{\rm cone}(\delta)$, $\sigma^2_\kappa$ | per-axis variances of $K_\delta$, of the cone kernel at defocus $\delta$, and of $\kappa$ (for per-ray residuals, the average over rays of the variances of $\kappa_{\hat s}$) | $\mathbb R_{\ge0}$ | µm² | 3.8 |
| $\Delta\hat\mu$ | a renderer's fitted $\hat\mu$ minus the reference's, same phantom | $\mathbb R$ | µm⁻¹ | 3.9 |
| $S$ | coherence parameter: condenser NA over objective NA | $[0,1]$ | dimensionless | 3.10 |
| $h_0$ | the objective's intensity point-spread function, of unit area | function $\mathbb R^2\to\mathbb R_{\ge0}$ | µm⁻² | 3.10 |

**Variants in the check scripts.** Each is evaluated on the same direction quadrature
as the ray world it is compared with.

| label | what is rendered | rays, weights, kernel | coefficient | script (printed label) |
|---|---|---|---|---|
| G | the ray world, (3) | the cone, weights $W$ | true $\mu$ | every ray-world check (G) |
| P | the partition, (5) | the same rays: kernel $K^{\rm cone}$, weights $W$ | true $\mu$ | `optics_points_check.py` BC (P) |
| P\* | the partition, (5) | kernel $K^{{\rm cone}*}$, weights $W^*$ | $\mu_{\rm ph}$, matched | `optics_points_check.py` matched (P\*) |
| Gv\* | the true ray history with vertical path elements | weights $W^*$ | its own $\mu_{\rm ph}$, matched | `optics_points_check.py` matched (Gv\*) |
| P\*s | as P\* | weights $W^*$ | true $\mu$, no matching | `followup_checks.py` (P\*); `hybrid_absorption_check.py` (P\*s) |
| Gv\*s | as Gv\* | weights $W^*$ | true $\mu$, no matching | `followup_checks.py` (Gv\*); `hybrid_absorption_check.py` (Gv\*s) |
| H | the hybrid, (13)–(14) | kernel $K^{{\rm cone}*}$ | true $\mu$ | `hybrid_absorption_check.py` (H) |

### Conventions

- Depth $z$ is in stage units and increases in the direction the light travels, from
  the condenser to the objective. Slices are numbered in the same direction. How
  Allen's plane index relates to this direction is not verified (configuration field
  `light_direction`).
- Intensities are in grey levels; $I/B$ is dimensionless.
- $O(a^2)$ denotes terms of second order in the stain's absorbance: in $\mu$ at fixed
  geometry, equivalently in the slice absorbances $a_j$ at fixed slicing.
- Every statement about the ray world, and about the hybrid's absorption, assumes
  geometric optics, an evenly filled NA 1.4 condenser cone, the sine condition and an
  index-matched specimen ((S6)).
- Statements about the partition "with the production kernel" use the ideal-Debye
  core table (mathematics §3.4), continued linearly beyond 0.84 µm with slope 0.79
  (D-024 (i)).
- Levels (precision rule R8): $d$, $\mu$ are model-level truths of a phantom;
  $\hat d$, $\hat\mu$, $\hat b$ are computed-level outputs of a fit. Wherever a check
  calls the ray world "the truth", the ray world stands in for the real microscope.
  It is a model, not a measurement.
- Letters that look alike but name unrelated objects: $h$ (a height above the skin,
  §3.5) and $h_0$ (a point-spread function, §3.10); $T$ (the fit's transmittance
  profile), $T_{<j}$ and $T^{\rm ray}_{<j}$ (histories), and $t_j$ (a factor in an
  identity).
- Source tags:
  - **[run]**: from a committed `checks/*.out`, script named;
  - **[src]**: from code on `sci/diameter-pipeline`, read at `3084991` and rechecked
    at `8a11f1e` (2026-10-09; the model and configuration files are unchanged
    between the two, and the SPEC lines cited here read the same);
  - **[reasoning]**: derived in this document;
  - **[arithmetic]**: a direct computation;
  - **[KB-repo]**: from a project document in this folder;
  - **[PubMed, full text]**: read in full through PubMed Central;
  - **[textbook]**: standard material, from memory (listed in §6).

---

## 2. Glossary

Ordered by first appearance, because the concepts build on each other.

- **Renderer** (§3.1). Code that maps a phantom tube and a plane depth to a synthetic
  image plane. Its output goes through the same pipeline as real data, so the table
  is the fit's bias in the renderer's world.
- **Slice** (§3.2). A thin horizontal layer of the object, at most 0.05 µm thick; the
  procedure calls it a "slab". It is not an image plane: every slice contributes to
  every plane, each time with a different blur.
- **Absorbance vs absorbed fraction** (§3.2). Absorbance is an exponent, additive
  along a path. The absorbed fraction is a share of the incident light, in $[0,1]$.
  The two coincide only for faint stain (mathematics §3.2).
- **History** (§3.2). The fraction of light still present when the light reaches a
  slice. The partition takes it along the vertical; the ray world takes it along the
  ray.
- **Partition** (§3.2). The default renderer, procedure Eqs. 5–6, also called the
  "absorbed-light partition". It books each slice's absorbed fraction using the
  vertical history, blurs it by the kernel for its defocus, and sums the slices.
- **Defocus kernel** (§3.2). How the shadow of a point at defocus $\delta$ spreads in a
  plane. In production it is a Gaussian with tabulated width: the ideal Debye table
  for now, measured after D-026 and D-027.
- **Ideal-Debye table** (§3.2). Core widths of the line-spread function of an ideal,
  index-matched NA 1.4 objective at 550 nm, computed from the Debye diffraction
  integral (mathematics §3.4). Illustrative, not calibrated.
- **Ray world** (§3.3). The geometric-optics reference renderer: Beer–Lambert along
  every ray of the condenser cone through the image point, averaged over the cone
  ((S6)). On the branch it is `absorption = "ray_world"` (Block 9).
- **Geometric optics** (§3.3). Light described as rays. It is exact in the limit of
  zero wavelength and has no diffraction or interference.
- **Sine condition, evenly filled aperture** (§3.3). The assumption that the
  illuminating directions are uniform in $(s_x,s_y)$ over the aperture disc, which is
  the aplanatic weighting (mathematics Eq. 17).
- **Block** (§3.3). A unit of the implementation in `specs/SPEC.md`: Block 2 is the
  tube geometry, Block 4 the renderer, Block 9 the ray world.
- **Gauss–Legendre quadrature** (§3.3). A standard rule for numerical integration; here
  used to average over directions.
- **Post-blur** (§3.3). The 0.08 µm Gaussian the ray world applies afterwards, as a
  stand-in for the in-focus diffraction core that geometric optics lacks.
- **Telescoping identity** (§3.4). $1-\prod_jt_j=\sum_j\big(\prod_{j'<j}t_{j'}\big)(1-t_j)$:
  the total loss is the sum, over steps, of each step's loss times what reached that
  step.
- **Kernel of the same rays** (§3.4). The blur the cone's own rays produce: a point at
  defocus $\delta$ spreads to the displacements $\delta\,\mathbf t_{\hat s}$, one per
  ray (the sign is immaterial, because the cone is symmetric). Also called the cone
  kernel.
- **Obliquity weighting** (§3.5). An oblique ray crosses more stain, a path longer by
  $1/\cos\vartheta$, so in faint stain it carries more of the shadow. $W^*$ weights the
  directions accordingly.
- **$\mu$ matched through $\hat\mu$** (§3.5). Choosing the phantom's $\mu$ so that the
  fit returns the same $\hat\mu$ as on the reference (procedure §3.5). It absorbs the
  first-order mismatch of path lengths.
- **Matched comparison** (§3.5). The checks' P\*: the partition with the cone kernel
  weighted by $W^*$ and $\mu$ matched through $\hat\mu$, compared with the ray world.
- **Node plane** (§3.5). The image plane through the tube's axis depth at the node.
- **Centre dip** (§3.5). $1-I/B$ at the tube's centre in the node plane.
- **Opaque limit** (§3.5). $\mu\to\infty$: all the light that enters the tube is
  absorbed where it enters. The partition, which follows vertical columns, books it
  at the tube's lower skin.
- **Leak, skin-crossing factor** (§3.5). In the opaque limit the partition books, along
  a ray, the factor (8) instead of 1 ((S9)). It is less than 1 for every ray through
  the centre with $s_v\ne0$, so light appears to pass through an opaque tube, and more
  than 1 for a ray whose lateral motion follows the skin's rise. Column sums stay
  conserved.
- **Diffraction** (§3.6). The spreading of light by the finite aperture. It sets the
  in-focus spot (the Airy pattern) and keeps it nearly constant within about
  ±0.15 µm of focus.
- **Airy pattern** (§3.6). The in-focus image of a point through a circular aperture:
  a bright core with faint rings. At NA 1.4 and 550 nm its first dark ring has a
  radius of 0.24 µm (optics §3.4).
- **Quarter-wave tolerance** (§3.6). The defocus at which the largest phase error
  across the pupil reaches $\pi/2$ (optics Eq. 6a). Within it the spot is essentially
  the in-focus one.
- **Like-for-like comparison** (§3.7). A comparison in which two renderers share
  everything except the approximation under test.
- **Inverse crime** (§3.7). Testing an inversion on data simulated with the same
  forward model it inverts, which leaves that model's own error out of the test.
  Paz-Linares et al. 2017 attribute the term to Kaipio and Somersalo; Hofmann et al.
  2022 use it too (§6).
- **Gate 2** (§3.7). The implementation's reduced end-to-end test of the correction
  (procedure §3.10, last row): a table, then independent-seed test phantoms
  corrected within 10 %. It was passed on 2026-10-06 with the same simulator on both
  sides.
- **Hybrid** (§3.8). The user's proposal: absorption from the ray world's geometry,
  blur from the measured kernel, slice by slice.
- **Direction averaging** (§3.8). Booking the absorbed light at a slice point as the
  average over all directions arriving there, because a kernel does not know which
  ray the light belonged to.
- **Ray-resolved kernel** (§3.8). A kernel written as the rays' own displacements
  followed by a per-ray residual blur. It would let each ray keep its history. By
  (15) the production kernel has no such form beyond about 0.1 µm of defocus; within
  it, (15) does not decide (§3.8).
- **Characteristic function** (§3.8). The Fourier transform of a probability
  distribution; for a blur kernel, its transfer function.
- **Passband** (§3.8). The band of spatial frequencies the objective transmits; for
  incoherent imaging it ends at $2\,\mathrm{NA}/\lambda$ (textbook).
- **Bookkeeping error** (§3.5, §3.9). Any error in where, and how much, light a
  renderer counts as absorbed, as distinct from an error in the blur.
- **Dark-field image** (§3.10). The image formed only by light the specimen scatters
  into the objective, with no direct light.

---

## 3. Main body

### 3.1 What the renderers are for, and where they sit

*Establishes: why a second renderer exists, and the map both implement.*

The corrected diameter of a real node $i$ solves
$\hat m(d,\varphi_i,\hat\alpha_i\mid\mathcal C)=\hat d_i$ (procedure Eq. 1, as amended
by D-031). Here $\hat m$ is the mean fitted diameter of phantoms rendered and fitted
like real nodes. The table therefore describes the fit's bias in the renderer's world.
If the renderer draws real nodes wrongly, the table corrects them wrongly, and the
table's own spread $\hat\tau$ does not show it (D-030, comment (a)).

Both renderers are maps with the same domain and codomain:
- **domain:** a phantom (true $d$, $\varphi$, $\theta$, $\mu$ and position;
  procedure §3.5) together with a plane depth $z_k$;
- **codomain:** a plane $I_k:\mathbb R^2\to[0,B]$, before the camera chain.

| | partition | ray world |
|---|---|---|
| role | builds the bias table | reference for checks; never builds the table |
| absorption | slice by slice, history along the vertical | Beer–Lambert along every ray of the cone |
| blur | the kernel, slice by slice (diffraction near focus) | the rays' own geometric spread, plus a 0.08 µm post-blur |
| status on the branch | default `absorption = "partition_vertical"` | option `absorption = "ray_world"` (Block 9) **[src]** |

Neither renderer is the real microscope. The real image (physics level) is what the
stacks record.

### 3.2 The partition

*Establishes: the formula, its three exact properties, and what it assumes.*

Number the slices in the direction the light travels. For each fixed $\mathbf x$ and
each slice $j$ (procedure Eq. 5; mathematics Eq. 12):

$$T_{<j}(\mathbf x)=\exp\Big(-\sum_{j'<j}a_{j'}(\mathbf x)\Big),\qquad \Delta A_j(\mathbf x)=T_{<j}(\mathbf x)\,\big(1-e^{-a_j(\mathbf x)}\big).\tag{1}$$

For each fixed plane $k$ (procedure Eq. 6):

$$I^{\rm P}_k(\mathbf x)=B\Big[1-\sum_{j=1}^{J}\big(\Delta A_j*K_{\zeta_j-z_k}\big)(\mathbf x)\Big].\tag{2}$$

In plain words: cut the tube into horizontal slices; let the light climb each vertical
column and lose a share in each slice; blur each slice's loss by its own defocus; add
the slices up.

Three properties hold exactly. Mathematics §3.2 proves each in one line, and checks
the first two numerically (`doc_checks.py` C1 and C2; no reference output of that
script is committed) **[KB-repo]**:
1. **Conservation per column.** $\sum_j\Delta A_j(\mathbf x)=1-\exp\big(-\sum_ja_j(\mathbf x)\big)$,
   which is Beer–Lambert for the vertical column.
2. **One slice reproduces the fit's model.** A single slice at $\delta=0$ with the fit's
   Gaussian as $K_0$ gives $B\,(T*g)$, the fit's model (handoff Eq. 11).
3. **Faint limit.** $\Delta A_j=a_j+O(a^2)$, so the image is the sum of the blurred
   slice absorbances (mathematics Eq. 11).

**What it assumes.** The light reaching a slice point is the light left on the
vertical column below it, and the path through a slice is its vertical thickness. In
reality the light crosses the specimen obliquely, along laterally displaced and
longer paths: at up to 67.5° under a fully open, evenly filled NA 1.4 condenser
(procedure §3.6, "What Eq. (6) is **not**"), a setting Allen does not report
(optics §3.6). The kernel $K_\delta$ carries all of the blur, including diffraction near
focus.

### 3.3 The ray world

*Establishes: the formula, what it computes exactly, and what it lacks.*

§3.2's assumption is about paths, so the natural reference follows the real paths.
For each fixed plane $k$ and point $\mathbf x$ ((S6)):

$$I^{\rm ray}_k(\mathbf x)=B\,\big\langle e^{-\mu L(\mathbf x,z_k,\hat s)}\big\rangle_W .\tag{3}$$

Each ray of the cone through the image point is followed through the whole tube. Its
transmission is Beer–Lambert along the ray, and the image point receives the average
over the cone. In the code (`model/ray_world.py` **[src]**):
- the chords are analytic;
- the average uses Gauss–Legendre quadrature in $\rho^2$, with $16\times32$ directions
  by default (most check scripts use $48\times96$; `hybrid_absorption_check.py` uses
  $16\times32$);
- a 0.08 µm Gaussian post-blur stands in for the in-focus diffraction core.

**What it computes exactly**, within geometric optics:
- every ray's history and its path length through the stain;
- the order-independence of absorption. Transmission along a line is a product of
  transmittances, so the order in which the light meets the stain does not matter.
  For a flat tube, mirror-symmetric about its axis depth, the planes at $c_z\pm\delta$
  are therefore equal (branch docstring and smoke test, to $10^{-12}$ **[src]**).

**What it lacks.**
- Diffraction: its blur near focus is the rays' own geometric spread (§3.6).
- Coherence.
- A kernel: its blur is fixed by the assumed cone and cannot take a measured kernel.
- Speed: it is much slower than the partition (SPEC: "minutes per phantom; the ray
  world much longer" **[src]**).

### 3.4 The two renderers in one formula

*Establishes: the precise sense in which, as this chat put it on 2026-10-09, the ray
world is the partition minus its approximation.*

(1)–(2) are written slice by slice and (3) ray by ray. Writing (3) slice by slice
puts the two on the same footing.

**The ray world, sliced.** The ray through $(\mathbf x,z_k)$ with direction $\hat s$
crosses slice $j$ at $\mathbf x_j=\mathbf x+(\zeta_j-z_k)\,\mathbf t_{\hat s}$. Its path
there is the vertical thickness inside the tube at $\mathbf x_j$ divided by
$\cos\vartheta$, so as $\delta\zeta\to0$ its transmission is
$\prod_j\exp\big(-a_j(\mathbf x_j)/\cos\vartheta\big)$. The telescoping identity
$1-\prod_jt_j=\sum_j\big(\prod_{j'<j}t_{j'}\big)(1-t_j)$ **[textbook]** turns (3), for
each fixed $k$ and $\mathbf x$, into

$$I^{\rm ray}_k(\mathbf x)=B\Big[1-\Big\langle\sum_j T^{\rm ray}_{<j}(\mathbf x,\hat s)\,\big(1-e^{-a_j(\mathbf x_j)/\cos\vartheta}\big)\Big\rangle_W\Big],\qquad T^{\rm ray}_{<j}(\mathbf x,\hat s)=\exp\Big(-\sum_{j'<j}\frac{a_{j'}(\mathbf x_{j'})}{\cos\vartheta}\Big),\tag{4}$$

which is exact as $\delta\zeta\to0$ ((S7)).

**The partition, on the same rays.** In (2), take the kernel of the same rays,
$K_\delta=K^{\rm cone}_\delta$, the law of $\delta\,\mathbf t_{\hat s}$ under $W$. The
cone is symmetric under $\mathbf t_{\hat s}\mapsto-\mathbf t_{\hat s}$, so
$(\Delta A_j*K^{\rm cone}_\delta)(\mathbf x)=\langle\Delta A_j(\mathbf x+\delta\,\mathbf t_{\hat s})\rangle_W$,
and (2) becomes, for each fixed $k$ and $\mathbf x$ ((S8)),

$$I^{\rm P}_k(\mathbf x)=B\Big[1-\Big\langle\sum_j T_{<j}(\mathbf x_j)\,\big(1-e^{-a_j(\mathbf x_j)}\big)\Big\rangle_W\Big].\tag{5}$$

(4) and (5) share the slices, the rays, the crossing points, the average over the
cone and Beer–Lambert. They differ in two places only:
1. **History.** (5) uses $T_{<j}(\mathbf x_j)$, the light left on the vertical line
   through the crossing point. (4) uses $T^{\rm ray}_{<j}(\mathbf x,\hat s)$, the light
   left along the ray itself.
2. **Path element.** (5) uses $a_j$, the slice's vertical thickness. (4) uses
   $a_j/\cos\vartheta$, the ray's actual path through the slice.

Undo both substitutions and (5) becomes (4). That is the whole sense in which the ray
world is the partition minus its approximation. It is not a claim that the ray world
is closer to the real microscope: §3.6 shows it is not, near focus. The figure
`TEEG_ray_world_vs_partition_2026-10-07.svg` shows one oblique ray and one slice:
- the ray world reduces the light by the stain the ray actually crossed;
- the partition reduces it by the stain straight below the crossing point.

### 3.5 What the two substitutions do

*Establishes: harmless in faint stain once matched; a qualitative failure in dark
stain; the numbers.*

§3.4 located the difference. This section measures it, first where it should vanish
and then where it does not.

**Faint stain.** At first order both histories equal 1 and the exponentials are
linear. The ray world's absorbed light at $\mathbf x$ is
$\langle\sum_ja_j(\mathbf x_j)/\cos\vartheta\rangle_W$; the partition's is
$\langle\sum_ja_j(\mathbf x_j)\rangle$, averaged with the weights of its kernel. For
any function $f$ of the direction,
$\langle f/\cos\vartheta\rangle_W=\langle1/\cos\vartheta\rangle_W\,\langle f\rangle_{W^*}$,
by the definition of $W^*$. Hence, suppose the partition's kernel is $K^{{\rm cone}*}$
(weights $W^*$) and its coefficient is

$$\mu^{(1)}_{\rm ph}=\Big\langle\frac{1}{\cos\vartheta}\Big\rangle_W\,\mu=\frac{2}{s_m^2}\Big(1-\sqrt{1-s_m^2}\Big)\,\mu=1.447\,\mu .\tag{6}$$

Then, for each fixed $k$ and $\mathbf x$ and any tube geometry, in the limit
$\delta\zeta\to0$,

$$I^{\rm P*}_k(\mathbf x)-I^{\rm ray}_k(\mathbf x)=B\cdot O(a^2).\tag{7}$$

The limit is needed because (4) is exact only as $\delta\zeta\to0$; at any fixed
slicing, (5) with these two choices matches the sliced form (4) to first order.
Sources: (S10); the derivation is **[reasoning]**; $\langle1/\cos\vartheta\rangle_W$ is
1.4470 by quadrature **[run, `optics_points_check.py`]**.

The production procedure already meets both conditions:
- **Kernel.** The kernel is calibrated on faint nodes (D-026, D-027). In a ray-world
  microscope, a faint thin node's shadow is spread by the rays weighted by
  $1/\cos\vartheta$, which is $W^*$ **[reasoning]**.
- **Coefficient.** The phantom $\mu$ is matched through $\hat\mu$ (procedure §3.5).
  That gives $\mu_{\rm ph}$, equal to $\mu^{(1)}_{\rm ph}$ at first order: 1.450 against
  1.447 at $\mu d=0.05$ **[run, `optics_points_check.out`, matched]**.

The checks' P\* is exactly this: "Renderer as it would be AFTER calibration on faint
nodes and mu-matching" (docstring of `section_matched`).

The two measures also blur differently. The per-axis RMS lateral slope of the rays is
$\gamma_W=0.7915$ under $W$ and $\gamma_{W^*}=0.8991$ under $W^*$ **[arithmetic;
quadrature, `hybrid_absorption_check.out`]**. These are the coefficients of
mathematics Eq. 17 for its aplanatic weighting ($\omega\propto\cos\vartheta$, 0.79)
and its isotropic weighting ($\omega$ constant, 0.90). Without $W^*$, that is with the
same rays, weights $W$ and the same $\mu$ (P), faint stain already gives
$\Delta\hat d/d=-0.040$ (P 1.162 against G 1.202 at $\mu d=0.05$) **[run,
`optics_points_check.out`, section BC]**.

**Dark stain.** Now the histories differ from 1 and from each other. The table is for
$d=1$ µm, flat, node plane; the ray world's own $\hat d/d$ is 1.20–1.22 throughout
**[run, `optics_points_check.out`, matched]**.

| $\mu d$ | ray world $\hat\mu$ (µm⁻¹) | P\*: $\mu_{\rm ph}/\mu$, $\Delta\hat d/d$ | Gv\*: $\Delta\hat d/d$ |
|---|---|---|---|
| 0.05 | 0.048 | 1.450, −0.002 | +0.000 |
| 0.5 | 0.472 | 1.519, −0.005 | +0.009 |
| 1.5 | 1.333 | no $\mu_{\rm ph}$ found in $[0.5\mu,8\mu]$ | +0.021 |
| 3 | 2.327 | no $\mu_{\rm ph}$ found | +0.050 |

For a 0.5 µm tube at 20° with $\mu d=0.5$: P\* has $\mu_{\rm ph}/\mu$ = 1.422 and
$\Delta\hat d/d=-0.003$; Gv\* has $+0.026$.

How to read the table:
- The history substitution is what breaks. P\* fails exactly where Gv\*, which keeps
  the true history, does not.
- The path-element substitution alone costs up to +0.05 at $\mu d=3$.
- "No $\mu_{\rm ph}$ found" rests on the end points of the bracket. At the four phantom
  coefficients probed for this tube (0.75, 1.5, 12 and 24 µm⁻¹, the bracket's end
  points), the partition's $\hat\mu$ was 0.47 to about 1.03 µm⁻¹ **[arithmetic from
  the rounded bracket values]**. A scan over the phantom coefficient would turn this
  into a statement.

The history substitution has three visible symptoms **[run]**:
1. **The centre lightens as the stain darkens.**
   - With the Debye kernel, the in-focus centre dip of a flat 1 µm tube is
     0.831 / 0.788 / 0.723 at $\mu d$ = 3 / 10 / 50, while the vertical column absorbs
     0.950 / 1.000 / 1.000 (`followup_checks.out`).
   - With the cone kernel weighted by $W^*$ at the same $\mu$ (P\*s), the dip is
     0.772 / 0.736 / 0.699 (same file, printed there as P\*).
   - With the true history (Gv\*s) the opaque centre is black, 1.000 at $\mu d=50$
     (same file, printed there as Gv\*).
2. **Opaque leak.** Take a ray that enters through the lower skin $z_{\rm lo}(v)$ at
   $v_e$ with lateral slope $s_v/s_z$, and let $h$ be the height above the skin of the
   column the ray is in. In the opaque limit the partition books along that ray
   ((S9))

   $$\int\mu\,e^{-\mu h}\,dz\ \xrightarrow{\ \mu\to\infty\ }\ \frac{1}{1-z_{\rm lo}'(v_e)\,s_v/s_z},\tag{8}$$

   the integral running along the ray; the denominator is positive for a ray that
   enters through the skin.
   - (8) is less than 1 for every ray through the centre point with $s_v\ne0$,
     because there $z_{\rm lo}'(v_e)\,s_v<0$: light leaks through an opaque tube. It
     is 1 for $s_v=0$. For a ray through the centre of a flat tube the entry point is
     $v_e=-r\,s_v/\sqrt{s_v^2+s_z^2}$, so $z'_{\rm lo}(v_e)=-s_v/s_z$ and (8) equals
     $s_z^2/(s_v^2+s_z^2)$ **[reasoning]**.
   - It is more than 1 when $z_{\rm lo}'(v_e)\,s_v/s_z>0$, where the ray's lateral
     motion follows the skin's rise. Column sums stay conserved.
   - Averaged with the weights $W^*$ over the rays through the centre of a flat 1 µm
     tube, (8) gives 0.691, and 0.737 with $W$ (`focus_scan.out`;
     `hybrid_absorption_check.out`). P\*s gives 0.699 at $\mu d=50$
     (`followup_checks.out`).
3. **Depth asymmetry and focus shift.**
   - The partition's planes three below and three above the axis differ: P's centre
     dips are 0.4698 / 0.3865 at $d=1$ µm, $\mu d=1.5$, light side first. The ray
     world's are equal, 0.4887 / 0.4887 (`optics_points_check.out`, BC).
   - With the Debye kernel, the dip-depth focus curve peaks one plane toward the light
     for $d=1$ µm, $\mu d=1.5$, three planes toward the light for $d=2$ µm,
     $\mu d=2$, and at the axis for faint stain (`focus_scan.out`).
   - Mechanism: $T_{<j}$ gives the slices the light meets first the largest share, so
     the shadow looks as if it sat on the light side. The ray world's transmission is
     order-independent (§3.3), so it has no such side.
   - Consequences: this is why `light_direction` matters only through the partition,
     and why the gradient-energy focus rule was preferred to the dip depth on rendered
     dark tubes (D-035, "Why").

These results are already recorded elsewhere:
- $\hat\mu$ is an effective coefficient ((S10));
- the dark label: D-024 (ii)'s threshold, now only a label (D-030's statement and
  its comment (c));
- the faint-only thick calibration set (D-026 (e));
- Open work 1.

### 3.6 What the ray world gets wrong: blur near focus

*Establishes: why the ray world cannot build the table, even for thick or dark
dendrites.*

§3.5 found the ray world's absorption exact. That makes it tempting as the table's
renderer, at least for thick or dark nodes. Its blur is what rules it out.

Post-blur aside, the ray world's only blur is the rays' own spread. A ray crossing a
slice at defocus $\delta$ lands a distance $|\delta|\,|\mathbf t_{\hat s}|$ to the side,
so a point's shadow spreads into a spot whose edge, set by the outermost rays, has
the radius

$$R_{\rm disc}(\delta)=|\delta|\tan\vartheta_{\max},\qquad \tan\vartheta_{\max}\approx2.4,\tag{9}$$

which grows linearly from focus (optics Eq. 6). The real objective behaves
differently. Defocus adds a phase error across the pupil whose largest value is

$$\Delta\phi_{\max}(\delta)=\frac{2\pi n_{\rm oil}}{\lambda}\,|\delta|\,\big(1-\cos\vartheta_{\max}\big)\approx10.7\ \mathrm{rad\,µm^{-1}}\times|\delta|\qquad(\lambda=0.55\ \text{µm}),\tag{10}$$

with $\vartheta_{\max}$ standing here for the objective's aperture angle, the same
67.5° because both apertures are NA 1.4 (optics Eq. 6a). Until that error reaches
$\pi/2$, at $|\delta|\approx0.15$ µm, the spot stays essentially the in-focus Airy
pattern (optics §3.5), whose first dark ring has a radius of 0.24 µm (optics §3.4).
At $\delta=0.15$ µm the cone's outermost rays already land 0.36 µm away, by (9).

A spot's edge is not its width, so compare one measure, the per-axis standard
deviation (µm). For the production kernel it is $\sigma_{\rm r}(\delta)$, the
ideal-Debye core width (optics §3.5, **[run]** there); for the cone it is
$\gamma_W\lvert\delta\rvert$ or $\gamma_{W^*}\lvert\delta\rvert$ (§3.5); the ray world
adds its 0.08 µm post-blur in quadrature **[arithmetic]**:

| $\delta$ (µm) | 0 | 0.14 | 0.28 | 0.42 | 0.56 | 0.84 |
|---|---|---|---|---|---|---|
| production kernel, $\sigma_{\rm r}(\delta)$ | 0.080 | 0.086 | 0.122 | 0.262 | 0.438 | 0.603 |
| cone, weights $W$: $\gamma_W\lvert\delta\rvert$ | 0 | 0.111 | 0.222 | 0.332 | 0.443 | 0.665 |
| cone, weights $W^*$: $\gamma_{W^*}\lvert\delta\rvert$ | 0 | 0.126 | 0.252 | 0.378 | 0.503 | 0.755 |
| ray world, faint stain: $W^*$ cone with the post-blur | 0.080 | 0.149 | 0.264 | 0.386 | 0.510 | 0.759 |

The last row is the ray world's own blur in faint stain, where each direction's
shadow is weighted by its path length (§3.5); in dark stain the weights change with
the absorption. Read the table in three ranges:
- **In focus** the ray world matches the kernel, because its post-blur was chosen
  to: 0.080 against 0.080. Within about 0.1 µm the cone's own spread is smaller than
  the kernel (§3.8).
- **From 0.14 µm on** the ray world is wider than the kernel at every tabulated
  $\delta$. The ratio is largest near 0.28 µm: 0.264 against 0.122, a factor of 2.2
  (1.8 for the bare $W$ cone).
- **Further out** the ratio falls, to 1.16 at 0.56 µm, where the bare $W$ cone and
  the kernel nearly meet (0.443 against 0.438), and rises again to 1.26 at 0.84 µm.
  The absolute excess is largest there, 0.156 µm. At 0.84 µm the kernel is 0.06 µm
  narrower than the $W$ cone, and its continuation with slope 0.79 keeps that gap
  (D-024 (i)) **[arithmetic]**.

**Why this matters for every tube, thick or dark** **[reasoning]**:
- A round tube's edges lie at its axis depth: by (S3), the vertical chord
  $\ell(v)\to0$ as $|v|\to r$ at depth $c_z+u\tan\varphi$. In the node plane the
  outermost rim of each edge is therefore drawn by slices near focus. How wide that
  rim is depends on $d$: at $v=0.9\,r$ the vertical chord spans $\pm0.436\,r$ about
  the axis depth, which is ±0.22 µm for a 1 µm tube, ±0.44 µm for 2 µm and ±0.65 µm
  for 3 µm **[arithmetic]**. Within a few tenths of a micrometre of focus the ray
  world over-blurs most, relative to the kernel (the table); for a thicker tube that
  range draws a thinner rim.
- At the edges the vertical chord, and so the stain along it, goes to zero. Even a
  dark tube is faint there.
- A thicker tube makes the edge error a smaller fraction of $d$. A darker tube leaves
  it roughly unchanged.

**The numbers** **[run, `kernel_confound_check.out`]**. Tubes in focus, flat except
the first row (20°); the same fit for both renderers ($\sigma_{\rm fit}=0.080$ µm,
oracle $\bar B=B$); the partition with the Debye kernel stands in for the real blur;
no $\mu$ matching.

| $d$ (µm) | $\mu d$ | partition $\hat d/d$ | ray world $\hat d/d$ | gap |
|---|---|---|---|---|
| 0.5 (tilt 20°) | 0.5 | 1.063 | 1.317 | −0.254 |
| 1 | 0.05 / 0.5 / 1.5 / 3 | 1.094 / 1.092 / 1.095 / 1.110 | 1.202 / 1.206 / 1.207 / 1.216 | −0.108 / −0.114 / −0.112 / −0.106 |
| 2 | 0.5 / 3 | 1.126 / 1.140 | 1.183 / 1.188 | −0.058 / −0.048 |
| 3 | 0.5 / 3 | 1.127 / 1.139 | 1.177 / 1.179 | −0.050 / −0.040 |

At $\mu d=3$ the gap also contains the partition's own bookkeeping error (§3.5).

**Consequence for a ray-world table.** Such a table would correct a node whose bias is
$\hat b_{\rm diff}$ to

$$\frac{\tilde d}{d}\approx\frac{\hat b_{\rm diff}(d)}{\hat b_{\rm ray}(d)}.\tag{11}$$

This holds where $\hat b_{\rm ray}$ changes little with $d$: here 1.183 → 1.177 from 2
to 3 µm (and $\hat b_{\rm diff}$ 1.126 → 1.127), at $\mu d=0.5$. The ratio is 0.95 at
2 µm and 0.96 at 3 µm, that is, 4–5 % too small against a whole correction of about
13 %; at 1 µm it is 0.91 **[arithmetic]**. At $\mu d=3$ the ratios are 0.96 and 0.97,
and they also contain the partition's bookkeeping error. A ray-world table would also
bypass the kernel calibration (D-026, D-027): the ray world has no kernel, and its
blur is set by an assumed cone whose real setting Allen does not report (optics
§3.6).

### 3.7 Comparing the renderers like-for-like

*Establishes: the conditions of a clean comparison, what an unmatched one measures,
the status on the branch, and the end-to-end acceptance.*

§3.5 measured the bookkeeping and §3.6 the blur. A comparison has to keep the two
apart.

A difference between the two renderers measures the bookkeeping only if they share
the blur. That needs three conditions, all met by `optics_points_check.py matched`:
- (a) the partition is rendered on the ray world's own rays ($K^{\rm cone}$);
- (b) the rays are weighted by obliquity ($W^*$);
- (c) $\mu$ is matched through $\hat\mu$.

Under (a)–(c), (7) holds: the difference is $O(a^2)$ and is the bookkeeping alone. If
instead the partition keeps its production kernel $K_\delta$, then for each fixed $k$
and the same absorbed fractions, (2) gives

$$I^{\rm P}_k[K]-I^{\rm P}_k[K^{{\rm cone}*}]=-B\sum_j\Delta A_j*\big(K_{\zeta_j-z_k}-K^{{\rm cone}*}_{\zeta_j-z_k}\big),\qquad\Delta A_j=a_j+O(a^2).\tag{12}$$

This term is first order and does not vanish in faint stain, so the comparison
measures blur first. It dominates the gap of §3.6, about −0.11 at $d=1$ µm whatever
the darkness. That gap is not (12) itself: it compares the partition with the ray
world, which adds its post-blur, at the same true $\mu$, and at $\mu d=3$ it also holds
the bookkeeping error **[reasoning]**. Either way the blur difference hides the dark
failure of §3.5.

**On the branch** **[src]**:
- the ray world is a full generator (`renderer.absorption = "ray_world"`);
- the partition has only the Gaussian table;
- SPEC lists the cone kernel as not built (Block 9: "it needs a cone kernel family in
  Block 4").

So the D-024 (i) study (Open work 1) can run in one of two ways: at profile level with
the `checks/` functions over the design's draws, or on the branch once Block 4 gains a
cone-kernel family.

**End-to-end acceptance** (procedure §3.10, last row).
- **What the procedure asks for:** test phantoms from "an alternative generator or
  kernel", because "with the same simulator it tests only the Monte Carlo and the
  inversion".
- **What gate 2 did:** it used the same simulator on both sides (2026-10-06;
  **[src]**, SPEC, Block 7 status).
- **Why that matters:** testing an inversion on data simulated with the model it
  inverts is what the inverse-problems literature calls the inverse crime. It is
  avoided by simulating the test data with a different model: a head model from a
  different subject in EEG source imaging (Paz-Linares et al. 2017), different
  discretisations plus noise in magnetic induction tomography (Hofmann et al. 2022)
  **[PubMed, full text]**.
- **Why the ray world is not that model here:** with a Gaussian-kernel table,
  ray-world test phantoms would fail the acceptance because of blur, not absorption,
  by (12). SPEC proposes the empirical kernel as the acceptance's alternative and
  keeps the ray world for absorption checks **[src]**.

### 3.8 The hybrid

*Establishes: the construction, the formula, why it must average over directions,
and what changes in the code.*

The partition has the right blur and, in dark stain, the wrong bookkeeping (§3.5).
The ray world has the right bookkeeping and the wrong blur near focus (§3.6). The
user's proposal of 2026-10-09 combines the right half of each: compute absorption with
the ray world's geometry (NA 1.4, $n_{\rm oil}$ 1.515), and render each image plane
with the kernel.

**Three ingredients.**
1. **Absorption from the rays.** The history of the light reaching a slice point is
   taken along each ray of the cone that reaches it. The path through the slice is
   $a_j/\cos\vartheta$.
2. **Blur from the kernel.** Each slice's absorbed light is blurred by
   $K_{\zeta_j-z_k}$ for every rendered plane; (2) is unchanged.
3. **The cone's own spread is dropped.** The rays say where the light was absorbed,
   not where it lands. Blurring the ray world's finished planes with the kernel would
   blur twice. Since a convolution with a non-negative kernel only widens, it also
   could not undo the ray world's over-blur near focus.

**The direction average.** In the ray world a ray's direction decides both what the
ray absorbed and where it lands. The kernel replaces "where it lands" and does not
know the direction. So the absorbed light at a slice point must be averaged over the
directions arriving there before it is spread. For each fixed $\mathbf x$ and slice
$j$, to first order in $\delta\zeta$,

$$\Delta A^{\rm hyb}_j(\mathbf x)=\delta\zeta\,\Big\langle\frac{\mu}{\cos\vartheta}\,e^{-\mu\,\ell_{\rm back}(\mathbf x,\zeta_j;\hat s)}\Big\rangle_W\quad\text{inside the tube, and 0 outside.}\tag{13}$$

At the slice point, $e^{-\mu\ell_{\rm back}}$ is the light left along the ray arriving
with direction $\hat s$, that ray's own history, and $\mu\,\delta\zeta/\cos\vartheta$ is
that ray's absorbed share in the slice. For each fixed plane $k$,

$$I^{\rm hyb}_k(\mathbf x)=B\Big[1-\sum_j\big(\Delta A^{\rm hyb}_j*K_{\zeta_j-z_k}\big)(\mathbf x)\Big].\tag{14}$$

For a flat tube, $\ell_{\rm back}$ follows from the cross-section alone. Let
$\mathbf q=(v,\,z-c_z)$ be the point in the across-branch plane,
$p=\sqrt{s_v^2+s_z^2}$ the length of the projection of $\hat s$ onto that plane, and
$\mathbf e=-(s_v,s_z)/p$ the backward direction projected onto it, as a unit vector.
The in-plane distance back to the circle is
$\ell^{\rm 2D}_{\rm back}=-\mathbf q\cdot\mathbf e+\sqrt{(\mathbf q\cdot\mathbf e)^2-|\mathbf q|^2+r^2}$,
and $\ell_{\rm back}=\ell^{\rm 2D}_{\rm back}/p$. Tilted tubes use Block 2's line
chords instead.

**Properties.**
- **First order.** In faint stain, (13) books
  $\langle\mu/\cos\vartheta\rangle_W\,\delta\zeta=1.447\,\mu\,\delta\zeta$ per slice
  inside the tube, that is $1.447\,\mu$ per unit vertical extent. That is the
  partition's faint booking at the coefficient $\mu^{(1)}_{\rm ph}$. With a kernel
  calibrated on faint nodes ($K^{{\rm cone}*}$ in a ray-world microscope), (14)
  therefore equals the ray world at first order by (7), with no rescaling of $\mu$
  **[reasoning; run: 0.057 against 0.057 at $\mu d=0.05$, §3.9]**. On real data $\mu$
  is matched through $\hat\mu$ anyway.
- **Code.** Only (1) changes, from $\Delta A_j$ to $\Delta A^{\rm hyb}_j$; (2) and the
  camera chain stay. On the branch, (1) is computed in the private generator `_slabs`
  of `model/render.py`, which `render_transmittance` uses; `"ray_world"` is sent to
  `model/ray_world.py` instead, and the test helper `absorbed_fractions` refuses it,
  because the ray world has no absorbed fractions **[src]**. The hybrid would be a new
  value of `absorption`, dispatched in `_slabs` and added to `ABSORPTIONS_RENDERED`;
  its name belongs in SPEC. That is one block, as D-021 intends.
- **Cost.** For each slice point, an average over directions of an analytic chord. It
  is computed once per phantom and reused for every plane, because (13) does not
  depend on $z_k$; the ray world recomputes for every plane **[reasoning; not timed]**.

**Why not keep each ray's own history?** Pairing each ray's history with that ray's
own landing point would require writing the kernel as the rays' displacements
followed by a residual blur around each landing point. With one residual $\kappa$ for
all rays this is $K_\delta=K^{\rm cone}_\delta*\kappa$, with $\kappa\ge0$ of unit area;
in general each ray may carry its own $\kappa_{\hat s}$, centred on its landing point,
and $K_\delta$ is the mixture of the shifted $\kappa_{\hat s}$ over the rays. Here
$K^{\rm cone}_\delta$ carries whichever weights the pairing uses: $W^*$ for a kernel
calibrated on faint nodes (§3.5), $W$ otherwise. For kernels with finite second
moments, per-axis variances add under convolution, and for the mixture the law of
total variance gives the same with $\sigma^2_\kappa$ replaced by the average of the
$\kappa_{\hat s}$ variances **[textbook]**. So for each fixed $\delta$

$$\sigma^2_{K_\delta}=\sigma^2_{\rm cone}(\delta)+\sigma^2_\kappa\ \ge\ \sigma^2_{\rm cone}(\delta).\tag{15}$$

The production kernel violates (15) beyond about 0.1 µm of defocus. Its width
$\sigma_{\rm r}(\delta)$ falls below $\gamma_{W^*}\lvert\delta\rvert$ at
$\lvert\delta\rvert=0.093$ µm, and below $\gamma_W\lvert\delta\rvert$ at 0.107 µm. It
stays below at every larger tabulated $\lvert\delta\rvert$, and between the entries
under linear interpolation (§3.6 table) **[arithmetic]**. The margin is smallest at 0.56 µm under
$W$: 0.438 against 0.443 µm. So beyond about 0.1 µm the production kernel has no such
decomposition **[reasoning]**.

(15) is a necessary condition only. Within about 0.1 µm of a plane it does not rule
the pairing out, and it does not establish it either. With one $\kappa$ for all rays,
an exact decomposition of the Gaussian fails there too: a convolution multiplies
Fourier transforms, and the cone kernel's transform has zeros while a Gaussian's has
none. (With a different $\kappa_{\hat s}$ per ray the transform of the mixture does
not factor, and this argument does not apply.) The characteristic function of the
per-axis ray slope first changes sign at an argument of about 7 radians per unit
slope, under $W$ and under $W^*$ (minimum −0.002 and −0.005) **[run,
`hybrid_absorption_check.out`; convolution theorem, textbook]**. At defocus $\delta$
that zero lies at the spatial frequency $7/(2\pi|\delta|)$, above the passband
($2\,\mathrm{NA}/\lambda=5.1$ µm⁻¹) for $|\delta|\lesssim0.2$ µm **[arithmetic]**. So an
approximate pairing near focus is not excluded. For heavy-tailed kernels, whose
second moment is infinite (mathematics §3.3), (15) needs another width measure.

The direction average is therefore forced for every slice farther than about 0.1 µm
from the rendered plane. For the nearer slices it is the simplest choice, not a
forced one.

### 3.9 How well the hybrid works

*Establishes: its accuracy where the answer is known, the near-opaque leak and its
cause, and what is still untested.*

The direction average of §3.8 is an approximation, so it has to be measured. The test
runs in the ray world with the cone's own $W^*$-weighted kernel, where the ray world
is exact. With that kernel a paired renderer would reproduce the ray world, so the
hybrid's error here is the cost of the direction average alone. Flat 1 µm tube,
centre dip $D_c$ at the node plane, geometric blur only, no post-blur **[run,
`hybrid_absorption_check.out`]**:

| $\mu d$ | 0.05 | 0.5 | 1.5 | 3 | 10 | 50 |
|---|---|---|---|---|---|---|
| hybrid H, (13)–(14) | 0.057 | 0.450 | 0.845 | 0.961 | 0.811 | 0.732 |
| ray world G (exact here) | 0.057 | 0.441 | 0.820 | 0.965 | 1.000 | 1.000 |
| partition P\*s (cone kernel $W^*$, same $\mu$, vertical path) | 0.040 | 0.323 | 0.641 | 0.772 | 0.736 | 0.699 |

The script uses $16\times32$ directions. Its P\*s and Gv\*s columns reproduce
`followup_checks.py`'s $48\times96$ values to three decimals at the four darknesses
both scripts run. The hybrid column itself was re-run at $24\times48$ and $32\times64$
directions only in the review's scratch run (not committed), which gave the same
values to four decimals. The partition's row uses vertical path elements at the same
$\mu$, so in faint stain it absorbs less by the factor 1.447 (0.040 against 0.057).
Compare its trend, not its level.

Up to $\mu d\approx3$ the hybrid is within 0.03 of the exact centre dip. Beyond that
its centre lightens as the stain darkens, as the partition's does: 0.961, 0.811 and
0.732 at $\mu d$ = 3, 10 and 50, against 0.965, 1.000 and 1.000.

**Why the centre is too light** **[reasoning; run where stated]**:
- Booking light at a point and then spreading it without the ray's direction is the
  partition's own structure; the hybrid keeps it.
- In the opaque limit each direction's light is absorbed where that direction enters
  the tube, and the hybrid books it there. At $\mu d=50$, 10.3 % of the booked
  absorption lies above the axis depth, on the side walls through which the oblique
  directions enter; at $\mu d=3$, where light penetrates further, 26.8 % **[run]**.
  The partition, which follows vertical columns, books all of it at the lower skin.
- The kernel then spreads each booked point's light as if every direction had lost
  light there, not only the directions that entered there. So along each line through
  the centre, the booked absorption sums to a skin-crossing factor like (8) instead
  of to 1. The value differs from the partition's because the booking does.
- Nothing is lost. For each direction, the hybrid's total booking over the
  cross-section equals the ray world's total shadow for that direction: both are
  $p/s_z$ times the integral, over the ray's in-plane offset from the axis, of one
  minus its transmission through the tube **[reasoning]**. Per unit length of a flat
  1 µm tube the run gives 1.114 against 1.114 µm at $\mu d=3$, and 1.240 against
  1.238 µm at $\mu d=50$, equal to within the error of the script's square grid (the
  opaque value, an upper bound, is $2r\langle p/s_z\rangle_W=1.239$ µm) **[run]**. The
  centre's missing darkness is light moved, not lost; where it goes has not been
  looked at.
- The ray world makes the opaque centre black because it pairs each ray's history
  with that ray's own landing point. For the production kernel, (15) forbids that
  pairing beyond about 0.1 µm of defocus.

**Untested so far:**
- fitted $\hat d$: only centre dips have been compared;
- tilted tubes, full profiles, and planes off focus;
- the cost;
- the hybrid with the production kernel, for which no exact reference exists.

**Proposed next tests:**
1. Profile-level fits of the hybrid against the ray world, both on the cone kernel,
   over a grid of $(d,\varphi,\mu d)$, reporting $\Delta\hat d/d$ and $\Delta\hat\mu$.
2. The full profile in the opaque regime, to see where the moved light lands.
3. The range of $\hat\mu$ and $\hat\alpha$ each renderer can reach at all (D-031
   coverage).
4. The real $\hat\mu_i$ distribution, which decides whether nodes beyond
   $\mu d\approx3$ exist in numbers that matter (D-024 open point).

### 3.10 What neither renderer contains

*Establishes: the limits the two share, so that their comparison is not over-read.*

Everything above compares two models of the same kind, so their agreement cannot
vouch for what both leave out.
- **Coherence.** Both renderers describe incoherent imaging. At $S=1$ the true image of
  a thin object is darker than $B\,(T*h_0)$ by the specimen's dark-field image from
  the directions outside the objective's aperture (Open work 2). In focus, for
  $d=1$ µm, the incoherent model is brighter by at most $0.009\,B$ ($\mu d=1$) and
  $0.045\,B$ ($\mu d=3$) **[run, `coherence_check.out`]**.
- **Other physics left out:**
  - refraction or phase in the node;
  - a stopped-down or unevenly filled condenser ($S<1$);
  - spherical aberration from index mismatch (optics §3.7);
  - the camera chain (procedure §3.6; D-025 for its noise).
- **The limits of the ray world's agreement.** Agreement with the ray world validates
  only the bookkeeping, and only in geometric optics. Near focus, "the history along a
  ray" is not defined in wave optics. The ray world therefore gives the sign and the
  mechanism of the bookkeeping error, not its size in the real microscope
  **[reasoning]**.

---

## 4. Summary of results

1. The two renderers share domain and codomain. The table is built with the
   partition; the ray world is a reference (§3.1).
2. The partition, (1)–(2): exact conservation per column, exact one-slice and faint
   limits (§3.2).
3. The ray world, (3): exact Beer–Lambert per ray, independent of order, so
   depth-symmetric for flat tubes. It has no diffraction and no kernel (§3.3).
4. Sliced, (4) against (5): the same slices, rays, crossing points and average. The
   partition makes two substitutions, the vertical history and the vertical path
   element. Undoing them gives the ray world (§3.4).
5. With $K^{{\rm cone}*}$ and $\mu^{(1)}_{\rm ph}=1.447\,\mu$, (6), the two agree to
   first order for any geometry as $\delta\zeta\to0$, (7). Without the $W^*$ weights
   the faint-stain gap is already −0.040 (§3.5).
6. In dark stain the history substitution breaks:
   - no probed phantom coefficient reaches the ray world's $\hat\mu$ at $\mu d$ = 1.5
     and 3, the two dark cases run;
   - the centre lightens as the stain darkens;
   - the opaque leak, (8), averages 0.691 at the centre under $W^*$ (0.737 under $W$);
   - the planes lose their depth symmetry and the focus shifts.

   The path element alone costs +0.009 / +0.021 / +0.050 at $\mu d$ = 0.5 / 1.5 / 3
   (§3.5).
7. Geometric blur, (9), against diffraction, (10): with its post-blur the ray world
   matches the kernel in focus and is wider at every tabulated defocus from 0.14 µm on,
   by a factor of about 2 near 0.28 µm. That is where every tube's edges are drawn.
   The gap in $\hat d/d$ is −0.25 / −0.11 / −0.05 to −0.06 / −0.04 to −0.05 for
   $d$ = 0.5 (20°) / 1 / 2 / 3 µm. A ray-world table would return faint 2–3 µm nodes
   ($\mu d=0.5$) 4–5 % too small, (11) (§3.6).
8. A like-for-like comparison needs (a)–(c). Otherwise (12) makes it measure blur. The
   branch does not yet have the cone kernel (§3.7).
9. The hybrid, (13)–(14), books absorption exactly to first order: with the kernel
   that faint nodes would calibrate in a ray-world microscope, it equals the ray world
   at first order. In the code it replaces only (1). By (15), direction averaging is
   forced for every slice farther than about 0.1 µm from the plane, because the
   production kernel is narrower than the cone there (§3.8).
10. At the centre of a flat 1 µm tube in the node plane, with the cone's own kernel,
    the hybrid stays within 0.03 of the exact dip up to $\mu d\approx3$. At
    $\mu d\ge10$ its centre is too light, 0.811 and 0.732 against 1.000. It books the
    light where each direction enters the tube, and the direction-blind kernel moves
    it; its total equals the ray world's (§3.9).

---

## 5. Open points, caveats, and assumptions

- **Illustrative optics.** Ideal Debye at 550 nm, index-matched, scalar. The ray world
  and the hybrid's absorption assume an evenly filled NA 1.4 cone, and the condenser
  setting is not reported (optics §3.6). Matching $\mu$ through $\hat\mu$ absorbs the
  first-order effect of a different cone, not the second-order one **[reasoning]**.
- **Run conditions.** Every run is noise-free and without the camera chain. Every
  fitted run uses an oracle background ($\bar B=B$) and $\sigma_{\rm fit}=0.080$ µm;
  the deliverable $\sigma_{\rm fit}$ is 0.099 µm (D-023). The centre-dip and coherence
  runs fit nothing.
- **Kernel.** The production kernel's Gaussian shape is itself provisional (I-002).
  The measured kernel comes with D-026 and D-027, and §3.6's widths and §3.8's
  0.1 µm threshold must be re-read on it.
- **Width measures.** §3.6 and (15) use the production kernel, a Gaussian whose width
  is the Debye core width. The real Debye line-spread function has heavy tails: its
  second moment is infinite at every defocus (mathematics §3.3). So (15), which needs
  finite second moments, is applied here only to the Gaussian, not to the real
  line-spread function.
- **Near-focus pairing.** Within about 0.1 µm of a plane, (15) does not decide whether
  a ray-resolved kernel exists. With one residual for all rays an exact one fails for
  the Gaussian, but only through Fourier zeros above the passband for
  $|\delta|\lesssim0.2$ µm; with per-ray residuals that argument does not apply. A
  hybrid that pairs rays for the nearest slices and averages for the rest is
  conceivable and untested (§3.8).
- **Scope of the hybrid test.** Only the centre of a flat 1 µm tube in the node plane,
  in the geometric world, with the cone's own kernel. Its fitted $\hat d$, tilted
  tubes, planes off focus, where the moved light lands, the production kernel and its
  cost are untested (§3.9).
- **Bracket end points.** "No $\mu_{\rm ph}$ found" rests on the end points of the
  bracket (§3.5).
- **Unknown real stain.** The real $\hat\mu_i$ and $\hat\alpha_i$ distribution is not
  known. It decides whether the hybrid's range ($\mu d\lesssim3$) suffices, and
  whether D-031's $\hat\alpha$ axis covers dark nodes (Open work 4).
- **Physics outside both renderers:** coherence (effect 3), refraction, and $S<1$
  (§3.10).
- **No decision yet.** Adopting the hybrid is not decided: D-024 (i) is still open,
  and the user's proposal of 2026-10-09 is not logged. The log's next free IDs are
  D-041 and I-003 (its 2026-10-09 rows).
- **Branch state.** Read at `3084991`, rechecked at `8a11f1e`; the branch may move
  again.

---

## 6. References and sources

**Knowledge base (repo, D-020):**
- design handoff `handoff_diameter_remeasurement.md`, Eq. 11 (the fit's model);
- procedure §3.1–3.10 (Eqs. 1, 5, 6, 7);
- mathematics §3.2 (Eqs. 11–13; `doc_checks.py` C1–C2 reported there), §3.3, §3.4
  and Eq. 17;
- optics §3.4–3.7 (Eqs. 6, 6a);
- implementation handoff (S1)–(S10), "Findings" and "Validation";
- theory handoff, Open work 1, 2 and 4;
- study notes;
- figure `TEEG_ray_world_vs_partition_2026-10-07.svg`;
- decisions D-018, D-019, D-021, D-023 to D-027, D-030, D-031 and D-035, and idea
  I-002 (project log `TEEG_decisions_and_ideas_log.md`; read again on 2026-10-09 for
  D-030's comments and the next free IDs).

**Check scripts and reference outputs, `../checks/` [run]:**
- `optics_points_check.py` (sections BC and matched);
- `followup_checks.py`;
- `focus_scan.py`;
- `kernel_confound_check.py` (2026-10-09);
- `hybrid_absorption_check.py` (2026-10-09; booking, conservation and kernel-split
  sections added later the same day);
- `coherence_check.py` (2026-10-08).

**Code [src], `sci/diameter-pipeline` at `3084991`, rechecked at `8a11f1e`:**
- `src/allen_diameter/model/ray_world.py` and `model/render.py`;
- `src/allen_diameter/config.py`;
- `scripts/end_to_end.py`;
- `specs/SPEC.md` (Blocks 4, 7 and 9; open questions).

**PubMed, full text read (2026-10-09; both re-read later that day to check the
attribution of the term):**
- Paz-Linares D. et al. (2017) Spatio temporal EEG source imaging with the
  hierarchical Bayesian elastic net and elitist lasso models. *Front Neurosci*
  11:635. PMC5696363. [DOI](https://doi.org/10.3389/fnins.2017.00635). Avoids the
  "inverse crime" by solving with a lead field from a different subject than the one
  that simulated the data, and adds noise at sources and sensors. It attributes the
  term to Kaipio and Somersalo.
- Hofmann A. et al. (2022) A deep residual neural network for image reconstruction in
  biomedical 3D magnetic induction tomography. *Sensors* 22:7925. PMC9610508.
  [DOI](https://doi.org/10.3390/s22207925). Avoids the inverse crime with different
  discretisations for simulation and reconstruction, plus 60 dB noise. In the
  retrieved text its citation for the term is not visible (citation markers are
  stripped).
- Kaipio and Somersalo, the book Paz-Linares et al. cite for the term, was not read.

**Searches (2026-10-09):**
- PubMed, earlier: "inverse crime" (11 records); ray-tracing simulation of
  brightfield images of absorbing specimens (0); geometric-optics simulation of
  transmitted-light images with an illumination cone (0).
- PubMed, later, for the hybrid's construction: bright-field image simulation with
  Beer–Lambert absorption and point-spread-function convolution for thick specimens
  (0); ray-tracing absorption combined with a defocus point-spread function in a
  transmitted-light forward model (0); brightfield image formation of absorbing
  specimens, simulation (0). No source for or against the construction was found.
- bioRxiv: one listing (bioengineering, last 30 days, 30 preprints); nothing relevant.
  The connector has no keyword search.
- Data repositories: not queried, since nothing here is a claim about data.

**Textbook, from memory:**
- the Beer–Lambert law and the multiplicativity of transmittances;
- the telescoping identity;
- geometric optics as the zero-wavelength limit;
- the quarter-wave (Rayleigh) tolerance;
- the additivity of variances under convolution;
- the convolution theorem, and that a Gaussian's Fourier transform is a Gaussian,
  without zeros;
- the incoherent cutoff frequency $2\,\mathrm{NA}/\lambda$.

**[reasoning], this document:** (7), (12), (15), the edge argument of §3.6, the
equality of the hybrid's and the ray world's totals, and the leak mechanism of §3.9.
The slopes $\gamma_W$ and $\gamma_{W^*}$, the zeros of the characteristic function and
the totals were computed in `hybrid_absorption_check.py` **[run]**.
