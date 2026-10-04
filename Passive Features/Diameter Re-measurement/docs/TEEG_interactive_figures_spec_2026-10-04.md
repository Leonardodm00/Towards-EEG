# Interactive figures of the diameter re-measurement docs: reproduction spec (for Claude Design)

| Date | Change |
|---|---|
| 2026-10-04 | v1. Specification of the four interactive figures built in chat on 2026-10-03/04, so that Claude Design (or any designer or developer) can rebuild them, restyle them, or extend them without re-deriving the physics. Reference implementations saved as standalone HTML in `../figures/`. Numbers in "Acceptance values" read from those files in headless Chromium on 2026-10-04 **[run]**. |

**How to use this file.** Each figure section is self-contained: purpose, the one claim the figure must make visible, controls, the model (equations and constants), the drawing, the readouts, and the values a correct rebuild must reproduce. The paste-ready prompt at the end of each section is written for Claude Design; it assumes this file and the reference HTML are attached or reachable. Physics background: `TEEG_microscope_optics_oil_immersion_2026-10-04.md` (optics doc) and `TEEG_diameter_bias_table_mathematics_2026-10-04.md` (mathematics doc), same folder.

---

## Figure index

| # | Reference file (`../figures/`) | Claim it makes visible | Used in |
|---|---|---|---|
| 1 | `fig1_defocus_cone_disc_lsf.html` | an out-of-focus point becomes a disc whose radius grows as $|\delta|\tan\theta_{\rm obj}$; summed along a branch, the disc gives a semicircular dip | optics doc §3.5 |
| 2 | `fig2_diffraction_focal_spot_vs_na.html` | waves from a cone of directions interfere into a spot of finite width; a narrower cone (lower NA) gives a wider spot | optics doc §3.4 |
| 3 | `fig3_tilted_plane_wave_lateral_frequency.html` | a plane wave tilted by $\vartheta$ ripples across the focal plane with period $\lambda/(n\sin\vartheta)$; the steepest accepted wave sets the finest ripple | optics doc §3.4 |
| 4 | `fig4_slab_rendering_one_output_plane.html` | one synthetic image plane is the sum of object slabs, each blurred by its own distance to that plane, weighted by the light it actually absorbs | procedure doc §3.6, mathematics doc §3.2 |

A first version of figure 4 (static SVG, three slabs, no partition) was shown on 2026-10-03; it is **superseded** and not reproduced.

---

## Shared design rules

| Rule | Value |
|---|---|
| Canvas width | 680 px design width; scale to container (`width:100%; height:auto`); usable at 380 px |
| Orientation | Microscope convention: optical axis $z$ **vertical**, light travelling **upward** (condenser below, objective above), focal plane horizontal. Figures 1, 3, 4 follow it. **Figure 2's reference implementation draws $z$ horizontally** (light left to right): a rebuild should rotate it to vertical **[open: improvement requested implicitly by the user's question on 2026-10-04]** |
| Units | µm everywhere; angles in degrees in labels |
| Fixed optical constants | $\lambda = 0.55$ µm (vacuum wavelength), $n = 1.515$ (oil), NA = 1.4, so $\theta_{\rm obj} = \arcsin(1.4/1.515) = 67.53°$, $\tan\theta_{\rm obj} = 2.418$ **[run: arithmetic]** |
| Colour roles | blue `#378ADD` = geometric rays / linear-sum comparison; orange `#D85A30` = wave result / plane $k$ / rendered profile; green `#1D9E75` = object slabs / defocused plane; amber `#BA7517` stroke + `#FAC775` fill = figure 1 cone, disc and dip; grey = individual contributions |
| Theme | Text and hairlines from CSS tokens (`--text-primary`, `--text-secondary`, `--border-strong`, `--surface-1`); light and dark mode both legible; transparent or `--surface-0` background |
| Typography | sentence case; labels ≥ 12 px; two weights only (400, 500) |
| Numbers on screen | always rounded: µm to 2-3 decimals, angles to 0.1° |
| Interaction | sliders recompute on `input`; no network; all computation in plain JavaScript (no libraries needed) |
| Honesty labels | each figure states in its caption what is idealised (scalar optics, 2-D, ideal objective, illustrative widths); keep these captions in any restyle |

---

## Figure 1 — Defocus: cone, disc, and the dip across a branch

**Claim.** Rays accepted by the objective form a cone of half-angle $\theta_{\rm obj}$ with its apex at the focus; a plane at defocus $\delta$ (µm, signed distance from the focal plane) cuts it in a disc of radius $R(\delta) = |\delta|\tan\theta_{\rm obj}$; summing that disc along the branch direction gives a semicircular dip across the branch.

**Layout.** Three panels: (left) side view $x$–$z$ with the cone, the focal plane, the defocused plane (green dashed) and the disc's chord (green bar); (top right) the disc seen from above, branch direction marked; (bottom right) the dip across the branch, $v$ axis.

**Control.**

| id | label | range | step | default |
|---|---|---|---|---|
| `dz` | Defocus Δz | 0 – 1 µm | 0.02 | 0.5 |

**Model.** $R = |\delta|\tan\theta_{\rm obj}$. Dip profile across the branch, for each fixed $\delta$: the disc projected along the branch is a semicircle $\sqrt{\max(0, R^2 - u^2)}$, convolved with a Gaussian of standard deviation 0.08 µm (the in-focus line-spread core, Debye, 550 nm), normalised to unit peak. Readout: $\sqrt{0.08^2 + R^2/4}$ µm.

**Readout caveat (must be kept).** $R/2$ is the RMS width of a **uniformly bright** disc. Real defocus discs are dim at the rim; with realistic angular weighting the RMS grows as $0.79$–$0.90\,|\delta|$, not $1.21\,|\delta|$ (mathematics doc Eq. 17, [corrected 2026-10-04]). The reference file labels the readout "uniform disc, upper bound". A rebuild may add a second readout using $0.79\,|\delta|$ (Debye weighting), labelled as such.

**Acceptance values** **[run]**: readout at Δz = 0, 0.28, 0.5, 1.0 µm → R = 0.00, 0.68, 1.21, 2.41 µm; RMS (uniform disc) = 0.08, 0.35, 0.61, 1.21 µm.

**Prompt for Claude Design.**
> Build an interactive figure (HTML, inline JS, no libraries), 680 px wide, light/dark aware, from section "Figure 1" of `TEEG_interactive_figures_spec_2026-10-04.md`, using `fig1_defocus_cone_disc_lsf.html` as the functional reference. Keep the optical axis vertical and light travelling upward. Show: side view with the cone (half-angle 67.5°) and a draggable defocused plane; top view of the disc; the dip across the branch (semicircle ⊗ Gaussian σ = 0.08 µm). Readouts: Δz, R = |Δz|·tan 67.5°, and the RMS width with the label "uniform disc, upper bound" plus a second value 0.79·|Δz| labelled "Debye weighting". Reproduce the acceptance values listed in the spec.

---

## Figure 2 — Diffraction: the focal spot versus numerical aperture

**Claim.** The geometric picture (rays meeting in a point) fails at focus; adding the same directions as waves gives a spot of finite width that grows as NA falls, close to $\lambda/(2\,{\rm NA})$.

**Layout.** Top: three stat tiles (cone half-angle in oil; $\lambda/(2\,{\rm NA})$; spot FWHM of this model). Middle canvas: (left) the fan of directions the lens supplies, up to $\pm\theta$; (right) intensity heat map of the focal region, $x \in [-1.2, 1.2]$ µm across, $z \in [-1.6, 1.6]$ µm along the axis, with the geometric cone overlaid as dashed blue lines. Bottom canvas: intensity across the focus at $z = 0$, with the geometric prediction (a single vertical blue line at $x = 0$) and the FWHM marked.

**Control.**

| id | label | range | step | default |
|---|---|---|---|---|
| `na` | Numerical aperture | 0.2 – 1.4 | 0.05 | 1.4 |

**Model (2-D scalar, equal weights).** Wavenumber in oil $K = 2\pi n/\lambda$ (rad/µm). Half-angle $\theta = \arcsin({\rm NA}/n)$. With $M = 121$ angles $\vartheta_m$ evenly spaced in $[-\theta, \theta]$, for each fixed point $(x, z)$,
$$E(x, z) = \sum_{m=1}^{M} \exp\big(iK(x\sin\vartheta_m + z\cos\vartheta_m)\big), \qquad I(x, z) = |E(x, z)|^2 / \max_{x,z}|E|^2 .$$
Heat-map grid 150 ($x$) × 110 ($z$); profile at $z = 0$ on 401 points; FWHM by linear interpolation of the half-maximum crossing. Intensity drawn as the alpha of orange `rgb(216,90,48)` with alpha $= I^{0.6}$.

**Must say in the caption.** 2-D (line focus from a fan of plane waves), scalar, equal weights per direction, ideal index-matched objective; its FWHM differs from the real 3-D Airy value (0.20 µm at NA 1.4); the trend with NA is the point, not the number.

**Acceptance values** **[run]**:

| NA | cone half-angle | $\lambda/(2\,{\rm NA})$ (µm) | model FWHM (µm) |
|---|---|---|---|
| 1.4 | 67.5° | 0.196 | 0.155 |
| 1.0 | 41.3° | 0.275 | 0.233 |
| 0.7 | 27.5° | 0.393 | 0.340 |
| 0.3 | 11.4° | 0.917 | 0.803 |

**Prompt for Claude Design.**
> Build the interactive figure of section "Figure 2" of `TEEG_interactive_figures_spec_2026-10-04.md`, functional reference `fig2_diffraction_focal_spot_vs_na.html`, **but with the optical axis vertical** (light upward), heat map with $x$ horizontal and $z$ vertical. One NA slider (0.2–1.4). Compute the field as the sum of 121 equal-weight plane waves exactly as specified; reproduce the acceptance table. Keep the caption that states the 2-D scalar idealisation.

---

## Figure 3 — A tilted plane wave ripples across the focal plane

**Claim.** A plane wave travelling at angle $\vartheta$ to the optical axis meets the horizontal focal plane obliquely, so along $x$ its crests repeat every $\Lambda = \lambda/(n\sin\vartheta)$; the axial wave ($\vartheta = 0$) has no pattern; the steepest wave the objective accepts ($n\sin\vartheta = {\rm NA} = 1.4$) gives the finest ripple, $\lambda/{\rm NA} = 0.393$ µm.

**Layout.** Controls row (angle slider, play/pause). Three stat tiles: wavelength in oil $\lambda/n = 0.363$ µm; $\Lambda$; $n\sin\vartheta$. Top canvas (680 × 340): $x \in [-2, 2]$ µm horizontal, $z$ vertical; animated wave crests (blue lines perpendicular to the travel direction, spaced $\lambda/n$); the focal plane as a horizontal line; orange dots where crests meet the plane; a bracket labelled $\Lambda$ between two dots; an arrow showing the travel direction with the axis dashed. Bottom canvas: ${\rm Re}\,E(x)$ along the focal plane at the current instant, $\cos\big(2\pi(x\sin\vartheta/(\lambda/n) - \phi(t))\big)$, peaks at the orange dots.

**Controls.**

| id | label | range | step | default |
|---|---|---|---|---|
| `th` | Tilt angle θ | 0 – 67.5° | 0.5 | 30 |
| `pp` | Pause / Play | toggle | — | playing |

**Model.** Crest $m$: $x\sin\vartheta + z\cos\vartheta = (m + \phi)\,\lambda/n$, $\phi(t) = 0.25\,t \bmod 1$ ($t$ in seconds). Dots at $x_m = (m + \phi)(\lambda/n)/\sin\vartheta$ on $z = 0$. Show "∞ (no pattern)" when $\sin\vartheta < 10^{-3}$.

**Acceptance values** **[run]**: θ = 0 → Λ = ∞, $n\sin\vartheta$ = 0.00; θ = 30° → Λ = 0.726 µm, 0.76; θ = 67.5° → Λ = 0.393 µm, 1.40.

**Prompt for Claude Design.**
> Build the animated figure of section "Figure 3" of `TEEG_interactive_figures_spec_2026-10-04.md`, functional reference `fig3_tilted_plane_wave_lateral_frequency.html`. Optical axis vertical, focal plane horizontal. Animate the crests moving along their travel direction; mark the crest–plane intersections; show the lateral field below. Reproduce the acceptance values. Optional extension: a second, faint wave at −θ and their sum, to show how two directions already build a standing pattern across the plane.

---

## Figure 4 — Rendering one image plane from object slabs

**Claim.** A synthetic image plane $k$ is built from the **object**, never from other image planes: the phantom tube is cut into thin horizontal slabs; each slab removes a share of the light that reaches it; that share is blurred by the kernel for the slab's own distance to plane $k$; the blurred shares are summed. Summing raw absorbances instead over-counts when the stain is dark.

**Layout.** Three slider rows; three stat tiles (centre-line absorbance $\mu d$; dip depth with the partition; dip depth with the plain linear sum, with "(impossible)" appended when > 1). Canvas 680 × 330: (left, 270 × 270) phantom cross-section, $x \in [-1.6, 1.6]$ µm, $z$ up, tube outline, 12 slabs filled green with alpha $\min(0.9, \max(0.12, 0.09/\sigma_j))$ so that slabs near plane $k$ are darker, plane $k$ as an orange dashed line, a "light ↑" marker at the bottom; (right, 340 × 270) profile panel, $I_k/\bar B$ from 1 (top) to 0, with one thin grey curve per slab (its blurred share, drawn as a dip from 1), the orange total, and the blue dashed linear sum. Legend row below.

**Controls.**

| id | label | range | step | default |
|---|---|---|---|---|
| `zk` | Output plane z_k (relative to the tube axis) | −1.2 – 1.2 µm | 0.02 | 0 |
| `d` | Tube diameter d | 0.3 – 2 µm | 0.05 | 1.0 |
| `mu` | Stain μ (per µm) | 0.1 – 5 | 0.1 | 1.5 |

**Model.** $r = d/2$; $J = 12$ slabs of thickness $\delta\zeta = d/J$ centred at $\zeta_j = -r + (j - \tfrac12)\delta\zeta$, numbered from the bottom (light enters from below). Lateral grid: 241 points on $[-1.6, 1.6]$ µm.

- Exact slab absorbance (no voxel staircase): for each fixed $x$ with $|x| < r$, $h(x) = \sqrt{r^2 - x^2}$, and $a_j(x) = \mu \cdot \max\big(0, \min(\zeta_j + \delta\zeta/2,\ h) - \max(\zeta_j - \delta\zeta/2,\ -h)\big)$.
- Absorbed-light partition: $T_{<1} = 1$; $\Delta A_j(x) = T_{<j}(x)\,(1 - e^{-a_j(x)})$; $T_{<j+1} = T_{<j}\,e^{-a_j}$.
- Blur width for a slab at defocus $\delta = \zeta_j - z_k$ (illustrative ideal Debye line-spread core widths, µm): piecewise-linear through $(|\delta|, \sigma)$ = (0, 0.080), (0.14, 0.086), (0.28, 0.122), (0.42, 0.262), (0.56, 0.438), (0.84, 0.603); for $|\delta| > 0.84$: $\sigma = 0.603 + 0.79\,(|\delta| - 0.84)$.
- Each share and each raw absorbance is convolved with a normalised 1-D Gaussian of that $\sigma$, truncated at $\pm 4\sigma$, zero outside the grid.
- Rendered profile: $I_k/\bar B = 1 - \sum_j (\Delta A_j * g_{\sigma_j})$; linear comparison: $1 - \sum_j (a_j * g_{\sigma_j})$, clipped for drawing at $-0.1$.

**Must say in the caption.** Blur widths illustrative (ideal Debye cores, then a linear continuation), not the calibrated kernel; 1-D blur across the tube only; the partition is exact for one slab and to first order for faint stain, a heuristic in between; vertical rays only.

**Acceptance values** **[run]**:

| z_k (µm) | d (µm) | μ (/µm) | μd | dip, partition | dip, linear sum |
|---|---|---|---|---|---|
| 0 | 1.0 | 1.5 | 1.50 | 0.691 | 1.328 (impossible) |
| 0 | 1.0 | 0.1 | 0.10 | 0.084 | 0.089 |
| 0.5 | 1.0 | 1.5 | 1.50 | 0.540 | 1.123 (impossible) |
| 0 | 2.0 | 1.5 | 3.00 | 0.804 | 2.556 (impossible) |
| 0 | 0.3 | 5.0 | 1.50 | 0.667 | 1.212 (impossible) |

Invariant to test: the area of the rendered dip, $\int (1 - I_k/\bar B)\,dx$, is the same for every $z_k$ at fixed $d$ and $\mu$ (blur moves absorbed light, it does not remove it), up to grid-edge truncation.

**Prompt for Claude Design.**
> Build the interactive figure of section "Figure 4" of `TEEG_interactive_figures_spec_2026-10-04.md`, functional reference `fig4_slab_rendering_one_output_plane.html`. Left: phantom tube cross-section with 12 horizontal slabs shaded by sharpness relative to a movable plane k (optical axis vertical, light from below). Right: per-slab blurred absorbed shares (thin grey), their sum (orange, the rendered image of plane k) and the plain linear sum (blue dashed). Implement the exact slab chords, the absorbed-light partition and the blur-width table exactly as specified; reproduce every row of the acceptance table and the area invariant. Optional extension: a fourth slider for a second output plane, drawn side by side, to show that the same slabs appear in every plane with different blur.

---

## Known gaps and open items

- **Figure 2 orientation** is horizontal in the reference file; the spec asks for vertical **[open]**.
- **Figure 1's dip** uses the uniform-disc picture; a faithful rebuild with the $\cos^4$ irradiance weighting (mathematics doc Eq. 17) would make the disc dim at the rim **[open; optional]**.
- **None of the figures uses the calibrated kernel**: they are explanatory, and every width in them is an ideal-objective value. They must not be used to read numbers for the bias table.
- **Not checked:** rendering in Claude Design itself; mobile layout below 380 px; screen-reader labels beyond the SVG `<title>`/`<desc>` of figure 1.

## Sources

Reference implementations: `../figures/*.html`, extracted from the chat widgets of 2026-10-03/04 and wrapped as standalone pages (only change: figure 1's readout label) **[run]**. Acceptance values: Playwright + Chromium, reading the readout elements after setting each slider **[run, 2026-10-04]**. Physics and constants: optics doc §3.4–3.5 and mathematics doc §3.2–3.4 (same folder); Debye widths from `checks/defocus_forms.out` **[run]**.
