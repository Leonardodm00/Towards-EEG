# Diameter re-measurement: ideas for the presentation

**Started:** 2026-10-10, by the theory chat (claude.ai session
`session_017oQJ14njZ9i5nSBhHUDoMh`), at the user's request. **Status:** a running
list of slides and figures to make. Nothing here is a decision; each idea points
to the decisions and documents it rests on.

| Date | Change |
|---|---|
| 2026-10-10 | v1. Idea 1, the defocus kernel in three domains, with its figure (the user's proposal of 2026-10-10, 15:54). |

Each idea gives:
- the message of the slide, in one sentence;
- the content, stated precisely;
- the figure to draw;
- speaker notes in plain words;
- caveats and sources.

---

## Idea 1 -- The defocus kernel in three domains

**Message.** Every slab of a dendrite is blurred by the kernel for its own
distance from the focal plane. That kernel has three domains, each set or
calibrated in a different way.

### Notation (as in the renderers document and procedure §3.4)

| Symbol | Meaning | Units |
|---|---|---|
| $z_k$ | depth of the rendered (focal) plane $k$ | µm |
| $z_{\rm ax}$ | depth of the branch's axis at the node | µm |
| $\zeta_j$ | centre depth of renderer slab $j$ (thickness $\le 0.05$ µm) | µm |
| $\delta=\zeta_j-z_k$ | signed defocus of a slab from the plane | µm |
| $K_\delta$, $\sigma_{\rm r}(\delta)$ | the defocus kernel, a circular Gaussian, and its per-axis width; model level | —, µm |
| $\Delta\sigma^2(\delta)$ | growth of the squared core width with defocus, $\Delta\sigma^2(0)=0$ (procedure Eq. 4) | µm² |
| $\gamma$ | slope of the far-defocus continuation (D-024 (i), D-026.1) | dimensionless |
| $\Delta z$ | recorded plane spacing, 0.28 µm | µm |
| $d$, $\hat d$, $\varphi$ | branch diameter, its fitted value, its tilt | µm, µm, rad |

The calibration returns estimates $\hat\sigma_{\rm r}(\delta)$ and
$\hat\gamma$ (computed level) of the model's $\sigma_{\rm r}(\delta)$ and
$\gamma$ (D-026, levels note).

### The three domains

Each slab $j$ is blurred once, by $K_{\zeta_j-z_k}$. The renderer sums the
slabs; nothing is applied on top of the in-focus blur (procedure §3.4, "the
point to keep").

| Domain | Defocus range | What sets $\sigma_{\rm r}(\delta)$ | How it is obtained | Status now / after calibration |
|---|---|---|---|---|
| 1 -- in focus | $\delta\approx0$; the width is nearly flat within about ±0.14 µm (0.080 → 0.086 µm in the ideal-Debye table) | diffraction: $\sigma_{\rm r}(0)$, the optical core of the in-focus line spread, ≈ 0.080 µm (ideal Debye, 550 nm) | **not identified by the stacks**: in procedure Eq. 4 it cancels into each node's constant $c_i$. Configured, and propagated by rebuilding tables at 0.066 and 0.095 µm (procedure §3.10) | configured / still configured |
| 2 -- slightly off focus | $0<\lvert\delta\rvert\le0.84$ µm (±3 planes) | the growth $\Delta\sigma^2(\delta)$, tabulated per plane offset | plane scans of **thin, faint branches**: $\hat d\lesssim0.3$ µm, faint, flat ($\varphi\lesssim10°$), isolated. Fit of procedure Eq. 4; offsets beyond ±2 planes trusted only after a residual check. Flat branches first (D-027) | configured (ideal-Debye table) / measured |
| 3 -- far off focus | $\lvert\delta\rvert>0.84$ µm | the continuation $\sigma_{\rm r}(\delta)=\sigma_{\rm r}(0.84)+\gamma\,(\lvert\delta\rvert-0.84)$ (D-026.1), further terms only if the residuals require them | the **thick set** (D-026): nodes whose Allen diameter is just below 0.8 µm, read on planes more than 0.84 µm from their axis, because thin shadows fade beyond about three planes. Flat branches first (D-027) | $\gamma=0.79$ configured (D-024 (i)) / $\hat\gamma$ fitted |

**Why the boundaries are where they are** **[KB-repo]**:
- **0.84 µm** is three planes ($3\Delta z$). A thin branch's shadow keeps a
  usable signal-to-noise ratio only up to about the third plane (D-026, "Why").
- **0.8 µm** is a diameter, not a defocus. The thick set's nodes are just below
  it, and a thicker faint node keeps its dip measurable further from focus
  (D-026, "Why").
- **The in-focus band of ±0.14 µm** is a drawing choice **[reasoning]**. It is
  half a plane, and it matches the quarter-wave tolerance (about ±0.15 µm),
  inside which the blur stays the in-focus one (optics Eq. 6a). The table
  itself has no such boundary.

### What it implies, for the slide **[reasoning]**

- **A thin dendrite rendered in its node plane** ($d\lesssim0.3$ µm) has all
  its slabs within $\lvert\delta\rvert\le d/2\approx0.15$ µm. It is drawn
  almost entirely by domain 1, so its corrected diameter depends most on the
  unidentified in-focus core.
- **A thick dendrite** ($d>1.68$ µm) has slabs in all three domains when the
  plane is at its axis.
- **The focus search renders $k^*\pm3$ planes.** That needs
  $\lvert\delta\rvert$ up to about $d/2+0.98$ µm, so every diameter relies
  on domain 3 there (procedure §3.4, "Range").

### Figure (the user's design, with the details filled in)

**Panel A -- the branch across depth.**
- The cross-section of a flat dendrite in the $(y,z)$ plane:
  - $y$ across the branch, horizontal, in µm;
  - $z$ the depth in stage units, vertical, in µm, with the light coming from
    below (direction convention as in the renderers document).
- A circle of diameter $d$ centred at $(0,z_{\rm ax})$. Use $d$ = 2.5–3 µm, so
  that all three domains fall inside it.
- The in-focus axis is the plane $z_k=z_{\rm ax}$, drawn as a solid line.
- The parts of the circle are coloured by $\lvert z-z_k\rvert$:
  - domain 1 for $\le0.14$ µm;
  - domain 2 for 0.14–0.84 µm;
  - domain 3 for $>0.84$ µm.
- The recorded planes $z_k\pm n\Delta z$ ($n$ = 1…5) are dashed lines. A few
  renderer slabs may be hatched to show that they are much thinner than a
  plane.

**Panel B -- the kernel width, sharing the vertical axis.**
- $\sigma_{\rm r}(\delta)$ on the horizontal axis (µm), against $\delta$ on
  the vertical axis (µm), with the same three colours.
- Curve values:
  - the ideal-Debye table, 0.080, 0.086, 0.122, 0.262, 0.438 and 0.603 µm at
    $\lvert\delta\rvert$ = 0, 0.14, 0.28, 0.42, 0.56 and 0.84 µm (optics
    §3.5);
  - then the linear continuation with $\gamma=0.79$.
- Labels on each segment:
  - "configured (diffraction)" on domain 1;
  - "measured on thin faint branches" on domain 2;
  - "thick set + continuation" on domain 3.
- After calibration, replace the curve with $\hat\sigma_{\rm r}(\delta)$.

**Panel C (optional) -- the two calibration sets.**
- A thin node ($d\lesssim0.3$ µm) seen from planes at 0, ±0.28, ±0.56 and
  ±0.84 µm, marked as domain 2.
- A node just below 0.8 µm seen from planes more than 0.84 µm from its axis,
  marked as domain 3.

Use three categorical colours that are safe for colour-blind viewers, and
label the domains in text, not by colour alone.

### Speaker notes (plain words)

"A dendrite is not flat. When we focus on its middle, its top and bottom are
out of focus, so each thin slice of it is blurred by a different amount. Right
at focus the blur is the smallest spot the lens can make, set by diffraction.
We take that spot from optics, because the data cannot tell it apart from the
dendrite's own width. A few tenths of a micrometre away, the blur grows. We
measure that growth on thin, faint branches by stepping through the planes,
and their shadows stay visible for about three planes, 0.84 µm. Further out
those shadows are too faint. There we use branches just under 0.8 µm thick,
whose shadows survive further, and extend the near-focus law with a fitted
slope."

### Caveats

- The user's description of 2026-10-10 says the thick branches "do not exceed
  0.84 µm in diameter". D-026 records "just below 0.8 µm" (Allen diameter); the
  0.84 µm is the defocus boundary, not a diameter.
- The kernel stays Gaussian for now; a non-Gaussian family is idea I-002.
- The kernel models the optics only. Pixel integration, interpolation and JPEG
  come later in the renderer (procedure §3.4).
- **The configured curve of domains 2 and 3 assumes a fully open condenser.**
  The ideal-Debye table is the objective's own defocused line spread, which is
  the incoherent, fully open limit. Allen does not report its condenser
  setting. At ZEISS's recommended 75–80 % aperture, the geometric cone's RMS
  slope would be about half the fully open value (optics v1.5, §3.6 note). The
  calibration ($\Delta\sigma^2(\delta)$, $\hat\gamma$) measures the real growth,
  and so removes that assumption **[reasoning]**.

### Sources

- D-023, D-024 (i), D-026 and D-027, and I-002 (project log; decision-log
  additions file).
- Procedure §3.4 (Eq. 4, "Range", the two kernel families) and §3.10.
- Optics §3.5 (the ideal-Debye table) and Eq. 6a.
- Renderers document §3.2 and §3.6.

All of these are project documents in this folder **[KB-repo]**.
