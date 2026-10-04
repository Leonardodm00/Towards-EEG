# How the diameter bias factor $b(d, \varphi \mid \mathcal{C})$ is tabulated

**Date:** 2026-10-04 (v1, revised the same day after an independent review;
v1.1 the same day adds a walkthrough of one output plane with an interactive
figure, §3.6; see §5, "Revision record"). **Project:** Towards EEG, diameter re-measurement
from the Allen 63× brightfield stacks (first specimen 529878215).
**Companion documents:** `claude/TEEG_diameter_bias_table_mathematics_2026-10-04.md`
("mathematics §n / Eq. (n)") and
`claude/TEEG_microscope_optics_oil_immersion_2026-10-04.md` ("optics §n").
**Sources it builds on:** the user's handoff `handoff_diameter_remeasurement.md`
(2026-09-30, the user's upload, not in project knowledge; "handoff Eq. n"),
`claude/TEEG_diameter_optics_notes.md` ("notes §n"), and decisions D-018 and
D-019 in `TEEG_decisions_and_ideas_log.md`.

**Abstract.** The per-node diameter fit (handoff Eq. 11, amended by D-018 and
D-019) models a dendrite as a flat absorbing tube blurred once, in focus. A
real dendrite is three-dimensional. Its parts above and below the focal plane,
which come from its own thickness and its tilt, add out-of-focus shadow to the
measured profile, so the fitted diameter $\hat d$ of a thick or steep branch
reads systematically different from its true diameter $d$. Other systematic
errors of the measurement chain add to that. The **bias factor**
$b(d, \varphi \mid \mathcal{C})$ measures the total systematic error on
synthetic stacks of known geometry, so it can be removed from the real
measurements. This document is the **procedure**: what has to be fixed before
tabulating, how the defocus kernel is obtained, how the synthetic stacks are
rendered, how they go through the real pipeline, how the table is estimated,
inverted and flagged, and which checks decide whether to trust it. The
derivations are in the mathematics document and the optics in the optics
document. Excluded: the real-data steps upstream of the fit (registration,
snap, focus score, line fit), which are in the handoff and are referred to
only where the synthetic stacks pass through them too. **Nothing here is
coded;** the procedure is a proposal awaiting the open choices in §5.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $i$, $N_{\rm nodes}$ | index of a real SWC dendrite node; number of nodes | $i \in \{1, \dots, N_{\rm nodes}\}$ | — | 3.1 |
| $d$ | true diameter of a phantom tube | $\mathbb{R}_{>0}$ | µm | Abstract |
| $d_i$ | true (unknown) diameter of real node $i$ | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $r$ | true radius, $r = d/2$ | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $\varphi$ | true tilt of a phantom's axis out of the image plane | $[0, \pi/2)$ | rad (quoted in °) | Abstract |
| $\varphi_i$ | tilt of node $i$ computed by the local line fit (handoff Eq. 5) | $[0, \pi/2)$ | rad | 3.1 |
| $\hat\varphi$, $\hat\varphi_n$ | tilt the pipeline computes on a synthetic stack (random variable; realized for replicate $n$) | $[0, \pi/2)$ | rad | 3.7 |
| $\theta_i$, $\theta$ | heading of a branch's projection in the image (handoff Eq. 5); $\theta$ for a phantom | $(-\pi, \pi]$ | rad | 3.5 |
| $\hat D$ | fitted diameter of one synthetic replicate **before** it is drawn (analytic level) | random variable, $\mathbb{R}_{>0}$ | µm | 3.2 |
| $\hat d$, $\hat d_n$ | realized fitted diameter (one number); $\hat d_n$ for replicate $n$ | $\mathbb{R}_{>0}$ | µm | Abstract |
| $\hat d_i$ | fitted diameter of real node $i$ (D-019.1) | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $d^\star_i$ | analytic-level corrected diameter: solves $m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$ | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $\tilde d_i$ | computed corrected diameter, the output: solves $\hat m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$ | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $\mathcal{C}$ | the configuration: every choice the table is conditional on (§3.3) | a fixed tuple of settings | — | Abstract |
| $\mathcal{S}$ | selection event: "fit converged, no bound hit, not flagged", applied identically to phantoms and real nodes | event | — | 3.2 |
| $\xi$ | nuisance draws of one replicate: heading, sub-pixel position, axis depth offset, noise | random vector | mixed | 3.2 |
| $\mathbb{E}_\xi$ | expectation over $\xi$ | operator | — | 3.2 |
| $m(d, \varphi \mid \mathcal{C})$ | mean response $\mathbb{E}_\xi[\hat D \mid d, \varphi, \mathcal{C}, \mathcal{S}]$ | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $b(d, \varphi \mid \mathcal{C})$ | bias factor, $m(d, \varphi \mid \mathcal{C}) / d$ (analytic level) | $\mathbb{R}_{>0}$ | dimensionless | Abstract |
| $\hat b(d, \varphi \mid \mathcal{C})$ | its Monte-Carlo estimate (computed level) | $\mathbb{R}_{>0}$ | dimensionless | 3.1 |
| $\hat m(d, \varphi \mid \mathcal{C})$ | interpolated estimate of $m$, $\hat b(d, \varphi \mid \mathcal{C})\,d$ | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $n$, $N$, $N_{\mathcal{S}}$ | replicate index; replicates run per grid point; replicates in $\mathcal{S}$ | $n \in \{1, \dots, N_{\mathcal{S}}\}$; $\mathbb{N}$ | — | 3.2 |
| $\tau(d, \varphi \mid \mathcal{C})$ | standard deviation of $\hat D / d$ (analytic level); $\hat\tau$ its estimate | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.8 |
| $\beta(d, \varphi \mid \mathcal{C})$ | elasticity $\partial\ln b / \partial\ln d$ (mathematics Eq. 21) | $\mathbb{R}$ | dimensionless | 3.8 |
| $\mathrm{SE}(\hat b)$ | Monte-Carlo standard error of $\hat b$ | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.8 |
| $\epsilon_{\rm MC}$ | target standard error used to choose $N$ | $\mathbb{R}_{>0}$ | dimensionless | 3.8 |
| $\mu$ | absorption coefficient of the stained cytoplasm | $\mathbb{R}_{\ge 0}$ | µm⁻¹ | 3.3 |
| $\hat\mu_i$, $\hat\mu_n$ | fitted $\mu$ of real node $i$; of replicate $n$ | $\mathbb{R}_{\ge 0}$ | µm⁻¹ | 3.3, 3.7 |
| $\alpha$ | centre-line absorbance, $\mu d / \cos\varphi$ | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.5 |
| $v$, $v_0$, $\hat v_{0,n}$ | position across the branch; tube-axis position on the measuring line; its fitted value for replicate $n$ | $\mathbb{R}$ | µm | 3.7 |
| $B$ | background brightness of a rendered stack (model level, configured) | $\mathbb{R}_{>0}$ | grey levels | 3.6 |
| $\bar B$, $\bar B_i$ | background estimate, a median of the focal plane (D-018.1; computed level) | $\mathbb{R}_{>0}$ | grey levels | 3.3 |
| $\mathcal{R}_i$ | region of the focal plane over which $\bar B_i$ is the median (D-018.1) | subset of $\mathbb{R}^2$ | µm² | 3.3 |
| $\sigma_{\rm fit}$ | the fixed blur width inside the fit (handoff Eq. 11) | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $\sigma_{\rm r}(\delta)$ | width of a Gaussian rendering kernel $K_\delta = G_{\sigma_{\rm r}(\delta)}$ | $\mathbb{R}_{>0}$ | µm | 3.3 |
| $\lambda$ | wavelength of the light | $\mathbb{R}_{>0}$ | µm | 3.3 |
| $z$ | depth along the optical axis, in **stage units** (the focus drive's coordinate) | $\mathbb{R}$ | µm | 3.1 |
| $z_{\rm axis}$ | depth of a tube's axis at the node | $\mathbb{R}$ | µm | 3.1 |
| $\Delta z$ | spacing between recorded planes, 0.28 µm | $\mathbb{R}_{>0}$ | µm | 3.3 |
| $k$, $z_k$ | index and depth of an image plane (recorded or rendered) | $k \in \mathbb{Z}$, $z_k \in \mathbb{R}$ | —, µm | 3.4 |
| $k^*$, $k^*_n$ | sharpest plane chosen by handoff Eq. 1; for replicate $n$ | $\mathbb{Z}$ | — | 3.1 |
| $\delta$ | signed defocus: depth of an object point **minus** depth of the plane being imaged, $\delta = z - z_k$ | $\mathbb{R}$ | µm | 3.3 |
| $K_\delta$ | rendering kernel for an object slab at defocus $\delta$ (2-D, unit integral) | function $\mathbb{R}^2 \to \mathbb{R}_{\ge 0}$ | µm⁻² | 3.3 |
| $G_\sigma$ | circular 2-D Gaussian, standard deviation $\sigma$ per axis, unit integral | function $\mathbb{R}^2 \to \mathbb{R}_{>0}$ | µm⁻² | 3.4 |
| $\mathrm{LSF}_\delta$ | line-spread function at defocus $\delta$, unit integral | function $\mathbb{R} \to \mathbb{R}_{\ge 0}$ | µm⁻¹ | 3.4 |
| $V$, $V_W$ | squared width of a dip or kernel (darkness-weighted mean squared distance from its centre); $V_W$ over a window of half-width $W$ | $[0, \infty]$; $\mathbb{R}_{\ge 0}$ | µm² | 3.4 |
| $W$ | half-width of the window of a windowed moment | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $V_{\rm tube}$ | squared width of the tube's own unblurred absorbed fraction | $\mathbb{R}_{\ge 0}$ | µm² | 3.4 |
| $\omega_{i,k}$ | per-plane width statistic of calibration node $i$ in plane $k$ (a windowed $V$, or a squared Gaussian-core width) | $\mathbb{R}_{\ge 0}$ | µm² | 3.4 |
| $A_{i,k}$ | area of that dip within the window | $\mathbb{R}_{\ge 0}$ | grey levels·µm | 3.4 |
| $c_i$ | per-node constant of the calibration fit | $\mathbb{R}$ | µm² | 3.4 |
| $z_{{\rm ax}, i}$ | depth of calibration node $i$'s axis (fitted) | $\mathbb{R}$ | µm | 3.4 |
| $\Delta\sigma^2(\delta)$ | growth of the width statistic with defocus, shared by all nodes, $\Delta\sigma^2(0) = 0$ | function $\mathbb{R} \to \mathbb{R}$ | µm² | 3.4 |
| $\varepsilon_{i,k}$ | residual of the calibration fit | $\mathbb{R}$ | µm² | 3.4 |
| $j$, $J$ | index and number of object slabs of a phantom | $j \in \{1, \dots, J\}$ | — | 3.6 |
| $\zeta_j$, $\delta\zeta$ | depth of slab $j$ (stage units); slab thickness | $\mathbb{R}$, $\mathbb{R}_{>0}$ | µm | 3.6 |
| $h_{\rm g}$ | lateral step of the fine rendering grid | $\mathbb{R}_{>0}$ | µm | 3.6 |
| $a_j(x, y)$ | absorbance of slab $j$ along the vertical ray through $(x, y)$ | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.6 |
| $T$, $T_{<j}$ | transmittance; fraction of the incident light reaching slab $j$ | $[0, 1]$; $(0, 1]$ | dimensionless | 3.6 |
| $\Delta A_j(x, y)$ | fraction of the light **incident on the column** that is absorbed in slab $j$ | $[0, 1]$ | dimensionless | 3.6 |
| $I_k(x, y)$ | grey level (real) or intensity (rendered) of plane $k$ at $(x, y)$ | $\mathbb{R}_{\ge 0}$ | grey levels | 3.6 |
| $(x, y)$ | lateral position in the global image frame, specimen-referred | $\mathbb{R}^2$ | µm | 3.6 |
| $p_{\rm x}$ | pixel pitch in the specimen, 0.1144 µm (handoff `res0`) | $\mathbb{R}_{>0}$ | µm | 3.3 |
| $g$ | handoff Eq. 11's 1-D Gaussian blur of width $\sigma_{\rm fit}$ | function $\mathbb{R} \to \mathbb{R}_{>0}$ | µm⁻¹ | 3.6 |
| $L$ | length of the line-fit window along the branch (handoff Eq. 4) | $\mathbb{R}_{>0}$ | µm | 3.3 |

### Conventions

- **Equation numbers.** "Eq. (n)" without prefix is this document's own;
  "handoff Eq. n", "mathematics Eq. (n)", "optics Eq. (n)" refer to the other
  documents.
- Lengths in µm. $x$, $y$ are specimen-referred (through the magnification);
  $z$ is in **stage units** (the distance the focus drive moves), which differs
  from the true depth inside the tissue when the mounting medium's refractive
  index differs from the oil's (optics §3.7). Every $z$-quantity here
  (planes, slabs, tilt, defocus calibration) is in stage units, so the
  procedure is internally consistent as far as the defocus calibration is
  concerned; a phantom that is round in ($x$, $y$, stage $z$) is round in the
  tissue only if the media are index-matched (§5).
- Angles in radians in formulas, quoted in degrees in text.
- **Level of an object (rule R8).** $\hat D$ is the fitted diameter as a random
  variable; $\hat d$ one realized number; $m$, $b$, $\tau$, $d^\star_i$ are
  analytic-level; $\hat b$, $\hat\tau$, $\hat m$, $\tilde d_i$ computed-level.
- **Scoped abuse of notation.** In §3.8–§3.10 and in tables, "$\mid \mathcal{C}$"
  is dropped from $b$, $\hat b$, $m$, $\hat m$, $\tau$ for readability; every
  such object there is conditional on $\mathcal{C}$ and $\mathcal{S}$. Full
  notation is restored in §4.
- "Real" means Allen data; "synthetic" or "phantom" means rendered here.
- Convolution $*$ is in the lateral plane $(x, y)$ unless stated.

---

## 2. Glossary

Ordered by first appearance, because the concepts build on each other.

- **Bias factor $b$** (Abstract). How much the whole measurement chain inflates
  (or shrinks) a tube of known diameter: the average fitted diameter divided by
  the true one. $b = 1$ means no systematic error.
- **Phantom** (§3.1). A synthetic object of known geometry, here a straight
  tube of chosen $d$ and $\varphi$, rendered into a fake image stack.
- **Configuration $\mathcal{C}$** (§3.2). Everything the table depends on
  besides $d$ and $\varphi$. Changing any element invalidates the table.
- **Selection $\mathcal{S}$** (§3.2). The rule deciding which fits count. The
  same rule must be applied to phantoms and to real nodes, or the table
  corrects a different population.
- **Nuisance draw $\xi$** (§3.2). A setting that varies at random between real
  nodes and is not of interest (heading, sub-pixel position, axis depth,
  noise). It is drawn at random and averaged over, never tabulated.
- **Analytic level / computed level** (§3.2). The first is a property of a
  distribution (an expectation); the second is a number computed from a finite
  batch of replicates that estimates it.
- **Defocus** (§3.3). Distance along the optical axis between an object point
  and the plane the microscope is focused on.
- **Line-spread function (LSF)** (§3.4). The blur profile across a straight
  thin line: the point-spread function (PSF) summed along the line.
- **Rendering kernel $K_\delta$** (§3.4). The 2-D blur the simulator applies to
  an object slab at defocus $\delta$. A model of the optics only; pixelation
  and interpolation are applied separately.
- **Squared width $V$** (§3.4). The darkness-weighted mean squared distance of
  a dip from its centre. *Everyday meaning differs:* often called "variance",
  but nothing random is involved. For real LSFs it depends on the window
  (mathematics §3.3).
- **Gaussian-core width** (§3.4). The width of a Gaussian fitted to the
  central part of a profile only; insensitive to the tails.
- **Object slab** (§3.6). A thin layer of the phantom, perpendicular to the
  optical axis. **Not** an image plane.
- **Absorbed-light partition** (§3.6). Splits the light a column of the object
  absorbs among the slabs it passes through, in the order the light meets
  them.
- **Beer–Lambert law** (§3.6). Transmitted fraction $= \exp(-\text{absorbance})$,
  absorbance $=$ absorption coefficient × path length.
- **Pixel integration** (§3.6). A camera pixel reports the light averaged over
  its area, not sampled at its centre.
- **Inversion** (§3.9). Recovering $d$ from a measured $\hat d_i$ by solving
  $\hat m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$.
- **Flag** (§3.9). A per-node mark that the correction is not trusted; flagged
  nodes are filled from neighbours (handoff D5).

---

## 3. Main body

### 3.1 What the table is for, and where it sits

This section fixes the problem the table solves. The per-node chain on real
data ends with

$$\tilde d_i = \text{the solution } d \text{ of } \; \hat m(d, \varphi_i \mid \mathcal{C}) = \hat d_i, \qquad \hat m(d, \varphi \mid \mathcal{C}) = \hat b(d, \varphi \mid \mathcal{C})\, d, \tag{1}$$

where $\hat d_i$ is the diameter fitted at node $i$ and $\varphi_i$ its tilt
from the line fit. Eq. (1) is the computed-level version of the analytic
inverse $d^\star_i$, which solves $m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$
(mathematics Eq. (20)). The handoff wrote the shortcut
$\hat d_i / b(\hat d_i, \varphi_i \mid \mathcal{C})$; it differs from Eq. (1) by
about $-\beta\,(b - 1)$ in relative terms (mathematics Eq. (21)), so it is close
only where $b \approx 1$ or $b$ hardly changes with $d$.

**Where the bias comes from.** The fit (handoff Eq. 11, as amended by D-018 and
D-019) assumes one flat shadow, blurred once by the in-focus kernel of fixed
width $\sigma_{\rm fit}$. Two things a real tube does are absent from it:

1. **Its own thickness.** At the node, a tube occupies the depths
   $z_{\rm axis} \pm r/\cos\varphi$ on the vertical ray through its axis
   ($\pm d/2$ when flat). Its upper and lower layers are out of focus in the
   plane $k^*$ where it is measured; their shadows are wider and widen the
   profile, even at $\varphi = 0$.
2. **Its tilt.** Along a tilted branch, stretches a little ahead and behind the
   node sit at other depths. Their defocused shadows spread onto the measuring
   line (handoff Eq. 13(b): a geometric estimate puts the onset near
   $\varphi \approx 22°$; not simulated).

Other systematic errors of the chain also enter $b$: a $\sigma_{\rm fit}$
different from the true in-focus blur, the $\bar B$ median's bias, pixelation,
JPEG, noise acting through a non-linear fit, and the tilt estimate
(mathematics §3.7). The handoff's check that a noise-free fit recovers $d$
exactly was made with the correct $\sigma$, no camera chain, and
$d \in \{0.5, 1, 2\}$ µm, $\alpha \in \{0.3, 3\}$ only. So: **the defocus part of
$b$ comes only from what the simulator adds beyond handoff Eq. 11** — the
tube's depth structure and the defocus kernel — and that is why most of this
procedure is about rendering; **$b$ also absorbs every other systematic error
the simulator reproduces.**

The chain, read left to right:

```
calibrate the defocus kernel K_delta from real stacks (3.4)
  -> render phantom stacks of known (d, phi) (3.5, 3.6)
  -> run the real pipeline, unchanged (3.7)
  -> estimate b_hat(d, phi | C) and its spread (3.8)
  -> invert at each real node, flag, fill (3.9)
```

### 3.2 What exactly is tabulated

§3.1 named the factor; this section says which object it is, on which level.
For a phantom of true diameter $d$ and tilt $\varphi$, one replicate draws the
nuisances $\xi$ (§3.5), renders a stack, and runs the pipeline. Before the
draw its output is the random variable $\hat D$. Replicates outside the
selection $\mathcal{S}$ are discarded, by the same rule used on real nodes.
The target is the analytic-level quantity

$$m(d, \varphi \mid \mathcal{C}) = \mathbb{E}_\xi\big[\hat D \mid d, \varphi, \mathcal{C}, \mathcal{S}\big], \qquad b(d, \varphi \mid \mathcal{C}) = \frac{m(d, \varphi \mid \mathcal{C})}{d}. \tag{2}$$

$b$ in Eq. (2) is estimated by the computed-level average over the
$N_{\mathcal{S}}$ retained replicates with realized outputs
$\hat d_1, \dots, \hat d_{N_{\mathcal{S}}}$:

$$\hat b(d, \varphi \mid \mathcal{C}) = \frac{1}{N_{\mathcal{S}}} \sum_{n=1}^{N_{\mathcal{S}}} \frac{\hat d_n}{d}. \tag{3}$$

For independent, identically drawn replicates, $\hat b$ is one realization of
a random quantity whose mean is $b$ and whose standard error is
$\tau/\sqrt{N_{\mathcal{S}}}$ (§3.8; mathematics §3.6).

The conditioning on $\mathcal{C}$ is written because the table is **a property
of the estimator**, not of the microscope alone. D-019 open point (c) records
this already: a per-node-$\mu$ fit and a shared-$\mu$ fit need their own
tables.

### 3.3 What must be fixed before tabulating: the configuration $\mathcal{C}$

§3.2 made $b$ conditional on $\mathcal{C}$; this section lists $\mathcal{C}$.
The rule: **the synthetic run must use exactly the code and settings used on
the real data**, and the simulator's own choices must be recorded too, so the
correction code can refuse a mismatched table (§3.11).

*Estimator settings (shared by real and synthetic runs):*

| Element | Status (2026-10-04) |
|---|---|
| Fit parameterisation: $(d, \mu, v_0)$ free, $\varphi_i$ fixed (D-019.1); `least_squares` bounds | decided (D-019); bounds not yet set |
| $\mu$ per node or shared | **open** (D-019 (a)); two tables if both are used |
| $\bar B$ rule: median over $\mathcal{R}_i$ (D-018.1); masking rule as applied to phantoms | decided; region **open** (D-018 (a)); phantoms need the same mask rule even though they have no Allen radius |
| Focus-score background rule $B_{j,k}$ of handoff Eq. 1 | **open** (D-018 (c)) |
| $\sigma_{\rm fit}$ | **open**: bracket of handoff Eq. 12; ≈ 0.099 µm at 550 nm as a heuristic budget (notes §2) |
| Profile: half-length (±3 µm), step, bilinear interpolation, optional ±0.5 µm averaging along the branch (handoff Eq. 9) | proposed in the handoff |
| Sub-plane depth (handoff Eq. 2) used or not | optional in the handoff; fix one choice |
| Line-fit window $L$, block size, planes per block | proposed defaults, handoff "Next actions" 4 |
| Selection rule $\mathcal{S}$ | to define |
| Mean or median of $\hat d/d$ | **open** (§3.8) |

*Simulator settings (synthetic run only):*

| Element | Status |
|---|---|
| Rendering kernel $K_\delta$, incl. its in-focus core $\sigma_{\rm r}(0)$ and the wavelength $\lambda$ it assumes | **open** (§3.4); $\sigma_{\rm r}(0)$ is **not identified** by the stacks (≈ 0.080 µm, ideal Debye at 550 nm; 0.066–0.095 µm over 450–650 nm, notes §2) |
| Phantom geometry: straight tube, **cross-section aspect ratio** (z / lateral) | round assumed; **open** (§3.5) |
| Phantom $\mu$ | matched to the real $\hat\mu_i$ distribution (§3.5) |
| Background $B$ | configured, matched to real backgrounds |
| Nuisance distributions (heading, sub-pixel offset, axis depth, noise) | §3.5 |
| Slab thickness $\delta\zeta$, fine grid step $h_{\rm g}$ | §3.6; by a convergence test |
| Camera chain: $p_{\rm x} = 0.1144$ µm, $\Delta z = 0.28$ µm, black-level offset and grey-level mapping, noise, 8-bit, JPEG tables | pitch and step fixed (handoff; Gouwens 2019 gives 0.114 µm and 0.28 µm for the mouse pipeline); offset and JPEG tables not yet read (notes §2, §8) |

Plainly: the table answers "how wrong is **this** measurement on a tube of
known size, under **this** model of the microscope". If either changes, so
does the answer.

### 3.4 Step A — the defocus kernel

§3.3 left the kernel open; this section says what it must be and how to get it
from the real stacks. It is the only place the microscope's 3-D behaviour
enters the simulator.

**What the kernel is.** For an object slab at signed defocus $\delta$ from the
plane being rendered, $K_\delta(x, y)$ is the blur its shadow receives. It
models **the optics only**. Pixel integration, bilinear interpolation (handoff
Eq. 9), quantisation and JPEG are applied later (§3.6) and must not be folded
into $K_\delta$, or they would be counted twice.

**The point to keep (notes §4).** Moving from the in-focus plane to a
defocused one does not put a second blur on top of the in-focus one: the
whole line-spread function changes shape. For a thin single-depth tube and
kernels with finite second moments, only squared widths add,
$V[\text{dip in plane } k] = V_{\rm tube} + V[\mathrm{LSF}_\delta]$
(mathematics Eq. (15)–(16)). An extra kernel $q$ with
$\mathrm{LSF}_\delta = \mathrm{LSF}_0 * q$ exists when both are Gaussian, not in
general. Real LSFs have tails whose full second moment is infinite at every
$\delta$, so on data this holds only approximately, for windowed $V_W$
(mathematics §3.3).

**Calibration from the stacks (the user's proposal, notes §5).** On thin
($\hat d \lesssim 0.3$ µm), faint, flat ($\varphi \lesssim 10°$), isolated nodes,
keep the measuring line fixed in $(x, y)$ and step through the neighbouring
planes $k$. In each plane compute a width statistic $\omega_{i,k}$ of the dip
and its area $A_{i,k}$, with the same window and $\bar B$ rule in every plane.
Fit

$$\omega_{i,k} = c_i + \Delta\sigma^2(z_{{\rm ax}, i} - z_k) + \varepsilon_{i,k}, \qquad \Delta\sigma^2(0) = 0, \tag{4}$$

with a per-node constant $c_i$ (it absorbs the tube's own width and the
in-focus blur), a per-node axis depth $z_{{\rm ax}, i}$, and one curve
$\Delta\sigma^2(\cdot)$ shared by all nodes and **tabulated per plane offset**,
not forced into an analytic form (mathematics §3.4). Three facts limit what it
gives (mathematics §3.5): the in-focus part is not identified (it cancels into
$c_i$); the curve is identified only up to a common shift, pinned by a
convention (symmetry, or origin at the minimum); and residuals beyond ±2
planes should be checked before trusting them (notes §5).

**From the calibration to a rendering kernel: what to match.**
[corrected 2026-10-04, twice] An earlier chat answer said to use "a circular
Gaussian with $\sigma(\delta)$ matched in second moment", and the first version
of this document said to match the growth of the windowed $V$. Both are wrong
for the same reason: the real LSF's windowed $V$ is dominated by its tails. In
the ideal Debye model in focus, the Gaussian-core width is 0.080 µm, while
$\sqrt{V_W}$ is 0.240 µm ($W = 1.5$ µm) and 0.348 µm ($W = 3$ µm) **[run,
`psf_check.py`, `doc_checks.py` C5]**, and a Gaussian matched to the $V_W$
growth comes out 1.5–2× too wide in the core within one plane of focus
(mathematics §3.4). A Gaussian can match one statistic only, and it must be
the one that drives $b$: the fit and the focus score respond mostly to the
core. So:

1. use, as $\omega_{i,k}$ in Eq. (4), the **squared Gaussian-core width** of
   the dip (fitted within about one FWHM), with the windowed $V$ kept as a
   cross-check;
2. tune $\sigma_{\rm r}(\delta)$ at each tabulated offset so that rendered
   thin-node phantoms — same $\hat d$ as the calibration nodes, full camera
   chain, same profile extraction — reproduce the **mean real profile** of the
   calibration nodes plane by plane (least squares on the profiles, or on
   their core widths);
3. treat the empirical-kernel comparison of §3.10 as the decisive test.

**Two kernel families.**

- *Gaussian:* $K_\delta = G_{\sigma_{\rm r}(\delta)}$, tuned as above. It is
  shift-invariant and depends on $\delta$ only — the "local" assumption, tested
  by the shallow-versus-deep comparison of Eq. (4). Its in-focus value
  $\sigma_{\rm r}(0)$ is the **optical** core width (≈ 0.080 µm, ideal Debye,
  550 nm), not the effective ≈ 0.099 µm, which already includes pixelation
  and interpolation; it is not identified by Eq. (4) and is a configured
  input whose uncertainty must be propagated (§3.10).
- *Empirical:* averaged, background-normalised real profiles at each offset.
  Usable only after four corrections: (i) they are 1-D, while Eq. (6) needs a
  2-D kernel — either reconstruct a circular 2-D PSF from the averaged LSF
  (inverse Abel transform, assuming rotational symmetry) or restrict the
  family to the $\varphi = 0$ check; (ii) they already contain pixel
  integration, interpolation and JPEG, which the simulator would add again —
  remove them (deconvolve the sampling chain) or bypass the camera chain for
  this family; (iii) they contain the tube's own width, which is **not**
  negligible near focus (at $d = 0.3$ µm, $V_{\rm tube} \approx 0.0056$ µm²
  against $0.080^2 = 0.0064$ µm² for the in-focus core), so subtract it in
  $V$ or deconvolve; (iv) register them to $\delta$ from the fitted
  $z_{{\rm ax}, i}$, not to the plane offset.

**Range.** The calibration covers the offsets scanned (±3 planes = ±0.84 µm;
trust beyond ±2 after a residual check). The measuring plane needs kernels out
to about $r/\cos\varphi + \Delta z/2$ from the axis: for a flat tube, ±0.84 µm
covers $d \lesssim 1.4$ µm. But the focus search renders $k^* \pm 3$, which
needs $|\delta|$ up to about $d/2 + 3.5\,\Delta z \approx d/2 + 0.98$ µm — beyond
the calibrated range for every $d$. So the synthetic $k^*$ always relies on
extrapolated kernels; the truncation test of §3.10 must include agreement of
$k^*$ between real and synthetic stacks.

### 3.5 Step B — the phantom grid and the nuisance draws

With the kernel fixed, this section fixes what is rendered. Every replicate
is a straight tube of true diameter $d$, tilt $\varphi$ and absorption
coefficient $\mu$, placed in a block of the size the real pipeline fetches
(≈ 10 × 10 µm, enough planes to contain the tube), and long enough for the
line-fit window $L$ plus margin.

**Grid (proposal, to confirm).** Uneven spacing where $b$ changes fast:

- $d \in \{0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0\}$ µm (roughly
  logarithmic; the cell's mean SWC diameter is 0.53 µm, handoff);
- $\varphi \in \{0, 5, 10, 15, 20, 25, 30, 40, 50, 60\}°$ (dense around the
  expected onset near 20–30°).

**Cross-section shape — an assumption with a large effect.** The thickness
halo (§3.1, item 1) is set by the tube's vertical extent. The handoff records
strong $z$-shrinkage of mounted slices (Mohan et al.: 63 ± 10 %, their
preparation; Allen's factor unverified), and D1 dropped the only test of
whether dendrites flatten with the slice or stay round. If they flatten,
round phantoms overstate the halo for thick branches. Add the aspect ratio
($z$ extent / lateral extent) to $\mathcal{C}$, start with round, and test
squashed phantoms at bracketing ratios (§3.10).

**Phantom $\mu$.** Choose it so that the **distribution of fitted $\hat\mu_n$**
on phantoms matches the distribution of real $\hat\mu_i$ over the same class
of nodes — matching through the measured statistic. Do not plug in the median
of $\hat\mu_i$ over thick nodes directly: D-019 (b) keeps thick or steep
nodes out of any $\mu$ estimate, because their measured darkness is not
$\mu d / \cos\varphi$ — the very effect $b$ measures. Then
$\alpha = \mu d / \cos\varphi$ follows from $d$ and $\varphi$. If $\hat\mu_i$ is
non-uniform, $\mu$ becomes a third axis, $b(d, \varphi, \mu \mid \mathcal{C})$.

**Nuisance draws $\xi$, one fresh set per replicate:**

| Draw | Distribution | Why |
|---|---|---|
| heading $\theta$ | uniform on $[0, \pi)$ | the pixel grid is not isotropic; see the hypothesis below |
| sub-pixel lateral offset | uniform within one pixel in $x$ and $y$ | the tube axis does not sit on a pixel centre |
| axis depth relative to the nearest plane | uniform on $[-\Delta z/2, \Delta z/2] = [-0.14, 0.14]$ µm | the axis does not sit on a recorded plane |
| noise | the real stacks' noise, matched after the full camera chain (§3.6) | the fit's scatter |

**Hypothesis.** Averaging over these distributions corrects real nodes only if
the real nodes follow the same distributions, or if $b$ does not depend on the
draw. Real headings are not uniform (apical dendrites run along the
pia–white-matter axis), so §3.10 checks that $\hat b$ does not depend on
$\theta$.

### 3.6 Step C — rendering a phantom stack

The grid says what to draw; this section says how one stack is drawn. It also
answers "do we blur the off-focus image planes and add them to the focal
one?" **No.** Each recorded plane already contains every depth's contribution,
each through its own defocus kernel. Blurring neighbouring planes and adding
them would count each slab several times, and stack blur on blur, while
defocus kernels do not compose that way ($K_a * K_b \ne K_{a+b}$). The
simulator starts from the **object**, cut into slabs, and builds **each output
plane separately**.

1. **Fine object grid.** Lateral step $h_{\rm g}$ finer than the pixel (e.g.
   $p_{\rm x}/8 \approx 0.014$ µm); slabs of thickness $\delta\zeta \le 0.05$ µm in
   $z$ (stage units). Settle both by a convergence test (halve them; $\hat b$
   must not change beyond its SE).
2. **Exact slab absorbance.** For slab $j$ (depths
   $[\zeta_j - \delta\zeta/2, \zeta_j + \delta\zeta/2]$), the absorbance along the
   vertical ray through $(x, y)$ is $\mu$ times the **length of the part of that
   ray lying both inside the slab and inside the tube**, which is closed-form
   from handoff Eqs. 6 and 10. Do not use a binary inside/outside voxel test:
   at $d = 0.2$ µm and $\delta\zeta = 0.05$ µm it gives four slabs with local
   chord errors up to about 25 % (reviewer's estimate, not run).
3. **Absorbed-light partition** (proposal, mathematics §3.2). Number the slabs
   in the direction the light travels (from the condenser up to the
   objective; the sign of Allen's plane index relative to that direction is
   **not verified**, and it matters only for dark tubes):
   $$T_{<j}(x, y) = \exp\Big(-\sum_{j' < j} a_{j'}(x, y)\Big), \qquad \Delta A_j(x, y) = T_{<j}(x, y)\,\big(1 - e^{-a_j(x, y)}\big). \tag{5}$$
4. **One output plane $k$.** Each slab's absorbed fraction is blurred by the
   kernel for its own defocus from that plane, and the results are summed:
   $$I_k(x, y) = B\,\Big[\,1 - \sum_{j=1}^{J} \big(\Delta A_j * K_{\zeta_j - z_k}\big)(x, y)\Big]. \tag{6}$$
   $B$ is the model-level background, configured to match real backgrounds;
   the pipeline then estimates it as $\bar B$ from the synthetic plane, exactly
   as on real data. With a Gaussian kernel this is one
   `scipy.ndimage.gaussian_filter` call per slab, with
   `sigma = sigma_r(zeta_j - z_k) / h_g`, `mode='constant'`, `cval=0`, on a block
   padded by at least $3\sigma_{\rm r}$ (the default `mode='reflect'` would mirror
   the tube at the block edges once the kernel is wide).
5. **Repeat for every output plane** the pipeline needs: the planes from the
   axis plane − 3 to + 3, plus the depth span of the tube over the line-fit
   window, so the synthetic stack also exercises the focus search (handoff
   Eq. 1). The same slab contributes to every plane, with a different kernel
   each time.
6. **Camera chain, in this order:** pixel integration (block-average the fine
   grid to $p_{\rm x} = 0.1144$ µm), black-level offset and grey-level mapping,
   noise, 8-bit quantisation, JPEG with Allen's tables. Choose the injected
   noise so that the background statistics **after** the whole chain match the
   real ones (noise measured on JPEG'd 8-bit data already contains
   quantisation and compression). Bilinear interpolation is **not** applied
   here: the pipeline applies it when it reads the profile (handoff Eq. 9).

**Why Eq. (6) is a reasonable proposal** (mathematics §3.2; checked
numerically **[run, `doc_checks.py` C1–C3]**):

- the absorbed fractions add up to the Beer–Lambert value of the whole column,
  $\sum_j \Delta A_j = 1 - e^{-\sum_j a_j}$: no light is created or lost;
- if the whole column's absorbance is placed in a single slab at $\delta = 0$
  and $K_0 = g$ (handoff Eq. 11's kernel), with $\varphi = 0$ and no camera
  chain, Eq. (6) reproduces handoff Eq. 11's model $B\,(T * g)$ exactly;
- for a faint tube it reduces to the linear sum of blurred slab absorbances.
  For a darker one the linear sum is badly wrong: at $d = 0.5$ µm,
  $\varphi = 0$, nine slabs with illustrative blur
  $0.08 + 1.2|\zeta_j - z_{\rm axis}|$ µm, it overstates the dip depth by 86 %
  at centre-line absorbance 1.5 and by 1 % at 0.025 **[run, C3]**.

What Eq. (6) is **not**: a validated model of 3-D image formation for thick
absorbers. It blurs each slab's absorbed light independently (light shaped by
lower slabs is ignored), and it traces absorption along **vertical** rays,
while under an NA 1.4 condenser the light crosses the specimen at up to 67.5°,
on longer, laterally displaced paths (mathematics §5).

**Walkthrough of one output plane, with the interactive figure** (added v1.1).
`../figures/fig4_slab_rendering_one_output_plane.html` draws steps 3–4 for a
flat tube ($\varphi = 0$) cut into 12 slabs, with the diameter $d$, the
absorption coefficient $\mu$ and the output plane $z_k$ as sliders. Read it in
this order:

1. **Slice the object** (left panel). The circle is the tube's cross-section;
   the horizontal bands are the slabs $j$, each with its exact chord through
   the circle (step 2). The dashed line is the output plane $z_k$; a slab's
   shading says how sharp it will be in that plane, darkest at $\zeta_j = z_k$.
2. **Partition the light, from below** (step 3, Eq. 5). The bottom slab takes
   its share of the full beam; each slab above takes its share of what is left.
   Equal slabs therefore get unequal shares (mathematics §3.2, worked table).
3. **Blur each share by its own distance** to the plane, $K_{\zeta_j - z_k}$
   (the grey curves in the right panel): narrow and tall for slabs near the
   plane, wide and low for slabs far from it.
4. **Sum** (Eq. 6): the orange curve is $I_k/B$, the profile that plane $k$
   records before the camera chain. The blue dashed curve is the linear sum of
   blurred slab absorbances; the gap between the two is the error the partition
   removes, and it grows with $\mu d$.
5. **Move the plane and repeat** (step 5). Dragging $z_k$ re-weights the same
   slabs with new kernels; nothing from the previous plane is reused.
6. **Camera chain** (step 6) is not in the figure.

Reference readouts of the figure, centre-line absorbance $\mu d$ and dip depth
($1 - I_k/B$ at the centre) with the partition and with the linear sum, are
**[run, Playwright probe of the figure, 2026-10-04]**: $z_k = 0$, $d = 1$ µm,
$\mu = 1.5$ µm⁻¹ → 1.50, 0.691, 1.328; $\mu = 0.1$ µm⁻¹ → 0.10, 0.084, 0.089.
The figure's blur widths are illustrative ideal-Debye values, not the
calibrated kernel of §3.4. Its reproduction instructions are in
`TEEG_interactive_figures_spec_2026-10-04.md`, figure 4.

### 3.7 Step D — measuring the phantoms with the real pipeline, unchanged

The stack now looks like Allen's; this section runs it through exactly the
code used on real nodes, so that the bias measured is the bias of that code.

For each replicate, in order (handoff "Per-node pipeline", steps 3–5): focus
score per plane and $k^*$ (handoff Eq. 1) → centres (handoff Eq. 3) → line fit
through centres, giving $\hat\varphi$ and the measuring axis (handoff Eqs. 4–5)
→ profile in plane $k^*$ with bilinear interpolation (handoff Eq. 9) →
$\bar B$ by the D-018 rule → fit (D-019.1) with the same $\sigma_{\rm fit}$ →
$\hat d_n$. To run the line fit, the phantom must provide centres over the
window $L$ (≈ 4 µm plus margin) and be measured at several consecutive
"nodes" along it; use the middle one.

**Record per replicate:** $\hat d_n$, $\hat\mu_n$, $\hat v_{0,n}$,
$\hat\varphi_n$, $k^*_n$ against the plane nearest the phantom's axis at that
node, fit status (converged, at a bound, failed), and every flag the real
pipeline would raise.

**Which tilt indexes the table.** The table is indexed by the **true**
$\varphi$, while at a real node only the estimate $\varphi_i$ is known. Using
$\varphi_i$ in Eq. (1) is harmless if
$|\partial b/\partial\varphi|\cdot|\hat\varphi - \varphi|$ — including any
systematic offset of $\hat\varphi$, read from the stored $\hat\varphi_n$ — is
below the tolerance on $b$ (or its SE). If not, average $b(d, \varphi)$ over
the conditional distribution of $\varphi$ given $\hat\varphi$ estimated from the
phantoms. (A table "indexed by $\hat\varphi$" directly is a different object,
which depends on the distribution of true tilts assumed.)

### 3.8 Step E — estimating the table

With the outputs per grid point, this section turns them into $\hat b$ and
says how large $N$ must be.

At each grid point $(d, \varphi)$, from the $N_{\mathcal{S}}$ retained replicates:

$$\hat b = \frac{1}{N_{\mathcal{S}}} \sum_{n=1}^{N_{\mathcal{S}}} \frac{\hat d_n}{d}, \qquad \hat\tau^2 = \frac{1}{N_{\mathcal{S}} - 1} \sum_{n=1}^{N_{\mathcal{S}}} \Big(\frac{\hat d_n}{d} - \hat b\Big)^2, \qquad \mathrm{SE}(\hat b) = \frac{\hat\tau}{\sqrt{N_{\mathcal{S}}}}. \tag{7}$$

- **Per-node error bar after correction.** $\hat\tau$ is the scatter of
  $\hat d/d$ **before** inversion. After inversion the relative scatter of a
  single corrected node is about $\hat\tau / [\hat b\,(1 + \beta)]$, because
  $\partial m/\partial d = b\,(1 + \beta)$ (mathematics §3.6). This is a lower
  bound: it omits the table's own SE and systematics and the real-world
  variability the phantoms lack (beads, spines, stain).
- **Choosing $N$.** Run a pilot (e.g. $N = 50$) at a few corners of the grid,
  read $\hat\tau$, set $N_{\mathcal{S}} = (\hat\tau / \epsilon_{\rm MC})^2$ for a
  target SE $\epsilon_{\rm MC}$ (e.g. 0.005). With $\hat\tau = 0.05$ that gives
  100. These numbers are illustrations, not measurements.
- **Failure rate.** Record $1 - N_{\mathcal{S}}/N$ per grid point. A high rate
  means the estimator breaks there; that region is flagged, not averaged
  over.
- **Mean or median.** Eq. (2) uses the mean. If $\hat d_n/d$ is skewed or has
  outliers (likely where the fit is degenerate, at small $d$), a median-based
  table is more robust but estimates a different quantity. Decide once and
  record it in $\mathcal{C}$.

### 3.9 Step F — applying the table at a real node

The table exists only at grid points; this section uses it at a real node.

1. **Interpolate** $\hat m(d, \varphi_i) = \hat b(d, \varphi_i)\,d$ between grid
   points: linearly in $\varphi$, and in $\log d$.
2. **Check the domain on what is known.** $d$ is unknown before inversion, so
   the domain check is on $\hat d_i$: it must lie in
   $[\hat m(d_{\min}, \varphi_i), \hat m(d_{\max}, \varphi_i)]$, and $\varphi_i$
   within the tabulated tilts.
3. **Invert** Eq. (1) with a 1-D root find (e.g. `scipy.optimize.brentq`). This
   needs $\hat m(\cdot, \varphi_i)$ strictly increasing in $d$; check it on the
   table allowing for Monte-Carlo noise (compare successive differences with
   their SE). Where it fails, flag.
4. **Flags** (handoff D5, proposal): $|\hat b - 1| > 0.2$; outside the domain
   (step 2) or the calibrated kernel range (§3.4); non-monotone $\hat m$; high
   failure rate; plus the real-data flags (steep, faint, crossings, stack
   edge). Flagged nodes take values from neighbours on the same branch, or
   Allen's radius if the whole stretch is bad.
5. **What the output is.** $\tilde d_i$ is corrected **in the mean**: across
   many real nodes of the same true $(d_i, \varphi_i)$ it removes the systematic
   part of the error, up to a Jensen gap that is small where $\tau$ is small
   (mathematics §3.6). A single node keeps the scatter of §3.8.

### 3.10 Step G — checks that decide whether the table is trusted

The table is only as good as the simulator; this section lists the tests that
can falsify it. Each is cheap compared with building the table. "Differs"
below means a difference larger than $\sqrt{\mathrm{SE}_1^2 + \mathrm{SE}_2^2}$
**and** larger than a stated tolerance (e.g. 0.02 in $b$); no significant
difference is not proof of equivalence.

| Check | How | What failure means |
|---|---|---|
| Consistency of the blurs | single-depth phantoms (all absorbance at $\delta = 0$), $\varphi = 0$, $d = 0.5$–1 µm: $\hat b$ should be ≈ 1 when $\sigma_{\rm fit}^2 \approx \sigma_{\rm r}(0)^2 + p_{\rm x}^2/4$ (heuristic budget, notes §2) | otherwise $\sigma_{\rm fit}$ and $\sigma_{\rm r}(0)$ are inconsistent, or the camera chain does more than budgeted |
| $\sigma_{\rm r}(0)$ and $\lambda$ (the unidentified input) | rebuild a reduced table at $\sigma_{\rm r}(0)$ = 0.066 and 0.095 µm (the 450–650 nm range, notes §2) | the spread of the **corrected** $\tilde d_i$ is a systematic error of the method |
| $\sigma_{\rm fit}$ | re-fit the real nodes **and** rebuild the table at each end of the handoff Eq. 12 bracket | the spread of the corrected $\tilde d_i$ should be ≈ 0 if the simulator is right (mathematics §3.7); a spread signals a simulator problem |
| Kernel shape | reduced table with the Gaussian kernel and with empirical kernels (§3.4) | if $\hat b$ differs, the Gaussian is not good enough |
| Truncation | for thick or steep phantoms, render with $\sigma_{\rm r}$ frozen beyond the calibrated $\delta_{\max}$ and with $\sigma_{\rm r}(\delta) = \sigma_{\rm r}(\delta_{\max})\,|\delta|/\delta_{\max}$; also compare $k^*$ on real and synthetic stacks | where $\hat b$ or $k^*$ differs, flag that region instead of trusting $b$ |
| Above/below | Eq. (4) separately for $\delta > 0$ and $\delta < 0$ (with a stated origin convention) | then $K_\delta \ne K_{-\delta}$ and must be tabulated per sign (causes: optics §3.7, mathematics §3.5) |
| Depth dependence | Eq. (4) on shallow vs deep nodes in the slice | the kernel depends on depth; add depth as an axis or restrict |
| Cross-section aspect | reduced table with round and with squashed phantoms at bracketing ratios | if $\hat b$ differs, the unknown shape is a systematic error of the method |
| $\mu$ | phantoms at the 10th and 90th percentile of the real $\hat\mu_i$ | if $\hat b$ changes, darkness becomes a table axis |
| Heading | $\hat b$ binned by $\theta$ | if it depends on $\theta$, the uniform draw must be replaced by the real heading distribution |
| Real vs synthetic profiles | mean real vs mean synthetic profiles at matched $(\hat d, \varphi)$ | a systematic residual is something the simulator lacks |
| End-to-end | recover known $d$ within ~10 % for $d \in \{0.5, 1, 2, 3\}$ µm, $\varphi \le 20°$ (handoff "Next actions" 5), on stacks from **independent seeds and an alternative generator or kernel** (empirical or Debye) | the acceptance test; with the same simulator it tests only the Monte Carlo and the inversion. $d \ge 1.4$ µm uses extrapolated kernels |

### 3.11 When the table must be rebuilt

Because $b$ is conditional on $\mathcal{C}$ (Eq. 2), any change to an element of
§3.3 invalidates it: a new $\sigma_{\rm fit}$; per-node vs shared $\mu$; a
different $\bar B$ region or mask; a recalibrated kernel or $\sigma_{\rm r}(0)$;
a different line-fit window or selection rule; a different specimen if its
optics, mounting, noise or JPEG tables differ. Store $\mathcal{C}$ as a
dictionary next to the table and have the correction code refuse a table whose
$\mathcal{C}$ does not match the fit's.

---

## 4. Summary of results

- Eq. (1) (§3.1): the corrected diameter $\tilde d_i$ solves $\hat m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$, the computed version of the analytic inverse $d^\star_i$ (mathematics Eq. (20)); the handoff's shortcut is off by about $-\beta(b - 1)$.
- Eq. (2) (§3.2): $b(d, \varphi \mid \mathcal{C}) = \mathbb{E}_\xi[\hat D \mid d, \varphi, \mathcal{C}, \mathcal{S}]/d$; $b$ contains the defocus bias and every other systematic error the simulator reproduces.
- Eqs. (3), (7) (§3.2, §3.8): $\hat b(d, \varphi \mid \mathcal{C})$ is the mean of $\hat d_n/d$ over the $N_{\mathcal{S}}$ retained replicates; SE $= \hat\tau/\sqrt{N_{\mathcal{S}}}$; the per-node scatter after correction is about $\hat\tau/[\hat b(1 + \beta)]$, a lower bound.
- Eq. (4) (§3.4): calibration of the defocus growth from plane scans of thin real nodes; in-focus part not identified, origin needs a convention.
- §3.4 [corrected 2026-10-04]: the rendering kernel is matched through the core-sensitive statistic — per-plane profiles of thin real nodes after the full rendering chain — not through second moments.
- Eqs. (5)–(6) (§3.6): each output plane is built from object slabs with exact chords, each blurred by its own defocus and weighted by the absorbed-light partition; recorded planes are never blurred and added.
- §3.7: the phantoms go through the real pipeline unchanged; the table is indexed by true $(d, \varphi)$, used with $\varphi_i$ only if the tilt error is negligible at the $b$ tolerance.
- §3.9: invert by root-finding; check the domain on $\hat d_i$; flag non-monotone, out-of-domain, large-correction and high-failure nodes.
- §3.10: the unidentified in-focus core $\sigma_{\rm r}(0)$, the cross-section shape and the kernel family are the main systematic uncertainties to propagate.

## 5. Open points, caveats, and assumptions

- **Open choices (the user's):** $\mu$ per node or shared (D-019 (a)); region of
  the $\bar B$ median (D-018 (a)) and the focus-score background (D-018 (c));
  $\sigma_{\rm fit}$ inside the handoff Eq. 12 bracket; kernel family; grid and
  $N$; mean vs median; selection rule $\mathcal{S}$. D5 (the bias-correction
  design itself) and D7 are still "proposed, confirm" in the handoff.
- **Assumption — slab rendering (Eq. 6).** Each slab's absorbed light is
  blurred independently; vertical rays; exact only for a single slab and to
  first order for faint tubes (mathematics §3.2).
- **Assumption — incoherent imaging.** Like handoff Eq. 11, the procedure treats
  the image as a blurred transmittance. Even at condenser NA = objective NA
  this holds only to first order in the absorbance, and Allen's
  aperture-diaphragm setting is not recorded (optics §3.6).
- **Assumption — round cross-section** (§3.5): unverified; mounting shrinkage
  may flatten dendrites; propagated by the aspect check.
- **Unidentified input — $\sigma_{\rm r}(0)$ and $\lambda$** (§3.4): fixed from an
  ideal model; propagated by the $\sigma_{\rm r}(0)$ check.
- **Assumption — shift invariance.** $K_\delta$ depends on $\delta$ only. Tested by
  the depth check.
- **Assumption — nuisance distributions** match the real nodes, or $b$ does not
  depend on them (§3.5).
- **Stage units vs tissue depth.** All $z$ are stage units. The defocus growth
  is calibrated in stage units, so it already includes the focal shift
  (optics §3.7); but a phantom that is round in stage units is round in the
  tissue only if index-matched. Not checked.
- **Mounting medium unknown for this cell.** Gouwens 2019 (mouse pipeline,
  full text) mounts in glycerol-based Mowiol or Aqua-Poly/Mount. Berg 2021
  says the human sections were imaged "as described previously". The specimen
  and treatment records of 529878215 in the Allen API contain no mounting or
  imaging fields (data inspected, 2026-10-04).
- **Kernel range.** Calibrated to ±0.84 µm (trust beyond ±2 planes after
  checking residuals); the focus search always uses extrapolated kernels.
- **Nothing coded.** Every numerical example comes from the ideal Debye model
  or from illustrative settings, not from Allen data.

**Revision record (2026-10-04).** After an independent review of v1:
the kernel-matching rule was changed from the windowed-$V$ growth to the
core-sensitive per-plane profiles (§3.4); the $\sigma_{\rm fit}$ sensitivity row
was replaced, and a $\sigma_{\rm r}(0)$/$\lambda$ row added (§3.10); the claim that
$b$ carries information "only" through defocus was qualified (§3.1); the round
cross-section was made an explicit assumption with a check (§3.5); the
selection event $\mathcal{S}$ was added (§3.2, §3.8); the per-node error bar,
the shortcut condition, the empirical-kernel caveats, the kernel range for the
focus search, the tilt-indexing criterion, the choice of phantom $\mu$, the
noise matching, the nuisance-distribution hypothesis, the exact slab chords,
the filter boundary mode, and the use of $B$ (not $\bar B$) in the renderer were
corrected; $\mathcal{C}$ was completed; notation was made consistent with the
mathematics document ($\tau$, $\beta$, $d^\star_i$, $\tilde d_i$).

**Revision record, v1.1 (2026-10-04, at the user's request).** Added the
walkthrough of one output plane tied to the interactive figure 4, with its
reference readouts (§3.6). No earlier statement changed.

## 6. References and sources

**Project knowledge / project documents (read this session):**
the user's handoff `handoff_diameter_remeasurement.md` (2026-09-30, the user's
upload, not in project knowledge): Eqs. 1–13, D1–D7, "Numbers checked", "Next
actions", Mohan et al. shrinkage figures; `claude/TEEG_diameter_optics_notes.md`
(v1, 2026-10-03); decisions D-018 and D-019 in
`TEEG_decisions_and_ideas_log.md`.

**PubMed, full text read:**
- Berg J. et al. (2021) Human neocortical expansion involves glutamatergic
  neuron diversification. *Nature*. PMC8494638.
  [DOI](https://doi.org/10.1038/s41586-021-03813-8). AxioImager Z2, Axiocam
  506, 0.63× Optivar, oil-immersion condenser NA 1.4, Plan-Apochromat 63×/1.4
  oil at 0.28 µm steps; mounting "as described previously".
- Gouwens N. W. et al. (2019) Classification of electrophysiological and
  morphological neuron types in the mouse visual cortex. *Nat Neurosci*.
  PMC8078853. [DOI](https://doi.org/10.1038/s41593-019-0417-0). Mouse
  pipeline: gelatin-coated slides, glycerol-based Mowiol or Aqua-Poly/Mount,
  0.114 µm pixels, 0.28 µm steps, 8-bit TIFF export, 4.54 µm camera pixel,
  "Tl VIS-LED" lamp.

**Data repository, data inspected:** Allen Brain Map API, `Specimen/529878215`
and `Treatment/680087734` JSON records (2026-10-04): no imaging or mounting
fields. Allen data terms of use apply; cite the Allen Cell Types Database and
Berg et al. 2021.

**Searches that returned nothing usable for this document:** PubMed
"partial coherence brightfield condenser numerical aperture image formation"
(0), "Allen Cell Types biocytin brightfield 63x reconstruction mounting medium
human neurons" (0), "human neocortex neuron morphology biocytin DAB brightfield
63x oil objective mounted reconstruction Allen Institute" (0); earlier
sessions' searches on phantom-based neurite-diameter correction (0, notes
"Sources"). bioRxiv, bioengineering, last 30 days: nothing relevant.

**Verified by running** (assistant's sandbox, scripts not in the project):
`doc_checks.py` C1–C3, C5; `defocus_forms.py`; `psf_check.py` (2026-10-02/04).
Independent review (2026-10-04) re-ran C1–C4 and checked every number against
the script outputs.

**From memory, not checked:** `scipy.ndimage.gaussian_filter` boundary modes
and `scipy.optimize.brentq` behaviour (standard SciPy API); the inverse Abel
transform as the route from a circularly symmetric LSF to its PSF.
