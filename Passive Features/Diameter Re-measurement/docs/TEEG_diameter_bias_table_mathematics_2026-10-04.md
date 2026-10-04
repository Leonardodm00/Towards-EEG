# The mathematics behind the diameter bias factor $b(d, \varphi \mid \mathcal{C})$

**Date:** 2026-10-04 (v1, revised the same day after an independent review;
v1.1 the same day adds the Beer–Lambert law, the slab, and absorbance versus
absorbed fraction; see §5, "Revision record"). **Project:** Towards EEG, diameter re-measurement
from the Allen 63× brightfield stacks. **Companion documents:**
`claude/TEEG_diameter_bias_table_procedure_2026-10-04.md` (cited as
"procedure §n / procedure Eq. (n)") and
`claude/TEEG_microscope_optics_oil_immersion_2026-10-04.md` ("optics §n").
**Sources it builds on:** the user's handoff `handoff_diameter_remeasurement.md`
(2026-09-30, the user's upload, not in project knowledge; cited as "handoff
Eq. n") and `claude/TEEG_diameter_optics_notes.md` ("notes §n").

**Abstract.** The procedure document says how the bias table is built; this
one says why each step has the form it has. The question: a measurement fits
a flat, once-blurred tube to the profile across a real three-dimensional
dendrite; what does it return, how can the microscope's out-of-focus
behaviour be put into a simulator, and how is the resulting bias estimated
and removed? Covered: (§3.1) the in-focus forward model of handoff Eq. 11 and
the line-spread function; (§3.2) the image of a thick absorber built from
slabs, with the absorbed-light partition and its limits; (§3.3) the squared
width $V$ and the theorem that squared widths add under blur, whatever the
shapes, with the finite-window caveat that applies to every real LSF;
(§3.4) why the defocused line-spread function changes shape, what geometric
optics predicts for its growth and why that prediction depends on how the
light is weighted over angle, and why "$\sigma(\delta)$" has several
non-equivalent meanings; (§3.5) the calibration regression and what it can
and cannot identify; (§3.6) the bias factor as an expectation, its
Monte-Carlo estimate, its inversion and the error of the handoff's shortcut;
(§3.7) what the table can and cannot correct. Excluded: the real-data steps
upstream of the fit (handoff Eqs. 1–9) and a full partially coherent 3-D
imaging theory, which this project does not use.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $(x, y)$ | lateral position in the image frame, specimen-referred | $\mathbb{R}^2$ | µm | 3.1 |
| $u$, $v$ | coordinates along and across the branch's projection (handoff Eq. 5 axes $\hat e_u$, $\hat y$) | $\mathbb{R}$ | µm | 3.1 |
| $z$ | depth along the optical axis, stage units | $\mathbb{R}$ | µm | 3.1 |
| $z_k$ | depth of image plane $k$ | $\mathbb{R}$ | µm | 3.1 |
| $\Delta z$ | spacing of recorded planes, 0.28 µm | $\mathbb{R}_{>0}$ | µm | 1 (Conventions) |
| $\delta$ | signed defocus: depth of the object point **minus** depth of the plane being imaged, $\delta = z - z_k$ | $\mathbb{R}$ | µm | 3.1 |
| $\delta_k$, $\delta_{i,k}$ | defocus of a single-depth tube from plane $k$; of calibration node $i$'s axis from plane $k$, $z_{{\rm ax}, i} - z_k$ | $\mathbb{R}$ | µm | 3.4, 3.5 |
| $d$, $r$ | true diameter and radius ($r = d/2$) of a tube | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $\varphi$ | tilt of the tube axis out of the image plane | $[0, \pi/2)$ | rad | 3.1 |
| $\mu$ | absorption coefficient of the stained cytoplasm | $\mathbb{R}_{\ge 0}$ | µm⁻¹ | 3.1 |
| $\ell(v)$ | stained path length of the vertical ray at offset $v$ (handoff Eq. 10) | $\mathbb{R}_{\ge 0}$ | µm | 3.1 |
| $\ell'$ | path coordinate along a ray (Beer–Lambert law only) | $[0, \ell]$ | µm | 3.1 |
| $I_0$, $I(\ell)$ | intensity entering an absorber; after a path $\ell$ | $\mathbb{R}_{>0}$ | arbitrary | 3.1 |
| $a$ | absorbance of a path, $\int \mu\,d\ell'$ (natural log) | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.1 |
| $A_{10}$, $\varepsilon_{\rm mol}$, $c_{\rm mol}$ | decadic absorbance; molar absorptivity; concentration (chemistry convention only) | $\mathbb{R}_{\ge 0}$; $\mathbb{R}_{>0}$; $\mathbb{R}_{\ge 0}$ | dimensionless; M⁻¹ µm⁻¹; M | 3.1 |
| $\delta\zeta$ | slab thickness | $\mathbb{R}_{>0}$ | µm | 3.2 |
| $s_d(v)$ | dome profile $\sqrt{1 - (2v/d)^2}$ for $|v| \le d/2$, else 0 | $[0, 1]$ | dimensionless | 3.1 |
| $\alpha$ | centre-line absorbance $\mu d / \cos\varphi$ | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.1 |
| $v_0$ | lateral position of the tube axis on the measuring line | $\mathbb{R}$ | µm | 3.1 |
| $T(x, y)$, $T(v)$ | transmittance: fraction of the incident light passing the object column | $[0, 1]$ | dimensionless | 3.1 |
| $B$ | brightness of the unobstructed background (model level, configured in a simulator) | $\mathbb{R}_{>0}$ | grey levels | 3.1 |
| $\bar B$ | median estimate of $B$ from a focal plane (D-018.1; computed level) | $\mathbb{R}_{>0}$ | grey levels | 3.1 |
| $h_\delta(x, y)$ | intensity point-spread function (PSF) for a point at defocus $\delta$, unit integral | function $\mathbb{R}^2 \to \mathbb{R}_{\ge 0}$ | µm⁻² | 3.1 |
| $\mathrm{LSF}_\delta(v)$ | line-spread function, $\int h_\delta(u, v)\,du$, unit integral | function $\mathbb{R} \to \mathbb{R}_{\ge 0}$ | µm⁻¹ | 3.1 |
| $g_\sigma$ | 1-D Gaussian of standard deviation $\sigma$, unit integral | function $\mathbb{R} \to \mathbb{R}_{>0}$ | µm⁻¹ | 3.1 |
| $G_\sigma$ | circular 2-D Gaussian, standard deviation $\sigma$ per axis, unit integral | function $\mathbb{R}^2 \to \mathbb{R}_{>0}$ | µm⁻² | 3.4 |
| $\sigma_{\rm fit}$ | fixed blur width inside the fit (handoff Eq. 11) | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $K$, $K_\delta$ | a unit-integral kernel; the rendering kernel for a slab at defocus $\delta$ (a model of $h_\delta$ used in the simulator) | function $\mathbb{R}^2 \to \mathbb{R}_{\ge 0}$ (or its 1-D LSF) | µm⁻² | 3.1, 3.2 |
| $I_k(x, y)$, $I_k(v)$ | image (intensity) of plane $k$; profile along the measuring line | $\mathbb{R}_{\ge 0}$ | grey levels | 3.1 |
| $w$, $w_k$ | a dip, $w = B - I$; the dip in plane $k$ | function $\mathbb{R} \to \mathbb{R}$ | grey levels | 3.1, 3.4 |
| $j$, $J$ | slab index and number of slabs | $j \in \{1, \dots, J\}$ | — | 3.2 |
| $\zeta_j$ | depth of slab $j$; slabs numbered in the direction the light travels | $\mathbb{R}$ | µm | 3.2 |
| $a_j(x, y)$ | absorbance of slab $j$ along the vertical ray through $(x, y)$ | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.2 |
| $T_{<j}(x, y)$ | fraction of light reaching slab $j$, $\exp(-\sum_{j' < j} a_{j'})$ | $(0, 1]$ | dimensionless | 3.2 |
| $\Delta A_j(x, y)$ | fraction of the light incident on the column that is absorbed in slab $j$ (*not* a difference of the area $A$) | $[0, 1]$ | dimensionless | 3.2 |
| $f$, $g$ | non-negative integrable profiles | $L^1(\mathbb{R})$, $\ge 0$ | any | 3.3 |
| $M_p[f]$ | $p$-th raw moment $\int v^p f(v)\,dv$, $p \in \{0, 1, 2\}$ | $\mathbb{R} \cup \{+\infty\}$ for $p = 2$ | µm$^{p}$ × units of $f$ | 3.3 |
| $A[f]$ | area $M_0[f]$ | $\mathbb{R}_{>0}$ | units of $f$ × µm | 3.3 |
| $\bar v[f]$ | centre $M_1[f]/M_0[f]$ | $\mathbb{R}$ | µm | 3.3 |
| $V[f]$ | squared width $M_2[f]/M_0[f] - \bar v[f]^2$ (may be $+\infty$) | $[0, +\infty]$ | µm² | 3.3 |
| $V_W[f]$ | the same computed over the window $|v - \bar v| \le W$ only | $\mathbb{R}_{\ge 0}$ | µm² | 3.3 |
| $W$ | half-width of the window used for a windowed moment | $\mathbb{R}_{>0}$ | µm | 3.3 |
| $X$, $Y$ | independent random variables with densities $f/A[f]$, $g/A[g]$ (only inside the proof of Eq. 15) | random variables | µm | 3.3 |
| $V_{\rm tube}$ | squared width of the tube's own unblurred absorbed fraction $1 - T$ | $\mathbb{R}_{\ge 0}$ | µm² | 3.3 |
| $q$ | a hypothetical extra kernel with $\mathrm{LSF}_\delta = \mathrm{LSF}_0 * q$ | function $\mathbb{R} \to \mathbb{R}_{\ge 0}$ | µm⁻¹ | 3.4 |
| $n_{\rm oil}$ | refractive index of the immersion oil (1.515, from memory) | $\mathbb{R}_{\ge 1}$ | dimensionless | 3.4 |
| $\theta_{\rm obj}$ | half-angle of the cone accepted by the objective, in oil | $(0, \pi/2)$ | rad | 3.4 |
| $\vartheta$ | angle of one ray to the optical axis, in oil | $[0, \theta_{\rm obj}]$ | rad | 3.4 |
| $\mathrm{NA}$ | numerical aperture, $n_{\rm oil}\sin\theta_{\rm obj}$ | $\mathbb{R}_{>0}$ | dimensionless | 3.4 |
| $R_{\rm disc}(\delta)$ | radius of the geometric defocus disc, $|\delta|\tan\theta_{\rm obj}$ | $\mathbb{R}_{\ge 0}$ | µm | 3.4 |
| $\rho$ | radial landing position of a ray in the defocused plane | $[0, R_{\rm disc}]$ | µm | 3.4 |
| $\omega(\vartheta)$ | angular weighting: light power per unit solid angle carried at angle $\vartheta$ | function, $\ge 0$ | arbitrary | 3.4 |
| $V_{\rm geo}(\delta)$, $\sigma_{\rm geo}(\delta)$ | geometric-optics squared width and RMS width of the defocused LSF | $\mathbb{R}_{\ge 0}$ | µm², µm | 3.4 |
| $\gamma_\omega$ | coefficient in $\sigma_{\rm geo} = \gamma_\omega |\delta|$ for weighting $\omega$ | $\mathbb{R}_{>0}$ | dimensionless | 3.4 |
| $\lambda$ | wavelength | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $\sigma_{\rm core}(\delta)$ | width of a Gaussian least-squares fit to the core (within one FWHM) of $\mathrm{LSF}_\delta$ | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $\sigma_{W}(\delta)$ | $\sqrt{V_W[\mathrm{LSF}_\delta]}$, the windowed RMS width | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $\sigma_{\rm r}(\delta)$ | width of the Gaussian rendering kernel used in the simulator | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $\kappa$, $\lambda'$, $\delta_0$ | fit constants of three trial growth forms (§3.4 only) | $\mathbb{R}$ | µm⁰, µm⁻², µm | 3.4 |
| $i$ | index of a calibration node (§3.5) or of a real node (§3.6) | $\mathbb{N}$ | — | 3.5 |
| $V_{i,k}$ | windowed squared width of calibration node $i$'s dip in plane $k$ (computed level) | $\mathbb{R}_{\ge 0}$ | µm² | 3.5 |
| $c_i$ | per-node constant, $V_{W}$ of the in-focus dip of node $i$ | $\mathbb{R}$ | µm² | 3.5 |
| $z_{{\rm ax}, i}$ | axis depth of calibration node $i$ | $\mathbb{R}$ | µm | 3.5 |
| $\Delta\sigma^2(\delta)$ | defocus growth, $\lim_{W \to \infty}\big(V_W[\mathrm{LSF}_\delta] - V_W[\mathrm{LSF}_0]\big)$ | function $\mathbb{R} \to \mathbb{R}$ | µm² | 3.5 |
| $\varepsilon_{i,k}$ | residual of the calibration regression | $\mathbb{R}$ | µm² | 3.5 |
| $t$ | a common shift of all axis depths (in the identifiability argument only) | $\mathbb{R}$ | µm | 3.5 |
| $\eta$ | height of a point of the tube above its axis | $[-r, r]$ | µm | 3.5 |
| $\xi$ | nuisance draws of one synthetic replicate (heading, sub-pixel offset, axis depth, noise) | random vector | mixed | 3.6 |
| $\mathcal{C}$ | configuration the table is conditional on (procedure §3.3) | fixed tuple | — | 3.6 |
| $\mathcal{S}$ | selection event "fit converged, no bound hit, not flagged", applied identically to phantoms and real nodes | event | — | 3.6 |
| $\hat D$ | fitted diameter of one replicate before it is drawn | random variable on $\mathbb{R}_{>0}$ | µm | 3.6 |
| $n$, $N_{\mathcal{S}}$ | replicate index; number of replicates in $\mathcal{S}$ at a grid point | $n \in \{1, \dots, N_{\mathcal{S}}\}$ | — | 3.6 |
| $\hat d$, $\hat d_n$, $\hat d_i$ | realized fitted diameters: one replicate, replicate $n$, real node $i$ | $\mathbb{R}_{>0}$ | µm | 3.6 |
| $m(d, \varphi \mid \mathcal{C})$ | mean response $\mathbb{E}_\xi[\hat D \mid d, \varphi, \mathcal{C}, \mathcal{S}]$ | $\mathbb{R}_{>0}$ | µm | 3.6 |
| $m^{-1}(\cdot \mid \varphi, \mathcal{C})$ | inverse of $d \mapsto m(d, \varphi \mid \mathcal{C})$ on its range | function | µm → µm | 3.6 |
| $\hat m(d, \varphi \mid \mathcal{C})$ | interpolated Monte-Carlo estimate of $m$, $\hat b\,d$ | $\mathbb{R}_{>0}$ | µm | 3.6 |
| $b(d, \varphi \mid \mathcal{C})$ | bias factor $m/d$ | $\mathbb{R}_{>0}$ | dimensionless | 3.6 |
| $\tau(d, \varphi \mid \mathcal{C})$ | standard deviation of $\hat D / d$ | $\mathbb{R}_{\ge 0}$ | dimensionless | 3.6 |
| $\hat b$, $\hat\tau$ | Monte-Carlo estimates of $b$, $\tau$ | $\mathbb{R}$ | dimensionless | 3.6 |
| $c_0$ | an illustrative constant offset (§3.6 only) | $\mathbb{R}$ | µm | 3.6 |
| $\varphi_i$ | tilt of real node $i$ from the line fit | $[0, \pi/2)$ | rad | 3.6 |
| $d^\star_i$ | analytic-level inverse: the $d$ solving $m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$ | $\mathbb{R}_{>0}$ | µm | 3.6 |
| $\tilde d_i$ | computed output: the $d$ solving $\hat m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$ | $\mathbb{R}_{>0}$ | µm | 3.6 |
| $d^{\rm s}_i$ | handoff shortcut $\hat d_i / b(\hat d_i, \varphi_i \mid \mathcal{C})$ | $\mathbb{R}_{>0}$ | µm | 3.6 |
| $b^\star$ | $b(d^\star_i, \varphi_i \mid \mathcal{C})$ | $\mathbb{R}_{>0}$ | dimensionless | 3.6 |
| $\beta(d, \varphi \mid \mathcal{C})$ | elasticity $\partial \ln b(d, \varphi \mid \mathcal{C}) / \partial \ln d$ | $\mathbb{R}$ | dimensionless | 3.6 |

### Conventions

- **Equation numbers.** "Eq. (n)" without a prefix is this document's own;
  "handoff Eq. n", "procedure Eq. (n)" and "optics Eq. (n)" refer to the other
  documents. The handoff's Eqs. 8–13 and this document's (8)–(13) are
  different equations.
- 1-D convolution $(f * g)(v) = \int f(v') g(v - v')\,dv'$; 2-D likewise in
  $(x, y)$. Every kernel ($h_\delta$, $\mathrm{LSF}_\delta$, $g_\sigma$,
  $G_\sigma$, $K$, $K_\delta$) has unit integral.
- $z$, $\delta$, $\zeta$ in stage units (procedure §1, Conventions).
- **Note on $\Delta z$.** In the handoff and here, $\Delta z = 0.28$ µm is the
  plane spacing. Some earlier chat answers used "$\Delta z$" for a defocus
  distance; here that is $\delta$.
- **Levels (rule R8).** $B$, $h_\delta$, $\mathrm{LSF}_\delta$, $m$, $b$,
  $\tau$, $\beta$, $d^\star_i$ are model- or distribution-level objects;
  $\bar B$, $V_{i,k}$, $\hat d$, $\hat b$, $\hat\tau$, $\hat m$, $\tilde d_i$ are
  numbers computed from images or replicates.
- **Debye numbers** in this document: scalar Debye integral with $\sqrt{\cos}$
  (aplanatic, point-source) amplitude apodisation, NA 1.4, $n = 1.515$
  (index-matched), $\lambda = 0.55$ µm, computed in the assistant's sandbox
  (`psf_check.py`, `defocus_forms.py`, `doc_checks.py`). They describe an
  ideal, aberration-free objective imaging a point; whether that apodisation
  describes a condenser-lit absorber is not established.

---

## 2. Glossary

Ordered by first appearance.

- **Beer–Lambert law** (§3.1). Light crossing an absorber of coefficient $\mu$
  over a path $\ell$ is reduced by the factor $e^{-\mu\ell}$. The exponent
  $\mu\ell$ is the *absorbance*.
- **Point-spread function (PSF)** (§3.1). The image of a single point. In
  incoherent intensity imaging every point of the object is replaced by a copy
  of it, and the copies add.
- **Line-spread function (LSF)** (§3.1). The image of an infinitely thin
  straight line: the PSF summed along the line's direction. Across a straight
  branch it is the LSF, not the PSF, that blurs the profile.
- **Incoherent imaging** (§3.1). Image formation in which intensities from
  different object points add without interference (optics §3.6).
- **Weak (faint) object** (§3.2). An object whose absorbance is small
  ($\mu\ell \ll 1$), so that $e^{-\mu\ell} \approx 1 - \mu\ell$ and its effect
  on the image is linear.
- **Slab** (§3.2). A thin layer of the object perpendicular to the optical
  axis, thin enough to be treated as lying at one depth. *Not* an image plane.
- **Absorbance vs absorbed fraction** (§3.2). *Flagged:* absorbance $a = \mu\ell$
  is an exponent, additive and unbounded; absorbed fraction is the share of
  the light actually removed, between 0 and 1. They agree only for faint
  stain.
- **Transmittance** (§3.1). The fraction of light that passes, $T = e^{-a}$.
- **Absorbed-light partition** (§3.2). A rule assigning to each slab the
  fraction of light it absorbs, given what the slabs before it let through.
- **Raw moment / squared width** (§3.3). Weighted averages of $v^p$ over a
  profile. The squared width $V$ is a centred, normalised second moment; *its
  everyday name "variance" is borrowed*: no randomness is involved.
- **Window** (§3.3). The finite range of $v$ over which a moment is computed.
  For every real LSF the result depends on it (§3.3).
- **Defocus disc** (§3.4). In geometric optics, the cone of light cuts a plane
  at distance $\delta$ from its apex in a disc of radius
  $|\delta|\tan\theta_{\rm obj}$.
- **Angular weighting / apodisation** (§3.4). How much light each ray
  direction carries. It decides how bright the disc is at each radius, so the
  disc is generally *not* uniformly bright.
- **Debye integral** (§3.4). A standard scalar model of the focal field of a
  high-NA lens; used here to compute ideal PSFs and LSFs.
- **Gaussian core fit** (§3.4). Fitting a Gaussian to the central part of a
  profile only (here, within one full width at half maximum, FWHM). It ignores
  the tails.
- **Identifiability** (§3.5). A parameter is identifiable if different values
  of it produce different distributions of the data. A non-identifiable one
  cannot be estimated however much data there is.
- **Monte-Carlo estimate** (§3.6). An average over simulated replicates used
  to estimate an expectation.
- **Calibration-curve inversion** (§3.6). Using a known mean response
  $m(d)$ to recover $d$ from a measured value by solving $m(d) = \hat d$.
- **Elasticity** (§3.6). $\partial \ln b / \partial \ln d$: the relative change
  of $b$ per relative change of $d$.
- **Jensen gap** (§3.6). For a non-linear function $f$, $\mathbb{E}[f(X)] \ne
  f(\mathbb{E}[X])$ in general; the difference is second order in the spread
  of $X$.

---

## 3. Main body

### 3.1 The in-focus forward model and the line-spread function

This section establishes the model the fit uses and the step, the
line-spread function, that turns a 2-D blur into the 1-D one across the
branch. Everything later is measured against it.

**The Beer–Lambert law** (added v1.1). Light crossing an absorbing medium
loses, per unit path length, a fixed fraction of the intensity that reaches
that point. Along a straight ray with path coordinate $\ell' \in [0, \ell]$
(µm) and absorption coefficient $\mu$ (µm⁻¹),

$$\frac{dI}{d\ell'} = -\mu\,I \quad\Longrightarrow\quad I(\ell) = I_0\,e^{-\mu\ell}, \qquad T = \frac{I(\ell)}{I_0} = e^{-a}, \quad a = \mu\ell,$$

with $I_0$ the intensity entering the medium, $T \in (0, 1]$ the
transmittance and $a \ge 0$ the absorbance (natural-log convention); if
$\mu$ varies along the ray, $a = \int_0^\ell \mu(\ell')\,d\ell'$, which is why
the absorbances of successive slabs add (§3.2). Chemists write the same law in
base 10, $A_{10} = -\log_{10} T = \varepsilon_{\rm mol}\,c_{\rm mol}\,\ell$,
so $\mu = \ln(10)\,\varepsilon_{\rm mol}\,c_{\rm mol}$ **[textbook, from
memory]**. *Hypotheses:* absorption only (no scattering), monochromatic light
(one $\mu$), linear absorption (no saturation), a straight ray. The DAB stain
is brown and the lamp a broadband LED, so "one $\mu$" holds only
approximately, one reason $\mu$ is fitted per node (D-019). Plainly: every
micrometre of stain passes the same fraction of whatever light reaches it, so
doubling the thickness squares the transmitted fraction.

**Absorption.** A vertical ray through the point at offset $v$ from the axis
of a round tube of radius $r$, tilted by $\varphi$, crosses stained cytoplasm
over the path (handoff Eq. 10)

$$\ell(v) = \frac{2\sqrt{r^2 - v^2}}{\cos\varphi} = \frac{d}{\cos\varphi}\, s_d(v), \qquad |v| \le r, \tag{8}$$

and Beer–Lambert gives the transmittance $T(v) = e^{-\mu\ell(v)} =
\exp(-\alpha\, s_d(v - v_0))$, with $\alpha = \mu d / \cos\varphi$ and the axis
at $v_0$. *Hypothesis:* the light crosses the specimen vertically. Under a
condenser of NA 1.4 it actually arrives at up to 67.5° from the axis
(optics §3.3); oblique paths are longer and laterally displaced, and for a
dark tube the average of $e^{-\mu\ell}$ over directions is not
$e^{-\mu\ell_{\rm vertical}}$ (§5).

**Blur, for each fixed plane.** If imaging is incoherent (optics §3.6) and
shift-invariant, the recorded intensity of an object lying in the focal plane
($\delta = 0$) is $I(x, y) = B\,(T * h_0)(x, y)$. For a straight branch $T$
depends on $v$ only — *hypothesis:* over the lateral support of $h_\delta$ —
so the convolution along $u$ integrates the PSF:

$$(T * h_\delta)(v) = \int T(v') \Big[\int h_\delta(u, v - v')\,du\Big] dv' = (T * \mathrm{LSF}_\delta)(v), \qquad \mathrm{LSF}_\delta(v) = \int h_\delta(u, v)\,du. \tag{9}$$

The fit replaces $\mathrm{LSF}_0$ by the Gaussian $g_{\sigma_{\rm fit}}$ and $B$
by its estimate $\bar B$, which gives handoff Eq. 11 in its D-019 form:
$I_{\rm model}(v) = \bar B\,(T * g_{\sigma_{\rm fit}})(v)$.

**An exact identity and an approximation, used later.** Because every kernel
$K$ has unit integral, exactly,

$$B - B\,(T * K) = B\,\big((1 - T) * K\big), \tag{10}$$

that is, the *dip* $w = B - I$ is the blurred *absorbed fraction* $1 - T$. And
for a faint tube ($\alpha \ll 1$), approximately, $1 - T \approx \mu\ell(v)$: the
absorbed fraction has the shape of the chord $2\sqrt{r^2 - v^2}$, a
semicircle, and the dip is that semicircle blurred.

**What this model leaves out** (the defocus part of $b$, procedure §3.1): it
puts the whole tube at one depth, $\delta = 0$. On the vertical ray at
$v = 0$ the tube occupies $\delta \in [-r/\cos\varphi, r/\cos\varphi]$ around its
axis, and a tilted tube also reaches other depths along $u$.

### 3.2 The image of a thick absorber, slab by slab

§3.1's model has one depth; this section builds the image of a tube that
spans several, and shows where an approximation enters.

**What a slab is** (added v1.1). A slab is a thin horizontal layer of the
phantom object, the part of the tube between two depths,
$\{(x, y, z) : \zeta_j - \delta\zeta/2 \le z < \zeta_j + \delta\zeta/2\} \cap
\text{tube}$, with mid-depth $\zeta_j$ and thickness $\delta\zeta$ (µm). It
exists because the blur depends on depth: a thick tube has no single defocus,
but a thin enough slab can be given one, $\delta = \zeta_j - z_k$, and one
kernel. A slab is **not** an image plane: image planes $z_k$ are the recorded
or rendered pictures, 0.28 µm apart; slabs are pieces of the object, much
thinner (≤ 0.05 µm in the procedure, settled by convergence), and every slab
contributes to every image plane, each time with a different blur. As
$\delta\zeta \to 0$ the sums over $j$ below become integrals over $z$.
Plainly: slice the fake dendrite horizontally like a loaf; blur each slice by
its distance from the focus; stack the slices back into one picture.

**First order in the absorbance.** Cut the object into slabs $j$ at depths
$\zeta_j$ with absorbances $a_j(x, y)$. If the total $\sum_j a_j$ is small, the
transmitted fraction of each column is $1 - \sum_j a_j$ to first order, and
each slab's missing light is imaged through the PSF for its own defocus.
Linearity then gives, for each fixed plane $k$,

$$I_k(x, y) = B\Big[1 - \sum_{j=1}^{J} \big(a_j * h_{\zeta_j - z_k}\big)(x, y)\Big] + B\cdot O\Big(\big(\textstyle\sum_j a_j\big)^2\Big). \tag{11}$$

This is the weak-object picture of incoherent imaging **[textbook, from
memory]**: each absorbing element removes light at its own position, and the
missing light is imaged like any other point.

**Why it fails for a dark tube.** Eq. (11) treats every slab as if it
received the full incident light. A lower slab that absorbs much leaves less
light for the slabs above it, so summing the $a_j$ over-counts the shadow; the
linear dip can even exceed 1, which is impossible.

**The absorbed-light partition (proposal).** Number the slabs in the
direction the light travels. The fraction of the incident light reaching slab
$j$ and the fraction absorbed in it are

$$T_{<j} = \exp\Big(-\sum_{j' < j} a_{j'}\Big), \qquad \Delta A_j = T_{<j}\big(1 - e^{-a_j}\big), \tag{12}$$

and the image of plane $k$ is rendered as

$$I_k = B\Big[1 - \sum_{j=1}^{J} \Delta A_j * K_{\zeta_j - z_k}\Big]. \tag{13}$$

Three properties, each proved in one line:

1. **Conservation.** $\Delta A_j = T_{<j} - T_{<j+1}$, so the sum telescopes:
   $\sum_j \Delta A_j = 1 - T_{<J+1} = 1 - \exp(-\sum_j a_j)$, the Beer–Lambert
   absorbed fraction of the whole column. Numerically equal to 12 digits
   **[run, `doc_checks.py` C1]**.
2. **One slab gives handoff Eq. 11's form.** With $J = 1$,
   $\Delta A_1 = 1 - T$, and by Eq. (10) $B[1 - (1 - T) * K] = B\,(T * K)$.
   Difference from a direct evaluation $< 10^{-15}$ **[run, C2]** (the whole
   column absorbance $\alpha s_d$ placed at one depth, $K$ a Gaussian).
3. **Faint limit gives Eq. (11).** $\Delta A_j = a_j + O\big(a_j \sum_{j' \le j} a_{j'}\big)$,
   so Eq. (13) agrees with Eq. (11) to first order when $K_\delta = h_\delta$.

**Absorbance versus absorbed fraction (rule R6; added v1.1).** The two are
different objects and coincide only for a faint stain. The absorbance $a_j$ is
an exponent: dimensionless, in $[0, \infty)$, additive along the ray. The
absorbed fraction $\Delta A_j$ of Eq. (12) is a share of the light incident on
the column, in $[0, 1]$: the light still present when the ray reaches slab
$j$, $T_{<j}$, times the fraction of it that slab $j$ removes,
$1 - e^{-a_j}$. What darkens the image is the absorbed fraction, so that is
what Eq. (13) blurs and sums. Three equal slabs, light entering from below
**[arithmetic]**:

| | slab 1 (bottom) | slab 2 | slab 3 (top) | total |
|---|---|---|---|---|
| dark: absorbance $a_j$ | 0.5 | 0.5 | 0.5 | 1.5 (as a share it would read "150 % absorbed") |
| dark: absorbed fraction $\Delta A_j$ | 0.393 | 0.239 | 0.145 | 0.777 $= 1 - e^{-1.5}$ |
| faint: absorbance $a_j$ | 0.05 | 0.05 | 0.05 | 0.150 |
| faint: absorbed fraction $\Delta A_j$ | 0.049 | 0.046 | 0.044 | 0.139 $= 1 - e^{-0.15}$ |

Identical slabs get unequal shares: the bottom one sees the full light, the
top one only what is left. For a faint stain $e^{-a} \approx 1 - a$ and
$T_{<j} \approx 1$, so $\Delta A_j \approx a_j$, which is why the linear sum
of Eq. (11) works there and fails for a dark stain. Plainly: absorbance says
how strongly a layer would block the full beam; the absorbed fraction says
how much it actually blocks, given what the layers below have already taken.

**Interactive figure.** `../figures/fig4_slab_rendering_one_output_plane.html`
draws Eqs. (12)–(13) for a flat tube with 12 slabs: slab shading by sharpness
relative to a movable plane $k$, each slab's blurred share (grey), their sum
(orange, $I_k/\bar B$), and the linear sum of Eq. (11) (blue dashed). Its
blur widths are illustrative ideal-Debye values; its specification is
`TEEG_interactive_figures_spec_2026-10-04.md`, figure 4.

**How much it matters** ($d = 0.5$ µm, $\varphi = 0$, nine slabs, illustrative
slab blur $0.08 + 1.2|\zeta_j - z_{\rm axis}|$ µm) **[run, C3]**:

| centre-line absorbance $\mu d$ | max dip, partition | max dip, linear sum | linear over partition |
|---|---|---|---|
| 0.025 | 0.0157 | 0.0159 | +1.1 % |
| 0.30 | 0.1668 | 0.1908 | +14.4 % |
| 1.50 | 0.5134 | 0.9538 | +85.8 % |

**What remains an assumption.** Eq. (13) blurs each slab's absorbed light
independently with the kernel for its own defocus. In a thick dark object the
light reaching an upper slab has already been shaped by the lower ones; that
shaping is ignored. Eq. (13) matches the incoherent model exactly for a single
slab and to first order in the faint limit, and is a heuristic between them. A
rigorous treatment would need 3-D partially coherent imaging theory
**[reasoning]**.

### 3.3 The squared width $V$ and why squared widths add

§3.2 builds images by convolution. This section gives the one quantity that
behaves simply under convolution, whatever the shapes, and the caveat that
applies to it on every real LSF.

**Definition.** For a non-negative profile $f$, let
$M_p[f] = \int v^p f(v)\,dv$. Then

$$A[f] = M_0[f], \qquad \bar v[f] = \frac{M_1[f]}{M_0[f]}, \qquad V[f] = \frac{M_2[f]}{M_0[f]} - \bar v[f]^2 = \frac{\int (v - \bar v)^2 f(v)\,dv}{\int f(v)\,dv}. \tag{14}$$

For a dip $w = B - I$, $V[w]$ is the darkness-weighted mean squared distance
from the dip's centre. Dividing by $A$ removes the overall amount of
darkness: $B$ and the lamp brightness drop out exactly; $\mu$ and the stain
concentration drop out only in the faint limit, because they change the shape
of $1 - T$ (below).

**Theorem (squared widths add).** For any two non-negative profiles $f$, $g$
with **finite** second moments,

$$V[f * g] = V[f] + V[g], \qquad \bar v[f * g] = \bar v[f] + \bar v[g], \qquad A[f * g] = A[f]\,A[g]. \tag{15}$$

*Proof.* Raw moments of a convolution follow from expanding
$v^p = ((v - v') + v')^p$ inside the double integral:
$M_0[f * g] = M_0[f]\,M_0[g]$, $M_1[f * g] = M_1[f]M_0[g] + M_0[f]M_1[g]$,
$M_2[f * g] = M_2[f]M_0[g] + 2M_1[f]M_1[g] + M_0[f]M_2[g]$. Dividing by
$M_0[f * g]$ and subtracting the squared centre leaves Eq. (15). Equivalently:
if $X$ and $Y$ are independent with densities $f/A[f]$ and $g/A[g]$, then
$(f * g)/A[f * g]$ is the density of $X + Y$, and
$\mathrm{Var}(X + Y) = \mathrm{Var}\,X + \mathrm{Var}\,Y$ **[textbook]**. $\square$

Eq. (15) needs no Gaussian. Checked on a semicircle and a box convolved with a
Gaussian, a uniform and a semicircular kernel: agreement to all printed digits
in six cases **[run, C4]**. Only the normalised moments add; the raw second
moments do not, which is why Eq. (14) divides by $A$.

**The tube's own squared width** ($r = d/2$, no blur), $V_{\rm tube} = V[1 - T]$:

- faint ($\alpha \ll 1$): $1 - T \propto \sqrt{r^2 - v^2}$, a semicircle, so
  $V_{\rm tube} = r^2/4 = d^2/16$;
- dark ($\alpha \gg 1$): $1 - T$ saturates to a box of width $d$, so
  $V_{\rm tube} = r^2/3 = d^2/12$;
- between, smoothly (notes §3) **[textbook; run]**.

**Plain widths do not add.** A faint 0.4 µm tube under a 0.10 µm Gaussian has
$V = 0.01 + 0.01 = 0.02$ µm², RMS width 0.141 µm, not 0.1 + 0.1.

**The finite-window caveat, which applies to every real LSF.** A circular
pupil has a sharp edge, which makes the PSF's far tail fall as $\rho^{-3}$ and
the LSF's as $|v|^{-2}$, **at every defocus** **[reasoning; reviewer's run
`rev_tail.py`, `rev_tail2.py`: $v^2\,\mathrm{LSF}_\delta(v)$ levels off for
$|v|$ of a few µm, at both $\delta = 0$ and $\delta = 0.84$ µm]**. Then
$M_2[\mathrm{LSF}_\delta] = +\infty$ for every $\delta$: the hypothesis of
Eq. (15) fails for the real kernels. On data, $V$ is always computed over a
finite window $|v - \bar v| \le W$, giving $V_W$; it grows roughly linearly
with $W$ (for the Debye LSF in focus, $V_W$ at $W = 3$ µm is 2.1 times its
value at $W = 1.5$ µm **[run, C5]**), and Eq. (15) holds for $V_W$ only
approximately. **Differences** of windowed widths can still converge: if the
$|v|^{-2}$ tail has the same coefficient at every $\delta$ (as the runs above
suggest), $V_W[\mathrm{LSF}_\delta] - V_W[\mathrm{LSF}_0]$ has a finite limit
as $W \to \infty$ **[reasoning]**. That limit is what §3.5 calls the growth.

### 3.4 The defocused line-spread function

§3.3 showed that the dip's squared width is the tube's plus the kernel's,
with a window. This section asks what the kernel does as the plane moves out
of focus, and why the answer depends on how "width" is measured.

**The shape changes; only (windowed) squared widths add.** For each plane $k$,
the dip of a thin, single-depth tube is
$w_k = B\,((1 - T) * \mathrm{LSF}_{\delta_k})$ by Eqs. (9) and (10), with
$\delta_k$ the tube's defocus from plane $k$. Moving to another plane
replaces $\mathrm{LSF}_{\delta_k}$ by a **different function**, not by
$\mathrm{LSF}_0$ convolved with something. In focus it is a sharp core with
weak rings; out of focus it broadens, flattens on top, and grows a rim. An
extra kernel $q \ge 0$ with $\mathrm{LSF}_\delta = \mathrm{LSF}_0 * q$ exists
when both are Gaussian, not in general. What does hold for any shapes with
finite moments, by Eq. (15), and approximately for windowed moments (§3.3), is

$$V_W[w_k] \approx V_{\rm tube} + V_W[\mathrm{LSF}_{\delta_k}] = V_{\rm tube} + V_W[\mathrm{LSF}_0] + \big(V_W[\mathrm{LSF}_{\delta_k}] - V_W[\mathrm{LSF}_0]\big). \tag{16}$$

**Geometric optics, far from focus.** Let the light from one object point
travel in rays at angles $\vartheta \in [0, \theta_{\rm obj}]$ to the axis, with
$\mathrm{NA} = n_{\rm oil}\sin\theta_{\rm obj}$ (optics §3.3). A ray at
$\vartheta$ crosses a plane at defocus $\delta$ at radius
$\rho = |\delta|\tan\vartheta$; the outermost ray lands at
$R_{\rm disc}(\delta) = |\delta|\tan\theta_{\rm obj}$. If $\omega(\vartheta)$ is
the power per unit solid angle, the mean squared landing radius is
$\langle\rho^2\rangle = \delta^2\,\langle\tan^2\vartheta\rangle_\omega$, and the
LSF, being a 1-D projection of a circularly symmetric spot, has half of it:

$$V_{\rm geo}(\delta) = \frac{\delta^2}{2}\,\frac{\int_0^{\theta_{\rm obj}} \tan^2\vartheta\;\omega(\vartheta)\sin\vartheta\,d\vartheta}{\int_0^{\theta_{\rm obj}} \omega(\vartheta)\sin\vartheta\,d\vartheta}, \qquad \sigma_{\rm geo}(\delta) = \gamma_\omega\,|\delta|. \tag{17}$$

The coefficient depends on the weighting **[reasoning; verified by
quadrature]**:

| angular weighting $\omega(\vartheta)$ | what it describes | $\gamma_\omega$ at NA 1.4, $n_{\rm oil} = 1.515$ |
|---|---|---|
| uniformly bright disc (irradiance constant over $\rho \le R_{\rm disc}$) | the simple picture used in earlier chat answers | 1.21 $= \tan\theta_{\rm obj}/2$ |
| isotropic, $\omega = $ const | a point radiating equally in all directions | 0.90 |
| $\omega \propto \cos\vartheta$ (the Debye model's $\sqrt{\cos}$ amplitude) | aplanatic objective, point source; irradiance on the plane $\propto \cos^4\vartheta$ | 0.79 |

[corrected 2026-10-04] Earlier chat answers and notes gave
"$\sigma \approx 1.2|\delta|$ far from focus". That number is the uniform-disc
case. A uniformly bright disc is not what geometric optics gives: a ray
bundle of fixed solid angle lands on an area that grows as
$1/\cos^3\vartheta$, so steep rays are spread thin and the rim is dim. For the
Debye model's own weighting the coefficient is 0.79, and the reviewer's runs
of the Debye LSF at $\delta = 1$–3 µm give $\sigma/|\delta| = 0.79$–0.86
(windows 1.05–1.2 $R_{\rm disc}$) **[reviewer's run, `rev_geo.py`]**. What
weighting describes the shadow of an absorber under a condenser of NA 1.4 is
not established; it is bounded by the condenser cone as well as the
objective's (optics §3.5). The robust content of Eq. (17) is only
$\sigma_{\rm geo} \propto |\delta|$ far from focus. Near focus it fails:
diffraction keeps the spot from shrinking below the Airy pattern.

**Several meanings of "$\sigma(\delta)$" (rule R6), and they differ.**

| $\delta$ (µm) | $\sigma_{\rm core}$ (Gaussian fit within one FWHM) | $\sigma_W$, $W = 1.5$ µm | $\sigma_W$, $W = 3$ µm | $\sigma_{\rm geo}$, uniform disc | $\sigma_{\rm geo}$, $\cos^4$ |
|---|---|---|---|---|---|
| 0.00 | 0.080 | 0.240 | 0.348 | 0 | 0 |
| 0.14 | 0.086 | 0.261 | 0.366 | 0.169 | 0.111 |
| 0.28 | 0.122 | 0.315 | 0.415 | 0.339 | 0.222 |
| 0.42 | 0.262 | 0.385 | 0.486 | 0.508 | 0.332 |
| 0.56 | 0.438 | 0.462 | 0.569 | 0.677 | 0.443 |
| 0.84 | 0.603 | 0.609 | 0.753 | 1.016 | 0.665 |

All in µm; scalar Debye LSF, index-matched, $\lambda = 0.55$ µm
**[run, `defocus_forms.py`, `doc_checks.py` C5; last two columns
Eq. (17)]**. What the table shows:

- **In focus, the tails dominate the second moment.** The core is 0.080 µm,
  but the windowed RMS width is 3.0 ($W = 1.5$ µm) to 4.4 ($W = 3$ µm) times
  larger, and grows with the window (§3.3). A "$\sigma$" matched to $V_W$ is
  not the width the fit sees.
- **Out of focus, the core catches up.** By $\delta = 0.84$ µm the core width
  (0.60) is close to the $\cos^4$ geometric value (0.67) and to $\sigma_W$ at
  $W = 1.5$ (0.61); the uniform-disc value (1.02) is far above all of them.
- **No simple analytic growth law fits the core.** Fitted on
  $|\delta| \le 0.28$ µm and extrapolated to 0.84 µm, $\kappa^2\delta^2$
  predicts 0.068 µm² against 0.357 µm² for $\sigma_{\rm core}^2 -
  \sigma_{\rm core}(0)^2$; $\kappa^2\delta^2 + \lambda'\delta^4$ predicts 0.576;
  $\kappa^2\delta^4/(\delta^2 + \delta_0^2)$ predicts 0.218 **[run,
  `defocus_forms.py`]**. Hence: tabulate per plane offset and interpolate; do
  not extrapolate an analytic form.

**Consequence for the simulator** [corrected 2026-10-04]. A Gaussian
rendering kernel $G_{\sigma_{\rm r}(\delta)}$ cannot match the real LSF in both
core and tails, so it can be matched through **one** statistic only, and that
statistic must be the one that drives $b$. The fit (handoff Eq. 11) and the
focus score (handoff Eq. 1) respond mostly to the core. A Gaussian matched to
the growth of the windowed $V$ would be too wide in the core near focus: with
$\sigma_{\rm r}(\delta)^2 = 0.080^2 + [\sigma_W(\delta)^2 - \sigma_W(0)^2]$ and
the table above, $\sigma_{\rm r} = 0.13$–0.14 µm at $\delta = 0.14$ µm and
0.22–0.24 µm at 0.28 µm, against core widths 0.086 and 0.122 µm (reviewer's
arithmetic on these numbers). The rule is therefore: match the per-plane
**profiles** (or their Gaussian-core widths) of thin real nodes, after the full
rendering chain, and confirm with empirical kernels that $\hat b$ does not
depend on the choice (procedure §3.4, §3.10). Two earlier phrasings are
withdrawn: "matched in second moment" (chat) and "match through the windowed
$V$ growth" (the first version of these documents).

### 3.5 The calibration regression and what it identifies

§3.4 showed the defocused LSF of the real microscope cannot be computed
reliably from first principles. This section shows how much of it the stacks
themselves determine.

**Model.** At calibration node $i$ (thin, faint, flat, isolated; procedure
§3.4) the measuring line is fixed and the planes $k$ are stepped through.
By Eq. (16), with $\delta_{i,k} = z_{{\rm ax}, i} - z_k$ (object minus plane, as
everywhere in this document),

$$V_{i,k} = c_i + \Delta\sigma^2(z_{{\rm ax}, i} - z_k) + \varepsilon_{i,k}, \qquad \Delta\sigma^2(0) = 0, \tag{18}$$

with $c_i = V_{{\rm tube}, i} + V_W[\mathrm{LSF}_0]$ and all widths windowed with
the same $W$. Unknowns: one $c_i$ and one $z_{{\rm ax}, i}$ per node; the values
of the shared curve $\Delta\sigma^2$ at tabulated offsets (interpolated in
between, since $z_{{\rm ax}, i}$ is not on a plane). Estimate by non-linear
least squares over all nodes jointly. The same structure holds if $V_{i,k}$
is replaced by the squared Gaussian-core width of the dip, the statistic §3.4
recommends for kernel matching; the cancellation of $V_{{\rm tube}, i}$ is then
only approximate (notes §5).

**Identifiability.**

- **The in-focus part is not identified.** Any change in
  $V_W[\mathrm{LSF}_0]$ is absorbed into every $c_i$ with no change in the
  data; the same holds for $V_{{\rm tube}, i}$. The plane scan gives the
  *growth*, never the in-focus value. The in-focus value has to come from
  elsewhere, and in the **same sense of width**: the handoff's Eq. 12 bracket
  ("optics lower bound, thinnest nodes upper bound") was written for a core
  width $\sigma$ (≈ 0.08–0.10 µm); for $V_W$ the optical "lower bound" is the
  windowed value (0.24–0.35 µm in the Debye model), not the core.
- **The growth is identified only up to a common shift.** For any $t$,
  replacing every $z_{{\rm ax}, i}$ by $z_{{\rm ax}, i} - t$, the curve
  $\Delta\sigma^2(x)$ by $\Delta\sigma^2(x + t) - \Delta\sigma^2(t)$ (still zero
  at $x = 0$), and every $c_i$ by $c_i + \Delta\sigma^2(t)$ leaves every
  $V_{i,k}$ unchanged. A convention pins it:
  symmetry $\Delta\sigma^2(\delta) = \Delta\sigma^2(-\delta)$, or the origin at
  the curve's minimum. If the two signs of $\delta$ are fitted separately (to
  test asymmetry), the asymmetry found depends on that convention.
- **The area is a diagnostic.** In the incoherent single-depth model and in
  Eq. (13), $A[w_k] = B\,A[1 - T]$ is the same in every plane, whatever the
  darkness (Eq. 15, $A[\mathrm{LSF}] = 1$; Eq. (13) by conservation). On data
  this holds only if the window is much larger than the defocused spot; with
  $W = 1.5$ µm the spot outgrows the window within ±3 planes, and the
  $|v|^{-2}$ tail loses mass $\propto 1/W$. A change of $A_W$ beyond that is a
  model violation (neighbour contamination, a non-linear camera, behaviour
  beyond Eq. (13)).

**Why "thin" matters.** A tube of finite thickness spans heights
$\eta \in [-r, r]$ about its axis, so plane $k$ sees a mixture of defocuses
$\delta_{i,k} + \eta$. For a faint tube at $\varphi = 0$, the plane-$k$ dip is a
weighted sum of slab dips that share the same lateral centre, and Eq. (15)
applied to the mixture gives $\mathbb{E}_\eta\big[V[\mathrm{LSF}_{\delta + \eta}]\big]$
in place of $V[\mathrm{LSF}_\delta]$. If that were exactly quadratic in
$\delta$, the extra term would be a constant $\propto \mathbb{E}[\eta^2]$ and
cancel into $c_i$; §3.4 shows it is not quadratic, so the cancellation is
approximate and better the thinner the node **[reasoning, not simulated]**.
For a dark tube the partition weights are asymmetric in $\eta$ (lower slabs
absorb more), so $\mathbb{E}[\eta] \ne 0$, which appears as a shift of the
fitted $z_{{\rm ax}, i}$ **[reasoning]**.

**Above and below.** Eq. (18) with one curve assumes symmetry in $\delta$.
Two effects break it: index mismatch with the mounting medium (optics §3.7),
and a refractive-index difference between tissue and mounting medium, which
in partially coherent brightfield adds a phase-contrast component that is odd
in $\delta$; that component has zero net area (so the area diagnostic cannot
see it) but adds a term linear in $\delta$ to $V$, confounding $z_{{\rm ax}, i}$
and the asymmetry test **[reasoning; textbook (transport-of-intensity
picture), from memory]**.

### 3.6 The bias factor: definition, estimate, inversion

§3.5 supplies the kernel, so phantoms can be rendered. This section defines
what is computed from them and how it is used.

**Definition (analytic level).** For true $(d, \varphi)$ and configuration
$\mathcal{C}$, a replicate's fitted diameter before it is drawn is the random
variable $\hat D$, whose randomness comes from the nuisance draws $\xi$.
Replicates whose fit fails, hits a bound or is flagged are excluded, and the
same rule $\mathcal{S}$ is applied to real nodes. Then

$$m(d, \varphi \mid \mathcal{C}) = \mathbb{E}_\xi\big[\hat D \mid d, \varphi, \mathcal{C}, \mathcal{S}\big], \quad b(d, \varphi \mid \mathcal{C}) = \frac{m(d, \varphi \mid \mathcal{C})}{d}, \quad \tau^2(d, \varphi \mid \mathcal{C}) = \mathrm{Var}_\xi\Big[\frac{\hat D}{d} \,\Big|\, d, \varphi, \mathcal{C}, \mathcal{S}\Big]. \tag{19}$$

($\mathcal{S}$ is part of the conditioning everywhere below and is written
explicitly only in Eq. (19).)

**Estimate (computed level).** From $N_{\mathcal{S}}$ independent replicates in
$\mathcal{S}$ with realized outputs $\hat d_1, \dots, \hat d_{N_{\mathcal{S}}}$,
$\hat b = N_{\mathcal{S}}^{-1}\sum_n \hat d_n / d$ is an unbiased estimate of
$b(d, \varphi \mid \mathcal{C})$: in expectation over repeated batches it equals
$b$, and its variance is $\tau^2/N_{\mathcal{S}}$ **[textbook]**, with $\tau$
estimated by the sample standard deviation $\hat\tau$ (procedure Eq. (7)).

**Multiplicative or additive does not matter.** Because $b$ is tabulated as a
function of $d$, any form of bias is representable: a constant offset $c_0$
appears as $b = 1 + c_0/d$, a quadrature blur as $b = \sqrt{1 + c_0^2/d^2}$, a
gain as a constant $b$. The form only matters for a parametric model of $b$
(notes §7).

**Inversion.** At a real node only $\hat d_i$ and $\varphi_i$ are known. The
analytic-level inverse is

$$d^\star_i \;\text{ such that }\; m(d^\star_i, \varphi_i \mid \mathcal{C}) = \hat d_i, \tag{20}$$

which exists if $\hat d_i$ lies in the range of $m(\cdot, \varphi_i \mid \mathcal{C})$
over the tabulated $d$, and is unique if $m(\cdot, \varphi_i \mid \mathcal{C})$ is
strictly increasing there. In practice $m$ is replaced by its interpolated
estimate $\hat m$, giving the computed output $\tilde d_i$ (procedure
Eq. (1)); monotonicity must then hold for $\hat m$ beyond its Monte-Carlo
noise.

**Error of the handoff's shortcut.** The handoff uses
$d^{\rm s}_i = \hat d_i / b(\hat d_i, \varphi_i \mid \mathcal{C})$, i.e. $b$ looked
up at the measured instead of the true diameter. Write
$b^\star = b(d^\star_i, \varphi_i \mid \mathcal{C})$ and
$\beta = \beta(d^\star_i, \varphi_i \mid \mathcal{C})$. Expanding $b$ about
$d^\star_i$ and using $\hat d_i = b^\star d^\star_i$ from Eq. (20):
$b(\hat d_i, \varphi_i \mid \mathcal{C}) \approx b^\star + \partial_d b\,(b^\star - 1)\,d^\star_i = b^\star\,[1 + \beta\,(b^\star - 1)]$.
Then $d^{\rm s}_i / d^\star_i = 1/[1 + \beta(b^\star - 1)]$, so

$$\frac{d^{\rm s}_i - d^\star_i}{d^\star_i} = -\frac{\beta\,(b^\star - 1)}{1 + \beta\,(b^\star - 1)} \approx -\,\beta\,(b^\star - 1), \qquad \beta(d, \varphi \mid \mathcal{C}) = \frac{\partial \ln b(d, \varphi \mid \mathcal{C})}{\partial \ln d}, \tag{21}$$

to first order in $(b^\star - 1)$ in the expansion of $b$ **[reasoning]**.
[corrected 2026-10-04: the first version displayed
$-\beta(b^\star - 1)/b^\star$, which is the error relative to $\hat d_i$, not
to $d^\star_i$.] The shortcut is harmless where $b \approx 1$ or where $b$
hardly changes with $d$. Illustration (numbers chosen, not measured):
$b^\star = 1.15$ and $\beta = 0.3$ give −4.3 % (−4.5 % at first order). The
sign flips if $\beta < 0$, and flips again if $b^\star < 1$ (as in §3.7's
example). Eq. (20) costs a 1-D root find and removes the question.

**What the inversion delivers.** $d^\star_i$ inverts the **mean** response.
Applied to a single noisy $\hat d_i$,
$\mathbb{E}\big[m^{-1}(\hat D \mid \varphi, \mathcal{C})\big] \ne d$ in general:
a Jensen gap of order
$\tfrac12\,(m^{-1})''\,\mathrm{Var}(\hat D \mid d, \varphi, \mathcal{C}, \mathcal{S})$
**[textbook]**. It is small where $\tau$ is small and $m$ is close to linear;
it is large near the small-$d$ end where the fit degenerates, one more reason
to flag there. The scatter of the corrected value is
$\tau / [b\,(1 + \beta)]$ in relative terms, since
$\partial m / \partial d = b\,(1 + \beta)$ (procedure §3.8).

### 3.7 What the table can and cannot correct

§3.6 defined the correction; this section states its reach.

**It corrects whatever the simulator reproduces.** $b$ is the systematic error
of the *estimator* on the *simulated* microscope. Any systematic error of the
estimator that the simulator also exhibits is removed by Eq. (20), including
errors not caused by defocus: a $\sigma_{\rm fit}$ different from the true
in-focus blur, the $\bar B$ median's bias from stained pixels in its region,
the JPEG and pixel chain, and the line-fit's tilt error — the last only if the
table is indexed by the estimated tilt or the tilt error is negligible at the
$b$ tolerance (procedure §3.7). Example: with $\sigma_{\rm fit} = 0.125$ µm on
a tube truly blurred by 0.10 µm, the noise-free fit of handoff Eq. 11's own
model reads 0.5 µm as 12 % too small at $\alpha = 0.3$ (handoff "Numbers
checked"); a simulator with the right blur would return $b \approx 0.88$ there
and undo it **[reasoning on handoff numbers]**. The scatter $\tau$ grows,
however, where the estimator is near-degenerate (D-019 (a): thin nodes with
per-node $\mu$).

**It cannot correct what the simulator gets wrong.** If $K_\delta$ is wrong, if
Eq. (13) misrepresents dark thick tubes, if the real illumination is more
coherent than assumed, if the real cross-section is not round, or if the real
stain is non-uniform while the phantoms are uniform, the table encodes the
wrong bias and the corrected diameter inherits the simulator's error. That is
why the procedure's checks (procedure §3.10) test the simulator against real
profiles and against alternative kernels.

**Hence the point to carry.** The quantities that have to be right are the
simulator's optics ($\sigma_{\rm r}(\delta)$ and Eq. (13)) and its phantoms; the
fit's $\sigma_{\rm fit}$ only has to be the same in the real and the synthetic
runs.

---

## 4. Summary of results

| # | Statement | Derived in |
|---|---|---|
| (8) | $\ell(v) = 2\sqrt{r^2 - v^2}/\cos\varphi$; $T(v) = \exp(-\alpha s_d(v - v_0))$, $\alpha = \mu d/\cos\varphi$; vertical rays assumed | §3.1 |
| (9) | Across a straight branch the 2-D blur acts through $\mathrm{LSF}_\delta(v) = \int h_\delta(u, v)\,du$ | §3.1 |
| (10) | The dip is the blurred absorbed fraction: $B - B(T * K) = B((1 - T) * K)$ | §3.1 |
| (11) | Faint thick object: $I_k = B[1 - \sum_j a_j * h_{\zeta_j - z_k}]$ to first order | §3.2 |
| (12)–(13) | Absorbed-light partition rendering; conserves Beer–Lambert; matches the incoherent model for one slab and to first order when faint; heuristic between | §3.2 |
| (14)–(15) | $V[f * g] = V[f] + V[g]$ for non-negative $f$, $g$ with finite second moments; areas multiply | §3.3 |
| — | Every real LSF has a $|v|^{-2}$ tail, so its full $V$ is infinite; only windowed $V_W$ and their differences are usable | §3.3 |
| — | $V_{\rm tube} = d^2/16$ (faint) to $d^2/12$ (dark) | §3.3 |
| (16) | Per plane: $V_W[w_k] \approx V_{\rm tube} + V_W[\mathrm{LSF}_{\delta_k}]$; the LSF changes shape, only squared widths add | §3.4 |
| (17) | Geometric law $\sigma_{\rm geo} = \gamma_\omega|\delta|$; $\gamma_\omega$ = 1.21 (uniform disc), 0.90 (isotropic), 0.79 ($\cos^4$, Debye weighting) | §3.4 |
| — | $\sigma_{\rm core}$ and $\sigma_W$ differ by 3–4.4× in focus; a Gaussian kernel is matched through the core-sensitive statistic (per-plane profiles) | §3.4 |
| (18) | $V_{i,k} = c_i + \Delta\sigma^2(z_{{\rm ax},i} - z_k)$; growth identified up to a common shift; in-focus value not identified | §3.5 |
| (19) | $b = \mathbb{E}_\xi[\hat D \mid d, \varphi, \mathcal{C}, \mathcal{S}]/d$; $\hat b$ unbiased with variance $\tau^2/N_{\mathcal{S}}$ | §3.6 |
| (20) | Correct by solving $m(d, \varphi_i \mid \mathcal{C}) = \hat d_i$; needs $\hat d_i$ in range and $m$ strictly increasing | §3.6 |
| (21) | Shortcut error $= -\beta(b^\star - 1)/[1 + \beta(b^\star - 1)] \approx -\beta(b^\star - 1)$ | §3.6 |
| — | The table removes every systematic error the simulator reproduces, and none that the simulator lacks | §3.7 |

## 5. Open points, caveats, and assumptions

- **Incoherent, shift-invariant imaging** (§3.1) is assumed throughout. Even at
  condenser NA = objective NA, brightfield is partially coherent, and the
  incoherent form $B(T * h)$ holds only to first order in the absorbance
  (optics §3.6); dark DAB dendrites are not weak objects. The diaphragm
  setting is unknown.
- **Vertical rays** (§3.1, §3.2): Beer–Lambert and the partition are evaluated
  along vertical rays, while the illumination arrives at up to 67.5°.
- **Pure absorber**: a refractive-index difference between tissue and mountant
  adds a phase term odd in $\delta$ (§3.5).
- **Eq. (13)** is a heuristic beyond the single-slab and faint limits (§3.2).
- **Debye numbers** are for an ideal, index-matched, scalar model at one
  wavelength, with point-source apodisation; the real objective, the
  broadband LED spectrum, the condenser and the mounting medium (optics §3.7)
  all change them. They show orders of magnitude and the disagreement between
  widths, never inputs to the table.
- **Windowed moments** (§3.3–3.5): the full second moment of every real LSF
  is infinite; Eq. (15) holds only approximately for $V_W$; the finiteness of
  the growth limit rests on the tail coefficient being independent of
  $\delta$ (reasoning plus the reviewer's runs).
- **Geometric weighting** (§3.4): which $\omega(\vartheta)$ describes a
  condenser-lit absorber is not established.
- **Thickness averaging** in the calibration (§3.5) cancels only if the
  growth is quadratic in $\delta$; it is not; not simulated.
- **Symmetry and origin** of the growth curve (§3.5) need a convention.
- **Eq. (21)** is first order in the expansion of $b$; nodes with
  $|b - 1| > 0.2$ are flagged anyway.
- **Mean vs median** of $\hat D/d$: Eq. (19) uses the mean; a median-based
  table would estimate a different object (open, procedure §5).

**Revision record (2026-10-04).** After an independent review of v1 the
following were corrected: the geometric defocus coefficient (uniform disc
1.21 replaced by the weighting-dependent $\gamma_\omega$, Eq. 17); the
divergence of the LSF's second moment holds at every $\delta$, not only in
focus, so all calibration quantities are windowed (§3.3, Eq. 18); the
kernel-matching rule now uses the core-sensitive statistic (§3.4); the sign
convention of Eq. (18) now matches $\delta = z - z_k$; a translation
degeneracy was added (§3.5); Eq. (21) displayed the error relative to the
wrong quantity; the selection event $\mathcal{S}$ was added (§3.6);
hypotheses were added for vertical rays, partial coherence, the phase term,
and the area diagnostic; the symbols $s$, $e$, $n$ (refractive index) were
renamed $\tau$, $\beta$, $n_{\rm oil}$ to remove clashes.

**Revision record, v1.1 (2026-10-04, at the user's request).** Added, from
the chat explanations of the same day: the Beer–Lambert law with its
hypotheses and the chemistry convention (§3.1); what a slab is (§3.2); the
distinction between absorbance and absorbed fraction with a worked
three-slab example (§3.2); a pointer to the interactive slab-rendering figure
and its specification. No earlier statement changed.

## 6. References and sources

**Project documents read this session:** the user's handoff
`handoff_diameter_remeasurement.md` (Eqs. 6, 10–13, "Numbers checked");
`claude/TEEG_diameter_optics_notes.md` (v1, 2026-10-03); D-018, D-019 in
`TEEG_decisions_and_ideas_log.md`.

**Verified by running** (assistant's sandbox; scripts not in the project):
`doc_checks.py` C1–C5 (2026-10-04); `defocus_forms.py` (re-run 2026-10-04);
`psf_check.py` (2026-10-02); quadrature of Eq. (17) for the three weightings
(2026-10-04). **Independent reviewer's runs** (same sandbox, 2026-10-04):
`rev_tail.py`, `rev_tail2.py` (LSF tails), `rev_geo.py` (Debye LSF at
$\delta$ = 1–3 µm), `rev_c14.py` (C1–C4 reproduced).

**Textbook, from memory (not checked against a source):** Beer–Lambert law;
weak-object linearisation of incoherent imaging (Eq. 11); moments of a
convolution and $\mathrm{Var}(X + Y) = \mathrm{Var}\,X + \mathrm{Var}\,Y$ for
independent $X$, $Y$; semicircle and box second moments; unbiasedness of a
sample mean; Jensen's inequality; the Debye integral; the $\rho^{-3}$ tail of
a sharp-edged pupil's PSF; the transport-of-intensity picture of phase
contrast in defocused brightfield.

**Literature searches:** for the physics of index mismatch and coherence, see
optics §6. Earlier sessions' PubMed searches on PSF-convolved cylinder fits
and 3-D brightfield image formation of absorbing specimens returned no full
text usable for Eqs. (11)–(13) (notes "Sources"). Zhang, Zerubia &
Olivo-Marin (2007), *Appl Opt*, [DOI](https://doi.org/10.1364/ao.46.001819) —
**abstract only**: states that no accurate Gaussian approximation exists for
the 3-D widefield PSF; consistent with §3.4, no numbers used.
