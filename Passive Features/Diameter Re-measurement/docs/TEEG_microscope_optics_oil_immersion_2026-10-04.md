# The physics of the 63× oil-immersion brightfield microscope

**Date:** 2026-10-04 (v1, revised the same day after an independent review;
v1.1 the same day adds diffraction, §3.4–3.5; v1.2 adds pointers to the
interactive figures; see §5, "Revision record"). **Project:** Towards EEG, diameter re-measurement
from the Allen 63× brightfield stacks. **Companion documents:**
`claude/TEEG_diameter_bias_table_procedure_2026-10-04.md` ("procedure §n /
Eq. (n)") and `claude/TEEG_diameter_bias_table_mathematics_2026-10-04.md`
("mathematics §n / Eq. (n)"). **Sources it builds on:** the user's handoff
`handoff_diameter_remeasurement.md` (2026-09-30, the user's upload, not in
project knowledge; "handoff Eq. n") and `claude/TEEG_diameter_optics_notes.md`
("notes §n").

**Abstract.** The diameter pipeline treats the Allen microscope as a fixed
blur, plus a defocus model for the bias table. This document explains the
physics behind those objects for a reader who has not worked with optics:
what each element in the light path does; why the oil between objective and
coverslip exists; what the numerical aperture is and why it limits both
resolution and the cone of light; why a defocused point becomes a spot that
grows with distance from focus; what the condenser does and why its aperture
decides whether the image is a simple blur of the specimen's transmittance;
and why the mounting medium's refractive index matters, including what it
does and does not change in a stack recorded in stage units. Each section ends
with what the result means for the diameter pipeline. Excluded: aberration
theory beyond its qualitative effects, vector (polarisation) effects, and the
camera's electronics. The scope is the microscope Allen reports (Zeiss
AxioImager Z2, Plan-Apochromat 63×/1.4 oil, oil condenser NA 1.4); every
Allen-specific fact is sourced and every gap marked.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $p_{\rm cam}$ | physical camera pixel size | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $M$ | total magnification, objective × Optivar | $\mathbb{R}_{>0}$ | dimensionless | 3.1 |
| $p_{\rm x}$ | pixel pitch referred to the specimen | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $\Delta z$ | spacing between recorded planes, 0.28 µm, in stage units | $\mathbb{R}_{>0}$ | µm | 3.1 |
| $n$, $n_1$, $n_2$ | refractive index of a medium (speed of light in vacuum over speed in the medium); of media 1 and 2 at a boundary | $\mathbb{R}_{\ge 1}$ | dimensionless | 3.2 |
| $n_{\rm g}$, $n_{\rm oil}$, $n_{\rm m}$ | indices of coverslip glass, immersion oil, mounting medium (with the tissue it soaks) | $\mathbb{R}_{\ge 1}$ | dimensionless | 3.2 |
| $\vartheta$, $\vartheta_1$, $\vartheta_2$ | angle of a ray to the optical axis (normal of the flat layers); in media 1 and 2 | $[0, \pi/2)$ | rad | 3.2 |
| $\vartheta_{\rm c}$ | critical angle for total internal reflection | $(0, \pi/2)$ | rad | 3.2 |
| $\theta_{\rm obj}$ | half-angle of the cone the objective accepts, measured in oil | $(0, \pi/2)$ | rad | 3.3 |
| $\mathrm{NA}$, $\mathrm{NA}_{\rm obj}$, $\mathrm{NA}_{\rm cond}$ | numerical aperture $n\sin\vartheta_{\max}$ of a lens, evaluated in any one flat layer; of the objective; of the condenser | $\mathbb{R}_{>0}$ | dimensionless | 3.3 |
| $\varphi$ | tilt of a branch out of the image plane (handoff Eq. 5) | $[0, \pi/2)$ | rad | 3.3 |
| $\lambda$ | vacuum wavelength of the light | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $(x, y)$ | lateral position in the image frame, specimen-referred | $\mathbb{R}^2$ | µm | 3.4 |
| $k_x(\vartheta)$ | lateral spatial frequency (phase advance per unit $x$ along the focal plane) of a plane wave at angle $\vartheta$ | $\mathbb{R}_{\ge 0}$ | rad µm⁻¹ | 3.4 |
| $\Lambda(\vartheta)$ | spacing of that wave's crests along the focal plane, $2\pi/k_x$ | $(0, \infty]$ | µm | 3.4 |
| $\Lambda_{\min,I}$ | finest period the intensity image can contain (Abbe), $\lambda/(2\,\mathrm{NA})$ | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $\Delta\phi_{\max}(\delta)$ | largest phase error across the pupil caused by defocus $\delta$ | $\mathbb{R}_{\ge 0}$ | rad | 3.5 |
| $v$ | position across a branch (handoff measuring axis) | $\mathbb{R}$ | µm | 3.4 |
| $h_\delta(x, y)$ | intensity point-spread function at defocus $\delta$; $h_0$ in focus | function $\mathbb{R}^2 \to \mathbb{R}_{\ge 0}$, unit integral | µm⁻² | 3.4 |
| $\mathrm{LSF}_\delta(v)$ | line-spread function, PSF summed along a line | function $\mathbb{R} \to \mathbb{R}_{\ge 0}$ | µm⁻¹ | 3.4 |
| $\sigma_{\rm core}(\delta)$ | Gaussian-core width of $\mathrm{LSF}_\delta$ (mathematics §3.4) | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $\sigma_{\rm fit}$ | fixed blur width inside the fit (handoff Eq. 11) | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $\sigma_{\rm r}(0)$ | in-focus width of the simulator's Gaussian rendering kernel (procedure §3.4) | $\mathbb{R}_{>0}$ | µm | 3.4 |
| $*$ | 2-D convolution in $(x, y)$ | operator | — | 3.4 |
| $z$, $z_k$ | depth along the optical axis in **stage units** (the focus drive's coordinate); depth of plane $k$ | $\mathbb{R}$ | µm | 3.5 |
| $\delta$ | signed defocus in **stage units**: depth of the object point minus depth of the focal plane | $\mathbb{R}$ | µm | 3.5 |
| $R_{\rm disc}(\delta)$ | radius of the geometric defocus spot | $\mathbb{R}_{\ge 0}$ | µm | 3.5 |
| $\gamma_\omega$ | coefficient of the geometric RMS width, $\gamma_\omega|\delta|$, for an angular weighting $\omega$ (mathematics Eq. 17) | $\mathbb{R}_{>0}$ | dimensionless | 3.5 |
| $E$, $E_1$, $E_2$ | complex amplitude of the light's electric field at a point (scalar model) | $\mathbb{C}$ | arbitrary (V/m in SI) | 3.6 |
| $E^*$, $\mathrm{Re}$ | complex conjugate; real part | operators on $\mathbb{C}$ | — | 3.6 |
| $I$, $I_1$, $I_2$ | intensity, $|E|^2$ averaged over time; what the camera records | $\mathbb{R}_{\ge 0}$ | arbitrary | 3.6 |
| $\Delta\phi$ | phase difference between two fields | $\mathbb{R}$ (mod $2\pi$) | rad | 3.6 |
| $\langle\cdot\rangle$ | average over the random phases of mutually incoherent sources | operator | — | 3.6 |
| $S$ | coherence parameter $\mathrm{NA}_{\rm cond}/\mathrm{NA}_{\rm obj}$ | $[0, 1]$ in transmitted brightfield | dimensionless | 3.6 |
| $T(x, y)$ | transmittance of the specimen column | $[0, 1]$ | dimensionless | 3.6 |
| $B$ | background brightness | $\mathbb{R}_{>0}$ | grey levels | 3.6 |
| $\theta_{\rm m}$ | angle inside the mounting medium of the steepest ray the objective accepts | $(\theta_{\rm obj}, \pi/2)$ when $1.4 < n_{\rm m} < n_{\rm oil}$ | rad | 3.7 |
| $D$ | depth of an object point below the coverslip, in the tissue (true µm) | $\mathbb{R}_{\ge 0}$ | µm | 3.7 |
| $s$ | nominal (stage-referred) focus position below the coverslip | $\mathbb{R}$ | µm | 3.7 |
| $K_\delta$ | rendering kernel at defocus $\delta$ (procedure §3.4) | function $\mathbb{R}^2 \to \mathbb{R}_{\ge 0}$ | µm⁻² | 3.7 |
| $\Delta\sigma^2(\delta)$ | calibrated defocus growth of the width (procedure Eq. 4) | function $\mathbb{R} \to \mathbb{R}$ | µm² | 3.7 |

### Conventions

- **Equation numbers.** "Eq. (n)" without prefix is this document's own; the
  other documents are cited with a prefix ("handoff Eq. n", "procedure
  Eq. (n)", "mathematics Eq. (n)").
- **Scalar optics.** Light is a scalar wave; polarisation effects, which matter
  at NA 1.4, are ignored.
- **Angles** are measured from the optical axis, which is perpendicular to all
  the flat layers. "A steep ray" means a ray at a large angle to the axis.
  Radians in formulas, degrees in text.
- **"Above" and "below"**: the light travels upward, from lamp and condenser
  under the stage, through the specimen, to the objective (upright
  microscope).
- **$\theta_{\rm obj}$ vs $\theta_i$.** The handoff's $\theta_i$ is the heading
  of a branch in the image; $\theta_{\rm obj}$ here is an optical cone angle.
  Unrelated.
- **Units of depth.** $z$, $\delta$ and $\Delta z$ are in stage units, as in the
  companion documents; true depth in the tissue is $D$ (§3.7).
- **"Debye" numbers** come from a scalar Debye-integral model of an ideal,
  aberration-free objective imaging a **point source**, with $\sqrt{\cos}$
  (aplanatic) amplitude apodisation, NA 1.4, $n = 1.515$, $\lambda = 0.55$ µm,
  computed in the assistant's sandbox. That this apodisation describes the
  shadow of a condenser-lit absorber is not established.

---

## 2. Glossary

Ordered by first appearance, because later terms use earlier ones. *Flagged*
terms have an everyday meaning that differs from the technical one.

- **Brightfield, transmitted light** (§3.1). The specimen is lit from below and
  imaged by the light that passes through it. Stained structures absorb and
  appear dark on a bright background.
- **Objective** (§3.1). The lens closest to the specimen. It collects the light
  leaving the specimen and forms the image; its numerical aperture sets the
  resolution.
- **Condenser** (§3.1). The lens below the specimen that concentrates the
  lamp's light onto it and sets the range of angles the light arrives from.
- **Mounting medium** (§3.1). The material the stained slice is embedded in
  between slide and coverslip. Its refractive index matters optically.
- **Optivar** (§3.1). A Zeiss intermediate magnification changer between
  objective and camera; here 0.63×.
- **Refractive index** (§3.2). How much a medium slows light; it sets how rays
  bend at a boundary.
- **Snell's law** (§3.2). At a flat boundary, $n\sin\vartheta$ is the same on
  both sides.
- **Total internal reflection (TIR)** (§3.2). A ray trying to pass into a
  lower-index medium at more than the critical angle to the axis is reflected
  back entirely.
- **Immersion oil** (§3.3). An oil of the same index as glass, placed between
  the objective's front lens and the coverslip, so that steep rays are not
  lost to TIR at a glass–air surface.
- **Numerical aperture (NA)** (§3.3). $n\sin\vartheta_{\max}$: the angular
  reach of a lens, in a form that does not change across flat layers.
  *Flagged:* "aperture" here measures an angle, not a hole's size.
- **Point-spread function (PSF)**, **Airy pattern** (§3.4). The image of a
  point; for a perfect circular lens in focus, a bright spot with faint rings
  (the Airy pattern).
- **Line-spread function (LSF)** (§3.4). The image of a thin line; the PSF
  summed along the line.
- **Diffraction** (§3.4). The spreading of a wave that has been cut off by a
  finite aperture. It is why a lens makes a spot, not a point. *Not* the same
  as refraction (bending at a change of index) or scattering (redirection by
  irregularities); it happens in vacuum too.
- **Huygens principle** (§3.4). Every point of a wavefront acts as a source of
  secondary waves; the field further on is their sum, with phases.
- **Lateral spatial frequency** (§3.4). How fast a wave's phase oscillates
  along a line in the focal plane. A tilted plane wave cuts that plane
  obliquely, so its crests fall at regular spacing $\Lambda$ across it; a wave
  along the axis has none.
- **Abbe limit** (§3.4). The finest period an image can contain, about
  $\lambda/(2\,\mathrm{NA})$ when condenser NA = objective NA. A resolution
  limit for periodic objects.
- **Resolution** (§3.4). *Flagged:* in optics, the smallest separation at which
  two points can be told apart, not the pixel count. Distinct from the width of
  the spot the lens forms.
- **Rayleigh criterion** (§3.4). Two points count as resolved when the peak of
  one sits on the first dark ring of the other, a separation of
  $0.61\,\lambda/\mathrm{NA}$.
- **Rayleigh quarter-wave tolerance** (§3.5). A wavefront whose phase error
  stays below a quarter of a wavelength ($\pi/2$) still gives an essentially
  perfect focal spot.
- **Defocus** (§3.5). A point not in the focal plane; it images as a spread-out
  spot whose size grows with distance from focus.
- **Electric field / intensity** (§3.6). Light is an oscillating electric field
  with an amplitude and a phase; detectors record its intensity, the squared
  magnitude averaged over time.
- **Coherence** (§3.6). Two light contributions are coherent if their phase
  difference is fixed: their fields add and they interfere. If it varies
  randomly they are incoherent: their intensities add.
- **Köhler illumination** (§3.6). The standard set-up of transmitted-light
  microscopes, in which each point of the condenser's aperture lights the whole
  field from one direction. Its aperture diaphragm sets $\mathrm{NA}_{\rm cond}$.
- **Partial coherence** (§3.6). The general case between fully coherent and
  fully incoherent illumination; described by Hopkins' theory.
- **Weak object** (§3.6). A specimen that absorbs (and shifts the phase of) the
  light only slightly; many imaging formulas are exact only to first order in
  that weakness.
- **Spherical aberration** (§3.7). A blur produced when rays at different
  angles come to focus at different depths, for example when focusing into a
  medium of a different index from the one the objective was designed for.
- **Focal shift** (§3.7). When the specimen's index differs from the oil's,
  moving the stage by a given distance moves the focus inside the specimen by
  a different distance.

---

## 3. Main body

### 3.1 The light path

This section establishes what lies between lamp and camera, in order, what
each element does, and where each fact comes from. The rest of the document
explains why each matters.

| # | Element (bottom to top) | What it does | Allen value and source |
|---|---|---|---|
| 1 | Lamp | light source | "Tl VIS-LED" (read here as transmitted-light visible LED **[reasoning]**): Gouwens 2019 (mouse pipeline, full text) |
| 2 | Condenser, with oil between it and the slide | concentrates light on the specimen; its aperture sets the illumination angles (§3.6) | oil-immersion condenser, NA 1.4: Berg 2021 (human), Gouwens 2019 (mouse), full text |
| 3 | Glass slide | carries the specimen | gelatin-coated slides: Gouwens 2019 (mouse) |
| 4 | Tissue slice in mounting medium | the specimen: biocytin-filled neuron, DAB-stained, coverslipped and dried | glycerol-based Mowiol or Aqua-Poly/Mount, slides dried ≈ 2 days: Gouwens 2019 (**mouse**); human 2016 preparation **not verified** (§3.7) |
| 5 | Coverslip | seals the specimen; glass of the thickness the objective is corrected for | type and thickness not reported in the sources read |
| 6 | Immersion oil | fills the gap between coverslip and front lens (§3.3) | "63×/1.4 Oil" objective implies it; oil type not reported |
| 7 | Objective | collects light, forms the image; sets resolution | Zeiss Plan-Apochromat 63×/1.4 oil: Berg 2021, Gouwens 2019 |
| 8 | Optivar | intermediate magnification | 0.63×: Berg 2021 (not mentioned in Gouwens 2019) |
| 9 | Camera | records intensity | Axiocam 506 monochrome (Berg 2021, Gouwens 2019); 4.54 µm pixels (Gouwens 2019) |

**The pixel pitch follows from this chain.** The specimen-referred pixel is
the camera pixel divided by the total magnification $M = 63 \times 0.63$:

$$p_{\rm x} = \frac{p_{\rm cam}}{M} = \frac{4.54\ \text{µm}}{63 \times 0.63} = 0.1144\ \text{µm}, \tag{1}$$

which matches the handoff's `res0` = 0.1144 µm and Gouwens 2019's "0.114 x
0.114 micron" **[run: arithmetic; that Gouwens' 0.114 µm comes from the same
Optivar is an inference, reasoning]**. Both papers report $\Delta z = 0.28$ µm
for the 1.4 NA objective; Berg 2021 also reports 0.44 µm for an alternative
63×/1.2 objective, so our stack's 0.28 µm step points to the 1.4 NA one
**[reasoning]**.

**For the pipeline:** the elements after the specimen (oil, objective, camera)
shape the blur; the condenser shapes how the specimen is lit, which decides
whether the blur model is valid at all (§3.6); the mounting medium sits
between them and can change both (§3.7).

### 3.2 Refraction, Snell's law and total internal reflection

The light path is a stack of flat layers (§3.1). This section gives the rule
that governs a ray crossing them, and the failure that the oil exists to
prevent.

**Refractive index.** Light travels more slowly in matter than in vacuum; the
refractive index is the ratio of the two speeds: air ≈ 1.000, cover glass and
immersion oil ≈ 1.515–1.518 **[textbook, from memory]**.

**Snell's law.** At a flat boundary between media 1 and 2, a ray at angle
$\vartheta_1$ to the normal continues at $\vartheta_2$ with

$$n_1 \sin\vartheta_1 = n_2 \sin\vartheta_2 \tag{2}$$

**[textbook]**. Through a stack of parallel layers Eq. (2) applies at each
boundary, so the product $n\sin\vartheta$ is the same in every layer: it is an
invariant of the ray. That is why the numerical aperture of §3.3 is well
defined across slide, specimen, coverslip and oil.

**Total internal reflection.** Going from a higher to a lower index
($n_1 > n_2$), Eq. (2) requires $\sin\vartheta_2 = (n_1/n_2)\sin\vartheta_1$,
which exceeds 1 when $\vartheta_1$ exceeds the critical angle

$$\vartheta_{\rm c} = \arcsin\frac{n_2}{n_1}. \tag{3}$$

Beyond it no transmitted ray exists; the light is reflected back. For glass
($n = 1.515$) to air, $\vartheta_{\rm c} = 41.3°$ **[run: arithmetic]**.

**For the pipeline:** without oil, every ray leaving the coverslip at more than
41° to the axis would be reflected at the glass–air surface and never reach
the objective.

### 3.3 Numerical aperture and why the oil is there

§3.2 showed that steep rays are lost at a glass–air surface. This section
shows that steep rays are what fine detail needs, defines the measure of how
many the objective gathers, and shows how the oil keeps them.

**Definition.** An objective accepts rays out to a half-angle $\theta_{\rm obj}$
from the axis. Its numerical aperture is

$$\mathrm{NA} = n\,\sin\theta_{\rm obj}, \tag{4}$$

with $n$ the index of the medium in which the angle is measured. By the
invariance of §3.2, NA is the same in every flat layer the accepted rays
cross, while the angle itself changes from layer to layer.

**Why it matters.** Fine detail in the specimen diffracts light into steep
angles; the finer the detail, the steeper. An objective that accepts only
shallow angles loses fine detail, so resolution scales as $\lambda/\mathrm{NA}$
(§3.4) **[textbook]**.

**The cap without oil.** A dry objective has air in front of its lens, so
$\mathrm{NA} = 1.000 \times \sin\theta_{\rm obj} < 1$, and rays beyond the
critical angle at the coverslip–air surface are lost anyway. **With oil** of
the glass's index, glass and front lens become optically continuous: no
boundary, no TIR, and NA can approach the oil's index. For the Allen objective,
NA = 1.4 in oil of $n_{\rm oil} = 1.515$ gives

$$\theta_{\rm obj} = \arcsin\frac{1.4}{1.515} = 67.5°. \tag{5}$$

**The general limit.** For an object point inside the specimen, away from the
coverslip, the NA that can be collected cannot exceed the lowest refractive
index between that point and the front lens. Kner et al. (2010, full text)
state it for their set-up: "Focusing into the sample away from the cover slip,
the NA cannot be greater than the sample index of refraction (by
definition)". (Right at the coverslip, evanescent coupling is an exception
not relevant here **[textbook, from memory]**.) The same reasoning caps the
condenser's useful NA at the lowest index on its side of the specimen. This is
why the mounting medium matters (§3.7).

**The condenser has its own NA.** It is a lens too, with oil between it and
the slide in Allen's set-up; its NA (up to 1.4) sets the range of illumination
angles (§3.6).

**For the pipeline:** $\theta_{\rm obj} = 67.5°$ is the cone of the geometric
defocus picture (§3.5) and of the handoff's tilt threshold: out-of-focus parts
along a branch reach the measuring line once $\tan\varphi > \cot\theta_{\rm obj}$,
i.e. $\varphi > 90° - 67.5° = 22.5°$ (handoff Eq. 13(b), a geometric estimate).

### 3.4 What the NA buys: the in-focus blur

§3.3 said resolution scales as $\lambda/\mathrm{NA}$. This section gives the
numbers for this objective and states which of them the diameter fit uses.

**The Airy pattern.** A perfect circular lens images a point in focus as a
bright central spot with faint rings. At NA 1.4 **[run, `psf_check.py`;
constants textbook]**:

| Quantity | Formula | 0.45 µm | 0.55 µm | 0.65 µm |
|---|---|---|---|---|
| first dark ring (radius) | $0.61\,\lambda/\mathrm{NA}$ | 0.196 | 0.240 | 0.283 |
| FWHM of the central spot | $0.514\,\lambda/\mathrm{NA}$ | 0.165 | 0.202 | 0.239 |

(µm). The handoff's "FWHM ≈ 0.24 µm" is the first dark ring, not the FWHM
(already corrected in D-018's status note and notes §2).

**Why a lens cannot focus to a point: diffraction** (added 2026-10-04, v1.1).
Geometric optics says that rays converging at the focus meet in a point; the
table above says the spot is about 0.2 µm wide. The gap is diffraction: the
spreading of light that comes from its being a wave cut off by a finite
aperture. It is **not** the same as refraction (bending at a change of index,
§3.2) or scattering (redirection by irregularities in the medium); it happens
even in a perfectly uniform medium or in vacuum **[textbook, from memory]**.

- *Huygens picture.* Each point of the wavefront leaving the objective acts as
  a small source. At the focus these contributions add with their phases.
  Only on the optical axis are they all in phase; a little way off axis, the
  waves from opposite edges of the aperture begin to cancel, so the bright
  region has a finite width.
- *Plane-wave picture.* The same field can be written as a sum of plane waves,
  one per direction the lens supplies. A plane wave travelling at angle
  $\vartheta$ to the axis in a medium of index $n$ cuts the focal plane
  obliquely, so along $x$ its phase advances at the lateral spatial frequency
  $$k_x(\vartheta) = \frac{2\pi n}{\lambda}\sin\vartheta, \tag{5a}$$
  i.e. its crests meet the focal plane every
  $$\Lambda(\vartheta) = \frac{2\pi}{k_x(\vartheta)} = \frac{\lambda}{n\sin\vartheta}. \tag{5b}$$
  A wave along the axis ($\vartheta = 0$) has $k_x = 0$: it is the same all
  across the plane and carries no pattern. The steepest wave the objective
  accepts has $n\sin\vartheta = \mathrm{NA}$, so the finest ripple available
  in the **field** has period $\lambda/\mathrm{NA} = 0.39$ µm at
  $\lambda = 0.55$ µm, NA 1.4 **[arithmetic]**. (Analogy: waves reaching a
  beach at an angle show crests along the shoreline spaced by
  $\lambda/\sin\vartheta$; waves arriving head-on hit the whole shore at once.)
- *Why that limits the spot.* The focal field is the Fourier sum of these
  plane waves (equivalently, the Fourier transform of the pupil). A field
  built only from lateral frequencies up to $2\pi\,\mathrm{NA}/\lambda$ cannot
  vary faster than that. The camera records the intensity $|E|^2$, and
  squaring doubles the highest frequency present, so the finest period the
  **intensity** can contain is
  $$\Lambda_{\min,I} = \frac{\lambda}{2\,\mathrm{NA}} = 0.196\ \text{µm} \quad (\lambda = 0.55\ \text{µm},\ \mathrm{NA} = 1.4), \tag{5c}$$
  which is Abbe's limit **[textbook, from memory]**. The index $n$ enters
  only through the shorter wavelength in the medium, $\lambda/n$; it is what
  lets an oil objective reach NA > 1.

**Forming is not resolving (rule R6).** Four widths are easily confused
**[textbook, from memory; Airy constants run, `psf_check.py`]**:

| Quantity | Meaning | At NA 1.4, $\lambda$ = 0.55 µm |
|---|---|---|
| finest ripple in the field | period of the steepest accepted plane wave across the focal plane, $\lambda/\mathrm{NA}$ (Eq. 5b) | 0.39 µm |
| finest ripple in the intensity | Abbe limit for a periodic object, $\lambda/(2\,\mathrm{NA})$ (Eq. 5c), with condenser NA = objective NA | 0.196 µm |
| narrowest spot formed | image of one point, Airy FWHM $0.514\,\lambda/\mathrm{NA}$ | 0.20 µm |
| smallest resolvable separation of two points | Rayleigh criterion: one peak on the other's first dark ring, $0.61\,\lambda/\mathrm{NA}$ | 0.24 µm |

*Forming* is about the system's output: how wide the image of a single point
is. *Resolving* is about telling two things apart, which needs a visible dip
between them, and so takes a separation somewhat larger than one spot width.
[corrected 2026-10-04: a chat answer said "the narrowest feature it can form
is about half of $\lambda/\mathrm{NA}$". As stated that is wrong:
$\lambda/(2\,\mathrm{NA})$ is the finest period of the intensity image, a
resolution limit, not the width of a spot; the spot's FWHM happens to be
numerically close, $0.514\,\lambda/\mathrm{NA}$.]

**For the pipeline:** neither number is the limit of the diameter
measurement. Measuring the width of one tube of known shape is an
**estimation** problem, not a resolution problem: the model fit of handoff
Eq. 11 can recover widths below 0.2 µm, provided the blur model and the bias
correction are right. There, noise and model error set the limit
**[reasoning]**.

**Across a branch, the LSF acts.** Summing the PSF along a straight branch
gives the LSF (mathematics Eq. (9)). Its Gaussian-core width
$\sigma_{\rm core}(0)$ (least-squares fit within one FWHM) is 0.080 µm (Debye)
and 0.082 µm (paraxial Airy, same method) at 550 nm, scaling linearly with
$\lambda$ **[run]**. Its tails decay as $|v|^{-2}$, so its full second moment is
infinite (mathematics §3.3).

**The wavelength is not one number.** Allen's lamp is a visible LED (§3.1);
its spectrum, filtered by the brown DAB stain and weighted by the camera's
sensitivity, sets an effective wavelength that is **not known**. The blur
widths above span ±18 % over 450–650 nm.

**Along the axis.** For the ideal Debye PSF, the on-axis intensity falls to
half at ±0.26 µm (axial FWHM 0.53 µm) and reaches its first minimum at
0.60 µm (numerically; closed form $\lambda/(n - \sqrt{n^2 - \mathrm{NA}^2}) =
0.59$ µm) **[run, `axial_check.py`; reviewer's run `ax2.py`]**. The paraxial
formulas give 0.75 µm (FWHM) and $2n\lambda/\mathrm{NA}^2 = 0.85$ µm (first
zero). [corrected 2026-10-04: the handoff's "axial ~0.85 µm (textbook)" (handoff
"Dropped or deferred", D1) is the paraxial **first-zero** distance; the
high-NA equivalent is ≈ 0.60 µm. A first version of this document compared it
with the FWHM instead.] None of these include aberration from the mounting
medium (§3.7).

**What falls off with defocus.** In the incoherent model with an unclipped
pupil, the laterally integrated darkness of a defocused shadow does not depend
on $\delta$, because every $h_\delta$ has unit integral; what falls is its peak
darkness, which is what the focus score of handoff Eq. 1 measures
**[reasoning]**. On data, a finite window loses part of a wide defocused shadow
(mathematics §3.5).

**For the pipeline:** the fit's fixed blur $\sigma_{\rm fit}$ (handoff Eq. 11) is,
as a heuristic budget, the LSF's core width plus the camera's sampling
(pixel integration and interpolation add $p_{\rm x}^2/4 = 0.0033$ µm² as if
they were variances), about 0.099 µm at 550 nm (notes §2; **[run,
`blur_chain_check.py`]**). The core is not a second moment, so this is a budget,
not a theorem. The simulator uses the optical part only, $\sigma_{\rm r}(0)$
(procedure §3.4).

**Interactive figures for this section** (added v1.2). Two standalone pages in
`../figures/` illustrate §3.4; their reproduction instructions are in
`TEEG_interactive_figures_spec_2026-10-04.md`.
`fig2_diffraction_focal_spot_vs_na.html`: an NA slider drives the cone
half-angle, the Abbe limit $\lambda/(2\mathrm{NA})$ and the Airy FWHM at
0.55 µm (reference readouts at NA 1.4: 67.5°, 0.196 µm, 0.155 µm as drawn
**[run, Playwright probe]**; the figure's FWHM comes from a 2-D scalar fan of
equal-weight plane waves, so the 3-D Airy table above is the reference and the
figure shows the trend with NA). It is drawn with the optical axis
horizontal, unlike the other figures (spec, known gaps).
`fig3_tilted_plane_wave_lateral_frequency.html`: one tilted plane wave in oil
and its lateral period $\Lambda = \lambda/(n\sin\theta)$ (Eq. 5b), 0.393 µm at
$\theta = 67.5°$ **[run, Playwright probe]**.

### 3.5 Defocus: the cone and the spot

§3.4 described a point in the focal plane. This section describes a point
above or below it, which is what the bias table is about.

**The cone.** Consider the rays that pass through one point of the specimen and
reach the camera. They fill a cone with its apex at that point. If the point
lies in the focal plane, the objective maps the apex to a point of the image.
If it lies a distance $\delta$ above or below, the focal plane cuts the cone
in a spot, and the image shows that spot:

$$R_{\rm disc}(\delta) = |\delta|\,\tan(\text{cone half-angle}). \tag{6}$$

**Which cone.** For a light-emitting point (fluorescence) the cone is the
objective's, $\theta_{\rm obj}$. In transmitted brightfield the point does not
emit; it **removes** light from the illumination rays that cross it, and only
rays that are both supplied by the condenser and accepted by the objective
contribute. So the shadow cone's half-angle is set by the smaller of the two
apertures and by how evenly the condenser aperture is filled
**[reasoning; textbook, from memory]**. With both NAs at 1.4 and the
condenser aperture fully open and evenly filled ($S \approx 1$, §3.6), the cone
is $\theta_{\rm obj} = 67.5°$, $\tan\theta_{\rm obj} = 2.41$, and a point one plane
(0.28 µm) off focus images as a spot of radius 0.68 µm. With the condenser
diaphragm partly closed the cone is narrower, the defocused shadow smaller,
and the depth of field larger.

The cone is symmetric about its apex, so the spot is the same size above and
below focus, **if** the system is index-matched and free of other spherical
aberration (§3.7).

**Interactive figure** (added v1.2):
`../figures/fig1_defocus_cone_disc_lsf.html` draws the cone, the spot at a
movable defocus $\Delta z$, and the spot summed along one direction. Its spot
is **uniformly bright**, so its RMS readout is the upper-bound coefficient 1.21
of the next paragraph, combined with a 0.08 µm in-focus core (at
$\Delta z = 0.28$ µm: $R = 0.68$ µm, RMS 0.35 µm **[run, Playwright probe]**).

**Across a branch, and the brightness of the spot.** Summing the spot along
one direction gives the LSF. Its RMS width grows as $\gamma_\omega|\delta|$,
where the coefficient depends on how the light is distributed over angle
(mathematics Eq. (17)): 1.21 for a uniformly bright spot, 0.90 for an
isotropic point, 0.79 for the Debye model's weighting. [corrected 2026-10-04:
earlier answers and the first version of this document used the
uniform-spot value 1.21, "$\sigma \approx 1.2|\delta|$". The spot is not uniformly
bright: a bundle of steep rays lands on a larger area (the area grows as
$1/\cos^3\vartheta$), so the rim is dim.] Which weighting fits a condenser-lit
absorber is not established; the calibration measures the real growth.

**Where geometry fails: near focus, diffraction dominates.** Eq. (6) says the
spot shrinks to nothing as $\delta \to 0$; diffraction (§3.4) says it can never
be smaller than the Airy pattern. Near focus the geometric spot is the smaller
of the two, so diffraction sets the size. The wave picture gives the
crossover (expanded 2026-10-04, v1.1). Defocus adds a phase error across the
pupil, largest for the steepest rays; for an ideal, index-matched objective,

$$\Delta\phi_{\max}(\delta) = \frac{2\pi n}{\lambda}\,\delta\,\big(1 - \cos\theta_{\rm obj}\big) \approx 10.7\ \mathrm{rad\,µm^{-1}} \times \delta \tag{6a}$$

($n = 1.515$, $\lambda = 0.55$ µm, $\theta_{\rm obj} = 67.5°$; **[textbook formula,
from memory; arithmetic]**), where $\delta$ is here the true defocus in the
index-matched medium. $\Delta\phi_{\max}$ reaches $\pi/2$ — a quarter of a
wavelength, the Rayleigh tolerance **[textbook, from memory]** — at
$|\delta| \approx 0.15$ µm. Below it the wavefront is close enough to perfect
that the spot is essentially the in-focus Airy pattern; beyond it the spot
starts to grow, and the growth eventually approaches the geometric law of
Eq. (6).

**What stays flat** is the width of the blur as a function of defocus within
about half a plane of focus. The Debye LSF's core width **[run,
`defocus_forms.py`]**:

| $\delta$ (µm) | 0 | 0.14 | 0.28 | 0.42 | 0.56 | 0.84 |
|---|---|---|---|---|---|---|
| $\sigma_{\rm core}(\delta)$ (µm) | 0.080 | 0.086 | 0.122 | 0.262 | 0.438 | 0.603 |

From 0 to 0.14 µm the width changes by +8 %; between 0.28 and 0.42 µm it more
than doubles; by 0.84 µm it is close to the $0.79|\delta| = 0.66$ µm of the
model's own geometric limit. That shape — flat, then steep — is why a parabola
in $\delta$ cannot fit the growth and why the calibration tabulates it plane by
plane (procedure §3.4). In plain words: very close to focus the blur is
already as small as a wave can make it, so stepping slightly out of focus
changes almost nothing; only past about a quarter-wave of error does the
out-of-focus cone start to show.

**For the pipeline:** this is the physics of the rendering kernel $K_\delta$
(procedure §3.4). Because the cone depends on the unrecorded condenser setting
and the brightness weighting is unknown, the kernel is calibrated from the
stacks themselves, not taken from Eq. (6).

### 3.6 The condenser, fields and coherence

§3.4–3.5 described how one point is imaged. This section asks how the images
of many points combine, which is decided by how the specimen is lit, and
therefore by the condenser.

**Fields and intensities.** Light is an oscillating electric field. At a point
it is described by a complex amplitude $E$, whose magnitude is the field's
strength and whose argument is its phase. The camera records
$I = |E|^2$, averaged over the exposure. If two contributions $E_1$, $E_2$ reach
the same pixel and are **coherent** (fixed phase difference $\Delta\phi$),
their fields add:

$$I = |E_1 + E_2|^2 = I_1 + I_2 + 2\,\mathrm{Re}\big(E_1 E_2^*\big) = I_1 + I_2 + 2\sqrt{I_1 I_2}\cos\Delta\phi . \tag{7}$$

The last term is **interference**: it depends on $\Delta\phi$ and can brighten
or darken the pixel, producing fringes. If the two contributions are
**incoherent** (phase difference random over the exposure), the average of
$\cos\Delta\phi$ is zero:

$$\langle I \rangle = I_1 + I_2 \tag{8}$$

**[textbook]**.

**What the condenser does.** In Köhler illumination **[textbook, from
memory]** each point of the condenser's aperture acts as an independent source
that lights the whole field with a plane wave from one direction; different
points of the aperture are mutually incoherent. The wider the aperture (the
higher $\mathrm{NA}_{\rm cond}$), the more directions, and the less coherent the
illumination at the specimen.

- **Wide condenser cone** ($\mathrm{NA}_{\rm cond} \approx \mathrm{NA}_{\rm obj}$):
  light from neighbouring specimen points is only weakly correlated, and the
  image is approximately the transmittance blurred by an intensity PSF,
  $I \approx B\,(T * h_0)$ — the form of handoff Eq. 11. Strictly, even at
  $S = 1$ the illumination is *partially* coherent (its coherence width,
  about $\lambda/\mathrm{NA}_{\rm cond}$, is comparable to the PSF), and
  $I = B\,(T * h_0)$ holds only **to first order in the specimen's absorbance**
  (weak object) **[textbook, from memory: Hopkins' theory; no full-text source
  found]**. Dark DAB dendrites are not weak, so for them it is an approximation
  of unchecked size.
- **Narrow condenser cone** (aperture diaphragm closed down): the specimen is
  lit by nearly one plane wave, which is coherent; fields add (Eq. 7), edges
  show fringes and ringing, and the image is no longer a blur of $T$, so
  handoff Eq. 11's form fails.

The usual measure is $S = \mathrm{NA}_{\rm cond}/\mathrm{NA}_{\rm obj}$: $S \to 0$ is
coherent, $S = 1$ the least coherent setting usable in transmitted brightfield
(illumination beyond the objective's aperture would make a darkfield
contribution instead) **[textbook, from memory]**.

**Allen's value.** The condenser is an oil-immersion NA 1.4 condenser (Berg
2021, Gouwens 2019, full text), so $S$ can reach 1. **But 1.4 is the
condenser's maximum.** The aperture-diaphragm setting used during acquisition
is not reported in the sources read. Closing it to gain contrast is common
practice **[from memory]**; it would lower $S$, make the illumination more
coherent, and narrow the defocus cone (§3.5).

**For the pipeline:** the incoherent form $B\,(T * h_0)$ underlies handoff
Eq. 11, the squared-width additivity (mathematics §3.3), the slab rendering
(procedure §3.6) and the bias table. If $S$ is well below 1, or the dendrites
are too dark for the weak-object approximation, all of these are
approximations of unknown quality. A data check: coherent edges ring, so a
systematic bright fringe beside dark branches in the real profiles is a
warning sign **[reasoning]**.

### 3.7 The mounting medium and index mismatch

§3.3 assumed that, between specimen and front lens, the light crosses only
media of the oil's index. This section drops that assumption, because the
specimen sits in a mounting medium whose index may be lower.

**What Allen used.** For the mouse pipeline, Gouwens 2019 (full text): "tissue
slices were mounted on gelatin coated slides and coverslipped with
glycerol-based Mowiol mounting media or Aqua-Poly/Mount Coverslipping Medium.
Slides were dried for approximately 2 days prior to imaging." Berg 2021 says
the human sections were imaged "as described previously" and does not name
the medium. The Allen API records for specimen 529878215 and its treatment
contain no mounting or imaging fields (data inspected, 2026-10-04). **The
medium for our cell, and its refractive index after drying, are not
verified.** The sources read give no index for these media; that
glycerol-based media lie below glass is from memory and is not used
quantitatively.

**What changes when $n_{\rm m} < n_{\rm oil}$, and in which units.** Take an
object point at true depth $D$ below the coverslip, and the focus drive at
nominal (stage-referred) depth $s$. A ray that leaves the point at angle
$\theta_{\rm m}$ in the medium continues in oil at the angle $\vartheta$ with
$n_{\rm oil}\sin\vartheta = n_{\rm m}\sin\theta_{\rm m}$. Traced back into the
objective's object space, it crosses the nominal focal plane at the lateral
offset

$$x(\vartheta) = D\tan\theta_{\rm m} - s\tan\vartheta \tag{9}$$

**[reasoning; reviewer's derivation]**. Three consequences follow.

1. **Inside the tissue the cone is wider.** For the steepest accepted ray,
   $n_{\rm m}\sin\theta_{\rm m} = 1.4$, so
   $\theta_{\rm m} = \arcsin(1.4/n_{\rm m})$ when $1.4 < n_{\rm m} < n_{\rm oil}$:
   for indices chosen as illustrations, $n_{\rm m}$ = 1.49, 1.45, 1.42 give
   $\theta_{\rm m}$ = 70.0°, 74.9°, 80.4° **[run, `doc_checks.py` C6]**. If
   $n_{\rm m} \le 1.4$, every propagating angle is accepted and the usable NA
   drops to $n_{\rm m}$ (§3.3).
2. **But per stage µm the spot grows as in oil.** By Eq. (9),
   $\partial x / \partial s = -\tan\vartheta$ for every ray: moving the stage
   moves the nominal focal plane through the **oil-equivalent** space, so the
   geometric growth per µm of stage travel is set by the angles in oil, not by
   the wider angles in the tissue. The wider tissue cone (item 1) and the focal
   shift (item 3) are two faces of one effect and cancel in stage units
   **[reasoning]**. If $n_{\rm m} < 1.4$, the steepest angle in oil drops to
   $\arcsin(n_{\rm m}/n_{\rm oil})$ and the growth per stage µm is **smaller**
   (e.g. $\tan/2 = 0.92$ instead of 1.21 at $n_{\rm m} = 1.33$, uniform-spot
   convention) **[run, arithmetic]**.
3. **What does not cancel: depth-dependent aberration and focal shift.**
   Different rays reach their best focus at different stage positions,
   $s = D\tan\theta_{\rm m}/\tan\vartheta$, which varies with the ray: that
   spread is spherical aberration, and it grows with the depth $D$. Kner et al.
   (2010, full text): high-NA objectives "only give well-corrected images just
   below the cover slip", and deeper, path-length errors from the index
   difference between sample and immersion medium (depth aberration) degrade
   the image. McGorty et al. (2014, full text), for an oil objective imaging
   into an **aqueous** medium: "[t]he physical distance between an emitter and
   the coverglass surface (actual focal position …) is shorter than the
   distance that the objective needs to move to refocus from the surface to the
   emitter (nominal focal position …)", and "the increasing amount of
   spherical aberration associated with deeper imaging broadens the PSF". So
   the 0.28 µm stage step is not 0.28 µm of tissue: a paraxial first estimate
   is tissue step ≈ $(n_{\rm m}/n_{\rm oil})$ × stage step **[textbook, from
   memory]**; at NA 1.4 the effective ratio depends on the ray and the depth,
   and is not computed here. Both papers are fluorescence experiments; the
   aberration arises on the path from the specimen to the objective, which
   brightfield shares, so the transfer is reasonable but not checked
   **[reasoning]**. A by-product **[textbook, from memory]**: the aberrated
   defocused PSF differs above and below focus.

**For the pipeline** (each item is a check, not a known problem):

| Effect | Where it enters | Check |
|---|---|---|
| wider cone in the tissue | nothing, in stage units: cancelled by the focal shift (item 2) | — (do not expect a wider cone in the calibrated $\Delta\sigma^2$); if $n_{\rm m} < 1.4$, expect a *slower* growth |
| depth-dependent aberration | blur depends on depth in the slice | shallow vs deep calibration nodes (procedure §3.10) |
| above/below asymmetry | $K_\delta \ne K_{-\delta}$ | calibrate the two signs separately |
| focal shift | absolute depths; the true shape of a phantom that is round in stage units; tilts compared with other datasets | internally consistent while everything stays in stage units (procedure §5) |

### 3.8 The objects of the diameter pipeline, in optical terms

The sections above each ended with a consequence; this section gathers them
into one map from optics to pipeline.

| Pipeline object | Optical meaning | Status |
|---|---|---|
| $\sigma_{\rm fit}$ (handoff Eq. 11) | heuristic budget: LSF core width plus pixel and interpolation blur | configured; ≈ 0.099 µm at 550 nm by the ideal model; open inside the handoff Eq. 12 bracket |
| $\sigma_{\rm r}(0)$ (procedure §3.4) | core width of the in-focus LSF only | configured; ≈ 0.080 µm (Debye, ideal, 550 nm); not identified by the stacks |
| $\Delta\sigma^2(\delta)$ (procedure Eq. 4) | growth of the defocused width in the real microscope, in stage units, condenser setting and mounting medium included | measured (to be), from the stacks |
| $\theta_{\rm obj} = 67.5°$ | cone half-angle in oil; the defocus cone only if the condenser is fully open | computed from NA 1.4 (reported) and $n_{\rm oil} = 1.515$ (from memory; oil type not reported) |
| incoherent form $B\,(T * h_0)$ | wide condenser aperture and weak object | assumed; diaphragm setting not verified; dark dendrites are not weak |
| $\Delta z = 0.28$ µm | stage step | configured acquisition setting, reported in Berg 2021 and Gouwens 2019; tissue-depth step not verified |

---

## 4. Summary of results

| # | Statement | Where |
|---|---|---|
| (1) | $p_{\rm x} = 4.54/(63 \times 0.63) = 0.1144$ µm, consistent with the handoff and Gouwens 2019 | §3.1 |
| (2) | Snell: $n_1\sin\vartheta_1 = n_2\sin\vartheta_2$; $n\sin\vartheta$ is invariant through flat layers | §3.2 |
| (3) | TIR beyond $\vartheta_{\rm c} = \arcsin(n_2/n_1)$; 41.3° from glass to air | §3.2 |
| (4)–(5) | $\mathrm{NA} = n\sin\theta_{\rm obj}$; oil lets NA reach 1.4; $\theta_{\rm obj} = 67.5°$ in oil | §3.3 |
| — | For a point inside the specimen, away from the coverslip, the usable NA cannot exceed the lowest index between it and the lens (Kner 2010) | §3.3 |
| — | Airy first ring 0.240 µm, FWHM 0.202 µm, LSF core 0.080 µm at 550 nm; axial first minimum ≈ 0.60 µm (paraxial 0.85), axial FWHM 0.53 µm (paraxial 0.75) | §3.4 |
| (5a)–(5c) | A plane wave at angle $\vartheta$ ripples across the focal plane with period $\lambda/(n\sin\vartheta)$; finest field ripple $\lambda/\mathrm{NA}$ = 0.39 µm, finest intensity period (Abbe) $\lambda/(2\,\mathrm{NA})$ = 0.196 µm | §3.4 |
| — | Forming is not resolving: spot FWHM 0.20 µm vs Rayleigh separation 0.24 µm; the diameter fit is an estimation problem, not limited by either | §3.4 |
| (6a) | Defocus phase error $\approx 10.7$ rad µm⁻¹ × $\delta$; reaches $\lambda/4$ at $|\delta| \approx 0.15$ µm, inside which the blur stays flat (core width +8 % at 0.14 µm) | §3.5 |
| (6) | Defocus spot radius $|\delta|\tan(\text{cone})$; in brightfield the cone is set by both condenser and objective; RMS growth $\gamma_\omega|\delta|$ with $\gamma_\omega$ = 0.79–1.21 depending on the angular weighting | §3.5 |
| (7)–(8) | Coherent: fields add, interference term; incoherent: intensities add | §3.6 |
| — | Wide condenser aperture → $I \approx B(T * h_0)$ to first order in the absorbance; Allen's diaphragm setting unknown | §3.6 |
| (9) | In a lower-index medium the tissue cone is wider but, per stage µm, the spot grows as in oil; depth-dependent spherical aberration and a focal shift remain | §3.7 |

## 5. Open points, caveats, and assumptions

- **Mounting medium of 529878215** and its refractive index: not verified
  (§3.7); Gouwens 2019 describes the mouse pipeline.
- **Condenser aperture-diaphragm setting**: not reported (§3.6); it changes both
  the coherence and the defocus cone (§3.5).
- **Weak-object approximation**: the incoherent blur form is first order in the
  absorbance; dark dendrites exceed it (§3.6).
- **Angular weighting of the defocused shadow**: not established (§3.5).
- **Effective wavelength** (LED spectrum × DAB absorption × camera
  sensitivity): not known (§3.4).
- **Coverslip thickness and oil type**: not reported; a mismatch with the
  objective's design adds spherical aberration **[textbook, from memory]**.
- **Scalar model**: polarisation effects at NA 1.4 are ignored.
- **Transfer from fluorescence to brightfield** of the index-mismatch results
  (§3.7): reasonable on the shared detection path, not checked.
- **Focal-shift ratio** at NA 1.4: not computed; $n_{\rm m}/n_{\rm oil}$ is a
  paraxial first estimate from memory.

**Revision record (2026-10-04).** After an independent review of v1: the
index-mismatch section was rewritten, because the wider tissue cone and the
focal shift cancel in stage units (§3.7, Eq. 9); the defocus cone is now
stated as set by condenser and objective together (§3.5); the geometric RMS
coefficient 1.21 was replaced by the weighting-dependent 0.79–1.21 (§3.5); the
axial correction now compares like with like (first zero 0.85 vs 0.60 µm;
FWHM 0.75 vs 0.53 µm) (§3.4); the incoherent form is stated as first order in
the absorbance (§3.6); Kner's condition ("away from the cover slip") was
restored (§3.3); the conservation statement was given its hypotheses (§3.4);
the Airy FWHM constant is 0.514, not 0.51; the paraxial LSF core is 0.082 µm by
the same fitting method as the Debye value; the $\sigma_{\rm fit}$ budget is
labelled heuristic; "cleared" was removed from the specimen description;
the McGorty quotation is now verbatim with elisions marked; memory-based
statements and inferences are tagged; the notation table was completed.

**Revision record, v1.1 (2026-10-04, at the user's request).** Added the
explanations given in chat after v1: what diffraction is and how it differs
from refraction and scattering; the Huygens and plane-wave pictures, with the
lateral spatial frequency of a tilted wave (Eqs. 5a–5c); the distinction
between the spot a lens forms and the separation it resolves, including the
correction of a chat sentence that conflated them (§3.4); and why the blur
stays flat near focus, via the defocus phase error and the quarter-wave
tolerance (Eq. 6a, §3.5). No earlier statement changed.

**Revision record, v1.2 (2026-10-04, at the user's request).** Added pointers
to the interactive figures 1–3 in `../figures/`, with their reference readouts
and the figure-1 caveat (uniformly bright spot, upper-bound coefficient).
No earlier statement changed.

## 6. References and sources

**PubMed, full text read (2026-10-04):**
- Berg J. et al. (2021) Human neocortical expansion involves glutamatergic
  neuron diversification. *Nature*. PMC8494638.
  [DOI](https://doi.org/10.1038/s41586-021-03813-8). AxioImager Z2, Axiocam
  506, 0.63× Optivar, oil condenser NA 1.4, Plan-Apochromat 63×/1.4 oil at
  0.28 µm (or LD LCI Plan-Apochromat 63×/1.2 Imm Corr at 0.44 µm); mounting
  "as described previously".
- Gouwens N. W. et al. (2019) Classification of electrophysiological and
  morphological neuron types in the mouse visual cortex. *Nat Neurosci*.
  PMC8078853. [DOI](https://doi.org/10.1038/s41593-019-0417-0). (The PMC
  author manuscript is titled "Classification of electrophysiological and
  morphological types in mouse visual cortex".) Mounting media; 4.54 µm camera
  pixel; 0.114 µm pixels; 0.28 µm step; 8-bit TIFF; "Tl VIS-LED" lamp; DAB
  histology.
- Kner P., Sedat J. W., Agard D. A., Kam Z. (2010) High-resolution wide-field
  microscopy with adaptive optics for spherical aberration correction and
  motionless focusing. *J Microsc*. PMC2897157.
  [DOI](https://doi.org/10.1111/j.1365-2818.2009.03315.x). Depth aberration
  from index mismatch; NA cannot exceed the sample index away from the
  coverslip.
- McGorty R., Schnitzbauer J., Zhang W., Huang B. (2014) Correction of
  depth-dependent aberrations in 3D single-molecule localization and
  super-resolution microscopy. *Opt Lett*. PMC4030053.
  [DOI](https://doi.org/10.1364/OL.39.000275). Focal shift into an aqueous
  medium; spherical aberration broadening the PSF with depth.

**PubMed, abstract only (not used for any claim):** Heine J. et al. (2018)
*Rev Sci Instrum*, [DOI](https://doi.org/10.1063/1.5020249) — index mismatch
between oil and aqueous media in STED; returned by the search, not needed.

**Searches that returned nothing (2026-10-04):** PubMed "Abbe diffraction
limit Rayleigh criterion resolution optical microscopy review" (0); "partial coherence
brightfield condenser numerical aperture image formation" (0); "numerical
aperture immersion objective resolution tutorial microscopy" (0); "Gibson
Lanni point spread function model widefield microscope" (0); "Koehler
illumination condenser aperture coherence transmitted light microscope" (0);
"Allen Cell Types biocytin brightfield 63x reconstruction mounting medium
human neurons" (0). bioRxiv bioengineering, last 30 days: nothing relevant.
The Allen Cell Types morphology white-paper URL redirected to a forum page
without the document.

**Data repository, data inspected:** Allen Brain Map API `Specimen/529878215`
and `Treatment/680087734` (2026-10-04): no mounting or imaging fields. Allen
data terms of use apply; cite the Allen Cell Types Database and Berg et al.
2021.

**Verified by running** (assistant's sandbox, not in the project):
`psf_check.py`, `defocus_forms.py`, `doc_checks.py` C5–C6, `axial_check.py`,
`blur_chain_check.py`; quadrature of the geometric coefficients. Independent
reviewer's runs (2026-10-04): `ax2.py` (axial first minimum; stage-unit growth
for $n_{\rm m} \le 1.4$).

**Textbook, from memory (not checked against a source):** diffraction and the
Huygens principle; plane-wave (Fourier) decomposition of the focal field and
the lateral wavenumber $k_x = (2\pi n/\lambda)\sin\vartheta$; the Abbe limit
and the Rayleigh resolution criterion; the defocus phase error of Eq. (6a) and
the Rayleigh quarter-wave tolerance; refractive-index
values of glass and oil; Snell's law; total internal reflection; Airy-pattern
constants 0.61 and 0.514; $\lambda/\mathrm{NA}$ scaling of resolution; Köhler
illumination; Hopkins' partial-coherence theory and the weak-object limit;
paraxial axial formulas and focal-shift ratio; evanescent coupling at the
coverslip; coverslip-thickness aberration; that glycerol-based media lie below
glass in index.
