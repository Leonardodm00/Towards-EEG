# TEEG-20 --- Spine membrane area: calculation, biases, and the F comparison

**Handoff v1 --- 19 August 2026**

## Abstract

This document specifies exactly how the Towards-EEG pipeline computes dendritic
spine membrane area, quantifies the systematic biases in that computation,
and re-examines the apparent deficit of the reconstruction-derived spine
correction factor $F$ against published human values. The scientific question
is: **is the H01-derived spine membrane area a measurement, an underestimate,
or a quantity not comparable to the published numbers it is being compared
against?** Covered: the frustum-chain area model and its exact formulae; the
four identified systematic biases with signs and magnitudes; the measured
result on four L3 excitatory cells; what the published comparison values
actually measure and by what method; and a new head-shape diagnostic that
brackets the uncertainty. Deliberately excluded: spine *density* validation
(the $\psi(\nu, b, d)$ profile comparison, which is S1.7 and a separate
document); spine neck axial resistance and the assumption-A2 question (settled
separately and unaffected by anything here --- see Section 3.7); and any change
to the pipeline's arithmetic, which this document argues against making until
the open points in Section 5 are closed.

---

## 1. Notation and Symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in Section |
|---|---|---|---|---|
| $\nu$ | Neuron identifier (H01 segment id) | $\nu \in \mathcal{N}$, a finite index set | dimensionless | 3.1 |
| $\sigma$ | A single dendritic spine | $\sigma \in \mathcal{S}(\nu)$, the spines of cell $\nu$ | dimensionless | 3.1 |
| $m$ | A skeleton node index | $m \in \mathcal{V}(\nu)$, the nodes of cell $\nu$ | dimensionless | 3.1 |
| $p(m)$ | Parent of node $m$ in the rooted tree | $p: \mathcal{V}(\nu)\setminus\{\text{root}\} \to \mathcal{V}(\nu)$ | dimensionless | 3.1 |
| $\mathbf{x}_m$ | Position of node $m$ | $\mathbf{x}_m \in \mathbb{R}^3$ | um (nm on disk) | 3.1 |
| $r_m$ | Radius recorded at node $m$ | $r_m \in \mathbb{R}_{>0}$ | um (nm on disk) | 3.1 |
| $\ell(m)$ | Compartment label of node $m$ | $\ell(m) \in \{\text{soma},\text{dendrite},\text{apical},\text{axon},\text{spine},\text{head},\text{neck},\ldots\}$ | dimensionless | 3.1 |
| $L_{ab}$ | Axial length of the segment $a \to b$ | $L_{ab} \in \mathbb{R}_{\ge 0}$ | um | 3.1 |
| $s_{ab}$ | Slant height of the frustum $a \to b$ | $s_{ab} \in \mathbb{R}_{\ge 0}$ | um | 3.1 |
| $A_{ab}$ | Lateral surface area of the frustum $a \to b$ | $A_{ab} \in \mathbb{R}_{\ge 0}$ | um^2 | 3.1 |
| $\mathcal{M}(\sigma)$ | Node set of spine $\sigma$ | $\mathcal{M}(\sigma) \subseteq \mathcal{V}(\nu)$, connected under $p$ | dimensionless | 3.1 |
| $\rho(\sigma)$ | Root node of spine $\sigma$ | $\rho(\sigma) \in \mathcal{M}(\sigma)$, unique | dimensionless | 3.1 |
| $b(\sigma)$ | Base node of spine $\sigma$ (a shaft node) | $b(\sigma) = p(\rho(\sigma)) \notin \mathcal{M}(\sigma)$ | dimensionless | 3.1 |
| $A_{\mathrm{spine}}(\sigma)$ | Total membrane area attributed to spine $\sigma$ | $\in \mathbb{R}_{\ge 0}$ | um^2 | 3.1 |
| $A_{\mathrm{head}}(\sigma)$ | Head-labelled part of $A_{\mathrm{spine}}(\sigma)$ | $\in \mathbb{R}_{\ge 0}$ | um^2 | 3.1 |
| $A_{\mathrm{neck}}(\sigma)$ | Neck-labelled part of $A_{\mathrm{spine}}(\sigma)$ | $\in \mathbb{R}_{\ge 0}$ | um^2 | 3.1 |
| $A_{\mathrm{other}}(\sigma)$ | Spine-labelled but neither head nor neck | $\in \mathbb{R}_{\ge 0}$ | um^2 | 3.1 |
| $A_{\mathrm{shaft}}(\nu, e)$ | Membrane area of shaft segment $e$ of cell $\nu$ | $\in \mathbb{R}_{\ge 0}$ | um^2 | 3.1 |
| $d(\nu, m)$ | Path distance from soma of $\nu$ to node $m$, along the cable | $\in \mathbb{R}_{\ge 0}$ | um | 3.1 |
| $d_{\mathrm{cut}}$ | Proximal cutoff for the literature-comparable $F$ | $d_{\mathrm{cut}} = 60$ (human), $30$ (mouse) | um | 3.1 |
| $\phi(\nu, b, d)$ | Spine-area line density along shaft branch $b$ at path distance $d$ | $\ge 0$, piecewise constant per segment | um^2 / um | 3.1 |
| $F(\nu)$ | Spine correction factor for cell $\nu$ | $F(\nu) \ge 1$ | dimensionless | 3.1 |
| $f_{\mathrm{implied}}(\nu)$ | $F$ over the WHOLE arbour, no cutoff | $\ge 1$ | dimensionless | 3.1 |
| $F_{\mathrm{lit}}(\nu)$ | $F$ restricted to $d \ge d_{\mathrm{cut}}$; see naming warning in Section 2 | $\ge 1$ | dimensionless | 3.1 |
| $r_{\max}(\sigma)$ | Largest radius among head-labelled nodes of $\sigma$ | $\in \mathbb{R}_{>0}$ | um | 3.5 |
| $A^{\mathrm{sph}}_{\mathrm{head}}(\sigma)$ | Head as a sphere of radius $r_{\max}(\sigma)$ | $= 4\pi r_{\max}^2$ | um^2 | 3.5 |
| $\chi(\sigma)$ | Lateral-to-sphere head area ratio | $\chi = A_{\mathrm{head}} / A^{\mathrm{sph}}_{\mathrm{head}}$, $> 0$ | dimensionless | 3.5 |
| $\lambda(\sigma)$ | Head path length over head radius | $\lambda = L_{\mathrm{head}} / r_{\max}$, $\ge 0$ | dimensionless | 3.5 |
| $L_{\mathrm{head}}(\sigma)$ | Summed length of head-labelled segments of $\sigma$ | $\in \mathbb{R}_{\ge 0}$ | um | 3.5 |
| $\rho_a$ | Cytoplasmic axial resistivity | $\rho_a \in [100, 400]$ | Ohm cm | 3.7 |
| $R_{\mathrm{neck}}(\sigma)$ | Axial resistance of the neck of $\sigma$ | $> 0$ | Ohm | 3.7 |
| $\kappa(\sigma)$ | Charge-transfer factor for spine-to-shaft relocation | $\kappa \in (0, 1]$ | dimensionless | 3.7 |
| $\hat{g}_{\mathrm{syn}}$ | Peak synaptic conductance | $> 0$ | nS | 3.7 |

### 1.1 Conventions

- **Units at the boundary.** H01 skeleton coordinates and radii are stored in
  nanometres. `spine_density._prepare_nodes` divides by
  `NM_PER_UM = 1000.0` on load, so every quantity in this document is in
  micrometres and square micrometres unless stated. `input_units="nm"` is the
  correct argument for every H01 frame; `"um"` exists only for test fixtures.
- **Tree orientation.** The skeleton is a rooted tree with the soma as root and
  $p(m)$ pointing toward the soma. Every node except the root therefore has
  exactly one *incoming* segment $(p(m), m)$. Summing a per-segment quantity
  over nodes enumerates segments exactly once, with no double counting. This
  identity is used throughout and is the reason the sums in Section 3.1 are
  written over nodes rather than edges.
- **Distal attribution.** Where a segment must be assigned to one compartment,
  it is assigned to the compartment its **distal** node carries. This is a
  convention, not a measurement, and it is what makes Eq. (5) an exact
  partition.
- **Path distance, not Euclidean.** $d(\nu, m)$ is always distance along the
  cable, accumulated by breadth-first traversal in
  `spine_density._path_distances_um`. It is never the Euclidean distance from
  the soma.
- **Index ranges.** Sums over $m \in \mathcal{M}(\sigma)$ include the spine
  root $\rho(\sigma)$, whose incoming segment runs from the shaft base
  $b(\sigma)$ into the spine. The base junction is therefore inside the spine
  area, not outside it.
- **"Spine" as a label set.** `spine_density.SPINE_LABELS = ("spine", "head",
  "neck")`, matched case-insensitively. A node is spine-labelled iff its
  `annotated_type` is in that set.

---

## 2. Glossary / Jargon

Ordered by first appearance, because the concepts build on each other.

**Frustum.** A truncated right circular cone. The pipeline's atomic geometric
unit: every skeleton segment is modelled as one, with the two node radii as its
end radii. *Operative from Section 3.1.*

**Lateral area.** The curved side surface of a solid, *excluding* its end
faces. The pipeline computes lateral area only, so no structure in the model
has a cap anywhere --- including at the distal tip of a spine head, which is
therefore an open tube. This is a term whose everyday reading ("the area of the
thing") differs from the technical one; the difference is a real bias, see
Section 3.4. *Operative from Section 3.1.*

**Slant height.** The distance along the sloping side of a frustum,
$\sqrt{(r_a - r_b)^2 + L^2}$, as opposed to its axial length $L$. Using the
slant rather than $L$ matters wherever radius changes fast relative to length,
which is exactly what happens where a neck meets a head. *Operative from
Section 3.1.*

**Spine (as an object in this pipeline).** A maximal parent/child-connected
component of spine-labelled nodes. Not a morphological classification --- a
"spine" here is whatever the labeller labelled, including structures a
histologist might call a short branch. *Operative from Section 3.1.*

**Base.** The shaft node a spine hangs off; the parent of the spine's root.
The point to which the spine's whole area is attributed, and to which its
synapses are redirected when spines are pruned. *Operative from Section 3.1.*

**F factor.** The ratio (dendritic membrane area + total spine area) /
(dendritic membrane area). Used to fold spine membrane into a model that has no
explicit spines, by multiplying $c_m$ by $F$ and dividing $R_m$ by $F$. The
method originates with Rapp, Segev and Yarom. *Operative from Section 3.1.*

**`F_lit` (naming warning).** A column name in the pipeline's outputs. It is
**not** a literature value: it is the reconstruction's own $F$ recomputed
under the literature's *convention*, i.e. restricted to segments at path
distance $\ge 60$ um from the soma. Comparing `f_implied` to `F_lit` is an
internal comparison of one cell against itself and is guaranteed to favour
`F_lit` for any cell whose spine density rises with distance. The comparison
against a published number is `F_lit` versus that number. *Operative from
Section 3.1.*

**Marching Cubes.** An algorithm that extracts a closed triangulated surface
from a 3-D voxel mask. Relevant because it is how the comparison values in the
literature were produced, and a closed triangulated surface is a different
mathematical object from a chain of open frustums. *Operative from Section
3.3.*

**Head-shape bracket.** A pair of bounds on head membrane area --- the frustum
lateral sum below, a circumscribed sphere above --- whose ratio $\chi(\sigma)$
reports how much the choice of area model matters for a given spine.
*Operative from Section 3.5.*

**Quasi-steady approximation.** The regime in which a spine head and its base
can be treated as exchanging current through a pure resistance with no
capacitive transient. Used for $\kappa$, and the reason $\kappa$ is a charge
ratio rather than a voltage ratio. *Operative from Section 3.7.*

---

## 3. Main body

### 3.1 What the pipeline computes, exactly

*This section establishes the precise algorithm and its formulae, so that any
proposed change can be stated as a change to a numbered equation.*

Implementation: `spine_density.py` v1.2.0 (`_frustum_lateral_area`,
`_attribute_spine_area`, `build_phi`) and `spine_geometry.py` v1.0.0
(`build_spine_geometry`). The head/neck split exists only in the latter;
`spine_density` computes the total only.

**The primitive.** For a segment from node $a$ to node $b$, for each fixed
pair $(a, b)$ with $p(b) = a$:

$$L_{ab} \;=\; \lVert \mathbf{x}_b - \mathbf{x}_a \rVert_2 \tag{1}$$

$$s_{ab} \;=\; \sqrt{(r_a - r_b)^2 + L_{ab}^2} \tag{2}$$

$$A_{ab} \;=\; \pi\,(r_a + r_b)\, s_{ab} \tag{3}$$

Eq. (3) is the exact lateral surface area of a truncated right circular cone
with end radii $r_a, r_b$ and end-face separation $L_{ab}$. Note it uses the
slant $s_{ab}$ of Eq. (2), not $L_{ab}$. The commonly seen simplification
$\pi(r_a + r_b)L_{ab}$ differs from Eq. (3) by under 0.5 percent on a gently
tapering neck but by an arbitrary factor where $|r_a - r_b| \gg L_{ab}$, which
is precisely the neck-to-head junction. The pipeline does **not** make that
simplification.

**Spine identification.** A node $m$ is spine-labelled iff
$\ell(m) \in \{\text{spine},\text{head},\text{neck}\}$. For each fixed $\nu$,
a spine $\sigma$ is a maximal connected component of spine-labelled nodes under
the parent relation $p$. Its root $\rho(\sigma)$ is the unique member whose
parent is not spine-labelled, and its base is
$b(\sigma) = p(\rho(\sigma))$, a shaft node.

**The area sum.** For each fixed $\sigma$:

$$A_{\mathrm{spine}}(\sigma) \;=\; \sum_{m \in \mathcal{M}(\sigma)} A_{p(m),\,m} \tag{4}$$

Because every node has exactly one incoming segment (see Conventions), Eq. (4)
counts each segment of the spine exactly once. The term $m = \rho(\sigma)$
contributes the segment from the shaft base $b(\sigma)$ into the spine, so the
base junction is included in the spine's area and not in the shaft's.

**The head/neck partition** (in `spine_geometry` only). For each fixed
$\sigma$, with the distal-attribution convention:

$$A_{\mathrm{head}}(\sigma) = \!\!\sum_{\substack{m \in \mathcal{M}(\sigma) \\ \ell(m) = \text{head}}}\!\! A_{p(m),\,m}, \qquad
A_{\mathrm{neck}}(\sigma) = \!\!\sum_{\substack{m \in \mathcal{M}(\sigma) \\ \ell(m) = \text{neck}}}\!\! A_{p(m),\,m} \tag{5}$$

with $A_{\mathrm{other}}(\sigma)$ defined analogously over spine-labelled nodes
carrying neither label. By construction, for each fixed $\sigma$,
$A_{\mathrm{head}} + A_{\mathrm{neck}} + A_{\mathrm{other}} = A_{\mathrm{spine}}$;
this identity is asserted on every build in `spine_geometry._self_check`.

**Where the area goes.** `spine_density._attribute_spine_area` assigns the
whole of $A_{\mathrm{spine}}(\sigma)$ to the shaft segment whose *distal* node
is $b(\sigma)$, and $\phi$ is that area divided by the segment length. Shaft
area $A_{\mathrm{shaft}}(\nu, e)$ uses the identical primitive, Eq. (3).

**The correction factors.** For each fixed $\nu$, with $\mathcal{E}(\nu)$ the
shaft segments and $d(\nu, e)$ the path distance to the proximal end of
segment $e$:

$$F(\nu; \mathcal{E}') \;=\; 1 \;+\; \frac{\sum_{e \in \mathcal{E}'} A^{\mathrm{sp}}(\nu, e)}{\sum_{e \in \mathcal{E}'} A_{\mathrm{shaft}}(\nu, e)} \tag{6}$$

where $A^{\mathrm{sp}}(\nu, e)$ is the spine area attributed to segment $e$.
Two instances are reported:

$$f_{\mathrm{implied}}(\nu) \;=\; F\!\left(\nu; \mathcal{E}(\nu)\right) \tag{7}$$

$$F_{\mathrm{lit}}(\nu) \;=\; F\!\left(\nu; \{e \in \mathcal{E}(\nu) : d(\nu, e) \ge d_{\mathrm{cut}}\}\right), \qquad d_{\mathrm{cut}} = 60\ \mathrm{um} \tag{8}$$

For each fixed $\nu$ whose spine density is non-decreasing with $d$, we have
$f_{\mathrm{implied}}(\nu) \le F_{\mathrm{lit}}(\nu)$, because the proximal
shaft contributes denominator with little matching numerator. Only Eq. (8) is
comparable to a published $F$.

### 3.2 What was measured

*This section establishes the empirical result the rest of the document
interprets.*

Four L3 excitatory H01 cells, run 19 August 2026. Radii were real on all four
(`radius_frac_at_default_r` $= 0$, `resistance_trustworthy` = True), so nothing
below is a fallback artefact.

| $\nu$ | $f_{\mathrm{implied}}$ | $F_{\mathrm{lit}}$ | $n$ spines | $A_{\mathrm{head}}$ median (um^2) | head share |
|---|---|---|---|---|---|
| 1302789404 | 1.6239 | 1.7106 | 6626 | 0.970 | 0.469 |
| 1317492596 | 1.5472 | 1.7251 | 4164 | 1.101 | 0.510 |
| 1333261412 | 1.4619 | 1.6243 | 3914 | 1.202 | 0.496 |
| 1376890291 | 1.2597 | 1.5751 | 5172 | 1.060 | 0.495 |

Derived: mean $F_{\mathrm{lit}} = 1.659$; median $A_{\mathrm{head}} = 1.081$
um^2; mean head share of spine area $= 0.492$; mean whole-spine area
$= 2.35$ um^2.

The $d \ge 60$ um restriction behaved as predicted: the enhancement
$(F_{\mathrm{lit}} - 1)/(f_{\mathrm{implied}} - 1)$ was 1.14, 1.33, 1.35, 2.21
(median 1.34), against 1.465 estimated beforehand from the aspiny interneuron
bank.

Neck dimensions landed inside published bands (length 0.94 um median against
$1.34 \pm 0.50$; equivalent diameter 0.30 um against 0.20--0.30). Head area did
not, on the comparison used at the time --- which Section 3.3 shows was the
wrong comparison.

### 3.3 What the published values measure, and why they are not the same quantity

*This section establishes that the head-area "deficit" was measured against a
number produced by an incompatible method, and that the literature spread on
the same quantity is larger than the deficit.*

**Method of the comparison values.** [Knowledge base, full text: Eyal et al.
2016, *Unique membrane properties...*] The $F$ values and spine areas were
computed from **confocal 3-D reconstructions**, not from a skeleton. Spines
were segmented in Imaris at manually selected intensity thresholds, often
merging surfaces from several thresholds per spine; the resulting meshes were
rasterised to a high-resolution 3-D mask, dilated and eroded to join
components, and a **closed mesh re-extracted by Marching Cubes** after Gaussian
filtering, with area computed from that mesh via VTK. A confocal
z-distension correction factor of 0.84 was applied.

That is a **closed triangulated surface**. It includes all curvature and all
end caps. Eq. (3)--(4) produce a **chain of open frustums**. These are
different mathematical objects, and there is no reason for them to agree on a
blob-shaped structure.

**The published $F$ values are not a single number.** [Knowledge base, full
text: Eyal et al. 2016]

| donor | region | dendrite | $F$ |
|---|---|---|---|
| 85 y | cingulate | basal | $1.81 \pm 0.34$ |
| 85 y | cingulate | apical | $1.78 \pm 0.33$ |
| 85 y | temporal | basal | $1.89 \pm 0.41$ |
| 85 y | temporal | apical | $1.87 \pm 0.13$ |
| 40 y | cingulate | basal | $1.98 \pm 0.38$ |
| 40 y | cingulate | apical | $2.00 \pm 0.28$ |
| 40 y | temporal | basal | $2.39 \pm 0.63$ |
| 40 y | temporal | apical | $2.39 \pm 0.27$ |

Average 1.946, from which $F = 1.9$ was adopted. **Temporal basal dendrites
differ by 26 percent between the two donors** (1.89 against 2.39) using one
method in one laboratory. H01 is a 45-year-old donor, temporal cortex.

Measured mean $F_{\mathrm{lit}} = 1.659$ sits $0.56$ standard deviations below
the 85-year-old temporal basal value of $1.89 \pm 0.41$. On that comparison it
is not a deficit at all.

**Head area in the literature spans a factor of 3.5.** [Knowledge base, full
text]

| source | quantity as reported | as a head area |
|---|---|---|
| Eyal 2018 / DeFelipe lab, $n = 150$ human L3 | head area $2.88 \pm 1.37$ um^2 (mesh) | 2.88 um^2 |
| *Comprehensive analysis of human dendritic spine morphology*, J Neurophysiol | head **diameter** $511 \pm 12$ nm (2-D manual) | 0.82 um^2 if spherical |
| this bank, H01, frustum chain | --- | 1.08 um^2 |

The H01 value sits **inside** the range the literature itself spans. The
earlier statement that head area is "a factor of 2.7 low" was measured against
the upper end of a 3.5-fold spread.

Whole-spine area for context: J Neurophysiol manual mesh $7.06 \pm 0.34$ um^2
(but with a $1$--$30$ um^2 filter applied, which excludes small spines and
biases the mean upward); Eyal's prototypical modelled spine 3.86 um^2 (head 2.8
plus a $1.35 \times 0.25$ um neck cylinder); this bank 2.35 um^2.

### 3.4 The four systematic biases, with signs

*This section establishes what is genuinely wrong with Eq. (3)--(4), separately
from the comparability problem of Section 3.3.*

**(a) No end caps --- underestimates.** Eq. (3) is lateral area only. Nothing in
the pipeline adds a cap anywhere, so every spine head terminates as an open
tube. For a head of radius $r$ the missing disc is $\pi r^2$; for an
Eyal-sized head ($r = 0.479$ um) that is 0.72 um^2, i.e. **25 percent of the
sphere area**. This is a real, one-signed, uncorrected bias.

**(b) Shaft footprint not removed --- overestimates.** Where a spine attaches,
that patch of shaft membrane does not exist, but both the shaft segment and the
spine's base segment count it. Magnitude $\approx \pi r_{b}^2$ with $r_b$ the
neck base radius: about 0.05 um^2 per spine. Opposite sign to (a) and roughly
14 times smaller.

**(c) Lateral area is the wrong model for a blob --- underestimates.** For a
structure of radius $r$ traversed by a skeleton path of length $L$, lateral
area is $2\pi r L$ against a sphere's $4\pi r^2$. These agree only at
$L = 2r$. A head represented by one or two skeleton nodes has $L \ll 2r$:

| head path $L$ | lateral as fraction of sphere |
|---|---|
| $2r$ | 1.00 |
| $1r$ | 0.50 |
| $0.5r$ | 0.25 |

**(d) Chorded lengths and circular cross-sections.** Eq. (1) is a straight-line
chord between nodes, so a curved neck sampled by few nodes has its length
underestimated. Cross-sections are assumed circular and taper linear. Signs are
not established.

Biases (a) and (c) act in the same direction on head-dominated spines, and
together are of the order of the difference between the frustum sum and a mesh
area. They do **not** apply to necks, which are genuine tubes traversed
longitudinally by the skeleton --- consistent with neck dimensions matching the
literature while head area does not.

**Consequence for $F$.** Eq. (6) is built from Eq. (4), so
$f_{\mathrm{implied}}$ and $F_{\mathrm{lit}}$ inherit (a)--(d). A negative bias
of order 20--25 percent on head-dominated spines is the same order as the gap
between measured $F_{\mathrm{lit}} = 1.659$ and the adopted $F = 1.9$.

### 3.5 The head-shape bracket

*This section establishes the diagnostic added to decide, per cell, whether
$A_{\mathrm{head}}$ is a measurement or a lower bound.*

`spine_geometry` v1.0.0 now emits, for each fixed $\sigma$ with at least one
head-labelled node:

$$A^{\mathrm{sph}}_{\mathrm{head}}(\sigma) \;=\; 4\pi\, r_{\max}(\sigma)^2, \qquad r_{\max}(\sigma) = \max_{\substack{m \in \mathcal{M}(\sigma) \\ \ell(m) = \text{head}}} r_m \tag{9}$$

$$\chi(\sigma) \;=\; \frac{A_{\mathrm{head}}(\sigma)}{A^{\mathrm{sph}}_{\mathrm{head}}(\sigma)} \tag{10}$$

$$\lambda(\sigma) \;=\; \frac{L_{\mathrm{head}}(\sigma)}{r_{\max}(\sigma)} \tag{11}$$

Columns: `A_head_sphere_um2`, `head_lateral_over_sphere`,
`head_path_over_radius`, `head_r_max_um`, `head_r_mean_um`, with medians in
`cell_spine_summary`. Both are NaN for a spine with no head-labelled node.

**Interpretation.** The ratio $\chi(\sigma)$ of Eq. (10) is what is read.
Eq. (9) is an upper bracket (a circumscribed sphere
ignores that a real head is not spherical and is partly occluded by its neck);
Eq. (4) restricted to head nodes is a lower bracket (no caps, blob traversed
by a short path). The truth lies between. $\chi \to 1$ means the two models
agree and $A_{\mathrm{head}}$ is safe; $\chi \ll 1$ means it is a lower bound.

**Offset caveat.** $\chi$ carries a constant positive offset from the
neck-to-head transition segment: the radius jumps from neck to head over almost
no axial length, so by Eq. (2) that frustum has a large slant and contributes a
"shoulder" annulus. That is real membrane where neck meets head, so $\chi$ can
slightly exceed 1 for a fully traversed head. Read changes in $\chi$ across a
population, not its absolute value against 1. In the module's own fixture the
offset is $\approx 0.23$.

**Predicted value on this bank.** From the observed
$A_{\mathrm{head}}/2.88 = 0.38$ and the table in Section 3.4(c), the implied
head path is $\lambda \approx 0.75$. A synthetic head built at
$\lambda = 0.73$ reproduces $A_{\mathrm{head}} = 0.945$,
$A^{\mathrm{sph}} = 2.883$, $\chi = 0.328$ --- close to the observed 1.081
against 2.88. **This is a prediction, not a measurement: it has not yet been
run on the real bank.** Running it is the first task in Section 5.

### 3.6 Two corrections to earlier statements in this project

*Recorded so they are not propagated.*

1. **The value 2.39 was attributed to Benavides-Piccione et al. as an "Htemp"
   measurement.** It is not: it is Eyal et al. 2016's own $F$ for the
   40-year-old donor's temporal cortex [knowledge base, full text]. The
   Benavides-Piccione work computed $F$ for MCA1 and HCA1 and took the Htemp
   spine-dimension values from Eyal 2018. Treating 1.9 and 2.39 as two
   independent comparison values was wrong; they are two donors in one study,
   and their difference is inter-individual variability.

2. **"The head-area deficit is more than sufficient to explain the F gap"**
   (stated after the four-cell run) was premature. It assumed
   $A_{\mathrm{head}}$ was a measurement comparable to Eyal's 2.88 um^2. Given
   Section 3.3, the comparison is between incompatible methods, and given
   Section 3.4 the pipeline value is a lower bound. The gap may be method,
   inter-individual variability, real biology, or any combination.

### 3.7 What is NOT affected

*Recorded so the next chat does not re-open a settled question.*

The spine-neck axial resistance result is independent of everything above.
Necks are tubes traversed longitudinally, so biases (a) and (c) do not apply,
and $R_{\mathrm{neck}}$ depends on radii and lengths, not on the area model at
all:

$$R_{\mathrm{neck}}(\sigma) \;=\; \rho_a \sum_{i \in \mathrm{neck}(\sigma)} \frac{L_i}{\pi\, r_{1,i}\, r_{2,i}} \tag{12}$$

$$\kappa(\sigma) \;=\; \frac{1}{1 + \hat{g}_{\mathrm{syn}}\, R_{\mathrm{neck}}(\sigma)} \tag{13}$$

Measured $R_{\mathrm{neck}}$ was 19--32 MOhm at $\rho_a = 200$ Ohm cm by
Eq. (12), giving via Eq. (13) a median $\kappa \ge 0.92$ across $\rho_a \in [100, 400]$ Ohm cm and
$\hat{g}_{\mathrm{syn}} \in [0.2, 2.0]$ nS. Nothing in this document changes
that.

---

## 4. Summary of results

1. The pipeline's spine membrane area is a sum of **open frustum lateral
   areas** along the skeleton, Eq. (3)--(4) of Section 3.1, using the slant
   height Eq. (2) rather than the axial length.
2. The head/neck partition Eq. (5) is exact by construction and asserted every
   build (Section 3.1).
3. Measured on four L3 excitatory cells, Eq. (7) and Eq. (8): mean
   $F_{\mathrm{lit}} = 1.659$,
   median $A_{\mathrm{head}} = 1.081$ um^2, head share 0.492 (Section 3.2).
4. The published comparison values were produced by **closed-mesh
   reconstruction from confocal stacks**, a different mathematical object from
   a frustum chain (Section 3.3).
5. Published $F$ for human **temporal basal** dendrites differs by **26
   percent between two donors** in one study, one method: 1.89 against 2.39.
   Measured 1.659 is 0.56 SD below the 85-year-old value (Section 3.3).
6. Published human head area spans a **factor of 3.5** (0.82 to 2.88 um^2)
   depending on measurement convention; the H01 value of 1.08 lies inside that
   range (Section 3.3).
7. Four systematic biases identified, with signs: no end caps (under, ~25
   percent of a sphere head), shaft footprint not removed (over, ~14x smaller),
   lateral area wrong for blobs (under, scales as $L/2r$), chorded lengths
   (sign unestablished) --- Section 3.4.
8. Biases (a) and (c) apply to heads but **not** to necks, consistent with the
   observed pattern that neck dimensions match the literature and head area
   does not (Section 3.4).
9. A head-shape bracket Eq. (9)--(11) is now emitted per spine, with a known
   positive offset from the neck-to-head shoulder (Section 3.5).
10. The neck-resistance and $\kappa$ results, Eq. (12)--(13), are unaffected
    (Section 3.7).

---

## 5. Open points, caveats, and assumptions

**Immediate, cheap, and decisive**

1. **Run the bracket on the real bank.** Re-run CELL 6c and read
   `head_lateral_over_sphere_median` and `head_path_over_radius_median`. The
   prediction is $\chi \approx 0.33$ by Eq. (10) and $\lambda \approx 0.75$ by
   Eq. (11) (Section 3.5).
   If $\chi \approx 1$, bias (c) is absent and the head area is a
   measurement --- which would make the whole of Section 3.4(c) inapplicable
   and point the investigation back at the labeller or at biology.

2. **Establish what `r` is.** Unresolved and load-bearing. Is the H01 skeleton
   radius a medial-axis inscribed-sphere radius, a fitted membrane radius, or
   something else, and at which mip level? Every area and every resistance in
   the pipeline is a function of it. Until this is known, no correction to
   Eq. (3) can be justified.

3. **Check whether skeleton X is compression-corrected.** The H01
   supplementary [knowledge base, full text] states that volume was computed
   "treating each voxel as 5.552 nm in X and 4 nm in Y", i.e. there is a real
   anisotropic sectioning compression in X. If the released skeleton
   coordinates are uncorrected, X extents are short by 28 percent, worth at
   most ~14 percent of area on a roughly isotropic head. Not sufficient alone,
   but it is a one-line check and it compounds with everything else.

**Assumed without proof**

4. Circular cross-sections and linear radius taper between adjacent nodes.
   Real spine necks are neither. No estimate of the resulting error exists.
5. That the labeller's head/neck split is correct. `spine_geometry` does not
   re-derive it. Spines whose label sequence has no neck (the labeller assigns
   everything to head when a path has fewer than 3 distinct nodes) are excluded
   from resistance statistics but still contribute area.
6. That a circumscribed sphere at $r_{\max}$ is a genuine upper bound on head
   area. It ignores that a real head is not spherical and is partly occluded
   where the neck joins it. It is a bracket, not a proof.

**Deliberately not done**

7. **No arithmetic was changed.** Adding end caps to Eq. (3) would be a
   plausible-looking fix that changes every $F$ in the project, and it should
   not be done until points 1 and 2 are closed. In particular, if the intended
   comparison target is a closed mesh area (Section 3.3), the right response
   may be to report a bracket rather than to patch the frustum sum toward one
   end of it.
8. **Spine density validation (S1.7) is untouched.** Whether $\psi(\nu, b, d)$
   matches published distance-resolved profiles is a separate question with a
   separate confound (the published profiles are basal-only), and it is not
   addressed here.

**Unresolved question of principle**

9. Which quantity does the model actually need? $F$ enters the model by scaling
   $c_m$ and $R_m$, so what matters is the **electrically relevant membrane
   area** --- the area that carries capacitance and leak. Whether that is better
   approximated by a closed mesh area or by a frustum sum is not obvious, and
   the answer may differ between head and neck. This has not been argued
   anywhere in the project and probably should be, before any correction is
   adopted.

---

## 6. References / further reading

Provenance is marked for every entry. Nothing below is stated from memory.

**Knowledge base, full text read**

- Eyal, Verhoog, Testa-Silva, Deitcher, Lodder, Benavides-Piccione,
  Morales, DeFelipe, de Kock, Mansvelder, Segev (2016). *Unique membrane
  properties and enhanced signal processing in human neocortical neurons.*
  eLife 5:e16553. --- Source of the $F$ definition as used here, the per-donor
  per-region $F$ table in Section 3.3, the $d_{\mathrm{cut}} = 60$ um
  convention, and the confocal/Imaris/Marching-Cubes area method.
- Eyal et al. (2018), *Human Cortical Pyramidal Neurons: From Spines to Spikes
  via Models*, and its supplementary material. --- Spine head area
  $2.88 \pm 1.37$ um^2, neck length $1.34 \pm 0.50$ um, neck diameter
  $0.24 \pm 0.08$ um ($n = 150$ human L3 spines); two-compartment spine model
  (neck cylinder $1.35 \times 0.25$ um, head as an isopotential compartment of
  2.8 um^2); neck resistance 50--80 MOhm with a 19--128 MOhm envelope.
- Benavides-Piccione, Rojo, Kastanauskaite, DeFelipe. *Principles for Dendritic
  Spine Size and Density in Human and Mouse Cortical Pyramidal Neurons.* ---
  Imaris multi-threshold spine reconstruction methodology; $F$ computed at
  $d \ge 60$ um for human and $\ge 30$ um for mouse; confirms Htemp spine
  dimensions were taken from Eyal 2018 rather than measured independently.
- *Comprehensive analysis of human dendritic spine morphology and density*,
  J Neurophysiol. --- Head diameter $511 \pm 12$ nm, neck diameter
  $258 \pm 10$ nm, whole-spine surface area $7.06 \pm 0.34$ um^2 (manual),
  spine length $1301 \pm 44$ nm; note the $1$--$30$ um^2 filter.
- Shapson-Coe et al. (2024). *A petavoxel fragment of human cerebral cortex
  reconstructed at nanoscale resolution.* Science 384:eadk4858, and
  supplementary. --- H01 acquisition geometry (4 x 4 nm^2 pixels, mean section
  thickness 33.9 nm, block approximately 3 x 2 mm in-plane and ~170 um deep);
  X-axis sectioning compression, volume computed treating each voxel as
  5.552 nm in X and 4 nm in Y.

**PubMed, abstract only --- flagged, no numeric content used**

- Rapp, Segev, Yarom (1994). J Physiol. doi:10.1113/jphysiol.1994.sp020006.
  PMC full text returns empty (scanned). The original reference for global
  spine incorporation via $F$, i.e. the source method for the assumption this
  document's Eq. (6) implements. **Should be obtained and read before any
  change to Eq. (3)--(6).**

**Project code, read directly**

- `spine_density.py` v1.2.0 --- `_frustum_lateral_area` (Eq. 3),
  `_attribute_spine_area` (Eq. 4), `_path_distances_um`, `build_phi`,
  `cell_f_implied_from_phi` (Eq. 7), `cell_f_beyond_cutoff` (Eq. 8).
- `spine_geometry.py` v1.0.0 --- `build_spine_geometry` (Eq. 5, 9--12),
  `cell_spine_summary`, `_self_check`.
- `phi_pipeline_colab.py` --- `radius_report`, the authoritative radius gate.

**Own reasoning, not from any source**

- The bias inventory of Section 3.4, its signs, and the magnitude estimates.
- The $\lambda \approx 0.75$ prediction of Section 3.5.
- The argument in Section 5 point 9 that the electrically relevant area, not
  the geometric area, is what $F$ should represent.
