# Validating skeleton-derived spine membrane area against H01 meshes

**Towards-EEG — handoff for a new chat**
**Date:** 26 August 2026
**Supersedes/extends:** `TEEG_20_spine_membrane_area_handoff_v1.md`
**Status of the code discussed here:** S1.3 complete, `ALL GREEN`, 4 cells processed

## Abstract

Stage S1 of the Towards-EEG pipeline derives per-spine membrane area
$A_{\mathrm{spine}}(\sigma)$ for 19,876 dendritic spines on four H01 human L2/3
pyramidal neurons, by integrating the lateral surface of a chain of truncated
cones fitted to the released skeleton nodes and their radii, and closing each
true skeleton leaf with a spherical cap of pole height $h = 0.100\ \mu m$ (the
documented H01 endpoint erosion). The resulting spine-head areas are
**2.5-fold smaller** than the values published for human L2/3 temporal cortex,
and the corresponding membrane-folding factor $F$ sits at the bottom of the
published band. The scientific question this document poses is: **is that
discrepancy a property of the tissue, of the H01 radius convention, or of the
surface-of-revolution model itself?** These three cannot be separated from
aggregate statistics alone, and the discrepancy must be resolved before
$F$ propagates into the passive-membrane fits and thence into the EEG forward
model, where it scales $c_m$ and $1/R_m$ multiplicatively.

The document specifies a validation that can separate them: a **per-spine
comparison of the skeleton-derived area against the area of the same spine's
patch of the released H01 triangle mesh**. It states the problem, the
quantities involved, the resources required, and what must be carried into the
new chat.

**Covered:** the definition of the estimand, the four candidate explanations
of the discrepancy, the mesh-comparison design, its known confounds, and the
inputs needed.
**Deliberately excluded:** implementation code (none is written yet); any
change to the cap model, which is settled; the downstream passive-fit and EEG
stages; and the base-segment attribution question, which is logged in §5 as a
separate open item.

---

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in § |
|---|---|---|---|---|
| $\sigma$ | A single spine: a maximal connected component of spine-labelled nodes | index, $\sigma \in \Sigma$ | — | §3.1 |
| $\Sigma$ | The set of all spines on one cell | finite set, $\lvert\Sigma\rvert = N_{\mathrm{sp}}$ | — | §3.1 |
| $N_{\mathrm{sp}}$ | Number of spines on a cell | $N_{\mathrm{sp}} \in \mathbb{N}$ | dimensionless | §3.1 |
| $\mathcal{M}(\sigma)$ | Node set of spine $\sigma$ | finite subset of node ids | — | §3.1 |
| $\mathcal{T}(\sigma)$ | Tips of $\sigma$ that are leaves of the **full** skeleton | $\mathcal{T}(\sigma) \subseteq \mathcal{M}(\sigma)$ | — | §3.2 |
| $N_{\mathrm{tip}}$ | Number of capped tips on a cell, $\sum_{\sigma} \lvert\mathcal{T}(\sigma)\rvert$ | $\in \mathbb{N}$ | dimensionless | §3.2 |
| $m$ | A skeleton node | index | — | §3.1 |
| $p(m)$ | Parent of node $m$ in the rooted skeleton tree | $p : \text{ids} \to \text{ids} \cup \{-1\}$ | — | §3.1 |
| $r_m$ | Radius recorded at node $m$ | $r_m \in \mathbb{R}_{>0}$ | µm | §3.1 |
| $r_t$ | Terminal radius, i.e. $r_m$ at a tip $t \in \mathcal{T}(\sigma)$ | $r_t \in \mathbb{R}_{>0}$ | µm | §3.2 |
| $\bar{r}_t$ | Arithmetic mean of $r_t$ over capped tips | $\in \mathbb{R}_{>0}$ | µm | §3.3 |
| $\tilde{r}_t$ | Root-mean-square of $r_t$, $\sqrt{\mathbb{E}[r_t^2]}$ | $\in \mathbb{R}_{>0}$ | µm | §3.3 |
| $L_{ab}$ | Euclidean distance between nodes $a$ and $b$ | $\in \mathbb{R}_{\ge 0}$ | µm | §3.1 |
| $s_{ab}$ | Slant length of the frustum $a \to b$ | $\in \mathbb{R}_{\ge 0}$ | µm | §3.1 |
| $A_{ab}$ | Lateral area of the frustum $a \to b$ | $\in \mathbb{R}_{\ge 0}$ | µm² | §3.1 |
| $h$ | Pole height of the tip cap; fixed at the H01 erosion, $h = 0.100$ | $h \in \mathbb{R}_{\ge 0}$ | µm | §3.2 |
| $R_{\mathrm{cap}}$ | Radius of the sphere the cap is a section of | $\in \mathbb{R}_{>0}$ | µm | §3.2 |
| $A_{\mathrm{cap}}(r_t, h)$ | Lateral area of one tip cap | $\in \mathbb{R}_{\ge 0}$ | µm² | §3.2 |
| $A_{\mathrm{spine}}^{\mathrm{skel}}(\sigma)$ | Skeleton-derived membrane area of spine $\sigma$, cap included | $\in \mathbb{R}_{>0}$ | µm² | §3.2 |
| $A_{\mathrm{spine}}^{\mathrm{mesh}}(\sigma)$ | Mesh-derived membrane area of the **same** spine | $\in \mathbb{R}_{>0}$ | µm² | §4.2 |
| $\kappa(\sigma)$ | Per-spine ratio $A^{\mathrm{mesh}}_{\mathrm{spine}}(\sigma)/A^{\mathrm{skel}}_{\mathrm{spine}}(\sigma)$ | $\in \mathbb{R}_{>0}$ | dimensionless | §4.3 |
| $A_{\mathrm{head}}(\sigma)$ | Portion of $A_{\mathrm{spine}}^{\mathrm{skel}}(\sigma)$ on head-labelled segments | $\in \mathbb{R}_{\ge 0}$ | µm² | §3.4 |
| $A_{\mathrm{shaft}}$ | Total dendritic shaft membrane area of a cell | $\in \mathbb{R}_{>0}$ | µm² | §3.4 |
| $F$ | Membrane-folding factor, $F = 1 + \left(\sum_{\sigma} A_{\mathrm{spine}}(\sigma)\right)/A_{\mathrm{shaft}}$ | $F \in [1, \infty)$ | dimensionless | §3.4 |
| $F_{\mathrm{lit}}$ | $F$ restricted to dendrite beyond 60 µm path distance from soma | $\in [1,\infty)$ | dimensionless | §3.4 |
| $c_m$ | Specific membrane capacitance | $\in \mathbb{R}_{>0}$ | µF/cm² | §2 |
| $R_m$ | Specific membrane resistance | $\in \mathbb{R}_{>0}$ | Ω·cm² | §2 |
| $\mathcal{V}$ | The H01 segmentation volume (voxel grid) | — | — | §4.1 |
| $\mathcal{S}_{\mathrm{id}}$ | Triangle mesh released for H01 segment id | set of triangles in $\mathbb{R}^3$ | — | §4.1 |
| $\mathrm{nn}(\cdot)$ | Nearest-skeleton-node assignment for a mesh face centroid | $\mathbb{R}^3 \to$ node ids | — | §4.2 |
| $d_{\mathrm{gap}}$ | Distance from the last skeleton node to the furthest mesh vertex at that tip | $\in \mathbb{R}_{\ge 0}$ | µm | §4.4 |

### 1.1 Conventions

- **Units.** All lengths in µm, all areas in µm², throughout. H01 releases
  coordinates and radii in nm; conversion happens once, at
  `spine_density._prepare_nodes`, and nothing downstream sees nm.
- **Indices.** $m$ always ranges over skeleton nodes; $\sigma$ always over
  spines; $t$ always over capped tips. Node ids are opaque integers, never
  assumed contiguous or ordered.
- **Tree orientation.** "Parent" means towards the soma; "distal", "tip" and
  "leaf" mean away from it. A **leaf** is a node with zero children in the
  **full, unfiltered** skeleton — never in a label-filtered subtree. This
  distinction is load-bearing and is the subject of §3.2.
- **Superscripts** `skel` and `mesh` denote which representation an area was
  computed from. They are never dropped: an unqualified $A_{\mathrm{spine}}$
  would be ambiguous in this document and does not appear.
- **Estimand vs estimate.** $A_{\mathrm{spine}}^{\mathrm{skel}}$ and
  $A_{\mathrm{spine}}^{\mathrm{mesh}}$ are two *estimators* of one physical
  quantity, the area of a 2-manifold in $\mathbb{R}^3$. Neither is the truth.
  §4.3 is explicit about this.

---

## 2. Glossary

Ordered by first appearance, because the concepts build on each other.

- **Frustum chain.** The geometric model underlying
  $A_{\mathrm{spine}}^{\mathrm{skel}}$: consecutive skeleton nodes are joined
  by truncated cones whose end radii are the node radii, and the membrane is
  taken to be the union of their lateral surfaces. Equivalent to a surface of
  revolution about the skeleton path with a piecewise-linear radius profile.
  Operative from §3.1.
- **Surface of revolution.** A surface generated by rotating a plane curve
  about an axis. The frustum chain is exactly this, which is why it can never
  represent a non-circular cross-section, however finely the skeleton is
  sampled. Operative in §5.
- **Endpoint erosion.** A step in the H01 skeletonisation: skeleton endpoints
  were trimmed back 100 nm from the extent of the segmentation. It is the
  reason the frustum chain leaves each tip open and the reason $h$ is a
  documented constant rather than a fitted parameter. Operative from §3.2.
- **Sparsification.** The companion H01 step: skeleton nodes were decimated to
  approximately 300 nm spacing, retaining all branch points and endpoints.
  Sets the spatial resolution at which any skeleton-derived shape claim can be
  made. Operative in §5.
- **Spherical cap (Archimedes' hat-box theorem).** The lateral area of a
  spherical zone equals $2\pi R \times (\text{axial height})$, independent of
  where on the sphere the zone sits. It is what makes $A_{\mathrm{cap}}$
  depend only on $(r_t, h)$. Operative from §3.2.
- **Marching Cubes.** The standard algorithm for extracting a triangulated
  isosurface from a voxel grid. Both the H01 meshes and the published confocal
  spine areas were produced this way, from different data. Operative in §4.1.
- **Membrane-folding factor $F$.** The multiplicative correction applied to
  $c_m$ and $1/R_m$ to account for spine membrane that is not explicitly
  represented in a compartmental model. **Note the everyday-vs-technical
  clash:** "folding" here has nothing to do with cortical gyrification. It is
  the ratio of true membrane area to modelled cable area. Operative in §3.4.
- **Label boundary.** A spine-labelled node whose only children are
  non-spine-labelled. It looks like a spine tip to a label-filtered traversal
  but is *not* a skeleton endpoint, so it never underwent erosion. Capping one
  would fabricate membrane. Operative in §3.2.
- **Truncation flag.** A per-tip classification (`taper`, `boundary`, `both`,
  `none`) marking dendritic tips that end because the imaged volume ended
  rather than because the neurite did. Operative in §5.
- **Estimand.** The quantity one is trying to estimate, as distinct from any
  particular estimator of it. Used here to insist that the mesh is not truth.
  Operative in §4.3.

---

## 3. What is established, and what the numbers are

This section states the current state of the pipeline. Everything in it is
measured output from the 26 August 2026 run, not projection.

### 3.1 The skeleton-derived area

For each ordered pair of nodes $(a,b)$ with $p(b) = a$, both carrying radii
$r_a = r_m\big|_{m=a} > 0$ and $r_b = r_m\big|_{m=b} > 0$ drawn from the node
radius field $r_m$:

$$L_{ab} = \lVert \mathbf{x}_b - \mathbf{x}_a \rVert_2,
\qquad
s_{ab} = \sqrt{(r_a - r_b)^2 + L_{ab}^2},
\qquad
A_{ab} = \pi (r_a + r_b)\, s_{ab}
\tag{1}$$

The slant $s_{ab}$, not $L_{ab}$, is used; this matters wherever
$\lvert r_a - r_b \rvert \gtrsim L_{ab}$, i.e. at the neck-to-head shoulder.

**Established property (proved, and asserted in the test suite):** with a
piecewise-linear radius profile, $\sum A_{ab}$ is *exactly invariant* under
subdivision of a frustum. Resampling the skeleton at finer spacing therefore
changes nothing, and any scheme that reports a different number after
resampling has changed the geometric model, not refined the arithmetic.

### 3.2 The tip cap

For each fixed tip $t \in \mathcal{T}(\sigma)$, a sphere is placed through the
rim circle of radius $r_t$ with its pole at axial height $h$ beyond it:

$$R_{\mathrm{cap}} = \frac{r_t^2 + h^2}{2h},
\qquad
A_{\mathrm{cap}}(r_t, h) = 2\pi R_{\mathrm{cap}} h = \pi\left(r_t^2 + h^2\right)
\tag{2}$$

$$A_{\mathrm{spine}}^{\mathrm{skel}}(\sigma)
= \sum_{m \in \mathcal{M}(\sigma)} A_{p(m),\,m}
\;+\; \sum_{t \in \mathcal{T}(\sigma)} \pi\left(r_t^2 + h^2\right),
\qquad h = 0.100\ \mu m
\tag{3}$$

$R_{\mathrm{cap}}$ cancels in (2). The construction never fails, since
$R_{\mathrm{cap}} \ge r_t$ reduces to $(r_t - h)^2 \ge 0$ for all
$r_t, h \ge 0$. Setting $h = 0$ gives the flat disc $\pi r_t^2$, used as the
lower bound of the reported bracket. Caps are applied symmetrically to spine
leaves and to shaft leaves; capping only the spine side would raise $F$ by
construction.

**Gate 1 result (measured):** across all four cells, **zero** label-boundary
ends and **zero** tips with $r_t \le 0$. Every capped node is a genuine
skeleton leaf. The cap is therefore legitimately applied.

### 3.3 The terminal radius distribution

| Cell | $N_{\mathrm{sp}}$ | $N_{\mathrm{tip}}$ | $\bar{r}_t$ (µm) | $\tilde{r}_t$ (µm) | $\mathrm{CV}(r_t)$ | neck radius (µm) |
|---|---|---|---|---|---|---|
| 1302789404 | 6626 | 7086 | 0.093 | 0.125 | 0.89 | 0.130 |
| 1317492596 | 4164 | 4391 | 0.095 | 0.137 | 1.03 | 0.150 |
| 1333261412 | 3914 | 4111 | 0.102 | 0.147 | 1.03 | 0.165 |
| 1376890291 | 5172 | 5677 | 0.094 | 0.133 | 0.99 | 0.151 |
| **pooled** | **19876** | **21265** | **0.096** | **0.134** | **~0.98** | — |

Three things follow, all measured:

1. **1.070 tips per spine** — about 7% of spines are branched.
2. **$r_t$ is strongly right-skewed**, $\mathrm{CV} \approx 1$. Consequently
   $\tilde{r}_t / \bar{r}_t \approx 1.4$. Because $A_{\mathrm{cap}} \propto
   r_t^2 + h^2$, the aggregate cap is governed by $\tilde{r}_t$, not
   $\bar{r}_t$; the two must never be interchanged.
3. **$\bar{r}_t \approx h$.** The mean tip cap is therefore close to a
   hemisphere, and in aggregate
   $A_{\mathrm{cap}}/A_{\mathrm{disc}} = 1 + h^2/\tilde{r}_t^{\,2} = 1.56$.
   The disc-vs-cap choice is a 56% effect on the cap, not a rounding detail,
   which is why the bracket is reported rather than a point estimate.

### 3.4 The folding factor, and the size of the cap

Pooled over the four cells, the cap adds $1868.5\ \mu m^2$ to a total spine
area of $48358.0\ \mu m^2$, i.e. **3.86%**.

| quantity | no cap | flat disc ($h=0$) | cap ($h=0.100$) |
|---|---|---|---|
| mean $f_{\mathrm{implied}}$ | 1.4732 | 1.4847 | 1.4913 |
| mean $F_{\mathrm{lit}}$ | 1.6588 | 1.6718 | **1.6810** |

Per-cell $A_{\mathrm{shaft}}$ ranges from 17,578 to 44,721 µm², which is what
makes the shaft-side caps (14–73 µm² per cell) negligible in $F$ while the
spine-side caps (396–569 µm² per cell) are not.

Published human L2/3 comparison values, from confocal Marching-Cubes
reconstructions: temporal basal $1.89 \pm 0.41$ (85 y) and $2.39 \pm 0.63$
(40 y); cingulate basal $1.81 \pm 0.34$ (85 y); the value adopted for L2/3
modelling is $F = 1.9$.

$$\boxed{\text{The cap closed } 3.9\% \text{ of spine area and moved } F_{\mathrm{lit}} \text{ by } +0.022. \text{ It is not the explanation for anything.}}$$

### 3.5 The discrepancy that motivates this document

| quantity | measured here | published (human L2/3 temporal) | ratio |
|---|---|---|---|
| median $A_{\mathrm{head}}(\sigma)$ | **1.16 µm²** | $2.88 \pm 1.37\ \mu m^2$ | **2.48** |
| $F_{\mathrm{lit}}$ | 1.681 | 1.89 ± 0.41 (85 y) | 1.12 |

And the puzzle that makes this non-obvious: the spines are **not**
undersampled. The labeller assigns 51287 / 31420 / 29842 / 37579 spine nodes
across the four cells, i.e. **7.3 to 7.7 nodes per spine**. At nominal 300 nm
spacing that is ~2.2 µm of path per spine, against a median neck length of
0.94 µm. There are enough nodes to resolve a head. Yet the mean terminal
radius, 0.096 µm, is **only 60–70% of the median neck radius** — the spine is
at its narrowest where a head should be widest.

---

## 4. The proposed validation

### 4.1 What is available

The H01 release contains, for each segment id, both the skeleton used here
*and* a triangle mesh $\mathcal{S}_{\mathrm{id}}$ extracted from the same voxel
segmentation $\mathcal{V}$. The mesh is an **independent representation of the
same physical membrane**: same tissue, same segmentation, different geometric
model. It is the only reference available that is not confounded by species,
preparation, fixation or imaging modality.

### 4.2 The comparison

For one cell, for each fixed spine $\sigma$:

1. Fetch $\mathcal{S}_{\mathrm{id}}$ at the **finest** level of detail.
   Multi-resolution meshes are decimated at coarse LODs and a coarse LOD would
   make the comparison meaningless.
2. Assign each mesh face to a skeleton node by nearest neighbour on the face
   centroid: $\mathrm{nn}(\mathbf{c}_f)$.
3. Define the spine's mesh patch as
   $\{f : \mathrm{nn}(\mathbf{c}_f) \in \mathcal{M}(\sigma)\}$ and set
   $A_{\mathrm{spine}}^{\mathrm{mesh}}(\sigma) = \sum_f \mathrm{area}(f)$.
4. Report the per-spine ratio

$$\kappa(\sigma) = \frac{A_{\mathrm{spine}}^{\mathrm{mesh}}(\sigma)}
{A_{\mathrm{spine}}^{\mathrm{skel}}(\sigma)}
\tag{4}$$

as a distribution over $\sigma \in \Sigma$, **and** its aggregate
$\sum_{\sigma \in \Sigma} A^{\mathrm{mesh}}_{\mathrm{spine}}(\sigma) \big/
\sum_{\sigma \in \Sigma} A^{\mathrm{skel}}_{\mathrm{spine}}(\sigma)$,
taken over all spines of the cell.

### 4.3 What $\kappa(\sigma)$ isolates, and what it does not

Both objects are **open at the base** — cutting a patch from a closed neuron
mesh leaves a boundary at the spine base, exactly as the frustum chain does —
so the base convention cancels in (4). What does not cancel, and is therefore
what $\kappa$ measures, is the sum of:

- non-circular cross-section, which a surface of revolution cannot represent;
- the H01 radius convention (see §5, item 1);
- residual tip closure error after the cap;
- mesh surface roughness inherited from the voxel grid.

Running the comparison with the cap **on** and **off** separates the third
term from the others: the change in $\kappa$ is attributable to the cap alone.

**The mesh is not ground truth.** Surfaces extracted from a voxel grid inherit
staircase roughness, and area is the quantity most sensitive to it — this is
the same non-continuity of the area functional that makes a stack of cylinders
converge to the wrong answer. Treat $\kappa$ as an upper-leaning reference and
report it as such.

### 4.4 The cheap precursor, which should be done first

Validating $h$ requires only one number, not the whole area comparison. For a
few dozen spines: take the mesh patch, project its vertices onto the local tip
axis, and measure $d_{\mathrm{gap}}$, the distance from the last skeleton node
to the furthest vertex. If the distribution of $d_{\mathrm{gap}}$ centres near
0.100 µm, the erosion assumption is confirmed by measurement rather than by a
sentence in a supplement. If it centres elsewhere, that value **is** the
measured $h$, and (3) still applies unchanged with the new constant.

An afternoon of work, and it converts $h$ from an assumption into a datum.

---

## 5. Summary of results

| # | Statement | Derived in |
|---|---|---|
| R1 | $A_{ab} = \pi(r_a+r_b)s_{ab}$; subdivision-invariant under linear interpolation, so resampling is a no-op | §3.1, Eq. (1) |
| R2 | $A_{\mathrm{cap}} = \pi(r_t^2 + h^2)$, independent of $R_{\mathrm{cap}}$; never fails; $h=0$ recovers the flat disc | §3.2, Eq. (2) |
| R3 | Gate 1 clean: 0 label-boundary ends, 0 non-positive radii, across 21265 tips | §3.2 |
| R4 | $\bar{r}_t = 0.096\ \mu m$, $\tilde{r}_t = 0.134\ \mu m$, $\mathrm{CV} \approx 1$; use $\tilde{r}_t$ for aggregates | §3.3 |
| R5 | $\bar{r}_t \approx h$, so $A_{\mathrm{cap}}/A_{\mathrm{disc}} = 1.56$ in aggregate; report the bracket | §3.3 |
| R6 | The cap adds 3.86% of spine area, $\Delta F_{\mathrm{lit}} = +0.022$ | §3.4 |
| R7 | Median $A_{\mathrm{head}}$ is 2.48× below the published human L2/3 value | §3.5 |
| R8 | 7.3–7.7 nodes per spine: the spines are **not** undersampled, so R7 is not simply a sampling artefact | §3.5 |
| R9 | $r_t$ is 60–70% of the neck radius: the spine is narrowest where a head should be widest | §3.5 |
| R10 | $\kappa(\sigma)$ from Eq. (4) isolates cross-section shape + radius convention + residual tip error + mesh roughness, with the base convention cancelling | §4.3 |

---

## 6. Open points, caveats and assumptions

**The four candidate explanations of R7, which the validation must separate:**

1. **Radius convention.** It is *not established* how the per-node radius in
   the H01 release is defined — inscribed-sphere radius from a distance
   transform, mean cross-sectional radius, or $\sqrt{A_{\mathrm{cross}}/\pi}$.
   These differ systematically for non-circular cross-sections, and an
   inscribed radius is a **lower bound**. The H01 supplement, as read, does
   not state it. This must be pinned down, in the Kimimaro source if not in
   the papers, before any conclusion is drawn. **This is the single most
   likely explanation of R9 and should be checked first.**
2. **Anisotropic voxels.** The segmentation grid is 8×8×33 nm and the
   skeletonisation ran at 32×32×33 nm. A radius derived from a distance
   transform on an anisotropic grid is not isotropically unbiased.
3. **Surface of revolution.** Even with perfect radii, the model cannot
   represent a non-circular cross-section. Spine heads are not surfaces of
   revolution.
4. **Genuine difference.** H01 is EM from a 45-year-old surgical sample;
   the published values are confocal from 40- and 85-year-old material, with a
   0.84 z-distension correction and manual intensity thresholds. These are not
   the same estimand, and a real difference cannot be excluded a priori.

**Assumptions made without proof:**

- That $h = 0.100\ \mu m$ applies uniformly at every leaf. Taken from the H01
  supplement; **not yet measured**. §4.4 is the test.
- That the spine label set correctly partitions spine from shaft. Gate 1's
  zero boundary count is consistent with this but does not prove it.
- That the nearest-node face assignment in §4.2 partitions the mesh sensibly
  at the spine base. It will not, exactly; this is why §4.3 insists on the
  aggregate ratio alongside the per-spine distribution.

**Separate open items, logged and deliberately not addressed here:**

- **Base-segment attribution.** The segment from the shaft node to the spine
  root runs from the *full shaft radius* down to the neck radius, and
  `_attribute_spine_area` counts all of it as spine. On a synthetic fixture
  this was 53–87% of $A_{\mathrm{spine}}$. Its magnitude on real H01 cells is
  **unmeasured** and could exceed the cap by an order of magnitude, with the
  opposite sign. `spine_cap.spine_profile` already returns `A_base_um2` and
  the gallery prints it per panel; the number simply has not been read off
  yet. **Do this before the mesh work — it is free.**
- **Shaft caps on truncated tips.** 35–42% of dendritic tips carry a
  truncation flag (basis `taper`, `both`, or `unresolved`), meaning they end
  where the imaged volume ends. Those tips have no membrane ending, so capping
  them invents surface. The effect on $F$ is small (shaft caps are 14–73 µm²
  per cell against shaft areas of 17,000–45,000 µm²) but it is a correctness
  issue. Blocked on: does the flag distinguish `boundary` proximity from
  `taper`? The printed basis counts show no bare `boundary` category.
- **Non-circular cross-section.** If the validation attributes most of
  $\kappa$ to this, no skeleton-based model can close the gap and the pipeline
  must either adopt an empirical per-spine correction calibrated on
  $\kappa(\sigma)$, or move to mesh-derived areas outright.

**Unresolved questions carried forward:**

- Does the H01 mesh, patched per spine, reproduce the published head areas? If
  it does, the fault is in the skeleton model. If it does not, the published
  comparison itself is the wrong reference.
- Is $\kappa(\sigma)$ roughly constant, or does it depend on spine size? A
  constant $\kappa$ licenses a single scalar correction; a size-dependent one
  does not.

---

## 7. Resources required

**Data:**

- H01 segment ids for the four cells: 1302789404, 1317492596, 1333261412,
  1376890291. Already in hand.
- The released H01 meshes for those ids, finest LOD, from the public GCS
  bucket (`gs://h01-release/...`; index at
  `h01-release.storage.googleapis.com/data.html`). **Not yet fetched.**
- The labelled node frames already produced by S1 (they carry `annotated_type`
  and the spine partition), so the mesh patches can be cut consistently with
  the areas being compared.

**Software:**

- A mesh reader for the released format (`cloud-volume` or the Neuroglancer
  precomputed-format reader), plus `trimesh` or equivalent for per-face areas.
- `scipy.spatial.cKDTree` for the nearest-node assignment.
- Colab, not this sandbox: the sandbox network allowlist does not include
  Google storage, so meshes cannot be fetched here.

**Documentation to locate:**

- The definition of the H01 skeleton node radius (Kimimaro / TEASAR
  implementation, or the H01 methods). Open point 1 above.

---

## 8. What to carry into the new chat

Upload these, and nothing else:

1. **This document.**
2. **`alignment_summary.csv`** — the 26 August run, the one with the 20 cap
   columns (`cap_tips`, `f_implied_nocap/disc/cap`, `F_lit_nocap/disc/cap`,
   `A_spine_um2_nocap/cap`, `total_spine_cap_um2`, `total_shaft_cap_um2`,
   `n_spine_true_leaf`, `n_spine_label_boundary_end`, `mean_r_tip_um`).
3. **`s1_spine_gallery.png`** — still unexamined, and the fastest way to see
   whether the meridians ever widen into a head. If they taper monotonically
   from base to tip, open point 3 is close to settled without any mesh work.
4. **`s1_cap_contribution.png`** and **`s1_tip_radius.png`** — the per-spine
   distribution of the correction and of $r_t$.
5. **`Stage_1/spine_cap.py`** and **`Stage_1/spine_geometry.py`** — the two
   modules the mesh comparison has to agree with. `spine_cap.spine_profile`
   already returns the per-spine node list, the meridian, and `A_base_um2`.
6. **The H01 supplementary methods PDF**, already in the project knowledge
   base, for the skeletonisation parameters.

**Do not re-upload** the notebook driver or the other Stage 1 modules; the
mesh work is a new, separate script and should not touch the S1 pipeline until
$\kappa$ is known.

**Opening request for the new chat, suggested wording:**

> Per the attached handoff, S1 gives spine head areas 2.5× below the published
> human L2/3 values, and the tip cap accounts for only 3.9% of spine area.
> I want to test whether the skeleton model or the radius convention is
> responsible, by comparing per-spine area against the H01 mesh. Start with
> §4.4 (measure $d_{\mathrm{gap}}$ to check $h$), then §4.2. Before either,
> read `A_base_um2` off the existing profiles — §6 says it may dominate.

---

## 9. References

**From the project knowledge base, full text read:**

- Shapson-Coe et al., *A petavoxel fragment of human cerebral cortex
  reconstructed at nanoscale resolution*, and its supplementary methods —
  source of the 100 nm endpoint erosion, the ~300 nm sparsification, the
  32×32×33 nm skeletonisation resolution, and the mesh/Neuroglancer release.
- Eyal et al. 2016, *eLife* — the Marching-Cubes spine-area protocol, the
  explicit statement that no head/neck border was applied when measuring the
  areas that produced $F$, the two-compartment spine model, and the $F$ values
  quoted in §3.4.
- Eyal et al. 2018, *Front Cell Neurosci* — the $F$-folding equation applied
  to $c_m$ and $R_m$.
- *Principles for Dendritic Spine Size and Density in Human and Mouse Cortical
  Pyramidal Neurons*, J Comp Neurol — the $2.88 \pm 1.37\ \mu m^2$ head area
  for human L2/3 temporal.
- *Comprehensive analysis of human dendritic spine morphology and density*,
  J Neurophysiol — the manual 2-D head/neck diameter protocol.

**From PubMed, full text retrieved:**

- Arellano, Benavides-Piccione, DeFelipe & Yuste 2007, *Front Neurosci*,
  doi:10.3389/neuro.01.1.1.010.2007 — serial-section EM spine reconstruction;
  the head treated as a **cylinder**,
  $d_{\mathrm{head}} = 2\sqrt{V_{\mathrm{head}}/(\pi L_{\mathrm{head}})}$; and
  the finding that stubby/thin/mushroom classes did not separate.

**From PubMed, abstract only — flagged, no numeric content used:**

- Rapp, Segev & Yarom 1994, *J Physiol*,
  doi:10.1113/jphysiol.1994.sp020006 — origin of the $F$-folding method.
  Full text not accessible through the connector; **if the PDF can be obtained
  through library access it should be uploaded to the new chat.**

**Searched and returned nothing relevant:**

- PubMed, six queries on skeleton-vs-mesh area validation, Marching-Cubes area
  accuracy, skeletonisation radius conventions, and spine head surface area
  from EM. All returned zero results. **No published method for the validation
  in §4 was found; it should be treated as novel and described as such.**
- bioRxiv/medRxiv: the connector supports category and date filtering only,
  with no keyword search, so it could not be aimed at this question. A targeted
  preprint search remains outstanding and should be done by hand on the
  bioRxiv website.

**Stated from this session's own computation, not from any source:** Eqs.
(1)–(4); the subdivision-invariance proof; the $\tilde{r}_t$-vs-$\bar{r}_t$
distinction and all values in §3.3; the aggregate cap fractions in §3.4; the
$\kappa$ design in §4 and its confound analysis in §4.3.
