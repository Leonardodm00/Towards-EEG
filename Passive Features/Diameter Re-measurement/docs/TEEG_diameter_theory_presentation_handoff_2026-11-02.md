# Handoff: the theory of the diameter pipeline, for the presentation (diameter theory chat -> presentation chat)

| Date | Change |
|---|---|
| 2026-11-02 | v1. Written by the diameter theory chat at the user's request: "the handoff with all the images you judge important for a presentation ... for the theory part of the diameter pipeline", covering all the theory, with the GitHub locations of the theory documents, for a chat that writes the documents for the design of the presentation. Sources: the repository at `main` 854004c and `sci/diameter-pipeline` 5015806 (2026-11-02); five located reading digests of the documents (line numbers re-grepped by each reader); spot checks of the key equations, numbers and decisions by this chat (section 16); this chat's optics checks (section 13); five new figures with their checks (`check_theory_figures.py`, 21 of 24 pass, the 3 failures kept with their diagnosed follow-ups). |

Scope: concepts only, i.e. the physics, mathematics and statistics under the pipeline, organised by the roadmap's theory graph. Implementation status, cluster runs and notebooks are out of scope (the roadmap and the runbook carry them).

Evidence tags used below:

| Tag | Meaning |
|---|---|
| **[run here]** | computed in this chat on 2026-11-02 by a script in the figure pack (output in `checks/`) |
| **[doc]** | stated in a repository document at the cited place, not re-derived here |
| **[log]** | stated in the decision log at the cited line |
| **[src]** | read in the code at 5015806 |
| **[KB]** | project knowledge (cited) |
| **[reasoning]** | this chat's inference |
| **[open]** | unresolved; present it as an open question, not as a result |

Line numbers ("l.") refer to the files at the commits in section 2.2.

---

## 1. First actions in the presentation chat

1. Read this file's sections 3 (the map and the talk order), 12 (cautions) and 14 (figure inventory) first. Then read the topic sections you are designing.
2. Get the documents. The repository is public; the theory documents are split across two branches (section 2.2):
   ```
   git clone https://github.com/Leonardodm00/Towards-EEG.git && cd Towards-EEG
   git fetch origin sci/diameter-pipeline
   D="Passive Features/Diameter Re-measurement"
   git show "origin/main:$D/docs/TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md" | less
   git show "origin/sci/diameter-pipeline:specs/SPEC.md" | less
   git show "origin/sci/diameter-pipeline:specs/roadmap/diameter-pipeline.md" | less
   ```
3. Get the figure pack. It reached the user as `TEEG_diameter_theory_presentation_2026-11-02.zip`; ask for it if it is not attached. After the user applies this chat's patch, its own material is also in the repository at `$D/figures/TEEG_diameter_theory_presentation_2026-11-02/` on `sci/diameter-pipeline`. This session could not push: the push was refused by the session's permissions, so only the patch exists.
4. Ask the user for the chat-only material you need (section 2.4). It covers mainly T4.2/T4.3 (the table's SE, the read-out and the inversion) and T6 (the far-field explanation, with its figures).
5. Standing rules that bind slide text too:
   - The project's full-text rule: no number from a paper whose full text was not read.
   - Evidence classes stay visible.
   - Read the decision log before stating anything a decision constrains.
   - Present open topics (T4.3, T7, T8, T9, T10) as open.
   - D-097: every document a chat writes is also committed to the repository.
   - Project knowledge is full (2.30 MB of 2.00 MB on 2026-11-02), so new documents may live only in the repository and as files.

---

## 2. Where things are

### 2.1 Repository, branches, commits

| Item | Value |
|---|---|
| Repository | `Leonardodm00/Towards-EEG` (public) |
| Folder | `Passive Features/Diameter Re-measurement/` (= `$D` below) |
| `main` | 854004c (2026-11-02). Holds the optics and renderer theory of 2026-10-04 to 10-12 and the four interactive figures |
| `sci/diameter-pipeline` | 5015806 (2026-11-02). Holds the code, `specs/SPEC.md`, `specs/roadmap/diameter-pipeline.{yaml,md}`, the decision-log copy `docs/project/TEEG_decisions_and_ideas_log.md`, and the handoffs from 2026-10-16 on |
| `diameter-remeasurement-docs` | 5a7dff1 (2026-10-05). Historical; nothing newer than `main` |

### 2.2 The theory documents: where the newest version is

All paths are under `$D/docs/` unless stated. "Commit" is the last commit that touched the file on that branch.

| Document | Newest on | Commit (date) | Theory it carries | Topics |
|---|---|---|---|---|
| `handoff_diameter_remeasurement.md` | main = branch | 589a0de (10-06) | Frames and geometry (Math reference Eqs. 5-8); the original per-node method (Eqs. 1-13). Partly superseded, see 12.1 | T1, T2 |
| `TEEG_microscope_optics_oil_immersion_2026-10-04.md` (v1.5) | **main** (the branch copy is older) | fdce497 (10-10) | Light path, NA, in-focus blur, defocus, condenser and coherence, mounting medium (§3.1-3.8) | T7 |
| `TEEG_diameter_optics_notes.md` | main = branch | 589a0de (10-06) | In-focus blur budget, squared widths, the calibration idea | T2, T3 |
| `TEEG_diameter_bias_table_mathematics_2026-10-04.md` | main = branch | 589a0de (10-06) | Forward model Eqs. 8-13; squared widths Eqs. 15-17; calibration regression Eq. 18; bias factor and inversion Eqs. 19-21. The table's design is superseded (12.1) | T2, T3, T4, T10 |
| `TEEG_diameter_bias_table_procedure_2026-10-04.md` | **sci/diameter-pipeline** (the main copy is older) | 76365a3 (10-28) | Kernel Step A §3.4 (Eq. 4); phantoms §3.5; rendering §3.6 (Eqs. 5-6); table steps §3.7-3.11. The table design is superseded | T3, T4, T10 |
| `TEEG_diameter_implementation_handoff_2026-10-06.md` | **main** (the branch copy is older) | bf14eb4 (10-09) | Equations S1-S10 (tube geometry, slabs, partition, ray world) | T3 |
| `TEEG_diameter_theory_handoff_2026-10-07.md` | main only | 81b3327 (10-12) | Findings of the theory chat, including the S = 1 dark-field correction of 2026-10-08 | T7 |
| `TEEG_diameter_study_notes_2026-10-07.md` | main only | bf14eb4 (10-09) | Dark nodes; the partition against the ray world | T3, T7 |
| `TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md` | main only | 2637194 (10-09) | Partition, ray world, hybrid (Eqs. 1-15); coherence §3.10 | T3, T7 |
| `TEEG_diameter_sine_condition_and_W_2026-10-12.md` | main only | 89fd2fb (10-12) | Sine condition, Koehler map, the measure W against W* | T7 |
| `TEEG_diameter_presentation_ideas_2026-10-10.md` | main only | 4f9c258 (10-10) | Idea 1, the kernel in three domains (partly stale, 12.1) | T3, T6 |
| `TEEG_interactive_figures_spec_2026-10-04.md` | main = branch | 589a0de (10-06) | Specs of the four interactive figures | T3, T7 |
| `TEEG_diameter_correction_table_handoff_2026-10-16.md` | branch only | ea49181 (10-16) | The SGT kernel and the fit (H1)-(H6) §3; Monte Carlo and identifiability §4 | T5 |
| `TEEG_diameter_correction_table_handoff_2026-10-19.md` | branch only | 84ea91f (10-19) | Table decisions; settled findings on the inversion (§5.3) | T4, T5 |
| `TEEG_diameter_table_implementation_plan_2026-10-20.md` | branch only | 2572e15 (10-27) | Stages of the table (S1-S6) | T4 |
| `TEEG_diameter_defocus_kernel_DL_handoff_2026-10-22.md` | branch only | 37b42ea (10-22) | The kernel in full notation (§4); what no method identifies (§5); the far-field Monte Carlo (§6) | T6, T10 |
| `TEEG_diameter_calibration_handoff_2026-10-29.md` | branch only | fdd6841 (10-29) | The kernel calibration on the pilot (Cell 7) | T6, T10 |
| `TEEG_thin_plate_spline_in_3D/TEEG_thin_plate_spline_in_3D.md` | branch only | 6a8dbf4 (10-19) | Thin-plate spline theory. Superseded as the table's estimator (D-063, D-067.3) | history |
| `specs/SPEC.md` (repository root) | branch | 1146494 (10-30) | §1 the question; §2.1-2.4 the mathematics of the deliverable; Blocks 3-5, 10, 12-15, 21; §8 open points | all |
| `docs/project/TEEG_decisions_and_ideas_log.md` (root) | branch (newer than project knowledge, see `docs/project/BRANCH_COPIES.md`) | e52a073 (10-30) | Decisions D-0xx, ideas I-00x | all |
| `specs/roadmap/diameter-pipeline.yaml` / `.md` (root) | branch | 5015806 (11-02) | The theory graph, statuses and notes | all |
| `$D/README.md` | main 81b3327 (10-12) / branch 779bec0 (10-21) | -- | Folder index. Both copies are stale for later documents | -- |

Index the decision log with `git show origin/sci/diameter-pipeline:docs/project/TEEG_decisions_and_ideas_log.md | grep -n '^## [DI]-'` (986 KB; read entries by line range).

### 2.3 Code that defines the theory's numbers (read only where a document is unclear)

The source tree is `$D/src/allen_diameter/` on the branch:
- `config.py`: kernel table, slope, absorption mode (l.186-192); design ranges; flags.
- `model/blur_kernels.py`: the SGT family (`sgt_kernel_pdf`, l.260).
- `model/kernel.py`: the law `sigma_r`.
- `model/render.py` and `model/hybrid.py`: rendering.
- `analysis/fit_sgt.py`: the fit.
- `analysis/calibration.py` and `analysis/far_field.py`: the kernel calibration.
- `analysis/table.py`: the estimator.
- `analysis/invert.py`: the inversion. Its `Inversion` record carries `d_tilde_um`, `d_root_um`, `b_hat` and `flags`, and **no SE**.

### 2.4 Material outside the repository (ask the user)

| Material | Where | Topic |
|---|---|---|
| `TEEG_diameter_table_SE_inversion_theory_handoff_2026-10-22.md` and `checks/table_theory_2026-10-22/`. Committed locally, never pushed (log l.117) | Chat "Bias-table diameter workstream theory", https://claude.ai/chat/e0ee7cf4-cc5f-4494-8776-b496241c8f71 | T4.2 (its §5-6.1), T4.3 (its §6.2, items 5a-5g) |
| `TEEG_diameter_table_readout_inversion_study_handoff_2026-10-27.md` and `table_theory_2026-10-27_full.zip`. Sent as files; project knowledge was full | Chat "Stage 4: SE calculation with cubic spline outdated", https://claude.ai/chat/59f4166f-d58b-4e95-be4e-c2bbb406303d | T4.2 (open items 4c-4f), T4.3 |
| The tensor-cubic read-out and the inversion, explained 2026-10-28 to 10-31. Checks `tensor_cubic_2026-10-28` (19 of 19 pass) and `inversion_2026-10-28` (9 of 10; the band predicted for mathematics Eq. 21 failed) | Chat "Tensor product cubic spline interpolation theory", https://claude.ai/chat/93921147-38df-4dcf-9a25-677ffe8f3609 | T4.3 |
| The S2b answers: "how I would proceed" (10-21; figures `fig1_far_line`, `fig2_far_fit`); the Statistic step (10-22; `fig1_twopass`, `fig2_selection`); the three roles of γ, c, s in F2. Also `s2b_checks_2026-10-21.zip` | Chat "Diameter bias-table implementation", https://claude.ai/chat/51e6efd7-0068-4ef5-9a90-67a3e8857db5 | T6 |
| The weak-object answers (2026-10-08); the sine-condition document's scripts (`check01`-`check03`, `make_W_figures.py`, zip of 2026-10-12) | Chat "Weak object approximation theory", https://claude.ai/chat/21ef843b-230e-47d0-8779-02b46a2a7e1d | T7 |
| The kernel from a deep-learning view (idea I-003) | Chat "Deep learning defocusing kernel", https://claude.ai/chat/7e693fd6-0971-4a5c-913b-e23a7a6fa7cc | T6, T10 |
| Cell 7's outputs on the pilot: `width_by_plane_near/far.png`, `kernel_law.png`, `example_near/far.png`, the CSVs | The user's Drive, `MyDrive/diameters/calibration/` | T6, T10 |
| Run-B tables (Cm by layer) | Project knowledge, `claude/TEEG_annual_report_2026_01_passive_properties.md` §6 | T9 |

---

## 3. The theory map and a talk order

### 3.1 The roadmap's theory graph (status at 5015806, `roadmap.py status`)

| Id | Title | Status | Covered in |
|---|---|---|---|
| TG1 | Image formation | group, open | -- |
| T1 | Frames and objects | closed, **DRAFT** (set at the roadmap's start, not by the user; yaml l.88) | §5 |
| T3 | Phantom geometry and rendering | closed, **DRAFT** (l.106) | §7 |
| T7 | Optics of the hybrid plane absorption and the defocus kernel | **open** | §6 |
| TG2 | Measuring a diameter at a node | group, closed | -- |
| T2 | Per-node measurement | closed, **DRAFT** (l.97; its `decision: D-030` is ambiguous, see 12.2) | §8 |
| T5 | Skewed generalized-t kernel and the plane fit | closed, **DRAFT** (l.151) | §9 |
| TG3 | Correction and calibration | group, open | -- |
| T4.1 | The table's design | closed (2026-10-21, D-071) | §11.1 |
| T4.2 | Step 4: the grid value and its SE | closed (2026-10-27); items 4c-4f left open | §11.2 |
| T4.3 | Step 5: the cubic read-out and the inversion | **studying** (the read-out is studied; the inversion remains) | §11.3 |
| T6 | Far-field calibration | closed (2026-10-26) | §10.2 |
| T8 | Uncertainty and flags of the corrected diameter | **open** | §11.4 |
| T10 | Kernel calibration, step 2: the width-diameter trade-off | **open** | §10.3 |
| T9 | Membrane area and the Cm refit | **open**, no group | §4 |

The near-field kernel calibration (Block 10, procedure Eq. 4) has no topic of its own; the roadmap files it under T3 (§10.1).

### 3.2 Proposed talk order (a storyline; the presentation chat decides)

| Part | Content | Topics | Figures (section 14) |
|---|---|---|---|
| 0 | The question: Cm comes out 2-3 times too high; are Allen's radii too thin? | T9 | to make: F-T9a, F-T9b |
| 1 | What is measured: a tube's width read along ŷ, and the forward model of its profile | T1, T2 | B1, B2 |
| 2 | How the image forms: microscope and Koehler, pupil and spot, defocus, coherence, sine condition, mountant | T7 | R1, R3, A6, A1, A2, R11, A3, A4, A5; backup R2, R4-R9, R12, R13 |
| 3 | Rendering phantoms: slabs, partition, ray world, hybrid; the defocus kernel | T3, T7 | R14, R10, B4 (a); to make: F-T7a, F-T7b |
| 4 | Measuring a node: focus rule, the Allen-guided plane, the SGT fit, its identifiability limits | T2, T5 | B3; to make: F-T5a, F-T5b |
| 5 | Calibrating the kernel on Allen's stacks: near field, far field, step 2 | Block 10, T6, T10 | B4; to make: F-T6a, F-T6b |
| 6 | The correction table and its uncertainty: design, local linear estimate, SE, read-out, inversion, flags | T4.1-T4.3, T8 | B5; to make: F-T4a, F-T4b, F-T8a |
| 7 | Back to the question: what the corrected area does to Cm; what else could cause the excess | T9 | to make: F-T9c |

---

## 4. T9 -- the question: membrane area and the Cm refit (open)

**Message.** The passive fits put human L2/L3 Cm 2-3 times above the reference. If Allen's SWC radii are too thin, the membrane area is too small, and the fit compensates with Cm. The diameter pipeline exists to test this.

**Numbers**

| Value | Meaning | Source |
|---|---|---|
| 1.478 µF cm⁻² (n 10), 1.062 (n 18) | Run B (free Ra) excitatory medians, L2 and L3 (also L4 0.785, L5 0.797, L6 1.100) | [KB] `claude/TEEG_annual_report_2026_01_passive_properties.md` §6 |
| 0.43-0.52 µF cm⁻² | Model-fitted Cm of six human L2/3 PCs | [KB, full text] Eyal et al. 2016, *eLife* 5:e16553, Fig. 1. The annual report quotes 0.47 ± 0.03 |
| 0.47 ± 0.15 and 0.51 ± 0.11 µF cm⁻² | Same paper, nucleated-patch estimates (Methods) | [KB, full text] |
| L2 about 3×, L3 about 2.3× | Run B against 0.47 | [KB] annual report §6 |
| Cm at its 3.0 bound | Pilot specimen 529878215 (L3 excitatory) | [doc] `handoff_diameter_remeasurement.md` l.34 |
| 205 of 337 nodes in S have d̂ > 1 µm, while Allen's median 2r is ≈ 0.5 µm in every d̂ bin | First, uncorrected evidence on the pilot (Gaussian fit at k*, before D-052) | [log] l.4012-4013, data read |

The reference value differs by method: say on the slide which one is used.

**Equations**

$$A_{\rm seg} = \pi\,(r_1 + r_2)\sqrt{h^2 + (r_1 - r_2)^2}$$

Reads: the lateral area of the frustum between a dendrite node of radius $r_1$ and its parent of radius $r_2$, at distance $h$ (µm). A dendrite node whose parent is the soma contributes a cylinder of its own radius. The area ratio is the area with $d_{\rm final}/2$ over the area with Allen's radii, the "direct test". Source: [doc] SPEC Block 8, l.535.

The refit's parameters are $\theta = (C_m, R_m, R_a, \bar g_h, \Delta v_h, \kappa_\tau)$ [log] D-083.1, l.7266-7267. No real cell gets the six-parameter fit before Stage 7's synthetic recovery has run on davinci (D-083.1).

**Concepts in order**
1. The excess and the hypothesis (SPEC §1, l.38-42).
2. The deliverable: an SWC with radius $d_{\rm final}/2$ on types 3 and 4, all else byte-identical (D-013, SPEC l.44-47).
3. The area ratio.
4. Into the I_h fitter via `--swc-dir` (SPEC l.49-50).
5. What else could produce the excess:
   - I_h absorbed into the passive skeleton, which is the direction the synthetic benchmark predicts [KB] annual report §6 and §4.2;
   - path length and z-shrinkage (Allen's z factor k ≈ 2.4, unverified) [doc] handoff l.260, l.288.
   - [reasoning] Only the corrected area can separate these.

**Open.**
- T9 is open and step S9 has not started.
- No repository document derives how a given area error maps into the fitted Cm [open].

**Figures.** None exist. To make: F-T9a, F-T9b, F-T9c (section 15).

---

## 5. T1 -- frames and objects (closed, DRAFT)

**Message.** Near a node the dendrite is a straight round tube. Its diameter is the width of its section along ŷ, the in-plane normal to the branch's projection. That width is d for every heading and tilt; any other line over-reads.

**Equations** (SPEC §2.1, l.61-65 = handoff Math reference Eq. 5; handoff Eqs. 6-8, l.195-211, written in Unicode there)

$$\hat t_i = (\cos\varphi_i\cos\theta_i,\ \cos\varphi_i\sin\theta_i,\ \sin\varphi_i),\quad \theta_i = \operatorname{atan2}(t_y, t_x) \in (-\pi, \pi],\quad \varphi_i = \arcsin|t_z| \in [0, \pi/2],\quad \hat y_i = (-\sin\theta_i, \cos\theta_i, 0)$$

Reads: node $i$'s axis $\hat t_i$, from its heading $\theta_i$ in the image and its tilt $\varphi_i$ out of the image plane. The measuring axis $\hat y_i$ is the in-plane normal to the projection. $(x, y)$ are in µm along the pixel columns and rows, and $z$ in µm of stage travel.

$$(u\sin\varphi)^2 + v^2 \le r^2$$

Reads: in the focal plane through the centre, with $u$ along $\hat e_u$ (the projection) and $v$ along $\hat y$, the tube's section is an ellipse with semi-axes $r/\sin\varphi$ and $r$. On $u = 0$ the width is $2r = d$ for every $\theta$ and $\varphi$ (Eq. 7).

$$\frac{\text{half-chord}}{r} = \frac{1}{\sqrt{\cos^2\alpha\,\sin^2\varphi + \sin^2\alpha}}$$

Reads: a line through the centre at angle $\alpha$ to $\hat e_u$ over-reads unless $\alpha = 90°$. A pixel row has $\alpha = \theta$.

**Numbers**
- 1.265 d for a pixel row at θ = 45°, φ = 30°. A flat branch measured on a line misaligned by ε reads d/cos ε: +1.5 / +6 / +15 % at 10 / 20 / 30° [doc] handoff l.210-211; [run here] G1-G2.
- Pixel 0.1144 µm; plane step Δz 0.28 µm (stage units); voxel anisotropy 2.45 [doc] handoff l.43-44, l.129, values from the Allen API, not re-queried.
- D6: everything is computed in µm in the global frame, never in pixel or plane units [doc] handoff l.129.

**Superseded**
- The handoff's "width in the sharpest plane k*" (D2): the plane is now the Allen-guided k^it (D-052.3).
- The handoff's steep-branch flag as a selection rule: now a label only, and every node is estimated at any tilt (D-024 (iv), log l.2743).
- α fitted: μ is now fitted, with α = μd/cos φ derived (D-019, log l.2426).

**Open.** Round against flattened sections, untested (handoff l.285). Stage z equals tissue z only if the media are index-matched (procedure l.695-698; see 6.6).

**Figures.** B1 (new): (a) the ellipse with the ŷ chord and a pixel-row chord; (b) chord/d against misalignment for φ = 0 to 90°.

---

## 6. T7 -- optics of the microscope, the renderers and the hybrid (open)

Fixed setting, unless stated:
- Allen's upright transmitted brightfield: Plan-Apo 63×/1.4 oil, oil condenser of NA up to 1.4, Koehler illumination;
- model wavelength λ = 0.55 µm, oil index n = 1.515 (the code's value, "from memory"; Immersol 518 F has n_e 1.518);
- s_o = NA_o/n = 0.924 [doc] optics §3.1, l.194-216, l.247-252.

### 6.1 Light path, Koehler illumination, numerical aperture

**Concepts**
- The light path: lamp, collector, field diaphragm, aperture diaphragm (the condenser's front focal plane), condenser, specimen, objective, tube lens, camera.
- Two sets of conjugate planes:
  - the field set: field diaphragm, specimen, camera;
  - the aperture set: lamp, aperture diaphragm, the objective's back focal plane.
- Under Koehler illumination each point of the aperture diaphragm becomes one plane wave across the whole field. Different points are mutually incoherent.
- Snell's law keeps $n\sin\vartheta$, so NA is the same in every flat layer. The oil removes the glass-air boundary at which steep rays would be totally reflected.

[doc] optics §3.1-3.3; sine doc §3.4.

$$\mathrm{NA} = n\sin\theta_{\rm obj}$$

Reads: the angular reach of a lens, $\theta_{\rm obj}$ in a medium of index $n$. At NA 1.4 in oil, $\theta_{\rm obj}$ = 67.5° [doc] optics Eq. 4, l.317; l.335.

Numbers:
- Pixel 4.54 µm/(63 × 0.63) = 0.1144 µm.
- Total internal reflection beyond 41.3° at a glass-air interface.
- ZEISS's visual default closes the aperture diaphragm by 20-25 %, so NA_c ≈ 1.05-1.12 and S = NA_c/NA_o ≈ 0.75-0.80. Allen does not report the setting.

[doc] optics l.209, l.302, l.646-661.

Figures: R1 (the microscope schematic, both conjugate sets, the cone at a specimen point F); R3 (one diaphragm point gives one plane wave).

### 6.2 Diffraction: the pupil function, the Debye integral, the in-focus spot

**Concepts**
- The objective accepts directions $\mathbf s$ (direction sines, $|\mathbf s| \le s_o$). The pupil function $P(\mathbf s) = A(\mathbf s)\,e^{i\Phi(\mathbf s)}$ says how much of each plane wave passes ($A$) and how late it arrives ($\Phi$, aberration and defocus).
- The point's image is the sum of these plane waves (the field); the camera records $|U|^2$.

[run here] the pupil-function figure and its checks T1-T4.

$$U(\mathbf x, \delta) = \iint_{|\mathbf s| \le s_o} P(\mathbf s)\,\exp\!\Big[i\,k n\,\big(\mathbf s\cdot\mathbf x + \sqrt{1 - |\mathbf s|^2}\,\delta\big)\Big]\,d^2\mathbf s, \qquad k = 2\pi/\lambda$$

Reads: the scalar Debye field at lateral position $\mathbf x$ (µm) and defocus $\delta$ (µm). It is one plane wave per pupil point, weighted by $P$. The intensity spot is $|U(\mathbf x, \delta)|^2$. Source: this chat's `optics_model.psf_field` (pack `scripts/`); the optics doc gives the plane-wave picture in §3.4.

$$\Lambda(\vartheta) = \frac{\lambda}{n\sin\vartheta}, \qquad \Lambda_{\min,I} = \frac{\lambda}{2\,\mathrm{NA}} = 0.196\ \mu\mathrm m$$

Reads: a plane wave tilted by $\vartheta$ ripples across the focal plane with period $\Lambda$. The finest intensity period is half the finest field period (Abbe's limit). Source: [doc] optics Eqs. 5b-5c, l.392, l.406.

Numbers (550 nm, NA 1.4, uniform pupil):

| Quantity | Value | Evidence |
|---|---|---|
| Airy first dark ring | 0.2396 µm (= 0.61 λ/NA) | [run here] D1, T2 |
| Airy FWHM | 0.2021 µm | [run here] |
| 50 %-energy radius | 0.1051 µm | [run here] |
| LSF core width (Debye) | 0.080 µm, the configured σ_r(0) | [doc] optics l.439-440 |
| Axial FWHM | 0.53 µm | [doc] optics l.449-453 |
| Pupil cases: NA 0.7; defocus δ = 0.3 µm; coma 2 rad | Peaks 0.25 / 0.43 / 0.60 of the ideal spot's; coma shifts the centroid by −0.12 µm | [run here] |

Rule R6 of the optics doc: forming a spot is not resolving two points. A diameter is an estimation problem, not a resolution limit [doc] optics l.411-435.

Superseded, all [corrected 2026-10-04]:
- "FWHM ≈ 0.24 µm": that value is the first dark ring.
- "narrowest feature ≈ λ/NA".
- Axial 0.85 µm, which is the paraxial first zero.

[doc] optics l.370-371, l.424-428, l.454-457.

Figures:
- A6: pupil cases and their spots.
- A1: pupil, then spot, then profile.
- R12: the interactive figure. It is a 2-D fan of waves, so its FWHM at NA 1.4 is 0.155 µm, not Airy's; say so if it is shown [doc] interactive spec l.29, l.161.
- R13: the tilted wave's ripple.

### 6.3 Defocus: the wave spot against the geometric cone

$$\Delta\phi_{\max}(\delta) = \frac{2\pi n}{\lambda}\,\delta\,\big(1 - \cos\theta_{\rm obj}\big)$$

Reads: the phase lag of the edge ray against the axial ray at defocus $\delta$. It reaches π/2 (the quarter-wave tolerance) at |δ| = 0.147 µm [doc] optics Eq. 6a, l.544-550; renderers Eqs. 9-10.

Numbers, all [run here], D2-D4, D3b:
- At the quarter-wave defocus 0.147 µm:
  - the core FWHM grows by only +1.3 % (0.2047 µm);
  - the peak keeps 0.82;
  - but the 50 %-energy radius grows 16 % (0.1221 µm), as light moves into the rings.
- So "the spot is unchanged within the quarter-wave tolerance" holds for the core only (claim D3 failed; D3b passes).
- The geometric 50 %-energy radius equals the in-focus wave value at δ = 0.1217 µm.
- At 0.84 µm (three planes) the wave and geometric radii agree within 6 % (0.685 against 0.725 µm).
- The core is defined up to |δ| = 0.42 µm and merges with the first ring beyond.

Ideal-Debye core width σ (µm) against δ:

| δ (µm) | 0 | 0.14 | 0.28 | 0.42 | 0.56 | 0.84 |
|---|---|---|---|---|---|---|
| σ (µm) | 0.080 | 0.086 | 0.122 | 0.262 | 0.438 | 0.603 |

This is the configured renderer table [doc] optics l.559-561; [src] config l.187-188.

The geometric RMS slope $\gamma_\omega$ in $\sigma_{\rm geo} = \gamma_\omega|\delta|$ depends on the angular weighting [doc] mathematics Eq. 17, l.515, table l.520-524:
- 1.21 for a uniform disc;
- 0.90 for an isotropic cone;
- 0.79 for the cos-weighted cone, the configured γ.

Superseded: "σ ≈ 1.2|δ|" [corrected 2026-10-04] (optics l.530-534).

Figures: A2 (spot through focus with the cone; widths; centre brightness); R11 (interactive cone, disc and LSF; it draws a uniform disc, so its RMS readout is an upper bound).

### 6.4 Coherence: the condenser setting, Abbe's sum, first-order transfer, phase

**Concepts**
- Each condenser point is an independent emitter. The image is the sum of the coherent images, one per source point (Abbe):
  $$I_S = B\,\big\langle |U_{\boldsymbol\nu_c}|^2 \big\rangle_W$$
  with $U_{\boldsymbol\nu_c}$ the coherent image under the plane wave of transverse frequency $\boldsymbol\nu_c$.
- How alike the light is at two points is the van Cittert-Zernike degree of coherence:
  $$\Gamma(\Delta\mathbf x) = \big\langle e^{i2\pi\boldsymbol\nu_c\cdot\Delta\mathbf x}\big\rangle_W = \frac{2J_1(\beta_c)}{\beta_c}, \qquad \beta_c = \frac{2\pi\,\mathrm{NA}_c\,|\Delta\mathbf x|}{\lambda}$$
  Its first zero is at 0.24 µm for S = 1 and 0.32 µm for S = 0.75 [run here] C1.
- First-order (weak-object) transfer of an absorber, in focus: $H_S(\boldsymbol\nu)$ is the fraction of the condenser disc (radius $S\,\mathrm{NA}_o/\lambda$) that stays inside the pupil (radius $\mathrm{NA}_o/\lambda$) when shifted by $\boldsymbol\nu$. It equals the incoherent OTF only at S = 1. Its cutoff is $(1+S)\,\mathrm{NA}_o/\lambda$ [run here] C2.
- The renderers' model is $I \approx B\,(T * h_0)$, with $T = |t|^2$ and $h_0 = |h_a|^2$ (square, then blur). The microscope blurs the field, then squares: $B\,|t * h_a|^2$ per direction.

[doc] study notes l.302.

$$I_{\rm model}(\mathbf x)-I_{\rm true}(\mathbf x)=\frac{B}{\int|h_a|^2}\int_{\mathbb R^2\setminus\Omega_{\rm obj}}\big|\big((w\,e_{\mathbf q})*h_a\big)(\mathbf x)\big|^2\,d^2\mathbf q\ \ge0$$

Reads: at S = 1, thin object, $w = 1 - t$ the removed field, $e_{\mathbf q}$ one illuminating plane wave of transverse frequency $\mathbf q$, $\Omega_{\rm obj}$ the objective's aperture. The incoherent model is too bright everywhere, by the stain's dark-field image from the directions the condenser does not supply. There is no constant or linear term in $w$, so the model is exact to first order. Source: [doc] theory handoff l.224, [reasoning, not run] there; reproduced numerically here [run here] C5 (residual 2.6 % and 1.2 % of the peak difference at μd = 1, 3).

Numbers [run here] C3-C6, tube of 1 µm, in focus, centre I/B:

| μd | incoherent | S = 1 | S = 0.75 | S → 0 | max(I_inc − I_S=1) |
|---|---|---|---|---|---|
| 1 | 0.412 | 0.408 | 0.362 | 0.380 | 0.0088 |
| 3 | 0.106 | 0.089 | 0.045 | 0.059 | 0.044 |
| 10 | 0.052 | 0.025 | 0.002 | 0.000 | 0.120 |

Findings of this chat's checks:
- **The incoherent model is first-order exact only at S = 1.** At S = 0.75 the difference is first order in μ [run here] C3.
- **Phase:** a weak phase tube shows no first-order contrast in focus, and at S = 1 none off focus either. At S = 0.75 and δ = 0.28 µm the phase contrast is first order: 0.0146 for 0.1 rad, against 0.0007 in focus [run here] C6b. This agrees with the repository's reading of Zuo et al. 2017 (*Sci Rep* 7:7654, PMC5550517, full text read by the theory chat on 2026-10-08): at S ≥ 1 the weak-object transfer is the pupil autocorrelation, with no phase transfer [doc] theory handoff l.219; optics l.927.
- The coherent case gives a bright fringe of +0.04 to +0.11 B [doc] theory handoff l.239-252.

Why it matters:
- Allen's condenser setting is unknown [open].
- Coherence is outside every renderer [doc] renderers l.892-896.
- The fringe check detects S < 1, not darkness at S = 1 [corrected 2026-10-08] [doc] optics l.673-678.
- The effect on d̂ is not computed [open].

Figures: A3 (Koehler: each emitter is one coherent image; Γ for S = 1 and 0.75); A4 (H_S for S = 0, 0.5, 0.75, 1); A5 (tube images at three condenser settings; the difference equals the dark-field image).

### 6.5 The sine condition and the ray world's measure W

$$h = n_{\rm oil}\,f\sin\vartheta, \qquad \mathbf s = -\frac{\mathbf h}{n_{\rm oil}f}, \qquad s_m = \frac{\mathrm{NA}_c}{n_{\rm oil}} = \sin\vartheta_{\max}$$

Reads:
- An aplanatic condenser (no spherical aberration, no coma) maps the height $h$ of a point in the aperture diaphragm to the sine of its ray's angle $\vartheta$ at the specimen. $f$ is the condenser's focal length.
- A diaphragm point $\mathbf h$ becomes the direction sines $\mathbf s$ (a rescaling plus an inversion).
- The lit disc of sines has radius $s_m$.

Source: [doc] sine doc Eqs. 2, 5, 6, l.331, l.481, l.492.

$$dW=\frac{ds_x\,ds_y}{\pi s_m^2}=\frac{\cos\vartheta\,d\Omega}{\pi s_m^2}\ \ (|\mathbf s|\le s_m), \qquad dW^*=\frac{d\Omega}{2\pi\,(1-\cos\vartheta_{\max})}\ \ (\vartheta\le\vartheta_{\max})$$

Reads:
- An evenly lit diaphragm gives a measure $W$ uniform in the direction sines, which is $\cos\vartheta$ per solid angle $d\Omega$.
- $W^*$ is the isotropic cone.
- The $\cos\vartheta$ is the obliquity of the specimen plane: equal radiance seen through an oblique plane.

Source: [doc] sine doc Eqs. 8-9, l.545, l.586.

$$\Big\langle\frac{1}{\cos\vartheta}\Big\rangle_W =\frac{2}{s_m^2}\Big(1-\sqrt{1-s_m^2}\Big)=1.447$$

Reads: the mean path per unit depth over the NA 1.4 cone. It is the factor by which faint absorption is booked in the ray world and the hybrid ($\mu^{(1)}_{\rm ph} = 1.447\,\mu$, renderers Eq. 6). Source: [doc] sine doc Eq. 10, l.589-590.

Numbers [doc] sine doc l.268-271, l.364-366, l.551-552, l.644-648:
- tan/sin = 2.62 at 67.5°.
- W's weight per steradian at the edge is 0.382 of its value on the axis.
- Median angle: 40.8° under W, 46.3° under W*.
- Slopes: γ_W = 0.7915 and γ_W* = 0.8991.
- At NA_c 1.12 / 1.05: s_m = 0.739 / 0.693, γ_W = 0.473 / 0.426.

Do not conflate the configured γ = 0.79 (D-024 (i)) with γ_W, even though the numbers coincide [doc] sine doc l.717-723.

Open:
- Whether Allen's condenser is aplanatic (H1) [Guessing in the source].
- How far the cone varies across the field (H2).
- The real direction distribution [open].

Figures:
- R2: one ray, from h to s to ŝ.
- R4: the sine rule against the tangent rule.
- R8: why a sine.
- R5: the Jacobian, map from diaphragm to directions.
- R7: obliquity.
- R6: W against W*.
- R1, R3 (6.1).
- R1, R7 and R8 are schematic [doc] sine doc l.906-908.

### 6.6 The mounting medium and index mismatch

**Concepts**
- The mounting medium of 529878215 is not verified; the Allen white paper names Mowiol, with no index.
- If $n_{\rm med} < n_{\rm oil}$:
  - rays steepen in the medium;
  - the cone loses its edge, since directions with $n_{\rm oil}|\mathbf s| > n_{\rm med}$ are reflected;
  - rays no longer meet at one point;
  - the focus shifts, with depth-dependent spherical aberration and an above/below asymmetry.

[doc] optics §3.7, l.686-755; sine doc §3.10, Eq. 13, l.782-784.

Numbers: at an illustrative n_med = 1.33, directions with |s| > 0.878 are reflected (the cone's edge 0.924 is lost), and rays cross the axis at 0.70-0.87 of the nominal depth [doc] sine doc l.785-796.

Open: the focal-shift ratio at NA 1.4 is not computed; the fluorescence literature's transfer to brightfield is not checked [doc] optics l.827-830.

Figure: R9.

### 6.7 Three renderers: the partition, the ray world, the hybrid

The partition (the default today):

$$T_{<j}(\mathbf x)=\exp\Big(-\sum_{j'<j}a_{j'}(\mathbf x)\Big),\quad \Delta A_j(\mathbf x)=T_{<j}(\mathbf x)\,\big(1-e^{-a_j(\mathbf x)}\big), \qquad I^{\rm P}_k(\mathbf x)=B\Big[1-\sum_{j=1}^{J}\big(\Delta A_j*K_{\zeta_j-z_k}\big)(\mathbf x)\Big]$$

Reads:
- $a_j$ is slab $j$'s absorbance along the vertical, and $T_{<j}$ the light reaching slab $j$ from below.
- $\Delta A_j$ is the light slab $j$ removes.
- Plane $k$ at depth $z_k$ is the background $B$ minus every slab's removed light, blurred by the kernel of the slab's own defocus $\zeta_j - z_k$.
- It conserves light per column and is exact for one slab and in faint stain.

Source: [doc] renderers Eqs. 1-2, l.335, l.339; = procedure Eqs. 5-6.

The ray world (geometric reference):

$$I^{\rm ray}_k(\mathbf x)=B\,\big\langle e^{-\mu L(\mathbf x,z_k,\hat s)}\big\rangle_W$$

Reads: Beer-Lambert along every ray of the cone, with $L$ the ray's path in the tube, averaged under $W$, then a 0.08 µm post-blur. There is no diffraction and no kernel. Source: [doc] renderers Eq. 3, l.370.

The hybrid (built as SPEC Block 21; D-084, D-092; not the default):

$$\Delta A^{\rm hyb}_j(\mathbf x)=\delta\zeta\,\Big\langle\frac{\mu}{\cos\vartheta}\,e^{-\mu\,\ell_{\rm back}(\mathbf x,\zeta_j;\hat s)}\Big\rangle_W, \qquad I^{\rm hyb}_k(\mathbf x)=B\Big[1-\sum_j\big(\Delta A^{\rm hyb}_j*K_{\zeta_j-z_k}\big)(\mathbf x)\Big]$$

Reads:
- Each ray brings the light it has left after its own path $\ell_{\rm back}$ in the tube before the point. It loses $\mu$ per unit path, with $1/\cos\vartheta$ of path per unit depth.
- The absorbed light is averaged over the cone, then every plane is rendered with the kernel as before.

Source: [doc] renderers Eqs. 13-14, l.739, l.745. The code uses the slab's exact absorbance times the cone average at the middle of the slab's inside part (SPEC §2.3 (H), l.230; PROVISIONAL, D-092.2).

$$\sigma^2_{K_\delta}=\sigma^2_{\rm cone}(\delta)+\sigma^2_\kappa\ \ge\ \sigma^2_{\rm cone}(\delta)$$

Reads: a kernel that keeps each ray's own landing point must be at least as wide as the cone. The production kernel is narrower than the cone beyond |δ| ≈ 0.09-0.11 µm, so the direction average is forced there. Source: [doc] renderers Eq. 15, l.788, l.790-796.

Numbers:
- Partition minus ray world, d̂/d at d = 0.5 / 1 / 2 / 3 µm: −0.25 / −0.11 / −0.05 / −0.04. The ray world is 2.2× wider than the kernel at 0.28 µm and over-blurs near focus [doc] renderers l.593-639.
- Hybrid against the ray world, centre dip of a flat 1 µm tube at μd = 0.05 / 0.5 / 1.5 / 3 / 10 / 50 [doc] renderers l.828-832:
  - hybrid 0.057 / 0.450 / 0.845 / 0.961 / 0.811 / 0.732;
  - ray world 0.057 / 0.441 / 0.820 / 0.965 / 1.000 / 1.000.
- Block 21 findings [log] D-092.3, l.7749-7765:
  - the hybrid conserves light per direction but not per column: the centre column books 1.03× its own light at μd 1.5 and 1.18× at μd 3;
  - with the production Gaussian kernel the node plane's minimum of τ = I/B is −0.130 / −0.096 / −0.034 / −0.013 at μd 3 (d = 0.5 / 1 / 2 / 3 µm);
  - like for like, max |τ − τ_ray| is 0.060 / 0.227 for the hybrid against 0.195 / 0.365 for the partition at μd 1 / 3;
  - it costs 4-50 times the partition per phantom.

Current state:
- D-084 decides that the hybrid becomes the renderer actually used once built.
- D-084.2 keeps the partition as the default until the user switches (config l.192).
- [corrected] The renderers doc calls the hybrid "not yet logged" and "no decision yet" (l.7-8, l.989-991); D-084, D-092 and Block 21 supersede that.

Open (SPEC §8, l.918), none decided:
- (a) keep the partition, cap the phantoms' μd, derive a variant that conserves light per column, or clip at 0;
- (b) when to switch the default;
- (c) the cost;
- (d) the kernel under the hybrid in dark stain (6.8).

Figures: R10 (one oblique ray: history along the ray against along the vertical); R14 (one plane rendered from slabs, partition). To make: F-T7a, F-T7b (section 15).

### 6.8 The kernel question under the hybrid (D-084)

- If the hybrid's rays only change how much a slab absorbs, the calibrated σ_r(δ) stays the defocus blur. If they also spread a slab's shadow with defocus, the kernel must be refitted under the hybrid [log] D-084, l.7331-7334.
- Faint stain: the hybrid books 1.447 a_j uniformly. A kernel calibrated through the partition on faint nodes therefore carries over with μ̂ rescaled [reasoning, log l.7341].
- Block 10 on hybrid-rendered thin phantoms (d = 0.3 µm, α = 0.1 / 1 / 1.5) gives σ̂_r(0.84) = 0.599 / 0.603 / 0.603 [doc] `checks/d094_2026-10-28/dark_node_growth_hybrid.out` l.7-9 (branch). That is internal to the renderer; whether either renderer images a dark tube as the microscope does is T7's question [log] l.7926-7928.
- Dark stain [open].

---

## 7. T3 -- phantom geometry and rendering (closed, DRAFT)

**Message.** A phantom is a round straight tube cut into thin slabs. Each image plane is rendered from the object, never by blurring other planes: each slab is blurred once, by the kernel of its own defocus. Then comes the camera chain.

**Equations** (SPEC §2.3, l.172-193; implementation handoff S1-S5, l.247-257)

$$u = (x - c_x)\cos\theta + (y - c_y)\sin\theta,\quad v = -(x - c_x)\sin\theta + (y - c_y)\cos\theta,\quad w = z - c_z;\qquad v^2 + (w\cos\varphi - u\sin\varphi)^2 \le r^2$$

Reads: the tube frame about the centre $c$, and the inside test (S1-S2).

$$z_{\rm lo/hi} = c_z + u\tan\varphi \mp \ell(v)/2,\qquad \ell(v) = 2\sqrt{r^2 - v^2}/\cos\varphi$$

Reads: each vertical line meets the tube on one interval (S3).

$$a_j(x, y) = \mu\max\{0, \min(z_{\rm hi}, \zeta_j + \tfrac{\delta\zeta}{2}) - \max(z_{\rm lo}, \zeta_j - \tfrac{\delta\zeta}{2})\}$$

Reads: slab $j$ (centre $\zeta_j$, thickness $\delta\zeta$ = 0.02 µm) absorbs $\mu$ times its chord (S4).

$$\Delta A_j = T_{<j}\,(1 - e^{-a_j}), \qquad I_k(x, y) = B\Big[1 - \sum_j \big(\Delta A_j * G_{\sigma_{\rm r}(\zeta_j - z_k)}\big)(x, y)\Big]$$

Reads: the partition (5-6) with a circular Gaussian $G_\sigma$ of SD $\sigma_{\rm r}(\delta)$. It is evaluated by FFT on a fine grid ($p_x/8$), then pixel integration, grey mapping, noise, 8-bit quantisation and JPEG with Allen's tables (SPEC l.458-461).

$$\sigma_{\rm r}(\delta) = \begin{cases} \text{the linear interpolation of } (t_m, \varsigma_m)_{m=0}^{M} \text{ at } \lvert\delta\rvert, & \lvert\delta\rvert \le t_M,\\ \varsigma_M + \gamma\,(\lvert\delta\rvert - t_M), & \lvert\delta\rvert > t_M \end{cases}$$

Reads: the kernel law. A table in $|\delta|$ up to $t_M$ = 0.84 µm, then linear with slope $\gamma$ from the anchor (D-080.1). Source: [doc] DL handoff §4.2, l.115.

**Numbers**
- Configured knots at δ = 0 / 0.14 / 0.28 / 0.42 / 0.56 / 0.84 µm: σ = 0.080 / 0.086 / 0.122 / 0.262 / 0.438 / 0.603 µm, γ = 0.79, marked "NOT calibrated" [src] config l.186-191; [run here] L1.
- U = 10 µm half length; δζ = 0.02 µm; axial end cut; squash k = 1 [src] config l.193-198.
- Linear sum against the partition, d = 0.5 µm: +1.1 / +14.4 / +85.8 % at μd 0.025 / 0.30 / 1.50. The linear sum is wrong in dark stain [doc] mathematics l.412-414.

**Current vs superseded**
- **Kernel family.** D-066 (2026-10-20, 16:11) put the SGT family into the renderer. D-067.2 withdrew it 30 minutes later, before any code [log] l.6197-6199, l.6226-6235. The renderer's kernel is a circular Gaussian (`gaussian_table`). Only the fit uses the SGT (T5). D-066's Abel result (every symmetric SGT member is the projection of a non-negative isotropic PSF) stays on record. I-002 is back to an idea.
- **Absorption.** Partition by default; hybrid built (6.7).
- **Continuation.** It used to start at the last knot; it now starts at the anchor 0.84 µm (D-080.1).
- **Calibration nodes.** The presentation-ideas doc's node criteria (d̂ ≲ 0.3 µm, faint, φ ≲ 10°) are out of date; see 10.1.

**Figures.** R14 (one plane from slabs, partition; illustrative widths); R10; B4 (a) (the configured table). Never made: presentation-ideas Idea 1, "the kernel in three domains", panels A-C (ideas l.80-119). B4 covers its panel B with today's numbers.

---

## 8. T2 -- the per-node measurement (closed, DRAFT)

**Message.** At each node a profile is sampled along ŷ, a plane is chosen, the background is fixed from the image, and a blurred Beer-Lambert tube is fitted. The fitted diameter is biased, and the table corrects it later.

**Concepts in order**
1. Profile: bilinear samples over ±3 µm in steps of the pixel, N = 2 round(3/0.1144) + 1 = 53, averaged over ±0.5 µm along the branch [doc] SPEC l.114-116; 10-16 handoff l.78.
2. Focus: the gradient energy of the profile picks the plane (D-035; its repository name is "D-030", see 12.2).
3. The pilot (Block 5): k*, sub-plane depth, a Gaussian centre, and a total-least-squares line over ±2 µm that gives θ, φ and ŷ.
4. Background fixed, not fitted (D-018, D-023).
5. Tube model with μ fitted and φ from the line fit (D-019).
6. Blur. The pilot fit uses a Gaussian with σ_fit = 0.099 µm, the in-focus budget $\sigma_{\rm fit}^2 \approx \sigma_{\rm r}(0)^2 + p_x^2/4$. The final fit uses the SGT kernel (T5).
7. Squared widths add, so thin tubes are nearly degenerate in (d, σ).
8. Selection S fails a node on:
   - its fit status;
   - `crossing`, `stack_edge`, `bbar_few`, `profile_nan`;
   - registration.

   `dark`, `faint`, `steep`, `vertical` and `iter_*` are labels only (D-030, D-096, D-024 (iv), D-068).

**Equations**

$$G_{j,k} = \frac{1}{B_{j,k}^2}\int_{|v| \le h_j}\Big(\frac{\partial\tilde I_{j,k}}{\partial v}(v)\Big)^2\,dv, \qquad k^*_j = \operatorname*{arg\,max}_k G_{j,k}$$

Reads:
- The edge energy of node $j$'s smoothed profile $\tilde I_{j,k}$ in plane $k$, over a window $h_j$ = radius + 0.5 µm, normalised by the plane's background $B_{j,k}$.
- At fixed shadow area $A$ a Gaussian dip of width $w$ gives $A^2/(4\sqrt\pi\,w^3)$, so $G$ peaks at focus.

Source: [doc] SPEC §2.2 (1), l.78, l.86-93.

$$T_{d,\mu,v_0}(v \mid \varphi_i) = \exp\!\Big(-\frac{\mu\,d}{\cos\varphi_i}\,s_d(v - v_0)\Big), \qquad s_d(v) = \sqrt{1 - (2v/d)^2}\ \ (|v| \le d/2)$$

Reads: Beer-Lambert across the tube along the vertical path $\ell(v) = d\,s_d(v)/\cos\varphi$; $\alpha = \mu d/\cos\varphi$ is the centre's absorbance. Source: [log] D-019.1, l.2444-2449; mathematics Eq. 8, l.246.

$$I_{\rm model}(v \mid d, \alpha, v_0, \sigma, \bar B) = \bar B\Big[1 - \frac{d}{2}\int_{-\pi/2}^{\pi/2}\big(1 - e^{-\alpha\cos\psi}\big)\cos\psi\; g_\sigma\big(v - v_0 - \tfrac{d}{2}\sin\psi\big)\,d\psi\Big]$$

Reads: the blurred absorbed fraction, $\bar B[1 - ((1 - T) * g_\sigma)(v)]$, written with $u = (d/2)\sin\psi$ so that the edge singularity disappears. Its dip area is exactly $\bar B\,\frac{\pi d}{2}(I_1(\alpha) - \mathbf L_1(\alpha))$ ($I_1$ the modified Bessel, $\mathbf L_1$ the modified Struve function). Source: [doc] SPEC (B3.1), l.433-438; [run here] P2b.

$$V[f * g] = V[f] + V[g], \qquad \bar v[f * g] = \bar v[f] + \bar v[g], \qquad A[f * g] = A[f]\,A[g]$$

Reads: under blur, normalised squared widths $V$, centres $\bar v$ and areas $A$ combine like this (finite moments only). $V_{\rm tube}$ runs from $d^2/16$ (faint, a semicircle) to $d^2/12$ (opaque, a box). Source: [doc] mathematics Eq. 15, l.444, l.460-466; [run here] P3.

**Numbers**
- [run here]: below d ≈ 0.34-0.40 µm the tube's own V is smaller than σ_fit², so the width trades against the diameter.
- Pilot run on 529878215 (uncorrected, Gaussian at k*): 393 nodes, 337 in S [log] l.3963-3964.
- The handoff's "0.24 µm" blur was wrong (it was the first dark ring) [log] l.2406-2418.

**Superseded**
- Focus by dip depth (handoff Eq. 1): now gradient energy (D-035, log l.3738).
- Background fitted (handoff Eq. 11): now fixed (D-018).
- α fitted: now μ.
- Gaussian σ bracketed: σ_fit is used only by the pilot and the SGT starts (D-052).
- Selection by α̂ ≤ 1 or by a bound hit: now only the rule of D-053.2.
- Stale text, not marked:
  - README "Method in one paragraph" (branch l.36-45);
  - SPEC §2.2 "Selection S", l.136-141, which still lists `faint`.

**Open**
- The bias of the median background from stained pixels (log l.2386-2393).
- The B̄ mask's scope (log l.6015-6017).
- Whether the vertical path d/cos φ holds for thick or steep branches (log l.2497-2500).
- D-052 (e): whether the pilot keeps the Gaussian (log l.4973-4974).

**Figures.** B2 (new): (a) T(v) for α = 0.3 / 3 / 30; (b) the (B3.1) profile for d = 0.3-2 µm with the 53 sample positions; (c) V against d with σ_fit². To make: F-T2a (focus scores through focus, section 15).

---

## 9. T5 -- the skewed generalized-t kernel and the plane fit (closed, DRAFT)

**Message.**
- A fixed Gaussian over-reads thin tubes, and a fitted-width Gaussian carries a shape bias on thick ones.
- So the fit's kernel is a flexible family of fixed FWHM w with shape κ = (β, ξ, ζ), fitted with the tube at the plane that an Allen-guided iteration picks.
- Thin tubes stay partly unidentifiable. That scatter is left to the table.

**Equations** (10-16 handoff §3.2-3.4, l.90-143 = SPEC Block 12; D-052.1-3)

$$h_{\beta,\xi}(x) = \frac{\beta\,\xi^{1/\beta}}{2\,B(1/\beta,\,1/\xi)}\,\bigl(1 + \xi\,\lvert x\rvert^{\beta}\bigr)^{-(1/\xi + 1/\beta)} \ \ (\xi > 0), \qquad h_{\beta,0}(x) = \frac{\beta}{2\,\Gamma(1/\beta)}\,e^{-\lvert x\rvert^{\beta}} \tag{H1}$$

Reads: the symmetric core. $\beta$ shapes the peak, and $\xi$ the tails, which fall as $|z|^{-(1+\beta/\xi)}$ ($B$ the beta function, $\Gamma$ the gamma function).

$$f(x \mid \kappa) = \frac{2}{\zeta + 1/\zeta}\begin{cases} h_{\beta,\xi}(x/\zeta), & x \ge 0,\\ h_{\beta,\xi}(\zeta x), & x < 0,\end{cases} \tag{H2}$$

Reads: the two-piece skew. Unit area, mode at 0, right-to-left half-width ratio $\zeta^2$.

$$W(\kappa) = \Bigl(\zeta + \frac1\zeta\Bigr)\Bigl(\frac{2^{\xi\beta/(\beta+\xi)} - 1}{\xi}\Bigr)^{1/\beta} \ (\xi > 0), \quad W(\beta, 0, \zeta) = \Bigl(\zeta + \frac1\zeta\Bigr)(\ln 2)^{1/\beta}, \qquad g_{w,\kappa}(z) = \frac{W(\kappa)}{w}\, f\Bigl(\frac{W(\kappa)}{w}\,z \Bigm| \kappa\Bigr) \tag{H3}$$

Reads: rescaled so that every member's FWHM is exactly $w$ (µm). Members:
- (2, 0, 1) Gaussian;
- (1, 0, 1) Laplace;
- (2, 2/ν, 1) Student $t_\nu$, so (2, 2, 1) is the Lorentzian;
- (β, 0, 1) generalized normal;
- ζ ≠ 1 their skewed versions.

$$I_{\rm model}(v \mid \theta;\, \varphi_j, \bar B_j(k)) = \bar B_j(k)\Bigl[1 - \frac d2 \int_{-\pi/2}^{\pi/2} \bigl(1 - e^{-\alpha\cos\psi}\bigr)\cos\psi\; g_{w,\kappa}\Bigl(v - v_0 - \frac d2 \sin\psi\Bigr)\, d\psi\Bigr] \tag{H4}$$

Reads: the T2 model with the SGT kernel. Seven parameters $\theta = (d, \mu, v_0, w, \beta, \xi, \zeta)$, with $\alpha = \mu d/\cos\varphi_j$.

$$Q_j(\theta \mid k, \omega) = \frac12 \sum_{n=1}^{N} \omega_n \Bigl[\frac{I_{\rm model}(v_n \mid \theta;\, \varphi_j, \bar B_j(k)) - I_j(v_n; k)}{\bar B_j(k)}\Bigr]^2, \qquad \omega^{(t)}_n = \exp\bigl(-\lvert v_n - \hat v_0^{(t)}\rvert / \lambda\bigr) \tag{H5}$$

Reads: least squares on the background-normalised profile. First unweighted, then re-weighted towards the fitted centre with $\lambda$ = 0.5 µm (IRLS), until $d$ and $v_0$ move by at most $10^{-4}$ µm, with at most 30 weighted solves. The result is a fixed point of that rule, not a joint minimum.

$$k_{j,i} = \arg\max_{k \in K_j} G^{(h_{j,i})}_{j,k}, \qquad G^{(h)}_{j,k} = B_{j,k}^{-2} \int_{-h}^{h} \Bigl(\frac{\partial \tilde I_{j}(v; k)}{\partial v}\Bigr)^2 dv, \qquad K_j = \{k_{{\rm SWC},j} - 6, \ldots, k_{{\rm SWC},j} + 6\} \tag{H6}$$

Reads: the Allen-guided plane.
- It starts from $d^{(0)}_j = 2r_j$ (Allen's radius) and uses lines of half-length $h_{j,i} = m\,d^{(i-1)}_j/2$, $m$ = 3.
- Each round's SGT fit sizes the next line. There are at most 5 rounds, and the iteration stops when a plane repeats.
- The result is $k^{\rm it}_j$ and $\hat\theta_j(k^{\rm it}_j)$.
- Phantoms and real nodes go through the same function (D-052.4, D-057.3).

**Numbers** [doc] 10-16 handoff §4 (sandbox Monte Carlo):
- 1.5 µm tube, t₄ kernel: d̂ = 1.493 ± 0.015 (SGT with IRLS), 1.530 ± 0.014 (Gaussian with fitted w), 1.639 ± 0.012 (Block 5) (l.185).
- 0.6 µm tube: 0.579 ± 0.182 and 0.539 ± 0.195 (Gaussian and t₄ kernels; 8 and 16 fits end at μ's bound), against 0.830 and 0.914 for Block 5 (l.187-188).
- Thin tubes have impostors: d trades against w roughly along $\sigma_w^2 + d^2/16$ = const. A larger family cannot fix this (log l.4365; 10-16 handoff §4.2).
- The box Θ is d ∈ [0.05, 6] µm, μ ∈ [0, 20] µm⁻¹, v₀ ∈ [−1, 1] µm, w ∈ [0.1, 2] µm, β ∈ [1, 10], ξ ∈ [0, 2], ζ ∈ [0.497, 2.014] (l.149-153).
- Bound rule of S (D-053.2): a fit fails only when d, μ, v₀ or w sits at a bound, or the IRLS does not converge.

**Open**
- Start points and the IRLS cap are PROVISIONAL (log l.4977-4978).
- Quadrature on the non-analytic members converges only algebraically (SPEC l.923).
- The thick, dark end (SPEC l.922).
- The real-data check of ŵ and κ̂ on thick nodes; no results in the documents read [open].
- The SGT literature references were never verified (10-16 handoff l.553-558).

**Figures.** B3 (new): the six named members at equal FWHM, peak-normalised and on a log scale. It is computed from (H1)-(H3) and matches the repository's `sgt_kernel_pdf` to 4e-16 of the peak [run here] K6. To make: F-T5a (identifiability, regenerable from `checks/estimator_2026-10-15/fig_sgt.py`), F-T5b (the iteration).

---

## 10. Calibrating the defocus kernel: near field (Block 10), far field (T6), step 2 (T10)

**Common message.**
- The kernel models the optics only; the camera chain comes after it.
- The condenser setting and the mountant are unknown, so σ_r(δ) is calibrated on Allen's own stacks:
  - near focus, from thin nodes scanned over three planes;
  - far from focus, from nodes of 0.6-0.8 µm, with a straight continuation.

**What no method can identify** [doc] DL handoff §5, l.188-219:
- an additive constant on the squared width;
- a common shift of δ;
- σ_r(0);
- knots finer than Δz;
- the far slope from one side of focus only;
- the kernel's shape beyond one statistic;
- width against diameter on thin nodes.

$$V_W[w_k] \approx V_{\rm tube} + V_W[\mathrm{LSF}_0] + \big(V_W[\mathrm{LSF}_{\delta_k}] - V_W[\mathrm{LSF}_0]\big)$$

Reads: a dip's (windowed) squared width = the tube's + the in-focus blur's + the growth with defocus. Only the growth varies with the plane. Real LSFs have $|v|^{-2}$ tails, so the statistic used is the Gaussian core width, not the second moment. Source: [doc] mathematics Eq. 16, l.503; l.471-484.

### 10.1 Near field (Block 10; roadmap: under T3)

$$\omega_{i,k} = c_i + \Delta\sigma^2(z_{{\rm ax}, i} - z_k) + \varepsilon_{i,k}, \qquad \Delta\sigma^2(0) = 0$$

Reads: node $i$'s core-width squared $\omega_{i,k}$ at plane $k$ is:
- a node constant $c_i$, which absorbs the tube and the in-focus blur, so σ_r(0) cancels;
- plus a growth $\Delta\sigma^2$ shared by all nodes, evaluated at the node's axis depth $z_{{\rm ax},i}$ (free within ±0.28 µm);
- plus noise.

Source: [doc] procedure Eq. 4, l.327 (branch) = mathematics Eq. 18, l.597-606.

$$\hat\sigma_{\rm r}(m\,\Delta z) = \big(\sigma_{\rm r}(0)^2 + \hat G(m\,\Delta z)\big)^{1/2}, \qquad \sigma_{\rm r}(0) = 0.080\ \mu\mathrm m\ \text{(configured)}$$

Reads: the table at the plane knots, from the fitted growth. Source: [doc] DL handoff §4.3, l.145; SPEC l.584-586.

Current node rules:
- selected by Allen's diameter, near $2r_A \le$ 0.6 µm (D-061.4);
- no faint criterion (D-094);
- flatness ≤ 10° (D-101.1);
- isolation by the stripe at τ = 0.3 gl (D-101, D-108);
- candidates from the whole cell's trace (D-109, D-112);
- 10-30 nodes per set, one kernel pooled across cells (D-088).

[log] l.5666, l.7876, l.8525, l.9048, l.9266, l.7515.

Pilot 529878215 (Cell 7, Colab, 2026-10-30), 30 near nodes scanned [log] l.139:
- σ_r = 0.080 / 0.143 / 0.384 / 0.547 µm at δ = 0 / 0.28 / 0.56 / 0.84;
- anchor SE 0.0438 µm with all 1000 resamples kept after D-115.4 (0.041 µm on 870 before);
- without three nodes whose dip is not one Gaussian core, the anchor is 0.484 µm.

[run here] L2-L3.

The method's own bias, from noise-free phantoms rendered with the configured kernel (α = 0.1): it reads low near focus [doc] `checks/d094_2026-10-28/dark_node_growth.out` (branch); [run here] L4.
- d = 0.3 µm: 0.092 / 0.389 / 0.597;
- d = 0.5 µm: 0.106 / 0.357 / 0.601;
- true values: 0.122 / 0.438 / 0.603.

Why: a tube spans δ ± r, and the core width of that mixture is narrower than its mean variance (SPEC l.595; mathematics l.635-646). That is step 2's subject (10.3).

Superseded:
- The procedure's node criteria (d̂ ≲ 0.3 µm, faint, φ ≲ 10°, isolated; l.320-321).
- "The same B̄ rule in every plane": now the stripe's planar background.
- "Never run on Allen's data".

[log] D-061.4, D-079, D-081, D-093, D-094, D-101, D-108, D-112.

### 10.2 Far field, S2b (T6, closed)

$$\sigma_{\rm r}(\delta) = \sigma_{\rm r}(0.84\ \mu{\rm m}) + \gamma\,\bigl(\lvert\delta\rvert - 0.84\ \mu{\rm m}\bigr)$$

Reads: beyond three planes the width grows linearly from the anchor with slope $\gamma$. Source: [log] D-026.1, l.2992-2993.

$$r_{j,k}(\theta) = c_j + \big(\hat\sigma_0 + \gamma\,x_{j,k}(s_j)\big)^2 - \hat\omega_{j,k}$$

Reads: the far fit's residual for node $j$ at plane $k$, where:
- $x_{j,k}$ is the plane's distance beyond the anchor, given the node's axis shift $s_j \in [-0.28, 0.28]$ µm;
- $\hat\sigma_0$ is Block 10's anchor, held fixed;
- $c_j$ is a node constant;
- $\hat\omega_{j,k}$ is the measured core width squared.

The fit is IRLS with model weights: one unweighted round, then 3 weighted rounds. The literal weighted objective F2 is a different, biased estimator. The SE is
$$\mathrm{SE}(\hat\gamma)^2 = \mathrm{SE}_{\rm boot}(\hat\gamma)^2 + (\partial\hat\gamma/\partial\sigma_0)^2\,\mathrm{SE}(\hat\sigma_0)^2$$
with intervals $\hat\gamma \pm t_{J-1,\,0.975}\,\mathrm{SE}$. Source: [doc] SPEC 15.4-15.5, l.688-689.

Design:
- Scan k* ± 16 planes (4.48 µm) on both sides, along a ±10 µm line, with the stripe background.
- Statistic: an iterated Gaussian core fit in a ±1 FWHM window, at most 10 passes, stopped at 1 SE (D-077, D-078).
- Planes with |k − k*| ≥ 5 enter when the *predicted* dip clears 3 RMS (D-074.1, D-078.3).
- At least 6 planes per node.

[doc] SPEC 15.1-15.3, l.685-687; [run here] L5.

Simulation evidence [doc] DL handoff §6, l.221-243, sandbox, J = 12 far nodes, 1 gl noise, 200 replicates per γ:

| Variant | γ = 0.79 | γ = 1.21 |
|---|---|---|
| Reference fit | bias +0.01 %, SD 0.0039 | bias −0.18 %, SD 0.0064 |
| One side of focus only | SD 4.3× | SD 4.8× |
| Selection on the observed dip, 2 gl | −1.14 % | −1.72 % |
| Selection on the predicted dip | +0.07 % | −0.09 % |
| Background +0.05 gl | +1.39 % | +1.42 % |

The node bootstrap's SE/SD is 0.83 and its 95 % intervals cover 92-93 %. Phase I through the camera chain gives γ̂ = 0.83 ± 0.017 against 0.79 (SPEC l.699).

On the pilot, only 2 of 19 far nodes survived. γ̂ = 0.408, t₁ interval [−0.01, 0.83], is a diagnostic only; no production kernel was written (far set insufficient, D-088.3) [log] l.139. **γ = 0.79 is still the configured value** [src] config l.190.

The geometric slopes are priors, not measurements (6.3, 6.5): 1.21 uniform disc, 0.90 isotropic, 0.79 cos-weighted at NA_c 1.4, 0.47 / 0.43 at NA_c 1.12 / 1.05. [reasoning] The pilot's 0.408 sits near the ZEISS-default value, but with J = 2 that is not evidence.

Superseded: the DL handoff calls the far design "unanswered" and "two-pass"; D-077 to D-081 settled it [log] l.6857-7211.

### 10.3 Step 2, the width-diameter trade-off (T10, open)

- Step 2 would tune σ_r at each knot until rendered phantoms, at each node's own d̂, reproduce the real profiles plane by plane, through the full chain [doc] procedure §3.4 step 2, l.354-358.
- Open [doc] SPEC §8, l.953:
  - what to match: the mean profiles or the core widths;
  - at which d and μ to render each calibration node;
  - the cost.
- Block 10's status says the first stage is not usable for rendering before step 2 (SPEC l.595).
- Rule: no calibration node may be selected by a d̂ computed under the kernel being calibrated (D-061.4).
- Numbers:
  - at d = 0.3 µm, V_tube ≈ 0.0056 µm² against σ_r(0)² = 0.0064 µm² [doc] procedure l.378-379;
  - the pilot's near nodes measure d̂ 0.6-2.5 µm, much thicker than the procedure assumed [doc] SPEC l.573; [reasoning].

**Figures (all of section 10).** B4 (new): (a) near field (configured table, pilot fit with the anchor SE, the phantoms' first stage); (b) far field (the configured law, the geometric slopes as priors, the pilot's diagnostic slope with its t₁ interval). Every constant drawn is read back from its source [run here] L1-L5. To make: F-T6a (far-fit sensitivity bars), F-T6b (calibration funnel), F-T10a (V_tube against the in-focus and one-plane growth). In chats only: `fig1_far_line`, `fig2_far_fit`, `fig1_twopass`, `fig2_selection` (2.4).

---

## 11. T4 and T8 -- the bias (correction) table and the corrected diameter's uncertainty

### 11.1 T4.1 -- the table's design (closed)

**Current target and design**

$$b(x \mid \mathcal C, \mathcal S) = \mathbb E[\hat D/d \mid d, \varphi, \hat M\hat D = \hat\mu\hat d, \mathcal C, \mathcal S]\ \text{ at }\ x = (\ln d, \varphi, \ln(\hat\mu\hat d)),\qquad m = d\,b$$

Reads: the mean ratio of fitted to true diameter over phantoms of true $(d, \varphi)$ whose fit returned the optical depth $\hat\mu\hat d$. It is taken among the fits in the selection $\mathcal S$, under the configuration $\mathcal C$ (the estimator signature). Every real node carries the third coordinate. Source: [doc] SPEC §2.4, l.256 (D-063.1).

$$x_{g,r} = (\ln(d_g/{\rm um}),\ \varphi_g,\ \ln(\hat\mu_{g,r}\hat d_{g,r})), \qquad x^{\rm fail}_{g,r} = (\ln d_g,\ \varphi_g,\ \ln(\mu_g d_g))$$

Reads: a row's table coordinates (draw $g$, realization $r$). The failure count places every realization at its true coordinates. Source: [doc] SPEC Block 14 notation, l.668.

Design [doc] SPEC l.256; [src] config l.228-235:
- $G$ = 100 000 independent cylinder draws $(d_g, \varphi_g, \theta_g, \mu_g)$, with log d uniform on [0.2, 4.0] µm (d_max TEMPORARY, D-086), φ uniform on [0°, 90°), θ uniform on [0, π), log μ uniform on [0.3, 3] µm⁻¹ (PROVISIONAL).
- Each draw is realized $n_{\rm real}$ = 10 times with fresh nuisances: sub-pixel offset, axis depth, node jitter, camera noise, and a branch radius $r_{{\rm br},g,r} = d_g/2 + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma_{\rm br}^2)$, σ_br = 0.2 µm, redrawn until positive.
- Phantoms go through exactly the real chain (D-057.3).

Why the fitted third coordinate: d̂ and α̂ are strongly anti-correlated (−0.94 / −0.90 at 0.6 µm) [log] l.5592, data inspected. Realizations of one draw are correlated (ρ), giving a design effect $\tau^2[1 + (n_{\rm real} - 1)\rho]/(G\,n_{\rm real})$ [log] l.5818.

Superseded:
- $b(d, \varphi \mid \mathcal C)$ on a 10 × 10 grid with grid-point means, in mathematics Eq. 19 and procedure Eqs. 2, 7. Replaced through D-031 → D-053.3 → D-063.1.
- The ln α̂ axis.
- The phantom radius d/2: now d/2 + ε (D-058, D-062.1).
- Only procedure l.612-620 is marked. SPEC l.349, 352 and 356 are stale and unmarked.

### 11.2 T4.2 -- the grid value and its SE (closed; 4c-4f open)

$$(\hat a(x), \hat\beta(x)) = \arg\min_{a \in \mathbb R,\ \beta \in \mathbb R^3} \sum_{n \in \mathcal S} \varpi_n(x)\,\big[y_n - a - \beta^\top (x_n - x)\big]^2, \qquad \hat b(x \mid \mathcal C, \mathcal S) = \hat a(x)$$

Reads: the local linear fit at grid point $x$. $y_n = \hat d_n/d_n$ is a row's ratio. The weights are Gaussian, $\varpi_n(x) = \exp(-\frac12\sum_{k=1}^{3}(x_{n,k} - x_k)^2/h_k^2)$, with bandwidths $h = c\,(0.15,\ 5°,\ 0.3)$. Source: [log] D-063.2, l.5879; SPEC l.668.

Equivalently, $\hat b(x) = \sum_n \ell_n(x)\,y_n$ with $\ell_n(x) = e_1^\top A(x)^{-1} z_n\,\varpi_n(x)$, where $A(x) = \sum_{n \in \mathcal S}\varpi_n(x)\,z_n z_n^\top$ and $z_n = (1,\ (x_n - x)/h)$. The weights sum to 1 and annihilate linear $b$ [doc] SPEC 14.4, l.672.

$$\widehat{\rm Var}(\hat b(x)) = \sum_g\Big(\sum_{r:\,(g,r) \in \mathcal S}\ell_{g,r}(x)\,e_{g,r}\Big)^2$$

Reads: the cluster-robust sandwich by draw, with residuals $e_n = y_n - \hat b(x_n)$ taken about the cubic read-out. Clustering by draw accounts for the realizations' correlation. The independent-rows SE stays beside it, labelled; it falls short (0.56-0.93 of the across-table SD, against 0.80-1.20 for the sandwich). Source: [doc] SPEC l.672 (D-063.3, D-071.2).

Numbers [doc] SPEC l.672 and `checks/table_2026-10-21/` (branch):
- c is chosen by 5-fold cross-validation with folds by draw: c = 1 on the fixture.
- The grid has spacing ≤ c·h/2, so (41, 37, 37) at c = 1.
- The bias is O(h²) in b's curvature, measured ≤ 0.0023 (1.0-1.3 SE) near the thin end.
- Support floor: NaN where rcond(A) < 1e-12 or N_eff < 10.
- Why not the thin-plate spline: at 10⁶ rows a dense TPS needs 8 TB, while the local window holds N_eff ≈ 1.5×10⁴ [log] l.5914-5917.

Open, items carried from the 10-27 study handoff (outside the repo):
- 4c: the support floor counts rows, not draws.
- 4d: the CV's choice of c is not in the table's spread.
- 4e: a row's ratio and its x₃ come from one fit.
- 4f: ρ on the real design is unknown.
- The bias is not in the reported SE [reasoning].

### 11.3 T4.3 -- the cubic read-out and the inversion (studying)

- **Read-out.** Every grid array is read by its tensor-product cubic interpolant (`RegularGridInterpolator`, method "cubic"). Its weights are products of 1-D not-a-knot cardinal weights, sum to 1, alternate in sign and decay by 2 − √3 ≈ 0.27 per grid step. The SE at a query is the interpolant of the grid SEs. Source: [src] `analysis/table.py`; the weight facts were established in the tensor-cubic chat (2.4), not in a repository document.
- **Inversion.** Classical calibration: solve the forward curve for $d$. Its bias does not depend on the population of real diameters; regressing d on d̂ (the "inverse estimator") does [doc] 10-19 handoff §5.3, l.277-284.

$$\tilde d_i:\quad \hat m(d, \varphi_i, \hat\mu_i\hat d_i \mid \mathcal C) = \hat d_i\ \text{ on } [d_{\min}, d_{\max}],\qquad \hat m = d\,\hat b(\ln d, \varphi_i, \ln(\hat\mu_i\hat d_i))$$

Reads:
- Find the true diameter whose expected measurement, at the node's tilt and fitted optical depth, equals what was measured.
- The root is bracketed on 256 log-spaced d and found by `brentq`.
- One sign change gives the root; none gives `out_of_domain`; more than one gives `non_monotone`.

Source: [doc] SPEC l.256, 14.5 l.673; [src] `analysis/invert.py` l.80-90.

$$\frac{d^{\rm s}_i - d^\star_i}{d^\star_i} = -\frac{\beta\,(b^\star - 1)}{1 + \beta\,(b^\star - 1)} \approx -\,\beta\,(b^\star - 1), \qquad \beta(d, \varphi \mid \mathcal{C}) = \frac{\partial \ln b(d, \varphi \mid \mathcal{C})}{\partial \ln d} \tag{Eq. 21}$$

Reads:
- The error of the superseded shortcut $d^{\rm s}_i = \hat d_i/b(\hat d_i, \varphi_i \mid \mathcal C)$, which looks b up at the measured instead of the true diameter.
- $d^\star_i$ is the analytic root and $b^\star = b(d^\star_i, \varphi_i \mid \mathcal C)$.
- Illustration: b* = 1.15 and β = 0.3 give −4.3 %.

Source: [doc] mathematics §3.6, l.707, l.712-715 (written in the old two-argument form).

Error pieces present in the repository [doc] mathematics l.718-727; procedure l.582-585; 10-19 handoff l.286:
- **Node scatter after correction:** $\tau/[b(1+\beta)]$ in relative terms, because $\partial m/\partial d = b(1+\beta)$; a lower bound.
- **Table term:** $(\tilde d - d)/d \approx -(\hat b - b)/\hat m'(d)$.
- **Jensen gap:** $\tfrac12 (m^{-1})''\,\mathrm{Var}(\hat D)$, because the inversion inverts the mean.
- **The simulator's systematics:** in neither τ̂ nor the SE.

Checks in the repository: `checks/table_2026-10-18/check_table_inversion.log` passes 10 of 10. For example, SD(d̃)/d = 0.17237 against τ/[b(1+β)] = 0.17195, and the shortcut error at d = 0.5 µm is +0.01085 against Eq. 21's +0.01314 [doc].

Open [open]:
- Items 5b-5g of the outside material: the error budget, and the SE of d̃.
- **No SE of d̃ exists in the code** (`Inversion` has no SE field; the per-node CSV has none).
- The Eq. 21 band failure of `inversion_2026-10-28` is unexplained in the repository.
- The monotonicity check counts sign changes only.

### 11.4 T8 -- uncertainty and flags of the corrected diameter (open)

There are three kinds of outcome [doc] SPEC l.256, l.673; [src] `analysis/invert.py` l.96-98, `analysis/fill.py`:
1. **The node fails S:** it is filled.
2. **An inversion flag:** the node is filled. The flags are:
   - `out_of_domain`, `non_monotone`;
   - `large_correction` ($|\hat b - 1| >$ 0.2, D5);
   - `high_failure` ($\hat p_{\rm fail} >$ 0.2, PROVISIONAL);
   - `off_support` ($M < q_{\rm LOO}$, the kernel mass at the root below the 5th leave-one-out percentile);
   - `phi_out_of_domain`.
3. **A label:** the node is corrected and kept. The labels are `dark`, `faint`, `steep`, `iter_*` and `mu_d_out_of_domain` (D-030, D-096, D-024 (iv), D-068, D-071.4).

Fill: the median of the nearest corrected node on each side within the stretch, else Allen's 2r, then a 3-node running median [doc] SPEC l.520.

Numbers [doc]:
- At local N_eff 15-28, P(p̂_fail > 0.2) = 0.45-0.47 when p = 0.2, so the flag is noisy (`checks/table_2026-10-18/check_table_steps45_extra_F.log`; SPEC l.932).
- The per-node scatter of the ratio is SD 0.008-0.010 at 1.5 µm and 0.17-0.20 at 0.6 µm (10-16 handoff l.407-410).

Open:
- No SE column.
- `max_failure_rate` waits for the cluster pilot's curve.
- The 0.2 threshold is to be revisited against the share of the domain it flags; that report is specified but not built.
- Procedure §3.10's checks have not run.
- No uncertainty is defined for filled values.

**Figures (all of section 11).**
- B5 (new): the inversion as classical calibration, with the toy numbers of mathematics §3.6. It is labelled as an illustration, not the project's table [run here] I1-I4.
- S1: the support band. Redraw it in today's coordinates before use (14.2).
- S5: TPS weights against a Gaussian window; a history slide only.
- To make: F-T4a (rows, grid, read-out, root from the code on a fixture), F-T4b (the support band in (ln d, ln μ̂d̂)), F-T8a (error budget bars, labelled "not computed by the pipeline").

---

## 12. Cross-cutting cautions for slide text

### 12.1 Statements that are superseded (do not present them as current)

| Old statement | Where | Replaced by | Current |
|---|---|---|---|
| The renderer's kernel becomes the SGT family | log D-066 (l.6102) | D-067.2 (l.6228), withdrawn the same day | The renderer is Gaussian, `gaussian_table`; the fit uses the SGT |
| The default renderer is the hybrid | D-084.1 | D-084.2; D-092.2 | The hybrid is built and smoke-tested; the default is still the partition (config l.192) |
| b(d, φ \| C) on a 10 × 10 grid; grid-point means; SE τ̂/√N | mathematics Eq. 19; procedure Eqs. 2, 7; l.400-405 | D-031, D-053.3, D-059, D-063, D-071 | Random design G × n_real; target at ln(μ̂d̂); local linear; sandwich SE (§11) |
| Thin-plate spline estimator | D-024 (v); the TPS document | D-063.2, D-067.3 | Local linear. `tps_spline` survives only for small tables and tests |
| The shortcut d̂/b(d̂) | handoff Eq. 13; D5 | mathematics Eqs. 20-21; D-024 (vi) | Root of the three-coordinate m̂ |
| Focus by dip depth; final plane k* | handoff Eq. 1 | D-035, D-043, D-044, D-052.3 | Gradient energy for the pilot; Allen-guided k^it for the fit |
| α fitted; background fitted | handoff Eqs. 10-11 | D-019, D-018 | μ fitted; background fixed |
| Calibration nodes "d̂ ≲ 0.3 µm, faint, φ ≲ 10°" | procedure l.320-321; ideas l.54-55; D-026 statement | D-061.4, D-094, D-101, D-108, D-112 | By Allen's diameter; no faint rule; ≤ 10° near / ≤ 15° far; stripe isolation |
| Far design "two-pass", "not decided" | DL handoff l.80, l.158, l.166-175 | D-077 to D-081 | Iterated core fit, N ≤ 10; sequential far fit with the anchor (§10.2) |
| "σ ≈ 1.2\|δ\|"; "FWHM 0.24 µm"; "narrowest feature ≈ λ/NA" | early handoffs | optics v1.3-v1.5 [corrected 2026-10-04] | 0.79-1.21 by weighting; Airy FWHM 0.20 µm; λ/(2NA) for intensity |
| The renderers doc's "hybrid not logged, no decision" | renderers l.7-8, l.989-991 | I-004, D-084, D-092 | Built as Block 21 (§6.7) |
| `alpha_out_of_domain` fills the node | log l.6049-6050 | D-071.4 | `mu_d_out_of_domain`, a label; the node is inverted |

### 12.2 Identifier collision: "D-030"

Before 2026-10-18 the SPEC and `config.py` call the gradient-energy focus rule "D-030". In the decision log that rule is **D-035**; the log's D-030 is "every real node gets its own estimate whatever its optical depth" [doc] SPEC l.69, l.381; [src] config l.50-51; [log] l.3738. The roadmap's T2 (`decision: D-030`, refs D-030) is therefore ambiguous. Cite the log's ids on slides.

### 12.3 Notation clashes across documents (choose one notation per slide deck)

| Symbol | Means, by document |
|---|---|
| w | depth z − c_z (implementation handoff); the Gaussian dip width (SPEC); the dip B − I (mathematics); the SGT FWHM (10-16 handoff); the removed field 1 − t (theory handoff) |
| B | background grey level; the beta function (H1) |
| β | SGT peak shape; the elasticity ∂ln b/∂ln d (Eq. 21) |
| λ | wavelength; the IRLS centre-weight scale (0.5 µm) |
| α | absorbance μd/cos φ; the angle of an oblique line to ê_u (handoff Eq. 8) |
| s | direction sines (optics); the far fit's axis shift s_j (SPEC 15.4); a fitted core SD in the S2b answer |
| γ | far slope of σ_r; γ_ω, γ_W geometric RMS slopes (priors, not the same quantity); in code `gamma` is also ζ of the SGT |

---

## 13. This chat's optics checks (2026-11-02): numbers and corrected claims

Setting: scalar optics, λ = 0.55 µm, n = 1.515, NA_o = 1.4, Koehler source uniform in direction sines, thin (projection) tube, in focus unless stated. Scripts: pack `scripts/check_optics_claims.py`, `check_conjugate_planes.py`, `check_diffraction.py`. Outputs: `checks/*.out`, `check_results.json`. All were re-run on 2026-11-02 for this handoff with the same results. Predictions were fixed before each run; failures are kept.

| Check | Claim tested | Result |
|---|---|---|
| D1, T1-T2 | In-focus Debye spot = Airy; first zero 0.61 λ/NA | PASS: 0.239578 µm; FWHM 0.2021 µm |
| D3 | At the quarter-wave defocus the 50 %-energy radius grows < 10 % | **FAIL**: +16 %, as light moves into the rings |
| D3b | ... the core is unchanged: FWHM < +5 %, peak ≥ 0.80 | PASS: +1.3 %, 0.8227 |
| D4 | At 0.84 µm the wave and geometric radii agree within 15 % | PASS: 0.685 against 0.725 µm |
| C1 | Γ = 2J₁(β_c)/β_c by source quadrature | PASS: 1e-8 |
| C2 | H_1 = incoherent OTF; cutoff (1+S)NA_o/λ | PASS: 1e-12 |
| C3 | S = 1 matches the incoherent model to first order (difference ∝ μ²); S = 0.75 does not (∝ μ) | PASS: ratios 3.80 and 1.94 |
| C4 | I_inc − I_S=1 ≥ 0 everywhere | **FAIL**: −1e-4 to −5e-4, 9-40 µm away; quadrature ripples |
| C4b | ≥ 0 within 2 µm of the axis; the far minimum shrinks with a finer source grid | PASS |
| C5 | I_inc = I_S=1 + the dark-field image of the directions outside the cone | PASS: residual 2.6 % and 1.2 % of the peak difference |
| C6 | Off focus, phase contrast becomes first order at S = 1 | **FAIL**: at S = 1 it stays second order (ratio 4.13) |
| C6b | At S = 1 second order off focus; at S = 0.75 first order (ratio 2.01) | PASS |
| G1 | A defocused point's rays cross the axis over a range of 1/cos θ_o | PASS: 2.616 against 2.617, so a defocused point has no stigmatic image at NA 1.4 |
| W1 | Refocusing the camera cannot restore a defocused point's peak | **FAIL**: at δ = 0.3 µm the peak goes from 0.422 to 0.990 at the best camera refocus |
| W1b | Peak ≥ 0.95 up to 0.6 µm after refocus | PASS: first below 0.8 at δ = 1.45 µm |

What to say, corrected:
- "Within the quarter-wave tolerance the core of the spot is unchanged; its energy is not."
- "At S = 1 the incoherent model is right to first order in the absorbance; partially closing the condenser breaks that, and adds phase contrast off focus."
- "A point is imaged to a point only in paraxial optics; at NA 1.4 a defocused point has no sharp image, although refocusing the camera recovers most of its peak for small δ."

---

## 14. Figure inventory (the pack)

Pack root: `TEEG_diameter_theory_presentation_2026-11-02/`. A and B are this chat's figures, regenerable from `scripts/`. R is copied from the repository (`main`); S is from the branch and must be redrawn or used only as history.

### 14.1 Use these

| Id | File | Topic | Shows | Origin | Caveats |
|---|---|---|---|---|---|
| R1 | `figures/R_repo_main/TEEG_microscope_schematic_2026-10-12.png` | T7 | Allen's microscope; both Koehler conjugate sets; the cone at F (67.5° open; 43.9-47.7° at ZEISS's setting) | `main` `$D/figures/` (e3cc4b0, 10-12); conjugate planes checked, 11 checks PASS [doc] README (main) | Schematic; its scripts are not in the repo |
| R3 | `.../fig01_kohler_one_point_one_wave.png` | T7 | One diaphragm point gives one plane wave through the specimen | `main` `$D/figures/TEEG_diameter_sine_condition_and_W_2026-10-12/` | Schematic |
| A6 | `figures/A_this_chat_optics/fig6_pupil_function.png` | T7 | Pupil P = A e^{iΦ}: ideal, NA 0.7, defocus 0.3 µm, coma; and their spots (peaks 1 / 0.25 / 0.43 / 0.60) | `make_pupil_figure.py`; T1-T4 PASS | -- |
| A1 | `.../fig1_pupil_to_spot.png` | T7 | Pupil disc, then Airy spot (log), then profile: first zero 0.24 µm, FWHM 0.20 µm | `make_figures.py`; D1 | -- |
| A2 | `.../fig2_defocus_wave_vs_rays.png` | T7, T3 | Spot through focus with the geometric cone; widths; centre brightness (0.82 at 0.147 µm) | `make_figures.py`; D2-D4, D3b | -- |
| A3 | `.../fig3_kohler_coherence.png` | T7 | Abbe's picture of Koehler; Γ(Δx) at S = 1 and 0.75 (zeros 0.24 / 0.32 µm) | `make_figures.py`; C1 | (a) is paraxial and schematic |
| A4 | `.../fig4_transfer_functions.png` | T7 | First-order transfer H_S for S = 0 / 0.5 / 0.75 / 1; cutoffs NA/λ and 2NA/λ | `make_figures.py`; C2 | Weak-object, in focus |
| A5 | `.../fig5_tube_coherence.png` | T7 | Tube images (μd = 1, 3) at S = 1 / 0.75 / → 0 against the incoherent model; the difference = the dark-field image | `make_figures.py`; C3-C5 | Thin-object projection |
| R6 | `.../fig04_W_vs_Wstar.png` | T7 | W (cos ϑ per sr) against W* (flat); cumulative; medians 40.8° / 46.3° | sine doc §3.8 | -- |
| R10 | `figures/R_repo_main/TEEG_ray_world_vs_partition_2026-10-07.svg` (preview in `previews/`) | T3, T7 | One oblique ray: light history along the ray (ray world) against along the vertical (partition) | `main` `$D/docs/` | -- |
| R14 | `.../fig4_slab_rendering_one_output_plane.html` (preview) | T3 | One plane rendered from slabs (partition) against the plain linear sum | `main` `$D/figures/`; spec §Figure 4 | Interactive; illustrative widths; partition only |
| R11 | `.../fig1_defocus_cone_disc_lsf.html` (preview) | T7, T3 | Defocus cone, disc and the dip across a branch | spec §Figure 1 | Uniform disc, so its RMS readout is an upper bound (0.79-0.90 realistic) |
| B1 | `figures/B_this_chat_theory/fig7_T1_measuring_line_geometry.png` | T1 | Ellipse section, ŷ chord = d, pixel-row chord 1.265 d; chord/d against misalignment | `make_theory_figures.py`; G1-G3 | -- |
| B2 | `.../fig8_T2_tube_forward_model.png` | T2 | T(v); the (B3.1) profile for d = 0.3-2 µm with sample positions; V against d and σ_fit² | P1b, P2b, P3 | Gaussian σ_fit (pilot); the final fit uses the SGT |
| B3 | `.../fig9_T5_sgt_kernel_family.png` | T5 | Six SGT members at equal FWHM; tails on a log scale | K1-K6 | -- |
| B4 | `.../fig10_T3-T6-T10_defocus_kernel_law.png` | T3, T6, T10 | Near field (configured, pilot, phantoms' first stage); far field (configured law, geometric priors, the pilot's diagnostic γ̂ with t₁ band) | L1-L5 | The pilot far slope is a 2-node diagnostic |
| B5 | `.../fig11_T4-3_inversion_illustration.png` | T4.3 | Inversion as classical calibration; the shortcut's error; scatter mapped through the slope | I1-I4 | ILLUSTRATION with the mathematics doc's chosen numbers |

### 14.2 Backup or with caveats

| Id | File | Topic | Shows / caveat |
|---|---|---|---|
| R2 | `figures/R_repo_main/TEEG_direction_sines_2026-10-12.png` | T7 | Why s has two components: diaphragm point h, then s, then the unit direction ŝ. Its label "Eq. (ii)" is the sine doc's Eq. (5) |
| R4 | `.../fig02_sine_rule.png` | T7 | The sine rule against the tangent rule (tan/sin = 2.62 at 67.5°) |
| R5 | `.../fig03_map_and_jacobian.png` | T7 | Diaphragm to directions, three charts; dot density shows the Jacobian |
| R7 | `.../fig05_obliquity.png` | T7 | Why W thins as cos ϑ per sr (schematic) |
| R8 | `.../fig06_why_the_sine.png` | T7 | The shifted-source argument (schematic) |
| R9 | `.../fig07_index_mismatch.png` | T7 | Snell geometry at n_med = 1.33 (illustrative, not Allen's) |
| R12 | `.../fig2_diffraction_focal_spot_vs_na.html` (preview) | T7 | 2-D wave fan; FWHM 0.155 µm at NA 1.4 is not Airy's 0.20 µm |
| R13 | `.../fig3_tilted_plane_wave_lateral_frequency.html` (preview) | T7 | A tilted plane wave's ripple, Λ = λ/(n sin ϑ) |
| S1 | `figures/S_superseded_redraw/fig_support_band.png` | T4.1, T8 | The phantoms fill a band of slope 1; the inversion line is supported only inside it. **Redraw before use**: its y-axis is the superseded ln α̂ (equal to ln μd at φ = 0); its legend's "root 0.492 µm unflagged" is spline-era, and today `off_support` fires for that node (SPEC l.673) |
| S5 | `.../tps_fig05_table_weights.png` | T4.2 history | Spline weights reach far and go negative; the Gaussian window is local. Use only as "why a local estimator". The TPS is superseded |
| S2-S4, S6 | `.../tps_fig01-04_*.png` | history | Spline theory and a spline-era table slice; not current |

Regenerate:
```
cd scripts
python3 check_optics_claims.py && python3 make_figures.py && python3 make_pupil_figure.py   # A1-A6
python3 check_conjugate_planes.py; python3 check_diffraction.py
python3 check_theory_figures.py --repo <clone at sci/diameter-pipeline>                       # 21 of 24 PASS (3 recorded FAILs)
python3 make_theory_figures.py ../figures/B_this_chat_theory                                   # B1-B5, byte-identical
python3 render_previews.py ../figures/R_repo_main ../figures/R_repo_main/previews              # needs Playwright
```
- The scripts are UTF-8. The plotting scripts contain non-ASCII labels.
- None of them is pipeline code.
- The palette is the dataviz reference palette. Its categorical slots 1-6 were validated: all checks pass, slots 3-5 are below 3:1 contrast, so every series also has a label or a line style.

---

## 15. Figures still to make (specs)

| Id | For | Content | Data |
|---|---|---|---|
| F-T9a | T9 | Bars: run-B excitatory median Cm by layer with n; reference band 0.43-0.52 µF cm⁻² (Eyal 2016 model fits); 529878215 at 3.0 | KB annual report §6; Eyal 2016 Fig. 1 |
| F-T9b | T9 | Pilot d̂ bins (x) against Allen's 2r (median, IQR); identity line; label "uncorrected, before D-052" | `MyDrive/diameters/pilot/pilot_529878215.csv` (user's Drive) |
| F-T9c | T9 | Chain: frustum area, area ratio, `--swc-dir`, six-parameter refit (gated by D-083.1), F applied beyond 60 µm | SPEC l.535, l.49-50; log l.7266, l.329 |
| F-T2a | T2 | G/max and F/max against plane offset −6 to 6 for a rendered dark tube (d 1.5 µm, μ 1/µm); argmax of each | SPEC Block 5 oracle l.491-495 |
| F-T5a | T5 | Δχ²_min against ρ = d/d_true for 0.6 and 1.5 µm truths, five kernels | `python fig_sgt.py --dir ident_sgt --student ident/identifiability.json` in `$D/checks/estimator_2026-10-15/` (branch) |
| F-T5b | T5 | The Allen-guided iteration: planes K (y) against v (x), each round's line ±1.5 d^(i−1), G/max | `gradient_iteration(..., fit_at=sgt_at_plane)` on `tests/smoke/fixtures_cell.py`; plotting `iteration_figure` |
| F-T7a | T7 | Where the hybrid books the light: ΔA^hyb over (v, z) at μd 3 and 50 with the share above the axis (26.8 % / 10.3 %); τ(v) at the node plane ± 0.28 / 0.56 µm for ray world, hybrid (cone kernel and production Gaussian) and partition, with τ = 0 drawn | `$D/checks/hybrid_2026-10-27/profile_survey.py`, `cone_survey.py` (extend them to print profiles); renderers l.849-864 |
| F-T7b | T7 | σ against \|δ\| (0-1.12 µm): the production σ_r; γ_W\|δ\|; γ_W*\|δ\|; the ray world (W* + 0.08 µm post-blur); NA_c 1.12 / 1.05 lines; shade \|δ\| > 0.093 µm "direction average forced"; mark 0.147 and 0.28 µm | Renderers l.593-598; sine doc l.644-648 |
| F-T6a | T6 | Bars of the far-fit sensitivities (table in §10.2) | DL handoff §6, l.230-243 |
| F-T6b | T6 | Two funnels, near and far: candidates, flat, stripe, planes, chosen, measured, scanned (7500 → 380 → 84 → 30; 723 → 19 → 2) | Log l.139; calibration handoff l.105-106 |
| F-T10a | T10 | V_tube(d), faint and opaque, against σ_r(0)² = 0.0064 and G(one plane) = 0.0085 µm²; shade the pilot's near d̂ range 0.6-2.5 µm | Procedure l.378-379; SPEC l.573; `dark_node_growth.out` l.3 |
| F-T4a | T4.2-T4.3 | (1) rows (grey), grid b̂ (markers), cubic read-out ± 2 sandwich SE (band) and ± 2 independent-rows SE (dashed), support floor hatched, along d at fixed (φ, μ̂d̂); (2) m̂ = d b̂ with the line at d̂_i, the root, the shortcut | `table.fit_table` on the SPEC l.672 fixture (τ = 0.03, ρ = 0.2) |
| F-T4b | T4.1, T8 | S1 redrawn in (ln d, ln μ̂d̂): rows with \|φ\| < 5°, band edges, the contour M = q_LOO, two inversion lines (mid-band; the C5 node, now `off_support`), the clamped ends hatched | `checks/table_2026-10-20/fig_support_band.py` adapted |
| F-T8a | T8 | Error budget of d̃ at d = 0.3 / 0.6 / 1.5 / 3 µm: node scatter, table term, Jensen gap; hatched "not in the SE": σ_r(0), kernel shape, cross-section, start radius. Label "not computed by the pipeline" | Mathematics l.718-727; 10-16 handoff l.407-410 |

---

## 16. Not checked, and the limits of this handoff

What this chat checked itself:
- Every repository path in sections 2 and 14, by script against the commits above.
- A scripted spot check of 125 cited line ranges against the files at those commits. 122 matched at once. The other 3:
  - a phrase wrapped across two lines (procedure l.582-585, correct);
  - the profile's sample count, cited from the wrong place and corrected (§8);
  - whitespace inside LaTeX (log l.2444-2449, correct).
- The decision texts of D-066/D-067.2, D-084, D-092, D-097 and the pilot row (log l.139).
- The configured kernel (config l.186-192), `Inversion` (invert.py l.49-54), SPEC l.256, l.584, l.668, l.672, l.685-688, l.699.
- The (H1)-(H6) text; the handoff's geometry (l.185-215); mathematics §3.6.
- The DL handoff's §6 table; the d094 phantom table; the run-B and Eyal numbers in project knowledge.
- The roadmap statuses (`roadmap.py status`, not shallow).

What rests on the readers' digests: other line numbers and doc-stated numbers. They are tagged [doc] or [log] with their location; each reader re-grepped its citations, but this chat did not re-read every one.

Not read by anyone for this handoff:
- The runbook (`TEEG_diameter_phase2_colab_runbook_2026-10-06.md`, operational).
- The notebooks.
- Most of SPEC Blocks 16-20 (operational).

Literature is as cited inside the documents (their reference sections). This chat ran no new literature search for the handoff. Slides that cite papers must follow the full-text rule.

Chat-only material (2.4) was located through the conversation index; its content was not re-read.

The project-knowledge copy of this file may have been refused (knowledge over capacity on 2026-11-02); the reply that delivered it says which. The repository copy exists only as a patch until the user applies it.
