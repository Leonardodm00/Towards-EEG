# Handoff: the defocus kernel $\sigma_{\rm r}(\delta)$ from a deep-learning perspective (diameter implementation chat -> kernel DL chat)

| Date | Change |
|---|---|
| 2026-10-22 (later) | Pushed to `sci/diameter-pipeline` at the user's agreement, merged with the remote's roadmap commit `9112bf3`. §3's roadmap row now names theory stage T7, the optics of the hybrid plane absorption (I-004), added at the user's request; §1 gives the next free IDs, D-075 and I-005. |
| 2026-10-22 | Created at the user's request (15:09) for a new chat that tackles the construction of the renderer's defocus kernel with deep-learning methods, logged as **I-003**. State described: `sci/diameter-pipeline` at `825bef1` (on the remote), estimator hash `0508e7ff12f48728`, simulation hash `0ea5a6d2ec069c68` (the rows carry `sim-0ea5a6d2ec069c68`), 25 of 25 smoke suites passing **[run, sandbox, 2026-10-22]**; S2b (the far-field calibration) designed, its fit not decided (D-074); Block 10 never run on Allen's data. Committed to `docs/` on `sci/diameter-pipeline` as a local commit (pushed only at the user's agreement). A project-knowledge copy under `claude/` was refused because the project is at its size limit, so this repository file is the one to read (D-020: the workstream's documents live in the repository). |

Basis tags: **[src]** read in the named file at `825bef1`; **[run, sandbox]** executed in the
assistant's sandbox (Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1), on the renderer's own model unless
stated; **[user]** said or decided by the user; **[KB]** the decision log in project knowledge;
**[doc]** a document of the workstream's `docs/` folder, section named (branch named where the copies
differ); **[PubMed, full text]**, **[PubMed, abstract only]**, **[preprint]** literature by access level
(§12); **[reasoning]** the assistant's inference, not checked; **[open]** neither decided nor verified.

Plainly: the diameter pipeline corrects every measured dendrite diameter with a table built by simulating
the microscope. The one part of that simulated microscope that has to be measured from Allen's own stacks
is how much a shadow blurs as it moves out of focus -- the defocus kernel. Today it is a Gaussian whose
width $\sigma_{\rm r}$ grows with the distance $\delta$ from focus. Its near-focus part comes from thin
dendrites (Block 10: built, never run on real data, biased within two planes of focus); its far part comes
from 0.6-0.8 um dendrites through a straight-line law (S2b: designed and checked on simulations, not
decided). The new chat asks whether deep learning builds this kernel better: more robustly at low signal,
with honest uncertainty, or with a non-Gaussian shape. Three things bound the answer: what the data can
identify at all (§5), what the user has decided (§3: the renderer stays Gaussian unless the user reopens
it), and what the classical route already achieves on simulated data (§6), the number to beat.

## 1. First actions in the kernel DL chat

1. Read, in this order:
   - this file;
   - in `TEEG_decisions_and_ideas_log.md` (project knowledge): **D-067** (.2: the renderer stays Gaussian,
     "without overengineering"), **I-002** and **D-066** (the withdrawn non-Gaussian plan; its "What it
     implies for code and tables" is the checklist for any non-Gaussian kernel), **D-026** with (D-026.1)
     and comments (a)-(d), **D-027**, **D-074** (S2b: what is settled, what is open), **D-061** (.4: nodes
     selected by Allen's diameter; (f): the kernel table is made in Colab and handed to davinci), **D-070**,
     **D-071** (.5), **D-025** (the image noise), **D-024** (i), **I-003**;
   - `specs/SPEC.md`: Blocks 4, 9, 10, 15 and the last three bullets of §8 (step 2, the kernel's
     resolution, the alternative generator);
   - `docs/TEEG_diameter_bias_table_procedure_2026-10-04.md` §3.4 (the kernel, Eq. 4, what to match, the
     empirical family) and §3.10 (the checks that decide whether the table is trusted);
     `docs/TEEG_diameter_bias_table_mathematics_2026-10-04.md` §3.3-3.5 (Eqs. 15-18);
   - the optics document **v1.5** §3.5-3.7, on `origin/main` only (the branch holds an older copy;
     command in §11).
2. Clone, run the suites, print the hashes (§11). At `825bef1`: 25 suites pass and the hashes are
   `0508e7ff12f48728 0ea5a6d2ec069c68` **[run, sandbox, 2026-10-22]**. The remote head moved to `9112bf3`
   at 15:19 the same day (the roadmap under `specs/roadmap/` and one `.gitignore` line, no code change), so
   the same results are expected there **[run: `git fetch`; reasoning for the expectation]**.
3. Settle §2's questions with the user before writing computational logic (scientific-coding §2).
4. Build the synthetic benchmark of §9 and reproduce the classical baseline of §6 on it before training
   any network.
5. The decision log is shared with the implementation chat and the noise chat: re-read it from the project
   before every write (on 2026-10-21 an entry was renumbered to D-074 after a collision). Next free IDs
   after this handoff: **D-075, I-005** (I-004, the hybrid renderer, was logged the same day).

## 2. Scope questions for the user, before any logic

The assistant's proposals, recommendation first **[reasoning]**; none is decided.

| # | Question | Options | Why it matters |
|---|---|---|---|
| Q1 | Which kernel does the DL work construct? | (a) the Gaussian width law $\sigma_{\rm r}(\delta)$ and the far slope $\gamma$, inside D-067.2; (b) a non-Gaussian $K_\delta$, which reopens I-002 / D-066 | (a) lands in existing configuration fields (§10); (b) changes Block 4, the FFT path, gate 1, Block 10's statistic and the simulation hash (D-066's list) |
| Q2 | Where does learning enter? | (a) components of the classical route: a learned far-plane background, a learned per-plane width statistic (C1, C2 of §8); (b) a differentiable forward model fitted to the raw profiles, i.e. procedure §3.4 step 2, never built (C3); (c) amortised neural posterior estimation over simulated scans (C4); (d) several of them, scored on one benchmark | each has its own data needs (§7) and failure modes (§8) |
| Q3 | Which data? | synthetic first (the sandbox cannot reach Allen's server); real profiles from Colab once Cell 7 has run and the profiles are exported (§7) | the scans file holds per-plane statistics only |
| Q4 | Which branch, and what merges back? | a new branch off `sci/diameter-pipeline` (e.g. `sci/kernel-dl`), merged only at the user's agreement | the implementation chat keeps working on `sci/diameter-pipeline` (Block 15) |
| Q5 | Which compute? | a Colab GPU runtime or davinci; whether either has a GPU available to this work is not checked **[open]** | the sandbox has 2 CPU cores and no PyTorch, JAX or `sbi` installed **[run, sandbox]** |
| Q6 | How does it meet S2b's open points (D-074.3-.4)? | the DL chat supplies evidence (e.g. on the far background) that the implementation chat uses; or it waits for them | two chats must not give two answers to one question |

## 3. Where the kernel stands

| Item | State | Source |
|---|---|---|
| Renderer kernel family | circular 2-D Gaussian of SD $\sigma_{\rm r}(\delta)$, even in $\delta$, the same at every depth in the slice (`kernel_family = "gaussian_table"`) | D-067.2 **[user]**; D-066 withdrawn; I-002 an idea, not scheduled **[KB]** |
| Fit-side kernel | the SGT family, $(w, \beta, \xi, \zeta)$ fitted per node | D-052 **[KB]**; SPEC Block 12 **[src]** |
| $\sigma_{\rm r}(0)$ | 0.080 um, configured; not identified by the scans (§5, I-3); its uncertainty is to be propagated by reduced tables at 0.066 and 0.095 um | procedure §3.4, §3.10 **[doc]**; not run **[open]** |
| The table in the code | the ideal-Debye core width at 550 nm, "NOT a calibration" | `model/kernel.py` docstring, `config.py` **[src]** |
| Near field, $\lvert\delta\rvert \le$ 0.84 um | Block 10 built and smoke-tested; never run on Allen's data (Cell 7 not run); its first stage is biased within two planes of focus (§5, I-6); step 2 not built | SPEC Block 10, §8 **[src]**; D-070 **[KB]** |
| Calibration nodes | chosen by Allen's diameter $D_{\rm A} = 2r_{\rm A}$: near set $D_{\rm A} \le$ 0.6 um, far set 0.6 < $D_{\rm A} \le$ 0.8 um; in $\mathcal S$, flat ($\varphi \le$ 10 deg) and faint ($\hat\alpha \le$ 0.5) by the pilot; the fit's $\hat d$ is no criterion, because "the kernel being calibrated must not choose its own nodes" | D-061.4, D-067.4 **[KB]** (values PROVISIONAL); SPEC Block 10 **[src]** |
| Width statistic | the squared Gaussian-core width $\omega$, fitted within one FWHM; the windowed variance only as a cross-check, because real line-spread functions have $\lvert v\rvert^{-2}$ tails | procedure §3.4 **[doc]**; D-026 (a): the user's "second moment" read as this core width, the reading not confirmed **[KB]** |
| Far law | (D-026.1): linear from 0.84 um, slope $\gamma$ fitted on the far set, further terms only if the residuals require them; $\gamma$ = 0.79 configured until S2b | D-026 **[user]**; D-070.1 **[KB]** |
| Far scan | planes $k^*_j \pm 16$ on both sides of focus, an SNR rule at 3, a +-10 um line | D-074.1-.2 **[user]** |
| Far background, statistic, fit, isolation and tilt, junction | five questions put to the user on 2026-10-21, unanswered (§4.4) | D-074.3-.4 **[open]** |
| Image noise | the renderer's 3.0 gl is provisional; D-025's measurement on clean background patches not run | D-025 **[KB]**; `config.py` **[src]** |
| Allen's camera chain | black level, gain, noise after JPEG and the light direction are unknown; the JPEG tables are read from the cached crops (Cell 5) | SPEC §8 **[src]** |
| Above/below and depth dependence | checks planned (per-sign Eq. 4, shallow vs deep nodes), not run; the renderer's kernel is even in $\delta$ and `kernel_from_growth` refuses a two-sided growth | procedure §3.10 **[doc]**; SPEC Block 10 **[src]** |
| Roadmap | `specs/roadmap/diameter-pipeline.md` (generated from its YAML, created 2026-10-22): theory stage T6 "Far-field calibration" studying, implementation stage I15 (Block 15) planned, and T7 "Optics of the hybrid plane absorption and the defocus kernel" open (added the same day at the user's request, I-004). The hybrid of the 2026-10-09 theory document (`TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md` §3.8-3.9, on `main`) takes each slice's absorbed light from the condenser's rays and then applies the defocus kernel, so a forward model fitted to real profiles (C3, C4) inherits whichever absorption the renderer uses. A stage for the DL work itself is added only at the user's word (the subproject-roadmap skill) | `9112bf3` and the commit that added T7 **[src]**; I-004 **[KB]** |

## 4. The kernel in the code, in full notation

### 4.1 The renderer (Block 4, `model/render.py`) [src]

For every plane $k$ (depth $z_k = k\,\Delta z$, $\Delta z$ = 0.28 um, stage units) and every lateral sample
$(x, y)$ of the fine grid, the transmittance is

$$\tau_k(x, y) = 1 - \sum_{\ell} \big(\Delta A_\ell * G_{\sigma_{\rm r}(\zeta_\ell - z_k)}\big)(x, y),$$

where $\ell$ indexes the slabs (centres $\zeta_\ell$, thickness $\Delta\zeta$ = 0.02 um; SPEC Block 4 writes
$j$, renamed here because $j$ indexes the far nodes below); $\Delta A_\ell(x, y) = T_{<\ell}(x, y)\,
(1 - e^{-a_\ell(x, y)})$ is the fraction of the light of the vertical ray through $(x, y)$ absorbed in slab
$\ell$ ($a_\ell$ the slab's absorbance, $T_{<\ell}$ the transmittance of the slabs before it;
`absorption = "partition_vertical"`, procedure Eq. 5); $G_\sigma$ is the circular 2-D Gaussian of SD
$\sigma$ (um); $*$ is the 2-D convolution in $(x, y)$; and $\delta = \zeta_\ell - z_k$ is the signed defocus
(object minus plane). The camera chain then maps $\tau_k$ to grey levels: pixel integration into
$\tau_{\rm px}$, the grey value `black_level_gl` + `gain` $\cdot\,B\,\tau_{\rm px}$ with $B$ (gl) the
background (`background_B_gl` = 210, PROVISIONAL), additive $\mathcal N(0, \texttt{noise\_sd\_gl}^2)$,
rounding and clipping to 8 bits, a JPEG round trip (Allen's tables when given). Backends: `fft`, with the
Gaussian transfer $e^{-2\pi^2\sigma^2(f_x^2 + f_y^2)}$ per (slab, plane) pair and the pairs with
$\sigma_{\rm r} >$ `fft_split_sigma_um` = 0.5 um summed exactly against the continuous Gaussian on coarse
grids; `direct`, `scipy.ndimage.gaussian_filter`. The kernel enters only through $\sigma_{\rm r}(\cdot)$,
and both backends assume it is Gaussian.

### 4.2 The kernel law (`model/kernel.py`, `sigma_r(delta, rcfg)`) [src]

With knots $0 = t_0 < t_1 < \dots < t_M$ (`kernel_table_delta_um`, um) and values
$\varsigma_0, \dots, \varsigma_M$ (`kernel_table_sigma_um`, um), for every $\delta \in \mathbb R$:

$$\sigma_{\rm r}(\delta) = \begin{cases} \text{the linear interpolation of } (t_m, \varsigma_m)_{m=0}^{M} \text{ at } \lvert\delta\rvert, & \lvert\delta\rvert \le t_M,\\ \varsigma_M + \gamma\,(\lvert\delta\rvert - t_M), & \lvert\delta\rvert > t_M, \end{cases}$$

with $\gamma$ = `kernel_continuation_slope` under `kernel_continuation = "linear"` ($\gamma = 0$ under
`"frozen"`, $\varsigma_M/t_M$ under `"proportional"`). Defaults: $t$ = (0, 0.14, 0.28, 0.42, 0.56, 0.84) um,
$\varsigma$ = (0.080, 0.086, 0.122, 0.262, 0.438, 0.603) um, $\gamma$ = 0.79 (D-024 (i); mathematics Eq. 17's
geometric slope $\gamma_\omega$ under the $\cos^4$ Debye weighting, against 0.90 for an isotropic and 1.21
for a uniformly bright disc **[doc]**). `kernel_family = "empirical"` raises `NotImplementedError`.
`validate()` requires the knots strictly increasing from 0, every $\varsigma_m > 0$, $\gamma \ge 0$ and
`sigma_r0_um` $= \varsigma_0$.

### 4.3 Near field: Block 10, procedure Eq. 4 = mathematics Eq. 18 [src; doc]

For every node $i$ of the near set with at least 3 finite values (`min_planes_per_node`) and every plane
$k$ with $\lvert k - k^*_i\rvert \le P$ ($P$ = `offsets_planes` = 3, $k^*_i$ the node's sharpest plane):

$$\omega_{i,k} = c_i + G(z_{{\rm ax},i} - z_k) + \varepsilon_{i,k}, \qquad G(0) = 0. \tag{Eq. 4}$$

Here $\omega_{i,k}$ (um$^2$) $= \hat s_{i,k}^2$, with $\hat s_{i,k}$ the SD of a Gaussian
$a\,e^{-(v - v_c)^2/(2s^2)}$ least-squares fitted to the dip $w_{i,k}(v_n) = \bar B_{i,k} - I_{i,k}(v_n)$
on the samples within one FWHM of its smoothed minimum (at least 5); $I_{i,k}(v_n)$ (grey levels) is plane
$k$'s profile along the node's fixed measuring line (the fit's extraction, Eq. 9: the mean of 9
parallel bilinear lines within +-0.5 um along the branch, sampled every 0.1144 um over +-3 um);
$\bar B_{i,k}$ is the node's background in plane $k$ (Block 5); $c_i$ (um$^2$) is a per-node constant
(the tube's own width and the in-focus blur); $z_{{\rm ax},i}$ (um) is the node's axis depth, free within
+-0.28 um of its sub-plane depth; $G$ (um$^2$) is the growth curve shared by all nodes (the procedure's
$\Delta\sigma^2$), even in its argument (`origin_convention = "symmetric"`), tabulated at the knots
$m\,\Delta z$, $m = 0, \dots, 4$, cubic between them; $\varepsilon_{i,k}$ is the error. `fit_growth`
minimises the sum of squared residuals (scipy `least_squares`, analytic Jacobian). The first-stage kernel
(`kernel_from_growth`), at every knot $m\,\Delta z$:

$$\hat\sigma_{\rm r}(m\,\Delta z) = \big(\sigma_{\rm r}(0)^2 + \hat G(m\,\Delta z)\big)^{1/2}, \qquad \sigma_{\rm r}(0) = 0.080\ {\rm um\ (configured)}.$$

Procedure §3.4's step 2 -- tune $\sigma_{\rm r}$ at each knot so that rendered thin phantoms (the
calibration nodes' $\hat d$, the full camera chain, the same extraction) reproduce the real mean profiles
plane by plane -- is not built; SPEC §8 leaves open what to match, which $d$ and $\mu$ to render each node
with ($\hat d_i$ reads 5-25 % wide) and the cost of a renderer in the loop **[src]**.

### 4.4 Far field: S2b (D-026.1; SPEC Block 15, drafted; D-074) [KB; src]

The law, for every $\delta$ with $\lvert\delta\rvert > \delta_0$, $\delta_0 = P\,\Delta z$ = 0.84 um:

$$\sigma_{\rm r}(\delta) = \sigma_0 + \gamma\,(\lvert\delta\rvert - \delta_0), \qquad \sigma_0 \equiv \sigma_{\rm r}(\delta_0)\ (\text{not } \sigma_{\rm r}(0)).$$

The fit recommended in D-074.4 (not decided): for every far node $j$ and every plane $k$ with
$\lvert k - k^*_j\rvert \ge 5$ (so that $x_{j,k} \ge$ 0.4 um $\ge r_j$, the node's radius), with
$x_{j,k} = \lvert z_{{\rm ax},j} - z_k\rvert - \delta_0$,

$$\omega_{j,k} = c_j + (\hat\sigma_0 + \gamma\,x_{j,k})^2 + e_{j,k},$$

with $e_{j,k}$ the error, the sum of weighted squared errors minimised over $\gamma \ge 0$, the per-node constants $c_j$ and the axis depths $z_{{\rm ax},j}$ (within
+-0.28 um of the pilot's), on both sides of focus; $\hat\sigma_0 = (\sigma_{\rm r}(0)^2 +
\hat G(\delta_0))^{1/2}$ from the near fit (sequential); $\omega$ from a two-pass core fit whose first
window comes from the configured law at the pilot's depth ("S2" in the checks); weights from the far law's
predicted width; the SE by a bootstrap over nodes with $t_{J-1}$ intervals plus the anchor's term; Block
10's table cut at 0.84 um (`kernel_from_growth(..., max_delta_um=0.84)`), so that the renderer's
continuation starts where (D-026.1) is anchored. Open for the user since 2026-10-21 (D-074): (1) the
background (proposed: a trimmed planar fit $b_0 + b_x x + b_y y$ to the pixels of a +-25 um square farther
than $r_q + 3\sigma_{\rm r}$ from every traced segment $q$ and the soma, the line-end mean as its check);
(2) the statistic (two-pass vs Block 10's); (3) the fit (sequential vs joint vs axis fixed); (4) isolation
and tilt (a per-plane neighbour rule, $\varphi \le$ 5 deg, straightness over +-10 um, vs per-node
isolation); (5) the junction (cut at 0.84 um vs anchor at the last knot).

### 4.5 What the kernel must and must not contain

| Constraint | Source |
|---|---|
| $K_\delta$ models the optics only: pixel integration, the profile's bilinear interpolation, quantisation and JPEG come after it in the chain and must not be folded into it, or they are counted twice. A kernel learned from real crops has to be learned through the camera chain, not in place of it | procedure §3.4 **[doc]** |
| The chain rests on the incoherent form $B\,(T * h_0)$ ($T$ the specimen's transmittance, $h_0$ the in-focus PSF). With the condenser well below full aperture, or dendrites too dark for the weak-object approximation, that form is an approximation of unknown quality; the defocus cone depends on the unrecorded condenser setting, which is why the kernel is calibrated from the stacks | optics v1.5 §3.5-3.6 **[doc, main]** |
| An index mismatch makes $K_\delta \ne K_{-\delta}$ and depth-dependent; the checks are Eq. 4 per sign and shallow vs deep nodes | optics v1.5 §3.7 **[doc, main]**; procedure §3.10 **[doc]** |
| An isotropic PSF gives an even profile across a straight branch. Every symmetric SGT member is the line projection of a non-negative isotropic PSF (inverse Abel transform); the two-piece skew is the projection of none | I-002 (a); D-066.1-.2, kept on record by D-067.2 **[KB; run, sandbox 2026-10-20: `checks/table_2026-10-20/check_abel_sgt.py`]** |
| Real line-spread functions have $\lvert v\rvert^{-2}$ tails at every defocus, so their second moment is infinite and a windowed one depends on the window; a Gaussian matches one statistic only, and the procedure matches the core (in focus, ideal Debye: a core width of 0.080 um against $\sqrt{V_W}$ = 0.240 um at $W$ = 1.5 um) | mathematics §3.3; procedure §3.4 **[doc; run, `checks/psf_check.py`]** |
| The renderer needs kernels beyond the calibrated range: the focus search renders $k^* \pm 3$, so $\lvert\delta\rvert$ up to about $d/2$ + 0.98 um, and the synthetic $k^*$ always uses extrapolated kernels | procedure §3.4, "Range" **[doc]** |

## 5. What the data cannot identify, by any method

- **(I-1) An additive constant on $\omega$**: it moves only the $c_i$; $G(0) = 0$ pins it.
- **(I-2) A common shift of $\delta$**: it moves only the $z_{{\rm ax},i}$; the symmetric convention (or
  `"min_origin"`) pins it. Both are oracles of `test_smoke_calibration.py` **[src]**; mathematics §3.5
  **[doc]**.
- **(I-3) $\sigma_{\rm r}(0)$** cancels into the $c_i$: width scans do not identify it; it is configured
  **[doc]**. A network trained on simulations returns, for such a direction, its training prior
  **[reasoning]**.
- **(I-4) Resolution**: knots at a step $h$ finer than $\Delta z$ are not identified -- a zig-zag of
  period $2h$ fits the same data with shifted $c_i$; `validate()` refuses `knot_step_um` < `dz_um`. The sub-plane shape (the
  default 0.086 um at 0.14 um) stays configured **[src: SPEC Block 10 finding, §8]**.
- **(I-5) The far slope**: with the axis free and planes on one side only, only the curvature identifies
  $\gamma$; planes on both sides let the linear term identify it too; an axis-depth error acts exactly like
  a change of the anchor (D-026 (b), checked in `checks/calib_reach_check.py` part 3 on `origin/main`)
  **[KB]**. Leaving $\sigma_0$ free multiplies the SD of $\hat\gamma$ by 3.7-4.1 (§6) **[run, sandbox]**.
- **(I-6) The first stage near focus**: the core width of a mixture of defocus values (a 0.25 um tube spans
  $\delta \pm$ 0.125 um) is narrower than its mean variance. On 8 noise-free thin phantoms the fitted
  growth is low by 0.010 um$^2$ at 1 plane and 0.012 at 2 planes, the size of the growth itself at 1 plane
  (0.0085 um$^2$), and the first-stage table came out 0.080, 0.049, 0.399, 0.604, 0.806 um at 0-4 planes
  against the true 0.080, 0.122, 0.438, 0.603, 0.824: not monotone, not usable for rendering before step 2
  **[run, sandbox, 2026-10-06: SPEC Block 10 status]**.
- **(I-7) Shape beyond one statistic**: one width statistic fixes one parameter per $\delta$; a shape
  parameter needs the profile's shape or further statistics, and for a heavy-tailed member the window
  becomes part of the definition (I-2 (b)) **[KB]**.
- **(I-8) Width against diameter on thin nodes**: the kernel's width and a thin node's diameter trade off
  ("the calibration's known weakness", D-066's implications) **[KB]**. Whatever learns the kernel must not
  select or weight nodes by a $\hat d$ computed under the kernel being learned (D-061.4).

Test for any learned estimator **[reasoning]**: train (or re-weight) under two priors that differ only along
a suspected unidentified direction; an output that follows the prior carries no information from the data
there, and must be reported as configured, not estimated.

## 6. The classical baseline on simulated data (the S2b checks)

Setting **[run, sandbox, 2026-10-21; recomputed where marked 2026-10-22]**: the renderer's own model
(Gaussian kernel, partition along vertical rays, an infinite flat tube, white noise, no camera chain);
$J$ = 12 far nodes per replicate, $d \in$ (0.6, 0.8) um, $\mu d \in$ (0.2, 0.5), $B$ = 210 gl; 1 gl of
white noise per profile sample (3 gl per pixel over 9 averaged lines, JPEG ignored) and a random
background error of 0.02 gl per plane; the pilot's depth error 0.07 um (assumed); planes $5 \le \lvert k - k^*_j\rvert \le 16$, both sides, SNR $\ge$ 3; 200 replicates per
$\gamma \in$ {0.79, 1.21}; "R" is the D-074.4 fit. None of it has run on real data.

| Variant | $\gamma$ = 0.79 | $\gamma$ = 1.21 | What it says |
|---|---|---|---|
| R (the reference) | bias +0.01 %, SD 0.0039 | bias -0.18 %, SD 0.0064 | the baseline a learned method must match or beat |
| R, one side of focus only | SD 0.0169 | SD 0.0307 | 4.3-4.8 times the SD: use both sides |
| R, $\sigma_0$ free (no anchor) | SD 0.0144; $\hat\sigma_0$ SD 0.051 um | SD 0.0261; $\hat\sigma_0$ SD 0.079 um | the far planes alone pin $\sigma_0$ poorly |
| R, anchor +0.05 um | -1.68 % | -1.48 % | the anchor's error passes into $\hat\gamma$ |
| R, axis fixed at the pilot | -0.60 % | -0.92 % | the axis depth must be free |
| R, background +0.05 gl (systematic) | +1.39 % | +1.42 % | far calibration is background-limited; noise-free, +0.1 gl gives +2.6 to +6.9 % |
| Block 10's statistic, its own SE as weights | -3.49 %; 165 planes locked onto noise | about -38 %; 647 planes | a single core fit collapses at low SNR (the two-pass fit: 0 such planes) |
| Selection by the observed amplitude, 2 gl | -1.14 % | -1.72 % | selecting on a noisy quantity biases; the predicted-dip rule gives +0.07 % and -0.09 % (SD 0.0120 and 0.0167) |
| Tilt 5 deg / 10 deg (noise-free) | +0.46 % / +1.85 % | +0.85 % / +3.04 % | the far set needs a tight tilt rule |
| Per-sign fits, $\gamma_+ - \gamma_-$ | SD 0.0097 | SD 0.0184 | the precision of the $K_\delta = K_{-\delta}$ test |
| Node bootstrap, 60 replicates | SE / SD 0.83 (jackknife 0.88); 95 % intervals cover 92 % ($\pm$1.96 SE) and 93 % ($t_{11}$) **[recomputed 2026-10-22 from `check_far_fit_mc_main.json`]** | -- | intervals at $J$ = 12 are slightly narrow |
| Fisher information, $\mu d$ 0.35, both sides, anchor | SE 0.0036; the SE at $K$ = 8, 10, 12, 14 planes is 2.71, 1.77, 1.36, 1.14 times that at 16 | SE 0.0065 | where the information is |

Facts about the near and far models from the same checks: Block 10's masked-median background reads
-0.03, -1.39, -2.50 and -2.62 gl at planes 4, 8, 12 and 16, which shrinks $\omega$ to 0.996, 0.678, 0.332
and 0.185 of itself, while a planar fit to far pixels is off by at most 0.003 gl; a line-end baseline
beyond 2 FWHM exists on a +-10 um line only up to plane 9 ($\gamma$ = 0.79) or 7 (1.21); the far law fits
noise-free core widths with residuals of at most $8 \times 10^{-3}$ um$^2$ and returns $\gamma$ within
0.03 %, with negative per-node constants (-0.010 to -0.067 um$^2$) and a fitted axis 0.009-0.022 um from
the geometric one, the shift growing with $\mu d$ **[run, sandbox]**. The scripts and logs are in
`s2b_checks_2026-10-21.zip`, sent to the user on 2026-10-21 and not in the repository (they enter
`checks/s2b_2026-10-21/` with Block 15).

## 7. Data, simulators and their cost

| Source | What it gives | Access | Cost and limits |
|---|---|---|---|
| Allen's stacks (specimen 529878215 first, D-024 (vi)) | 8-bit JPEG crops per plane: `allen_image_io.fetch_zblock(fetcher, planes, k_lo, k_hi, left, top, width, height, res0_um_px) -> (block (n_planes, H, W) uint8, ks, valid, CropFrame)`; $p_{\rm x}$ = 0.1144 um, $\Delta z$ = 0.28 um | Colab only: `api.brain-map.org` returns 403 from the sandbox; Phase II crops cached in `MyDrive/allen_cache` | one request per uncached crop; see `claude/TEEG_diameter_HPC_paths_reference.md` §1, §3 for the Drive layout |
| The pilot table, `pilot/pilot_<sp>.csv` (Cells 4a-4b) | per node: centre, heading, $\varphi$, $k^*$, the fit, flags, Allen's radius; since S2 also `calibration_set` and the pilot's columns | Drive, `MyDrive/diameters/` | the copy last read (2026-10-08) was written on 2026-10-07, 393 rows; whether it was re-run after S2 is not known here **[open]** |
| Block 10's scans (`scripts/calibrate_kernel.py real`, Cell 7) | per node and plane: `ks, z_um, omega_um2, vw_um2, area_gl_um, depth_gl, B_bar`, plus `node_id, d_hat_um, phi_rad, alpha_hat, k_star, z_sub_um` | not run yet (D-070); the far set is counted, not scanned | statistics only: no profiles, no crops. `plane_scan` computes each plane's `profiles.sample_profile(...)` and `node_background(...)` and keeps only the statistics, so a method that learns from profiles needs an export **[src: `analysis/calibration.py`, `loading/calibration_io.py`]** |
| Rendered phantoms (`render.synthetic_block`; `calibrate_kernel.calibration_phantom`; `allen_image_io.SyntheticFetcher`) | the full chain in the `fetch_zblock` contract | sandbox | Block 13's chain 5-132 s per realization (mean 48 s, 74 % rendering, 10 phantoms); a 60-degree 1 um tube over 21 planes renders in 28 s **[run, sandbox: SPEC Blocks 4, 13]** |
| `farmodel.dip(v, d, mu, delta, gamma, B, nq)` (the S2b zip) | the exact 1-D dip of an infinite flat tube under the renderer's model (no pixel grid, no camera chain) | the zip (§11); its `REPO_SRC` must point at the clone's `src` | about 4 ms per 175-sample profile on one sandbox core **[run, sandbox, 2026-10-22]** |
| The ray world (Block 9, `absorption = "ray_world"`) | geometric optics, NA 1.4, an evenly filled condenser, no diffraction, a 0.08 um post blur: an out-of-family generator | sandbox | minutes per plane for a full phantom block at $p_{\rm x}/8$ **[src: SPEC Block 9]** |
| `checks/psf_check.py` | the scalar Debye PSF and LSF (NA 1.4, $n$ 1.515, 550 nm), in focus and defocused: the source of the default table, and a non-Gaussian, heavy-tailed generator | sandbox | -- |
| Real noise | not measured (D-025); `scripts/camera_calibration.py` (Cell 5) reads Allen's JPEG tables from the cached crops | Colab | -- |

## 8. Candidate deep-learning formulations (none decided)

All rows are the assistant's proposals **[reasoning]** unless a source is named.

| | Candidate | Learns | Trained on | Keeps D-067.2? | Main risk | First test |
|---|---|---|---|---|---|---|
| C1 | A learned far-plane background (a U-net, as BGnet) | the background of a far plane's crop, per pixel | rendered crops with structured backgrounds (other neurites' blurred shadows, tissue texture), kernels drawn from a prior over $\gamma$ | yes | BGnet's authors state their method works because the PSF model is known **[PubMed, full text]**; here the kernel is what is being calibrated, so a training set at one kernel would bias $\hat\gamma$ | $\hat\gamma$ with the learned background against the planar far-pixel fit of D-074 on renderer data with structured backgrounds; the budget is +1.4 % per +0.05 gl (§6) |
| C2 | A learned per-plane width statistic (a 1-D CNN on the profile, with an SE head) | $\hat\omega_{j,k}$ and its SE, or a reject flag | `farmodel` or renderer profiles over $d$, $\mu$, $\delta$ and the noise | yes | it defines its own statistic: on non-Gaussian real profiles its output need not be the core width that Eq. 4 and the far law assume | against S2 at 1 and 2 gl: collapse rate, bias of $\hat\gamma$, SE calibration; then on ray-world and Debye profiles |
| C3 | Differentiable analysis by synthesis: procedure §3.4 step 2 with automatic differentiation | the knot values of $\sigma_{\rm r}$ (or $G$) and $\gamma$, jointly with each node's $d$, $\mu$, $z_{\rm ax}$, $v_0$ and background planes | no training set: the calibration nodes' own profiles or crops | yes, with the Gaussian parameterisation | the width-diameter trade-off (I-8); a Gaussian fitted to profiles with $\lvert v\rvert^{-2}$ tails compromises between core and tails unless the loss weights the core; at small scale scipy's `least_squares` already does this (ecosystem first) | recover a known table from rendered thin phantoms, where the first stage fails near focus (I-6) |
| C4 | Amortised neural posterior estimation (simulation-based inference) | $q(\theta \mid \text{scans})$ for $\theta$ = (knot values, $\gamma$), the nuisances marginalised; a per-node embedding network pooled over nodes | simulated scans from a prior over $\theta$ and the nuisances | yes ($\theta$ parameterises the Gaussian law) | the posterior is only as right as the simulator: a Gaussian-kernel simulator facing non-Gaussian real optics can give confidently wrong posteriors (two PubMed queries on misspecified neural posterior estimation returned no record); unidentified directions come back as the prior (I-3) | coverage in family (simulation-based calibration), then ray-world and Debye scans out of family, then posterior predictive checks on real nodes |
| C5 | A non-Gaussian kernel: a Gaussian mixture $K_\delta = \sum_m \pi_m(\delta)\,G_{\sigma_m(\delta)}$, the SGT through the inverse Abel transform (D-066.1), or a pupil-function model fitted as in uiPSF | the kernel's shape at each $\delta$ | as C3 or C4 | no: it reopens I-002 and D-066 | everything D-066 listed (Block 4, the FFT split, padding, gate 1, Block 10's statistic, the simulation hash); a mixture of Gaussians would keep the Gaussian backends as a sum of Gaussian blurs per slab | a labelled comparison on the benchmark; production use is the user's decision |

Rules that hold for every candidate **[reasoning, except where a source is named]**:

- **No circularity.** Calibration nodes are chosen by Allen's diameter (D-061.4), never by a $\hat d$
  computed under the kernel being learned; nodes that calibrate the kernel do not validate the corrected
  diameters; real data are split by cell or specimen, never by node or plane.
- **The camera chain stays outside the kernel** (§4.5): a model trained on real crops must not absorb
  JPEG, quantisation or pixel integration into $K_\delta$.
- **Selection on noisy quantities biases** (§6: -1.1 to -1.7 % at 2 gl); a learned quality gate must
  select on predicted signal, not on the plane's own noisy amplitude.
- **Scale.** The renderer is custom code ("no library renders a depth-dependent blur", SPEC Block 4)
  **[src]**; for flat calibration nodes the 1-D integral of `farmodel` is exact for the renderer's model
  and hundreds of times cheaper per plane (about 4 ms against 1.3-1.6 s per rendered plane in SPEC Block
  4's timings, for blocks smaller than the far scan needs) **[run, sandbox; the ratio is reasoning]**;
  whole crops need the 2-D renderer, on a GPU for training-set sizes.
- **Validation** of an amortised posterior: sample from the inferred model and look for systematic
  differences from the data, and run several initialisations, as Gonçalves et al. advise for SNPE
  **[PubMed, full text]**.

## 9. Acceptance tests (proposed)

Proposals **[reasoning]**; the reference numbers are §6's **[run, sandbox]**.

| Test | How | Reference or pass rule |
|---|---|---|
| T0 What accuracy matters | render a reduced phantom set under two kernels ($\gamma$ 0.79 against 0.85; one knot +-10 %) and compare $\hat d/d$, or reduced tables $\hat b$ (procedure §3.10's kernel rows) | not run: the target accuracy is unknown until this runs **[open]** |
| T1 Recovery in family | known table and $\gamma \in$ {0.79, 1.21}, $J$ = 12, 1 and 2 gl, at least 100 replicates | R: bias +0.01 % / -0.18 %, SD 0.0039 / 0.0064 at 1 gl; at 2 gl with the predicted-dip rule, SD 0.0120 / 0.0167 |
| T2 Honest uncertainty | coverage of nominal 95 % intervals over the replicates; simulation-based calibration for C4 | the node bootstrap: 92 % ($\pm$1.96 SE), 93 % ($t_{11}$) at $J$ = 12 |
| T3 Out of family | ray-world or Debye-LSF scans, the target being their own noise-free core-width law | a learned method must not be further off than the classical fit |
| T4 Real data | per-sign fits, shallow vs deep nodes, rendered against real profiles plane by plane, agreement with the classical fit within the SEs | procedure §3.10 |
| T5 Downstream | a reduced table under the learned kernel against the classical one | procedure §3.10, "Kernel shape" row |

## 10. Where a result lands in the code

| Target | Fields or files | Consequence |
|---|---|---|
| A Gaussian law (inside D-067.2) | `RendererConfig.kernel_table_delta_um`, `kernel_table_sigma_um`, `sigma_r0_um` (= the first value), `kernel_continuation`, `kernel_continuation_slope` | every one is in the simulation signature: a new kernel is a new simulation hash, the merge refuses rows rendered under the old one, and the table is rendered again; per D-067.1 the production waits for the final configuration. A finished table is checked at application only against the estimator hash, which the kernel does not enter, so a table rendered with an old kernel would still be applied: its record (`full_signature`) is the place to check **[src: `config.py`, `build_table.merge`, `analysis/table.py`; KB]** |
| The calibration record | `loading/calibration_io.save_calibration(path, growth, kernel, kernel_error, area_ratio, meta)`; Block 15's planned output adds $\hat\gamma$, its SE, range and residuals beside the near table | the calibration settings are in no signature **[src]** |
| Handing it to davinci | the configuration made in Colab (Cell 8, `config_production.json`), passed as `--config-json` to `build_table.py`; every task prints the simulation hash | D-061 (f) **[KB]** |
| A non-Gaussian kernel | `config.KERNEL_FAMILIES` holds `"empirical"`, which raises `NotImplementedError`; D-066's "What it implies" lists the code to change | needs the user's decision first (I-002) |
| New code | a SPEC entry first, then modules and smoke tests per the scientific-coding skill (plus hpc-python-compat and hpc-git-delivery for anything that runs on davinci: pure-ASCII `.py`, LF shell and PBS files); proposal: a separate package or `src/allen_diameter/learn/`, so that `MeasureConfig` and the estimator hash stay untouched | project instructions; **[reasoning]** for the placement |

## 11. Commands

Clone and check (in an existing clone, `git fetch origin && git checkout sci/diameter-pipeline && git pull
--ff-only` instead):

```
cd /home/claude && git clone https://github.com/Leonardodm00/Towards-EEG.git towards-eeg && cd towards-eeg && git checkout sci/diameter-pipeline && git log --oneline -3
```

Every suite once, then the hashes (the suites take more than 10 min in the sandbox; run them in the
background or one by one):

```
cd "/home/claude/towards-eeg/Passive Features/Diameter Re-measurement" && for f in tests/smoke/test_smoke_*.py; do python3 "$f" > /dev/null 2>&1 && echo "pass $f" || echo "FAIL $f"; done
PYTHONPATH=src python3 -c "from allen_diameter.config import DiameterConfig as D; c = D(); print(c.signature_hash('estimator'), c.signature_hash('simulation'))"
```

The optics document v1.5 from `main`:

```
cd /home/claude/towards-eeg && git fetch origin && git show "origin/main:Passive Features/Diameter Re-measurement/docs/TEEG_microscope_optics_oil_immersion_2026-10-04.md" > /tmp/optics_v1.5.md
```

Block 10 on rendered phantoms, then its fit (the Phase I path of `calibrate_kernel.py`; outputs outside the
repository, since `cal/` is not git-ignored). About 30 s for 4 phantoms with 2 workers; it prints the growth
fit and a first-stage table that is not increasing, with the tool's own warning "do not render with it
before step 2" **[run, sandbox, 2026-10-22]**:

```
cd "/home/claude/towards-eeg/Passive Features/Diameter Re-measurement" && mkdir -p /tmp/kernel_cal && python3 scripts/calibrate_kernel.py synthetic --n 4 --seed 5 --out /tmp/kernel_cal/scans_syn.json --workers 2 && python3 scripts/calibrate_kernel.py fit --scans /tmp/kernel_cal/scans_syn.json --out /tmp/kernel_cal/calibration_syn.json
```

The S2b checks, after the user uploads `s2b_checks_2026-10-21.zip`. `farmodel.py` reads the package from
`REPO_SRC` = `/home/claude/towards-eeg/Passive Features/Diameter Re-measurement/src` (edit it if the clone
lives elsewhere). The last command prints 6.527, the dip depth in grey levels of a 0.7 um node at
$\mu d$ = 0.35, 2.8 um from focus, $\gamma$ = 0.79 **[run, sandbox, 2026-10-22]**:

```
mkdir -p /home/claude/s2b && cd /home/claude/s2b && unzip -q <upload>/s2b_checks_2026-10-21.zip && cd s2b_checks_2026-10-21 && python3 -c "import numpy as np, farmodel as fm; v = np.linspace(-10, 10, 175); print(round(float(fm.dip(v, 0.7, 0.5, 2.8, 0.79).max()), 3))"
```

PyTorch and `sbi` are not installed in the sandbox; the package index offers torch 2.14.1 (`pip index
versions torch`), not installed or tested here **[open]**:

```
pip install --break-system-packages torch sbi
```

## 12. Literature and software leads

| Lead | Access | What it offers here |
|---|---|---|
| Liu S. et al. 2024, uiPSF, *Nat Methods* 21:1082-93, PMID 38831208, PMC12330227, [DOI](https://doi.org/10.1038/s41592-024-02282-x) | **[PubMed, full text]** (introduction and the inverse-modelling methods read) | inverse modelling through a differentiable forward model (TensorFlow's automatic differentiation): in the spatial domain a voxel PSF, with every bead's position, photons and background, fitted by L-BFGS on a mean-squared-error loss with regularisers; in the Fourier domain a pupil function (pixelated or Zernike) with a likelihood loss; from bead stacks or directly from blinking emitters whose depths are fitted: the closest published analogue of C3 |
| Möckl L. et al. 2020 (online 2019), BGnet, *PNAS* 117:60-67, PMID 31871202, PMC6955367, [DOI](https://doi.org/10.1073/pnas.1916219117) | **[PubMed, full text]** | a U-net trained on simulated PSF images with Perlin-noise structured backgrounds that returns the background per pixel; subtracting it improved localisation precision against a constant background, on simulated and experimental data; "the method works because the PSF model is known"; data and code on request: the analogue of C1, with its caveat |
| Gonçalves P.J. et al. 2020, SNPE, *eLife* 9:e56261, PMID 32940606, PMC7581433, [DOI](https://doi.org/10.7554/eLife.56261) | **[PubMed, full text]** (introduction, method overview, discussion and limitations, code availability read; the application sections not) | neural posterior estimation for mechanistic neuroscience models: a conditional density network trained on simulations, applied to new data without new simulations (amortised), able to learn summary features from raw data; validation by sampling the inferred model against the data and by several initialisations; parameter spaces above about 30 dimensions called challenging; a PyTorch toolbox (the `sbi` package, by name **[from memory]**): the basis of C4 |
| Cranmer K., Brehmer J., Louppe G. 2020, "The frontier of simulation-based inference", *PNAS* 117:30055-62, PMID 32471948, PMC7720103, [DOI](https://doi.org/10.1073/pnas.1912789117) | **[PubMed, abstract only]**: the PMC record returned no body text | a review of simulation-based inference; nothing cited from it |
| Maloberti et al. 2025, *Biomed Opt Express* 16:1310-26, PMID 40322002, PMC12047720, [DOI](https://doi.org/10.1364/BOE.551278) | **[PubMed, full text]** (read for D-074) | a non-numeric analogy only: planes on both sides of focus improve a joint PSF estimate in SMLM |
| Droste I. et al., calibration-free estimation of field-dependent aberrations for SMLM, bioRxiv 2024, [DOI](https://doi.org/10.1101/2024.12.11.627909) (PMC11661230); published in *Optica*, [DOI](https://doi.org/10.1364/OPTICA.560721) | **[preprint, abstract only]**; the published version not read | a few-parameter physical PSF model (vectorial, nodal aberration theory) estimated from the specimen's own single-molecule data without bead calibration, compared with a deep-learning approach: the analogue of a physical $K_\delta$ (I-002 (c)) |
| Quan Y. et al. 2024, Gaussian kernel mixture for defocus deblurring, *IEEE TPAMI* 46:11361-77, PMID 39255177, [DOI](https://doi.org/10.1109/TPAMI.2024.3457856) | **[PubMed, abstract only]** | parameterises spatially varying defocus PSFs as pixel-wise Gaussian kernel mixtures, motivated by their isotropy: the C5 mixture idea in photography |
| Oberlaender M. et al. 2009, *J Microsc*, PMID 19220694, [DOI](https://doi.org/10.1111/j.1365-2818.2009.03118.x) | **[PubMed, abstract only]** (I-002 (c)) | transmitted-light stacks of biocytin-filled rat dendrites with an oil objective, deconvolved with a model PSF from refraction between media and pupil diffraction; Allen's mounting medium is unknown, so nothing is transplanted |

Searches run for this handoff (PubMed, 2026-10-22): physics-informed neural network deconvolution in
microscopy (1 record: an unrolled Richardson-Lucy network for multi-cluster PSFs, PMID 42271986, not
relevant here); brightfield microscopy with deep learning and defocus (1 record: nuclei detection from
defocused brightfield images, PMID 33796381, not relevant); deep-learning PSF estimation with aberrations
(3 records, among them Droste and NeuroDecon, PMID 41009339); uiPSF; SNPE; in-situ PSF retrieval (Xu F. et
al. 2020, INSPR, *Nat Methods* 17:531-40, PMID 32371980, [DOI](https://doi.org/10.1038/s41592-020-0816-x):
exists, applied among others to dendrites in brain tissue, not read); two queries on misspecification in simulation-based or amortised
inference, 0 records. bioRxiv: only the published-version lookup for Droste (the connector has no keyword
search). Not searched: arXiv, where most of the machine-learning literature on simulation-based inference
and differentiable rendering sits **[open]**.
