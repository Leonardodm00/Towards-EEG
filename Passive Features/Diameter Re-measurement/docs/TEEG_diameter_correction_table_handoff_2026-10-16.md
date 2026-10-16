# Handoff: the bias (correction) table under the settled estimator (diameter chat -> table chat)

| Date | Change |
|---|---|
| 2026-10-16 | Created at the close of the estimator study (2026-10-13 to 2026-10-16), for the chat that builds the correction table. The estimator is **D-052** (decision log, project knowledge). Code facts read on `sci/diameter-pipeline` at `9e3296c` **[src]**; numbers re-read on 2026-10-16 from the result files of the sandbox runs of 2026-10-15 **[run, sandbox]**; three of those checks re-run from the repository copy on 2026-10-16 with the same output. Fact-checked the same day by an independent agent against the code and the result files; its corrections are in (the cause of the two IRLS failures, the kernels' shape counts, the $\bar B$ mask, the source of $\pm 6$ planes, Q3's mechanism, Q6's numbers without the collapses, the strict `high_failure` rule, the search minima as upper bounds), and §10 gained Block 10's dependence on `measure_node`. |

Basis tags: **[src]** read in the named file at `9e3296c`; **[run, sandbox]**
executed in the assistant's sandbox; **[user]** said or decided by the user;
**[KB]** project knowledge (the decision log); **[reasoning]** the assistant's
inference, not checked; **[open]** neither decided nor verified.

Plainly: the per-node measurement is now fixed. The table chat's job is to
make the bias table measure its phantoms with exactly that measurement,
build the table on the axes the decisions ask for, and get it through the
gates and onto davinci.

## 1. First actions in the table chat

1. Read, in this order: **D-052**; D-024, D-030, D-031, D-023 and D-043 (a)
   in `TEEG_decisions_and_ideas_log.md` (project knowledge); `specs/SPEC.md`
   §2.2, §2.4, Blocks 5-7 and §8; the procedure,
   `docs/TEEG_diameter_bias_table_procedure_2026-10-04.md` §3.5-3.11.
   **ID collision:** the repo (SPEC.md, docstrings, config comments) calls the
   gradient-energy focus rule "D-030"; in the log that decision is **D-035**,
   and the log's D-030 is "every real node gets its own estimate whatever its
   optical depth" **[KB]**.
2. Get the branch (in an existing clone, `git pull --ff-only` instead):

   ```
   cd /home/claude && git clone https://github.com/Leonardodm00/Towards-EEG.git towards-eeg && cd towards-eeg && git checkout sci/diameter-pipeline && git log --oneline -3
   ```

   Head: `9e3296c`, plus the commit carrying this handoff and
   `checks/estimator_2026-10-15/` if the user agreed to push it.
3. Run every suite once before changing anything (23 suites passed at
   `9e3296c`, D-047 **[KB]**):

   ```
   cd "Passive Features/Diameter Re-measurement" && for f in tests/smoke/test_smoke_*.py; do python3 "$f" > /dev/null 2>&1 && echo "pass $f" || echo "FAIL $f"; done
   ```

4. Settle §7 with the user before writing any computational logic
   (scientific-coding §2), in batches of multiple-choice questions. Q1, Q2
   and Q4 shape the code of Blocks 13-14; the others are settings or later
   steps.
5. Blocks 12-14 of §8, each through the block protocol (SPEC entry first,
   code, verify twice, commit), then the gates of §8.4.

## 2. Where things are

| What | Where | Note |
|---|---|---|
| Repository | `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline` | head `9e3296c` on 2026-10-16, in sync with origin **[run]** |
| Workstream | `Passive Features/Diameter Re-measurement/` | `src/allen_diameter/` (`config.py`, `analysis/`, `model/`, `loading/`, `plotting/`), `scripts/`, `tests/smoke/`, `configs/gate2_reduced.json`, `docs/`, `checks/`, `notebooks/phase2_colab.ipynb` |
| Spec | `specs/SPEC.md` (repo root) | Coverage names the workstream; Block 6 = phantoms and table, Block 7 = inversion, flags, fill |
| Decision log | project knowledge, `TEEG_decisions_and_ideas_log.md` | D-001 to D-052; next free D-053, I-003 (2026-10-16) |
| Documents | `docs/` of the workstream (D-020, not project knowledge) | procedure and mathematics (2026-10-04), implementation handoff (2026-10-06), Colab runbook v17, this file |
| Evidence of D-052 | `checks/estimator_2026-10-15/` with its `README.md` | the check scripts and their result files (JSON, NPZ) |
| Iteration code today | `scripts/plane_differences.py` | `ITER_LINE_MULT` = 3.0 (line 114), `node_stack` (148), `line_profiles` (252), `block_background` (270), `check_iteration` (430), `iterate_planes` (447), `refit_at_plane` (497), `gradient_iteration` (547) **[src]** |
| Kernels and fit variants | `model/blur_kernels.py`, `analysis/fit_variants.py`, `tube_model.model_profile_kernel` | 6 scipy families: `gauss` and `lorentz` without a shape parameter (`NO_SHAPE`), `voigt`, `student`, `gennorm`, `skewnorm` with one; `_Layout` holds at most one **[src]** |
| Table code | `analysis/phantoms.py`, `analysis/table.py`, `analysis/invert.py`, `analysis/fill.py`, `loading/table_io.py`, `scripts/build_table.py`, `scripts/end_to_end.py`, `scripts/pbs/build_table.pbs` | **[src]** |
| Cluster | davinci-1, clone `/davinci-1/home/ldellamea/TEEG/Towards-EEG`, tables under `/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_tables/<RUN_TAG>` | commands, knobs, exit codes, logs: project doc `claude/TEEG_diameter_HPC_paths_reference.md` §2 **[KB]**; no diameter job has run there yet **[KB]** |
| Real-data runs | Colab, `MyDrive/diameters/pilot/...` (Cells 4j-4m) | Cell 4m (the variants' own iterations) has not run on Allen's data; D-052's fit has never run on Allen's data **[KB]** |

## 3. The estimator the table is built for (D-052)

### 3.1 Inputs of a node

For a dendrite node $j$, real or the middle node of a phantom:

- $r_j$, the radius its branch carries (um): Allen's on a real node, the true
  $d/2$ on a phantom today (§6);
- $k_{{\rm SWC},j}$, the plane nearest its traced depth (planes 0.28 um apart);
- Block 5's **pilot** at $k^*_j$ (unchanged, D-035): the centre, the heading
  $\theta_j$ and the tilt $\varphi_j$ from the line fit of handoff Eqs. 3-5,
  whose centres come from Block 5's Gaussian fit at each window node's $k^*$
  (`node_pipeline._node_pass`) **[src]**;
- $v_n$, $n = 1, \ldots, N$, the profile offsets (um): $N = 2\,\mathrm{round}(3/0.1144) + 1 = 53$
  (`profile_half_um` 3, `profile_step_um` 0.1144) **[src]**;
- $I_j(v_n; k)$, the profile in plane $k$ through the pilot's centre across
  its heading, averaged along the branch as Block 5 does, and $\bar B_j(k)$
  (gl), the D-018.1 masked median of plane $k$ about the node
  (`refit_at_plane`, every input but the plane the pilot's) **[src]**.

### 3.2 Kernel: skewed generalized $t$, two-piece form

With $B$ the beta function, $\Gamma$ the gamma function and
$\kappa = (\beta, \xi, \zeta)$, for every $x \in \mathbb R$:

$$h_{\beta,\xi}(x) = \frac{\beta\,\xi^{1/\beta}}{2\,B(1/\beta,\,1/\xi)}\,\bigl(1 + \xi\,\lvert x\rvert^{\beta}\bigr)^{-(1/\xi + 1/\beta)} \ \ (\xi > 0), \qquad h_{\beta,0}(x) = \frac{\beta}{2\,\Gamma(1/\beta)}\,e^{-\lvert x\rvert^{\beta}}, \tag{H1}$$

$$f(x \mid \kappa) = \frac{2}{\zeta + 1/\zeta}\begin{cases} h_{\beta,\xi}(x/\zeta), & x \ge 0,\\ h_{\beta,\xi}(\zeta x), & x < 0,\end{cases} \tag{H2}$$

a density of unit integral with its mode at 0. Its FWHM and the kernel of
FWHM $w$ (um), for every $z \in \mathbb R$:

$$W(\kappa) = \Bigl(\zeta + \frac1\zeta\Bigr)\Bigl(\frac{2^{\xi\beta/(\beta+\xi)} - 1}{\xi}\Bigr)^{1/\beta} \ (\xi > 0), \quad W(\beta, 0, \zeta) = \Bigl(\zeta + \frac1\zeta\Bigr)(\ln 2)^{1/\beta}, \qquad g_{w,\kappa}(z) = \frac{W(\kappa)}{w}\, f\Bigl(\frac{W(\kappa)}{w}\,z \Bigm| \kappa\Bigr). \tag{H3}$$

Members: $(2, 0, 1)$ Gaussian; $(1, 0, 1)$ Laplace; $(2, 2/\nu, 1)$ Student
$t_\nu$, so $(2, 2, 1)$ Lorentzian; $(\beta, 0, 1)$ the generalized normal
(box-like as $\beta$ grows); $\zeta \ne 1$ their two-piece skewed versions,
right-to-left half-width ratio $\zeta^2$. Tails: $\lvert z\rvert^{-(1 + \beta/\xi)}$
for $\xi > 0$. In code (`sgt_kernel.py`, the result files) the shape is
`(p, r, gamma)` = $(\beta, \xi, \zeta)$; for small $\xi$ the factor
$2^{e} - 1$ must be computed as `expm1(e * ln 2)` (cancellation found
2026-10-15: FWHM off by 6e-8 without it) **[run, sandbox]**.

### 3.3 Model and fit at a plane

With $\theta = (d, \mu, v_0, w, \beta, \xi, \zeta)$ and $\alpha = \mu d/\cos\varphi_j$,
for every $v \in \mathbb R$ (D-046's model, `tube_model.model_profile_kernel`):

$$I_{\rm model}(v \mid \theta;\, \varphi_j, \bar B_j(k)) = \bar B_j(k)\Bigl[1 - \frac d2 \int_{-\pi/2}^{\pi/2} \bigl(1 - e^{-\alpha\cos\psi}\bigr)\cos\psi\; g_{w,\kappa}\Bigl(v - v_0 - \frac d2 \sin\psi\Bigr)\, d\psi\Bigr]. \tag{H4}$$

For weights $\omega = (\omega_1, \ldots, \omega_N)$:

$$Q_j(\theta \mid k, \omega) = \frac12 \sum_{n=1}^{N} \omega_n \Bigl[\frac{I_{\rm model}(v_n \mid \theta;\, \varphi_j, \bar B_j(k)) - I_j(v_n; k)}{\bar B_j(k)}\Bigr]^2, \qquad \omega^{(t)}_n = \exp\bigl(-\lvert v_n - \hat v_0^{(t)}\rvert / \lambda\bigr). \tag{H5}$$

$\hat\theta^{(0)}_j(k) = \arg\min_{\theta \in \Theta} Q_j(\theta \mid k, \mathbf 1)$
(unweighted); then
$\hat\theta^{(t+1)}_j(k) = \arg\min_{\theta \in \Theta} Q_j(\theta \mid k, \omega^{(t)})$,
$\hat v_0^{(t)}$ the centre of iterate $t$, until $d$ and $v_0$ both move by at
most $10^{-4}$ um, at most 30 weighted solves (D-046's IRLS). The node's fit at
plane $k$ is the last iterate, $\hat\theta_j(k)$.

### 3.4 Plane: the Allen-guided iteration driven by this fit

$d^{(0)}_j = 2 r_j$. For rounds $i = 1, \ldots, 5$, the line of half-length
$h_{j,i} = m\, d^{(i-1)}_j / 2$, $m = 3$ (D-044), through the traced node
across the pilot's heading, and

$$k_{j,i} = \arg\max_{k \in K_j} G^{(h_{j,i})}_{j,k}, \qquad G^{(h)}_{j,k} = B_{j,k}^{-2} \int_{-h}^{h} \Bigl(\frac{\partial \tilde I_{j}(v; k)}{\partial v}\Bigr)^2 dv, \qquad K_j = \{k_{{\rm SWC},j} - 6, \ldots, k_{{\rm SWC},j} + 6\}, \tag{H6}$$

$\tilde I_j(\cdot\,; k)$ the line's profile smoothed by `focus_smooth_px`
(1 px), $B_{j,k}$ `block_background`'s background of plane $k$ (masked
radius + 4 um, shrinking) (`focus.plane_gradient_energies`) **[src]**. A plane
already picked ends the iteration (`converged` if it is the previous round's,
`cycle` otherwise); else $d^{(i)}_j = \hat d_j(k_{j,i})$ from §3.3. A line
reaching within 2 pixels of the square's edge rebuilds a larger square
(`restack`) **[src]**. Output: $k^{\rm it}_j$, the last plane picked, and
$\hat\theta_j(k^{\rm it}_j)$, with $\hat\alpha_j = \hat\mu_j \hat d_j / \cos\varphi_j$.
This $\hat d_j$ is the $\hat d_i$ the inversion solves for, and phantoms go
through the same chain (D-052.4; procedure §3.7) **[KB]**.

### 3.5 Constants

| Quantity | Value | Source |
|---|---|---|
| $d$ bounds | [0.05, 6] um | `fit_d_bounds_um` **[src]** |
| $\mu$ bounds | [0, 20] /um | `fit_mu_bounds_per_um` **[src]** |
| $v_0$ bounds | [-1, 1] um | `fit_v0_bounds_um` **[src]** |
| $w$ bounds | [0.1, 2] um (FWHM) | `fit_variants.FITTED_FWHM_BOUNDS_UM`, a module constant, not in `MeasureConfig` **[src]** |
| $\beta$, $\xi$, $\zeta$ bounds | [1, 10], [0, 2], [$e^{-0.7}$, $e^{0.7}$] = [0.497, 2.014]; searched in $(\log_{10}\beta, \xi, \ln\zeta)$ | `identifiability_sgt.LB_Q`, `UB_Q` **[run, sandbox]** |
| $\lambda$ | 0.5 um | D-052 **[user]** |
| IRLS stop, cap | $10^{-4}$ um on $d$ and $v_0$; 30 | `fit_variants.IRLS_TOL_UM`, `IRLS_MAX_ITER` (module constants) **[src]** |
| Solver | `least_squares`, `method="trf"`, `jac="2-point"`, `x_scale="jac"`, `ftol = xtol = gtol` = 1e-10 | `fit_tol` **[src]** |
| Per-solve `max_nfev` | the check: 3000; `fit_variants`' rule: max(`fit_max_nfev` 300, 100 x n_params) = 700 for 7 parameters | **[src]**; not settled by D-052: make it a `MeasureConfig` field and decide it (the two IRLS failures of §4.4 were solves that hit 3000) |
| Quadrature | $N_q = \max(64, \lceil 6 \cdot 6 / \sigma_{\min} \rceil)$ = 848, $\sigma_{\min} = 0.1/2.3548$ um | `fit_variants.n_nodes_for`, `fit_quad_*` **[src]** |
| Start points | Gaussian of fitted width x shapes (2, 0, 1), (1, 0, 1), (5, 0, 1), (2, 1, 1); plus Block 5's $(d, \mu, v_0)$ at $w$ 0.7 um, Gaussian shape; least objective wins | `identifiability_sgt.fit_sgt` **[run, sandbox]**; PROVISIONAL |
| Iteration | $m$ = 3; at most 5 rounds; `bg_margin_um` 4.0; $K_j = k_{\rm SWC} \pm 6$ | D-044, D-042 **[KB]**; the first three are `gradient_iteration`'s defaults, the $\pm 6$ is `node_stack(planes_half=6)`, `run(planes_half=6)` and the CLI's `--planes-half 6` **[src]** |
| Pilot | Block 5 unchanged: $k^*$ by gradient energy over $\lvert v\rvert \le r + 0.5$ um, $\pm 3$ planes widened for tilt, Gaussian fit at $\sigma_{\rm fit}$ = 0.099 um | `planes_half` 3, `focus_grad_margin_um` 0.5, `sigma_fit_um` **[src]** |

### 3.6 Default and comparisons

The deliverable is D-052's fit at D-052's plane. Labelled comparisons only:
Block 5's Gaussian at $\sigma_{\rm fit}$ at $k^*$ (today's estimate, keep it as a
column, e.g. `d_pilot_um`), the unweighted SGT, the Gaussian of fitted width.
A smoke check must assert the defaults (project-decision-log: a default is a
decision).

## 4. Evidence behind D-052 (simulation only)

### 4.1 Monte Carlo

Profiles from (H4) at the 53 offsets: tubes of 0.6 and 1.5 um, $\mu$ = 1 /um,
$v_0$ = 0.1 um, $\varphi$ = 0.2 rad, $\bar B$ = 180 gl, true kernels of FWHM
0.5 um (Gaussian; Student $t_4$ = sgt(2, 0.5, 1); generalized normal
$\beta$ = 4), Gaussian noise of SD 2 gl, 40 draws per truth, seed 20261017.
Median $\pm$ SD of $\hat d$ (um); in brackets, fits stopped at $\mu$'s upper
bound **[run, sandbox: `ident_sgt/identifiability_sgt.json`]**:

| Truth | D-052 ($\lambda$ = 0.5) | SGT unweighted | Gaussian, $w$ fitted | Block 5 (Gaussian, $w$ 0.233) |
|---|---|---|---|---|
| 1.5 um, Gaussian | 1.498 ± 0.012 | 1.499 ± 0.012 | 1.499 ± 0.011 | 1.583 ± 0.007 ($t_{100}$) |
| 1.5 um, $t_4$ | 1.493 ± 0.015 | 1.494 ± 0.015 | 1.530 ± 0.014 | 1.639 ± 0.012 |
| 1.5 um, gen. normal 4 | 1.500 ± 0.012 | 1.499 ± 0.011 | 1.495 ± 0.011 | -- |
| 0.6 um, Gaussian | 0.579 ± 0.182 [8] | 0.564 ± 0.182 [8] | 0.624 ± 0.215 [14] | 0.830 ± 0.019 ($t_{100}$) |
| 0.6 um, $t_4$ | 0.539 ± 0.195 [16] | 0.511 ± 0.194 [15] | 0.231 ± 0.083 [39] | 0.914 ± 0.028 |

Block 5's column is from the Student-only run (seed 20261016, its own draws,
truths $t_{100}$ and $t_4$; `ident/identifiability.json`). 10 % / 90 %
quantiles of D-052's fit: 0.241 / 0.767 um (0.6, Gaussian), 0.232 / 0.677 um
(0.6, $t_4$) **[run, sandbox]**. Read: thick tubes are measured without the
kernel-shape bias the Gaussian has (1.530 on $t_4$); thin tubes keep an SD of
0.18-0.20 um and a collapse branch at $\hat d$ 0.23-0.25 um, which the table
can report but not remove **[reasoning]**.

### 4.2 Impostor search (is $d$ identified by one profile?)

$\Delta\chi^2_{\min}(d \mid \theta_{\rm true}, \sigma_\varepsilon)$ = the
smallest $\sigma_\varepsilon^{-2} \sum_n [I_{\rm model}(v_n \mid d, \cdot) - I_{\rm model}(v_n \mid \theta_{\rm true})]^2$
over all other parameters, the shape included, at $d = \rho\, d_{\rm true}$,
$\rho \in$ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.35, 1.5}; five
truth kernels (Gaussian, $t_4$, Laplace, generalized normal 4, skewed Gaussian
$\zeta$ = 1.5) **[run, sandbox]**:

- 0.6 um truths, $\sigma_\varepsilon$ = 1 gl: 9 pairs (truth, $\rho$) with
  $\lvert\rho - 1\rvert \ge 0.2$ reach $\Delta\chi^2_{\min} < 1$ (Gaussian 4,
  $t_4$ 4, Laplace 1); 17 at 2 gl. Example: the Gaussian truth is matched by
  $d$ = 0.30 um at $\Delta\chi^2$ = 0.026 (1 gl), max $\lvert\Delta I\rvert$ =
  0.064 gl. None is exact: the nine differ from their truth by 0.027-0.49 gl
  at their worst sample (0.027-0.075 gl for the Gaussian truth).
- 1.5 um truths: no impostor was found below $\Delta\chi^2$ = 20.8 at
  $\rho$ = 0.95, 1.05 or below 67.6 at 0.9, 1.1 (1 gl), for all five kernels,
  the Laplace weakest.
- The minima come from a local multi-start search, so each reported value is
  an upper bound on the true $\Delta\chi^2_{\min}$: the thin-tube impostors
  are certain, the thick-tube margins are "none found" **[reasoning]**. The
  one cross-check: at the truth both runs share ($t_4$) the SGT minimum was
  below the Student run's at every $\rho$ (worst excess $-0.0035$), as nesting
  requires **[run, sandbox]**.
- Exact: a larger kernel family can only lower $\Delta\chi^2_{\min}$ (a
  minimum over a larger set), so more shape freedom cannot make thin nodes
  identifiable **[reasoning, exact]**.

### 4.3 Kernel pinned from one thick node (a labelled comparison, Q7)

For thin draw $n$, $(w, \kappa)$ fixed at the unweighted seven-parameter fit of
thick draw $n$ (1.5 um, the same truth kernel, its own noise), and only
$(d, \mu, v_0)$ fitted on the 0.6 um draw: 0.614 ± 0.052 and 0.604 ± 0.060 um unweighted,
0.607 ± 0.053 and 0.601 ± 0.058 um at $\lambda$ = 0.5 (Gaussian and $t_4$
truths), no fit at $\mu$'s bound, smallest 0.42 um **[run, sandbox:
`ident_sgt/pinning_sgt.json`, re-run 2026-10-16]**. Whether a real thick
node's kernel carries over to thin nodes of other depths is untested **[open]**.

### 4.4 Bound hits and convergence (feeds Q2)

Per 40 fits, unweighted / $\lambda$ = 0.5 **[run, sandbox]**:

| Truth | $\xi$ = 0 | $\xi$ = 2 | $\beta$ = 10 | $\mu$ = 20 | converged |
|---|---|---|---|---|---|
| 1.5, Gaussian | 20 / 26 | 2 / 3 | 0 / 0 | 0 / 0 | 40 / 40 |
| 1.5, $t_4$ | 6 / 8 | 3 / 5 | 0 / 0 | 0 / 0 | 40 / 38 |
| 1.5, gen. normal 4 | 16 / 19 | 9 / 3 | 7 / 5 | 0 / 0 | 40 / 40 |
| 0.6, Gaussian | 22 / 19 | 10 / 9 | 2 / 5 | 8 / 8 | 40 / 40 |
| 0.6, $t_4$ | 10 / 12 | 8 / 9 | 0 / 0 | 15 / 16 | 40 / 40 |

"Converged": unweighted, at least one start's solve succeeded; at
$\lambda$ = 0.5, the IRLS rule was met. No fit reached a bound of $d$, $v_0$ or
$w$. $\zeta$ at a bound: at most 4 per 40; $\beta$ = 1: 2 per 40 on the thin
$t_4$ truth. $\xi$ = 0 is the generalized-normal edge of the family, not a
failure. The two unconverged weighted runs (draws 36 and 39 of the thick
$t_4$ truth) did not reach the 30-solve cap: one weighted solve hit
`max_nfev` = 3000 (scipy status 0), and the run stopped there; draw 39's
$\hat d$ was 1.464 um **[run, sandbox: draw 39 recomputed 2026-10-16, the
stored $d$ and $v_0$ reproduced]**. At `fit_variants`' budget of 700 such
stops would be more frequent **[reasoning]**.

Under Q2 (A) below (a fit at a bound of $d$, $\mu$, $v_0$ or $w$, or not
converged, leaves $\mathcal S$), D-052's thin fits keep 32 of 40 (Gaussian truth:
mean 0.603 um, SD 0.120 um) and 24 of 40 ($t_4$: mean 0.597 um, SD
0.101 um); the thick ones keep 40 (Gaussian), 38 ($t_4$) and 40 (generalized
normal) of 40 **[run, sandbox: from the JSON, 2026-10-16]**. The SD of $\hat w$ (true $w$ 0.5 um) is 0.03-0.07 um on
the thick truths and 0.11-0.13 um on the thin ones **[run, sandbox]**.

### 4.5 Cost

The fits of one simulated profile (Block 5's, the Gaussian of fitted width,
the SGT, the SGT at $\lambda$ = 0.5) took at most about 5.5 CPU-s together at
$N_q$ = 848: part B ran 551 s wall on 2 processes for 200 profiles, in 5
tasks, so one process sat idle part of the time **[run, sandbox; derived,
an upper bound]**. The
iteration fits at 1 to 5 planes per node (2 on the synthetic 3.0 um tube,
D-047 **[KB]**). Rendering under today's chain: 10-140 s per replicate, peak 0.7 GB
**[KB: HPC paths reference §2]**; the new chain renders 13 planes (today
$\pm 3$ widened for tilt) and squares up to $\pm 1.5\,d$ + 2 px, so expect
more **[reasoning]**; measure it (Q6).

### 4.6 What the simulation did not contain

No rendered stack: no slab-by-slab defocus (whose sum over depth makes the
projected kernel a mixture of Gaussians of different widths: at a given FWHM
it is not Gaussian, though its far tails are still those of its widest
component **[reasoning]**), no pixel integration, no 8-bit
rounding or JPEG, no neighbouring neurites, no tilt other than 0.2 rad, noise
1-2 gl where the renderer's default is 3.0 gl (`noise_sd_gl`, PROVISIONAL;
D-025's measurement of the real noise has not run) **[src; KB]**. Gate 2
under the new chain is the first test of D-052's fit on rendered phantoms.

## 5. Today's code against D-052 and the standing decisions

| Component | Today at `9e3296c` **[src]** | Required | By |
|---|---|---|---|
| Plane of the final fit | $k^*$ (gradient energy over $\lvert v\rvert \le r$ + 0.5 um) | the iteration (H6) after the pilot | D-052 |
| Final fit | `fit.fit_profile`: Gaussian at $\sigma_{\rm fit}$, $(d, \mu, v_0)$, starts $d_0$ x (0.7, 1.0, 1.4) | (H4)-(H5): seven parameters, IRLS at $\lambda$ = 0.5 | D-052 |
| Kernel family | `blur_kernels.FAMILIES`: gauss, lorentz (no shape), voigt, student, gennorm, skewnorm (one shape each) | add `sgt` with a shape vector $(\beta, \xi, \zeta)$; `_Layout` a shape vector | D-052 |
| Iteration code | in `scripts/plane_differences.py`; `node_stack` built on `survey.node_planes` (real SWC, provider, transform) | in `src/allen_diameter/analysis/`, fed by `measure_node`'s branch, provider and pilot; scripts import it | D-052 |
| Estimator hash | `estimator_signature` = acquisition + all of `MeasureConfig` (minus `sigma_fit_study_um`) | every D-052 setting a `MeasureConfig` field (today $w$ bounds, IRLS tolerance and cap are `fit_variants` module constants, outside the hash) | procedure §3.11 |
| Phantom radius | true $d/2$ (`phantom_branch`) | open | Q1 (D-043 (a), D-052 (a)) |
| $\mathcal S$, bounds | `fit_status == "converged"`, which excludes any parameter at a bound | restated for the shape parameters | Q2 (D-052 (b)) |
| $\mathcal S$, dark | `dark` ($\hat\alpha$ > `alpha_dark_flag`) fails $\mathcal S$ (`phantoms.SELECTION_FLAGS`; SPEC §2.2, Block 6) | `dark` a label only, for phantoms and real nodes | **D-030, decided, not implemented** |
| Failure rate | `failure_rate_ignore = ("dark",)` (PROVISIONAL, gate 2) | revisit once `dark` leaves $\mathcal S$ | Q3 |
| Table axes | `table.table_inputs(d, phi)`: $(\ln d, \varphi)$; `tau_kernel_bandwidth` (0.15, 5 deg) | $(\ln d, \varphi, \hat\alpha)$; `BiasTable` methods, `invert_node` at $\hat\alpha_i$, coverage in $\hat\alpha$ | **D-031, decided, not implemented**; details Q4 |
| $\sigma_{\rm fit}$ study | three tables at 0.080, 0.099, 0.125 um (Cell 8 writes the configs) | under a fitted width $\sigma_{\rm fit}$ reaches only the pilot and the starts | Q5 (D-023) |
| Profile line | $\pm 3$ um (`profile_half_um`); phantom $d$ on [0.2, 4] um, TEMPORARY until Cell 6 sets $d_{\max}$ | unchanged unless decided | Q8 (D-038, D-043 (c)) |
| Renderer kernel | `kernel_family = "gaussian_table"`, $\sigma_{\rm r}(0)$ = 0.080 um | unchanged | D-026, D-052 (f); Q10 |
| Gate 2 | passed 2026-10-06 at estimator hash `535847535e83cf7d` (dip-depth focus rule) | re-run under the new hash | D-052 |
| Production job | `build_table.pbs`: `-J 0-99`, 1 core, 4 GB, 12 h, $N$ = 2000; never run on davinci | resized after Q6 | D-022 |

## 6. Where the node's radius enters (Q1)

On a phantom $r_j$ is the true $d/2$ (`phantoms.phantom_branch`:
`Branch.from_points(xyz, 0.5 * draw.d_um)`), on a real node Allen's radius
**[src]**. It enters four places:

| Place | Rule | Code |
|---|---|---|
| Iteration start (new) | $d^{(0)}_j = 2 r_j$ | `gradient_iteration(d_start_um=None)` uses `2 * allen_radius_um` |
| Pilot's focus window | $\lvert v\rvert \le r_j$ + 0.5 um | `focus_grad_window = "radius_margin"`; the comment on `focus_grad_margin_um` (`config.py` l. 93): "Allen's on real nodes, the true one on phantoms" |
| $\bar B_j(k)$ mask (D-018.1) | pixels within (the larger end radius + 1 um) of a segment of the node's own Branch masked: on real data the stretch plus its parent node (`cell.stretch_branch`), on a phantom its axis | `node_pipeline._bbar` -> `background.mask_near_branch`, `bbar_mask_margin_um` |
| $B_{j,k}$ of the gradient energy | pixels within $r$ + 4 um (shrinking to 0.5) of traced segments and soma masked | `plane_differences.block_background` |

SPEC §2.2 and D-023 (ii) state the $\bar B$ mask as "any SWC segment in the
block"; the code masks only the node's own Branch, so on real data other
traced dendrites in the block stay in the median **[src: SPEC.md §2.2,
`node_pipeline._bbar`]** -- a deviation to record in SPEC.md, and one the
phantoms (a single tube) cannot show **[reasoning]**.

SPEC §8 records, for specimen 529878215, that Allen's $2r$ is about 0.5 um
in every bin of $\hat d$, and that a window radius at half the true one undoes
the window's gain on thick tubes **[src: SPEC.md §8]**. On the synthetic 3.0 um
tube the iteration ended one or two planes from the axis from an Allen start
five times too small **[src: SPEC.md §8]**.

## 7. Open questions for the user (settle before writing logic)

**Q1. The phantoms' radius** (D-043 (a), D-052 (a)).
- (A) Keep the true radius: the table is conditional on an exact radius, real
  nodes use Allen's.
- (B) Give each phantom an Allen-like radius drawn from the specimen's Allen
  radius distribution (Cell 6, `scripts/allen_radius_distribution.py`,
  `radius/radius_nodes_<sp>.csv`), independently of $d$, used in all four
  places of §6; a phantom draw (simulation hash), not an estimator setting.
  (B') The same, conditional on $d$ through the real nodes' joint
  distribution of Allen's radius and $\hat d$.
- (C) First measure the sensitivity on a few rendered phantoms
  ($d \in$ {0.5, 1, 2, 3} um, the chain run with $r \in \{d/2,\ 0.25\ \text{um},\ d\}$):
  if $k^{\rm it}$ and $\hat d$ agree within their SE, (A); otherwise (B).
- Recommendation: (C), then (B) if needed **[reasoning]**.

**Q2. Which bound hits fail $\mathcal S$** (D-052 (b); evidence §4.4).
- (A) A fit fails $\mathcal S$ when $d$, $\mu$, $v_0$ or $w$ is at a bound or the
  IRLS did not converge (its 30-solve cap, or a failed solve such as one at
  `max_nfev`); the shape parameters at their box edges do not fail
  (nuisance coordinates; $\xi$ = 0 and $\beta$ = 1 are members). Collapses
  ($\mu$ = 20 /um) then count in the failure rate, and real ones are filled
  (D5).
- (B) As (A), but $\mu$ at its upper bound stays in $\mathcal S$ and the
  $\hat\alpha$ axis carries the collapse branch ($\hat\alpha \approx 20 \times 0.24 / \cos\varphi \approx 4.8/\cos\varphi$).
- (C) Literal D-024: any bound fails, which fails about half the thick fits on
  $\xi$ = 0.
- Recommendation: (A). A collapsed fit's $\hat d$ (0.23-0.25 um) is set by the
  $\mu$ bound almost whatever the true $d$, so $\hat m$ is flat there and the
  inversion could not use it; (B) gains nothing **[reasoning]**.
- Consequence to raise with it: the failure rate on thin phantoms will be of
  order 0.2-0.4 (8 and 16 of 40 at 0.6 um, §4.4), around
  `max_failure_rate` = 0.2 (PROVISIONAL; `invert_node` flags only a rate
  strictly above it), so many thin real nodes would be flagged
  `high_failure` and filled; that threshold needs a decision too
  **[reasoning]**. The fits that stay are nearly unbiased in the mean (§4.4:
  0.603 and 0.597 um on a 0.6 um tube), so the table's correction is small
  there and its $\hat\tau$ large **[run, sandbox; reasoning]**.

**Q3. `failure_rate_ignore` once D-030 is implemented.** A replicate leaves
the failure-rate count only when its `reject` reasons meet the screen
(`table.fit_table`, the `counted` mask) **[src]**; with `dark` out of
`SELECTION_FLAGS` no replicate is rejected for darkness, so `("dark",)`
becomes inert. Recommendation: default `()`, so that the configuration says
what the code does **[reasoning]**; confirm, or name another screen a real
node applies to itself.

**Q4. The $\hat\alpha$ axis** (D-031's open: scaling, coordinate, budget, how
far outside coverage a node may be corrected).
- Coordinate: (A) $\ln\hat\alpha$ (it spans decades: phantom $\mu d$ from 0.06
  to 12, times $1/\cos\varphi$) or (B) $\hat\alpha$.
- Scaling: the thin-plate spline in `RBFInterpolator` is radial, so the three
  inputs' relative scale changes the fit **[reasoning]**: (A) each input
  divided by its design range, (B) per-axis scales chosen by CV with the
  smoothing.
- A third bandwidth for $\hat\tau$, SE and failure rate (e.g. 0.3 in
  $\ln\hat\alpha$, PROVISIONAL); `config.validate` now requires exactly two
  values (l. 357) **[src]**.
- Recommendation: $\ln\hat\alpha$, range-normalised axes, CV on the
  smoothing only, coverage in $\hat\alpha$ reported (D-030) **[reasoning]**.

**Q5. D-023's $\sigma_{\rm fit}$ axis under a fitted width** (D-052 (d)).
$\sigma_{\rm fit}$ no longer sets the final fit's kernel; it still sets the
pilot's Gaussian fit (centres, heading, tilt), the starts, and the renderer's
consistency check (procedure §3.10, first row).
- (A) Keep the three tables (x 3 cost).
- (B) One reduced sensitivity check (the gate-2 design at 0.080 and 0.125 um),
  reporting the spread of the corrected $\tilde d$.
- (C) Drop it.
- Recommendation: (B) **[reasoning]**; D-023 is the user's, so it is theirs
  to change.

**Q6. Budget and target SE.** Procedure §3.8: $N_{\mathcal S} = (\hat\tau/\epsilon_{\rm MC})^2$
per neighbourhood. From §4.1 and §4.4, the ratio $\hat d/d$ has an SD of
0.008-0.010 at 1.5 um; at 0.6 um, 0.30-0.32 with the collapses and
0.17-0.20 without them (Q2 (A)). At $\epsilon_{\rm MC}$ = 0.01 a thin
neighbourhood then needs about 280-400 retained replicates under Q2 (A)
(900-1050 if collapses stayed), and a three-axis spline spreads them over
one more axis (D-031) **[run, sandbox; reasoning]**. Ask: the target
$\epsilon_{\rm MC}$, the davinci CPU-hours available, and whether to densify the
design at small $d$. Measure the new chain's per-replicate wall time in the
sandbox (about 10 phantoms over $d$ and $\varphi$) first.

**Q7. Thin nodes** (the user: "for what regards the errors we would have to
leverage on the correction table" **[user]**; D-052 (g)).
- (A) The default: the table's mean correction, $\hat\tau$, the failure rate
  and the D5 flags and fill.
- (B) Also a labelled comparison with the kernel pinned from the cell's thick
  nodes (§4.3), which needs a rule for which thick nodes, and per defocus.
- (C) A kernel shared across a stretch's nodes, fitted jointly (a new
  estimator).
- Recommendation: (A) now, (B) after the first table **[reasoning]**.

**Q8. Profile length and $d_{\max}$** (D-038, D-043 (c)). Keep $\pm 3$ um and set
$d_{\max}$ from Cell 6's radius distribution; real nodes above
$\hat m(d_{\max})$ are flagged `out_of_domain` **[reasoning]**; the trunk of
node 2 reached the 6 um bound (D-043 (c)) **[KB]**.

**Q9. Plane range and depth jitter** (D-043 (b), D-024 (vii)). Phantoms'
$k_{\rm SWC}$ is exact (axis depth jitter $\pm$0.14 um only, `jitter_z_um` = 0),
real nodes' is not; keep $\pm 6$ planes and set `jitter_z_um` from the
registration step's $\Delta z^*$ (notebook Cell 3b, the old "cell 13") when
measured **[reasoning]**.

**Q10. The renderer's kernel shape** (procedure §3.10 "Kernel shape", I-002).
Defer until gate 2 passes **[reasoning]**: on thick tubes D-052's fit moved by
at most 0.007 um across three truth kernels, on thin ones its median by
0.04 um (§4.1).

**A real-data check worth asking for before production.** After Block 12
(Cell 4l's grid has no SGT kernel), run D-052's fit on Allen's thick nodes in
Colab and read $\hat w$, $\hat\kappa$ in focus; compare with the same fit on
phantoms of the same $\hat d$. On thick tubes $\hat w$ scattered by 0.03-0.07 um
about a true 0.5 um, on thin ones by 0.11-0.13 um (§4.4): the fitted widths
of 1-2 um on Allen's thin nodes (D-046's figures **[KB]**) say little about
the blur, while a mismatch on thick nodes would say the renderer's blur
($\sigma_{\rm r}(0)$ = 0.080 um, FWHM 0.19 um in focus) is not the real one
**[reasoning]**.

## 8. Block plan

Block numbers continue SPEC.md (Blocks 0-11 exist).

### 8.1 Block 12: the SGT kernel and the seven-parameter fit

- Modules: `model/blur_kernels.py` (family `sgt`: (H1)-(H3), shape
  $(\beta, \xi, \zeta)$, closed-form FWHM); `analysis/fit_variants.py`
  (`_Layout` with a shape vector) or a new `analysis/fit_sgt.py` for the
  deliverable fit (H5) with its IRLS; new `MeasureConfig` fields for every
  constant of §3.5.
- Port `checks/estimator_2026-10-15/sgt_kernel.py` (pure ASCII already);
  do not import from `checks/`.
- Oracles: unit integral (quadrature) to 1e-10; FWHM = $w$ by root-finding;
  mode at 0; the members against `scipy.stats` `norm`, `laplace`, `t(4)`,
  `cauchy`, `gennorm(beta)`, each rescaled to FWHM $w$, to 1e-12; right/left
  half-width ratio $\zeta^2$; continuity as $\xi \to 0$ (the `expm1` branch);
  noise-free recovery of $\theta$ on a 1.5 um tube; nesting (SGT objective
  $\le$ the fitted-width Gaussian's on the same profile); IRLS deterministic
  and stopping by its rule; five draws of the stored Monte Carlo reproduced
  from seed 20261017; defaults asserted.

### 8.2 Block 13: the iteration inside Block 5

- New module (e.g. `analysis/plane_iteration.py`): `iterate_planes`,
  `gradient_iteration`, `refit_at_plane` (its `fit_at` the Block 12 fit),
  `line_profiles`, `block_background` with `far_from_dendrites` and
  `plane_background`, a square builder fed by `measure_node`'s branch,
  provider and pilot instead of `survey.node_planes`, `check_iteration`,
  `ITER_STATUSES`. `scripts/plane_differences.py` then imports them (one
  implementation; Cells 4j-4m's suites must pass unchanged).
- `measure_node`: after the passes, the iteration for node $i$, then the final
  fit at $k^{\rm it}$. `NodeResult` gains `k_iter`, `iter_status`, `n_rounds`,
  `n_restacks`, `w_hat_um`, `beta_hat`, `xi_hat`, `zeta_hat`, `irls_ok`,
  `at_bound`, and `d_pilot_um` (comparison); `run_replicate`'s row and
  `run_cell.py`'s CSV carry them.
- `MeasureConfig`: iteration fields (line multiplier 3.0, rounds 5, plane
  half-range 6, background margin 4.0 um, start "allen_radius").
- Oracles: on `tests/smoke/fixtures_cell.py`, the moved iteration with the old
  `fit_at` reproduces the scripts' `gradient_iteration` exactly; phantoms and
  real nodes share the code path (one function object); the estimator hash
  differs from `535847535e83cf7d` (gate 2) and from `6f0397236447fe6f` (the
  default configuration at `9e3296c`, computed 2026-10-16 **[run, sandbox]**);
  a table of the old hash is refused.

### 8.3 Block 14: Blocks 6-7 under D-030, D-031 and D-052

- $\mathcal S$ per Q2; `dark` a column only (D-030); `failure_rate_ignore` per
  Q3; `table_inputs` $(\ln d, \varphi, \cdot)$ per Q4; `BiasTable.b_hat`,
  `m_hat`, `tau_hat`, `se_hat`, `failure_rate`, `n_eff` with $\hat\alpha$;
  `tau_kernel_bandwidth` with three values; `invert.invert_node(d_hat, phi,
  alpha_hat, table, cfg)`; coverage in $\hat\alpha$; the phantom radius per Q1.
- Oracles: the fixture of `test_smoke_table.py` extended to
  $b(d, \varphi, \alpha)$ (e.g. $1 + 0.06/d + 0.15\sin^2\varphi + 0.05\ln\alpha$),
  recovered within 4 SE; inversion at fixed $\hat\alpha$; D-030's check (a node
  with $\hat\alpha > 1$ corrected, not filled, `dark` = True); the dark
  replicates kept in the table.

### 8.4 Gates

1. Every suite, then a clean clone of the pushed branch (as Blocks 11's
   commits were checked).
2. Gate 2, reduced end-to-end, sandbox, new hash (`configs/gate2_reduced.json`
   in the new schema; under D-030 `--alpha-range` is no longer needed to pass
   a dark screen):

   ```
   python scripts/build_table.py run --start 0 --stop 240 --workers 2 --out-dir T --config-json configs/gate2_reduced.json
   python scripts/build_table.py merge --out-dir T --config-json configs/gate2_reduced.json
   python scripts/end_to_end.py phantoms --d 0.5,1,2,3 --reps 8 --seed 777 --out T/test.csv --config-json configs/gate2_reduced.json
   python scripts/end_to_end.py evaluate --table T/bias_table_<hash> --rows T/test.csv --config-json configs/gate2_reduced.json
   ```

   Pass: every mean $\tilde d/d$ within 0.10 of 1 (`--tol` 0.10). Same simulator
   for table and test: this tests the Monte Carlo, the estimator and the
   inversion, not the simulator (procedure §3.10).
3. davinci (`hpc-git-delivery`, `hpc-python-compat`): the byte checks,
   `qsub -v DIAM_DRYRUN=1`, the two-replicate probe, a pilot of about 200
   replicates read for $\hat\tau$ by region (Q6), then production and the merge
   (commands: HPC paths reference §2.1 **[KB]**).
4. Procedure §3.10 on the production table: blur consistency, $\sigma_{\rm r}(0)$,
   kernel shape (Q10), $\mu$ percentiles, heading bins, end-to-end with an
   alternative generator.

## 9. Files delivered with this handoff

| File | Content |
|---|---|
| `docs/TEEG_diameter_correction_table_handoff_2026-10-16.md` | this file |
| `checks/estimator_2026-10-15/` | `sgt_kernel.py` (the reference kernel), the impostor and Monte Carlo scripts of both runs, the pinning and independent-quadrature checks, the figure and equation scripts, `ident/` and `ident_sgt/` results; `README.md` with commands and the lines a correct run prints. Pure ASCII, compiled **[run, sandbox]** |
| `identifiability_2026-10-15.zip`, `flexible_kernel_2026-10-15.zip` | the same scripts and results, sent to the user on 2026-10-15 (the figures `fig3`-`fig6` and the equation images are in the zips only; `fig_identifiability.py` and `fig_sgt.py` redraw them) |

## 10. Not checked

- D-052's fit on a rendered phantom or on Allen's data: never run **[open]**.
- The iteration inside `measure_node`: not written; on phantoms it has never
  run **[open]**.
- The real image noise (D-025) and the camera chain's black level and gain:
  not measured; the renderer's 3.0 gl and 210 gl are PROVISIONAL **[src]**.
- The SGT family's standard references (McDonald & Newey 1988, Theodossiou
  1998, Fernández & Steel 1998) are named from memory and not verified; they
  are not in PubMed. (H1)-(H3) were checked numerically instead
  (`sgt_kernel.py`, ALL PASS, re-run 2026-10-16) **[run, sandbox]**. Practical
  identifiability by the profile likelihood follows Simpson & Maclaren 2023,
  *PLoS Comput Biol*, PMC10566698 (full text read 2026-10-15) **[PubMed FT]**.
- `independent_model_check.py` matched the pipeline's quadrature within 6e-6
  gl on three cases; on the cusped Laplace kernel at 1.5 um it was 8.3e-4 gl
  off (over its 1e-4 gl tolerance), changing that $\Delta\chi^2$ by 0.002: a
  cusped kernel needs more nodes than the 848 rule gives **[run, sandbox]**.
- Block 10 depends on `measure_node`: `analysis/calibration.py` calls it
  (l. 222) and selects its calibration nodes on the result's $\hat d$,
  $\hat\alpha$ and $k^*$ **[src]**, so Block 13 changes Block 10's selection
  unless the pilot's values stay available (e.g. `d_pilot_um`, `k_star`) and
  Block 10 reads those.
- The $\bar B$ mask covers only the node's own Branch, not "any SWC segment in
  the block" as SPEC §2.2 says (§6) **[src]**.
- Exists, not checked for this change: `scripts/run_cell.py` and the
  notebook's Cells 8-10 (they read `NodeResult` and the table; their columns
  and the inversion's arguments change in Blocks 13-14).

## 11. Working rules

- Load `scientific-coding` before writing or reviewing code; for anything that
  runs on davinci also `hpc-python-compat` and `hpc-git-delivery`.
- Read the decision log before acting on anything it constrains
  (`project-decision-log`); cite IDs; log a decision the moment the user
  makes it. Next free IDs on 2026-10-16: D-053, I-003.
- Commit and push only when the user agrees; never to `main`; never
  force-push.
- `.py` sources pure ASCII, shell and PBS files LF only.
- In the chat, displayed equations go out as LaTeX rendered to PNG; no
  complex Unicode math in text.
- Documents of this workstream live in `docs/` (D-020); project docs follow
  DOCUMENT UPKEEP (re-read, merge, dated changelog rows, "[corrected <date>]"
  marks); end every reply with one line naming every doc written.
