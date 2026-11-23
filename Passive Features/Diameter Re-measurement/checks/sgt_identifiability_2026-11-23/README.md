# The SGT kernel in the per-node joint fit: an identifiability check (D-221, D-222)

2026-11-23. Standalone check, not pipeline code. Branch `wip/diameter-pipeline-sgt-check` from `f1bc7cb`; the pipeline
(`src/`, `scripts/`, `tests/`, `specs/`) is read, never changed. Every claim below carries its basis: **[run here]**
(a number this folder's scripts produced), **[reasoning]**, or **[SPEC]** / **[code]** / **[read]** (read in the
repository: the SPEC, the code, a project document).

## Question

D-221 and D-222 (the user, 2026-11-22, as the brief summarises them): the node's diameter fit gets an option to replace
the Gaussian defocus kernel by the skewed generalized-t (SGT) family of Block 12. At each slab-plane defocus
$\delta$ the SGT's FWHM is the Gaussian law's, $w(\delta) = 2\sqrt{2\ln 2}\,\sigma_{\rm r}(\delta)$; the shape
$(\beta, \xi)$ is fitted per node with $\zeta = 1$ by default (a symmetric profile kernel); an option also fits
$\zeta$; the member $(2, 0, 1)$ is the Gaussian, today's fit. Before the option is built:

1. In the per-node joint fit across planes, can $(\beta, \xi)$ -- and $\zeta$ under the option -- be estimated together
   with $\theta = (d, \mu, v_0, z_{\rm ax})$ without a degeneracy that makes $d$ unreliable?
2. How large is the error in $d$ when the true kernel is an SGT and the fit assumes the Gaussian (the option's reason)?

## The model

The 1-D model the line renderers (Blocks 26, 35) reduce to for a straight tube invariant along its axis
(`sgt_line_model.py`) **[code, run here]**:

- **Geometry and sampling.** A flat tube along the pixel grid (tilt 0, heading 0), diameter $d$, coefficient $\mu$,
  axis $v_0$ = 0.04 um aside of the line's origin (a generic sub-pixel offset) at depth $z_{\rm ax}$. The measuring
  line crosses it through a pixel centre: Block 5's offsets $v_n = n\,p$, $n = -26..26$ ($\pm 3$ um, step $p$ =
  0.1144 um: `profile_half_um`, `profile_step_um`), so each sample is a pixel value and the $2A + 1$ = 9 lines along
  the tube read equal pixels. Planes $k = -4..4$ at $z_k = k\,\Delta z$, $\Delta z$ = 0.28 um (Block 36's
  `cleaning.plane_half_range` = 4 about $k^*$ = 0); Block 30's `node_inference.select_planes` (peak dip $\ge$
  `plane_snr_min` = 1 noise SD) kept all 9 in every case **[run here]**.
- **Absorption.** Block 26's fine lattice across the line ($h = p/8$, `inference.line_fine_factor`), each sample's
  column interval (Block 2 `column_interval`), the slabs (Block 2 `slab_centres`, `dzeta_um` = 0.02 um) and the
  partition absorption $\Delta A_j = T_{<j}(1 - e^{-a_j})$ (Block 2 `slab_absorbance`, `transmitted_before`, light
  +1), as the brief asks (the per-node fit itself uses the hybrid, D-130.O10).
- **Kernel.** Each slab-plane pair at defocus $\delta = \zeta_j - z_k$ is blurred by the 1-D profile kernel: the
  Gaussian of SD $\sigma_{\rm r}(\delta)$ (Block 4 `kernel.sigma_r`, the default table: 0.080 um in focus, knots to
  0.84 um, linear continuation 0.79), or the SGT $g_{w(\delta), \kappa}$ of Block 12 ((H1)-(H3);
  `blur_kernels.sgt_unit_fwhm`, `sgt_unit_pdf`) at $w(\delta) = 2\sqrt{2\ln 2}\,\sigma_{\rm r}(\delta)$, one shape
  $\kappa = (\beta, \xi, \zeta)$ for every pair. For an isotropic PSF the profile kernel is its line projection (the
  1-D Gaussian of the same SD for the Gaussian law); for the option the SGT is taken as the profile kernel itself
  (D-222).
- **Pixel mean.** Each fine sample's kernel is integrated exactly over the pixel, as Block 26 does with `ndtr`; for
  the SGT with its closed-form CDF (the Block 12 test oracle's: $\tfrac12 I_t(1/\beta, 1/\xi)$ or
  $\tfrac12 P(1/\beta, y^\beta)$ on each side of the two-piece density).
- **Data and fit.** $I_k(v_n) = B_k\,\tau_k(v_n) + \varepsilon$, $B_k$ = 150 gl, $\varepsilon$ white Gaussian of SD
  $\sigma$ = 1.1 gl per line sample. The values: 1.1 gl and $b_0$ = 150 gl are the ones Block 42's identifiability
  check uses (`checks/sigma_calibration_2026-11-20/identifiability.py`, "the pilot cell's 5353 sits on about 140 gl")
  **[code]**, inside the SPEC's "profile's noise SD on the pilot, 0.85 to 1.31 gl" (Block 36, amendment 9) **[SPEC]**.
  SEs scale with $\sigma/B$; correlations, inflation factors, condition numbers and the noise-free biases of (b) do
  not depend on it **[reasoning]**. The fit (`fitting.py`) is Block 30's: $\min \sum_{k,n}[I_k(v_n) -
  B_k\tau_k(v_n;\theta)]^2/\sigma^2$, `scipy.optimize.least_squares` (trf, 2-point, `x_scale="jac"`, `max_nfev` 200,
  scipy's tolerances), Block 30's bounds ($d \in [0.1, 6]$ um, $\mu \in [0.01, 10]$ /um, $v_0 \in [-1, 1]$ um,
  $z_{\rm ax}$ = start $\pm 1$ um) and Block 12's (`sgt_beta_bounds` [1, 10] searched in $\log_{10}\beta$,
  `sgt_xi_bounds` [0, 2], `sgt_ln_zeta_bounds` [-0.7, 0.7]), the covariance Block 30's $(J^\top J)^{-1}$ with the
  rows of a parameter at a bound blanked. **The background is one scale per plane solved exactly** (variable
  projection, $B_k = \sum_n I_k\tau_k / \sum_n \tau_k^2$), as the brief states. Note: Block 30's `fit_node` itself
  holds each $B_k$ at the measured background (`node_inference.py`; `infer_nodes.py` passes `node.B_gl`) **[code]**,
  so (a) also reports the SEs with the backgrounds known (column "infl bk").
- **Weights.** The fit is unweighted (Block 30's entry). Block 39's centre weights ($w_n = e^{-|v_n - v_0|/\lambda}$,
  $\lambda$ = 1.25 x `allen_fwhm_um`(d, 0.080 um) with Allen's diameter = the true $d$; the default of the clear-node
  path, D-169.4) enter (a) only, as a labelled variant (sandwich covariance, Block 30's amendment 2).
- **Kernel truths** (the brief's): `gauss` (2, 0, 1); `heavy` (2, 0.5, 1); `b13` (1.3, 0, 1), the brief's "flatter
  core" -- at equal FWHM a generalized normal with $\beta < 2$ has a sharper, cusp-like core and heavier shoulders than
  the Gaussian (a flatter core is $\beta > 2$) **[reasoning]**, run as given; `skew` (2, 0.3, 1.3), the $\zeta$
  option's case. Geometries: $d$ = 0.4, 0.8, 1.5, 2.5 um x $\mu$ = 0.5, 1.5 /um x the axis on the plane of best
  focus ($z_{\rm ax}$ = 0) and 0.5 um off it.
- **Fitted models.** `gauss` (today's), `z1` ($(\beta, \xi)$ free, $\zeta$ = 1, D-222's default), `zf`
  ($(\beta, \xi, \zeta)$ free, the option), and in (a) `known` (the true shape held; SE(d)'s reference).

## Validation of the 1-D model

`validate_model.py` -> `validate_model.out`, `validation.json` **[run here]**:

| check | result |
|---|---|
| V1: 1-D model vs Block 35 `line_profiles` (partition, Gaussian), 12 tubes | max abs diff in tau 1.8e-10 (min tau of the cases -0.020) |
| V2: SGT CDF vs quad of `blur_kernels.sgt_kernel_pdf`, 11 shapes x 3 widths | max abs 1.5e-13 |
| V2: dF/dt vs the density (central difference) | max rel 4.4e-09 |
| V2: SGT (2, 0, 1) vs ndtr; xi = 1e-10 vs xi = 0 | 2.1e-15; 9.4e-12 |
| V3: fast (Hermite) H vs exact: CDF; model tau over the 64 truth cases | 1.3e-11; 4.9e-13 |
| V4: grouped FFT correlation vs the per-pair sum | 4.4e-16 |

Accepted differences: V1's 1.8e-10 is the 2-D renderer's caps (half-length 12 um) and its reach rule (6 kernel SDs),
neither in the 1-D model; it is $2.5 \times 10^{-8}$ of the noise in $\tau$ units (7.3e-3). Both renderers give
$\tau < 0$ (down to -0.02) for the dark thick tube ($d$ 2.5 um, $\mu$ 1.5 /um): the slab-by-slab blur of Eqs. 5-6
spreads the defocused slabs' light from the tube's edge columns over its centre, where the in-focus slabs have
already absorbed nearly everything; a property of the model, identical in Block 35 **[run here; reasoning]**.

## Run

```
cd "Passive Features/Diameter Re-measurement/checks/sgt_identifiability_2026-11-23"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 validate_model.py > validate_model.out          # 1 min
python3 fisher_bias.py > fisher_bias.out                 # (a), (b): 3 min
python3 recovery.py --d 2.5 --n-sets 15 --n-robust 1 > recovery_d2.5.out   # (c), one process per diameter,
python3 recovery.py --d 1.5 > recovery_d1.5.out          #   at most two at a time: 15-35 min each on a shared CPU
python3 recovery.py --d 0.8 > recovery_d0.8.out
python3 recovery.py --d 0.4 > recovery_d0.4.out
python3 discretization_check.py > discretization_check.out   # slabs and fine lattice halved: 2 min
python3 lattice_check.py > lattice_check.out             # fine factor 8..64: 1 min (reads recovery_d*.json)
python3 corr_structure.py > corr_structure.out           # the correlation matrices' weakest direction: 1 min
python3 figures.py                                       # fig_d_error_se.png, fig_correlations.png, fig_bias_gauss.png
python3 tables.py > tables.md                            # the tables below
```

Files: `sgt_line_model.py` (the model, the SGT's CDF, the Fisher information), `fitting.py` (Block 30's fit with the
backgrounds projected out), `cases.py`, `validate_model.py`, `fisher_bias.py`, `recovery.py`,
`discretization_check.py`, `lattice_check.py`, `corr_structure.py`, `figures.py`, `tables.py`; results
`validation.json`, `fisher_a.csv`, `bias_b.csv`, `fisher_bias.json`, `recovery_d*.json` (every fit),
`discretization_check.json`, `lattice_check.json`, `corr_structure.json`, the `.out` logs, `tables.md`; figures
`fig_d_error_se.png`, `fig_correlations.png`, `fig_bias_gauss.png`. Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, one
thread per process, on 2 CPUs shared with another agent's tests (load about 4).
`recovery_d2.5_aborted30.out` is the log of a first d = 2.5 um batch at 30 data sets, stopped after one case for time
(its numbers are not used).

## Results

### (a) Fisher information at the truth

Per diameter, over its four geometries. SE(d)/d in %; infl = SE(d) / SE(d | shape known); |corr| the largest; cond = condition number of the correlation matrix; w = Block 39's centre weights (sandwich); bk = backgrounds known (Block 30's fit_node) instead of solved.

| truth | model | d (um) | SE(d)/d % | infl | infl w | infl bk | max abs corr(d,beta) | (d,xi) | (d,zeta) | SE(beta) | SE(xi) | SE(zeta) | cond |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss | known | 0.4 | 1.99..4.98 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 88..103 |
| gauss | known | 0.8 | 0.43..0.93 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 24..28 |
| gauss | known | 1.5 | 0.16..0.28 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 13..15 |
| gauss | known | 2.5 | 0.08..0.12 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 8..13 |
| gauss | z1 | 0.4 | 2.15..5.41 | 1.08..1.09 | 1.14..1.15 | 1.10..1.11 | 0.18 | 0.25 | n/a | 0.28..0.75 | 0.22..0.58 | n/a | 357..379 |
| gauss | z1 | 0.8 | 0.43..0.93 | 1.01 | 1.00..1.01 | 1.02..1.03 | 0.08 | 0.09 | n/a | 0.16..0.37 | 0.12..0.28 | n/a | 399..454 |
| gauss | z1 | 1.5 | 0.16..0.29 | 1.00..1.03 | 1.00..1.02 | 1.02..1.07 | 0.13 | 0.11 | n/a | 0.14..0.32 | 0.10..0.24 | n/a | 859..1327 |
| gauss | z1 | 2.5 | 0.08..0.13 | 1.00..1.06 | 1.00..1.06 | 1.01..1.17 | 0.07 | 0.05 | n/a | 0.12..0.32 | 0.09..0.24 | n/a | 921..3366 |
| gauss | zf | 0.4 | 2.17..5.44 | 1.08..1.09 | 1.15..1.16 | 1.11..1.12 | 0.18 | 0.25 | 0.12 | 0.28..0.75 | 0.22..0.58 | 0.014..0.038 | 358..379 |
| gauss | zf | 0.8 | 0.44..0.94 | 1.01..1.02 | 1.01..1.02 | 1.03..1.04 | 0.08 | 0.09 | 0.15 | 0.16..0.37 | 0.12..0.28 | 0.006..0.013 | 407..463 |
| gauss | zf | 1.5 | 0.17..0.30 | 1.05..1.07 | 1.04..1.06 | 1.07..1.11 | 0.12 | 0.11 | 0.29 | 0.14..0.32 | 0.10..0.24 | 0.003..0.006 | 872..1372 |
| gauss | zf | 2.5 | 0.09..0.14 | 1.05..1.10 | 1.05..1.10 | 1.04..1.22 | 0.06 | 0.05 | 0.30 | 0.12..0.32 | 0.09..0.24 | 0.002..0.004 | 954..3388 |
| heavy | known | 0.4 | 2.41..6.03 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 110..128 |
| heavy | known | 0.8 | 0.49..1.06 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 26..31 |
| heavy | known | 1.5 | 0.17..0.31 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 14..16 |
| heavy | known | 2.5 | 0.09..0.14 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 8..13 |
| heavy | z1 | 0.4 | 2.64..6.59 | 1.09..1.10 | 1.13 | 1.12 | 0.09 | 0.18 | n/a | 0.23..0.62 | 0.29..0.76 | n/a | 505..518 |
| heavy | z1 | 0.8 | 0.51..1.12 | 1.04..1.06 | 1.06..1.08 | 1.07..1.08 | 0.26 | 0.30 | n/a | 0.13..0.30 | 0.15..0.36 | n/a | 653..706 |
| heavy | z1 | 1.5 | 0.18..0.32 | 1.00..1.02 | 1.00..1.01 | 1.03..1.07 | 0.05 | 0.05 | n/a | 0.10..0.24 | 0.12..0.29 | n/a | 1312..2108 |
| heavy | z1 | 2.5 | 0.09..0.14 | 1.01..1.07 | 1.01..1.07 | 1.01..1.16 | 0.09 | 0.09 | n/a | 0.09..0.25 | 0.09..0.29 | n/a | 1335..6245 |
| heavy | zf | 0.4 | 2.65..6.63 | 1.09..1.10 | 1.14 | 1.12..1.13 | 0.09 | 0.18 | 0.11 | 0.23..0.62 | 0.29..0.76 | 0.017..0.044 | 507..520 |
| heavy | zf | 0.8 | 0.52..1.13 | 1.05..1.07 | 1.07..1.09 | 1.08..1.09 | 0.26 | 0.29 | 0.15 | 0.13..0.30 | 0.15..0.36 | 0.007..0.015 | 654..708 |
| heavy | zf | 1.5 | 0.18..0.33 | 1.04..1.06 | 1.04..1.06 | 1.08..1.10 | 0.05 | 0.06 | 0.30 | 0.10..0.24 | 0.12..0.29 | 0.004..0.007 | 1310..2141 |
| heavy | zf | 2.5 | 0.09..0.15 | 1.06..1.11 | 1.06..1.11 | 1.05..1.21 | 0.08 | 0.09 | 0.31 | 0.09..0.25 | 0.09..0.29 | 0.002..0.004 | 1359..6238 |
| b13 | known | 0.4 | 2.84..7.16 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 138..163 |
| b13 | known | 0.8 | 0.54..1.16 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 28..33 |
| b13 | known | 1.5 | 0.19..0.34 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 14..16 |
| b13 | known | 2.5 | 0.09..0.15 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 8..13 |
| b13 | z1 | 0.4 | 3.13..7.84 | 1.09..1.10 | 1.13..1.14 | 1.13 | 0.09 | 0.18 | n/a | 0.12..0.32 | 0.14..0.37 | n/a | 509..523 |
| b13 | z1 | 0.8 | 0.57..1.25 | 1.05..1.08 | 1.08..1.11 | 1.08..1.10 | 0.31 | 0.34 | n/a | 0.06..0.16 | 0.07..0.18 | n/a | 598..693 |
| b13 | z1 | 1.5 | 0.19..0.34 | 1.00..1.01 | 0.99..1.00 | 1.02..1.06 | 0.04 | 0.03 | n/a | 0.05..0.12 | 0.06..0.14 | n/a | 1139..1698 |
| b13 | z1 | 2.5 | 0.10..0.16 | 1.01..1.09 | 1.01..1.09 | 1.01..1.16 | 0.22 | 0.20 | n/a | 0.05..0.13 | 0.06..0.15 | n/a | 1936..6342 |
| b13 | zf | 0.4 | 3.15..7.87 | 1.10..1.11 | 1.14..1.15 | 1.13 | 0.10 | 0.19 | 0.10 | 0.12..0.32 | 0.14..0.37 | 0.017..0.046 | 510..525 |
| b13 | zf | 0.8 | 0.57..1.26 | 1.06..1.08 | 1.09..1.12 | 1.09..1.10 | 0.31 | 0.34 | 0.14 | 0.06..0.16 | 0.07..0.18 | 0.007..0.016 | 600..695 |
| b13 | zf | 1.5 | 0.20..0.35 | 1.03..1.05 | 1.03..1.05 | 1.07..1.09 | 0.05 | 0.04 | 0.30 | 0.05..0.12 | 0.06..0.14 | 0.004..0.007 | 1138..1741 |
| b13 | zf | 2.5 | 0.10..0.16 | 1.05..1.13 | 1.05..1.13 | 1.06..1.21 | 0.20 | 0.18 | 0.31 | 0.05..0.13 | 0.06..0.15 | 0.003..0.005 | 1952..6313 |
| skew | known | 0.4 | 2.23..5.58 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 101..117 |
| skew | known | 0.8 | 0.47..1.00 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 25..30 |
| skew | known | 1.5 | 0.17..0.30 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 14..15 |
| skew | known | 2.5 | 0.08..0.13 | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | n/a | n/a | 8..12 |
| skew | zf | 0.4 | 2.46..6.14 | 1.09..1.10 | 1.13..1.14 | 1.12..1.13 | 0.13 | 0.21 | 0.13 | 0.24..0.64 | 0.25..0.67 | 0.021..0.057 | 449..463 |
| skew | zf | 0.8 | 0.49..1.05 | 1.04..1.05 | 1.04..1.06 | 1.06..1.07 | 0.19 | 0.22 | 0.16 | 0.13..0.30 | 0.13..0.31 | 0.009..0.020 | 498..536 |
| skew | zf | 1.5 | 0.17..0.31 | 1.03..1.06 | 1.03..1.05 | 1.07..1.10 | 0.04 | 0.07 | 0.28 | 0.10..0.24 | 0.10..0.24 | 0.005..0.010 | 974..1627 |
| skew | zf | 2.5 | 0.09..0.14 | 1.06..1.10 | 1.06..1.10 | 1.03..1.22 | 0.10 | 0.11 | 0.28 | 0.09..0.24 | 0.08..0.24 | 0.003..0.006 | 982..4267 |

Step check (every finite-difference step x 0.5 and x 2; heavy truth, axis 0.5 um off, zf model):

| geometry | step factor | SE(d) rel change | SE(v0) rel change | shape SEs, max rel change | inflation (ref -> step) | corr(d, shape) max abs change |
|---|---|---|---|---|---|---|
| d0.4_mu0.5_z0.5 | 0.5 | +0.002 | +0.006 | 3.2e-04 | 1.101 -> 1.101 | 0.004 |
| d0.4_mu0.5_z0.5 | 2.0 | +0.008 | -0.055 | 3.4e-03 | 1.101 -> 1.106 | 0.043 |
| d0.4_mu1.5_z0.5 | 0.5 | +0.002 | +0.008 | 3.0e-04 | 1.104 -> 1.103 | 0.005 |
| d0.4_mu1.5_z0.5 | 2.0 | +0.012 | -0.066 | 3.2e-03 | 1.104 -> 1.110 | 0.048 |
| d0.8_mu0.5_z0.5 | 0.5 | +0.000 | +0.006 | 8.9e-05 | 1.066 -> 1.065 | 0.004 |
| d0.8_mu0.5_z0.5 | 2.0 | +0.011 | -0.057 | 4.8e-04 | 1.066 -> 1.074 | 0.054 |
| d0.8_mu1.5_z0.5 | 0.5 | -0.000 | +0.008 | 7.7e-05 | 1.070 -> 1.069 | 0.005 |
| d0.8_mu1.5_z0.5 | 2.0 | +0.021 | -0.076 | 3.1e-04 | 1.070 -> 1.082 | 0.065 |
| d1.5_mu0.5_z0.5 | 0.5 | -0.014 | +0.036 | 3.2e-03 | 1.050 -> 1.041 | 0.035 |
| d1.5_mu0.5_z0.5 | 2.0 | -0.092 | +0.005 | 4.1e-03 | 1.050 -> 1.027 | 0.117 |
| d1.5_mu1.5_z0.5 | 0.5 | -0.025 | +0.048 | 1.6e-03 | 1.058 -> 1.045 | 0.041 |
| d1.5_mu1.5_z0.5 | 2.0 | -0.146 | -0.005 | 7.3e-03 | 1.058 -> 1.025 | 0.141 |
| d2.5_mu0.5_z0.5 | 0.5 | -0.012 | +0.038 | 6.4e-03 | 1.087 -> 1.076 | 0.035 |
| d2.5_mu0.5_z0.5 | 2.0 | -0.096 | +0.117 | 1.9e-02 | 1.087 -> 1.062 | 0.101 |
| d2.5_mu1.5_z0.5 | 0.5 | -0.029 | +0.040 | 4.4e-03 | 1.060 -> 1.043 | 0.052 |
| d2.5_mu1.5_z0.5 | 2.0 | -0.180 | +0.116 | 1.2e-02 | 1.060 -> 1.016 | 0.158 |

### (b) Bias in d when the truth is an SGT and the fit assumes the Gaussian kernel (noise-free)

Per diameter, over its four geometries. fit = the actual least-squares fit; lin = the first Gauss-Newton step from the truth; bias/SE in units of the Gaussian model's SE(d) at the truth; chi2 = the fit's noise-free chi2 in noise units (the misfit's visibility: 477 residuals, 4 + 9 fitted parameters, 464 degrees of freedom, so a chi2 SD of 30).

| truth | fit model | d (um) | bias fit % | bias lin % | bias/SE | chi2 noise-free |
|---|---|---|---|---|---|---|
| heavy | gauss | 0.4 | +6.40..+6.72 | +6.78..+7.15 | +1.3..+3.3 | 10..70 |
| heavy | gauss | 0.8 | +1.33..+1.66 | +1.14..+1.61 | +1.4..+3.8 | 75..422 |
| heavy | gauss | 1.5 | +0.31..+1.02 | +0.26..+1.00 | +2.0..+6.4 | 226..1132 |
| heavy | gauss | 2.5 | -1.48..+0.71 | -1.46..+0.67 | -18.1..+5.7 | 251..2450 |
| b13 | gauss | 0.4 | +12.68..+13.14 | +11.96..+12.54 | +2.6..+6.5 | 34..239 |
| b13 | gauss | 0.8 | +2.00..+2.85 | +1.77..+2.58 | +2.2..+6.5 | 251..1410 |
| b13 | gauss | 1.5 | +0.77..+2.05 | +0.44..+1.81 | +3.6..+12.7 | 754..3582 |
| b13 | gauss | 2.5 | -2.59..+1.43 | -2.52..+1.24 | -31.7..+11.5 | 827..6910 |
| skew | gauss | 0.4 | +3.88..+4.81 | +2.09..+2.73 | +0.8..+2.4 | 43..303 |
| skew | gauss | 0.8 | +0.67..+1.03 | -1.61..-1.05 | +0.7..+2.4 | 353..1889 |
| skew | gauss | 1.5 | +0.30..+0.66 | +3.20..+4.70 | +1.1..+4.1 | 1500..5995 |
| skew | gauss | 2.5 | -1.08..+0.76 | +1.15..+3.06 | -13.2..+6.1 | 3483..12925 |
| skew | z1 | 0.4 | -0.44..-0.05 | -4.36..-3.62 | -0.2..-0.0 | 37..261 |
| skew | z1 | 0.8 | -0.30..-0.26 | -2.99..-2.29 | -0.5..-0.3 | 312..1650 |
| skew | z1 | 1.5 | -0.21..+0.21 | +2.90..+4.01 | -0.7..+1.2 | 1371..5430 |
| skew | z1 | 2.5 | -0.26..+0.05 | +2.34..+2.93 | -2.8..+0.4 | 3343..11681 |

### (c) Recovery on noisy profiles (B 150 gl, noise 1.1 gl per line sample)

Per diameter, over its geometries run. err = mean (d_hat - d)/d %; rmse %; cover = share of |d_hat - d| <= SE (nominal 0.68) and <= 1.96 SE (0.95), SE by Block 30's rule; SD/SE = the scatter of d_hat over the median reported SE; bound = share of fits with any parameter at a bound (the bounds met); robust = data sets where one of Block 12's other start shapes found a lower chi2 / sets tried.

| truth | fit | d (um) | sets | err % | rmse % | cover 1 SE | cover 1.96 SE | SD/SE | bound | bounds met | median beta | median xi | median zeta | SD beta | SD xi | SD zeta | robust | chi2/dof |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss | gauss | 0.4 | 30 x 4 | -0.31..+0.25 | 1.61..4.16 | 0.67..0.80 | 0.90..1.00 | 0.86..0.94 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 0.98..1.02 |
| gauss | gauss | 0.8 | 30 x 4 | -0.10..+0.09 | 0.43..0.77 | 0.63..0.83 | 0.90..1.00 | 0.68..1.00 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 0.99..1.01 |
| gauss | gauss | 1.5 | 30 x 4 | -0.00..+0.04 | 0.15..0.28 | 0.67..0.77 | 0.87..0.97 | 0.83..1.20 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 0.99..1.02 |
| gauss | gauss | 2.5 | 15 x 4 | -0.02..+0.05 | 0.06..0.16 | 0.47..0.80 | 0.87..0.93 | 0.75..1.24 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 0.99..1.01 |
| gauss | z1 | 0.4 | 30 x 4 | -0.71..+0.03 | 1.60..4.63 | 0.57..0.77 | 0.90..0.97 | 0.81..1.07 | 0.33..0.60 | xi_lo 0.48 | 2.04..2.23 | 0.00..0.08 | n/a | 0.20..0.52 | 0.13..0.39 | n/a | 0/24 | 0.98..1.03 |
| gauss | z1 | 0.8 | 30 x 4 | -0.14..+0.04 | 0.43..0.78 | 0.60..0.73 | 0.90..1.00 | 0.74..1.04 | 0.43..0.60 | xi_lo 0.51 | 2.00..2.04 | 0.00..0.03 | n/a | 0.08..0.41 | 0.07..0.30 | n/a | 0/24 | 0.99..1.02 |
| gauss | z1 | 1.5 | 30 x 4 | +0.00..+0.05 | 0.15..0.29 | 0.63..0.77 | 0.87..0.97 | 0.84..1.12 | 0.47..0.60 | xi_lo 0.52 | 2.00..2.07 | 0.00..0.03 | n/a | 0.09..0.23 | 0.07..0.18 | n/a | 0/24 | 0.99..1.02 |
| gauss | z1 | 2.5 | 15 x 4 | -0.01..+0.05 | 0.06..0.17 | 0.40..0.80 | 0.87..1.00 | 0.71..1.27 | 0.40..0.60 | xi_lo 0.50 | 2.01..2.07 | 0.00..0.03 | n/a | 0.06..0.14 | 0.05..0.11 | n/a | 0/12 | 0.99..1.01 |
| gauss | zf | 0.4 | 30 x 4 | -0.62..+0.24 | 1.64..4.33 | 0.60..0.83 | 0.97..1.00 | 0.77..0.94 | 0.37..0.60 | xi_lo 0.50 | 2.06..2.20 | 0.00..0.11 | 1.000..1.005 | 0.21..0.52 | 0.13..0.37 | 0.013..0.037 | 1/24 (max chi2 gain 0.02, max abs d change 0.0003 um) | 0.98..1.03 |
| gauss | zf | 0.8 | 30 x 4 | -0.10..+0.04 | 0.40..0.75 | 0.67..0.77 | 0.90..1.00 | 0.76..0.99 | 0.40..0.60 | xi_lo 0.51 | 2.00..2.03 | 0.00..0.02 | 0.999..1.007 | 0.08..0.40 | 0.07..0.29 | 0.005..0.013 | 3/24 (max chi2 gain 0.00, max abs d change 0.0000 um) | 0.99..1.02 |
| gauss | zf | 1.5 | 30 x 4 | -0.01..+0.05 | 0.15..0.28 | 0.63..0.73 | 0.83..0.93 | 0.78..1.19 | 0.43..0.63 | xi_lo 0.51 | 2.00..2.06 | 0.00..0.01 | 0.999..1.001 | 0.09..0.20 | 0.07..0.16 | 0.003..0.007 | 1/24 (max chi2 gain 0.14, max abs d change 0.0000 um) | 0.99..1.02 |
| gauss | zf | 2.5 | 15 x 4 | -0.01..+0.04 | 0.06..0.16 | 0.53..0.87 | 0.80..1.00 | 0.69..1.22 | 0.33..0.67 | xi_lo 0.50 | 2.01..2.07 | 0.00..0.03 | 0.999..1.002 | 0.06..0.14 | 0.05..0.11 | 0.002..0.004 | 0/12 | 0.99..1.01 |
| heavy | gauss | 0.4 | 30 x 4 | +4.42..+7.51 | 5.61..8.80 | 0.00..0.60 | 0.07..0.83 | 0.65..1.09 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.01..1.16 |
| heavy | gauss | 0.8 | 30 x 4 | +1.12..+1.51 | 1.42..1.58 | 0.00..0.43 | 0.17..0.83 | 0.93..1.07 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.15..1.90 |
| heavy | gauss | 1.5 | 30 x 4 | +0.38..+1.05 | 0.41..1.06 | 0.00..0.37 | 0.00..0.60 | 0.86..1.10 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.49..3.45 |
| heavy | gauss | 2.5 | 15 x 4 | -1.42..+0.70 | 0.11..1.43 | 0.00..0.53 | 0.00..1.00 | 0.90..1.54 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.59..6.19 |
| heavy | z1 | 0.4 | 30 x 4 | -1.87..+1.50 | 2.13..4.98 | 0.63..0.70 | 0.87..0.97 | 0.72..1.01 | 0.00..0.20 | xi_lo 0.07, xi_hi 0.03 | 1.91..2.03 | 0.34..0.62 | n/a | 0.20..0.55 | 0.27..0.71 | n/a | 1/24 (max chi2 gain 0.00, max abs d change 0.0001 um) | 0.98..0.99 |
| heavy | z1 | 0.8 | 30 x 4 | -0.16..+0.12 | 0.53..1.07 | 0.57..0.67 | 0.80..0.93 | 0.90..1.20 | 0.00..0.07 | xi_lo 0.02 | 1.86..2.04 | 0.33..0.53 | n/a | 0.11..0.31 | 0.14..0.39 | n/a | 0/24 | 0.98..1.00 |
| heavy | z1 | 1.5 | 30 x 4 | -0.08..+0.05 | 0.16..0.34 | 0.53..0.77 | 0.90..0.97 | 0.91..1.06 | 0.00..0.03 | xi_lo 0.02 | 1.93..2.03 | 0.43..0.54 | n/a | 0.09..0.25 | 0.10..0.30 | n/a | 0/24 | 0.98..1.01 |
| heavy | z1 | 2.5 | 15 x 4 | -0.02..+0.02 | 0.10..0.14 | 0.60..0.73 | 0.93..1.00 | 0.88..1.24 | 0.00 | - | 1.85..2.07 | 0.35..0.56 | n/a | 0.10..0.25 | 0.10..0.29 | n/a | 0/12 | 1.00..1.05 |
| heavy | zf | 0.4 | 30 x 4 | -1.61..+1.26 | 2.08..5.00 | 0.63..0.80 | 0.93..1.00 | 0.74..0.96 | 0.00..0.20 | xi_lo 0.07, xi_hi 0.02 | 1.93..2.05 | 0.33..0.61 | 0.981..1.000 | 0.20..0.52 | 0.27..0.68 | 0.013..0.046 | 0/24 | 0.98..1.00 |
| heavy | zf | 0.8 | 30 x 4 | -0.19..+0.14 | 0.51..1.07 | 0.57..0.63 | 0.83..0.97 | 0.95..1.15 | 0.00..0.07 | xi_lo 0.02 | 1.86..2.02 | 0.34..0.51 | 0.998..1.003 | 0.11..0.31 | 0.14..0.38 | 0.006..0.017 | 0/24 | 0.98..1.00 |
| heavy | zf | 1.5 | 30 x 4 | -0.08..+0.04 | 0.15..0.34 | 0.53..0.77 | 0.83..0.90 | 0.84..1.07 | 0.00..0.03 | xi_lo 0.01 | 1.93..2.03 | 0.43..0.53 | 0.999..1.000 | 0.09..0.25 | 0.10..0.30 | 0.004..0.007 | 0/24 | 0.98..1.01 |
| heavy | zf | 2.5 | 15 x 4 | -0.03..+0.01 | 0.10..0.15 | 0.60..0.73 | 0.93..1.00 | 0.86..1.16 | 0.00 | - | 1.83..2.07 | 0.32..0.56 | 1.001..1.002 | 0.10..0.25 | 0.10..0.29 | 0.002..0.005 | 0/12 | 1.00..1.05 |
| b13 | gauss | 0.4 | 30 x 4 | +10.88..+14.12 | 11.91..14.81 | 0.00..0.13 | 0.00..0.57 | 0.81..0.97 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.04..1.52 |
| b13 | gauss | 0.8 | 30 x 4 | +2.06..+2.75 | 2.20..2.80 | 0.00..0.13 | 0.00..0.57 | 0.76..1.16 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.51..4.06 |
| b13 | gauss | 1.5 | 30 x 4 | +0.77..+2.11 | 0.78..2.12 | 0.00 | 0.00..0.03 | 0.78..1.03 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 2.66..8.65 |
| b13 | gauss | 2.5 | 15 x 4 | -2.54..+1.50 | 0.43..2.55 | 0.00 | 0.00..0.27 | 0.83..2.14 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 2.78..15.96 |
| b13 | z1 | 0.4 | 30 x 4 | -0.86..+1.16 | 2.60..6.48 | 0.60..0.73 | 0.80..0.97 | 0.90..1.00 | 0.23..0.47 | xi_lo 0.38 | 1.32..1.42 | 0.00..0.05 | n/a | 0.06..0.29 | 0.07..0.46 | n/a | 0/24 | 0.97..1.02 |
| b13 | z1 | 0.8 | 30 x 4 | -0.10..+0.00 | 0.49..1.00 | 0.63..0.80 | 0.90..0.97 | 0.80..1.04 | 0.40..0.63 | xi_lo 0.50 | 1.31..1.35 | 0.00..0.03 | n/a | 0.03..0.10 | 0.03..0.13 | n/a | 0/24 | 0.99..1.01 |
| b13 | z1 | 1.5 | 30 x 4 | -0.20..+0.03 | 0.17..0.37 | 0.47..0.80 | 0.87..0.97 | 0.75..0.96 | 0.27..0.60 | xi_lo 0.40 | 1.30..1.34 | 0.00..0.06 | n/a | 0.03..0.09 | 0.04..0.13 | n/a | 3/24 (max chi2 gain 1.70, max abs d change 0.0002 um) | 0.99..1.02 |
| b13 | z1 | 2.5 | 15 x 4 | -0.03..+0.10 | 0.10..0.16 | 0.67..0.80 | 0.87..0.93 | 0.77..1.13 | 0.33..0.60 | xi_lo 0.50 | 1.30..1.32 | 0.00..0.01 | n/a | 0.02..0.09 | 0.02..0.12 | n/a | 0/12 | 0.97..1.01 |
| b13 | zf | 0.4 | 30 x 4 | -0.82..+1.07 | 2.59..6.50 | 0.67..0.77 | 0.87..0.97 | 0.83..1.00 | 0.23..0.43 | xi_lo 0.34 | 1.31..1.43 | 0.00..0.06 | 0.992..1.009 | 0.06..0.28 | 0.08..0.45 | 0.019..0.046 | 0/24 | 0.98..1.02 |
| b13 | zf | 0.8 | 30 x 4 | -0.12..+0.02 | 0.47..1.04 | 0.67..0.77 | 0.87..1.00 | 0.79..1.06 | 0.33..0.63 | xi_lo 0.46 | 1.31..1.34 | 0.00..0.03 | 0.999..1.002 | 0.03..0.11 | 0.04..0.15 | 0.006..0.015 | 8/24 (max chi2 gain 1.77, max abs d change 0.0017 um) | 0.99..1.01 |
| b13 | zf | 1.5 | 30 x 4 | -0.19..+0.04 | 0.17..0.35 | 0.59..0.77 | 0.83..0.97 | 0.74..0.94 | 0.30..0.60 | xi_lo 0.41 | 1.30..1.33 | 0.00..0.03 | 0.999..1.003 | 0.03..0.10 | 0.04..0.13 | 0.004..0.011 | 0/24 | 0.99..1.02 |
| b13 | zf | 2.5 | 15 x 4 | -0.02..+0.11 | 0.09..0.17 | 0.67..0.80 | 0.87..0.93 | 0.78..1.05 | 0.33..0.67 | xi_lo 0.45 | 1.30..1.33 | 0.00..0.01 | 0.998..1.001 | 0.02..0.10 | 0.02..0.14 | 0.003..0.006 | 0/12 | 0.97..1.01 |
| skew | gauss | 0.4 | 30 x 4 | +3.79..+5.16 | 4.52..6.11 | 0.03..0.50 | 0.23..0.87 | 0.76..1.01 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.10..1.68 |
| skew | gauss | 0.8 | 30 x 4 | +0.79..+1.14 | 0.98..1.43 | 0.10..0.53 | 0.47..0.90 | 0.89..1.08 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 1.78..5.06 |
| skew | gauss | 1.5 | 30 x 4 | +0.35..+0.67 | 0.38..0.68 | 0.00..0.33 | 0.00..0.63 | 0.88..1.26 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 4.25..13.96 |
| skew | gauss | 2.5 | 15 x 4 | -0.96..+0.73 | 0.14..1.00 | 0.00..0.67 | 0.00..0.67 | 0.95..3.19 | 0.00 | - | n/a | n/a | n/a | n/a | n/a | n/a | - | 8.48..28.64 |
| skew | zf | 0.4 | 30 x 4 | -1.04..+0.95 | 1.39..4.79 | 0.60..0.83 | 0.93..0.97 | 0.70..0.96 | 0.03..0.23 | xi_lo 0.11, xi_hi 0.02 | 1.92..1.98 | 0.22..0.35 | 1.301..1.317 | 0.17..0.57 | 0.19..0.66 | 0.020..0.058 | 5/24 (max chi2 gain 0.27, max abs d change 0.0015 um) | 1.00..1.02 |
| skew | zf | 0.8 | 30 x 4 | +0.04..+0.24 | 0.49..0.90 | 0.53..0.77 | 0.90..1.00 | 0.78..1.14 | 0.00..0.13 | xi_lo 0.04 | 1.94..2.02 | 0.22..0.27 | 1.297..1.301 | 0.13..0.35 | 0.15..0.38 | 0.009..0.024 | 0/24 | 0.98..1.03 |
| skew | zf | 1.5 | 30 x 4 | -0.04..+0.01 | 0.15..0.27 | 0.50..0.87 | 0.83..0.97 | 0.79..0.93 | 0.00..0.07 | xi_lo 0.03 | 1.96..2.04 | 0.26..0.35 | 1.298..1.301 | 0.09..0.25 | 0.09..0.25 | 0.005..0.010 | 1/24 (max chi2 gain 0.00, max abs d change 0.0001 um) | 0.99..1.01 |
| skew | zf | 2.5 | 15 x 4 | -0.03..+0.04 | 0.06..0.13 | 0.67..0.93 | 0.80..1.00 | 0.61..0.85 | 0.00..0.07 | xi_lo 0.02 | 1.96..2.05 | 0.27..0.38 | 1.299..1.301 | 0.09..0.28 | 0.08..0.27 | 0.003..0.006 | 0/12 | 0.98..1.01 |

Pooled over every case run (each data set counted once):

| fit | truth | data sets | failed | SE(d) not finite | cover 1 SE | cover 1.96 SE | share at a bound |
|---|---|---|---|---|---|---|---|
| gauss | gauss | 420 | 0 | 0 | 0.717 | 0.936 | 0.000 |
| z1 | gauss | 420 | 0 | 0 | 0.683 | 0.943 | 0.500 |
| zf | gauss | 420 | 0 | 0 | 0.705 | 0.931 | 0.505 |
| gauss | heavy | 420 | 0 | 0 | 0.186 | 0.390 | 0.000 |
| z1 | heavy | 420 | 0 | 0 | 0.657 | 0.924 | 0.038 |
| zf | heavy | 420 | 0 | 0 | 0.664 | 0.917 | 0.033 |
| gauss | b13 | 420 | 0 | 0 | 0.029 | 0.150 | 0.000 |
| z1 | b13 | 420 | 0 | 0 | 0.686 | 0.917 | 0.438 |
| zf | b13 | 420 | 1 | 0 | 0.702 | 0.916 | 0.411 |
| gauss | skew | 420 | 0 | 0 | 0.250 | 0.500 | 0.000 |
| zf | skew | 420 | 0 | 0 | 0.719 | 0.929 | 0.057 |

### (d) Cost

One forward evaluation of the 1-D model (`validate_model.py` V5; the fast path is the cubic-Hermite H used by the
recovery fits); per fit, the mean wall time and scipy's `nfev` (the Jacobian's evaluations not counted: about
1 + p per iteration more) from the recovery, on a CPU shared with another agent's processes (load about 4 on 2 cores), so slower than
alone **[reasoning]**. For reference only: the 2-D line renderer (Block 35, partition) took 0.03-0.9 s per call on the
same tubes (V1).

| d (um) | forward, Gaussian (ndtr) ms | SGT exact ms | SGT fast ms | fit: gauss s (nfev) | z1 s (nfev) | zf s (nfev) |
|---|---|---|---|---|---|---|
| 0.4 | 2.0 | 11.4 | 3.5 | 0.31 (11) | 1.17 (14) | 1.44 (15) |
| 0.8 | 2.7 | 11.4 | 6.3 | 0.31 (8) | 1.18 (10) | 1.43 (12) |
| 1.5 | 7.8 | 16.6 | 6.2 | 0.38 (8) | 1.22 (11) | 1.45 (12) |
| 2.5 | 6.7 | 25.0 | 9.5 | 0.76 (11) | 1.93 (12) | 2.38 (13) |

### Extra checks: the discretisation and the correlation structure

`discretization_check.py` (the slabs halved, the fine lattice halved; heavy, b13, skew at 0.4 and 2.5 um, mu 1.5 /um, the axis 0.5 um off) -> `discretization_check.out` **[run here]**:

```
default (dzeta 0.02, f 8)  d 0.4 heavy Gaussian-fit bias +6.662 %; SE(d) known 0.00964, zf 0.01064 (infl 1.104); corr(d, beta, xi, zeta) -0.092 -0.181 +0.113
default (dzeta 0.02, f 8)  d 0.4 b13   Gaussian-fit bias +13.022 %; SE(d) known 0.01137, zf 0.01260 (infl 1.108); corr(d, beta, xi, zeta) -0.096 -0.185 +0.100
default (dzeta 0.02, f 8)  d 0.4 skew  Gaussian-fit bias +4.812 %; SE(d) known 0.00893, zf 0.00987 (infl 1.105); corr(d, beta, xi, zeta) -0.130 -0.214 +0.129
default (dzeta 0.02, f 8)  d 2.5 heavy Gaussian-fit bias +0.072 %; SE(d) known 0.00222, zf 0.00236 (infl 1.060); corr(d, beta, xi, zeta) +0.083 +0.093 -0.312
default (dzeta 0.02, f 8)  d 2.5 b13   Gaussian-fit bias +0.496 %; SE(d) known 0.00236, zf 0.00252 (infl 1.067); corr(d, beta, xi, zeta) +0.094 +0.105 -0.311
default (dzeta 0.02, f 8)  d 2.5 skew  Gaussian-fit bias +0.037 %; SE(d) known 0.00212, zf 0.00224 (infl 1.056); corr(d, beta, xi, zeta) +0.104 +0.110 -0.282
dzeta 0.01                 d 0.4 heavy Gaussian-fit bias +6.604 %; SE(d) known 0.00963, zf 0.01063 (infl 1.104); corr(d, beta, xi, zeta) -0.093 -0.182 +0.114
dzeta 0.01                 d 0.4 b13   Gaussian-fit bias +13.038 %; SE(d) known 0.01137, zf 0.01259 (infl 1.108); corr(d, beta, xi, zeta) -0.098 -0.187 +0.101
dzeta 0.01                 d 0.4 skew  Gaussian-fit bias +4.783 %; SE(d) known 0.00892, zf 0.00986 (infl 1.105); corr(d, beta, xi, zeta) -0.130 -0.214 +0.131
dzeta 0.01                 d 2.5 heavy Gaussian-fit bias +0.072 %; SE(d) known 0.00222, zf 0.00236 (infl 1.060); corr(d, beta, xi, zeta) +0.083 +0.093 -0.313
dzeta 0.01                 d 2.5 b13   Gaussian-fit bias +0.495 %; SE(d) known 0.00236, zf 0.00252 (infl 1.067); corr(d, beta, xi, zeta) +0.093 +0.105 -0.311
dzeta 0.01                 d 2.5 skew  Gaussian-fit bias +0.036 %; SE(d) known 0.00212, zf 0.00224 (infl 1.057); corr(d, beta, xi, zeta) +0.104 +0.110 -0.282
f 16                       d 0.4 heavy Gaussian-fit bias +6.667 %; SE(d) known 0.00671, zf 0.00745 (infl 1.110); corr(d, beta, xi, zeta) -0.051 -0.152 +0.025
f 16                       d 0.4 b13   Gaussian-fit bias +12.675 %; SE(d) known 0.00800, zf 0.00897 (infl 1.121); corr(d, beta, xi, zeta) -0.026 -0.131 +0.022
f 16                       d 0.4 skew  Gaussian-fit bias +4.534 %; SE(d) known 0.00622, zf 0.00688 (infl 1.107); corr(d, beta, xi, zeta) -0.089 -0.183 +0.037
f 16                       d 2.5 heavy Gaussian-fit bias +0.107 %; SE(d) known 0.00225, zf 0.00227 (infl 1.009); corr(d, beta, xi, zeta) +0.082 +0.090 +0.065
f 16                       d 2.5 b13   Gaussian-fit bias +0.379 %; SE(d) known 0.00240, zf 0.00244 (infl 1.017); corr(d, beta, xi, zeta) +0.107 +0.117 +0.062
f 16                       d 2.5 skew  Gaussian-fit bias +0.171 %; SE(d) known 0.00215, zf 0.00216 (infl 1.005); corr(d, beta, xi, zeta) +0.046 +0.052 +0.075
wrote discretization_check.json
```

`lattice_check.py` (the fine factor f = 8, 16, 32, 64; heavy truth, mu 1.5 /um, the axis 0.5 um off) -> `lattice_check.out` **[run here]**:

```
d 0.4 f  8: SE(d) known 0.00964, zf 0.01064 (infl 1.104); SE(v0) zf 0.00417; corr(d, v0) known -0.404, zf -0.334; corr(d, beta, xi, zeta) -0.092 -0.181 +0.113
d 0.4 f 16: SE(d) known 0.00671, zf 0.00745 (infl 1.110); SE(v0) zf 0.00342; corr(d, v0) known -0.346, zf -0.234; corr(d, beta, xi, zeta) -0.051 -0.152 +0.025
d 0.4 f 32: SE(d) known 0.00783, zf 0.00865 (infl 1.106); SE(v0) zf 0.00357; corr(d, v0) known +0.221, zf +0.113; corr(d, beta, xi, zeta) -0.067 -0.165 +0.029
d 0.4 f 64: SE(d) known 0.00785, zf 0.00868 (infl 1.105); SE(v0) zf 0.00354; corr(d, v0) known -0.003, zf +0.003; corr(d, beta, xi, zeta) -0.067 -0.165 -0.007
d 0.4: recovery (f = 8 data and fits), empirical SD of d_hat: gauss-truth model n/a; heavy truth z1 0.00943, zf 0.00915
d 2.5 f  8: SE(d) known 0.00222, zf 0.00236 (infl 1.060); SE(v0) zf 0.00122; corr(d, v0) known +0.603, zf +0.599; corr(d, beta, xi, zeta) +0.083 +0.093 -0.312
d 2.5 f 16: SE(d) known 0.00225, zf 0.00227 (infl 1.009); SE(v0) zf 0.00128; corr(d, v0) known -0.273, zf -0.212; corr(d, beta, xi, zeta) +0.082 +0.090 +0.065
d 2.5 f 32: SE(d) known 0.00204, zf 0.00205 (infl 1.004); SE(v0) zf 0.00124; corr(d, v0) known +0.034, zf +0.021; corr(d, beta, xi, zeta) +0.070 +0.076 -0.002
d 2.5 f 64: SE(d) known 0.00214, zf 0.00215 (infl 1.005); SE(v0) zf 0.00123; corr(d, v0) known -0.009, zf -0.009; corr(d, beta, xi, zeta) +0.074 +0.080 +0.005
d 2.5: recovery (f = 8 data and fits), empirical SD of d_hat: gauss-truth model n/a; heavy truth z1 0.00284, zf 0.00269
wrote lattice_check.json
```

`corr_structure.py` (the zf model's correlation matrix at the truth, all 64 cases) -> `corr_structure.out`, last line **[run here]**:

```
ranges over the 64 cases: corr(beta, xi) +0.971..+0.997; corr(v0, zeta) -0.863..-0.738; corr(d, mu) -0.963..-0.142; the smallest eigenvector's d component <= 0.42; its two largest components beta/xi in 64 of 64
```

Figures: `fig_d_error_se.png` (top: the recovered d's relative error, mean +- SD, for the three fits, with the Gaussian fit's noise-free bias; bottom: SE(d)/d from the Fisher information with the shape known, (beta, xi) free and (beta, xi, zeta) free, and the recovery's SD of d_hat), `fig_correlations.png` (corr(d, beta), corr(d, xi), corr(d, zeta) of the zf model, unweighted and with the centre weights), `fig_bias_gauss.png` (part (b) against d).


## Reading

Every number here comes from the tables above or the extra checks' outputs **[run here]** unless marked
**[reasoning]** or with a source.

**1. For which nodes the shape is identifiable alongside d.**

- No class of node shows a degeneracy between $d$ and the shape. Freeing $(\beta, \xi)$ with $\zeta = 1$ multiplies
  SE(d) by 1.00-1.10, freeing $(\beta, \xi, \zeta)$ by 1.01-1.13. With Block 39's centre weights the factor is
  1.00-1.16, and with the backgrounds known (Block 30's `fit_node`) it is 1.01-1.22. $|{\rm corr}(d, \beta)| \le 0.31$,
  $|{\rm corr}(d, \xi)| \le 0.34$ and $|{\rm corr}(d, \zeta)| \le 0.31$ (with the weights $\le 0.42$). On a converged
  fine lattice (f = 64) the inflation is 1.105 at 0.4 um and 1.005 at 2.5 um, with every $|{\rm corr}(d, {\rm shape})|
  \le 0.17$ (`lattice_check.out`).
- In recovery the SGT fits' mean error of $d$ stays within $\pm$1.9 % at 0.4 um (RMSE 1.4-6.5 %), within $\pm$0.24 % at
  0.8 um, $\pm$0.20 % at 1.5 um and $\pm$0.11 % at 2.5 um. Pooled over the 420 data sets of each (truth, model), $d$'s
  SE covers the truth in 0.66-0.72 of them at 1 SE (nominal 0.68). At 1.96 SE it covers 0.92-0.94 (nominal 0.95; the
  binomial SE is about 0.01, so $\hat d$'s tails are slightly wider than its SE says). Of the 4620 fits, only $\xi$
  ever ended at a bound ($\xi = 0$ in 824 SGT fits, $\xi = 2$ in 8); never $d$, $\mu$, $v_0$, $z_{\rm ax}$,
  $\beta$ or $\zeta$.
- The ill-conditioning lies inside the shape: condition numbers of 357-6342, against 8-163 with the shape known.
  ${\rm corr}(\beta, \xi)$ = 0.971-0.997, and the weakest direction of the correlation matrix is $\beta$ against
  $\xi$ in all 64 cases. That direction's $d$ component is at most 0.03 at 2.5 um, 0.07 at 1.5, 0.16 at 0.8 and 0.42 at
  0.4 um (`corr_structure.out`). The data fix one combination of $(\beta, \xi)$ -- how much weight the shoulders and
  tails carry at the given FWHM -- not the two separately **[reasoning]**.
- **Thick nodes (1.5, 2.5 um):** the shape is resolved (SE($\beta$) 0.05-0.32, SE($\xi$) 0.06-0.29, SE($\zeta$)
  0.002-0.010). Identifiable alongside $d$.
- **0.8 um:** SE($\beta$) 0.06-0.37, SE($\xi$) 0.07-0.36, SE($\zeta$) 0.006-0.020. Identifiable; the correlations of $d$
  with $(\beta, \xi)$ are the largest here (0.31, 0.34), at a cost of at most 8 % in SE(d).
- **Thin nodes (0.4 um):** $d$ stays identifiable (SE(d) x 1.08-1.11), but the shape is loosely determined. On faint
  ones ($\mu$ 0.5 /um) SE($\beta$) is 0.31-0.75 and SE($\xi$) 0.36-0.76, and in recovery the SD of $\hat\beta$
  reaches 0.57 and of $\hat\xi$ 0.71. At $\mu$ 1.5 /um SE($\beta$) is 0.12-0.29 and SE($\xi$) 0.14-0.30. The fitted
  $(\beta, \xi)$ of a faint thin node says little about the kernel; its $d$ stays unbiased because the shapes the data
  allow all give about the same $d$ **[reasoning, from the small corr(d, shape)]**.
- **Practical points.** (i) $\xi$ ends at its lower bound 0 in about half of the fits when the true $\xi$ is 0 (0.41-0.51
  pooled, the Gaussian and the $\beta$ 1.3 truths), as a parameter on its boundary does **[reasoning]**; otherwise in
  0.03-0.06 of them. Under D-053 (ii) a shape at its bound keeps a fit in $\mathcal S$ **[SPEC]**. (ii) Started from
  today's Gaussian fit, the solver sometimes stops early along the $\beta$-$\xi$ ridge. Block 12's other start shapes
  found a lower $\chi^2$ in 23 of 588 tries (by at most 1.77) and moved $d$ by at most 0.0017 um (at most 0.4 SE). One of the
  2940 SGT fits (zf, $\beta$ 1.3 truth, 1.5 um) spent `max_nfev` = 200 on the ridge, with its $d$ within 0.45 SE.

**2. Whether $\zeta$ should be freed.**

- $\zeta$ is the best-determined shape parameter: SE($\zeta$) is 0.002-0.057 (the largest on faint thin nodes), so a
  skew of 1.3 lies 5-150 SE from 1. On the skewed truth the zf fits' median $\hat\zeta$ is 1.297-1.317 (SD
  0.003-0.058).
- Freeing it multiplies SE(d) by 1.004-1.053 over the $\zeta = 1$ fit (f = 8); on the converged lattice the whole
  shape $(\beta, \xi, \zeta)$ costs x 1.105 at 0.4 um and x 1.005 at 2.5 um. $\zeta$ is odd about the axis and $d$ even, so the two are nearly orthogonal **[reasoning]**: the
  $|{\rm corr}(d, \zeta)| \le 0.31$ seen at f = 8 comes from the lattice and is about 0 at f = 64. $\zeta$ trades
  with the lateral offset instead: ${\rm corr}(v_0, \zeta)$ = -0.74 to -0.86, and SE($v_0$) grows 1.5-2.0 times.
- For $d$, freeing $\zeta$ hardly matters. On the skewed truth the symmetric option ($\zeta = 1$) finds $d$ within
  -0.44..+0.21 % (at most 2.8 of its SE), against -1.08..+4.81 % for today's Gaussian fit (part (b)). What
  $\zeta = 1$ leaves behind is a visible misfit (noise-free $\chi^2$ 37-11681 on 462 degrees of freedom) and a
  shifted $v_0$.
- Reading **[reasoning]**: freeing $\zeta$ is safe (no degeneracy, well determined), but it is not needed to protect
  $d$; the case for it is the fit's adequacy ($\chi^2$) and the lateral position. Keeping $\zeta = 1$ as the default
  (D-222) and freeing it where the residuals are asymmetric is consistent with these numbers. So is freeing it always,
  at a cost of at most 5 % in SE(d) over $\zeta = 1$ (13 % over the shape known).

**3. How large the Gaussian assumption's bias in d is.**

- Noise-free (part (b), the actual fit; each cell spans $\mu$ 0.5, 1.5 /um and the axis on and 0.5 um off $k^*$):

  | truth | 0.4 um | 0.8 um | 1.5 um | 2.5 um |
  |---|---|---|---|---|
  | heavy (2, 0.5, 1) | +6.4..+6.7 % | +1.3..+1.7 % | +0.3..+1.0 % | -1.5..+0.7 % |
  | $\beta$ 1.3 (1.3, 0, 1) | +12.7..+13.1 % | +2.0..+2.9 % | +0.8..+2.1 % | -2.6..+1.4 % |
  | skewed (2, 0.3, 1.3) | +3.9..+4.8 % | +0.7..+1.0 % | +0.3..+0.7 % | -1.1..+0.8 % |

  With noise, the recovery's mean errors agree: +3.8..+14.1 % at 0.4 um, +0.8..+2.8 % at 0.8 um.
- The bias shrinks as the tube widens: the kernel's shoulders matter less against a wide chord profile
  **[reasoning]**. It is positive on thin and mid-sized nodes, where the Gaussian widens the tube to make up for the
  SGT's shoulders **[reasoning]**, and turns negative on the dark thick tube with its axis on the focus plane.
- Against the noise it is 0.8-6.5 SE on 0.4 um nodes, up to 12.7 SE at 1.5 um and 31.7 SE at 2.5 um ($\mu$ 1.5 /um).
  The Gaussian fit's SE therefore does not cover the truth: pooled coverage 0.03-0.25 at 1 SE and 0.15-0.50 at
  1.96 SE, while the SGT fits cover it (point 1).
- The misfit shows in $\chi^2$ (noise-free excess 10-12925; recovery $\chi^2$/dof 1.01-28.6), except on faint thin
  nodes with heavy tails. There (heavy truth, 0.4 um, $\mu$ 0.5 /um) the excess is 10-11, a third of $\chi^2$'s SD of
  30, so a bias of 6.4-6.7 % in $d$ is invisible in the fit's quality.
- The linearised bias (the first Gauss-Newton step) matches the fit to within 0.8 points for the symmetric truths. For
  the skewed truth it does not (up to 4 points off): that misfit is too large for a first-order step
  **[reasoning]**. The fit is the reference.

**4. A side finding: the line renderers' lattice and the Jacobian.** Not asked; reported because it bears on how the
option's SEs read.

- Blocks 26 and 35 point-sample the cross-section on the fine lattice $h = p/8$. The model's values converge as $h$
  shrinks, but its local derivative in $d$ (and in $v_0$) is that of the sampled tube: the strip between the
  outermost sample and the edge, where the chord's derivative diverges, is missing, and the two edges are sampled
  unequally **[reasoning]**.
- Measured (`lattice_check.out`, `discretization_check.out`, (a)'s step check): SE(d) at $d$ 0.4 um, $\mu$ 1.5 /um is
  0.0096 um at f = 8, 0.0067 at 16, and 0.0078 at 32 and 64. corr($d$, $v_0$) at 2.5 um is +0.60 at f = 8, -0.27 at
  16, and about 0 at 32-64. Doubling every finite-difference step moves the SE(d) of thick tubes by up to 18 %. The
  shape's SEs move by less than 2 %, and the (b) biases by at most 0.35 points (f 16) or 0.06 points (dzeta 0.01).
  On data from the f = 8 model, the recovery's SD of $\hat d$ is 0.6-1.3 times the solver's SE.
- Block 30 reports SE(d) from scipy's finite-difference Jacobian (relative steps near 1e-8) on this lattice, so its
  SE(d) and its corr($d$, $v_0$) carry this error, tens of percent on thin nodes **[reasoning]**. An exact integral
  of the cross-section across its edges, or a finer lattice, would remove it **[reasoning]**. This is no departure
  from the SPEC: Block 26's oracles bound the values, not their derivatives **[SPEC]**.

**5. Limits [reasoning].**

- The model is one-dimensional: a flat tube along the pixel grid, read at pixel centres. There is no tilt, no heading
  off the grid (bilinear reads between pixel centres), no caps, no neighbours and no hybrid absorption (the per-node
  fit's).
- Every truth has the Gaussian law's FWHM at every defocus, which is the option's own assumption. A kernel whose width
  departs from the law (the in-focus-width question, D-165, D-201) is not tested, and it is where the confound between
  $d$ and the kernel is strongest (`docs/TEEG_clear_node_fit_handoff_2026-11-19.md`, section 7) **[read]**.
- The noise is white, 1.1 gl on 150 gl. Real profiles carry background texture ($\chi^2/N$ 3.5-50 on the pilot's five
  nodes, the same handoff, section 4 **[read]**), which a free shape may absorb.
- The $\beta$ = 1.3 truth has a sharper core, not a flatter one; a flatter core ($\beta > 2$) was not run.
- Recovery at 2.5 um used 15 data sets per case (time); the other diameters used 30.
