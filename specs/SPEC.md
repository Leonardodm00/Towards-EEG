# SPEC -- Allen dendrite-diameter re-measurement (`allen_diameter`)

Single source of truth for what the code must do. The code follows this file;
when the two disagree, fix whichever is wrong and record why in the decision
log (`TEEG_decisions_and_ideas_log.md`, project knowledge; entries D-018 to
D-024 bind this spec). The integration tester reads only the repository, so
every constraint the code must satisfy is stated here.

Notation rules: every symbol is defined at first use; a conditional quantity
keeps its conditioning every time it appears; quantifiers are stated.

Method documents (decision D-020; cited below as "handoff Eq. n", "procedure
Eq. (n) / s.n", "mathematics Eq. (n) / s.n", "impl-handoff (S n)"):
`Passive Features/Diameter Re-measurement/docs/handoff_diameter_remeasurement.md`,
`.../TEEG_diameter_bias_table_procedure_2026-10-04.md`,
`.../TEEG_diameter_bias_table_mathematics_2026-10-04.md`,
`.../TEEG_diameter_implementation_handoff_2026-10-06.md`.

Last reviewed at commit: (first commit of branch `sci/diameter-pipeline`, 2026-10-06)

## 0. Coverage

Paths this spec covers:

- `Passive Features/Diameter Re-measurement/src/`
- `Passive Features/Diameter Re-measurement/scripts/`
- `Passive Features/Diameter Re-measurement/tests/`

Everything else in the repository is not specified here and is out of scope
for the tester. The four flat modules `src/allen_image_*.py` and the two
suites `tests/smoke/smoke_allen_image.py`, `tests/smoke/robustness_registration.py`
are the 2026-09-23 image-access code imported byte-identical (Block 0); their
entries are `Confirmed: inferred` and their own suites are their tests.

## 1. Scientific goal

Run B of the passive fit gives L2/L3 membrane capacitance 2-3 times the
reference value; the working hypothesis is that the Allen SWC radii are too
thin. This code re-measures dendrite diameters from Allen's 63x brightfield
image stacks (specimen 529878215 first, D7) and writes corrected SWC files in
the archive's own format (D-013) for the Cm refit.

Deliverables, per cell: a per-node CSV (identity, registration, focus and
path, background, fit, correction, flags, final diameter; columns in
Block 8), a corrected SWC with radius = d_final / 2 on dendrite nodes (types
3, 4) and everything else byte-identical (D-013), and the dendritic
membrane-area ratio against Allen's radii. Per configuration: a bias table
(Block 6) with its configuration dictionary C. Consumers: the user; the I_h
fitter (`Passive Features/HPC script/Ih Fit/run_ih_fit.py --swc-dir`).

## 2. Mathematical formulation

### 2.1 Frames and objects

Global image frame: $(x, y)$ in um along pixel columns and rows
(pitch $p_{\rm x} = 0.1144$ um), $z$ along the optical axis in stage units
(plane spacing $\Delta z = 0.28$ um; plane index $k$; depth $z_k$). A dendrite
node $i$ has SWC position $p_i \in \mathbb{R}^3$ (um) and refined centre
$c_i \in \mathbb{R}^3$. The branch near node $i$ is a round straight tube of
diameter $d$ (um), direction $\hat t_i = (\cos\varphi_i\cos\theta_i,
\cos\varphi_i\sin\theta_i, \sin\varphi_i)$ with heading
$\theta_i = \operatorname{atan2}(t_y, t_x) \in (-\pi, \pi]$ and tilt
$\varphi_i = \arcsin|t_z| \in [0, \pi/2]$; measuring axis
$\hat y_i = (-\sin\theta_i, \cos\theta_i, 0)$ (handoff Eq. 5).

### 2.2 Per-node measurement (Blocks 3, 5; handoff Eqs. 1-5, 9; D-018; D-019)

Focus score, for node $j$ and plane $k$, with $\tilde I_{j,k}$ the raw profile
along $\hat y_j$ through $p_{j,\parallel}$ smoothed by Gaussian weights of
$s$ = `focus_smooth_px` pixels, and $B_{j,k}$ the median of the profile ends
$|v| > 1.5$ um (D-023, `focus_bg_rule = "profile_ends_median"`):

$$F_{j,k} = -\ln\big(\tilde I_{j,k,\min} / B_{j,k}\big), \qquad k^*_j = \operatorname*{arg\,max}_k F_{j,k}. \tag{1}$$

Sub-plane depth (handoff Eq. 2; on by default), with
$F_\mp = F_{j,k^*_j \mp 1}$, $F_0 = F_{j,k^*_j}$:
$z_j = z_{k^*_j} + \Delta z\,(F_- - F_+)/[2(F_- - 2F_0 + F_+)]$, clipped to
$z_{k^*_j} \pm \Delta z/2$; the middle of the plateau when $F$ is flat-topped.

Centre: $c_j = (p_{j,\parallel} + \hat v_{0,j}(-\sin\theta_j, \cos\theta_j),\ z_j)$
with $\hat v_{0,j}$ from the fit (3) in plane $k^*_j$ (handoff Eq. 3).

Direction for node $i$: total-least-squares line through the centres
$c_j$ with $|s_j - s_i| \le L/2$ ($s$ = path length, $L$ = `line_fit_window_um`),
in um in the global frame (D6: never in pixel/plane units):
$\hat t_i$ = leading eigenvector of $\frac{1}{|W_i|}\sum_{j \in W_i}(c_j - \bar c_i)(c_j - \bar c_i)^\top$,
oriented away from the soma (handoff Eq. 4). The first pass uses the raw
SWC direction; one redraw with the fitted direction (`direction_redraws`).

Profile (handoff Eq. 9): $I_i(v_n) = I_{k^*_i}(c_{i,\parallel} + v_n \hat y_i)$,
$v_n \in [-3, 3]$ um in steps of $p_{\rm x}$, bilinear interpolation,
averaged over $\pm$`along_branch_avg_um` along $\hat e_u$ when enabled.

Background (D-018.1, D-023): $\bar B_i$ = median of the node's block in
plane $k^*_i$ with every pixel within (Allen radius + `bbar_mask_margin_um`)
of any SWC segment in the block masked out; phantoms mask the same way
around their known axis.

Fit (D-019.1; $\bar B_i$ fixed, $\sigma_{\rm fit}$ fixed, $\varphi_i$ from the
line fit), with $s_d(u) = \sqrt{1 - (2u/d)^2}$ for $|u| \le d/2$, else 0, and
$g_\sigma$ the 1-D Gaussian of standard deviation $\sigma$ (unit integral):

$$T_{d,\mu,v_0}(v \mid \varphi_i) = \exp\!\Big(-\frac{\mu\,d}{\cos\varphi_i}\,s_d(v - v_0)\Big), \qquad (\hat d_i, \hat\mu_i, \hat v_{0,i}) = \operatorname*{arg\,min}_{d, \mu, v_0} \sum_n \Big[I_i(v_n) - \bar B_i\,\big(T_{d,\mu,v_0}(\cdot \mid \varphi_i) * g_{\sigma_{\rm fit}}\big)(v_n)\Big]^2 \tag{3}$$

over the bounds of `MeasureConfig` (fit_d_bounds_um, fit_mu_bounds_per_um,
fit_v0_bounds_um), `scipy.optimize.least_squares`, multi-start at
$d_0 \times$ `fit_multistart_factors`. Derived: $\hat\alpha_i = \hat\mu_i \hat d_i / \cos\varphi_i$
(diagnostics only). With $\mu$ free per node (D-023) the fit has the same
minimiser in $(d, v_0)$ as the $\alpha$-parameterised D-018.2 (regression
guard, Block 3).

Selection $\mathcal S$ (procedure s.3.2; D-024), applied identically to
phantoms and real nodes: converged, no parameter at a bound, registration
verdict on the branch (real data only), not `faint`, no second dip
(`crossing`), not `stack_edge`, $\hat\alpha_i \le$ `alpha_dark_flag`.
**Tilt is never a selection criterion** (D-024); `steep`
($\tan\varphi_i \ge$ `steep_tan_diagnostic`) and `vertical`
($\varphi_i >$ `phi_vertical_deg`) are reported columns.

### 2.3 Phantom geometry and rendering (Blocks 2, 4; impl-handoff (S1)-(S5); procedure Eqs. 5-6)

For a tube of radius $r = d/2$, tilt $\varphi$, heading $\theta$, node point
$c$, and every $(x, y, z)$: $u = (x - c_x)\cos\theta + (y - c_y)\sin\theta$,
$v = -(x - c_x)\sin\theta + (y - c_y)\cos\theta$, $w = z - c_z$ (S1);
membership $v^2 + (w\cos\varphi - u\sin\varphi)^2 \le r^2$ (S2). The
vertical line through $(x, y)$ is inside the tube iff $|v| \le r$ (and
inside the end cut at half-length $U$ = `U_um`: $|u| \le U$ for
`end_cut = "vertical"`, the axial arc length $|u\cos\varphi + w\sin\varphi| \le U$
for `"axial"`; Block 2) and
$z \in [z_{\rm lo}, z_{\rm hi}]$, $z_{\rm lo/hi} = c_z + u\tan\varphi \mp \ell(v)/2$,
$\ell(v) = 2\sqrt{r^2 - v^2}/\cos\varphi$ (S3). These are the round-section
forms ($k$ = `cross_section_aspect` = 1); Block 2 states the squashed-section
generalisation $k \ne 1$. Slab $j$ spans
$[\zeta_j - \delta\zeta/2, \zeta_j + \delta\zeta/2]$; its absorbance along the
vertical ray and the light reaching it (slabs numbered in the light's
direction, `light_direction`):

$$a_j(x, y) = \mu\max\{0, \min(z_{\rm hi}, \zeta_j + \tfrac{\delta\zeta}{2}) - \max(z_{\rm lo}, \zeta_j - \tfrac{\delta\zeta}{2})\}, \qquad T_{<j}(x, y) = \exp\!\big(-\mu\max\{0, \min(z_{\rm hi}, \zeta_j - \tfrac{\delta\zeta}{2}) - z_{\rm lo}\}\big) \tag{S4}$$

(both 0 resp. 1 for $|v| > r$; $\sum_j a_j = \mu\ell(v)$ when the slabs tile
the column). Absorbed-light partition and one output plane $k$ with a
circular Gaussian kernel $G_{\sigma_{\rm r}(\delta)}$, $\delta = \zeta_j - z_k$:

$$\Delta A_j = T_{<j}\,(1 - e^{-a_j}), \qquad I_k(x, y) = B\Big[1 - \sum_j \big(\Delta A_j * G_{\sigma_{\rm r}(\zeta_j - z_k)}\big)(x, y)\Big], \tag{5-6}$$

computed as one forward FFT per slab and one inverse FFT per plane on an
array zero-padded to twice the grid (linear, not circular, convolution;
impl-handoff "FFT form"); the `direct` backend is the reference. Kernel
table $\sigma_{\rm r}(\delta)$: linear interpolation of
`kernel_table_(delta|sigma)_um` in $|\delta|$, continued beyond the last
tabulated offset by `kernel_continuation` (`linear`:
$\sigma_{\rm r}(\delta_{\max}) + \gamma(|\delta| - \delta_{\max})$ with
$\gamma$ = `kernel_continuation_slope`; `frozen`; `proportional`:
$\sigma_{\rm r}(\delta_{\max})|\delta|/\delta_{\max}$). Camera chain in this
order (procedure s.3.6 step 6): block-average to $p_{\rm x}$, black level
and gain, Gaussian noise `noise_sd_gl`, 8-bit rounding, JPEG (quality or
Allen's own tables). Alternative absorption treatments by name:
`linear` ($I_k = B[1 - \sum_j a_j * G]$, mathematics Eq. 11) and
`ray_world` (impl-handoff (S6), geometric optics; Block 9).

### 2.4 Bias table and correction (Blocks 6, 7; procedure Eqs. 1-3, 7; mathematics Eqs. 19-21; D-024)

For a phantom of true $(d, \varphi)$ and nuisances $\xi$ (heading, sub-pixel
offset, axis depth, noise, node jitter), the pipeline's fitted diameter
before the draw is the random variable $\hat D$; the target is
$m(d, \varphi \mid \mathcal C) = \mathbb E_\xi[\hat D \mid d, \varphi, \mathcal C, \mathcal S]$
and $b = m/d$. Design (D-024): $N$ independent replicates with
$\log d \sim U[\log d_{\min}, \log d_{\max}]$, $\varphi \sim U[0, 90°)$,
$\theta \sim U[0, \pi)$, $\log\mu \sim U[\log\mu_{\min}, \log\mu_{\max}]$,
from one `numpy.random.Generator(seed)`. Estimator: the smoothing thin-plate
spline $\hat m(\log d, \varphi)$ through the retained $(\log d_n, \varphi_n, \hat d_n)$
(`scipy.interpolate.RBFInterpolator`, `kernel="thin_plate_spline"`,
smoothing by k-fold cross-validation); $\hat\tau$ and the failure rate from
Gaussian kernel weights of bandwidth `tau_kernel_bandwidth`. Correction at a
real node: $\tilde d_i$ solves $\hat m(d, \varphi_i \mid \mathcal C) = \hat d_i$
by `brentq` on $d \in [d_{\min}, d_{\max}]$, requiring $\hat m(\cdot, \varphi_i)$
strictly increasing there (else flag `non_monotone`) and
$\hat d_i \in [\hat m(d_{\min}, \varphi_i), \hat m(d_{\max}, \varphi_i)]$ (else
`out_of_domain`); flag `large_correction` when
$|\hat b(\tilde d_i, \varphi_i) - 1| >$ `bias_flag_threshold` (D5),
`high_failure` when the local failure rate $>$ `max_failure_rate`. The
correction refuses a table whose estimator signature (`config.estimator_signature()`)
differs from the fit's (procedure s.3.11). Fill (D5): flagged nodes take the
median of unflagged same-branch neighbours, else Allen's radius; then a
running median of `median_window_nodes` along the branch.

### Notation

| Symbol | Meaning | Type / shape | Units |
|---|---|---|---|
| $p_{\rm x}$, $\Delta z$ | pixel pitch, plane spacing | float | um |
| $i, j$ | SWC dendrite node indices | int | - |
| $k$, $z_k$, $k^*_i$ | plane index, its depth, sharpest plane of node $i$ | int, float, int | -, um, - |
| $d, r, \mu$ | tube diameter, radius, absorption coefficient | float > 0, > 0, >= 0 | um, um, 1/um |
| $\varphi, \theta$ | tilt out of the image plane, heading | float in [0, pi/2], (-pi, pi] | rad |
| $\hat d_i, \hat\mu_i, \hat v_{0,i}, \hat\alpha_i$ | fitted diameter, coefficient, axis offset; derived centre-line absorbance | float | um, 1/um, um, - |
| $\bar B_i$, $B$ | background estimate (computed), rendered background (configured) | float | grey levels |
| $\sigma_{\rm fit}$, $\sigma_{\rm r}(\delta)$ | fit blur; rendering kernel width at defocus $\delta$ | float > 0 | um |
| $a_j, T_{<j}, \Delta A_j$ | slab absorbance, light reaching slab $j$, absorbed fraction | arrays (n_y, n_x) | -, -, - |
| $I_k(x, y)$ | rendered or real plane $k$ | array (n_y, n_x) | grey levels |
| $m, b, \hat m, \hat b, \hat\tau$ | mean response, bias factor; their estimates; scatter of $\hat D/d$ | float | um, -, um, -, - |
| $\tilde d_i$, $d_{{\rm final},i}$ | corrected diameter; after flags and fill | float | um |
| $\mathcal C$, $\mathcal S$ | configuration dictionary; selection event | dict, predicate | - |

## 3. Data contracts

| Item | Source / format | Shape and axis order | dtype | Units | Conventions |
|---|---|---|---|---|---|
| SWC file | Allen `reconstruction.swc`; `swc_io.read_swc` | N nodes: ids, types, parent (N,), xyz (N, 3), radius (N,) | int, float64 | um | 7 fields per data line; `#` comments kept verbatim; types 3, 4 = dendrite |
| image block | `allen_image_io.fetch_zblock` (Colab) or the renderer | `block` (n_planes, height, width); `ks` plane indices; `valid` (n_planes,); `frame` CropFrame | uint8 (real) / float64 (synthetic before the camera chain) | grey levels | row = y, column = x; missing planes white (255) and `valid = False` |
| profile | Block 5 | $v_n$ (n_v,), $I_i(v_n)$ (n_v,) | float64 | um, grey levels | $v = 0$ at the node; step $p_{\rm x}$; bilinear |
| per-node CSV | Block 8 | one row per dendrite node | mixed | um, rad, grey levels, 1/um | columns listed in Block 8 |
| replicate table | Block 6 | one row per replicate: draws, outputs, status, flags | mixed | um, rad, 1/um | stored before any selection |
| bias table | Block 6 | `npz` (spline weights/centres, tau grid, failure grid) + JSON of `config.full_signature()` | float64 | - | file name carries `signature_hash("estimator")` |

Conventions: 0-based array indices; SWC node ids are the file's own (not
assumed 1-based); `plane_index = section_number - min(section_number)`
(handoff); z in stage units throughout; angles in radians inside the code,
degrees in config and CSV column names ending `_deg`.

## 4. Global conventions

- Units used internally: um, radians, grey levels, 1/um; converted at the loading boundary.
- Floating-point precision: float64 (uint8 only for real image blocks).
- Randomness: a `numpy.random.Generator` passed explicitly; the seed is `PhantomConfig.seed`, set once in the script.
- Parameters only in `src/allen_diameter/config.py` (D-021); every field carries a `# source:` tag (Block 1 asserts it).
- Code destined for the cluster is pure ASCII with LF line endings (`hpc-python-compat`); `.gitattributes` pins `*.py` to LF.
- Binding decisions:
  - D-013: corrected morphologies = one SWC per specimen, same format as the archive's; only radii change.
  - D-018: $\bar B_i$ fixed from a median (never fitted); $\sigma_{\rm fit}$ fixed.
  - D-019: the fitted darkness parameter is $\mu$; $\varphi_i$ from the line fit; $\alpha$ derived.
  - D-021: open choices are config fields with provisional defaults; alternatives selectable by name; comparisons labelled.
  - D-022: layout as in Coverage; branch `sci/diameter-pipeline`; production table on davinci.
  - D-023: `mu_mode = per_node`, `bbar_region = block_masked`, `focus_bg_rule = profile_ends_median`; $\sigma_{\rm fit}$ a study axis (`sigma_fit_study_um`), deliverable 0.099 um (provisional).
  - D-024: `absorption = partition_vertical`, `kernel_continuation = linear` (0.79); dark flag $\hat\alpha > 1.0$; random design, $U = 10$ um, $\varphi \in [0°, 90°)$, $d$ range temporary; `response_estimator = tps_spline`; no tilt exclusion; D5 (flag 0.2 + fill) and D7 confirmed; jitter 0 until cell 13.
  - D5/D6 (design handoff): correct by inversion of $\hat m$ (not the shortcut $\hat d/b(\hat d)$); compute directions in um in the global frame.

## 5. Blocks

### Block 0: scaffolding and the 2026-09-23 image modules

- Module: `Passive Features/Diameter Re-measurement/src/allen_image_io.py`, `src/allen_image_align.py`, `src/allen_image_measure.py`, `src/allen_image_plot.py`, `src/allen_diameter/__init__.py`, `tests/smoke/smoke_allen_image.py`, `tests/smoke/robustness_registration.py`
- Public API (used downstream): `allen_image_io.fetch_zblock(fetcher, planes, k_lo, k_hi, left, top, width, height, res0_um_px) -> (block, ks, valid, frame)`; `allen_image_io.fetch_swc(specimen_id, out_dir) -> path`; `allen_image_io.SyntheticFetcher`; `allen_image_align.path_through_node`, `plan_path_block`, `registration_check(...) -> dict` (keys include `verdict`, `lateral_offset_um`, `p_value`, `coverage`); `allen_image_align.swc_to_full_px`; `allen_image_measure.perpendicular_profile`, `fwhm_um`; `allen_image_plot.plot_registration_check`
- Inputs/Outputs: as documented in the modules' docstrings (design handoff s.Code)
- Parameters: none from `config.py` (the modules keep their own call-site arguments)
- Library calls relied on: numpy, scipy.ndimage, pandas, requests, Pillow, matplotlib
- Custom code: the modules themselves (byte-identical imports; sha256 prefixes io dae5d24f7f2c, align 14dd899057a8, measure 4614cd949ba1, plot 3b9f1a07e544, smoke d7811c681b0f, robustness 18059c7109ee)
- Test oracles: `python tests/smoke/smoke_allen_image.py` prints `20/20 passed`; `python tests/smoke/robustness_registration.py` prints `40/40 correct`, 0/10 false positives (run from `tests/smoke/` with `src/` on `PYTHONPATH`); the sha256 prefixes above
- Data flow: feeds Blocks 5 (real blocks), 8 (SWC fetch), 11
- Confirmed: inferred (2026-10-06; the modules' intent is the design handoff's)
- Status: smoke-tested 2026-10-06 (20/20, 40/40 re-run in the sandbox; sha256 verified)

### Block 1: configuration

- Module: `src/allen_diameter/config.py`
- Public API: `default_config() -> DiameterConfig`; `DiameterConfig.validate() -> None` (raises ValueError); `estimator_signature() -> dict`; `full_signature() -> dict`; `signature_hash(which) -> str` (16 hex); `to_json() -> str`; `config_from_dict(d) -> DiameterConfig`; `with_sigma_fit(cfg, sigma) -> DiameterConfig`
- Inputs: none (defaults) or a signature dict
- Outputs: frozen dataclasses `AcquisitionConfig`, `MeasureConfig`, `CorrectionConfig`, `RendererConfig`, `PhantomConfig`, `CalibrationConfig` inside `DiameterConfig`; canonical ASCII JSON
- Parameters: every pipeline parameter, with its `# source:` tag (D-021); the choice tuples `MU_MODES`, `BBAR_REGIONS`, `FOCUS_BG_RULES`, `ABSORPTIONS`, `KERNEL_FAMILIES`, `KERNEL_CONTINUATIONS`, `RESPONSE_ESTIMATORS`, `PHANTOM_DESIGNS`, `TABLE_STATISTICS`, `FILL_POLICIES`, `END_CUTS`, first entry = deliverable default
- Library calls relied on: dataclasses, json, hashlib
- Custom code: none
- Test oracles (`tests/smoke/test_smoke_config.py`): the defaults equal the decided values (D-018, D-019, D-023, D-024, D5, D7; exact equality); every field line of the source carries `# source:` (parsed from the file; count = number of fields); `sigma_fit_study_um` is not in the estimator signature and `sigma_fit_um` is; a renderer change leaves the estimator hash unchanged, a `sigma_fit` change changes it; JSON round trip is identity; frozen; ASCII; `validate()` refuses the illegal values listed in the test; `with_sigma_fit` refuses a value outside the study set
- Data flow: feeds every block
- Confirmed: yes 2026-10-06 (values from the D-021 batch, D-022 to D-024; provisional ones tagged PROVISIONAL)
- Status: smoke-tested 2026-10-06 (6 pass, 1 skip)

### Block 2: geometry (S1)-(S6)

- Module: `src/allen_diameter/model/geometry.py`
- Public API: `Tube(c, r, phi, theta, aspect=1.0, half_length=None, end_cut="axial")` (frozen; properties `d`, `phi0`, `e_u`, `e_v`, `t_hat`, `t_hat0`, `half_chord_factor`); `axis_direction(phi, theta) -> (3,)`; `to_local(x, y, z, tube) -> (u, v, w)`; `inside_tube(x, y, z, tube) -> bool array`; `column_interval(x, y, tube) -> (z_lo, z_hi, inside)`; `depth_extent(tube) -> (z_min, z_max)`; `lateral_half_extent(tube) -> (along e_u, along e_v)`; `slab_centres(z_min, z_max, dzeta) -> zeta (J,)`; `slab_absorbance(z_lo, z_hi, zeta, dzeta, mu) -> a_j`; `transmitted_before(z_lo, z_hi, zeta, dzeta, mu, light_direction) -> T_<j`; `line_interval(P, s, tube) -> (t1, t2, hit)`
- Inputs: positions in um (x, y specimen-referred; z stage units), broadcasting arrays; `r` um > 0; `phi` rad in [0, pi/2) (observed tilt); `theta` rad; `aspect` k > 0; `half_length` U um > 0 or None; `mu` 1/um >= 0; `light_direction` +1 (light toward +z) or -1
- Outputs: float64 arrays of the broadcast input shape (absorbances dimensionless, lengths um); `z_lo = z_hi = c_z` and `inside = False` where a vertical line misses (never NaN); `line_interval` returns line parameters, arc length = (t2 - t1)|s|
- Generalisations of impl-handoff (S1)-(S6), both reducing to the handoff at k = 1:
  - squashed cross-section: the phantom is a round tube of radius r in unsquashed space mapped by S = diag(1, 1, k) about c (mounting shrinkage along global z, procedure s.3.5); unsquashed tilt phi0 = arctan(tan(phi)/k); the vertical-line half-chord is h(v) = sqrt(r^2 - v^2) sqrt(k^2 + tan^2 phi); the width along e_v stays d for every phi and k (handoff Eq. 7);
  - end cut `RendererConfig.end_cut`: `"vertical"` keeps |u| <= U (impl-handoff (S5)); `"axial"` keeps |s0| <= U with s0 = u cos(phi0) + (w/k) sin(phi0) the unsquashed arc length along the axis, so the depth span stays bounded for every phi < 90 deg. Depth extent: vertical c_z -+ (U tan phi + r sqrt(k^2 + tan^2 phi)); axial c_z -+ k (U sin phi0 + r cos phi0).
- Parameters: `RendererConfig.U_um`, `end_cut` (default `"axial"`, PROVISIONAL, to be confirmed), `cross_section_aspect`, `dzeta_um`; `AcquisitionConfig.light_direction`
- Library calls relied on: numpy
- Custom code: all (the closed forms of the spec; no library computes tube chords)
- Test oracles (`tests/smoke/test_smoke_geometry.py`): width along e_v = d to 1e-12 for random phi <= 85 deg, k in [0.4, 1.6], all cuts; handoff Eq. 8 oblique cut to 1e-10 relative and its 1.265 d example; vertical line through c = 2 r sqrt(k^2 + tan^2 phi) (or the cap-limited length) to 1e-10; a line along the axis with axial caps has length 2U/|S^-1 t_hat| to 1e-9; column intervals vs an independent (S2) membership and vs handoff Eq. 6: 0 mismatches away from 1e-9 of the ends; slab chords vs 20001-sample brute force to 2 sampling steps; line chords vs 2e-4 um sampling to 4 steps; identical to `checks/stack_geometry_check.py` (`ray_interval`, `inside_3d`) to 1e-12 at k = 1; sum_j a_j = mu (z_hi - z_lo) to 1e-12; T_<j = cumulative product of exp(-a) for both light directions to 1e-12; sum_j T_<j (1 - e^{-a_j}) = 1 - e^{-sum a_j} (C1) to 1e-12; the depth extent bounds every column and is attained at its extreme point (to 1e-9 (1 + tan phi)); rotation by pi with theta -> theta + pi is a symmetry (S3); tangent and on-surface lines miss; invalid tubes raise ValueError
- Data flow: feeds Block 4 (renderer) and Block 9 (ray world)
- Confirmed: yes (S1)-(S6) as written in the impl-handoff; the squash and the axial cut are the assistant's generalisations, the axial default PROVISIONAL (2026-10-06)
- Status: smoke-tested 2026-10-06 (5 pass, 2 skip)

### Block 3: forward model and fitter (handoff Eqs. 10-11; D-018.2; D-019.1)

- Module: `src/allen_diameter/model/tube_model.py` (forward model), `src/allen_diameter/analysis/fit.py` (fitter)
- Forward model: for each fixed $d > 0$ (um), $\alpha \ge 0$, $v_0$ (um), $\sigma > 0$ (um), $\bar B > 0$ (grey levels) and every $v$ (um), handoff Eq. 11 with $B$ fixed (D-018.2) is $I_{\rm model}(v) = \bar B\,(T_{d,\alpha,v_0} * g_\sigma)(v)$ with $T_{d,\alpha,v_0}(v) = e^{-\alpha s_d(v - v_0)}$. Since $1 - T$ vanishes for $|v - v_0| > d/2$ and $g_\sigma$ has unit integral, the substitution $u = (d/2)\sin\psi$ gives

  $$I_{\rm model}(v \mid d, \alpha, v_0, \sigma, \bar B) = \bar B\Big[1 - \frac{d}{2}\int_{-\pi/2}^{\pi/2}\big(1 - e^{-\alpha\cos\psi}\big)\cos\psi\; g_\sigma\big(v - v_0 - \tfrac{d}{2}\sin\psi\big)\,d\psi\Big] \tag{B3.1}$$

  whose integrand is analytic in $\psi$ (no endpoint square root), evaluated by $N$-node Gauss-Legendre quadrature in $\psi$ (`numpy.polynomial.legendre.leggauss`, nodes cached per $N$). In a fit, $N = \max(\texttt{fit\_quad\_min\_nodes}, \lceil \texttt{fit\_quad\_nodes\_per\_sigma}\cdot d_{\rm hi}/\sigma_{\rm fit}\rceil)$ with $d_{\rm hi}$ = `fit_d_bounds_um[1]`, fixed for the whole fit so the model is smooth in the parameters. The $\mu$ form (D-019.1): $\alpha = \mu d/\cos\varphi_i$ for $\varphi_i \in [0, \pi/2)$.
- Exact dip area (oracle, for every $\alpha \ge 0$): $\int_{\mathbb R}\big(\bar B - I_{\rm model}(v)\big)\,dv = \bar B\,\frac{\pi d}{2}\big(I_1(\alpha) - \mathbf L_1(\alpha)\big)$ ($I_1$ modified Bessel, $\mathbf L_1$ modified Struve), with limits $\bar B\mu\pi d^2/(4\cos\varphi_i)$ as $\alpha \to 0$ and $\bar B d$ as $\alpha \to \infty$
- Fit (deliverable, D-019.1): parameters $(d, \mu, v_0)$ in the box `fit_d_bounds_um` x `fit_mu_bounds_per_um` x `fit_v0_bounds_um`; residuals $r_n = (I_{\rm model}(v_n) - I_i(v_n))/\bar B_i$ (same minimiser as (3), $\bar B_i > 0$ a constant); `scipy.optimize.least_squares(method="trf", jac="2-point", x_scale="jac", ftol = xtol = gtol = fit_tol, max_nfev = fit_max_nfev)`, one run per start; the reported run is the converged run (scipy status > 0, finite) of least residual sum of squares, the first in start order on ties
- Starts (`fit_start_rule`, PROVISIONAL): $n^*$ = index of the smallest $I_n$ among samples with $v_n$ inside the $v_0$ bounds; $v_{0,{\rm start}} = v_{n^*}$. `"profile"`: $w$ = width of the dip at half depth, level $(\bar B + I_{n^*})/2$, crossings found walking out from $n^*$ and placed by linear interpolation between samples; $d_0 = \sqrt{\max(w^2 - 8\ln 2\,\sigma_{\rm fit}^2, 0)}$ (the blur's FWHM removed in quadrature), or `fit_d0_um` when $I_{n^*} \ge \bar B$ or a crossing is missing on either side. `"fixed"`: $d_0$ = `fit_d0_um`. Start $k$: $d = d_0 f_k$ ($f_k$ = `fit_multistart_factors`) and $\mu = \alpha_0\cos\varphi_i/d$ with $\alpha_0 = -\ln\min(1, \max(I_{n^*}/\bar B, \text{smallest positive float}))$ (the start's centre-line absorbance equals the observed depth), each clipped into its bounds
- Status: `"failed"` when no start converged (the finite run of least residual sum of squares is then reported); `"at_bound"` when a reported parameter lies within `fit_at_bound_rel_tol` x (its bound range) of a bound, the bounds named in `at_bound` (`d_lo`, `d_hi`, `mu_lo`, `mu_hi`, `v0_lo`, `v0_hi`; `alpha_lo`, `alpha_hi` in the alpha form); otherwise `"converged"`. For a dark tube (alpha above about 5) the profile hardly depends on $\mu$ and the fit may stop at `mu_hi` with a sound $\hat d$ (seen on the `mu_tie.py` case $\mu = 3$, $d = 2$ um, $\hat d/d = 0.977$); such nodes are excluded by $\mathcal S$ twice over (bound, dark flag)
- alpha form (D-018.2; labelled comparison only, never the deliverable): the same fit over $(d, \alpha, v_0)$ with $\alpha$ in the box $[\mu_{\rm lo} d_{\rm lo}/\cos\varphi_i,\ \mu_{\rm hi} d_{\rm hi}/\cos\varphi_i]$ that contains the image of the $\mu$ box; same starts ($\alpha = \alpha_0$); $\hat\mu$ reported as $\hat\alpha\cos\varphi_i/\hat d$. The two forms have the same minimiser in $(d, v_0)$ whenever no bound is active (bijection, D-019)
- Public API: `tube_model.transmittance(v, d, alpha, v0) -> T`; `tube_model.model_profile(v, d, alpha, v0, sigma, B_bar, n_nodes) -> I_model`; `tube_model.alpha_from_mu(mu, d, phi) -> alpha`; `tube_model.quadrature_rule(n_nodes) -> (sin_psi, cos_psi, weights)`; `tube_model.n_quadrature_nodes(d_hi, sigma, min_nodes, nodes_per_sigma) -> int`; `fit.half_depth_width(v, I, B_bar, n_star) -> float or None`; `fit.start_points(v, I, phi, B_bar, cfg, form) -> (n_starts, 3) array`; `fit.fit_profile(v, I, phi, B_bar, cfg) -> FitResult`; `fit.fit_profile_alpha(v, I, phi, B_bar, cfg) -> FitResult`
- `FitResult` (frozen): `d_hat_um, mu_hat_per_um, v0_hat_um, alpha_hat` (= $\hat\mu\hat d/\cos\varphi_i$), `phi_rad`, `form` (`"mu"` or `"alpha"`), `status`, `at_bound` (tuple of bound names), `rss_gl2` ($\sum_n (I_n - I_{\rm model}(v_n))^2$, grey levels squared), `n_samples`, `n_starts`, `best_start`, `n_converged`, `nfev` (summed over starts), `n_nodes`, `message`
- Inputs: `v` (n,) strictly increasing, um; `I` (n,) finite, grey levels; `phi` rad in [0, pi/2); `B_bar` > 0 finite, grey levels; `cfg` a `MeasureConfig` ($\sigma_{\rm fit}$ = `sigma_fit_um`); n >= 4; at least one sample inside the $v_0$ bounds. Anything else raises ValueError (Block 5 flags a profile that leaves the image before calling the fit). `mu_mode` other than `per_node` raises NotImplementedError (a shared $\mu$ is a planned labelled comparison, D-023)
- Parameters: `sigma_fit_um`, `fit_d_bounds_um`, `fit_mu_bounds_per_um`, `fit_v0_bounds_um`, `fit_multistart_factors`, `fit_start_rule`, `fit_d0_um`, `fit_tol`, `fit_max_nfev`, `fit_at_bound_rel_tol`, `fit_quad_min_nodes`, `fit_quad_nodes_per_sigma`
- Library calls relied on: `numpy.polynomial.legendre.leggauss`, `scipy.optimize.least_squares`; in the test `scipy.integrate.quad`, `scipy.special.iv`, `scipy.special.modstruve`
- Custom code: the substitution (B3.1) (no library convolves a Beer-Lambert dome with a Gaussian) and the start rule
- Test oracles (`tests/smoke/test_smoke_fit.py`): (B3.1) against `scipy.integrate.quad` of the unsubstituted integral at $d \in \{0.05, 0.5, 2, 6\}$ um, $\alpha \in \{0.3, 3, 30\}$, $\sigma \in \{0.08, 0.125\}$: $|\Delta| \le 10^{-12}\bar B$ at the fit's $N$; the exact dip area to $10^{-9}$ relative at $\alpha \in \{10^{-3}, 0.3, 3, 10\}$; faint-limit centroid $v_0$ and variance $\sigma^2 + d^2/16$ (handoff Eq. 12) to $10^{-6}$ relative at $\alpha = 10^{-6}$; quadrature error decreasing in $N$ and $\le 10^{-12}\bar B$ at the configured $N$ for the worst case $d = d_{\rm hi}$, $\sigma = \min$ `sigma_fit_study_um`; `alpha_from_mu` equals $\mu d/\cos\varphi$ to 1 ulp and is the only route from $\mu$ to the model; agreement with `checks/mu_tie.py`'s grid model within $10^{-3}$ (its grid and 4-sigma truncation); noise-free recovery of $(d, \mu, v_0)$ to $10^{-6}$ relative at $\alpha \in \{0.3, 3\}$, $d \in \{0.5, 1, 2\}$ um, $\varphi \in \{0°, 30°, 60°\}$ (handoff "Numbers checked" asks $10^{-3}$); on the `mu_tie.py` profiles ($\sigma_{\rm true} = 0.10$, $\sigma_{\rm fit} = 0.125$, $\varphi = 15°$, $\mu \in \{0.6, 3\}$, $d \in \{0.3, 0.5, 1, 2\}$) the $\mu$ and $\alpha$ forms return the same $\hat d$ to $2\cdot10^{-3} d$ wherever neither is at a bound, and both report `at_bound` where one is; the same runs reproduce the logged D-019 (a) values $\hat d/d$ = 0.88, 0.98, 1.00 ($\mu = 0.6$) and 0.89, 0.98, 0.98 ($\mu = 3$) at $d$ = 0.5, 1, 2 um to $\pm 0.006$ (two-decimal rounding plus the grid model's error); the half-depth width of a sampled triangular dip is exact; a flat profile returns `at_bound` with `mu_lo`, an 8 um dip `d_hi`; determinism (bit-identical repeat); invalid inputs raise ValueError
- Data flow: consumes Block 5's profile, $\varphi_i$ and $\bar B_i$; feeds Blocks 5 (centres, handoff Eq. 3), 6, 10; `model_profile` is the C2 oracle of Block 4
- Confirmed: yes for the model and the fit (D-018, D-019, D-023); the quadrature, the start rule and the tolerances are the assistant's, PROVISIONAL (2026-10-06)
- Status: smoke-tested 2026-10-06 (7 pass): model vs quad 4e-14 B over 216 points; vs `mu_tie.py` 7e-4 (its grid); quadrature error 4e-15 B at N = 450 (d = 6 um, sigma = 0.08 um), 4e-4 B at N/4; noise-free recovery 1e-14; D-019 (a) reproduced (0.879, 0.982, 0.996; 0.886, 0.975, 0.977)

### Block 4: renderer (procedure Eqs. 5-6; impl-handoff (S4), FFT form; procedure s.3.6 step 6)

- Module: `src/allen_diameter/model/kernel.py`, `src/allen_diameter/model/render.py`, `src/allen_diameter/model/camera.py`, `src/allen_diameter/loading/jpeg_tables.py`
- Kernel (`kernel_family = "gaussian_table"`): for every defocus $\delta$ (um), $\sigma_{\rm r}(\delta)$ = linear interpolation of (`kernel_table_delta_um`, `kernel_table_sigma_um`) at $|\delta| \le \delta_{\max}$ (the last knot); beyond, $\sigma_{\rm r}(\delta_{\max}) + \gamma(|\delta| - \delta_{\max})$ with $\gamma$ = `kernel_continuation_slope` (`"linear"`), $\gamma = 0$ (`"frozen"`), $\gamma = \sigma_{\rm r}(\delta_{\max})/\delta_{\max}$ (`"proportional"`, the rule of `checks/stack_geometry_check.py`). `sigma_r0_um` is the table's value at 0 (one number under two names; `validate()` keeps them equal). `"empirical"` raises NotImplementedError until Block 10
- Fine grid: a `FineGrid(x0, y0, h, nx, ny)` of sample centres $x_n = x_0 + n h$, $y_m = y_0 + m h$ (um); the object is drawn on every sample of the grid and nowhere else (the caller chooses the grid: Block 6 sizes it from the block, `pad_um` and the tube's footprint). For a block of pixels (`left`, `top`, `width`, `height`, full-resolution pixel coordinates whose integers are pixel centres; um = coordinate x $p_{\rm x}$), `block_fine_grid` subdivides every pixel into $f \times f$ samples, $f = p_{\rm x}/h_{\rm g}$ an integer ($h_{\rm g}$ = `h_g_um_thin` for $d \le$ `h_g_switch_d_um`, else `h_g_um_thick`), and adds `pad_px` whole pixels on each side
- Slabs: centres $\zeta_j$ from Block 2 `slab_centres` over $[\min z_{\rm lo}, \max z_{\rm hi}]$ of the tube's columns on the grid, thickness `dzeta_um`. Per slab, on the grid: $a_j$ and $T_{<j}$ by Block 2 (`slab_absorbance`, `transmitted_before` with `AcquisitionConfig.light_direction`); the absorbed fraction is $\Delta A_j = T_{<j}(1 - e^{-a_j})$ for `absorption = "partition_vertical"` (procedure Eq. 5) and $\Delta A_j = a_j$ for `"linear"` (mathematics Eq. 11); `"ray_world"` raises NotImplementedError until Block 9
- Planes: transmittance $\tau_k = I_k/B = 1 - \sum_j (\Delta A_j * G_{\sigma_{\rm r}(\zeta_j - z_k)})$ on the grid (procedure Eq. 6 divided by $B$), at plane depths $z_k = k\,\Delta z$ (plane index $k$; $z$ in stage units, increasing in the light's direction when `light_direction = +1`). Backend `"fft"` (deliverable), split by kernel width: (slab, plane) pairs with $\sigma_{\rm r} \le$ `fft_split_sigma_um` go through one `scipy.fft.rfft2` per slab, zero-padded on each axis by $p \ge$ `fft_wrap_sigmas` $\cdot\,\sigma/h$ samples ($\sigma$ the widest such kernel), the Gaussian transfer $e^{-2\pi^2\sigma^2(f_x^2 + f_y^2)}$ as an outer product per pair, one `irfft2` per plane (a source's periodic images stay at least $p\,h$ from every output sample: wrap error below $e^{-p^2h^2/(2\sigma^2)}$ of a slab's peak contribution); wider pairs (far defocus: steep, long or thick tubes) are summed exactly from the sampled object against the continuous Gaussian onto a coarse output grid of spacing $\le \sigma_{\min,\rm far}/$`far_grid_per_sigma`, then bicubic-spline interpolated (`scipy.interpolate.RectBivariateSpline`) to the fine grid. Without the split the padding grows with the widest kernel (9 um and a 4840-sample FFT for a 60-degree tube with $U$ = 10 um). The grid must resolve the narrowest kernel, $h \le \sigma_{\min}/2$ (ValueError otherwise; spectral truncation $< 3\cdot10^{-9}$). Backend `"direct"` (reference): `scipy.ndimage.gaussian_filter(sigma = sigma_r / h, mode="constant", cval=0, truncate=direct_truncate)` per (slab, plane), as procedure s.3.6 step 4 writes it
- Camera chain (procedure s.3.6 step 6, in this order): pixel integration (mean of the $f \times f$ samples of each pixel), grey mapping $s = $ `black_level_gl` $+$ `gain` $\cdot$ `background_B_gl` $\cdot\,\tau_{\rm px}$, additive Gaussian noise of SD `noise_sd_gl` from the passed `numpy.random.Generator`, rounding to the nearest integer and clipping to $[0, 2^{\rm bit\_depth} - 1]$, JPEG round trip (Pillow, `quality = jpeg_quality`, or Allen's quantization tables when given) when `jpeg` is true. No bilinear interpolation (Block 5 applies it when reading profiles)
- Public API: `kernel.sigma_r(delta, rcfg) -> ndarray`; `render.FineGrid`; `render.block_fine_grid(left, top, width, height, p_x, factor, pad_px) -> (FineGrid, (row slice, col slice))`; `render.fine_factor(d, rcfg, p_x) -> int`; `render.slab_grid(tube, grid, dzeta) -> zeta (J,)`; `render.absorbed_fractions(tube, mu, grid, zeta, dzeta, light_direction, absorption) -> (J, ny, nx)` (tests and small cases); `render.render_transmittance(tube, mu, z_planes, grid, rcfg, light_direction, backend=None) -> RenderResult(tau (n_planes, ny, nx), zeta, sigma_max_um, fft_shape, backend)`; `render.pixel_integrate(fine, factor) -> array`; `render.synthetic_block(tube, mu, ks, left, top, width, height, cfg, rng, pad_px, qtables=None, backend=None) -> (block uint8 (n_planes, height, width), ks, valid, frame)` in the `fetch_zblock` contract (Section 3; `valid` all true; `frame = CropFrame(left, top, 0, p_x)`); `camera.camera_chain(tau_px, rcfg, rng, qtables=None) -> uint8`; `camera.jpeg_roundtrip(img8, quality=None, qtables=None) -> uint8`; `jpeg_tables.load_qtables(path) -> list of 64-integer lists`
- Inputs: a Block 2 `Tube` (um, rad), `mu` (1/um, >= 0), plane depths (um), a `FineGrid`, `RendererConfig`, `light_direction` (+1 or -1), a `numpy.random.Generator` for the camera noise
- Parameters: all of `RendererConfig`; `AcquisitionConfig.res0_um`, `dz_um`, `light_direction`
- Library calls relied on: `scipy.fft` (`rfft2`, `irfft2`, `rfftfreq`, `fftfreq`, `next_fast_len`), `scipy.interpolate.RectBivariateSpline` (far path), `scipy.ndimage.gaussian_filter` (direct backend), Pillow (`Image.save(format="JPEG", quality | qtables)`, `Image.open(f).quantization`)
- Custom code: the slab loop with the partition (Eqs. 5-6), the per-(slab, plane) transfer functions and the far-field separable sum (no library renders a depth-dependent blur)
- Test oracles (`tests/smoke/test_smoke_render.py`): $\sigma_{\rm r}$ exact at the knots, linear between, the three continuations, even in $\delta$, equal to `stack_geometry_check.sigma_r` for `"proportional"`; C1: $\sum_j \Delta A_j = 1 - e^{-\sum_j a_j}$ per column to 1e-12; C2: one slab holding the whole tube with a constant kernel $\sigma$ and $\varphi = 0$ equals the discrete convolution of the sampled $1 - T$ with $g_\sigma$ to 1e-12, and converges to Block 3's `model_profile` as $h \to 0$ (error falling monotonically over $h = p_{\rm x}/8, /16, /32$, at most 1e-4 at $/32$; the decrease is irregular because point samples meet the tube's edges at varying sub-sample positions); the node dip moves by at most 1e-4 when $h_{\rm g}$ halves and 5e-4 when `dzeta_um` halves from 0.04 um; `fft` without the split vs `direct` to 1e-10, with the default split to 1e-6; agreement with `checks/stack_geometry_check.render_planes` (vertical cut, proportional continuation) to 1e-10; mass: $\iint (1 - \tau_k) = \sum_j \iint \Delta A_j$ for every plane to 1e-10 relative when the grid holds the blurred tube; exact symmetries: a 90-degree rotation of the tube rotates the image, and $\tau(z_k; +1, \theta) = \tau(2c_z - z_k; -1, \theta + \pi)$ (light reversed = depth mirrored); faint limit: partition and linear agree to $O(\mu d)$; $\mu = 0$ gives $\tau = 1$; camera: pixel integration is the block mean, the background of a flat field has mean `black + gain B` and SD `noise_sd_gl` within 2 % before JPEG, values are integers in range, JPEG with explicit tables round-trips the tables (`Image.open(f).quantization`); determinism under a fixed seed; invalid inputs raise
- Data flow: consumes Block 2 (geometry) and Block 3 (C2 oracle); feeds Blocks 5 (synthetic blocks), 6, 10
- Confirmed: yes for the model (procedure s.3.6, D-024); the FFT padding rule, the reference backend's truncation and the grey mapping formula are the assistant's, PROVISIONAL (2026-10-06)
- Status: smoke-tested 2026-10-06 (7 pass): fft vs direct 9e-16, with the split 5e-8; vs `stack_geometry_check.render_planes` 4e-16; C2 exact 4e-16 and vs Eq. 11 2.2e-3 / 5.8e-4 / 6.0e-5 at $p_{\rm x}$/8, /16, /32 (point sampling of the object: 0.4 % of a 0.6 dip at the production $p_{\rm x}/8$ for this flat $d = 0.8$ um tube). Timing (sandbox, one core, block of 87 pixels + 17 padding): 60-degree $d = 1$ um tube, $U = 10$ um, 21 planes: 28 s at `fft_split_sigma_um` 0.5, 48 s at 1.0; flat $d = 0.5$ um at $p_{\rm x}/16$, 7 planes: 11 s / 15 s

### Block 5: the shared per-node chain (handoff Eqs. 1-5, 9; D-018.1)

- Module: `src/allen_diameter/analysis/focus.py`, `analysis/path.py`, `analysis/background.py`, `analysis/node_pipeline.py`
- Public API (planned): `focus_scores(block, ks, valid, line, cfg) -> F_k`; `best_plane(F_k, ks, dz) -> (k_star, z_sub, flat_top)`; `line_fit(centres_um, s_path, L) -> (t_hat, theta, phi, y_hat)`; `background_median(plane, mask) -> B_bar`; `measure_node(block_provider, swc, node_id, cfg) -> NodeResult` (the one function of impl-handoff "What the whole pipeline is")
- Inputs: a block provider returning `(block, ks, valid, frame)` for a node (real: `fetch_zblock`; synthetic: Block 4); the SWC; config
- Outputs: `NodeResult` with every per-node CSV column of Block 8 except the correction ones
- Parameters: `MeasureConfig`
- Library calls relied on: `scipy.ndimage.map_coordinates` (bilinear, order=1), `numpy.linalg.eigh` (TLS line)
- Custom code: the focus score (1) and the per-node orchestration
- Test oracles: recovers known $\theta, \varphi$ of a synthetic tube to 2 deg; $\hat y_i \perp \hat t_{i,\parallel}$ for any $\theta$; D6 guard: a direction fitted in pixel/plane units inflates $\varphi$ (asserted on a tilted phantom); flags fire for empty tissue (`faint`) and at the stack edge; parabola vertex within $\pm\Delta z/2$; **gate 1**: single-depth phantoms ($\varphi = 0$, all absorbance at $\delta = 0$, $d \in [0.5, 1]$ um) give $\hat b \approx 1$ within 2 % when $\sigma_{\rm fit}^2 \approx \sigma_{\rm r}(0)^2 + p_{\rm x}^2/4$
- Data flow: consumes Blocks 0, 3, 4; feeds Blocks 6, 8, 10, 11
- Confirmed: yes (handoff Method; D-023; D-024 tilt policy)
- Status: drafted

### Block 6: phantoms and the bias table (procedure s.3.5, 3.8; D-024)

- Module: `src/allen_diameter/analysis/phantoms.py`, `analysis/table.py`, `scripts/build_table.py`
- Public API (planned): `draw_replicates(cfg, rng, n) -> list of Phantom + nuisances`; `run_replicate(phantom, cfg, rng) -> ReplicateRow`; `fit_table(rows, cfg) -> BiasTable(m_hat spline, tau_hat, failure_rate, C)`; `BiasTable.save(path)`, `load(path)`; `BiasTable.b_hat(d, phi)`, `m_hat(d, phi)`, `tau_hat(d, phi)`, `failure(d, phi)`
- Inputs: config; rng; replicate rows
- Outputs: `npz` + JSON (Section 3)
- Parameters: `PhantomConfig`, `CorrectionConfig.response_estimator`, `spline_*`, `tau_kernel_bandwidth`, `table_statistic`
- Library calls relied on: `scipy.interpolate.RBFInterpolator`, numpy; `multiprocessing` for the sandbox, a PBS array on davinci (D-022)
- Custom code: the local kernel estimates of $\hat\tau$ and the failure rate (no library gives them under one weighting)
- Test oracles: seed determinism (bit-identical rows); nuisance laws by Kolmogorov-Smirnov on 2000 draws; on a fixture with known $m$ (synthetic $\hat d_n = m(d, \varphi) + \tau\epsilon_n$) the spline recovers $m$ within 2 SE and $\hat\tau$ within 15 %; `grid` design as a labelled comparison agrees with `random` within SE; the table refuses a mismatched $\mathcal C$; file round trip
- Data flow: consumes Blocks 4, 5; feeds Block 7, 11
- Confirmed: yes (D-024)
- Status: drafted

### Block 7: inversion, flags, fill (procedure Eq. 1, s.3.9; mathematics Eqs. 20-21; D5)

- Module: `src/allen_diameter/analysis/invert.py`, `analysis/fill.py`
- Public API (planned): `invert(d_hat, phi, table, cfg) -> (d_tilde, b_hat_at_solution, flags)`; `correct_nodes(node_results, table, cfg) -> rows with d_tilde, flags`; `fill_and_smooth(rows, swc, cfg) -> d_final, filled_from`
- Inputs: `NodeResult` rows; a `BiasTable`; the SWC (branch structure)
- Outputs: the correction columns of Block 8
- Parameters: `CorrectionConfig`
- Library calls relied on: `scipy.optimize.brentq`, `scipy.ndimage.median_filter` per branch
- Custom code: none beyond orchestration
- Test oracles: on a synthetic table with $b(d) = 1 + c_0/d$ the root solution differs from the shortcut by $-\beta(b - 1)$ to first order (mathematics Eq. 21); non-monotone and out-of-domain cases flagged; fill rules on a hand-made branch; the mismatched-$\mathcal C$ refusal; **gate 2**: synthetic stacks from independent seeds recover $d$ within 10 % for $d \in \{0.5, 1, 2, 3\}$ um, $\varphi \le 20°$
- Data flow: consumes Blocks 5, 6; feeds Block 8
- Confirmed: yes (D5 confirmed by D-024)
- Status: drafted

### Block 8: cell level -- SWC I/O, per-node CSV, membrane area (handoff step 7; D-013)

- Module: `src/allen_diameter/loading/swc_io.py` (written), `src/allen_diameter/analysis/cell.py` (planned), `scripts/run_cell.py` (planned), `scripts/allen_radius_distribution.py` (written)
- Public API: `swc_io.read_swc(path) -> SWC`; `swc_io.write_swc(swc, path, radius=None)`; `swc_io.rewrite_radius_lines(swc, radius) -> lines`; `swc_io.segment_lengths_um(swc)`; `swc_io.frustum_areas_um2(swc, radius=None)`; planned: `cell.area_ratio(swc, d_final) -> float`, `cell.write_outputs(...)`
- Inputs: SWC path; radii (um, > 0, shape (N,))
- Outputs: `SWC` (Section 3); written file byte-identical except the radius tokens, which keep the token's decimals (>= 4)
- Per-node CSV columns: `node_id, type, x_um, y_um, z_um, path_um, reg_verdict, s_star_um, dz_star_um, k_star, z_sub_um, cx_um, cy_um, cz_um, theta_rad, phi_rad, steep, vertical, B_bar, B_bar_region, d_hat_um, mu_hat_per_um, v0_hat_um, alpha_hat, fit_status, b_hat, d_tilde_um, flags, filled_from, d_final_um, allen_radius_um, sigma_fit_um, d_tilde_sigma_spread_um`
- Parameters: `AcquisitionConfig.dendrite_swc_types`; `MeasureConfig.sigma_fit_study_um` (the spread column, D-023)
- Library calls relied on: numpy; the Allen API through `allen_image_io.fetch_swc` (Colab only)
- Custom code: the separator-preserving tokeniser (`_split_keep`) -- `str.split` discards separators
- Test oracles (`tests/smoke/test_smoke_swc_io.py`): hand-written file parses to known arrays; frustum areas of a cylinder ($2\pi r h$) and a cone exact to 1e-12; agreement with `allen_image_plot.read_swc`; write without new radii is byte-identical; with new radii only the 6th token of data lines changes (separators, CRLF, comments preserved); malformed lines and bad radii refused. `scripts/allen_radius_distribution.py`: pooled percentiles of the per-node CSV equal `numpy.percentile` on the same values (checked by running it on a synthetic archive)
- Data flow: consumes Block 7 (planned part); feeds the I_h fitter (`--swc-dir`, D-013) and Block 11
- Confirmed: yes (D-013; D-024 (iii) for the radius script)
- Status: loading half smoke-tested 2026-10-06 (6 pass, 1 skip); `cell.py` drafted

### Block 9: ray-world generator and the independent end-to-end check (impl-handoff (S6)-(S8))

- Module: `src/allen_diameter/model/ray_world.py`, `scripts/end_to_end.py`
- Public API (planned): `directions(n_rho, n_psi, na, n_oil) -> (S, W)`; `chord_interval(P, S, r, phi, U)`; `ray_plane(phantom, z_k, grid, S, W) -> I_k/B`; selectable as `absorption = "ray_world"`
- Oracles: chord vs brute force on sampled lines (2e-4 um); $\langle 1/\cos\vartheta\rangle = 1.447$ by quadrature (S10); faint limit: dip area ratio ray/partition $= \langle 1/\cos\vartheta\rangle$ within 1 %; reproduces `checks/optics_points_check.py` self-checks
- Data flow: consumes Block 2; feeds the end-to-end gate of Block 7 (independent generator)
- Confirmed: yes (D-024: alternative generator for checks only)
- Status: drafted

### Block 10: kernel calibration (procedure Eq. 4, s.3.4; mathematics s.3.5)

- Module: `src/allen_diameter/analysis/calibration.py`, `scripts/calibrate_kernel.py`
- Public API (planned): `plane_scan(block_provider, node, cfg) -> omega_k, A_k`; `fit_growth(scans, cfg) -> Delta_sigma2 per offset, c_i, z_ax_i`; `kernel_from_growth(...) -> RendererConfig with a new table`
- Oracles: synthetic plane scans with a known growth recovered up to the common shift (Phase I); residual check beyond +-2 planes; area constancy diagnostic
- Status: drafted

### Block 11: real-data runs (Phase II, Colab)

- Module: `scripts/colab_bootstrap.py`, `scripts/run_node.py`, `scripts/run_cell.py`, `scripts/compare_profiles.py`
- Steps: cell 13 (registration) on node 4505 and 5-10 stretches; camera-chain calibration from fetched crops (JPEG tables, background SD, grey mapping); real $\hat\mu_i$ distribution -> phantom $\mu$; the radius distribution (`allen_radius_distribution.py --fetch 529878215`) -> `d_range_um`; the production table on davinci; apply to 529878215
- Status: drafted; not executable in the sandbox (Section 7)

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python tests/smoke/test_smoke_config.py` | Block 1 checks | none | 1 s |
| `python tests/smoke/test_smoke_swc_io.py` | Block 8 loading checks | none | 1 s |
| `python tests/smoke/test_smoke_geometry.py` | Block 2 checks | none | 10 s |
| `python tests/smoke/test_smoke_fit.py` | Block 3 checks | none | 3 s |
| `python tests/smoke/test_smoke_render.py` | Block 4 checks | none | 15 s |
| `cd tests/smoke && PYTHONPATH=../../src python smoke_allen_image.py` | Block 0 (2026-09-23 modules) | none | 30 s |
| `cd tests/smoke && PYTHONPATH=../../src python robustness_registration.py` | Block 0 robustness | none | 2-3 min |
| `python scripts/allen_radius_distribution.py --swc FILE` | Allen radius distribution (D-024) | any SWC file | 1 s |
| planned: `scripts/build_table.py --n 50 --sandbox` | reduced table | `n_replicates=50` | minutes |
| planned: `scripts/end_to_end.py --generator ray_world` | gate 2 | small $d$ set | minutes |

All commands run from `Passive Features/Diameter Re-measurement/`.

## 7. Not executable in the sandbox

- Anything that reaches `api.brain-map.org` (403 from the sandbox): real blocks, `fetch_swc`, cell 13, Phase II. Substitute: `allen_image_io.SyntheticFetcher` and the renderer; a synthetic archive of SWC files for `allen_radius_distribution.py`.
- The production bias table (davinci, D-022). Substitute: `n_replicates` of order 50-200 on a coarse range in the sandbox.
- The full `robustness_registration.py` takes > 2 min; it was run once here (40/40).

## 8. Open questions

- $d_{\max}$ of the phantom range: after `allen_radius_distribution.py` on 529878215 (Colab) and on the davinci archive (raised 2026-10-06).
- Phantom $\mu$ range and whether $\mu$ becomes a regression axis: after the real $\hat\mu_i$ distribution (2026-10-06).
- Dark-flag threshold on real data; D5's 0.2 once the share of the $(d, \varphi)$ domain it flags is known (2026-10-06).
- Jitter amplitudes after cell 13 (2026-10-06).
- Allen camera chain (black level, gain, noise after JPEG, JPEG tables) and the light direction relative to the plane index: facts nobody has (impl-handoff, Known gaps).
- Whether a root-level `tests/smoke/` redirect is needed for the scheduled tester (D-022).
- `RendererConfig.end_cut` default `"axial"` (Block 2): the handoff's vertical cut makes the depth span $2U\tan\varphi$ diverge as $\varphi \to 90°$ (raised 2026-10-06).
- The fit's start rule `fit_start_rule = "profile"` and its fallback `fit_d0_um` (Block 3; raised 2026-10-06).
