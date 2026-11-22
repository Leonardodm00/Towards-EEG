# The soma fit's recovery on simulated stacks (2026-11-22; D-216.2)

The check asked for before the pilot's stack is fitted (D-216.2): Block 46's fit (`analysis.soma_fit.fit_soma`) with
Block 45's forward model (`model.soma_render.SomaRenderer`) at the production settings, run on simulated stacks like the
pilot's, in which the soma is known. It follows `../soma_fit_2026-11-22/`. That check gave the precision (Fisher
information) and linearised biases; this one runs the full non-linear fit, from the pipeline's start, on rendered, noisy,
quantised stacks.

## Question

Does `fit_soma`, with `SomaRenderer` as its forward model at the production settings, recover a known soma? The
settings are the `SomaFitConfig` defaults: `fine_factor` 1, `dzeta_um` 0.14, `far_slope` "fit". The soma's height c
comes first, then a, b, the membrane area S and the equivalent radius r_eq = sqrt(S / 4 pi). How large are the errors
from four sources?

- (i) The fit's own discretisation (the truth rendered finer than the fit).
- (ii) A far slope that differs from the configured one.
- (iii) A soma that is not an ellipsoid.
- (iv) Stems leaving the soma.

## Settings

- **Acquisition.** The pilot's: pixel 0.1144 um; planes 0.28 um apart (plane k at depth k dz, stage units); light
  toward +z.
- **Kernel.** The pilot's cleaning configuration:
  - `kernel_table_delta_um` (0, 0.28, 0.56, 0.84);
  - `kernel_table_sigma_um` (0.08, 0.15360, 0.39243, 0.56202);
  - `sigma_r0_um` 0.08;
  - the linear continuation at 0.79 um/um, or 0.95 in case B's truth;
  - `fft_wrap_sigmas` 6.

  The configuration passes `DiameterConfig`'s validation.
- **Soma.** Centre (503.317, 287.641, 32.76) um. Its depth is plane 117's; x and y were chosen off the pixel grid.
  a = 6.5 um along phi = 0.6 rad, b = 5.0 um, mu = 0.15 /um.
- **Truth.** Rendered finer than the fit: fine_factor 2, dzeta 0.07 um. Cases A and B use `SomaRenderer`; cases C and D
  use the check renderer (below). The planes are rendered in groups, each with its own FFT box, to bound the memory;
  grouping changes tau by at most 2.2e-15.
- **Stack.** Each plane is tau_k times B_k = 110 (1 + 0.05 sin(1.5 pi t_k)) gl, with t_k = (k - 88) / 70 across the
  planes, so B_k runs from 104.5 to 115.5 gl. Gaussian noise of SD 3 gl is added, drawn by `numpy.random.default_rng`
  with seed 20261122. Every case uses the same noise field (common random numbers), so the cases differ only in their
  truth. The result is rounded to integers and clipped to [0, 255]. Nothing was clipped: the data span 25-129 gl.
- **Start.** As the pipeline takes it from Allen's trace:
  - the traced centre is the truth's plus (0.8, -0.6, +1.5) um, i.e. (504.117, 287.041, 34.26) um;
  - the traced radius is 0.9 times the truth's r_eq: 4.320, 4.785, 5.245 um for A with c = 3.0, 4.5, 6.0 and for B,
    and 5.032 um for C and D (0.9 times the union's r_eq).
- **Crop and planes.**
  - The crop follows Block 46's data flow: the fit region's bounding box (13 um about the traced centre) with a margin of
    1 um. That gives (left, top, width, height) = (4284, 2386, 246, 247) px.
  - Block 44's window holds the planes with |k dz - z_s| <= r_s + 6 um. Of these, `select_planes` takes 36: k = 88 to
    158 in steps of 2 (k_0 = 122). Only those 36 planes are rendered; the fit uses no other.
- **Fit.**
  - `fit_soma` at the `SomaFitConfig` defaults, with `far_slope` "fixed" in B_fixed.
  - Forward model: `SomaRenderer(crop, p, z, rcfg, scfg, +1).render` at fine_factor 1, dzeta 0.14 um.
  - slope0 = 0.79.
  - No mask in A to C. In D, the stems' footprints widened by 2 um (below).
- **Case C, the pyramidal soma.** The ellipsoid of A (c = 4.5 um) united with a right circular cone.
  - The cone's axis lies in the image plane at depth z0, along phi + pi/2 (the b axis). Its apex is 11 um from the
    centre.
  - Its base is the disc of radius c in the ellipsoid's central section perpendicular to the axis, the widest section.
    That fixes tan(alpha) = c / 11 = 0.409, alpha = 22.25 deg.
  - The base touches the ellipsoid's surface at z0 +- c and lies inside it elsewhere. So the cone meets the ellipsoid at
    its widest section, and the union's depth profile has no step.
  - In the image plane the cone emerges from the ellipse 4.6 um from the centre. The union spans -5 to +11 um along the
    axis and +-6.5 um across it.
  - Every part is symmetric about z0, so each column's depth interval is the longer of the two.
  - The union's half-height is 4.5 um, the same as c: the centre column is the deepest.
  - Its area is 392.77 um2 (marching cubes at 0.05 um), 10.6 % more than the ellipsoid's 355.28 um2. Its volume is
    657.6 um3 (the ellipsoid's is 612.6).
  - D-216.2 left the cone's base and cross-section open; these are the check's choices. A round cone with a wider angle
    would stand out of the ellipsoid in depth, so its base would make a step in the depth profile.
- **Case D, case C plus stems.** All the stems lie at depth z0 and have the soma's mu: the check renderer renders one
  material, as Block 45 does.
  - An apical trunk of radius 1.0 um, from the apex outward along the cone's axis for 15 um.
  - Two basal stems of radius 0.6 um, 12 um long. Each runs radially from the centre, starting on the ellipse's outline,
    at 125 and -130 deg from the apical direction (the check's choice). Each has a 1 um attachment inside the
    ellipsoid, which the ellipsoid hides.
  - The mask is Block 44's `segment_mask` of the stems' segments at radius r + `mask_widen_um` (2 um), as the pipeline
    will build it from Allen's trace.
  - The truth is rendered on a crop holding the whole solid (316 x 367 px), so that the stems outside the fit's crop blur
    into it, and then cut to the fit's crop.
  - D's references for S and c are C's: the stems are not soma.

## Run

```
cd "Passive Features/Diameter Re-measurement/checks/soma_recovery_2026-11-22"
OMP_NUM_THREADS=1 python3 validate_check.py            # the checks of the check, 1 min -> results/validation.json
sh run_all.sh                                          # the seven cases, two processes at a time
OMP_NUM_THREADS=1 python3 run_case.py C > logs/C.log   # one case: A_c4.5 B_fit B_fixed C D A_c3.0 A_c6.0
python3 summarize.py                                   # the results table (markdown), from results/*.json
```

Each case writes `results/<CASE>.json` when it finishes, `figures/<CASE>.png` and `logs/<CASE>.log`. The figure shows
the data, the fit and the residual on the fitted planes nearest the true depth z0 and z0 +- 3 um, +- 8 um.
`run_case.py --smoke --out DIR` is a fast functional run (3 planes, 3 evaluations), not part of the check.

## The checks of the check (`validate_check.py`; `results/validation.json`, `logs/validate_check.log`) **[run, sandbox]**

- **The check renderer against `SomaRenderer` on the plain ellipsoid: bit-identical** (max |difference| 0) in every
  comparison:
  - a 141 px crop at the fit's settings (f 1, dzeta 0.14 um) and at the truth's (f 2, dzeta 0.07 um), each with the
    configured slope and with 0.95;
  - the cases' fit crop at the truth's settings.

  Rendering the planes in groups changes tau by at most 2.2e-15, for the check renderer and for `SomaRenderer` alike.
- **The marching-cubes area against `ellipsoid_area`.**

  | ellipsoid (a x b x c, um) | grid spacing (um) | relative error |
  |---|---|---|
  | 6.5 x 5 x 3.0 | 0.05 | -4.4e-5 |
  | 6.5 x 5 x 4.5 | 0.1 | -1.17e-4 |
  | 6.5 x 5 x 4.5 | 0.05 | -2.9e-5 |
  | 6.5 x 5 x 4.5 | 0.035 | -1.4e-5 |
  | 6.5 x 5 x 6.0 | 0.05 | -2.4e-5 |

  The error converges at second order.
- **The union of case C.**
  - Area: 392.677, 392.768 and 392.787 um2 at 0.1, 0.05 and 0.035 um. The 0.05 um value, used as S_ref, is within 5e-5
    of the finer one, i.e. 0.02 um2.
  - Half-height: 4.5 um analytically and on a 0.01 um grid of columns.
  - The cone's base lies inside the ellipsoid (least level 0, at the two touching points).
  - At the base, the ellipsoid's half-depth is at least the cone's everywhere, so the depth profile has no step.
- **Context for source (i).** On a small crop, the in-focus plane and +-3 um were rendered at f = 1, 2 and 4 (dzeta
  0.14, 0.07, 0.035 um).
  - f 1 against f 4: at most 0.016 in tau, on the in-focus plane's rim pixels (1.7 gl at B = 110 gl).
  - f 2 against f 4: at most 0.0056.
  - The difference f 1 - f 2 projects to 0.99 of f 1 - f 4. So the truth at f 2 holds about all of the fit's
    discretisation error relative to f 4. SPEC Block 45's status had named f 4 for the truth; the brief fixed f 2.

## Results **[run, sandbox]** (`python3 summarize.py`; `logs/summarize.out`)

Errors are fitted minus truth. For C and D, c is compared with the union's half-height (4.5 um), and S and r_eq with the
union's area (392.77 um2) and its r_eq (5.591 um). The SE of r_eq is SE(S) / (8 pi r_eq), by the delta method. A "*"
marks an error outside 2 SE. Wall times are on 2 CPUs shared with another agent's tests: 7.7-12.4 s per evaluation,
against 3.1-3.7 s for one render alone.

| case | c err | a err | b err | S err | r_eq err | status | n_eval | time | chi2/dof |
|---|---|---|---|---|---|---|---|---|---|
| A_c4.5 | -0.0022 um (-0.05 %, -0.6 SE) | +0.0013 um (+0.02 %, +1.2 SE) | +0.0005 um (+0.01 %, +0.5 SE) | -0.02 um2 (-0.01 %, -0.1 SE) | -0.0002 um (-0.00 %, -0.1 SE) | converged | 60 | 11 min | 9.093 |
| B_fit | -0.0001 um (-0.00 %, -0.0 SE) | +0.0012 um (+0.02 %, +1.0 SE) | -0.0001 um (-0.00 %, -0.1 SE) | +0.04 um2 (+0.01 %, +0.3 SE) | +0.0003 um (+0.01 %, +0.3 SE) | converged | 52 | 11 min | 9.098 |
| B_fixed | -0.7435 um (-16.52 %, -187.9 SE) * | +0.0804 um (+1.24 %, +62.5 SE) * | +0.0844 um (+1.69 %, +77.1 SE) * | -26.54 um2 (-7.47 %, -170.3 SE) * | -0.2025 um (-3.81 %, -167.0 SE) * | converged | 55 | 11 min | 9.805 |
| C | +0.0223 um (+0.50 %, +4.6 SE) * | -0.1950 um (-3.00 %, -147.6 SE) * | +0.5412 um (+10.82 %, +438.7 SE) * | -19.91 um2 (-5.07 %, -99.5 SE) * | -0.1435 um (-2.57 %, -98.3 SE) * | converged | 103 | 21 min | 13.297 |
| D | -0.0260 um (-0.58 %, -4.5 SE) * | -0.2762 um (-4.25 %, -176.7 SE) * | +0.5775 um (+11.55 %, +403.0 SE) * | -24.07 um2 (-6.13 %, -103.6 SE) * | -0.1740 um (-3.11 %, -101.9 SE) * | converged | 96 | 16 min | 13.858 |
| A_c3.0 | -0.0090 um (-0.30 %, -1.6 SE) | +0.0020 um (+0.03 %, +1.3 SE) | +0.0013 um (+0.03 %, +1.0 SE) | -0.24 um2 (-0.08 %, -1.2 SE) | -0.0020 um (-0.04 %, -1.2 SE) | converged | 92 | 13 min | 9.095 |
| A_c6.0 | -0.0079 um (-0.13 %, -2.2 SE) * | +0.0020 um (+0.03 %, +2.0 SE) * | +0.0011 um (+0.02 %, +1.2 SE) | -0.24 um2 (-0.06 %, -1.5 SE) | -0.0016 um (-0.03 %, -1.5 SE) | converged | 60 | 8 min | 9.096 |

The fitted parameters with their SEs:

| case | c_ref, S_ref | x0 (um) | y0 (um) | z0 (um) | a (um) | b (um) | c (um) | phi (rad) | mu (1/um) | gamma (um/um) | S (um2) | r_eq (um) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_c4.5 | 4.5, 355.28 | 503.3174 +- 0.0006 | 287.6414 +- 0.0006 | 32.7607 +- 0.0015 | 6.5013 +- 0.0012 | 5.0005 +- 0.0010 | 4.4978 +- 0.0040 | 0.6001 +- 0.0005 | 0.1500 +- 0.0001 | 0.7899 +- 0.0004 | 355.26 +- 0.16 | 5.3170 +- 0.0012 |
| B_fit | 4.5, 355.28 | 503.3175 +- 0.0007 | 287.6416 +- 0.0007 | 32.7610 +- 0.0014 | 6.5012 +- 0.0012 | 4.9999 +- 0.0011 | 4.4999 +- 0.0037 | 0.5999 +- 0.0005 | 0.1500 +- 0.0001 | 0.9499 +- 0.0005 | 355.32 +- 0.15 | 5.3175 +- 0.0011 |
| B_fixed | 4.5, 355.28 | 503.3183 +- 0.0007 | 287.6413 +- 0.0007 | 32.5312 +- 0.0013 | 6.5804 +- 0.0013 | 5.0844 +- 0.0011 | 3.7565 +- 0.0040 | 0.5995 +- 0.0006 | 0.1512 +- 0.0001 | 0.79 held | 328.73 +- 0.16 | 5.1147 +- 0.0012 |
| C | 4.5, 392.77 | 503.1265 +- 0.0007 | 287.9194 +- 0.0008 | 32.7754 +- 0.0017 | 6.3050 +- 0.0013 | 5.5412 +- 0.0012 | 4.5223 +- 0.0049 | 0.6030 +- 0.0012 | 0.1450 +- 0.0001 | 0.7910 +- 0.0005 | 372.86 +- 0.20 | 5.4471 +- 0.0015 |
| D | 4.5, 392.77 | 503.1248 +- 0.0010 | 287.9521 +- 0.0009 | 32.7666 +- 0.0020 | 6.2238 +- 0.0016 | 5.5775 +- 0.0014 | 4.4740 +- 0.0057 | 0.5497 +- 0.0022 | 0.1450 +- 0.0001 | 0.7811 +- 0.0005 | 368.70 +- 0.23 | 5.4166 +- 0.0017 |
| A_c3.0 | 3.0, 289.56 | 503.3177 +- 0.0008 | 287.6414 +- 0.0008 | 32.7602 +- 0.0014 | 6.5020 +- 0.0015 | 5.0013 +- 0.0013 | 2.9910 +- 0.0055 | 0.6000 +- 0.0006 | 0.1503 +- 0.0002 | 0.7896 +- 0.0005 | 289.32 +- 0.19 | 4.7982 +- 0.0016 |
| A_c6.0 | 6.0, 426.74 | 503.3174 +- 0.0006 | 287.6410 +- 0.0005 | 32.7582 +- 0.0017 | 6.5020 +- 0.0010 | 5.0011 +- 0.0009 | 5.9921 +- 0.0036 | 0.6003 +- 0.0004 | 0.1500 +- 0.0001 | 0.7896 +- 0.0004 | 426.50 +- 0.16 | 5.8258 +- 0.0011 |

The truth: x0 503.317, y0 287.641, z0 32.76 um; a 6.5, b 5.0 um; phi 0.6 rad; mu 0.15 /um; gamma 0.79 (0.95 in B).

Every fit converged ("converged": scipy status 2, 3 or 4) from the offset start, with no parameter at a bound. The
estimated scales B_k are within 0.04 gl of the true ones in A and B_fit, and within 1 gl in B_fixed, C and D.

## Reading

At the production settings, with the far slope fitted, the fit recovers the height c to within 0.01 um for an
ellipsoidal soma and to within 0.03 um (0.6 %) for the pyramidal one, with or without stems. The membrane area is right
for an ellipsoid, but 5-6 % low for the pyramidal soma. Source by source:

- **(i) The fit's discretisation (case A).** Fitting at f 1 / 0.14 um a truth rendered at f 2 / 0.07 um costs at most:
  - 0.009 um in c (0.3 %);
  - 0.24 um2 in S (0.08 %);
  - 0.002 um in a, b and r_eq.

  All of these are within 2.2 SE (c: -1.6, -0.6, -2.2 SE at c = 3, 4.5, 6 um). chi2/dof is 9.093-9.096. The noise and
  the rounding alone give 9 + 1/12 = 9.083, with an SD of the ratio of 0.011. The residuals have no structure. So D-219.1's
  discretisation error is at most of the order of the noise's SE (0.004-0.006 um in c). One noisy data set cannot
  separate a bias below that level from the noise; the noise field is common to the cases, so the three A errors are not
  independent.
- **(ii) The far slope (case B).**
  - Fitted (D-218.1, the default): the 20 % steeper slope is recovered, 0.9499 +- 0.0005, and c and S are unbiased (c
    -0.0001 um, S +0.04 um2).
  - Held at the configured 0.79 (D-218.2): c is 0.74 um low (-16.5 %), z0 0.23 um low, S 26.5 um2 low (-7.5 %), r_eq
    0.20 um low (-3.8 %). chi2/dof rises to 9.81, and the residual is structured: a negative core (data darker than
    the fit) at +-3 um and a negative ring in focus.

  This matches the Fisher check's linearised bias: -0.76 um in c and -29 um2 in S at c = 5 um, for a slightly different
  ellipsoid.
- **(iii) A soma that is not an ellipsoid (case C).** The union's half-height is recovered to +0.022 um (+0.5 %). The
  shape mismatch goes into the in-plane axes and the centre:
  - b is 0.54 um long (+10.8 %), along the cone's axis;
  - a is 0.20 um short (-3.0 %);
  - the centre moves 0.34 um toward the apex;
  - mu is 3 % low.

  S is 19.9 um2 below the union's 392.8 um2 (-5.1 %), and r_eq 0.14 um low (-2.6 %). The fitted area, 372.9 um2, lies
  between the ellipsoid's (355.3) and the union's (392.8). The SEs (0.005 um, 0.2 um2) do not cover these errors, which
  are 5 to 440 SE: they are model error, not noise. chi2/dof flags them at 13.3, 46 % above the noise's level, and the
  cone's shadow is the structure in the residual.
- **(iv) Stems leaving the soma (case D against C).** With the stems masked at their footprints plus 2 um:
  - c moves by -0.048 um (+0.022 um in C, -0.026 um in D);
  - S by -4.2 um2 (-1.1 %);
  - a by -0.08 um, phi by -0.05 rad;
  - the fitted far slope by -0.010.

  chi2/dof is 13.9. This difference includes both the stems' blur beyond the mask and the soma pixels the mask removes
  (the trunk's mask covers the cone's last 3 um); the check does not separate the two. Its sign is the opposite of the
  Fisher check's +0.1 um for a thicker stem (radius 1.3 um, mu 0.258 /um) leaking past its mask on one side, so the
  stems' effect depends on their geometry. Here it is small next to (iii).

For the pilot this means:

- The height needs the free far slope. Holding the slope at a wrong value costs 17 % of c.
- S and r_eq of a pyramidal soma carry a few-percent negative bias that the ellipsoid model cannot remove and its SE
  does not express.
- chi2/dof against the noise's level, and the residual's structure, are the signs that the shape is not an ellipsoid.
  On real data the noise level must first be measured.

## The check's own choices (not set by D-216.2's brief)

- The true centre's x and y, off the pixel grid.
- The crop's margin of 1 um about the region's bounding box. It does not affect the fitted pixels, since every source
  of A-C lies inside the crop.
- The stack holds only the 36 planes the fit selects; the fit ignores the window's other planes.
- One common noise seed for all cases.
- The cone's base and cross-section, and therefore its half-angle (Settings).
- The basal stems' angles (125 and -130 deg), their 1 um hidden attachment, and the stems' mu, which is the soma's.
- The truth at f 2 rather than SPEC Block 45's f 4. f 2 is the brief's setting; check (4) above shows it holds about all
  of the f 1 error relative to f 4.

## Files

- `recovery_lib.py`: the settings and cases, the solids, the check renderer, the mesh area, the stack, the fit's inputs,
  the errors.
- `run_case.py`: the entry point, one case by name.
- `run_all.sh`: the seven cases, two at a time.
- `validate_check.py`: the checks of the check.
- `recovery_figures.py`: the figure.
- `summarize.py`: the tables above.
- `results/<CASE>.json`: per case the settings, the truth and its references, the start, the stack, theta_hat, the SEs
  and the correlations, the errors, the residual per plane and the environment.
- `results/validation.json`.
- `figures/<CASE>.png`.
- `logs/<CASE>.log`, `logs/validate_check.log`, `logs/run_all.log`, `logs/summarize.out`.
