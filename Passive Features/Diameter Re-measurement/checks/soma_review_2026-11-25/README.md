# Soma fit: checks prompted by the independent review, 2026-11-25

Standalone checks (not pipeline code) run while discussing the review handoff
`TEEG_soma_dendrite_approach_review_handoff_2026-11-25.md` (the user's upload; not in the repository) with the user.
Simulated data only. Python 3 with the repository's `src/` (set `TEEG_W` to the workstream folder if moved).

`sensitivity.py OUT.json` -- three simulated somas (in-plane axes 7.0 x 5.6 um with c 2.5 and 4.5 um; 8.08 x 6.52 um
with c 7.26 um; 2 mu c = 0.87; centre 1.8 um above the traced depth), the pilot's acquisition and **the pilot's own
blur table** (0.080, 0.1536, 0.3924, 0.5620 um at 0, 0.28, 0.56, 0.84 um; D-170), far slope 0.79 held, 35 planes,
background 110 gl, noise SD 3 gl. For five fit configurations (region radius 5.97, 10 or 13 um; each plane's brightness
B_k re-solved, as today, or held at the value measured outside the soma) it gives, at the truth: the noise SE of each
parameter from the residuals' Jacobian; and the bias after one Gauss-Newton step when the data come from a world the
model does not hold (far slope 0.95 or 0.60; in-focus blur SD 0.258 um; a black level of 10 gl). The bias is
linearised: a sensitivity, not a fit's result. Cross-check of the linearisation: c 4.5 um, 13 um region, B_k re-solved,
slope 0.95 gives c -17.9 %, S -7.6 %; the full fit of handoff 14.4 gave -16.5 % and -7.5 %.

Results (`sensitivity.out`; run in the sandbox, 2026-11-25):

| Question | Result |
|---|---|
| Noise | SE of c 0.06-0.35 %, of S under 0.09 %, in every configuration: noise is not what limits the soma's area |
| The held far slope | world 0.95 against 0.79: c -4.5 to -59 %, S -1.3 to -18 %; world 0.60: c +7.9 to +90 %, S +2.9 to +28 %; worst for the flat soma (c 2.5 um), least on the D-241 region with B_k re-solved |
| Holding B_k (the review's proposal) | SE of c smaller by a factor 1.0-1.7; the slope's bias about twice as large on the 5.97 um region (c 4.5 um: -22.2 against -10.5 %), unchanged on 10-13 um regions |
| The region without the soma's edge (8.08 x 6.52 um soma, 5.97 um disc) | a, b still identified: SE 0.11 and 0.09 %, largest bias over the four worlds 1.4 and 2.1 % |
| In-plane axes in general | largest bias of a, b over the four worlds and all configurations 3.9 % |
| In-focus blur 0.258 against 0.080 um | c at most -4.0 %, S at most -1.3 % |
| Black level 10 gl | c at most 1.9 %, S at most 1.0 % |
| A thresholded outline for a, b | the in-focus plane's half-depth contour encloses 0.89, 0.94, 0.97 of pi a b at c 2.5, 4.5, 7.26 um: it depends on c |

What it does not show: anything about the pilot's stack; the effect of structure the model lacks (processes crossing
the soma, a non-Gaussian far blur, phase contrast, a tilted illumination cone).

`figures_revised.py sensitivity.json OUT_DIR` -- the three figures of the explanation of the revised approach (2026-11-25, 14:12):
fig5 the approach (schematic), fig6 the bias of c and of a against the true far slope (from `sensitivity.json`), fig7 S against
c with an illustrative carried range. The revised approach is proposed, not decided (handoff section 9).

## The notation figures and the profile over c (2026-11-25, 14:17-14:43)

`notation_figures.py OUT_DIR` -- figN1 where each symbol comes from (schematic; statuses read from `fit_soma.py`, `soma_fit.py`
and the notebook's `cleaning_config`); figN2 the ellipsoid's parameters (schematic to scale); figN3 D_k, R, tau_k, B_k on
three planes (simulated, noise 3 gl, the pilot's table); figN4 sigma_r(delta) and gamma (the pilot's table, D-170).

`profile_c.py GAMMA_TRUE OUT.json` (run with OMP_NUM_THREADS=1) -- the misfit, a_hat, b_hat and S with c held at 2.5-6.5 um,
the far slope held at 0.79 in the fit, the world's slope GAMMA_TRUE; noise-free; then a fit with c free. **Stopped early**
(the user's call, 14:43): the logs `profile_g079.out` (c 2.5-5.5) and `profile_g060.out` (c 2.5-5.0) hold the points
finished; c 6.0-6.5 and the c-free fits were not run. `profile_figure.py A B OUT_DIR` draws figN5 from the JSONs or the logs.

| Result (simulated, a 7.0, b 5.6, c 4.5 um) | world gamma 0.79 (held value right) | world gamma 0.60 (held value wrong) |
|---|---|---|
| misfit's minimum over held c | at the true c (RMS 0) | above 5.0 um (RMS still falling: 0.95 gl at 5.0); the linearised check put c_hat near 5.3 um |
| a_hat, b_hat over the held c | +11.7 / +13.9 % at c 2.5, -3.9 / -4.1 % at c 5.5 | +19.2 / +21.9 % at c 2.5 |
| S(a_hat, b_hat, c) over the held c | 380-431 um^2 (truth 405.6): -6.2 to +6.3 % over c 2.5-5.5 | 407-433 um^2 over c 2.5-5.0 |
| S(7.0, 5.6, c), the axes held at the truth | 315 to about 455 um^2 over c 2.5-5.5 | -- |

Readings: (1) with gamma held, the misfit's minimum over c is the c-free fit's c_hat by construction (a profile's minimum is
the joint minimum), so it cannot test c; (2) a_hat and b_hat compensate the held c, so they are not fixed independently of
it; (3) because they compensate, the stack's S varies far less over the held c than with fixed axes. All three are for this
simulated configuration with the model otherwise true; the real stack is untested.

`sigma0_identifiability.py` (2026-11-25, 14:50) -- if the in-focus blur sigma_r(0) were a ninth free parameter of the soma
fit: at a true 0.080 um its noise SE would be 0.0069 um, at 0.258 um 0.0052 um; its largest correlations 0.20 (a, mu) and
0.40 (a, mu); SE(c) unchanged (0.0074 and 0.0082 um, free or held). Model-true simulation, D-241 region, noise 3 gl.

`literature_c_range.py` (2026-11-25, 15:40; D-248 [corrected 2026-11-25, 16:10: logged on `main` as D-250; D-248 was taken meanwhile by another chat]) -- the arithmetic behind the literature-based range of c: c/r = 3V/(4A)
over sqrt(A/pi) from Benavides-Piccione et al. 2024's group means (human CA1 1.02, mouse CA1 1.07); the pilot's in-plane
radius 5.94-6.56 um (silhouette 111-135 um^2, handoff 14.2); living c 5.1-7.0 um (c/r 0.86-1.07); stack c 1.4-3.3 um
(x 0.27-0.47, Mohan et al. 2015's mounted/original thickness); living S 403-565 um^2. Assumptions in the docstring.

`background_check.py` (2026-11-25, 15:45) -- a background measured beside the soma (median of an 8-14 um annulus per
plane) and then held, in a simulated world where the model is otherwise true: the annulus reads 0.1-4.9 gl below the true
110 gl, more on defocused planes (the soma's own blurred halo); held, it biases c by -15.7 % and S by -10.2 % (one
linearised Gauss-Newton step). B_k re-solved at every step (as now) has no bias in that world.

## The tilted background and the region's radius (2026-11-25, 16:05-16:20; D-254)

`tilt_check.py OUT.json` (run with OMP_NUM_THREADS=1) -- simulated soma a 7.0, b 5.6, c 4.5 um, 2 mu c 0.87, the pilot's
blur table (D-170), far slope 0.79 in world and model, 35 planes, background tilted along x (B = 110 + 0.15 x gl, x in um
from the traced centre); the region a bare disc of radius r_R about the traced centre. For each r_R and each background
model -- 'const' (B_k, one column: the fit as it stands) or 'tilt' (b_0k + b_1k x + b_2k y, three columns, solved per plane
by least squares at every theta: D-254.1) -- the noise SEs of x0, y0, c, S (noise SD 3 gl, linearised) and the bias after
one Gauss-Newton step in the tilted world (noise-free). Results in `tilt_check.out` / `.json`.

`tilt_cost_check.py OUT.json [TAUS.npz]` -- the same soma and tilt; worlds with far slope 0.79, 0.95 or 0.60 while the
model holds 0.79; noise-free; the c-free fit's bias (one Gauss-Newton step) by r_R and model; and the background check of
D-254.3: on the 4 planes within 1.2 um of z0, the fitted background at the traced centre minus a plane fitted by least
squares to the data at r >= 12 um. Caches its 19 renders in TAUS.npz (287 MB, not committed; 33 s on 2 CPUs).
`tilt_figures.py` draws figT1 (the nested regression, schematic), figT2 (the shift column and the tilt column's best
imitation of it on R 5.97 and 10 um) and figT3 (the trade-off over r_R) from the two JSONs and the cache, and prints
rho^2, the share of the shift column (each plane's constant column removed) that the per-plane tilts reproduce
(`tilt_figures.out`).

| r_R (um) | 5.97 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| pixels per plane (bare disc) | 8556 | 15356 | 23984 | 34568 | 47028 |
| SE(x0), SE(y0): tilt / const (noise 3 gl) | 2.88, 2.74 | 1.47, 1.61 | 1.19, 1.24 | 1.10, 1.13 | 1.06, 1.07 |
| rho^2, the shift column mimicked by the tilts | 0.76 | 0.50 | 0.29 | 0.17 | 0.11 |
| const in the tilted world: bias x0 (um), S (%) | -0.054, -0.77 | -0.062, -0.37 | -0.082, -0.35 | -0.098, -0.32 | -0.110, -0.31 |
| tilt in the tilted world: any bias | 0 | 0 | 0 | 0 | 0 |
| tilt, world slope 0.95: bias c, S (%) | -9.5, -3.8 | -13.4, -5.9 | -16.6, -7.3 | -17.7, -7.6 | -17.9, -7.6 |
| tilt, world slope 0.60: bias c, S (%) | +17.0, +7.5 | +24.4, +11.0 | +30.0, +13.2 | +32.0, +13.7 | +32.5, +13.7 |
| background check (gl), slope 0.95 / 0.60 | -1.84 / +3.25 | -1.87 / +3.12 | -1.45 / +2.28 | -1.00 / +1.51 | -0.69 / +1.04 |

Readings: (1) the tilted projection removes the tilt's bias at every radius (model otherwise exact); without it, a 0.15
gl/um tilt (2 gl across the crop) moves x0 by 0.05-0.11 um and S by 0.3-0.8 %. (2) Its cost is the trade with the
centre: the per-plane tilts can mimic 76 % of the shift column on Allen's radius and 29 % at 10 um; in noise terms the
cost is negligible in absolute value (SE x0 with the tilt 0.0036 um at 5.97 um) and matters as the amplifier of
systematic errors along the shift, which was not tested. (3) A held wrong far slope biases c and S more on a larger
region, with or without the tilt (the constant model's values differ by at most 1.2 points), up to a plateau near 10-12
um. (4) The background check sees that slope error at 0.7-3.3 gl against a floor of at most 0.04 gl (the far field's own
halo), more on a smaller region; its sign gives the error's direction. (5) The 35 per-plane solves cost 0.009-0.034 s per
residual evaluation (r_R 5.97-10 um), against about 3.5 s for a render. All for this simulated configuration; the real
tilt near the soma, the far field's other structures and the real slope are untested.
