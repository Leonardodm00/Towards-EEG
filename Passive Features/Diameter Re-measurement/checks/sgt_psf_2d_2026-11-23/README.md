# The SGT kernel in 2-D for the line renderer: the isotropic PSF against the separable construction (D-225)

2026-11-23. Standalone check, not pipeline code: it reads `src/` and changes nothing. Every number below carries its
basis: **[run here]** (this folder's scripts), **[reasoning]**, or **[code]** / **[SPEC]** / **[log]** (read in the
repository).

## Question

D-222.1 makes the SGT option's kernel, with $\zeta = 1$, the isotropic PSF whose line projection is the SGT profile
kernel (D-066.1); D-224 left its construction in 2-D, and its cost, to wave 7's contract. Block 35's line renderer
integrates each sample's Gaussian over the pixels exactly because the Gaussian is separable in the image axes
**[code]**; no other isotropic kernel is. The cheaper construction is a kernel that is the SGT across the measuring line
and the law's Gaussian along it: both have the same line projection across the line, so for a tube parallel to the
image plane they give the same profile **[reasoning, exact]**. How far apart are they for a tilted tube, and does the
difference move the fitted diameter?

## Method

`fft_line.py` renders Block 35's quantity -- the pixel means of $\sum_j \Delta A_j * K_{jk}$ read by
`profiles.sample_profile` -- for any kernel given by its Fourier transform: the slab maps of Block 35 (partition
absorption, `line_fast._footprint`, `geometry.slab_centres`, `slab_absorbance`, `transmitted_before`) on the fine
lattice $h = p/f$, one zero-padded periodic grid, the transfer times the pixel box's $\mathrm{sinc}(p\nu_x)
\mathrm{sinc}(p\nu_y)$ and the half-sample phase that puts pixel centres on the output lattice ($f$ even). The kernels:

- `gauss_transfer`: Block 4's Gaussian, $e^{-2\pi^2\sigma^2|\nu|^2}$;
- `iso_transfer`: the isotropic PSF of D-066.1, $\hat h(a|\nu|)$ (projection-slice theorem), $\hat h$ the unit SGT's
  characteristic function by a 1-D FFT of `blur_kernels.sgt_unit_pdf` on $[-2000, 2000)$, $2^{23}$ points;
- `sep_transfer`: the SGT across the line, the law's Gaussian along it, $\hat h(a\,\nu\cdot\hat y)\,e^{-2\pi^2\sigma^2
  (\nu\cdot\hat e)^2}$;

with $a = 2\sqrt{2\ln 2}\,\sigma_{\rm r}(\delta)/W(\kappa)$ (D-222.2). Settings (`run_compare.py`): `line_fine_factor`
4, `dzeta_um` 0.04 (both kernels alike; coarser than the pipeline's 8 and 0.02, for time), half-length 8 um, the line
across the tube's heading through the tube's axis 0.04 um aside, $v = n\,p$, $n = -26..26$, `n_avg` 4, planes
$-4..4$, $\mu$ = 1 /um, $z_{\rm ax}$ = 0.1 um; pairs whose law SD exceeds 1.3 um are left out for every kernel alike
(their Gaussian share is in the last column below), the grid padded by 8 um. Shapes: `heavy` (2, 0.5, 1), `b13`
(1.3, 0, 1), `flat4` (4, 0, 1), `tails1` (2, 1, 1). Noise: the identifiability check's $\sigma/B$ = 1.1/150 in
$\tau$ units.

`bias.py`: the separable model fitted to the isotropic model's noise-free profiles, linearised: one Gauss-Newton step
from the truth over $(d, \mu, v_0, z_{\rm ax}, \log_{10}\beta, \xi)$ (central differences of the separable model;
the backgrounds known), and the same with the shape held at the truth.

```
cd "Passive Features/Diameter Re-measurement/checks/sgt_psf_2d_2026-11-23"
python3 run_compare.py ../../src validate.json validate   # the FFT reference against Block 35: 1 min
python3 run_compare.py ../../src compare.json compare > compare.out   # 18 geometries x 4 shapes: about 3 h
python3 bias.py ../../src bias.json > bias.out            # 4 cases: about 40 min
python3 tables.py                                         # the two tables below, from the .out files
```

## Results

**Validation [run here]** (`validate.json`): the FFT reference with the Gaussian against Block 35's
`line_fast._profiles` (partition, the same lattice and slabs): max $|\Delta\tau|$ 8.9e-12 (flat, heading 0),
2.2e-16 (flat, heading 0.52 rad), 1.1e-16 (tilt 0.3 rad, heading 0.52 rad, half-length 3.2 um). The reference computes
Block 35's quantity.

**The two constructions [run here]** (`compare.out`; the 53 of the 72 rows that finished: the run was stopped to free
the processors for wave 7's implementers, the $d$ = 1.5 um rows and two $d$ = 0.8 um ones left out): max and RMS of $|\tau_{\rm iso} - \tau_{\rm sep}|$ over the 9 planes x 53 offsets, in noise
SD; beside, the shape's own effect $\max|\tau_{\rm iso} - \tau_{\rm Gauss}|$ and the Gaussian share of the pairs left
out.

<!-- compare -->
| d (um) | tilt (rad) | heading (rad) | shape | max abs(iso - sep) | RMS | max abs(iso - Gauss) | pairs left out (Gauss), max |
|---|---|---|---|---|---|---|---|
| 0.4 | 0.0 | 0.00 | heavy | 0.001 | 0.000 | 3.05 | 0.00 |
| 0.4 | 0.0 | 0.00 | b13 | 0.000 | 0.000 | 5.01 | 0.00 |
| 0.4 | 0.0 | 0.00 | flat4 | 0.000 | 0.000 | 3.39 | 0.00 |
| 0.4 | 0.0 | 0.00 | tails1 | 0.012 | 0.006 | 6.11 | 0.00 |
| 0.4 | 0.0 | 0.52 | heavy | 0.001 | 0.000 | 2.87 | 0.00 |
| 0.4 | 0.0 | 0.52 | b13 | 0.001 | 0.000 | 4.70 | 0.00 |
| 0.4 | 0.0 | 0.52 | flat4 | 0.002 | 0.000 | 3.19 | 0.00 |
| 0.4 | 0.0 | 0.52 | tails1 | 0.010 | 0.004 | 5.76 | 0.00 |
| 0.4 | 0.3 | 0.00 | heavy | 0.436 | 0.146 | 2.96 | 0.73 |
| 0.4 | 0.3 | 0.00 | b13 | 0.781 | 0.252 | 4.82 | 0.73 |
| 0.4 | 0.3 | 0.00 | flat4 | 0.584 | 0.193 | 3.13 | 0.73 |
| 0.4 | 0.3 | 0.00 | tails1 | 0.606 | 0.220 | 5.96 | 0.73 |
| 0.4 | 0.3 | 0.52 | heavy | 0.434 | 0.147 | 2.78 | 0.74 |
| 0.4 | 0.3 | 0.52 | b13 | 0.778 | 0.253 | 4.51 | 0.74 |
| 0.4 | 0.3 | 0.52 | flat4 | 0.581 | 0.192 | 2.96 | 0.74 |
| 0.4 | 0.3 | 0.52 | tails1 | 0.605 | 0.220 | 5.66 | 0.74 |
| 0.4 | 0.6 | 0.00 | heavy | 0.363 | 0.114 | 4.13 | 1.66 |
| 0.4 | 0.6 | 0.00 | b13 | 0.493 | 0.215 | 6.66 | 1.66 |
| 0.4 | 0.6 | 0.00 | flat4 | 1.482 | 0.424 | 3.14 | 1.66 |
| 0.4 | 0.6 | 0.00 | tails1 | 0.805 | 0.184 | 8.17 | 1.66 |
| 0.4 | 0.6 | 0.52 | heavy | 0.354 | 0.114 | 3.93 | 1.60 |
| 0.4 | 0.6 | 0.52 | b13 | 0.492 | 0.215 | 6.32 | 1.60 |
| 0.4 | 0.6 | 0.52 | flat4 | 1.473 | 0.426 | 3.00 | 1.60 |
| 0.4 | 0.6 | 0.52 | tails1 | 0.792 | 0.183 | 7.83 | 1.60 |
| 0.8 | 0.0 | 0.00 | heavy | 0.005 | 0.002 | 4.82 | 0.00 |
| 0.8 | 0.0 | 0.00 | b13 | 0.001 | 0.000 | 8.44 | 0.00 |
| 0.8 | 0.0 | 0.00 | flat4 | 0.000 | 0.000 | 6.73 | 0.00 |
| 0.8 | 0.0 | 0.00 | tails1 | 0.040 | 0.020 | 8.99 | 0.00 |
| 0.8 | 0.0 | 0.52 | heavy | 0.004 | 0.002 | 4.77 | 0.00 |
| 0.8 | 0.0 | 0.52 | b13 | 0.001 | 0.000 | 8.32 | 0.00 |
| 0.8 | 0.0 | 0.52 | flat4 | 0.004 | 0.000 | 6.55 | 0.00 |
| 0.8 | 0.0 | 0.52 | tails1 | 0.032 | 0.016 | 8.90 | 0.00 |
| 0.8 | 0.3 | 0.00 | heavy | 1.032 | 0.408 | 5.25 | 2.40 |
| 0.8 | 0.3 | 0.00 | b13 | 1.863 | 0.709 | 9.08 | 2.40 |
| 0.8 | 0.3 | 0.00 | flat4 | 1.286 | 0.467 | 6.36 | 2.40 |
| 0.8 | 0.3 | 0.00 | tails1 | 1.417 | 0.618 | 9.70 | 2.40 |
| 0.8 | 0.3 | 0.52 | heavy | 1.030 | 0.409 | 5.19 | 2.43 |
| 0.8 | 0.3 | 0.52 | b13 | 1.857 | 0.710 | 8.98 | 2.43 |
| 0.8 | 0.3 | 0.52 | flat4 | 1.277 | 0.466 | 6.27 | 2.43 |
| 0.8 | 0.3 | 0.52 | tails1 | 1.411 | 0.615 | 9.61 | 2.43 |
| 0.8 | 0.6 | 0.00 | heavy | 0.663 | 0.305 | 5.41 | 5.14 |
| 0.8 | 0.6 | 0.00 | b13 | 1.174 | 0.562 | 8.61 | 5.14 |
| 0.8 | 0.6 | 0.00 | flat4 | 3.132 | 1.015 | 6.31 | 5.14 |
| 0.8 | 0.6 | 0.00 | tails1 | 1.727 | 0.524 | 11.51 | 5.14 |
| 0.8 | 0.6 | 0.52 | heavy | 0.662 | 0.303 | 5.38 | 5.18 |
| 0.8 | 0.6 | 0.52 | b13 | 1.176 | 0.557 | 8.56 | 5.18 |
| 0.8 | 0.6 | 0.52 | flat4 | 3.062 | 0.997 | 6.32 | 5.18 |
| 0.8 | 0.6 | 0.52 | tails1 | 1.718 | 0.518 | 11.48 | 5.18 |
| 1.5 | 0.0 | 0.00 | heavy | 0.014 | 0.006 | 7.98 | 2.19 |
| 1.5 | 0.0 | 0.00 | b13 | 0.004 | 0.001 | 13.29 | 2.19 |
| 1.5 | 0.0 | 0.00 | flat4 | 0.000 | 0.000 | 10.49 | 2.19 |
| 1.5 | 0.0 | 0.00 | tails1 | 0.110 | 0.056 | 15.47 | 2.19 |
| 1.5 | 0.0 | 0.52 | heavy | 0.014 | 0.006 | 7.93 | 2.19 |
<!-- /compare -->

Flat tubes: at most 0.04 noise SD (the caps and the 2-D tails of `tails1`); tilted tubes: up to 1.9 noise SD, a
sizeable fraction of the shape's own effect (3-10 noise SD). A tilted tube's slabs lie at different places along its
axis, and the kernel's part along the line weighs them; the separable kernel's Gaussian along the line is not the
isotropic kernel's **[reasoning]**.

**The diameter [run here]** (`bias.out`; linearised; SE from the same Jacobian, noise 1.1/150; three of the four cases,
the fourth stopped with the comparison):

<!-- bias -->
| d (um) | tilt (rad) | heading (rad) | shape | bias of d, shape fitted | in SE | bias of d, shape held | in SE | SE(d)/d | RMS residual (noise SD) |
|---|---|---|---|---|---|---|---|---|---|
| 0.4 | 0.3 | 0.52 | b13 | -5.03 % | -1.23 | +0.66 % | +0.18 | 4.10 % | 0.25 |
| 0.4 | 0.6 | 0.52 | heavy | -0.09 % | -0.03 | -0.36 % | -0.14 | 2.76 % | 0.11 |
| 0.8 | 0.3 | 0.52 | b13 | -3.08 % | -4.52 | +0.22 % | +0.37 | 0.68 % | 0.71 |
<!-- /bias -->

With the shape fitted -- the option's use -- the construction's error moves a tilted node's $\hat d$ by -5 % (-1.2 SE)
at 0.4 um and by -3 % (-4.5 SE) at 0.8 um (`b13`, tilt 0.3 rad): the shape absorbs part of the misfit and drags $d$
with it. With the shape held at the truth the effect is at most 0.4 SE; for `heavy` at tilt 0.6 rad it is small either
way. The size depends on the shape and the geometry.

## Conclusion (D-225.1)

The separable construction is not an acceptable stand-in for D-222.1's isotropic PSF on tilted nodes; the line
renderer's SGT path computes the isotropic PSF (Block 49), whose exact pixel integral needs a non-separable
convolution -- here an FFT. `fft_line.py` is a brute-force reference for Block 49's oracles (iii).

## Limits

The coarser lattice and slabs (both kernels alike); pairs above 1.3 um left out; the partition absorption (the fit
uses the hybrid); the linearised bias (one Gauss-Newton step) on four cases; noise-free.
