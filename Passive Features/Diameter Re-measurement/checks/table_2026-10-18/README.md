# Table checks of 2026-10-16 to 2026-10-18 (the table chat)

Standalone checks, not pipeline code. They test, on simulated data with a known
truth, the claims the table chat made while the user settled D-053 to D-060:
how the bias table's spline behaves on the three-axis design, what its local
statistics report, how the inversion propagates the table's error, and what the
design of D-059/D-060 changes. Only `check_quadrature_bias.py` imports the
workstream (`allen_diameter`, Block 12's fit at commit `bee1e0b`); every other
script draws its own fixture (an analytic bias factor, Gaussian scatter, and
the fitted optical depth set to the true one unless the script says otherwise),
so nothing here renders a stack, runs the SGT fit on rendered phantoms or reads
Allen's data.

Needs numpy and scipy (run with NumPy 2.5.3, SciPy 1.18.1, Python 3.13.16);
`check_tps_deep.py` T1 also needs sympy. The `.py` files are pure ASCII.

Handoff that uses these results:
`docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` (§5, §8). Also cited
by `specs/SPEC.md` §8 (the Steps 4-5 findings, the design of D-059/D-060) and by
D-054 and D-059 in the decision log (project knowledge).

## Files

| Script | What it checks | Log (what a correct run prints) |
|---|---|---|
| `check_tps_scaling.py` (2026-10-16) | C1 an isotropic rescale of the inputs equals a change of the smoothing; C2 rescaling one axis changes the fit; C3 raw against range-normalised inputs, each with its CV smoothing; C4 per-axis scales chosen by CV (D-054's option B) | `check_tps_scaling.log` (re-run 2026-10-19) |
| `check_tps_scaling_seeds.py` (2026-10-16) | C3-C4 over further seeds | `check_tps_scaling_seeds.log` (re-run 2026-10-19) |
| `check_table_inversion.py` | C1 the inversion's mean correction against the Jensen term; C2 the corrected scatter $\tau/[b(1+\beta)]$; C3 the shortcut's error against mathematics Eq. 21 | `check_table_inversion.log` |
| `check_inverse_vs_classical.py` | inverting the mean response (classical calibration) against regressing $d$ on $\hat d$ over the phantoms, on two populations unlike the design | `check_inverse_vs_classical.log` (first saved 2026-10-19) |
| `check_table_se_coverage.py` | C1 the spline is a linear smoother; C2-C3 the table's reported SE against the spline's own sampling SD and the across-table SD (20 tables); C4 the phantoms' support along the inversion line; C5 a node darker than every phantom of its size. Writes `check_table_se_coverage_<mode>.npz` | `check_table_se_coverage_hetero.log`, `check_table_se_coverage_homo.log` |
| `check_table_steps45_extra.py` | A the CV choice's share of the across-table SD; B and B2 the table term of a node's corrected error (B's tolerance used the true slope and failed 19 of 35; B2, with the table's own slope, passes 35 of 35); C the reach of the spline's weights; D the $\ln\alpha$ range of the normalisation; E the running median's SD; F the failure rate's noise. Reads the two `.npz` files | `check_table_steps45_extra.log` (A-E), `..._B2.log`, `..._F.log` |
| `check_grouped_design.py` | at an equal number of measured phantoms, 300 draws x 5 realizations against 1500 independent draws; the design effect $1 + (k-1)\rho$ | `check_grouped_design.log` |
| `check_tps_deep.py` | T1 the biharmonic Green's function in 2-D and 3-D; T2 the two limits of the smoothing; T3 the variational form; T4 the effective degrees of freedom; T5 affine reproduction; T6 the cost of one fit; T7 cross-validation folds under D-059/D-060 (plain against by draw) | `check_tps_deep.log` |
| `check_b_target.py` | on a toy of the D-059/D-060 design with a closed form, what the table estimates: the rows' mean at a fitted $\ln\hat\alpha$ against the draws' own means | `check_b_target.log` |
| `check_quadrature_bias.py` | the bias that D-052.2's quadrature rule ($N$ = 848) puts on the IRLS $\hat d$ of Block 12's fit, on noise-free profiles computed at $16N$ | `check_quadrature_bias.log` (first saved 2026-10-19) |
| `check_table_se_coverage_hetero.npz`, `..._homo.npz` | the 20 tables' query-point results of `check_table_se_coverage.py`, read by `check_table_steps45_extra.py` A | -- |

`check_table_se_coverage.py` and `check_tps_deep.py` are byte-identical to the
copies in `docs/TEEG_thin_plate_spline_in_3D/scripts/` (sha256 checked
2026-10-19), which that document's own checks import.

## Commands

Run from this folder (the scripts import each other by name):

```
cd "Passive Features/Diameter Re-measurement/checks/table_2026-10-18"
python3 check_tps_scaling.py            # a few minutes (2.4 min on 2 busy cores)
python3 check_tps_scaling_seeds.py      # about 10 min (4 seeds)
python3 check_table_inversion.py        # seconds
python3 check_inverse_vs_classical.py   # seconds
python3 check_table_se_coverage.py      # hetero, about 9 min; writes check_table_se_coverage_hetero.npz
python3 check_table_se_coverage.py homo # about 9 min
python3 check_table_steps45_extra.py    # A-E, about 3 min (needs both .npz)
python3 check_table_steps45_extra.py B2 # about 2 min
python3 check_table_steps45_extra.py F  # seconds
python3 check_grouped_design.py         # a few minutes
python3 check_tps_deep.py               # about 6 min
python3 check_b_target.py               # seconds
python3 check_quadrature_bias.py "$(cd ../.. && pwd)"   # the workstream; about 1-2 min
```

Re-run on 2026-10-19 from this folder: `check_b_target.py`,
`check_table_inversion.py` and `check_table_steps45_extra.py F` reproduced
their logs byte for byte; the two scaling checks and `check_quadrature_bias.py`
wrote the logs now here, which agree with the numbers recorded before (D-054's
Why: range-normalised RMS 0.0073-0.0100, raw 0.0092-0.0106, option B
0.0062-0.0087 over the five seeds; SPEC Block 12's status: at most 0.07 grey
levels and 7.5e-4 um in the fitted regime); `check_inverse_vs_classical.py`
had no earlier saved record **[run, sandbox]**.

## Lines to expect

- `check_tps_scaling.log`: `C1 isotropic c=7, s=10 vs s/c^2 PASS max|delta| = 2.34e-13`; C3 RMS 0.0101 raw, 0.0073 range-normalised.
- `check_table_inversion.log`: every row `PASS` (10).
- `check_inverse_vs_classical.log`: `thin -0.00035 0.00046 -0.03966 0.00039 classical unbiased (PASS), inverse biased`.
- `check_table_se_coverage_hetero.log`: `C1 linear smoother: ... -> PASS`; smoothing 10 in 14 of 20 batches; C5 `root 0.4921 um`.
- `check_table_steps45_extra_B2.log`: `B2: 35 of 35 pass`.
- `check_table_steps45_extra_F.log`: 21 rows, every one `PASS`.
- `check_grouped_design.log`: `pooled SD ratio 1.062, predicted 1.000 -> PASS` and `1.335, predicted 1.342 -> PASS`.
- `check_tps_deep.log`: `T7 plain : lambda chosen [0.01, 0.1, 0.01, 0.01, 0.1, 0.1]; mean RMS error of the table 0.0752` and `T7 by draw : lambda chosen [10.0, 100.0, 100.0, 0.1, 10.0, 10.0]; mean RMS error of the table 0.0295`.
- `check_b_target.log`: `overall: PASS`.
- `check_quadrature_bias.log`: 32 rows `converged`; largest $|\hat d - d|$ 7.5e-04 um (Laplace, $d$ 5.5 um, $w$ 0.5 um).
