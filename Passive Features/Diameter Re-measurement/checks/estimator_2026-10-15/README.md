# Estimator checks of 2026-10-15 (the evidence behind D-052)

Standalone checks, not pipeline code. They import `allen_diameter` from the
workstream through `check_objective.load(repo)` and were written against
commit `9e3296c` of `sci/diameter-pipeline`. They simulate profiles with the
pipeline's own model (`tube_model.model_profile_kernel`) at Block 5's 53
offsets and fit them; nothing here renders a stack or reads Allen's data.

Needs numpy, scipy, matplotlib; `render_tex.py` also needs pdflatex and
pdftoppm. The equation sources are `eqs_identifiability*.tex` and
`eqs_flexible*.tex`. In `sgt_kernel.py` and the result files the shape
parameters are named `(p, r, gamma)`; the documents and D-052 call them
`(beta, xi, zeta)`.

Handoff that uses these results:
`docs/TEEG_diameter_correction_table_handoff_2026-10-16.md`.

## Files

| File | What it is |
|---|---|
| `check_objective.py` | loader of the workstream (`load(repo)`) and the objective checks C2, C3, C5 of 2026-10-15 |
| `sgt_kernel.py` | the skewed generalized t kernel, two-piece form, of FWHM `w` and mode 0: `sgt_pdf(fwhm, p, r, gamma)`; `python sgt_kernel.py "$REPO"` checks it (unit integral, FWHM, members against the pipeline's kernels) |
| `identifiability_check.py` | Student-only run (seed 20261016): impostor search, cumulants, Monte Carlo of Block 5, the Gaussian of fitted width and the Student fit |
| `identifiability_example.py`, `kernel_pinning_check.py`, `kernel_formula_check.py`, `fig_identifiability.py` | the Student run's example, pinned-kernel check, kernel formulas, figures |
| `identifiability_sgt.py` | three-shape run (seed 20261017): impostor search (part A) and Monte Carlo (part B) of the SGT fit, unweighted and centre-weighted at lambda 0.5 um (IRLS), against the Gaussian of fitted width |
| `identifiability_sgt_example.py`, `independent_model_check.py`, `sgt_pinning_check.py`, `fig_sgt.py` | the SGT run's example, an independent quadrature of the model, the pinned-kernel check, figures |
| `ident/` | the Student run's results: `identifiability.json`, `pinning.json`, `example.npz` |
| `ident_sgt/` | the SGT run's results: `identifiability_sgt.json`, `pinning_sgt.json`, `example_sgt.npz` |

## Commands

Run from this folder, with the workstream as `REPO`:

```
REPO="<clone>/Passive Features/Diameter Re-measurement"
python sgt_kernel.py "$REPO"
python identifiability_check.py --repo "$REPO" --out ident --n-mc 40 --procs 2
python identifiability_sgt.py --repo "$REPO" --student ident/identifiability.json --out ident_sgt --n-mc 40 --procs 2
python identifiability_sgt_example.py --repo "$REPO" --res ident_sgt/identifiability_sgt.json
python independent_model_check.py --repo "$REPO" --res ident_sgt/identifiability_sgt.json
python sgt_pinning_check.py --repo "$REPO" --res ident_sgt/identifiability_sgt.json
python kernel_pinning_check.py --repo "$REPO" --res ident/identifiability.json
python kernel_formula_check.py --repo "$REPO"
python fig_identifiability.py --dir ident
python fig_sgt.py --dir ident_sgt --student ident/identifiability.json
python render_tex.py eqs_flexible_body.tex eqs_flexible
```

The two Monte Carlo runs take about 15 min (Student) and 11 min (SGT: part A
about 2 min, part B about 9 min) on 2 processes in the sandbox. They overwrite
the result files in `ident/` and `ident_sgt/`; run them in a copy of this
folder to keep the recorded ones.

## What a correct run prints

- `sgt_kernel.py`: one PASS line per check, ending with `ALL PASS`.
- `identifiability_sgt.py`: verdict `refuted_in_practice_thin` true
  (`n_practical_thin` 9), `refuted_exactly_thin` false,
  `refuted_in_practice_thick` false, `thick_identifiable` true,
  `search_check_pass` true; `approach` false for the 0.6 um truths, true for
  the 1.5 um truths.
- `independent_model_check.py`: PASS on three cases; the thick Laplace case
  FAILS the 1e-4 gl per-sample tolerance (quadrature error of the cusped
  kernel, 8.3e-4 gl) while its Delta chi^2 agrees to 0.002 (20.8413 against
  20.8431), and the script ends with `SOME FAIL`. This is the recorded result,
  not a regression.
- `sgt_pinning_check.py`: medians 0.601-0.614 um, SD 0.052-0.060 um,
  `identifies False` on all four lines.
- `identifiability_check.py`: verdict `refuted_in_practice` True,
  `n_practical` 2, `refuted_exactly` False, `approach_identifies_d` False.
- `kernel_pinning_check.py`: `oracle|none ... identifies True` for both thin
  truths. `kernel_formula_check.py`: ends with PASS.

## Re-run on 2026-10-16

From a copy of this folder, against the workstream at `9e3296c`:
`sgt_kernel.py` ALL PASS; `independent_model_check.py` the four lines above,
the same numbers as on 2026-10-15; `sgt_pinning_check.py` medians 0.614,
0.607, 0.604, 0.601 um, SD 0.052, 0.053, 0.060, 0.058 um. The Monte Carlo
runs were not repeated.
