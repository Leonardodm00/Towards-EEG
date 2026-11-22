# Diameter reach of cell 529878215 (check, 2026-11-22)

**Question (the user, 2026-11-20 18:33).** The refined morphologies feed the passive and Ih fits from somatic
stimulation, so accuracy matters most where diameters still change what those fits see: how far from the soma do
wrongly defined diameters still have an effect?

**Measure (the user's answers, D-205).** Every dendritic diameter beyond path distance x from the soma is scaled
by s = 1.2 or 0.8; the "data" are the model's own somatic responses with the correct diameters; the shift the
passive fit makes in (Cm, Rm, Ra) is computed by one Gauss-Newton step of the fitter's cost (linearised), and
x* is the smallest distance beyond which every shift stays below 5 %. Parameters: Eyal et al. 2016, Table 1, cell
060308 (Cm 0.452 uF/cm2, Rm 38.91 kOhm cm2, Ra 203.23 Ohm cm), spines F = 1.9 beyond 60 um, passive, no Ih.

**Model.** The fitter's `PassiveCell` (`HPC script/Biological Fit/passive_fitting_hpc_fixed.py`, Cell 7) rebuilt
here: Import3d of the pilot's repaired SWC (3,136 basal and 5,354 apical sections, one segment each), the Hay stub
axon, `pas`, F on dendritic segments beyond 60 um, IClamp at soma(0.5), dt 0.025 ms. Fit targets: the fitter's
training target (the -200 pA, 0.5 ms pulse; RMSD on 1-100 ms after onset) alone, and with a -110 pA, 1 s step.

## Results [run, sandbox; NEURON 9.0.2]

| parameter set | fit target | x* Cm | x* Rm | x* Ra |
|---|---|---|---|---|
| Eyal 2016 cell 060308 (model R_in 220 MOhm) | pulse | 175 um | **250 um** | 175 um |
| Eyal 2016 cell 060308 | pulse + step | 200-250 um | **250 um** | 200-250 um |
| Rm 3.8 k, Cm 5.39 (matched to the cell's R_in 54.9 MOhm, tau 20.5 ms) | pulse | 90 um | 150 um | 80 um |
| same | pulse + step | 60 um | 150 um | 80 um |

(x* the same for s = 1.2 and 0.8 except where a range is given: s 1.2 / 0.8.)

- With all dendrites scaled by 1.2 (x = 0) the fit absorbs the error almost entirely: Cm -15 %, Rm +18 %, Ra +37 %
  (linearised; the full refit -15.5, +18.5, +47.6 %), near the exact invariance of the cable equation
  (Cm/s, s Rm, s^2 Ra: -16.7, +20, +44 %), broken only by the unscaled soma and axon stub (2 % of the membrane).
- Ra is set by the first 80-175 um; Rm and Cm by everything out to 150-250 um. Under Eyal's parameters 59 % of
  the cell's membrane (spines included) lies within 250 um; beyond it, scaling every diameter by 1.2 moves Rm by
  +3.2 % (full refit, pulse) and by 0.7 % beyond 400 um.
- Linearised against full Gauss-Newton refits (s = 1.2): within 0.5 percentage points at x = 200 and 250 um
  (Rm +6.0 / +3.2 % refit against +5.7 / +3.1 % linearised, pulse fit); the linearisation underestimates only the
  large proximal shifts (Ra at x = 0: +47.6 % refit, +36.9 % linearised).

## Caveats

- **A systematic error**: every diameter beyond x wrong by the same factor. Random errors partly average out and
  reach less far; this is the conservative case for "is the far tree negligible".
- **Passive only** (the user's choice): Ih at rest adds a conductance that shortens the electrotonic length, as
  the matched case does.
- **The morphology cannot carry plausible passive parameters as reconstructed**: Eyal's values give a model R_in of
  220 MOhm against the cell's 54.9 MOhm, and matching the cell's R_in and tau needs Cm 5.4 uF/cm2, ten times any
  measured value. Ih at rest, underestimated diameters and missing membrane are the candidates; the matched case
  is therefore a bracket (the electrotonically longest cell consistent with R_in and tau), not a model.

## Files and how to rerun

`diameter_reach.py` (the calculation), `smoke_diameter_reach.py` (5 checks: the analytic R_in of a ball-and-stick,
the cable invariance by refit and by the linearised step, the mask and restore, determinism; `smoke_diameter_reach.out`),
`plot_diameter_reach.py`, the outputs `diameter_reach_<label>.csv/.json`, `diameter_reach_529878215.png`, and the
run logs `run_*.out`. The SWC is not committed (the user's upload of 2026-11-22).

    python3 diameter_reach.py <reconstruction.swc> <out>                      # Eyal's parameters, about 5 min
    python3 diameter_reach.py <swc> <out> --no-grid --refit 250 --refit-scales 1.2 --label refit250
    python3 diameter_reach.py <swc> <out> --match-rin-mohm 54.894264 --match-tau-ms 20.495 --label matched
    python3 plot_diameter_reach.py fig.png <out>/diameter_reach_eyal2016_060308.csv <out>/diameter_reach_matched.csv
