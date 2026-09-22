# Ih Fit -- the I_h fitting pipeline (TEEG_Ih_fit_staged_plan.md, decisions D-005 .. D-007)

This folder started on 2026-09-22 as a byte-copy of `../Biological Fit/` and is
the copy that the staged plan edits. `../Biological Fit/` is never touched: it
is the oracle for `regression_passive_identity.py`, which must PASS after every
stage. See `CHANGELOG_ih_fit.md` for what each stage added.

Files that are NEW relative to `Biological Fit/` (Stage 0-1):

| file | role |
|---|---|
| `mod/Ih.mod`, `mod/Ih_human.mod` | Kole 2006 / Rich 2021 h-current with RANGE `vshift`, `vshift_minf`, `tau_scale` (defaults = published models) |
| `human_ih_params.py` | literature constants + pure-Python reference kinetics (from the synthetic benchmark) + shift/scale-aware `minf`, `mtau_ms`, `v_half_mV` |
| `ih_mechanism.py` | `IhSpec`, `attach_ih`, `set_ih`, `balance_e_pas` (rest balance, plan Eq. 11.3), `ih_rest_summary`, `rest_drift`, `sag_metrics` |
| `synthetic_ground_truth.py` | copied from the synthetic benchmark: ball-and-stick SWC, archive-format cell writer (fixtures; Stage 7 extends it) |
| `smoke_ih_fit.py` | Stage 1 smoke suite S1-S5, S10 |
| `regression_passive_identity.py` | Stage 0 gate: passive 3-D path identical to `Biological Fit/` |

First run on the cluster (login node, conda env `prova`):

    cd "<repo>/Passive Features/HPC script/Ih Fit"
    nrnivmodl mod                          # once per architecture; creates x86_64/
    python smoke_ih_fit.py                 # expect "smoke_ih_fit: 6/6 passed"
    python regression_passive_identity.py --archive-cell "<ARCHIVE_ROOT>/L3_exc/specimen_<id>"
                                           # expect "regression_passive_identity: PASS (...)"

Sign convention (pinned by S5): `vshift` and `vshift_minf` move the curves along
v; positive = more activation at a given v. Kalmbach 2018's "-20 mV" shift of
the Kole rates is `vshift = +20` here.
