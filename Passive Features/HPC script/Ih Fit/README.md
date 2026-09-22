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
| `param_spec.py` | `ParamSpec`: which parameters are fitted, in which order and coordinates, and how they are applied to a cell. `PASSIVE_3D` (legacy) and `make_ih_spec()` (4/5/6-D) |
| `smoke_ih_fit.py` | smoke suite S1-S7, S10, S11 |
| `regression_passive_identity.py` | Stage 0 gate: passive 3-D path identical to `Biological Fit/` |

Files EDITED relative to `Biological Fit/` (Stages 2-4) -- all with legacy
defaults, so every existing call behaves exactly as before:

| file | change |
|---|---|
| `passive_fitting_hpc_fixed.py` | loader: `ls_max_amplitude_pA=None`, depolarising bundles, `swc_dir`, `sweep_has_spike`, `bundle_trough_mV`; fit: `spec=` through `fit_one_cell`, `_build_loss_function`, `_gp_parameter_uncertainty`, `_estimate_residual_noise_at_mle`; `PassiveFitResult.params` / `.sigmas_by_name` / `.param_spec`; `OptimiserInputs.param_spec` |
| `passive_long_step_training.py` | `ls_window_mode` (`after_onset` / `step` / `sweep`), `assign_ls_roles`, `split_train_validation_ih`, variadic `loss(*q)`, `integrate_long_step(spec=, ih_protocol=, ...)` |
| `synthetic_ground_truth.py` | `IhConfig.vshift_mV` / `.vshift_minf_mV` / `.tau_scale` |

First run on the cluster (login node, conda env `prova`):

    cd "<repo>/Passive Features/HPC script/Ih Fit"
    nrnivmodl mod                          # once per architecture; creates x86_64/
    python smoke_ih_fit.py                 # expect "smoke_ih_fit: 9/9 passed"
    python regression_passive_identity.py --archive-cell "<ARCHIVE_ROOT>/L3_exc/specimen_<id>"
                                           # expect "regression_passive_identity: PASS (...)"

Sign convention (pinned by S5): `vshift` and `vshift_minf` move the curves along
v; positive = more activation at a given v. Kalmbach 2018's "-20 mV" shift of
the Kole rates is `vshift = +20` here.
