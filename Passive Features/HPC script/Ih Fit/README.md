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
| `smoke_ih_fit.py` | smoke suite S1-S8, S10-S12 |
| `run_ih_fit.py` | **the entrypoint**: one arm x one group. Arms, the I_h configuration, the D-006 protocol, and the three CSVs a campaign is read from |
| `submit_ih_fit.sh` | PBS wrapper: `qsub -v GROUP=L3_exc,ARM=ih6`. Idempotent mtime-aware `nrnivmodl` guard, mechanism verification, `DRY_RUN=1` |
| `synth_gt_grid.py` | the synthetic cohort's manifest: per-cell ground truth, incl. the two fitted kinetic knobs, the false-positive control, and the noise level read out of a real run |
| `gen_from_manifest.py` | manifest row -> Phase-0 archive (pure mapping + a NEURON-side generator) |
| `ih_recovery_report.py` | **pure**: truth vs estimate, the recovery table, the C_m inflation, the false-positive table, the gate |
| `run_ih_recovery.py` | **Stage 7's entrypoint**: draw -> generate -> fit (via `run_ih_fit.main`) -> report -> gate. Exit 0 = gate passed, 2 = gate failed |
| `submit_ih_recovery.sh` | PBS wrapper for Stage 7: `qsub -v MORPH_ROOT=...,NOISE_FROM_RESULTS=...` |
| `smoke_ih_recovery.py` | Stage 7's smoke suite R1-R9 |
| `regression_passive_identity.py` | Stage 0 gate: passive 3-D path identical to `Biological Fit/` |

Files EDITED relative to `Biological Fit/` (Stages 2-4) -- all with legacy
defaults, so every existing call behaves exactly as before:

| file | change |
|---|---|
| `passive_fitting_hpc_fixed.py` | loader: `ls_max_amplitude_pA=None`, depolarising bundles, `swc_dir`, `sweep_has_spike`, `bundle_trough_mV`; fit: `spec=` through `fit_one_cell`, `_build_loss_function`, `_gp_parameter_uncertainty`, `_estimate_residual_noise_at_mle`; `PassiveFitResult.params` / `.sigmas_by_name` / `.param_spec`; `OptimiserInputs.param_spec` |
| `passive_long_step_training.py` | `ls_window_mode` (`after_onset` / `step` / `sweep`), `assign_ls_roles`, `split_train_validation_ih`, variadic `loss(*q)`, `integrate_long_step(spec=, ih_protocol=, ...)` |
| `passive_fitting_hpc_fixed.py` (Phase 3) | the bootstrap and the GP diagnostic key on the result's own `ParamSpec` instead of a module `PARAM_NAMES` triple; `BootstrapCIResult.param_names`; per-axis q -> physical transform |
| `synthetic_ground_truth.py` | `IhConfig.vshift_mV` / `.vshift_minf_mV` / `.tau_scale` |

## Paths

| | |
|---|---|
| Laptop clone | `~/Towards-EEG` |
| Cluster clone | `/davinci-1/home/ldellamea/TEEG/Towards-EEG` |
| This folder, on the cluster | `<cluster clone>/Passive Features/HPC script/Ih Fit` |
| Phase-0 archives (`<GROUP>/specimen_*/`) | **not in git**; default `/davinci-1/home/ldellamea/Human Neurons Fitting`, knob `IH_ARCHIVE_ROOT` |

**Nothing in the submit scripts needs editing before the first run.** The code
directory is `$PBS_O_WORKDIR` -- wherever you ran `qsub` -- so submit from
this folder and a `git pull` can never leave a job running last week's code.
Everything else is a knob with a default, and the run header prints every
resolved path before any work starts.

The knobs are prefixed `IH_` (`IH_CODE_DIR`, `IH_ARCHIVE_ROOT`,
`IH_OUTPUT_BASE`, `IH_ENV`) because the login shell exports a bare `CODE` and
PBS jobs source `.bashrc`; a bare `CODE`, `ROOT`, `ARCHIVE_ROOT`,
`OUTPUT_BASE` or `CODE_DIR` is reported and ignored. Line endings are already
pinned by the repo's own `.gitattributes` (`*.sh`, `*.py`, `*.mod` -> `eol=lf`),
so a Windows clone cannot ship CRLF to the cluster.

A `qsub -v` list is comma-separated, so a value containing a **space** or a
**comma** cannot travel in it. Export such a knob before `qsub` and pass `-V`,
or keep the path out of `-v` -- which is the other reason the code directory
is taken from `$PBS_O_WORKDIR`: its path contains a space.

## First run on the cluster (login node, conda env `prova`)

    cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/HPC script/Ih Fit"
    nrnivmodl mod                          # once per architecture; creates x86_64/
    python smoke_ih_fit.py                 # expect "smoke_ih_fit: 11/11 passed"
    python smoke_ih_recovery.py            # expect "smoke_ih_recovery: 9/9 passed"
    python regression_passive_identity.py --ref-dir "../Biological Fit" \
        --archive-cell "/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc/specimen_<id>"
                                           # expect "regression_passive_identity: PASS (...)"

Then, before committing walltime to anything (submit FROM this folder):

    qsub -v GROUP=L3_exc,ARM=ih6,DRY_RUN=1 submit_ih_fit.sh
        # checks conda, the nrnivmodl guard, that h.Ih and h.Ih_human load,
        # every resolved path, and the whole CLI contract -- and runs no fit.

## The three arms (plan section 5)

`baseline_runB` normally is NOT re-run; its run-B CSV is read from disk.

| ARM | theta | long steps admitted | RMSD window | gate |
|---|---|---|---|---|
| `baseline_runB` | C_m, R_m, R_a | 2 smallest, 12 mV cap, <= 100 pA | 60 ms after onset | legacy early-window validation |
| `passive_fullstep` | C_m, R_m, R_a | h_2..h_{n-1}, no cap | the whole step | same window both sides |
| `ih6` | + gbar_h, dv_h, kappa_tau | h_2..h_{n-1}, no cap | the whole step | same window both sides |

`passive_fullstep` is not optional: without it, a loss improvement in `ih6`
could be the window and the extra steps rather than I_h.

## Stage 7 -- the gate to the campaign

Nothing in Stage 8 should run before this has passed: until it does, a fitted
`dv_h` or `kappa_tau` is a number the optimiser returned, not a measurement.

Submit from this folder; `MORPH_ROOT` defaults to `$IH_ARCHIVE_ROOT/L3_exc`.

    qsub -v DRY_RUN=1 submit_ih_recovery.sh
    qsub -v MAX_CELLS=2,N_CALLS=20,RUN_TAG=pilot submit_ih_recovery.sh
                                           # shakedown; its verdict is NOT a verdict
    qsub -v NOISE_FROM_RESULTS=/path/to/runB/phase2_results.csv submit_ih_recovery.sh

Read the **exit status**: `0` the gate passed, `2` it failed (and
`gate_verdict.csv` names the axis), `1` the run broke before a verdict.

`NOISE_FROM_RESULTS` is not optional in spirit. Without it the cohort is
generated at the benchmark's 0.05 mV, and a gate that passes on data cleaner
than the campaign's says nothing about the campaign. The run header prints
`*** BENCHMARK DEFAULT, NOT MEASURED ***` when it is missing.

On a failure, D-005 says to FREEZE the failing kinetic knob rather than carry
it: `--fit-params Cm,Rm,Ra,gbar` (drop both) or `Cm,Rm,Ra,gbar,dv_h` (keep the
shift, drop the time-constant scale). A C_m or gbar failure is not fixed by a
smaller spec -- the protocol is what has to change.

Sign convention (pinned by S5): `vshift` and `vshift_minf` move the curves along
v; positive = more activation at a given v. Kalmbach 2018's "-20 mV" shift of
the Kole rates is `vshift = +20` here.
