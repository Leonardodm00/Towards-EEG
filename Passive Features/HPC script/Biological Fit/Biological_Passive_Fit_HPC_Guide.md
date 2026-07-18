# Biological Passive-Fit Pipeline — HPC User Guide

**Scope.** This document covers the four files that run the passive-cable
fit (C<sub>m</sub>, R<sub>m</sub>, R<sub>a</sub>) on **real** human cortical
pyramidal-neuron recordings on the davinci cluster: `run_biological_fit.py`
(orchestrator), `submit_biological_fit.sh` (per-group PBS job),
`submit_all_groups_biological.sh` (fan-out wrapper), and
`smoke_run_biological_fit.py` (NEURON-free unit test). It explains what to
install, how to launch a run, what the pipeline does internally, and — in
detail — every file and column the pipeline writes to disk, so a result can
be interpreted without re-reading the source.

This is the real-data counterpart of the validated synthetic benchmark: it
runs the **same fitting algorithm** (multi-protocol, `relative`-weighted,
time-weighted loss + two-pass auto-τ<sub>w</sub> selection), not a
re-derivation of it. Where this pipeline deliberately differs from the
synthetic driver, it is called out explicitly.

---

## 1. What the pipeline estimates — and what it does not

The fit recovers three passive cable parameters per cell:

| Symbol | Meaning | Bounds (`DEFAULT_*_BOUNDS`) |
|---|---|---|
| C<sub>m</sub> | specific membrane capacitance | 0.3 – 3.0 µF/cm² |
| R<sub>m</sub> | specific membrane resistance | 1 000 – 100 000 Ω·cm² |
| R<sub>a</sub> | axial (cytoplasmic) resistivity | 50 – 1 000 Ω·cm |

The NEURON model (`PassiveCell`) inserts **only** the `pas` mechanism —
no active or voltage-gated conductance, and in particular **no I<sub>h</sub>**,
is part of the biological fit (verified directly against the model-build
code: `sec.insert("pas")` is the only mechanism-insertion call in
`PassiveCell`). The search space handed to the Bayesian optimiser is exactly
the three log-parameters above — there is no fourth dimension for any
H-current conductance. I<sub>h</sub> is treated as a *nuisance contaminant*
to be minimized by protocol choice (fitting on the hyperpolarizing target by
default, where I<sub>h</sub> activation is smallest) and by the deflection
guard on long-step training folds, not as something the model represents.

---

## 2. Required files (`CODE_DIR`)

All four `.py` files below must sit in the **same directory** on the
cluster, because `run_biological_fit.py` imports the other three by module
name:

| File | Role | Source |
|---|---|---|
| `run_biological_fit.py` | Orchestrator entrypoint (new) | delivered |
| `passive_fitting_hpc_fixed.py` | Fitter, loader, Phase 2.5, Phase 3 (the "monolith") | your existing file, **unchanged** |
| `cm_profile_sweep.py` | Two-pass auto-τ<sub>w</sub> sweep | synthetic-folder module, reused as-is |
| `passive_long_step_training.py` | Loss/split patch (`integrate_long_step`, `bundle_rmsd`) | synthetic-folder module, reused as-is |

Every top-level `import` in all three dependency files was traced
(including lazily-imported ones inside functions) — there is no fifth
file. The two commented-out `# from phase1_data_loader import (...)` lines
inside the monolith are dead code from an earlier refactor and can be
ignored; `AllenSDK` is imported in a `try/except` and is **not required**
for the biological fit.

**Environment.** A conda env with `numpy`, `pandas`, `scipy`,
`scikit-optimize` (`skopt`), `matplotlib`, and NEURON (with its Python
bindings). **No `nrnivmodl` step is needed** — the biological model is
passive-only, so `Ih.mod` (which exists solely to *generate* the synthetic
benchmark's sag) never needs to be compiled here.

---

## 3. Expected data layout

```
<ARCHIVE_ROOT>/
├── L2_exc/
│   ├── specimen_<id>/
│   ├── specimen_<id>/
│   └── ...
├── L2_inh/
├── L3_exc/
└── ...
```

One call to `run_biological_fit.py --archive-dir <ARCHIVE_ROOT>/<GROUP>`
processes one **group** (one layer × dendrite-type folder) and loads every
`specimen_*/` inside it via `load_cells_from_archive`
(`specimen_ids=None`). The group label used throughout the outputs (Phase
2.5's `group_label`, job names, output subfolder) is simply
`Path(archive_dir).name`, e.g. `L3_exc`.

---

## 4. Quickstart — validate before you commit compute

Because the smoke test only covers the NEURON-free pure helpers (argument
parsing, τ<sub>w</sub> selection, subset selection, the gate arithmetic,
result serialization — see §10), run a **1-cell dry run** first to validate
the NEURON-side path (model build, sweep, fit, gate, Phase 2.5) on real
hardware before launching a full group:

1. Edit `submit_biological_fit.sh`: set `MAX_CELLS=1` and
   `PHASE3_SUBSET="none"`.
2. `qsub -v GROUP=<one_group> submit_biological_fit.sh`
3. Confirm `phase2_results.csv` has one row with a finite `cm_uF_per_cm2`,
   and that `validation_status` is not blank.
4. Revert `MAX_CELLS` / `PHASE3_SUBSET` to their production values and
   submit the full run (§5–6).

This takes minutes rather than the hours a full group takes, and it is the
only way to catch archive-path or environment problems before they cost
real wall-time.

---

## 5. Running one group

```bash
qsub -v GROUP=L3_exc submit_biological_fit.sh
```

`submit_biological_fit.sh` is a single-node PBS job (`select=1:ncpus=1`,
`walltime=100:00:00` — the fit is sequential, one cell at a time, because
NEURON's global section list is reset between cells and the monolith's
worker count is fixed at 1). It:

1. resolves `ARCHIVE_DIR = $ARCHIVE_ROOT/$GROUP` and
   `OUTPUT_DIR = $OUTPUT_ROOT/$GROUP`, failing loudly if `$GROUP` is unset
   or the archive directory does not exist;
2. activates the conda environment (`conda activate "$CONDA_ENV"` — **not**
   `module load python`, which produced Lmod errors on this cluster);
3. builds the full CLI argument list from the **USER CONFIG** block at the
   top of the file (edit this block, not the argument-assembly logic below
   it) and calls `run_biological_fit.py`.

All configuration lives in that USER CONFIG block, organized into four
groups that mirror the pipeline stages: data/fit, two-pass auto-τ<sub>w</sub>
+ loss, Phase 2.5, Phase 3. See §9 for the full flag reference.

`GROUP`, `F_FACTOR`, `SKIP_PHASE2P5`, `N_FLOOR`, `N_RA_PROFILE`, and
`PHASE3_SUBSET` are also overridable at submission time via `qsub -v`
without editing the file — this is how the fan-out wrapper (§6) drives
per-group values.

---

## 6. Running all groups

```bash
./submit_all_groups_biological.sh                 # every group in DEFAULT_GROUPS
./submit_all_groups_biological.sh L3_exc L5_exc    # only the named groups
```

Submits **one PBS job per group**, each pinned to its own node. Two things
are configured once, at the top of this wrapper, for the whole sweep:

- **`F_PER_GROUP`** — the per-group spine-area correction factor
  (Eyal-style scaling), keyed by group name, with `F_FACTOR_DEFAULT=1.9`
  as a fallback (with a printed warning) for any group not explicitly
  listed. Excitatory L2/L3 default to 1.9, L4/L5/L6 excitatory to 1.5,
  and all inhibitory groups to 1 (no spine correction) — edit
  `F_PER_GROUP` if your own literature values differ.
- **`SKIP_PHASE2P5` / `N_FLOOR` / `N_RA_PROFILE` / `PHASE3_SUBSET`** —
  methodological constants forwarded identically to every job via
  `qsub -v`, so you set the Phase 2.5 / Phase 3 policy for the whole
  sweep in one place rather than per group.

The wrapper skips (with a printed message, not an error) any group whose
archive directory doesn't exist, and reports a final `Submitted: N
Skipped: M` count.

---

## 7. What the pipeline does internally

For each group, `run_biological_fit.py` runs five stages, logged with
timestamps (`[HH:MM:SS | +MM:SS]`) so `tail -f` on the PBS `.o` file shows
live progress:

### [1/5] Patch the loss (once, group-wide)

`passive_long_step_training.integrate_long_step(mono, ...)` monkey-patches
`mono.prepare_optimiser_inputs` and `mono._build_loss_function` in place,
installing:

- the **multi-protocol** training set: the smallest-|amplitude|
  hyperpolarizing long-square steps (`n_long_train`, default 2) are folded
  into the training bundles alongside the short square-subthreshold pulse,
  subject to an I<sub>h</sub> deflection guard (`ls_deflection_cap`, default
  12 mV) that rejects long-step folds whose deflection would let residual
  sag leak into the passive fit;
- the **`relative`-weighted, time-weighted loss** (`weighting="relative"`,
  `ss_time_weight="exp"` by default): short-pulse residuals are weighted by
  an exponential time-decay profile and normalized per bundle by its own
  deflection, so bundles of very different amplitude contribute
  comparably to the total loss rather than the largest deflection
  dominating.

This split (train vs. held-out validation bundles) is τ<sub>w</sub>-independent,
so it is applied once per group, before any per-cell loop.

### [2/5] Load

`mono.load_cells_from_archive(archive_dir, specimen_ids=None, ...)` reads
every `specimen_*/` folder in the group, and
`mono.prepare_optimiser_inputs(cd, fit_target=...)` builds each cell's
training/validation bundle split, search-space bounds, and the Allen
reference scalars (R<sub>in</sub>, τ<sub>m</sub>, V<sub>rest</sub>) used later for
sanity-checking.

### [3/5] Two-pass auto-τ<sub>w</sub> fit + absolute-mV gate (per cell)

For each cell, sequentially:

1. **Sweep** — `cm_profile_sweep.sweep_tau_w_per_cell` profiles the
   training loss over the τ<sub>w</sub> grid (`--tau-w-grid-ms`, default a
   single point `5.0`) × a log-C<sub>m</sub> grid (`--sweep-n-grid`, default
   15 points), and `pick_winning_tau_w` selects the τ<sub>w</sub><sup>\*</sup>
   whose C<sub>m</sub> profile is sharpest (smallest finite `HW_rho`, the
   half-width of the profile-likelihood ridge). With the validated
   single-point default this selection is trivial; a multi-point grid
   (e.g. `"3,5,7"`) makes it a real per-cell choice.
2. **Re-patch** — the loss is re-patched at τ<sub>w</sub><sup>\*</sup> and the
   orchestrator **verifies** (`_verify_tau_w_applied`) that the patched
   loss numerically matches a fresh reference loss built at
   τ<sub>w</sub><sup>\*</sup> at an off-optimum probe point, raising loudly if
   not — this guards against a silent double-patch bug rather than
   letting a wrong τ<sub>w</sub> propagate into the fit undetected.
3. **Fit** — `mono.fit_one_cell(cell, cd, oi, F=..., n_calls=..., ...)` runs
   the Bayesian (Gaussian-process) optimisation over
   (log C<sub>m</sub>, log R<sub>m</sub>, log R<sub>a</sub>).
4. **Absolute-mV gate** — see §8 below.

A dead/non-simulating cell is caught immediately via `_assert_loss_live`,
which evaluates the loss at three C<sub>m</sub> values and raises if it is
constant (±10⁻⁹) — a flat loss means the simulation silently failed, not
that C<sub>m</sub> genuinely doesn't matter.

Per-cell errors are **logged and the group continues** by default (pass
`--fail-fast` to abort the whole group on the first error instead — not
recommended for production, since one bad archive shouldn't cost the rest
of the group's compute).

### [4/5] Phase 2.5 — fix R<sub>a</sub> at the cohort median

`mono.run_phase2p5_for_group(...)` (on by default; `--skip-phase2p5` to
disable) profiles R<sub>a</sub> for every cell with a finite Phase-2 fit,
takes the **cohort median** of the per-cell R<sub>a</sub> argmins (or falls
back to a literature R<sub>a</sub> if fewer than `--n-floor` cells qualify),
fixes R<sub>a</sub> at that single group-wide value, and **refits**
C<sub>m</sub>, R<sub>m</sub> for every cell at that fixed R<sub>a</sub>. This
mutates each result **in place**: `cm_uF_per_cm2`, `rm_Ohm_cm2`,
`ra_Ohm_cm`, `train_rmsd_mV` are overwritten with the Phase-2.5 values, and
the pre-2.5 values are stashed on `cm_phase2`, `rm_phase2`, `ra_phase2`,
`train_rmsd_phase2` so both remain inspectable. Phase 2.5's pass-1 filter
only requires a finite Phase-2 (C<sub>m</sub>, R<sub>m</sub>) — it is
**independent of `validation_status`**, so the absolute-mV gate (§8, which
overwrites `validation_status`) does not affect which cells enter Phase
2.5; it only affects which cells are later eligible for Phase 3.

### [5/5] Phase 3 — bootstrap confidence intervals, on a subset

`select_phase3_subset(fittable_ids, spec)` resolves which cells get
bootstrapped (`--phase3-subset`, default `frac:0.5` — half the *fittable*
cells, i.e. those with `validation_status` in `{good, to_refine}` and a
retained GP result; grammar: `none | all | first:N | frac:F`, deterministic
by specimen-ID sort). For each selected cell,
`mono.phase3_full_for_cell(...)` runs the bootstrap (nonparametric by
default — resampling your real recorded pulses; `--bootstrap-mode
parametric` uses simulated noise instead) plus a GP diagnostic, and
`mono.save_replot_bundle(...)` persists everything needed to replot the
cell later without re-fitting.

---

## 8. The absolute-mV validation gate — what it is and why it exists

**The problem.** Under the patched `relative` loss, `fit_one_cell` returns
`train_rmsd_mV = result.fun`, which — despite the name — is the
**unitless** relative loss (a deflection-normalized quantity), while the
held-out validation RMSD computed by
`mono._rmsd_for_validation_bundle` stays in **absolute mV**. Feeding both
into `valid_to_train_ratio` mixes units, and the monolith's mV-calibrated
thresholds misfire — this is the documented root cause of the synthetic
benchmark report's "almost every cell reads good" caveat (report §§8.1–8.2).
On synthetic data there was a ground-truth recovery ratio to fall back on
for sanity-checking; on real recordings there is none, so a working gate
matters more, not less.

**The fix.** `_apply_absolute_gate` recomputes **both** sides in absolute
mV at the fitted point θ̂ = (Ĉ<sub>m</sub>, R̂<sub>m</sub>, R̂<sub>a</sub>), then
re-classifies with the monolith's own already-calibrated thresholds — it
does not invent new fractional thresholds (that would require the
synthetic benchmark's `phase2_results.csv`, which was not available; if
you'd like the fractional variant added once your τ<sub>w</sub> = 3/7
benchmark produces that distribution, that's a small follow-up).

Formally, with `oi.train_bundles` the B<sub>train</sub> training bundles and
`oi.validation_bundles` the B<sub>valid</sub> held-out bundles, both simulated
at θ̂ with `bundle_rmsd(cell, bundle, v_rest_mV; ss_window_ms, ls_window_ms_after_onset, r_in_target, ss_sample_weight_fn=None)`:

$$
\text{train\_rmsd\_abs\_mV}(\hat\theta) \;=\; \frac{1}{B_{\text{train}}}\sum_{b=1}^{B_{\text{train}}} \text{RMSD}_{\text{train},b}(\hat\theta)
\qquad
\text{valid\_rmsd\_abs\_mV}(\hat\theta) \;=\; \frac{1}{B_{\text{valid}}}\sum_{b=1}^{B_{\text{valid}}} \text{RMSD}_{\text{valid},b}(\hat\theta)
$$

(non-finite per-bundle RMSDs are dropped from their respective mean, not
counted as zero), and

$$
\text{valid\_to\_train\_ratio\_abs}(\hat\theta) \;=\; \frac{\text{valid\_rmsd\_abs\_mV}(\hat\theta)}{\max\!\big(\text{train\_rmsd\_abs\_mV}(\hat\theta),\, 10^{-9}\big)}.
$$

Note the training-RMSD here is computed **unweighted**
(`ss_sample_weight_fn=None`), matching how the validation RMSD is computed,
so train and valid are on the same footing — this is deliberately *not*
the same training loss the optimiser minimized (which is the time-weighted
`relative` loss), because that loss is what's unitless in the first place.

The status is then `mono._classify_fit(train_rmsd_abs_mV, valid_rmsd_abs_mV,
k_good, k_fail, train_fail_mV, valid_rmsd_good_mV=valid_good_mV)`, applied
**in this exact order** (verbatim from the monolith, re-verified against
source for this document):

1. if `train_rmsd_abs_mV` is non-finite or `> train_fail_mV` (default 2.0
   mV) → **`failed`** — training itself was poor; validation is not even
   consulted.
2. else if `valid_rmsd_abs_mV` is non-finite → **`failed`**.
3. else if `valid_rmsd_abs_mV ≤ valid_good_mV` (default 0.2 mV) →
   **`good`**, regardless of the ratio — an escape hatch for the regime
   where training fits extremely tightly and the ratio would otherwise be
   dominated by floor-level noise rather than real model error.
4. else if `valid_to_train_ratio_abs ≤ k_good` (default 3.0) → **`good`**.
5. else if `valid_to_train_ratio_abs ≤ k_fail` (default 10.0) →
   **`to_refine`**.
6. else → **`failed`**.

`_apply_absolute_gate` then **mutates** the result object: it stashes the
fitter's own relative-loss verdict on `validation_status_relative` /
`train_rel_loss` (so it remains inspectable), writes
`train_rmsd_abs_mV`, `valid_rmsd_abs_mV`, `valid_to_train_ratio_abs`, and
**overwrites `validation_status`** with the absolute-mV verdict — this is
the status Phase 3's eligibility filter (§7, stage 5) consumes. One
documented limitation: this gate reflects the **Phase-2** (free-R<sub>a</sub>)
fit; Phase 2.5 has no validation pass of its own and does not recompute it,
so after Phase 2.5 the gate's status still describes the pre-2.5 fit while
`train_rmsd_mV`/`cm_uF_per_cm2`/etc. have been overwritten with the
post-2.5 values.

---

## 9. Full CLI / configuration reference

All flags are set in the **USER CONFIG** block of `submit_biological_fit.sh`
(shell variable → CLI flag mapping shown). Defaults below are the
as-shipped values, matching the validated synthetic benchmark configuration
plus the four project decisions (τ<sub>w</sub>=5.0 default, Phase 3 on half
the cells, nonparametric bootstrap default, Phase 2.5 on).

**Data / fit**

| Shell var | CLI flag | Default | Meaning |
|---|---|---|---|
| — | `--archive-dir` | *(required)* | `<ARCHIVE_ROOT>/<GROUP>` |
| — | `--output-dir` | *(required)* | `<OUTPUT_ROOT>/<GROUP>` |
| — | `--code-dir` | *(required)* | dir with the 4 `.py` files |
| `N_AVG_GROUPS` | `--n-avg-groups` | 1 | sweep-average groups per polarity |
| `FIT_TARGET` | `--fit-target` | `hyp` | `dep`\|`hyp`\|`both` (hyp minimises I<sub>h</sub>) |
| `F_FACTOR` | `--F` | 1.9 | spine-area correction |
| `N_CALLS` / `N_INITIAL` | `--n-calls` / `--n-initial` | 100 / 50 | GP optimiser budget per cell |
| `MAX_CELLS` | `--max-cells` | *(all)* | cap cells — use `1` for the dry run (§4) |

**Two-pass auto-τ<sub>w</sub> + multi-protocol loss**

| Shell var | CLI flag | Default | Meaning |
|---|---|---|---|
| `N_LONG_TRAIN` | `--n-long-train` | 2 | smallest-\|amp\| hyp long steps folded into training |
| `LS_DEFLECTION_CAP_MV` | `--ls-deflection-cap` | 12.0 | I<sub>h</sub> deflection guard (mV) for long-step admission |
| `MAX_SAG_AMPLITUDE_MV` | `--max-sag-amplitude-mV` | −1 (disabled) | optional sag-gated guard; needs `sag_ratio` on `CellData` (not currently carried by the loader — falls back to the deflection cap) |
| `R_IN_TARGET` | `--r-in-target` | `peak` | `peak` (pre-sag) \| `steady` (sagged) R<sub>in</sub> |
| `WEIGHTING` | `--weighting` | `relative` | cross-bundle loss weighting |
| `SS_WINDOW_MS` | `--ss-window-ms` | `0.5,100.0` | short-pulse RMSD window (start = pulse offset) |
| `SS_T0_MS` | `--ss-t0-ms` | *(= SS window start)* | override the time-weight's t₀ |
| `SS_TIME_WEIGHT` | `--ss-time-weight` | `exp` | `exp`\|`gauss`\|`none` |
| `LS_WINDOW_MS` | `--ls-window-ms` | 150.0 | long-step RMSD window length from onset |
| `TAU_W_GRID_MS` | `--tau-w-grid-ms` | `5.0` | per-cell τ<sub>w</sub> grid; winner = sharpest `HW_rho` |
| `SWEEP_RHO` | `--sweep-rho` | 0.5 | relative-rise threshold defining `HW_rho` |
| `SWEEP_N_GRID` | `--sweep-n-grid` | 15 | log-C<sub>m</sub> grid points for the profile |

**Phase 2.5**

| Shell var | CLI flag | Default | Meaning |
|---|---|---|---|
| `SKIP_PHASE2P5` | `--skip-phase2p5` | off (Phase 2.5 **on**) | `1` = legacy free-R<sub>a</sub> diagnostic mode |
| `N_FLOOR` | `--n-floor` | 4 | min qualifying cells for cohort-median R<sub>a</sub>; below this, literature-R<sub>a</sub> fallback |
| `N_RA_PROFILE` | `--n-ra-profile` | 50 | R<sub>a</sub> grid points for the RMSD-vs-R<sub>a</sub> profile |

**Phase 3**

| Shell var | CLI flag | Default | Meaning |
|---|---|---|---|
| `PHASE3_SUBSET` | `--phase3-subset` | `frac:0.5` | `none`\|`all`\|`first:N`\|`frac:F` |
| `BOOTSTRAP_B` | `--bootstrap-B` | 200 | bootstrap replicates |
| `BOOTSTRAP_MODE` | `--bootstrap-mode` | `nonparametric` | `nonparametric`\|`parametric` |
| `NOISE_MODE` | `--noise-mode` | `block` | `iid`\|`ar1`\|`block` (parametric only) |
| `BOOTSTRAP_N_CALLS` / `BOOTSTRAP_N_INITIAL` | `--bootstrap-n-calls` / `--bootstrap-n-initial` | 60 / 20 | GP budget per bootstrap replicate |

**Robustness**

| Shell var | CLI flag | Default | Meaning |
|---|---|---|---|
| — | `--fail-fast` | off | abort the group on the first per-cell error instead of logging and continuing |

---

## 10. Outputs produced

Directory tree for one group, `<OUTPUT_DIR>/<GROUP>/`, with the stage that
writes each item:

```
<OUTPUT_DIR>/<GROUP>/
├── tau_w_choice.csv                 [3/5, orchestrator]
├── phase2_results.csv               [3/5, orchestrator]
├── failed_cells.txt                 [3/5, orchestrator — only if any cell failed]
├── phase2p5_results.csv             [4/5, monolith — before/after table]
├── phase2p5_profiles.npz            [4/5, monolith — raw R_a-profile arrays]
├── phase2p5_profiles/
│   └── <specimen_id>.png            [4/5, monolith — one plot per profiled cell]
├── phase2p5_combined_results.csv    [4/5, orchestrator — full result table, post-2.5]
├── phase3_full_summary.csv          [5/5, orchestrator — CI summary, Phase-3 subset only]
└── cell_<specimen_id>/              [5/5, monolith — one per Phase-3 cell]
    ├── bootstrap/                   *.pkl, *.npy, *.csv, histogram_*.png, pairwise_*.png
    ├── gp_diagnostic/                *.pkl, profile_*.png
    └── replot/
        ├── core.pkl                  bootstrap + GP diagnostic + fit summary + traces
        ├── pulse_pool.pkl            (optional) the real pulses used for nonparametric resampling
        └── meta.json                 human-readable index of shapes/versions
```

### `tau_w_choice.csv` — one row per cell

| Column | Meaning |
|---|---|
| `specimen_id` | cell identifier |
| `tau_w_chosen_ms` | the winning τ<sub>w</sub><sup>\*</sup> |
| `reason` | `"sharpest_hw_rho"` or `"no_finite_hw_rho->fallback_mid"` (degenerate sweep — treat as not-identified, not a silent default) |
| `hw_rho` | half-width of the winning C<sub>m</sub> profile-likelihood ridge |
| `kappa` | the winning profile's curvature diagnostic |

### `phase2_results.csv` — one row per cell, Phase-2 (free-R<sub>a</sub>) fit

Core fit columns: `specimen_id, layer, dendrite_type, F, fit_target,
cm_uF_per_cm2, rm_Ohm_cm2, ra_Ohm_cm, cm_sigma, rm_sigma, ra_sigma,
train_rmsd_mV, valid_rmsd_mV, valid_to_train_ratio, rin_MOhm_allen,
tau_ms_allen, v_rest_mV, validation_status, n_calls, n_initial,
wall_time_s, noise_sigma_mV, noise_rho_lag1, error_message`.

Added by this driver (see §8 for the gate columns, §7 for τ<sub>w</sub>):
`tau_w_chosen_ms, tau_w_hw_rho, tau_w_reason, validation_status_relative,
train_rel_loss, train_rmsd_abs_mV, valid_rmsd_abs_mV,
valid_to_train_ratio_abs`.

**Read `validation_status` here as the absolute-mV verdict** (§8) — not the
optimiser's own relative-loss verdict, which is preserved separately as
`validation_status_relative` for inspection.

### `phase2p5_results.csv` — one row per profiled cell, before/after table

Written by the monolith's `Phase2p5CellResult` (verified against source):
`specimen_id, layer, dendrite_type, F, ra_fixed_group, ra_source
("cohort_median"|"literature_fallback"), ra_argmin_cell, ra_argmin_status
("ok"|"boundary"|"profile_failed"), ra_in_median_pool, cm_before, cm_after,
d_cm, rm_before, rm_after, d_rm, train_rmsd_before, train_rmsd_after,
d_train_rmsd, cm_rail_after, rm_rail_after, refit_ok, train_rmsd_blewup,
error_message`.

`refit_ok=True` means the 2-D (C<sub>m</sub>, R<sub>m</sub>) refit at fixed
R<sub>a</sub> succeeded (finite, no blow-up) — it is **not** a
good/to_refine/failed validation verdict, since Phase 2.5 is training-only
and has no held-out data to validate against; that verdict lives in
`validation_status` (Phase 2's absolute-mV gate, carried through). Watch
`train_rmsd_blewup`: `True` flags a cell whose training RMSD materially
worsened once R<sub>a</sub> was fixed at the cohort value — a falsification
signal that the cell's *data*, not just its optimizer search, may be
unreliable at the shared R<sub>a</sub>.

### `phase2p5_combined_results.csv`

The full result table (same columns as `phase2_results.csv`, via the same
`results_to_dataframe`) re-serialized **after** Phase 2.5 has mutated the
results in place — so `cm_uF_per_cm2`/`rm_Ohm_cm2`/`ra_Ohm_cm`/`train_rmsd_mV`
here are the Phase-2.5 (fixed-R<sub>a</sub>) values, with the pre-2.5 values
recoverable from `cm_phase2, rm_phase2, ra_phase2, train_rmsd_phase2`.

### `phase3_full_summary.csv` — one row per (cell × parameter), Phase-3 subset only

| Column | Meaning |
|---|---|
| `specimen_id` | cell identifier |
| `parameter` | `Cm`\|`Rm`\|`Ra` |
| `mle` | bootstrap point estimate (from `bootstrap.mle_physical`) |
| `ci_bca_lo`, `ci_bca_hi` | 95% bias-corrected-and-accelerated (BCa) CI |
| `ci_perc_lo`, `ci_perc_hi` | 95% percentile CI |
| `ci_norm_lo`, `ci_norm_hi` | 95% normal-approximation CI |
| `n_kept` | bootstrap replicates retained after rejection filtering |
| `mode` | `nonparametric`\|`parametric` |

BCa is generally the recommended interval of the three for the same reason
it is standard practice in the bootstrap literature: it corrects for both
bias and skewness in the bootstrap distribution, which the percentile and
normal-approximation intervals do not.

### `cell_<specimen_id>/` — full Phase-3 artefact bundle

Written by the monolith's own `bootstrap_ci_for_cell`,
`gp_diagnostic_for_cell` (both called from `phase3_full_for_cell`), and
`save_replot_bundle`. `bootstrap/` and `gp_diagnostic/` hold the raw
pickled bootstrap/GP objects, arrays, and diagnostic plots (histograms,
pairwise scatter, profile curves). `replot/core.pkl` is the single
self-contained artefact needed to regenerate any Phase-3 plot later without
re-fitting or re-simulating; `replot/pulse_pool.pkl` additionally preserves
the real pulses used for nonparametric resampling, if available.

### `failed_cells.txt`

One `specimen_id` per line, written only if at least one cell's Phase-2 fit
raised an exception (loading, sweep, fit, or gate). Check this file first
if `phase2_results.csv` has fewer rows than the archive has `specimen_*/`
folders.

---

## 11. Validation record

Everything below was verified before delivery, not assumed:

- **Smoke test** (`python smoke_run_biological_fit.py`) — 5 test groups, 2
  independent controls each, covering CLI parsing, τ<sub>w</sub> selection
  (sharpest / single-point / all-`inf` fallback), the `none|all|first:N|
  frac:F` subset grammar, the absolute-gate arithmetic (mean/ratio/NaN-drop/
  short-circuit/ceiling), and result serialization (base + new + Phase-2.5
  columns; absent attributes → NaN/`''`, no crash). **All 5 pass.**
- **ASCII byte-scan** — all four delivered files are pure ASCII (per the
  HPC-compatibility requirement for this cluster; verified byte-by-byte,
  not just "should be").
- **Syntax** — `py_compile` clean on both `.py` files; `bash -n` clean on
  both `.sh` files.
- **Import surface** — `run_biological_fit.py`'s top-level imports are
  confirmed NEURON-free (`argparse, datetime, numpy, pandas, pathlib, sys,
  time, typing` only); the heavy imports (`neuron`, the monolith, the
  sweep, the loss patch) are inside `main()`, so the smoke test never needs
  NEURON.
- **CLI defaults** — re-parsed and printed to confirm they match the
  validated benchmark configuration plus your four decisions (τ<sub>w</sub>
  = `5.0`, Phase 2.5 on, `phase3_subset = frac:0.5`, bootstrap =
  `nonparametric`).
- **API signatures** — every call the orchestrator makes into the three
  dependency modules (`sweep_tau_w_per_cell`, `build_relative_loss_for_tau_w`,
  `CellSweepInput`, `integrate_long_step`, `bundle_rmsd`, `_classify_fit`,
  `_rmsd_for_validation_bundle`, `run_phase2p5_for_group`,
  `phase3_full_for_cell`, `save_replot_bundle`) was checked against the
  actual function signatures in source, not written from memory — including
  the exact `_classify_fit` branch order reproduced in §8.
- **Negative-valued flag** — `--max-sag-amplitude-mV` is passed as a single
  `=`-joined token in the submit script so argparse cannot mistake a
  leading `-1` for an unrecognised option; both `-1` and `8.0` were parsed
  and checked.

---

## 12. Known caveats / tunables to revisit

- **`N_FLOOR=4`**: any group with fewer than 4 qualifying cells falls back
  to a per-(layer, type) literature R<sub>a</sub> instead of a cohort
  median. Lower it if you trust small-group medians more than the
  literature fallback for your groups; raise it to be more conservative.
- **Sag-gated I<sub>h</sub> guard is currently inert**: `--max-sag-amplitude-mV`
  defaults to disabled (`-1`) because `sag_ratio` is not carried on
  `CellData` by the current loader — enabling it needs that field added to
  the Phase-0 metadata and loader first. Until then, the fixed-mV
  deflection cap (`LS_DEFLECTION_CAP_MV`) is the only I<sub>h</sub> guard in
  effect.
- **Optimiser budget**: `n_calls=100 / n_initial=50` matches the validated
  benchmark; for a production biological run you may want to raise this
  (e.g. 200/120) for a more thorough per-cell search, at proportionally
  higher wall-time cost.
- **The absolute-mV gate reflects the Phase-2 fit only** (§8) — Phase 2.5
  has no validation pass of its own, so a cell's `validation_status` after
  Phase 2.5 still describes the pre-2.5 fit quality even though the
  parameter values it sits next to are post-2.5.

---

## 13. Sourcing note

Every mechanical claim in this document — file lists, CLI defaults, output
filenames and columns, the `_classify_fit` branch order, function
signatures — was verified directly against the repository source (the
monolith `passive_fitting_hpc_fixed.py` and the two synthetic-folder
wrapper modules) and against the delivered files themselves, not stated
from memory. No PubMed or bioRxiv/medRxiv literature check was needed for
this document, since it describes an implementation, not a scientific
claim requiring external corroboration; the biophysical rationale
referenced in passing (I<sub>h</sub>/sag, the BCa interval's bias-correction
property) is carried over from the project's own prior documentation
(`phase1_conceptual_guide.md`, `phase2_patch.md`) and standard bootstrap
methodology, respectively.
