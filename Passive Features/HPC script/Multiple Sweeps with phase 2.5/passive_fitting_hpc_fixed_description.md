# `passive_fitting_hpc_fixed.py` — Technical Reference

## Overview

`passive_fitting_hpc_fixed.py` is a self-contained, HPC-adapted pipeline for estimating the passive biophysical parameters — specific membrane capacitance $C_m$ (µF/cm²), specific membrane resistance $R_m$ (Ω·cm²), and axial resistance $R_a$ (Ω·cm) — of human cortical pyramidal neurons and interneurons from Allen Institute Cell Types Database electrophysiology data.

The pipeline runs in four sequential phases per group (one PBS job = one `(layer, cell-type)` cohort), reads pre-downloaded cell archives from disk (no Allen SDK network dependency on the cluster), and writes self-documenting CSV summaries and plots at the end of each phase.

---

## Architecture

The file is a single monolith (~7,500 lines). Its logical sections are:

| Section | Lines (approx.) | Purpose |
|---|---|---|
| HPC patches & archive loader | 1–500 | `load_cell_from_archive`, `load_cells_from_archive` |
| Core data structures | 500–1,700 | `CellData`, `SweepBundle`, `PassiveCell`, `OptimiserInputs` |
| NEURON model | 1,400–1,670 | `PassiveCell`, `build_neuron_model` |
| Phase 2 fitting primitives | 1,700–2,860 | Loss, noise estimator, classifier, `fit_one_cell` |
| Phase 2 batch driver | 2,860–3,370 | `fit_cells`, `aggregate_population` |
| **Phase 2.5** | 3,370–4,100 | `run_phase2p5_for_group` and helpers |
| Phase 3 bootstrap | 4,100–6,100 | `bootstrap_ci_for_cell`, CI helpers, GP diagnostic |
| Plotting | 6,100–7,180 | Profile plots, pairwise scatter, histograms |
| `__main__` | 7,181–7,545 | Argparse, orchestration |

All phases share the **same `PassiveCell` → NEURON → RMSD** compute path and run **strictly sequentially** (one cell at a time). This is deliberate: NEURON's section list is process-global; `PassiveCell.destroy()` is called between cells to prevent section contamination.

---

## Phase 1 — Data Loading

**Entry point:** `load_cells_from_archive(archive_dir, n_avg_groups, max_cells)`

Each `specimen_<id>/` subdirectory in the archive contains:

| File | Contents |
|---|---|
| `metadata.json` | Scalar features: $R_{in}$, $\tau_m$, $V_{rest}$, layer, dendrite type, LJP correction, LS sweep metadata |
| `ss_pulses.npz` | All Square Subthreshold individual pulses (stacked or variable-length format) |
| `ls_sweeps.npz` | Long Square voltage and current traces, one per sweep index |
| `reconstruction.swc` | Morphological reconstruction |

**Loading steps:**

1. Parse `metadata.json` for scalar features and LS sweep metadata.
2. Load all Square Subthreshold (SS) individual pulses from `ss_pulses.npz`. Each pulse is stored as a dict `{t, v, i, polarity, peak_pA, stim_duration_s, sampling_rate_Hz}`.
3. Partition the SS pulses by polarity (`dep`/`hyp`) into `n_avg_groups` equal groups. Average voltage and current within each group to produce `SweepBundle` objects. This produces $2 \times n\_avg\_groups$ training bundles (depolarising + hyperpolarising), and retains all individual pulses in `ss_individual_pulses` for Phase 3's nonparametric bootstrap.
4. Load Long Square (LS) sweeps. Filter to hyperpolarising steps with $|\text{amplitude}| \leq 100\ \text{pA}$ (fallback: 300 pA). Group by amplitude (rounded to nearest 10 pA), average within each group, and detect stim onset/duration from the averaged current. Each amplitude group becomes one `SweepBundle`.
5. Assemble a `CellData` object, which is the sole input to all downstream phases.

---

## Core Data Structures

### `SweepBundle`
One averaged voltage trace at one stimulus amplitude and polarity. Carries: `t` (time array, s), `v_mV`, `i_pA`, `polarity`, `amplitude_pA`, `stim_onset_s`, `stim_duration_s`, `n_repeats_averaged`, `sampling_rate_Hz`, `stimulus_name`.

### `CellData`
Aggregates all data for one specimen: scalar features ($R_{in}$, $\tau_m$, $V_{rest}$, LJP), the SWC path, the SS and LS `SweepBundle` lists, and the raw `ss_individual_pulses` pool.

### `OptimiserInputs`
Produced by `prepare_optimiser_inputs`. Contains: `train_bundles`, `validation_bundles`, `skopt_dimensions` (the 3-D log-space search space over $(C_m, R_m, R_a)$), and the scalar features needed by the loss function.

### `PassiveFitResult`
One row of Phase 2 output. Stores the best-fit $(C_m, R_m, R_a)$, GP-posterior uncertainties $(\sigma_{Cm}, \sigma_{Rm}, \sigma_{Ra})$, training and validation RMSDs, Allen reference features, validation status, noise statistics, and wall time. Pre-2.5 values are stashed on `cm_phase2`, `rm_phase2`, `ra_phase2`, `train_rmsd_phase2` after Phase 2.5 mutates the object in place.

---

## Phase 2 — Bayesian Optimisation

**Entry point:** `fit_cells(cells_data, opt_inputs, F, n_calls, n_initial)`

### Data split
`prepare_optimiser_inputs` partitions bundles as follows:

- **Training set:** all hyperpolarising SS bundles (default `fit_target="hyp"`). Hyperpolarising pulses are preferred because they deactivate resting $I_h$, minimising contamination of the passive transient by active currents.
- **Validation set:** all depolarising SS bundles + all LS bundles.

### The NEURON model (`PassiveCell`)

The reconstruction SWC is imported via NEURON's `Import3d_SWC_read`. The morphology is processed as follows:

- **Spine area correction:** all dendritic surface areas are multiplied by factor $F$ ($F = 1.9$ for L2/3, following Eyal 2016; $F = 2.0$ for L5/L6 per Rich et al. 2021). Spines on segments within 50 µm of the soma are excluded (proximal cutoff).
- **Axon replacement:** the original axon is replaced by the Hay stub (60 µm unmyelinated initial segment) to avoid numerical issues from long, thin axon reconstructions.
- **Passive mechanism:** `pas` is inserted in all sections. `set_passive(Cm, Rm, Ra)` sets `cm`, `g_pas = 1/Rm`, and `Ra` uniformly. `e_pas` is set to $V_{rest}$.

### Loss function (`_build_loss_function`)

For a candidate $(C_m, R_m, R_a)$ in log-space:

1. Call `set_passive(Cm, Rm, Ra)` and `set_e_pas(V_rest)`.
2. For each training `SweepBundle`, replay the SS pulse via `cell.simulate` (IClamp at soma, $dt = 0.025\ \text{ms}$, 10 ms pre-pad + 100 ms post-pad).
3. Compute the **baseline-subtracted RMSD** over the 1–100 ms post-pulse window (Eyal 2016 convention):

$$\mathcal{L}(C_m, R_m, R_a) = \frac{1}{N_{train}} \sum_{j=1}^{N_{train}} \sqrt{ \frac{1}{|W|} \sum_{t \in W} \left[ (v^{exp}_j(t) - \bar{v}^{exp}_{j,pre}) - (v^{sim}_j(t) - \bar{v}^{sim}_{j,pre}) \right]^2 }$$

where $W = [1, 100]\ \text{ms}$ post-pulse and $\bar{v}_{pre}$ denotes the pre-pulse mean. Baseline subtraction removes any offset between $V_{rest}^{exp}$ and $V_{rest}^{sim}$ so the optimiser fits the **shape** of the transient, not the absolute voltage level.

NEURON crashes during evaluation return $10^6$ (a large but finite penalty) so `gp_minimize` steers away from the offending region without aborting.

### Gaussian-process optimisation

`gp_minimize` (scikit-optimize) is run with:

| Parameter | Default | Notes |
|---|---|---|
| `n_calls` | 150 | Total NEURON simulations |
| `n_initial_points` | 20 | Random Sobol samples before GP takes over |
| `acq_func` | `"gp_hedge"` | Automatically selects EI / PI / LCB |
| Search space | log-uniform | $C_m \in [0.3, 3.0]$, $R_m \in [10^3, 10^5]$, $R_a \in [50, 1000]$ |

The search is conducted in log-space throughout, keeping the surrogate landscape approximately symmetric for positive scale parameters.

### GP-posterior uncertainty

After optimisation, 5,000 points are sampled uniformly from the search space and their loss predicted by the final GP surrogate. Points within $\Delta = 0.1\ \text{mV}$ of the optimum are retained; their log-space standard deviation along each axis gives $(\sigma_{Cm}, \sigma_{Rm}, \sigma_{Ra})$. A large $\sigma_{Ra}$ (≈ 1.0–1.3) signals that $R_a$ is not constrained by the loss, which is the primary diagnostic that motivated Phase 2.5.

### Noise estimation

After the fit, the pre-pulse baseline of each training bundle is analysed:

- **$\sigma_{noise}$:** standard deviation of pre-pulse samples (noise on the averaged trace).
- **$\rho_{lag1}$:** lag-1 autocorrelation coefficient. Values $> 0.5$ trigger a warning that the effective sample size is reduced and that iid-based confidence intervals will be optimistic.

### Validation

The best-fit model is evaluated on the validation set:

- **SS depolarising bundles:** same 1–100 ms RMSD as training.
- **LS bundles:** baseline-subtracted RMSD over the first 50 ms after step onset (early window, where $I_h$ sag has not yet fully activated).
- **Per-bundle RMSDs** are printed, then averaged with equal weight across all validation bundles.

### Classification (`_classify_fit`)

The validation status is assigned by the following ordered rules:

| Condition | Status |
|---|---|
| $\text{train\_RMSD} > 2.0\ \text{mV}$ | `failed` |
| $\text{valid\_RMSD} \leq 0.2\ \text{mV}$ (absolute escape) | `good` |
| $\text{valid\_RMSD} / \text{train\_RMSD} \leq 3$ | `good` |
| $3 < \text{ratio} \leq 10$ | `to_refine` |
| $\text{ratio} > 10$ | `failed` |

The absolute escape at 0.2 mV prevents the classifier from calling a biophysically excellent fit "failed" when the optimiser achieves an extremely tight train RMSD (making the ratio blow up artificially).

### Phase 2 outputs

- `phase2_results.csv`: one row per cell with all `PassiveFitResult` fields.
- Population summary printed to stdout: mean/std/median of $(C_m, R_m, R_a)$ and status counts across cells with `good` or `to_refine` status.

---

## Phase 2.5 — Ra Identification and Correction

**Scientific rationale:** $R_a$ is not identifiable from a somatic subthreshold transient. The RMSD-vs-$R_a$ profile at the fitted $(C_m, R_m)$ is flat by construction, so the Phase 2 optimiser treats $R_a$ as a free noise dimension, causing it to drift to box edges and drag $C_m$ to its ceiling along with it (the two rails are one degeneracy, not two). This is the same finding reported by Eyal et al. (2016) on the identical protocol. Phase 2.5 removes the degeneracy by fixing $R_a$ at a single data-driven value per group, then refitting $(C_m, R_m)$ in 2-D.

**Entry point:** `run_phase2p5_for_group(results, cells_data, opt_inputs, ...)`

Phase 2.5 runs strictly after the entire Phase 2 batch and before Phase 3. It consists of two passes over the group.

### Pass 1 — RMSD-vs-$R_a$ profile

For every cell with a finite Phase 2 fit:

1. Rebuild a fresh `PassiveCell` from the SWC.
2. Evaluate $\mathcal{L}(C_m^*, R_m^*, R_a)$ at $N_{profile}$ log-spaced $R_a$ values in $[50, 1000]\ \Omega\text{·cm}$ (default $N_{profile} = 50$), holding $(C_m, R_m)$ fixed at the Phase 2 best-fit point.
3. Fit a **smoothing spline** (`scipy.interpolate.UnivariateSpline`, degree $k = 4$, smoothing factor $s = n \cdot \sigma^2_{RMSD}$) through $({\log R_a},\, \text{RMSD})$.
4. Locate the argmin of the spline using `scipy.optimize.minimize_scalar` with `method="bounded"` over $[\log(50), \log(1000)]$.
5. Flag the argmin as `"boundary"` if it falls within 2% of the grid endpoints in log-space (via `np.isclose`); otherwise `"ok"`. Flag as `"profile_failed"` if fewer than 5 finite RMSD points exist or if the curve is perfectly flat (zero scatter).
6. Destroy the NEURON cell.

The per-cell argmin is recorded as a **diagnostic only**; it is never used directly for refitting.

For each cell a PNG profile plot is saved (`phase2p5_profiles/<sid>.png`): RMSD grid points, smoothing spline overlay, per-cell argmin (green dashed), and group fixed $R_a$ (red solid).

### Cohort median and literature fallback

Cells with `"ok"` argmins form the **median pool**. The group-fixed $R_a$ is the **interpolated cohort median** of the pool:

$$R_a^{fixed} = \text{median}\bigl\{ \hat{R}_a^{(i)} : i \in \text{ok} \bigr\}$$

If the pool size is below `n_floor` (default 4), the pipeline falls back to a literature value, resolved by `(layer, dendrite_type)`:

| Cell type | $R_a$ (Ω·cm) | Source | Inherited? |
|---|---|---|---|
| Any interneuron (aspiny/sparsely spiny) | 100 | Yao et al. 2021 | No |
| L2/3 excitatory | 250 | Eyal et al. 2016 | No |
| L5 excitatory | 495 | Rich et al. 2021 | No |
| L4 excitatory | 250 | No human data; nearest = L2/3 | **Yes** |
| L6 excitatory | 495 | No human data; nearest = L5 | **Yes** |
| Unknown | 250 | Default | **Yes** |

Inherited values are explicitly flagged in the output (`ra_source = "literature_fallback"`) and in the verbose log. Groups with zero finite fits still receive the literature value and proceed to the refit.

### Pass 2 — 2-D refit at fixed $R_a$

For every profiled cell, $R_a$ is pinned at $R_a^{fixed}$ and a fresh `gp_minimize` is run over $(C_m, R_m)$ alone, reusing the same loss function and the same `n_calls`/`n_initial` budget as Phase 2. The loss wrapper injects `ra_fixed_log = \log R_a^{fixed}` as a constant.

### In-place mutation

On success, the `PassiveFitResult` is mutated in place:

- `cm_uF_per_cm2`, `rm_Ohm_cm2`, `ra_Ohm_cm`, `train_rmsd_mV` → Phase 2.5 values.
- `cm_phase2`, `rm_phase2`, `ra_phase2`, `train_rmsd_phase2` → original Phase 2 values (stashed for auditing).

Phase 3 consumes the mutated result and therefore always operates at the Phase 2.5 corrected point.

### Falsification signal

`train_rmsd_blewup = True` is set when:

$$\text{train\_RMSD}^{2.5} > 3 \times \text{train\_RMSD}^{2} \quad \text{AND} \quad \text{train\_RMSD}^{2.5} > 2.0\ \text{mV}$$

This flags cells whose data are genuinely incompatible with the fixed $R_a$ — a falsifiable signal distinguishing bad-data cells from merely under-constrained search (which Phase 2.5 should recover without RMSD degradation).

### Phase 2.5 outputs

| File | Contents |
|---|---|
| `phase2p5_results.csv` | Before/after table: one row per cell with all Phase 2 and Phase 2.5 parameter values, $\Delta C_m$, $\Delta R_m$, RMSD before/after, boundary/rail flags, `refit_ok`, `train_rmsd_blewup` |
| `phase2p5_profiles.npz` | Raw $R_a$ grids and RMSD arrays per cell, group median argmins |
| `phase2p5_profiles/<sid>.png` | One RMSD-vs-$R_a$ plot per cell |

**`refit_ok`** is `True` iff the 2-D refit returned finite $(C_m, R_m)$, threw no error, and did not trip the blow-up test. It is the only quality flag Phase 2.5 can honestly emit — no validation pass is run here, so no good/to_refine/failed verdict is possible.

### Self-validation structure

The before/after table is its own correctness proof:
- Non-railed cells: $|\Delta C_m|$ and $|\Delta R_m|$ small → fixing an insensitive parameter barely perturbs good fits.
- Railed cells (e.g. $C_m = 3.0$, $R_a = 1000$): large $\Delta C_m$ (off the ceiling), $R_m$ rises to maintain $\tau_m = C_m \cdot R_m$, train RMSD improves or is unchanged.
- Bad-data cells: train RMSD blows up when forced off the rail → `train_rmsd_blewup = True`.

---

## Phase 3 — Bootstrap Confidence Intervals and GP Diagnostic

**Entry point:** `bootstrap_ci_for_cell(fit_result, B, bootstrap_mode, ...)`

### Coupling to Phase 2.5

Because Phase 2.5 fixed $R_a$, bootstrapping all three parameters would re-open the flat direction on every replicate. Phase 3 therefore bootstraps only $(C_m, R_m)$ at the fixed $R_a$:

- `ra_fixed_log = log(fit_result.ra_Ohm_cm)` is passed to every replicate.
- The per-replicate refit (`_refit_from_bundles`) collapses to a 2-D `gp_minimize` over $(C_m, R_m)$, with $R_a$ injected as a constant into the loss.
- The returned $R_a$ for every replicate is identically $R_a^{fixed}$, so the 3-tuple $(C_m, R_m, R_a)$ structure is preserved downstream (Ra becomes a zero-variance, degenerate column).
- If Phase 2.5 was skipped via `--skip-phase2p5`, `fix_ra=False` and Phase 3 reverts to the legacy full 3-D bootstrap.

### Parametric mode

For each replicate $b = 1, \ldots, B$:

1. Simulate the MLE trace at the Phase 2.5 corrected $(C_m, R_m, R_a^{fixed})$.
2. Add synthetic noise to the training bundles. Three noise modes are available:
   - `iid`: Gaussian i.i.d. with $\sigma_{residual}$.
   - `ar1`: AR(1) coloured noise with estimated $(\sigma, \rho_{lag1})$.
   - `block`: block noise preserving empirical residual blocks.
3. Refit $(C_m, R_m)$ at $R_a^{fixed}$ on the synthetic bundles using `gp_minimize` in fast mode (warm-started ball centred on the MLE in 2-D log-space).

### Nonparametric mode

For each replicate $b = 1, \ldots, B$:

1. Draw $n_{pulses}$ individual pulse traces with replacement from `ss_individual_pulses` (the full pool loaded in Phase 1).
2. Separate by polarity, partition into `n_avg_groups` sub-groups, and average within each to form `SweepBundle` objects — exactly mirroring Phase 1's bundle construction.
3. Refit $(C_m, R_m)$ at $R_a^{fixed}$.

**Pulse-pool diversity log:** at the start of each cell's bootstrap, the number of distinct `(polarity, amplitude)` clusters in the pool is reported. Pools with $\leq 5$ clusters trigger a warning that iid-pulse resampling may be optimistic because pulses sharing a protocol step are not independent — in that regime, whole-sweep resampling should be considered.

**RMSD outlier rejection:** bootstrap samples whose refit RMSD exceeds $5 \times \text{RMSD}^{MLE}$ are discarded before CI computation.

### Confidence intervals

Three CI types are computed for each parameter from the $N_{kept}$ bootstrap samples $\{\theta_b\}$:

- **Percentile:** $\bigl[Q_{\alpha/2}(\{\theta_b\}),\; Q_{1-\alpha/2}(\{\theta_b\})\bigr]$.
- **BCa (bias-corrected and accelerated):** accounts for bias in $\hat{\theta}^{MLE}$ and skewness in the bootstrap distribution using $z_0$ and the jackknife acceleration $a$.
- **Normal:** $\hat{\theta}^{MLE} \pm z_{1-\alpha/2} \cdot \hat{\sigma}_{boot}$.

All three use `scipy.stats.norm` for $\Phi$ and $\Phi^{-1}$; no CI math is hand-rolled.

For the fixed $R_a$: the CI is a degenerate point interval $[R_a^{fixed}, R_a^{fixed}]$ for all three types (BCa is ill-defined on a constant array and is bypassed). The 3×3 bootstrap covariance $\Sigma_{\log}$ has a zero $R_a$ row/column; the existing `d[d==0]=1` guard ensures the correlation matrix is computed without division by zero.

### GP diagnostic (`gp_diagnostic_for_cell`)

After the bootstrap, the GP surrogate trained on Phase 2's full $n_{calls}$ evaluations is inspected by profiling each parameter along a 1-D grid while holding the other two at their MLE values. The profile is compared to the bootstrap CI edges: if the GP predicts that the loss rises by more than $k \times \text{RMSD}_{MLE}$ before reaching the CI boundary, the surrogate has under-resolved the loss in that region and `gp_ok = False` is flagged for that parameter.

### Phase 3 outputs

Per cell (in `<output_dir>/cell_<sid>/`):

| File | Contents |
|---|---|
| `bootstrap_results.pkl` | Full `BootstrapCIResult` object (all samples, CI tables, covariance) |
| `bootstrap_histograms.png` | Marginal histograms of bootstrap samples with CI overlays |
| `bootstrap_pairwise.png` | Pairwise scatter plots with 2-D ellipse (from $\Sigma_{\log}$) |
| `gp_diagnostic_*.png` | Per-parameter GP profile with bootstrap CI edges |
| `replot/core.pkl` | MLE traces, fit result, opt_inputs — for post-hoc figure generation |
| `replot/pulse_pool.pkl` | Full pulse pool for post-hoc nonparametric bootstrap replication |

Group-level: `phase3_full_summary.csv` — one row per `(specimen_id, parameter)` with MLE, all three CI types (lo/hi), `n_kept`, `mode`, and `gp_ok`.

---

## CLI Reference

Invoked per group by `submit_passive_fit.sh`, which sets `$GROUP` and passes arguments via `qsub -v`.

```
python3 passive_fitting_hpc_fixed.py \
    --archive-dir   <path>/<GROUP>   \
    --output-dir    <path>/<GROUP>   \
    --n-avg-groups  3                \
    --fit-target    hyp              \
    --F             1.9              \
    --n-calls       100              \
    --n-initial     50               \
    --n-floor       4                \
    --n-ra-profile  50               \
    --bootstrap-B   200              \
    --bootstrap-mode nonparametric   \
    --noise-mode    block            \
    --bootstrap-n-calls   40         \
    --bootstrap-n-initial 20
```

| Flag | Default | Description |
|---|---|---|
| `--archive-dir` | required | `specimen_*/` directory root |
| `--output-dir` | `pipeline_outputs` | Root for all output files |
| `--n-avg-groups` | 1 | SS averaging groups per polarity |
| `--fit-target` | `hyp` | Training polarity: `hyp`, `dep`, or `both` |
| `--F` | 1.9 | Spine-area correction factor |
| `--n-calls` | 150 | GP budget (Phase 2 and Phase 2.5 refit) |
| `--n-initial` | 20 | Random initial points before GP |
| `--max-cells` | (all) | Cap on cells processed |
| `--skip-phase2p5` | off | Diagnostic escape hatch: skip Phase 2.5 and revert Phase 3 to full 3-D bootstrap |
| `--n-floor` | 4 | Min qualifying cells for cohort-median $R_a$; below → literature fallback |
| `--n-ra-profile` | 50 | $R_a$ grid points for the RMSD-vs-$R_a$ profile |
| `--skip-phase3` | off | Skip bootstrap and GP diagnostic entirely |
| `--bootstrap-B` | 200 | Bootstrap replicates |
| `--bootstrap-mode` | `parametric` | `parametric` or `nonparametric` |
| `--noise-mode` | `block` | Parametric noise model: `iid`, `ar1`, or `block` |
| `--bootstrap-n-calls` | 100 | GP budget per bootstrap replicate |
| `--bootstrap-n-initial` | 50 | Initial random points per replicate |

---

## Key Design Decisions and Known Limitations

**Non-identifiability of $R_a$.** The somatic subthreshold transient constrains $\tau_m = C_m \cdot R_m$ but leaves the Cm/Rm split and $R_a$ nearly unconstrained (Eyal 2016, Fig. 1 supplement). Phase 2's free-$R_a$ fit therefore drifts to box edges in those dimensions. Phase 2.5 closes this degeneracy by fixing $R_a$ per group. The GP-posterior $\sigma_{Ra} \approx 1.0$–$1.3$ in log-space (reported in `phase2_results.csv`) is the diagnostic; it signals that the full log-width of the prior is equally plausible.

**Validation failures on high-$R_{in}$ cells.** High-input-resistance cells show pronounced $I_h$ sag in the LS validation traces. A purely passive model structurally cannot reproduce sag, so these cells consistently show elevated validation RMSD ratios and are classified as `to_refine` or `failed` even when the SS training fit is excellent and the $\tau_m$ cross-check passes. Validation status driven by $I_h$ sag is therefore not equivalent to fit failure.

**Autocorrelation and effective sample size.** Pre-stimulus residuals show $\rho_{lag1} = 0.3$–$0.99$ across cells. The pipeline warns when $\rho_{lag1} > 0.5$. In the nonparametric bootstrap, whole-pulse resampling preserves within-trace autocorrelation (the critical concern), but between-pulse dependence within the same protocol step is not fully accounted for. The pulse-pool diversity log provides a coarse effective-$N$ sanity check; pools with $\leq 5$ distinct `(polarity, amplitude)` clusters trigger an explicit warning. Parametric confidence intervals based on iid or AR1 noise models inherit the same caveat.

**Sequential execution.** NEURON's process-global section list makes parallelism unsafe. The pipeline forces `n_workers=1` for both Phase 2 and Phase 3 bootstrap, and calls `cell.destroy()` between cells. This is the only configuration confirmed free of `ReferenceError: can't access a deleted section` and `No soma section found` failures on this cluster.

**SWC parse failures.** Cells with malformed morphologies fail inside `Import3d_SWC_read` before the optimiser is reached (`n_calls=0` in the output). These are data defects, not fit failures. A pre-flight SWC validator (queued as a follow-up) should screen these before submitting jobs.

---

## Output File Summary

| File | Phase | Contents |
|---|---|---|
| `phase2_results.csv` | 2 | Per-cell: $(C_m, R_m, R_a)$, sigmas, RMSDs, status, noise stats, wall time |
| `phase2p5_results.csv` | 2.5 | Per-cell: before/after $(C_m, R_m)$, $R_a^{fixed}$, argmins, `refit_ok`, `train_rmsd_blewup` |
| `phase2p5_profiles.npz` | 2.5 | Raw RMSD-vs-$R_a$ arrays and group median argmins |
| `phase2p5_profiles/<sid>.png` | 2.5 | Per-cell RMSD-vs-$R_a$ profile plot |
| `phase3_full_summary.csv` | 3 | Per `(cell, parameter)`: MLE, percentile/BCa/normal CIs, `n_kept`, `gp_ok` |
| `cell_<sid>/bootstrap_results.pkl` | 3 | Full `BootstrapCIResult` |
| `cell_<sid>/bootstrap_histograms.png` | 3 | Marginal bootstrap histograms |
| `cell_<sid>/bootstrap_pairwise.png` | 3 | Pairwise scatter + covariance ellipse |
| `cell_<sid>/gp_diagnostic_*.png` | 3 | Per-parameter GP profile |
| `cell_<sid>/replot/core.pkl` | 3 | MLE traces + result for post-hoc figures |
| `cell_<sid>/replot/pulse_pool.pkl` | 3 | Full pulse pool for post-hoc bootstrap |
