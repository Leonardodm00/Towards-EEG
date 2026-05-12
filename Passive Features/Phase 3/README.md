# Phase 3 — Bootstrap Confidence Intervals & GP Diagnostic

Post-optimisation uncertainty quantification for passive cable parameters (Cm, Rm, Ra)
fitted to human cortical neurons from the Allen Cell Types Database.

---

## Overview

Phase 3 takes the `PassiveFitResult` produced by Phase 2 (Bayesian GP optimisation) and
computes confidence intervals on the three fitted parameters. It runs two complementary
analyses:

| Output | Method | Purpose |
|--------|--------|---------|
| `bootstrap/` | Parametric residual bootstrap | Publication-quality CIs |
| `gp_diagnostic/` | GP surrogate profile + envelope | Identifiability sanity check |

---

## Requirements

- Phase 2 patch sections (1)–(6) applied to `phase2_optimiser.py`
- `fit_result.neuron_cell`, `.gp_result`, `.opt_inputs` populated (via `fit_one_cell` or
  `rebuild_neuron_cells_for_phase3` after parallel `fit_cells`)
- `scikit-optimize`, `scipy`, `numpy`, `pandas`, `matplotlib`
- NEURON accessible in the session (`from neuron import h` must succeed)

---

## Quick Start

```python
from phase3_profile_likelihood import phase3_full_for_cell

phase3 = phase3_full_for_cell(
    fit_result=results[0],          # PassiveFitResult from Phase 2
    root_dir="phase3_outputs",
    bootstrap_kwargs=dict(
        B=200,                      # bootstrap iterations (1000+ for publication)
        fit_mode="fast",            # "rigorous" for final runs
        noise_mode="iid",           # "ar1" | "block"
        n_workers=1,                # >1 requires phase2_optimiser as importable module
    ),
    gp_kwargs=dict(envelope_k=2.0),
)

# Inspect CIs
b = phase3.bootstrap
for p in ("Cm", "Rm", "Ra"):
    print(f"{p}: MLE={b.mle_physical[('Cm','Rm','Ra').index(p)]:.4g}"
          f"  percentile={b.ci_marginal_percentile[p]}"
          f"  BCa={b.ci_marginal_bca[p]}"
          f"  joint={b.ci_joint_projected[p]}")
```

---

## Entry Points

### `bootstrap_ci_for_cell(**kwargs) → BootstrapCIResult`

Runs the parametric residual bootstrap.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fit_result` | required | `PassiveFitResult` from Phase 2 |
| `root_dir` | required | Output root; writes to `cell_<id>/bootstrap/` |
| `B` | 200 | Number of bootstrap iterations |
| `alpha` | 0.95 | Confidence level |
| `fit_mode` | `"fast"` | `"fast"` (warm-start, n_calls=50) or `"rigorous"` (n_calls=150) |
| `n_calls` | None | Override n_calls (None → fit_mode default) |
| `n_initial` | None | Override n_initial (None → fit_mode default) |
| `noise_mode` | `"iid"` | `"iid"` / `"ar1"` / `"block"` |
| `joint_method` | `"gaussian"` | `"gaussian"` (Σ-ellipsoid) or `"mahalanobis"` (empirical depth) |
| `rmsd_reject_mult` | 5.0 | Reject samples with RMSD > k × MLE_RMSD |
| `n_workers` | 1 | Workers for parallel bootstrap |
| `seed` | 0 | Master RNG seed |

### `gp_diagnostic_for_cell(**kwargs) → GpDiagnosticResult`

Sweeps the GP surrogate across the full search space, draws the μ ± k σ envelope, overlays
bootstrap CIs, and validates the GP at each CI boundary with real NEURON calls.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fit_result` | required | `PassiveFitResult` from Phase 2 |
| `bootstrap_result` | None | If provided, overlays its CIs on the plot |
| `root_dir` | required | Output root; writes to `cell_<id>/gp_diagnostic/` |
| `envelope_k` | 2.0 | Envelope width in GP σ units |
| `n_grid` | 80 | Grid points along each parameter axis |
| `inner_grid_per_axis` | 30 | Inner-optimisation grid for the other two parameters |
| `n_validation_per_bound` | 5 | Real NEURON calls per CI boundary point |

### `phase3_full_for_cell(**kwargs) → Phase3Result`

Convenience wrapper that runs both in sequence and writes all outputs.

```python
phase3_full_for_cell(
    fit_result=fr,
    root_dir="phase3_outputs",
    bootstrap_kwargs={...},   # forwarded to bootstrap_ci_for_cell
    gp_kwargs={...},          # forwarded to gp_diagnostic_for_cell
)
```

---

## Output Files

```
cell_<id>/
├── bootstrap/
│   ├── bootstrap_result.pkl       # BootstrapCIResult dataclass (pickle)
│   ├── bootstrap_samples.npy      # (n_kept, 3) samples, physical units
│   ├── bootstrap_summary.csv      # per-parameter MLE + 3 CI types + correlations
│   ├── histogram_{Cm,Rm,Ra}.png   # density + MLE + percentile / BCa / joint CIs
│   └── pairwise_{Cm_Rm,...}.png   # scatter + joint-CR ellipse
└── gp_diagnostic/
    ├── gp_diagnostic_result.pkl   # GpDiagnosticResult dataclass (pickle)
    └── profile_{Cm,Rm,Ra}.png     # GP envelope + bootstrap CI lines + NEURON dots
```

### `bootstrap_summary.csv` columns

`specimen_id`, `parameter`, `mle`,
`ci_perc_lo`, `ci_perc_hi` (percentile CI),
`ci_bca_lo`, `ci_bca_hi` (BCa CI),
`ci_joint_lo`, `ci_joint_hi` (joint-projected CI),
`sigma_log` (log-space σ),
`corr_CmRm`, `corr_CmRa`, `corr_RmRa` (log-space correlations),
`n_kept`, `B`, `alpha`, `fit_mode`, `noise_mode`

---

## CI Types Explained

**Marginal percentile CI** — α/2 and 1−α/2 quantiles of the 1-D bootstrap distribution.
Coverage is nominal α for each parameter individually.

**Marginal BCa CI** — Bias-corrected accelerated interval (DiCiccio & Efron 1996).
Second-order accurate; corrects for skew and bias in the bootstrap distribution.
Preferred over percentile for small effective sample sizes.

**Joint-projected CI** — Projection of the 3-D joint χ²₃(α) confidence ellipsoid onto each
parameter axis. Always wider than the marginal CI. Appropriate when reporting uncertainty
over all three parameters simultaneously.

---

## Key Design Decisions

**Residual bootstrap, not baseline σ.** The noise σ is estimated from (data − model)
residuals at the MLE, pooling the pre-stimulus baseline and the 1–100 ms training window.
This captures model misspecification in addition to instrument noise, producing honest CIs
even when the passive model fits the training window imperfectly.

**Integer-indexed trace cache.** The MLE model traces are cached before the bootstrap loop
keyed by integer index (not object identity), because opt_inputs and mle_traces are both
pickled and unpickled per task even in sequential mode.

**GP diagnostic is not the CI source.** The GP envelope plot uses the bootstrap CIs as
reference lines, not the Wilks χ² threshold. The GP profile is a visualisation of the loss
landscape shape; the bootstrap is the inferential method.

**Parallel bootstrap requires importable modules.** When `n_workers > 1`, workers are spawned
fresh and cannot see notebook globals. Both `phase2_optimiser` and `phase1_data_loader` must
be importable from PYTHONPATH. In Colab, use `n_workers=1`.

---

## Reproducing the Test-Cell Results (specimen 614635228)

```python
phase3 = phase3_full_for_cell(
    fit_result=results[0],
    root_dir="phase3_outputs",
    bootstrap_kwargs=dict(B=200, fit_mode="fast",
                          n_calls=50, n_initial=5, seed=0),
    gp_kwargs=dict(envelope_k=2.0, seed=0),
)
```

Expected approximate CIs (fast mode, B=200):
- Cm ≈ [0.97, 1.02] µF/cm² (marginal percentile)
- Ra ≈ [116, 129] Ω·cm (marginal percentile)
- Rm: one-sided lower bound only (practically non-identifiable above)

---

## References

Efron B & Tibshirani RJ (1993) *An Introduction to the Bootstrap.*
DiCiccio TJ & Efron B (1996) Bootstrap confidence intervals. *Statistical Science* 11:189–228.
Raue A et al. (2009) Structural and practical identifiability. *Bioinformatics* 25:1923–1929.
Eyal G et al. (2016) Human neocortical neurons. *eLife* 5:e16553.
