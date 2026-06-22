# Synthetic Passive-Fit Benchmark — Usage Guide

A complete reference for running the synthetic benchmark on da Vinci HPC,
from directory setup to reading the output figures.

---

## 1. File inventory

Place every file below in the same directory on the cluster (`CODE_DIR`).
The pipeline imports everything from there; no sub-packages, no `pip install`
beyond what your `prova` conda env already has.

### Pipeline modules (files you already own)

| File | What it does |
|---|---|
| `passive_fitting_hpc_fixed.py` | The monolith: archive loader, Phase 2 fit, Phase 2.5 Ra-fix, Phase 3 bootstrap |
| `synthetic_ground_truth.py` | Synthetic archive generator |
| `passive_long_step_training.py` | `integrate_long_step`, multi-protocol loss, `exp_time_weight` |
| `cm_profile_sweep.py` | Per-cell τ_w profile-likelihood sweep |
| `mod/Ih.mod` | NEURON I_h mechanism (required when I_h is on) |

### New benchmark layer (files delivered in this session)

| File | What it does |
|---|---|
| `synth_gt_grid.py` | Builds the seeded manifest (morphology × random GT × per-cell I_h/noise) |
| `gen_from_manifest.py` | Translates manifest rows into generation configs; drives `synthetic_ground_truth` |
| `run_synth_benchmark.py` | Per-cohort Python entrypoint: generate → two-pass fit → Phase 2.5 → Phase 3 |
| `submit_synth_benchmark.sh` | PBS job wrapper (one job per cohort) |
| `submit_all_cohorts.sh` | Login-node launcher: builds manifest, fans out jobs |
| `aggregate_synth_results.py` | Joins all cohort outputs, writes summary CSV + paper/talk figures |

### Smoke tests (run once before submitting)

| File | What it validates | Needs NEURON? |
|---|---|---|
| `smoke_synth_gt_grid.py` | Manifest determinism, log-uniform draws, cohort-shared Ra, I_h/noise jitter | No |
| `smoke_gen_from_manifest.py` | Row → config mapping, factory choice, per-cell noise | No |
| `smoke_cm_profile_sweep.py` | `profile_cm`/`profile_sharpness` numerics, dead-cell detection | No |
| `smoke_run_synth_benchmark.py` | τ_w winner selection, subset parsing, CLI parsers, result serialisation | No |
| `smoke_aggregate_synth_results.py` | Join/ratio/coverage maths, headless figure rendering | No |

---

## 2. Directory layout on da Vinci

```
$CODE_DIR/
├── passive_fitting_hpc_fixed.py
├── synthetic_ground_truth.py
├── passive_long_step_training.py
├── cm_profile_sweep.py
├── synth_gt_grid.py
├── gen_from_manifest.py
├── run_synth_benchmark.py
├── aggregate_synth_results.py
├── submit_synth_benchmark.sh
├── submit_all_cohorts.sh
├── smoke_*.py
└── mod/
    └── Ih.mod          ← (+ na.mod / kv.mod if you enable active channels)

$MORPH_ROOT/
└── specimen_*/
    └── reconstruction.swc     ← your real morphologies

$CODE_DIR/manifest.csv          ← built once by submit_all_cohorts.sh
$CODE_DIR/synthetic_archive/    ← written by each PBS job
$CODE_DIR/synthetic_out/        ← fit results from each PBS job
$CODE_DIR/summary/              ← final summary CSV + figures
```

---

## 3. Step-by-step: running a full analysis

### Step 3.1 — One-time check on the login node

Verify the three monolith function names that `run_synth_benchmark.py`
calls but that I could not confirm without reading the full 7544-line file:

```bash
cd $CODE_DIR
conda activate prova
python3 -c "
import passive_fitting_hpc_fixed as m
print('phase3_full_for_cell :', hasattr(m, 'phase3_full_for_cell'))
print('save_replot_bundle   :', hasattr(m, 'save_replot_bundle'))
print('DEFAULT_ACQ_FUNC     :', hasattr(m, 'DEFAULT_ACQ_FUNC'))
"
```

All three must print `True`. If `phase3_full_for_cell` is `False`, find the
actual name:

```bash
grep "^def " passive_fitting_hpc_fixed.py | grep -i phase3
```

Then open `run_synth_benchmark.py`, search for `phase3_full_for_cell`, and
replace it with whatever that grep returned — one line change.

### Step 3.2 — Run the five smoke tests

Each takes about one second and requires only numpy/pandas (no NEURON, no
morphologies):

```bash
python smoke_synth_gt_grid.py
python smoke_gen_from_manifest.py
python smoke_cm_profile_sweep.py
python smoke_run_synth_benchmark.py
python smoke_aggregate_synth_results.py
```

All five must end with `ALL CHECKS PASSED`. Any failure prints the exact
contract that is broken; fix before submitting.

### Step 3.3 — Edit the two USER CONFIG blocks

Open `submit_all_cohorts.sh` and fill in the paths and the manifest design:

```bash
CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting/synthetic_benchmark"
MORPH_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting/Morphologies"
MORPH_GLOB="specimen_*/reconstruction.swc"
DRAWS_PER_MORPH=2          # GT draws per morphology
CELLS_PER_COHORT=10        # cells per PBS job / Phase 2.5 unit
```

Open `submit_synth_benchmark.sh` and fill in the same `CODE_DIR`.
Everything else has sensible defaults (see Section 4 for tuning).

### Step 3.4 — Dry run with 4 cells

Before committing to the full cohort fan-out, set `MAX_CELLS=4` in
`submit_all_cohorts.sh` to force a single cohort of 4 cells. Submit it:

```bash
bash submit_all_cohorts.sh
```

This builds `manifest.csv`, submits one PBS job, and writes outputs to
`synthetic_out/cohort_0000/`. Once it finishes, confirm these files exist:

```
synthetic_out/cohort_0000/
├── phase2_results.csv
├── tau_w_choice.csv
├── phase2p5_results.csv
├── phase2p5_combined_results.csv
└── phase3_full_summary.csv      ← only if PHASE3_SUBSET is non-empty
```

If the dry run passes, remove `MAX_CELLS=4` (set to empty string) in
`submit_all_cohorts.sh`.

### Step 3.5 — Full submission

```bash
bash submit_all_cohorts.sh
```

Because `manifest.csv` now exists, the launcher skips manifest generation
and goes straight to fanning out one PBS job per cohort. Watch progress:

```bash
qstat -u $USER
```

### Step 3.6 — Aggregate after all jobs finish

```bash
python3 aggregate_synth_results.py \
    --manifest      $CODE_DIR/manifest.csv \
    --output-root   $CODE_DIR/synthetic_out \
    --archive-root  $CODE_DIR/synthetic_archive \
    --out-dir       $CODE_DIR/summary \
    --code-dir      $CODE_DIR
```

Outputs written to `$CODE_DIR/summary/`:

```
summary/
├── benchmark_summary.csv          ← one row per cell: injected + recovered + ratios
├── phase3_coverage.csv            ← CI coverage for the calibration subset
└── figures/
    ├── paper/                     ← vector PDF + 300 dpi PNG, small fonts
    │   ├── recovery_distributions.pdf/.png
    │   ├── bias_vs_injected.pdf/.png
    │   ├── iso_tau_identifiability.pdf/.png
    │   ├── recovered_vs_injected.pdf/.png
    │   ├── tau_w_selection.pdf/.png
    │   ├── phase2p5_effect.pdf/.png
    │   ├── validation_health.pdf/.png
    │   └── ci_calibration.pdf/.png
    └── talk/                      ← large fonts, PNG only
        └── (same eight figures)
```

---

## 4. Knob-tuning reference

Every knob below lives in one of two places: the USER CONFIG block of
`submit_all_cohorts.sh` (manifest design) or `submit_synth_benchmark.sh`
(fit/sweep/Phase 2.5/3). The tables below show which file each knob lives
in, what it does, and how to choose it.

---

### 4.1 Manifest design (`submit_all_cohorts.sh`)

| Knob | Default | What it controls | How to tune |
|---|---|---|---|
| `DRAWS_PER_MORPH` | 2 | Number of independent random GT triples per morphology | 1 → benchmark is morphology-driven (one GT per cell); ≥3 → separates morphology effects from GT effects. More draws = more total cells = longer total runtime. |
| `CELLS_PER_COHORT` | 10 | Cells per PBS job and per Phase 2.5 unit | Must be comfortably above `N_FLOOR` (default 4). Smaller = shorter per-job walltime; larger = Phase 2.5 has more cells to estimate the cohort Ra from. 8–15 is the practical range. |
| `SEED` | 0 | Master random seed for all GT and noise draws | Change to get an independent replicate of the whole benchmark. |
| `RA_MODE` | `per_cohort` | Whether Ra is shared within a cohort | `per_cohort` (default): Phase 2.5's premise holds; Ra recovery is interpretable. `per_cell`: stress test of Phase 2.5 misspecification — Ra recovery is NOT interpretable but Cm/Rm bias under a wrong Ra is measured. |
| `IH_GIHBAR` | 2e-4 | Nominal I_h maximal conductance [S/cm²] | Match your target cell type. L2/3 human pyramidal: ~1–3×10⁻⁴ S/cm². |
| `IH_GIHBAR_CV` | 0.5 | Cell-to-cell variability of gIhbar (log-normal CV) | 0.3–0.5 covers the biological range observed in L2/3. 0 = no jitter (all cells at the nominal; unrealistic). |
| `NOISE_SIGMA` | 0.05 | Nominal within-sweep noise level [mV] | Match your real recordings. Typical Allen patch-clamp: 0.03–0.08 mV. |
| `NOISE_CV` | 0.3 | Cell-to-cell noise-level variability (log-normal CV) | 0.2–0.4 is realistic. Controls the spread of the noise distribution, not its mean. |
| `USE_IH` | 1 | Inject I_h (1) or pure-passive baseline (0) | Keep 1 for the realistic benchmark. Set 0 to isolate the passive-estimator bias from I_h contamination (useful control experiment). |

---

### 4.2 Fit budget (`submit_synth_benchmark.sh`)

| Knob | Default | What it controls | How to tune |
|---|---|---|---|
| `N_CALLS` | 100 | Bayesian optimisation evaluations per cell (Phase 2 + Phase 2.5 refit) | 100 is the minimum for reliable convergence on real morphologies. 150–200 for publication. Each call = one NEURON simulation. |
| `N_INITIAL` | 50 | Random initial points before the GP takes over | ~40–50% of `N_CALLS`. Fewer → GP starts earlier on a sparser landscape; more → better initial coverage but slower. |
| `F` | 1.9 | Spine-area correction factor | L2/3 spiny: 1.9 (Eyal 2016). L5 spiny: 2.0 (Rich). Aspiny: 1.0. Match your morphologies. |
| `FIT_TARGET` | `hyp` | Which polarity to train on | `hyp` for the standard passive pipeline (hyperpolarising steps activate minimal channels). |

---

### 4.3 Auto-τ_w sweep (`submit_synth_benchmark.sh`)

These are the most important knobs for Cm identifiability.

| Knob | Default | What it controls | How to tune |
|---|---|---|---|
| `TAU_W_GRID_MS` | `2.0,5.0,10.0` | The three τ_w values evaluated per cell | 2 ms emphasises the early capacitive band (best for multi-protocol mode where LS steps already pin τ_m). 10 ms keeps more tail weight (needed in SS-only mode). The winner is picked per cell, so all three are always evaluated — widening the grid costs proportionally more NEURON time. |
| `SWEEP_RHO` | 0.5 | Relative-rise threshold for HW_ρ | The profile half-width is measured where PL rises to (1+ρ)×PL_min. Smaller ρ = more conservative sharpness criterion (only very sharp profiles win). 0.5 is a good default; 0.3–0.7 is the sensible range. |
| `SWEEP_N_GRID` | 41 | Number of Cm grid points for the profile | More points = smoother profile, better-resolved minimum, slower sweep. 41 is the right balance; below 21 the parabolic refinement can miss the true minimum. |
| `SS_WINDOW_MS` | `0.5,100.0` | Fitting window for brief pulses [start, end] in ms | Start = pulse offset (0.5 ms). Moving the start later throws away early-band Cm information. End controls how much slow tail is included. Keep at default unless you have good reason. |
| `N_LONG_TRAIN` | 2 | Number of smallest-amplitude LS steps folded into training | 2 is the standard. 0 = SS-only mode (then set `TAU_W_GRID_MS` to include 10 ms). |
| `LS_DEFLECTION_CAP_MV` | 12.0 | Maximum estimated voltage deflection for a LS step to be admitted to training [mV] | 12 mV corresponds to ~Vss ≈ −82 mV, the foot of the I_h activation curve for L2/3. Increase (up to ~20 mV) for very low-Rin cells that barely reach I_h threshold; decrease for high-sag cells. |

---

### 4.4 Phase 2.5 (`submit_synth_benchmark.sh`)

| Knob | Default | What it controls | How to tune |
|---|---|---|---|
| `SKIP_PHASE2P5` | 0 | 0 = run Phase 2.5 (standard). 1 = skip (legacy free-Ra diagnostic) | Never set to 1 for a real benchmark run. Use only to reproduce the old free-Ra behaviour for comparison. |
| `N_FLOOR` | 4 | Minimum qualifying cells for a data-driven cohort Ra. Below this → literature fallback | Must be < `CELLS_PER_COHORT`. With 10 cells per cohort, N_FLOOR=4 means up to 6 cells can have failed/degenerate Ra profiles before the fallback fires. |
| `N_RA_PROFILE` | 50 | Ra grid points for the per-cell RMSD-vs-Ra profile | 50 is sufficient for smooth interpolation. Below 30 the spline can miss the argmin; above 100 is diminishing returns vs cost. |

---

### 4.5 Phase 3 — calibration subset (`submit_synth_benchmark.sh`)

| Knob | Default | What it controls | How to tune |
|---|---|---|---|
| `PHASE3_SUBSET` | `first:2` | Which cells in a cohort get bootstrapped | `""` → none. `"first:2"` → first 2 cells (cheapest, enough for a calibration check). `"frac:0.2"` → ~20% of the cohort. `"all"` → every cell (full pipeline; expensive). |
| `BOOTSTRAP_B` | 200 | Bootstrap replicates | 200 is the minimum for stable BCa CIs. Use 500–1000 for publication-quality calibration curves. Each replicate = N_CALLS simulations. |
| `BOOTSTRAP_MODE` | `nonparametric` | Resample strategy | `nonparametric` (resample whole pulses with replacement) is the default in this benchmark because the noise model is already injected synthetically. `parametric` (residual bootstrap) is the standard for real data. |
| `BOOTSTRAP_N_CALLS` | 40 | GP budget per bootstrap replicate | Keep at 30–50 (each replicate starts near the MLE; fewer calls are needed for convergence than for the initial fit). |

---

### 4.6 Walltime and cost estimate

The dominant costs per cell are the Phase 2 fit (N_CALLS simulations), the
τ_w sweep (3 grid points × SWEEP_N_GRID × one inner minimisation each ≈
3×41×~20 = ~2500 simulations), and Phase 2.5 (N_RA_PROFILE × N_CALLS
simulations for the refit). A rough per-cell estimate on da Vinci:

| Stage | Simulations | Time (typical L2/3 morphology) |
|---|---|---|
| Phase 2 initial fit | N_CALLS = 100 | ~3–5 min |
| τ_w sweep (3 values) | 3 × 41 × ~20 = ~2500 | ~15–25 min |
| Phase 2.5 refit | N_RA_PROFILE + N_CALLS ≈ 150 | ~5 min |
| Phase 3 (2 cells) | 2 × B × N_BOOT_CALLS ≈ 16 000 | ~30–60 min |
| **Total per cohort (10 cells, Phase 3 on 2)** | — | **~5–7 h** |

The 10 h walltime in `submit_synth_benchmark.sh` is conservative for this
budget. If you increase `N_CALLS` to 150 or `SWEEP_N_GRID` to 61 consider
bumping walltime to 14 h.

---

## 5. Re-running selectively

**Re-run only the aggregation** (all jobs already done):
```bash
python3 aggregate_synth_results.py --manifest manifest.csv \
    --output-root synthetic_out --archive-root synthetic_archive \
    --out-dir summary --code-dir $CODE_DIR
```

**Re-run one failed cohort** (e.g. cohort_0005 timed out):
```bash
qsub -v GROUP=cohort_0005 submit_synth_benchmark.sh
```
The manifest already exists so generation picks up from where it was; any
already-written archives are overwritten (generation is idempotent).

**Re-run everything from scratch** (new seed, new morphologies, etc.):
Delete `manifest.csv` first, update the config in `submit_all_cohorts.sh`,
then run `bash submit_all_cohorts.sh`. The launcher rebuilds the manifest
and fans out fresh jobs.

---

## 6. Reading the outputs

### `benchmark_summary.csv`

One row per cell. Key columns:

| Column | Meaning |
|---|---|
| `cm_true / rm_true / ra_true` | Injected ground truth |
| `cm_uF_per_cm2 / rm_Ohm_cm2 / ra_Ohm_cm` | Recovered (post Phase 2.5) |
| `cm_ratio / rm_ratio / ra_ratio / tau_ratio` | Recovered / injected (1.0 = perfect) |
| `cm_dex / rm_dex / tau_dex` | log₁₀(ratio): bias in decades |
| `cm_phase2 / rm_phase2 / ra_phase2` | Recovered before Phase 2.5 |
| `tau_w_chosen_ms` | Per-cell auto-selected τ_w |
| `tau_w_hw_rho` | HW_ρ of the winning profile (smaller = sharper = better-identified Cm) |
| `validation_status` | `good` / `to_refine` / `failed` |
| `ih_gihbar_S_cm2` | Injected I_h conductance (the contamination axis) |

### `phase3_coverage.csv`

One row per (cell × parameter). Key columns: `covered_bca`, `covered_perc`
(True/False: does the CI contain the injected value?), `degenerate` (True for
Ra when Phase 2.5 was run — the bootstrap fixes Ra so its CI has zero width
by design).

### Figures

| Figure | What to look for |
|---|---|
| `recovery_distributions` | Violin medians should hug 1.0. Cm and Rm violins wide → iso-τ_m valley active; τ_m violin narrow → τ_m product well-identified despite individual parameters not being so. |
| `bias_vs_injected` | Horizontal trend → no systematic bias across the box. A slope → the estimator is biased for extreme parameter values (e.g. very high Rm). |
| `iso_tau_identifiability` | The correlation printed in the title quantifies the sloppy direction. A correlation of −0.97 (as in your Colab results) means Cm and Rm are nearly exchangeable; the iso-τ_m guide lines show where the data fall along those valleys. |
| `phase2p5_effect` | The left panel (cohort Ra recovered vs injected) should cluster along the identity line. The right panel (τ_m histogram before vs after 2.5) should show the post-2.5 distribution tightening toward 1.0. |
| `validation_health` | A high `failed` fraction at large gIhbar confirms the deflection guard is correctly flagging I_h-contaminated cells. |
| `ci_calibration` | The BCa bars should be close to the nominal 95% dashed line. Bars below → CIs are anti-conservative (too narrow). Bars above → conservative. |
