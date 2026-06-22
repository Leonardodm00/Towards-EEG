#!/bin/bash
##########################################################################
# submit_all_cohorts.sh — build the manifest ONCE, then fan out one
# PBS job per cohort (qsub -v GROUP=<cohort>).
#
# Run on the LOGIN node:
#     bash submit_all_cohorts.sh
#
# The manifest is the single seeded source of truth (morphology x GT x
# cohort x per-cell I_h/noise). Building it once here guarantees every
# job sees the same draws; the per-cohort jobs only READ their slice.
##########################################################################

set -euo pipefail

# ─── USER CONFIG ────────────────────────────────────────────────────────
CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting/synthetic_benchmark"
MANIFEST="$CODE_DIR/manifest.csv"
JOB_SCRIPT="$CODE_DIR/submit_synth_benchmark.sh"

# Morphology pool (real reconstructions to simulate on):
MORPH_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting/Morphologies"
MORPH_GLOB="specimen_*/reconstruction.swc"   # or "*.swc" for a flat folder

# ─── Manifest design (see synth_gt_grid.py) ─────────────────────────────
SEED=0
DRAWS_PER_MORPH=2          # GT draws per morphology (n_cells = n_morph * this)
CELLS_PER_COHORT=10        # cohort size = PBS job size = Phase-2.5 unit (>= N_FLOOR)
RA_MODE="per_cohort"       # per_cohort (Phase-2.5-coherent) | per_cell (stress test)
E_PAS=-70.0
F_FACTOR=1.9
MAX_CELLS=""               # empty = all (n_morph * DRAWS_PER_MORPH)

# Per-cell variability:
IH_GIHBAR=2e-4             # nominal I_h maximal conductance [S/cm^2]
IH_GIHBAR_CV=0.5           # log-normal CV of gIhbar across cells
IH_EHCN=-45.0
IH_DIST="hay_exponential"
NOISE_SIGMA=0.05           # nominal within-sweep noise [mV]
NOISE_BASELINE=0.05
NOISE_DRIFT=0.10
NOISE_CV=0.3               # log-normal CV of the per-cell noise level
USE_IH=1                   # 1 = inject I_h (default); 0 = passive baseline
# ────────────────────────────────────────────────────────────────────────

cd "$CODE_DIR"

# ─── 1) Build the manifest once (idempotent: skip if present) ───────────
if [ ! -f "$MANIFEST" ]; then
    echo "[fanout] building manifest -> $MANIFEST"
    ARGS=(
        --morph-root "$MORPH_ROOT" --morph-glob "$MORPH_GLOB"
        --out "$MANIFEST" --seed "$SEED"
        --draws-per-morph "$DRAWS_PER_MORPH"
        --cells-per-cohort "$CELLS_PER_COHORT" --ra-mode "$RA_MODE"
        --e-pas "$E_PAS" --F "$F_FACTOR"
        --ih-gihbar "$IH_GIHBAR" --ih-gihbar-cv "$IH_GIHBAR_CV"
        --ih-ehcn "$IH_EHCN" --ih-dist "$IH_DIST"
        --noise-sigma "$NOISE_SIGMA" --noise-baseline "$NOISE_BASELINE"
        --noise-drift "$NOISE_DRIFT" --noise-cv "$NOISE_CV"
    )
    [ -n "$MAX_CELLS" ] && ARGS+=(--max-cells "$MAX_CELLS")
    [ "$USE_IH" = "0" ] && ARGS+=(--no-ih)
    python3 synth_gt_grid.py "${ARGS[@]}"
else
    echo "[fanout] manifest exists, reusing: $MANIFEST"
    echo "         (delete it to redraw with new settings)"
fi

# ─── 2) Read the unique cohort labels from the manifest ─────────────────
GROUPS=$(python3 -c "import pandas as pd; print(' '.join(sorted(pd.read_csv(r'''$MANIFEST''')['group'].unique())))")
if [ -z "$GROUPS" ]; then
    echo "[FATAL] no cohorts found in $MANIFEST" >&2; exit 1
fi
echo "[fanout] cohorts to submit: $GROUPS"

# ─── 3) One job per cohort ──────────────────────────────────────────────
for g in $GROUPS; do
    echo "[fanout] qsub -v GROUP=$g $JOB_SCRIPT"
    qsub -v GROUP="$g" "$JOB_SCRIPT"
done
echo "[fanout] submitted $(echo "$GROUPS" | wc -w) cohort job(s)."
