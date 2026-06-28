#!/bin/bash
##########################################################################
# submit_all_cohorts.sh -- build the manifest ONCE, then fan out one
# PBS job per cohort (qsub -v GROUP=<cohort>).
#
# Run on the LOGIN node, AFTER activating the env:
#     conda activate prova
#     bash submit_all_cohorts.sh
#
# v3 changes vs pilot:
#   - Physiological (Cm, tau_m) rejection filter active by default.
#   - CELLS_PER_COHORT raised to 8 so Phase 2.5 median is meaningful.
#   - MAX_CELLS removed (empty = all morphologies).
#   - Ra stream isolated: changing Cm/tau_m constraints does not shift Ra.
##########################################################################

set -uo pipefail

# --- USER CONFIG -----------------------------------------------------------
CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting/Synthetic Test"
MANIFEST="$CODE_DIR/manifest.csv"
JOB_SCRIPT="$CODE_DIR/submit_synth_benchmark.sh"
CONDA_ENV="prova"

# Morphology pool:
MORPH_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc"
MORPH_GLOB="specimen_*/reconstruction.swc"

# Manifest design:
SEED=0
DRAWS_PER_MORPH=2       # increase if you want more draws per morphology
CELLS_PER_COHORT=8      # >=8 so cohort-median Ra is a real estimate
RA_MODE="per_cohort"
E_PAS=-70.0
F_FACTOR=1.9
# MAX_CELLS intentionally omitted -> use all morphologies

# --- Physiological prior constraints (v3) ----------------------------------
# Rejection-sample (Cm, Rm) so that:
#   CM_PHYS_LO <= Cm [uF/cm^2] <= CM_PHYS_HI
#   TAU_LO_MS  <= tau_m = Rm*Cm*1e-3 [ms] <= TAU_HI_MS
#
# Rationale:
#   Human L2/3 Cm ~0.45 uF/cm^2 (Eyal 2016); standard ~1.0. Window [0.4, 1.5]
#   covers the anomalously-low human value and the standard value with margin.
#   tau_m [3, 40] ms spans the physiological range for L3 pyramidal neurons
#   fitted by this pipeline (Allen data) without reaching the 100+ ms regime
#   where the 100 ms fit window becomes the binding constraint.
#
# Individual (Cm, Rm) draws remain within the fitter search box
# ([0.3, 3.0] and [1000, 100000]), so assert_bounds_match_phase1 still passes.
# Set --no-phy-filter on the synth_gt_grid.py call to disable (v2 behaviour).
CM_PHYS_LO=0.4      # uF/cm^2
CM_PHYS_HI=1.5      # uF/cm^2
TAU_LO_MS=3.0       # ms
TAU_HI_MS=40.0      # ms

# Per-cell variability:
IH_GIHBAR=2e-4
IH_GIHBAR_CV=0.5
IH_EHCN=-45.0
IH_DIST="hay_exponential"
NOISE_SIGMA=0.05
NOISE_BASELINE=0.05
NOISE_DRIFT=0.10
NOISE_CV=0.3
USE_IH=1
# --------------------------------------------------------------------------

cd "$CODE_DIR" \
    || { echo "[FATAL] cannot cd to CODE_DIR: $CODE_DIR" >&2; exit 1; }

# --- 0) Ensure the conda env is active (provides python, pandas) -----------
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    echo "[fanout] activating conda env '$CONDA_ENV'..."
    source "$(conda info --base)/etc/profile.d/conda.sh" \
        || { echo "[FATAL] cannot source conda.sh" >&2; exit 1; }
    conda activate "$CONDA_ENV" \
        || { echo "[FATAL] cannot activate env '$CONDA_ENV'" >&2; exit 1; }
fi
echo "[fanout] python: $(which python)   env: ${CONDA_DEFAULT_ENV:-none}"

# --- 1) Build the manifest once (idempotent) --------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "[fanout] building manifest -> $MANIFEST"
    ARGS=(
        --morph-root       "$MORPH_ROOT"
        --morph-glob       "$MORPH_GLOB"
        --out              "$MANIFEST"
        --seed             "$SEED"
        --draws-per-morph  "$DRAWS_PER_MORPH"
        --cells-per-cohort "$CELLS_PER_COHORT"
        --ra-mode          "$RA_MODE"
        --e-pas            "$E_PAS"
        --F                "$F_FACTOR"
        --ih-gihbar        "$IH_GIHBAR"
        --ih-gihbar-cv     "$IH_GIHBAR_CV"
        --ih-ehcn          "$IH_EHCN"
        --ih-dist          "$IH_DIST"
        --noise-sigma      "$NOISE_SIGMA"
        --noise-baseline   "$NOISE_BASELINE"
        --noise-drift      "$NOISE_DRIFT"
        --noise-cv         "$NOISE_CV"
        # physiological prior constraints (v3)
        --cm-phys-lo       "$CM_PHYS_LO"
        --cm-phys-hi       "$CM_PHYS_HI"
        --tau-lo-ms        "$TAU_LO_MS"
        --tau-hi-ms        "$TAU_HI_MS"
    )
    [ "$USE_IH" = "0" ] && ARGS+=(--no-ih)
    python synth_gt_grid.py "${ARGS[@]}" \
        || { echo "[FATAL] synth_gt_grid.py failed" >&2; exit 1; }
else
    echo "[fanout] manifest exists, reusing -> $MANIFEST"
    echo "         (delete it to redraw with new settings)"
fi

# --- 2) Read unique cohort labels via the manifest's own loader -------------
echo "[fanout] reading cohort labels..."
COHORT_LIST="$(python -c '
import sys
from synth_gt_grid import load_manifest, list_groups
print("\n".join(list_groups(load_manifest(sys.argv[1]))))
' "$MANIFEST")"
STATUS=$?

if [ $STATUS -ne 0 ] || [ -z "$COHORT_LIST" ]; then
    echo "[FATAL] could not read cohort labels from $MANIFEST" >&2
    echo "        Run the reader directly to see the traceback:" >&2
    echo "        cd '$CODE_DIR' && python -c \"from synth_gt_grid import " \
         "load_manifest, list_groups; print(list_groups(load_manifest('$MANIFEST')))\"" >&2
    exit 1
fi

mapfile -t COHORTS <<< "$COHORT_LIST"
if [ "${#COHORTS[@]}" -eq 0 ]; then
    echo "[FATAL] no cohort labels found in $MANIFEST" >&2
    exit 1
fi
echo "[fanout] cohorts found: ${COHORTS[*]}"

# --- 3) One PBS job per cohort ----------------------------------------------
N=0
for g in "${COHORTS[@]}"; do
    echo "[fanout] submitting GROUP=$g ..."
    qsub -v GROUP="$g" "$JOB_SCRIPT" \
        || { echo "[WARN] qsub failed for GROUP=$g" >&2; }
    N=$((N + 1))
done
echo "[fanout] done -- $N cohort job(s) submitted."
