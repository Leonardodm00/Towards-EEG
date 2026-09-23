#!/bin/bash

#PBS -S /bin/bash
#PBS -N "ih_recovery"
#PBS -q cpu
#PBS -l select=1:ncpus=1,walltime=200:00:00
#PBS -k eo

##########################################################################
# Stage 7 -- synthetic six-parameter recovery, and the gate to Stage 8.
#
# Entry point: run_ih_recovery.py. It draws a cohort of synthetic cells on the
# REAL archive morphologies, generates Phase-0 archives from them, fits them
# with the campaign's OWN arms (run_ih_fit.py), and reports how much of the
# injected ground truth came back.
#
# Read the exit status, not the log's last line:
#     0  the gate PASSED   -- Stage 8 may start
#     2  the gate FAILED   -- gate_verdict.csv names the axis and, when the
#                             failure is a kinetic knob, D-005 says to freeze
#                             it (--fit-params without it) and record it
#     1  the run itself broke before a verdict existed
#
# -- Dispatch ------------------------------------------------------------------
#     qsub -v MORPH_ROOT=/path/to/L3_exc submit_ih_recovery.sh
#     qsub -v MORPH_ROOT=...,RUN_TAG=pilot,MAX_CELLS=4,N_CALLS=40 submit_ih_recovery.sh
#     qsub -v MANIFEST=/path/manifest.csv,SKIP_GENERATE=1,ARCHIVE_DIR=... \
#          submit_ih_recovery.sh          # re-fit a cohort already generated
#
# -- COST ----------------------------------------------------------------------
# MAX_CELLS x |ARMS| fits at N_CALLS/N_INITIAL. The default is 20 x 4 at
# 200/100 = 80 fits. The budget is deliberately NOT reduced: a recovery
# failure at 60 calls would say nothing about a campaign that runs at 200.
# Shake the pipeline down with MAX_CELLS=2,N_CALLS=20 first, then run the real
# cohort -- and note that the pilot's verdict is not a verdict.
##########################################################################

# --- Configuration (nothing needs editing; every path is a knob) ------------
# --- Paths: derived, not hard-coded -----------------------------------------
# SUBMIT FROM THE "Ih Fit" DIRECTORY. PBS sets PBS_O_WORKDIR to the directory
# qsub was invoked from, and that is where this script looks for the code:
#
#   cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/HPC script/Ih Fit"
#   qsub -v GROUP=L3_exc,ARM=ih6 submit_ih_recovery.sh
#
# Deriving the code directory this way rather than hard-coding it means a
# `git pull` into a different checkout, or a renamed clone, cannot leave the
# job running last week's code. It also avoids passing a path containing
# SPACES through `qsub -v`, which is comma-separated and does not survive one.
#
# RESERVED NAMES. The login shell on this cluster exports CODE (and ENV_NAME),
# and PBS jobs source .bashrc -- which has already sent one campaign into the
# wrong directory (paths reference, sections 8.2 and 8.4). Every knob here is
# therefore prefixed IH_, and a bare name is reported and ignored.
for _n in CODE ROOT ARCHIVE_ROOT OUTPUT_BASE CODE_DIR; do
    eval "_v=\${$_n:-}"
    if [ -n "$_v" ]; then
        echo "NOTE: $_n is set ($_v) and is IGNORED; the knobs are"
        echo "      IH_CODE_DIR, IH_ARCHIVE_ROOT, IH_OUTPUT_BASE."
    fi
done
unset _n _v

CODE_DIR="${IH_CODE_DIR:-${PBS_O_WORKDIR:-$(pwd)}}"
# The Phase-0 archives are NOT in git (there is no specimen_* anywhere in the
# repo), so their root is independent of the checkout. The value below is the
# passive-fit working directory the earlier runs used; override with
# IH_ARCHIVE_ROOT if the archives live elsewhere.
ARCHIVE_ROOT="${IH_ARCHIVE_ROOT:-/davinci-1/home/ldellamea/Human Neurons Fitting}"
OUTPUT_BASE="${IH_OUTPUT_BASE:-$ARCHIVE_ROOT/stage7_recovery}"
RUN_TAG="${RUN_TAG:-default}"
CONDA_ENV="${IH_ENV:-prova}"
ENTRYPOINT="$CODE_DIR/run_ih_recovery.py"
MOD_DIR="$CODE_DIR/mod"

if [ ! -f "$ENTRYPOINT" ]; then
    echo "[FATAL] run_ih_recovery.py not found under CODE_DIR." >&2
    echo "        CODE_DIR resolved to: $CODE_DIR" >&2
    echo "        Submit from the 'Ih Fit' directory:" >&2
    echo "          cd \"/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/HPC script/Ih Fit\"" >&2
    echo "          qsub -v MORPH_ROOT=... submit_ih_recovery.sh" >&2
    echo "        (or export IH_CODE_DIR before qsub)." >&2
    exit 2
fi

# The morphologies the cohort is drawn on. D-007/C4: these are the cells the
# campaign will actually fit, because identifiability depends on the arbour.
MORPH_ROOT="${MORPH_ROOT:-$ARCHIVE_ROOT/L3_exc}"
MORPH_GLOB="${MORPH_GLOB:-specimen_*/reconstruction.swc}"

# RUN B's results, from which the synthetic noise level is taken. WITHOUT IT
# the cohort is generated at the benchmark default, which may be cleaner than
# any real recording -- and a gate that passes on cleaner data than the
# campaign's says nothing about the campaign.
NOISE_FROM_RESULTS="${NOISE_FROM_RESULTS:-}"

# Reuse an existing cohort instead of drawing one.
MANIFEST="${MANIFEST:-}"
ARCHIVE_DIR="${ARCHIVE_DIR:-}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
OVERWRITE="${OVERWRITE:-0}"
CLEAN_ARCHIVE="${CLEAN_ARCHIVE:-0}"

# --- Cohort -----------------------------------------------------------------
MAX_CELLS="${MAX_CELLS:-20}"
DRAWS_PER_MORPH="${DRAWS_PER_MORPH:-1}"
CELLS_PER_COHORT="${CELLS_PER_COHORT:-10}"
SEED="${SEED:-0}"
FP_CONTROL_FRAC="${FP_CONTROL_FRAC:-0.25}"   # the false-positive control
E_PAS="${E_PAS:--73.5}"

# --- The I_h ground truth (must agree with the fit, or it is misspecification)
IH_MECHANISM="${IH_MECHANISM:-Ih_human}"
IH_DISTRIBUTION="${IH_DISTRIBUTION:-uniform}"
IH_REGIONS="${IH_REGIONS:-soma,dend,apic}"
VSHIFT_BASE="${VSHIFT_BASE:-0.0}"
EHCN="${EHCN:-}"                              # empty = the mechanism default
GBAR_RANGE="${GBAR_RANGE:-2e-5,3e-4}"         # spans Rich / Kalmbach / Hay
DVH_RANGE="${DVH_RANGE:--5,5}"                # inside the fit box [-10,10]
KAPPA_RANGE="${KAPPA_RANGE:-0.7,1.4}"         # inside the fit box [0.5,2]

# --- The generated protocol --------------------------------------------------
LS_HYP_AMPS="${LS_HYP_AMPS:--10,-30,-50,-70,-90,-110,-150}"
LS_DEP_AMPS="${LS_DEP_AMPS:-20,50}"           # the D-006 validation set
SS_N_REPEATS="${SS_N_REPEATS:-30}"

# --- The fit (identical to the campaign's) ----------------------------------
ARMS="${ARMS:-baseline_runB,passive_fullstep,ih4,ih6}"
F_FACTOR="${F_FACTOR:-1.9}"
N_CALLS="${N_CALLS:-200}"
N_INITIAL="${N_INITIAL:-100}"
DT_BRIEF_MS="${DT_BRIEF_MS:-0.1}"
DT_LONG_MS="${DT_LONG_MS:-0.1}"
PHASE3_SUBSET="${PHASE3_SUBSET:-none}"

# --- The gate (the tolerances are the assistant's PROPOSAL, plan section 8) --
GATE_ARM="${GATE_ARM:-ih6}"
TOL_CM_FACTOR="${TOL_CM_FACTOR:-1.25}"
TOL_GBAR_FACTOR="${TOL_GBAR_FACTOR:-1.25}"
TOL_DVH_MV="${TOL_DVH_MV:-3.0}"
TOL_KAPPA_FACTOR="${TOL_KAPPA_FACTOR:-1.5}"
TOL_FP_LOW_RAIL_FRAC="${TOL_FP_LOW_RAIL_FRAC:-0.8}"
TOL_FP_CM_FACTOR="${TOL_FP_CM_FACTOR:-1.25}"
# ----------------------------------------------------------------------------

case "$RUN_TAG" in
    ""|*[!A-Za-z0-9_.-]*)
        echo "[FATAL] RUN_TAG must be non-empty and contain only letters," >&2
        echo "        digits, '_', '.', '-'  (got: '$RUN_TAG')" >&2
        exit 4 ;;
esac

OUTPUT_DIR="$OUTPUT_BASE/$RUN_TAG"

if [ -z "$MANIFEST" ] && [ ! -d "$MORPH_ROOT" ]; then
    echo "[FATAL] MORPH_ROOT does not exist: $MORPH_ROOT" >&2
    echo "        Give the group directory whose specimen_*/ hold the SWCs," >&2
    echo "        or pass MANIFEST=<csv> to reuse a cohort." >&2
    exit 3
fi
if [ -n "$MANIFEST" ] && [ ! -f "$MANIFEST" ]; then
    echo "[FATAL] MANIFEST does not exist: $MANIFEST" >&2
    exit 3
fi
if [ -n "$NOISE_FROM_RESULTS" ] && [ ! -f "$NOISE_FROM_RESULTS" ]; then
    echo "[FATAL] NOISE_FROM_RESULTS does not exist: $NOISE_FROM_RESULTS" >&2
    exit 3
fi

source "$(conda info --base)/etc/profile.d/conda.sh" \
    || { echo "[FATAL] cannot source conda.sh" >&2; exit 5; }
conda activate "$CONDA_ENV" \
    || { echo "[FATAL] cannot activate conda env '$CONDA_ENV'" >&2; exit 5; }

# NEURON auto-loads ./x86_64 from the PROCESS working directory.
cd "$CODE_DIR" || { echo "[FATAL] cannot cd to CODE_DIR: $CODE_DIR" >&2; exit 5; }

# ============================================================================
#  nrnivmodl guard -- same contract as submit_ih_fit.sh
# ============================================================================
SPECIAL="$CODE_DIR/x86_64/special"
LOCK="$CODE_DIR/.nrnivmodl.lock"
[ -d "$MOD_DIR" ] || { echo "[FATAL] mod dir not found: $MOD_DIR" >&2; exit 6; }

need_build=0; build_reason=""
if [ ! -x "$SPECIAL" ]; then
    need_build=1; build_reason="x86_64/special missing or not executable"
else
    newer_mod=$(find "$MOD_DIR" -maxdepth 1 -name '*.mod' -newer "$SPECIAL" \
                     -print 2>/dev/null | head -n 1)
    [ -n "$newer_mod" ] && { need_build=1
        build_reason="$(basename "$newer_mod") is newer than x86_64/special"; }
fi

if [ "$need_build" = "1" ]; then
    echo "[nrnivmodl] rebuild needed: $build_reason"
    if mkdir "$LOCK" 2>/dev/null; then
        trap 'rmdir "$LOCK" 2>/dev/null' EXIT
        rm -rf "$CODE_DIR/x86_64"
        nrnivmodl "$MOD_DIR"; rc=$?
        rmdir "$LOCK" 2>/dev/null; trap - EXIT
        [ $rc -ne 0 ] && { echo "[FATAL] nrnivmodl failed (exit $rc)" >&2; exit 6; }
    else
        echo "[nrnivmodl] another job holds the lock; waiting (<= 10 min) ..."
        waited=0
        while [ -d "$LOCK" ] && [ $waited -lt 600 ]; do sleep 5; waited=$((waited+5)); done
        echo "[nrnivmodl] waited ${waited}s"
    fi
else
    echo "[nrnivmodl] up to date: $SPECIAL"
fi
[ -x "$SPECIAL" ] || { echo "[FATAL] $SPECIAL still absent." >&2; exit 6; }

python - <<'PYCHK'
import sys
from neuron import h
missing = [m for m in ("Ih", "Ih_human") if not hasattr(h, m)]
if missing:
    sys.stderr.write("[FATAL] mechanism(s) not loaded: %s\n" % ", ".join(missing))
    sys.exit(7)
print("[nrnivmodl] mechanisms loaded: Ih, Ih_human")
PYCHK
[ $? -ne 0 ] && { echo "[FATAL] mechanism verification failed." >&2; exit 7; }
# ============================================================================

mkdir -p "$OUTPUT_DIR"

echo "Running on node:        $(hostname)"
echo "Job ID:                 $PBS_JOBID"
echo "Run tag:                $RUN_TAG"
echo "Python:                 $(which python)   | conda: $CONDA_DEFAULT_ENV"
echo "Working dir (process):  $(pwd)"
echo "Code dir (resolved):    $CODE_DIR"
echo "Output dir:             $OUTPUT_DIR"
echo "Morphologies:           ${MANIFEST:-$MORPH_ROOT/$MORPH_GLOB}"
echo "Cohort:                 max $MAX_CELLS cell(s), seed $SEED, FP fraction $FP_CONTROL_FRAC"
echo "I_h truth:              $IH_MECHANISM / $IH_DISTRIBUTION / regions=$IH_REGIONS"
echo "                        gbar [$GBAR_RANGE]  dv_h [$DVH_RANGE] mV  kappa [$KAPPA_RANGE]"
echo "Generated protocol:     LS hyp [$LS_HYP_AMPS] pA, LS dep [$LS_DEP_AMPS] pA, SS x$SS_N_REPEATS"
if [ -n "$NOISE_FROM_RESULTS" ]; then
    echo "Noise:                  MEASURED, from $NOISE_FROM_RESULTS"
else
    echo "Noise:                  *** BENCHMARK DEFAULT, NOT MEASURED ***"
    echo "                        Pass NOISE_FROM_RESULTS=<run B phase2_results.csv>"
    echo "                        or the gate passes on data cleaner than the campaign's."
fi
echo "Arms:                   $ARMS   at $N_CALLS/$N_INITIAL, dt ${DT_LONG_MS}ms"
echo "Gate arm:               $GATE_ARM"
echo "-----------------------------------------"

ARGS=(
    --output-dir         "$OUTPUT_DIR"
    --code-dir           "$CODE_DIR"
    --morph-glob         "$MORPH_GLOB"
    --draws-per-morph    "$DRAWS_PER_MORPH"
    --cells-per-cohort   "$CELLS_PER_COHORT"
    --max-cells          "$MAX_CELLS"
    --seed               "$SEED"
    "--e-pas=$E_PAS"
    --fp-control-frac    "$FP_CONTROL_FRAC"
    --ih-mechanism       "$IH_MECHANISM"
    --ih-distribution    "$IH_DISTRIBUTION"
    --ih-regions         "$IH_REGIONS"
    --vshift-base        "$VSHIFT_BASE"
    "--gbar-range=$GBAR_RANGE"
    "--dvh-range=$DVH_RANGE"
    "--kappa-range=$KAPPA_RANGE"
    "--ls-hyp-amps=$LS_HYP_AMPS"
    "--ls-dep-amps=$LS_DEP_AMPS"
    --ss-n-repeats       "$SS_N_REPEATS"
    --arms               "$ARMS"
    --F                  "$F_FACTOR"
    --n-calls            "$N_CALLS"
    --n-initial          "$N_INITIAL"
    --dt-brief-ms        "$DT_BRIEF_MS"
    --dt-long-ms         "$DT_LONG_MS"
    --phase3-subset      "$PHASE3_SUBSET"
    --gate-arm           "$GATE_ARM"
    --tol-cm-factor      "$TOL_CM_FACTOR"
    --tol-gbar-factor    "$TOL_GBAR_FACTOR"
    --tol-dvh-mV         "$TOL_DVH_MV"
    --tol-kappa-factor   "$TOL_KAPPA_FACTOR"
    --tol-fp-low-rail-frac "$TOL_FP_LOW_RAIL_FRAC"
    --tol-fp-cm-factor   "$TOL_FP_CM_FACTOR"
)

[ -n "$MANIFEST" ]           && ARGS+=(--manifest "$MANIFEST")
[ -z "$MANIFEST" ]           && ARGS+=(--morph-root "$MORPH_ROOT")
[ -n "$ARCHIVE_DIR" ]        && ARGS+=(--archive-dir "$ARCHIVE_DIR")
[ -n "$NOISE_FROM_RESULTS" ] && ARGS+=(--noise-from-results "$NOISE_FROM_RESULTS")
[ -n "$EHCN" ]               && ARGS+=("--ehcn=$EHCN")
[ "$SKIP_GENERATE" = "1" ]   && ARGS+=(--skip-generate)
[ "$OVERWRITE" = "1" ]       && ARGS+=(--overwrite)
[ "$CLEAN_ARCHIVE" = "1" ]   && ARGS+=(--clean-archive)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY RUN] would run:"
    echo "  python $ENTRYPOINT \\"
    printf '    %s\n' "${ARGS[@]}"
    echo "[DRY RUN] env + mechanisms verified; nothing generated, nothing fitted."
    conda deactivate
    exit 0
fi

python "$ENTRYPOINT" "${ARGS[@]}"
status=$?

conda deactivate
case $status in
    0) echo "[done] run '$RUN_TAG': STAGE 7 GATE PASSED -- Stage 8 may start." ;;
    2) echo "[done] run '$RUN_TAG': STAGE 7 GATE FAILED -- see $OUTPUT_DIR/gate_verdict.csv" ;;
    *) echo "[done] run '$RUN_TAG': the run broke before a verdict (exit $status)" ;;
esac
sleep 5s
exit $status
