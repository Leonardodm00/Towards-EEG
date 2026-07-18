#!/bin/bash
##########################################################################
# submit_all_groups_biological.sh -- fan out one PBS job per (layer x type)
# group folder, each pinned to its own node, for the REAL-DATA passive fit.
#
# Each job receives $GROUP (+ per-group F and the Phase 2.5 / Phase 3 knobs)
# via `qsub -v`; submit_biological_fit.sh then loads cells from
# <ARCHIVE_ROOT>/<GROUP> and writes results into <OUTPUT_ROOT>/<GROUP>.
#
# Usage:
#     ./submit_all_groups_biological.sh                # every group in DEFAULT_GROUPS
#     ./submit_all_groups_biological.sh L3_exc L5_exc  # only the named groups
#
# Logs (kept in $HOME via `#PBS -k eo`) are named
#     bio_passive_fit_<group>.o<jobid>   bio_passive_fit_<group>.e<jobid>
##########################################################################

set -uo pipefail

# --- USER CONFIG ------------------------------------------------------------
# Must match ARCHIVE_ROOT inside submit_biological_fit.sh -- used here only to
# sanity-check that each group directory exists before wasting a qsub.
ARCHIVE_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting"
SUBMIT_SCRIPT="/davinci-1/home/ldellamea/Human Neurons Fitting/submit_biological_fit.sh"

# Default groups to submit when no CLI args are given. Comment out any to skip.
DEFAULT_GROUPS=(
    L2_exc
    L2_inh
    L3_exc
    L3_inh
    L4_exc
    L4_inh
    L5_exc
    L5_inh
    L6_exc
    L6_inh
)

# Per-group spine-area correction factor F (Eyal-style scaling).
# Any group not listed falls back to F_FACTOR_DEFAULT (with a warning).
F_FACTOR_DEFAULT=1.9
declare -A F_PER_GROUP=(
    [L2_exc]=1.9
    [L2_inh]=1
    [L3_exc]=1.9
    [L3_inh]=1
    [L4_exc]=1.5
    [L4_inh]=1
    [L5_exc]=1.5
    [L5_inh]=1
    [L6_exc]=1.5
    [L6_inh]=1
)

# --- Phase 2.5 / Phase 3 controls (methodological constants, not per-group) --
# Forwarded to every job via `qsub -v`. Edit once for the whole sweep.
SKIP_PHASE2P5=0     # 1 = skip Phase 2.5 everywhere (legacy free-Ra), 0 = run it
N_FLOOR=4           # min qualifying cells for a cohort-median Ra; a group with
                    #   FEWER than this uses the per-(layer,type) LITERATURE Ra
                    #   fallback. Lower it (e.g. 2) if you trust small-group
                    #   medians; raise it to be more conservative.
N_RA_PROFILE=50     # Ra grid points for the RMSD-vs-Ra profile
PHASE3_SUBSET="frac:0.5"   # none | all | first:N | frac:F  (bootstrap subset)
# ----------------------------------------------------------------------------

# If groups were passed on the command line, use those; else use defaults.
if [ "$#" -gt 0 ]; then
    SEL_GROUPS=( "$@" )
else
    SEL_GROUPS=( "${DEFAULT_GROUPS[@]}" )
fi

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "[FATAL] Submit script not found: $SUBMIT_SCRIPT" >&2
    exit 1
fi

echo "Submitting ${#SEL_GROUPS[@]} group(s) -- one PBS job each."
echo "Archive root:  $ARCHIVE_ROOT"
echo "Submit script: $SUBMIT_SCRIPT"
if [ "$SKIP_PHASE2P5" = "1" ]; then
    echo "Phase 2.5:     SKIPPED for all groups (legacy free-Ra)"
else
    echo "Phase 2.5:     ON for all groups (N_FLOOR=$N_FLOOR  N_RA_PROFILE=$N_RA_PROFILE)"
fi
echo "Phase 3:       subset='$PHASE3_SUBSET' for all groups"
echo "----------------------------------------"

n_ok=0
n_skip=0
for g in "${SEL_GROUPS[@]}"; do
    if [ ! -d "$ARCHIVE_ROOT/$g" ]; then
        echo "[skip] $g -- directory does not exist: $ARCHIVE_ROOT/$g"
        n_skip=$((n_skip + 1))
        continue
    fi
    f="${F_PER_GROUP[$g]:-}"
    if [ -z "$f" ]; then
        f="$F_FACTOR_DEFAULT"
        echo "[warn] $g not in F_PER_GROUP -- falling back to F=$f"
    fi
    echo "[submit] $g  (F=$f)"
    qsub -v "GROUP=$g,F_FACTOR=$f,SKIP_PHASE2P5=$SKIP_PHASE2P5,N_FLOOR=$N_FLOOR,N_RA_PROFILE=$N_RA_PROFILE,PHASE3_SUBSET=$PHASE3_SUBSET" \
         -N "bio_passive_fit_${g}" "$SUBMIT_SCRIPT" \
         || { echo "[WARN] qsub failed for GROUP=$g" >&2; }
    n_ok=$((n_ok + 1))
done

echo "----------------------------------------"
echo "Submitted: $n_ok   Skipped: $n_skip"
