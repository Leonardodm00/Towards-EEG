#!/bin/bash
#
# Wrapper: submits one PBS job per group folder, each pinned to its own node.
#
# Each job receives $GROUP as an environment variable; submit_passive_fit.sh
# then loads cells from <ARCHIVE_ROOT>/<GROUP> and writes results into
# <OUTPUT_ROOT>/<GROUP>.
#
# Usage:
#     ./submit_all_groups.sh                  # submit every group in DEFAULT_GROUPS
#     ./submit_all_groups.sh L2_exc L3_inh    # submit only the named groups
#
# Logs (kept in $HOME via `#PBS -k eo`) will be named
#     passive_fit_<group>.o<jobid>   passive_fit_<group>.e<jobid>
# because we override the job name on the qsub command line.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Must match ARCHIVE_ROOT inside submit_passive_fit.sh — used here only to
# sanity-check that each group directory exists before we waste a qsub.
ARCHIVE_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting"
SUBMIT_SCRIPT="/davinci-1/home/ldellamea/Human Neurons Fitting/submit_passive_fit.sh"

# Default list of groups to submit when no CLI args are given.
# Comment out any you don't want to run.
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
# Any group not listed here falls back to F_FACTOR_DEFAULT (with a warning).
F_FACTOR_DEFAULT=1.9
declare -A F_PER_GROUP=(
    [L2_exc]=1.9
    [L2_inh]=1
    [L3_exc]=1.9
    [L3_inh]=1
    [L4_exc]=1.9
    [L4_inh]=1
    [L5_exc]=2.0
    [L5_inh]=1
    [L6_exc]=2.0
    [L6_inh]=1
)
# ────────────────────────────────────────────────────────────────────────

# If groups were passed on the command line, use those; else use defaults.
if [ "$#" -gt 0 ]; then
    GROUPS=( "$@" )
else
    GROUPS=( "${DEFAULT_GROUPS[@]}" )
fi

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "[FATAL] Submit script not found: $SUBMIT_SCRIPT" >&2
    exit 1
fi

echo "Submitting ${#GROUPS[@]} group(s) — one PBS job each."
echo "Archive root:  $ARCHIVE_ROOT"
echo "Submit script: $SUBMIT_SCRIPT"
echo "----------------------------------------"

n_ok=0
n_skip=0
for g in "${GROUPS[@]}"; do
    if [ ! -d "$ARCHIVE_ROOT/$g" ]; then
        echo "[skip] $g — directory does not exist: $ARCHIVE_ROOT/$g"
        n_skip=$((n_skip + 1))
        continue
    fi
    f="${F_PER_GROUP[$g]}"
    if [ -z "$f" ]; then
        f="$F_FACTOR_DEFAULT"
        echo "[warn] $g not in F_PER_GROUP — falling back to F=$f"
    fi
    echo "[submit] $g  (F=$f)"
    qsub -v "GROUP=$g,F_FACTOR=$f" -N "passive_fit_${g}" "$SUBMIT_SCRIPT"
    n_ok=$((n_ok + 1))
done

echo "----------------------------------------"
echo "Submitted: $n_ok   Skipped: $n_skip"
