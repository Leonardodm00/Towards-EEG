#!/bin/bash
# ---------------------------------------------------------------------------
# Run every smoke test in this directory, on the cluster, and say plainly
# which passed. Cheap enough for the login node (about a minute, single core,
# no network); submit it only if your site forbids that.
#
#   cd h01_code
#   bash run_smoke_tests.sh                 # all suites
#   bash run_smoke_tests.sh p0 p3           # only suites whose name matches
#   ENV_NAME=other_env bash run_smoke_tests.sh
#   qsub run_smoke_tests.sh                 # same script, as a job
#
# Exit status 0 only if every selected suite passed. Any failure leaves the
# full output in logs/smoke_<suite>_<stamp>.log and prints the path.
#
# Suites are DISCOVERED, not listed: a new smoke_test_*.py is picked up with
# no edit here. That is deliberate -- a hard-coded list silently stops testing
# whatever was added last.
# ---------------------------------------------------------------------------
#PBS -N smoke
#PBS -q cpu
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o logs/smoke.log

set -eo pipefail

# SKIP_CONDA=1 runs the script against whatever python3 is already on PATH.
# It exists so this script can be exercised off-cluster before it is shipped;
# on the cluster, leave it unset so the env is the one the jobs will use.
if [ -z "${SKIP_CONDA:-}" ]; then
    ENV_NAME="${ENV_NAME:-spine_env}"
    # set +e AS WELL AS set +u. `conda activate` runs the env's activate.d
    # hooks, and those can return non-zero while still having activated
    # correctly -- binutils on this cluster prints its INFO block and returns
    # 1. Under `set -e` that killed the script here, after the INFO block and
    # before any test ran: no error, no output, exit 1. Verified by
    # reproduction, 2026-09-16.
    set +u
    set +e
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    set -e
    set -u
    # So trust the OUTCOME, not the status: activation is real only if
    # python3 now resolves inside the env.
    case "$(command -v python3 || true)" in
        *"/envs/$ENV_NAME/"*) ;;
        *)
            echo "ERROR: 'conda activate $ENV_NAME' did not take effect."
            echo "  python3 is $(command -v python3 || echo '<none>')"
            echo "  Check the env exists:  conda env list"
            echo "  Or bypass:             SKIP_CONDA=1 bash run_smoke_tests.sh"
            exit 1
            ;;
    esac
else
    set -u
    echo "SKIP_CONDA set -- using $(which python3) without activating an env"
fi

# When submitted with qsub, start in the directory the job was submitted from;
# when run directly, in the directory the script lives in.
CODE="${CODE:-${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$CODE"
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

echo "==========================================================="
echo " smoke tests | $(hostname) | $(date -u +%FT%TZ)"
echo " dir    $CODE"
echo " env    ${CONDA_DEFAULT_ENV:-<none>}"
echo " python $(which python3)  ($(python3 -V 2>&1))"
echo "==========================================================="

# --- 1. environment report --------------------------------------------------
# Reported, never asserted: a missing plotly skips figure tests and is not a
# failure, while a missing numpy is caught by the suites themselves with a
# better message than anything this script could print.
python3 - <<'PYEOF'
import importlib
for m in ("numpy", "pandas", "scipy", "skimage", "neuron", "LFPy", "plotly",
          "cloudvolume"):
    try:
        mod = importlib.import_module(m)
        print("  %-12s %s" % (m, getattr(mod, "__version__", "present")))
    except Exception as exc:
        print("  %-12s MISSING (%s)" % (m, type(exc).__name__))
PYEOF

# --- 2. the Stage 1 symlink farm -------------------------------------------
# This IS asserted. Every suite that touches the real partition imports through
# stage1/, and an unlinked farm makes them fail in a way that looks like a code
# fault rather than a setup step never run.
echo
echo "--- stage1 symlink farm ---"
if ! bash stage1_link.sh --check; then
    echo "  farm incomplete -- creating it"
    bash stage1_link.sh
fi

# --- 3. the suites ----------------------------------------------------------
ALL=()
while IFS= read -r f; do
    [ -n "$f" ] && ALL+=("$f")
done < <(ls smoke_test_*.py 2>/dev/null | sort)
if [ "${#ALL[@]}" -eq 0 ]; then
    echo "no smoke_test_*.py in $CODE"
    exit 1
fi

SUITES=()
if [ "$#" -gt 0 ]; then
    for pat in "$@"; do
        for s in "${ALL[@]}"; do
            case "$s" in *"$pat"*) SUITES+=("$s");; esac
        done
    done
    if [ "${#SUITES[@]}" -eq 0 ]; then
        echo "no suite matches: $*"
        printf '  available: %s\n' "${ALL[@]}"
        exit 1
    fi
else
    SUITES=("${ALL[@]}")
fi

PASSED=(); FAILED=()
for s in "${SUITES[@]}"; do
    name="${s%.py}"; name="${name#smoke_test_}"
    log="logs/smoke_${name}_${STAMP}.log"
    echo
    echo "--- $s ---"
    # `set -e` must not abort the loop: one failing suite should not hide the
    # verdicts of the others. Hence the explicit if, and || true on the tail.
    if python3 "$s" > "$log" 2>&1; then
        tail -n 2 "$log" | sed 's/^/  /'
        PASSED+=("$name")
        rm -f "$log"
    else
        tail -n 25 "$log" | sed 's/^/  /' || true
        echo "  FULL LOG: $CODE/$log"
        FAILED+=("$name")
    fi
done

# --- 4. verdict -------------------------------------------------------------
echo
echo "==========================================================="
echo " passed ${#PASSED[@]}/${#SUITES[@]}: ${PASSED[*]:-none}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo " FAILED ${#FAILED[@]}: ${FAILED[*]}"
    echo "==========================================================="
    exit 1
fi
echo " ALL SUITES PASSED"
echo "==========================================================="
