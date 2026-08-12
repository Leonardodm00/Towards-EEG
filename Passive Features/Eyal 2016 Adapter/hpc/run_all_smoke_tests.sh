#!/bin/bash
# run_all_smoke_tests.sh -- every check, in dependency order, on the LOGIN NODE.
#
# Run AFTER verify_on_cluster.py has passed and AFTER the patches are applied.
# No allocation, no network. Nothing here submits a job.
#
#   cd <bundle>
#   bash run_all_smoke_tests.sh
#
# Override the two paths at the top, or export them:
#   CODE_DIR=/path/to/Biological\ Fit bash run_all_smoke_tests.sh
#
# Stages, cheapest and most fundamental first. A later stage is meaningless
# if an earlier one failed, so each is reported separately and the script
# keeps going to give you the full picture in one pass.
#
#   1  bytes           encoding + line endings + compile        (no deps)
#   2  archive         11 NEURON-free tests vs the raw traces   (no deps)
#   3  patches         all anchors applied, none ambiguous      (needs CODE_DIR)
#   4  Allen unchanged before/after fingerprint comparison      (needs CODE_DIR)
#   5  loader          tests 12-13 through the patched pipeline (needs CODE_DIR)
#   6  NEURON          .asc import, axon deleted, Rin*          (needs NEURON)
#   7  dry run         one cell, tiny budget, end to end        (needs both)

set -u

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${CODE_DIR:-/davinci-1/home/ldellamea/Human Neurons Fitting}"
CONDA_ENV="${CONDA_ENV:-prova}"
ARCHIVE="$BUNDLE_DIR/archive/eyal_archive"
SOURCE="$BUNDLE_DIR/source/195667-master"
TOOLS="$BUNDLE_DIR/tools"
TMP="${TMPDIR:-/tmp}/eyal_smoke_$$"

PASS=0
FAIL=0

stage () {
    echo
    echo "########################################################################"
    echo "# $1"
    echo "########################################################################"
}

report () {   # report <exit_code> <name>
    if [ "$1" -eq 0 ]; then
        echo ">>> PASS: $2"
        PASS=$((PASS + 1))
    else
        echo ">>> FAIL: $2"
        FAIL=$((FAIL + 1))
    fi
}

# --- environment ------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
        && conda activate "$CONDA_ENV" 2>/dev/null \
        && echo "conda env: $CONDA_DEFAULT_ENV" \
        || echo "WARNING: could not activate conda env '$CONDA_ENV'; using \$PATH python3"
fi
echo "python  : $(python3 --version 2>&1)"
echo "bundle  : $BUNDLE_DIR"
echo "code dir: $CODE_DIR"
mkdir -p "$TMP"

# --- 1. bytes ---------------------------------------------------------------
stage "1. BYTES -- integrity, encoding, line endings, syntax"
python3 "$BUNDLE_DIR/verify_on_cluster.py"
report $? "byte-level verification"

# --- 2. archive (NEURON-free, no pipeline needed) ---------------------------
stage "2. ARCHIVE -- 11 NEURON-free tests against the raw ModelDB traces"
( cd "$TOOLS" && python3 smoke_eyal_archive_builder.py \
    --eyal-root "$SOURCE" --out-root "$TMP/archive_a" )
report $? "archive smoke tests (expect pass=11 skip=2)"

if [ ! -d "$CODE_DIR" ]; then
    echo
    echo "CODE_DIR does not exist: $CODE_DIR"
    echo "Stages 3-5 and 7 need the Towards-EEG 'Biological Fit' directory."
    echo "Set CODE_DIR and re-run. Skipping them."
else
    # --- 3. patches ---------------------------------------------------------
    stage "3. PATCHES -- are all 21 edits applied, and none ambiguous?"
    python3 "$TOOLS/patch_eyal_support.py" --code-dir "$CODE_DIR" --check
    report $? "patch anchors resolve"

    # --- 4. Allen unchanged -------------------------------------------------
    stage "4. ALLEN REGRESSION -- did the patches change Allen behaviour?"
    python3 "$TOOLS/regression_allen_unchanged.py" --code-dir "$CODE_DIR"
    report $? "Allen behaviour unchanged (expect FAILURES: 0)"

    # --- 5. loader ----------------------------------------------------------
    stage "5. LOADER -- tests 12-13 through the PATCHED pipeline"
    ( cd "$TOOLS" && python3 smoke_eyal_archive_builder.py \
        --eyal-root "$SOURCE" --out-root "$TMP/archive_b" \
        --monolith-dir "$CODE_DIR" )
    report $? "loader round-trip (expect pass=13 fail=0)"
fi

# --- 6. NEURON --------------------------------------------------------------
stage "6. NEURON -- .asc import, axon deleted, Rin* per cell"
if python3 -c "import neuron" 2>/dev/null; then
    ( cd "$TOOLS" && python3 smoke_eyal_neuron_build.py \
        --archive-root "$ARCHIVE" )
    report $? "NEURON build (expect pass=6 fail=0)"
else
    echo "NEURON not importable in this env -- SKIPPED"
fi

# --- 7. dry run -------------------------------------------------------------
stage "7. DRY RUN -- one cell, tiny budget, the real entrypoint"
if [ -d "$CODE_DIR" ] && python3 -c "import neuron, skopt" 2>/dev/null; then
    python3 "$CODE_DIR/run_biological_fit.py" \
        --archive-dir  "$ARCHIVE" \
        --output-dir   "$TMP/dryrun" \
        --code-dir     "$CODE_DIR" \
        --no-long-square \
        --group-ss-by-amplitude \
        --axon-replacement none \
        --fit-target dep \
        --F 1.9 \
        --ss-window-ms 3.0,102.0 \
        --n-long-train 0 \
        --skip-phase2p5 \
        --phase3-subset none \
        --n-calls 20 --n-initial 10 \
        --max-cells 1 \
        --tau-w-grid-ms 5.0
    report $? "one-cell dry run"
    echo
    echo "Expected: 1 cell loaded (NOT 'No cells loaded'), a finite Cm,"
    echo "and status=not_evaluated -- which for this dataset means"
    echo "'no held-out data existed', NOT 'the fit is bad'."
else
    echo "need CODE_DIR + neuron + skopt -- SKIPPED"
fi

# --- summary ----------------------------------------------------------------
echo
echo "########################################################################"
echo "# SUMMARY:  pass=$PASS  fail=$FAIL"
if [ "$FAIL" -eq 0 ]; then
    echo "# Everything that could run, passed. Safe to qsub."
else
    echo "# DO NOT SUBMIT until the failures above are understood."
fi
echo "########################################################################"
rm -rf "$TMP"
exit $FAIL
