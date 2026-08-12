#!/bin/bash
# push_eyal_adapter.sh -- clone Towards-EEG, unpack the bundle from Downloads,
# commit the Eyal adapter, and push.
#
# RUN THIS ON YOUR LOCAL MACHINE, NOT ON THE CLUSTER.
# The cluster has no internet access, so it can neither clone nor push. This
# script exists to get the work into GitHub from the machine that downloaded
# the tarball; the cluster then receives files by SFTP, not by git.
#
# On Windows use Git Bash (ships with Git for Windows). Do NOT use cmd.exe or
# PowerShell -- this is a bash script, and Git Bash also gives you the same
# LF handling the repo's .gitattributes expects.
#
#   bash push_eyal_adapter.sh
#   bash push_eyal_adapter.sh --dry-run          # stage + show, do not push
#   TARBALL=~/Downloads/eyal_hpc_bundle.tar.gz bash push_eyal_adapter.sh
#
# What it does NOT commit, and why:
#   source/195667-master/   upstream ModelDB release; not ours to vendor, and
#                           it is ~5 MB of third-party code
#   archive/eyal_archive/   derived data. Regenerable from the builder in one
#                           Colab run, and .npz binaries bloat history. The
#                           metadata.json + comparison_targets.csv ARE small
#                           and are committed, so the published targets and
#                           the per-cell provenance stay in version control.
#
# The repo's .gitattributes already pins '* text=auto eol=lf' plus explicit
# rules for .py/.sh/.md, so a Windows checkout will not rewrite line endings
# under you. This script verifies that rather than assuming it.

set -euo pipefail

# --- config ------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/Leonardodm00/Towards-EEG.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/towards-eeg-push}"
TARBALL="${TARBALL:-$HOME/Downloads/eyal_hpc_bundle.tar.gz}"
DEST="Passive Features/Eyal 2016 Adapter"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

echo "repo    : $REPO_URL ($BRANCH)"
echo "tarball : $TARBALL"
echo "workdir : $WORKDIR"
echo

# --- 0. preconditions --------------------------------------------------------
command -v git >/dev/null 2>&1 || { echo "FATAL: git not found"; exit 1; }
if [ ! -f "$TARBALL" ]; then
    echo "FATAL: tarball not found: $TARBALL"
    echo "Download eyal_hpc_bundle.tar.gz from Drive first, or set TARBALL=..."
    echo "On Windows/Git Bash the Downloads folder is usually:"
    echo "    /c/Users/<you>/Downloads/eyal_hpc_bundle.tar.gz"
    exit 1
fi

# git identity: a commit with no author fails late, after all the copying.
if ! git config --get user.email >/dev/null 2>&1; then
    echo "FATAL: git has no user.email. Set it once:"
    echo "    git config --global user.email 'you@example.com'"
    echo "    git config --global user.name  'Your Name'"
    exit 1
fi
echo "author  : $(git config --get user.name) <$(git config --get user.email)>"

# GitHub stopped accepting account passwords over HTTPS in 2021. If the push
# at the end fails with 'Authentication failed', you need a Personal Access
# Token (Settings -> Developer settings -> Tokens, scope 'repo') and should
# use it as the PASSWORD at the prompt, or switch REPO_URL to SSH:
#     REPO_URL=git@github.com:Leonardodm00/Towards-EEG.git bash push_eyal_adapter.sh

# --- 1. clone (or reuse) -----------------------------------------------------
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ -d "Towards-EEG/.git" ]; then
    echo "[1/6] reusing existing clone"
    cd Towards-EEG
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
else
    echo "[1/6] cloning"
    git clone --branch "$BRANCH" "$REPO_URL" Towards-EEG
    cd Towards-EEG
fi
REPO_ROOT="$(pwd)"
echo "      at $REPO_ROOT"

# --- 2. confirm the line-ending policy is in force ---------------------------
echo "[2/6] checking .gitattributes"
if grep -q "eol=lf" .gitattributes 2>/dev/null; then
    echo "      OK: repo pins LF endings"
else
    echo "      WARNING: no eol=lf rule found. A Windows clone may have"
    echo "      rewritten line endings, which will corrupt the .sh files."
fi

# --- 3. unpack the bundle to a scratch dir -----------------------------------
echo "[3/6] unpacking $TARBALL"
SCRATCH="$WORKDIR/_unpacked"
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"
tar -xzf "$TARBALL" -C "$SCRATCH"

# Two tarball layouts are accepted, because two exist in circulation:
#
#   eyal_hpc_bundle.tar.gz      -> eyal_hpc_bundle/{tools,archive,source,...}
#       the full offline bundle built by make_hpc_bundle.py in Colab.
#       Carries the archive metadata as well as the tooling.
#
#   eyal_adapter_complete.tar.gz -> eyal_adapter/*.py, *.sh   (flat)
#       tooling only, no archive. Fine to push -- you just get no
#       archive_metadata/ directory in the commit.
#
# Detected rather than assumed, so you can push from whichever you have.
BUNDLE=""
LAYOUT=""
if [ -d "$SCRATCH/eyal_hpc_bundle/tools" ]; then
    BUNDLE="$SCRATCH/eyal_hpc_bundle"; LAYOUT="bundle"
else
    for d in "$SCRATCH"/*/; do
        if [ -f "${d}eyal_archive_builder.py" ]; then
            BUNDLE="${d%/}"; LAYOUT="flat"; break
        fi
    done
fi
if [ -z "$BUNDLE" ]; then
    echo "FATAL: unrecognised tarball layout. Expected either"
    echo "  eyal_hpc_bundle/tools/...   or   <dir>/eyal_archive_builder.py"
    echo "Found:"; ls -R "$SCRATCH" | head -20
    exit 1
fi
echo "      layout: $LAYOUT  ($BUNDLE)"

# verify the bundle survived the round trip before committing anything
if [ -f "$BUNDLE/CHECKSUMS.sha256" ] && command -v sha256sum >/dev/null 2>&1; then
    ( cd "$BUNDLE" && sha256sum -c CHECKSUMS.sha256 --quiet ) \
        && echo "      OK: bundle checksums match" \
        || { echo "FATAL: bundle is corrupted; re-download it"; exit 1; }
else
    echo "      (no CHECKSUMS.sha256 in this layout; skipping integrity check)"
fi

# --- 4. copy the parts we version-control ------------------------------------
echo "[4/6] staging into '$DEST'"
mkdir -p "$REPO_ROOT/$DEST/tools"
mkdir -p "$REPO_ROOT/$DEST/hpc"

if [ "$LAYOUT" = "bundle" ]; then
    SRC_TOOLS="$BUNDLE/tools"
    SRC_ROOT="$BUNDLE"
else
    SRC_TOOLS="$BUNDLE"
    SRC_ROOT="$BUNDLE"
fi

# Python tooling
for f in eyal_archive_builder.py eyal_reference_scalars.py \
         smoke_eyal_archive_builder.py smoke_eyal_neuron_build.py \
         patch_eyal_support.py regression_allen_unchanged.py \
         make_hpc_bundle.py Save_Eyal2016_Data.py; do
    [ -f "$SRC_TOOLS/$f" ] && cp "$SRC_TOOLS/$f" "$REPO_ROOT/$DEST/tools/"
    [ -f "$SRC_ROOT/$f" ]  && cp "$SRC_ROOT/$f"  "$REPO_ROOT/$DEST/tools/"
done
[ -f "$SRC_ROOT/verify_on_cluster.py" ] \
    && cp "$SRC_ROOT/verify_on_cluster.py" "$REPO_ROOT/$DEST/"
[ -f "$SRC_ROOT/README_HPC.md" ] \
    && cp "$SRC_ROOT/README_HPC.md" "$REPO_ROOT/$DEST/"

# cluster scripts, wherever they ended up
for f in run_all_smoke_tests.sh submit_eyal_fit.sh push_eyal_adapter.sh; do
    [ -f "$SRC_ROOT/$f" ]  && cp "$SRC_ROOT/$f"  "$REPO_ROOT/$DEST/hpc/"
    [ -f "$SRC_TOOLS/$f" ] && cp "$SRC_TOOLS/$f" "$REPO_ROOT/$DEST/hpc/"
done

# small, text-only provenance from the archive: the published targets and the
# per-cell metadata. The .npz arrays and the .asc morphologies stay out.
if [ -d "$BUNDLE/archive/eyal_archive" ]; then
    mkdir -p "$REPO_ROOT/$DEST/archive_metadata"
    cp "$BUNDLE"/archive/eyal_archive/comparison_targets.csv \
       "$REPO_ROOT/$DEST/archive_metadata/" 2>/dev/null || true
    cp "$BUNDLE"/archive/eyal_archive/manifest.json \
       "$REPO_ROOT/$DEST/archive_metadata/" 2>/dev/null || true
    for d in "$BUNDLE"/archive/eyal_archive/specimen_*; do
        [ -d "$d" ] || continue
        cp "$d/metadata.json" \
           "$REPO_ROOT/$DEST/archive_metadata/$(basename "$d").json"
    done
    echo "      archive metadata included"
else
    echo "      no archive in this tarball; committing tooling only"
fi

# --- 5. byte checks on everything we are about to commit ---------------------
# This is the whole point of the HPC discipline: a non-ASCII byte in a .py or
# a CR in a .sh is invisible in every editor and only surfaces as a failed
# job. It is checked here, before the commit, so the repo can never carry it.
#
# Python is preferred but NOT required. On Windows, `python3` often exists
# only as the Microsoft Store alias, which prints a message and exits
# non-zero -- so we test that the interpreter actually RUNS, not merely that
# the name resolves. Failing that, the pure-shell path below is exactly
# equivalent: tr -d '\000-\177' leaves only bytes > 0x7F, and tr -dc '\r'
# leaves only carriage returns.
echo "[5/6] byte checks"
cd "$REPO_ROOT"

PYBIN=""
for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "pass" >/dev/null 2>&1; then
        PYBIN="$c"
        break
    fi
done

if [ "${SKIP_BYTE_CHECK:-0}" = "1" ]; then
    echo "      SKIPPED (SKIP_BYTE_CHECK=1) -- committing UNVERIFIED bytes"
elif [ -n "$PYBIN" ]; then
    echo "      using $PYBIN"
    "$PYBIN" - "$DEST" <<'PYEOF'
import os, sys
dest = sys.argv[1]
bad_ascii, bad_crlf = [], []
for root, dirs, files in os.walk(dest):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        p = os.path.join(root, f)
        b = open(p, "rb").read()
        if f.endswith(".py") and any(c > 127 for c in b):
            bad_ascii.append(p)
        if f.endswith((".sh", ".pbs", ".slurm")) and b.count(b"\r"):
            bad_crlf.append(p)
for p in bad_ascii:
    print("  NON-ASCII .py :", p)
for p in bad_crlf:
    print("  CRLF shebang  :", p)
if bad_ascii or bad_crlf:
    print("REFUSING TO COMMIT -- fix the files above first")
    sys.exit(1)
print("  OK: every .py pure ASCII, every .sh LF-only")
PYEOF
else
    echo "      no working python found; using the POSIX shell fallback"
    _bad=0
    _n_py=0
    _n_sh=0
    while IFS= read -r p; do
        [ -f "$p" ] || continue
        case "$p" in
            */__pycache__/*) continue ;;
            *.py)
                _n_py=$((_n_py + 1))
                n=$(LC_ALL=C tr -d '\000-\177' < "$p" | wc -c | tr -d '[:space:]')
                if [ "${n:-0}" -gt 0 ]; then
                    echo "  NON-ASCII .py : $p ($n byte(s) > 0x7F)"
                    _bad=1
                fi
                ;;
            *.sh|*.pbs|*.slurm)
                _n_sh=$((_n_sh + 1))
                n=$(LC_ALL=C tr -dc '\r' < "$p" | wc -c | tr -d '[:space:]')
                if [ "${n:-0}" -gt 0 ]; then
                    echo "  CRLF shebang  : $p ($n CR byte(s))"
                    _bad=1
                fi
                ;;
        esac
    done <<EOF
$(find "$DEST" -type f)
EOF
    if [ "$_bad" -ne 0 ]; then
        echo "REFUSING TO COMMIT -- fix the files above first"
        exit 1
    fi
    echo "  OK: $_n_py .py pure ASCII, $_n_sh .sh LF-only"
fi

# --- 6. commit and push ------------------------------------------------------
echo "[6/6] commit"
git add -A -- "$DEST"
if git diff --cached --quiet; then
    echo "      nothing to commit; working tree already matches the bundle"
    exit 0
fi
git status --short -- "$DEST"

MSG="Eyal 2016 adapter: archive builder, 7 guarded pipeline patches, tests

Adds the data-adaptation layer that feeds the six Eyal et al. (2016) human
L2/3 cells into the passive-fitting pipeline, so that the fitted (Cm, Rm, Ra)
can be compared against an independent group's published values.

tools/
  eyal_archive_builder.py        ModelDB 195667 -> Phase 0 archive (no NEURON)
  eyal_reference_scalars.py      Eyal-faithful reference model; Rin*, areas
  smoke_eyal_archive_builder.py  13 tests (11 NEURON-free, 2 loader)
  smoke_eyal_neuron_build.py     .asc import, axon deletion, Rin* per cell
  patch_eyal_support.py          21 guarded edits, idempotent, --check/--revert
  regression_allen_unchanged.py  proves Allen behaviour is bit-identical

Pipeline patches (each defaults to current behaviour):
  1 Square-Subthreshold averaging may group by AMPLITUDE, not polarity alone
  2 brief-pulse VALIDATION window no longer hard-coded to (1, 100) ms
  3 morphology filename read from metadata['morphology_file']
  4 Import3d dispatched on suffix (.asc -> Import3d_Neurolucida3)
  5 require_long_square, for datasets with no Long Square sweeps
  6 CLI flags in run_biological_fit.py so 1-5 are reachable
  7 an EMPTY validation set yields 'not_evaluated', not 'failed'

Also fixes a pre-existing indexing bug: SweepBundle.sweep_numbers addressed
the full npz array with an index into the polarity-filtered pulse list, so
hyperpolarising bundles reported depolarising sweep numbers.

Archive .npz/.asc payloads are NOT committed (regenerable); per-cell
metadata.json and comparison_targets.csv are, as provenance."

git commit -m "$MSG"

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "--dry-run: committed locally, NOT pushed. Inspect with:"
    echo "    cd '$REPO_ROOT' && git show --stat"
    echo "Then push with:  git push origin $BRANCH"
    exit 0
fi

echo "pushing to $BRANCH ..."
git push origin "$BRANCH"
echo
echo "done: https://github.com/Leonardodm00/Towards-EEG/tree/$BRANCH/$DEST"
