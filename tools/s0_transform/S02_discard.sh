#!/usr/bin/env bash
# S02_discard.sh -- the deletion half of stage S0.2.
#
# Kept out of decolab.py deliberately: a tool that rewrites files should not
# also be able to delete them, and the deletion should read as a deliberate act
# in git history rather than as a side effect of a transform.
#
# Target: Population/population_CHANGED.py
#   Dead branch. The working-branch file of the same name was committed at S0.1
#   to its own ancestor path (HybridLFPy Tweaked/Population_multiMorph.py). Any
#   tree containing both resolves the import by whichever is seen last.
#   Verified: nothing in the tree imports this module.
#
# Run from the repository root, on branch s0-packaging.

set -euo pipefail

TARGET="Population/population_CHANGED.py"

if [ ! -f "${TARGET}" ]; then
    echo "already absent: ${TARGET}"
    exit 0
fi

echo "sha256 before removal:"
python3 - "${TARGET}" <<'EOF'
import hashlib, sys
p = sys.argv[1]
with open(p, "rb") as fh:
    print("  %s  %s" % (hashlib.sha256(fh.read()).hexdigest(), p))
EOF

# Re-verify the import claim at the moment of deletion rather than trusting
# a measurement taken earlier in the session.
echo "importers found (expect none):"
grep -rnI --include='*.py' -e 'population_CHANGED' . \
    | grep -v -e '^\./Population/population_CHANGED\.py' \
              -e '^\./tools/' \
    | grep -E 'import|from' || echo "  none"

git rm -- "${TARGET}"
echo "removed ${TARGET}"
echo "the ledger row keeps verdict=discard, stage=S0.2; stamp_phase.py will"
echo "record sha256_post_s02 = '-' for it."
