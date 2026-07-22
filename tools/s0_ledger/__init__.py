"""
s0_ledger -- reconciliation ledger builder for Towards-EEG stage S0.0.

Three strictly separated layers:

    scan.py      pure I/O.  Walks a tree, reads bytes, computes measurements.
                 Makes no judgement about ancestry or verdicts.
    analyse.py   pure logic.  Consumes scan results plus a declared ancestor
                 map and assigns relationship / transform / verdict / scope.
                 Performs no I/O.
    render.py    pure output.  Emits LEDGER.md and ledger.csv from analysed
                 rows.  Performs no scanning and no classification.

The orchestrator is tools/build_ledger.py.

Source text is pure ASCII by policy (HPC transfer safety); see
tools/build_ledger.py --self-check.
"""

TOOL_NAME = "s0_ledger"
TOOL_VERSION = "1.0.0"

__all__ = ["TOOL_NAME", "TOOL_VERSION"]
