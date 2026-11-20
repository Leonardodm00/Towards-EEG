# -*- coding: utf-8 -*-
"""
run_twin_recovery.py -- the Stage-7 run on the twins: plan, one twin (build and fit), merge (Block 8)
=====================================================================================================

specs/SPEC.md, the entry of Block 8 (D-184, D-185, D-188, D-182.3). Entry
point: wiring only. Sub-commands (the contract is the entry's "Public API"
and "Outputs"):

    python run_twin_recovery.py plan  --group-dir <dir> --noise-dir <dir> --out <dir> [options]
    python run_twin_recovery.py list  --out <dir>
    python run_twin_recovery.py twin  --out <dir> --twin <id> [--force]
    python run_twin_recovery.py merge --out <dir> [--no-plot]

CONTRACT STUB (contract: wave 2b). Written by Block 8's implementer.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    raise NotImplementedError("Block 8")


if __name__ == "__main__":
    sys.exit(main())
