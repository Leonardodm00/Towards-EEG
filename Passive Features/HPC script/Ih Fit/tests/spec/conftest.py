"""Shared set-up for the spec-derived tests (integration tester, 2026-11-17).

The code base under test is two directories up (Ih Fit/). The synthetic cell
is written with smoke_noise_windows.write_fixture (Block 3's writer-like
fixture, the same writer the smoke suites use) and read back through Block 3's
own pipeline; only the INPUT is built with it, every expected value in the
tests is computed independently.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
sys.dont_write_bytecode = True


@pytest.fixture(scope="session")
def cell(tmp_path_factory):
    import noise_fit as nf
    import noise_windows as nw
    import smoke_noise_windows as sw
    d = tmp_path_factory.mktemp("cell") / "specimen_1"
    truth = sw.write_fixture(d, np.random.default_rng(2026), sigma_b=0.1, ls_tail_n=sw.LS_LONG_TAIL_N)
    ss = nw.load_ss_windows(d)
    ls = nw.load_ls_prewindows(d)
    nwin = nw.ss_noise_windows(ss)
    lay = nf.CellLayout.from_cell(nwin, ls)
    tg = nf.CellTarget.from_cell(nwin, ls, lay)
    return {"dir": d, "truth": truth, "ss": ss, "ls": ls, "nwin": nwin, "layout": lay, "target": tg,
            "law": sw.THETA_CF}
