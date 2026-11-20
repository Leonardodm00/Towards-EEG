"""Smoke test for kernel.with_kernel -- Block 4, amendment 2 in specs/SPEC.md (D-201).

Checks
    test_known_answer   the knots' variances are c (sigma_m^2 - sigma_0^2) + s0^2; the first knot is s0
    test_reference      growth_scale 1 equals with_in_focus_width field for field, bit for bit
    test_convergence    skipped (no discretisation)
    test_invariants     beyond the last knot: the linear slope times sqrt(c), sigma' / sigma -> sqrt(c) far out;
                        frozen stays frozen; proportional follows the new table; the result validates as a config
    test_contract       a RendererConfig is returned; the input is unchanged; sigma_r reads it
    test_determinism    two calls are equal
    test_edge_cases     the refusals

Run (from the code folder "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_kernel_scale.py
Pure ASCII, for the cluster.
"""
from __future__ import annotations

import dataclasses
import math
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter import config as C  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402

RC = C.default_config().renderer


def _raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised" % exc.__name__)


def test_known_answer():
    old = np.asarray(RC.kernel_table_sigma_um)
    for s0, c in ((0.15, 1.3), (0.25, 0.6), (0.08, 2.0)):
        new = np.asarray(K.with_kernel(RC, s0, c).kernel_table_sigma_um)
        assert new[0] == s0
        assert_allclose(new ** 2, c * (old ** 2 - old[0] ** 2) + s0 ** 2, rtol=1e-14, atol=1e-15)


def test_reference():
    for s0 in (0.05, 0.080, 0.2, 0.31):
        a, b = K.with_kernel(RC, s0, 1.0), K.with_in_focus_width(RC, s0)
        for f in dataclasses.fields(a):
            assert getattr(a, f.name) == getattr(b, f.name), f.name      # bit for bit, every field


def test_convergence():
    raise unittest.SkipTest("a table re-anchored at the knots: no discretisation")


def test_invariants():
    c, s0 = 1.7, 0.18
    new = K.with_kernel(RC, s0, c)
    assert RC.kernel_continuation == "linear"
    assert_allclose(new.kernel_continuation_slope, RC.kernel_continuation_slope * math.sqrt(c), rtol=1e-15)
    far = 1.0e4                                   # um: where the continuation dominates both tables
    assert_allclose(K.sigma_r(far, new) / K.sigma_r(far, RC), math.sqrt(c), rtol=1e-3)
    dm = RC.kernel_table_delta_um[-1]
    d1, d2 = dm + 0.3, dm + 0.9                   # the slope beyond the last knot
    assert_allclose((K.sigma_r(d2, new) - K.sigma_r(d1, new)) / 0.6, new.kernel_continuation_slope, rtol=1e-12)
    for kind in ("frozen", "proportional"):
        rc = dataclasses.replace(RC, kernel_continuation=kind)
        nk = K.with_kernel(rc, s0, c)
        assert nk.kernel_continuation == kind and nk.kernel_continuation_slope == rc.kernel_continuation_slope
        tab = np.asarray(nk.kernel_table_sigma_um)
        want = tab[-1] if kind == "frozen" else tab[-1] + tab[-1] / dm * 0.5
        assert_allclose(K.sigma_r(dm + 0.5, nk), want, rtol=1e-12)
    cfg = C.with_overrides(C.default_config(), {"renderer.sigma_r0_um": new.sigma_r0_um,
                                                "renderer.kernel_table_sigma_um": list(new.kernel_table_sigma_um),
                                                "renderer.kernel_continuation_slope": new.kernel_continuation_slope})
    assert cfg.renderer == new                    # the result is a valid configuration
    s = K.sigma_r(np.array([0.0, 0.14, 0.3]), new)    # monotone where the table is
    assert np.all(np.diff(s) > 0)


def test_contract():
    keep = dataclasses.asdict(RC)
    out = K.with_kernel(RC, 0.2, 0.8)
    assert isinstance(out, type(RC)) and dataclasses.asdict(RC) == keep
    assert out.sigma_r0_um == 0.2 and len(out.kernel_table_sigma_um) == len(RC.kernel_table_sigma_um)
    assert out.kernel_table_delta_um == RC.kernel_table_delta_um


def test_determinism():
    assert K.with_kernel(RC, 0.17, 1.4) == K.with_kernel(RC, 0.17, 1.4)


def test_edge_cases():
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        _raises(ValueError, K.with_kernel, RC, bad, 1.0)
        _raises(ValueError, K.with_kernel, RC, 0.1, bad)
    _raises(ValueError, K.with_kernel, dataclasses.replace(RC, kernel_family="empirical"), 0.1, 1.0)
    shrink = dataclasses.replace(RC, kernel_table_sigma_um=(0.2, 0.1) + tuple(RC.kernel_table_sigma_um[2:]))
    _raises(ValueError, K.with_kernel, shrink, 0.05, 5.0)      # a knot narrower than focus: variance <= 0


def main():
    checks = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    print("== %s" % Path(__file__).name)
    counts = dict(PASS=0, FAIL=0, ERROR=0, SKIP=0)
    for name, func in checks:
        t0 = time.perf_counter()
        try:
            func(); st, msg = "PASS", ""
        except unittest.SkipTest as e:
            st, msg = "SKIP", str(e)
        except AssertionError as e:
            st, msg = "FAIL", str(e) or traceback.format_exc().strip().splitlines()[-2]
        except Exception:
            st, msg = "ERROR", traceback.format_exc()
        counts[st] += 1
        print("%-5s  %-18s %7.3fs  %s" % (st, name, time.perf_counter() - t0, msg.splitlines()[0] if msg else ""))
    print("-- %d pass, %d fail, %d error, %d skip" % (counts["PASS"], counts["FAIL"], counts["ERROR"], counts["SKIP"]))
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
