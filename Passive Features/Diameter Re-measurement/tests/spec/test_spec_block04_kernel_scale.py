"""Spec-derived tests of Block 4, amendments 1 and 2 (kernel.with_in_focus_width, kernel.with_kernel; tester,
2026-11-22). Written from specs/SPEC.md (Block 4 amendments 1 and 2; D-165, D-201) before the code was read.

Expected values are the amendments' formulas evaluated with numpy, and Block 4's kernel law as an earlier
tester wrote it (tests/spec/spec_helpers.ref_sigma_r); the code under test only produces what is checked.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from spec_helpers import ref_sigma_r  # noqa: E402

from allen_diameter import config as C  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402


def rcfg(**kw):
    r = C.default_config().renderer
    return dataclasses.replace(r, **kw) if kw else r


def test_with_in_focus_width_amendment1():
    r = rcfg()
    old = np.asarray(r.kernel_table_sigma_um)
    for s0 in (0.05, 0.20, 0.30):
        n = K.with_in_focus_width(r, s0)
        new = np.asarray(n.kernel_table_sigma_um)
        # new minus old variance = s0^2 - sigma_r(0)^2 at every knot (1e-15 um^2: a few float64 operations on 0.1)
        assert_allclose(new ** 2 - old ** 2, s0 ** 2 - old[0] ** 2, rtol=0, atol=1e-15)
        assert new[0] == s0 and n.sigma_r0_um == s0                      # exact
        assert (n.kernel_continuation, n.kernel_continuation_slope) == (r.kernel_continuation,
                                                                         r.kernel_continuation_slope)
        assert n.kernel_table_delta_um == r.kernel_table_delta_um
        dataclasses.replace(C.default_config(), renderer=n).validate()
    same = K.with_in_focus_width(r, float(old[0]))
    assert_allclose(same.kernel_table_sigma_um, old, rtol=0, atol=1e-15)
    falling = rcfg(kernel_table_sigma_um=(0.2, 0.1, 0.12, 0.26, 0.44, 0.6), sigma_r0_um=0.2)
    for bad_r, s0 in ((rcfg(kernel_family="empirical"), 0.1), (r, 0.0), (r, -0.1), (r, np.nan), (r, np.inf),
                      (falling, 0.05)):
        with pytest.raises(ValueError):
            K.with_in_focus_width(bad_r, s0)


@pytest.mark.parametrize("cont", ["linear", "frozen", "proportional"])
def test_with_kernel_amendment2(cont):
    r = rcfg(kernel_continuation=cont)
    old = np.asarray(r.kernel_table_sigma_um)
    for s0, c in ((0.15, 1.3), (0.08, 0.5), (0.3, 4.0), (0.12, 1.0)):
        n = K.with_kernel(r, s0, c)
        new = np.asarray(n.kernel_table_sigma_um)
        assert_allclose(new ** 2, c * (old ** 2 - old[0] ** 2) + s0 ** 2, rtol=1e-14)
        assert new[0] == s0 and n.sigma_r0_um == s0
        dataclasses.replace(C.default_config(), renderer=n).validate()
        # the law read back through sigma_r equals Block 4's law (independent) on the new table
        delta = np.concatenate([np.linspace(-3.0, 3.0, 241), [1e4]])
        assert_allclose(K.sigma_r(delta, n), ref_sigma_r(delta, n), rtol=1e-14, atol=0)
        if cont == "linear":
            assert_allclose(n.kernel_continuation_slope, r.kernel_continuation_slope * math.sqrt(c), rtol=1e-15)
            ratio = float(K.sigma_r(1e4, n)) / float(K.sigma_r(1e4, r))
            assert abs(ratio - math.sqrt(c)) <= 1e-3                     # where the growth dominates
        else:
            assert n.kernel_continuation_slope == r.kernel_continuation_slope
        if cont == "proportional":                                       # follows the new table's last knot
            assert_allclose(K.sigma_r(2.0, n), new[-1] * 2.0 / r.kernel_table_delta_um[-1], rtol=1e-14)
        if cont == "frozen":
            assert_allclose(K.sigma_r(5.0, n), new[-1], rtol=0, atol=0)
    one = K.with_kernel(r, 0.2, 1.0)
    ref = K.with_in_focus_width(r, 0.2)
    assert dataclasses.asdict(one) == dataclasses.asdict(ref)            # c = 1: every field equal, bit for bit


def test_with_kernel_refusals():
    r = rcfg()
    falling = rcfg(kernel_table_sigma_um=(0.2, 0.1, 0.12, 0.26, 0.44, 0.6), sigma_r0_um=0.2)
    for args in ((rcfg(kernel_family="empirical"), 0.1, 1.0), (r, 0.0, 1.0), (r, 0.1, 0.0), (r, 0.1, -1.0),
                 (r, np.nan, 1.0), (r, 0.1, np.inf), (falling, 0.05, 1.0)):
        with pytest.raises(ValueError):
            K.with_kernel(*args)
    r0 = rcfg()
    K.with_kernel(r0, 0.2, 2.0)
    assert r0 == rcfg()                                                  # the input is a frozen value, unchanged
