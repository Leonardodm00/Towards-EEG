"""Spec-derived tests of Block 5, amendments 2 and 3 (measure_pilot's d_fixed and measure.pilot_diameter; D-173,
D-174, D-178; tester, 2026-11-22). Written from specs/SPEC.md before the code was read.

Oracles: "d_fixed=None reads cfg.measure.pilot_diameter"; "an explicit argument replaces the configuration";
"allen" holds the diameter at twice the branch's traced radius at the node; a radius that is not finite and > 0
is fitted and flagged allen_radius_missing. The phantoms are rendered by the smoke suite's provider
(tests/smoke/test_smoke_node_pipeline.make_provider: Block 4 behind the fetch_zblock contract), a fixture only.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis import node_pipeline as NP  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
import test_smoke_node_pipeline as SN  # noqa: E402  (fixture: the phantom provider)


def _fields(res):
    """Every NodeResult field as a comparable tuple (arrays by value, the fit by its own fields)."""
    out = []
    for f in dataclasses.fields(res):
        x = getattr(res, f.name)
        if dataclasses.is_dataclass(x):
            x = _fields(x)
        elif isinstance(x, np.ndarray):
            x = (x.shape, x.tobytes())
        elif isinstance(x, float) and math.isnan(x):
            x = "nan"
        out.append((f.name, x))
    return tuple(out)


@pytest.fixture(scope="module")
def scene():
    cfg = C.default_config()
    d_true, phi, theta, c = 0.9, 0.0, 0.5, (0.02, 0.01, 0.04)
    tube = G.Tube(c, 0.5 * d_true, phi, theta, 1.0, 6.0, "axial")
    return cfg, tube, (c, phi, theta)


def _run(scene, radius, cfg=None, d_fixed=None):
    cfg0, tube, (c, phi, theta) = scene
    br = SN.phantom_branch(c, phi, theta, radius)
    return NP.measure_pilot(br, 4, SN.make_provider([tube], [1.0], cfg0, 3), cfg or cfg0, d_fixed=d_fixed)


def test_default_is_allen_and_the_argument_wins(scene):
    cfg = scene[0]
    free_cfg = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, pilot_diameter="free"))
    default = _run(scene, 0.35)
    held = _run(scene, 0.35, d_fixed="allen")
    assert _fields(default) == _fields(held)                        # the whole NodeResult, exactly
    assert held.d_hat_um == 0.7 and held.fit.d_hat_um == 0.7          # 2 x 0.35, exactly
    by_cfg = _run(scene, 0.35, cfg=free_cfg)
    by_arg = _run(scene, 0.35, d_fixed="free")
    assert _fields(by_cfg) == _fields(by_arg)
    assert by_arg.d_hat_um != 0.7
    over = _run(scene, 0.35, cfg=free_cfg, d_fixed="allen")         # an explicit argument replaces the configuration
    assert _fields(over) == _fields(held)


def test_missing_radius_is_fitted_and_flagged(scene):
    """A traced radius of 0 (finite, not > 0) reaches the hold's rule under the default focus window."""
    res = _run(scene, 0.0)
    assert "allen_radius_missing" in res.flags
    assert math.isfinite(res.d_hat_um) and res.d_hat_um > 0.5       # fitted freely (the tube is 0.9 um)


def test_non_finite_radius_reaches_the_flag_only_without_the_radius_window(scene):
    """Amendment 3 says a radius "not finite and > 0" is fitted and flagged. Under the default focus window
    ("radius_margin", Block 5: a missing, non-finite or negative radius raises ValueError) a NaN or negative
    radius stops the pilot before the hold; under "whole_profile" a NaN radius is fitted and flagged. Recorded
    as a problem in the spec (the two rules meet), not as a failure of either."""
    cfg = scene[0]
    whole = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, focus_grad_window="whole_profile"))
    res = _run(scene, float("nan"), cfg=whole)
    assert "allen_radius_missing" in res.flags and res.d_hat_um > 0.5
    for r in (float("nan"), -0.2):
        with pytest.raises(ValueError):
            _run(scene, r)
