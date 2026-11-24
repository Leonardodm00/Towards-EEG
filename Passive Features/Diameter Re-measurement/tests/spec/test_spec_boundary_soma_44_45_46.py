"""Boundaries of wave 5 (tester, 2026-11-22, incremental run at 97f5104): Block 45 -> Block 46 (the renderer as the
fit's forward model, SPEC Block 46 "Data flow": forward_factory = lambda z: SomaRenderer(crop, p, z, rcfg, scfg,
acquisition.light_direction).render) and Block 44 -> Block 46 (the crop `soma_<S>_block.npz` and `soma_<S>.json` of
scripts/show_soma.py, cut to the region's bounding box with a margin, and a mask from Block 44's segment_mask widened
by soma_fit.mask_widen_um). Block 47, the caller, is not built: the wiring below is the one SPEC states.

    python3 -m pytest -q -p no:cacheprovider tests/spec/test_spec_boundary_soma_44_45_46.py

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter import config as C                           # noqa: E402
from allen_diameter.analysis import soma_fit as SF               # noqa: E402
from allen_diameter.analysis import soma_view as SV              # noqa: E402
from allen_diameter.model import soma_render as SR               # noqa: E402

P = 0.1144
DZ = 0.28


def test_renderer_as_the_fits_forward_recovers_its_own_truth():
    """A noise-free stack rendered by Block 45 (f = 1, the default kernel) is fitted by Block 46 with Block 45 as the
    forward model, wired as SPEC states; the planes reach the renderer as ks[K] dz in ascending order."""
    cfg = C.default_config()
    rcfg = cfg.renderer
    # [2026-11-24, D-240] far_slope "fit": the fitted-slope path, an opt-in comparison since D-240
    scfg = dataclasses.replace(cfg.soma_fit, far_slope="fit", plane_half_um=3.0, plane_step=2,
                               region_radius_um=4.0, max_nfev=150,
                               ftol=1e-10, xtol=1e-10)
    light = cfg.acquisition.light_direction
    crop = (-42, -40, 85, 81)
    ks = np.arange(-14, 15)
    truth = np.array([0.15, -0.1, 0.3, 2.2, 1.7, 1.5, 0.75, 0.3, 0.9])
    tau = SR.SomaRenderer(crop, P, ks * DZ, rcfg, cfg.soma_fit, light).render(truth[:8], truth[8])
    B = 110.0 + 3.0 * np.cos(ks)
    D = B[:, None, None] * tau
    seen = []

    def factory(z):
        seen.append(np.array(z, dtype=float))
        return SR.SomaRenderer(crop, P, z, rcfg, cfg.soma_fit, light).render

    centre, rs = (truth[0] + 0.5, truth[1] - 0.5, truth[2] + 0.56), 1.15 * 1.8
    fit = SF.fit_soma(D, ks, np.ones(ks.size, bool), crop[0], crop[1], P, DZ, centre, rs, np.zeros(D.shape[1:], bool),
                      factory, scfg, rcfg.kernel_continuation_slope)
    err = np.abs(fit.theta - truth)
    print("B45->46 (tester): status %s, %d evaluations, %.1f s, |error| %s" % (fit.status, fit.n_eval, fit.seconds,
                                                                             np.array2string(err, precision=2)))
    assert len(seen) == 1
    assert np.all(np.diff(seen[0]) > 0)
    assert_array_equal(seen[0], fit.ks * DZ)
    assert fit.status == "converged", fit.message
    # tolerance: the model is piecewise smooth in its slab grid (the slabs start at z0 - c), so trf may stop a little
    # short of the exact minimum; 1e-3 in each unit is far below the noise-level SEs (about 1e-2 um at 1 % noise)
    assert np.all(err <= 1e-3), err
    assert np.all(np.abs(fit.B_gl - B[np.isin(ks, fit.ks)]) <= 1e-3 * B[np.isin(ks, fit.ks)])


def test_fixed_slope_under_a_non_linear_continuation():
    """far_slope "fixed" passes slope0 to the forward model on every call (Block 46: 'slope a float ... slope0 when
    held'), and Block 45 refuses any slope unless the continuation is "linear". Under a "frozen" or "proportional"
    kernel the production wiring therefore cannot run, even with the slope held."""
    cfg = C.default_config()
    rcfg = dataclasses.replace(cfg.renderer, kernel_continuation="frozen")
    scfg = dataclasses.replace(cfg.soma_fit, far_slope="fixed", plane_half_um=1.0, plane_step=1, region_radius_um=2.0,
                               max_nfev=2)
    crop = (-20, -20, 41, 41)
    ks = np.arange(-5, 6)
    D = np.full((ks.size, 41, 41), 100.0)
    D[:, 15:26, 15:26] = 80.0
    factory = lambda z: SR.SomaRenderer(crop, P, z, rcfg, cfg.soma_fit, +1).render     # noqa: E731
    with pytest.raises(ValueError, match="linear"):
        SF.fit_soma(D, ks, np.ones(ks.size, bool), crop[0], crop[1], P, DZ, (0.0, 0.0, 0.0), 1.0,
                    np.zeros((41, 41), bool), factory, scfg, 0.0)


@pytest.fixture(scope="module")
def soma_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("soma44")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    res = subprocess.run([sys.executable, str(WS / "scripts" / "show_soma.py"), "--specimen", "7", "--synthetic",
                          "--half-um", "16", "--out", str(out)], cwd=str(WS), env=env, capture_output=True, text=True,
                         timeout=900)
    assert res.returncode == 0, res.stdout + res.stderr
    return out


def test_block44_outputs_feed_block46(soma_outputs):
    """The keys Block 46 reads (block, ks, valid, left, top, res0_um; centre_um, radius_um, dz_um) exist with the
    types it accepts; the crop cut to the region's bounding box with a margin and a widened trace mask run through
    fit_soma with Block 45 as the forward (a plumbing run: the synthetic stack holds no soma, one solver step)."""
    cfg = C.default_config()
    # [2026-11-24, D-241] the explicit-radius path (a number overrides the traced radius) with a widened mask: under
    # the defaults (region = the traced radius, no widening) the synthetic cell has no traced branch inside the cut
    # (the soma-rooted segments are left out by Block 44), the mask is empty and this plumbing check would test
    # nothing; the default path is run end to end in test_spec_e2e_fit_soma.py
    q = dataclasses.replace(cfg.soma_fit, region_radius_um=13.0, mask_widen_um=2.0)
    rec = json.load(open(soma_outputs / "soma_7.json"))
    z = np.load(soma_outputs / "soma_7_block.npz")
    block, ks, valid = z["block"], z["ks"], z["valid"]
    left, top, p = int(z["left"]), int(z["top"]), float(z["res0_um"])
    centre, radius, dz = rec["centre_um"], rec["radius_um"], rec["dz_um"]
    assert block.dtype == np.uint8 and ks.dtype.kind == "i" and valid.dtype == bool
    assert p == cfg.acquisition.res0_um and dz == cfg.acquisition.dz_um
    # the cut: the region's bounding box plus a margin of 1 um, in the crop's pixel frame
    # [2026-11-24, D-241] region_radius_um None (the default) = the traced soma radius, as fit_soma.py resolves it
    r_reg = float(radius) if q.region_radius_um is None else float(q.region_radius_um)
    margin = 1.0
    c0 = max(0, int(math.floor((centre[0] - r_reg - margin) / p)) - left)
    c1 = min(block.shape[2], int(math.ceil((centre[0] + r_reg + margin) / p)) - left + 1)
    r0 = max(0, int(math.floor((centre[1] - r_reg - margin) / p)) - top)
    r1 = min(block.shape[1], int(math.ceil((centre[1] + r_reg + margin) / p)) - top + 1)
    sub = block[:, r0:r1, c0:c1]
    sl, st = left + c0, top + r0
    # the trace mask (Block 44) on the cut, widened by mask_widen_um (0 by default since D-241)
    from allen_diameter.analysis import cell                               # noqa: E402
    seg = None
    box = (sl * p, (sl + sub.shape[2]) * p, st * p, (st + sub.shape[1]) * p)
    import clean_cell                                                         # the synthetic SWC, as show_soma reads it
    ns = type("A", (), dict(synthetic=True, specimen="7", swc=None, cache_dir=None, shift_x=None, shift_y=None,
                            flip_h=None, z0=None))()
    swc, _prov, _lab = clean_cell.load_cell(ns, cfg, lambda m: None, need_images=False)
    xyz = cell.to_image_um(swc.xyz, p, **clean_cell.transform_of(ns))
    index_of = {int(i): k for k, i in enumerate(swc.ids)}
    parent_index = np.array([index_of.get(int(pp), -1) for pp in swc.parent])
    seg = SV.trace_segments(xyz, swc.radius, parent_index, np.asarray(swc.types), box)
    mask = SV.segment_mask(sub.shape[1:], sl, st, p, seg["p1"], seg["p2"], seg["r1"] + q.mask_widen_um,
                           seg["r2"] + q.mask_widen_um)
    assert mask.shape == sub.shape[1:] and mask.dtype == bool and mask.any()
    seen = []

    def factory(zp):
        seen.append(np.array(zp))
        return SR.SomaRenderer((sl, st, sub.shape[2], sub.shape[1]), p, zp, cfg.renderer, q,
                               cfg.acquisition.light_direction).render

    scfg = dataclasses.replace(q, max_nfev=1)
    fit = SF.fit_soma(sub, ks, valid, sl, st, p, dz, centre, radius, mask, factory, scfg,
                      cfg.renderer.kernel_continuation_slope)
    assert fit.status in ("converged", "max_nfev", "failed")
    want = SF.select_planes(ks, valid, centre[2], dz, q.plane_half_um, q.plane_step)
    assert_array_equal(fit.ks, ks[want])
    assert_array_equal(seen[0], ks[want] * dz)
    print("B44->46 (tester): %d planes %s, region %d px, status %s" % (fit.ks.size, list(fit.ks),
                                                                   fit.n_samples // max(fit.ks.size, 1), fit.status))
