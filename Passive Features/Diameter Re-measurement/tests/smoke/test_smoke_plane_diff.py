"""Smoke test for the consecutive-plane differences -- Block 11 in specs/SPEC.md
(the user's proposal of 2026-10-08; a diagnostic, not a focus rule).

For planes I_k in increasing plane index and the pixels p counted (N of them):
    D_n(p) = I_{n+1}(p) - I_n(p)
    S+_n   = (1/N) sum_p max(D_n(p), 0)
    S-_n   = (1/N) sum_p max(-D_n(p), 0)
and the dip: the minimum of S+ between its two largest local maxima (the ends
count as maxima), else its global minimum.

Checks
    test_known_answer   a 3-plane 2 x 2 stack worked by hand (S+ 0.75, S- 0.5;
                        S+ 0, S- 0.25); the dip of hand-made curves, with NaN
    test_reference      a random stack with a pixel mask against loops written
                        out by hand
    test_convergence    skipped: nothing is discretised (the measure is exact
                        on the samples)
    test_invariants     reversing the plane order swaps S+ and S-; a constant
                        added to every plane changes nothing; a gain a > 0
                        scales both by a; S+ - S- equals the mean of D_n; a
                        blur that conserves each plane's total (Gaussian, wrap
                        boundary) gives S+ = S- to roundoff
    test_noise_floor    planes of white Gaussian noise of SD s: S+ and S- are
                        s / sqrt(pi) within 5 standard errors (the mean of the
                        positive part of a N(0, 2 s^2) variable)
    test_contract       shapes and NaN: an invalid plane makes its two pairs
                        NaN; a non-finite pixel counts only inside the mask;
                        refusals of a single plane, a bad valid, an empty mask
    test_determinism    a pure function: equal inputs, equal outputs
    test_edge_cases     a rendered thin tube (Block 4 renderer, no noise, d
                        0.5 um, centre in plane 0): the dip of S+ is a pair
                        next to plane 0; scripts/plane_differences.run on the
                        synthetic cell (fixtures_cell): planes +-8 about the
                        SWC plane with the two missing ones NaN, a PNG and a
                        record per node, the dip next to plane 0, the soma
                        refused and recorded as skipped, --band-um counts fewer
                        pixels

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_plane_diff.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (SRC, HERE, WS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from allen_diameter.analysis import focus as FO  # noqa: E402

SEED = 20261008
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")


def _hand_stack():
    I0 = [[0.0, 1.0], [2.0, 3.0]]
    I1 = [[1.0, 1.0], [0.0, 5.0]]
    I2 = [[1.0, 0.0], [0.0, 5.0]]
    return np.array([I0, I1, I2])


def test_known_answer():
    r = FO.plane_differences(_hand_stack())
    # D0 = [[1, 0], [-2, 2]]: positive 1 + 2 over 4 pixels, negative 2 over 4; D1 = [[0, -1], [0, 0]]
    assert np.allclose(r["pos"], [0.75, 0.0], atol=0, rtol=0) and np.allclose(r["neg"], [0.5, 0.25], atol=0, rtol=0), r
    assert np.array_equal(r["diff"][0], [[1.0, 0.0], [-2.0, 2.0]])
    # the dip: [3, 1, 2, 0.5, 4] has maxima at 0, 2, 4; the two largest are 4 (index 4) and 3 (index 0) -> index 3
    assert FO.difference_dip([3.0, 1.0, 2.0, 0.5, 4.0]) == 3
    assert FO.difference_dip([5.0, 4.0, 3.0, 2.0, 1.0]) == 4          # one maximum: the global minimum
    assert FO.difference_dip([3.0, float("nan"), 1.0, 4.0]) == 2      # NaN skipped
    assert FO.difference_dip([2.0, 2.0]) == 0                          # adjacent maxima: the global minimum (first)
    assert FO.difference_dip([float("nan")] * 3) is None


def test_reference():
    rng = np.random.default_rng(SEED)
    st = rng.normal(100.0, 10.0, size=(5, 7, 6))
    mask = rng.random((7, 6)) < 0.6
    r = FO.plane_differences(st, mask=mask)
    for n in range(4):
        sp = sn = 0.0
        cnt = 0
        for i in range(7):
            for j in range(6):
                if mask[i, j]:
                    d = float(st[n + 1, i, j]) - float(st[n, i, j])
                    sp += d if d > 0 else 0.0
                    sn += -d if d < 0 else 0.0
                    cnt += 1
        assert abs(r["pos"][n] - sp / cnt) < 1e-12 and abs(r["neg"][n] - sn / cnt) < 1e-12, (n, r["pos"][n], sp / cnt)


def test_convergence():
    raise unittest.SkipTest("nothing is discretised: the measure is exact on the samples")


def test_invariants():
    rng = np.random.default_rng(SEED + 1)
    st = rng.normal(150.0, 20.0, size=(6, 16, 16))
    r = FO.plane_differences(st)
    rev = FO.plane_differences(st[::-1])
    assert np.allclose(rev["pos"], r["neg"][::-1], rtol=0, atol=1e-12) and np.allclose(rev["neg"], r["pos"][::-1],
                                                                                          rtol=0, atol=1e-12)
    off = FO.plane_differences(st + 37.5)
    assert np.allclose(off["pos"], r["pos"], rtol=0, atol=1e-10) and np.allclose(off["neg"], r["neg"], rtol=0, atol=1e-10)
    g = FO.plane_differences(2.5 * st)
    assert np.allclose(g["pos"], 2.5 * r["pos"], rtol=1e-12) and np.allclose(g["neg"], 2.5 * r["neg"], rtol=1e-12)
    assert np.allclose(r["pos"] - r["neg"], np.diff(st, axis=0).mean(axis=(1, 2)), rtol=0, atol=1e-10)
    # a blur that conserves each plane's total: the positive and negative parts of every difference balance
    X = rng.gamma(2.0, 20.0, size=(64, 64))
    planes = np.stack([ndimage.gaussian_filter(X, s, mode="wrap") for s in (0.5, 1.0, 2.0, 3.0, 2.0, 1.0)])
    b = FO.plane_differences(planes)
    assert np.allclose(b["pos"], b["neg"], rtol=1e-9, atol=0), (b["pos"], b["neg"])


def test_noise_floor():
    rng = np.random.default_rng(SEED + 2)
    s, n_px = 3.0, 200 * 200
    r = FO.plane_differences(rng.normal(100.0, s, size=(5, 200, 200)))
    expect = s / math.sqrt(math.pi)
    se = s * math.sqrt(1.0 - 1.0 / math.pi) / math.sqrt(n_px)    # SD of max(D, 0), D ~ N(0, 2 s^2), over the pixels
    for x in np.concatenate([r["pos"], r["neg"]]):
        assert abs(x - expect) < 5.0 * se, (x, expect, se)


def test_contract():
    st = np.full((4, 3, 3), 10.0)
    st[2] += 1.0
    r = FO.plane_differences(st, valid=[True, True, False, True])
    assert r["diff"].shape == (3, 3, 3) and r["pos"].shape == (3,) and r["neg"].shape == (3,)
    assert r["pos"][0] == 0.0 and np.isnan(r["pos"][1]) and np.isnan(r["pos"][2]) and np.all(np.isnan(r["diff"][1]))
    st2 = st.copy()
    st2[1, 0, 0] = np.nan
    m = np.ones((3, 3), dtype=bool)
    assert np.isnan(FO.plane_differences(st2, mask=m)["pos"][0])
    m[0, 0] = False
    assert FO.plane_differences(st2, mask=m)["pos"][0] == 0.0
    for bad in (dict(stack=st[:1]), dict(stack=st, valid=[True, True]), dict(stack=st, mask=np.zeros((3, 3), bool)),
                dict(stack=st, mask=np.ones((2, 2), bool))):
        try:
            FO.plane_differences(**bad)
        except ValueError:
            continue
        raise AssertionError("plane_differences accepted %r" % (list(bad),))


def test_determinism():
    rng = np.random.default_rng(SEED + 3)
    st = rng.normal(0.0, 1.0, size=(4, 9, 9))
    a, b = FO.plane_differences(st), FO.plane_differences(st.copy())
    assert np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["neg"], b["neg"]) and \
        np.array_equal(a["diff"], b["diff"], equal_nan=True)


def test_edge_cases():
    import test_smoke_node_pipeline as T
    from allen_diameter.model import geometry as G
    # a rendered thin tube, no noise: the image is stationary at focus, so the dip sits next to plane 0
    cfg = T.small_cfg(noise_sd_gl=0.0, jpeg=False)
    tube = G.Tube((0.03, -0.02, 0.05), 0.25, 0.0, 0.5, 1.0, 9.0, "axial")
    p = cfg.acquisition.res0_um
    block, ks, valid, _ = T.make_provider([tube], [1.0], cfg, SEED)(int(math.floor(-3.5 / p)), int(math.floor(-3.5 / p)),
                                                                     int(math.ceil(7.0 / p)), int(math.ceil(7.0 / p)), -5, 5)
    r = FO.plane_differences(block.astype(float), valid)
    d = FO.difference_dip(r["pos"])
    assert ks[d] + 0.5 in (-0.5, 0.5), ("rendered thin tube: dip at %d->%d" % (ks[d], ks[d] + 1), r["pos"])
    # the script on the synthetic cell: planes +-8 about the SWC plane (the fixture has -7..7)
    import run_cell
    import plane_differences as PD
    from fixtures_cell import synthetic_cell
    from allen_diameter.config import default_config
    base = default_config()
    ccfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    with tempfile.TemporaryDirectory() as tmp:
        swc, fetcher, planes, _ = synthetic_cell(tmp, ccfg, d_true=0.8, mu=1.0)
        prov = run_cell.real_provider(fetcher, planes, ccfg.acquisition.res0_um)
        out = os.path.join(tmp, "pd")
        recs = PD.run(swc, prov, ccfg, [4, 1], out, "999", planes_half=8, log=lambda m: None)
        rec, skip = recs
        assert rec["ks"] == list(range(-8, 9)) and rec["n_missing"] == 2 and rec["valid"][0] is False, rec["ks"]
        assert math.isnan(rec["pos"][0]) and math.isnan(rec["pos"][-1]) and all(math.isfinite(x) for x in rec["pos"][1:-1])
        assert os.path.exists(rec["png"]) and rec["k_swc"] == 0 and rec["dip_pair"] in ([-1, 0], [0, 1]), rec["dip_pair"]
        assert "skipped" in skip and skip["node_id"] == 1
        with open(os.path.join(out, "planediff_999.json")) as f:
            assert len(json.load(f)) == 2
        band = PD.run(swc, prov, ccfg, [4], os.path.join(tmp, "band"), "999", planes_half=3, band_um=1.0,
                      log=lambda m: None)[0]
        assert 0 < band["n_pixels"] < rec["n_pixels"], (band["n_pixels"], rec["n_pixels"])


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
