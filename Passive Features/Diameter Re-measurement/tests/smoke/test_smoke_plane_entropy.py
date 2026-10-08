"""Smoke test for the entropy evaluation -- Block 11 in specs/SPEC.md (the
user's proposal of 2026-10-08, 17:07; a diagnostic, not a focus rule).

For N finite values x_i (grey levels) and a bin width w, value x goes to bin
j = floor(x / w + 1/2), the half-open interval [(j - 1/2) w, (j + 1/2) w);
with n_j values in bin j and p_j = n_j / N,
    H = -sum over occupied bins of p_j log2 p_j      [bits]
(focus.histogram_entropy, the plug-in estimate). focus.plane_entropies takes,
plane by plane, H of the bilinear samples along the measuring line and H of
the pixels of the strip around it (profiles.stripe_mask: |v| <= h across the
branch, |u| <= s along it); the script frames each curve's dip
(focus.difference_dip).

Checks
    test_known_answer   eight distinct levels: 3 bits; one level: 0; two
                        levels 1:3: 0.811278 bits (closed form); bins by
                        hand at the half-integer boundaries (0.5 goes up,
                        -0.5 goes up) and with w = 2; grey_histogram's centres
                        and counts; stripe_mask by hand, axis-aligned and
                        turned by 90 degrees (boundary pixels included)
    test_reference      random values against a count written out with a dict
                        and math.log2; stripe_mask against a pixel loop;
                        plane_entropies against per-plane calls
    test_convergence    the plug-in mean of a rounded Gaussian (SD 3 gl) over
                        800 draws: a tenfold N (100 -> 1000) shrinks the bias
                        5- to 15-fold (to first order it falls as 1/N)
    test_invariants     adding a multiple of w, permuting, or scaling values
                        and w together leave H unchanged; a gain of 2 on a
                        wide Gaussian (SD 20 gl, 1e6 values) adds 1 bit within
                        0.01 (the continuous limit: h(aX) = h(X) + log2 a)
    test_noise_floor    1e6 values of a rounded Gaussian (SD 3 gl, mean 0.3):
                        H within 0.005 bits of the exact entropy of the
                        discretised distribution (sum over the normal CDF)
    test_contract       types; NaN dropped; (NaN, 0, 0) with nothing finite;
                        refusals of w <= 0; plane_entropies: an invalid plane
                        or a non-finite sample used makes that entropy NaN
                        with N = 0, a non-finite pixel outside the strip does
                        not; refusals of mismatched shapes and an empty strip
    test_determinism    pure functions: equal inputs, equal outputs
    test_edge_cases     a rendered thin tube (Block 4 renderer, no noise, d
                        0.5 um, centre in plane 0): both entropies dip at
                        plane 0; scripts/plane_differences.run --evaluation
                        entropy on the synthetic cell (fixtures_cell): planes
                        +-8 with the two missing ones NaN, 53 samples and 917
                        pixels in every other plane, planeentropy_*.png and
                        .json, both dips at plane 0 (k* is 0 there), the soma
                        refused; a plane shifted by +3 or -5 grey levels after
                        the camera leaves every entropy unchanged (exactly);
                        made 2 % darker it moves the strip's entropy by less
                        than 0.05 bits; a strip of half-width 0.5 um holds
                        fewer pixels

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_plane_entropy.py

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
from scipy import stats

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (SRC, HERE, WS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from allen_diameter.analysis import focus as FO  # noqa: E402
from allen_diameter.analysis import profiles as PR  # noqa: E402

SEED = 20261008
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")


def _frame(left=0, top=0, p=1.0):
    from allen_image_io import CropFrame
    return CropFrame(left, top, 0, p)


def _dict_entropy(values, w=1.0):
    """Reference: counts in a dict, -sum p log2 p in a loop."""
    counts = {}
    for x in values:
        if math.isfinite(x):
            j = math.floor(x / w + 0.5)
            counts[j] = counts.get(j, 0) + 1
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values()), n, len(counts)


def _discretised_gaussian_entropy(mu, s, w=1.0):
    j = np.arange(math.floor((mu - 15 * s) / w), math.ceil((mu + 15 * s) / w) + 1)
    pj = stats.norm.cdf(((j + 0.5) * w - mu) / s) - stats.norm.cdf(((j - 0.5) * w - mu) / s)
    pj = pj[pj > 0]
    return float(-(pj * np.log2(pj)).sum())


def test_known_answer():
    assert FO.histogram_entropy(np.arange(8)) == (3.0, 8, 8)
    assert FO.histogram_entropy([5.0, 5.0, 5.0]) == (0.0, 3, 1)
    h, n, m = FO.histogram_entropy([0.0, 1.0, 1.0, 1.0])
    assert abs(h - (-(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75)))) < 1e-12 and (n, m) == (4, 2), h
    assert abs(h - 0.8112781244591328) < 1e-12, h
    # the boundaries: bin j holds [j - 1/2, j + 1/2)
    assert FO.histogram_entropy([0.4, 0.6])[0] == 1.0             # bins 0 and 1
    assert FO.histogram_entropy([0.49, -0.49])[0] == 0.0          # both in bin 0
    assert FO.histogram_entropy([0.5, 1.49])[0] == 0.0            # both in bin 1: 0.5 goes up
    assert FO.histogram_entropy([-0.5, 0.49])[0] == 0.0           # both in bin 0: -0.5 goes up
    # w = 2: 0 and 0.9 -> bin 0, 1.1 -> bin 1, 3 -> bin 2; p = 1/2, 1/4, 1/4 -> 1.5 bits
    assert FO.histogram_entropy([0.0, 0.9, 1.1, 3.0], bin_width=2.0) == (1.5, 4, 3)
    c, k = FO.grey_histogram([3.0, 5.0, 5.0])
    assert c.tolist() == [3.0, 4.0, 5.0] and k.tolist() == [1, 0, 2], (c, k)
    c, k = FO.grey_histogram([0.0, 0.9, 1.1, 3.0], bin_width=2.0)
    assert c.tolist() == [0.0, 2.0, 4.0] and k.tolist() == [2, 1, 1], (c, k)
    # the strip by hand: pixel centres at (col, row) um; origin at pixel (4, 3); across = y, along = x
    m = PR.stripe_mask((7, 9), _frame(), (4.0, 3.0), (0.0, 1.0), (1.0, 0.0), 2.0, 1.0)
    want = np.zeros((7, 9), bool)
    want[1:6, 3:6] = True                                          # |dy| <= 2 (rows 1..5), |dx| <= 1 (cols 3..5)
    assert m.dtype == bool and np.array_equal(m, want), m.astype(int)
    m90 = PR.stripe_mask((7, 9), _frame(), (4.0, 3.0), (-1.0, 0.0), (0.0, 1.0), 2.0, 1.0)
    want90 = np.zeros((7, 9), bool)
    want90[2:5, 2:7] = True                                        # |dx| <= 2 (cols 2..6), |dy| <= 1 (rows 2..4)
    assert np.array_equal(m90, want90), m90.astype(int)
    # a frame offset and a pixel size: the same strip, shifted
    m2 = PR.stripe_mask((7, 9), _frame(10, 20, 0.5), (7.0, 11.5), (0.0, 1.0), (1.0, 0.0), 1.0, 0.5)
    want2 = np.zeros((7, 9), bool)
    want2[1:6, 3:6] = True                                         # centre col 4 (x 7.0), row 3 (y 11.5)
    assert np.array_equal(m2, want2), m2.astype(int)


def test_reference():
    rng = np.random.default_rng(SEED)
    for w in (1.0, 2.0, 0.5):
        for x in (np.round(rng.normal(120.0, 6.0, 300)), rng.normal(120.0, 6.0, 300), rng.uniform(0, 255, 50)):
            h, n, m = FO.histogram_entropy(x, w)
            hr, nr, mr = _dict_entropy(x.tolist(), w)
            assert abs(h - hr) < 1e-12 and (n, m) == (nr, mr), (w, h, hr, n, nr, m, mr)
    th = 0.7
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    fr = _frame(-12, 5, 0.25)
    o = (-1.3, 3.1)
    m = PR.stripe_mask((40, 50), fr, o, y_hat, e_u, 3.0, 1.0)
    for r in range(40):
        for c in range(50):
            dx, dy = (fr.left + c) * 0.25 - o[0], (fr.top + r) * 0.25 - o[1]
            inside = abs(dx * y_hat[0] + dy * y_hat[1]) <= 3.0 + 1e-9 and abs(dx * e_u[0] + dy * e_u[1]) <= 1.0 + 1e-9
            assert m[r, c] == inside, (r, c)
    assert 0 < m.sum() < m.size
    st = np.round(rng.normal(100.0, 5.0, size=(4, 40, 50)))
    prof = rng.normal(100.0, 5.0, size=(4, 21))
    e = FO.plane_entropies(prof, st, m)
    for k in range(4):
        assert e["h_line"][k] == FO.histogram_entropy(prof[k])[0] and e["h_strip"][k] == FO.histogram_entropy(st[k][m])[0]
        assert e["n_line"][k] == 21 and e["n_strip"][k] == m.sum()


def test_convergence():
    rng = np.random.default_rng(SEED + 1)
    ref = _discretised_gaussian_entropy(0.3, 3.0)
    bias = {}
    for N in (100, 1000):
        H = [FO.histogram_entropy(np.round(0.3 + 3.0 * rng.standard_normal(N)))[0] for _ in range(800)]
        bias[N] = float(np.mean(H)) - ref
    # both negative (the plug-in reads low); the standard errors of the two means are about 0.004 and 0.0012 bits
    assert bias[100] < 0 and bias[1000] < 0, bias
    ratio = bias[100] / bias[1000]
    assert 5.0 < ratio < 15.0, (bias, ratio)


def test_invariants():
    rng = np.random.default_rng(SEED + 2)
    x = rng.normal(80.0, 7.0, 2000)
    h0 = FO.histogram_entropy(x)
    assert FO.histogram_entropy(x + 7.0) == h0 and FO.histogram_entropy(x - 31.0) == h0
    assert FO.histogram_entropy(rng.permutation(x)) == h0
    assert FO.histogram_entropy(x + 6.0, bin_width=2.0) == FO.histogram_entropy(x, bin_width=2.0)
    assert FO.histogram_entropy(4.0 * x, bin_width=4.0) == h0          # data and bins scaled together
    xi = np.round(x)
    assert FO.histogram_entropy(xi + 3.0) == FO.histogram_entropy(xi)
    # a gain a on a distribution wide against the bins adds log2 a (h(aX) = h(X) + log2 a in the continuous limit);
    # bias difference (m - 1)/(2 N ln 2) for m ~ 160 vs 320 at N = 1e6 is below 1e-3 bits
    g = rng.normal(0.0, 20.0, 1_000_000)
    d = FO.histogram_entropy(2.0 * g)[0] - FO.histogram_entropy(g)[0]
    assert abs(d - 1.0) < 0.01, d
    # plane_entropies: a different integer offset per plane leaves every entropy unchanged
    st = np.round(rng.normal(100.0, 5.0, size=(3, 12, 12)))
    prof = rng.normal(100.0, 5.0, size=(3, 30))
    strip = np.zeros((12, 12), bool)
    strip[3:9, 2:10] = True
    off = np.array([0.0, 4.0, -9.0])
    a = FO.plane_entropies(prof, st, strip)
    b = FO.plane_entropies(prof + off[:, None], st + off[:, None, None], strip)
    assert np.array_equal(a["h_line"], b["h_line"]) and np.array_equal(a["h_strip"], b["h_strip"])


def test_noise_floor():
    rng = np.random.default_rng(SEED + 3)
    ref = _discretised_gaussian_entropy(0.3, 3.0)
    h, n, m = FO.histogram_entropy(np.round(0.3 + 3.0 * rng.standard_normal(1_000_000)))
    # SD of the plug-in at N = 1e6 is about sqrt(Var(log2 p) / N) ~ 1e-3 bits; bias ~ 2e-5 bits
    assert abs(h - ref) < 0.005, (h, ref)
    # the continuous approximation 0.5 log2(2 pi e s^2) is within 0.01 bits of the discretised entropy for s = 3
    assert abs(ref - 0.5 * math.log2(2.0 * math.pi * math.e * 9.0)) < 0.01, ref


def test_contract():
    h, n, m = FO.histogram_entropy([1.0, float("nan"), 2.0, float("inf")])
    assert isinstance(h, float) and isinstance(n, int) and isinstance(m, int) and (h, n, m) == (1.0, 2, 2)
    for empty in ([], [float("nan")], np.full((2, 2), np.nan)):
        h, n, m = FO.histogram_entropy(empty)
        assert math.isnan(h) and (n, m) == (0, 0)
        c, k = FO.grey_histogram(empty)
        assert c.size == 0 and k.size == 0
    for w in (0.0, -1.0, float("nan")):
        for f in (FO.histogram_entropy, FO.grey_histogram):
            try:
                f([1.0, 2.0], w)
            except ValueError:
                continue
            raise AssertionError("%s accepted bin_width %r" % (f.__name__, w))
    st = np.zeros((4, 5, 5))
    st[:, 0, 0] = np.arange(4)
    prof = np.tile(np.arange(6.0), (4, 1))
    strip = np.zeros((5, 5), bool)
    strip[1:4, 1:4] = True
    st[1, 2, 2] = np.nan            # inside the strip: plane 1's strip entropy is NaN
    st[2, 4, 4] = np.nan            # outside: no effect
    prof[3, 0] = np.nan             # plane 3's line entropy is NaN
    e = FO.plane_entropies(prof, st, strip, valid=[True, True, True, False])
    for key in ("h_line", "h_strip", "n_line", "m_line", "n_strip", "m_strip"):
        assert e[key].shape == (4,), key
    assert math.isnan(e["h_strip"][1]) and e["n_strip"][1] == 0 and e["h_line"][1] == math.log2(6)
    assert e["h_strip"][2] == 0.0 and e["n_strip"][2] == 9
    assert all(math.isnan(e[k][3]) for k in ("h_line", "h_strip")) and e["n_line"][3] == 0 and e["n_strip"][3] == 0
    e2 = FO.plane_entropies(prof, st, strip)          # plane 3 valid: its strip is fine, its line has a NaN
    assert math.isnan(e2["h_line"][3]) and e2["n_line"][3] == 0 and e2["h_strip"][3] == 0.0
    for bad in (dict(profiles=prof[:3], stack=st, strip=strip), dict(profiles=prof, stack=st[0], strip=strip),
                dict(profiles=prof, stack=st, strip=np.zeros((5, 5), bool)),
                dict(profiles=prof, stack=st, strip=np.ones((4, 4), bool)),
                dict(profiles=prof, stack=st, strip=strip, valid=[True])):
        try:
            FO.plane_entropies(**bad)
        except ValueError:
            continue
        raise AssertionError("plane_entropies accepted %r" % (list(bad),))


def test_determinism():
    rng = np.random.default_rng(SEED + 4)
    st = np.round(rng.normal(90.0, 4.0, size=(5, 15, 15)))
    prof = rng.normal(90.0, 4.0, size=(5, 25))
    strip = rng.random((15, 15)) < 0.5
    a, b = FO.plane_entropies(prof, st, strip), FO.plane_entropies(prof.copy(), st.copy(), strip.copy())
    for key in a:
        assert np.array_equal(a[key], b[key], equal_nan=True), key


def test_edge_cases():
    import test_smoke_node_pipeline as T
    from allen_diameter.model import geometry as G
    # a rendered thin tube, no noise: its neighbourhood has the fewest grey levels in focus
    cfg = T.small_cfg(noise_sd_gl=0.0, jpeg=False)
    tube = G.Tube((0.03, -0.02, 0.05), 0.25, 0.0, 0.5, 1.0, 9.0, "axial")
    p = cfg.acquisition.res0_um
    left = top = int(math.floor(-3.5 / p))
    block, ks, valid, _ = T.make_provider([tube], [1.0], cfg, SEED)(left, top, int(math.ceil(7.0 / p)),
                                                                   int(math.ceil(7.0 / p)), -5, 5)
    th = 0.5
    o, y_hat, e_u = np.array([0.03, -0.02]), np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = PR.profile_offsets(cfg.measure)
    fr = _frame(left, top, p)
    prof = np.array([PR.sample_profile(block[i].astype(float), fr, o, y_hat, e_u, v) for i in range(ks.size)])
    strip = PR.stripe_mask(block.shape[1:], fr, o, y_hat, e_u, cfg.measure.profile_half_um, 1.0)
    e = FO.plane_entropies(prof, block.astype(float), strip, valid)
    assert ks[FO.difference_dip(e["h_line"])] == 0 and ks[FO.difference_dip(e["h_strip"])] == 0, (e["h_line"], e["h_strip"])
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
        out = os.path.join(tmp, "ent")
        rec, skip = PD.run(swc, prov, ccfg, [4, 1], out, "999", planes_half=8, evaluation="entropy", log=lambda m: None)
        hl, hs = np.array(rec["h_line"], dtype=float), np.array(rec["h_strip"], dtype=float)
        assert rec["ks"] == list(range(-8, 9)) and rec["n_missing"] == 2 and rec["evaluation"] == "entropy"
        assert np.isnan(hl[0]) and np.isnan(hl[-1]) and np.isnan(hs[0]) and np.isnan(hs[-1])
        assert np.all(np.isfinite(hl[1:-1])) and np.all(np.isfinite(hs[1:-1]))
        assert set(rec["n_line"][1:-1]) == {53} and rec["n_line"][0] == 0, rec["n_line"]
        assert set(rec["n_strip"][1:-1]) == {917} and rec["n_strip"][-1] == 0, rec["n_strip"]
        assert rec["k_dip_h_line"] == 0 and rec["k_dip_h_strip"] == 0 and rec["k_star"] == 0, rec
        assert os.path.basename(rec["png"]) == "planeentropy_4.png" and os.path.exists(rec["png"])
        assert "skipped" in skip and skip["node_id"] == 1
        with open(os.path.join(out, "planeentropy_999.json")) as f:
            assert len(json.load(f)) == 2
        assert not os.path.exists(os.path.join(out, "planediff_999.json"))
        for off in (3.0, -5.0):
            def shifted(left, top, w, h, k_lo, k_hi, off=off):       # one plane brighter or darker, after the camera
                block, ks, valid, frame = prov(left, top, w, h, k_lo, k_hi)
                block = block.astype(float)
                block[ks == 2] += off
                return block, ks, valid, frame
            rs = PD.run(swc, shifted, ccfg, [4], os.path.join(tmp, "shift"), "999", planes_half=8,
                        evaluation="entropy", log=lambda m: None)[0]
            assert np.array_equal(rs["h_line"], rec["h_line"], equal_nan=True) and \
                np.array_equal(rs["h_strip"], rec["h_strip"], equal_nan=True), off

        def darker(left, top, w, h, k_lo, k_hi):                     # plane +2 made 2 % darker after the camera
            block, ks, valid, frame = prov(left, top, w, h, k_lo, k_hi)
            block = block.astype(float)
            block[ks == 2] *= 0.98
            return block, ks, valid, frame
        rd = PD.run(swc, darker, ccfg, [4], os.path.join(tmp, "dark"), "999", planes_half=8, evaluation="entropy",
                    log=lambda m: None)[0]
        i2 = rec["ks"].index(2)
        assert abs(rd["h_strip"][i2] - rec["h_strip"][i2]) < 0.05, (rd["h_strip"][i2], rec["h_strip"][i2])
        rn = PD.run(swc, prov, ccfg, [4], os.path.join(tmp, "narrow"), "999", planes_half=8, evaluation="entropy",
                    stripe_half_um=0.5, log=lambda m: None)[0]
        assert 0 < max(rn["n_strip"]) < 917 and rn["stripe_half_um"] == 0.5, rn["n_strip"]


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
