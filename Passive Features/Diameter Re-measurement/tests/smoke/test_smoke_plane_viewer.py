"""Smoke test for the soma's plane viewer (plotting/plane_viewer.py, loading/soma_io.py, show_soma.py --z-margin-um) -- Block 44 amendment 4 in specs/SPEC.md (D-239).

Oracles: the grey mapping's known answer (lo -> 0, hi -> 255, linear between, clipped outside; whole-pixel zoom), the
grey ranges against numpy.percentile written out, the view's pixels against a brute-force selection of the pixel
centres; invariance of the shown image under an affine change of the grey levels and its range; the contract (shapes,
dtypes, NaN rows for planes not recorded); determinism; the refusals; the reader against show_soma.py's own files
(--synthetic, end to end, with and without --z-margin-um); the viewer's widgets driven from Python (skipped when
ipywidgets is not installed): the slider, the contrast toggle, a plane not recorded, the pick and its callback.

Checks, from the strongest oracle to the weakest
    test_known_answer   analytic or tabulated solution
    test_reference      independent implementation (library function, or
                        brute force on a tiny input)
    test_convergence    discretisations only: observed order p in
                        e(h) ~ C * h**p, where e(h) is the error at step h
    test_invariants     conservation laws, symmetries, bounds, normalisation
    test_contract       output shapes, dtypes and finiteness
    test_determinism    identical output for an identical seed
    test_edge_cases     degenerate or boundary inputs named in the spec
    test_viewer         the widget driven from Python (no browser)

Run
    python tests/smoke/test_smoke_plane_viewer.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python -m pytest tests/smoke -q
        The same checks through pytest (TODO checks show up as failures).

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz
    Inspect it with
        d = np.load("tests/smoke/artifacts/<check name>.npz")
        diff = d["actual"] - d["expected"]
        np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

Add a check
    Define a function test_<something>() with no arguments that raises
    AssertionError on failure; the runner below and pytest both pick it up.

Keep this file pure ASCII (no Greek letters, dashes or arrows outside ASCII)
so it survives transfer to the cluster.
"""
from __future__ import annotations

import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "scripts"))
from allen_diameter.plotting import plane_viewer as PV   # noqa: E402
from allen_diameter.loading import soma_io                # noqa: E402

SEED = 20261124
REPORT_PACKAGES = ("numpy", "Pillow", "ipywidgets")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def observed_order(h, err):
    """Least-squares slope p of log e(h) = log C + p * log h.

    h: step sizes (any order); err: positive errors e(h) at those steps.
    """
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


# ---------------------------------------------------------------- checks ---

def _decode(png):
    """The PNG's pixels (uint8, 2-D) and its mode."""
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    return np.asarray(im), im.mode


def _stack(n=5, h=40, w=50, seed=SEED):
    """A small stack: plane i a ramp of contrast (i + 1) plus noise, grey levels in 0..255 (uint8); plane 3 not
    recorded (left white, as fetch_zblock leaves it)."""
    rng = make_rng(seed)
    x = np.linspace(-1.0, 1.0, w)[None, :] * np.ones((h, 1))
    block = np.empty((n, h, w), dtype=np.uint8)
    for i in range(n):
        block[i] = np.clip(np.rint(128 + 20 * (i + 1) * x + rng.normal(0.0, 3.0, (h, w))), 0, 255)
    valid = np.ones(n, bool)
    valid[3] = False
    block[3] = 255
    return block, np.arange(10, 10 + n), valid


def test_known_answer():
    """plane_png maps lo to 0 and hi to 255, linearly between (rounded), clipped outside; zoom repeats each pixel
    zoom x zoom times exactly."""
    img = np.array([[10.0, 20.0, 30.0, 40.0], [5.0, 50.0, 25.0, 35.0]])
    lo, hi = 10.0, 40.0
    u, mode = _decode(PV.plane_png(img, lo, hi, 1))
    expected = np.clip(np.rint(255.0 * (img - lo) / (hi - lo)), 0, 255).astype(np.uint8)
    assert mode == "L", mode
    assert_array_equal(u, expected)
    assert u[0, 0] == 0 and u[0, 3] == 255 and u[1, 0] == 0 and u[1, 1] == 255, u      # ends and clipping
    assert u[0, 1] == 85 and u[0, 2] == 170, u                                        # 255/3 and 2*255/3
    u3, _ = _decode(PV.plane_png(img, lo, hi, 3))
    assert u3.shape == (6, 12), u3.shape
    assert_array_equal(u3, np.repeat(np.repeat(expected, 3, axis=0), 3, axis=1))


def test_reference():
    """grey_ranges against numpy.percentile written out; view_slices against the pixel centres selected one by
    one."""
    block, ks, valid = _stack()
    fixed = PV.grey_ranges(block, valid, "fixed", (0.5, 99.5))
    ref = np.percentile(block[valid].astype(float), (0.5, 99.5))
    # exact: the same numpy call on the same values (float64 roundoff only)
    check_close("grey_ranges fixed", fixed[valid], np.tile(ref, (int(valid.sum()), 1)), rtol=1e-12)
    per = PV.grey_ranges(block, valid, "per plane", (2.0, 98.0))
    for i in np.flatnonzero(valid):
        check_close("grey_ranges per plane %d" % i, per[i], np.percentile(block[i].astype(float), (2.0, 98.0)),
                    rtol=1e-12)
    H, W, left, top, p = 60, 70, -30, 12, 0.25
    centre, half = (-1.3, 10.4), 3.1
    rows, cols = PV.view_slices((H, W), left, top, p, centre, half)
    x = (left + np.arange(W)) * p
    y = (top + np.arange(H)) * p
    want_c = np.flatnonzero(np.abs(x - centre[0]) <= half)
    want_r = np.flatnonzero(np.abs(y - centre[1]) <= half)
    assert_array_equal(np.arange(W)[cols], want_c)
    assert_array_equal(np.arange(H)[rows], want_r)


def test_convergence():
    raise unittest.SkipTest("no discretisation: a display mapping and a reader")


def test_invariants():
    """The image shown does not change under g -> a g + b (a > 0) with its range mapped the same way; the fixed
    range is one row for every recorded plane; per-plane ranges follow each plane's contrast."""
    block, ks, valid = _stack()
    img = block[1].astype(float)
    lo, hi = 100.0, 160.0
    a, b = 2.5, -17.0
    assert PV.plane_png(img, lo, hi, 2) == PV.plane_png(a * img + b, a * lo + b, a * hi + b, 2)
    fixed = PV.grey_ranges(block, valid, "fixed")
    assert np.all(fixed[valid] == fixed[valid][0]), fixed
    per = PV.grey_ranges(block, valid, "per plane")
    widths = per[valid, 1] - per[valid, 0]
    assert np.all(np.diff(widths) > 0), widths          # planes 0, 1, 2, 4: contrast growing with the index


def test_contract():
    """Shapes and dtypes: the ranges (n_planes, 2) float with NaN rows for planes not recorded; the PNG 8-bit grey
    of (H zoom, W zoom); the view's slices inside the crop; the reader's fields."""
    block, ks, valid = _stack()
    for contrast in PV.CONTRASTS:
        r = PV.grey_ranges(block, valid, contrast)
        assert r.shape == (len(ks), 2) and r.dtype == float, (r.shape, r.dtype)
        assert np.all(np.isnan(r[~valid])) and np.all(np.isfinite(r[valid])) and np.all(r[valid, 1] > r[valid, 0])
    u, mode = _decode(PV.plane_png(block[0], 0, 255, 2))
    assert u.dtype == np.uint8 and mode == "L" and u.shape == (2 * block.shape[1], 2 * block.shape[2]), u.shape
    rows, cols = PV.view_slices(block.shape[1:], 0, 0, 0.1, (2.0, 2.0), 100.0)      # clipped to the crop
    assert (rows, cols) == (slice(0, block.shape[1]), slice(0, block.shape[2])), (rows, cols)
    assert PV.view_slices(block.shape[1:], 0, 0, 0.1) == (slice(0, 40), slice(0, 50))


def test_determinism():
    """The same plane and range give the same bytes."""
    block, ks, valid = _stack()
    r = PV.grey_ranges(block, valid, "per plane")
    assert PV.plane_png(block[2], *r[2], 3) == PV.plane_png(block[2].copy(), *r[2], 3)


def test_edge_cases():
    """The refusals; a flat plane's range; show_soma.py end to end on the synthetic cell, by default and with
    --z-margin-um (the window widens, the JSON records the margin, a negative margin is refused); the reader on its
    files, and its refusals (a file missing, the two files disagreeing)."""
    block, ks, valid = _stack()
    for bad in (lambda: PV.grey_ranges(block, valid, "auto"),
                lambda: PV.grey_ranges(block, valid, "fixed", (50.0, 50.0)),
                lambda: PV.grey_ranges(block, valid, "fixed", (-1.0, 99.0)),
                lambda: PV.grey_ranges(block, np.zeros(len(ks), bool)),
                lambda: PV.grey_ranges(block[0], valid),
                lambda: PV.grey_ranges(block, valid[:3]),
                lambda: PV.plane_png(block[0], 5.0, 5.0),
                lambda: PV.plane_png(block[0], 0.0, np.nan),
                lambda: PV.plane_png(block, 0.0, 255.0),
                lambda: PV.plane_png(block[0], 0.0, 255.0, 0),
                lambda: PV.plane_png(block[0], 0.0, 255.0, 1.5),
                lambda: PV.view_slices((10, 10), 0, 0, 0.1, (0.5, 0.5), 0.0),
                lambda: PV.view_slices((10, 10), 0, 0, 0.1, None, 1.0),
                lambda: PV.view_slices((10, 10), 0, 0, 0.1, (50.0, 50.0), 1.0)):      # misses the crop
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("no ValueError")
    flat = np.full((2, 4, 4), 77, dtype=np.uint8)
    r = PV.grey_ranges(flat, np.ones(2, bool), "fixed")
    assert_array_equal(r, [[77.0, 78.0], [77.0, 78.0]])
    import show_soma
    out = tempfile.mkdtemp(prefix="plane_viewer_smoke_")
    assert show_soma.main(["--specimen", "7", "--synthetic", "--out", out]) == 0
    v = soma_io.read_soma_view(out, 7)
    rec = json.load(open(os.path.join(out, "soma_7.json")))
    assert v.block.shape == (len(rec["ks"]), rec["crop"]["height"], rec["crop"]["width"]) and v.block.dtype == np.uint8
    assert list(v.ks) == rec["ks"] and list(v.valid) == rec["valid"] and v.k_traced == rec["k_traced"]
    assert (v.left, v.top, v.res0_um, v.dz_um) == (rec["crop"]["left"], rec["crop"]["top"], rec["crop"]["res0_um"],
                                                   rec["dz_um"])
    assert v.k_by_eye is None and rec["z_margin_um"] == 6.0, rec["z_margin_um"]
    out2 = tempfile.mkdtemp(prefix="plane_viewer_smoke_")
    assert show_soma.main(["--specimen", "7", "--synthetic", "--out", out2, "--z-margin-um", "8", "--soma-k", "2"]) == 0
    v2 = soma_io.read_soma_view(out2, 7)
    rec2 = json.load(open(os.path.join(out2, "soma_7.json")))
    # the window: |k dz| <= radius + margin; 2 um more margin is floor(2 / 0.28) = 7 planes more on each side
    n_more = int(np.floor(2.0 / rec["dz_um"] + 1e-9))
    assert v2.ks[0] == v.ks[0] - n_more and v2.ks[-1] == v.ks[-1] + n_more, (v.ks[[0, -1]], v2.ks[[0, -1]])
    assert rec2["z_margin_um"] == 8.0 and v2.k_by_eye == 2
    for argv in (["--z-margin-um", "-1"], ["--z-margin-um", "nan"]):
        try:
            show_soma.main(["--specimen", "7", "--synthetic", "--out", out2] + argv)
        except SystemExit as exc:
            assert exc.code == 2, (argv, exc.code)
            continue
        raise AssertionError("not refused: %s" % argv)
    try:
        soma_io.read_soma_view(out, 8)
    except FileNotFoundError as exc:
        assert "show_soma.py" in str(exc), exc
    else:
        raise AssertionError("no FileNotFoundError")
    rec_bad = dict(rec)
    rec_bad["ks"] = [k + 1 for k in rec["ks"]]
    with open(os.path.join(out, "soma_7.json"), "w") as f:
        json.dump(rec_bad, f)
    try:
        soma_io.read_soma_view(out, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for files that disagree")


def test_viewer():
    """The widget driven from Python: the first plane (k_start, else k_ref, else the middle), the slider's range
    over the recorded planes, the image following the slider and the contrast, a plane not recorded shown grey and
    said so, the pick handed to on_pick (its message shown; its exception shown, not raised), a k_start that is not
    recorded said in the status line; the refusals."""
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("ipywidgets not installed (the notebook's dependency)")
    block, ks, valid = _stack()
    picked = []

    def on_pick(k):
        picked.append(k)
        return "SOMA_K = %d" % k

    v = PV.PlaneViewer(block, ks, valid, left=0, top=0, res0_um=0.1, dz_um=0.28, k_ref=12, title="t",
                       zoom=2, on_pick=on_pick)
    assert v.k == 12 and (v.slider.min, v.slider.max) == (10, 14), (v.k, v.slider.min, v.slider.max)
    assert v.image.value == v.png(12, "fixed") and "k = <b>12</b>" in v.info.value, v.info.value
    u, _ = _decode(v.image.value)
    assert u.shape == (2 * block.shape[1], 2 * block.shape[2]), u.shape
    r = PV.grey_ranges(block, valid, "fixed")
    assert v.image.value == PV.plane_png(block[2], *r[2], 2)
    v.slider.value = 14
    assert v.image.value == PV.plane_png(block[4], *r[4], 2) and "z - z(k 12) = +0.56" in v.info.value, v.info.value
    v.contrast.value = "per plane"
    rp = PV.grey_ranges(block, valid, "per plane")
    assert v.image.value == PV.plane_png(block[4], *rp[4], 2)
    v.slider.value = 13                                                       # not recorded
    u13, _ = _decode(v.image.value)
    assert np.all(u13 == PV.PLACEHOLDER_GREY) and "not recorded" in v.info.value, v.info.value
    v.pick.click()
    assert picked == [] and "not picked" in v.status.value, v.status.value
    v.slider.value = 11
    v.pick.click()
    assert picked == [11] and v.status.value == "SOMA_K = 11", v.status.value

    def failing(k):
        raise RuntimeError("boom")

    w = PV.PlaneViewer(block, ks, valid, left=0, top=0, res0_um=0.1, dz_um=0.28, on_pick=failing, k_start=13)
    assert w.k == 12 and "not a recorded plane" in w.status.value, (w.k, w.status.value)   # the middle recorded
    w.pick.click()
    assert "on_pick failed" in w.status.value and "boom" in w.status.value, w.status.value
    x = PV.PlaneViewer(block, ks, valid, left=0, top=0, res0_um=0.1, dz_um=0.28, k_start=14, k_ref=12,
                       centre_xy_um=(2.0, 1.0), view_half_um=0.5)
    assert x.k == 14 and x.view.shape == (5, 11, 11), x.view.shape                 # +-0.5 um at 0.1 um/px
    assert x.pick not in x.widget.children[3].children                              # no on_pick: no button
    for bad in (dict(contrast="auto"), dict(zoom=0), dict(valid=np.zeros(5, bool)), dict(ks=ks[::-1].copy()),
                dict(view_half_um=1.0)):
        kw = dict(left=0, top=0, res0_um=0.1, dz_um=0.28)
        args = [block, bad.pop("ks", ks), bad.pop("valid", valid)]
        kw.update(bad)
        try:
            PV.PlaneViewer(*args, **kw)
        except ValueError:
            continue
        raise AssertionError("no ValueError: %s" % bad)


def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
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
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
