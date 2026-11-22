"""Smoke test for the soma view (analysis/soma_view.py, plotting/soma_figures.py, scripts/show_soma.py) -- Block 44 in specs/SPEC.md (D-210, D-211).

Oracles: the fine-detail energy's known answer (largest on the least blurred plane of a blurred-disk stack), a
direct computation with scipy and the disk written out, its invariants (contrast squared, offset, a quadratic
field); the reference plane (the one chosen by eye, else the recorded plane nearest the traced depth) and the
montage's planes (the whole window at the step, aligned on the reference plane); the geometry's arithmetic,
determinism, the refusals, and the script end to end on the synthetic cell (D-211: no plane picked automatically).

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

Run
    python tests/smoke/test_smoke_<algorithm>.py
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
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

import os
WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WS, "src"))
from allen_diameter.analysis import soma_view as SV   # noqa: E402

SEED = 20261122
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
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

def _disk_stack(blurs_um, radius_um=4.0, res=0.1144, half_um=8.0, dark=80.0, bright=200.0, seed=SEED):
    """A dark disk at the origin on a bright field, blurred by a Gaussian of SD blurs_um[k] on plane k, plus mild
    noise (SD 0.5 gl): the focus measure's known answer is the least blurred plane."""
    from scipy.ndimage import gaussian_filter
    n = int(round(2 * half_um / res)) + 1
    left = top = int(round(-half_um / res))
    x = (left + np.arange(n)) * res
    X, Y = np.meshgrid(x, x)
    base = np.where(X ** 2 + Y ** 2 <= radius_um ** 2, dark, bright)
    rng = make_rng(seed)
    planes = [gaussian_filter(base, b / res) + rng.normal(0.0, 0.5, base.shape) for b in blurs_um]
    return np.stack(planes), left, top, res


def test_known_answer():
    """The fine-detail energy of a blurred-disk stack is largest on the least blurred plane and falls off on both
    sides; reference_plane returns the chosen plane, or the recorded plane nearest the traced depth; montage_planes
    spans the window at the step, aligned on the reference plane (the pilot's window: k = 74..160, k_ref 117, step
    2 -> 43 planes with 117 among them; 105..145 step 1 -> 41)."""
    blurs = [0.15 + 0.4 * abs(k - 3) for k in range(9)]
    block, left, top, res = _disk_stack(blurs)
    E = SV.detail_energy(block, np.ones(9, bool), left, top, res, (0.0, 0.0), 5.0, 1.0)
    assert int(np.nanargmax(E)) == 3, ("argmax", E)
    assert np.all(np.diff(E[:4]) > 0) and np.all(np.diff(E[3:]) < 0), E
    ks = np.arange(74, 161)
    v = np.ones(ks.size, bool)
    assert SV.reference_plane(ks, v, 117 * 0.28, 0.28) == (117 - 74, "traced depth")
    assert SV.reference_plane(ks, v, 117 * 0.28, 0.28, 110) == (110 - 74, "chosen by eye")
    v2 = v.copy()
    v2[117 - 74] = False                                   # the traced plane missing: the nearest recorded, lower first
    assert SV.reference_plane(ks, v2, 117 * 0.28, 0.28)[0] == 116 - 74
    idx = SV.montage_planes(ks, v, 117, 2)
    got = [int(ks[i]) for i in idx]
    assert len(got) == 43 and 117 in got and got[0] == 75 and got[-1] == 159 and set(np.diff(got)) == {2}, got
    assert [int(ks[i]) for i in SV.montage_planes(ks, v, 117, 1, 105, 145)] == list(range(105, 146))


def test_reference():
    """Against a direct computation: scipy.ndimage.gaussian_laplace and the disk mask written out here."""
    from scipy.ndimage import gaussian_laplace
    block, left, top, res = _disk_stack([0.3, 0.6])
    E = SV.detail_energy(block, np.ones(2, bool), left, top, res, (0.5, -0.25), 3.0, 1.5)
    x = (left + np.arange(block.shape[2])) * res
    y = (top + np.arange(block.shape[1])) * res
    m = (x[None, :] - 0.5) ** 2 + (y[:, None] + 0.25) ** 2 <= 9.0
    ref = [np.var(gaussian_laplace(block[j].astype(float), 1.5)[m]) for j in range(2)]
    check_close("detail_energy against the direct form", E, ref, rtol=1e-12)   # same operations: roundoff only


def test_convergence():
    raise unittest.SkipTest("no discretisation: the blocks draw and measure recorded planes")


def test_invariants():
    """E scales with the square of the contrast and ignores a constant offset; a quadratic field (constant
    Laplacian) has E near 0 away from the borders; the montage skips planes not recorded and keeps the alignment."""
    block, left, top, res = _disk_stack([0.3, 0.8])
    v = np.ones(2, bool)
    E1 = SV.detail_energy(block, v, left, top, res, (0.0, 0.0), 5.0, 1.0)
    E2 = SV.detail_energy(2.0 * block + 7.0, v, left, top, res, (0.0, 0.0), 5.0, 1.0)
    check_close("E(2 I + 7) = 4 E(I)", E2, 4.0 * E1, rtol=1e-10)       # linear filter: exact up to roundoff
    n = 141
    x = (np.arange(n) - 70) * res
    X, Y = np.meshgrid(x, x)
    quad = (2.0 * X ** 2 + 3.0 * Y ** 2)[None]
    Eq = SV.detail_energy(quad, np.ones(1, bool), -70, -70, res, (0.0, 0.0), 2.0, 1.0)
    assert Eq[0] < 1e-6, Eq        # the Laplacian of Gaussian of a quadratic is a constant: zero variance
    ks = np.arange(-5, 6)
    vv = np.ones(ks.size, bool)
    vv[[3, 7]] = False             # planes -2 and 2 missing
    assert [int(ks[i]) for i in SV.montage_planes(ks, vv, 0, 2)] == [-4, 0, 4]


def test_contract():
    """Geometry, crop and plane range; the outputs' shapes and types."""
    xyz = np.array([[10.0, 20.0, 5.0], [12.0, 22.0, 7.0], [30.0, 30.0, 6.0]])
    c, r = SV.soma_geometry(xyz, [3.0, 5.0, 1.0], [1, 1, 3])
    check_close("centre", c, [11.0, 21.0, 6.0], rtol=0, atol=1e-12)
    assert r == 5.0
    left, top, w, h = SV.crop_box((11.0, 21.0), 20.0, 0.1144)
    assert (left, top) == (int(np.floor(-9.0 / 0.1144)), int(np.floor(1.0 / 0.1144))) and w == h == int(np.ceil(40 / 0.1144)) + 1
    assert SV.plane_range(10.0, 3.0, 0.28) == (25, 46)
    assert SV.plane_range(0.0, 0.0, 0.28) == (0, 0)
    block, left, top, res = _disk_stack([0.3, 0.5, 0.7])
    E = SV.detail_energy(block, np.array([True, False, True]), left, top, res, (0.0, 0.0), 5.0, 1.0)
    assert E.shape == (3,) and np.isnan(E[1]) and np.all(np.isfinite(E[[0, 2]]))
    j, lab = SV.reference_plane([0, 1, 2], [True, False, True], 0.29, 0.28)
    assert isinstance(j, int) and j in (0, 2) and lab == "traced depth"
    assert all(isinstance(i, int) for i in SV.montage_planes([0, 1, 2], [True, True, True], 0, 1))
    assert not hasattr(SV, "sharpest") and not hasattr(SV, "focus_measure")   # D-211: no automatic pick


def test_determinism():
    block, left, top, res = _disk_stack([0.3, 0.5])
    a = SV.detail_energy(block, np.ones(2, bool), left, top, res, (0.0, 0.0), 5.0, 1.0)
    b = SV.detail_energy(block, np.ones(2, bool), left, top, res, (0.0, 0.0), 5.0, 1.0)
    assert np.array_equal(a, b)


def test_edge_cases():
    """The refusals; a disk outside the crop; the script end to end on the synthetic cell (no network): by default
    no plane is chosen and the montage spans the window with the traced depth's plane among its planes (the failure
    of the first real run: the montage centred on the automatic pick never reached the soma); with --soma-k,
    --soma-k-range and --montage-k the choices are recorded; a plane outside the window is refused."""
    ks = np.arange(0, 10)
    v = np.ones(10, bool)
    vbad = v.copy()
    vbad[4] = False
    for bad in (lambda: SV.soma_geometry(np.zeros((2, 3)), [1.0, 1.0], [3, 3]),
                lambda: SV.crop_box((0.0, 0.0), -1.0, 0.1),
                lambda: SV.plane_range(0.0, 1.0, 0.0),
                lambda: SV.reference_plane(ks, v, 0.0, 0.28, 12),          # outside the window
                lambda: SV.reference_plane(ks, vbad, 0.0, 0.28, 4),        # not recorded
                lambda: SV.reference_plane(ks, np.zeros(10, bool), 0.0, 0.28),
                lambda: SV.montage_planes(ks, v, 0, 0),
                lambda: SV.montage_planes(ks, v, 0, 1, 7, 3)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("no ValueError")
    block, left, top, res = _disk_stack([0.3])
    E = SV.detail_energy(block, np.ones(1, bool), left, top, res, (500.0, 500.0), 2.0, 1.0)
    assert np.isnan(E[0]), E
    import json
    import tempfile
    sys.path.insert(0, os.path.join(WS, "scripts"))
    import show_soma
    out = tempfile.mkdtemp(prefix="soma_smoke_")
    assert show_soma.main(["--specimen", "7", "--synthetic", "--out", out]) == 0
    for name in ("soma_7_focus.png", "soma_7_planes.png", "soma_7.json", "soma_7_block.npz"):
        assert os.path.exists(os.path.join(out, name)), name
    rec = json.load(open(os.path.join(out, "soma_7.json")))
    rec_ks = [k for k, ok in zip(rec["ks"], rec["valid"]) if ok]
    assert rec["k_by_eye"] is None and rec["k_range_by_eye"] is None and rec["k_shown"] == rec["k_traced"]
    assert rec["k_traced"] in rec["montage_ks"]
    assert rec["montage_ks"] == [k for k in rec_ks if (k - rec["k_traced"]) % 2 == 0], rec["montage_ks"]
    assert "k_sharpest" not in rec and "focus" not in rec and len(rec["detail_energy"]) == len(rec["ks"])
    out2 = tempfile.mkdtemp(prefix="soma_smoke_")
    assert show_soma.main(["--specimen", "7", "--synthetic", "--out", out2, "--soma-k", "2", "--soma-k-range", "-3",
                           "5", "--montage-k", "-6", "6", "1"]) == 0
    rec2 = json.load(open(os.path.join(out2, "soma_7.json")))
    assert (rec2["k_by_eye"], rec2["k_range_by_eye"], rec2["k_shown"]) == (2, [-3, 5], 2), rec2
    assert rec2["montage_ks"] == [k for k in rec_ks if -6 <= k <= 6]
    for argv in (["--soma-k", "99"], ["--soma-k-range", "5", "-3"], ["--montage-k", "40", "50", "1"]):
        try:
            show_soma.main(["--specimen", "7", "--synthetic", "--out", out2] + argv)
        except SystemExit as exc:
            assert exc.code == 2, (argv, exc.code)
            continue
        raise AssertionError("not refused: %s" % argv)


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
