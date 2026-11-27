"""Smoke test for a twin's measurement -- Block 55 in specs/SPEC.md (section 5, "Kernel calibration, step 2").

WIP (2026-11-27): the oracles that run a scan (M2, M3, M5, M6, M7) are blocked by the ISSUE reported to the
orchestrator (twin_scene's node lacks B_bar and mu_hat_per_um, which the scans' "stripe" background support reads);
they are reported as TODO so the run fails until the issue is settled. M1 and the noise half of M4 run.

Run
    python tests/smoke/test_smoke_step2_measure.py

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import dataclasses
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

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (SRC, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import step2_measure as M  # noqa: E402
from allen_diameter.analysis.step2_render import TwinStack  # noqa: E402
from allen_diameter.analysis.step2_twins import Twin  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import camera  # noqa: E402

SEED = 20261127  # Step2Config.seed
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

BLOCKED = ("blocked by the Block 55 ISSUE: twin_scene's node has no B_bar / mu_hat_per_um, read by "
           "calibration.support_masks under background_support 'stripe'")


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

def _twin(n, ks, B, sd, set_="near"):
    z = np.zeros(n)
    return Twin(cell_id=7, node_id=11, set=set_, role="train", d_um=0.5, mu_per_um=0.2, k_star=int(ks[n // 2]),
                z_sub_um=0.0, cx_um=50.0, cy_um=50.0, theta_rad=0.3, phi_rad=0.1, ks=np.asarray(ks, np.int64),
                B_gl=np.asarray(B, float), noise_sd_gl=np.asarray(sd, float), omega_um2=z, se_omega_um2=z,
                area_gl_um=z, status=("ok",) * n)


def _toy_stack(n=5, H=12, W=14, depth=0.3):
    """A synthetic transmittance: a Gaussian dip along x, deeper in the middle plane (no render needed for the
    camera checks M1 and M4; the scan-dependent checks would use render.render_transmittance)."""
    y = np.arange(H)[:, None] + 0 * np.arange(W)[None, :]
    tau = np.empty((n, H, W))
    for q in range(n):
        tau[q] = 1.0 - depth * np.exp(-0.5 * ((y - H / 2.0) / (1.0 + abs(q - n // 2))) ** 2) / (1 + abs(q - n // 2))
    return TwinStack(tau=tau, ks=np.arange(-(n // 2), n - n // 2, dtype=np.int64), left=400, top=420, res0_um=0.114)


def test_known_answer():
    # M1: one background and one SD for every plane: camera_planes equals camera_chain on the whole stack with the
    # matching per-plane generators, bit for bit (jpeg on, the default renderer)
    cfg = default_config()
    rc = cfg.renderer
    st = _toy_stack()
    n = st.tau.shape[0]
    B = rc.black_level_gl + rc.gain * 180.0
    tw = _twin(n, st.ks, np.full(n, B), np.full(n, 2.5))
    blk, valid = M.camera_planes(st, tw.B_gl, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))
    rq = dataclasses.replace(rc, background_B_gl=180.0, noise_sd_gl=2.5)
    ref = np.stack([camera.camera_chain(st.tau[q], rq, g) for q, g in enumerate(M.noise_rngs(tw, n, cfg.step2, 0))])
    assert valid.all() and blk.dtype == np.uint8 and blk.shape == st.tau.shape
    assert_array_equal(blk, ref)


def test_reference():
    # M2: needs plane_scan on the twin's node
    raise NotImplementedError("M2 " + BLOCKED)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the block composes a camera chain and a scan")


def test_invariants():
    # M3 (background and SD as measured by the scan) and M5 (twin_statistics) need the scan
    raise NotImplementedError("M3, M5 " + BLOCKED)


def test_contract():
    # M6: a far twin through far_plane_scan
    raise NotImplementedError("M6 " + BLOCKED)


def test_determinism():
    # M4 (noise half): two calls bit for bit; another replicate differs; a different stack with the same seed uses
    # the same noise field: with bit_depth 16 and jpeg off the difference of the two blocks is the difference of
    # the grey mappings to 1 gl (each block's rounding contributes at most 0.5 gl)
    cfg = default_config()
    rc = dataclasses.replace(cfg.renderer, bit_depth=16, jpeg=False)
    s1, s2 = _toy_stack(depth=0.3), _toy_stack(depth=0.6)
    n = s1.tau.shape[0]
    B = np.full(n, rc.black_level_gl + rc.gain * 3000.0)
    tw = _twin(n, s1.ks, B, np.full(n, 20.0))
    a = M.camera_planes(s1, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))[0]
    b = M.camera_planes(s1, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))[0]
    c = M.camera_planes(s1, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 1))[0]
    d = M.camera_planes(s2, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))[0]
    assert_array_equal(a, b)
    assert np.any(a != c)
    gm = (camera.grey_mapping(s1.tau, dataclasses.replace(rc, background_B_gl=3000.0))
          - camera.grey_mapping(s2.tau, dataclasses.replace(rc, background_B_gl=3000.0)))
    assert np.max(np.abs((a.astype(float) - d.astype(float)) - gm)) <= 1.0


def test_edge_cases():
    # M7 first half: a non-finite B_gl (and a B_gl at or below the black level) gives an invalid, all-zero plane;
    # the status half needs the scan
    cfg = default_config()
    rc = cfg.renderer
    st = _toy_stack()
    n = st.tau.shape[0]
    B = np.full(n, rc.black_level_gl + rc.gain * 180.0)
    B[1] = np.nan
    B[3] = rc.black_level_gl
    tw = _twin(n, st.ks, B, np.full(n, 2.5))
    blk, valid = M.camera_planes(st, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))
    assert_array_equal(valid, [True, False, True, False, True])
    assert not blk[1].any() and not blk[3].any() and blk[0].any()
    raise NotImplementedError("M7 status half " + BLOCKED)


# ---------------------------------------------------------------- runner ---

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
