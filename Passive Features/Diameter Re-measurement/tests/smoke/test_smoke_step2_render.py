"""Smoke test for a twin's rendered stack under a trial kernel table -- Block 54 in specs/SPEC.md (section 5,
"Kernel calibration, step 2"; D-270.1, D-270.3, D-265, D-272): analysis.step2_render.trial_renderer, twin_tube,
render_twin and TwinStack.

Checks (the oracles R1 to R5 of the Block 54 entry)
    test_known_answer   R2: trial_renderer's fields; kernel.sigma_r at the knots returns (sigma_r(0), theta);
                        "near" gives "tangent", "far" "linear" with gamma (sigma_r beyond the last knot)
    test_reference      R1: render_twin's tau through camera.camera_chain with one generator equals
                        render.synthetic_block with the same tube, absorption, planes, patch and generator, bit for
                        bit (full hybrid quadrature, the camera chain with noise and JPEG)
    test_convergence    not a discretisation of its own (Block 4's convergence is test_smoke_render's): SkipTest
    test_invariants     R3: tau in (-0.1, 1 + 1e-12]; tau = 1 to 1e-12 far from the tube; the dip on the twin's
                        line deeper at focus than three planes away. R4: theta_3 larger by 0.02 um widens the dip
                        three planes from focus and leaves the focus plane within 1e-3
    test_contract       R5: shapes, dtypes, the patch's corner and pitch; twin_tube's fields; a ks override
    test_determinism    R5: two calls bit for bit
    test_edge_cases     R2: each refusal of trial_renderer; a non-finite axis depth

Test inputs: Twin objects built here from the contract (Block 53's functions are not called). For time the patch
is small (calibration.patch_half_um replaced by 1.5 um in R1, R5; by 6.5 um in R3, R4, where tau = 1 must be
checked far from the tube), the tubes thin (d 0.4 um), the planes few (7, k = -3 .. 3), and R3, R4, R5 use a
reduced condenser quadrature (ray_n_rho 4, ray_n_psi 8) of the hybrid: this does not cover the accuracy of the
production quadrature (Block 21's check), only the plumbing, which is the same; R1 uses the default quadrature.

Run (from Passive Features/Diameter Re-measurement/)
    python tests/smoke/test_smoke_step2_render.py
    python -m pytest tests/smoke/test_smoke_step2_render.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import camera, geometry, kernel, render  # noqa: E402
from allen_diameter.analysis import calibration_scene  # noqa: E402
from allen_diameter.analysis.step2_twins import Twin  # noqa: E402
from allen_diameter.analysis import step2_render as SR  # noqa: E402

SEED = 12345  # the camera noise of R1
REPORT_PACKAGES = ("numpy", "scipy")
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


# ------------------------------------------------------------ fixtures ---

SIGMA0 = 0.080                      # um, the default in-focus width
THETA = (0.1536, 0.3924, 0.5620)    # um, the pilot cell's table (SPEC Block 4, amendment 4)
GAMMA = 0.79                        # um per um
KS = np.arange(-3, 4, dtype=np.int64)


def make_twin(d_um=0.4, mu=1.0, cx=100.03, cy=50.02, theta=0.3, phi=0.1, ks=KS, node_set="near"):
    ks = np.asarray(ks, dtype=np.int64)
    n = ks.size
    arrs = dict(B_gl=np.full(n, 200.0), noise_sd_gl=np.full(n, 3.0), omega_um2=np.full(n, 0.1),
                se_omega_um2=np.full(n, 0.01), area_gl_um=np.full(n, 10.0))
    for a in arrs.values():
        a.setflags(write=False)
    ks.setflags(write=False)
    return Twin(cell_id=7, node_id=11, set=node_set, role="train", d_um=float(d_um), mu_per_um=float(mu),
                k_star=0, z_sub_um=0.0, cx_um=float(cx), cy_um=float(cy), theta_rad=float(theta),
                phi_rad=float(phi), ks=ks, status=("ok",) * n, **arrs)


def make_cfg(patch_half_um, fast=False):
    cfg = default_config()
    cfg = dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, patch_half_um=patch_half_um))
    if fast:   # reduced condenser quadrature of the hybrid (module docstring: what it does not cover)
        cfg = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, ray_n_rho=4, ray_n_psi=8))
    return cfg


def near_rcfg(cfg, theta=THETA):
    return SR.trial_renderer(cfg.renderer, SIGMA0, theta, None, cfg.step2, "near")


def cross_profile(tau, stack, twin):
    """(n_planes, H) the mean over columns, and v (H,) um, the rows' offsets from the axis (heading 0)."""
    rows = (stack.top + np.arange(tau.shape[1])) * stack.res0_um - twin.cy_um
    return tau.mean(axis=2), rows


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    """R2: the trial table's fields and its reading."""
    cfg = default_config()
    s2 = cfg.step2
    base = cfg.renderer
    rn = SR.trial_renderer(base, SIGMA0, THETA, None, s2, "near")
    rf = SR.trial_renderer(base, SIGMA0, list(THETA), GAMMA, s2, "far")
    knots = (0.0,) + tuple(s2.knots_um)
    for r in (rn, rf):
        assert r.kernel_family == s2.kernel_family == "gaussian_table_spline", r.kernel_family
        assert r.kernel_table_delta_um == knots, r.kernel_table_delta_um
        assert r.kernel_table_sigma_um == (SIGMA0,) + THETA, r.kernel_table_sigma_um
        assert r.sigma_r0_um == SIGMA0 and r.absorption == s2.absorption == "hybrid"
        for f in dataclasses.fields(base):    # every other field is the given configuration's
            if f.name not in ("kernel_family", "kernel_table_delta_um", "kernel_table_sigma_um", "sigma_r0_um",
                              "absorption", "kernel_continuation", "kernel_continuation_slope"):
                assert getattr(r, f.name) == getattr(base, f.name), f.name
        dataclasses.replace(cfg, renderer=r).validate()
        # the knots read back: sqrt(s0^2 + (theta^2 - s0^2)) is exact to float64 roundoff
        check_close("knot_values_" + r.kernel_continuation, kernel.sigma_r(np.array(knots), r),
                    np.array((SIGMA0,) + THETA), rtol=1e-12)
    assert rn.kernel_continuation == "tangent", rn.kernel_continuation
    assert rn.kernel_continuation_slope == base.kernel_continuation_slope   # gamma ignored
    assert rf.kernel_continuation == "linear" and rf.kernel_continuation_slope == GAMMA
    # beyond the last knot: "linear" is theta_3 + gamma (|delta| - 0.84); "tangent" is D-265's continuation of G
    dd = np.array([0.98, 1.12, 1.28])
    check_close("far_linear", kernel.sigma_r(dd, rf), THETA[-1] + GAMMA * (dd - 0.84), rtol=1e-12)
    sp = kernel.growth_spline(rn)
    g = float(sp(0.84)) + float(sp(0.84, 1)) * (dd - 0.84)
    check_close("near_tangent", kernel.sigma_r(dd, rn), np.sqrt(SIGMA0 ** 2 + g), rtol=1e-12)
    # D-265's numbers on the pilot table (SPEC Block 4, amendment 4: 0.618, 0.669, 0.723 um, log digits)
    check_close("near_tangent_D265", kernel.sigma_r(dd, rn), [0.618, 0.669, 0.723], atol=5e-4)


def test_reference():
    """R1: render_twin then camera_chain == synthetic_block, bit for bit (full hybrid quadrature)."""
    cfg = make_cfg(1.5)
    twin = make_twin()
    r = near_rcfg(cfg)
    z_ax = 0.07
    stack = SR.render_twin(twin, z_ax, r, cfg, cfg.step2)
    img = camera.camera_chain(stack.tau, r, np.random.default_rng(SEED))
    # the reference, built independently of twin_tube: the tube and patch written out from the contract
    tube = geometry.Tube(c=(twin.cx_um, twin.cy_um, z_ax), r=twin.d_um / 2, phi=twin.phi_rad, theta=twin.theta_rad,
                         aspect=cfg.renderer.cross_section_aspect, half_length=cfg.step2.tube_half_length_um,
                         end_cut=cfg.renderer.end_cut)
    left, top, w, h = calibration_scene.patch_bounds(twin.cx_um, twin.cy_um, cfg.calibration.patch_half_um,
                                                     cfg.acquisition.res0_um)
    pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))
    ref, ks, valid, frame = render.synthetic_block(tube, twin.mu_per_um, twin.ks, left, top, w, h,
                                                   dataclasses.replace(cfg, renderer=r),
                                                   np.random.default_rng(SEED), pad)
    assert img.dtype == ref.dtype == np.uint8 and img.shape == ref.shape, (img.shape, ref.shape)
    assert_array_equal(img, ref, err_msg="camera_chain(render_twin) vs synthetic_block")
    assert_array_equal(stack.ks, ks)
    assert (stack.left, stack.top, stack.res0_um) == (frame.left, frame.top, frame.res_um_px)
    assert int(ref.min()) < int(ref.max()), "the block shows no tube"


def test_convergence():
    raise unittest.SkipTest("not a discretisation of its own: Block 4's grid and slab convergence is "
                            "test_smoke_render's; this block only composes its calls")


def test_invariants():
    """R3 bounds, tau = 1 far away, focus dip; R4 theta_3 widens the dip at 3 planes, focus within 1e-3."""
    cfg = make_cfg(6.5, fast=True)
    twin = make_twin(theta=0.0, phi=0.0, cx=100.0, cy=50.0)     # flat, heading along x: rows are across
    r = near_rcfg(cfg)
    stack = SR.render_twin(twin, 0.0, r, cfg, cfg.step2)
    tau = stack.tau
    assert tau.min() > -0.1 and tau.max() <= 1.0 + 1e-12, (tau.min(), tau.max())
    prof, v = cross_profile(tau, stack, twin)
    # far from the tube: beyond r + 8 sigma_max, sigma_max the widest kernel any slab meets on these planes
    dmax = float(np.max(np.abs(KS))) * cfg.acquisition.dz_um + twin.d_um / 2
    smax = float(kernel.sigma_r(dmax, r))
    far = np.abs(v) >= twin.d_um / 2 + 8.0 * smax
    assert far.sum() >= 4, "no far rows: sigma_max %.3f um" % smax
    dev = float(np.max(np.abs(tau[:, far, :] - 1.0)))
    print("    R3: tau in [%.4f, %.6f]; %d far rows (|v| >= %.2f um), max |tau - 1| there %.2e"
          % (tau.min(), tau.max(), int(far.sum()), twin.d_um / 2 + 8 * smax, dev))
    assert dev <= 1e-12, dev
    i0, i3m, i3p = [int(np.flatnonzero(KS == k)[0]) for k in (0, -3, 3)]
    depth = 1.0 - prof.min(axis=1)
    print("    R3: dip depth by plane %s" % np.array2string(depth, precision=4))
    assert depth[i0] > depth[i3m] and depth[i0] > depth[i3p], depth
    # R4: theta_3 larger by 0.02 um (4 x step2.diff_step_knot_um). The not-a-knot spline is global, so theta_3
    # also moves sigma_r at the focus plane's slabs (|delta| <= r = 0.2 um): measured here (fast quadrature, this
    # tube) max |d tau| at focus 1.9e-4, 7.5e-4, 2.1e-3, 4.8e-3 for +0.005, 0.02, 0.05, 0.1 um; the 1e-3 of the
    # oracle holds for steps up to about 0.025 um on a d = 0.4 um tube, not for large changes of theta_3
    r2 = near_rcfg(cfg, THETA[:2] + (THETA[2] + 0.02,))
    tau2 = SR.render_twin(twin, 0.0, r2, cfg, cfg.step2).tau
    prof2, _ = cross_profile(tau2, stack, twin)

    def width(p):
        a = 1.0 - p
        return math.sqrt(float(np.sum(a * v ** 2) / np.sum(a)))

    w1, w2 = width(prof[i3p]), width(prof2[i3p])
    d0 = float(np.max(np.abs(tau2[i0] - tau[i0])))
    print("    R4: rms half-width at k = +3: %.4f -> %.4f um; focus plane max |d tau| %.2e" % (w1, w2, d0))
    assert w2 > w1, (w1, w2)
    assert width(prof2[i3m]) > width(prof[i3m])
    assert d0 <= 1e-3, d0


def test_contract():
    """R5: shapes, dtypes, the patch; twin_tube's fields; a ks override."""
    cfg = make_cfg(1.5, fast=True)
    s2 = cfg.step2
    twin = make_twin()
    tube = SR.twin_tube(twin, 0.07, cfg, s2)
    assert isinstance(tube, geometry.Tube)
    assert tube.c == (twin.cx_um, twin.cy_um, 0.07) and tube.r == twin.d_um / 2, (tube.c, tube.r)
    assert (tube.phi, tube.theta, tube.aspect, tube.half_length, tube.end_cut) == (
        twin.phi_rad, twin.theta_rad, cfg.renderer.cross_section_aspect, s2.tube_half_length_um,
        cfg.renderer.end_cut)
    r = near_rcfg(cfg)
    st = SR.render_twin(twin, 0.0, r, cfg, s2)
    left, top, w, h = calibration_scene.patch_bounds(twin.cx_um, twin.cy_um, 1.5, cfg.acquisition.res0_um)
    assert isinstance(st, SR.TwinStack)
    assert st.tau.shape == (KS.size, h, w) and st.tau.dtype == np.float64, (st.tau.shape, st.tau.dtype)
    assert np.all(np.isfinite(st.tau))
    assert st.ks.dtype == np.int64 and st.ks.shape == (KS.size,)
    assert_array_equal(st.ks, twin.ks)
    assert (st.left, st.top) == (left, top) and type(st.left) is int and type(st.top) is int
    assert isinstance(st.res0_um, float) and st.res0_um == cfg.acquisition.res0_um
    st2 = SR.render_twin(twin, 0.0, r, cfg, s2, ks=[0, 2])
    assert st2.tau.shape == (2, h, w) and st2.ks.dtype == np.int64
    assert_array_equal(st2.ks, [0, 2])
    # the same planes as in the full call, up to the FFT's plane chunking (roundoff)
    check_close("ks_override", st2.tau, st.tau[[3, 5]], rtol=0, atol=1e-12)
    try:
        st.tau = None
        raise AssertionError("TwinStack is not frozen")
    except dataclasses.FrozenInstanceError:
        pass


def test_determinism():
    """R5: two calls bit for bit."""
    cfg = make_cfg(1.5, fast=True)
    twin = make_twin()
    r = SR.trial_renderer(cfg.renderer, SIGMA0, THETA, GAMMA, cfg.step2, "far")
    a = SR.render_twin(twin, 0.03, r, cfg, cfg.step2)
    b = SR.render_twin(twin, 0.03, r, cfg, cfg.step2)
    assert_array_equal(a.tau, b.tau)
    assert_array_equal(a.ks, b.ks)
    assert (a.left, a.top, a.res0_um) == (b.left, b.top, b.res0_um)


def test_edge_cases():
    """R2: each refusal; a non-finite axis depth."""
    cfg = default_config()
    s2, base = cfg.step2, cfg.renderer
    bad = [
        ("stage", dict(stage="mid")),
        ("theta length", dict(theta_um=THETA[:2])),
        ("theta length", dict(theta_um=THETA + (0.7,))),
        ("theta nan", dict(theta_um=(0.15, float("nan"), 0.56))),
        ("theta inf", dict(theta_um=(0.15, 0.39, float("inf")))),
        ("theta zero", dict(theta_um=(0.0, 0.39, 0.56))),
        ("theta negative", dict(theta_um=(0.15, -0.39, 0.56))),
        ("sigma0 zero", dict(sigma_r0_um=0.0)),
        ("sigma0 nan", dict(sigma_r0_um=float("nan"))),
        ("sigma0 negative", dict(sigma_r0_um=-0.08)),
        ("far gamma None", dict(stage="far", gamma=None)),
        ("far gamma nan", dict(stage="far", gamma=float("nan"))),
        ("far gamma inf", dict(stage="far", gamma=float("inf"))),
        ("far gamma negative", dict(stage="far", gamma=-0.01)),
    ]
    for name, over in bad:
        kw = dict(rcfg=base, sigma_r0_um=SIGMA0, theta_um=THETA, gamma=GAMMA, s2cfg=s2, stage="near")
        kw.update(over)
        try:
            SR.trial_renderer(**kw)
        except ValueError:
            continue
        raise AssertionError("no ValueError for %s" % name)
    # accepted: "near" with gamma None (ignored), "far" with gamma 0
    SR.trial_renderer(base, SIGMA0, THETA, None, s2, "near")
    assert SR.trial_renderer(base, SIGMA0, THETA, 0.0, s2, "far").kernel_continuation_slope == 0.0
    try:
        SR.twin_tube(make_twin(), float("nan"), cfg, s2)
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for a non-finite axis depth")


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
