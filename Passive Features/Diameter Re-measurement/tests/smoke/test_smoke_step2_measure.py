"""Smoke test for a twin's measurement -- Block 55 in specs/SPEC.md (section 5, "Kernel calibration, step 2";
amendment 1 of 2026-11-27).

Oracles M1 to M7 of the entry. The stacks are rendered here by render.render_transmittance with the default
renderer (absorption "partition_vertical", the partition absorption, for time; Block 54 and its hybrid are not
imported) on small patches; the Twin and TwinStack objects are constructed here (Block 53's and 54's functions are
not called).

Shortcuts and what they do not cover
    near patch +-8 um (default 25 um): the stripe support and the background on a smaller square only
    far patch +-10.5 um, a single far tube: the far line (+-far_profile_half_um = 10 um) fits, but neighbours,
        the soma and a cell's scene are not exercised
    gof_check off in M2 only: without noise the background SD is near 0 and the goodness of fit drops most planes;
        the scan's code path with the check on is covered by M3 to M7

Run
    python tests/smoke/test_smoke_step2_measure.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table. Exit code 1 if any check failed.
    python -m pytest tests/smoke/test_smoke_step2_measure.py -q

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import dataclasses
import math
import importlib.metadata
import platform
import re
import sys
import time
import traceback
import unittest
import warnings
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
from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import calibration_scene as CS  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.model import camera  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402

SEED = 20261127  # Step2Config.seed
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


# ---------------------------------------------------------------- checks ---

NEAR_HALF_UM = 8.0      # um: the near patch (see the module docstring)
FAR_HALF_UM = 10.5      # um: the far patch, just wider than the far line's half-length
_CACHE = {}


def _cfg(**cal):
    cfg = default_config()
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def _render(tube, mu, ks, cx, cy, half_um, cfg):
    """TwinStack of the planes ks on the patch about (cx, cy): render.synthetic_block's steps before the camera."""
    acq = cfg.acquisition
    left, top, width, height = CS.patch_bounds(cx, cy, half_um, acq.res0_um)
    pad = int(math.ceil(cfg.renderer.pad_um / acq.res0_um))
    f = R.fine_factor(tube.d, cfg.renderer, acq.res0_um)
    grid, inner = R.block_fine_grid(left, top, width, height, acq.res0_um, f, pad)
    res = R.render_transmittance(tube, mu, np.asarray(ks) * acq.dz_um, grid, cfg.renderer, acq.light_direction,
                                 None, reduce=lambda plane: R.pixel_integrate(plane[inner[0], inner[1]], f))
    return TwinStack(tau=np.asarray(res.tau, dtype=float), ks=np.asarray(ks, dtype=np.int64), left=int(left),
                     top=int(top), res0_um=float(acq.res0_um))


def _twin(ks, B, sd, set_="near", d=0.2, mu=1.5, cx=0.02, cy=-0.03, z=0.04, theta=0.7, phi=0.0, k_star=0,
          B_pilot=210.0, d_pilot=None, alpha_pilot=None):
    n = len(ks)
    zero = np.zeros(n)
    d_pilot = d if d_pilot is None else d_pilot
    alpha_pilot = mu * d / math.cos(phi) if alpha_pilot is None else alpha_pilot
    return Twin(cell_id=7, node_id=11, set=set_, role="train", d_um=d, mu_per_um=mu, k_star=int(k_star),
                z_sub_um=z, cx_um=cx, cy_um=cy, theta_rad=theta, phi_rad=phi, B_pilot_gl=B_pilot,
                d_pilot_um=d_pilot, alpha_pilot=alpha_pilot, ks=np.asarray(ks, np.int64),
                B_gl=np.asarray(B, float), noise_sd_gl=np.asarray(sd, float), omega_um2=zero, se_omega_um2=zero,
                area_gl_um=zero, status=("ok",) * n)


def _near():
    """A thin flat tube (d 0.2 um, mu 1.5 /um, heading 0.7 rad) rendered on the near patch over k = -3..3."""
    if "near" not in _CACHE:
        cfg = _cfg(patch_half_um=NEAR_HALF_UM)
        d, mu, c, theta = 0.2, 1.5, (0.02, -0.03, 0.04), 0.7
        tube = G.Tube(c, 0.5 * d, 0.0, theta, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
        P = cfg.calibration.offsets_planes
        _CACHE["near"] = (cfg, tube, _render(tube, mu, np.arange(-P, P + 1), c[0], c[1], NEAR_HALF_UM, cfg))
    return _CACHE["near"]


def test_known_answer():
    # M1: one background and one SD for every plane: camera_planes equals camera_chain plane by plane on the
    # rendered stack with the matching per-plane generators, bit for bit (jpeg on, the default renderer)
    cfg, _tube, st = _near()
    rc = cfg.renderer
    n = st.tau.shape[0]
    B = rc.black_level_gl + rc.gain * 180.0
    tw = _twin(st.ks, np.full(n, B), np.full(n, 2.5))
    blk, valid = M.camera_planes(st, tw.B_gl, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))
    rq = dataclasses.replace(rc, background_B_gl=180.0, noise_sd_gl=2.5)
    ref = np.stack([camera.camera_chain(st.tau[q], rq, g) for q, g in enumerate(M.noise_rngs(tw, n, cfg.step2, 0))])
    assert valid.all() and blk.dtype == np.uint8 and blk.shape == st.tau.shape
    assert_array_equal(blk, ref)


def _m2_case():
    """scan_node's scan of a flat thin phantom (noise off, gof_check off) and the twin built from its pilot."""
    if "m2" in _CACHE:
        return _CACHE["m2"]
    cfg0 = _cfg(patch_half_um=NEAR_HALF_UM, gof_check=False)
    cfg = dataclasses.replace(cfg0, renderer=dataclasses.replace(cfg0.renderer, noise_sd_gl=0.0))
    d, mu, c, theta = 0.2, 1.5, np.array([0.02, -0.03, 0.04]), 0.7
    tube = G.Tube(tuple(c), 0.5 * d, 0.0, theta, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
    t = G.axis_direction(0.0, theta)
    br = Branch.from_points(c[None, :] + np.arange(-10, 11)[:, None] * 1.18 * t[None, :], 0.5 * d)
    pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))
    rng = make_rng()

    def provider(left, top, width, height, k_lo, k_hi):
        return R.synthetic_block(tube, mu, np.arange(k_lo, k_hi + 1), left, top, width, height, cfg, rng, pad)
    res, scan, reasons = CAL.scan_node(br, 10, provider, cfg)
    assert scan is not None and scan.set == "near", reasons
    st = _render(tube, mu, scan.ks, res.cx_um, res.cy_um, NEAR_HALF_UM, cfg)
    n = scan.ks.size
    Bg = cfg.renderer.black_level_gl + cfg.renderer.gain * cfg.renderer.background_B_gl
    tw = _twin(scan.ks, np.full(n, Bg), np.full(n, 5.0), d=d, mu=float(res.mu_hat_per_um), cx=float(res.cx_um),
               cy=float(res.cy_um), z=float(res.z_sub_um), theta=float(res.theta_rad), phi=float(res.phi_rad),
               k_star=int(res.k_star), B_pilot=float(res.B_bar), d_pilot=float(res.d_hat_um),
               alpha_pilot=float(res.alpha_hat))
    _CACHE["m2"] = (cfg, br, scan, st, tw)
    return _CACHE["m2"]


def _compare_m2(ms, scan, name):
    assert_array_equal(ms.ks, scan.ks)
    both = (np.array(ms.status) == "ok") & (np.array(scan.status) == "ok")
    assert both.sum() >= 3, (scan.status, ms.status)
    check_close(name, ms.omega_um2[both], scan.omega_um2[both], rtol=0.0, atol=1e-9)


def test_reference():
    # M2 as SPEC.md writes it: noise off, a flat phantom: measure_twin equals calibration.scan_node's scan of the
    # same tube rendered through render.synthetic_block with noise_sd_gl = 0, the core widths to 1e-9 um^2 where
    # both are ok, the twin's pilot values those of scan_node's own pilot (amendment 1). The scans' branches differ:
    # scan_node's is the phantom's trace along the TRUE axis, the twin's is twin_scene's straight branch along the
    # PILOT's axis; the stripe support follows the branch, so a plane's background can differ (reported as ISSUE).
    cfg, _br, scan, st, tw = _m2_case()
    _compare_m2(M.measure_twin(tw, st, cfg, cfg.step2, noise=False), scan, "M2_omega")


def test_reference_same_branch():
    # M2a: the same comparison with twin_scene's branch replaced by scan_node's own branch (the node stays
    # twin_scene's): isolates the camera, the patch and the node from the branch. Exact to 1e-9 um^2.
    cfg, br, scan, st, tw = _m2_case()
    orig = M.twin_scene

    def same_branch(twin, cfg_, s2cfg):
        node, _b, _s = orig(twin, cfg_, s2cfg)
        return node, br, Scene.from_branch(br)
    M.twin_scene = same_branch
    try:
        ms = M.measure_twin(tw, st, cfg, cfg.step2, noise=False)
    finally:
        M.twin_scene = orig
    _compare_m2(ms, scan, "M2a_omega")


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the block composes a camera chain and a scan")


def test_invariants():
    # M3: jpeg off, per-plane backgrounds 180..216 gl and SD 3 gl: the measured background equals B_gl within
    # 0.5 gl per plane and bg_sd_gl is within 30 % of 3 gl. The support has >= 500 pixels per plane
    # (stripe_min_pixels, asserted below), so the SE of a mean is <= 3 / sqrt(500) = 0.13 gl and the relative SE of
    # an SD <= 1 / sqrt(2 x 500) = 3 %: both tolerances are > 3.5 SE (quantisation adds 1/12 gl^2 to the variance,
    # +0.5 % on the SD).
    cfg0, tube, st = _near()
    cfg = dataclasses.replace(cfg0, renderer=dataclasses.replace(cfg0.renderer, jpeg=False))
    n = st.ks.size
    B = 180.0 + 6.0 * np.arange(n)
    tw = _twin(st.ks, B, np.full(n, 3.0))
    node, branch, scene = M.twin_scene(tw, cfg, cfg.step2)
    sup, _info = CAL.support_masks(st.tau.shape[1:], CS_frame(st), st.ks, node, branch, scene, cfg)
    assert sup.reshape(n, -1).sum(axis=1).min() >= 500, sup.reshape(n, -1).sum(axis=1)
    ms = M.measure_twin(tw, st, cfg, cfg.step2)
    assert np.all(np.isfinite(ms.B_bar)), ms.status
    assert np.max(np.abs(ms.B_bar - B)) <= 0.5, ms.B_bar - B
    assert np.max(np.abs(ms.bg_sd_gl / 3.0 - 1.0)) <= 0.30, ms.bg_sd_gl
    # M5: twin_statistics against the replicates' scans; the NaN rule on a plane made invalid (B_gl NaN); R = 1
    # without noise
    B5 = B.copy()
    B5[0] = np.nan
    tw5 = _twin(st.ks, B5, np.full(n, 3.0))
    s2 = cfg.step2
    reps = [M.measure_twin(tw5, st, cfg, s2, replicate=r) for r in range(s2.noise_replicates)]
    om = np.vstack([r.omega_um2 for r in reps])
    out = M.twin_statistics(tw5, st, cfg, s2)
    cnt = np.isfinite(om).sum(axis=0)
    need = math.ceil(s2.noise_replicates / 2.0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # nanmean / nanstd of the all-NaN plane 0
        exp_mean = np.where(cnt >= need, np.nanmean(om, axis=0), np.nan)
        exp_sd = np.where(cnt >= 2, np.nanstd(om, axis=0, ddof=1), np.nan)
        exp_area = np.where(np.isfinite(np.vstack([r.area_gl_um for r in reps])).any(axis=0),
                            np.nanmean(np.vstack([r.area_gl_um for r in reps]), axis=0), np.nan)
        exp_bsd = np.nanmean(np.vstack([r.bg_sd_gl for r in reps]), axis=0)
    assert set(out) == {"omega_um2", "omega_sd_um2", "area_gl_um", "bg_sd_gl", "n_ok"}, set(out)
    assert_array_equal(out["n_ok"], cnt)
    assert out["n_ok"].dtype.kind == "i" and out["n_ok"][0] == 0 and np.isnan(out["omega_um2"][0])
    assert np.isnan(out["bg_sd_gl"][0]) and np.isnan(out["omega_sd_um2"][0])
    check_close("M5_mean", out["omega_um2"], exp_mean, rtol=1e-12)
    check_close("M5_sd", out["omega_sd_um2"], exp_sd, rtol=1e-12)
    check_close("M5_area", out["area_gl_um"], exp_area, rtol=1e-12)
    check_close("M5_bgsd", out["bg_sd_gl"], exp_bsd, rtol=1e-12)
    assert np.sum(cnt >= 2) >= 3, cnt
    one = M.twin_statistics(tw5, st, cfg, s2, noise=False)
    ref = M.measure_twin(tw5, st, cfg, s2, noise=False)
    assert np.all(one["n_ok"] <= 1)
    check_close("M5_R1", one["omega_um2"], ref.omega_um2, rtol=0.0)
    assert np.all(np.isnan(one["omega_sd_um2"]))


def CS_frame(st):
    from allen_image_io import CropFrame
    return CropFrame(int(st.left), int(st.top), 0, float(st.res0_um))


def test_contract():
    # M6: a far twin (d 0.7 um, Allen's far set) goes through far_plane_scan: set "far", 2 x 16 + 1 planes. Its
    # values are not checked against an oracle here (Block 15's own test does)
    cfg = _cfg(patch_half_um=FAR_HALF_UM)
    d, mu, c, theta = 0.7, 0.5, (0.02, -0.03, 0.04), 0.7
    tube = G.Tube(c, 0.5 * d, 0.0, theta, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
    P = cfg.calibration.far_offsets_planes
    assert P == 16
    st = _render(tube, mu, np.arange(-P, P + 1), c[0], c[1], FAR_HALF_UM, cfg)
    n = st.ks.size
    Bg = cfg.renderer.black_level_gl + cfg.renderer.gain * cfg.renderer.background_B_gl
    tw = _twin(st.ks, np.full(n, Bg), np.full(n, 3.0), set_="far", d=d, mu=mu)
    ms = M.measure_twin(tw, st, cfg, cfg.step2)
    assert ms.set == "far" and ms.ks.size == 33 and len(ms.status) == 33, (ms.set, ms.ks)
    assert ms.omega_um2.dtype == np.float64 and ms.omega_um2.shape == (33,)
    assert sum(s == "ok" for s in ms.status) >= 10, ms.status
    # the refusal: stack.ks not the scan's stage_planes
    try:
        M.measure_twin(tw, TwinStack(st.tau[1:], st.ks[1:], st.left, st.top, st.res0_um), cfg, cfg.step2)
    except ValueError:
        pass
    else:
        raise AssertionError("measure_twin accepted planes other than stage_planes")
    # node attributes are the pilot's (amendment 1)
    node, branch, _scene = M.twin_scene(tw, cfg, cfg.step2)
    assert (node.B_bar, node.mu_hat_per_um, node.d_hat_um, node.alpha_hat) == (210.0, mu, d, mu * d)
    assert np.allclose(branch.radius_um, 0.5 * d)
    assert abs(branch.xyz_um[:, 0].max() - branch.xyz_um[:, 0].min()
               - 2 * math.floor(40.0 / 1.18) * 1.18 * math.cos(theta)) < 1e-9


def test_determinism():
    # M4: two calls bit for bit; another replicate differs; a different stack with the same seed uses the same
    # noise field: with bit_depth 16 and jpeg off the difference of the two blocks is the difference of the grey
    # mappings to 1 gl (each block's rounding contributes at most 0.5 gl)
    cfg, _tube, s1 = _near()
    rc = dataclasses.replace(cfg.renderer, bit_depth=16, jpeg=False)
    s2 = TwinStack(1.0 - 2.0 * (1.0 - s1.tau), s1.ks, s1.left, s1.top, s1.res0_um)   # a twice deeper dip
    n = s1.ks.size
    B = np.full(n, rc.black_level_gl + rc.gain * 3000.0)
    tw = _twin(s1.ks, B, np.full(n, 20.0))
    a = M.camera_planes(s1, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))[0]
    b = M.camera_planes(s1, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))[0]
    c = M.camera_planes(s1, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 1))[0]
    d = M.camera_planes(s2, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))[0]
    assert_array_equal(a, b)
    assert np.any(a != c)
    gm = (camera.grey_mapping(s1.tau, dataclasses.replace(rc, background_B_gl=3000.0))
          - camera.grey_mapping(s2.tau, dataclasses.replace(rc, background_B_gl=3000.0)))
    assert np.max(np.abs((a.astype(float) - d.astype(float)) - gm)) <= 1.0
    # and through the scan: two measure_twin calls bit for bit
    x = M.measure_twin(tw, s1, cfg, cfg.step2)
    y = M.measure_twin(tw, s1, cfg, cfg.step2)
    assert_array_equal(x.omega_um2, y.omega_um2)


def test_edge_cases():
    # M7: a non-finite B_gl (and a B_gl at the black level) gives an invalid, all-zero plane and a status that is
    # not "ok"
    cfg, _tube, st = _near()
    rc = cfg.renderer
    n = st.ks.size
    B = np.full(n, rc.black_level_gl + rc.gain * 180.0)
    B[1] = np.nan
    B[3] = rc.black_level_gl
    tw = _twin(st.ks, B, np.full(n, 2.5))
    blk, valid = M.camera_planes(st, B, tw.noise_sd_gl, rc, M.noise_rngs(tw, n, cfg.step2, 0))
    assert_array_equal(valid, [True, False, True, False, True, True, True])
    assert not blk[1].any() and not blk[3].any() and blk[0].any()
    ms = M.measure_twin(tw, st, cfg, cfg.step2)
    assert ms.status[1] != "ok" and ms.status[3] != "ok", ms.status
    assert np.isnan(ms.omega_um2[1]) and np.isnan(ms.omega_um2[3])
    # a negative noise SD on a valid plane is refused
    try:
        M.camera_planes(st, B, np.full(n, -1.0), rc, M.noise_rngs(tw, n, cfg.step2, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("a negative SD was accepted")


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
