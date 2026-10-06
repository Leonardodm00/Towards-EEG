"""Smoke test for the renderer and the camera chain -- Block 4 in specs/SPEC.md
(procedure Eqs. 5-6, s.3.6; impl-handoff (S4) and FFT form).

Checks
    test_known_answer   sigma_r at the knots, between them and on the three
                        continuations; C1 (the partition telescopes per column,
                        both light directions); C2 (one slab, constant kernel,
                        flat tube = the discrete convolution of the sampled
                        1 - T with g_sigma)
    test_reference      fft (split off) vs direct (scipy.ndimage.gaussian_filter);
                        fft with the near/far split vs direct; agreement with
                        checks/stack_geometry_check.render_planes and sigma_r;
                        pixel_integrate vs explicit block means
    test_convergence    C2 converges to Block 3's model_profile as h -> 0;
                        the node dip settles when h_g and dzeta are halved
    test_invariants     mass: the dip volume of every plane equals the absorbed
                        light; a 90-degree rotation of the tube rotates the image;
                        light reversed = depth mirrored with theta + pi; faint
                        limit: partition -> linear; mu = 0 gives tau = 1
    test_contract       synthetic_block in the fetch_zblock contract; the dip
                        sits on the tube; camera background mean and SD; JPEG
                        tables round trip; integer values in range
    test_determinism    a synthetic block is bit-identical under one seed
    test_edge_cases     tube off the grid; invalid inputs raise

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_render.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import io
import math
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import jpeg_tables  # noqa: E402
from allen_diameter.model import camera as C  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402
from allen_diameter.model import tube_model as M  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy", "Pillow")
PX = 0.1144


def rcfg(**changes):
    return dataclasses.replace(default_config().renderer, **changes)


def exact_rcfg(**changes):
    """No near/far split and generous FFT padding: the FFT path is then exact
    to roundoff for the sampled object."""
    base = dict(fft_split_sigma_um=1e9, fft_wrap_sigmas=10.0)
    base.update(changes)
    return rcfg(**base)


def centred_grid(half_um, h, cx=0.0, cy=0.0):
    """Square grid of odd size, symmetric about (cx, cy)."""
    m = int(round(half_um / h))
    n = 2 * m + 1
    return R.FineGrid(cx - m * h, cy - m * h, h, n, n)


def load_check_script():
    spec = importlib.util.spec_from_file_location("stack_geometry_check", str(WS / "checks" / "stack_geometry_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    rc = rcfg()
    knots = np.array(rc.kernel_table_delta_um)
    vals = np.array(rc.kernel_table_sigma_um)
    assert np.array_equal(K.sigma_r(knots, rc), vals) and np.array_equal(K.sigma_r(-knots, rc), vals)
    mids = 0.5 * (knots[1:] + knots[:-1])
    assert np.allclose(K.sigma_r(mids, rc), 0.5 * (vals[1:] + vals[:-1]), rtol=0, atol=1e-15)
    far = np.array([1.0, 2.5, 7.0])
    d_max, s_max = knots[-1], vals[-1]
    for name, gamma in (("linear", rc.kernel_continuation_slope), ("frozen", 0.0), ("proportional", s_max / d_max)):
        got = K.sigma_r(far, rcfg(kernel_continuation=name))
        assert np.allclose(got, s_max + gamma * (far - d_max), rtol=0, atol=1e-14), name
    # C1: sum_j Delta A_j = 1 - exp(-sum_j a_j) in every column, both light directions
    tube = G.Tube((0.01, -0.02, 0.03), 0.45, math.radians(35.0), math.radians(110.0), 1.0, 1.2, "axial")
    grid = centred_grid(1.6, PX / 4)
    zeta = R.slab_grid(tube, grid, 0.02)
    a = R.absorbed_fractions(tube, 2.0, grid, zeta, 0.02, +1, "linear")
    for light in (+1, -1):
        dA = R.absorbed_fractions(tube, 2.0, grid, zeta, 0.02, light, "partition_vertical")
        err = np.max(np.abs(dA.sum(axis=0) - (-np.expm1(-a.sum(axis=0)))))
        assert err <= 1e-12, ("C1", light, err)
    # C2: one slab holding a flat tube, constant kernel: the discrete convolution of 1 - T
    sig = 0.099
    rc2 = exact_rcfg(kernel_table_sigma_um=(sig,) * 6, sigma_r0_um=sig, kernel_continuation="frozen", dzeta_um=5.0)
    d, mu, y0 = 0.8, 1.2, 0.0137
    flat = G.Tube((0.0, y0, 0.0), 0.5 * d, 0.0, 0.0, 1.0, None, "axial")
    g2 = centred_grid(1.5, PX / 8)
    res = R.render_transmittance(flat, mu, [0.0], g2, rc2, +1, "fft")
    assert res.zeta.size == 1
    ys = g2.ys
    prof = res.tau[0][:, g2.nx // 2]
    T = M.transmittance(ys, d, mu * d, y0)
    g = np.exp(-0.5 * ((ys[:, None] - ys[None, :]) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    disc = 1.0 - g @ ((1.0 - T) * g2.h)
    inner = np.abs(ys) < 0.9          # 6 sigma away from the grid's ends along y
    assert np.max(np.abs(prof - disc)[inner]) <= 1e-12, np.max(np.abs(prof - disc)[inner])


def test_reference():
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(25.0), math.radians(30.0), 1.0, 1.0, "axial")
    grid = centred_grid(1.8, PX / 4)
    zk = np.array([-0.28, 0.0, 0.28])
    ref = R.render_transmittance(tube, 1.0, zk, grid, exact_rcfg(), +1, "direct")
    exact = R.render_transmittance(tube, 1.0, zk, grid, exact_rcfg(), +1, "fft")
    assert np.max(np.abs(exact.tau - ref.tau)) <= 1e-10, ("fft vs direct", np.max(np.abs(exact.tau - ref.tau)))
    split = R.render_transmittance(tube, 1.0, zk, grid, rcfg(), +1, "fft")
    assert split.n_far_pairs > 0, "the default split must exercise the far-field path here"
    # far path: bicubic interpolation from spacing sigma / 8, error ~ (1/8)^4 / 384 of the far field
    assert np.max(np.abs(split.tau - ref.tau)) <= 1e-6, ("split vs direct", np.max(np.abs(split.tau - ref.tau)))
    # checks/stack_geometry_check.py: vertical cut, proportional continuation, dzeta 0.05, light toward +z
    sgc = load_check_script()
    r, phi, theta, mu = 0.25, math.radians(20.0), math.radians(30.0), 1.0
    xs, planes, _J = sgc.render_planes(r, phi, theta, mu, np.zeros(3), [0.0, 0.28], half_xy=2.5, h_g=PX / 8,
                                       dzeta=0.05, pad=1.0, U=0.8)
    g3 = R.FineGrid(float(xs[0]), float(xs[0]), PX / 8, xs.size, xs.size)
    mine = R.render_transmittance(G.Tube((0.0, 0.0, 0.0), r, phi, theta, 1.0, 0.8, "vertical"), mu, [0.0, 0.28], g3,
                                  exact_rcfg(kernel_continuation="proportional", dzeta_um=0.05), +1, "fft")
    assert np.max(np.abs(mine.tau - planes)) <= 1e-10, ("vs stack_geometry_check", np.max(np.abs(mine.tau - planes)))
    deltas = np.linspace(-3, 3, 61)
    assert np.allclose(K.sigma_r(deltas, rcfg(kernel_continuation="proportional")), sgc.sigma_r(deltas), rtol=0, atol=1e-15)
    # pixel integration = explicit block means
    rng = np.random.default_rng(SEED)
    fine = rng.random((2, 12, 18))
    got = R.pixel_integrate(fine, 3)
    want = np.array([[[fine[p, 3 * i:3 * i + 3, 3 * j:3 * j + 3].mean() for j in range(6)] for i in range(4)] for p in range(2)])
    assert np.allclose(got, want, rtol=0, atol=1e-15)


def test_convergence():
    # C2 against the continuous handoff Eq. 11 (Block 3) as h -> 0; point samples meet the
    # tube's edges at varying sub-sample positions, so the decrease is monotone but irregular
    sig, d, mu, y0 = 0.099, 0.8, 1.2, 0.0137
    rc2 = exact_rcfg(kernel_table_sigma_um=(sig,) * 6, sigma_r0_um=sig, kernel_continuation="frozen", dzeta_um=5.0)
    flat = G.Tube((0.0, y0, 0.0), 0.5 * d, 0.0, 0.0, 1.0, None, "axial")
    errs = []
    for f in (8, 16, 32):
        grid = centred_grid(1.5, PX / f)
        prof = R.render_transmittance(flat, mu, [0.0], grid, rc2, +1, "fft").tau[0][:, grid.nx // 2]
        cont = M.model_profile(grid.ys, d, mu * d, y0, sig, 1.0, 256)
        inner = np.abs(grid.ys) < 0.9
        errs.append(float(np.max(np.abs(prof - cont)[inner])))
    assert errs[0] > errs[1] > errs[2] and errs[2] <= 1e-4, ("C2 vs Eq. 11", errs)
    # the node dip settles when h_g and dzeta are halved (tilted thin tube, default kernel)
    tube = G.Tube((0.0, 0.0, 0.0), 0.25, math.radians(20.0), 0.3, 1.0, 1.5, "axial")
    dips = {}
    for f, dz in ((8, 0.04), (16, 0.04), (16, 0.02), (16, 0.01)):
        grid = centred_grid(1.5, PX / f)
        tau = R.render_transmittance(tube, 1.0, [0.0], grid, rcfg(dzeta_um=dz), +1, "fft").tau[0]
        dips[(f, dz)] = 1.0 - tau[grid.ny // 2, grid.nx // 2]
    dh = abs(dips[(16, 0.04)] - dips[(8, 0.04)])
    dz1 = abs(dips[(16, 0.02)] - dips[(16, 0.04)])
    dz2 = abs(dips[(16, 0.01)] - dips[(16, 0.02)])
    assert dz2 < dz1, ("dzeta halving does not converge", dips)
    # observed: 3e-6 (h_g) and 1e-4 (dzeta) at a dip of 0.39; bounds 30x and 5x above
    assert dh <= 1e-4 and dz1 <= 5e-4, ("node dip moves", dips)


def test_invariants():
    rc = rcfg()
    # mass: every plane's dip volume equals the absorbed light (the grid holds the blurred tube)
    # (margin >= 6.5 sigma_max between the tube and the grid's edge: leak < 1e-10)
    tube = G.Tube((0.0, 0.0, 0.0), 0.35, math.radians(40.0), math.radians(15.0), 1.0, 0.8, "axial")
    grid = centred_grid(6.0, PX / 4)
    zk = np.array([-0.28, 0.0, 0.28])
    zeta = R.slab_grid(tube, grid, rc.dzeta_um)
    absorbed = R.absorbed_fractions(tube, 1.5, grid, zeta, rc.dzeta_um, +1).sum() * grid.h ** 2
    for name, cfg in (("exact", exact_rcfg()), ("split", rc)):
        res = R.render_transmittance(tube, 1.5, zk, grid, cfg, +1, "fft")
        vol = (1.0 - res.tau).sum(axis=(1, 2)) * grid.h ** 2
        tol = 1e-10 if name == "exact" else 1e-6       # far path: spline-interpolated field
        assert np.all(np.abs(vol / absorbed - 1) <= tol), (name, vol / absorbed - 1)
    # a 90-degree rotation of the tube about the grid centre rotates the image
    t0 = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(30.0), 0.4, 1.0, 1.2, "axial")
    t90 = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(30.0), 0.4 + 0.5 * math.pi, 1.0, 1.2, "axial")
    g = centred_grid(2.0, PX / 4)
    a0 = R.render_transmittance(t0, 1.0, [0.0, 0.28], g, rc, +1).tau
    a90 = R.render_transmittance(t90, 1.0, [0.0, 0.28], g, rc, +1).tau
    assert np.max(np.abs(a90 - np.rot90(a0, k=-1, axes=(1, 2)))) <= 1e-12, "90-degree rotation"
    # light reversed = depth mirrored about c_z with theta -> theta + pi (exact when the slab
    # lattice spans the depth range exactly, so it mirrors onto itself)
    cz = 0.11
    tp = G.Tube((0.0, 0.0, cz), 0.3, math.radians(35.0), 0.7, 1.0, 1.0, "axial")
    tm = G.Tube((0.0, 0.0, cz), 0.3, math.radians(35.0), 0.7 + math.pi, 1.0, 1.0, "axial")
    X, Y = g.mesh()
    lo, hi, ins = G.column_interval(X, Y, tp)
    dz = (hi[ins].max() - lo[ins].min()) / 40.0
    zk = np.array([cz - 0.3, cz, cz + 0.45])
    up = R.render_transmittance(tp, 2.0, zk, g, rcfg(dzeta_um=dz), +1).tau
    down = R.render_transmittance(tm, 2.0, 2 * cz - zk, g, rcfg(dzeta_um=dz), -1).tau
    assert np.max(np.abs(up - down)) <= 1e-12, ("light/mirror", np.max(np.abs(up - down)))
    assert np.max(np.abs(up - R.render_transmittance(tp, 2.0, zk, g, rcfg(dzeta_um=dz), -1).tau)) > 1e-3, \
        "the light direction must matter for a dark tube"
    # faint limit: the partition tends to the linear sum; its dip is the smaller one
    mu, d = 1e-3, 0.6
    tf = G.Tube((0.0, 0.0, 0.0), 0.5 * d, math.radians(20.0), 0.2, 1.0, 1.0, "axial")
    p = R.render_transmittance(tf, mu, [0.0], g, rc, +1).tau
    lin = R.render_transmittance(tf, mu, [0.0], g, rcfg(absorption="linear"), +1).tau
    rel = (1 - lin).max() / (1 - p).max() - 1
    assert 0 < rel <= 2 * mu * d / math.cos(math.radians(20.0)), ("faint limit", rel)
    # mu = 0: nothing is drawn
    assert np.array_equal(R.render_transmittance(tf, 0.0, [0.0], g, rc).tau, np.ones((1, g.ny, g.nx)))


def test_contract():
    cfg = default_config()
    rc = cfg.renderer
    tube = G.Tube((0.3 * PX, -0.2 * PX, 0.05), 0.4, math.radians(10.0), 1.1, 1.0, 10.0, "axial")
    ks = np.arange(-2, 3)
    block, ks_out, valid, frame = R.synthetic_block(tube, 1.0, ks, -20, -15, 41, 31, cfg,
                                                    np.random.default_rng(SEED), pad_px=4)
    assert block.shape == (5, 31, 41) and block.dtype == np.uint8
    assert np.array_equal(ks_out, ks) and valid.dtype == bool and valid.all()
    assert (frame.left, frame.top, frame.downsample, frame.res0_um_px) == (-20, -15, 0, cfg.acquisition.res0_um)
    # placement: without noise and JPEG, the darkest pixel of the node plane is within half a
    # pixel diagonal of the axis line, and the dip is centred on it across the tube
    quiet = dataclasses.replace(cfg, renderer=dataclasses.replace(rc, noise_sd_gl=0.0, jpeg=False))
    img = R.synthetic_block(tube, 1.0, ks, -20, -15, 41, 31, quiet, np.random.default_rng(SEED), pad_px=4)[0][2]
    rows, cols = np.mgrid[0:31, 0:41]
    xg, yg = (frame.left + cols) * PX, (frame.top + rows) * PX
    vg = -(xg - tube.c[0]) * math.sin(tube.theta) + (yg - tube.c[1]) * math.cos(tube.theta)
    i, j = np.unravel_index(np.argmin(img), img.shape)
    assert abs(vg[i, j]) <= 0.75 * PX, ("darkest pixel off the axis", vg[i, j])
    w = np.clip(float(img.max()) - img.astype(float), 0, None)
    centroid = float((w * vg).sum() / w.sum())
    assert abs(centroid) <= 0.02, ("dip not centred on the tube", centroid)
    # camera: flat field statistics before JPEG; integer values in range
    flat = np.ones((20, 80, 80))
    q = C.camera_chain(flat, dataclasses.replace(rc, jpeg=False), np.random.default_rng(SEED))
    assert q.dtype == np.uint8 and q.min() >= 0 and q.max() <= 255
    expect = rc.black_level_gl + rc.gain * rc.background_B_gl
    assert abs(q.mean() - expect) <= 0.02 * rc.noise_sd_gl, (q.mean(), expect)
    assert abs(q.std() / rc.noise_sd_gl - 1) <= 0.02, q.std()   # rounding adds 1/12 gl^2 (0.5 %)
    # JPEG with explicit tables: the tables survive encoding, JSON storage and decoding
    from PIL import Image
    table = [int(2 + (k % 7)) for k in range(64)]
    buf = io.BytesIO()
    Image.fromarray(q[0]).save(buf, format="JPEG", qtables=[table])
    read = jpeg_tables.qtables_from_jpeg(buf.getvalue())
    assert read[0] == table, read
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "qtables.json")
        jpeg_tables.save_qtables(read, path)
        assert jpeg_tables.load_qtables(path) == read
    out = C.jpeg_roundtrip(q[0], qtables=[table])
    assert out.shape == q[0].shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - q[0].astype(int)).mean() < 3.0


def test_determinism():
    cfg = default_config()
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, math.radians(15.0), 0.5, 1.0, 10.0, "axial")
    runs = [R.synthetic_block(tube, 1.0, [0, 1], -12, -12, 25, 25, cfg, np.random.default_rng(SEED), pad_px=3)[0]
            for _ in range(2)]
    assert np.array_equal(runs[0], runs[1])
    other = R.synthetic_block(tube, 1.0, [0, 1], -12, -12, 25, 25, cfg, np.random.default_rng(SEED + 1), pad_px=3)[0]
    assert not np.array_equal(runs[0], other), "the noise must follow the generator"


def test_edge_cases():
    rc = rcfg()
    g = centred_grid(1.0, PX / 4)
    off = G.Tube((30.0, 30.0, 0.0), 0.3, 0.2, 0.1, 1.0, 1.0, "axial")
    res = R.render_transmittance(off, 1.0, [0.0], g, rc)
    assert np.array_equal(res.tau, np.ones((1, g.ny, g.nx))) and res.zeta.size == 0
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, 0.2, 0.1, 1.0, 1.0, "axial")
    bad = [lambda: R.render_transmittance(tube, -1.0, [0.0], g, rc),
           lambda: R.render_transmittance(tube, 1.0, [0.0], g, rc, light_direction=0),
           lambda: R.render_transmittance(tube, 1.0, [0.0], g, rc, backend="gpu"),
           lambda: R.render_transmittance(tube, 1.0, [np.nan], g, rc),
           lambda: R.FineGrid(0.0, 0.0, 0.0, 10, 10),
           lambda: R.FineGrid(0.0, 0.0, 0.1, 0, 10),
           lambda: R.block_fine_grid(0, 0, 10, 10, PX, 2.5),
           lambda: R.fine_factor(1.0, rcfg(h_g_um_thick=0.03), PX),
           lambda: R.pixel_integrate(np.ones((10, 9)), 2),
           lambda: K.sigma_r(np.array([np.inf]), rc),
           lambda: K.sigma_r(0.1, rcfg(kernel_continuation="cubic")),
           lambda: C.quantize(np.array([np.nan]), 8),
           lambda: C.jpeg_roundtrip(np.zeros((4, 4), dtype=np.uint16), quality=80),
           lambda: R.render_transmittance(tube, 1.0, [0.0], R.FineGrid(-1.0, -1.0, 0.05, 41, 41), rc)]
    for k, call in enumerate(bad):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid input %d accepted" % k)
    try:
        R.absorbed_fractions(tube, 1.0, g, np.array([0.0]), 0.02, +1, "ray_world")
    except ValueError:
        pass
    else:
        raise AssertionError("absorbed fractions are not defined in the ray world")
    try:
        K.sigma_r(0.1, rcfg(kernel_family="empirical"))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("a planned option must raise NotImplementedError")


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
