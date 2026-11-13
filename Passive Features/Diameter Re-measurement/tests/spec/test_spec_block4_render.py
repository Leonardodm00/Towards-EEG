"""Spec-derived tests of Block 4 (the renderer: kernel, slabs, planes, camera,
fine grid) -- integration tester, run D1, 2026-11-13. Expected values: the
kernel law and the identities of SPEC Block 4 / section 2.3 written out by
hand, the column intervals from the round tube's quadratic (spec_helpers_D1)
and numpy for the camera statistics; the code under test only produces the
outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block4_render.py -q
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D1 as H

from allen_diameter.config import default_config
from allen_diameter.model import camera as CAM
from allen_diameter.model import render as R
from allen_diameter.model.geometry import Tube
from allen_diameter.model.kernel import sigma_r

CFG = default_config()
RC = CFG.renderer
PX = CFG.acquisition.res0_um


def ref_sigma(delta, rc):
    d = np.abs(np.asarray(delta, dtype=float))
    xs = np.asarray(rc.kernel_table_delta_um, float)
    ys = np.asarray(rc.kernel_table_sigma_um, float)
    dmax, smax = xs[-1], ys[-1]
    if rc.kernel_continuation == "linear":
        g = rc.kernel_continuation_slope
    elif rc.kernel_continuation == "frozen":
        g = 0.0
    else:
        g = smax / dmax
    return np.where(d <= dmax, np.interp(np.minimum(d, dmax), xs, ys), smax + g * (d - dmax))


# ----------------------------------------------------------------- O4.1 ---

@pytest.mark.parametrize("cont", ["linear", "frozen", "proportional"])
def test_kernel_law(cont):
    rc = dataclasses.replace(RC, kernel_continuation=cont)
    knots = np.asarray(rc.kernel_table_delta_um)
    assert_array_equal(sigma_r(knots, rc), np.asarray(rc.kernel_table_sigma_um))   # exact at the knots
    d = np.linspace(-3.0, 3.0, 1201)
    assert_allclose(sigma_r(d, rc), ref_sigma(d, rc), rtol=0, atol=1e-15)
    assert_array_equal(sigma_r(d, rc), sigma_r(-d, rc))                            # even in delta
    if cont == "proportional":
        dm, sm = knots[-1], rc.kernel_table_sigma_um[-1]
        assert_allclose(sigma_r(2 * dm, rc), 2 * sm, rtol=1e-15)


# -------------------------------------------------------- O4.2 slabs, C1 ---

def _small_grid(n=40, h=0.0286, x0=None):
    x0 = -n * h / 2 if x0 is None else x0
    return R.FineGrid(x0, x0, h, n, n)


@pytest.mark.parametrize("absorption", ["partition_vertical", "linear"])
def test_column_sums(absorption):
    phi, theta, r, mu = math.radians(35.0), 0.6, 0.35, 2.5
    c = (0.02, -0.03, 0.1)
    tube = Tube(c, r, phi, theta, 1.0, 1.2, "axial")
    g = _small_grid()
    zeta = R.slab_grid(tube, g, RC.dzeta_um)
    dA = R.absorbed_fractions(tube, mu, g, zeta, RC.dzeta_um, +1, absorption)
    X, Y = np.meshgrid(g.x0 + g.h * np.arange(g.nx), g.y0 + g.h * np.arange(g.ny))
    z_lo, z_hi, ins = H.ref_column(X, Y, np.asarray(c), r, phi, theta, 1.2)
    total = mu * (z_hi - z_lo).reshape(X.shape)
    col = dA.sum(axis=0)
    if absorption == "partition_vertical":
        assert_allclose(col, -np.expm1(-total), rtol=0, atol=1e-12)    # C1
    else:
        assert_allclose(col, total, rtol=0, atol=1e-12)                # linear: DeltaA_j = a_j
    assert np.all(dA >= 0)


def test_absorbed_fractions_refuses_ray_world():
    tube = Tube((0.0, 0.0, 0.0), 0.3, 0.2, 0.0, 1.0, 1.0, "axial")
    g = _small_grid(10)
    with pytest.raises(ValueError):
        R.absorbed_fractions(tube, 1.0, g, np.array([0.0]), 0.02, +1, "ray_world")


# ------------------------------------------------- O4.3, O4.4, O4.5 ------

def test_mass_is_conserved_per_plane():
    tube = Tube((0.0, 0.0, 0.0), 0.25, math.radians(10.0), 0.3, 1.0, 0.4, "axial")
    # depth extent +-0.32 um, planes within 0.28 um: |delta| <= 0.6 um, sigma_r <= 0.47 um;
    # the tube's footprint is within +-0.45 um, so a +-3.6 um square keeps > 6 sigma of margin
    g = _small_grid(180, 0.04)
    zs = np.array([-0.28, 0.0, 0.28])
    res = R.render_transmittance(tube, 1.5, zs, g, RC)
    zeta = R.slab_grid(tube, g, RC.dzeta_um)
    dA = R.absorbed_fractions(tube, 1.5, g, zeta, RC.dzeta_um, +1, RC.absorption)
    mass = dA.sum()
    for k in range(len(zs)):
        assert abs((1.0 - res.tau[k]).sum() - mass) <= 1e-10 * mass


def test_fft_matches_direct():
    tube = Tube((0.0, 0.0, 0.0), 0.3, math.radians(50.0), 0.2, 1.0, 1.0, "axial")
    g = _small_grid(64, 0.0286)
    zs = np.array([-1.12, -0.28, 0.0, 0.84])
    d = R.render_transmittance(tube, 1.0, zs, g, RC, backend="direct").tau
    nos = dataclasses.replace(RC, fft_split_sigma_um=1e9)
    f0 = R.render_transmittance(tube, 1.0, zs, g, nos, backend="fft").tau
    res = R.render_transmittance(tube, 1.0, zs, g, RC, backend="fft")
    assert np.max(np.abs(f0 - d)) < 1e-10
    assert res.n_far_pairs > 0                       # the split is exercised
    assert np.max(np.abs(res.tau - d)) < 1e-6


def test_mu_zero_and_light_mirror():
    phi, theta = math.radians(25.0), 0.5
    t1 = Tube((0.0, 0.0, 0.0), 0.3, phi, theta, 1.0, 1.0, "axial")
    t2 = Tube((0.0, 0.0, 0.0), 0.3, phi, theta + math.pi, 1.0, 1.0, "axial")
    g = _small_grid(50)
    assert_array_equal(R.render_transmittance(t1, 0.0, [0.0], g, RC).tau, 1.0)
    X, Y = np.meshgrid(g.x0 + g.h * np.arange(g.nx), g.y0 + g.h * np.arange(g.ny))
    z_lo, z_hi, ins = H.ref_column(X, Y, np.zeros(3), 0.3, phi, theta, 1.0)
    span = float(z_hi[ins].max() - z_lo[ins].min())
    rc = dataclasses.replace(RC, dzeta_um=span / round(span / 0.02))   # the slabs tile the range exactly
    zs = np.array([-0.56, -0.28, 0.0, 0.28])
    a = R.render_transmittance(t1, 2.0, zs, g, rc, light_direction=+1).tau
    b = R.render_transmittance(t2, 2.0, -zs, g, rc, light_direction=-1).tau
    assert np.max(np.abs(a - b)) < 1e-12


def test_grid_must_resolve_the_narrowest_kernel():
    tube = Tube((0.0, 0.0, 0.0), 0.3, 0.0, 0.0, 1.0, 1.0, "axial")
    g = R.FineGrid(-1.0, -1.0, 0.05, 40, 40)         # h > sigma_r(0)/2 = 0.04 um
    with pytest.raises(ValueError):
        R.render_transmittance(tube, 1.0, [0.0], g, RC)


# ------------------------------------------------ O4.7 grid, integration --

@pytest.mark.parametrize("left,top,width,height,f,pad", [(100, 37, 5, 4, 8, 2), (0, 0, 3, 3, 4, 0), (-7, 12, 2, 6, 2, 1)])
def test_block_fine_grid_pixel_centres(left, top, width, height, f, pad):
    grid, (rs, cs) = R.block_fine_grid(left, top, width, height, PX, f, pad)
    assert abs(grid.h - PX / f) < 1e-15
    assert grid.nx == (width + 2 * pad) * f and grid.ny == (height + 2 * pad) * f
    xs = grid.x0 + grid.h * np.arange(grid.nx)
    ys = grid.y0 + grid.h * np.arange(grid.ny)
    bx = xs[cs].reshape(width, f).mean(axis=1)
    by = ys[rs].reshape(height, f).mean(axis=1)
    assert_allclose(bx, (left + np.arange(width)) * PX, rtol=0, atol=1e-12)
    assert_allclose(by, (top + np.arange(height)) * PX, rtol=0, atol=1e-12)
    # the samples of one pixel lie inside it
    assert np.all(np.abs(xs[cs].reshape(width, f) - bx[:, None]) < PX / 2)


def test_pixel_integrate_is_the_block_mean():
    rng = np.random.default_rng(3)
    fine = rng.normal(size=(2, 6 * 4, 5 * 4))
    out = R.pixel_integrate(fine, 4)
    exp = fine.reshape(2, 6, 4, 5, 4).mean(axis=(2, 4))
    assert_allclose(out, exp, rtol=0, atol=1e-14)


# ------------------------------------------------------- O4.8 camera ------

def test_camera_chain_statistics_rounding_clipping():
    rc = dataclasses.replace(RC, jpeg=False, noise_sd_gl=3.0, black_level_gl=5.0, gain=1.1)
    tau = np.ones((300, 300))
    img = CAM.camera_chain(tau, rc, np.random.default_rng(11))
    assert img.dtype == np.uint8
    mean_exp = 5.0 + 1.1 * rc.background_B_gl * 1.0
    # 9e4 samples: the mean's SE is 3/300 = 0.01 gl; the SD of an integer-rounded N(m, 3^2) is
    # sqrt(9 + 1/12) = 3.014 (rounding adds 1/12 gl^2), within 2 % of 3 as the spec allows
    assert abs(img.mean() - mean_exp) < 0.1
    assert abs(img.std() - 3.0) < 0.02 * 3.0 + 0.02
    # rounding to the nearest integer and clipping to [0, 255], no noise
    rc0 = dataclasses.replace(rc, noise_sd_gl=0.0, black_level_gl=0.0, gain=1.0)
    B = rc0.background_B_gl
    vals = np.array([[100.4 / B, 100.6 / B, 2.0, -1.0]])
    out = CAM.camera_chain(vals, rc0, np.random.default_rng(0))
    assert_array_equal(out, [[100, 101, 255, 0]])
    # the generator passed is the one used
    a = CAM.camera_chain(tau[:50, :50], rc, np.random.default_rng(5))
    b = CAM.camera_chain(tau[:50, :50], rc, np.random.default_rng(5))
    c = CAM.camera_chain(tau[:50, :50], rc, np.random.default_rng(6))
    assert_array_equal(a, b)
    assert np.any(a != c)


# ------------------------------------------------- O4.9 synthetic_block ---

def test_synthetic_block_contract_and_determinism():
    cfg = CFG
    tube = Tube((60 * PX, 40 * PX, 2 * cfg.acquisition.dz_um), 0.3, 0.0, 0.4, 1.0, 10.0, "axial")
    ks = np.array([0, 1, 2, 3, 4])
    out1 = R.synthetic_block(tube, 1.0, ks, 50, 30, 20, 20, cfg, np.random.default_rng(4), 4)
    out2 = R.synthetic_block(tube, 1.0, ks, 50, 30, 20, 20, cfg, np.random.default_rng(4), 4)
    block, ks_out, valid, frame = out1
    assert block.dtype == np.uint8 and block.shape == (5, 20, 20)
    assert_array_equal(np.asarray(ks_out), ks)
    assert np.asarray(valid).dtype == bool and np.all(valid)
    assert (frame.left, frame.top, frame.downsample, frame.res0_um_px) == (50, 30, 0, PX)
    assert_array_equal(block, out2[0])                    # bit-identical under one seed
    # plane k sits at z = k dz: under the linear absorption (no focus shift) and no
    # noise the planes k = 1 and k = 3 are mirror images about the tube at k = 2
    lin = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, absorption="linear",
                                                                 noise_sd_gl=0.0, jpeg=False))
    b = R.synthetic_block(tube, 0.5, ks, 50, 30, 20, 20, lin, np.random.default_rng(4), 4)[0].astype(int)
    dips = 255 - b.reshape(5, -1).min(axis=1)
    assert int(np.argmax(dips)) == 2
    assert np.max(np.abs(b[1] - b[3])) <= 1               # equal up to 8-bit rounding


# ------------------------------------------------- JPEG tables (Block 4) --

def _pillow_jpeg(tables, path):
    """A greyscale JPEG written by Pillow itself with the given tables."""
    import io
    from PIL import Image
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (16, 16), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", qtables=tables)
    data = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(data)
    return data


def test_jpeg_tables_read_save_load_collect(tmp_path):
    from PIL import Image
    from allen_diameter.loading import jpeg_tables as J
    t0 = [int(1 + (k * 7) % 50) for k in range(64)]
    a, b = str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")
    data = _pillow_jpeg([t0], a)
    _pillow_jpeg([t0], b)
    expect = [list(v) for _k, v in sorted(Image.open(a).quantization.items())]
    assert J.qtables_from_jpeg(data) == expect and J.qtables_from_jpeg(a) == expect
    p = str(tmp_path / "t.json")
    J.save_qtables(expect, p)
    assert [list(x) for x in J.load_qtables(p)] == expect
    other = str(tmp_path / "c.txt")
    open(other, "w").write("not a jpeg")
    sets = J.collect_qtables([a, b, other])
    assert [list(map(list, s[0])) if s[0] is not None else None for s in sets][0] == expect
    assert sets[0][1] == 2 and any(s[0] is None and s[1] == 1 for s in sets)   # most common first
    with pytest.raises(RuntimeError):
        J.check_pillow_table_order("8.2.0")
    J.check_pillow_table_order("8.3.0")
    # camera_chain with explicit tables equals the configuration's tables, bit for bit
    rc = dataclasses.replace(RC, jpeg=True, noise_sd_gl=0.0)
    tau = np.full((1, 16, 16), 0.8)
    x1 = CAM.camera_chain(tau, rc, np.random.default_rng(0), qtables=expect)
    x2 = CAM.camera_chain(tau, dataclasses.replace(rc, jpeg_qtables=tuple(tuple(t) for t in expect)),
                          np.random.default_rng(0))
    assert_array_equal(x1, x2)
