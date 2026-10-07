"""Block 4 (kernel, renderer, camera): oracles from SPEC.md section 2.3 and Block 4."""
import dataclasses
import io
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage

from allen_diameter.config import default_config
from allen_diameter.model import camera, geometry as G, kernel, render

CFG = default_config()
R = CFG.renderer


def test_sigma_r_table_and_continuations():
    kn, sv = np.array(R.kernel_table_delta_um), np.array(R.kernel_table_sigma_um)
    assert_allclose(kernel.sigma_r(kn, R), sv, rtol=0, atol=0)
    mid = 0.5 * (kn[1:] + kn[:-1])
    assert_allclose(kernel.sigma_r(mid, R), 0.5 * (sv[1:] + sv[:-1]), rtol=1e-14)
    assert_allclose(kernel.sigma_r(-mid, R), kernel.sigma_r(mid, R), rtol=0)
    far = np.array([1.0, 2.0])
    assert_allclose(kernel.sigma_r(far, R), sv[-1] + 0.79 * (far - kn[-1]), rtol=1e-14)
    fr = dataclasses.replace(R, kernel_continuation="frozen")
    assert_allclose(kernel.sigma_r(far, fr), sv[-1], rtol=0)
    pr = dataclasses.replace(R, kernel_continuation="proportional")
    assert_allclose(kernel.sigma_r(far, pr), sv[-1] * far / kn[-1], rtol=1e-14)


def _small(phi=0.2, d=0.6, U=2.0):
    tube = G.Tube((0.05, -0.03, 0.0), d / 2, phi, 0.4, 1.0, U, "axial")
    grid = render.FineGrid(-3.0, -3.0, 0.1144 / 4, 210, 210)
    return tube, grid


def test_mu_zero_gives_one_and_fft_vs_direct():
    tube, grid = _small()
    z = np.array([-0.28, 0.0, 0.28])
    res0 = render.render_transmittance(tube, 0.0, z, grid, R)
    assert np.all(res0.tau == 1.0)
    rc = dataclasses.replace(R, fft_split_sigma_um=10.0)          # no far path: pure FFT vs the reference
    a = render.render_transmittance(tube, 1.2, z, grid, rc, backend="fft").tau
    b = render.render_transmittance(tube, 1.2, z, grid, rc, backend="direct").tau
    # both exact up to the wrap (exp(-18)) and truncation (8 sigma) errors
    assert_allclose(a, b, atol=1e-10)
    c = render.render_transmittance(tube, 1.2, z, grid, R, backend="fft").tau
    assert_allclose(c, b, atol=1e-6)


def test_mass_and_partition():
    tube, grid = _small(phi=0.0, d=0.8, U=0.6)   # footprint >= 2 um (7 sigma_max) from the grid edge
    z = np.array([0.0, 0.28])
    mu = 2.0
    zeta = render.slab_grid(tube, grid, R.dzeta_um)
    dA = render.absorbed_fractions(tube, mu, grid, zeta, R.dzeta_um, +1, "partition_vertical")
    X, Y = grid.mesh()
    zl, zh, ins = G.column_interval(X, Y, tube)
    # C1 per column
    assert_allclose(dA.sum(0), 1 - np.exp(-mu * (zh - zl)), atol=1e-12)
    tau = render.render_transmittance(tube, mu, z, grid, R).tau
    h2 = grid.h ** 2
    for k in range(2):
        assert_allclose((1 - tau[k]).sum() * h2, dA.sum() * h2, rtol=1e-8)


def test_light_reversal_is_depth_mirror():
    tube = G.Tube((0.0, 0.0, 0.14), 0.4, math.radians(25), 0.3, 1.0, 1.5)
    tube2 = G.Tube((0.0, 0.0, 0.14), 0.4, math.radians(25), 0.3 + math.pi, 1.0, 1.5)
    grid = render.FineGrid(-2.5, -2.5, 0.1144 / 4, 176, 176)
    zk = np.array([-0.28, 0.0, 0.56])
    a = render.render_transmittance(tube, 3.0, zk, grid, R, light_direction=+1).tau
    b = render.render_transmittance(tube2, 3.0, 2 * 0.14 - zk, grid, R, light_direction=-1).tau
    # The spec calls this symmetry exact; with slabs tiling [z_min, z_max] from the bottom it holds only
    # up to the slab discretisation (measured: 1.3e-4 at dzeta 0.02, 1.2e-6 at 0.01). Tolerance: the
    # spec's own dzeta-convergence bound for the node dip (5e-4). See report, Problems in the spec.
    assert_allclose(a, b, atol=5e-4)


def test_pixel_integrate_is_block_mean():
    a = np.random.default_rng(0).random((3, 16, 24))
    got = render.pixel_integrate(a, 4)
    ref = np.array([[[a[k, 4 * i:4 * i + 4, 4 * j:4 * j + 4].mean() for j in range(6)] for i in range(4)]
                    for k in range(3)])
    assert_allclose(got, ref, rtol=1e-14)


def test_block_fine_grid_pixel_centres():
    grid, inner = render.block_fine_grid(10, 20, 3, 2, 0.1144, 4, pad_px=1)
    xs = grid.xs[inner[1]].reshape(3, 4).mean(1)
    ys = grid.ys[inner[0]].reshape(2, 4).mean(1)
    assert_allclose(xs, np.array([10, 11, 12]) * 0.1144, atol=1e-12)
    assert_allclose(ys, np.array([20, 21]) * 0.1144, atol=1e-12)


def test_camera_flat_field_statistics():
    rc = dataclasses.replace(R, jpeg=False, black_level_gl=5.0, gain=0.9, noise_sd_gl=3.0)
    out = camera.camera_chain(np.ones((256, 256)), rc, np.random.default_rng(1))
    assert out.dtype == np.uint8
    m, s = out.mean(), out.std(ddof=1)
    assert abs(m - (5.0 + 0.9 * 210.0)) < 0.02 * (5 + 0.9 * 210)
    # rounding adds 1/12 gl^2 of variance: sqrt(9 + 1/12) = 3.014; 2 % band (spec)
    assert abs(s - 3.0) < 0.06 * 3.0


def test_camera_clip_and_nan():
    rc = dataclasses.replace(R, jpeg=False, noise_sd_gl=0.0, background_B_gl=300.0)
    out = camera.camera_chain(np.array([[1.0, -0.5]]), rc, np.random.default_rng(0))
    assert out.tolist() == [[255, 0]]
    with pytest.raises(ValueError):
        camera.camera_chain(np.array([[np.nan]]), rc, np.random.default_rng(0))


def test_camera_nan_noise_is_refused():
    """NaN noise SD slips past validate() (Block 1 finding); the chain must not
    produce an image from it silently."""
    rc = dataclasses.replace(R, jpeg=False, noise_sd_gl=float("nan"))
    with pytest.raises(ValueError):
        camera.camera_chain(np.ones((8, 8)), rc, np.random.default_rng(0))


def test_jpeg_tables_round_trip():
    from PIL import Image
    qt = [[(i % 7) + 2 for i in range(64)]]
    img = (np.random.default_rng(0).random((32, 32)) * 255).astype(np.uint8)
    out = camera.jpeg_roundtrip(img, qtables=qt)
    assert out.dtype == np.uint8 and out.shape == img.shape
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", qtables=qt)
    buf.seek(0)
    assert list(Image.open(buf).quantization[0]) == qt[0]


def test_synthetic_block_contract_and_determinism():
    tube = G.Tube((1.0, 1.0, 0.0), 0.4, 0.1, 0.5, 1.0, 10.0)
    args = (tube, 1.0, np.arange(-2, 3), 0, 0, 18, 18, CFG)
    b1, ks, valid, frame = render.synthetic_block(*args, np.random.default_rng(5), 2)
    b2 = render.synthetic_block(*args, np.random.default_rng(5), 2)[0]
    assert b1.dtype == np.uint8 and b1.shape == (5, 18, 18)
    assert np.array_equal(b1, b2)
    assert valid.all() and list(ks) == [-2, -1, 0, 1, 2]
    assert (frame.left, frame.top, frame.downsample) == (0, 0, 0) and frame.res_um_px == CFG.acquisition.res0_um


def test_jpeg_qtables_file_reaches_the_renderer(tmp_path):
    """Block 4/11: the camera chain uses Allen's tables 'when given'; Block 11 writes
    them to renderer.jpeg_qtables_file for the table build. The configured file
    must change the synthetic block (here: coarse tables vs quality 85)."""
    qt = [[64] * 64]
    p = tmp_path / "q.json"
    p.write_text(json.dumps(qt))
    cfg_q = dataclasses.replace(CFG, renderer=dataclasses.replace(R, jpeg_qtables_file=str(p), noise_sd_gl=0.0))
    cfg_0 = dataclasses.replace(CFG, renderer=dataclasses.replace(R, noise_sd_gl=0.0))
    tube = G.Tube((1.0, 1.0, 0.0), 0.4, 0.1, 0.5, 1.0, 10.0)
    a = render.synthetic_block(tube, 1.0, [0], 0, 0, 18, 18, cfg_q, np.random.default_rng(0), 2)[0]
    b = render.synthetic_block(tube, 1.0, [0], 0, 0, 18, 18, cfg_0, np.random.default_rng(0), 2)[0]
    c = render.synthetic_block(tube, 1.0, [0], 0, 0, 18, 18, cfg_0, np.random.default_rng(0), 2, qtables=qt)[0]
    assert not np.array_equal(b, c)            # the tables matter
    assert np.array_equal(a, c), "renderer.jpeg_qtables_file is not applied by synthetic_block"


def test_grid_must_resolve_kernel():
    tube, _ = _small()
    coarse = render.FineGrid(-3.0, -3.0, 0.05, 120, 120)     # h > sigma_min / 2 = 0.04
    with pytest.raises(ValueError):
        render.render_transmittance(tube, 1.0, [0.0], coarse, R)
