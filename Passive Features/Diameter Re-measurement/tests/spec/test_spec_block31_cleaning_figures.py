"""Spec-derived tests, Block 31 (the cleaning figures) -- tester 2026-11-13.

Oracles (SPEC.md Block 31): figure_node returns a Figure with 3 image axes
and 1 profile axis per plane shown; the three images of a row share
vmin/vmax; inputs unchanged; image extent in the pixel-centre convention of
section 2.1 / 3; figure_rounds one point per round; save writes a non-empty
PNG; ValueError for shapes that disagree and an empty `planes`.
Swappability (directive audit): the thresholds drawn follow the
configuration in use.
"""
from __future__ import annotations

import numpy as np
import pytest

import nc_helpers as H  # noqa: F401
import matplotlib
matplotlib.use("Agg")
from allen_diameter.plotting import cleaning_figures as CFG_
from allen_image_io import CropFrame

P = 0.1144


def arrays():
    rng = np.random.default_rng(31)
    blk = rng.integers(150, 220, size=(3, 20, 24)).astype(np.uint8)
    sh = rng.random((3, 20, 24)) * 30
    cl = blk + sh
    v = np.arange(-8, 9) * P
    pb, pa = rng.random((3, v.size)) * 200, rng.random((3, v.size)) * 200
    return blk, sh, cl, v, pb, pa


def test_figure_node_layout_scale_extent_and_inputs():
    blk, sh, cl, v, pb, pa = arrays()
    copies = [a.copy() for a in (blk, sh, cl, pb, pa)]
    fr = CropFrame(100, 200, 0, P)
    fig = CFG_.figure_node(blk, sh, cl, [4, 5, 6], fr, [[100 * P, 210 * P], [110 * P, 210 * P]], v, pb, pa,
                           {"node_id": 7, "class": "diffuse", "round": 1, "features": {"d": 0.5}}, planes=[6, 4])
    axes = fig.axes
    imgs = [ax for ax in axes if ax.images]
    prof = [ax for ax in axes if not ax.images]
    assert len(imgs) == 3 * 2 and len(prof) == 2
    for r in range(2):
        lims = {im.get_clim() for ax in imgs[3 * r:3 * r + 3] for im in ax.images}
        assert len(lims) == 1
    ext = imgs[0].images[0].get_extent()
    np.testing.assert_allclose(ext, [(100 - 0.5) * P, (124 - 0.5) * P, (220 - 0.5) * P, (200 - 0.5) * P], rtol=0, atol=1e-12)
    for a, c in zip((blk, sh, cl, pb, pa), copies):
        np.testing.assert_array_equal(a, c)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_figure_rounds_and_save(tmp_path):
    hist = [{"max_rel_d": 0.2, "max_line_change_gl": 5.0, "n_refined": 30},
            {"max_rel_d": 0.03, "max_line_change_gl": 1.0, "n_refined": 12},
            {"max_rel_d": 0.004, "max_line_change_gl": 0.2, "n_refined": 3}]
    from allen_diameter import config as _C   # adapted 2026-11-13: figure_rounds(history, cfg), Block 31 amendment 1 (D-136.4)
    run = _C.with_overrides(_C.default_config(), {"cleaning.stop_rel_d": 0.05, "cleaning.max_rounds": 7})
    fig = CFG_.figure_rounds(hist, run)
    ax0 = fig.axes[0]
    ys = [np.unique(np.asarray(ln.get_ydata(), dtype=float)) for ln in ax0.lines]
    assert any(y.size == 1 and abs(y[0] - 0.05) < 1e-12 for y in ys), "the run's stop_rel_d (0.05) is not drawn"
    xs = [np.unique(np.asarray(ln.get_xdata(), dtype=float)) for ln in ax0.lines]
    assert any(x.size == 1 and abs(x[0] - 7) < 1e-12 for x in xs), "the run's max_rounds (7) is not drawn"
    data = [ln for ax in fig.axes for ln in ax.lines if ln.get_linestyle() == "-"]   # the per-round series
    assert len(data) == 3 and all(len(ln.get_xdata()) == 3 for ln in data)
    p = CFG_.save(fig, str(tmp_path / "sub" / "r.png"), 60)
    with open(p, "rb") as f:
        head = f.read(8)
    assert head == b"\x89PNG\r\n\x1a\n"
    import os
    assert os.path.getsize(p) > 1000


def test_value_errors():
    blk, sh, cl, v, pb, pa = arrays()
    fr = CropFrame(0, 0, 0, P)
    with pytest.raises(ValueError):
        CFG_.figure_node(blk, sh[:, :-1], cl, [0, 1, 2], fr, [[0, 0], [1, 1]], v, pb, pa, {})
    with pytest.raises(ValueError):
        CFG_.figure_node(blk, sh, cl, [0, 1, 2], fr, [[0, 0], [1, 1]], v, pb, pa, {}, planes=[])


def test_rounds_threshold_follows_configuration():
    """Directive audit (swappability): a run with stop_rel_d = 0.05 should see 0.05 drawn.
    figure_rounds takes no configuration and always draws CleaningConfig() defaults."""
    import inspect
    sig = inspect.signature(CFG_.figure_rounds)
    assert any(p in sig.parameters for p in ("cfg", "config", "stop_rel_d")), \
        "figure_rounds(history) draws the default stop_rel_d / max_rounds, not the run's"
