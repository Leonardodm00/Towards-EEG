"""Smoke test for Cell 7's isolation figures -- Block 10.F (5) in
specs/SPEC.md (D-098.4, 2026-10-28): for every candidate the calibration
evaluates, the patch its background is fitted on, with the traced
neighbours whose predicted shadows decide not_isolated.

The figure computes nothing new: the view must hold exactly what the
selection used -- the patch of 10.P at the pilot's sharpest plane, its far
pixels and planar background (patch_planes), and every neighbour's shadow
(isolation_breakdown, whose deepest is isolation_depth's).

Checks
    test_known_answer   a branch along x and a parallel neighbour 3 um away
                        on a planar ramp image: the view's deepest shadow and
                        worst neighbour are isolation_depth's, the offenders
                        are the neighbour's segments, not_isolated in
                        calibration_reasons; the background is the ramp
                        (b0 at the centre and both slopes, to 1e-9)
    test_reference      the far mask and background at k* equal the
                        scan's (patch_planes over the whole stage: the
                        soma's blur is the stage's largest offset);
                        the image is the provider's plane k*; one provider
                        call of one plane; isolation_breakdown's deepest
                        equals a verbatim copy of isolation_depth before
                        2026-10-28 on random scenes (value and worst id)
    test_convergence    not a discretisation: skipped
    test_invariants     the neighbour moved 3 patch widths away: no
                        offender, isolated; the node's own branch never an
                        offender; the depth rises as the neighbour comes
                        closer
    test_contract       the view's keys and shapes; the figure has two
                        panels and writes a PNG; the overview pages hold 36
                        thumbnails each; under the stripe (D-101) the view
                        carries the stripe of k*, drawn by the figure and the
                        overview; Cell 7's hook on the synthetic cell
                        of fixtures_cell.py through scan_cell: one PNG per
                        measured candidate, the soma or the segment joining
                        it to the dendrite the worst neighbour at the
                        default threshold, none at a threshold of 50 gl,
                        the verdict carried and consistent with the reasons
    test_determinism    two views of the same node are identical
    test_edge_cases     a pilot without a centre: no view; without a sharpest
                        plane: the SWC node's plane, isolation not evaluated;
                        a NaN B_bar: not evaluated; the hook skips a node
                        without a result and reports a failing figure
                        without raising

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_isolation_view.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
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

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (SRC, HERE, WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import calibration_report as CR  # noqa: E402
from allen_diameter.analysis import calibration_scene as CS  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch, NodeResult  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model.kernel import sigma_r  # noqa: E402
from allen_diameter.plotting import calibration_figures as CF  # noqa: E402
from allen_image_io import CropFrame  # noqa: E402

SEED = 20261028
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
NAN = float("nan")
HALF = 8.0                       # um: the patch of these checks (the default 25 um is only slower)
RAMP = (180.0, 0.4, -0.25)       # gl, gl/um, gl/um: the provider's plane b0 + bx (x - cx) + by (y - cy)


# --------------------------------------------------------------- helpers ---

def cfg_with(**cal):
    """The square rule (D-078.1, D-081.1) unless the check says otherwise: these checks are about the figure's
    contents under it; the stripe (D-101, the default since 2026-10-28) has test_smoke_stripe.py and one check in
    test_contract below."""
    cfg = default_config()
    cal = dict({"background_support": "square"}, **cal)
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def pilot(cx=0.0, cy=0.0, z=0.0, theta=0.0, d=0.4, mu=1.0, B_bar=200.0, z_sub=0.0, k_star=0):
    """A pilot NodeResult with a centre, a heading, a sharpest plane and finite B_bar and mu."""
    return NodeResult(12, 3, cx, cy, z, 0.0, "", NAN, NAN, k_star, z_sub, cx, cy, z, theta, 0.0, False, False,
                      B_bar, "block_masked", d, mu, 0.0, mu * d, "converged", (),
                      None, "gradient_energy", np.empty(0, dtype=int), np.empty(0), np.empty(0), -1)


def scene_of(y_other=3.0, r_other=0.3, z_other=0.0, with_soma=False):
    """(own Branch along x, ids 100..124, r 0.2 um; the Scene with a parallel neighbour at y = y_other, ids 200..204)."""
    x = np.linspace(-12, 12, 25)
    own = Branch.from_points(np.c_[x, np.zeros(25), np.zeros(25)], 0.2, ids=np.arange(100, 125))
    other = Branch.from_points(np.c_[np.linspace(-8, 8, 5), np.full(5, y_other), np.full(5, z_other)], r_other,
                               ids=np.arange(200, 205))
    s1, s2 = Scene.from_branch(own), Scene.from_branch(other)
    soma = np.array([[0.0, -9.0, 2.0]]) if with_soma else np.empty((0, 3))
    return own, Scene(np.vstack([s1.seg_xyz, s2.seg_xyz]), np.vstack([s1.seg_radius, s2.seg_radius]),
                      np.vstack([s1.seg_ids, s2.seg_ids]), soma)


def ramp_provider(cfg, cx=0.0, cy=0.0):
    """The fetch_zblock contract over a planar ramp (every plane the same); the calls are kept."""
    p = cfg.acquisition.res0_um
    calls = []

    def provider(left, top, width, height, k_lo, k_hi):
        calls.append((int(left), int(top), int(width), int(height), int(k_lo), int(k_hi)))
        ks = np.arange(int(k_lo), int(k_hi) + 1)
        cols, rows = np.meshgrid(np.arange(int(width)), np.arange(int(height)))
        x, y = (left + cols) * p, (top + rows) * p
        plane = RAMP[0] + RAMP[1] * (x - cx) + RAMP[2] * (y - cy)
        return np.repeat(plane[None], ks.size, axis=0), ks, np.ones(ks.size, dtype=bool), \
            CropFrame(int(left), int(top), 0, float(p))
    return provider, calls


def old_isolation_depth(branch, index, scene, cx, cy, half_um, planes_z, B_bar, mu, cfg):
    """isolation_depth as it stood before 2026-10-28 (verbatim but for the names): the reference."""
    rc = cfg.renderer
    z = np.asarray(planes_z, dtype=float).reshape(-1)
    lo = np.array([cx - half_um, cy - half_um])
    hi = np.array([cx + half_um, cy + half_um])
    in_branch = {int(x) for x in np.asarray(branch.ids).ravel()}
    worst, worst_id = 0.0, None
    for q in range(scene.seg_xyz.shape[0]):
        c_id, p_id = int(scene.seg_ids[q, 0]), int(scene.seg_ids[q, 1])
        if c_id in in_branch and p_id in in_branch:
            continue
        a, b = scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:]
        rho = CS._rect_distance(a[:2], b[:2], lo, hi)
        r = float(max(scene.seg_radius[q, 0], scene.seg_radius[q, 1]))
        depth = 0.0
        for z_end in (a[2], b[2]):
            for s in np.atleast_1d(sigma_r(np.abs(z_end - z), rc)):
                if rho > r + 6.0 * s:
                    continue
                depth = max(depth, CS.shadow_depth(rho, r, mu, s, B_bar))
        if depth > worst:
            worst, worst_id = depth, c_id
    span = 0.5 * (z.max() - z.min()) if z.size > 1 else 0.0
    sig_soma = float(sigma_r(span, rc))
    for x, y, r in np.atleast_2d(np.asarray(scene.soma, dtype=float)).reshape(-1, 3):
        rho = float(np.hypot(*(np.clip([x, y], lo, hi) - np.array([x, y]))))
        depth = CS.shadow_depth(rho, float(r), mu, sig_soma, B_bar)
        if depth > worst:
            worst, worst_id = depth, "soma"
    return float(worst), worst_id


def view_of(cfg, y_other=3.0, node=None, with_soma=False, verdict="probe"):
    br, scene = scene_of(y_other, with_soma=with_soma)
    prov, calls = ramp_provider(cfg)
    node = node or pilot()
    return CR.isolation_view(br, 12, node, scene, prov, cfg, verdict=verdict), br, scene, node, calls


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = cfg_with(patch_half_um=HALF)
    view, br, scene, node, _ = view_of(cfg)
    planes_z = CAL.stage_planes(node, cfg, "near") * cfg.acquisition.dz_um
    depth, who = CS.isolation_depth(br, 12, scene, 0.0, 0.0, HALF, planes_z, 200.0, 1.0, cfg)
    assert view["evaluated"] and abs(view["max_depth_gl"] - depth) == 0.0 and view["worst"]["child_id"] == who, \
        (view["max_depth_gl"], depth, view["worst"], who)
    thr = cfg.calibration.isolation_depth_gl
    assert depth > thr and view["offenders"] and all(o["child_id"] >= 200 for o in view["offenders"]), view["offenders"]
    assert [o["depth_gl"] for o in view["offenders"]] == sorted((o["depth_gl"] for o in view["offenders"]), reverse=True)
    assert "not_isolated" in CAL.calibration_reasons(node, cfg, 0.4, br, 12, scene, planes_z)
    # each drawn neighbour segment carries its own breakdown depth (the rows map to the scene's segments)
    rows = np.atleast_2d(view["seg_rows"])
    nb = rows[:, 8] < 0.5
    assert nb.sum() == 4 and np.all(np.isfinite(view["seg_depth_gl"][nb])), view["seg_depth_gl"]
    assert np.allclose(np.sort(view["seg_depth_gl"][nb]),
                       np.sort([o["depth_gl"] for o in view["offenders"] if o["kind"] == "segment"]), rtol=0, atol=0)
    assert np.all(rows[nb][:, 1] == 3.0) and np.all(rows[~nb][:, 1] == 0.0)
    # the background drawn is the plane fitted through the far pixels: the ramp itself
    b = view["background"]
    assert view["background_ok"] and np.allclose(b, RAMP, atol=1e-9), (b, RAMP)
    assert view["set"] == "near" and view["k_view"] == 0 and view["plane_rule"] == "k_star"
    assert view["verdict"] == "probe" and view["threshold_gl"] == thr and view["offsets_planes"] == 3
    # a neighbour 2 um deep is sharpest at the stage's deepest plane, not at k*: the view's depth is the stage's
    br2, sc2 = scene_of(y_other=3.0, z_other=2.0)
    prov, _ = ramp_provider(cfg)
    v2 = CR.isolation_view(br2, 12, node, sc2, prov, cfg)
    d2 = CS.isolation_depth(br2, 12, sc2, 0.0, 0.0, HALF, planes_z, 200.0, 1.0, cfg)[0]
    d2_star = CS.isolation_depth(br2, 12, sc2, 0.0, 0.0, HALF, np.array([0.0]), 200.0, 1.0, cfg)[0]
    assert v2["max_depth_gl"] == d2 and d2 > d2_star, (v2["max_depth_gl"], d2, d2_star)


def test_reference():
    cfg = cfg_with(patch_half_um=HALF)
    view, br, scene, node, calls = view_of(cfg, with_soma=True)
    assert len(calls) == 1 and calls[0][4] == calls[0][5] == 0, calls           # one call, one plane: k*
    kk = CAL.stage_planes(node, cfg, "near")
    prov, _ = ramp_provider(cfg)
    full = CS.fetch_patch(prov, 0.0, 0.0, HALF, kk[0], kk[-1], cfg.acquisition.res0_um)
    _idx, far, _b, _ok, _fr, _st = CAL.patch_planes(full, kk, node, br, scene, cfg)
    q = int(np.flatnonzero(kk == 0)[0])
    assert np.array_equal(view["far"], far[q]) and 0.05 < view["far"].mean() < 1.0, view["far"].mean()
    assert np.array_equal(view["image"], full[0][q])
    # isolation_breakdown's deepest against the function as it was, on random scenes
    rng = np.random.default_rng(SEED)
    for trial in range(25):
        y_o, z_o, r_o = rng.uniform(-14, 14), rng.uniform(-3, 3), rng.uniform(0.1, 0.6)
        b2, sc = scene_of(y_o, r_o, z_o, with_soma=bool(trial % 2))
        z = np.arange(-int(rng.integers(1, 17)), int(rng.integers(1, 17)) + 1) * cfg.acquisition.dz_um
        cx, cy, B, mu = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(150, 230), rng.uniform(0.3, 3.0)
        new = CS.isolation_depth(b2, 12, sc, cx, cy, HALF, z, B, mu, cfg)
        old = old_isolation_depth(b2, 12, sc, cx, cy, HALF, z, B, mu, cfg)
        assert new == old, (trial, new, old)
        parts = CS.isolation_breakdown(b2, 12, sc, cx, cy, HALF, z, B, mu, cfg)
        assert len(parts) == 4 + sc.soma.shape[0] and max([0.0] + [p["depth_gl"] for p in parts]) == old[0]


def test_convergence():
    raise unittest.SkipTest("the figure's inputs are read, not computed by a discretisation")


def test_invariants():
    cfg = cfg_with(patch_half_um=HALF)
    far_view = view_of(cfg, y_other=3 * 2 * HALF)[0]
    assert far_view["offenders"] == [] and far_view["max_depth_gl"] <= far_view["threshold_gl"], far_view["max_depth_gl"]
    assert far_view["worst"] is None or far_view["worst"]["child_id"] >= 200
    depths = [view_of(cfg, y_other=y)[0]["max_depth_gl"] for y in (HALF + 3.0, HALF + 1.5, HALF + 0.5)]
    assert depths[0] < depths[1] < depths[2], depths
    for v in (far_view, view_of(cfg)[0]):
        own = np.atleast_2d(v["seg_rows"])[:, 8] >= 0.5
        assert own.any() and np.all(np.isnan(v["seg_depth_gl"][own])), "the own branch carries no shadow depth"


def test_contract():
    cfg = cfg_with(patch_half_um=HALF)
    view = view_of(cfg, with_soma=True)[0]
    keys = {"node_id", "set", "allen_diameter_um", "k_view", "plane_rule", "offsets_planes", "image", "image_valid",
            "left", "top", "res_um_px", "far", "background", "background_ok", "cx_um", "cy_um", "theta_rad", "half_um",
            "line_half_um", "seg_rows", "seg_depth_gl", "soma", "soma_depth_gl", "threshold_gl", "evaluated",
            "max_depth_gl", "worst", "offenders", "n_neighbours", "verdict", "support", "stripe"}
    assert keys == set(view), sorted(set(view) ^ keys)
    assert view["support"] == "square" and view["stripe"] is None
    H, W = view["image"].shape
    assert view["far"].shape == (H, W) and view["far"].dtype == bool and view["seg_rows"].shape[1] == 9
    assert view["seg_depth_gl"].shape == (view["seg_rows"].shape[0],) and view["soma_depth_gl"].shape == (1,)
    assert abs(W * view["res_um_px"] - 2 * HALF) <= 2 * view["res_um_px"], (W, view["res_um_px"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory() as tmp:
        fig = CF.isolation_figure(view)
        assert len(fig.axes) == 2
        png = os.path.join(tmp, "v.png")
        fig.savefig(png, dpi=80)
        plt.close(fig)
        assert os.path.getsize(png) > 10000
        pages = CF.isolation_overview([view] * 40)
        assert len(pages) == 2 and len([a for a in pages[0].axes if a.images]) == 36
        assert len([a for a in pages[1].axes if a.images]) == 4
        for f in pages:
            plt.close(f)
        # under the stripe (D-101, the default): the view carries the stripe of plane k*, its mask is the support
        cfg_s = cfg_with(patch_half_um=HALF, background_support="stripe")
        view_s = view_of(cfg_s)[0]
        st = view_s["stripe"]
        assert view_s["support"] == "stripe" and isinstance(st, dict) and st["n_pixels"] == int(view_s["far"].sum())
        assert "stripe at k*" in CF._isolation_line(view_s)
        fig = CF.isolation_figure(view_s)
        assert "stripe" in fig.axes[1].get_title(loc="left")
        plt.close(fig)
        page = CF.isolation_overview([view_s])[0]
        assert "stripe" in page._suptitle.get_text()
        plt.close(page)
    # Cell 7's hook through scan_cell on the synthetic cell (fixtures_cell): one figure per measured candidate
    import calibrate_kernel as CK
    import run_cell
    from fixtures_cell import synthetic_cell
    for thr, want_offender in ((cfg.calibration.isolation_depth_gl, True), (50.0, False)):
        cfg_cell = dataclasses.replace(cfg_with(patch_half_um=3.5, isolation_depth_gl=thr),
                                       measure=dataclasses.replace(cfg.measure, block_half_um=3.5))
        with tempfile.TemporaryDirectory() as tmp:
            swc, fetcher, planes, _ = synthetic_cell(tmp, cfg_cell, d_true=0.2, mu=1.5, allen_radius=0.1)
            provider = run_cell.real_provider(fetcher, planes, cfg_cell.acquisition.res0_um)
            views, logs = [], []
            fig_dir = os.path.join(tmp, "isolation")
            hook = CK.isolation_hook(provider, cfg_cell, fig_dir, views, log=logs.append)
            out = CAL.scan_cell(swc, provider, cfg_cell, candidates={4, 5, 6}, on_node=hook)
            made = sorted(os.listdir(fig_dir))
            measured = [nid for nid, res, _, _ in out if res is not None]
            assert made == ["near_node_%d.png" % n for n in sorted(measured)] and len(views) == len(measured) == 3, \
                (made, measured, logs)
            assert not any("failed" in m for m in logs), logs
            for v, (nid, res, scan, reasons) in zip(views, out):
                assert v["node_id"] == nid and v["verdict"] == CAL.scan_verdict(scan, reasons) and "far" not in v
                bad = v["evaluated"] and v["max_depth_gl"] > v["threshold_gl"]
                assert bad == ("not_isolated" in reasons), (nid, v["max_depth_gl"], reasons)
            if want_offender:
                # the cell's soma (r 4 um at the origin) and the segment joining it to the dendrite's first node --
                # not of the node's stretch, so a neighbour -- shade every candidate's patch
                assert all(v["worst"]["kind"] == "soma" or 1 in (v["worst"]["child_id"], v["worst"]["parent_id"])
                           for v in views), [v["worst"] for v in views]
            pages = CK.write_isolation_overview(views, fig_dir)
            assert [os.path.basename(x) for x in pages] == ["isolation_overview_1.png"]
            if want_offender:
                _real_command_line(tmp, cfg_cell, swc, fetcher, planes)


def _real_command_line(tmp, cfg_cell, swc, fetcher, planes):
    """`calibrate_kernel.py real` in process on the synthetic cell, Allen's server replaced (the SWC by --swc, the
    image table and the fetcher by the fixture's): the figures land in figures/isolation beside --out, with the
    overview and the summary line; --no-isolation-figures draws none."""
    import contextlib
    import io
    import json
    import allen_image_io as aio
    import calibrate_kernel as CK
    from allen_diameter.loading import table_io
    csv_path = os.path.join(tmp, "nodes.csv")
    table_io.write_rows([dict(node_id=n, fit_status="converged", flags="", phi_rad=0.0, alpha_hat=0.3,
                              allen_radius_um=0.1) for n in (4, 5, 6)], csv_path)
    cj = os.path.join(tmp, "config.json")
    with open(cj, "w") as f:
        json.dump(dict(json.loads(cfg_cell.to_json()), calibration=dataclasses.asdict(cfg_cell.calibration)), f)
    swc_path = os.path.join(tmp, "cell.swc")
    saved = (aio.list_images, aio.plane_table, aio.HttpFetcher)
    aio.list_images, aio.plane_table = (lambda sid: None), (lambda images: planes)
    aio.HttpFetcher = lambda cache_dir=None: fetcher
    try:
        for extra, want in (([], True), (["--no-isolation-figures"], False)):
            out_dir = os.path.join(tmp, "cal_%s" % ("on" if want else "off"))
            os.makedirs(out_dir)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                CK.main(["real", "--specimen", "1", "--nodes-csv", csv_path, "--swc", swc_path, "--cache-dir", tmp,
                         "--out", os.path.join(out_dir, "scans.json"), "--config-json", cj] + extra)
            text = buf.getvalue()
            iso = os.path.join(out_dir, "figures", "isolation")
            if want:
                made = sorted(os.listdir(iso))
                assert made == ["isolation_overview_1.png", "near_node_4.png", "near_node_5.png", "near_node_6.png"], made
                assert "isolation figures: 3 candidates (3 with a neighbour's predicted shadow above 0.03 gl)" in text, text
            else:
                assert not os.path.exists(iso) and "isolation figures" not in text, text
            assert "3 candidate nodes" in text and os.path.exists(os.path.join(out_dir, "scans.json")), text
    finally:
        aio.list_images, aio.plane_table, aio.HttpFetcher = saved


def test_determinism():
    cfg = cfg_with(patch_half_um=HALF)
    a, b = view_of(cfg, with_soma=True)[0], view_of(cfg, with_soma=True)[0]
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k
        else:
            assert a[k] == b[k], k


def test_edge_cases():
    cfg = cfg_with(patch_half_um=HALF)
    assert view_of(cfg, node=pilot(cx=NAN))[0] is None
    v = view_of(cfg, node=pilot(z_sub=NAN, k_star=-1))[0]
    assert v["plane_rule"] == "swc" and v["k_view"] == 0 and not v["evaluated"] and v["offenders"] == []
    assert np.all(np.isnan(v["seg_depth_gl"])) and v["max_depth_gl"] == 0.0
    v = view_of(cfg, node=pilot(B_bar=NAN))[0]
    assert v["plane_rule"] == "k_star" and not v["evaluated"] and v["worst"] is None
    import calibrate_kernel as CK
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory() as tmp:
        fig = CF.isolation_figure(v)                        # not evaluated: drawn, dashed neighbours
        plt.close(fig)
        logs, views = [], []
        br, scene = scene_of()

        def broken(*args):
            raise RuntimeError("no such plane")
        hook = CK.isolation_hook(broken, cfg, os.path.join(tmp, "iso"), views, log=logs.append)
        hook(12, br, 12, None, None, ["status:failed"], scene)          # no result: skipped silently
        hook(12, br, 12, pilot(), None, ["not_isolated"], scene)         # a failing figure: reported, not raised
        assert views == [] and len(logs) == 1 and "isolation figure failed" in logs[0] and "no such plane" in logs[0]
        a = type("A", (), dict(no_isolation_figures=True, isolation_figures="", out=os.path.join(tmp, "s.json")))()
        assert CK.isolation_dir(a) == ""
        a.no_isolation_figures = False
        assert CK.isolation_dir(a) == os.path.join(tmp, "figures", "isolation")


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
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
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
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
