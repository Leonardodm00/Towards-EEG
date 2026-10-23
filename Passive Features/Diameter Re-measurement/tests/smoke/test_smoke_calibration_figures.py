"""Smoke test for the calibration's figure inputs and figures -- Block 10.F
in specs/SPEC.md (D-075, D-080.2, D-081.5; 2026-10-23).

Checks
    test_known_answer   stage_inputs on exact near data (omega = c_j +
                        G_true(delta), G_true in the fit's own basis) centres
                        by c_j, finds the median by offset
                        over the planted values and leaves residuals below
                        1e-9; on exact far data (omega = c_j + (sigma_0 +
                        gamma x)^2) likewise; growth_curve equals the family
                        curve on a grid; kernel_law_inputs recovers the
                        planted sigma_r at every used plane
    test_reference      the figures against matplotlib itself: every PNG
                        write_all names exists, is non-empty and opens as an
                        image of the expected size class (Pillow)
    test_convergence    not a discretisation: skipped
    test_invariants     a constant added to every omega leaves the centred
                        values and residuals unchanged; the example node is
                        the one planted on the median
    test_contract       figure_inputs keys and shapes; write_all with no
                        calibration, no far fit and no patches still writes
                        the width figures; gaussian_inputs is None for a scan
                        without profiles
    test_determinism    two runs of write_all give byte-identical PNGs
    test_edge_cases     an empty scan list -> no stage, nothing written; a
                        stage with one node

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_calibration_figures.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
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
for _p in (SRC, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import calibration_report as CR  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import calibration_io as CIO  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.plotting import calibration_figures as CF  # noqa: E402

SEED = 20261023
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib", "Pillow")
FW = math.sqrt(8.0 * math.log(2.0))


# --------------------------------------------------------------- helpers ---

def truth_growth(cfg):
    """The planted growth G_true(delta) (um^2): the renderer table's sigma^2 -
    sigma_r(0)^2 at the fit's own knots (m h, m = 0..M, M as fit_growth sets
    it for the near scan), interpolated with the fit's own basis, so that the
    near fit can represent it exactly. Returns (knots, G at the knots, curve)."""
    cal, dz, h = cfg.calibration, cfg.acquisition.dz_um, float(cfg.calibration.knot_step_um)
    M = max(1, int(math.ceil((cal.offsets_planes + 0.5) * dz / h - 1e-9)))
    knots = np.arange(M + 1) * h
    g = K.sigma_r(knots, cfg.renderer) ** 2 - cfg.renderer.sigma_r0_um ** 2

    def curve(delta):
        B, _ = CAL._basis(np.asarray(delta, dtype=float).ravel(), cal.origin_convention, h, M, cal.growth_interp)
        return (B @ g).reshape(np.shape(delta))
    return knots, g, curve


def make_scans(cfg, which, n_nodes, rng, gamma=0.79, noise=0.0):
    """Synthetic PlaneScans with profiles: near omega = c_j + G_true(delta) (+ noise), G_true from truth_growth;
    far omega = c_j + (sigma_0 + gamma x)^2 beyond the anchor, c_j + sigma_r(delta)^2 inside (those planes are
    not fitted)."""
    dz, d0 = cfg.acquisition.dz_um, cfg.calibration.table_max_delta_um
    P = cfg.calibration.offsets_planes if which == "near" else cfg.calibration.far_offsets_planes
    v = np.arange(-26, 27) * cfg.measure.profile_step_um if which == "near" else np.arange(-87, 88) * cfg.measure.profile_step_um
    s0 = float(K.sigma_r(d0, cfg.renderer))
    _, _, g_true = truth_growth(cfg)
    scans, truth = [], []
    for j in range(n_nodes):
        z_ax = rng.uniform(-0.5 * dz, 0.5 * dz)
        c = rng.uniform(0.01, 0.03) if which == "near" else rng.uniform(-0.05, 0.02)
        ks = np.arange(-P, P + 1)
        z = ks * dz
        delta = z_ax - z
        if which == "near":
            om = c + g_true(delta)
        else:
            x = np.abs(delta) - d0
            om = np.where(x >= 0, c + (s0 + gamma * x) ** 2, c + K.sigma_r(delta, cfg.renderer) ** 2)
        om = om + (rng.normal(0.0, noise, ks.size) if noise > 0 else 0.0)
        sig = np.sqrt(np.maximum(om, 1e-4))
        amp = 35.0 / (math.sqrt(2.0 * math.pi) * sig)
        w = amp[:, None] * np.exp(-0.5 * (v[None, :] / sig[:, None]) ** 2)
        n = ks.size
        scans.append(CAL.PlaneScan(j, 0.7, 0.0, 0.3, 0, z_ax, ks, z, om, np.full(n, np.nan), np.full(n, 35.0),
                                   amp, np.full(n, 210.0), ("ok",) * n, which, 0.05 * om, amp, np.zeros(n),
                                   np.full(n, 1.0), np.full(n, 21), np.full(n, 2), FW * sig, np.zeros(n),
                                   np.tile([210.0, 0.0, 0.0], (n, 1)), v, w, np.full((n, v.size), 210.0)))
        truth.append((c, z_ax))
    return scans, truth


def fake_calib(cfg, scans, truth):
    g = CAL.fit_growth(scans, cfg)
    kernel = CAL.kernel_from_growth(g, cfg.renderer, max_delta_um=cfg.calibration.table_max_delta_um)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        CIO.save_calibration(path, g, kernel, "", {}, {}, dict(sigma0_um=float(kernel.kernel_table_sigma_um[-1]),
                                                               se_um=0.005, n_boot_ok=10, seed=0, n_resamples=10,
                                                               delta0_um=cfg.calibration.table_max_delta_um))
        return CIO.load_calibration(path)


def fake_far_fit(cfg, scans, truth, gamma, s0):
    from allen_diameter.analysis import far_field as FF
    sel = FF.select_far_planes(scans, cfg)
    fit = FF.fit_far_slope(sel, s0, cfg, k_star={int(s.node_id): 0 for s in scans})
    summary = FF.far_fit_summary(sel, fit, None, None, 0.01, 10, -0.3, s0, 0.005, cfg)
    return {"far_fit": summary}


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    rng = np.random.default_rng(SEED)
    near, t_near = make_scans(cfg, "near", 6, rng)
    calib = fake_calib(cfg, near, t_near)
    st = CR.stage_inputs(near, "near", cfg, calib)
    assert st is not None and st["has_fit"] and st["law"] is not None
    assert st["offsets"] == list(range(-3, 4)) and sorted(st["median"]) == st["offsets"]
    for nd, (c, z_ax) in zip(st["nodes"], t_near):
        # the fit recovers c_j and z_ax to 1e-9 on exact data (Block 10's reference check), so the centred values are
        # omega - c_j and the residuals from G_hat vanish
        assert np.max(np.abs(nd["centred"] - (nd["omega"] - c))) <= 1e-8, nd["node_id"]
        assert np.max(np.abs(nd["residual"])) <= 1e-8, nd["residual"]
        assert nd["used"].all()
    for o in st["offsets"]:
        vals = [nd["omega"][list(nd["offsets"]).index(o)] for nd in st["nodes"]]
        assert abs(st["median"][o] - np.median(vals)) <= 1e-12
    # growth_curve equals fit_growth's own curve on a grid
    g = CAL.fit_growth(near, cfg)
    grid = np.linspace(-1.0, 1.0, 41)
    assert np.max(np.abs(CR.growth_curve(calib["growth"])(grid) - g.growth(grid))) <= 1e-9
    # far: exact (F1) data beyond the anchor
    gamma = 0.9
    s0 = float(K.sigma_r(cfg.calibration.table_max_delta_um, cfg.renderer))
    far, t_far = make_scans(cfg, "far", 5, rng, gamma=gamma)
    ff = fake_far_fit(cfg, far, t_far, gamma, s0)
    assert abs(ff["far_fit"]["gamma"] - gamma) <= 1e-6, ff["far_fit"]["gamma"]
    stf = CR.stage_inputs(far, "far", cfg, calib, ff)
    assert stf["has_fit"] and stf["offsets"] == list(range(-16, 17))
    for nd, (c, z_ax) in zip(stf["nodes"], t_far):
        used = nd["used"]
        assert used.sum() == 24 and np.all(np.abs(nd["offsets"][used]) >= 5), used.sum()
        assert np.max(np.abs(nd["centred"][used] - (nd["omega"][used] - c))) <= 1e-6
        assert np.nanmax(np.abs(nd["residual"][used])) <= 1e-6, np.nanmax(np.abs(nd["residual"][used]))
    # the kernel law: the planted sigma_r at every used far plane and every near plane
    kl = CR.kernel_law_inputs(near + far, cfg, calib, ff)
    _, _, g_true = truth_growth(cfg)
    for d, s, side, used, which in kl["points"]:
        if used:
            expect = (math.sqrt(cfg.renderer.sigma_r0_um ** 2 + float(g_true(d))) if which == "near"
                      else s0 + gamma * (d - cfg.calibration.table_max_delta_um))
            assert abs(s - expect) <= 1e-5, (which, d, s, expect)
    assert kl["anchor"] is not None and abs(kl["anchor"][1] - s0) <= 1e-9
    assert abs(kl["table"][0][-1] - kl["anchor"][0]) <= 1e-9 and kl["table"][1][-1] == kl["anchor"][1]


def test_reference():
    from PIL import Image
    cfg = default_config()
    rng = np.random.default_rng(SEED + 1)
    near, t_near = make_scans(cfg, "near", 5, rng, noise=1e-3)
    far, t_far = make_scans(cfg, "far", 4, rng, noise=1e-3)
    calib = fake_calib(cfg, near, t_near)
    s0 = float(calib["anchor"]["sigma0_um"])
    ff = fake_far_fit(cfg, far, t_far, 0.79, s0)
    with tempfile.TemporaryDirectory() as tmp:
        written = CF.write_all(near + far, calib, ff, {}, cfg, tmp)
        names = sorted(os.path.relpath(w, tmp) for w in written)
        expect = ["example_far.png", "example_near.png", "kernel_law.png", "width_by_plane_far.png",
                  "width_by_plane_near.png"] + sorted("gaussians/%s_node_%d.png" % (w, i) for w, n in (("near", 5), ("far", 4))
                                                      for i in range(n))
        assert names == sorted(expect), names
        for w in written:
            assert os.path.getsize(w) > 2000, w
            with Image.open(w) as im:
                assert im.size[0] >= 600 and im.size[1] >= 300, (w, im.size)


def test_convergence():
    raise unittest.SkipTest("the figures have no discretisation to converge")


def test_invariants():
    cfg = default_config()
    rng = np.random.default_rng(SEED + 2)
    near, t_near = make_scans(cfg, "near", 6, rng, noise=2e-3)
    calib = fake_calib(cfg, near, t_near)
    st = CR.stage_inputs(near, "near", cfg, calib)
    shifted = [dataclasses.replace(sc, omega_um2=sc.omega_um2 + 0.5) for sc in near]
    calib2 = fake_calib(cfg, shifted, t_near)
    st2 = CR.stage_inputs(shifted, "near", cfg, calib2)
    for a, b in zip(st["nodes"], st2["nodes"]):
        assert np.max(np.abs(a["centred"] - b["centred"])) <= 1e-7 and np.max(np.abs(a["residual"] - b["residual"])) <= 1e-7
    med = CAL.median_by_offset(near)
    planted = dataclasses.replace(near[0], node_id=77, omega_um2=np.array([med[int(o)] for o in near[0].offsets()]))
    assert CR.stage_inputs(near + [planted], "near", cfg, calib)["example_id"] == 77


def test_contract():
    cfg = default_config()
    rng = np.random.default_rng(SEED + 3)
    near, t_near = make_scans(cfg, "near", 4, rng)
    far, t_far = make_scans(cfg, "far", 3, rng)
    inputs = CR.figure_inputs(near + far, cfg)
    assert set(inputs) == {"stages", "kernel_law", "examples", "gaussians"} and set(inputs["stages"]) == {"near", "far"}
    assert inputs["kernel_law"] is None and len(inputs["gaussians"]) == 7
    for which, st in inputs["stages"].items():
        assert not st["has_fit"] and st["law"] is None and st["example_id"] is not None
        assert all(np.isnan(nd["centred"]).all() for nd in st["nodes"])
    g = inputs["gaussians"][0]
    assert set(g) == {"node_id", "set", "v", "planes", "k_star"} and set(g["planes"][0]) == {
        "offset", "k", "w", "amp", "v_c", "sigma", "window_half", "omega", "status", "passes"}
    assert CR.gaussian_inputs(dataclasses.replace(near[0], w_gl=None)) is None
    with tempfile.TemporaryDirectory() as tmp:
        written = CF.write_all(near + far, None, None, None, cfg, tmp, every_node=False)
        assert sorted(os.path.basename(w) for w in written) == ["example_far.png", "example_near.png",
                                                                "width_by_plane_far.png", "width_by_plane_near.png"]


def test_determinism():
    cfg = default_config()
    rng = np.random.default_rng(SEED + 4)
    near, t_near = make_scans(cfg, "near", 3, rng, noise=1e-3)
    calib = fake_calib(cfg, near, t_near)
    outs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            w = CF.write_all(near, calib, None, {}, cfg, tmp, every_node=False)
            outs.append({os.path.basename(p): open(p, "rb").read() for p in w})
    assert outs[0].keys() == outs[1].keys() and all(outs[0][k] == outs[1][k] for k in outs[0])


def test_edge_cases():
    cfg = default_config()
    assert CR.stage_inputs([], "near", cfg) is None and CR.figure_inputs([], cfg)["stages"] == {}
    with tempfile.TemporaryDirectory() as tmp:
        assert CF.write_all([], None, None, None, cfg, tmp) == []
        one, _ = make_scans(cfg, "near", 1, np.random.default_rng(SEED + 5))
        st = CR.stage_inputs(one, "near", cfg)
        assert len(st["nodes"]) == 1 and st["example_id"] == 0
        assert len(CF.write_all(one, None, None, None, cfg, tmp, every_node=False)) == 2


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
