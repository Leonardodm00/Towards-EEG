"""Smoke test for the Phase II tools -- Block 11 in specs/SPEC.md (pilot
measurement, registration survey, camera-chain inputs, Colab bootstrap).

The real archive is out of reach here (api.brain-map.org, 403), so every tool
runs on a synthetic cell served through allen_image_io.fetch_zblock
(fixtures_cell.py) or on files written here.

Checks
    test_known_answer   survey.summarize on hand-made rows: percentiles equal
                        numpy.percentile, the suggested phantom mu range is
                        [p10, p90] of mu_hat over the nodes in S, the dark
                        share is counted over the converged nodes; the
                        registration summary's robust SD is 1.4826 x MAD;
                        the selection S reads registration_check sentences by
                        category (ON passes; ALONGSIDE, NOT_ON fail); plane
                        montage (2026-10-07): the image extent puts a pixel's
                        centre exactly where analysis.profiles.sample_profile
                        reads it, depth_alpha at its closed-form points, the
                        node selection ('differ': k_star != k_star_depth) and
                        the pilot comparison on hand-made rows
    test_reference      the registration survey on the synthetic cell's own
                        traced path (the 2026-09-23 registration_check): ON,
                        |s*| <= 0.2 um, |dz*| <= 0.3 um; JSON as run_cell.py
                        reads it; survey.node_planes equals the pilot's
                        measurement of the node (survey.measure_nodes) field by
                        field and makes the same provider request; on the
                        fixture's known geometry, the darkness centroid across
                        the tube in plane k*, with pixels placed by the
                        montage's extent, lies on Allen's centre line within
                        0.05 um
    test_convergence    skipped: no discretisation parameter
    test_invariants     skipped: wiring of tested blocks
    test_contract       run_node.run on the synthetic cell: the CSV and summary
                        (with the focus rule and k_star_vs_dip_depth, D-030)
                        files, every node measured, the background median
                        within 4 grey levels of background_B_gl and its clipped
                        SD within 15 % of a flat field's post-chain SD; a
                        figure per node; camera_calibration.run on a cache of
                        JPEGs saved with known tables (+ one that is not a
                        JPEG): the tables recovered exactly, the background
                        carried into the suggested fields, and the injected
                        noise matched back through the camera chain to within
                        15 % of the fixture's 3 grey levels; the bootstrap on
                        this clone (no pull): sys.path and imports;
                        node_planes.run: one PNG per node, the record says
                        'same as the pilot', the montage has one image panel
                        per valid focus plane, the segments (n, 9) flag the
                        stretch's own segments and give the soma's child its
                        own radius at both ends
    test_determinism    skipped: deterministic wiring over tested blocks
    test_edge_cases     an empty cache: no tables, a note; run_node with node
                        ids on no stretch: no rows, n_nodes 0; a registration
                        entry with null s* and dz* (a NOT ON node, as the
                        survey writes it): NaN columns and the node out of S;
                        node_planes on the soma and on a missing id: refused,
                        and node_planes.run records them as skipped; a missing
                        plane is drawn as a 'missing' panel without an image;
                        the bootstrap
                        without a clone and clone=False: FileNotFoundError

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_phase2.py

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
for _p in (SRC, HERE, WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import camera_fit, survey  # noqa: E402
from allen_diameter.analysis.phantoms import reject_reasons  # noqa: E402
from allen_diameter.analysis.node_pipeline import NodeResult  # noqa: E402
from allen_diameter.config import config_from_dict, default_config  # noqa: E402
from allen_diameter.analysis import profiles  # noqa: E402
from allen_diameter.loading import jpeg_tables, table_io  # noqa: E402
from allen_diameter.plotting import figures  # noqa: E402
from fixtures_cell import synthetic_cell  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "Pillow", "matplotlib")


def cell_cfg():
    base = default_config()
    return dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))


def _same_value(a, b):
    """Field-by-field equality of two results: dataclasses recursively, arrays element-wise (NaN equal
    to NaN for float arrays), floats with NaN equal to NaN, anything else by ==."""
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        return type(a) is type(b) and all(_same_value(getattr(a, f.name), getattr(b, f.name))
                                          for f in dataclasses.fields(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        floats = a.dtype.kind in "fc" and b.dtype.kind in "fc"
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=floats))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(40):
        mu, d = float(rng.uniform(0.3, 3.0)), float(rng.uniform(0.3, 2.0))
        conv = k % 5 != 0
        alpha = mu * d
        rows.append(dict(in_S=conv and alpha <= 1.0, reject="" if conv and alpha <= 1.0 else ("status:at_bound" if not conv else "dark"),
                         fit_status="converged" if conv else "at_bound", alpha_hat=alpha, mu_hat_per_um=mu,
                         d_hat_um=d, allen_radius_um=0.25, phi_rad=0.1, calibration_node=False))
    s = survey.summarize(rows, cfg)
    inS = [r for r in rows if r["in_S"]]
    mus = np.array([r["mu_hat_per_um"] for r in inS])
    assert s["n_in_S"] == len(inS) and s["mu_hat_per_um"]["n"] == len(inS)
    for q in survey.PERCENTILES:
        assert s["mu_hat_per_um"]["p%d" % q] == float(np.percentile(mus, q))
    assert s["suggested_phantom_mu_range_per_um"] == [float(np.percentile(mus, 10)), float(np.percentile(mus, 90))]
    conv = [r for r in rows if r["fit_status"] == "converged"]
    assert s["dark_share_of_converged"] == sum(1 for r in conv if r["alpha_hat"] > cfg.measure.alpha_dark_flag) / len(conv)
    # the selection reads the registration_check sentences by category
    nan = float("nan")

    def node(verdict):
        return NodeResult(1, 3, 0.0, 0.0, 0.0, 0.0, verdict, nan, nan, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False,
                          200.0, "block_masked", 1.0, 0.5, 0.0, 0.5, "converged", (),
                          None, "gradient_energy", np.empty(0, dtype=int), np.empty(0), np.empty(0), -1)
    assert reject_reasons(node("ON THE PROCESS (lateral offset +0.12 um)")) == []
    assert reject_reasons(node("")) == []
    assert reject_reasons(node("ALONGSIDE: a ridge +1.20 um to the side -- snap needed before measuring")) == \
        ["registration:ALONGSIDE"]
    assert reject_reasons(node("NOT ON A VISIBLE PROCESS (p=0.300 > 0.05): wrong structure")) == ["registration:NOT_ON"]
    import registration_survey as RS
    res = {i: dict(verdict="ON", lateral_offset_um=x, z_offset_um=0.1 * x) for i, x in enumerate([-0.3, 0.1, 0.2, 0.5, 0.05])}
    res[9] = dict(verdict="NOT ON", lateral_offset_um=None, z_offset_um=None)
    summ = RS.summarize(res)
    x = np.array([-0.3, 0.1, 0.2, 0.5, 0.05])
    assert summ["verdicts"] == {"ON": 5, "NOT_ON": 1}
    assert abs(summ["lateral_offset_um"]["robust_sd"] - 1.4826 * np.median(np.abs(x - np.median(x)))) <= 1e-15
    # plane montage. The extent puts pixel (row, col) at ((left + col) p, (top + row) p): one dark pixel,
    # sampled by profiles.sample_profile at the centre the extent gives it, is read back as itself
    # (bilinear interpolation at a pixel centre; 1e-6 gl covers the float64 rounding of the coordinate,
    # ~1e-15 px times the 183 gl step). Half a pixel off would read (17 + 200) / 2.
    from allen_image_io import CropFrame
    p = cfg.acquisition.res0_um
    fr = CropFrame(37, -12, 0, p)
    img = np.full((5, 7), 200.0)
    img[3, 4] = 17.0
    ext = figures.image_extent_um(fr, img.shape)
    assert np.allclose(ext, [36.5 * p, 43.5 * p, -7.5 * p, -12.5 * p], rtol=0, atol=1e-12), ext
    at = (ext[0] + 4.5 * (ext[1] - ext[0]) / 7, ext[3] + 3.5 * (ext[2] - ext[3]) / 5)
    got = profiles.sample_profile(img, fr, at, (0.0, 1.0), (1.0, 0.0), np.zeros(1))
    assert abs(got[0] - 17.0) <= 1e-6, got
    # depth_alpha at its closed-form points: inside the segment's depth range and at solid_um 1, half-way
    # to fade_um 1 - 0.8 / 2, beyond fade_um the floor; the order of the ends does not matter
    da = figures.depth_alpha
    assert da(0.0, 0.5, 0.3, 1.0, 3.0) == 1.0 and da(0.0, 0.0, 1.0, 1.0, 3.0) == 1.0
    assert abs(da(0.0, 0.0, 2.0, 1.0, 3.0) - 0.6) <= 1e-12 and abs(da(0.5, 0.0, -2.0, 1.0, 3.0) - 0.6) <= 1e-12
    assert da(0.0, 0.0, 5.0, 1.0, 3.0) == 0.2
    # node selection and the pilot comparison of scripts/node_planes.py
    import node_planes as NP
    prow = [dict(node_id=10, z_sub_um=0.1, k_star=5, k_star_depth=6), dict(node_id=11, z_sub_um=nan, k_star=-1,
                                                                            k_star_depth=3),
            dict(node_id=12, z_sub_um=0.2, k_star=7, k_star_depth=7), dict(node_id=13, z_sub_um=0.0, k_star=2,
                                                                            k_star_depth=0)]
    assert NP.select_nodes("differ", prow, 6) == [10, 13] and NP.select_nodes("differ", prow, 1) == [10]
    assert NP.select_nodes(" 13, 4,", None, 6) == [13, 4]
    for rows_bad in (None, [dict(node_id=1, z_sub_um=0.0, k_star=1)]):
        try:
            NP.select_nodes("differ", rows_bad, 6)
        except ValueError:
            continue
        raise AssertionError("'differ' without pilot rows carrying k_star_depth must be refused")
    nr = node("")
    row = dict(k_star=0, k_star_depth=-1, z_sub_um=0.0, d_hat_um=1.0, mu_hat_per_um=0.5, focus_rule="gradient_energy")
    assert NP.compare_with_pilot(nr, row) == []
    assert NP.compare_with_pilot(nr, dict(row, d_hat_um=1.0 + 1e-6, k_star=1)) == ["k_star", "d_hat_um"]
    assert NP.compare_with_pilot(dataclasses.replace(nr, d_hat_um=nan), dict(row, d_hat_um=nan)) == []
    assert NP.compare_with_pilot(nr, dict(row, d_hat_um=nan, focus_rule="dip_depth")) == ["d_hat_um", "focus_rule"]


def test_reference():
    import allen_image_plot as aip
    import registration_survey as RS
    cfg = cell_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        swc, fetcher, planes, path = synthetic_cell(tmp, cfg, d_true=0.8, mu=1.0, n_nodes=8)
        res, summ = RS.run(aip.read_swc(path), fetcher, planes, [5], os.path.join(tmp, "reg"), "999",
                           cfg.acquisition.res0_um, cfg.acquisition.dz_um, n_each_way=6, log=lambda m: None)
        r = res[5]
        assert r["verdict"].startswith("ON THE PROCESS") and abs(r["lateral_offset_um"]) <= 0.2 \
            and abs(r["z_offset_um"]) <= 0.3, r
        with open(os.path.join(tmp, "reg", "registration_999.json")) as f:
            back = {int(k): v for k, v in json.load(f).items()}
        assert set(back[5]) >= {"verdict", "lateral_offset_um", "z_offset_um"} and summ["verdicts"] == {"ON": 1}
    # plane montage: node_planes measures the node as the pilot does (survey.measure_nodes), with the same
    # single provider request -- hence, on real data, a block served by the image cache
    import run_cell
    with tempfile.TemporaryDirectory() as tmp:
        swc, fetcher, planes, _ = synthetic_cell(tmp, cfg, d_true=0.8, mu=1.0)
        provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
        calls = []

        def recorder(*args):
            calls.append(args)
            return provider(*args)
        meas = survey.measure_nodes(swc, recorder, cfg, only={4})
        assert len(calls) == len(meas)                     # one request per measured node
        at = [i for i, m in enumerate(meas) if m[0].node_id == 4][0]
        pilot_call, pilot_res = calls[at], meas[at][0]
        del calls[:]
        pl = survey.node_planes(swc, recorder, cfg, 4)
        assert calls == [pilot_call], (calls, pilot_call)
        bad = [f.name for f in dataclasses.fields(pilot_res)
               if not _same_value(getattr(pl["result"], f.name), getattr(pilot_res, f.name))]
        assert not bad, bad
        assert pl["index"] == meas[at][3] and pl["allen_radius_um"] == meas[at][1]
        # the same with a global alignment (2 px in x): node_planes passes it on, so the request moves
        shift = dict(shift_full_px=(2.0, 0.0))
        del calls[:]
        meas_s = survey.measure_nodes(swc, recorder, cfg, shift, only={4})
        call_s = calls[at]
        del calls[:]
        pl_s = survey.node_planes(swc, recorder, cfg, 4, shift)
        assert calls == [call_s] and call_s != pilot_call and _same_value(pl_s["result"], meas_s[at][0]), (calls, call_s)
        # alignment through the montage's extent, on the fixture's known geometry (the tube is rendered on
        # the traced path at heading 0.3 rad): in plane k*, the darkness-weighted mean of the offset across
        # the path, over the pixels within 1 um along and 1.5 um across the path of node 4, placed by
        # image_extent_um, is 0. Tolerance 0.05 um: below half a pixel (0.057 um), above the centroid's noise;
        # the half-pixel convention itself is pinned exactly in test_known_answer.
        res = pl["result"]
        assert math.isfinite(res.z_sub_um), res
        img = np.asarray(pl["block"][int(res.k_star) - int(pl["ks"][0])], dtype=float)
        ext = figures.image_extent_um(pl["frame"], img.shape)
        H, W = img.shape
        XX, YY = np.meshgrid(ext[0] + (np.arange(W) + 0.5) * (ext[1] - ext[0]) / W,
                             ext[3] + (np.arange(H) + 0.5) * (ext[2] - ext[3]) / H)
        th = 0.3
        u = (XX - res.x_um) * math.cos(th) + (YY - res.y_um) * math.sin(th)
        v = -(XX - res.x_um) * math.sin(th) + (YY - res.y_um) * math.cos(th)
        sel = (np.abs(u) <= 1.0) & (np.abs(v) <= 1.5)
        w = np.clip(np.median(img) - img, 0.0, None)[sel]
        v_bar = float(np.sum(w * v[sel]) / np.sum(w))
        assert abs(v_bar) <= 0.05, v_bar


def test_convergence():
    raise unittest.SkipTest("no discretisation parameter")


def test_invariants():
    raise unittest.SkipTest("wiring of tested blocks (5, 6, 8)")


def test_contract():
    import run_node
    import camera_calibration as CC
    import colab_bootstrap as CB
    cfg = cell_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        swc, fetcher, planes, _ = synthetic_cell(tmp, cfg, d_true=0.8, mu=1.0)
        import run_cell
        provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
        out = os.path.join(tmp, "pilot")
        rows, summ = run_node.run(swc, provider, cfg, out, "999", only={4}, figures=True, background=True,
                                  log=lambda m: None)
        assert len(rows) == 6 and summ["n_nodes"] == 6 and summ["n_in_S"] >= 4, (summ, [r["reject"] for r in rows])
        back = table_io.read_rows([os.path.join(out, "pilot_999.csv")])
        assert len(back) == 6 and {"in_S", "reject", "calibration_node", "bg_median_gl", "allen_radius_um"} <= set(back[0])
        with open(os.path.join(out, "pilot_summary_999.json")) as f:
            js = json.load(f)
        assert js["estimator_hash"] == cfg.signature_hash("estimator") and "suggested_phantom_mu_range_per_um" in js
        # D-030: every row carries the focus rule and the dip depth's plane; the summary counts where they differ
        assert all(r["focus_rule"] == cfg.measure.focus_rule and isinstance(r["k_star_depth"], int) for r in back)
        found = [r for r in rows if math.isfinite(r["z_sub_um"])]
        agree = js["k_star_vs_dip_depth"]
        assert js["focus_rule"] == "gradient_energy" and agree["n_nodes"] == len(found), agree
        assert agree["n_differ"] == sum(1 for r in found if r["k_star"] != r["k_star_depth"]), agree
        bg = summ["background"]
        assert abs(bg["B_bar_gl"]["p50"] - cfg.renderer.background_B_gl) <= 4.0, bg
        # the background is read after JPEG: its clipped SD is the flat field's post-chain SD, not the
        # injected 3 grey levels (2.1 at quality 85); 15 % covers the sampling of ~6000 pixels per node
        # and the defocused tube's faint tails beyond the mask
        flat = camera_fit.post_chain_sd(cfg.renderer.noise_sd_gl, cfg.renderer, np.random.default_rng(SEED))
        assert abs(bg["clipped_sd_gl"]["p50"] / flat - 1) <= 0.15, (bg, flat)
        pngs = [p for p in os.listdir(os.path.join(out, "figures")) if p.endswith(".png")]
        assert len(pngs) == sum(1 for r in rows if math.isfinite(r["z_sub_um"])) and all(
            os.path.getsize(os.path.join(out, "figures", p)) > 1000 for p in pngs)
        # plane montage of node 4 (scripts/node_planes.py): a PNG, a record that says the re-measurement
        # equals the pilot row, one image panel per valid focus plane, the segments of the trace
        import node_planes as NP
        import matplotlib.pyplot as plt
        recs = NP.run(swc, provider, cfg, [4], os.path.join(tmp, "planes"), "999", pilot_rows=back,
                      log=lambda m: None)
        assert recs[0]["pilot"] == "same" and os.path.getsize(recs[0]["png"]) > 1000, recs
        with open(os.path.join(tmp, "planes", "planes_999.json")) as f:
            assert json.load(f)[0]["node_id"] == 4
        pl = survey.node_planes(swc, provider, cfg, 4)
        fk = np.asarray(pl["result"].focus_planes, dtype=int)
        fig = figures.plane_montage(pl, cfg.acquisition.dz_um, cfg.measure.profile_half_um)
        n_valid = int(np.sum(pl["valid"][fk - int(pl["ks"][0])]))
        assert fk.size >= 7 and sum(1 for ax in fig.axes if ax.images) == n_valid == fk.size, (fk, n_valid)
        plt.close(fig)
        seg, br = pl["segments"], pl["branch"]
        own = seg[:, 8] > 0.5
        # the fixture's straight stretch lies inside the block: its len - 1 segments are the stretch's own,
        # each from one Branch node to the next (parent end first); the soma's child segment (soma at the
        # origin) is not, and carries the child's radius at both ends (the soma's 4 um is not a neurite's)
        assert seg.shape == (len(br.ids), 9) and int(own.sum()) == len(br.ids) - 1, seg
        for row_ in seg[own]:
            i0 = int(np.flatnonzero(np.all(np.abs(br.xyz_um - row_[0:3]) <= 1e-12, axis=1))[0])
            assert np.all(np.abs(br.xyz_um[i0 + 1] - row_[3:6]) <= 1e-12) and row_[6] == row_[7] == 0.3
        soma = seg[np.all(np.abs(seg[:, 0:3]) <= 1e-12, axis=1)]
        assert soma.shape[0] == 1 and soma[0, 8] == 0.0 and soma[0, 6] == soma[0, 7] == 0.3, soma
        # camera inputs: a cache of JPEGs saved with known tables, and one file that is not a JPEG
        from PIL import Image
        cache = os.path.join(tmp, "cache")
        os.makedirs(cache)
        tables = [[2 + (j * 7) % 31 for j in range(64)]]     # a greyscale JPEG carries one (luminance) table
        img = Image.fromarray(np.random.default_rng(SEED).integers(150, 220, (64, 64), dtype=np.uint8))
        for k in range(3):
            img.save(os.path.join(cache, "%d_abc.img" % k), format="JPEG", qtables=tables)
        with open(os.path.join(cache, "bad.img"), "wb") as f:
            f.write(b"not a jpeg")
        res = CC.run(cache, os.path.join(tmp, "camera"), "999", os.path.join(out, "pilot_summary_999.json"))
        # the tables themselves are suggested for the configuration [corrected 2026-10-07: a file path was
        # suggested, which the renderer never read]; the JSON file stays as a record
        assert res["renderer"]["jpeg_qtables"] == tables
        assert jpeg_tables.load_qtables(res["jpeg_qtables_file"]) == tables and res["pillow_version"]
        sig = json.loads(cfg.to_json())
        sig["renderer"].update({k: v for k, v in res["renderer"].items()})
        cfg_cam = config_from_dict(sig)
        assert cfg_cam.renderer.jpeg_qtables == tuple(tuple(t) for t in tables), cfg_cam.renderer.jpeg_qtables
        assert res["table_sets"][0]["count"] == 3 and res["table_sets"][-1]["n_tables"] is None
        assert res["renderer"]["background_B_gl"] == bg["B_bar_gl"]["p50"]
        # closed loop through the camera chain: with the fixture's own JPEG (no cache tables) the matched
        # injected SD comes back near the configured 3 grey levels
        cache2 = os.path.join(tmp, "cache2")
        os.makedirs(cache2)
        res2 = CC.run(cache2, os.path.join(tmp, "camera2"), "999", os.path.join(out, "pilot_summary_999.json"), cfg=cfg)
        assert abs(res2["renderer"]["noise_sd_gl"] / cfg.renderer.noise_sd_gl - 1) <= 0.15, res2["renderer"]
    env = CB.bootstrap(repo_dir=str(WS.parent.parent), pull=False, verbose=False)
    assert env["src"] in sys.path and env["commit"] and os.path.isdir(env["workstream"])


def test_determinism():
    raise unittest.SkipTest("deterministic wiring over tested blocks (covered by Blocks 4-6)")


def test_edge_cases():
    import run_node
    import camera_calibration as CC
    import colab_bootstrap as CB
    cfg = cell_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "empty"))
        res = CC.run(os.path.join(tmp, "empty"), os.path.join(tmp, "camera"), "1")
        assert res["renderer"] == {} and any("no JPEG tables" in n for n in res["notes"]), res
        swc, fetcher, planes, _ = synthetic_cell(tmp, cfg, d_true=0.8, mu=1.0, ks=np.arange(-2, 3))
        import run_cell
        provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
        rows, summ = run_node.run(swc, provider, cfg, os.path.join(tmp, "pilot"), "1", only={12345}, log=lambda m: None)
        assert rows == [] and summ["n_nodes"] == 0
        # a registration file as registration_survey writes it for a NOT ON node: s* and dz* are null
        regs = {4: dict(verdict="NOT ON A VISIBLE PROCESS (p=0.300 > 0.05): wrong structure",
                        lateral_offset_um=None, z_offset_um=None)}
        rows, _ = run_node.run(swc, provider, cfg, os.path.join(tmp, "pilot2"), "1", regs=regs, only={4},
                               log=lambda m: None)
        r4 = [r for r in rows if r["node_id"] == 4][0]
        assert math.isnan(r4["s_star_um"]) and "registration:NOT_ON" in r4["reject"], r4
        # plane montage: the soma (id 1) and an id not in the SWC are refused, and node_planes.run records
        # them as skipped; the planes outside the 5-plane stack (and one made invalid here) are drawn as
        # 'missing' panels with no image; a downsampled frame is refused by the extent
        for bad_id in (1, 12345):
            try:
                survey.node_planes(swc, provider, cfg, bad_id)
            except ValueError:
                continue
            raise AssertionError("node_planes(%d) must be refused" % bad_id)
        import node_planes as NP
        import matplotlib.pyplot as plt
        recs = NP.run(swc, provider, cfg, [1, 12345], os.path.join(tmp, "planes"), "1", log=lambda m: None)
        assert all(r.get("skipped") for r in recs) and not any("png" in r for r in recs), recs
        pl = survey.node_planes(swc, provider, cfg, 4)
        fk = np.asarray(pl["result"].focus_planes, dtype=int)
        valid = np.array(pl["valid"], dtype=bool)
        valid[int(fk[len(fk) // 2]) - int(pl["ks"][0])] = False
        pl["valid"] = valid
        fig = figures.plane_montage(pl, cfg.acquisition.dz_um, cfg.measure.profile_half_um)
        n_valid = int(np.sum(valid[fk - int(pl["ks"][0])]))
        n_missing = sum(1 for ax in fig.axes for t in ax.texts if t.get_text() == "missing")
        assert sum(1 for ax in fig.axes if ax.images) == n_valid and n_missing == fk.size - n_valid >= 3, \
            (fk, n_valid, n_missing)
        plt.close(fig)
        from allen_image_io import CropFrame
        try:
            figures.image_extent_um(CropFrame(0, 0, 1, cfg.acquisition.res0_um), (4, 4))
        except ValueError:
            pass
        else:
            raise AssertionError("a downsampled frame must be refused by image_extent_um")
        try:
            CB.bootstrap(repo_dir=os.path.join(tmp, "nothing"), pull=False, clone=False, verbose=False)
        except FileNotFoundError:
            return
        raise AssertionError("a missing clone with clone=False must be refused")


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
