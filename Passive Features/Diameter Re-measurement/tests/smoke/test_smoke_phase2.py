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
                        category (ON passes; ALONGSIDE, NOT_ON fail)
    test_reference      the registration survey on the synthetic cell's own
                        traced path (the 2026-09-23 registration_check): ON,
                        |s*| <= 0.2 um, |dz*| <= 0.3 um; JSON as run_cell.py
                        reads it
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
                        this clone (no pull): sys.path and imports
    test_determinism    skipped: deterministic wiring over tested blocks
    test_edge_cases     an empty cache: no tables, a note; run_node with node
                        ids on no stretch: no rows, n_nodes 0; a registration
                        entry with null s* and dz* (a NOT ON node, as the
                        survey writes it): NaN columns and the node out of S;
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
from allen_diameter.loading import jpeg_tables, table_io  # noqa: E402
from fixtures_cell import synthetic_cell  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "Pillow", "matplotlib")


def cell_cfg():
    base = default_config()
    return dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))


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
