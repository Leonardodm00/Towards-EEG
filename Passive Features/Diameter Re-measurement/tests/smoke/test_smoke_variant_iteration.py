"""Smoke test for Cell 4m, each fit variant's own Allen-guided iteration --
Block 11 in specs/SPEC.md (D-047, the user's request of 2026-10-14, 15:35).

For node j of a Cell 4j record and variant V: Cell 4j's rule (round 1's line
+-m 2 r_A / 2, later lines +-m d_{i-1} / 2, the plane of the largest gradient
energy over the line, until a plane repeats or after R rounds), with m and R
read from the record, Cell 4j's squares, and d_i = V's fit at the plane
(Block 5's inputs there). The reference (Block 5's fit) must redo Cell 4j's
record. Outcome: k_iter, d_iter, the selection line +-h_sel through the SWC
node, the edges s + v0 -+ d / 2 on it.

Checks
    test_known_answer   iteration_params reads the record's m and R (2 and 3 when
                        the record says so); the reference redoes Cell 4j's
                        record to the last bit (planes, k_iter, d_iter, status,
                        restacks) on the 0.8 um cell and on the 3.0 um tube
                        (restacks); the variant Block 5 would be (Gaussian, fixed
                        width, unweighted) gives Cell 4j's planes and d_iter
                        (1e-6 um)
    test_reference      each round of a variant's iteration: round 1's line
                        m r_A, a later line m d_(i-1) / 2, its plane the argmax
                        of the gradient energy computed here, its d the fit of
                        fit_variants called here on refit_at_plane's inputs; the
                        selection line, s and the edges from their definitions
    test_convergence    SKIP: Cell 4j's loop and the fits are tested in their own
                        suites; nothing new iterates or is discretised
    test_invariants     restack: the smallest square made that holds the line,
                        a new one only when none does, each holding its line; a
                        gain of 0.75 on every plane leaves the planes and d of
                        two variants (1e-6 um)
    test_contract       select_specs (50 at the fitted width by default);
                        flag_of; NodeSquares refuses planes that are not
                        k_SWC +- n; a plane without a profile gives a NaN fit and
                        stops the variant's iteration (fit_failed); a square the
                        provider cannot make ends that variant's iteration with
                        an error row, not the run; run() skips Cell 4j's
                        skipped records with their reason
    test_determinism    two runs on one node: the same rows
    test_edge_cases     run() end to end on the 0.8 um cell (a reduced set):
                        files, CSV rows (the reference first), JSON keys, the
                        log line; the full 50 on one node with both figures,
                        every text inside, the edges drawn along the branch at
                        the fitted edges; with both width modes a row per
                        kernel and width; the CLI passes its options (server
                        stubbed) and refuses a bad width mode, lambda, family

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_variant_iteration.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import importlib.metadata
import inspect
import io
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
for p in (SRC, HERE, WS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from allen_diameter.config import default_config  # noqa: E402

SEED = 20261014
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
NAN = float("nan")


def _cell4j(tmp, d_true=0.8, nodes=(3, 4, 1), line_mult=None, max_rounds=5, provider_wrap=None):
    """The synthetic cell (fixtures_cell), Cell 4j's records read back from its JSON, and the provider."""
    import iteration_fits as IF
    import plane_differences as PD
    import run_cell
    from fixtures_cell import synthetic_cell
    base = default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    swc, fetcher, planes, swc_path = synthetic_cell(tmp, cfg, d_true=d_true, mu=1.0)
    prov = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    if provider_wrap is not None:
        prov = provider_wrap(prov)
    out = os.path.join(tmp, "iter")
    PD.run(swc, prov, cfg, list(nodes), out, "999", planes_half=6, evaluation="iterate", line_mult=line_mult,
           max_rounds=max_rounds, log=lambda m: None)
    path = os.path.join(out, "planeiter_999.json")
    return cfg, swc, prov, swc_path, IF.read_records(path), path


def _texts_inside(fig):
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = list(fig.texts) + [ax.title for ax in fig.axes]
    texts += [t for leg in fig.legends for t in leg.get_texts()]
    for ax in fig.axes:
        texts += list(ax.texts)
    for t in texts:
        if t.get_text():
            bb = t.get_window_extent(renderer=ren)
            assert bb.x0 >= -1.0 and bb.x1 <= fig.bbox.width + 1.0 and bb.y0 >= -1.0 and \
                bb.y1 <= fig.bbox.height + 1.0, (t.get_text()[:60], bb.x0, bb.x1, bb.y0, bb.y1)


def _refuses(func, *args, exc=ValueError):
    try:
        func(*args)
    except exc:
        return
    raise AssertionError("%s accepted %r" % (getattr(func, "__name__", func), args))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    import variant_iteration as VI
    from allen_diameter.analysis import fit_variants as FV
    assert VI.iteration_params(dict(line_mult=2.0, max_rounds=3)) == (2.0, 3)
    assert VI.iteration_params(dict(line_mult=3.0, max_rounds=5)) == (3.0, 5)
    base = FV.VariantSpec("gauss", None, False, False, None)
    for d_true, nodes in ((0.8, (3, 4)), (3.0, (4,))):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, swc, prov, _p, recs, _j = _cell4j(tmp, d_true, nodes)
            for rec in recs:
                sq = VI.NodeSquares(swc, prov, cfg, rec)
                ev = VI.run_iteration(sq, rec, None)
                assert VI.compare_reference(ev, rec) == [], VI.compare_reference(ev, rec)
                rounds = [dict(x) for x in rec["rounds"]]
                rounds[0]["k"] = -99
                bad = VI.compare_reference(ev, dict(rec, d_iter_um=rec["d_iter_um"] + 1e-9, rounds=rounds,
                                                    iter_status="cycle", n_restacks=rec["n_restacks"] + 1))
                assert len(bad) == 4 and all(n.startswith("the reference") for n in bad), bad
                assert ev["d_final_um"] == rec["d_iter_um"] and ev["k_final"] == rec["k_iter"]
                if d_true == 3.0:
                    assert rec["n_restacks"] >= 1 and ev["n_restacks"] == rec["n_restacks"]
                evb = VI.run_iteration(sq, rec, base)        # Block 5's fit through the variants' machinery
                assert [x["k"] for x in evb["rounds"]] == [x["k"] for x in rec["rounds"]]
                assert abs(evb["d_final_um"] - rec["d_iter_um"]) <= 1e-6, (evb["d_final_um"], rec["d_iter_um"])


def test_reference():
    from allen_diameter.analysis import fit_variants as FV
    _check_rounds(0.8, (FV.VariantSpec("lorentz", None, False, True, None),
                        FV.VariantSpec("gennorm", None, True, True, 0.5)))
    _check_rounds(3.0, (FV.VariantSpec("gauss", None, False, True, None),), min_planes=2)


def _check_rounds(d_true, specs, min_planes=1):
    import plane_differences as PD
    import variant_iteration as VI
    from allen_diameter.analysis import fit_variants as FV
    from allen_diameter.analysis import focus
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, d_true, (4,))
        rec = recs[0]
        sq = VI.NodeSquares(swc, prov, cfg, rec)
        B, _f, _m = PD.block_background(sq.first, float(rec["bg_margin_um"]))
        for spec in specs:
            ev = VI.run_iteration(sq, rec, spec)
            assert abs(ev["rounds"][0]["half_um"] - sq.m * float(rec["allen_radius_um"])) <= 1e-12
            d_prev = None
            for x in ev["rounds"]:
                if x["k"] is None:
                    continue
                if d_prev is not None:
                    assert abs(x["half_um"] - 0.5 * sq.m * d_prev) <= 1e-12, (x["half_um"], d_prev)
                st = sq.restack(x["half_um"])
                _h, v, _y, _e, prof = PD.line_profiles(st, cfg, x["half_um"])
                G = focus.plane_gradient_energies(prof, v, B, cfg.measure, st["valid"])
                assert x["k"] == int(st["ks"][int(np.nanargmax(G))]), (x["k"], G)
                inp = PD.refit_at_plane(sq.first, cfg, x["k"], with_profile=True)
                fits, _b = FV.fit_grid(np.asarray(inp["v_um"]), np.asarray(inp["profile"]), inp["phi_rad"],
                                       inp["B_bar"], cfg.measure, [spec])
                assert fits[0].d_um == x["d_hat_um"] and fits[0].v0_um == x["v0_hat_um"], (fits[0].d_um, x)
                d_prev = x["d_hat_um"]
            sel = VI.selection(ev, sq.first)
            picked = [x for x in ev["rounds"] if x["k"] is not None]
            assert len({x["k"] for x in picked}) >= min_planes, [x["k"] for x in picked]
            assert sel["h_sel"] == picked[-1]["half_um"] and sel["k_iter"] == ev["k_final"]
            r = sq.first["result"]
            th = float(sq.first["theta"])
            y_hat = np.array([-math.sin(th), math.cos(th)])
            o = np.asarray(sq.first["o"], dtype=float)
            s = float(np.dot(np.array([r.cx_um, r.cy_um]) - o, y_hat))
            assert abs(sel["s_um"] - s) <= 1e-12
            assert np.allclose(sel["line_xy"], [o - sel["h_sel"] * y_hat, o + sel["h_sel"] * y_hat], rtol=0, atol=1e-12)
            want = np.array([s + sel["v0"] - 0.5 * sel["d"], s + sel["v0"] + 0.5 * sel["d"]])
            assert np.allclose(sel["edges_v"], want, rtol=0, atol=1e-12)
            assert np.allclose(sel["edges_xy"], o[None, :] + want[:, None] * y_hat[None, :], rtol=0, atol=1e-12)


def test_convergence():
    raise unittest.SkipTest("Cell 4j's loop and the fits are tested in their own suites; nothing new iterates or is "
                            "discretised here")


def test_invariants():
    import variant_iteration as VI
    from allen_diameter.analysis import fit_variants as FV
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 0.8, (4,))
        rec = recs[0]
        sq = VI.NodeSquares(swc, prov, cfg, rec)
        half0 = float(sq.first["square_half_um"])
        p = float(sq.first["frame"].res_um_px)
        h_in = half0 - 2 * p - 0.01
        assert sq.restack(h_in) is sq.first and len(sq.made) == 1          # it holds the line: no new square
        assert not VI.NodeSquares.holds(sq.first, half0 - p)              # a line within 2 pixels of the edge
        edge = sq.restack(half0 - p)
        assert edge is not sq.first and len(sq.made) == 2 and VI.NodeSquares.holds(edge, half0 - p)
        sq.made = [x for x in sq.made if x is not edge]     # by identity: the squares hold arrays
        big = sq.restack(half0 + 1.0)
        assert len(sq.made) == 2 and VI.NodeSquares.holds(big, half0 + 1.0)
        assert sq.restack(half0 + 0.5) is big and len(sq.made) == 2        # the smallest square that holds it
        bigger = sq.restack(half0 + 2.0)
        assert len(sq.made) == 3 and sq.restack(half0 + 0.9) is big and sq.largest is bigger
        specs = [FV.VariantSpec("gauss", None, False, True, None), FV.VariantSpec("student", None, True, True, 1.0)]
        sq = VI.NodeSquares(swc, prov, cfg, rec)
        ref = [VI.run_iteration(sq, rec, s) for s in specs]

    def gain(pr):
        def g(*a, **k):
            block, ks, valid, x = pr(*a, **k)
            return 0.75 * np.asarray(block, dtype=float), ks, valid, x
        return g
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 0.8, (4,), provider_wrap=gain)
        sq = VI.NodeSquares(swc, prov, cfg, recs[0])
        for s, ev0 in zip(specs, ref):
            ev = VI.run_iteration(sq, recs[0], s)
            assert [x["k"] for x in ev["rounds"]] == [x["k"] for x in ev0["rounds"]]
            assert abs(ev["d_final_um"] - ev0["d_final_um"]) <= 1e-6, (s.key, ev["d_final_um"], ev0["d_final_um"])


def test_contract():
    import variant_iteration as VI
    from allen_diameter.analysis import fit_variants as FV
    specs = VI.select_specs()
    assert len(specs) == 50 and all(s.width_fitted for s in specs) and specs[0].key == "gauss|w fitted|none"
    assert [s.key for s in VI.select_specs(["lorentz"], [0.5])] == [
        "gauss|w fitted|none", "gauss|w fitted|lam=0.5", "lorentz|w fitted|none", "lorentz|w fitted|lam=0.5"]
    row = dict(fit_status="converged", d_iter_um=0.5, irls_ok=None)
    assert VI.flag_of(row) == "" and VI.flag_of(dict(row, fit_status="at_bound")) == "*"
    assert VI.flag_of(dict(row, d_iter_um=NAN)) == "x" and VI.flag_of(dict(row, fit_status="failed")) == "x"
    assert VI.flag_of(dict(row, irls_ok=False)) == "~" and VI.flag_of(dict(row, fit_status="at_bound",
                                                                         irls_ok=False)) == "*~"
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 0.8, (4, 1))
        rec = recs[0]
        _refuses(VI.NodeSquares, swc, prov, cfg, dict(rec, ks=rec["ks"][:-2]))
        sq = VI.NodeSquares(swc, prov, cfg, rec)
        k_dark = int(rec["k_iter"])
        sq._inputs[k_dark] = dict(sq.inputs(k_dark), fit_status="profile_nan")      # a plane without a profile
        spec = FV.VariantSpec("lorentz", None, False, True, None)
        d = sq.fit_dict(k_dark, spec)
        assert math.isnan(d["d_hat_um"]) and d["fit_status"] == "profile_nan"
        ev = VI.run_iteration(sq, rec, spec)
        if ev["rounds"][0]["k"] == k_dark:
            assert ev["status"] == "fit_failed" and math.isnan(ev["d_final_um"]), ev["status"]
        def broken(h):                                    # a square the provider cannot make
            raise ValueError("no square")
        sq2 = VI.NodeSquares(swc, prov, cfg, rec)
        sq2.restack = broken
        sq2.first = dict(sq2.first, square_half_um=0.0)   # so that round 1 already asks for a larger square
        ev = VI.run_iteration_safe(sq2, rec, spec)
        row = VI.variant_record(spec, ev, VI.selection(ev, sq.first), 1.0)
        assert row["iter_status"] == "error: no square" and math.isnan(row["d_iter_um"]), row["iter_status"]
        assert VI.flag_of(row) == "x" and row["planes"] == "" and row["k_iter"] is None
        out = VI.run(swc, prov, cfg, recs, os.path.join(tmp, "o"), "999", specs=[specs[0]], log=lambda m: None)
        assert out[1]["node_id"] == 1 and out[1]["skipped"].startswith("Cell 4j: "), out[1]


def test_determinism():
    import variant_iteration as VI
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 0.8, (4,))
        specs = VI.select_specs(["skewnorm"], [0.5])
        a = VI.run(swc, prov, cfg, recs, os.path.join(tmp, "a"), "999", specs=specs, log=lambda m: None)
        b = VI.run(swc, prov, cfg, recs, os.path.join(tmp, "b"), "999", specs=specs, log=lambda m: None)
        assert json.dumps(a[0]["variants"]) == json.dumps(b[0]["variants"])


def test_edge_cases():
    import matplotlib.pyplot as plt
    import variant_iteration as VI
    from allen_diameter.plotting import figures as FG
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, swc_path, recs, path = _cell4j(tmp, 0.8, (3, 4, 1))
        logs = []
        specs = VI.select_specs(["lorentz", "skewnorm"], [0.5])
        out = VI.run(swc, prov, cfg, recs, os.path.join(tmp, "v"), "999", specs=specs, log=logs.append)
        keys = {"node_id", "k_swc", "k_star", "line_mult", "max_rounds", "allen_2r_um", "d_iter_cell4j_um",
                "k_iter_cell4j", "variants", "notes", "png_stack", "png_planes", "seconds", "n_squares",
                "n_planes_fitted"}
        for x, r in zip(out[:2], recs[:2]):
            assert set(x) == keys, set(x) ^ keys
            assert x["notes"] == [] and len(x["variants"]) == 1 + len(specs) == 7
            assert x["variants"][0]["variant"] == VI.REFERENCE and x["variants"][0]["d_iter_um"] == r["d_iter_um"]
            assert x["line_mult"] == 3.0 and x["max_rounds"] == 5
            for v in x["variants"][1:]:
                assert v["ratio_ref"] == v["d_iter_um"] / x["variants"][0]["d_iter_um"]
            assert os.path.exists(x["png_stack"]) and os.path.exists(x["png_planes"])
        with open(os.path.join(tmp, "v", "variter_999.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 * 7 and rows[0]["variant"] == VI.REFERENCE and rows[1]["variant"] == "gauss|w fitted|none"
        assert any("redone identically on 2 of 2" in m for m in logs), logs[-1:]
        # the full 50 on one node: both figures, every text inside, the edges along the branch at the fitted edges
        full = VI.run(swc, prov, cfg, recs, os.path.join(tmp, "f"), "999", nodes=[4], log=lambda m: None)
        assert len(full[0]["variants"]) == 51
        sq = VI.NodeSquares(swc, prov, cfg, recs[1])
        specs50 = VI.select_specs()
        evs = [VI.run_iteration(sq, recs[1], None)] + [VI.run_iteration(sq, recs[1], s) for s in specs50]
        sels = [VI.selection(ev, sq.first) for ev in evs]
        rows = [VI.variant_record(s, ev, sl, sels[0]["d"]) for s, ev, sl in zip([None] + specs50, evs, sels)]
        stack, planes = VI.node_blocks(recs[1], sq, sels[0], rows, sels, specs50, [])
        assert len(stack["rows"]) == 51 and len(planes["panels"]) == 51
        for fig in (FG.variant_stack_figure(stack, "a title"), FG.variant_planes_figure(planes, "a title")):
            _texts_inside(fig)
            plt.close(fig)
        fig = FG.variant_planes_figure(planes, "a title")
        e_u = np.asarray(planes["e_u"])
        view = planes["view"]
        half = 0.42 * abs(view[1] - view[0]) / 2.0
        ax = fig.axes[1]                                  # the first variant's panel (gauss, unweighted)
        segs = [np.column_stack([ln.get_xdata(), ln.get_ydata()]) for ln in ax.get_lines() if len(ln.get_xdata()) == 2]
        E = np.asarray(planes["panels"][1]["edges_xy"])
        for P in E:
            want = np.array([P - half * e_u, P + half * e_u])
            assert any(np.allclose(sg, want, rtol=0, atol=1e-9) for sg in segs), (want, segs[:3])
        plt.close(fig)
        # both width modes: a row per kernel and width mode, no two panels in one cell
        specs2 = VI.select_specs(["lorentz"], [0.5], ["fixed", "fitted"])
        evs = [VI.run_iteration(sq, recs[1], None)] + [VI.run_iteration(sq, recs[1], s) for s in specs2]
        sels = [VI.selection(ev, sq.first) for ev in evs]
        rows = [VI.variant_record(s, ev, sl, sels[0]["d"]) for s, ev, sl in zip([None] + specs2, evs, sels)]
        _stack, planes = VI.node_blocks(recs[1], sq, sels[0], rows, sels, specs2, [])
        assert planes["kernels"] == ["gauss, w fixed", "gauss, w fitted", "lorentz, w fixed", "lorentz, w fitted"]
        cells = [(p_["kernel"], p_["weight"]) for p_ in planes["panels"][1:]]
        assert len(set(cells)) == len(cells) == 8, cells
        fig = FG.variant_planes_figure(planes, "a title")
        _texts_inside(fig)
        plt.close(fig)
        _check_cli(VI, swc_path, path, tmp)


def _check_cli(VI, swc_path, iter_json, tmp):
    import allen_image_io as aio
    import run_cell

    class _Fetcher:
        def __init__(self, cache_dir="", verbose=True):
            self.n_cache_hits, self.n_requests, self.bytes_downloaded = 0, 0, 0
    saved = (aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, VI.run)
    calls = []
    try:
        aio.HttpFetcher, aio.list_images, aio.plane_table = _Fetcher, (lambda specimen: None), (lambda images: None)
        run_cell.real_provider = lambda *x, **k: None
        VI.run = lambda *x, **k: calls.append((x, k)) or []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert VI.main(["--specimen", "999", "--iter-json", iter_json, "--out-dir", os.path.join(tmp, "cli"),
                            "--swc", swc_path, "--nodes", "4,3", "--families", "voigt", "--lambdas", "1",
                            "--half-um", "3.5", "--shift-x", "2.5", "--z0", "1.0"]) == 0
            for bad in (["--width-modes", "free"], ["--lambdas", "-1"], ["--families", "cosine"], ["--nodes", "a"]):
                try:
                    VI.main(["--specimen", "999", "--iter-json", iter_json, "--out-dir", tmp, "--swc", swc_path] + bad)
                except SystemExit as e:
                    assert e.code == 2, bad
                else:
                    raise AssertionError("%r accepted" % (bad,))
    finally:
        aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, VI.run = saved
    (x, k), = calls
    got = inspect.signature(VI.run).bind(*x, **k).arguments
    assert got["nodes"] == [4, 3] and got["half_um"] == 3.5 and got["specimen"] == "999", got
    assert got["transform"]["shift_full_px"] == (2.5, 0.0) and got["transform"]["z0_um"] == 1.0
    assert sorted(s.key for s in got["specs"]) == sorted(["gauss|w fitted|none", "gauss|w fitted|lam=1",
                                                          "voigt r=1|w fitted|none", "voigt r=1|w fitted|lam=1",
                                                          "voigt r fit|w fitted|none", "voigt r fit|w fitted|lam=1"])


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
