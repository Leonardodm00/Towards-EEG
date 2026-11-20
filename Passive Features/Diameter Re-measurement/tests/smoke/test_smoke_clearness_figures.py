"""Smoke test for the clearness figures and the gallery's entry point -- Block 41 and Block 36, amendment 18,
in specs/SPEC.md (D-194 [orchestrator]).

Checks
    test_known_answer   figure_node draws what it is given: the profile lines' data equal the inputs, the
                        half-depth bar's ends are the crossings given, Allen's bar is d wide, one image and one
                        profile panel per plane on one grey-level axis; figure_sheet has one panel per node
    test_reference      scripts/show_clear_gallery.py on files written by hand (a clean band, a band with a
                        parallel neighbour, a band with a blob beside the line): the table's verdicts and
                        numbers equal clearness.screen_node called directly, by default (D-198: the blob is
                        judged clean) and with --judge-along (the blob rejected); what is shown follows the rule
    test_convergence    skipped (no discretisation)
    test_invariants     choose(): the bins' order, the cap per bin, the fill with rejected nodes, an explicit
                        list; bin_of() and parse_bins()
    test_contract       the files written, the CSV's columns, the JSON's keys, the PNGs not empty; the inputs
                        unchanged
    test_determinism    the script twice: the same CSV
    test_edge_cases     the refusals of the figures; a folder without nodes returns 1; a node without its npz
                        is skipped

Run (from the code folder "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_clearness_figures.py
    python3 -m pytest tests/smoke/test_smoke_clearness_figures.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_smoke_clearness as T  # noqa: E402  (the synthetic band and its geometry)
from allen_diameter.analysis import clearness as CL  # noqa: E402
from allen_diameter.analysis import profiles  # noqa: E402
from allen_diameter.config import ClearnessConfig  # noqa: E402
from allen_diameter.plotting import clearness_figures as CF  # noqa: E402

_spec = importlib.util.spec_from_file_location("show_clear_gallery", str(WS / "scripts" / "show_clear_gallery.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

SEED = 20261119
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
_SAVED_ARTIFACTS = []
CFG = ClearnessConfig()
SPECIMEN = "7"
BLOCKS = {
    11: (0.5, lambda: T.band_block()),                                                       # clean
    12: (0.5, lambda: T.band_block(extra=lambda u, v: 30.0 * np.exp(-0.5 * ((v + 1.4) / 0.35) ** 2))),   # neighbour
    13: (1.0, lambda: T.band_block(extra=lambda u, v: 50.0 * np.exp(-0.5 * (((u - 1.5) / 0.4) ** 2
                                                                           + ((v - 0.6) / 0.4) ** 2)))),  # blob
    14: (1.0, lambda: T.band_block(width=lambda u: 0.5 + 0.0 * u)),                              # clean, wider
}


def _raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised by %s" % (exc.__name__, getattr(func, "__name__", func)))


def write_run(folder, ids=tuple(BLOCKS), sigma=1.0):
    """A preparation's files as clean_cell.py clean --no-clean writes them, for the nodes of BLOCKS."""
    origin, y_hat, e_u = T._geometry()
    frame = T.SimpleNamespace(left=T.LEFT, top=T.TOP, res_um_px=T.P_UM)
    nodes = {}
    for nid in ids:
        d_allen, make = BLOCKS[nid]
        block = make()
        line = np.stack([profiles.sample_profile(block[i], frame, origin, y_hat, e_u, T.V, T.N_AVG, T.STEP)
                         for i in range(T.KS.size)])
        np.savez(os.path.join(folder, "cleaning_%s_node%d.npz" % (SPECIMEN, nid)), block=block, ks=T.KS,
                 B_gl=np.full(T.KS.size, T.B_GL), v=T.V, profiles_before=line, profiles_after=line, origin_xy=origin,
                 y_hat=y_hat, e_u=e_u, n_avg=T.N_AVG, step=T.STEP, left=T.LEFT, top=T.TOP, res0_um=T.P_UM)
        nodes[str(nid)] = dict(node_id=nid, sigma_gl=sigma, d_allen_um=d_allen,
                               pilot=dict(k_star=T.K_STAR, mu_hat_per_um=0.7, phi_rad=0.1))
    with open(os.path.join(folder, "cleaning_%s.json" % SPECIMEN), "w") as fh:
        json.dump(dict(example_ids=list(ids), nodes=nodes, dropped={}), fh)


def panel(block=None):
    origin, y_hat, e_u = T._geometry()
    block = T.band_block() if block is None else block
    r = T.screen(block)
    frame = T.SimpleNamespace(left=T.LEFT, top=T.TOP, res_um_px=T.P_UM)
    line = np.stack([profiles.sample_profile(block[i], frame, origin, y_hat, e_u, T.V, T.N_AVG, T.STEP)
                     for i in range(T.KS.size)])
    p = T.P_UM
    extent = ((T.LEFT - 0.5) * p, (T.LEFT + block.shape[2] - 0.5) * p, (T.TOP + block.shape[1] - 0.5) * p,
              (T.TOP - 0.5) * p)
    return dict(images=block, extent=extent, ks=T.KS, k_index=r.k_index, origin_xy=origin, y_hat=y_hat, e_u=e_u,
                v_um=T.V, profiles_gl=line, B_gl=np.full(T.KS.size, T.B_GL), u_um=r.u_um, strips_gl=r.strips_gl,
                focus=r.focus, d_allen_um=0.6, x_lo_um=np.linspace(-0.9, -0.4, T.KS.size),
                x_hi_um=np.linspace(0.4, 0.9, T.KS.size), title="node 11", text="JUDGED CLEAN")


def _by_gid(fig, gid):
    return [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_gid() == gid]


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    import matplotlib.pyplot as plt
    pn = panel()
    keep = {k: np.array(v, copy=True) for k, v in pn.items() if isinstance(v, np.ndarray)}
    fig = CF.figure_node(pn, zoom_um=5.0)
    n_k = T.KS.size
    lines = _by_gid(fig, "profile")
    assert len(lines) == n_k
    for i, ln in enumerate(lines):
        assert_array_equal(ln.get_xdata(), T.V)
        assert_array_equal(ln.get_ydata(), pn["profiles_gl"][i])
    bars = _by_gid(fig, "fwhm")
    assert len(bars) == n_k
    for i, b in enumerate(bars):
        assert_allclose(b.get_xdata(), [pn["x_lo_um"][i], pn["x_hi_um"][i]], atol=1e-12)
    allen = [ln for ln in _by_gid(fig, "allen") if len(ln.get_xdata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
             and ln.axes.get_xlabel() == "v (um)"]
    assert len(allen) == n_k and all(abs((a.get_xdata()[1] - a.get_xdata()[0]) - 0.6) < 1e-12 for a in allen)
    images = [ax for ax in fig.axes if ax.images]
    assert len(images) == n_k + 1                                       # the sharpest plane large, then every plane
    assert len({im.get_clim() for ax in images for im in ax.images}) == 1       # one grey scale
    prof_axes = [ln.axes for ln in lines]
    assert len({ax.get_ylim() for ax in prof_axes}) == 1                # one grey-level axis
    assert len([ln for ln in fig.axes[1].get_lines() if (ln.get_gid() or "").startswith("strip")]) == pn["u_um"].size
    assert len([ln for ln in fig.axes[2].get_lines() if (ln.get_gid() or "").startswith("plane")]) == n_k
    for k, v in keep.items():
        assert_array_equal(pn[k], v)
    plt.close(fig)
    sheet = CF.figure_sheet([pn, pn, pn], ncols=2)
    assert len([ax for ax in sheet.axes if ax.images]) == 3
    plt.close(sheet)


def _rows(folder):
    with open(os.path.join(folder, "gallery_7.csv")) as fh:
        return {int(r["node_id"]): r for r in csv.DictReader(fh)}


def _same_as_direct(rows, cfg):
    """Every row of the table equals clearness.screen_node called directly with cfg."""
    for nid, (d_allen, make) in BLOCKS.items():
        ref = T.screen(make(), cfg=cfg)
        r = rows[nid]
        assert (r["clean"] == "True") == ref.clean, (nid, r["clean"], ref.reasons)
        assert_allclose(float(r["score"]), ref.score, rtol=1e-12)
        assert r["reasons"] == " ".join(ref.reasons)
        assert_allclose(float(r["fwhm_kstar_um"]), ref.line["fwhm_um"][ref.k_index], rtol=1e-12)
        assert_allclose(float(r["fwhm_over_allen"]), ref.line["fwhm_um"][ref.k_index] / d_allen, rtol=1e-12)
        assert_allclose(float(r["width_range"]), ref.width_range, rtol=1e-12, atol=1e-15)
        assert int(r["k_star"]) == T.K_STAR


def test_reference():
    tmp = tempfile.mkdtemp(prefix="gallery_")
    try:
        write_run(tmp)
        base = ["--cleaning-dir", tmp, "--specimen", SPECIMEN, "--d-bins", "0.4,0.8,5.0"]
        # the default verdict (D-198): the measuring line alone. The neighbour on the line (12) is rejected; the
        # blob beside the line (13) is seen only by the lines along the dendrite, so it is judged clean.
        assert G.main(base) == 0
        rows = _rows(os.path.join(tmp, "gallery"))
        assert set(rows) == set(BLOCKS)
        _same_as_direct(rows, T.CFG)
        assert [n for n in BLOCKS if rows[n]["clean"] == "True"] == [11, 13, 14]
        assert "outside" in rows[12]["reasons"]
        rec = json.load(open(os.path.join(tmp, "gallery", "gallery_7.json")))
        in_bin = sorted((13, 14), key=lambda i: float(rows[i]["score"]))      # one bin: the lower score first
        assert rec["shown"] == [11] + in_bin and rec["judged_clean"] == [11] + in_bin
        assert rows[11]["figure"] == "gallery_node11.png" and rows[12]["figure"] == "" and rows[12]["shown"] == "False"
        assert set(rec["nodes"]["11"]["ratios"]) == set(CL.LINE_CRITERIA)
        assert rec["clearness"]["judge_along"] is False
        # --judge-along (D-194.3, D-194.4): the blob is rejected too; the table equals the direct call with it
        assert G.main(base + ["--judge-along", "--out", os.path.join(tmp, "ja")]) == 0
        rows_j = _rows(os.path.join(tmp, "ja"))
        _same_as_direct(rows_j, T.JUDGE)
        assert [n for n in BLOCKS if rows_j[n]["clean"] == "True"] == [11, 14]
        rec_j = json.load(open(os.path.join(tmp, "ja", "gallery_7.json")))
        assert rec_j["shown"] == [11, 14] and rec_j["clearness"]["judge_along"] is True
        assert set(rec_j["nodes"]["13"]["ratios"]) == set(CL.CRITERIA)
        # with a fill, the best-scoring rejected nodes follow, labelled in their file names
        assert G.main(base + ["--judge-along", "--fill-to", "3", "--out", os.path.join(tmp, "g2")]) == 0
        rec2 = json.load(open(os.path.join(tmp, "g2", "gallery_7.json")))
        best_rejected = min((12, 13), key=lambda i: float(rows_j[i]["score"]))
        assert rec2["shown"] == [11, 14, best_rejected]
        assert os.path.exists(os.path.join(tmp, "g2", "gallery_rejected_node%d.png" % best_rejected))
        # a threshold given on the command line is the one applied
        assert G.main(["--cleaning-dir", tmp, "--specimen", SPECIMEN, "--outside-max", "0.9", "--skew-max", "0.9",
                       "--out", os.path.join(tmp, "g3")]) == 0
        rec3 = json.load(open(os.path.join(tmp, "g3", "gallery_7.json")))
        assert rec3["clearness"]["outside_max"] == 0.9 and 12 in rec3["judged_clean"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("figures and an entry point: no discretisation")


def test_invariants():
    nan = float("nan")
    rows = [dict(node_id=1, clean=True, score=0.5, bin_lo_um=0.8), dict(node_id=2, clean=True, score=0.2, bin_lo_um=0.8),
            dict(node_id=3, clean=True, score=0.9, bin_lo_um=0.4), dict(node_id=4, clean=False, score=1.4, bin_lo_um=0.4),
            dict(node_id=5, clean=False, score=1.1, bin_lo_um=0.8), dict(node_id=6, clean=True, score=0.3, bin_lo_um=0.8),
            dict(node_id=7, clean=False, score=float("inf"), bin_lo_um=nan)]
    assert G.choose(rows, 0, 0) == [3, 2, 6, 1]                 # the thinner bin first, then the lowest score
    assert G.choose(rows, 2, 0) == [3, 2, 6]                    # two per bin at most
    assert G.choose(rows, 1, 4) == [3, 2, 5, 4]                 # filled with the best-scoring rejected
    assert G.choose(rows, 1, 9) == [3, 2, 5, 4, 7]
    assert G.choose(rows, 4, 0, nodes=[7, 99, 1]) == [7, 1]     # an explicit list, the unknown id left out
    assert G.bin_of(0.45, [0.4, 0.45, 0.6]) == (0.45, 0.6) and G.bin_of(0.4, [0.4, 0.45]) == (0.4, 0.45)
    assert all(math.isnan(x) for x in G.bin_of(0.7, [0.4, 0.6])) and all(math.isnan(x) for x in G.bin_of(0.5, None))
    assert G.parse_bins("") is None and G.parse_bins("0.4, 0.8,5") == [0.4, 0.8, 5.0]
    for bad in ("0.4", "0.8,0.4", "0,1", "a,b"):
        _raises(ValueError, G.parse_bins, bad)


def test_contract():
    tmp = tempfile.mkdtemp(prefix="gallery_")
    try:
        write_run(tmp, ids=(11, 12))
        before = {f: open(os.path.join(tmp, f), "rb").read() for f in os.listdir(tmp)}
        assert G.main(["--cleaning-dir", tmp, "--specimen", SPECIMEN, "--fill-to", "2"]) == 0
        for f, data in before.items():
            assert open(os.path.join(tmp, f), "rb").read() == data          # the run's files unchanged
        out = os.path.join(tmp, "gallery")
        names = sorted(os.listdir(out))
        assert names == ["gallery_7.csv", "gallery_7.json", "gallery_node11.png", "gallery_rejected_node12.png",
                         "gallery_sheet_7.png"], names
        assert all(os.path.getsize(os.path.join(out, n)) > 2000 for n in names if n.endswith(".png"))
        with open(os.path.join(out, "gallery_7.csv")) as fh:
            assert tuple(csv.DictReader(fh).fieldnames) == G.COLUMNS
        rec = json.load(open(os.path.join(out, "gallery_7.json")))
        assert {"clearness", "shown", "judged_clean", "nodes", "skipped", "d_bins_um"} <= set(rec)
        node = rec["nodes"]["11"]
        assert set(node["ratios"]) == set(CL.LINE_CRITERIA) and len(node["line"]["fwhm_um"]) == T.KS.size
        assert len(node["along"]["fwhm_um"]) == len(node["u_um"]) == 9 and sum(node["focus"]) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_determinism():
    tmp = tempfile.mkdtemp(prefix="gallery_")
    try:
        write_run(tmp, ids=(11, 13))
        texts = []
        for name in ("a", "b"):
            assert G.main(["--cleaning-dir", tmp, "--specimen", SPECIMEN, "--out", os.path.join(tmp, name)]) == 0
            texts.append(open(os.path.join(tmp, name, "gallery_7.csv")).read())
        assert texts[0] == texts[1]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_cases():
    pn = panel()
    _raises(ValueError, CF.figure_node, [pn])
    _raises(ValueError, CF.figure_node, {k: v for k, v in pn.items() if k != "strips_gl"})
    _raises(ValueError, CF.figure_node, dict(pn, profiles_gl=pn["profiles_gl"][:, :5]))
    _raises(ValueError, CF.figure_node, dict(pn, d_allen_um=-1.0))
    _raises(ValueError, CF.figure_node, dict(pn, k_index=99))
    _raises(ValueError, CF.figure_node, dict(pn, focus=np.ones(3, dtype=bool)))
    _raises(ValueError, CF.figure_node, pn, zoom_um=0.0)
    _raises(ValueError, CF.figure_sheet, [])
    _raises(ValueError, CF.figure_sheet, [pn], ncols=0)
    import matplotlib.pyplot as plt
    bare = {k: pn[k] for k in CF.NODE_REQUIRED}                 # the optional keys left out; the whole patch
    plt.close(CF.figure_node(bare, zoom_um=None))
    tmp = tempfile.mkdtemp(prefix="gallery_")
    try:
        write_run(tmp, ids=(11, 12))
        os.remove(os.path.join(tmp, "cleaning_7_node12.npz"))   # a node without its arrays is skipped
        assert G.main(["--cleaning-dir", tmp, "--specimen", SPECIMEN]) == 0
        rec = json.load(open(os.path.join(tmp, "gallery", "gallery_7.json")))
        assert list(rec["nodes"]) == ["11"] and "12" in rec["skipped"]
        os.remove(os.path.join(tmp, "cleaning_7_node11.npz"))   # none left: 1
        assert G.main(["--cleaning-dir", tmp, "--specimen", SPECIMEN, "--out", os.path.join(tmp, "none")]) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ main ---

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
