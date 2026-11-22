"""Spec-derived tests of Block 36, amendments 12 to 18 (the scripts of the clear-node path; tester, 2026-11-22).
Written from specs/SPEC.md before the code was read. The chain runs through the real entry points on the
synthetic cell (--synthetic) with the FAST configuration of tests/smoke/test_smoke_pilot_run.py:
clean_cell.py rank -> clear (--d-bins) -> clean --no-clean -> infer_nodes.py real -> show_focal.py and
show_clear_gallery.py.

Oracles:
- clear (am. 13, 15): the picks satisfy Block 27's rule on rank's table (the tester's own statement of the rule,
  with the table's text read back as numbers); bins [e_j-1, e_j) of Allen's diameter = twice the traced radius;
  an empty bin is skipped; `sizes` records each pick's diameter and bin; no qualifying bin returns 1;
- clean --no-clean (am. 16, 17): stopped "not_cleaned", the first stage's diameter held at Allen's;
- clean --sigma-r0-um (am. 14): the stored configuration carries with_in_focus_width's table;
- infer_nodes.py real on a not-cleaned run (am. 16): lambda = weight_kappa x allen_fwhm_um(D_A, sigma_r0) by
  hand; the CSV's columns weights, lambda_um, v_cut_lo_um, v_cut_hi_um, n_solves before seconds, the last;
- show_focal (am. 12): axis_xy = origin_xy + (v0 - v_t) y_hat, v_t by hand [1e-9 um]; d_um the fit's estimate;
- show_clear_gallery (am. 18): the CSV's columns as listed; the score of each node equals an independent
  computation of Block 40's line verdict from the node's npz (the tester's ref_features); the JSON is strict.

Pure ASCII.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from test_spec_block40_clearness import ref_features  # noqa: E402  (the tester's Block 40 reference)

FAST = {"cleaning": {"shadow_absorption": "partition_vertical", "shadow_fine_factor": 2, "soma_mu_per_um": 0.0,
                     "example_counts": [1, 0, 0, 0, 0], "max_rounds": 1, "refine_max_nfev": 40,
                     "rank_min_separation_um": 3.0},
        "inference": {"absorption": "partition_vertical", "ray_n_rho": 4, "ray_n_psi": 8,
                      "multistart_factors": [1.0], "max_nfev": 40}}
SC = WS / "scripts" / "clean_cell.py"


def _run(args, ok=True):
    r = subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(WS), capture_output=True, text=True,
                       env=dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1"))
    if ok:
        assert r.returncode == 0, (args, r.stdout[-2000:], r.stderr[-2000:])
    return r


@pytest.fixture(scope="module")
def chain():
    base = Path(tempfile.mkdtemp(prefix="b36_am12_18_"))
    cfgj = base / "fast.json"
    cfgj.write_text(json.dumps(FAST))
    _run([SC, "rank", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0", "--out", base,
          "--config-json", cfgj])
    clear = _run([SC, "clear", "--synthetic", "--specimen", "7", "--out", base, "--n", "2", "--min-soma-um", "0",
                  "--d-bins", "0.2,0.5,1.0", "--config-json", cfgj])
    cdir, idir = base / "cleaning", base / "inference"
    _run([SC, "clean", "--synthetic", "--specimen", "7", "--examples", base / "examples_clear_7.json", "--out",
          cdir, "--no-clean", "--no-own", "--config-json", cfgj])
    _run([WS / "scripts" / "infer_nodes.py", "real", "--specimen", "7", "--cleaning-dir", cdir, "--out", idir,
          "--no-own"])
    _run([WS / "scripts" / "show_focal.py", "--cleaning-dir", cdir, "--specimen", "7", "--inference-dir", idir])
    _run([WS / "scripts" / "show_clear_gallery.py", "--cleaning-dir", cdir, "--specimen", "7", "--fill-to", "2"])
    yield base, cdir, idir, cfgj, clear.stdout + clear.stderr
    shutil.rmtree(base, ignore_errors=True)


def _num(r):
    out = {}
    for k, v in r.items():
        if k == "near_end":
            out[k] = str(v).strip().lower() in ("1", "true", "yes")
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def test_clear_picks_follow_the_rule_and_the_bins(chain):
    base, cdir, idir, cfgj, log = chain
    rows = [_num(r) for r in csv.DictReader(open(base / "rank_tier1_7.csv"))]
    ex = json.loads((base / "examples_clear_7.json").read_text())
    picked = [int(i) for i in ex["example_ids"]]
    assert picked and ex["clear"]["d_bins_um"] == [0.2, 0.5, 1.0] and ex["clear"]["n_per_bin"] == 2
    ok = sorted([r for r in rows if not r["near_end"] and r["n_tier2"] == 0 and r["worst2_peak_gl"] == 0],
                key=lambda r: (r["worst1_peak_gl"], -r["own_dip_gl"], r["node_id"]))
    want, pos = [], []
    for r in ok:                                     # the synthetic cell: Allen's diameter 0.6 um everywhere
        p = np.array([r["x_um"], r["y_um"], r["z_um"]])
        if len(want) < 2 and all(np.linalg.norm(p - q) >= 3.0 for q in pos):
            want.append(int(r["node_id"]))
            pos.append(p)
    assert picked == want
    for i in picked:
        assert ex["sizes"][str(i)] == {"d_allen_um": 0.6, "bin_um": [0.5, 1.0]}
        assert ex["classes"][str(i)] == "clear"
    assert "[0.2, 0.5) um: 0 ranked nodes" in log                     # the empty bin is logged
    r = _run([SC, "clear", "--synthetic", "--specimen", "7", "--out", base, "--n", "2", "--min-soma-um", "0",
              "--d-bins", "1.0,2.0", "--config-json", cfgj], ok=False)
    assert r.returncode == 1                                          # no node in any bin
    for bad in ("0.5", "0.5,0.2", "0,1", "a,b"):
        r = _run([SC, "clear", "--synthetic", "--specimen", "7", "--out", base, "--n", "1", "--d-bins", bad,
                  "--config-json", cfgj], ok=False)
        assert r.returncode != 0, bad


def test_no_clean_record_and_infer_weights(chain):
    from allen_diameter.analysis.line_weights import allen_fwhm_um
    base, cdir, idir, cfgj, _ = chain
    rec = json.loads((cdir / "cleaning_7.json").read_text())
    assert rec["stopped"] == "not_cleaned" and rec["rounds"] == 0 and rec["pilot_diameter"] == "allen"
    inf = json.loads((idir / "inference_7.json").read_text())
    s0 = rec["config"]["renderer"]["sigma_r0_um"]
    for nid in rec["example_ids"]:
        node = rec["nodes"][str(nid)]
        assert node["pilot"]["d_hat_um"] == node["d_allen_um"] == 0.6
        r = inf["nodes"][str(nid)]
        d_a = node["d_allen_um"]
        fwhm = math.sqrt((math.sqrt(3) / 2 * d_a) ** 2 + (2 * math.sqrt(2 * math.log(2)) * s0) ** 2)
        assert abs(r["lambda_um"] - 1.25 * fwhm) <= 1e-12 and abs(allen_fwhm_um(d_a, s0) - fwhm) <= 1e-15
        assert r["weights_mode"] == "allen" and 1 <= r["n_solves"] <= 3
    with open(idir / "inference_7.csv") as f:
        header = next(csv.reader(f))
    assert header[-6:] == ["weights", "lambda_um", "v_cut_lo_um", "v_cut_hi_um", "n_solves", "seconds"]


def test_show_focal_axis_point_by_hand(chain):
    import show_focal
    base, cdir, idir, cfgj, _ = chain
    rec = json.loads((cdir / "cleaning_7.json").read_text())
    inf = json.loads((idir / "inference_7.json").read_text())["nodes"]
    rows = list(csv.DictReader(open(cdir / "focal_7.csv")))
    assert [int(r["node_id"]) for r in rows] == [int(i) for i in rec["example_ids"]]
    for nid in rec["example_ids"]:
        node = rec["nodes"][str(nid)]
        with np.load(cdir / ("cleaning_7_node%d.npz" % nid)) as z:
            panel, row = show_focal.focal_panel(node, z, inf[str(nid)])
            th, pil = float(node["pilot"]["theta_rad"]), node["pilot"]
            y_hat = np.array([-math.sin(th), math.cos(th)])
            t = np.asarray(z["node_xyz"], float)
            v_t = (pil["cx_um"] - t[0]) * y_hat[0] + (pil["cy_um"] - t[1]) * y_hat[1]
            want = np.asarray(z["origin_xy"], float) + (float(inf[str(nid)]["theta"][2]) - v_t) * np.asarray(
                z["y_hat"], float)
        assert np.max(np.abs(panel["axis_xy"] - want)) <= 1e-9
        assert panel["d_um"] == float(inf[str(nid)]["theta"][0]) and row["drawn"] == "fit"


def test_gallery_columns_scores_and_strict_json(chain):
    from allen_diameter import config as C
    base, cdir, idir, cfgj, _ = chain
    gdir = cdir / "gallery"
    with open(gdir / "gallery_7.csv") as f:
        header = next(csv.reader(f))
    assert header == ["node_id", "d_allen_um", "bin_lo_um", "bin_hi_um", "clean", "score", "reasons", "k_star",
                      "depth_gl", "depth_sd", "sigma_gl", "fwhm_kstar_um", "fwhm_over_allen", "fwhm_min_um",
                      "k_fwhm_min", "outside_focus", "skew_focus", "outside_along", "skew_along", "width_range",
                      "depth_range", "centre_range", "shown", "figure"]
    rows = {int(r["node_id"]): r for r in csv.DictReader(open(gdir / "gallery_7.csv"))}
    rec = json.loads((cdir / "cleaning_7.json").read_text())
    q = C.default_config().clearness
    for nid in rec["example_ids"]:
        node = rec["nodes"][str(nid)]
        with np.load(cdir / ("cleaning_7_node%d.npz" % nid)) as z:
            ks, B, prof, v = z["ks"].astype(int), z["B_gl"], z["profiles_before"], z["v"]
        j = int(np.argmin(np.abs(ks - int(node["pilot"]["k_star"]))))
        E, S = [], []
        for i in range(ks.size):
            if abs(ks[i] - ks[j]) <= q.focus_half_planes:
                f = ref_features(gaussian_filter1d(B[i] - prof[i], q.smooth_samples, mode="nearest"), v, 0.0,
                                 q.window_um, q.skew_levels, q.outside_fwhm)
                E.append(f["outside"])
                S.append(f["skew"])
        score = max(max(E) / q.outside_max, max(S) / q.skew_max)
        assert abs(float(rows[nid]["score"]) - score) <= 1e-12
    text = (gdir / "gallery_7.json").read_text()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("non-strict JSON: " + c)))


def test_sigma_r0_option_stores_the_reanchored_table(chain):
    from allen_diameter import config as C
    from allen_diameter.model.kernel import with_in_focus_width
    base, cdir, idir, cfgj, _ = chain
    out = base / "cleaning_s020"
    _run([SC, "clean", "--synthetic", "--specimen", "7", "--examples", base / "examples_clear_7.json", "--out", out,
          "--no-clean", "--no-own", "--sigma-r0-um", "0.20", "--config-json", cfgj])
    ren = json.loads((out / "cleaning_7.json").read_text())["config"]["renderer"]
    want = with_in_focus_width(C.default_config().renderer, 0.20)
    assert ren["sigma_r0_um"] == 0.20
    assert np.allclose(ren["kernel_table_sigma_um"], want.kernel_table_sigma_um, rtol=0, atol=1e-15)


def test_gallery_choose_rule_by_hand():
    """Amendment 18: the judged-clean nodes by the lower edge of their bin, then by increasing score, at most
    max_per_bin per bin (0: all); then, up to fill_to, the rejected nodes by increasing score; --nodes: exactly
    those (the ones that were screened). Bins are [lo, hi)."""
    import show_clear_gallery as G
    edges = G.parse_bins("0.4,0.6,1.0")
    assert edges == [0.4, 0.6, 1.0]
    assert G.bin_of(0.6, edges) == (0.6, 1.0) and G.bin_of(0.4, edges) == (0.4, 0.6)
    assert all(math.isnan(x) for x in G.bin_of(1.0, edges)) and all(math.isnan(x) for x in G.bin_of(0.5, None))
    rows = []
    for nid, d, clean, score in ((1, 0.8, True, 0.5), (2, 0.5, True, 0.9), (3, 0.5, True, 0.2), (4, 0.8, True, 0.1),
                                 (5, 0.5, False, 1.5), (6, 0.8, False, 1.2), (7, 2.0, True, 0.3), (8, 0.5, True, 0.4)):
        lo, hi = G.bin_of(d, edges)
        rows.append(dict(node_id=nid, clean=clean, score=score, bin_lo_um=lo, bin_hi_um=hi))
    # bins by lower edge: [0.4, 0.6): 3 (0.2), 8 (0.4), 2 (0.9); [0.6, 1.0): 4 (0.1), 1 (0.5); no bin last: 7
    assert G.choose(rows, 0, 0) == [3, 8, 2, 4, 1, 7]
    assert G.choose(rows, 2, 0) == [3, 8, 4, 1, 7]
    assert G.choose(rows, 1, 6) == [3, 4, 7, 6, 5]           # 3 shown, then the rejected by score up to 6 ... 5 in all
    assert G.choose(rows, 1, 4) == [3, 4, 7, 6]
    assert G.choose(rows, 0, 0, nodes=[6, 99, 1]) == [6, 1]
    for bad in ("0.4", "0.6,0.4", "0,1", "1,nan"):
        with pytest.raises(ValueError):
            G.parse_bins(bad)
