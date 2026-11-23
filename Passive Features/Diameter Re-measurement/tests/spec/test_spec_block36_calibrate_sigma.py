"""Spec-derived tests of Block 36, amendment 19 (scripts/calibrate_sigma.py) and its boundaries with Blocks 42
and 43 (tester, 2026-11-22). Written from specs/SPEC.md before the code was read.

Oracles (SPEC Block 42's data flow and Block 36 amendment 19):
- each node's fit uses its own record entry's sigma_gl ("the record's top-level sigma_gl is the median over
  nodes and is not used") and the planes k* +- half_planes of its recorded profiles;
- "a node enters the summary when it converged with neither sigma0 nor c at a bound"; the summary is
  Block 42's summarize, recomputed here by hand;
- sigma_<S>.csv, sigma_<S>.json, sigma_node<id>.png, sigma_summary_<S>.png are written;
- the result does not depend on the pool (one worker or two).
The preparation is made by the real entry points (clean_cell.py rank, clear, clean --no-clean --synthetic) on
the FAST configuration of tests/smoke/test_smoke_pilot_run.py with Cell 23A's thick-node settings.

Pure ASCII.
"""
from __future__ import annotations

import copy
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

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FAST3 = {"cleaning": {"shadow_absorption": "partition_vertical", "shadow_fine_factor": 2, "soma_mu_per_um": 0.0,
                      "example_counts": [1, 0, 0, 0, 0], "max_rounds": 1, "refine_max_nfev": 40,
                      "rank_min_separation_um": 3.0, "plane_half_range": 2},
         "measure": {"profile_half_um": 4.5},
         "inference": {"absorption": "partition_vertical", "ray_n_rho": 4, "ray_n_psi": 8,
                       "multistart_factors": [1.0], "max_nfev": 40}}


def _run(args):
    r = subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(WS), capture_output=True, text=True,
                       env=dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, (args, r.stdout[-2000:], r.stderr[-2000:])
    return r.stdout


@pytest.fixture(scope="module")
def prep():
    base = Path(tempfile.mkdtemp(prefix="b36_sigma_"))
    cfgj = base / "fast3.json"
    cfgj.write_text(json.dumps(FAST3))
    sc = WS / "scripts" / "clean_cell.py"
    _run([sc, "rank", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0", "--out", base,
          "--config-json", cfgj])
    _run([sc, "clear", "--synthetic", "--specimen", "7", "--out", base, "--n", "2", "--min-soma-um", "0",
          "--config-json", cfgj])
    cdir = base / "cleaning"
    _run([sc, "clean", "--synthetic", "--specimen", "7", "--examples", base / "examples_clear_7.json", "--out", cdir,
          "--no-clean", "--no-own", "--config-json", cfgj])
    yield base, cdir, cfgj
    shutil.rmtree(base, ignore_errors=True)


def _fit_row(cdir, nid, out):
    import calibrate_sigma
    os.makedirs(out, exist_ok=True)
    row, _extra = calibrate_sigma.fit_one(str(cdir), "7", int(nid), None, str(out))
    return row


def test_entry_point_outputs_and_summary_by_hand(prep):
    base, cdir, _ = prep
    import calibrate_sigma
    out = base / "sigma"
    assert calibrate_sigma.main(["--cleaning-dir", str(cdir), "--specimen", "7", "--out", str(out),
                                 "--workers", "1"]) == 0
    rec = json.loads((cdir / "cleaning_7.json").read_text())
    ids = [int(i) for i in rec["example_ids"]]
    for name in ["sigma_7.csv", "sigma_7.json", "sigma_summary_7.png"] + ["sigma_node%d.png" % i for i in ids]:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    js = json.loads((out / "sigma_7.json").read_text())
    rows = list(csv.DictReader(open(out / "sigma_7.csv")))
    assert [int(r["node_id"]) for r in rows] == ids == js["nodes"]
    for f in js["fits"]:
        assert f["n_planes"] == 5 and f["k_star"] == rec["nodes"][str(f["node_id"])]["pilot"]["k_star"]
        assert all(math.isfinite(f[k]) for k in ("d_um", "mu_per_um", "v0_um", "z_ax_um", "sigma0_um",
                                                 "growth_scale", "se_sigma0_um"))
    ok = [f for f in js["fits"] if f["status"] == "converged" and not ({"sigma0_um", "growth_scale"}
                                                                         & set(f["at_bound"].split()))]
    x = np.array([f["sigma0_um"] for f in ok])
    w = 1.0 / np.array([f["se_sigma0_um"] for f in ok]) ** 2
    s = js["summary"]
    assert s["n"] == len(ok) >= 2
    mean = float(np.sum(w * x) / np.sum(w))
    assert abs(s["ivw_mean_um"] - mean) <= 1e-12 and abs(s["median_um"] - np.median(x)) <= 1e-15
    assert abs(s["ivw_se_um"] - 1.0 / math.sqrt(np.sum(w))) <= 1e-12
    assert abs(s["chi2_cons"] - float(np.sum(w * (x - mean) ** 2))) <= 1e-9 and s["dof"] == len(ok) - 1


def test_the_nodes_own_sigma_is_used_not_the_median(prep):
    """The top-level sigma_gl changed: nothing changes. The node's own sigma_gl doubled: the solver sees the
    residuals halved, the same minimiser; chi2 quarters, SE doubles, SE_s stays (Block 42's formulas)."""
    base, cdir, _ = prep
    rec = json.loads((cdir / "cleaning_7.json").read_text())
    nid = int(rec["example_ids"][0])
    ref = _fit_row(cdir, nid, base / "s_ref")
    for variant in ("top", "node"):
        vdir = base / ("c_" + variant)
        if vdir.exists():
            shutil.rmtree(vdir)
        shutil.copytree(cdir, vdir)
        r2 = copy.deepcopy(rec)
        if variant == "top":
            r2["sigma_gl"] = 99.0 * float(rec.get("sigma_gl", 1.0))
        else:
            r2["nodes"][str(nid)]["sigma_gl"] = 2.0 * float(rec["nodes"][str(nid)]["sigma_gl"])
        (vdir / "cleaning_7.json").write_text(json.dumps(r2))
        got = _fit_row(vdir, nid, base / ("s_" + variant))
        if variant == "top":
            for k in ("d_um", "sigma0_um", "growth_scale", "se_sigma0_um", "chi2", "n_eval"):
                assert got[k] == ref[k], (k, got[k], ref[k])
        else:
            assert abs(got["d_um"] - ref["d_um"]) <= 1e-6 and abs(got["sigma0_um"] - ref["sigma0_um"]) <= 1e-6
            assert abs(got["chi2"] / ref["chi2"] - 0.25) <= 1e-4
            assert abs(got["se_sigma0_um"] / ref["se_sigma0_um"] - 1.0) <= 1e-3     # SE_s: sigma cancels


def test_a_node_at_a_sigma0_bound_leaves_the_summary(prep):
    base, cdir, cfgj = prep
    import calibrate_sigma
    tight = json.loads(cfgj.read_text())
    tight["sigma_calibration"] = {"sigma0_bounds_um": [0.02, 0.06], "sigma0_start_um": 0.05}
    tj = base / "tight.json"
    tj.write_text(json.dumps(tight))
    out = base / "sigma_tight"
    rec = json.loads((cdir / "cleaning_7.json").read_text())
    nid = int(rec["example_ids"][0])
    assert calibrate_sigma.main(["--cleaning-dir", str(cdir), "--specimen", "7", "--out", str(out), "--nodes",
                                 str(nid), "--workers", "1", "--config-json", str(tj)]) == 0
    js = json.loads((out / "sigma_7.json").read_text())
    f = js["fits"][0]
    assert "sigma0_um" in f["at_bound"].split() and f["ok"] is False
    assert js["summary"]["n"] == 0


def _strict(text):
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("non-strict JSON: " + c)))


def test_pool_and_repeat_give_the_same_fits(prep):
    """Determinism of the entry point and its independence of the pool (one worker, two workers)."""
    base, cdir, _ = prep
    import calibrate_sigma
    outs = []
    for tag, w in (("r1", "1"), ("r2", "2")):
        out = base / ("sigma_" + tag)
        assert calibrate_sigma.main(["--cleaning-dir", str(cdir), "--specimen", "7", "--out", str(out),
                                     "--workers", w]) == 0
        js = json.loads((out / "sigma_7.json").read_text())
        outs.append(([{k: v for k, v in f.items() if k != "seconds"} for f in js["fits"]], js["summary"]))
    assert outs[0] == outs[1]


@pytest.mark.xfail(strict=True, reason="2026-11-22 report F3: sigma_<S>.json writes NaN as the bare token NaN "
                                       "(json.dump's default), unlike the gallery's and the cleaning's strict JSON")
def test_sigma_json_is_strict_when_the_summary_is_empty(prep):
    base, cdir, cfgj = prep
    import calibrate_sigma
    tight = json.loads(cfgj.read_text())
    tight["sigma_calibration"] = {"sigma0_bounds_um": [0.02, 0.06], "sigma0_start_um": 0.05}
    tj = base / "tight2.json"
    tj.write_text(json.dumps(tight))
    out = base / "sigma_tight2"
    rec = json.loads((cdir / "cleaning_7.json").read_text())
    assert calibrate_sigma.main(["--cleaning-dir", str(cdir), "--specimen", "7", "--out", str(out), "--nodes",
                                 str(int(rec["example_ids"][0])), "--workers", "1", "--config-json", str(tj)]) == 0
    _strict((out / "sigma_7.json").read_text())


@pytest.mark.xfail(strict=True, reason="2026-11-22 report F4: one node that raises stops calibrate_sigma.py and "
                                       "no CSV or JSON is written for the nodes already fitted")
def test_one_bad_node_does_not_lose_the_others(prep):
    base, cdir, _ = prep
    import calibrate_sigma
    bad = base / "c_bad"
    if bad.exists():
        shutil.rmtree(bad)
    shutil.copytree(cdir, bad)
    rec = json.loads((bad / "cleaning_7.json").read_text())
    ids = [int(i) for i in rec["example_ids"]]
    rec["nodes"][str(ids[-1])]["sigma_gl"] = 0.0            # the last node's record cannot be fitted
    (bad / "cleaning_7.json").write_text(json.dumps(rec))
    out = bad / "sigma"
    try:
        calibrate_sigma.main(["--cleaning-dir", str(bad), "--specimen", "7", "--out", str(out), "--workers", "1"])
    except Exception:
        pass
    assert (out / "sigma_7.csv").exists() and (out / "sigma_7.json").exists()
