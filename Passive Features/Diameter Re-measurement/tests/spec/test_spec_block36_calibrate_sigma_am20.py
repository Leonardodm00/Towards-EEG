"""Spec-derived tests of scripts/calibrate_sigma.py under Block 36 amendment 20 (the settlements D-212 of the final
check of wave 4: F3, F4, F5, S3, S5), with Blocks 42 and 43 behind it. Tester, 2026-11-22 (incremental run at
97f5104).

The preparation is the section 6 minimal case: `clean_cell.py rank`, `clear` and `clean --no-clean` on the synthetic
cell (--synthetic) with the smoke tests' FAST configuration, the line +-4.5 um and the planes k* +- 2 (Cell 23A's two
changes), three example nodes written by hand into the examples file. Expected values are computed here: the summary
by hand from the rows, strict JSON by a parser that refuses NaN and Infinity.

    python3 -m pytest -q -p no:cacheprovider tests/spec/test_spec_block36_calibrate_sigma_am20.py

Pure ASCII.
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FAST = {"cleaning": {"shadow_absorption": "partition_vertical", "shadow_fine_factor": 2, "soma_mu_per_um": 0.0,
                     "example_counts": [1, 0, 0, 0, 0], "max_rounds": 1, "refine_max_nfev": 40},
        "inference": {"absorption": "partition_vertical", "ray_n_rho": 4, "ray_n_psi": 8,
                      "multistart_factors": [1.0], "max_nfev": 40}}
ENV = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def _run(args, cwd=WS, timeout=1800):
    res = subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(cwd), env=ENV, capture_output=True,
                         text=True, timeout=timeout)
    return res.returncode, res.stdout + res.stderr


def strict_load(path):
    def refuse(name):
        raise ValueError("non-standard JSON constant %s" % name)
    with open(path) as f:
        return json.loads(f.read(), parse_constant=refuse)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    base = tmp_path_factory.mktemp("cs_prep")
    fast = base / "fast.json"
    fast.write_text(json.dumps(FAST))
    thick = dict(json.loads(json.dumps(FAST)))
    thick["cleaning"]["plane_half_range"] = 2
    thick["measure"] = {"profile_half_um": 4.5}
    (base / "thick.json").write_text(json.dumps(thick))
    sc = WS / "scripts" / "clean_cell.py"
    rc, out = _run([sc, "rank", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0", "--out", base,
                    "--config-json", fast])
    assert rc == 0, out
    rc, out = _run([sc, "clear", "--synthetic", "--specimen", "7", "--out", base, "--n", "3", "--min-soma-um", "0",
                    "--config-json", fast])
    assert rc == 0, out
    ex = json.load(open(base / "examples_clear_7.json"))
    ex["example_ids"] = [5, 8, 11]                       # three nodes that pass the clear rule (not near_end, no tier 2)
    ex["classes"] = {str(i): "clear" for i in ex["example_ids"]}
    (base / "examples_three_7.json").write_text(json.dumps(ex))
    rc, out = _run([sc, "clean", "--synthetic", "--no-clean", "--specimen", "7", "--examples",
                    base / "examples_three_7.json", "--out", base / "thick", "--config-json", base / "thick.json"])
    assert rc == 0, out
    return base / "thick"


def _main_in_process(argv, monkeypatch, spies=None):
    """calibrate_sigma.main(argv) in this process (workers 1), with the stdout captured and the spies applied."""
    import calibrate_sigma as CS
    for target, name, value in (spies or ()):
        monkeypatch.setattr(target, name, value)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CS.main([str(a) for a in argv])
    return rc, buf.getvalue()


def test_outputs_written_as_it_goes_strict_json_and_titles(prep, tmp_path, monkeypatch):
    import calibrate_sigma as CS
    from allen_diameter.plotting import sigma_figures as SFIG
    writes, titles = [], []
    real_write, real_fig = CS.write_outputs, SFIG.figure_node_fit

    def spy_write(out, a, ids, rows, scfg, summary, t0):
        real_write(out, a, ids, rows, scfg, summary, t0)
        rec = strict_load(os.path.join(out, "sigma_%s.json" % a.specimen))     # strict JSON at every write (F3)
        writes.append((rec["complete"], len(rec["fits"]), len(read_csv(os.path.join(out, "sigma_%s.csv" % a.specimen)))))

    def spy_fig(node, *args, **kw):
        titles.append(node["title"])
        return real_fig(node, *args, **kw)

    out = tmp_path / "sigma"
    rc, log = _main_in_process(["--cleaning-dir", prep, "--specimen", "7", "--out", out, "--workers", "1",
                                "--nodes", "5,8"], monkeypatch,
                               [(CS, "write_outputs", spy_write), (SFIG, "figure_node_fit", spy_fig)])
    assert rc == 0, log
    # (iv) rewritten after every node, `complete` false until the end
    assert writes == [(False, 1, 1), (False, 2, 2), (True, 2, 2)], writes
    # (ii) the forward model named in the log and in every page's title: the preparation's own (FAST)
    named = "partition_vertical line model, 4 x 8 directions, kernel sigma_r(0) 0.0800 um"
    assert "forward model: " + named in log
    assert len(titles) == 2 and all(named in t for t in titles), titles
    # (vi) the CSV's columns: in_summary right after ok; ok and in_summary as the rule states
    rows = read_csv(out / "sigma_7.csv")
    cols = list(rows[0].keys())
    assert cols.index("in_summary") == cols.index("ok") + 1
    rec = strict_load(out / "sigma_7.json")
    for r, f in zip(rows, rec["fits"]):
        at = set(r["at_bound"].split())
        ok = r["status"] == "converged" and not ({"sigma0_um", "growth_scale"} & at)
        assert r["ok"] == str(ok) and f["ok"] == ok
        ins = ok and math.isfinite(float(r["sigma0_um"])) and math.isfinite(float(r["se_sigma0_um"])) \
            and float(r["se_sigma0_um"]) > 0
        assert r["in_summary"] == str(ins) and f["in_summary"] == ins
    # the summary over the in_summary nodes, by hand: median, MAD (unscaled), inverse-variance mean, its SE, chi2_cons
    s = np.array([f["sigma0_um"] for f in rec["fits"] if f["in_summary"]])
    e = np.array([f["se_sigma0_um"] for f in rec["fits"] if f["in_summary"]])
    w = 1 / e ** 2
    m = np.sum(w * s) / np.sum(w)
    summ = rec["summary"]
    assert summ["n"] == s.size == 2
    assert abs(summ["median_um"] - np.median(s)) <= 1e-12
    assert abs(summ["mad_um"] - np.median(np.abs(s - np.median(s)))) <= 1e-12
    assert abs(summ["ivw_mean_um"] - m) <= 1e-12 and abs(summ["ivw_se_um"] - 1 / math.sqrt(np.sum(w))) <= 1e-12
    assert abs(summ["chi2_cons"] - np.sum((s - m) ** 2 * w)) <= 1e-9 and summ["dof"] == 1
    for name in ("sigma_node5.png", "sigma_node8.png", "sigma_summary_7.png"):
        assert (out / name).exists()


def test_config_json_merged_over_the_preparation(prep, tmp_path):
    import calibrate_sigma as CS
    rec = json.load(open(prep / "cleaning_7.json"))
    only = tmp_path / "only_sigma.json"
    only.write_text(json.dumps({"sigma_calibration": {"max_nfev": 5}}))
    cfg = CS.run_configuration(rec, str(only))
    base = CS.run_configuration(rec, None)
    assert cfg.sigma_calibration.max_nfev == 5
    assert cfg.inference == base.inference and cfg.renderer == base.renderer      # the forward model as prepared
    assert (cfg.inference.absorption, cfg.inference.ray_n_rho, cfg.inference.ray_n_psi) == ("partition_vertical", 4, 8)
    part = tmp_path / "part.json"
    part.write_text(json.dumps({"inference": {"ray_n_rho": 2}}))
    cfg2 = CS.run_configuration(rec, str(part))
    assert (cfg2.inference.absorption, cfg2.inference.ray_n_rho, cfg2.inference.ray_n_psi) == \
        ("partition_vertical", 2, 8)                                                 # field by field
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sigma_calibraton": {"max_nfev": 5}}))
    with pytest.raises(ValueError):
        CS.run_configuration(rec, str(bad))                                          # an unknown section is refused
    # end to end: the run with only the sigma_calibration section keeps the preparation's forward model in the log
    rc, log = _run([WS / "scripts" / "calibrate_sigma.py", "--cleaning-dir", prep, "--specimen", "7", "--out",
                    tmp_path / "s", "--workers", "1", "--nodes", "11", "--config-json", only])
    assert rc == 0, log
    assert "forward model: partition_vertical line model, 4 x 8 directions" in log
    assert strict_load(tmp_path / "s" / "sigma_7.json")["config"]["max_nfev"] == 5


@pytest.mark.parametrize("workers", ["1", "2"])
def test_a_raising_node_is_recorded_and_the_run_goes_on(prep, tmp_path, workers):
    bad = tmp_path / "prep"
    shutil.copytree(prep, bad)
    rec = json.load(open(bad / "cleaning_7.json"))
    rec["nodes"]["8"]["sigma_gl"] = 0.0                  # fit_sigma_node refuses a noise SD that is not > 0
    (bad / "cleaning_7.json").write_text(json.dumps(rec))
    out = tmp_path / "sigma"
    cfgj = tmp_path / "n.json"
    cfgj.write_text(json.dumps({"sigma_calibration": {"max_nfev": 6}}))
    rc, log = _run([WS / "scripts" / "calibrate_sigma.py", "--cleaning-dir", bad, "--specimen", "7", "--out", out,
                    "--workers", workers, "--nodes", "8,11", "--config-json", cfgj])
    assert rc == 0, log
    assert "node 8: error" in log and "the run goes on" in log
    rows = read_csv(out / "sigma_7.csv")
    assert [r["node_id"] for r in rows] == ["8", "11"]                     # the order of --nodes
    assert rows[0]["status"] == "error" and rows[0]["ok"] == "False" and rows[0]["in_summary"] == "False"
    assert "Error" in rows[0]["message"]
    assert rows[1]["status"] in ("converged", "max_nfev")
    js = strict_load(out / "sigma_7.json")
    assert js["complete"] is True and js["fits"][0]["status"] == "error"
    assert (out / "sigma_summary_7.png").exists() and (out / "sigma_node11.png").exists()


def test_in_summary_excludes_a_non_finite_se(prep, tmp_path, monkeypatch):
    import calibrate_sigma as CS
    real = CS.SC.fit_sigma_node

    def no_se(*a, **kw):
        fit = real(*a, **kw)
        se = np.array(fit.se_scaled, dtype=float).copy()
        se[4] = np.nan
        return dataclasses.replace(fit, status="converged", at_bound=(), se_scaled=se)

    cfgj = tmp_path / "n.json"
    cfgj.write_text(json.dumps({"sigma_calibration": {"max_nfev": 5}}))
    out = tmp_path / "sigma"
    rc, log = _main_in_process(["--cleaning-dir", prep, "--specimen", "7", "--out", out, "--workers", "1", "--nodes",
                                "11", "--config-json", cfgj], monkeypatch, [(CS.SC, "fit_sigma_node", no_se)])
    assert rc == 0, log
    row = read_csv(out / "sigma_7.csv")[0]
    assert row["ok"] == "True" and row["in_summary"] == "False"
    js = strict_load(out / "sigma_7.json")                                   # NaN written as null (strict JSON)
    assert js["summary"]["n"] == 0 and js["summary"]["median_um"] is None and js["fits"][0]["se_sigma0_um"] is None


def test_determinism_and_one_or_two_workers(prep, tmp_path):
    cfgj = tmp_path / "n.json"
    cfgj.write_text(json.dumps({"sigma_calibration": {"max_nfev": 8}}))
    outs = []
    for tag, workers in (("a", "1"), ("b", "1"), ("c", "2")):
        rc, log = _run([WS / "scripts" / "calibrate_sigma.py", "--cleaning-dir", prep, "--specimen", "7", "--out",
                        tmp_path / tag, "--workers", workers, "--nodes", "5,11", "--config-json", cfgj])
        assert rc == 0, log
        outs.append([{k: v for k, v in r.items() if k not in ("seconds",)} for r in read_csv(tmp_path / tag /
                                                                                           "sigma_7.csv")])
    assert outs[0] == outs[1]                       # the same seedless inputs, the same rows (timings aside)
    assert outs[0] == outs[2]                       # one worker or two: the same fits
