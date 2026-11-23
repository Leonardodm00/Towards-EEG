"""Spec-derived tests, Block 36 amendment 21 (D-220.3): scripts/calibrate_sigma.py's last lines -- tester 2026-11-23.

Oracles (SPEC Block 36 amendments 19-21, section 6's row): on a `clean_cell.py clean --no-clean --synthetic`
preparation,
  - with no node fitted (every node raises) the script exits 1 with a message, after recording each node with status
    "error" in a strict-JSON sigma_<S>.json and the CSV, and names no summary figure (none is written);
  - with one node fitted and one raising, it exits 0 and its "wrote" line names the CSV, the JSON, the summary figure
    written and the one node page, and counts the node that failed; sigma_node<id>.png exists for the fitted node only.
A node is made to raise by removing `profiles_before` from its npz (a copy of the preparation).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import soma_spec_helpers as H

WS = str(H.WS)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", OMP_NUM_THREADS="1")


def _py(*args, timeout=1800):
    p = subprocess.run([sys.executable] + list(args), cwd=WS, env=ENV, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout + p.stderr


@pytest.fixture(scope="module")
def prep():
    d = tempfile.mkdtemp(prefix="tester_sigma_prep_")
    rc, log = _py("scripts/clean_cell.py", "survey", "--synthetic", "--specimen", "0", "--B", "210", "--mu", "1.0",
                  "--out", d)
    assert rc == 0, log[-2000:]
    rc, log = _py("scripts/clean_cell.py", "clean", "--synthetic", "--specimen", "0", "--examples",
                  os.path.join(d, "examples_0.json"), "--out", d, "--no-clean", "--B", "210", "--mu", "1.0")
    assert rc == 0, log[-2000:]
    return d


def _copy_breaking(prep, broken):
    d = tempfile.mkdtemp(prefix="tester_sigma_case_")
    shutil.rmtree(d)
    shutil.copytree(prep, d)
    for nid in broken:
        fn = os.path.join(d, "cleaning_0_node%d.npz" % nid)
        z = dict(np.load(fn, allow_pickle=False))
        z.pop("profiles_before")
        np.savez(fn, **z)
    return d


def _strict(path):
    with open(path) as f:
        return json.loads(f.read(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError("non-strict: " + c)))


def test_no_node_fitted_exits_1(prep):
    d = _copy_breaking(prep, (8, 9))
    rc, log = _py("scripts/calibrate_sigma.py", "--cleaning-dir", d, "--specimen", "0", "--nodes", "8,9",
                  "--workers", "1")
    lines = [ln for ln in log.strip().splitlines() if ln.strip()]
    assert rc == 1
    assert "no node was fitted" in lines[-1]
    wrote = [ln for ln in lines if ln.startswith("wrote ")]
    assert len(wrote) == 1 and "sigma_summary" not in wrote[0] and "0 node page" in wrote[0] and "2 node(s) failed" \
        in wrote[0]
    rec = _strict(os.path.join(d, "sigma", "sigma_0.json"))
    assert [f["status"] for f in rec["fits"]] == ["error", "error"] and rec["complete"] is True
    assert not os.path.exists(os.path.join(d, "sigma", "sigma_summary_0.png"))


def test_one_fitted_one_failed_exits_0(prep):
    d = _copy_breaking(prep, (9,))
    rc, log = _py("scripts/calibrate_sigma.py", "--cleaning-dir", d, "--specimen", "0", "--nodes", "8,9",
                  "--workers", "1")
    lines = [ln for ln in log.strip().splitlines() if ln.strip()]
    assert rc == 0, log[-3000:]
    last = lines[-1]
    assert last.startswith("wrote ") and "sigma_0.csv" in last and ".json" in last
    assert "sigma_summary_0.png" in last and "1 node page" in last and "1 node(s) failed" in last
    out = os.path.join(d, "sigma")
    assert os.path.exists(os.path.join(out, "sigma_node8.png")) and not os.path.exists(os.path.join(out,
                                                                                                   "sigma_node9.png"))
    assert os.path.exists(os.path.join(out, "sigma_summary_0.png"))
    rec = _strict(os.path.join(out, "sigma_0.json"))
    assert [f["status"] for f in rec["fits"]][1] == "error"
