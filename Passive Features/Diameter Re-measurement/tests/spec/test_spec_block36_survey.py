"""Spec-derived tests of Block 36, amendment 5 (2026-11-14, 14:55): `clean_cell.py survey` visits the dendrite
nodes in a seeded random order (--seed, default phantom.seed, plus 1), --max-nodes N stops after N, the
outputs are rewritten every --checkpoint-every nodes and when the script is stopped (SIGINT, SIGTERM), and
examples_S.json records n_target, n_dendrite_nodes, partial and sample; the example nodes are drawn from the
rows written. Integration tester, 2026-11-14 18:30. Oracles: the prefix property of a seeded order (a sample
of N is the first N rows of the full run under the same seed; a stopped run's rows are a prefix too), row
equality with the full run, determinism, the recorded fields.

Run: python3 -m pytest -q tests/spec/test_spec_block36_survey.py   (from the code folder; about 2 min)
Pure ASCII.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import nc_helpers  # noqa: F401  (sys.path)

WS = Path(__file__).resolve().parent.parent.parent
SCRIPT = str(WS / "scripts" / "clean_cell.py")


def survey(out, *extra):
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, SCRIPT, "survey", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0",
           "--out", str(out)] + [str(x) for x in extra]
    r = subprocess.run(cmd, cwd=str(WS), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return read(out)


def read(out):
    rows = list(csv.DictReader(open(os.path.join(out, "contamination_7.csv"))))
    ex = json.load(open(os.path.join(out, "examples_7.json")))
    return rows, ex


_FULL = {}


def full(tmp_path_factory):
    if "a" not in _FULL:
        _FULL["a"] = survey(tmp_path_factory.mktemp("survey_full_a"))
    return _FULL["a"]


def test_full_run_deterministic_and_fields(tmp_path_factory):
    """Two full runs under the default seed are identical (rows in visiting order, the examples file); the
    file records n_target = n_dendrite_nodes, partial False, the sample; every example is a row written and not
    near_end."""
    rows, ex = full(tmp_path_factory)
    rows2, ex2 = survey(tmp_path_factory.mktemp("survey_full_b"))
    assert rows == rows2
    assert ex == ex2
    assert ex["n_target"] == ex["n_dendrite_nodes"] and ex["partial"] is False
    assert ex["n_surveyed"] + len(ex["failed"]) == ex["n_target"] == len(rows) + len(ex["failed"])
    assert "sample" in ex and isinstance(ex["sample"], str)
    by_id = {int(r["node_id"]): r for r in rows}
    assert len(by_id) == len(rows)                                # no node twice
    for i in ex["example_ids"]:
        assert i in by_id and by_id[i]["near_end"] in ("False", "false", "0")


def test_max_nodes_is_a_prefix_of_the_seeded_order(tmp_path_factory):
    """--max-nodes 5: the 5 rows are the first 5 rows of the full run under the same seed, equal field by field
    (the order is one seeded permutation); n_target 5, n_dendrite_nodes the full count, partial False; the
    examples are drawn from those 5 rows. Another --seed gives another sample."""
    rows, ex = full(tmp_path_factory)
    r5, e5 = survey(tmp_path_factory.mktemp("survey_5"), "--max-nodes", 5)
    assert len(r5) == 5
    assert r5 == rows[:5]
    assert e5["n_target"] == 5 and e5["n_dendrite_nodes"] == ex["n_dendrite_nodes"] and e5["partial"] is False
    assert set(e5["example_ids"]) <= {int(r["node_id"]) for r in r5}
    seed = int(ex["seed"])
    r5b, _e = survey(tmp_path_factory.mktemp("survey_5_seed"), "--max-nodes", 5, "--seed", seed + 17)
    assert [r["node_id"] for r in r5b] != [r["node_id"] for r in r5]
    by_id = {r["node_id"]: r for r in rows}
    assert all(by_id[r["node_id"]] == r for r in r5b)           # a node's row does not depend on the order


def test_sigterm_writes_a_partial_prefix(tmp_path_factory):
    """Stopped by SIGTERM after the first checkpoint (--checkpoint-every 1): the script exits 0 and leaves
    outputs of the nodes done, partial True, rows a prefix of the full run's (equal field by field)."""
    rows, ex = full(tmp_path_factory)
    out = tmp_path_factory.mktemp("survey_term")
    cmd = [sys.executable, SCRIPT, "survey", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0",
           "--out", str(out), "--checkpoint-every", "1"]
    p = subprocess.Popen(cmd, cwd=str(WS), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    path = os.path.join(out, "examples_7.json")
    t0 = time.time()
    while not os.path.exists(path) and p.poll() is None and time.time() - t0 < 300:
        time.sleep(0.02)
    if p.poll() is not None:
        p.communicate()
        pytest.skip("the survey finished before the signal could be sent")
    p.send_signal(signal.SIGTERM)
    so, se = p.communicate(timeout=300)
    assert p.returncode == 0, so[-2000:] + se[-2000:]
    r, e = read(out)
    if not e["partial"]:
        pytest.skip("the signal arrived after the last node (the run completed)")
    assert 1 <= len(r) < len(rows)
    assert r == rows[:len(r)]
    assert e["n_surveyed"] == len(r) and e["n_target"] == ex["n_target"]
    assert set(e["example_ids"]) <= {int(q["node_id"]) for q in r}
    print("stopped after %d of %d nodes" % (len(r), len(rows)))
