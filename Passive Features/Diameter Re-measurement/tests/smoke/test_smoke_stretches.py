"""Smoke test for the stretch list -- Block 11 in specs/SPEC.md (2026-10-07):
survey.path_distance_um, survey.stretch_table, survey.suggest_nodes and
scripts/list_stretches.py.

The fixture is a hand-built SWC whose every number is known by hand: a soma
(id 1) at the origin; basal stretch A (ids 2-13, along +x from x = 10 um,
every 2 um, d 1.0) and basal stretch B (ids 14-18, along -x from x = -8,
every 1 um, d 0.8); an apical trunk along +y (T1: ids 19-24, y = 10..25 every
3 um, d 3.0) with an oblique off its last node (O: ids 25-34, from x = 2 at
y = 25 every 2 um, d 0.6), the trunk continuing (T2: ids 35-44, y = 28..55
every 3 um, d 2.0) to the main bifurcation (id 44) and two tuft stretches
(L: ids 45-56, R: ids 57-60, 5 um steps along (-3, 4)/5 and (3, 4)/5, d 1.0
and 0.8).

    stretch  type  n   first  mid  last  length  path start  order  d    terminal
    0 A      3     12  2      8    13    32      10          0      1.0  yes
    1 B      3     5   14     16   18    12      8           0      0.8  yes
    2 T1     4     6   19     22   24    25      10          0      3.0  no
    3 O      4     10  25     30   34    20      27          1      0.6  yes
    4 T2     4     10  35     40   44    30      28          1      2.0  no
    5 L      4     12  45     51   56    60      60          2      1.0  yes
    6 R      4     4   57     59   60    20      60          2      0.8  yes

What it checks (strongest first)
    test_known_answer   the table above, exactly (to 1e-9 um)
    test_reference      the stretches are those of cell.stretches (Block 8);
                        path distances equal a brute-force walk to the root
    test_convergence    skipped: no discretisation
    test_invariants     suggest_nodes for per_type 1, 2, 3 and min_nodes 4,
                        10 equals the ranks derived by hand; an include id
                        comes first and stands for its stretch
    test_contract       the script writes the CSV (7 rows) and the JSON
                        (suggestion, nodes_arg, rule, counts) and prints the
                        NODES line
    test_determinism    the same input gives the same lists
    test_edge_cases     a non-dendrite or unknown include id, per_type 0 and
                        a detached cycle in the parent links raise ValueError

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_stretches.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table.
        Exit code 1 if any check failed, errored or is still TODO.
    python -m pytest tests/smoke/test_smoke_stretches.py -q

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import importlib.metadata
import json
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
for _p in (SRC, WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import cell, survey  # noqa: E402
from allen_diameter.loading import swc_io  # noqa: E402

SEED = 0      # nothing random here
REPORT_PACKAGES = ("numpy",)
EXPECTED = [  # stretch, type, n, first, mid, last, length, path start, order, d, terminal
    (0, 3, 12, 2, 8, 13, 32.0, 10.0, 0, 1.0, True),
    (1, 3, 5, 14, 16, 18, 12.0, 8.0, 0, 0.8, True),
    (2, 4, 6, 19, 22, 24, 25.0, 10.0, 0, 3.0, False),
    (3, 4, 10, 25, 30, 34, 20.0, 27.0, 1, 0.6, True),
    (4, 4, 10, 35, 40, 44, 30.0, 28.0, 1, 2.0, False),
    (5, 4, 12, 45, 51, 56, 60.0, 60.0, 2, 1.0, True),
    (6, 4, 4, 57, 59, 60, 20.0, 60.0, 2, 0.8, True),
]
KEYS = ("stretch", "type", "n_nodes", "first_node", "mid_node", "last_node", "length_um", "path_start_um", "order",
        "allen_diameter_um", "terminal")


def fixture_lines():
    """SWC lines of the hand-built cell (see the module docstring)."""
    rows = [(1, 1, 0.0, 0.0, 0.0, 5.0, -1)]

    def run(first_id, parent, start, step, n, radius, typ):
        for k in range(n):
            x, y = start[0] + k * step[0], start[1] + k * step[1]
            rows.append((first_id + k, typ, x, y, 0.0, radius, parent if k == 0 else first_id + k - 1))

    run(2, 1, (10.0, 0.0), (2.0, 0.0), 12, 0.5, 3)        # A
    run(14, 1, (-8.0, 0.0), (-1.0, 0.0), 5, 0.4, 3)       # B
    run(19, 1, (0.0, 10.0), (0.0, 3.0), 6, 1.5, 4)        # T1, ends at (0, 25)
    run(25, 24, (2.0, 25.0), (2.0, 0.0), 10, 0.3, 4)      # O
    run(35, 24, (0.0, 28.0), (0.0, 3.0), 10, 1.0, 4)      # T2, ends at (0, 55)
    run(45, 44, (-3.0, 59.0), (-3.0, 4.0), 12, 0.5, 4)    # L
    run(57, 44, (3.0, 59.0), (3.0, 4.0), 4, 0.4, 4)       # R
    return ["# hand-built\n"] + ["%d %d %.4f %.4f %.4f %.4f %d\n" % r for r in rows]


def fixture_swc(tmp, lines=None):
    path = os.path.join(tmp, "hand.swc")
    with open(path, "w") as f:
        f.writelines(fixture_lines() if lines is None else lines)
    return swc_io.read_swc(path)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    with tempfile.TemporaryDirectory() as tmp:
        rows = survey.stretch_table(fixture_swc(tmp))
    assert len(rows) == len(EXPECTED), len(rows)
    for r, e in zip(rows, EXPECTED):
        for k, v in zip(KEYS, e):
            if isinstance(v, float):
                assert abs(r[k] - v) <= 1e-9, (k, r, e)      # straight segments: exact to roundoff
            else:
                assert r[k] == v, (k, r, e)


def test_reference():
    with tempfile.TemporaryDirectory() as tmp:
        swc = fixture_swc(tmp)
        rows = survey.stretch_table(swc)
        runs = cell.stretches(swc)
        assert [(r["first_node"], r["last_node"], r["n_nodes"]) for r in rows] == [
            (int(swc.ids[run[0]]), int(swc.ids[run[-1]]), len(run)) for run in runs]
        dist = survey.path_distance_um(swc)
        pidx = swc.parent_index()
        for i in range(len(swc)):                     # brute force: walk to the root
            d, j = 0.0, i
            while pidx[j] >= 0:
                d += float(np.linalg.norm(swc.xyz[j] - swc.xyz[pidx[j]]))
                j = pidx[j]
            assert abs(dist[i] - d) <= 1e-9, (int(swc.ids[i]), dist[i], d)


def test_convergence():
    raise unittest.SkipTest("no discretisation: a table of the SWC's own nodes")


def test_invariants():
    with tempfile.TemporaryDirectory() as tmp:
        swc = fixture_swc(tmp)
        # min_nodes 10: basal eligible {A}; apical eligible by path start {O 27, T2 28, L 60}
        assert survey.suggest_nodes(swc, per_type=3, min_nodes=10) == [8, 30, 40, 51]
        assert survey.suggest_nodes(swc, per_type=2, min_nodes=10) == [8, 30, 51]      # ranks 0, 2
        assert survey.suggest_nodes(swc, per_type=1, min_nodes=10) == [8, 40]          # rank (3 - 1) // 2 = 1
        # min_nodes 4: basal {B 8, A 10} (both); apical {T1 10, O 27, T2 28, L 60, R 60}: ranks 0 and 4
        assert survey.suggest_nodes(swc, per_type=2, min_nodes=4) == [16, 8, 22, 59]
        # an include id comes first and stands for its stretch (44 is T2's last node)
        assert survey.suggest_nodes(swc, per_type=3, min_nodes=10, include=[44]) == [44, 8, 30, 51]
        assert survey.suggest_nodes(swc, per_type=3, min_nodes=10, include=[8]) == [8, 30, 40, 51]
        assert survey.suggest_nodes(swc, types=(4,), per_type=3, min_nodes=10) == [30, 40, 51]
    # the rank rule itself, half-way cases included (floor(x + 1/2), not round-half-to-even: 2.5 -> 3)
    for (m, k), want in {(6, 3): [0, 3, 5], (5, 2): [0, 4], (3, 1): [1], (4, 1): [1], (2, 3): [0, 1], (0, 3): [],
                         (4, 3): [0, 2, 3], (10, 4): [0, 3, 6, 9], (7, 7): [0, 1, 2, 3, 4, 5, 6]}.items():
        assert survey.spread_ranks(m, k) == want, (m, k, survey.spread_ranks(m, k), want)
    for m in range(1, 40):
        for k in range(1, 12):
            r = survey.spread_ranks(m, k)
            assert len(r) == min(m, k) and r == sorted(set(r)) and all(0 <= x < m for x in r), (m, k, r)
            if 1 < k < m:
                assert r[0] == 0 and r[-1] == m - 1, (m, k, r)


def test_contract():
    import list_stretches as LS
    with tempfile.TemporaryDirectory() as tmp:
        swc = fixture_swc(tmp)
        out = os.path.join(tmp, "out")
        lines = []
        rows, info = LS.run(swc, out, "hand", per_type=3, min_nodes=10, include=[44], log=lines.append)
        assert info["suggested_nodes"] == [44, 8, 30, 51] and info["nodes_arg"] == "44,8,30,51"
        assert info["n_nodes_measured_by_pilot"] == 10 + 12 + 10 + 12          # T2, A, O, L
        assert info["by_type"]["basal"] == dict(n_stretches=2, n_eligible=1, n_nodes=17, max_order=0)
        assert info["by_type"]["apical"] == dict(n_stretches=5, n_eligible=3, n_nodes=42, max_order=2)
        with open(os.path.join(out, "stretches_hand.json")) as f:
            assert json.load(f) == info
        with open(os.path.join(out, "stretches_hand.csv")) as f:
            csv_lines = f.read().splitlines()
        assert len(csv_lines) == 1 + 7 and csv_lines[0].split(",") == list(KEYS)
        assert lines[-1] == 'NODES = "44,8,30,51"', lines[-1]
        # the CLI with --swc reaches the same answer (no network)
        path = os.path.join(tmp, "hand.swc")
        assert LS.main(["--specimen", "hand2", "--out-dir", out, "--swc", path, "--include", "44"]) == 0
        with open(os.path.join(out, "stretches_hand2.json")) as f:
            assert json.load(f)["suggested_nodes"] == [44, 8, 30, 51]


def test_determinism():
    with tempfile.TemporaryDirectory() as tmp:
        swc = fixture_swc(tmp)
        assert survey.stretch_table(swc) == survey.stretch_table(swc)
        assert survey.suggest_nodes(swc, per_type=2, min_nodes=4) == survey.suggest_nodes(swc, per_type=2, min_nodes=4)


def test_edge_cases():
    with tempfile.TemporaryDirectory() as tmp:
        swc = fixture_swc(tmp)
        for kw in (dict(include=[1]), dict(include=[999]), dict(per_type=0), dict(min_nodes=0)):
            try:
                survey.suggest_nodes(swc, **kw)
            except ValueError:
                pass
            else:
                raise AssertionError("suggest_nodes accepted %r" % (kw,))
        # min_nodes above every stretch: only the include ids are left
        assert survey.suggest_nodes(swc, min_nodes=100, include=[16]) == [16]
        # a detached cycle (ids 61 <-> 62) is refused, not given NaN distances
        bad = fixture_swc(tmp, fixture_lines() + ["61 3 1.0 1.0 0.0 0.3 62\n", "62 3 2.0 1.0 0.0 0.3 61\n"])
        try:
            survey.path_distance_um(bad)
        except ValueError as exc:
            assert "61" in str(exc) or "62" in str(exc), exc
        else:
            raise AssertionError("a cycle in the parent links was accepted")


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
        except Exception:  # unexpected errors are reported, never hidden
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
