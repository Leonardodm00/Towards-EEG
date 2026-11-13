"""Spec-derived tests, tools: noise_fit_gates.py (bytes, compile, ss-pulses)
and cluster_gates_noise_fit.sh's static properties (SPEC section 5, Tools).
Expected values come from the planted files and from the tester's own
writer-like fixture (its layout known by construction).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import _fixtures as fx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATES = os.path.join(ROOT, "noise_fit_gates.py")


def _run(args, cwd=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, GATES] + args, capture_output=True, text=True, cwd=cwd, env=env)
    return p.returncode, p.stdout


def test_OT_1_bytes(tmp_path):
    d = str(tmp_path)
    os.makedirs(os.path.join(d, "mod"))
    open(os.path.join(d, "good.py"), "wb").write(b"x = 1\n")
    open(os.path.join(d, "good.sh"), "wb").write(b"#!/bin/bash\necho ok\n")
    rc, out = _run(["bytes", "--folder", d])
    assert rc == 0 and out.startswith("BYTES: ok")
    open(os.path.join(d, "bad.py"), "wb").write("y = 'é'\n".encode("utf-8"))
    open(os.path.join(d, "bad.sh"), "wb").write(b"#!/bin/bash\r\necho x\r\n")
    open(os.path.join(d, "mod", "bad.mod"), "wb").write(b"NEURON {\r\n}\r\n")
    rc, out = _run(["bytes", "--folder", d])
    assert rc == 1 and out.startswith("BYTES: FAIL")
    for name in ("bad.py", "bad.sh", os.path.join("mod", "bad.mod")):
        assert name in out
    assert "good.py" not in out and "good.sh" not in out


def test_OT_2_compile(tmp_path):
    d = str(tmp_path)
    open(os.path.join(d, "a.py"), "w").write("x = 1\n")
    rc, out = _run(["compile", "--folder", d])
    assert rc == 0 and "COMPILE: ok" in out
    open(os.path.join(d, "b.py"), "w").write("def f(:\n    pass\n")
    rc, out = _run(["compile", "--folder", d])
    assert rc == 1 and "COMPILE: FAIL" in out and "b.py" in out
    assert not any(n.endswith(".pyc") or n == "__pycache__" for n in os.listdir(d))


def test_OT_3_ss_pulses_summary_from_planted_layout(tmp_path):
    d = fx.write_cell(str(tmp_path), specimen_id=444, drop=[(0, 3), (0, 4), (1, 10), (1, 11), (1, 12)])
    rc, out = _run(["ss-pulses", d])
    line = [ln for ln in out.splitlines() if ln.startswith("SS_PULSES 444:")][0]
    exp = ("SS_PULSES 444: format=A sweep_number=yes n_pulses=69 n_sweeps=2 per_sweep=35,34 adjacent=65/67 "
           "opposite_polarity_adjacent=65/65 spacing_ms=200.500 fs_Hz=50000 window_ms=-10.000..200.000")
    assert rc == 0 and line == exp, line


def test_OT_3_no_sweep_number_exit_1(tmp_path):
    d = fx.write_cell(str(tmp_path), specimen_id=445, drop=[(0, 3)])
    p = os.path.join(d, "ss_pulses.npz")
    z = dict(np.load(p))
    z.pop("sweep_number")
    np.savez(p, **z)
    rc, out = _run(["ss-pulses", d])
    assert rc == 1 and "sweep_number=no" in out and "n_chains=" in out


def test_OT_4_runner_static_properties():
    src = open(os.path.join(ROOT, "cluster_gates_noise_fit.sh"), "rb").read()
    assert b"\r" not in src and all(c < 128 for c in src)
    assert subprocess.run(["bash", "-n", os.path.join(ROOT, "cluster_gates_noise_fit.sh")]).returncode == 0
    text = src.decode()
    for check in ("G0 ", "G1 ", "G2 ", "G3 ", "G4 ", "G5 ", "G6 ", "G6b ", "G6c ", "G6d ", "G7 ", "G8 ", "G9 ",
                  "G10 ", "G11 ", "G12 "):
        assert '"%s' % check in text, check
    for expected in ("SMOKE noise_stats: 6/6 passed", "SMOKE noise_law: 10/10 passed",
                     "SMOKE noise_windows: 15/15 passed", "SMOKE noise_fit: 11/11 passed",
                     "SMOKE noise_sweep_fit: 11/11 passed"):
        assert expected in text, expected
