"""Smoke test of diameter_reach.py (the check of 2026-11-22). Oracles, strongest first:
  T1 known answer  a ball-and-stick model reproduces the analytic input resistance of a sealed finite cable,
                   R_in = 1 / (G_soma + G_inf tanh(L / lambda)), G_inf = pi d^2 / (4 Ra lambda),
                   lambda = sqrt(Rm d / (4 Ra))  [0.5 %: the dendrite cut into 201 segments for this test]
  T2 invariance    scaling every dendritic diameter by s is undone exactly by Cm/s, s Rm, s^2 Ra (the cable
                   equation's parameters per unit length); a Gauss-Newton refit with all dendrites scaled
                   by 1.1 recovers q - q0 = (-ln s, +ln s, +2 ln s)  [0.005 in q: the soma, 0.1 % of the
                   membrane, is not scaled]
  T3 linearised    one Gauss-Newton step at s = 1.01 gives the same shifts  [3 % relative: second order in ln s]
  T4 contract      the mask scales exactly the sections whose midpoint lies beyond x, and restore_diameters
                   returns every 3-d diameter bit for bit; the targets' shapes
  T5 determinism   two simulations are identical
Run:  python3 smoke_diameter_reach.py     (about a minute)
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diameter_reach as R   # noqa: E402

RESULTS = []


def check(name, fn):
    t = time.time()
    try:
        msg = fn()
        RESULTS.append((name, "PASS", time.time() - t, msg or ""))
    except AssertionError as e:
        RESULTS.append((name, "FAIL", time.time() - t, str(e)))
    except Exception as e:                    # noqa: BLE001 -- reported, the run continues
        RESULTS.append((name, "ERROR", time.time() - t, "%s: %s" % (type(e).__name__, e)))


def write_swc(rows):
    fd, path = tempfile.mkstemp(suffix=".swc")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write("%d %d %.4f %.4f %.4f %.4f %d\n" % tuple(r))
    return path


def ball_and_stick(r_soma=10.0, length=500.0, radius=1.0):
    rows = [(1, 1, 0.0, 0.0, 0.0, r_soma, -1)]
    for i in range(int(length) + 1):
        rows.append((i + 2, 3, r_soma + i, 0.0, 0.0, radius, i + 1))
    return write_swc(rows)


def tree(r_soma=0.5):
    """A trunk of 50 um (radius 1.0) splitting twice into 100 um branches (radius 0.7, then 0.5): 7 sections,
    midpoints at 25, 100, 100, 200, 200, 200, 200 um from the soma."""
    rows, nid = [(1, 1, 0.0, 0.0, 0.0, r_soma, -1)], [1]

    def branch(parent, start, direction, length, radius):
        last = parent
        for i in range(1, int(length) + 1):
            nid[0] += 1
            p = start + direction * i
            rows.append((nid[0], 3, p[0], p[1], p[2], radius, last))
            last = nid[0]
        return last, start + direction * length

    e1, p1 = branch(1, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0, 0]), 50, 1.0)
    for sy in (1, -1):
        d2 = np.array([1.0, sy, 0]) / math.sqrt(2)
        e2, p2 = branch(e1, p1, d2, 100, 0.7)
        for sz in (1, -1):
            d3 = np.array([1.0, sy, sz]) / math.sqrt(3)
            branch(e2, p2, d3, 100, 0.5)
    return write_swc(rows)


def t1_known_answer():
    cfg = R.ReachConfig(axon_stub=False, spine_cutoff_um=1e9)
    cell = R.build_cell(ball_and_stick(), cfg)
    den = cell["dendritic"][0]
    den.nseg = 201                                    # this test only: the analytic cable is continuous
    for k in range(den.nseg):
        cell["mult"][(den.name(), k)] = 1.0           # F nowhere (cutoff beyond the dendrite)
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    y = R.targets(cell, q0, ("step",), cfg)["step"]
    r_model = y[-1] / cfg.protocols["step"]["amp_nA"]                    # MOhm
    d_cm, L_cm = den.diam * 1e-4, den.L * 1e-4
    lam = math.sqrt(cfg.rm_ohm_cm2 * d_cm / (4 * cfg.ra_ohm_cm))
    g_inf = math.pi * d_cm ** 2 / (4 * cfg.ra_ohm_cm * lam)
    a_soma = sum(seg.area() for seg in cell["soma"][0]) * 1e-8            # cm2
    g_soma = a_soma / cfg.rm_ohm_cm2
    r_true = 1e-6 / (g_soma + g_inf * math.tanh(L_cm / lam))              # MOhm
    rel = abs(r_model - r_true) / r_true
    assert rel < 0.005, "R_in %.3f MOhm against the analytic %.3f (%.2f %%)" % (r_model, r_true, 100 * rel)
    return "R_in %.3f MOhm, analytic %.3f MOhm (%.3f %%); L %.1f um, d %.2f um, L/lambda %.3f" % (
        r_model, r_true, 100 * rel, den.L, den.diam, L_cm / lam)


def _tree_cell():
    cfg = R.ReachConfig(axon_stub=False)
    return cfg, R.build_cell(tree(), cfg)


def t2_invariance():
    cfg, cell = _tree_cell()
    names = ("pulse", "step")
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    data = R.targets(cell, q0, names, cfg)
    s = 1.1
    R.scale_beyond(cell, -1.0, s)
    qh, it = R.refit(cell, q0, data, names, cfg)
    R.restore_diameters(cell)
    want = np.array([-math.log(s), math.log(s), 2 * math.log(s)])
    err = np.abs((qh - q0) - want)
    assert np.all(err < 0.005), "refit q - q0 %s against %s" % (np.round(qh - q0, 5), np.round(want, 5))
    return "refit shifts (ln) %s, exact %s, %d iterations" % (np.round(qh - q0, 5), np.round(want, 5), it)


def t3_linearised():
    cfg, cell = _tree_cell()
    names = ("pulse", "step")
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    data = R.targets(cell, q0, names, cfg)
    J = R.jacobian(cell, q0, names, cfg)
    s = 1.01
    R.scale_beyond(cell, -1.0, s)
    y = R.targets(cell, q0, names, cfg)
    R.restore_diameters(cell)
    dq = R.gauss_newton_step(J, {b: y[b] - data[b] for b in names}, names)
    want = np.array([-math.log(s), math.log(s), 2 * math.log(s)])
    rel = np.abs(dq - want) / np.abs(want)
    assert np.all(rel < 0.03), "linearised %s against %s" % (dq, want)
    return "linearised (ln) %s, exact %s, worst %.2f %%" % (np.round(dq, 6), np.round(want, 6), 100 * rel.max())


def t4_contract():
    cfg, cell = _tree_cell()
    d_before = [d.copy() for d in cell["d0"]]
    n = R.scale_beyond(cell, 100.0, 1.3)
    want = int(np.sum(cell["dist_um"] > 100.0))
    assert n == want == 4, "scaled %d sections, %d lie beyond 100 um (4 expected)" % (n, want)
    # Import3d starts a child section with a copy of its parent's last point (the parent's diameter): the
    # section's own diameter is read at its last point
    scaled = [sec.diam3d(int(sec.n3d()) - 1) for sec, dist in zip(cell["dendritic"], cell["dist_um"]) if dist > 100.0]
    assert np.allclose(scaled, 0.5 * 2 * 1.3), scaled
    R.restore_diameters(cell)
    now = [np.array([sec.diam3d(i) for i in range(int(sec.n3d()))]) for sec in cell["dendritic"]]
    assert all(np.array_equal(a, b) for a, b in zip(now, d_before)), "diameters not restored bit for bit"
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    y = R.targets(cell, q0, ("pulse", "step"), cfg)
    assert y["pulse"].shape == (991,) and y["step"].shape == (1001,), (y["pulse"].shape, y["step"].shape)
    assert np.all(np.isfinite(y["pulse"])) and np.all(np.isfinite(y["step"]))
    assert R.x_star([0, 10, 20, 30], [0.2, 0.06, 0.04, 0.01], 0.05) == 20.0
    assert R.x_star([0, 10, 20], [0.2, 0.01, 0.06], 0.05) != R.x_star([0, 10, 20], [0.2, 0.01, 0.01], 0.05)
    return "4 of 7 sections scaled beyond 100 um, restored bit for bit; pulse 991, step 1001 samples"


def t5_determinism():
    cfg, cell = _tree_cell()
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    a = R.targets(cell, q0, ("pulse",), cfg)["pulse"]
    b = R.targets(cell, q0, ("pulse",), cfg)["pulse"]
    assert np.array_equal(a, b)
    return "identical"


if __name__ == "__main__":
    import neuron
    print("python %s | numpy %s | NEURON %s" % (sys.version.split()[0], np.__version__, neuron.__version__))
    for name, fn in (("T1 known answer", t1_known_answer), ("T2 invariance", t2_invariance),
                     ("T3 linearised", t3_linearised), ("T4 contract", t4_contract), ("T5 determinism", t5_determinism)):
        check(name, fn)
    for name, status, sec, msg in RESULTS:
        print("%-5s  %-16s %6.1fs  %s" % (status, name, sec, msg))
    n_bad = sum(1 for r in RESULTS if r[1] != "PASS")
    print("-- %d pass, %d not" % (len(RESULTS) - n_bad, n_bad))
    sys.exit(1 if n_bad else 0)
