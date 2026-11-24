"""Smoke test for the sweeps of one loss evaluation in parallel -- Block 19 in
specs/SPEC.md (section 2.9 (c)).

build_multi_protocol_loss(..., n_workers=W) and integrate_long_step(...,
n_workers=W) in passive_long_step_training.py: the pairs (RMSD_b, D_b) of
the training bundles computed by a fork pool, the loss EXACTLY equal (== on
float64) to the serial one, penalties included.

Checks (the oracles of the Block 19 entry, W1-W10, plus W11-W12)
    W1   >= 8 points q over the box of the six-parameter spec (two on its
         faces), weighting ls_amplitude + peak_decay and relative + none:
         loss_2(q) == loss_1(q) and loss_3(q) == loss_1(q); and loss_1 equals
         the contract commit's loss bit for bit (W = 1 is the serial code)
    W2   after loss_W(q) the parent cell is at q: a serial loss right after
         gives the same value; cm, g_pas, gbar (and e_pas) of the soma equal
         those after loss_1(q)
    W3   lazy pool; worker pids distinct from the parent, at most
         min(W, n_bundles); none alive after close(); close() twice is
         harmless; a call after close() works, equals the serial value
    W4   a bundle with a NaN in its RMSD window: 1e6 for every W
    W5   a DtEnforcementError raised in a worker reaches the caller with that
         class name (the monolith's class, and an unpicklable one); a plain
         exception in a worker gives 1e6; the first failing bundle in the
         bundles' order decides, as in the serial loop
    W6   n_workers = 0, -1, 1.5 raise ValueError (build_multi_protocol_loss
         and integrate_long_step)
    W7   through integrate_long_step(mono, ..., n_workers=2): the loss
         mono._build_loss_function returns has n_workers == 2 and equals the
         serial one at 3 points
    W8   wall time of 5 evaluations, W = 2 against W = 1: REPORTED only
    W9   smoke_ls_protocol.py still 9/9
    W10  ASCII / LF of the block's files
    W13  [amendment 2] one live pool per process: two W = 2 losses on the
         same cell (different bundle subsets) evaluated alternately, 3 times
         each, with warnings turned into errors: every value equals its
         serial loss, no warning, at most min(2, n_bundles) live worker
         children after each call
    W11  (beyond the entry) a worker that dies: BrokenProcessPool in the
         parent, not the 1e6 penalty, and no hang; the next call makes a new
         pool and equals the serial value
    W12  (beyond the entry) the pool is terminated when the loss is
         garbage-collected, and at interpreter exit (a child interpreter that
         never calls close())

Run (from this directory; NEURON loads ./x86_64 from the CWD)
    env -u DISPLAY python3 smoke_ls_pool.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    W1 needs git (it reads the contract commit's module with `git show`).
    The sandbox has 2 cores: W8 reports the speed, it asserts nothing.

Keep this file pure ASCII.
"""
from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                  # noqa: E402

import param_spec as PS             # noqa: E402
import passive_long_step_training as plst  # noqa: E402
# The fixtures the existing smokes use (bundles, stub cell, roles cell).
from smoke_ls_protocol import (V_REST, ls_bundle, ss_bundle, _train_set,  # noqa: E402
                               roles_cell)

CONTRACT = "1e3fe25"
SEED = 20261124
REPORT_PACKAGES = ("numpy", "scipy", "NEURON")
DT = 0.1                            # ms, brief and long (as the existing smokes)
SPEC6 = PS.make_ih_spec()           # the six-parameter spec, its default box


# --------------------------------------------------------------- helpers ---

def expect_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn),
                                                  exc.__name__))


class _Timeout(Exception):
    pass


def _alarm(seconds):
    """A guard against a hang: raises _Timeout after `seconds`."""
    def _h(signum, frame):
        raise _Timeout("timed out after %d s (a hang)" % seconds)
    signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(seconds))


def _children():
    return {p.pid: p for p in multiprocessing.active_children()}


_CONTRACT_MOD = None


def contract_module():
    """passive_long_step_training.py as of the contract commit."""
    global _CONTRACT_MOD
    if _CONTRACT_MOD is None:
        src = subprocess.check_output(
            ["git", "show", "%s:./passive_long_step_training.py" % CONTRACT],
            cwd=str(HERE))
        d = Path(tempfile.mkdtemp(prefix="smoke_pool_contract_"))
        p = d / "plst_contract_b19.py"
        p.write_bytes(src)
        spec = importlib.util.spec_from_file_location("plst_contract_b19", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CONTRACT_MOD = mod
    return _CONTRACT_MOD


# --------------------------------------------------------------- fixtures --
_CELL = {}


def neuron_cell():
    """One ball-and-stick NEURON cell with I_h (Ih_human, uniform), built
    once: PassiveCell.set_passive walks h.allsec(), so a second live cell
    would break the first."""
    if "cell" not in _CELL:
        import passive_fitting_hpc_fixed as mono
        import ih_mechanism as IM
        from synthetic_ground_truth import write_ball_and_stick_swc
        tmp = Path(tempfile.mkdtemp(prefix="smoke_pool_"))
        swc = write_ball_and_stick_swc(tmp / "bas.swc", soma_r_um=10.0,
                                       dend_len_um=300.0, dend_r_um=1.0,
                                       apic_len_um=500.0, apic_r_um=1.2,
                                       step_um=20.0)
        cell = mono.build_neuron_model(swc, F=1.9)
        IM.attach_ih(cell, IM.IhSpec(mechanism="Ih_human", distribution="uniform"))
        _CELL["mono"], _CELL["cell"] = mono, cell
    return _CELL["mono"], _CELL["cell"]


def box_points(spec, n_inner=6, seed=SEED):
    """n_inner random points inside the box (q coordinates) and two on its
    faces: the lower corner, and the upper bound on half the axes."""
    rng = np.random.default_rng(seed)
    lo = np.array([d.low for d in spec.as_skopt_dimensions()])
    hi = np.array([d.high for d in spec.as_skopt_dimensions()])
    pts = [tuple(float(x) for x in lo + rng.uniform(0.05, 0.95, lo.size) * (hi - lo))
           for _ in range(n_inner)]
    pts.append(tuple(float(x) for x in lo))
    mixed = lo + 0.4 * (hi - lo)
    mixed[::2] = hi[::2]
    pts.append(tuple(float(x) for x in mixed))
    return pts


def build(tb, W, *, weighting, lsw, spec=SPEC6, mod=plst):
    _mono, cell = neuron_cell()
    kw = dict(ls_window_mode="step", weighting=weighting, ls_sample_weight=lsw,
              ss_sample_weight_fn=plst.exp_time_weight(5e-4, 5e-3),
              dt_brief_ms=DT, dt_long_ms=DT, spec=spec)
    if mod is plst:
        kw["n_workers"] = W
    return mod.build_multi_protocol_loss(cell, tb, V_REST, **kw)


CONFIGS = (("ls_amplitude", "peak_decay"), ("relative", "none"))


def soma_state(cell):
    seg = cell.soma[0](0.5)
    return (seg.cm, seg.g_pas, seg.e_pas, seg.Ih_human.gIhbar)


# ------------------------------------------------------------ the checks ---

def test_W1_exact_over_the_box():
    pts = box_points(SPEC6)
    assert len(pts) >= 8
    old = contract_module()
    notes = []
    for weighting, lsw in CONFIGS:
        tb = _train_set()
        l1 = build(tb, 1, weighting=weighting, lsw=lsw)
        l0 = build(tb, 1, weighting=weighting, lsw=lsw, mod=old)
        l2 = build(tb, 2, weighting=weighting, lsw=lsw)
        l3 = build(tb, 3, weighting=weighting, lsw=lsw)
        try:
            v1 = [l1(*q) for q in pts]
            v0 = [l0(*q) for q in pts]
            v2 = [l2(*q) for q in pts]
            v3 = [l3(*q) for q in pts]
        finally:
            l2.close()
            l3.close()
        assert v1 == v0, ("W=1 is not the contract's serial loss", weighting, v1, v0)
        assert v2 == v1, ("W=2", weighting, v2, v1)
        assert v3 == v1, ("W=3", weighting, v3, v1)
        assert max(v1) < 1e6 and len(set(v1)) == len(pts), v1   # live values
        notes.append("%s/%s: %d points, range %.6g..%.6g"
                     % (weighting, lsw, len(pts), min(v1), max(v1)))
    print("    W1 " + "; ".join(notes))


def test_W2_parent_cell_at_q():
    _mono, cell = neuron_cell()
    pts = box_points(SPEC6)
    qa, qb = pts[1], pts[4]
    tb = _train_set()
    l1 = build(tb, 1, weighting="ls_amplitude", lsw="peak_decay")
    l2 = build(tb, 2, weighting="ls_amplitude", lsw="peak_decay")
    try:
        va = l1(*qa)
        state_serial = soma_state(cell)
        apic_serial = [(s.cm, s.g_pas) for s in cell.apic[0]]
        l1(*qb)                                   # move the cell elsewhere
        assert soma_state(cell) != state_serial
        vw = l2(*qa)
        assert soma_state(cell) == state_serial, (soma_state(cell), state_serial)
        assert [(s.cm, s.g_pas) for s in cell.apic[0]] == apic_serial
        assert l1(*qa) == va == vw, (va, vw)
        # a serial loss evaluated on the parent's cell, without applying q
        # again, gives the same value: build one whose spec applies nothing
        class _NoApply:
            def apply_q(self, c, q, v):
                return None
        l2(*qa)
        lnone = build(tb, 1, weighting="ls_amplitude", lsw="peak_decay",
                      spec=_NoApply())
        assert lnone(*qa) == va
    finally:
        l2.close()


def test_W3_processes_and_close():
    tb = _train_set()
    n_b = len(tb)
    for W in (2, 3, 8):
        before = _children()
        l1 = build(tb, 1, weighting="relative", lsw="none")
        lw = build(tb, W, weighting="relative", lsw="none")
        q = box_points(SPEC6)[2]
        assert set(_children()) == set(before), "the pool must not exist before the first call"
        v = lw(q[0], q[1], q[2], q[3], q[4], q[5])
        kids = {pid: p for pid, p in _children().items() if pid not in before}
        assert os.getpid() not in kids
        assert 1 <= len(kids) <= min(W, n_b), (W, len(kids))
        assert len(kids) == min(W, n_b), (W, len(kids))     # fork: all at once
        lw(*q)                                               # reused, not re-forked
        assert set(pid for pid in _children() if pid not in before) == set(kids)
        lw.close()
        time.sleep(0.05)
        alive = [pid for pid, p in kids.items() if p.is_alive()]
        assert not alive, alive
        assert not [pid for pid in _children() if pid not in before]
        lw.close()                                           # twice: harmless
        v_again = lw(*q)                                     # a new pool
        new = {pid for pid in _children() if pid not in before}
        assert new and not (new & set(kids)), (new, set(kids))
        assert v_again == v == l1(*q), (v_again, v)
        assert lw.n_workers == W and isinstance(lw.n_workers, int)
        lw.close()
        assert not [pid for pid in _children() if pid not in before]
    # W = 1 has both attributes; close() does nothing
    l1 = build(tb, 1, weighting="relative", lsw="none")
    assert l1.n_workers == 1
    assert l1.close() is None and l1.close() is None


def test_W4_nonfinite_rmsd_penalty():
    nanb = ls_bundle(-50.0)
    k = np.flatnonzero(nanb.t > nanb.stim_onset_s + 0.05)[0]
    nanb.v_mV = nanb.v_mV.copy()
    nanb.v_mV[k] = np.nan                         # inside the RMSD window
    pts = box_points(SPEC6)[:3]
    for pos in (0, 2, 4):
        tb = _train_set()
        tb.insert(pos, nanb)
        vals = {}
        for W in (1, 2, 3):
            lw = build(tb, W, weighting="relative", lsw="none")
            try:
                vals[W] = [lw(*q) for q in pts]
            finally:
                lw.close()
        assert vals[1] == [1e6] * 3, vals[1]
        assert vals[2] == vals[1] and vals[3] == vals[1], vals


def _dt_cls():
    import passive_fitting_hpc_fixed as mono
    return mono.DtEnforcementError


def _patched_rmsd(fail):
    """bundle_rmsd that raises fail[amplitude] for the bundles listed."""
    orig = plst.bundle_rmsd

    def f(cell, bundle, *a, **k):
        exc = fail.get(float(bundle.amplitude_pA))
        if exc is not None:
            raise exc()
        return orig(cell, bundle, *a, **k)
    return orig, f


def _run_both(tb, fail, q, W=2):
    """(serial outcome, parallel outcome): a value, or the exception's class
    name. The patch is in place before the first call (the workers fork)."""
    orig, f = _patched_rmsd(fail)
    plst.bundle_rmsd = f
    out = []
    try:
        for w in (1, W):
            lw = build(tb, w, weighting="relative", lsw="none")
            try:
                out.append(lw(*q))
            except Exception as exc:  # noqa: BLE001 - the outcome under test
                out.append((type(exc).__name__, exc))
            finally:
                lw.close()
    finally:
        plst.bundle_rmsd = orig
    return out


def test_W5_dt_enforcement_and_exceptions():
    q = box_points(SPEC6)[0]
    tb = _train_set()                             # amplitudes -200, -150, -30, -70, -110
    DtE = _dt_cls()

    class DtEnforcementError(RuntimeError):       # local: cannot be pickled
        pass

    for cls in (DtE, DtEnforcementError):
        s, p = _run_both(tb, {-70.0: cls}, q)
        assert isinstance(s, tuple) and s[0] == "DtEnforcementError", s
        assert isinstance(p, tuple) and p[0] == "DtEnforcementError", p
        if cls is DtE:
            assert isinstance(p[1], DtE), type(p[1])     # the original class
    # a plain exception in a worker: the penalty
    s, p = _run_both(tb, {-30.0: ValueError}, q)
    assert s == p == 1e6, (s, p)
    s, p = _run_both(tb, {-30.0: ValueError}, q, W=3)
    assert s == p == 1e6, (s, p)
    # the order rule: an earlier failing bundle decides
    s, p = _run_both(tb, {-150.0: ValueError, -110.0: DtE}, q)
    assert s == p == 1e6, (s, p)
    s, p = _run_both(tb, {-150.0: DtE, -110.0: ValueError}, q)
    assert s[0] == p[0] == "DtEnforcementError", (s, p)
    nanb = ls_bundle(-50.0)
    nanb.v_mV = nanb.v_mV.copy()
    nanb.v_mV[np.flatnonzero(nanb.t > 0.2)[0]] = np.nan
    s, p = _run_both([nanb] + tb, {-110.0: DtE}, q)
    assert s == p == 1e6, (s, p)
    s, p = _run_both(tb + [nanb], {-110.0: DtE}, q)
    assert s[0] == p[0] == "DtEnforcementError", (s, p)
    # the parent's own apply_q raising DtEnforcementError is not swallowed
    class _BadSpec:
        def apply_q(self, c, qq, v):
            raise DtE("parent")
    lw = build(tb, 2, weighting="relative", lsw="none", spec=_BadSpec())
    expect_raises(DtE, lw, *q)
    lw.close()


def test_W6_refusals():
    _mono, cell = neuron_cell()
    tb = _train_set()
    for bad in (0, -1, 1.5):
        expect_raises(ValueError, plst.build_multi_protocol_loss, cell, tb, V_REST,
                      n_workers=bad)
        expect_raises(ValueError, plst.integrate_long_step, object(), n_workers=bad)


def test_W7_through_integrate_long_step():
    mono, cell = neuron_cell()
    cd = roles_cell(6)
    common = dict(spec=SPEC6, ih_protocol=True, ls_window_mode="step",
                  dt_brief_ms=DT, dt_long_ms=DT, verbose=False,
                  ss_time_weight="exp", ss_tau_w_ms=5.0, role_rule="d207",
                  weighting="ls_amplitude", ls_sample_weight="peak_decay")
    pts = box_points(SPEC6)[:3]
    vals = {}
    try:
        for W in (1, 2):
            plst.integrate_long_step(mono, n_workers=W, **common)
            oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
            L = mono._build_loss_function(cell=cell, train_bundles=oi.train_bundles,
                                          v_rest_mV=V_REST,
                                          train_window_ms=oi.train_window_ms)
            try:
                assert L.n_workers == W, (L.n_workers, W)
                vals[W] = [L(*q) for q in pts]
            finally:
                L.close()
    finally:
        plst.integrate_long_step(mono, verbose=False)
    assert vals[2] == vals[1], vals
    assert max(vals[1]) < 1e6, vals[1]


def test_W8_wall_time_reported():
    tb = _train_set()
    pts = box_points(SPEC6)[:5]
    out = {}
    for W in (1, 2):
        lw = build(tb, W, weighting="ls_amplitude", lsw="peak_decay")
        try:
            lw(*pts[0])                           # the pool's creation excluded
            t0 = time.perf_counter()
            for q in pts:
                lw(*q)
            out[W] = time.perf_counter() - t0
        finally:
            lw.close()
    print("    W8 5 evaluations, %d bundles, %d cores: W=1 %.3f s, W=2 %.3f s "
          "(speed-up %.2f; reported, not asserted)"
          % (len(tb), os.cpu_count() or 0, out[1], out[2], out[1] / out[2]))


def test_W9_smoke_ls_protocol():
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    r = subprocess.run([sys.executable, "smoke_ls_protocol.py"], cwd=str(HERE),
                       env=env, capture_output=True, text=True, timeout=1800)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    print("    W9 smoke_ls_protocol.py: %s (exit %d)" % (tail, r.returncode))
    assert r.returncode == 0 and "9 pass" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


def test_W10_ascii_lf():
    for f in ("passive_long_step_training.py", "smoke_ls_pool.py"):
        b = (HERE / f).read_bytes()
        assert all(c < 128 for c in b), f + " non-ASCII"
        assert b"\r" not in b, f + " CR bytes"


def test_W13_one_live_pool():
    import warnings
    tb = _train_set()
    subsets = (tb[:3], tb[2:])                    # 3 bundles each, different
    pts = box_points(SPEC6)
    before = set(_children())
    pooled = [build(s, 2, weighting="ls_amplitude", lsw="peak_decay")
              for s in subsets]
    serial = [build(s, 1, weighting="ls_amplitude", lsw="peak_decay")
              for s in subsets]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for r in range(3):
                for k in (0, 1):
                    q = pts[r + 3 * k]
                    v = pooled[k](*q)
                    live = [pid for pid in _children() if pid not in before]
                    assert len(live) <= min(2, len(subsets[k])), (r, k, live)
                    assert v == serial[k](*q), (r, k, v)
    finally:
        for lw in pooled:
            lw.close()
    assert not [pid for pid in _children() if pid not in before]


def test_W11_worker_death():
    q = box_points(SPEC6)[3]
    tb = _train_set()
    orig = plst.bundle_rmsd

    def dies(cell, bundle, *a, **k):
        if float(bundle.amplitude_pA) == -70.0:
            os._exit(3)                           # a crash, not an exception
        return orig(cell, bundle, *a, **k)
    plst.bundle_rmsd = dies
    lw = build(tb, 2, weighting="relative", lsw="none")
    try:
        _alarm(120)
        try:
            lw(*q)
        except BrokenProcessPool:
            pass
        else:
            raise AssertionError("a dead worker was not reported")
        finally:
            signal.alarm(0)
    finally:
        plst.bundle_rmsd = orig
    # the next call makes a new pool (forked now, without the patch's death)
    _alarm(120)
    try:
        v = lw(*q)
    finally:
        signal.alarm(0)
        lw.close()
    l1 = build(tb, 1, weighting="relative", lsw="none")
    assert v == l1(*q), (v, l1(*q))


_CHILD = textwrap.dedent("""
    import gc, os, sys, multiprocessing
    sys.path.insert(0, os.getcwd())
    import smoke_ls_pool as S
    import passive_long_step_training as plst
    tb = S._train_set()
    q = S.box_points(S.SPEC6)[0]
    lw = S.build(tb, 2, weighting="relative", lsw="none")
    lw(*q)
    pids = [p.pid for p in multiprocessing.active_children()]
    if sys.argv[1] == "gc":
        del lw
        gc.collect()
        left = [p.pid for p in multiprocessing.active_children()]
        print("GC", pids, left)
    else:
        print("EXIT", pids)
    sys.stdout.flush()
""")


def _pid_alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return False


def test_W12_gc_and_exit():
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    for mode in ("gc", "exit"):
        r = subprocess.run([sys.executable, "-c", _CHILD, mode], cwd=str(HERE),
                           env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        line = [x for x in r.stdout.splitlines() if x.startswith(("GC", "EXIT"))][-1]
        nums = [int(x) for x in line.replace("[", " ").replace("]", " ")
                .replace(",", " ").split()[1:]]
        if mode == "gc":
            pids, left = nums[:2], nums[2:]
            assert len(pids) == 2 and not left, line
        else:
            pids = nums
            assert len(pids) == 2, line
        time.sleep(0.2)
        assert not [p for p in pids if _pid_alive(p)], (mode, line)


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("%d cores" % (os.cpu_count() or 0))
    parts.append("seed %d, contract %s" % (SEED, CONTRACT))
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
            _alarm(1800)
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc) or traceback.format_exc()
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        finally:
            signal.alarm(0)
        results.append((name, status, time.perf_counter() - start, detail))
    if "cell" in _CELL:
        _CELL["cell"].destroy()
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        head = lines[0] if lines else ""
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, head)).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
