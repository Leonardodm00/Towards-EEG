"""Spec-derived tests of Block 19 (passive_long_step_training.py: the pool of
build_multi_protocol_loss; SPEC 2.9 (c) and the Block 19 entry). Integration
tester, 2026-11-24.

Oracle of 2.9 (c): for every q and every W >= 1, L_W(q) == L_1(q) exactly
(float64 equality), penalties included; W = 1 is the serial code. The serial
loss is the reference (it is the code the requirement names); the pooled loss
is compared with it with `==`, never with a tolerance. Most checks run on the
duck-typed analytic stub cell of the existing smokes (no NEURON: the pool's
arithmetic and bookkeeping do not depend on the simulator); one check runs on a
NEURON ball-and-stick cell with I_h.

Inputs only (synthetic bundles, the stub cell) come from smoke_ls_protocol.
"""
import multiprocessing
import os
import warnings

import numpy as np
import pytest

import param_spec as PS
import passive_long_step_training as plst
from smoke_ls_protocol import V_REST, StubCell, ls_bundle, ss_bundle

DT = 0.1


def _train():
    return [ss_bundle(-200.0), ss_bundle(-150.0), ls_bundle(-30.0, sag=0.1), ls_bundle(-70.0, sag=0.3),
            ls_bundle(-110.0, sag=0.5)]


def _pts(n=6, seed=7):
    rng = np.random.default_rng(seed)
    lo = np.log([0.3, 1e3, 50.0])
    hi = np.log([3.0, 1e5, 1e3])
    pts = [tuple(lo + rng.uniform(0, 1, 3) * (hi - lo)) for _ in range(n)]
    pts += [tuple(lo), tuple(hi)]
    return [tuple(float(x) for x in p) for p in pts]


def _build(cell, tb, W, **kw):
    base = dict(ls_window_mode="step", dt_brief_ms=DT, dt_long_ms=DT)
    base.update(kw)
    return plst.build_multi_protocol_loss(cell, tb, V_REST, n_workers=W, **base)


def _children():
    return [p for p in multiprocessing.active_children()]


@pytest.mark.parametrize("weighting,extra", [
    ("relative", {}), ("equal", {}), ("ls_amplitude", dict(ls_sample_weight="peak_decay")),
    ("custom", dict(weights=[1.0, 2.0, 0.5, 3.0, 0.25])),
    ("some_unknown_name", {}),                 # falls through to 'equal' in the serial code (Block 14 as built)
])
@pytest.mark.parametrize("W", [2, 4, 9])
def test_exact_equality_every_weighting(weighting, extra, W):
    cell = StubCell()
    tb = _train()
    l1 = _build(cell, tb, 1, weighting=weighting, **extra)
    lw = _build(cell, tb, W, weighting=weighting, **extra)
    try:
        v1 = [l1(*q) for q in _pts()]
        vw = [lw(*q) for q in _pts()]
        assert vw == v1, (weighting, W)
        assert all(np.isfinite(v1)) and max(v1) < 1e6
        assert lw.n_workers == W and l1.n_workers == 1
        # at most min(W, n_bundles) workers (2.9 (c), the entry)
        assert len(_children()) <= min(W, len(tb))
    finally:
        lw.close()
    assert _children() == []


def test_default_dt_none_resolved_in_the_parent_after_the_fork():
    """dt None: the serial loop resolves the module defaults at each call. A change of the
    module default AFTER the pool forked must be seen by the pooled loss too."""
    cell = StubCell()
    tb = _train()
    old = (plst.DEFAULT_DT_BRIEF_MS, plst.DEFAULT_DT_LONG_MS)
    l1 = _build(cell, tb, 1, dt_brief_ms=None, dt_long_ms=None)
    l2 = _build(cell, tb, 2, dt_brief_ms=None, dt_long_ms=None)
    try:
        q = _pts()[0]
        assert l2(q[0], q[1], q[2]) == l1(*q)          # pool created with the old defaults
        plst.DEFAULT_DT_BRIEF_MS, plst.DEFAULT_DT_LONG_MS = 0.05, 0.2
        a, b = l1(*q), l2(*q)
        assert a == b
        plst.DEFAULT_DT_BRIEF_MS, plst.DEFAULT_DT_LONG_MS = old
        assert l1(*q) != a or old == (0.05, 0.2)       # the default matters (the check is not vacuous)
    finally:
        plst.DEFAULT_DT_BRIEF_MS, plst.DEFAULT_DT_LONG_MS = old
        l2.close()


def test_parent_cell_at_q_and_spec_none_is_passive_3d():
    tb = _train()
    c1, c2 = StubCell(), StubCell()
    l1 = _build(c1, tb, 1)
    l2 = _build(c2, tb, 2)                     # spec None -> param_spec.PASSIVE_3D in both paths
    try:
        q = _pts()[3]
        assert l2(*q) == l1(*q)
        th = PS.PASSIVE_3D.to_physical(q)
        for c in (c1, c2):
            assert (c.Cm, c.Rm, c.Ra, c.e) == (th["Cm"], th["Rm"], th["Ra"], V_REST)
    finally:
        l2.close()


def test_no_bundles_no_pool_same_value():
    cell = StubCell()
    l1 = _build(cell, [], 1)
    l2 = _build(cell, [], 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # numpy's mean of an empty array
        a, b = l1(*_pts()[0]), l2(*_pts()[0])
    assert (np.isnan(a) and np.isnan(b)) or a == b
    assert _children() == []
    l2.close()


class _Boom(Exception):
    pass


class _BadSpec:
    """A spec whose apply raises in the parent (not a dt failure): 1e6 in both paths."""
    names = ("Cm", "Rm", "Ra")

    def apply_q(self, cell, q, v):
        raise _Boom("no")


def test_parent_apply_failure_is_penalty_both_paths():
    cell = StubCell()
    tb = _train()
    l1 = _build(cell, tb, 1, spec=_BadSpec())
    l2 = _build(cell, tb, 2, spec=_BadSpec())
    assert l1(0.0, 9.0, 4.0) == l2(0.0, 9.0, 4.0) == 1e6
    l2.close()


def test_nan_bundle_order_decides():
    """A NaN in the second bundle's window: 1e6 for every W (the serial rule)."""
    cell = StubCell()
    t = np.arange(int(round(0.9 * 1e4))) / 1e4
    v = np.full_like(t, V_REST)
    v[2000:2100] = np.nan
    tb = _train()
    tb.insert(1, ls_bundle(-50.0, v=v))
    l1 = _build(cell, tb, 1)
    l3 = _build(cell, tb, 3)
    try:
        q = _pts()[1]
        assert l1(*q) == 1e6 and l3(*q) == 1e6
    finally:
        l3.close()


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, "2", None, 2.0])
def test_n_workers_refusals(bad):
    with pytest.raises(ValueError):
        plst.build_multi_protocol_loss(StubCell(), _train(), V_REST, n_workers=bad)


def test_n_workers_numpy_integer_accepted():
    l = plst.build_multi_protocol_loss(StubCell(), _train(), V_REST, ls_window_mode="step", dt_brief_ms=DT,
                                       dt_long_ms=DT, n_workers=np.int64(2))
    try:
        assert l.n_workers == 2 and isinstance(l.n_workers, int)
        assert np.isfinite(l(*_pts()[0]))
    finally:
        l.close()


def test_close_idempotent_and_reopen():
    cell = StubCell()
    tb = _train()
    l1 = _build(cell, tb, 1)
    l2 = _build(cell, tb, 2)
    q = _pts()[2]
    assert _children() == []                     # lazy: no pool before the first call
    v = l2(*q)
    kids = _children()
    assert 1 <= len(kids) <= 2 and all(k.pid != os.getpid() for k in kids)
    l2.close()
    l2.close()
    assert _children() == []
    assert l2(*q) == v == l1(*q)
    l2.close()
    l1.close()                                   # a no-op for W = 1


def test_one_live_pool_per_process():
    cell = StubCell()
    tb = _train()
    a1, b1 = _build(cell, tb[:3], 1), _build(cell, tb[2:], 1)
    a2, b2 = _build(cell, tb[:3], 2), _build(cell, tb[2:], 2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for q in _pts(3):
                assert a2(*q) == a1(*q)
                assert len(_children()) <= 2
                assert b2(*q) == b1(*q)
                assert len(_children()) <= 2
    finally:
        a2.close()
        b2.close()


# ---------------------------------------------------------------------------------------------- NEURON

def test_neuron_ih_cell_exact_and_cell_state():
    pytest.importorskip("neuron")
    import ih_mechanism as IM
    import passive_fitting_hpc_fixed as mono
    from synthetic_ground_truth import write_ball_and_stick_swc
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="tb19_"))
    swc = write_ball_and_stick_swc(tmp / "bas.swc", soma_r_um=10.0, dend_len_um=200.0, dend_r_um=1.0,
                                   apic_len_um=300.0, apic_r_um=1.2, step_um=25.0)
    cell = mono.build_neuron_model(swc, F=1.9)
    IM.attach_ih(cell, IM.IhSpec(mechanism="Ih_human", distribution="uniform"))
    spec = PS.make_ih_spec()
    lo = np.array([d.low for d in spec.as_skopt_dimensions()])
    hi = np.array([d.high for d in spec.as_skopt_dimensions()])
    rng = np.random.default_rng(1)
    pts = [tuple(float(x) for x in lo + rng.uniform(0.1, 0.9, lo.size) * (hi - lo)) for _ in range(3)]
    tb = _train()
    kw = dict(weighting="ls_amplitude", ls_sample_weight="peak_decay", spec=spec)
    l1 = _build(cell, tb, 1, **kw)
    l2 = _build(cell, tb, 2, **kw)
    try:
        for q in pts:
            a = l1(*q)
            s1 = (cell.soma[0](0.5).cm, cell.soma[0](0.5).g_pas, cell.soma[0](0.5).e_pas,
                  cell.soma[0](0.5).Ih_human.gIhbar)
            b = l2(*q)
            s2 = (cell.soma[0](0.5).cm, cell.soma[0](0.5).g_pas, cell.soma[0](0.5).e_pas,
                  cell.soma[0](0.5).Ih_human.gIhbar)
            assert a == b and a < 1e6
            assert s1 == s2                       # the parent's cell is left at q (the serial side effect)
    finally:
        l2.close()
