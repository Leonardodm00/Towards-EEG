"""Smoke test for hoc_qc (the S1 propagation gate).

Runs standalone (python3 test_hoc_qc.py) or under pytest. Needs only numpy and
pandas: the LFPy cell is a STUB with hand-planted vmem, so every expected value
is a number written down in this file rather than something the module computed.
The real-NEURON path is exercised only if LFPy imports, and skipped otherwise,
exactly as test_morphology_exporter::E8 does.

The synthetic cell (allseclist order, which is what LFPy flattens in):

    idx  0        soma[0]   nseg 1   root
    idx  1,2,3    dend[0]   nseg 3   child of soma[0]
    idx  4,5      dend[1]   nseg 2   child of dend[0]
    idx  6,7      dend[2]   nseg 2   child of dend[0]
    idx  8        axon[0]   nseg 1   child of soma[0]

    totnsegs = 9

Planted peak deflections, monotone decreasing away from the soma:

    soma[0]  5.0
    dend[0]  4.0  3.0  2.5
    dend[1]  2.0  1.5
    dend[2]  2.2  1.0
    axon[0]  3.5

dend[2] proximal (2.2) is deliberately LARGER than dend[1] proximal (2.0) while
both are below dend[0] distal (2.5): sibling order must not matter, only the
parent-child relation does.

Coverage
  Q1  peak_deflection: baseline is the last pre-stimulus sample, dv is exact
  Q2  check_finite catches NaN and inf and reports the first offender
  Q3  check_connected passes a healthy cell and catches a planted orphan
  Q4  check_monotone: healthy passes; an across-section increase is caught; a
      within-section increase INSIDE the loose tolerance is tolerated and one
      outside it is caught
  Q5  check_soma window, both sides
  Q6  section_slices reproduces the planted runs; non-contiguous raises
  Q7  check_propagation verdicts: pass / fail(orphan) / fail(NaN, early out) /
      pass_low_confidence(non-monotone)
  Q8  simulate_step wiring: stimulus kwargs forwarded, rec_vmem=True requested
  Q9  a section absent from the section table is reported, not fatal
  Q10 passive_cell_factory rejects non-positive Rm, cm, Ra
  Q11 gate_hoc returns the live cell when asked and releases it when not
  Q12 REAL NEURON (skipped without LFPy): a two-section hoc passes the gate
"""

import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import hoc_qc as hq


E_PAS = -70.0
DV_PLANTED = np.array([5.0,                  # soma[0]
                       4.0, 3.0, 2.5,        # dend[0]
                       2.0, 1.5,             # dend[1]
                       2.2, 1.0,             # dend[2]
                       3.5])                 # axon[0]
SEC_OF_IDX = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4])
SEC_NAMES = ["soma[0]", "dend[0]", "dend[1]", "dend[2]", "axon[0]"]
NSEG = [1, 3, 2, 2, 1]
LENGTHS = [10.0, 60.0, 40.0, 40.0, 30.0]

DELAY_MS = 2.5
TVEC = np.arange(10, dtype=float)            # 0..9 ms, dt = 1 ms


# --------------------------------------------------------------------------- #
#  Stubs                                                                       #
# --------------------------------------------------------------------------- #
class _StubSec(object):
    def __init__(self, name, L, nseg):
        self._name = name
        self.L = float(L)
        self.nseg = int(nseg)

    def name(self):
        return self._name


class _StubCell(object):
    """Minimal LFPy.Cell surface: allseclist, totnsegs, vmem, tvec, get_idx."""

    def __init__(self, dv=None, names=None, simulate_hook=None):
        names = names or SEC_NAMES
        self.allseclist = [_StubSec(n, L, k)
                           for n, L, k in zip(names, LENGTHS, NSEG)]
        self.totnsegs = int(sum(NSEG))
        self._dv = DV_PLANTED.copy() if dv is None else np.asarray(dv, float)
        self._simulate_hook = simulate_hook
        self.simulate_calls = []
        self.vmem = None
        self.tvec = None
        self._materialise()

    def _materialise(self):
        v = np.full((self.totnsegs, TVEC.size), E_PAS, dtype=float)
        post = TVEC >= DELAY_MS
        v[:, post] = E_PAS + self._dv[:, None]
        self.vmem = v
        self.tvec = TVEC.copy()

    def get_idx(self, name):
        if name == "soma":
            return np.array([0])
        raise KeyError(name)

    def simulate(self, **kwargs):
        self.simulate_calls.append(dict(kwargs))
        if self._simulate_hook is not None:
            self._simulate_hook(self)


def _section_table(names=None):
    names = names or SEC_NAMES
    rows = [(0, "soma", 0, -1), (1, "dend", 0, 0), (2, "dend", 1, 1),
            (3, "dend", 2, 1), (4, "axon", 0, 0)]
    return pd.DataFrame(rows, columns=["section_id", "array", "type_idx",
                                       "parent_sec_id"])


def _parent_of():
    import synapse_redirect_audit as sra
    return sra.build_section_tree(_section_table())


# --------------------------------------------------------------------------- #
#  Q1                                                                          #
# --------------------------------------------------------------------------- #
def test_q1_peak_deflection():
    cell = _StubCell()
    v0, dv = hq.peak_deflection(cell.vmem, cell.tvec, DELAY_MS)
    assert np.allclose(v0, E_PAS), v0
    assert np.allclose(dv, DV_PLANTED), dv
    # the baseline must be the LAST pre-stimulus sample, not the first
    shifted = cell.vmem.copy()
    shifted[:, 0] = E_PAS - 5.0                    # a spurious early excursion
    v0b, dvb = hq.peak_deflection(shifted, cell.tvec, DELAY_MS)
    assert np.allclose(v0b, E_PAS), v0b
    # too short a delay is a caller error, not a silent baseline of zero samples
    try:
        hq.peak_deflection(cell.vmem, cell.tvec, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative baseline window should raise")


# --------------------------------------------------------------------------- #
#  Q2                                                                          #
# --------------------------------------------------------------------------- #
def test_q2_finite():
    cell = _StubCell()
    ok, rep = hq.check_finite(cell.vmem)
    assert ok and rep["n_nonfinite"] == 0, rep

    bad = cell.vmem.copy()
    bad[6, 4] = np.nan
    bad[6, 5] = np.inf
    ok, rep = hq.check_finite(bad)
    assert not ok, rep
    assert rep["n_nonfinite"] == 2, rep
    assert rep["n_compartments_affected"] == 1, rep
    assert rep["first_bad_compartment"] == 6, rep
    assert rep["first_bad_sample"] == 4, rep


# --------------------------------------------------------------------------- #
#  Q3                                                                          #
# --------------------------------------------------------------------------- #
def test_q3_connected():
    ok, rep = hq.check_connected(DV_PLANTED)
    assert ok, rep
    assert math.isclose(rep["dv_min_mV"], 1.0), rep
    assert math.isclose(rep["dv_max_mV"], 5.0), rep

    orphan = DV_PLANTED.copy()
    orphan[6] = 0.0                                # dend[2] detached
    orphan[7] = 0.0
    ok, rep = hq.check_connected(orphan)
    assert not ok, rep
    assert rep["n_zero_deflection"] == 2, rep
    assert rep["zero_deflection_idx"] == [6, 7], rep


# --------------------------------------------------------------------------- #
#  Q4                                                                          #
# --------------------------------------------------------------------------- #
def test_q4_monotone():
    pof = _parent_of()

    ok, rep = hq.check_monotone(DV_PLANTED, SEC_OF_IDX, SEC_NAMES, pof)
    assert ok, rep["worst_violations"]
    assert rep["n_unknown_sections"] == 0, rep

    # across: dend[1] proximal jumps above dend[0] distal (2.5)
    bad = DV_PLANTED.copy()
    bad[4] = 3.0                                   # 3.0 > 2.5*1.05 + 1e-3
    ok, rep = hq.check_monotone(bad, SEC_OF_IDX, SEC_NAMES, pof)
    assert not ok and rep["n_violations"] == 1, rep
    v = rep["worst_violations"][0]
    assert v["kind"] == "across" and v["section"] == "dend[1]", v
    assert v["parent"] == "dend[0]", v

    # within, INSIDE the loose tolerance: 3.0 -> 3.05, bound is 3.0*1.05+1e-3
    tol_ok = DV_PLANTED.copy()
    tol_ok[3] = 3.05
    tol_ok[4] = 2.0
    ok, rep = hq.check_monotone(tol_ok, SEC_OF_IDX, SEC_NAMES, pof)
    assert ok, rep["worst_violations"]

    # within, OUTSIDE it
    tol_bad = DV_PLANTED.copy()
    tol_bad[3] = 3.4
    ok, rep = hq.check_monotone(tol_bad, SEC_OF_IDX, SEC_NAMES, pof)
    assert not ok, rep
    kinds = [v["kind"] for v in rep["worst_violations"]]
    assert "within" in kinds, rep["worst_violations"]


# --------------------------------------------------------------------------- #
#  Q5                                                                          #
# --------------------------------------------------------------------------- #
def test_q5_soma_window():
    ok, rep = hq.check_soma(DV_PLANTED, 0)
    assert ok and math.isclose(rep["dv_soma_mV"], 5.0), rep

    huge = DV_PLANTED.copy()
    huge[0] = 5000.0
    ok, _ = hq.check_soma(huge, 0)
    assert not ok

    tiny = DV_PLANTED.copy()
    tiny[0] = 1e-6
    ok, _ = hq.check_soma(tiny, 0)
    assert not ok

    try:
        hq.check_soma(DV_PLANTED, 99)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range soma index should raise")


# --------------------------------------------------------------------------- #
#  Q6                                                                          #
# --------------------------------------------------------------------------- #
def test_q6_section_slices():
    sl = hq.section_slices(SEC_OF_IDX)
    assert sl == {0: (0, 1), 1: (1, 4), 2: (4, 6), 3: (6, 8), 4: (8, 9)}, sl

    try:
        hq.section_slices([0, 1, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous section run should raise")


# --------------------------------------------------------------------------- #
#  Q7                                                                          #
# --------------------------------------------------------------------------- #
def test_q7_verdicts():
    pof = _parent_of()

    good = hq.check_propagation(_StubCell(), pof, baseline_end_ms=DELAY_MS)
    assert good["qc_status"] == hq.QC_PASS, good
    assert good["reasons"] == [], good
    assert good["totnsegs"] == 9 and good["n_sections"] == 5, good
    assert math.isclose(good["cable_length_um"], sum(LENGTHS)), good

    orphan_dv = DV_PLANTED.copy()
    orphan_dv[6:8] = 0.0
    orphan = hq.check_propagation(_StubCell(dv=orphan_dv), pof,
                                  baseline_end_ms=DELAY_MS)
    assert orphan["qc_status"] == hq.QC_FAIL, orphan
    assert any(r.startswith("zero_deflection") for r in orphan["reasons"]), orphan

    def _nan_hook(c):
        c.vmem[3, 7] = np.nan
    nan_cell = _StubCell(simulate_hook=None)
    nan_cell.vmem[3, 7] = np.nan
    nan = hq.check_propagation(nan_cell, pof, baseline_end_ms=DELAY_MS)
    assert nan["qc_status"] == hq.QC_FAIL, nan
    assert "C2_connected" not in nan, "C1 must short-circuit before dv is used"

    nonmono_dv = DV_PLANTED.copy()
    nonmono_dv[4] = 3.0
    nm = hq.check_propagation(_StubCell(dv=nonmono_dv), pof,
                              baseline_end_ms=DELAY_MS)
    assert nm["qc_status"] == hq.QC_LOW, nm
    assert any(r.startswith("monotonicity") for r in nm["reasons"]), nm

    # an unsimulated cell is a CALLER error and must raise, not return a verdict
    blank = _StubCell()
    blank.vmem = None
    try:
        hq.check_propagation(blank, pof)
    except ValueError:
        pass
    else:
        raise AssertionError("unsimulated cell should raise")


# --------------------------------------------------------------------------- #
#  Q8                                                                          #
# --------------------------------------------------------------------------- #
def test_q8_simulate_step_wiring():
    seen = {}

    def _stub_factory(cell, idx, amp_nA, delay_ms, dur_ms):
        seen.update(idx=idx, amp_nA=amp_nA, delay_ms=delay_ms, dur_ms=dur_ms)
        return object()

    cell = _StubCell()
    stim = hq.simulate_step(cell, amp_nA=0.25, delay_ms=7.0, dur_ms=120.0,
                            stim_factory=_stub_factory)
    assert stim is not None
    assert seen == {"idx": 0, "amp_nA": 0.25, "delay_ms": 7.0,
                    "dur_ms": 120.0}, seen
    assert cell.simulate_calls == [{"rec_vmem": True}], cell.simulate_calls

    # an explicit idx overrides the soma lookup
    hq.simulate_step(cell, idx=4, stim_factory=_stub_factory)
    assert seen["idx"] == 4, seen


# --------------------------------------------------------------------------- #
#  Q9                                                                          #
# --------------------------------------------------------------------------- #
def test_q9_unknown_section_is_soft():
    pof = _parent_of()
    names = list(SEC_NAMES)
    names[3] = "dend[99]"                          # present in hoc, not in table
    cell = _StubCell(names=names)
    rep = hq.check_propagation(cell, pof, baseline_end_ms=DELAY_MS)
    assert rep["qc_status"] == hq.QC_LOW, rep
    assert any(r.startswith("sections_absent") for r in rep["reasons"]), rep
    assert rep["C3_monotone"]["n_unknown_sections"] == 1, rep["C3_monotone"]
    assert rep["C3_monotone"]["unknown_sections"] == ["dend[99]"], rep


# --------------------------------------------------------------------------- #
#  Q10                                                                         #
# --------------------------------------------------------------------------- #
def test_q10_factory_guards():
    for kw in (dict(Rm=0.0), dict(Rm=-1.0)):
        try:
            hq.passive_cell_factory("nofile.hoc", cm=0.5, Ra=200.0, **kw)
        except ValueError:
            pass
        except ImportError:
            raise AssertionError("guard must fire BEFORE importing LFPy")
        else:
            raise AssertionError("non-positive Rm should raise: %r" % kw)

    for kw in (dict(cm=0.0, Ra=200.0), dict(cm=0.5, Ra=-3.0)):
        try:
            hq.passive_cell_factory("nofile.hoc", **kw)
        except ValueError:
            pass
        except ImportError:
            raise AssertionError("guard must fire BEFORE importing LFPy")
        else:
            raise AssertionError("non-positive cm/Ra should raise: %r" % kw)

    assert math.isclose(1.0 / hq.DEFAULT_RM_OHM_CM2, 4.1667e-5, rel_tol=1e-3)


# --------------------------------------------------------------------------- #
#  Q11                                                                         #
# --------------------------------------------------------------------------- #
def test_q11_gate_hoc_cell_lifetime():
    released = {"n": 0}

    class _Releasable(_StubCell):
        def __del__(self):
            released["n"] += 1

    def _factory(hoc_path, **kwargs):
        return _Releasable()

    def _stim(cell, idx, amp_nA, delay_ms, dur_ms):
        return object()

    rep, cell = hq.gate_hoc("ignored.hoc", _section_table(), cm=0.5, Ra=200.0,
                            delay_ms=DELAY_MS, cell_factory=_factory,
                            stim_factory=_stim, keep_cell=True)
    assert rep["qc_status"] == hq.QC_PASS, rep
    assert cell is not None
    assert rep["passive"]["Rm_ohm_cm2"] == hq.DEFAULT_RM_OHM_CM2, rep["passive"]
    assert math.isclose(rep["passive"]["g_pas_S_cm2"],
                        1.0 / hq.DEFAULT_RM_OHM_CM2), rep["passive"]
    assert rep["stimulus"]["amp_nA"] == hq.DEFAULT_AMP_NA, rep["stimulus"]

    before = released["n"]
    rep2, cell2 = hq.gate_hoc("ignored.hoc", _section_table(), cm=0.5, Ra=200.0,
                              delay_ms=DELAY_MS, cell_factory=_factory,
                              stim_factory=_stim, keep_cell=False)
    assert cell2 is None
    assert released["n"] > before, "keep_cell=False must release the cell"

    # a factory that explodes yields a verdict, not a traceback
    def _boom(hoc_path, **kwargs):
        raise RuntimeError("no such morphology")

    try:
        hq.gate_hoc("ignored.hoc", _section_table(), cm=0.5, Ra=200.0,
                    cell_factory=_boom, stim_factory=_stim)
    except RuntimeError:
        pass                        # construction failure propagates, by design
    else:
        raise AssertionError("factory failure should propagate")

    # but a failure INSIDE simulate/judge is caught and reported
    class _BadSim(_StubCell):
        def simulate(self, **kwargs):
            raise RuntimeError("nrn blew up")

    rep3, cell3 = hq.gate_hoc("ignored.hoc", _section_table(), cm=0.5, Ra=200.0,
                              cell_factory=lambda p, **k: _BadSim(),
                              stim_factory=_stim)
    assert rep3["qc_status"] == hq.QC_FAIL and cell3 is None, rep3
    assert rep3["reasons"] == ["qc_raised:RuntimeError"], rep3


# --------------------------------------------------------------------------- #
#  Q12 -- real NEURON, skipped when LFPy is absent                             #
# --------------------------------------------------------------------------- #
HOC_TWO_SECTION = """\
create soma[1]
create dend[1]
soma[0] {
    pt3dclear()
    pt3dadd(0, 0, -5, 10)
    pt3dadd(0, 0, 5, 10)
}
dend[0] {
    pt3dclear()
    pt3dadd(0, 0, 5, 2)
    pt3dadd(0, 0, 205, 2)
}
connect dend[0](0), soma[0](1)
"""


def test_q12_real_neuron():
    try:
        import LFPy                                            # noqa: F401
    except Exception:                                          # noqa: BLE001
        print("    Q12 skipped: LFPy/NEURON not installed")
        return

    tmp = tempfile.mkdtemp(prefix="hoc_qc_")
    try:
        path = os.path.join(tmp, "two_section.hoc")
        with open(path, "w", newline="\n") as fh:
            fh.write(HOC_TWO_SECTION)
        table = pd.DataFrame([(0, "soma", 0, -1), (1, "dend", 0, 0)],
                             columns=["section_id", "array", "type_idx",
                                      "parent_sec_id"])
        rep, cell = hq.gate_hoc(path, table, cm=0.5, Ra=200.0, keep_cell=False)
        assert rep["qc_status"] == hq.QC_PASS, rep
        assert rep["dv_min_mV"] > 0.0, rep
        assert rep["C4_soma"]["dv_soma_mV"] > rep["dv_min_mV"], rep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main():
    def _order(item):
        name = item[0]
        digits = name.split("_")[1][1:]
        return int(digits) if digits.isdigit() else 999

    tests = [v for _, v in sorted(
        [(k, v) for k, v in globals().items() if k.startswith("test_")],
        key=_order)]
    passed = 0
    for fn in tests:
        fn()
        passed += 1
        print("  ok  %s" % fn.__name__)
    print("%d/%d hoc_qc tests passed" % (passed, len(tests)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
