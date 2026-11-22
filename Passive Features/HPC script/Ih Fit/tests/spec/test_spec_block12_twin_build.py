"""Spec-derived tests of Block 12 (twin_build.py; SPEC 2.7 (c), (d), (f), (h) and
the Block 12 entry with amendment 6). Integration tester, 2026-11-20.

Fixture: Block 3's writer-like archive cell (smoke_noise_windows.write_fixture)
with a ball-and-stick reconstruction.swc; the twins' table is Block 5's own
output on hand-written noise tables (so the Block 5 -> Block 12 boundary is
exercised as the pipeline uses it). Expected values: the noise of (TW1) /
(TW2) is written out here from the seed rule of 2.7 (e) and the recursions of
(L2') (independent of noise_law and twin_noise); A_p, d_p from the fixture's
planted peak_pA / stim_duration_s; tau_ms = Rm Cm 1e-3 (amendment 6 (iii)).
"""
import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("neuron")

import twin_build as tb          # noqa: E402
import twin_table as tt          # noqa: E402
from twin_config import TwinConfig   # noqa: E402

REAL_ID = 900000003              # the fixture writer's specimen id
TWIN_ID = 900000777
FS = 50000.0
LS_AMPS = (-110.0, -50.0, 0.0, 30.0, 60.0)       # sweeps 54..58; 56 (0 pA) no step
THETA_P = dict(Cm=0.9, Rm=18000.0, Ra=120.0)
THETA_IH = dict(THETA_P, gbar=1.0e-4, dv_h=1.5, kappa_tau=1.2)
CFG = TwinConfig()


def _units(entropy, n):
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)


def _law_trace(row, dt, fast, mid, slow):
    from scipy.signal import lfilter   # the AR(1) of (L2') with its stationary start, written out
    out = row["sigma_f_mV"] * fast
    for sig, tau, xi in ((row["sigma_m_mV"], row["tau_m_s"], mid), (row["sigma_s_mV"], row["tau_s_s"], slow)):
        r = math.exp(-dt / tau)
        eta = sig * math.sqrt(1 - r * r) * xi.copy()
        eta[0] = sig * xi[0]
        out = out + lfilter([1.0], [1.0, -r], eta)
    return out


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    import csv
    import smoke_noise_windows as sw
    import synthetic_ground_truth as sgt
    root = tmp_path_factory.mktemp("b12")
    real = root / "real" / ("specimen_%d" % REAL_ID)
    truth = sw.write_fixture(real, np.random.default_rng(11), ls_amps=LS_AMPS)
    sgt.write_ball_and_stick_swc(real / "reconstruction.swc")
    # spike test: the 60 pA sweep (index 4 -> sweep 58) "spiked"
    with np.load(real / "ls_sweeps.npz") as z:
        v58 = np.array(z["v_4"])
    spike_fn = lambda v, fs: "action_potential" if np.array_equal(v, v58) else ""   # noqa: E731
    prot = tt.load_cell_protocol(real, spike_reason_fn=spike_fn)
    cells_csv, sweeps_csv = root / "c.csv", root / "s.csv"
    with open(cells_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["specimen_id", "status", "reason", "noise_sigma_f_mV", "noise_sigma_m_mV", "noise_tau_m_ms",
                    "noise_sigma_s_mV", "noise_tau_s_ms"])
        w.writerow([REAL_ID, "ok", "", 0.034, 0.017, 0.07, 0.08, 210.0])
    with open(sweeps_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["specimen_id", "noise_sweep_protocol", "noise_sweep_number", "noise_tilt_mV_per_s",
                    "noise_sweep_sigma_f_mV", "noise_sweep_sigma_m_mV", "noise_sweep_sigma_s_mV", "noise_sweep_flags"])
        w.writerow([REAL_ID, "SS", 92, 0.021, 0.033, 0.018, 0.09, ""])
        w.writerow([REAL_ID, "SS", 93, -0.013, 0.036, 0.016, 0.07, ""])
        for k, s in enumerate((54, 55, 57, 58)):
            w.writerow([REAL_ID, "LS", s, -0.05 + 0.03 * k, 0.03 + 0.001 * k, 0.015, 0.06 + 0.01 * k, ""])
    cells, sweeps = tt.read_noise_tables(cells_csv, sweeps_csv)
    table = tt.build_twin_sweep_table([prot], cells, sweeps)
    return dict(root=root, real=real, truth=truth, prot=prot, table=table, spike_fn=spike_fn)


def _spec(conf, twin_id=TWIN_ID, theta=None, v_rest=-60.0):
    th = theta if theta is not None else (dict(THETA_IH) if tb.HAS_IH[conf] else dict(THETA_P))
    return tb.TwinSpec(twin_id=twin_id, twin_of_specimen_id=REAL_ID, configuration=conf, theta=th,
                       v_rest_mV=v_rest, F=1.9)


def _model(conf):
    import ih_mechanism as IM
    import param_spec as PS
    return IM.IhSpec(), (PS.make_ih_spec().apply if tb.HAS_IH[conf] else PS.make_passive_spec().apply)


def _clear():
    import synthetic_ground_truth as sgt
    sgt._clear_neuron_sections()


@pytest.fixture(scope="module")
def built(fx):
    """One 'full' twin cell, the traces with and without noise."""
    _clear()
    ih_spec, apply = _model("full")
    cell = tb.build_twin_cell(fx["real"] / "reconstruction.swc", _spec("full"), ih_spec=ih_spec, apply_theta=apply)
    noisy = tb.simulate_twin(cell, _spec("full"), fx["prot"], fx["table"])
    clean = tb.simulate_twin(cell, _spec("full"), fx["prot"], fx["table"], with_noise=False)
    other = tb.simulate_twin(cell, _spec("full", twin_id=TWIN_ID + 1), fx["prot"], fx["table"])
    return dict(cell=cell, noisy=noisy, clean=clean, other=other)


def test_traces_shapes_and_polarity_means(fx, built):
    tr, truth = built["noisy"], fx["truth"]
    K = truth["K"]
    assert tr.ss_v_mV.shape == (K, 10500) and tr.ss_v_mV.dtype == np.float64
    assert_array_equal(tr.ss_sweep_number, truth["sweeps"])
    assert_array_equal(tr.ss_polarity_is_dep, truth["pol"])
    assert_array_equal(tr.ss_peak_pA, np.where(truth["pol"], 200.0, -200.0))     # A_p = mean of +-200
    assert_allclose(tr.ss_stim_duration_s, 25 / FS, rtol=1e-15)
    # in_twin False sweep 58 absent; order of CellProtocol
    assert_array_equal(tr.ls_sweep_number, [54, 55, 57])
    assert_array_equal(tr.ls_amplitude_pA, [-110.0, -50.0, 30.0])
    assert all(np.all(np.isfinite(v)) for v in tr.ls_v_mV) and np.all(np.isfinite(tr.ss_v_mV))
    # one clean response per polarity: every clean window of a polarity identical
    cl = built["clean"].ss_v_mV
    for p in (True, False):
        rows = cl[truth["pol"] == p]
        assert np.all(rows == rows[0])
    # the SS current on samples 500 .. 524 (the archive's layout)
    on = np.flatnonzero(tr.ss_i_pA[0])
    assert on[0] == 500 and on[-1] == 524


def test_noise_composition_written_out(fx, built):
    """(TW1), (TW2): noisy - clean = the noise of 2.7 (c)-(e) written out here.
    1e-12 mV: float64 roundoff of mV-sized sums."""
    tr, cl, prot, table = built["noisy"], built["clean"], fx["prot"], fx["table"]
    dt = 1.0 / FS
    n_w = 10500
    for s in (92, 93):
        row = table[(table["protocol"] == "SS") & (table["sweep_number"] == s)].iloc[0]
        m = prot.ss_sweep_number == s
        pos = prot.ss_position[m]
        u = (pos[:, None] * prot.ss_period_samples + np.arange(n_w)[None, :]) * dt
        ubar = u.mean()
        diff = tr.ss_v_mV[m] - cl.ss_v_mV[m]
        for k, pk in enumerate(pos):
            e = _law_trace(row, dt, *_units([CFG.base_seed, TWIN_ID, 0, s, int(pk)], n_w))
            assert_allclose(diff[k], e + row["tilt_mV_per_s"] * (u[k] - ubar), rtol=0, atol=1e-12)
    for idx, s in enumerate(tr.ls_sweep_number):
        row = table[(table["protocol"] == "LS") & (table["sweep_number"] == s)].iloc[0]
        n = tr.ls_v_mV[idx].size
        e = _law_trace(row, dt, *_units([CFG.base_seed, TWIN_ID, 1, int(s)], n))
        exp = e + row["tilt_mV_per_s"] * (np.arange(n) - (n - 1) / 2) * dt
        assert_allclose(tr.ls_v_mV[idx] - cl.ls_v_mV[idx], exp, rtol=0, atol=1e-12)


def test_determinism_and_twin_id(built, fx):
    _clear()
    ih_spec, apply = _model("full")
    cell = tb.build_twin_cell(fx["real"] / "reconstruction.swc", _spec("full"), ih_spec=ih_spec, apply_theta=apply)
    again = tb.simulate_twin(cell, _spec("full"), fx["prot"], fx["table"])
    assert_array_equal(again.ss_v_mV, built["noisy"].ss_v_mV)
    for a, b in zip(again.ls_v_mV, built["noisy"].ls_v_mV):
        assert_array_equal(a, b)
    assert not np.array_equal(built["other"].ss_v_mV, built["noisy"].ss_v_mV)
    # the clean part does not depend on the twin id: other - noise(other) == clean
    d_other = built["other"].ss_v_mV - built["clean"].ss_v_mV
    d_this = built["noisy"].ss_v_mV - built["clean"].ss_v_mV
    assert not np.allclose(d_other, d_this)


@pytest.mark.parametrize("conf", ["full", "active_fit"])
def test_archive_round_trip_and_metadata(fx, conf, tmp_path):
    import passive_fitting_hpc_fixed as mono
    _clear()
    ih_spec, apply = _model(conf)
    d = tb.build_twin(_spec(conf), fx["real"], fx["table"], tmp_path / "grp", ih_spec=ih_spec, apply_theta=apply,
                      spike_reason_fn=fx["spike_fn"], provenance={"by": "tester"})
    _clear()
    meta = json.load(open(d / "metadata.json"))
    assert meta["specimen_id"] == TWIN_ID and meta["v_rest_mV"] == -60.0 and meta["ljp_correction_mV"] == 0.0
    th = THETA_IH if conf == "full" else THETA_P
    assert meta["tau_ms"] == pytest.approx(th["Rm"] * th["Cm"] * 1e-3, rel=1e-12)
    assert math.isfinite(meta["rin_MOhm"]) and meta["rin_MOhm"] > 0
    tw = meta["twin"]
    assert tw["theta"] == th and tw["configuration"] == conf and tw["twin_of_specimen_id"] == REAL_ID
    assert tw["has_ih"] == (conf == "full") and tw["ls_sweeps_left_out"] == {"58": "action_potential"}
    assert tw["provenance"] == {"by": "tester"} and abs(tw["rest_drift_mV"]) <= 1e-3
    assert [x["sweep_number"] for x in meta["ls_sweeps"]] == [54, 55, 57]
    with np.load(d / "ss_pulses.npz") as z:
        assert set(z.files) == set(tb.SS_NPZ_KEYS)
        v, pol = z["v"], z["polarity_is_dep"]
    cd = mono.load_cell_from_archive(d, load_depolarising_ls=True, verbose=False)
    assert cd.specimen_id == TWIN_ID and cd.v_rest_mV == -60.0
    for b in cd.square_subthreshold:
        m = pol == (b.polarity == "dep")
        assert_allclose(b.v_mV, v[m].mean(axis=0), rtol=0, atol=1e-12)
        assert b.amplitude_pA == (200.0 if b.polarity == "dep" else -200.0)
    ls_amps = sorted(round(b.amplitude_pA, 6) for b in cd.long_square_subthreshold)
    assert set(ls_amps) <= {-110.0, -50.0, 30.0}


def test_active_fit_has_no_ih(fx):
    _clear()
    ih_spec, apply = _model("active_fit")
    cell = tb.build_twin_cell(fx["real"] / "reconstruction.swc", _spec("active_fit"), ih_spec=ih_spec,
                              apply_theta=apply)
    from neuron import h
    names = set()
    for sec in h.allsec():
        for seg in sec:
            for mech in seg:
                names.add(mech.name())
    assert not any(n.lower().startswith("ih") for n in names), names
    _clear()


@pytest.mark.parametrize("case", ["unknown_conf", "same_id", "keys_missing", "keys_extra", "nan_theta",
                                  "no_ih_spec", "nan_vrest", "drift"])
def test_cell_refusals(fx, case):
    _clear()
    ih_spec, apply = _model("full")
    spec = _spec("full")
    if case == "unknown_conf":
        spec = dataclasses.replace(spec, configuration="ih6")
    elif case == "same_id":
        spec = dataclasses.replace(spec, twin_id=REAL_ID)
    elif case == "keys_missing":
        spec = _spec("full", theta=dict(THETA_P))
    elif case == "keys_extra":
        spec = _spec("active_fit", theta=dict(THETA_IH))
    elif case == "nan_theta":
        spec = _spec("full", theta=dict(THETA_IH, Ra=float("nan")))
    elif case == "no_ih_spec":
        ih_spec = None
    elif case == "nan_vrest":
        spec = _spec("full", v_rest=float("nan"))
    elif case == "drift":
        def shifted(cell, theta, v):
            apply(cell, theta, v)
            from neuron import h
            for sec in h.allsec():
                for seg in sec:
                    if hasattr(seg, "pas"):
                        seg.pas.e = seg.pas.e + 1.0
        apply_used = shifted
    if case != "drift":
        apply_used = apply
    with pytest.raises(tb.TwinBuildError):
        tb.build_twin_cell(fx["real"] / "reconstruction.swc", spec, ih_spec=ih_spec, apply_theta=apply_used)
    _clear()


@pytest.mark.parametrize("case", ["n_windows", "amplitude", "no_rows", "in_twin"])
def test_table_refusals(fx, built, case):
    t = fx["table"].copy()
    if case == "n_windows":
        t.loc[t["sweep_number"] == 92, "n_windows"] += 1
    elif case == "amplitude":
        t.loc[t["sweep_number"] == 54, "amplitude_pA"] = -111.0
    elif case == "no_rows":
        t = t.iloc[0:0]
    elif case == "in_twin":
        t.loc[t["sweep_number"] == 58, "in_twin"] = True
    with pytest.raises(tb.TwinBuildError):
        tb.simulate_twin(built["cell"], _spec("full"), fx["prot"], t)


def test_existing_dir_refused(fx, tmp_path):
    (tmp_path / ("specimen_%d" % TWIN_ID)).mkdir()
    ih_spec, apply = _model("full")
    with pytest.raises(tb.TwinBuildError):
        tb.build_twin(_spec("full"), fx["real"], fx["table"], tmp_path, ih_spec=ih_spec, apply_theta=apply,
                      spike_reason_fn=fx["spike_fn"])


def test_swap_config(fx):
    """Directive audit, swappability: TwinConfig(base_seed=...) changes the noise
    and not the clean part; TwinConfig(neuron_dt_ms=0.05) changes the clean part
    by a step-size amount and the run completes (bound 0.2 mV: the largest
    difference sits on the 0.5 ms pulse edge, 10 steps at 0.05 ms; measured 0.06 mV)."""
    _clear()
    ih_spec, apply = _model("full")
    cell = tb.build_twin_cell(fx["real"] / "reconstruction.swc", _spec("full"), ih_spec=ih_spec, apply_theta=apply)
    a = tb.simulate_twin(cell, _spec("full"), fx["prot"], fx["table"])
    b = tb.simulate_twin(cell, _spec("full"), fx["prot"], fx["table"], cfg=TwinConfig(base_seed=1))
    c0 = tb.simulate_twin(cell, _spec("full"), fx["prot"], fx["table"], with_noise=False)
    c1 = tb.simulate_twin(cell, _spec("full"), fx["prot"], fx["table"], with_noise=False,
                          cfg=TwinConfig(neuron_dt_ms=0.05))
    assert not np.array_equal(a.ss_v_mV, b.ss_v_mV)
    d = np.max(np.abs(c1.ss_v_mV - c0.ss_v_mV))
    assert 0 < d < 0.2, d
    _clear()


# ===========================================================================
#  Block 18: a twin without noise (SPEC 2.8 (e), the Block 18 entry).
#  Integration tester, 2026-11-22. Expected values: the clean traces of
#  simulate_twin(with_noise=False) (the entry's reference, "bit for bit"); the
#  default build against twin_build.py as it stood at the base commit 216ecf1
#  (`git show`; TB12's reference); the metadata keys the entry lists.
# ===========================================================================
import importlib.util     # noqa: E402
import subprocess         # noqa: E402
import tempfile           # noqa: E402
from pathlib import Path  # noqa: E402

BASE18 = "216ecf13ccca455903b2cb05d740dd64f3feff08"


def _old_twin_build(_c={}):
    if "m" in _c:
        return _c["m"]
    code = Path(__file__).resolve().parents[2]
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=code, capture_output=True, text=True,
                         check=True).stdout.strip()
    rel = (code.relative_to(top) / "twin_build.py").as_posix()
    src = subprocess.run(["git", "show", "%s:%s" % (BASE18, rel)], cwd=top, capture_output=True, check=True).stdout
    p = Path(tempfile.mkdtemp()) / "twin_build_base.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("twin_build_base", p)
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["twin_build_base"] = m          # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(m)
    _c["m"] = m
    return m


def _read_archive(d):
    with np.load(d / "ss_pulses.npz") as z, np.load(d / "ls_sweeps.npz") as y:
        return {k: z[k].copy() for k in z.files}, {k: y[k].copy() for k in y.files}, \
            json.load(open(d / "metadata.json"))


@pytest.mark.parametrize("conf", ["full", "full_no_noise"])
def test_b18_no_noise_archive_equals_clean_traces(fx, built, conf, tmp_path):
    _clear()
    ih_spec, apply = _model(conf)
    d = tb.build_twin(_spec(conf), fx["real"], fx["table"], tmp_path / "g", ih_spec=ih_spec, apply_theta=apply,
                      spike_reason_fn=fx["spike_fn"], with_noise=False)
    _clear()
    ss, ls, meta = _read_archive(d)
    cl = built["clean"]                         # the 'full' cell's clean traces (same truth, same real cell)
    assert_array_equal(ss["v"], cl.ss_v_mV)
    assert_array_equal(ss["i_pA"], cl.ss_i_pA)
    for m, (v, i) in enumerate(zip(cl.ls_v_mV, cl.ls_i_pA)):
        assert_array_equal(ls["v_%d" % m], v)
        assert_array_equal(ls["i_%d" % m], i)
    tw = meta["twin"]
    assert tw["with_noise"] is False and tw["noise_laws_applied"] is False
    assert tw["configuration"] == conf and tw["has_ih"] is True
    # read back by the fitter's loader: the SS bundles are the clean polarity responses
    import passive_fitting_hpc_fixed as mono
    cd = mono.load_cell_from_archive(d, load_depolarising_ls=True, verbose=False)
    for b in cd.square_subthreshold:
        m = ss["polarity_is_dep"] == (b.polarity == "dep")
        assert_allclose(b.v_mV, cl.ss_v_mV[m][0], rtol=0, atol=1e-12)


def test_b18_with_noise_metadata_and_difference(fx, built, tmp_path):
    _clear()
    ih_spec, apply = _model("full")
    d = tb.build_twin(_spec("full"), fx["real"], fx["table"], tmp_path / "g", ih_spec=ih_spec, apply_theta=apply,
                      spike_reason_fn=fx["spike_fn"])
    _clear()
    ss, ls, meta = _read_archive(d)
    assert meta["twin"]["with_noise"] is True
    assert meta["twin"].get("noise_laws_applied", True) is not False
    assert_array_equal(ss["v"], built["noisy"].ss_v_mV)
    assert not np.array_equal(ss["v"], built["clean"].ss_v_mV)


def test_b18_default_reproduces_base_archive(fx, tmp_path):
    """TB12: the default (with_noise=True) gives the base commit's archive arrays bit for bit."""
    old = _old_twin_build()
    ih_spec, apply = _model("full")
    arrs = []
    for mod, sub in ((tb, "new"), (old, "old")):
        _clear()
        spec = mod.TwinSpec(twin_id=TWIN_ID, twin_of_specimen_id=REAL_ID, configuration="full",
                            theta=dict(THETA_IH), v_rest_mV=-60.0, F=1.9)
        d = mod.build_twin(spec, fx["real"], fx["table"], tmp_path / sub, ih_spec=ih_spec, apply_theta=apply,
                           spike_reason_fn=fx["spike_fn"])
        arrs.append(_read_archive(d))
    _clear()
    for part in (0, 1):
        assert arrs[0][part].keys() == arrs[1][part].keys()
        for k in arrs[0][part]:
            assert_array_equal(arrs[0][part][k], arrs[1][part][k])
    new_tw, old_tw = arrs[0][2]["twin"], arrs[1][2]["twin"]
    assert set(new_tw) - set(old_tw) == {"with_noise"}
    assert {k: v for k, v in new_tw.items() if k != "with_noise"} == old_tw


@pytest.mark.parametrize("bad", ["False", 0, 1, None, "true"])
def test_b18_with_noise_must_be_bool(fx, bad, tmp_path):
    ih_spec, apply = _model("full")
    with pytest.raises(tb.TwinBuildError):
        tb.build_twin(_spec("full"), fx["real"], fx["table"], tmp_path / "g", ih_spec=ih_spec, apply_theta=apply,
                      spike_reason_fn=fx["spike_fn"], with_noise=bad)
    assert not (tmp_path / "g" / ("specimen_%d" % TWIN_ID)).exists()
    _clear()


def test_b18_full_no_noise_configuration():
    assert tb.CONFIGURATIONS == ("full", "passive_fit", "active_fit", "full_no_noise")
    assert tb.HAS_IH["full_no_noise"] is True
