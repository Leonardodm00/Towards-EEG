"""Smoke test for the twin build -- Block 12 in specs/SPEC.md (TB1-TB9, and the
optional slow oracle on a real cell).

The fixture: a real-layout archive cell written by
smoke_noise_windows.write_fixture (two SS sweeps with trace-QC drops; LS
sweeps of -110, -50, 0, 30, 60 pA, the 0 pA one not a step), a ball-and-stick
reconstruction.swc from synthetic_ground_truth.write_ball_and_stick_swc, a
spike test written here that gives the 60 pA sweep the reason
'action_potential', and a twins' table written by hand from the planted laws
and the fixture's constants (then through twin_table's CSV writer and
reader). The fit's own IhSpec() and make_ih_spec() / make_passive_spec()
defaults.

Checks
    test_TB1_zero_noise_identity   the fitter's own simulation at theta(g) on the
                                   loader's bundles of the written clean archive
                                   (full, active_fit) at dt = cfg.neuron_dt_ms:
                                   RMSD < 1e-6 mV per bundle; the RMSD at the
                                   fit's default 0.1 ms is printed, not gated
    test_TB2_composition           noisy minus clean = Block 10's noise recomputed
                                   here (1e-12 mV); the in_twin False sweep absent
    test_TB3_archive_round_trip    load_cell_from_archive on the twin
    test_TB4_rest_guard            drift printed and < 1e-3 mV; a shifted leak refused
    test_TB5_configurations        the mechanism where it belongs, theta keys
    test_TB6_determinism           same inputs, same arrays; another twin id
    test_TB7_metadata              the keys of metadata.json
    test_TB8_refusals              every refusal of the entry
    test_TB9_ascii_lf              the two files
    test_TB10_real_twin            optional, slow (--archive-cell, --nf2-dir)

Run (from this folder: NEURON loads ./x86_64 from the working directory)
    python smoke_twin_build.py [--build] [--archive-cell DIR --nf2-dir DIR]
        Prints a PASS / FAIL / ERROR / SKIP table and the full message of every
        failure. Exit code 1 if any check failed or errored.

Pure ASCII, LF.
"""
from __future__ import annotations

import argparse
import atexit
import dataclasses
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                     # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

import twin_build as tb            # noqa: E402
import twin_noise as tn            # noqa: E402
import twin_table as tt            # noqa: E402
from noise_law import NoiseLaw     # noqa: E402
from twin_config import TwinConfig  # noqa: E402
import smoke_noise_windows as snw  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "neuron")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []
ARGS = argparse.Namespace(archive_cell=None, nf2_dir=None)
CFG = TwinConfig()

# ---------------------------------------------------------------- fixture ---
REAL_ID = 900000003                       # write_fixture's specimen id (the "real" cell)
TWIN_ID, TWIN_ID_2 = 900000501, 900000502
SS_SWEEPS = (92, 93)
SS_DROPPED = {92: (2, 3, 11, 20), 93: (0, 5, 27)}
LS_AMPS = (-110.0, -50.0, 0.0, 30.0, 60.0)  # sweeps 54..58; 56 (0 pA) is no step
SPIKE_SWEEP = 58                            # its real trace gets 'action_potential'
SPIKE_REASON = "action_potential"
V_REST = -60.0                              # the fixture's metadata v_rest_mV
F = 1.9
THETA_P = dict(Cm=1.0, Rm=20000.0, Ra=150.0)
THETA_IH = dict(THETA_P, gbar=1.2e-4, dv_h=2.0, kappa_tau=1.3)
TAU_M_S, TAU_S_S = 7.1e-5, 0.21             # the cell's two time scales (s)


def _planted_law(pr, s):
    """(NoiseLaw, tilt mV/s) planted for each sweep: distinct values per sweep."""
    k = s - (92 if pr == "SS" else 54)
    base = 0.0 if pr == "SS" else 0.01
    law = NoiseLaw(0.031 + 1e-3 * k + base, 0.017 + 5e-4 * k, TAU_M_S, 0.081 + 2e-3 * k + base, TAU_S_S)
    tilt = (0.0123 * (k + 1) - 0.02) if pr == "SS" else (-0.0567 + 0.031 * k)
    return law, tilt


_STATE = {}


class PlantedSpikeTest:
    """spike_reason_fn written here: the reason planted for SPIKE_SWEEP, ''
    for every other sweep (identified by the bytes of its stored voltage)."""

    def __init__(self, by_v):
        self.by_v = by_v
        self.calls = []

    def __call__(self, v_mV, sampling_rate_Hz):
        s = self.by_v[np.asarray(v_mV).tobytes()]
        self.calls.append(s)
        return SPIKE_REASON if s == SPIKE_SWEEP else ""


def _fixture():
    if _STATE:
        return _STATE
    root = Path(tempfile.mkdtemp(prefix="smoke_twin_build_"))
    atexit.register(shutil.rmtree, str(root), True)
    real = root / "real" / ("specimen_%d" % REAL_ID)
    snw.write_fixture(real, np.random.default_rng(SEED), with_response=True, dropped=SS_DROPPED,
                      sweeps=SS_SWEEPS, ls_amps=LS_AMPS)
    import synthetic_ground_truth as sgt
    sgt.write_ball_and_stick_swc(real / "reconstruction.swc")
    meta = json.load(open(real / "metadata.json"))
    by_v = {}
    with np.load(real / "ls_sweeps.npz") as z:
        for info in meta["ls_sweeps"]:
            by_v[np.asarray(z["v_%d" % info["index"]], dtype=np.float64).tobytes()] = int(info["sweep_number"])
    spike_fn = PlantedSpikeTest(by_v)
    protocol = tt.load_cell_protocol(real, spike_reason_fn=spike_fn)
    table_path = root / "twin_sweeps_FIX.csv"
    tt.write_twin_sweep_table(_hand_table(protocol), table_path)
    table = tt.read_twin_sweep_table(table_path)
    _STATE.update(root=root, real=real, meta=meta, spike_fn=spike_fn, protocol=protocol, table=table, n=0)
    return _STATE


def _hand_table(protocol):
    """The twins' table written out from the planted values and the fixture's
    constants (snw.*): the onsets, durations and lengths are the fixture's,
    not read back from the protocol."""
    rows = []
    for s in SS_SWEEPS:
        law, tilt = _planted_law("SS", s)
        n_win = 37 - len(SS_DROPPED[s])
        rows.append(dict(specimen_id=REAL_ID, protocol="SS", sweep_number=s, n_windows=n_win, amplitude_pA=np.nan,
                         onset_s=np.nan, duration_s=np.nan, n_samples=snw.PRE_N + snw.POST_N, fs_Hz=snw.FS,
                         real_spike_reason="", in_twin=True, **_law_cols(law, tilt)))
    for k, amp in enumerate(LS_AMPS):
        s = 54 + k
        if amp == 0.0:
            continue                          # not a step: no row (Block 5, amendment 3)
        law, tilt = _planted_law("LS", s)
        why = SPIKE_REASON if s == SPIKE_SWEEP else ""
        rows.append(dict(specimen_id=REAL_ID, protocol="LS", sweep_number=s, n_windows=1, amplitude_pA=amp,
                         onset_s=snw.LS_PRE_N / snw.FS,
                         duration_s=(snw.LS_PRE_N + snw.LS_STEP_N) / snw.FS - snw.LS_PRE_N / snw.FS,
                         n_samples=snw.LS_PRE_N + snw.LS_STEP_N + snw.LS_TAIL_N, fs_Hz=snw.FS,
                         real_spike_reason=why, in_twin=(why == ""), **_law_cols(law, tilt)))
    df = pd.DataFrame(rows)[list(tt.TWIN_SWEEP_COLUMNS)]
    for c in ("specimen_id", "sweep_number", "n_windows", "n_samples"):
        df[c] = df[c].astype(np.int64)
    for c in ("in_twin", "cell_tau_s_above_window"):
        df[c] = df[c].astype(bool)
    return df


def _law_cols(law, tilt):
    return dict(sigma_f_mV=law.sigma_f, sigma_m_mV=law.sigma_m, tau_m_s=law.tau_m, sigma_s_mV=law.sigma_s,
                tau_s_s=law.tau_s, tilt_mV_per_s=tilt, law_source="sweep", noise_flags="",
                cell_tau_s_above_window=False)


def _spec(configuration, twin_id=TWIN_ID, theta=None, **kw):
    if theta is None:
        theta = dict(THETA_IH) if tb.HAS_IH[configuration] else dict(THETA_P)
    return tb.TwinSpec(twin_id=twin_id, twin_of_specimen_id=kw.pop("real_id", REAL_ID),
                       configuration=configuration, theta=theta, v_rest_mV=kw.pop("v_rest", V_REST), F=F)


def _model(configuration):
    """(ih_spec, apply_theta): the fit's own IhSpec() and ParamSpec defaults."""
    import ih_mechanism as IM
    import param_spec as PS
    if tb.HAS_IH[configuration]:
        return IM.IhSpec(), PS.make_ih_spec().apply
    return IM.IhSpec(), PS.make_passive_spec().apply


def _clear():
    import synthetic_ground_truth as sgt
    sgt._clear_neuron_sections()


def _new_group():
    st = _fixture()
    st["n"] += 1
    return st["root"] / ("group_%d" % st["n"])


def _build_traces(configuration, *, twin_id=TWIN_ID, with_noise=True, cfg=CFG):
    """(cell, traces) of one twin on the fixture; sections cleared first."""
    st = _fixture()
    _clear()
    ih_spec, apply_theta = _model(configuration)
    spec = _spec(configuration, twin_id=twin_id)
    cell = tb.build_twin_cell(st["real"] / "reconstruction.swc", spec, ih_spec=ih_spec, apply_theta=apply_theta,
                              cfg=cfg)
    return cell, tb.simulate_twin(cell, spec, st["protocol"], st["table"], cfg=cfg, with_noise=with_noise)


def _build_twin(configuration, *, twin_id=TWIN_ID, group=None):
    st = _fixture()
    _clear()
    ih_spec, apply_theta = _model(configuration)
    out = tb.build_twin(_spec(configuration, twin_id=twin_id), st["real"], st["table"], group or _new_group(),
                        ih_spec=ih_spec, apply_theta=apply_theta, spike_reason_fn=st["spike_fn"], cfg=CFG)
    _clear()
    return out


def _load(d):
    import passive_fitting_hpc_fixed as mono
    return mono.load_cell_from_archive(d, load_depolarising_ls=True, ls_max_amplitude_pA=None,
                                       ls_dep_max_amplitude_pA=None, verbose=False)


def check_close(name, actual, expected, rtol=0.0, atol=0.0):
    """assert_allclose that keeps the evidence (smoke_artifacts/<name>.npz) on failure."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / ("%s.npz" % re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def _expect_refusal(label, func):
    try:
        func()
    except tb.TwinBuildError as exc:
        return str(exc)
    except Exception as exc:
        raise AssertionError("%s: raised %s, not TwinBuildError: %s" % (label, type(exc).__name__, exc))
    raise AssertionError("%s: nothing raised" % label)


# ---------------------------------------------------------------- checks ---
TB1_TOL_MV = 1e-6
TB1_MODES = ("after_onset", "step", "sweep")


def _tb1_rmsds(cd, configuration, dt_ms):
    """{(kind, amplitude, mode): RMSD} of the fitter's own simulation at
    theta(g) (passive_long_step_training.bundle_rmsd on a cell built by
    (TC1)) against every bundle of the loaded clean twin, at dt_ms."""
    import ih_mechanism as IM
    import passive_fitting_hpc_fixed as mono
    import passive_long_step_training as plst
    _clear()
    ih_spec, apply_theta = _model(configuration)
    spec = _spec(configuration)
    cell = mono.build_neuron_model(cd.swc_path, F=F)
    if tb.HAS_IH[configuration]:
        IM.attach_ih(cell, ih_spec)
    apply_theta(cell, dict(spec.theta), float(cd.v_rest_mV))
    out = {}
    bundles = ([("SS", b) for b in cd.square_subthreshold] + [("LS", b) for b in cd.long_square_subthreshold]
               + [("LS", b) for b in cd.long_square_depolarising])
    for kind, b in bundles:
        modes = TB1_MODES if kind == "LS" else ("ss_window",)
        for mode in modes:
            r, _ = plst.bundle_rmsd(cell, b, float(cd.v_rest_mV), ls_window_mode=(mode if kind == "LS" else
                                                                                 "after_onset"),
                                    dt_brief_ms=dt_ms, dt_long_ms=dt_ms)
            out[(kind, float(b.amplitude_pA), mode)] = float(r)
    _clear()
    return out


def test_TB1_zero_noise_identity():
    st = _fixture()
    lines, worst = [], 0.0
    for configuration in ("full", "active_fit"):
        cell, traces = _build_traces(configuration, with_noise=False)
        _clear()
        d = tb.write_twin_archive(_new_group(), _spec(configuration), traces, swc_path=st["real"] / "reconstruction.swc",
                                  real_metadata=st["meta"], features=dict(rin_MOhm=1.0, tau_ms=1.0, sag_ratio=0.0))
        cd = _load(d)
        n_ss, n_ls = len(cd.square_subthreshold), len(cd.long_square_subthreshold) + len(cd.long_square_depolarising)
        assert n_ss == 2 and n_ls == len(traces.ls_sweep_number), (n_ss, n_ls)
        eq = _tb1_rmsds(cd, configuration, CFG.neuron_dt_ms)
        fit = _tb1_rmsds(cd, configuration, 0.1)
        for key in sorted(eq):
            worst = max(worst, eq[key])
            lines.append("%s %s %+.0f pA %s: dt %.3g ms %.2e mV%s | dt 0.1 ms %.2e mV"
                         % (configuration, key[0], key[1], key[2], CFG.neuron_dt_ms, eq[key],
                            "" if eq[key] < TB1_TOL_MV else " FAIL", fit[key]))
    print("\n  [TB1] " + "\n  [TB1] ".join(lines))
    bad = [ln for ln in lines if " FAIL" in ln]
    assert not bad, ("TB1: %d bundle/window RMSD(s) at equal steps >= %g mV (worst %.3g mV): %s"
                     % (len(bad), TB1_TOL_MV, worst, "; ".join(bad)))


def test_TB2_composition():
    st = _fixture()
    prot = st["protocol"]
    _cell, noisy = _build_traces("full", with_noise=True)
    _cell, clean = _build_traces("full", with_noise=False)
    _clear()
    dt_ss = 1.0 / prot.ss_fs_Hz
    for s in SS_SWEEPS:
        m = prot.ss_sweep_number == s
        law, tilt = _planted_law("SS", s)
        e = tn.ss_sweep_noise(law, tilt, dt_ss, prot.ss_t_s.size, prot.ss_position[m], prot.ss_period_samples,
                              base_seed=CFG.base_seed, twin_id=TWIN_ID, sweep_number=s)
        check_close("TB2_ss_%d" % s, noisy.ss_v_mV[m] - clean.ss_v_mV[m], e, atol=1e-12)
    want_ls = [54 + k for k, a in enumerate(LS_AMPS) if a != 0.0 and 54 + k != SPIKE_SWEEP]
    assert noisy.ls_sweep_number.tolist() == want_ls, (noisy.ls_sweep_number.tolist(), want_ls)
    for m, s in enumerate(noisy.ls_sweep_number):
        law, tilt = _planted_law("LS", int(s))
        e = tn.ls_sweep_noise(law, tilt, 1.0 / prot.ls_fs_Hz, noisy.ls_v_mV[m].size, base_seed=CFG.base_seed,
                              twin_id=TWIN_ID, sweep_number=int(s))
        check_close("TB2_ls_%d" % s, noisy.ls_v_mV[m] - clean.ls_v_mV[m], e, atol=1e-12)
    d = _build_twin("full")
    meta = json.load(open(d / "metadata.json"))
    got = [int(x["sweep_number"]) for x in meta["ls_sweeps"]]
    assert got == want_ls, ("archive LS sweeps", got, want_ls)
    assert meta["twin"]["ls_sweeps_left_out"] == {str(SPIKE_SWEEP): SPIKE_REASON}, meta["twin"]["ls_sweeps_left_out"]
    with np.load(d / "ls_sweeps.npz") as z:
        assert int(z["n_sweeps"][0]) == len(want_ls)


def test_TB3_archive_round_trip():
    d = _build_twin("full")
    cd = _load(d)
    assert int(cd.specimen_id) == TWIN_ID, cd.specimen_id
    assert cd.v_rest_mV == V_REST, (cd.v_rest_mV, V_REST)
    st = _fixture()
    prot = st["protocol"]
    with np.load(d / "ss_pulses.npz") as z:
        assert set(z.files) == set(tb.SS_NPZ_KEYS), sorted(z.files)
        v, pol = z["v"], z["polarity_is_dep"]
    assert len(cd.square_subthreshold) == 2, len(cd.square_subthreshold)
    for b in cd.square_subthreshold:
        m = pol == (b.polarity == "dep")
        check_close("TB3_ss_%s" % b.polarity, b.v_mV, v[m].mean(axis=0), atol=1e-12)
        pm = prot.ss_polarity_is_dep == (b.polarity == "dep")
        a_p = float(np.mean(prot.ss_peak_pA[pm]))
        d_p = float(np.mean(prot.ss_stim_duration_s[pm]))
        assert b.amplitude_pA == a_p and b.stim_duration_s == d_p, (b.amplitude_pA, a_p, b.stim_duration_s, d_p)
    ls = list(cd.long_square_subthreshold) + list(cd.long_square_depolarising)
    got = sorted((int(b.sweep_numbers[0]), float(b.amplitude_pA)) for b in ls)
    want = sorted((54 + k, a) for k, a in enumerate(LS_AMPS) if a != 0.0 and 54 + k != SPIKE_SWEEP)
    assert got == want, ("LS bundles", got, want)
    with np.load(d / "ls_sweeps.npz") as z:
        keys = set(z.files)
    want_keys = {"n_sweeps"} | {"%s_%d" % (p, m) for p in ("v", "i") for m in range(len(want))}
    assert keys == want_keys, sorted(keys ^ want_keys)


def test_TB4_rest_guard():
    import ih_mechanism as IM
    st = _fixture()
    drifts = {}
    for configuration in tb.CONFIGURATIONS:
        cell, _t = _build_traces(configuration, with_noise=False)
        drifts[configuration] = float(cell.twin_rest_drift_mV)
    _clear()
    print("\n  [TB4] rest drift over %g ms: %s" % (CFG.rest_drift_ms, ", ".join("%s %.2e mV" % kv
                                                                                for kv in drifts.items())))
    assert all(v < CFG.rest_drift_max_mV for v in drifts.values()), drifts
    ih_spec, apply_theta = _model("full")

    def shifted(cell, theta, v_rest):
        apply_theta(cell, theta, v_rest)
        for sec in IM._all_sections(cell):
            for seg in sec:
                seg.e_pas = seg.e_pas + 0.01            # 0.01 mV: ten times the bound
    msg = _expect_refusal("shifted leak", lambda: tb.build_twin_cell(
        st["real"] / "reconstruction.swc", _spec("full"), ih_spec=ih_spec, apply_theta=shifted, cfg=CFG))
    _clear()
    assert "drift" in msg, msg


def test_TB5_configurations():
    import ih_mechanism as IM
    regions = IM.IhSpec().regions
    for configuration in ("full", "passive_fit"):
        cell, _t = _build_traces(configuration, with_noise=False)
        n = 0
        for r in ("soma", "dend", "apic", "axon"):
            for sec in getattr(cell, r):
                has = sec.has_membrane("Ih_human")
                assert has == (r in regions), (configuration, sec.name(), has)
                if has:
                    for seg in sec:
                        assert seg.Ih_human.gIhbar == THETA_IH["gbar"], (sec.name(), seg.Ih_human.gIhbar)
                        n += 1
        assert n > 0
    cell, _t = _build_traces("active_fit", with_noise=False)
    for r in ("soma", "dend", "apic", "axon"):
        for sec in getattr(cell, r):
            assert not sec.has_membrane("Ih_human") and not sec.has_membrane("Ih"), (sec.name(),)
    _clear()
    st = _fixture()
    ih_spec, apply_theta = _model("full")
    swc = st["real"] / "reconstruction.swc"
    for label, conf, theta in (("full missing kappa_tau", "full", {k: v for k, v in THETA_IH.items()
                                                                   if k != "kappa_tau"}),
                               ("passive_fit three keys", "passive_fit", dict(THETA_P)),
                               ("active_fit with gbar", "active_fit", dict(THETA_P, gbar=1e-4)),
                               ("full extra key", "full", dict(THETA_IH, e_pas=-70.0))):
        _expect_refusal(label, lambda: tb.build_twin_cell(swc, _spec(conf, theta=theta), ih_spec=ih_spec,
                                                          apply_theta=apply_theta, cfg=CFG))


def test_TB6_determinism():
    a = _build_twin("full")
    b = _build_twin("full")
    c = _build_twin("full", twin_id=TWIN_ID_2)
    for name in ("ss_pulses.npz", "ls_sweeps.npz"):
        with np.load(a / name) as za, np.load(b / name) as zb, np.load(c / name) as zc:
            for k in za.files:
                assert np.array_equal(za[k], zb[k]), ("same inputs differ", name, k)
                if k.startswith("v"):
                    assert not np.array_equal(za[k], zc[k]), ("another twin id, same noise", name, k)
    _c1, clean1 = _build_traces("full", twin_id=TWIN_ID, with_noise=False)
    _c2, clean2 = _build_traces("full", twin_id=TWIN_ID_2, with_noise=False)
    _clear()
    assert np.array_equal(clean1.ss_v_mV, clean2.ss_v_mV)
    assert all(np.array_equal(x, y) for x, y in zip(clean1.ls_v_mV, clean2.ls_v_mV))


def test_TB7_metadata():
    st = _fixture()
    d = _build_twin("full")
    meta = json.load(open(d / "metadata.json"))
    keys = ("specimen_id", "v_rest_mV", "ljp_correction_mV", "rin_MOhm", "tau_ms", "sag_ratio", "layer",
            "dendrite_type", "full_allen_metadata", "ls_sweeps", "ss_extraction", "twin")
    missing = [k for k in keys if k not in meta]
    assert not missing, missing
    assert meta["specimen_id"] == TWIN_ID and meta["v_rest_mV"] == V_REST and meta["ljp_correction_mV"] == 0.0
    assert meta["layer"] == st["meta"]["layer"] and meta["dendrite_type"] == st["meta"]["dendrite_type"]
    fam = meta["full_allen_metadata"]
    assert fam["structure_layer_name"] == st["meta"]["layer"] and fam["dendrite_type"] == st["meta"]["dendrite_type"]
    for k in ("rin_MOhm", "tau_ms"):
        assert np.isfinite(meta[k]) and meta[k] > 0, (k, meta[k])
    assert np.isfinite(meta["sag_ratio"]) and meta["sag_ratio"] >= 0, meta["sag_ratio"]
    prot = st["protocol"]
    K = int(prot.ss_polarity_is_dep.size)
    assert meta["ss_extraction"] == dict(n_pulses_qc=K, n_dep=int(prot.ss_polarity_is_dep.sum()),
                                         n_hyp=int(K - prot.ss_polarity_is_dep.sum())), meta["ss_extraction"]
    for m, e in enumerate(meta["ls_sweeps"]):
        assert set(e) == {"index", "sweep_number", "detected_amplitude_pA", "sampling_rate_Hz", "stimulus_name",
                          "n_samples"} and e["index"] == m and e["stimulus_name"] == "Long Square", e
    tw = meta["twin"]
    want = {"twin_of_specimen_id", "configuration", "theta", "F", "has_ih", "base_seed", "neuron_dt_ms",
            "settle_ms", "rest_drift_mV", "ls_sweeps_left_out", "provenance"}
    assert set(tw) == want, sorted(set(tw) ^ want)
    assert tw["theta"] == THETA_IH and tw["twin_of_specimen_id"] == REAL_ID and tw["configuration"] == "full"
    assert tw["has_ih"] is True and tw["F"] == F and tw["base_seed"] == CFG.base_seed
    assert tw["neuron_dt_ms"] == CFG.neuron_dt_ms and tw["settle_ms"] == CFG.settle_ms
    assert 0.0 <= tw["rest_drift_mV"] < CFG.rest_drift_max_mV, tw["rest_drift_mV"]
    print("\n  [TB7] features: rin_MOhm %.4g, tau_ms %.4g, sag_ratio %.4g"
          % (meta["rin_MOhm"], meta["tau_ms"], meta["sag_ratio"]))


def test_TB8_refusals():
    st = _fixture()
    real, table, prot = st["real"], st["table"], st["protocol"]
    swc = real / "reconstruction.swc"
    ih_spec, apply_ih = _model("full")
    _p, apply_p = _model("active_fit")
    caught = []

    def refuse(label, func):
        _expect_refusal(label, func)
        _clear()
        caught.append(label)

    refuse("unknown configuration", lambda: tb.build_twin_cell(
        swc, dataclasses.replace(_spec("full"), configuration="ih6"), ih_spec=ih_spec, apply_theta=apply_ih))
    refuse("theta keys", lambda: tb.build_twin_cell(swc, _spec("full", theta=dict(THETA_P)), ih_spec=ih_spec,
                                                     apply_theta=apply_ih))
    refuse("theta non-finite", lambda: tb.build_twin_cell(swc, _spec("full", theta=dict(THETA_IH, Ra=float("nan"))),
                                                           ih_spec=ih_spec, apply_theta=apply_ih))
    refuse("I_h without ih_spec", lambda: tb.build_twin_cell(swc, _spec("passive_fit"), ih_spec=None,
                                                              apply_theta=apply_ih))
    refuse("twin id = real id", lambda: tb.build_twin(_spec("active_fit", twin_id=REAL_ID), real, table,
                                                      _new_group(), ih_spec=None, apply_theta=apply_p,
                                                      spike_reason_fn=st["spike_fn"]))

    def sim_with(tab):
        _clear()
        spec = _spec("active_fit")
        cell = tb.build_twin_cell(swc, spec, ih_spec=None, apply_theta=apply_p)
        return tb.simulate_twin(cell, spec, prot, tab, with_noise=False)

    refuse("table without the cell", lambda: sim_with(table.assign(specimen_id=np.int64(REAL_ID + 1))))
    refuse("table missing an SS sweep", lambda: sim_with(table[~((table.protocol == "SS")
                                                                 & (table.sweep_number == 93))]))
    refuse("table missing an LS sweep", lambda: sim_with(table[table.sweep_number != 54]))
    t = table.copy()
    t.loc[(t.protocol == "SS") & (t.sweep_number == 92), "n_windows"] += 1
    refuse("windows count", lambda: sim_with(t))
    t = table.copy()
    t.loc[t.sweep_number == 55, "amplitude_pA"] = -60.0
    refuse("amplitude", lambda: sim_with(t))
    t = table.copy()
    t.loc[t.sweep_number == SPIKE_SWEEP, "real_spike_reason"] = ""
    t.loc[t.sweep_number == SPIKE_SWEEP, "in_twin"] = True
    refuse("real_spike_reason", lambda: sim_with(t))
    t = table.copy()
    t.loc[t.sweep_number == SPIKE_SWEEP, "in_twin"] = True
    refuse("in_twin vs reason", lambda: sim_with(t))
    refuse("build_twin, table disagrees", lambda: tb.build_twin(_spec("active_fit"), real, t, _new_group(),
                                                                ih_spec=None, apply_theta=apply_p,
                                                                spike_reason_fn=st["spike_fn"]))
    # a real cell without reconstruction.swc: a copy of the fixture without it
    noswc = st["root"] / "noswc" / ("specimen_%d" % REAL_ID)
    if not noswc.exists():
        shutil.copytree(real, noswc)
        (noswc / "reconstruction.swc").unlink()
    refuse("no reconstruction.swc", lambda: tb.build_twin(_spec("active_fit"), noswc, table, _new_group(),
                                                          ih_spec=None, apply_theta=apply_p,
                                                          spike_reason_fn=st["spike_fn"]))
    # rest drift above the bound (TB4's shifted leak, again through build_twin_cell)
    import ih_mechanism as IM

    def shifted(cell, theta, v_rest):
        apply_p(cell, theta, v_rest)
        for sec in IM._all_sections(cell):
            for seg in sec:
                seg.e_pas = seg.e_pas - 0.01
    refuse("rest drift", lambda: tb.build_twin_cell(swc, _spec("active_fit"), ih_spec=None, apply_theta=shifted))
    # an existing directory without overwrite: build_twin and write_twin_archive
    group = _new_group()
    d = _build_twin("active_fit", group=group)
    refuse("existing dir (build_twin)", lambda: tb.build_twin(_spec("active_fit"), real, table, group,
                                                              ih_spec=None, apply_theta=apply_p,
                                                              spike_reason_fn=st["spike_fn"]))
    _c, traces = _build_traces("active_fit", with_noise=False)
    refuse("existing dir (write_twin_archive)", lambda: tb.write_twin_archive(
        group, _spec("active_fit"), traces, swc_path=swc, real_metadata=st["meta"],
        features=dict(rin_MOhm=1.0, tau_ms=1.0, sag_ratio=0.0)))
    out = tb.write_twin_archive(group, _spec("active_fit"), traces, swc_path=swc, real_metadata=st["meta"],
                                features=dict(rin_MOhm=1.0, tau_ms=1.0, sag_ratio=0.0), overwrite=True)
    assert out == d and not any(p.name.startswith(".") for p in group.iterdir()), list(group.iterdir())
    print("\n  [TB8] refused: " + "; ".join(caught))


def test_TB9_ascii_lf():
    for name in ("twin_build.py", "smoke_twin_build.py"):
        data = (HERE / name).read_bytes()
        assert all(b < 128 for b in data), "%s holds a non-ASCII byte" % name
        assert b"\r" not in data, "%s holds a CR" % name


def test_TB10_real_twin():
    if not (ARGS.archive_cell and ARGS.nf2_dir):
        raise unittest.SkipTest("needs --archive-cell <specimen dir> and --nf2-dir <dir> (real data)")
    import param_spec as PS
    import ih_mechanism as IM
    import passive_fitting_hpc_fixed as mono
    real = Path(ARGS.archive_cell).resolve()
    nf2 = Path(ARGS.nf2_dir).resolve()
    group = Path(tempfile.mkdtemp(prefix="smoke_twin_real_"))
    atexit.register(shutil.rmtree, str(group), True)
    cells, sweeps = tt.read_noise_tables(nf2 / "noise_cells_L3_exc.csv", nf2 / "noise_sweeps_L3_exc.csv")
    prot = tt.load_cell_protocol(real, spike_reason_fn=mono.sweep_spike_reason)
    table = tt.build_twin_sweep_table([prot], cells, sweeps)
    real_cd = mono.load_cell_from_archive(real, verbose=False)
    box = PS.make_ih_spec()
    theta = box.to_physical([0.5 * sum(a.q_bounds()) for a in box.axes])   # the centre of the fit's box (q-space)
    spec = tb.TwinSpec(twin_id=900000901, twin_of_specimen_id=int(prot.specimen_id), configuration="full",
                       theta=theta, v_rest_mV=float(real_cd.v_rest_mV), F=F)
    _clear()
    t0 = time.time()
    d = tb.build_twin(spec, real, table, group, ih_spec=IM.IhSpec(), apply_theta=box.apply, cfg=CFG)
    secs = time.time() - t0
    _clear()
    meta = json.load(open(d / "metadata.json"))
    cd = _load(d)
    left = meta["twin"]["ls_sweeps_left_out"]
    n_ls = len(cd.long_square_subthreshold) + len(cd.long_square_depolarising)
    print("\n  [TB10] real twin of %d: theta %s; built in %.1f s; %d LS sweeps in the twin (%d stored), %d left "
          "out %s; loader: %d SS bundles, %d LS bundles; drift %.2e mV; rin %.4g MOhm"
          % (prot.specimen_id, {k: round(v, 6) for k, v in theta.items()}, secs, len(meta["ls_sweeps"]),
             len(prot.ls_sweep_number), len(left), sorted(set(left.values())), len(cd.square_subthreshold), n_ls,
             meta["twin"]["rest_drift_mV"], meta["rin_MOhm"]))
    assert int(cd.specimen_id) == 900000901
    assert len(meta["ls_sweeps"]) == 15 and len(prot.ls_sweep_number) == 22, (len(meta["ls_sweeps"]),)
    assert len(left) == 7 and set(left.values()) == {"action_potential"}, left
    assert n_ls == 15, n_ls


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


def ensure_build(build: bool) -> None:
    special = HERE / "x86_64" / "special"
    if special.exists():
        return
    if not build:
        sys.exit("x86_64/special not found; run 'nrnivmodl mod' here or pass --build")
    r = subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0 or not special.exists():
        sys.exit("nrnivmodl failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="run nrnivmodl mod if x86_64/special is missing")
    ap.add_argument("--archive-cell", default=None, help="a real archive cell (specimen_508282493) for TB10")
    ap.add_argument("--nf2-dir", default=None, help="the nf2 noise-fit tables for TB10")
    ap.add_argument("--only", default=None, help="run only the checks whose name contains this string")
    args = ap.parse_args(argv)
    ARGS.archive_cell, ARGS.nf2_dir = args.archive_cell, args.nf2_dir
    ensure_build(args.build)
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj) and (not args.only or args.only in name)]
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
        except AssertionError as exc:
            status, detail = "FAIL", str(exc) or traceback.format_exc()
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    print()
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, lines[0][:150] if lines else "")).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print("-- arrays from a failed comparison: %s" % path)
    return 1 if counts["FAIL"] + counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
