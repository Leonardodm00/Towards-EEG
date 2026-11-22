"""Smoke test for the twins' manifest -- Blocks 13 and 16 in specs/SPEC.md
(sections 2.7 (i) and 2.8 (e)).

Checks (the oracles TM1-TM9 of the Block 13 entry, plus TM10, adapted to the
four configurations of Block 16; TM11-TM13 of the Block 16 entry)
    TM1  counts and order: 3 n + n_no_noise rows, n per drawn configuration,
         n_no_noise full_no_noise rows, configurations in order, ids
         id_base + 100 q + i (full_no_noise: id_base + 300 + i), unique and
         disjoint from the real ids
    TM2  the cells: n distinct real cells per configuration, equal to
         default_rng(SeedSequence([seed, q])).choice(sorted ids, n,
         replace=False) recomputed here; v_rest_mV, real_swc_path and the
         flag those of the real cell
    TM3  seeds: same seed -> same manifest; another seed -> another one; the
         cells of a configuration depend on (seed, q) alone (row order of the
         input, the truth kwargs and the other configurations do not move them)
    TM4  truths (independent reference): full and passive_fit equal
         synth_gt_grid.draw_manifest called here with the stated arguments,
         exactly; finite and inside synth_gt_grid.SearchBox; the dropped
         columns are absent
    TM5  active_fit: its I_h truth columns equal those of a false-positive row
         of draw_manifest (fp_control_frac = 0.5), at the default floor and at
         a floor passed in truth_kwargs; is_fp_control True there only
    TM6  arms and theta: arm follows ARM_OF; twin_theta has 6 keys (full,
         passive_fit) or 3 (active_fit) with the row's values; twin_spec_of
         returns a TwinSpec with the row's ids, configuration, theta, v_rest, F
    TM7  CSV round trip: equal values, dtypes and column order
    TM8  every refusal of the entry raises ValueError
    TM9  the two files are pure ASCII and LF; git diff <contract> --
         synth_gt_grid.py is empty
    TM10 sanity (Block 8's use): the manifest passed with a hand-made frame of
         fit results to ih_recovery_report.recovery_long is read without error
         and gives truth values equal to twin_theta's
    TM11 regression: the first 3 n rows equal the manifest of the contract
         commit's twin_manifest.py (read with git show, imported here) for the
         same arguments in every column except arm (ih6 -> ih5) and ih_dvh_mV
         (0 in the rows with I_h); dvh_truth_mV=None restores the drawn
         ih_dvh_mV bit for bit; another dvh_truth_mV is set in the rows with
         I_h only
    TM12 the full_no_noise rows: ids id_base + 300 + i, paired_with the i-th
         full twin's id, every other column equal to the partner's,
         with_noise False there and True elsewhere, paired_with -1 elsewhere;
         n_no_noise = 0 gives no such row and leaves the rest unchanged;
         n_no_noise = n works; its refusals raise ValueError
    TM13 twin_theta of a full_no_noise row equals its partner's (and
         twin_spec_of carries the same theta, its own ids and configuration)

Fixture: 18 fictitious real cells (ids unsorted in the frame, swc paths
.../specimen_<id>/reconstruction.swc that do not exist: draw_manifest reads
the paths only as strings -- morph names and, with a noise table, the
specimen id in the path; no noise table is passed here).

Run (from the code folder)
    python smoke_twin_manifest.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.

Keep this file pure ASCII.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.dont_write_bytecode = True   # leave no __pycache__ in the code folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import twin_manifest as TM  # noqa: E402

SEED = 0                    # the entry's default seed
CONTRACT_COMMIT = "aac46d3"
REPORT_PACKAGES = ("numpy", "pandas")
N = 10                      # n per configuration (Stage 7)
N0 = 5                      # n_no_noise (the entry's default)
DRAWN = ("full", "passive_fit", "active_fit")
ID_BASE = 900_000_000

# 18 fictitious real cells, written in a non-sorted row order on purpose
_RNG_FIX = np.random.default_rng(20261120)
REAL_IDS = (500_000_000 + _RNG_FIX.choice(90_000_000, size=18, replace=False)).astype(np.int64)


def make_real_cells(ids=REAL_IDS) -> pd.DataFrame:
    ids = np.asarray(ids, dtype=np.int64)
    return pd.DataFrame({
        "specimen_id": ids,
        "swc_path": ["/fake/L3_exc/specimen_{}/reconstruction.swc".format(int(i)) for i in ids],
        "v_rest_mV": -80.0 + (ids % 1000) / 100.0,
        "cell_tau_s_above_window": (ids % 3) == 0,
    })


def truth_kwargs(**over):
    """The keyword arguments run_ih_recovery.phase_manifest passes to
    draw_manifest at its _parse_args defaults (minus the ones the function sets
    and the noise table), plus the noise-only knobs."""
    import human_ih_params as hip
    kw = dict(cells_per_cohort=10, ra_mode="per_cohort", e_pas_mV=-73.5, F=1.9,
              cm_phys_lo=0.4, cm_phys_hi=1.5, tau_lo_ms=3.0, tau_hi_ms=40.0,
              ih_kinetics="Ih_human", ih_dist="uniform",
              ih_ehcn_mV=float(hip.ehcn_default_mV("Ih_human")),
              ih_regions=("soma", "dend", "apic"), ih_vshift_base_mV=0.0,
              ih_gbar_range_S_cm2=(2e-5, 3e-4), ih_dvh_range_mV=(-5.0, 5.0),
              ih_kappa_range=(0.7, 1.4), ih_gbar_floor_S_cm2=1e-6,
              ra_phys_lo=100.0, ra_phys_hi=500.0,
              noise_sigma_nominal_mV=0.05, noise_baseline_nominal_mV=0.05,
              noise_drift_nominal_mV=0.10, noise_cv=0.3, noise_rho_nominal=0.0)
    kw.update(over)
    return kw


def quiet(fn, *a, **k):
    """Call fn with stdout captured (draw_manifest prints a summary)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


_DEFAULT = object()


def manifest(seed=SEED, real=None, n=N, n0=_DEFAULT, dvh=_DEFAULT, **tk):
    """draw_twin_manifest at the entry's defaults for n_no_noise and
    dvh_truth_mV unless given (n0, dvh)."""
    real = make_real_cells() if real is None else real
    kw = {}
    if n0 is not _DEFAULT:
        kw["n_no_noise"] = n0
    if dvh is not _DEFAULT:
        kw["dvh_truth_mV"] = dvh
    return quiet(TM.draw_twin_manifest, real, n_per_configuration=n, seed=seed,
                 id_base=ID_BASE, truth_kwargs=truth_kwargs(**tk), **kw)


def reference_cells(seed, q, n=N, ids=REAL_IDS):
    rng = np.random.default_rng(np.random.SeedSequence([seed, q]))
    return np.asarray(rng.choice(np.sort(np.asarray(ids, dtype=np.int64)), size=n, replace=False))


def reference_truths(swcs, seed, q, **over):
    import synth_gt_grid as G
    s = int(np.random.SeedSequence([seed, q, 1]).generate_state(1)[0])
    return quiet(G.draw_manifest, swcs, draws_per_morph=1, seed=s, id_base=ID_BASE + 100 * q,
                 use_ih=True, fp_control_frac=0.0, **truth_kwargs(**over))


TRUTH_COLS = ("cm_true", "rm_true", "ra_true", "ih_gihbar_S_cm2", "ih_dvh_mV", "ih_kappa_tau")
IH_COLS = ("ih_gihbar_S_cm2", "ih_dvh_mV", "ih_kappa_tau")

_CACHE = {}


def base_manifest():
    if "m" not in _CACHE:
        _CACHE["m"] = manifest()
    return _CACHE["m"].copy()


# ---------------------------------------------------------------- checks ---

def test_tm1_counts_and_order():
    m = base_manifest()
    assert TM.CONFIGURATIONS == DRAWN + ("full_no_noise",), TM.CONFIGURATIONS
    assert len(m) == 3 * N + N0, len(m)
    exp_conf = [c for c in DRAWN for _ in range(N)] + ["full_no_noise"] * N0
    assert m["configuration"].tolist() == exp_conf, m["configuration"].tolist()
    exp_ids = [ID_BASE + 100 * q + i for q in range(3) for i in range(N)] + [ID_BASE + 300 + i for i in range(N0)]
    assert m["specimen_id"].tolist() == exp_ids, m["specimen_id"].tolist()
    assert m["specimen_id"].is_unique
    assert not set(m["specimen_id"]) & set(int(x) for x in REAL_IDS)
    assert m["specimen_id"].dtype == np.int64 and m["twin_of_specimen_id"].dtype == np.int64


def test_tm2_cells():
    m = base_manifest()
    real = make_real_cells().set_index("specimen_id")
    for q, conf in enumerate(TM.CONFIGURATIONS):
        sub = m[m["configuration"] == conf]
        got = sub["twin_of_specimen_id"].to_numpy()
        k = N0 if conf == "full_no_noise" else N
        assert len(set(got.tolist())) == k, (conf, got)
        # full_no_noise: the cells of the first N0 full twins (2.8 (e))
        exp = reference_cells(SEED, 0)[:N0] if conf == "full_no_noise" else reference_cells(SEED, q)
        assert np.array_equal(got, exp), (conf, got, exp)
        for _, r in sub.iterrows():
            rc = real.loc[int(r["twin_of_specimen_id"])]
            assert r["v_rest_mV"] == rc["v_rest_mV"], (r["v_rest_mV"], rc["v_rest_mV"])
            assert r["real_swc_path"] == rc["swc_path"]
            assert r["swc"] == rc["swc_path"]
            assert bool(r["cell_tau_s_above_window"]) == bool(rc["cell_tau_s_above_window"])
    # the three draws are not all the same set (independent draws)
    sets = [frozenset(m.loc[m["configuration"] == c, "twin_of_specimen_id"]) for c in DRAWN]
    assert len(set(sets)) > 1, "every configuration drew the same cells"
    assert (m["manifest_seed"] == SEED).all()


def test_tm3_seeds():
    a = base_manifest()
    b = manifest()
    assert_frame_equal(a, b, check_exact=True)
    c = manifest(seed=7)
    assert not a["twin_of_specimen_id"].equals(c["twin_of_specimen_id"])
    assert not np.array_equal(a["cm_true"].to_numpy(), c["cm_true"].to_numpy())
    # input row order does not matter (the ids are sorted before the draw)
    real = make_real_cells().iloc[::-1].reset_index(drop=True)
    d = manifest(real=real)
    assert_frame_equal(a, d, check_exact=True)
    # the cells of configuration q depend on (seed, q) alone: other truth
    # kwargs (which change every truth) leave the drawn cells unchanged
    e = manifest(ra_mode="per_cell", ih_kappa_range=(0.5, 2.0))
    assert a["twin_of_specimen_id"].equals(e["twin_of_specimen_id"])
    assert not np.array_equal(a["ra_true"].to_numpy(), e["ra_true"].to_numpy())
    for q in range(3):
        sub = c[c["configuration"] == TM.CONFIGURATIONS[q]]["twin_of_specimen_id"].to_numpy()
        assert np.array_equal(sub, reference_cells(7, q))


def test_tm4_truths_reference():
    import synth_gt_grid as G
    m = base_manifest()
    box = G.SearchBox()
    m_drawn = manifest(dvh=None)   # ih_dvh_mV as drawn (TM11 covers the default 0)
    for q, conf in enumerate(DRAWN):
        sub = m[m["configuration"] == conf].reset_index(drop=True)
        subd = m_drawn[m_drawn["configuration"] == conf].reset_index(drop=True)
        ref = reference_truths(sub["real_swc_path"].tolist(), SEED, q)
        cols = TRUTH_COLS if TM.HAS_IH[conf] else TRUTH_COLS[:3]
        if TM.HAS_IH[conf]:
            assert (sub["ih_dvh_mV"] == 0.0).all(), (conf, sub["ih_dvh_mV"].tolist())
            assert np.array_equal(subd["ih_dvh_mV"].to_numpy(), ref["ih_dvh_mV"].to_numpy()), conf
        for col in [c for c in cols if c != "ih_dvh_mV"] + ["tau_m_true_ms", "F", "ih_ehcn_mV",
                                                             "ih_vshift_base_mV"]:
            assert np.array_equal(sub[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float)), (conf, col)
        for col in ("ra_mode", "ih_kinetics", "ih_dist", "ih_regions", "morph_name", "group"):
            assert sub[col].tolist() == ref[col].tolist(), (conf, col)
        assert np.array_equal(sub["specimen_id"].to_numpy(), ref["specimen_id"].to_numpy())
        v = sub[list(TRUTH_COLS)].to_numpy(dtype=float)
        assert np.all(np.isfinite(v)), conf
        for col, (lo, hi) in (("cm_true", box.cm_bounds), ("rm_true", box.rm_bounds),
                              ("ra_true", box.ra_bounds)):
            x = sub[col].to_numpy()
            assert np.all((x >= lo) & (x <= hi)), (conf, col, x)
    dropped = [c for c in m.columns if c.startswith(("noise_", "acq_")) or c == "e_pas_mV"]
    assert not dropped, dropped
    # every other draw_manifest column is kept
    kept = [c for c in G.MANIFEST_COLUMNS if not c.startswith(("noise_", "acq_")) and c != "e_pas_mV"]
    assert all(c in m.columns for c in kept), [c for c in kept if c not in m.columns]
    assert list(m.columns[:len(TM.LEAD_COLUMNS)]) == list(TM.LEAD_COLUMNS)
    assert list(m.columns[-len(TM.TAIL_COLUMNS):]) == list(TM.TAIL_COLUMNS) == ["with_noise", "paired_with"]


def _fp_row(swcs, floor):
    import synth_gt_grid as G
    kw = truth_kwargs(ih_gbar_floor_S_cm2=floor)
    d = quiet(G.draw_manifest, swcs, draws_per_morph=1, seed=3, use_ih=True, fp_control_frac=0.5, **kw)
    fp = d[d["is_fp_control"]]
    assert len(fp) >= 1
    return fp.iloc[0]


def test_tm5_active_fit():
    m = base_manifest()
    swcs = make_real_cells()["swc_path"].tolist()[:4]
    for floor, mm in ((1e-6, m), (2.5e-6, manifest(ih_gbar_floor_S_cm2=2.5e-6))):
        fp = _fp_row(swcs, floor)
        act = mm[mm["configuration"] == "active_fit"]
        for col in IH_COLS:
            assert (act[col].to_numpy(dtype=float) == float(fp[col])).all(), (floor, col, act[col].tolist(), fp[col])
        assert act["is_fp_control"].all()
        assert not mm.loc[mm["configuration"] != "active_fit", "is_fp_control"].any()
        assert mm["is_fp_control"].dtype == bool
        # the I_h truths of full / passive_fit are drawn (not the floor)
        oth = mm[mm["configuration"] != "active_fit"]
        assert (oth["ih_gihbar_S_cm2"] > floor).all()
    # the passive truths of active_fit are still draw_manifest's (unchanged)
    act = m[m["configuration"] == "active_fit"].reset_index(drop=True)
    ref = reference_truths(act["real_swc_path"].tolist(), SEED, 2)
    for col in ("cm_true", "rm_true", "ra_true"):
        assert np.array_equal(act[col].to_numpy(), ref[col].to_numpy()), col


def test_tm6_arms_theta_spec():
    from twin_build import TwinSpec
    m = base_manifest()
    assert TM.ARM_OF == {"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5",
                         "full_no_noise": "ih5"}, TM.ARM_OF
    assert TM.HAS_IH == {"full": True, "passive_fit": True, "active_fit": False, "full_no_noise": True}
    assert TM.WITH_NOISE == {"full": True, "passive_fit": True, "active_fit": True, "full_no_noise": False}
    assert (m["arm"] == m["configuration"].map(TM.ARM_OF)).all()
    names6 = {"Cm": "cm_true", "Rm": "rm_true", "Ra": "ra_true",
              "gbar": "ih_gihbar_S_cm2", "dv_h": "ih_dvh_mV", "kappa_tau": "ih_kappa_tau"}
    for _, r in m.iterrows():
        th = TM.twin_theta(r)
        conf = r["configuration"]
        exp_keys = list(names6) if conf in ("full", "passive_fit", "full_no_noise") else ["Cm", "Rm", "Ra"]
        assert list(th) == exp_keys, (conf, list(th))
        for k in th:
            assert th[k] == float(r[names6[k]]) and type(th[k]) is float, (k, th[k])
        sp = TM.twin_spec_of(r)
        assert isinstance(sp, TwinSpec)
        assert sp.twin_id == int(r["specimen_id"]) and sp.twin_of_specimen_id == int(r["twin_of_specimen_id"])
        assert sp.configuration == conf and dict(sp.theta) == th
        assert sp.v_rest_mV == float(r["v_rest_mV"]) and sp.F == float(r["F"]) == 1.9
        assert TM.twin_spec_of(r, F=2.3).F == 2.3


def test_tm7_csv_round_trip():
    for m in (base_manifest(), manifest(n0=0), manifest(dvh=None)):
        assert m["with_noise"].dtype == bool and m["paired_with"].dtype == np.int64
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "twin_manifest.csv"
            TM.save_twin_manifest(m, p)
            raw = p.read_bytes()
            assert b"\r" not in raw and all(b < 128 for b in raw)
            back = TM.load_twin_manifest(p)
        assert list(back.columns) == list(m.columns)
        assert list(back.dtypes) == list(m.dtypes), [(c, a, b) for c, a, b in zip(m.columns, m.dtypes, back.dtypes)
                                                     if a != b]
        assert_frame_equal(back, m, check_exact=True)


def _raises(fn, *a, **k):
    try:
        quiet(fn, *a, **k)
    except ValueError:
        return True
    return False


def test_tm8_refusals():
    real = make_real_cells()
    f = TM.draw_twin_manifest
    bad = []

    def expect(label, *a, **k):
        if not _raises(f, *a, **k):
            bad.append(label)

    tk = truth_kwargs()
    expect("n=0", real, n_per_configuration=0, truth_kwargs=tk)
    big = make_real_cells(np.arange(1, 131, dtype=np.int64) * 1000 + 7)
    expect("n=101 (130 cells)", big, n_per_configuration=101, truth_kwargs=tk)
    expect("n=19 > 18 cells", real, n_per_configuration=19, truth_kwargs=tk)
    expect("n=2.5", real, n_per_configuration=2.5, truth_kwargs=tk)
    expect("n=True", real, n_per_configuration=True, truth_kwargs=tk)
    dup = real.copy()
    dup.loc[3, "specimen_id"] = dup.loc[5, "specimen_id"]
    expect("duplicate ids", dup, truth_kwargs=tk)
    flt = real.copy()
    flt["specimen_id"] = flt["specimen_id"].astype(float)
    flt.loc[2, "specimen_id"] = 500000000.5
    expect("non-integer id 1.5", flt, truth_kwargs=tk)
    st = real.copy()
    st["specimen_id"] = st["specimen_id"].astype(object)
    st.loc[0, "specimen_id"] = "abc"
    expect("string id", st, truth_kwargs=tk)
    nan_id = real.copy()
    nan_id["specimen_id"] = nan_id["specimen_id"].astype(float)
    nan_id.loc[0, "specimen_id"] = np.nan
    expect("NaN id", nan_id, truth_kwargs=tk)
    clash = make_real_cells(np.concatenate([REAL_IDS[:17], [ID_BASE + 100 * 2 + 3]]))
    expect("twin id equals a real id", clash, id_base=ID_BASE, truth_kwargs=tk)
    clash0 = make_real_cells(np.concatenate([REAL_IDS[:17], [ID_BASE + 300 + 3]]))
    expect("full_no_noise twin id equals a real id", clash0, id_base=ID_BASE, truth_kwargs=tk)
    for v in (np.nan, np.inf, "0", True):
        expect("dvh_truth_mV " + repr(v), real, dvh_truth_mV=v, truth_kwargs=tk)
    for col in TM.REAL_CELL_COLUMNS:
        expect("missing " + col, real.drop(columns=[col]), truth_kwargs=tk)
    for v in (np.nan, np.inf):
        r2 = real.copy()
        r2.loc[4, "v_rest_mV"] = v
        expect("v_rest " + str(v), r2, truth_kwargs=tk)
    for k in ("draws_per_morph", "use_ih", "fp_control_frac", "seed", "id_base", "noise_table"):
        expect("truth_kwargs " + k, real, truth_kwargs=dict(tk, **{k: 1}))
    assert not bad, "not refused: {}".format(bad)
    # and the valid boundary values pass
    quiet(f, real, n_per_configuration=1, n_no_noise=1, truth_kwargs=tk)   # n_no_noise <= n (Block 16)
    quiet(f, real, n_per_configuration=18, truth_kwargs=tk)
    quiet(f, big, n_per_configuration=100, truth_kwargs=tk)
    # an id that only a full_no_noise twin beyond n_no_noise would take is free
    quiet(f, clash0, id_base=ID_BASE, n_no_noise=3, truth_kwargs=tk)


def test_tm9_ascii_lf_and_frozen_generator():
    for name in ("twin_manifest.py", "smoke_twin_manifest.py"):
        raw = (HERE / name).read_bytes()
        assert all(b < 128 for b in raw), name + " is not pure ASCII"
        assert b"\r" not in raw, name + " has CR"
    try:
        out = subprocess.run(["git", "diff", CONTRACT_COMMIT, "--", "synth_gt_grid.py"], cwd=str(HERE),
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest("git diff not available here: {}".format(exc))
    assert out == "", "synth_gt_grid.py differs from the contract commit"


def test_tm10_recovery_report_reads_it():
    import ih_recovery_report as R
    assert R.TRUTH_COLUMN == dict(TM.PASSIVE_TRUTH + TM.IH_TRUTH), R.TRUTH_COLUMN
    m = base_manifest()
    results = []
    for arm in ("ih5", "passive_fullstep"):
        confs = [c for c in TM.CONFIGURATIONS if TM.ARM_OF[c] == arm]
        sub = m[m["configuration"].isin(confs)]
        names = ["Cm", "Rm", "Ra"] + (["gbar", "kappa_tau"] if arm == "ih5" else [])
        rows = []
        for _, r in sub.iterrows():
            th = TM.twin_theta(r)
            row = {"specimen_id": int(r["specimen_id"]), "arm": arm, "param_names": "|".join(names)}
            for k in names:
                tru = th.get(k, float(r[R.TRUTH_COLUMN[k]]))
                row[k] = tru + 0.5 if k == "dv_h" else tru * 1.1
            rows.append(row)
        results.append(pd.DataFrame(rows))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.csv"
        TM.save_twin_manifest(m, p)
        mm = TM.load_twin_manifest(p)
    long_df = R.recovery_long(mm, results)
    assert len(long_df) == 5 * (N + N + N0) + 3 * N, len(long_df)
    idx = m.set_index("specimen_id")
    for _, lr in long_df.iterrows():
        rowd = dict(idx.loc[lr["specimen_id"]].to_dict())
        rowd["specimen_id"] = lr["specimen_id"]
        th = TM.twin_theta(rowd)
        par = lr["parameter"]
        if par in th:
            assert lr["truth"] == th[par], (lr["specimen_id"], par)
        else:   # an ih5 fit of an active_fit twin: the false-positive values
            assert idx.loc[lr["specimen_id"], "configuration"] == "active_fit"
            assert lr["truth"] == {"gbar": 1e-6, "dv_h": 0.0, "kappa_tau": 1.0}[par], (lr["specimen_id"], par)
        assert bool(lr["is_fp_control"]) == (idx.loc[lr["specimen_id"], "configuration"] == "active_fit")
    fp = R.fp_control_summary(long_df)
    assert list(fp["arm"]) == ["ih5"] and int(fp["n_fp_cells"].iloc[0]) == N, fp


def _contract_module():
    """twin_manifest.py of the contract commit, imported under another name."""
    import importlib.util
    if "contract" in _CACHE:
        return _CACHE["contract"]
    try:
        src = subprocess.run(["git", "show", CONTRACT_COMMIT + ":./twin_manifest.py"], cwd=str(HERE),
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest("git show not available here: {}".format(exc))
    d = tempfile.mkdtemp(prefix="tm_contract_")
    path = Path(d) / "twin_manifest_contract.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("twin_manifest_contract", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CONFIGURATIONS == DRAWN, "the contract module is not Block 13's"
    _CACHE["contract"] = mod
    return mod


def test_tm11_regression_against_contract():
    old_mod = _contract_module()
    real = make_real_cells()
    cases = [dict(seed=SEED, n=N, tk={}), dict(seed=7, n=4, tk=dict(ra_mode="per_cell")),
             dict(seed=3, n=18, tk=dict(ih_dvh_range_mV=(-8.0, 2.0)))]
    for case in cases:
        n, sd = case["n"], case["seed"]
        k0 = min(N0, n)   # the default 5 is refused for n < 5 (0 <= n_no_noise <= n)
        old = quiet(old_mod.draw_twin_manifest, real, n_per_configuration=n, seed=sd, id_base=ID_BASE,
                    truth_kwargs=truth_kwargs(**case["tk"]))
        assert len(old) == 3 * n
        new = manifest(seed=sd, n=n, n0=k0, **case["tk"]).iloc[:3 * n].reset_index(drop=True)
        new_none = manifest(seed=sd, n=n, n0=k0, dvh=None, **case["tk"]).iloc[:3 * n].reset_index(drop=True)
        new_m25 = manifest(seed=sd, n=n, n0=k0, dvh=-2.5, **case["tk"]).iloc[:3 * n].reset_index(drop=True)
        for nm in (new, new_none, new_m25):
            assert list(nm.columns) == list(old.columns) + list(TM.TAIL_COLUMNS), list(nm.columns)
            assert nm["configuration"].tolist() == old["configuration"].tolist()
            exp_arm = old["arm"].map({"ih6": "ih5", "passive_fullstep": "passive_fullstep"})
            assert nm["arm"].tolist() == exp_arm.tolist(), (nm["arm"].tolist(), old["arm"].tolist())
        has_ih = old["configuration"].map(TM.HAS_IH).to_numpy(dtype=bool)
        assert has_ih.sum() == 2 * n
        others = [c for c in old.columns if c not in ("arm", "ih_dvh_mV")]
        assert_frame_equal(new[others], old[others], check_exact=True)
        assert_frame_equal(new_none[others], old[others], check_exact=True)
        assert_frame_equal(new_m25[others], old[others], check_exact=True)
        # the default 0 mV in the rows with I_h; active_fit keeps the false-positive 0
        assert (new.loc[has_ih, "ih_dvh_mV"] == 0.0).all()
        assert np.array_equal(new.loc[~has_ih, "ih_dvh_mV"].to_numpy(), old.loc[~has_ih, "ih_dvh_mV"].to_numpy())
        # the drawn values were not 0 (so the test can see the setting)
        assert (old.loc[has_ih, "ih_dvh_mV"] != 0.0).all()
        # dvh_truth_mV=None: the drawn ih_dvh_mV bit for bit
        a = new_none["ih_dvh_mV"].to_numpy(dtype=np.float64)
        b = old["ih_dvh_mV"].to_numpy(dtype=np.float64)
        assert a.tobytes() == b.tobytes(), "dvh_truth_mV=None does not restore the drawn ih_dvh_mV"
        # another value: set in the rows with I_h only
        assert (new_m25.loc[has_ih, "ih_dvh_mV"] == -2.5).all()
        assert np.array_equal(new_m25.loc[~has_ih, "ih_dvh_mV"].to_numpy(), b[~has_ih])
        # the drawn rows are the same whatever n_no_noise
        for n0 in (0, n):
            alt = manifest(seed=sd, n=n, n0=n0, **case["tk"]).iloc[:3 * n].reset_index(drop=True)
            assert_frame_equal(alt, new, check_exact=True)


COPY_EXCLUDED = ("specimen_id", "configuration", "arm", "with_noise", "paired_with")


def test_tm12_no_noise_rows():
    for dvh in (_DEFAULT, None):
        m = manifest(dvh=dvh)
        nn = m[m["configuration"] == "full_no_noise"].reset_index(drop=True)
        full = m[m["configuration"] == "full"].reset_index(drop=True)
        assert len(nn) == N0
        assert nn["specimen_id"].tolist() == [ID_BASE + 300 + i for i in range(N0)]
        assert nn["paired_with"].tolist() == full["specimen_id"].tolist()[:N0]
        assert nn["paired_with"].tolist() == [ID_BASE + i for i in range(N0)]
        assert (nn["arm"] == "ih5").all()
        assert (~nn["with_noise"]).all()
        rest = m[m["configuration"] != "full_no_noise"]
        assert rest["with_noise"].all() and (rest["paired_with"] == -1).all()
        cols = [c for c in m.columns if c not in COPY_EXCLUDED]
        assert "manifest_seed" in cols and "real_swc_path" in cols and "cm_true" in cols
        assert_frame_equal(nn[cols], full.iloc[:N0][cols].reset_index(drop=True), check_exact=True)
    # n_no_noise = 0: no such row, the rest unchanged
    m0 = manifest(n0=0)
    assert len(m0) == 3 * N and "full_no_noise" not in set(m0["configuration"])
    assert list(m0.columns) == list(base_manifest().columns)
    assert_frame_equal(m0, base_manifest().iloc[:3 * N], check_exact=True)
    assert m0["with_noise"].all() and (m0["paired_with"] == -1).all()
    # n_no_noise = n: one per full twin
    mN = manifest(n0=N)
    assert (mN.loc[mN["configuration"] == "full_no_noise", "paired_with"].tolist()
            == mN.loc[mN["configuration"] == "full", "specimen_id"].tolist())
    # refusals
    real = make_real_cells()
    bad = []
    for v in (-1, N + 1, 2.5, True, "3", None):
        if not _raises(TM.draw_twin_manifest, real, n_per_configuration=N, n_no_noise=v,
                       truth_kwargs=truth_kwargs()):
            bad.append(v)
    if not _raises(TM.draw_twin_manifest, real, n_per_configuration=2, n_no_noise=3,
                   truth_kwargs=truth_kwargs()):
        bad.append("n0=3 > n=2")
    # the default n_no_noise = 5 with n_per_configuration = 4 is the same refusal
    if not _raises(TM.draw_twin_manifest, real, n_per_configuration=4, truth_kwargs=truth_kwargs()):
        bad.append("default 5 > n=4")
    assert not bad, "n_no_noise not refused: {}".format(bad)


def test_tm13_theta_of_paired_row():
    m = base_manifest()
    idx = m.set_index("specimen_id", drop=False)
    nn = m[m["configuration"] == "full_no_noise"]
    assert len(nn) == N0
    for _, r in nn.iterrows():
        partner = idx.loc[int(r["paired_with"])]
        assert partner["configuration"] == "full"
        th, thp = TM.twin_theta(r), TM.twin_theta(partner)
        assert th == thp and list(th) == list(thp), (th, thp)
        sp, spp = TM.twin_spec_of(r), TM.twin_spec_of(partner)
        assert dict(sp.theta) == dict(spp.theta)
        assert sp.twin_id == int(r["specimen_id"]) != spp.twin_id
        assert sp.configuration == "full_no_noise" and sp.twin_of_specimen_id == spp.twin_of_specimen_id
        assert sp.v_rest_mV == spp.v_rest_mV and sp.F == spp.F


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python {}".format(platform.python_version())]
    for package in REPORT_PACKAGES:
        try:
            parts.append("{} {}".format(package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("{} (not installed)".format(package))
    parts.append(platform.platform())
    parts.append("seed {}".format(SEED))
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== {}".format(Path(__file__).name))
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
            status, detail = "FAIL", str(exc) or traceback.format_exc()
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print("{:<5}  {:<{w}}  {:8.3f}s  {}".format(status, name, seconds, headline[:160], w=width).rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- {}: {}\n{}".format(status, name, detail.strip()))

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("{} {}".format(n, s.lower()) for s, n in counts.items()))
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
