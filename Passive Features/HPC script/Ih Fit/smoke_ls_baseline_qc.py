# -*- coding: utf-8 -*-
"""
smoke_ls_baseline_qc.py -- smoke suite for ls_baseline_qc.py (L1-L10)
======================================================================

No network, no simulation (NEURON is only imported, through the loader).
Builds a synthetic Phase-0 archive group in a temporary directory, with
traces whose answers are known in closed form, and checks:

  L1  archive arithmetic on a known creep: delta_end_mV and creep_pred_mV
      equal slope x (centre of end window - centre of pre-window), both
      signs; tail_ms, end_gap_ms, pre_ms as constructed; flag 'ok'
  L2  a recording that ends in non-finite samples: the end window is taken
      from the last finite sample, and n_trailing_nonfinite counts the rest
  L3  a tail shorter than the end window: flag 'tail_short', delta NaN,
      tail_ms still reported
  L4  the Allen URL is AllenSDK's (model_query EphysSweep, num_rows 'all',
      quoted with safe=';/?:@&=+$,')
  L5  the RMA envelope: success -> records; success false -> ValueError;
      fetch mode downloads once, validates BEFORE caching, then reads the
      cache without calling the network again
  L6  the join: by sweep number only; a 'Test' sweep in the Allen list is
      not joined; allen_pre_offset_mV = (pre mean - LJP) - pre_vm_mv
  L7  CLI end to end, --allen cache: exit 0, both CSVs written, the two
      positive marker lines printed, a broken cell reported and skipped, a
      cell without cache marked 'error: FileNotFoundError ...' with its
      archive columns intact, and urlopen never called
  L8  CLI --allen off writes no Allen column; --print-urls lists one URL per
      cell and exits 0 without loading any cell; an empty group exits 2
  L9  byte safety: this file and ls_baseline_qc.py are pure ASCII
  L10 fetch mode without a network: after --allen-max-network-failures
      consecutive network errors no further download is attempted, the
      remaining cells read the cache only, and every row says why

Run from this folder (any env that runs noise_calibration.py):

    python smoke_ls_baseline_qc.py

Expected last line: "SMOKE ls_baseline_qc: 10/10 passed". Exit 1 on any FAIL.
"""

import contextlib
import io
import json
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ls_baseline_qc as q  # noqa: E402

FS = 20000.0          # Hz
T_PRE = 0.270         # s, onset after the first stored sample
T_STEP = 1.000        # s
LJP = 14.0            # mV, as the Allen archive cells carry
V0 = -70.0            # mV, LJP-corrected baseline
SIGMA = 0.02          # mV, white noise


def _trace(slope_mV_per_s, amp_pA, tail_s, rng, nan_tail_s=0.0):
    """One stored Long Square sweep: baseline + linear creep + a step
    response that relaxes with tau = 20 ms (gone well before the end
    window) + white noise. Returns (v_mV, i_pA)."""
    n = int(round((T_PRE + T_STEP + tail_s) * FS))
    t = np.arange(n) / FS
    on = int(round(T_PRE * FS))
    off = int(round((T_PRE + T_STEP) * FS))
    i = np.zeros(n)
    i[on:off] = amp_pA
    dv = 5.0 * amp_pA / 100.0          # mV, negative for hyperpolarising
    resp = np.zeros(n)
    ts = t[on:off] - t[on]
    resp[on:off] = dv * (1.0 - np.exp(-ts / 0.02))
    td = t[off:] - t[off]
    resp[off:] = resp[off - 1] * np.exp(-td / 0.02)
    v = V0 + slope_mV_per_s * t + resp + rng.normal(0.0, SIGMA, n)
    if nan_tail_s > 0:
        v[n - int(round(nan_tail_s * FS)):] = np.nan
    return v, i


def _write_cell(group, sid, sweeps, *, broken=False):
    """sweeps: list of (sweep_number, slope, amp_pA, tail_s, nan_tail_s)."""
    d = Path(group) / ("specimen_%d" % sid)
    d.mkdir(parents=True)
    if broken:
        (d / "metadata.json").write_text("{not json")
        return d
    rng = np.random.default_rng(sid)
    arrays, info = {}, []
    for k, (sn, slope, amp, tail, nan_tail) in enumerate(sweeps):
        v, i = _trace(slope, amp, tail, rng, nan_tail)
        arrays["v_%d" % k], arrays["i_%d" % k] = v, i
        info.append(dict(index=k, sweep_number=int(sn),
                         detected_amplitude_pA=float(amp),
                         sampling_rate_Hz=FS, stimulus_name="Long Square",
                         n_samples=int(v.size)))
    arrays["n_sweeps"] = np.array([len(sweeps)], dtype=np.int64)
    np.savez_compressed(d / "ls_sweeps.npz", **arrays)
    meta = dict(specimen_id=sid, ljp_correction_mV=LJP, layer="synthetic",
                dendrite_type="spiny", ls_sweeps=info,
                full_allen_metadata=dict(structure_layer_name="synthetic",
                                         dendrite_type="spiny", id=sid))
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


# cell A: three sweeps with a creep of +/-0.8 mV/s and a 1.5 s tail; the last
# one ends in 0.1 s of NaN. cell B: two sweeps with a 0.3 s tail (short).
SID_A, SID_B, SID_C = 900000201, 900000202, 900000203
SWEEPS_A = [(31, +0.8, -50.0, 1.5, 0.0), (32, +0.8, -90.0, 1.5, 0.0),
            (33, -0.8, -130.0, 1.5, 0.1)]
SWEEPS_B = [(41, 0.0, -50.0, 0.3, 0.0), (42, 0.0, -90.0, 0.3, 0.0)]
PRE_OFFSET = 0.05     # mV: the Allen pre_vm_mv we write = pre mean - LJP - this


def _expected_delta(slope, tail_s, nan_tail_s):
    """slope x (centre of end window - centre of pre-window), in mV."""
    n = int(round((T_PRE + T_STEP + tail_s) * FS))
    t = np.arange(n) / FS
    pre = t < (T_PRE - 5e-3)
    last = n - int(round(nan_tail_s * FS))
    n_end = int(round(q.END_WINDOW_MS * 1e-3 * FS))
    return slope * (t[last - n_end:last].mean() - t[pre].mean())


def _envelope(records, success=True):
    return {"success": success, "id": 0, "start_row": 0,
            "num_rows": len(records), "total_rows": len(records),
            "msg": records}


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
                RESULTS.append((name, True, ""))
                print("PASS  %s" % name)
            except BaseException as exc:   # SystemExit included
                RESULTS.append((name, False, repr(exc)))
                print("FAIL  %s  --  %r" % (name, exc))
                traceback.print_exc()
        return run
    return deco


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lsqc_smoke_"))
    group = tmp / "L3_exc"
    _write_cell(group, SID_A, SWEEPS_A)
    _write_cell(group, SID_B, SWEEPS_B)
    _write_cell(group, SID_C, [], broken=True)
    import noise_calibration as nc
    import passive_fitting_hpc_fixed as mono
    rows_a = q.measure_specimen(group / ("specimen_%d" % SID_A), mono=mono, nc=nc)
    rows_b = q.measure_specimen(group / ("specimen_%d" % SID_B), mono=mono, nc=nc)
    by_sn = {r["sweep_number"]: r for r in rows_a}

    @check("L1 archive arithmetic on a known creep (both signs)")
    def _l1():
        assert sorted(by_sn) == [31, 32, 33], sorted(by_sn)
        for sn, slope, amp, tail, nan_tail in SWEEPS_A[:2]:
            r = by_sn[sn]
            exp = _expected_delta(slope, tail, nan_tail)
            assert r["flag"] == "ok", r["flag"]
            assert abs(r["delta_end_mV"] - exp) < 0.01, (r["delta_end_mV"], exp)
            assert abs(r["creep_pred_mV"] - exp) < 0.06, (r["creep_pred_mV"], exp)
            assert abs(r["tail_ms"] - 1e3 * tail) <= 1e3 / FS + 1e-9, r["tail_ms"]
            assert abs(r["end_gap_ms"] - (1e3 * tail - q.END_WINDOW_MS)) <= 1e3 / FS + 1e-9
            assert abs(r["pre_ms"] - 1e3 * (T_PRE - 5e-3)) <= 1e3 / FS + 1e-9, r["pre_ms"]
            assert abs(r["pre_slope_mV_per_s"] - slope) < 0.03, r["pre_slope_mV_per_s"]
            assert abs(r["end_slope_mV_per_s"] - slope) < 0.03, r["end_slope_mV_per_s"]
            assert abs(r["amplitude_pA"] - amp) < 1e-6, r["amplitude_pA"]
        r = by_sn[33]
        assert r["delta_end_mV"] < -1.0 and r["creep_pred_mV"] < -1.0, r

    @check("L2 trailing non-finite samples are excluded from the end window")
    def _l2():
        sn, slope, amp, tail, nan_tail = SWEEPS_A[2]
        r = by_sn[sn]
        assert r["n_trailing_nonfinite"] == int(round(nan_tail * FS)), r["n_trailing_nonfinite"]
        exp = _expected_delta(slope, tail, nan_tail)
        assert np.isfinite(r["delta_end_mV"]), r
        assert abs(r["delta_end_mV"] - exp) < 0.01, (r["delta_end_mV"], exp)
        assert abs(r["tail_ms"] - 1e3 * (tail - nan_tail)) <= 1e3 / FS + 1e-9, r["tail_ms"]

    @check("L3 a tail shorter than the end window is flagged, not used")
    def _l3():
        assert len(rows_b) == 2, len(rows_b)
        for r in rows_b:
            assert r["flag"] == "tail_short", r["flag"]
            assert np.isnan(r["delta_end_mV"]) and np.isnan(r["creep_pred_mV"]), r
            assert abs(r["tail_ms"] - 300.0) <= 1e3 / FS + 1e-9, r["tail_ms"]
            assert np.isfinite(r["pre_mean_mV"]), r

    @check("L4 the Allen URL is AllenSDK's get_ephys_sweeps query")
    def _l4():
        url = q.allen_query_url(526785799)
        exp = ("http://api.brain-map.org/api/v2/data/query.json?q=model::EphysSweep,"
               "rma::criteria,%5Bspecimen_id$eq526785799%5D,"
               "rma::options%5Bnum_rows$eq%27all%27%5D")
        assert url == exp, url
        assert q.allen_query_url(1, "https://x.org/").startswith("https://x.org/api/v2/")

    @check("L5 envelope parsing; fetch validates before caching, then reads the cache")
    def _l5():
        recs = [{"sweep_number": 5, "vm_delta_mv": 0.3}]
        assert q.parse_allen_envelope(_envelope(recs)) == recs
        for bad in (_envelope(recs, success=False), {"success": True}, ["x"]):
            try:
                q.parse_allen_envelope(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("no ValueError for %r" % (bad,))
        cdir = tmp / "cache_l5"
        calls = []

        def opener_bad(url, timeout):
            calls.append(url)
            return _FakeResponse(json.dumps(_envelope([], success=False)).encode())
        try:
            q.fetch_allen_records(7, cache_dir=cdir, mode="fetch", opener=opener_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsuccessful envelope accepted")
        assert not q.cache_path(cdir, 7).exists(), "bad envelope was cached"

        def opener_ok(url, timeout):
            calls.append(url)
            return _FakeResponse(json.dumps(_envelope(recs)).encode())
        got = q.fetch_allen_records(7, cache_dir=cdir, mode="fetch", opener=opener_ok)
        assert got == recs and q.cache_path(cdir, 7).exists()
        n = len(calls)

        def opener_never(url, timeout):
            raise AssertionError("network used although the cache exists")
        assert q.fetch_allen_records(7, cache_dir=cdir, mode="fetch",
                                     opener=opener_never) == recs
        assert q.fetch_allen_records(7, cache_dir=cdir, mode="cache",
                                     opener=opener_never) == recs
        assert len(calls) == n
        try:
            q.fetch_allen_records(8, cache_dir=cdir, mode="cache", opener=opener_never)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("cache mode without a cache file did not raise")

    def _allen_records_a():
        recs = [{"sweep_number": 0, "stimulus_name": "Test", "vm_delta_mv": 9.9,
                 "pre_vm_mv": 0.0}]
        for k, (sn, slope, amp, tail, nan_tail) in enumerate(SWEEPS_A):
            pre = by_sn[sn]["pre_mean_mV"]
            recs.append({"sweep_number": sn, "stimulus_name": "Long Square",
                         "stimulus_absolute_amplitude": amp,
                         "vm_delta_mv": 0.1 * (k + 1), "slow_noise_rms_mv": 0.06,
                         "pre_vm_mv": pre - LJP - PRE_OFFSET, "post_vm_mv": None})
        return recs

    @check("L6 the join is by sweep number; a Test sweep is not joined")
    def _l6():
        rows = [dict(r) for r in rows_a]
        q.join_allen(rows, _allen_records_a(), ljp_mV=LJP)
        for k, r in enumerate(rows):
            assert r["allen_status"] == "matched", r["allen_status"]
            assert r["allen_stimulus_name"] == "Long Square"
            assert abs(r["allen_pre_offset_mV"] - PRE_OFFSET) < 1e-9, r["allen_pre_offset_mV"]
            assert np.isnan(r["allen_post_vm_mv"])
            assert abs(r["allen_stimulus_absolute_amplitude"] - r["amplitude_pA"]) < 1e-9
        assert np.allclose(sorted(r["allen_vm_delta_mv"] for r in rows), [0.1, 0.2, 0.3])
        rows2 = [dict(rows_a[0], sweep_number=99)]
        q.join_allen(rows2, _allen_records_a(), ljp_mV=LJP)
        assert rows2[0]["allen_status"] == "no_record"

    @check("L7 CLI end to end with --allen cache (no network)")
    def _l7():
        out = tmp / "run7" / "lsqc_L3.csv"
        cdir = tmp / "run7" / "allen_ephys_sweeps"
        cdir.mkdir(parents=True)
        q.cache_path(cdir, SID_A).write_text(json.dumps(_envelope(_allen_records_a())))
        real = urllib.request.urlopen

        def never(*a, **k):
            raise AssertionError("urlopen called in cache mode")
        urllib.request.urlopen = never
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = q.main(["--group-dir", str(group), "--out", str(out),
                             "--allen", "cache"])
        finally:
            urllib.request.urlopen = real
        log = buf.getvalue()
        assert rc == 0, (rc, log)
        assert "[lsqc] archive: 5 sweep(s) in 2 cell(s)" in log, log
        assert "end window (500 ms) usable in 3/5 sweeps" in log, log
        assert "[lsqc] allen: mode cache; 3/5 sweep(s) matched" in log, log
        assert "specimen_%d  ARCHIVE ERROR" % SID_C in log, log
        df = pd.read_csv(out)
        assert list(df.columns) == q.ARCHIVE_COLUMNS + q.ALLEN_COLUMNS, list(df.columns)
        b = df[df["specimen_id"] == SID_B]
        assert len(b) == 2 and b["allen_status"].str.startswith(
            "error: FileNotFoundError").all(), b["allen_status"].tolist()
        assert b["pre_mean_mV"].notna().all() and (b["flag"] == "tail_short").all()
        cells = pd.read_csv(out.with_name("lsqc_L3_cells.csv"))
        a = cells[cells["specimen_id"] == SID_A].iloc[0]
        assert int(a["n_valid"]) == 3 and int(a["sign_agree"]) == 3, a.to_dict()
        assert int(a["n_allen_matched"]) == 3, a.to_dict()
        assert abs(float(a["allen_abs_pre_offset_median_mV"]) - PRE_OFFSET) < 1e-9

    @check("L8 --allen off, --print-urls, empty group")
    def _l8():
        out = tmp / "run8" / "lsqc.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = q.main(["--group-dir", str(group), "--out", str(out)])
        assert rc == 0
        assert list(pd.read_csv(out).columns) == q.ARCHIVE_COLUMNS
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = q.main(["--group-dir", str(group), "--out", str(tmp / "x.csv"),
                         "--print-urls"])
        log = buf.getvalue()
        assert rc == 0 and log.count("api.brain-map.org") == 3, log
        assert "ephys_sweeps_%d.json" % SID_C in log, log
        assert not (tmp / "x.csv").exists()
        empty = tmp / "empty_group"
        empty.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = q.main(["--group-dir", str(empty), "--out", str(tmp / "y.csv")])
        assert rc == 2

    @check("L9 byte safety: pure ASCII sources")
    def _l9():
        for p in (HERE / "ls_baseline_qc.py", Path(__file__).resolve()):
            bad = [b for b in p.read_bytes() if b > 127]
            assert not bad, (p.name, bad[:5])

    @check("L10 fetch without a network stops downloading after N failures")
    def _l10():
        import urllib.error
        out = tmp / "run10" / "lsqc.csv"
        calls = []
        real = urllib.request.urlopen

        def offline(url, timeout=None):
            calls.append(url)
            raise urllib.error.URLError("no route to host (smoke)")
        urllib.request.urlopen = offline
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = q.main(["--group-dir", str(group), "--out", str(out),
                             "--allen", "fetch", "--allen-timeout-s", "1",
                             "--allen-max-network-failures", "1"])
        finally:
            urllib.request.urlopen = real
        log = buf.getvalue()
        assert rc == 0, (rc, log)
        assert len(calls) == 1, calls
        assert "1 consecutive network error(s): no more downloads" in log, log
        assert "[lsqc] allen: mode fetch (network disabled); 0/5 sweep(s) matched" in log, log
        df = pd.read_csv(out)
        a = df[df["specimen_id"] == SID_A]["allen_status"]
        b = df[df["specimen_id"] == SID_B]["allen_status"]
        assert a.str.startswith("error: URLError").all(), a.tolist()
        assert b.str.startswith("error: FileNotFoundError").all(), b.tolist()
        assert not any((tmp / "run10" / "allen_ephys_sweeps").glob("*.json"))

    for fn in (_l1, _l2, _l3, _l4, _l5, _l6, _l7, _l8, _l9, _l10):
        fn()
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("SMOKE ls_baseline_qc: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
