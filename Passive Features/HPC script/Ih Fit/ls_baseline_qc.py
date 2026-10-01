# -*- coding: utf-8 -*-
"""
ls_baseline_qc.py
=================

Does the Long Square baseline change across a sweep? (D-017, open points (a)
and (c); topic document claude/TEEG_Stage7_noise_model_2026-09-28.md, 3.6.7.)

Why this exists
---------------
The fitter's Long Square pre-window is 265 ms long on every L3_exc cell. A
window that short cannot tell a CREEP (the baseline moving steadily in one
direction, which would continue through the 1 s step and bias the I_h
parameters) from a slow stationary WANDER (which would not). Records longer
than the pre-window can. Two are read here, per hyperpolarising single Long
Square sweep -- the same sweeps `noise_calibration.py` measures:

ARCHIVE (always; no network; data we hold)
    From the stored sweep itself:
      pre window   the fitter's LS pre-window [t0, t_on - 5 ms), exactly as
                   noise_calibration reads it (_pre_window_mask)
      end window   the last END_WINDOW_MS (500 ms, Allen's post-stimulus
                   stability epoch) of the stored sweep, used only if it lies
                   entirely after the step ends; otherwise NaN and flag
                   'tail_short'
    Columns:
      tail_ms        stored recording after the step (t_last - t_off)
      end_gap_ms     start of the end window minus t_off
      delta_end_mV   mean(end window) - mean(pre window), SIGNED
      pre_slope_mV_per_s, end_slope_mV_per_s   least-squares slopes
      creep_pred_mV  pre_slope x (centre of end window - centre of pre
                     window): what a creep that CONTINUED linearly at the
                     pre-window's own slope would give
    A creep that continues makes delta_end_mV follow creep_pred_mV (same
    sign, similar size); a wander leaves them unrelated. After a
    hyperpolarising step the rebound decays during the tail, so read
    delta_end_mV with end_gap_ms and end_slope_mV_per_s beside it.

ALLEN (optional: --allen fetch | cache)
    The Allen Institute's own per-sweep QC features, from the public RMA API
    (model EphysSweep -- the records AllenSDK's CellTypesApi.get_ephys_sweeps
    returns; same query URL), joined to the archive sweeps by sweep number:
      vm_delta_mv        |mean Vm over the 500 ms before the stimulus -
                         mean Vm over the last 500 ms of the recording|
                         (IPFX 2.1.2 qc_feature_extractor; criterion < 1 mV)
      slow_noise_rms_mv  SD of the 500 ms before the stimulus (ddof 0)
      pre_vm_mv, post_vm_mv, slow_vm_mv, pre_noise_rms_mv, post_noise_rms_mv,
      stimulus_name, stimulus_absolute_amplitude, stimulus_start_time,
      stimulus_duration, num_spikes
    plus a join check: allen_pre_offset_mV = (archive pre-window mean - LJP)
    - allen_pre_vm_mv. The archive is LJP-corrected (+14 mV for Allen cells)
    and the Allen values are not; the two windows differ (265 vs 500 ms), so
    a correct join reads a few tenths of a mV at most, a wrong one does not.
    Raw JSON is cached per specimen, so a second run needs no network
    (--allen cache). The cluster reaches the API after `module load proxy`
    (urllib honours http_proxy). Without network anywhere, --print-urls lists
    the URLs: open each in a browser, save it under the printed name in the
    cache directory, then run with --allen cache.

Nothing downstream reads these columns; they inform a decision the user has
not taken (D-017 (a), (c)). No simulation; seconds per cell.

Smoke: smoke_ls_baseline_qc.py (synthetic archive with a known creep, a
known short tail, cached Allen records; no network).
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


#: Allen's post-stimulus stability epoch (IPFX epochs.POSTSTIM_STABILITY_EPOCH).
END_WINDOW_MS = 500.0

#: AllenSDK's default API host (allensdk.api.api.Api.default_api_url).
ALLEN_BASE_URL = "http://api.brain-map.org"

#: EphysSweep fields kept, in output order (prefixed 'allen_').
ALLEN_FIELDS = (
    "stimulus_name", "stimulus_absolute_amplitude", "stimulus_start_time",
    "stimulus_duration", "num_spikes", "vm_delta_mv", "slow_noise_rms_mv",
    "slow_vm_mv", "pre_vm_mv", "post_vm_mv", "pre_noise_rms_mv",
    "post_noise_rms_mv",
)

ARCHIVE_COLUMNS = [
    "specimen_id", "sweep_number", "amplitude_pA", "fs_Hz", "n_samples",
    "t_on_ms", "t_off_ms", "tail_ms", "n_trailing_nonfinite",
    "pre_ms", "pre_mean_mV", "pre_slope_mV_per_s",
    "end_window_ms", "end_gap_ms", "end_mean_mV", "end_slope_mV_per_s",
    "delta_end_mV", "creep_pred_mV", "flag",
]
ALLEN_COLUMNS = (["allen_status"] + ["allen_" + f for f in ALLEN_FIELDS]
                 + ["allen_pre_offset_mV", "ljp_mV"])


# ---------------------------------------------------------------------------
# Pure numerics (no I/O)
# ---------------------------------------------------------------------------
def mean_and_slope(t_s: np.ndarray, v_mV: np.ndarray):
    """Mean (mV) and least-squares slope (mV/s) of one window; NaN pair when
    fewer than 3 samples."""
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(v_mV, dtype=float)
    if t.size < 3:
        return float("nan"), float("nan")
    slope = float(np.polyfit(t - t.mean(), v, 1)[0])
    return float(v.mean()), slope


def baseline_record(t_s: np.ndarray, v_mV: np.ndarray, *, t_on_s: float,
                    t_off_s: float, pre_mask: np.ndarray, fs_Hz: float,
                    end_window_ms: float = END_WINDOW_MS) -> Dict[str, Any]:
    """Archive columns for ONE stored sweep (see the module docstring).

    t_s, v_mV : the stored sweep (s from its first sample; mV, LJP-corrected)
    t_on_s, t_off_s : step onset and end (s), as the loader detected them
    pre_mask : boolean mask of the fitter's pre-window on t_s
    """
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(v_mV, dtype=float)
    out: Dict[str, Any] = {c: np.nan for c in ARCHIVE_COLUMNS}
    out["flag"] = "ok"
    finite = np.isfinite(v)
    # a recording that stopped early ends in non-finite samples; the end
    # window is taken from the last FINITE sample (IPFX's recording epoch)
    last = int(np.flatnonzero(finite)[-1]) + 1 if finite.any() else 0
    out["n_trailing_nonfinite"] = int(v.size - last)
    out["n_samples"] = int(v.size)
    out["fs_Hz"] = float(fs_Hz)
    out["t_on_ms"] = 1e3 * float(t_on_s)
    out["t_off_ms"] = 1e3 * float(t_off_s)
    if last == 0:
        out["flag"] = "no_finite_samples"
        return out
    t_last = float(t[last - 1])
    out["tail_ms"] = 1e3 * (t_last - float(t_off_s))

    pm = np.asarray(pre_mask, dtype=bool) & finite
    out["pre_ms"] = 1e3 * float(np.count_nonzero(pm)) / float(fs_Hz)
    pre_mean, pre_slope = mean_and_slope(t[pm], v[pm])
    out["pre_mean_mV"], out["pre_slope_mV_per_s"] = pre_mean, pre_slope
    if not np.isfinite(pre_mean):
        out["flag"] = "pre_window_short"
        return out

    n_end = int(round(end_window_ms * 1e-3 * float(fs_Hz)))
    out["end_window_ms"] = float(end_window_ms)
    i0 = last - n_end
    if n_end < 3 or i0 < 0 or float(t[i0]) <= float(t_off_s):
        out["flag"] = "tail_short"
        return out
    out["end_gap_ms"] = 1e3 * (float(t[i0]) - float(t_off_s))
    te, ve = t[i0:last], v[i0:last]
    end_mean, end_slope = mean_and_slope(te, ve)
    out["end_mean_mV"], out["end_slope_mV_per_s"] = end_mean, end_slope
    out["delta_end_mV"] = end_mean - pre_mean
    out["creep_pred_mV"] = pre_slope * (float(te.mean()) - float(t[pm].mean()))
    return out


# ---------------------------------------------------------------------------
# Allen RMA API (EphysSweep): query, envelope, join
# ---------------------------------------------------------------------------
def allen_query_url(specimen_id: int, base_url: str = ALLEN_BASE_URL) -> str:
    """The URL AllenSDK's CellTypesApi.get_ephys_sweeps requests:
    model_query('EphysSweep', criteria='[specimen_id$eq<id>]', num_rows='all'),
    quoted as AllenSDK quotes it (safe=';/?:@&=+$,')."""
    raw = ("%s/api/v2/data/query.json?q=model::EphysSweep,rma::criteria,"
           "[specimen_id$eq%d],rma::options[num_rows$eq'all']"
           % (base_url.rstrip("/"), int(specimen_id)))
    return urllib.parse.quote(raw, safe=";/?:@&=+$,")


def cache_path(cache_dir: Union[str, Path], specimen_id: int) -> Path:
    return Path(cache_dir) / ("ephys_sweeps_%d.json" % int(specimen_id))


def parse_allen_envelope(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The records of an RMA response (allensdk Api.read_data: doc['msg']).
    Raises ValueError on an unsuccessful or malformed envelope."""
    if not isinstance(doc, Mapping) or not doc.get("success", False):
        msg = doc.get("msg") if isinstance(doc, Mapping) else doc
        raise ValueError("RMA query not successful: %r" % (msg,))
    recs = doc.get("msg")
    if not isinstance(recs, list):
        raise ValueError("RMA envelope has no record list")
    return [dict(r) for r in recs if isinstance(r, Mapping)]


def fetch_allen_records(specimen_id: int, *, cache_dir: Union[str, Path],
                        mode: str, base_url: str = ALLEN_BASE_URL,
                        timeout_s: float = 60.0,
                        opener: Optional[Callable] = None,
                        ) -> List[Dict[str, Any]]:
    """EphysSweep records of one specimen. mode 'cache' reads the cached JSON
    only; mode 'fetch' uses the cache when present and downloads otherwise,
    writing the raw JSON to the cache. Raises on any failure (the caller
    records it per specimen). `opener` defaults to urllib.request.urlopen,
    looked up at call time (the smoke suite replaces it)."""
    if opener is None:
        opener = urllib.request.urlopen
    p = cache_path(cache_dir, specimen_id)
    if p.exists():
        return parse_allen_envelope(json.loads(p.read_text()))
    if mode != "fetch":
        raise FileNotFoundError("no cached Allen JSON at %s" % p)
    with opener(allen_query_url(specimen_id, base_url), timeout=timeout_s) as r:
        raw = r.read()
    doc = json.loads(raw.decode("utf-8"))
    recs = parse_allen_envelope(doc)         # validate BEFORE caching
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return recs


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def allen_row(rec: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The 'allen_*' columns of one EphysSweep record (NaN when absent)."""
    out: Dict[str, Any] = {}
    for f in ALLEN_FIELDS:
        val = None if rec is None else rec.get(f)
        out["allen_" + f] = (val if f == "stimulus_name"
                             else _num(val))
    return out


def join_allen(rows: List[Dict[str, Any]], records: Sequence[Mapping[str, Any]],
               *, ljp_mV: float) -> None:
    """Add the Allen columns to the archive rows of ONE specimen, in place,
    by sweep number."""
    by_sweep = {}
    for r in records:
        try:
            by_sweep[int(r.get("sweep_number"))] = r
        except (TypeError, ValueError):
            continue
    for row in rows:
        rec = by_sweep.get(int(row["sweep_number"])) if np.isfinite(
            _num(row.get("sweep_number"))) else None
        row.update(allen_row(rec))
        row["ljp_mV"] = float(ljp_mV)
        row["allen_status"] = "matched" if rec is not None else "no_record"
        row["allen_pre_offset_mV"] = (
            (_num(row.get("pre_mean_mV")) - float(ljp_mV))
            - _num(row["allen_pre_vm_mv"]) if rec is not None else np.nan)


def allen_failed(rows: List[Dict[str, Any]], err: str, *, ljp_mV: float) -> None:
    """Mark the rows of one specimen whose Allen records could not be read."""
    for row in rows:
        row.update(allen_row(None))
        row["ljp_mV"] = float(ljp_mV)
        row["allen_status"] = "error: " + err
        row["allen_pre_offset_mV"] = np.nan


# ---------------------------------------------------------------------------
# Archive I/O (reuses noise_calibration, so the sweeps and the pre-window are
# exactly the ones the noise table was measured on)
# ---------------------------------------------------------------------------
def archive_ljp_mV(specimen_dir: Union[str, Path]) -> float:
    meta = json.loads((Path(specimen_dir) / "metadata.json").read_text())
    return float(meta.get("ljp_correction_mV", np.nan))


def measure_specimen(specimen_dir: Union[str, Path], *, mono, nc,
                     end_window_ms: float = END_WINDOW_MS) -> List[Dict[str, Any]]:
    """Archive columns, one row per hyperpolarising SINGLE Long Square sweep.
    Raises on a cell that cannot be loaded (the caller records it)."""
    d = Path(specimen_dir)
    sid = nc.specimen_id_of(d)
    cd = nc.load_for_noise(d, mono=mono)
    if sid is None:
        sid = int(getattr(cd, "specimen_id", -1))
    rows = []
    for b in list(getattr(cd, "long_square_subthreshold", []) or []):
        if int(getattr(b, "n_repeats_averaged", 1)) != 1:
            continue                       # averaged bundles: not one sweep
        t = np.asarray(b.t, dtype=float)
        row = baseline_record(
            t, np.asarray(b.v_mV, dtype=float),
            t_on_s=float(b.stim_onset_s),
            t_off_s=float(b.stim_onset_s) + float(b.stim_duration_s),
            pre_mask=nc._pre_window_mask(mono, b),
            fs_Hz=float(b.sampling_rate_Hz), end_window_ms=end_window_ms)
        row["specimen_id"] = int(sid)
        sn = list(getattr(b, "sweep_numbers", []) or [])
        row["sweep_number"] = int(sn[0]) if sn else -1
        row["amplitude_pA"] = float(b.amplitude_pA)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-cell summary (pure pandas)
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per specimen. sign_agree counts sweeps where delta_end_mV and
    creep_pred_mV have the same sign, out of n_valid (both finite)."""
    out = []
    for sid, g in df.groupby("specimen_id", sort=True):
        ok = g[np.isfinite(g["delta_end_mV"]) & np.isfinite(g["creep_pred_mV"])]
        r = {
            "specimen_id": int(sid),
            "n_sweeps": int(len(g)),
            "n_valid": int(len(ok)),
            "n_tail_short": int((g["flag"] == "tail_short").sum()),
            "tail_ms_min": float(np.nanmin(g["tail_ms"])) if g["tail_ms"].notna().any() else np.nan,
            "tail_ms_median": float(np.nanmedian(g["tail_ms"])) if g["tail_ms"].notna().any() else np.nan,
            "delta_end_median_mV": float(np.median(ok["delta_end_mV"])) if len(ok) else np.nan,
            "abs_delta_end_median_mV": float(np.median(np.abs(ok["delta_end_mV"]))) if len(ok) else np.nan,
            "creep_pred_median_mV": float(np.median(ok["creep_pred_mV"])) if len(ok) else np.nan,
            "sign_agree": int((np.sign(ok["delta_end_mV"]) == np.sign(ok["creep_pred_mV"])).sum()),
        }
        if "allen_status" in g.columns:
            m = g[g["allen_status"] == "matched"]
            r["n_allen_matched"] = int(len(m))
            vd = m["allen_vm_delta_mv"].dropna()
            r["allen_vm_delta_median_mV"] = float(np.median(vd)) if len(vd) else np.nan
            off = m["allen_pre_offset_mV"].dropna()
            r["allen_abs_pre_offset_median_mV"] = float(np.median(np.abs(off))) if len(off) else np.nan
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-sweep Long Square baseline change: archive tail "
                    "(always) and Allen QC features (optional).")
    ap.add_argument("--group-dir", required=True,
                    help="archive group, e.g. <ARCHIVE_ROOT>/L3_exc")
    ap.add_argument("--out", required=True, help="per-sweep CSV to write")
    ap.add_argument("--summary-out", default=None,
                    help="per-cell CSV (default: <out stem>_cells.csv)")
    ap.add_argument("--allen", choices=("off", "fetch", "cache"), default="off",
                    help="Allen QC features: off (default), fetch (network; "
                         "cached), cache (cached JSON only)")
    ap.add_argument("--allen-cache-dir", default=None,
                    help="default: <out dir>/allen_ephys_sweeps")
    ap.add_argument("--allen-base-url", default=ALLEN_BASE_URL)
    ap.add_argument("--allen-timeout-s", type=float, default=60.0)
    ap.add_argument("--allen-max-network-failures", type=int, default=2,
                    help="after this many CONSECUTIVE network errors in fetch "
                         "mode, stop using the network (remaining cells read "
                         "the cache only), so a run without a proxy fails in "
                         "minutes rather than one timeout per cell")
    ap.add_argument("--print-urls", action="store_true",
                    help="print each specimen's Allen URL and cache file "
                         "name, then exit (for a manual download)")
    ap.add_argument("--end-window-ms", type=float, default=END_WINDOW_MS)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out = Path(args.out)
    cache_dir = (Path(args.allen_cache_dir) if args.allen_cache_dir
                 else out.parent / "allen_ephys_sweeps")
    dirs = sorted(p for p in Path(args.group_dir).glob("specimen_*") if p.is_dir())
    if args.max_cells is not None:
        dirs = dirs[:int(args.max_cells)]
    if not dirs:
        print("[lsqc] ERROR: no specimen_* folder under %s" % args.group_dir)
        return 2

    sys.path.insert(0, args.code_dir)
    import noise_calibration as nc

    if args.print_urls:
        for d in dirs:
            sid = nc.specimen_id_of(d)
            print("%s  ->  %s" % (allen_query_url(sid, args.allen_base_url),
                                  cache_path(cache_dir, sid)))
        print("[lsqc] %d URL(s); save each response under the name shown, "
              "then run with --allen cache" % len(dirs))
        return 0

    import passive_fitting_hpc_fixed as mono

    rows_all: List[Dict[str, Any]] = []
    n_cells_ok = 0
    allen_mode = args.allen
    net_failures = 0
    for i, d in enumerate(dirs):
        sid = nc.specimen_id_of(d)
        try:
            rows = measure_specimen(d, mono=mono, nc=nc,
                                    end_window_ms=args.end_window_ms)
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised
            print("[lsqc] %3d/%d %s  ARCHIVE ERROR %s: %s"
                  % (i + 1, len(dirs), d.name, type(exc).__name__, exc))
            continue
        if not rows:
            print("[lsqc] %3d/%d %s  no single hyperpolarising Long Square sweep"
                  % (i + 1, len(dirs), d.name))
            continue
        n_cells_ok += 1
        if args.allen != "off":
            ljp = archive_ljp_mV(d)
            try:
                recs = fetch_allen_records(sid, cache_dir=cache_dir,
                                           mode=allen_mode,
                                           base_url=args.allen_base_url,
                                           timeout_s=args.allen_timeout_s)
                join_allen(rows, recs, ljp_mV=ljp)
                net_failures = 0
            except Exception as exc:  # noqa: BLE001 -- recorded per specimen
                allen_failed(rows, "%s: %s" % (type(exc).__name__, exc),
                             ljp_mV=ljp)
                if allen_mode == "fetch" and isinstance(exc, OSError) \
                        and not isinstance(exc, FileNotFoundError):
                    net_failures += 1
                    if net_failures >= args.allen_max_network_failures:
                        allen_mode = "cache"
                        print("[lsqc] %d consecutive network error(s): no more "
                              "downloads in this run (last: %s); remaining "
                              "cells read the cache only. Did you run "
                              "'module load proxy'?" % (net_failures, exc))
        g = pd.DataFrame(rows)
        msg = ("[lsqc] %3d/%d %s  %d sweep(s) | tail %.0f-%.0f ms | "
               "delta_end median %+.3f mV | creep_pred median %+.3f mV"
               % (i + 1, len(dirs), d.name, len(rows),
                  np.nanmin(g["tail_ms"]), np.nanmax(g["tail_ms"]),
                  np.nanmedian(g["delta_end_mV"]) if g["delta_end_mV"].notna().any() else np.nan,
                  np.nanmedian(g["creep_pred_mV"]) if g["creep_pred_mV"].notna().any() else np.nan))
        if args.allen != "off":
            n_m = int((g["allen_status"] == "matched").sum())
            msg += " | Allen %d/%d matched" % (n_m, len(rows))
            if n_m == 0:
                msg += " (%s)" % str(g["allen_status"].iloc[0])[:80]
        print(msg, flush=True)
        rows_all.extend(rows)

    if not rows_all:
        print("[lsqc] ERROR: no sweep measured in %d cell folder(s)" % len(dirs))
        return 2

    cols = ARCHIVE_COLUMNS + (ALLEN_COLUMNS if args.allen != "off" else [])
    df = pd.DataFrame(rows_all).reindex(columns=cols)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summ = summarise(df)
    sout = (Path(args.summary_out) if args.summary_out
            else out.with_name(out.stem + "_cells.csv"))
    summ.to_csv(sout, index=False)

    n_end = int(np.isfinite(df["delta_end_mV"]).sum())
    print("[lsqc] archive: %d sweep(s) in %d cell(s); tail after the step: "
          "min %.0f ms, median %.0f ms; end window (%.0f ms) usable in %d/%d sweeps"
          % (len(df), n_cells_ok, np.nanmin(df["tail_ms"]),
             np.nanmedian(df["tail_ms"]), args.end_window_ms, n_end, len(df)))
    if args.allen != "off":
        m = df[df["allen_status"] == "matched"]
        off = np.abs(m["allen_pre_offset_mV"].dropna())
        print("[lsqc] allen: mode %s%s; %d/%d sweep(s) matched by sweep number; "
              "median |pre-window offset| %s mV (LJP removed); cache %s"
              % (args.allen,
                 "" if allen_mode == args.allen else " (network disabled)",
                 len(m), len(df),
                 ("%.2f" % float(np.median(off))) if len(off) else "nan",
                 cache_dir))
    print("[lsqc] wrote %s and %s" % (out, sout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
