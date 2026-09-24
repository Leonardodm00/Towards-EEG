# -*- coding: utf-8 -*-
"""
check_dep_sweeps.py
===================

Which depolarising Long Square sweeps of a REAL archive group enter the I_h
fit's validation set, under the spike screen the loader uses now (D-016) and
under the one it replaced -- sweep by sweep, with the numbers that decide it.

Why this exists
---------------
The depolarising validation set of D-006 is built from the Long Square sweeps
above +5 pA and below the amplitude cap that contain no action potential. The
screen that decided "no action potential" used to be: peak above -20 mV, or
max |raw sample difference| / dt above 10 mV/ms. Recording noise alone
crosses the second test. For a stationary window with SD sigma and lag-1
correlation rho, the raw differences have SD sigma * sqrt(2 (1 - rho)); at the
L3_exc medians (0.0594 mV, 0.668, 50 kHz) that is 2.42 mV/ms, and the
maximum over one 1.3 s synthetic sweep came out at about 10 mV/ms. Every
synthetic twin of Stage 7 lost its depolarising steps to it. This script
shows what the old and the new screen do to the real sweeps, so the change
is judged on the recordings it applies to, not on the synthetic ones.

The new screen (passive_fitting_hpc_fixed.sweep_spike_reason) is the Allen
white paper's action-potential rule: dV/dt after a 10 kHz 4-pole Bessel
filter >= 20 mV/ms, peak >= -30 mV, height >= 2 mV, plus the old -20 mV
catch-all.

Output
------
--out: one row per depolarising sweep (every sweep above +5 pA, including
those above the cap, which are flagged `above_cap` and never kept):

    specimen_id, sweep_number, amplitude_pA, sampling_rate_Hz, duration_s,
    above_cap, v_max_mV,
    max_raw_dvdt_mV_per_ms       max |raw sample difference| / dt, whole sweep
    max_filtered_dvdt_mV_per_ms  max of the filtered dV/dt, whole sweep
    filtered                     False where the 10 kHz filter cannot apply
    sd_raw_dvdt_pre_mV_per_ms    SD of the raw differences before the step:
                                 the noise part of the first column
    legacy_flag                  the old screen's verdict (True = rejected)
    reason                       the new screen's verdict: "subthreshold", or
                                 why not (action_potential,
                                 peak_above_threshold, unusable)
    n_events                     action-potential events found
    kept_legacy, kept_new        in the validation set under each screen

Nothing is simulated and nothing is fitted; seconds per cell.

Run:
    python check_dep_sweeps.py --group-dir "<ARCHIVE_ROOT>/L3_exc" \\
        --out dep_screen_L3.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

SWEEP_COLUMNS: List[str] = [
    "specimen_id", "sweep_number", "amplitude_pA", "sampling_rate_Hz",
    "duration_s", "above_cap", "v_max_mV",
    "max_raw_dvdt_mV_per_ms", "max_filtered_dvdt_mV_per_ms", "filtered",
    "sd_raw_dvdt_pre_mV_per_ms",
    "legacy_flag", "reason", "n_events", "kept_legacy", "kept_new",
]


def _specimen_id(d: Path) -> Optional[int]:
    name = d.name
    if name.startswith("specimen_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def _pre_step_samples(v: np.ndarray, i_pA: np.ndarray) -> np.ndarray:
    """The samples before the current step, found the way the loader finds
    the onset (first sample where |I| exceeds half its maximum)."""
    i = np.asarray(i_pA, dtype=float)
    if i.size != v.size or not np.any(np.abs(i) > 0):
        return v[:0]
    active = np.abs(i) > 0.5 * float(np.max(np.abs(i)))
    return v[:int(np.argmax(active))]


def screen_specimen(specimen_dir: Union[str, Path], *, mono,
                    cap_pA: Optional[float] = 100.0) -> List[Dict[str, Any]]:
    """One row per depolarising Long Square sweep of one archive cell."""
    d = Path(specimen_dir)
    meta = json.loads((d / "metadata.json").read_text())
    infos = meta.get("ls_sweeps", []) or []
    f = d / "ls_sweeps.npz"
    if not infos or not f.exists():
        return []
    npz = np.load(f)
    sid = _specimen_id(d)
    if sid is None:
        sid = int(meta.get("specimen_id", -1))
    rows: List[Dict[str, Any]] = []
    for info in infos:
        amp = info.get("detected_amplitude_pA")
        if amp is None or amp <= 5.0:
            continue
        v = np.asarray(npz["v_%d" % info["index"]], dtype=float)
        key_i = "i_%d" % info["index"]
        i_pA = (np.asarray(npz[key_i], dtype=float) if key_i in npz.files
                else np.zeros_like(v))
        fs = float(info["sampling_rate_Hz"])
        above = cap_pA is not None and abs(float(amp)) > float(cap_pA)
        finite = v.size >= 3 and bool(np.all(np.isfinite(v)))
        raw = np.abs(np.diff(v)) * fs / 1e3 if finite else np.array([np.nan])
        if finite:
            dvdt, filt = mono.filtered_dvdt_mV_per_ms(v, fs)
            n_ev = len(mono.spike_events(v, fs))
        else:
            dvdt, filt, n_ev = np.array([np.nan]), False, 0
        pre = _pre_step_samples(v, i_pA)
        sd_pre = (float(np.std(np.diff(pre) * fs / 1e3, ddof=1))
                  if pre.size >= 4 else float("nan"))
        legacy = bool(mono.sweep_has_spike_legacy(v, fs))
        why = mono.sweep_spike_reason(v, fs)
        rows.append({
            "specimen_id": sid,
            "sweep_number": int(info.get("sweep_number", -1)),
            "amplitude_pA": float(amp),
            "sampling_rate_Hz": fs,
            "duration_s": v.size / fs,
            "above_cap": bool(above),
            "v_max_mV": float(np.max(v)) if finite else float("nan"),
            "max_raw_dvdt_mV_per_ms": float(np.nanmax(raw)),
            "max_filtered_dvdt_mV_per_ms": float(np.nanmax(dvdt)),
            "filtered": bool(filt),
            "sd_raw_dvdt_pre_mV_per_ms": sd_pre,
            "legacy_flag": legacy,
            "reason": why or "subthreshold",
            "n_events": int(n_ev),
            "kept_legacy": (not above) and not legacy,
            "kept_new": (not above) and why == "",
        })
    return rows


def screen_group(group_dir: Union[str, Path], *, mono,
                 cap_pA: Optional[float] = 100.0,
                 specimen_glob: str = "specimen_*",
                 max_cells: Optional[int] = None,
                 verbose: bool = True) -> pd.DataFrame:
    dirs = sorted(p for p in Path(group_dir).glob(specimen_glob) if p.is_dir())
    if max_cells is not None:
        dirs = dirs[:int(max_cells)]
    rows: List[Dict[str, Any]] = []
    for k, d in enumerate(dirs):
        try:
            r = screen_specimen(d, mono=mono, cap_pA=cap_pA)
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            if verbose:
                print("[dep] %3d/%d %-22s ERROR %s: %s"
                      % (k + 1, len(dirs), d.name, type(exc).__name__, exc))
            continue
        rows.extend(r)
        if verbose:
            sub = [x for x in r if not x["above_cap"]]
            print("[dep] %3d/%d %-22s %2d dep sweep(s) <= cap | kept: legacy %d, "
                  "new %d | max raw dV/dt %s mV/ms"
                  % (k + 1, len(dirs), d.name, len(sub),
                     sum(x["kept_legacy"] for x in sub),
                     sum(x["kept_new"] for x in sub),
                     ", ".join("%.1f" % x["max_raw_dvdt_mV_per_ms"] for x in sub)
                     or "-"), flush=True)
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per cell: sweeps under the cap, and how many each screen keeps."""
    if df.empty:
        return pd.DataFrame(columns=["specimen_id", "n_dep_under_cap",
                                     "kept_legacy", "kept_new"])
    sub = df[~df["above_cap"].astype(bool)]
    g = sub.groupby("specimen_id")
    out = pd.DataFrame({
        "n_dep_under_cap": g.size(),
        "kept_legacy": g["kept_legacy"].sum().astype(int),
        "kept_new": g["kept_new"].sum().astype(int),
    }).reset_index()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Old vs new spike screen on the depolarising Long Square "
                    "sweeps of a real archive group (no simulation).")
    ap.add_argument("--group-dir", required=True, help="e.g. <ARCHIVE_ROOT>/L3_exc")
    ap.add_argument("--out", required=True, help="per-sweep CSV to write")
    ap.add_argument("--cap-pA", default="100",
                    help="depolarising amplitude cap in pA, or 'none' "
                         "(default 100, the loader's default)")
    ap.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args(argv)
    cap = None if str(args.cap_pA).lower() == "none" else float(args.cap_pA)
    sys.path.insert(0, args.code_dir)
    import passive_fitting_hpc_fixed as mono
    df = screen_group(args.group_dir, mono=mono, cap_pA=cap,
                      max_cells=args.max_cells)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    s = summarise(df)
    n_cells = int(len(s))
    sub = df[~df["above_cap"].astype(bool)] if not df.empty else df
    n_dirs = len([q for q in Path(args.group_dir).glob("specimen_*") if q.is_dir()])
    if args.max_cells is not None:
        n_dirs = min(n_dirs, int(args.max_cells))
    print("\n[dep] %d cell folder(s) screened, %d with depolarising sweeps <= "
          "cap, %d such sweep(s) -> %s" % (n_dirs, n_cells, len(sub), args.out))
    if len(sub):
        print("[dep] kept by the OLD screen: %d sweep(s) in %d cell(s) | by the "
              "NEW screen: %d sweep(s) in %d cell(s)"
              % (int(sub["kept_legacy"].sum()), int((s["kept_legacy"] > 0).sum()),
                 int(sub["kept_new"].sum()), int((s["kept_new"] > 0).sum())))
        print("[dep] new-screen verdicts: %s" % sub["reason"].value_counts().to_dict())
        print("[dep] median SD of raw dV/dt before the step: %.2f mV/ms; median "
              "max raw |dV/dt| over the sweep: %.1f mV/ms (old threshold %g)"
              % (float(np.nanmedian(sub["sd_raw_dvdt_pre_mV_per_ms"])),
                 float(np.nanmedian(sub["max_raw_dvdt_mV_per_ms"])),
                 mono.LEGACY_SPIKE_DVDT_MV_PER_MS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
