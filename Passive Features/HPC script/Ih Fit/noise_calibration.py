# -*- coding: utf-8 -*-
"""
noise_calibration.py
====================

Measure the recording noise of REAL cells in the form the synthetic generator
needs it, so a Stage-7 cohort is as noisy as the bench -- per sweep, per cell.

Why this exists: two objects, one name
--------------------------------------
Two different quantities in this pipeline are both called "sigma":

    fitter   PassiveFitResult.noise_sigma_mV
             the SD of the pre-stimulus samples of each AVERAGED training
             bundle, then averaged over the training bundles. An SS bundle
             is a mean over N pulses, so its fast noise is ~ sigma/sqrt(N);
             a Long Square bundle is often a single sweep. The value is
             therefore a protocol-dependent MIXTURE, and it changes when
             the training set changes (run B's 1 SS + 2 LS is not D-006's
             1 SS + ~5 LS).

    generator  synthetic_ground_truth.NoiseConfig.sigma_mV
             the RMS of the fast noise added to EACH SWEEP, before any
             averaging.

Feeding the first into the second injects the wrong object. And the
generator has an AR(1) coefficient, `rho_lag1`, that no manifest has ever
set -- so every synthetic cohort so far had WHITE noise, while run B
(`pipeline_outputs/dt01ls60`) measured rho_lag1 > 0.5 in 61 of its 68 cells.
For AR(1) noise the effective number of independent samples in a window is
reduced by about (1 - rho)/(1 + rho): white synthetic noise gives the fitter
several times more information than a real recording does, which is exactly
the "gate passes on data cleaner than the bench" failure.

What this measures, and how
---------------------------
For each specimen of a real Phase-0 archive group, on SINGLE SWEEPS only:

    * every individual Square-Subthreshold pulse (ss_individual_pulses:
      one pulse, one sweep, never averaged);
    * every Long-Square bundle with n_repeats_averaged == 1 (one sweep).

On each it applies the fitter's OWN estimator, passive_fitting_hpc_fixed.
_estimate_noise, so "sigma" and "rho_lag1" here are, by construction, the
same statistic the fitter computes -- evaluated on the object the generator
parameterises. It also records the sampling rate of each protocol and the
number of SS pulses per polarity, because both change what the fitter sees:
lag-1 correlation depends on the sampling rate, and the SS bundle's noise on
how many pulses are averaged into it.

No simulation happens here: the pre-stimulus window of a sweep has no
stimulus response in it, so reading it needs only the archive's arrays. What
it contains is the TOTAL voltage fluctuation at rest in that sweep --
recording-chain noise plus whatever the membrane itself does (channel
gating, spontaneous synaptic input) -- and no estimator on one window can
tell those apart. For the generator it does not need to: the fitter sees the
total.

Is AR(1) enough? (the adequacy diagnostic)
------------------------------------------
The generator's fast noise is a stationary AR(1): matched to the data at lag
0 (sigma) and lag 1 (rho), it then IMPOSES rho^L at every lag L, i.e. one
exponential correlation time. Real baseline noise need not have one: an
anti-aliasing filter makes the autocorrelation flatter than exponential at
short lags, and slower biological components add a tail an exponential
fitted at lag 1 cannot carry. The table therefore also reports, per
protocol, the measured autocorrelation at fixed PHYSICAL lags
(ACF_LAGS_MS) beside the value AR(1) implies there, rho^L with L the lag in
samples. acf >> ar1 at 1 ms means slow correlated power the generator does
not reproduce -- the direction that makes synthetic data more informative
than real data. Diagnostic only: nothing downstream reads these columns.

Smoke: smoke_ih_recovery.py R10 closes the loop -- it generates a synthetic
archive with KNOWN per-sweep (sigma, rho), measures it back with this module,
and requires the injected values to come out.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


#: The per-specimen columns written by measure_group_noise, in order.
NOISE_TABLE_COLUMNS: List[str] = [
    "specimen_id",
    "sigma_ls_mV", "rho_ls", "n_ls_single",
    "sigma_ss_mV", "rho_ss", "n_ss_pulses",
    "fs_ls_Hz", "fs_ss_Hz",
    "n_ss_hyp", "n_ss_dep",
    "n_ls_multi_skipped",
    "n_pre_ls", "n_pre_ss",
]

#: Physical lags (ms) of the AR(1) adequacy diagnostic. In samples they are
#: round(lag * fs): 5 and 50 at 50 kHz, 20 and 200 at 200 kHz.
ACF_LAGS_MS = (0.1, 1.0)


def _lag_tag(lag_ms: float) -> str:
    return ("%gms" % lag_ms).replace(".", "p")


for _p in ("ls", "ss"):
    for _l in ACF_LAGS_MS:
        NOISE_TABLE_COLUMNS += ["acf_%s_%s" % (_p, _lag_tag(_l)),
                                "ar1_%s_%s" % (_p, _lag_tag(_l))]
NOISE_TABLE_COLUMNS.append("error")

_SPECIMEN_RE = re.compile(r"specimen_(\d+)")


def specimen_id_of(path: Union[str, Path]) -> Optional[int]:
    """The Allen specimen id encoded in a path (`.../specimen_<id>/...`), or
    None. Used to hand each synthetic cell the noise of the real cell whose
    morphology it was drawn on."""
    m = None
    for part in reversed(Path(path).parts):
        m = _SPECIMEN_RE.fullmatch(part) or _SPECIMEN_RE.match(part)
        if m:
            break
    return int(m.group(1)) if m else None


def _pulse_as_bundle(mono, p: Mapping[str, Any]):
    """One individual SS pulse (a dict from the archive loader) as a
    SweepBundle, so the fitter's estimator can read it unchanged. The SS
    convention is t = 0 at pulse onset; the estimator's brief-pulse branch
    then takes t < 0 as the pre-stimulus window."""
    return mono.SweepBundle(
        polarity=str(p.get("polarity", "hyp")),
        amplitude_pA=float(p.get("peak_pA", 0.0)),
        t=np.asarray(p["t"], dtype=float),
        v_mV=np.asarray(p["v"], dtype=float),
        i_pA=np.asarray(p.get("i", np.zeros_like(p["v"])), dtype=float),
        stim_onset_s=0.0,
        stim_duration_s=float(p.get("stim_duration_s", 5e-4)),
        n_repeats_averaged=1,
        sweep_numbers=[],
        sampling_rate_Hz=float(p.get("sampling_rate_Hz", np.nan)),
        stimulus_name="Square Subthreshold (single pulse)",
    )


def _pre_window_mask(mono, bundle) -> np.ndarray:
    """The samples mono._estimate_noise reads for `bundle`, as a boolean
    mask: the same two windows, written out because the estimator does not
    return them. Brief pulse: t < 0. Long step: [t[0], stim_onset - 5 ms)."""
    if mono._is_brief_pulse(bundle):
        return (bundle.t >= float(bundle.t[0])) & (bundle.t < 0.0)
    end_s = max(float(bundle.t[0]), bundle.stim_onset_s - 5e-3)
    return (bundle.t >= float(bundle.t[0])) & (bundle.t < end_s)


def pre_window_samples(mono, bundle) -> np.ndarray:
    """The pre-stimulus samples (mV) of one sweep, as the fitter reads them."""
    return np.asarray(bundle.v_mV[_pre_window_mask(mono, bundle)], dtype=float)


def _pre_window_n(mono, bundle) -> int:
    """Number of samples mono._estimate_noise reads for `bundle`.

    Why it is recorded: the estimator demeans the window before taking the
    SD, and ddof=1 corrects that only for independent samples. On an AR(1)
    window of n samples the mean absorbs a share ~ (1+rho)/((1-rho) n) of the
    variance, so a short, strongly correlated window reads sigma LOW. The
    window length is what decides whether that matters, so it goes in the
    table next to the sigma it qualifies."""
    return int(np.count_nonzero(_pre_window_mask(mono, bundle)))


def lag_autocorr(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation of one window at `lag` samples, in the SAME
    convention as the fitter's lag-1 statistic (mono._estimate_noise):
    demean the window, then the mean of the n - lag available products over
    the mean square of all n samples. lag = 1 therefore reproduces the
    fitter's rho_lag1 exactly (smoke R10 asserts it). NaN when fewer than 10
    products are available or the window is flat.

    Written out in numpy rather than taken from statsmodels.tsa.acf, whose
    normalisation (a sum over n - lag products divided by n times the
    variance) is a different estimator and would not reproduce the fitter's
    rho at lag 1."""
    u = np.asarray(x, dtype=float)
    n = int(u.size)
    lag = int(lag)
    if lag < 1 or n - lag < 10:
        return float("nan")
    u = u - u.mean()
    var = float(np.mean(u * u))
    if not var > 0.0:
        return float("nan")
    return float(np.mean(u[lag:] * u[:-lag]) / var)


def _acf_diagnostic(mono, bundles: Sequence, fs_of) -> Dict[str, float]:
    """For each lag in ACF_LAGS_MS: the median over sweeps of the measured
    autocorrelation at that physical lag, and of the value AR(1) implies
    there from the same sweep's lag-1 coefficient, rho1^L."""
    out: Dict[str, float] = {}
    for lag_ms in ACF_LAGS_MS:
        acf, ar1 = [], []
        for b in bundles:
            fs = float(fs_of(b))
            if not (np.isfinite(fs) and fs > 0):
                continue
            lag = int(round(lag_ms * 1e-3 * fs))
            x = pre_window_samples(mono, b)
            r1 = lag_autocorr(x, 1)
            acf.append(lag_autocorr(x, lag))
            ar1.append(r1 ** lag if np.isfinite(r1) else float("nan"))
        out[_lag_tag(lag_ms)] = (_median(acf), _median(ar1))
    return out


def load_for_noise(specimen_dir: Union[str, Path], *, mono):
    """The one way this module loads a cell: every hyperpolarising Long
    Square amplitude (no current ceiling -- the PRE-step window is what is
    read, and it does not depend on the amplitude), no depolarising steps."""
    return mono.load_cell_from_archive(Path(specimen_dir),
                                       ls_max_amplitude_pA=None,
                                       load_depolarising_ls=False,
                                       verbose=False)


def _median(xs: Sequence[float]) -> float:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def measure_specimen_noise(specimen_dir: Union[str, Path], *, mono,
                           ) -> Dict[str, Any]:
    """Per-sweep noise of ONE real cell. Returns one row of the noise table.

    Every statistic is a MEDIAN over single sweeps of that protocol, so one
    bad sweep does not move it. A cell with no single-sweep Long Square
    reports NaN for the LS columns and the count of multi-sweep bundles it
    skipped, rather than a sigma inflated or deflated by an unknown average.
    """
    d = Path(specimen_dir)
    row: Dict[str, Any] = {c: np.nan for c in NOISE_TABLE_COLUMNS}
    row["specimen_id"] = specimen_id_of(d)
    row["error"] = ""
    try:
        cd = load_for_noise(d, mono=mono)
    except Exception as exc:  # noqa: BLE001 -- recorded, not raised
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        return row
    if row["specimen_id"] is None:
        row["specimen_id"] = int(getattr(cd, "specimen_id", -1))

    ls_all = list(getattr(cd, "long_square_subthreshold", []) or [])
    ls_single = [b for b in ls_all if int(getattr(b, "n_repeats_averaged", 1)) == 1]
    ls_stats = [mono._estimate_noise(b) for b in ls_single]
    row["sigma_ls_mV"] = _median([s[0] for s in ls_stats])
    row["rho_ls"] = _median([s[1] for s in ls_stats])
    for _tag, (_a, _r) in _acf_diagnostic(
            mono, ls_single, lambda b: b.sampling_rate_Hz).items():
        row["acf_ls_" + _tag], row["ar1_ls_" + _tag] = _a, _r
    row["n_ls_single"] = len(ls_single)
    row["n_ls_multi_skipped"] = len(ls_all) - len(ls_single)
    row["fs_ls_Hz"] = _median([float(b.sampling_rate_Hz) for b in ls_single])
    row["n_pre_ls"] = _median([_pre_window_n(mono, b) for b in ls_single])

    pulses = list(getattr(cd, "ss_individual_pulses", []) or [])
    ss_bundles = [_pulse_as_bundle(mono, p) for p in pulses]
    ss_stats = [mono._estimate_noise(b) for b in ss_bundles]
    row["n_pre_ss"] = _median([_pre_window_n(mono, b) for b in ss_bundles])
    for _tag, (_a, _r) in _acf_diagnostic(
            mono, ss_bundles, lambda b: b.sampling_rate_Hz).items():
        row["acf_ss_" + _tag], row["ar1_ss_" + _tag] = _a, _r
    row["sigma_ss_mV"] = _median([s[0] for s in ss_stats])
    row["rho_ss"] = _median([s[1] for s in ss_stats])
    row["n_ss_pulses"] = len(pulses)
    row["fs_ss_Hz"] = _median([float(p.get("sampling_rate_Hz", np.nan))
                               for p in pulses])
    row["n_ss_hyp"] = sum(1 for p in pulses if p.get("polarity") == "hyp")
    row["n_ss_dep"] = sum(1 for p in pulses if p.get("polarity") == "dep")
    return row


def measure_group_noise(group_dir: Union[str, Path], *, mono,
                        specimen_glob: str = "specimen_*",
                        max_cells: Optional[int] = None,
                        verbose: bool = True) -> pd.DataFrame:
    """The noise table of a whole archive group: one row per specimen."""
    dirs = sorted(p for p in Path(group_dir).glob(specimen_glob) if p.is_dir())
    if max_cells is not None:
        dirs = dirs[:int(max_cells)]
    rows = []
    for i, d in enumerate(dirs):
        r = measure_specimen_noise(d, mono=mono)
        rows.append(r)
        if verbose:
            print("[noise] %3d/%d %-22s LS sigma %.4f rho %.3f (%s sweep(s))  "
                  "SS sigma %.4f rho %.3f (%s pulse(s))%s"
                  % (i + 1, len(dirs), d.name, r["sigma_ls_mV"], r["rho_ls"],
                     r["n_ls_single"], r["sigma_ss_mV"], r["rho_ss"],
                     r["n_ss_pulses"],
                     ("  ERROR " + r["error"]) if r["error"] else ""),
                  flush=True)
    return pd.DataFrame(rows, columns=NOISE_TABLE_COLUMNS)


def _has_error(value) -> bool:
    """True only for a non-empty error string. An empty `error` cell reads
    back from CSV as NaN, and NaN is truthy -- testing the raw value would
    silently drop every cell of a reloaded table."""
    return isinstance(value, str) and value.strip() != ""


def summarise_noise_table(df: pd.DataFrame) -> Dict[str, float]:
    """Cohort medians -- the protocol settings a synthetic cohort should be
    generated with so that its bundles look, to the fitter, like real ones.
    Per-cell noise itself is taken from the table row by row; these medians
    are the fallback for a morphology with no usable row, and the protocol
    constants (sampling rates, SS repeat count) that are per cohort."""
    ok = df[[not _has_error(e) for e in df["error"]]]

    def med(c):
        v = pd.to_numeric(ok[c], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    return {
        "n_cells": float(len(ok)),
        "sigma_ls_mV": med("sigma_ls_mV"), "rho_ls": med("rho_ls"),
        "sigma_ss_mV": med("sigma_ss_mV"), "rho_ss": med("rho_ss"),
        "fs_ls_Hz": med("fs_ls_Hz"), "fs_ss_Hz": med("fs_ss_Hz"),
        # per polarity, because the generator's ss_n_repeats is per polarity
        "n_ss_per_polarity": med("n_ss_hyp"),
        "frac_rho_ls_gt_0p5": (float(np.mean(pd.to_numeric(
            ok["rho_ls"], errors="coerce").dropna() > 0.5))
            if ok["rho_ls"].notna().any() else float("nan")),
        # the AR(1) adequacy diagnostic, cohort medians
        **{c: med(c) for c in NOISE_TABLE_COLUMNS
           if c.startswith(("acf_", "ar1_"))},
    }


#: What `noise_lookup` can hand the generator.
NOISE_PROTOCOLS = ("per_protocol", "ls", "ss")


def noise_lookup(df: pd.DataFrame, *, protocol: str = "per_protocol"
                 ) -> Dict[int, Dict[str, float]]:
    """specimen_id -> what the synthetic twin of that cell is generated with.

    Keys: "sigma_mV", "rho_lag1" (the noise of the Long Square sweeps, and of
    the SS pulses unless SS keys follow); in "per_protocol" mode also
    "ss_sigma_mV", "ss_rho_lag1" when the cell's SS pulses were measurable;
    and in EVERY mode the acquisition twin -- "fs_ss_Hz", "fs_ls_Hz",
    "ss_n_repeats" -- because rho is a correlation between successive
    samples and transfers only together with the sampling interval it was
    measured at.

    protocol:
      "per_protocol" (default, D-014) -- LS sweeps carry the LS measurement,
          SS pulses the SS measurement: each protocol's pre-window sees the
          noise at roughly the bandwidth its own residual is judged on (the
          SS residual is exponentially weighted over its first ~10 ms, the
          LS residual spans the whole step).
      "ls" / "ss" -- one measurement drives both protocols (D-012's form).
    A cell is skipped -- and generated at the nominal level, labelled so --
    when the measurement that drives its LS sweeps is not finite.
    """
    if protocol not in NOISE_PROTOCOLS:
        raise ValueError("protocol must be one of %s, got %r"
                         % (NOISE_PROTOCOLS, protocol))
    main = "ss" if protocol == "ss" else "ls"

    def _f(r, key):
        try:
            v = float(r.get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    out: Dict[int, Dict[str, float]] = {}
    for _, r in df.iterrows():
        if _has_error(r.get("error")):
            continue
        s, rho = _f(r, "sigma_%s_mV" % main), _f(r, "rho_%s" % main)
        if not (np.isfinite(s) and np.isfinite(rho)):
            continue
        e: Dict[str, float] = {"sigma_mV": s, "rho_lag1": rho}
        if protocol == "per_protocol":
            ss_s, ss_r = _f(r, "sigma_ss_mV"), _f(r, "rho_ss")
            if np.isfinite(ss_s) and np.isfinite(ss_r):
                e["ss_sigma_mV"], e["ss_rho_lag1"] = ss_s, ss_r
        for src, key in (("fs_ss_Hz", "fs_ss_Hz"), ("fs_ls_Hz", "fs_ls_Hz"),
                         ("n_ss_hyp", "ss_n_repeats")):
            v = _f(r, src)
            if np.isfinite(v) and v > 0:
                e[key] = v
        out[int(r["specimen_id"])] = e
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure per-sweep recording noise of a real archive "
                    "group (no simulation; seconds per cell).")
    ap.add_argument("--group-dir", required=True,
                    help="e.g. <ARCHIVE_ROOT>/L3_exc")
    ap.add_argument("--out", required=True, help="noise table CSV to write")
    ap.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args(argv)
    sys.path.insert(0, args.code_dir)
    import passive_fitting_hpc_fixed as mono
    df = measure_group_noise(args.group_dir, mono=mono, max_cells=args.max_cells)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    s = summarise_noise_table(df)
    print("\n[noise] %d cell(s) -> %s" % (int(s["n_cells"]), args.out))
    print("[noise] per-sweep medians: LS sigma %.4f mV rho %.3f @ %.0f Hz | "
          "SS sigma %.4f mV rho %.3f @ %.0f Hz | SS pulses/polarity %.0f"
          % (s["sigma_ls_mV"], s["rho_ls"], s["fs_ls_Hz"], s["sigma_ss_mV"],
             s["rho_ss"], s["fs_ss_Hz"], s["n_ss_per_polarity"]))
    print("[noise] fraction of cells with LS rho_lag1 > 0.5: %.2f"
          % s["frac_rho_ls_gt_0p5"])
    for p in ("ls", "ss"):
        print("[noise] AR(1) adequacy, %s: " % p.upper() + " | ".join(
            "lag %s: measured %.3f vs AR(1) %.3f"
            % (_lag_tag(l), s["acf_%s_%s" % (p, _lag_tag(l))],
               s["ar1_%s_%s" % (p, _lag_tag(l))]) for l in ACF_LAGS_MS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
