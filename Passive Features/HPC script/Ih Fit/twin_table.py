# -*- coding: utf-8 -*-
"""
twin_table.py -- the twins' table: each twin sweep's law, tilt and protocol (Block 5)
===================================================================================

specs/SPEC.md, section 2.7 (a), (b) and the entry of Block 5 (D-073, D-076
(iii), D-122.3, D-182.3, D-186). Loading layer: the only twin module that
reads the real archive and the noise-fit tables. No NEURON, no plotting.

What it does
    load_cell_protocol      one real cell's stimulation protocol (SS windows
                            and LS sweeps), through Block 3's loaders
    read_noise_tables       the two tables of `run_noise_fit.py merge`
    sweep_laws_from_tables  (P0') of 2.7 (b): each sweep's law and tilt
    build_twin_sweep_table  the twins' table, one row per real sweep
    write_ / read_twin_sweep_table, sweep_law

Units: mV, s, mV/s, pA, Hz (the per-cell table's `_ms` columns are divided by
1000 here, at the loading boundary). Custom code: none (a join and checks).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from noise_law import NoiseLaw
from noise_windows import WindowConfig, load_ls_prewindows, load_ss_windows
from twin_config import TwinConfig

# Columns of the twins' table, in this order (SPEC, Block 5, "Outputs").
TWIN_SWEEP_COLUMNS = (
    "specimen_id", "protocol", "sweep_number", "n_windows", "amplitude_pA", "onset_s", "duration_s", "n_samples",
    "fs_Hz", "sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s", "tilt_mV_per_s", "law_source",
    "noise_flags", "cell_tau_s_above_window")

# Columns of sweep_laws_from_tables, in this order (SPEC, Block 5, "Public API").
SWEEP_LAW_COLUMNS = (
    "specimen_id", "protocol", "sweep_number", "sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s",
    "tilt_mV_per_s", "law_source", "noise_flags", "cell_tau_s_above_window")

_INT_COLUMNS = ("specimen_id", "sweep_number", "n_windows", "n_samples")
_FLOAT_COLUMNS = ("amplitude_pA", "onset_s", "duration_s", "fs_Hz", "sigma_f_mV", "sigma_m_mV", "tau_m_s",
                  "sigma_s_mV", "tau_s_s", "tilt_mV_per_s")
_STR_COLUMNS = ("protocol", "law_source", "noise_flags")
_BOOL_COLUMNS = ("cell_tau_s_above_window",)
_LAW_COLUMNS = ("sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s")
_PROTOCOLS = ("SS", "LS")
_FAILED_PREFIX = "sweep_fit_failed:"


class TwinTableError(RuntimeError):
    """A cell or a sweep the twins' table refuses (SPEC, Block 5, "Refused")."""


@dataclass(frozen=True)
class CellProtocol:
    """The stimulation protocol of one real cell, read from its archive.

    SS, per stored window in the archive's order, shape (K,):
        ss_sweep_number (int64), ss_position (int64, 0-based place in the
        sweep's train), ss_polarity_is_dep (bool), ss_peak_pA (float64,
        signed pA), ss_stim_duration_s (float64, s);
        ss_t_s float64 (n_w,), s, 0 at the pulse onset; ss_fs_Hz (Hz);
        ss_pre_n (samples before the onset); ss_period_samples (onset to
        onset); ss_train_length (pulses per sweep).
    LS, per stored sweep in the order of metadata.json['ls_sweeps'], (K_LS,):
        ls_index (int64, the <index> of v_<index>), ls_sweep_number (int64),
        ls_amplitude_pA (float64, signed pA), ls_onset_s (float64, s),
        ls_duration_s (float64, s: step end - onset), ls_n_samples (int64);
        ls_fs_Hz (Hz).
    """
    specimen_id: int
    ss_fs_Hz: float
    ss_t_s: np.ndarray
    ss_pre_n: int
    ss_period_samples: int
    ss_train_length: int
    ss_sweep_number: np.ndarray
    ss_position: np.ndarray
    ss_polarity_is_dep: np.ndarray
    ss_peak_pA: np.ndarray
    ss_stim_duration_s: np.ndarray
    ls_fs_Hz: float
    ls_index: np.ndarray
    ls_sweep_number: np.ndarray
    ls_amplitude_pA: np.ndarray
    ls_onset_s: np.ndarray
    ls_duration_s: np.ndarray
    ls_n_samples: np.ndarray


# ===========================================================================
#  The archive: one real cell's protocol
# ===========================================================================
def load_cell_protocol(spec_dir, cfg: WindowConfig = WindowConfig()) -> CellProtocol:
    """The protocol of the real cell in `spec_dir` (an archive cell directory,
    SPEC section 3), through noise_windows.load_ss_windows and
    load_ls_prewindows, plus `peak_pA` and `stim_duration_s` of ss_pulses.npz
    and `index` of metadata.json['ls_sweeps']. Raises TwinTableError when a
    sweep of metadata.json['ls_sweeps'] is not returned by load_ls_prewindows.
    The loaders' own LayoutError passes through unchanged."""
    spec_dir = str(spec_dir)
    ss = load_ss_windows(spec_dir, cfg)
    ls = load_ls_prewindows(spec_dir, cfg)
    specimen = int(ss.specimen_id)
    K = ss.n_windows
    with np.load(os.path.join(spec_dir, "ss_pulses.npz")) as z:
        peak = np.asarray(z["peak_pA"], dtype=np.float64).copy()
        dur = np.asarray(z["stim_duration_s"], dtype=np.float64).copy()
    if peak.shape != (K,) or dur.shape != (K,):
        raise TwinTableError("cell %d: ss_pulses.npz peak_pA %s / stim_duration_s %s, expected (%d,)"
                             % (specimen, peak.shape, dur.shape, K))
    with open(os.path.join(spec_dir, "metadata.json")) as fh:
        infos = json.load(fh).get("ls_sweeps") or []
    meta_sweeps = [int(info["sweep_number"]) for info in infos]
    got = [int(s) for s in ls.sweep_number]
    if got != meta_sweeps:
        # the loader returns the sweeps it keeps in the order of metadata.json['ls_sweeps']
        missing = [s for s in meta_sweeps if s not in got]
        raise TwinTableError("cell %d: LS sweep(s) %s of metadata.json['ls_sweeps'] not returned by "
                             "load_ls_prewindows (the twin must have every stored LS sweep, D-186.1)"
                             % (specimen, missing if missing else meta_sweeps))
    ls_index = np.asarray([int(info["index"]) for info in infos], dtype=np.int64)
    return CellProtocol(
        specimen_id=specimen,
        ss_fs_Hz=float(ss.fs),
        ss_t_s=np.asarray(ss.t, dtype=np.float64).copy(),
        ss_pre_n=int(ss.pre_n),
        ss_period_samples=int(ss.period_samples),
        ss_train_length=int(ss.train_length),
        ss_sweep_number=np.asarray(ss.sweep_number, dtype=np.int64).copy(),
        ss_position=np.asarray(ss.position, dtype=np.int64).copy(),
        ss_polarity_is_dep=np.asarray(ss.polarity_is_dep, dtype=bool).copy(),
        ss_peak_pA=peak,
        ss_stim_duration_s=dur,
        ls_fs_Hz=float(ls.fs),
        ls_index=ls_index,
        ls_sweep_number=np.asarray(ls.sweep_number, dtype=np.int64).copy(),
        ls_amplitude_pA=np.asarray(ls.amplitude_pA, dtype=np.float64).copy(),
        ls_onset_s=np.asarray(ls.onset_s, dtype=np.float64).copy(),
        ls_duration_s=(np.asarray(ls.step_end_s, dtype=np.float64) - np.asarray(ls.onset_s, dtype=np.float64)),
        ls_n_samples=np.asarray(ls.n_samples, dtype=np.int64).copy(),
    )


# ===========================================================================
#  The noise-fit run's tables and (P0')
# ===========================================================================
def read_noise_tables(cells_csv, sweeps_csv) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(cells, sweeps): the per-cell and the per-sweep table written by
    `run_noise_fit.py merge` (SPEC section 3), as DataFrames; an empty CSV
    cell is a missing value (and only an empty cell: no other string is read
    as missing)."""
    # float_precision="round_trip": the C parser's default float conversion is not exact (it can differ from
    # the written value in the last digit); round_trip parses as Python's float() does.
    kw = dict(keep_default_na=False, na_values=[""], float_precision="round_trip")
    cells = pd.read_csv(cells_csv, dtype={"specimen_id": np.int64}, **kw)
    sweeps = pd.read_csv(sweeps_csv, dtype={"specimen_id": np.int64, "noise_sweep_number": np.int64}, **kw)
    return cells, sweeps


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _flags_str(v) -> str:
    return "" if _missing(v) else str(v)


def _is_failed(flags: str) -> bool:
    return any(item.strip().startswith(_FAILED_PREFIX) for item in flags.split(";"))


def _cell_lookup(cells: pd.DataFrame) -> Dict[int, dict]:
    ids = [int(x) for x in cells["specimen_id"]]
    seen, dup = set(), set()
    for i in ids:
        (dup if i in seen else seen).add(i)
    if dup:
        raise TwinTableError("cell(s) %s: more than one row in the per-cell table" % sorted(dup))
    return {int(r["specimen_id"]): r for r in cells.to_dict("records")}


def _cell_law(specimen: int, crow: dict) -> NoiseLaw:
    try:
        return NoiseLaw(float(crow["noise_sigma_f_mV"]), float(crow["noise_sigma_m_mV"]),
                        float(crow["noise_tau_m_ms"]) / 1000.0, float(crow["noise_sigma_s_mV"]),
                        float(crow["noise_tau_s_ms"]) / 1000.0)
    except (ValueError, TypeError) as exc:
        raise TwinTableError("cell %d: the per-cell table's law is refused: %s" % (specimen, exc)) from None


def _check_cell_ok(specimen: int, lookup: Dict[int, dict]) -> dict:
    if specimen not in lookup:
        raise TwinTableError("cell %d: absent from the per-cell table" % specimen)
    crow = lookup[specimen]
    status = crow.get("status")
    if _missing(status) or str(status) != "ok":
        raise TwinTableError("cell %d: status %r in the per-cell table, not 'ok' (reason %r)"
                             % (specimen, status, crow.get("reason")))
    return crow


def sweep_laws_from_tables(cells: pd.DataFrame, sweeps: pd.DataFrame,
                           cfg: TwinConfig = TwinConfig()) -> pd.DataFrame:
    """(P0') of SPEC 2.7 (b): one row per row of `sweeps`, in its order, with
    columns specimen_id, protocol ('SS' | 'LS'), sweep_number, sigma_f_mV,
    sigma_m_mV, tau_m_s, sigma_s_mV, tau_s_s, tilt_mV_per_s, law_source
    ('sweep' | 'cell'), noise_flags (str, '' when none),
    cell_tau_s_above_window (bool: tau_s_s > cfg.slow_flag_tau_s). mV, s,
    mV/s. The three SDs are the per-sweep table's columns as they are (for a
    failed sweep they already hold theta_hat(c): checked, never replaced);
    the two time scales the per-cell table's `_ms` columns / 1000.
    Refusals (TwinTableError): SPEC, Block 5."""
    lookup = _cell_lookup(cells)
    laws: Dict[int, NoiseLaw] = {}
    out: List[dict] = []
    for r in sweeps.to_dict("records"):
        specimen = int(r["specimen_id"])
        pr = r["noise_sweep_protocol"]
        sweep = int(r["noise_sweep_number"])
        if _missing(pr) or str(pr) not in _PROTOCOLS:
            raise TwinTableError("cell %d sweep %d: protocol %r, expected 'SS' or 'LS'" % (specimen, sweep, pr))
        pr = str(pr)
        where = "cell %d %s sweep %d" % (specimen, pr, sweep)
        crow = _check_cell_ok(specimen, lookup)
        if specimen not in laws:
            laws[specimen] = _cell_law(specimen, crow)
        cell_law = laws[specimen]
        flags = _flags_str(r.get("noise_sweep_flags"))
        failed = _is_failed(flags)
        sds = tuple(float(r[c]) for c in ("noise_sweep_sigma_f_mV", "noise_sweep_sigma_m_mV",
                                          "noise_sweep_sigma_s_mV"))
        try:
            law = NoiseLaw(sds[0], sds[1], cell_law.tau_m, sds[2], cell_law.tau_s)
        except ValueError as exc:
            raise TwinTableError("%s: the law is refused: %s" % (where, exc)) from None
        tilt = float(r["noise_tilt_mV_per_s"])
        if not np.isfinite(tilt):
            raise TwinTableError("%s: tilt %r is not finite" % (where, tilt))
        if failed:
            ref = (cell_law.sigma_f, cell_law.sigma_m, cell_law.sigma_s)
            for name, a, b in zip(("sigma_f", "sigma_m", "sigma_s"), sds, ref):
                if not abs(a - b) <= cfg.failed_law_rtol * abs(b):
                    raise TwinTableError("%s: failed sweep whose %s %r differs from the cell's %r by more than "
                                         "rtol %g (the row must hold theta_hat(c), D-122.3)"
                                         % (where, name, a, b, cfg.failed_law_rtol))
        out.append({
            "specimen_id": specimen, "protocol": pr, "sweep_number": sweep,
            "sigma_f_mV": law.sigma_f, "sigma_m_mV": law.sigma_m, "tau_m_s": law.tau_m,
            "sigma_s_mV": law.sigma_s, "tau_s_s": law.tau_s, "tilt_mV_per_s": tilt,
            "law_source": "cell" if failed else "sweep", "noise_flags": flags,
            "cell_tau_s_above_window": bool(law.tau_s > cfg.slow_flag_tau_s),
        })
    df = pd.DataFrame(out, columns=list(SWEEP_LAW_COLUMNS))
    return _typed(df, SWEEP_LAW_COLUMNS)


def _typed(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The dtypes of SPEC, Block 5, "Outputs", on the columns present."""
    df = df.copy()
    for c in columns:
        if c in _INT_COLUMNS:
            df[c] = df[c].astype(np.int64)
        elif c in _FLOAT_COLUMNS:
            df[c] = df[c].astype(np.float64)
        elif c in _STR_COLUMNS:
            df[c] = df[c].astype(str)
        elif c in _BOOL_COLUMNS:
            df[c] = df[c].astype(bool)
    return df


# ===========================================================================
#  The twins' table
# ===========================================================================
def build_twin_sweep_table(protocols: Sequence[CellProtocol], cells: pd.DataFrame, sweeps: pd.DataFrame,
                           cfg: TwinConfig = TwinConfig()) -> pd.DataFrame:
    """The twins' table for the cells in `protocols`: one row per real sweep
    (SS and LS), columns TWIN_SWEEP_COLUMNS, dtypes and sorting as SPEC,
    Block 5, "Outputs" states. Refusals: SPEC, Block 5."""
    ids = [int(p.specimen_id) for p in protocols]
    if len(set(ids)) != len(ids):
        raise TwinTableError("cell(s) %s: more than one protocol given"
                             % sorted({i for i in ids if ids.count(i) > 1}))
    lookup = _cell_lookup(cells)
    for specimen in ids:
        _check_cell_ok(specimen, lookup)
    mine = sweeps[sweeps["specimen_id"].astype(np.int64).isin(ids)]
    laws = sweep_laws_from_tables(cells, mine, cfg)
    by_key: Dict[Tuple[int, str, int], List[dict]] = {}
    for r in laws.to_dict("records"):
        by_key.setdefault((int(r["specimen_id"]), r["protocol"], int(r["sweep_number"])), []).append(r)

    rows: List[dict] = []
    used = set()
    for p in protocols:
        specimen = int(p.specimen_id)
        ss_sweeps, ss_counts = np.unique(np.asarray(p.ss_sweep_number, dtype=np.int64), return_counts=True)
        ls_sweeps = [int(s) for s in p.ls_sweep_number]
        dup = sorted({s for s in ls_sweeps if ls_sweeps.count(s) > 1})
        if dup:
            raise TwinTableError("cell %d: LS sweep number(s) %s stored more than once" % (specimen, dup))
        stored = [("SS", int(s)) for s in ss_sweeps] + [("LS", s) for s in ls_sweeps]
        for pr, s in stored:
            hits = by_key.get((specimen, pr, s), [])
            if len(hits) != 1:
                raise TwinTableError("cell %d %s sweep %d: %d rows in the per-sweep table, expected exactly 1"
                                     % (specimen, pr, s, len(hits)))
            used.add((specimen, pr, s))
        for key in by_key:
            if key[0] == specimen and key not in used:
                raise TwinTableError("cell %d %s sweep %d: a row of the per-sweep table without a stored sweep"
                                     % key)
        n_w = int(np.asarray(p.ss_t_s).size)
        for s, k in zip(ss_sweeps, ss_counts):
            law = by_key[(specimen, "SS", int(s))][0]
            rows.append(_row(law, n_windows=int(k), amplitude=np.nan, onset=np.nan, duration=np.nan,
                             n_samples=n_w, fs=float(p.ss_fs_Hz)))
        for i, s in enumerate(ls_sweeps):
            law = by_key[(specimen, "LS", s)][0]
            rows.append(_row(law, n_windows=1, amplitude=float(p.ls_amplitude_pA[i]),
                             onset=float(p.ls_onset_s[i]), duration=float(p.ls_duration_s[i]),
                             n_samples=int(p.ls_n_samples[i]), fs=float(p.ls_fs_Hz)))
    rows.sort(key=lambda r: (r["specimen_id"], _PROTOCOLS.index(r["protocol"]), r["sweep_number"]))
    df = pd.DataFrame(rows, columns=list(TWIN_SWEEP_COLUMNS))
    return _typed(df, TWIN_SWEEP_COLUMNS).reset_index(drop=True)


def _row(law: dict, *, n_windows: int, amplitude: float, onset: float, duration: float, n_samples: int,
         fs: float) -> dict:
    row = {c: law[c] for c in SWEEP_LAW_COLUMNS}
    row.update({"n_windows": n_windows, "amplitude_pA": amplitude, "onset_s": onset, "duration_s": duration,
                "n_samples": n_samples, "fs_Hz": fs})
    return row


def write_twin_sweep_table(table: pd.DataFrame, path) -> None:
    """Write the table as CSV (ASCII, LF), columns TWIN_SWEEP_COLUMNS in that
    order, floats with 17 significant digits ('%.17g'), NaN as an empty
    field; the file is replaced atomically."""
    if tuple(table.columns) != TWIN_SWEEP_COLUMNS:
        raise TwinTableError("the table's columns %s differ from TWIN_SWEEP_COLUMNS" % list(table.columns))
    path = Path(path)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    table.to_csv(tmp, index=False, float_format="%.17g", na_rep="", encoding="ascii", lineterminator="\n")
    os.replace(tmp, path)


def read_twin_sweep_table(path) -> pd.DataFrame:
    """Read a table written by write_twin_sweep_table: the same values,
    dtypes and column order."""
    dtype = {c: np.int64 for c in _INT_COLUMNS}
    dtype.update({c: np.float64 for c in _FLOAT_COLUMNS})
    dtype.update({c: str for c in _STR_COLUMNS + _BOOL_COLUMNS})
    df = pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values={c: [""] for c in _FLOAT_COLUMNS},
                     float_precision="round_trip")
    if tuple(df.columns) != TWIN_SWEEP_COLUMNS:
        raise TwinTableError("%s: columns %s differ from TWIN_SWEEP_COLUMNS" % (path, list(df.columns)))
    for c in _BOOL_COLUMNS:
        vals = set(df[c].tolist())
        if not vals <= {"True", "False"}:
            raise TwinTableError("%s: column %s holds %s, expected True / False" % (path, c, sorted(vals)))
        df[c] = df[c].map({"True": True, "False": False})
    return _typed(df, TWIN_SWEEP_COLUMNS)


def sweep_law(table: pd.DataFrame, specimen_id: int, protocol: str, sweep_number: int) -> Tuple[NoiseLaw, float]:
    """(law, tilt_mV_per_s) of one row: NoiseLaw(sigma_f, sigma_m, tau_m,
    sigma_s, tau_s) in mV, mV, s, mV, s and the tilt in mV/s. KeyError when
    the row is absent, TwinTableError when it is not unique."""
    m = ((table["specimen_id"] == int(specimen_id)) & (table["protocol"] == str(protocol))
         & (table["sweep_number"] == int(sweep_number)))
    hit = table[m]
    if len(hit) == 0:
        raise KeyError("no row for cell %d %s sweep %d" % (int(specimen_id), protocol, int(sweep_number)))
    if len(hit) > 1:
        raise TwinTableError("cell %d %s sweep %d: %d rows, expected 1"
                             % (int(specimen_id), protocol, int(sweep_number), len(hit)))
    r = hit.iloc[0]
    law = NoiseLaw(*(float(r[c]) for c in _LAW_COLUMNS))
    return law, float(r["tilt_mV_per_s"])
