"""noise_fit_gates.py -- read-only helpers of cluster_gates_noise_fit.sh.

Three subcommands; each prints its details, then ONE summary line, and exits
0 when the check holds, 1 on a finding, 2 on a usage error:

  python noise_fit_gates.py bytes [--folder DIR]
      every top-level *.py of DIR (default: this folder) is pure ASCII, and
      *.sh, *.pbs and mod/*.mod carry no CR byte (hpc-python-compat)
      -> "BYTES: ok (...)" or "BYTES: FAIL (...)"
  python noise_fit_gates.py compile [--folder DIR]
      every top-level *.py compiles; compiled in memory, no .pyc written
      -> "COMPILE: ok (...)" or "COMPILE: FAIL (...)"
  python noise_fit_gates.py ss-pulses SPECIMEN_DIR [SPECIMEN_DIR ...]
      what a Phase-0 ss_pulses.npz holds -- the input Block 3 of
      specs/SPEC.md reads: its keys, dtypes and shapes; per sweep number, the
      stored windows' count, polarity sequence and which consecutive windows
      are ADJACENT in the recording; and the metadata.json keys about SS.
      -> one line per cell, "SS_PULSES <id>: format=A sweep_number=yes ..."

Adjacency. The archive writer cuts every window from one sweep's trace,
(-pre, +post) ms around each pulse onset, and stores a sweep's windows in
onset order (`_extract_windows_around_pulses`, `_detect_pulses_in_current`,
`save_alleninstitute_data (6).py`). Two windows k, k+1 of one sweep whose
onsets lie D samples apart, D < n, therefore share n - D samples exactly:
v_k[D:] == v_{k+1}[:n - D], and likewise the current. The smallest D >= 1 for
which both the voltage and the current agree sample for sample over at least
MIN_OVERLAP samples is taken as the onset spacing (D / f_s); a pair with no
such D is a gap (a pulse dropped by the writer's QC, or onsets more than one
window apart). MIN_OVERLAP guards against chance agreement of a few samples
of an ADC-quantised trace.

Pure ASCII. Needs numpy only.
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Shortest overlap accepted as adjacency (samples): at 50 kHz 2 ms; the
# writer's (-10, +200) ms windows 200 ms apart overlap by 500 samples there.
MIN_OVERLAP = 100


# ===========================================================================
#  bytes and compile
# ===========================================================================
def scan_bytes(folder: str):
    """(files with a CR byte among *.sh, *.pbs, mod/*.mod; *.py with a byte
    above 0x7F; number of *.py scanned), all top-level in `folder`."""
    shebang = sorted(glob.glob(os.path.join(folder, "*.sh")) + glob.glob(os.path.join(folder, "*.pbs"))
                     + glob.glob(os.path.join(folder, "mod", "*.mod")))
    pys = sorted(glob.glob(os.path.join(folder, "*.py")))
    crlf = [os.path.relpath(p, folder) for p in shebang if b"\r" in open(p, "rb").read()]
    nonascii = [os.path.relpath(p, folder) for p in pys if any(c > 127 for c in open(p, "rb").read())]
    return crlf, nonascii, len(pys), len(shebang)


def compile_all(folder: str):
    """(number of top-level *.py compiled, list of 'file: error')."""
    errors = []
    pys = sorted(glob.glob(os.path.join(folder, "*.py")))
    for p in pys:
        try:
            compile(open(p, "rb").read(), p, "exec", dont_inherit=True)
        except SyntaxError as exc:
            errors.append("%s: line %s: %s" % (os.path.basename(p), exc.lineno, exc.msg))
    return len(pys), errors


# ===========================================================================
#  ss_pulses.npz
# ===========================================================================
def load_ss_pulses(path: str) -> Dict:
    """Read ss_pulses.npz in either archive format into per-pulse lists.

    Returns dict: format 'A' or 'B'; keys [(name, dtype, shape)]; v, i (lists
    of float arrays, one per pulse); t (list of time axes, s); polarity_is_dep
    (bool array); sampling_rate_Hz (float array); sweep_number (int array, or
    None when the file has no such key)."""
    z = np.load(path)
    keys = [(k, str(z[k].dtype), tuple(z[k].shape)) for k in z.files]
    variable = "variable_length" in z.files and bool(z["variable_length"][0])
    if not variable:
        v2 = np.asarray(z["v"], dtype=float)
        i2 = np.asarray(z["i_pA"], dtype=float)
        t = np.asarray(z["t"], dtype=float)
        v = [v2[k] for k in range(v2.shape[0])]
        i = [i2[k] for k in range(i2.shape[0])]
        ts = [t] * len(v)
        fmt = "A"
    else:
        n = int(z["n_pulses"][0])
        v = [np.asarray(z["v_%d" % k], dtype=float) for k in range(n)]
        i = [np.asarray(z["i_%d" % k], dtype=float) for k in range(n)]
        ts = [np.asarray(z["t_%d" % k], dtype=float) for k in range(n)]
        fmt = "B"
    sweep = np.asarray(z["sweep_number"], dtype=np.int64) if "sweep_number" in z.files else None
    return dict(format=fmt, keys=keys, v=v, i=i, t=ts,
                polarity_is_dep=np.asarray(z["polarity_is_dep"], dtype=bool),
                sampling_rate_Hz=np.asarray(z["sampling_rate_Hz"], dtype=float),
                sweep_number=sweep)


def onset_spacing(v_a: np.ndarray, i_a: np.ndarray, v_b: np.ndarray, i_b: np.ndarray,
                  min_overlap: int = MIN_OVERLAP) -> Optional[int]:
    """Smallest D >= 1 (samples) with v_a[D:D+L] == v_b[:L] and i_a[D:D+L] ==
    i_b[:L], L = min(len(v_a) - D, len(v_b)) >= min_overlap; None if none."""
    n_a, n_b = len(v_a), len(v_b)
    if n_a == 0 or n_b == 0 or min_overlap < 1:
        return None
    cand = np.flatnonzero(v_a == v_b[0])
    cand = cand[(cand >= 1) & (np.minimum(n_a - cand, n_b) >= min_overlap)]
    # vectorised pre-filter on the first samples of the overlap (a quantised
    # trace repeats single values often; a run of 16 by chance essentially never)
    for k in range(1, min(16, min_overlap)):
        if cand.size == 0:
            return None
        cand = cand[v_a[cand + k] == v_b[k]]
    for d in cand:                                   # ascending: the smallest D first
        d = int(d)
        L = min(n_a - d, n_b)
        if np.array_equal(v_a[d:d + L], v_b[:L]) and np.array_equal(i_a[d:d + L], i_b[:L]):
            return d
    return None


def sweep_structure(data: Dict, min_overlap: int = MIN_OVERLAP) -> List[Dict]:
    """Group the stored windows by sweep number (stored order kept), or, when
    the file has no sweep numbers, into chains of consecutive adjacent
    windows. Per group: label, pulse indices, polarity string ('+' dep, '-'
    hyp), the spacing D (samples) of each consecutive pair or None, f_s."""
    n = len(data["v"])
    sweep = data["sweep_number"]
    pol = data["polarity_is_dep"]
    fs = data["sampling_rate_Hz"]

    def spacing(a, b):
        if fs[a] != fs[b] or len(data["v"][a]) != len(data["v"][b]):
            return None
        return onset_spacing(data["v"][a], data["i"][a], data["v"][b], data["i"][b], min_overlap)

    groups: List[Dict] = []
    if sweep is not None:
        order = []
        for k in range(n):
            if not order or order[-1][0] != sweep[k]:
                order.append((int(sweep[k]), [k]))
            else:
                order[-1][1].append(k)
        for label, idx in order:
            groups.append(dict(label=str(label), idx=idx,
                               spacing=[spacing(a, b) for a, b in zip(idx[:-1], idx[1:])]))
    else:
        idx = [0] if n else []
        sp: List[Optional[int]] = []
        for k in range(1, n):
            d = spacing(k - 1, k)
            if d is None:
                groups.append(dict(label="chain%d" % (len(groups) + 1), idx=idx, spacing=sp))
                idx, sp = [k], []
            else:
                idx.append(k)
                sp.append(d)
        if idx:
            groups.append(dict(label="chain%d" % (len(groups) + 1), idx=idx, spacing=sp))
    for g in groups:
        g["polarity"] = "".join("+" if pol[k] else "-" for k in g["idx"])
        g["fs"] = float(fs[g["idx"][0]])
    return groups


def _metadata_lines(path: str) -> List[str]:
    out = []
    try:
        m = json.load(open(path))
    except Exception as exc:                                # report, do not stop the listing
        return ["  metadata.json: unreadable (%r)" % (exc,)]
    out.append("  metadata.json top-level keys: %s" % sorted(m.keys()))
    for k in sorted(m.keys()):
        if not any(s in k.lower() for s in ("ss", "sweep", "pulse", "square")):
            continue
        val = m[k]
        if isinstance(val, list):
            head = val[:3]
            if head and isinstance(head[0], dict):
                prev = "list of %d dicts, keys %s" % (len(val), sorted(head[0].keys()))
            else:
                prev = "list of %d: %s%s" % (len(val), head, " ..." if len(val) > 3 else "")
        elif isinstance(val, dict):
            prev = "dict, keys %s" % sorted(val.keys())[:12]
        else:
            prev = repr(val)[:120]
        out.append("  metadata.json[%r]: %s" % (k, prev))
    return out


def describe_specimen(spec_dir: str, min_overlap: int = MIN_OVERLAP):
    """(detail lines, summary line, ok) for one Phase-0 specimen directory;
    ok is True when the file was read and carries sweep numbers."""
    sid = os.path.basename(os.path.normpath(spec_dir)).replace("specimen_", "")
    path = os.path.join(spec_dir, "ss_pulses.npz")
    if not os.path.isfile(path):
        return [], "SS_PULSES %s: MISSING %s" % (sid, path), False
    try:
        data = load_ss_pulses(path)
    except Exception as exc:
        return [], "SS_PULSES %s: UNREADABLE %s (%r)" % (sid, path, exc), False
    lines = ["== %s" % path]
    for name, dtype, shape in data["keys"]:
        lines.append("  key %-18s %-8s %s" % (name, dtype, shape))
    t0 = data["t"][0]
    fs_all = sorted(set(float(f) for f in data["sampling_rate_Hz"]))
    dt = 1.0 / fs_all[0] if fs_all else float("nan")
    win = "%.3f..%.3f" % (t0[0] * 1e3, (t0[-1] + dt) * 1e3) if len(t0) else "empty"
    lines.append("  pulses %d, window %s ms (%d samples), f_s %s Hz"
                 % (len(data["v"]), win, len(t0), ",".join("%g" % f for f in fs_all)))
    groups = sweep_structure(data, min_overlap)
    n_pairs = n_adj = n_opp = 0
    spacings = set()
    for g in groups:
        sp = g["spacing"]
        adj = [d for d in sp if d is not None]
        n_pairs += len(sp)
        n_adj += len(adj)
        n_opp += sum(1 for j, d in enumerate(sp)
                     if d is not None and g["polarity"][j] != g["polarity"][j + 1])
        ms = sorted(set(round(d / g["fs"] * 1e3, 3) for d in adj))
        spacings.update(ms)
        gaps = [j + 1 for j, d in enumerate(sp) if d is None]
        lines.append("  %s %-10s windows %3d  polarity %s  adjacent %d/%d  spacing_ms %s%s"
                     % ("sweep" if data["sweep_number"] is not None else "group", g["label"],
                        len(g["idx"]), g["polarity"], len(adj), len(sp),
                        ",".join("%.3f" % s for s in ms) or "-",
                        ("  gap before window(s) %s" % gaps) if gaps else ""))
    meta = os.path.join(spec_dir, "metadata.json")
    if os.path.isfile(meta):
        lines.extend(_metadata_lines(meta))
    has_sweep = data["sweep_number"] is not None
    summary = ("SS_PULSES %s: format=%s sweep_number=%s n_pulses=%d n_%s=%d per_%s=%s adjacent=%d/%d "
               "opposite_polarity_adjacent=%d/%d spacing_ms=%s fs_Hz=%s window_ms=%s"
               % (sid, data["format"], "yes" if has_sweep else "no", len(data["v"]),
                  "sweeps" if has_sweep else "chains", len(groups), "sweep" if has_sweep else "chain",
                  ",".join(str(len(g["idx"])) for g in groups), n_adj, n_pairs, n_opp, n_adj,
                  ",".join("%.3f" % s for s in sorted(spacings)) or "-",
                  ",".join("%g" % f for f in fs_all), win))
    return lines, summary, has_sweep


# ===========================================================================
#  CLI
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    for name in ("bytes", "compile"):
        p = sub.add_parser(name)
        p.add_argument("--folder", default=HERE)
    p = sub.add_parser("ss-pulses")
    p.add_argument("specimen_dirs", nargs="+")
    p.add_argument("--min-overlap", type=int, default=MIN_OVERLAP)
    args = ap.parse_args(argv)
    if args.cmd == "bytes":
        crlf, nonascii, n_py, n_sh = scan_bytes(args.folder)
        ok = not crlf and not nonascii
        print("BYTES: %s (%d .py pure ASCII required, %d .sh/.pbs/.mod LF-only required; CR in %s; non-ASCII in %s)"
              % ("ok" if ok else "FAIL", n_py, n_sh, crlf or "none", nonascii or "none"))
        return 0 if ok else 1
    if args.cmd == "compile":
        n, errors = compile_all(args.folder)
        for e in errors:
            print("  " + e)
        print("COMPILE: %s (%d .py, %d with errors)" % ("ok" if not errors else "FAIL", n, len(errors)))
        return 0 if not errors else 1
    if args.cmd == "ss-pulses":
        all_ok = True
        summaries = []
        for d in args.specimen_dirs:
            lines, summary, ok = describe_specimen(d, args.min_overlap)
            for ln in lines:
                print(ln)
            summaries.append(summary)
            all_ok &= ok
        for s in summaries:
            print(s)
        return 0 if all_ok else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
