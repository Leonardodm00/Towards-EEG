"""
smoke_noise_fit_gates.py -- smoke suite for noise_fit_gates.py and
cluster_gates_noise_fit.sh (T1-T8)
=====================================================================

No NEURON, no archive; about 20 s (T7 runs the two fast suites). The fixtures
are written the way the Phase-0 archive writer writes them
(`save_alleninstitute_data (6).py`: windows (-10, +200) ms around each pulse
onset, cut from one sweep's trace, stored in onset order with their sweep
number), so every expected value below is known by construction.

  T1  bytes: on a fixture folder, exactly the planted files are flagged (a
      .py with one byte above 0x7F; a .sh and a mod/*.mod with CRLF); a clean
      folder gives "BYTES: ok" and exit 0
  T2  compile: a .py with a syntax error is named and gives exit 1; no
      __pycache__ is written by the check
  T3  ss-pulses, Format A, two sweeps at 50 kHz of an ADC-quantised trace
      (steps of 1/32 mV, so equal values recur and only a long overlap can
      prove adjacency): sweep 41, 20 pulses exactly 200 ms apart; sweep 42,
      20 pulses 200.04 ms apart with the 8th dropped (as the writer's QC
      drops a pulse); alternating polarity. Expected summary exactly:
      n_sweeps=2 per_sweep=20,19 adjacent=36/37 opposite_polarity_adjacent=
      36/36 spacing_ms=200.000,200.040 window_ms=-10.000..200.000, the gap
      reported before window 7 of sweep 42, the metadata key listed, exit 0
  T4  Format B (the same pulses stored one by one): the same summary with
      format=B
  T5  a file without sweep_number: sweep_number=no, three chains of
      consecutive adjacent windows (20, 7, 12), exit 1
  T6  onset_spacing: two windows of DIFFERENT quantised traces are never
      adjacent (200 random pairs, although their values recur); a true
      overlap shorter than MIN_OVERLAP is refused, and found when the
      minimum is lowered
  T7  the gate script: bash -n; QUICK=1 run here prints PASS for G0, G1, G3,
      G4, G5, G6, a "GATES noise-fit:" line and a "VERDICT: NOT SETTLED"
      line (QUICK can never settle); with PYTHON=/nonexistent G0 FAILs, the
      summary is still printed and the exit status is 1
  T8  byte safety: the three files are pure ASCII with LF endings

Run from this folder:

    python smoke_noise_fit_gates.py

Expected last line: "SMOKE noise_fit_gates: 8/8 passed". Exit 1 on any FAIL.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_fit_gates as G  # noqa: E402

RESULTS = []
FS = 50_000.0
PRE_N, POST_N = 500, 10_000                      # 10 ms, 200 ms at 50 kHz
QUANT = 1.0 / 32.0                                # mV, an ADC-like step


def report(name, ok, note=""):
    RESULTS.append((name, bool(ok), note))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + note) if note else ""))


def quantised_trace(rng, n):
    """A -70 mV baseline plus correlated noise, rounded to QUANT (mV)."""
    e = np.empty(n)
    e[0] = rng.normal(0.0, 0.06)
    z = rng.normal(0.0, 0.06 * np.sqrt(1 - 0.9 ** 2), n)
    for j in range(1, n):
        e[j] = 0.9 * e[j - 1] + z[j]
    return -70.0 + np.round(e / QUANT) * QUANT


def make_sweep(rng, onsets, n_samples):
    """One sweep: trace, current with +/-200 pA 0.5 ms pulses (alternating,
    first depolarising) at `onsets` (samples)."""
    v = quantised_trace(rng, n_samples)
    i = np.zeros(n_samples)
    for k, s in enumerate(onsets):
        i[s:s + 25] = 200.0 if k % 2 == 0 else -200.0
        v[s:s + 500] += (1.0 if k % 2 == 0 else -1.0) * np.exp(-np.arange(500) / 100.0)
    return v, i


def cut_windows(v, i, onsets, sweep_number, keep):
    """The writer's _extract_windows_around_pulses: (-PRE_N, +POST_N) samples."""
    out = []
    for k, s in enumerate(onsets):
        if k not in keep or s - PRE_N < 0 or s + POST_N > len(v):
            continue
        out.append(dict(v=v[s - PRE_N:s + POST_N].copy(), i=i[s - PRE_N:s + POST_N].copy(),
                        dep=(k % 2 == 0), sweep=sweep_number))
    return out


def build_fixture_pulses(seed=11):
    rng = np.random.default_rng(seed)
    on_a = [25_000 + 10_000 * k for k in range(20)]
    v_a, i_a = make_sweep(rng, on_a, on_a[-1] + POST_N + 2_000)
    on_b = [25_000 + 10_002 * k for k in range(20)]
    v_b, i_b = make_sweep(rng, on_b, on_b[-1] + POST_N + 2_000)
    pulses = cut_windows(v_a, i_a, on_a, 41, set(range(20)))
    pulses += cut_windows(v_b, i_b, on_b, 42, set(range(20)) - {7})
    return pulses


def write_specimen(root, sid, pulses, fmt="A", with_sweep=True):
    d = Path(root) / ("specimen_%d" % sid)
    d.mkdir(parents=True)
    t = (np.arange(PRE_N + POST_N) - PRE_N) / FS
    meta = dict(polarity_is_dep=np.array([p["dep"] for p in pulses], dtype=bool),
                peak_pA=np.array([200.0 if p["dep"] else -200.0 for p in pulses]),
                stim_duration_s=np.full(len(pulses), 5e-4),
                sampling_rate_Hz=np.full(len(pulses), FS))
    if with_sweep:
        meta["sweep_number"] = np.array([p["sweep"] for p in pulses], dtype=np.int64)
    if fmt == "A":
        data = dict(t=t, v=np.stack([p["v"] for p in pulses]), i_pA=np.stack([p["i"] for p in pulses]), **meta)
    else:
        data = dict(n_pulses=np.array([len(pulses)]), variable_length=np.array([True]), **meta)
        for k, p in enumerate(pulses):
            data["t_%d" % k], data["v_%d" % k], data["i_%d" % k] = t, p["v"], p["i"]
    np.savez_compressed(d / "ss_pulses.npz", **data)
    json.dump({"specimen_id": sid, "ss": {"sweep_numbers": [41, 42], "n_pulses": len(pulses)},
               "ls_sweeps": [{"sweep_number": 7}]}, open(d / "metadata.json", "w"))
    return d


def run_gates(argv):
    proc = subprocess.run([sys.executable, str(HERE / "noise_fit_gates.py")] + argv,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout


def _t1():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "mod").mkdir()
        (Path(tmp) / "clean.py").write_bytes(b"x = 1\n")
        (Path(tmp) / "bad.py").write_bytes(b"# caf\xc3\xa9\nx = 1\n")
        (Path(tmp) / "ok.sh").write_bytes(b"#!/bin/bash\necho ok\n")
        (Path(tmp) / "bad.sh").write_bytes(b"#!/bin/bash\r\necho bad\r\n")
        (Path(tmp) / "mod" / "x.mod").write_bytes(b"NEURON {\r\n}\r\n")
        rc_bad, out_bad = run_gates(["bytes", "--folder", tmp])
        planted = ("bad.py" in out_bad and "bad.sh" in out_bad and os.path.join("mod", "x.mod") in out_bad
                   and "clean.py" not in out_bad and "ok.sh" not in out_bad)
        for f in ("bad.py", "bad.sh", os.path.join("mod", "x.mod")):
            os.remove(os.path.join(tmp, f))
        rc_ok, out_ok = run_gates(["bytes", "--folder", tmp])
    report("T1 bytes: the planted .py, .sh and .mod flagged, nothing else; a clean folder passes",
           rc_bad == 1 and planted and rc_ok == 0 and out_ok.startswith("BYTES: ok"),
           out_bad.strip().splitlines()[-1])


def _t2():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "good.py").write_text("def f():\n    return 1\n")
        (Path(tmp) / "broken.py").write_text("def f(:\n    return 1\n")
        rc, out = run_gates(["compile", "--folder", tmp])
        no_cache = not (Path(tmp) / "__pycache__").exists()
    report("T2 compile: the broken file named, exit 1, no .pyc written",
           rc == 1 and "broken.py" in out and "good.py" not in out and no_cache and "COMPILE: FAIL" in out,
           out.strip().splitlines()[-1])


EXPECT_A = ("SS_PULSES 900000001: format=A sweep_number=yes n_pulses=39 n_sweeps=2 per_sweep=20,19 "
            "adjacent=36/37 opposite_polarity_adjacent=36/36 spacing_ms=200.000,200.040 fs_Hz=50000 "
            "window_ms=-10.000..200.000")


def _t3():
    with tempfile.TemporaryDirectory() as tmp:
        d = write_specimen(tmp, 900000001, build_fixture_pulses(), fmt="A")
        rc, out = run_gates(["ss-pulses", str(d)])
    lines = out.strip().splitlines()
    gap_ok = any("sweep 42" in ln and "gap before window(s) [7]" in ln for ln in lines)
    meta_ok = any("metadata.json['ss']" in ln for ln in lines) and any("metadata.json['ls_sweeps']" in ln for ln in lines)
    report("T3 ss-pulses Format A: counts, adjacency, both spacings, the gap, metadata keys",
           rc == 0 and lines[-1] == EXPECT_A and gap_ok and meta_ok, lines[-1])


def _t4():
    with tempfile.TemporaryDirectory() as tmp:
        d = write_specimen(tmp, 900000001, build_fixture_pulses(), fmt="B")
        rc, out = run_gates(["ss-pulses", str(d)])
    last = out.strip().splitlines()[-1]
    report("T4 ss-pulses Format B: the same summary, format=B", rc == 0 and last == EXPECT_A.replace("format=A", "format=B"), last)


def _t5():
    with tempfile.TemporaryDirectory() as tmp:
        d = write_specimen(tmp, 900000001, build_fixture_pulses(), fmt="A", with_sweep=False)
        rc, out = run_gates(["ss-pulses", str(d)])
    last = out.strip().splitlines()[-1]
    ok = rc == 1 and "sweep_number=no" in last and "n_chains=3 per_chain=20,7,12" in last and "adjacent=36/36" in last
    report("T5 no sweep_number: reported, three adjacency chains (20, 7, 12), exit 1", ok, last)


def _t6():
    rng = np.random.default_rng(6)
    false_hits = 0
    for _ in range(200):
        a = quantised_trace(rng, PRE_N + POST_N)
        b = quantised_trace(rng, PRE_N + POST_N)
        ia = np.zeros_like(a)
        ib = np.zeros_like(b)
        false_hits += G.onset_spacing(a, ia, b, ib) is not None
    v, i = make_sweep(rng, [1000, 1000 + 10_450], 30_000)
    wa, wb = v[500:11_000], v[10_950:21_450]                      # true overlap: 50 samples
    ca, cb = i[500:11_000], i[10_950:21_450]
    refused = G.onset_spacing(wa, ca, wb, cb) is None
    found = G.onset_spacing(wa, ca, wb, cb, min_overlap=20) == 10_450
    report("T6 onset_spacing: no false adjacency in 200 independent pairs; short overlaps refused unless allowed",
           false_hits == 0 and refused and found, "false hits %d; 50-sample overlap refused %s, found at min 20 %s"
           % (false_hits, refused, found))


def _t7():
    script = HERE / "cluster_gates_noise_fit.sh"
    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True).returncode == 0
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, QUICK="1", GATE_LOG_DIR=os.path.join(tmp, "logs"), PYTHON=sys.executable)
        good = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, cwd=tmp)
        env_bad = dict(env, PYTHON="/nonexistent/python", GATE_LOG_DIR=os.path.join(tmp, "logs_bad"))
        bad = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env_bad, cwd=tmp)
    out = good.stdout
    need = ("PASS  G0 interpreter", "PASS  G1 numpy, scipy, statsmodels", "PASS  G3 bytes", "PASS  G4 syntax",
            "PASS  G5 smoke_noise_stats (Block 1)", "PASS  G6 smoke_noise_law (Block 2)")
    good_ok = all(any(ln.startswith(n) for ln in out.splitlines()) for n in need)
    tail_ok = "GATES noise-fit:" in out and "VERDICT: NOT SETTLED" in out and "VERDICT: SETTLED" not in out
    bad_ok = (bad.returncode == 1 and any(ln.startswith("FAIL  G0 interpreter") for ln in bad.stdout.splitlines())
              and "GATES noise-fit:" in bad.stdout)
    missing = [n for n in need if not any(ln.startswith(n) for ln in out.splitlines())]
    report("T7 gate script: bash -n; QUICK run passes G0, G1, G3-G6 and cannot settle; a bad PYTHON fails G0, exit 1",
           syntax and good_ok and tail_ok and bad_ok,
           "missing %s; bad-python exit %d" % (missing or "none", bad.returncode))


def _t8():
    bad = []
    for f in ("noise_fit_gates.py", "smoke_noise_fit_gates.py", "cluster_gates_noise_fit.sh"):
        b = (HERE / f).read_bytes()
        if any(c > 127 for c in b):
            bad.append(f + ":non-ascii")
        if b"\r" in b:
            bad.append(f + ":CR")
    report("T8 ASCII + LF on the three files", not bad, ", ".join(bad) if bad else "ok")


def main():
    for fn in (_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8):
        try:
            fn()
        except BaseException as exc:          # SystemExit included (D-010 (iii) lineage)
            traceback.print_exc()
            report(fn.__name__ + " raised", False, repr(exc))
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("SMOKE noise_fit_gates: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
