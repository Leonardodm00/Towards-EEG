"""
smoke_eyal_archive_builder.py -- NEURON-free acceptance tests for the
Eyal et al. (2016) archive builder.

Run:
    python3 smoke_eyal_archive_builder.py --eyal-root ./195667-master \\
                                          --out-root  ./eyal_archive_smoke
    python3 smoke_eyal_archive_builder.py --eyal-root ./195667-master \\
                                          --out-root  ./eyal_archive_smoke -v

Exit code 0 iff every test passes. Run it TWICE before trusting a build:
the first run creates the archive, the second exercises the
already-exists / overwrite path.

Optional loader round-trip
--------------------------
Tests 12-13 exercise the real pipeline loader. They are SKIPPED unless
--monolith-dir points at the directory containing
passive_fitting_hpc_fixed.py. Test 13 is EXPECTED TO FAIL against the
unpatched loader; that failure is the acceptance criterion for the
amplitude-grouping patch, and the message says so explicitly.

ASCII-only, LF-only by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import eyal_archive_builder as eab


# ---------------------------------------------------------------------------
#  Tiny test harness (no pytest dependency -- this must run inside Colab and
#  on a login node with a minimal environment)
# ---------------------------------------------------------------------------

_TESTS: List[Tuple[str, Callable]] = []


def test(name: str):
    def deco(fn):
        _TESTS.append((name, fn))
        return fn
    return deco


class Ctx(object):
    """Shared state built once and handed to every test."""

    def __init__(self, eyal_root: Path, out_root: Path,
                 monolith_dir: Optional[Path]):
        self.eyal_root = Path(eyal_root)
        self.out_root = Path(out_root)
        self.monolith_dir = Path(monolith_dir) if monolith_dir else None
        self.dirs: List[Path] = []
        self.traces: Dict[str, Dict[str, Any]] = {}


def _all_trace_entries():
    for entry in eab.MANIFEST:
        for rel, amp in entry["traces"]:
            yield entry, rel, amp


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

@test("01 manifest integrity (paths, ids, onsets, 11 traces)")
def t01(ctx: Ctx):
    eab.validate_manifest(ctx.eyal_root)
    assert len(eab.MANIFEST) == 6
    n = sum(len(e["traces"]) for e in eab.MANIFEST)
    assert n == 11, "expected 11 traces, got %d" % n


@test("02 trace parsing: 11 files, 2 columns, dt = 0.02 ms")
def t02(ctx: Ctx):
    for entry, rel, amp in _all_trace_entries():
        tr = eab.parse_eyal_trace(ctx.eyal_root / rel)
        ctx.traces[rel] = tr
        assert abs(tr["dt_ms"] - 0.02) < 1e-6, rel
        assert tr["n_samples"] in (6376, 6454), (rel, tr["n_samples"])
    assert len(ctx.traces) == 11


@test("03 record ends exactly 102.000 ms after onset (all 11 traces)")
def t03(ctx: Ctx):
    # This is the cheapest strong check on t_inj: the record length is
    # t_inj + D + 100 ms by construction for every file in the release, so a
    # wrong onset shows up here immediately.
    for entry, rel, amp in _all_trace_entries():
        tr = ctx.traces[rel]
        end_rel = tr["t_ms"][-1] - entry["t_inj_ms"]
        assert abs(end_rel - eab.EYAL_RECORD_END_POST_ONSET_MS) < 0.021, (
            "%s: record ends %.3f ms after onset, expected %.1f"
            % (rel, end_rel, eab.EYAL_RECORD_END_POST_ONSET_MS))


@test("04 baseline within 0.5 mV of -86 (LJP already applied, not twice)")
def t04(ctx: Ctx):
    for entry, rel, amp in _all_trace_entries():
        tr = ctx.traces[rel]
        t_rel = tr["t_ms"] - entry["t_inj_ms"]
        pre = tr["v_mV"][t_rel < -1.5]
        assert pre.size > 100, rel
        base = float(pre.mean())
        assert abs(base - eab.EYAL_E_PAS_MV) < 0.5, (
            "%s: baseline %.3f mV, expected within 0.5 mV of %.1f"
            % (rel, base, eab.EYAL_E_PAS_MV))


@test("05 time origin: t[0] < 0, t[-1] > 0, a sample sits exactly at t = 0")
def t05(ctx: Ctx):
    for entry, rel, amp in _all_trace_entries():
        rec = eab.build_ss_pulse_record(ctx.traces[rel],
                                        t_inj_ms=entry["t_inj_ms"],
                                        amp_pA=amp)
        t = rec["t"]
        assert t[0] < 0.0, rel
        assert t[-1] > 0.0, rel
        assert abs(t[int(np.argmin(np.abs(t)))]) < 1e-12, (
            "%s: nearest sample to onset is at t = %.3e s" % (rel, t[int(np.argmin(np.abs(t)))]))
        # the pipeline's brief-pulse baseline window is [-10 ms, 0]
        assert t[0] <= -10e-3, "%s: only %.2f ms of pre-stimulus baseline" % (
            rel, -t[0] * 1e3)


@test("06 per-cell onset: 0603_cell03 uses 25.50 ms, the other five 27.06 ms")
def t06(ctx: Ctx):
    # Regression test for the single nastiest trap in this dataset: applying
    # the common 27.06 ms onset to 0603_cell03 misaligns its window by
    # 1.56 ms, landing inside the bridge artefact.
    for entry in eab.MANIFEST:
        expected = 25.50 if entry["cell_tag"] == "0603_cell03" else 27.06
        assert abs(entry["t_inj_ms"] - expected) < 1e-9, entry["cell_tag"]


@test("07 current synthesis: 0 before onset, A on [0, 2 ms), 0 after")
def t07(ctx: Ctx):
    for entry, rel, amp in _all_trace_entries():
        rec = eab.build_ss_pulse_record(ctx.traces[rel],
                                        t_inj_ms=entry["t_inj_ms"],
                                        amp_pA=amp)
        t_ms = rec["t"] * 1e3
        i = rec["i"]
        assert i.shape == rec["v"].shape == rec["t"].shape, rel
        assert np.all(i[t_ms < -0.01] == 0.0), rel
        on = (t_ms >= -0.01) & (t_ms < 1.99)
        assert np.all(i[on] == float(amp)), rel
        assert np.all(i[t_ms > 2.01] == 0.0), rel
        assert int(np.count_nonzero(i)) == 100, (
            "%s: %d active samples, expected 100" % (rel, np.count_nonzero(i)))


@test("08 filename-implied amplitude matches the manifest (all 11)")
def t08(ctx: Ctx):
    seen = 0
    for entry, rel, amp in _all_trace_entries():
        parsed = eab.parse_amplitude_from_filename(Path(rel).name)
        assert parsed is not None, rel
        assert abs(parsed - amp) < 1e-9, (rel, parsed, amp)
        seen += 1
    assert seen == 11


@test("09 artefact exclusion: |dv| in the fit window < |dv| in [0, 3] ms")
def t09(ctx: Ctx):
    # Direct guard against fitting an instrumentation artefact that is
    # several times larger than the physiological signal.
    for entry, rel, amp in _all_trace_entries():
        tr = ctx.traces[rel]
        t_rel = tr["t_ms"] - entry["t_inj_ms"]
        base = float(tr["v_mV"][t_rel < -1.5].mean())
        dv = tr["v_mV"] - base
        art = np.abs(dv[(t_rel >= 0) & (t_rel < 3.0)]).max()
        win = np.abs(dv[(t_rel >= 3.0) & (t_rel <= 102.0)]).max()
        assert win < art, (
            "%s: window max |dv| = %.3f mV is NOT below the [0,3) ms max of "
            "%.3f mV" % (rel, win, art))


@test("10 amplitude homogeneity is recorded, not assumed")
def t10(ctx: Ctx):
    for entry in eab.MANIFEST:
        amps = sorted(set(a for _, a in entry["traces"]))
        pulses = [{"polarity": "dep" if a > 0 else "hyp", "peak_pA": a}
                  for _, a in entry["traces"]]
        groups = eab.group_pulses_by_amplitude(pulses)
        # every group must be amplitude-homogeneous by construction
        for g in groups:
            vals = set(round(pulses[k]["peak_pA"], 6) for k in g)
            assert len(vals) == 1, (entry["cell_tag"], vals)
        assert len(groups) == len(amps), (entry["cell_tag"], len(groups),
                                          len(amps))
        if entry["cell_tag"] == "0603_cell08":
            assert len(groups) == 6, "cell08 must yield 6 amplitude groups"
        else:
            assert len(groups) == 1, entry["cell_tag"]


@test("11 build archive; npz keys/shapes/dtypes match the loader contract")
def t11(ctx: Ctx):
    ctx.dirs = eab.build_eyal_archive(ctx.eyal_root, ctx.out_root,
                                      verbose=False)
    assert len(ctx.dirs) == 6
    required = {"t", "v", "i_pA", "polarity_is_dep", "peak_pA",
                "stim_duration_s", "sampling_rate_Hz", "sweep_number"}
    for d in ctx.dirs:
        assert (d / "metadata.json").is_file(), d
        assert (d / "morphology.asc").is_file(), d
        assert (d / "ss_pulses.npz").is_file(), d
        # Absence of ls_sweeps.npz is deliberate and must stay absent.
        assert not (d / "ls_sweeps.npz").exists(), (
            "%s: a Long Square file was written; the Eyal release has none "
            "and a synthetic one must never be fabricated" % d)

        z = np.load(d / "ss_pulses.npz")
        assert required.issubset(set(z.files)), (d, sorted(z.files))
        assert "variable_length" not in z.files, d  # Format A only
        n = z["v"].shape[0]
        assert z["t"].ndim == 1
        assert z["v"].shape == z["i_pA"].shape == (n, z["t"].size)
        assert z["polarity_is_dep"].dtype == np.bool_
        assert z["sweep_number"].dtype == np.int64
        for key in ("peak_pA", "stim_duration_s", "sampling_rate_Hz"):
            assert z[key].dtype == np.float64 and z[key].shape == (n,)
        assert np.allclose(z["stim_duration_s"], 2.0e-3)
        assert np.allclose(z["sampling_rate_Hz"], 50000.0)

        meta = json.loads((d / "metadata.json").read_text(encoding="ascii"))
        assert meta["ljp_correction_mV"] == 16.0
        assert meta["v_rest_mV"] == -86.0
        assert meta["morphology_file"] == "morphology.asc"
        assert meta["ls_sweeps"] == []
        assert meta["full_allen_metadata"]["structure_layer_name"] == "2/3"
        assert meta["reference_scalars_are_derived"] is True
        # tau surrogate must equal Cm* x Rm* exactly
        rp = meta["reference_published"]
        tau = rp["cm_uF_per_cm2"] * rp["rm_Ohm_cm2"] * 1e-3
        assert abs(meta["tau_ms"] - tau) < 1e-9, d


def _loader_supports(mono, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(
            mono.load_cell_from_archive).parameters
    except Exception:
        return False


@test("12 [optional] pipeline loader round-trip: 6 cells, non-empty SS")
def t12(ctx: Ctx):
    mono = _try_import_monolith(ctx)
    if mono is None:
        raise _Skip("no --monolith-dir given")
    n_ok = 0
    for d in sorted(ctx.out_root.glob("specimen_*")):
        cd = mono.load_cell_from_archive(d, n_avg_groups=1, verbose=False)
        assert cd.square_subthreshold, d
        assert not cd.long_square_subthreshold, d
        n_ok += 1
    assert n_ok == 6, n_ok

    # The batch loader drops every cell unless require_long_square=False.
    import inspect
    if "require_long_square" in inspect.signature(
            mono.load_cells_from_archive).parameters:
        cells = mono.load_cells_from_archive(
            ctx.out_root, require_long_square=False, verbose=False)
        assert len(cells) == 6, (
            "load_cells_from_archive(require_long_square=False) returned %d "
            "cells, expected 6" % len(cells))
        dropped = mono.load_cells_from_archive(ctx.out_root, verbose=False)
        assert len(dropped) == 0, (
            "require_long_square=True should still drop all six Eyal cells, "
            "got %d" % len(dropped))


@test("13 [optional] loader must not average distinct amplitudes together")
def t13(ctx: Ctx):
    mono = _try_import_monolith(ctx)
    if mono is None:
        raise _Skip("no --monolith-dir given")
    nominal = {50.0, 100.0, 200.0}
    d = ctx.out_root / "specimen_60308"

    patched = _loader_supports(mono, "group_ss_by_amplitude")
    kw = {"group_ss_by_amplitude": True} if patched else {}
    cd = mono.load_cell_from_archive(d, n_avg_groups=1, verbose=False, **kw)

    for b in cd.square_subthreshold:
        assert abs(b.amplitude_pA) in nominal, (
            "loader produced a bundle labelled %+.1f pA, which is not a "
            "nominal amplitude. This is the amplitude-averaging defect: "
            "load_cell_from_archive groups SS pulses by POLARITY only, so "
            "cell 0603_cell08's +50/+100/+200 pA traces are averaged into "
            "one meaningless bundle. Apply patch 1 "
            "(patch_eyal_support.py) and pass group_ss_by_amplitude=True, "
            "or run with --n-avg-groups 3 as a stopgap."
            % b.amplitude_pA)

    if patched:
        assert len(cd.square_subthreshold) == 6, (
            "expected 6 amplitude-homogeneous bundles for cell08, got %d"
            % len(cd.square_subthreshold))
        # With the flag OFF the pool is heterogeneous and must warn loudly
        # rather than silently averaging.
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            mono.load_cell_from_archive(d, n_avg_groups=1, verbose=False)
        assert any("group_ss_by_amplitude" in str(x.message) for x in caught), (
            "the heterogeneous-amplitude pool did not raise a warning when "
            "group_ss_by_amplitude was left False")


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

class _Skip(Exception):
    pass


def _try_import_monolith(ctx: Ctx):
    if ctx.monolith_dir is None:
        return None
    if str(ctx.monolith_dir) not in sys.path:
        sys.path.insert(0, str(ctx.monolith_dir))
    import importlib
    return importlib.import_module("passive_fitting_hpc_fixed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eyal-root", required=True,
                    help="extracted ModelDB 195667 directory")
    ap.add_argument("--out-root", required=True,
                    help="where to build the test archive")
    ap.add_argument("--monolith-dir", default=None,
                    help="directory containing passive_fitting_hpc_fixed.py; "
                         "enables tests 12-13")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    ctx = Ctx(Path(args.eyal_root), Path(args.out_root),
              Path(args.monolith_dir) if args.monolith_dir else None)

    n_pass = n_fail = n_skip = 0
    print("=" * 72)
    print("smoke_eyal_archive_builder -- %d test(s)" % len(_TESTS))
    print("=" * 72)
    for name, fn in _TESTS:
        try:
            fn(ctx)
        except _Skip as s:
            n_skip += 1
            print("SKIP  %s  (%s)" % (name, s))
        except Exception as exc:
            n_fail += 1
            print("FAIL  %s" % name)
            print("      %s: %s" % (type(exc).__name__, exc))
            if args.verbose:
                traceback.print_exc()
        else:
            n_pass += 1
            print("PASS  %s" % name)
    print("=" * 72)
    print("pass=%d  fail=%d  skip=%d" % (n_pass, n_fail, n_skip))
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
