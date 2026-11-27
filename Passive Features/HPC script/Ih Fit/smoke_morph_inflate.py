#!/usr/bin/env python3
"""smoke_morph_inflate.py -- Block 29 (SPEC 2.11, D-268): twins simulated on a
morphology whose proximal radii are inflated, fitted on the original one.

    python smoke_morph_inflate.py [--build]

M1-M3 are pure (morph_inflate.py); M4-M6 need NEURON, ./x86_64 and the
fixture of smoke_twin_recovery.py. One PASS/FAIL line per check; exit code 0
when nothing failed. About 1 min.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import morph_inflate as MI          # noqa: E402
import smoke_twin_recovery as STR   # noqa: E402  (the fixture, the stubs, the runner's helpers)

FACTOR, MAX_PATH = 1.5, 300.0
_S = {}

# a hand SWC: soma root (r 5), a basal branch of nodes 100 um apart (r 1) out to 500 um, an apical node at
# 250 um (r 2) with a child at 350 um (r 2), an axon node at 50 um (r 0.5)
HAND = """# hand morphology
1 1 0 0 0 5.0 -1
2 3 100 0 0 1.0 1
3 3 200 0 0 1.0 2
4 3 300 0 0 1.0 3
5 3 400 0 0 1.0 4
6 3 500 0 0 1.0 5
7 4 0 250 0 2.0 1
8 4 0 350 0 2.0 7
9 2 0 0 50 0.5 1
"""


def _fail_parts(parts):
    bad = [k for k, v in parts.items() if not v]
    assert not bad, "failed parts: " + "; ".join(bad)


def _hand(tmp):
    src = Path(tmp) / "hand.swc"
    src.write_text(HAND)
    return src


def test_M1_path_length_and_step():
    """The path length is the sum along the parent links; the factor applies up to and INCLUDING max_path_um
    (a hard step), to soma and dendrites, not to the axon."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _hand(tmp)
        rows, header = MI.read_swc(src)
        d = MI.path_length_from_root(rows)
        dst = Path(tmp) / "out.swc"
        c = MI.inflate_swc(src, dst, factor=FACTOR, max_path_um=MAX_PATH)
        out, header2 = MI.read_swc(dst)
        parts = {
            "path lengths 0,100,...,500 / 250,350 / 50": np.allclose(d, [0, 100, 200, 300, 400, 500, 250, 350, 50],
                                                                    rtol=0, atol=1e-12),
            # tolerance: the radii are written with 10 significant digits
            "radii: soma 7.5; basal 1.5 to 300 um then 1; apical 3 at 250, 2 at 350; axon 0.5": np.allclose(
                out[:, 5], [7.5, 1.5, 1.5, 1.5, 1.0, 1.0, 3.0, 2.0, 0.5], rtol=1e-9, atol=0),
            "ids, types, coordinates, parents unchanged": np.array_equal(out[:, [0, 1, 2, 3, 4, 6]],
                                                                         rows[:, [0, 1, 2, 3, 4, 6]]),
            "counts: 5 of 9 inflated (1 soma, 3 basal, 1 apical)": (
                c["n_nodes"] == 9 and c["n_inflated"] == 5 and c["n_by_type"] == {1: 1, 3: 3, 4: 1}),
            "header kept, one comment added": header2[:len(header)] == header and len(header2) == len(header) + 1,
            "the source file is untouched": src.read_text() == HAND,
            "every value but the inflated radii reads back exactly": (
                np.array_equal(out[:, :5], rows[:, :5]) and np.array_equal(out[[4, 5, 7, 8], 5], rows[[4, 5, 7, 8], 5])),
        }
        # the final check's F3: a detached tree is not inflated; no soma root or two of them is refused
        two = Path(tmp) / "two.swc"
        two.write_text(HAND + "10 3 0 0 900 1.0 -1\n11 3 0 0 910 1.0 10\n")
        c2 = MI.inflate_swc(two, Path(tmp) / "two_out.swc", factor=FACTOR, max_path_um=MAX_PATH)
        o2, _ = MI.read_swc(Path(tmp) / "two_out.swc")
        parts["a detached dendrite tree is copied unchanged and counted"] = (
            np.array_equal(o2[9:, 5], [1.0, 1.0]) and c2["n_outside_soma_tree"] == 2 and c2["n_inflated"] == 5)
        for lab, text in (("no soma root", HAND.replace("1 1 0 0 0 5.0 -1", "1 3 0 0 0 5.0 -1")),
                          ("two soma roots", HAND + "10 1 0 0 900 5.0 -1\n")):
            p = Path(tmp) / "bad.swc"
            p.write_text(text)
            try:
                MI.inflate_swc(p, Path(tmp) / "bad_out.swc", factor=FACTOR, max_path_um=MAX_PATH)
                parts[lab + ": refused"] = False
            except ValueError:
                parts[lab + ": refused"] = True
        _fail_parts(parts)


def test_M2_identity_and_limits():
    """factor 1 reproduces the radii; max_path 0 inflates the root only; a huge max_path inflates every soma
    and dendrite node and never the axon."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _hand(tmp)
        rows, _ = MI.read_swc(src)
        r1, _ = MI.read_swc(_run_inflate(src, tmp, "a", 1.0, 300.0))
        r0, _ = MI.read_swc(_run_inflate(src, tmp, "b", 2.0, 0.0))
        rb, _ = MI.read_swc(_run_inflate(src, tmp, "c", 2.0, 1e9))
        parts = {"factor 1: radii unchanged": np.allclose(r1[:, 5], rows[:, 5], rtol=1e-9, atol=0),
                 "max_path 0: the root only": np.allclose(r0[:, 5] / rows[:, 5], [2, 1, 1, 1, 1, 1, 1, 1, 1]),
                 "max_path huge: all but the axon": np.allclose(rb[:, 5] / rows[:, 5], [2, 2, 2, 2, 2, 2, 2, 2, 1])}
        _fail_parts(parts)


def _run_inflate(src, tmp, tag, factor, max_path):
    dst = Path(tmp) / ("o_%s.swc" % tag)
    MI.inflate_swc(src, dst, factor=factor, max_path_um=max_path)
    return dst


def test_M3_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        src = _hand(tmp)
        bad = {}
        for lab, kw in (("factor 0", dict(factor=0.0, max_path_um=1.0)), ("factor nan", dict(factor=float("nan"),
                                                                                           max_path_um=1.0)),
                        ("max_path < 0", dict(factor=1.5, max_path_um=-1.0)),
                        ("max_path inf", dict(factor=1.5, max_path_um=float("inf")))):
            try:
                MI.inflate_swc(src, Path(tmp) / "x.swc", **kw)
                bad[lab] = False
            except ValueError:
                bad[lab] = True
        try:
            MI.inflate_swc(src, src, factor=1.5, max_path_um=1.0)
            bad["dst is src"] = False
        except ValueError:
            bad["dst is src"] = True
        cyc = Path(tmp) / "cyc.swc"
        cyc.write_text("1 1 0 0 0 1 2\n2 3 1 0 0 1 1\n")
        orphan = Path(tmp) / "orph.swc"
        orphan.write_text("1 1 0 0 0 1 -1\n2 3 1 0 0 1 7\n")
        for lab, p in (("a cycle", cyc), ("a parent that is not a node", orphan)):
            try:
                MI.path_length_from_root(MI.read_swc(p)[0])
                bad[lab] = False
            except ValueError:
                bad[lab] = True
        _fail_parts(bad)


def _runs():
    """Two planned runs on the fixture: one plain, one with --twin-diam-factor; the noise-free twin of each
    built by the `twin` sub-command with a minimal fit budget."""
    if "plain" not in _S:
        import run_twin_recovery as T
        for key, extra in (("plain", ()), ("infl", ("--twin-diam-factor", str(FACTOR),
                                                    "--twin-diam-max-path-um", str(MAX_PATH)))):
            out = STR._new_out("mi_" + key)
            code, so, se = STR._plan(out, "--refine-max-nfev", "2", *extra)
            assert code == 0, "plan (%s) exited %d: %s" % (key, code, se[-2000:])
            man = STR._manifest(out)
            tid = {str(c): int(t) for c, t in zip(man["configuration"], man["specimen_id"])}
            r = STR._sub(["twin", "--out", out, "--twin", tid["full_no_noise"]])
            assert r.returncode == 0, "twin (%s) exited %d: %s" % (key, r.returncode, r.stderr[-2000:])
            _S[key] = dict(out=out, tid=tid["full_no_noise"], man=man,
                           cfg=json.loads((out / "run_config.json").read_text()))
        _S["T"] = T
    return _S


def _spec_dir(run):
    return run["out"] / "twins" / ("twin_%d" % run["tid"]) / ("specimen_%d" % run["tid"])


def test_M4_twin_on_inflated_fit_on_original():
    """The inflated run's archive stores the ORIGINAL reconstruction (the fit's morphology) and records the
    inflation; the plain run records none; the plan's options carry the two keys only when inflated."""
    st = _runs()
    T = st["T"]
    real = Path(st["plain"]["cfg"]["options"]["group_dir"]) / ("specimen_%d" % int(
        st["plain"]["man"].set_index("specimen_id").loc[st["plain"]["tid"], "twin_of_specimen_id"]))
    orig = (real / "reconstruction.swc").read_bytes()
    mi = json.loads((_spec_dir(st["infl"]) / "metadata.json").read_text())["twin"]
    mp = json.loads((_spec_dir(st["plain"]) / "metadata.json").read_text())["twin"]
    sim = st["infl"]["out"] / "twins" / ("twin_%d" % st["infl"]["tid"]) / ("sim_morphology_%d.swc" % st["infl"]["tid"])
    r_o, _ = MI.read_swc(real / "reconstruction.swc")
    r_s, _ = MI.read_swc(sim) if sim.is_file() else (np.zeros((0, 7)), [])
    d = MI.path_length_from_root(r_o)
    typ = r_o[:, 1].astype(int)
    exp = np.where(np.isin(typ, (1, 3, 4)) & (d <= MAX_PATH), FACTOR, 1.0)
    oi, op = st["infl"]["cfg"]["options"], st["plain"]["cfg"]["options"]
    sm = (mi.get("provenance") or {}).get("sim_morphology") or {}
    parts = {
        "inflated run: the archive's reconstruction.swc is the real cell's, byte for byte": (
            (_spec_dir(st["infl"]) / "reconstruction.swc").read_bytes() == orig),
        "plain run: likewise": (_spec_dir(st["plain"]) / "reconstruction.swc").read_bytes() == orig,
        "the simulation morphology exists beside the archive, radii = original x expected factor": (
            r_s.shape == r_o.shape and np.allclose(r_s[:, 5], r_o[:, 5] * exp, rtol=1e-9, atol=0)),
        "provenance.sim_morphology: factor, max_path_um, counts": (
            sm.get("factor") == FACTOR and sm.get("max_path_um") == MAX_PATH
            and sm.get("n_inflated") == int((exp != 1.0).sum()) and sm.get("n_nodes") == int(r_o.shape[0])),
        "plain run: no sim_morphology in the provenance": "sim_morphology" not in (mp.get("provenance") or {}),
        "run_config options: the two keys only in the inflated run": (
            oi.get("twin_diam_factor") == FACTOR and oi.get("twin_diam_max_path_um") == MAX_PATH
            and "twin_diam_factor" not in op and "twin_diam_max_path_um" not in op),
        "twin_inflation(options)": (T.twin_inflation(oi) == dict(factor=FACTOR, max_path_um=MAX_PATH)
                                    and T.twin_inflation(op) is None),
    }
    print("\n  [M4] fixture morphology: %d of %d nodes inflated (x %g within %g um)"
          % (int((exp != 1.0).sum()), r_o.shape[0], FACTOR, MAX_PATH))
    _fail_parts(parts)


def test_M5_traces_differ_as_a_thicker_cell():
    """Same truth, same stimuli, no noise: the inflated twin's long-step deflections are smaller in magnitude
    (a thicker cell has a lower input resistance), and its recorded features say so."""
    st = _runs()
    lp = np.load(_spec_dir(st["plain"]) / "ls_sweeps.npz", allow_pickle=False)
    li = np.load(_spec_dir(st["infl"]) / "ls_sweeps.npz", allow_pickle=False)
    # the archive stores one voltage array per long sweep: v_<m>
    vk = sorted((k for k in lp.files if k.startswith("v_")), key=lambda k: int(k.split("_")[1]))
    assert vk and vk == sorted((k for k in li.files if k.startswith("v_")), key=lambda k: int(k.split("_")[1])), \
        "the two archives do not hold the same long sweeps"
    same_shape = all(lp[k].shape == li[k].shape for k in vk)
    dp = np.array([np.max(np.abs(np.asarray(lp[k], float) - float(lp[k][0]))) for k in vk])
    di = np.array([np.max(np.abs(np.asarray(li[k], float) - float(li[k][0]))) for k in vk])
    mp = json.loads((_spec_dir(st["plain"]) / "metadata.json").read_text())
    mi = json.loads((_spec_dir(st["infl"]) / "metadata.json").read_text())

    def rin(m):
        for blk in (m, m.get("twin") or {}, m.get("features") or {}, (m.get("twin") or {}).get("features") or {}):
            for k in ("rin_MOhm", "input_resistance_MOhm", "ri"):
                if k in blk:
                    return float(blk[k])
        return float("nan")
    rp, ri = rin(mp), rin(mi)
    parts = {"same sweeps and samples": same_shape,
             "every long step: the inflated twin's peak deflection is smaller": bool(np.all(di < dp)),
             "the truths are equal": mp["twin"]["theta"] == mi["twin"]["theta"]}
    if np.isfinite(rp) and np.isfinite(ri):
        parts["recorded input resistance: inflated below plain"] = ri < rp
    print("\n  [M5] peak deflection, inflated / plain, per long sweep: %s; R_in %.1f -> %.1f MOhm"
          % (np.round(di / dp, 3).tolist(), rp, ri))
    _fail_parts(parts)


def test_M6_archive_check_and_fit_morphology():
    """An archive is reused only by a run with the same inflation; the fit of the inflated twin ran on the
    archive's original morphology (its morphology_source is the archive)."""
    import pandas as pd
    st = _runs()
    T = st["T"]
    import twin_manifest as TM
    from twin_config import TwinConfig
    import dataclasses
    out = {}
    for key in ("plain", "infl"):
        run = st[key]
        row = run["man"].set_index("specimen_id", drop=False).loc[run["tid"]]
        spec = TM.twin_spec_of(row, F=float(run["cfg"]["options"]["F"]))
        tcfg = dataclasses.replace(TwinConfig(), base_seed=int(run["cfg"]["options"]["base_seed"]))
        out[key] = (spec, tcfg, bool(row["with_noise"]))
    infl = dict(factor=FACTOR, max_path_um=MAX_PATH)
    a = T._archive_matches(_spec_dir(st["infl"]), *out["infl"], infl)
    b = T._archive_matches(_spec_dir(st["infl"]), *out["infl"], None)
    c = T._archive_matches(_spec_dir(st["plain"]), *out["plain"], None)
    d = T._archive_matches(_spec_dir(st["plain"]), *out["plain"], infl)
    e = T._archive_matches(_spec_dir(st["infl"]), *out["infl"], dict(factor=2.0, max_path_um=MAX_PATH))
    fit = pd.read_csv(st["infl"]["out"] / "fits" / ("twin_%d" % st["infl"]["tid"]) / "phase2_results.csv")
    src = str(fit["morphology_source"].iloc[0]) if "morphology_source" in fit.columns else "?"
    parts = {"inflated archive, inflated run: matches": a is None,
             "inflated archive, plain run: refused": b is not None and "inflation" in b,
             "plain archive, plain run: matches": c is None,
             "plain archive, inflated run: refused": d is not None and "inflation" in d,
             "inflated archive, another factor: refused": e is not None and "inflation" in e,
             "the inflated twin's fit read the archive's morphology": src == "archive"}
    _fail_parts(parts)


def test_M8_plan_refusals():
    """The final check's F1: an invalid inflation option is a clean refusal of plan (exit 2, one message, no
    traceback), and nothing is written."""
    import run_twin_recovery as T
    parts = {}
    for lab, extra in (("factor 0", ("--twin-diam-factor", "0")), ("factor nan", ("--twin-diam-factor", "nan")),
                       ("path -1", ("--twin-diam-factor", "1.5", "--twin-diam-max-path-um", "-1"))):
        out = STR._new_out("mi_bad")
        code, so, se = STR._plan(out, *extra)
        parts["%s: exit %d, no traceback, nothing written" % (lab, T.EXIT_USAGE)] = (
            code == T.EXIT_USAGE and "Traceback" not in se and not (out / "run_config.json").exists())
    _fail_parts(parts)


def test_M7_ascii_lf():
    for name in ("morph_inflate.py", "smoke_morph_inflate.py", "twin_build.py", "run_twin_recovery.py"):
        raw = (HERE / name).read_bytes()
        assert all(b < 128 for b in raw), "%s: non-ASCII byte" % name
        assert b"\r" not in raw, "%s: CR" % name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="run `nrnivmodl mod` when ./x86_64/special is missing")
    args = ap.parse_args(argv)
    if args.build and not (HERE / "x86_64" / "special").exists():
        subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), check=True)
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(STR._environment())
    results = []
    t_all = time.perf_counter()
    for name, func in checks:
        start = time.perf_counter()
        try:
            ret = func()
            status, detail = "PASS", (ret if isinstance(ret, str) else "")
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(name) for name, _, _, _ in results] + [4])
    print()
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, (lines[0] if lines else "")[:400])).rstrip())
    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items())
          + "; %.1f s in all" % (time.perf_counter() - t_all))
    return 1 if counts["FAIL"] + counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
