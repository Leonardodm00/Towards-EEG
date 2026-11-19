"""Smoke test for the twin's clean responses -- Block 11 in specs/SPEC.md.

simulate_ss_clean (one Square Subthreshold pulse from rest on a stored
window's time axis) and simulate_ls_clean (one Long Square sweep at its real
amplitude and timing), both in synthetic_ground_truth.py, below the banner
"The twins' clean responses" (SPEC 2.7 (f); D-183.1, D-186.4, D-187).

Checks TR1-TR9 of the Block 11 entry, on ball-and-stick cells written by
synthetic_ground_truth.write_ball_and_stick_swc, passive and with I_h
(GroundTruthParams.ih, the Rich 2021 human kinetics "Ih_human"):
    TR1 legacy parity      v of the new functions == the zero-noise bundles of
                           simulate_ss_protocol / simulate_ls_protocol, 1e-9 mV
    TR2 linearity          passive: v(2A) - v0 == 2 (v(A) - v0), 1e-8 mV
    TR3 baseline           samples before the onset == v_init: passive 1e-9 mV,
                           I_h cell at its measured rest 5e-3 mV
    TR4 steady state       passive: end-of-step deflection == R_in A, 1e-3 rel,
                           at dt 0.025 and 0.05 ms
    TR5 I_h                sag and rebound > 1e-3 mV with I_h; none (1e-9 mV)
                           for the passive cell
    TR6 contract           shapes, float64, finite; i_pA on exactly the stated
                           samples, legacy and real geometry
    TR7 no state           identical repeat calls; a call after another
                           amplitude == a fresh cell's call, 1e-12 mV
    TR8 refusals           every ValueError of the entry's "Refused"
    TR9 bytes and legacy   this file pure ASCII and LF; git diff against the
                           contract commit removes or changes no line above
                           the banner

Needs NEURON and the compiled mechanisms (./x86_64, `nrnivmodl mod`; pass
--build to compile them when x86_64/special is missing). TR9's second half
needs git and the contract commit.

Run (from this folder)
    python smoke_twin_response.py [--build]
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.

Debug
    A failed check_close() saves both arrays to
        <tmp>/smoke_twin_response_artifacts/<check name>.npz
    (outside the repository; the path is printed at the end of the run).

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                     # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                                                  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal       # noqa: E402,F401

import synthetic_ground_truth as sgt                                # noqa: E402
from twin_config import TwinConfig                                  # noqa: E402

SEED = 12345                       # only the legacy protocol functions take an rng
REPORT_PACKAGES = ("numpy", "scipy", "neuron")
ARTIFACT_DIR = Path(tempfile.gettempdir()) / "smoke_twin_response_artifacts"
_SAVED_ARTIFACTS = []

CONTRACT_COMMIT = "f54fdde"        # the wave-1 contract (stubs of Block 11)
BANNER = "#  The twins' clean responses (Block 11"
DT_MS = TwinConfig().neuron_dt_ms  # 0.025 ms, the generator's step

# The real archive's geometry (SPEC section 3, 508282493, data inspected):
# SS window 10500 samples at 50 kHz from -10 ms; LS sweep 363500 samples,
# onset 0.27 s, duration 1.0 s, current on samples 13500 .. 63499.
REAL_FS = 50000.0
REAL_SS_T = (np.arange(10500) - 500) / REAL_FS
REAL_LS = dict(onset_s=0.27, duration_s=1.0, n_samples=363500, fs_Hz=REAL_FS)
# The legacy geometry (ProtocolConfig defaults): onset 100 ms, step 1000 ms,
# 200 ms after, 20 kHz -> 26000 samples.
LEGACY_LS = dict(onset_s=0.1, duration_s=1.0, n_samples=26000, fs_Hz=20000.0)

_TMP = Path(tempfile.mkdtemp(prefix="smoke_twin_response_"))
_SWC = sgt.write_ball_and_stick_swc(_TMP / "bas.swc")


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """assert_allclose that saves both arrays (outside the repo) on failure."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def gt_for(kind: str) -> sgt.GroundTruthParams:
    """The two test cells: passive, and the same cell with uniform human I_h
    (the configuration of smoke_ih_fit's synthetic I_h archive). Spine factor
    F = 1.9 and the I_h regions soma, dend, apic are the defaults."""
    ih = None
    if kind == "ih":
        ih = sgt.IhConfig(gIhbar_S_cm2=1.2e-4, ehcn_mV=-49.85,
                          distribution="uniform", mechanism="Ih_human")
    elif kind != "passive":
        raise ValueError(kind)
    return sgt.GroundTruthParams(cm_uF_cm2=0.9, rm_Ohm_cm2=30000.0,
                                 ra_Ohm_cm=200.0, e_pas_mV=-78.0, ih=ih)


def make_cell(kind: str, measure: bool = True):
    """(cell, gt, v_init, rin_MOhm). Only one cell lives in NEURON at a time
    (Import3d creates top-level sections), so every build clears the last.
    v_init: e_pas for the passive cell, the measured rest for the I_h cell."""
    sgt._clear_neuron_sections()
    gt = gt_for(kind)
    cell = sgt.build_synthetic_cell(_SWC, gt)
    if not measure:
        return cell, gt, None, None
    v_rest, rin, _tau, _sag = sgt.measure_rin_tau_sag(cell, gt)
    v0 = gt.e_pas_mV if kind == "passive" else v_rest
    return cell, gt, float(v0), float(rin)


def ss(cell, amp, v0, t_s=REAL_SS_T, dur=0.0005, dt=DT_MS):
    return sgt.simulate_ss_clean(cell, amplitude_pA=amp, duration_s=dur,
                                 t_s=t_s, v_init_mV=v0, dt_ms=dt)


def ls(cell, amp, v0, geom=LEGACY_LS, dt=DT_MS):
    return sgt.simulate_ls_clean(cell, amplitude_pA=amp, v_init_mV=v0,
                                 dt_ms=dt, **geom)


def ls_samples(geom):
    j_on = int(round(geom["onset_s"] * geom["fs_Hz"]))
    n_on = int(round(geom["duration_s"] * geom["fs_Hz"]))
    return j_on, n_on


# ---------------------------------------------------------------- checks ---

def test_TR1_legacy_parity():
    # Same NEURON run (same dt, delay, duration, v_init; the run's steps up to
    # the last sample do not depend on tstop) and the same numpy.interp; the
    # sample times differ by float roundoff only (<= 1e-12 ms), the trace's
    # slope is at most tens of mV/ms -> 1e-9 mV is ample.
    notes = []
    for kind in ("passive", "ih"):
        cell, gt, v0, _ = make_cell(kind)
        proto = sgt.ProtocolConfig(ss_n_repeats=1, ls_n_repeats=1)
        noise = sgt.NoiseConfig(sigma_mV=0.0, seed=0)
        bundles, _ind = sgt.simulate_ss_protocol(cell, gt, proto, noise,
                                                 make_rng(), v_init_mV=v0)
        worst = 0.0
        for b in bundles:
            v, i = ss(cell, b.amplitude_pA, v0, t_s=b.t, dur=b.stim_duration_s)
            check_close(f"TR1_ss_{kind}_{b.polarity}", v, b.v_mV, rtol=0.0, atol=1e-9)
            assert_array_equal(i, b.i_pA, err_msg=f"TR1 SS i_pA {kind} {b.polarity}")
            worst = max(worst, float(np.max(np.abs(v - b.v_mV))))
        amps = (-90.0, 50.0)
        lbs = sgt.simulate_ls_protocol(cell, gt, proto, noise, make_rng(), amps,
                                       "mixed", v_init_mV=v0)
        worst_ls = 0.0
        for b in lbs:
            geom = dict(onset_s=proto.ls_onset_ms * 1e-3,
                        duration_s=proto.ls_duration_ms * 1e-3,
                        n_samples=len(b.t), fs_Hz=proto.ls_sampling_rate_Hz)
            v, _i = ls(cell, b.amplitude_pA, v0, geom=geom)
            check_close(f"TR1_ls_{kind}_{b.amplitude_pA:g}", v, b.v_mV, rtol=0.0, atol=1e-9)
            worst_ls = max(worst_ls, float(np.max(np.abs(v - b.v_mV))))
        notes.append(f"{kind}: SS max|dv| {worst:.1e} mV, LS {worst_ls:.1e} mV "
                     f"(n_LS {len(lbs[0].t)})")
    print("    TR1 " + "; ".join(notes))


def test_TR2_linearity():
    # Passive cable + IClamp under NEURON's fixed-step implicit scheme is an
    # affine system in the amplitude: v(2A) - v0 == 2 (v(A) - v0) to roundoff
    # on mV-sized values (~1e-13); 1e-8 mV as the entry states.
    cell, _gt, v0, _ = make_cell("passive", measure=False)
    v0 = _gt.e_pas_mV
    a1, _ = ss(cell, 100.0, v0)
    a2, _ = ss(cell, 200.0, v0)
    check_close("TR2_ss", a2 - v0, 2.0 * (a1 - v0), rtol=0.0, atol=1e-8)
    b1, _ = ls(cell, -40.0, v0)
    b2, _ = ls(cell, -80.0, v0)
    check_close("TR2_ls", b2 - v0, 2.0 * (b1 - v0), rtol=0.0, atol=1e-8)
    print("    TR2 SS max|v(2A)-v0-2(v(A)-v0)| %.1e mV, LS %.1e mV"
          % (np.max(np.abs(a2 - v0 - 2 * (a1 - v0))),
             np.max(np.abs(b2 - v0 - 2 * (b1 - v0)))))


def test_TR3_baseline():
    # Passive, v_init = e_pas: the uniform resting state is exact -> 1e-9 mV.
    # I_h cell at its measured rest: 5e-3 mV, a sixth of the smallest sigma_f
    # of the nf2 run (0.031 mV), as the entry states.
    out = []
    for kind, tol in (("passive", 1e-9), ("ih", 5e-3)):
        cell, _gt, v0, _ = make_cell(kind)
        v_ss, _ = ss(cell, -200.0, v0)
        pre_ss = v_ss[REAL_SS_T < 0.0]
        v_ls, _ = ls(cell, -110.0, v0, geom=REAL_LS)
        j_on, _n_on = ls_samples(REAL_LS)
        pre_ls = v_ls[:j_on]
        d_ss = float(np.max(np.abs(pre_ss - v0)))
        d_ls = float(np.max(np.abs(pre_ls - v0)))
        out.append((kind, tol, d_ss, d_ls, v0))
    msg = "; ".join(f"{k}: max|v - v_init| before onset SS {a:.2e} mV, LS {b:.2e} mV "
                    f"(tol {t:g}, v_init {v:.4f})" for k, t, a, b, v in out)
    print("    TR3 " + msg)
    for kind, tol, d_ss, d_ls, _v in out:
        assert d_ss <= tol and d_ls <= tol, "TR3 " + msg


def test_TR4_steady_state():
    # Passive: after 1 s (> 35 tau_m of 27 ms) the step response has settled;
    # the deflection equals R_in A (measure_rin_tau_sag, -30 pA probe, linear
    # cell) to 1e-3 relative, at two NEURON steps.
    cell, _gt, v0, rin = make_cell("passive")
    amp = -50.0
    j_on, n_on = ls_samples(LEGACY_LS)
    ev = []
    for dt in (0.025, 0.05):
        v, _ = ls(cell, amp, v0, dt=dt)
        defl = float(v[j_on + n_on - 1] - v0)
        expected = rin * amp * 1e-3          # MOhm * pA = uV -> mV
        rel = abs(defl - expected) / abs(expected)
        ev.append((dt, defl, expected, rel))
    print("    TR4 R_in %.4f MOhm; " % rin + "; ".join(
        f"dt {d:g} ms: deflection {a:.6f} mV vs R_in A {b:.6f} mV (rel {r:.1e})"
        for d, a, b, r in ev))
    for _d, _a, _b, rel in ev:
        assert rel <= 1e-3, f"TR4 relative error {rel:.3e} > 1e-3"


def test_TR5_ih_sag_rebound():
    j_on, n_on = ls_samples(REAL_LS)
    res = {}
    for kind in ("ih", "passive"):
        cell, _gt, v0, _ = make_cell(kind)
        v, _ = ls(cell, -110.0, v0, geom=REAL_LS)
        step = v[j_on:j_on + n_on]
        after = v[j_on + n_on:]
        sag = float(step[-1] - step.min())       # > 0: the trough lies below the end
        rebound = float(after.max() - v0)        # > 0: overshoot above rest
        res[kind] = (sag, rebound)
    print("    TR5 sag (v_end - v_min in step) / rebound (max after - v_init): "
          "I_h %.4f / %.4f mV; passive %.2e / %.2e mV"
          % (res["ih"] + res["passive"]))
    assert res["ih"][0] > 1e-3, f"TR5 no sag with I_h: {res['ih'][0]}"
    assert res["ih"][1] > 1e-3, f"TR5 no rebound with I_h: {res['ih'][1]}"
    # passive: monotone approach and return (roundoff only)
    assert res["passive"][0] <= 1e-9, f"TR5 passive sag {res['passive'][0]}"
    assert res["passive"][1] <= 1e-9, f"TR5 passive rebound {res['passive'][1]}"


def test_TR6_contract():
    cell, _gt, v0, _ = make_cell("passive", measure=False)
    v0 = _gt.e_pas_mV
    ev = []
    # SS on the real window axis (10500 samples from -10 ms at 50 kHz)
    for amp in (-200.0, 200.0):
        v, i = ss(cell, amp, v0)
        for name, a in (("v", v), ("i", i)):
            assert a.shape == (10500,), f"TR6 SS {name} shape {a.shape}"
            assert a.dtype == np.float64, f"TR6 SS {name} dtype {a.dtype}"
            assert np.all(np.isfinite(a)), f"TR6 SS {name} not finite"
        expected_on = np.nonzero((REAL_SS_T >= 0.0) & (REAL_SS_T <= 0.0005))[0]
        on = np.nonzero(i)[0]
        assert_array_equal(on, expected_on, err_msg="TR6 SS current samples")
        # on this axis 0.0005 s is a sample (j = 525): the legacy closed rule
        # puts the current on samples 500 .. 525
        assert_array_equal(on, np.arange(500, 526), err_msg="TR6 SS 500..525")
        assert np.all(i[on] == amp) and np.all(i[np.setdiff1d(np.arange(10500), on)] == 0.0)
    ev.append(f"SS (10500,) current on {on[0]}..{on[-1]} ({on.size} samples)")
    # LS: legacy and real geometry
    for label, geom, first, last in (("legacy", LEGACY_LS, 2000, 21999),
                                     ("real", REAL_LS, 13500, 63499)):
        v, i = ls(cell, -70.0, v0, geom=geom)
        n = geom["n_samples"]
        for name, a in (("v", v), ("i", i)):
            assert a.shape == (n,), f"TR6 LS {label} {name} shape {a.shape}"
            assert a.dtype == np.float64, f"TR6 LS {label} {name} dtype {a.dtype}"
            assert np.all(np.isfinite(a)), f"TR6 LS {label} {name} not finite"
        on = np.nonzero(i)[0]
        assert_array_equal(on, np.arange(first, last + 1),
                           err_msg=f"TR6 LS {label} current samples")
        assert np.all(i[on] == -70.0)
        ev.append(f"LS {label} ({n},) current on {on[0]}..{on[-1]}")
    print("    TR6 " + "; ".join(ev))


def test_TR7_no_state():
    # I_h cell (gating state that a leak would carry between calls).
    cell, _gt, v0, _ = make_cell("ih")
    a1 = ls(cell, -70.0, v0)
    a2 = ls(cell, -70.0, v0)
    assert_array_equal(a1[0], a2[0], err_msg="TR7 LS repeat v")
    assert_array_equal(a1[1], a2[1], err_msg="TR7 LS repeat i")
    s1 = ss(cell, -200.0, v0)
    s2 = ss(cell, -200.0, v0)
    assert_array_equal(s1[0], s2[0], err_msg="TR7 SS repeat v")
    ls(cell, 50.0, v0)                   # another amplitude first
    after_ls = ls(cell, -30.0, v0)[0]
    ss(cell, 200.0, v0)
    after_ss = ss(cell, -200.0, v0)[0]
    cell2, _gt2, _, _ = make_cell("ih", measure=False)   # fresh cell, first calls
    fresh_ls = ls(cell2, -30.0, v0)[0]
    sgt._clear_neuron_sections()
    cell3, _gt3, _, _ = make_cell("ih", measure=False)
    fresh_ss = ss(cell3, -200.0, v0)[0]
    check_close("TR7_ls_fresh", after_ls, fresh_ls, rtol=0.0, atol=1e-12)
    check_close("TR7_ss_fresh", after_ss, fresh_ss, rtol=0.0, atol=1e-12)
    print("    TR7 repeats identical; after another amplitude vs fresh cell: "
          "LS max|dv| %.1e mV, SS %.1e mV"
          % (np.max(np.abs(after_ls - fresh_ls)), np.max(np.abs(after_ss - fresh_ss))))


def test_TR8_refusals():
    cell, _gt, _v0, _ = make_cell("passive", measure=False)
    v0 = _gt.e_pas_mV
    good_ss = dict(amplitude_pA=-200.0, duration_s=0.0005, t_s=REAL_SS_T,
                   v_init_mV=v0, dt_ms=DT_MS)
    good_ls = dict(amplitude_pA=-70.0, onset_s=0.27, duration_s=1.0,
                   n_samples=63500, fs_Hz=REAL_FS, v_init_mV=v0, dt_ms=DT_MS)
    bad_axis = REAL_SS_T.copy()
    bad_axis[3000:] += 2e-5 * 2e-6           # one step longer by 2e-6 relative
    nan_axis = REAL_SS_T.copy(); nan_axis[100] = np.nan
    ss_cases = [
        ("t_s one sample", dict(t_s=np.array([-0.01]))),
        ("t_s empty", dict(t_s=np.array([]))),
        ("t_s 2-D", dict(t_s=REAL_SS_T.reshape(2, -1))),
        ("t_s not uniform", dict(t_s=bad_axis)),
        ("t_s decreasing", dict(t_s=REAL_SS_T[::-1].copy() - 1.0)),
        ("t_s[0] == 0", dict(t_s=REAL_SS_T - REAL_SS_T[0])),
        ("t_s[0] > 0", dict(t_s=REAL_SS_T + 1.0)),
        ("t_s NaN", dict(t_s=nan_axis)),
        ("duration 0", dict(duration_s=0.0)),
        ("duration < 0", dict(duration_s=-0.0005)),
        ("dt 0", dict(dt_ms=0.0)),
        ("dt < 0", dict(dt_ms=-0.025)),
        ("amplitude NaN", dict(amplitude_pA=np.nan)),
        ("amplitude inf", dict(amplitude_pA=np.inf)),
        ("duration NaN", dict(duration_s=np.nan)),
        ("v_init NaN", dict(v_init_mV=np.nan)),
        ("dt NaN", dict(dt_ms=np.nan)),
    ]
    ls_cases = [
        ("n_samples 1", dict(n_samples=1)),
        ("n_samples 0", dict(n_samples=0)),
        ("n_samples not integer", dict(n_samples=63500.5)),
        ("fs 0", dict(fs_Hz=0.0)),
        ("fs < 0", dict(fs_Hz=-REAL_FS)),
        ("duration 0", dict(duration_s=0.0)),
        ("dt 0", dict(dt_ms=0.0)),
        ("onset < 0", dict(onset_s=-1e-3)),
        ("step past the end", dict(n_samples=63499)),
        ("amplitude NaN", dict(amplitude_pA=np.nan)),
        ("onset inf", dict(onset_s=np.inf)),
        ("duration NaN", dict(duration_s=np.nan)),
        ("fs inf", dict(fs_Hz=np.inf)),
        ("v_init NaN", dict(v_init_mV=np.nan)),
        ("dt NaN", dict(dt_ms=np.nan)),
    ]
    missed = []
    for fn, good, cases, tag in ((sgt.simulate_ss_clean, good_ss, ss_cases, "SS"),
                                 (sgt.simulate_ls_clean, good_ls, ls_cases, "LS")):
        for name, over in cases:
            kw = dict(good); kw.update(over)
            try:
                fn(cell, **kw)
            except ValueError:
                continue
            except Exception as exc:          # wrong exception type
                missed.append(f"{tag} {name}: {type(exc).__name__}")
                continue
            missed.append(f"{tag} {name}: accepted")
    assert not missed, "TR8 not refused: " + "; ".join(missed)
    # boundaries accepted: spread 5e-7 (< 1e-6), and the step ending on the
    # last sample (13500 + 50000 == 63500 samples)
    ok_axis = REAL_SS_T.copy(); ok_axis[3000:] += 2e-5 * 5e-7
    v, _ = sgt.simulate_ss_clean(cell, **dict(good_ss, t_s=ok_axis))
    v2, i2 = sgt.simulate_ls_clean(cell, **good_ls)
    assert v.shape == (10500,) and v2.shape == (63500,) and i2[-1] == -70.0
    print(f"    TR8 {len(ss_cases)} SS and {len(ls_cases)} LS refusals raised ValueError; "
          "boundaries (spread 5e-7, step ending on the last sample) accepted")


def test_TR9_bytes_and_legacy():
    raw = Path(__file__).read_bytes()
    assert all(b < 128 for b in raw), "TR9 smoke_twin_response.py is not pure ASCII"
    assert b"\r" not in raw, "TR9 smoke_twin_response.py holds CR"
    src_now = (HERE / "synthetic_ground_truth.py").read_bytes()
    assert b"\r" not in src_now, "TR9 synthetic_ground_truth.py holds CR"
    try:
        old = subprocess.run(["git", "show", f"{CONTRACT_COMMIT}:./synthetic_ground_truth.py"],
                             cwd=str(HERE), capture_output=True, check=True).stdout
        diff = subprocess.run(["git", "diff", "-U0", CONTRACT_COMMIT, "--",
                               "synthetic_ground_truth.py"],
                              cwd=str(HERE), capture_output=True, check=True,
                              text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest(f"TR9 legacy half: git or commit {CONTRACT_COMMIT} "
                                f"not available ({exc}); ASCII/LF half passed")
    old_lines = old.decode("ascii", errors="replace").split("\n")
    now_lines = src_now.decode("ascii", errors="replace").split("\n")
    idx = [k for k, ln in enumerate(old_lines) if ln.startswith(BANNER)]
    assert len(idx) == 1, f"TR9 banner found {len(idx)} times in {CONTRACT_COMMIT}"
    first_new = idx[0] - 1               # the "# ====" line opening the banner
    assert now_lines[:first_new] == old_lines[:first_new], \
        "TR9 a line above the banner differs from the contract commit"
    # every removed or changed line of the diff lies below the banner
    removed = added = 0
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, re.M):
        a0, na = int(m.group(1)), int(m.group(2) or 1)
        nb = int(m.group(4) or 1)
        removed += na; added += nb
        if na > 0:
            assert a0 > first_new + 1, \
                f"TR9 lines {a0}..{a0 + na - 1} of {CONTRACT_COMMIT} removed above the banner"
    # the additions below the banner are pure ASCII
    tail = "\n".join(now_lines[first_new:])
    assert all(ord(ch) < 128 for ch in tail), "TR9 non-ASCII below the banner"
    print(f"    TR9 ASCII/LF; {first_new} lines above the banner identical to "
          f"{CONTRACT_COMMIT}; diff: {removed} lines removed, {added} added, all below "
          f"line {first_new + 1}")


# ---------------------------------------------------------------- runner ---

def ensure_build(build: bool) -> None:
    special = HERE / "x86_64" / "special"
    if special.exists():
        return
    if not build:
        sys.exit("x86_64/special not found; run 'nrnivmodl mod' here or pass --build")
    r = subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0 or not special.exists():
        sys.exit("nrnivmodl failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"dt {DT_MS} ms")
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true",
                    help="run nrnivmodl mod if x86_64/special is missing")
    args = ap.parse_args(argv)
    ensure_build(args.build)

    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    t_all = time.perf_counter()
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
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    sgt._clear_neuron_sections()

    print()
    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline[:160]}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items())
          + f" ({time.perf_counter() - t_all:.1f} s)")
    print(f"smoke_twin_response: {counts['PASS']}/{len(results)} passed")
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
