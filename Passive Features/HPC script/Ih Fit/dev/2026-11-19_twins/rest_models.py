"""Standalone check (not pipeline code), 2026-11-19: three ways to start an I_h twin's response.
 U  uniform e_pas, started by finitialize at the measured soma rest (the legacy generator; Block 11 as first written)
 S  uniform e_pas, started from the cell's true settled state (a 6 s unstimulated lead before the stimulus)
 B  e_pas balanced per segment so that the measured soma rest is a uniform equilibrium (the fitter's model, D-005.2)
Reports max |difference| of the baseline-subtracted responses (baseline = mean of the 10 ms before onset for SS,
of the 270 ms before onset for LS) on the windows the fit reads: SS 0.5-100 ms after onset, LS the 1 s step."""
import sys, tempfile, pathlib
import numpy as np
sys.path.insert(0, ".")
from neuron import h
import synthetic_ground_truth as sgt

def build(balanced_at=None):
    for s in list(h.allsec()): h.delete_section(sec=s)
    p = sgt.write_ball_and_stick_swc(pathlib.Path(tempfile.mkdtemp()) / "bs.swc")
    gt = sgt.GroundTruthParams(cm_uF_cm2=0.9, rm_Ohm_cm2=30000.0, ra_Ohm_cm=200.0, e_pas_mV=-78.0,
                               ih=sgt.IhConfig(gIhbar_S_cm2=1.2e-4, ehcn_mV=-49.85, mechanism="Ih_human"))
    cell = sgt.build_synthetic_cell(p, gt)
    if balanced_at is not None:
        h.finitialize(balanced_at)
        for sec in h.allsec():
            for seg in sec:
                i_h = getattr(seg, "ihcn_Ih_human", 0.0) if h.ismembrane("Ih_human", sec=sec) else 0.0   # mA/cm2
                seg.e_pas = balanced_at + i_h / seg.g_pas   # mA/cm2 over S/cm2 = mV
    return cell, gt

def run(cell, amp, delay, dur, tail, v0, lead=0.0):
    t, v = cell.simulate(stim_amp_pA=amp, stim_delay_ms=lead + delay, stim_dur_ms=dur, tstop_ms=lead + delay + dur + tail,
                         v_init_mV=v0, dt_ms=0.025)
    grid = np.arange(0.0, delay + dur + tail, 0.025)        # common output axis (ms): no one-sample misalignment
    return grid, np.interp(grid + lead, t, v)

cell, gt = build(); v_rest = sgt.measure_rin_tau_sag(cell, gt)[0]
print("measured soma rest %.4f mV" % v_rest)
LEAD = 6000.0
out = {}
for name in ("U", "S", "B"):
    cell, gt = build(balanced_at=v_rest if name == "B" else None)
    lead = LEAD if name == "S" else 0.0
    ts, vs = run(cell, -200.0, 10.0, 0.5, 200.0, v_rest, lead)
    tl, vl = run(cell, -50.0, 270.0, 1000.0, 300.0, v_rest, lead)
    n = min(len(ts), 8400); out[name] = (ts[:n], vs[:n], tl[:62000], vl[:62000])
    pre_s = vs[ts < 10.0]; pre_l = vl[tl < 270.0]
    print("%s: SS pre-onset max|v - v[0]| %.2e mV, drift across the 10 ms %.2e; LS pre-window drift %.2e mV; v at t=0 %.4f"
          % (name, np.max(np.abs(pre_s - pre_s[0])), abs(pre_s[-1] - pre_s[0]), np.max(np.abs(pre_l - pre_l[0])), vs[0]))
def bsub(t, v, t0, t1): return v - v[(t >= t0) & (t < t1)].mean()
for a, b in (("U", "B"), ("S", "B"), ("U", "S")):
    ta, sa, la_t, la = out[a]; tb, sb, lb_t, lb = out[b]
    ds = bsub(ta, sa, 0.0, 10.0) - bsub(tb, sb, 0.0, 10.0); w = (ta >= 10.5) & (ta <= 110.0)
    dl = bsub(la_t, la, 0.0, 270.0) - bsub(lb_t, lb, 0.0, 270.0); wl = (la_t >= 270.0) & (la_t <= 1270.0)
    print("%s vs %s: SS window max|diff| %.4f mV (RMS %.4f); LS step max|diff| %.4f mV (RMS %.4f)"
          % (a, b, np.max(np.abs(ds[w])), np.sqrt(np.mean(ds[w] ** 2)), np.max(np.abs(dl[wl])), np.sqrt(np.mean(dl[wl] ** 2))))
ta, sa, la_t, la = out["B"]
print("scale: SS peak deflection %.3f mV; LS deflection at step end %.3f mV" % (np.max(np.abs(bsub(ta, sa, 0.0, 10.0))), abs(bsub(la_t, la, 0.0, 270.0)[(la_t >= 1260) & (la_t < 1270)].mean())))
