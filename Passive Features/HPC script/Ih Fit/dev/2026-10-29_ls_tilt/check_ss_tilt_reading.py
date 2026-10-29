"""check_ss_tilt_reading.py -- standalone check (not pipeline code), 2026-10-29:
how precisely an SS sweep's realised tilt is read (D-102.2) once D-103's
per-sweep, per-polarity median subtraction and its later window starts apply.

Reading (D-102.2): per SS sweep s, the ordinary least-squares slope of the
window means <x~_k> over [t_SS, 200) ms on the windows' centre times, with one
intercept per polarity. Variants of the subtraction compared on the same
simulated windows: none; the per-polarity mean window over the cell (D-051.1,
what the tilt study's check03 (f) used); the per-sweep, per-polarity median
window (D-103.1), implemented here with numpy.median (Block 3 does not have it
yet). Starts: 80 ms (the floor and the fallback), 130 ms (D-103's start of the
RC test cell at kappa_tr 0.1), 150 ms (the latest start before D-104's
fallback; a 50 ms window).

Truth (chosen): theta_cf, sigma_b = 0.1 mV/s, the four reference layouts of the
tilt study (make_banks.py ref: the same layout draws), continuous sweeps from
the tilt study's simulator (tilt_common.simulate_windows, Block 2 underneath);
the realised tilt of sweep s is b_s = sigma_b xi_tilt[s].

Predictions, fixed before running:
 P1  the three subtraction variants give the same slopes to 1e-9 mV/s: each
     subtracted window shifts every window of its group by one constant, and
     a group is one polarity of one sweep, so the polarity intercept absorbs it;
 P2  at every start, the Monte Carlo RMS of b_hat - b lies within 3 SE of the
     exact value sqrt(mean_s [(X'X)^-1 X' Sigma X (X'X)^-1]_00), Sigma the
     covariance of the sweep's window means under the law (noise_law's
     var_window_mean and window_mean_cov), SE = RMS / sqrt(2 N);
 P3  at 80 ms the RMS is within 10 % of check03 (f)'s 0.0036, 0.0028, 0.0028,
     0.0027 mV/s for the four layouts.
Seed of the cells 20261040; numpy. Run, from this folder (about 5 min):
    python check_ss_tilt_reading.py ../2026-10-26_tilt_sigma_b/scripts
Pure ASCII, LF.
"""
import sys
import time
from pathlib import Path

import numpy as np

CFG = dict(bank_seed=20261026, cell_seed=20261040, sigma_b=0.1,
           layouts=(("2sw_K40", 2, 40, 2000), ("2sw_K63", 2, 63, 2000), ("6sw_K199", 6, 199, 400),
                    ("10sw_K348", 10, 348, 400)),
           t_ss_ms=(80.0, 130.0, 150.0), check03_rms=(0.0036, 0.0028, 0.0028, 0.0027), p1_tol=1e-9,
           n_se=3.0, p3_rel=0.10)


# ---------------------------------------------------------------- analysis
def segment(v, tc, t_ss_ms):
    j0 = tc.PRE_N + int(round(t_ss_ms * 1e-3 * tc.FS))
    return v[:, j0:tc.N_W], j0


def subtract(seg, lay, mode):
    if mode == "none":
        return seg
    out = seg.copy()
    dep = lay.pos % 2 == 0
    if mode == "polarity_mean":
        for p in (True, False):
            g = dep == p
            out[g] -= seg[g].mean(axis=0)
        return out
    for s in range(lay.n_sweeps):                                   # "group_median" (D-103.1)
        for p in (True, False):
            g = (lay.sweep == s) & (dep == p)
            if g.any():
                out[g] -= np.median(seg[g], axis=0)
    return out


def design(lay, s, centres):
    g = lay.sweep == s
    t = centres[g]
    dep = (lay.pos[g] % 2 == 0).astype(float)
    return g, np.column_stack([t - t.mean(), dep, 1.0 - dep])


def slopes(means, lay, centres):
    out = np.empty(lay.n_sweeps)
    for s in range(lay.n_sweeps):
        g, X = design(lay, s, centres)
        out[s] = np.linalg.lstsq(X, means[g], rcond=None)[0][0]
    return out


def exact_rms(tc, lay, j0):
    m = tc.N_W - j0
    centres = (lay.pos * tc.P + j0 + 0.5 * (m - 1)) * tc.DT
    v0 = tc.R.nl.var_window_mean(tc.LAW_CF, tc.DT, m)
    cov = {d: tc.R.nl.window_mean_cov(tc.LAW_CF, tc.DT, m, m, d * tc.P) for d in range(1, tc.TRAIN)}
    var = []
    for s in range(lay.n_sweeps):
        g, X = design(lay, s, centres)
        pos = lay.pos[g]
        dd = np.abs(pos[:, None] - pos[None, :])
        Sig = np.where(dd == 0, v0, 0.0)
        for d, c in cov.items():
            Sig = np.where(dd == d, c, Sig)
        A = np.linalg.solve(X.T @ X, X.T)
        var.append((A @ Sig @ A.T)[0, 0])
    return float(np.sqrt(np.mean(var))), centres


def main(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    import tilt_common as tc
    t0 = time.time()
    ss = np.random.SeedSequence([CFG["bank_seed"], 1]).spawn(len(CFG["layouts"]))
    crng = np.random.default_rng(CFG["cell_seed"])
    ok = dict(P1=True, P2=True, P3=True)
    for (name, S, K, n_cells), s_seq, ref03 in zip(CFG["layouts"], ss, CFG["check03_rms"]):
        lay = tc.build_layout(S, K, np.random.default_rng(s_seq), name)   # the reference layout of make_banks
        ex, cen = {}, {}
        for t in CFG["t_ss_ms"]:
            j0 = tc.PRE_N + int(round(t * 1e-3 * tc.FS))
            ex[t], cen[t] = exact_rms(tc, lay, j0)
        err = {t: [] for t in CFG["t_ss_ms"]}
        dmax = 0.0
        for c in range(n_cells):
            v, units = tc.simulate_windows(tc.LAW_CF, lay, crng, CFG["sigma_b"])
            b_true = CFG["sigma_b"] * units.tilt
            for t in CFG["t_ss_ms"]:
                seg, _ = segment(v, tc, t)
                sl = {mode: slopes(subtract(seg, lay, mode).mean(axis=1), lay, cen[t])
                      for mode in ("none", "polarity_mean", "group_median")}
                dmax = max(dmax, np.max(np.abs(sl["group_median"] - sl["none"])),
                           np.max(np.abs(sl["polarity_mean"] - sl["none"])))
                err[t].append(sl["group_median"] - b_true)
        print("== %s: %d sweeps, K %d, %d cells; windows per sweep %s"
              % (name, S, lay.K, n_cells, np.bincount(lay.sweep, minlength=S).tolist()))
        p1 = dmax < CFG["p1_tol"]
        ok["P1"] &= p1
        print("   P1 largest slope difference between the subtraction variants: %.1e mV/s %s"
              % (dmax, "PASS" if p1 else "FAIL"))
        for t in CFG["t_ss_ms"]:
            e = np.concatenate(err[t])
            rms = float(np.sqrt(np.mean(e ** 2)))
            se = ex[t] / np.sqrt(2.0 * e.size)
            p2 = abs(rms - ex[t]) < CFG["n_se"] * se
            ok["P2"] &= p2
            line = ("   t_SS %5.1f ms (window %5.1f ms): RMS(b_hat - b) %.5f mV/s (Monte Carlo, N %d, SE %.5f) against "
                    "exact %.5f -> P2 %s; %.1f %% of sigma_b; x%.3f of the 80 ms value"
                    % (t, 200.0 - t, rms, e.size, se, ex[t], "PASS" if p2 else "FAIL", 100 * rms / CFG["sigma_b"],
                       ex[t] / ex[80.0]))
            if t == 80.0:
                p3 = abs(rms / ref03 - 1) < CFG["p3_rel"]
                ok["P3"] &= p3
                line += "; check03 (f) %.4f -> P3 %s" % (ref03, "PASS" if p3 else "FAIL")
            print(line)
    print("\n" + "  ".join("%s %s" % (k, "PASS" if v else "FAIL") for k, v in ok.items())
          + "   (%.0f s)" % (time.time() - t0))
    return all(ok.values())


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "2026-10-26_tilt_sigma_b" / "scripts"
    raise SystemExit(0 if main(arg.resolve()) else 1)
