"""Seed loop over C3/C4 of check_tps_scaling.py (same truth, N, tau): RMS vs truth
of raw / range-normalised / CV-scaled (option B) fits, and the CV error that chose
them. Run: python3 check_tps_scaling_seeds.py"""
import numpy as np
import check_tps_scaling as C
rows = []
for seed in (1, 2, 3, 4):
    rng = np.random.default_rng(seed)
    X = C.design(rng, C.CFG["n"]); y = C.truth(*X.T) + C.CFG["tau"] * rng.standard_normal(C.CFG["n"])
    Xt = C.design(rng, C.CFG["n_test"]); yt = C.truth(*Xt.T)
    lo = X.min(0); span = X.max(0) - lo; Xn = (X - lo) / span; Xtn = (Xt - lo) / span
    g, f, sd = C.CFG["smoothing_grid"], C.CFG["folds"], seed
    s_raw, cv_raw = C.choose_smoothing(X, y, g, f, sd); rms_raw = np.sqrt(np.mean((C.fit(X, y, s_raw)(Xt) - yt) ** 2))
    s_n, cv_n = C.choose_smoothing(Xn, y, g, f, sd); rms_n = np.sqrt(np.mean((C.fit(Xn, y, s_n)(Xtn) - yt) ** 2))
    best = None
    for a in C.CFG["scale_grid"]:
        for b in C.CFG["scale_grid"]:
            S = np.diag([1.0, a, b]); s_b, cv_b = C.choose_smoothing(Xn @ S, y, g, f, sd)
            if best is None or cv_b < best[0]: best = (cv_b, a, b, s_b)
    cv_b, a, b, s_b = best; S = np.diag([1.0, a, b])
    rms_b = np.sqrt(np.mean((C.fit(Xn @ S, y, s_b)(Xtn @ S) - yt) ** 2))
    print("seed %d | raw: s=%g cv=%.5f rms=%.4f | norm: s=%g cv=%.5f rms=%.4f | B: scales=(%g,%g) s=%g cv=%.5f rms=%.4f"
          % (seed, s_raw, cv_raw, rms_raw, s_n, cv_n, rms_n, a, b, s_b, cv_b, rms_b))
