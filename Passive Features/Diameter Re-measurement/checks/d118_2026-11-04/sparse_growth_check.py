"""Check by computation (D-119, 2026-11-04): Block 10's growth fit with the
sparse Jacobian (least_squares tr_solver="lsmr", lsq_linear lsq_solver="lsmr")
against the dense fit (tr_solver="exact"), on synthetic near scans, and the
time per fit at the pool's sizes.

Synthetic scans (truth chosen): N nodes, 7 planes each (k* - 3 .. k* + 3),
sub-plane depth uniform in +- dz/2, omega = c_i + G(z_ax,i - z_k) + noise with
G(delta) = 0.3 delta^2 (um^2), c_i uniform in [0.02, 0.06] um^2, noise SD 0.003
um^2; 10 % of the planes set to NaN. Compared, per fit: max |dG| over the knots
(um^2), |d anchor| (um, anchor_value at delta_0), max |dc|, max |dz_ax|, and
the relative cost difference. Run from the workstream folder:
    python3 checks/d118_2026-11-04/sparse_growth_check.py
"""
import dataclasses
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

CFG = dict(seed=20261104, compare_N=(10, 30, 100, 300), compare_reps=3, time_N=(300, 1000, 1530),
           noise_um2=0.003, nan_share=0.1)


def scans(rng, n, cfg, dz):
    out = []
    for i in range(n):
        ks = np.arange(-3, 4)
        zsub = rng.uniform(-0.5, 0.5) * dz
        zk = ks * dz
        om = rng.uniform(0.02, 0.06) + 0.3 * (zsub - zk) ** 2 + rng.normal(0, cfg["noise_um2"], ks.size)
        om[rng.random(ks.size) < cfg["nan_share"]] = np.nan
        m = ks.size
        out.append(CAL.PlaneScan(node_id=i, d_hat_um=0.4, phi_rad=0.0, alpha_hat=0.2, k_star=0, z_sub_um=zsub, ks=ks,
                                 z_um=zk, omega_um2=om, vw_um2=om.copy(), area_gl_um=np.full(m, 5.0),
                                 depth_gl=np.full(m, 20.0), B_bar=np.full(m, 200.0),
                                 status=tuple("ok" for _ in range(m)), set="near"))
    return out


def with_jac(cfg0, kind, **cal):
    return dataclasses.replace(cfg0, calibration=dataclasses.replace(cfg0.calibration, growth_jacobian=kind, **cal))


def main(cfg=CFG):
    rng = np.random.default_rng(cfg["seed"])
    base = default_config()
    dz = base.acquisition.dz_um
    print("== sparse vs dense, the same scans (seed %d)" % cfg["seed"])
    worst = dict(G=0.0, anchor=0.0, c=0.0, z=0.0, cost=0.0)
    for variant in (dict(), dict(growth_interp="linear"), dict(origin_convention="min_origin")):
        for n in cfg["compare_N"]:
            for _ in range(cfg["compare_reps"]):
                sc = scans(rng, n, cfg, dz)
                gd = CAL.fit_growth(sc, with_jac(base, "dense", **variant))
                gs = CAL.fit_growth(sc, with_jac(base, "sparse", **variant))
                fin = np.isfinite(gd.g_um2)
                assert np.array_equal(fin, np.isfinite(gs.g_um2))
                d = dict(G=float(np.max(np.abs(gd.g_um2[fin] - gs.g_um2[fin]))),
                         c=float(np.max(np.abs(gd.c_um2 - gs.c_um2))), z=float(np.max(np.abs(gd.z_ax_um - gs.z_ax_um))),
                         cost=abs(gd.cost - gs.cost) / max(gd.cost, 1e-300))
                if variant.get("origin_convention", "symmetric") == "symmetric":
                    cd = with_jac(base, "dense", **variant)
                    d["anchor"] = abs(CAL.anchor_value(gd, cd) - CAL.anchor_value(gs, cd))
                for k, v in d.items():
                    worst[k] = max(worst[k], v)
                print("  %-28s N = %4d: max|dG| %.2e um2, |d anchor| %s um, max|dc| %.2e, max|dz| %.2e um, "
                      "rel d cost %.2e; nfev %d / %d" % (variant or "cubic, symmetric", n, d["G"],
                      "%.2e" % d["anchor"] if "anchor" in d else "  n/a  ", d["c"], d["z"], d["cost"], gd.nfev, gs.nfev))
    print("  worst: " + ", ".join("%s %.2e" % kv for kv in worst.items()))
    print("== time per fit (cubic, symmetric)")
    for n in cfg["time_N"]:
        sc = scans(rng, n, cfg, dz)
        t0 = time.perf_counter()
        CAL.fit_growth(sc, with_jac(base, "sparse"))
        ts = time.perf_counter() - t0
        line = "  N = %4d: sparse %.2f s (x 1000 resamples ~ %.2f h)" % (n, ts, ts * 1000 / 3600)
        if n <= 300:
            t0 = time.perf_counter()
            CAL.fit_growth(sc, with_jac(base, "dense"))
            line += "; dense %.2f s" % (time.perf_counter() - t0)
        print(line, flush=True)


if __name__ == "__main__":
    main()
