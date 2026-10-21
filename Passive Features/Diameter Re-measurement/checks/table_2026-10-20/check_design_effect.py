"""Check by computation (explanation-modes v12): the design effect of n_real realizations per draw.

Claim: with y_{g,r} = b + u_g + e_{g,r}, u_g ~ N(0, s_u^2) i.i.d. over draws, e_{g,r} ~ N(0, s_e^2) i.i.d.,
tau^2 = s_u^2 + s_e^2 and rho = s_u^2 / tau^2, the mean of all G*n rows has
    Var(ybar) = tau^2 / (G n) * [1 + (n - 1) rho]   (exact for this model).
Test: Monte Carlo variance of ybar over R replications against the formula; also the naive
"independent rows" value tau^2/(G n), which the formula says is too small by the bracket.
Tolerance: the MC variance has relative SE sqrt(2/(R-1)) = 0.020 at R = 5000, so agreement within 3 SE (6 %).
"""
import numpy as np

CFG = dict(G=2000, n=10, s_u=0.1, s_e=0.2, R=5000, seed=20261020)

def run(cfg):
    rng = np.random.default_rng(cfg["seed"])
    G, n, s_u, s_e, R = cfg["G"], cfg["n"], cfg["s_u"], cfg["s_e"], cfg["R"]
    tau2 = s_u ** 2 + s_e ** 2
    rho = s_u ** 2 / tau2
    means = np.empty(R)
    for i in range(R):
        u = rng.normal(0.0, s_u, size=(G, 1))
        e = rng.normal(0.0, s_e, size=(G, n))
        means[i] = (u + e).mean()
    v_mc = means.var(ddof=1)
    v_formula = tau2 / (G * n) * (1.0 + (n - 1) * rho)
    v_naive = tau2 / (G * n)
    rel_se = np.sqrt(2.0 / (R - 1))
    print("config:", cfg)
    print("rho = %.4f  design effect 1+(n-1)rho = %.4f" % (rho, 1 + (n - 1) * rho))
    print("Var(ybar): MC %.4e | formula %.4e | naive (independent rows) %.4e" % (v_mc, v_formula, v_naive))
    print("MC / formula = %.4f (MC relative SE %.3f);  MC / naive = %.3f" % (v_mc / v_formula, rel_se, v_mc / v_naive))
    ok = abs(v_mc / v_formula - 1.0) < 3 * rel_se
    print("PASS" if ok else "FAIL", "(within 3 relative SE of the formula)")
    return ok

if __name__ == "__main__":
    run(CFG)
