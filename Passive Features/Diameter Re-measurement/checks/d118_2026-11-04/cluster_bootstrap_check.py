"""Check by computation (D-118.3, 2026-11-04): how a node bootstrap, a cell
bootstrap (cells resampled, each with all its nodes) and a two-level
bootstrap (cells, then nodes inside each drawn cell) estimate the SE of an
average over nodes nested in cells.

Model (simulated, truth chosen): y_{c,j} = mu + b_c + e_{c,j}, c = 1..C cells,
j = 1..n nodes per cell, b_c ~ N(0, s_b^2) (the cell's own offset), e_{c,j} ~
N(0, s_w^2), all independent; ICC = s_b^2 / (s_b^2 + s_w^2). Estimator: the
grand mean of the C n values, whose true SE is
    SE_true = sqrt(s_b^2 / C + s_w^2 / (C n)).
For each (C, ICC): R datasets; in each, B resamples per scheme; reported:
the median over datasets of SE_hat / SE_true, and the coverage of
mean +- 1.96 SE_hat for mu (nominal 0.95). Added the same day: "larger" =
max(node SE, cell SE) per dataset, the rule offered for 2-9 cells.

This is a stand-in for the pooled kernel fit (an average of node-level
information over nodes grouped by cell); it tests the resampling schemes, not
the fit. Standalone; numpy only. Run:
    python3 cluster_bootstrap_check.py            (about a minute)
"""
import numpy as np

CFG = dict(seed=20261104, R=400, B=300, n=30, s_total=1.0,
           C_values=(2, 3, 5, 10, 20, 51), icc_values=(0.0, 0.05, 0.2))


def simulate(rng, C, n, icc, s_total):
    s_b, s_w = np.sqrt(icc) * s_total, np.sqrt(1.0 - icc) * s_total
    return rng.normal(0.0, s_b, (C, 1)) + rng.normal(0.0, s_w, (C, n))   # mu = 0


def se_node(rng, y, B):
    flat = y.ravel()
    idx = rng.integers(0, flat.size, (B, flat.size))
    return flat[idx].mean(axis=1).std(ddof=1)


def se_cell(rng, y, B):
    C = y.shape[0]
    pick = rng.integers(0, C, (B, C))
    return y.mean(axis=1)[pick].mean(axis=1).std(ddof=1)     # equal n: mean of drawn cells' means


def se_two_level(rng, y, B):
    C, n = y.shape
    pick = rng.integers(0, C, (B, C))
    inner = rng.integers(0, n, (B, C, n))
    vals = y[pick[:, :, None], inner]                          # B x C x n
    return vals.mean(axis=(1, 2)).std(ddof=1)


def main(cfg=CFG):
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n"]
    print("n = %d nodes per cell, R = %d datasets, B = %d resamples, seed %d" % (n, cfg["R"], cfg["B"], cfg["seed"]))
    print("%4s %5s | %22s | %22s | %22s | %22s" % ("C", "ICC", "node: SE/true, cov.", "cell: SE/true, cov.",
                                                   "two-level: SE/true, cov.", "larger: SE/true, cov."))
    for icc in cfg["icc_values"]:
        for C in cfg["C_values"]:
            s_b2, s_w2 = icc * cfg["s_total"] ** 2, (1 - icc) * cfg["s_total"] ** 2
            se_true = np.sqrt(s_b2 / C + s_w2 / (C * n))
            out = {k: ([], []) for k in ("node", "cell", "two", "max")}
            for _ in range(cfg["R"]):
                y = simulate(rng, C, n, icc, cfg["s_total"])
                m = y.mean()
                got = {}
                for k, f in (("node", se_node), ("cell", se_cell), ("two", se_two_level)):
                    got[k] = f(rng, y, cfg["B"])
                got["max"] = max(got["node"], got["cell"])
                for k, s in got.items():
                    out[k][0].append(s / se_true)
                    out[k][1].append(abs(m) <= 1.96 * s)
            cells = ["%7.2f, %5.3f" % (np.median(out[k][0]), np.mean(out[k][1])) for k in ("node", "cell", "two", "max")]
            print("%4d %5.2f | %22s | %22s | %22s | %22s" % (C, icc, cells[0], cells[1], cells[2], cells[3]), flush=True)
        print()


if __name__ == "__main__":
    main()
