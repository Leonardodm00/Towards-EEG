"""Smoke test for the inversion, flags and fill -- Blocks 7 and 14.5 in
specs/SPEC.md (procedure Eq. 1, s.3.9; mathematics Eqs. 20-21; D5, D-030,
D-063.1, D-063.4, D-071.4).

The inversion's checks read tables built from known grid values
(grid_table: b, the failure rate, the mass and the support set on the grid of
the design's ranges), so that they test the inversion, not an estimator.

Checks
    test_known_answer   on a grid table of b = 1 + c0/d, d_tilde recovers d
                        to 1e-4; the shortcut d_hat / b(d_hat) has the
                        closed-form error c0^2 / (d (d + 2 c0)) and agrees with
                        mathematics Eq. 21 to first order
    test_reference      fill rules on hand-made stretches, values by hand;
                        off_support (D-063.4) on a local linear table fitted to
                        a 3-D fixture: it fires at the node of
                        check_table_se_coverage.py C5 (d = 0.5 um at mu = 4
                        /um, outside the design's band) and not at a node in
                        the band's middle
    test_convergence    skipped: no discretisation parameter
    test_invariants     m_hat(d_tilde, phi, mu_hat d_hat) = d_hat at every
                        inverted node (1e-9), b depending on all three
                        coordinates
    test_contract       Inversion fields; correct_nodes keeps the selection
                        reasons of nodes outside S, inverts a dark node (D-030:
                        a label, Block 14.1) and reads each node's own mu_hat;
                        mu_d_out_of_domain is a label (D-071.4): the node is
                        inverted at the axis' end and keeps its d_tilde;
                        end_to_end.phantom_config draws mu on [lo / d, hi / d]
    test_determinism    skipped: deterministic closed forms and root finding
    test_edge_cases     non-monotone table, out-of-domain d_hat, large
                        correction, high failure rate, low mass and a region
                        without support (off_support), tilt outside the table
                        (flagged and filled, the root kept), missing estimate
                        (d_hat, mu_hat), mismatched estimator signature

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_invert.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import fill as FI  # noqa: E402
from allen_diameter.analysis import invert as IV  # noqa: E402
from allen_diameter.analysis import table as TB  # noqa: E402
from allen_diameter.analysis.node_pipeline import NodeResult  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy")


def grid_table(b_fn, cfg=None, fail_fn=None, mass_fn=None, support_fn=None, q_loo=10.0):
    """A BiasTable of known grid values on the design's ranges (grid_axes without rows, c = 1): b_hat = b_fn(d,
    phi, mu_d), the failure rate fail_fn (default 0), the mass mass_fn (default 1000), the support support_fn
    (default everywhere), q_loo given; the SEs and tau constant."""
    cfg = cfg or default_config()
    axes = TB.grid_axes(np.empty((0, 3)), cfg, 1.0)
    x1, x2, x3 = np.meshgrid(*axes, indexing="ij")
    d, phi, md = np.exp(x1), x2, np.exp(x3)
    ones = np.ones(x1.shape)
    arrays = dict(b_hat=b_fn(d, phi, md) * ones, se_hat=0.01 * ones, se_indep=0.01 * ones, tau_hat=0.03 * ones,
                  n_eff=100.0 * ones, mass=(1000.0 * ones if mass_fn is None else mass_fn(d, phi, md) * ones),
                  p_fail=(0.0 * ones if fail_fn is None else fail_fn(d, phi, md) * ones))
    supported = np.ones(x1.shape, bool) if support_fn is None else support_fn(d, phi, md) & np.ones(x1.shape, bool)
    meta = dict(estimator="grid", q_loo=q_loo, estimator_hash=cfg.signature_hash("estimator"),
                signature=cfg.full_signature())
    return TB.BiasTable(axes, arrays, supported, meta), cfg


def node(d_hat, phi, status="converged", flags=(), mu_hat=1.0):
    nan = float("nan")
    return NodeResult(0, 3, 0.0, 0.0, 0.0, 0.0, "", nan, nan, 0, 0.0, 0.0, 0.0, 0.0, 0.0, phi, False, False, 200.0,
                      "block_masked", d_hat, mu_hat, 0.0, 1.0, status, tuple(flags),
                      None, "gradient_energy", np.empty(0, dtype=int), np.empty(0), np.empty(0), -1)


def fixture_rows(G, n_real, seed, cfg, tau=0.03, eta=0.1):
    """Rows of the G x n_real design on cfg's ranges with b = 1 + 0.06/d + 0.15 sin^2 phi + 0.05 ln(mu d) at the
    rows' own ln(mu_hat d_hat) (mu_hat d_hat = mu d exp(N(0, eta^2)) per realization), every row in S."""
    rng = np.random.default_rng(seed)
    ph = cfg.phantom
    (dlo, dhi), (mlo, mhi) = ph.d_range_um, ph.mu_range_per_um
    rows = []
    for g in range(1, G + 1):
        d = math.exp(rng.uniform(math.log(dlo), math.log(dhi)))
        phi = math.radians(rng.uniform(*ph.phi_range_deg))
        mu = math.exp(rng.uniform(math.log(mlo), math.log(mhi)))
        for r in range(1, n_real + 1):
            md = mu * d * math.exp(rng.normal(0.0, eta))
            y = 1.0 + 0.06 / d + 0.15 * math.sin(phi) ** 2 + 0.05 * math.log(md) + rng.normal(0.0, tau)
            rows.append(dict(draw_index=g, realization_index=r, d_um=d, phi_rad=phi, mu_per_um=mu, d_hat_um=y * d,
                             mu_hat_per_um=md / (y * d), ratio=y, in_S=True, reject=""))
    return rows


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    c0 = 0.1
    T, cfg = grid_table(lambda d, p, m: 1.0 + c0 / d)
    for d in (0.4, 0.8, 1.5, 3.0):
        phi = math.radians(25.0)
        inv = IV.invert_node(d + c0, phi, 1.0, T, cfg)        # d_hat = m(d) = d b(d) = d + c0
        assert abs(inv.d_root_um / d - 1) <= 1e-4, (d, inv)
        assert inv.flags in ((), ("large_correction",)), inv            # b(0.4) = 1.25: above the threshold
        if not inv.flags:
            assert inv.d_tilde_um == inv.d_root_um, inv
        rel = IV.shortcut(d + c0, phi, 1.0, T) / d - 1
        exact = c0 ** 2 / (d * (d + 2 * c0))
        b, beta = 1 + c0 / d, -(c0 / d) / (1 + c0 / d)
        eq21 = -beta * (b - 1) / (1 + beta * (b - 1))
        assert abs(rel - exact) <= 1e-4, ("shortcut vs closed form", d, rel, exact)
        assert abs(rel - eq21) <= 2 * (b - 1) ** 2, ("Eq. 21 to first order", d, rel, eq21)


def test_reference():
    cfg = default_config()
    nan = float("nan")
    d, src = FI.fill_stretch([1.0, nan, 1.2, nan, nan, 0.8], 0.5, cfg)
    assert np.allclose(d, [1.0, 1.1, 1.1, 1.0, 1.0, 0.8]) and list(src) == ["self", "neighbours", "self",
                                                                             "neighbours", "neighbours", "self"], (d, src)
    d, src = FI.fill_stretch([nan, nan, nan], [0.4, 0.5, 0.6], cfg)
    assert np.allclose(d, [0.4, 0.5, 0.6]) and set(src) == {"allen"}
    d, src = FI.fill_stretch([1.0, 1.0, 3.0, 1.0, 1.0], 0.5, cfg)
    assert np.allclose(d, 1.0), "the running median removes a one-node spike"
    allen = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, fill_policy="allen_only"))
    d, src = FI.fill_stretch([1.0, nan, 1.2], 0.5, allen)
    assert np.allclose(d, [1.0, 1.0, 1.2]) and src[1] == "allen", d    # median of (1.0, 0.5, 1.2) at the middle
    none = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, fill_policy="none"))
    d, src = FI.fill_stretch([1.0, nan, 1.2, 1.4], 0.5, none)
    assert math.isnan(d[1]) and src[1] == "none" and np.allclose(d[[0, 2, 3]], [1.0, 1.2, 1.4])
    # off_support (D-063.4) on a fitted local linear table, the design's band mu d in [mu_min d, mu_max d]: the C5
    # node of check_table_se_coverage.py (d = 0.5 um at mu = 4 /um, mu d = 2: above mu_max d) against one in the
    # band's middle (d = 1 um, mu d = 1)
    fit = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, bandwidth_cv_grid=(1.0,)))
    T = TB.fit_table(fixture_rows(3000, 2, SEED, fit), fit)
    phi = math.radians(30.0)
    for d0, md0, want in ((0.5, 2.0, True), (1.0, 1.0, False)):
        d_hat = float(T.m_hat(d0, phi, md0))
        inv = IV.invert_node(d_hat, phi, md0 / d_hat, T, fit)
        assert abs(inv.d_root_um / d0 - 1) <= 1e-9, (d0, inv)
        assert ("off_support" in inv.flags) == want, (d0, md0, inv.flags, float(T.mass(d0, phi, md0)), T.q_loo)
    assert float(T.mass(0.5, phi, 2.0)) < T.q_loo < float(T.mass(1.0, phi, 1.0))


def test_convergence():
    raise unittest.SkipTest("no discretisation parameter in Block 7")


def test_invariants():
    T, cfg = grid_table(lambda d, p, m: 1.05 + 0.08 / d + 0.1 * np.sin(p) ** 2 + 0.03 * np.log(m))
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        d, phi = math.exp(rng.uniform(math.log(0.3), math.log(3.5))), math.radians(rng.uniform(0, 85))
        md = math.exp(rng.uniform(math.log(0.1), math.log(8.0)))
        d_hat = float(T.m_hat(d, phi, md))
        inv = IV.invert_node(d_hat, phi, md / d_hat, T, cfg)
        assert abs(float(T.m_hat(inv.d_root_um, phi, md)) - d_hat) <= 1e-9 * d_hat, inv
        assert abs(inv.d_root_um / d - 1) <= 1e-9, (d, inv)


def test_contract():
    T, cfg = grid_table(lambda d, p, m: 1.0 + 0.05 / d)
    out = IV.correct_nodes([node(1.05, 0.2), node(1.05, 0.2, flags=("faint",)), node(float("nan"), 0.2, status="none"),
                            node(1.05, 0.2, flags=("dark", "steep"))], T, cfg)
    assert isinstance(out[0], IV.Inversion) and abs(out[0].d_tilde_um - 1.0) <= 1e-4 and out[0].flags == ()
    assert math.isnan(out[1].d_tilde_um) and out[1].flags == ("faint",)
    assert math.isnan(out[2].d_tilde_um) and out[2].flags == ("status:none",)
    assert out[3].d_tilde_um == out[0].d_tilde_um and out[3].flags == (), out[3]       # D-030: dark is a label
    # each node's own mu_hat is read (D-063.1): b = 1 + 0.05 ln(mu d) inverts mu d = 1 and mu d = e differently
    Tm, _ = grid_table(lambda d, p, m: 1.0 + 0.05 * np.log(m))
    a, b = IV.correct_nodes([node(1.0, 0.2, mu_hat=1.0), node(1.0, 0.2, mu_hat=math.e)], Tm, cfg)
    assert abs(a.d_tilde_um - 1.0) <= 1e-6 and abs(b.d_tilde_um - 1.0 / 1.05) <= 1e-6, (a, b)
    # mu_d_out_of_domain is a label (D-071.4): mu d above the axis is clamped to its end, the node keeps d_tilde
    hi = T.mu_d_range[1]
    inv = IV.invert_node(1.05, 0.2, 3.0 * hi / 1.05, T, cfg)
    assert inv.flags == ("mu_d_out_of_domain",) and abs(inv.d_tilde_um - 1.0) <= 1e-4, inv
    inv2 = IV.invert_node(1.0, 0.2, 3.0 * hi, Tm, cfg)
    want = 1.0 / (1.0 + 0.05 * math.log(hi))                 # inverted at the axis' end
    assert inv2.flags == ("mu_d_out_of_domain",) and abs(inv2.d_tilde_um / want - 1) <= 1e-6, (inv2, want)
    # gate 2's test phantoms: --alpha-range lo,hi draws mu on [lo / d, hi / d] for each diameter
    sys.path.insert(0, str(WS / "scripts"))
    import end_to_end
    base = default_config()
    c2 = end_to_end.phantom_config(base, 2.0, (0.15, 0.9))
    assert c2.phantom.mu_range_per_um == (0.075, 0.45) and end_to_end.phantom_config(base, 2.0) is base


def test_determinism():
    raise unittest.SkipTest("deterministic closed forms and root finding")


def test_edge_cases():
    # non-monotone: b peaks at d = 1, so m = d b falls after it
    T, cfg = grid_table(lambda d, p, m: 1.0 + 0.8 * np.exp(-((np.log(d)) / 0.2) ** 2))
    assert "non_monotone" in IV.invert_node(1.6, 0.3, 1.0, T, cfg).flags
    T, cfg = grid_table(lambda d, p, m: 1.0 + 0.05 / d)
    lo, hi = T.d_range
    assert "out_of_domain" in IV.invert_node(hi * 1.3, 0.3, 1.0, T, cfg).flags
    assert "out_of_domain" in IV.invert_node(lo * 0.5, 0.3, 1.0, T, cfg).flags
    for bad in ((float("nan"), 1.0), (1.0, 0.0), (1.0, float("nan")), (-1.0, 1.0)):
        assert IV.invert_node(bad[0], 0.3, bad[1], T, cfg).flags == ("no_estimate",), bad
    inv = IV.invert_node(1.05, math.radians(95.0), 1.0, T, cfg)
    assert inv.flags == ("phi_out_of_domain",) and math.isnan(inv.d_tilde_um) and abs(inv.d_root_um - 1) <= 1e-4, inv
    T, cfg = grid_table(lambda d, p, m: 1.3 + 0.0 * d)
    inv = IV.invert_node(1.3, 0.3, 1.0, T, cfg)
    assert "large_correction" in inv.flags and math.isnan(inv.d_tilde_um) and abs(inv.d_root_um - 1.0) <= 1e-4, inv
    T, cfg = grid_table(lambda d, p, m: 1.0 + 0.0 * d, fail_fn=lambda d, p, m: (p > np.radians(60.0)).astype(float))
    assert "high_failure" in IV.invert_node(1.0, math.radians(80.0), 1.0, T, cfg).flags
    assert "high_failure" not in IV.invert_node(1.0, math.radians(20.0), 1.0, T, cfg).flags
    # off_support: the mass at the root below q_loo, or a grid point around the root without support
    T, cfg = grid_table(lambda d, p, m: 1.0 + 0.0 * d, mass_fn=lambda d, p, m: np.where(d < 0.5, 5.0, 1000.0))
    assert "off_support" in IV.invert_node(0.3, 0.3, 1.0, T, cfg).flags
    assert IV.invert_node(1.0, 0.3, 1.0, T, cfg).flags == ()
    T, cfg = grid_table(lambda d, p, m: 1.0 + 0.0 * d, support_fn=lambda d, p, m: d > 0.5)
    inv = IV.invert_node(0.3, 0.3, 1.0, T, cfg)
    assert inv.flags == ("off_support",) and math.isnan(inv.d_tilde_um) and abs(inv.d_root_um - 0.3) <= 1e-4, inv
    assert IV.invert_node(1.0, 0.3, 1.0, T, cfg).flags == ()
    try:
        IV.correct_nodes([node(1.0, 0.2)], T, dataclasses.replace(cfg, measure=dataclasses.replace(
            cfg.measure, sigma_fit_um=0.125)))       # another estimator (sigma_fit_um is in its signature)
    except ValueError:
        pass
    else:
        raise AssertionError("a table must refuse a fit with another estimator")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
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
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
