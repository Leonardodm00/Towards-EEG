"""Smoke test for the phantoms and the bias table -- Block 6 in specs/SPEC.md
(procedure s.3.5, 3.7, 3.8; D-024).

Checks
    test_known_answer   on a fixture with a known b(ln d, phi), Gaussian scatter
                        tau and a known rejection pattern: b_hat within 4 SE
                        everywhere and within 2 SE at >= 85 % of the points,
                        median tau_hat within 10 % of tau, the failure rate
                        within 4 SD of its expectation given the draws;
                        m_hat = d * b_hat
    test_reference      the draws follow their laws (Kolmogorov-Smirnov on 2000
                        draws g = 1..2000, realization 1); Block 14.2: the
                        branch radius's noise at d = 0.2 um follows the normal
                        truncated below at -d/2 (KS against scipy.stats.truncnorm,
                        4000 realizations) and the rejections per kept draw
                        average Phi(-d/2s) / (1 - Phi(-d/2s)) within 4 SE;
                        D-062's keys and draw orders, recomputed from NumPy
                        alone, give the same features, nuisances and radius
    test_convergence    the SE falls as 1/sqrt(N): N = 500 vs 2000
    test_invariants     Block 14.2: a draw depends only on (seed, g) and a
                        realization only on (seed, g, r), in any order; a draw's
                        realizations share its features and differ in their
                        nuisances; the feature, nuisance and radius streams
                        differ for g, r in 1..3; index 0 is refused; the phantom
                        branch lies on the tube's axis at the configured spacing
                        and carries radius_um at every node; sd 0 gives d/2;
                        the grid design visits cell (g - 1) mod (n_d n_phi)
    test_contract       two rendered replicates end to end: row contract; CSV
                        and table files round trip exactly; the CLI runs and
                        merges; a changed estimator setting is refused
    test_determinism    a rendered replicate repeats bit for bit
    test_edge_cases     unsupported estimators raise; too few rows; bad inputs;
                        failure_rate_ignore: () (the default, D-057.1)
                        counts every row; a named screen ("dark" or
                        "stack_edge") removes the rows it rejected, whatever
                        else they failed; "crossing" and an unlabelled
                        rejection count as failures

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_table.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import phantoms as PH  # noqa: E402
from allen_diameter.analysis.table import check_rows_config, fit_table, row_provenance, table_inputs  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy")
PX = 0.1144


def b_true(d, phi):
    return 1.0 + 0.06 / d + 0.15 * np.sin(phi) ** 2


def fixture(n, seed, tau=0.03):
    rng = np.random.default_rng(seed)
    lo, hi = default_config().phantom.d_range_um
    d = np.exp(rng.uniform(math.log(lo), math.log(hi), n))
    phi = np.radians(rng.uniform(0, 90, n))
    ratio = b_true(d, phi) + rng.normal(0, tau, n)
    p_fail = np.where(phi > np.radians(45), 0.5, 0.0)
    fail = rng.random(n) < p_fail
    rows = [dict(d_um=a, phi_rad=b, ratio=c, in_S=not f) for a, b, c, f in zip(d, phi, ratio, fail)]
    return rows, d, phi, p_fail


def small_cfg(**phantom):
    """Short rendered replicates: low tilts, short tubes, smaller block, p_x/8 everywhere; 2 realizations a draw."""
    base = default_config()
    rc = dataclasses.replace(base.renderer, U_um=4.0, pad_um=1.0, h_g_um_thin=PX / 8)
    ms = dataclasses.replace(base.measure, block_half_um=3.5)
    ph = dataclasses.replace(base.phantom, phi_range_deg=(0.0, 30.0), d_range_um=(0.5, 1.2), nodes_each_way=4,
                             n_realizations=2, **phantom)
    return dataclasses.replace(base, renderer=rc, measure=ms, phantom=ph)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    rows, d, phi, p_fail = fixture(1500, SEED)
    T = fit_table(rows, cfg)
    dq = np.exp(np.linspace(math.log(0.3), math.log(3.0), 9))
    D, P = np.meshgrid(dq, np.radians(np.linspace(5, 85, 9)))
    z = np.abs(T.b_hat(D, P) - b_true(D, P)) / T.se_hat(D, P)
    assert z.max() <= 4.0 and (z > 2).mean() <= 0.15, ("b_hat", z.max(), (z > 2).mean())
    th = T.tau_hat(D, P)
    assert abs(np.median(th) / 0.03 - 1) <= 0.10 and th.min() >= 0.5 * 0.03 and th.max() <= 1.6 * 0.03, th
    X, Q = table_inputs(d, phi), table_inputs(D, P)
    w = np.exp(-0.5 * (((Q[:, :1] - X[None, :, 0]) / T.bandwidth[0]) ** 2 + ((Q[:, 1:2] - X[None, :, 1]) / T.bandwidth[1]) ** 2))
    expect = (w * p_fail).sum(1) / w.sum(1)
    sd = np.sqrt((w ** 2 * p_fail * (1 - p_fail)).sum(1)) / w.sum(1)
    zf = np.abs(T.failure_rate(D, P).ravel() - expect) / np.maximum(sd, 1e-12)
    # at phi = 20 deg the nearest rejections (phi > 45 deg) are 5 bandwidths away: weight ~ exp(-12.5)
    assert np.all(zf[sd > 0] <= 4.0) and float(T.failure_rate(1.0, math.radians(20.0))) <= 1e-3, zf.max()
    assert np.array_equal(T.m_hat(D, P), D * T.b_hat(D, P))


def test_reference():
    cfg = default_config()
    ph, acq = cfg.phantom, cfg.acquisition
    draws = [PH.draw_replicate(cfg, SEED, g, 1)[0] for g in range(1, 2001)]
    lo, hi = ph.d_range_um
    mlo, mhi = ph.mu_range_per_um
    laws = {
        "ln d": [(math.log(x.d_um) - math.log(lo)) / (math.log(hi) - math.log(lo)) for x in draws],
        "phi": [math.degrees(x.phi_rad) / 90.0 for x in draws],
        "theta": [x.theta_rad / math.pi for x in draws],
        "ln mu": [(math.log(x.mu_per_um) - math.log(mlo)) / (math.log(mhi) - math.log(mlo)) for x in draws],
        "x offset": [x.cx_um / acq.res0_um + 0.5 for x in draws],
        "y offset": [x.cy_um / acq.res0_um + 0.5 for x in draws],
        "depth": [x.cz_um / acq.dz_um + 0.5 for x in draws],
    }
    for name, u in laws.items():
        p = stats.kstest(u, "uniform").pvalue
        assert p > 1e-3, (name, p)
    # Block 14.2 (D-062.1): r_br = d/2 + eps, eps ~ N(0, sd^2) redrawn until positive; at d = 0.2 um, sd = 0.2 um a
    # draw is rejected with probability Phi(-0.5) = 0.31, so the kept eps follow the normal truncated below at -d/2
    d0, sd = 0.2, ph.radius_noise_sd_um
    reals = [PH.draw_replicate(cfg, SEED, g, 1, d_override_um=d0)[0] for g in range(1, 4001)]
    eps = np.array([x.radius_um - 0.5 * d0 for x in reals])
    p = stats.kstest(eps, stats.truncnorm(-0.5 * d0 / sd, np.inf, loc=0.0, scale=sd).cdf).pvalue
    assert p > 0.01 and np.all(eps > -0.5 * d0), ("radius law", p)
    q = 1.0 - stats.norm.cdf(-0.5 * d0 / sd)             # the probability that a draw is kept
    n_rej = np.array([x.radius_redraws for x in reals], dtype=float)
    mean, se = (1 - q) / q, math.sqrt((1 - q) / q ** 2 / n_rej.size)
    assert abs(n_rej.mean() - mean) <= 4 * se, ("rejections per kept draw", n_rej.mean(), mean, se)
    # D-062's keys and orders, recomputed from NumPy alone: the features of draw g from default_rng([seed, g]) in
    # the order d, phi, theta, mu; the nuisances of realization r from default_rng([seed, g, r]) in the order
    # (cx, cy), cz, the returned generator positioned after them; the radius from default_rng([seed, g, r, 1])
    assert ph.d_log_uniform and ph.mu_log_uniform and ph.subpixel_offset and ph.axis_depth_jitter
    for g, r in ((1, 1), (5, 3), (12, 10)):
        x, gen = PH.draw_replicate(cfg, SEED, g, r)
        f = np.random.default_rng([SEED, g])
        want = (math.exp(f.uniform(math.log(lo), math.log(hi))), math.radians(f.uniform(*ph.phi_range_deg)),
                f.uniform(*ph.theta_range_rad), math.exp(f.uniform(math.log(mlo), math.log(mhi))))
        assert (x.d_um, x.phi_rad, x.theta_rad, x.mu_per_um) == want, ("feature key", g, r)
        n = np.random.default_rng([SEED, g, r])
        off = n.uniform(-0.5 * acq.res0_um, 0.5 * acq.res0_um, 2)
        cz = n.uniform(-0.5 * acq.dz_um, 0.5 * acq.dz_um)
        assert (x.cx_um, x.cy_um, x.cz_um) == (off[0], off[1], cz) and gen.random() == n.random(), ("nuisance key", g, r)
        e = np.random.default_rng([SEED, g, r, 1])
        for k in range(100):
            rb = 0.5 * want[0] + ph.radius_noise_sd_um * e.standard_normal()
            if rb > 0:
                break
        assert (x.radius_um, x.radius_redraws) == (rb, k), ("radius key", g, r)


def test_convergence():
    cfg = default_config()
    se = []
    for n in (500, 2000):
        T = fit_table(fixture(n, SEED + n)[0], cfg)
        se.append(float(T.se_hat(1.0, math.radians(30.0))))
    assert abs(se[0] / se[1] / 2.0 - 1) <= 0.25, se


def test_invariants():
    cfg = default_config()
    a = PH.draw_replicate(cfg, SEED, 7, 2)[0]
    for g, r in ((3, 1), (11, 4), (7, 1), (1, 1)):
        PH.draw_replicate(cfg, SEED, g, r)
    assert PH.draw_replicate(cfg, SEED, 7, 2)[0] == a
    assert PH.draw_replicate(cfg, SEED + 1, 7, 2)[0] != a
    # a draw's realizations share its features and differ in their nuisances and radius
    reals = [PH.draw_replicate(cfg, SEED, 7, r)[0] for r in (1, 2, 3)]
    assert len({(x.d_um, x.phi_rad, x.theta_rad, x.mu_per_um) for x in reals}) == 1
    assert len({(x.cx_um, x.cy_um, x.cz_um) for x in reals}) == 3 and len({x.radius_um for x in reals}) == 3
    assert PH.draw_features(cfg, SEED, 7) == (a.d_um, a.phi_rad, a.theta_rad, a.mu_per_um)
    # the streams of D-062's keys are distinct for indices from 1 (NumPy pads a key with zeros: [s, g, 0] = [s, g])
    first = {}
    for g in (1, 2, 3):
        first[(g,)] = np.random.default_rng([SEED, g]).random()
        for r in (1, 2, 3):
            first[(g, r)] = np.random.default_rng([SEED, g, r]).random()
            first[(g, r, 1)] = np.random.default_rng([SEED, g, r, 1]).random()
    assert len(set(first.values())) == len(first), "two generator keys give one stream"
    assert np.random.default_rng([SEED, 7, 0]).random() == np.random.default_rng([SEED, 7]).random()  # why from 1
    for bad in ((0, 1), (1, 0), (-2, 1), (1.5, 1)):
        try:
            PH.draw_replicate(cfg, SEED, *bad)
        except ValueError:
            continue
        raise AssertionError("index %r accepted" % (bad,))
    zero = dataclasses.replace(cfg, phantom=dataclasses.replace(cfg.phantom, radius_noise_sd_um=0.0))
    z = PH.draw_replicate(zero, SEED, 7, 2)[0]
    assert z.radius_um == 0.5 * z.d_um and z.radius_redraws == 0 and (z.cx_um, z.cz_um) == (a.cx_um, a.cz_um)
    # the grid design (an option; D-024 is random): draw g takes cell (g - 1) mod (n_d n_phi), d-major, so the
    # draws 1..n_d n_phi visit every cell once, in order, and draw n_d n_phi + 1 starts again
    grid = dataclasses.replace(cfg, phantom=dataclasses.replace(cfg.phantom, design="grid"))
    gd, gp = grid.phantom.grid_d_um, grid.phantom.grid_phi_deg
    cells = [PH.draw_features(grid, SEED, g)[:2] for g in range(1, len(gd) * len(gp) + 2)]
    want = [(float(x), math.radians(y)) for x in gd for y in gp]
    assert cells[:-1] == want and cells[-1] == want[0], "grid design cells"
    draw, rng = PH.draw_replicate(cfg, SEED, 5, 1)
    br = PH.phantom_branch(draw, cfg, rng)
    assert np.all(np.asarray(br.radius_um) == draw.radius_um) and draw.radius_um > 0
    tube = PH.phantom_tube(draw, cfg)
    q = br.xyz_um - np.array(tube.c)
    t = tube.t_hat
    off_axis = np.linalg.norm(q - (q @ t)[:, None] * t[None, :], axis=1)
    assert off_axis.max() <= 1e-12 and np.allclose(np.diff(br.s_um), cfg.phantom.phantom_node_step_um, atol=1e-12)
    assert np.allclose(br.xyz_um[cfg.phantom.nodes_each_way], tube.c, atol=0)


def test_contract():
    cfg = small_cfg()
    rows = [PH.run_replicate(cfg, SEED, 1, r) for r in (1, 2)]
    for r in rows:
        for k in ("draw_index", "realization_index", "seed", "d_um", "phi_rad", "theta_rad", "mu_per_um", "cx_um",
                  "cy_um", "cz_um", "radius_um", "radius_redraws", "d_hat_um", "mu_hat_per_um", "fit_status", "flags",
                  "in_S", "reject", "ratio", "meas_phi_rad", "seconds"):
            assert k in r, k
        assert isinstance(r["in_S"], bool) and r["in_S"] == (r["reject"] == "")
        assert abs(r["ratio"] - r["d_hat_um"] / r["d_um"]) <= 1e-15 * abs(r["ratio"])
    assert [(r["draw_index"], r["realization_index"]) for r in rows] == [(1, 1), (1, 2)]
    assert rows[0]["d_um"] == rows[1]["d_um"] and rows[0]["radius_um"] != rows[1]["radius_um"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rows.csv")
        table_io.write_rows(rows, path)
        back = table_io.read_rows(path)
        for a, b in zip(rows, back):
            for k, v in a.items():
                assert (v == b[k]) or (isinstance(v, float) and math.isnan(v) and math.isnan(b[k])), (k, v, b[k])
        T = fit_table(fixture(300, SEED)[0], default_config())
        stem = os.path.join(tmp, "table")
        table_io.save_table(T, stem)
        T2 = table_io.load_table(stem)
        D, P = np.meshgrid(np.linspace(0.3, 3, 5), np.radians(np.linspace(0, 80, 5)))
        assert np.array_equal(T.b_hat(D, P), T2.b_hat(D, P)) and T2.estimator_hash == T.estimator_hash
        T2.check_estimator(default_config())
        try:
            base = default_config()
            T2.check_estimator(dataclasses.replace(base, measure=dataclasses.replace(base.measure,
                                                                                     sigma_fit_um=0.125)))
        except ValueError:
            pass
        else:
            raise AssertionError("a table must refuse a fit with another sigma_fit")
        # the CLI, on 4 rendered replicates with a fixed smoothing (too few rows for CV)
        cfgf = small_cfg()
        cfgf = dataclasses.replace(cfgf, correction=dataclasses.replace(cfgf.correction, spline_smoothing="1.0"))
        cj = os.path.join(tmp, "cfg.json")
        with open(cj, "w") as f:
            f.write(cfgf.to_json())
        script = str(WS / "scripts" / "build_table.py")
        out = os.path.join(tmp, "tab")
        for a, b in ((1, 2), (2, 3)):             # two draws x 2 realizations (small_cfg)
            subprocess.run([sys.executable, script, "run", "--draw-start", str(a), "--draw-stop", str(b), "--out-dir",
                            out, "--config-json", cj], check=True, capture_output=True)
        refused = subprocess.run([sys.executable, script, "run", "--draw-start", "0", "--draw-stop", "1", "--out-dir",
                                  out, "--config-json", cj], capture_output=True, text=True)
        assert refused.returncode != 0 and "--draw-start" in refused.stderr, refused.stderr[-300:]
        subprocess.run([sys.executable, script, "merge", "--out-dir", out, "--config-json", cj], check=True,
                       capture_output=True)
        h = cfgf.signature_hash("estimator")
        meta = json.load(open(os.path.join(out, "bias_table_%s.json" % h)))
        assert meta["n_all"] == 4 and meta["estimator_hash"] == h, meta["n_all"]
        # 2026-10-07: every row carries the simulation hash, and the merge refuses rows rendered under another
        # simulation configuration -- the dangerous case is the same estimator hash, so the row files are found --
        # while a change of the correction settings, which only the merge reads, is accepted
        import glob as _glob
        rows4 = table_io.read_rows(sorted(_glob.glob(os.path.join(out, "rows_*.csv"))))
        assert {r["sim_hash"] for r in rows4} == {row_provenance(cfgf)}, {r["sim_hash"] for r in rows4}
        other = dataclasses.replace(cfgf, phantom=dataclasses.replace(cfgf.phantom, d_range_um=(0.5, 2.0)))
        corr = dataclasses.replace(cfgf, correction=dataclasses.replace(cfgf.correction, spline_smoothing="2.0"))
        assert other.signature_hash("estimator") == h and corr.signature_hash("estimator") == h
        for c, ok in ((other, False), (corr, True)):
            cj2 = os.path.join(tmp, "cfg2.json")
            with open(cj2, "w") as f:
                f.write(c.to_json())
            res = subprocess.run([sys.executable, script, "merge", "--out-dir", out, "--config-json", cj2],
                                 capture_output=True, text=True)
            assert (res.returncode == 0) == ok, (ok, res.returncode, res.stderr[-300:])
            if not ok:
                assert "merge refused" in res.stderr and "--config-json" in res.stderr, res.stderr[-300:]


def test_determinism():
    cfg = small_cfg()
    a, b = PH.run_replicate(cfg, SEED, 3, 2), PH.run_replicate(cfg, SEED, 3, 2)
    a.pop("seconds"), b.pop("seconds")
    assert a == b or all((a[k] == b[k]) or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])) for k in a)


def test_edge_cases():
    # row provenance (2026-10-07): rows without the column (older runs) are counted, not refused; one row from
    # another configuration refuses the lot; the prefix keeps a numeric-looking hash a string through the CSV
    cfg0 = default_config()
    good = row_provenance(cfg0)
    assert check_rows_config([dict(sim_hash=good), dict(sim_hash=good)], cfg0) == 0
    assert check_rows_config([dict(sim_hash=good), dict(), dict(sim_hash="")], cfg0) == 2
    try:
        check_rows_config([dict(sim_hash=good), dict(sim_hash="sim-0000000000000000")], cfg0)
    except ValueError:
        pass
    else:
        raise AssertionError("rows from another configuration were accepted")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "r.csv")
        table_io.write_rows([dict(sim_hash="sim-1234567890123456"), dict(sim_hash="sim-1e10000000000000")], p)
        assert [r["sim_hash"] for r in table_io.read_rows(p)] == ["sim-1234567890123456", "sim-1e10000000000000"]
    rows = fixture(200, SEED)[0]
    for change in (dict(response_estimator="local_linear"), dict(table_statistic="median")):
        cfg = default_config()
        cfg = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, **change))
        try:
            fit_table(rows, cfg)
        except NotImplementedError:
            continue
        raise AssertionError("unsupported estimator accepted: %r" % change)
    # failure_rate_ignore: the default () counts every row (D-057.1); a named screen removes the rows it rejected,
    # whatever else they failed; a row failing other screens, or with no reason given, is a failure
    cfg = default_config()
    assert cfg.correction.failure_rate_ignore == ()
    base = [dict(r, reject="") for r in rows]
    t0 = fit_table(base, cfg)
    extra = [dict(d_um=1.0, phi_rad=0.2, ratio=float("nan"), in_S=False, reject=why) for why in
             ("dark", "dark", "stack_edge;dark", "crossing", "")]
    t2 = fit_table(base + extra, cfg)
    assert t2.X_all.shape[0] == t0.X_all.shape[0] + 5 and int((~t2.ok_all).sum()) == int((~t0.ok_all).sum()) + 5
    assert t2.n_rows == len(base) + 5
    for named, left in ((("dark",), 2), (("stack_edge",), 4)):
        t1 = fit_table(base + extra, dataclasses.replace(cfg, correction=dataclasses.replace(
            cfg.correction, failure_rate_ignore=named)))
        assert t1.X_all.shape[0] == t0.X_all.shape[0] + left, (named, t1.X_all.shape[0] - t0.X_all.shape[0])
    for bad in (lambda: fit_table(rows[:6], default_config()), lambda: table_inputs([0.0], [0.1]),
                lambda: table_inputs([1.0], [np.nan])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("bad input accepted")


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
