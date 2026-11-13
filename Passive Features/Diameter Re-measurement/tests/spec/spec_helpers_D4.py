"""Independent references for run D4's spec-derived tests (Blocks 10, 15, 19
and their boundaries; integration tester, 2026-11-13). Written from
specs/SPEC.md only. Nothing here calls the logic under test: the only code of
the package used is the plain data containers (PlaneScan, FarSelection, the
configuration dataclasses), which the tests need to feed the functions.

  cfg_with           default_config() with fields of the calibration, survey,
                     renderer or measure sections replaced (validated)
  ref_sigma_r        Block 4's kernel law (section 2.3): linear interpolation of
                     the table in |delta|, linear continuation beyond
  ref_growth         Block 10's growth family (Eq. 4): G through the knots
                     (+-delta_m, G_m), scipy CubicSpline not-a-knot or
                     numpy.interp, linear continuation beyond the end knots
  eq4_scans          near PlaneScans whose omega follow Eq. 4 exactly (plus
                     optional noise and NaN planes)
  ref_fit_eq4        Eq. 4 minimised independently (scipy least_squares with a
                     numerical Jacobian; the spec's start rule and bounds;
                     optional per-node weights, Block 19.7)
  f1_scans           far PlaneScans whose omega follow (F1) of 15.4 exactly
  ref_select_far     15.3's plane selection written from the spec
  ref_se_model       15.4's model SE: 2 w RMS [(J^T J)^-1]^(1/2)_ss on +-FWHM
  ref_pool           19.4's round-robin pool with the per-cell and pool caps
  ref_anova          19.8's one-way random-effects ANOVA (textbook moments)
  ref_cell_z         19.8's leave-one-out z and p of every cell

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize, stats

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DZ = 0.28                      # um, section 2.1 (plane spacing)
FW = math.sqrt(8.0 * math.log(2.0))   # FWHM / SD of a Gaussian (15.2 notation: FWHM = 2 sqrt(2 ln 2) sigma)
SQ2PI = math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------ configuration --

def cfg_with(calibration=None, survey=None, renderer=None, measure=None):
    """default_config() with the named fields of each section replaced; validated."""
    from allen_diameter.config import default_config
    cfg = default_config()
    for name, over in (("calibration", calibration), ("survey", survey), ("renderer", renderer),
                       ("measure", measure)):
        if over:
            cfg = dataclasses.replace(cfg, **{name: dataclasses.replace(getattr(cfg, name), **over)})
    cfg.validate()
    return cfg


def node_result(**over):
    """A NodeResult container with neutral values (a converged pilot in focus at plane 0, no flag); fields
    replaced by name."""
    from allen_diameter.analysis.node_pipeline import NodeResult
    base = dict(node_id=1, type=3, x_um=0.0, y_um=0.0, z_um=0.0, path_um=0.0, reg_verdict="", s_star_um=0.0,
                dz_star_um=0.0, k_star=0, z_sub_um=0.0, cx_um=0.0, cy_um=0.0, cz_um=0.0, theta_rad=0.0, phi_rad=0.0,
                steep=False, vertical=False, B_bar=180.0, B_bar_region="block_masked", d_hat_um=0.3,
                mu_hat_per_um=0.5, v0_hat_um=0.0, alpha_hat=0.15, fit_status="converged", flags=(), fit=None,
                focus_rule="gradient_energy", focus_planes=np.arange(3), focus_score=np.zeros(3),
                focus_depth=np.zeros(3), k_star_depth=0)
    base.update(over)
    return NodeResult(**base)


# ------------------------------------------------------------ Block 4's law --

def ref_sigma_r(delta, table_delta, table_sigma, slope):
    """sigma_r(delta) of Block 4 with the linear continuation: numpy.interp of
    the table at |delta| <= the last knot, sigma(last) + slope (|delta| - last)
    beyond."""
    d = np.abs(np.asarray(delta, dtype=float))
    xs, ys = np.asarray(table_delta, dtype=float), np.asarray(table_sigma, dtype=float)
    return np.where(d <= xs[-1], np.interp(np.minimum(d, xs[-1]), xs, ys), ys[-1] + slope * (d - xs[-1]))


# ------------------------------------------------------------ Block 10, Eq. 4 --

def knots_symmetric(M, h=DZ):
    return np.arange(M + 1) * h


def ref_growth(delta, G, h=DZ, interp="cubic", convention="symmetric"):
    """G(delta) (um^2). symmetric: G = (G_0 .. G_M) at m h, mirrored (even);
    min_origin: G = (G_0 .. G_2M) at (m - M) h. cubic: CubicSpline not-a-knot
    through the signed knots; linear: numpy.interp. Linear continuation
    beyond +-M h with the end slope."""
    G = np.asarray(G, dtype=float)
    if convention == "symmetric":
        M = G.size - 1
        y = np.r_[G[::-1], G[1:]]
    else:
        M = (G.size - 1) // 2
        y = G
    x = np.arange(-M, M + 1) * h
    d = np.asarray(delta, dtype=float)
    e = M * h
    if interp == "cubic":
        cs = interpolate.CubicSpline(x, y, bc_type="not-a-knot")
        inside = cs(np.clip(d, -e, e))
        lo_val, lo_slope = cs(-e), cs(-e, 1)
        hi_val, hi_slope = cs(e), cs(e, 1)
    else:
        inside = np.interp(np.clip(d, -e, e), x, y)
        lo_val, lo_slope = y[0], (y[1] - y[0]) / h
        hi_val, hi_slope = y[-1], (y[-1] - y[-2]) / h
    out = np.where(d > e, hi_val + hi_slope * (d - e), inside)
    out = np.where(d < -e, lo_val + lo_slope * (d + e), out)
    return out


def make_scan(node_id, k_star, z_sub, omega, ks=None, z_um=None, which="near", status=None, **extra):
    """A PlaneScan with the given omega (NaN where absent); the other per-plane
    arrays NaN; status "ok" where omega is finite unless given."""
    from allen_diameter.analysis.calibration import PlaneScan
    omega = np.asarray(omega, dtype=float)
    n = omega.size
    if ks is None:
        P = (n - 1) // 2
        ks = int(k_star) + np.arange(-P, P + 1)
    ks = np.asarray(ks, dtype=int)
    if z_um is None:
        z_um = ks * DZ
    nan = np.full(n, np.nan)
    if status is None:
        status = tuple("ok" if np.isfinite(w) else "no_dip" for w in omega)
    return PlaneScan(int(node_id), float(extra.pop("d_hat_um", 0.2)), float(extra.pop("phi_rad", 0.0)),
                     float(extra.pop("alpha_hat", 0.3)), int(k_star), float(z_sub), ks, np.asarray(z_um, float),
                     omega, nan.copy(), nan.copy(), nan.copy(), np.full(n, 200.0), tuple(status), which, **extra)


def eq4_scans(G, c, z_ax, z_sub, k_star, interp="cubic", convention="symmetric", P=3, h=DZ,
              noise_sd=0.0, nan_frac=0.0, rng=None, node_ids=None, z_shift=0.0):
    """One near PlaneScan per node i: planes k*_i - P .. k*_i + P at z_k = k dz
    (+ z_shift), omega = c_i + G(z_ax_i - z_k) (+ noise), a share nan_frac of
    the planes NaN (never the node's k*)."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for i in range(len(c)):
        ks = int(k_star[i]) + np.arange(-P, P + 1)
        z = ks * DZ + z_shift
        om = c[i] + ref_growth(z_ax[i] - z, G, h, interp, convention)
        if noise_sd:
            om = om + rng.normal(0.0, noise_sd, om.size)
        if nan_frac:
            drop = rng.random(om.size) < nan_frac
            drop[P] = False
            om = np.where(drop, np.nan, om)
        nid = i + 1 if node_ids is None else node_ids[i]
        out.append(make_scan(nid, k_star[i], z_sub[i], om, ks=ks, z_um=z))
    return out


def _basis_matrix(delta, M, h, interp, convention):
    """(n, K + 1): column m = ref_growth of the unit vector of knot m."""
    K = M if convention == "symmetric" else 2 * M
    cols = []
    for m in range(K + 1):
        e = np.zeros(K + 1)
        e[m] = 1.0
        cols.append(ref_growth(delta, e, h, interp, convention))
    return np.column_stack(cols)


def ref_fit_eq4(scans, M=4, h=DZ, interp="cubic", convention="symmetric", z_bound=0.28, weights=None,
                min_planes=3):
    """Eq. (4) minimised independently: residual sqrt(w_i) (c_i + G(z_ax,i - z_k)
    - omega_ik) over c (N), z_ax (N, within z_bound of the start), the free knots
    (G(0) = 0; >= 0 under min_origin); start z_ax = z_sub (k* dz when NaN), (c, G)
    by linear least squares (lsq_linear) there; scipy least_squares (trf,
    3-point Jacobian, tolerances 1e-15). Nodes with fewer than min_planes finite
    omega are left out. Returns dict(c, z, G (K + 1,), cost (0.5 sum of weighted
    squares), node_ids)."""
    used = [s for s in scans if int(np.isfinite(s.omega_um2).sum()) >= min_planes]
    N = len(used)
    K = M if convention == "symmetric" else 2 * M
    zero = 0 if convention == "symmetric" else M
    free = np.array([m for m in range(K + 1) if m != zero])
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=float)
    oi, oz, ow = [], [], []
    z0 = np.empty(N)
    for i, s in enumerate(used):
        ok = np.isfinite(s.omega_um2)
        oi.append(np.full(int(ok.sum()), i))
        oz.append(np.asarray(s.z_um)[ok])
        ow.append(np.asarray(s.omega_um2)[ok])
        z0[i] = s.z_sub_um if math.isfinite(s.z_sub_um) else s.k_star * DZ
    oi, oz, ow = (np.concatenate(a) for a in (oi, oz, ow))
    sw = np.sqrt(w)[oi]

    def model(q):
        c, z, g = q[:N], q[N:2 * N], np.zeros(K + 1)
        g[free] = q[2 * N:]
        B = _basis_matrix(z[oi] - oz, M, h, interp, convention)
        return c[oi] + B @ g

    B0 = _basis_matrix(z0[oi] - oz, M, h, interp, convention)
    A = np.zeros((oi.size, N + free.size))
    A[np.arange(oi.size), oi] = 1.0
    A[:, N:] = B0[:, free]
    lb = np.r_[np.full(N, -np.inf), np.full(free.size, 0.0 if convention == "min_origin" else -np.inf)]
    lin = optimize.lsq_linear(sw[:, None] * A, sw * ow, bounds=(lb, np.full(lb.size, np.inf)), tol=1e-14,
                              method="bvls")
    q0 = np.r_[lin.x[:N], z0, lin.x[N:]]
    lo = np.r_[np.full(N, -np.inf), z0 - z_bound, lb[N:]]
    hi = np.r_[np.full(N, np.inf), z0 + z_bound, np.full(free.size, np.inf)]
    q0 = np.clip(q0, lo, hi)
    r = optimize.least_squares(lambda q: sw * (model(q) - ow), q0, bounds=(lo, hi), method="trf", jac="3-point",
                               x_scale="jac", ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=5000)
    g = np.zeros(K + 1)
    g[free] = r.x[2 * N:]
    return dict(c=r.x[:N], z=r.x[N:2 * N], G=g, cost=float(r.cost), node_ids=[s.node_id for s in used],
                success=bool(r.success))


# ------------------------------------------------------------ Block 15 ------

def gauss_dip(v, a, vc, s):
    return a * np.exp(-0.5 * ((np.asarray(v, float) - vc) / s) ** 2)


def f1_omega(z_p, s_j, z_k, c_j, sigma0, gamma, d0=0.84, gamma2=0.0):
    """(F1) of 15.4: omega = c_j + (sigma0 + gamma x + gamma2 x^2)^2, x = |z_p + s_j - z_k| - d0."""
    x = np.abs(z_p + s_j - np.asarray(z_k, float)) - d0
    return c_j + (sigma0 + gamma * x + gamma2 * x * x) ** 2


def f1_scans(c, s, z_p, k_star, sigma0, gamma, P=16, amp=40.0, rms=1.0, node_ids=None, noise_sd=0.0, rng=None,
             gamma_per_node=None):
    """Far PlaneScans (set "far") over k*_j - P .. k*_j + P with omega exactly (F1)
    (+ noise), status ok everywhere, amp_gl = amp, rms_gl = rms."""
    rng = np.random.default_rng(1) if rng is None else rng
    out = []
    for j in range(len(c)):
        ks = int(k_star[j]) + np.arange(-P, P + 1)
        z = ks * DZ
        g = gamma if gamma_per_node is None else gamma_per_node[j]
        om = f1_omega(z_p[j], s[j], z, c[j], sigma0, g)
        if noise_sd:
            om = om + rng.normal(0.0, noise_sd, om.size)
        n = ks.size
        nid = j + 1 if node_ids is None else node_ids[j]
        out.append(make_scan(nid, k_star[j], z_p[j], om, ks=ks, z_um=z, which="far",
                             amp_gl=np.full(n, float(amp)), rms_gl=np.full(n, float(rms)),
                             se_omega_um2=np.full(n, 0.01)))
    return out


def ref_select_far(scans, cal, table_delta, table_sigma, slope):
    """15.3 from the spec: per node A_j = median over status-ok planes with
    |k - k*| in far_area_offsets of a sigma sqrt(2 pi) (sigma = sqrt(omega)),
    RMS_j = median RMS over the ok planes; K_j = {|k - k*| >= far_min_k_offset,
    ok, A_j / (sqrt(2 pi) sigma_p(k)) >= far_min_snr RMS_j}, sigma_p(k) =
    sigma_r(|z_p - z_k|); nodes without A_j or with fewer than
    far_min_planes_per_node planes dropped. Returns (kept: {node_id: (A, RMS,
    ks kept)}, dropped node ids, n_not_converged, n_snr_dropped)."""
    lo, hi = cal.far_area_offsets
    kept, dropped, n_nc, n_snr = {}, set(), 0, 0
    for sc in scans:
        st = np.array(sc.status)
        off = np.abs(np.asarray(sc.ks) - sc.k_star)
        ok = st == "ok"
        n_nc += int(np.sum(st == "not_converged"))
        area_pl = ok & (off >= lo) & (off <= hi) & np.isfinite(sc.omega_um2) & np.isfinite(sc.amp_gl)
        if not area_pl.any():
            dropped.add(int(sc.node_id))
            continue
        A = float(np.median(sc.amp_gl[area_pl] * np.sqrt(sc.omega_um2[area_pl]) * SQ2PI))
        rms = float(np.median(sc.rms_gl[ok]))
        sp = ref_sigma_r(sc.z_sub_um - np.asarray(sc.z_um), table_delta, table_sigma, slope)
        cand = ok & (off >= cal.far_min_k_offset) & np.isfinite(sc.omega_um2)
        keep = cand & (A / (SQ2PI * sp) >= cal.far_min_snr * rms)
        n_snr += int(np.sum(cand & ~keep))
        if keep.sum() < cal.far_min_planes_per_node:
            dropped.add(int(sc.node_id))
            continue
        kept[int(sc.node_id)] = (A, rms, np.asarray(sc.ks)[keep])
    return kept, dropped, n_nc, n_snr


def ref_core_fit_iterative(v, w, depth, v_peak, sigma_p, fwhm_factor, width_bounds, min_samples, max_passes,
                           stop_tol):
    """15.2 written from the spec: pass i fits a exp(-(v - c)^2 / (2 s^2)) by
    least squares to w on |v - c^(i-1)| <= fwhm_factor FWHM(s^(i-1)), with
    s^(0) = sigma_p, c^(0) = v_peak, s in width_bounds x sigma_p (a fit at a
    bound ends the plane: at_bound), a >= 0, c free (the spec's 'within half
    the window of its start' never binds on these data); after pass i >= 2 the
    loop stops when |s^(i) - s^(i-1)| <= stop_tol SE(s^(i)), SE(s) = RMS
    [(J^T J)^-1]^(1/2)_ss, RMS^2 = sum (fit - w)^2 / (n_win - 3); no stop by
    max_passes: not_converged. Returns dict(status, omega, se_omega, a, c, s,
    rms, n_win, passes)."""
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    slo, shi = width_bounds[0] * sigma_p, width_bounds[1] * sigma_p
    c_prev, s_prev = float(v_peak), float(sigma_p)
    out = dict(status="ok")
    for i in range(1, max_passes + 1):
        win = np.abs(v - c_prev) <= fwhm_factor * FW * s_prev
        vv, ww = v[win], w[win]
        out.update(n_win=int(win.sum()), passes=i)
        if vv.size < min_samples:
            out["status"] = "few_samples"
            return out

        def res(q):
            return q[0] * np.exp(-0.5 * ((vv - q[1]) / q[2]) ** 2) - ww
        q0 = [max(depth, 1e-3), c_prev, min(max(s_prev, slo * 1.01), shi * 0.99)]
        r = optimize.least_squares(res, q0, bounds=([0.0, -np.inf, slo], [np.inf, np.inf, shi]), method="trf",
                                   x_scale="jac", ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=1000)
        a, c, s = r.x
        e = np.exp(-0.5 * ((vv - c) / s) ** 2)
        J = np.column_stack([e, a * e * (vv - c) / s ** 2, a * e * (vv - c) ** 2 / s ** 3])
        rms = math.sqrt(float(np.sum(res(r.x) ** 2)) / (vv.size - 3))
        se_s = rms * math.sqrt(float(np.linalg.inv(J.T @ J)[2, 2]))
        out.update(a=a, c=c, s=s, rms=rms, omega=s * s, se_omega=2 * s * se_s)
        if abs(s - slo) <= 1e-6 * slo or abs(s - shi) <= 1e-6 * shi:
            out["status"] = "at_bound"
            return out
        if i >= 2 and abs(s - s_prev) <= stop_tol * se_s:
            return out
        c_prev, s_prev = c, s
    out["status"] = "not_converged"
    return out


def ref_se_model(amp, sigma, rms, step, half):
    """15.4's SE_m of omega = s^2 for one plane: a Gaussian of amplitude amp
    and SD sigma sampled every step on +-FWHM(sigma) (cut at half) about its
    centre, white noise rms: 2 sigma rms [(J^T J)^-1]^(1/2)_ss, J the
    derivatives of a exp(-(v - c)^2 / (2 s^2)) in (a, c, s)."""
    W = min(FW * sigma, half)
    m = int(math.floor(W / step + 1e-12))
    u = np.arange(-m, m + 1) * step
    e = np.exp(-0.5 * (u / sigma) ** 2)
    J = np.column_stack([e, amp * e * u / sigma ** 2, amp * e * u ** 2 / sigma ** 3])
    cov = np.linalg.inv(J.T @ J) * rms ** 2
    return 2.0 * sigma * math.sqrt(cov[2, 2])


# ------------------------------------------------------------ Block 19 ------

def ref_pool(per_cell, per_cell_cap, pool_cap):
    """19.4 from the spec: per set, the cells in increasing specimen id, each
    cell's first per_cell_cap scans of the set in file order, taken
    round-robin (every cell's first, then every cell's second, ...) up to
    pool_cap in all (None: no cap). Returns {set: [(specimen, node_id), ...]}."""
    out = {}
    for which in ("near", "far"):
        lists = [(int(s), [x for x in scans if x.set == which][:per_cell_cap]) for s, scans in
                 sorted(per_cell, key=lambda t: int(t[0]))]
        taken = []
        depth = 0
        longest = max([len(x) for _, x in lists] + [0])
        while depth < longest:
            for s, xs in lists:
                if depth < len(xs):
                    taken.append((s, int(xs[depth].node_id)))
            depth += 1
        if pool_cap is not None:
            taken = taken[:pool_cap]
        out[which] = taken
    return out


def ref_anova(values, labels):
    """One-way random-effects ANOVA (textbook method of moments, unequal sizes):
    dict(C, N, ms_w, ms_b, n0, sigma_b2, sigma_w2, means {label: mean}, sizes)."""
    values = np.asarray(values, dtype=float)
    labs = list(labels)
    cells = []
    for lab in labs:
        if lab not in cells:
            cells.append(lab)
    groups = {c: values[[lab == c for lab in labs]] for c in cells}
    C, N = len(cells), values.size
    grand = values.mean()
    ms_w = sum(((g - g.mean()) ** 2).sum() for g in groups.values()) / (N - C) if N > C else float("nan")
    if C >= 2:
        ms_b = sum(g.size * (g.mean() - grand) ** 2 for g in groups.values()) / (C - 1)
        n0 = (N - sum(g.size ** 2 for g in groups.values()) / N) / (C - 1)
        sb2 = max(0.0, (ms_b - ms_w) / n0) if math.isfinite(ms_w) else float("nan")
    else:
        ms_b = n0 = sb2 = float("nan")
    return dict(C=C, N=N, ms_w=ms_w, ms_b=ms_b, n0=n0, sigma_b2=sb2, sigma_w2=ms_w,
                means={c: float(g.mean()) for c, g in groups.items()}, sizes={c: g.size for c, g in groups.items()})


def ref_cell_z(values, labels):
    """19.8: for each cell c, the ANOVA of the OTHER cells' node values and the
    unweighted mean of their cell means m_ref; z_c = (m_c - m_ref) /
    sqrt(sigma_b^2 + sigma_w^2 / N_c), p_c = 2 Phi(-|z_c|). Returns {cell:
    (z, p)}; NaN where the yardstick cannot be formed."""
    values = np.asarray(values, dtype=float)
    labs = list(labels)
    full = ref_anova(values, labs)
    out = {}
    for c in full["means"]:
        mine = np.array([lab == c for lab in labs])
        oth = ref_anova(values[~mine], [lab for lab, m in zip(labs, mine) if not m])
        if oth["C"] < 2 or not math.isfinite(oth["sigma_w2"]):
            out[c] = (float("nan"), float("nan"))
            continue
        m_ref = float(np.mean(list(oth["means"].values())))
        var = oth["sigma_b2"] + oth["sigma_w2"] / full["sizes"][c]
        if not var > 0:
            out[c] = (float("nan"), float("nan"))
            continue
        z = (full["means"][c] - m_ref) / math.sqrt(var)
        out[c] = (z, float(2.0 * stats.norm.sf(abs(z))))
    return out
