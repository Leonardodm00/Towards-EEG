"""The bias table b(x | C, S) -- Block 14 (14.3, 14.4) in specs/SPEC.md
(procedure s.3.8; D-057.1, D-060, D-062, D-063, D-067.3, D-071).

Rows (Block 14.2): draw g = 1..G, realization r = 1..n_real, the ratio
y = d_hat / d_g. Coordinates (D-063.1, D-062.5):

  a row in S         x = (ln(d_g / um), phi, ln(mu_hat d_hat)); phi the true
                     tilt phi_g (table_phi_axis "true") or the line-fit tilt
                     (table_phi_axis "measured", a labelled comparison)
  a counted row      x_fail = (ln d_g, phi_g, ln(mu_g d_g)): the true values,
                     which every row has; every row is counted except those a
                     screen of failure_rate_ignore rejected (D-057.1)

The estimator (response_estimator "local_linear", D-063.2): with the Gaussian
weight w_n(x) = exp(-sum_k (x_nk - x_k)^2 / (2 h_k^2)), h = c (h1, h2, h3)
(tau_kernel_bandwidth in ln d, deg, ln(mu d); the multiplier c chosen by
k-fold cross-validation with folds by draw over bandwidth_cv_grid, D-071.1),
at each point x of a regular grid (spacing at most c h_k / 2 on axis k,
D-071.3) the weighted least squares of y on z_n = (1, (x_n - x) / h) over the
rows in S: A(x) beta = u(x), A = sum w z z', u = sum w z y, b_hat(x) =
beta_0. Every kernel sum over the grid is exact: the weight is a product over
the axes, so a sum over the rows is a product of per-axis factor matrices,
computed in row chunks by matrix products (no truncation of the kernel).

  SE (D-063.3, D-071.2)  the sandwich by draw,
                         Var b_hat(x) = a' [sum_g v_g v_g'] a, a = A^-1 e1,
                         v_g = sum_r w_gr z_gr e_gr, e = y - b_hat(x_n)
  independent rows       tau_hat / sqrt(N_eff) (labelled), tau_hat^2 =
                         sum w e^2 / sum w * N_eff / (N_eff - 1), N_eff =
                         (sum w)^2 / sum w^2 (rows in S)
  mass, failure rate     M(x) = sum w, p_fail(x) = sum w [n not in S] / M(x)
                         over the counted rows at x_fail
  q_loo                  the loo_mass_percentile-th percentile of the counted
                         rows' leave-one-out masses M(x_fail_n) - 1 (D-063.4)

b_hat at the rows (the residuals) and M at the rows read the interpolants.
Under table_phi_axis "true" the rows of a draw share their first two
coordinates, so v_g factors into the draw's two one-dimensional factors and two
sums over its realizations on the third axis; under "measured" the sandwich
runs over the pairs (r, r') of rows of each draw (both exact).

The support floor: where the reciprocal condition number of A is below
RCOND_MIN or N_eff is below min_neff, the arrays of the rows in S (b_hat, the
two SEs, tau_hat, N_eff) are NaN, and their interpolants read the nearest
supported grid value; the mass and the failure rate are kept. Queries read
tensor cubic interpolants of the grid arrays (RegularGridInterpolator,
method "cubic", solver rtol INTERP_RTOL); NaN outside the grid.

"tps_spline" (D-067.3: the small-table code path of the fixtures, never the
deliverable): the smoothing thin-plate spline through the rows in S on the
coordinates range-normalised by the grid's axes (D-054.1), the smoothing by
k-fold cross-validation with folds by draw over spline_smoothing_grid
(D-062.3-4), evaluated on the grid at c = 1; its SE is the independent-rows
value. It refuses more than tps_max_rows rows in S.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator

ESTIMATORS_BUILT = ("local_linear", "tps_spline")
SE_METHODS = {"local_linear": "sandwich_by_draw", "tps_spline": "independent_rows"}
ARRAYS = ("b_hat", "se_hat", "se_indep", "tau_hat", "n_eff", "mass", "p_fail")
S_ARRAYS = ("b_hat", "se_hat", "se_indep", "tau_hat", "n_eff")      # NaN off the support floor
RCOND_MIN = 1e-12                    # PROVISIONAL (Block 14.4): A(x) closer to singular holds no estimate
AXIS_ROW_PERCENTILES = (0.1, 99.9)   # PROVISIONAL (Block 14.4): the rows' share of an axis' range, outliers left out
MAX_GRID_POINTS = 4000000            # a guard on the grid's size (about 30 MB per array)
CHUNK_DOUBLES = 4000000              # the size of one chunk's product matrix (32 MB)
INTERP_RTOL = 1e-13                  # the cubic interpolant's solver tolerance: at SciPy's default (1e-6) a grid of
#                                      41 x 37 x 37 values is reproduced only to about 2.5e-5 (checked, SciPy 1.18.1)

# the kernel sums of the local linear fit: (axis-1 factor, axis-2 factor, axis-3 factor, row field); the factor "e"
# is exp(-u^2/2), "e1" e u, "e2" e u^2, "q" e^2 (the squared weight), u = (x_nk - x_k) / h_k
LL_SPECS = (("e", "e", "e", None), ("e1", "e", "e", None), ("e", "e1", "e", None), ("e", "e", "e1", None),
            ("e2", "e", "e", None), ("e", "e2", "e", None), ("e", "e", "e2", None),
            ("e1", "e1", "e", None), ("e1", "e", "e1", None), ("e", "e1", "e1", None),
            ("e", "e", "e", "y"), ("e1", "e", "e", "y"), ("e", "e1", "e", "y"), ("e", "e", "e1", "y"),
            ("q", "q", "q", None))
# the ten entries (p, q), p <= q, of B = sum_g v_g v_g', z = (1, u1, u2, u3)
B_ENTRIES = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 2), (1, 2), (1, 3), (2, 3), (3, 3))


# ------------------------------------------------------------------ rows --

def table_inputs(d_um, phi_rad, mu_hat_per_um, d_hat_um):
    """(n, 3) table coordinates (ln(d / um), phi rad, ln(mu_hat d_hat)) (broadcasting; Block 14.3, D-063.1).
    ValueError unless d is finite and > 0, phi finite and mu_hat d_hat finite and > 0. The failure count's
    coordinates of a draw are table_inputs(d_g, phi_g, mu_g, d_g)."""
    d, phi, mu, dh = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in
                                           (d_um, phi_rad, mu_hat_per_um, d_hat_um)))
    md = mu * dh
    with np.errstate(invalid="ignore"):
        good = np.isfinite(d) & (d > 0) & np.isfinite(phi) & np.isfinite(md) & (md > 0)
    if not np.all(good):
        raise ValueError("table inputs need finite d > 0, finite phi and finite mu_hat d_hat > 0")
    return np.column_stack([np.log(d.ravel()), phi.ravel(), np.log(md.ravel())])


@dataclass(frozen=True)
class TableRows:
    """The rows of one table fit as arrays (table_rows)."""

    X: np.ndarray          # (n_S, 3) coordinates of the rows in S
    y: np.ndarray          # (n_S,) their ratios d_hat / d
    draw: np.ndarray       # (n_S,) their draw index g
    X_fail: np.ndarray     # (n_c, 3) failure-count coordinates of the counted rows
    failed: np.ndarray     # (n_c,) bool: the counted row is outside S
    n_rows: int            # every row given
    n_draws_rows: int      # distinct draws among them


def _column(rows, key, default=None):
    """Column `key` of the rows -- a list of dicts, or a dict of equal-length columns (a merge reading only the
    columns it needs) -- as an array; KeyError when it is missing and has no default."""
    if isinstance(rows, dict):
        if key in rows:
            return np.asarray(rows[key])
        if default is None:
            raise KeyError(key)
        return np.full(len(next(iter(rows.values()))), default, dtype=object)
    if default is None:
        return np.array([r[key] for r in rows])
    return np.array([r.get(key, default) for r in rows], dtype=object)


def n_rows_of(rows):
    """The number of rows of a list of dicts or of a dict of columns."""
    return len(next(iter(rows.values()))) if isinstance(rows, dict) else len(rows)


def _flag(values):
    """bools from bools or their CSV text ("True"/"False")."""
    return np.array([v if isinstance(v, (bool, np.bool_)) else str(v) == "True" for v in values], dtype=bool)


def _row_arrays(rows, cfg):
    """The columns of the rows the table reads, the selection ok (in S with a finite ratio, tilt and a finite
    mu_hat d_hat > 0) and the counted rows (failure_rate_ignore, D-057.1)."""
    c = cfg.correction

    def col(key):
        return _column(rows, key).astype(float)
    g = _column(rows, "draw_index").astype(np.int64)
    d, mu, phi_true = col("d_um"), col("mu_per_um"), col("phi_rad")
    phi = phi_true if c.table_phi_axis == "true" else col("meas_phi_rad")
    y, d_hat, mu_hat = col("ratio"), col("d_hat_um"), col("mu_hat_per_um")
    with np.errstate(invalid="ignore"):
        md = mu_hat * d_hat
        ok = _flag(_column(rows, "in_S")) & np.isfinite(y) & np.isfinite(phi) & np.isfinite(md) & (md > 0)
    ignore = set(c.failure_rate_ignore)
    if ignore:
        reasons = [set(x for x in str(v).split(";") if x) for v in _column(rows, "reject", "")]
        counted = np.array([bool(k) or not (rs & ignore) for k, rs in zip(ok, reasons)], dtype=bool)
    else:
        counted = np.ones(ok.shape, dtype=bool)
    return dict(g=g, d=d, mu=mu, phi_true=phi_true, phi=phi, y=y, d_hat=d_hat, mu_hat=mu_hat, ok=ok,
                counted=counted)


def table_rows(rows, cfg):
    """TableRows from replicate rows -- a list of dicts, or a dict of columns -- with draw_index, d_um, phi_rad
    (and meas_phi_rad under table_phi_axis "measured"), mu_per_um, d_hat_um, mu_hat_per_um, ratio, in_S and
    optionally reject, the ';'-joined reasons. A row in S without a finite mu_hat d_hat > 0 is counted as a
    failure."""
    n = n_rows_of(rows) if rows else 0
    if not n:
        raise ValueError("no rows")
    a = _row_arrays(rows, cfg)
    ok, cnt = a["ok"], a["counted"]
    X = (table_inputs(a["d"][ok], a["phi"][ok], a["mu_hat"][ok], a["d_hat"][ok]) if ok.any()
         else np.empty((0, 3)))
    X_fail = (table_inputs(a["d"][cnt], a["phi_true"][cnt], a["mu"][cnt], a["d"][cnt]) if cnt.any()
              else np.empty((0, 3)))
    return TableRows(X, a["y"][ok], a["g"][ok], X_fail, ~ok[cnt], n, int(np.unique(a["g"]).size))


def per_draw_table(rows, cfg):
    """The per-draw table (D-060.2): one dict per draw g, in order of g -- draw_index, d_um, phi_rad,
    theta_rad, mu_per_um, n_rows, n_S, y_mean and y_sd (the mean and SD, ddof 1, of its ratios in S; NaN
    without rows in S, and the SD with fewer than two), failure_share = 1 - n_S / n_rows. The selection is the
    table's (table_rows)."""
    a = _row_arrays(rows, cfg)
    theta = _column(rows, "theta_rad", float("nan")).astype(float)
    out = []
    for g in np.unique(a["g"]):
        idx = np.flatnonzero(a["g"] == g)
        ys = a["y"][idx][a["ok"][idx]]
        k = int(idx[0])
        out.append(dict(draw_index=int(g), d_um=float(a["d"][k]), phi_rad=float(a["phi_true"][k]),
                        theta_rad=float(theta[k]), mu_per_um=float(a["mu"][k]), n_rows=int(idx.size),
                        n_S=int(ys.size), y_mean=float(ys.mean()) if ys.size else float("nan"),
                        y_sd=float(ys.std(ddof=1)) if ys.size > 1 else float("nan"),
                        failure_share=1.0 - ys.size / float(idx.size)))
    return out


def fold_of_draw(draw, folds):
    """The cross-validation fold of draw g: (g - 1) mod folds, so the realizations of a draw share a fold
    (D-062.3)."""
    return (np.asarray(draw, dtype=np.int64) - 1) % int(folds)


# ------------------------------------------------------------------ grid --

def bandwidths(cfg, c=1.0):
    """h = c (h1, h2, h3) in the coordinates' units (ln d, rad, ln(mu d))."""
    h1, h2, h3 = cfg.correction.tau_kernel_bandwidth
    return (float(c) * h1, float(c) * math.radians(h2), float(c) * h3)


def design_ranges(cfg):
    """The design's range of each coordinate: [ln d_min, ln d_max], the tilt range (rad), and the band
    [ln(mu_min d_min), ln(mu_max d_max)]."""
    ph = cfg.phantom
    (dlo, dhi), (mlo, mhi) = ph.d_range_um, ph.mu_range_per_um
    return ((math.log(dlo), math.log(dhi)), (math.radians(ph.phi_range_deg[0]), math.radians(ph.phi_range_deg[1])),
            (math.log(mlo * dlo), math.log(mhi * dhi)))


def grid_axes(X, cfg, c=1.0):
    """The three regular axes at multiplier c (D-071.3): axis k spans the design's range of coordinate k joined
    with the AXIS_ROW_PERCENTILES range of the rows' coordinate k (X: (n, 3), the rows in S), with n_k =
    ceil(L_k / (c h_k / 2)) + 1 points (at least 4), so the spacing is at most c h_k / 2."""
    h = bandwidths(cfg, c)
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    axes = []
    for k, (lo, hi) in enumerate(design_ranges(cfg)):
        if X.shape[0]:
            qlo, qhi = np.percentile(X[:, k], AXIS_ROW_PERCENTILES)
            lo, hi = min(lo, float(qlo)), max(hi, float(qhi))
        if not hi > lo:
            raise ValueError("table axis %d has an empty range [%g, %g]" % (k + 1, lo, hi))
        n = max(4, int(math.ceil((hi - lo) / (0.5 * h[k]) - 1e-9)) + 1)
        axes.append(np.linspace(lo, hi, n))
    size = int(np.prod([a.size for a in axes]))
    if size > MAX_GRID_POINTS:
        raise ValueError("the table grid would hold %d points (axes %s) at multiplier %g, more than %d: check the "
                         "ranges" % (size, [a.size for a in axes], c, MAX_GRID_POINTS))
    return tuple(axes)


def clamp_to_grid(axes, X):
    """X (n, 3) with each coordinate clipped to its axis."""
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    return np.column_stack([np.clip(X[:, k], axes[k][0], axes[k][-1]) for k in range(3)])


# ---------------------------------------------------------- kernel sums --

def _factors(axis, s, h, kinds):
    """{kind: (len(axis), len(s))} for u = (s_n - axis_i) / h: "e" exp(-u^2/2), "e1" e u, "e2" e u^2, "q" e^2,
    "q1" e^2 u, "q2" e^2 u^2."""
    u = (np.asarray(s, dtype=float)[None, :] - np.asarray(axis, dtype=float)[:, None]) / h
    e = np.exp(-0.5 * u * u)
    out = {}
    for k in kinds:
        base = e if k[0] == "e" else e * e
        p = int(k[1:]) if len(k) > 1 else 0
        out[k] = base * u ** p if p else base
    return out


def _contract(F1, F2, F3, specs, fields, out):
    """out[s] += sum_n f_n F1[k1][i, n] F2[k2][j, n] F3[k3][l, n] for each spec s = (k1, k2, k3, f), f None or a
    key of fields; F1, F2, F3: dicts of (n_axis, m) matrices over the same m columns."""
    n1 = next(iter(F1.values())).shape[0]
    n2 = next(iter(F2.values())).shape[0]
    n3, m = next(iter(F3.values())).shape
    plan = {}
    for s, (k1, k2, k3, f) in enumerate(specs):
        plan.setdefault((k1, k2, f), []).append((s, k3))
    for (k1, k2, f), lst in plan.items():
        Q = F1[k1][:, None, :] * F2[k2][None, :, :]
        if f is not None:
            Q = Q * fields[f][None, None, :]
        R = np.concatenate([F3[k3] for _, k3 in lst], axis=0)
        res = (Q.reshape(n1 * n2, m) @ R.T).reshape(n1, n2, len(lst), n3)
        for t, (s, _) in enumerate(lst):
            out[s] += res[:, :, t, :]


def grid_sums(axes, h, X, specs, fields=None, groups=None, n_groups=1):
    """(n_groups, len(specs), n1, n2, n3) kernel sums over the rows X (n, 3): for spec (k1, k2, k3, f), at grid
    point (i, j, l), sum over the rows n of group q of fields[f][n] (1 when f is None) times the factors k1, k2,
    k3 of _factors on the three axes; groups (n,) ints in [0, n_groups) or None (one group)."""
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    n1, n2, n3 = (a.size for a in axes)
    out = np.zeros((n_groups, len(specs), n1, n2, n3))
    kinds = [{sp[k] for sp in specs} for k in range(3)]
    m = max(16, CHUNK_DOUBLES // (n1 * n2))
    gid = np.zeros(X.shape[0], dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
    for q in range(n_groups):
        rows = np.flatnonzero(gid == q)
        for a in range(0, rows.size, m):
            idx = rows[a:a + m]
            F = [_factors(axes[k], X[idx, k], h[k], kinds[k]) for k in range(3)]
            _contract(F[0], F[1], F[2], specs, {key: np.asarray(v)[idx] for key, v in (fields or {}).items()},
                      out[q])
    return out


# ------------------------------------------------------ local linear fit --

def _assemble(S):
    """(A (..., 4, 4), u (..., 4), sum w, sum w^2) from the 15 sums of LL_SPECS."""
    s000, s100, s010, s001, s200, s020, s002, s110, s101, s011, y0, y1, y2, y3, sq = S
    A = np.stack([np.stack([s000, s100, s010, s001], -1), np.stack([s100, s200, s110, s101], -1),
                  np.stack([s010, s110, s020, s011], -1), np.stack([s001, s101, s011, s002], -1)], -2)
    return A, np.stack([y0, y1, y2, y3], -1), s000, sq


def local_linear(S, min_neff):
    """(b_hat, a = A^-1 e1 (..., 4), N_eff, supported) on the grid from the 15 sums of LL_SPECS; b_hat and a
    NaN where unsupported (rcond(A) < RCOND_MIN or N_eff < min_neff)."""
    A, u, sw, sw2 = _assemble(S)
    shape = sw.shape
    A2, u2 = A.reshape(-1, 4, 4), u.reshape(-1, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        neff = sw ** 2 / sw2
        sv = np.linalg.svd(A2, compute_uv=False)
        rcond = sv[:, -1] / sv[:, 0]
    ok = np.isfinite(rcond) & (rcond >= RCOND_MIN) & np.isfinite(neff.ravel()) & (neff.ravel() >= min_neff)
    b, a = np.full(A2.shape[0], np.nan), np.full((A2.shape[0], 4), np.nan)
    if ok.any():
        rhs = np.zeros((int(ok.sum()), 4, 2))
        rhs[:, :, 0], rhs[:, 0, 1] = u2[ok], 1.0
        sol = np.linalg.solve(A2[ok], rhs)
        b[ok], a[ok] = sol[:, 0, 0], sol[:, :, 1]
    return b.reshape(shape), a.reshape(shape + (4,)), neff, ok.reshape(shape)


def filled(V, good):
    """V with every value where good is False replaced by the nearest good one (in grid index space, where the
    spacings are about c/2 bandwidths on every axis); ValueError when nothing is good."""
    good = np.asarray(good, dtype=bool)
    if good.all():
        return np.asarray(V, dtype=float)
    if not good.any():
        raise ValueError("no grid point holds a value")
    idx = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
    return np.asarray(V, dtype=float)[tuple(idx)]


def interpolant(axes, V):
    """The tensor cubic interpolant of grid values V (finite), NaN outside the grid."""
    return RegularGridInterpolator(tuple(axes), np.asarray(V, dtype=float), method="cubic", bounds_error=False,
                                   fill_value=np.nan, solver_args=dict(rtol=INTERP_RTOL, atol=0.0))


def _sorted_by_draw(X, e, draw):
    order = np.argsort(np.asarray(draw), kind="stable")
    X, e, draw = np.asarray(X)[order], np.asarray(e)[order], np.asarray(draw)[order]
    starts = np.flatnonzero(np.r_[True, draw[1:] != draw[:-1]])
    return X, e, draw, starts, np.r_[starts[1:], draw.size]


def sandwich_true(axes, h, X, e, draw):
    """(10, n1, n2, n3) entries B_pq (B_ENTRIES) of sum_g v_g v_g' when the rows of a draw share their first
    two coordinates (table_phi_axis "true"): v_g = e1_g e2_g (t_g, u1_g t_g, u2_g t_g, s_g) with t_g = sum_r
    e3 e_r and s_g = sum_r e3 u3 e_r on the third axis. ValueError when a draw's rows differ in x1 or x2."""
    X, e, draw, starts, stops = _sorted_by_draw(X, e, draw)
    first = np.repeat(starts, stops - starts)
    if np.any(X[:, :2] != X[first, :2]):
        raise ValueError("sandwich_true: the rows of a draw differ in ln d or phi (use sandwich_pairs)")
    n1, n2, n3 = (a.size for a in axes)
    specs = (("q", "q", "tt", None), ("q1", "q", "tt", None), ("q", "q1", "tt", None), ("q", "q", "ts", None),
             ("q2", "q", "tt", None), ("q", "q2", "tt", None), ("q1", "q1", "tt", None), ("q1", "q", "ts", None),
             ("q", "q1", "ts", None), ("q", "q", "ss", None))
    out = np.zeros((len(specs), n1, n2, n3))
    m = max(16, CHUNK_DOUBLES // (n1 * n2))
    for a in range(0, starts.size, m):
        st, sp = starts[a:a + m], stops[a:a + m]
        rows = np.arange(st[0], sp[-1])
        F3 = _factors(axes[2], X[rows, 2], h[2], ("e", "e1"))
        t = np.add.reduceat(F3["e"] * e[rows][None, :], st - st[0], axis=1)
        s = np.add.reduceat(F3["e1"] * e[rows][None, :], st - st[0], axis=1)
        F1 = _factors(axes[0], X[st, 0], h[0], ("q", "q1", "q2"))
        F2 = _factors(axes[1], X[st, 1], h[1], ("q", "q1", "q2"))
        _contract(F1, F2, {"tt": t * t, "ts": t * s, "ss": s * s}, specs, {}, out)
    return out


def sandwich_pairs(axes, h, X, e, draw):
    """(10, n1, n2, n3) entries B_pq (B_ENTRIES) of sum_g v_g v_g' by the pairs (r, r') of rows of each draw
    (any table_phi_axis): B_pq = sum_g sum_{r, r'} w_r z_p(r) e_r w_r' z_q(r') e_r'."""
    X, e, draw, starts, stops = _sorted_by_draw(X, e, draw)
    I = np.concatenate([np.repeat(np.arange(a, b), b - a) for a, b in zip(starts, stops)])
    J = np.concatenate([np.tile(np.arange(a, b), b - a) for a, b in zip(starts, stops)])
    specs = []
    for p, q in B_ENTRIES:
        kinds = ["00", "00", "00"]
        if p and q and p != q:
            kinds[p - 1], kinds[q - 1] = "10", "01"
        elif p and p == q:
            kinds[p - 1] = "11"
        elif q:
            kinds[q - 1] = "01"
        specs.append(tuple(kinds) + ("ee",))
    n1, n2, n3 = (a.size for a in axes)
    out = np.zeros((len(specs), n1, n2, n3))
    m = max(16, CHUNK_DOUBLES // (n1 * n2))
    for a in range(0, I.size, m):
        i, j = I[a:a + m], J[a:a + m]
        F = []
        for k in range(3):
            fi = _factors(axes[k], X[i, k], h[k], ("e", "e1"))
            fj = _factors(axes[k], X[j, k], h[k], ("e", "e1"))
            F.append({"00": fi["e"] * fj["e"], "10": fi["e1"] * fj["e"], "01": fi["e"] * fj["e1"],
                      "11": fi["e1"] * fj["e1"]})
        _contract(F[0], F[1], F[2], specs, {"ee": e[i] * e[j]}, out)
    return out


def sandwich_variance(B, a):
    """a' B a on the grid, B the B_ENTRIES of a symmetric 4 x 4 matrix, a (..., 4); clipped at 0."""
    var = np.zeros(a.shape[:-1])
    for t, (p, q) in enumerate(B_ENTRIES):
        var = var + (1.0 if p == q else 2.0) * a[..., p] * a[..., q] * B[t]
    return np.maximum(var, 0.0)


def choose_multiplier(R, cfg):
    """(c, CV scores, at the grid's edge) by k-fold cross-validation with folds by draw (D-071.1, D-062.3): for
    each c of bandwidth_cv_grid and fold f, the local linear table of the other folds' rows in S (on the grid at
    c, with the support floor; its sums are the total less fold f's) read by its interpolant at fold f's rows
    (clamped to the grid); the score is the mean squared error over all rows in S, the smallest wins. One grid
    value fixes c (no scores)."""
    c = cfg.correction
    grid = tuple(float(x) for x in c.bandwidth_cv_grid)
    if len(grid) == 1:
        return grid[0], np.empty(0), False
    K = int(c.spline_cv_folds)
    fold = fold_of_draw(R.draw, K)
    if np.unique(fold).size < K:
        raise ValueError("cross-validation needs rows in S from at least %d draws, got %d"
                         % (K, np.unique(R.draw).size))
    scores = []
    for mult in grid:
        axes, h = grid_axes(R.X, cfg, mult), bandwidths(cfg, mult)
        S = grid_sums(axes, h, R.X, LL_SPECS, {"y": R.y}, groups=fold, n_groups=K)
        total, err = S.sum(0), 0.0
        for f in range(K):
            b, _, _, ok = local_linear(total - S[f], c.min_neff)
            if not ok.any():
                err = math.inf
                break
            held = fold == f
            pred = interpolant(axes, filled(b, ok))(clamp_to_grid(axes, R.X[held]))
            err += float(np.sum((R.y[held] - pred) ** 2))
        scores.append(err / R.y.size)
    scores = np.array(scores)
    k = int(np.argmin(scores))
    if not math.isfinite(scores[k]):
        raise ValueError("no bandwidth multiplier gives a finite cross-validation error (too few rows in S)")
    return grid[k], scores, k in (0, len(grid) - 1)


def choose_smoothing(X, y, draw, grid, folds):
    """(smoothing, CV mean squared error per grid value, at the grid's edge) of the thin-plate spline by k-fold
    CV with folds by draw (D-062.3); a value whose fit fails (singular system) scores inf; the first minimum
    wins; a choice at either end of a grid of two or more values is flagged (D-062.4)."""
    fold = fold_of_draw(draw, folds)
    if np.unique(fold).size < folds or y.size < 2 * folds:
        raise ValueError("cross-validation needs rows in S from at least %d draws and %d rows, got %d draws, %d "
                         "rows" % (folds, 2 * folds, np.unique(draw).size, y.size))
    scores = []
    for lam in grid:
        err = 0.0
        try:
            for f in range(folds):
                tr = fold != f
                err += float(np.sum((RBFInterpolator(X[tr], y[tr], kernel="thin_plate_spline",
                                                     smoothing=float(lam))(X[~tr]) - y[~tr]) ** 2))
        except np.linalg.LinAlgError:
            err = math.inf
        scores.append(err / y.size)
    k = int(np.argmin(scores))
    if not math.isfinite(scores[k]):
        raise ValueError("no smoothing value gives a finite cross-validation error")
    return float(grid[k]), np.array(scores), len(grid) > 1 and k in (0, len(grid) - 1)


# ------------------------------------------------------------- the table --

class BiasTable:
    """The table of one configuration C on its grid (Block 14.4): the axes (ln d, phi rad, ln(mu d)), the arrays
    of ARRAYS, the support mask and the record (meta: the estimator, the bandwidths and multiplier, the CV, q_loo,
    the counts, the signature, the estimator hash). Queries take d (um), phi (rad) and mu_d, the node's fitted
    optical depth mu_hat d_hat (dimensionless), broadcasting; NaN outside the grid."""

    def __init__(self, axes, arrays, supported, meta):
        self.axes = tuple(np.asarray(a, dtype=float) for a in axes)
        shape = tuple(a.size for a in self.axes)
        self.arrays = {k: np.asarray(arrays[k], dtype=float) for k in ARRAYS}
        self.supported = np.asarray(supported, dtype=bool)
        if any(v.shape != shape for v in self.arrays.values()) or self.supported.shape != shape:
            raise ValueError("table arrays do not match the axes %s" % (shape,))
        self.meta = dict(meta)
        self.estimator_hash = str(self.meta["estimator_hash"])
        self.signature = self.meta.get("signature")
        self.q_loo = float(self.meta["q_loo"])
        self._interp = {}
        for k in ARRAYS:
            v = self.arrays[k]
            good = np.isfinite(v) & (self.supported if k in S_ARRAYS else True)
            self._interp[k] = interpolant(self.axes, filled(v, good))
        self._support = RegularGridInterpolator(self.axes, self.supported.astype(float), method="linear",
                                                bounds_error=False, fill_value=0.0)

    # ranges of the axes
    @property
    def d_range(self):
        return (float(math.exp(self.axes[0][0])), float(math.exp(self.axes[0][-1])))

    @property
    def phi_range_rad(self):
        return (float(self.axes[1][0]), float(self.axes[1][-1]))

    @property
    def mu_d_range(self):
        return (float(math.exp(self.axes[2][0])), float(math.exp(self.axes[2][-1])))

    def _query(self, d_um, phi_rad, mu_d):
        d, p, m = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (d_um, phi_rad, mu_d)))
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = np.column_stack([np.log(d.ravel()), p.ravel(), np.log(m.ravel())])
        return Q, np.isfinite(Q).all(axis=1), d.shape

    def at(self, name, d_um, phi_rad, mu_d):
        """The interpolated array `name` (ARRAYS) at the query points."""
        Q, fin, shape = self._query(d_um, phi_rad, mu_d)
        out = np.full(Q.shape[0], np.nan)
        if fin.any():
            out[fin] = self._interp[name](Q[fin])
        return out.reshape(shape)

    def b_hat(self, d_um, phi_rad, mu_d):
        return self.at("b_hat", d_um, phi_rad, mu_d)

    def m_hat(self, d_um, phi_rad, mu_d):
        return np.asarray(d_um, dtype=float) * self.b_hat(d_um, phi_rad, mu_d)

    def se_hat(self, d_um, phi_rad, mu_d):
        """The reported SE (meta["se_method"]: the sandwich by draw, or the independent-rows value)."""
        return self.at("se_hat", d_um, phi_rad, mu_d)

    def se_indep(self, d_um, phi_rad, mu_d):
        return self.at("se_indep", d_um, phi_rad, mu_d)

    def tau_hat(self, d_um, phi_rad, mu_d):
        return self.at("tau_hat", d_um, phi_rad, mu_d)

    def n_eff(self, d_um, phi_rad, mu_d):
        return self.at("n_eff", d_um, phi_rad, mu_d)

    def mass(self, d_um, phi_rad, mu_d):
        return np.maximum(self.at("mass", d_um, phi_rad, mu_d), 0.0)

    def failure_rate(self, d_um, phi_rad, mu_d):
        return np.clip(self.at("p_fail", d_um, phi_rad, mu_d), 0.0, 1.0)

    def support(self, d_um, phi_rad, mu_d):
        """The trilinear interpolant of the support mask: 1 where the eight grid points around the query are all
        supported, less otherwise, 0 outside the grid."""
        Q, fin, shape = self._query(d_um, phi_rad, mu_d)
        out = np.zeros(Q.shape[0])
        if fin.any():
            out[fin] = self._support(Q[fin])
        return out.reshape(shape)

    def check_estimator(self, cfg):
        """ValueError unless cfg's estimator signature is the table's (procedure s.3.11)."""
        h = cfg.signature_hash("estimator")
        if h != self.estimator_hash:
            raise ValueError("bias table built for estimator %s, the fit uses %s: rebuild the table or use the "
                             "matching configuration" % (self.estimator_hash, h))


SIM_PREFIX = "sim-"       # keeps the hash a string in the CSV (table_io parses numeric-looking cells)


def row_provenance(cfg):
    """The value of the sim_hash column of a replicate row rendered under cfg."""
    return SIM_PREFIX + cfg.signature_hash("simulation")


def check_rows_config(rows, cfg):
    """ValueError unless every replicate row that carries a sim_hash was rendered
    under cfg's simulation settings (build_table.run writes the column since
    2026-10-07; the correction settings may differ: only the merge reads them).
    Returns the number of rows without the column (older runs: not checked).
    rows: a list of dicts or a dict of columns (Block 14.6)."""
    want = row_provenance(cfg)
    values = [str(v) for v in _column(rows, "sim_hash", "")]
    blank = ("", "None", "nan")
    seen = {v for v in values if v not in blank}
    if seen - {want}:
        raise ValueError("the rows were rendered under %s, the merge was given %s: pass the --config-json the "
                         "run used (without it the default configuration is assumed)" % (sorted(seen), want))
    return sum(1 for v in values if v in blank)


def build_table(R, cfg, estimator, c, cv=None):
    """BiasTable of the rows R (TableRows) at multiplier c: the fit's b_hat on the grid (local_linear, or the
    tps_spline of cv["spline"]), then the residuals, tau_hat, N_eff, the SEs, the mass, the failure rate and
    q_loo (see the module docstring). cv: the record of the choice (cv_parameter, cv_grid, cv_scores,
    cv_at_grid_edge, and for tps_spline smoothing and spline)."""
    cc = cfg.correction
    cv = dict(cv or {})
    axes, h = grid_axes(R.X, cfg, c), bandwidths(cfg, c)
    S = grid_sums(axes, h, R.X, LL_SPECS, {"y": R.y})[0]
    b, a, neff, ok = local_linear(S, cc.min_neff)
    if estimator == "tps_spline":
        lo = np.array([ax[0] for ax in axes])
        span = np.array([ax[-1] - ax[0] for ax in axes])
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        b = cv["spline"]((pts - lo) / span).reshape(neff.shape)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(neff) & (neff >= cc.min_neff) & np.isfinite(b)
    if not ok.any():
        raise ValueError("no grid point has N_eff >= min_neff (%g) and a regular local fit: too few rows in S "
                         "(%d)" % (cc.min_neff, R.y.size))
    b = np.where(ok, b, np.nan)
    e = R.y - interpolant(axes, filled(b, ok))(clamp_to_grid(axes, R.X))
    sw = S[0]
    sw_e2 = grid_sums(axes, h, R.X, (("e", "e", "e", "e2"),), {"e2": e * e})[0, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.sqrt(np.where(neff > 1, sw_e2 / sw * neff / (neff - 1), np.nan))
        se_indep = tau / np.sqrt(neff)
    if estimator == "local_linear":
        B = (sandwich_true if cc.table_phi_axis == "true" else sandwich_pairs)(axes, h, R.X, e, R.draw)
        se = np.sqrt(sandwich_variance(B, np.where(ok[..., None], a, 0.0)))
    else:
        se = se_indep
    F = grid_sums(axes, h, R.X_fail, (("e", "e", "e", None), ("e", "e", "e", "f")),
                  {"f": R.failed.astype(float)})[0]
    mass = F[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        p_fail = np.where(mass > 0, F[1] / mass, np.nan)
    m_rows = interpolant(axes, filled(mass, np.isfinite(mass)))(clamp_to_grid(axes, R.X_fail))
    q_loo = float(np.percentile(m_rows - 1.0, cc.loo_mass_percentile)) if m_rows.size else float("nan")
    arrays = dict(b_hat=b, se_hat=np.where(ok, se, np.nan), se_indep=np.where(ok, se_indep, np.nan),
                  tau_hat=np.where(ok, tau, np.nan), n_eff=np.where(ok, neff, np.nan), mass=mass, p_fail=p_fail)
    meta = dict(estimator=estimator, se_method=SE_METHODS[estimator],
                tau_kernel_bandwidth=list(cc.tau_kernel_bandwidth), multiplier=float(c), bandwidth=list(h),
                cv_parameter=cv.get("cv_parameter", ""), cv_grid=list(cv.get("cv_grid", ())),
                cv_scores=[float(x) for x in cv.get("cv_scores", ())],
                cv_at_grid_edge=bool(cv.get("cv_at_grid_edge", False)), cv_folds=int(cc.spline_cv_folds),
                smoothing=cv.get("smoothing"), min_neff=float(cc.min_neff), rcond_min=RCOND_MIN,
                axis_row_percentiles=list(AXIS_ROW_PERCENTILES), loo_mass_percentile=float(cc.loo_mass_percentile),
                q_loo=q_loo, table_phi_axis=cc.table_phi_axis, grid_shape=[ax.size for ax in axes],
                n_supported=int(ok.sum()), n_all=int(R.n_rows), n_kept=int(R.y.size),
                n_failure_counted=int(R.X_fail.shape[0]), n_draws=int(R.n_draws_rows),
                n_draws_kept=int(np.unique(R.draw).size), signature=cfg.full_signature(),
                estimator_hash=cfg.signature_hash("estimator"))
    return BiasTable(axes, arrays, ok, meta)


def fit_table(rows, cfg):
    """BiasTable from replicate rows (table_rows) under cfg.correction: response_estimator "local_linear" (the
    multiplier c by choose_multiplier) or "tps_spline" (at most tps_max_rows rows in S; the smoothing by
    choose_smoothing when spline_smoothing is "cv", else the given float); table_statistic "mean". Any other
    choice raises NotImplementedError."""
    c = cfg.correction
    if c.response_estimator not in ESTIMATORS_BUILT or c.table_statistic != "mean":
        raise NotImplementedError("response_estimator %r / table_statistic %r: only %s / mean are built (Block 14)"
                                  % (c.response_estimator, c.table_statistic, " or ".join(ESTIMATORS_BUILT)))
    R = table_rows(rows, cfg)
    if R.y.size < 4:
        raise ValueError("a table needs at least 4 rows in S, got %d" % R.y.size)
    if c.response_estimator == "local_linear":
        mult, scores, edge = choose_multiplier(R, cfg)
        return build_table(R, cfg, "local_linear", mult,
                           dict(cv_parameter="bandwidth_multiplier" if scores.size else "",
                                cv_grid=c.bandwidth_cv_grid if scores.size else (), cv_scores=scores,
                                cv_at_grid_edge=edge))
    if R.y.size > c.tps_max_rows:
        raise ValueError("tps_spline: %d rows in S, more than tps_max_rows (%d): the dense spline is for small "
                         "tables (D-063.2, D-067.3); use local_linear" % (R.y.size, c.tps_max_rows))
    axes = grid_axes(R.X, cfg, 1.0)
    lo = np.array([ax[0] for ax in axes])
    span = np.array([ax[-1] - ax[0] for ax in axes])
    Xs = (R.X - lo) / span
    if c.spline_smoothing == "cv":
        lam, scores, edge = choose_smoothing(Xs, R.y, R.draw, c.spline_smoothing_grid, c.spline_cv_folds)
        grid = c.spline_smoothing_grid
    else:
        lam, scores, edge, grid = float(c.spline_smoothing), np.empty(0), False, ()
    spline = RBFInterpolator(Xs, R.y, kernel="thin_plate_spline", smoothing=lam)
    return build_table(R, cfg, "tps_spline", 1.0,
                       dict(cv_parameter="smoothing" if scores.size else "", cv_grid=grid, cv_scores=scores,
                            cv_at_grid_edge=edge, smoothing=lam, spline=spline))
