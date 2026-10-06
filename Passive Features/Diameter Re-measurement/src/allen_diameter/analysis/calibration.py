"""Defocus-kernel calibration, first stage -- Block 10 in specs/SPEC.md
(procedure Eq. 4 and s.3.4; mathematics Eq. 18 and s.3.5).

  plane_scan          at one thin, faint, flat calibration node the measuring
                      line is kept fixed in (x, y) and the planes
                      k* - P .. k* + P are stepped through (P = offsets_planes).
                      In plane k the dip w_k(v) = B_bar_k - I_k(v) (the Eq. 9
                      profile; B_bar_k by the node's own background rule in
                      plane k) gives omega_k = s_k^2, the squared width of a
                      Gaussian fitted to the dip's core (samples with
                      |v - v_peak| <= core_fit_fwhm_factor x FWHM), the
                      windowed variance V_W (cross-check) and the windowed
                      area A_W (diagnostic: the model keeps it constant
                      across planes, mathematics s.3.5)
  fit_growth          Eq. (4): omega_{i,k} = c_i + G(z_ax,i - z_k) + eps_{i,k},
                      G(0) = 0, jointly over the nodes; G is tabulated at the
                      plane offsets m * knot_step_um and interpolated between
                      them (cubic, or linear; linear extrapolation beyond the
                      last knot); convention "symmetric" (G even in delta) or
                      "min_origin" (two-sided knots, G >= 0, so that its
                      minimum sits at delta = 0)
  kernel_from_growth  sigma_r(delta_m) = sqrt(sigma_r(0)^2 + G(delta_m)), with
                      sigma_r(0) configured: the first-stage Gaussian table

delta = z_ax,i - z_k is object minus plane, as in the mathematics document.
The in-focus part is not identified (it cancels into c_i) and the curve only
up to a common shift, which the convention pins (mathematics s.3.5). Step 2
of procedure s.3.4 -- tuning sigma_r so that rendered thin phantoms
reproduce the mean real profile plane by plane -- is not in this module.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, lsq_linear

from . import cell, phantoms, profiles
from .fit import half_depth_width
from .node_pipeline import measure_node, node_background

STATUS_OK = "ok"


# ------------------------------------------------------------ one plane ---

def _core_fit(v, w, depth, v_peak, fwhm):
    """Least-squares Gaussian a exp(-(v - c)^2 / (2 s^2)) through the samples
    (v, w); returns (a, c, s) or None."""
    s0 = fwhm / math.sqrt(8.0 * math.log(2.0))

    def resid(q):
        return q[0] * np.exp(-0.5 * ((v - q[1]) / q[2]) ** 2) - w

    lo = [0.0, v_peak - fwhm, 1e-3 * s0]
    hi = [np.inf, v_peak + fwhm, 1e3 * s0]
    try:
        r = least_squares(resid, [depth, v_peak, s0], bounds=(lo, hi), method="trf", x_scale="jac",
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200)
    except ValueError:
        return None
    if r.status <= 0 or not np.all(np.isfinite(r.x)):
        return None
    return float(r.x[0]), float(r.x[1]), float(r.x[2])


def dip_statistics(v, I, B_bar, mcfg, cal):
    """Core width, windowed variance and area of one plane's dip.

    v (n,) um, increasing with a uniform step; I (n,) grey levels; B_bar > 0
    grey levels; mcfg a MeasureConfig, cal a CalibrationConfig. The peak
    v_peak is the smallest sample of I smoothed by focus_smooth_px samples
    among those inside the v0 bounds; FWHM is the half-depth width of the
    smoothed dip there (Block 3's half_depth_width). Returns a dict:
    omega_um2 (= s^2), vw_um2, area_gl_um, depth_gl (smoothed), v_c_um (the
    core centre, or v_peak when the core fit did not run), status ("ok",
    "no_dip", "no_fwhm", "few_samples", "fit_failed"). V_W and A_W are
    computed over |v - v_c| <= window_half_um whenever there is a dip."""
    v = np.asarray(v, dtype=float)
    I = np.asarray(I, dtype=float)
    nan = float("nan")
    out = dict(omega_um2=nan, vw_um2=nan, area_gl_um=nan, depth_gl=nan, v_c_um=nan, status=STATUS_OK)
    sm = ndimage.gaussian_filter1d(I, mcfg.focus_smooth_px, mode="nearest") if mcfg.focus_smooth_px > 0 else I
    lo, hi = mcfg.fit_v0_bounds_um
    cand = np.flatnonzero((v >= lo) & (v <= hi))
    if cand.size == 0:
        raise ValueError("dip_statistics: no sample inside the v0 bounds")
    n0 = int(cand[np.argmin(sm[cand])])
    depth = float(B_bar - sm[n0])
    out["depth_gl"] = depth
    if not depth > 0:
        out["status"] = "no_dip"
        return out
    w = B_bar - I
    v_c = float(v[n0])
    fwhm = half_depth_width(v, sm, B_bar, n0)
    if fwhm is None:
        out["status"] = "no_fwhm"
    else:
        win = np.abs(v - v[n0]) <= cal.core_fit_fwhm_factor * fwhm
        if int(win.sum()) < cal.min_window_samples:
            out["status"] = "few_samples"
        else:
            core = _core_fit(v[win], w[win], depth, float(v[n0]), fwhm)
            if core is None:
                out["status"] = "fit_failed"
            else:
                out["omega_um2"] = core[2] ** 2
                v_c = core[1]
    out["v_c_um"] = v_c
    win2 = np.abs(v - v_c) <= cal.window_half_um
    step = (v[-1] - v[0]) / (v.size - 1)
    w2, v2 = w[win2], v[win2]
    out["area_gl_um"] = float(np.sum(w2) * step)
    total = float(np.sum(w2))
    if total > 0:
        vbar = float(np.sum(v2 * w2)) / total
        out["vw_um2"] = float(np.sum((v2 - vbar) ** 2 * w2)) / total
    return out


# ------------------------------------------------------------ one node ----

@dataclass(frozen=True)
class PlaneScan:
    """The plane scan of one calibration node; arrays over the planes ks."""

    node_id: int
    d_hat_um: float
    phi_rad: float
    alpha_hat: float
    k_star: int
    z_sub_um: float
    ks: np.ndarray            # (n,) plane indices k* - P .. k* + P
    z_um: np.ndarray          # (n,) plane depths k dz (um)
    omega_um2: np.ndarray     # (n,) squared core width (NaN unless status ok)
    vw_um2: np.ndarray        # (n,) windowed variance
    area_gl_um: np.ndarray    # (n,) windowed area (grey levels x um)
    depth_gl: np.ndarray      # (n,) depth of the smoothed dip
    B_bar: np.ndarray         # (n,) background in plane k
    status: Tuple[str, ...]   # (n,) "ok" or the reason the plane has no omega

    def values(self, statistic):
        if statistic == "gaussian_core_width2":
            return self.omega_um2
        if statistic == "windowed_variance":
            return self.vw_um2
        raise ValueError("unknown statistic %r" % (statistic,))


def plane_scan(blk, node, branch, cfg):
    """Scan the planes k* - P .. k* + P of a measured node along its fixed line.

    blk = (block, ks, valid, frame) in the fetch_zblock contract; node a Block 5
    NodeResult with a sharpest plane (finite z_sub_um); branch the node's Branch (the background mask); cfg a
    DiameterConfig. The line: origin (cx, cy) um, measuring axis
    (-sin theta, cos theta), along-branch axis (cos theta, sin theta); the
    profile is sampled as the fit's (Eq. 9, along-branch averaging on)."""
    block, ks, valid, frame = blk
    m, cal, dz = cfg.measure, cfg.calibration, cfg.acquisition.dz_um
    # NodeResult.k_star is -1 when no plane was found, but -1 is also a plane of a phantom
    # stack (planes numbered about 0), so the test is on the sub-plane depth
    if not (math.isfinite(node.z_sub_um) and math.isfinite(node.cx_um) and math.isfinite(node.cy_um)):
        raise ValueError("plane_scan: the node has no sharpest plane")
    th = float(node.theta_rad)
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    o = np.array([node.cx_um, node.cy_um])
    v = profiles.profile_offsets(m)
    kk = np.arange(node.k_star - cal.offsets_planes, node.k_star + cal.offsets_planes + 1)
    n = kk.size
    cols = {name: np.full(n, np.nan) for name in ("omega_um2", "vw_um2", "area_gl_um", "depth_gl", "B_bar")}
    status = []
    ks = np.asarray(ks)
    for q, k in enumerate(kk):
        idx = int(k - ks[0])
        if not (0 <= idx < ks.size) or not bool(valid[idx]):
            status.append("invalid_plane")
            continue
        plane = block[idx]
        B, ok = node_background(plane, frame, o, branch, m)
        cols["B_bar"][q] = B
        if not ok or not (B > 0):
            status.append("bbar_few")
            continue
        I = profiles.sample_profile(plane, frame, o, y_hat, e_u, v, profiles.n_along(m), m.profile_step_um)
        if not np.all(np.isfinite(I)):
            status.append("profile_nan")
            continue
        st = dip_statistics(v, I, B, m, cal)
        for name in ("omega_um2", "vw_um2", "area_gl_um", "depth_gl"):
            cols[name][q] = st[name]
        status.append(st["status"])
    return PlaneScan(int(node.node_id), float(node.d_hat_um), float(node.phi_rad), float(node.alpha_hat),
                     int(node.k_star), float(node.z_sub_um), kk, kk * dz, cols["omega_um2"], cols["vw_um2"],
                     cols["area_gl_um"], cols["depth_gl"], cols["B_bar"], tuple(status))


def calibration_reasons(result, cfg):
    """Why a NodeResult is not a calibration node (empty: it is one): outside
    the selection S (Block 6's rule), or not thin, flat and faint."""
    cal = cfg.calibration
    out = phantoms.reject_reasons(result)
    if not result.d_hat_um <= cal.node_dhat_max_um:
        out.append("not_thin")
    if not result.phi_rad <= math.radians(cal.node_phi_max_deg):
        out.append("not_flat")
    if not result.alpha_hat <= cal.node_alpha_max:
        out.append("not_faint")
    return out


def scan_node(branch, i, provider, cfg, reg=None):
    """Measure node i (Block 5) and, when it is a calibration node, fetch its
    block over k* - P .. k* + P (a second provider call) and scan it.
    Returns (NodeResult, PlaneScan or None, reasons)."""
    res = measure_node(branch, i, provider, cfg, reg)
    reasons = calibration_reasons(res, cfg)
    if reasons:
        return res, None, reasons
    m, p, P = cfg.measure, cfg.acquisition.res0_um, cfg.calibration.offsets_planes
    left = int(math.floor((res.cx_um - m.block_half_um) / p)) - 1
    top = int(math.floor((res.cy_um - m.block_half_um) / p)) - 1
    width = int(math.ceil((res.cx_um + m.block_half_um) / p)) + 2 - left
    height = int(math.ceil((res.cy_um + m.block_half_um) / p)) + 2 - top
    blk = provider(left, top, width, height, res.k_star - P, res.k_star + P)
    return res, plane_scan(blk, res, branch, cfg), []


def scan_cell(swc, provider, cfg, transform=None, regs=None, candidates=None, log=None):
    """Plane scans of one cell's calibration nodes. Every dendrite stretch
    (Block 8) holding a node id of `candidates` (None: every dendrite node) is
    built in the image frame (transform: keyword arguments of cell.to_image_um)
    and each such node goes through scan_node (regs: {node_id: registration
    dict}). Returns a list of (node_id, NodeResult, PlaneScan or None, reasons)."""
    types = cfg.acquisition.dendrite_swc_types
    xyz_img = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um, **(transform or {}))
    out = []
    for run in cell.stretches(swc, types):
        ids = [int(swc.ids[r]) for r in run]
        if candidates is not None and not any(i in candidates for i in ids):
            continue
        branch, off = cell.stretch_branch(swc, run, xyz_img)
        for t, nid in enumerate(ids):
            if candidates is not None and nid not in candidates:
                continue
            res, scan, reasons = scan_node(branch, off + t, provider, cfg, (regs or {}).get(nid))
            out.append((nid, res, scan, reasons))
            if log is not None:
                log("node %d: %s" % (nid, "scanned" if scan is not None else "not a calibration node (%s)"
                                     % ";".join(reasons)))
    return out


def area_ratio_by_offset(scans):
    """{plane offset k - k*: median over nodes of A_W(k) / A_W(k*)} (diagnostic)."""
    ratios: Dict[int, List[float]] = {}
    for sc in scans:
        a0 = sc.area_gl_um[sc.ks == sc.k_star]
        if a0.size != 1 or not (a0[0] > 0):
            continue
        for k, a in zip(sc.ks, sc.area_gl_um):
            if np.isfinite(a):
                ratios.setdefault(int(k - sc.k_star), []).append(float(a / a0[0]))
    return {o: float(np.median(r)) for o, r in sorted(ratios.items())}


# ------------------------------------------------------------ the growth --

def _weights(delta, convention, h, M):
    """Knot index j, fraction t and d(abscissa)/d(delta) for linear
    interpolation (linear extrapolation beyond both ends) of the growth curve
    at delta (um). symmetric: abscissa |delta| on knots m h, m = 0..M;
    min_origin: abscissa delta on knots (m - M) h, m = 0..2M."""
    delta = np.asarray(delta, dtype=float)
    if convention == "symmetric":
        x, x0, K, dx = np.abs(delta), 0.0, M, np.sign(delta)
    else:
        x, x0, K, dx = delta, -M * h, 2 * M, np.ones_like(delta)
    u = (x - x0) / h
    j = np.clip(np.floor(u).astype(int), 0, K - 1)
    return j, u - j, dx


_CUBIC_CACHE: Dict[Tuple[str, float, int], list] = {}


def _cubic_basis_splines(convention, h, M):
    """One scipy CubicSpline (not-a-knot) per knot value: the spline through the
    unit vector of that knot on the signed knots -M h .. M h (symmetric: the unit
    is placed at +-m h, so every basis function is even)."""
    key = (convention, float(h), int(M))
    if key not in _CUBIC_CACHE:
        x = np.arange(-M, M + 1) * h
        out = []
        for m in range((M if convention == "symmetric" else 2 * M) + 1):
            y = np.zeros(2 * M + 1)
            if convention == "symmetric":
                y[M + m] = y[M - m] = 1.0
            else:
                y[m] = 1.0
            out.append(CubicSpline(x, y, bc_type="not-a-knot"))
        _CUBIC_CACHE[key] = out
    return _CUBIC_CACHE[key]


def _basis(delta, convention, h, M, interp):
    """(B (n, K + 1), dB/ddelta (n, K + 1)): the growth curve at delta is B @ G,
    G the knot values. Linear: piecewise-linear hat functions (in |delta| under
    the symmetric convention). Cubic: not-a-knot cubic splines through the knot
    values on the signed knots. Both extrapolate linearly beyond the end knots."""
    delta = np.asarray(delta, dtype=float).ravel()
    n = delta.size
    K = M if convention == "symmetric" else 2 * M
    B, dB = np.zeros((n, K + 1)), np.zeros((n, K + 1))
    if interp == "linear":
        j, t, dx = _weights(delta, convention, h, M)
        rows = np.arange(n)
        B[rows, j] = 1.0 - t
        B[rows, j + 1] += t
        dB[rows, j] -= dx / h
        dB[rows, j + 1] += dx / h
        return B, dB
    if interp != "cubic":
        raise ValueError("unknown growth interpolation %r" % (interp,))
    edge = M * h
    x = np.clip(delta, -edge, edge)
    for m, sp in enumerate(_cubic_basis_splines(convention, h, M)):
        d1 = sp(x, 1)
        B[:, m] = sp(x) + d1 * (delta - x)       # linear continuation outside [-M h, M h]
        dB[:, m] = d1
    return B, dB


@dataclass(frozen=True)
class GrowthFit:
    """Eq. (4) fitted jointly over the calibration nodes."""

    convention: str
    statistic: str
    interp: str
    knot_step_um: float
    knots_um: np.ndarray          # (K + 1,): m h (symmetric) or (m - M) h (min_origin)
    g_um2: np.ndarray             # (K + 1,): G at the knots; 0 at delta = 0
    node_ids: np.ndarray          # (N,) the nodes used
    c_um2: np.ndarray             # (N,)
    z_ax_um: np.ndarray           # (N,)
    obs_node: np.ndarray          # (n_obs,) index into node_ids
    obs_offset: np.ndarray        # (n_obs,) k - k*_i (planes)
    delta_um: np.ndarray          # (n_obs,) z_ax,i - z_k
    residual_um2: np.ndarray      # (n_obs,) model - omega
    n_obs_per_interval: np.ndarray  # (K,) observations whose abscissa falls in each knot interval
    dropped_nodes: Tuple[int, ...]  # nodes with fewer than min_planes_per_node finite values
    cost: float
    success: bool
    message: str
    nfev: int

    def growth(self, delta_um):
        """G(delta) (um^2) for every delta in the array, interpolated as fitted."""
        M = (self.knots_um.size - 1) if self.convention == "symmetric" else (self.knots_um.size - 1) // 2
        B, _ = _basis(delta_um, self.convention, self.knot_step_um, M, self.interp)
        return (B @ self.g_um2).reshape(np.shape(delta_um))

    def rms_by_offset(self):
        """{|k - k*|: RMS residual (um^2)} -- the residual check beyond +-trust_planes."""
        out = {}
        for o in sorted(set(np.abs(self.obs_offset).tolist())):
            r = self.residual_um2[np.abs(self.obs_offset) == o]
            out[int(o)] = float(np.sqrt(np.mean(r ** 2)))
        return out


def fit_growth(scans, cfg):
    """Fit Eq. (4) to the plane scans (cfg.calibration.statistic, convention,
    knot_step_um, z_ax_bound_um, min_planes_per_node).

    Start: z_ax,i = the node's sub-plane depth (k* dz when NaN); (c, G) by
    linear least squares at those depths (scipy lsq_linear; G >= 0 under
    min_origin); then all of (c, z_ax, G) by scipy least_squares (trf, the
    analytic Jacobian), z_ax,i within z_ax_bound_um of its start. Knots: m h,
    m = 0..M, M = ceil((P + 1/2) dz / h) with P the largest plane offset
    scanned (both signs under min_origin); a knot with no observation in
    either adjacent interval is returned as NaN (not identified). Raises
    ValueError with fewer than two usable nodes."""
    cal, dz = cfg.calibration, cfg.acquisition.dz_um
    conv, h = cal.origin_convention, float(cal.knot_step_um)
    if conv not in ("symmetric", "min_origin"):
        raise ValueError("unknown origin_convention %r" % (conv,))
    used, dropped = [], []
    for sc in scans:
        vals = sc.values(cal.statistic)
        (used if int(np.isfinite(vals).sum()) >= cal.min_planes_per_node else dropped).append(sc)
    if len(used) < 2:
        raise ValueError("fit_growth: need at least two nodes with >= %d finite values" % cal.min_planes_per_node)
    N = len(used)
    obs_i, obs_z, obs_w, obs_o = [], [], [], []
    z0 = np.empty(N)
    for i, sc in enumerate(used):
        vals = sc.values(cal.statistic)
        ok = np.isfinite(vals)
        obs_i.append(np.full(int(ok.sum()), i))
        obs_z.append(sc.z_um[ok])
        obs_w.append(vals[ok])
        obs_o.append(sc.ks[ok] - sc.k_star)
        z0[i] = sc.z_sub_um if math.isfinite(sc.z_sub_um) else sc.k_star * dz
    oi, oz, ow, oo = (np.concatenate(a) for a in (obs_i, obs_z, obs_w, obs_o))
    # knots cover |delta| <= P dz + dz / 2 (P the largest plane offset scanned, the axis within half a
    # plane of z_k*); a fitted axis outside that is reached by linear extrapolation
    P_max = int(np.max(np.abs(oo)))
    M = max(1, int(math.ceil((P_max + 0.5) * dz / h - 1e-9)))
    K = M if conv == "symmetric" else 2 * M
    zero = 0 if conv == "symmetric" else M
    free = np.array([m for m in range(K + 1) if m != zero])
    n_obs = oi.size

    interp = cal.growth_interp

    def design(z):
        return _basis(z[oi] - oz, conv, h, M, interp)

    def unpack(q):
        G = np.zeros(K + 1)
        G[free] = q[2 * N:]
        return q[:N], q[N:2 * N], G

    def resid(q):
        c, z, G = unpack(q)
        W, _ = design(z)
        return c[oi] + W @ G - ow

    def jac(q):
        c, z, G = unpack(q)
        W, dW = design(z)
        J = np.zeros((n_obs, 2 * N + free.size))
        J[np.arange(n_obs), oi] = 1.0
        J[np.arange(n_obs), N + oi] = dW @ G
        J[:, 2 * N:] = W[:, free]
        return J

    # linear start at the starting depths
    W0, _ = design(z0)
    A = np.zeros((n_obs, N + free.size))
    A[np.arange(n_obs), oi] = 1.0
    A[:, N:] = W0[:, free]
    lb = np.r_[np.full(N, -np.inf), np.full(free.size, 0.0 if conv == "min_origin" else -np.inf)]
    lin = lsq_linear(A, ow, bounds=(lb, np.full(N + free.size, np.inf)))
    q0 = np.r_[lin.x[:N], z0, lin.x[N:]]
    lo = np.r_[np.full(N, -np.inf), z0 - cal.z_ax_bound_um, lb[N:]]
    hi = np.r_[np.full(N, np.inf), z0 + cal.z_ax_bound_um, np.full(free.size, np.inf)]
    q0 = np.clip(q0, lo, hi)
    r = least_squares(resid, q0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac",
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    c, z, G = unpack(r.x)
    delta = z[oi] - oz
    j, _, _ = _weights(delta, conv, h, M)
    counts = np.bincount(j, minlength=K)[:K]
    touched = np.zeros(K + 1, dtype=bool)          # a knot is identified when an adjacent interval holds data
    touched[:-1] |= counts > 0
    touched[1:] |= counts > 0
    G = np.where(touched | (np.arange(K + 1) == zero), G, np.nan)
    knots = np.arange(K + 1) * h - (0.0 if conv == "symmetric" else M * h)
    return GrowthFit(conv, cal.statistic, interp, h, knots, G, np.array([sc.node_id for sc in used]), c, z, oi, oo,
                     delta, resid(r.x), counts, tuple(sc.node_id for sc in dropped),
                     float(r.cost), bool(r.success), str(r.message), int(r.nfev))


def kernel_from_growth(growth, rcfg, sigma_r0_um=None, max_delta_um=None):
    """A RendererConfig whose Gaussian kernel table is sigma_r(delta_m) =
    sqrt(sigma_r(0)^2 + G(delta_m)) at the growth's knots (delta_m <=
    max_delta_um when given). sigma_r(0) = rcfg.sigma_r0_um unless given: it
    is not identified by the scan (mathematics s.3.5). The renderer's kernel
    is even in delta, so a "min_origin" growth is refused; so is a knot where
    sigma_r(0)^2 + G <= 0. The continuation beyond the last knot stays
    rcfg's."""
    if growth.convention != "symmetric":
        raise ValueError("kernel_from_growth: the renderer's kernel is even in delta; "
                         "a two-sided growth needs a per-sign table (procedure s.3.10, above/below)")
    s0 = float(rcfg.sigma_r0_um if sigma_r0_um is None else sigma_r0_um)
    keep = np.ones(growth.knots_um.size, dtype=bool) if max_delta_um is None \
        else growth.knots_um <= float(max_delta_um) + 1e-12
    if keep.sum() < 2:
        raise ValueError("kernel_from_growth: fewer than two knots kept")
    s2 = s0 ** 2 + growth.g_um2[keep]
    if not (s0 > 0) or not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        raise ValueError("kernel_from_growth: sigma_r(0)^2 + G must be > 0 at every knot")
    sig = np.sqrt(s2)
    sig[0] = s0
    return replace(rcfg, kernel_family="gaussian_table", sigma_r0_um=s0,
                   kernel_table_delta_um=tuple(float(x) for x in growth.knots_um[keep]),
                   kernel_table_sigma_um=tuple(float(x) for x in sig))
