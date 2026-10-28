"""The far-field calibration -- Block 15 in specs/SPEC.md (S2b of the
2026-10-20 plan; D-026.1, D-074 to D-081).

  core_fit_iterative    the far statistic (15.2; D-077, D-078.2, D-081.4): a
                        Gaussian core fitted to the dip within +-1 FWHM of the
                        previous pass's width, starting from the prior width
                        sigma_p of the renderer's current law, up to N passes,
                        stopped when the width moves by no more than kappa of
                        its own SE; the width bounded to [0.25, 4] sigma_p
  far_plane_statistics  one far plane: the peak, the depth, the core fit and
                        the line-end mean of the dip (the background's check)
  far_plane_scan        the far scan of one node over k* +- far_offsets_planes
                        on the +-patch_half_um patch with the planar far-pixel
                        background (15.1; Block 10.P)
  select_far_planes     the planes of the far fit (15.3): |k - k*| >= 5 and the
                        PREDICTED dip A_j / (sqrt(2 pi) sigma_p) >= 3 RMS_j
  se_model              the SE a core fit would have at a plane if the dip were
                        the Gaussian the far law predicts (the IRLS weights)
  fit_far_slope         the far fit (15.4): omega_jk = c_j + (sigma_0_hat +
                        gamma x_jk(s_j))^2, sigma_0_hat held fixed, gamma >= 0,
                        both sides, the axis shifts s_j free within +-0.28 um,
                        the weights frozen per IRLS round (D-078.3)
  bootstrap_gamma       the node bootstrap's SE of gamma_hat (15.5)
  anchor_sensitivity    d gamma_hat / d sigma_0 by a central difference (15.5)
  far_fit_summary       everything the far-fit file records

Notation: delta = z_ax - z_k (object minus plane, as Block 10); x = |delta| -
delta_0, delta_0 = table_max_delta_um = 0.84 um; sigma_r(delta) = sigma_0 +
gamma (|delta| - delta_0) beyond delta_0 (D-026.1). The symbol kappa is the
core fit's stopping tolerance only; the quadratic comparison coefficient is
gamma_2 (D-081.6). Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from . import calibration_scene, profiles
from .calibration import (FW, PlaneScan, gaussian_core_fit, line_geometry, patch_images, patch_planes,
                          stage_planes)
from ..model.kernel import sigma_r

STATUS_OK = "ok"


# ------------------------------------------------------------ the statistic

def core_fit_iterative(v, w, depth, v_peak, sigma_p, cal, mcfg=None):
    """The iterative core fit (15.2) of the dip w (gl) on the offsets v (um).

    Pass 1 fits the Gaussian core within +-core_fit_fwhm_factor x FWHM(sigma_p)
    of v_peak; pass i + 1 within the same factor x FWHM(s_i) of the fitted
    centre; the width s is bounded to far_core_width_bounds x sigma_p (a fit at
    a bound: status at_bound), the centre to half the window about its start,
    the amplitude >= 0. From pass 2 on the loop stops when |s_i - s_(i-1)| <=
    far_core_stop_tol x SE(s_i) (SE from the fit's own residual RMS, Block 10's
    gaussian_core_fit); no stop by pass far_core_max_passes: status
    not_converged, the last pass's values kept. Returns a dict: status,
    omega_um2 (= s^2), se_omega_um2 (2 s SE(s)), amp_gl, v_c_um, sigma_um,
    rms_gl, n_win, passes, window_half_um."""
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    nan = float("nan")
    out = dict(status=STATUS_OK, omega_um2=nan, se_omega_um2=nan, amp_gl=nan, v_c_um=float(v_peak), sigma_um=nan,
               rms_gl=nan, n_win=0, passes=0, window_half_um=nan)
    sp = float(sigma_p)
    slo, shi = cal.far_core_width_bounds[0] * sp, cal.far_core_width_bounds[1] * sp
    c_cen, s_cur = float(v_peak), sp
    f = None
    for i in range(1, int(cal.far_core_max_passes) + 1):
        W = cal.core_fit_fwhm_factor * FW * s_cur
        win = np.abs(v - c_cen) <= W
        out["n_win"], out["window_half_um"], out["passes"] = int(win.sum()), float(W), i
        if int(win.sum()) < cal.min_window_samples:
            out["status"] = "few_samples"
            return out
        f = gaussian_core_fit(v[win], w[win], [max(float(depth), 1e-3), c_cen, min(max(s_cur, slo * 1.01), shi * 0.99)],
                              [0.0, c_cen - 0.5 * W, slo], [np.inf, c_cen + 0.5 * W, shi])
        if f is None:
            out["status"] = "fit_failed"
            return out
        out.update(omega_um2=f["s"] ** 2, se_omega_um2=2.0 * f["s"] * f["se_s"], amp_gl=f["a"], v_c_um=f["c"],
                   sigma_um=f["s"], rms_gl=f["rms"])
        if f["at_bound"]:
            out["status"] = "at_bound"
            return out
        s_prev = s_cur
        c_cen, s_cur = f["c"], f["s"]
        if i >= 2 and abs(s_cur - s_prev) <= cal.far_core_stop_tol * f["se_s"]:
            return out
    out["status"] = "not_converged"
    return out


def far_plane_statistics(v, I_flat, B0, sigma_p, mcfg, cal):
    """One far plane (15.1-15.2): the flattened profile I_flat (gl) on the
    offsets v (um) against the scalar background B0 = B_hat(line(0)); the peak
    is the smallest sample of I_flat smoothed by max(sigma_p / (2 step),
    focus_smooth_px) samples among those within fit_v0_bounds_um, the depth B0
    - that value (<= 0: no_dip); then core_fit_iterative with the prior width
    sigma_p and, for a kept fit, the line-end mean of the dip over the samples
    with |v - v_c| >= far_line_end_fwhm_factor x FWHM when both sides hold at
    least far_line_end_min_samples of them (NaN otherwise). Returns the core
    fit's dict plus depth_gl and line_end_gl."""
    v = np.asarray(v, dtype=float)
    I = np.asarray(I_flat, dtype=float)
    step = float(v[1] - v[0])
    sig = max(0.5 * float(sigma_p) / step, float(mcfg.focus_smooth_px))
    sm = ndimage.gaussian_filter1d(I, sig, mode="nearest") if sig > 0 else I
    lo, hi = mcfg.fit_v0_bounds_um
    cand = np.flatnonzero((v >= lo) & (v <= hi))
    if cand.size == 0:
        raise ValueError("far_plane_statistics: no sample inside the v0 bounds")
    n0 = int(cand[np.argmin(sm[cand])])
    depth = float(B0 - sm[n0])
    out = dict(status="no_dip", omega_um2=np.nan, se_omega_um2=np.nan, amp_gl=np.nan, v_c_um=float(v[n0]),
               sigma_um=np.nan, rms_gl=np.nan, n_win=0, passes=0, window_half_um=np.nan, depth_gl=depth,
               line_end_gl=np.nan)
    if not depth > 0:
        return out
    w = B0 - I
    out.update(core_fit_iterative(v, w, depth, float(v[n0]), sigma_p, cal, mcfg))
    out["depth_gl"] = depth
    if math.isfinite(out["sigma_um"]):
        zone = np.abs(v - out["v_c_um"]) >= cal.far_line_end_fwhm_factor * FW * out["sigma_um"]
        n_left, n_right = int(np.sum(zone & (v < out["v_c_um"]))), int(np.sum(zone & (v > out["v_c_um"])))
        if min(n_left, n_right) >= cal.far_line_end_min_samples:
            out["line_end_gl"] = float(np.mean(w[zone]))
    return out


def far_offsets(cfg):
    """v_n (um) of the far line: every profile_step_um to +-far_profile_half_um."""
    m, cal = cfg.measure, cfg.calibration
    n = int(round(cal.far_profile_half_um / m.profile_step_um))
    return np.arange(-n, n + 1) * m.profile_step_um


def far_plane_scan(patch, node, branch, scene, cfg, keep_images=False):
    """The far scan of one node (15.1): the planes stage_planes(node, cfg,
    "far") of the patch (+-patch_half_um about the pilot's centre), each with
    its far pixels and planar background (Block 10.P), the Eq. 9 profile along
    the node's fixed line of half-length far_profile_half_um, the flattened
    profile against B_hat(line(0)), the prior width sigma_p = sigma_r(|z_sub -
    z_k|) of the renderer's current law, and far_plane_statistics. Returns a
    PlaneScan with set "far", the per-plane profiles and, when asked, the
    images."""
    block, ks, valid, frame = patch
    m, cal, dz = cfg.measure, cfg.calibration, cfg.acquisition.dz_um
    if not (math.isfinite(node.z_sub_um) and math.isfinite(node.cx_um) and math.isfinite(node.cy_um)):
        raise ValueError("far_plane_scan: the node has no sharpest plane")
    o, y_hat, e_u = line_geometry(node)
    v = far_offsets(cfg)
    kk = stage_planes(node, cfg, "far")
    n = kk.size
    idx, far, bco, bok, frame, stripe = patch_planes(patch, kk, node, branch, scene, cfg)
    names = ("omega_um2", "depth_gl", "B_bar", "se_omega_um2", "amp_gl", "v_c_um", "rms_gl", "window_half_um",
             "line_end_gl")
    cols = {name: np.full(n, np.nan) for name in names}
    n_win, passes = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    w_gl, bhat = np.full((n, v.size), np.nan), np.full((n, v.size), np.nan)
    status = []
    xy = o[None, :] + v[:, None] * y_hat[None, :]
    nan_n = np.full(n, np.nan)
    for q, k in enumerate(kk):
        if idx[q] < 0:
            status.append("invalid_plane")
            continue
        if not bok[q] or not (bco[q, 0] > 0):
            status.append("bg_few")
            continue
        plane = block[idx[q]]
        B_line = calibration_scene.background_on_line(bco[q], node.cx_um, node.cy_um, xy)
        B0 = float(bco[q, 0])
        cols["B_bar"][q] = B0
        bhat[q] = B_line
        I = profiles.sample_profile(plane, frame, o, y_hat, e_u, v, profiles.n_along(m), m.profile_step_um)
        if not np.all(np.isfinite(I)):
            status.append("profile_nan")
            continue
        w_gl[q] = B_line - I
        sp = float(sigma_r(abs(float(node.z_sub_um) - float(k) * dz), cfg.renderer))
        st = far_plane_statistics(v, I - B_line + B0, B0, sp, m, cal)
        for name in names:
            if name != "B_bar":
                cols[name][q] = st[name]
        n_win[q], passes[q] = st["n_win"], st["passes"]
        status.append(st["status"])
    return PlaneScan(int(node.node_id), float(node.d_hat_um), float(node.phi_rad), float(node.alpha_hat),
                     int(node.k_star), float(node.z_sub_um), kk, kk * dz, cols["omega_um2"], nan_n.copy(),
                     nan_n.copy(), cols["depth_gl"], cols["B_bar"], tuple(status), "far",
                     cols["se_omega_um2"], cols["amp_gl"], cols["v_c_um"], cols["rms_gl"], n_win, passes,
                     cols["window_half_um"], cols["line_end_gl"], bco, v, w_gl, bhat,
                     patch_images(patch, idx, far, kk, node) if keep_images else None, stripe)


# ------------------------------------------------------------ the selection

@dataclass(frozen=True)
class FarSelection:
    """The observations of the far fit (15.3), one row per kept (node, plane)."""

    node_ids: np.ndarray        # (J,) the nodes kept
    area_gl_um: np.ndarray      # (J,) A_j
    rms_gl: np.ndarray          # (J,) RMS_j
    z_p_um: np.ndarray          # (J,) the pilot's sub-plane depth
    obs_node: np.ndarray        # (n_obs,) index into node_ids
    obs_k: np.ndarray           # (n_obs,) plane index
    z_um: np.ndarray            # (n_obs,) plane depth
    omega_um2: np.ndarray       # (n_obs,)
    obs_rms_gl: np.ndarray      # (n_obs,) the plane's residual RMS
    sigma_p_um: np.ndarray      # (n_obs,) the prior width used at that plane
    dropped: Dict[int, str]     # node_id -> reason
    n_not_converged: int
    n_snr_dropped: int
    n_planes_per_node: np.ndarray   # (J,)

    def subset(self, nodes):
        """A FarSelection over the node indices `nodes` (with repetition: the
        bootstrap), renumbered 0..len(nodes) - 1."""
        nodes = np.asarray(nodes, dtype=int)
        parts = []
        for t, j in enumerate(nodes):
            m = self.obs_node == j
            parts.append((np.full(int(m.sum()), t), self.obs_k[m], self.z_um[m], self.omega_um2[m], self.obs_rms_gl[m],
                          self.sigma_p_um[m]))
        cat = [np.concatenate([p[i] for p in parts]) for i in range(6)]
        return FarSelection(self.node_ids[nodes], self.area_gl_um[nodes], self.rms_gl[nodes], self.z_p_um[nodes],
                            cat[0], cat[1], cat[2], cat[3], cat[4], cat[5], dict(self.dropped), self.n_not_converged,
                            self.n_snr_dropped, self.n_planes_per_node[nodes])


def select_far_planes(scans, cfg):
    """The planes of the far fit (15.3; D-074.1, D-078.3) from the far scans:
    per node A_j = the median over the ok planes with |k - k*| in
    far_area_offsets of amp x sigma x sqrt(2 pi), RMS_j = the median RMS over
    its ok planes; a plane is kept when |k - k*| >= far_min_k_offset, its
    status is ok and A_j / (sqrt(2 pi) sigma_p(k)) >= far_min_snr x RMS_j; a
    node without A_j (no_area) or with fewer than far_min_planes_per_node kept
    planes (few_planes) is dropped and named. sigma_p is recomputed from the
    renderer's current law at the pilot's depth. Scans of another set raise."""
    cal, dz = cfg.calibration, cfg.acquisition.dz_um
    lo_a, hi_a = cal.far_area_offsets
    ids, areas, rmss, zps, counts = [], [], [], [], []
    obs = {k: [] for k in ("node", "k", "z", "omega", "rms", "sp")}
    dropped: Dict[int, str] = {}
    n_nc = n_snr = 0
    for sc in scans:
        if sc.set != "far":
            raise ValueError("select_far_planes: scan of node %d is not a far scan" % sc.node_id)
        st = np.array(sc.status)
        off = np.abs(sc.offsets())
        ok = st == STATUS_OK
        n_nc += int(np.sum(st == "not_converged"))
        in_area = ok & (off >= lo_a) & (off <= hi_a) & np.isfinite(sc.amp_gl) & np.isfinite(sc.omega_um2)
        if not in_area.any():
            dropped[int(sc.node_id)] = "no_area"
            continue
        A = float(np.median(sc.amp_gl[in_area] * np.sqrt(sc.omega_um2[in_area]) * math.sqrt(2.0 * math.pi)))
        rms_j = float(np.median(sc.rms_gl[ok & np.isfinite(sc.rms_gl)]))
        sp = sigma_r(np.abs(float(sc.z_sub_um) - sc.z_um), cfg.renderer)
        pred = A / (math.sqrt(2.0 * math.pi) * sp)
        cand = ok & (off >= cal.far_min_k_offset) & np.isfinite(sc.omega_um2)
        keep = cand & (pred >= cal.far_min_snr * rms_j)
        n_snr += int(np.sum(cand & ~keep))
        if int(keep.sum()) < cal.far_min_planes_per_node:
            dropped[int(sc.node_id)] = "few_planes"
            continue
        j = len(ids)
        ids.append(int(sc.node_id)); areas.append(A); rmss.append(rms_j); zps.append(float(sc.z_sub_um))
        counts.append(int(keep.sum()))
        obs["node"].append(np.full(int(keep.sum()), j))
        obs["k"].append(np.asarray(sc.ks)[keep])
        obs["z"].append(sc.z_um[keep])
        obs["omega"].append(sc.omega_um2[keep])
        obs["rms"].append(sc.rms_gl[keep])
        obs["sp"].append(np.asarray(sp)[keep])
    cat = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in obs.items()}
    return FarSelection(np.array(ids, dtype=int), np.array(areas), np.array(rmss), np.array(zps),
                        cat["node"].astype(int), cat["k"].astype(int), cat["z"], cat["omega"], cat["rms"], cat["sp"],
                        dropped, n_nc, n_snr, np.array(counts, dtype=int))


# ------------------------------------------------------------ the far fit

def se_model(amp_gl, sigma_um, rms_gl, step_um, half_um):
    """(n,) the SE of omega = s^2 (um^2) a Gaussian-core fit would have on a
    Gaussian of amplitude amp_gl and SD sigma_um sampled every step_um within
    +-1 FWHM (cut at +-half_um, at least three samples each side) under white
    noise of SD rms_gl: 2 s x the inverse Fisher information of (a, c, s) in s
    (15.4; the model weights of D-078.3). Custom: a closed-form Fisher
    information on the fit's own grid; no library function."""
    a = np.asarray(amp_gl, dtype=float).reshape(-1)
    s = np.maximum(np.asarray(sigma_um, dtype=float).reshape(-1), 0.05)
    r = np.asarray(rms_gl, dtype=float).reshape(-1)
    W = np.maximum(np.minimum(FW * s, float(half_um)), 3.0 * float(step_um))
    m = (W / float(step_um)).astype(int)
    idx = np.arange(-int(m.max()), int(m.max()) + 1)
    u = idx[None, :] * float(step_um)
    mask = np.abs(idx)[None, :] <= m[:, None]
    e = np.exp(-0.5 * (u / s[:, None]) ** 2) * mask
    G = np.stack([e, a[:, None] * e * u / s[:, None] ** 2, a[:, None] * e * u ** 2 / s[:, None] ** 3], axis=-1)
    M = np.einsum("nli,nlj->nij", G, G)
    out = np.full(a.size, np.inf)
    for n in range(a.size):
        try:
            cov = np.linalg.inv(M[n]) * r[n] ** 2
            out[n] = 2.0 * s[n] * math.sqrt(max(float(cov[2, 2]), 0.0))
        except np.linalg.LinAlgError:
            pass
    return out


@dataclass(frozen=True)
class FarFit:
    """The far fit of 15.4."""

    gamma: float
    gamma_minus: float          # the per-sign fit's slope on delta < 0 (NaN unless per_sign); gamma is then delta > 0
    gamma2: float               # the quadratic comparison's coefficient (NaN unless quadratic)
    sigma0_um: float
    c_um2: np.ndarray           # (J,)
    s_um: np.ndarray            # (J,)
    node_ids: np.ndarray        # (J,)
    obs_node: np.ndarray        # (n_obs,)
    obs_k: np.ndarray           # (n_obs,)
    obs_offset: np.ndarray      # (n_obs,) k - k*_j (sign of the plane relative to the fitted axis: see delta_um)
    delta_um: np.ndarray        # (n_obs,) z_p + s_j - z_k at the fit
    residual_um2: np.ndarray    # (n_obs,) model - omega
    weights: np.ndarray         # (n_obs,) the last round's 1 / SE_m
    range_um: float             # delta_0 + max x
    rounds: int
    success: bool
    at_bound: bool
    message: str
    nfev: int

    def rms_by_offset(self):
        out = {}
        for o in sorted(set(np.abs(self.obs_offset).tolist())):
            r = self.residual_um2[np.abs(self.obs_offset) == o]
            out[int(o)] = float(np.sqrt(np.mean(r ** 2)))
        return out


def fit_far_slope(sel, sigma0_um, cfg, per_sign=False, quadratic=False, weights="model", k_star=None):
    """The far fit (15.4) over a FarSelection with the anchor sigma0_um held
    fixed. theta = (c in R^J, s in [-b, b]^J, gamma in [0, gamma_max] (two
    slopes when per_sign: delta > 0, delta < 0), gamma_2 in R when quadratic);
    residual c_j + (sigma_0 + gamma x + gamma_2 x^2)^2 - omega, x = |z_p,j +
    s_j - z_k| - delta_0; least_squares (trf, analytic Jacobian, tolerances
    1e-10) from c = 0, s = 0, gamma = 1, gamma_2 = 0. weights "model": an
    unweighted round then far_irls_rounds refits with 1 / SE_m frozen at the
    previous round (D-078.3); "none": the unweighted fit only. k_star: {node_id:
    k*} for obs_offset (None: offsets from the nearest plane to z_p).
    ValueError below two nodes or six observations."""
    cal = cfg.calibration
    J = int(sel.node_ids.size)
    om, idx, zk, zp = sel.omega_um2, sel.obs_node, sel.z_um, sel.z_p_um[sel.obs_node]
    if J < 2 or om.size < 6:
        raise ValueError("fit_far_slope: need at least two far nodes and six planes (got %d, %d)" % (J, om.size))
    d0, s0 = float(cal.table_max_delta_um), float(sigma0_um)
    ng, nq = (2 if per_sign else 1), (1 if quadratic else 0)
    step, half = float(cfg.measure.profile_step_um), float(cal.far_profile_half_um)

    def parts(q):
        c, s = q[:J], q[J:2 * J]
        g = q[2 * J:2 * J + ng]
        g2 = q[2 * J + ng] if quadratic else 0.0
        dl = zp + s[idx] - zk
        x = np.abs(dl) - d0
        gg = np.where(dl > 0, g[0], g[-1])
        sig = s0 + gg * x + g2 * x * x
        return c, s, g, g2, dl, x, gg, sig

    def predict(q):
        c, s, g, g2, dl, x, gg, sig = parts(q)
        return c[idx] + sig ** 2

    def jac_pred(q):
        c, s, g, g2, dl, x, gg, sig = parts(q)
        n = om.size
        Jc = np.zeros((n, J)); Jc[np.arange(n), idx] = 1.0
        Js = np.zeros((n, J)); Js[np.arange(n), idx] = 2.0 * sig * (gg + 2.0 * g2 * x) * np.sign(dl)
        cols = [Jc, Js]
        if per_sign:
            Jg = np.zeros((n, 2))
            Jg[dl > 0, 0] = (2.0 * sig * x)[dl > 0]
            Jg[dl <= 0, 1] = (2.0 * sig * x)[dl <= 0]
            cols.append(Jg)
        else:
            cols.append((2.0 * sig * x)[:, None])
        if quadratic:
            cols.append((2.0 * sig * x * x)[:, None])
        return np.hstack(cols)

    lo = np.r_[np.full(J, -np.inf), np.full(J, -cal.z_ax_bound_um), np.zeros(ng), np.full(nq, -np.inf)]
    hi = np.r_[np.full(J, np.inf), np.full(J, cal.z_ax_bound_um), np.full(ng, cal.far_gamma_max), np.full(nq, np.inf)]
    q0 = np.r_[np.zeros(J), np.zeros(J), np.ones(ng), np.zeros(nq)]

    def solve(wts, start):
        r = least_squares(lambda q: wts * (predict(q) - om), np.clip(start, lo, hi),
                          jac=lambda q: wts[:, None] * jac_pred(q), bounds=(lo, hi), method="trf", x_scale="jac",
                          ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=500)
        return r

    wts = np.ones_like(om)
    r = solve(wts, q0)
    nfev, rounds = int(r.nfev), 0
    if weights == "model":
        for _ in range(int(cal.far_irls_rounds)):
            s_pred = np.sqrt(np.maximum(predict(r.x), 0.01))
            a_pred = sel.area_gl_um[idx] / (math.sqrt(2.0 * math.pi) * s_pred)
            se = se_model(a_pred, s_pred, sel.obs_rms_gl, step, half)
            if not np.all(np.isfinite(se)) or np.any(se <= 0):
                break
            wts = 1.0 / se
            r = solve(wts, r.x)
            nfev += int(r.nfev)
            rounds += 1
    elif weights != "none":
        raise ValueError("fit_far_slope: weights must be 'model' or 'none'")
    c, s, g, g2, dl, x, gg, sig = parts(r.x)
    at_bound = bool(np.any(np.isclose(g, 0.0, atol=1e-9)) or np.any(np.isclose(g, cal.far_gamma_max, rtol=1e-9)))
    if k_star is None:
        offs = np.round(zk / cfg.acquisition.dz_um).astype(int) - np.round(zp / cfg.acquisition.dz_um).astype(int)
    else:
        offs = np.asarray(sel.obs_k) - np.array([k_star[int(n)] for n in sel.node_ids[idx]])
    return FarFit(float(g[0]), float(g[1]) if per_sign else float("nan"), float(g2) if quadratic else float("nan"),
                  s0, c, s, sel.node_ids.copy(), idx.copy(), np.asarray(sel.obs_k).copy(), offs, dl,
                  predict(r.x) - om, wts.copy(), float(d0 + np.max(x)), rounds, bool(r.success) and not at_bound,
                  at_bound, str(r.message), nfev)


def bootstrap_gamma(sel, sigma0_um, cfg, rng, n_resamples=None):
    """(se, n_ok, draws): the node bootstrap of the far fit (15.5): n_resamples
    (default bootstrap_resamples) draws of the nodes with replacement, each
    refitted by fit_far_slope with the model weights; a draw whose fit fails
    (or has fewer than two distinct nodes' worth of planes) is dropped. se is
    the SD of the kept gamma_hat*, NaN below two."""
    n = int(cfg.calibration.bootstrap_resamples if n_resamples is None else n_resamples)
    J = int(sel.node_ids.size)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, J, J)
        try:
            f = fit_far_slope(sel.subset(pick), sigma0_um, cfg)
        except ValueError:
            continue
        if f.success:
            draws.append(f.gamma)
    draws = np.array(draws, dtype=float)
    return (float(np.std(draws, ddof=1)) if draws.size >= 2 else float("nan")), int(draws.size), draws


def anchor_sensitivity(sel, sigma0_um, cfg):
    """d gamma_hat / d sigma_0 (per um) by the central difference over +-
    anchor_sensitivity_step_um about the anchor (15.5, D-081.7)."""
    h = float(cfg.calibration.anchor_sensitivity_step_um)
    up = fit_far_slope(sel, float(sigma0_um) + h, cfg).gamma
    dn = fit_far_slope(sel, float(sigma0_um) - h, cfg).gamma
    return float((up - dn) / (2.0 * h))


def far_fit_summary(sel, fit, fit_sign, fit_quad, se_boot, n_boot_ok, dgamma_dsigma0, sigma0_um, se_sigma0_um, cfg,
                    seed=None):
    """The far-fit record (15.6) as a JSON-ready dict: gamma_hat, its SE and
    the two parts, the t_(J-1) interval, c, s and the planes used per node, the
    range, the residual RMS by offset, the per-sign and quadratic comparisons,
    the anchor, the dropped nodes and the plane counts."""
    from scipy import stats
    J = int(sel.node_ids.size)
    se2 = float(se_boot) ** 2 + float(dgamma_dsigma0) ** 2 * float(se_sigma0_um) ** 2 if math.isfinite(se_sigma0_um) \
        else float("nan")
    se = math.sqrt(se2) if math.isfinite(se2) else float("nan")
    tq = float(stats.t.ppf(0.975, J - 1)) if J >= 2 else float("nan")

    def num(x):
        x = float(x)
        return x if math.isfinite(x) else None
    return dict(
        gamma=num(fit.gamma), se=num(se), se_boot=num(se_boot), n_boot_ok=int(n_boot_ok),
        dgamma_dsigma0_per_um=num(dgamma_dsigma0), t_quantile=num(tq),
        interval=[num(fit.gamma - tq * se), num(fit.gamma + tq * se)] if math.isfinite(se) else [None, None],
        anchor=dict(sigma0_um=num(sigma0_um), se_um=num(se_sigma0_um), delta0_um=float(cfg.calibration.table_max_delta_um)),
        n_nodes=J, node_ids=[int(i) for i in sel.node_ids], c_um2=[num(x) for x in fit.c_um2],
        s_um=[num(x) for x in fit.s_um], n_planes_per_node=[int(x) for x in sel.n_planes_per_node],
        n_obs=int(sel.omega_um2.size), range_um=num(fit.range_um), rounds=int(fit.rounds), success=bool(fit.success),
        at_bound=bool(fit.at_bound), message=fit.message, nfev=int(fit.nfev),
        rms_by_offset={str(k): v for k, v in fit.rms_by_offset().items()},
        per_sign=dict(gamma_plus=num(fit_sign.gamma), gamma_minus=num(fit_sign.gamma_minus),
                      success=bool(fit_sign.success)) if fit_sign is not None else None,
        quadratic=dict(gamma=num(fit_quad.gamma), gamma2_per_um=num(fit_quad.gamma2), success=bool(fit_quad.success),
                       label="labelled comparison, not the deliverable (D-026, D-081.6)") if fit_quad is not None else None,
        dropped_nodes={str(k): v for k, v in sel.dropped.items()}, n_not_converged=int(sel.n_not_converged),
        n_snr_dropped=int(sel.n_snr_dropped), bootstrap_resamples=int(cfg.calibration.bootstrap_resamples),
        seed=None if seed is None else int(seed), extrapolated_beyond_range=True,
        settings=dict(far_min_k_offset=cfg.calibration.far_min_k_offset, far_min_snr=cfg.calibration.far_min_snr,
                      far_irls_rounds=cfg.calibration.far_irls_rounds, z_ax_bound_um=cfg.calibration.z_ax_bound_um,
                      far_gamma_max=cfg.calibration.far_gamma_max))
