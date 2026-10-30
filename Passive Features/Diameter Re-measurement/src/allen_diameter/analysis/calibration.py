"""Defocus-kernel calibration, first stage -- Block 10 in specs/SPEC.md
(procedure Eq. 4 and s.3.4; mathematics Eq. 18 and s.3.5).

  calibration_reasons the selection of the calibration nodes (D-061.4,
                      D-067.4): Allen's diameter D_A = 2 x the branch radius
                      at the node puts a node in the near-field set (D_A <=
                      node_allen_diameter_max_um, scanned up to +-P planes)
                      or the far-field set (lo < D_A <= hi,
                      far_field_allen_diameter_um; the offsets beyond, D-026);
                      in S, flat and faint are read from the pilot
                      (pilot_view); since D-094 (2026-10-28) neither the
                      optical depth (faint, node_alpha_check) nor the flag
                      faint (node_faint_flag_check; not in S since D-096) is
                      applied unless switched on. The fit's d_hat is no
                      criterion. The
                      far-field scan is not built: such a node is selected and
                      reported, not scanned (FAR_FIELD_NOT_BUILT)
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
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, lsq_linear

from . import calibration_scene, cell, phantoms, profiles
from .fit import half_depth_width
from .node_pipeline import measure_pilot, node_background, pilot_view
from .plane_iteration import Scene
from ..model.kernel import sigma_r

STATUS_OK = "ok"


# ------------------------------------------------------------ one plane ---

FW = math.sqrt(8.0 * math.log(2.0))          # FWHM / SD of a Gaussian


def gaussian_core_fit(v, w, q0, lo, hi):
    """Least-squares Gaussian a exp(-(v - c)^2 / (2 s^2)) through the samples
    (v, w) from the start q0 = (a, c, s) within bounds lo, hi (scipy
    least_squares, trf, tolerances 1e-12). Returns None on failure, else a dict:
    a, c, s, se_s (the Gauss-Newton SE of s: RMS x [(J^T J)^-1]_ss^(1/2)),
    rms (the residual RMS with n - 3 degrees of freedom, gl), n, at_bound
    (s at either bound). Shared by Block 10's dip_statistics and Block 15's
    iterative core fit."""
    def resid(q):
        return q[0] * np.exp(-0.5 * ((v - q[1]) / q[2]) ** 2) - w

    try:
        r = least_squares(resid, q0, bounds=(lo, hi), method="trf", x_scale="jac",
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200)
    except ValueError:
        return None
    if r.status <= 0 or not np.all(np.isfinite(r.x)):
        return None
    a, c, s = (float(x) for x in r.x)
    e = np.exp(-0.5 * ((v - c) / s) ** 2)
    J = np.column_stack([e, a * e * (v - c) / s ** 2, a * e * (v - c) ** 2 / s ** 3])
    n = int(v.size)
    s2 = float(np.sum(resid(r.x) ** 2)) / max(n - 3, 1)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        se_s = math.sqrt(max(float(cov[2, 2]), 0.0))
    except np.linalg.LinAlgError:
        se_s = float("nan")
    at_bound = bool(np.isclose(s, lo[2], rtol=1e-6, atol=0.0) or np.isclose(s, hi[2], rtol=1e-6, atol=0.0))
    return dict(a=a, c=c, s=s, se_s=se_s, rms=math.sqrt(s2), n=n, at_bound=at_bound)


def _core_fit(v, w, depth, v_peak, fwhm):
    """Block 10's core fit: start at the dip's depth, peak and FWHM; the width
    free within 1e-3 .. 1e3 of its start. Returns gaussian_core_fit's dict or None."""
    s0 = fwhm / FW
    return gaussian_core_fit(v, w, [depth, v_peak, s0], [0.0, v_peak - fwhm, 1e-3 * s0], [np.inf, v_peak + fwhm, 1e3 * s0])


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
    out = dict(omega_um2=nan, vw_um2=nan, area_gl_um=nan, depth_gl=nan, v_c_um=nan, status=STATUS_OK,
               amp_gl=nan, sigma_um=nan, se_omega_um2=nan, rms_gl=nan, n_win=0, window_half_um=nan, fwhm_um=nan,
               passes=0)
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
        out["fwhm_um"] = float(fwhm)
        out["window_half_um"] = float(cal.core_fit_fwhm_factor * fwhm)
        win = np.abs(v - v[n0]) <= cal.core_fit_fwhm_factor * fwhm
        out["n_win"] = int(win.sum())
        if int(win.sum()) < cal.min_window_samples:
            out["status"] = "few_samples"
        else:
            core = _core_fit(v[win], w[win], depth, float(v[n0]), fwhm)
            if core is None:
                out["status"] = "fit_failed"
            else:
                out["omega_um2"] = core["s"] ** 2
                out["amp_gl"], out["sigma_um"], out["rms_gl"] = core["a"], core["s"], core["rms"]
                out["se_omega_um2"] = 2.0 * core["s"] * core["se_s"]
                out["passes"] = 1
                v_c = core["c"]
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
    B_bar: np.ndarray         # (n,) background in plane k: the planar background at the line's origin (10.P)
    status: Tuple[str, ...]   # (n,) "ok" or the reason the plane has no omega
    # 2026-10-23 (10.P, D-075, Block 15): the set, the core fit's details and the per-plane profiles; defaults keep
    # older constructions and files valid. images: the patch stack, only when the scan was asked to keep it
    set: str = "near"
    se_omega_um2: Optional[np.ndarray] = None    # (n,) SE of omega from the plane's own fit (2 s SE(s))
    amp_gl: Optional[np.ndarray] = None          # (n,) fitted core amplitude
    v_c_um: Optional[np.ndarray] = None          # (n,) fitted core centre
    rms_gl: Optional[np.ndarray] = None          # (n,) residual RMS of the core fit
    n_win: Optional[np.ndarray] = None           # (n,) samples in the last window
    passes: Optional[np.ndarray] = None          # (n,) core-fit passes (1 for Block 10's statistic)
    window_half_um: Optional[np.ndarray] = None  # (n,) half-width of the last window
    line_end_gl: Optional[np.ndarray] = None     # (n,) line-end mean of the dip (the background check), far scans
    b_coef: Optional[np.ndarray] = None          # (n, 3) the planar background's b0, bx, by
    v_um: Optional[np.ndarray] = None            # (n_v,) the line's offsets
    w_gl: Optional[np.ndarray] = None            # (n, n_v) the dip B_hat(line(v)) - I(v)
    bhat_gl: Optional[np.ndarray] = None         # (n, n_v) the background along the line
    images: Optional[dict] = field(default=None, compare=False, repr=False)
    stripe: Optional[tuple] = None               # (n,) stripe_support's info per plane (Block 10.S), None under "square"

    def values(self, statistic):
        if statistic == "gaussian_core_width2":
            return self.omega_um2
        if statistic == "windowed_variance":
            return self.vw_um2
        raise ValueError("unknown statistic %r" % (statistic,))

    def offsets(self):
        """k - k* per plane."""
        return np.asarray(self.ks) - int(self.k_star)


def line_geometry(node):
    """(origin (2,), y_hat (2,), e_u (2,)) of a node's fixed measuring line: the
    pilot's centre, the measuring axis (-sin theta, cos theta) and the
    along-branch axis (cos theta, sin theta)."""
    th = float(node.theta_rad)
    return (np.array([node.cx_um, node.cy_um]), np.array([-math.sin(th), math.cos(th)]),
            np.array([math.cos(th), math.sin(th)]))


def stage_planes(node, cfg, which):
    """The plane indices a stage scans about the node's sharpest plane: k* +-
    offsets_planes ("near") or +- far_offsets_planes ("far")."""
    P = cfg.calibration.offsets_planes if which == "near" else cfg.calibration.far_offsets_planes
    return np.arange(int(node.k_star) - P, int(node.k_star) + P + 1)


def support_masks(shape, frame, kk, node, branch, scene, cfg, sigma_soma_um=None):
    """The background's support of each plane of kk on a patch of `shape` and
    `frame` (10.P / 10.S): (support (n, H, W) bool, stripe (n,) tuple of info
    dicts or None). background_support "square": 10.P's far pixels
    (far_pixel_mask with far_pixel_sigmas), stripe None. "stripe" (Block
    10.S, D-101): the pixels no neighbour shades at tau = isolation_depth_gl
    and the node's own fitted cylinder (d_hat_um) does not shade, cut to the
    stripe along the measuring line with most clear pixels (stripe_support),
    plane by plane. The own cylinder is the trace's own rows and the same rows
    moved across to the pilot's fitted centre (fitted_own_rows; the fitted
    cylinder of D-101.4, wherever the trace lies). sigma_soma_um: the soma's
    blur (default: the stage's, sigma_r at half the span of kk -- a caller
    testing one plane of a stage passes the stage's, F1 of 2026-10-28). The
    segments considered reach the patch within their radius plus 6 sigma_r at
    the largest defocus any segment end has from any plane (a deep segment far
    outside the patch can still shade it)."""
    cal, dz = cfg.calibration, cfg.acquisition.dz_um
    z = np.asarray(kk, dtype=float) * dz
    H, W = int(shape[0]), int(shape[1])
    seg_z = scene.seg_xyz[:, [2, 5]].ravel() if scene.seg_xyz.shape[0] else np.array([float(node.z_sub_um)])
    d_max = float(np.max(np.abs(seg_z[:, None] - z[None, :])))
    r_max = float(np.max(np.concatenate([scene.seg_radius.ravel(), [0.0]])))
    if cal.background_support == "square":
        reach = r_max + cal.far_pixel_sigmas * float(sigma_r(d_max, cfg.renderer))
        rows = calibration_scene.segment_rows(scene, frame, (H, W), branch, reach)
        return calibration_scene.far_pixel_mask((H, W), frame, rows, scene.soma, z, cfg), None
    d_own = float(node.d_hat_um)
    reach = max(r_max, 0.5 * d_own if math.isfinite(d_own) else 0.0) + 6.0 * float(sigma_r(d_max, cfg.renderer))
    rows = calibration_scene.segment_rows(scene, frame, (H, W), branch, reach)
    rows = np.vstack([rows, calibration_scene.fitted_own_rows(rows, node.cx_um, node.cy_um)])
    nb, own = calibration_scene.shadow_masks((H, W), frame, rows, scene.soma, z, node.B_bar, node.mu_hat_per_um,
                                             cfg, d_own, sigma_soma_um)
    support = np.zeros((z.size, H, W), dtype=bool)
    infos = []
    for q in range(z.size):
        support[q], info = calibration_scene.stripe_support(nb[q], own[q], frame, node.cx_um, node.cy_um,
                                                            node.theta_rad, cfg)
        infos.append(info)
    return support, tuple(infos)


def patch_planes(patch, kk, node, branch, scene, cfg):
    """The per-plane preparation shared by the near and the far scan (10.P,
    10.S): for each plane k of kk, its index in the patch (or None), the
    background's support (the far pixels under "square", the stripe under
    "stripe"; support_masks) and the planar background on it. Returns
    (ks_index (n,) int, -1 when the plane is not in the patch or not valid;
    far (n, H, W) bool; b (n, 3); ok (n,) bool; frame; stripe (n,) tuple of
    info dicts or None). Under "stripe" ok also needs the stripe's own ok
    (stripe_min_pixels, stripe_min_side_um); under "square" the far fraction
    background_min_far_frac."""
    block, ks, valid, frame = patch
    cal = cfg.calibration
    ks = np.asarray(ks)
    H, W = np.shape(block)[1:]
    far, stripe = support_masks((H, W), frame, kk, node, branch, scene, cfg)
    min_frac = cal.background_min_far_frac if stripe is None else 0.0
    idx = np.full(len(kk), -1, dtype=int)
    b = np.full((len(kk), 3), np.nan)
    ok = np.zeros(len(kk), dtype=bool)
    for q, k in enumerate(kk):
        i = int(k - ks[0])
        if 0 <= i < ks.size and bool(valid[i]):
            idx[q] = i
            b[q], _n, ok[q] = calibration_scene.planar_background(block[i], far[q], frame, node.cx_um, node.cy_um,
                                                                  cal.background_trim_sigmas, min_frac)
            if stripe is not None and not stripe[q]["ok"]:
                ok[q] = False
    return idx, far, b, ok, frame, stripe


def patch_images(patch, idx, far, kk, node):
    """The images record of a scan that keeps them (never written to the scans JSON): the patch stack (uint8),
    the far masks, the frame and the line's geometry (the pilot's centre and heading)."""
    block, ks, valid, frame = patch
    keep = idx >= 0
    return dict(ks=np.asarray(kk)[keep], stack=np.asarray(block, dtype=np.uint8)[idx[keep]], far=far[keep],
                left=int(frame.left), top=int(frame.top), res_um_px=float(frame.res_um_px),
                cx_um=float(node.cx_um), cy_um=float(node.cy_um), theta_rad=float(node.theta_rad))


def plane_scan(patch, node, branch, scene, cfg, keep_images=False):
    """Scan the planes k* - P .. k* + P of a measured node along its fixed line
    on the +-patch_half_um patch (10.P; D-078.1, D-081.2).

    patch = (block, ks, valid, frame) in the fetch_zblock contract, over the
    planes stage_planes(node, cfg, "near") about the pilot's centre; node a
    Block 5 NodeResult with a sharpest plane (finite z_sub_um); branch the
    node's Branch; scene the cell's Scene (Block 13.2; the far pixels mask
    every traced segment, the node's own included); cfg a DiameterConfig. In
    plane k: the far pixels and the planar background B_hat_k (10.P); the
    Eq. 9 profile I_k(v) along the line (origin (cx, cy), measuring axis
    (-sin theta, cos theta), along-branch averaging on); the flattened profile
    I'(v) = I(v) - B_hat(line(v)) + B_hat(line(0)); dip_statistics(v, I',
    B_hat(line(0))), unchanged inside. Per plane the scan keeps the dip
    w = B_hat(line(v)) - I(v) and the background along the line (D-075)."""
    block, ks, valid, frame = patch
    m, cal, dz = cfg.measure, cfg.calibration, cfg.acquisition.dz_um
    # NodeResult.k_star is -1 when no plane was found, but -1 is also a plane of a phantom
    # stack (planes numbered about 0), so the test is on the sub-plane depth
    if not (math.isfinite(node.z_sub_um) and math.isfinite(node.cx_um) and math.isfinite(node.cy_um)):
        raise ValueError("plane_scan: the node has no sharpest plane")
    o, y_hat, e_u = line_geometry(node)
    v = profiles.profile_offsets(m)
    kk = stage_planes(node, cfg, "near")
    n = kk.size
    idx, far, bco, bok, frame, stripe = patch_planes(patch, kk, node, branch, scene, cfg)
    names = ("omega_um2", "vw_um2", "area_gl_um", "depth_gl", "B_bar", "se_omega_um2", "amp_gl", "v_c_um", "rms_gl",
             "window_half_um")
    cols = {name: np.full(n, np.nan) for name in names}
    n_win, passes = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    w_gl, bhat = np.full((n, v.size), np.nan), np.full((n, v.size), np.nan)
    status = []
    xy = o[None, :] + v[:, None] * y_hat[None, :]
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
        st = dip_statistics(v, I - B_line + B0, B0, m, cal)
        for name in names:
            if name != "B_bar":
                cols[name][q] = st[name]
        n_win[q], passes[q] = st["n_win"], st["passes"]
        status.append(st["status"])
    return PlaneScan(int(node.node_id), float(node.d_hat_um), float(node.phi_rad), float(node.alpha_hat),
                     int(node.k_star), float(node.z_sub_um), kk, kk * dz, cols["omega_um2"], cols["vw_um2"],
                     cols["area_gl_um"], cols["depth_gl"], cols["B_bar"], tuple(status), "near",
                     cols["se_omega_um2"], cols["amp_gl"], cols["v_c_um"], cols["rms_gl"], n_win, passes,
                     cols["window_half_um"], None, bco, v, w_gl, bhat,
                     patch_images(patch, idx, far, kk, node) if keep_images else None, stripe)


CALIBRATION_SETS = ("near", "far")      # D-067.4: the near-field (thin) set and the far-field set (D-026)


def calibration_set(allen_diameter_um, cfg):
    """The calibration set of a node by Allen's diameter D_A = allen_diameter_um
    (um; D-061.4, D-067.4): "near" when 0 < D_A <= node_allen_diameter_max_um,
    "far" when lo < D_A <= hi with (lo, hi) = far_field_allen_diameter_um, ""
    otherwise (D_A not finite or <= 0 included). validate() keeps the two sets
    disjoint."""
    cal = cfg.calibration
    d = float(allen_diameter_um)
    if not (math.isfinite(d) and d > 0):
        return ""
    if d <= cal.node_allen_diameter_max_um:
        return "near"
    lo, hi = cal.far_field_allen_diameter_um
    return "far" if lo < d <= hi else ""


def criteria_reasons(phi_rad, alpha_hat, allen_diameter_um, cfg):
    """The calibration criteria other than S, on numbers (D-067.4), in this
    order: Allen's diameter (allen_diameter_missing when it is not finite or
    <= 0, allen_diameter_out when it is in neither set), flat (phi_rad <= the
    set's bound: node_phi_max_deg for the near set and a node in neither set,
    node_phi_max_deg_far for the far set; D-101.1; else not_flat), faint
    (alpha_hat <= node_alpha_max, else not_faint) -- only when
    node_alpha_check, off by default (D-094); a NaN fails its criterion.
    Empty: all met."""
    cal = cfg.calibration
    d = float(allen_diameter_um)
    out = []
    which = ""
    if not (math.isfinite(d) and d > 0):
        out.append("allen_diameter_missing")
    else:
        which = calibration_set(d, cfg)
        if not which:
            out.append("allen_diameter_out")
    phi_max = cal.node_phi_max_deg_far if which == "far" else cal.node_phi_max_deg
    if not float(phi_rad) <= math.radians(phi_max):
        out.append("not_flat")
    if cal.node_alpha_check and not float(alpha_hat) <= cal.node_alpha_max:
        out.append("not_faint")
    return out


def selection_flags(cfg):
    """The flags that fail a calibration node: Block 6's SELECTION_FLAGS (the
    flags of S), preceded by "faint" when node_faint_flag_check (D-094). Since
    D-096 S itself has no flag faint, so the switch, off by default, adds it
    for the calibration only."""
    return (("faint",) if cfg.calibration.node_faint_flag_check else ()) + tuple(phantoms.SELECTION_FLAGS)


def selection_reasons(result, cfg):
    """The reasons of Block 6's selection S that fail a calibration node:
    phantoms.reject_reasons (the fit's status, the registration, the flags of
    SELECTION_FLAGS), with the flag "faint" put first among the flags when
    node_faint_flag_check and the node carries it (D-094, D-096)."""
    base = phantoms.reject_reasons(result)
    if not (cfg.calibration.node_faint_flag_check and "faint" in result.flags):
        return base
    head = [r for r in base if r not in phantoms.SELECTION_FLAGS]          # the status and the registration
    return head + ["faint"] + [r for r in base if r in phantoms.SELECTION_FLAGS]


def trace_reasons(view, cfg, allen_diameter_um, branch, index, scene, planes_z):
    """The criteria read from the trace and the pilot (10.I; D-081.1, D-081.3,
    D-093), in this order: not_straight -- only when node_straight_check (off
    by default, D-093: the calibration evaluates only the measuring line) --
    the branch's points within +- node_straight_half_um along the trace stray
    farther than node_straight_tol_radii x (Allen's diameter / 2) from their
    own total-least-squares line; not_isolated -- a traced neighbour (not
    the node's own branch; the soma included) throws a shadow deeper than
    isolation_depth_gl into the +-patch_half_um patch about the pilot's centre
    in some plane of planes_z (depths, um), under the renderer's current law,
    with the pilot's B_bar and mu. The isolation test is skipped (no reason)
    when the pilot's centre, B_bar or mu is not finite: such a node fails S
    already."""
    cal = cfg.calibration
    out = []
    if cal.node_straight_check:
        excess = calibration_scene.straightness_excess(branch, index, cal.node_straight_half_um)
        r = 0.5 * float(allen_diameter_um)
        if math.isfinite(excess) and not (math.isfinite(r) and r > 0 and excess <= cal.node_straight_tol_radii * r):
            out.append("not_straight")
    if scene is not None and all(math.isfinite(float(x)) for x in (view.cx_um, view.cy_um, view.B_bar,
                                                                    view.mu_hat_per_um)):
        if cal.background_support == "stripe":
            out += stripe_reasons(view, cfg, branch, index, scene, planes_z)
        else:
            depth, _who = calibration_scene.isolation_depth(branch, index, scene, view.cx_um, view.cy_um,
                                                            cal.patch_half_um, planes_z, view.B_bar,
                                                            view.mu_hat_per_um, cfg)
            if depth > cal.isolation_depth_gl:
                out.append("not_isolated")
    return out


def stripe_reasons(view, cfg, branch, index, scene, planes_z):
    """10.I (vi) under background_support "stripe" (Block 10.S, D-101): the
    stripe of the plane of planes_z (the stage's planes, um) nearest the
    pilot's depth, computed on the patch's frame without its image
    (support_masks with patch_frame) and with the soma at the stage's largest
    blur, sigma_r at half the span of planes_z -- so that the verdict is the
    scan's stripe at that plane (F1 of 2026-10-28); the reason
    "not_isolated:too_few" (which takes precedence) or
    "not_isolated:side_short" when it is not ok, else nothing. Returns a
    list."""
    cal, p = cfg.calibration, cfg.acquisition.res0_um
    z = np.asarray(planes_z, dtype=float).reshape(-1)
    z_star = float(z[int(np.argmin(np.abs(z - float(view.z_sub_um))))]) if math.isfinite(float(view.z_sub_um)) else float(z[z.size // 2])
    span = 0.5 * (z.max() - z.min()) if z.size > 1 else 0.0
    shape, frame = calibration_scene.patch_frame(view.cx_um, view.cy_um, cal.patch_half_um, p)
    _support, stripe = support_masks(shape, frame, np.array([z_star / cfg.acquisition.dz_um]), view, branch, scene,
                                     cfg, sigma_soma_um=float(sigma_r(span, cfg.renderer)))
    info = stripe[0]
    return [] if info["ok"] else ["not_isolated:" + info["reason"]]


def calibration_reasons(result, cfg, allen_diameter_um, branch=None, index=None, scene=None, planes_z=None):
    """Why a NodeResult is not a calibration node (empty: it is one, of the set
    calibration_set(allen_diameter_um, cfg)): outside the selection S (Block 6's
    rule; selection_reasons: the flag faint only with node_faint_flag_check,
    D-094), then criteria_reasons -- every criterion read from the pilot
    (pilot_view; Block 13.3b, D-061.2, D-067.4) except Allen's diameter
    allen_diameter_um (um; 2 x the branch radius at the node, D-061.4) --,
    then, when branch and index are given, trace_reasons (10.I: not_straight
    with node_straight_check, D-093; not_isolated when scene and planes_z are
    given too). The fit's d_hat is no
    criterion: the kernel being calibrated must not choose its own nodes."""
    view = pilot_view(result)
    out = selection_reasons(view, cfg) + criteria_reasons(view.phi_rad, view.alpha_hat, allen_diameter_um, cfg)
    if branch is not None and index is not None:
        if planes_z is None:
            scene = None
        out += trace_reasons(view, cfg, allen_diameter_um, branch, int(index), scene, planes_z)
    return out


def scan_verdict(scan, reasons):
    """The log phrase of a scan_node result: scanned (near or far field), or
    not a calibration node with its reasons."""
    if scan is not None:
        return "scanned (%s field)" % scan.set
    return "not a calibration node (%s)" % ";".join(reasons)


def scan_node(branch, i, provider, cfg, reg=None, scene=None, keep_images=False):
    """Measure node i's pilot (measure_pilot: Block 5, D-061.2) and select it by
    calibration_reasons with Allen's diameter 2 x branch.radius_um[i] (D-061.4,
    D-067.4) and the trace criteria of 10.I (scene: the cell's Scene, Block
    13.2; None: Scene.from_branch(branch), a phantom's own tube). A calibration
    node: fetch its patch (+-patch_half_um about the pilot's centre, one
    provider call) over the stage's planes and scan it -- the near set by
    plane_scan, the far set by far_field.far_plane_scan (Block 15.1). Returns
    (NodeResult, PlaneScan or None, reasons)."""
    from . import far_field
    res = measure_pilot(branch, i, provider, cfg, reg)
    allen_d = 2.0 * float(branch.radius_um[i])
    scene = Scene.from_branch(branch) if scene is None else scene
    which = calibration_set(allen_d, cfg) or "near"
    dz = cfg.acquisition.dz_um
    planes_z = stage_planes(res, cfg, which) * dz if math.isfinite(res.z_sub_um) else None
    reasons = calibration_reasons(res, cfg, allen_d, branch, i, scene, planes_z)
    if reasons:
        return res, None, reasons
    kk = stage_planes(res, cfg, which)
    patch = calibration_scene.fetch_patch(provider, res.cx_um, res.cy_um, cfg.calibration.patch_half_um, kk[0], kk[-1],
                                          cfg.acquisition.res0_um)
    if which == "near":
        return res, plane_scan(patch, res, branch, scene, cfg, keep_images), []
    return res, far_field.far_plane_scan(patch, res, branch, scene, cfg, keep_images), []


def scan_cell(swc, provider, cfg, transform=None, regs=None, candidates=None, log=None, keep_images=False,
              order_rng=None, max_per_set=None, on_node=None):
    """Plane scans of one cell's calibration nodes, both sets. Every dendrite
    stretch (Block 8) holding a node id of `candidates` (None: every dendrite
    node) is built in the image frame (transform: keyword arguments of
    cell.to_image_um) and each such node goes through scan_node with the
    cell's scene (cell.cell_scene, D-064; regs: {node_id: registration dict};
    Allen's diameter from the SWC radius the branch carries). Block 19.1
    (D-088.3): with order_rng the nodes are taken in the order of
    order_rng.permutation over them (the SWC's order without it); with
    max_per_set a node whose set (by Allen's diameter, calibration_set)
    already holds max_per_set scans is not measured, reason set_full.
    Returns a list of (node_id, NodeResult or None, PlaneScan or None,
    reasons); log, when given, gets one scan_verdict line per measured node
    and, at the end, one line counting the set_full nodes (2026-10-29: the
    whole-cell search of Block 10.C hands over hundreds of candidates); on_node,
    when given, is called after each measured node as on_node(node_id,
    branch, index, NodeResult, PlaneScan or None, reasons, scene) -- Cell 7's
    isolation figure (10.F (5), D-098.4); not for a set_full node."""
    types = cfg.acquisition.dendrite_swc_types
    xyz_img = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um, **(transform or {}))
    scene = cell.cell_scene(swc, types, cfg.acquisition.res0_um, transform)
    work = []
    for run in cell.stretches(swc, types):
        ids = [int(swc.ids[r]) for r in run]
        if candidates is not None and not any(i in candidates for i in ids):
            continue
        branch, off = cell.stretch_branch(swc, run, xyz_img)
        for t, nid in enumerate(ids):
            if candidates is None or nid in candidates:
                work.append((nid, branch, off + t))
    if order_rng is not None:
        work = [work[k] for k in order_rng.permutation(len(work))]
    counts, out, full = {}, [], {}
    for nid, branch, i in work:
        which = calibration_set(2.0 * float(branch.radius_um[i]), cfg)
        if max_per_set is not None and which is not None and counts.get(which, 0) >= int(max_per_set):
            out.append((nid, None, None, ["set_full"]))
            full[which] = full.get(which, 0) + 1
            continue
        res, scan, reasons = scan_node(branch, i, provider, cfg, (regs or {}).get(nid), scene, keep_images)
        if scan is not None:
            counts[scan.set] = counts.get(scan.set, 0) + 1
        out.append((nid, res, scan, reasons))
        if log is not None:
            log("node %d: %s" % (nid, scan_verdict(scan, reasons)))
        if on_node is not None:
            on_node(nid, branch, i, res, scan, reasons, scene)
    if log is not None and full:
        log("not measured, their set already holding %d scans (set_full): %s" % (
            int(max_per_set), ", ".join("%d %s" % (full[w], w) for w in sorted(full))))
    return out


def median_by_offset(scans, which=None):
    """{k - k*: median over the nodes of omega_hat at that offset} over the scans
    of set `which` (None: all), finite values only."""
    acc: Dict[int, List[float]] = {}
    for sc in scans:
        if which is not None and sc.set != which:
            continue
        for o, w in zip(sc.offsets(), sc.omega_um2):
            if np.isfinite(w):
                acc.setdefault(int(o), []).append(float(w))
    return {o: float(np.median(v)) for o, v in sorted(acc.items())}


def example_node(scans, which=None):
    """The node nearest the median width by plane (10.F, D-081.5): the scan of
    set `which` whose mean relative deviation |omega_hat - median_offset| /
    median_offset over its finite planes is smallest (at least two such
    planes). Returns its node_id, or None."""
    med = median_by_offset(scans, which)
    best, best_id = None, None
    for sc in scans:
        if which is not None and sc.set != which:
            continue
        dev = [abs(w - med[int(o)]) / med[int(o)] for o, w in zip(sc.offsets(), sc.omega_um2)
               if np.isfinite(w) and int(o) in med and med[int(o)] > 0]
        if len(dev) >= 2 and (best is None or float(np.mean(dev)) < best):
            best, best_id = float(np.mean(dev)), int(sc.node_id)
    return best_id


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


# ------------------------------------------------------------ the anchor --

def anchor_from_growth(growth, cfg, sigma_r0_um=None):
    """The anchor sigma_0_hat = sigma_r(table_max_delta_um) (um) of a growth fit
    (10.A): the last value of the renderer table kernel_from_growth builds with
    the cut at cfg.calibration.table_max_delta_um -- (sigma_r(0)^2 +
    G_hat(delta_0))^(1/2) with sigma_r(0) configured or given. ValueError when
    that knot is not identified (NaN) or the table is refused."""
    k = kernel_from_growth(growth, cfg.renderer, sigma_r0_um, max_delta_um=cfg.calibration.table_max_delta_um)
    if abs(k.kernel_table_delta_um[-1] - cfg.calibration.table_max_delta_um) > 1e-9:
        raise ValueError("anchor_from_growth: the growth has no knot at table_max_delta_um = %g um"
                         % cfg.calibration.table_max_delta_um)
    return float(k.kernel_table_sigma_um[-1])


def anchor_value(growth, cfg, sigma_r0_um=None):
    """The anchor (um) read from the growth's knot at delta_0 =
    cfg.calibration.table_max_delta_um alone (10.A, D-115.4):
    (sigma_r(0)^2 + G(delta_0))^(1/2), sigma_r(0) = cfg.renderer.sigma_r0_um
    unless given. Unlike anchor_from_growth, the other knots need not give an
    accepted table. ValueError for a two-sided growth (the renderer's kernel
    is even in delta), when no knot sits at delta_0, or when sigma_r(0)^2 +
    G(delta_0) is not a positive number (G not identified there, or too
    low)."""
    if growth.convention != "symmetric":
        raise ValueError("anchor_value: the renderer's kernel is even in delta; a two-sided growth has no one anchor")
    d0 = float(cfg.calibration.table_max_delta_um)
    hit = np.flatnonzero(np.abs(np.asarray(growth.knots_um, dtype=float) - d0) <= 1e-9)
    if hit.size != 1:
        raise ValueError("anchor_value: the growth has no knot at table_max_delta_um = %g um" % d0)
    s0 = float(cfg.renderer.sigma_r0_um if sigma_r0_um is None else sigma_r0_um)
    s2 = s0 ** 2 + float(growth.g_um2[int(hit[0])])
    if not (s0 > 0 and math.isfinite(s2) and s2 > 0):
        raise ValueError("anchor_value: sigma_r(0)^2 + G(delta_0) = %r is not a positive number" % s2)
    return math.sqrt(s2)


def bootstrap_sigma0(scans, cfg, rng, sigma_r0_um=None, n_resamples=None):
    """(se_um, n_ok, draws): the SE of the anchor by a node bootstrap (10.A,
    D-081.7) -- n_resamples (default cfg.calibration.bootstrap_resamples)
    resamples of the scans with replacement (rng: numpy Generator), each
    refitted by fit_growth and read by anchor_value, from G(delta_0) alone
    (D-115.4: a table refused at another knot no longer drops a resample); a
    resample with fewer than two usable nodes, a failed fit or an anchor not
    identified at delta_0 is dropped (n_ok counts the kept ones). se_um is
    NaN below two kept draws."""
    n = int(cfg.calibration.bootstrap_resamples if n_resamples is None else n_resamples)
    scans = list(scans)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(scans), len(scans))
        sample = [replace(scans[j], node_id=1000000 + t) for t, j in enumerate(pick)]
        try:
            g = fit_growth(sample, cfg)
            draws.append(anchor_value(g, cfg, sigma_r0_um))
        except ValueError:
            continue
    draws = np.array(draws, dtype=float)
    se = float(np.std(draws, ddof=1)) if draws.size >= 2 else float("nan")
    return se, int(draws.size), draws
