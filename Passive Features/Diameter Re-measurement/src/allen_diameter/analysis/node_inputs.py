"""Node inputs of the pilot run of the cleaning and the per-node fit -- Block 36 in specs/SPEC.md
(plan steps C0, C5-C6, F5; D-129, D-130, D-132, D-136, D-137, D-138).

The functions the two entry points (scripts/clean_cell.py, scripts/infer_nodes.py) need to turn a
traced node and its pilot (Block 5's measure_pilot) into the inputs of Blocks 32 (the cleaning
iteration) and 30 (the per-node fit):

  own_branch        the stretch's own nodes as a Branch, without the parent node that
                    cell.stretch_branch prepends (D-136.1)
  own_ids_within    the node's own nodes (amendment 3, D-143): on each side along the stretch up to
                    and including the first node whose arc exceeds inference.tube_half_length_um
                    (or the stretch's end); U_minus, U_plus = the arcs to those outermost nodes
  tube_extent       (U_minus, U_plus) of own_ids_within at tube_half_length_um; near_end when
                    min(U_minus, U_plus) < inference.extrusion_cap_margin_um (D-138.2, D-140.3)
  axis_extent       the stretch's two sides mapped onto the model tube's axis a_hat (see there)
  local_tube        the node's straight capped tube, shifts relative to the trace (amendment 1,
                    D-141): centre at the foot of the traced position on the measuring line plus
                    v0 y_hat, i.e. (c_x, c_y) + (v0 - v_t) y_hat with v_t = ((c_x, c_y) - (x_t, y_t))
                    . y_hat, depth z_t + dz; radius d / 2, heading pilot.theta_rad, tilt
                    pilot.phi_rad, aspect 1, from -U_minus to +U_plus along a_hat (amendment 3);
                    y_hat = (-sin theta, cos theta)
  pilot_start       theta0 = (d_hat, mu_hat, v_t, z_sub - z_t), the first-stage start
  node_planes       k* - P .. k* + P, P = cleaning.plane_half_range
  build_node        the NodeData of Block 32 from one patch fetch (calibration_scene.fetch_patch),
                    each plane's planar background and noise SD (calibration.patch_planes,
                    with_sd=True, the support cleaning.background_support), B_k the background at
                    the line's origin (calibration_scene.background_on_line), the measuring line of
                    Block 5's pilot, node_xyz the traced position
  all_shadow        Block 25's predicted shadow of every traced segment and soma disc on a node's
                    patch (amendment 4, D-144)
  model_background  the planar background fitted through the shadow model, per plane (amendment 4)
  background_factory  run_rounds' background_fn: model_background refitted each round (amendment 4)
  predict_factory   Block 32's predict: 1 - line_fast.line_profiles of the local tube at
                    (v0, dz) about the trace (D-138.4); ExtrusionInvalid -> NaN
  forward_factory   Block 30's forward: line_fast.line_profiles of the local tube at absolute
                    depth z_ax = z_t + dz
  work_list         the example nodes and the end nodes of their strong tier-1 neighbours

Units: um, 1/um, rad, grey levels (gl). Custom code: none numerical. No file access, no
plotting: the provider is an argument. Pure ASCII, LF, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np

from . import calibration, calibration_scene, cell, cleaning, profiles
from .clean_iteration import NodeData
from .node_pipeline import Branch
from ..model import geometry, line_fast, shadow

# reasons build_node gives for a refusal (info["reason"])
REASON_NEAR_END = "near_end"                 # D-138.2: the tube's caps are within the margin of the line
REASON_NO_PLANE = "no_sharpest_plane"        # the pilot found no sharpest plane (z_sub_um not finite)
REASON_NO_FIT = "no_pilot_fit"               # the pilot's (d_hat, mu_hat) are not usable as a start
REASON_MISSING = "missing_plane"             # a plane of node_planes is not in the patch or not valid
REASON_NO_BG = "no_background"               # a plane of node_planes has no valid planar background
REASON_LINE = "line_leaves_patch"            # the measuring line reads outside the patch (NaN profile)
REFUSALS = (REASON_NEAR_END, REASON_NO_PLANE, REASON_NO_FIT, REASON_MISSING, REASON_NO_BG, REASON_LINE)


# ------------------------------------------------------------ geometry ---

def own_branch(swc, rows, xyz_img):
    """(Branch, index_offset) of a stretch's own nodes (row indices `rows` of the SWC, proximal -> distal,
    as cell.stretches returns them), positions xyz_img (N, 3) um in the image frame: no parent node
    prepended (D-136.1), so index_offset is 0 and branch.ids are exactly the stretch's node ids.
    None for a stretch of fewer than 2 nodes."""
    idx = np.asarray([int(r) for r in rows], dtype=int)
    if idx.size < 2:
        return None
    P = np.asarray(xyz_img, dtype=float)[idx]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return Branch(np.asarray(swc.ids)[idx], np.asarray(swc.types)[idx], P,
                  np.asarray(swc.radius, dtype=float)[idx], s), 0


def own_ids_within(branch, index, half_length_um):
    """-> (ids, U_minus, U_plus) (amendment 3, D-143): the node's own nodes on its stretch -- on each side,
    the nodes up to and including the first whose arc length from node `index` exceeds half_length_um, or
    up to the stretch's end -- and U_minus, U_plus (um), the arc lengths from the node to the outermost own
    node toward the stretch's first node and toward its last. The own segments are those between own nodes;
    the stretch's segments beyond are neighbours for Blocks 25, 28 and 32, and the tube covers exactly the
    own segments (local_tube)."""
    s = np.asarray(branch.s_um, dtype=float)
    ids = np.asarray(branch.ids)
    i = int(index)
    if not 0 <= i < s.size:
        raise ValueError("index %d outside the branch of %d nodes" % (i, s.size))
    T = float(half_length_um)
    lo = i
    while lo > 0 and s[i] - s[lo] <= T:
        lo -= 1
    hi = i
    while hi < s.size - 1 and s[hi] - s[i] <= T:
        hi += 1
    return tuple(int(x) for x in ids[lo:hi + 1]), float(s[i] - s[lo]), float(s[hi] - s[i])


def tube_extent(branch, index, cfg):
    """(U_minus, U_plus) um of own_ids_within at cfg.inference.tube_half_length_um, toward the stretch's first
    and last node. The branch is the node's own stretch (own_branch)."""
    _ids, um, up = own_ids_within(branch, index, cfg.inference.tube_half_length_um)
    return um, up


def is_near_end(branch, index, cfg):
    """True when min(tube_extent) is below cfg.inference.extrusion_cap_margin_um: the node is flagged
    near_end and skipped by the fits (D-138.2, D-140.3; amendment 3)."""
    return bool(min(tube_extent(branch, index, cfg)) < float(cfg.inference.extrusion_cap_margin_um))


def axis_extent(pilot, branch, index, U_minus, U_plus):
    """(U_back, U_fwd) um: the stretch's extents (U_minus toward the first node, U_plus toward the last)
    mapped onto the model tube's axis a_hat = (cos phi cos theta, cos phi sin theta, sin phi) of the pilot.
    Block 5's tilt is phi = asin|t_z| (no sign), so the model tube rises along +e_u = (cos theta, sin theta)
    whether or not the dendrite does. With side A the stretch side whose outermost own node lies further
    along +e_u and B the other: when the trace rises toward A (z_A >= z_B) the model is the dendrite and
    U_fwd = U_A, U_back = U_B; when it falls toward A the model is the dendrite mirrored across the
    measuring line (u -> -u), which the profile reads alike (the line and its 2 n_avg + 1 parallel lines
    are symmetric in u), and U_fwd = U_B, U_back = U_A."""
    s = np.asarray(branch.s_um, dtype=float)
    P = np.asarray(branch.xyz_um, dtype=float)
    i = int(index)
    j_lo = int(np.searchsorted(s, s[i] - float(U_minus) - 1e-9))
    j_hi = int(np.searchsorted(s, s[i] + float(U_plus) - 1e-9))
    j_hi = min(j_hi, s.size - 1)
    eu = np.array([math.cos(float(pilot.theta_rad)), math.sin(float(pilot.theta_rad))])
    u_lo = float(np.dot(P[j_lo, :2] - P[i, :2], eu))
    u_hi = float(np.dot(P[j_hi, :2] - P[i, :2], eu))
    if u_hi >= u_lo:                              # A = the last-node side
        U_A, U_B, z_A, z_B = float(U_plus), float(U_minus), P[j_hi, 2], P[j_lo, 2]
    else:
        U_A, U_B, z_A, z_B = float(U_minus), float(U_plus), P[j_lo, 2], P[j_hi, 2]
    if z_A >= z_B:
        return U_B, U_A
    return U_A, U_B


def trace_offset(pilot, node_xyz_trace):
    """v_t (um): the pilot centre's offset from the traced position along y_hat = (-sin theta, cos theta),
    ((cx, cy) - (x_t, y_t)) . y_hat."""
    t = np.asarray(node_xyz_trace, dtype=float)
    th = float(pilot.theta_rad)
    return float((float(pilot.cx_um) - t[0]) * (-math.sin(th)) + (float(pilot.cy_um) - t[1]) * math.cos(th))


def pilot_start(pilot, node_xyz_trace):
    """theta0 = (d_hat, mu_hat, v_t, z_sub - z_t) (amendment 1, D-141): the pilot's fit as shifts about the trace."""
    t = np.asarray(node_xyz_trace, dtype=float)
    return np.array([float(pilot.d_hat_um), float(pilot.mu_hat_per_um), trace_offset(pilot, t),
                     float(pilot.z_sub_um) - float(t[2])])


def local_tube(pilot, node_xyz_trace, d_um, v0_um, dz_um, U_minus, U_plus):
    """The node's straight tube (Block 2 Tube), shifts about the trace (amendment 1, D-141) and extents along
    the axis (amendment 3, D-143): the node's point (cx, cy) + (v0 - v_t) y_hat -- the foot of the traced
    position on the measuring line plus v0 y_hat -- at depth z_t + dz_um; the tube runs from -U_minus to
    +U_plus along a_hat = (cos phi cos theta, cos phi sin theta, sin phi): centre moved by
    (U_plus - U_minus) / 2 a_hat, half-length (U_plus + U_minus) / 2, axial caps; radius d_um / 2, heading
    pilot.theta_rad, tilt pilot.phi_rad, aspect 1; y_hat = (-sin theta, cos theta). U_minus, U_plus are
    along a_hat (axis_extent maps the stretch's sides onto it). pilot: any object with cx_um, cy_um,
    theta_rad, phi_rad; node_xyz_trace (3,) um, the traced position in the image frame."""
    th, ph = float(pilot.theta_rad), float(pilot.phi_rad)
    yx, yy = -math.sin(th), math.cos(th)
    a = (math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph))
    t = np.asarray(node_xyz_trace, dtype=float)
    s = float(v0_um) - trace_offset(pilot, t)
    m = 0.5 * (float(U_plus) - float(U_minus))
    c = (float(pilot.cx_um) + s * yx + m * a[0], float(pilot.cy_um) + s * yy + m * a[1],
         float(t[2]) + float(dz_um) + m * a[2])
    return geometry.Tube(c=c, r=0.5 * float(d_um), phi=ph, theta=th, aspect=1.0,
                         half_length=0.5 * (float(U_plus) + float(U_minus)), end_cut="axial")


def node_planes(pilot, cfg):
    """The plane indices k* - P .. k* + P (P = cfg.cleaning.plane_half_range), k* = pilot.k_star."""
    P = int(cfg.cleaning.plane_half_range)
    k = int(pilot.k_star)
    return np.arange(k - P, k + P + 1, dtype=np.int64)


def _has_sharpest_plane(pilot):
    return all(math.isfinite(float(getattr(pilot, name))) for name in ("z_sub_um", "cx_um", "cy_um"))


def profile_noise_factor(frame, shape, origin_xy, y_hat, e_u, v, n_avg, step):
    """-> float: sqrt((1 / N_v) sum_n sum_px w_{n,px}^2), w_{n,px} the weight of pixel px in profile sample
    n of profiles.sample_profile (bilinear, the mean over 2 n_avg + 1 lines) -- the profile's noise SD over
    a pixel's for white pixel noise (amendment 2, F2). The read is linear in the image: the weights are read
    from unit impulses over the pixels the line touches (line_render.pixels_needed), each on the box of
    those pixels with one pixel of margin. frame: the patch's CropFrame (left, top, res_um_px); shape: the
    patch's (H, W). ValueError when the line touches a pixel outside the patch."""
    from ..model import line_render
    p = float(frame.res_um_px)
    cols, rows = line_render.pixels_needed(origin_xy, y_hat, e_u, v, n_avg, step, p)
    H, W = int(shape[0]), int(shape[1])
    if (cols.min() < frame.left or rows.min() < frame.top or cols.max() >= frame.left + W
            or rows.max() >= frame.top + H):
        raise ValueError("profile_noise_factor: the line touches pixels outside the patch")
    c0, r0 = int(cols.min()) - 1, int(rows.min()) - 1
    box = np.zeros((int(rows.max()) - r0 + 2, int(cols.max()) - c0 + 2))
    sub = line_render._Frame(c0, r0, p)
    vv = np.asarray(v, dtype=float)
    total = np.zeros(vv.size)
    for c, r in zip(cols, rows):
        box[r - r0, c - c0] = 1.0
        w = profiles.sample_profile(box, sub, origin_xy, y_hat, e_u, vv, n_avg, step)
        box[r - r0, c - c0] = 0.0
        total += w * w
    return float(math.sqrt(np.mean(total)))


# --------------------------------------------------------- node inputs ---

_DUMMY_IDS = (-910000001, -910000002)       # a stand-in own stretch no traced id equals (all_shadow)


def all_shadow(fscene, node_like, cfg, floor_B_gl=None):
    """S_all (n_planes, H, W) float64, dimensionless: Block 25's predicted shadow (cleaning.shadow_absorption)
    of every traced segment and soma disc of fscene -- the node's own stretch included, no floor, nothing left
    out -- on the node's patch (node_like: left, top, block (n_planes, H, W) or shape, planes_z). shadow.
    shadow_image needs an own stretch with a segment, so it is given a stand-in: one segment of zero
    coefficient whose two ends carry ids no traced node has, 1 km from the patch (amendment 4, D-144).
    floor_B_gl (gl): when given, a segment whose shadow's peak times floor_B_gl is under cleaning.shadow_floor_gl
    is left out (D-146.2: such shadows are smooth over the patch and the planar background takes them up);
    None keeps every segment (slow on a whole cell)."""
    from .node_features import FeaturedScene
    shape = np.shape(node_like.block)[1:] if hasattr(node_like, "block") else tuple(node_like.shape)[-2:]
    H, W = int(shape[0]), int(shape[1])
    p = float(cfg.acquisition.res0_um)
    far = np.array([(float(node_like.left) - 1e3 / p) * p, float(node_like.top) * p, 0.0])
    xyz = np.vstack([np.asarray(fscene.seg_xyz, dtype=float).reshape(-1, 6),
                     np.concatenate([far, far + np.array([0.1, 0.0, 0.0])])[None, :]])
    fs = FeaturedScene(seg_xyz=xyz,
                       seg_d_um=np.vstack([np.asarray(fscene.seg_d_um, dtype=float).reshape(-1, 2), [[0.1, 0.1]]]),
                       seg_mu_per_um=np.vstack([np.asarray(fscene.seg_mu_per_um, dtype=float).reshape(-1, 2),
                                                [[0.0, 0.0]]]),
                       seg_ids=np.vstack([np.asarray(fscene.seg_ids, dtype=np.int64).reshape(-1, 2),
                                          [list(_DUMMY_IDS)]]),
                       seg_type=np.concatenate([np.asarray(fscene.seg_type, dtype=np.int64).reshape(-1), [3]]),
                       soma=np.asarray(fscene.soma, dtype=float).reshape(-1, 3))
    return shadow.shadow_image(fs, _DUMMY_IDS, int(node_like.left), int(node_like.top), W, H,
                               np.asarray(node_like.planes_z, dtype=float), cfg, B_gl=floor_B_gl, include=None).S


def model_background(block, S_all, frame, origin_xy, cfg):
    """(B, sd, frac, ok), each (n_planes,): the planar background fitted through the shadow model (amendment
    4, D-144). Per plane k, b(x, y) = b0 + bx (x - cx) + by (y - cy) ((cx, cy) the patch's centre, um)
    minimises sum [I - b (1 - S_all)]^2 over the pixels with S_all <= cleaning.background_shadow_max (and I
    finite), by numpy.linalg.lstsq, then once more without the pixels whose residual exceeds
    calibration.background_trim_sigmas x 1.4826 x MAD about their median (D-145.2). B = b at
    the line's origin (gl); sd = the SD of the kept residuals (gl, 3 parameters fitted); frac = the kept share
    of the patch; ok = frac >= calibration.background_min_far_frac and B finite and > 0."""
    I_all = np.asarray(block, dtype=float)
    S_all = np.asarray(S_all, dtype=float)
    if I_all.shape != S_all.shape or I_all.ndim != 3:
        raise ValueError("block and S_all must be (n_planes, H, W) of one shape")
    n, H, W = I_all.shape
    p = float(frame.res_um_px)
    px, py = calibration_scene.pixel_centres((H, W), frame)
    cx, cy = float(px.mean()), float(py.mean())
    o = np.asarray(origin_xy, dtype=float)
    smax = float(cfg.cleaning.background_shadow_max)
    trim = float(cfg.calibration.background_trim_sigmas)
    min_frac = float(cfg.calibration.background_min_far_frac)
    B, sd, frac = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    X, Y = (px - cx).ravel(), (py - cy).ravel()
    for k in range(n):
        I, S = I_all[k].ravel(), S_all[k].ravel()
        keep = (S <= smax) & np.isfinite(I) & np.isfinite(S)
        if keep.sum() < 6:
            continue
        A = (1.0 - S)[:, None] * np.column_stack([np.ones_like(X), X, Y])
        b = np.linalg.lstsq(A[keep], I[keep], rcond=None)[0]
        res = I[keep] - A[keep] @ b
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med)))
        if mad > 0:
            # residuals centred on their median before the comparison (final check of 2026-11-14 18:30, F2;
            # D-145.2): uncentred, a small offset of the model beside a small MAD dropped the whole support
            inner = np.abs(res - med) <= trim * 1.4826 * mad
            if 6 <= inner.sum() < inner.size:
                idx = np.flatnonzero(keep)[inner]
                keep = np.zeros_like(keep)
                keep[idx] = True
                b = np.linalg.lstsq(A[keep], I[keep], rcond=None)[0]
                res = I[keep] - A[keep] @ b
        B[k] = b[0] + b[1] * (o[0] - cx) + b[2] * (o[1] - cy)
        sd[k] = float(np.sqrt(np.sum(res * res) / max(res.size - 3, 1)))
        frac[k] = keep.sum() / float(H * W)
    ok = (frac >= min_frac) & np.isfinite(B) & (B > 0)
    return B, sd, frac, ok


def background_factory(cfg, log=None):
    """-> background_fn(node, F) -> (n_planes,) gl for clean_iteration.run_rounds (Block 32, amendment 2):
    model_background on all_shadow(F, node, cfg), refitted each round from the featured scene of the round
    before (amendment 4, D-144). A round in which a plane of a node has no valid background keeps that
    node's previous backgrounds (NodeData.B_gl at first) and calls log(message). The function keeps
    background_fn.history: node_id -> list of dicts per call (B_gl, sd_gl, frac, ok, kept_previous,
    seconds)."""
    from allen_image_io import CropFrame
    last, history = {}, {}

    def background_fn(node, F):
        import time as _time
        t0 = _time.time()
        nid = int(node.node_id)
        prev = last.get(nid, np.asarray(node.B_gl, dtype=float))
        S_all = all_shadow(F, node, cfg, floor_B_gl=float(np.max(prev)))     # the floor at the last backgrounds
        B, sd, frac, ok = model_background(node.block, S_all, CropFrame(int(node.left), int(node.top), 0,
                                                                         float(cfg.acquisition.res0_um)),
                                           node.origin_xy, cfg)
        kept = not bool(np.all(ok))
        out = np.array(prev, dtype=float) if kept else B
        if kept and log is not None:
            log("node %d: planes %s without a valid model background this round; the previous backgrounds kept"
                % (nid, [int(k) for k, g in zip(node.ks, ok) if not g]))
        last[nid] = out
        history.setdefault(nid, []).append(dict(B_gl=out, B_fit_gl=B, sd_gl=sd, frac=frac, ok=ok,
                                                kept_previous=kept, seconds=_time.time() - t0))
        return out
    background_fn.history = history
    return background_fn


def unshaded_support(shape, frame, ks, pilot, branch, scene, cfg):
    """(n_planes, H, W) bool: the pixels of each plane of ks that neither a traced neighbour nor the node's
    own branch shades by more than calibration.isolation_depth_gl (amendment 3, D-143): Block 10.S's masks
    (calibration_scene.shadow_masks from the pilot's B_bar and mu_hat, the own branch at the pilot's d_hat
    and moved to its fitted centre) without the stripe's cut. The rows are built as
    calibration.support_masks builds them: segment_rows with its reach (the larger of the scene's radii
    and d_hat / 2, plus 6 sigma_r at the largest defocus of any segment end from any plane), then
    fitted_own_rows."""
    from ..model.kernel import sigma_r
    dz = float(cfg.acquisition.dz_um)
    z = np.asarray(ks, dtype=float) * dz
    H, W = int(shape[0]), int(shape[1])
    seg_z = scene.seg_xyz[:, [2, 5]].ravel() if scene.seg_xyz.shape[0] else np.array([float(pilot.z_sub_um)])
    d_max = float(np.max(np.abs(seg_z[:, None] - z[None, :])))
    r_max = float(np.max(np.concatenate([np.asarray(scene.seg_radius, dtype=float).ravel(), [0.0]])))
    d_own = float(pilot.d_hat_um)
    reach = max(r_max, 0.5 * d_own if math.isfinite(d_own) else 0.0) + 6.0 * float(sigma_r(d_max, cfg.renderer))
    rows = calibration_scene.segment_rows(scene, frame, (H, W), branch, reach)
    rows = np.vstack([rows, calibration_scene.fitted_own_rows(rows, pilot.cx_um, pilot.cy_um)])
    nb, own = calibration_scene.shadow_masks((H, W), frame, rows, scene.soma, z, pilot.B_bar, pilot.mu_hat_per_um,
                                             cfg, d_own)
    return ~(np.asarray(nb, dtype=bool) | np.asarray(own, dtype=bool))


def unshaded_planes(patch, ks, pilot, branch, scene, cfg):
    """The per-plane background on unshaded_support (amendment 3): (idx (n,) int, the plane's index in the
    patch or -1; b (n, 3); ok (n,) bool; frame; sd (n,) the noise SD on the support, gl; frac (n,) the
    support's share of the patch). calibration_scene.planar_background with background_trim_sigmas, valid
    when the support holds at least background_min_far_frac of the patch."""
    block, pks, valid, frame = patch
    pks = np.asarray(pks)
    H, W = np.shape(block)[1:]
    cal = cfg.calibration
    sup = unshaded_support((H, W), frame, ks, pilot, branch, scene, cfg)
    n = len(ks)
    idx = np.full(n, -1, dtype=int)
    b = np.full((n, 3), np.nan)
    ok = np.zeros(n, dtype=bool)
    sd = np.full(n, np.nan)
    for q, k in enumerate(ks):
        i = int(k - pks[0])
        if 0 <= i < pks.size and bool(valid[i]):
            idx[q] = i
            b[q], _n, ok[q], sd[q] = calibration_scene.planar_background(
                block[i], sup[q], frame, float(pilot.cx_um), float(pilot.cy_um), cal.background_trim_sigmas,
                cal.background_min_far_frac, return_sd=True)
    return idx, b, ok, frame, sd, sup.mean(axis=(1, 2))


def build_node(branch, index, pilot, provider, scene, cfg, own_ids, fscene, theta0=None):
    """-> (NodeData or None, info). branch: the node's own stretch (own_branch); index: the node's index
    in it; pilot: the node's Block 5 pilot (NodeResult: cx_um, cy_um, theta_rad, phi_rad, z_sub_um,
    k_star, d_hat_um, mu_hat_per_um, B_bar); provider(left, top, width, height, k_lo, k_hi) -> (block,
    ks, valid, frame); scene: the cell's plane_iteration.Scene; cfg: DiameterConfig; own_ids: the
    node's own ids (own_ids_within); fscene: the round-0 featured scene (node_features.featured_scene),
    read under "model"; theta0: (d, mu, v0, dz) about the trace, default pilot_start.

    One provider call: the patch of half-size cleaning.patch_half_um about the pilot's centre over
    node_planes; per plane its planar background and noise SD; B_k the background at the line's origin.
    The support is cfg.cleaning.background_support: "model" (amendment 4, D-144, the default) -- the
    planar background fitted through the shadow model (model_background on all_shadow of fscene), refitted
    each round by background_factory; "unshaded" (amendment 3, D-143) -- the
    pixels neither a traced neighbour nor the own branch shades above calibration.isolation_depth_gl
    (unshaded_support), the planar background by calibration_scene.planar_background on them; "square" and
    "stripe" -- calibration.patch_planes with calibration.background_support replaced by it (amendment 1). The line:
    calibration.line_geometry (origin the pilot's centre), v = profiles.profile_offsets, n_avg =
    profiles.n_along, step = profile_step_um (Block 5's pilot). node_xyz = the traced position,
    branch.xyz_um[index].

    info: node_id, U_stretch_um (U_minus, U_plus toward the stretch's first and last node, tube_extent),
    near_end, ks, U_axis_um (the same along the model axis, axis_extent; the tube's extents), and on success sd_gl (the planes' pixel noise SDs, gl),
    sigma_pixel_gl (their median), noise_factor (profile_noise_factor of the line) and sigma_gl = their
    product, the profile sample's noise SD (amendment 2, F2); on a refusal reason (one of REFUSALS) and detail. Refusals:
    near_end (checked first, no fetch), no sharpest plane, no usable pilot fit for the default start,
    a plane missing or invalid, a plane with no valid background, a line that leaves the patch."""
    m, dz = cfg.measure, float(cfg.acquisition.dz_um)
    i = int(index)
    nid = int(np.asarray(branch.ids)[i])
    Um, Up = tube_extent(branch, i, cfg)
    U = min(Um, Up)
    info = {"node_id": nid, "U_stretch_um": (Um, Up), "near_end": U < float(cfg.inference.extrusion_cap_margin_um)}

    def refuse(reason, detail=""):
        info.update(reason=reason, detail=detail)
        return None, info

    if info["near_end"]:
        return refuse(REASON_NEAR_END, "tube extent %.3f um on one side < margin %.3f um"
                      % (U, float(cfg.inference.extrusion_cap_margin_um)))
    if not _has_sharpest_plane(pilot):
        return refuse(REASON_NO_PLANE, "the pilot has no sharpest plane (z_sub_um not finite)")
    trace = np.asarray(branch.xyz_um, dtype=float)[i]
    info["U_axis_um"] = axis_extent(pilot, branch, i, Um, Up)
    if theta0 is None:
        theta0 = pilot_start(pilot, trace)
    th0 = np.asarray(theta0, dtype=float)
    if th0.shape != (4,) or not np.all(np.isfinite(th0)) or not th0[0] > 0 or th0[1] < 0:
        return refuse(REASON_NO_FIT, "start %r not usable (the pilot's fit status %r)"
                      % (tuple(float(x) for x in th0.ravel()), str(getattr(pilot, "fit_status", ""))))
    ks = node_planes(pilot, cfg)
    info["ks"] = ks
    patch = calibration_scene.fetch_patch(provider, float(pilot.cx_um), float(pilot.cy_um),
                                          float(cfg.cleaning.patch_half_um), int(ks[0]), int(ks[-1]),
                                          float(cfg.acquisition.res0_um))
    if cfg.cleaning.background_support == "model":
        block_, pks, valid, frame = patch
        pks = np.asarray(pks)
        idx = np.array([int(k - pks[0]) if 0 <= int(k - pks[0]) < pks.size and bool(valid[int(k - pks[0])])
                        else -1 for k in ks])
        bco = np.full((ks.size, 3), np.nan)
        ok = np.zeros(ks.size, dtype=bool)
        sd = np.full(ks.size, np.nan)
        if np.all(idx >= 0):
            sub = np.asarray(block_, dtype=float)[idx]
            like = SimpleNamespace(left=int(frame.left), top=int(frame.top), block=sub, planes_z=ks * dz)
            origin0 = calibration.line_geometry(pilot)[0]
            B_floor = float(pilot.B_bar) if (math.isfinite(float(pilot.B_bar)) and float(pilot.B_bar) > 0) else None
            Bm, sd, frac, ok = model_background(sub, all_shadow(fscene, like, cfg, floor_B_gl=B_floor), frame,
                                                origin0, cfg)
            bco[:, 0] = Bm                       # B at the line's origin: background_on_line reads b0 there
            bco[:, 1:] = 0.0
            info["support_frac"] = frac
    elif cfg.cleaning.background_support == "unshaded":
        idx, bco, ok, frame, sd, frac = unshaded_planes(patch, ks, pilot, branch, scene, cfg)
        info["support_frac"] = frac
    else:
        cfg_bg = dataclasses.replace(cfg, calibration=dataclasses.replace(
            cfg.calibration, background_support=cfg.cleaning.background_support))
        idx, _far, bco, ok, frame, _stripe, sd = calibration.patch_planes(patch, ks, pilot, branch, scene, cfg_bg,
                                                                          with_sd=True)
    missing = [int(k) for k, j in zip(ks, idx) if j < 0]
    if missing:
        return refuse(REASON_MISSING, "planes %s not in the patch or not valid" % missing)
    no_bg = [int(k) for k, good, b in zip(ks, ok, bco) if not (good and math.isfinite(b[0]) and b[0] > 0)]
    if no_bg:
        return refuse(REASON_NO_BG, "planes %s without a valid background" % no_bg)
    origin, y_hat, e_u = calibration.line_geometry(pilot)
    B = np.array([float(calibration_scene.background_on_line(bco[q], float(pilot.cx_um), float(pilot.cy_um),
                                                              origin[None, :])[0]) for q in range(ks.size)])
    if not np.all(np.isfinite(B)) or np.any(B <= 0):
        return refuse(REASON_NO_BG, "background at the line's origin not finite and > 0: %r" % (B.tolist(),))
    block = np.asarray(patch[0])[np.asarray(idx)]
    v = profiles.profile_offsets(m)
    n_avg, step = profiles.n_along(m), float(m.profile_step_um)
    for q in range(ks.size):
        if not np.all(np.isfinite(profiles.sample_profile(block[q], frame, origin, y_hat, e_u, v, n_avg, step))):
            return refuse(REASON_LINE, "the measuring line leaves the patch in plane %d" % int(ks[q]))
    sd = np.asarray(sd, dtype=float)
    info["sd_gl"] = sd
    # the profile sample's noise SD (amendment 2, F2): the pixel's times the read's noise factor
    info["sigma_pixel_gl"] = float(np.median(sd)) if np.any(np.isfinite(sd)) else float("nan")
    info["noise_factor"] = profile_noise_factor(frame, block.shape[1:], origin, y_hat, e_u, v, n_avg, step)
    info["sigma_gl"] = info["sigma_pixel_gl"] * info["noise_factor"]
    node = NodeData(node_id=nid, own_ids=tuple(int(x) for x in own_ids),
                    node_xyz=trace,
                    block=block, B_gl=B, ks=ks, planes_z=ks * dz, left=int(frame.left), top=int(frame.top),
                    origin_xy=origin, y_hat=y_hat, e_u=e_u, v=v, n_avg=n_avg, step=step, theta0=th0)
    return node, info


# ------------------------------------------------------- forward models ---

def _line_args(node):
    return (node.ks, node.origin_xy, node.y_hat, node.e_u, node.v, int(node.n_avg), float(node.step))


def predict_factory(pilots, half_lengths, cfg, cache):
    """-> callable(NodeData) -> predict(d, mu, v0, dz) -> (n_planes, n_v) dimensionless shadow, the predictor
    of Block 32 (D-138.4): 1 - line_fast.line_profiles(local_tube(pilot, node.node_xyz, d, v0, dz, U_minus,
    U_plus), mu, node.ks, node's line, cfg, cache=cache); node.node_xyz is the traced position (v0, dz about
    it). pilots, half_lengths: dicts node_id -> pilot, (U_minus, U_plus) um along the model axis (build_node's
    info["U_axis_um"], amendment 3); cache: a
    line_fast.SectionCache shared by the calls (or None). An ExtrusionInvalid returns NaN (the fit then fails
    and Block 32 flags the node)."""
    def factory(node):
        nid = int(node.node_id)
        pilot = pilots[nid]
        Um, Up = (float(x) for x in half_lengths[nid])
        trace = np.asarray(node.node_xyz, dtype=float)
        args = _line_args(node)
        shape = (len(node.ks), len(node.v))

        def predict(d, mu, v0, dz):
            try:
                tau = line_fast.line_profiles(local_tube(pilot, trace, d, v0, dz, Um, Up), mu, *args, cfg,
                                              cache=cache)
            except line_fast.ExtrusionInvalid:
                return np.full(shape, np.nan)
            return 1.0 - tau
        return predict
    return factory


def forward_factory(pilot, half_length, node, cfg, cache):
    """-> forward(d, mu, v0, z_ax) -> (n_planes, n_v) dimensionless, Block 30's forward model:
    line_fast.line_profiles(local_tube(pilot, node.node_xyz, d, v0, z_ax - z_t, U_minus, U_plus), mu,
    node.ks, node's line, cfg, cache=cache): v0 about the trace, z_ax absolute (um) = z_t + dz; half_length =
    (U_minus, U_plus) um along the model axis (amendment 3). An ExtrusionInvalid returns NaN (Block 30 fails
    that start)."""
    Um, Up = (float(x) for x in half_length)
    trace = np.asarray(node.node_xyz, dtype=float)
    args = _line_args(node)
    shape = (len(node.ks), len(node.v))

    def forward(d, mu, v0, z_ax):
        try:
            return line_fast.line_profiles(local_tube(pilot, trace, d, v0, float(z_ax) - float(trace[2]), Um, Up),
                                           mu, *args, cfg, cache=cache)
        except line_fast.ExtrusionInvalid:
            return np.full(shape, np.nan)
    return forward


# ------------------------------------------------------------ work list ---

def _stretch_rows_by_id(swc, cfg):
    """node id -> (rows of its dendrite stretch, its index in them) over cell.stretches with the
    acquisition's dendrite types (the stretches of Blocks 8 and 27)."""
    out = {}
    ids = np.asarray(swc.ids)
    for run in cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types)):
        for t, r in enumerate(run):
            out[int(ids[r])] = (run, t)
    return out


def _node_position(fscene, nid):
    """(3,) um: node nid's position in the featured scene (the child end of its own segment, else a parent
    end that carries it)."""
    ids = np.asarray(fscene.seg_ids).astype(np.int64)
    xyz = np.asarray(fscene.seg_xyz, dtype=float)
    q = np.flatnonzero(ids[:, 0] == int(nid))
    if q.size:
        return xyz[int(q[0]), 3:].copy()
    q = np.flatnonzero(ids[:, 1] == int(nid))
    if q.size:
        return xyz[int(q[0]), :3].copy()
    raise ValueError("node %d is in no segment of the scene" % int(nid))


def work_list(example_ids, fscene, swc, cfg, peak_min_gl, B_gl):
    """-> list of node ids: the example nodes (in their order), then the end nodes of the example nodes'
    tier-1 neighbour segments whose predicted peak on the example's patch times B_gl is at least
    peak_min_gl (gl), by decreasing peak (ties by node id), the whole list capped at
    cleaning.work_max_nodes, the examples first (more examples than that: the examples cut; amendment 2, F5,
    the caller logs it). For each example: its own nodes within its tube's half-length on its stretch
    (own_ids_within, amendment 2, F1), its position in fscene, the patch of half-size cleaning.patch_half_um about it,
    the planes round(z / dz) +- cleaning.plane_half_range, the tiers (cleaning.neighbour_tiers) and the
    peaks (shadow.shadow_image's per_neighbour_peak, under partition_vertical for time, the tier-2
    neighbours left out). An end node joins only when it lies on a dendrite stretch and is not near_end
    (a near_end node is skipped by the fits, D-138.2). B_gl: scalar background (gl, > 0)."""
    B = float(B_gl)
    if not (math.isfinite(B) and B > 0):
        raise ValueError("B_gl must be finite and > 0")
    examples = [int(i) for i in example_ids]
    if len(set(examples)) != len(examples):
        raise ValueError("duplicate example ids")
    by_id = _stretch_rows_by_id(swc, cfg)
    xyz_rows = {}
    pv = dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, shadow_absorption="partition_vertical"))
    p, dz = float(cfg.acquisition.res0_um), float(cfg.acquisition.dz_um)
    P = int(cfg.cleaning.plane_half_range)
    seg_ids = np.asarray(fscene.seg_ids).astype(np.int64)
    ids = np.asarray(swc.ids)

    def candidate(nid):
        """nid lies on a dendrite stretch of >= 2 nodes and is not near_end there."""
        if nid not in by_id:
            return False
        run, t = by_id[nid]
        key = int(run[0])
        if key not in xyz_rows:
            pos = np.zeros((len(ids), 3))
            for r in run:
                pos[int(r)] = _node_position(fscene, int(ids[r]))
            xyz_rows[key] = own_branch(swc, run, pos)
        ob = xyz_rows[key]
        return ob is not None and not is_near_end(ob[0], t, cfg)

    cap = int(cfg.cleaning.work_max_nodes)
    if len(examples) >= cap:                 # the examples alone fill the work list (F5): no neighbour can join
        return examples[:cap]
    best = {}
    for nid in examples:
        if nid not in by_id:
            raise ValueError("example node %d is not on a dendrite stretch" % nid)
        run, t_ex = by_id[nid]
        pos = np.zeros((len(ids), 3))
        for r in run:
            pos[int(r)] = _node_position(fscene, int(ids[r]))
        ob = own_branch(swc, run, pos)
        if ob is None:
            raise ValueError("example node %d is on a one-node stretch" % nid)
        own = own_ids_within(ob[0], t_ex, cfg.inference.tube_half_length_um)[0]    # amendments 2 (F1), 3
        xyz = _node_position(fscene, nid)
        left, top, width, height = calibration_scene.patch_bounds(float(xyz[0]), float(xyz[1]),
                                                                  float(cfg.cleaning.patch_half_um), p)
        k0 = int(round(float(xyz[2]) / dz))
        planes_z = np.arange(k0 - P, k0 + P + 1) * dz
        tiers = cleaning.neighbour_tiers(fscene, own, xyz, cfg)
        res = shadow.shadow_image(fscene, own, left, top, width, height, planes_z, pv,
                                  B_gl=B if float(peak_min_gl) >= float(cfg.cleaning.shadow_floor_gl) else None,
                                  include=(tiers == 1))    # B: the floor's prefilter (D-146); peaks >= peak_min_gl kept
        segs, _somas = shadow.neighbour_indices(fscene, own)
        for m, q in enumerate(segs):
            if tiers[m] != 1:
                continue
            peak = B * float(res.per_neighbour_peak[m])
            if peak < float(peak_min_gl):
                continue
            for end in seg_ids[int(q)]:
                e = int(end)
                if e in examples or e in own:
                    continue
                if candidate(e):
                    best[e] = max(best.get(e, -np.inf), peak)
    cap = int(cfg.cleaning.work_max_nodes)
    extra = sorted(best, key=lambda e: (-best[e], e))[:max(0, cap - len(examples))]
    return (examples + extra)[:cap]
