"""Node inputs of the pilot run of the cleaning and the per-node fit -- Block 36 in specs/SPEC.md
(plan steps C0, C5-C6, F5; D-129, D-130, D-132, D-136, D-137, D-138).

The functions the two entry points (scripts/clean_cell.py, scripts/infer_nodes.py) need to turn a
traced node and its pilot (Block 5's measure_pilot) into the inputs of Blocks 32 (the cleaning
iteration) and 30 (the per-node fit):

  own_branch        the stretch's own nodes as a Branch, without the parent node that
                    cell.stretch_branch prepends (D-136.1)
  tube_half_length  U = min(arc length to the stretch's first node, to its last node,
                    inference.tube_half_length_um); the node is near_end when U is below
                    inference.extrusion_cap_margin_um (D-138.2, D-140.3)
  local_tube        the node's straight capped tube, shifts relative to the trace (amendment 1,
                    D-141): centre at the foot of the traced position on the measuring line plus
                    v0 y_hat, i.e. (c_x, c_y) + (v0 - v_t) y_hat with v_t = ((c_x, c_y) - (x_t, y_t))
                    . y_hat, depth z_t + dz; radius d / 2, heading pilot.theta_rad, tilt
                    pilot.phi_rad, aspect 1, caps "axial" at +-U; y_hat = (-sin theta, cos theta)
  pilot_start       theta0 = (d_hat, mu_hat, v_t, z_sub - z_t), the first-stage start
  node_planes       k* - P .. k* + P, P = cleaning.plane_half_range
  build_node        the NodeData of Block 32 from one patch fetch (calibration_scene.fetch_patch),
                    each plane's planar background and noise SD (calibration.patch_planes,
                    with_sd=True, the support cleaning.background_support), B_k the background at
                    the line's origin (calibration_scene.background_on_line), the measuring line of
                    Block 5's pilot, node_xyz the traced position
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


def tube_half_length(branch, index, cfg):
    """U (um) = min(arc length from node `index` to the branch's first node, to its last node,
    cfg.inference.tube_half_length_um). The branch is the node's own stretch (own_branch)."""
    s = np.asarray(branch.s_um, dtype=float)
    i = int(index)
    if not 0 <= i < s.size:
        raise ValueError("index %d outside the branch of %d nodes" % (i, s.size))
    return float(min(s[i] - s[0], s[-1] - s[i], float(cfg.inference.tube_half_length_um)))


def is_near_end(branch, index, cfg):
    """True when tube_half_length is below cfg.inference.extrusion_cap_margin_um: the node is flagged
    near_end and skipped by the fits (D-138.2, D-140.3)."""
    return bool(tube_half_length(branch, index, cfg) < float(cfg.inference.extrusion_cap_margin_um))


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


def local_tube(pilot, node_xyz_trace, d_um, v0_um, dz_um, half_length_um):
    """The node's straight tube (Block 2 Tube), shifts about the trace (amendment 1, D-141): centre
    (cx, cy) + (v0 - v_t) y_hat -- the foot of the traced position on the measuring line plus v0 y_hat --
    at depth z_t + dz_um; radius d_um / 2, heading pilot.theta_rad, tilt pilot.phi_rad, aspect 1,
    half-length half_length_um, axial caps; y_hat = (-sin theta, cos theta). pilot: any object with cx_um,
    cy_um, theta_rad, phi_rad; node_xyz_trace (3,) um, the traced position in the image frame."""
    th = float(pilot.theta_rad)
    yx, yy = -math.sin(th), math.cos(th)
    t = np.asarray(node_xyz_trace, dtype=float)
    s = float(v0_um) - trace_offset(pilot, t)
    c = (float(pilot.cx_um) + s * yx, float(pilot.cy_um) + s * yy, float(t[2]) + float(dz_um))
    return geometry.Tube(c=c, r=0.5 * float(d_um), phi=float(pilot.phi_rad), theta=th, aspect=1.0,
                         half_length=float(half_length_um), end_cut="axial")


def node_planes(pilot, cfg):
    """The plane indices k* - P .. k* + P (P = cfg.cleaning.plane_half_range), k* = pilot.k_star."""
    P = int(cfg.cleaning.plane_half_range)
    k = int(pilot.k_star)
    return np.arange(k - P, k + P + 1, dtype=np.int64)


def _has_sharpest_plane(pilot):
    return all(math.isfinite(float(getattr(pilot, name))) for name in ("z_sub_um", "cx_um", "cy_um"))


def own_ids_within(branch, index, half_length_um):
    """-> tuple of node ids: the stretch's own nodes whose arc length from node `index` along the stretch is
    at most half_length_um (the tube's U), and at least the node and its neighbours on the stretch
    (amendment 2, F1). The stretch's segments beyond are neighbours for Blocks 25, 28 and 32."""
    s = np.asarray(branch.s_um, dtype=float)
    ids = np.asarray(branch.ids)
    i = int(index)
    if not 0 <= i < s.size:
        raise ValueError("index %d outside the branch of %d nodes" % (i, s.size))
    keep = np.abs(s - s[i]) <= float(half_length_um) * (1.0 + 1e-12) + 1e-12
    keep[max(0, i - 1):min(s.size, i + 2)] = True
    return tuple(int(x) for x in ids[keep])


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

def build_node(branch, index, pilot, provider, scene, cfg, own_ids, theta0=None):
    """-> (NodeData or None, info). branch: the node's own stretch (own_branch); index: the node's index
    in it; pilot: the node's Block 5 pilot (NodeResult: cx_um, cy_um, theta_rad, phi_rad, z_sub_um,
    k_star, d_hat_um, mu_hat_per_um, B_bar); provider(left, top, width, height, k_lo, k_hi) -> (block,
    ks, valid, frame); scene: the cell's plane_iteration.Scene; cfg: DiameterConfig; own_ids: the
    stretch's node ids (D-136.1); theta0: (d, mu, v0, dz) about the trace, default pilot_start.

    One provider call: the patch of half-size cleaning.patch_half_um about the pilot's centre over
    node_planes; per plane its planar background and noise SD (calibration.patch_planes, with_sd);
    B_k the background at the line's origin; the support is cfg.cleaning.background_support (amendment 1,
    D-141: patch_planes is called with calibration.background_support replaced by it). The line:
    calibration.line_geometry (origin the pilot's centre), v = profiles.profile_offsets, n_avg =
    profiles.n_along, step = profile_step_um (Block 5's pilot). node_xyz = the traced position,
    branch.xyz_um[index].

    info: node_id, half_length_um, near_end, ks, and on success sd_gl (the planes' pixel noise SDs, gl),
    sigma_pixel_gl (their median), noise_factor (profile_noise_factor of the line) and sigma_gl = their
    product, the profile sample's noise SD (amendment 2, F2); on a refusal reason (one of REFUSALS) and detail. Refusals:
    near_end (checked first, no fetch), no sharpest plane, no usable pilot fit for the default start,
    a plane missing or invalid, a plane with no valid background, a line that leaves the patch."""
    m, dz = cfg.measure, float(cfg.acquisition.dz_um)
    i = int(index)
    nid = int(np.asarray(branch.ids)[i])
    U = tube_half_length(branch, i, cfg)
    info = {"node_id": nid, "half_length_um": U, "near_end": U < float(cfg.inference.extrusion_cap_margin_um)}

    def refuse(reason, detail=""):
        info.update(reason=reason, detail=detail)
        return None, info

    if info["near_end"]:
        return refuse(REASON_NEAR_END, "tube half-length %.3f um < margin %.3f um"
                      % (U, float(cfg.inference.extrusion_cap_margin_um)))
    if not _has_sharpest_plane(pilot):
        return refuse(REASON_NO_PLANE, "the pilot has no sharpest plane (z_sub_um not finite)")
    trace = np.asarray(branch.xyz_um, dtype=float)[i]
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
    of Block 32 (D-138.4): 1 - line_fast.line_profiles(local_tube(pilot, node.node_xyz, d, v0, dz, U), mu,
    node.ks, node's line, cfg, cache=cache); node.node_xyz is the traced position (v0, dz about it). pilots, half_lengths: dicts node_id -> pilot, U (um); cache: a
    line_fast.SectionCache shared by the calls (or None). An ExtrusionInvalid returns NaN (the fit then fails
    and Block 32 flags the node)."""
    def factory(node):
        nid = int(node.node_id)
        pilot, U = pilots[nid], float(half_lengths[nid])
        trace = np.asarray(node.node_xyz, dtype=float)
        args = _line_args(node)
        shape = (len(node.ks), len(node.v))

        def predict(d, mu, v0, dz):
            try:
                tau = line_fast.line_profiles(local_tube(pilot, trace, d, v0, dz, U), mu, *args, cfg, cache=cache)
            except line_fast.ExtrusionInvalid:
                return np.full(shape, np.nan)
            return 1.0 - tau
        return predict
    return factory


def forward_factory(pilot, half_length, node, cfg, cache):
    """-> forward(d, mu, v0, z_ax) -> (n_planes, n_v) dimensionless, Block 30's forward model:
    line_fast.line_profiles(local_tube(pilot, node.node_xyz, d, v0, z_ax - z_t, U), mu, node.ks, node's
    line, cfg, cache=cache): v0 about the trace, z_ax absolute (um) = z_t + dz. An ExtrusionInvalid returns
    NaN (Block 30 fails that start)."""
    U = float(half_length)
    trace = np.asarray(node.node_xyz, dtype=float)
    args = _line_args(node)
    shape = (len(node.ks), len(node.v))

    def forward(d, mu, v0, z_ax):
        try:
            return line_fast.line_profiles(local_tube(pilot, trace, d, v0, float(z_ax) - float(trace[2]), U), mu,
                                           *args, cfg, cache=cache)
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
        own = own_ids_within(ob[0], t_ex, tube_half_length(ob[0], t_ex, cfg))      # amendment 2, F1
        xyz = _node_position(fscene, nid)
        left, top, width, height = calibration_scene.patch_bounds(float(xyz[0]), float(xyz[1]),
                                                                  float(cfg.cleaning.patch_half_um), p)
        k0 = int(round(float(xyz[2]) / dz))
        planes_z = np.arange(k0 - P, k0 + P + 1) * dz
        tiers = cleaning.neighbour_tiers(fscene, own, xyz, cfg)
        res = shadow.shadow_image(fscene, own, left, top, width, height, planes_z, pv, B_gl=None,
                                  include=(tiers == 1))
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
