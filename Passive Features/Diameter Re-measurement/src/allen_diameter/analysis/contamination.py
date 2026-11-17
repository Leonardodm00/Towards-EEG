"""The contamination survey of a node's measuring line -- Block 27 in specs/SPEC.md (plan step C0; D-129, D-130).

For node i of a Branch (Allen radius r_i, traced position p_i), the measuring
line runs through p_i along y_hat_i (Block 5's raw direction: the TLS line of
the trace's points within line_fit_window_um / 2 of path length, oriented
distally), sampled at Block 5's offsets v_n; the plane is k_i = round(z_i / dz),
z_k = k_i dz. Each traced segment q of the scene that is not on the node's
branch (not both ends' ids in branch.ids) is a neighbour; its predicted dip at
sample n is today's predictor (calibration_scene.shadow_depths, already in gl):

    B delta_q(v_n) = shadow_depths(rho_qn, max(r_q0, r_q1), mu, sigma_r(z_qn - z_k), B),

with c_qn = a_q + t_qn (b_q - a_q) the segment's point nearest the sample
o + v_n y_hat in projection (t_qn in [0, 1]), rho_qn its 2-D distance (um) and
z_qn its depth, linear in the segment's arc: one point sets both the distance
and the blur (amendment 1, D-131). Then

    P_i(v_n) = sum_q B delta_q(v_n)  (gl),   D_i = shadow_depth(0, r_i, mu, sigma_r(0), B)  (gl),
    C_i = (sum_n w_n P_i(v_n) / sum_n w_n) / D_i,   w_n = exp(-|v_n| / lambda),  lambda = sgt_lambda_um.

Classes, first that holds (amendment 1, D-131): clean (C_i < clean_index_max);
overlap (a neighbour sharing no node with the own stretch whose footprint --
within its larger end radius of its axis -- meets the own branch's footprint at
a point of the line's square |u|, |v| <= profile_half_um, read on the grid
v_n x v_n); close_in_focus (one neighbour whose point nearest the line's centre
lies within close_lateral_um of it laterally and close_defocus_um of the plane
carries >= 1/2 of sum_n w_n P); diffuse.

The soma is not in P_i (the Scene carries no soma depth); soma_lateral_um is
the node's 2-D distance to the nearest soma disc's edge, 0 inside a disc, NaN
without a soma. The own footprint is the branch's consecutive nodes, each
segment at its larger end radius; n_neighbours counts the neighbours whose dip
on the line reaches cleaning.shadow_floor_gl.

Amendment 2 (D-150.2): tier1_contamination ranks a node by its worst tier-1
neighbour -- the neighbours the cleaning subtracts (D-132) -- on the same line
and with the same predictor, by the structured part of each segment's dip,
R_q = max_n B delta_q(v_n) - min_n B delta_q(v_n) (gl): a dip that shapes the
profile, not a uniform rise that the background takes. pick_tier1_examples
draws the example nodes from that ranking.

No file access, no plotting. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from . import background, path
from .calibration_scene import OWN, _GL_CACHE, leggauss, patch_frame, segment_rows, shadow_depth, shadow_depths
from .profiles import profile_offsets
from ..model.kernel import sigma_r

CLASSES = ("overlap", "close_in_focus", "diffuse", "clean")   # the labels; precedence: clean first, then overlap, close_in_focus, diffuse (amendment 1, D-131)
NEAREST_SOMA = "nearest_soma"
EXAMPLE_KEYS = ("clean", "diffuse", "close_in_focus", "overlap", NEAREST_SOMA)  # CleaningConfig.example_counts' order


def _check(branch, index, B_gl, mu_per_um):
    n = len(branch.ids)
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= int(index) < n:
        raise ValueError("index %r is outside the branch (%d nodes)" % (index, n))
    if not (math.isfinite(float(B_gl)) and float(B_gl) > 0):
        raise ValueError("B_gl must be finite and > 0")
    if not (math.isfinite(float(mu_per_um)) and float(mu_per_um) > 0):
        raise ValueError("mu_per_um must be finite and > 0")
    r = float(np.asarray(branch.radius_um, dtype=float)[int(index)])
    if not (math.isfinite(r) and r > 0):
        raise ValueError("the node's Allen radius must be finite and > 0 (the own dip D_i divides the index)")
    return int(index)


def _line(branch, i, cfg):
    """(origin (2,), y_hat (2,), e_u (2,), z_k): the trace's line through node i and the plane nearest its depth."""
    xyz = np.asarray(branch.xyz_um, dtype=float)
    idx = path.window(branch.s_um, i, 0.5 * cfg.measure.line_fit_window_um)
    if idx.size < 2:
        idx = np.arange(max(i - 1, 0), min(i + 2, xyz.shape[0]))
    t = path.orient(path.tls_direction(xyz[idx]), xyz[idx[0]], xyz[idx[-1]])
    _theta, _phi, y_hat, e_u = path.angles(t)
    dz = float(cfg.acquisition.dz_um)
    return xyz[i, :2].copy(), y_hat, e_u, round(xyz[i, 2] / dz) * dz


def _neighbours(branch, scene, o, cfg):
    """Indices (n_q,) of the scene's segments off the branch (segment_rows' own rule: both ends' ids in
    branch.ids), every one of them whatever its distance or depth (reach infinite, D-130.O5)."""
    shape, frame = patch_frame(o[0], o[1], cfg.measure.profile_half_um, cfg.acquisition.res0_um)
    rows, idx = segment_rows(scene, frame, shape, branch, np.inf, return_index=True)
    return np.asarray(idx, dtype=np.int64)[rows[:, 8] != OWN] if rows.shape[0] else np.empty(0, dtype=np.int64)


def _nearest(px, py, a, b):
    """(distance (um), t in [0, 1]) of the points (px, py) to the segment a-b in projection: the closed form of
    background.segment_distance, returning the nearest point's parameter as well (its depth is then linear in t)."""
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    L2 = dx * dx + dy * dy
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    t = np.zeros(px.shape) if L2 == 0.0 else np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


_UNDERFLOW_SIGMAS = 40.0   # exp(-40^2 / 2) underflows to 0 in float64: such samples are exactly 0, not truncated


def _segment_dips(rho, sig, r_q, mu, B):
    """shadow_depths at each sample with its own kernel width, one call per distinct width."""
    out = np.zeros(rho.shape)
    live = rho - r_q <= _UNDERFLOW_SIGMAS * np.maximum(sig, 1e-6)
    for s_val in np.unique(sig[live]):
        sel = live & (sig == s_val)
        out[sel] = shadow_depths(rho[sel], r_q, mu, float(s_val), B)
    return out


def _profile(branch, i, scene, v, B_gl, mu_per_um, cfg):
    o, y_hat, e_u, z_k = _line(branch, i, cfg)
    v = np.asarray(v, dtype=float).reshape(-1)
    qs = _neighbours(branch, scene, o, cfg)
    px, py = o[0] + v * y_hat[0], o[1] + v * y_hat[1]
    per = np.zeros((qs.size, v.size))
    lateral = np.zeros(qs.size)
    defocus = np.zeros(qs.size)
    for m, q in enumerate(qs):
        a, b = scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:]
        d0, t0 = _nearest(o[0], o[1], a, b)
        lateral[m], defocus[m] = float(d0), abs(float(a[2] + t0 * (b[2] - a[2])) - z_k)
        rho, t = _nearest(px, py, a, b)
        sig = np.asarray(sigma_r(a[2] + t * (b[2] - a[2]) - z_k, cfg.renderer), dtype=float)
        per[m] = _segment_dips(rho, sig, float(np.max(scene.seg_radius[q])), float(mu_per_um), float(B_gl))
    return per.sum(axis=0), per, qs, lateral, defocus, (o, y_hat, e_u)


def pollution_profile(branch, index, scene, v, B_gl, mu_per_um, cfg):
    """-> (P (n_v,) gl, per_neighbour (n_q, n_v) gl), n_q the scene's segments off the branch, in the scene's order."""
    i = _check(branch, index, B_gl, mu_per_um)
    P, per = _profile(branch, i, scene, v, B_gl, mu_per_um, cfg)[:2]
    return P, per


def _overlap(branch, scene, qs, frame_line, cfg):
    """Per neighbour (n_q,) bool: its footprint meets the own branch's on the line's square (grid v_n x v_n)."""
    o, y_hat, e_u = frame_line
    g = profile_offsets(cfg.measure)
    V, U = np.meshgrid(g, g, indexing="ij")
    px, py = o[0] + V * y_hat[0] + U * e_u[0], o[1] + V * y_hat[1] + U * e_u[1]
    xyz, r = np.asarray(branch.xyz_um, dtype=float), np.asarray(branch.radius_um, dtype=float)
    own = np.zeros(V.shape, dtype=bool)
    for j in range(xyz.shape[0] - 1):
        own |= background.segment_distance(px, py, xyz[j], xyz[j + 1]) <= max(r[j], r[j + 1])
    out = np.zeros(qs.size, dtype=bool)
    own_ids = {int(x) for x in np.asarray(branch.ids).ravel()}
    if own.any():
        for m, q in enumerate(qs):
            if int(scene.seg_ids[q, 0]) in own_ids or int(scene.seg_ids[q, 1]) in own_ids:
                continue                                   # attached at a branch point or root: no overlap (D-131)
            d = background.segment_distance(px[own], py[own], scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:])
            out[m] = bool(np.any(d <= float(np.max(scene.seg_radius[q]))))
    return out


def node_contamination(branch, index, scene, B_gl, mu_per_um, cfg):
    """-> dict: node_id, own_dip_gl, index, class, n_neighbours, worst_child_id, worst_lateral_um, worst_defocus_um,
    worst_share, soma_lateral_um. n_neighbours: neighbours whose predicted dip on the line reaches cleaning.shadow_floor_gl;
    worst: the neighbour of the largest weighted share of sum_n w_n P (-1 / NaN when none carries any), its
    lateral distance and unsigned defocus taken at its point nearest the line's centre."""
    i = _check(branch, index, B_gl, mu_per_um)
    m_cfg, c_cfg = cfg.measure, cfg.cleaning
    v = profile_offsets(m_cfg)
    P, per, qs, lateral, defocus, frame_line = _profile(branch, i, scene, v, B_gl, mu_per_um, cfg)
    w = np.exp(-np.abs(v) / float(m_cfg.sgt_lambda_um))
    D = shadow_depth(0.0, float(branch.radius_um[i]), float(mu_per_um), float(sigma_r(0.0, cfg.renderer)), float(B_gl))
    total = float(np.sum(w * P))
    C = (total / float(np.sum(w))) / D
    contrib = per @ w
    if qs.size and total > 0:
        m = int(np.argmax(contrib))
        share = contrib / total
        worst = (int(scene.seg_ids[qs[m], 0]), float(lateral[m]), float(defocus[m]), float(share[m]))
    else:
        share = np.zeros(qs.size)
        worst = (-1, float("nan"), float("nan"), float("nan"))
    close = (lateral <= c_cfg.close_lateral_um) & (defocus <= c_cfg.close_defocus_um) & (share >= 0.5)
    if C < c_cfg.clean_index_max:
        cls = "clean"
    elif qs.size and np.any(_overlap(branch, scene, qs, frame_line, cfg)):
        cls = "overlap"
    elif np.any(close):
        cls = "close_in_focus"
    else:
        cls = "diffuse"
    n_nb = int(np.sum(np.max(per, axis=1) >= c_cfg.shadow_floor_gl)) if qs.size else 0
    return dict(node_id=int(branch.ids[i]), own_dip_gl=float(D), index=float(C), **{"class": cls},
                n_neighbours=n_nb, worst_child_id=worst[0], worst_lateral_um=worst[1],
                worst_defocus_um=worst[2], worst_share=worst[3], soma_lateral_um=_soma_lateral(branch, i, scene))


def _soma_lateral(branch, i, scene):
    """2-D distance (um) from node i to the nearest soma disc's edge (0 inside a disc); NaN without a soma."""
    soma = np.asarray(scene.soma, dtype=float).reshape(-1, 3)
    if soma.shape[0] == 0:
        return float("nan")
    x, y = float(branch.xyz_um[i, 0]), float(branch.xyz_um[i, 1])
    return float(np.max([0.0, np.min(np.hypot(soma[:, 0] - x, soma[:, 1] - y) - soma[:, 2])]))


def pick_examples(rows, counts, rng):
    """-> list of node ids (D-130.O7). counts: a dict {key: n} or a sequence in EXAMPLE_KEYS' order (the order of
    CleaningConfig.example_counts); a class key draws n of its rows at random (rng, without replacement),
    NEAREST_SOMA takes the n rows of smallest 'soma_lateral_um' (node_contamination's key). No node twice; a short
    key is reported by a RuntimeWarning and in the warning's text (wanted, got)."""
    items = list(counts.items()) if isinstance(counts, dict) else list(zip(EXAMPLE_KEYS, counts))
    for key, n in items:
        if key not in CLASSES and key != NEAREST_SOMA:
            raise ValueError("unknown example key %r" % (key,))
        if int(n) < 0:
            raise ValueError("counts must be >= 0")
    picked, short = [], []
    rows = list(rows)
    for key, n in items:
        n = int(n)
        free = [r for r in rows if int(r["node_id"]) not in picked]
        if key == NEAREST_SOMA:
            free = sorted((r for r in free if np.isfinite(r.get("soma_lateral_um", np.nan))),
                          key=lambda r: (float(r["soma_lateral_um"]), int(r["node_id"])))
            take = [int(r["node_id"]) for r in free[:n]]
        else:
            ids = sorted({int(r["node_id"]) for r in free if r["class"] == key})
            take = [ids[j] for j in rng.permutation(len(ids))[:n]] if ids else []
        picked.extend(take)
        if len(take) < n:
            short.append("%s: wanted %d, got %d" % (key, n, len(take)))
    if short:
        warnings.warn("pick_examples: short -- " + "; ".join(short), RuntimeWarning, stacklevel=2)
    return picked


# --------------------------------------------- tier-1 ranking (amendment 2) --

_BRANCH_AND_BOUND = True   # tier1_contamination stops evaluating tier-1 neighbours whose bound is under the best found
TIER1_COLUMNS = ("node_id", "x_um", "y_um", "z_um", "own_dip_gl", "n_tier2", "n_evaluated", "worst1_range_gl",
                 "worst1_peak_gl", "worst1_child_id", "worst1_lateral_um", "worst1_defocus_um", "worst2_peak_gl",
                 "soma_lateral_um")


def _gl_n(r, sigma):
    """The Gauss-Legendre order shadow_depths uses for a tube of radius r and blur sigma (_gauss_legendre's rule)."""
    return np.minimum(512, np.maximum(16, np.ceil(6.0 * r / np.maximum(sigma, 1e-6)))).astype(np.int64)


def _dips_segments(rho, sigma, r, mu, B):
    """shadow_depths at every (segment, sample) pair: rho, sigma (m, n_v) um, r, mu (m,) um, 1/um -> (m, n_v) gl,
    each pair on the Gauss-Legendre rule shadow_depths would take (_gauss_legendre). Custom: shadow_depths'
    formula with the tube's absorbed fraction at the rule's nodes computed once per segment and rule, not per
    sample (shadow_depths and _depths_rule take one tube or recompute it per element)."""
    m, n_v = rho.shape
    out = np.zeros((m, n_v))
    n_rule = _gl_n(r[:, None], sigma)
    for nn in np.unique(n_rule):
        nn = int(nn)
        if nn not in _GL_CACHE:
            _GL_CACHE[nn] = leggauss(nn)
        x, w = _GL_CACHE[nn]
        qi, vi = np.nonzero(n_rule == nn)
        segs, inv = np.unique(qi, return_inverse=True)
        vp_s = r[segs, None] * x[None, :]
        deficit = 1.0 - np.exp(-2.0 * mu[segs, None] * np.sqrt(np.maximum(r[segs, None] ** 2 - vp_s ** 2, 0.0)))
        s = np.maximum(sigma[qi, vi], 1e-6)[:, None]
        g = np.exp(-0.5 * ((rho[qi, vi][:, None] - vp_s[inv]) / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
        out[qi, vi] = B * r[qi] * np.sum(w * deficit[inv] * g, axis=1)
    return out


def tier1_contamination(fscene, branch, index, own_ids, B_gl, cfg, tiers=None):
    """-> dict (TIER1_COLUMNS): node i's worst tier-1 neighbour on its measuring line (amendment 2, D-150.2).

    fscene: Block 23 FeaturedScene (um; each segment's d and mu at its two ends); branch: the node's own stretch
    (a Branch whose xyz_um are the positions in fscene's frame); index: the node's index in it; own_ids: the
    node's own nodes (node_inputs.own_ids_within: its tube); B_gl: background (gl, > 0); tiers: the tiers of the
    neighbours (cleaning.neighbour_tiers' output for this node), None = computed here at branch.xyz_um[index].
    The line and the predictor are the survey's (module docstring), each segment q at its larger end radius
    r_q and larger end coefficient mu_q; the neighbours are the segments with an end outside own_ids, in the
    scene's order (cleaning.neighbour_tiers' order). Per neighbour, the dip on the line B delta_q(v_n), its
    peak P_q = max_n and its structured part R_q = max_n - min_n (gl). Bounds (exact, they need sigma_r
    non-decreasing in |defocus|, else every neighbour is evaluated): A_q = 2 r_q (1 - exp(-2 mu_q r_q)) bounds
    the integral of the tube's absorbed fraction; with sigma_min, sigma_max the kernel at the segment's smallest
    and largest defocus, P_q <= B min(1 - exp(-2 mu_q r_q), A_q / (sqrt(2 pi) sigma_min)), and over the line's
    length ell R_q <= B A_q ell (exp(-1/2) + gamma tan(phi_q)) / (sqrt(2 pi) sigma_min^2) (gamma sigma_r's
    largest slope, phi_q the segment's tilt); a segment whose distance to every sample, less r_q, is at least
    6 sigma_max is left out. A tier-2 neighbour is evaluated when its peak bound reaches
    cleaning.rank_bound_gl; a tier-1 one when its bound min(peak, range) does, by decreasing bound, until the
    bound falls under the largest R_q found (branch and bound: the worst is exact).
    worst1: the tier-1 neighbour of the largest R_q (worst1_child_id its child id; lateral distance and unsigned
    defocus at its point nearest the line's centre, as the survey's worst); worst2_peak_gl: the largest P_q of
    the tier-2 neighbours evaluated (0 without one); n_evaluated: the neighbours evaluated. No tier-1 neighbour
    evaluated: worst1_range_gl = worst1_peak_gl = 0, worst1_child_id = -1, the distances NaN."""
    from . import cleaning                                   # the tier rule (D-132), Block 28's module
    i = _check(branch, index, B_gl, 1.0)
    B = float(B_gl)
    xyz_all = np.asarray(fscene.seg_xyz, dtype=float)
    ids = np.asarray(fscene.seg_ids, dtype=np.int64)
    own = np.asarray(sorted({int(x) for x in own_ids}), dtype=np.int64)
    nb = np.flatnonzero(~(np.isin(ids[:, 0], own) & np.isin(ids[:, 1], own)))
    node = np.asarray(branch.xyz_um, dtype=float)[i]
    if tiers is None:
        tiers = cleaning.neighbour_tiers(fscene, own, node, cfg)
    tiers = np.asarray(tiers)[:nb.size]
    o, y_hat, _e_u, z_k = _line(branch, i, cfg)
    v = profile_offsets(cfg.measure)
    r_own = float(np.asarray(branch.radius_um, dtype=float)[i])
    mu_own = float(np.max(np.asarray(fscene.seg_mu_per_um, dtype=float)[np.isin(ids[:, 0], own)
                                                                         & np.isin(ids[:, 1], own)], initial=0.0))
    row = dict(node_id=int(branch.ids[i]), x_um=float(node[0]), y_um=float(node[1]), z_um=float(node[2]),
               own_dip_gl=float(shadow_depth(0.0, r_own, mu_own, float(sigma_r(0.0, cfg.renderer)), B))
               if mu_own > 0 else float("nan"),
               n_tier2=int(np.sum(tiers == 2)), n_evaluated=0, worst1_range_gl=0.0, worst1_peak_gl=0.0,
               worst1_child_id=-1, worst1_lateral_um=float("nan"), worst1_defocus_um=float("nan"),
               worst2_peak_gl=0.0, soma_lateral_um=_soma_lateral(branch, i, fscene))
    if nb.size == 0:
        return row
    a, b = xyz_all[nb, :3], xyz_all[nb, 3:]
    r = 0.5 * np.max(np.asarray(fscene.seg_d_um, dtype=float)[nb], axis=1)
    mu = np.max(np.asarray(fscene.seg_mu_per_um, dtype=float)[nb], axis=1)
    # the samples' nearest points on every neighbour (n_nb, n_v), as _nearest
    px, py = o[0] + v * y_hat[0], o[1] + v * y_hat[1]
    dx, dy = (b[:, 0] - a[:, 0])[:, None], (b[:, 1] - a[:, 1])[:, None]
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(L2 == 0.0, 0.0, np.clip(((px[None, :] - a[:, :1]) * dx + (py[None, :] - a[:, 1:2]) * dy) / L2,
                                             0.0, 1.0))
    rho = np.hypot(px[None, :] - (a[:, :1] + t * dx), py[None, :] - (a[:, 1:2] + t * dy))
    # the bounds
    za, zb = a[:, 2] - z_k, b[:, 2] - z_k
    d_min = np.where(za * zb <= 0.0, 0.0, np.minimum(np.abs(za), np.abs(zb)))
    d_max = np.maximum(np.abs(za), np.abs(zb))
    s_min = np.asarray(sigma_r(d_min, cfg.renderer), dtype=float)
    s_max = np.asarray(sigma_r(d_max, cfg.renderer), dtype=float)
    absorbed = 1.0 - np.exp(-2.0 * mu * r)
    A = 2.0 * r * absorbed                                   # bounds the integral of the absorbed fraction (um)
    s2 = np.maximum(s_min, 1e-6)
    peak_bound = B * np.minimum(absorbed, A / (math.sqrt(2.0 * math.pi) * s2))
    # the structured part over the line's length ell: |d phi_s(x)/dx| <= exp(-1/2) / (sqrt(2 pi) s^2),
    # |d phi_s(x)/ds| <= 1 / (sqrt(2 pi) s^2), rho is 1-Lipschitz in v, and sigma along the line moves by at most
    # gamma tan(phi_q) per um (gamma: sigma_r's largest slope; the nearest point's depth along a segment of tilt phi_q)
    rc = cfg.renderer
    knots, vals = (np.asarray(x, dtype=float) for x in (rc.kernel_table_delta_um, rc.kernel_table_sigma_um))
    slope_c = {"linear": float(rc.kernel_continuation_slope), "frozen": 0.0,
               "proportional": float(vals[-1] / knots[-1])}.get(rc.kernel_continuation, np.inf)
    gamma = max(float(np.max(np.diff(vals) / np.diff(knots))), slope_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        tan = np.where(L2[:, 0] == 0.0, 0.0, np.abs(zb - za) / np.sqrt(L2[:, 0]))
    ell = float(v[-1] - v[0])
    range_bound = B * A * ell * (math.exp(-0.5) + gamma * tan) / (math.sqrt(2.0 * math.pi) * s2 ** 2)
    far = rho.min(axis=1) - r >= 6.0 * s_max
    # the bounds need sigma_r non-decreasing in |defocus| (between its knots and beyond them)
    monotone = bool(np.all(np.diff(vals) >= 0.0)) and slope_c >= 0.0
    bound = float(cfg.cleaning.rank_bound_gl)
    r_bound = np.minimum(peak_bound, range_bound)
    if monotone:
        two = np.flatnonzero(~far & (tiers == 2) & (peak_bound >= bound))
        one = np.flatnonzero(~far & (tiers == 1) & (r_bound >= bound))
    else:
        two, one = np.flatnonzero(tiers == 2), np.flatnonzero(tiers == 1)

    def exact(idx):
        if idx.size == 0:
            return np.empty(0), np.empty(0)
        zq = a[idx, 2:3] + t[idx] * (b[idx, 2:3] - a[idx, 2:3]) - z_k
        dips = _dips_segments(rho[idx], np.asarray(sigma_r(zq, cfg.renderer), dtype=float), r[idx], mu[idx], B)
        return dips.max(axis=1), dips.max(axis=1) - dips.min(axis=1)

    P2, _R2 = exact(two)
    if P2.size:
        row["worst2_peak_gl"] = float(P2.max())
    # the tier-1 neighbours by decreasing bound, in chunks; one whose bound is under the largest R found so far
    # cannot be the worst, so the evaluation stops there (exact; switched off with _BRANCH_AND_BOUND = False)
    if monotone and _BRANCH_AND_BOUND:
        one = one[np.argsort(-r_bound[one], kind="stable")]
    best_R, best_m, done, lo, size = -1.0, -1, 0, 0, 16
    while lo < one.size:
        idx = one[lo:lo + size]
        lo, size = lo + size, min(2 * size, 1024)            # chunks of 16, 32, ... up to 1024
        if monotone and _BRANCH_AND_BOUND and best_R >= 0.0:
            idx = idx[r_bound[idx] >= best_R]
            if idx.size == 0:
                break
        P1, R1 = exact(idx)
        done += idx.size
        k = int(np.argmax(R1))
        if R1[k] > best_R:
            best_R, best_m, best_P = float(R1[k]), int(idx[k]), float(P1[k])
    row["n_evaluated"] = int(two.size + done)
    if best_m >= 0:
        q = nb[best_m]
        d0, t0 = _nearest(o[0], o[1], xyz_all[q, :3], xyz_all[q, 3:])
        row.update(worst1_range_gl=best_R, worst1_peak_gl=best_P, worst1_child_id=int(ids[q, 0]),
                   worst1_lateral_um=float(d0),
                   worst1_defocus_um=abs(float(xyz_all[q, 2] + float(t0) * (xyz_all[q, 5] - xyz_all[q, 2])) - z_k))
    return row


def pick_tier1_examples(rows, n, max_tier2_gl, min_separation_um):
    """-> list of node ids (amendment 2, D-150.2): from the rows (tier1_contamination's dicts with a bool
    'near_end'), the not-near_end ones whose tier-2 neighbours put at most max_tier2_gl on the line, by
    decreasing worst1_range_gl (ties by node id), skipping a node within min_separation_um (3-D) of one already
    taken, until n; rows with worst1_range_gl = 0 are never taken."""
    if int(n) < 0 or not (float(min_separation_um) >= 0.0) or not math.isfinite(float(max_tier2_gl)):
        raise ValueError("n >= 0, min_separation_um >= 0 and a finite max_tier2_gl are required")
    ok = [r for r in rows if not bool(r["near_end"]) and float(r["worst2_peak_gl"]) <= float(max_tier2_gl)
          and float(r["worst1_range_gl"]) > 0.0]
    ok.sort(key=lambda r: (-float(r["worst1_range_gl"]), int(r["node_id"])))
    taken, pos = [], []
    for r in ok:
        if len(taken) >= int(n):
            break
        p = np.array([float(r["x_um"]), float(r["y_um"]), float(r["z_um"])])
        if any(float(np.linalg.norm(p - q)) < float(min_separation_um) for q in pos):
            continue
        taken.append(int(r["node_id"]))
        pos.append(p)
    return taken

