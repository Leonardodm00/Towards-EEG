"""The contamination survey of a node's measuring line -- Block 27 in specs/SPEC.md (plan step C0; D-129, D-130).

For node i of a Branch (Allen radius r_i, traced position p_i), the measuring
line runs through p_i along y_hat_i (Block 5's raw direction: the TLS line of
the trace's points within line_fit_window_um / 2 of path length, oriented
distally), sampled at Block 5's offsets v_n; the plane is k_i = round(z_i / dz),
z_k = k_i dz. Each traced segment q of the scene that is not on the node's
branch (not both ends' ids in branch.ids) is a neighbour; its predicted dip at
sample n is today's predictor (calibration_scene.shadow_depths, gl):

    B delta_q(v_n) = shadow_depths(rho_qn, max(r_q0, r_q1), mu, sigma_r(dz_q), B),

rho_qn the 2-D distance (um) from the sample o + v_n y_hat to the segment,
dz_q = min_ends |z_end - z_k| (the end nearer the plane in depth). Then

    P_i(v_n) = sum_q B delta_q(v_n)  (gl),   D_i = shadow_depth(0, r_i, mu, sigma_r(0), B)  (gl),
    C_i = (sum_n w_n P_i(v_n) / sum_n w_n) / D_i,   w_n = exp(-|v_n| / lambda),  lambda = sgt_lambda_um.

Classes, first that holds: overlap (a neighbour's footprint -- within its
larger end radius of its axis -- meets the own branch's footprint at a point of
the line's square |u|, |v| <= profile_half_um, read on the grid v_n x v_n);
close_in_focus (one neighbour with lateral distance from the line's centre
<= close_lateral_um and dz_q <= close_defocus_um carries >= 1/2 of sum_n w_n P);
diffuse (C_i >= clean_index_max); clean.

Readings of the entry that SPEC.md leaves open (the implementer's, reported to
the orchestrator, PROVISIONAL): the 2-D point-to-segment distance sets rho_qn
and the lateral distance from the line's centre; "the nearer end" is the end
nearer the plane in depth; the soma is not among the neighbours (per_neighbour
is (n_q, n_v), the predictor is a segment's, the Scene carries no soma depth);
the own footprint is the branch's consecutive nodes, each segment at its larger
end radius; n_neighbours counts the neighbours whose dip on the line reaches
cleaning.shadow_floor_gl.

No file access, no plotting. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from . import background, path
from .calibration_scene import OWN, patch_frame, segment_rows, shadow_depth, shadow_depths
from .profiles import profile_offsets
from ..model.kernel import sigma_r

CLASSES = ("overlap", "close_in_focus", "diffuse", "clean")   # in this order of precedence
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
        defocus[m] = min(abs(a[2] - z_k), abs(b[2] - z_k))
        lateral[m] = float(background.segment_distance(o[0], o[1], a, b))
        rho = background.segment_distance(px, py, a, b)
        r_q = float(np.max(scene.seg_radius[q]))
        per[m] = shadow_depths(rho, r_q, float(mu_per_um), float(sigma_r(defocus[m], cfg.renderer)), float(B_gl))
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
    if own.any():
        for m, q in enumerate(qs):
            d = background.segment_distance(px[own], py[own], scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:])
            out[m] = bool(np.any(d <= float(np.max(scene.seg_radius[q]))))
    return out


def node_contamination(branch, index, scene, B_gl, mu_per_um, cfg):
    """-> dict: node_id, own_dip_gl, index, class, n_neighbours, worst_child_id, worst_lateral_um, worst_defocus_um,
    worst_share. n_neighbours: neighbours whose predicted dip on the line reaches cleaning.shadow_floor_gl;
    worst: the neighbour of the largest weighted share of sum_n w_n P (-1 / NaN when none carries any)."""
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
    if qs.size and np.any(_overlap(branch, scene, qs, frame_line, cfg)):
        cls = "overlap"
    elif np.any(close):
        cls = "close_in_focus"
    elif C >= c_cfg.clean_index_max:
        cls = "diffuse"
    else:
        cls = "clean"
    n_nb = int(np.sum(np.max(per, axis=1) >= c_cfg.shadow_floor_gl)) if qs.size else 0
    return dict(node_id=int(branch.ids[i]), own_dip_gl=float(D), index=float(C), **{"class": cls},
                n_neighbours=n_nb, worst_child_id=worst[0], worst_lateral_um=worst[1],
                worst_defocus_um=worst[2], worst_share=worst[3])


def pick_examples(rows, counts, rng):
    """-> list of node ids (D-130.O7). counts: a dict {key: n} or a sequence in EXAMPLE_KEYS' order (the order of
    CleaningConfig.example_counts); a class key draws n of its rows at random (rng, without replacement),
    NEAREST_SOMA takes the n rows of smallest 'soma_distance_um' (a key the caller adds). No node twice; a short
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
            free = sorted((r for r in free if np.isfinite(r.get("soma_distance_um", np.nan))),
                          key=lambda r: (float(r["soma_distance_um"]), int(r["node_id"])))
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
