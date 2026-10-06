"""The per-node chain -- Block 5 in specs/SPEC.md (handoff steps 1-5, Eqs. 1-5, 9).

ONE function, measure_node, for real and synthetic blocks: the block provider
is either allen_image_io.fetch_zblock (real, Colab) or model/render.py's
synthetic_block (phantoms), so the bias table measures the bias of the code
actually used on real nodes (procedure s.3.7).

Passes (direction_redraws = R, line_fit_window_um = L): pass 0 draws each
node's measuring line from the raw SWC direction (the line through the raw
positions in the node's window) on the nodes within (R + 1) L/2 of node i;
pass p redraws from the line through the pass p-1 centres on the nodes within
(R + 1 - p) L/2; node i's final direction is the line through the pass-R
centres in its window. Each pass at node j: focus scores over the node's
planes (Eq. 1), k* and sub-plane depth (Eq. 2), B_bar in k* (D-018.1), the
Eq. 9 profile through the node fitted (Block 3), centre c_j (Eq. 3).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from . import background, focus, path, profiles
from .fit import FitResult, fit_profile


@dataclass(frozen=True)
class Branch:
    """Nodes proximal -> distal; xyz in the global image frame (x, y um; z stage um)."""

    ids: np.ndarray
    types: np.ndarray
    xyz_um: np.ndarray
    radius_um: np.ndarray
    s_um: np.ndarray

    def __post_init__(self):
        n = len(self.ids)
        if n < 2 or np.shape(self.xyz_um) != (n, 3) or len(self.types) != n or len(self.radius_um) != n \
                or len(self.s_um) != n or not np.all(np.isfinite(self.xyz_um)) or np.any(np.diff(self.s_um) < 0):
            raise ValueError("Branch: need >= 2 nodes, xyz (n, 3), n types/radii/s, s non-decreasing")

    @staticmethod
    def from_points(xyz_um, radius_um, ids=None, types=None):
        P = np.asarray(xyz_um, dtype=float)
        n = P.shape[0]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        return Branch(np.arange(n) if ids is None else np.asarray(ids), np.full(n, 3) if types is None
                      else np.asarray(types), P, np.broadcast_to(np.asarray(radius_um, float), (n,)).copy(), s)


@dataclass
class _Pass:
    t: np.ndarray
    theta: float
    phi: float
    y_hat: np.ndarray
    e_u: np.ndarray
    k_star: Optional[int] = None
    z_sub: float = float("nan")
    at_edge: bool = False
    B_bar: float = float("nan")
    bbar_ok: bool = False
    centre: Optional[np.ndarray] = None
    fit: Optional[FitResult] = None
    F: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class NodeResult:
    node_id: int
    type: int
    x_um: float
    y_um: float
    z_um: float
    path_um: float
    reg_verdict: str
    s_star_um: float
    dz_star_um: float
    k_star: int
    z_sub_um: float
    cx_um: float
    cy_um: float
    cz_um: float
    theta_rad: float
    phi_rad: float
    steep: bool
    vertical: bool
    B_bar: float
    B_bar_region: str
    d_hat_um: float
    mu_hat_per_um: float
    v0_hat_um: float
    alpha_hat: float
    fit_status: str
    flags: Tuple[str, ...]
    fit: Optional[FitResult]
    focus_F: np.ndarray


def _raw_direction(branch, j, L):
    w = path.window(branch.s_um, j, 0.5 * L)
    if w.size < 2:
        w = np.arange(max(0, j - 1), min(len(branch.ids), j + 2))
    P = branch.xyz_um[w]
    return path.orient(path.tls_direction(P), P[0], P[-1])


def _plane_range(z, phi, m, dz):
    k = int(round(z / dz))
    extra = int(math.ceil(0.5 * m.line_fit_window_um * math.sin(phi) / dz)) if m.planes_widen_for_tilt else 0
    return k - m.planes_half - extra, k + m.planes_half + extra


def _bbar(block2d, frame, o, branch, m):
    """Masked median of the node's block (block_half_um around o) in one plane."""
    p = float(frame.res_um_px)
    H, W = block2d.shape
    c0 = max(0, int(math.ceil((o[0] - m.block_half_um) / p - frame.left)))
    c1 = min(W, int(math.floor((o[0] + m.block_half_um) / p - frame.left)) + 1)
    r0 = max(0, int(math.ceil((o[1] - m.block_half_um) / p - frame.top)))
    r1 = min(H, int(math.floor((o[1] + m.block_half_um) / p - frame.top)) + 1)
    sub = np.asarray(block2d[r0:r1, c0:c1], dtype=float)
    if m.bbar_region == "block":
        mask = np.zeros(sub.shape, dtype=bool)
    elif m.bbar_region == "block_masked":
        mask = background.mask_near_branch(sub.shape, frame.left + c0, frame.top + r0, p, branch.xyz_um,
                                           branch.radius_um, m.bbar_mask_margin_um)
    else:
        raise NotImplementedError("bbar_region %r needs the whole plane (not in Block 5)" % (m.bbar_region,))
    B, _frac, ok = background.masked_median(sub, mask, m.bbar_min_unmasked_frac)
    return B, ok


node_background = _bbar   # public name (Block 10's plane scan applies the same rule in every plane)


def _node_pass(blk, branch, j, t, m, dz):
    block, ks, valid, frame = blk
    theta, phi, y_hat, e_u = path.angles(t)
    out = _Pass(t, theta, phi, y_hat, e_u)
    o = branch.xyz_um[j, :2]
    v = profiles.profile_offsets(m)
    k_lo, k_hi = _plane_range(branch.xyz_um[j, 2], phi, m, dz)
    k_lo, k_hi = max(k_lo, int(ks[0])), min(k_hi, int(ks[-1]))
    kk = np.arange(k_lo, k_hi + 1)          # empty when the node's planes miss the block
    F = np.full(kk.size, np.nan)
    for n, k in enumerate(kk):
        idx = int(k - ks[0])
        if valid[idx]:
            B_ovr = _bbar(block[idx], frame, o, branch, m)[0] if m.focus_bg_rule == "same_as_bbar" else None
            I = profiles.sample_profile(block[idx], frame, o, y_hat, e_u, v)
            F[n] = focus.focus_score(I, v, m, B_ovr)[0]
    out.F = F
    n_star, z_sub, _plateau, at_edge = focus.best_plane(F, kk * dz, m, dz)
    if n_star is None:
        out.centre = np.array([o[0], o[1], branch.xyz_um[j, 2]])
        return out
    out.k_star, out.z_sub, out.at_edge = int(kk[n_star]), z_sub, at_edge
    plane = block[int(kk[n_star] - ks[0])]
    out.B_bar, out.bbar_ok = _bbar(plane, frame, o, branch, m)
    I = profiles.sample_profile(plane, frame, o, y_hat, e_u, v, profiles.n_along(m), m.profile_step_um)
    out.centre = np.array([o[0], o[1], z_sub])
    if np.all(np.isfinite(I)) and out.B_bar > 0:
        out.fit = fit_profile(v, I, phi, out.B_bar, m)
        out.centre[:2] = o + out.fit.v0_hat_um * y_hat
    return out


def _profile_flags(I, v, B_bar, m):
    flags = []
    smooth = ndimage.gaussian_filter1d(I, m.focus_smooth_px, mode="nearest") if m.focus_smooth_px > 0 else I
    n0 = int(np.argmin(smooth))
    depth = B_bar - smooth[n0]
    if depth < m.faint_min_dip_gl:
        flags.append("faint")
    interior = np.arange(1, smooth.size - 1)
    minima = interior[(smooth[interior] < smooth[interior - 1]) & (smooth[interior] <= smooth[interior + 1])]
    far = minima[np.abs(v[minima] - v[n0]) >= m.second_dip_min_sep_um]
    if depth > 0 and np.any(B_bar - smooth[far] >= m.second_dip_rel * depth):
        flags.append("crossing")
    return flags


def _num(x):
    """A registration value as float: None (JSON null, e.g. s* of a NOT ON node) and a missing key are NaN."""
    return float("nan") if x is None else float(x)


def measure_node(branch, i, provider, cfg, reg=None):
    """Measure node i of a Branch with one provider call (see the module docstring).

    provider(left, top, width, height, k_lo, k_hi) -> (block, ks, valid, frame);
    cfg: DiameterConfig; reg: optional dict returned by
    allen_image_align.registration_check (keys verdict, lateral_offset_um,
    z_offset_um are read) -- real data only. Returns NodeResult.
    """
    m, p, dz = cfg.measure, cfg.acquisition.res0_um, cfg.acquisition.dz_um
    R, L = int(m.direction_redraws), float(m.line_fit_window_um)
    nodes = path.window(branch.s_um, i, 0.5 * (R + 1) * L)
    t_raw = {int(j): _raw_direction(branch, int(j), L) for j in nodes}
    P = branch.xyz_um[nodes]
    ranges = [_plane_range(branch.xyz_um[j, 2], path.angles(t_raw[int(j)])[1], m, dz) for j in nodes]
    left = int(math.floor((P[:, 0].min() - m.block_half_um) / p)) - 1
    top = int(math.floor((P[:, 1].min() - m.block_half_um) / p)) - 1
    width = int(math.ceil((P[:, 0].max() + m.block_half_um) / p)) + 2 - left
    height = int(math.ceil((P[:, 1].max() + m.block_half_um) / p)) + 2 - top
    blk = provider(left, top, width, height, min(r[0] for r in ranges), max(r[1] for r in ranges))
    passes = {int(j): _node_pass(blk, branch, int(j), t_raw[int(j)], m, dz) for j in nodes}
    for q in range(1, R + 1):
        current = path.window(branch.s_um, i, 0.5 * (R + 1 - q) * L)
        new = {}
        for j in current:
            w = [int(x) for x in path.window(branch.s_um, int(j), 0.5 * L) if int(x) in passes]
            C = np.array([passes[x].centre for x in w])
            t = path.orient(path.tls_direction(C), C[0], C[-1]) if len(w) >= 2 else passes[int(j)].t
            new[int(j)] = _node_pass(blk, branch, int(j), t, m, dz)
        passes = new
    w = [int(x) for x in path.window(branch.s_um, i, 0.5 * L) if int(x) in passes]
    C = np.array([passes[x].centre for x in w])
    t_i = path.orient(path.tls_direction(C), C[0], C[-1]) if len(w) >= 2 else passes[i].t
    theta, phi, y_hat, e_u = path.angles(t_i)
    flags = []
    vertical = phi > math.radians(m.phi_vertical_deg)
    if vertical:
        flags.append("vertical")
        th = path.circular_mean([passes[x].theta for x in w if x != i and passes[x].phi <= math.radians(m.phi_vertical_deg)])
        if th is not None:
            theta, y_hat, e_u = th, np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    last = passes[i]
    steep = math.tan(phi) >= m.steep_tan_diagnostic
    fit, I = None, None
    if last.k_star is not None:
        block, ks, valid, frame = blk
        v = profiles.profile_offsets(m)
        I = profiles.sample_profile(block[int(last.k_star - ks[0])], frame, last.centre[:2], y_hat, e_u, v,
                                    profiles.n_along(m), m.profile_step_um)
        if not np.all(np.isfinite(I)):
            flags.append("profile_nan")
        elif last.B_bar > 0:
            fit = fit_profile(v, I, phi, last.B_bar, m)
            flags += _profile_flags(I, v, last.B_bar, m)
    if last.at_edge:
        flags.append("stack_edge")
    if not last.bbar_ok:
        flags.append("bbar_few")
    if steep:
        flags.append("steep")
    if fit is not None and fit.alpha_hat > m.alpha_dark_flag:
        flags.append("dark")
    if fit is not None and fit.status != "converged":
        flags.append(fit.status)
    nan = float("nan")
    reg = reg or {}
    xyz = branch.xyz_um[i]
    return NodeResult(
        node_id=int(branch.ids[i]), type=int(branch.types[i]), x_um=float(xyz[0]), y_um=float(xyz[1]),
        z_um=float(xyz[2]), path_um=float(branch.s_um[i]), reg_verdict=str(reg.get("verdict") or ""),
        s_star_um=_num(reg.get("lateral_offset_um")), dz_star_um=_num(reg.get("z_offset_um")),
        k_star=-1 if last.k_star is None else int(last.k_star), z_sub_um=float(last.z_sub),
        cx_um=float(last.centre[0]), cy_um=float(last.centre[1]), cz_um=float(last.centre[2]),
        theta_rad=float(theta), phi_rad=float(phi), steep=bool(steep), vertical=bool(vertical),
        B_bar=float(last.B_bar), B_bar_region=str(m.bbar_region),
        d_hat_um=nan if fit is None else fit.d_hat_um, mu_hat_per_um=nan if fit is None else fit.mu_hat_per_um,
        v0_hat_um=nan if fit is None else fit.v0_hat_um, alpha_hat=nan if fit is None else fit.alpha_hat,
        fit_status="none" if fit is None else fit.status, flags=tuple(flags), fit=fit, focus_F=last.F)
