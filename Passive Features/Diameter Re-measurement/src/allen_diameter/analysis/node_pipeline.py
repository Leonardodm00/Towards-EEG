"""The per-node chain -- Blocks 5 and 13 in specs/SPEC.md (handoff steps 1-5,
Eqs. 1-5, 9; D-052.3, D-061).

ONE function, measure_node, for real and synthetic blocks: the block provider
is either allen_image_io.fetch_zblock (real, Colab) or model/render.py's
synthetic_block (phantoms), so the bias table measures the bias of the code
actually used on real nodes (procedure s.3.7). Since Block 13 measure_node is
the pilot (Block 5, below; measure_pilot alone, for the diagnostics) followed
by the Allen-guided plane iteration and Block 12's fit at the plane it ends
on (analysis/plane_iteration.py); both make node_request's one request.

Passes (direction_redraws = R, line_fit_window_um = L): pass 0 draws each
node's measuring line from the raw SWC direction (the line through the raw
positions in the node's window) on the nodes within (R + 1) L/2 of node i;
pass p redraws from the line through the pass p-1 centres on the nodes within
(R + 1 - p) L/2; node i's final direction is the line through the pass-R
centres in its window. Each pass at node j: focus scores over the node's
planes (the configured rule, D-030, and the dip depth of handoff Eq. 1 beside
it as a diagnostic), k* and sub-plane depth (Eq. 2) from the configured rule,
B_bar in k* (D-018.1), the Eq. 9 profile through the node fitted (Block 3),
centre c_j (Eq. 3).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from . import background, focus, path, profiles
from .fit import FitResult, fit_profile
from .plane_iteration import Scene, gradient_iteration, iteration_state, scene_segments, sgt_at_plane


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
    kk: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))   # the node's planes
    F: np.ndarray = field(default_factory=lambda: np.empty(0))   # configured focus rule (D-030)
    D: np.ndarray = field(default_factory=lambda: np.empty(0))   # dip depth (handoff Eq. 1), diagnostic


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
    focus_rule: str                  # measure.focus_rule of the run (D-030)
    focus_planes: np.ndarray         # plane indices of the final pass's focus curves
    focus_score: np.ndarray          # the configured rule's score on those planes (k* is its argmax)
    focus_depth: np.ndarray          # the dip depth (handoff Eq. 1) on the same planes, diagnostic
    k_star_depth: int                # argmax plane of focus_depth (-1 when none), diagnostic
    # Block 13 (D-052.3, D-061): the iteration, the SGT fit's shape at k_iter and the pilot's columns (pilot_view);
    # the defaults are a pilot's, so that a NodeResult built before Block 13 stays valid
    k_iter: int = -1                 # the deliverable's plane, the iteration's last pick; -1 without one
    iter_status: str = ""            # plane_iteration.ITER_STATUSES; "" when no iteration ran (a pilot)
    n_rounds: int = 0
    n_restacks: int = 0              # squares rebuilt for longer lines (cut from the block, or one provider call)
    w_hat_um: float = float("nan")   # the SGT fit at k_iter: the kernel's FWHM and shape (D-052.1-2)
    beta_hat: float = float("nan")
    xi_hat: float = float("nan")
    zeta_hat: float = float("nan")
    irls_ok: Optional[bool] = None
    at_bound: Tuple[str, ...] = ()
    d_pilot_um: float = float("nan")      # the pilot's Block 3 fit at k* (the fitted columns of measure_pilot)
    mu_pilot_per_um: float = float("nan")
    v0_pilot_um: float = float("nan")
    alpha_pilot: float = float("nan")
    B_bar_pilot: float = float("nan")
    pilot_status: str = ""           # the pilot's fit_status; "" in a pilot itself
    pilot_flags: Tuple[str, ...] = ()
    plane_fit: Optional[object] = field(default=None, compare=False, repr=False)   # fit_sgt.PlaneSGT at k_iter
    iteration: Optional[dict] = field(default=None, compare=False, repr=False)     # gradient_iteration's record, no squares


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
    F = np.full(kk.size, np.nan)            # the configured rule (D-030)
    D = np.full(kk.size, np.nan)            # the dip depth of handoff Eq. 1, diagnostic
    radius = float(branch.radius_um[j])
    for n, k in enumerate(kk):
        idx = int(k - ks[0])
        if valid[idx]:
            B_ovr = _bbar(block[idx], frame, o, branch, m)[0] if m.focus_bg_rule == "same_as_bbar" else None
            I = profiles.sample_profile(block[idx], frame, o, y_hat, e_u, v)
            scores = focus.plane_scores(I, v, m, B_ovr, radius)
            F[n], D[n] = scores[m.focus_rule], scores["dip_depth"]
    out.kk, out.F, out.D = kk, F, D
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


def _window(branch, i, m):
    """The pilot's window: the nodes within (R + 1) L / 2 of node i along the path and their raw SWC directions."""
    R, L = int(m.direction_redraws), float(m.line_fit_window_um)
    nodes = path.window(branch.s_um, i, 0.5 * (R + 1) * L)
    return nodes, {int(j): _raw_direction(branch, int(j), L) for j in nodes}


def _pilot_planes(branch, nodes, t_raw, cfg):
    """(k_lo, k_hi): Block 5's planes, the lowest to the highest of the window
    nodes' pilot ranges from their raw directions -- the planes the pilot reads
    (D-061.2: the pilot unchanged; a pass whose redrawn direction widens its
    range is clipped to them, as Block 5's own request clipped it)."""
    m, dz = cfg.measure, cfg.acquisition.dz_um
    ranges = [_plane_range(branch.xyz_um[j, 2], path.angles(t_raw[int(j)])[1], m, dz) for j in nodes]
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def _request(branch, i, nodes, t_raw, cfg):
    m, p, dz = cfg.measure, cfg.acquisition.res0_um, cfg.acquisition.dz_um
    P = branch.xyz_um[nodes]
    lo, hi = _pilot_planes(branch, nodes, t_raw, cfg)
    k_swc = int(round(float(branch.xyz_um[i, 2]) / dz))
    left = int(math.floor((P[:, 0].min() - m.block_half_um) / p)) - 1
    top = int(math.floor((P[:, 1].min() - m.block_half_um) / p)) - 1
    width = int(math.ceil((P[:, 0].max() + m.block_half_um) / p)) + 2 - left
    height = int(math.ceil((P[:, 1].max() + m.block_half_um) / p)) + 2 - top
    return (left, top, width, height, min(lo, k_swc - int(m.iter_planes_half)),
            max(hi, k_swc + int(m.iter_planes_half)))


def _planes_of(blk, lo, hi):
    """The block blk = (block, ks, valid, frame) restricted to its planes lo..hi."""
    block, ks, valid, frame = blk
    ks = np.asarray(ks)
    sel = (ks >= int(lo)) & (ks <= int(hi))
    return np.asarray(block)[sel], ks[sel], np.asarray(valid, dtype=bool)[sel], frame


def node_request(branch, i, cfg):
    """(left, top, width, height, k_lo, k_hi): node i's one provider request
    (D-061.1; Block 13.3b), made by measure_pilot and measure_node alike: Block
    5's xy block (block_half_um about the pilot's window nodes, full-resolution
    pixels) over the planes from the lowest to the highest of Block 5's planes
    (their pilot ranges, +-planes_half widened for tilt by their raw
    directions) and of K_i = k_SWC +- iter_planes_half, k_SWC = round(z_i /
    dz). The pilot reads Block 5's planes of it, the iteration K_i."""
    nodes, t_raw = _window(branch, i, cfg.measure)
    return _request(branch, i, nodes, t_raw, cfg)


def _pilot(branch, i, blk, nodes, t_raw, cfg, reg):
    """Block 5 on the block blk (the module docstring's passes, unchanged): the
    pilot's NodeResult, the Block 13 fields at their defaults."""
    m, dz = cfg.measure, cfg.acquisition.dz_um
    R, L = int(m.direction_redraws), float(m.line_fit_window_um)
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
    k_depth = int(last.kk[int(np.nanargmax(last.D))]) if np.any(np.isfinite(last.D)) else -1
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
        fit_status="none" if fit is None else fit.status, flags=tuple(flags), fit=fit, focus_rule=str(m.focus_rule),
        focus_planes=np.asarray(last.kk, dtype=int), focus_score=last.F, focus_depth=last.D, k_star_depth=k_depth)


def measure_pilot(branch, i, provider, cfg, reg=None):
    """The pilot of node i of a Branch (Block 5; D-035, D-061.2): with one
    provider call, node_request's (the request measure_node makes, so the
    diagnostics that call this are served from the image cache), the passes of
    the module docstring on Block 5's planes of the block (_pilot_planes: the
    planes beyond them are not read, so the pilot is Block 5's), k* by the
    configured focus rule, the centre, the heading, the tilt, and Block 3's
    Gaussian fit at k* with its flags.

    provider(left, top, width, height, k_lo, k_hi) -> (block, ks, valid, frame);
    cfg: DiameterConfig; reg: optional dict returned by
    allen_image_align.registration_check (keys verdict, lateral_offset_um,
    z_offset_um are read) -- real data only. Returns NodeResult with the Block
    13 fields at their defaults (pilot_status "": pilot_view is the identity).
    """
    nodes, t_raw = _window(branch, i, cfg.measure)
    blk = provider(*_request(branch, i, nodes, t_raw, cfg))
    return _pilot(branch, i, _planes_of(blk, *_pilot_planes(branch, nodes, t_raw, cfg)), nodes, t_raw, cfg, reg)


# the NodeResult fields Block 13 added, at their defaults in a pilot
ITER_FIELDS = ("k_iter", "iter_status", "n_rounds", "n_restacks", "w_hat_um", "beta_hat", "xi_hat", "zeta_hat",
               "irls_ok", "at_bound", "d_pilot_um", "mu_pilot_per_um", "v0_pilot_um", "alpha_pilot", "B_bar_pilot",
               "pilot_status", "pilot_flags", "plane_fit", "iteration")


def pilot_view(result):
    """The pilot's NodeResult of a measured node (Block 13.3b): result with
    its fitted columns, status, flags and B_bar replaced by the pilot's
    (d_pilot_um, mu_pilot_per_um, v0_pilot_um, alpha_pilot, pilot_status,
    pilot_flags, B_bar_pilot) and the Block 13 fields at their defaults, i.e.
    what measure_pilot returns for the same node and block; result itself when
    pilot_status is "" (a pilot, or a NodeResult built before Block 13)."""
    if not result.pilot_status:
        return result
    defaults = {f.name: f.default for f in dataclasses.fields(NodeResult) if f.name in ITER_FIELDS}
    return dataclasses.replace(result, d_hat_um=result.d_pilot_um, mu_hat_per_um=result.mu_pilot_per_um,
                               v0_hat_um=result.v0_pilot_um, alpha_hat=result.alpha_pilot,
                               fit_status=result.pilot_status, flags=result.pilot_flags, B_bar=result.B_bar_pilot,
                               **defaults)


def _iter_at_edge(rec, ks, k, margin):
    """Block 5's stack_edge rule (focus.best_plane) on the iteration's curve: k within margin planes of the first
    or last plane with a finite G in the round rec that picked it."""
    ok = np.flatnonzero(np.isfinite(np.asarray(rec.get("G"), dtype=float)))
    if not ok.size:
        return False
    n = int(k) - int(ks[0])
    return n - int(ok[0]) < margin or int(ok[-1]) - n < margin


def deliverable_flags(pilot, rec, ks, m, ending=None):
    """The flags of the selection S at the deliverable's plane (D-061.3), Block
    5's rules and order applied to the record rec of the round that picked
    k_iter (plane_iteration.sgt_at_plane's keys, with the round's G over the
    planes ks), or None when no round picked a plane: vertical (the pilot's);
    profile_nan when the profile there holds a NaN, else, with a fit, faint and
    crossing from it and its B_bar; stack_edge (_iter_at_edge); bbar_few when
    the plane's B_bar is flagged (and when there is no plane, as Block 5 without
    k*); steep (the pilot's); dark when the fit's alpha_hat > alpha_dark_flag;
    the fit's status when it is not converged; and, last, the label
    iter_<ending> when the iteration's ending (plane_iteration.ITER_STATUSES)
    is given and is not converged (D-068: a label, not in
    phantoms.SELECTION_FLAGS). Returns a tuple."""
    flags = ["vertical"] if pilot.vertical else []
    fitted = rec is not None and rec.get("plane_fit") is not None
    if rec is not None and rec.get("profile") is not None:
        I, v = np.asarray(rec["profile"], dtype=float), np.asarray(rec["v_um"], dtype=float)
        if not np.all(np.isfinite(I)):
            flags.append("profile_nan")
        elif fitted:
            flags += _profile_flags(I, v, float(rec["B_bar"]), m)
    if rec is not None and _iter_at_edge(rec, ks, rec["k"], m.edge_margin_planes):
        flags.append("stack_edge")
    if rec is None or not rec["bbar_ok"]:
        flags.append("bbar_few")
    if pilot.steep:
        flags.append("steep")
    if fitted and rec["alpha_hat"] > m.alpha_dark_flag:
        flags.append("dark")
    if fitted and rec["fit_status"] != "converged":
        flags.append(str(rec["fit_status"]))
    if ending is not None and ending != "converged":
        flags.append("iter_%s" % ending)                # D-068: kept, flagged
    return tuple(flags)


def measure_node(branch, i, provider, cfg, reg=None, scene=None):
    """Measure node i of a Branch (Block 13.3b; D-052.3, D-061): the
    deliverable, the same function for phantoms and real nodes (D-057 (iii)).

    One provider call, node_request's, and one more per restack that leaves
    the block (D-061.1). (1) The pilot on that block (as measure_pilot). (2)
    The Allen-guided iteration (plane_iteration.gradient_iteration; D-042,
    D-044) on the block's planes K_i = k_SWC +- iter_planes_half, from d_0 =
    2 branch.radius_um[i] (Allen's on a real node, the phantom's on a phantom),
    with line_mult = iter_line_mult, max_rounds = iter_max_rounds, bg_margin_um
    = iter_bg_margin_um, round 1's line at least iter_min_first_samples
    samples (D-069), the square +-block_half_um about the SWC node, a
    restack of half-width h + 2 pixels, and fit_at = plane_iteration.
    sgt_at_plane (Block 12's fit, D-052.2); the background masks the scene's
    segments and soma (D-064; scene None: Scene.from_branch(branch), a
    phantom's own tube). (3) k_iter = the last plane picked; the deliverable is
    the fit of the round that picked it, whatever the iteration's ending, the
    flags deliverable_flags' with the label of a non-converged ending (D-068).

    Returns NodeResult: the centre, heading, tilt, k*, sub-plane depth,
    steep, vertical, the focus curves and `fit` (Block 3's at k*) are the
    pilot's; d_hat_um, mu_hat_per_um, v0_hat_um, alpha_hat, B_bar (of k_iter),
    fit_status (the SGT fit's, "none" without one) and flags are the
    deliverable's; the Block 13 fields hold the iteration, the SGT fit's shape
    and the pilot's columns (pilot_view). provider, cfg, reg: as measure_pilot.
    """
    m = cfg.measure
    nodes, t_raw = _window(branch, i, m)
    blk = provider(*_request(branch, i, nodes, t_raw, cfg))
    pilot = _pilot(branch, i, _planes_of(blk, *_pilot_planes(branch, nodes, t_raw, cfg)), nodes, t_raw, cfg, reg)
    k_swc = int(round(float(branch.xyz_um[i, 2]) / cfg.acquisition.dz_um))
    bb, kb, vb, frame = _planes_of(blk, k_swc - int(m.iter_planes_half), k_swc + int(m.iter_planes_half))
    scene = Scene.from_branch(branch) if scene is None else scene
    segments = scene_segments(scene, frame, np.shape(bb)[1:], branch)
    px = float(frame.res_um_px)

    def square(half_um):
        return iteration_state(bb, kb, vb, frame, branch, i, pilot.theta_rad, pilot, segments, scene.soma, half_um,
                               provider, k_swc, label="node %d" % int(branch.ids[i]))
    res = gradient_iteration(square(float(m.block_half_um)), cfg, d_start_um=2.0 * float(branch.radius_um[i]),
                             line_mult=m.iter_line_mult, max_rounds=m.iter_max_rounds,
                             bg_margin_um=m.iter_bg_margin_um, restack=lambda h: square(float(h) + 2.0 * px),
                             fit_at=lambda cur, k: sgt_at_plane(cur, cfg, k),
                             min_first_samples=m.iter_min_first_samples)
    picked = [r for r in res["rounds"] if r["k"] is not None]
    rec = picked[-1] if picked else None
    fitted = rec is not None and rec.get("plane_fit") is not None
    nan = float("nan")

    def num(key):
        return float(rec[key]) if fitted else nan
    return dataclasses.replace(
        pilot, B_bar=nan if rec is None else float(rec["B_bar"]), d_hat_um=num("d_hat_um"),
        mu_hat_per_um=num("mu_hat_per_um"), v0_hat_um=num("v0_hat_um"), alpha_hat=num("alpha_hat"),
        fit_status=str(rec["fit_status"]) if fitted else "none",
        flags=deliverable_flags(pilot, rec, kb, m, str(res["status"])),
        k_iter=-1 if rec is None else int(rec["k"]), iter_status=str(res["status"]), n_rounds=int(res["n_rounds"]),
        n_restacks=int(res["n_restacks"]), w_hat_um=num("w_hat_um"), beta_hat=num("beta_hat"), xi_hat=num("xi_hat"),
        zeta_hat=num("zeta_hat"), irls_ok=rec["irls_ok"] if fitted else None,
        at_bound=tuple(rec["at_bound"]) if fitted else (), d_pilot_um=pilot.d_hat_um,
        mu_pilot_per_um=pilot.mu_hat_per_um, v0_pilot_um=pilot.v0_hat_um, alpha_pilot=pilot.alpha_hat,
        B_bar_pilot=pilot.B_bar, pilot_status=pilot.fit_status, pilot_flags=pilot.flags,
        plane_fit=rec["plane_fit"] if fitted else None, iteration={k: v for k, v in res.items() if k != "st"})
