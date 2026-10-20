"""The Allen-guided plane iteration of the gradient energy -- Block 13 in
specs/SPEC.md (D-042, D-044, D-052.3). Moved on 2026-10-20 from
scripts/plane_differences.py (Block 11), the bodies unchanged; the script
imports these names, so its evaluations and this module are one
implementation.

For node j with the starting diameter d_0 (twice the radius its branch
carries: Allen's on a real node), the multiplier m (ITER_LINE_MULT = 3,
D-044: the lines +-1.5 d) and at most R rounds (5, D-042): round i takes the
line of half-length h_i = m d_{i-1} / 2 through the SWC node across the
pilot's heading, the gradient energy over the whole line,
    G^(h_i)_k = B_k^-2 * trapezoid over the line of (dI~_k/dv)^2,
in every plane k (focus.plane_gradient_energies; B_k = block_background: the
interquartile mean of the block's pixels farther than the radius + a margin
from every traced segment and from the soma), and its plane
k_i = argmax_k G (the first of equal values). A plane picked before ends the
iteration -- "converged" when it is the previous round's, "cycle" otherwise --
and the fit made there is reused; else d_i is the fit at k_i (fit_at; by
default refit_at_plane, Block 5's Gaussian fit with every other input the
pilot's: the diagnostics' fit). Other endings: ITER_STATUSES.

The functions read a state dict st (built by scripts/plane_differences.
node_stack): stack, ks, valid, frame (the square about the node), o (the SWC
node's xy, um), theta (the pilot's heading), block, block_frame (the pilot's
block over the same planes), segments (n, 9), soma (m, 3), square_half_um,
allen_radius_um, result (the pilot's NodeResult), branch.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import background, focus, profiles

# the line multiplier of the Allen-guided iteration, a line's full width in units of the diameter sizing it (D-044,
# the user's choice of 2026-10-12: m = 3, the lines +-1.5 d)
ITER_LINE_MULT = 3.0


# how the Allen-guided iteration ends (D-042): a plane picked again (the previous round's, or an earlier one), the
# round limit, or a round that cannot go on
ITER_STATUSES = ("converged", "cycle", "max_rounds", "fit_failed", "no_plane", "line_short", "square_small")


def check_iteration(line_mult, max_rounds):
    """(line_mult as float, max_rounds as int) of the Allen-guided iteration;
    ValueError unless line_mult is finite and positive and max_rounds an
    integer >= 1 (5 and 5.0 pass; 5.5, True, "5" do not)."""
    m = float(line_mult)
    if not (math.isfinite(m) and m > 0):
        raise ValueError("the iteration's line multiplier must be finite and > 0, got %r" % (line_mult,))
    try:
        R = int(max_rounds)
        ok = not isinstance(max_rounds, (bool, str)) and R == max_rounds and R >= 1
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise ValueError("max_rounds must be an integer >= 1, got %r" % (max_rounds,))
    return m, R


def iterate_planes(d_start_um, line_mult, plane_of, fit_at, max_rounds=5):
    """The Allen-guided iteration's rule (D-042), on two callables so that it
    can be checked by hand. Round i = 1..max_rounds: the line of half-length
    h_i = line_mult * d_{i-1} / 2, d_0 = d_start_um; (k_i, info) = plane_of(h_i),
    k_i the plane that line picks, or None with info["stop"] (default
    "no_plane") saying why. If k_i was picked in an earlier round the iteration
    stops, "converged" when it is the previous round's plane and "cycle"
    otherwise, and the fit made there is reused; else fit = fit_at(k_i), a dict
    with d_hat_um, and d_i = d_hat_um sizes the next line ("fit_failed" when it
    is not finite and positive). "max_rounds" after max_rounds rounds without a
    repeat. ValueError unless d_start_um and line_mult are finite and positive
    and max_rounds is an integer >= 1. Returns dict(rounds: [dict(round,
    d_in_um, d_in_k (None in round 1, else the plane whose fit sized the line),
    half_um, k, **info, **fit, refit)], status, n_rounds, k_first, k_final (the
    plane of the last round that picked one, or None), d_final_um (the fit
    there; NaN when it failed or there is none))."""
    # Custom: the loop ends on a repeated plane, a discrete event. Checked: scipy.optimize.fixed_point (Steffensen
    # acceleration of a continuous map; this map is piecewise constant in d, its value set by a discrete plane).
    m, R = check_iteration(line_mult, max_rounds)
    d = float(d_start_um)
    if not (math.isfinite(d) and d > 0):
        raise ValueError("the iteration needs a finite positive starting diameter, got %r" % (d_start_um,))
    rounds, fits, status, d_k = [], {}, "max_rounds", None
    for i in range(1, R + 1):
        h = 0.5 * m * d
        k, info = plane_of(h)
        rnd = dict(round=i, d_in_um=d, d_in_k=d_k, half_um=h, k=None if k is None else int(k))
        rnd.update(info or {})
        stop = rnd.pop("stop", None)
        rounds.append(rnd)
        if k is None:
            status = stop or "no_plane"
            break
        k = int(k)
        if k in fits:                           # the plane repeats: its fit is known
            rnd.update(fits[k], refit=False)
            status = "converged" if k == rounds[-2]["k"] else "cycle"
            break
        fits[k] = dict(fit_at(k))
        rnd.update(fits[k], refit=True)
        d, d_k = float(fits[k].get("d_hat_um", float("nan"))), k
        if not (math.isfinite(d) and d > 0):
            status = "fit_failed"
            break
    last = [r for r in rounds if r["k"] is not None]
    return dict(rounds=rounds, status=status, n_rounds=len(rounds), k_first=rounds[0]["k"],
                k_final=last[-1]["k"] if last else None,
                d_final_um=float(last[-1].get("d_hat_um", float("nan"))) if last else float("nan"))


def far_from_dendrites(shape, frame, segments, margin_um, discs=()):
    """(H, W) bool: True where a pixel centre is farther than max(r0, r1) +
    margin_um from every traced dendrite segment (the rows of
    survey.node_planes' segments: x0, y0, z0, x1, y1, z1, r0, r1, in_stretch)
    and farther than r + margin_um from every disc (x, y, r) (the soma)."""
    H, W = int(shape[0]), int(shape[1])
    p = float(frame.res_um_px)
    rows, cols = np.mgrid[0:H, 0:W]
    px, py = (frame.left + cols) * p, (frame.top + rows) * p
    near = np.zeros((H, W), dtype=bool)
    for s in np.atleast_2d(segments):
        if s.size < 8:
            continue
        near |= background.segment_distance(px, py, s[0:2], s[3:5]) <= max(s[6], s[7]) + margin_um
    for x, y, r in np.atleast_2d(np.asarray(discs, dtype=float)).reshape(-1, 3):
        near |= np.hypot(px - x, py - y) <= r + margin_um
    return ~near


def plane_background(block, valid, far, min_frac=0.05):
    """B_k (n,): the interquartile mean (25th-75th percentile) of each valid
    plane's pixels in `far`; NaN when fewer than min_frac of the pixels are far."""
    B = np.full(block.shape[0], np.nan)
    if far.mean() < min_frac:
        return B
    for k in range(block.shape[0]):
        if valid[k]:
            x = np.sort(block[k][far])
            lo, hi = int(np.floor(0.25 * x.size)), int(np.ceil(0.75 * x.size))
            B[k] = float(x[lo:max(hi, lo + 1)].mean())
    return B


def block_background(st, bg_margin_um=4.0):
    """Each plane's background B_k (the profile evaluation's, D-036): plane_background
    over the pilot block's pixels farther than Allen's radius + the margin from
    every traced dendrite segment and from the soma. The margin shrinks
    (bg_margin_um, then 3, 2, 1, 0.5 um below it) until 5 % of the block's
    pixels are far: next to the soma a wide margin leaves no pixel. Returns
    (B (n,), the far fraction, the margin used or None when no plane has a
    background)."""
    margins = [float(bg_margin_um)] + [x for x in (3.0, 2.0, 1.0, 0.5) if x < float(bg_margin_um)]
    for margin in margins:
        far = far_from_dendrites(st["block"].shape[1:], st["block_frame"], st["segments"], margin, st["soma"])
        if far.mean() >= 0.05:
            break
    B = plane_background(st["block"], st["valid"], far)
    return B, float(far.mean()), (margin if np.isfinite(B).any() else None)


def line_profiles(st, cfg, profile_half_um=None):
    """The node's measuring line (through the node, across its fitted heading,
    |v| <= h, h = profile_half_um or the measurement's) and the bilinear
    profile I_k(v) on it in every valid plane of node_stack's square (NaN rows
    for the invalid planes). Returns (h, v, y_hat, e_u, prof (n, M))."""
    m = cfg.measure
    h = float(m.profile_half_um if profile_half_um is None else profile_half_um)
    v = profiles.profile_offsets(dataclasses.replace(m, profile_half_um=h))
    th = st["theta"]
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    stack, valid, fr = st["stack"], st["valid"], st["frame"]
    prof = np.full((stack.shape[0], v.size), np.nan)
    for k in range(stack.shape[0]):
        if valid[k]:
            prof[k] = profiles.sample_profile(stack[k], fr, st["o"], y_hat, e_u, v)
    return h, v, y_hat, e_u, prof


def _argmax_plane(ks, A):
    """The plane of the largest finite value of A (the first of equal values); None without a finite value."""
    A = np.asarray(A, dtype=float)
    return None if not np.isfinite(A).any() else int(ks[int(np.nanargmax(A))])


def refit_at_plane(st, cfg, k, with_profile=False):
    """Block 5's final fit (fit.fit_profile, D-019.1) at plane k of
    node_stack's block (D-042), every other input the pilot's: the
    measurement's offsets (profiles.profile_offsets), the profile through the
    pilot's fitted centre (cx_um, cy_um) across its heading theta_rad, averaged
    over 2 A + 1 lines along the branch (profiles.sample_profile with
    profiles.n_along and profile_step_um), its tilt phi_rad, and B_bar =
    node_pipeline.node_background of plane k (the masked median of the node's
    block about the SWC node, D-018.1). At k = k* these are the pilot's own
    inputs, so d_hat is the pilot's. Returns dict(d_hat_um, fit_status,
    v0_hat_um, mu_hat_per_um, B_bar, bbar_ok); without a fit d_hat_um is NaN
    and fit_status says why: "no_plane" (k not among the block's planes, or
    missing), "profile_nan" (a sample outside the block), "bbar_nonpositive".
    with_profile=True (Cell 4k, D-045) also returns what drawing the fit
    needs: v_um (the offsets) and profile (the samples; both None without a
    plane), phi_rad, and the fit's alpha_hat and n_nodes (NaN and None
    without a fit), for tube_model.model_profile."""
    from allen_diameter.analysis import node_pipeline
    from allen_diameter.analysis.fit import fit_profile
    m, r, nan = cfg.measure, st["result"], float("nan")
    out = dict(d_hat_um=nan, fit_status="no_plane", v0_hat_um=nan, mu_hat_per_um=nan, B_bar=nan, bbar_ok=False)
    if with_profile:
        out.update(v_um=None, profile=None, phi_rad=float(r.phi_rad), alpha_hat=nan, n_nodes=None)
    ks = np.asarray(st["ks"])
    j = int(k) - int(ks[0])
    if not (0 <= j < ks.size and int(ks[j]) == int(k) and bool(st["valid"][j])):
        return out
    plane, frame = st["block"][j], st["block_frame"]
    B_bar, ok = node_pipeline.node_background(plane, frame, st["o"], st["branch"], m)
    th = float(r.theta_rad)
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = profiles.profile_offsets(m)
    I = profiles.sample_profile(plane, frame, np.array([float(r.cx_um), float(r.cy_um)]), y_hat, e_u, v,
                                profiles.n_along(m), m.profile_step_um)
    out.update(B_bar=float(B_bar), bbar_ok=bool(ok))
    if with_profile:
        out.update(v_um=v, profile=I)
    if not np.all(np.isfinite(I)):
        out["fit_status"] = "profile_nan"
    elif not B_bar > 0:
        out["fit_status"] = "bbar_nonpositive"
    else:
        fit = fit_profile(v, I, float(r.phi_rad), float(B_bar), m)
        out.update(d_hat_um=float(fit.d_hat_um), fit_status=str(fit.status), v0_hat_um=float(fit.v0_hat_um),
                   mu_hat_per_um=float(fit.mu_hat_per_um))
        if with_profile:
            out.update(alpha_hat=float(fit.alpha_hat), n_nodes=int(fit.n_nodes))
    return out


def gradient_iteration(st, cfg, d_start_um=None, line_mult=ITER_LINE_MULT, max_rounds=5, bg_margin_um=4.0,
                       restack=None, fit_at=None):
    """The Allen-guided iteration of the gradient energy (D-042, the user's
    choice of 2026-10-10): iterate_planes from d_0 = d_start_um (default
    Allen's diameter, 2 x allen_radius_um) and the multiplier line_mult
    (default ITER_LINE_MULT = 3, the lines +-1.5 d, D-044), with plane_of(h)
    the plane of the largest G over the whole line of half-length h (line_profiles and
    focus.plane_gradient_energies with block_background's B_k, the same in
    every round; the first of equal values; "line_short" for fewer than 3
    samples) and fit_at = refit_at_plane. A line reaching closer than 2 pixels
    to the square's edge makes restack(h) build a larger square (a node_stack
    dict of the same node and planes; on real data cut from the cached block
    when it lies inside it); without restack, or when the new square is still
    too small, the round stops with "square_small". fit_at(st, k), when given,
    replaces refit_at_plane(st, cfg, k) (the fit variants' own iterations,
    D-047; st the square in use). Returns iterate_planes' dict with
    background, bg_frac, bg_margin_used, st (the last square used),
    n_restacks, d_start_um. ValueError as iterate_planes, or when restack
    returns other planes."""
    d0 = 2.0 * float(st["allen_radius_um"]) if d_start_um is None else float(d_start_um)
    B, bg_frac, margin_used = block_background(st, bg_margin_um)
    step = float(cfg.measure.profile_step_um)
    box = dict(st=st, n_restacks=0)

    def fits_in(cur, h):
        return h + 2.0 * float(cur["frame"].res_um_px) <= float(cur["square_half_um"]) + 1e-9

    def plane_of(h):
        if int(round(h / step)) < 1:
            return None, dict(stop="line_short", n_samples=2 * int(round(h / step)) + 1)
        cur = box["st"]
        if not fits_in(cur, h):
            if restack is None:
                return None, dict(stop="square_small")
            cur = restack(h)
            if not np.array_equal(np.asarray(cur["ks"]), np.asarray(st["ks"])):
                raise ValueError("restack returned planes %r, not %r" % (list(cur["ks"]), list(st["ks"])))
            box["st"], box["n_restacks"] = cur, box["n_restacks"] + 1
            if not fits_in(cur, h):
                return None, dict(stop="square_small")
        _h, v, _y, _e, prof = line_profiles(cur, cfg, h)
        G = focus.plane_gradient_energies(prof, v, B, cfg.measure, cur["valid"])
        k = _argmax_plane(cur["ks"], G)
        info = dict(G=G, n_samples=int(v.size))
        if k is None:
            info["stop"] = "no_plane"
        return k, info

    if fit_at is None:
        def fit_at(cur, k):
            return refit_at_plane(cur, cfg, k)
    res = iterate_planes(d0, line_mult, plane_of, lambda k: fit_at(box["st"], k), max_rounds)
    res.update(background=B, bg_frac=bg_frac, bg_margin_used=margin_used, st=box["st"],
               n_restacks=box["n_restacks"], d_start_um=d0)
    return res
