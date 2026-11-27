"""The twins of step 2: settings and real measurements per chosen node, and the training/validation split --
Block 53 in specs/SPEC.md (section 5, "Kernel calibration, step 2"). No file access, no rendering.

Contract of wave 9 (2026-11-27; D-261.5, D-274.1, D-274.4), with its amendment 1 (the pilot's B_bar, d_hat_um and
alpha_hat held; the scan's k_star and z_sub_um checked against the pilot's; the split's capped seats passed on). Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

SETS = ("near", "far")
ROLES = ("train", "validation")

# The pilot fields a twin holds (besides k_star): each must be a finite number (D-274.1; SPEC Block 53 and its
# amendment 1); those in _PILOT_POSITIVE must also be > 0.
_PILOT_FLOAT_FIELDS = ("mu_hat_per_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "z_sub_um", "alpha_hat", "B_bar",
                       "d_hat_um")
_PILOT_POSITIVE = ("mu_hat_per_um", "B_bar", "d_hat_um")
# A scan's z_sub_um and its pilot's are one number written twice: they agree to this (um; SPEC amendment 1).
_Z_SUB_TOL_UM = 1e-9
# Added to f N + 1/2 before the floor of V_s (SPEC Block 53, amendment 3).
_HALF_GUARD = 1e-9


@dataclass(frozen=True)
class Twin:
    """One chosen calibration node: the held settings of its twin and the real scan's measurements. The arrays
    are (n,) over the real scan's planes, read-only. Units: um, 1/um, rad, grey levels (gl), um^2, gl x um."""

    cell_id: int
    node_id: int
    set: str                    # "near" | "far"
    role: str                   # "train" | "validation"
    d_um: float                 # Allen's diameter, taken as true (D-261.5)
    mu_per_um: float            # the pilot's absorption coefficient, held (D-274.1)
    k_star: int                 # the pilot's sharpest plane
    z_sub_um: float             # the pilot's sub-plane depth: the start of the twin's axis depth
    cx_um: float                # the pilot's lateral centre (image frame)
    cy_um: float
    theta_rad: float            # heading
    phi_rad: float              # tilt
    B_pilot_gl: float           # the pilot's B_bar (gl): read by the scan's stripe support (amendment 1)
    d_pilot_um: float           # the pilot's d_hat_um (um): the scan's own-cylinder diameter (amendment 1)
    alpha_pilot: float          # the pilot's alpha_hat (amendment 1)
    ks: np.ndarray              # (n,) int64 plane indices of the real scan
    B_gl: np.ndarray            # (n,) the real plane's background at the node (PlaneScan.B_bar)
    noise_sd_gl: np.ndarray     # (n,) the real plane's background SD (PlaneScan.bg_sd_gl)
    omega_um2: np.ndarray       # (n,) the real squared core width (NaN unless status ok)
    se_omega_um2: np.ndarray    # (n,) its SE from the plane's own fit
    area_gl_um: np.ndarray      # (n,) the real dip's windowed area
    status: Tuple[str, ...]     # (n,) the real scan's status per plane


# ------------------------------------------------------------- helpers ---

def _finite_number(x):
    """True for a real number (not bool, not None) that is finite."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _ro(a, dtype):
    """A read-only (n,) copy of a, in dtype (the copy leaves the scan's own array writeable)."""
    out = np.array(a, dtype=dtype, copy=True).reshape(-1)
    out.setflags(write=False)
    return out


def _twin_problems(scan, rec):
    """The reasons a chosen node cannot make a twin (empty: it can). scan: PlaneScan or None; rec: node record."""
    if scan is None or rec is None:
        return (["no scan"] if scan is None else []) + (["no record"] if rec is None else [])
    why = []
    rec_set = rec.get("set")
    if rec_set != scan.set:
        why.append("record set %r differs from the scan's set %r" % (rec_set, scan.set))
    if scan.set not in SETS:
        why.append("scan set %r not in %s" % (scan.set, SETS))
    d = rec.get("allen_diameter_um")
    if not (_finite_number(d) and float(d) > 0.0):
        why.append("allen_diameter_um %r not finite and > 0" % (d,))
    pilot = rec.get("pilot")
    if not isinstance(pilot, dict):
        why.append("no pilot")
    else:
        for name in _PILOT_FLOAT_FIELDS:
            x = pilot.get(name)
            if not _finite_number(x):
                why.append("pilot %s %r not finite" % (name, x))
            elif name in _PILOT_POSITIVE and not float(x) > 0.0:
                why.append("pilot %s %r not > 0" % (name, x))
        k = pilot.get("k_star")
        if not _finite_number(k) or float(k) != int(float(k)):
            why.append("pilot k_star %r missing or not an integer" % (k,))
        elif int(float(k)) != int(scan.k_star):
            why.append("scan k_star %d differs from the pilot's %d" % (int(scan.k_star), int(float(k))))
        z = pilot.get("z_sub_um")
        if _finite_number(z) and not abs(float(scan.z_sub_um) - float(z)) <= _Z_SUB_TOL_UM:
            why.append("scan z_sub_um %r differs from the pilot's %r by more than %g um"
                       % (scan.z_sub_um, z, _Z_SUB_TOL_UM))
    if scan.bg_sd_gl is None:
        why.append("scan without bg_sd_gl")
    # The (n,) arrays must agree with ks (the contract's shapes); a malformed scan is refused, not truncated.
    n = np.asarray(scan.ks).reshape(-1).size
    arrays = dict(B_bar=scan.B_bar, bg_sd_gl=scan.bg_sd_gl, omega_um2=scan.omega_um2,
                  se_omega_um2=scan.se_omega_um2, area_gl_um=scan.area_gl_um)
    for name, a in arrays.items():
        if a is not None and np.asarray(a).size != n:
            why.append("scan %s of size %d, ks of size %d" % (name, np.asarray(a).size, n))
    if len(tuple(scan.status)) != n:
        why.append("scan status of length %d, ks of size %d" % (len(tuple(scan.status)), n))
    return why


# ---------------------------------------------------------------- API ---

def build_twins(cell_id, scans, nodes, chosen_ids):
    """list[Twin] in the order of chosen_ids, role "train": scans a sequence of PlaneScan, nodes {node id: node
    record} (Block 52's CellScans). ValueError naming every offending node id: no scan, no record, a record whose
    set differs from the scan's, a non-finite or non-positive allen_diameter_um, a pilot without finite
    mu_hat_per_um > 0, cx_um, cy_um, theta_rad, phi_rad, z_sub_um, alpha_hat, finite B_bar > 0 and d_hat_um > 0, or
    without k_star; a scan without bg_sd_gl; a scan whose k_star differs from the pilot's or whose z_sub_um differs
    from the pilot's by more than 1e-9 um (amendment 1)."""
    by_id = {}
    dup_scan = set()
    for s in scans:
        nid = int(s.node_id)
        if nid in by_id:
            dup_scan.add(nid)
        by_id[nid] = s
    recs = {int(k): v for k, v in dict(nodes).items()}
    chosen = [int(i) for i in chosen_ids]

    problems = {}
    seen = set()
    for nid in chosen:
        why = []
        if nid in seen:
            why.append("chosen twice")
        seen.add(nid)
        if nid in dup_scan:
            why.append("two scans with this node id")
        why += _twin_problems(by_id.get(nid), recs.get(nid))
        if why:
            problems.setdefault(nid, [])
            for w in why:
                if w not in problems[nid]:
                    problems[nid].append(w)
    if problems:
        lines = ["node %d: %s" % (nid, "; ".join(why)) for nid, why in problems.items()]
        raise ValueError("cell %d: %d chosen node(s) cannot make a twin: %s"
                         % (int(cell_id), len(problems), " | ".join(lines)))

    twins = []
    for nid in chosen:
        scan, rec = by_id[nid], recs[nid]
        pilot = rec["pilot"]
        n = np.asarray(scan.ks).reshape(-1).size
        se = scan.se_omega_um2 if scan.se_omega_um2 is not None else np.full(n, np.nan)
        twins.append(Twin(
            cell_id=int(cell_id), node_id=nid, set=str(scan.set), role="train",
            d_um=float(rec["allen_diameter_um"]), mu_per_um=float(pilot["mu_hat_per_um"]),
            k_star=int(float(pilot["k_star"])), z_sub_um=float(pilot["z_sub_um"]),
            cx_um=float(pilot["cx_um"]), cy_um=float(pilot["cy_um"]),
            theta_rad=float(pilot["theta_rad"]), phi_rad=float(pilot["phi_rad"]),
            B_pilot_gl=float(pilot["B_bar"]), d_pilot_um=float(pilot["d_hat_um"]),
            alpha_pilot=float(pilot["alpha_hat"]),
            ks=_ro(scan.ks, np.int64), B_gl=_ro(scan.B_bar, np.float64),
            noise_sd_gl=_ro(scan.bg_sd_gl, np.float64), omega_um2=_ro(scan.omega_um2, np.float64),
            se_omega_um2=_ro(se, np.float64), area_gl_um=_ro(scan.area_gl_um, np.float64),
            status=tuple(str(x) for x in scan.status)))
    return twins


# Custom: the largest-remainder (Hamilton) allocation with a per-cell cap of n - 1; no library function
# fits the caps (SPEC Block 53 allows it). Checked: numpy, scipy (no apportionment routine).
def _allocate(V, counts):
    """{cell_id: V_c}: V seats shared among the cells in proportion to counts {cell_id: n_c} by largest
    remainders, ties to the lower cell_id, each V_c <= n_c - 1. Exact integer arithmetic: the quota of cell c
    is V n_c / N, its floor (V n_c) // N and its remainder (V n_c) % N (over N).
    Placement of the seats left after the floors (SPEC amendment 1): they are offered to the cells in the order (remainder descending, cell_id ascending), one
    seat per cell per pass, a cell at its cap skipped; passes repeat until every seat is placed or every cell is
    at its cap; what is left then is dropped, so sum_c V_c = min(V, sum_c (n_c - 1))."""
    N = sum(counts.values())
    if V <= 0 or N <= 0:
        return {c: 0 for c in counts}
    cap = {c: max(n - 1, 0) for c, n in counts.items()}
    alloc = {c: min((V * n) // N, cap[c]) for c, n in counts.items()}
    left = V - sum(alloc.values())
    order = sorted(counts, key=lambda c: (-((V * counts[c]) % N), c))
    while left > 0:
        placed = False
        for c in order:
            if left == 0:
                break
            if alloc[c] < cap[c]:
                alloc[c] += 1
                left -= 1
                placed = True
        if not placed:
            break       # every cell at its cap: the rest is dropped
    return alloc


def split_twins(twins, validation_fraction, seed):
    """The twins in the same order with their roles set (D-274.4): by node, stratified by set and by cell, the
    rule of SPEC Block 53 (largest remainders, a cell never emptied of training twins, a seat a capped cell cannot
    take passed on in repeated passes, a seeded permutation of the sorted node ids). Independent of the input order. ValueError: a fraction outside [0, 1); two twins with one
    (cell_id, node_id)."""
    twins = list(twins)
    f = validation_fraction
    if not (_finite_number(f) and 0.0 <= float(f) < 1.0):
        raise ValueError("validation_fraction %r outside [0, 1)" % (f,))
    f = float(f)
    seen, dup, bad_set = set(), [], []
    for t in twins:
        key = (int(t.cell_id), int(t.node_id))
        if key in seen:
            dup.append(key)
        seen.add(key)
        if t.set not in SETS:
            bad_set.append((key, t.set))
    if dup:
        raise ValueError("two twins with one (cell_id, node_id): %s" % (sorted(set(dup)),))
    if bad_set:
        raise ValueError("twins with a set not in %s: %s" % (SETS, bad_set))

    validation = set()
    for s_index, s in enumerate(SETS):
        ids = {}
        for t in twins:
            if t.set == s:
                ids.setdefault(int(t.cell_id), []).append(int(t.node_id))
        N = sum(len(v) for v in ids.values())
        # floor(f N + 1/2), with 1e-9 so that an exact half is not lost to floating point (amendment 3:
        # 0.7 * 45 + 0.5 = 31.999999999999996 in float64)
        V = int(math.floor(f * N + 0.5 + _HALF_GUARD))
        alloc = _allocate(V, {c: len(v) for c, v in ids.items()})
        for c, nodes_c in ids.items():
            if alloc[c] == 0:
                continue
            perm = np.random.default_rng([int(seed), c, s_index]).permutation(np.array(sorted(nodes_c),
                                                                                       dtype=np.int64))
            for nid in perm[:alloc[c]]:
                validation.add((c, int(nid)))
    return [dataclasses.replace(t, role="validation" if (int(t.cell_id), int(t.node_id)) in validation
                                else "train") for t in twins]


def fit_planes(twin, s2cfg, which):
    """(n,) bool over twin.ks: which "near": |k - k*| <= s2cfg.near_planes; "far": far_planes[0] <= |k - k*| <=
    far_planes[1]; in both, status "ok" and a finite omega_um2. ValueError for another which."""
    j = np.abs(np.asarray(twin.ks, dtype=np.int64) - int(twin.k_star))
    if which == "near":
        band = j <= int(s2cfg.near_planes)
    elif which == "far":
        lo, hi = s2cfg.far_planes
        band = (j >= int(lo)) & (j <= int(hi))
    else:
        raise ValueError("which %r not 'near' or 'far'" % (which,))
    ok = np.array([s == "ok" for s in twin.status], dtype=bool)
    return band & ok & np.isfinite(np.asarray(twin.omega_um2, dtype=np.float64))
