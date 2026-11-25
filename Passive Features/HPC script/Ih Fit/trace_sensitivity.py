"""trace_sensitivity.py -- the stored sensitivities of a cell at a point
(SPEC section 2.10 (d), Block 26).

No NEURON import: the model enters through an evaluator (Block 23's
build_trace_evaluator, or any callable of the same form): evaluator(Q), Q
float64 (m, p) in optimiser coordinates q, returns a list of m entries, each
the list of the bundles' traces (float64 (n_t_b,), mV, minus their own
pre-window mean) or None when the point failed.

S_{b,s,i}(q) = d V_hat_b(t_s; q) / d q_i (mV per unit of q_i) over every
recorded sample of each bundle, by SPEC 2.10 (b) (default: central, absolute
step h = 1e-3 in unit coordinates), in ONE fd_jacobian call whose single
evaluator call also carries the base point q.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import fd_jacobian as FJ

__all__ = ["compute_sensitivities", "save_sensitivities", "load_sensitivities"]


def _point_traces(entry, k, n_bundles=None, lengths=None):
    """One evaluator entry checked: a list of 1-D finite float64 arrays."""
    if entry is None:
        raise RuntimeError("the evaluator returned None for point %d (the model could not "
                           "be simulated there)" % k)
    traces = [np.asarray(x, dtype=np.float64) for x in entry]
    if n_bundles is not None and len(traces) != n_bundles:
        raise RuntimeError("point %d returned %d bundles, the base point %d"
                           % (k, len(traces), n_bundles))
    for b, x in enumerate(traces):
        if x.ndim != 1:
            raise RuntimeError("point %d, bundle %d: trace of shape %s is not 1-D"
                               % (k, b, x.shape))
        if lengths is not None and x.shape[0] != lengths[b]:
            raise RuntimeError("point %d, bundle %d: %d samples, the base point %d"
                               % (k, b, x.shape[0], lengths[b]))
    return traces


def compute_sensitivities(evaluator: Callable, q, bounds: Sequence[Tuple[float, float]], *,
                          h: float = 1e-3, scheme: str = "central") -> Dict[str, object]:
    """q (p,), h, scheme, traces0 (list of (n_t_b,), mV), S (list of (n_t_b, p),
    mV per unit of q_i), n_points (rows asked of the evaluator).

    RuntimeError when the evaluator returns None for any point (or a malformed
    entry). ValueError (from fd_jacobian) when q lies outside the box, h is not
    in (0, 0.5], the scheme is unknown, or a trace is non-finite."""
    q = np.array(q, dtype=np.float64).reshape(-1)
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    if len(bounds) != q.shape[0]:
        raise ValueError("q has %d entries, bounds %d" % (q.shape[0], len(bounds)))
    if not np.all(np.isfinite(q)):
        raise ValueError("q holds non-finite values")
    u = np.asarray(FJ.to_unit(q, bounds), dtype=np.float64).reshape(-1)
    state = {"traces0": None, "lengths": None, "n_points": 0}

    def batch_fun(U):
        U = np.asarray(U, dtype=np.float64)
        # The base point goes into the same evaluator call: prepended unless
        # fd_jacobian already put u as the first row (no perturbed row equals u).
        has_base = U.shape[0] > 0 and np.array_equal(U[0], u)
        Q = np.asarray(FJ.to_q(U, bounds), dtype=np.float64).reshape(U.shape[0], -1)
        if has_base:
            Q[0] = q
        else:
            Q = np.vstack([q[None, :], Q])
        out = evaluator(Q)
        state["n_points"] += Q.shape[0]
        if out is None or len(out) != Q.shape[0]:
            raise RuntimeError("the evaluator returned %s entries for %d points"
                               % ("no" if out is None else len(out), Q.shape[0]))
        base = _point_traces(out[0], 0)
        lengths = [x.shape[0] for x in base]
        rows = [np.concatenate(base)]
        for k in range(1, Q.shape[0]):
            tr = _point_traces(out[k], k, len(base), lengths)
            rows.append(np.concatenate(tr))
        state["traces0"] = [np.array(x, dtype=np.float64, copy=True) for x in base]
        state["lengths"] = lengths
        F = np.vstack(rows)
        return F if has_base else F[1:]

    res = FJ.fd_jacobian(batch_fun, u, h=h, scheme=scheme)
    if state["traces0"] is None:
        raise RuntimeError("fd_jacobian did not call the evaluator")
    J_q = np.asarray(FJ.jac_unit_to_q(res.J, bounds), dtype=np.float64)
    S: List[np.ndarray] = []
    start = 0
    for n_t in state["lengths"]:
        S.append(np.ascontiguousarray(J_q[start:start + n_t, :]))
        start += n_t
    return {"q": q, "h": float(h), "scheme": str(scheme), "traces0": state["traces0"],
            "S": S, "n_points": int(state["n_points"])}


def save_sensitivities(path, sens: Dict[str, object], *, t_list: Sequence[np.ndarray],
                       names: Sequence[str], meta: Optional[dict] = None) -> None:
    """One .npz (np.savez, no pickle; numpy appends '.npz' to a path without it)
    holding q, h, scheme, names, per bundle t_<b>, V0_<b>, S_<b>, and meta as a
    JSON string (json.dumps; None is stored as 'null')."""
    q = np.asarray(sens["q"], dtype=np.float64).reshape(-1)
    traces0 = [np.asarray(x, dtype=np.float64) for x in sens["traces0"]]
    S = [np.asarray(x, dtype=np.float64) for x in sens["S"]]
    t_list = [np.asarray(t, dtype=np.float64) for t in t_list]
    names = [str(n) for n in names]
    p = q.shape[0]
    if not (len(traces0) == len(S) == len(t_list)):
        raise ValueError("traces0, S and t_list hold %d, %d and %d bundles"
                         % (len(traces0), len(S), len(t_list)))
    if len(names) != p:
        raise ValueError("names holds %d entries, q %d" % (len(names), p))
    for b in range(len(S)):
        n_t = traces0[b].shape[0]
        if traces0[b].ndim != 1 or t_list[b].shape != (n_t,) or S[b].shape != (n_t, p):
            raise ValueError("bundle %d: t %s, V0 %s, S %s are not (n_t,), (n_t,), (n_t, %d)"
                             % (b, t_list[b].shape, traces0[b].shape, S[b].shape, p))
    arrays = {"q": q, "h": np.float64(sens["h"]), "scheme": np.array(str(sens["scheme"])),
              "names": np.array(names, dtype=np.str_),
              "meta": np.array(json.dumps(meta))}
    for b in range(len(S)):
        arrays["t_%d" % b] = t_list[b]
        arrays["V0_%d" % b] = traces0[b]
        arrays["S_%d" % b] = S[b]
    np.savez(path, **arrays)


def load_sensitivities(path) -> Dict[str, object]:
    """The dict save_sensitivities wrote: q, h, scheme, traces0, S, t_list,
    names, meta (read with allow_pickle=False)."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        n_b = 0
        while ("t_%d" % n_b) in keys:
            n_b += 1
        return {"q": np.array(z["q"], dtype=np.float64),
                "h": float(z["h"]),
                "scheme": str(z["scheme"]),
                "traces0": [np.array(z["V0_%d" % b]) for b in range(n_b)],
                "S": [np.array(z["S_%d" % b]) for b in range(n_b)],
                "t_list": [np.array(z["t_%d" % b]) for b in range(n_b)],
                "names": [str(n) for n in z["names"]],
                "meta": json.loads(str(z["meta"]))}
