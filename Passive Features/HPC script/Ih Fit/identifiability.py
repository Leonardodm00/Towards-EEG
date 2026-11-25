"""identifiability.py -- the identifiability table of a protocol from stored
sensitivities (SPEC section 2.10 (d), Block 26).

Pure module: numpy only, no NEURON, no file I/O, no global state.
`layout` is duck-typed on Block 23's ResidualLayout (masks, pre_masks,
sqrt_a, D, slices, n_residuals).

For a protocol (window, weights, training bundles) and stored sensitivities
S_b (n_t_b, p) (mV per unit of q_i):

    J[(b, s), i]   = sqrt(a_{b,s}) * S_{b,s,i} / D_b,        s in W_b
    r_eta[(b, s)]  = -sqrt(a_{b,s}) * eta_tilde_b(t_s) / D_b,
                     eta_tilde_b = eta_b - mean(eta_b over the pre-window)
    dq^(k)         = -(J^T J)^{-1} J^T r_eta^(k)   (least-squares solution)

The predicted standard deviation of q_hat_i is the sd of dq_i over k
(ddof 1); the predicted correlations are those of dq. First order in the
noise; the weights held fixed; the box ignored.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["jacobian_from_sensitivities", "svd_summary", "noise_residuals",
           "predicted_scatter", "relative_difference", "identifiability_row"]


def _layout_parts(layout):
    """The duck-typed layout's attributes, checked for consistency."""
    masks = [np.asarray(m, dtype=bool) for m in layout.masks]
    pre_masks = [np.asarray(m, dtype=bool) for m in layout.pre_masks]
    sqrt_a = [np.asarray(s, dtype=np.float64) for s in layout.sqrt_a]
    D = np.asarray(layout.D, dtype=np.float64)
    slices = list(layout.slices)
    n_res = int(layout.n_residuals)
    nb = len(masks)
    if not (len(pre_masks) == len(sqrt_a) == len(slices) == D.shape[0] == nb):
        raise ValueError("layout: masks, pre_masks, sqrt_a, D and slices differ in length "
                         "(%d, %d, %d, %d, %d)" % (nb, len(pre_masks), len(sqrt_a),
                                                   D.shape[0], len(slices)))
    for b in range(nb):
        n_w = int(np.count_nonzero(masks[b]))
        if sqrt_a[b].shape != (n_w,):
            raise ValueError("layout: bundle %d has %d window samples but sqrt_a of shape %s"
                             % (b, n_w, sqrt_a[b].shape))
        sl = slices[b]
        if len(range(*sl.indices(n_res))) != n_w:
            raise ValueError("layout: bundle %d's slice does not hold its %d window samples"
                             % (b, n_w))
        if pre_masks[b].shape != masks[b].shape:
            raise ValueError("layout: bundle %d's pre_mask and mask differ in shape" % b)
    return masks, pre_masks, sqrt_a, D, slices, n_res


def jacobian_from_sensitivities(S_list: Sequence[np.ndarray], layout) -> np.ndarray:
    """J (N, p) of SPEC 2.10 (d): sqrt_a[b][:, None] * S_list[b][masks[b]] / D[b],
    bundle b in the rows layout.slices[b]. S_list[b] float64 (n_t_b, p), mV per
    unit of q_i; the layout's bundles are S_list's, in order."""
    masks, _pre, sqrt_a, D, slices, n_res = _layout_parts(layout)
    if len(S_list) != len(masks):
        raise ValueError("S_list holds %d bundles, the layout %d" % (len(S_list), len(masks)))
    p = None
    J = None
    for b, S in enumerate(S_list):
        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] != masks[b].shape[0]:
            raise ValueError("S_list[%d] has shape %s; expected (%d, p)"
                             % (b, S.shape, masks[b].shape[0]))
        if p is None:
            p = S.shape[1]
            J = np.empty((n_res, p), dtype=np.float64)
        elif S.shape[1] != p:
            raise ValueError("S_list[%d] has %d columns, S_list[0] %d" % (b, S.shape[1], p))
        J[slices[b], :] = sqrt_a[b][:, None] * S[masks[b], :] / D[b]
    if J is None:
        raise ValueError("S_list is empty")
    return J


def svd_summary(J: np.ndarray) -> Dict[str, object]:
    """sigma (p,) descending; V (p, p), row j = v_{j+1}, each row's component of
    largest magnitude made positive (first such on a tie); ratio = sigma_p / sigma_1
    (nan when sigma_1 == 0)."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] < 1 or J.shape[0] < J.shape[1]:
        raise ValueError("J must be 2-D (N, p) with N >= p >= 1; got shape %s" % (J.shape,))
    if not np.all(np.isfinite(J)):
        raise ValueError("J holds non-finite values")
    _U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
    V = np.array(Vt, dtype=np.float64, copy=True)
    for j in range(V.shape[0]):
        k = int(np.argmax(np.abs(V[j])))
        if V[j, k] < 0.0:
            V[j] = -V[j]
    ratio = float(sigma[-1] / sigma[0]) if sigma[0] > 0.0 else float("nan")
    return {"sigma": np.asarray(sigma, dtype=np.float64), "V": V, "ratio": ratio}


def noise_residuals(layout, noise_list: Sequence[np.ndarray]) -> np.ndarray:
    """r_eta (N,) of SPEC 2.10 (d): per bundle, the noise (n_t_b,) (mV) minus its own
    mean over pre_masks[b], then -sqrt_a[b] * eta_tilde[masks[b]] / D[b]."""
    masks, pre_masks, sqrt_a, D, slices, n_res = _layout_parts(layout)
    if len(noise_list) != len(masks):
        raise ValueError("noise_list holds %d bundles, the layout %d"
                         % (len(noise_list), len(masks)))
    r = np.empty(n_res, dtype=np.float64)
    for b, eta in enumerate(noise_list):
        eta = np.asarray(eta, dtype=np.float64)
        if eta.shape != masks[b].shape:
            raise ValueError("noise_list[%d] has shape %s; expected %s"
                             % (b, eta.shape, masks[b].shape))
        if not np.any(pre_masks[b]):
            raise ValueError("bundle %d has no pre-window sample" % b)
        eta_t = eta - np.mean(eta[pre_masks[b]])
        r[slices[b]] = -sqrt_a[b] * eta_t[masks[b]] / D[b]
    if not np.all(np.isfinite(r)):
        raise ValueError("r_eta holds non-finite values (non-finite noise in a used sample)")
    return r


def predicted_scatter(J: np.ndarray, R_eta: np.ndarray) -> Dict[str, np.ndarray]:
    """dq (K, p) = the least-squares solutions of J dq = -r_eta^(k)
    (numpy.linalg.lstsq(J, -R_eta.T)); sd (p,) with ddof 1; corr (p, p)."""
    J = np.asarray(J, dtype=np.float64)
    R_eta = np.asarray(R_eta, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] < 1:
        raise ValueError("J must be 2-D (N, p); got shape %s" % (J.shape,))
    if R_eta.ndim != 2 or R_eta.shape[1] != J.shape[0]:
        raise ValueError("R_eta must be (K, N) with N = %d; got shape %s"
                         % (J.shape[0], R_eta.shape))
    if R_eta.shape[0] < 2:
        raise ValueError("K = %d noise realisations; at least 2 are needed" % R_eta.shape[0])
    if not np.all(np.isfinite(J)):
        raise ValueError("J holds non-finite values")
    if not np.all(np.isfinite(R_eta)):
        raise ValueError("R_eta holds non-finite values")
    sol, _res, rank, _sv = np.linalg.lstsq(J, -R_eta.T, rcond=None)
    p = J.shape[1]
    if rank < p:
        raise ValueError("J has rank %d below p = %d: the linearised scatter is undefined"
                         % (rank, p))
    dq = np.ascontiguousarray(sol.T, dtype=np.float64)
    sd = np.std(dq, axis=0, ddof=1)
    if p == 1:
        corr = np.ones((1, 1), dtype=np.float64)
    else:
        corr = np.atleast_2d(np.corrcoef(dq, rowvar=False))
    return {"dq": dq, "sd": sd, "corr": corr}


def relative_difference(J_a: np.ndarray, J_b: np.ndarray) -> float:
    """||J_a - J_b||_F / ||J_b||_F."""
    J_a = np.asarray(J_a, dtype=np.float64)
    J_b = np.asarray(J_b, dtype=np.float64)
    if J_a.shape != J_b.shape:
        raise ValueError("J_a %s and J_b %s differ in shape" % (J_a.shape, J_b.shape))
    if not (np.all(np.isfinite(J_a)) and np.all(np.isfinite(J_b))):
        raise ValueError("non-finite values in J_a or J_b")
    nb = float(np.linalg.norm(J_b))
    if nb == 0.0:
        raise ValueError("||J_b||_F = 0: the relative difference is undefined")
    return float(np.linalg.norm(J_a - J_b) / nb)


def identifiability_row(J: np.ndarray, names: Sequence[str],
                        R_eta: Optional[np.ndarray] = None) -> Dict[str, float]:
    """One table row: sigma_1..sigma_p, sigma_ratio, v_p_<name> (the right singular
    vector of the smallest singular value), v_pm1_<name> (the next one; only when
    p >= 2), and with R_eta: sd_lin_<name>, corr_<a>_<b> for every pair a < b in
    names' order. Values are Python floats."""
    J = np.asarray(J, dtype=np.float64)
    names = [str(n) for n in names]
    if J.ndim != 2 or len(names) != J.shape[1]:
        raise ValueError("names holds %d entries, J has shape %s" % (len(names), J.shape))
    if len(set(names)) != len(names):
        raise ValueError("names must be distinct")
    p = J.shape[1]
    svd = svd_summary(J)
    row: Dict[str, float] = {}
    for j in range(p):
        row["sigma_%d" % (j + 1)] = float(svd["sigma"][j])
    row["sigma_ratio"] = float(svd["ratio"])
    for i, n in enumerate(names):
        row["v_p_%s" % n] = float(svd["V"][p - 1, i])
    if p >= 2:
        for i, n in enumerate(names):
            row["v_pm1_%s" % n] = float(svd["V"][p - 2, i])
    if R_eta is not None:
        sc = predicted_scatter(J, R_eta)
        for i, n in enumerate(names):
            row["sd_lin_%s" % n] = float(sc["sd"][i])
        for i in range(p):
            for k in range(i + 1, p):
                row["corr_%s_%s" % (names[i], names[k])] = float(sc["corr"][i, k])
    return row
