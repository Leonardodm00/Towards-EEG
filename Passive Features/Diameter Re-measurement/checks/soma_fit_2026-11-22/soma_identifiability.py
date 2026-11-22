"""Can an ellipsoid soma's height be recovered from a brightfield stack? (2026-11-22; D-216.2) -- a Fisher check.

Forward model -- the pipeline's partition absorption (procedure Eqs. 5-6; geometry.slab_absorbance and
transmitted_before) with the pilot's calibrated kernel table (cleaning_config.json, Cell 7) and its configured
continuation (linear, slope 0.79 um/um):
  the ellipsoid E(theta): centre (x0, y0, z0), in-plane semi-axes a, b at angle phi, depth semi-axis c, absorption mu.
  Column (x, y): u, v its coordinates rotated by phi about (x0, y0); q = 1 - (u/a)^2 - (v/b)^2; inside when q > 0, on
  [z0 - c sqrt(q), z0 + c sqrt(q)]. Slab j (centre zeta_j, thickness dzeta): a_j = mu |interval cap slab|,
  T_<j = exp(-mu * path before the slab) (light toward +z), Delta A_j = T_<j (1 - exp(-a_j)).
  tau_k = 1 - sum_j G[sigma_r(zeta_j - z_k)] (*) Delta A_j (FFT, the Gaussian transfer per (slab, plane)); I_k = B_k tau_k.
Simplifications (a check, not pipeline code): the pixel grid itself (0.1144 um), the kernel floored at half a pixel,
slabs of 0.14 um, planes every 0.56 um within +-10 um of z0, a periodic FFT box of 512 px.
Fisher information at the truth: J = dI/d(x0, y0, z0, a, b, c, phi, mu, B_1..B_K) over the fit region's pixels (a disk
of 13 um about the centre, minus a stem's mask where given), noise SD sigma_n: Cov = sigma_n^2 (J^T J)^-1.
Linearised biases: data rendered with the kernel's slope x 1.2 (or with an apical stem leaking outside its mask) and
fitted with the configured model: d theta = (J^T J)^-1 J^T (I_data - I_model) at the truth.
Membrane area S(a, b, c): the ellipsoid's surface, numerical integral; its SE by the delta method.

    python3 soma_identifiability.py <workstream folder> <cleaning_config.json>
Pure ASCII.
"""
import json
import math
import sys
import time

import numpy as np
from scipy import integrate

sys.path.insert(0, sys.argv[1] + "/src")
from allen_diameter.config import config_from_dict, default_config   # noqa: E402
from allen_diameter.model import geometry                            # noqa: E402
from allen_diameter.model.kernel import sigma_r                       # noqa: E402

CFG = dict(res=0.1144, nfft=512, half_crop_um=14.0, dzeta=0.14, dz=0.28, plane_step=2, plane_half_um=10.0,
           B=110.0, noise=3.0, a=6.6, b=5.8, phi_deg=20.0, tau_centre=0.5, fit_radius_um=13.0,
           cs=(3.5, 5.0, 6.5, 8.0), slope_error=1.2, stem_r_um=1.3, stem_mu=0.258, stem_mask_extra_um=2.0,
           floor_px=0.5)


def kernel_cfg(path, slope_factor=1.0):
    d = json.load(open(path))
    base = default_config()
    import dataclasses
    rd = dataclasses.asdict(base.renderer)
    rd.update(d["renderer"])
    rd["kernel_continuation_slope"] = rd["kernel_continuation_slope"] * slope_factor
    full = dataclasses.asdict(base)
    full["renderer"] = rd
    return config_from_dict(full).renderer


class Box:
    def __init__(self, c):
        n, p = c["nfft"], c["res"]
        x = (np.arange(n) - n // 2) * p
        self.X, self.Y = np.meshgrid(x, x)
        fx = np.fft.fftfreq(n, d=p)
        fy = np.fft.rfftfreq(n, d=p)
        self.f2 = (fx[:, None] ** 2 + fy[None, :] ** 2).T if False else (np.fft.fftfreq(n, d=p)[:, None] ** 2
                                                                          + np.fft.rfftfreq(n, d=p)[None, :] ** 2)
        self.n, self.p = n, p


def column_interval(box, th, stem=None):
    x0, y0, z0, a, b, c, phi, mu = th
    u = (box.X - x0) * math.cos(phi) + (box.Y - y0) * math.sin(phi)
    v = -(box.X - x0) * math.sin(phi) + (box.Y - y0) * math.cos(phi)
    q = 1.0 - (u / a) ** 2 - (v / b) ** 2
    inside = q > 0
    h = c * np.sqrt(np.clip(q, 0.0, None))
    lo, hi = z0 - h, z0 + h
    MU = np.where(inside, mu, 0.0)
    if stem is not None:                         # an apical stem along -y from the soma's top, at the soma's depth
        rs, mus = stem
        top = y0 - b
        w2 = rs * rs - (box.X - x0) ** 2
        s_in = (w2 > 0) & (box.Y < top) & ~inside
        hs = np.sqrt(np.clip(w2, 0.0, None))
        lo = np.where(s_in, z0 - hs, lo)
        hi = np.where(s_in, z0 + hs, hi)
        MU = np.where(s_in, mus, MU)
        inside = inside | s_in
    return lo, hi, inside, MU


def render(box, th, zk, rcfg, c, stem=None):
    lo, hi, inside, MU = column_interval(box, th, stem)
    z0, cc = th[2], th[5]
    zmin = min(float(lo[inside].min()), z0 - cc)
    zmax = max(float(hi[inside].max()), z0 + cc)
    zeta = geometry.slab_centres(zmin, zmax, c["dzeta"])
    F = []
    for zj in zeta:
        aj = geometry.slab_absorbance(lo, hi, zj, c["dzeta"], 1.0) * MU
        T = np.exp(-MU * np.clip(np.minimum(hi, zj - 0.5 * c["dzeta"]) - lo, 0.0, None))
        dA = np.where(inside, T * (-np.expm1(-aj)), 0.0)
        F.append(np.fft.rfft2(dA))
    F = np.array(F)
    floor = c["floor_px"] * c["res"]
    out = []
    for z in zk:
        s = np.maximum(sigma_r(zeta - z, rcfg), floor)
        acc = np.einsum("j,jxy->xy", np.ones(len(zeta)), F * np.exp(-2.0 * math.pi ** 2 * s[:, None, None] ** 2 * box.f2[None]))
        out.append(1.0 - np.fft.irfft2(acc, s=(box.n, box.n)))
    return np.array(out)


def ellipsoid_area(a, b, c):
    f = lambda t, ph: (math.sin(t) * math.sqrt((b * c * math.sin(t) * math.cos(ph)) ** 2 + (a * c * math.sin(t) *
                       math.sin(ph)) ** 2 + (a * b * math.cos(t)) ** 2))
    return integrate.dblquad(lambda t, ph: f(t, ph), 0, 2 * math.pi, 0, math.pi, epsabs=1e-6)[0]


def analyse(cc, rcfg, rcfg_slope, c, with_stem):
    box = Box(c)
    mu = -math.log(c["tau_centre"]) / (2.0 * cc)
    th = np.array([0.0, 0.0, 0.0, c["a"], c["b"], cc, math.radians(c["phi_deg"]), mu])
    zk = np.arange(-c["plane_half_um"], c["plane_half_um"] + 1e-9, c["dz"] * c["plane_step"])
    region = (box.X ** 2 + box.Y ** 2) <= c["fit_radius_um"] ** 2
    stem = (c["stem_r_um"], c["stem_mu"]) if with_stem else None
    if with_stem:
        region &= ~((np.abs(box.X) < c["stem_r_um"] + c["stem_mask_extra_um"]) & (box.Y < -c["b"] + 1.0))
    t0 = time.time()
    tau = render(box, th, zk, rcfg, c)
    steps = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.004, mu * 1e-3])
    D = []
    for i in range(8):
        t2 = th.copy()
        t2[i] += steps[i]
        D.append((render(box, t2, zk, rcfg, c) - tau) / steps[i] * c["B"])
    D = np.array(D)                                   # (8, K, n, n): dI/dtheta
    K = len(zk)
    P = 8 + K
    A = np.zeros((P, P))
    for k in range(K):
        Dk = D[:, k][:, region].T                      # (npix, 8)
        tk = tau[k][region]
        A[:8, :8] += Dk.T @ Dk
        A[:8, 8 + k] = Dk.T @ tk
        A[8 + k, :8] = A[:8, 8 + k]
        A[8 + k, 8 + k] = tk @ tk
    cov = c["noise"] ** 2 * np.linalg.inv(A)
    se = np.sqrt(np.diag(cov)[:8])
    corr = cov[:8, :8] / np.outer(se, se)

    def bias(I_data):
        g = np.zeros(P)
        for k in range(K):
            rk = (I_data[k] - c["B"] * tau[k])[region]
            g[:8] += D[:, k][:, region] @ rk
            g[8 + k] = tau[k][region] @ rk
        return np.linalg.solve(A, g)[:8]

    b_slope = bias(c["B"] * render(box, th, zk, rcfg_slope, c))
    b_stem = bias(c["B"] * render(box, th, zk, rcfg, c, stem=stem)) if with_stem else np.full(8, np.nan)
    S = ellipsoid_area(c["a"], c["b"], cc)
    gS = np.array([(ellipsoid_area(c["a"] + 1e-3, c["b"], cc) - S) / 1e-3,
                   (ellipsoid_area(c["a"], c["b"] + 1e-3, cc) - S) / 1e-3,
                   (ellipsoid_area(c["a"], c["b"], cc + 1e-3) - S) / 1e-3])
    idx = [3, 4, 5]
    se_S = math.sqrt(gS @ cov[np.ix_(idx, idx)] @ gS)
    return dict(c=cc, mu=mu, se=se, corr=corr, b_slope=b_slope, b_stem=b_stem, S=S, se_S=se_S,
                bS_slope=float(gS @ b_slope[idx]), bS_stem=float(gS @ b_stem[idx]) if with_stem else float("nan"),
                planes=K, seconds=time.time() - t0)


if __name__ == "__main__":
    rcfg = kernel_cfg(sys.argv[2])
    rcfg_s = kernel_cfg(sys.argv[2], CFG["slope_error"])
    print("kernel: table %s um at %s um, continuation %s %.2f um/um; noise %.1f gl on B %.0f gl; a %.1f, b %.1f um"
          % (list(rcfg.kernel_table_sigma_um), list(rcfg.kernel_table_delta_um), rcfg.kernel_continuation,
             rcfg.kernel_continuation_slope, CFG["noise"], CFG["B"], CFG["a"], CFG["b"]))
    print("  c true | SE(c)  corr(c,mu) corr(c,z0) | bias(c): slope x%.1f, stem | S true  SE(S)  bias(S): slope, stem"
          % CFG["slope_error"])
    res = []
    for cc in CFG["cs"]:
        for with_stem in (False, True):
            r = analyse(cc, rcfg, rcfg_s, CFG, with_stem)
            res.append(dict(stem=with_stem, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}))
            print("  %4.1f%s | %.3f   %+.2f      %+.2f     | %+.3f  %s | %6.1f  %5.1f   %+.1f  %s   (%d planes, %.0f s)"
                  % (cc, " +stem" if with_stem else "      ", r["se"][5], r["corr"][5, 7], r["corr"][5, 2],
                     r["b_slope"][5], ("%+.3f" % r["b_stem"][5]) if with_stem else "  -   ", r["S"], r["se_S"],
                     r["bS_slope"], ("%+.1f" % r["bS_stem"]) if with_stem else "  -", r["planes"], r["seconds"]),
                  flush=True)
    json.dump(dict(config={k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}, results=res),
              open("soma_identifiability.json", "w"), indent=1)
