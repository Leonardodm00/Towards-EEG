"""Hybrid renderer vs ray world: does booking absorbed light with the history along the
rays, then spreading it with a kernel, cure the partition's dark failure? (2026-10-09)

NOT pipeline code. Theory chat check for the user's proposal (theory handoff, Open
work 1): compute absorption with the ray world's geometry (NA 1.4, n_oil 1.515) and
render each image plane with a kernel. Because a kernel does not know a ray's
direction, the absorbed light at a slice point is averaged over the directions
arriving there before it is spread. Tested where the answer is known exactly: in
the ray world, with the kernel of the cone's own rays weighted W* (the faint-limit
kernel, as in optics_points_check.py `matched`). Flat d = 1 um tube, centre point of
the node plane. Reference output in hybrid_absorption_check.out. Run from inside
checks/ (imports optics_points_check):
    python hybrid_absorption_check.py      # about 30 s

Added later on 2026-10-09, after an independent review of the renderers document
(docs/TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md, sections
3.8-3.9): where the hybrid books the light over the cross-section, its total against the
ray world's total shadow, the opaque-limit leak (S9) under W and W*, and why a kernel
cannot be split into the cone's rays plus a per-ray blur (RMS slopes; zeros of the cone
kernel's characteristic function). The same-mu variants are labelled P*s and Gv*s here.
"""
import numpy as np
from optics_points_check import directions, chord_interval, ray_models


r = 0.5
S_line, W_line = directions(16, 32)            # lines through the image point
Wst = W_line / S_line[:, 2]; Wst /= Wst.sum()  # W* = faint-limit kernel weights
S_abs, W_abs = directions(16, 32)               # directions arriving at a slice point

def A_density(v, z, mu):
    """direction-averaged absorption density per unit depth at points (v, z) inside the
    flat tube: < (mu / cos) exp(-mu * back-traced path inside the tube) >_W."""
    out = np.zeros(v.size)
    inside = v**2 + z**2 < r**2
    vv, zz = v[inside], z[inside]
    sv, sz = S_abs[:, 1], S_abs[:, 2]
    p2 = np.sqrt(sv**2 + sz**2)                  # length of the (v, z) projection of s
    dv, dz = -sv / p2, -sz / p2                  # unit back-trace direction in (v, z)
    acc = np.zeros(vv.size)
    for a in range(0, vv.size, 4000):
        pv, pz = vv[a:a + 4000, None], zz[a:a + 4000, None]
        pd = pv * dv[None, :] + pz * dz[None, :]
        c = pv**2 + pz**2 - r**2
        t2 = -pd + np.sqrt(np.clip(pd**2 - c, 0, None))   # 2-D distance back to the circle
        ell = t2 / p2[None, :]                            # 3-D path length inside the tube
        acc[a:a + 4000] = (mu / sz[None, :] * np.exp(-mu * ell)) @ W_abs
    out[inside] = acc
    return out

def hybrid_centre_dip(mu, n_t=600):
    P = np.zeros(3)
    t1, t2, good = chord_interval(P, S_line, r, 0.0, 6.0)
    x, wx = np.polynomial.legendre.leggauss(n_t)
    dip = 0.0
    for k in np.flatnonzero(good):
        tt = (t1[k] + t2[k]) / 2 + (t2[k] - t1[k]) / 2 * x
        v = tt * S_line[k, 1]; z = tt * S_line[k, 2]
        A = A_density(v, z, mu)
        # booked absorption along the line: integral of A dz = integral of A s_z dt
        dip += Wst[k] * np.sum(wx * A * S_line[k, 2]) * (t2[k] - t1[k]) / 2
    return dip

print("flat d = 1 um, node plane, centre point; dip = 1 - I/B; geometric blur only, no post-blur")
print("(P*s, Gv*s: weights W* but the same mu as the ray world, no mu matching; followup_checks.py prints them as P*, Gv*)")
print("mu*d   ray world G (W)   hybrid H (W*)   partition P*s (W*, vert. path)  true history Gv*s (W*, vert. path)")
for mud in (0.05, 0.5, 1.5, 3.0, 10.0, 50.0):
    mu = mud / 1.0
    _, _, _, G = ray_models(0.0, 0.0, r, 0.0, mu, S_line, W_line)
    _, P, Gv, _ = ray_models(0.0, 0.0, r, 0.0, mu, S_line, Wst, n_t=200)
    H = hybrid_centre_dip(mu)
    print(f"{mud:5.2f}   {1 - G:.3f}             {H:.3f}           {1 - P:.3f}                          {1 - Gv:.3f}", flush=True)

# Where the hybrid books the light (added 2026-10-09 after an independent review): the
# booked absorption per unit length of the tube, integrated over its cross-section, and
# the share of it above the axis depth; and the opaque-limit leak (S9) under W and W*.
print()
print("hybrid booking over the cross-section (flat d = 1 um):")
print("   opaque-limit total 2 r <sqrt(s_v^2 + s_z^2) / s_z>_W = %.4f um" % (
    2 * r * np.sum(W_abs * np.sqrt(S_abs[:, 1]**2 + S_abs[:, 2]**2) / S_abs[:, 2])))
for mud, h in ((3.0, 0.002), (50.0, 0.001)):
    g = np.arange(-r + h / 2, r, h)
    V, Z = np.meshgrid(g, g, indexing="ij")
    A = A_density(V.ravel(), Z.ravel(), mud / 1.0).reshape(V.shape)
    tot = A.sum() * h * h
    up = A[:, g > 0].sum() * h * h
    print("   mu*d = %4.1f: total booked %.3f um per unit length; share above the axis depth %.3f" % (mud, tot, up / tot))
t1, t2, good = chord_interval(np.zeros(3), S_line, r, 0.0, 6.0)
v_e = t1 * S_line[:, 1]
slope = np.abs(v_e) / np.sqrt(np.clip(r**2 - v_e**2, 1e-12, None))
absorbed = 1.0 / (1.0 + slope * np.abs(S_line[:, 1]) / S_line[:, 2])
print("opaque-limit partition leak at the centre, (S9): %.3f with W* weights, %.3f with W weights" % (
    np.sum(Wst * absorbed), np.sum(W_line * absorbed)))

# The ray world's total shadow per unit length, <integral of (1 - exp(-mu L)) dv>_W, for the
# same tube: per direction it equals the hybrid's total booking (both are
# (p / s_z) * integral of (1 - exp(-mu c(w) / p)) dw, c the in-plane chord at offset w), so
# the hybrid moves light, it does not lose it.
p_abs = np.sqrt(S_abs[:, 1]**2 + S_abs[:, 2]**2)
xw, ww = np.polynomial.legendre.leggauss(400)
chord = 2 * np.sqrt(r**2 - (r * xw)**2)
for mud in (3.0, 50.0):
    per_dir = (p_abs / S_abs[:, 2]) * ((1 - np.exp(-mud * chord[None, :] / p_abs[:, None])) @ (r * ww))
    print("   ray world total shadow at mu*d = %4.1f: %.4f um per unit length" % (mud, np.sum(W_abs * per_dir)))

# Why a kernel cannot be split into the cone's own rays plus a per-ray blur, K = K_cone * kappa:
# (i) variances would add, so sigma_K >= the cone's RMS; (ii) the Fourier transform of K_cone has
# zeros, so a kernel whose transform has none (a Gaussian) has no such kappa at any defocus.
# Characteristic function of the per-axis slope t_x = s_x / s_z, continuous measures:
# W: (s_x, s_y) uniform on the disc of radius s_m; W*: W reweighted by 1 / cos.
from scipy import integrate, special
s_m = 1.4 / 1.515
def char_fun(k, star):
    wgt = (lambda q: q / np.sqrt(1 - q**2)) if star else (lambda q: q)
    num = integrate.quad(lambda q: special.j0(k * q / np.sqrt(1 - q**2)) * wgt(q), 0, s_m,
                         limit=2000, epsabs=1e-12, epsrel=1e-10)[0]
    return num / integrate.quad(wgt, 0, s_m, limit=200)[0]
ks = np.linspace(0.0, 12.0, 1201)
print()
print("cone kernel: per-axis RMS slope and zeros of its characteristic function (k in rad per unit slope)")
for name, star, w in (("W ", False, W_line), ("W*", True, Wst)):
    cf = np.array([char_fun(k, star) for k in ks])
    sc = np.flatnonzero(np.diff(np.sign(cf)))
    print("   %s: RMS slope %.4f; first sign changes at k = %s; min over k <= 12: %.4f" % (
        name, np.sqrt(np.sum(w * (S_line[:, 0] / S_line[:, 2])**2)), np.round(ks[sc[:2]], 2), cf.min()))
