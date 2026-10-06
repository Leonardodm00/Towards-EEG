"""Sandbox illustrations for the three renderer assumptions (2026-10-06).

NOT pipeline code. Source of the [run] numbers in
docs/TEEG_diameter_implementation_handoff_2026-10-06.md ("Findings"); reference
output in optics_points_check.out. Run from inside checks/:
    python optics_points_check.py checks A BC matched   # ~4 min; default: checks A BC
Purpose: put numbers on
  (A) the kernel K_delta: how far the renderer needs it vs the planned
      calibration range, and how much the extrapolation rule moves d_hat;
  (B) the absorbed-light partition and (C) vertical rays, against a
      geometric-optics ray reference that uses the SAME rays for the blur.

Conventions: local frame (u along the branch projection, v across, z depth in
stage um, light travelling toward +z); tube axis through the origin with
direction (cos phi, 0, sin phi); delta = object depth - plane depth.
Illustrative kernel: ideal-Debye LSF core widths (mathematics Sec. 3.4),
linear interpolation, three continuations beyond 0.84 um.
"""
import numpy as np
from scipy import ndimage, optimize

DZ = 0.28                      # plane spacing, um
PX = 0.1144                    # pixel pitch, um
DELTA_TAB = np.array([0.00, 0.14, 0.28, 0.42, 0.56, 0.84])
SIGMA_TAB = np.array([0.080, 0.086, 0.122, 0.262, 0.438, 0.603])
D_MAX = 0.84                   # planned calibration range (+-3 planes)
SLOPES = {"frozen": 0.0, "debye_0.79": 0.79, "disc_1.21": 1.21}
V_PROF = np.arange(-26, 27) * PX          # pipeline profile samples, +-3 um


def sigma_r(delta, rule):
    d = np.abs(np.asarray(delta, float))
    s = np.interp(d, DELTA_TAB, SIGMA_TAB)
    return np.where(d > D_MAX, SIGMA_TAB[-1] + SLOPES[rule] * (d - D_MAX), s)


# ------------------------------------------------------------------ partition renderer
def render_partition_points(r, phi, mu, z_k, rule, U=6.0, hu=0.02, hv=0.01,
                            dzeta=0.02, v_eval=V_PROF, split_dmax=False):
    """Node-plane-type intensity I/B at (u=0, v_eval) in plane z_k, procedure
    Eqs. 5-6 with circular Gaussian kernels. Separable sum over the tube grid.
    Returns I (and, if split_dmax, the part of the dip from |delta| > D_MAX)."""
    us = np.arange(-U + hu / 2, U, hu)
    vs = np.arange(-r + hv / 2, r, hv)
    Uu, Vv = np.meshgrid(us, vs, indexing="ij")
    ell = 2 * np.sqrt(np.clip(r**2 - Vv**2, 0, None)) / np.cos(phi)
    z_lo = Uu * np.tan(phi) - ell / 2
    z_hi = z_lo + ell
    zetas = np.arange(z_lo.min() + dzeta / 2, z_hi.max() + dzeta, dzeta)
    dip = np.zeros(v_eval.size)
    dip_far = np.zeros(v_eval.size)
    cell = hu * hv
    for zj in zetas:
        lo = zj - dzeta / 2
        hi = zj + dzeta / 2
        a = mu * np.clip(np.minimum(z_hi, hi) - np.maximum(z_lo, lo), 0, None)
        if not a.any():
            continue
        below = np.clip(np.minimum(z_hi, lo) - z_lo, 0, None)   # column inside tube below slab
        dA = np.exp(-mu * below) * (1 - np.exp(-a))             # Eq. 5
        s = float(sigma_r(zj - z_k, rule))
        gu = np.exp(-us**2 / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)
        R = gu @ dA                                              # sum over u -> (n_v,)
        gv = np.exp(-(vs[:, None] - v_eval[None, :])**2 / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)
        contrib = (R @ gv) * cell
        dip += contrib
        if abs(zj - z_k) > D_MAX:
            dip_far += contrib
    I = 1.0 - dip
    return (I, dip_far) if split_dmax else I


# ------------------------------------------------------------------ the fit (handoff Eq. 11, D-018, D-019)
def fit_profile(I, v, phi, sigma_fit, d0, mu0, Bbar=1.0):
    w = np.arange(-4.5, 4.5, 0.0025)

    def model(p):
        d, mu, v0 = p
        x = 2 * (w - v0) / d
        sd = np.sqrt(np.clip(1 - x**2, 0, None))
        f = 1 - np.exp(-mu * d / np.cos(phi) * sd)
        conv = ndimage.gaussian_filter1d(f, sigma_fit / 0.0025, mode="constant", truncate=8)
        return Bbar * (1 - np.interp(v, w, conv))

    best = None
    for fac in (1.0, 0.7, 1.4):
        res = optimize.least_squares(lambda p: model(p) - I, x0=[d0 * fac, mu0, 0.0],
                                     bounds=([0.02, 1e-5, -0.5], [8.0, 40.0, 0.5]),
                                     xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or res.cost < best.cost:
            best = res
    return best.x, best.cost


# ------------------------------------------------------------------ geometric ray reference
def directions(n_rho=48, n_psi=96, na=1.4, n=1.515):
    """Directions uniform in (s_u, s_v) over the NA disc (sine-condition condenser,
    evenly filled aperture). Returns s (N,3) and weights summing to 1."""
    sm = na / n
    x, wx = np.polynomial.legendre.leggauss(n_rho)
    rho = (x + 1) / 2 * sm**2
    wr = wx / 2                                   # sums to 1 over [0, sm^2] after / sm^2 * sm^2
    psi = (np.arange(n_psi) + 0.5) * 2 * np.pi / n_psi
    R, P = np.meshgrid(rho, psi, indexing="ij")
    W = np.repeat(wr[:, None], n_psi, 1) / n_psi
    s = np.sqrt(R)
    su, sv = s * np.cos(P), s * np.sin(P)
    sz = np.sqrt(1 - s**2)
    S = np.stack([su.ravel(), sv.ravel(), sz.ravel()], 1)
    return S, W.ravel() / W.sum()


def chord_interval(P, S, r, phi, U):
    """Arc-length interval [t1, t2] of the line P + t s inside the tube capped at |u| <= U."""
    t = np.array([np.cos(phi), 0.0, np.sin(phi)])
    q = np.asarray(P, float)
    st = S @ t
    A = 1 - st**2
    qt = q @ t
    B = S @ q - qt * st
    C = q @ q - qt**2 - r**2
    disc = B**2 - A * C
    ok = (disc > 0) & (A > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    t1 = np.where(ok, (-B - sq) / np.where(ok, A, 1), 0)
    t2 = np.where(ok, (-B + sq) / np.where(ok, A, 1), 0)
    su = S[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (-U - q[0]) / su
        tb = (U - q[0]) / su
    lo = np.where(su != 0, np.minimum(ta, tb), -np.inf)
    hi = np.where(su != 0, np.maximum(ta, tb), np.inf)
    t1c = np.maximum(t1, lo)
    t2c = np.minimum(t2, hi)
    good = ok & (t2c > t1c)
    return np.where(good, t1c, 0), np.where(good, t2c, 0), good


def ray_models(v0, z_k, r, phi, mu, S, W, U=6.0, n_t=32):
    """Returns I/B for: L (linear), P (partition), Gv (ray, vertical path element),
    G (ray, true oblique path), at the point (0, v0, z_k). Geometric blur only."""
    P = np.array([0.0, v0, z_k])
    t1, t2, good = chord_interval(P, S, r, phi, U)
    Lc = t2 - t1                                   # arc length inside the tube
    sz = S[:, 2]
    I_G = np.sum(W * np.exp(-mu * Lc))
    I_Gv = np.sum(W * np.exp(-mu * Lc * sz))
    I_L = 1 - np.sum(W * mu * Lc * sz)
    # partition along the line: integrand mu*sz*exp(-mu*(z - z_lo(column at that point)))
    x, wx = np.polynomial.legendre.leggauss(n_t)
    idx = np.flatnonzero(good)
    tt = (t1[idx, None] + t2[idx, None]) / 2 + (t2[idx, None] - t1[idx, None]) / 2 * x[None, :]
    uu = tt * S[idx, 0:1]
    vv = v0 + tt * S[idx, 1:2]
    zz = z_k + tt * S[idx, 2:3]
    ell = 2 * np.sqrt(np.clip(r**2 - vv**2, 0, None)) / np.cos(phi)
    zlo = uu * np.tan(phi) - ell / 2
    integ = mu * S[idx, 2:3] * np.exp(-mu * np.clip(zz - zlo, 0, None))
    absorbed = (integ @ wx) * (t2[idx] - t1[idx]) / 2
    I_P = 1 - np.sum(W[idx] * absorbed)
    return I_L, I_P, I_Gv, I_G


def ray_profiles(r, phi, mu, z_k, S, W, v_fine, post_sigma=0.08):
    out = np.array([ray_models(v, z_k, r, phi, mu, S, W) for v in v_fine])  # (n, 4)
    h = v_fine[1] - v_fine[0]
    prof = {}
    for k, name in enumerate(("L", "P", "Gv", "G")):
        dip = ndimage.gaussian_filter1d(1 - out[:, k], post_sigma / h, mode="constant", truncate=6)
        prof[name] = 1 - np.interp(V_PROF, v_fine, dip)
    return prof, out


# ------------------------------------------------------------------ sections
def section_A():
    print("=== A. Kernel ===")
    # A1: how fast a thin, flat node fades with defocus (partition renderer, phi = 0)
    for d, mu in ((0.3, 0.6), (1.0, 0.6)):
        r = d / 2
        dips = []
        for off in (0.0, 0.28, 0.56, 0.84, 1.12, 1.40):
            I = render_partition_points(r, 0.0, mu, -off, "debye_0.79", U=6.0,
                                        v_eval=np.array([0.0]), hu=0.05, hv=r / 50, dzeta=0.01)
            dips.append(1 - I[0])
        rel = np.array(dips) / dips[0]
        print(f"A1 d={d} mu={mu}: in-focus dip {dips[0]:.4f}; dip / in-focus at 1,2,3,4,5 planes: "
              + ", ".join(f"{x:.2f}" for x in rel[1:]))
    # A2: node-plane share from |delta| > D_MAX and fitted d_hat/d under three continuations
    print("A2 node plane (z_k = axis depth), U = 6 um, sigma_fit = 0.080, Bbar = B (oracle):")
    for d, ph_deg in ((0.5, 0), (0.5, 20), (0.5, 45), (0.5, 60), (2.0, 0), (2.0, 20)):
        r, phi, mu = d / 2, np.radians(ph_deg), 1.0
        hu, hv, dz = (0.02, 0.01, 0.02) if d < 1 else (0.04, 0.02, 0.04)
        row = []
        far_share = None
        for rule in ("frozen", "debye_0.79", "disc_1.21"):
            I, far = render_partition_points(r, phi, mu, 0.0, rule, hu=hu, hv=hv, dzeta=dz,
                                             split_dmax=True)
            if rule == "debye_0.79":
                far_share = far[26] / (1 - I[26])
            p, cost = fit_profile(I, V_PROF, phi, 0.080, d, mu)
            row.append((p[0] / d, 1 - I[26]))
        txt = "; ".join(f"{k}: d^/d {x:.3f} (dip {y:.3f})" for k, (x, y) in
                        zip(("frozen", "0.79", "1.21"), row))
        print(f"   d={d} phi={ph_deg:2d}: dip share from |delta|>0.84 (0.79 rule) {far_share:.1%}; {txt}")
    # A3: the focus-search planes for a flat tube
    for d in (0.5, 2.0):
        r = d / 2
        txt = []
        for off in (-3 * DZ, 3 * DZ):
            vals = []
            for rule in ("frozen", "debye_0.79", "disc_1.21"):
                I = render_partition_points(r, 0.0, 1.0, off, rule, U=6.0, v_eval=np.array([0.0]),
                                            hu=0.05, hv=r / 40, dzeta=0.01)
                vals.append(1 - I[0])
            txt.append(f"z_k={off:+.2f}: " + "/".join(f"{x:.4f}" for x in vals))
        print(f"A3 d={d} phi=0 mu=1, dip at v=0 in the +-3-plane focus-search planes "
              f"(frozen/0.79/1.21): " + "; ".join(txt))


def section_BC():
    print("=== B, C. Partition and vertical rays vs geometric ray reference ===")
    S, W = directions()
    sz = S[:, 2]
    sm = 1.4 / 1.515
    print(f"<1/cos> over the cone: quadrature {np.sum(W / sz):.4f}, analytic "
          f"{2 / sm**2 * (1 - np.sqrt(1 - sm**2)):.4f}; max 1/cos = {1 / np.sqrt(1 - sm**2):.3f}")
    v_fine = np.arange(-3.5, 3.5001, 0.01)
    cases = [(1.0, 0, 0.05), (1.0, 0, 0.5), (1.0, 0, 1.5), (1.0, 0, 3.0), (0.5, 20, 1.0)]
    for d, ph_deg, mu in cases:
        r, phi = d / 2, np.radians(ph_deg)
        res = {}
        for zk in (-3 * DZ, 0.0, 3 * DZ):
            prof, raw = ray_profiles(r, phi, mu, zk, S, W, v_fine)
            res[zk] = (prof, raw)
        prof0, raw0 = res[0.0]
        i0 = np.argmin(np.abs(v_fine))
        dips = {k: 1 - raw0[i0, j] for j, k in enumerate(("L", "P", "Gv", "G"))}
        asym = {}
        for j, k in enumerate(("P", "Gv", "G")):
            jj = ("L", "P", "Gv", "G").index(k)
            lo = 1 - res[-3 * DZ][1][i0, jj]
            hi = 1 - res[3 * DZ][1][i0, jj]
            asym[k] = (lo, hi)
        fits = {}
        for k in ("P", "Gv", "G"):
            p, cost = fit_profile(prof0[k], V_PROF, phi, 0.080, d, mu)
            fits[k] = p
        print(f"d={d} phi={ph_deg} mu={mu} (mu*d={mu * d:.2f}):")
        print("   node-plane dip at v=0 (geometric blur only): " +
              ", ".join(f"{k} {v:.4f}" for k, v in dips.items()))
        print("   dip at z_k = -3dz (light side) / +3dz: " +
              ", ".join(f"{k} {a:.4f}/{b:.4f}" for k, (a, b) in asym.items()))
        print("   fit at node plane (post-blur 0.08, sigma_fit 0.08, Bbar=B): " +
              ", ".join(f"{k}: d^/d {p[0] / d:.3f}, mu^/mu {p[1] / mu:.3f}" for k, p in fits.items()))


def section_matched():
    """Renderer as it would be AFTER calibration on faint nodes and mu-matching:
    P* = partition with the faint-limit kernel of the truth (ray weights W/cos, renormalised),
    phantom mu chosen so that the fitted mu^ equals the truth's (procedure Sec. 3.5).
    Truth = G (oblique paths, weights W). Gv* splits the gap: true ray history,
    vertical path element, same W* weights."""
    print("=== matched comparison (truth G; renderer P*; split Gv*) ===")
    S, W = directions()
    Wst = W / S[:, 2]
    Wst = Wst / Wst.sum()
    v_fine = np.arange(-3.5, 3.5001, 0.02)
    i0 = np.argmin(np.abs(v_fine))
    cases = [(1.0, 0, 0.05), (1.0, 0, 0.5), (1.0, 0, 1.5), (1.0, 0, 3.0), (0.5, 20, 1.0)]

    def prof(r, phi, mu, zk, weights, col):
        out = np.array([ray_models(v, zk, r, phi, mu, S, weights)[col] for v in v_fine])
        h = v_fine[1] - v_fine[0]
        dip = ndimage.gaussian_filter1d(1 - out, 0.08 / h, mode="constant", truncate=6)
        return 1 - np.interp(V_PROF, v_fine, dip)

    for d, ph_deg, mu in cases:
        r, phi = d / 2, np.radians(ph_deg)
        pG, _ = fit_profile(prof(r, phi, mu, 0.0, W, 3), V_PROF, phi, 0.080, d, mu)
        line = [f"d={d} phi={ph_deg} mu*d={mu * d:.2f}: truth G d^/d {pG[0] / d:.3f} (mu^ {pG[1]:.3f})"]
        for name, col in (("P*", 1), ("Gv*", 2)):
            def f(m):
                p, _ = fit_profile(prof(r, phi, m, 0.0, Wst, col), V_PROF, phi, 0.080, d, m)
                return p[1] - pG[1]
            lo, hi = 0.5 * mu, 8.0 * mu
            flo, fhi = f(lo), f(hi)
            if flo * fhi > 0:
                line.append(f"{name}: no bracket ({flo:.3f}, {fhi:.3f})")
                continue
            m_ph = optimize.brentq(f, lo, hi, xtol=1e-4 * mu, rtol=1e-6)
            p, _ = fit_profile(prof(r, phi, m_ph, 0.0, Wst, col), V_PROF, phi, 0.080, d, m_ph)
            a_lo = 1 - ray_models(0.0, -3 * DZ, r, phi, m_ph, S, Wst)[col]
            a_hi = 1 - ray_models(0.0, 3 * DZ, r, phi, m_ph, S, Wst)[col]
            line.append(f"{name}: mu_ph/mu {m_ph / mu:.3f}, d^/d {p[0] / d:.3f} "
                        f"(diff vs truth {p[0] / d - pG[0] / d:+.3f}); dip -3dz/+3dz {a_lo:.4f}/{a_hi:.4f}")
        gl = 1 - ray_models(0.0, -3 * DZ, r, phi, mu, S, W)[3]
        gh = 1 - ray_models(0.0, 3 * DZ, r, phi, mu, S, W)[3]
        line.append(f"truth dip -3dz/+3dz {gl:.4f}/{gh:.4f}")
        print("\n   ".join(line))


def self_checks():
    print("=== self-checks ===")
    rng = np.random.default_rng(1)
    S, W = directions(16, 32)
    # chord vs brute-force membership along each line
    worst = 0.0
    for _ in range(5):
        r, phi = rng.uniform(0.2, 1.0), np.radians(rng.uniform(0, 30))
        P = np.array([0.0, rng.uniform(-r, r), rng.uniform(-0.5, 0.5)])
        t1, t2, good = chord_interval(P, S, r, phi, 6.0)
        tt = np.linspace(-40, 40, 400001)
        ax = np.array([np.cos(phi), 0, np.sin(phi)])
        for k in range(0, S.shape[0], 37):
            pts = P[None, :] + tt[:, None] * S[k][None, :]
            dd = (pts**2).sum(1) - (pts @ ax)**2
            inside = (dd <= r**2) & (np.abs(pts[:, 0]) <= 6.0)
            L_bf = inside.sum() * (tt[1] - tt[0])
            worst = max(worst, abs(L_bf - (t2[k] - t1[k])))
    print(f"chord length vs brute force: worst |diff| {worst:.2e} um (grid step 2e-4)")
    # small-mu agreement P ~ Gv ~ L (first order) for a flat tube
    for mu in (0.01, 0.1):
        L_, P_, Gv_, G_ = ray_models(0.1, 0.2, 0.5, 0.0, mu, S, W)
        print(f"mu={mu}: (P-L)/(1-L) {(P_ - L_) / (1 - L_):.2e}, (Gv-L)/(1-L) {(Gv_ - L_) / (1 - L_):.2e}")
    # partition renderer: column absorbed fraction conservation (one column, many slabs)
    r, mu = 0.5, 2.0
    I = render_partition_points(r, 0.0, mu, 0.0, "frozen", U=3.0, v_eval=np.array([0.0]),
                                hu=0.05, hv=0.005, dzeta=0.005)
    print(f"partition renderer node dip {1 - I[0]:.4f} (sanity: below column absorbed fraction "
          f"{1 - np.exp(-mu * 2 * r):.4f})")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["checks", "A", "BC"]
    if "checks" in which:
        self_checks()
    if "A" in which:
        section_A()
    if "BC" in which:
        section_BC()
    if "matched" in which:
        section_matched()
