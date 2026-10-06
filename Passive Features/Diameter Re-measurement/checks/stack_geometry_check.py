"""Sandbox checks for the restatement of the synthetic-stack renderer (2026-10-05).

Not pipeline code. Verifies:
  G1  the vertical-ray interval of a tube with tilt phi and heading theta
      (local-frame closed form) against the 3-D membership test of handoff Eq. 6;
  G2  exact slab chords: closed form vs brute-force z-sampling, and the sum
      over slabs equals mu * l(v) of handoff Eq. 10;
  R1  rotation equivariance: with a circular kernel and vertical rays, the
      fine-grid (pre-camera) profile across the tube does not depend on theta;
      after pixel integration it does (grid effect);
  R2  truncation: node-plane dip and flank haze vs the tube half-length U;
  E1  the worked example numbers (r = 0.25 um, phi = 20 deg, theta = 30 deg).
Illustrative kernel: ideal-Debye core widths of mathematics Sec. 3.4,
interpolated, continued IN PROPORTION beyond 0.84 um
(sigma_r = 0.603 |delta| / 0.84, slope 0.72 um/um; not the 0.79 rule of
optics_points_check.py). [corrected 2026-10-06: said "extrapolated linearly"]
Source of the (S1)-(S4), heading, grid and U numbers in
docs/TEEG_diameter_implementation_handoff_2026-10-06.md; reference output in
stack_geometry_check.out. Run from inside checks/ (~3 min):
    python stack_geometry_check.py            # geometry example truncation rotation
"""
import numpy as np
from scipy import ndimage

rng = np.random.default_rng(20261005)

# ---------------------------------------------------------------- geometry
def axis_dir(phi, theta):
    return np.array([np.cos(phi) * np.cos(theta),
                     np.cos(phi) * np.sin(theta),
                     np.sin(phi)])

def local_uv(x, y, c, theta):
    dx, dy = x - c[0], y - c[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return u, v

def ray_interval(x, y, c, r, phi, theta, U=None):
    """Depth interval [z_lo, z_hi] of the vertical ray through (x, y) inside the tube.
    U: optional half-length of the tube along u (vertical end cuts, |u| <= U)."""
    u, v = local_uv(x, y, c, theta)
    inside = np.abs(v) <= r
    if U is not None:
        inside = inside & (np.abs(u) <= U)
    z_axis_u = c[2] + u * np.tan(phi)                      # axis depth above u
    half = np.sqrt(np.clip(r**2 - v**2, 0.0, None)) / np.cos(phi)  # l(v)/2
    return z_axis_u - half, z_axis_u + half, inside

def inside_3d(p, c, r, t):
    d = p - c
    return (d * d).sum(-1) - (d @ t) ** 2 <= r**2         # handoff Eq. 6

def slab_absorbance(z_lo, z_hi, inside, zeta, dzeta, mu):
    lo = np.maximum(z_lo, zeta - dzeta / 2)
    hi = np.minimum(z_hi, zeta + dzeta / 2)
    return mu * np.clip(hi - lo, 0.0, None) * inside

# ---------------------------------------------------------------- G1, G2
def check_geometry(n_rays=4000):
    worst_g1 = 0.0
    worst_g2 = 0.0
    worst_sum = 0.0
    for _ in range(20):
        r = rng.uniform(0.1, 2.0)
        phi = rng.uniform(0.0, np.radians(75.0))
        theta = rng.uniform(-np.pi, np.pi)
        c = rng.uniform(-1, 1, 3)
        t = axis_dir(phi, theta)
        x = c[0] + rng.uniform(-3, 3, n_rays)
        y = c[1] + rng.uniform(-3, 3, n_rays)
        z_lo, z_hi, ins = ray_interval(x, y, c, r, phi, theta)
        # G1: points just inside/outside the closed-form interval vs 3-D test
        eps = 1e-6
        for z, expect in ((z_lo + eps, True), (z_hi - eps, True),
                          (z_lo - 1e-3, False), (z_hi + 1e-3, False)):
            p = np.stack([x, y, z], -1)
            got = inside_3d(p, c, r, t)
            sel = ins & (z_hi - z_lo > 4e-3)
            bad = np.mean(got[sel] != expect)
            worst_g1 = max(worst_g1, bad)
        # rays with |v| > r must miss the tube everywhere
        sel = ~ins
        zz = np.linspace(-15, 15, 3001)
        p = np.stack([np.repeat(x[sel][:200, None], zz.size, 1),
                      np.repeat(y[sel][:200, None], zz.size, 1),
                      np.broadcast_to(zz, (min(200, sel.sum()), zz.size))], -1)
        worst_g1 = max(worst_g1, float(inside_3d(p, c, r, t).any()))
        # G2: slab chords vs brute force, and sum over slabs
        dzeta = 0.05
        zmin, zmax = np.nanmin(z_lo[ins]) - 0.1, np.nanmax(z_hi[ins]) + 0.1
        zetas = np.arange(zmin, zmax + dzeta, dzeta)
        mu = 1.0
        a = np.stack([slab_absorbance(z_lo, z_hi, ins, zj, dzeta, mu) for zj in zetas])
        ell = np.where(ins, 2 * np.sqrt(np.clip(r**2 - local_uv(x, y, c, theta)[1]**2, 0, None)) / np.cos(phi), 0.0)
        worst_sum = max(worst_sum, np.max(np.abs(a.sum(0) - mu * ell)))
        # brute force on 60 rays x 4 slabs
        idx = np.flatnonzero(ins)[:60]
        for i in idx:
            j_hit = np.flatnonzero(a[:, i] > 0)
            for j in j_hit[:4]:
                zs = np.linspace(zetas[j] - dzeta / 2, zetas[j] + dzeta / 2, 20001)
                p = np.stack([np.full_like(zs, x[i]), np.full_like(zs, y[i]), zs], -1)
                frac = inside_3d(p, c, r, t).mean()
                worst_g2 = max(worst_g2, abs(frac * dzeta - a[j, i] / mu))
    print(f"G1 closed-form interval vs 3-D membership: worst mismatch fraction {worst_g1:.3g}")
    print(f"G2 slab chord vs brute force (20001 z-samples per slab): worst |error| {worst_g2:.2e} um")
    print(f"G2 sum over slabs - mu*l(v): worst |error| {worst_sum:.2e} (absorbance)")

# ---------------------------------------------------------------- kernel (illustrative)
DELTA_TAB = np.array([0.00, 0.14, 0.28, 0.42, 0.56, 0.84])
SIGMA_TAB = np.array([0.080, 0.086, 0.122, 0.262, 0.438, 0.603])

def sigma_r(delta):
    d = np.abs(delta)
    s = np.interp(d, DELTA_TAB, SIGMA_TAB)
    return np.where(d > 0.84, SIGMA_TAB[-1] * d / 0.84, s)

# ---------------------------------------------------------------- renderer (fine grid)
def render_planes(r, phi, theta, mu, c, z_planes, half_xy, h_g, dzeta, pad, U=None):
    """Continuous-level planes I_k/B on a fine grid; light travels toward +z.
    The object is drawn over the padded grid (the tube continues into the margin
    unless capped at |u| <= U); convolution in Fourier space, array zero-padded 2x
    so the convolution is linear, not circular."""
    n = int(round(2 * (half_xy + pad) / h_g))
    n += n % 2
    xs = (np.arange(n) - n / 2 + 0.5) * h_g
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    z_lo, z_hi, ins = ray_interval(X, Y, c, r, phi, theta, U)
    zmin = np.min(z_lo[ins]); zmax = np.max(z_hi[ins])
    zetas = np.arange(zmin + dzeta / 2, zmax + dzeta, dzeta)
    # zero-pad by 2x for linear (not circular) convolution of the margin region
    N = 2 * n
    fx = np.fft.fftfreq(N, d=h_g)
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    f2 = FX**2 + FY**2
    acc = [np.zeros((N, N), complex) for _ in z_planes]
    T_before = np.ones_like(X)
    for zj in zetas:                                     # slabs in light order
        a = slab_absorbance(z_lo, z_hi, ins, zj, dzeta, mu)
        if not a.any():
            continue
        dA = T_before * (1.0 - np.exp(-a))               # procedure Eq. 5
        T_before = T_before * np.exp(-a)
        big = np.zeros((N, N)); big[:n, :n] = dA
        F = np.fft.fft2(big)
        for k, zk in enumerate(z_planes):
            s = float(sigma_r(zj - zk))
            acc[k] += F * np.exp(-2 * np.pi**2 * s**2 * f2)   # Gaussian transfer fn
    planes = []
    for k in range(len(z_planes)):
        img = 1.0 - np.real(np.fft.ifft2(acc[k]))[:n, :n]      # procedure Eq. 6
        planes.append(img)
    return xs, np.array(planes), len(zetas)

def profile(img, xs, c_xy, theta, half=3.0, step=0.1144):
    """Bilinear profile along e_v(theta) through c_xy (handoff Eq. 9)."""
    v = np.arange(-half, half + 1e-9, step)
    px = c_xy[0] - v * np.sin(theta)
    py = c_xy[1] + v * np.cos(theta)
    h = xs[1] - xs[0]
    col = (px - xs[0]) / h
    row = (py - xs[0]) / h
    return v, ndimage.map_coordinates(img, [row, col], order=1, mode="nearest")

def pixel_integrate(img, xs, factor):
    n = (img.shape[0] // factor) * factor
    im = img[:n, :n].reshape(n // factor, factor, n // factor, factor).mean((1, 3))
    xp = xs[:n].reshape(-1, factor).mean(1)
    return im, xp

def check_rotation():
    """Same finite tube (|u| <= U) at three headings; fine grid at two resolutions."""
    r, phi, mu, U = 0.25, np.radians(20.0), 1.0, 3.5
    c = np.array([0.0, 0.0, 0.0])
    z_planes = [0.0]
    res = {}
    for fac in (8, 16):
        h_g = 0.1144 / fac
        for theta in (0.0, np.radians(30.0), np.radians(45.0)):
            if fac == 16 and theta == np.radians(45.0):
                continue
            xs, planes, J = render_planes(r, phi, theta, mu, c, z_planes,
                                          half_xy=3.5, h_g=h_g, dzeta=0.05, pad=1.5, U=U)
            v, p_fine = profile(planes[0], xs, c[:2], theta)
            im_px, xp = pixel_integrate(planes[0], xs, fac)
            _, p_px = profile(im_px, xp, c[:2], theta)
            res[(fac, theta)] = (p_fine, p_px)
    dip = 1 - res[(8, 0.0)][0].min()
    print(f"R1 finite tube r=0.25, phi=20, U=3.5 um; node-plane dip depth (fine, theta=0) = {dip:.4f}")
    for fac in (8, 16):
        f0, p0 = res[(fac, 0.0)]
        for th in (np.radians(30.0), np.radians(45.0)):
            if (fac, th) not in res:
                continue
            f, p = res[(fac, th)]
            print(f"   h_g = p_x/{fac:2d}, theta={np.degrees(th):3.0f}: fine max|diff| = "
                  f"{np.max(np.abs(f - f0)):.2e} ({np.max(np.abs(f - f0)) / dip:.2%} of dip); "
                  f"pixel-integrated max|diff| = {np.max(np.abs(p - p0)):.2e} "
                  f"({np.max(np.abs(p - p0)) / dip:.2%} of dip)")
    f8, _ = res[(8, 0.0)]; f16, _ = res[(16, 0.0)]
    print(f"   discretisation check, theta=0: |profile(p_x/8) - profile(p_x/16)| max = "
          f"{np.max(np.abs(f8 - f16)):.2e}")
    for key, (f, p) in sorted(res.items()):
        print(f"   dip 1 - I/B at node: h_g = p_x/{key[0]:2d}, theta={np.degrees(key[1]):3.0f}: "
              f"fine {1 - f.min():.4f}, pixel-integrated {1 - p.min():.4f}")
    alpha = mu * 2 * r / np.cos(phi)
    print(f"   reference: column absorbed fraction 1 - exp(-alpha) = {1 - np.exp(-alpha):.4f} (alpha = {alpha:.4f})")

def check_truncation():
    """Node-plane intensity at v = 0 and v = 2.5 um vs tube half-length U.
    Direct sum over the tube's own grid points (no FFT, no wrap-around)."""
    r, mu, dzeta = 0.25, 1.0, 0.05
    h_g = 0.1144 / 4
    print("R2 truncation: dip at node (v=0) and haze at the flank (v=2.5 um), node plane z_k = axis depth")
    for ph_deg in (20, 45, 60):
        phi = np.radians(ph_deg)
        row = []
        for U in (2.0, 4.0, 6.0, 10.0, 15.0):
            us = np.arange(-U, U + 1e-9, h_g)
            vs = np.arange(-r, r + 1e-9, h_g / 2)
            Uu, Vv = np.meshgrid(us, vs, indexing="ij")
            z_axis_u = Uu * np.tan(phi)
            half = np.sqrt(np.clip(r**2 - Vv**2, 0, None)) / np.cos(phi)
            z_lo, z_hi = z_axis_u - half, z_axis_u + half
            zmin, zmax = z_lo.min(), z_hi.max()
            zetas = np.arange(zmin + dzeta / 2, zmax + dzeta, dzeta)
            T_before = np.ones_like(Uu)
            pts = np.array([[0.0, 0.0], [0.0, 2.5]])         # (u, v) of evaluation points
            acc = np.zeros(len(pts))
            cell = h_g * (h_g / 2)
            for zj in zetas:
                lo = np.maximum(z_lo, zj - dzeta / 2); hi = np.minimum(z_hi, zj + dzeta / 2)
                a = mu * np.clip(hi - lo, 0, None)
                if not a.any():
                    continue
                dA = T_before * (1 - np.exp(-a)); T_before *= np.exp(-a)
                m = dA > 0
                s = float(sigma_r(zj - 0.0))
                for k, (pu, pv) in enumerate(pts):
                    d2 = (Uu[m] - pu) ** 2 + (Vv[m] - pv) ** 2
                    acc[k] += np.sum(dA[m] * np.exp(-d2 / (2 * s**2))) * cell / (2 * np.pi * s**2)
            row.append((U, acc[0], acc[1]))
        txt = "; ".join(f"U={U:4.1f}: dip {a0:.4f}, flank {a1:.4f}" for U, a0, a1 in row)
        print(f"   phi={ph_deg:2d}: {txt}")

# ---------------------------------------------------------------- E1 worked example
def worked_example():
    r, phi, theta, mu = 0.25, np.radians(20.0), np.radians(30.0), 1.0
    dz, L, dzeta = 0.28, 4.0, 0.05
    d = 2 * r
    ell0 = 2 * r / np.cos(phi)
    alpha = mu * d / np.cos(phi)
    span_L = L * np.sin(phi)                      # depth span over the line-fit window (path length L)
    lo = -(L / 2) * np.sin(phi) - ell0 / 2 - 3 * dz
    hi = +(L / 2) * np.sin(phi) + ell0 / 2 + 3 * dz
    n_planes = int(np.floor(hi / dz) - np.ceil(lo / dz) + 1)
    # tube's depth range inside a 10 x 10 um block centred on the node
    half = 5.0
    corners = np.array([[sx * half, sy * half] for sx in (-1, 1) for sy in (-1, 1)])
    u_c = corners @ np.array([np.cos(theta), np.sin(theta)])
    z_rng = (u_c.max() - u_c.min()) * np.tan(phi) + ell0
    print("E1 r=0.25 um, phi=20 deg, theta=30 deg, mu=1 um^-1 (illustrative):")
    print(f"   alpha = mu d / cos phi = {alpha:.3f};  l(0) = {ell0:.3f} um "
          f"= {ell0 / dz:.2f} plane spacings = {ell0 / dzeta:.1f} slabs of 0.05 um")
    print(f"   rise tan(phi) = {np.tan(phi):.3f} um per um in-plane; depth span over L=4 um of path = {span_L:.3f} um "
          f"= {span_L / dz:.2f} planes")
    print(f"   planes needed (node z +- (L/2) sin phi +- l(0)/2 +- 3 dz): [{lo:.3f}, {hi:.3f}] um -> {n_planes} planes")
    print(f"   tube depth range across the 10x10 um block (axis + l(0)) = {z_rng:.2f} um -> "
          f"{z_rng / dzeta:.0f} slabs; largest slab defocus from the node plane ~ {z_rng / 2:.2f} um, "
          f"sigma_r there {float(sigma_r(z_rng / 2)):.2f} um -> 3 sigma margin {3 * float(sigma_r(z_rng / 2)):.1f} um")
    for ph_deg in (20, 40, 60):
        ph = np.radians(ph_deg)
        z_rng = (u_c.max() - u_c.min()) * np.tan(ph) + 2 * r / np.cos(ph)
        print(f"   phi={ph_deg:2d}: block depth range {z_rng:5.2f} um, {z_rng / dzeta:4.0f} slabs, "
              f"sigma_r(range/2) {float(sigma_r(z_rng / 2)):.2f} um")
    # alternative bound: geometric onset of the tilt halo (handoff Eq. 13b) for comparison
    for gamma, lab in ((0.79, "Debye cos^4"), (0.90, "isotropic"), (1.21, "uniform disc")):
        print(f"   RMS-width crossover gamma*tan(phi)=1 at phi = {np.degrees(np.arctan(1 / gamma)):.1f} deg ({lab})")

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["geometry", "example", "rotation", "truncation"]
    if "geometry" in which:
        check_geometry()
    if "example" in which:
        worked_example()
    if "truncation" in which:
        check_truncation()
    if "rotation" in which:
        check_rotation()
