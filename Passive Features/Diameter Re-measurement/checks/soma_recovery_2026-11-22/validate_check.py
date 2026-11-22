"""The checks of the check (D-216.2), before the cases use them.

(1) The check renderer (recovery_lib.render_columns: given column intervals through Block 45's slabs, partition
    absorption and sampled-Gaussian blur) against SomaRenderer on the plain ellipsoid: on a small crop at the fit's
    settings (f 1, dzeta 0.14 um) and the truth's (f 2, dzeta 0.07 um), with the configured slope and 0.95; on the
    cases' fit crop at the truth's settings; and the planes in groups (each group its own box) against one group.
    Expected: agreement to round-off (one group: the same arithmetic as SomaRenderer).
(2) The soma's membrane area by marching cubes (recovery_lib.mesh_area) against analysis.soma_fit.ellipsoid_area on
    the plain ellipsoids of case A, and its convergence in the grid spacing (0.1, 0.05, 0.035 um).
(3) The union of case C (the ellipsoid and the cone): its area at the same spacings, its half-height (analytic and on
    a fine grid of columns), the cone's base inside the ellipsoid, the depth profile continuous at the cone's base.
(4) Context for source (i): on a small crop, the in-focus plane and +-3 um rendered at f 1, 2, 4 (dzeta 0.14,
    0.07, 0.035 um): how far the fit's discretisation is from the truth's, and the truth's from a finer one.

Run:  OMP_NUM_THREADS=1 python3 validate_check.py      (about 5 min; writes results/validation.json)
Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recovery_lib as L                                                  # noqa: E402
from allen_diameter.analysis.soma_fit import ellipsoid_area               # noqa: E402
from allen_diameter.config import SomaFitConfig                           # noqa: E402
from allen_diameter.model.render import block_fine_grid                   # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer                 # noqa: E402

S = L.SETTINGS
OUT = {}


def log(msg):
    print(msg, flush=True)


def small_crop(solid, half_um):
    p = S.p_um
    left = int(math.floor((solid.x0 - half_um) / p))
    top = int(math.floor((solid.y0 - half_um) / p))
    n = int(math.ceil(2 * half_um / p)) + 1
    return left, top, n, n


def check_renderer():
    rcfg = L.renderer_config()
    solid = L.make_solid(L.CASES["A_c4.5"])
    crop = small_crop(solid, 8.0)
    z0 = solid.z0
    planes = z0 + np.array([-8.0, -3.0, -0.28, 0.28, 3.0, 8.0])
    rows = []
    for f, dzeta in ((1, 0.14), (2, 0.07)):
        scfg = dataclasses.replace(SomaFitConfig(), fine_factor=f, dzeta_um=dzeta)
        for slope in (None, 0.95):
            t0 = time.perf_counter()
            ref = SomaRenderer(crop, S.p_um, planes, rcfg, scfg, S.light_direction).render(
                np.asarray(solid.theta_g), slope=slope)
            t1 = time.perf_counter()
            chk = L.render_solid(solid, crop, S.p_um, planes, rcfg, f, dzeta, S.light_direction, slope=slope)
            t2 = time.perf_counter()
            d = float(np.max(np.abs(chk - ref)))
            rows.append(dict(crop=crop, f=f, dzeta_um=dzeta, slope=slope, planes_minus_z0_um=planes - z0,
                             max_abs_diff=d, bit_identical=bool(np.array_equal(chk, ref)),
                             seconds_somarenderer=t1 - t0, seconds_check=t2 - t1, tau_min=float(ref.min())))
            log("  (1) small crop %s f %d dzeta %.2f slope %s: max|check - SomaRenderer| = %.3g (bit-identical %s)"
                % (crop[2:], f, dzeta, slope, d, rows[-1]["bit_identical"]))
    # the cases' fit crop, at the truth's settings, three planes
    scfg_fit = SomaFitConfig()
    ref_vals = L.soma_reference(solid)
    centre, _r = L.traced_start(solid, ref_vals)
    crop_fit = L.fit_crop(centre[:2], scfg_fit)
    planes3 = z0 + np.array([-3.0, 0.0, 3.0])
    scfg_t = dataclasses.replace(SomaFitConfig(), fine_factor=S.truth_fine_factor, dzeta_um=S.truth_dzeta_um)
    t0 = time.perf_counter()
    ref = SomaRenderer(crop_fit, S.p_um, planes3, rcfg, scfg_t, S.light_direction).render(np.asarray(solid.theta_g))
    chk = L.render_solid(solid, crop_fit, S.p_um, planes3, rcfg, S.truth_fine_factor, S.truth_dzeta_um,
                         S.light_direction)
    d = float(np.max(np.abs(chk - ref)))
    rows.append(dict(crop=crop_fit, f=S.truth_fine_factor, dzeta_um=S.truth_dzeta_um, slope=None,
                     planes_minus_z0_um=planes3 - z0, max_abs_diff=d, bit_identical=bool(np.array_equal(chk, ref)),
                     seconds_both=time.perf_counter() - t0))
    log("  (1) fit crop %s (truth's settings): max|check - SomaRenderer| = %.3g (bit-identical %s)"
        % (crop_fit[2:], d, rows[-1]["bit_identical"]))
    # the planes in groups (each its own box) against one group, on the small crop at the truth's settings
    chk1 = L.render_solid(solid, crop, S.p_um, planes, rcfg, 2, 0.07, S.light_direction)
    chkg = L.render_solid(solid, crop, S.p_um, planes, rcfg, 2, 0.07, S.light_direction, acc_mb=1.0)
    dg = float(np.max(np.abs(chkg - chk1)))
    rows.append(dict(crop=crop, f=2, dzeta_um=0.07, groups="one plane per group vs one group", max_abs_diff=dg))
    log("  (1) planes in groups vs one group: max|diff| = %.3g" % dg)
    # SomaRenderer in groups (render_ellipsoid_grouped, used for the truths of A and B) against one render
    scfg_t2 = dataclasses.replace(SomaFitConfig(), fine_factor=2, dzeta_um=0.07)
    ref1 = SomaRenderer(crop, S.p_um, planes, rcfg, scfg_t2, S.light_direction).render(np.asarray(solid.theta_g))
    refg = L.render_ellipsoid_grouped(solid.theta_g, crop, S.p_um, planes, rcfg, scfg_t2, S.light_direction,
                                      acc_mb=1.0)
    dr = float(np.max(np.abs(refg - ref1)))
    rows.append(dict(crop=crop, f=2, dzeta_um=0.07, groups="SomaRenderer one plane per group vs one render",
                     max_abs_diff=dr))
    log("  (1) SomaRenderer in groups vs one render: max|diff| = %.3g" % dr)
    OUT["renderer"] = rows


def check_area():
    rows = []
    for c in (3.0, 4.5, 6.0):
        solid = L.make_solid(dict(L.CASES["A_c4.5"], c_um=c))
        exact = ellipsoid_area(S.a_um, S.b_um, c)
        spacings = (0.1, 0.05, 0.035) if c == 4.5 else (0.05,)
        for h in spacings:
            t0 = time.perf_counter()
            area, nf = L.mesh_area(solid, h)
            rows.append(dict(a=S.a_um, b=S.b_um, c=c, spacing_um=h, mesh_um2=area, exact_um2=exact,
                             rel_err=(area - exact) / exact, n_faces=nf, seconds=time.perf_counter() - t0))
            log("  (2) ellipsoid %.1f x %.1f x %.1f, spacing %.3f: mesh %.3f um2, ellipsoid_area %.3f um2, "
                "rel. err %+.2e" % (S.a_um, S.b_um, c, h, area, exact, rows[-1]["rel_err"]))
    OUT["area_ellipsoid"] = rows


def check_union():
    case = L.CASES["C"]
    solid = L.make_solid(case)
    rows = []
    for h in (0.1, 0.05, 0.035):
        t0 = time.perf_counter()
        area, nf = L.mesh_area(solid, h)
        rows.append(dict(spacing_um=h, mesh_um2=area, n_faces=nf, seconds=time.perf_counter() - t0))
        log("  (3) union, spacing %.3f: mesh %.3f um2 (%d faces)" % (h, area, nf))
    # the half-height on a fine grid of columns (0.01 um) about the centre, in the soma's frame
    x0, y0, z0, a, b, c, phi, mu = solid.theta_g
    g = np.arange(-1.0, 1.0 + 1e-9, 0.01)
    cp, sp = math.cos(phi), math.sin(phi)
    U, V = np.meshgrid(g, g, indexing="ij")
    X = x0 + U * cp - V * sp
    Y = y0 + U * sp + V * cp
    H = solid.half_depth(X, Y)
    # the cone's base inside the ellipsoid: on the disc of radius c at v = 0 the ellipsoid's level >= 0
    psi = np.linspace(0, 2 * math.pi, 721)
    rr = np.linspace(0, solid.cone_apex_um * solid.cone_tan_alpha, 201)
    PU = rr[:, None] * np.sin(psi)[None, :]
    PZ = rr[:, None] * np.cos(psi)[None, :]
    lev = 1.0 - (PU / a) ** 2 - (PZ / c) ** 2
    # the depth profile at v = 0: the ellipsoid's half-depth >= the cone's (no step at the cone's base)
    uu = np.linspace(-a, a, 2001)
    hE = c * np.sqrt(np.clip(1 - (uu / a) ** 2, 0, None))
    R0 = solid.cone_apex_um * solid.cone_tan_alpha
    hC = np.sqrt(np.clip(R0 ** 2 - uu ** 2, 0, None))
    # the union's volume (column integral, 0.02 um columns) and in-plane extent
    gu = np.arange(-a - 0.1, a + 0.1, 0.02)
    gv = np.arange(-b - 0.1, solid.cone_apex_um + 0.1, 0.02)
    U2, V2 = np.meshgrid(gu, gv, indexing="ij")
    H2 = solid.half_depth(x0 + U2 * cp - V2 * sp, y0 + U2 * sp + V2 * cp)
    vol = float(2.0 * H2.sum() * 0.02 * 0.02)
    foot = H2 > 0
    OUT["union"] = dict(
        cone_apex_um=solid.cone_apex_um, cone_tan_alpha=solid.cone_tan_alpha,
        cone_half_angle_deg=math.degrees(math.atan(solid.cone_tan_alpha)), areas=rows,
        area_change_005_to_0035_rel=(rows[2]["mesh_um2"] - rows[1]["mesh_um2"]) / rows[2]["mesh_um2"],
        half_height_analytic_um=solid.half_height, half_height_grid_um=float(H.max()),
        cone_base_min_ellipsoid_level=float(lev.min()),
        v0_profile_min_hE_minus_hC_um=float(np.min(hE - hC)),
        volume_um3=vol, ellipsoid_volume_um3=4.0 / 3.0 * math.pi * a * b * c,
        ellipsoid_area_um2=ellipsoid_area(a, b, c),
        footprint_u_extent_um=[float(U2[foot].min()), float(U2[foot].max())],
        footprint_v_extent_um=[float(V2[foot].min()), float(V2[foot].max())])
    u = OUT["union"]
    log("  (3) union: half-height analytic %.4f um, on a 0.01 um grid %.4f um; cone half-angle %.2f deg; base's "
        "least ellipsoid level %.3g (>= 0: inside); min(hE - hC) at v = 0: %.3g um (>= 0: continuous)"
        % (u["half_height_analytic_um"], u["half_height_grid_um"], u["cone_half_angle_deg"],
           u["cone_base_min_ellipsoid_level"], u["v0_profile_min_hE_minus_hC_um"]))
    log("  (3) union: volume %.1f um3 (ellipsoid %.1f), footprint u %s, v %s um; area change 0.05 -> 0.035 um "
        "spacing %+.2e" % (u["volume_um3"], u["ellipsoid_volume_um3"], np.round(u["footprint_u_extent_um"], 2),
                           np.round(u["footprint_v_extent_um"], 2), u["area_change_005_to_0035_rel"]))


def check_discretisation():
    rcfg = L.renderer_config()
    solid = L.make_solid(L.CASES["A_c4.5"])
    crop = small_crop(solid, 7.5)
    z0 = solid.z0
    planes = z0 + np.array([-3.0, 0.0, 3.0])
    taus = {}
    for f, dzeta in ((1, 0.14), (2, 0.07), (4, 0.035)):
        scfg = dataclasses.replace(SomaFitConfig(), fine_factor=f, dzeta_um=dzeta)
        t0 = time.perf_counter()
        taus[f] = SomaRenderer(crop, S.p_um, planes, rcfg, scfg, S.light_direction).render(np.asarray(solid.theta_g))
        log("  (4) f %d, dzeta %.3f: rendered in %.0f s" % (f, dzeta, time.perf_counter() - t0))
    rows = []
    for (fa, fb) in ((1, 4), (2, 4), (1, 2)):
        d = taus[fa] - taus[fb]
        rows.append(dict(pair="f%d - f%d" % (fa, fb), max_abs_per_plane=np.max(np.abs(d), axis=(1, 2)),
                         rms_per_plane=np.sqrt(np.mean(d * d, axis=(1, 2))),
                         max_abs_gl_at_B110=110.0 * float(np.max(np.abs(d)))))
        log("  (4) %s: max |d tau| per plane (z0 - 3, z0, z0 + 3) %s, RMS %s"
            % (rows[-1]["pair"], np.array2string(rows[-1]["max_abs_per_plane"], precision=4),
               np.array2string(rows[-1]["rms_per_plane"], precision=2)))
    # the share of the fit's discretisation error that the truth's captures: corr and ratio of (f1 - f4), (f1 - f2)
    d14 = (taus[1] - taus[4]).ravel()
    d12 = (taus[1] - taus[2]).ravel()
    OUT["discretisation"] = dict(crop=crop, planes_minus_z0_um=planes - z0, pairs=rows,
                                 share_f1f2_of_f1f4=float(np.dot(d12, d14) / np.dot(d14, d14)))
    log("  (4) the projection of (f1 - f2) on (f1 - f4): %.3f (the share of the fit's error the f 2 truth captures)"
        % OUT["discretisation"]["share_f1f2_of_f1f4"])


if __name__ == "__main__":
    t_all = time.perf_counter()
    log("validate_check.py: the checks of the check (D-216.2), git %s" % L.git_head())
    log("(1) the check renderer against SomaRenderer, plain ellipsoid")
    check_renderer()
    log("(2) marching-cubes area against ellipsoid_area")
    check_area()
    log("(3) the union of case C")
    check_union()
    log("(4) discretisation context")
    check_discretisation()
    OUT["environment"] = L.environment()
    OUT["git"] = L.git_head()
    OUT["seconds"] = time.perf_counter() - t_all
    L.write_json(os.path.join(HERE, "results", "validation.json"), OUT)
    log("done in %.0f s; results/validation.json written" % OUT["seconds"])
