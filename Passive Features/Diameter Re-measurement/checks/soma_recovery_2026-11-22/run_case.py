"""The soma fit's recovery on a simulated stack, one case (2026-11-22; D-216.2).

    OMP_NUM_THREADS=1 python3 run_case.py CASE [--out DIR] [--smoke]

CASE one of A_c4.5, B_fit, B_fixed, C, D, A_c3.0, A_c6.0 (recovery_lib.CASES). The case's truth (an ellipsoid; for C
the ellipsoid united with a cone; for D with stems) is rendered finer than the fit (SomaRenderer, or for C and D the
check renderer, at fine_factor 2, dzeta 0.07 um), on the planes the fit takes; each plane is scaled by a drifting
background, noise is added, the stack rounded and clipped; then analysis.soma_fit.fit_soma fits it at the production
settings (the SomaFitConfig defaults, far_slope as the case says) with SomaRenderer at its defaults (fine_factor 1,
dzeta 0.14 um) as the forward model, slope0 = 0.79, from the start the pipeline takes from Allen's trace (the centre
off by (0.8, -0.6, +1.5) um, the radius 0.9 r_eq). Writes results/<CASE>.json (when the case finishes) and
figures/<CASE>.png; the log on stdout. --smoke: a fast functional run (3 planes, 3 evaluations) into --out.
Pure ASCII.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recovery_lib as L                                                  # noqa: E402
from allen_diameter.analysis import soma_fit as SF                        # noqa: E402
from allen_diameter.config import SomaFitConfig                           # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer, ellipsoid_columns  # noqa: E402

T_START = time.perf_counter()


def log(msg):
    print("[%7.1f s] %s" % (time.perf_counter() - T_START, msg), flush=True)


class LoggedForward(object):
    """A pass-through of the forward model that logs every n-th evaluation (the count, the time, theta)."""

    def __init__(self, fn, every=10):
        self.fn, self.every, self.n, self.t0 = fn, int(every), 0, time.perf_counter()

    def __call__(self, theta_g, slope):
        self.n += 1
        out = self.fn(theta_g, slope)
        if self.n == 1 or self.n % self.every == 0:
            log("  forward %4d, %.1f s/eval, theta %s, slope %.4f"
                % (self.n, (time.perf_counter() - self.t0) / self.n,
                   np.array2string(np.asarray(theta_g), precision=4, max_line_width=200), slope))
        return out


def truth_tau(case, solid, crop, z_planes, rcfg, s):
    """tau (K, height, width) of the truth on the fit's crop, at the truth's discretisation."""
    f, dzeta, light = s.truth_fine_factor, s.truth_dzeta_um, s.light_direction
    left, top, width, height = crop
    box = solid.footprint_box_xy()
    inside = (box[0] > left * s.p_um and box[1] < (left + width - 1) * s.p_um
              and box[2] > top * s.p_um and box[3] < (top + height - 1) * s.p_um)
    info = dict(fine_factor=f, dzeta_um=dzeta, footprint_box_um=box)
    if case["shape"] == "ellipsoid":
        if not inside:
            raise RuntimeError("the ellipsoid's footprint is not inside the fit's crop")
        scfg_t = dataclasses.replace(SomaFitConfig(), fine_factor=f, dzeta_um=dzeta)
        info["renderer"] = "SomaRenderer (planes in groups)"
        info["crop"] = crop
        return L.render_ellipsoid_grouped(solid.theta_g, crop, s.p_um, z_planes, rcfg, scfg_t, light,
                                          slope=case["slope_true"], acc_mb=s.truth_acc_mb, log=log), info
    if inside:
        info["renderer"] = "check renderer on the fit's crop (the solid lies inside it)"
        info["crop"] = crop
        return L.render_solid(solid, crop, s.p_um, z_planes, rcfg, f, dzeta, light, slope=case["slope_true"],
                              acc_mb=s.truth_acc_mb, log=log), info
    big = L.box_holding(crop, box, s.p_um, margin_um=1.0)
    info["renderer"] = "check renderer on a crop holding the whole solid, cut to the fit's crop"
    info["crop"] = big
    tau_big = L.render_solid(solid, big, s.p_um, z_planes, rcfg, f, dzeta, light, slope=case["slope_true"],
                             acc_mb=s.truth_acc_mb, log=log)
    r0, c0 = top - big[1], left - big[0]
    return tau_big[:, r0:r0 + height, c0:c0 + width].copy(), info


def run(name, out_dir, smoke=False):
    s = L.SETTINGS
    case = L.CASES[name]
    res_dir, fig_dir = os.path.join(out_dir, "results"), os.path.join(out_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    rcfg = L.renderer_config()
    scfg = dataclasses.replace(SomaFitConfig(), far_slope=case["far_slope"])
    if smoke:
        scfg = dataclasses.replace(scfg, plane_half_um=2.0, plane_step=4, max_nfev=3)
    solid = L.make_solid(case)
    log("case %s: %s, c %.2f um, truth's slope %.2f, far_slope %r; git %s"
        % (name, case["shape"], case["c_um"], case["slope_true"], case["far_slope"], L.git_head()))
    ref = L.soma_reference(solid)
    log("truth: c_ref %.4f um, S_ref %.3f um2 (%s), r_eq_ref %.4f um"
        % (ref["c_ref_um"], ref["S_ref_um2"], ref["S_ref_method"], ref["r_eq_ref_um"]))
    centre, r_s = L.traced_start(solid, ref)
    crop = L.fit_crop(centre[:2], scfg)
    left, top, width, height = crop
    ks_window, ks = L.stack_planes(centre[2], r_s, scfg)
    z_planes = ks * s.dz_um
    log("start: traced centre %s um, radius %.4f um; crop %s px; window k %d..%d, fitted planes %d (k %d..%d)"
        % (np.round(centre, 4), r_s, crop, ks_window[0], ks_window[-1], ks.size, ks[0], ks[-1]))

    t0 = time.perf_counter()
    tau, tinfo = truth_tau(case, solid, crop, z_planes, rcfg, s)
    t_truth = time.perf_counter() - t0
    log("truth rendered (%s) in %.0f s; tau in [%.4f, %.4f]" % (tinfo["renderer"], t_truth, tau.min(), tau.max()))
    data, B_k = L.simulate_stack(tau, ks)
    mask = L.stem_mask(solid, (height, width), left, top, s.p_um, scfg.mask_widen_um)
    region = L.region_of((height, width), crop, centre[:2], scfg, mask)
    valid = np.ones(ks.size, dtype=bool)
    i0 = int(np.argmin(np.abs(ks * s.dz_um - centre[2])))
    th0 = SF.start_theta(data[i0], left, top, s.p_um, centre, r_s, region, s.slope_cfg)
    log("stack: B_k %.2f..%.2f gl, data %d..%d gl, region %d px, masked %d px; start theta0 %s"
        % (B_k.min(), B_k.max(), data.min(), data.max(), region.sum(), mask.sum(),
           np.array2string(th0, precision=4, max_line_width=200)))

    forwards = []

    def factory(z):
        fw = LoggedForward(SomaRenderer(crop, s.p_um, z, rcfg, scfg, s.light_direction).render)
        forwards.append(fw)
        return fw

    log("fit_soma: far_slope %r, max_nfev %d, ftol %g, xtol %g" % (scfg.far_slope, scfg.max_nfev, scfg.ftol,
                                                                  scfg.xtol))
    fit = SF.fit_soma(data, ks, valid, left, top, s.p_um, s.dz_um, centre, r_s, mask, factory, scfg, s.slope_cfg)
    log("fit: %s, n_eval %d, %.0f s; %s" % (fit.status, fit.n_eval, fit.seconds, fit.message))
    log("  theta %s" % np.array2string(fit.theta, precision=5, max_line_width=200))
    log("  se    %s" % np.array2string(fit.se, precision=5, max_line_width=200))
    log("  S %.3f +- %.3f um2, r_eq %.4f um, chi2/dof %.4f, at_bound %s"
        % (fit.area_um2, fit.se_area_um2, fit.r_eq_um, fit.chi2 / fit.dof, fit.at_bound))
    err = L.fit_errors(fit, ref, case["slope_true"]) if fit.status != "failed" else None
    if err is not None:
        for key in ("c", "a", "b", "S", "r_eq"):
            e = err[key]
            log("  error %-4s %+9.4f (%+7.2f %%, %s SE)%s"
                % (key, e["err"], e["err_pct"], ("%+.2f" % e["err_se"]) if e["err_se"] is not None else "n/a",
                   "" if e["within_2se"] else "  OUTSIDE 2 SE"))

    # the fitted model, for the figure and the residual summaries
    tau_hat = SomaRenderer(crop, s.p_um, z_planes, rcfg, scfg, s.light_direction).render(fit.theta[:8],
                                                                                         float(fit.theta[8]))
    model = fit.B_gl[:, None, None] * tau_hat
    resid = (data.astype(float) - model)[:, region]
    rms_plane = np.sqrt(np.mean(resid * resid, axis=1))
    mean_plane = np.mean(resid, axis=1)

    out = dict(
        case=name, case_spec=case, question="D-216.2: the soma fit's recovery on simulated stacks",
        settings=dataclasses.asdict(s), smoke=bool(smoke),
        renderer_config=dict(kernel_table_delta_um=rcfg.kernel_table_delta_um,
                             kernel_table_sigma_um=rcfg.kernel_table_sigma_um, sigma_r0_um=rcfg.sigma_r0_um,
                             kernel_continuation=rcfg.kernel_continuation,
                             kernel_continuation_slope=rcfg.kernel_continuation_slope,
                             fft_wrap_sigmas=rcfg.fft_wrap_sigmas),
        soma_fit_config=dataclasses.asdict(scfg),
        truth=dict(ref, slope=case["slope_true"], half_height_um=solid.half_height,
                   stems=[dataclasses.asdict(st) for st in solid.stems], render=tinfo, render_seconds=t_truth),
        start=dict(centre_um=centre, radius_um=r_s, theta0_unclipped=th0, plane_k0=int(ks[i0])),
        data=dict(crop=crop, ks=ks, ks_window=[int(ks_window[0]), int(ks_window[-1])], B_k_gl=B_k,
                  region_px=int(region.sum()), masked_px=int(mask.sum()), data_min=int(data.min()),
                  data_max=int(data.max())),
        fit=dict(status=fit.status, params=SF.PARAMS, theta=fit.theta, se=fit.se, fitted=fit.fitted,
                 at_bound=fit.at_bound, n_eval=fit.n_eval, seconds=fit.seconds, chi2=fit.chi2, dof=fit.dof,
                 chi2_dof=fit.chi2 / fit.dof if fit.dof > 0 else None, n_samples=fit.n_samples,
                 area_um2=fit.area_um2, se_area_um2=fit.se_area_um2, volume_um3=fit.volume_um3,
                 r_eq_um=fit.r_eq_um,
                 se_r_eq_um=fit.se_area_um2 / (8 * math.pi * fit.r_eq_um) if fit.r_eq_um > 0 else None,
                 B_gl=fit.B_gl, B_true_gl=B_k, message=fit.message,
                 corr=(fit.cov / np.outer(fit.se, fit.se)) if np.all(np.isfinite(fit.se[:8])) else None,
                 seconds_per_eval=fit.seconds / max(1, fit.n_eval)),
        errors=err,
        residual=dict(rms_per_plane_gl=rms_plane, mean_per_plane_gl=mean_plane,
                      rms_all_gl=float(np.sqrt(np.mean(resid * resid)))),
        environment=L.environment(), git=L.git_head(), total_seconds=time.perf_counter() - T_START)
    path = os.path.join(res_dir, "%s.json" % name)
    L.write_json(path, out)
    log("wrote %s" % path)

    # the figure: the fitted planes nearest the true centre's depth + the offsets
    from recovery_figures import figure_case
    offs = np.asarray(s.figure_offsets_um)
    show = []
    for o in offs:
        j = int(np.argmin(np.abs(z_planes - (solid.z0 + o))))
        if j not in show:
            show.append(j)
    x_um = (left + np.arange(width)) * s.p_um - solid.x0
    y_um = (top + np.arange(height)) * s.p_um - solid.y0
    Xp, Yp = np.meshgrid(x_um + solid.x0, y_um + solid.y0)
    H_true = solid.half_depth(Xp, Yp)
    zl, zh, _ins = ellipsoid_columns(Xp, Yp, fit.theta[:8])
    H_fit = 0.5 * (zh - zl)
    labels = ["k %d: z - z0 = %+.2f um" % (ks[j], z_planes[j] - solid.z0) for j in show]
    e = err
    title = ("D-216.2 recovery, case %s (%s, c %.1f um, truth's slope %.2f, far slope %s): c %.3f +- %.3f um "
             "(err %+.3f um, %+.1f %%), S %.1f +- %.1f um2 (err %+.1f %%)"
             % (name, case["shape"], case["c_um"], case["slope_true"], case["far_slope"], fit.theta[5], fit.se[5],
                e["c"]["err"], e["c"]["err_pct"], fit.area_um2, fit.se_area_um2, e["S"]["err_pct"])
             if e is not None else "D-216.2 recovery, case %s: fit %s" % (name, fit.status))
    subtitle = ("truth rendered at f %d, dzeta %.2f um; fit at f 1, dzeta 0.14 um (SomaRenderer); %s, %d evaluations,"
                " %.0f s; chi2/dof %.3f (noise SD 3 gl)"
                % (s.truth_fine_factor, s.truth_dzeta_um, fit.status, fit.n_eval, fit.seconds, fit.chi2 / fit.dof))
    figure_case(os.path.join(fig_dir, "%s.png" % name), title, data[show].astype(float), model[show], region, mask,
                H_true, H_fit, np.abs(z_planes[show] - solid.z0), np.abs(z_planes[show] - fit.theta[2]), labels,
                x_um, y_um, (centre[0] - solid.x0, centre[1] - solid.y0), scfg.region_radius_um,
                4.0 * s.noise_sd_gl, subtitle=subtitle)
    log("wrote %s; total %.0f s" % (os.path.join(fig_dir, "%s.png" % name), time.perf_counter() - T_START))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", choices=list(L.CASES))
    ap.add_argument("--out", default=HERE, help="folder for results/ and figures/ (default: this check's folder)")
    ap.add_argument("--smoke", action="store_true", help="a fast functional run (not the check)")
    a = ap.parse_args(argv)
    try:
        run(a.case, a.out, smoke=a.smoke)
    except Exception:                                   # noqa: BLE001 -- recorded, then re-raised
        tb = traceback.format_exc()
        os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
        L.write_json(os.path.join(a.out, "results", "%s.error.json" % a.case),
                     dict(case=a.case, error=tb, git=L.git_head()))
        print(tb, flush=True)
        raise


if __name__ == "__main__":
    main()
