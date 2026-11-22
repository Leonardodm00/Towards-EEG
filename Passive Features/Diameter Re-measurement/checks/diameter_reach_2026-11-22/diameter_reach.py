"""Diameter reach -- how far from the soma do dendritic diameter errors still bias the somatic passive fit?

Check of 2026-11-22. The user's request (2026-11-20, 18:33): "calculate the most far position from the soma where
the effects of wrongly defined diameters has an effect", for the passive (and Ih) fits from somatic stimulation;
their answers: the linearised bias of the fitted (Cm, Rm, Ra), x* where every shift is below 5 %, every dendritic
diameter beyond x scaled by 0.8 and by 1.2, Eyal et al. 2016 cell 060308's passive parameters, no Ih, the pilot
cell 529878215's own morphology.

Model -- mirrors PassiveCell of "Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py" (Cell 7):
NEURON's Import3d of the SWC; the axon replaced by a Hay stub (2 sections, 30 um x 1 um, nseg 5) at soma(1);
`pas` everywhere; cm and g_pas multiplied by F on dendritic segments (dend, apic) whose path distance from
soma(0.5) exceeds the cutoff; nseg as imported (1 per section); IClamp at soma(0.5); fixed dt.

Targets -- y_b(q, D): the somatic voltage minus its value at stimulus onset t0, sampled on the protocol's window:
  pulse  -200 pA for 0.5 ms; t0 + 1 .. t0 + 100 ms every 0.1 ms (the fitter's training target and window)
  step   -110 pA for 1000 ms; t0 .. t0 + 1000 ms every 1 ms (the cell's first long square)
The fit's cost: C(q) = sum_b mean_n [y_b,n(q, D_model) - y_b,n(data)]^2 over the protocols of the fit (the
fitter's mean RMSD over bundles, squared per bundle: its Gauss-Newton form).

Linearised bias -- data from (q0, D); the model with D_x, every dendritic diameter beyond path distance x scaled
by s; one Gauss-Newton step from q0, q = log(Cm, Rm, Ra):
  dq(x, s) = -(J^T W J)^{-1} J^T W [y(q0, D_x) - y(q0, D)],   J = dy/dq at (q0, D) by central differences,
W = diag(1/N_b) row by row; the relative shift of parameter j is exp(dq_j) - 1.
x*_j(s) = the smallest grid distance from which on |exp(dq_j) - 1| < tol at every larger grid distance.
The step is the fit's shift to first order in the perturbation; full Gauss-Newton refits check it (--refit).

Usage:  python3 diameter_reach.py <swc> <out dir> [--refit X [X ...]] [--quick]
Pure ASCII, LF.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

PARAMS = ("Cm", "Rm", "Ra")
PROTOCOL_SETS = {"pulse": ("pulse",), "pulse+step": ("pulse", "step")}


@dataclass(frozen=True)
class ReachConfig:
    """Every number of the check. Units in the names."""
    cm_uF_cm2: float = 0.452            # Eyal et al. 2016, Table 1, cell 060308 (their Figure 1a), best fit
    rm_ohm_cm2: float = 38910.0         # same cell: 38.91 kOhm cm2
    ra_ohm_cm: float = 203.23           # same cell
    F: float = 1.9                      # the fitter's DEFAULT_F (Eyal 2016)
    spine_cutoff_um: float = 60.0       # the fitter's SPINE_PROXIMAL_CUTOFF_UM
    axon_stub: bool = True              # the fitter's axon_replacement "hay_stub"
    e_pas_mV: float = -70.0             # irrelevant to the baseline-subtracted, linear responses
    dt_ms: float = 0.025                # the fitter's DEFAULT_DT_BRIEF_MS and DEFAULT_DT_LONG_MS
    t0_ms: float = 10.0                 # stimulus onset (the fitter's pre_ms)
    protocols: Dict[str, dict] = field(default_factory=lambda: {
        "pulse": dict(amp_nA=-0.200, dur_ms=0.5, window_ms=(1.0, 100.0), sample_ms=0.1),
        "step": dict(amp_nA=-0.110, dur_ms=1000.0, window_ms=(0.0, 1000.0), sample_ms=1.0)})
    scales: Tuple[float, ...] = (0.8, 1.2)
    x_grid_um: Tuple[float, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300,
                                    350, 400, 500, 600)
    fd_step: float = 0.01               # central-difference step in q = log(parameter)
    tol: float = 0.05                   # the user's threshold on |relative shift|
    refit_max_iter: int = 10
    refit_tol: float = 1e-6             # stop when max |Gauss-Newton step| in q is below this


# ------------------------------------------------------------------ loading --

def _h():
    from neuron import h
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    h.load_file("stdrun.hoc")
    return h


def clear_all_sections():
    """Delete every NEURON section (one cell per process at a time, as the fitter's destroy())."""
    h = _h()
    for sec in list(h.allsec()):
        h.delete_section(sec=sec)


def build_cell(swc_path, cfg: ReachConfig):
    """The fitter's PassiveCell, rebuilt here (see the module docstring). Returns a dict of NEURON handles and
    per-section geometry: soma, dend (list), apic (list), axon (list), iclamp, v (recording Vector), dist_um
    (midpoint path distance from soma(0.5) of every dendritic section), d0 (each dendritic section's original
    3-d diameters), mult (the F multiplier of each segment, keyed by section and segment index)."""
    h = _h()
    clear_all_sections()
    reader = h.Import3d_SWC_read()
    reader.input(str(swc_path))
    gui = h.Import3d_GUI(reader, 0)
    gui.instantiate(None)
    soma, dend, apic, axon = [], [], [], []
    for sec in h.allsec():
        n = sec.name().lower()
        (soma if "soma" in n else apic if "apic" in n else dend if "dend" in n else axon if "axon" in n
         else dend).append(sec)
    if not soma:
        raise RuntimeError("no soma section after the SWC import")
    if cfg.axon_stub:
        for sec in list(axon):
            h.delete_section(sec=sec)
        axon = []
        for i in range(2):
            ax = h.Section(name="axon_stub[%d]" % i)
            ax.L, ax.diam, ax.nseg = 30.0, 1.0, 5
            axon.append(ax)
        axon[0].connect(soma[0](1.0), 0)
        axon[1].connect(axon[0](1.0), 0)
    for sec in h.allsec():
        sec.insert("pas")
    origin = soma[0](0.5)
    dendritic = dend + apic
    dist = np.array([h.distance(origin, sec(0.5)) for sec in dendritic])
    mult = {}
    dnames = set(sec.name() for sec in dendritic)          # names are unique after the import
    for sec in h.allsec():
        for k, seg in enumerate(sec):
            m = 1.0
            if sec.name() in dnames and h.distance(origin, seg) > cfg.spine_cutoff_um:
                m = cfg.F
            mult[(sec.name(), k)] = m
    d0 = [np.array([sec.diam3d(i) for i in range(int(sec.n3d()))]) for sec in dendritic]
    stim = h.IClamp(soma[0](0.5))
    stim.delay, stim.dur, stim.amp = 0.0, 0.0, 0.0
    v = h.Vector().record(soma[0](0.5)._ref_v)
    return dict(h=h, soma=soma, dend=dend, apic=apic, axon=axon, dendritic=dendritic, dist_um=dist, d0=d0,
                mult=mult, iclamp=stim, v=v)


# -------------------------------------------------------------------- model --

def set_passive(cell, q, cfg: ReachConfig):
    """q = log(Cm uF/cm2, Rm Ohm cm2, Ra Ohm cm); F on the dendritic segments beyond the cutoff."""
    cm, rm, ra = (float(math.exp(x)) for x in q)
    h = cell["h"]
    for sec in h.allsec():
        sec.Ra = ra
        for k, seg in enumerate(sec):
            m = cell["mult"][(sec.name(), k)]
            seg.cm = cm * m
            seg.g_pas = m / rm
            seg.e_pas = cfg.e_pas_mV


def scale_beyond(cell, x_um, s):
    """Every dendritic section whose midpoint path distance exceeds x_um: its 3-d diameters times s. Returns the
    number of sections scaled. restore_diameters() undoes it exactly."""
    h = cell["h"]
    n = 0
    for sec, d, dist in zip(cell["dendritic"], cell["d0"], cell["dist_um"]):
        if dist > x_um:
            for i in range(d.size):
                h.pt3dchange(i, float(d[i] * s), sec=sec)
            n += 1
    return n


def restore_diameters(cell):
    h = cell["h"]
    for sec, d in zip(cell["dendritic"], cell["d0"]):
        for i in range(d.size):
            h.pt3dchange(i, float(d[i]), sec=sec)


def simulate(cell, name, cfg: ReachConfig):
    """y_b: the somatic voltage minus its value at t0 on the protocol's sampling grid (mV)."""
    h, p = cell["h"], cfg.protocols[name]
    stim = cell["iclamp"]
    stim.delay, stim.dur, stim.amp = cfg.t0_ms, p["dur_ms"], p["amp_nA"]
    h.cvode.active(0)
    h.dt = cfg.dt_ms
    h.steps_per_ms = 1.0 / cfg.dt_ms
    t_end = cfg.t0_ms + p["window_ms"][1] + 2 * cfg.dt_ms
    h.tstop = t_end
    h.finitialize(cfg.e_pas_mV)
    h.continuerun(t_end)
    v = np.array(cell["v"], dtype=float)
    k0 = int(round(cfg.t0_ms / cfg.dt_ms))
    grid = np.arange(p["window_ms"][0], p["window_ms"][1] + 1e-9, p["sample_ms"])
    idx = k0 + np.round(grid / cfg.dt_ms).astype(int)
    return v[idx] - v[k0]


def targets(cell, q, names, cfg: ReachConfig):
    set_passive(cell, q, cfg)
    return {b: simulate(cell, b, cfg) for b in names}


# ----------------------------------------------------------------- analysis --

def jacobian(cell, q, names, cfg: ReachConfig):
    """dy_b/dq at q by central differences, per protocol: {b: (N_b, 3)}."""
    J = {b: np.zeros((0, 3)) for b in names}
    cols = {b: [] for b in names}
    for j in range(3):
        e = np.zeros(3)
        e[j] = cfg.fd_step
        yp = targets(cell, np.asarray(q) + e, names, cfg)
        ym = targets(cell, np.asarray(q) - e, names, cfg)
        for b in names:
            cols[b].append((yp[b] - ym[b]) / (2 * cfg.fd_step))
    set_passive(cell, q, cfg)
    for b in names:
        J[b] = np.column_stack(cols[b])
    return J


def gauss_newton_step(J, r, names):
    """-(J^T W J)^{-1} J^T W r with W = 1/N_b per row of protocol b (the cost sum_b mean_n r^2)."""
    A = np.zeros((3, 3))
    g = np.zeros(3)
    for b in names:
        w = 1.0 / J[b].shape[0]
        A += w * J[b].T @ J[b]
        g += w * J[b].T @ r[b]
    return -np.linalg.solve(A, g)


def refit(cell, q_start, data, names, cfg: ReachConfig):
    """Gauss-Newton on the cost against `data` with the cell's current (perturbed) diameters. -> (q, n_iter)."""
    q = np.asarray(q_start, dtype=float).copy()
    for it in range(1, cfg.refit_max_iter + 1):
        y = targets(cell, q, names, cfg)
        J = jacobian(cell, q, names, cfg)
        step = gauss_newton_step(J, {b: y[b] - data[b] for b in names}, names)
        q = q + step
        if np.max(np.abs(step)) < cfg.refit_tol:
            return q, it
    return q, cfg.refit_max_iter


def membrane_beyond(cell, x_um):
    """(shaft area beyond x / all dendritic shaft area, F-weighted membrane beyond x / the whole cell's
    F-weighted membrane)."""
    h = cell["h"]
    shaft = np.array([sum(seg.area() for seg in sec) for sec in cell["dendritic"]])
    eff = np.array([sum(seg.area() * cell["mult"][(sec.name(), k)] for k, seg in enumerate(sec))
                    for sec in cell["dendritic"]])
    other = sum(seg.area() for sec in cell["soma"] + cell["axon"] for seg in sec)
    far = cell["dist_um"] > x_um
    return float(shaft[far].sum() / shaft.sum()), float(eff[far].sum() / (eff.sum() + other))


def x_star(xs, rel, tol):
    """The smallest grid x from which on |rel| < tol at every larger x (NaN if never)."""
    xs, rel = np.asarray(xs, float), np.abs(np.asarray(rel, float))
    for i in range(xs.size):
        if np.all(rel[i:] < tol):
            return float(xs[i])
    return float("nan")


def match_rin_tau(swc, cfg: ReachConfig, rin_MOhm, tau_ms, log=print, max_iter=8):
    """Rm (and Cm = tau / Rm) such that the model's R_in after the 1 s step equals rin_MOhm, Ra kept: the secant
    method on log Rm. -> a ReachConfig with the matched Cm, Rm."""
    from dataclasses import replace
    cell = build_cell(swc, cfg)

    def rin(rm):
        q = np.log([tau_ms * 1e-3 / rm * 1e6, rm, cfg.ra_ohm_cm])     # Cm in uF/cm2 = tau (s) / Rm * 1e6
        return targets(cell, q, ("step",), cfg)["step"][-1] / cfg.protocols["step"]["amp_nA"]

    a, b = math.log(cfg.rm_ohm_cm2), math.log(cfg.rm_ohm_cm2 / 2.0)
    fa, fb = math.log(rin(math.exp(a))) - math.log(rin_MOhm), math.log(rin(math.exp(b))) - math.log(rin_MOhm)
    for _ in range(max_iter):
        if abs(fb) < 1e-4:
            break
        a, b, fa = b, b - fb * (b - a) / (fb - fa), fb
        fb = math.log(rin(math.exp(b))) - math.log(rin_MOhm)
    rm = math.exp(b)
    log("matched: Rm %.0f Ohm cm2, Cm %.3f uF/cm2 give R_in %.2f MOhm (target %.2f) and tau = Rm Cm = %.1f ms"
        % (rm, tau_ms * 1e3 / rm, math.exp(fb) * rin_MOhm, rin_MOhm, tau_ms))
    return replace(cfg, rm_ohm_cm2=rm, cm_uF_cm2=tau_ms * 1e3 / rm)


def run(swc, out_dir, cfg: ReachConfig, refit_at=(), refit_scales=None, log=print, label="eyal2016_060308"):
    t_all = time.time()
    names = tuple(cfg.protocols)
    cell = build_cell(swc, cfg)
    q0 = np.log([cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm])
    n_d = len(cell["dendritic"])
    log("cell: %d dendritic sections (%d basal, %d apical), path distance max %.0f um; soma %.2f um; axon %s"
        % (n_d, len(cell["dend"]), len(cell["apic"]), cell["dist_um"].max(), cell["soma"][0].L,
           "Hay stub" if cfg.axon_stub else "as reconstructed"))
    t = time.time()
    data = targets(cell, q0, names, cfg)
    r_in = data["step"][-1] / (cfg.protocols["step"]["amp_nA"])       # mV / nA = MOhm (after 1 s)
    log("reference (Eyal 2016 cell 060308: Cm %.3f uF/cm2, Rm %.0f Ohm cm2, Ra %.1f Ohm cm; F %.1f beyond %.0f um):"
        " R_in(1 s) %.1f MOhm; pulse peak %.3f mV; %.1f s"
        % (cfg.cm_uF_cm2, cfg.rm_ohm_cm2, cfg.ra_ohm_cm, cfg.F, cfg.spine_cutoff_um, r_in,
           float(np.min(data["pulse"])), time.time() - t))
    t = time.time()
    J = jacobian(cell, q0, names, cfg)
    log("Jacobian (central differences, step %.3g in log p): %.1f s" % (cfg.fd_step, time.time() - t))
    rows = []
    for s in cfg.scales:
        for x in cfg.x_grid_um:
            n = scale_beyond(cell, x, s)
            y = targets(cell, q0, names, cfg)
            restore_diameters(cell)
            dy = {b: y[b] - data[b] for b in names}
            shaft, eff = membrane_beyond(cell, x)
            row = dict(scale=s, x_um=x, n_scaled=n, shaft_area_beyond=shaft, membrane_beyond=eff,
                       rmsd_pulse_mV=float(np.sqrt(np.mean(dy["pulse"] ** 2))),
                       rmsd_step_mV=float(np.sqrt(np.mean(dy["step"] ** 2))))
            for key, sub in PROTOCOL_SETS.items():
                dq = gauss_newton_step(J, dy, sub)       # the fit's shift: -(J^T W J)^-1 J^T W dy
                for j, p in enumerate(PARAMS):
                    row["%s_%s" % (p, key)] = float(math.exp(dq[j]) - 1.0)
            rows.append(row)
        log("scale %.2f done (%d distances)" % (s, len(cfg.x_grid_um)))
    stars = {}
    for s in cfg.scales:
        sub_rows = [r for r in rows if r["scale"] == s]
        xs = [r["x_um"] for r in sub_rows]
        for key in PROTOCOL_SETS:
            for p in PARAMS:
                stars["%.2f|%s|%s" % (s, key, p)] = x_star(xs, [r["%s_%s" % (p, key)] for r in sub_rows], cfg.tol)
    checks = []
    for x in refit_at:
        for s in (cfg.scales if not refit_scales else refit_scales):
            t = time.time()
            scale_beyond(cell, x, s)
            for key, sub in PROTOCOL_SETS.items():
                qh, it = refit(cell, q0, {b: data[b] for b in sub}, sub, cfg)
                lin = [r for r in rows if r["scale"] == s and r["x_um"] == x]
                checks.append(dict(x_um=x, scale=s, fit=key, iterations=it,
                                   **{"%s_refit" % p: float(math.exp(qh[j] - q0[j]) - 1.0) for j, p in enumerate(PARAMS)},
                                   **{"%s_linear" % p: (lin[0]["%s_%s" % (p, key)] if lin else float("nan"))
                                      for p in PARAMS}))
            restore_diameters(cell)
            log("refit at x %.0f um, scale %.2f: %.1f s" % (x, s, time.time() - t))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diameter_reach_%s.csv" % label), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")   # LF, as every file here
        w.writeheader()
        w.writerows(rows)
    summary = dict(label=label, config={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                   swc=os.path.basename(str(swc)), r_in_1s_MOhm=r_in, x_star_um=stars, refit_checks=checks,
                   seconds=time.time() - t_all)
    with open(os.path.join(out_dir, "diameter_reach_%s.json" % label), "w") as f:
        json.dump(summary, f, indent=1)
    return rows, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("swc")
    ap.add_argument("out_dir")
    ap.add_argument("--refit", type=float, nargs="*", default=())
    ap.add_argument("--refit-scales", type=float, nargs="*", default=None)
    ap.add_argument("--no-grid", action="store_true", help="refits only (the grid reduced to the refit distances)")
    ap.add_argument("--quick", action="store_true", help="a short distance grid (for a trial run)")
    ap.add_argument("--match-rin-mohm", type=float, default=None, help="set Rm so that the model's R_in is this")
    ap.add_argument("--match-tau-ms", type=float, default=None, help="with --match-rin-mohm: Cm = tau / Rm")
    ap.add_argument("--label", default="eyal2016_060308")
    a = ap.parse_args(argv)
    from dataclasses import replace
    cfg = ReachConfig()
    if a.quick:
        cfg = replace(cfg, x_grid_um=(0, 50, 100, 200, 400))
    if a.no_grid:
        cfg = replace(cfg, x_grid_um=tuple(a.refit))
    if a.match_rin_mohm is not None:
        cfg = match_rin_tau(a.swc, cfg, a.match_rin_mohm, a.match_tau_ms)
    rows, summary = run(a.swc, a.out_dir, cfg, refit_at=a.refit, refit_scales=a.refit_scales, label=a.label)
    print("x* (um): the smallest distance beyond which scaling every dendritic diameter by s shifts the fitted "
          "parameter by less than %.0f %%" % (100 * cfg.tol))
    for k, v in summary["x_star_um"].items():
        s_, fit_, par_ = k.split("|")
        print("  s %s  fit %-10s  %-2s  x* = %s" % (s_, fit_, par_, ("%.0f" % v) if v == v else "never"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
