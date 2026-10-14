"""Cell 4m (D-047, the user's request of 2026-10-14, 15:35, and the answers to
three questions): each fit variant at the fitted width runs its own
Allen-guided iteration. A diagnostic: no measurement, Block 5 fit or
estimator hash changes.

For each node of a Cell 4j run, Cell 4j's iteration (plane_differences.
gradient_iteration, D-042) is run once per variant V of the fit-variant grid
(analysis.fit_variants; by default the 50 at the fitted width: 10 kernels x
unweighted and 4 centre weightings), with Cell 4j's own multiplier m and
round limit R read from its record (inherited: the lines +-m d / 2, at most R
rounds), its squares and its background: round 1's line sized from Allen's
2 r, each later line from V's fit at the plane the previous line picked
(Block 5's inputs at that plane: the profile through the pilot's fitted
centre, the tilt, B_bar of the plane; a weighted V re-weighted from the
unweighted fit of its kernel at that plane). Per V: the plane k_iter where it
stops, d_iter there, and the line that chose that plane (the last round that
picked a plane: +-m d / 2 through the SWC node) with V's fitted edges on it
(v = s + v0 -+ d_iter / 2, s the pilot's fitted centre's offset along the
line, as in Cell 4k). Cell 4j's own iteration (Block 5's fit) is redone first
with the same machinery, as the reference: it must give the record's planes
and d_iter, a difference being noted.

The fits at a plane are made once per node and shared by every variant that
visits the plane (fit_variants.PlaneFits); a square for a longer line is made
once (NodeSquares). Writes, into --out-dir, per node variter_stack_<id>.png
(one row per variant: its edges v0 -+ d/2 and the planes its rounds picked)
and variter_planes_<id>.png (each variant's chosen plane with its line and
edges and d), variter_<specimen>.csv (one row per node and variant, the
reference first) and variter_<specimen>.json.

    python scripts/variant_iteration.py --specimen 529878215 \
        --iter-json OUT/pilot/planeiter/bydiameter_m3/planeiter_529878215.json \
        --out-dir OUT/pilot/planeiter/bydiameter_m3/variter --cache-dir CACHE

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fit_variants as FVS  # noqa: E402
import iteration_fits as IF  # noqa: E402
import plane_differences as PD  # noqa: E402
from allen_diameter.analysis import fit_variants as FV  # noqa: E402
from allen_diameter.model import blur_kernels as BK  # noqa: E402
from build_table import load_config  # noqa: E402

PREFIX = "variter"
NO_INPUT = ("no_plane", "profile_nan", "bbar_nonpositive")
REFERENCE = "Block 5 (Cell 4j)"


# --------------------------------------------------------------- analysis ---

def iteration_params(rec):
    """(m, R): the line multiplier and round limit of Cell 4j's run, from its
    record (D-047: inherited). ValueError as plane_differences.check_iteration."""
    return PD.check_iteration(float(rec["line_mult"]), rec.get("max_rounds", 5))


class NodeSquares:
    """One node's squares as Cell 4j makes them: the first, node_stack with the
    record's planes and m for the Allen line and the pilot's d-line, and the
    larger ones made for longer lines. restack(h) returns the smallest square
    made so far that holds a line of half-length h (h + 2 pixels within its
    half-width, gradient_iteration's test), else makes one for h, once
    (PROVISIONAL, the assistant's: Cell 4j makes a square per request; a
    reference run first makes Cell 4j's own squares). Block 5's inputs at a
    plane come from the pilot's block, the same in every square, and are taken
    once per plane (inputs); the variants' fits there once per plane and
    variant (fit_dict, through fit_variants.PlaneFits)."""

    def __init__(self, swc, provider, cfg, rec, transform=None, half_um=None):
        nid = int(rec["node_id"])
        ks = [int(k) for k in rec["ks"]]
        k_swc = int(rec["k_swc"])
        n = k_swc - ks[0]
        if n < 0 or ks != list(range(k_swc - n, k_swc + n + 1)):
            raise ValueError("node %d: the record's planes %s are not k_SWC %d +- n" % (nid, ks, k_swc))
        self.m, self.R = iteration_params(rec)
        self.cfg = cfg
        self._args = (swc, provider, cfg, nid, transform, n, half_um, 0.0)
        self.first = PD.node_stack(*self._args, None, self.m, self.m)
        if [int(k) for k in self.first["ks"]] != ks:
            raise ValueError("node %d: the square's planes %s, the record's %s"
                             % (nid, [int(k) for k in self.first["ks"]], ks))
        self.made = [self.first]
        self._inputs, self._fits = {}, {}

    @staticmethod
    def holds(st, h):
        return float(h) + 2.0 * float(st["frame"].res_um_px) <= float(st["square_half_um"]) + 1e-9

    def restack(self, h):
        holding = [st for st in self.made if self.holds(st, h)]
        if holding:
            return min(holding, key=lambda st: float(st["square_half_um"]))
        st = PD.node_stack(*self._args, float(h), self.m, self.m)
        self.made.append(st)
        return st

    @property
    def largest(self):
        return max(self.made, key=lambda st: float(st["square_half_um"]))

    def inputs(self, k):
        """refit_at_plane(first, cfg, k, with_profile=True), once per plane."""
        if k not in self._inputs:
            self._inputs[k] = PD.refit_at_plane(self.first, self.cfg, k, with_profile=True)
        return self._inputs[k]

    def plane_fits(self, k):
        if k not in self._fits:
            inp = self.inputs(k)
            self._fits[k] = FV.PlaneFits(np.asarray(inp["v_um"], dtype=float), np.asarray(inp["profile"], dtype=float),
                                         float(inp["phi_rad"]), float(inp["B_bar"]), self.cfg.measure)
        return self._fits[k]

    def fit_dict(self, k, spec):
        """What a round of V's iteration records at plane k: d_hat_um (NaN when the
        plane gives no profile or no positive B_bar, with fit_status saying why),
        fit_status, v0_hat_um, mu_hat_per_um, B_bar, bbar_ok, fwhm_um, shape,
        at_bound, n_irls, irls_ok."""
        inp, nan = self.inputs(int(k)), float("nan")
        out = dict(d_hat_um=nan, fit_status=inp["fit_status"], v0_hat_um=nan, mu_hat_per_um=nan,
                   B_bar=inp["B_bar"], bbar_ok=inp["bbar_ok"], fwhm_um=nan, shape=None, at_bound="", n_irls=0,
                   irls_ok=None)
        if inp["fit_status"] in NO_INPUT:
            return out
        f = self.plane_fits(int(k)).fit(spec)
        out.update(d_hat_um=f.d_um, fit_status=f.status, v0_hat_um=f.v0_um, mu_hat_per_um=f.mu_per_um,
                   fwhm_um=f.fwhm_um, shape=f.shape, at_bound=";".join(f.at_bound), n_irls=f.n_irls,
                   irls_ok=f.irls_ok)
        return out


def run_iteration(sq, rec, spec=None):
    """gradient_iteration on sq's first square with Cell 4j's m, R and
    background margin, restacking through sq; spec None runs Block 5's fit
    (Cell 4j's own iteration), else variant spec's."""
    fit_at = None if spec is None else (lambda st, k: sq.fit_dict(k, spec))
    return PD.gradient_iteration(sq.first, sq.cfg, line_mult=sq.m, max_rounds=sq.R,
                                 bg_margin_um=float(rec.get("bg_margin_um", 4.0)), restack=sq.restack, fit_at=fit_at)


def run_iteration_safe(sq, rec, spec):
    """run_iteration, or, when it raises ValueError (a square the provider cannot
    make for a variant's longer line), an iteration without rounds whose status
    says why, so that the other variants still run."""
    try:
        return run_iteration(sq, rec, spec)
    except ValueError as e:
        return dict(rounds=[], status="error: %s" % e, n_rounds=0, k_first=None, k_final=None,
                    d_final_um=float("nan"), n_restacks=0, d_start_um=2.0 * float(rec["allen_radius_um"]))


def selection(ev, st):
    """The variant's outcome from its iteration ev on square st (any of the
    node's): k_iter, the fit there (d, v0), the line that chose k_iter (the last
    round that picked a plane; None when no round did), s along it, and the
    edges v = s + v0 -+ d / 2 on it (None without a finite d)."""
    picked = [x for x in ev["rounds"] if x["k"] is not None]
    r = st["result"]
    th = float(st["theta"])
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    o = np.asarray(st["o"], dtype=float)
    c = np.array([float(r.cx_um), float(r.cy_um)])
    s_um = float(np.dot(c - o, y_hat))
    out = dict(k_iter=ev["k_final"], sel_round=None, h_sel=None, d=float("nan"), v0=float("nan"), s_um=s_um,
               line_xy=None, edges_v=None, edges_xy=None, o=o, y_hat=y_hat, e_u=e_u,
               planes=[x["k"] for x in ev["rounds"]], status=ev["status"], n_rounds=ev["n_rounds"])
    if not picked:
        return out
    sel = picked[-1]
    h = float(sel["half_um"])
    d, v0 = float(sel.get("d_hat_um", float("nan"))), float(sel.get("v0_hat_um", float("nan")))
    out.update(sel_round=int(sel["round"]), h_sel=h, d=d, v0=v0, line_xy=np.array([o - h * y_hat, o + h * y_hat]),
               fit=sel)
    if math.isfinite(d) and math.isfinite(v0):
        ev_v = np.array([s_um + v0 - 0.5 * d, s_um + v0 + 0.5 * d])
        out.update(edges_v=ev_v, edges_xy=o[None, :] + ev_v[:, None] * y_hat[None, :])
    return out


def compare_reference(ev, rec):
    """Notes where the reference run differs from Cell 4j's record: the planes of
    its rounds, k_iter, d_iter (to the last bit), the status, the restacks."""
    notes = []
    planes = [x["k"] for x in ev["rounds"]]
    rec_planes = [x.get("k") for x in rec.get("rounds") or []]
    if planes != rec_planes:
        notes.append("the reference's planes %s, Cell 4j's %s" % (planes, rec_planes))
    d, d_rec = float(ev["d_final_um"]), rec.get("d_iter_um")
    d_rec = float("nan") if d_rec is None else float(d_rec)
    if not (d == d_rec or (math.isnan(d) and math.isnan(d_rec))):
        notes.append("the reference's d_iter %s um, Cell 4j's %s um" % (_f(d, "%.4f"), _f(d_rec, "%.4f")))
    if ev["status"] != rec.get("iter_status"):
        notes.append("the reference's status %s, Cell 4j's %s" % (ev["status"], rec.get("iter_status")))
    if int(ev["n_restacks"]) != int(rec.get("n_restacks", 0)):
        notes.append("the reference restacked %d times, Cell 4j %s" % (int(ev["n_restacks"]), rec.get("n_restacks")))
    return notes


# ----------------------------------------------------------------- wiring ---

def _f(x, fmt="%.2f"):
    return fmt % x if x is not None and math.isfinite(float(x)) else "n/a"


def _planes_text(planes):
    return ">".join("-" if k is None else "%d" % k for k in planes)


def variant_record(spec, ev, sel, ref_d):
    """The scalar outcome of one variant's iteration (spec None: the reference)."""
    fit = sel.get("fit") or {}
    d = sel["d"]
    row = dict(variant=REFERENCE if spec is None else spec.key,
               family="gauss" if spec is None else spec.family,
               kernel="gauss (Block 5)" if spec is None else spec.kernel_label,
               shape_mode="fixed" if spec is None else ("fitted" if spec.shape_fitted else "fixed"),
               width_mode="fixed" if spec is None else spec.width_mode,
               weight="none" if spec is None else spec.weight_label, lam_um=None if spec is None else spec.lam_um,
               k_allen=ev["k_first"], k_iter=ev["k_final"], d_iter_um=d, iter_status=ev["status"],
               n_rounds=ev["n_rounds"], planes=_planes_text(sel["planes"]), trajectory=PD.trajectory_text(ev),
               v0_um=sel["v0"], mu_per_um=fit.get("mu_hat_per_um"), fwhm_um=fit.get("fwhm_um"),
               shape=fit.get("shape"), fit_status=fit.get("fit_status"), at_bound=fit.get("at_bound"),
               n_irls=fit.get("n_irls"), irls_ok=fit.get("irls_ok"), B_bar=fit.get("B_bar"),
               sel_round=sel["sel_round"], sel_half_um=sel["h_sel"], s_um=sel["s_um"],
               edges_v_um=None if sel["edges_v"] is None else [float(x) for x in sel["edges_v"]],
               edges_xy_um=None if sel["edges_xy"] is None else sel["edges_xy"].tolist(),
               line_xy_um=None if sel["line_xy"] is None else sel["line_xy"].tolist(), n_restacks=ev["n_restacks"],
               ratio_ref=(d / ref_d) if (math.isfinite(d) and math.isfinite(ref_d) and ref_d > 0) else float("nan"))
    return row


def flag_of(row):
    """'x' a fit failed or no plane, '*' a fit parameter at its bound, '~' IRLS not settled, after FVS.flag_of."""
    s = "x" if (row["fit_status"] == "failed" or not math.isfinite(float(row["d_iter_um"]))) else (
        "*" if row["fit_status"] == "at_bound" else "")
    if row.get("irls_ok") is False and "x" not in s:
        s += "~"
    return s


def node_blocks(rec, sq, ref, rows, sels, specs, notes):
    """The two figures' blocks (figures.variant_stack_figure, figures.variant_planes_figure)."""
    from allen_diameter.plotting import figures as fg
    nid = int(rec["node_id"])
    st = sq.largest
    r = st["result"]
    ks = [int(k) for k in st["ks"]]
    styles = {kern: fg.kernel_style(kern[0], kern[2], second) for kern, _lab, second in FVS.kernel_rows(specs)}
    two_widths = len({s.width_mode for s in specs}) > 1         # then a row per kernel and width mode

    def row_key(spec):
        return spec.kernel_label + (", w %s" % spec.width_mode if two_widths else "")
    items = []
    for spec, row, sel in zip([None] + list(specs), rows, sels):
        if spec is None:
            colour, ls = fg.lam_colour(None), "-"
        else:
            colour, ls = styles[spec.kernel]
        items.append(dict(label=row["kernel"] if spec is None else "%s, %s" % (row_key(spec), spec.weight_label),
                          kernel=row["kernel"] if spec is None else row_key(spec), weight=row["weight"],
                          d=row["d_iter_um"], v0=row["v0_um"],
                          colour=colour, ls=ls, planes=sel["planes"], k_iter=row["k_iter"], status=row["iter_status"],
                          flag=flag_of(row), reference=spec is None, line_xy=sel["line_xy"], edges_xy=sel["edges_xy"],
                          h_sel=sel["h_sel"]))
    head = ("node %d: %d variants, each with its own Allen-guided iteration (m = %g, at most %d rounds, Cell 4j's); "
            "Cell 4j's (Block 5's fit) redone: k_iter %s, d %s um%s"
            % (nid, len(specs), sq.m, sq.R, _f(ref["k_iter"], "%d") if ref["k_iter"] is not None else "n/a",
               _f(ref["d"]), "; " + "; ".join(notes) if notes else ""))
    sub = ("planes %d..%d, k* %d, SWC plane %d; Allen 2r %.2f um (round 1's line +-%.2f um for every variant); "
           "edges v0 -+ d/2 from the pilot's fitted centre at each variant's own plane"
           % (ks[0], ks[-1], int(r.k_star), int(st["k_swc"]), 2.0 * float(rec["allen_radius_um"]),
              0.5 * sq.m * 2.0 * float(rec["allen_radius_um"])))
    stack = dict(label=head, sublabel=sub, rows=items, ks=ks, k_star=int(r.k_star), k_swc=int(st["k_swc"]))
    hs = [x["h_sel"] for x in items if x["h_sel"] is not None]
    view = min(max(hs + [1.0]) + 0.5, float(st["square_half_um"]))
    o = np.asarray(st["o"], dtype=float)
    panels = []
    for it in items:
        k = it["k_iter"]
        j = None if k is None else int(k) - ks[0]
        panels.append(dict(it, plane=None if j is None else np.asarray(st["stack"][j], dtype=float),
                           title="k %s, d %s%s" % ("n/a" if k is None else "%d" % k, _f(it["d"]), it["flag"])))
    planes = dict(label=head, sublabel=sub, panels=panels, extent=st["extent"], o=o, e_u=sels[0]["e_u"],
                  view=(o[0] - view, o[0] + view, o[1] + view, o[1] - view),
                  kernels=list(dict.fromkeys(row_key(sp) for sp in specs)),
                  weights=list(dict.fromkeys(s.weight_label for s in specs)))
    return stack, planes


def run(swc, provider, cfg, records, out_dir, specimen, transform=None, half_um=None, nodes=None, specs=None,
        dpi=110, log=print):
    """Per node with a plane chosen by Cell 4j: the reference and every
    variant's iteration, two figures, the rows; then the CSV and the JSON (the
    module docstring). nodes: ids (default every node of the records); specs:
    the variants (default the 50 at the fitted width). A record skipped by
    Cell 4j, without k_iter, or refused by NodeSquares is listed with its
    reason. Returns the per-node JSON records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from allen_diameter.plotting import figures as fg
    os.makedirs(out_dir, exist_ok=True)
    specs = select_specs() if specs is None else list(specs)
    want = None if not nodes else {int(x) for x in nodes}
    out, csv_rows = [], []
    for rec in records:
        nid = int(rec["node_id"])
        if want is not None and nid not in want:
            continue
        if "skipped" in rec or rec.get("k_iter") is None:
            why = ("Cell 4j: " + str(rec["skipped"])) if "skipped" in rec else (
                "no plane chosen (iteration %s)" % rec.get("iter_status"))
            log("[variter] node %d skipped: %s" % (nid, why))
            out.append(dict(node_id=nid, skipped=why))
            continue
        t0 = time.perf_counter()
        try:
            sq = NodeSquares(swc, provider, cfg, rec, transform, half_um)
            ev_ref = run_iteration(sq, rec, None)
        except ValueError as e:
            log("[variter] node %d skipped: %s" % (nid, e))
            out.append(dict(node_id=nid, skipped=str(e)))
            continue
        notes = compare_reference(ev_ref, rec)
        sel_ref = selection(ev_ref, sq.first)
        evs, sels = [ev_ref], [sel_ref]
        for spec in specs:
            ev = run_iteration_safe(sq, rec, spec)
            evs.append(ev)
            sels.append(selection(ev, sq.first))
        ref_d = sel_ref["d"]
        rows = [variant_record(s, ev, sl, ref_d) for s, ev, sl in zip([None] + specs, evs, sels)]
        rows[0]["fwhm_um"] = FV.fixed_fwhm_um(cfg.measure)          # Block 5's Gaussian at its fixed width
        sec = time.perf_counter() - t0
        stack, planes = node_blocks(rec, sq, sel_ref, rows, sels, specs, notes)
        title = "Each fit variant's own Allen-guided iteration, specimen %s" % specimen
        png_stack = os.path.join(out_dir, "%s_stack_%d.png" % (PREFIX, nid))
        png_planes = os.path.join(out_dir, "%s_planes_%d.png" % (PREFIX, nid))
        fig = fg.variant_stack_figure(stack, title)
        fig.savefig(png_stack, dpi=dpi)
        plt.close(fig)
        fig = fg.variant_planes_figure(planes, title)
        fig.savefig(png_planes, dpi=dpi)
        plt.close(fig)
        meta = dict(node_id=nid, k_swc=int(rec["k_swc"]), k_star=int(sq.first["result"].k_star),
                    line_mult=sq.m, max_rounds=sq.R, allen_2r_um=2.0 * float(rec["allen_radius_um"]),
                    d_iter_cell4j_um=rec.get("d_iter_um"), k_iter_cell4j=rec.get("k_iter"))
        for row in rows:
            csv_rows.append(dict(meta, **{k: v for k, v in row.items()
                                          if k not in ("edges_xy_um", "line_xy_um", "edges_v_um", "trajectory")}))
        out.append(dict(meta, variants=rows, notes=notes, png_stack=png_stack, png_planes=png_planes, seconds=sec,
                        n_squares=len(sq.made), n_planes_fitted=len(sq._fits)))
        log(node_log(nid, rows, notes, sec, len(sq.made), len(sq._fits)))
    if csv_rows:
        with open(os.path.join(out_dir, "%s_%s.csv" % (PREFIX, specimen)), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    with open(os.path.join(out_dir, "%s_%s.json" % (PREFIX, specimen)), "w") as f:
        json.dump(out, f, indent=1)
    done = [x for x in out if "skipped" not in x]
    same = sum(1 for x in done if not any(n.startswith("the reference") for n in x["notes"]))
    log("[variter] %d nodes, %d skipped; %d variants per node, each with its own iteration; Cell 4j's iteration "
        "redone identically on %d of %d" % (len(done), len(out) - len(done), len(specs), same, len(done)))
    return out


def node_log(nid, rows, notes, sec, n_squares, n_planes):
    ref, var = rows[0], rows[1:]
    ks = [r["k_iter"] for r in var if r["k_iter"] is not None]
    counts = {}
    for k in ks:
        counts[k] = counts.get(k, 0) + 1
    ds = [r["d_iter_um"] for r in var if math.isfinite(float(r["d_iter_um"]))]
    unw = [r for r in var if r["weight"] == "none"]
    return ("[variter] node %d: Cell 4j redone k_iter %s d %s um (%s); %d variants: k_iter %s; d_iter %s-%s um; "
            "unweighted %s; %d planes fitted, %d squares; %.0f s%s"
            % (nid, ref["k_iter"], _f(ref["d_iter_um"]), ref["iter_status"], len(var),
               ", ".join("%d x%d" % (k, n) for k, n in sorted(counts.items())) or "none",
               _f(min(ds) if ds else None), _f(max(ds) if ds else None),
               ", ".join("%s %s (k %s)" % (r["kernel"], _f(r["d_iter_um"]), r["k_iter"]) for r in unw),
               n_planes, n_squares, sec, "; " + "; ".join(notes) if notes else ""))


def select_specs(families=None, lambdas_um=None, width_modes=("fitted",)):
    """The variants: the grid at the fitted width by default (D-047: 10 kernels x
    unweighted and 4 lambdas = 50), restricted as fit_variants.select_specs
    (the Gaussian always kept)."""
    return FVS.select_specs(families, lambdas_um, width_modes)


def _csv_list(text, conv, what, ap):
    try:
        return [conv(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as e:
        ap.error("%s: %s" % (what, e))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--iter-json", required=True, help="Cell 4j's planeiter_<specimen>.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nodes", default="", help="comma-separated node ids (default: every node of the record)")
    ap.add_argument("--families", default="", help="comma-separated kernel families (default all: %s); the Gaussian "
                                                   "is always fitted" % ",".join(BK.FAMILIES))
    ap.add_argument("--lambdas", default=",".join("%g" % x for x in FV.WEIGHT_LAMBDAS_UM),
                    help="comma-separated lambdas (um) of the centre weights; empty for unweighted fits only")
    ap.add_argument("--width-modes", default="fitted", help="fitted (default, D-047), fixed or both")
    ap.add_argument("--half-um", type=float, default=None,
                    help="Cell 4j's --half-um, if it was given one (Cell 4j passes none)")
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment), as in Cell 4j")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    nodes = _csv_list(a.nodes, int, "--nodes takes comma-separated node ids", ap)
    lams = _csv_list(a.lambdas, float, "--lambdas takes comma-separated numbers", ap)
    modes = _csv_list(a.width_modes, str, "--width-modes", ap)
    if not modes or any(m not in FV.WIDTH_MODES for m in modes):
        ap.error("--width-modes takes fitted, fixed or both")
    if any(not (math.isfinite(x) and x > 0) for x in lams):
        ap.error("--lambdas must be finite and > 0")
    try:
        specs = select_specs(_csv_list(a.families, str, "--families", ap), lams, modes)
    except ValueError as e:
        ap.error(str(e))
    cfg = load_config(a.config_json)
    records = IF.read_records(a.iter_json)
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    fetcher = aio.HttpFetcher(cache_dir=a.cache_dir)
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    t0 = time.perf_counter()
    print("[variter] %d variants per node, each with its own iteration (m and the round limit from Cell 4j's "
          "record)" % len(specs), flush=True)
    run(swc, provider, cfg, records, a.out_dir, a.specimen,
        dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0), a.half_um, nodes, specs,
        log=lambda m: print(m, flush=True))
    print("[variter] files in %s; crops: %d from the cache, %d downloaded (%.1f MB); %.1f min"
          % (a.out_dir, fetcher.n_cache_hits, fetcher.n_requests, fetcher.bytes_downloaded / 1e6,
             (time.perf_counter() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
