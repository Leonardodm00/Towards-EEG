"""S3a's pooled analysis: the coverage of the phantoms' band by the real
nodes and the diameters by layer and class -- Block 17.3-17.4 in
specs/SPEC.md (D-059.1, D-065.1, D-085.5, D-087). Pure functions of node
rows and the configuration; nothing here reads or writes a file.

  node_table      the per-node CSV rows of the cells, pooled, with the cell's class and layer, S
                  recomputed from each row's columns under the current rule (D-098.3)
  selection_summary  how S was read: rows recomputed, rows whose stored in_S differs
  coverage_table  per bin of d_hat: counts, quantiles, the shares inside the analytic band (and the phantoms')
  verdict         the pass rule of D-085.5
  diameter_table  d_hat (in S) and Allen's 2r by (class, layer), by class, by layer

The analytic band of D-065.1, ln(mu_hat d_hat) in [ln mu_min + ln d_hat,
ln mu_max + ln d_hat] at the node's own d_hat, holds exactly when
mu_min <= mu_hat <= mu_max. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _true(v):
    return v is True or str(v).strip().lower() in ("true", "1")


def node_table(rows_by_cell, flags=None):
    """rows_by_cell: [(cell record with specimen_id, neuron_class, layer; the
    cell's per-node CSV rows)]. Returns one dict per node: specimen_id,
    neuron_class, layer, node_id, in_S, reject, d_hat, mu_hat, alpha, D_allen,
    and in_S_stored, s_source. in_S and reject are recomputed from the row's
    fit_status, reg_verdict and flags under the current S
    (phantoms.row_reject_reasons; `flags` overrides SELECTION_FLAGS), not read
    from the stored columns, which a CSV written under another S carries
    (D-098.3); in_S_stored is the column as written; s_source is "recomputed",
    or "stored" for a row without fit_status or flags, which keeps the stored
    in_S and reject."""
    from .phantoms import row_reject_reasons
    out = []
    for rec, rows in rows_by_cell:
        for r in rows:
            stored = _true(r.get("in_S"))
            try:
                reasons = row_reject_reasons(r, flags)
                in_s, reject, source = not reasons, ";".join(reasons), "recomputed"
            except KeyError:
                in_s, reject, source = stored, str(r.get("reject") or ""), "stored"
            out.append(dict(specimen_id=int(rec["specimen_id"]), neuron_class=str(rec.get("neuron_class") or ""),
                            layer=str(rec.get("layer") or ""), node_id=int(r["node_id"]), in_S=in_s,
                            reject=reject, d_hat=_f(r.get("d_hat_um")),
                            mu_hat=_f(r.get("mu_hat_per_um")), alpha=_f(r.get("alpha_hat")),
                            D_allen=2.0 * _f(r.get("allen_radius_um")), in_S_stored=stored, s_source=source))
    return out


def selection_summary(nodes, flags=None):
    """How S was read for the pooled nodes (D-098.3): the flags of S applied,
    the rows recomputed and the rows whose stored in_S was kept, and how many
    recomputed rows differ from their stored in_S (a CSV written under another
    S, e.g. before D-096), by direction."""
    from ..config import SELECTION_FLAGS
    rec = [n for n in nodes if n.get("s_source") == "recomputed"]
    return dict(flags=list(SELECTION_FLAGS if flags is None else flags), n_rows=len(nodes), n_recomputed=len(rec),
                n_stored_used=len(nodes) - len(rec),
                n_now_in_S=sum(1 for n in rec if n["in_S"] and not n["in_S_stored"]),
                n_now_out_of_S=sum(1 for n in rec if not n["in_S"] and n["in_S_stored"]))


def _usable(n):
    return n["in_S"] and math.isfinite(n["d_hat"]) and n["d_hat"] > 0 and math.isfinite(n["mu_hat"]) and n["mu_hat"] > 0


def bin_edges(cfg):
    """[(label, lo, hi)]: below the first edge, the edges' bins, from the last edge; each [lo, hi)."""
    e = [float(x) for x in cfg.survey.coverage_dhat_bins_um]
    full = [0.0] + e + [math.inf]
    out = []
    for lo, hi in zip(full[:-1], full[1:]):
        label = "<%g" % hi if lo == 0.0 else (">=%g" % lo if hi == math.inf else "%g-%g" % (lo, hi))
        out.append((label, lo, hi))
    return out


def _q(x):
    x = np.asarray(x, dtype=float)
    return [float(v) for v in np.quantile(x, QUANTILES)] if x.size else [float("nan")] * len(QUANTILES)


def coverage_table(nodes, cfg, phantom=None):
    """Block 17.3: one dict per bin of d_hat over the usable nodes (in S,
    d_hat > 0, mu_hat > 0, finite). phantom: optional dict of arrays d_hat,
    mud (the phantoms' rows in S), for their band per bin."""
    sv = cfg.survey
    mu_lo, mu_hi = (float(x) for x in cfg.phantom.mu_range_per_um)
    use = [n for n in nodes if _usable(n)]
    out = []
    for label, lo, hi in bin_edges(cfg):
        sel = [n for n in use if lo <= n["d_hat"] < hi]
        lnd = np.log([n["d_hat"] for n in sel]) if sel else np.zeros(0)
        lnm = np.log([n["mu_hat"] * n["d_hat"] for n in sel]) if sel else np.zeros(0)
        below = lnm < math.log(mu_lo) + lnd
        above = lnm > math.log(mu_hi) + lnd
        n = len(sel)
        share = (lambda m: float(np.mean(m)) if n else float("nan"))
        alphas = [n_["alpha"] for n_ in sel if math.isfinite(n_["alpha"]) and n_["alpha"] > 0]
        row = dict(bin=label, lo=lo, hi=hi, n=n, n_cells=len({x["specimen_id"] for x in sel}),
                   share_inside=share(~below & ~above), share_below=share(below), share_above=share(above),
                   q_ln_mud=_q(lnm), q_ln_alpha=_q(np.log(alphas)) if alphas else _q([]),
                   status="too_few" if n < sv.coverage_min_nodes else (
                       "pass" if np.mean(~below & ~above) >= sv.coverage_pass_share else "fail"))
        row.update(phantom_n=0, phantom_lo=float("nan"), phantom_hi=float("nan"),
                   share_inside_phantom=float("nan"), status_phantom="")
        if phantom is not None:
            pd_, pm = np.asarray(phantom["d_hat"], dtype=float), np.asarray(phantom["mud"], dtype=float)
            inb = (pd_ >= lo) & (pd_ < hi)
            row["phantom_n"] = int(inb.sum())
            if inb.sum() >= 20:
                plo, phi = np.quantile(np.log(pm[inb]), sv.coverage_phantom_quantiles)
                row.update(phantom_lo=float(plo), phantom_hi=float(phi))
                if n:
                    ins = float(np.mean((lnm >= plo) & (lnm <= phi)))
                    row["share_inside_phantom"] = ins
                    row["status_phantom"] = "too_few" if n < sv.coverage_min_nodes else (
                        "pass" if ins >= sv.coverage_pass_share else "fail")
            else:
                row["status_phantom"] = "no_phantoms"
        out.append(row)
    return out


def verdict(bins, cfg):
    """D-085.5's rule: every bin with at least coverage_min_nodes nodes has a
    share inside of at least coverage_pass_share (the analytic band; the
    phantoms' band reported beside it when present)."""
    sv = cfg.survey
    failing = [b["bin"] for b in bins if b["status"] == "fail"]
    out = dict(passed=not failing, failing=failing, too_few=[b["bin"] for b in bins if b["status"] == "too_few"],
               counted=[b["bin"] for b in bins if b["status"] in ("pass", "fail")],
               rule="every bin of d_hat with >= %d nodes in S has >= %.0f %% of them inside the band (D-085.5)"
                    % (sv.coverage_min_nodes, 100 * sv.coverage_pass_share),
               n_nodes=int(sum(b["n"] for b in bins)))
    ph = [b for b in bins if b.get("status_phantom") in ("pass", "fail")]
    if ph:
        out["phantom"] = dict(passed=not any(b["status_phantom"] == "fail" for b in ph),
                              failing=[b["bin"] for b in ph if b["status_phantom"] == "fail"])
    return out


def diameter_table(nodes):
    """Block 17.4: per (class, layer), per class (layer 'all'), per layer
    (class 'all'): n nodes, n in S, n cells, the quantiles of d_hat (in S,
    finite, > 0; fitted, not corrected) and of Allen's 2r (finite)."""
    keys = sorted({(n["neuron_class"], n["layer"]) for n in nodes})
    groups = [(c, l) for c, l in keys] + [(c, "all") for c in sorted({c for c, _ in keys})] \
        + [("all", l) for l in sorted({l for _, l in keys})]
    out = []
    for c, l in groups:
        sel = [n for n in nodes if (c == "all" or n["neuron_class"] == c) and (l == "all" or n["layer"] == l)]
        d = [n["d_hat"] for n in sel if n["in_S"] and math.isfinite(n["d_hat"]) and n["d_hat"] > 0]
        D = [n["D_allen"] for n in sel if math.isfinite(n["D_allen"])]
        out.append(dict(neuron_class=c, layer=l, n=len(sel), n_in_S=len(d), n_cells=len({n["specimen_id"] for n in sel}),
                        q_d_hat=_q(d), q_D_allen=_q(D)))
    return out


def outside_counts(nodes):
    """{reason: count} of the nodes outside S (a node with several reasons counts under each)."""
    out = {}
    for n in nodes:
        if n["in_S"]:
            continue
        for r in (n["reject"].split(";") if n["reject"] else ["unknown"]):
            out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items()))
