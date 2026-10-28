"""Smoke test for S3a, the real-cell survey: the node sample, the
measurement, the coverage of the phantoms' band, the diameters by layer and
class -- Block 17 in specs/SPEC.md (D-059.1, D-065.1, D-085.4-5, D-087;
2026-10-26).

Checks
    test_known_answer   the stratified sample on an SWC of known stretches:
                        min(N, available) per bin of Allen's 2r, round-robin
                        over the stretches; `all` = every dendrite node; the
                        coverage on hand-made nodes: shares, n cells,
                        quantiles and the pass rule as computed by hand (19
                        nodes outside: too_few; 20 with 1 outside: 0.95,
                        passes; 2 outside: fails); S recomputed from a row's
                        columns (D-098.3): a row written before D-096 with
                        the flag faint (in_S False, reject faint) is in S,
                        crossing, a fit at a bound and a registration other
                        than ON are out
    test_reference      the sample equals an independent re-implementation;
                        the analytic share equals the share from the mu-range
                        rule; the quantiles numpy's; the phantoms' band
                        numpy's quantiles of their rows
    test_convergence    not a discretisation: skipped
    test_invariants     the rows' order and nodes outside S change nothing;
                        the stored in_S and reject change nothing when the
                        row has fit_status and flags (D-098.3)
    test_contract       measure_nodes(whole_stretches=False) measures exactly
                        the ids given, True the whole stretch; the step
                        `nodes` end to end on the synthetic cell of
                        fixtures_cell.py (through fetch_zblock): every
                        sampled node measured, CSV, summary, background
                        statistics; `pool --what coverage` over it: CSVs,
                        JSON keys, figures; the stored row's S equals
                        reject_reasons of its NodeResult through the CSV
                        round trip, and a CSV rewritten as an older run's
                        (in_S False, reject faint) is pooled with its S
                        recomputed (D-098.3)
    test_determinism    the sample twice with the seed; another seed changes
                        a bin with more than N candidates
    test_edge_cases     an empty bin, a NaN radius, a cell without nodes in S,
                        the configuration's refusals; a row without
                        fit_status or flags keeps its stored S; empty and
                        NaN cells read as no flag and no registration

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_s3a.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import survey_cells as SC  # noqa: E402
from allen_diameter.analysis import cell as cellmod  # noqa: E402
from allen_diameter.analysis import coverage as COV  # noqa: E402
from allen_diameter.analysis import phantoms  # noqa: E402
from allen_diameter.analysis import s3_nodes as SN  # noqa: E402
from allen_diameter.analysis import survey  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import cell_pool, survey_io, swc_io, table_io  # noqa: E402

SEED = 20261026
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib")

# stretches from the soma: (number of nodes, radius um) per segment of the stretch
STRETCHES = (((30, 0.15),), ((30, 0.25),), ((10, 0.15),), ((5, 0.15), (5, 1.5)), ((8, float("nan")),))


def write_tree(path, stretches=STRETCHES):
    """An SWC whose stretches leave the soma (each one a run of cell.stretches) with the given radii."""
    lines, nid = ["1 1 0 0 0 5 -1"], 2
    for s, segs in enumerate(stretches):
        par, a = 1, 2 * math.pi * s / len(stretches)
        x, y = 0.0, 0.0
        for n, r in segs:
            for _ in range(n):
                x, y = x + math.cos(a), y + math.sin(a)
                lines.append("%d %d %.3f %.3f 1.0 %s %d" % (nid, 3 if s % 2 else 4, x, y, "nan" if r != r else "%.3f" % r, par))
                par, nid = nid, nid + 1
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def tree(tmp):
    return swc_io.read_swc(write_tree(os.path.join(tmp, "tree.swc")))


def stretch_of(swc, cfg):
    out = {}
    for k, run in enumerate(cellmod.stretches(swc, cfg.acquisition.dendrite_swc_types)):
        for r in run:
            out[int(swc.ids[r])] = k
    return out


def node(sid, cls, layer, d, mu, in_S=True, alpha=None, D=0.5, nid=0, reject=""):
    return dict(specimen_id=sid, neuron_class=cls, layer=layer, node_id=nid, in_S=in_S, reject=reject, d_hat=d,
                mu_hat=mu, alpha=mu * d if alpha is None else alpha, D_allen=D)


def bin_nodes(n_in, n_out, d=0.5, sid0=1):
    """n_in nodes inside the analytic band (mu 1 /um) and n_out above it (mu 5 /um), at d_hat = d."""
    return [node(sid0 + (i % 3), "exc", "3", d, 1.0, nid=i) for i in range(n_in)] + \
        [node(sid0 + (i % 3), "exc", "3", d, 5.0, nid=1000 + i) for i in range(n_out)]


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    tmp = tempfile.mkdtemp()
    swc = tree(tmp)
    ids, counts = SN.stratified_nodes(swc, cfg, 7)
    # bins of 2r: [0, .4) holds 30 + 10 + 5 nodes of radius 0.15, [.4, .6) the 30 of 0.25, [2, inf) the 5 of 1.5
    assert counts == {0: (45, 20), 1: (30, 20), 5: (5, 5)}, counts
    assert len(ids) == 45 and len(set(ids)) == 45
    st = stretch_of(swc, cfg)
    rad = {int(i): float(r) for i, r in zip(swc.ids, swc.radius)}
    for b, lo, hi in ((0, 0.0, 0.4), (1, 0.4, 0.6), (5, 2.0, math.inf)):
        taken = [i for i in ids if lo <= 2 * rad[i] < hi]
        per, avail = {}, {}
        for i in taken:
            per[st[i]] = per.get(st[i], 0) + 1
        for i, r in rad.items():
            if i in st and r == r and lo <= 2 * r < hi:
                avail[st[i]] = avail.get(st[i], 0) + 1
        open_ = [per.get(s, 0) for s in avail if per.get(s, 0) < avail[s]]    # stretches not exhausted
        if open_:
            assert max(per.values()) - min(open_) <= 1, (b, per, avail)    # round-robin: within one of each other
    assert sorted(SN.all_nodes(swc, cfg)) == sorted(int(i) for i, t in zip(swc.ids, swc.types) if t in (3, 4))
    # the coverage on hand-made nodes
    nodes = bin_nodes(19, 0, d=0.3) + bin_nodes(19, 1, d=0.5) + bin_nodes(18, 2, d=0.7) + bin_nodes(0, 19, d=1.0)
    bins = {b["bin"]: b for b in COV.coverage_table(nodes, cfg)}
    assert bins["0.2-0.4"]["n"] == 19 and bins["0.2-0.4"]["status"] == "too_few"
    assert bins["0.4-0.6"]["share_inside"] == 0.95 and bins["0.4-0.6"]["status"] == "pass"
    assert abs(bins["0.6-0.8"]["share_inside"] - 0.9) < 1e-12 and bins["0.6-0.8"]["status"] == "fail"
    assert abs(bins["0.6-0.8"]["share_above"] - 0.1) < 1e-12 and bins["0.6-0.8"]["share_below"] == 0.0
    assert bins["0.8-1.2"]["status"] == "too_few" and bins["0.8-1.2"]["share_inside"] == 0.0
    assert bins["0.4-0.6"]["n_cells"] == 3 and bins["<0.2"]["n"] == 0
    assert list(bins) == ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.2", "1.2-2", "2-4", ">=4"], list(bins)
    far = {b["bin"]: b["n"] for b in COV.coverage_table([node(1, "exc", "3", 5.0, 1.0), node(1, "exc", "3", 0.1, 1.0),
                                                         node(1, "exc", "3", 4.0, 1.0)], cfg)}
    assert far[">=4"] == 2 and far["<0.2"] == 1, far                    # [4, inf) holds 4.0 and 5.0
    # node_table parses the CSV's strings: in_S "True"/"False", numbers, Allen's radius doubled
    rows = [dict(node_id="3", in_S="False", reject="dark", d_hat_um="0.5", mu_hat_per_um="1.0", alpha_hat="0.6",
                 allen_radius_um="0.25"), dict(node_id="4", in_S=True, reject="", d_hat_um="", mu_hat_per_um="2",
                                               alpha_hat="nan", allen_radius_um="0.3")]
    tab = COV.node_table([(dict(specimen_id=9, neuron_class="inh", layer="5"), rows)])
    assert [n["in_S"] for n in tab] == [False, True] and tab[0]["D_allen"] == 0.5 and tab[0]["reject"] == "dark"
    assert tab[1]["d_hat"] != tab[1]["d_hat"] and tab[0]["specimen_id"] == 9 and tab[0]["layer"] == "5"
    assert [n["s_source"] for n in tab] == ["stored", "stored"]          # no fit_status, flags: the stored S kept
    # D-098.3: S recomputed from the row's own columns. Rows as a run before D-096 wrote them: the flag faint
    # failed S then (in_S False, reject faint), and is a label now; crossing, a fit at a bound and a registration
    # other than ON still fail S, in reject_reasons' order (status, registration, the flags of S)
    old = [dict(node_id=1, fit_status="converged", reg_verdict="", flags="faint;iter_max_rounds", in_S=False,
                reject="faint", d_hat_um=0.5, mu_hat_per_um=1.0, alpha_hat=0.2, allen_radius_um=0.25),
           dict(node_id=2, fit_status="converged", reg_verdict="", flags="faint;crossing", in_S=False,
                reject="faint;crossing", d_hat_um=0.5, mu_hat_per_um=1.0, alpha_hat=0.2, allen_radius_um=0.25),
           dict(node_id=3, fit_status="at_bound", reg_verdict="FAR: nearest matching ridge 2.1 um away", flags="faint",
                in_S=False, reject="status:at_bound;registration:FAR;faint", d_hat_um=0.5, mu_hat_per_um=1.0, alpha_hat=0.2,
                allen_radius_um=0.25),
           dict(node_id=4, fit_status="converged", reg_verdict="ON THE PROCESS (lateral offset +0.05 um)", flags="dark",
                in_S=True, reject="",
                d_hat_um=0.5, mu_hat_per_um=1.0, alpha_hat=1.3, allen_radius_um=0.25)]
    tab = COV.node_table([(dict(specimen_id=9, neuron_class="inh", layer="5"), old)])
    assert [n["in_S"] for n in tab] == [True, False, False, True], [n["in_S"] for n in tab]
    assert [n["reject"] for n in tab] == ["", "crossing", "status:at_bound;registration:FAR", ""], tab
    assert [n["in_S_stored"] for n in tab] == [False, False, False, True]
    summ = COV.selection_summary(tab)
    assert (summ["n_rows"], summ["n_recomputed"], summ["n_stored_used"], summ["n_now_in_S"], summ["n_now_out_of_S"]) \
        == (4, 4, 0, 1, 0), summ
    assert summ["flags"] == ["crossing", "stack_edge", "bbar_few", "profile_nan"], summ["flags"]
    assert COV.outside_counts(tab) == {"crossing": 1, "registration:FAR": 1, "status:at_bound": 1}
    v = COV.verdict(list(bins.values()), cfg)
    assert v["passed"] is False and v["failing"] == ["0.6-0.8"] and set(v["too_few"]) >= {"0.2-0.4", "0.8-1.2"}
    v2 = COV.verdict([b for b in bins.values() if b["bin"] != "0.6-0.8"], cfg)
    assert v2["passed"] is True
    # the diameters by layer and class
    nodes = [node(1, "exc", "2", 0.5, 1.0, D=0.4), node(1, "exc", "2", 0.7, 1.0, D=0.6, in_S=False),
             node(2, "inh", "3", 0.9, 1.0, D=1.0), node(3, "sparsely_spiny", "3", 1.1, 1.0, D=1.2)]
    dia = {(d["neuron_class"], d["layer"]): d for d in COV.diameter_table(nodes)}
    assert dia[("exc", "2")]["n"] == 2 and dia[("exc", "2")]["n_in_S"] == 1 and dia[("exc", "2")]["q_d_hat"][2] == 0.5
    assert dia[("exc", "2")]["q_D_allen"][2] == 0.5 and dia[("all", "3")]["n_cells"] == 2
    assert dia[("sparsely_spiny", "all")]["n"] == 1


def independent_sample(swc, cfg, sid):
    """Block 17.1 written again: per bin, stretches and nodes permuted by default_rng([seed, sid, b]), round-robin."""
    edges = list(cfg.survey.nodes_allen_bins_um)
    runs = {}
    for run in cellmod.stretches(swc, cfg.acquisition.dendrite_swc_types):
        per = {}
        for r in run:
            rr = float(swc.radius[r])
            if rr == rr and rr >= 0:
                b = sum(1 for e in edges if 2 * rr >= e)
                per.setdefault(b, []).append(int(swc.ids[r]))
        for b, ids in per.items():
            runs.setdefault(b, []).append(ids)
    out = {}
    for b, lists in runs.items():
        g = np.random.default_rng([cfg.survey.selection_seed, sid, b])
        lists = [lists[i] for i in g.permutation(len(lists))]
        lists = [[ids[j] for j in g.permutation(len(ids))] for ids in lists]
        taken, depth = [], 0
        while len(taken) < min(cfg.survey.nodes_per_bin, sum(len(x) for x in lists)):
            for ids in lists:
                if depth < len(ids) and len(taken) < cfg.survey.nodes_per_bin:
                    taken.append(ids[depth])
            depth += 1
        out[b] = taken
    return out


def test_reference():
    cfg = default_config()
    tmp = tempfile.mkdtemp()
    swc = tree(tmp)
    ids, counts = SN.stratified_nodes(swc, cfg, 7)
    ref = independent_sample(swc, cfg, 7)
    assert ids == [i for b in sorted(ref) for i in ref[b]], "the sample must equal the independent re-implementation"
    # a bin's draw does not depend on another bin: without the 0.25 um stretch, bins 0 and 5 are unchanged
    swc2 = swc_io.read_swc(write_tree(os.path.join(tmp, "tree2.swc"),
                                      tuple(s for s in STRETCHES if s != ((30, 0.25),))))
    ids2, counts2 = SN.stratified_nodes(swc2, cfg, 7)
    assert 1 not in counts2 and counts2[0] == counts[0]
    rng = np.random.default_rng(SEED)
    nodes = [node(int(rng.integers(1, 5)), "exc", "3", float(np.exp(rng.uniform(np.log(0.15), np.log(5)))),
                  float(np.exp(rng.uniform(np.log(0.05), np.log(10)))), alpha=float(np.exp(rng.normal())), nid=i)
             for i in range(600)]
    mu_lo, mu_hi = cfg.phantom.mu_range_per_um
    for b in COV.coverage_table(nodes, cfg):
        sel = [n for n in nodes if b["lo"] <= n["d_hat"] < b["hi"]]
        if not sel:
            continue
        direct = np.mean([mu_lo <= n["mu_hat"] <= mu_hi for n in sel])        # the mu-range rule
        assert abs(b["share_inside"] - direct) < 1e-12, (b["bin"], b["share_inside"], direct)
        q = np.quantile(np.log([n["mu_hat"] * n["d_hat"] for n in sel]), COV.QUANTILES)
        assert np.allclose(b["q_ln_mud"], q, rtol=0, atol=1e-12)
    ph = dict(d_hat=np.exp(rng.uniform(np.log(0.2), np.log(4), 5000)), mud=np.exp(rng.normal(0, 1, 5000)))
    for b in COV.coverage_table(nodes, cfg, ph):
        inb = (ph["d_hat"] >= b["lo"]) & (ph["d_hat"] < b["hi"])
        if inb.sum() >= 20:
            lo, hi = np.quantile(np.log(ph["mud"][inb]), cfg.survey.coverage_phantom_quantiles)
            assert abs(b["phantom_lo"] - lo) < 1e-12 and abs(b["phantom_hi"] - hi) < 1e-12
            sel = np.log([n["mu_hat"] * n["d_hat"] for n in nodes if b["lo"] <= n["d_hat"] < b["hi"]])
            if sel.size:
                assert abs(b["share_inside_phantom"] - np.mean((sel >= lo) & (sel <= hi))) < 1e-12
        else:
            assert b["status_phantom"] == "no_phantoms" and b["phantom_lo"] != b["phantom_lo"]


def test_convergence():
    raise unittest.SkipTest("no discretisation: a sample and counts")


def test_invariants():
    cfg = default_config()
    nodes = bin_nodes(30, 2, d=0.5) + bin_nodes(25, 0, d=1.5)
    ref = COV.coverage_table(nodes, cfg)
    rng = np.random.default_rng(SEED)
    shuffled = [nodes[i] for i in rng.permutation(len(nodes))]
    noise = [node(9, "inh", "5", 0.5, 50.0, in_S=False, reject="dark") for _ in range(40)]
    for other in (shuffled, nodes + noise):
        got = COV.coverage_table(other, cfg)
        assert json.dumps(survey_io.clean(got), sort_keys=True) == json.dumps(survey_io.clean(ref), sort_keys=True)
    # D-098.3: with fit_status and flags present, the stored in_S and reject change nothing
    rows = [dict(node_id=k, fit_status=("converged", "failed")[k % 2], reg_verdict="", flags=("faint", "crossing", "")[k % 3],
                 d_hat_um=0.5, mu_hat_per_um=1.0, alpha_hat=0.2, allen_radius_um=0.25) for k in range(12)]
    rec = dict(specimen_id=5, neuron_class="exc", layer="3")
    a = COV.node_table([(rec, [dict(r, in_S=True, reject="") for r in rows])])
    b = COV.node_table([(rec, [dict(r, in_S=False, reject="faint;whatever") for r in rows])])
    strip = lambda t: [{k: v for k, v in n.items() if k != "in_S_stored"} for n in t]
    assert strip(a) == strip(b)


class CellSource:
    """The Allen source of Block 16.2 over fixtures_cell's synthetic cell."""

    def __init__(self, swc_path, fetcher, planes):
        self._swc, self.fetcher, self._planes = swc_path, fetcher, planes

    def swc_path(self, sid):
        return self._swc

    def images(self, sid):
        raise AssertionError("the nodes step reads the planes, not the image table")

    def planes(self, sid):
        return self._planes


def test_contract():
    import fixtures_cell
    cfg = default_config()
    tmp = tempfile.mkdtemp()
    swc, fetcher, planes, swc_path = fixtures_cell.synthetic_cell(tmp, cfg)
    import run_cell
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    one = survey.measure_nodes(swc, provider, cfg, only={4}, whole_stretches=False)
    assert [r.node_id for r, _, _, _ in one] == [4], [r.node_id for r, _, _, _ in one]
    whole = survey.measure_nodes(swc, provider, cfg, only={4})
    assert sorted(r.node_id for r, _, _, _ in whole) == list(range(2, 8))
    assert one[0][0].d_hat_um == next(r for r, _, _, _ in whole if r.node_id == 4).d_hat_um
    # D-098.3, reference: the S of a stored row equals reject_reasons of its NodeResult, through the CSV round trip
    # (survey.pilot_row, table_io.write_rows, read_rows), for every status, registration verdict and flag mix tried
    variants = []
    for res, r_allen, _br, _i in whole[:3]:
        for st in ("converged", "at_bound", "failed"):
            for reg in ("", "ON THE PROCESS (lateral offset +0.00 um)", "ALONGSIDE: a ridge 0.4 um off -- snap "
                        "needed before measuring", "NOT ON A VISIBLE PROCESS (no ridge)"):
                for fl in ((), ("faint",), ("faint", "crossing"), ("dark", "stack_edge", "iter_max_rounds")):
                    variants.append((dataclasses.replace(res, fit_status=st, reg_verdict=reg, flags=fl), r_allen))
    trip = os.path.join(tmp, "trip.csv")
    table_io.write_rows([survey.pilot_row(v, r_allen, cfg) for v, r_allen in variants], trip)
    back = table_io.read_rows(trip)
    assert len(back) == len(variants) == 144
    for (v, _), row in zip(variants, back):
        assert phantoms.row_reject_reasons(row) == phantoms.reject_reasons(v), (row["fit_status"], row["flags"])
    # the step `nodes` through the runner (align from a manual identity alignment), then the pooled coverage
    out = os.path.join(tmp, "survey")
    rows = [dict(specimen_id=11, species="Homo Sapiens", layer="3", dendrite_type="spiny", reconstruction_type="full",
                 area="MTG", donor_id="9", hemisphere="left", normalized_depth=0.3, neuron_class="exc", group="exc|3",
                 rank=0)]
    sel = os.path.join(tmp, "selection.csv")
    cell_pool.write_table(sel, rows, cell_pool.SELECTION_COLUMNS)
    cell = survey_io.CellDir(out, 11)
    survey_io.write_json(cell.path_of("align", "alignment_manual.json"),
                         dict(shift_x_px=0.0, shift_y_px=0.0, flip_h_px=None, z0_um=0.0))
    src = CellSource(swc_path, fetcher, planes)
    logs = []
    st = SC.run(cfg, sel, out, src, ["align", "nodes"], log=logs.append)
    assert st[0]["align"] == "done" and st[0]["nodes"] == "done", (st, logs)
    got = table_io.read_rows(cell.path_of("nodes", "pilot_11.csv"))
    assert sorted(int(r["node_id"]) for r in got) == list(range(2, 8))          # 2r = 0.6: one bin, all 6 nodes
    for col in ("d_hat_um", "mu_hat_per_um", "alpha_hat", "in_S", "allen_radius_um", "bg_median_gl", "bg_clipped_sd_gl"):
        assert col in got[0], col
    summ = survey_io.read_json(cell.path_of("nodes", "pilot_summary_11.json"))
    assert summ["mode"] == "stratified" and summ["n_sampled"] == 6 and "background" in summ
    assert summ["sample_counts"] == {"2": [6, 6]}, summ["sample_counts"]
    bins, ver = SC.pool_coverage(cfg, out, sel, log=logs.append)
    s3a = os.path.join(out, "s3a")
    for f in ("coverage.csv", "coverage.json", "diameters.csv", "coverage_mud.png", "coverage_share.png",
              "coverage_alpha.png", "diameters.png"):
        assert os.path.exists(os.path.join(s3a, f)), f
    js = survey_io.read_json(os.path.join(s3a, "coverage.json"))
    assert {"bins", "verdict", "per_cell", "outside_S", "mu_range_per_um", "dhat_bins_um"} <= set(js)
    assert js["per_cell"]["11"]["n"] == 6 and sum(b["n"] for b in js["bins"]) == js["verdict"]["n_nodes"]
    assert js["selection"]["n_recomputed"] == 6 and js["selection"]["n_now_in_S"] == 0, js["selection"]
    assert any(ln.startswith("coverage: S recomputed") for ln in logs), logs
    # an older run's CSV (in_S False and reject faint on every row, as a run before D-096 could write them): the pool
    # recomputes S, so the coverage and the per-cell counts are those of the current rule
    n_in = js["per_cell"]["11"]["n_in_S"]
    path = cell.path_of("nodes", "pilot_11.csv")
    table_io.write_rows([dict(r, in_S=False, reject="faint") for r in table_io.read_rows(path)], path)
    bins2, ver2 = SC.pool_coverage(cfg, out, sel, log=logs.append)
    js2 = survey_io.read_json(os.path.join(s3a, "coverage.json"))
    assert js2["per_cell"]["11"]["n_in_S"] == n_in and js2["per_cell"]["11"]["n_in_S_stored"] == 0, js2["per_cell"]
    assert js2["selection"]["n_now_in_S"] == n_in and ver2["n_nodes"] == ver["n_nodes"], (js2["selection"], ver2)
    head = open(os.path.join(s3a, "coverage.csv")).readline().strip().split(",")
    assert {"bin", "n", "share_inside", "q50_ln_mud", "q05_ln_alpha", "status"} <= set(head), head
    head = open(os.path.join(s3a, "diameters.csv")).readline().strip().split(",")
    assert {"neuron_class", "layer", "n", "q50_d_hat_um", "q50_D_allen_um"} <= set(head), head


def test_determinism():
    cfg = default_config()
    tmp = tempfile.mkdtemp()
    swc = tree(tmp)
    a, _ = SN.stratified_nodes(swc, cfg, 7)
    b, _ = SN.stratified_nodes(swc, cfg, 7)
    assert a == b
    other = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, selection_seed=cfg.survey.selection_seed + 1))
    c, _ = SN.stratified_nodes(swc, other, 7)
    d, _ = SN.stratified_nodes(swc, cfg, 8)                       # another specimen: another draw
    assert set(a) != set(c) and set(a) != set(d)


def test_edge_cases():
    cfg = default_config()
    tmp = tempfile.mkdtemp()
    swc = tree(tmp)
    ids, counts = SN.stratified_nodes(swc, cfg, 7)
    nan_ids = [int(i) for i, r in zip(swc.ids, swc.radius) if r != r]
    assert nan_ids and not set(nan_ids) & set(ids), "a NaN radius belongs to no bin"
    assert 2 not in counts and 3 not in counts and 4 not in counts      # empty bins are absent, not errors
    assert SN.allen_bin(float("nan"), (0.4,)) is None and SN.allen_bin(-1.0, (0.4,)) is None
    assert SN.allen_bin(0.2, (0.4,)) == 1 and SN.allen_bin(0.19, (0.4,)) == 0      # [0, 0.4), [0.4, inf)
    small = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, nodes_per_bin=1))
    ids1, c1 = SN.stratified_nodes(swc, small, 7)
    assert all(t == 1 for _, t in c1.values()) and len(ids1) == len(c1)
    every = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, nodes_mode="all"))
    assert SN.sample_nodes(swc, every, 7) == (SN.all_nodes(swc, cfg), {})
    # a cell without nodes in S: zero counts, too_few everywhere, the verdict passes vacuously but says so
    bins = COV.coverage_table([node(1, "exc", "2", 0.5, 1.0, in_S=False, reject="dark")], cfg)
    assert all(b["n"] == 0 and b["status"] == "too_few" for b in bins)
    v = COV.verdict(bins, cfg)
    assert v["passed"] and v["counted"] == [] and v["n_nodes"] == 0
    assert COV.outside_counts([node(1, "exc", "2", 0.5, 1.0, in_S=False, reject="dark;flat")]) == {"dark": 1, "flat": 1}
    # D-098.3: a row with one of fit_status and flags only keeps its stored S; empty and NaN cells are no flag and
    # no registration verdict (a NaN read as text would be the category UNKNOWN)
    half = [dict(node_id=1, fit_status="converged", in_S=False, reject="faint", d_hat_um=0.5, mu_hat_per_um=1.0),
            dict(node_id=2, flags="", in_S=True, reject="", d_hat_um=0.5, mu_hat_per_um=1.0)]
    t = COV.node_table([(dict(specimen_id=3), half)])
    assert [(n["in_S"], n["reject"], n["s_source"]) for n in t] == [(False, "faint", "stored"), (True, "", "stored")], t
    nan = float("nan")
    assert phantoms.row_reject_reasons(dict(fit_status="converged", flags=nan, reg_verdict=nan)) == []
    assert phantoms.row_reject_reasons(dict(fit_status="converged", flags="", reg_verdict="")) == []
    assert phantoms.row_reject_reasons(dict(fit_status="converged", flags="crossing")) == ["crossing"]
    try:
        phantoms.row_reject_reasons(dict(flags="crossing"))
    except KeyError:
        pass
    else:
        raise AssertionError("a row without fit_status must raise KeyError")
    for bad in (dict(nodes_mode="some"), dict(nodes_per_bin=0), dict(nodes_allen_bins_um=(0.6, 0.4)),
                dict(coverage_dhat_bins_um=(0.0, 1.0)), dict(coverage_pass_share=1.5),
                dict(coverage_phantom_quantiles=(0.9, 0.1))):
        try:
            dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, **bad)).validate()
        except ValueError:
            continue
        raise AssertionError("survey %s must be refused" % bad)


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
