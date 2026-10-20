"""Block 13.2 oracle: survey.node_planes and scripts/plane_differences.node_stack,
rebuilt on the shared rules of analysis/plane_iteration.py (Scene,
scene_segments, cut_square, iteration_state) and analysis/cell.cell_scene,
return what they returned at commit 84ea91f.

On the synthetic cell of tests/smoke/fixtures_cell.py (tubes of 0.8 and
3.0 um), nodes 3 and 5: node_planes of survey.py at 84ea91f (read with
`git show`, its relative imports resolved in the current package) against the
current one -- the result's fields, the block, its planes and frame, k_swc,
Allen's radius and the segments, bit for bit; node_stack of the script at
84ea91f (on the old node_planes) against the current one, for a square inside
the block, a square fetched (half 5 um > block_half_um 3.5 um) and one sized
by a measuring line (4 um) -- every key of the old dict, arrays bit for bit;
and gradient_iteration on each pair of squares (m = 3, with restacks) --
equal records.
[2026-10-20, Block 13.3b] The old survey's measure_node was Block 5's chain,
today's node_pipeline.measure_pilot (check_13_3_pilot.py), while the name
measure_node is Block 13's chain since 13.3b: the old module is bound to
measure_pilot, so that its node_planes measures as it did; both then make
node_request's one request. Re-run after 13.3.

Run (from the workstream folder, in a git clone that holds 84ea91f):
    python checks/block13_2026-10-20/check_13_2_square.py
Expected last line: PASS (24 comparisons equal)

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
for p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from check_13_1_move import equal  # noqa: E402  (the recursive comparison of the 13.1 check)

OLD_COMMIT = "84ea91f"
REL = "Passive Features/Diameter Re-measurement/"


def git_show(rel):
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=WS, capture_output=True, text=True,
                         check=True).stdout.strip()
    return subprocess.run(["git", "show", "%s:%s%s" % (OLD_COMMIT, REL, rel)], cwd=top, capture_output=True,
                          text=True, check=True).stdout


def load(name, text, tmp, package=None):
    path = os.path.join(tmp, name + ".py")
    with open(path, "w") as f:
        f.write(text)
    full = (package + "." + name) if package else name
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    if package:
        mod.__package__ = package
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def scalars(res):
    return {k: v for k, v in dataclasses.asdict(res).items() if not isinstance(v, (np.ndarray, dict)) and k != "fit"}


def main():
    import run_cell
    import plane_differences as PD_NEW
    from allen_diameter.analysis import survey as SV_NEW
    from allen_diameter.config import default_config
    from fixtures_cell import synthetic_cell
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        SV_OLD = load("survey_84ea91f", git_show("src/allen_diameter/analysis/survey.py"), tmp, "allen_diameter.analysis")
        PD_OLD = load("pd_84ea91f", git_show("scripts/plane_differences.py"), tmp)
        PD_OLD.survey = SV_OLD                     # the old node_stack on the old node_planes
        from allen_diameter.analysis import node_pipeline as NP
        SV_OLD.measure_node = NP.measure_pilot     # Block 5's chain under its 13.3b name (see the docstring)
        base = default_config()
        cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
        for d_true in (0.8, 3.0):
            sub = os.path.join(tmp, "d%g" % d_true)
            os.makedirs(sub)
            swc, fetcher, planes, _ = synthetic_cell(sub, cfg, d_true=d_true, mu=1.0)
            prov = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
            for nid in (3, 5):
                old, new = SV_OLD.node_planes(swc, prov, cfg, nid), SV_NEW.node_planes(swc, prov, cfg, nid)
                assert set(new) - set(old) == {"scene"} and not set(old) - set(new), (sorted(old), sorted(new))
                d = equal(scalars(old["result"]), scalars(new["result"]), "node_planes.result")
                for k in ("block", "ks", "valid", "k_swc", "allen_radius_um", "segments", "index"):
                    d = d or equal(old[k], new[k], "node_planes.%s" % k)
                fo, fn = old["frame"], new["frame"]
                d = d or (None if (fo.left, fo.top, fo.res_um_px) == (fn.left, fn.top, fn.res_um_px) else "frame")
                n += 1
                print("d %.1f um node %d node_planes: %s (%d segments)" % (d_true, nid, d or "equal", new["segments"].shape[0]))
                assert d is None, d
                for half, line_h, mult in ((None, None, 3.0), (5.0, None, 3.0), (None, 4.0, None)):
                    args = (swc, prov, cfg, nid, None, 6, half, 0.0, line_h, mult, mult)
                    so, sn = PD_OLD.node_stack(*args), PD_NEW.node_stack(*args)
                    assert set(sn) - set(so) == {"index"} and not set(so) - set(sn), (sorted(so), sorted(sn))
                    d = None
                    for k in so:
                        if k in ("result", "branch", "frame", "block_frame"):
                            continue
                        d = d or equal(so[k], sn[k], "node_stack.%s" % k)
                    for k in ("frame", "block_frame"):
                        a, b = so[k], sn[k]
                        d = d or (None if (a.left, a.top, a.res_um_px) == (b.left, b.top, b.res_um_px) else k)
                    d = d or equal(scalars(so["result"]), scalars(sn["result"]), "node_stack.result")
                    n += 1
                    print("   node_stack half %s line %s mult %s: %s (fetched %s, half %.3f um)"
                          % (half, line_h, mult, d or "equal", sn["square_fetched"], sn["square_half_um"]))
                    assert d is None, d
                    if mult is not None:
                        out = []
                        for PD, st in ((PD_OLD, so), (PD_NEW, sn)):
                            out.append(PD.gradient_iteration(st, cfg, line_mult=3.0, restack=lambda h, PD=PD: PD.node_stack(
                                swc, prov, cfg, nid, None, 6, half, 0.0, h, 3.0, 3.0)))
                        d = equal(out[0], out[1], "iteration")
                        n += 1
                        print("   iteration: %s (status %s, k_final %s, d %.4f um, restacks %d)"
                              % (d or "equal", out[1]["status"], out[1]["k_final"], out[1]["d_final_um"], out[1]["n_restacks"]))
                        assert d is None, d
    print("PASS (%d comparisons equal)" % n)


if __name__ == "__main__":
    main()
