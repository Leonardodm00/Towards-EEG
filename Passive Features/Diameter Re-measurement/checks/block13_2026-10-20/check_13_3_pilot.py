"""Block 13.3b oracle: the pilot is Block 5 at commit 84ea91f, bit for bit.

measure_pilot (and the pilot inside measure_node, through pilot_view) makes
node_request's one request -- Block 5's xy block over Block 5's planes and
the iteration's K_i = k_SWC +- iter_planes_half -- and reads only Block 5's
planes of it. On a provider that serves the same pixels for any request (the
synthetic cell of tests/smoke/fixtures_cell.py, like Allen's image cache),
its NodeResult must equal, field by field (arrays bit for bit, NaN equal to
NaN), what analysis/node_pipeline.py's measure_node returned at 84ea91f (read
with `git show` and run on this tree's Block 5 dependencies, unchanged since
84ea91f: background, focus, path, profiles, fit). Tubes of 0.8 um and
3.0 um (the thick tube's pilot ends on the edge of its planes, where a pass
whose redrawn direction is tilted would read more planes if it were not
clipped to Block 5's), every node of the stretch; the request's xy equals the
old request's and its planes contain the old ones; on two nodes per tube
pilot_view(measure_node(...)) equals measure_pilot(...) as well.

Run (from the workstream folder, in a git clone that holds 84ea91f):
    python checks/block13_2026-10-20/check_13_3_pilot.py
Expected last line: PASS (... pilots equal)

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
for p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OLD_COMMIT = "84ea91f"
DEPS = ("background", "focus", "path", "profiles", "fit")


def git_show(rel):
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=WS, capture_output=True, text=True,
                         check=True).stdout.strip()
    return subprocess.run(["git", "show", "%s:%s" % (OLD_COMMIT, rel)], cwd=top, capture_output=True, text=True,
                          check=True).stdout


def load_old_node_pipeline():
    """node_pipeline.py at 84ea91f as a module of the current package (its relative imports resolve to this
    tree's Block 5 dependencies, which must be unchanged since 84ea91f)."""
    base = "Passive Features/Diameter Re-measurement/src/allen_diameter/analysis/"
    for dep in DEPS:
        now = (WS / "src" / "allen_diameter" / "analysis" / (dep + ".py")).read_text(encoding="ascii")
        assert now == git_show(base + dep + ".py"), "analysis/%s.py changed since %s" % (dep, OLD_COMMIT)
    src = git_show(base + "node_pipeline.py")
    mod = types.ModuleType("allen_diameter.analysis._node_pipeline_84ea91f")
    mod.__package__ = "allen_diameter.analysis"
    mod.__file__ = "node_pipeline_84ea91f.py"
    sys.modules[mod.__name__] = mod               # dataclasses resolves the module's annotations through it
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def same(a, b):
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        return all(same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        A, B = np.asarray(a), np.asarray(b)
        return A.shape == B.shape and A.dtype.kind == B.dtype.kind and bool(
            np.array_equal(A, B, equal_nan=A.dtype.kind in "fc"))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def main():
    import run_cell
    from allen_diameter.analysis import cell
    from allen_diameter.analysis import node_pipeline as NP
    from allen_diameter.config import default_config
    from fixtures_cell import synthetic_cell
    OLD = load_old_node_pipeline()
    base = default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    n_equal = 0
    with tempfile.TemporaryDirectory() as tmp:
        for d_true in (0.8, 3.0):
            sub = os.path.join(tmp, "d%g" % d_true)
            os.makedirs(sub)
            swc, fetcher, planes, _ = synthetic_cell(sub, cfg, d_true=d_true, mu=1.0)
            prov = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
            types_ = cfg.acquisition.dendrite_swc_types
            branch, off = cell.stretch_branch(swc, cell.stretches(swc, types_)[0],
                                              cell.to_image_um(swc.xyz, cfg.acquisition.res0_um))
            scene = cell.cell_scene(swc, types_, cfg.acquisition.res0_um)
            for i in range(len(branch.ids)):
                old_calls, new_calls = [], []

                def rec(calls):
                    def provider(*args):
                        calls.append(args)
                        return prov(*args)
                    return provider
                old = OLD.measure_node(branch, i, rec(old_calls), cfg)
                new = NP.measure_pilot(branch, i, rec(new_calls), cfg)
                bad = [f.name for f in dataclasses.fields(old) if not same(getattr(old, f.name), getattr(new, f.name))]
                (oa,), (na,) = old_calls, new_calls
                req_ok = oa[:4] == na[:4] and na[4] <= oa[4] and oa[5] <= na[5]
                line = ("d %.1f um node %d: %s; request %s, old %s; k* %d, d_hat %.6f um, flags %s"
                        % (d_true, int(branch.ids[i]), "equal" if not bad else "DIFFER " + ",".join(bad),
                           na[4:], oa[4:], new.k_star, new.d_hat_um, new.flags))
                print(line, flush=True)
                assert not bad and req_ok, line
                n_equal += 1
                if i in (2, 3):
                    full = NP.measure_node(branch, i, prov, cfg, None, scene)
                    view = NP.pilot_view(full)
                    bad = [f.name for f in dataclasses.fields(new) if not same(getattr(new, f.name), getattr(view, f.name))]
                    print("    pilot_view(measure_node): %s (k_iter %d, %s, d_hat %.4f um)"
                          % ("equal" if not bad else "DIFFER " + ",".join(bad), full.k_iter, full.iter_status,
                             full.d_hat_um), flush=True)
                    assert not bad, bad
    print("PASS (%d pilots equal)" % n_equal)


if __name__ == "__main__":
    main()
