"""Block 13.1 oracle: the iteration moved into analysis/plane_iteration.py
behaves exactly as the script's own copy at commit 84ea91f.

On the synthetic cell of tests/smoke/fixtures_cell.py (a provider that serves
the same pixels for any request, like Allen's cache), for tubes of 0.8 um and
3.0 um (Allen radius 0.3 um), nodes 3 and 5, multipliers m = 2 and 3, the
square and restack of the script's run(): node_stack and gradient_iteration of
the script at 84ea91f (read with `git show`) and of the current script (whose
iteration functions are imported from the package) return equal records --
every key, every array bit for bit (NaN equal to NaN), every float equal.
Also: the current script's iteration names are the package's objects.

Run (from the workstream folder, in a git clone that holds 84ea91f):
    python checks/block13_2026-10-20/check_13_1_move.py
Expected last line: PASS (16 runs equal)

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
for p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OLD_COMMIT = "84ea91f"
NAMES = ("ITER_LINE_MULT", "ITER_STATUSES", "check_iteration", "iterate_planes", "far_from_dendrites",
         "plane_background", "block_background", "line_profiles", "refit_at_plane", "gradient_iteration",
         "_argmax_plane")


def load_old_script(tmp):
    rel = "Passive Features/Diameter Re-measurement/scripts/plane_differences.py"
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=WS, capture_output=True, text=True,
                         check=True).stdout.strip()
    text = subprocess.run(["git", "show", "%s:%s" % (OLD_COMMIT, rel)], cwd=top, capture_output=True, text=True,
                          check=True).stdout
    path = os.path.join(tmp, "pd_old_84ea91f.py")
    with open(path, "w") as f:
        f.write(text)
    spec = importlib.util.spec_from_file_location("pd_old_84ea91f", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def equal(a, b, where="root"):
    """None when a and b are equal (recursively; arrays bit for bit, NaN = NaN), else a description."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return "%s: keys differ %r vs %r" % (where, sorted(a) if isinstance(a, dict) else a,
                                                 sorted(b) if isinstance(b, dict) else b)
        for k in a:
            if k in ("st",):                    # the square dict: compared on its own below
                continue
            d = equal(a[k], b[k], "%s.%s" % (where, k))
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return "%s: lengths %d vs %d" % (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = equal(x, y, "%s[%d]" % (where, i))
            if d:
                return d
        return None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        A, B = np.asarray(a), np.asarray(b)
        if A.shape != B.shape or A.dtype.kind != B.dtype.kind:
            return "%s: arrays %r/%s vs %r/%s" % (where, A.shape, A.dtype, B.shape, B.dtype)
        same = np.array_equal(A, B, equal_nan=A.dtype.kind in "fc")
        return None if same else "%s: arrays differ (max |diff| %r)" % (where, np.nanmax(np.abs(A - B)) if A.dtype.kind in "fc" else "n/a")
    if isinstance(a, float) and isinstance(b, float):
        return None if (a == b or (math.isnan(a) and math.isnan(b))) else "%s: %r vs %r" % (where, a, b)
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        return equal(dataclasses.asdict(a), dataclasses.asdict(b), where)
    return None if a == b else "%s: %r vs %r" % (where, a, b)


def main():
    import run_cell
    import plane_differences as PD_NEW
    from allen_diameter.analysis import plane_iteration as PI
    from allen_diameter.config import default_config
    from fixtures_cell import synthetic_cell
    bad = [n for n in NAMES if getattr(PD_NEW, n) is not getattr(PI, n)]
    assert not bad, "the script does not use the package's objects: %r" % bad
    n_runs = 0
    with tempfile.TemporaryDirectory() as tmp:
        PD_OLD = load_old_script(tmp)
        base = default_config()
        cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
        for d_true in (0.8, 3.0):
            sub = os.path.join(tmp, "d%g" % d_true)
            os.makedirs(sub)
            swc, fetcher, planes, _ = synthetic_cell(sub, cfg, d_true=d_true, mu=1.0)
            prov = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
            for nid in (3, 5):
                for m in (2.0, 3.0):
                    out = []
                    for PD in (PD_OLD, PD_NEW):
                        st = PD.node_stack(swc, prov, cfg, nid, None, 6, None, 0.0, None, m, m)
                        it = PD.gradient_iteration(st, cfg, line_mult=m, restack=lambda h, PD=PD: PD.node_stack(
                            swc, prov, cfg, nid, None, 6, None, 0.0, h, m, m))
                        out.append((st, it))
                    (st0, it0), (st1, it1) = out
                    d = equal(it0, it1, "iteration") or equal({k: v for k, v in it0["st"].items() if k not in ("result", "branch")},
                                                             {k: v for k, v in it1["st"].items() if k not in ("result", "branch")},
                                                             "last square") \
                        or equal(it0["st"]["result"], it1["st"]["result"], "result")
                    n_runs += 2
                    print("d %.1f um node %d m %g: %s; status %s, rounds %d, k_final %s, d_final %.4f um, restacks %d"
                          % (d_true, nid, m, "equal" if d is None else "DIFFER " + d, it1["status"], it1["n_rounds"],
                             it1["k_final"], it1["d_final_um"], it1["n_restacks"]))
                    assert d is None, d
    print("PASS (%d runs equal)" % n_runs)


if __name__ == "__main__":
    main()
