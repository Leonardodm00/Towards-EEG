"""Smoke test for the kernel's in-focus width switch -- SPEC Block 36, amendment 22 (D-221.1, D-222.3, D-225.7):
loading/sigma_io.read_calibrated_kernel, scripts/kernel_width.apply and scripts/calibrate_sigma.growth_scale_summary.

Run
    python tests/smoke/test_smoke_kernel_width.py                     (seconds; the "-- n pass" line, exit 0/1)
    python -m pytest tests/smoke/test_smoke_kernel_width.py -q        (the same checks through pytest)

The script runner is the package's (D-233: the final check's F2 found unittest.main() running none of these
module-level checks). Pure ASCII.
"""
import dataclasses
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(WS, "src"), os.path.join(WS, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from allen_diameter import config as C                      # noqa: E402
from allen_diameter.loading import sigma_io                 # noqa: E402
from allen_diameter.model import kernel                     # noqa: E402
import kernel_width                                         # noqa: E402


def _write(d):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    return path


GOOD = dict(specimen=7, complete=True,
            summary=dict(n=4, median_um=0.07, mad_um=0.004, ivw_mean_um=0.0712, ivw_se_um=0.0021, chi2_cons=3.1, dof=3),
            summary_growth_scale=dict(n=3, median=1.1, mad=0.05, ivw_mean=1.083, ivw_se=0.031, chi2_cons=2.0, dof=2),
            fits=[])


def test_read_good():
    path = _write(GOOD)
    v = sigma_io.read_calibrated_kernel(path)
    assert (v["sigma0_um"], v["se_sigma0_um"], v["n_sigma0"]) == (0.0712, 0.0021, 4)
    assert (v["growth_scale"], v["se_growth_scale"], v["n_growth_scale"]) == (1.083, 0.031, 3)
    assert v["file"] == os.path.abspath(path) and v["specimen"] == 7


def test_read_refusals():
    bad = []
    d = dict(GOOD, complete=False)
    bad.append(d)
    d = dict(GOOD)
    d.pop("summary_growth_scale")
    bad.append(d)
    bad.append(dict(GOOD, summary=dict(GOOD["summary"], n=0, ivw_mean_um=None)))
    bad.append(dict(GOOD, summary_growth_scale=dict(GOOD["summary_growth_scale"], ivw_mean=-1.0)))
    bad.append(dict(GOOD, summary=dict(GOOD["summary"], ivw_mean_um=None)))
    for d in bad:
        try:
            sigma_io.read_calibrated_kernel(_write(d))
        except ValueError:
            continue
        raise AssertionError("accepted %r" % (d,))


def test_fixed_keeps_the_configuration():
    cfg = C.default_config()
    out, rec = kernel_width.apply(cfg, None, "")
    assert out.renderer == cfg.renderer and rec["mode"] == "fixed" and rec["sigma0_um"] == cfg.renderer.sigma_r0_um
    out2, rec2 = kernel_width.apply(cfg, "fixed", "/nonexistent.json")      # the file is not read when fixed
    assert out2.renderer == cfg.renderer


def test_calibrated_is_with_kernel_of_the_base():
    cfg = C.default_config()
    path = _write(GOOD)
    out, rec = kernel_width.apply(cfg, "calibrated", path)
    ref = kernel.with_kernel(cfg.renderer, 0.0712, 1.083)
    assert out.renderer == ref, (out.renderer, ref)
    assert rec["base_renderer"]["sigma_r0_um"] == cfg.renderer.sigma_r0_um
    assert rec["table_sigma_um"] == list(ref.kernel_table_sigma_um)
    # re-anchoring an already calibrated configuration from its recorded base gives the same law (no compounding)
    again, _ = kernel_width.apply(out, "calibrated", path, base=rec["base_renderer"])
    assert again.renderer == ref
    # ... whereas re-anchoring the calibrated table itself would compound c
    twice = kernel.with_kernel(out.renderer, 0.0712, 1.083)
    assert max(abs(a - b) for a, b in zip(twice.kernel_table_sigma_um, ref.kernel_table_sigma_um)) > 1e-4


def test_calibrated_refusals():
    cfg = C.default_config()
    missing = os.path.join(tempfile.mkdtemp(), "no_such_sigma.json")          # D-233 (S6): a ValueError, no traceback
    for mode, path in (("calibrated", ""), ("other", ""), ("calibrated", _write(dict(GOOD, complete=False))),
                       ("calibrated", missing)):
        try:
            kernel_width.apply(cfg, mode, path)
        except ValueError:
            continue
        raise AssertionError("accepted %r %r" % (mode, path))


def test_recorded_base_with_jpeg_tables():
    """D-233 (the final check's F1): a preparation's recorded base renderer, after its JSON round trip, carries
    Allen's JPEG tables as nested lists; the switch reads it back with config_from_dict's rule."""
    cfg0 = C.default_config()
    cfg = dataclasses.replace(cfg0, renderer=dataclasses.replace(cfg0.renderer,
                                                                 jpeg_qtables=(tuple(range(1, 65)),)))
    cfg.validate()
    _, rec = kernel_width.apply(cfg, "fixed", "")
    rec_rt = json.loads(json.dumps(rec))
    assert isinstance(rec_rt["base_renderer"]["jpeg_qtables"][0], list)
    out, _ = kernel_width.apply(cfg, "fixed", "", base=rec_rt["base_renderer"])
    assert out.renderer == cfg.renderer
    out, _ = kernel_width.apply(cfg, "calibrated", _write(GOOD), base=rec_rt["base_renderer"])
    assert out.renderer == kernel.with_kernel(cfg.renderer, 0.0712, 1.083)


def test_growth_scale_summary_by_hand():
    import calibrate_sigma
    rows = [dict(ok=True, growth_scale=1.0, se_growth_scale=0.1), dict(ok=True, growth_scale=1.2, se_growth_scale=0.2),
            dict(ok=False, growth_scale=5.0, se_growth_scale=0.1), dict(ok=True, growth_scale=float("nan"),
                                                                         se_growth_scale=0.1),
            dict(ok=True, growth_scale=0.9, se_growth_scale=0.0)]
    s = calibrate_sigma.growth_scale_summary(rows)
    w = np.array([1 / 0.1 ** 2, 1 / 0.2 ** 2])
    x = np.array([1.0, 1.2])
    mean = float(np.sum(w * x) / np.sum(w))
    assert s["n"] == 2 and abs(s["ivw_mean"] - mean) < 1e-15 and abs(s["ivw_se"] - 1 / math.sqrt(w.sum())) < 1e-15
    assert abs(s["median"] - 1.1) < 1e-15 and s["dof"] == 1
    assert abs(s["chi2_cons"] - float(np.sum(w * (x - mean) ** 2))) < 1e-12
    empty = calibrate_sigma.growth_scale_summary([dict(ok=False, growth_scale=1.0, se_growth_scale=0.1)])
    assert empty["n"] == 0 and math.isnan(empty["ivw_mean"])


def main():
    """The package's script runner: a PASS/FAIL table and the "-- n pass, ..." line; exit 1 on a FAIL or ERROR."""
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print("python %s | numpy %s | %s" % (platform.python_version(), np.__version__, platform.platform()))
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
        except Exception:                       # unexpected errors are reported, never hidden
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
