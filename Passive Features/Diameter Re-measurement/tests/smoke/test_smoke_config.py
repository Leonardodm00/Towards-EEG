"""Smoke test for the configuration module -- Block 1 in specs/SPEC.md.

What it checks (strongest first)
    test_known_answer   the deliverable defaults are the decided values
                        (D-018, D-019, D-021 to D-024, D5, D7)
    test_reference      every parameter line of config.py carries a
                        '# source:' tag (D-021), read from the source file
    test_convergence    skipped: no discretisation
    test_invariants     signatures: deterministic, the study set is not an
                        estimator setting, the renderer is not either, a
                        different sigma_fit is a different estimator
    test_contract       JSON round trip, frozen dataclasses, ASCII output
    test_determinism    two constructions give the same hashes
    test_edge_cases     validate() refuses every illegal name or range named
                        in the spec; with_sigma_fit refuses a value outside
                        the study set

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_config.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table.
        Exit code 1 if any check failed, errored or is still TODO.
    python -m pytest tests/smoke/test_smoke_config.py -q

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter import config as C  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = HERE / "artifacts"


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = C.default_config()
    m, c, r, p, a = cfg.measure, cfg.correction, cfg.renderer, cfg.phantom, cfg.acquisition
    # D-018 (B fixed) + D-019 (mu, not alpha; phi from the line fit)
    assert m.fit_params == ("d", "mu", "v0"), m.fit_params
    # D-023
    assert m.mu_mode == "per_node", m.mu_mode
    assert m.bbar_region == "block_masked", m.bbar_region
    assert m.focus_bg_rule == "profile_ends_median", m.focus_bg_rule
    # D-030: gradient energy is the focus rule; its window (radius + 0.5 um) is PROVISIONAL
    assert m.focus_rule == "gradient_energy" and C.FOCUS_RULES[0] == "gradient_energy" and "dip_depth" in C.FOCUS_RULES
    assert m.focus_grad_window == "radius_margin" and m.focus_grad_margin_um == 0.5, (m.focus_grad_window,
                                                                                      m.focus_grad_margin_um)
    assert m.sigma_fit_um == 0.099 and 0.080 in m.sigma_fit_study_um and 0.125 in m.sigma_fit_study_um
    # D-024
    assert r.absorption == "partition_vertical", r.absorption
    assert r.kernel_continuation == "linear" and abs(r.kernel_continuation_slope - 0.79) < 1e-12
    assert m.alpha_dark_flag == 1.0
    assert r.U_um == 10.0
    assert p.design == "random"
    assert p.phi_range_deg == (0.0, 90.0), p.phi_range_deg
    assert p.jitter_xy_um == 0.0 and p.jitter_z_um == 0.0
    assert c.response_estimator == "tps_spline"
    # D5, D7 (confirmed by D-024)
    assert c.bias_flag_threshold == 0.2
    assert c.fill_policy == "same_branch_then_allen"
    assert a.specimen_id == 529878215
    # D-024: tilt is never a selection criterion -- the only 'steep' knob is the diagnostic one
    steep_fields = [f.name for f in dataclasses.fields(C.MeasureConfig) if "steep" in f.name]
    assert steep_fields == ["steep_tan_diagnostic"], steep_fields
    # acquisition facts (handoff)
    assert a.res0_um == 0.1144 and a.dz_um == 0.28
    # the first entry of every choice tuple is the default actually used
    assert C.MU_MODES[0] == m.mu_mode and C.BBAR_REGIONS[0] == m.bbar_region
    assert C.ABSORPTIONS[0] == r.absorption and C.KERNEL_CONTINUATIONS[0] == r.kernel_continuation
    assert C.RESPONSE_ESTIMATORS[0] == c.response_estimator and C.PHANTOM_DESIGNS[0] == p.design


def test_reference():
    """Every dataclass field line in config.py must carry a '# source:' tag."""
    src = (SRC / "allen_diameter" / "config.py").read_text(encoding="ascii")
    in_class = False
    missing = []
    n_tagged = 0
    for line in src.splitlines():
        if line.startswith("@dataclass"):
            in_class = True
            continue
        if in_class and line and not line.startswith(" ") and not line.startswith("class"):
            in_class = False
        if not in_class:
            continue
        m = re.match(r"^    ([A-Za-z_0-9]+): [^=]+= (.+)$", line)
        if m is None:
            continue
        name = m.group(1)
        if name in ("acquisition", "measure", "correction", "renderer", "phantom", "calibration"):
            continue  # the container's sub-config fields
        if "# source:" in line:
            n_tagged += 1
        else:
            missing.append(name)
    assert not missing, "fields without a '# source:' tag: %s" % missing
    assert n_tagged >= 60, "only %d tagged fields found; parser broken?" % n_tagged
    # every field of every sub-config was seen by the parser
    n_fields = sum(len(dataclasses.fields(k)) for k in (C.AcquisitionConfig, C.MeasureConfig, C.CorrectionConfig,
                                                        C.RendererConfig, C.PhantomConfig, C.CalibrationConfig))
    assert n_tagged == n_fields, (n_tagged, n_fields)
    # every field is read by some code outside config.py (attribute access or getattr, parsed with ast,
    # so a docstring naming a field does not count) [added 2026-10-07: renderer.jpeg_qtables_file was
    # read by nothing]; the exceptions are records and declarations that validate() checks
    import ast
    records = {"acquisition.specimen_id", "measure.node_step", "measure.fit_params", "measure.sigma_fit_study_um",
               "correction.inversion_method", "phantom.n_replicates"}
    read = set()
    roots = (SRC / "allen_diameter", SRC.parent / "scripts")
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name == "config.py" and path.parent.name == "allen_diameter":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="ascii"))):
                if isinstance(node, ast.Attribute):
                    read.add(node.attr)
                elif (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr"
                      and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)):
                    read.add(str(node.args[1].value))
    cfg = C.default_config()
    unread = sorted("%s.%s" % (s.name, f.name) for s in dataclasses.fields(cfg)
                    for f in dataclasses.fields(getattr(cfg, s.name)) if f.name not in read)
    assert set(unread) == records, "config fields no code reads: %s (declared records: %s)" % (
        sorted(set(unread) - records), sorted(records - set(unread)))


def test_convergence():
    raise unittest.SkipTest("configuration has no discretisation")


def test_invariants():
    cfg = C.default_config()
    e1 = cfg.estimator_signature()
    assert "sigma_fit_study_um" not in e1["measure"], "the study set is not an estimator setting (D-023)"
    assert "sigma_fit_um" in e1["measure"]
    assert set(e1) == {"acquisition", "measure"}
    f1 = cfg.full_signature()
    assert set(f1) == {"acquisition", "measure", "renderer", "phantom", "correction"}
    # a renderer change does not change the estimator signature; a sigma_fit change does
    cfg_r = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, U_um=8.0))
    assert cfg_r.signature_hash("estimator") == cfg.signature_hash("estimator")
    assert cfg_r.signature_hash("full") != cfg.signature_hash("full")
    # the simulation signature (2026-10-07): the full one without the correction settings
    assert set(cfg.simulation_signature()) == {"acquisition", "measure", "renderer", "phantom"}
    cfg_c = dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, spline_cv_folds=4))
    assert cfg_c.signature_hash("simulation") == cfg.signature_hash("simulation")
    assert cfg_c.signature_hash("full") != cfg.signature_hash("full")
    assert cfg_r.signature_hash("simulation") != cfg.signature_hash("simulation")
    try:
        cfg.signature_hash("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("signature_hash accepted an unknown signature name")
    cfg_s = C.with_sigma_fit(cfg, 0.125)
    assert cfg_s.signature_hash("estimator") != cfg.signature_hash("estimator")
    assert cfg_s.measure.sigma_fit_um == 0.125 and cfg_s.measure.sigma_fit_study_um == cfg.measure.sigma_fit_study_um
    assert len(cfg.signature_hash()) == 16 and re.fullmatch(r"[0-9a-f]{16}", cfg.signature_hash())


def test_contract():
    cfg = C.default_config()
    s = cfg.to_json()
    assert s.isascii(), "signature JSON must be ASCII"
    d = json.loads(s)
    back = C.config_from_dict(d)
    assert back == cfg, "JSON round trip changed the configuration"
    assert back.to_json() == s
    try:
        cfg.measure.sigma_fit_um = 0.2  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("MeasureConfig is not frozen")
    for sub in (cfg.acquisition, cfg.measure, cfg.correction, cfg.renderer, cfg.phantom, cfg.calibration):
        for f in dataclasses.fields(sub):
            v = getattr(sub, f.name)
            assert not isinstance(v, list), "%s: lists are mutable; use tuples" % f.name
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                assert np.isfinite(v), f.name
    # Allen's JPEG tables travel inside the configuration (2026-10-07): nested tuples survive the JSON
    # round trip, the configuration stays hashable, and the full hash is reproduced
    tables = (tuple(range(1, 65)), tuple(64 - j for j in range(64)))
    cfg_q = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, jpeg_qtables=tables))
    cfg_q.validate()
    back = C.config_from_dict(json.loads(cfg_q.to_json()))
    assert back == cfg_q and back.renderer.jpeg_qtables == tables
    assert all(isinstance(t, tuple) for t in back.renderer.jpeg_qtables)
    assert hash(back) == hash(cfg_q) and back.signature_hash("full") == cfg_q.signature_hash("full")
    assert cfg_q.signature_hash("full") != cfg.signature_hash("full"), "the tables must enter the table's signature"
    assert cfg_q.signature_hash("estimator") == cfg.signature_hash("estimator"), "the renderer is not the estimator"
    # with_overrides (2026-10-07): JSON-like values are put in the declared form, so the same values give
    # the same configuration and hash (0 and 0.0, a list and a tuple)
    assert C.with_overrides(cfg, {}) == cfg
    same = C.with_overrides(cfg, {"phantom.jitter_xy_um": 0, "phantom.d_range_um": [0.2, 4],
                                  "correction.failure_rate_ignore": ["dark"]})
    assert same == cfg and same.signature_hash("full") == cfg.signature_hash("full")
    assert isinstance(same.phantom.jitter_xy_um, float) and isinstance(same.phantom.d_range_um[1], float)
    got = C.with_overrides(cfg, {"renderer.jpeg_qtables": [list(range(1, 65))], "phantom.d_range_um": (0.2, 6.0),
                                 "renderer.noise_sd_gl": 1.4})
    assert got.renderer.jpeg_qtables == (tuple(range(1, 65)),) and got.phantom.d_range_um == (0.2, 6.0)
    assert got.renderer.noise_sd_gl == 1.4 and cfg.phantom.d_range_um == (0.2, 4.0), "the input was modified"
    assert C.config_from_dict(json.loads(got.to_json())) == got


def test_determinism():
    a, b = C.default_config(), C.default_config()
    assert a == b and a.signature_hash("full") == b.signature_hash("full")
    assert a.to_json() == b.to_json()


def test_edge_cases():
    cfg = C.default_config()

    def refuses(**changes):
        sub, kw = changes.popitem()
        bad = dataclasses.replace(cfg, **{sub: dataclasses.replace(getattr(cfg, sub), **kw)})
        try:
            bad.validate()
        except ValueError:
            return
        raise AssertionError("validate() accepted %s=%r" % (sub, kw))

    refuses(measure=dict(mu_mode="per_cell"))
    refuses(measure=dict(fit_params=("d", "alpha", "v0")))          # D-019
    refuses(measure=dict(fit_params=("d", "mu", "v0", "B")))        # D-018
    refuses(measure=dict(sigma_fit_um=0.2))                          # D-023: not in the study set
    refuses(measure=dict(bbar_region="whole_plane"))
    refuses(measure=dict(focus_rule="tenengrad"))                    # D-030: the rules are named
    refuses(measure=dict(focus_grad_window="core"))
    refuses(measure=dict(focus_grad_margin_um=0.05))                 # fewer than 3 samples in the window
    refuses(renderer=dict(absorption="partition_oblique"))
    refuses(renderer=dict(kernel_continuation="quadratic"))
    refuses(renderer=dict(kernel_table_delta_um=(0.0, 0.28, 0.14)))
    refuses(renderer=dict(kernel_table_delta_um=(0.1, 0.28)))
    refuses(renderer=dict(dzeta_um=0.0))
    refuses(phantom=dict(phi_range_deg=(0.0, 95.0)))
    refuses(phantom=dict(d_range_um=(4.0, 0.2)))
    refuses(phantom=dict(design="latin_hypercube"))
    refuses(correction=dict(fill_policy="interpolate"))
    refuses(acquisition=dict(light_direction=0))
    try:
        C.with_sigma_fit(cfg, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("with_sigma_fit accepted a value outside the study set")
    # config_from_dict on a signature with a bad name refuses too
    d = json.loads(cfg.to_json())
    d["measure"]["mu_mode"] = "nonsense"
    try:
        C.config_from_dict(d)
    except ValueError:
        pass
    else:
        raise AssertionError("config_from_dict accepted a bad mu_mode")
    # JPEG tables (2026-10-07): lists, wrong length, out-of-range entries, too many tables are refused
    good = tuple(range(1, 65))
    refuses(renderer=dict(jpeg_qtables=(list(good),)))
    refuses(renderer=dict(jpeg_qtables=[good]))
    refuses(renderer=dict(jpeg_qtables=(good[:63],)))
    refuses(renderer=dict(jpeg_qtables=((0,) + good[1:],)))
    refuses(renderer=dict(jpeg_qtables=((1.5,) + good[1:],)))
    refuses(renderer=dict(jpeg_qtables=(good,) * 5))
    # the legacy key: an empty jpeg_qtables_file (it had no effect) is dropped, a path is refused by name
    d = json.loads(cfg.to_json())
    del d["renderer"]["jpeg_qtables"]
    d["renderer"]["jpeg_qtables_file"] = ""
    assert C.config_from_dict(d) == cfg
    d["renderer"]["jpeg_qtables_file"] = "/content/drive/MyDrive/diameters/camera/jpeg_qtables_1.json"
    try:
        C.config_from_dict(d)
    except ValueError as exc:
        assert "jpeg_qtables" in str(exc) and "never" in str(exc), exc
    else:
        raise AssertionError("config_from_dict accepted a jpeg_qtables_file path that nothing would read")
    # with_overrides refuses an unknown section or field by name, and an illegal value through validate()
    for bad in ({"phantom.d_max_um": 6.0}, {"render.U_um": 8.0}, {"U_um": 8.0}, {"phantom.d_range_um": [6.0, 0.2]},
                {"renderer.jpeg_qtables": [[1] * 63]}):
        try:
            C.with_overrides(cfg, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("with_overrides accepted %r" % (bad,))


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
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
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
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
