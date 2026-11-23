"""The kernel's in-focus width, fixed or calibrated: the switch shared by clean_cell.py, infer_nodes.py and
fit_soma.py -- SPEC Block 36, amendment 22 (D-221.1, D-222.3, D-225.7).

    add_arguments(parser, default_text=...)     --kernel-width fixed|calibrated, --sigma-json PATH
    apply(cfg, mode, sigma_json, base=None)  -> (cfg', record)

"fixed" keeps the configuration's kernel table (sigma_r(0) = 0.080 um by default, D-172). "calibrated" replaces the
renderer's table by kernel.with_kernel(base, sigma_0, c), sigma_0 and c the inverse-variance weighted means of Cell 23B's
calibration (loading.sigma_io.read_calibrated_kernel; D-222.3), `base` the renderer before any re-anchoring (the
configuration's own, or a preparation's recorded base: applying with_kernel twice would compound c). The record names
the mode, the values, their SEs and counts, the file and the base renderer, for the script's JSON. A recorded base
renderer is read back with config.config_from_dict's rule (nested tuples restored: renderer.jpeg_qtables), and a
missing --sigma-json file is refused like an incomplete one (D-233: the final check's F1 and S6). Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import os

from allen_diameter.config import RendererConfig, config_from_dict
from allen_diameter.loading import sigma_io
from allen_diameter.model import kernel

MODES = ("fixed", "calibrated")


def add_arguments(p, default_text="Default: fixed"):
    """The two flags; default_text ends --kernel-width's help (infer_nodes.py: the preparation's law, D-233)."""
    p.add_argument("--kernel-width", choices=MODES, default=None,
                   help="the kernel's in-focus width: fixed = the configuration's table (sigma_r(0) 0.080 um by "
                        "default); calibrated = re-anchored at Cell 23B's weighted means of sigma0 and of the growth "
                        "scale c (--sigma-json; Block 36 amendment 22, D-222.3). " + default_text)
    p.add_argument("--sigma-json", default="", help="Cell 23B's sigma_<S>.json (calibrate_sigma.py), with "
                   "--kernel-width calibrated")


def _renderer(base):
    """A RendererConfig from a RendererConfig or a dict of its fields (a JSON record), read back by
    config.config_from_dict's rule: nested tuples restored (renderer.jpeg_qtables), the renderer validated
    (D-233, the final check's F1)."""
    if isinstance(base, RendererConfig):
        return base
    names = {f.name for f in dataclasses.fields(RendererConfig)}
    return config_from_dict({"renderer": {k: v for k, v in base.items() if k in names}}).renderer


def apply(cfg, mode, sigma_json, base=None):
    """-> (cfg with its renderer set by the switch, record dict). mode None or "fixed": the renderer `base` (default:
    cfg's own) unchanged; "calibrated": kernel.with_kernel(base, sigma_0, c) from sigma_json. ValueError: an unknown
    mode, "calibrated" without sigma_json, a missing or unreadable file, a file sigma_io refuses, a table with_kernel
    refuses."""
    mode = "fixed" if mode is None else mode
    if mode not in MODES:
        raise ValueError("--kernel-width must be one of %r, got %r" % (MODES, mode))
    ren0 = cfg.renderer if base is None else _renderer(base)
    record = dict(mode=mode, base_renderer=dataclasses.asdict(ren0))
    if mode == "fixed":
        ren = ren0
        record.update(sigma0_um=float(ren0.sigma_r0_um), growth_scale=1.0, file="")
    else:
        if not sigma_json:
            raise ValueError("--kernel-width calibrated needs --sigma-json (Cell 23B's sigma_<S>.json)")
        if not os.path.isfile(sigma_json):           # D-233 (the final check's S6): refused, not a traceback
            raise ValueError("--sigma-json %s: no such file" % sigma_json)
        try:
            vals = sigma_io.read_calibrated_kernel(sigma_json)
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError("--sigma-json %s: not readable (%s: %s)" % (sigma_json, type(exc).__name__, exc))
        ren = kernel.with_kernel(ren0, vals["sigma0_um"], vals["growth_scale"])
        record.update(vals)
    out = dataclasses.replace(cfg, renderer=ren)
    out.validate()
    record["table_sigma_um"] = [float(x) for x in ren.kernel_table_sigma_um]
    record["table_delta_um"] = [float(x) for x in ren.kernel_table_delta_um]
    return out, record


def describe(record):
    """One log line."""
    if record["mode"] == "fixed":
        return "kernel width: fixed, sigma_r(0) %.4f um (the configuration's table)" % record["sigma0_um"]
    return ("kernel width: calibrated, sigma_r(0) %.4f +- %.4f um (n %d), growth scale c %.4f +- %.4f (n %d), from %s; "
            "table %s um" % (record["sigma0_um"], record["se_sigma0_um"], record["n_sigma0"], record["growth_scale"],
                             record["se_growth_scale"], record["n_growth_scale"], record["file"],
                             ["%.4f" % s for s in record["table_sigma_um"]]))
