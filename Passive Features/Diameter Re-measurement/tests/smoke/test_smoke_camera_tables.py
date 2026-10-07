"""Smoke test: Allen's JPEG tables reach the camera chain from the
configuration -- Block 4 in specs/SPEC.md (corrected 2026-10-07).

Before 2026-10-07 the tables entered the camera chain only through a
`qtables` argument that no table build passed, and the configuration field
meant to carry them (renderer.jpeg_qtables_file) was read by nothing, so a
table "built with Allen's tables" was encoded at jpeg_quality. The tables now
travel in the configuration itself (renderer.jpeg_qtables).

What it checks (strongest first)
    test_known_answer   camera_chain with configured tables equals an
                        independent construction (grey mapping, the same
                        noise draws, rounding, Pillow encode with the tables,
                        decode), bit for bit; the encoded JPEG carries
                        exactly the configured tables
    test_reference      the table build's own path: synthetic_block with a
                        configuration carrying tables equals synthetic_block
                        with the tables passed explicitly, and differs from
                        the configuration without tables; run_replicate
                        hands the configured tables to every JPEG encode
    test_convergence    skipped: no discretisation here
    test_invariants     no tables -> the jpeg_quality path, unchanged; an
                        explicit argument overrides the configured tables
    test_contract       dtype and shape; the configuration is not mutated
    test_determinism    the same seed gives the same block
    test_edge_cases     Pillow < 8.3 (zigzag-order tables) is refused when
                        tables are read or applied, not when quality is used

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_camera_tables.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table.
        Exit code 1 if any check failed, errored or is still TODO.
    python -m pytest tests/smoke/test_smoke_camera_tables.py -q

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import io
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import phantoms  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import jpeg_tables  # noqa: E402
from allen_diameter.model import camera as CAM  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402

SEED = 20261007
REPORT_PACKAGES = ("numpy", "scipy", "Pillow")
# two distinctive tables: a coarse one (strong quantisation, so the output cannot coincide with quality 85)
# and a graded one; entries in [1, 255], natural order
TABLES = (tuple(40 + (j % 8) for j in range(64)), tuple(2 + (j * 7) % 31 for j in range(64)))


def with_tables(cfg, tables=TABLES):
    return dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, jpeg_qtables=tables))


def tau_planes():
    """(3, 40, 48) transmittance with a dark line, different in every plane."""
    y, x = np.mgrid[0:40, 0:48]
    return np.stack([1.0 - (0.3 + 0.2 * k) * np.exp(-0.5 * ((x - 24.0 - k) / 2.5) ** 2) for k in range(3)])


def reference_chain(tau, rcfg, rng, tables):
    """The camera chain built from its parts, with Pillow called directly."""
    from PIL import Image
    s = CAM.grey_mapping(tau, rcfg) + rng.normal(0.0, rcfg.noise_sd_gl, size=tau.shape)
    q = CAM.quantize(s, 8)
    out = np.empty_like(q)
    for k in range(q.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(q[k])).save(buf, format="JPEG", qtables=[list(t) for t in tables])
        buf.seek(0)
        with Image.open(buf) as im:
            out[k] = np.array(im.convert("L"), dtype=np.uint8)
    return out


def small_block(cfg, rng, qtables=None):
    """A flat 0.4 um tube over 3 planes in a 9 x 9 um block (seconds)."""
    p = cfg.acquisition.res0_um
    tube = G.Tube((0.3, -0.2, 0.0), 0.2, 0.0, 0.4, 1.0, 4.0, "axial")
    left, top = int(math.floor(-4.5 / p)), int(math.floor(-4.5 / p))
    n = int(round(9.0 / p))
    blk, ks, valid, frame = R.synthetic_block(tube, 0.8, np.arange(-1, 2), left, top, n, n, cfg, rng,
                                              int(math.ceil(1.0 / p)), qtables=qtables)
    return blk


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    from PIL import Image
    cfg = with_tables(default_config())
    tau = tau_planes()
    out = CAM.camera_chain(tau, cfg.renderer, np.random.default_rng(SEED))
    ref = reference_chain(tau, cfg.renderer, np.random.default_rng(SEED), TABLES)
    assert out.dtype == np.uint8 and np.array_equal(out, ref), int(np.abs(out.astype(int) - ref).max())
    # the table written into the JPEG is the configured one (natural order, Pillow >= 8.3). A greyscale
    # plane has one component, so Pillow writes and uses table 0 only (the luminance table); a second
    # (chroma) table of Allen's set has no counterpart in the grey synthetic planes
    buf = io.BytesIO()
    Image.fromarray(ref[0]).save(buf, format="JPEG", qtables=[list(t) for t in cfg.renderer.jpeg_qtables])
    buf.seek(0)
    with Image.open(buf) as im:
        q = im.quantization
    assert [tuple(q[k]) for k in sorted(q)] == [TABLES[0]], "Pillow did not write the configured luminance table"


def test_reference():
    base = default_config()
    cfg = with_tables(base)
    a = small_block(cfg, np.random.default_rng(SEED))
    b = small_block(base, np.random.default_rng(SEED), qtables=[list(t) for t in TABLES])
    c = small_block(base, np.random.default_rng(SEED))
    assert np.array_equal(a, b), "a configuration with tables must render as the explicit tables do"
    assert not np.array_equal(a, c), "the configured tables changed nothing (jpeg_quality used instead)"
    # run_replicate, the table build's entry, hands the configured tables to every JPEG encode
    seen = []
    original = CAM.jpeg_roundtrip

    def recorder(img8, quality=None, qtables=None):
        seen.append(qtables)
        return original(img8, quality=quality, qtables=qtables)

    small = dataclasses.replace(cfg, phantom=dataclasses.replace(cfg.phantom, d_range_um=(0.5, 0.6),
                                                                 phi_range_deg=(0.0, 1.0), mu_range_per_um=(1.0, 1.1)),
                                renderer=dataclasses.replace(cfg.renderer, U_um=4.0))
    small.validate()
    CAM.jpeg_roundtrip = recorder
    try:
        phantoms.run_replicate(small, SEED, 0)
    finally:
        CAM.jpeg_roundtrip = original
    assert seen, "run_replicate encoded no plane"
    assert all(q == [list(t) for t in TABLES] for q in seen), "a JPEG encode did not receive the configured tables"


def test_convergence():
    raise unittest.SkipTest("no discretisation: the camera chain is exact integer arithmetic plus Pillow")


def test_invariants():
    base = default_config()
    tau = tau_planes()
    # without tables, the jpeg_quality path as before 2026-10-07
    out0 = CAM.camera_chain(tau, base.renderer, np.random.default_rng(SEED))
    from PIL import Image
    rng = np.random.default_rng(SEED)
    q = CAM.quantize(CAM.grey_mapping(tau, base.renderer) + rng.normal(0.0, base.renderer.noise_sd_gl, tau.shape), 8)
    ref = np.empty_like(q)
    for k in range(q.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(q[k]).save(buf, format="JPEG", quality=base.renderer.jpeg_quality)
        buf.seek(0)
        with Image.open(buf) as im:
            ref[k] = np.array(im.convert("L"))
    assert np.array_equal(out0, ref), "the no-table path changed"
    out1 = CAM.camera_chain(tau, with_tables(base).renderer, np.random.default_rng(SEED))
    assert not np.array_equal(out0, out1)
    # an explicit argument wins over the configuration (camera_calibration matches noise with fresh tables)
    other = [list(TABLES[1]), list(TABLES[0])]
    x = CAM.camera_chain(tau, with_tables(base).renderer, np.random.default_rng(SEED), qtables=other)
    y = CAM.camera_chain(tau, base.renderer, np.random.default_rng(SEED), qtables=other)
    assert np.array_equal(x, y)


def test_contract():
    cfg = with_tables(default_config())
    before = cfg.renderer
    out = CAM.camera_chain(tau_planes(), cfg.renderer, np.random.default_rng(SEED))
    assert out.shape == (3, 40, 48) and out.dtype == np.uint8
    assert cfg.renderer == before and isinstance(cfg.renderer.jpeg_qtables[0], tuple)
    two_d = CAM.camera_chain(tau_planes()[0], cfg.renderer, np.random.default_rng(SEED))
    assert two_d.shape == (40, 48)


def test_determinism():
    cfg = with_tables(default_config())
    a = small_block(cfg, np.random.default_rng(SEED))
    b = small_block(cfg, np.random.default_rng(SEED))
    assert np.array_equal(a, b)


def test_edge_cases():
    import PIL
    for v in ("8.2.0", "8.2", "7.0.0", "8.2.9.post1"):
        try:
            jpeg_tables.check_pillow_table_order(v)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Pillow %s accepted" % v)
    for v in ("8.3.0", "8.3", "9.0.0.dev0", "10.4.0", "12.3.0"):
        jpeg_tables.check_pillow_table_order(v)
    img = np.full((16, 16), 200, dtype=np.uint8)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(img).save(buf, format="JPEG", qtables=[list(TABLES[0])])
    data = buf.getvalue()
    real = PIL.__version__
    PIL.__version__ = "8.2.0"
    try:
        for call in (lambda: CAM.jpeg_roundtrip(img, qtables=[list(TABLES[0])]),
                     lambda: jpeg_tables.qtables_from_jpeg(data)):
            try:
                call()
            except RuntimeError as exc:
                assert "8.3" in str(exc), exc
            else:
                raise AssertionError("tables accepted under Pillow 8.2.0")
        CAM.jpeg_roundtrip(img, quality=85)          # the quality path does not depend on the table order
    finally:
        PIL.__version__ = real


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
