"""JPEG quantization tables: read them from a fetched Allen crop, save them,
load them for the camera chain -- Block 4 in specs/SPEC.md.

The camera chain (model/camera.py) re-encodes synthetic planes with Allen's
own tables when they are known (procedure s.3.6 step 6); they reach it as
RendererConfig.jpeg_qtables, the tables themselves [corrected 2026-10-07: a
file path, jpeg_qtables_file, was never read]. Pillow exposes the tables of a
decoded JPEG as Image.open(f).quantization, a dict {index: 64 integers}; they
are stored here as a JSON list of 64-integer lists, in index order. File I/O
lives in loading/ (scientific-coding layout).

Order of the 64 entries: natural (row-major), Pillow's convention for both
reading (quantization) and writing (save(qtables=...)) from 8.3.0 on.
Pillow 8.2.0's JpegImagePlugin.DQT stored the file's zigzag order; 8.3.0
de-zigzags it (read from the source of both wheels, 2026-10-07). Tables read
under one convention and written under the other are permuted, so both ends
refuse a Pillow older than 8.3 (check_pillow_table_order).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import io
import json

PILLOW_NATURAL_ORDER = (8, 3)       # first Pillow whose quantization tables are in natural order


def check_pillow_table_order(version=None):
    """RuntimeError when Pillow (or the given version string) is older than
    8.3, whose JPEG tables are in zigzag order instead of natural order."""
    if version is None:
        import PIL
        version = PIL.__version__
    parts = []
    for p in str(version).split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts) < PILLOW_NATURAL_ORDER:
        raise RuntimeError("Pillow %s orders JPEG quantization tables differently from Pillow >= 8.3 (zigzag "
                           "vs natural order): upgrade Pillow before reading or applying Allen's tables" % version)


def _check_tables(tables):
    out = []
    for t in tables:
        t = [int(x) for x in t]
        if len(t) != 64 or any(not (1 <= x <= 65535) for x in t):
            raise ValueError("a quantization table has 64 integers in [1, 65535]")
        out.append(t)
    if not out:
        raise ValueError("no quantization table")
    return out


def qtables_from_jpeg(data_or_path):
    """Quantization tables of a JPEG given as bytes or a path, as a list of
    64-integer lists in table-index order (Pillow's Image.quantization)."""
    from PIL import Image

    check_pillow_table_order()
    src = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else data_or_path
    with Image.open(src) as image:
        q = getattr(image, "quantization", None)
        if not q:
            raise ValueError("not a JPEG with quantization tables")
        return _check_tables([q[k] for k in sorted(q)])


def save_qtables(tables, path):
    """Write the tables as JSON (a list of 64-integer lists)."""
    with open(path, "w") as f:
        json.dump(_check_tables(tables), f)


def load_qtables(path):
    """Read tables written by save_qtables (or any JSON list of 64-integer
    lists, or a dict {index: list}) for camera.camera_chain(qtables=...)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data[k] for k in sorted(data, key=lambda s: int(s))]
    return _check_tables(data)


def collect_qtables(paths):
    """Distinct table sets among JPEG files, most common first: a list of
    (tables, count, first path). Files that are not JPEGs with quantization
    tables are counted under tables None (an HttpFetcher cache holds the
    bytes as served, so every crop of one stack should carry one set)."""
    groups = {}
    for p in paths:
        try:
            key = tuple(tuple(t) for t in qtables_from_jpeg(p))
        except (ValueError, OSError):
            key = None
        if key not in groups:
            groups[key] = [0, p]
        groups[key][0] += 1
    out = [(None if k is None else [list(t) for t in k], n, first) for k, (n, first) in groups.items()]
    return sorted(out, key=lambda x: (x[0] is None, -x[1]))
