"""JPEG quantization tables: read them from a fetched Allen crop, save them,
load them for the camera chain -- Block 4 in specs/SPEC.md.

The camera chain (model/camera.py) re-encodes synthetic planes with Allen's
own tables when they are known (procedure s.3.6 step 6). Pillow exposes the
tables of a decoded JPEG as Image.open(f).quantization, a dict {index: 64
integers}; they are stored here as a JSON list of 64-integer lists, in index
order. File I/O lives in loading/ (scientific-coding layout).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import io
import json


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
