"""The soma's view as scripts/show_soma.py wrote it, read back -- SPEC Block 44 amendment 4 (D-239).

read_soma_view(out_dir, specimen) reads soma_<S>_block.npz (the crop: block, ks, valid, left, top, res0_um) and
soma_<S>.json (the traced soma's centre and radius, dz_um, k_traced, k_by_eye) from show_soma.py's --out directory and
checks that the two describe the same crop. Pure ASCII, LF.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SomaView:
    """The crop of the soma's planes and its geometry (image frame, full resolution)."""
    specimen: str
    block: np.ndarray        # (n_planes, H, W) grey levels as fetched (uint8 on the real stack; 255 where not recorded)
    ks: np.ndarray           # (n_planes,) int, the plane indices, increasing
    valid: np.ndarray        # (n_planes,) bool, True where the plane was recorded
    left: int                # px, the crop's first column in the image frame (pixel j's centre at (left + j) res0_um)
    top: int                 # px, the crop's first row
    res0_um: float           # um per px
    dz_um: float             # um per plane: plane k lies at depth k dz_um
    centre_um: np.ndarray    # (3,) um, the traced soma's centre in the image frame
    radius_um: float         # um, the traced soma's radius
    k_traced: int            # the recorded plane nearest the traced depth
    k_by_eye: Optional[int]  # the plane given to show_soma.py as --soma-k, else None


def read_soma_view(out_dir, specimen) -> SomaView:
    """Read soma_<S>_block.npz and soma_<S>.json from out_dir (show_soma.py --out DIR).

    FileNotFoundError naming the missing file; ValueError when the block is not (n_planes, height, width) of the
    JSON's crop, or when the two files disagree on the planes (ks, valid), the crop's corner or the pixel size, or
    when ks is not increasing."""
    s = str(specimen)
    npz_path = os.path.join(str(out_dir), "soma_%s_block.npz" % s)
    json_path = os.path.join(str(out_dir), "soma_%s.json" % s)
    for path in (npz_path, json_path):
        if not os.path.exists(path):
            raise FileNotFoundError("%s not found: run scripts/show_soma.py --out %s first (Cell 24 or 24 (b))"
                                    % (path, out_dir))
    with np.load(npz_path) as z:
        block = np.asarray(z["block"])
        ks = np.asarray(z["ks"]).astype(int).reshape(-1)
        valid = np.asarray(z["valid"]).astype(bool).reshape(-1)
        left, top, res0 = int(z["left"]), int(z["top"]), float(z["res0_um"])
    with open(json_path) as f:
        rec = json.load(f)
    crop = rec["crop"]
    shape = (len(ks), int(crop["height"]), int(crop["width"]))
    if block.ndim != 3 or tuple(block.shape) != shape or len(valid) != len(ks):
        raise ValueError("soma_%s_block.npz: block %s, ks %d, valid %d; the JSON's crop wants %s"
                         % (s, tuple(block.shape), len(ks), len(valid), shape))
    if [int(k) for k in ks] != [int(k) for k in rec["ks"]] or [bool(v) for v in valid] != [bool(v) for v in
                                                                                           rec["valid"]]:
        raise ValueError("soma_%s_block.npz and soma_%s.json disagree on the planes: run show_soma.py again" % (s, s))
    if (left, top) != (int(crop["left"]), int(crop["top"])) or not np.isclose(res0, float(crop["res0_um"]),
                                                                              rtol=1e-12, atol=0.0):
        raise ValueError("soma_%s_block.npz and soma_%s.json disagree on the crop's corner or pixel size" % (s, s))
    if len(ks) > 1 and np.any(np.diff(ks) <= 0):
        raise ValueError("soma_%s_block.npz: the planes are not in increasing order" % s)
    k_eye = rec.get("k_by_eye")
    return SomaView(specimen=s, block=block, ks=ks, valid=valid, left=left, top=top, res0_um=res0,
                    dz_um=float(rec["dz_um"]), centre_um=np.asarray(rec["centre_um"], dtype=float).reshape(3),
                    radius_um=float(rec["radius_um"]), k_traced=int(rec["k_traced"]),
                    k_by_eye=None if k_eye is None else int(k_eye))
