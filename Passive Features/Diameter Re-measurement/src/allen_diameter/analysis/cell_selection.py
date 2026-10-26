"""The class of a cell, the acquisition check's comparison and summary, and
the stratified selection of S3 -- Block 16.1 in specs/SPEC.md (D-085.1,
D-085.3). Pure functions of records (loading/cell_pool.py) and the
configuration; nothing here reads a file or the network.

  neuron_class          Allen's dendrite type -> "exc", "inh" or None (survey.class_map)
  matches_acquisition   a cell's plane spacing and pixel size against acquisition.dz_um, res0_um
  acquisition_summary   the distinct values over the cells, the matching count, the cells that differ
  select_cells          survey.cells_per_group cells per (class, layer), drawn by a seeded permutation

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def neuron_class(dendrite_type, class_map):
    """The class class_map gives the dendrite type (exact match after strip and
    lower case); None for an unmapped or missing type."""
    if dendrite_type is None:
        return None
    key = str(dendrite_type).strip().lower()
    for dtype, cls in class_map:
        if key == str(dtype).strip().lower():
            return str(cls).strip()
    return None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def matches_acquisition(record, cfg):
    """(ok, reason) for one acquisition record (Block 16.1 (c)): ok when the
    record has no error, at least one plane, and its plane spacing dz_um and
    pixel size pixel_um each lie within survey.acquisition_rel_tol (relative)
    of acquisition.dz_um and acquisition.res0_um; a NaN never matches. reason:
    '' or 'error', 'no_planes', 'dz', 'pixel', 'dz;pixel'."""
    if record.get("error"):
        return False, "error"
    if not record.get("n_planes"):
        return False, "no_planes"
    tol = float(cfg.survey.acquisition_rel_tol)
    reasons = []
    for key, ref, name in (("dz_um", cfg.acquisition.dz_um, "dz"), ("pixel_um", cfg.acquisition.res0_um, "pixel")):
        v = _num(record.get(key))
        if not (math.isfinite(v) and abs(v - ref) <= tol * ref):
            reasons.append(name)
    return (not reasons), ";".join(reasons)


def _value_counts(records, key):
    out = {}
    for r in records:
        v = _num(r.get(key))
        label = "%.4f" % v if math.isfinite(v) else (
            "multiple: " + str(r.get("pixel_values")) if key == "pixel_um" and r.get("pixel_values") else "nan")
        out[label] = out.get(label, 0) + 1
    return dict(sorted(out.items()))


def acquisition_summary(records, cfg):
    """The acquisition check over the cells: how many were checked and failed,
    the distinct plane spacings and pixel sizes (um, rounded to 1e-4) with
    their counts, how many match the configuration and why the others do not."""
    mismatch = {}
    n_match = 0
    for r in records:
        ok, why = matches_acquisition(r, cfg)
        if ok:
            n_match += 1
        else:
            mismatch[str(int(r["specimen_id"]))] = why
    return dict(n_cells=len(records), n_failed=sum(1 for r in records if r.get("error")),
                dz_um_counts=_value_counts(records, "dz_um"), pixel_um_counts=_value_counts(records, "pixel_um"),
                n_match=n_match, mismatch=mismatch,
                configuration=dict(dz_um=cfg.acquisition.dz_um, pixel_um=cfg.acquisition.res0_um,
                                   rel_tol=cfg.survey.acquisition_rel_tol))


@dataclass(frozen=True)
class Selection:
    selected: Tuple[dict, ...]              # the pool records chosen, with neuron_class, group, rank
    groups: Dict[str, Tuple[int, int]]      # "class|layer" -> (cells available, cells selected)
    excluded: Dict[int, str]                # specimen_id -> why it is in no group
    seed: int


def group_rng(seed, key):
    """The generator of group `key`: numpy's default_rng seeded with (seed,
    crc32 of the key), so a group's draw does not depend on the other groups."""
    return np.random.default_rng([int(seed), zlib.crc32(str(key).encode("utf-8"))])


def select_cells(pool, cfg, acquisition=None, require_match=False):
    """Block 16.1 (d): the pool's cells with a class (and, with require_match,
    a matching acquisition record) grouped by (class, layer); in each group,
    taken in increasing specimen_id, the first survey.cells_per_group of the
    permutation drawn by group_rng(survey.selection_seed, key), or all of a
    smaller group. Returns a Selection."""
    sv = cfg.survey
    acq = None if acquisition is None else {int(r["specimen_id"]): r for r in acquisition}
    if require_match and acq is None:
        raise ValueError("select_cells: require_match needs the acquisition records")
    groups, classes, excluded = {}, {}, {}
    for rec in sorted(pool, key=lambda r: int(r["specimen_id"])):
        sid = int(rec["specimen_id"])
        cls = neuron_class(rec.get("dendrite_type"), sv.class_map)
        if cls is None:
            excluded[sid] = "no_class"
            continue
        if rec.get("layer") is None or str(rec.get("layer")).strip() == "":
            excluded[sid] = "no_layer"
            continue
        if require_match:
            a = acq.get(sid)
            if a is None:
                excluded[sid] = "acquisition_missing"
                continue
            ok, why = matches_acquisition(a, cfg)
            if not ok:
                excluded[sid] = "acquisition_mismatch:" + why
                continue
        key = "%s|%s" % (cls, str(rec["layer"]).strip())
        groups.setdefault(key, []).append(rec)
        classes[key] = cls
    selected, counts = [], {}
    for key in sorted(groups):
        cells = groups[key]
        order = group_rng(sv.selection_seed, key).permutation(len(cells))
        n = min(int(sv.cells_per_group), len(cells))
        for rank, idx in enumerate(order[:n]):
            selected.append(dict(cells[int(idx)], neuron_class=classes[key], group=key, rank=rank))
        counts[key] = (len(cells), n)
    return Selection(tuple(selected), counts, excluded, int(sv.selection_seed))
