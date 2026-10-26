"""The S3 survey folder -- Block 16.2 (g) in specs/SPEC.md (D-085.12): one
folder per cell, the markers of its finished steps, the status table.

  <out>/<specimen>/cell.json          the selection record, the SWC path, the alignment once known
  <out>/<specimen>/done/<step>.json   the marker of a finished step (its summary, seconds, config hash, time)
  <out>/<specimen>/errors.log         the tracebacks of failed steps, appended
  <out>/status.csv                    one row per cell: the state of each step, the last error

Every JSON is written to a temporary file and moved into place, with NaN and
infinities written as null. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os

from . import cell_pool


def clean(obj):
    """obj with numpy scalars and arrays made Python, NaN and infinities None,
    tuples made lists: what strict JSON holds."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, obj):
    """clean(obj) as indented JSON, written to path.tmp and moved into place."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(clean(obj), f, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


class CellDir:
    """The folder of one cell of the survey."""

    def __init__(self, out_dir, specimen_id):
        self.sid = int(specimen_id)
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, str(self.sid))

    def path_of(self, *parts):
        """A path inside the cell's folder, its directory created."""
        p = os.path.join(self.path, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def marker_path(self, step):
        return os.path.join(self.path, "done", "%s.json" % step)

    def done(self, step):
        return os.path.exists(self.marker_path(step))

    def context(self):
        p = os.path.join(self.path, "cell.json")
        return read_json(p) if os.path.exists(p) else {}

    def save_context(self, ctx):
        write_json(self.path_of("cell.json"), ctx)

    def log_error(self, step, text):
        with open(self.path_of("errors.log"), "a") as f:
            f.write("== %s %s\n%s\n" % (_utc_now(), step, text))


def write_marker(cell, step, summary, seconds, config_hash):
    """The marker of a finished step, written after the step's own files."""
    write_json(cell.path_of("done", "%s.json" % step),
               dict(step=step, specimen_id=cell.sid, summary=summary, seconds=float(seconds), config_hash=config_hash,
                    finished_utc=_utc_now()))


def read_marker(cell, step):
    p = cell.marker_path(step)
    return read_json(p) if os.path.exists(p) else None


def clear_markers(cell, steps):
    """Remove the markers of steps (those that exist); returns the ones removed."""
    gone = []
    for s in steps:
        p = cell.marker_path(s)
        if os.path.exists(p):
            os.remove(p)
            gone.append(s)
    return gone


def read_selection(path):
    """The selected cells of Block 16.1 (selection.csv), in the file's order."""
    return cell_pool.read_table(path, cell_pool.SELECTION_TYPES)


def write_status(out_dir, rows, steps):
    """status.csv: specimen_id, group, one column per step, error; rows are
    dicts with those keys (a missing step is 'pending')."""
    cols = ["specimen_id", "group"] + list(steps) + ["error"]
    path = os.path.join(out_dir, "status.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "pending" if c in steps else "") for c in cols})
    os.replace(tmp, path)
    return path
