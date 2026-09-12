"""sma_run -- offline helpers for the Spine Mesh Analysis driver (S0 -> S2).

Pure computation. NO network, NO plotting, NO Colab imports, so every function
here is exercised by smoke_test_sma_run.py with no credentials and no internet.
ASCII only, LF only, per TEEG_27 sec. 9.

Why this module exists
----------------------
The notebook previously called into `h01_segid_census` and `h01_fetch` with
hard-coded function names. That couples the driver to an API that changes, and
a rename surfaces as a NameError three cells deep. Everything that is not a
call into those modules now lives here, is named once, and is tested:

  * `resolve` / `call_with` -- find an entry point among candidate names and
    pass only the keyword arguments its signature actually accepts. A missing
    entry point raises immediately, with the module's full public surface in
    the message, instead of failing later with a wrong-argument TypeError.
  * `join_spine_labels` -- attach the labeller's spine partition BY POSITION.
    The reconstruction is re-rooted by navis (`reroot_entire_neuron_navis`),
    which renumbers nodes, so joining on `id` is unsafe.
  * `spine_components` / `g0_report` -- the G0 quantity: what fraction of
    spines sit on a segment id other than the cell's primary one.
  * `g2_verdict` -- the G2 gate, tolerant of whatever shape
    `h01_fetch.registration_metrics` returns.

Gate constants are stated once, here, and imported by the notebook.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Gate constants (TEEG_27 sec. 6)                                              #
# --------------------------------------------------------------------------- #
G2_MEDIAN_LO = 0.90
G2_MEDIAN_HI = 1.15
G2_MAX_OUTLIER_FRACTION = 0.01
# Used ONLY when the fetch module reports no outlier fraction of its own. It is
# this module's definition, not the pipeline's, and g2_verdict says so in its
# output so the number is never mistaken for the module's own.
FALLBACK_OUTLIER_BAND = (0.5, 2.0)

JOIN_TOL_NM = 1.0


# --------------------------------------------------------------------------- #
# 1. Module surface discovery and safe dispatch                                #
# --------------------------------------------------------------------------- #
class EntryPointNotFound(LookupError):
    """No candidate name matched a callable in the module."""


def public_callables(module) -> Dict[str, str]:
    """Map {name: signature} for functions defined IN this module (not imported)."""
    out: Dict[str, str] = {}
    for name, obj in vars(module).items():
        if name.startswith("_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != getattr(module, "__name__", None):
            continue
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = "(signature unavailable)"
        out[name] = sig
    return dict(sorted(out.items()))


def public_constants(module) -> Dict[str, Any]:
    """UPPER_CASE scalar/tuple module constants, e.g. DEFAULT_SEG_CLOUDPATH."""
    out: Dict[str, Any] = {}
    for name, obj in vars(module).items():
        if name.startswith("_") or callable(obj) or inspect.ismodule(obj):
            continue
        if name.isupper() and isinstance(obj, (str, int, float, bool, tuple, list)):
            out[name] = obj
    return dict(sorted(out.items()))


def resolve(module, candidates: Sequence[str]) -> Tuple[str, Callable]:
    """Return (name, fn) for the first candidate present as a callable.

    Raises EntryPointNotFound listing the module's public surface, so the fix is
    one edit to the candidate list rather than a hunt through the source.
    """
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    raise EntryPointNotFound(
        "none of %s found in %r.\npublic callables: %s"
        % (list(candidates), getattr(module, "__name__", module),
           json.dumps(public_callables(module), indent=2))
    )


def call_with(fn: Callable, *positional, **candidate_kwargs):
    """Call fn with only the keyword arguments its signature accepts.

    Returns (result, record) where record documents what was actually passed --
    keep it, it goes in the run log. Keyword arguments that would collide with a
    supplied positional argument are dropped.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    takes_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kw = dict(candidate_kwargs) if takes_var_kw else {
        k: v for k, v in candidate_kwargs.items() if k in params}
    positional_names = [n for n, p in params.items()
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for n in positional_names[:len(positional)]:
        kw.pop(n, None)
    record = {"called": getattr(fn, "__name__", repr(fn)),
              "n_positional": len(positional),
              "kwargs_passed": sorted(kw),
              "kwargs_dropped": sorted(set(candidate_kwargs) - set(kw))}
    return fn(*positional, **kw), record


# --------------------------------------------------------------------------- #
# 2. Spine partition: positional join and connected components                 #
# --------------------------------------------------------------------------- #
def join_spine_labels(nodes: pd.DataFrame, spines: pd.DataFrame,
                      tol_nm: float = JOIN_TOL_NM,
                      label_col: str = "annotated_type",
                      out_col: str = "spine_label"):
    """Attach `spines[label_col]` to `nodes` by nearest position in nanometres.

    Returns (nodes_copy_with_out_col, report). Nodes with no partner inside
    tol_nm get 'unmatched'. A large max_distance_nm means the two files are NOT
    in the same frame -- do not proceed on the join in that case.
    """
    from scipy.spatial import cKDTree

    for frame, name in ((nodes, "nodes"), (spines, "spines")):
        missing = [c for c in ("x", "y", "z") if c not in frame.columns]
        if missing:
            raise KeyError("%s is missing %s" % (name, missing))
    if label_col not in spines.columns:
        raise KeyError("spines has no column %r; have %s" % (label_col, list(spines.columns)))

    tgt = spines[["x", "y", "z"]].to_numpy(dtype=float)
    src = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    dist, idx = cKDTree(tgt).query(src, k=1)
    lab = spines[label_col].astype(str).str.strip().str.lower().to_numpy()
    matched = dist <= tol_nm
    out = nodes.copy()
    out[out_col] = np.where(matched, lab[idx], "unmatched")

    report = {
        "n_nodes": int(len(out)),
        "n_spine_rows": int(len(spines)),
        "max_distance_nm": float(dist.max()) if len(dist) else float("nan"),
        "median_distance_nm": float(np.median(dist)) if len(dist) else float("nan"),
        "n_unmatched": int((~matched).sum()),
        "tol_nm": float(tol_nm),
        "label_counts": pd.Series(out[out_col]).value_counts().to_dict(),
        "frame_consistent": bool(len(dist) and dist.max() <= tol_nm),
    }
    return out, report


def label_vocabularies(spine_density=None, spine_geometry=None) -> Dict[str, Any]:
    """The canonical label sets, taken FROM THE PROJECT MODULES when supplied.

    Hardcoding ("spine", "head", "neck") here would be a second definition of
    the partition, free to drift from spine_density's. Pass the modules and the
    values come from them; the fallbacks below exist only so this module
    imports on a bare environment, and `sourced_from_project` records which
    happened.
    """
    out = {
        "spine_labels": tuple(getattr(spine_density, "SPINE_LABELS",
                                      _FALLBACK_SPINE_LABELS)),
        "shaft_regex": str(getattr(spine_density, "SHAFT_REGEX", _FALLBACK_SHAFT_REGEX)),
        "default_radius_nm": float(getattr(spine_density, "DEFAULT_RADIUS_NM", 50.0)),
        "head_labels": tuple(getattr(spine_geometry, "HEAD_LABELS", ("head",))),
        "neck_labels": tuple(getattr(spine_geometry, "NECK_LABELS", ("neck",))),
        "sourced_from_project": bool(spine_density is not None),
        "spine_density_version": getattr(spine_density, "MODULE_VERSION", None),
        "spine_geometry_version": getattr(spine_geometry, "MODULE_VERSION", None),
    }
    return out


class LabellerFrameError(RuntimeError):
    """The labeller returned a frame that is not the input frame relabelled."""


def label_spines_project(nodes: pd.DataFrame, nid: int, labeller, threshold_nm: float,
                         *, output_dir: Optional[str] = None,
                         labeller_kwargs: Optional[Dict[str, Any]] = None):
    """Run the PROJECT's own spine labeller on an in-memory node table.

    This calls `labeller.label_dendritic_spines_robust` -- the verbatim
    extraction pinned by spine_labeller.SOURCE_SHA256 -- through the same
    tempdir adapter Stage 1 uses in `make_label_fn`, because that function is
    disk-based and keys on the filename. No part of the spine definition is
    reimplemented here.

    `threshold_nm` MUST be passed explicitly. The extracted function's own
    default is 5000.0 nm, left unchanged so its hash still matches the source;
    the project value is morphology_exporter.SPINE_LENGTH_THRESHOLD_NM =
    4000.0, supplied at the call site. Using the default silently would give a
    partition that disagrees with every F in the Stage 1 bank.

    If `output_dir` is given, the labeller itself writes
    neuron_{nid}_spines.csv there -- this module never writes that file.

    Returns (labelled_frame, provenance). Raises LabellerFrameError if the
    returned frame is not the input with `annotated_type` rewritten: the
    positional join downstream depends on the geometry being untouched.
    """
    import shutil
    import tempfile

    fn = getattr(labeller, "label_dendritic_spines_robust", None)
    if not callable(fn):
        raise EntryPointNotFound(
            "labeller module %r has no label_dendritic_spines_robust"
            % getattr(labeller, "__name__", labeller))

    required = {"id", "p", "x", "y", "z", "annotated_type"}
    missing = sorted(required - set(nodes.columns))
    if missing:
        raise KeyError("node table is missing %s, which the labeller requires" % missing)

    kwargs = dict(labeller_kwargs or {})
    tmp = tempfile.mkdtemp()
    try:
        nodes.to_csv(os.path.join(tmp, "neuron_%s.csv" % nid), index=False)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        out = fn([nid], input_dir=tmp, output_dir=output_dir,
                 spine_length_threshold_nm=float(threshold_nm), **kwargs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    frame = out[nid] if isinstance(out, dict) and nid in out else out
    if not isinstance(frame, pd.DataFrame):
        raise LabellerFrameError(
            "labeller returned %s, not a DataFrame -- neuron %s was probably "
            "skipped (it prints [WARN] and continues)." % (type(frame).__name__, nid))

    # The labeller must relabel, not move. Verify rather than assume: the
    # positional join in the driver is exact, and a silently transformed frame
    # would produce a plausible-looking but wrong partition.
    if len(frame) != len(nodes):
        raise LabellerFrameError("labeller returned %d rows for %d input rows"
                                 % (len(frame), len(nodes)))
    for col in ("id", "p", "x", "y", "z"):
        if not np.array_equal(np.asarray(frame[col]), np.asarray(nodes[col])):
            raise LabellerFrameError(
                "labeller altered column %r -- the frame is not the input "
                "relabelled, and the positional join would be invalid." % col)

    before = nodes["annotated_type"].astype(str).str.lower()
    after = frame["annotated_type"].astype(str).str.lower()
    provenance = {
        "function": "label_dendritic_spines_robust",
        "module": getattr(labeller, "__name__", None),
        "module_file": getattr(labeller, "__file__", None),
        "source_file": getattr(labeller, "SOURCE_FILE", None),
        "source_lines": list(getattr(labeller, "SOURCE_LINES", []) or []),
        "source_sha256": getattr(labeller, "SOURCE_SHA256", None),
        "threshold_nm": float(threshold_nm),
        "threshold_is_project_value": None,   # filled in by the caller
        "extra_kwargs": {k: v for k, v in kwargs.items()},
        "output_dir": output_dir,
        "spines_csv": (os.path.join(output_dir, "neuron_%s_spines.csv" % nid)
                       if output_dir else None),
        "n_nodes": int(len(frame)),
        "n_relabelled": int((before != after).sum()),
        "labels_before": before.value_counts().to_dict(),
        "labels_after": after.value_counts().to_dict(),
    }
    return frame, provenance


_FALLBACK_SPINE_LABELS = ("spine", "head", "neck")
_FALLBACK_SHAFT_REGEX = r"dendrite|apical|^1$"


def spine_mask(nodes: pd.DataFrame, col: str = "spine_label",
               spine_values: Optional[Iterable[str]] = None) -> np.ndarray:
    """Boolean mask of spine-labelled nodes.

    `spine_values` should come from label_vocabularies(spine_density=...) so the
    vocabulary has exactly one definition in the project.
    """
    vals = set(v.lower() for v in (spine_values if spine_values is not None
                                   else _FALLBACK_SPINE_LABELS))
    return nodes[col].astype(str).str.lower().isin(vals).to_numpy()


def spine_components(nodes: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Label maximal connected components of the spine-labelled subgraph.

    Edges are the (id, p) parent links with BOTH endpoints spine-labelled, which
    is the definition of a spine sigma in TEEG_21 sec. 3.1. Union-find, no
    external graph dependency. Non-spine nodes get -1.
    """
    ids = nodes["id"].to_numpy(dtype=np.int64)
    par = nodes["p"].to_numpy(dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(ids)}
    parent = np.arange(len(ids), dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(ids)):
        if not mask[i] or par[i] == -1:
            continue
        j = pos.get(int(par[i]))
        if j is None or not mask[j]:
            continue
        ra, rb = find(i), find(j)
        if ra != rb:
            parent[ra] = rb

    comp = np.full(len(ids), -1, dtype=np.int64)
    relabel: Dict[int, int] = {}
    for i in range(len(ids)):
        if not mask[i]:
            continue
        r = find(i)
        if r not in relabel:
            relabel[r] = len(relabel)
        comp[i] = relabel[r]
    return comp


def g0_report(nodes: pd.DataFrame, comp: np.ndarray, segid_col: str = "segid") -> Dict[str, Any]:
    """G0: the manifest and the foreign + mixed spine fraction.

    primary segid = the one carrying the most nodes overall.
    For each spine component sigma: 'primary' if every node is on the primary
    id, 'foreign' if none is, 'mixed' otherwise. A mixed spine straddles a
    segmentation boundary and its mesh cannot be pulled from one id.
    """
    if segid_col not in nodes.columns:
        raise KeyError("nodes has no %r column -- run S0 first" % segid_col)
    seg = nodes[segid_col].to_numpy()
    valid = pd.notna(seg)
    counts = pd.Series(seg[valid]).value_counts()
    primary = counts.index[0] if len(counts) else None

    classes: Dict[str, int] = {"primary": 0, "foreign": 0, "mixed": 0, "unknown": 0}
    for c in range(int(comp.max()) + 1 if comp.size and comp.max() >= 0 else 0):
        sel = comp == c
        s = pd.Series(seg[sel]).dropna().unique()
        if len(s) == 0:
            classes["unknown"] += 1
        elif len(s) == 1:
            classes["primary" if s[0] == primary else "foreign"] += 1
        else:
            classes["mixed"] += 1

    n_sigma = sum(classes.values())
    frac = (lambda k: (classes[k] / n_sigma) if n_sigma else float("nan"))
    return {
        "primary_segid": None if primary is None else int(primary),
        "manifest": [int(v) for v in counts.index.tolist()],
        "n_segids": int(len(counts)),
        "nodes_per_segid": {int(k): int(v) for k, v in counts.items()},
        "n_spine_components": int(n_sigma),
        "spine_class_counts": classes,
        "foreign_fraction": frac("foreign"),
        "mixed_fraction": frac("mixed"),
        "foreign_plus_mixed_fraction": (frac("foreign") + frac("mixed")) if n_sigma else float("nan"),
        "n_nodes_no_segid": int((~valid).sum()),
    }


# --------------------------------------------------------------------------- #
# 3. G2 verdict                                                                #
# --------------------------------------------------------------------------- #
_MEDIAN_KEYS = ("median_d_over_r", "median_dr", "d_over_r_median", "median_ratio", "median")
_OUTLIER_KEYS = ("outlier_fraction", "frac_outliers", "outlier_frac", "fraction_outliers")
_RATIO_KEYS = ("d_over_r", "dr", "ratio", "ratios", "d_over_r_per_node")


def _lookup(result: Any, names: Sequence[str]) -> Optional[Any]:
    """Case-insensitive fetch from a dict, Series, DataFrame column, or attribute."""
    lower = [n.lower() for n in names]
    if isinstance(result, dict):
        low = {str(k).lower(): v for k, v in result.items()}
        for n in lower:
            if n in low:
                return low[n]
    if isinstance(result, pd.Series):
        low = {str(k).lower(): v for k, v in result.items()}
        for n in lower:
            if n in low:
                return low[n]
    if isinstance(result, pd.DataFrame):
        low = {str(c).lower(): c for c in result.columns}
        for n in lower:
            if n in low:
                return result[low[n]].to_numpy()
    for n in names:
        if hasattr(result, n):
            got = getattr(result, n)
            # A pandas object exposes .median, .ratio-like helpers etc. as BOUND
            # METHODS. Returning one would be silently wrong, so require data.
            if not callable(got):
                return got
    if isinstance(result, (tuple, list)):
        for item in result:
            got = _lookup(item, names)
            if got is not None:
                return got
    return None


def g2_verdict(result: Any, band: Tuple[float, float] = FALLBACK_OUTLIER_BAND) -> Dict[str, Any]:
    """Apply G2 to whatever registration_metrics returned.

    Prefers the module's own median and outlier fraction. Falls back to a
    per-node d/r array, in which case `outlier_definition` records that the
    band is THIS module's, not the pipeline's.
    """
    med = _lookup(result, _MEDIAN_KEYS)
    frac = _lookup(result, _OUTLIER_KEYS)
    definition = "reported by fetch module"
    n = None

    if med is None or frac is None:
        arr = _lookup(result, _RATIO_KEYS)
        if arr is not None:
            a = np.asarray(arr, dtype=float)
            a = a[np.isfinite(a)]
            n = int(a.size)
            if a.size:
                if med is None:
                    med = float(np.median(a))
                if frac is None:
                    frac = float(np.mean((a < band[0]) | (a > band[1])))
                    definition = "computed here, |d/r| outside [%.2f, %.2f]" % band

    if med is None:
        return {"parsed": False, "reason": "no median or per-node d/r found",
                "raw_type": type(result).__name__, "raw_repr": repr(result)[:400]}

    med = float(med)
    median_ok = G2_MEDIAN_LO <= med <= G2_MEDIAN_HI
    outlier_ok = None if frac is None else bool(float(frac) < G2_MAX_OUTLIER_FRACTION)
    return {
        "parsed": True,
        "median_d_over_r": med,
        "median_ok": bool(median_ok),
        "median_band": [G2_MEDIAN_LO, G2_MEDIAN_HI],
        "outlier_fraction": None if frac is None else float(frac),
        "outlier_ok": outlier_ok,
        "outlier_max": G2_MAX_OUTLIER_FRACTION,
        "outlier_definition": definition,
        "n_finite_ratios": n,
        "G2_PASS": bool(median_ok and (outlier_ok is True)),
    }


# --------------------------------------------------------------------------- #
# 4. Mesh cache and run log                                                    #
# --------------------------------------------------------------------------- #
def mesh_cache_path(cache_dir: str, segid: int, lod: int = 0) -> str:
    return os.path.join(cache_dir, "mesh_%d_lod%d.npz" % (int(segid), int(lod)))


def save_mesh_npz(path: str, vertices: np.ndarray, faces: np.ndarray, **meta) -> str:
    """Cache a mesh as float64 vertices + int32 faces. float32 would quantise
    nanometre coordinates near 1e6 to ~0.1 nm, inside the registration tolerance."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path,
                        vertices=np.asarray(vertices, dtype=np.float64),
                        faces=np.asarray(faces, dtype=np.int32),
                        meta=json.dumps(meta))
    return path


def load_mesh_npz(path: str):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return z["vertices"], z["faces"], meta


def mesh_sanity(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, Any]:
    """Cheap structural checks on a fetched mesh, before any geometry is done."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    ext = (v.max(axis=0) - v.min(axis=0)) if v.size else np.zeros(3)
    return {
        "n_vertices": int(v.shape[0]),
        "n_faces": int(f.shape[0]),
        "finite": bool(np.isfinite(v).all()),
        "extent_nm": [float(e) for e in ext],
        "extent_um": [float(e) / 1000.0 for e in ext],
        "face_index_in_range": bool(f.size == 0 or (f.min() >= 0 and f.max() < v.shape[0])),
        "degenerate_faces": int(np.sum([len(set(row)) < 3 for row in f])) if f.size else 0,
    }


def bbox_agreement(vertices: np.ndarray, nodes: pd.DataFrame) -> Dict[str, Any]:
    """G1: mesh and node-table bounding boxes should agree to a few micrometres."""
    v = np.asarray(vertices, dtype=float)
    p = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    dlo = v.min(axis=0) - p.min(axis=0)
    dhi = v.max(axis=0) - p.max(axis=0)
    worst = float(np.max(np.abs(np.concatenate([dlo, dhi]))))
    return {"delta_low_nm": [float(x) for x in dlo],
            "delta_high_nm": [float(x) for x in dhi],
            "worst_abs_nm": worst, "worst_abs_um": worst / 1000.0}


def write_run_log(run_dir: str, cell_id: int, payload: Dict[str, Any]) -> str:
    """Append-only provenance record. Filename carries the timestamp, so a rerun
    never overwrites the previous attempt."""
    os.makedirs(run_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(run_dir, "run_%d_%s.json" % (int(cell_id), stamp))
    body = dict(payload)
    body["cell_id"] = int(cell_id)
    body["utc"] = stamp
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        json.dump(_jsonable(body), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (pd.Series,)):
        return _jsonable(obj.to_dict())
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


_MAGIC_PREFIXES = ("!",)


def _strip_ipython_magics(source: str) -> Tuple[str, list]:
    """Comment out IPython shell-magic lines ('!pip install ...') so the REST
    of the file can still be syntax-checked. Line numbers are preserved --
    each magic line becomes '#' + the same content, same length, same
    position -- so a genuine SyntaxError elsewhere still points at the right
    line.

    Only '!' is treated as a magic marker, deliberately NOT '%'. '!' can never
    legally start ANY Python line -- not a statement, not a continuation of an
    open bracket -- so stripping it is always safe. '%' is NOT safe: this
    codebase's own printf-style formatting routinely continues a statement
    onto a new line starting with '%', e.g.

        print("... %s ..."
              % (value,))

    A per-line '%'-prefix check cannot tell that apart from a '%timeit'-style
    line magic without tracking bracket depth, and getting it wrong silently
    deletes part of a live expression rather than merely mis-classifying a
    comment. This project only ever uses '!pip install', so the unambiguous
    '!'-only rule is what is implemented; if '%'-magics are ever needed here,
    they need a bracket-depth-aware detector, not this one.
    """
    out_lines, magic_lines = [], []
    for i, line in enumerate(source.splitlines(), start=1):
        if line.strip().startswith(_MAGIC_PREFIXES):
            out_lines.append("#" + line[1:])
            magic_lines.append(i)
        else:
            out_lines.append(line)
    return "\n".join(out_lines), magic_lines


def transfer_safety(paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Per-file ASCII / CRLF / compile audit. Two separate checks: an ASCII scan
    cannot find CRLF, because CR is itself ASCII (TEEG_27 sec. 9).

    IPython magic lines ('!pip install ...') are commented out before the
    syntax check, not excluded from it: a Colab driver file (CELL markers,
    '!pip install') is expected to live in CODE_DIR next to the library
    modules -- Stage 1's own colab_run_s1_full.py does exactly this -- and a
    naive py_compile on it fails on the FIRST magic line with no information
    about the rest of the file. Stripping just the magic lines means a genuine
    syntax error anywhere else in that same file is still caught, and is
    reported at its correct line number. Files with no magic lines behave
    exactly as before this function was fixed.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        rec: Dict[str, Any] = {}
        try:
            b = open(p, "rb").read()
        except OSError as exc:
            out[p] = {"readable": False, "error": str(exc)}
            continue
        rec["readable"] = True
        rec["bytes"] = len(b)
        rec["non_ascii"] = int(sum(1 for c in b if c > 127))
        rec["cr"] = int(b.count(b"\r"))
        rec["sha256_12"] = hashlib.sha256(b).hexdigest()[:12]
        if p.endswith(".py"):
            try:
                text = b.decode("ascii")
            except UnicodeDecodeError as exc:
                rec["compiles"] = False
                rec["compile_error"] = "not ASCII-decodable: %s" % exc
            else:
                stripped, magic_lines = _strip_ipython_magics(text)
                if magic_lines:
                    rec["ipython_magic_lines"] = magic_lines
                try:
                    compile(stripped, p, "exec")
                    rec["compiles"] = True
                except SyntaxError as exc:
                    rec["compiles"] = False
                    rec["compile_error"] = "line %s: %s" % (exc.lineno, exc.msg)
        rec["clean"] = bool(rec["non_ascii"] == 0 and rec["cr"] == 0
                            and rec.get("compiles", True))
        out[p] = rec
    return out
