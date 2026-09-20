#!/usr/bin/env python3
"""Structural audit of an emitted .hoc, quarantine, and the arbour-angle check.

Ports, for the headless P1 driver, of the Colab-only helpers in
`Stage 1/Morphology_exporter_notebook.py` CELL 6a (audit_hoc_geometry,
classify_violations, _verdict, _merge_hoc_verdict, neuron_validate,
_quarantine; lines 483-706 on 2026-09-20) and of
`towards_eeg/structure/alignment_plots.py` (arbour_direction, angle_from_z_deg;
lines 101-140), which P1 cannot import because that module imports
matplotlib at load time. Logic unchanged; only the notebook globals became
arguments.

The structural audit is COMPLEMENTARY to the propagation gate, not a
duplicate: the gate asks "does a transient reach every compartment", this
asks "is the FILE a well-formed NEURON model". Neither promotes -- the export
already committed -- so the audit's job is to quarantine what the electrical
test did not catch.

Pure ASCII, LF only. NEURON is used only in `neuron_validate`, in a
SUBPROCESS (h.load_file is global and cumulative; one interpreter per file).
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

MODULE_VERSION = "p1_hoc_audit v1.0"

STRUCTURAL, SOFT = "structural", "soft"
MIN_SOMA_DIAM_UM = 4.0
RETIRED_DOMAIN_ARRAYS = frozenset({"apic_dend", "basal_dend"})


def audit_hoc_geometry(path):
    """Structural audit of an emitted .hoc. No NEURON needed."""
    txt = open(path, "r", encoding="ascii").read()

    arrays = {m[0]: int(m[1]) for m in re.findall(r"create (\w+)\[(\d+)\]", txt)}
    secs = {"%s[%d]" % (a, n) for a, k in arrays.items() for n in range(k)}

    conn = re.findall(
        r"connect\s+(\w+\[\d+\])\(([\d.]+)\),\s*(\w+\[\d+\])\(([\d.]+)\)", txt)
    child_of = {c: p for c, _, p, _ in conn}
    kids = {}
    for c, _, p, _ in conn:
        kids.setdefault(p, []).append(c)

    roots = sorted(secs - set(child_of))
    somatic = sorted(s for s in secs if s.startswith("soma"))

    start = "soma[0]" if "soma[0]" in secs else (roots[0] if roots else None)
    seen, stack = set(), ([start] if start else [])
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        stack.extend(kids.get(s, []))
    orphans = sorted(secs - seen)

    cur, pts, n_declared, unparsable = None, {}, {}, []
    for line in txt.splitlines():
        m = re.match(r"\s*(\w+\[\d+\])\s*\{", line)
        if m:
            cur = m.group(1)
            pts.setdefault(cur, [])
        m2 = re.search(r"pt3dadd\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)",
                       line)
        if m2 and cur is not None:
            n_declared[cur] = n_declared.get(cur, 0) + 1
            try:
                pts[cur].append([float(g.strip()) for g in m2.groups()])
            except ValueError:
                unparsable.append(cur)

    zero_len, bad_diam, few_pts, nonfinite, lengths = [], [], [], [], {}
    for s in sorted(secs):
        P = pts.get(s, [])
        if len(P) < 2:
            few_pts.append(s)
            continue
        A = np.asarray(P, dtype=float)
        if not np.isfinite(A).all():
            nonfinite.append(s)
            continue
        L = float(np.sum(np.linalg.norm(np.diff(A[:, :3], axis=0), axis=1)))
        lengths[s] = L
        if L <= 1e-9:
            zero_len.append(s)
        if A[:, 3].min() <= 0.0:
            bad_diam.append(s)

    dropped = sorted(s for s, n in n_declared.items() if len(pts.get(s, [])) != n)
    diam_max = {s: float(np.asarray(P)[:, 3].max()) for s, P in pts.items() if P}
    biggest = max(diam_max, key=diam_max.get) if diam_max else None

    return {"path": str(path), "n_sections": len(secs), "arrays": arrays,
            "n_connect": len(conn), "roots": roots, "somatic_sections": somatic,
            "n_reachable": len(seen), "orphans": orphans,
            "zero_length": zero_len, "nonpositive_diam": bad_diam,
            "few_pt3d": few_pts, "nonfinite": nonfinite,
            "unparsable_pt3d": sorted(set(unparsable)), "dropped_pt3d": dropped,
            "total_length_um": float(sum(lengths.values())),
            "largest_diameter_section": biggest,
            "largest_diameter_um": (float(diam_max[biggest]) if biggest
                                    else float("nan")),
            "soma_diameter_um": float(diam_max.get("soma[0]", float("nan")))}


def classify_violations(geo, neuron_rep=None, min_soma_diam_um=MIN_SOMA_DIAM_UM):
    """Q6 ladder: STRUCTURAL -> fail, SOFT -> pass_low_confidence."""
    v = {STRUCTURAL: [], SOFT: []}

    if len(geo["roots"]) != 1:
        v[STRUCTURAL].append("I-15: %d root sections %r" % (
            len(geo["roots"]), geo["roots"][:6]))
    if len(geo["somatic_sections"]) != 1:
        v[STRUCTURAL].append("I-15: %d somatic sections" % (
            len(geo["somatic_sections"]),))
    if geo["roots"] and geo["somatic_sections"] and (
            geo["roots"][0] != geo["somatic_sections"][0]):
        v[STRUCTURAL].append("I-15: root %s is not the soma" % geo["roots"][0])
    if geo["orphans"]:
        v[STRUCTURAL].append("debris: %d of %d sections unreachable, e.g. %r" % (
            len(geo["orphans"]), geo["n_sections"], geo["orphans"][:5]))
    if geo["n_connect"] != geo["n_sections"] - 1:
        v[STRUCTURAL].append("not a tree: %d connect for %d sections" % (
            geo["n_connect"], geo["n_sections"]))

    for key, msg in (("zero_length", "zero-length sections (NaN nseg)"),
                     ("nonpositive_diam", "non-positive diameters"),
                     ("few_pt3d", "sections with fewer than 2 pt3d points"),
                     ("nonfinite", "non-finite coordinates"),
                     ("unparsable_pt3d", "unparsable pt3dadd coordinates"),
                     ("dropped_pt3d", "pt3d points lost during parsing")):
        if geo[key]:
            v[STRUCTURAL].append("%s: %d %r" % (msg, len(geo[key]), geo[key][:5]))

    retired = sorted(set(geo["arrays"]) & set(RETIRED_DOMAIN_ARRAYS))
    if retired:
        v[STRUCTURAL].append("retired domain arrays present: %r" % retired)

    if geo["largest_diameter_section"] != "soma[0]":
        v[SOFT].append("I-18: largest-calibre section is %s (%.3f um), not the "
                       "soma (%.3f um)" % (geo["largest_diameter_section"],
                                           geo["largest_diameter_um"],
                                           geo["soma_diameter_um"]))
    if not (geo["soma_diameter_um"] >= float(min_soma_diam_um)):
        v[SOFT].append("I-18: soma diameter %.3f um is below the %.2f um floor" % (
            geo["soma_diameter_um"], float(min_soma_diam_um)))

    if neuron_rep is not None and neuron_rep.get("neuron_available"):
        if neuron_rep.get("I16_no_synapse_sections") is False:
            v[STRUCTURAL].append("I-16: synapse label in the section vocabulary")
        if neuron_rep.get("I13a_vocabulary_admissible") is False:
            v[STRUCTURAL].append("I-13a: inadmissible arrays %r" % (
                neuron_rep.get("arrays"),))
        if neuron_rep.get("I15_single_soma_is_root") is False:
            v[STRUCTURAL].append("I-15 (NEURON): roots=%r" % (
                neuron_rep.get("root_sections"),))
        n_sec = neuron_rep.get("n_sections")
        if n_sec is not None and n_sec != geo["n_sections"]:
            v[STRUCTURAL].append("NEURON loaded %d sections, file declares %d" % (
                n_sec, geo["n_sections"]))
    return v


def verdict(v):
    if v[STRUCTURAL]:
        return "fail"
    return "pass_low_confidence" if v[SOFT] else "pass"


def merge_verdict(existing, hoc_verdict):
    """The emitted file can only LOWER confidence, never raise it."""
    order = {"pass": 0, "pass_low_confidence": 1, "fail": 2}
    return max([existing, hoc_verdict], key=lambda x: order.get(x, 2))


_NEURON_SNIPPET = r"""
import json, sys
sys.path[:0] = %r
import morphology_exporter as mx
print("@@@" + json.dumps(mx.validate_hoc(%r, min_soma_diam_um=%r), default=str))
"""


def neuron_validate(hoc_path, sys_path=None, timeout=300):
    """morphology_exporter.validate_hoc in an isolated interpreter. Returns
    {'neuron_available': False, ...} rather than raising when NEURON is
    absent, so the audit degrades to the file-only checks."""
    paths = [p for p in (sys_path if sys_path is not None else sys.path) if p]
    code = _NEURON_SNIPPET % (paths, str(hoc_path), float(MIN_SOMA_DIAM_UM))
    try:
        r = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=timeout)
        for line in r.stdout.splitlines():
            if line.startswith("@@@"):
                return json.loads(line[3:])
        return {"neuron_available": False, "import_error": (r.stderr or "")[-300:]}
    except Exception as exc:                                 # noqa: BLE001
        return {"neuron_available": False, "import_error": repr(exc)}


def quarantine(output_dir, nid, files=None):
    """Move every artefact of a structurally invalid cell into
    <output_dir>/_quarantine. Matches on the neuron_<nid> prefix rather than
    on a registered file list, so an artefact the exporter wrote but did not
    register travels too. Returns (qdir, moved_names). `files`, if given, is
    rewritten in place to the new locations."""
    qdir = os.path.join(output_dir, "_quarantine")
    os.makedirs(qdir, exist_ok=True)
    prefix = "neuron_%s" % nid
    moved = []
    for f in sorted(os.listdir(output_dir)):
        src_p = os.path.join(output_dir, f)
        if f.startswith(prefix) and os.path.isfile(src_p):
            shutil.move(src_p, os.path.join(qdir, f))
            moved.append(f)
    if files:
        for k, val in list(files.items()):
            pth = val if isinstance(val, str) else (
                val.get("path") if isinstance(val, dict) else None)
            if not pth:
                continue
            newp = os.path.join(qdir, os.path.basename(pth))
            if isinstance(val, str):
                files[k] = newp
            else:
                val["path"] = newp
    return qdir, moved


def arbour_direction(df, class_column="compartment_class",
                     classes=("dend", "apic_dend", "basal_dend"),
                     percentile=90.0):
    """Unit vector from the soma to the centre of mass of the DISTAL dendrite
    (alignment_plots.arbour_direction, verbatim). The frame must be ALIGNED
    (soma at the origin). Returns None when there is no dendrite."""
    if class_column in df.columns:
        d = df[df[class_column].astype(str).isin(classes)]
    else:
        d = df
    if not len(d):
        return None
    P = d[["x", "y", "z"]].to_numpy(float)
    r = np.linalg.norm(P, axis=1)
    if not np.isfinite(r).any() or r.max() <= 0:
        return None
    tips = P[r >= np.percentile(r, percentile)]
    if not len(tips):
        return None
    com = tips.mean(axis=0)
    n = np.linalg.norm(com)
    return None if n == 0 else com / n


def angle_from_z_deg(v):
    """Angle between a unit vector and +z, in degrees. None -> NaN."""
    if v is None:
        return float("nan")
    c = float(np.clip(np.dot(v, [0.0, 0.0, 1.0]), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))
