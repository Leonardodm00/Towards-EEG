#!/usr/bin/env python3
"""Build the P1 manifest for ONE population (layer, cell_type).

    python3 build_p1_manifest.py --root ../h01 --layer L3 --cell-type exc \
        --ids 1302789404,1317492596,1333261412,1376890291 --out ../h01/p1/manifests/L3_exc.csv

    python3 build_p1_manifest.py --root ../h01 --layer L3 --cell-type exc \
        --ids-file p23_nids.txt --out ../h01/p1/manifests/L3_exc.csv

One manifest per population, one array job per manifest: that is the
campaign structure the user set (10 populations, L2-L6 x exc/inh, each
layer with its own alignment bank). The label carried per cell is ONLY
(layer, cell_type); nothing finer exists on the cluster.

For every id the builder CHECKS, rather than assumes, that
  * neurons/neuron_<id>.csv exists (refused otherwise);
  * synapses/neuron_<id>_synapses.csv exists (a missing file is allowed:
    the column is left empty and P1 exports without the redirect for that
    cell, and says so);
  * the id is in the layer's bank, alignment_metadata_L<n>.csv, column
    neuron_id. Bank membership is what ties the id to its layer on the
    cluster, so an id outside its bank is REFUSED unless --allow-unbanked
    is given, in which case layer_source is 'manifest' instead of 'bank'.

WHERE THE BANK LIVES. Not hard-coded, because it has moved once already.
`Alignment Metadata/Usage.py` writes the bank into its own `input_dir` --
the folder holding neuron_<id>.csv -- so on the cluster the banks sit in
<root>/neurons/ beside the skeletons (user, 2026-09-20), not in a separate
<root>/alignment/. Rather than swap one hard-coded subdirectory for
another, this builder SEARCHES, in order, <root>/neurons/,
<root>/alignment/ and <root>/, prints the location it used, and refuses
with every path it tried when the file is in none of them. `--bank`
(an explicit file) and `--bank-dir` (a directory) override the search;
the same name present in two searched locations is a WARNING naming both,
because two banks with one name can disagree and only the user knows
which is current.

Columns written: cell_id, layer, cell_type, neuron_csv, alignment_metadata,
synapse_csv, layer_source. Paths are relative to --root so the manifest is
portable between the laptop and the cluster.

Pure ASCII, LF only.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BUILDER_VERSION = "build_p1_manifest v1.1"
LAYERS = ("L2", "L3", "L4", "L5", "L6")
CELL_TYPES = ("exc", "inh")
# Search order for alignment_metadata_<layer>.csv, relative to --root.
# neurons/ first: that is where the extraction script writes it (its
# output_directory is its own input_dir). "." keeps a bank dropped at the
# campaign root working without a flag.
BANK_SUBDIRS = ("neurons", "alignment", ".")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="campaign root (H01_ROOT)")
    p.add_argument("--layer", required=True, choices=LAYERS)
    p.add_argument("--cell-type", required=True, choices=CELL_TYPES)
    p.add_argument("--ids", default=None, help="comma-separated cell ids")
    p.add_argument("--ids-file", default=None,
                   help="one id per line (np.savetxt %%d format, as Save nids writes)")
    p.add_argument("--out", required=True, help="manifest CSV to write")
    p.add_argument("--allow-unbanked", action="store_true",
                   help="accept ids absent from the layer's alignment bank")
    p.add_argument("--bank", default=None,
                   help="explicit bank CSV (absolute, or relative to --root); "
                        "overrides the search")
    p.add_argument("--bank-dir", default=None,
                   help="directory holding alignment_metadata_<layer>.csv "
                        "(absolute, or relative to --root); overrides the search")
    return p


def read_ids(ids=None, ids_file=None):
    if (ids is None) == (ids_file is None):
        raise SystemExit("give exactly one of --ids or --ids-file")
    if ids is not None:
        raw = [s.strip() for s in str(ids).split(",") if s.strip()]
    else:
        if not os.path.isfile(ids_file):
            raise SystemExit("--ids-file does not exist: %s" % ids_file)
        raw = [ln.strip() for ln in open(ids_file) if ln.strip()
               and not ln.strip().startswith("#")]
    out = []
    for s in raw:
        try:
            out.append(int(float(s)))
        except ValueError:
            raise SystemExit("not an integer id: %r" % s)
    if not out:
        raise SystemExit("no ids given")
    dupes = sorted({v for v in out if out.count(v) > 1})
    if dupes:
        raise SystemExit("duplicate ids: %s" % dupes[:5])
    return out


def bank_filename(layer):
    return "alignment_metadata_%s.csv" % layer


def _under_root(path, root):
    """POSIX path relative to root when it is inside root, else the absolute
    path. The manifest carries this, so a bank inside the campaign tree keeps
    the manifest portable laptop <-> cluster and one outside still works."""
    path, root = os.path.abspath(path), os.path.abspath(root)
    rel = os.path.relpath(path, root)
    if rel.startswith(os.pardir):
        return path
    return rel.replace(os.sep, "/")


def resolve_bank(root, layer, bank=None, bank_dir=None):
    """Locate the layer's alignment bank. Returns (abs_path, note).

    Search order and overrides are documented in the module docstring. Every
    path tried is named on failure: a bank that moved is the single most
    likely reason this builder refuses, and guessing costs a cluster round
    trip.
    """
    root = os.path.abspath(root)
    name = bank_filename(layer)
    if bank is not None and bank_dir is not None:
        raise SystemExit("give at most one of --bank or --bank-dir")
    if bank is not None:
        cand = bank if os.path.isabs(bank) else os.path.join(root, bank)
        if not os.path.isfile(cand):
            raise SystemExit("--bank does not exist: %s" % cand)
        return os.path.abspath(cand), "--bank"
    if bank_dir is not None:
        d = bank_dir if os.path.isabs(bank_dir) else os.path.join(root, bank_dir)
        cand = os.path.join(d, name)
        if not os.path.isfile(cand):
            raise SystemExit("--bank-dir has no %s: %s" % (name, cand))
        return os.path.abspath(cand), "--bank-dir"

    tried = [os.path.normpath(os.path.join(root, sub, name))
             for sub in BANK_SUBDIRS]
    found = [p for p in tried if os.path.isfile(p)]
    if not found:
        raise SystemExit(
            "alignment bank %s not found. Looked in:\n  %s\n"
            "Give --bank <file> or --bank-dir <dir> if it lives elsewhere."
            % (name, "\n  ".join(tried)))
    note = "found in %s/" % os.path.basename(os.path.dirname(found[0]))
    if len(found) > 1:
        note += " -- WARNING: %s also exists at %s; using the first. Pass " \
                "--bank to choose." % (name, ", ".join(found[1:]))
    return found[0], note


def read_bank_ids(path):
    """The set of neuron_ids in a bank. `neuron_id` is written by
    extract_alignment_metadata (Alignment Metadata/Extract_metadata.py,
    metadata_records); a file without it is a different table, not a bank."""
    df = pd.read_csv(path)
    if "neuron_id" not in df.columns:
        raise SystemExit("bank %s has no neuron_id column (has %s)"
                         % (path, list(df.columns)[:8]))
    ids = set(int(v) for v in df["neuron_id"].dropna().astype(np.int64))
    if not ids:
        raise SystemExit("bank %s has no usable neuron_id values" % path)
    return ids


def build_manifest(root, layer, cell_type, ids, allow_unbanked=False,
                   bank_path=None):
    """One manifest frame. `bank_path` is an already-resolved bank (main
    resolves it, so the location it used can be reported); when it is None the
    default search of resolve_bank applies."""
    root = os.path.abspath(root)
    if bank_path is None:
        bank_path, _ = resolve_bank(root, layer)
    bank_set = read_bank_ids(bank_path)
    bank_rel = _under_root(bank_path, root)
    rows, problems = [], []
    for cid in ids:
        # POSIX separators on purpose: the manifest is data that travels
        # laptop -> cluster; run_p1_export resolves it against --root
        ncsv = "neurons/neuron_%d.csv" % cid
        scsv = "synapses/neuron_%d_synapses.csv" % cid
        if not os.path.isfile(os.path.join(root, ncsv)):
            problems.append("%d: no skeleton at %s" % (cid, ncsv))
            continue
        in_bank = cid in bank_set
        if not in_bank and not allow_unbanked:
            problems.append("%d: not in %s (use --allow-unbanked to accept)"
                            % (cid, bank_rel))
            continue
        rows.append({"cell_id": cid, "layer": layer, "cell_type": cell_type,
                     "neuron_csv": ncsv,
                     "alignment_metadata": bank_rel,
                     "synapse_csv": scsv if os.path.isfile(os.path.join(root, scsv)) else "",
                     "layer_source": "bank" if in_bank else "manifest"})
    if problems:
        raise SystemExit("manifest refused, %d problem(s):\n  %s"
                         % (len(problems), "\n  ".join(problems)))
    return pd.DataFrame(rows, columns=["cell_id", "layer", "cell_type", "neuron_csv",
                                       "alignment_metadata", "synapse_csv",
                                       "layer_source"])


def main(argv=None):
    args = build_parser().parse_args(argv)
    ids = read_ids(args.ids, args.ids_file)
    bank_path, bank_note = resolve_bank(args.root, args.layer, bank=args.bank,
                                        bank_dir=args.bank_dir)
    m = build_manifest(args.root, args.layer, args.cell_type, ids,
                       args.allow_unbanked, bank_path=bank_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    m.to_csv(args.out, index=False, lineterminator="\n")
    n_syn = int((m["synapse_csv"] != "").sum())
    print("%s: %d cells (%s %s), %d with a synapse file -> %s"
          % (BUILDER_VERSION, len(m), args.layer, args.cell_type, n_syn, args.out))
    print("  bank %s (%s)" % (m["alignment_metadata"].iloc[0], bank_note))
    if n_syn < len(m):
        print("  NOTE: %d cell(s) without a synapse file will be exported WITHOUT "
              "the redirect" % (len(m) - n_syn))
    print("  array width for p1_export.pbs: #PBS -J 0-%d" % (len(m) - 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
