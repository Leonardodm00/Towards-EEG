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
  * the id is in the layer's bank, alignment/alignment_metadata_L<n>.csv,
    column neuron_id. Bank membership is what ties the id to its layer on
    the cluster, so an id outside its bank is REFUSED unless
    --allow-unbanked is given, in which case layer_source is 'manifest'
    instead of 'bank'.

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

BUILDER_VERSION = "build_p1_manifest v1.0"
LAYERS = ("L2", "L3", "L4", "L5", "L6")
CELL_TYPES = ("exc", "inh")


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


def bank_ids(root, layer):
    path = os.path.join(root, "alignment", "alignment_metadata_%s.csv" % layer)
    if not os.path.isfile(path):
        raise SystemExit("alignment bank not found: %s" % path)
    bank = pd.read_csv(path)
    if "neuron_id" not in bank.columns:
        raise SystemExit("bank %s has no neuron_id column (has %s)"
                         % (path, list(bank.columns)[:8]))
    return path, set(int(v) for v in bank["neuron_id"].dropna().astype(np.int64))


def build_manifest(root, layer, cell_type, ids, allow_unbanked=False):
    root = os.path.abspath(root)
    bank_path, bank = bank_ids(root, layer)
    rows, problems = [], []
    for cid in ids:
        # POSIX separators on purpose: the manifest is data that travels
        # laptop -> cluster; run_p1_export resolves it against --root
        ncsv = "neurons/neuron_%d.csv" % cid
        scsv = "synapses/neuron_%d_synapses.csv" % cid
        if not os.path.isfile(os.path.join(root, ncsv)):
            problems.append("%d: no skeleton at %s" % (cid, ncsv))
            continue
        in_bank = cid in bank
        if not in_bank and not allow_unbanked:
            problems.append("%d: not in %s (use --allow-unbanked to accept)"
                            % (cid, os.path.basename(bank_path)))
            continue
        rows.append({"cell_id": cid, "layer": layer, "cell_type": cell_type,
                     "neuron_csv": ncsv,
                     "alignment_metadata": "alignment/alignment_metadata_%s.csv" % layer,
                     "synapse_csv": scsv if os.path.isfile(os.path.join(root, scsv)) else "",
                     "layer_source": "bank" if in_bank else "manifest"})
    if problems:
        raise SystemExit("manifest refused, %d problem(s):\n  %s"
                         % (len(problems), "\n  ".join(problems)))
    return pd.DataFrame(rows, columns=["cell_id", "layer", "cell_type", "neuron_csv",
                                       "alignment_metadata", "synapse_csv", "layer_source"])


def main(argv=None):
    args = build_parser().parse_args(argv)
    ids = read_ids(args.ids, args.ids_file)
    m = build_manifest(args.root, args.layer, args.cell_type, ids, args.allow_unbanked)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    m.to_csv(args.out, index=False, lineterminator="\n")
    n_syn = int((m["synapse_csv"] != "").sum())
    print("%s: %d cells (%s %s), %d with a synapse file, bank %s -> %s"
          % (BUILDER_VERSION, len(m), args.layer, args.cell_type, n_syn,
             os.path.basename(m["alignment_metadata"].iloc[0]), args.out))
    if n_syn < len(m):
        print("  NOTE: %d cell(s) without a synapse file will be exported WITHOUT "
              "the redirect" % (len(m) - n_syn))
    print("  array width for p1_export.pbs: #PBS -J 0-%d" % (len(m) - 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
