#!/usr/bin/env python3
"""Smoke test for P0 partition unification, against the REAL Stage 1 chain.

Runs morphology_exporter.demote_shaft_continuations_three_vote -- the single
entry point every call site must route through -- with the real
shaft_continuation and continuation_inspect, on a synthetic cell built to
contain one of each decision the rule can make:

  KEEP     a genuine spine (thin neck, bulged head): fails the calibre vote.
  DEMOTE   a shaft continuation (same calibre, straight, monotone taper):
           all three votes agree it is shaft.
  RESCUE   a shaft-calibre, shaft-direction protrusion whose radius has a
           distal maximum: the two-observable rule calls it shaft, the taper
           vote holds it back. This is the vote that is new in 1.2.0.

Needs only numpy + pandas + the repo itself; no network, no scheduler. Run it
in the sandbox before shipping and on the cluster login node after pulling:

    cd h01_code && python3 smoke_test_p0_partition.py

Expected: "9 checks passed, 0 failed" and "ALL GREEN" on the last two lines.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Prefer the stage1 symlink farm (what the runner uses); fall back to the
# canonical source directories so this runs before stage1_link.sh has.
_stage1 = os.path.join(HERE, "stage1")
if os.path.isfile(os.path.join(_stage1, "morphology_exporter.py")):
    _paths = [HERE, _stage1]
else:
    _paths = [HERE, os.path.join(REPO, "towards_eeg", "structure"),
              os.path.join(REPO, "Stage 1")]
for _p in _paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import spine_density as sd                            # noqa: E402
import morphology_exporter as mx                      # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILURES.append(name)
    return ok


def synthetic_cell():
    """Shaft along +x with three labelled protrusions: keep / demote / rescue.

    Radii and angles are chosen against the rule's own thresholds
    (RHO_SHAFT_MIN=0.50, COS_SHAFT_MIN=0.70, MIN_LEN_NM=150, BULGE_MIN=1.25,
    PEAK_FRAC_MIN=0.30), with margin on every inequality.
    """
    rows = []
    n_shaft, step, r_shaft = 21, 250.0, 300.0
    for i in range(n_shaft):                       # ids 0..20, root id 0
        rows.append(dict(id=i, p=i - 1 if i else -1, x=step * i, y=0.0, z=0.0,
                         r=r_shaft, annotated_type="dendrite",
                         compartment_class=""))
    nid = 1000

    def add(px, py, pz, parent, radii, direction, tag):
        first, par = nid_box[0], parent
        for k, r in enumerate(radii, start=1):
            rows.append(dict(id=nid_box[0], p=par,
                             x=px + direction[0] * step * k,
                             y=py + direction[1] * step * k,
                             z=pz + direction[2] * step * k,
                             r=r, annotated_type="spine",
                             compartment_class=""))
            par = nid_box[0]
            nid_box[0] += 1
        return first

    nid_box = [nid]
    # KEEP: off node 5, straight +y, thin neck then head. rho=70/300=0.23.
    keep_root = add(5 * step, 0.0, 0.0, 5,
                    [70, 70, 70, 70, 70, 250, 250, 250], (0.0, 1.0, 0.0), "keep")
    # DEMOTE: continues +x off the LAST shaft node, same calibre, monotone
    # taper. rho=290/300=0.97, cos=1.0, bulge=1.0 (peak at s=0).
    demote_root = add((n_shaft - 1) * step, 0.0, 0.0, n_shaft - 1,
                      [290, 280, 270, 260, 250, 235, 220, 200],
                      (1.0, 0.0, 0.0), "demote")
    # RESCUE: off node 12, mostly +x (cos ~ 0.90), shaft calibre at the root
    # (rho=280/300=0.93), radius dips to 150 then rises to a DISTAL maximum
    # of 330 (bulge=330/150=2.2, s_peak_frac=1.0).
    d = np.array([0.9, 0.436, 0.0])
    d /= np.linalg.norm(d)
    rescue_root = add(12 * step, 0.0, 0.0, 12,
                      [280, 240, 190, 150, 170, 230, 290, 330],
                      tuple(d), "rescue")
    return pd.DataFrame(rows), keep_root, demote_root, rescue_root


def labels_of(df, roots):
    return {r: df.loc[df["id"] == r, "annotated_type"].item() for r in roots}


def main():
    print("smoke_test_p0_partition  (exporter %s, scorer %s)"
          % (mx.MODULE_VERSION,
             getattr(mx.shc, "MODULE_VERSION", "?") if mx.shc else "MISSING"))
    check("real modules on path (not stubs)",
          "stub" not in mx.MODULE_VERSION
          and "stub" not in sd.MODULE_VERSION,
          "%s / %s" % (mx.MODULE_VERSION, sd.MODULE_VERSION))

    df, keep_root, demote_root, rescue_root = synthetic_cell()

    # ---- the default rule: all three votes -------------------------------
    out, rep = mx.demote_shaft_continuations_three_vote(df)
    check("report: applied, radius trusted",
          rep.get("applied") is True and rep.get("use_radius") is True,
          "use_radius=%s method=%s" % (rep.get("use_radius"), rep.get("method")))
    lab = labels_of(out, (keep_root, demote_root, rescue_root))
    check("KEEP: genuine spine stays a spine",
          lab[keep_root] == "spine", str(lab[keep_root]))
    check("DEMOTE: shaft continuation relabelled to its base's label",
          lab[demote_root] == "dendrite", str(lab[demote_root]))
    check("RESCUE: distal-maximum protrusion held back by the taper vote",
          lab[rescue_root] == "spine",
          "demoted_roots=%s rescued=%s" % (rep.get("demoted_roots"),
                                           rep.get("n_rescued_by_taper")))
    check("counts: exactly one demotion, at least one taper rescue",
          rep.get("n_demoted") == 1 and rep.get("n_rescued_by_taper", 0) >= 1,
          "n_demoted=%s n_rescued=%s" % (rep.get("n_demoted"),
                                         rep.get("n_rescued_by_taper")))
    check("geometry untouched: ids, parents, coordinates, radii identical",
          all(np.array_equal(np.asarray(df[c]), np.asarray(out[c]))
              for c in ("id", "p", "x", "y", "z", "r")))

    # ---- require_taper=False reduces to the two-observable rule ----------
    out2, rep2 = mx.demote_shaft_continuations_three_vote(
        df, require_taper=False)
    lab2 = labels_of(out2, (keep_root, demote_root, rescue_root))
    check("two-observable fallback demotes the rescue case too",
          lab2[rescue_root] == "dendrite" and lab2[demote_root] == "dendrite"
          and lab2[keep_root] == "spine",
          "n_demoted=%s" % rep2.get("n_demoted"))

    # ---- thresholds surface in the report (what the fingerprint reads) ---
    check("report carries the thresholds the fingerprint records",
          all(k in rep for k in ("rho_shaft_min", "cos_shaft_min",
                                 "min_len_nm", "bulge_min", "require_taper")),
          str({k: rep.get(k) for k in ("rho_shaft_min", "cos_shaft_min",
                                       "min_len_nm", "bulge_min")}))

    n = 9
    print("\n%d checks passed, %d failed" % (n - len(FAILURES), len(FAILURES)))
    print("ALL GREEN" if not FAILURES else "FAILURES: %s" % FAILURES)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
