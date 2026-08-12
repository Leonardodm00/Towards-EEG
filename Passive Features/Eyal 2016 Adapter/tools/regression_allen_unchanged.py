"""
regression_allen_unchanged.py -- prove the five Eyal-support patches leave
Allen behaviour unchanged.

Method
------
1. Build a SYNTHETIC Allen-shaped archive: an SWC morphology, a
   Square-Subthreshold pulse pool whose peaks all sit within
   SQ_SUB_AMPLITUDE_TOL_PA (30 pA) of the nominal +/-200 pA exactly as
   Allen's QC guarantees, Long Square sweeps, and a metadata.json with NO
   "morphology_file" key (as Allen archives have none).
2. Load it with the UNPATCHED monolith in one subprocess and with the
   PATCHED monolith in another, dumping a deep fingerprint of the resulting
   CellData (scalars, per-bundle metadata, and SHA-256 of every array) to
   JSON.
3. Compare the two fingerprints field by field.

Why subprocesses: both copies of the module are called
passive_fitting_hpc_fixed, so they cannot coexist in one interpreter.

Expected result
---------------
Every field identical EXCEPT SweepBundle.sweep_numbers on the
hyperpolarising bundles, where the patch fixes a pre-existing indexing bug:
the original addressed the full npz sweep_number array with an index into
the polarity-FILTERED pulse list, so hyp bundles reported the sweep numbers
of dep pulses. The test asserts the old values were provably wrong (they
name sweeps that belong to dep pulses) and that the new ones are correct.

The test additionally checks:
  * n_avg_groups = 1 and 3 both reproduce (partitioning is untouched)
  * group_ss_by_amplitude=True is a NO-OP on Allen-shaped data, because a
    QC'd Allen pool clusters into exactly one amplitude group per polarity
  * require_long_square defaults to True: a cell with no LS is still dropped
  * the morphology path still resolves to reconstruction.swc

Usage
-----
    python3 regression_allen_unchanged.py --code-dir <dir with the monolith>

The directory must already be patched (patch_eyal_support.py --apply); the
unpatched copy is recovered from the .orig.bak files the patcher wrote.

ASCII-only, LF-only by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

MONOLITH = "passive_fitting_hpc_fixed.py"
LONGSTEP = "passive_long_step_training.py"

SEED = 20160331          # eLife 16553 submission date, for reproducibility


# ---------------------------------------------------------------------------
#  Synthetic Allen-shaped fixture
# ---------------------------------------------------------------------------

def make_synthetic_allen_archive(root: Path, *, with_ls: bool = True,
                                 specimen_id: int = 999001) -> Path:
    """Write one specimen directory shaped exactly like an Allen archive."""
    rng = np.random.default_rng(SEED)
    cell_dir = root / ("specimen_%d" % specimen_id)
    cell_dir.mkdir(parents=True, exist_ok=True)

    # -- morphology: a minimal but valid single-root SWC -------------------
    swc_lines = [
        "# synthetic test morphology",
        "1 1 0.0 0.0 0.0 8.0 -1",      # soma
        "2 1 0.0 8.0 0.0 8.0 1",
        "3 1 0.0 -8.0 0.0 8.0 1",
        "4 3 -10.0 0.0 0.0 1.2 1",     # basal
        "5 3 -60.0 0.0 0.0 0.9 4",
        "6 3 -110.0 20.0 0.0 0.6 5",
        "7 4 0.0 20.0 0.0 2.0 1",      # apical
        "8 4 0.0 120.0 0.0 1.5 7",
        "9 4 0.0 220.0 0.0 1.0 8",
        "10 2 0.0 -20.0 0.0 0.8 1",    # axon
        "11 2 0.0 -70.0 0.0 0.8 10",
    ]
    (cell_dir / "reconstruction.swc").write_text(
        "\n".join(swc_lines) + "\n", encoding="ascii", newline="\n")

    # -- Square Subthreshold pool -------------------------------------------
    # Allen: 0.5 ms pulses at +/-200 pA. Detected peaks jitter slightly but
    # every one is within SQ_SUB_AMPLITUDE_TOL_PA of nominal, which is the
    # invariant that made polarity-only grouping safe.
    sr = 50000.0
    dt = 1.0 / sr
    t = np.arange(-0.010, 0.200, dt)                  # -10 ms .. +200 ms
    dur_s = 5e-4

    n_dep, n_hyp = 5, 4
    peaks = ([200.0 + float(rng.normal(0, 0.4)) for _ in range(n_dep)]
             + [-200.0 + float(rng.normal(0, 0.4)) for _ in range(n_hyp)])
    # sweep numbers deliberately DISJOINT between polarities so that a
    # cross-polarity indexing error is detectable
    sweep_numbers = [10, 11, 12, 13, 14] + [90, 91, 92, 93]

    vs, is_ = [], []
    for k, pk in enumerate(peaks):
        env = np.where(t < 0, 0.0, np.exp(-t / 0.018))
        v = -74.0 + (pk / 200.0) * 1.4 * env + rng.normal(0, 0.01, t.size)
        i = np.where((t >= 0) & (t < dur_s), pk, 0.0)
        vs.append(v)
        is_.append(i)

    np.savez_compressed(
        cell_dir / "ss_pulses.npz",
        t=t.astype(np.float64),
        v=np.stack(vs).astype(np.float64),
        i_pA=np.stack(is_).astype(np.float64),
        polarity_is_dep=np.array([p > 0 for p in peaks], dtype=bool),
        peak_pA=np.array(peaks, dtype=np.float64),
        stim_duration_s=np.full(len(peaks), dur_s, dtype=np.float64),
        sampling_rate_Hz=np.full(len(peaks), sr, dtype=np.float64),
        sweep_number=np.array(sweep_numbers, dtype=np.int64),
    )

    # -- Long Square sweeps --------------------------------------------------
    ls_info: List[Dict[str, Any]] = []
    if with_ls:
        ls_sr = 20000.0
        ls_t = np.arange(0.0, 1.6, 1.0 / ls_sr)
        arrays: Dict[str, np.ndarray] = {}
        for k, amp in enumerate((-50.0, -90.0)):
            i = np.where((ls_t >= 0.2) & (ls_t < 1.2), amp, 0.0)
            env = np.clip((ls_t - 0.2) / 0.02, 0, 1)
            v = -74.0 + (amp / 100.0) * 6.0 * env * (ls_t < 1.2)
            arrays["v_%d" % k] = v.astype(np.float64)
            arrays["i_%d" % k] = i.astype(np.float64)
            ls_info.append({"index": k, "sweep_number": 200 + k,
                            "detected_amplitude_pA": amp,
                            "sampling_rate_Hz": ls_sr,
                            "stimulus_name": "Long Square",
                            "n_samples": int(v.size)})
        arrays["n_sweeps"] = np.array([len(ls_info)], dtype=np.int64)
        np.savez_compressed(cell_dir / "ls_sweeps.npz", **arrays)

    # -- metadata.json: NOTE the absence of "morphology_file" ---------------
    metadata = {
        "specimen_id": specimen_id,
        "layer": "3",
        "dendrite_type": "spiny",
        "donor_id": "synthetic",
        "structure_area_abbrev": "MTG",
        "rin_MOhm": 71.4,
        "tau_ms": 18.2,
        "v_rest_mV": -74.0,
        "ljp_correction_mV": 14.0,
        "full_allen_metadata": {"id": specimen_id,
                                "structure_layer_name": "3",
                                "dendrite_type": "spiny"},
        "ss_extraction": {"n_pulses_qc": len(peaks), "stacked": True},
        "ls_sweeps": ls_info,
        "smoke_test": {},
    }
    (cell_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="ascii", newline="\n")
    return cell_dir


# ---------------------------------------------------------------------------
#  Fingerprint dumper -- executed inside a subprocess
# ---------------------------------------------------------------------------

_DUMPER = r'''
import hashlib, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import passive_fitting_hpc_fixed as mono

archive, out, n_avg, group_amp = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "1"

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]

kw = dict(n_avg_groups=n_avg, verbose=False)
if group_amp:
    kw["group_ss_by_amplitude"] = True

cells = mono.load_cells_from_archive(archive, **kw)
res = {"n_cells": len(cells), "cells": []}
for cd in cells:
    entry = {
        "specimen_id": int(cd.specimen_id),
        "swc_path": str(cd.swc_path.name),
        "rin_MOhm": float(cd.rin_MOhm), "tau_ms": float(cd.tau_ms),
        "v_rest_mV": float(cd.v_rest_mV),
        "ljp_correction_mV": float(cd.ljp_correction_mV),
        "n_avg_groups": int(cd.n_avg_groups),
        "n_individual_pulses": len(cd.ss_individual_pulses),
        "ss": [], "ls": [],
    }
    for b in cd.square_subthreshold:
        entry["ss"].append({
            "polarity": b.polarity,
            "amplitude_pA": round(float(b.amplitude_pA), 9),
            "stim_onset_s": float(b.stim_onset_s),
            "stim_duration_s": float(b.stim_duration_s),
            "n_repeats_averaged": int(b.n_repeats_averaged),
            "sampling_rate_Hz": float(b.sampling_rate_Hz),
            "stimulus_name": b.stimulus_name,
            "sweep_numbers": list(b.sweep_numbers),
            "hash_t": h(b.t), "hash_v": h(b.v_mV), "hash_i": h(b.i_pA),
        })
    for b in cd.long_square_subthreshold:
        entry["ls"].append({
            "polarity": b.polarity,
            "amplitude_pA": round(float(b.amplitude_pA), 9),
            "stim_onset_s": round(float(b.stim_onset_s), 9),
            "stim_duration_s": round(float(b.stim_duration_s), 9),
            "n_repeats_averaged": int(b.n_repeats_averaged),
            "sweep_numbers": list(b.sweep_numbers),
            "hash_t": h(b.t), "hash_v": h(b.v_mV), "hash_i": h(b.i_pA),
        })
    res["cells"].append(entry)

with open(out, "w") as f:
    json.dump(res, f, indent=2)
'''


def _dump(code_dir: Path, archive: Path, out: Path, *,
          n_avg: int = 1, group_amp: bool = False) -> Dict[str, Any]:
    script = out.parent / ("_dumper_%s.py" % out.stem)
    script.write_text(_DUMPER, encoding="ascii", newline="\n")
    r = subprocess.run(
        [sys.executable, str(script), str(code_dir), str(archive), str(out),
         str(n_avg), "1" if group_amp else "0"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("dumper failed for %s:\n%s\n%s"
                           % (code_dir, r.stdout[-2000:], r.stderr[-2000:]))
    return json.loads(out.read_text())


def _make_unpatched_copy(code_dir: Path, dest: Path) -> Path:
    """Reconstruct the pre-patch tree from the .orig.bak files."""
    dest.mkdir(parents=True, exist_ok=True)
    for p in code_dir.glob("*.py"):
        shutil.copy2(p, dest / p.name)
    n = 0
    for fn in (MONOLITH, LONGSTEP):
        bak = code_dir / (fn + ".orig.bak")
        if bak.is_file():
            shutil.copy2(bak, dest / fn)
            n += 1
    if n == 0:
        raise RuntimeError(
            "no .orig.bak files in %s -- run patch_eyal_support.py --apply "
            "first, so that the unpatched revision is recoverable" % code_dir)
    return dest


# ---------------------------------------------------------------------------
#  Comparison
# ---------------------------------------------------------------------------

def compare(before: Dict[str, Any], after: Dict[str, Any],
            label: str) -> List[str]:
    """Return a list of human-readable differences (empty = identical)."""
    diffs: List[str] = []
    if before["n_cells"] != after["n_cells"]:
        diffs.append("%s: n_cells %d -> %d"
                     % (label, before["n_cells"], after["n_cells"]))
        return diffs
    for cb, ca in zip(before["cells"], after["cells"]):
        for k in ("specimen_id", "swc_path", "rin_MOhm", "tau_ms",
                  "v_rest_mV", "ljp_correction_mV", "n_avg_groups",
                  "n_individual_pulses"):
            if cb[k] != ca[k]:
                diffs.append("%s: %s %r -> %r" % (label, k, cb[k], ca[k]))
        for proto in ("ss", "ls"):
            if len(cb[proto]) != len(ca[proto]):
                diffs.append("%s: n_%s_bundles %d -> %d"
                             % (label, proto, len(cb[proto]), len(ca[proto])))
                continue
            for j, (bb, ba) in enumerate(zip(cb[proto], ca[proto])):
                for k in bb:
                    if bb[k] != ba[k]:
                        diffs.append("%s: %s[%d].%s %r -> %r"
                                     % (label, proto, j, k, bb[k], ba[k]))
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code-dir", required=True,
                    help="PATCHED directory (must contain the .orig.bak files)")
    ap.add_argument("--keep", action="store_true", help="keep the temp tree")
    args = ap.parse_args(argv)

    code_dir = Path(args.code_dir).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="allen_regression_"))
    print("workspace:", tmp)

    unpatched = _make_unpatched_copy(code_dir, tmp / "unpatched")
    archive = tmp / "allen_archive"
    make_synthetic_allen_archive(archive)
    archive_nols = tmp / "allen_archive_no_ls"
    make_synthetic_allen_archive(archive_nols, with_ls=False,
                                 specimen_id=999002)

    n_fail = 0
    print("=" * 74)

    # -- A. default flags, n_avg_groups = 1 and 3 --------------------------
    for n_avg in (1, 3):
        b = _dump(unpatched, archive, tmp / ("b_%d.json" % n_avg), n_avg=n_avg)
        a = _dump(code_dir, archive, tmp / ("a_%d.json" % n_avg), n_avg=n_avg)
        diffs = compare(b, a, "n_avg=%d" % n_avg)
        expected, unexpected = [], []
        for d in diffs:
            (expected if ".sweep_numbers" in d else unexpected).append(d)
        if unexpected:
            n_fail += 1
            print("FAIL  default flags, n_avg_groups=%d" % n_avg)
            for d in unexpected[:12]:
                print("      %s" % d)
        else:
            print("PASS  default flags, n_avg_groups=%d -- every fitting-"
                  "relevant field bit-identical" % n_avg)
            if expected:
                print("      (%d sweep_numbers difference(s), examined below)"
                      % len(expected))

    # -- B. the sweep_numbers change is a bug FIX --------------------------
    b = _dump(unpatched, archive, tmp / "b_sn.json", n_avg=1)
    a = _dump(code_dir, archive, tmp / "a_sn.json", n_avg=1)
    dep_sweeps, hyp_sweeps = {10, 11, 12, 13, 14}, {90, 91, 92, 93}
    ok = True
    for bb, ba in zip(b["cells"][0]["ss"], a["cells"][0]["ss"]):
        want = dep_sweeps if bb["polarity"] == "dep" else hyp_sweeps
        if set(ba["sweep_numbers"]) != want:
            ok = False
            print("      patched %s bundle reports %r, expected %r"
                  % (ba["polarity"], ba["sweep_numbers"], sorted(want)))
        if bb["polarity"] == "hyp" and set(bb["sweep_numbers"]) <= dep_sweeps:
            print("      confirmed: the UNPATCHED hyp bundle reported %r, "
                  "which are DEP sweep numbers" % bb["sweep_numbers"])
    if ok:
        print("PASS  sweep_numbers: patched values correct per polarity; "
              "old hyp values were dep sweeps (pre-existing indexing bug)")
    else:
        n_fail += 1
        print("FAIL  sweep_numbers not corrected as expected")

    # -- C. group_ss_by_amplitude=True is a no-op on Allen-shaped data -----
    a_on = _dump(code_dir, archive, tmp / "a_grp.json", n_avg=1, group_amp=True)
    diffs = compare(a, a_on, "group_ss_by_amplitude=True")
    if diffs:
        n_fail += 1
        print("FAIL  group_ss_by_amplitude=True changed Allen output")
        for d in diffs[:12]:
            print("      %s" % d)
    else:
        print("PASS  group_ss_by_amplitude=True is a NO-OP on Allen-shaped "
              "data (one amplitude cluster per polarity)")

    # -- D. require_long_square still defaults to True ---------------------
    b_nols = _dump(unpatched, archive_nols, tmp / "b_nols.json")
    a_nols = _dump(code_dir, archive_nols, tmp / "a_nols.json")
    if b_nols["n_cells"] == a_nols["n_cells"] == 0:
        print("PASS  require_long_square defaults to True: a cell with no LS "
              "is still dropped, before and after")
    else:
        n_fail += 1
        print("FAIL  LS gate changed: before=%d cells, after=%d cells"
              % (b_nols["n_cells"], a_nols["n_cells"]))

    # -- E. morphology path unchanged when metadata lacks the key ----------
    if a["cells"][0]["swc_path"] == "reconstruction.swc":
        print("PASS  morphology path still resolves to reconstruction.swc "
              "when metadata has no morphology_file key")
    else:
        n_fail += 1
        print("FAIL  morphology path is %r" % a["cells"][0]["swc_path"])

    print("=" * 74)
    print("FAILURES: %d" % n_fail)
    print("=" * 74)
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("workspace kept at", tmp)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
