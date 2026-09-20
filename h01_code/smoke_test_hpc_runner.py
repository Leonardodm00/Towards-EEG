#!/usr/bin/env python3
"""End-to-end test of run_spine_area_F.py and merge_spine_area_F.py.

  python3 smoke_test_hpc_runner.py        (quiet)
  python3 smoke_test_hpc_runner.py -v     (every check)

Builds a throwaway tree shaped like the cluster layout -- neurons/, stage1/,
out/, a g table -- with STUB Stage 1 modules and a STUB network reader, then
runs two shards and the merge for real. It proves the CLI, path resolution,
sharding, fingerprinting, ledger union and output writing work; it does NOT
validate the science (smoke_test_h01_spine_area_F.py does that) and the Stage
1 stubs are not the real modules.

Negative paths are tested too: a missing CSV, a bad --task, shards built with
different parameters, and a missing shard.

Section B (2026-09-20) tests the JOB SCRIPTS, which the Python checks above
cannot see: spine_area_F.pbs is parsed for its two guards (the --g-table flag
that points at H01_CODE, and the activation block that trusts the outcome of
`conda activate` rather than its exit status) and is then RUN with bash
against a fixture tree, with stub python3 / conda / curl on PATH, asserting on
the argv the stub interpreter received and on every refusal path. probe_net.pbs
gets the same treatment. Needs bash on PATH (the cluster and the sandbox both
have it); no network.

Pure ASCII, LF only.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h01_area_calibration as CAL
import smoke_test_h01_spine_area_F as T

VERBOSE = "-v" in sys.argv
CELL = 424242
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    if VERBOSE or not ok:
        print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


# Stubs for ONLY the four Stage 1 modules that are not in this bundle. sma_run,
# s0_ingest and shaft_continuation are the REAL ones, so this test exercises
# the genuine glue -- entry-point resolution, the labeller tempdir adapter, the
# frame-integrity checks -- not a reimplementation of it.
STUBS = {}

STUBS["spine_density"] = '''"""Stub spine_density for the HPC runner smoke test. NOT the real module."""
import sys
sys.path.insert(0, "__CODE_DIR__")
import smoke_test_h01_spine_area_F as T

MODULE_VERSION = "stub spine_density (smoke test)"
_sd = T.FakeSD()
SHAFT_REGEX = _sd.SHAFT_REGEX
SPINE_LABELS = tuple(_sd.SPINE_LABELS)
DEFAULT_RADIUS_NM = _sd.DEFAULT_RADIUS_NM
CAP_H_UM_DEFAULT = _sd.CAP_H_UM_DEFAULT
_prepare_nodes = _sd._prepare_nodes
_frustum_lateral_area = _sd._frustum_lateral_area
_segment_length_um = _sd._segment_length_um
build_phi = _sd.build_phi
cell_f_beyond_cutoff = _sd.cell_f_beyond_cutoff
'''

STUBS["spine_geometry"] = '''"""Stub spine_geometry for the HPC runner smoke test."""
MODULE_VERSION = "stub spine_geometry (smoke test)"
HEAD_LABELS = ("head",)
NECK_LABELS = ("neck",)
'''

STUBS["morphology_exporter"] = '''"""Stub morphology_exporter for the HPC runner smoke test.

demote_shaft_continuations_three_vote mirrors the real entry point's contract
(same-frame-out, (df, report) return, report keys the fingerprint reads) and
demotes nothing: the phantom's spines are genuinely spine-like, which is also
what the real three-vote rule would decide.
"""
MODULE_VERSION = "stub morphology_exporter (smoke test)"
SPINE_LENGTH_THRESHOLD_NM = 4000.0


def demote_shaft_continuations_three_vote(df, **kw):
    report = {"applied": True, "module_version": MODULE_VERSION,
              "scorer_version": "stub", "inspect_version": "stub",
              "method": "stub", "use_radius": True,
              "rho_shaft_min": 0.50, "cos_shaft_min": 0.70,
              "require_taper": bool(kw.get("require_taper", True)),
              "min_len_nm": 150.0, "bulge_min": 1.25,
              "n_spine_roots": 0, "n_shaft_like_rho_cos": 0,
              "n_demoted": 0, "n_nodes_demoted": 0,
              "n_rescued_by_taper": 0, "n_undecidable": 0,
              "demoted_roots": []}
    return df, report
'''

# The real sma_run.label_spines_project writes the node table to a tempdir and
# calls this by filename, then checks the returned frame is the input relabelled
# (same rows, same order, geometry untouched). The phantom CSV already carries
# its labels, so the stub relabels nothing and that check is what is exercised.
STUBS["spine_labeller"] = '''"""Stub spine_labeller for the HPC runner smoke test."""
import os
import pandas as pd

MODULE_VERSION = "stub spine_labeller (smoke test)"
SOURCE_FILE = "<stub>"
SOURCE_LINES = []
SOURCE_SHA256 = None


def label_dendritic_spines_robust(nids, input_dir=None, output_dir=None,
                                  spine_length_threshold_nm=5000.0, **kw):
    out = {}
    for nid in nids:
        df = pd.read_csv(os.path.join(input_dir, "neuron_%s.csv" % nid))
        out[nid] = df
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    return out
'''

N_SPINES = 4
SPACING = 3000.0


def multi_nodes():
    """A shaft along +x with N_SPINES identical spines on it, so the sharding,
    union and merge logic is exercised with more spines than shards."""
    import pandas as pd
    o = T.ORIGIN
    rows, n_shaft = [], int(SPACING * N_SPINES / 250.0) + 1
    for i in range(n_shaft):
        rows.append(dict(id=i, p=i - 1 if i else -1, x=o[0] + 250.0 * i,
                         y=o[1], z=o[2], r=300.0, annotated_type="dendrite"))
    nid = 1000
    for k in range(N_SPINES):
        xk = 1500.0 + SPACING * k
        par = int(round(xk / 250.0))
        for yy in (380., 480., 580., 680., 780., 900., 1050., 1200.):
            rows.append(dict(id=nid, p=par, x=o[0] + xk, y=o[1] + yy, z=o[2],
                             r=70.0 if yy < 850 else 250.0,
                             annotated_type="spine"))
            par = nid
            nid += 1
    return pd.DataFrame(rows)


def multi_stub_factory():
    """Fabricates the segmentation for that phantom instead of downloading."""
    def factory(cloudpath, mip=0, **kw):
        info = {"cloudpath": cloudpath, "mip": 0,
                "resolution_nm": list(T.RES), "dtype": "uint64",
                "bounds_vox": [[0, 0, 0], [10 ** 7] * 3], "available_mips": [0]}

        def reader(lo, hi):
            lo, hi = np.asarray(lo), np.asarray(hi)
            X, Y, Z = T.grid(tuple(int(v) for v in hi - lo), lo)
            x, y, z = X - T.ORIGIN[0], Y - T.ORIGIN[1], Z - T.ORIGIN[2]
            m = (y ** 2 + z ** 2 <= 300.0 ** 2) & (x >= 0) & (x <= SPACING * N_SPINES)
            for k in range(N_SPINES):
                xk = 1500.0 + SPACING * k
                m |= ((x - xk) ** 2 + z ** 2 <= 70.0 ** 2) & (y >= 0) & (y <= 850.0)
                m |= (x - xk) ** 2 + (y - 1050.0) ** 2 + z ** 2 <= 250.0 ** 2
            a = np.zeros(X.shape, dtype=np.uint64)
            a[m] = CELL
            return a
        return reader, info
    return factory


def build_tree(tmp, code_dir):
    root = os.path.join(tmp, "campaign")
    for d in ("neurons", "stage1", "out"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for name, src in STUBS.items():
        with open(os.path.join(root, "stage1", name + ".py"), "w") as fh:
            fh.write(src.replace("__CODE_DIR__", code_dir))
    multi_nodes().to_csv(os.path.join(root, "neurons", "neuron_%d.csv" % CELL),
                         index=False)
    th, ph = np.arange(46) * 2.0, np.arange(23) * 2.0
    CAL.save_table(os.path.join(root, "g_table_cyl_2deg.npz"),
                   {"theta_deg": th, "phi_deg": ph,
                    "g": 1.0 + 0.04 * np.ones((46, 23)),
                    "meta": {"resolution_nm": [8.0, 8.0, 33.0]}})
    return root


def base_argv(root, task, ntasks, *extra):
    return (["--root", root, "--cell", str(CELL), "--task", str(task),
             "--ntasks", str(ntasks), "--stage1-dir", os.path.join(root, "stage1")]
            + list(extra))



# ---------------------------------------------------------------------------
# Section B: the job scripts. Two defects shipped in spine_area_F.pbs that no
# Python suite could catch -- the runner's --g-table default resolves into
# H01_ROOT while the table is tracked in H01_CODE (every array task died at
# resolve() after the queue wait), and the activation block obeyed the exit
# status of `conda activate`, which is 1 on this cluster while activation is
# real (binutils activate.d hook). Both live in bash, so both are tested in
# bash: text checks for the guards, then real runs against a fixture.
# ---------------------------------------------------------------------------
PBS = "spine_area_F.pbs"
PROBE = "probe_net.pbs"
JOB_CELL = 424242
_ARGV_N = [0]

STUB_PY = """#!/bin/bash
# stub python3 for the job-script test: records argv, runs nothing
printf '%s\\n' "$@" >> "$STUB_ARGV_OUT"
exit 0
"""

# The hook evaluates to nothing; `activate` prints an INFO line and returns 1,
# which is exactly what the binutils hook does on the cluster while the env
# HAS been activated. Whether it "took effect" is decided by where python3
# resolves, which the test controls through PATH.
STUB_CONDA = """#!/bin/bash
case "$1" in
    shell) exit 0 ;;
    activate) echo "INFO: stub activate.d hook"; exit 1 ;;
esac
exit 0
"""

STUB_CURL = """#!/bin/bash
printf '000'
exit 0
"""


def _write_exec(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o755)


def job_fixture(tmp, with_table=True):
    """A tree shaped like the cluster: H01_CODE with the runner file and the g
    table, H01_ROOT with neurons/, bin/ with the stubs, envs/spine_env/bin and
    home/.conda/envs/spine_env/bin each with a python3 stub."""
    fx = os.path.join(tmp, "jobfix")
    shutil.rmtree(fx, ignore_errors=True)
    code = os.path.join(fx, "h01_code")
    root = os.path.join(fx, "h01")
    pydirs = (os.path.join(fx, "bin"),
              os.path.join(fx, "envs", "spine_env", "bin"),
              os.path.join(fx, "home", ".conda", "envs", "spine_env", "bin"))
    for d in (os.path.join(code, "stage1"), os.path.join(root, "neurons")) + pydirs:
        os.makedirs(d, exist_ok=True)
    open(os.path.join(code, "run_spine_area_F.py"), "w").close()
    if with_table:
        open(os.path.join(code, "g_table_cyl_2deg.npz"), "w").close()
    for d in pydirs:
        _write_exec(os.path.join(d, "python3"), STUB_PY)
    _write_exec(os.path.join(fx, "bin", "conda"), STUB_CONDA)
    _write_exec(os.path.join(fx, "bin", "curl"), STUB_CURL)
    return fx, code, root


def run_job(script, fx, code, root, extra_env, env_python=False):
    """bash <script> with a scrubbed environment: no inherited CODE/ROOT/H01_*/
    PBS_*/CONDA*, HOME inside the fixture, the stub bin first on PATH and, when
    env_python, the envs/spine_env/bin stub ahead of it. Returns (rc, stdout,
    argv), argv being what the stub interpreter received ([] if never run)."""
    _ARGV_N[0] += 1
    argv_out = os.path.join(fx, "argv_%d.txt" % _ARGV_N[0])
    path = os.path.join(fx, "bin") + os.pathsep + os.environ.get("PATH", "")
    if env_python:
        path = os.path.join(fx, "envs", "spine_env", "bin") + os.pathsep + path
    drop = ("CODE", "ROOT", "CELL", "SUBSET", "NTASKS", "DRY_RUN", "SKIP_CONDA",
            "ENV_NAME", "MIN_SPINE_VALUE", "NO_MEASURE_BASE", "STUB_ARGV_OUT")
    env = {k: v for k, v in os.environ.items()
           if not (k in drop or k.startswith("H01_") or k.startswith("PBS_")
                   or k.startswith("CONDA"))}
    env.update({"PATH": path, "HOME": os.path.join(fx, "home"),
                "H01_CODE": code, "H01_ROOT": root, "STUB_ARGV_OUT": argv_out})
    env.update(extra_env)
    p = subprocess.run(["bash", script], cwd=fx, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=120)
    argv = open(argv_out).read().splitlines() if os.path.isfile(argv_out) else []
    return p.returncode, p.stdout, argv


def pbs_python_invocation(text):
    """The `python3 run_spine_area_F.py ...` command with its backslash
    continuations joined into one line, or '' if absent."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("python3 run_spine_area_F.py"):
            parts = []
            for j in range(i, len(lines)):
                parts.append(lines[j].rstrip().rstrip("\\").strip())
                if not lines[j].rstrip().endswith("\\"):
                    break
            return " ".join(parts)
    return ""


def _first_line(lines, needle, start=0):
    for i in range(start, len(lines)):
        if lines[i].strip() == needle:
            return i
    return -1


def _argval(argv, flag):
    if flag in argv and argv.index(flag) + 1 < len(argv):
        return argv[argv.index(flag) + 1]
    return None


def job_script_checks(code_dir, tmp):
    pbs_path = os.path.join(code_dir, PBS)
    probe_path = os.path.join(code_dir, PROBE)
    text = open(pbs_path).read()
    lines = text.splitlines()

    # ---- B1 text: the g-table flag and its preflight
    inv = pbs_python_invocation(text)
    check("B1 .pbs passes --g-table under H01_CODE to the runner",
          '--g-table "$H01_CODE/g_table_cyl_2deg.npz"' in inv, inv[:90])
    check("B1' .pbs checks the g table exists before starting python",
          'if [ ! -f "$H01_CODE/g_table_cyl_2deg.npz" ]' in text)

    # ---- B2 text: activation trusts the OUTCOME, guards in the right order
    i_hook = _first_line(lines, 'eval "$(conda shell.bash hook)"')
    i_act = _first_line(lines, 'conda activate "$ENV_NAME"', i_hook + 1)
    i_pe = _first_line(lines, "set +e")
    i_pu = _first_line(lines, "set +u")
    i_me = _first_line(lines, "set -e", i_act + 1)
    i_mu = _first_line(lines, "set -u", i_act + 1)
    check("B2 activation: set +u and set +e BEFORE the hook, -e/-u restored AFTER activate",
          0 <= i_pu < i_hook and 0 <= i_pe < i_hook and i_hook < i_act
          and i_act < i_me and i_act < i_mu,
          "+u %d +e %d hook %d act %d -e %d -u %d" % (i_pu, i_pe, i_hook, i_act, i_me, i_mu))
    check("B2' activation verified by where python3 resolves, not by exit status",
          '*"/envs/$ENV_NAME/"*)' in text and "did not take effect" in text)
    check("B2'' path knobs are H01_ROOT / H01_CODE and stale CODE/ROOT are reported",
          '"${H01_CODE:-' in text and '"${H01_ROOT:-' in text
          and "for _stale in ROOT CODE" in text)

    # ---- B3 text: probe_net.pbs no longer points at the retired repo
    ptext = open(probe_path).read()
    plines = [ln for ln in ptext.splitlines() if not ln.lstrip().startswith("#")]
    check("B3 probe_net.pbs has no live reference to TEEG/Spines",
          not any("TEEG/Spines" in ln for ln in plines))
    check("B3' probe_net.pbs uses the H01_CODE knob with the Towards-EEG default",
          'H01_CODE="${H01_CODE:-/davinci-1/home/ldellamea/TEEG/Towards-EEG/h01_code}"' in ptext
          and "$CODE" not in ptext)

    # ---- B4 syntax and bytes of every shell script shipped
    for name in (PBS, PROBE, "run_smoke_tests.sh", "stage1_link.sh"):
        fp = os.path.join(code_dir, name)
        rc = subprocess.run(["bash", "-n", fp], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
        raw = open(fp, "rb").read()
        check("B4 %s: bash -n passes, LF only, pure ASCII" % name,
              rc.returncode == 0 and b"\r" not in raw and all(b < 128 for b in raw),
              rc.stdout.strip()[:80])

    # ---- B5 run spine_area_F.pbs: the positive path, no conda
    fx, code, root = job_fixture(tmp)
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            {"SKIP_CONDA": "1", "CELL": str(JOB_CELL),
                             "PBS_ARRAY_INDEX": "3", "SUBSET": "7", "DRY_RUN": "1"})
    check("B5 .pbs runs to completion and invokes the runner",
          rc == 0 and argv[:1] == ["run_spine_area_F.py"],
          "rc=%d argv=%s\n%s" % (rc, argv[:2], out[-400:]) if rc else "")
    check("B5' the runner receives --g-table = H01_CODE/g_table_cyl_2deg.npz",
          _argval(argv, "--g-table") == os.path.join(code, "g_table_cyl_2deg.npz"),
          str(_argval(argv, "--g-table")))
    check("B5'' --root, --stage1-dir, --cell, --task, --ntasks, --subset, --dry-run as set",
          _argval(argv, "--root") == root
          and _argval(argv, "--stage1-dir") == os.path.join(code, "stage1")
          and _argval(argv, "--cell") == str(JOB_CELL)
          and _argval(argv, "--task") == "3" and _argval(argv, "--ntasks") == "40"
          and _argval(argv, "--subset") == "7" and "--dry-run" in argv
          and "task 3/40" in out, str(argv))
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            {"SKIP_CONDA": "1", "CELL": str(JOB_CELL), "PBS_ARRAYID": "5"})
    check("B5''' Torque fallback: PBS_ARRAYID sets --task when PBS_ARRAY_INDEX is unset",
          rc == 0 and _argval(argv, "--task") == "5" and "--subset" not in argv
          and "--dry-run" not in argv)

    # ---- B6 the activation block, positive and negative
    # Positive: the stub `conda activate` RETURNS 1 (as binutils does) but
    # python3 resolves under envs/spine_env/bin -> the job must proceed.
    rc, out, argv = run_job(pbs_path, fx, code, root, {"CELL": str(JOB_CELL)},
                            env_python=True)
    check("B6 conda activate returning 1 with python3 inside the env: job PROCEEDS",
          rc == 0 and "INFO: stub activate.d hook" in out
          and argv[:1] == ["run_spine_area_F.py"],
          "rc=%d\n%s" % (rc, out[-400:]) if rc else "")
    # Negative: same hook, but python3 does NOT resolve inside the env.
    rc, out, argv = run_job(pbs_path, fx, code, root, {"CELL": str(JOB_CELL)},
                            env_python=False)
    check("B6' activation that did not take effect: job REFUSES before python",
          rc != 0 and "did not take effect" in out and argv == [],
          "rc=%d" % rc)

    # ---- B7 the other refusals, each before the interpreter starts
    rc, out, argv = run_job(pbs_path, fx, code, root, {"SKIP_CONDA": "1"})
    check("B7 unset CELL: refused, python never invoked",
          rc != 0 and "CELL" in out and argv == [], "rc=%d" % rc)
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            {"SKIP_CONDA": "1", "CELL": str(JOB_CELL),
                             "CODE": "/nowhere/other_project", "ROOT": "/nowhere/else"})
    check("B7' stale CODE / ROOT in the environment: reported, ignored, run proceeds",
          rc == 0 and "NOTE: CODE is set" in out and "NOTE: ROOT is set" in out
          and _argval(argv, "--root") == root
          and _argval(argv, "--g-table") == os.path.join(code, "g_table_cyl_2deg.npz"))
    rc, out, argv = run_job(pbs_path, fx, code, os.path.join(fx, "not_a_root"),
                            {"SKIP_CONDA": "1", "CELL": str(JOB_CELL)})
    check("B7'' H01_ROOT without neurons/: refused",
          rc != 0 and "neurons/" in out and argv == [])
    fx2, code2, root2 = job_fixture(tmp, with_table=False)
    rc, out, argv = run_job(pbs_path, fx2, code2, root2,
                            {"SKIP_CONDA": "1", "CELL": str(JOB_CELL)})
    check("B7''' missing g table: refused with the path named, python never invoked",
          rc != 0 and "calibration table missing" in out
          and os.path.join(code2, "g_table_cyl_2deg.npz") in out and argv == [],
          "rc=%d" % rc)

    # ---- B8 probe_net.pbs runs offline against the fixture (stub curl, stub
    # interpreter under HOME/.conda) and refuses a wrong H01_CODE
    fx, code, root = job_fixture(tmp)
    rc, out, argv = run_job(probe_path, fx, code, root, {})
    check("B8 probe_net.pbs runs to 'PROBE done' from H01_CODE",
          rc == 0 and "PROBE done" in out and "PROBE code dir" in out,
          "rc=%d\n%s" % (rc, out[-300:]) if rc else "")
    rc, out, argv = run_job(probe_path, fx, os.path.join(fx, "nope"), root, {})
    check("B8' probe_net.pbs refuses an H01_CODE that is not h01_code",
          rc != 0 and "PROBE FATAL" in out)


def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    import run_spine_area_F as R
    import merge_spine_area_F as M

    tmp = tempfile.mkdtemp()
    try:
        root = build_tree(tmp, code_dir)

        # ---- dry run first: no network, no ledger
        rc = R.main(base_argv(root, 0, 2, "--dry-run"))
        shards = os.path.join(root, "out", "cell%d_shards" % CELL)
        check("A1 dry run exits 0 and writes no ledger",
              rc == 0 and not os.path.isdir(shards))

        # ---- sharding tiles the list exactly once, contiguously
        args = R.resolve(R.build_parser().parse_args(base_argv(root, 0, 2)))
        st = R.prepare_all(args, R.import_modules(args))
        allsig = st["sigmas_all"]
        parts = [R.shard(allsig, k, 3) for k in range(3)]
        flat = [s for p in parts for s in p]
        check("A2 shards tile the spine list exactly once, in order",
              flat == list(allsig) and sum(len(p) for p in parts) == len(allsig)
              and len(allsig) == N_SPINES,
              "%d spines -> %s" % (len(allsig), [len(p) for p in parts]))
        check("A3 more shards than spines is harmless",
              [R.shard(allsig, k, 99) for k in range(99)].count([]) == 99 - len(allsig))

        # ---- two real shards against the stub network
        fac = multi_stub_factory()
        for k in (0, 1):
            rc = R.main(base_argv(root, k, 2), reader_factory=fac)
            check("A4.%d shard %d ran and exited 0" % (k, k), rc == 0)
        led = [R.task_paths(root + "/out", CELL, k) for k in (0, 1)]
        check("A5 both shards wrote a ledger and a meta sidecar",
              all(os.path.isfile(p["ledger"]) and os.path.isfile(p["meta"])
                  for p in led))
        fps = {json.load(open(p["meta"]))["fingerprint"] for p in led}
        check("A6 both shards agree on the parameter fingerprint", len(fps) == 1,
              str(fps))
        st6 = R.prepare_all(args, R.import_modules(args))
        part = st6["fingerprint_detail"].get("partition", {})
        check("A6' fingerprint records the three-vote partition",
              part.get("rule") == "three_vote"
              and st6["continuation_report"].get("applied") is True
              and part.get("rho_shaft_min") == 0.50
              and part.get("require_taper") is True, str(part))
        argsx = R.resolve(R.build_parser().parse_args(
            base_argv(root, 0, 2, "--no-shaft-stub-fix")))
        stx = R.prepare_all(argsx, R.import_modules(argsx))
        partx = stx["fingerprint_detail"].get("partition", {})
        check("A6'' --no-shaft-stub-fix yields partition rule 'none' and a "
              "DIFFERENT fingerprint",
              partx.get("rule") == "none"
              and stx["fingerprint"] != st6["fingerprint"], str(partx))
        SAF = R.import_modules(args)["h01_spine_area_F"]
        ids = set()
        for p in led:
            ids |= set(SAF.load_ledger(p["ledger"])[0])
        check("A7 the union of shard ledgers covers every selected spine",
              ids == set(allsig), "%d vs %d" % (len(ids), len(allsig)))

        # ---- merge
        margv = ["--root", root, "--cell", str(CELL), "--ntasks", "2",
                 "--stage1-dir", os.path.join(root, "stage1"),
                 "--kappa-min-per-bin", "1"]
        check("A7' the REAL sma_run / s0_ingest / shaft_continuation ran",
              all(os.path.join(root, "stage1") not in
                  getattr(R.import_modules(args)[m], "__file__", "")
                  for m in ("sma_run", "s0_ingest", "shaft_continuation")))
        check("A8 merge exits 0", M.main(margv) == 0)
        summ = os.path.join(root, "out", "spine_area_F_summary.csv")
        check("A9 merge wrote the S1 outputs",
              all(os.path.isfile(os.path.join(root, "out", f)) for f in (
                  "spine_area_F_summary.csv",
                  "neuron_%d_phi_mesh.csv" % CELL,
                  "cell%d_spines.csv" % CELL,
                  "cell%d_kappa_function.csv" % CELL)))
        import pandas as pd
        row = pd.read_csv(summ).iloc[-1]
        # The phantom dendrite is 12 um long, so NO segment lies beyond the
        # 60 um literature cutoff and F_lit is NaN by construction. F_whole is
        # the one with content here.
        check("A10 F_whole mesh is finite and the gate passed",
              np.isfinite(row["F_whole_mesh"]) and row["gate_max_abs_diff_um2"] < 1e-9,
              "F_whole_mesh %.4f" % row["F_whole_mesh"])
        check("A10' F_lit is NaN on a 12 um phantom, as it should be",
              not np.isfinite(row["F_lit_mesh"]))
        check("A11 the junction columns reached the summary",
              np.isfinite(row["kappa_pooled_norind"])
              and np.isfinite(row["base_frac_of_skel_pooled"]),
              "kappa_norind %.3f" % row["kappa_pooled_norind"])

        # ---- P3 (2026-09-15): deliverable, coverage accounting, QC
        check("A16 deliverable is mesh_beyond, fully covered, qc pass",
              row["deliverable_variant"] == "mesh_beyond"
              and row["qc_status"] == "pass"
              and abs(row["coverage_count_deliverable"] - 1.0) < 1e-12
              and int(row["n_fallback_mesh_beyond"]) == 0
              and np.isfinite(row["F_whole_deliverable"])
              and bool(row["cap_tips"]),
              "qc=%s cov=%.3f fb=%s F_whole=%.4f" % (
                  row["qc_status"], row["coverage_count_deliverable"],
                  row["n_fallback_mesh_beyond"], row["F_whole_deliverable"]))
        pm = pd.read_csv(os.path.join(root, "out", "neuron_%d_phi_mesh.csv" % CELL))
        ps = pd.read_csv(os.path.join(root, "out", "neuron_%d_phi_skel.csv" % CELL))
        check("A16' phi_mesh.csv is the deliverable's phi and phi_skel.csv exists",
              abs(1.0 + pm["spine_area_um2"].sum() / pm["shaft_area_um2"].sum()
                  - row["F_whole_deliverable"]) < 1e-9
              and "spine_cap_um2" in ps.columns and len(ps) == len(pm))

        # Tamper: one spine FAILED, one CLIPPED -> both must fall back to the
        # skeleton, and the summary must say so.
        import copy as _copy
        saved = {}
        for p in led:
            r_, h_ = SAF.load_ledger(p["ledger"])
            saved[p["ledger"]] = (_copy.deepcopy(r_), _copy.deepcopy(h_))
        sids = sorted(int(s) for p in led for s in SAF.load_ledger(p["ledger"])[0])
        s_fail, s_clip = sids[0], sids[-1]
        for p in led:
            r_, h_ = SAF.load_ledger(p["ledger"])
            if s_fail in r_:
                r_[s_fail]["ok"] = False
            if s_clip in r_:
                r_[s_clip]["clipped"] = True
            SAF.save_ledger(p["ledger"], r_, h_)
        check("A17 tampered merge exits 0", M.main(margv) == 0)
        row2 = pd.read_csv(os.path.join(root, "out", "spine_area_F_summary.csv")).iloc[-1]
        check("A17' one failed + one clipped -> 2 on fallback, qc low confidence",
              int(row2["n_failed"]) == 1 and int(row2["n_clipped"]) == 1
              and int(row2["n_fallback_mesh_beyond"]) == 2
              and abs(row2["coverage_count_deliverable"] - (N_SPINES - 2) / N_SPINES) < 1e-12
              and row2["qc_status"] == "pass_low_confidence"
              and "fallback" in str(row2["qc_reason"])
              and 0.0 < row2["fallback_area_frac_deliverable"] < 1.0
              and np.isfinite(row2["F_whole_deliverable"]),
              "failed=%s clipped=%s fb=%s cov=%.3f qc=%s area_fb=%.3f" % (
                  row2["n_failed"], row2["n_clipped"], row2["n_fallback_mesh_beyond"],
                  row2["coverage_count_deliverable"], row2["qc_status"],
                  row2["fallback_area_frac_deliverable"]))
        # Tamper: strip the base measurement from every record -> the
        # deliverable track is EMPTY -> F NaN and qc fail, never a silent copy.
        for p in led:
            r_, h_ = SAF.load_ledger(p["ledger"])
            for rec in r_.values():
                rec.pop("A_beyond_um2", None)
                rec.pop("s_base_nm", None)
            SAF.save_ledger(p["ledger"], r_, h_)
        check("A18 empty deliverable track merges (exit 0)", M.main(margv) == 0)
        row3 = pd.read_csv(os.path.join(root, "out", "spine_area_F_summary.csv")).iloc[-1]
        check("A18' empty deliverable track -> F NaN and qc fail, mesh track intact",
              row3["qc_status"] == "fail"
              and not np.isfinite(row3["F_whole_mesh_beyond"])
              and int(row3["n_measured_mesh_beyond"]) == 0
              and np.isfinite(row3["F_whole_mesh"]),
              "qc=%s F_beyond=%s F_mesh=%.4f" % (row3["qc_status"],
                                                 row3["F_whole_mesh_beyond"],
                                                 row3["F_whole_mesh"]))
        for path_, (r_, h_) in saved.items():
            SAF.save_ledger(path_, r_, h_)
        # Sidecar hole: a ledger with no meta must be REFUSED, not accepted.
        meta_bak = led[1]["meta"] + ".bak"
        os.rename(led[1]["meta"], meta_bak)
        try:
            M.main(margv)
            check("A19 merge refuses a ledger with no meta sidecar", False, "accepted")
        except SystemExit as e:
            check("A19 merge refuses a ledger with no meta sidecar",
                  "NO META SIDECAR" in str(e), str(e)[:70])
        os.rename(meta_bak, led[1]["meta"])

        # ---- negative paths
        # min-spine-value 1e9 demotes EVERY spine: the merge must reject the
        # shards on their fingerprint, and must not crash on the empty table.
        try:
            M.main(margv + ["--min-spine-value", "1e9"])
            check("A12 merge REFUSES shards built with other parameters", False,
                  "accepted")
        except SystemExit as e:
            check("A12 merge REFUSES shards built with other parameters",
                  "different parameters" in str(e), str(e)[:60])
        rc = R.main(base_argv(root, 0, 1, "--min-spine-value", "1e9", "--dry-run"))
        check("A12' REGRESSION: a cell with every spine demoted is handled",
              rc == 0)
        os.remove(led[1]["ledger"])
        try:
            M.main(margv)
            check("A13 merge refuses a missing shard by default", False, "accepted")
        except SystemExit as e:
            check("A13 merge refuses a missing shard by default",
                  "no ledger for task" in str(e), str(e)[:60])
        check("A14 --allow-missing proceeds anyway",
              M.main(margv + ["--allow-missing"]) == 0)
        for argv, why in ((base_argv(root, 5, 2), "task >= ntasks"),
                          (["--root", root, "--cell", "999", "--task", "0",
                            "--ntasks", "1", "--stage1-dir",
                            os.path.join(root, "stage1")], "missing neuron CSV"),
                          (["--root", os.path.join(tmp, "nope"), "--cell",
                            str(CELL), "--task", "0", "--ntasks", "1"],
                           "missing root")):
            try:
                R.main(argv)
                check("A15 runner refuses: %s" % why, False, "accepted")
            except SystemExit:
                check("A15 runner refuses: %s" % why, True)

        # ---- Section B: the job scripts
        job_script_checks(code_dir, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n%d checks passed, %d failed" % (len(RESULTS) - n_fail, n_fail))
    print("ALL GREEN" if n_fail == 0 else "FAILURES")
    return n_fail


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
