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
cannot see: spine_area_F.pbs is parsed for its three guards (the --g-table
flag that points at H01_CODE; the activation block that trusts the outcome of
`conda activate` rather than its exit status; the env knob H01_ENV, with a
stale ENV_NAME from the login shell reported and ignored) and is then RUN with
bash against a fixture tree, with stub python3 / conda / curl on PATH -- the
conda stub installs a real shell function through the hook, so activation
changes PATH as the real one does -- asserting on the argv the stub interpreter
received, on which env's interpreter ran, and on every refusal path.
run_smoke_tests.sh gets the text checks, probe_net.pbs the same treatment.
Needs bash on PATH (the cluster and the sandbox both have it); no network.

Section C (2026-09-22, decision D-004) tests campaign.pbs, the orchestrator
that runs P1 -> P2 -> P3 for every cell of a population from ONE submission:
the (row, shard) decomposition of PBS_ARRAY_INDEX, P1 on shard 0 only, the
cell id of every stage read from the same manifest row, --stage1-dir and
--g-table under H01_CODE in every call, --deliverable mesh_beyond explicit
and --allow-missing absent, the fingerprint knobs mirrored to P2 and P3, the
deferred merge (SHARDS > 1) and PHASE=merge, DRY_RUN, a P1 failure stopping
the cell before P2, and every refusal (MANIFEST, SHARDS, PHASE, an index
past the manifest, a non-integer cell id, a bad first column).

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
# Section B: the job scripts. Three defects shipped in the job scripts that no
# Python suite could catch -- the runner's --g-table default resolves into
# H01_ROOT while the table is tracked in H01_CODE (every array task died at
# resolve() after the queue wait); the activation block obeyed the exit
# status of `conda activate`, which is 1 on this cluster while activation is
# real (binutils activate.d hook); and the env knob was the bare ENV_NAME,
# which the login shell exports for another project (sbi_export), so the
# scripts activated THAT env and the outcome check passed (2026-09-20). All
# live in bash, so all are tested in bash: text checks for the guards, then
# real runs against a fixture with a stub conda whose hook installs a real
# shell function, so activation changes PATH the way the real one does.
# ---------------------------------------------------------------------------
PBS = "spine_area_F.pbs"
PROBE = "probe_net.pbs"
RUNNER_SH = "run_smoke_tests.sh"
JOB_CELL = 424242
_ARGV_N = [0]

STUB_PY = """#!/bin/bash
# stub python3 for the job-script test: records argv, runs nothing.
# STUB_FAIL_ON=<script.py> makes that one invocation exit 1 (section C).
printf '%s\\n' "$@" >> "$STUB_ARGV_OUT"
if [ -n "${STUB_FAIL_ON:-}" ] && [ "$1" = "$STUB_FAIL_ON" ]; then
    echo "STUB: failing $1 on request"
    exit 1
fi
exit 0
"""

# `module load proxy` on a compute node; here a stub that only says it ran,
# so section C can assert it precedes P2 and nothing else.
STUB_MODULE = """#!/bin/bash
echo "STUB module $*"
exit 0
"""

# `conda shell.bash hook` prints a conda() function, as the real hook does.
# `conda activate <env>` prepends $STUB_ENVS/<env>/bin to PATH when that env
# exists, sets CONDA_DEFAULT_ENV, prints an INFO line and RETURNS 1 -- which
# is what the binutils activate.d hook does on the cluster while the env HAS
# been activated. An env that does not exist changes nothing (and returns 1).
# The heredoc is unquoted so $STUB_ENVS is baked in at hook time; the \\$ are
# what keep $1, $2 and $PATH for the function body.
STUB_CONDA = r"""#!/bin/bash
case "$1" in
    shell.bash)
        cat <<EOF
conda() {
    if [ "\$1" = activate ]; then
        echo "INFO: stub activate.d hook (\$2)"
        if [ -d "$STUB_ENVS/\$2/bin" ]; then
            export PATH="$STUB_ENVS/\$2/bin:\$PATH"
            export CONDA_DEFAULT_ENV="\$2"
        fi
        return 1
    fi
    return 0
}
EOF
        ;;
esac
exit 0
"""

STUB_CURL = """#!/bin/bash
printf '000'
exit 0
"""

FIXTURE_ENVS = ("spine_env", "sbi_export", "other_env")


def _write_exec(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o755)


def job_fixture(tmp, with_table=True):
    """A tree shaped like the cluster: H01_CODE with the runner file and the g
    table, H01_ROOT with neurons/, bin/ with the stubs and a python3 that is
    in NO env, envs/<name>/bin/python3 for each of FIXTURE_ENVS, and
    home/.conda/envs/spine_env/bin/python3 for probe_net.pbs."""
    fx = os.path.join(tmp, "jobfix")
    shutil.rmtree(fx, ignore_errors=True)
    code = os.path.join(fx, "h01_code")
    root = os.path.join(fx, "h01")
    pydirs = [os.path.join(fx, "bin"),
              os.path.join(fx, "home", ".conda", "envs", "spine_env", "bin")]
    pydirs += [os.path.join(fx, "envs", e, "bin") for e in FIXTURE_ENVS]
    for d in [os.path.join(code, "stage1"), os.path.join(root, "neurons")] + pydirs:
        os.makedirs(d, exist_ok=True)
    open(os.path.join(code, "run_spine_area_F.py"), "w").close()
    if with_table:
        open(os.path.join(code, "g_table_cyl_2deg.npz"), "w").close()
    for d in pydirs:
        _write_exec(os.path.join(d, "python3"), STUB_PY)
    _write_exec(os.path.join(fx, "bin", "conda"), STUB_CONDA)
    _write_exec(os.path.join(fx, "bin", "curl"), STUB_CURL)
    _write_exec(os.path.join(fx, "bin", "module"), STUB_MODULE)
    return fx, code, root


def run_job(script, fx, code, root, extra_env):
    """bash <script> with a scrubbed environment: no inherited CODE/ROOT/
    ENV_NAME/H01_*/PBS_*/CONDA*, HOME inside the fixture, the stub bin first
    on PATH. Returns (rc, stdout, argv), argv being what the stub interpreter
    received ([] if never run)."""
    _ARGV_N[0] += 1
    argv_out = os.path.join(fx, "argv_%d.txt" % _ARGV_N[0])
    path = os.path.join(fx, "bin") + os.pathsep + os.environ.get("PATH", "")
    drop = ("CODE", "ROOT", "CELL", "SUBSET", "NTASKS", "DRY_RUN", "SKIP_CONDA",
            "ENV_NAME", "MIN_SPINE_VALUE", "NO_MEASURE_BASE", "STUB_ARGV_OUT",
            "STUB_ENVS", "STUB_FAIL_ON", "MANIFEST", "SHARDS", "PHASE",
            "PASSIVE_TABLE", "OUT_DIR", "FORCE", "NO_RIGIDITY",
            "NO_NEURON_VALIDATE", "MIN_COVERAGE")
    env = {k: v for k, v in os.environ.items()
           if not (k in drop or k.startswith("H01_") or k.startswith("PBS_")
                   or k.startswith("CONDA"))}
    env.update({"PATH": path, "HOME": os.path.join(fx, "home"),
                "H01_CODE": code, "H01_ROOT": root, "STUB_ARGV_OUT": argv_out,
                "STUB_ENVS": os.path.join(fx, "envs")})
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


def _ran_python_from(out, fx, env_name):
    """The .pbs echoes `python <path>` after activation; True if that path is
    the fixture's envs/<env_name>/bin/python3."""
    want = "python " + os.path.join(fx, "envs", env_name, "bin", "python3")
    return any(ln.strip() == want for ln in out.splitlines())


def job_script_checks(code_dir, tmp):
    pbs_path = os.path.join(code_dir, PBS)
    probe_path = os.path.join(code_dir, PROBE)
    text = open(pbs_path).read()
    lines = text.splitlines()
    rtext = open(os.path.join(code_dir, RUNNER_SH)).read()

    # ---- B1 text: the g-table flag and its preflight
    inv = pbs_python_invocation(text)
    check("B1 .pbs passes --g-table under H01_CODE to the runner",
          '--g-table "$H01_CODE/g_table_cyl_2deg.npz"' in inv, inv[:90])
    check("B1' .pbs checks the g table exists before starting python",
          'if [ ! -f "$H01_CODE/g_table_cyl_2deg.npz" ]' in text)

    # ---- B2 text: activation trusts the OUTCOME, guards in the right order,
    # env knob is H01_ENV in BOTH scripts and ENV_NAME is reported, not read
    i_hook = _first_line(lines, 'eval "$(conda shell.bash hook)"')
    i_act = _first_line(lines, 'conda activate "$H01_ENV"', i_hook + 1)
    i_pe = _first_line(lines, "set +e")
    i_pu = _first_line(lines, "set +u")
    i_me = _first_line(lines, "set -e", i_act + 1)
    i_mu = _first_line(lines, "set -u", i_act + 1)
    check("B2 activation: set +u and set +e BEFORE the hook, -e/-u restored AFTER activate",
          0 <= i_pu < i_hook and 0 <= i_pe < i_hook and i_hook < i_act
          and i_act < i_me and i_act < i_mu,
          "+u %d +e %d hook %d act %d -e %d -u %d" % (i_pu, i_pe, i_hook, i_act, i_me, i_mu))
    check("B2' activation verified by where python3 resolves, not by exit status",
          '*"/envs/$H01_ENV/"*)' in text and "did not take effect" in text)
    check("B2'' path knobs are H01_ROOT / H01_CODE and stale CODE/ROOT are reported",
          '"${H01_CODE:-' in text and '"${H01_ROOT:-' in text
          and "for _stale in ROOT CODE" in text)
    for name, t in ((PBS, text), (RUNNER_SH, rtext)):
        check("B2''' %s: env knob is H01_ENV; ENV_NAME reported and ignored" % name,
              'H01_ENV="${H01_ENV:-spine_env}"' in t
              and "NOTE: ENV_NAME is set" in t
              and "${ENV_NAME:-spine_env}" not in t
              and 'conda activate "$ENV_NAME"' not in t)

    # ---- B3 text: probe_net.pbs no longer points at the retired repo
    ptext = open(probe_path).read()
    plines = [ln for ln in ptext.splitlines() if not ln.lstrip().startswith("#")]
    check("B3 probe_net.pbs has no live reference to TEEG/Spines",
          not any("TEEG/Spines" in ln for ln in plines))
    check("B3' probe_net.pbs uses the H01_CODE knob with the Towards-EEG default",
          'H01_CODE="${H01_CODE:-/davinci-1/home/ldellamea/TEEG/Towards-EEG/h01_code}"' in ptext
          and "$CODE" not in ptext)

    # ---- B4 syntax and bytes of every shell script shipped
    for name in (PBS, PROBE, RUNNER_SH, "stage1_link.sh"):
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

    # ---- B6 the activation block, through the stub hook
    # Positive: `conda activate spine_env` RETURNS 1 (as binutils does) but
    # python3 now resolves under envs/spine_env/bin -> the job must proceed,
    # and the interpreter that ran must be the env's.
    rc, out, argv = run_job(pbs_path, fx, code, root, {"CELL": str(JOB_CELL)})
    check("B6 conda activate returning 1 with the env really activated: job PROCEEDS "
          "under envs/spine_env",
          rc == 0 and "INFO: stub activate.d hook (spine_env)" in out
          and _ran_python_from(out, fx, "spine_env")
          and argv[:1] == ["run_spine_area_F.py"],
          "rc=%d\n%s" % (rc, out[-500:]) if rc else "")
    # Negative: an env that does not exist -> PATH unchanged -> refused.
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            {"CELL": str(JOB_CELL), "H01_ENV": "no_such_env"})
    check("B6' activation that did not take effect: job REFUSES before python",
          rc != 0 and "did not take effect" in out and argv == [],
          "rc=%d" % rc)
    # The 2026-09-20 collision: ENV_NAME=sbi_export in the environment and an
    # sbi_export env that exists. The scripts must NOT activate it: NOTE line,
    # spine_env activated, the env's interpreter ran.
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            {"CELL": str(JOB_CELL), "ENV_NAME": "sbi_export"})
    check("B6'' stale ENV_NAME=sbi_export in the environment: reported, IGNORED, "
          "spine_env activated",
          rc == 0 and "NOTE: ENV_NAME is set (sbi_export)" in out
          and "INFO: stub activate.d hook (spine_env)" in out
          and "hook (sbi_export)" not in out
          and _ran_python_from(out, fx, "spine_env")
          and argv[:1] == ["run_spine_area_F.py"],
          "rc=%d\n%s" % (rc, out[-500:]))
    # The real knob still works.
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            {"CELL": str(JOB_CELL), "H01_ENV": "other_env"})
    check("B6''' H01_ENV=other_env selects that env",
          rc == 0 and "hook (other_env)" in out and _ran_python_from(out, fx, "other_env")
          and argv[:1] == ["run_spine_area_F.py"], "rc=%d" % rc)

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


# ---------------------------------------------------------------------------
# Section C: campaign.pbs (decision D-004, 2026-09-22). One submission must
# leave every cell of a population complete -- P1, P2 and P3 -- with the mesh
# spine area as THE F. The script decomposes PBS_ARRAY_INDEX into (manifest
# row, shard), so the checks below assert the argv of all three interpreter
# calls at non-trivial (N, S), not merely that bash exited 0.
# ---------------------------------------------------------------------------
CAMPAIGN = "campaign.pbs"
CAMPAIGN_CELLS = (1302789404, 1317492596, 1333261412)
CAMPAIGN_SCRIPTS = ("run_p1_export.py", "run_spine_area_F.py", "merge_spine_area_F.py")


def campaign_fixture(tmp, cells=CAMPAIGN_CELLS, first_col="cell_id"):
    """job_fixture plus what campaign.pbs checks for: the two other drivers,
    a passive table, and a three-row manifest under H01_ROOT/p1/manifests."""
    fx, code, root = job_fixture(tmp)
    for name in ("run_p1_export.py", "merge_spine_area_F.py"):
        open(os.path.join(code, name), "w").close()
    with open(os.path.join(code, "passive_params.csv"), "w") as fh:
        fh.write("layer,cell_type,cm_uF_cm2,Ra_ohm_cm\nL3,exc,0.5,268.5\n")
    mdir = os.path.join(root, "p1", "manifests")
    os.makedirs(mdir, exist_ok=True)
    man = os.path.join(mdir, "L3_exc.csv")
    with open(man, "w") as fh:
        fh.write("%s,layer,cell_type,neuron_csv,alignment_metadata,synapse_csv,layer_source\n"
                 % first_col)
        for c in cells:
            fh.write("%s,L3,exc,neurons/neuron_%s.csv,neurons/alignment_metadata_L3.csv,"
                     "synapses/neuron_%s_synapses.csv,bank\n" % (c, c, c))
    return fx, code, root, man


def split_calls(argv):
    """The stub appends every invocation's argv to one file; split it back
    into one list per interpreter call, keyed by script name (in order)."""
    calls = []
    for a in argv:
        if a in CAMPAIGN_SCRIPTS:
            calls.append([a])
        elif calls:
            calls[-1].append(a)
    return calls


def _scripts(calls):
    return [c[0] for c in calls]


def campaign_checks(code_dir, tmp):
    pbs_path = os.path.join(code_dir, CAMPAIGN)
    text = open(pbs_path).read()
    lines = text.splitlines()
    live = [ln for ln in lines if not ln.lstrip().startswith("#")]

    # ---- C1 text: the same three guards as spine_area_F.pbs / p1_export.pbs
    i_hook = _first_line(lines, 'eval "$(conda shell.bash hook)"')
    i_act = _first_line(lines, 'conda activate "$H01_ENV"', i_hook + 1)
    i_pe = _first_line(lines, "set +e")
    i_pu = _first_line(lines, "set +u")
    i_me = _first_line(lines, "set -e", i_act + 1)
    i_mu = _first_line(lines, "set -u", i_act + 1)
    check("C1 campaign.pbs activation block: +u/+e before the hook, -e/-u after activate",
          0 <= i_pu < i_hook and 0 <= i_pe < i_hook and i_hook < i_act
          and i_act < i_me and i_act < i_mu)
    check("C1a campaign.pbs: outcome check, H01_ENV knob, ENV_NAME reported not read",
          '*"/envs/$H01_ENV/"*)' in text and "did not take effect" in text
          and 'H01_ENV="${H01_ENV:-spine_env}"' in text
          and "NOTE: ENV_NAME is set" in text
          and "${ENV_NAME:-spine_env}" not in text
          and 'conda activate "$ENV_NAME"' not in text)
    check("C1b campaign.pbs: path knobs H01_ROOT / H01_CODE, stale CODE/ROOT reported",
          '"${H01_CODE:-' in text and '"${H01_ROOT:-' in text
          and "for _stale in ROOT CODE" in text)
    check("C1c campaign.pbs: --allow-missing appears NOWHERE (a dead shard fails the merge)",
          "--allow-missing" not in text)
    check("C1d campaign.pbs: --deliverable is mesh_beyond, explicitly, in live code (D-004)",
          any("--deliverable mesh_beyond" in ln for ln in live)
          and not any("--deliverable" in ln and "mesh_beyond" not in ln for ln in live))
    check("C1e campaign.pbs: g table checked before python, MANIFEST is :? required",
          'if [ ! -f "$H01_CODE/g_table_cyl_2deg.npz" ]' in text
          and 'MANIFEST="${MANIFEST:?' in text)
    rc = subprocess.run(["bash", "-n", pbs_path], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, universal_newlines=True)
    raw = open(pbs_path, "rb").read()
    check("C1f campaign.pbs: bash -n passes, LF only, pure ASCII",
          rc.returncode == 0 and b"\r" not in raw and all(b < 128 for b in raw),
          rc.stdout.strip()[:80])

    # ---- C2 the decomposition at (N=3, S=4): index 5 -> row 1, shard 1
    fx, code, root, man = campaign_fixture(tmp)
    stage1 = os.path.join(code, "stage1")
    gtab = os.path.join(code, "g_table_cyl_2deg.npz")
    base = {"SKIP_CONDA": "1", "MANIFEST": man}
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, SHARDS="4", PBS_ARRAY_INDEX="5"))
    calls = split_calls(argv)
    check("C2 (N=3,S=4) index 5 -> row 1 shard 1: P2 only, cell = manifest row 1",
          rc == 0 and _scripts(calls) == ["run_spine_area_F.py"]
          and _argval(calls[0], "--cell") == str(CAMPAIGN_CELLS[1])
          and _argval(calls[0], "--task") == "1" and _argval(calls[0], "--ntasks") == "4"
          and "row 1 shard 1/4" in out,
          "rc=%d scripts=%s\n%s" % (rc, _scripts(calls), out[-400:]))
    check("C2a P1 does NOT run on shard 1", "run_p1_export.py" not in _scripts(calls))
    check("C2b with S > 1 the merge is deferred; the hint names PHASE=merge and -J 0-2",
          "merge_spine_area_F.py" not in _scripts(calls)
          and "P3 merge deferred" in out and "PHASE=merge" in out and "-J 0-2" in out)
    # index 8 -> row 2, shard 0: P1 then P2, same cell in both, P3 deferred
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, SHARDS="4", PBS_ARRAY_INDEX="8"))
    calls = split_calls(argv)
    p1 = calls[0] if calls else []
    p2 = calls[1] if len(calls) > 1 else []
    check("C3 (N=3,S=4) index 8 -> row 2 shard 0: P1 then P2, in that order",
          rc == 0 and _scripts(calls) == ["run_p1_export.py", "run_spine_area_F.py"],
          "rc=%d scripts=%s\n%s" % (rc, _scripts(calls), out[-400:]))
    check("C3a P1 --task is the manifest row (2) and P2 --cell is that row's cell id",
          _argval(p1, "--task") == "2" and _argval(p1, "--manifest") == man
          and _argval(p2, "--cell") == str(CAMPAIGN_CELLS[2])
          and _argval(p2, "--task") == "0" and _argval(p2, "--ntasks") == "4",
          "p1=%s p2=%s" % (p1, p2))
    check("C3b P1 receives --root, --stage1-dir under H01_CODE, the passive table, --out-dir root/p1, -v",
          _argval(p1, "--root") == root and _argval(p1, "--stage1-dir") == stage1
          and _argval(p1, "--passive-table") == os.path.join(code, "passive_params.csv")
          and _argval(p1, "--out-dir") == os.path.join(root, "p1") and "-v" in p1
          and "--force" not in p1 and "--dry-run" not in p1)
    check("C3c P2 receives --stage1-dir and --g-table under H01_CODE, no --subset/--dry-run",
          _argval(p2, "--root") == root and _argval(p2, "--stage1-dir") == stage1
          and _argval(p2, "--g-table") == gtab
          and "--subset" not in p2 and "--dry-run" not in p2)
    o_lines = out.splitlines()
    i_p1 = next((i for i, ln in enumerate(o_lines) if ln.startswith("=== P1 export")), -1)
    i_mod = next((i for i, ln in enumerate(o_lines) if ln.startswith("STUB module load proxy")), -1)
    i_p2 = next((i for i, ln in enumerate(o_lines) if ln.startswith("=== P2 measure")), -1)
    check("C3d module load proxy runs after P1 and before P2 (once)",
          0 <= i_p1 < i_p2 <= i_mod
          and sum(1 for ln in o_lines if ln.startswith("STUB module load proxy")) == 1,
          "p1 %d p2 %d module %d" % (i_p1, i_p2, i_mod))

    # ---- C4 S=1 (the default): P1, P2, P3 in one task, all on the same cell
    rc, out, argv = run_job(pbs_path, fx, code, root, dict(base, PBS_ARRAY_INDEX="1"))
    calls = split_calls(argv)
    check("C4 SHARDS unset -> S=1: P1, P2, P3 run in one task, in order",
          rc == 0 and _scripts(calls) == list(CAMPAIGN_SCRIPTS),
          "rc=%d scripts=%s\n%s" % (rc, _scripts(calls), out[-400:]))
    if len(calls) == 3:
        p1, p2, p3 = calls
        check("C4a the same cell in all three: P1 row 1, P2/P3 --cell = that row's id",
              _argval(p1, "--task") == "1"
              and _argval(p2, "--cell") == str(CAMPAIGN_CELLS[1])
              and _argval(p3, "--cell") == str(CAMPAIGN_CELLS[1])
              and _argval(p2, "--ntasks") == "1" and _argval(p3, "--ntasks") == "1")
        check("C4b P3 receives --stage1-dir AND --g-table under H01_CODE (its resolve() needs both), "
              "--deliverable mesh_beyond, no --allow-missing, no --min-coverage unless asked",
              _argval(p3, "--root") == root and _argval(p3, "--stage1-dir") == stage1
              and _argval(p3, "--g-table") == gtab
              and _argval(p3, "--deliverable") == "mesh_beyond"
              and "--allow-missing" not in p3 and "--min-coverage" not in p3, str(p3))
    else:
        check("C4a (skipped: three calls expected)", False)
        check("C4b (skipped: three calls expected)", False)

    # ---- C5 the fingerprint knobs reach P2 AND P3; P1-only knobs stay on P1
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, PBS_ARRAY_INDEX="0", SUBSET="300",
                                 MIN_SPINE_VALUE="1.5", NO_MEASURE_BASE="1",
                                 MIN_COVERAGE="0.9", FORCE="1", NO_RIGIDITY="1",
                                 NO_NEURON_VALIDATE="1"))
    calls = split_calls(argv)
    if rc == 0 and len(calls) == 3:
        p1, p2, p3 = calls
        check("C5 SUBSET / MIN_SPINE_VALUE / NO_MEASURE_BASE mirrored to P2 and P3 identically",
              all(_argval(c, "--subset") == "300" and _argval(c, "--min-spine-value") == "1.5"
                  and "--no-measure-base" in c for c in (p2, p3))
              and "--subset" not in p1)
        check("C5a MIN_COVERAGE reaches P3 only; FORCE / NO_RIGIDITY / NO_NEURON_VALIDATE reach P1 only",
              _argval(p3, "--min-coverage") == "0.9" and "--min-coverage" not in p2
              and "--force" in p1 and "--no-rigidity-control" in p1
              and "--no-neuron-validate" in p1
              and not any("--force" in c or "--no-rigidity-control" in c for c in (p2, p3)))
    else:
        check("C5 (run failed: rc=%d, %d calls)" % (rc, len(calls)), False, out[-300:])
        check("C5a (run failed)", False)

    # ---- C6 DRY_RUN: --dry-run to P1 and P2, P3 skipped with a reason
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, PBS_ARRAY_INDEX="0", DRY_RUN="1"))
    calls = split_calls(argv)
    check("C6 DRY_RUN=1: P1 and P2 get --dry-run, P3 is skipped and says why",
          rc == 0 and _scripts(calls) == ["run_p1_export.py", "run_spine_area_F.py"]
          and all("--dry-run" in c for c in calls) and "P3 merge SKIPPED (DRY_RUN)" in out,
          "rc=%d scripts=%s" % (rc, _scripts(calls)))

    # ---- C7 PHASE=merge: index = row, P3 only, --ntasks = SHARDS
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, SHARDS="4", PHASE="merge", PBS_ARRAY_INDEX="2"))
    calls = split_calls(argv)
    check("C7 PHASE=merge, S=4, index 2: P3 only, --cell = row 2, --ntasks 4, no module load",
          rc == 0 and _scripts(calls) == ["merge_spine_area_F.py"]
          and _argval(calls[0], "--cell") == str(CAMPAIGN_CELLS[2])
          and _argval(calls[0], "--ntasks") == "4"
          and _argval(calls[0], "--deliverable") == "mesh_beyond"
          and "STUB module" not in out,
          "rc=%d scripts=%s\n%s" % (rc, _scripts(calls), out[-300:]))

    # ---- C8 a P1 failure stops the cell before P2 (and the task exits 1)
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, PBS_ARRAY_INDEX="0", STUB_FAIL_ON="run_p1_export.py"))
    calls = split_calls(argv)
    check("C8 P1 exits 1 -> the task exits non-zero, names the cell, and P2/P3 never run",
          rc != 0 and _scripts(calls) == ["run_p1_export.py"]
          and "P1 export failed for cell %d" % CAMPAIGN_CELLS[0] in out,
          "rc=%d scripts=%s" % (rc, _scripts(calls)))
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, PBS_ARRAY_INDEX="0", STUB_FAIL_ON="run_spine_area_F.py"))
    calls = split_calls(argv)
    check("C8a P2 exits 1 -> the task exits non-zero and P3 never runs (no silent merge)",
          rc != 0 and _scripts(calls) == ["run_p1_export.py", "run_spine_area_F.py"],
          "rc=%d scripts=%s" % (rc, _scripts(calls)))

    # ---- C9 refusals, each before any interpreter call
    rc, out, argv = run_job(pbs_path, fx, code, root, {"SKIP_CONDA": "1"})
    check("C9 unset MANIFEST: refused naming the knob, python never invoked",
          rc != 0 and "MANIFEST" in out and argv == [], "rc=%d" % rc)
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, MANIFEST=os.path.join(root, "nope.csv")))
    check("C9a missing manifest file: refused", rc != 0 and "does not exist" in out and argv == [])
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, SHARDS="4", PBS_ARRAY_INDEX="12"))
    check("C9b index past the manifest (N=3,S=4, index 12): refused, names -J 0-11",
          rc != 0 and "-J 0-11" in out and argv == [], out[-200:])
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, SHARDS="4", PHASE="merge", PBS_ARRAY_INDEX="3"))
    check("C9c PHASE=merge index past N: refused", rc != 0 and argv == [])
    for bad, why in (("0", "SHARDS=0"), ("x", "SHARDS=x")):
        rc, out, argv = run_job(pbs_path, fx, code, root, dict(base, SHARDS=bad))
        check("C9d %s: refused" % why, rc != 0 and "SHARDS" in out and argv == [])
    rc, out, argv = run_job(pbs_path, fx, code, root, dict(base, PHASE="later"))
    check("C9e PHASE=later: refused", rc != 0 and "PHASE" in out and argv == [])
    fxb, codeb, rootb, manb = campaign_fixture(tmp, cells=("12ab",))
    rc, out, argv = run_job(pbs_path, fxb, codeb, rootb, {"SKIP_CONDA": "1", "MANIFEST": manb})
    check("C9f non-integer cell_id in the manifest row: refused", rc != 0
          and "not an integer" in out and argv == [])
    fxc, codec, rootc, manc = campaign_fixture(tmp, first_col="neuron_id")
    rc, out, argv = run_job(pbs_path, fxc, codec, rootc, {"SKIP_CONDA": "1", "MANIFEST": manc})
    check("C9g manifest whose first column is not cell_id: refused", rc != 0
          and "expected cell_id" in out and argv == [])
    for missing in ("run_p1_export.py", "merge_spine_area_F.py"):
        fxd, coded, rootd, mand = campaign_fixture(tmp)
        os.remove(os.path.join(coded, missing))
        rc, out, argv = run_job(pbs_path, fxd, coded, rootd, {"SKIP_CONDA": "1", "MANIFEST": mand})
        check("C9h H01_CODE without %s: refused" % missing,
              rc != 0 and missing in out and argv == [])

    # ---- C10 the environment collisions, same as B6''/B7'
    fx, code, root, man = campaign_fixture(tmp)
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            {"MANIFEST": man, "ENV_NAME": "sbi_export",
                             "CODE": "/nowhere/other", "ROOT": "/nowhere/else"})
    calls = split_calls(argv)
    check("C10 stale ENV_NAME / CODE / ROOT: NOTEs printed, spine_env activated, right root used",
          rc == 0 and "NOTE: ENV_NAME is set (sbi_export)" in out
          and "NOTE: CODE is set" in out and "NOTE: ROOT is set" in out
          and "hook (sbi_export)" not in out and _ran_python_from(out, fx, "spine_env")
          and len(calls) == 3 and all(_argval(c, "--root") == root for c in calls),
          "rc=%d\n%s" % (rc, out[-400:]))
    rc, out, argv = run_job(pbs_path, fx, code, root, {"MANIFEST": man, "H01_ENV": "no_such_env"})
    check("C10a activation that did not take effect: refused before python",
          rc != 0 and "did not take effect" in out and argv == [])

    # ---- C11 relative MANIFEST / PASSIVE_TABLE / OUT_DIR resolve like p1_export.pbs
    rel_man = os.path.relpath(man, code)
    with open(os.path.join(code, "passive_params_inh_SST.csv"), "w") as fh:
        fh.write("layer,cell_type,cm_uF_cm2,Ra_ohm_cm\nL3,inh,1.0,100\n")
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, MANIFEST=rel_man, PASSIVE_TABLE="passive_params_inh_SST.csv",
                                 OUT_DIR="p1_inh_SST", PBS_ARRAY_INDEX="0"))
    calls = split_calls(argv)
    p1 = calls[0] if calls else []
    check("C11 relative MANIFEST (vs H01_CODE), PASSIVE_TABLE (vs H01_CODE), OUT_DIR (vs H01_ROOT)",
          rc == 0 and len(calls) == 3
          and os.path.normpath(_argval(p1, "--manifest") or "") == os.path.normpath(man)
          and _argval(p1, "--passive-table") == os.path.join(code, "passive_params_inh_SST.csv")
          and _argval(p1, "--out-dir") == os.path.join(root, "p1_inh_SST"),
          "rc=%d p1=%s" % (rc, p1))
    check("C11a P2/P3 outputs are NOT per passive tree: no --out-dir passed to P2 or P3 (D-004 item 2)",
          len(calls) == 3 and all("--out-dir" not in c for c in calls[1:]))

    # ---- C12 a manifest that crossed a Windows boundary: CRLF and a blank
    # line. Rows are counted as pandas counts them (blank lines skipped), the
    # CR never reaches the cell id, and zero-padded knobs are decimal.
    fx, code, root, man = campaign_fixture(tmp)
    with open(man, "wb") as fh:
        fh.write(b"cell_id,layer,cell_type,neuron_csv,alignment_metadata,synapse_csv,layer_source\r\n"
                 b"111,L3,exc,a,b,,bank\r\n\r\n222,L3,exc,a,b,,bank\r\n333,L3,exc,a,b,,bank\r\n")
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, MANIFEST=man, SHARDS="08", PBS_ARRAY_INDEX="09"))
    calls = split_calls(argv)
    check("C12 CRLF manifest with a blank line, SHARDS=08, index 09: row 1 shard 1/8, cell 222, no CR",
          rc == 0 and _scripts(calls) == ["run_spine_area_F.py"]
          and _argval(calls[0], "--cell") == "222" and _argval(calls[0], "--ntasks") == "8"
          and "row 1 shard 1/8" in out, "rc=%d %s\n%s" % (rc, _scripts(calls), out[-300:]))
    rc, out, argv = run_job(pbs_path, fx, code, root,
                            dict(base, MANIFEST=man, SHARDS="8", PBS_ARRAY_INDEX="24"))
    check("C12a ...and index 24 (row 3 of 3) is refused naming -J 0-23",
          rc != 0 and "-J 0-23" in out and argv == [])


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
        # ---- Section C: the campaign orchestrator (D-004)
        campaign_checks(code_dir, tmp)
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
