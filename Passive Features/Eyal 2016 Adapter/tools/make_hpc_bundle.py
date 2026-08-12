"""
make_hpc_bundle.py -- assemble ONE self-verifying, offline-complete bundle in
Google Drive for transfer to an internet-restricted cluster.

Run this in Colab AFTER Save_Eyal2016_Data.py has built the archive.

Why a bundle rather than just the archive tarball
-------------------------------------------------
The cluster cannot reach the internet, so anything it will ever need must
travel in this one file: the Phase 0 archive, the ModelDB source release
(without which smoke tests 1-11 cannot re-run), the patch tooling, and the
tests themselves. Nothing here downloads anything at run time.

Why checksums
-------------
The transfer path (Drive -> browser -> Windows -> MobaXterm/SFTP -> cluster)
crosses a Windows boundary. Two corruption classes are silent there and
invisible in an editor:

  * non-ASCII bytes in .py source mangled by a cp1252 re-encode
  * CRLF injected into any file with a shebang (.sh), which breaks it with
    "bad interpreter: /bin/bash^M" -- and which an ASCII scan CANNOT catch,
    because \\r is itself ASCII

Because the cluster cannot re-download a reference copy, corruption must be
detectable from the bundle alone. Every file is therefore SHA-256'd here,
before packing, and verify_on_cluster.py re-checks them there.

Everything is packed into a single .tar.gz: a binary archive cannot have its
line endings rewritten in transit, whereas loose .py/.sh files can.

Output
------
    <DRIVE_OUT>/eyal_hpc_bundle.tar.gz     the one file to move
    <DRIVE_OUT>/eyal_hpc_bundle.sha256     its own checksum, to compare after

ASCII-only, LF-only by construction.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

BUNDLE_NAME = "eyal_hpc_bundle"

TOOLS = [
    "eyal_archive_builder.py",
    "eyal_reference_scalars.py",
    "smoke_eyal_archive_builder.py",
    "smoke_eyal_neuron_build.py",
    "patch_eyal_support.py",
    "regression_allen_unchanged.py",
    # Colab-side, not used on the cluster, but bundled so that one artefact
    # contains the whole reproducible chain (and so the git push captures it).
    "Save_Eyal2016_Data.py",
    "make_hpc_bundle.py",
]

# Cluster-side scripts, placed at the bundle ROOT (not in tools/) because
# they are the things you actually invoke.
SCRIPTS = [
    "run_all_smoke_tests.sh",
    "submit_eyal_fit.sh",
    "push_eyal_adapter.sh",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_text_files(root: Path) -> Tuple[List[str], List[str]]:
    """Return (non_ascii_py, crlf_shebang) offenders under `root`."""
    bad_ascii: List[str] = []
    bad_crlf: List[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".py":
            data = p.read_bytes()
            if any(b > 127 for b in data):
                bad_ascii.append(str(p.relative_to(root)))
        if p.suffix in (".sh", ".pbs", ".slurm"):
            if p.read_bytes().count(b"\r"):
                bad_crlf.append(str(p.relative_to(root)))
    return bad_ascii, bad_crlf


VERIFY_PY = r'''"""
verify_on_cluster.py -- run on the LOGIN NODE, from inside the extracted
bundle directory, BEFORE any qsub. No network, no allocation needed.

    python3 verify_on_cluster.py

WHY THIS IS PYTHON AND NOT A SHELL SCRIPT
-----------------------------------------
A .sh verifier cannot survive the corruption it exists to detect. If CRLF
is injected in transit, the kernel looks for an interpreter named
"/bin/bash\r", and the script dies with a cryptic parse error instead of
reporting the problem -- exactly when you most need it to work.

Invoked as `python3 verify_on_cluster.py`, the interpreter is named
explicitly, so no shebang is read; and Python 3 accepts universal newlines,
so this file runs correctly even if every line ending was rewritten. The
verifier is therefore immune to the failure mode it checks for.

Checks run in order, cheapest-and-nastiest first.
"""

import hashlib
import os
import subprocess
import sys


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _banner(n, title):
    print("\n" + "=" * 66)
    print(" %d. %s" % (n, title))
    print("=" * 66)


def check_integrity():
    _banner(1, "INTEGRITY -- did every byte survive the transfer?")
    if not os.path.isfile("CHECKSUMS.sha256"):
        print("FAIL: CHECKSUMS.sha256 not found; are you inside the bundle?")
        return False
    bad = missing = 0
    with open("CHECKSUMS.sha256") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            want, path = line.split("  ", 1)
            if not os.path.isfile(path):
                print("MISSING  ", path)
                missing += 1
                continue
            if _sha256(path) != want:
                print("MISMATCH ", path)
                bad += 1
    if bad or missing:
        print("FAIL: %d corrupted, %d missing -- re-transfer in BINARY mode"
              % (bad, missing))
        return False
    print("OK: all checksums match")
    return True


# Scope for the ENCODING and SYNTAX checks. Deliberately excludes
# source/195667-master: that is the upstream ModelDB release, it contains
# Python 2 code from 2016 that will never compile under python3, and we
# never execute it -- it is data. Its bytes are still covered by the
# checksum check, which is what actually matters for it.
OURS = ("tools", "verify_on_cluster.py")


def _walk(exts, scope=None):
    roots = ["."] if scope is None else [s for s in scope
                                         if os.path.isdir(s)]
    for r in roots:
        for root, dirs, files in os.walk(r):
            dirs[:] = [d for d in dirs
                       if d not in ("__pycache__", ".git")]
            for f in files:
                if f.endswith(exts):
                    yield os.path.join(root, f)
    if scope is not None:
        for s in scope:
            if os.path.isfile(s) and s.endswith(exts):
                yield s


def check_line_endings():
    _banner(2, "LINE ENDINGS -- CRLF in anything with a shebang")
    bad = []
    for p in _walk((".sh", ".pbs", ".slurm")):
        n = open(p, "rb").read().count(b"\r")
        if n:
            bad.append((p, n))
            print("CRLF", p, "(%d CR bytes)" % n)
    if bad:
        print("FAIL: %d file(s). Fix with:  sed -i 's/\\r$//' <file>"
              % len(bad))
        return False
    print("OK: every shebang'd file is LF-only")
    return True


def check_encoding():
    _banner(3, "ENCODING -- non-ASCII bytes in our Python source")
    bad = []
    for p in _walk((".py",), scope=OURS):
        b = [hex(c) for c in open(p, "rb").read() if c > 127]
        if b:
            bad.append(p)
            print("NON-ASCII", p, b[:6])
    if bad:
        print("FAIL: %d file(s)" % len(bad))
        return False
    print("OK: every .py file is pure ASCII")
    return True


def check_syntax():
    _banner(4, "SYNTAX -- our whole import chain")
    files = sorted(_walk((".py",), scope=OURS))
    r = subprocess.run([sys.executable, "-m", "py_compile"] + files)
    if r.returncode:
        print("FAIL: something does not compile")
        return False
    print("OK: %d file(s) compile" % len(files))
    return True


def check_smoke():
    _banner(5, "ARCHIVE SMOKE TESTS (NEURON-free, offline)")
    if not os.path.isdir("source/195667-master"):
        print("SKIP: source/195667-master not bundled")
        return True
    r = subprocess.run(
        [sys.executable, "smoke_eyal_archive_builder.py",
         "--eyal-root", os.path.join("..", "source", "195667-master"),
         "--out-root", os.path.join("/tmp", "eyal_smoke_%d" % os.getpid())],
        cwd="tools")
    if r.returncode:
        print("FAIL: archive smoke tests did not pass")
        return False
    return True


def main():
    print("=" * 66)
    print(" verify_on_cluster.py")
    print("=" * 66)
    print("python3 :", sys.version.split()[0])
    print("conda   :", os.environ.get("CONDA_DEFAULT_ENV", "none"))
    print("cwd     :", os.getcwd())

    ok = True
    # Integrity first: if bytes are wrong, every later result is meaningless.
    if not check_integrity():
        ok = False
    for fn in (check_line_endings, check_encoding, check_syntax, check_smoke):
        if not fn():
            ok = False

    print("\n" + "=" * 66)
    print(" ALL CHECKS PASSED -- safe to proceed to the patch step" if ok
          else " SOMETHING FAILED -- do NOT qsub until it is resolved")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
'''


README = """# Eyal et al. (2016) -> Towards-EEG passive fit: offline HPC bundle

Built {created} in Colab. Everything needed on an internet-restricted
cluster is inside this bundle; nothing here downloads anything at run time.

## Contents

    archive/eyal_archive/     Phase 0 archive, 6 specimens, ready to fit
    source/195667-master/     ModelDB 195667 (Eyal 2016) -- needed only so
                              smoke tests 1-11 can be re-run offline
    tools/                    builder, tests, and the pipeline patcher
    CHECKSUMS.sha256          SHA-256 of every file, computed BEFORE transfer
    verify_on_cluster.py      run this first (Python, not shell, on purpose:
                              a .sh verifier cannot survive the CRLF
                              corruption it exists to detect)

## Procedure on the cluster

Transfer `eyal_hpc_bundle.tar.gz` in BINARY mode (SFTP binary, or scp).
Never paste it as text.

    tar -xzf eyal_hpc_bundle.tar.gz
    cd {bundle}
    python3 verify_on_cluster.py

Only if that prints ALL CHECKS PASSED:

    # 1. patch the pipeline (guarded; Allen behaviour is unchanged)
    python3 tools/patch_eyal_support.py --code-dir <Biological Fit> --check
    python3 tools/patch_eyal_support.py --code-dir <Biological Fit> --apply

    # 2. prove Allen is unaffected
    python3 tools/regression_allen_unchanged.py --code-dir <Biological Fit>

    # 3. confirm the patched loader handles the archive
    cd tools
    python3 smoke_eyal_archive_builder.py \\
        --eyal-root ../source/195667-master \\
        --out-root /tmp/eyal_smoke \\
        --monolith-dir <Biological Fit>
    cd ..

    # 4. NEURON build check (needs NEURON on the cluster)
    cd tools
    python3 smoke_eyal_neuron_build.py --archive-root ../archive/eyal_archive
    cd ..

Expected: step 2 reports FAILURES: 0; step 3 reports pass=13 fail=0;
step 4 reports pass=6 fail=0.

## Then run the fit

Point --archive-dir at `archive/eyal_archive`. Non-default flags:

    --F 1.9
    --ss-window-ms 3.0,102.0
    --n-long-train 0
    --axon-replacement none
    --skip-phase2p5
    --phase3-subset none

plus `require_long_square=False` and `group_ss_by_amplitude=True` on the
archive loader.

## Known gaps (carry these into any write-up)

* No Long Square data exists in this dataset. Held-out validation rests on
  the opposite-polarity SS bundles, which exist for cell 0603_cell08 only.
* No individual (pre-average) sweeps: the Phase 3 nonparametric bootstrap
  is degenerate. Use --phase3-subset none.
* rin_MOhm and tau_ms in metadata.json are DERIVED from the published
  triplet theta*. They are useful for orientation but CIRCULAR as validation
  gates. Only v_rest_mV = -86.0 is independent.
* PassiveCell discretises these morphologies to nseg=1 (127 segments for
  cell08) where Eyal's own rule gives 738. Measure the effect on Cm before
  trusting the comparison.
"""


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="/content",
                    help="where Save_Eyal2016_Data.py put things")
    ap.add_argument("--drive-out",
                    default="/content/drive/MyDrive/Eyal Data",
                    help="Drive folder to write the bundle into")
    ap.add_argument("--tools-dir", default=None,
                    help="where the .py tools live (default: --work)")
    args = ap.parse_args(argv)

    work = Path(args.work)
    tools_src = Path(args.tools_dir) if args.tools_dir else work
    drive_out = Path(args.drive_out)

    archive = work / "eyal_archive"
    source = work / "195667-master"
    if not archive.is_dir():
        print("missing %s -- run Save_Eyal2016_Data.py first" % archive)
        return 1
    if not source.is_dir():
        print("missing %s -- run Cell 3 of the driver first" % source)
        return 1

    staging = work / BUNDLE_NAME
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "tools").mkdir(parents=True)
    (staging / "archive").mkdir(parents=True)
    (staging / "source").mkdir(parents=True)

    # -- contents -----------------------------------------------------------
    shutil.copytree(archive, staging / "archive" / "eyal_archive")
    shutil.copytree(source, staging / "source" / "195667-master")
    missing = []
    for name in TOOLS:
        src = tools_src / name
        if src.is_file():
            shutil.copy2(src, staging / "tools" / name)
        else:
            missing.append(name)
    if missing:
        print("WARNING: tools not found and therefore NOT bundled: %s"
              % missing)
        print("         the cluster will not be able to run them offline.")

    for name in SCRIPTS:
        src = tools_src / name
        if src.is_file():
            shutil.copy2(src, staging / name)
        else:
            print("WARNING: %s not found; not bundled" % name)

    (staging / "verify_on_cluster.py").write_text(
        VERIFY_PY, encoding="ascii", newline="\n")
    (staging / "README_HPC.md").write_text(
        README.format(created=datetime.now().isoformat(timespec="seconds"),
                      bundle=BUNDLE_NAME),
        encoding="ascii", newline="\n")

    # -- pre-flight byte scan, BEFORE checksumming --------------------------
    bad_ascii, bad_crlf = scan_text_files(staging)
    if bad_ascii or bad_crlf:
        print("REFUSING TO PACK -- the bundle is already unclean:")
        for p in bad_ascii:
            print("  non-ASCII .py :", p)
        for p in bad_crlf:
            print("  CRLF shebang  :", p)
        return 1
    print("byte scan OK: every .py pure ASCII, every .sh LF-only")

    # -- checksums ----------------------------------------------------------
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    lines = ["%s  %s" % (sha256_file(p), p.relative_to(staging).as_posix())
             for p in files]
    (staging / "CHECKSUMS.sha256").write_text(
        "\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print("checksummed %d file(s)" % len(files))

    # -- pack ---------------------------------------------------------------
    drive_out.mkdir(parents=True, exist_ok=True)
    tgz = drive_out / (BUNDLE_NAME + ".tar.gz")
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(staging, arcname=BUNDLE_NAME)
    digest = sha256_file(tgz)
    (drive_out / (BUNDLE_NAME + ".sha256")).write_text(
        "%s  %s\n" % (digest, tgz.name), encoding="ascii", newline="\n")

    print()
    print("=" * 66)
    print("bundle : %s" % tgz)
    print("size   : %.1f MB" % (tgz.stat().st_size / 1e6))
    print("sha256 : %s" % digest)
    print("=" * 66)
    print("""
On the cluster, after transferring in BINARY mode:

    sha256sum eyal_hpc_bundle.tar.gz      # must equal the sha256 above
    tar -xzf eyal_hpc_bundle.tar.gz
    cd %s
    python3 verify_on_cluster.py
""" % BUNDLE_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
