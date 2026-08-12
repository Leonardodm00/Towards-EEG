"""
Save_Eyal2016_Data.py -- Colab driver: Eyal et al. (2016) -> Phase 0 archive
(corrected: Drive bootstrap + working-directory handling)

This is the Eyal counterpart of 'Save_AllenInstitute_Data.ipynb'. Paste the
cells below into Colab in order (the '# %%' markers are cell breaks), or run
the whole file with `python3 Save_Eyal2016_Data.py`.

WHY CELL 0 EXISTS
-----------------
Mounting Google Drive makes files VISIBLE, not IMPORTABLE. Python imports
from sys.path, and Colab's default working directory is /content, so a
module sitting in Drive raises ModuleNotFoundError. Two separate things must
therefore be fixed before anything else runs:

  * sys.path  -- so that `import eyal_archive_builder` resolves
  * the CWD   -- because cells 6 and 8 launch subprocesses using BARE
                 FILENAMES ("smoke_eyal_archive_builder.py"), which are
                 resolved against the working directory, NOT sys.path.
                 Fixing only sys.path defers the failure by two cells.

Cell 0 copies the modules out of Drive onto local disk and chdir's there.
Copying rather than working in place is deliberate: local disk is far faster
than the Drive FUSE mount, __pycache__ is not written into your Drive
folder, and the space in "Colab Notebooks" stops being a quoting hazard.

os.chdir() is used instead of the %cd magic so that this file remains valid
Python and can be byte-compiled; likewise pip is invoked through subprocess
rather than the ! shell escape.

FILES REQUIRED ALONGSIDE THIS ONE (all five, same folder)
---------------------------------------------------------
    eyal_archive_builder.py        pure I/O, no NEURON
    eyal_reference_scalars.py      NEURON reference model
    smoke_eyal_archive_builder.py  NEURON-free tests
    smoke_eyal_neuron_build.py     NEURON tests
    Save_Eyal2016_Data.py          this driver

NOT needed in Colab: patch_eyal_support.py, regression_allen_unchanged.py,
patches_eyal_support.diff -- those patch the pipeline and belong wherever
the 'Biological Fit' checkout lives.

ASCII-only, LF-only by construction.
"""

# %% Cell 0 -- BOOTSTRAP: locate the modules, copy them local, chdir =========
#
# Run this FIRST, before any `import eyal_archive_builder`. Re-run it after
# editing any module in Drive (then reload, see the note at the end).

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---- edit this list if your modules live somewhere else -------------------
MODULE_SRC_CANDIDATES = [
    "/content/drive/MyDrive/Colab Notebooks/Eyal Data",
    "/content/eyal_adapter",
    "/content",
]

REQUIRED = [
    "eyal_archive_builder.py",
    "eyal_reference_scalars.py",
    "smoke_eyal_archive_builder.py",
    "smoke_eyal_neuron_build.py",
]

IN_COLAB = "google.colab" in sys.modules or Path("/content").is_dir()
WORK = Path("/content") if IN_COLAB else Path.cwd()

# ---- mount Drive if any candidate path needs it ---------------------------
if IN_COLAB and any(c.startswith("/content/drive")
                    for c in MODULE_SRC_CANDIDATES):
    if not Path("/content/drive/MyDrive").is_dir():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as exc:
            print("Drive mount skipped/failed: %s" % exc)


def _locate_modules():
    """First candidate directory holding ALL required modules."""
    for cand in MODULE_SRC_CANDIDATES:
        d = Path(cand)
        if d.is_dir() and all((d / f).is_file() for f in REQUIRED):
            return d
    return None


MODULE_SRC = _locate_modules()
if MODULE_SRC is None:
    print("Could not find the five modules. Looked in:")
    for c in MODULE_SRC_CANDIDATES:
        d = Path(c)
        if not d.is_dir():
            print("  %-55s (no such directory)" % c)
        else:
            miss = [f for f in REQUIRED if not (d / f).is_file()]
            print("  %-55s missing: %s" % (c, miss))
    raise SystemExit(
        "Add the correct folder to MODULE_SRC_CANDIDATES and re-run Cell 0.")

print("modules found in:", MODULE_SRC)

# ---- copy to local disk (no-op when already local) ------------------------
WORK.mkdir(parents=True, exist_ok=True)
for f in sorted(MODULE_SRC.glob("*.py")):
    dst = WORK / f.name
    try:
        if dst.exists() and dst.resolve() == f.resolve():
            continue
        shutil.copy2(f, dst)
    except shutil.SameFileError:
        pass

# ---- fix BOTH the working directory and sys.path --------------------------
os.chdir(WORK)
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

print("cwd            :", Path.cwd())
present = sorted(p.name for p in WORK.glob("*.py"))
print("modules present:", present)
_missing = [f for f in REQUIRED if f not in present]
assert not _missing, "still missing after copy: %s" % _missing

# ---- prove it imports -----------------------------------------------------
import eyal_archive_builder as eab                       # noqa: E402

print("import OK -- manifest has %d cells, %d traces"
      % (len(eab.MANIFEST), sum(len(e["traces"]) for e in eab.MANIFEST)))

# If you EDIT a module in Drive, re-run Cell 0 and then:
#     import importlib; importlib.reload(eab)
# A bare re-import silently reuses the already-loaded module object.


# %% Cell 1 -- NEURON ========================================================
# Needed by Cell 8 only. Installed via subprocess so this file stays valid
# Python (no ! escape). Skipped when NEURON is already importable.

try:
    import neuron                                        # noqa: F401
    HAVE_NEURON = True
except ImportError:
    print("installing NEURON (about a minute) ...")
    _r = subprocess.run([sys.executable, "-m", "pip", "install",
                         "--quiet", "neuron"])
    HAVE_NEURON = (_r.returncode == 0)

if HAVE_NEURON:
    import neuron
    print("NEURON", neuron.__version__)
else:
    print("NEURON unavailable -- Cell 8 will be skipped.")


# %% Cell 2 -- configuration =================================================
# Set SAVE_TO_DRIVE = True to leave a copy of the finished archive in Drive.

SAVE_TO_DRIVE = False

EYAL_ROOT = WORK / "195667-master"          # extracted ModelDB release
ARCHIVE_ROOT = WORK / "eyal_archive"        # what we build
SMOKE_ROOT = WORK / "eyal_archive_smoke"    # scratch archive for the tests
TARBALL = WORK / "eyal_archive.tar.gz"      # what we ship to the cluster

if SAVE_TO_DRIVE:
    DRIVE_OUT = Path("/content/drive/MyDrive/Eyal Data")
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
else:
    DRIVE_OUT = None

print("EYAL_ROOT    :", EYAL_ROOT)
print("ARCHIVE_ROOT :", ARCHIVE_ROOT)
print("DRIVE_OUT    :", DRIVE_OUT)


# %% Cell 3 -- fetch the ModelDB release =====================================
# ModelDB accession 195667 -- code and data accompanying
# Eyal et al. (2016) eLife 5:e16553, DOI 10.7554/eLife.16553

TARGZ = WORK / "eyal2016.tar.gz"
URL = ("https://codeload.github.com/ModelDBRepository/195667/"
       "tar.gz/refs/heads/master")

if not EYAL_ROOT.is_dir():
    subprocess.run(["curl", "-sL", URL, "-o", str(TARGZ)], check=True)
    subprocess.run(["tar", "-xzf", str(TARGZ), "-C", str(WORK)], check=True)

assert EYAL_ROOT.is_dir(), "extraction failed: %s" % EYAL_ROOT
n_dat = len(list((EYAL_ROOT / "Fig1").rglob("*.dat")))
n_asc = len(list((EYAL_ROOT / "morphs").glob("*.ASC")))
n_hoc = len(list((EYAL_ROOT / "PassiveModels").glob("model_*.hoc")))
print("traces=%d  morphologies=%d  passive models=%d" % (n_dat, n_asc, n_hoc))
assert (n_dat, n_asc, n_hoc) == (11, 6, 6), "unexpected release contents"


# %% Cell 4 -- validate the manifest BEFORE building anything ================
# The trace <-> morphology <-> onset mapping is the likeliest place for a
# silent transcription error, and a wrong pairing yields a plausible-looking
# but meaningless fit. Check it first, loudly.

eab.validate_manifest(EYAL_ROOT)
print("manifest OK -- %d cells, %d traces\n"
      % (len(eab.MANIFEST), sum(len(e["traces"]) for e in eab.MANIFEST)))

for e in eab.MANIFEST:
    r = e["reference_published"]
    print("  %-12s id=%-7d t_inj=%6.2f ms  %d trace(s)  "
          "Cm*=%.5f Rm*=%6.0f Ra*=%6.2f  tau_m*=%5.2f ms"
          % (e["cell_tag"], e["specimen_id"], e["t_inj_ms"], len(e["traces"]),
             r["cm_uF_per_cm2"], r["rm_Ohm_cm2"], r["ra_Ohm_cm"],
             r["cm_uF_per_cm2"] * r["rm_Ohm_cm2"] * 1e-3))


# %% Cell 5 -- build the archive =============================================

dirs = eab.build_eyal_archive(EYAL_ROOT, ARCHIVE_ROOT,
                              write_reference_scalars=True, verbose=True)

print()
for d in sorted(ARCHIVE_ROOT.glob("specimen_*")):
    print(" ", d.name, "->", sorted(p.name for p in d.iterdir()))
print("\ncomparison targets:")
print((ARCHIVE_ROOT / "comparison_targets.csv").read_text())


# %% Cell 6 -- NEURON-free smoke tests =======================================
# Run BEFORE the NEURON work: if the archive is malformed there is no point
# building a model from it. The subprocess uses a BARE FILENAME, which is
# exactly why Cell 0 had to chdir as well as touch sys.path.

rc = subprocess.run([sys.executable, "smoke_eyal_archive_builder.py",
                     "--eyal-root", str(EYAL_ROOT),
                     "--out-root", str(SMOKE_ROOT)]).returncode
print("\nsmoke (NEURON-free) exit code:", rc)
assert rc == 0, "archive smoke tests failed"
# Expected in Colab: pass=11 fail=0 skip=2 (tests 12-13 need MONOLITH_DIR).


# %% Cell 7 -- OPTIONAL: round-trip through the real pipeline loader =========
# Leave as None in Colab. Against an UNPATCHED pipeline, test 13 reports the
# amplitude-averaging defect (a bundle labelled +116.7 pA); with
# patch_eyal_support.py applied, both tests pass. This is more useful on the
# cluster, where the patched tree lives.
#
# To enable here:
#   subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
#                   "scikit-optimize"])
#   subprocess.run(["git", "clone", "--quiet",
#                   "https://github.com/Leonardodm00/Towards-EEG.git",
#                   str(WORK / "Towards-EEG")])
#   MONOLITH_DIR = (WORK / "Towards-EEG" / "Passive Features" /
#                   "HPC script" / "Biological Fit")

MONOLITH_DIR = None

if MONOLITH_DIR is not None:
    rc = subprocess.run([sys.executable, "smoke_eyal_archive_builder.py",
                         "--eyal-root", str(EYAL_ROOT),
                         "--out-root", str(SMOKE_ROOT),
                         "--monolith-dir", str(MONOLITH_DIR)]).returncode
    print("smoke (with loader) exit code:", rc)


# %% Cell 8 -- NEURON: import check + reference scalars + backfill ===========
# Builds each morphology in its OWN child process (NEURON's section namespace
# is process-global), asserts the .asc imports and the axon is deleted,
# measures Rin* implied by the published triplet, and merges the results into
# each metadata.json.

if HAVE_NEURON:
    rc = subprocess.run([sys.executable, "smoke_eyal_neuron_build.py",
                         "--archive-root", str(ARCHIVE_ROOT),
                         "--backfill"]).returncode
    print("\nsmoke (NEURON) exit code:", rc)
    assert rc == 0, "NEURON build tests failed"
else:
    print("skipped: NEURON not available")


# %% Cell 9 -- ship it =======================================================

eab.make_tarball(ARCHIVE_ROOT, TARBALL)

if DRIVE_OUT is not None:
    shutil.copy2(TARBALL, DRIVE_OUT / TARBALL.name)
    print("copied to Drive:", DRIVE_OUT / TARBALL.name)

print("""
Next steps on the cluster
-------------------------
  1. Transfer %s as a BINARY file (never paste it as text).
  2. tar -xzf eyal_archive.tar.gz
  3. Apply the pipeline patches, then verify:
       python3 patch_eyal_support.py --code-dir <Biological Fit> --check
       python3 patch_eyal_support.py --code-dir <Biological Fit> --apply
       python3 regression_allen_unchanged.py --code-dir <Biological Fit>
  4. Run the cluster-side verification block (line endings, encoding,
     py_compile, imports) before any qsub.
  5. Suggested first invocation once patched:
       --F 1.9 --ss-window-ms 3.0,102.0 --n-long-train 0
       --axon-replacement none --skip-phase2p5 --phase3-subset none
     plus require_long_square=False and group_ss_by_amplitude=True on the
     archive loader.
""" % TARBALL)
