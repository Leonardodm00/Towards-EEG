#!/usr/bin/env python3
"""Colab bootstrap for the diameter pipeline -- Block 11 in specs/SPEC.md
(impl-handoff "Colab bootstrap"; D-020, D-022).

The repository is public, so Colab clones it without a token; nothing is
copied to Drive (copies on Drive are how stale module versions happened).
Drive keeps only data: the HttpFetcher cache, outputs, tables.

First Colab cell (the clone has to exist before this file can run):

    import os, subprocess, sys
    REPO, BRANCH = "/content/Towards-EEG", "sci/diameter-pipeline"
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--depth", "50", "-b", BRANCH,
                        "https://github.com/Leonardodm00/Towards-EEG.git", REPO], check=True)
    sys.path.insert(0, os.path.join(REPO, "Passive Features", "Diameter Re-measurement", "scripts"))
    import colab_bootstrap
    ENV = colab_bootstrap.bootstrap(REPO, BRANCH)      # pulls, sets sys.path, prints versions and the commit

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

REPO_URL = "https://github.com/Leonardodm00/Towards-EEG.git"
WORKSTREAM = os.path.join("Passive Features", "Diameter Re-measurement")
PACKAGES = ("numpy", "scipy", "pandas", "PIL", "matplotlib", "requests")


def bootstrap(repo_dir="/content/Towards-EEG", branch="sci/diameter-pipeline", url=REPO_URL, pull=True, clone=True,
              depth=50, verbose=True):
    """Clone (or fast-forward) the repository on `branch`, put the workstream's
    src/ and scripts/ on sys.path, import the packages the pipeline needs and
    report their versions and the commit. Returns a dict of paths and the
    commit. With clone=False a missing clone raises FileNotFoundError."""
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        if not clone:
            raise FileNotFoundError("no git clone at %s" % repo_dir)
        subprocess.run(["git", "clone", "--depth", str(depth), "-b", branch, url, repo_dir], check=True)
    elif pull:
        subprocess.run(["git", "-C", repo_dir, "fetch", "--depth", str(depth), "origin", branch], check=True)
        subprocess.run(["git", "-C", repo_dir, "checkout", branch], check=True)
        subprocess.run(["git", "-C", repo_dir, "merge", "--ff-only", "FETCH_HEAD"], check=True)
    ws = os.path.join(repo_dir, WORKSTREAM)
    src, scripts = os.path.join(ws, "src"), os.path.join(ws, "scripts")
    for p in (scripts, src):
        if p not in sys.path:
            sys.path.insert(0, p)
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "?")
        except ImportError:
            versions[name] = "MISSING"
    import allen_diameter  # noqa: F401  (the package resolves from src/)
    import allen_image_io  # noqa: F401
    commit = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    env = dict(repo=repo_dir, workstream=ws, src=src, scripts=scripts, commit=commit, branch=branch,
               python=sys.version.split()[0], versions=versions)
    if verbose:
        print("[diameter] commit %s on %s; python %s" % (commit, branch, env["python"]))
        print("[diameter] " + ", ".join("%s %s" % kv for kv in versions.items()))
        print("[diameter] src %s" % src)
    return env


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Clone or update the repository and set sys.path (Colab).")
    ap.add_argument("--repo-dir", default="/content/Towards-EEG")
    ap.add_argument("--branch", default="sci/diameter-pipeline")
    ap.add_argument("--no-pull", action="store_true")
    a = ap.parse_args(argv)
    bootstrap(a.repo_dir, a.branch, pull=not a.no_pull)
    return 0


if __name__ == "__main__":
    sys.exit(main())
