"""Block 11 amendment (F6; D-212.6, SPEC "The final check of wave 4"): colab_bootstrap.bootstrap takes <repo>/.git
that exists -- a directory in a clone, a file in a git worktree -- for an existing clone, and runs no `git clone`
there. Tester, 2026-11-22 (incremental run at 97f5104). The test builds a fake repository folder with a `.git`
file and checks that bootstrap(clone=True) does not call `git clone` on it (subprocess.run is replaced by a spy).

    python3 -m pytest -q -p no:cacheprovider tests/spec/test_spec_block11_colab_bootstrap.py

Pure ASCII.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import colab_bootstrap as CB      # noqa: E402


@pytest.mark.parametrize("kind", ["file", "dir", "none"])
def test_existing_git_entry_counts_as_a_clone(tmp_path, monkeypatch, kind):
    repo = tmp_path / "Towards-EEG"
    repo.mkdir()
    if kind == "file":
        (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    elif kind == "dir":
        (repo / ".git").mkdir()
    calls = []

    class Stop(Exception):
        pass

    def spy(cmd, *a, **kw):
        calls.append(list(cmd))
        raise Stop()            # stop at the first git call: what is checked is whether it is a clone

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr(CB.subprocess, "run", spy)
    if kind == "none":
        with pytest.raises(FileNotFoundError):
            CB.bootstrap(repo_dir=str(repo), clone=False)
    try:
        CB.bootstrap(repo_dir=str(repo), clone=True)
    except Stop:
        pass
    except Exception:
        pass                    # later steps of the bootstrap need a real clone: not under test here
    cloned = any(len(c) > 1 and c[0] == "git" and c[1] == "clone" for c in calls)
    assert cloned == (kind == "none"), calls
