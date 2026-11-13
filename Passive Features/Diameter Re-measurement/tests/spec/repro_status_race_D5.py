"""Reproducer (tester run D5, 2026-11-13): two survey tasks that finish at the
same time both rewrite <survey>/status.csv through the same temporary name,
status.csv.tmp (survey_io.write_status, called by survey_cells.run after each
cell). On davinci every task of the PBS array writes into the one survey
folder (Block 16.4 (b)); here two processes stand for two array tasks.

Run (from the workstream directory):
    python tests/spec/repro_status_race_D5.py
Prints how many of the writes raised (FileNotFoundError from os.replace, when
the other task moved status.csv.tmp first) and whether the final file holds
one row per cell of the two tasks.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from allen_diameter.loading import survey_io  # noqa: E402

STEPS = ("align", "nodes")


def task(out_dir, sid, n, q):
    errors = 0
    row = dict(specimen_id=sid, group="exc|3", align="done", nodes="done", error="")
    for _ in range(n):
        try:
            survey_io.write_status(out_dir, [row], STEPS)        # what survey_cells.run writes for its one cell
        except OSError as exc:                                    # FileNotFoundError: the other task moved the tmp
            errors += 1
            last = repr(exc)
    q.put((sid, errors, last if errors else ""))


def main(n=3000):
    out = tempfile.mkdtemp()
    q = mp.Queue()
    ps = [mp.Process(target=task, args=(out, sid, n, q)) for sid in (101, 102)]
    for p in ps:
        p.start()
    res = [q.get() for _ in ps]
    for p in ps:
        p.join()
    for sid, errors, last in sorted(res):
        print("task %d: %d of %d status writes raised%s" % (sid, errors, n, (": " + last) if last else ""))
    with open(os.path.join(out, "status.csv")) as f:
        lines = f.read().splitlines()
    print("final status.csv: %d data rows (the two tasks' cells: 2 expected by 16.2 (g)); rows: %s"
          % (len(lines) - 1, lines[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
