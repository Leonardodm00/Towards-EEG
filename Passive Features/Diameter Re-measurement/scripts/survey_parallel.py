#!/usr/bin/env python3
"""Run scripts/survey_cells.py over a selection with several worker processes (Block 63; D-283, D-284).

The cells of the selection (or of the --cells subset) whose requested steps are not all done are split round-robin
among --workers processes; worker k runs

    python survey_cells.py run --selection S --out-dir O --cache-dir C --cache-per-cell --steps T
                               [--config-json J] [--include-set-aside] --cells <its cells>

with its output in <out-dir>/logs/parallel_<UTC stamp>_w<k>.log. Each cell has its own folder under --out-dir and its
own crop cache under --cache-dir, so the workers share no file but status.csv, which each process writes through a
temporary of its own (survey_io.write_status). A finished step leaves a marker, so the run is resumable: started
again after an interruption, it skips the finished steps and gives the workers only the cells left.

Every --poll-s seconds one line: of the cells asked for, those with every requested step done, set aside at the
alignment (D-085.2; their later steps do not run without --include-set-aside), failed (a step without its marker
and an errors.log in the cell's folder), and the rest; and the workers still running. Ctrl-C, or SIGTERM (stopping
a Colab cell), stops the workers. Exit code: 0 when every worker exited 0, 1 when one did not, 2 on bad arguments,
130 when stopped.

    python scripts/survey_parallel.py --workers 4 --selection SEL --out-dir OUT --cache-dir CACHE \\
        --steps align,nodes,calibration,screening --config-json CFG [--cells ID,ID] [--poll-s 300]
"""
from __future__ import annotations

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
if os.path.join(WS, "src") not in sys.path:
    sys.path.insert(0, os.path.join(WS, "src"))

from allen_diameter.loading import survey_io  # noqa: E402

STATES = ("done", "set_aside", "failed", "pending")


def split(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def cell_state(out_dir, specimen_id, steps):
    """One of STATES for a cell of the survey in out_dir: 'done' when every step of `steps` has its marker;
    'set_aside' when its alignment set it aside (cell.json); 'failed' when a step lacks its marker and the cell's
    errors.log exists; 'pending' otherwise (not started, or running)."""
    cell = survey_io.CellDir(out_dir, specimen_id)
    if all(cell.done(s) for s in steps):
        return "done"
    if ((cell.context() or {}).get("alignment") or {}).get("set_aside"):
        return "set_aside"
    if os.path.exists(os.path.join(cell.path, "errors.log")):
        return "failed"
    return "pending"


def partition(cells, n_workers):
    """Round-robin: worker k gets the cells k, k + W, k + 2W, ... of `cells`; no empty group."""
    groups = [list(cells[k::n_workers]) for k in range(max(1, int(n_workers)))]
    return [g for g in groups if g]


def counts(out_dir, cells, steps):
    out = dict((s, 0) for s in STATES)
    for c in cells:
        out[cell_state(out_dir, c, steps)] += 1
    return out


def _say(t0, out_dir, cells, steps, procs):
    n = counts(out_dir, cells, steps)
    running = sum(1 for p in procs if p["proc"].poll() is None)
    print("[parallel] %5.2f h | of %d cells: %d done, %d set aside, %d failed, %d to go | workers running %d/%d"
          % ((time.time() - t0) / 3600.0, len(cells), n["done"], n["set_aside"], n["failed"], n["pending"],
             running, len(procs)), flush=True)
    return n


class _Stop(Exception):
    pass


def _on_term(signum, frame):            # stopping a Colab cell terminates this process: stop the workers too
    raise _Stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--steps", required=True, help="comma-separated, as survey_cells.py run --steps")
    ap.add_argument("--config-json", default="")
    ap.add_argument("--cells", default="", help="comma-separated specimen ids of the selection (default: all)")
    ap.add_argument("--include-set-aside", action="store_true")
    ap.add_argument("--poll-s", type=float, default=300.0, help="seconds between progress lines")
    ap.add_argument("--runner", default=os.path.join(HERE, "survey_cells.py"),
                    help="the survey runner (default: scripts/survey_cells.py; the smoke test passes a stub)")
    a = ap.parse_args(argv)
    steps = split(a.steps)
    if a.workers < 1 or not steps or a.poll_s <= 0:
        print("[parallel] --workers >= 1, --steps non-empty and --poll-s > 0 are required", file=sys.stderr)
        return 2
    sel = [int(r["specimen_id"]) for r in survey_io.read_selection(a.selection)]
    wanted = [int(c) for c in split(a.cells)]
    if wanted:
        missing = sorted(set(wanted) - set(sel))
        if missing:
            print("[parallel] not in the selection %s: %s" % (a.selection, ",".join(str(m) for m in missing)),
                  file=sys.stderr)
            return 2
        keep = set(wanted)
        cells = [c for c in sel if c in keep]
    else:
        cells = sel
    states = dict((c, cell_state(a.out_dir, c, steps)) for c in cells)
    todo = [c for c in cells if states[c] in ("pending", "failed")
            or (states[c] == "set_aside" and a.include_set_aside)]
    t0 = time.time()
    if not todo:
        _say(t0, a.out_dir, cells, steps, [])
        print("[parallel] nothing to do: every cell asked for is done or set aside", flush=True)
        return 0
    groups = partition(todo, a.workers)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = os.path.join(a.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    base = [sys.executable, "-u", a.runner, "run", "--selection", a.selection, "--out-dir", a.out_dir,
            "--cache-dir", a.cache_dir, "--cache-per-cell", "--steps", ",".join(steps)]
    if a.config_json:
        base += ["--config-json", a.config_json]
    if a.include_set_aside:
        base += ["--include-set-aside"]
    old_term = signal.signal(signal.SIGTERM, _on_term)
    procs = []
    try:
        for k, g in enumerate(groups):
            path = os.path.join(log_dir, "parallel_%s_w%d.log" % (stamp, k))
            f = open(path, "w")
            p = subprocess.Popen(base + ["--cells", ",".join(str(c) for c in g)], stdout=f,
                                 stderr=subprocess.STDOUT, cwd=WS)
            procs.append(dict(proc=p, log=f, path=path, cells=g))
            print("[parallel] worker %d (pid %d): %d cells, %s%s -> %s" % (
                k, p.pid, len(g), ",".join(str(c) for c in g[:4]), ",..." if len(g) > 4 else "", path), flush=True)
        _say(t0, a.out_dir, cells, steps, procs)
        last = time.time()
        while any(d["proc"].poll() is None for d in procs):
            time.sleep(min(1.0, a.poll_s))
            if time.time() - last >= a.poll_s:
                _say(t0, a.out_dir, cells, steps, procs)
                last = time.time()
    except (KeyboardInterrupt, _Stop):
        for d in procs:
            if d["proc"].poll() is None:
                d["proc"].terminate()
        deadline = time.time() + 30.0
        for d in procs:
            try:
                d["proc"].wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                d["proc"].kill()
                d["proc"].wait()
        _say(t0, a.out_dir, cells, steps, procs)
        print("[parallel] stopped: workers terminated; run again to resume (finished steps are skipped)", flush=True)
        return 130
    finally:
        signal.signal(signal.SIGTERM, old_term)
        for d in procs:
            d["log"].close()
    n = _say(t0, a.out_dir, cells, steps, procs)
    bad = [(k, d["proc"].returncode, d["path"]) for k, d in enumerate(procs) if d["proc"].returncode != 0]
    for k, rc, path in bad:
        print("[parallel] worker %d exited %d: see %s" % (k, rc, path), flush=True)
    print("[parallel] finished: %d done, %d set aside, %d failed, %d not done; %d of %d workers exited 0"
          % (n["done"], n["set_aside"], n["failed"], n["pending"], len(procs) - len(bad), len(procs)), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
