#!/usr/bin/env python3
"""Will this campaign array fit the queue? Ask before submitting, not after.

  python3 campaign_queue_check.py --manifest ../h01/p1/manifests/L3_exc.csv
  python3 campaign_queue_check.py --manifest ... --shards 4
  python3 campaign_queue_check.py --manifest ... --hours-per-task 3.5
  python3 campaign_queue_check.py --rows 537 --shards 4 --queue cpu

Two different things are checked, and they fail differently. The QUEUE
LIMITS (from qstat) can REFUSE a submission or block it. The CPU BUDGET
(D-008: 200 CPUs at once, walltime ceiling 300 h) cannot refuse anything --
it decides how many subjobs run concurrently and therefore how long the
campaign takes. The ncpus and walltime per subjob are read from
campaign.pbs's own #PBS directives, so this check cannot drift from the
script it is checking. With --hours-per-task (from the SUBSET probe) the
budget becomes a wall-clock estimate.

Six P1 arrays already came close to this cluster's cap, and a campaign array
is N x SHARDS subjobs rather than N, so the first full-population submission
is the one that hits it. PBS rejects an oversized array at submission with a
terse message, which is survivable; the worse outcome is an accepted array
that starves every other job of the user's slots.

This reads the limits from the scheduler instead of guessing them: it runs

  qstat -Qf <queue>     the queue's own limits
  qstat -Bf             the server's (max_array_size lives here on PBS Pro)
  qstat -u <user>       what is already queued or running

unwraps PBS's 80-column line folding (a long value continues on the next
line after a TAB, with no field name repeated -- a naive grep truncates it),
collects every limit-shaped attribute it finds, and compares the ones it
recognises with N x SHARDS. Attributes it does not recognise are PRINTED
rather than ignored, because this script's list of names is not authoritative
for every PBS build.

Needs only the standard library, so it runs on the login node with no conda
env activated. It submits nothing.

  --from-queue FILE / --from-server FILE / --from-user FILE  read captured
  output instead of running qstat (used by the smoke test; also handy for
  pasting output from a node where this script is not installed).

Exit status: 0 the array fits every limit found, 1 a limit is exceeded,
2 no limit could be read at all (then the answer is unknown, not yes).

Pure ASCII, LF only.
"""

import argparse
import os
import re
import subprocess
import sys

CHECK_VERSION = "campaign_queue_check v1.1"

# Decision D-008 (2026-09-22) [user]: the allocation is 200 CPUs concurrently
# and a walltime ceiling of 300 h. These are DEFAULTS, overridable, and they
# are a throughput budget, not a rejection cap: a subjob that exceeds the CPU
# budget is not refused, it waits. Only the qstat limits below can refuse.
ALLOC_MAX_CPUS = 200
ALLOC_MAX_WALLTIME_H = 300.0

# Attribute names that cap how many jobs may exist. PBS Pro and Torque spell
# these differently, so both vocabularies are listed; whatever is present is
# used and the rest are simply absent.
ARRAY_SIZE_KEYS = ("max_array_size",)
QUEUED_KEYS = ("max_queued", "max_user_queued", "max_queuable",
               "max_user_queuable", "queued_jobs_threshold")
RUN_KEYS = ("max_run", "max_user_run", "max_running")
LIMIT_RE = re.compile(r"^(max_|.*_max$|queued_jobs_threshold)", re.I)


# --------------------------------------------------------------------- #
# reading the scheduler                                                 #
# --------------------------------------------------------------------- #
def unwrap(text):
    """PBS folds long values at ~80 columns, continuing after a newline+TAB.
    Rejoin those before parsing, exactly as the paths reference's qstat
    recipe does."""
    return text.replace("\n\t", "")


def parse_attrs(text):
    """{attribute: value} from `qstat -Qf` / `-Bf` output."""
    attrs = {}
    for line in unwrap(text).splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and not k.startswith("#"):
            attrs[k] = v
    return attrs


def as_int(value):
    """The leading integer of a PBS limit value, or None.

    Values come as plain integers, and on PBS Pro also as limit specs such as
    `[u:PBS_GENERIC=40]`; the number is what matters here."""
    if value is None:
        return None
    m = re.search(r"(-?\d+)", str(value))
    return int(m.group(1)) if m else None


def run(cmd):
    """(stdout, error). Never raises: a missing qstat is a reportable state,
    not a crash."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return "", "%s: %s" % (cmd[0], exc)
    if p.returncode != 0:
        return p.stdout, (p.stderr or "").strip() or "exit %d" % p.returncode
    return p.stdout, ""


def read_source(from_file, cmd):
    if from_file:
        if not os.path.isfile(from_file):
            raise SystemExit("--from-* file does not exist: %s" % from_file)
        with open(from_file) as fh:
            return fh.read(), ""
    return run(cmd)


def count_user_jobs(text):
    """Rows in `qstat -u <user>` that are jobs, not header or rule lines."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("-") or s.lower().startswith("job id"):
            continue
        if s.split()[0][0].isdigit():
            n += 1
    return n


def parse_pbs_script(path):
    """{ncpus, mem, walltime_h} from a job script's own #PBS -l directives.

    Read rather than assumed, so this check cannot drift from the script it
    is checking: campaign.pbs asks for `select=1:ncpus=2:mem=16gb` and a
    walltime, and those two decide how many subjobs fit a CPU budget and
    whether the request is inside the allocation's ceiling."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("#PBS"):
                continue
            m = re.search(r"ncpus=(\d+)", s)
            if m:
                out["ncpus"] = int(m.group(1))
            m = re.search(r"mem=(\S+?)(?::|\s|$)", s)
            if m and "mem" not in out:
                out["mem"] = m.group(1)
            m = re.search(r"walltime=(\d+):(\d+):(\d+)", s)
            if m:
                h, mi, sec = (int(g) for g in m.groups())
                out["walltime_h"] = h + mi / 60.0 + sec / 3600.0
    return out


def budget_view(subjobs, ncpus_per_task, max_cpus, walltime_h, max_walltime_h,
                hours_per_task=None):
    """Throughput under a CPU allocation. Nothing here can refuse a job; it
    says how long the campaign takes and whether the walltime request is
    inside the ceiling."""
    conc = max(1, int(max_cpus) // max(1, int(ncpus_per_task)))
    out = {"ncpus_per_task": int(ncpus_per_task), "max_cpus": int(max_cpus),
           "concurrency": conc, "subjobs": int(subjobs),
           "waves": int((int(subjobs) + conc - 1) // conc),
           "walltime_h": walltime_h, "max_walltime_h": float(max_walltime_h)}
    out["walltime_ok"] = (walltime_h is None
                          or float(walltime_h) <= float(max_walltime_h))
    if hours_per_task is not None:
        out["hours_per_task"] = float(hours_per_task)
        out["elapsed_h_estimate"] = out["waves"] * float(hours_per_task)
        out["cpu_hours_estimate"] = (int(subjobs) * float(hours_per_task)
                                     * int(ncpus_per_task))
        out["fits_task_walltime"] = (walltime_h is None
                                     or float(hours_per_task) <= float(walltime_h))
    return out


def print_budget(b):
    print("  CPU BUDGET (D-008: %d CPUs, walltime ceiling %.0f h)"
          % (b["max_cpus"], b["max_walltime_h"]))
    print("    %d CPU(s) per subjob -> %d subjob(s) run at once; %d subjobs "
          "is %d wave(s)" % (b["ncpus_per_task"], b["concurrency"],
                             b["subjobs"], b["waves"]))
    if b.get("walltime_h") is not None:
        mark = "ok" if b["walltime_ok"] else "OVER THE CEILING"
        print("    the job script asks for %.2f h per subjob -- %s"
              % (b["walltime_h"], mark))
    if "elapsed_h_estimate" in b:
        print("    at %.2f h per cell: about %.1f h wall clock (%.0f CPU-hours)"
              % (b["hours_per_task"], b["elapsed_h_estimate"],
                 b["cpu_hours_estimate"]))
        if not b.get("fits_task_walltime", True):
            print("    WARNING: %.2f h per cell exceeds the %.2f h the script "
                  "requests; raise it with  qsub -l walltime=HH:MM:SS  (up to "
                  "%.0f h)" % (b["hours_per_task"], b["walltime_h"],
                               b["max_walltime_h"]))
    print("    exceeding the CPU budget does not refuse a subjob, it queues it.")


def manifest_rows(path):
    """Non-empty data lines, counted as pandas counts them (and as
    campaign.pbs's own awk does): header excluded, blank lines skipped."""
    n = 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            if line.replace("\r", "").strip():
                n += 1
    return n


# --------------------------------------------------------------------- #
# the verdict -- pure, no I/O                                           #
# --------------------------------------------------------------------- #
def limits_from(qattrs, battrs):
    """{kind: (name, value, where)} for every cap recognised."""
    found = {}
    for kind, keys in (("array_size", ARRAY_SIZE_KEYS),
                       ("queued", QUEUED_KEYS), ("run", RUN_KEYS)):
        for where, attrs in (("queue", qattrs), ("server", battrs)):
            for k in keys:
                v = as_int(attrs.get(k))
                if v is not None and kind not in found:
                    found[kind] = (k, v, where)
    return found


def verdict(n_rows, shards, limits, in_flight):
    """Does an array of n_rows*shards subjobs fit? Returns a dict; the caller
    prints it."""
    total = int(n_rows) * int(shards)
    out = {"n_rows": int(n_rows), "shards": int(shards), "subjobs": total,
           "in_flight": in_flight, "breaches": [], "checked": []}
    for kind, (name, cap, where) in sorted(limits.items()):
        used = total if kind == "array_size" else total + (in_flight or 0)
        out["checked"].append((kind, name, cap, where, used))
        if used > cap:
            out["breaches"].append((kind, name, cap, where, used))
    out["ok"] = not out["breaches"]
    # The largest whole number of cells that fits every cap, so the advice is
    # a submission rather than a complaint.
    caps = [cap for _, (_, cap, _) in limits.items()]
    if caps:
        # The tightest cap, less what is already in flight -- except that a
        # max_array_size cap is per array, not per user, so nothing in flight
        # counts against it. Conservative on purpose: "at most", not "exactly".
        counts_in_flight = any(kind != "array_size" for kind in limits)
        room = min(caps) - ((in_flight or 0) if counts_in_flight else 0)
        out["max_cells_per_submission"] = max(0, int(room) // max(1, int(shards)))
    return out


def print_verdict(v, limits, notes):
    print("%s | %d manifest rows x %d shard(s) = %d subjobs; %s already queued "
          "or running" % (CHECK_VERSION, v["n_rows"], v["shards"], v["subjobs"],
                          v["in_flight"] if v["in_flight"] is not None else "?"))
    for n in notes:
        print("  NOTE: %s" % n)
    if not limits:
        print("  NO LIMIT FOUND in qstat output. The answer is UNKNOWN, not yes.")
        print("  Read it by hand before submitting:")
        print("    qstat -Qf cpu | sed ':a;N;$!ba;s/\\n\\t//g' | grep -i max")
        print("    qstat -Bf     | sed ':a;N;$!ba;s/\\n\\t//g' | grep -i max")
        return
    for kind, name, cap, where, used in v["checked"]:
        mark = "FAIL" if used > cap else "ok  "
        print("  [%s] %-18s %-22s cap %-8d (%s) vs %d" % (mark, kind, name, cap,
                                                          where, used))
    if v["ok"]:
        print("  FITS: submit with -J 0-%d" % (v["subjobs"] - 1))
    else:
        print("  DOES NOT FIT. Either lower SHARDS, or split the population:")
        k = v.get("max_cells_per_submission", 0)
        if k > 0:
            print("    at most %d cell(s) per submission with SHARDS=%d, i.e. "
                  "-J 0-%d at a time" % (k, v["shards"], k * v["shards"] - 1))
            print("    (campaign.pbs takes the manifest whole; make per-chunk "
                  "manifests, or submit the same manifest with a narrower -J "
                  "range and repeat -- P1 and P2 both resume, so the ranges "
                  "may overlap harmlessly)")
        else:
            print("    nothing fits at this SHARDS; lower it.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="the manifest whose rows are the cells")
    g.add_argument("--rows", type=int, help="cell count, if no manifest to hand")
    p.add_argument("--shards", type=int, default=1, help="SHARDS (default 1)")
    p.add_argument("--queue", default="cpu", help="queue name (default cpu)")
    p.add_argument("--user", default=None, help="default $USER")
    p.add_argument("--from-queue", default=None, help="captured qstat -Qf output")
    p.add_argument("--from-server", default=None, help="captured qstat -Bf output")
    p.add_argument("--from-user", default=None, help="captured qstat -u output")
    p.add_argument("--max-cpus", type=int, default=ALLOC_MAX_CPUS,
                   help="CPUs usable at once (default %d, D-008)" % ALLOC_MAX_CPUS)
    p.add_argument("--max-walltime-h", type=float, default=ALLOC_MAX_WALLTIME_H,
                   help="walltime ceiling in hours (default %.0f, D-008)"
                        % ALLOC_MAX_WALLTIME_H)
    p.add_argument("--ncpus-per-task", type=int, default=None,
                   help="default: read from --pbs-script")
    p.add_argument("--pbs-script", default=None,
                   help="job script whose #PBS -l directives set ncpus and "
                        "walltime (default campaign.pbs beside this file)")
    p.add_argument("--hours-per-task", type=float, default=None,
                   help="measured hours for one cell; turns the budget into a "
                        "wall-clock estimate for the campaign")
    p.add_argument("--show-all", action="store_true",
                   help="print every limit-shaped attribute found, recognised or not")
    args = p.parse_args(argv)

    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")
    n_rows = (manifest_rows(args.manifest) if args.manifest else int(args.rows))
    if n_rows < 1:
        raise SystemExit("no cells: %s has no data rows"
                         % (args.manifest or "--rows"))
    user = args.user or os.environ.get("USER") or os.environ.get("LOGNAME") or ""

    notes = []
    qtext, qerr = read_source(args.from_queue, ["qstat", "-Qf", args.queue])
    if qerr:
        notes.append("qstat -Qf %s: %s" % (args.queue, qerr))
    btext, berr = read_source(args.from_server, ["qstat", "-Bf"])
    if berr:
        notes.append("qstat -Bf: %s" % berr)
    utext, uerr = read_source(args.from_user, ["qstat", "-u", user])
    if uerr:
        notes.append("qstat -u %s: %s" % (user, uerr))

    qattrs, battrs = parse_attrs(qtext), parse_attrs(btext)
    in_flight = count_user_jobs(utext) if not uerr else None
    limits = limits_from(qattrs, battrs)
    v = verdict(n_rows, args.shards, limits, in_flight)
    print_verdict(v, limits, notes)

    script = args.pbs_script or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "campaign.pbs")
    spec = parse_pbs_script(script)
    if not spec and args.pbs_script:
        notes.append("could not read #PBS directives from %s" % script)
    ncpus = args.ncpus_per_task or spec.get("ncpus") or 1
    print()
    print_budget(budget_view(v["subjobs"], ncpus, args.max_cpus,
                             spec.get("walltime_h"), args.max_walltime_h,
                             args.hours_per_task))

    if args.show_all:
        print("\n  every limit-shaped attribute found:")
        for where, attrs in (("queue", qattrs), ("server", battrs)):
            for k in sorted(attrs):
                if LIMIT_RE.match(k):
                    print("    %-8s %-28s %s" % (where, k, attrs[k]))

    if not limits:
        return 2
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
