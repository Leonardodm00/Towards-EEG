"""Roadmap of a subproject: theory stages and implementation stages as two
connected graphs, with the implementation colours computed from the
repository.

One YAML file per subproject (specs/roadmap/<name>.yaml) is the source of
truth: the stages, their dependencies, and the theory statuses (set by
hand). The implementation statuses are never written by hand: this script
derives them from git, specs/SPEC.md, the integration-tester reports and
the HPC job markers on the hpc-status branch.

Usage
    python roadmap.py init   --name N --branch B [--repo OWNER/NAME]
                             [--spec specs/SPEC.md] [--root PATH]
    python roadmap.py check  --yaml specs/roadmap/N.yaml [--root PATH]
    python roadmap.py status --yaml specs/roadmap/N.yaml [--root PATH]
                             [--json OUT] [--no-fetch] [--source LABEL]
    python roadmap.py render --yaml specs/roadmap/N.yaml [--root PATH]
                             [--status STATUS.json] [--out specs/roadmap/N.md]
    python roadmap.py set    --yaml specs/roadmap/N.yaml STAGE STATUS
                             [--note TEXT] [--decision D-nnn] [--on DATE]

    --root is the repository root (default: the root of the repository that
    holds the YAML file, else the current directory).

Implementation states (one colour each; the page and the Mermaid rendering
take them from STATES below):
    planned   red      no code: the SPEC entry is drafted, or the modules
                       the entry names do not exist on the branch
    local     orange   code exists but is not on origin (uncommitted changes
                       or commits not pushed); only visible from a working
                       clone
    pushed    amber    on origin, no tester report covers the current code
    testing   blue     the night tester is running with this block in scope
    tested    teal     tester finished and found no BLOCKER/MAJOR; the HPC
                       run is still missing (stages with needs_hpc)
    running   violet   an HPC job of this stage is running (marker on the
                       hpc-status branch)
    failing   crimson  an open BLOCKER/MAJOR finding on the current code, or
                       the last HPC job failed, or its running marker is
                       stale
    done      green    pushed, tested, and (if needs_hpc) the HPC run
                       finished on the current code

Theory states: open (grey), studying (blue), closed (green). A stage whose
prerequisites are not satisfied (a theory stage not closed, an
implementation stage not done) is reported with blocked_by; the renderers
draw it with a dashed border.

Groups: a stage with `parent: <id>` is a member of that stage, to any
depth. A stage with members is drawn as a frame around them, coloured by
its least advanced member (failing < planned < local < pushed < testing <
tested < running < done; open < studying < closed), with its own node
inside when it has a block of its own. An implementation stage without a
block is a pure group: it has no state of its own and is not counted.

Exit codes: 0 on success, 1 on a validation error, 2 if git fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

SCHEMA_VERSION = 1

# ----------------------------------------------------------------- states ---

# (state, fill colour, text colour, meaning)
STATES = {
    "implementation": [
        ("planned", "#d62828", "#ffffff", "not implemented"),
        ("local", "#f77f00", "#1a1a1a", "implemented, not pushed"),
        ("pushed", "#e9c46a", "#1a1a1a", "pushed, not tested"),
        ("testing", "#3a86ff", "#ffffff", "independent tester running"),
        ("tested", "#2a9d8f", "#ffffff", "tester passed, HPC run missing"),
        ("running", "#8338ec", "#ffffff", "HPC job running"),
        ("failing", "#b5174e", "#ffffff", "tester or HPC failure on the current code"),
        ("done", "#2d9a3d", "#ffffff", "pushed, tested, HPC run done"),
    ],
    "theory": [
        ("open", "#8d99ae", "#1a1a1a", "not started"),
        ("studying", "#3a86ff", "#ffffff", "being studied"),
        ("closed", "#2d9a3d", "#ffffff", "closed"),
    ],
}
THEORY_STATES = [s[0] for s in STATES["theory"]]
IMPL_STATES = [s[0] for s in STATES["implementation"]]

BLOCK_HEADER = re.compile(r"^###\s+Block\s+(\d+)\s*:\s*(.*?)\s*$")
FIELD_LINE = re.compile(r"^-\s+([A-Za-z][A-Za-z /]*?)\s*:")
BACKTICKED = re.compile(r"`([^`]+)`")
BLOCK_REFS = re.compile(r"\bBlocks?\s+(\d+(?:\s*(?:,|and|&|-)\s*\d+)*)")
REPORT_COMMIT = re.compile(r"^Commit:\s*([0-9a-fA-F]{7,40})\b", re.MULTILINE)
REPORT_MODE = re.compile(r"^Mode:\s*(\w+)", re.MULTILINE)
REPORT_SCOPE = re.compile(r"^Scope:\s*(.*)$", re.MULTILINE)
REPORT_DATE = re.compile(r"^#.*?(\d{4}-\d{2}-\d{2})", re.MULTILINE)
VERDICT = re.compile(r"^##\s+Verdict\s*\n\s*(PASS WITH ISSUES|PASS|FAIL)",
                     re.MULTILINE)
STATUS_FIELD = re.compile(r"Run status:\s*(\w+)")
STATUS_COMMIT = re.compile(r"Commit under test:\s*([0-9a-fA-F]{7,40})")
STATUS_MODE = re.compile(r"Mode:\s*(\w+)")
STATUS_STARTED = re.compile(r"Started \(UTC\):\s*([0-9: -]+)")
RESOLVED_WORDS = re.compile(
    r"not a problem|resolved|fixed|closed|withdrawn|superseded|no longer",
    re.IGNORECASE)
HPC_WORDS = re.compile(r"davinci|cluster|\bHPC\b|\bPBS\b|qsub|sbatch|SLURM",
                       re.IGNORECASE)


class RoadmapError(RuntimeError):
    pass


class GitError(RoadmapError):
    pass


# -------------------------------------------------------------------- git ---

def git(root, *args, check=True):
    result = subprocess.run(["git", "-C", root, *args],
                            capture_output=True, text=True)
    if check and result.returncode != 0:
        raise GitError("git %s: %s" % (" ".join(args), result.stderr.strip()))
    return result.stdout


def git_ok(root, *args):
    return subprocess.run(["git", "-C", root, *args],
                          capture_output=True, text=True).returncode == 0


def resolve_rev(root, rev):
    out = subprocess.run(["git", "-C", root, "rev-parse", "--verify",
                          "--quiet", rev + "^{commit}"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def read_at(root, rev, path):
    out = subprocess.run(["git", "-C", root, "show", "%s:%s" % (rev, path)],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def is_ancestor(root, older, newer):
    return git_ok(root, "merge-base", "--is-ancestor", older, newer)


def tree_paths(root, rev):
    return git(root, "ls-tree", "-r", "--name-only", rev).splitlines()


def repo_root_of(path):
    out = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(path))
                          or ".", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ------------------------------------------------------------------- yaml ---

def need_yaml():
    if yaml is None:
        raise RoadmapError("PyYAML is required: pip install pyyaml")


def load_roadmap(path):
    need_yaml()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("theory", [])
    data.setdefault("implementation", [])
    for stage in data["theory"]:
        # a theory stage with members and no status of its own is a pure
        # group: its colour is its least advanced member's
        stage["status_given"] = bool(stage.get("status"))
        stage["status"] = stage.get("status") or "open"
        stage["depends_on"] = as_list(stage.get("depends_on"))
        stage["refs"] = as_list(stage.get("refs"))
        stage["summary"] = stage.get("summary") or ""
        stage["parent"] = stage.get("parent") or None
    for stage in data["implementation"]:
        stage["depends_on"] = as_list(stage.get("depends_on"))
        stage["needs_theory"] = as_list(stage.get("needs_theory"))
        stage["needs_hpc"] = bool(stage.get("needs_hpc"))
        stage["jobs"] = as_list(stage.get("jobs"))
        stage["summary"] = stage.get("summary") or ""
        stage["parent"] = stage.get("parent") or None
        blocks = stage.get("blocks")
        if blocks is None:
            block = stage.get("block")
            blocks = [] if block is None else [block]
        stage["blocks"] = [int(b) for b in blocks]
    data.setdefault("spec", "specs/SPEC.md")
    data.setdefault("hpc_branch", "hpc-status")
    data.setdefault("stale_after_hours", 48)
    return data


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def check_roadmap(data):
    """Return a list of problems (empty when the roadmap is consistent)."""
    problems = []
    for key in ("name", "branch"):
        if not data.get(key):
            problems.append("missing top-level field: %s" % key)
    theory = {s.get("id"): s for s in data["theory"]}
    impl = {s.get("id"): s for s in data["implementation"]}
    for kind, stages in (("theory", data["theory"]),
                         ("implementation", data["implementation"])):
        seen = set()
        for stage in stages:
            sid = stage.get("id")
            if not sid or not re.match(r"^[A-Za-z][A-Za-z0-9_.-]*$", str(sid)):
                problems.append("%s stage without a valid id: %r" % (kind, stage))
                continue
            if sid in seen:
                problems.append("duplicate id %s" % sid)
            seen.add(sid)
            if not stage.get("title"):
                problems.append("%s has no title" % sid)
            if kind == "theory" and stage["status"] not in THEORY_STATES:
                problems.append("%s: status %r not in %s"
                                % (sid, stage["status"], THEORY_STATES))
            if kind == "implementation":
                manual = stage.get("manual_status")
                if manual is not None and manual not in IMPL_STATES:
                    problems.append("%s: manual_status %r not in %s"
                                    % (sid, manual, IMPL_STATES))
                for ref in as_list(stage.get("needs_theory")):
                    if ref not in theory:
                        problems.append("%s needs_theory %s: no such theory stage"
                                        % (sid, ref))
            pool = theory if kind == "theory" else impl
            for ref in as_list(stage.get("depends_on")):
                if ref not in pool:
                    problems.append("%s depends_on %s: no such %s stage"
                                    % (sid, ref, kind))
                elif ref == sid:
                    problems.append("%s depends on itself" % sid)
            parent = stage.get("parent")
            if parent is not None:
                if parent not in pool:
                    problems.append("%s parent %s: no such %s stage" % (sid, parent, kind))
                elif parent == sid:
                    problems.append("%s is its own parent" % sid)
    if set(theory) & set(impl):
        problems.append("ids shared between theory and implementation: %s"
                        % sorted(set(theory) & set(impl)))
    for kind, pool in (("theory", theory), ("implementation", impl)):
        cycle = find_cycle({k: as_list(v.get("depends_on")) for k, v in pool.items()})
        if cycle:
            problems.append("%s dependencies form a cycle: %s" % (kind, " -> ".join(cycle)))
        cycle = find_cycle({k: ([v["parent"]] if v.get("parent") else [])
                            for k, v in pool.items()})
        if cycle:
            problems.append("%s parents form a cycle: %s" % (kind, " -> ".join(cycle)))
        for sid, stage in pool.items():
            parent = stage.get("parent")
            if parent in pool and parent in as_list(stage.get("depends_on")):
                problems.append("%s depends on its own group %s" % (sid, parent))
    return problems


def find_cycle(graph):
    state, stack = {}, []

    def visit(node):
        state[node] = 1
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt not in graph:
                continue
            if state.get(nxt) == 1:
                return stack[stack.index(nxt):] + [nxt]
            if nxt not in state:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in graph:
        if node not in state:
            found = visit(node)
            if found:
                return found
    return None


# ------------------------------------------------------------------- spec ---

def parse_spec(text):
    """{number: {"name": str, "lines": [...], "text": str}}."""
    blocks, current, field = {}, None, None
    for line in (text or "").splitlines():
        header = BLOCK_HEADER.match(line)
        if header:
            current = {"name": header.group(2), "lines": [], "raw": []}
            blocks[int(header.group(1))] = current
            field = None
            continue
        if line.startswith("#"):
            current, field = None, None
            continue
        if current is None:
            continue
        match = FIELD_LINE.match(line)
        if match:
            field = match.group(1).strip().lower()
        elif not line.strip() or not line.startswith((" ", "\t")):
            field = None
        current["lines"].append((field, line.rstrip()))
        current["raw"].append(line.rstrip())
    for block in blocks.values():
        block["text"] = "\n".join(block["raw"])
    return blocks


def field_text(block, *names):
    return " ".join(line.strip() for field, line in block["lines"]
                    if field in names)


SPEC_STATUS_WORDS = ("integration-tested", "smoke-tested", "implemented", "drafted",
                     "verified")
# longest alternatives first, so "smoke-tested" is one match, not "tested";
# "not implemented" does not count as implemented
SPEC_STATUS_RE = re.compile(r"(?<!not )(?<!un)\b(%s)\b" % "|".join(SPEC_STATUS_WORDS),
                            re.IGNORECASE)


def spec_status_word(block):
    """The block's current status word from its Status line.

    A Status line is a history ("drafted 2026-10-21; 14.1-14.6 implemented
    2026-10-21; gate 2 next"), so the LAST known status word in it is the
    current one. An entry can also hold other "Status:" lines (a fit's
    status field, say); words outside the known set are ignored when a
    known one exists.
    """
    text = field_text(block, "status")
    if not text:
        # no "- Status:" field: the status is embedded in another line
        # ("... and the runbook. Status: drafted 2026-10-21 ..."); take the
        # text after the last "Status:" of the entry
        holders = [line for line in block["raw"] if "Status:" in line]
        text = holders[-1].split("Status:", 1)[1] if holders else ""
    known = [m.group(1).lower() for m in SPEC_STATUS_RE.finditer(text)]
    if known:
        return known[-1]
    first = re.findall(r"Status:\s*(\w[\w-]*)", text)
    return first[0].lower() if first else ""


def looks_like_path(token):
    token = token.strip()
    if ":" in token or "(" in token or "=" in token or " " in token:
        return False
    return "/" in token or bool(re.search(r"\.[A-Za-z0-9]+$", token))


def module_tokens(block):
    """Backticked paths of the Module/Modules field that are the block's own.

    Modules are listed comma-separated; parenthesised annotations ("(the
    fitter)", "(`measure_node`)") and everything after the first semicolon
    outside parentheses ("; selected by ... through `scripts/end_to_end.py`")
    are prose about consumers, not modules of the block.
    """
    text = field_text(block, "module", "modules")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\([^()]*\)", "", text)
    text = text.split(";")[0]
    return [t.strip() for t in BACKTICKED.findall(text) if looks_like_path(t)]


def coverage_paths(text):
    paths, inside, found = [], False, False
    for line in (text or "").splitlines():
        if line.startswith("#"):
            inside = bool(re.match(r"^#{1,6}\s+.*\bCoverage\b", line, re.I))
            found = found or inside
            continue
        if inside:
            for snippet in BACKTICKED.findall(line):
                token = snippet.strip()
                if token and not token.startswith("<") and looks_like_path(token):
                    paths.append(token)
    return paths if found else []


def resolve_modules(tokens, paths, coverage):
    """Map the backticked module tokens of a SPEC entry to paths of the tree.

    A token is taken as is when the tree has it; otherwise as a suffix
    (`analysis/fit.py` -> `<...>/src/allen_diameter/analysis/fit.py`),
    preferring paths under the coverage and then the shortest path. A token
    ending in "/" claims a directory; a token with * ? [ is a glob.
    """
    resolved, unresolved = [], []
    path_set = set(paths)
    for token in tokens:
        if token in path_set:
            resolved.append(token)
            continue
        if token.endswith("/"):
            hits = [p for p in paths if p.startswith(token)
                    or ("/" + token) in p]
            dirs = sorted({p[:p.index(token) + len(token)] for p in hits
                           if token in p})
            if dirs:
                resolved.extend(dirs)
                continue
        if any(ch in token for ch in "*?["):
            hits = [p for p in paths if fnmatch.fnmatchcase(p, token)
                    or fnmatch.fnmatchcase(os.path.basename(p), token)]
            if hits:
                resolved.extend(hits)
                continue
        hits = [p for p in paths if p.endswith("/" + token)]
        if hits:
            covered = [p for p in hits if any(claims(c, p) for c in coverage)]
            pool = covered or hits
            pool.sort(key=lambda p: (len(p), p))
            resolved.append(pool[0])
            continue
        unresolved.append(token)
    return sorted(set(resolved)), unresolved


def claims(prefix, path):
    prefix = prefix.strip()
    if any(ch in prefix for ch in "*?["):
        return fnmatch.fnmatchcase(path, prefix)
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


def block_refs(text):
    """Block numbers named as "Block 3", "Blocks 2, 4 and 7", "Blocks 0-2".

    Parenthesised annotations are dropped first, so "Blocks 0 (real
    provider), 3 (fit), 4 (synthetic provider)" gives {0, 3, 4}.
    """
    numbers = set()
    text = re.sub(r"\([^)]*\)", "", text or "")
    for match in BLOCK_REFS.finditer(text):
        chunk = match.group(1)
        ranges = re.findall(r"(\d+)\s*-\s*(\d+)", chunk)
        for lo, hi in ranges:
            numbers.update(range(int(lo), int(hi) + 1))
        chunk = re.sub(r"\d+\s*-\s*\d+", "", chunk)
        numbers.update(int(n) for n in re.findall(r"\d+", chunk))
    return numbers


def consumed_blocks(block):
    """Block numbers named after 'consumes' on the Data flow line."""
    text = field_text(block, "data flow", "consumes / produces for")
    match = re.search(r"consumes(.*?)(?:;|feeds|$)", text, re.I | re.S)
    return block_refs(match.group(1)) if match else set()


# ---------------------------------------------------------- tester reports ---

def parse_report(text, path):
    commit = REPORT_COMMIT.search(text or "")
    if not commit:
        return None
    mode = REPORT_MODE.search(text)
    scope = REPORT_SCOPE.search(text)
    date = REPORT_DATE.search(text)
    verdict = VERDICT.search(text)
    report = {
        "path": path,
        "commit": commit.group(1),
        "mode": (mode.group(1).lower() if mode else "full"),
        "scope": (scope.group(1).strip() if scope else ""),
        "date": (date.group(1) if date else ""),
        "verdict": (verdict.group(1) if verdict else ""),
        "findings": [],
        "decisions": {},
    }
    scope_text = report["scope"]
    report["scope_blocks"] = (None if report["mode"] == "full"
                              or "whole" in scope_text.lower()
                              else block_refs(scope_text))
    report["findings"] = parse_findings(text)
    report["decisions"] = parse_decisions(text)
    return report


def table_rows(text, heading_regex):
    """Rows (list of cells) of the first table under a heading."""
    rows, inside, header_seen = [], False, False
    for line in (text or "").splitlines():
        if line.startswith("#"):
            inside = bool(re.match(heading_regex, line, re.I))
            header_seen = False
            continue
        if not inside:
            continue
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not header_seen:
                header_seen = True  # header row
                continue
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                continue
            rows.append(cells)
        elif rows:
            break
    return rows


def parse_findings(text):
    findings = []
    for cells in table_rows(text, r"^#{1,6}\s+Findings"):
        if len(cells) < 3 or not re.match(r"^F\d+", cells[0]):
            continue
        # "4, 6, 11", "0 -> 5 boundary", "1 (10)", "7 (gate 2 tool)":
        # drop parentheses that hold words, keep bare numbers
        cell = re.sub(r"\([^)]*[A-Za-z][^)]*\)", "", cells[2])
        findings.append({
            "id": cells[0],
            "severity": cells[1].upper(),
            "blocks": sorted({int(n) for n in re.findall(r"\d+", cell)}),
            "finding": cells[4] if len(cells) > 4 else "",
        })
    return findings


def parse_decisions(text):
    decisions = {}
    for cells in table_rows(text, r"^#{1,6}\s+Triage"):
        if cells and re.match(r"^[FS]\d+", cells[0]) and len(cells) >= 2:
            decisions[cells[0]] = cells[-1]
    for cells in table_rows(text, r"^#{1,6}\s+Tester findings"):
        if cells and re.match(r"^[FS]\d+", cells[0]) and len(cells) >= 2:
            decisions.setdefault(cells[0], cells[-1])
    return decisions


def collect_reports(root, head, reports_dir="specs/test_reports"):
    """Reports on the branch head and on tester/* branches, newest first."""
    refs = [head] + git(root, "for-each-ref", "--format=%(refname)",
                        "refs/heads/tester", "refs/remotes/origin/tester").split()
    reports, decisions, seen = [], {}, set()
    for ref in refs:
        listing = subprocess.run(
            ["git", "-C", root, "ls-tree", "--name-only", ref, reports_dir + "/"],
            capture_output=True, text=True).stdout.splitlines()
        for path in listing:
            if not path.endswith(".md"):
                continue
            text = read_at(root, ref, path) or ""
            base = os.path.basename(path)
            if "triage" in base.lower():
                # a triage file (<date>_triage.md) belongs to the report(s)
                # of the same date; its table may sit under any heading
                day = base[:10]
                found = decisions.setdefault(day, {})
                for cells in table_rows_any(text):
                    if cells and re.match(r"^[FS]\d+", cells[0]) and cells[-1].strip():
                        found.setdefault(cells[0], cells[-1])
                continue
            report = parse_report(text, path)
            if report and report["commit"] not in seen:
                seen.add(report["commit"])
                reports.append(report)
    for report in reports:
        day = os.path.basename(report["path"])[:10]
        for fid, cell in decisions.get(day, {}).items():
            if not report["decisions"].get(fid, "").strip():
                report["decisions"][fid] = cell
    reports.sort(key=lambda r: (r["date"], r["path"]), reverse=True)
    return reports


def table_rows_any(text):
    rows, header_seen = [], False
    for line in (text or "").splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                continue
            if not header_seen:
                header_seen = True
                continue
            rows.append(cells)
        else:
            header_seen = False
    return rows


def tester_runs(root, reports_dir="specs/test_reports"):
    """Status files of tester/* branches: running runs and their commits."""
    runs = []
    refs = git(root, "for-each-ref", "--format=%(refname)",
               "refs/heads/tester", "refs/remotes/origin/tester").split()
    for ref in refs:
        listing = subprocess.run(
            ["git", "-C", root, "ls-tree", "--name-only", ref, reports_dir + "/"],
            capture_output=True, text=True).stdout.splitlines()
        for path in listing:
            if not path.endswith(".status"):
                continue
            text = read_at(root, ref, path) or ""
            status = STATUS_FIELD.search(text)
            commit = STATUS_COMMIT.search(text)
            if not status or not commit:
                continue
            mode = STATUS_MODE.search(text)
            started = STATUS_STARTED.search(text)
            runs.append({
                "branch": ref.replace("refs/remotes/origin/", "").replace("refs/heads/", ""),
                "status": status.group(1).lower(),
                "commit": commit.group(1),
                "mode": (mode.group(1).lower() if mode else "full"),
                "started": (started.group(1).strip() if started else ""),
            })
    return runs


# ------------------------------------------------------------ hpc markers ---

def hpc_markers(root, hpc_branch):
    """Markers runs/<stage>/<job>.json on the hpc-status branch (origin first)."""
    ref = None
    for candidate in ("refs/remotes/origin/" + hpc_branch, "refs/heads/" + hpc_branch):
        if resolve_rev(root, candidate):
            ref = candidate
            break
    if ref is None:
        return []
    markers = []
    for path in subprocess.run(
            ["git", "-C", root, "ls-tree", "-r", "--name-only", ref, "runs/"],
            capture_output=True, text=True).stdout.splitlines():
        if not path.endswith(".json"):
            continue
        try:
            marker = json.loads(read_at(root, ref, path) or "{}")
        except ValueError:
            continue
        marker["path"] = path
        marker.setdefault("stage", path.split("/")[1] if path.count("/") >= 2 else "")
        markers.append(marker)
    return markers


def parse_time(value):
    if not value:
        return None
    value = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = dt.datetime.strptime(value, "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


# ------------------------------------------------------------------ status ---

def compute_status(data, root, fetch=True, source="chat"):
    branch = data["branch"]
    if fetch:
        subprocess.run(["git", "-C", root, "fetch", "--quiet", "origin",
                        "+refs/heads/tester/*:refs/remotes/origin/tester/*"],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", root, "fetch", "--quiet", "origin",
                        "+refs/heads/%s:refs/remotes/origin/%s"
                        % (data["hpc_branch"], data["hpc_branch"])],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", root, "fetch", "--quiet", "origin",
                        "+refs/heads/%s:refs/remotes/origin/%s" % (branch, branch)],
                       capture_output=True, text=True)

    head = resolve_rev(root, branch) or resolve_rev(root, "origin/" + branch)
    if head is None:
        raise GitError("branch %s not found locally or on origin" % branch)
    remote_head = resolve_rev(root, "origin/" + branch)
    current = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    worktree_is_branch = current == branch
    shallow = git(root, "rev-parse", "--is-shallow-repository").strip() == "true"

    spec_text = read_at(root, head, data["spec"])
    blocks = parse_spec(spec_text)
    coverage = coverage_paths(spec_text)
    paths = tree_paths(root, head)
    reports = collect_reports(root, head)
    runs = tester_runs(root)
    markers = hpc_markers(root, data["hpc_branch"])
    dirty = set()
    if worktree_is_branch:
        for line in git(root, "status", "--porcelain", "--untracked-files=no").splitlines():
            dirty.add(line[3:].split(" -> ")[-1].strip('"'))
    now = dt.datetime.now(dt.timezone.utc)
    stale_hours = float(data.get("stale_after_hours", 48))

    last_finished = max((r for r in reports), key=lambda r: r["date"], default=None)

    theory = {}
    for stage in data["theory"]:
        theory[stage["id"]] = {
            "id": stage["id"],
            "title": stage.get("title", stage["id"]),
            "summary": stage.get("summary", ""),
            "refs": as_list(stage.get("refs")),
            "depends_on": as_list(stage.get("depends_on")),
            "parent": stage.get("parent"),
            "status": stage.get("status", "open"),
            "closed_on": str(stage.get("closed_on") or ""),
            "decision": str(stage.get("decision") or ""),
            "note": stage.get("note", ""),
            "has_own": stage.get("status_given", True),
        }
    # a theory stage without members always has a status of its own
    members_of = {}
    for stage in data["theory"]:
        if stage.get("parent") in theory:
            members_of.setdefault(stage["parent"], []).append(stage["id"])
    for sid, stage in theory.items():
        if sid not in members_of:
            stage["has_own"] = True

    impl = {}
    for stage in data["implementation"]:
        impl[stage["id"]] = evaluate_stage(
            stage, data, root, head, remote_head, worktree_is_branch, dirty,
            blocks, coverage, paths, reports, runs, markers, now, stale_hours,
            last_finished)

    # groups: children, depth, and the aggregate of the descendants' states
    attach_groups(theory, "theory")
    attach_groups(impl, "implementation")

    # blocked_by: unmet prerequisites (does not change the colour)
    for sid, stage in impl.items():
        blocked = []
        for ref in stage["needs_theory"]:
            if ref in theory and theory[ref]["status"] != "closed":
                blocked.append(ref)
        for ref in stage["depends_on"]:
            if ref in impl and impl[ref]["status"] != "done":
                blocked.append(ref)
        stage["blocked_by"] = blocked if stage["has_own"] else []
    for sid, stage in theory.items():
        stage["blocked_by"] = [ref for ref in stage["depends_on"]
                               if ref in theory and theory[ref]["status"] != "closed"
                               ] if stage["has_own"] else []
        stage["feeds"] = [i for i, s in impl.items() if sid in s["needs_theory"]]

    counts = {s: 0 for s in IMPL_STATES}
    for stage in impl.values():
        if stage["has_own"]:
            counts[stage["status"]] = counts.get(stage["status"], 0) + 1
    tcounts = {s: 0 for s in THEORY_STATES}
    for stage in theory.values():
        if stage["has_own"]:
            tcounts[stage["status"]] = tcounts.get(stage["status"], 0) + 1

    warnings = []
    if shallow:
        warnings.append("shallow clone: a block's last change may be reported "
                        "as the clone's oldest commit, which can turn done into "
                        "pushed; deepen with git fetch --depth=2000 origin %s" % branch)
    if spec_text is None:
        warnings.append("%s not found on %s" % (data["spec"], branch))
    unresolved = {s["id"]: s["unresolved_modules"] for s in impl.values()
                  if s.get("unresolved_modules")}
    if unresolved:
        warnings.append("module tokens of SPEC.md not found in the tree: %s"
                        % json.dumps(unresolved))

    return {
        "schema": SCHEMA_VERSION,
        "key": doc_key(data),
        "name": data["name"],
        "title": data.get("title", data["name"]),
        "repo": data.get("repo", ""),
        "branch": branch,
        "commit": head,
        "commit_short": head[:7],
        "commit_date": git(root, "log", "-1", "--format=%cI", head).strip(),
        "synced_at": now_iso(),
        "synced_from": source,
        "shallow_clone": shallow,
        "spec": data["spec"],
        "hpc_branch": data["hpc_branch"],
        "theory": [theory[s["id"]] for s in data["theory"]],
        "implementation": [impl[s["id"]] for s in data["implementation"]],
        "counts": {"implementation": counts, "theory": tcounts},
        "legend": {kind: [{"state": s, "color": c, "text": t, "meaning": m}
                          for s, c, t, m in states]
                   for kind, states in STATES.items()},
        "reports_seen": [{"path": r["path"], "commit": r["commit"][:7],
                          "date": r["date"], "verdict": r["verdict"]}
                         for r in reports],
        "hpc_markers_seen": len(markers),
        "warnings": warnings,
    }


# least advanced first; a group takes the colour of its least advanced member
AGGREGATE_ORDER = {
    "implementation": ["failing", "planned", "local", "pushed", "testing",
                       "tested", "running", "done"],
    "theory": ["open", "studying", "closed"],
}


def attach_groups(stages, kind):
    """Fill parent/children/depth and, for stages with children, aggregate.

    aggregate = {"worst": state, "counts": {state: n}} over every descendant
    that has a state of its own (has_own), the stage's own state included.
    A stage without a state of its own (an implementation group with no
    block) takes the aggregate as its status, so it can be drawn.
    """
    order = AGGREGATE_ORDER[kind]
    children = {sid: [] for sid in stages}
    for sid, stage in stages.items():
        parent = stage.get("parent")
        if parent in stages and parent != sid:
            children[parent].append(sid)
        else:
            stage["parent"] = None
    for sid, stage in stages.items():
        stage["children"] = children[sid]
        depth, cur, seen = 0, stage.get("parent"), set()
        while cur in stages and cur not in seen:
            seen.add(cur)
            depth += 1
            cur = stages[cur].get("parent")
        stage["depth"] = depth

    def own_states(sid, seen):
        if sid in seen:
            return []
        seen.add(sid)
        found = [stages[sid]["status"]] if stages[sid]["has_own"] else []
        for child in children[sid]:
            found += own_states(child, seen)
        return found

    for sid, stage in stages.items():
        if not children[sid]:
            stage["aggregate"] = None
            continue
        states = own_states(sid, set())
        counts = {}
        for state in states:
            counts[state] = counts.get(state, 0) + 1
        worst = min(states, key=lambda s: order.index(s) if s in order else -1) if states else None
        stage["aggregate"] = {"worst": worst, "counts": counts, "n": len(states)}
        if not stage["has_own"] and worst:
            stage["status"] = worst
            stage["reason"] = "group: least advanced member is %s" % worst


def doc_key(data):
    repo = (data.get("repo") or "repo").split("/")[-1].lower()
    return "%s:%s" % (repo, data["name"])


def evaluate_stage(stage, data, root, head, remote_head, worktree_is_branch,
                   dirty, blocks, coverage, paths, reports, runs, markers,
                   now, stale_hours, last_finished):
    sid = stage["id"]
    result = {
        "id": sid,
        "title": stage.get("title", sid),
        "summary": stage.get("summary", ""),
        "blocks": stage["blocks"],
        "depends_on": as_list(stage.get("depends_on")),
        "needs_theory": as_list(stage.get("needs_theory")),
        "needs_hpc": bool(stage.get("needs_hpc")),
        "jobs": as_list(stage.get("jobs")),
        "note": stage.get("note", ""),
        "parent": stage.get("parent"),
        # a stage has a state of its own when it has a block or a manual
        # status; otherwise it is a pure group and takes its members' aggregate
        "has_own": bool(stage["blocks"]) or bool(stage.get("manual_status")),
        "modules": [],
        "unresolved_modules": [],
        "spec_status": "",
        "last_change": "",
        "last_change_date": "",
        "tester": {},
        "hpc": {},
        "status": "planned",
        "reason": "",
        "history": [],
    }

    # --- code: modules named by the SPEC entries of the stage's blocks
    tokens, spec_words, missing_blocks = [], {}, []
    for number in stage["blocks"]:
        block = blocks.get(number)
        if block is None:
            missing_blocks.append(number)
            continue
        tokens.extend(module_tokens(block))
        spec_words[number] = spec_status_word(block)
    modules, unresolved = resolve_modules(tokens, paths, coverage)
    result["modules"] = modules
    result["unresolved_modules"] = unresolved
    result["spec_status"] = ", ".join("%d: %s" % (n, w) for n, w in spec_words.items() if w)

    if stage.get("manual_status"):
        result["status"] = stage["manual_status"]
        result["manual"] = True
        result["reason"] = "set by hand: %s" % stage.get("manual_reason", "no reason given")
        return result
    if not stage["blocks"]:
        result["reason"] = "no SPEC block assigned (a group, if it has members)"
        return result
    if missing_blocks and len(missing_blocks) == len(stage["blocks"]):
        result["reason"] = "no entry in SPEC.md for Block %s" % ", ".join(map(str, missing_blocks))
        return result
    drafted = [str(n) for n, w in spec_words.items() if w.startswith("drafted")]
    if drafted:
        result["reason"] = "SPEC.md status: drafted (Block %s)" % ", ".join(drafted)
        return result
    if not modules:
        result["reason"] = ("the modules of the SPEC entry do not exist on %s"
                            % data["branch"]) + (": %s" % ", ".join(unresolved)
                                                   if unresolved else "")
        return result

    # --- last change of the code, on the branch
    log = git(root, "log", "-1", "--format=%H%x09%cI", head, "--", *modules).strip()
    if log:
        result["last_change"], result["last_change_date"] = log.split("\t")
    last_change = result["last_change"] or head

    # --- local: not pushed
    if worktree_is_branch:
        touched = [p for p in dirty if any(claims(m, p) for m in modules)]
        if touched:
            result["status"] = "local"
            result["reason"] = "uncommitted changes in %s" % ", ".join(sorted(touched)[:3])
            return result
    if remote_head is None:
        result["status"] = "local"
        result["reason"] = "branch %s is not on origin" % data["branch"]
        return result
    if not is_ancestor(root, last_change, remote_head):
        result["status"] = "local"
        result["reason"] = "commit %s is not on origin/%s" % (last_change[:7], data["branch"])
        return result

    # --- HPC markers of this stage
    mine = [m for m in markers if marker_matches(m, sid, result["jobs"])]
    mine.sort(key=lambda m: (m.get("started") or ""), reverse=True)
    hpc = {}
    if mine:
        marker = mine[0]
        started = parse_time(marker.get("started"))
        age_h = ((now - started).total_seconds() / 3600.0) if started else None
        run_commit = resolve_rev(root, marker["commit"]) if marker.get("commit") else None
        hpc = {
            "state": marker.get("state", ""),
            "job": marker.get("job_id", ""),
            "job_name": marker.get("job_name", ""),
            "commit": (marker.get("commit") or "")[:7],
            "commit_known": run_commit is not None,
            "started": marker.get("started", ""),
            "ended": marker.get("ended", ""),
            "n_tasks": marker.get("n_tasks"),
            "n_failed": marker.get("n_failed"),
            "age_hours": (round(age_h, 1) if age_h is not None else None),
            "covers_current_code": bool(
                run_commit and is_ancestor(root, last_change, run_commit)),
            "stale": bool(marker.get("state") == "running" and age_h is not None
                          and age_h > stale_hours),
        }
    result["hpc"] = hpc
    if hpc.get("state") == "running" and not hpc.get("stale"):
        result["status"] = "running"
        result["reason"] = "HPC job %s running since %s" % (hpc["job"] or "?", hpc["started"])
        return result

    # --- tester
    covering = [r for r in reports
                if report_covers(r, stage["blocks"])
                and resolve_rev(root, r["commit"])
                and is_ancestor(root, last_change, r["commit"])]
    running = []
    for run in runs:
        if run["status"] != "running" or not resolve_rev(root, run["commit"]):
            continue
        if not is_ancestor(root, last_change, run["commit"]):
            continue
        in_scope = (run["mode"] == "full" or last_finished is None
                    or not resolve_rev(root, last_finished["commit"])
                    or not is_ancestor(root, last_change, last_finished["commit"]))
        if in_scope:
            running.append(run)
    tester = {}
    if covering:
        report = covering[0]
        open_findings = [f for f in report["findings"]
                         if f["severity"] in ("BLOCKER", "MAJOR")
                         and set(f["blocks"]) & set(stage["blocks"])
                         and not RESOLVED_WORDS.search(report["decisions"].get(f["id"], ""))]
        tester = {
            "report": report["path"], "commit": report["commit"][:7],
            "date": report["date"], "verdict": report["verdict"],
            "open_findings": [f["id"] + " " + f["severity"] for f in open_findings],
        }
    elif reports:
        older = [r for r in reports if report_covers(r, stage["blocks"])]
        if older:
            tester = {"report": older[0]["path"], "commit": older[0]["commit"][:7],
                      "date": older[0]["date"], "verdict": older[0]["verdict"],
                      "outdated": True}
    result["tester"] = tester
    if running:
        result["status"] = "testing"
        result["reason"] = "tester running on %s (%s)" % (running[0]["commit"][:7], running[0]["branch"])
        return result
    if tester.get("open_findings"):
        result["status"] = "failing"
        result["reason"] = "open %s in %s" % (", ".join(tester["open_findings"]), tester["report"])
        return result
    if hpc and hpc.get("state") == "failed" and hpc.get("covers_current_code"):
        result["status"] = "failing"
        result["reason"] = "HPC job %s failed (%s of %s tasks)" % (
            hpc["job"] or "?", hpc.get("n_failed"), hpc.get("n_tasks"))
        return result
    if hpc.get("stale"):
        result["status"] = "failing"
        result["reason"] = "HPC running marker is %.0f h old with no end marker" % hpc["age_hours"]
        return result
    if not covering:
        result["status"] = "pushed"
        result["reason"] = ("no tester report covers commit %s" % last_change[:7]
                            + (" (last report %s at %s)" % (tester["date"], tester["commit"])
                               if tester.get("outdated") else ""))
        return result
    if result["needs_hpc"] and not (hpc.get("state") == "done" and hpc.get("covers_current_code")):
        if hpc.get("state") == "done":
            why = ("at a commit this clone does not know" if not hpc.get("commit_known")
                   else "predates the last change")
        else:
            why = "missing"
        result["status"] = "tested"
        result["reason"] = "tester %s on %s; HPC run %s" % (
            tester["verdict"], tester["commit"], why)
        return result
    result["status"] = "done"
    result["reason"] = "tester %s on %s" % (tester["verdict"], tester["commit"]) + (
        "; HPC job %s done" % hpc["job"] if result["needs_hpc"] else "")
    return result


def marker_matches(marker, sid, jobs):
    if marker.get("stage") == sid:
        return True
    name = marker.get("job_name") or ""
    script = marker.get("script") or ""
    for pattern in jobs:
        for candidate in (name, script, os.path.basename(script)):
            if candidate and (candidate == pattern or fnmatch.fnmatchcase(candidate, pattern)):
                return True
    return False


def report_covers(report, numbers):
    if report["scope_blocks"] is None:
        return True
    return bool(set(numbers) & set(report["scope_blocks"]))


# ------------------------------------------------------------------ render ---

def mermaid(data, status):
    colors = {s: (fill, text) for s, fill, text, _ in STATES["implementation"]}
    colors.update({s: (fill, text) for s, fill, text, _ in STATES["theory"]})
    lines = ["flowchart LR"]
    for state, (fill, text) in colors.items():
        lines.append("    classDef %s fill:%s,color:%s,stroke:%s" % (state, fill, text, fill))
    lines.append("    classDef blocked stroke-dasharray:5 3,stroke:#333")
    styles = []

    def node_label(stage, kind):
        if kind == "theory":
            label = "%s %s" % (stage["id"], stage["title"])
            if stage["closed_on"]:
                label += "<br/>closed %s" % stage["closed_on"]
            return label
        blocks = ("Block %s" % ", ".join(map(str, stage["blocks"]))) if stage["blocks"] else ""
        return "%s %s<br/>%s %s" % (stage["id"], stage["title"], blocks, stage["status"])

    def emit(stages, kind, parent, indent):
        pad = "    " * indent
        for stage in stages:
            if stage["parent"] != parent:
                continue
            if stage["children"]:
                agg = stage.get("aggregate") or {}
                counts = ", ".join("%d %s" % (n, s) for s, n in sorted(
                    agg.get("counts", {}).items(), key=lambda kv: -kv[1]))
                title = "%s %s" % (stage["id"], stage["title"])
                if counts:
                    title += " (" + counts + ")"
                lines.append('%ssubgraph grp_%s["%s"]' % (pad, stage["id"], esc(title)))
                lines.append("%s    direction TB" % pad)
                if stage["has_own"]:
                    lines.append('%s    %s["%s"]:::%s' % (
                        pad, stage["id"], esc(node_label(stage, kind)), stage["status"]))
                emit(stages, kind, stage["id"], indent + 1)
                lines.append("%send" % pad)
                worst = agg.get("worst") or stage["status"]
                styles.append("    style grp_%s stroke:%s,stroke-width:3px"
                              % (stage["id"], colors.get(worst, ("#8d99ae",))[0]))
            else:
                lines.append('%s%s["%s"]:::%s' % (
                    pad, stage["id"], esc(node_label(stage, kind)), stage["status"]))

    def endpoint(stage_map, sid):
        stage = stage_map.get(sid)
        if stage and stage["children"] and not stage["has_own"]:
            return "grp_" + sid
        return sid

    tmap = {s["id"]: s for s in status["theory"]}
    imap = {s["id"]: s for s in status["implementation"]}
    lines.append('    subgraph theory["Theory"]')
    lines.append("        direction TB")
    emit(status["theory"], "theory", None, 2)
    for stage in status["theory"]:
        for ref in stage["depends_on"]:
            lines.append("        %s --> %s" % (endpoint(tmap, ref), endpoint(tmap, stage["id"])))
    lines.append("    end")
    lines.append('    subgraph impl["Implementation"]')
    lines.append("        direction TB")
    emit(status["implementation"], "implementation", None, 2)
    for stage in status["implementation"]:
        for ref in stage["depends_on"]:
            lines.append("        %s --> %s" % (endpoint(imap, ref), endpoint(imap, stage["id"])))
    lines.append("    end")
    for stage in status["implementation"]:
        for ref in stage["needs_theory"]:
            lines.append("    %s -.-> %s" % (endpoint(tmap, ref), endpoint(imap, stage["id"])))
    blocked = [s["id"] for s in status["theory"] + status["implementation"]
               if s.get("blocked_by") and (s["has_own"] or not s["children"])]
    if blocked:
        lines.append("    class %s blocked" % ",".join(blocked))
    lines.extend(styles)
    return "\n".join(lines)


def esc(text):
    return text.replace('"', "#quot;")


def render_markdown(data, status):
    out = []
    out.append("# Roadmap -- %s" % status["title"])
    out.append("")
    out.append("Generated by `specs/roadmap/roadmap.py render` from `specs/roadmap/%s.yaml`; "
               "do not edit. Branch `%s` at `%s` (%s); synced %s from %s."
               % (status["name"], status["branch"], status["commit_short"],
                  status["commit_date"][:10], status["synced_at"], status["synced_from"]))
    out.append("")
    out.append("```mermaid")
    out.append(mermaid(data, status))
    out.append("```")
    out.append("")
    out.append("## Legend")
    out.append("")
    out.append("| State | Colour | Meaning |")
    out.append("|---|---|---|")
    for kind in ("implementation", "theory"):
        for state, color, _, meaning in STATES[kind]:
            out.append("| %s (%s) | %s | %s |" % (state, kind, color, meaning))
    out.append("")
    out.append("A dashed border marks a stage whose prerequisites are not met "
               "(theory not closed, or an earlier implementation stage not done).")
    out.append("")
    out.append("## Theory")
    out.append("")
    out.append("| Stage | Group | Status | Closed | Decision | Depends on | Feeds | Summary |")
    out.append("|---|---|---|---|---|---|---|---|")
    for stage in status["theory"]:
        out.append("| %s %s | %s | %s | %s | %s | %s | %s | %s |" % (
            stage["id"], stage["title"], stage["parent"] or "", stage["status"],
            stage["closed_on"], stage["decision"], ", ".join(stage["depends_on"]),
            ", ".join(stage["feeds"]), cell(stage["summary"])))
    out.append("")
    out.append("## Implementation")
    out.append("")
    out.append("| Stage | Group | Blocks | Status | Why | Last change | Tester | HPC | Needs | Summary |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for stage in status["implementation"]:
        tester = stage["tester"]
        hpc = stage["hpc"]
        needs = ", ".join(stage["needs_theory"] + stage["depends_on"])
        out.append("| %s %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            stage["id"], stage["title"], stage["parent"] or "",
            ", ".join(map(str, stage["blocks"])),
            stage["status"], cell(stage["reason"]),
            (stage["last_change"][:7] + " " + stage["last_change_date"][:10]).strip(),
            ((tester.get("verdict", "") + " " + tester.get("commit", "")).strip()
             + (" (outdated)" if tester.get("outdated") else "")) if tester else "",
            ((hpc.get("state", "") + " " + (hpc.get("job") or "")).strip()) if hpc else "",
            needs, cell(stage["summary"])))
    out.append("")
    return "\n".join(out) + "\n"


def cell(text):
    return str(text or "").replace("|", "/").replace("\n", " ")


# -------------------------------------------------------------------- set ---

def set_theory_status(path, stage_id, new_status, note=None, decision=None, on=None):
    """Edit the status of a theory stage in place, keeping the file's layout."""
    if new_status not in THEORY_STATES:
        raise RoadmapError("status must be one of %s" % THEORY_STATES)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines(True)
    # find the stage block: a line "- id: <stage_id>" inside the theory list
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*-\s*id:\s*['\"]?%s['\"]?\s*$" % re.escape(stage_id), line):
            start = i
            break
    if start is None:
        raise RoadmapError("stage %s not found in %s" % (stage_id, path))
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].lstrip()
        cur = len(lines[j]) - len(stripped)
        if stripped and cur <= indent and not stripped.startswith("#"):
            end = j
            break
    field_indent = " " * (indent + 2)
    on = on or dt.date.today().isoformat()
    updates = {"status": new_status}
    if new_status == "closed":
        updates["closed_on"] = on
    if decision:
        updates["decision"] = decision
    if note:
        updates["note"] = note
    present = {}
    for j in range(start + 1, end):
        match = re.match(r"^(\s*)(\w+):", lines[j])
        if match:
            present[match.group(2)] = j
    for key, value in updates.items():
        text = "%s%s: %s\n" % (field_indent, key, yaml_scalar(value))
        if key in present:
            lines[present[key]] = text
        else:
            lines.insert(end, text)
            end += 1
    if new_status != "closed" and "closed_on" in present:
        lines[present["closed_on"]] = "%sclosed_on:\n" % field_indent
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(lines))


def yaml_scalar(value):
    text = str(value)
    if re.match(r"^[A-Za-z0-9_.:-]+$", text):
        return text
    return json.dumps(text)


# ------------------------------------------------------------------- init ---

def init_roadmap(root, name, branch, repo=None, spec="specs/SPEC.md", out=None, title=None):
    head = resolve_rev(root, branch) or resolve_rev(root, "origin/" + branch)
    if head is None:
        raise GitError("branch %s not found" % branch)
    spec_text = read_at(root, head, spec) or ""
    blocks = parse_spec(spec_text)
    title = title or (re.match(r"^#\s*SPEC\s*--\s*(.*)$", spec_text.splitlines()[0]).group(1)
                      if spec_text and re.match(r"^#\s*SPEC\s*--", spec_text) else name)
    if repo is None:
        url = git(root, "remote", "get-url", "origin", check=False).strip()
        match = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        repo = match.group(1) if match else ""
    lines = [
        "# Roadmap of the subproject %s. Source of truth for the stages and the" % name,
        "# theory statuses; the implementation statuses are computed by",
        "# specs/roadmap/roadmap.py (never written by hand).",
        "#",
        "# Drafted by `roadmap.py init` from %s: review every stage, the" % spec,
        "# dependencies and needs_hpc, and replace the theory stages with the",
        "# topics you actually have to study. Lines marked DRAFT are guesses.",
        "schema: %d" % SCHEMA_VERSION,
        "name: %s" % name,
        "title: %s" % yaml_scalar(title),
        "repo: %s" % repo,
        "branch: %s" % branch,
        "spec: %s" % spec,
        "hpc_branch: hpc-status",
        "stale_after_hours: 48      # a running HPC marker older than this is reported as failing",
        "",
        "theory:",
    ]
    # theory stages drafted from the subsections of the mathematical
    # formulation ("### 2.k Title (Blocks n, m; ...)"): one stage each, fed
    # to the blocks its heading names
    sections = re.findall(r"^###\s+2\.(\d+)\s+(.*?)\s*$", spec_text, re.MULTILINE)
    if not sections:
        sections = [("1", "DRAFT: first topic to study")]
    section_blocks = {}
    for k, (num, heading) in enumerate(sections, start=1):
        sid = "T%d" % k
        # the blocks a section covers sit in its parentheses:
        # "### 2.2 Per-node measurement (Blocks 3, 5; handoff Eqs. 1-5)"
        section_blocks[sid] = block_refs(heading.replace("(", " ").replace(")", " "))
        short = re.sub(r"\s*\(.*?\)\s*$", "", heading)
        lines += [
            "  - id: %s" % sid,
            "    title: %s" % yaml_scalar(short),
            "    summary: \"DRAFT: from SPEC section 2.%s\"" % num,
            "    refs: []",
            "    parent:                 # id of the group stage this belongs to, if any",
            "    depends_on: %s" % ("[]" if k == 1 else "[T%d]" % (k - 1)),
            "    status: open",
            "    closed_on:",
            "    decision:",
        ]
    lines.append("")
    lines.append("implementation:")
    for number in sorted(blocks):
        block = blocks[number]
        consumed = sorted(n for n in consumed_blocks(block) if n in blocks and n != number)
        # a stage depends on the EARLIER blocks it consumes; a dependency on
        # a later block (a feedback loop in the data flow) is dropped so the
        # roadmap stays a DAG, and noted for review
        deps = [n for n in consumed if n < number]
        dropped = [n for n in consumed if n > number]
        needs_hpc = bool(HPC_WORDS.search(block["text"]))
        header = block["name"]
        needs_theory = [sid for sid, nums in section_blocks.items() if number in nums]
        short = re.sub(r"\s*\(.*$", "", header).strip()
        lines += [
            "  - id: I%d" % number,
            "    title: %s" % yaml_scalar(short),
            "    summary: %s" % yaml_scalar("DRAFT: " + short),
            "    block: %d" % number,
            "    parent:                 # id of the group stage this belongs to, if any",
            "    depends_on: [%s]" % ", ".join("I%d" % n for n in deps)
            + ("   # DRAFT: also consumes later Block %s (loop), not drawn"
               % ", ".join(map(str, dropped)) if dropped else ""),
            "    needs_theory: [%s]" % ", ".join(sorted(set(needs_theory))),
            "    needs_hpc: %s" % ("true" if needs_hpc else "false")
            + ("      # DRAFT: the entry mentions the cluster" if needs_hpc else ""),
            "    jobs: []                # job script paths or PBS/SLURM job names that run this stage",
        ]
    text = "\n".join(lines) + "\n"
    out = out or os.path.join(root, "specs", "roadmap", name + ".yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise RoadmapError("%s exists; delete it first or choose another --name" % out)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    # install the tool beside the roadmap and ignore the cluster's state dir
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("roadmap.py", "hpc_status.sh"):
        src, dst = os.path.join(here, fname), os.path.join(os.path.dirname(out), fname)
        if os.path.abspath(src) != os.path.abspath(dst) and os.path.exists(src):
            shutil.copyfile(src, dst)
    gitignore = os.path.join(root, ".gitignore")
    marker = ".roadmap/"
    existing = open(gitignore, encoding="utf-8").read() if os.path.exists(gitignore) else ""
    if marker not in existing.splitlines():
        with open(gitignore, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(("" if existing.endswith("\n") or not existing else "\n")
                     + "# roadmap: the cluster-side state of hpc_status.sh\n" + marker + "\n")
    return out


# ------------------------------------------------------------------- main ---

def find_root(args):
    if getattr(args, "root", None):
        return os.path.abspath(args.root)
    if getattr(args, "yaml", None):
        root = repo_root_of(args.yaml)
        if root:
            return root
    return os.path.abspath(os.getcwd())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="draft a roadmap from SPEC.md")
    p.add_argument("--name", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--repo", default=None, help="owner/name on GitHub")
    p.add_argument("--spec", default="specs/SPEC.md")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--root", default=None)

    p = sub.add_parser("check", help="validate a roadmap file")
    p.add_argument("--yaml", required=True)
    p.add_argument("--root", default=None)

    p = sub.add_parser("status", help="compute the stage statuses")
    p.add_argument("--yaml", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--json", default=None, help="write the status document here")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--source", default="chat")

    p = sub.add_parser("render", help="write the Mermaid markdown")
    p.add_argument("--yaml", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--status", default=None, help="a status JSON (default: compute)")
    p.add_argument("--out", default=None)
    p.add_argument("--no-fetch", action="store_true")

    p = sub.add_parser("set", help="set a theory stage's status")
    p.add_argument("--yaml", required=True)
    p.add_argument("stage")
    p.add_argument("status", choices=THEORY_STATES)
    p.add_argument("--note", default=None)
    p.add_argument("--decision", default=None)
    p.add_argument("--on", default=None, help="date, default today")

    args = parser.parse_args(argv)
    try:
        return run(args)
    except GitError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except RoadmapError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


def run(args):
    root = find_root(args)
    if args.command == "init":
        out = init_roadmap(root, args.name, args.branch, args.repo, args.spec,
                           args.out, args.title)
        print("wrote %s" % out)
        data = load_roadmap(out)
        for problem in check_roadmap(data):
            print("warning: %s" % problem)
        return 0
    data = load_roadmap(args.yaml)
    if args.command == "set":
        set_theory_status(args.yaml, args.stage, args.status, args.note,
                          args.decision, args.on)
        print("%s -> %s" % (args.stage, args.status))
        return 0
    problems = check_roadmap(data)
    if args.command == "check":
        for problem in problems:
            print("problem: %s" % problem)
        print("ok" if not problems else "%d problem(s)" % len(problems))
        return 0 if not problems else 1
    if problems:
        raise RoadmapError("; ".join(problems))
    if args.command == "status":
        status = compute_status(data, root, fetch=not args.no_fetch, source=args.source)
        text = json.dumps(status, indent=2, sort_keys=False)
        if args.json:
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text + "\n")
            print("wrote %s: %s" % (args.json, summary(status)))
        else:
            print(text)
        return 0
    if args.command == "render":
        if args.status:
            with open(args.status, encoding="utf-8") as fh:
                status = json.load(fh)
        else:
            status = compute_status(data, root, fetch=not args.no_fetch)
        out = args.out or os.path.splitext(args.yaml)[0] + ".md"
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_markdown(data, status))
        print("wrote %s: %s" % (out, summary(status)))
        return 0
    return 1


def summary(status):
    parts = ["%s %d" % (s, n) for s, n in status["counts"]["implementation"].items() if n]
    tparts = ["%s %d" % (s, n) for s, n in status["counts"]["theory"].items() if n]
    return "implementation " + ", ".join(parts) + "; theory " + ", ".join(tparts)


if __name__ == "__main__":
    sys.exit(main())
