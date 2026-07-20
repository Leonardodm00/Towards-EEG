#!/usr/bin/env python3
"""
decolab.py -- stage S0.2 transform for Towards-EEG.

WHAT IT DOES
------------
Neutralises notebook-only syntax so that every target file satisfies the S0.2
exit test (`ast.parse` and `py_compile` succeed), WITHOUT deleting anything and
WITHOUT changing what the file does when it is executed in a Colab notebook by
some other means.  Two transforms, both declared, both logged:

  T7  notebook-magic neutralisation.  A physical line whose first non-blank
      character is `!` or `%` is turned into a comment that carries the
      original text verbatim:

          !pip install neuron          ->   #S0.2:T7# !pip install neuron

      Line count, line numbering, indentation and line endings are preserved,
      so every line number recorded elsewhere (defect O7, LEDGER.md) stays
      valid, and `git blame` still lines up.

  T7p same, for the case where the magic is the ONLY statement of an indented
      block.  A bare comment would leave the block empty and produce an
      IndentationError, so the replacement carries a `pass`:

              !pip install x           ->       pass  #S0.2:T7p# !pip install x

      The tool escalates T7 -> T7p only when the parse demands it, never
      speculatively, and records which form it used.

  T8  a declared, single-line leading-whitespace correction.  This is NOT
      inferred: each fix names the file, the line, the exact expected original
      text and the exact replacement, and the tool aborts if the file on disk
      does not match the declaration byte for byte.

WHAT IT REFUSES TO DO
---------------------
A line that looks like a magic but lies INSIDE a string literal is left alone.
Two such lines exist in the real target set (a docstring in
`Passive Features/Plot/passive_result_plot (5).py` that documents the Colab
install command).  Commenting them out would silently change the value of a
string at run time -- a defect fix wearing the costume of a mechanical
transform, which clause (2) of the byte-identity rule does not permit.

COLAB IS STILL THE EXECUTION ENVIRONMENT
----------------------------------------
These scripts continue to run in Colab against the H01 database; only their
outputs move to the HPC.  T7 therefore neutralises the magics rather than
deleting them, and the transform log records, per file and in order, every
shell command that was neutralised.  That log is the machine-readable input
from which a Colab bootstrap orchestrator can later be generated -- see
`--emit-colab-manifest`.  Generating that orchestrator is NOT part of S0.2.

USAGE
-----
    python3 tools/s0_transform/decolab.py --root . --dry-run
    python3 tools/s0_transform/decolab.py --root . --apply
    python3 tools/s0_transform/decolab.py --root . --verify
    python3 tools/s0_transform/decolab.py --root . --restore
    python3 tools/s0_transform/decolab.py --root . --emit-colab-manifest

Dependencies: Python 3.8+ standard library only.  ASCII source throughout.
"""

import argparse
import ast
import datetime
import hashlib
import io
import json
import os
import re
import sys
import tokenize

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.dirname(_HERE)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from s0_transform import TOOL_NAME, TOOL_VERSION, T7, T7P, T8  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

CANDIDATE = re.compile(br"^([ \t]*)([!%].*)$")
MARK_T7 = b"#S0.2:T7# "
MARK_T7P = b"pass  #S0.2:T7p# "
RESTORE_T7 = re.compile(br"^([ \t]*)#S0\.2:T7# (.*)$")
RESTORE_T7P = re.compile(br"^([ \t]*)pass  #S0\.2:T7p# (.*)$")

MAX_ESCALATIONS = 64

DEFAULT_TARGETS = os.path.join(_HERE, "s02_targets.json")
DEFAULT_LOG = os.path.join(_HERE, "s02_transform_log.json")
DEFAULT_MANIFEST = os.path.join(_HERE, "s02_colab_commands.json")


# ---------------------------------------------------------------------------
# byte-exact line handling
# ---------------------------------------------------------------------------

def split_lines(data):
    """Split bytes into [(content, eol), ...] such that join_lines inverts it.

    `content` never contains a line terminator; `eol` is b'', b'\\n' or
    b'\\r\\n'.  A lone b'\\r' is deliberately NOT treated as a terminator: S0.3
    owns line-ending normalisation, and S0.2 must not anticipate it.
    """
    parts = data.split(b"\n")
    out = []
    last_i = len(parts) - 1
    for i, p in enumerate(parts):
        if i == last_i:
            out.append((p, b""))
        elif p.endswith(b"\r"):
            out.append((p[:-1], b"\r\n"))
        else:
            out.append((p, b"\n"))
    return out


def join_lines(lines):
    return b"".join(c + e for c, e in lines)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parses(data):
    """True if the bytes parse as Python. Decoding failure counts as False."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        # ValueError covers source containing NUL bytes, which is not a
        # SyntaxError but is equally not parseable.
        return False


def parse_error(data):
    try:
        ast.parse(data.decode("utf-8"))
        return None
    except UnicodeDecodeError as exc:
        return ("decode", 0, str(exc))
    except SyntaxError as exc:
        return ("syntax", exc.lineno or 0, exc.msg)


# ---------------------------------------------------------------------------
# string-literal awareness
# ---------------------------------------------------------------------------

def string_line_numbers(data):
    """1-based line numbers covered by a string/f-string token.

    Tokenisation is performed on the CANDIDATE-NEUTRALISED text, because the
    original does not tokenise (a bare `!` is not a Python token).  T7 is
    line-count and column preserving, so the line numbers agree.

    Returns (set_of_lines, ok).  `ok` is False if the text could not be
    tokenised at all, in which case the caller must abort rather than guess.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return set(), False
    covered = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            name = tokenize.tok_name.get(tok.type, "")
            if name in ("STRING", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
                for ln in range(tok.start[0], tok.end[0] + 1):
                    covered.add(ln)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set(), False
    return covered, True


# ---------------------------------------------------------------------------
# T7 / T7p
# ---------------------------------------------------------------------------

def _neutralise_all(lines):
    """Comment out every candidate line. Returns (new_lines, candidate_idxs)."""
    new = list(lines)
    idxs = []
    for i, (content, eol) in enumerate(lines):
        m = CANDIDATE.match(content)
        if not m:
            continue
        indent, body = m.group(1), m.group(2)
        new[i] = (indent + MARK_T7 + body, eol)
        idxs.append(i)
    return new, idxs


def apply_t7(data, path_for_msg="<bytes>"):
    """Apply T7/T7p to one file's bytes.

    Returns (new_data, edits, notes).  `edits` is a list of dicts; `notes`
    carries non-fatal observations.  Raises RuntimeError on any condition that
    would make the transform unsafe (untokenisable text, escalation loop).
    """
    lines = split_lines(data)
    staged, idxs = _neutralise_all(lines)
    if not idxs:
        return data, [], ["no candidate lines"]

    covered, ok = string_line_numbers(join_lines(staged))
    if not ok:
        raise RuntimeError(
            "cannot tokenise %s after neutralisation; refusing to guess which "
            "candidate lines lie inside string literals" % path_for_msg
        )

    notes = []
    kept = []
    for i in idxs:
        if (i + 1) in covered:
            staged[i] = lines[i]          # revert: it is inside a string
            notes.append(
                "line %d looks like a magic but lies inside a string literal; "
                "left byte-identical" % (i + 1)
            )
        else:
            kept.append(i)

    if len(kept) != len(idxs):
        # Reverting a line INSIDE a string literal restores bytes that could,
        # in principle, contain a quote sequence that terminates the string
        # early and so changes tokenisation. Re-derive the covered set and
        # require it to still cover every reverted line.
        covered2, ok2 = string_line_numbers(join_lines(staged))
        if not ok2:
            raise RuntimeError(
                "reverting in-string candidates made %s untokenisable"
                % path_for_msg)
        for i in idxs:
            if i not in kept and (i + 1) not in covered2:
                raise RuntimeError(
                    "line %d of %s left a string literal when reverted; "
                    "refusing to proceed" % (i + 1, path_for_msg))

    forms = dict((i, T7) for i in kept)

    # Escalate T7 -> T7p only where the parser demands it.
    for _ in range(MAX_ESCALATIONS):
        err = parse_error(join_lines(staged))
        if err is None:
            break
        kind, lineno, msg = err
        # Escalate ONLY on the one error an empty block produces. Any other
        # syntax error is not ours to fix inside S0.2 and must be surfaced.
        if kind != "syntax" or "expected an indented block" not in msg.lower():
            break
        target = None
        for i in sorted(kept, reverse=True):
            if i + 1 < lineno and forms[i] == T7:
                # everything between the edit and the reported line must be
                # blank or comment, otherwise this edit is not the empty block
                between = [staged[j][0].strip() for j in range(i + 1, lineno - 1)]
                if all((not b) or b.startswith(b"#") for b in between):
                    target = i
                break
        if target is None:
            break
        content, eol = lines[target]
        m = CANDIDATE.match(content)
        indent, body = m.group(1), m.group(2)
        staged[target] = (indent + MARK_T7P + body, eol)
        forms[target] = T7P
        notes.append(
            "line %d escalated to the pass-carrying form: it is the only "
            "statement of its block" % (target + 1)
        )
    else:
        raise RuntimeError("escalation did not converge on %s" % path_for_msg)

    edits = []
    for i in sorted(kept):
        edits.append({
            "line": i + 1,
            "transform": forms[i],
            "original": lines[i][0].decode("utf-8", "backslashreplace"),
            "replacement": staged[i][0].decode("utf-8", "backslashreplace"),
            "command": CANDIDATE.match(lines[i][0]).group(2).decode(
                "utf-8", "backslashreplace"),
            "reason": "notebook-only syntax; neutralised so the module parses",
        })

    return join_lines(staged), edits, notes


def restore_t7(data):
    """Inverse of apply_t7. Returns (new_data, n_restored)."""
    lines = split_lines(data)
    n = 0
    for i, (content, eol) in enumerate(lines):
        m = RESTORE_T7P.match(content) or RESTORE_T7.match(content)
        if m:
            lines[i] = (m.group(1) + m.group(2), eol)
            n += 1
    return join_lines(lines), n


# ---------------------------------------------------------------------------
# T8
# ---------------------------------------------------------------------------

def apply_t8(data, fixes, path_for_msg="<bytes>"):
    """Apply declared single-line corrections. Aborts on any mismatch."""
    lines = split_lines(data)
    edits = []
    for fx in fixes:
        idx = fx["line"] - 1
        if idx < 0 or idx >= len(lines):
            raise RuntimeError("T8 line %d out of range in %s"
                               % (fx["line"], path_for_msg))
        have = lines[idx][0]
        want = fx["expect"].encode("utf-8")
        new = fx["replacement"].encode("utf-8")
        if have == new and have != want:
            # already applied: T8 must be idempotent like the rest of the
            # package, so this is a no-op, not an error.
            continue
        if have != want:
            raise RuntimeError(
                "T8 declaration does not match %s line %d.\n  expected: %r\n"
                "  found   : %r" % (path_for_msg, fx["line"], want, have))
        lines[idx] = (new, lines[idx][1])
        edits.append({
            "line": fx["line"],
            "transform": T8,
            "original": fx["expect"],
            "replacement": fx["replacement"],
            "command": "",
            "reason": fx.get("rationale", "declared whitespace correction"),
        })
    return join_lines(lines), edits


def restore_t8(data, fixes, path_for_msg="<bytes>"):
    lines = split_lines(data)
    for fx in fixes:
        idx = fx["line"] - 1
        have = lines[idx][0]
        want = fx["replacement"].encode("utf-8")
        if have != want:
            raise RuntimeError(
                "T8 restore: %s line %d is not in the transformed state"
                % (path_for_msg, fx["line"]))
        lines[idx] = (fx["expect"].encode("utf-8"), lines[idx][1])
    return join_lines(lines)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def load_targets(path):
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    for key in ("t7_targets", "t8_fixes", "discards"):
        if key not in spec:
            raise KeyError("target spec is missing required key '%s'" % key)
    return spec


def process(root, spec, write):
    """Transform every target. Returns (log, failures)."""
    t8_by_path = {}
    for fx in spec["t8_fixes"]:
        t8_by_path.setdefault(fx["path"], []).append(fx)

    entries = []
    failures = []
    for rel in spec["t7_targets"]:
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            failures.append("missing target: %s" % rel)
            continue
        with open(full, "rb") as fh:
            before = fh.read()

        try:
            after, edits, notes = apply_t7(before, rel)
            t8_edits = []
            if rel in t8_by_path:
                after, t8_edits = apply_t8(after, t8_by_path[rel], rel)
            edits = edits + t8_edits
        except RuntimeError as exc:
            failures.append("%s: %s" % (rel, exc))
            continue

        ok_after = parses(after)
        if not ok_after:
            err = parse_error(after)
            failures.append("%s still does not parse after transform: line %s: %s"
                            % (rel, err[1], err[2]))

        entries.append({
            "path": rel,
            "sha256_before": sha256(before),
            "sha256_after": sha256(after),
            "size_before": len(before),
            "size_after": len(after),
            "n_lines_before": len(split_lines(before)),
            "n_lines_after": len(split_lines(after)),
            "parses_before": parses(before),
            "parses_after": ok_after,
            "n_edits": len(edits),
            "edits": edits,
            "colab_commands": [e["command"] for e in edits if e["command"]],
            "notes": notes,
        })

        if write and after != before:
            with open(full, "wb") as fh:
                fh.write(after)

    log = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "stage": "S0.2",
        "generated": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "root": os.path.abspath(root),
        "files": entries,
        "discards": spec["discards"],
    }
    return log, failures


def verify(root, log):
    """Check the tree against a log. Detects tampering with either."""
    problems = []
    for entry in log["files"]:
        full = os.path.join(root, entry["path"])
        if not os.path.isfile(full):
            problems.append("missing: %s" % entry["path"])
            continue
        with open(full, "rb") as fh:
            data = fh.read()
        if sha256(data) != entry["sha256_after"]:
            problems.append("%s: on-disk hash does not match sha256_after"
                            % entry["path"])
            continue
        if entry["n_lines_before"] != entry["n_lines_after"]:
            problems.append("%s: line count changed (%d -> %d)"
                            % (entry["path"], entry["n_lines_before"],
                               entry["n_lines_after"]))
        # the strong check: the logged edits must invert to the logged input
        back, _ = restore_t7(data)
        t8 = [e for e in entry["edits"] if e["transform"] == T8]
        if t8:
            fixes = [{"line": e["line"], "expect": e["original"],
                      "replacement": e["replacement"]} for e in t8]
            try:
                back = restore_t8(back, fixes, entry["path"])
            except RuntimeError as exc:
                problems.append("%s: %s" % (entry["path"], exc))
                continue
        if sha256(back) != entry["sha256_before"]:
            problems.append(
                "%s: restoring the logged edits does not reproduce "
                "sha256_before -- the log and the file disagree" % entry["path"])
        if not parses(data):
            problems.append("%s: does not parse" % entry["path"])
    return problems


def restore(root, log):
    problems = []
    for entry in log["files"]:
        full = os.path.join(root, entry["path"])
        with open(full, "rb") as fh:
            data = fh.read()
        back, _ = restore_t7(data)
        t8 = [e for e in entry["edits"] if e["transform"] == T8]
        if t8:
            fixes = [{"line": e["line"], "expect": e["original"],
                      "replacement": e["replacement"]} for e in t8]
            back = restore_t8(back, fixes, entry["path"])
        if sha256(back) != entry["sha256_before"]:
            problems.append("%s: restore does not reproduce sha256_before"
                            % entry["path"])
            continue
        with open(full, "wb") as fh:
            fh.write(back)
    return problems


def emit_manifest(log, out_path):
    manifest = {
        "generated_from": "s02_transform_log.json",
        "stage": "S0.2",
        "purpose": (
            "Ordered shell commands neutralised by T7, per file. These scripts "
            "still run in Colab; a bootstrap orchestrator generated from this "
            "manifest can reinstate the installs without reinstating the "
            "unparseable syntax. Generating that orchestrator is out of scope "
            "for S0.2 (see the deferred sub-step in the S0.2 report)."),
        "files": dict(
            (e["path"], e["colab_commands"]) for e in log["files"]
            if e["colab_commands"]),
    }
    with open(out_path, "w", encoding="ascii") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.2 de-Colab transform.")
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--restore", action="store_true")
    mode.add_argument("--emit-colab-manifest", action="store_true")
    args = ap.parse_args(argv)

    if args.verify or args.restore or args.emit_colab_manifest:
        with open(args.log, "r", encoding="utf-8") as fh:
            log = json.load(fh)
        if args.emit_colab_manifest:
            man = emit_manifest(log, args.manifest)
            print("wrote %s (%d files)" % (args.manifest, len(man["files"])))
            return 0
        problems = (verify(args.root, log) if args.verify
                    else restore(args.root, log))
        for p in problems:
            print("  - %s" % p)
        if problems:
            print("\n%s: FAIL -- %d problem(s)"
                  % ("VERIFY" if args.verify else "RESTORE", len(problems)))
            return 1
        print("%s: PASS" % ("VERIFY" if args.verify else "RESTORE"))
        return 0

    spec = load_targets(args.targets)

    if args.apply and os.path.isfile(args.log):
        print("A transform log already exists at %s.\n"
              "The log IS the ledger entry for this transform, so it is not "
              "overwritten silently.\nUse --verify to check the tree against "
              "it, or --restore then delete the log to redo the transform."
              % args.log, file=sys.stderr)
        return 3

    log, failures = process(args.root, spec, write=bool(args.apply))

    for e in log["files"]:
        flag = " " if e["parses_after"] else "!"
        print("%s %-72s %2d edit(s)  %s -> %s"
              % (flag, e["path"], e["n_edits"],
                 "parse:FAIL" if not e["parses_before"] else "parse:ok",
                 "parse:ok" if e["parses_after"] else "parse:FAIL"))
        for n in e["notes"]:
            print("      note: %s" % n)

    if args.apply:
        with open(args.log, "w", encoding="ascii") as fh:
            json.dump(log, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("\nwrote %s" % args.log)

    print("\nfiles          : %d" % len(log["files"]))
    print("edits          : %d" % sum(e["n_edits"] for e in log["files"]))
    print("parse after    : %d/%d"
          % (sum(1 for e in log["files"] if e["parses_after"]), len(log["files"])))
    print("discards (git rm, not performed by this tool): %d"
          % len(log["discards"]))
    for d in log["discards"]:
        print("  - %s" % d["path"])

    if failures:
        print("\nEXIT TEST: FAIL -- %d problem(s):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("\nEXIT TEST: PASS -- every target parses; transform is logged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
