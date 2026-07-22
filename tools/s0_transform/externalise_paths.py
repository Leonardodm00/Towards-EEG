#!/usr/bin/env python3
"""
externalise_paths.py -- apply transform T11 for sub-step S0.6.

    python3 tools/s0_transform/externalise_paths.py --root . --apply
    python3 tools/s0_transform/externalise_paths.py --root . --check
    python3 tools/s0_transform/externalise_paths.py --root . --invert-check

WHAT T11 IS
-----------
Declared externalisation of a configuration path literal: a string literal in a
keep-verdict module is replaced by a call to towards_eeg.config.resolve(key),
where the shipped default towards_eeg/config/paths.json maps key to the
literal's exact prior bytes. This is byte-identity rule clause (2) -- the
output of a stated, reproducible mechanical transform of an identified
ancestor -- and clause (2) is the one that requires the transform be STATED,
which is what tools/s0_transform/s06_targets.json does and what this tool
enforces.

WHY THE LOG IS INVERTIBLE
-------------------------
TEEG_10 section 3.3 requires the transform be provably behaviour-preserving.
Comparing resolve(key) against the recorded literal proves the VALUE is
preserved but says nothing about whether anything ELSE in the file changed at
the same time. The stronger statement, and the one this tool produces, is that
the log carries enough information to invert the transform, and that applying
the inverse to the current bytes reproduces sha256_before exactly. If any byte
outside a declared edit had changed, the inverse would not reproduce the prior
hash. That is a proof about the whole file, not about the five lines the tool
happens to have looked at.

POSTCONDITION ON BYTES, NOT ON INTENT
-------------------------------------
Doc 7 section 2 generalised the ledger-CRLF defect into a rule: assert your
postcondition on the bytes you just wrote. This tool re-reads every file it
touches and raises if a byte >= 0x80 appears, if a CRLF appears, if the result
does not parse, or if the recorded inverse fails to reproduce sha256_before.
The S0.2/S0.3 defect got through because nothing checked the artefacts the
tools themselves produced.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import base64
import datetime
import hashlib
import json
import os
import sys

MARKER = "#S0.6:T11"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def b64(text):
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def unb64(text):
    return base64.b64decode(text.encode("ascii")).decode("utf-8")


def read_lines(full):
    """Return (list_of_lines_without_terminator, trailing_newline_flag).

    Split on "\\n" only. The tree is LF-only by S0.3 and .gitattributes, and a
    universal-newline read would silently normalise a CRLF that ought to be a
    failure rather than a fixup.
    """
    with open(full, "rb") as fh:
        raw = fh.read()
    if b"\r\n" in raw:
        raise SystemExit("CRLF in %s -- S0.3 should have removed it; stop" % full)
    text = raw.decode("ascii")          # strict: S0.3 made the tree ASCII
    ends_nl = text.endswith("\n")
    lines = text.split("\n")
    if ends_nl:
        lines = lines[:-1]
    return lines, ends_nl


def join_lines(lines, ends_nl):
    return "\n".join(lines) + ("\n" if ends_nl else "")


def replace_literal(line, literal, key):
    """Replace exactly one quoted occurrence of `literal` with resolve(key).

    Returns the new line. Raises if the literal is absent, or if it occurs
    more than once on the line under either quoting style -- an ambiguous line
    must be declared as two occurrences, not silently half-transformed.
    """
    candidates = ["'" + literal + "'", '"' + literal + '"']
    hits = [c for c in candidates if c in line]
    if not hits:
        raise SystemExit("literal %r not found (quoted) in line: %r"
                         % (literal, line))
    if len(hits) > 1:
        raise SystemExit("literal %r appears under both quoting styles: %r"
                         % (literal, line))
    quoted = hits[0]
    if line.count(quoted) != 1:
        raise SystemExit("literal %r occurs %d times on one line; declare each "
                         "occurrence separately: %r"
                         % (literal, line.count(quoted), line))
    return line.replace(quoted, "resolve('%s')" % key, 1)


def plan_file(root, spec):
    """Compute the edit plan for one declared file, without writing anything.

    Returns a dict carrying everything needed both to apply and to invert.
    """
    rel = spec["path"]
    full = os.path.join(root, rel)
    lines, ends_nl = read_lines(full)
    before_bytes = join_lines(lines, ends_nl).encode("ascii")

    edits = []
    for occ in spec["occurrences"]:
        n = occ["line"]
        if not (1 <= n <= len(lines)):
            raise SystemExit("%s: declared line %d is out of range (%d lines)"
                             % (rel, n, len(lines)))
        original = lines[n - 1]
        replacement = replace_literal(original, occ["literal"], occ["key"])
        if MARKER not in replacement:
            replacement = replacement + "  " + MARKER
        edits.append({
            "line": n,
            "key": occ["key"],
            "literal": occ["literal"],
            "original_b64": b64(original),
            "replacement_b64": b64(replacement),
            "transform": "T11",
        })

    ins = spec["insert_import"]
    anchor = ins["after_line"]
    if lines[anchor - 1] != ins["expect_line_text"]:
        raise SystemExit("%s: line %d is %r, declaration expects %r"
                         % (rel, anchor, lines[anchor - 1],
                            ins["expect_line_text"]))
    insertion = {
        "after_line": anchor,
        "text_b64": b64(ins["text"]),
        "transform": "T11",
        "rationale": "resolve() must be in scope at the call sites; the import "
                     "is part of T11 and is recorded so the transform inverts.",
    }

    # Apply: literals first, in pre-edit numbering, then the insertion.
    out = list(lines)
    for e in edits:
        out[e["line"] - 1] = unb64(e["replacement_b64"])
    out.insert(anchor, unb64(insertion["text_b64"]))
    after_text = join_lines(out, ends_nl)

    return {
        "path": rel,
        "sha256_before": sha256_bytes(before_bytes),
        "sha256_after": sha256_bytes(after_text.encode("ascii")),
        "n_lines_before": len(lines),
        "n_lines_after": len(out),
        "insert_import": insertion,
        "edits": edits,
        "ends_with_newline": ends_nl,
        "_after_text": after_text,
    }


def invert(after_text, entry):
    """Reconstruct the pre-transform text from the post-transform text.

    The inverse of T11: remove the inserted import line, then restore each
    edited line from its recorded original. Line numbers in the log are
    PRE-edit, so the insertion is removed first and the rest then line up.
    """
    ends_nl = entry["ends_with_newline"]
    lines = after_text.split("\n")
    if ends_nl:
        lines = lines[:-1]
    idx = entry["insert_import"]["after_line"]          # 0-based index of insert
    expect = unb64(entry["insert_import"]["text_b64"])
    if lines[idx] != expect:
        raise SystemExit("inverse: line %d is %r, expected the inserted import %r"
                         % (idx + 1, lines[idx], expect))
    del lines[idx]
    for e in entry["edits"]:
        n = e["line"]
        if lines[n - 1] != unb64(e["replacement_b64"]):
            raise SystemExit("inverse: line %d is %r, expected %r"
                             % (n, lines[n - 1], unb64(e["replacement_b64"])))
        lines[n - 1] = unb64(e["original_b64"])
    return join_lines(lines, ends_nl)


def assert_postcondition(full, entry):
    """Re-read what was just written and check it on the BYTES.

    Doc 7 section 2. Checks: pure ASCII, no CRLF, hash matches the plan, the
    file still parses, the marker is present once per edit, and the recorded
    inverse reproduces sha256_before.
    """
    with open(full, "rb") as fh:
        raw = fh.read()
    bad = [(i + 1, hex(b)) for i, b in enumerate(raw) if b > 127]
    if bad:
        raise SystemExit("POSTCONDITION: non-ASCII in %s at %r" % (full, bad[:5]))
    if b"\r\n" in raw:
        raise SystemExit("POSTCONDITION: CRLF in %s" % full)
    got = sha256_bytes(raw)
    if got != entry["sha256_after"]:
        raise SystemExit("POSTCONDITION: %s hashes %s, planned %s"
                         % (full, got, entry["sha256_after"]))
    text = raw.decode("ascii")
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise SystemExit("POSTCONDITION: %s no longer parses: %s" % (full, exc))
    n_marker = text.count(MARKER)
    if n_marker != len(entry["edits"]) + 1:
        raise SystemExit("POSTCONDITION: %s carries %d marker(s), expected %d"
                         % (full, n_marker, len(entry["edits"]) + 1))
    back = invert(text, entry)
    if sha256_bytes(back.encode("ascii")) != entry["sha256_before"]:
        raise SystemExit("POSTCONDITION: %s does not invert to sha256_before; "
                         "something changed outside a declared edit" % full)


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.6 transform T11.")
    ap.add_argument("--root", default=".")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true",
                      help="plan only; write nothing")
    mode.add_argument("--invert-check", action="store_true",
                      help="verify the existing log inverts the tree on disk")
    args = ap.parse_args(argv)

    tdir = os.path.join(args.root, "tools", "s0_transform")
    with open(os.path.join(tdir, "s06_targets.json"), "r", encoding="ascii") as fh:
        targets = json.load(fh)
    log_path = os.path.join(tdir, "s06_transform_log.json")

    if args.invert_check:
        with open(log_path, "r", encoding="ascii") as fh:
            doc = json.load(fh)
        for entry in doc["files"]:
            full = os.path.join(args.root, entry["path"])
            with open(full, "rb") as fh:
                raw = fh.read()
            back = invert(raw.decode("ascii"), entry)
            ok = sha256_bytes(back.encode("ascii")) == entry["sha256_before"]
            print("%-52s inverse -> %s  %s"
                  % (entry["path"], entry["sha256_before"][:12],
                     "OK" if ok else "MISMATCH"))
            if not ok:
                return 1
        print("\nT11 INVERTS: every edited file reconstructs its prior bytes")
        return 0

    plans = [plan_file(args.root, spec) for spec in targets["files"]]
    for p in plans:
        print("%-52s %d edit(s)  %s -> %s"
              % (p["path"], len(p["edits"]), p["sha256_before"][:12],
                 p["sha256_after"][:12]))
    if args.check:
        print("\nCHECK ONLY: nothing written")
        return 0

    for p in plans:
        full = os.path.join(args.root, p["path"])
        with open(full, "w", encoding="ascii", newline="") as fh:
            fh.write(p["_after_text"])

    doc = {
        "stage": "S0.6",
        "transform_id": "T11",
        "declaration": targets["declaration"],
        "scope_decision": targets["scope_decision"],
        "tool": "tools/s0_transform/externalise_paths.py",
        "version": 1,
        "generated": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_file": "towards_eeg/config/paths.json",
        "files": [],
    }
    for p in plans:
        entry = dict((k, v) for k, v in p.items() if not k.startswith("_"))
        doc["files"].append(entry)
        assert_postcondition(os.path.join(args.root, p["path"]), entry)

    with open(log_path, "w", encoding="ascii", newline="") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("\nwrote %s" % log_path)
    print("POSTCONDITION HELD on every file: ASCII, LF, parses, inverts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
