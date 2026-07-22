#!/usr/bin/env python3
"""
asciify.py -- stage S0.3 transform: ASCII sweep and LF normalisation.

    python3 tools/s0_transform/asciify.py --root . --dry-run
    python3 tools/s0_transform/asciify.py --root . --apply
    python3 tools/s0_transform/asciify.py --root . --verify
    python3 tools/s0_transform/asciify.py --root . --restore

TWO TRANSFORMS
--------------
  T9   transliteration. Every byte >= 0x80 is replaced by an ASCII rendering
       declared in s03_translit_map.json. NO PEP-263 cookie is added anywhere.
       Doc 5 E-12 is the argument: once a multi-byte UTF-8 character has
       collapsed to a single cp1252 byte in transit, the declared codec fails
       to decode it and the cookie is worthless. ASCII is a fixed point of
       cp1252, latin-1 and UTF-8 alike, so pure-ASCII source cannot be
       corrupted in transit at all. Cookies already present are left alone;
       removing them would be a second change for no benefit.

  T10  line-ending normalisation, CRLF -> LF.

HONEST CAVEAT, STATED IN THE TOOL AND NOT ONLY IN THE REPORT
------------------------------------------------------------
T9 changes printed output. Status glyphs become tokens ([OK], [FAIL], [WARN]);
decorative emoji are deleted; box-drawing banners become runs of '=' and '-'.
Nothing computational changes, but console output does. This satisfies clause
(2) of the byte-identity rule ONLY because the transform is scripted, declared
and logged -- the script and its log ARE the ledger entry.

EXHAUSTIVENESS IS ENFORCED
--------------------------
A non-ASCII character with no rule is a hard failure naming the character, the
file and the line. The tool never passes an unmapped character through and
never guesses one, because a sweep that silently skips what it does not
recognise leaves exactly the bytes that caused the original problem.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import base64
import datetime
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import unicodedata

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from s0_paths import Resolver  # noqa: E402

_RESOLVERS = {}


def resolved(root, rel):
    """Filesystem path of a declared or logged path, after any T6 move.

    s03_targets.json and s03_transform_log.json are both keyed by the paths
    the files had when S0.3 ran. S0.4 relocates nine of them into
    towards_eeg/. Resolving forward here keeps --verify and --restore honest
    instead of reporting a target as missing.
    """
    R = _RESOLVERS.get(root)
    if R is None:
        R = _RESOLVERS[root] = Resolver(root)
    return R.full(rel)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.dirname(_HERE)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from s0_transform import TOOL_NAME, TOOL_VERSION  # noqa: E402

T9 = "T9"
T10 = "T10"

DEFAULT_MAP = os.path.join(_HERE, "s03_translit_map.json")
DEFAULT_TARGETS = os.path.join(_HERE, "s03_targets.json")
DEFAULT_LOG = os.path.join(_HERE, "s03_transform_log.json")


# ---------------------------------------------------------------------------
# byte-exact line handling (shared conventions with decolab.py)
# ---------------------------------------------------------------------------

def split_lines(data):
    parts = data.split(b"\n")
    out = []
    last = len(parts) - 1
    for i, p in enumerate(parts):
        if i == last:
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


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------

class Translit(object):
    def __init__(self, doc):
        self.ordered = [(r["from"], r["to"]) for r in doc["ordered_rules"]]
        self.single = dict((r["from"], r["to"]) for r in doc["single_rules"])
        self.drop = set(r["from"] for r in doc["drop_decorative"])
        overlap = set(self.single) & self.drop
        if overlap:
            raise ValueError("character is both mapped and dropped: %r" % sorted(overlap))
        for _src, dst in self.ordered + list(self.single.items()):
            if any(ord(ch) > 127 for ch in dst):
                raise ValueError("replacement is not ASCII: %r" % dst)

    def apply(self, text):
        """Returns (new_text, used_rules, unmapped)."""
        used = []
        for src, dst in self.ordered:
            if src in text:
                used.append(("ordered", src, dst, text.count(src)))
                text = text.replace(src, dst)
        out = []
        unmapped = []
        for ch in text:
            if ord(ch) < 128:
                out.append(ch)
            elif ch in self.single:
                out.append(self.single[ch])
                used.append(("single", ch, self.single[ch], 1))
            elif ch in self.drop:
                used.append(("drop", ch, "", 1))
            else:
                unmapped.append(ch)
                out.append(ch)
        return "".join(out), used, unmapped


# ---------------------------------------------------------------------------
# per-file transform
# ---------------------------------------------------------------------------

def transform_file(data, translit, do_t9=True, do_t10=True):
    """Returns (new_data, edits, unmapped). Raises on a non-UTF-8 input."""
    text_lines = split_lines(data)
    edits = []
    unmapped_all = []
    out_lines = []
    for i, (content, eol) in enumerate(text_lines, start=1):
        new_content, new_eol = content, eol
        rules = []

        if do_t9 and any(b > 127 for b in content):
            try:
                s = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError("line %d is not valid UTF-8: %s" % (i, exc))
            s2, used, unmapped = translit.apply(s)
            if unmapped:
                for ch in unmapped:
                    unmapped_all.append((i, ch))
            new_content = s2.encode("ascii", "strict") if not unmapped \
                else s2.encode("utf-8")
            if used:
                rules.append(T9)

        if do_t10 and new_eol == b"\r\n":
            new_eol = b"\n"
            rules.append(T10)

        if rules:
            edits.append({
                "line": i,
                "transforms": rules,
                "original_b64": base64.b64encode(content + eol).decode("ascii"),
                "replacement_b64": base64.b64encode(new_content + new_eol).decode("ascii"),
            })
        out_lines.append((new_content, new_eol))

    return join_lines(out_lines), edits, unmapped_all


def assert_clean(data, path):
    """Write-time postcondition. The CRLF bug got through because nothing
    checked the bytes a tool had just produced."""
    bad = [(i + 1, hex(b)) for i, b in enumerate(data) if b > 127]
    if bad:
        raise RuntimeError("%s: output still contains non-ASCII at byte offsets %r"
                           % (path, bad[:5]))
    if b"\r\n" in data:
        raise RuntimeError("%s: output still contains CRLF" % path)


# ---------------------------------------------------------------------------
# target selection
# ---------------------------------------------------------------------------

def tracked_files(root):
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                         cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("git ls-tree failed: %s" % out.stderr.strip())
    return out.stdout.splitlines()


def select_targets(root, spec):
    files = tracked_files(root)
    chosen = []
    for p in files:
        if not any(fnmatch.fnmatch(p, g) for g in spec["include_globs"]):
            continue
        if any(fnmatch.fnmatch(p, g) for g in spec["exclude_globs"]):
            continue
        if p in spec["exclude_paths"]:
            continue
        chosen.append(p)
    return sorted(chosen)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def process(root, spec, translit, write):
    entries, failures = [], []
    for rel in select_targets(root, spec):
        full = resolved(root, rel)
        if not os.path.isfile(full):
            failures.append("target missing from the working tree: %s" % rel)
            continue
        with open(full, "rb") as fh:
            before = fh.read()
        if not any(b > 127 for b in before) and b"\r\n" not in before:
            continue                                   # already clean: no-op
        try:
            after, edits, unmapped = transform_file(before, translit)
        except RuntimeError as exc:
            failures.append("%s: %s" % (rel, exc))
            continue
        if unmapped:
            for ln, ch in unmapped[:20]:
                failures.append(
                    "%s line %d: no rule for %r (U+%04X %s) -- add it to the map"
                    % (rel, ln, ch, ord(ch), unicodedata.name(ch, "<unnamed>")))
            continue
        try:
            assert_clean(after, rel)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue

        entries.append({
            "path": rel,
            "sha256_before": sha256(before),
            "sha256_after": sha256(after),
            "n_lines": len(split_lines(before)),
            "n_edits": len(edits),
            "n_nonascii_before": sum(1 for b in before if b > 127),
            "n_crlf_before": before.count(b"\r\n"),
            "edits": edits,
        })
        if write:
            with open(full, "wb") as fh:
                fh.write(after)

    log = {
        "tool": TOOL_NAME, "version": TOOL_VERSION, "stage": "S0.3",
        "generated": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "transforms": {T9: "transliteration to ASCII, no cookie added",
                       T10: "CRLF -> LF"},
        "files": entries,
    }
    return log, failures


def verify(root, log):
    problems = []
    for e in log["files"]:
        full = resolved(root, e["path"])
        if not os.path.isfile(full):
            problems.append("missing: %s" % e["path"])
            continue
        with open(full, "rb") as fh:
            data = fh.read()
        if sha256(data) != e["sha256_after"]:
            problems.append("%s: on-disk hash does not match sha256_after" % e["path"])
            continue
        try:
            assert_clean(data, e["path"])
        except RuntimeError as exc:
            problems.append(str(exc))
        if len(split_lines(data)) != e["n_lines"]:
            problems.append("%s: line count changed" % e["path"])
        back = restore_bytes(data, e)
        if sha256(back) != e["sha256_before"]:
            problems.append("%s: logged edits do not invert to sha256_before"
                            % e["path"])
    return problems


def restore_bytes(data, entry):
    lines = split_lines(data)
    for ed in entry["edits"]:
        orig = base64.b64decode(ed["original_b64"])
        i = ed["line"] - 1
        if orig.endswith(b"\r\n"):
            lines[i] = (orig[:-2], b"\r\n")
        elif orig.endswith(b"\n"):
            lines[i] = (orig[:-1], b"\n")
        else:
            lines[i] = (orig, b"")
    return join_lines(lines)


def restore(root, log):
    problems = []
    for e in log["files"]:
        full = resolved(root, e["path"])
        with open(full, "rb") as fh:
            data = fh.read()
        back = restore_bytes(data, e)
        if sha256(back) != e["sha256_before"]:
            problems.append("%s: restore does not reproduce sha256_before" % e["path"])
            continue
        with open(full, "wb") as fh:
            fh.write(back)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.3 ASCII sweep and LF normalisation.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--log", default=DEFAULT_LOG)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--restore", action="store_true")
    args = ap.parse_args(argv)

    with open(args.map, "r", encoding="utf-8") as fh:
        translit = Translit(json.load(fh))
    with open(args.targets, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    if args.verify or args.restore:
        with open(args.log, "r", encoding="utf-8") as fh:
            log = json.load(fh)
        problems = verify(root=args.root, log=log) if args.verify \
            else restore(args.root, log)
        for p in problems:
            print("  - %s" % p)
        label = "VERIFY" if args.verify else "RESTORE"
        print("%s: %s" % (label, "FAIL" if problems else "PASS"))
        return 1 if problems else 0

    if args.apply and os.path.isfile(args.log):
        print("A transform log already exists at %s; it is the ledger entry for "
              "this transform and is not overwritten silently. Use --verify, or "
              "--restore then delete it." % args.log, file=sys.stderr)
        return 3

    log, failures = process(args.root, spec, translit, write=bool(args.apply))

    for e in log["files"]:
        print("  %-70s %5d edit(s)  %6d non-ASCII  %5d CRLF"
              % (e["path"][:70], e["n_edits"], e["n_nonascii_before"],
                 e["n_crlf_before"]))
    print("\nfiles touched   : %d" % len(log["files"]))
    print("lines edited    : %d" % sum(e["n_edits"] for e in log["files"]))
    print("non-ASCII bytes : %d" % sum(e["n_nonascii_before"] for e in log["files"]))
    print("CRLF lines      : %d" % sum(e["n_crlf_before"] for e in log["files"]))

    if args.apply:
        with open(args.log, "w", encoding="ascii") as fh:
            json.dump(log, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote %s" % args.log)

    if failures:
        print("\nEXIT TEST: FAIL -- %d problem(s):" % len(failures))
        for f in failures[:30]:
            print("  - %s" % f)
        return 1
    print("\nEXIT TEST: PASS -- every target is pure ASCII and LF-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
