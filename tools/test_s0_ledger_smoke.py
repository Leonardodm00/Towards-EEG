#!/usr/bin/env python3
"""
test_s0_ledger_smoke.py -- smoke test for the S0.0 ledger builder.

Runs standalone (no pytest needed) and is also pytest-discoverable:

    python3 tools/test_s0_ledger_smoke.py      # standalone, prints a table
    python3 -m pytest tools/ -q                # if pytest is available

What it proves
--------------
Every check below builds a SYNTHETIC fixture tree whose correct answer is
known by construction, so a failure localises to one function rather than to
"the ledger looks wrong".  The checks are ordered from most primitive to most
composed:

    1  known-answer SHA-256 (NIST vectors)         scan.sha256_bytes
    2  derived-hash lattice                        scan.measure_file
    3  line-ending census                          scan.measure_file
    4  non-ASCII count and PEP 263 cookie          scan.measure_file
    5  parse failure on a Colab magic              scan.measure_file
    6  top-level definition extraction             scan.measure_file
    7  O7: later definition is the effective one   analyse.find_shadowed_defs
    8  duplicate class names across files          analyse.find_duplicate_classes
    9  relationship lattice on real pairs          analyse.measured_relationship
    10 scope assignment, first prefix wins         analyse.assign_scope
    11 payload detection                           analyse.is_payload
    12 every row gets a final verdict              analyse.build_rows
    13 a FALSE declaration is caught               analyse.build_rows   <-- key
    14 CSV round trip                              render.write_csv/read_csv
    15 determinism across two runs                 render.write_markdown
    16 ASCII purity of the tool sources            build_ledger.self_check

Check 13 is the one that matters most.  A ledger that merely restates the
document it was built from is worthless; it has to be able to contradict it.
Check 13 declares a relationship that is false, and asserts the builder
notices, refuses the row, and fails the exit test.

Debugging
---------
Each check prints its own diagnostic on failure.  To inspect a fixture tree
instead of deleting it, run with KEEP=1:

    KEEP=1 python3 tools/test_s0_ledger_smoke.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_ledger                                    # noqa: E402
from s0_ledger import analyse, render, scan            # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

# ancestor: LF, no trailing-newline oddity, one shadowed pair
ANCESTOR_PY = (
    b"def alpha():\n"
    b"    return 1\n"
    b"\n"
    b"def beta():\n"
    b"    return 2\n"
    b"\n"
    b"def alpha():\n"
    b"    return 3\n"
)

COLAB_PY = b"!pip install numpy\nimport numpy as np\n"

NONASCII_PY = "# -*- coding: utf-8 -*-\n# box: \u2500\u2500 check: \u2705\nx = 1\n".encode("utf-8")

CLASSY_A = b"class Widget:\n    pass\n"
CLASSY_B = b"class Widget:\n    pass\n\nclass Other:\n    pass\n"


def build_fixture(base):
    """Create a synthetic repository plus a local working directory.

    Returns (repo_dir, local_dir, spec_path).
    """
    repo = base / "repo"
    local = base / "local"
    (repo / "pkg").mkdir(parents=True)
    (repo / "data").mkdir(parents=True)
    (repo / "notebooks").mkdir(parents=True)
    local.mkdir()

    # ancestors inside the repository
    (repo / "pkg" / "ancestor.py").write_bytes(ANCESTOR_PY)
    (repo / "pkg" / "eol_ancestor.py").write_bytes(ANCESTOR_PY)
    (repo / "pkg" / "nl_ancestor.py").write_bytes(ANCESTOR_PY)
    (repo / "pkg" / "edited_ancestor.py").write_bytes(ANCESTOR_PY)
    (repo / "pkg" / "classes_a.py").write_bytes(CLASSY_A)
    (repo / "notebooks" / "classes_b.py").write_bytes(CLASSY_B)
    (repo / "notebooks" / "colab.py").write_bytes(COLAB_PY)
    (repo / "notebooks" / "unicode.py").write_bytes(NONASCII_PY)
    (repo / "data" / "payload.zip").write_bytes(b"PK\x03\x04not-a-real-zip")
    (repo / "data" / "doc.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    (repo / "dead.py").write_bytes(b"def leftover(:\n")          # deliberate SyntaxError

    # local working files, one per relationship in the lattice
    (local / "identical.py").write_bytes(ANCESTOR_PY)
    (local / "eol.py").write_bytes(ANCESTOR_PY.replace(b"\n", b"\r\n"))
    (local / "trailing.py").write_bytes(ANCESTOR_PY + b"\n")
    (local / "edited.py").write_bytes(ANCESTOR_PY + b"\ndef gamma():\n    return 4\n")

    (repo / "toolshed").mkdir(parents=True)
    (repo / "toolshed" / "helper.py").write_bytes(b"x = 1\n")
    (repo / "STANDALONE.md").write_bytes(b"# note\n")

    spec = {
        "schema_version": 1,
        "snapshot": "fixture",
        "snapshot_date": "1970-01-01",
        "transforms": {"T0": "none", "T1": "trailing newline", "T2": "line endings"},
        "scopes": [["orchestrator", ["pkg/"]], ["notebooks", ["notebooks/"]]],
        "new_infrastructure": ["toolshed/"],
        "working_branch": [
            {"local": "identical.py", "ancestor": "pkg/ancestor.py",
             "expected_relationship": "identical", "stage": "S0.1", "rationale": "fixture"},
            {"local": "eol.py", "ancestor": "pkg/eol_ancestor.py",
             "expected_relationship": "mechanical", "stage": "S0.1", "rationale": "fixture"},
            {"local": "trailing.py", "ancestor": "pkg/nl_ancestor.py",
             "expected_relationship": "mechanical", "stage": "S0.1", "rationale": "fixture"},
            {"local": "edited.py", "ancestor": "pkg/edited_ancestor.py",
             "expected_relationship": "working_branch_edit", "stage": "S0.1",
             "rationale": "fixture"},
        ],
        "discards": [{"path": "dead.py", "stage": "S0.2", "rationale": "fixture dead branch"}],
        "binary_payload": {"extensions": [".zip", ".pdf"], "paths": [],
                           "rationale": "fixture payload"},
    }
    spec_path = base / "ancestors.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="ascii")
    return repo, local, spec_path


def _rows(repo, local, spec):
    repo_recs = scan.scan_tree(repo)
    pairs = [(local / e["local"], e["local"]) for e in spec["working_branch"]]
    local_recs = scan.scan_files(pairs)
    return repo_recs, local_recs, analyse.build_rows(repo_recs, local_recs, spec)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_01_known_answer_sha256(ctx):
    assert scan.sha256_bytes(b"") == SHA256_EMPTY, "empty-string vector wrong"
    assert scan.sha256_bytes(b"abc") == SHA256_ABC, "abc vector wrong"
    p = ctx["base"] / "vector.bin"
    p.write_bytes(b"abc")
    assert scan.sha256_file(p) == SHA256_ABC, "chunked file hash != in-memory hash"


def check_02_derived_hash_lattice(ctx):
    base = ctx["base"]
    (base / "a.txt").write_bytes(b"x\ny\n")
    (base / "b.txt").write_bytes(b"x\r\ny\r\n")
    (base / "c.txt").write_bytes(b"x\ny\n\n")
    a = scan.measure_file(base / "a.txt", "a.txt")
    b = scan.measure_file(base / "b.txt", "b.txt")
    c = scan.measure_file(base / "c.txt", "c.txt")
    assert a.sha256 != b.sha256, "CRLF variant must differ at byte level"
    assert a.sha256_lf == b.sha256_lf, "CRLF variant must match after LF normalisation"
    assert a.sha256_lf != c.sha256_lf, "extra trailing newline must differ after LF norm"
    assert a.sha256_stripped == c.sha256_stripped, "trailing newlines must strip equal"


def check_03_line_ending_census(ctx):
    r = scan.measure_file(ctx["local"] / "eol.py", "eol.py")
    assert r.n_crlf == 8, "expected 8 CRLF, got %d" % r.n_crlf
    assert r.n_lf == 8, "every CRLF contains an LF; got %d" % r.n_lf
    r2 = scan.measure_file(ctx["repo"] / "pkg" / "ancestor.py", "ancestor.py")
    assert r2.n_crlf == 0, "ancestor must be pure LF"


def check_04_nonascii_and_cookie(ctx):
    r = scan.measure_file(ctx["repo"] / "notebooks" / "unicode.py", "notebooks/unicode.py")
    assert r.n_nonascii > 0, "non-ASCII bytes not counted"
    assert r.has_cookie is True, "PEP 263 cookie on line 1 not detected"
    r2 = scan.measure_file(ctx["repo"] / "pkg" / "ancestor.py", "pkg/ancestor.py")
    assert r2.n_nonascii == 0 and r2.has_cookie is False, "false positive on clean file"


def check_05_parse_failure(ctx):
    r = scan.measure_file(ctx["repo"] / "notebooks" / "colab.py", "notebooks/colab.py")
    assert r.parses is False, "a leading '!' magic must not parse as Python"
    assert r.parse_error, "parse_error must be populated when parses is False"
    r2 = scan.measure_file(ctx["repo"] / "dead.py", "dead.py")
    assert r2.parses is False, "malformed def must not parse"


def check_06_toplevel_defs(ctx):
    r = scan.measure_file(ctx["repo"] / "pkg" / "ancestor.py", "pkg/ancestor.py")
    names = [n for n, _, _ in r.toplevel_defs]
    assert names == ["alpha", "beta", "alpha"], "got %r" % (names,)
    assert all(k == "def" for _, _, k in r.toplevel_defs), "kind must be 'def'"


def check_07_o7_effective_is_last(ctx):
    recs = scan.scan_tree(ctx["repo"])
    o7 = analyse.find_shadowed_defs(recs)
    hits = [s for s in o7 if s.name == "alpha" and s.path == "pkg/ancestor.py"]
    assert len(hits) == 1, "expected exactly one shadowed pair for alpha, got %d" % len(hits)
    s = hits[0]
    assert s.lines == (1, 7), "definition lines wrong: %r" % (s.lines,)
    assert s.effective_line == 7, "the LATER definition must win at module scope"
    assert s.dead_lines == (1,), "dead lines wrong: %r" % (s.dead_lines,)


def check_08_duplicate_classes(ctx):
    recs = scan.scan_tree(ctx["repo"])
    dup = analyse.find_duplicate_classes(recs)
    assert "Widget" in dup, "Widget defined in two files but not reported"
    assert len(dup["Widget"]) == 2, "expected 2 locations, got %r" % (dup["Widget"],)
    assert "Other" not in dup, "single-location class must not be reported"


def check_09_relationship_lattice(ctx):
    repo_recs, local_recs, _ = _rows(ctx["repo"], ctx["local"], ctx["spec"])
    L = {r.path: r for r in local_recs}
    R = {r.path: r for r in repo_recs}
    cases = [
        ("identical.py", "pkg/ancestor.py", "identical", "T0"),
        ("eol.py", "pkg/eol_ancestor.py", "mechanical", "T2"),
        ("trailing.py", "pkg/nl_ancestor.py", "mechanical", "T1"),
        ("edited.py", "pkg/edited_ancestor.py", "working_branch_edit", ""),
    ]
    for loc, anc, want_rel, want_t in cases:
        rel, tid, note = analyse.measured_relationship(L[loc], R[anc])
        assert rel == want_rel, "%s: want %s got %s (%s)" % (loc, want_rel, rel, note)
        assert tid == want_t, "%s: want transform %r got %r" % (loc, want_t, tid)


def check_10_scope_first_prefix_wins(ctx):
    scopes = [("orchestrator", ("pkg/",)), ("notebooks", ("notebooks/",))]
    assert analyse.assign_scope("pkg/x.py", scopes) == "orchestrator"
    assert analyse.assign_scope("notebooks/x.py", scopes) == "notebooks"
    assert analyse.assign_scope("elsewhere/x.py", scopes) == "colab", "default not applied"
    # order sensitivity: a broader prefix declared first must shadow a later one
    ordered = [("first", ("a/",)), ("second", ("a/b/",))]
    assert analyse.assign_scope("a/b/c.py", ordered) == "first", "declaration order ignored"


def check_11_payload_detection(ctx):
    rule = {"extensions": [".zip", ".pdf"], "paths": ["logs/job.o1"]}
    assert analyse.is_payload("data/payload.zip", rule) is True
    assert analyse.is_payload("data/DOC.PDF", rule) is True, "extension match must be case-insensitive"
    assert analyse.is_payload("logs/job.o1", rule) is True
    assert analyse.is_payload("pkg/ancestor.py", rule) is False


def check_12_every_row_has_final_verdict(ctx):
    _, _, (rows, problems) = _rows(ctx["repo"], ctx["local"], ctx["spec"])
    bad = [r.path for r in rows if r.verdict not in analyse.VERDICTS]
    assert not bad, "non-final verdicts: %r" % (bad,)
    assert not problems, "unexpected problems on a consistent fixture: %r" % (problems,)
    paths = {r.path for r in rows}
    assert "identical.py" in paths, "working files missing from the ledger"
    assert "dead.py" in paths, "discarded file must still appear, with verdict discard"
    disc = [r for r in rows if r.path == "dead.py"][0]
    assert disc.verdict == "discard", "declared discard not honoured"
    zip_row = [r for r in rows if r.path == "data/payload.zip"][0]
    assert zip_row.verdict == "discard" and zip_row.stage == "S0.7", "payload not routed to S0.7"


def check_12b_new_infrastructure_not_misclassified(ctx):
    """Regression check.

    Caught for real against the actual Towards-EEG repository: the S0.0
    tooling itself (tools/) was scanned as pre-existing content and defaulted
    to scope 'colab', verdict 'retain' -- as if it were a Colab script nobody
    had reviewed, rather than the new infrastructure clause 3 of the
    byte-identity rule describes. A file under a declared new_infrastructure
    prefix must get verdict 'new', not fall through to the scope default.
    """
    _, _, (rows, _) = _rows(ctx["repo"], ctx["local"], ctx["spec"])
    by_path = {r.path: r for r in rows}

    helper = by_path["toolshed/helper.py"]
    assert helper.verdict == "new", "new-infra file got verdict %r, want 'new'" % helper.verdict
    assert helper.scope == "infrastructure", (
        "new-infra file got scope %r, want 'infrastructure'" % helper.scope
    )
    assert helper.stage == "S0.0", "new-infra file got stage %r, want 'S0.0'" % helper.stage

    # a file NOT under the declared prefix must be unaffected
    standalone = by_path["STANDALONE.md"]
    assert standalone.verdict != "new", "unrelated file wrongly classified as new infrastructure"


def check_13_false_declaration_is_caught(ctx):
    """THE IMPORTANT ONE.

    Declare that 'edited.py' is byte-identical to its ancestor, which it is
    not.  The builder must notice, mark the row 'unknown', and fail the exit
    test.  A ledger that cannot contradict the document that produced it is
    not evidence.
    """
    spec = json.loads(json.dumps(ctx["spec"]))          # deep copy
    for e in spec["working_branch"]:
        if e["local"] == "edited.py":
            e["expected_relationship"] = "identical"
    _, _, (rows, problems) = _rows(ctx["repo"], ctx["local"], spec)
    assert problems, "a false declaration produced no problem report"
    assert any("MISMATCH" in p for p in problems), "problem not labelled MISMATCH: %r" % (problems,)
    row = [r for r in rows if r.path == "edited.py"][0]
    assert row.verdict == "unknown", "mismatched row must not receive a final verdict"


def check_14_csv_round_trip(ctx):
    _, _, (rows, _) = _rows(ctx["repo"], ctx["local"], ctx["spec"])
    out = ctx["base"] / "rt.csv"
    render.write_csv(rows, out)
    back = render.read_csv(out)
    assert len(back) == len(rows), "row count changed on round trip"
    assert list(back[0].keys()) == render.CSV_COLUMNS, "column set changed"
    for r, d in zip(rows, back):
        assert d["path"] == r.path, "row order changed on round trip"
        assert d["sha256_pre_s0"] == r.sha256_pre_s0, "hash mangled on round trip"
        assert d["verdict"] == r.verdict, "verdict mangled on round trip"


def check_15_determinism(ctx):
    meta = {"generated": "fixed", "tool": "t", "version": "0", "snapshot": "s",
            "snapshot_date": "d", "root": "r", "n_repo": 0, "n_local": 0}
    outs = []
    for i in (1, 2):
        repo_recs, local_recs, (rows, problems) = _rows(ctx["repo"], ctx["local"], ctx["spec"])
        o7 = analyse.find_shadowed_defs(repo_recs)
        dup = analyse.find_duplicate_classes(repo_recs)
        p = ctx["base"] / ("det%d.md" % i)
        render.write_markdown(rows, o7, dup, problems, meta, p)
        outs.append(p.read_bytes())
    assert outs[0] == outs[1], "two runs over the same tree produced different bytes"


def check_16_tool_sources_are_ascii(ctx):
    tools = Path(__file__).resolve().parent
    offenders = []
    for p in sorted(tools.rglob("*.py")):
        data = p.read_bytes()
        if any(b > 127 for b in data):
            offenders.append(p.name)
    assert not offenders, "non-ASCII bytes in tool sources: %r" % (offenders,)
    assert build_ledger.self_check(tools) is True, "self_check disagrees with the byte scan"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _collect():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("check_")]


def run_all():
    base = Path(tempfile.mkdtemp(prefix="s0_ledger_smoke_"))
    keep = os.environ.get("KEEP") == "1"
    try:
        repo, local, spec_path = build_fixture(base)
        ctx = {
            "base": base,
            "repo": repo,
            "local": local,
            "spec": json.loads(spec_path.read_text(encoding="ascii")),
        }
        results = []
        for name, fn in _collect():
            try:
                fn(ctx)
                results.append((name, "PASS", ""))
            except AssertionError as exc:
                results.append((name, "FAIL", str(exc)))
            except Exception as exc:                       # noqa: BLE001
                results.append((name, "ERROR", "%s: %s" % (type(exc).__name__, exc)))

        width = max(len(n) for n, _, _ in results)
        print("")
        print("S0.0 ledger builder -- smoke test")
        print("fixture tree: %s" % base)
        print("-" * (width + 30))
        for name, status, msg in results:
            print("%-*s  %-5s  %s" % (width, name, status, msg))
        print("-" * (width + 30))
        n_pass = sum(1 for _, s, _ in results if s == "PASS")
        print("%d/%d checks passed" % (n_pass, len(results)))
        if keep:
            print("KEEP=1 -- fixture retained at %s" % base)
        return 0 if n_pass == len(results) else 1
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)


# pytest entry points ------------------------------------------------------

def test_s0_ledger_smoke():
    assert run_all() == 0


if __name__ == "__main__":
    sys.exit(run_all())
