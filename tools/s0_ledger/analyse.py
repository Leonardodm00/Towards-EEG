"""
analyse.py -- pure logic layer of the S0.0 ledger builder.

Responsibility
--------------
Turn measurements (scan.FileRecord) plus a declared ancestor map into
ledger rows.  Performs NO I/O: everything it needs is already measured.

The central idea
----------------
The ancestor map DECLARES a relationship ("this local file is byte-identical
to that repository file").  This module MEASURES the relationship from the
hashes and compares the two.  A disagreement is not silently resolved in
favour of either side: the row is marked verdict "unknown", which fails the
S0.0 exit test.  The ledger is only worth having if it can contradict the
document that produced it.

Vocabulary
----------
relationship
    origin              file predates S0; it is its own ancestor at pre-s0
    identical           byte-identical to a declared ancestor
    mechanical          output of a stated reproducible transform (clause 2
                        of the byte-identity rule)
    working_branch_edit hand edit made before S0 on the working branch.  NOT
                        a mechanical transform, and legitimate only because
                        S0.1 commits the file, making it an ancestor for
                        everything downstream
    new                 new infrastructure nothing pre-existing depends on

verdict
    keep      survives into the installed package towards_eeg/
    retain    stays in the repository but outside the installed package
    discard   removed from the working tree
    new       created by S0

Note that "retain" is not in the roadmap's vocabulary.  It is introduced
here because the roadmap simultaneously declares the Colab data-prep scripts
"not a target" (decision register) and instructs S0.2 to strip magics from
them.  Both are satisfied if they stay in the tree and out of the package.
The distinction is recorded, not decided; see the S0.0 report.
"""

from dataclasses import dataclass, asdict
from typing import Tuple

RELATIONSHIPS = ("origin", "identical", "mechanical", "working_branch_edit", "new")
VERDICTS = ("keep", "retain", "discard", "new")

# Sentinel written into hash-chain columns that a later sub-step will fill.
NOT_YET = "-"


@dataclass(frozen=True)
class LedgerRow:
    path: str
    scope: str
    stage: str
    verdict: str
    relationship: str
    transform_id: str
    ancestor_path: str
    ancestor_sha256: str
    sha256_pre_s0: str
    sha256_post_s02: str
    sha256_post_s03: str
    sha256_post_s04: str
    size_bytes: int
    is_python: bool
    parses: bool
    n_syntax_warnings: int
    n_crlf: int
    n_lf: int
    n_nonascii: int
    has_cookie: bool
    rationale: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ShadowedDef:
    """One module-level name defined more than once in one file (defect O7)."""

    path: str
    name: str
    kind: str
    lines: Tuple[int, ...]
    effective_line: int
    dead_lines: Tuple[int, ...]


# ---------------------------------------------------------------------------
# measured relationship
# ---------------------------------------------------------------------------

def measured_relationship(local, ancestor):
    """Compare two FileRecords using only their hashes.

    Returns (relationship, transform_id, note).  The lattice is documented in
    scan.py: each test is a strictly weaker equality than the one above it.
    """
    if local.sha256 == ancestor.sha256:
        return "identical", "T0", "bytes equal"
    if local.sha256_lf == ancestor.sha256_lf:
        return "mechanical", "T2", "line endings only (LF <-> CRLF)"
    if local.sha256_stripped == ancestor.sha256_stripped:
        return "mechanical", "T1", "trailing newline only (line endings may also differ)"
    return "working_branch_edit", "", "content differs beyond line endings"


# ---------------------------------------------------------------------------
# defect O7
# ---------------------------------------------------------------------------

def find_shadowed_defs(records):
    """Module-level names defined more than once.

    At module scope the LAST definition wins, unconditionally.  In a notebook
    the winner depends on cell execution order, which version control does not
    record -- which is why this must be written down before the files move.
    """
    out = []
    for r in records:
        if not r.is_python or not r.parses:
            continue
        by_name = {}
        for name, lineno, kind in r.toplevel_defs:
            by_name.setdefault(name, []).append((lineno, kind))
        for name in sorted(by_name):
            entries = sorted(by_name[name])
            if len(entries) < 2:
                continue
            lines = tuple(ln for ln, _ in entries)
            out.append(
                ShadowedDef(
                    path=r.path,
                    name=name,
                    kind=entries[-1][1],
                    lines=lines,
                    effective_line=lines[-1],
                    dead_lines=lines[:-1],
                )
            )
    out.sort(key=lambda s: (s.path, s.lines[0]))
    return out


def find_duplicate_classes(records):
    """Top-level class names defined in more than one file.

    Reported for awareness at S0.0; it is smoke-test assertion 6 that acts on
    it at S0.9.
    """
    by_name = {}
    for r in records:
        if not r.is_python or not r.parses:
            continue
        for name, lineno, kind in r.toplevel_defs:
            if kind == "class":
                by_name.setdefault(name, []).append((r.path, lineno))
    return {k: sorted(v) for k, v in sorted(by_name.items()) if len(v) > 1}


# ---------------------------------------------------------------------------
# scope and payload rules
# ---------------------------------------------------------------------------

def assign_scope(path, scopes, default="colab"):
    """First matching prefix wins; declaration order in the spec is therefore
    significant and is preserved by build_ledger.py."""
    for scope_name, prefixes in scopes:
        for prefix in prefixes:
            if path == prefix or path.startswith(prefix):
                return scope_name
    return default


def is_payload(path, rule):
    """True if the file is part of the committed binary payload (D-8, S0.7)."""
    for ext in rule.get("extensions", []):
        if path.lower().endswith(ext.lower()):
            return True
    for pat in rule.get("suffix_patterns", []):
        if pat in path:
            return True
    return path in set(rule.get("paths", []))


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

def build_rows(repo_records, local_records, spec, phases=None, as_of="post_s01"):
    """Assign a verdict to every file.

    repo_records  : FileRecords for the repository tree at pre-s0
    local_records : FileRecords for the four local working files
    spec          : the declared ancestor map (see tools/ancestors.json)
    phases        : optional hash-chain record (see tools/phase_hashes.json,
                    written by tools/stamp_phase.py).  Without it the chain
                    columns stay at the NOT_YET sentinel and sha256_pre_s0 is
                    whatever is on disk -- correct at S0.0 and WRONG from S0.2
                    onward, because a rescan would overwrite the pre-S0 state
                    with the post-transform hash.  Pass it after S0.2.

    Returns (rows, problems).  problems is a list of human-readable strings;
    a non-empty list means the ledger disagrees with the documents and S0.0
    must not be signed off until each entry is resolved.
    """
    problems = []
    by_path = {r.path: r for r in repo_records}
    chain = (phases or {}).get("phases", {})

    def chained(path, measured_sha):
        """(pre_s0, post_s02, post_s03, post_s04) for one path."""
        pre = chain.get("pre_s0", {}).get(path, measured_sha)
        return (pre,
                chain.get("post_s02", {}).get(path, NOT_YET),
                chain.get("post_s03", {}).get(path, NOT_YET),
                chain.get("post_s04", {}).get(path, NOT_YET))
    local_by_name = {r.path: r for r in local_records}

    scopes = [(k, tuple(v)) for k, v in spec["scopes"]]
    payload_rule = spec.get("binary_payload", {})
    discards = {d["path"]: d for d in spec.get("discards", [])}

    new_infra = tuple(spec.get("new_infrastructure", []))

    rows = []

    # -- the four local working files, entering the tree at S0.1 ------------
    declared_ancestors = set()
    for entry in spec["working_branch"]:
        name = entry["local"]
        anc_path = entry["ancestor"]
        declared_ancestors.add(anc_path)

        loc = local_by_name.get(name)
        anc = by_path.get(anc_path)
        if loc is None:
            problems.append("working file not measured: %s" % name)
            continue
        if anc is None:
            problems.append(
                "declared ancestor absent from snapshot: %s (for %s)" % (anc_path, name)
            )
            continue

        rel, tid, note = measured_relationship(loc, anc)
        expected = entry["expected_relationship"]
        if as_of == "post_s01" and "expected_relationship_post_s01" in entry:
            # S0.1 wrote the working-branch bytes to the ancestor path, so
            # from that commit onward the two are the same file. Doc 4 s3
            # calls this out as expected; declaring it keeps the exit test
            # meaningful instead of permanently red.
            expected = entry["expected_relationship_post_s01"]
        if rel != expected:
            problems.append(
                "MISMATCH %s: declared '%s', measured '%s' (%s)"
                % (name, expected, rel, note)
            )
            verdict = "unknown"
        else:
            verdict = "keep"

        extra = []
        if loc.n_crlf and not anc.n_crlf:
            extra.append("ancestor LF, local CRLF (%d)" % loc.n_crlf)
        elif anc.n_crlf and not loc.n_crlf:
            extra.append("ancestor CRLF (%d), local LF" % anc.n_crlf)

        rows.append(
            LedgerRow(
                path=name,
                scope="orchestrator",
                stage=entry.get("stage", "S0.1"),
                verdict=verdict,
                relationship=rel,
                transform_id=tid,
                ancestor_path=anc_path,
                ancestor_sha256=anc.sha256,
                sha256_pre_s0=chained(name, loc.sha256)[0],
                sha256_post_s02=chained(name, loc.sha256)[1],
                sha256_post_s03=chained(name, loc.sha256)[2],
                sha256_post_s04=chained(name, loc.sha256)[3],
                size_bytes=loc.size,
                is_python=loc.is_python,
                parses=loc.parses,
                n_syntax_warnings=loc.n_syntax_warnings,
                n_crlf=loc.n_crlf,
                n_lf=loc.n_lf,
                n_nonascii=loc.n_nonascii,
                has_cookie=loc.has_cookie,
                rationale="; ".join([entry["rationale"], note] + extra),
            )
        )

    # -- every file already in the repository -------------------------------
    for r in repo_records:
        scope = assign_scope(r.path, scopes)

        if r.path in discards:
            d = discards[r.path]
            verdict, stage, rationale = "discard", d.get("stage", "S0.2"), d["rationale"]
        elif is_payload(r.path, payload_rule):
            verdict, stage = "discard", "S0.7"
            rationale = payload_rule.get(
                "rationale", "committed binary payload; moves to a separate repository (D-8)"
            )
        elif r.path in declared_ancestors:
            verdict, stage = "keep", "S0.4"
            rationale = "ancestor of a working-branch file; superseded at S0.1, moved at S0.4"
        elif any(r.path == pre or r.path.startswith(pre) for pre in new_infra):
            verdict, stage = "new", "S0.0"
            rationale = (
                "new infrastructure (byte-identity rule clause 3): nothing "
                "pre-existing depends on it"
            )
            scope = "infrastructure"
        elif scope == "orchestrator":
            verdict, stage = "keep", "S0.4"
            rationale = "orchestrator source; enters the installed package"
        else:
            verdict, stage = "retain", "S0.2" if (r.is_python and not r.parses) else "-"
            rationale = (
                "%s scope; stays in the repository, outside the installed package" % scope
            )
            if r.is_python and not r.parses:
                rationale += "; magics stripped at S0.2 so ast.parse succeeds"

        rows.append(
            LedgerRow(
                path=r.path,
                scope=scope,
                stage=stage,
                verdict=verdict,
                relationship="origin",
                transform_id="",
                ancestor_path=r.path,
                ancestor_sha256=r.sha256,
                sha256_pre_s0=chained(r.path, r.sha256)[0],
                sha256_post_s02=chained(r.path, r.sha256)[1],
                sha256_post_s03=chained(r.path, r.sha256)[2],
                sha256_post_s04=chained(r.path, r.sha256)[3],
                size_bytes=r.size,
                is_python=r.is_python,
                parses=r.parses,
                n_syntax_warnings=r.n_syntax_warnings,
                n_crlf=r.n_crlf,
                n_lf=r.n_lf,
                n_nonascii=r.n_nonascii,
                has_cookie=r.has_cookie,
                rationale=rationale,
            )
        )

    rows.sort(key=lambda x: (x.scope, x.path))

    unknown = [x.path for x in rows if x.verdict not in VERDICTS]
    if unknown:
        problems.append("rows with a non-final verdict: %s" % ", ".join(unknown))

    return rows, problems
