"""
render.py -- pure output layer of the S0.0 ledger builder.

Responsibility
--------------
Turn analysed rows into two artefacts:

    LEDGER.md    the committed, human-readable ledger
    ledger.csv   the machine-readable form that S0.2, S0.3, S0.4 and the
                 S0.9 smoke test read

No scanning, no classification.  Swapping the report layout must not be able
to change a verdict, which is why this module receives finished rows and can
only format them.

Determinism: rows arrive pre-sorted from analyse.py and are not reordered
here, so two runs over the same tree produce byte-identical output and the
ledger can itself be diffed.
"""

import csv

CSV_COLUMNS = [
    "path",
    "scope",
    "stage",
    "verdict",
    "relationship",
    "transform_id",
    "ancestor_path",
    "ancestor_sha256",
    "sha256_pre_s0",
    "sha256_post_s02",
    "sha256_post_s03",
    "sha256_post_s04",
    "sha256_post_s05",
    "sha256_post_s06",
    "sha256_post_s07",
    "sha256_post_s08",
    "size_bytes",
    "is_python",
    "parses",
    "n_syntax_warnings",
    "n_crlf",
    "n_lf",
    "n_nonascii",
    "has_cookie",
    "rationale",
]


def write_csv(rows, out_path):
    """Emit ledger.csv.  newline='' per the csv module contract; LF line
    terminator so the file survives .gitattributes normalisation at S0.3."""
    with open(str(out_path), "w", newline="", encoding="ascii", errors="strict") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.as_dict()[k] for k in CSV_COLUMNS})


def write_csv_dicts(dict_rows, fieldnames, out_path):
    """Emit a ledger CSV from plain dicts, with the SAME conventions as
    write_csv above: LF terminator, ASCII, strict.

    It exists because any tool that rewrites ledger.csv in place -- the phase
    stamper, the move applier -- must not reinvent those conventions. A bare
    csv.DictWriter defaults to a CRLF terminator, which silently converted the
    whole file to CRLF once already; in a project whose next sub-step is line
    ending normalisation, that is not an acceptable failure mode.
    """
    with open(str(out_path), "w", newline="", encoding="ascii", errors="strict") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(dict_rows)
    with open(str(out_path), "rb") as fh:
        data = fh.read()
    if b"\r\n" in data:
        raise RuntimeError("write_csv_dicts emitted CRLF into %s" % out_path)


def read_csv(in_path):
    """Read ledger.csv back as a list of dicts.  Used by the smoke test to
    prove the round trip, and by later sub-steps to fill the hash chain."""
    with open(str(in_path), "r", newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


def _short(h, n=12):
    return h[:n] if h and h != "-" else h


def _counts(rows, key):
    out = {}
    for r in rows:
        out[getattr(r, key)] = out.get(getattr(r, key), 0) + 1
    return dict(sorted(out.items()))


def _table(header, widths, lines):
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |"
    body = [
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
        for row in lines
    ]
    return "\n".join([head, sep] + body)


def write_markdown(rows, o7, dup_classes, problems, meta, out_path):
    """Emit LEDGER.md."""
    L = []
    a = L.append

    a("# Towards-EEG -- S0.0 Reconciliation Ledger")
    a("")
    a("**Stage:** S0.0 (roadmap rev 3, section 3.2)  ")
    a("**Generated:** %s by `%s` v%s  " % (meta["generated"], meta["tool"], meta["version"]))
    a("**Snapshot:** `%s`, %s  " % (meta["snapshot"], meta["snapshot_date"]))
    a("**Tree root:** `%s`  " % meta["root"])
    a("**Rows:** %d (%d repository files + %d local working files)"
      % (len(rows), meta["n_repo"], meta["n_local"]))
    a("")
    a("This ledger is the evidence base for the byte-identity rule (handoff")
    a("brief section 3). Every file in the reorganised tree must be (1) byte-identical")
    a("to an identified ancestor, (2) the output of a stated reproducible mechanical")
    a("transform of one, or (3) new infrastructure nothing pre-existing depends on.")
    a("Each row below records which, with the hash that proves it.")
    a("")
    a("Regenerate with:")
    a("")
    a("```bash")
    a("python3 tools/build_ledger.py --root . --spec tools/ancestors.json \\")
    a("        --local-dir <dir containing the four working files> --out-dir .")
    a("```")
    a("")

    # -- 0. exit test ------------------------------------------------------
    a("## 0. Exit test")
    a("")
    if problems:
        a("**FAIL** -- %d unresolved item(s). S0.0 is not complete." % len(problems))
        a("")
        for p in problems:
            a("- %s" % p)
    else:
        a("**PASS** -- every row carries a final verdict; every declared")
        a("relationship agrees with the measured hashes.")
    a("")

    # -- 1. summary --------------------------------------------------------
    a("## 1. Summary")
    a("")
    a("### By verdict")
    a("")
    a(_table(["verdict", "files"], [12, 6],
             [[k, v] for k, v in _counts(rows, "verdict").items()]))
    a("")
    a("### By scope")
    a("")
    a(_table(["scope", "files"], [14, 6],
             [[k, v] for k, v in _counts(rows, "scope").items()]))
    a("")
    a("### By acting sub-step")
    a("")
    a(_table(["stage", "files"], [8, 6],
             [[k, v] for k, v in _counts(rows, "stage").items()]))
    a("")

    py = [r for r in rows if r.is_python]
    a("### Python surface at pre-s0")
    a("")
    a("- python files: **%d**" % len(py))
    a("- failing `ast.parse`: **%d**" % sum(1 for r in py if not r.parses))
    a("- containing bytes >= 0x80: **%d**" % sum(1 for r in py if r.n_nonascii))
    a("- non-ASCII and no PEP 263 cookie: **%d**"
      % sum(1 for r in py if r.n_nonascii and not r.has_cookie))
    a("- containing CRLF: **%d**" % sum(1 for r in py if r.n_crlf))
    a("- raising SyntaxWarning (latent, e.g. invalid escape sequences): **%d**"
      % sum(1 for r in py if r.n_syntax_warnings))
    a("")

    # -- 2. working branch -------------------------------------------------
    a("## 2. Working-branch reconciliation (S0.1)")
    a("")
    a("The four local files are committed BEFORE the `pre-s0` tag, so that")
    a("ancestry is verifiable by `git` rather than by assertion. Relationship")
    a("`working_branch_edit` marks a hand edit that is NOT a mechanical")
    a("transform; it is admissible only because the commit makes the file its")
    a("own ancestor for every later sub-step.")
    a("")
    wb = [r for r in rows if r.relationship != "origin"]
    a(_table(
        ["file", "relationship", "T", "ancestor", "sha256(pre-s0)"],
        [38, 20, 3, 52, 14],
        [[r.path, r.relationship, r.transform_id or "-", r.ancestor_path,
          _short(r.sha256_pre_s0)] for r in wb]))
    a("")
    for r in wb:
        a("- **%s** -- %s" % (r.path, r.rationale))
    a("")

    # -- 3. discards -------------------------------------------------------
    a("## 3. Discards")
    a("")
    disc = [r for r in rows if r.verdict == "discard"]
    a("%d files. Payload discards move to the separate repository per D-8;" % len(disc))
    a("no history is rewritten.")
    a("")
    a(_table(["path", "stage", "sha256(pre-s0)", "bytes"], [62, 6, 14, 10],
             [[r.path, r.stage, _short(r.sha256_pre_s0), r.size_bytes] for r in disc]))
    a("")

    # -- 4. O7 -------------------------------------------------------------
    a("## 4. Defect O7 register -- shadowed module-level definitions")
    a("")
    a("At module scope the LAST definition of a name wins, unconditionally. In")
    a("a notebook the winner depends on cell execution order, which version")
    a("control does not record. Once these files are renamed and moved at S0.4")
    a("this table is the only surviving record of which definition ran.")
    a("")
    a("**%d shadowed pairs across %d files.**"
      % (len(o7), len({s.path for s in o7})))
    a("")
    a(_table(["file", "name", "defined at", "effective", "dead"],
             [56, 34, 16, 10, 12],
             [[s.path, s.name, ", ".join("L%d" % x for x in s.lines),
               "L%d" % s.effective_line,
               ", ".join("L%d" % x for x in s.dead_lines)] for s in o7]))
    a("")

    # -- 5. duplicate class names -----------------------------------------
    a("## 5. Duplicate top-level class names")
    a("")
    a("Recorded for awareness. Smoke-test assertion 6 acts on this at S0.9 and")
    a("must be scoped, or it can never pass while the passive sub-project and")
    a("the Colab scripts remain in the tree.")
    a("")
    a("**%d names defined in more than one file.**" % len(dup_classes))
    a("")
    a(_table(["class", "copies", "locations"], [28, 6, 96],
             [[k, len(v), "; ".join("%s:%d" % (p, ln) for p, ln in v)]
              for k, v in dup_classes.items()]))
    a("")

    # -- 6. full ledger ----------------------------------------------------
    a("## 6. Full ledger")
    a("")
    a("Hash-chain columns: `pre` is measured now. `s02`, `s03`, `s04` are")
    a("filled by rerunning this tool after each of those sub-steps. A dash means")
    a("the sub-step has not run yet. S0.4 must leave `s04` equal to `s03`:")
    a("that equality IS the exit test for the move.")
    a("")
    a(_table(
        ["path", "scope", "stage", "verdict", "pre", "s02", "s03", "s04"],
        [76, 13, 6, 8, 13, 4, 4, 4],
        [[r.path, r.scope, r.stage, r.verdict, _short(r.sha256_pre_s0),
          _short(r.sha256_post_s02, 3), _short(r.sha256_post_s03, 3),
          _short(r.sha256_post_s04, 3)] for r in rows]))
    a("")
    a("Full detail, including rationale for every row, is in `ledger.csv`.")
    a("")

    with open(str(out_path), "w", encoding="ascii", errors="strict", newline="\n") as fh:
        fh.write("\n".join(L))
