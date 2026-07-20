"""
scan.py -- pure I/O layer of the S0.0 ledger builder.

Responsibility
--------------
Walk a directory tree, read each file's bytes, and record measurements.

This module deliberately makes NO judgement.  It does not know what an
ancestor is, what a verdict is, or which sub-step of S0 touches a file.
Everything it emits is a measurement that can be reproduced by rerunning
it on the same bytes.  All classification lives in analyse.py.

Measurements taken per file
---------------------------
  size            bytes on disk
  sha256          hex digest of the exact bytes
  sha256_lf       hex digest after CRLF -> LF normalisation
  sha256_stripped hex digest after CRLF -> LF then stripping trailing "\n"
  is_python       filename suffix is .py
  n_crlf, n_lf    line-ending census
  n_nonascii      count of bytes >= 0x80
  has_cookie      PEP 263 encoding declaration on line 1 or 2
  parses          ast.parse succeeded (python files only)
  n_syntax_warnings  SyntaxWarnings raised while parsing (latent defects)
  parse_error     first SyntaxError message, empty if none
  toplevel_defs   ((name, lineno, kind), ...) for module-scope def/class

toplevel_defs is a raw fact, not an analysis.  Deciding which of two
definitions of one name is effective at module scope is analyse.py's job.

The two derived hashes exist so that analyse.py can decide how two files
differ WITHOUT reading bytes itself.  They form a lattice of strictly
weakening equalities:

    sha256          equal  =>  byte-identical
    sha256_lf       equal  =>  differs in line endings only
    sha256_stripped equal  =>  differs in line endings and/or trailing
                               newlines only
    none equal             =>  content differs; not a mechanical transform
"""

import ast
import hashlib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# PEP 263: the declaration must match this on line 1 or line 2.
_COOKIE_RE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")

_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class FileRecord:
    """One measured file.  Immutable: a record is evidence, not a workspace."""

    path: str
    size: int
    sha256: str
    sha256_lf: str
    sha256_stripped: str
    is_python: bool
    n_crlf: int
    n_lf: int
    n_nonascii: int
    has_cookie: bool
    parses: bool
    n_syntax_warnings: int = 0
    parse_error: str = ""
    toplevel_defs: Tuple[Tuple[str, int, str], ...] = field(default_factory=tuple)


def sha256_bytes(data):
    """Hex SHA-256 of an in-memory bytes object."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    """Hex SHA-256 of a file, read in chunks so large payloads do not
    have to be held in memory."""
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _has_pep263_cookie(data):
    """True if a PEP 263 encoding declaration sits on line 1 or line 2."""
    for line in data.split(b"\n", 2)[:2]:
        if _COOKIE_RE.match(line):
            return True
    return False


def _toplevel_defs(tree):
    """Module-scope def / async def / class definitions, in source order."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append((node.name, node.lineno, "class"))
        elif isinstance(node, ast.AsyncFunctionDef):
            out.append((node.name, node.lineno, "async def"))
        elif isinstance(node, ast.FunctionDef):
            out.append((node.name, node.lineno, "def"))
    return tuple(out)


def measure_file(abs_path, rel_path):
    """Measure one file and return an immutable FileRecord.

    abs_path : path used to read the bytes
    rel_path : POSIX-style path recorded in the ledger (tree-relative)
    """
    abs_path = Path(abs_path)
    data = abs_path.read_bytes()

    is_python = abs_path.suffix == ".py"
    parses = False
    parse_error = ""
    defs = ()

    n_warnings = 0
    if is_python:
        # ast.parse emits SyntaxWarning for things that are legal today but
        # scheduled to become errors (chiefly invalid escape sequences such as
        # "\m" in an unraw docstring).  Capturing rather than printing keeps a
        # CI log readable AND keeps the information: the count is a latent
        # defect measure, recorded per file.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                # errors="replace" so that an undecodable byte becomes a parse
                # question rather than an I/O crash.  A file that only fails to
                # decode is still reported, with its parse error recorded.
                tree = ast.parse(data.decode("utf-8", errors="replace"))
                defs = _toplevel_defs(tree)
                parses = True
            except SyntaxError as exc:
                parse_error = "line %s: %s" % (exc.lineno, exc.msg)
            except ValueError as exc:
                # e.g. source containing NUL bytes
                parse_error = "ValueError: %s" % (exc,)
            n_warnings = len(caught)

    data_lf = data.replace(b"\r\n", b"\n")

    return FileRecord(
        path=rel_path,
        size=len(data),
        sha256=sha256_bytes(data),
        sha256_lf=sha256_bytes(data_lf),
        sha256_stripped=sha256_bytes(data_lf.rstrip(b"\n")),
        is_python=is_python,
        n_crlf=data.count(b"\r\n"),
        n_lf=data.count(b"\n"),
        n_nonascii=sum(1 for b in data if b >= 0x80),
        has_cookie=_has_pep263_cookie(data),
        parses=parses,
        n_syntax_warnings=n_warnings,
        parse_error=parse_error,
        toplevel_defs=defs,
    )


def scan_tree(root, skip_dirs=(".git", "__pycache__", ".ipynb_checkpoints")):
    """Measure every regular file under root.

    Returns a list of FileRecord sorted by path, so that two runs over the
    same bytes produce byte-identical output (a ledger that reorders itself
    between runs cannot be diffed).
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError("scan root is not a directory: %s" % (root,))

    skip = set(skip_dirs)
    records = []
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if skip.intersection(rel.parts):
            continue
        records.append(measure_file(p, rel.as_posix()))
    records.sort(key=lambda r: r.path)
    return records


def scan_files(pairs):
    """Measure an explicit list of (abs_path, rel_path) pairs.

    Used for the four local working files, which live outside the repository
    tree but must appear in the ledger because S0.1 commits them.
    """
    records = [measure_file(a, r) for a, r in pairs]
    records.sort(key=lambda r: r.path)
    return records
