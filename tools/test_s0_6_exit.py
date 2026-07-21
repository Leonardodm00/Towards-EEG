#!/usr/bin/env python3
"""
test_s0_6_exit.py -- exit test for stage S0.6 (path literals externalised).

    python3 tools/test_s0_6_exit.py --root .

WHAT S0.6 CLAIMS
----------------
The configuration path literals inside the installed package are externalised
to towards_eeg/config/paths.json and reached through towards_eeg.config.
resolve(). The transform is declared (T11), logged per occurrence, invertible
from its own log, and behaviour-preserving under the shipped default
configuration. Nothing else changed.

WHAT THIS FILE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
That every declaration in s06_exit_scope.json still applies; that the package
sweep is clean under the declared exemptions; that the resolver reads nothing
at import; that the shipped default resolves byte-identically to every literal
it replaced; and that the transform inverts.

It does NOT assert that the edited modules import or run. They do not, and
they did not before S0.6 either -- see the unimportable_by_construction
declaration, which this file asserts is still TRUE. A file that started
importing would mean the declaration is stale, which is an error and not good
news.

It also does not re-assert chain continuity. That is tools/test_s0_chain.py,
checks 11 to 13, because a chain assertion that lived in one sub-step's exit
test would have to be rewritten at the next.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import ast
import base64
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MARKER = "#S0.6:T11"


def package_py_files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "towards_eeg")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if f.endswith(".py"):
                full = os.path.join(dirpath, f)
                out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


def string_constants(src):
    """Every str constant in a parsed module, with its line number."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((getattr(node, "lineno", -1), node.value))
    return out


def module_scope_body(tree):
    """Module-scope statements that are neither imports nor definitions."""
    return [n for n in tree.body
            if not isinstance(n, (ast.Import, ast.ImportFrom,
                                  ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Expr))]


def imports_module_at_scope(tree, module):
    return any(isinstance(n, ast.ImportFrom) and n.module == module
               for n in tree.body)


def name_is_used_unbound_at_scope(tree, name):
    """True if `name` is READ at module scope and never BOUND at module scope.

    Deliberately conservative: only module-scope binding counts, because a
    binding inside a function body does not exist when the module body runs.
    """
    bound = set()
    for n in tree.body:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                bound.add(sub.id)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for a in sub.names:
                    bound.add((a.asname or a.name).split(".")[0])
    used = False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) \
                    and sub.id == name:
                used = True
    return used and name not in bound


def run(root):
    tdir = os.path.join(root, "tools", "s0_transform")
    with open(os.path.join(tdir, "s06_exit_scope.json"), "r", encoding="utf-8") as fh:
        scope = json.load(fh)
    with open(os.path.join(tdir, "s06_transform_log.json"), "r", encoding="utf-8") as fh:
        log = json.load(fh)
    with open(os.path.join(tdir, "s06_targets.json"), "r", encoding="utf-8") as fh:
        targets = json.load(fh)

    results = []

    def add(name, ok, msg):
        results.append((name, ok, msg))

    # -- 1: the shipped configuration is present and clean -------------------
    problems = []
    cfg_rel = scope["config_data_file"]
    cfg = os.path.join(root, cfg_rel)
    if not os.path.isfile(cfg):
        problems.append("missing: %s" % cfg_rel)
    else:
        data = open(cfg, "rb").read()
        if any(b > 127 for b in data):
            problems.append("non-ASCII bytes in %s" % cfg_rel)
        if b"\r\n" in data:
            problems.append("CRLF in %s" % cfg_rel)
        try:
            doc = json.loads(data.decode("ascii"))
            if "paths" not in doc:
                problems.append("%s has no 'paths' object" % cfg_rel)
        except Exception as exc:                            # noqa: BLE001
            problems.append("%s does not parse: %s" % (cfg_rel, exc))
    add("check_01_shipped_configuration_present_and_clean", not problems,
        "%s is present, pure ASCII, LF-only, valid JSON" % cfg_rel
        if not problems else "; ".join(problems[:4]))

    # -- 2: the declared keys are exactly the keys that exist ----------------
    # Every check from here on that touches the resolver must survive a tree
    # whose configuration is missing or unreadable. The mutation harness found
    # this the hard way: an earlier version let load_paths() raise out of
    # run(), so removing paths.json produced a TRACEBACK rather than a FAIL,
    # and an exit test that crashes returns no verdict on any of its other
    # thirteen checks. Report red; do not abort.
    from towards_eeg import config as CFG                    # noqa: E402
    declared = tuple(scope["path_keys"])
    try:
        got = CFG.path_keys()
        ok = got == declared
        msg = ("paths.json declares exactly %r" % (declared,) if ok
               else "paths.json holds %r, scope declares %r" % (got, declared))
    except Exception as exc:                                # noqa: BLE001
        ok, msg = False, "the configuration could not be read: %s" % exc
    add("check_02_declared_keys_are_the_keys_that_exist", ok, msg)

    # -- 3: every declared occurrence still applies --------------------------
    # A declaration that only skips is a declaration that hides (Doc 7 s3).
    # Each declared occurrence must still be found, at its declared line, in
    # its transformed form -- not merely "not contradicted".
    problems, n_occ = [], 0
    for entry in log["files"]:
        rel = entry["path"]
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            problems.append("%s: absent" % rel)
            continue
        lines = open(full, "r", encoding="ascii").read().split("\n")
        ins_idx = entry["insert_import"]["after_line"]
        expect_imp = base64.b64decode(
            entry["insert_import"]["text_b64"].encode("ascii")).decode("utf-8")
        if ins_idx >= len(lines) or lines[ins_idx] != expect_imp:
            problems.append("%s: the declared import insertion is not at line %d"
                            % (rel, ins_idx + 1))
        # Edited-line numbers in the log are PRE-edit, so shift by the insert.
        for ed in entry["edits"]:
            n_occ += 1
            idx = ed["line"] - 1 + (1 if ed["line"] - 1 >= ins_idx else 0)
            want = base64.b64decode(
                ed["replacement_b64"].encode("ascii")).decode("utf-8")
            if idx >= len(lines) or lines[idx] != want:
                problems.append("%s:%d does not carry its declared replacement"
                                % (rel, idx + 1))
            if "resolve('%s')" % ed["key"] not in want:
                problems.append("%s:%d does not call resolve(%r)"
                                % (rel, idx + 1, ed["key"]))
    if n_occ != scope["externalised_occurrences"]:
        problems.append("log holds %d occurrence(s), scope declares %d"
                        % (n_occ, scope["externalised_occurrences"]))
    add("check_03_every_declared_occurrence_still_applies", not problems,
        "%d occurrence(s) across %d file(s) carry their declared replacement "
        "and the declared import is in place"
        % (n_occ, len(log["files"])) if not problems else "; ".join(problems[:4]))

    # -- 4: the resolver reads nothing at import (decision N-20) -------------
    # Asserted in a subprocess so it is a statement about a COLD import, not
    # about whatever this process has already cached.
    code = ("import builtins, sys\n"
            "sys.path.insert(0, %r)\n"
            "seen = []\n"
            "_real = builtins.open\n"
            "builtins.open = lambda f, *a, **k: (seen.append(str(f)), _real(f, *a, **k))[1]\n"
            "import towards_eeg.config\n"
            "builtins.open = _real\n"
            "print(len(seen))\n" % root)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=root)
    n_open = (r.stdout or "").strip().splitlines()[-1] if r.stdout else "?"
    ok = n_open == "0"
    add("check_04_resolver_reads_nothing_at_import", ok,
        "importing towards_eeg.config opens no file; the package stays "
        "importable with no configuration present (N-20)" if ok
        else "cold import opened %s file(s); stderr=%s" % (n_open, (r.stderr or "")[-160:]))

    # -- 5: behaviour preservation, by value ---------------------------------
    # The shipped default must resolve to exactly the bytes that stood at the
    # call site. TEEG_10 s3.3: anything else is S0.6 silently FIXING the
    # Colab-dependence defect rather than externalising it.
    problems = []
    for entry in log["files"]:
        for ed in entry["edits"]:
            try:
                got = CFG.resolve(ed["key"])
            except Exception as exc:                        # noqa: BLE001
                problems.append("%s:%d resolve(%r) raised %s"
                                % (entry["path"], ed["line"], ed["key"], exc))
                continue
            if got != ed["literal"]:
                problems.append("%s:%d resolve(%r) -> %r, literal was %r"
                                % (entry["path"], ed["line"], ed["key"],
                                   got, ed["literal"]))
    add("check_05_shipped_default_resolves_to_the_prior_literal", not problems,
        "every key resolves byte-identically to the literal it replaced; the "
        "externalisation is behaviour-preserving under the shipped default"
        if not problems else "; ".join(problems[:3]))

    # -- 6: behaviour preservation, by inversion -----------------------------
    # Stronger than check_05: value equality proves the five lines are right,
    # inversion proves nothing ELSE in the file moved.
    problems = []
    for entry in log["files"]:
        full = os.path.join(root, entry["path"])
        text = open(full, "r", encoding="ascii").read()
        lines = text.split("\n")
        if text.endswith("\n"):
            lines = lines[:-1]
        try:
            del lines[entry["insert_import"]["after_line"]]
            for ed in entry["edits"]:
                lines[ed["line"] - 1] = base64.b64decode(
                    ed["original_b64"].encode("ascii")).decode("utf-8")
            back = "\n".join(lines) + ("\n" if entry["ends_with_newline"] else "")
            h = hashlib.sha256(back.encode("ascii")).hexdigest()
            if h != entry["sha256_before"]:
                problems.append("%s: inverse hashes %s, sha256_before is %s"
                                % (entry["path"], h[:12], entry["sha256_before"][:12]))
        except Exception as exc:                            # noqa: BLE001
            problems.append("%s: inversion failed: %s" % (entry["path"], exc))
    add("check_06_t11_inverts_to_its_prior_bytes", not problems,
        "the recorded inverse reproduces the pre-S0.6 bytes of all %d edited "
        "file(s) exactly" % len(log["files"])
        if not problems else "; ".join(problems[:3]))

    # -- 7: the package sweep is clean under the declared exemptions ---------
    sweep = scope["package_sweep"]
    forbidden = sweep["forbidden_substrings"]
    exempt = set(sweep["exempt_files"])
    problems = []
    for rel in package_py_files(root):
        if rel in exempt:
            continue
        src = open(os.path.join(root, rel), "r", encoding="ascii").read()
        for lineno, val in string_constants(src):
            for bad in forbidden:
                if bad in val:
                    problems.append("%s:%d contains %r" % (rel, lineno, bad))
    add("check_07_no_undeclared_path_literal_in_the_package", not problems,
        "no module under towards_eeg/ carries an absolute or drive-rooted path "
        "literal outside the %d declared exemption(s)" % len(exempt)
        if not problems else "; ".join(problems[:5]))

    # -- 8: the exemption did not widen --------------------------------------
    # An exemption for a whole file is only safe if what it exempts is pinned.
    # connectivity_buildup.py is exempt because of ONE declared drive.mount
    # call; if a second literal appeared there, the exemption would hide it.
    excl = scope["excluded_by_declaration"][0]
    rel = "towards_eeg/connectome/connectivity_buildup.py"
    src = open(os.path.join(root, rel), "r", encoding="ascii").read()
    hits = [(n, v) for n, v in string_constants(src) if "/content/" in v]
    ok = len(hits) == 1 and hits[0][1] == "/content/drive"
    add("check_08_file_exemption_still_covers_exactly_one_literal", ok,
        "%s retains exactly one '/content/' literal and it is the declared "
        "drive.mount mount point" % rel.split("/")[-1] if ok
        else "expected exactly one '/content/drive' literal, found %r" % (hits[:4],))

    # -- 9: the declared exclusion has not gone stale ------------------------
    lines = src.split("\n")
    want = targets["excluded_in_package"][0]["text"]
    ok = any(ln.strip() == want for ln in lines)
    add("check_09_declared_exclusion_still_present", ok,
        "the excluded drive.mount line is still present verbatim; the "
        "exclusion is live, not a leftover" if ok
        else "declared exclusion %r no longer appears" % want)

    # -- 10: the S0.5 declaration is still honoured --------------------------
    # s05_exit_scope.json declares that io/schema.py resolves schemas/ from
    # __file__ and is NOT configuration. A sweep that regressed into flagging
    # it would fail here rather than be argued about.
    with open(os.path.join(tdir, "s05_exit_scope.json"), "r", encoding="utf-8") as fh:
        s05 = json.load(fh)
    problems = []
    if "package_relative_resource_access" not in s05:
        problems.append("s05_exit_scope.json no longer carries the declaration")
    else:
        # Assert this STRUCTURALLY, not by string match. The first version of
        # this check looked for the literal "os.path.dirname(__file__)" while
        # the source reads "os.path.dirname(os.path.abspath(__file__))", so it
        # failed on a file that was entirely correct. A check that names an
        # exact spelling rots the moment the spelling changes -- trap T-15's
        # family, arriving here as a false FAIL rather than a false PASS.
        sch = os.path.join(root, "towards_eeg", "io", "schema.py")
        tree = ast.parse(open(sch, "r", encoding="ascii").read())
        names = set()
        consts = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                consts.add(node.value)
        if "__file__" not in names:
            problems.append("io/schema.py no longer references __file__")
        if "schemas" not in consts:
            problems.append("io/schema.py no longer names the schemas directory")
        if "towards_eeg/io/schema.py" not in json.dumps(scope["excluded_by_declaration"]):
            problems.append("S0.6 does not carry forward the S0.5 carve-out")
    add("check_10_s05_package_relative_declaration_honoured", not problems,
        "io/schema.py's package-relative resolution is declared in S0.5, "
        "carried forward in S0.6, and not swept"
        if not problems else "; ".join(problems[:3]))

    # -- 11: config/ imports nothing scientific ------------------------------
    # Same reasoning as check_07 of the S0.5 exit test: importing one of these
    # is the cheapest way for configuration code to acquire the ability to read
    # data without anyone noticing.
    forbidden_mods = set(s05["s0_populates_nothing"]["forbidden_tokens"])
    problems = []
    cdir = os.path.join(root, "towards_eeg", "config")
    for f in sorted(os.listdir(cdir)):
        if not f.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(cdir, f), "r", encoding="ascii").read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in forbidden_mods:
                    problems.append("config/%s imports %r" % (f, m))
    add("check_11_config_cannot_read_data", not problems,
        "no module under towards_eeg/config/ imports a scientific library; "
        "S0 populates nothing" if not problems else "; ".join(problems[:4]))

    # -- 12: paths.json is declared package data -----------------------------
    body = open(os.path.join(root, "pyproject.toml"), "r", encoding="utf-8").read()
    ok = '"towards_eeg.config" = ["paths.json"]' in body
    add("check_12_configuration_is_declared_package_data", ok,
        "paths.json is declared package data, so a non-editable install "
        "carries it" if ok
        else "pyproject.toml does not declare towards_eeg.config package-data; "
             "a wheel install would ship a resolve() that raises")

    # -- 13: the unimportability declaration is still TRUE -------------------
    # H-3. Both edited files were unimportable before S0.6 and still are; S0.6
    # neither caused that nor fixed it, because fixing it is a content
    # operation. Asserted STATICALLY and never by importing -- see
    # asserted_statically_not_by_importing in the declaration. In Colab, where
    # TEEG_08 says these harnesses run, importing connectivity_buildup would
    # block on a Drive authorisation prompt and then execute the notebook.
    decl = scope["unimportable_by_construction"]
    problems = []
    blockers = decl.get("blockers", {})
    if not blockers:
        problems.append("no blocker is declared; this check has no subject")
    for rel, b in sorted(blockers.items()):
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            problems.append("%s: absent" % rel)
            continue
        tree = ast.parse(open(full, "r", encoding="ascii").read())
        if not module_scope_body(tree):
            problems.append("%s no longer has an executable module-scope body; "
                            "the declaration is stale" % rel)
        if b["kind"] == "module_scope_import":
            if not imports_module_at_scope(tree, b["module"]):
                problems.append("%s no longer imports %s at module scope; it may "
                                "now import, and the O10 assignment needs "
                                "revisiting" % (rel, b["module"]))
        elif b["kind"] == "module_scope_unbound_name":
            if not name_is_used_unbound_at_scope(tree, b["name"]):
                problems.append("%s no longer uses %r unbound at module scope; "
                                "the declaration is stale"
                                % (rel, b["name"]))
        else:
            problems.append("%s: unknown blocker kind %r" % (rel, b["kind"]))
    if "O10" not in decl.get("proposed_defect_class", ""):
        problems.append("the declaration no longer proposes a defect class")
    add("check_13_unimportability_declaration_still_true", not problems,
        "both edited modules still carry a module-scope notebook body and each "
        "declared blocker is still present; S0.6 neither caused this nor fixed "
        "it (H-3, proposed O10)"
        if not problems else "; ".join(problems[:3]))

    # -- 14: every transform id in any log is registered ---------------------
    # H-5. Before S0.6 the register held T0-T6 while T7-T10 were in use. A
    # declaration a tool cannot read is not a declaration.
    with open(os.path.join(root, "tools", "ancestors.json"), "r", encoding="utf-8") as fh:
        registered = set(json.load(fh)["transforms"].keys())
    used = set()
    for name in sorted(os.listdir(tdir)):
        if not name.endswith("_transform_log.json"):
            continue
        with open(os.path.join(tdir, name), "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if "transform_id" in doc:
            used.add(doc["transform_id"])
        for entry in doc.get("files", []):
            for ed in entry.get("edits", []):
                if ed.get("transform"):
                    used.add(ed["transform"])
                for t in ed.get("transforms", []) or []:
                    used.add(t)
    missing = sorted(used - registered)
    add("check_14_every_used_transform_id_is_registered", not missing,
        "all %d transform id(s) used in the logs are registered in "
        "ancestors.json" % len(used)
        if not missing else "unregistered: %r" % missing)

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="S0.6 exit test (path externalisation).")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    results = run(args.root)
    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    print("\nS0.6 EXIT TEST: %s" % ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
