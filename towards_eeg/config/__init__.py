"""
Resolved configuration and paths.json. Populated in S0.6.

WHAT THIS MODULE IS
-------------------
The single place a filesystem path literal is allowed to live inside the
installed package. S0.6 replaced five string literals in two orchestrator
modules with resolve() calls; the strings themselves now sit in paths.json
next to this file, byte-identical to what they replaced.

WHY IT READS NOTHING AT IMPORT
------------------------------
Decision N-20: resolution is LAZY. `import towards_eeg.config` performs no
filesystem access whatsoever, so the package remains importable when the
configuration is absent or unreadable, and the S0.9 import-surface assertions
stay reachable. TEEG_10 section 4.3 justifies this by the call sites, which is
not the real reason: the five call sites in scope are all at MODULE scope in
files that already cannot be imported (they die on `from google.colab import
drive` and on an undefined name), so at those five sites "use time" and
"import time" are the same instant. The property that actually matters is a
property of THIS module's API, not of its callers, and it is asserted by
check_04 of tools/test_s0_6_exit.py.

WHY JSON AND NOT YAML
---------------------
See paths.json's format_rationale. Decision N-19.

DELIBERATE NON-FEATURES
-----------------------
No environment-variable lookup, no search path, no user-config merging, no
writing. Each would be a behaviour change relative to a hard-coded literal and
S0 fixes no defects (TEEG_00 section 3). `overrides` exists so that a later
stage can inject a mapping explicitly, at its own call site, without this
module acquiring a policy about where configuration comes from.

Standard library only, ASCII source, Python 3.8+.
"""

import json
import os

__all__ = ["resolve", "path_keys", "load_paths", "config_path", "PathKeyError"]

_PATHS_FILENAME = "paths.json"

# Populated on first successful load_paths(); never at import.
_CACHE = None


class PathKeyError(KeyError):
    """Raised for a key that is not declared in paths.json.

    A subclass of KeyError so that existing except-clauses behave, and a named
    type so that a caller can distinguish "this key is not configured" from
    "some dict lookup failed".
    """


def config_path():
    """Absolute path of the shipped paths.json.

    Package-relative, like towards_eeg/io/schema.py's resolution of schemas/,
    and declared in s05_exit_scope.json under package_relative_resource_access
    for the same reason: a package locates its own data inside itself. This
    function touches the filesystem only to build a string; it does not read.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        _PATHS_FILENAME)


def load_paths(refresh=False):
    """Return the parsed paths.json document, caching after the first read.

    For each fixed call, the returned object is the cached document; callers
    must not mutate it. Pass refresh=True to force a re-read, which exists for
    the mutation harness rather than for production use.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    with open(config_path(), "r", encoding="ascii") as fh:
        doc = json.load(fh)
    if "paths" not in doc or not isinstance(doc["paths"], dict):
        raise ValueError("%s has no 'paths' object" % config_path())
    _CACHE = doc
    return _CACHE


def path_keys():
    """Sorted tuple of every declared key, for each state of paths.json."""
    return tuple(sorted(load_paths()["paths"].keys()))


def resolve(key, overrides=None):
    """Return the configured path string for `key`.

    Parameters
    ----------
    key : str
        A key declared in paths.json.
    overrides : mapping or None
        If given and it contains `key`, its value is returned instead of the
        shipped default, and paths.json is not read at all. Provided so that a
        later stage can redirect a path at its own call site without this
        module acquiring a search policy. S0 ships no caller that passes it.

    Returns
    -------
    str
        For the shipped default configuration and for each of the five
        occurrences externalised at S0.6, this is byte-identical to the string
        literal that stood at that call site before S0.6. That equality is the
        behaviour-preservation proof and is asserted by check_05 of
        tools/test_s0_6_exit.py.

    Raises
    ------
    PathKeyError
        If `key` is not declared. The message lists the declared keys, because
        the failure mode this replaces -- a mistyped literal -- used to be a
        FileNotFoundError far from its cause.
    """
    if overrides is not None and key in overrides:
        return overrides[key]
    entries = load_paths()["paths"]
    if key not in entries:
        raise PathKeyError(
            "%r is not a declared path key; declared keys are %s (see %s)"
            % (key, ", ".join(sorted(entries)), config_path()))
    entry = entries[key]
    if not isinstance(entry, dict) or "value" not in entry:
        raise ValueError("path entry %r has no 'value'" % key)
    return entry["value"]
