"""pytest configuration for the tester's spec-derived tests (tests/spec/).

Puts the Ih Fit folder (the noise fit's root, two levels up) and this folder
on sys.path so the tests import the code under test exactly as the smoke
suites do (flat modules, no package). Run from the Ih Fit folder:

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/spec -q

Tests marked `slow` run reduced fits (minutes); deselect with -m "not slow".
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: reduced-size fits that take minutes")
