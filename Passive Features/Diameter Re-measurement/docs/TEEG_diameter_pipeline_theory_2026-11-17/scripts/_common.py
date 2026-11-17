"""Shared setup of the figure and check scripts of TEEG_diameter_pipeline_theory_2026-11-17.md.

The scripts run inside the repository (they import the package allen_diameter of this workstream and the
diagnostic checks/own_shadow_clearance_check.py). Parameters of every figure: the CONFIG block of its script.
No randomness anywhere (no seed needed). Pure ASCII.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent.parent                      # "Passive Features/Diameter Re-measurement"
FIG = HERE.parent / "figures"
sys.path.insert(0, str(WS / "src"))


def load_check():
    """The module checks/own_shadow_clearance_check.py (the split of line_fast's slab loop, in memory)."""
    spec = importlib.util.spec_from_file_location("own_check", str(WS / "checks" / "own_shadow_clearance_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def line_geometry(default_config, ks):
    import numpy as np
    p = default_config().acquisition.res0_um
    return (np.array([437 * p, 211 * p]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.arange(-26, 27) * 0.1144,
            np.asarray(ks))
