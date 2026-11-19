"""Block 6 (run_noise_fit.py) against SPEC 2.6 (g) and the Block 6 entry.

Oracles: the two tables' columns exactly as 2.6 (g) lists them, in order; tau columns in ms
(= 1e3 x the law's seconds); a JSON whose Block 9 keys differ from the constants stops the
merge (exit 1, as built); NaN / None written as empty fields.
"""
import json
import subprocess
import sys
from pathlib import Path

import run_noise_fit as rnf

CODE_DIR = Path(__file__).resolve().parents[2]
CELL_COLS = ["specimen_id", "status", "reason", "noise_sigma_f_mV", "noise_sigma_m_mV", "noise_tau_m_ms",
             "noise_sigma_s_mV", "noise_tau_s_ms", "noise_J", "noise_dof", "noise_p", "noise_kappa",
             "noise_n_solves", "noise_flags", "noise_t_ss_ms", "noise_K", "noise_K_LS",
             "noise_cell_tau2_log_sigma_m_ss", "noise_cell_mu_log_sigma_m_ss", "noise_cell_tau2_log_sigma_m_ls",
             "noise_cell_mu_log_sigma_m_ls", "noise_cell_tau2_log_sigma_s_ss", "noise_cell_mu_log_sigma_s_ss",
             "noise_cell_tau2_log_sigma_s_ls", "noise_cell_mu_log_sigma_s_ls", "noise_code_commit"]
SWEEP_COLS = ["specimen_id", "noise_sweep_protocol", "noise_sweep_number", "noise_tilt_mV_per_s",
              "noise_sweep_sigma_f_mV", "noise_sweep_sigma_m_raw_mV", "noise_sweep_sigma_m_mV",
              "noise_sweep_v_log_sigma_m", "noise_sweep_w_m", "noise_sweep_sigma_s_raw_mV",
              "noise_sweep_sigma_s_mV", "noise_sweep_v_log_sigma_s", "noise_sweep_w", "noise_sweep_flags"]


def test_table_columns():
    assert list(rnf.CELL_COLUMNS) == CELL_COLS
    assert list(rnf.SWEEP_COLUMNS) == SWEEP_COLS
    assert "noise_rho_f" not in rnf.CELL_COLUMNS and "noise_sweep_rho_f" not in rnf.SWEEP_COLUMNS


def test_cell_row_units():
    rec = {"code_commit": "x", "windows": {"t_ss_ms": 80.0, "K": 63, "K_LS": 5},
           "cell_fit": {"law": {"sigma_f_mV": 0.03, "sigma_m_mV": 0.02, "tau_m_s": 3.5e-4, "sigma_s_mV": 0.05,
                                "tau_s_s": 0.04}, "J": 1.0, "dof": 16, "p_value": 0.5, "kappa": 10.0,
                        "n_solves": 3, "flags": ["a", "b"]}}
    row = rnf.cell_row(1, "ok", "", rec)
    assert row["noise_tau_m_ms"] == 1e3 * 3.5e-4 and row["noise_tau_s_ms"] == 1e3 * 0.04
    assert row["noise_sigma_m_mV"] == 0.02 and row["noise_flags"] == "a;b"


def test_merge_stops_on_two_component_tables(tmp_path):
    cells = tmp_path / "cells"
    cells.mkdir()
    old_cc = {"specimen_id": 5, "noise_cell_tau2_log_sigma_s_ss": 0.0, "noise_cell_mu_log_sigma_s_ss": -3.0,
              "noise_cell_tau2_log_sigma_s_ls": 0.0, "noise_cell_mu_log_sigma_s_ls": -3.0}
    rec = {"specimen_id": 5, "status": "ok", "reason": "", "tables": {"cell_columns": old_cc, "sweep_rows": []},
           "cell_fit": {"law": {"sigma_f_mV": 0.03, "rho_f": 0.2, "sigma_s_mV": 0.05, "tau_s_s": 0.04}}}
    (cells / "specimen_5.json").write_text(json.dumps(rec))
    p = subprocess.run([sys.executable, "run_noise_fit.py", "merge", "--out", str(tmp_path), "--group", "G"],
                       cwd=str(CODE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    assert p.returncode == 1, (p.returncode, p.stdout, p.stderr)
