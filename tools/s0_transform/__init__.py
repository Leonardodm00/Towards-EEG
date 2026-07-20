"""s0_transform -- scripted, reproducible byte-level transforms for stage S0.

New infrastructure under clause (3) of the byte-identity rule: nothing
pre-existing depends on it.  Each transform in this package is required to be

  * declarative   -- the target set is data, never inferred at run time;
  * idempotent    -- applying twice equals applying once;
  * invertible    -- --restore reproduces the input bytes exactly;
  * logged        -- a machine-readable JSON log is the ledger entry that
                     clause (2) of the byte-identity rule demands.

Pure standard library, ASCII source, Python 3.8+.  It must run in a bare HPC
login shell and in a Colab runtime without any environment work.
"""

TOOL_NAME = "s0_transform"
TOOL_VERSION = "1.0.0"

# Transform identifiers used across S0.  T0-T6 are defined by S0.0.
T7 = "T7"   # notebook-magic neutralisation (comment form)
T7P = "T7p"  # notebook-magic neutralisation (pass-comment form, block-preserving)
T8 = "T8"   # declared single-line leading-whitespace correction
