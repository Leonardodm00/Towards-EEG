"""Figure 2 (plotted): the defocus kernel's SD sigma_r(delta) as the pilot runs use it. python3 fig02_kernel.py"""
import dataclasses
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import FIG
from allen_diameter.config import default_config
from allen_diameter.model.kernel import sigma_r

# ---- CONFIG ----
TABLE_DELTA = (0.0, 0.28, 0.56, 0.84)                 # um, the pilot's table (Cell 7, specimen 529878215)
TABLE_SIGMA = (0.0800, 0.1536, 0.3924, 0.5620)        # um
SLOPES = ((0.79, "far slope 0.79 (configured, D-024)", "-"), (0.397, "far slope 0.397 (pilot's diagnostic fit, 2 nodes)", "--"))
DELTA_MAX = 2.4                                        # um
# ----------------

base = default_config()
delta = np.linspace(-DELTA_MAX, DELTA_MAX, 961)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for gamma, label, ls in SLOPES:
    rcfg = dataclasses.replace(base.renderer, kernel_table_delta_um=TABLE_DELTA, kernel_table_sigma_um=TABLE_SIGMA,
                               sigma_r0_um=TABLE_SIGMA[0], kernel_continuation_slope=gamma)
    ax.plot(delta, sigma_r(delta, rcfg), ls, color="#1f5fa8", lw=1.6, label=label)
kn = np.array(TABLE_DELTA)
ax.plot(np.r_[-kn[::-1], kn], np.r_[TABLE_SIGMA[::-1], TABLE_SIGMA], "o", color="k", ms=4, label="table knots (measured on the pilot)")
ax.axvspan(-0.84, 0.84, color="0.92", zorder=0, label="the table's range (near field)")
ax.set_xlabel(r"defocus $\delta = \zeta_j - z_k$ (um)")
ax.set_ylabel(r"kernel SD $\sigma_{\rm r}(\delta)$ (um)")
ax.set_ylim(0, 1.8)
ax.legend(fontsize=7.5, loc="upper center")
fig.savefig(FIG / "fig02_kernel.png", dpi=170, bbox_inches="tight")
print("wrote", FIG / "fig02_kernel.png")
