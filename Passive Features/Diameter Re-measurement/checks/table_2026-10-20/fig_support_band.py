"""Figure for open point 11 (off_support): where phantoms exist along a real node's inversion line.
Plotted, not schematic: 1500 draws of (d, mu) from the D-024 laws at a fixed tilt phi = 0, with the
fitted optical depth alpha_hat replaced by the true alpha = mu d / cos(phi) (the table chat's stand-in;
the fit's scatter in alpha_hat blurs the band edges). The band is analytic from the mu range; the
inversion line is a real node's own alpha_hat_i (D-053 (iii)); the example node is the C5 case of
checks/table_2026-10-18/check_table_se_coverage.py (d = 0.5 um, mu = 4 /um, alpha_hat = 2.03).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, N = 20261020, 1500
D_RANGE, MU_RANGE, PHI = (0.2, 4.0), (0.3, 3.0), 0.0
NODE_D, NODE_ALPHA = 0.5, 2.03

rng = np.random.default_rng(SEED)
d = np.exp(rng.uniform(np.log(D_RANGE[0]), np.log(D_RANGE[1]), N))
mu = np.exp(rng.uniform(np.log(MU_RANGE[0]), np.log(MU_RANGE[1]), N))
alpha = mu * d / np.cos(PHI)
ld = np.linspace(np.log(D_RANGE[0]), np.log(D_RANGE[1]), 2)
fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
ax.scatter(np.log(d), np.log(alpha), s=6, color="#8a8a8a", alpha=0.6, label="phantoms (1500 draws, $\\varphi$ = 0)")
ax.plot(ld, np.log(MU_RANGE[0]) + ld - np.log(np.cos(PHI)), color="#2a78d6", lw=1.5,
        label="band edges: $\\ln\\hat\\alpha = \\ln\\mu + \\ln d - \\ln\\cos\\varphi$, $\\mu$ = 0.3 and 3 /um")
ax.plot(ld, np.log(MU_RANGE[1]) + ld - np.log(np.cos(PHI)), color="#2a78d6", lw=1.5)
la = np.log(NODE_ALPHA)
lo = max(np.log(NODE_ALPHA * np.cos(PHI) / MU_RANGE[1]), np.log(D_RANGE[0]))
hi = min(np.log(NODE_ALPHA * np.cos(PHI) / MU_RANGE[0]), np.log(D_RANGE[1]))
ax.axhline(la, color="#ff1744", lw=1.2, ls="--", label="inversion line of a real node: $\\ln\\hat\\alpha_i$ = ln 2.03 fixed, $d$ solved")
ax.plot([lo, hi], [la, la], color="#ff1744", lw=5, alpha=0.6, label="support on that line: $d \\in [\\hat\\alpha_i\\cos\\varphi_i/\\mu_{\\max},\\ \\hat\\alpha_i\\cos\\varphi_i/\\mu_{\\min}]$")
ax.plot(np.log(NODE_D), la, marker="*", ms=14, color="#008300", ls="none",
        label="the C5 node: $d$ = 0.5 um, $\\hat\\alpha_i$ = 2.03 (needs $\\mu$ = 4.06 /um): off support, root 0.492 um unflagged")
ax.set_xlabel("$\\ln(d\\,/\\,\\mathrm{um})$  (true diameter of the phantom; the unknown solved for at a real node)")
ax.set_ylabel("$\\ln\\hat\\alpha$  (fitted optical depth, computed level; here the true $\\alpha$ as stand-in)")
ax.set_title("At fixed tilt the phantoms fill a band of slope 1; an inversion line is supported only inside it", fontsize=9.5)
ax.legend(fontsize=6.8, loc="lower right", framealpha=0.95)
ax.set_xlim(np.log(D_RANGE[0]) - 0.1, np.log(D_RANGE[1]) + 0.1)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_support_band.png")
print("support bracket on this line: ln d in [%.3f, %.3f] (d in [%.3f, %.3f] um); node at ln d = %.3f -> %s"
      % (lo, hi, np.exp(lo), np.exp(hi), np.log(NODE_D), "OFF support" if not (lo <= np.log(NODE_D) <= hi) else "on support"))
