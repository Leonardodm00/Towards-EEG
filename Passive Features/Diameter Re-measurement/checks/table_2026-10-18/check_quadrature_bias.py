"""check_quadrature_bias.py -- how much the D-052.2 quadrature rule (N = 848) biases the SGT fit's d_hat
(2026-10-18; standalone, not pipeline code). Noise-free profiles of the SGT model computed with 16 N
Gauss-Legendre nodes (the 'truth'), fitted by fit_sgt.fit_plane at the rule's N; reported: the model error at N
on the truth (max over Block 5's 53 offsets, grey levels) and d_hat - d (um) of the IRLS fit.
Run: python3 check_quadrature_bias.py "<workstream>"
"""
import sys, time
import numpy as np
ws = sys.argv[1]
sys.path.insert(0, ws + "/src")
from allen_diameter.config import default_config
from allen_diameter.analysis import fit_sgt as FS, profiles
cfg = default_config().measure
N = FS.n_nodes_for(cfg)
v = profiles.profile_offsets(cfg)
phi, B, mu, v0 = 0.2, 180.0, 1.0, 0.1
print("N = %d; truth model at 16 N; fit at N (the D-052.2 rule)" % N)
print("%-6s %-5s %-18s %-12s %-12s %-10s %s" % ("d um", "w um", "shape", "model err gl", "d_hat - d um", "status", "fit s"))
for shape in ((1.0, 0.0, 1.0), (2.0, 0.5, 1.5), (10.0, 0.0, 1.0), (1.5, 0.0, 1.0)):
    for d in (0.6, 1.5, 3.0, 5.5):
        for w in (0.3, 0.5):
            q = FS.to_q(d, mu, v0, w, *shape)
            I = FS.sgt_model(v, q, phi, B, 16 * N)
            em = float(np.max(np.abs(FS.sgt_model(v, q, phi, B, N) - I)))
            t0 = time.perf_counter()
            r = FS.fit_plane(v, I, phi, B, cfg).final
            print("%-6.1f %-5.1f %-18s %-12.1e %-+12.1e %-10s %.1f" % (d, w, shape, em, r.d_um - d, r.status, time.perf_counter() - t0))
