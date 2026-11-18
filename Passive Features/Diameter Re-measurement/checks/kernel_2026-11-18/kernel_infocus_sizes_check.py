#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: if the true in-focus kernel width is larger than the table's
0.080 um, is the inflation of the fitted diameter the same for every size? One flat tube of true diameter d,
rendered with sigma_true(delta)^2 = sigma_T(delta)^2 + (s0^2 - 0.080^2), fitted with the table T (the pipeline's
model). The absorption is chosen so that mu x d = 0.4 for every size. Set-up of kernel_sensitivity_check.py.

Run:  python3 kernel_infocus_sizes_check.py <src>
"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("ks", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "kernel_sensitivity_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SIZES, S0 = (0.5, 0.8, 1.2, 2.0), (0.22, 0.30)
for s0 in S0:
    extra = s0 ** 2 - m.SIGMA_T[0] ** 2
    cfg = m.cfg_for([np.sqrt(s ** 2 + extra) for s in m.SIGMA_T], m.SLOPE)
    print("\ntrue in-focus sigma %.2f um (table 0.08)" % s0, flush=True)
    for d in SIZES:
        th = np.array([d, 0.4 / d, 0.0, 0.0])
        data = m.B_GL * m.tau(th, cfg, "s%.2f" % s0)
        x, rms = m.fit(data, th * [1.3, 0.8, 1, 1])
        print("  true d %.2f um: fitted %.3f um = %.2f x true, mu x d %.3f (true 0.400), rms %.2f gl"
              % (d, x[0], x[0] / d, x[0] * x[1], rms), flush=True)
