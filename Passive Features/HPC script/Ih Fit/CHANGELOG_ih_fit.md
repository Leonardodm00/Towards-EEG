# Ih Fit -- stage log

| Date | Stage | Change |
|---|---|---|
| 2026-09-22 | 0 | Folder created as a byte-copy of `Biological Fit/` (monolith `cmp`-identical). `regression_passive_identity.py` added: run-B configuration (dt 0.1 ms, 60 ms window, 2 smallest steps, 12 mV cap), same seed, loss at 20 random log-theta points and `gp_minimize` output compared between the two folders. PASS in the assistant's sandbox on a synthetic archive cell (max dL = 0, x and fun identical). Cluster check (Step-0 shell block of the v1 handoff) and the run-B config record are still to be done by the user. |
| 2026-09-22 | 1 | `mod/Ih.mod`, `mod/Ih_human.mod` with RANGE `vshift`, `vshift_minf`, `tau_scale`; `human_ih_params.py` extended (section 5); `ih_mechanism.py`; `smoke_ih_fit.py` S1-S5 + S10. All 6 checks PASS with NEURON 9.0.2 (sandbox). Sign convention fixed and pinned by S5: Kalmbach = `vshift = +20`. Axon stub carries no I_h by default (`IhSpec.regions`). |
