import sys
import numpy as np
a = np.loadtxt(sys.argv[1], comments="#")
nid, typ, xyz, r, par = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:5], a[:, 5], a[:, 6].astype(int)
idx = {n: i for i, n in enumerate(nid)}
soma = np.flatnonzero(typ == 1)
c = xyz[soma].mean(axis=0)
pl = np.full(len(nid), np.nan)
pl[soma] = 0.0
changed = True
while changed:
    changed = False
    for i in range(len(nid)):
        if np.isfinite(pl[i]) or par[i] == -1:
            continue
        j = idx[par[i]]
        if np.isfinite(pl[j]):
            pl[i] = pl[j] + np.linalg.norm(xyz[i] - (c if typ[j] == 1 else xyz[j]))
            changed = True
seg = [(i, idx[par[i]]) for i in range(len(nid)) if par[i] != -1 and typ[idx[par[i]]] != 1 and typ[i] in (3, 4)]
L = np.array([np.linalg.norm(xyz[i] - xyz[j]) for i, j in seg])
inside = np.array([pl[i] <= 250.0 for i, j in seg])
Li = L[inside]
print("dendritic segments within 250 um: %d; spacing between successive nodes (um): median %.2f, 10-90 %% %.2f-%.2f, max %.2f"
      % (Li.size, np.median(Li), np.percentile(Li, 10), np.percentile(Li, 90), Li.max()))
print("share of segments longer than 2 um: %.1f %%; longer than 3 um: %.1f %%" % (100 * np.mean(Li > 2), 100 * np.mean(Li > 3)))
