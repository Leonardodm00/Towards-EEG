"""
robustness_registration.py -- the second control of registration_check.

smoke_allen_image.py checks one synthetic stack. This sweeps several
independent stacks (different speck layouts and noise) and, on each, four
cases with known answers. It reports the error rates rather than a single pass,
because a statistical test is only as good as its behaviour across draws.

Run:  python robustness_registration.py      (~1-2 min)
"""
import numpy as np
import smoke_allen_image as sm

RES = sm.RES0


def displaced(swc, d_um):
    sw = swc.copy()
    x, y = sw["x"].values / RES, sw["y"].values / RES
    tc, tr = np.gradient(x), np.gradient(y)
    n = np.hypot(tc, tr)
    tc, tr = tc / n, tr / n
    d = d_um / RES
    sw["x"] = (x - d * tr) * RES
    sw["y"] = (y + d * tc) * RES
    return sw


def main():
    seeds = [11, 23, 37, 41, 59]
    rows = []
    for sd, amp in [(s_, 25.0) for s_ in seeds] + [(s_, 0.0) for s_ in seeds]:
        R = sm._rc_stack(sd, amp)
        base = R["swc"]
        cases = [("on", base, "ON", 0.0),
                 ("+1.0um", displaced(base, +1.0), "ALONGSIDE", -1.0),
                 ("-2.0um", displaced(base, -2.0), "ALONGSIDE", +2.0)]
        empty = base.copy()
        empty["y"] = empty["y"] + 300 * RES
        cases.append(("empty", empty, "NOT ON", np.nan))
        for name, sw, want, s_exp in cases:
            name = ("wavy " if amp else "strght ") + name
            chk = sm._rc_run(sw, seed=sd, null_seed=sd, amp=amp)
            v = chk["verdict"]
            got = ("ON" if v.startswith("ON THE") else "ALONGSIDE" if v.startswith("ALONGSIDE")
                   else "NOT ON" if v.startswith("NOT ON") else v.split(":")[0])
            s_err = (chk["lateral_offset_um"] - s_exp) if np.isfinite(s_exp) else np.nan
            ok = (got == want) and (not np.isfinite(s_err) or abs(s_err) <= 0.3)
            rows.append((sd, name, want, got, chk["lateral_offset_um"], s_err,
                         chk["p_value"], chk["coverage"], ok))
    print("%-5s %-14s %-10s %-10s %8s %7s %7s %6s  %s"
          % ("seed", "case", "expected", "got", "s*(um)", "s err", "p", "cov", ""))
    for sd, name, want, got, s_, e_, p_, c_, ok in rows:
        print("%-5d %-14s %-10s %-10s %+8.2f %7s %7.3f %5.0f%%  %s"
              % (sd, name, want, got, s_, ("%+.2f" % e_) if np.isfinite(e_) else "-",
                 p_, 100 * c_, "ok" if ok else "MISS"))
    n_ok = sum(r[-1] for r in rows)
    fp = sum(1 for r in rows if r[1].endswith("empty") and r[3] != "NOT ON")
    fn = sum(1 for r in rows if not r[1].endswith("empty") and r[3] == "NOT ON")
    errs = [abs(r[5]) for r in rows if np.isfinite(r[5])]
    fp_p_only = sum(1 for r in rows if r[1].endswith("empty") and r[6] <= 0.05)
    print("\nwithout the coverage gate (p alone), empty tissue would pass in %d/%d stacks"
          % (fp_p_only, 2 * len(seeds)))
    print("%d/%d correct | false positives on empty tissue: %d/%d | missed real "
          "processes: %d/%d | max |s* error| = %.2f um"
          % (n_ok, len(rows), fp, 2 * len(seeds), fn, 6 * len(seeds), max(errs)))
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
