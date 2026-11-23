"""Rewrite README.md's two tables from compare.out and bias.out (between the <!-- compare --> and <!-- bias --> marks)."""
import json
import re


def rows(fn):
    return [json.loads(l) for l in open(fn) if l.startswith("{")]


cmp_lines = ["| d (um) | tilt (rad) | heading (rad) | shape | max abs(iso - sep) | RMS | max abs(iso - Gauss) | "
             "pairs left out (Gauss), max |", "|---|---|---|---|---|---|---|---|"]
for r in rows("compare.out"):
    cmp_lines.append("| %.1f | %.1f | %.2f | %s | %.3f | %.3f | %.2f | %.2f |" % (
        r["d"], r["phi"], r["theta"], r["shape"], r["max_iso_sep_noise"], r["rms_iso_sep_noise"],
        r["max_iso_gauss_noise"], r["excluded_pairs_max_noise"]))
bias_lines = ["| d (um) | tilt (rad) | heading (rad) | shape | bias of d, shape fitted | in SE | bias of d, shape held | "
              "in SE | SE(d)/d | RMS residual (noise SD) |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows("bias.out"):
    bias_lines.append("| %.1f | %.1f | %.2f | %s | %+.2f %% | %+.2f | %+.2f %% | %+.2f | %.2f %% | %.2f |" % (
        r["d"], r["phi"], r["theta"], r["shape"], 100 * r["bias_d_rel"], r["bias_d_over_se"],
        100 * r["bias_d_rel_shape_known"], r["bias_d_over_se_shape_known"], 100 * r["se_d_rel"], r["rms_r"]))
s = open("README.md").read()
for tag, lines in (("compare", cmp_lines), ("bias", bias_lines)):
    block = "<!-- %s -->\n%s\n<!-- /%s -->" % (tag, "\n".join(lines), tag)
    if "<!-- %s -->" % tag in s:
        s = re.sub(r"<!-- %s -->.*?<!-- /%s -->" % (tag, tag), lambda m: block, s, flags=re.S)
    else:
        s = s.replace("%s_TABLE" % tag.upper(), block)
open("README.md", "w").write(s)
print(len(cmp_lines) - 2, "comparison rows,", len(bias_lines) - 2, "bias rows")
