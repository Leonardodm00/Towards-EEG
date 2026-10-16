#!/usr/bin/env python3
"""Typeset a LaTeX body (amsmath) into a PNG for the chat: standalone class,
pdflatex, pdftoppm. Usage:

    python render_tex.py body.tex out_basename [--dpi 230] [--width-cm 17.6]

Writes out_basename.tex, .pdf and .png next to out_basename; prints the PNG
size, or the first LaTeX error. Pure ASCII.
"""
import argparse
import os
import subprocess
import sys

PREAMBLE = r"""\documentclass[11pt,varwidth=%(width)scm,border=14pt]{standalone}
\usepackage[Euler]{upgreek}
\usepackage[fleqn]{amsmath}
\usepackage{amssymb,mathtools,bm}
\setlength{\mathindent}{0pt}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator{\diag}{diag}
\newcommand{\um}{\,\upmu\mathrm{m}}
\newcommand{\Bb}{\bar B_j(k)}
\newcommand{\Imod}{I_{\mathrm{model}}}
\setlength{\parindent}{0pt}
\begin{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body")
    ap.add_argument("out")
    ap.add_argument("--dpi", type=int, default=230)
    ap.add_argument("--width-cm", default="17.6")
    a = ap.parse_args()
    body = open(a.body).read()
    out_dir, base = os.path.dirname(os.path.abspath(a.out)), os.path.basename(a.out)
    tex = os.path.join(out_dir, base + ".tex")
    with open(tex, "w") as fh:
        fh.write(PREAMBLE % dict(width=a.width_cm) + body + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", base + ".tex"], cwd=out_dir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        errs = [ln for ln in r.stdout.splitlines() if ln.startswith("!") or ln.startswith("l.")]
        print("LaTeX failed:", "\n".join(errs[:6]))
        sys.exit(1)
    subprocess.run(["pdftoppm", "-png", "-r", str(a.dpi), "-singlefile", base + ".pdf", base], cwd=out_dir, check=True)
    from PIL import Image
    print("wrote", os.path.join(out_dir, base + ".png"), Image.open(os.path.join(out_dir, base + ".png")).size)


if __name__ == "__main__":
    main()
