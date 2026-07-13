#!/usr/bin/env python
"""FIGURE: the period-3 comb IS a cosine.  Renders straight from fig_phase_data.json.

House style: LABELS ARE SHORT.  A few words per component, sentence case, no sentences inside the
panels.  The narrative lives in fig_phase_wave.CAPTION.md, not in the artwork.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = json.loads((REPO / "results/phase-lattice/fig_phase_data.json").read_text())
OUT = REPO / "figures/splicebert/fig_phase_wave.svg"

BLUE, GRAY, GREEN, PURPLE = "#0B6FB8", "#9AA7B2", "#00845F", "#6A4A9C"
INK, SEC, MUT, RULE, PANEL = "#16232E", "#5A6672", "#8794A0", "#E1E7EC", "#F7F9FB"
W, H = 1400, 790
s: list[str] = []
w3 = 2 * math.pi / 3
OFFS = D["Rk_offsets"]
WAVE = D["wave"]["heads"]
HEADS = [f"L{l}H{h}" for l, h in D["combs"]]


def txt(x, y, t, *, size=11, fill=SEC, anchor="start", weight=None):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    w = f' font-weight="{weight}"' if weight else ""
    s.append(f'<text class="sans" x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
             f'{a}{w}>{t}</text>')


def line(x1, y1, x2, y2, *, stroke=RULE, w=1, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{op}"' if op != 1.0 else ""
    s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
             f'stroke-width="{w}"{d}{o}/>')


def rect(x, y, w_, h_, *, fill="none", stroke=None, rx=0):
    st = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w_:.1f}" height="{max(h_,0):.1f}" rx="{rx}" '
             f'fill="{fill}"{st}/>')


def path(pts, *, stroke, w=2, dash=None, op=1.0):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{op}"' if op != 1.0 else ""
    s.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{w}" '
             f'stroke-linejoin="round" stroke-linecap="round"{da}{o}/>')


def dot(x, y, r, fill, *, rw=1.5):
    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#FFFFFF" '
             f'stroke-width="{rw}"/>')


def plabel(x, y, letter, title):
    txt(x, y, letter, size=15, fill=INK, weight="700")
    txt(x + 18, y, title, size=13, fill=INK, weight="700")


def env(coef, k):
    return sum(c * k ** p for p, c in enumerate(coef))


s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')
txt(W / 2, 36, "The period-3 comb is a single cosine", size=21, fill=INK, anchor="middle",
    weight="700")

# ============================================================ A. the jaggedness was the connector
AX, AY, AW, AH = 70, 96, 620, 300
plabel(AX, AY - 12, "A", "The jaggedness was the connector")
rect(AX, AY + 10, AW, AH, fill=PANEL)

HD = "L2H15"
fi = WAVE[HD]["intact"]
y_int = D["Rk"][HD]["intact"]
lo_y, hi_y = 1.9, 5.5


def axx(k):
    return AX + 54 + (k + 12) / 24 * (AW - 84)


def axy(v):
    return AY + 26 + (AH - 66) - (v - lo_y) / (hi_y - lo_y) * (AH - 66)


for gv in (2, 3, 4, 5):
    line(AX + 50, axy(gv), AX + AW - 20, axy(gv), stroke="#E7EDF2")
    txt(AX + 44, axy(gv) + 3.5, str(gv), size=9.5, fill=MUT, anchor="end")
for k in (-12, -6, 0, 6, 12):
    txt(axx(k), AY + AH + 22, f"{k:+d}" if k else "0", size=9.5, fill=MUT, anchor="middle")
txt(AX + AW / 2 + 20, AY + AH + 40, "Attention offset k (nucleotides)", size=10, fill=SEC,
    anchor="middle")
s.append(f'<text class="sans" transform="translate({AX+22},{AY+10+AH/2}) rotate(-90)" '
         f'text-anchor="middle" font-size="10" fill="{SEC}">R(k)  (attention logits)</text>')

# 1. what we used to draw: straight lines between the samples
path([(axx(k), axy(y_int[j])) for j, k in enumerate(OFFS)], stroke="#C2CBD3", w=1.6)
# 2. the fitted envelope
fine = [-12 + 24 * t / 400 for t in range(401)]
path([(axx(k), axy(env(fi["env_coef"], k))) for k in fine], stroke="#8A97A2", w=1.4, dash="4 3")
# 3. the exact interpolant: envelope + ONE cosine
path([(axx(k), axy(env(fi["env_coef"], k) + fi["A"] * math.cos(w3 * k + fi["phi"])))
      for k in fine], stroke=BLUE, w=2.4)
# 4. the samples
for j, k in enumerate(OFFS):
    dot(axx(k), axy(y_int[j]), 3.0, BLUE, rw=1.4)

txt(AX + 66, AY + 34, HD, size=11, fill=PURPLE, weight="700")
txt(AX + 66, AY + 48, "The addresser", size=9, fill=MUT)
line(AX + 300, AY + 30, AX + 320, AY + 30, stroke="#C2CBD3", w=1.6)
txt(AX + 326, AY + 33.5, "Straight-line join", size=9.5, fill=SEC)
line(AX + 300, AY + 46, AX + 320, AY + 46, stroke="#8A97A2", w=1.4, dash="4 3")
txt(AX + 326, AY + 49.5, "Envelope", size=9.5, fill=SEC)
line(AX + 460, AY + 30, AX + 480, AY + 30, stroke=BLUE, w=2.4)
txt(AX + 486, AY + 33.5, "Envelope + one cosine", size=9.5, fill=SEC)
dot(AX + 470, AY + 46, 3.0, BLUE, rw=1.4)
txt(AX + 486, AY + 49.5, "Measured", size=9.5, fill=SEC)

# ============================================================ B. six heads, three codon positions
BX, BY, BW, BH = 740, 96, 590, 300
plabel(BX, BY - 12, "B", "Six heads, three codon positions")
rect(BX, BY + 10, BW, BH, fill=PANEL)

by_pos: dict[int, list[str]] = {0: [], 1: [], 2: []}
for hd in HEADS:
    by_pos[WAVE[hd]["intact"]["codon_position"]].append(hd)

# ONE STRIP PER HEAD.  Two heads that share a codon position have the SAME phase, so overlaying
# them just hides one curve under the other and reads as a missing line.  Give each its own row.
sh, sgap, ggap = 36, 4, 12
sy0 = BY + 34
LEFT = BX + 116


def bx_(k):
    return LEFT + (k + 8) / 16 * (BW - LEFT + BX - 22)


row = 0
for c in (0, 1, 2):
    ytop = sy0 + row * (sh + sgap) + c * ggap
    for hd in by_pos[c]:
        y0 = sy0 + row * (sh + sgap) + c * ggap
        mid = y0 + sh / 2
        f = WAVE[hd]["intact"]

        # the columns this head peaks on.  The grid is the SAME in every strip, so the one-
        # nucleotide step from one codon position to the next shows up as the wave sliding across.
        for k in range(-8, 9):
            if k % 3 == c:
                line(bx_(k), y0 + 1, bx_(k), y0 + sh - 1, stroke=BLUE, w=1.4, op=0.20)
        line(LEFT - 6, mid, bx_(8) + 6, mid, stroke="#D9E0E6")

        # amplitude-normalised: this panel is about PHASE.  Amplitude is panel C.
        path([(bx_(k), mid - math.cos(w3 * k + f["phi"]) * (sh / 2 - 4))
              for k in [-8 + 16 * t / 400 for t in range(401)]], stroke=BLUE, w=2.0)
        txt(BX + 106, mid + 3.5, hd, size=9.5, fill=PURPLE if hd == "L2H15" else SEC,
            weight="700" if hd == "L2H15" else None, anchor="end")
        row += 1

    ybot = sy0 + (row - 1) * (sh + sgap) + c * ggap + sh
    line(BX + 16, ytop + 2, BX + 16, ybot - 2, stroke="#C3CCD4", w=1.5)
    txt(BX + 24, (ytop + ybot) / 2 + 3.5, f"Codon {c}", size=10, fill=INK, weight="700")

ylast = sy0 + 6 * sh + 5 * sgap + 2 * ggap
for k in (-6, -3, 0, 3, 6):
    txt(bx_(k), ylast + 16, f"{k:+d}" if k else "0", size=9, fill=MUT, anchor="middle")
txt((LEFT + bx_(8)) / 2, ylast + 34, "Attention offset k (nucleotides)", size=10, fill=SEC,
    anchor="middle")

# ============================================================ C. the wave dies with the clock
CX, CY, CW, CH = 70, 470, 1260, 250
plabel(CX, CY - 12, "C", "Delete the clock and the wave is gone")
rect(CX, CY + 10, CW, CH, fill=PANEL)

hi_a = 2.6


def cy(v):
    return CY + 26 + (CH - 62) - v / hi_a * (CH - 62)


for gv in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
    line(CX + 60, cy(gv), CX + CW - 20, cy(gv), stroke="#E7EDF2")
    txt(CX + 54, cy(gv) + 3.5, f"{gv:.1f}", size=9.5, fill=MUT, anchor="end")
s.append(f'<text class="sans" transform="translate({CX+24},{CY+10+CH/2}) rotate(-90)" '
         f'text-anchor="middle" font-size="10" fill="{SEC}">Clock amplitude A (logits)</text>')

gw = (CW - 90) / 6
for i, hd in enumerate(HEADS):
    x0 = CX + 70 + i * gw
    trio = [(WAVE[hd]["intact"]["A"], BLUE), (WAVE[hd]["no_carrier"]["A"], GRAY),
            (WAVE[hd]["no_control"]["A"], GREEN)]
    for j, (v, col) in enumerate(trio):
        bx = x0 + gw / 2 - 33 + j * 22
        bh = max(cy(0) - cy(v), 2.0)            # a floor: a ~0 bar must read as a bar, not a gap
        rect(bx, cy(0) - bh, 20, bh, fill=col, rx=3)
    txt(x0 + gw / 2, CY + CH + 4, hd, size=10, fill=PURPLE if hd == "L2H15" else SEC,
        weight="700" if hd == "L2H15" else None, anchor="middle")

sm = D["wave"]["summary"]
for lx, lab, col in ((CX + 70, "Intact", BLUE),
                     (CX + 170, f"Clock deleted  ({sm['kept_clock_deleted']*100:.0f}%)", GRAY),
                     (CX + 380, f"Control band deleted  ({sm['kept_control_deleted']*100:.0f}%)",
                      GREEN)):
    rect(lx, CY + CH + 20, 11, 11, fill=col, rx=2)
    txt(lx + 16, CY + CH + 29.5, lab, size=9.5, fill=SEC)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" version="1.1">\n'
       '<title>The period-3 comb is a single cosine</title>\n'
       '<style type="text/css"><![CDATA[ .sans{font-family:"Helvetica Neue",Helvetica,Arial,'
       'sans-serif} text{shape-rendering:geometricPrecision} ]]></style>\n'
       + "\n".join(s) + "\n</svg>\n")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(svg)
print(f"saved -> {OUT}  ({len(svg)/1024:.0f} KB)")
