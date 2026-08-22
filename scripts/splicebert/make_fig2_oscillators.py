#!/usr/bin/env python
"""FIGURE 2 -- Two learned oscillators in one position table.

Reads ONLY results/phase-lattice/fig_report_data.json.

Panel B is the one that carries the argument, and it is worth saying why it looks the way it
does.  Bin k of the table is not a number -- it is a complex VECTOR in C^512, one phase per
embedding dimension.  Its contribution sweeps an ELLIPSE in the plane spanned by Re F[k] and
Im F[k].  Both carriers come out as near-perfect CIRCLES (0.977 and 0.992).  A circle means
every phase is realised in some embedding direction, so THE BIN HAS NO PHASE OF ITS OWN.  The
table supplies a rotating coordinate system; a head chooses a phase from it through its QK
bilinear form.  That is why this report says "carrier", never "phase".

And the period-3 carrier lands on exactly THREE points -- 1026/342 = 3 exactly, so position p
visits only three angles, corresponding to position modulo 3.  This geometry does not identify
codon phase or reading frame.  The ~10.63
carrier has no such luck: 1026/96 = 10.6875 is not an integer, so it sweeps the whole circle.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/splicebert"))
from figstyle import (BLUE, ORANGE, RED, SLATE, INK, SEC, MUT, RULE, PANEL, MIST,  # noqa: E402
                      Canvas)

D = json.loads((REPO / "results/phase-lattice/fig_report_data.json").read_text())
OUT = REPO / "figures/splicebert/fig2_oscillators.svg"

W, H = 1440, 880
c = Canvas(W, H)

SP = D["spectra"]["final_1024_multispecies"]
per, rel = np.array(SP["period"]), np.array(SP["rel"])
P3B, CARB = set(D["bands"]["period3"]), set(D["bands"]["control"])

c.txt(W / 2, 40, "Two learned oscillators, in orthogonal subspaces of one table", size=22,
      fill=INK, anchor="middle", weight="700")
c.txt(W / 2, 64, "SpliceBERT.1024nt  -  position-embedding table, 1026 x 512", size=12.5,
      fill=MUT, anchor="middle")

# ============================================================ A. the full spectrum
AX, AY, AW, AH = 80, 118, 1280, 232
c.plabel(AX, AY - 14, "A", "Both carriers are narrowband spikes in a spectrum otherwise made "
                           "of slow drift")
c.rect(AX, AY, AW, AH, fill=PANEL)

lo_p, hi_p = math.log10(1.9), math.log10(1200)
lo_v, hi_v = math.log10(0.3), math.log10(1600)
px = lambda p: AX + (math.log10(p) - lo_p) / (hi_p - lo_p) * AW           # noqa: E731
py = lambda v: AY + AH - (math.log10(max(v, .31)) - lo_v) / (hi_v - lo_v) * AH   # noqa: E731

# the slow-mode region, neutral
c.rect(px(22), AY, AX + AW - px(22), AH, fill=MIST, op=.75)
c.txt(px(150), AY + 20, "Slow positional modes", size=11, fill=SEC, anchor="middle",
      weight="700")
c.txt(px(150), AY + 36, "period > 22 nt", size=10, fill=MUT, anchor="middle")

for gv in (1, 10, 100, 1000):
    c.line(AX, py(gv), AX + AW, py(gv), stroke=RULE, w=1)
    c.txt(AX - 8, py(gv) + 4, f"{gv}x", size=10, fill=MUT, anchor="end", cls="mono")
for gp in (2, 3, 5, 10, 20, 50, 100, 300, 1000):
    c.txt(px(gp), AY + AH + 18, str(gp), size=10, fill=MUT, anchor="middle", cls="mono")
c.txt(AX + AW / 2, AY + AH + 38, "Period  (nt, log scale)", size=11.5, fill=SEC, anchor="middle")
c.txt(AX - 46, AY + AH / 2, "Power / median bin", size=11.5, fill=SEC, anchor="middle", rot=-90)

# every bin, as a recessive stem
for k in range(1, len(rel)):
    if k in P3B or k in CARB:
        continue
    c.line(px(per[k - 1]), py(rel[k - 1]), px(per[k - 1]), py(0.31), stroke=MUT, w=1, op=.30)

# the two carriers
for band, col, name in ((P3B, BLUE, "period-3"), (CARB, ORANGE, "carrier")):
    for k in sorted(band):
        c.line(px(per[k - 1]), py(rel[k - 1]), px(per[k - 1]), py(0.31), stroke=col, w=2.2)

kp3 = max(P3B, key=lambda k: rel[k - 1])
kcar = max(CARB, key=lambda k: rel[k - 1])
for k, col, lab, sub in ((kp3, BLUE, "Period 3", f"{rel[kp3-1]:.0f}x median"),
                         (kcar, ORANGE, "~10.63", f"{rel[kcar-1]:.0f}x median")):
    c.dot(px(per[k - 1]), py(rel[k - 1]), 5, col)
    c.txt(px(per[k - 1]), py(rel[k - 1]) - 26, lab, size=12, fill=col, anchor="middle",
          weight="700")
    c.txt(px(per[k - 1]), py(rel[k - 1]) - 12, sub, size=10, fill=SEC, anchor="middle",
          cls="mono")

# ============================================================ B. carrier geometry
BX, BY, BW, BH = 80, 424, 596, 330
c.plabel(BX, BY - 14, "B", "Each carrier is a rotating 2-D coordinate system, not a phase")
c.rect(BX, BY, BW, BH, fill=PANEL)

for i, (key, col, ttl) in enumerate((("period3", BLUE, "Period-3 carrier  (bin 342)"),
                                     ("carrier", ORANGE, "~10.63 carrier  (bin 96)"))):
    pl = D["planes"][key]
    xy = np.array(pl["xy"])
    cx, cy = BX + 152 + i * 292, BY + 142
    R = 82
    sc = R / np.abs(xy).max()
    c.ring(cx, cy, R, RULE, w=1)
    c.line(cx - R - 12, cy, cx + R + 12, cy, stroke=RULE, w=1)
    c.line(cx, cy - R - 12, cx, cy + R + 12, stroke=RULE, w=1)
    uniq = np.unique(np.round(xy, 6), axis=0)
    if len(uniq) <= 6:                       # period 3 -> exactly three stations
        for j, (x, y) in enumerate(uniq):
            ang = math.atan2(y, x)
            order = int(round((ang % (2 * math.pi)) / (2 * math.pi / 3))) % 3
            c.dot(cx + x * sc, cy - y * sc, 8, col)
            lr = R + 32                       # clear the dot, not a multiple of its radius
            c.txt(cx + math.cos(ang) * lr, cy - math.sin(ang) * lr + 4,
                  f"codon {order}", size=10, fill=col, anchor="middle", weight="700")
    else:
        for x, y in xy[::4]:
            c.dot(cx + x * sc, cy - y * sc, 2.4, col, rw=0, op=.75)
    c.txt(cx, BY + 26, ttl, size=11.5, fill=INK, anchor="middle", weight="700")
    c.txt(cx, BY + 292, "Three stations, 120 deg apart" if len(uniq) <= 6
          else "Sweeps every phase", size=10, fill=SEC, anchor="middle")
    c.txt(cx, BY + 312, f"circularity  {pl['circularity']:.3f}", size=11, fill=col,
          anchor="middle", weight="700", cls="mono")

# ============================================================ C. orthogonality
CX, CY_, CW_, CH_ = 712, 424, 300, 330
c.plabel(CX, CY_ - 14, "C", "Disjoint subspaces")
c.rect(CX, CY_, CW_, CH_, fill=PANEL)

bins = D["overlaps"]["matrix_bins"]
M = np.array(D["overlaps"]["matrix"])
n = len(bins)
CELL = 31
GX, GY = CX + 82, CY_ + 60


def ramp(v):
    v = max(0.0, min(1.0, v))
    a, b = (242, 245, 247), (22, 35, 46)
    return "#%02X%02X%02X" % tuple(round(a[i] + (b[i] - a[i]) * v) for i in range(3))


for i in range(n):
    for j in range(n):
        c.rect(GX + j * CELL, GY + i * CELL, CELL - 2, CELL - 2, fill=ramp(M[i, j]), rx=2)
        c.txt(GX + j * CELL + (CELL - 2) / 2, GY + i * CELL + (CELL - 2) / 2 + 3.5,
              f"{M[i, j]:.2f}".lstrip("0") if M[i, j] < .995 else "1",
              size=8.5, fill="#FFFFFF" if M[i, j] > .5 else SEC, anchor="middle", cls="mono")
for i, b in enumerate(bins):
    col = BLUE if b in P3B else ORANGE
    c.txt(GX - 8, GY + i * CELL + 20, str(b), size=9.5, fill=col, anchor="end", cls="mono")
    c.txt(GX + i * CELL + 16, GY - 8, str(b), size=9.5, fill=col, anchor="middle", cls="mono")
c.txt(GX + n * CELL / 2, GY - 26, "Fourier bin", size=10.5, fill=SEC, anchor="middle")
c.rect(GX, GY, 3 * CELL - 2, 3 * CELL - 2, stroke=BLUE, sw=2, rx=2)
c.rect(GX + 3 * CELL, GY + 3 * CELL, 3 * CELL - 2, 3 * CELL - 2, stroke=ORANGE, sw=2, rx=2)
c.txt(CX + CW_ / 2, CY_ + 292, f"period-3 vs carrier  =  {D['overlaps']['p3_vs_carrier']:.4f}",
      size=11, fill=INK, anchor="middle", weight="700", cls="mono")
c.txt(CX + CW_ / 2, CY_ + 312, "a bin vs its own sidebands  =  "
      f"{D['overlaps']['p3_vs_p3_sidebands']:.2f}", size=10, fill=MUT, anchor="middle",
      cls="mono")

# ============================================================ D. the grid label was not the answer
DX, DY, DW, DH = 1048, 424, 312, 330
c.plabel(DX, DY - 14, "D", "10.67 was a grid label")
c.rect(DX, DY, DW, DH, fill=PANEL)

sb = D["subbin"]
lo_w, hi_w = 10.24, 10.74
wx = lambda t: DX + 40 + (t - lo_w) / (hi_w - lo_w) * (DW - 80)          # noqa: E731
AXY = DY + 236
c.line(wx(lo_w), AXY, wx(hi_w), AXY, stroke=SEC, w=1.4)
for t in (10.3, 10.5, 10.7):
    c.line(wx(t), AXY, wx(t), AXY + 5, stroke=SEC, w=1.2)
    c.txt(wx(t), AXY + 19, f"{t:.1f}", size=9.5, fill=MUT, anchor="middle", cls="mono")
c.txt(DX + DW / 2, AXY + 38, "Wavelength  (nt)", size=11, fill=SEC, anchor="middle")

rows = [("510-row grid", sb["stage1 multi-species (510nt)"]["bin_label_wavelength"],
         sb["stage1 multi-species (510nt)"]["wavelength"]),
        ("1026-row grid", sb["final (1024nt)"]["bin_label_wavelength"],
         sb["final (1024nt)"]["wavelength"])]
for i, (nm, lab, fit) in enumerate(rows):
    yy = DY + 68 + i * 66
    c.txt(DX + 16, yy - 22, nm, size=10.5, fill=SEC, weight="700")
    c.ring(wx(lab), yy, 5.5, MUT, w=1.6)
    c.txt(wx(lab) + 10, yy + 4, f"{lab:.3f}", size=9, fill=MUT, cls="mono")
    c.dot(wx(fit), yy, 5.5, ORANGE)
    c.txt(wx(fit) - 10, yy + 4, f"{fit:.3f}", size=9, fill=ORANGE, anchor="end", cls="mono",
          weight="700")
    c.arrow(wx(lab) - 7, yy, wx(fit) + 8, yy, stroke=MUT, w=1.2, head=4)

cw = sb["consensus_wavelength"]
c.line(wx(cw), DY + 140, wx(cw), AXY, stroke=ORANGE, w=1.4, dash="4 3")
c.dot(wx(cw), AXY, 6, ORANGE)
c.txt(wx(cw), AXY - 10, f"{cw:.3f}", size=11, fill=ORANGE, anchor="middle", weight="700",
      cls="mono")
c.txt(wx(cw), DY + 200, "Both grids agree", size=10, fill=ORANGE, anchor="middle", weight="700")

bw = sb["biology_wavelength"]
c.line(wx(bw), DY + 198, wx(bw), AXY, stroke=RED, w=1.4, dash="4 3")
c.dot(wx(bw), AXY, 6, RED)
c.txt(wx(bw), AXY - 10, f"{bw:.3f}", size=11, fill=RED, anchor="middle", weight="700", cls="mono")
c.txt(wx(bw), DY + 170, "Nucleosomal", size=10, fill=RED, anchor="middle", weight="700")
c.txt(wx(bw), DY + 186, "(a different signal)", size=9, fill=MUT, anchor="middle")

c.swatch(DX + 16, DY + 300, "grid bin label", MUT)
c.swatch(DX + 152, DY + 300, "measured, sub-bin", ORANGE)

c.line(80, H - 44, W - 80, H - 44, stroke=RULE, w=1)
c.txt(80, H - 24, "Circularity = sigma2/sigma1 of [Re F(k); Im F(k)].  Subspace overlap = mean "
                  "squared cosine of the principal angles.  Both carriers are near-circles, so "
                  "neither has a phase of its own.", size=10, fill=MUT)
c.render(OUT)
print(f"saved -> {OUT}")
