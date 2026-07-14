#!/usr/bin/env python
"""FIGURE 4 -- The ~10.63 circuit: real, specific, unexplained.

Reads ONLY results/phase-lattice/fig_report_data.json.

WHY PANEL A SHOWS BOTH MODELS.  The obvious objection to a null is that the instrument is
blind.  The answer is in the panel: the SAME estimator, the same frozen gates, the same 400
permutations, run on two checkpoints that differ only in TRAINING DATA.  The multi-species model
puts eight heads in the high-coherence / low-retention corner.  The human-only model puts none
there.  A blind instrument cannot do that.

WHY THERE IS NO VERTICAL SIGNIFICANCE LINE.  Significance is NOT an m_T cutoff -- it is BH-FDR
over each head's OWN permutation null.  Several heads with HIGH coherence are not significant,
because their own spiky residual reproduces that coherence when shuffled.  Drawing a vertical
threshold would tell a lie that is convenient for the story.  Significance is drawn as fill.

p-VALUES ARE EXACT ADD-ONE, floor 1/(400+1) = 0.0025.  Never 0.0000.  BH-FDR q = 0.05 is applied
across the whole scanned head family (79 heads multi-species, 75 human-only).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/splicebert"))
from figstyle import (ORANGE, RED, SLATE, INK, SEC, MUT, RULE, PANEL, MIST, SAND,  # noqa: E402
                      OLIVE, ASH, LAYER, Canvas)

D = json.loads((REPO / "results/phase-lattice/fig_report_data.json").read_text())
OUT = REPO / "figures/splicebert/fig4_carrier_circuit.svg"

W, H = 1440, 940
c = Canvas(W, H)
BM = D["band_mass"]
COLLAPSE = D["levelA_frozen"]["collapse_max"]

c.txt(W / 2, 40, "The ~10.63 carrier is read through the softmax - and only in one checkpoint",
      size=22, fill=INK, anchor="middle", weight="700")
c.txt(W / 2, 64, "Same estimator, same frozen gates, two checkpoints that differ only in "
                 "training data", size=12.5, fill=MUT, anchor="middle")

# ============================================================ A. reader landscape
AX, AY, AW, AH = 80, 118, 636, 372
c.plabel(AX, AY - 14, "A", "Eight heads in the corner - and in the human-only twin, none")
c.rect(AX, AY, AW, AH, fill=PANEL)

SW = (AW - 46) / 2
for i, (mdl, ttl) in enumerate((("multispecies", "Multi-species  (SpliceBERT.510nt)"),
                                ("human_only", "Human-only  (SpliceBERT-human.510nt)"))):
    ox, oy, ow, oh = AX + 40 + i * (SW + 6), AY + 58, SW - 44, 232
    sx = lambda v: ox + min(v, 1.0) * ow                                  # noqa: E731
    sy = lambda v: oy + oh - min(max(v, 0), 1.2) / 1.2 * oh               # noqa: E731
    c.rect(ox, oy, ow, oh, fill="#FFFFFF")
    for gv in (0, .25, .5, .75, 1.0):
        c.line(ox, sy(gv), ox + ow, sy(gv), stroke=RULE, w=1)
        if i == 0:
            c.txt(ox - 7, sy(gv) + 4, f"{gv:.2f}", size=8.5, fill=MUT, anchor="end", cls="mono")
    for gv in (0, .5, 1.0):
        c.txt(sx(gv), oy + oh + 15, f"{gv:.1f}", size=8.5, fill=MUT, anchor="middle", cls="mono")
    # the ONE fixed threshold there is: the carrier-deletion collapse criterion
    c.line(ox, sy(COLLAPSE), ox + ow, sy(COLLAPSE), stroke=INK, w=1.4, dash="5 3")
    if i == 0:
        c.txt(ox + ow - 4, sy(COLLAPSE) - 6, f"collapse criterion  {COLLAPSE:.2f}", size=8.5,
              fill=INK, anchor="end", weight="700", cls="mono")
    rows = D["levelA"][mdl]
    for r in sorted(rows, key=lambda x: x["LEVEL_A"]):
        rad = 2.6 + 7 * math.sqrt(min(BM[mdl].get(r["head"], 0.0), 1.0))
        if r["LEVEL_A"]:
            c.dot(sx(r["m_T"]), sy(r["kept_carrier"]), rad, ORANGE, ring=INK, rw=1.4)
        elif r["significant"]:
            c.dot(sx(r["m_T"]), sy(r["kept_carrier"]), rad, "#FFFFFF", ring=ORANGE, rw=1.6)
        else:
            c.dot(sx(r["m_T"]), sy(r["kept_carrier"]), rad, LAYER[r["layer"]], ring="#FFFFFF",
                  rw=.8, op=.9)
    nA = sum(1 for r in rows if r["LEVEL_A"])
    c.txt(ox + ow / 2, AY + 26, ttl, size=10.5, fill=INK, anchor="middle", weight="700")
    c.txt(ox + ow / 2, AY + 42, f"{len(rows)} of 96 heads clear the 2% band-mass floor",
          size=9, fill=MUT, anchor="middle")
    c.txt(ox + ow / 2, oy + oh + 36, f"{nA} heads reach Level A" if nA else
          "0 heads reach Level A", size=11.5, fill=ORANGE if nA else SLATE, anchor="middle",
          weight="700")
c.txt(AX + AW / 2, AY + 58 + 232 + 55, "Post-softmax coherence  m_T  at T = 10.61", size=11,
      fill=SEC, anchor="middle")
c.txt(AX + 12, AY + 58 + 116, "Kept after deleting the carrier", size=11, fill=SEC,
      anchor="middle", rot=-90)
LGY = AY + 58 + 232 + 76
c.dot(AX + 34, LGY - 4, 5.5, ORANGE, ring=INK, rw=1.4)
c.txt(AX + 44, LGY, "Level A", size=9, fill=SEC)
c.dot(AX + 112, LGY - 4, 5.5, "#FFFFFF", ring=ORANGE, rw=1.6)
c.txt(AX + 122, LGY, "significant only", size=9, fill=SEC)
c.dot(AX + 224, LGY - 4, 5.5, LAYER[2])
c.txt(AX + 234, LGY, "not significant (shade = layer)", size=9, fill=SEC)
c.dot(AX + 412, LGY - 4, 3, MUT)
c.dot(AX + 424, LGY - 4, 6.5, MUT)
c.txt(AX + 436, LGY, "attention mass in the band", size=9, fill=SEC)

# ============================================================ B. the controls
BX, BY, BW, BH = 756, 118, 604, 372
c.plabel(BX, BY - 14, "B", "Carrier-dependent, content-independent")
c.rect(BX, BY, BW, BH, fill=PANEL)

A8 = sorted([r for r in D["levelA"]["multispecies"] if r["LEVEL_A"]], key=lambda x: -x["m_T"])
CONDS = [("kept_carrier", SLATE, "Carrier deleted"),
         ("kept_reciprocal", SAND, "Reciprocal carrier deleted"),
         ("kept_low_leak", OLIVE, "Low-leak band deleted"),
         ("kept_shuffled", ASH, "Content shuffled")]
gx0, gy0, gw, gh = BX + 46, BY + 44, BW - 66, 226
byv = lambda v: gy0 + gh - min(v, 1.15) / 1.15 * gh                       # noqa: E731
for gv in (0, .25, .5, .75, 1.0):
    c.line(gx0, byv(gv), gx0 + gw, byv(gv), stroke=RULE, w=1)
    c.txt(gx0 - 7, byv(gv) + 4, f"{gv*100:.0f}%", size=8.5, fill=MUT, anchor="end", cls="mono")
c.line(gx0, byv(1.0), gx0 + gw, byv(1.0), stroke=INK, w=1.4)
c.txt(gx0, byv(1.0) - 7, "intact = 100%  (reference)", size=9, fill=INK, weight="700",
      cls="mono")
GRP = gw / len(A8)
for i, r in enumerate(A8):
    for j, (k, col, _) in enumerate(CONDS):
        bw_ = (GRP - 12) / len(CONDS)
        bx = gx0 + i * GRP + 6 + j * bw_
        c.bar(bx, byv(r[k]), bw_ - 1.6, gy0 + gh - byv(r[k]), col, rx=2)
    c.txt(gx0 + i * GRP + GRP / 2, gy0 + gh + 15, r["head"], size=8.5, fill=INK,
          anchor="middle", weight="700")
    c.txt(gx0 + i * GRP + GRP / 2, gy0 + gh + 28, f"{r['p_exact']:.4f}", size=7.5, fill=SEC,
          anchor="middle", cls="mono")
c.txt(gx0 + gw / 2, gy0 + gh + 48, "Exact add-one permutation p, BH-FDR across 79 scanned heads",
      size=9.5, fill=MUT, anchor="middle")
for j, (_, col, lab) in enumerate(CONDS):
    c.swatch(BX + 30 + (j % 2) * 290, BY + 334 + (j // 2) * 20, lab, col)

# ============================================================ C. identity exclusions
CX, CY, CW, CH_ = 80, 552, 1280, 288
c.plabel(CX, CY - 14, "C", "Same neighbourhood, different wavelengths, different evidence")
c.rect(CX, CY, CW, CH_, fill=PANEL)

WV = D["wavelengths"]
# --- upper tier: the whole scan range, where the Kogan window cannot resolve anything
ux, uy, uw = CX + 60, CY + 58, CW - 200
kx = lambda t: ux + (t - 7.0) / (16.0 - 7.0) * uw                         # noqa: E731
c.line(ux, uy, ux + uw, uy, stroke=SEC, w=1.4)
for t in (7, 9, 11, 13, 15):
    c.line(kx(t), uy, kx(t), uy + 5, stroke=SEC, w=1.2)
    c.txt(kx(t), uy + 17, str(t), size=9, fill=MUT, anchor="middle", cls="mono")
lo, hi = WV["kogan_unresolvable"]["lo"], WV["kogan_unresolvable"]["hi"]
c.rect(kx(lo), uy - 30, kx(hi) - kx(lo), 25, fill=MIST, rx=3)
c.txt((kx(lo) + kx(hi)) / 2, uy - 13, "Kogan splice-junction phasing: real, but its wavelength "
      "is not identifiable at its coherence length", size=10, fill=SEC, anchor="middle",
      weight="700")
c.txt(ux + uw + 12, uy + 1, "51-nt window", size=9.5, fill=SEC, weight="700")
c.txt(ux + uw + 12, uy + 15, "dT ~ 4 nt at T = 10", size=9, fill=MUT, cls="mono")
c.txt(CX + 22, uy + 1, "Full", size=9.5, fill=MUT, anchor="middle")
c.txt(CX + 22, uy + 14, "scan", size=9.5, fill=MUT, anchor="middle")

# --- the zoom.  Connectors run ABOVE the zoom axis; every marker label hangs BELOW it, so the
#     two never fight for the same space.
zlo, zhi = 10.10, 10.80
zx, zy, zw = CX + 60, CY + 150, CW - 200
c.rect(kx(zlo), uy - 4, max(kx(zhi) - kx(zlo), 2), 8, fill=INK, op=.16, rx=2)
c.path([(kx(zlo), uy + 5), (zx, zy - 12)], stroke=MUT, w=1, dash="3 3")
c.path([(kx(zhi), uy + 5), (zx + zw, zy - 12)], stroke=MUT, w=1, dash="3 3")

wx = lambda t: zx + (t - zlo) / (zhi - zlo) * zw                          # noqa: E731
c.line(zx, zy, zx + zw, zy, stroke=SEC, w=1.4)
for t in (10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8):
    c.line(wx(t), zy, wx(t), zy + 5, stroke=SEC, w=1.2)
    c.txt(wx(t), zy + 18, f"{t:.1f}", size=9, fill=MUT, anchor="middle", cls="mono")
c.txt(zx + zw + 12, zy + 4, "nt", size=10, fill=SEC)
c.txt(CX + 22, zy + 4, "Zoom", size=9.5, fill=MUT, anchor="middle")

MK = [(WV["human_only"]["T"], SLATE, "Human-only component", "0 of 14 preregistered readers",
       "DOES NOT REACH LEVEL A", False),
      (WV["nucleosomal"]["T"], RED, "Nucleosomal signal",
       f"z = +{WV['nucleosomal']['z']:.0f}, depth {WV['nucleosomal']['depth']*100:.1f}%",
       "measured in DNA, not in the model", True),
      (WV["carrier"]["T"], ORANGE, "Multi-species carrier", "8 post-softmax readers",
       "LEVEL A", True)]
for t, col, name, sub, badge, solid in MK:
    c.line(wx(t), zy + 5, wx(t), zy + 36, stroke=col, w=1.4, dash=None if solid else "4 3")
    if solid:
        c.dot(wx(t), zy, 6.5, col)
    else:
        c.ring(wx(t), zy, 6, col, w=2)
        c.line(wx(t) - 4.2, zy - 4.2, wx(t) + 4.2, zy + 4.2, stroke=col, w=1.8)
    c.txt(wx(t), zy + 52, f"{t:.3f}", size=12, fill=col, anchor="middle", weight="700",
          cls="mono")
    c.txt(wx(t), zy + 68, name, size=10.5, fill=col, anchor="middle", weight="700")
    c.txt(wx(t), zy + 83, sub, size=9, fill=SEC, anchor="middle")
    c.txt(wx(t), zy + 97, badge, size=9, fill=col if solid else SLATE, anchor="middle",
          weight="700" if badge.isupper() else None)

c.line(80, H - 44, W - 80, H - 44, stroke=RULE, w=1)
c.txt(80, H - 24, "The multi-species carrier is NOT the measured nucleosomal periodicity - the "
                  "two differ by 0.33 nt, and the model's is fitted to +/-0.005.  Its biological "
                  "or training origin remains unresolved.  At 400 permutations BH needs >= 4 "
                  "heads at the p-floor before any is called significant.", size=10, fill=MUT)
c.render(OUT)
print(f"saved -> {OUT}")
