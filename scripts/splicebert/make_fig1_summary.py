#!/usr/bin/env python
"""FIGURE 1 -- The complete mechanistic summary.

Reads ONLY results/phase-lattice/fig_report_data.json.

TWO RULES THIS FIGURE OBEYS, BOTH OF THEM SCIENTIFIC RATHER THAN AESTHETIC:

  1. The two branches are drawn APART, with a measured gap between them, because their carrier
     planes are ORTHOGONAL in embedding space (subspace overlap 0.0003, against 0.99 for a bin
     and its own sidebands).  Drawing them merging into one code would assert something the
     measurement refutes.

  2. The ~10.63 branch STOPS at "function unknown."  There is no arrow from it to splicing, to
     chromatin, or to translation.  Eight heads read it through the softmax; what they do with
     it is not established, and the figure must not imply otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/splicebert"))
from figstyle import (BLUE, ORANGE, SLATE, INK, SEC, MUT, RULE, PANEL, MIST, Canvas)  # noqa: E402

D = json.loads((REPO / "results/phase-lattice/fig_report_data.json").read_text())
OUT = REPO / "figures/splicebert/fig1_summary.svg"

W, H = 1440, 860
c = Canvas(W, H)

med, dose = D["mediation"], D["dose"]
d1 = next(r for r in dose if r["band"] == "period-3" and r["alpha"] == 1.0)
d0 = next(r for r in dose if r["band"] == "period-3" and r["alpha"] == 0.0)
exon_dmg, intron_dmg = d1["exon"] - d0["exon"], d1["intron"] - d0["intron"]
_LEVEL_A = [r for r in D["levelA"]["multispecies"] if r["LEVEL_A"]]
nA = len(_LEVEL_A)


def kept_range(key: str) -> str:
    """Retention range across the Level-A heads, derived from the artifact.

    These four numbers used to be typed as literals, and one of them drifted: the
    carrier-deleted row read "40-44%" while the artifact says 27-44% (six of the eight
    heads are below 40%), contradicting fig4_carrier_circuit and EVIDENCE.md. Deriving
    them here means the card cannot disagree with its own data package again.
    """
    v = [r[key] for r in _LEVEL_A]
    return f"{min(v) * 100:.0f}-{max(v) * 100:.0f}%"
ov = D["overlaps"]["p3_vs_carrier"]

c.txt(W / 2, 40, "One position table, two carriers, two very different stories", size=22,
      fill=INK, anchor="middle", weight="700")
c.txt(W / 2, 64, "SpliceBERT", size=13, fill=MUT, anchor="middle")

# ------------------------------------------------------------------ the position table
TX, TY, TW, TH = W / 2 - 150, 92, 300, 54
c.rect(TX, TY, TW, TH, fill=PANEL, stroke=RULE, rx=5)
for i in range(14):
    c.line(TX + 12 + i * 20, TY + 12, TX + 12 + i * 20, TY + TH - 12, stroke=MUT, w=1, op=.35)
c.txt(W / 2, TY + 24, "Position-embedding table  P", size=13.5, fill=INK, anchor="middle",
      weight="700")
c.txt(W / 2, TY + 42, "1026 x 512", size=10.5, fill=MUT, anchor="middle", cls="mono")

# the split -- drawn as two separate stems, never touching
LX, RX = 400, 1040
SPLIT = TY + TH + 30
c.line(W / 2, TY + TH, W / 2, SPLIT, stroke=MUT, w=1.6)
c.path([(W / 2, SPLIT), (LX, SPLIT), (LX, SPLIT + 22)], stroke=BLUE, w=2.2)
c.path([(W / 2, SPLIT), (RX, SPLIT), (RX, SPLIT + 22)], stroke=ORANGE, w=2.2)

# the orthogonality note, sitting IN the gap that the two branches leave
c.rect(W / 2 - 118, SPLIT + 44, 236, 46, fill="#FFFFFF", stroke=RULE, rx=5)
c.txt(W / 2, SPLIT + 63, "Orthogonal subspaces", size=11.5, fill=INK, anchor="middle",
      weight="700")
c.txt(W / 2, SPLIT + 80, f"plane overlap  {ov:.4f}", size=10.5, fill=SEC, anchor="middle",
      cls="mono")


def node(x, y, w_, h_, title, sub, colour, *, fill=PANEL, dash=None, bold=True):
    c.rect(x - w_ / 2, y, w_, h_, fill=fill, stroke=colour, rx=6, sw=1.8, dash=dash)
    c.txt(x, y + (21 if sub else h_ / 2 + 4), title, size=12.5, fill=INK, anchor="middle",
          weight="700" if bold else None)
    if sub:
        c.txt(x, y + 38, sub, size=10.5, fill=SEC, anchor="middle")


def flow(x, y1, y2, colour, *, dash=None):
    c.arrow(x, y1, x, y2, stroke=colour, w=2.2, dash=dash, head=6)


# ------------------------------------------------------------------ LEFT: period-3 branch
y = SPLIT + 22
node(LX, y, 300, 52, "Period-3 carrier", "bins 339-345    circularity 0.977", BLUE)
y += 52
flow(LX, y, y + 30, BLUE)
y += 30
node(LX, y, 300, 52, "Six comb heads", "L0H0  L0H11  L1H10  L1H12  L2H2  L2H15", BLUE)
y += 52
flow(LX, y, y + 30, BLUE)
y += 30
node(LX, y, 300, 52, "L2H15", "the acceptor addresser", BLUE)
y += 52
flow(LX, y, y + 30, BLUE)
y += 30
node(LX, y, 300, 52, "L3H1", "the reader", BLUE)
READ_Y = y
y += 52
flow(LX, y, y + 30, BLUE)
y += 30
node(LX, y, 300, 52, "Acceptor prediction", "donor-acceptor coupling", BLUE, fill="#EAF3FA")
BEHAV_Y = y

# the donor enters the reader from the side -- content, not position
c.rect(LX - 300, READ_Y + 8, 108, 36, fill="#FFFFFF", stroke=MUT, rx=5)
c.txt(LX - 246, READ_Y + 31, "Donor", size=12, fill=INK, anchor="middle", weight="700")
c.arrow(LX - 192, READ_Y + 26, LX - 156, READ_Y + 26, stroke=SEC, w=1.8, head=5)

# ------------------------------------------------------------------ RIGHT: 10.63 branch
y = SPLIT + 22
node(RX, y, 300, 52, "~10.63 carrier", "bins 94-100    circularity 0.992", ORANGE)
y += 52
flow(RX, y, y + 30, ORANGE)
y += 30
node(RX, y, 300, 52, f"{nA} Level-A heads",
     "L2H12  L3H3  L3H5  L3H8  L3H9  L3H11  L3H13  L4H13", ORANGE)
y += 52
flow(RX, y, y + 30, ORANGE)
y += 30
node(RX, y, 300, 52, "Survives the softmax", "carrier-dependent, content-independent", ORANGE)
y += 52
flow(RX, y, y + 30, SLATE, dash="5 4")
y += 30
node(RX, y, 300, 52, "Function unknown", "no downstream effect established", SLATE,
     fill="#FFFFFF", dash="5 4")
UNK_Y = y

# the branch STOPS.  An explicit terminal bar, so the eye cannot continue it.
c.line(RX - 46, UNK_Y + 68, RX + 46, UNK_Y + 68, stroke=SLATE, w=2.5, cap="round")
c.txt(RX, UNK_Y + 90, "No causal path to splicing, chromatin", size=10.5, fill=MUT,
      anchor="middle")
c.txt(RX, UNK_Y + 105, "or translation has been demonstrated.", size=10.5, fill=MUT,
      anchor="middle")

# ------------------------------------------------------------------ the numbers
def card(x, y_, w_, lines, colour):
    h_ = 26 + 19 * len(lines)
    c.rect(x, y_, w_, h_, fill=PANEL, stroke=RULE, rx=5)
    c.line(x, y_, x, y_ + h_, stroke=colour, w=3)
    for i, (k, v) in enumerate(lines):
        c.txt(x + 14, y_ + 24 + 19 * i, k, size=10.5, fill=SEC)
        c.txt(x + w_ - 14, y_ + 24 + 19 * i, v, size=10.5, fill=INK, anchor="end",
              weight="700", cls="mono")
    return h_


CY, CW = 660, 320
c.txt(LX - CW / 2, CY - 12, "The clock does a job", size=11.5, fill=BLUE, weight="700")
card(LX - CW / 2, CY, CW, [
    ("Coupling, carrier intact", f"{d1['coupling']:.3f}"),
    ("Coupling, carrier deleted", f"{d0['coupling']:.3f}"),
    ("Restoring L2H15 recovers", f"{med['named']['L2H15  (the ADDRESSER)']['rescue']*100:.1f}%"),
    ("Exon damage", f"{exon_dmg:.3f} nats"),
    ("Intron damage", f"{intron_dmg:.3f} nats"),
], BLUE)

c.txt(RX - CW / 2, CY - 12, "The carrier is read, and that is all we know", size=11.5,
      fill=ORANGE, weight="700")
card(RX - CW / 2, CY, CW, [
    ("Post-softmax readers", f"{nA}"),
    ("Kept after carrier deleted", kept_range("kept_carrier")),
    ("Kept after reciprocal band", kept_range("kept_reciprocal")),
    ("Kept after content shuffle", kept_range("kept_shuffled")),
    ("Downstream function", "unknown"),
], ORANGE)

c.line(58, H - 46, W - 58, H - 46, stroke=RULE, w=1)
c.txt(58, H - 26, "Period-3 circuit measured in SpliceBERT.1024nt.  Level-A heads measured in "
                  "SpliceBERT.510nt (multi-species), against its human-only twin.", size=10,
      fill=MUT)
c.render(OUT)
print(f"saved -> {OUT}")
