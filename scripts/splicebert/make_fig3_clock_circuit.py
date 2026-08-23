#!/usr/bin/env python
"""FIGURE 3 -- causal use of the period-3 positional carrier.

Reads ONLY results/phase-lattice/fig_report_data.json.

PANEL D IS SCORED AGAINST SIZE-MATCHED NULLS, and that is not a detail.  The random-head
controls come in two sizes: 24 that restore ONE random head, and 24 that restore SIX.  The
one-head nulls top out at 0.042 -- so L2H15's 0.363 is 8.6x the best of 24 random single heads.
The SIX-head nulls are a different animal: their median is 0.063, but one of them reaches 0.939.
A random six-head set CAN nearly fully restore the coupling.  Pooling the two null families
would have buried that, so they are drawn separately, in full, tail and all.

"ALL 16 layer-2 heads" has NO size-matched null.  It is drawn greyed out and labelled as
unsupported rather than quietly compared against the wrong distribution.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/splicebert"))
from figstyle import (BLUE, PURPLE, GREEN, INK, SEC, MUT, RULE, PANEL, MIST, SLATE,  # noqa: E402
                      PALEBLUE, PHASE, Canvas)

D = json.loads((REPO / "results/phase-lattice/fig_report_data.json").read_text())
OUT = REPO / "figures/splicebert/fig3_clock_circuit.svg"

W, H = 1440, 940
c = Canvas(W, H)

COMBS = D["combs"]
DOM = D["dom_phase"]
OFF = np.array(D["Rk_offsets"])

c.txt(W / 2, 40, "The period-3 positional carrier supports donor-acceptor routing", size=22,
      fill=INK, anchor="middle", weight="700")
c.txt(W / 2, 64, "SpliceBERT.1024nt", size=12.5, fill=MUT, anchor="middle")

# ============================================================ A. six comb profiles
AX, AY, AW, AH = 80, 118, 872, 272
c.plabel(AX, AY - 14, "A", "Each comb head reads one carrier phase")
c.rect(AX, AY, AW, AH, fill=PANEL)

MW = AW / 6
for i, h in enumerate(COMBS):
    r = np.array(D["Rk"][h]["intact"])
    col = PHASE[DOM[h]]
    ox, oy, ow, oh = AX + i * MW + 16, AY + 52, MW - 32, 158
    lo, hi = float(r.min()), float(r.max())
    pad = (hi - lo) * .16 or 1.0
    gx = lambda k: ox + (k - OFF[0]) / (OFF[-1] - OFF[0]) * ow            # noqa: E731
    gy = lambda v: oy + oh - (v - lo + pad) / (hi - lo + 2 * pad) * oh    # noqa: E731
    # the teeth this head reads: offsets congruent to its carrier phase
    for k in OFF:
        if k % 3 == DOM[h]:
            c.line(gx(k), oy, gx(k), oy + oh, stroke=col, w=1, op=.16)
    if lo <= 0 <= hi:                     # only if zero is actually inside this head's range
        c.line(ox, gy(0), ox + ow, gy(0), stroke=RULE, w=1)
    c.path([(gx(k), gy(v)) for k, v in zip(OFF, r)], stroke=col, w=2)
    for k, v in zip(OFF, r):
        if k % 3 == DOM[h]:
            c.dot(gx(k), gy(v), 2.8, col, rw=0)
    c.txt(ox + ow / 2, AY + 30, h, size=11.5, fill=INK, anchor="middle", weight="700")
    c.txt(ox + ow / 2, AY + 44, f"phase {DOM[h]}", size=9.5, fill=col, anchor="middle",
          weight="700")
    c.txt(ox + ow / 2, oy + oh + 16, "-12", size=8.5, fill=MUT, anchor="start", cls="mono")
    c.txt(ox + ow, oy + oh + 16, "+12", size=8.5, fill=MUT, anchor="end", cls="mono")
c.txt(AX + AW / 2, AY + AH - 12, "Query-key offset  k  (nt)", size=11, fill=SEC, anchor="middle")
c.txt(AX - 12, AY + AH / 2, "R(k)", size=11, fill=SEC, anchor="middle", rot=-90)

# ---- the phase wheel
PX, PY, PW, PH = 984, 118, 376, 272
c.plabel(PX, PY - 14, "", "Heads choose their phase")
c.txt(PX, PY - 14, "", size=1)
c.rect(PX, PY, PW, PH, fill=PANEL)
wcx, wcy, WR = PX + PW / 2, PY + 112, 66
c.ring(wcx, wcy, WR, RULE, w=1)
for t in range(3):
    a = math.radians(90 - 120 * t)
    c.line(wcx, wcy, wcx + math.cos(a) * WR, wcy - math.sin(a) * WR, stroke=RULE, w=1, dash="3 3")

# Heads of the SAME carrier-phase class land on the SAME point -- that IS the result.  So label the
# CLASS once, naming both heads, instead of stacking two labels on top of each other.
klass: dict[int, list] = {}
for hp in D["head_phases"]:
    klass.setdefault(DOM[hp["head"]], []).append(hp)
for dom, hps in sorted(klass.items()):
    col = PHASE[dom]
    for hp in hps:
        ap, am = hp["phase_pred"], hp["phase_meas"]
        xp, yp = wcx + math.cos(ap) * WR, wcy - math.sin(ap) * WR
        xm, ym = wcx + math.cos(am) * (WR + 14), wcy - math.sin(am) * (WR + 14)
        c.line(xp, yp, xm, ym, stroke=col, w=1.4)
        c.dot(xp, yp, 5.5, col)
        c.ring(xm, ym, 3.6, col, w=1.6)
    ma = math.atan2(sum(math.sin(h["phase_pred"]) for h in hps),
                    sum(math.cos(h["phase_pred"]) for h in hps))
    lx, ly = wcx + math.cos(ma) * (WR + 47), wcy - math.sin(ma) * (WR + 47)
    c.txt(lx, ly, f"phase {dom}", size=9.5, fill=col, anchor="middle", weight="700")
    c.txt(lx, ly + 13, "  ".join(h["head"] for h in hps), size=9, fill=col, anchor="middle",
          cls="mono")

c.txt(PX + PW / 2, PY + 232,
      f"Predicted vs measured:  worst {D['worst_phase_err_deg']:.2f} deg", size=10.5,
      fill=INK, anchor="middle", weight="700", cls="mono")
c.txt(PX + PW / 2, PY + 249, "against 120 deg between carrier phases", size=10, fill=MUT,
      anchor="middle")
c.dot(PX + 78, PY + 262, 5, SLATE)
c.txt(PX + 90, PY + 266, "predicted from QK", size=9.5, fill=SEC)
c.ring(PX + 214, PY + 262, 3.6, SLATE, w=1.6)
c.txt(PX + 226, PY + 266, "measured", size=9.5, fill=SEC)

# ============================================================ B. deletion specificity
BX, BY, BW, BH = 80, 448, 416, 400
c.plabel(BX, BY - 14, "B", "Only the carrier's own band matters")
c.rect(BX, BY, BW, BH, fill=PANEL)

pm = D["phase_mass"]
CH = 1 / 3
gx0, gy0, gw, gh = BX + 52, BY + 46, BW - 78, 254
by_ = lambda v: gy0 + gh - (v - .25) / (1.0 - .25) * gh                   # noqa: E731
for gv in (.3, .4, .5, .6, .7, .8, .9, 1.0):
    c.line(gx0, by_(gv), gx0 + gw, by_(gv), stroke=RULE, w=1)
    c.txt(gx0 - 8, by_(gv) + 4, f"{gv:.1f}", size=9, fill=MUT, anchor="end", cls="mono")
GRP = gw / 6
for i, h in enumerate(COMBS):
    for j, (cond, col) in enumerate((("intact", BLUE), ("no_carrier", SLATE),
                                     ("no_control", PALEBLUE))):
        v = pm[cond][h]
        bx = gx0 + i * GRP + 6 + j * ((GRP - 14) / 3)
        bw_ = (GRP - 14) / 3 - 2
        c.bar(bx, by_(v), bw_, gy0 + gh - by_(v), col)
    c.txt(gx0 + i * GRP + GRP / 2, gy0 + gh + 16, h, size=8.5, fill=SEC, anchor="middle")
c.line(gx0, by_(CH), gx0 + gw, by_(CH), stroke=INK, w=1.5, dash="5 3")
c.txt(gx0 + 4, by_(CH) - 6, "chance  1/3", size=9.5, fill=INK, weight="700", cls="mono")
c.txt(BX - 6, gy0 + gh / 2, "Phase mass", size=11, fill=SEC, anchor="middle", rot=-90)
c.swatch(BX + 20, BY + 352, "Intact", BLUE)
c.swatch(BX + 96, BY + 352, "Period-3 band deleted", SLATE)
c.swatch(BX + 20, BY + 376, "Energy-matched control band deleted", PALEBLUE)

# ============================================================ C. dose-response
CX, CY, CW, CH_ = 528, 448, 400, 400
c.plabel(CX, CY - 14, "C", "Coupling collapses first, and fastest")
c.rect(CX, CY, CW, CH_, fill=PANEL)

dose3 = sorted([r for r in D["dose"] if r["band"] == "period-3"], key=lambda r: r["alpha"])
doseC = sorted([r for r in D["dose"] if r["band"] != "period-3"], key=lambda r: r["alpha"])
cx0, cy0, cw_, ch_ = CX + 52, CY + 40, CW - 76, 268
ax_ = lambda a: cx0 + a * cw_                                            # noqa: E731
ay_ = lambda v: cy0 + ch_ - max(min(v, 1.08), -.02) / 1.08 * ch_         # noqa: E731
for gv in (0, .25, .5, .75, 1.0):
    c.line(cx0, ay_(gv), cx0 + cw_, ay_(gv), stroke=RULE, w=1)
    c.txt(cx0 - 8, ay_(gv) + 4, f"{gv:.2f}", size=9, fill=MUT, anchor="end", cls="mono")
for ga in (0, .25, .5, .75, 1.0):
    c.txt(ax_(ga), cy0 + ch_ + 18, f"{ga:.2f}", size=9, fill=MUT, anchor="middle", cls="mono")
c.txt(cx0 + cw_ / 2, cy0 + ch_ + 40, "Retained carrier amplitude  a", size=11, fill=SEC,
      anchor="middle")
c.txt(CX - 6, cy0 + ch_ / 2, "Fraction of headroom kept", size=11, fill=SEC, anchor="middle",
      rot=-90)

c.path([(ax_(r["alpha"]), ay_(r["coupling_kept"])) for r in doseC], stroke=MUT, w=1.5,
       dash="4 3")
c.txt(ax_(.5), ay_(doseC[3]["coupling_kept"]) - 10, "control band", size=9.5, fill=MUT,
      anchor="middle")
for key, col, lab, dash in (("intron_kept", GREEN, "Intron", None),
                            ("exon_kept", INK, "Exon", None),
                            ("coupling_kept", PURPLE, "Donor-acceptor coupling", None)):
    pts = [(ax_(r["alpha"]), ay_(r[key])) for r in dose3]
    c.path(pts, stroke=col, w=2.6 if "coupling" in key else 2, dash=dash)
    for x, y in pts:
        c.dot(x, y, 3.4, col)
    yv = dose3[0][key]
    c.txt(cx0 - 6, ay_(yv) + 4, "", size=1)
c.txt(ax_(1.0) - 4, ay_(dose3[-1]["intron_kept"]) - 12, "Intron", size=10.5, fill=GREEN,
      anchor="end", weight="700")
c.txt(ax_(1.0) - 4, ay_(dose3[-1]["exon_kept"]) + 18, "Exon", size=10.5, fill=INK,
      anchor="end", weight="700")
c.txt(ax_(.62), ay_(dose3[3]["coupling_kept"]) - 14, "Donor-acceptor coupling", size=10.5,
      fill=PURPLE, anchor="middle", weight="700")
c.txt(cx0 + 4, ay_(dose3[0]['coupling_kept']) - 9, f"{dose3[0]['coupling_kept']*100:.0f}%",
      size=10, fill=PURPLE, weight="700", cls="mono")
c.txt(cx0 + 4, ay_(dose3[0]["intron_kept"]) - 8, f"{dose3[0]['intron_kept']*100:.0f}%",
      size=10, fill=GREEN, weight="700", cls="mono")
c.txt(cx0 + 4, ay_(dose3[0]["exon_kept"]) - 8, f"{dose3[0]['exon_kept']*100:.0f}%", size=10,
      fill=INK, weight="700", cls="mono")
c.txt(CX + CW / 2, CY + 366, "Carrier fully deleted at the left edge", size=10, fill=MUT,
      anchor="middle")

# ============================================================ D. mediation
DX, DY, DW, DH = 960, 448, 400, 400
c.plabel(DX, DY - 14, "D", "Comparative mediation, against matched nulls")
c.rect(DX, DY, DW, DH, fill=PANEL)

med = D["mediation"]
dx0, dw_ = DX + 22, DW - 130
mx = lambda v: dx0 + min(max(v, 0), 1.0) * dw_                           # noqa: E731

for gv in (0, .25, .5, .75, 1.0):
    c.line(mx(gv), DY + 30, mx(gv), DY + 330, stroke=RULE, w=1)
    c.txt(mx(gv), DY + 348, f"{gv*100:.0f}%", size=9, fill=MUT, anchor="middle", cls="mono")
c.txt(DX + DW / 2 - 30, DY + 370, "Lost coupling recovered", size=11, fill=SEC, anchor="middle")


def bar_row(y, key, lab):
    v = med["named"][key]
    unsupported = v["p_add_one"] is None
    col = MIST if unsupported else PURPLE
    c.txt(dx0, y, lab, size=10, fill=MUT if unsupported else SEC)
    c.bar(dx0, y + 6, mx(v["rescue"]) - dx0, 13, col)
    c.line(mx(v["lo"]), y + 12.5, mx(v["hi"]), y + 12.5, stroke=INK, w=1.2, op=.5)
    c.txt(min(mx(v["hi"]) + 9, dx0 + dw_ + 8), y + 17, f"{v['rescue']*100:.1f}%", size=10,
          fill=MUT if unsupported else INK, weight="700", cls="mono")
    c.txt(DX + DW - 14, y + 17,
          "no matched null" if unsupported else f"p = {v['p_add_one']:.3f}",
          size=9, fill=MUT if unsupported else SEC, anchor="end", cls="mono")
    return y + 38


def null_strip(y, vals, label):
    c.rect(dx0 - 4, y, dw_ + 8, 24, fill="#FFFFFF", rx=4)
    c.line(dx0, y + 12, dx0 + dw_, y + 12, stroke=RULE, w=1)
    for v in vals:
        c.dot(mx(v), y + 12, 3.4, MIST, ring=SLATE, rw=1)
    c.txt(dx0, y - 3, label, size=9.5, fill=SLATE, weight="700")
    c.txt(DX + DW - 14, y + 16, f"max {max(vals)*100:.1f}%", size=9, fill=SLATE,
          anchor="end", cls="mono")
    return y + 46


yy = DY + 42
yy = bar_row(yy, "L2H15  (the ADDRESSER)", "L2H15   the addresser")
yy = bar_row(yy, "L3H1   (the READER)", "L3H1    the reader")
yy = bar_row(yy, "L1H12  (a comb, not in C)", "L1H12   a comb, not in the circuit")
yy = null_strip(yy, med["nulls_by_size"]["1"], "24 random single heads")
yy = bar_row(yy, "all 6 COMBS", "All six combs")
yy = null_strip(yy, med["nulls_by_size"]["6"], "24 random six-head sets")
yy = bar_row(yy, "ALL 16 layer-2 heads", "All 16 layer-2 heads")

c.line(80, H - 44, W - 80, H - 44, stroke=RULE, w=1)
c.txt(80, H - 24, "Panel C is the dose run (n = 256): coupling 0.946 -> 0.026.  Panel D is the "
                  f"mediation run (n = 160), with its own baseline {med['intact']:.3f} -> "
                  f"{med['killed']:.3f}; percentages are of THAT lost coupling.  Each restoration "
                  "is scored only against random head sets OF ITS OWN SIZE (exact add-one p, 24 "
                  "matched nulls).", size=10, fill=MUT)
c.render(OUT)
print(f"saved -> {OUT}")
