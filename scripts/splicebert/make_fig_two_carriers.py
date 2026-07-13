#!/usr/bin/env python
"""FIGURE: SpliceBERT's position table carries TWO clocks, and they do different jobs.

Renders from results/phase-lattice/{second_carrier,fig_phase_data}.json.

House style: LABELS ARE SHORT.  Narrative lives in fig_two_carriers.CAPTION.md.

Palette:
    BLUE   #0B6FB8   the codon clock       (period 3,     bins 339-345)
    ORANGE #C25A00   the second carrier    (period 10.69, bins 94-100)
    GRAY   #9AA7B2   deleted
    PURPLE #6A4A9C   the donor-acceptor coupling
    GREEN  #00845F   intron
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
S = json.loads((REPO / "results/phase-lattice/second_carrier.json").read_text())
D = json.loads((REPO / "results/phase-lattice/fig_phase_data.json").read_text())
OUT = REPO / "figures/splicebert/fig_two_carriers.svg"

BLUE, ORANGE, GRAY, PURPLE, GREEN = "#0B6FB8", "#C25A00", "#9AA7B2", "#6A4A9C", "#00845F"
INK, SEC, MUT, RULE, PANEL = "#16232E", "#5A6672", "#8794A0", "#E1E7EC", "#F7F9FB"
W, H = 1440, 940
s: list[str] = []

K = np.array(S["offsets"], float)
BY = {r["name"]: r for r in S["heads"]}
ED = S["envdeg"]
W3 = 2 * math.pi / 3
T10 = S["period_p10_exact"]
FLOOR10 = float(np.percentile([r["no_p10"]["A10"] for r in S["heads"]], 95))


def txt(x, y, t, *, size=11, fill=SEC, anchor="start", weight=None, rot=None):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    w = f' font-weight="{weight}"' if weight else ""
    if rot is not None:
        s.append(f'<text class="sans" transform="translate({x:.1f},{y:.1f}) rotate({rot})" '
                 f'text-anchor="{anchor}" font-size="{size}" fill="{fill}"{w}>{t}</text>')
    else:
        s.append(f'<text class="sans" x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
                 f'{a}{w}>{t}</text>')


def line(x1, y1, x2, y2, *, stroke=RULE, w=1, dash=None, cap="butt", op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{op}"' if op != 1.0 else ""
    s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
             f'stroke-width="{w}" stroke-linecap="{cap}"{d}{o}/>')


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


def dot(x, y, r, fill, *, rw=1.3):
    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#FFFFFF" '
             f'stroke-width="{rw}"/>')


def plabel(x, y, ltr, title):
    txt(x, y, ltr, size=15, fill=INK, weight="700")
    txt(x + 18, y, title, size=13, fill=INK, weight="700")


def fit(y, T2=None, deg=ED):
    cols = [K**p for p in range(deg + 1)] + [np.cos(W3 * K), np.sin(W3 * K)]
    if T2:
        w2 = 2 * math.pi / T2
        cols += [np.cos(w2 * K), np.sin(w2 * K)]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    den = (y - np.polyval(b[:deg + 1][::-1], K)).var()
    return b, X @ b, float(1 - res.var() / den), np.polyval(b[:deg + 1][::-1], K)


s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')
txt(W / 2, 36, "SpliceBERT's position table carries two clocks", size=21, fill=INK,
    anchor="middle", weight="700")

# ======================================================================= A. the spectrum
AX, AY, AW, AH = 74, 96, 810, 230
plabel(AX, AY - 12, "A", "Two high-frequency carriers, and nothing else")
rect(AX, AY + 10, AW, AH, fill=PANEL)

sp = D["spectrum"]["bins"]
b3, b10 = set(D["band"]), set(D["ctrl_band"])
lo_p, hi_p = math.log10(2.7), math.log10(1100)
lo_v, hi_v = math.log10(0.25), math.log10(1200)
ax = lambda p: AX + (math.log10(p) - lo_p) / (hi_p - lo_p) * AW          # noqa: E731
ay = lambda v: AY + 10 + AH - (math.log10(max(v, .26)) - lo_v) / (hi_v - lo_v) * AH  # noqa: E731

for gv in (1, 10, 100, 1000):
    line(AX, ay(gv), AX + AW, ay(gv), stroke="#E7EDF2")
    txt(AX - 7, ay(gv) + 3.5, f"{gv}×", size=9.5, fill=MUT, anchor="end")
for gp in (3, 10, 30, 100, 300, 1000):
    txt(ax(gp), AY + AH + 28, str(gp), size=9.5, fill=MUT, anchor="middle")
txt(AX + AW / 2, AY + AH + 45, "Period (nucleotides)", size=10, fill=SEC, anchor="middle")
txt(AX - 46, AY + 10 + AH / 2, "Power / median bin", size=10, fill=SEC, anchor="middle", rot=-90)

for b in sp:
    if 2.7 <= b["period"] <= 1100 and b["k"] not in b3 and b["k"] not in b10:
        line(ax(b["period"]), ay(.26), ax(b["period"]), ay(b["rel"]), stroke="#B9C4CD")
for b in sp:
    if b["k"] in b3:
        line(ax(b["period"]), ay(.26), ax(b["period"]), ay(b["rel"]), stroke=BLUE, w=2.4,
             cap="round")
    elif b["k"] in b10:
        line(ax(b["period"]), ay(.26), ax(b["period"]), ay(b["rel"]), stroke=ORANGE, w=2.4,
             cap="round")

line(ax(3), ay(330), ax(3) + 60, ay(760), stroke=BLUE, w=1, op=.55)
txt(ax(3) + 66, ay(760) - 2, "Period 3", size=12, fill=BLUE, weight="700")
txt(ax(3) + 66, ay(760) + 13, "The codon clock", size=9.5, fill=SEC)
line(ax(10.69), ay(590), ax(10.69) + 54, ay(1000), stroke=ORANGE, w=1, op=.55)
txt(ax(10.69) + 60, ay(1000) - 2, "Period 10.69", size=12, fill=ORANGE, weight="700")
txt(ax(10.69) + 60, ay(1000) + 13, "The second carrier", size=9.5, fill=SEC)
txt(ax(150), ay(1.4), "Everything else strong is slow drift (period ≥ 22)", size=10, fill=MUT,
    anchor="middle")

# ======================================================================= B. L0H11 needs two
# Shown in the +-12 window: that is the window where the raggedness was SEEN (it is the window
# every other figure in this study plots), and this panel is the completion of that observation.
# The wide window is what panel C uses, because that is what resolves the FREQUENCY.  Over +-60
# two tones explain 89.5% (from 70.2%) -- there is further structure at long offsets.  Caption.
BX, BY_, BW, BH = 74, 410, 810, 230
plabel(BX, BY_ - 12, "B", "L0H11 needed a second cosine")
rect(BX, BY_ + 10, BW, BH, fill=PANEL)

K12 = np.array(D["Rk_offsets"], float)
y11 = np.array(D["Rk"]["L0H11"]["intact"], float)
W10 = 2 * math.pi / T10


def fit12(y, T2=None, deg=3):
    cols = [K12**p for p in range(deg + 1)] + [np.cos(W3 * K12), np.sin(W3 * K12)]
    if T2:
        cols += [np.cos(W10 * K12), np.sin(W10 * K12)]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    den = (y - np.polyval(b[:deg + 1][::-1], K12)).var()
    fine = np.linspace(-12, 12, 400)
    cf = [fine**p for p in range(deg + 1)] + [np.cos(W3 * fine), np.sin(W3 * fine)]
    if T2:
        cf += [np.cos(W10 * fine), np.sin(W10 * fine)]
    return fine, np.column_stack(cf) @ b, float(1 - res.var() / den)


fine, f1, v1 = fit12(y11, None)
_, f2, v2 = fit12(y11, T10)
lo_y, hi_y = float(y11.min()) - .6, float(y11.max()) + .9
bx = lambda k: BX + 54 + (k + 12) / 24 * (BW - 80)                        # noqa: E731
byy = lambda v: BY_ + 30 + (BH - 68) - (v - lo_y) / (hi_y - lo_y) * (BH - 68)  # noqa: E731

for gv in range(int(lo_y) + 1, int(hi_y) + 1):
    line(BX + 50, byy(gv), BX + BW - 16, byy(gv), stroke="#E7EDF2")
    txt(BX + 44, byy(gv) + 3.5, str(gv), size=9.5, fill=MUT, anchor="end")
for gk in (-12, -6, 0, 6, 12):
    txt(bx(gk), BY_ + BH + 24, f"{gk:+d}" if gk else "0", size=9.5, fill=MUT, anchor="middle")
txt(BX + BW / 2, BY_ + BH + 42, "Attention offset k (nucleotides)", size=10, fill=SEC,
    anchor="middle")
txt(BX - 46, BY_ + 10 + BH / 2, "R(k)  (attention logits)", size=10, fill=SEC, anchor="middle",
    rot=-90)

path([(bx(k), byy(v)) for k, v in zip(fine, f1)], stroke=GRAY, w=1.8, dash="5 3")
path([(bx(k), byy(v)) for k, v in zip(fine, f2)], stroke=ORANGE, w=2.4)
for k, v in zip(K12, y11):
    dot(bx(k), byy(v), 3.0, INK, rw=1.2)

lx = BX + 300
line(lx, BY_ + 32, lx + 20, BY_ + 32, stroke=GRAY, w=1.8, dash="5 3")
txt(lx + 26, BY_ + 35.5, f"One cosine (period 3) — {v1*100:.1f}% of the ripple", size=10, fill=SEC)
line(lx, BY_ + 50, lx + 20, BY_ + 50, stroke=ORANGE, w=2.4)
txt(lx + 26, BY_ + 53.5, f"Plus period 10.69 — {v2*100:.1f}%", size=10, fill=ORANGE, weight="700")
dot(lx + 10, BY_ + 68, 3.0, INK, rw=1.2)
txt(lx + 26, BY_ + 71.5, "Measured", size=10, fill=SEC)

# ======================================================================= C. the periodogram
CX, CY, CW, CH = 928, 96, 440, 230
plabel(CX, CY - 12, "C", "The data names the period")
rect(CX, CY + 10, CW, CH, fill=PANEL)

per = np.arange(6.0, 20.01, 0.05)


def pgram(y):
    out = []
    for T in per:
        w2 = 2 * math.pi / T
        X = np.column_stack([K**p for p in range(ED + 1)] +
                            [np.cos(W3 * K), np.sin(W3 * K), np.cos(w2 * K), np.sin(w2 * K)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.append(math.hypot(b[ED + 3], b[ED + 4]))
    return np.array(out)


# the periodogram uses the WIDE window (+-60) -- that is what resolves the frequency.
# (y11 above is the +-12 window; do not reuse it here.)
pg11 = pgram(np.array(BY["L0H11"]["Rk_intact"]))
pg15 = pgram(np.array(BY["L2H15"]["Rk_intact"]))
hi_a = math.ceil(pg11.max() * 10) / 10 + .1
cx = lambda t: CX + 48 + (t - 6) / 14 * (CW - 74)                          # noqa: E731
cy = lambda a: CY + 28 + (CH - 62) - a / hi_a * (CH - 62)                  # noqa: E731

for gv in (0, .2, .4, .6):
    line(CX + 44, cy(gv), CX + CW - 14, cy(gv), stroke="#E7EDF2")
    txt(CX + 38, cy(gv) + 3.5, f"{gv:.1f}", size=9.5, fill=MUT, anchor="end")
for gt in (6, 8, 10, 12, 14, 16, 18, 20):
    txt(cx(gt), CY + CH + 22, str(gt), size=9.5, fill=MUT, anchor="middle")
txt(CX + CW / 2, CY + CH + 40, "Candidate period T (nucleotides)", size=10, fill=SEC,
    anchor="middle")
txt(CX + 6, CY + 10 + CH / 2, "Fitted amplitude", size=10, fill=SEC, anchor="middle", rot=-90)

line(cx(T10), cy(0), cx(T10), CY + 26, stroke=ORANGE, w=1.2, dash="4 3", op=.7)
txt(cx(T10) + 6, CY + 38, f"P's bin 96 = {T10:.2f}", size=9.5, fill=ORANGE, weight="700")
path([(cx(t), cy(a)) for t, a in zip(per, pg15)], stroke=GRAY, w=1.8, dash="4 3")
path([(cx(t), cy(a)) for t, a in zip(per, pg11)], stroke=ORANGE, w=2.4)
pk = per[int(pg11.argmax())]
dot(cx(pk), cy(pg11.max()), 3.4, ORANGE, rw=1.4)
txt(cx(pk) + 8, cy(pg11.max()) - 6, f"L0H11 peaks at {pk:.2f}", size=9.5, fill=ORANGE,
    weight="700")
txt(cx(15.5), cy(0.055), "L2H15 — flat, no second tone", size=9.5, fill=SEC)

# ======================================================================= D. double dissociation
DX, DY, DW, DH = 928, 410, 440, 230
plabel(DX, DY - 12, "D", "Each carrier deletes on its own")
rect(DX, DY + 10, DW, DH, fill=PANEL)

r11 = BY["L0H11"]
cells = [
    ("Codon clock (period 3)", BLUE, r11["intact"]["A3"],
     r11["no_p3"]["A3"], r11["no_p10"]["A3"]),
    ("Second carrier (10.69)", ORANGE, r11["intact"]["A10"],
     r11["no_p3"]["A10"], r11["no_p10"]["A10"]),
]
gw = (DW - 90) / 2
for i, (nm, col, a_i, a_n3, a_n10) in enumerate(cells):
    x0 = DX + 62 + i * gw
    hi_b = 1.35
    dy = lambda v: DY + 52 + (DH - 96) - v / hi_b * (DH - 96)              # noqa: E731
    for j, (v, c) in enumerate([(1.0, col), (a_n3 / a_i, GRAY), (a_n10 / a_i, GRAY)]):
        bxx = x0 + 8 + j * 34
        rect(bxx, dy(v), 26, max(dy(0) - dy(v), 2), fill=c, rx=3)
        txt(bxx + 13, dy(v) - 5, f"{v*100:.0f}%", size=9.5, anchor="middle",
            fill=col if j == 0 else SEC, weight="700")
    txt(x0 + gw / 2 - 6, DY + 40, nm, size=10, fill=col, weight="700", anchor="middle")
    for j, lab in enumerate(("Intact", "− clock", "− carrier")):
        txt(x0 + 21 + j * 34, DY + DH + 2, lab, size=8.5, fill=MUT, anchor="middle")
txt(DX + 8, DY + 10 + DH / 2, "Amplitude kept", size=10, fill=SEC, anchor="middle", rot=-90)
txt(DX + DW / 2, DY + DH + 24, "L0H11", size=10, fill=INK, weight="700", anchor="middle")

# ======================================================================= E. two jobs
EX, EY, EW, EH = 74, 720, 1294, 150
plabel(EX, EY - 12, "E", "But only the codon clock does splicing")
rect(EX, EY + 10, EW, EH, fill=PANEL)

d = {(r["band"], r["alpha"]): r for r in D["dose"]}
base = d[("period-3", 1.0)]
sets = [
    ("Coding-exon damage", "nats", [(base["exon"] - d[("period-3", 0.0)]["exon"], BLUE),
                                    (base["exon"] - d[("control", 0.0)]["exon"], ORANGE)], 0.20),
    ("Intron damage", "nats", [(base["intron"] - d[("period-3", 0.0)]["intron"], BLUE),
                               (base["intron"] - d[("control", 0.0)]["intron"], ORANGE)], 0.20),
    ("Donor–acceptor coupling kept", "%", [(d[("period-3", 0.0)]["coupling_kept"], BLUE),
                                           (d[("control", 0.0)]["coupling_kept"], ORANGE)], 1.15),
]
sw = EW / 3
for i, (nm, unit, bars, hi_v2) in enumerate(sets):
    x0 = EX + i * sw
    if i:
        line(x0, EY + 24, x0, EY + EH - 10, stroke="#DCE3E9")
    txt(x0 + sw / 2, EY + 34, nm, size=10.5, fill=INK, weight="700", anchor="middle")
    ey = lambda v: EY + 48 + (EH - 78) - v / hi_v2 * (EH - 78)             # noqa: E731
    line(x0 + 130, ey(0), x0 + sw - 40, ey(0), stroke="#CBD4DB")
    for j, (v, c) in enumerate(bars):
        bxx = x0 + 150 + j * 90
        rect(bxx, ey(v), 34, max(ey(0) - ey(v), 2), fill=c, rx=3)
        lab = f"{v:.3f}" if unit == "nats" else f"{v*100:.0f}%"
        txt(bxx + 17, ey(v) - 6, lab, size=10, fill=c, weight="700", anchor="middle")
        txt(bxx + 17, EY + EH - 2, ("− clock", "− carrier")[j], size=8.5, fill=MUT,
            anchor="middle")

line(74, 902, W - 74, 902, stroke=RULE)
for x, c, lab in ((90, BLUE, "Codon clock deleted (bins 339–345)"),
                  (400, ORANGE, "Second carrier deleted (bins 94–100)"),
                  (740, GRAY, "Amplitude after deletion")):
    rect(x, 916, 11, 11, fill=c, rx=2)
    txt(x + 16, 925.5, lab, size=10, fill=SEC)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" version="1.1">\n'
       "<title>SpliceBERT's position table carries two clocks</title>\n"
       '<style type="text/css"><![CDATA[ .sans{font-family:"Helvetica Neue",Helvetica,Arial,'
       'sans-serif} text{shape-rendering:geometricPrecision} ]]></style>\n'
       + "\n".join(s) + "\n</svg>\n")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(svg)
print(f"saved -> {OUT}  ({len(svg)/1024:.0f} KB)")
print(f"  L0H11: one cosine {v1*100:.1f}%  ->  two cosines {v2*100:.1f}%")
print(f"  periodogram peak {pk:.2f}  vs  P's bin 96 = {T10:.4f}")
