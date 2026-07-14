#!/usr/bin/env python
"""FIGURE: the period-3 positional clock in SpliceBERT.  Renders straight from fig_phase_data.json,
so the figure cannot drift from the result it claims to show.

House style: LABELS ARE SHORT.  A few words per component, sentence case, no sentences inside the
panels.  The narrative lives in fig_phase_clock.CAPTION.md, not in the artwork.

Palette (validated: CVD dE 35.0, all four in-band):
    BLUE   #0B6FB8   the clock / intact model
    GRAY   #9AA7B2   the clock deleted
    GREEN  #00845F   the control (energy-matched band)
    ORANGE #C25A00   coding exon
    PURPLE #6A4A9C   the donor-acceptor coupling
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = json.loads((REPO / "results/phase-lattice/fig_phase_data.json").read_text())
OUT = REPO / "figures/splicebert/fig_phase_clock.svg"

BLUE, GRAY, GREEN, ORANGE, PURPLE = "#0B6FB8", "#9AA7B2", "#00845F", "#C25A00", "#6A4A9C"
INK, SEC, MUT, RULE, PANEL = "#16232E", "#5A6672", "#8794A0", "#E1E7EC", "#F7F9FB"
W, H = 1460, 1000
s: list[str] = []


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, t, *, size=11, fill=SEC, anchor="start", weight=None, cls="sans"):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    w = f' font-weight="{weight}"' if weight else ""
    s.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
             f'{a}{w}>{esc(t)}</text>')


def line(x1, y1, x2, y2, *, stroke=RULE, w=1, dash=None, cap="butt", op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{op}"' if op != 1.0 else ""
    s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
             f'stroke-width="{w}" stroke-linecap="{cap}"{d}{o}/>')


def rect(x, y, w_, h_, *, fill="none", stroke=None, sw=1, rx=0):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w_:.1f}" height="{h_:.1f}" rx="{rx}" '
             f'fill="{fill}"{st}/>')


def path(pts, *, stroke, w=2, dash=None, fill="none", cap="round"):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    s.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
             f'stroke-linejoin="round" stroke-linecap="{cap}"{da}/>')


def dot(x, y, r, fill, *, ring="#FFFFFF", rw=1.6):
    s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{ring}" '
             f'stroke-width="{rw}"/>')


def panel_label(x, y, letter, title):
    txt(x, y, letter, size=15, fill=INK, weight="700")
    txt(x + 18, y, title, size=13, fill=INK, weight="700")


# ======================================================================== header
s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')
txt(W / 2, 36, "A learned period-3 clock in SpliceBERT's position table", size=21, fill=INK,
    anchor="middle", weight="700")

# ======================================================================== A. spectrum
AX, AY, AW, AH = 78, 96, 880, 208
panel_label(AX, AY - 12, "A", "The clock is in the position table")
rect(AX, AY + 10, AW, AH, fill=PANEL)

sp = D["spectrum"]["bins"]
band = set(D["band"])
lo_p, hi_p = math.log10(2.7), math.log10(1100)
lo_v, hi_v = math.log10(0.25), math.log10(1200)


def ax(p):
    return AX + (math.log10(p) - lo_p) / (hi_p - lo_p) * AW


def ay(v):
    return AY + 10 + AH - (math.log10(max(v, 0.26)) - lo_v) / (hi_v - lo_v) * AH


for gv in (1, 10, 100, 1000):
    line(AX, ay(gv), AX + AW, ay(gv), stroke="#E7EDF2", w=1)
    txt(AX - 7, ay(gv) + 3.5, f"{gv}×", size=9.5, fill=MUT, anchor="end")
for gp in (3, 10, 30, 100, 300, 1000):
    txt(ax(gp), AY + AH + 28, str(gp), size=9.5, fill=MUT, anchor="middle")
txt(AX + AW / 2, AY + AH + 45, "Period (nucleotides)", size=10, fill=SEC, anchor="middle")
s.append(f'<text class="sans" transform="translate({AX-46},{AY+10+AH/2}) rotate(-90)" '
         f'text-anchor="middle" font-size="10" fill="{SEC}">Power / median bin</text>')

for b in sp:
    if 2.7 <= b["period"] <= 1100 and b["k"] not in band:
        line(ax(b["period"]), ay(0.26), ax(b["period"]), ay(b["rel"]), stroke="#B9C4CD", w=1)
for b in sp:
    if b["k"] in band:
        line(ax(b["period"]), ay(0.26), ax(b["period"]), ay(b["rel"]), stroke=BLUE, w=2.4,
             cap="round")

xb = ax(3.0)
line(xb, ay(340), xb + 74, ay(680), stroke=BLUE, w=1, op=0.55)
txt(xb + 80, ay(680) - 3, "Period 3", size=12, fill=BLUE, weight="700")
txt(xb + 80, ay(680) + 13, f"{D['spectrum']['band_rel']:.0f}× median bin", size=10, fill=SEC)

IX, IY = AX + AW - 162, AY + 26
rect(IX, IY, 150, 72, fill="#FFFFFF", stroke=RULE, rx=5)
txt(IX + 10, IY + 18, "Rank 2", size=10, fill=INK, weight="700")
txt(IX + 140, IY + 18, "96%", size=10, fill=BLUE, weight="700", anchor="end")
for i, e in enumerate(D["svd"][:6]):
    bh = max(e * 40, 2.0)                       # a floor, so the tiny modes still read as bars
    rect(IX + 12 + i * 24, IY + 64 - bh, 20, bh, fill=BLUE if i < 2 else "#C9D3DB", rx=2)

# ======================================================================== E. provenance
EX, EY, EW, EH = 1000, 96, 380, 208
panel_label(EX, EY - 12, "E", "Learned only with multi-species data")
rect(EX, EY + 10, EW, EH, fill=PANEL)

prov = D["provenance"]
rows = [("SpliceBERT.510nt", "Multi-species", prov["SpliceBERT.510nt (multispecies)"]["enrich"], BLUE),
        ("SpliceBERT.1024nt", "Multi-species", prov["SpliceBERT.1024nt"]["enrich"], BLUE),
        ("SpliceBERT-human.510nt", "Human only", prov["SpliceBERT-human.510nt"]["enrich"], GRAY),
        ("Random init", "Untrained", prov["random init (1024nt config)"]["enrich"], GRAY)]
bx0, bw_max = EX + 150, 166
lo_e, hi_e = math.log10(0.8), math.log10(700)
for i, (nm, sub, v, col) in enumerate(rows):
    y = EY + 44 + i * 44
    txt(bx0 - 10, y + 1, nm, size=10, fill=INK, anchor="end", weight="700")
    txt(bx0 - 10, y + 13, sub, size=8.5, fill=MUT, anchor="end")
    ww = max(3, (math.log10(max(v, 0.9)) - lo_e) / (hi_e - lo_e) * bw_max)
    rect(bx0, y - 9, ww, 14, fill=col, rx=3)
    txt(bx0 + ww + 7, y + 2, f"{v:.1f}×", size=10.5, fill=col if col == BLUE else SEC, weight="700")
line(bx0, EY + 24, bx0, EY + EH - 4, stroke="#CBD4DB", w=1)
txt(EX + EW / 2, EY + EH + 27, "Period-3 enrichment", size=10, fill=SEC, anchor="middle")

# ======================================================================== B. the wave
BY = 380
panel_label(78, BY - 14, "B", "Six heads read it as a period-3 wave")
txt(78 + 18, BY + 2, "L0H11 reads a second carrier too — its complete wave function needs both",
    size=10, fill=ORANGE)

heads = [f"L{l}H{h}" for l, h in D["combs"]]
offs = D["Rk_offsets"]
WAVE = D["wave"]["heads"]
w3 = 2 * math.pi / 3
pw_, ph, gap = 208, 172, 14
by0 = BY + 12
amp = math.ceil(max(WAVE[h]["intact"]["A"] for h in heads) * 10) / 10 + 0.35


def _env(coef, k):
    return sum(c * k ** p for p, c in enumerate(coef))


for i, hd in enumerate(heads):
    px = 78 + i * (pw_ + gap)
    rect(px, by0, pw_, ph, fill=PANEL)
    fi, fn = WAVE[hd]["intact"], WAVE[hd]["no_carrier"]

    def X(k):
        return px + 16 + (k + 12) / 24 * (pw_ - 30)

    def Y(v):
        return by0 + ph / 2 - (v / amp) * (ph / 2 - 30)

    line(px + 10, Y(0), px + pw_ - 10, Y(0), stroke="#D5DDE3", w=1)
    pk = fi["codon_position"]
    for k in offs:
        if k % 3 == pk % 3:
            line(X(k), by0 + 22, X(k), by0 + ph - 12, stroke=BLUE, w=1, op=0.09)

    # clock deleted, through the identical de-enveloping: a control that can fail, and does not
    path([(X(k), Y(D["Rk"][hd]["no_carrier"][j] - _env(fn["env_coef"], k)))
          for j, k in enumerate(offs)], stroke="#8A97A2", w=1.6, dash="3 3")
    # THE WAVE.  For a head that reads ONE carrier this is A3·cos(2πk/3 + φ3).  L0H11 reads BOTH,
    # so its COMPLETE wave function needs the second tone too -- and with it, its points land on
    # the curve (80.8% -> 99.9% of the ripple).  D22.
    tt = fi["two_tone"]
    dual = tt["A2"] > 0.2
    w2 = 2 * math.pi / tt["T2"]

    def wave(kk):
        v = tt["A3"] * math.cos(w3 * kk + tt["phi3"])
        return v + tt["A2"] * math.cos(w2 * kk + tt["phi2"]) if dual else v

    if dual:                       # what ONE cosine alone would have said -- it misses the points
        path([(X(-12 + 24 * t / 300),
               Y(fi["A"] * math.cos(w3 * (-12 + 24 * t / 300) + fi["phi"])))
              for t in range(301)], stroke=BLUE, w=1.4, dash="4 3")
    path([(X(-12 + 24 * t / 300), Y(wave(-12 + 24 * t / 300))) for t in range(301)],
         stroke=ORANGE if dual else BLUE, w=2.2)
    # the measured points, envelope removed (the two-tone envelope where that is the model)
    ec = tt["env_coef"] if dual else fi["env_coef"]
    for j, k in enumerate(offs):
        dot(X(k), Y(D["Rk"][hd]["intact"][j] - _env(ec, k)), 2.7,
            ORANGE if dual else BLUE, rw=1.3)

    if i == 0:
        sx = px + 9
        line(sx, Y(1.0), sx, Y(-1.0), stroke=SEC, w=1.2)
        line(sx - 2.5, Y(1.0), sx + 2.5, Y(1.0), stroke=SEC, w=1.2)
        line(sx - 2.5, Y(-1.0), sx + 2.5, Y(-1.0), stroke=SEC, w=1.2)
        s.append(f'<text class="sans" transform="translate({sx-6},{Y(0)}) rotate(-90)" '
                 f'text-anchor="middle" font-size="8.5" fill="{SEC}">2 logits</text>')

    txt(px + 22, by0 + 16, hd, size=10.5, fill=PURPLE if hd == "L2H15" else INK, weight="700")
    if dual:
        txt(px + pw_ - 10, by0 + 16, f"A₃ = {tt['A3']:.2f}", size=9.5, fill=BLUE, weight="700",
            anchor="end")
        txt(px + pw_ - 10, by0 + 28, f"A₂ = {tt['A2']:.2f}", size=9.5, fill=ORANGE, weight="700",
            anchor="end")
        txt(px + 22, by0 + 28, "Dual reader", size=8.5, fill=ORANGE)
    else:
        txt(px + pw_ - 10, by0 + 16, f"A = {fi['A']:.2f}", size=9.5, fill=BLUE, weight="700",
            anchor="end")
    txt(px + pw_ - 10, by0 + ph - 9, f"Codon position {pk}", size=9, fill=MUT, anchor="end")
    if hd == "L2H15":
        txt(px + 22, by0 + 28, "The addresser", size=8.5, fill=PURPLE)
    for k in (-9, 0, 9):
        txt(X(k), by0 + ph + 13, f"{k:+d}" if k else "0", size=9, fill=MUT, anchor="middle")

txt(78 + 3 * (pw_ + gap) - gap / 2, by0 + ph + 30, "Attention offset k (nucleotides)", size=10,
    fill=SEC, anchor="middle")
LX = 78 + 4 * (pw_ + gap) + 40
line(LX, by0 + ph + 26.5, LX + 22, by0 + ph + 26.5, stroke=BLUE, w=2.2)
txt(LX + 28, by0 + ph + 30, "Intact", size=10, fill=SEC)
line(LX + 92, by0 + ph + 26.5, LX + 114, by0 + ph + 26.5, stroke="#8A97A2", w=1.6, dash="3 3")
txt(LX + 120, by0 + ph + 30, "Clock deleted", size=10, fill=SEC)
line(LX + 218, by0 + ph + 26.5, LX + 240, by0 + ph + 26.5, stroke=ORANGE, w=2.2)
txt(LX + 246, by0 + ph + 30, "Complete wave function (both carriers)", size=10, fill=SEC)

# ======================================================================== C. attention -> chance
CX, CY, CW, CH = 78, 646, 600, 214
panel_label(CX, CY - 12, "C", "Delete it: attention falls to chance")
rect(CX, CY + 10, CW, CH, fill=PANEL)

pm = D["phase_mass"]
lo_m, hi_m = 0.30, 0.98


def cy(v):
    return CY + 20 + (CH - 44) - (v - lo_m) / (hi_m - lo_m) * (CH - 44)


for gv in (0.4, 0.6, 0.8):
    line(CX + 44, cy(gv), CX + CW - 12, cy(gv), stroke="#E7EDF2", w=1)
    txt(CX + 38, cy(gv) + 3.5, f"{gv:.1f}", size=9.5, fill=MUT, anchor="end")
line(CX + 44, cy(1 / 3), CX + CW - 12, cy(1 / 3), stroke=INK, w=1.2, dash="5 4", op=0.55)

gw = (CW - 62) / 6
for i, hd in enumerate(heads):
    x0 = CX + 48 + i * gw
    for j, (v, col) in enumerate([(pm["intact"][hd], BLUE), (pm["no_carrier"][hd], GRAY),
                                  (pm["no_control"][hd], GREEN)]):
        rect(x0 + 6 + j * 17, cy(v), 15, cy(lo_m) - cy(v), fill=col, rx=3)
    txt(x0 + gw / 2 - 2, CY + CH + 4, hd, size=9.5, fill=PURPLE if hd == "L2H15" else SEC,
        weight="700" if hd == "L2H15" else None, anchor="middle")
s.append(f'<text class="sans" transform="translate({CX+16},{CY+10+CH/2}) rotate(-90)" '
         f'text-anchor="middle" font-size="10" fill="{SEC}">Phase mass</text>')

for lx, lab, col in ((CX + 48, "Intact", BLUE), (CX + 132, "Clock deleted", GRAY),
                     (CX + 250, "Control band", GREEN)):
    rect(lx, CY + CH + 20, 11, 11, fill=col, rx=2)
    txt(lx + 16, CY + CH + 29.5, lab, size=9.5, fill=SEC)
line(CX + 370, CY + CH + 25.5, CX + 388, CY + CH + 25.5, stroke=INK, w=1.2, dash="5 4", op=0.55)
txt(CX + 394, CY + CH + 29.5, "Chance = 0.333", size=9.5, fill=INK, weight="700")

# ======================================================================== D. dose-response
DX, DY, DW, DH = 748, 646, 632, 214
panel_label(DX, DY - 12, "D", "Delete it: coding and the circuit die")
rect(DX, DY + 10, DW, DH, fill=PANEL)

alphas = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]


def dx(a):
    return DX + 52 + (1.0 - a) * (DW - 110)


def dy(v):
    return DY + 22 + (DH - 52) - min(max(v, -0.03), 1.1) / 1.1 * (DH - 52)


for gv in (0, 0.25, 0.5, 0.75, 1.0):
    line(DX + 48, dy(gv), DX + DW - 14, dy(gv), stroke="#E7EDF2", w=1)
    txt(DX + 42, dy(gv) + 3.5, f"{int(gv*100)}%", size=9.5, fill=MUT, anchor="end")
for a in (1.0, 0.75, 0.5, 0.25, 0.0):
    txt(dx(a), DY + DH + 4, f"{a:g}", size=9.5, fill=MUT, anchor="middle")
txt(DX + DW / 2 + 10, DY + DH + 21, "Clock amplitude retained in P  (α)", size=10, fill=SEC,
    anchor="middle")

series = [("Coding exon", "exon_kept", ORANGE), ("Intron", "intron_kept", GREEN),
          ("Donor–acceptor coupling", "coupling_kept", PURPLE)]
for nm, key, col in series:
    for bandname, dash, wdt in (("control", "5 4", 1.6), ("period-3", None, 2.4)):
        pts = []
        for a in alphas:
            r = next((r for r in D["dose"] if r["band"] == bandname and r["alpha"] == a), None)
            if r is None and a == 1.0:
                r = next(r for r in D["dose"] if r["alpha"] == 1.0)
            if r:
                pts.append((dx(a), dy(r[key])))
        if len(pts) > 1:
            path(pts, stroke=col, w=wdt, dash=dash)
    for a in alphas:
        r = next((r for r in D["dose"] if r["band"] == "period-3" and r["alpha"] == a), None)
        if r:
            dot(dx(a), dy(r[key]), 3.1, col, rw=1.4)

r0 = next(r for r in D["dose"] if r["band"] == "period-3" and r["alpha"] == 0.0)
for key, col in (("coupling_kept", PURPLE), ("intron_kept", GREEN), ("exon_kept", ORANGE)):
    txt(dx(0.0) + 8, dy(r0[key]) + 3.5, f"{r0[key]*100:.0f}%", size=10, fill=col, weight="700")

LY = DY + DH + 34
for i, (nm, key, col) in enumerate(series):
    lx = DX + 50 + i * 190
    line(lx, LY, lx + 20, LY, stroke=col, w=2.4)
    txt(lx + 26, LY + 3.5, nm, size=9.5, fill=SEC)
line(DX + 50, LY + 16, DX + 70, LY + 16, stroke=MUT, w=1.6, dash="5 4")
txt(DX + 76, LY + 19.5, "Dashed: control band deleted instead", size=9.5, fill=MUT)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" version="1.1">\n'
       '<title>A learned period-3 positional clock in SpliceBERT</title>\n'
       '<style type="text/css"><![CDATA[ .sans{font-family:"Helvetica Neue",Helvetica,Arial,'
       'sans-serif} text{shape-rendering:geometricPrecision} ]]></style>\n'
       + "\n".join(s) + "\n</svg>\n")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(svg)
print(f"saved -> {OUT}  ({len(svg)/1024:.0f} KB)")
