#!/usr/bin/env python
"""FIGURE: the donor-acceptor circuit, and the period-3 clock that addresses it.

Was a hand-authored SVG.  It had already DRIFTED from the data (it printed phase mass 0.786 where
the run says 0.794), and its clock glyph was a hand-drawn SAWTOOTH -- the artefact this whole
thread retired.  It is now generated, like the other two, straight from fig_phase_data.json, and
the clock glyph is L2H15's ACTUAL fitted wave.

House style: LABELS ARE SHORT.  A few words per component, sentence case, no sentences inside the
artwork.  The narrative lives in fig_phase_circuit.CAPTION.md.

Palette -- five IDENTITY colours, validated (all pass; worst adjacent CVD dE 28.4 protan):
    BLUE    #0B6FB8   the donor / the query Q
    GREEN   #00845F   the acceptor / the address K
    ORANGE  #C25A00   the acceptor's payload V
    PURPLE  #6A4A9C   L3H1, the read
    MAGENTA #B23A6E   the period-3 clock (the position code)
Grey is "absent/inert", never an identity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D = json.loads((REPO / "results/phase-lattice/fig_phase_data.json").read_text())
OUT = REPO / "figures/splicebert/fig_phase_circuit.svg"

BLUE, GREEN, ORANGE, PURPLE, CLOCK = "#0B6FB8", "#00845F", "#C25A00", "#6A4A9C", "#B23A6E"
BLUE_D, GREEN_D, ORANGE_D, PURPLE_D, CLOCK_D = "#084E82", "#04654A", "#8F4200", "#4E3576", "#8E2856"
INK, SEC, MUT, RULE, PANEL = "#16232E", "#5A6672", "#8794A0", "#E1E7EC", "#F7F9FB"
W, H = 1180, 800
s: list[str] = []

# ---------------------------------------------------------------- the numbers, from the data
pm = D["phase_mass"]
PM_INT = float(np.mean(list(pm["intact"].values())))
PM_KILL = float(np.mean(list(pm["no_carrier"].values())))
dose = {(r["band"], r["alpha"]): r for r in D["dose"]}
CPL_INT = dose[("period-3", 1.0)]["coupling"]
CPL_KILL = dose[("period-3", 0.0)]["coupling"]
CPL_CTRL = dose[("control", 0.0)]["coupling_kept"]
RESCUE = next(v for k, v in D["mediation"]["rescue"].items() if k.startswith("L2H15"))
BAND_REL = D["spectrum"]["band_rel"]
RANK2 = 100 * (D["svd"][0] + D["svd"][1])
NBAND, NBINS = len(D["band"]), len(D["spectrum"]["bins"])
WV = D["wave"]["heads"]["L2H15"]["intact"]


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, t, *, size=11, fill=SEC, anchor="start", weight=None, cls="sans", rot=None):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    w = f' font-weight="{weight}"' if weight else ""
    if rot is not None:
        s.append(f'<text class="{cls}" transform="translate({x:.1f},{y:.1f}) rotate({rot})" '
                 f'text-anchor="{anchor}" font-size="{size}" fill="{fill}"{w}>{esc(t)}</text>')
    else:
        s.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
                 f'{a}{w}>{esc(t)}</text>')


def rect(x, y, w_, h_, *, fill="none", stroke=None, sw=1.0, rx=0):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w_:.1f}" height="{h_:.1f}" rx="{rx}" '
             f'fill="{fill}"{st}/>')


def line(x1, y1, x2, y2, *, stroke=RULE, w=1, dash=None, cap="butt", op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' stroke-opacity="{op}"' if op != 1.0 else ""
    s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
             f'stroke-width="{w}" stroke-linecap="{cap}"{d}{o}/>')


def arrow(d, *, stroke, w=2, dash=None, marker=None, cap="butt", op=1.0):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    o = f' stroke-opacity="{op}"' if op != 1.0 else ""
    s.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{w}" '
             f'stroke-linecap="{cap}"{da}{mk}{o}/>')


def circle(cx, cy, r, *, fill, stroke, sw=1.9):
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
             f'stroke-width="{sw}"/>')


# ---------------------------------------------------------------- defs
defs = ['<defs>', '<style type="text/css"><![CDATA[',
        ".sans{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif}",
        ".mono{font-family:'SFMono-Regular',Menlo,Consolas,monospace}",
        "text{shape-rendering:geometricPrecision}", ']]></style>']
for nm, col in (("ahBlue", BLUE), ("ahGreen", GREEN), ("ahOrange", ORANGE), ("ahClock", CLOCK),
                ("ahPurple", PURPLE)):
    defs.append(f'<marker id="{nm}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
                f'markerHeight="5" orient="auto-start-reverse" markerUnits="strokeWidth">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
defs.append('<linearGradient id="gWrite" gradientUnits="userSpaceOnUse" x1="540" y1="628" '
            f'x2="410" y2="668"><stop offset="0" stop-color="{PURPLE}"/>'
            f'<stop offset="1" stop-color="{BLUE}"/></linearGradient>')
defs.append('<linearGradient id="gWriteL" gradientUnits="userSpaceOnUse" x1="880" y1="0" '
            f'x2="910" y2="0"><stop offset="0" stop-color="{PURPLE}"/>'
            f'<stop offset="1" stop-color="{BLUE}"/></linearGradient>')
defs.append("</defs>")
s.append("\n".join(defs))
s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# ---------------------------------------------------------------- title
txt(W / 2, 34, "Donor–acceptor binding runs on a learned period-3 clock", size=21, fill=INK,
    anchor="middle", weight="700")

# ---------------------------------------------------------------- stage bands
rect(300, 66, 856, 72, fill=PANEL)
rect(300, 230, 856, 212, fill=PANEL)   # must reach past the Q/K/V circles (cy 414, r 26)
rect(300, 606, 856, 86, fill=PANEL)
for y, lab in ((102, "1 Sequence"), (186, "2 Token states"), (318, "3 Projections"),
               (490, "4 The read"), (649, "5 Write-back")):
    txt(286, y, lab, size=8.5, fill="#A3B0BA", weight="700", anchor="middle", rot=-90)

# ================================================================ stage 0: THE CLOCK
rect(40, 140, 224, 266, fill="#FBF0F5", stroke=CLOCK, sw=1.9, rx=10)
txt(56, 164, "0 Position code", size=8.5, fill=CLOCK, weight="700")
txt(56, 188, "The period-3 clock", size=13, fill=CLOCK_D, weight="700")
txt(56, 206, "Position table P [1026 × 512]", size=10, fill=SEC)

# THE WAVE.  L2H15's actual fitted cosine -- not a hand-drawn sawtooth.
rect(56, 218, 192, 66, fill="#FFFFFF", stroke="#EBD3DF", sw=1, rx=5)
w3 = 2 * math.pi / 3


def gx(k):
    return 56 + 12 + (k + 6) / 12 * 168


def gy(v):
    return 218 + 33 - (v / WV["A"]) * 22


arrow("M " + " L ".join(f"{gx(-6 + 12*t/400):.1f},{gy(WV['A']*math.cos(w3*(-6 + 12*t/400) + WV['phi'])):.1f}"
                        for t in range(401)), stroke=CLOCK, w=2, cap="round")
for k in range(-6, 7):
    circle(gx(k), gy(WV["A"] * math.cos(w3 * k + WV["phi"])), 1.9, fill=CLOCK, stroke="#FFFFFF",
           sw=0.9)
txt(152, 296, "One oscillation every 3 nucleotides", size=9, fill="#B0849A", anchor="middle")

for i, b in enumerate([f"Rank 2 · a rotary phase code ({RANK2:.0f}% of its energy)",
                       f"{BAND_REL:.0f}× the median Fourier bin",
                       "Learned with multi-species data only"]):
    txt(56, 326 + i * 20, "· " + b, size=10, fill=SEC)

# clock -> every token
arrow("M 264 250 C 300 250, 320 254, 344 256", stroke=CLOCK, w=1.8, dash="2 3.5", cap="round",
      marker="ahClock")
txt(270, 241, "To every token", size=10, fill=CLOCK)
# clock -> L2H15
arrow("M 264 372 C 380 388, 470 372, 560 348", stroke=CLOCK, w=2.4, marker="ahClock")
txt(324, 332, "L2H15 reads it", size=10, fill=CLOCK, weight="700")

# ---------------------------------------------------------------- donor residual spine
arrow("M 400 290 L 400 666", stroke=BLUE, w=2.5, dash="1 7", cap="round", op=0.35)
txt(388, 600, "Donor residual stream", size=10, fill=BLUE, anchor="middle", rot=-90)

# ================================================================ stage 1: SEQUENCE
txt(320, 76, "Illustrative genomic input (5′ → 3′)", size=10, fill=SEC)
rect(320, 82, 812, 38, fill="#F4F7FA", stroke="#9AA7B2", sw=1.1, rx=6)
rect(500, 82, 396, 38, fill="#ECEFF2")
rect(372, 82, 58, 38, fill="#D6E7F5", stroke=BLUE, sw=1.5)
rect(896, 82, 58, 38, fill="#D3EDE2", stroke=GREEN, sw=1.5)
txt(345, 107, "…CAG", size=13, fill=INK, anchor="middle", cls="mono")
txt(401, 107, "GT", size=13, fill=BLUE_D, anchor="middle", weight="700", cls="mono")
txt(698, 107, "… intron …", size=12, fill="#7A868F", anchor="middle")
txt(925, 107, "AG", size=13, fill=GREEN_D, anchor="middle", weight="700", cls="mono")
txt(1050, 107, "GTA…", size=13, fill=INK, anchor="middle", cls="mono")
txt(406, 136, "Donor site", size=10.5, fill=BLUE_D, weight="700")
txt(918, 136, "Acceptor site", size=10.5, fill=GREEN_D, weight="700", anchor="end")
arrow("M 401 122 L 401 226", stroke=BLUE, w=2, marker="ahBlue")
arrow("M 925 122 C 925 158, 934 190, 940 226", stroke=GREEN, w=2, marker="ahGreen")

# ================================================================ stage 2: TOKEN STATES
rect(344, 230, 184, 52, fill="#EAF2FA", stroke=BLUE, sw=1.7, rx=8)
txt(436, 252, "Donor state", size=13, fill=BLUE_D, anchor="middle", weight="700")
txt(436, 270, "Produces the query", size=10, fill=SEC, anchor="middle")

rect(846, 230, 200, 52, fill="#E6F4EE", stroke=GREEN, sw=1.7, rx=8)
txt(946, 252, "Acceptor state", size=13, fill=GREEN_D, anchor="middle", weight="700")
txt(946, 270, "Address + payload", size=10, fill=SEC, anchor="middle")

arrow("M 400 282 L 400 386", stroke=BLUE, w=2, marker="ahBlue")
arrow("M 900 282 C 860 300, 820 318, 786 330", stroke=GREEN, w=2, dash="7 4", marker="ahGreen")
arrow("M 1024 282 C 1030 320, 1024 356, 1020 384", stroke=ORANGE, w=2, dash="1.5 4.5", cap="round",
      marker="ahOrange")

# ================================================================ stage 3: L2H15 + Q/K/V
rect(560, 322, 228, 52, fill="#E6F4EE", stroke=GREEN, sw=1.9, rx=26)
txt(674, 342, "L2H15 · The addresser", size=12.5, fill=GREEN_D, anchor="middle", weight="700")
txt(674, 360, "Builds the acceptor's address", size=9.5, fill=SEC, anchor="middle")
arrow("M 674 374 L 674 386", stroke=GREEN, w=2, dash="7 4", marker="ahGreen")

circle(400, 414, 26, fill="#EAF2FA", stroke=BLUE)
txt(400, 421, "Q", size=20, fill=BLUE_D, anchor="middle", weight="700")
txt(367, 418, "Donor query", size=10, fill=SEC, anchor="end")

circle(674, 414, 26, fill="#E6F4EE", stroke=GREEN)
txt(674, 421, "K", size=20, fill=GREEN_D, anchor="middle", weight="700")
txt(708, 410, "Acceptor address", size=10, fill=SEC)
txt(708, 424, "Phase-coded", size=10, fill=CLOCK_D, weight="700")

circle(1020, 414, 26, fill="#FBEEE2", stroke=ORANGE)
txt(1020, 421, "V", size=20, fill=ORANGE_D, anchor="middle", weight="700")
txt(1053, 418, "Acceptor payload", size=10, fill=SEC)

# ================================================================ stage 4: L3H1
arrow("M 400 440 C 400 476, 418 498, 456 508", stroke=BLUE, w=2, marker="ahBlue")
arrow("M 674 440 L 674 482", stroke=GREEN, w=2, dash="7 4", marker="ahGreen")
arrow("M 1020 440 C 1020 476, 986 498, 906 508", stroke=ORANGE, w=2, dash="1.5 4.5", cap="round",
      marker="ahOrange")

rect(466, 486, 430, 66, fill="#F1ECF8", stroke=PURPLE, sw=1.9, rx=33)
txt(681, 513, "L3H1 · The read", size=13.5, fill=PURPLE_D, anchor="middle", weight="700")
txt(681, 532, "Q·K selects the partner · attention transports V", size=10.5, fill=SEC,
    anchor="middle")

# ================================================================ stage 5: WRITE-BACK
arrow("M 540 552 C 500 586, 460 610, 424 650", stroke="url(#gWrite)", w=2.8, marker="ahPurple")
txt(552, 590, "Payload enters the donor's stream", size=10, fill=SEC)

rect(330, 654, 220, 52, fill="#EAF2FA", stroke=BLUE, sw=1.9, rx=8)
txt(440, 676, "Donor update", size=13, fill=BLUE_D, anchor="middle", weight="700")
txt(440, 694, "Partner-informed output", size=10, fill=SEC, anchor="middle")

# ================================================================ the causal test
rect(600, 606, 556, 106, fill="#FFFFFF", stroke=CLOCK, sw=1.6, rx=9)
txt(620, 630, f"Delete the clock  ({NBAND} of {NBINS} Fourier bins)", size=12, fill=CLOCK_D,
    weight="700")
for i, (lab, val, col) in enumerate([
        ("Comb attention → chance", f"{PM_INT:.2f} → {PM_KILL:.2f}", INK),
        ("Donor–acceptor coupling", f"{CPL_INT:.2f} → {CPL_KILL:.2f}", PURPLE_D),
        ("Restore L2H15 alone", f"{RESCUE*100:.0f}% recovered", GREEN_D)]):
    txt(620, 652 + i * 18, "· " + lab, size=10.5, fill="#46525C")
    txt(1000, 652 + i * 18, val, size=10.5, fill=col, weight="700", anchor="end")
txt(620, 704, f"Control band deleted instead: coupling {CPL_CTRL*100:.1f}%", size=9.5, fill=MUT)

# ================================================================ legend
line(40, 736, 1140, 736, stroke=RULE, w=1)
leg = [(56, CLOCK, 2.4, None, "Period-3 clock"), (250, BLUE, 2, None, "Q · Donor query"),
       (420, GREEN, 2, "7 4", "K · Acceptor address"),
       (620, ORANGE, 2, "1.5 4.5", "V · Acceptor payload")]
for x, col, w_, dash, lab in leg:
    line(x, 752, x + 30, 752, stroke=col, w=w_, dash=dash, cap="round" if dash else "butt")
    txt(x + 37, 756, lab, size=10, fill="#46525C")
s.append('<line x1="830" y1="752" x2="860" y2="752" stroke="url(#gWriteL)" stroke-width="2.8"/>')
txt(867, 756, "Write-back into donor", size=10, fill="#46525C")
txt(56, 780, "Box = token state · Pill = attention head · Circle = projected vector", size=9.5,
    fill=MUT)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" version="1.1">\n'
       '<title>Donor–acceptor binding in SpliceBERT runs on a learned period-3 clock</title>\n'
       + "\n".join(s) + "\n</svg>\n")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(svg)
print(f"saved -> {OUT}  ({len(svg)/1024:.0f} KB)")
print(f"  phase mass {PM_INT:.3f} -> {PM_KILL:.3f} | coupling {CPL_INT:.3f} -> {CPL_KILL:.3f} "
      f"| L2H15 rescue {RESCUE*100:.1f}%")
