"""Shared SVG primitives and the VALIDATED palette for the four publication figures.

THE PALETTE IS NOT A TASTE DECISION -- IT WAS COMPUTED.  Every pair of colours that appears
together in a panel was checked for separation under normal vision and under simulated
deuteranopia / protanopia / tritanopia (Vienot dichromacy model, dE in CIE Lab).  Floor: 12.

The palette that the earlier figures used FAILED that check on the pairs these figures actually
compare -- BLUE vs GRAY came out at dE = 0.9 under deuteranopia, and those two are "intact" vs
"deleted", the single most important contrast in the causal figure.  BLUE vs GREEN was 1.3 under
tritanopia, and those are the exon/intron curves.  Both are fixed here.

ONE HUE, ONE MEANING, ACROSS ALL FOUR FIGURES:

    BLUE     period-3 carrier                     ORANGE   ~10.63 carrier
    PURPLE   donor-acceptor coupling (behaviour)  GREEN    intron
    INK      exon (coding)                        RED      the nucleosomal 10.298 signal
    SLATE    deleted / failed / not Level A       MIST     unresolvable / other bins

BLUE vs ORANGE -- the two carriers, the central distinction of the whole report -- separate by
dE = 74 under every colour-vision model.  Nothing else in the set comes close to colliding.

The layer ramp in Figure 4A is SEQUENTIAL (layer is ordinal, read by lightness order), so the
categorical dE >= 12 floor does not apply to its adjacent steps; the Level-A highlight clears
every step of it by dE >= 39.
"""
from __future__ import annotations

import math

# ---- semantic palette (validated; see module docstring)
BLUE = "#0B6FB8"      # period-3 carrier
ORANGE = "#C25A00"    # ~10.63 carrier
PURPLE = "#6A4A9C"    # donor-acceptor coupling
GREEN = "#00845F"     # intron
RED = "#9E2A2B"       # nucleosomal 10.298
SLATE = "#55606B"     # deleted / failed
MIST = "#E3E9EE"      # recessive context
GOLD = "#B08800"      # codon phase 1
MAGENTA = "#A5195F"   # codon phase 2
PALEBLUE = "#A8CCE8"  # control band deleted
SAND = "#D9A05B"      # reciprocal-carrier deleted
OLIVE = "#8C6D3F"     # low-leak band deleted
ASH = "#9AA7B2"       # content shuffled

INK, SEC, MUT, RULE, PANEL = "#16232E", "#5A6672", "#8794A0", "#E1E7EC", "#F7F9FB"
PHASE = [BLUE, GOLD, MAGENTA]                       # codon position 0 / 1 / 2
LAYER = ["#D3DAE0", "#B4BEC7", "#94A1AC", "#74848F", "#586772", "#3C4A55"]   # sequential

FONT = ('<style type="text/css"><![CDATA[ .sans{font-family:"Helvetica Neue",Helvetica,Arial,'
        'sans-serif} .mono{font-family:"SF Mono",Menlo,Consolas,monospace} ]]></style>')


class Canvas:
    """A minimal SVG page.  Short labels only -- the narrative lives in the .CAPTION.md."""

    def __init__(self, w: int, h: int):
        self.w, self.h, self.s = w, h, []
        self.s.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#FFFFFF"/>')

    # -- primitives
    def txt(self, x, y, t, *, size=11, fill=SEC, anchor="start", weight=None, rot=None,
            cls="sans", op=1.0):
        t = str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        w = f' font-weight="{weight}"' if weight else ""
        o = f' fill-opacity="{op}"' if op != 1.0 else ""
        if rot is not None:
            self.s.append(f'<text class="{cls}" transform="translate({x:.1f},{y:.1f}) rotate({rot})"'
                          f' text-anchor="{anchor}" font-size="{size}" fill="{fill}"{w}{o}>{t}</text>')
        else:
            a = f' text-anchor="{anchor}"' if anchor != "start" else ""
            self.s.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                          f'fill="{fill}"{a}{w}{o}>{t}</text>')

    def line(self, x1, y1, x2, y2, *, stroke=RULE, w=1, dash=None, cap="butt", op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' stroke-opacity="{op}"' if op != 1.0 else ""
        self.s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{stroke}" stroke-width="{w}" stroke-linecap="{cap}"{d}{o}/>')

    def rect(self, x, y, w_, h_, *, fill="none", stroke=None, rx=0, sw=1, op=1.0, dash=None):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        o = f' fill-opacity="{op}"' if op != 1.0 else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w_,0):.1f}" '
                      f'height="{max(h_,0):.1f}" rx="{rx}" fill="{fill}"{st}{o}{d}/>')

    def path(self, pts, *, stroke, w=2, dash=None, op=1.0, fill="none"):
        if not pts:
            return
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' stroke-opacity="{op}"' if op != 1.0 else ""
        self.s.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
                      f'stroke-linejoin="round" stroke-linecap="round"{da}{o}/>')

    def dot(self, x, y, r, fill, *, ring="#FFFFFF", rw=1.2, op=1.0):
        o = f' fill-opacity="{op}"' if op != 1.0 else ""
        self.s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
                      f'stroke="{ring}" stroke-width="{rw}"{o}/>')

    def ring(self, x, y, r, stroke, *, w=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" '
                      f'stroke="{stroke}" stroke-width="{w}"{d}/>')

    def arrow(self, x1, y1, x2, y2, *, stroke=INK, w=2, dash=None, head=6):
        a = math.atan2(y2 - y1, x2 - x1)
        xs, ys = x2 - head * math.cos(a), y2 - head * math.sin(a)
        self.line(x1, y1, xs, ys, stroke=stroke, w=w, dash=dash, cap="round")
        p = [(x2, y2),
             (x2 - head * 1.5 * math.cos(a - .38), y2 - head * 1.5 * math.sin(a - .38)),
             (x2 - head * 1.5 * math.cos(a + .38), y2 - head * 1.5 * math.sin(a + .38))]
        self.s.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in p) +
                      f'" fill="{stroke}"/>')

    # -- composites
    def plabel(self, x, y, ltr, title):
        self.txt(x, y, ltr, size=15, fill=INK, weight="700")
        self.txt(x + 17, y, title, size=12.5, fill=INK, weight="700")

    def bar(self, x, y, w_, h_, fill, *, rx=3, op=1.0):
        """A bar with 4px-rounded data-end, anchored to the baseline."""
        self.rect(x, y, w_, h_, fill=fill, rx=min(rx, w_ / 2), op=op)
        if h_ > rx:
            self.rect(x, y + h_ - rx, w_, rx, fill=fill, op=op)

    def swatch(self, x, y, label, fill, *, size=10.5, sw=11, hatch=False):
        self.rect(x, y - sw + 2.5, sw, sw, fill=fill, rx=2)
        if hatch:
            self.rect(x, y - sw + 2.5, sw, sw, fill="url(#hatch)", rx=2)
        self.txt(x + sw + 6, y, label, size=size, fill=SEC)

    def render(self, path):
        # NO <?xml?> prolog.  The PNG preview nests this whole document inside a square wrapper
        # <svg>, and an XML declaration is only legal at byte zero -- a prolog here turns the
        # preview into a WebKit parse-error page while the SVG itself still validates.
        body = "".join(self.s)
        doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}">'
               f'<defs>{FONT}'
               f'<pattern id="hatch" width="5" height="5" patternTransform="rotate(45)" '
               f'patternUnits="userSpaceOnUse">'
               f'<line x1="0" y1="0" x2="0" y2="5" stroke="#FFFFFF" stroke-width="2.2"/></pattern>'
               f'</defs>{body}</svg>')
        path.write_text(doc)
        return path


def nice_p(p: float) -> str:
    """Format an exact add-one permutation p-value.  NEVER prints zero."""
    return f"p = {p:.4f}" if p >= 1e-4 else f"p < {1e-4:.4f}"
