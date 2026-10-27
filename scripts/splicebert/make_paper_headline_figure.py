#!/usr/bin/env python3
"""Correct the Submission 39 headline figure (paper Fig. 2) from recorded artifacts.

The paper's Fig. 2 raster has no surviving renderer, so this script edits it in place:
panel A is kept as published; panel B loses the 22.30-nt marker (that bin was never
intervened on) and relabels the carrier ≈10.63 nt; panel C is redrawn from
``second_carrier.json`` on one shared colour scale with L3H1 added and the meaningless
sequence row removed; panels D-E are redrawn from a masked-donor validation window
exported by ``export_short_intron_attention.py --source val``.

    python scripts/splicebert/make_paper_headline_figure.py --base PAPER/headline.png --out OUT.png

No model is run here. Every input is checked against a pinned sha256, and a provenance
sidecar is written beside the output. Fonts are the macOS SF Pro / SF Mono system fonts
the published raster was typeset with; another platform cannot reproduce the bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402

BASE_SHA256 = "f4a0472d4f68de07ea77fda9a4a4281fab62f4ed2806dc77510267493cfd73d5"
SOURCES = {
    "spectrum": ("results/phase-lattice/fig_report_data.json",
                 "3bca5b20be7c51a158aa3214a288bf49f1bdd76636f412f6728e9ea9bd835057"),
    "rk": ("results/phase-lattice/second_carrier.json",
           "c06016929dc6188abb006b72e05b09081250b9616bc39202bcb98289066214b4"),
    "attention": ("figures/splicebert/viz/live_attention_val_window.json",
                  "d326c371c56b0a8ae0f5474d977fc80f39ba8edf7dc51273afce9c03a2cc2692"),
}
SF = "/System/Library/Fonts/SFNS.ttf"
SF_MONO = "/System/Library/Fonts/SFNSMono.ttf"

WHITE = (255, 255, 255)
INK = (26, 39, 51)
GREY = (107, 114, 128)
BLUE = (22, 104, 173)
ORANGE = (192, 88, 0)
PURPLE = (111, 74, 155)
DONOR_GREEN = (52, 224, 161)
ACCEPTOR_PINK = (255, 143, 150)
TAG_BOX = (9, 15, 23)
SPECTRUM_LINE = (74, 84, 98)
SPECTRUM_FILL = (232, 235, 239)

# Panel B: log-log axis fitted to the published stems, dots, median line and line peaks.
B_X0, B_X_PER_DECADE = 4302.5, 1062.96   # x at period 3 nt; px per decade of period
B_Y_UNIT, B_Y_PER_DECADE = 1591.0, 402.4  # y at power/median 1; px per decade
B_STRIP = (5186, 425, 5272, 1597)          # the removed 22.30-nt stem and dot
B_LEGEND = (4040, 2246, 7060, 2376)

C_HEADS = ("L0H0", "L0H11", "L1H10", "L1H12", "L2H2", "L2H15", "L3H1")
C_OFFSETS = tuple(range(-5, 13))
C_LIMIT = 2.5                              # logit; |deviation| max over shown rows is 2.43

ATTN_FLOOR = 8e-4                          # the published colour floor, kept
D_E_SHIFT = 4924                           # E is D translated right by this many px
MAP = 2240                                 # published map edge in px
ABS_X, ABS_Y, REL_X, REL_Y = 240, 3348, 2560, 3192


def font(size: float, weight: int = 400, mono: bool = False) -> ImageFont.FreeTypeFont:
    face = ImageFont.truetype(SF_MONO if mono else SF, round(size))
    face.set_variation_by_axes([weight, weight] if mono else [100, 20, 400, weight])
    return face


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    out = {}
    for key, (relative, expected) in SOURCES.items():
        path = REPO / relative
        if sha256(path) != expected:
            raise ValueError(f"source identity changed; review before rendering: {relative}")
        out[key] = json.loads(path.read_text())
    return out


def blend(a, b, t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def ramp(anchors, t: np.ndarray) -> np.ndarray:
    """Piecewise-linear colour ramp; anchors are (t, rgb) with t ascending in [0, 1]."""
    ts = np.array([a for a, _ in anchors])
    rgb = np.array([c for _, c in anchors], dtype=float)
    return np.stack([np.interp(t, ts, rgb[:, i]) for i in range(3)], -1).round().astype(np.uint8)


# ----------------------------------------------------------------------------- panel A
A_PLOT_X = (1072, 3375)                    # extent of the margin rows and axis
A_HEADING_INK = (105, 1232, 1442, 1348)    # "Masked-donor margin"
A_SEALED_INK = (2310, 1239, 3366, 1326)    # "Sealed test · 95% CI"
A_SEALED_TOP = 2266                        # level with panel B's legend


def patch_a(img: Image.Image) -> None:
    """Centre the heading over the plot; move the sealed-test note below the axis.

    The published pixels are moved, not retypeset, so the type is unchanged.
    """
    pad = 10
    centre = sum(A_PLOT_X) / 2
    blocks = [(box, img.crop((box[0] - pad, box[1] - pad, box[2] + pad + 1, box[3] + pad + 1)))
              for box in (A_HEADING_INK, A_SEALED_INK)]
    draw = ImageDraw.Draw(img)
    draw.rectangle((A_HEADING_INK[0] - pad, min(A_HEADING_INK[1], A_SEALED_INK[1]) - pad,
                    A_SEALED_INK[2] + pad, max(A_HEADING_INK[3], A_SEALED_INK[3]) + pad), fill=WHITE)
    (heading_box, heading), (sealed_box, sealed) = blocks
    img.paste(heading, (round(centre - heading.width / 2), heading_box[1] - pad))
    img.paste(sealed, (round(centre - sealed.width / 2), A_SEALED_TOP - pad))


# ----------------------------------------------------------------------------- panel B
def b_xy(period: float, power: float) -> tuple[float, float]:
    return (B_X0 + B_X_PER_DECADE * math.log10(period / 3),
            B_Y_UNIT - B_Y_PER_DECADE * math.log10(power))


def patch_b(img: Image.Image, spectrum: dict) -> None:
    """Repaint the spectrum under the removed 22.30-nt stem; rewrite the legend."""
    x0, y0, x1, y1 = B_STRIP
    ss = 4
    canvas = Image.new("RGB", ((x1 - x0) * ss, (y1 - y0) * ss), WHITE)
    draw = ImageDraw.Draw(canvas)
    points = sorted(zip(spectrum["period"], spectrum["rel"]))
    xy = [((x - x0) * ss, (y - y0) * ss) for x, y in (b_xy(p, v) for p, v in points)]
    near = [pt for pt in xy if -200 * ss < pt[0] < (x1 - x0 + 200) * ss]
    draw.polygon(near + [(near[-1][0], 4000 * ss), (near[0][0], 4000 * ss)], fill=SPECTRUM_FILL)
    # dashed median (power/median = 1): 28-px dashes every 48 px, phase taken from the figure
    for start in range(5315 - 48 * 4, x1, 48):
        left, right = max(start, x0), min(start + 28, x1)
        if right > left:
            draw.rectangle(((left - x0) * ss, (1587.75 - y0) * ss, (right - x0) * ss - 1,
                            (1593.75 - y0) * ss - 1), fill=(152, 160, 176))
    draw.line(near, fill=SPECTRUM_LINE, width=5 * ss, joint="curve")
    # Replace only what the teal marker touched (its hue, dilated 3 px); keep every other
    # published pixel, so no seam can form where the repaint meets the original line.
    patch = canvas.resize((x1 - x0, y1 - y0), Image.BOX)
    before = np.asarray(img.crop(B_STRIP)).astype(int)
    teal = (before[..., 1] - before[..., 0] > 18).astype(np.uint8) * 255
    mask = Image.fromarray(teal).filter(ImageFilter.MaxFilter(7))
    img.paste(patch, (x0, y0), mask)

    draw = ImageDraw.Draw(img)
    draw.rectangle(B_LEGEND, fill=WHITE)
    face = font(106.7, 700)
    items = [("3 nt · 289×", BLUE), ("≈10.63 nt · 546×", ORANGE)]
    dot, gap_dot, gap_item = 60, 39, 156
    widths = [dot + gap_dot + face.getlength(text) for text, _ in items]
    x = (4116 + 6986) / 2 - (sum(widths) + gap_item) / 2
    for (text, color), width in zip(items, widths):
        draw.ellipse((x, 2280, x + dot, 2280 + dot), fill=color)
        draw.text((x + dot + gap_dot, 2310), text, font=face, fill=color, anchor="lm")
        x += width + gap_item


# ----------------------------------------------------------------------------- panel C
def redraw_c(img: Image.Image, rk: dict) -> dict:
    draw = ImageDraw.Draw(img)
    draw.rectangle((7300, 300, 9868, 2500), fill=WHITE)
    draw.text((7385, 339), "Mean query–key logit by key offset k", font=font(120), fill=GREY,
              anchor="lt")

    offsets = rk["offsets"]
    by_name = {head["name"]: head for head in rk["heads"]}
    rows = {}
    for name in C_HEADS:
        values = np.array([by_name[name]["Rk_intact"][offsets.index(k)] for k in C_OFFSETS])
        rows[name] = values - values.mean()
    if max(float(np.abs(v).max()) for v in rows.values()) > C_LIMIT:
        raise ValueError("a row exceeds the shared colour limit; raise C_LIMIT")

    col_x0, col_pitch, col_w = 7880, 100, 92
    header = font(88, 400, mono=True)
    draw.text((7790, 605), "k", font=font(110), fill=GREY, anchor="rm")
    for j, k in enumerate(C_OFFSETS):
        if k % 3 == 0:
            draw.text((col_x0 + j * col_pitch + col_w / 2, 605), f"{k:d}".replace("-", "−"),
                      font=header, fill=GREY, anchor="mm")

    top, row_h, row_gap = 700, 208, 16
    label = font(115, 700, mono=True)
    neutral, low = (243, 244, 246), (156, 163, 175)
    for i, name in enumerate(C_HEADS):
        y = top + i * (row_h + row_gap)
        color = BLUE if name == "L2H15" else PURPLE if name == "L3H1" else INK
        draw.text((7790, y + row_h / 2), name, font=label, fill=color, anchor="rm")
        for j, value in enumerate(rows[name]):
            t = min(abs(value) / C_LIMIT, 1.0)
            fill = blend(neutral, BLUE if value > 0 else low, t)
            x = col_x0 + j * col_pitch
            draw.rectangle((x, y, x + col_w - 1, y + row_h - 1), fill=fill)

    # one shared scale, drawn as a continuous bar
    legend = font(110)
    y_mid, bar_x, bar_w, bar_h = 2381, 7600, 600, 44
    draw.text((bar_x - 30, y_mid), f"−{C_LIMIT:g}", font=legend, fill=GREY, anchor="rm")
    t = np.linspace(-1, 1, bar_w)
    strip = np.stack([np.array(blend(neutral, BLUE if v > 0 else low, abs(v))) for v in t])
    img.paste(Image.fromarray(np.repeat(strip[None].astype(np.uint8), bar_h, 0)),
              (bar_x, y_mid - bar_h // 2))
    draw.text((bar_x + bar_w + 30, y_mid), f"+{C_LIMIT:g} logit vs head mean", font=legend,
              fill=GREY, anchor="lm")
    return {name: [round(float(v), 4) for v in values] for name, values in rows.items()}


# --------------------------------------------------------------------------- panels D-E
D_RAMP = ((0.0, (9, 15, 23)), (0.35, (14, 52, 85)), (0.62, (22, 104, 173)),
          (0.84, (131, 175, 211)), (1.0, (226, 237, 247)))
E_RAMP = ((0.0, (9, 15, 23)), (0.3, (43, 35, 68)), (0.56, (92, 63, 130)),
          (0.82, (165, 142, 193)), (1.0, (237, 232, 246)))


def heat(values: np.ndarray, vmax: float, anchors) -> Image.Image:
    lo, hi = math.log10(ATTN_FLOOR), math.log10(vmax)
    t = (np.log10(np.clip(values, ATTN_FLOOR, vmax)) - lo) / (hi - lo)
    return Image.fromarray(ramp(anchors, t)).resize((MAP, MAP), Image.NEAREST)


def dashed(draw, start, end, color, width=5, dash=30, gap=20) -> None:
    (xa, ya), (xb, yb) = start, end
    length = math.hypot(xb - xa, yb - ya)
    pos = 0.0
    while pos < length:
        a, b = pos / length, min(pos + dash, length) / length
        draw.line((xa + (xb - xa) * a, ya + (yb - ya) * a, xa + (xb - xa) * b, ya + (yb - ya) * b),
                  fill=color, width=width)
        pos += dash + gap


def tag(img: Image.Image, text: str, color, x: float, y: float, align: str) -> None:
    face = font(84, 700)
    width = face.getlength(text)
    pad_x, height = 26, 104
    left = x if align == "left" else x - width - 2 * pad_x
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((left, y, left + width + 2 * pad_x, y + height), radius=16,
                         fill=TAG_BOX + (225,))
    od.text((left + pad_x, y + height / 2 + 2), text, font=face, fill=color + (255,), anchor="lm")
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def redraw_head(img: Image.Image, att: dict, head: str, dx: int, anchors, accent,
                subtitle: str) -> dict:
    draw = ImageDraw.Draw(img)
    left = 0 if dx == 0 else 4920
    right = 4900 if dx == 0 else 9868
    draw.rectangle((left, 2845, right, 5852), fill=WHITE)
    draw.text((106 + (5020 - 106) * (dx > 0), 2866), subtitle, font=font(124), fill=GREY,
              anchor="lt")

    title = font(121, 600)
    draw.text((ABS_X + dx - 1, 3059), "Absolute-Position Frame", font=title, fill=INK, anchor="lt")
    draw.text((REL_X + dx, 3059), "Relative-Offset Frame", font=title, fill=INK, anchor="lt")
    draw.text((REL_X + dx + MAP, 3059), "A[q, q+δ]", font=font(116), fill=GREY, anchor="rt")

    maps = att["head_maps"][head]
    absolute, relative = np.array(maps["absolute"]), np.array(maps["relative"])
    vmax = float(max(absolute.max(), relative.max()))
    img.paste(heat(absolute, vmax, anchors), (ABS_X + dx, ABS_Y))
    img.paste(heat(relative, vmax, anchors), (REL_X + dx, REL_Y))

    # key-column attention mass over the crop's queries; blue above the mean, grey below
    n = absolute.shape[0]
    cell = MAP / n
    mass = absolute.mean(0)
    base_y, top_y = ABS_Y - 20, ABS_Y - 150
    scale = (base_y - top_y) / mass.max()
    for j, value in enumerate(mass):
        x = ABS_X + dx + j * cell
        draw.rectangle((x + 2, base_y - value * scale, x + cell - 3, base_y),
                       fill=accent if value > mass.mean() else (196, 199, 207))
    mean_y = base_y - mass.mean() * scale
    dashed(draw, (ABS_X + dx, mean_y), (ABS_X + dx + MAP, mean_y), (200, 203, 210), width=6,
           dash=26, gap=18)

    crop_lo = att["crop"][0]
    donor, acceptor = att["donor_nt"] - crop_lo, att["acceptor_nt"] - crop_lo
    for index, color in ((donor, DONOR_GREEN), (acceptor, ACCEPTOR_PINK)):
        centre = (index + 0.5) * cell
        dashed(draw, (ABS_X + dx + centre, ABS_Y), (ABS_X + dx + centre, ABS_Y + MAP), color)
        dashed(draw, (ABS_X + dx, ABS_Y + centre), (ABS_X + dx + MAP, ABS_Y + centre), color)
        draw.rectangle((ABS_X + dx + centre - 4, base_y + 2, ABS_X + dx + centre + 4, base_y + 16),
                       fill=color)
    tag(img, "DONOR GT", DONOR_GREEN, ABS_X + dx + 48, ABS_Y + 24, "left")
    tag(img, "ACCEPTOR AG", ACCEPTOR_PINK, ABS_X + dx + MAP - 8, ABS_Y + 24, "right")

    draw = ImageDraw.Draw(img)
    label = font(118.6)
    draw.text((ABS_X + dx + 9, 5747), "Key →", font=label, fill=GREY, anchor="ls")
    draw.text((REL_X + dx + 9, REL_Y + MAP + 160), "Offset δ →", font=label, fill=GREY, anchor="ls")
    draw.text((ABS_X + dx + MAP - 8, 5747), f"log {ATTN_FLOOR:g}–{vmax:.3f}",
              font=font(114, 400, mono=True), fill=GREY, anchor="rs")
    query = Image.new("RGBA", (600, 140), (0, 0, 0, 0))
    ImageDraw.Draw(query).text((300, 70), "Query →", font=label, fill=GREY + (255,), anchor="mm")
    query = query.rotate(90, expand=True)
    img.paste(query, (round(176 + dx - 70), round(ABS_Y + MAP / 2 - 300)), query)
    return {"vmax": vmax, "crop_nt": n, "donor_index": donor, "acceptor_index": acceptor}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="published Fig. 2 raster")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    provenance_path = args.out.with_suffix(".provenance.json")
    for path in (args.out, provenance_path):
        guard_output_path(path, overwrite=args.overwrite)
    if sha256(args.base) != BASE_SHA256:
        raise ValueError("base raster is not the published Fig. 2")
    data = load()
    att = data["attention"]

    img = Image.open(args.base).convert("RGB")
    patch_a(img)
    patch_b(img, data["spectrum"]["spectra"]["final_1024_multispecies"])
    rows = redraw_c(img, data["rk"])
    window = (f"val {att['chrom']}, {att['intron_len']}-nt intron, donor masked")
    n = att["crop"][1] - att["crop"][0]
    d = redraw_head(img, att, "L2H15", 0, D_RAMP, BLUE,
                    f"Periodic diagonal, {n} × {n} nt crop ({window})")
    e = redraw_head(img, att, "L3H1", D_E_SHIFT, E_RAMP, PURPLE,
                    f"Vertical stripes beside the masked donor and at the acceptor AG; same crop")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, optimize=True)

    sources = [{"path": "paper:headline.png (published Fig. 2, Submission 39)", "sha256": BASE_SHA256}]
    sources += [{"path": relative, "sha256": sha256(REPO / relative)} for relative, _ in SOURCES.values()]
    sources.append({"path": "scripts/splicebert/make_paper_headline_figure.py",
                    "sha256": sha256(Path(__file__))})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip())
    provenance = {
        "git_commit": revision,
        "git_dirty": dirty,
        "command": ["python", *sys.argv],
        "output_sha256": sha256(args.out),
        "sources": sources,
        "checkpoint_weights_sha256": "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d",
        "rendering": {
            "A": "Published pixels; the 'Masked-donor margin' heading is moved to centre over the plot "
                 "and the 'Sealed test · 95% CI' note is moved below the axis.",
            "B": "22.30-nt (bin 46) stem and legend removed and the spectrum line repainted from the "
                 "recorded bins; that bin was never intervened on. Carrier relabelled ≈10.63 nt "
                 "(peak bin 96, period 10.69). Shading unchanged: only the 3-nt and carrier bands.",
            "C": "Mean query-key logit R(k) = mean_q q·k(q+k)/sqrt(d), intact embeddings, minus each "
                 "head's mean over k = -5..12, on one shared diverging scale (±2.5 logit). Rows: the "
                 "six period-3 comb heads plus L3H1. The sequence row is removed.",
            "D-E": "Post-softmax attention, one masked-donor validation window; log colour scale "
                   "from 0.0008 to the head's maximum over both frames. Relative frame "
                   "A[q, q+δ], δ = k − q. Bars: mean attention received per key over crop queries.",
        },
        "values": {"C_rows_logit_minus_head_mean": rows, "D": d, "E": e,
                   "example_id": att["example_id"], "example_coupling": att["example_coupling"]},
        "limitations": [
            "Panel A (apart from two moved labels) and the rest of B are the published pixels; "
            "their renderer is lost.",
            "D-E show one example; descriptive only (no OV, mediation or cohort claim).",
            "C uses second_carrier.json (32 sequences, positions 260-760), not fig_phase_data.json; "
            "the five previously plotted rows differ from it by < 0.1 logit.",
        ],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(args.out)
    print(provenance_path)


if __name__ == "__main__":
    main()
