"""Inline plots for the notebooks, on the repo's own figure style, plus the saturation guard.

These are notebook helpers, not the tracked figure pipeline (`scripts/fig_phase_*.py` owns that, and
never re-derives a number). They take the dataclasses/arrays the assays return and draw them, so a
study notebook reads as: measure -> plot -> read the verdict off the plot.

The one non-cosmetic thing here is `saturation_guard`. Four times this project nearly shipped a wrong
conclusion because a metric sat on its ceiling and reported "nothing here" exactly where the effect
was largest (D20 softmax, D27 accuracy=1.0, D31 accuracy mid-grok). The guard makes that failure
loud: hand it the metric you are about to conclude from, and it warns if you are reading at a ceiling.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from gfmcd.figstyle import (
    ANCHORED,
    ANCHOR_FREE,
    AQUA,
    BAG,
    BLUE,
    COUNTER,
    INK,
    INK_2,
    INK_MUTED,
    RED,
    RELAY,
    VIOLET,
    apply_figure_style,
    annotate,
)

apply_figure_style()


# ----------------------------------------------------------------------------
# the house-rule guard
# ----------------------------------------------------------------------------


def saturation_guard(value: float, name: str = "metric", ceiling: float = 1.0, tol: float = 1e-3) -> bool:
    """Warn if `value` sits within `tol` of a ceiling (or floor 0). Returns True if saturated.

    Call this on any metric you are about to draw a conclusion from. The repo's recurring failure
    class is a saturated measurement reporting nothing where the effect is largest -- accuracy pinned
    at 1.0 (D27), accuracy read mid-grok (D31), a softmax gradient annihilated (D20). If this fires,
    switch to a non-saturating readout: loss instead of accuracy, an unsaturated distance bin, an
    integrated gradient instead of a point gradient.
    """
    hi = abs(value - ceiling) <= tol
    lo = abs(value) <= tol
    if hi or lo:
        edge = f"ceiling {ceiling}" if hi else "floor 0"
        warnings.warn(
            f"[saturation guard] {name} = {value:.6g} is at its {edge}. A saturated metric reports "
            f"nothing where the effect is largest (D20/D27/D31). Read a non-saturating quantity "
            f"(loss, not accuracy) before concluding.",
            stacklevel=2,
        )
        return True
    return False


# ----------------------------------------------------------------------------
# PILLAR 1 -- discovery
# ----------------------------------------------------------------------------


def plot_attention_survey(attn: np.ndarray, title: str = "attention to target, per head", ax=None):
    """Heatmap of [n_layers, n_heads] attention. The 'who looks responsible' screen."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(attn, aspect="auto", cmap="magma", origin="upper")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(attn.shape[1]))
    ax.set_yticks(range(attn.shape[0]))
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean attention")
    return ax


# ----------------------------------------------------------------------------
# PILLAR 3 -- ablation: attention rank vs causal effect, side by side (the L1H2 lesson)
# ----------------------------------------------------------------------------


def plot_head_effects(effects, top: int = 10, title: str = "attention rank vs causal effect", ax=None):
    """Two bars per head: attention-to-target and Δloss when ablated. The point of the panel is that
    they DISAGREE -- a head can top attention and be causally inert (L1H2), which is why only the loss
    bar decides. `effects` is a list of HeadEffect (from head_ablation_table)."""
    effects = sorted(effects, key=lambda e: e.delta_loss, reverse=True)[:top]
    labels = [f"L{e.layer}H{e.head}" for e in effects]
    dloss = np.array([e.delta_loss for e in effects])
    attn = np.array([e.attn_to_target for e in effects])
    x = np.arange(len(labels))

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - 0.2, dloss, width=0.4, color=BLUE, label="Δloss when ablated (decides)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, attn, width=0.4, color=INK_2, alpha=0.6, label="attention to target (ranks)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δ copy/aa loss", color=BLUE)
    ax2.set_ylabel("attention to target", color=INK_2)
    ax.set_title(title)
    ax.grid(False)
    ax2.grid(False)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=7.5)
    return ax


def plot_redundancy(red, title: str = "single vs joint ablation", ax=None):
    """Bar of each head's solo Δloss, with the sum-of-singles and the joint Δloss as reference lines.
    joint << sum => partial redundancy (the copy-circuit finding). `red` is a RedundancyResult."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.4))
    keys = list(red.singles)
    labels = [f"L{l}H{h}" for l, h in keys]
    vals = [red.singles[k] for k in keys]
    x = np.arange(len(labels))
    ax.bar(x, vals, color=BLUE, alpha=0.85)
    ax.axhline(red.sum_of_singles, color=RED, ls="--", lw=1.3, label=f"Σ singles = {red.sum_of_singles:.2f}")
    ax.axhline(red.subset_delta, color=RELAY, ls="-", lw=1.6, label=f"joint = {red.subset_delta:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δloss")
    ax.set_title(title)
    ax.legend(fontsize=7.5)
    return ax


# ----------------------------------------------------------------------------
# PILLAR 2/4 -- the ceiling
# ----------------------------------------------------------------------------


def plot_ceiling(report, title: str = "shortcut ceiling vs model", ax=None):
    """Three bars: pooled-statistic ceiling, model, and the objective's demand. If the model only
    matches the ceiling, it is pooling. `report` is a CeilingReport."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.4))
    names = ["pooled\nceiling", "model\n(phase)", "demanded"]
    vals = [report.pooled_oracle_acc, report.model_phase_acc, report.demanded]
    colors = [INK_2, ANCHORED if report.circuit_predicted else ANCHOR_FREE, "#b9b8b3"]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("frame accuracy")
    ax.set_title(title)
    return ax


# ----------------------------------------------------------------------------
# PILLAR 5 -- conflict + the mechanism map
# ----------------------------------------------------------------------------


def plot_cue_conflict(curve, title: str = "cue conflict: anchor vs content by distance", ax=None):
    """Anchor-share and content-share vs distance from the frameshift edit. A counter holds anchor at
    range; a bag crosses to content. `curve` is the list from cue_conflict_curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.4))
    d = [c["distance"] for c in curve]
    a = [c["anchor_share"] for c in curve]
    ct = [c["content_share"] for c in curve]
    ax.plot(d, a, "-o", color=ANCHORED, label="follows anchor (counter)")
    ax.plot(d, ct, "-o", color=ANCHOR_FREE, label="follows content (bag)")
    ax.axhline(0.5, color=INK_2, ls=":", lw=1)
    ax.set_xlabel("distance from edit (nt)")
    ax.set_ylabel("share")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=7.5)
    return ax


def plot_grokking(history, acc_key: str, loss_key: str, transition: dict | None = None,
                  base_floor: float | None = None, title: str = "circuit formation (grokking)", ax=None):
    """Training curve: CE loss and accuracy vs step, with the transition step marked.

    `history` is the list from load_history; `transition` is the dict from grok_transition. The point of
    the panel is *when the circuit forms* -- the abrupt accuracy rise, not the smooth loss decline.
    """
    steps = [r["step"] for r in history]
    acc = [r[acc_key] for r in history]
    loss = [r[loss_key] for r in history]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(steps, loss, "-o", ms=2.8, color=INK_2, label="CE loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("CE loss", color=INK_2)
    ax.grid(False)
    ax2 = ax.twinx()
    ax2.plot(steps, acc, "-o", ms=2.8, color=BLUE, label=acc_key)
    ax2.set_ylabel(acc_key, color=BLUE)
    ax2.set_ylim(0, 1.03)
    ax2.grid(False)
    if base_floor is not None:
        ax2.axhline(base_floor, color=RED, ls="--", lw=1.1, label=f"base-rate floor {base_floor:.3f}")
    if transition and transition.get("transition_step"):
        ts = transition["transition_step"]
        ax.axvline(ts, color=RELAY, ls=":", lw=1.4)
        ax.text(ts, ax.get_ylim()[1] * 0.95, f"  circuit forms\n  ~step {ts}", color=RELAY, fontsize=8, va="top")
    ax.set_title(title)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=7.5)
    return ax


def plot_lambda_sweep(points, title: str = "mechanism vs shortcut strength", ax=None):
    """Anchor share vs shortcut information content (I(nt;phase)), one point per λ arm.

    As the shortcut carries more of the frame the mechanism shifts counter -> bag, while accuracy stays
    ~1.0 -- so only the mechanism axis (colour + y) separates the arms. `points` is from lambda_sweep.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.8))
    xs = [p.shortcut_nats for p in points]
    ys = [p.anchor_share for p in points]
    for p in points:
        color = COUNTER if "counter" in p.verdict else (RELAY if "relay" in p.verdict else BAG)
        ax.scatter([p.shortcut_nats], [p.anchor_share], color=color, s=70, zorder=3)
        lbl = f"λ={p.lam:g}" if not np.isnan(p.lam) else p.name
        ax.annotate(lbl, (p.shortcut_nats, p.anchor_share), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color=INK_2)
    ax.plot(xs, ys, "-", color=INK_MUTED, lw=1.0, zorder=1)
    ax.axhline(0.5, color=INK_2, ls=":", lw=1)
    ax.set_xlabel("shortcut information  I(nt; phase)  [nats]")
    ax.set_ylabel("anchor share (counter → bag)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(title)
    return ax


def plot_2x2_grid(verdicts, title: str = "2×2 mechanism verdict", ax=None):
    """The 2x2 cell grid: shortcut (absent/present) x detector (easy/hard), coloured by verdict.

    `verdicts` is a dict {cell_letter: Verdict2x2}. Cell A top-left through D bottom-right, each cell
    annotated with its anchor share, ablation effect, and formed/not verdict.
    """
    layout = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.grid(False)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["detector easy", "detector hard"])
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["shortcut absent", "shortcut present"])
    for cell, (r, c) in layout.items():
        v = verdicts.get(cell)
        x0, y0 = c, 1 - r
        if v is None:
            ax.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor="#f0efec", edgecolor="#d8d7d2"))
            ax.text(x0 + 0.5, y0 + 0.5, f"{cell}\n(absent)", ha="center", va="center", fontsize=9, color=INK_MUTED)
            continue
        color = COUNTER if v.circuit_formed else BAG
        ax.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor=color, alpha=0.14, edgecolor=color, lw=1.5))
        verdict = "CIRCUIT" if v.circuit_formed else "no circuit"
        ax.text(x0 + 0.5, y0 + 0.72, f"cell {cell}", ha="center", fontsize=10, weight="bold", color=INK)
        ax.text(x0 + 0.5, y0 + 0.48, verdict, ha="center", fontsize=9, color=color, weight="bold")
        ax.text(x0 + 0.5, y0 + 0.24, f"y={v.anchor_share:.2f}  Δabl={v.ablation_effect:+.2f}",
                ha="center", fontsize=7.5, color=INK_2)
    ax.set_title(title)
    return ax


def plot_circuit_flow(stages, title: str = "circuit — information flow", ax=None,
                      probe_by_layer: list[float] | None = None):
    """A layer-by-layer flow diagram of the circuit: what each layer computes and which heads carry it.

    `stages` is an ordered list of dicts, one per drawn stage, each:
        {"layer": "L0", "role": "locate the anchor", "heads": "L1H3",
         "detail": "attends the ATG", "kind": "anchor"|"propagate"|"readout"|"setup"}
    Drawn bottom-to-top (input at the bottom, readout at the top), matching residual-stream flow. This
    is a schematic to trace the logic by hand -- the numbers come from the notebook's own pillar cells,
    not from this function. `probe_by_layer` (optional) overlays where the target becomes decodable.
    """
    kind_color = {"setup": INK_MUTED, "anchor": BLUE, "propagate": VIOLET, "readout": AQUA}
    n = len(stages)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 1.05 * n + 1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1)
    ax.axis("off")
    ax.set_title(title)

    # input token row at the bottom
    ax.add_patch(plt.Rectangle((1.5, 0.15), 7, 0.55, facecolor="#f0efec", edgecolor="#d8d7d2"))
    ax.text(5, 0.42, "input: DNA tokens  {A,C,G,T,N}", ha="center", va="center", fontsize=8.5, color=INK_2)

    for i, st in enumerate(stages):
        y = i + 1
        color = kind_color.get(st.get("kind", "setup"), INK_MUTED)
        ax.add_patch(plt.Rectangle((1.5, y + 0.12), 7, 0.66, facecolor=color, alpha=0.13,
                                   edgecolor=color, lw=1.6))
        ax.text(1.75, y + 0.45, st.get("layer", ""), ha="left", va="center", fontsize=9,
                weight="bold", color=color)
        ax.text(3.1, y + 0.58, st.get("role", ""), ha="left", va="center", fontsize=8.8, color=INK)
        detail = st.get("detail", "")
        heads = st.get("heads", "")
        sub = "  ".join([s for s in [heads, detail] if s])
        ax.text(3.1, y + 0.30, sub, ha="left", va="center", fontsize=7.6, color=INK_2)
        # upward arrow (residual stream)
        if i < n - 1:
            ax.annotate("", xy=(5, y + 1.10), xytext=(5, y + 0.80),
                        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.2))
        else:
            ax.annotate("", xy=(5, y + 1.02), xytext=(5, y + 0.80),
                        arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.4))
        # arrow from input to first stage
        if i == 0:
            ax.annotate("", xy=(5, y + 0.10), xytext=(5, 0.72),
                        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.2))
        if probe_by_layer is not None and i < len(probe_by_layer):
            ax.text(8.7, y + 0.45, f"decodable\n{probe_by_layer[i]:.2f}", ha="center", va="center",
                    fontsize=6.8, color=INK_MUTED)
    return ax


def plot_mechanism_map(signatures, title: str = "mechanism signature", ax=None):
    """Scatter the 2-D mechanism signature: x = anchor dependence, y = anchor share. The three regions
    are counter (top-right), relay (bottom-right), bag (bottom-left). `signatures` is one
    MechanismSignature or a list of them."""
    if not isinstance(signatures, (list, tuple)):
        signatures = [signatures]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.axhspan(0.5, 1.05, 0.0, 1.0, color=BLUE, alpha=0.05)
    ax.axhline(0.5, color=INK_2, ls=":", lw=1)
    ax.axvline(0.10, color=INK_2, ls=":", lw=1)
    for s in signatures:
        if "counter" in s.verdict:
            color = COUNTER
        elif "relay" in s.verdict:
            color = RELAY
        else:
            color = BAG
        ax.scatter([s.anchor_dependence], [s.anchor_share], color=color, s=70, zorder=3)
        ax.annotate(s.name, (s.anchor_dependence, s.anchor_share), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color=INK_2)
    ax.text(0.02, 0.96, "counter", fontsize=8, color=COUNTER, transform=ax.transAxes)
    ax.text(0.72, 0.03, "relay", fontsize=8, color=RELAY, transform=ax.transAxes)
    ax.text(0.02, 0.03, "bag", fontsize=8, color=BAG, transform=ax.transAxes)
    ax.set_xlabel("anchor dependence (far-field ablation, peak over depth)")
    ax.set_ylabel("anchor share under cue conflict")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(title)
    return ax
