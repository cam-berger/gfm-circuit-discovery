"""Trace-hardening controls for the repeat-copy circuit.

Runs two Phase-C checks:
  1. single vs subset head ablations for the L1 source-copy heads;
  2. prefix-scramble control: edit the distal continuation as in the cue conflict, then corrupt the
     preceding source prefix while leaving the local second-copy prefix intact.

This is not a replacement for a full IG-QK decomposition, but it directly tests the two immediate
risks exposed by the trace screen: redundant heads and apparent source-copying without prefix keying.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gfmcd.paths import project_paths

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.repeat_copy import IGNORE, ar_inputs_targets, make_batch, single_conflict_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig


def parse_heads(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not (part.startswith("L") and "H" in part):
            raise ValueError(f"bad head spec: {part}")
        layer, head = part[1:].split("H", 1)
        out.append((int(layer), int(head)))
    return out


@torch.no_grad()
def copy_loss(model: GFMNano, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, device: str, batch: int) -> float:
    total = n = 0.0
    for i in range(0, len(x), batch):
        xb, yb, mb = x[i : i + batch].to(device), y[i : i + batch].to(device), mask[i : i + batch].to(device)
        logits = model(xb)
        total += float(F.cross_entropy(logits[mb], yb[mb], reduction="sum"))
        n += float(mb.sum())
    return total / n


def set_head_masks(model: GFMNano, heads: list[tuple[int, int]], value: float) -> None:
    for layer, head in heads:
        model.blocks[layer].attn.head_mask[head] = value


def reset_head_masks(model: GFMNano) -> None:
    for block in model.blocks:
        block.attn.head_mask[:] = 1


def prefix_scramble(tokens: np.ndarray, data, score_mask: np.ndarray, seed: int) -> np.ndarray:
    """Corrupt the source prefix before the edited source continuation, preserving local prefix."""

    rng = np.random.default_rng(seed)
    out = tokens.copy()
    rows, pred_positions = np.nonzero(score_mask)
    for r, p in zip(rows, pred_positions):
        src = int(data.source_index[r, p])
        start = int(data.first_start[r])
        for q in range(start, src):
            out[r, q] = (out[r, q] + int(rng.integers(1, 4))) % 4
    return out


@torch.no_grad()
def conflict_shares(model: GFMNano, tokens: np.ndarray, distal: np.ndarray, local: np.ndarray, mask: np.ndarray, device: str, batch: int) -> dict:
    x = torch.from_numpy(tokens[:, :-1]).long()
    preds = []
    for i in range(0, len(x), batch):
        preds.append(model(x[i : i + batch].to(device)).argmax(-1).cpu().numpy())
    pred = np.concatenate(preds)
    return {
        "distal": float((pred[mask] == distal[mask]).mean()),
        "local": float((pred[mask] == local[mask]).mean()),
        "neither": float(1.0 - (pred[mask] == distal[mask]).mean() - (pred[mask] == local[mask]).mean()),
    }


@torch.no_grad()
def mean_attention_to_source(model: GFMNano, tokens: np.ndarray, data, mask: np.ndarray, heads: list[tuple[int, int]], device: str) -> dict[str, float]:
    x = torch.from_numpy(tokens[: min(len(tokens), 256), :-1]).long().to(device)
    _, cache = model.run_with_cache(x)
    rows, pred_positions = np.nonzero(mask[: len(x)])
    out = {}
    for layer, head in heads:
        pattern = cache[f"L{layer}.attn.pattern"].cpu().numpy()
        vals = []
        for r, p in zip(rows, pred_positions):
            vals.append(pattern[r, head, p, int(data.source_index[r, p])])
        out[f"L{layer}H{head}"] = float(np.mean(vals))
    return out


def apply_style() -> None:
    from gfmcd.figstyle import apply_figure_style

    apply_figure_style()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/repeat_copy_s0_step8000.pt"))
    ap.add_argument("--seqs", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=6262)
    ap.add_argument("--heads", default="L1H7,L1H0,L1H5,L1H1,L1H6")
    ap.add_argument("--out-dir", type=Path, default=project_paths().results / "repeat-copy")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    heads = parse_heads(args.heads)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    data = make_batch(args.seqs, window=256, seed=args.seed)
    inputs, targets, copy_mask = ar_inputs_targets(data)
    x = torch.from_numpy(inputs).long()
    y = torch.from_numpy(targets).long()
    m = torch.from_numpy(copy_mask)

    base = copy_loss(model, x, y, m, device, args.batch)
    ablation_rows = []
    for layer, head in heads:
        reset_head_masks(model)
        set_head_masks(model, [(layer, head)], 0)
        loss = copy_loss(model, x, y, m, device, args.batch)
        ablation_rows.append({"condition": f"ablate L{layer}H{head}", "n_heads": 1, "loss": loss, "delta": loss - base})
    reset_head_masks(model)
    set_head_masks(model, heads[:3], 0)
    loss_top3 = copy_loss(model, x, y, m, device, args.batch)
    ablation_rows.append({"condition": "ablate top3", "n_heads": 3, "loss": loss_top3, "delta": loss_top3 - base})
    reset_head_masks(model)
    set_head_masks(model, heads, 0)
    loss_all = copy_loss(model, x, y, m, device, args.batch)
    ablation_rows.append({"condition": "ablate nominated", "n_heads": len(heads), "loss": loss_all, "delta": loss_all - base})
    reset_head_masks(model)

    conflict_tokens, distal, local, score_mask = single_conflict_batch(data, seed=args.seed + 1)
    conflict = conflict_shares(model, conflict_tokens, distal, local, score_mask, device, args.batch)
    scrambled = prefix_scramble(conflict_tokens, data, score_mask, seed=args.seed + 2)
    conflict_scrambled = conflict_shares(model, scrambled, distal, local, score_mask, device, args.batch)
    attn_intact = mean_attention_to_source(model, conflict_tokens, data, score_mask, heads, device)
    attn_scrambled = mean_attention_to_source(model, scrambled, data, score_mask, heads, device)

    fig_data = args.out_dir / "fig_data"
    fig_data.mkdir(parents=True, exist_ok=True)
    ablation_csv = fig_data / "copy_head_ablation.csv"
    with ablation_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "n_heads", "loss", "delta"])
        writer.writeheader()
        writer.writerows(ablation_rows)
    prefix_csv = fig_data / "copy_prefix_control.csv"
    with prefix_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "distal", "local", "neither"])
        writer.writeheader()
        writer.writerow({"condition": "single conflict", **conflict})
        writer.writerow({"condition": "source prefix scrambled", **conflict_scrambled})
    attn_csv = fig_data / "copy_prefix_attention.csv"
    with attn_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["head", "attn_single_conflict", "attn_prefix_scrambled"])
        writer.writeheader()
        for name in attn_intact:
            writer.writerow({"head": name, "attn_single_conflict": attn_intact[name], "attn_prefix_scrambled": attn_scrambled[name]})

    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    singles = [r for r in ablation_rows if r["n_heads"] == 1]
    axes[0].bar([r["condition"].replace("ablate ", "") for r in singles], [r["delta"] for r in singles], color="#0b5cad")
    axes[0].set_ylabel("delta copy loss")
    axes[0].set_title("Single-head ablations")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar([r["condition"] for r in ablation_rows if r["n_heads"] > 1], [r["delta"] for r in ablation_rows if r["n_heads"] > 1], color="#b03a2e")
    axes[1].set_title("Subset ablations")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig_path = args.out_dir / "fig_copy_heads.png"
    fig.savefig(fig_path)

    trace_md = args.out_dir / "COPY_TRACE.md"
    sum_singles = sum(r["delta"] for r in singles)
    trace_md.write_text(
        "\n".join(
            [
                "# Repeat-Copy Trace Controls",
                "",
                f"Checkpoint: `{args.checkpoint}`. Base copy loss: {base:.4f}.",
                "",
                "## Head Ablations",
                "",
                "| Condition | Heads | Copy Loss | Delta |",
                "|---|---:|---:|---:|",
                *[
                    f"| {r['condition']} | {r['n_heads']} | {r['loss']:.4f} | {r['delta']:+.4f} |"
                    for r in ablation_rows
                ],
                "",
                f"Sum of single-head deltas: {sum_singles:.4f}.",
                f"Top3 subset delta: {loss_top3 - base:.4f}. Nominated-head subset delta: {loss_all - base:.4f}.",
                "Subset cost is far below the sum of singles if the heads are partially redundant/non-independent.",
                "",
                "## Prefix-Scramble Control",
                "",
                "| Condition | Distal | Local | Neither |",
                "|---|---:|---:|---:|",
                f"| single conflict | {conflict['distal']:.4f} | {conflict['local']:.4f} | {conflict['neither']:.4f} |",
                f"| source prefix scrambled | {conflict_scrambled['distal']:.4f} | {conflict_scrambled['local']:.4f} | {conflict_scrambled['neither']:.4f} |",
                "",
                "Scrambling the source prefix while preserving the local second-copy prefix tests whether the distal source is selected by a prefix match.",
                "",
                f"Raw data: `{ablation_csv}`, `{prefix_csv}`, `{attn_csv}`. Figure: `{fig_path}`.",
                "",
            ]
        )
    )
    print(trace_md.read_text())


if __name__ == "__main__":
    main()
