"""Trace screens for the repeat-copy circuit.

This is deliberately a screen, not a final attribution graph. It asks two causal questions first:
which heads attend to the copied source positions, and which heads increase copy loss when ablated?

    python scripts/trace_repeat_copy.py --checkpoint checkpoints/repeat_copy_s0_step8000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.repeat_copy import ar_inputs_targets, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig


@torch.no_grad()
def copy_loss(model: GFMNano, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, device: str, batch: int) -> float:
    total = n = 0.0
    for i in range(0, len(x), batch):
        xb, yb, mb = x[i : i + batch].to(device), y[i : i + batch].to(device), mask[i : i + batch].to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits[mb], yb[mb], reduction="sum")
        total += float(loss)
        n += float(mb.sum())
    return total / n


@torch.no_grad()
def attention_to_sources(model: GFMNano, x: torch.Tensor, mask: np.ndarray, source: np.ndarray, device: str) -> list[dict]:
    _, cache = model.run_with_cache(x.to(device))
    rows, pred_pos = np.nonzero(mask)
    out = []
    for layer in range(model.cfg.n_layers):
        pattern = cache[f"L{layer}.attn.pattern"].cpu().numpy()
        for head in range(model.cfg.n_heads):
            to_cont = []
            to_key = []
            for r, p in zip(rows, pred_pos):
                src = int(source[r, p])
                to_cont.append(pattern[r, head, p, src])
                if src > 0:
                    to_key.append(pattern[r, head, p, src - 1])
            out.append(
                {
                    "head": f"L{layer}H{head}",
                    "attn_to_source_continuation": float(np.mean(to_cont)),
                    "attn_to_source_key": float(np.mean(to_key)),
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seqs", type=int, default=512)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=5151)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    data = make_batch(args.seqs, window=args.window, seed=args.seed)
    inputs, targets, copy_mask = ar_inputs_targets(data)
    x = torch.from_numpy(inputs).long()
    y = torch.from_numpy(targets).long()
    m = torch.from_numpy(copy_mask)

    base = copy_loss(model, x, y, m, device, args.batch)
    attn = attention_to_sources(model, x[: min(args.seqs, 128)], copy_mask[: min(args.seqs, 128)], data.source_index[: min(args.seqs, 128)], device)
    by_head = {row["head"]: row for row in attn}

    rows = []
    for layer, block in enumerate(model.blocks):
        original = block.attn.head_mask.clone()
        for head in range(model.cfg.n_heads):
            block.attn.head_mask[:] = original
            block.attn.head_mask[head] = 0
            loss = copy_loss(model, x, y, m, device, args.batch)
            name = f"L{layer}H{head}"
            rows.append(
                {
                    "head": name,
                    "delta_copy_loss": loss - base,
                    **by_head.get(name, {"attn_to_source_continuation": float("nan"), "attn_to_source_key": float("nan")}),
                }
            )
        block.attn.head_mask[:] = original

    print(f"{args.checkpoint.name}   step {blob['step']}   seed {blob.get('seed', '?')}")
    print(f"base copy loss: {base:.4f}\n")

    print("top heads by ablation delta")
    print(f"{'head':<8} {'d_copy_loss':>12} {'attn->src':>12} {'attn->key':>12}")
    for row in sorted(rows, key=lambda r: r["delta_copy_loss"], reverse=True)[: args.top]:
        print(
            f"{row['head']:<8} {row['delta_copy_loss']:>12.6f} "
            f"{row['attn_to_source_continuation']:>12.6f} {row['attn_to_source_key']:>12.6f}"
        )

    print("\ntop heads by attention to copied source continuation")
    print(f"{'head':<8} {'attn->src':>12} {'d_copy_loss':>12}")
    for row in sorted(rows, key=lambda r: r["attn_to_source_continuation"], reverse=True)[: args.top]:
        print(f"{row['head']:<8} {row['attn_to_source_continuation']:>12.6f} {row['delta_copy_loss']:>12.6f}")


if __name__ == "__main__":
    main()
