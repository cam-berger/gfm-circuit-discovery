"""Integrated-gradient QK edge check for repeat-copy heads.

For copy query positions, measure how much QK attribution in a layer lands on the edge to the copied
source continuation versus simple controls. This is the repeat-copy analogue of the marker-head
QK-IG check: it asks whether the decide-where-to-look circuit is concentrated on the source-copy edge.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from gfmcd.data.repeat_copy import ar_inputs_targets, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig


def forward_with_scores(model: GFMNano, tokens: torch.Tensor, target_layer: int, supplied_scores: torch.Tensor) -> torch.Tensor:
    cfg = model.cfg
    n = tokens.shape[-1]
    resid = model.embed(tokens) + model.pos_embed(torch.arange(n, device=tokens.device))
    causal_mask = torch.ones(n, n, dtype=torch.bool, device=tokens.device).tril()

    for layer, block in enumerate(model.blocks):
        x = block.ln1(resid)
        b = x.shape[0]
        h, dh = cfg.n_heads, cfg.d_head
        v = block.attn.W_V(x).view(b, n, h, dh)

        if layer == target_layer:
            scores = supplied_scores
        else:
            q = block.attn.W_Q(x).view(b, n, h, dh)
            k = block.attn.W_K(x).view(b, n, h, dh)
            scores = torch.einsum("bphc,bqhc->bhpq", q, k) / dh**0.5
            if cfg.causal:
                scores = scores.masked_fill(~causal_mask, float("-inf"))

        pattern = scores.softmax(-1)
        z = torch.einsum("bhpq,bqhc->bphc", pattern, v)
        resid = resid + block.attn.W_O(z.reshape(b, n, -1))
        resid = resid + block.mlp(block.ln2(resid), {}, layer)

    return model.unembed(model.ln_f(resid))


def copy_metric(logits: torch.Tensor, targets: torch.Tensor, copy_mask: torch.Tensor) -> torch.Tensor:
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[copy_mask].sum()


def ig_scores(model: GFMNano, tokens: torch.Tensor, targets: torch.Tensor, copy_mask: torch.Tensor, layer: int, steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    real = cache[f"L{layer}.attn.scores"].detach()
    finite = torch.isfinite(real)
    baseline = torch.zeros_like(real).masked_fill(~finite, float("-inf"))
    content = torch.where(finite, real, torch.zeros_like(real))

    total = torch.zeros_like(real).masked_fill(~finite, 0.0)
    for i in range(steps):
        alpha = (i + 0.5) / steps
        interp = torch.where(finite, alpha * content, baseline)
        interp = interp.detach().requires_grad_(True)
        metric = copy_metric(forward_with_scores(model, tokens, layer, interp), targets, copy_mask)
        grad = torch.autograd.grad(metric, interp)[0]
        total += torch.where(finite, grad, torch.zeros_like(grad))
    mean_grad = total / steps
    attr = torch.where(finite, content * mean_grad, torch.zeros_like(content))
    return attr.detach(), cache[f"L{layer}.attn.pattern"].detach(), finite.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/repeat_copy_s0_step8000.pt"))
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7373)
    ap.add_argument("--out-dir", type=Path, default=Path("branches/codex-repeat-induction"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    data = make_batch(args.seqs, window=256, seed=args.seed)
    inputs, targets_np, copy_mask_np = ar_inputs_targets(data)
    tokens = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets_np).long().to(device)
    copy_mask = torch.from_numpy(copy_mask_np).to(device)
    attr, pattern, finite = ig_scores(model, tokens, targets, copy_mask, args.layer, args.steps)

    rows_idx, pred_pos = np.nonzero(copy_mask_np)
    source = data.source_index[rows_idx, pred_pos]
    rng = np.random.default_rng(args.seed + 1)
    # A valid causal control key that is neither the source continuation nor the previous source key.
    controls = []
    for p, src in zip(pred_pos, source):
        choices = [q for q in range(p + 1) if q not in (int(src), int(src) - 1)]
        controls.append(int(rng.choice(choices)))
    controls = np.array(controls)

    rows = []
    for head in range(model.cfg.n_heads):
        a = attr[:, head].abs().cpu().numpy()
        pat = pattern[:, head].cpu().numpy()
        source_attr = a[rows_idx, pred_pos, source].sum()
        prev_attr = a[rows_idx, pred_pos, source - 1].sum()
        ctl_attr = a[rows_idx, pred_pos, controls].sum()
        total_attr = a[rows_idx, pred_pos, :].sum()
        rows.append(
            {
                "head": f"L{args.layer}H{head}",
                "source_edge_share": source_attr / max(total_attr, 1e-12),
                "prev_key_edge_share": prev_attr / max(total_attr, 1e-12),
                "control_edge_share": ctl_attr / max(total_attr, 1e-12),
                "attn_to_source": pat[rows_idx, pred_pos, source].mean(),
                "attn_to_prev_key": pat[rows_idx, pred_pos, source - 1].mean(),
            }
        )

    out_data = args.out_dir / "fig_data"
    out_data.mkdir(parents=True, exist_ok=True)
    csv_path = out_data / "copy_qk_ig.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "head",
                "source_edge_share",
                "prev_key_edge_share",
                "control_edge_share",
                "attn_to_source",
                "attn_to_prev_key",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out_dir / "COPY_QK_IG.md"
    top = sorted(rows, key=lambda r: r["source_edge_share"], reverse=True)
    md_path.write_text(
        "\n".join(
            [
                "# Repeat-Copy QK-IG Edge Check",
                "",
                f"Checkpoint: `{args.checkpoint}`. Layer: L{args.layer}. IG steps: {args.steps}.",
                "",
                "| Head | Source Edge Share | Prev-Key Share | Control Share | Attention To Source |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {r['head']} | {r['source_edge_share']:.4f} | {r['prev_key_edge_share']:.4f} | {r['control_edge_share']:.4f} | {r['attn_to_source']:.4f} |"
                    for r in top
                ],
                "",
                f"Raw data: `{csv_path}`.",
                "",
            ]
        )
    )
    print(md_path.read_text())


if __name__ == "__main__":
    main()
