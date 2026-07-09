"""Does the QK instrument tell the causal marker head from the inert one?

The forcing-task model contains, spontaneously, the control we hand-built in Rung 0:

    L1H3   attends the start codon 230x over decoys.  Ablating it: 1.0000 -> 0.3875
    L1H2   attends the start codon 406x over decoys.  Ablating it: 1.0000 -> 1.0000

Attention inspection cannot separate them -- L1H2 looks *more* like a marker head. This is exactly
the failure mode `build_bag_with_unread_marker_head` was written to catch (D20).

Naive pattern-gradient attribution vanishes on a confident head (measured 3.7e-15 on the planted
twin), so we integrate the gradient from a uniform-attention baseline to the real scores and report,
per head, the total content-driven contribution to the amino-acid readout.

    python scripts/qk_ig_nano.py --checkpoint checkpoints/nano_uniform_step12000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import IGNORE
from gfmcd.data.uniform_cds import N_AA, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

IG_STEPS = 16


def forward_with_scores(model: GFMNano, tokens: torch.Tensor, target_layer: int, scores: torch.Tensor):
    """Re-run the model with `target_layer`'s attention scores supplied."""
    cfg = model.cfg
    n = tokens.shape[-1]
    resid = model.embed(tokens) + model.pos_embed(torch.arange(n, device=tokens.device))

    for l, block in enumerate(model.blocks):
        x = block.ln1(resid)
        b = x.shape[0]
        h, dh = cfg.n_heads, cfg.d_head
        v = block.attn.W_V(x).view(b, n, h, dh)

        if l == target_layer:
            pattern = scores.softmax(-1)
        else:
            q = block.attn.W_Q(x).view(b, n, h, dh)
            k = block.attn.W_K(x).view(b, n, h, dh)
            s = torch.einsum("bphc,bqhc->bhpq", q, k) / dh**0.5
            if cfg.causal:
                s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=x.device).tril(), float("-inf"))
            pattern = s.softmax(-1)

        z = torch.einsum("bhpq,bqhc->bphc", pattern, v)
        resid = resid + block.attn.W_O(z.reshape(b, n, -1))
        resid = resid + block.mlp(block.ln2(resid), {}, l)

    return model.unembed(model.ln_f(resid))


def metric(logits: torch.Tensor, aa: torch.Tensor) -> torch.Tensor:
    m = aa != IGNORE
    return logits.gather(-1, aa.clamp(min=0).unsqueeze(-1)).squeeze(-1)[m].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--layer", type=int, default=1)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = NanoConfig(**blob["cfg"])
    model = GFMNano(cfg).to(device).eval()
    model.load_state_dict(blob["state"])

    tok_np, aa_np, ph_np = make_batch(args.seqs, cfg.max_len, seed=555)
    tokens = torch.from_numpy(tok_np).long().to(device)
    aa = torch.from_numpy(aa_np).to(device)
    start = np.argmax(ph_np == 0, axis=1)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    scores = cache[f"L{args.layer}.attn.scores"].detach()

    # Naive gradient at the real scores.
    s = scores.clone().requires_grad_(True)
    naive = torch.autograd.grad(metric(forward_with_scores(model, tokens, args.layer, s), aa), s)[0]

    # Integrated gradient from uniform attention (all scores zero) to the real scores.
    total = torch.zeros_like(scores)
    for i in range(IG_STEPS):
        alpha = (i + 0.5) / IG_STEPS
        s = (alpha * scores).detach().requires_grad_(True)
        total += torch.autograd.grad(metric(forward_with_scores(model, tokens, args.layer, s), aa), s)[0]
    mean_grad = total / IG_STEPS

    attr = (scores * mean_grad).sum(dim=(0, 2, 3))  # per head
    naive_attr = (scores * naive).sum(dim=(0, 2, 3))

    pat = cache[f"L{args.layer}.attn.pattern"]
    print(f"{args.checkpoint.name}   layer {args.layer}\n")
    print(f"{'head':<7}{'attn->start':>13}{'naive grad attr':>18}{'IG attr':>14}{'IG share':>11}")

    ig = attr.abs()
    for h in range(cfg.n_heads):
        with torch.no_grad():
            a = np.mean([float(pat[b, h, start[b] + 15 :, start[b] + 1].mean()) for b in range(len(tokens))])
        print(f"L{args.layer}H{h:<4}{a:>13.4f}{float(naive_attr[h]):>18.3e}{float(attr[h]):>14.2f}{float(ig[h]/ig.sum()):>10.1%}")

    print(f"\nnaive gradient, max |attr| over heads: {float(naive_attr.abs().max()):.3e}")
    print("(a confident head saturates its softmax; the gradient at the real scores is annihilated)")


if __name__ == "__main__":
    main()
