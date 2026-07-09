"""Trace the anchored counter: which head finds the start codon, and which components matter.

We know the circuit exists (D26) and we know the tracer's error rate on planted ground truth
(D12-D17). This is the first time both are true of a real learned model with LayerNorm.

    python scripts/trace_uniform.py --checkpoint checkpoints/nano_uniform_step12000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import torch.nn.functional as F

from gfmcd.data.genome import IGNORE
from gfmcd.data.uniform_cds import ATG, N_AA, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig


def fit_probe(x, y, steps: int = 400):
    torch.manual_seed(0)
    probe = torch.nn.Linear(x.shape[-1], 3)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe((x - mu) / sd), y).backward()
        opt.step()
    return probe, mu, sd


@torch.no_grad()
def phase_accuracy(model, tok, phase, probe, mu, sd, layer, device) -> float:
    """Probe fit on the intact model, applied unchanged. Measures the REPRESENTATION, not the output."""
    t = torch.from_numpy(tok).long()
    m = torch.from_numpy(phase >= 0)
    y = torch.from_numpy(phase[phase >= 0]).long()
    xs = []
    for i in range(0, len(t), 64):
        _, cache = model.run_with_cache(t[i : i + 64].to(device))
        xs.append(cache[layer][m[i : i + 64]].cpu())
    x = torch.cat(xs)
    return float((probe((x - mu) / sd).argmax(-1) == y).float().mean())


@torch.no_grad()
def aa_accuracy(model, tok: np.ndarray, aa: np.ndarray, device: str) -> float:
    t, y = torch.from_numpy(tok).long(), torch.from_numpy(aa)
    correct = total = 0.0
    for i in range(0, len(t), 64):
        lg = model(t[i : i + 64].to(device)).cpu()
        m = y[i : i + 64] != IGNORE
        correct += float((lg.argmax(-1)[m] == y[i : i + 64][m]).sum())
        total += float(m.sum())
    return correct / total


@torch.no_grad()
def attention_to(model, tok: np.ndarray, targets: np.ndarray, device: str, min_dist: int = 10):
    """Mean attention paid by downstream positions to each target column, per (layer, head)."""
    cfg = model.cfg
    acc = np.zeros((cfg.n_layers, cfg.n_heads))
    n_seen = 0
    t = torch.from_numpy(tok).long()

    for i in range(0, len(t), 32):
        chunk = t[i : i + 32].to(device)
        _, cache = model.run_with_cache(chunk)
        tgt = targets[i : i + 32]
        for b in range(len(chunk)):
            if tgt[b] < 0:
                continue
            lo = tgt[b] + min_dist
            if lo >= chunk.shape[1] - 1:
                continue
            for l in range(cfg.n_layers):
                pat = cache[f"L{l}.attn.pattern"][b, :, lo:, tgt[b]]
                acc[l] += pat.mean(dim=-1).cpu().numpy()
            n_seen += 1
    return acc / max(n_seen, 1)


def first_and_decoy_atgs(tok: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.argmax(phase == 0, axis=1)
    decoy = np.full(len(tok), -1, dtype=int)
    for r in range(len(tok)):
        for p in range(start[r] + 6, tok.shape[1] - 3):
            if tuple(tok[r, p : p + 3]) == tuple(ATG):
                decoy[r] = p
                break
    return start, decoy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seqs", type=int, default=384)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = NanoConfig(**blob["cfg"])
    model = GFMNano(cfg).to(device).eval()
    model.load_state_dict(blob["state"])

    tok, aa, phase = make_batch(args.seqs, cfg.max_len, seed=555)
    start, decoy = first_and_decoy_atgs(tok, phase)
    have_decoy = decoy >= 0

    base = aa_accuracy(model, tok, aa, device)
    print(f"{args.checkpoint.name}  intact amino-acid accuracy {base:.4f}\n")

    a_start = attention_to(model, tok, start, device)
    a_decoy = attention_to(model, tok[have_decoy], decoy[have_decoy], device)

    print("mean attention from downstream positions to the START codon (and to a DECOY ATG)")
    print(f"{'':>8}" + "".join(f"{'h'+str(h):>16}" for h in range(cfg.n_heads)))
    for l in range(cfg.n_layers):
        row = "".join(f"{a_start[l,h]:>8.3f}/{a_decoy[l,h]:<7.3f}" for h in range(cfg.n_heads))
        print(f"L{l:<7}" + row)

    flat = a_start.reshape(-1)
    order = np.argsort(-flat)[:4]
    print("\ntop heads by attention to the start codon:")
    for k in order:
        l, h = divmod(int(k), cfg.n_heads)
        print(f"  L{l}H{h}: start {a_start[l,h]:.3f}   decoy {a_decoy[l,h]:.3f}   ratio {a_start[l,h]/max(a_decoy[l,h],1e-6):.1f}x")

    # Fit the phase probe once on the intact model, at the layer where phase saturates.
    layer = "L2.resid_post"
    t = torch.from_numpy(tok).long()
    m = torch.from_numpy(phase >= 0)
    xs = []
    with torch.no_grad():
        for i in range(0, len(t), 64):
            _, cache = model.run_with_cache(t[i : i + 64].to(device))
            xs.append(cache[layer][m[i : i + 64]].cpu())
    x_fit = torch.cat(xs)
    y_fit = torch.from_numpy(phase[phase >= 0]).long()
    probe, mu, sd = fit_probe(x_fit, y_fit)
    base_phase = phase_accuracy(model, tok, phase, probe, mu, sd, layer, device)

    print(f"\nhead ablation. aa chance-with-guessing 0.3768; phase chance 0.3333")
    print(f"intact: aa {base:.4f}   phase({layer}) {base_phase:.4f}\n")
    print(f"{'head':<8}{'aa acc':>9}{'aa drop':>10}{'phase acc':>12}{'phase drop':>12}{'attn->start':>13}")

    results = []
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            model.blocks[l].attn.head_mask[h] = 0.0
            acc = aa_accuracy(model, tok, aa, device)
            ph = phase_accuracy(model, tok, phase, probe, mu, sd, layer, device)
            model.blocks[l].attn.head_mask[h] = 1.0
            results.append((base_phase - ph, base - acc, l, h, acc, ph))

    results.sort(reverse=True)
    for pdrop, adrop, l, h, acc, ph in results[:6]:
        print(f"L{l}H{h:<5}{acc:>9.4f}{adrop:>+10.4f}{ph:>12.4f}{pdrop:>+12.4f}{a_start[l,h]:>13.3f}")
    print("  ...")
    for pdrop, adrop, l, h, acc, ph in results[-3:]:
        print(f"L{l}H{h:<5}{acc:>9.4f}{adrop:>+10.4f}{ph:>12.4f}{pdrop:>+12.4f}{a_start[l,h]:>13.3f}")

    print("\nheads that destroy the AMINO ACID but not the PHASE = local triplet readers, not anchors")
    for pdrop, adrop, l, h, acc, ph in sorted(results, key=lambda r: -r[1])[:4]:
        role = "ANCHOR" if pdrop > 0.2 else "triplet reader"
        print(f"  L{l}H{h}: aa drop {adrop:+.4f}  phase drop {pdrop:+.4f}  -> {role}")


if __name__ == "__main__":
    main()
