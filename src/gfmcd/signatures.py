"""The 2-D mechanism signature, measured the same way for every model.

Both axes are **probe-based and objective-independent**, so a nucleotide-head LM (μ = 0) and an
amino-acid-head model (μ = 1) land on the same plot. A far-field Δloss cannot do that: the two heads
have different label spaces.

    x   far-field Δ(phase-probe accuracy) under start-codon ablation, minus the decoy-Met control
    y   far-field anchor share under cue conflict

Both were validated on the planted twins before being pointed at a trained model (D3, D24). Neither is
the training accuracy, which saturates (D27, D31). Δloss is reported alongside for the aa-head models,
per the D27 house rule, but it is not an axis.

Three mechanisms occupy distinct regions (D32/D33):

    bag       x -> 0      y -> 0      the anchor is not used at range
    relay     x >  0      y -> 0      anchor initialises; local structure propagates
    counter   x >> 0      y -> 1      counted from the anchor at every distance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import IGNORE
from gfmcd.data.uniform_cds import ATG

NON_START = np.array([1, 3, 1], dtype=np.int8)  # CTC
FAR = 64

# LAYER was chosen when the only trained model we had was the counter, which carries the anchor at L2.
# On real CDS the anchor does not enter until L4, so BOTH axes of the 2x2 -- read here -- reported the
# wrong mechanism for exactly the cells that produced a new claim (D34 on x, D35 on y). Nothing that
# names a mechanism may be read at a fixed layer. Use `peak_anchor_dependence` for the x-axis and
# `scripts/phase_c_conflict_layers.py`'s output-level readout for the y-axis.
LAYER = "L2.resid_post"


@dataclass(frozen=True)
class Signature:
    name: str
    ablation_effect: float      # x -- superseded by peak_anchor_dependence (D34)
    anchor_share_l2probe: float  # NOT the anchor share. A probe at LAYER. See D35 before using.
    content_share_l2probe: float
    ablation_raw: float
    control_raw: float
    n_far: int


def fit_probe(x: torch.Tensor, y: torch.Tensor, steps: int = 300):
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
def residuals(model, tokens: np.ndarray, device: str, layer: str = LAYER) -> torch.Tensor:
    t = torch.from_numpy(tokens).long()
    return torch.cat([model.run_with_cache(t[i : i + 64].to(device))[1][layer].cpu() for i in range(0, len(t), 64)])


@torch.no_grad()
def all_residuals(model, tokens: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    """Every layer from ONE forward pass. Re-running the model per layer is what made this slow."""
    names = ["resid.embed"] + [f"L{i}.resid_post" for i in range(model.cfg.n_layers)]
    chunks: dict[str, list] = {n: [] for n in names}
    t = torch.from_numpy(tokens).long()
    for i in range(0, len(t), 64):
        _, cache = model.run_with_cache(t[i : i + 64].to(device))
        for n in names:
            chunks[n].append(cache[n].cpu())
    return {n: torch.cat(v) for n, v in chunks.items()}


@torch.no_grad()
def probe_accuracy(probe, mu, sd, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((probe((x - mu) / sd).argmax(-1) == y).float().mean())


def start_index(phase: np.ndarray) -> np.ndarray:
    return np.argmax(phase == 0, axis=1)


def downstream_met(tokens: np.ndarray, start: np.ndarray, gap: int = 30) -> np.ndarray:
    """An in-frame ATG well past the start: an ordinary Met, not the anchor."""
    out = np.full(len(tokens), -1, dtype=int)
    for r in range(len(tokens)):
        for p in range(start[r] + gap, tokens.shape[1] - 3, 3):
            if tuple(tokens[r, p : p + 3]) == tuple(ATG):
                out[r] = p
                break
    return out


def ablate(tokens: np.ndarray, at: np.ndarray) -> np.ndarray:
    out = tokens.copy()
    for r, i in enumerate(at):
        if i >= 0:
            out[r, i : i + 3] = NON_START
    return out


def frameshift(tokens: np.ndarray, at: np.ndarray, seed: int = 0) -> np.ndarray:
    """Insert one nucleotide at `at`, shifting the rest right and dropping the final base."""
    rng = np.random.default_rng(seed)
    out = tokens.copy()
    for r, i in enumerate(at):
        out[r, i + 1 :] = tokens[r, i:-1]
        out[r, i] = rng.integers(0, 4)
    return out


def anchor_content_by_distance(model, tokens, phase, start, device, insert_at=None, bins=None,
                               cached=None):
    """Cue conflict. Returns [(bin_centre, anchor_share, content_share, n)], measured from the insertion.

    `cached` = (r_intact_full, r_shift_full) at LAYER, so the model runs twice, not once per bin.
    """
    bins = bins or [(4, 15), (16, 31), (32, 63), (64, 127), (128, 255)]
    ins = start + 3 if insert_at is None else insert_at
    mask = phase >= 0

    if cached is None:
        r_int = residuals(model, tokens, device)
        r_sh = residuals(model, frameshift(tokens, ins), device)
    else:
        r_int, r_sh = cached

    probe, mu, sd = fit_probe(r_int[torch.from_numpy(mask)], torch.from_numpy(phase[mask]).long())

    pos = np.tile(np.arange(tokens.shape[1]), (len(tokens), 1))
    d_ins = pos - ins[:, None]
    d_atg = pos - start[:, None]

    out = []
    for lo, hi in bins:
        sel = mask & (d_ins >= lo) & (d_ins <= hi)
        if sel.sum() < 200:
            continue
        pred = probe((r_sh[torch.from_numpy(sel)] - mu) / sd).argmax(-1).numpy()
        da = d_atg[sel]
        out.append(((lo + hi) / 2, float((pred == da % 3).mean()), float((pred == (da - 1) % 3).mean()), int(sel.sum())))
    return out, (probe, mu, sd)


def signature(model, tokens, phase, device, name: str) -> tuple[Signature, dict, dict]:
    """Four forward passes total: intact, ATG-ablated, decoy-Met-ablated, frameshifted."""
    start = start_index(phase)
    mask = phase >= 0
    met = downstream_met(tokens, start)
    have = met >= 0

    r_int = residuals(model, tokens, device)
    r_abl = residuals(model, ablate(tokens, start), device)
    r_ctl = residuals(model, ablate(tokens, np.where(have, met, -1)), device)
    r_sh = residuals(model, frameshift(tokens, start + 3), device)

    probe, mu, sd = fit_probe(r_int[torch.from_numpy(mask)], torch.from_numpy(phase[mask]).long())

    pos = np.tile(np.arange(tokens.shape[1]), (len(tokens), 1))
    d_atg = pos - start[:, None]
    far = mask & (d_atg >= FAR)
    far_ctl = far & have[:, None]

    def acc(r, sel):
        return probe_accuracy(probe, mu, sd, r[torch.from_numpy(sel)], torch.from_numpy(phase[sel]).long())

    intact_far, ablated_far = acc(r_int, far), acc(r_abl, far)
    ctl_intact, ctl_ablated = acc(r_int, far_ctl), acc(r_ctl, far_ctl)

    ablation_raw = intact_far - ablated_far
    control_raw = ctl_intact - ctl_ablated

    curve, _ = anchor_content_by_distance(model, tokens, phase, start, device,
                                          bins=[(FAR, 255)], cached=(r_int, r_sh))
    _, anchor, content, n = curve[0]

    sig = Signature(name, ablation_raw - control_raw, anchor, content, ablation_raw, control_raw, n)
    extra = dict(intact_far=intact_far, ablated_far=ablated_far, ctl_intact=ctl_intact, ctl_ablated=ctl_ablated)
    return sig, extra, dict(r_int=r_int, r_sh=r_sh)


def layer_probe(model, tokens, phase, device, steps: int = 200) -> dict[str, float]:
    """Phase decodability per layer, from ONE forward pass. Counter builds across depth; a bag is flat."""
    mask = torch.from_numpy(phase >= 0)
    y = torch.from_numpy(phase[phase >= 0]).long()
    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n) :]

    reps = all_residuals(model, tokens, device)
    out = {}
    for name, r in reps.items():
        rr = r[mask]
        probe, mu, sd = fit_probe(rr[tr], y[tr], steps=steps)
        out[name] = probe_accuracy(probe, mu, sd, rr[te], y[te])
    return out


def peak_anchor_dependence(model, tokens, phase, device, steps: int = 200):
    """Max over depth of [far-field probe drop under ATG ablation] − [same for the decoy-Met control].

    Probing at a FIXED layer measures whichever variable that layer happens to carry. In the relay
    models the pooled estimate saturates the anchor-free ceiling by L2 and the anchor's contribution is
    only added at L4, so a probe at L2 reports ~0 anchor dependence for a model whose output collapses
    when the anchor is removed. Taking the max over depth is layer-agnostic and comparable across
    architectures and objectives.

    Returns (peak, best_layer, per_layer_dict).
    """
    start = start_index(phase)
    met = downstream_met(tokens, start)
    have = met >= 0

    R0 = all_residuals(model, tokens, device)
    R1 = all_residuals(model, ablate(tokens, start), device)
    R2 = all_residuals(model, ablate(tokens, np.where(have, met, -1)), device)

    mask = phase >= 0
    pos = np.tile(np.arange(tokens.shape[1]), (len(tokens), 1))
    far = mask & ((pos - start[:, None]) >= FAR)
    far_ctl = far & have[:, None]

    per_layer = {}
    for name in R0:
        probe, mu, sd = fit_probe(R0[name][torch.from_numpy(mask)], torch.from_numpy(phase[mask]).long(), steps=steps)

        def acc(R, sel):
            return probe_accuracy(probe, mu, sd, R[name][torch.from_numpy(sel)], torch.from_numpy(phase[sel]).long())

        drop = acc(R0, far) - acc(R1, far)
        ctl = acc(R0, far_ctl) - acc(R2, far_ctl)
        per_layer[name] = dict(intact=acc(R0, far), ablated=acc(R1, far), drop=drop, control=ctl, effect=drop - ctl)

    best = max(per_layer, key=lambda k: per_layer[k]["effect"])
    return per_layer[best]["effect"], best, per_layer
