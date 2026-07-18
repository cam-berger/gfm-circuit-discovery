"""A0 -- stage-wise provenance of L3H1's acceptor targeting (donor-graph experiment 4).

Reusable kernel for ``PREREGISTRATION_A0_TARGETING_PROVENANCE.md``.  It answers *where in the
residual supplied to L3H1 the donor-query preference for the intron-mate acceptor becomes
sufficient*, and whether that sufficiency rides the key side, the query side, or a Q x K
interaction -- by reconstructing L3's query/key at a set of tracked positions from a hybrid
residual whose stages after ``s`` are clamped to a mean-resample reference.

Design rules inherited from the study (do not relax without a prereg amendment):

* SpliceBERT is **post-LayerNorm**, so the residual is not an additive sum of writes; the stage
  reconstruction is by INTERVENTION (``clamp_write`` on block dense outputs + an embedding-output
  clamp), never a frozen-LayerNorm linear decomposition.  The model's own forward (every LN,
  attention mixing) runs; we only override component writes at the tracked positions.
* Every LayerNorm is positionwise, so reconstructing ONLY ``{donor, a+, a(other)}`` is exact for
  L3's Q/K at those positions (preflight memorial ``positionwise_reconstruction``).
* The reference is a mean-resample over the eligible set at the matched positional role (D-A0-1);
  the CORRUPT run is never the reference (it perturbs the a+ token at the embedding stage).
* The three curves ``S_K/S_Q/S_J`` are pre-softmax QK logits: mechanistic intermediates, never a
  behavioural partner-selection claim (the natural matched-decoy behavioural task was withdrawn).
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator

import numpy as np
import torch

from gfmcd.splicing.address import qk_logit_to_window
from gfmcd.splicing.circuit import READER, _block_write, capture_proj, clamp_write

L3, H3 = READER

# The 7 residual components feeding L3's input, in depth order.  Stage ``s`` (1..7) = clean
# through ``COMPONENTS[s-1]``, reference after; stage 0 = all reference; stage 7 = true L3 input.
COMPONENTS: tuple = (
    "embed", (0, "attn"), (0, "mlp"), (1, "attn"), (1, "mlp"), (2, "attn"), (2, "mlp"),
)
N_STAGES = len(COMPONENTS)


def clamp_after(stage: int) -> tuple:
    """Components to force to reference at reconstruction ``stage`` (those strictly after it)."""
    return tuple(COMPONENTS[stage:])          # stage 0 -> all; stage 7 -> none


# --------------------------------------------------------------------------------------
# Embedding-output clamp -- the only new primitive (mirrors circuit.clamp_write for blocks).
# --------------------------------------------------------------------------------------
def _walk(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def embed_module(ad):
    """The BERT embeddings module whose OUTPUT is the post-LayerNorm token+position residual."""
    for path in ("bert.embeddings", "base_model.embeddings", "base_model.bert.embeddings",
                 "splicebert.embeddings", "encoder.embeddings", "embeddings"):
        m = _walk(ad.model, path)
        if m is not None and hasattr(m, "register_forward_hook"):
            return m
    raise AttributeError("could not locate a hookable embeddings module")


@contextmanager
def clamp_embed(ad, source: torch.Tensor, positions: torch.Tensor) -> Iterator[None]:
    """Force the embedding output to ``source`` at ``positions`` ([B] or [B,P])."""
    mod = embed_module(ad)

    def hook(_m, _i, out):
        src = source.to(device=out.device, dtype=out.dtype)
        out = out.clone()
        pos = positions if positions.dim() == 2 else positions[:, None]
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(pos)
        out[b, pos] = src[b, pos]
        return out

    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


# --------------------------------------------------------------------------------------
# Token-vs-position 2x2 (A1a): clamp the word/position embedding COMPONENTS separately, so a
# shared reference can neutralise one dimension's a+/decoy discrimination while the other stays
# clean (prereg A1_TOKEN_VS_POSITION).  BERT: embeddings = LayerNorm(word + position + type).
# --------------------------------------------------------------------------------------
_EMBED_COMPONENTS = {"word": "word_embeddings", "position": "position_embeddings"}


def embed_component(ad, which: str):
    """The hookable ``word_embeddings`` / ``position_embeddings`` sub-module."""
    return getattr(embed_module(ad), _EMBED_COMPONENTS[which])


def _bcast_gather(tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Gather ``tensor`` [B or 1, S, H] at ``pos`` [B,K] -> [B,K,H] (position_embeddings may be
    broadcast [1,S,H] over the batch)."""
    if tensor.shape[0] == 1 and pos.shape[0] > 1:
        tensor = tensor.expand(pos.shape[0], -1, -1)
    b = torch.arange(tensor.shape[0], device=tensor.device)[:, None].expand_as(pos)
    return tensor[b, pos]


@contextmanager
def clamp_embed_component(ad, which: str, source: torch.Tensor,
                          positions: torch.Tensor) -> Iterator[None]:
    """Force the word/position embedding output to ``source`` at ``positions`` ([B,P])."""
    mod = embed_component(ad, which)

    def hook(_m, _i, out):
        src = source.to(device=out.device, dtype=out.dtype)
        if out.shape[0] == 1 and positions.shape[0] > 1:     # position_embeddings broadcast
            out = out.expand(positions.shape[0], -1, -1)
        out = out.clone()
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(positions)
        out[b, positions] = src[b, positions]
        return out

    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def capture_embed_components(ad, run) -> dict:
    """Cache the word and position embedding outputs from one forward pass ({which: [.,S,H]})."""
    store: dict = {}
    handles = []
    for which in ("word", "position"):
        handles.append(embed_component(ad, which).register_forward_hook(
            lambda _m, _i, o, w=which: store.__setitem__(w, o.detach().clone())))
    try:
        run()
    finally:
        for h in handles:
            h.remove()
    return store


def sa_difference(query, key, donor_tok, wplus, wother, Dh) -> torch.Tensor:
    """S_A = mean-window (q_donor . k_a+) - (q_donor . k_other), per example [B] (tensor)."""
    return (qk_logit_to_window(query, key, H3, Dh, donor_tok, wplus)
            - qk_logit_to_window(query, key, H3, Dh, donor_tok, wother))


def shared_window_reference(refc: torch.Tensor, wplus: torch.Tensor, wother: torch.Tensor,
                            seq_len: int, device) -> torch.Tensor:
    """[B,S,H] with a per-offset shared reference ``refc`` [K,H] written at BOTH the a+ and decoy
    windows -- so the two windows become identical at matched offsets (the A1a neutralisation)."""
    B, K = wplus.shape
    H = refc.shape[1]
    t = torch.zeros(B, seq_len, H, device=device, dtype=refc.dtype)
    b = torch.arange(B, device=device)[:, None].expand(B, K)
    t[b, wplus] = refc[None].expand(B, K, H).to(t.dtype)
    t[b, wother] = refc[None].expand(B, K, H).to(t.dtype)
    return t


def capture_all(ad, run) -> dict:
    """Cache the embedding output and the layer 0-2 block writes from one forward pass.

    Returns ``{"embed": [B,S,H], (l,kind): [B,S,H]}`` keyed exactly like ``COMPONENTS``.
    """
    store: dict = {}
    handles = []

    def make(key):
        def hook(_m, _i, out):
            store[key] = out.detach().clone()
        return hook

    try:
        handles.append(embed_module(ad).register_forward_hook(make("embed")))
        for c in COMPONENTS:
            if c == "embed":
                continue
            layer, kind = c
            handles.append(_block_write(ad, layer, kind).register_forward_hook(make(c)))
        run()
    finally:
        for h in handles:
            h.remove()
    return store


# --------------------------------------------------------------------------------------
# Reconstruction: L3 query/key from a hybrid residual (stages after ``clamp`` -> reference).
# --------------------------------------------------------------------------------------
def reconstruct_qk(ad, enc, tracked_pos: torch.Tensor, clamp: tuple, ref: dict):
    """Return L3's (query, key) projection outputs under a stage-``s`` reconstruction.

    ``clamp`` is the tuple of components to hold at ``ref`` (from ``clamp_after(stage)``);
    ``ref`` maps each component to a ``[B,S,H]`` tensor whose values at ``tracked_pos`` are the
    mean-resample reference.  With ``clamp`` empty this is exactly the clean forward.
    """
    blocks = [c for c in clamp if c != "embed"]
    with ExitStack() as st:
        if "embed" in clamp:
            st.enter_context(clamp_embed(ad, ref["embed"], tracked_pos))
        if blocks:
            src = {c: ref[c] for c in blocks}
            st.enter_context(clamp_write(ad, blocks, src, positions=tracked_pos))
        proj = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc))
    return proj["query"], proj["key"]


# --------------------------------------------------------------------------------------
# Reference construction: mean-resample over the eligible set at matched positional role.
# --------------------------------------------------------------------------------------
def gather_at(tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """``tensor`` [B,S,H], ``pos`` [B,P] -> [B,P,H] (the write at each tracked position)."""
    b = torch.arange(tensor.shape[0], device=tensor.device)[:, None].expand_as(pos)
    return tensor[b, pos]


class RoleMean:
    """Accumulates the per-(component, tracked-column) mean write across all batches (seed-free:
    an exact running mean, order-independent)."""

    def __init__(self, ncol: int, hidden: int, device):
        self.sum = {c: torch.zeros(ncol, hidden, device=device) for c in COMPONENTS}
        self.n = 0

    def update(self, captured: dict, tracked_pos: torch.Tensor) -> None:
        self.n += tracked_pos.shape[0]
        for c in COMPONENTS:
            self.sum[c] += gather_at(captured[c], tracked_pos).sum(0)   # [P,H]

    def means(self) -> dict:
        if self.n == 0:
            raise RuntimeError("RoleMean got no examples")
        return {c: self.sum[c] / self.n for c in COMPONENTS}             # {c: [P,H]}


def scatter_reference(means: dict, tracked_pos: torch.Tensor, seq_len: int, device) -> dict:
    """Build ``{component: [B,S,H]}`` with the global role-mean written at every tracked position.

    Only the tracked positions are ever read by the clamps, so untracked entries stay zero.
    """
    B, P = tracked_pos.shape
    b = torch.arange(B, device=device)[:, None].expand_as(tracked_pos)
    ref: dict = {}
    for c in COMPONENTS:
        H = means[c].shape[1]
        t = torch.zeros(B, seq_len, H, device=device, dtype=means[c].dtype)
        t[b, tracked_pos] = means[c][None, :, :].expand(B, P, H).to(t.dtype)
        ref[c] = t
    return ref


# --------------------------------------------------------------------------------------
# The three targeting curves and the absolute-attraction diagnostic.
# --------------------------------------------------------------------------------------
def _diff(query, key, donor_tok, wplus, wother, Dh):
    """S = mean_win qk(a+) - mean_win qk(a_other), per example [B] (numpy)."""
    p = qk_logit_to_window(query, key, H3, Dh, donor_tok, wplus)
    o = qk_logit_to_window(query, key, H3, Dh, donor_tok, wother)
    return (p - o).cpu().numpy()


def three_curves(q_s, k_s, q_F, k_F, donor_tok, wplus, wother, Dh):
    """Per-example ``S_K``, ``S_Q``, ``S_J`` for one stage (prereg §2).

    ``S_K`` fixes the query at its final value and varies the keys (key provenance);
    ``S_Q`` fixes the keys and varies the query (query provenance);
    ``S_J`` varies both (realized targeting).  Their divergence is the Q x K interaction readout.
    """
    return {
        "S_K": _diff(q_F, k_s, donor_tok, wplus, wother, Dh),
        "S_Q": _diff(q_s, k_F, donor_tok, wplus, wother, Dh),
        "S_J": _diff(q_s, k_s, donor_tok, wplus, wother, Dh),
    }


def window_logits(query, key, donor_tok, win, Dh) -> torch.Tensor:
    """Per-token pre-softmax logits from the donor query to a key window, [B,K] (un-meaned)."""
    b = torch.arange(query.shape[0], device=query.device)
    q = query[b, donor_tok, H3 * Dh:(H3 + 1) * Dh]                       # [B,Dh]
    k = key[:, :, H3 * Dh:(H3 + 1) * Dh]                                 # [B,S,Dh]
    kk = k[b[:, None], win]                                              # [B,K,Dh]
    return (kk * q[:, None, :]).sum(-1) / (Dh ** 0.5)                    # [B,K]


def attraction(query, key, donor_tok, wplus, wother, wbg, Dh) -> dict:
    """Absolute window attraction vs a matched non-site background, per example (prereg §2).

    ``LSE_win(a) - LSE_bg`` catches a query effect that lifts the whole acceptor class relative to
    background while barely moving ``a+ - a_other`` (which the S-difference metrics would miss).
    """
    def lse(win):
        return torch.logsumexp(window_logits(query, key, donor_tok, win, Dh), dim=-1)
    bg = lse(wbg)
    return {
        "a_plus_vs_bg": (lse(wplus) - bg).cpu().numpy(),
        "a_other_vs_bg": (lse(wother) - bg).cpu().numpy(),
    }
