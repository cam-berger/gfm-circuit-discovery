"""Two transformers with identical behaviour and different mechanism.

The compositional twin attends to the start marker and computes (i - start - 1) % 3
exactly. The heuristic twin reads a per-nucleotide lookup table and never represents
frame at all. Both are hand-weighted, so the ground-truth edge set is known rather
than inferred, and `PLANTED_EDGES` is what the tracer is scored against.

Behaviour is matched by gating the compositional twin's readout on a fixed, position-
indexed mask: on gated positions it emits a uniform distribution. Solving
  p * 1 + (1 - p) * (1/3) = bayes_unigram_accuracy()
for p makes the two twins equally accurate while leaving their computations untouched.

Residual layout (d_model = 16):
    0:3    one-hot(i % 3)                from pos_embed
    3:6    one-hot(start % 3)            written by the layer-0 attention head
    6:9    one-hot(phase)                written by the layer-1 MLP
    9:14   one-hot(token)                from embed
    14     unused
    15     constant 1                    query source for the marker-finding head
"""

from __future__ import annotations

import torch

from gfmcd.data.synthetic import M, bayes_unigram_accuracy, posterior_phase_given_nt
from gfmcd.models.tiny import TinyConfig, TinyTransformer

POS_PHASE = slice(0, 3)
START_PHASE = slice(3, 6)
PHASE_FEAT = slice(6, 9)
TOK = slice(9, 14)
CONST = 15

MARKER_SCORE = 6.0
LOGIT_SCALE = 8.0

# Edges carried by the OV path. Frozen-attention attribution can see these.
PLANTED_EDGES_OV: frozenset[tuple[str, str]] = frozenset(
    {
        ("pos_embed.phase", "L0.attn.z"),
        ("L0.attn.z", "L1.mlp.acts"),
        ("pos_embed.phase", "L1.mlp.acts"),
        ("L1.mlp.acts", "logits"),
    }
)

# The marker-finding edge lives in QK: token M enters through the keys and moves the
# attention pattern. A tracer that freezes attention has no representation for this edge
# and will report zero attribution on it -- while ablating it collapses the circuit.
PLANTED_EDGES_QK: frozenset[tuple[str, str]] = frozenset({("embed.tok", "L0.attn.pattern")})

PLANTED_EDGES: frozenset[tuple[str, str]] = PLANTED_EDGES_OV | PLANTED_EDGES_QK


def _base(cfg: TinyConfig) -> TinyTransformer:
    model = TinyTransformer(cfg)
    with torch.no_grad():
        for t in range(cfg.vocab):
            model.embed[t, TOK.start + t] = 1.0
        for i in range(cfg.max_len):
            model.pos_embed[i, POS_PHASE.start + (i % 3)] = 1.0
            model.pos_embed[i, CONST] = 1.0
        for k in range(3):
            model.W_U[PHASE_FEAT.start + k, k] = LOGIT_SCALE
    return model


def build_compositional(cfg: TinyConfig | None = None, match_accuracy: bool = True) -> TinyTransformer:
    cfg = cfg or TinyConfig()
    model = _base(cfg)

    with torch.no_grad():
        attn = model.attn[0]
        attn.W_Q[CONST, 0] = MARKER_SCORE
        attn.W_K[TOK.start + M, 0] = MARKER_SCORE
        for k in range(3):
            attn.W_V[POS_PHASE.start + k, k] = 1.0
            attn.W_O[k, START_PHASE.start + k] = 1.0

        mlp = model.mlp[1]
        for a in range(3):
            for b in range(3):
                j = 3 * a + b
                mlp.W_in[POS_PHASE.start + a, j] = 1.0
                mlp.W_in[START_PHASE.start + b, j] = 1.0
                # Bias -1 would put the one-input-matches case exactly on the ReLU kink,
                # making frozen-gate and real-ReLU interventions coincide and the
                # faithfulness gap identically zero. That is a degenerate coincidence, not
                # a property of transformers. -1.5 places it strictly below the kink.
                mlp.b_in[j] = -1.5
                mlp.W_out[j, PHASE_FEAT.start + (a - b - 1) % 3] = 2.0

        if match_accuracy:
            p = (3.0 * bayes_unigram_accuracy() - 1.0) / 2.0
            g = torch.Generator().manual_seed(0xC0FFEE)
            # Fix the open-gate count exactly; sampling it i.i.d. leaves the realised
            # fraction off by ~sqrt(p(1-p)/max_len), which is visible at this length.
            gate = torch.zeros(cfg.max_len)
            gate[torch.randperm(cfg.max_len, generator=g)[: round(p * cfg.max_len)]] = 1.0
            model.readout_gate.copy_(gate)

    return model


def build_heuristic(cfg: TinyConfig | None = None) -> TinyTransformer:
    cfg = cfg or TinyConfig()
    model = _base(cfg)
    posterior = posterior_phase_given_nt()

    with torch.no_grad():
        mlp = model.mlp[1]
        for t in range(4):
            mlp.W_in[TOK.start + t, t] = 1.0
            for ph in range(3):
                mlp.W_out[t, PHASE_FEAT.start + ph] = float(posterior[t, ph])

    return model


@torch.no_grad()
def phase_accuracy(model: TinyTransformer, tokens: torch.Tensor, labels: torch.Tensor) -> float:
    logits = model(tokens)
    pred = logits.argmax(dim=-1)
    mask = labels >= 0
    return float((pred[mask] == labels[mask]).float().mean())
