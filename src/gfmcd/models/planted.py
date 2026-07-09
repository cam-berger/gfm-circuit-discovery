"""Two transformers with identical behaviour and different mechanism.

The compositional twin attends to the start marker and computes (i - start - 1) % 3 exactly:
two input features feed one output feature. The heuristic twin is a genuine *bag of motifs* --
five local nucleotide detectors at offsets -2..+2, each casting an independent weak vote for
the phase, summed. It never represents frame.

An earlier heuristic twin was a unigram lookup: one neuron, one input. That is a single
shortcut rather than a bag, and it defeated the point of Test D, because a single shortcut is
*maximally* sparse and scored as more compositional than composition itself. A bag of motifs
must actually be a bag.

Both are hand-weighted, so the ground-truth edge set is known rather than inferred. Behaviour
is matched by gating the compositional twin's readout on a fixed, position-indexed mask:
solving p*1 + (1-p)*(1/3) = heuristic accuracy makes the twins equally accurate while leaving
their computations untouched.

Residual layout (d_model = 36):
    0:3    one-hot(i % 3)              from pos_embed
    3:6    one-hot(start % 3)          written by head 0 (compositional only)
    6:9    phase readout               written by the layer-1 MLP
    9:14   one-hot(token at i)         from embed
   14:19   one-hot(token at i-2)       written by head 1 (heuristic only)
   19:24   one-hot(token at i-1)       written by head 2
   24:29   one-hot(token at i+1)       written by head 3
   29:34   one-hot(token at i+2)       written by head 4
   34      unused
   35      constant 1                  query source for the marker-finding head
"""

from __future__ import annotations

import torch

from gfmcd.data.synthetic import EMIT, M, VOCAB
from gfmcd.models.tiny import NEG, TinyConfig, TinyTransformer

POS_PHASE = slice(0, 3)
START_PHASE = slice(3, 6)
PHASE_FEAT = slice(6, 9)
TOK = slice(9, 14)
CONST = 35

CONTEXT_OFFSETS = (-2, -1, 1, 2)
CONTEXT_SLOTS = {-2: 14, -1: 19, 1: 24, 2: 29}
BAG_OFFSETS = (-2, -1, 0, 1, 2)

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
# and reports zero attribution on it -- while ablating it collapses the circuit.
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
    return model


def build_compositional(
    cfg: TinyConfig | None = None,
    match_accuracy: bool = True,
    target_accuracy: float | None = None,
) -> TinyTransformer:
    cfg = cfg or TinyConfig()
    model = _base(cfg)

    with torch.no_grad():
        attn = model.attn[0]
        attn.W_Q[CONST, 0, 0] = MARKER_SCORE
        attn.W_K[TOK.start + M, 0, 0] = MARKER_SCORE
        for c in range(3):
            attn.W_V[POS_PHASE.start + c, 0, c] = 1.0
            attn.W_O[0, c, START_PHASE.start + c] = 1.0

        mlp = model.mlp[1]
        for a in range(3):
            for b in range(3):
                j = 3 * a + b
                mlp.W_in[POS_PHASE.start + a, j] = 1.0
                mlp.W_in[START_PHASE.start + b, j] = 1.0
                # Bias -1 would put the one-input-matches case exactly on the ReLU kink,
                # making frozen-gate and real-ReLU interventions coincide and the
                # faithfulness gap identically zero. -1.5 places it strictly below the kink.
                mlp.b_in[j] = -1.5
                mlp.W_out[j, PHASE_FEAT.start + (a - b - 1) % 3] = 2.0

        for k in range(3):
            model.W_U[PHASE_FEAT.start + k, k] = LOGIT_SCALE

        if match_accuracy:
            target = heuristic_accuracy_ceiling() if target_accuracy is None else target_accuracy
            p = (3.0 * target - 1.0) / 2.0
            g = torch.Generator().manual_seed(0xC0FFEE)
            gate = torch.zeros(cfg.max_len)
            gate[torch.randperm(cfg.max_len, generator=g)[: round(p * cfg.max_len)]] = 1.0
            model.readout_gate.copy_(gate)

    return model


def build_heuristic(cfg: TinyConfig | None = None, seq_len: int = 256) -> TinyTransformer:
    """Five local motif detectors, one per offset, summing independent votes for the phase."""
    cfg = cfg or TinyConfig()
    model = _base(cfg)
    log_emit = EMIT.log()

    with torch.no_grad():
        attn = model.attn[0]
        for h, off in enumerate(CONTEXT_OFFSETS, start=1):
            bias = torch.full((cfg.max_len, cfg.max_len), NEG)
            for i in range(cfg.max_len):
                bias[i, min(max(i + off, 0), seq_len - 1)] = 0.0
            attn.attn_bias[h] = bias
            for t in range(5):
                attn.W_V[TOK.start + t, h, t] = 1.0
                attn.W_O[h, t, CONTEXT_SLOTS[off] + t] = 1.0

        mlp = model.mlp[1]
        for oi, off in enumerate(BAG_OFFSETS):
            src = TOK.start if off == 0 else CONTEXT_SLOTS[off]
            for t in range(5):
                j = 5 * oi + t
                mlp.W_in[src + t, j] = 1.0
                mlp.b_in[j] = -0.5
                if VOCAB[t] == "M":
                    continue
                for ph in range(3):
                    mlp.W_out[j, PHASE_FEAT.start + ph] = 2.0 * float(log_emit[(ph + off) % 3, t])

        for k in range(3):
            model.W_U[PHASE_FEAT.start + k, k] = 1.0

    return model


def heuristic_accuracy_ceiling() -> float:
    """Bayes accuracy of the 5-offset bag, computed once from the emission table."""
    global _CEILING
    try:
        return _CEILING
    except NameError:
        pass

    from gfmcd.data.synthetic import make_batch

    cfg = TinyConfig()
    tokens, labels = make_batch(n_seq=64, body_len=90, seed=99)
    _CEILING = phase_accuracy(build_heuristic(cfg, seq_len=tokens.shape[-1]), tokens, labels)
    return _CEILING


@torch.no_grad()
def phase_accuracy(model: TinyTransformer, tokens: torch.Tensor, labels: torch.Tensor) -> float:
    logits = model(tokens)
    pred = logits.argmax(dim=-1)
    mask = labels >= 0
    return float((pred[mask] == labels[mask]).float().mean())


def build_shortcut(cfg: TinyConfig | None = None, seq_len: int = 256) -> TinyTransformer:
    """A single unigram motif: one neuron, one input feature, no context, no frame.

    Included to show what Jacobian sparsity actually measures. A model that cheats with ONE
    trick has in-degree 1 and is therefore *maximally* sparse -- so H3's discriminator can
    separate composition from a bag of motifs, but not from a lone shortcut.
    """
    cfg = cfg or TinyConfig()
    model = _base(cfg)
    log_emit = EMIT.log()

    with torch.no_grad():
        mlp = model.mlp[1]
        for t in range(5):
            mlp.W_in[TOK.start + t, t] = 1.0
            mlp.b_in[t] = -0.5
            if VOCAB[t] == "M":
                continue
            for ph in range(3):
                mlp.W_out[t, PHASE_FEAT.start + ph] = 2.0 * float(log_emit[ph, t])

        for k in range(3):
            model.W_U[PHASE_FEAT.start + k, k] = 1.0

    return model


def build_compositional_unread(cfg: TinyConfig | None = None) -> TinyTransformer:
    """Superseded by `build_bag_with_unread_marker_head`; kept for the record.

    Zeroing W_O here also silences the MLP, because the phase neurons need the head's output to
    fire. The logits go to zero, the loss goes constant, and *every* gradient vanishes. Zero QK
    attribution then proves nothing about the instrument's specificity.
    """
    model = build_compositional(cfg, match_accuracy=False)
    with torch.no_grad():
        model.attn[0].W_O[0].zero_()
    return model


def build_bag_with_unread_marker_head(cfg: TinyConfig | None = None, seq_len: int = 256) -> TinyTransformer:
    """The strong QK falsification control: a working model with a live but unread head.

    Head 0 attends to the start marker exactly as the compositional twin does -- W_Q and W_K are
    intact, so its score coefficients are nonzero and its pattern is content-driven. But W_O[0] is
    zero, so nothing downstream reads it. The readout is produced entirely by the bag-of-motifs
    pathway on heads 1-4, so the loss is non-constant and gradients elsewhere are alive.

    `build_compositional_unread` is the weak version: zeroing W_O there also silences the MLP, so
    the loss is constant and *every* gradient vanishes. Zero attribution then proves nothing. Here
    the instrument must single out the one head whose pattern does not matter, while the model is
    busy doing real work through other heads.
    """
    cfg = cfg or TinyConfig()
    model = build_heuristic(cfg, seq_len=seq_len)
    with torch.no_grad():
        model.attn[0].W_Q[CONST, 0, 0] = MARKER_SCORE
        model.attn[0].W_K[TOK.start + M, 0, 0] = MARKER_SCORE
        for c in range(3):
            model.attn[0].W_V[POS_PHASE.start + c, 0, c] = 1.0
        model.attn[0].W_O[0].zero_()
    return model
