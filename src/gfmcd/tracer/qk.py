"""Attribution through the attention pattern — the edges frozen-attention tracing cannot see.

Test A established the blind spot exactly: the compositional twin's marker-finding edge scores
0.000 under OV attribution, while ablating that token costs 0.617 accuracy. Freezing the pattern
is what makes the replacement model linear, and it is what deletes the QK circuit from the graph.
Reading frame is anchored by such an edge -- locating the ATG *is* the decide-where-to-look step --
so Rung 1 cannot be traced without this instrument.

**Plain gradients do not work here, and the reason generalises.** A head that has learned to find a
marker is confident: its score on the marker key is ~36 and ~0 elsewhere, so the softmax is fully
saturated and dL/dscore vanishes. Gradient attribution then reports the edge is irrelevant *because*
it is so relevant the head is certain. Measured on the compositional twin, naive pattern-gradient
attribution is identically zero on every edge.

So we integrate the gradient along a path from a content-free baseline to the real scores. The
baseline is the head's positional bias alone -- the pattern it would have with no content driving
it. This is the attention-pattern analogue of EAP-IG.

    score[p, q]   = (W_Q^T x_p) . (W_K^T x_q) + bias[p, q]
    content[p, q] = score[p, q] - bias[p, q]
    mean_grad     = mean over alpha of  dL/dscore  at  bias + alpha * content

and each score decomposes linearly into the input dimensions that produced it:

    key-side   attr(q, d) = mean_grad[p, q] * x[q, d] * (W_Q^T x_p) . W_K[d]
    query-side attr(p, d) = mean_grad[p, q] * x[p, d] * (W_K^T x_q) . W_Q[d]

The falsification control is `build_compositional_unread`: a model whose marker head still attends
to the marker (W_Q, W_K intact, coefficients nonzero) but whose output is never read (W_O = 0). The
instrument must report no edge into a pattern that does not matter, and it must discover that from
the gradient rather than from a zeroed coefficient.
"""

from __future__ import annotations

import torch

from gfmcd.models.tiny import TinyTransformer
from gfmcd.tracer.graph import GROUPS

IG_STEPS = 32


def phase_logit_metric(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Summed logit of the correct phase over labelled positions."""
    mask = labels >= 0
    safe = labels.clamp(min=0)
    return logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)[mask].sum()


def _forward_from_scores(model: TinyTransformer, tokens: torch.Tensor, scores0: torch.Tensor) -> torch.Tensor:
    """Re-run the model with layer-0 attention scores supplied, everything else unchanged."""
    n = tokens.shape[-1]
    x0 = model.embed[tokens] + model.pos_embed[:n]

    attn0 = model.attn[0]
    v = torch.einsum("bqd,dhc->bqhc", x0, attn0.W_V)
    z = torch.einsum("bhpq,bqhc->bphc", scores0.softmax(dim=-1), v)
    resid = x0 + torch.einsum("bphc,hcd->bpd", z, attn0.W_O)

    scratch: dict = {}
    resid = resid + model.mlp[0](resid, scratch, 0)
    for layer in range(1, model.cfg.n_layers):
        resid = resid + model.attn[layer](resid, scratch, layer)
        resid = resid + model.mlp[layer](resid, scratch, layer)

    return (resid * model.readout_gate[:n].unsqueeze(-1)) @ model.W_U


def naive_score_gradient(model: TinyTransformer, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gradient evaluated at the real scores, no integration. Retained to show it vanishes."""
    logits, cache = model.run_with_cache(tokens)
    return torch.autograd.grad(phase_logit_metric(logits, labels), cache["L0.attn.scores"])[0]


def integrated_score_gradient(
    model: TinyTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    steps: int = IG_STEPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (mean_grad, content, resid_embed). Baseline is the head's positional bias alone."""
    n = tokens.shape[-1]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    scores = cache["L0.attn.scores"].detach()
    bias = model.attn[0].attn_bias[:, :n, :n]
    content = scores - bias

    total = torch.zeros_like(scores)
    for i in range(steps):
        alpha = (i + 0.5) / steps
        s = (bias + alpha * content).detach().requires_grad_(True)
        loss = phase_logit_metric(_forward_from_scores(model, tokens, s), labels)
        total += torch.autograd.grad(loss, s)[0]

    return total / steps, content, cache["resid.embed"].detach()


@torch.no_grad()
def _coefficients(model: TinyTransformer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attn = model.attn[0]
    W_Q, W_K = attn.W_Q.detach(), attn.W_K.detach()
    q = torch.einsum("bpd,dhc->bphc", x, W_Q)
    k = torch.einsum("bqd,dhc->bqhc", x, W_K)
    return torch.einsum("bphc,dhc->bhpd", q, W_K), torch.einsum("bqhc,dhc->bhqd", k, W_Q)


def qk_attribution(
    model: TinyTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    steps: int = IG_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (key_side, query_side), each [batch, head, pos, key_pos, d_model]."""
    mean_grad, _content, x = integrated_score_gradient(model, tokens, labels, steps)
    key_coeff, query_coeff = _coefficients(model, x)

    with torch.no_grad():
        # key_coeff is indexed by the query position p, query_coeff by the key position q.
        key_side = torch.einsum("bhpq,bqd,bhpd->bhpqd", mean_grad, x, key_coeff)
        query_side = torch.einsum("bhpq,bpd,bhqd->bhpqd", mean_grad, x, query_coeff)

    return key_side, query_side


def qk_edges(
    model: TinyTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    steps: int = IG_STEPS,
) -> dict[tuple[str, str], float]:
    """Collapse per-dimension QK attributions into named edges into the attention pattern."""
    key_side, query_side = qk_attribution(model, tokens, labels, steps)

    out: dict[tuple[str, str], float] = {}
    for name, sl in GROUPS.items():
        out[(name, "L0.attn.pattern")] = float(key_side[..., sl].abs().mean())
        out[(name, "L0.attn.pattern.query")] = float(query_side[..., sl].abs().mean())
    return out


def key_attribution_by_position(
    model: TinyTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    head: int = 0,
    steps: int = IG_STEPS,
) -> torch.Tensor:
    """[batch, key_pos] -- total key-side attribution landing on each source position."""
    key_side, _ = qk_attribution(model, tokens, labels, steps)
    return key_side[:, head].abs().sum(dim=(1, 3))
