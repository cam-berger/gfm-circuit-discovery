"""Calibrate what freezing LayerNorm costs the tracer (TO_FIX.md §16).

Every tracer error rate in Gate 0 was measured on the LN-free planted model, where a frozen attention
pattern + frozen ReLU gate make the model exactly linear and attribution exact. GFM-nano and SpliceBERT
have LayerNorm; the replacement model freezes it (reuses clean-pass mean/std under intervention). This
module measures the resulting error on an LN-equipped copy of the SAME planted circuit:

- conservation: how far the LN-ignoring direct-path reconstruction misses (the unmodeled LN term),
- edge recall/precision: whether the planted OV structure still survives,
- frozen-LN faithfulness gap: replacement (frozen LN) vs ground truth (recomputed LN) causal effect,
  as a function of intervention magnitude.
"""

from __future__ import annotations

import torch

from gfmcd.models.planted import PLANTED_EDGES_OV, build_compositional
from gfmcd.models.tiny import TinyConfig
from gfmcd.tracer.graph import abstract_edges, conservation_residuals, recovered_edges


def _relnorm(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(b.norm())
    return float((a - b).norm() / denom) if denom > 1e-12 else 0.0


@torch.no_grad()
def edge_recovery(model, tokens) -> dict[str, float]:
    """Recall/precision of the planted OV edges under the (LN-ignoring) direct-path tracer."""
    found = recovered_edges(abstract_edges(model, tokens))
    hits = found & PLANTED_EDGES_OV
    recall = len(hits) / len(PLANTED_EDGES_OV)
    precision = len(hits) / len(found) if found else 0.0
    return {"recall": recall, "precision": precision, "n_found": len(found)}


@torch.no_grad()
def frozen_ln_faithfulness_gap(model, tokens, alphas: list[float]) -> list[dict]:
    """Frozen-LN vs recomputed-LN causal effect of ablating head 0's OV, per magnitude alpha.

    gap = ||pred_delta - true_delta|| / ||true_delta||, where pred_delta uses frozen LN stats (the
    replacement model) and true_delta recomputes LN (ground truth). For an LN-free model the two paths
    are identical and the gap is exactly 0 (a control).
    """
    ln = model.cfg.layernorm
    stats = model.capture_ln_stats(tokens) if ln else None
    clean = model(tokens)
    W_O0 = model.attn[0].W_O
    saved = W_O0.data.clone()

    rows = []
    for alpha in alphas:
        W_O0.data[0] = saved[0] * (1.0 - alpha)
        true_ab = model(tokens)  # recompute LN
        repl_ab = model(tokens, ln_frozen=stats) if ln else true_ab
        W_O0.data.copy_(saved)

        true_delta = clean - true_ab
        pred_delta = clean - repl_ab
        cos = torch.nn.functional.cosine_similarity(pred_delta.reshape(1, -1), true_delta.reshape(1, -1))
        rows.append(
            {
                "alpha": alpha,
                "true_effect_norm": float(true_delta.norm()),
                "faithfulness_gap": _relnorm(pred_delta, true_delta),
                "direction_cos": float(cos),
            }
        )
    return rows


def calibrate(seed: int = 3, n_seq: int = 8, body_len: int = 60, alphas: list[float] | None = None) -> dict:
    """Run the full LN calibration and return a JSON-able report (LN-free control + LN model)."""
    from gfmcd.data.synthetic import make_batch

    alphas = alphas or [0.1, 0.25, 0.5, 0.75, 1.0]
    tokens, _ = make_batch(n_seq=n_seq, body_len=body_len, seed=seed)

    free = build_compositional(cfg=TinyConfig(layernorm=False), match_accuracy=False)
    ln = build_compositional(cfg=TinyConfig(layernorm=True), match_accuracy=False)

    return {
        "config": {"seed": seed, "n_seq": n_seq, "body_len": body_len, "alphas": alphas},
        "ln_free": {
            "conservation_max_err": max(conservation_residuals(free, tokens).values()),
            "edges": edge_recovery(free, tokens),
            "gap_curve": frozen_ln_faithfulness_gap(free, tokens, alphas),
        },
        "layernorm": {
            "conservation_max_err": max(conservation_residuals(ln, tokens).values()),
            "edges": edge_recovery(ln, tokens),
            "gap_curve": frozen_ln_faithfulness_gap(ln, tokens, alphas),
        },
    }
