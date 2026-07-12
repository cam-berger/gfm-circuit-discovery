"""Regression guard for attribution patching: the linear AtP ranking must track
exact head-patch recovery. Skipped unless an additive-frozen checkpoint and the
discovery split are present (they are build artifacts, not committed)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.attribution import head_attribution_scan
from gfmcd.circuits.hf_bert import capture_head_outputs, head_patch_scan
from gfmcd.splicing.config import load_config
from gfmcd.splicing.context_controls import mask_true_partner_neighborhood
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.readout import collate_pair_indices, true_pair_margin
from gfmcd.splicing.training import load_pair_checkpoint

REPO = Path(__file__).resolve().parents[1]
CFG = load_config(REPO / "configs/splicebert_partner_selection/base.json")
CKPT = REPO / "colab_results/splicebert_run/runs/additive-frozen_seed0/checkpoint.pt"
DISC = REPO / "results/splicebert_partner_selection/circuits/discovery/discovery.jsonl"
MODEL = REPO / CFG["model"]["path"]

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and DISC.exists() and MODEL.exists()),
    reason="tracing build artifacts (checkpoint/discovery split) not present",
)


def _spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    return cov / (va * vb)


def test_attribution_tracks_exact_patching():
    rep = int(CFG["readout"]["representation_layer"])
    mc = CFG["model"]
    adapter = SpliceBertAdapter.from_pretrained(
        MODEL, tokenizer_mode=mc.get("tokenizer_mode", "auto"), device="cpu",
        trust_remote_code=False, local_files_only=True, model_class=mc.get("model_class", "masked_lm"))
    readout, _ = load_pair_checkpoint(CKPT, adapter=adapter)
    ex = load_jsonl(DISC)[:16]
    clean_tok = adapter.tokenize([x.sequence for x in ex])
    corrupt_tok = adapter.tokenize([mask_true_partner_neighborhood(x, kind="donor", flank=12) for x in ex])
    pair_batch = collate_pair_indices(ex, token_offset=clean_tok.prefix_tokens, device=adapter.device)

    def margin_tensor(tok, grad):
        out = adapter.forward(tok, output_hidden_states=True, output_attentions=False, grad=grad)
        return true_pair_margin(readout(out.hidden_states[rep], pair_batch), pair_batch).mean()

    def score(tok):
        with torch.no_grad():
            return float(margin_tensor(tok, grad=False))

    clean_score, corrupt_score = score(clean_tok), score(corrupt_tok)
    assert clean_score - corrupt_score > 0.05  # corruption is behaviorally effective
    cache = capture_head_outputs(adapter, lambda: score(clean_tok))
    exact = head_patch_scan(adapter, clean_score=clean_score, corrupt_score=corrupt_score,
                            source_cache=cache, corrupt_score_fn=lambda: score(corrupt_tok))
    attr = head_attribution_scan(adapter, cache, lambda: margin_tensor(corrupt_tok, grad=True))
    keys = sorted((e.layer, e.head) for e in exact)
    em = {(e.layer, e.head): e.recovery_fraction for e in exact}
    am = {(a.layer, a.head): a.attribution for a in attr}
    rho = _spearman([am[k] for k in keys], [em[k] for k in keys])
    assert rho >= 0.6, f"AtP does not track exact patching (Spearman={rho:.3f})"
