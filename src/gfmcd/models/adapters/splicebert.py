"""SpliceBERT tracing adapter (TO_FIX.md §19).

Wraps the pretrained BertForMaskedLM (community mirror yangheng/SpliceBERT-510nt — verified by
scripts/verify_model_source.py) behind a uniform surface: nucleotide<->token alignment, per-layer
activations (residual, Q/K/V, attention, per-head context, MLP), MLM logits, and self-restoring head
interventions. No circuit CLAIM may be made until the smoke tests in tests/test_splicebert_adapter.py
pass (they check hook idempotence, alignment, shapes, the length guard, and localized ablation).

Empirical tokenizer fact (do not forget): SpliceBERT tokenizes SPACE-SEPARATED nucleotides. A raw
string collapses to [UNK]; `encode()` inserts the spacing and records the nt->token offset.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from gfmcd.config import SUBSTRATES, assert_holdout_valid


class SpliceBERTAdapter:
    def __init__(self, hf_id: str | None = None, device: str | None = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.hf_id = hf_id or SUBSTRATES["splicebert"].hf_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        # eager attention so output_attentions=True returns probs (sdpa silently drops them).
        self.model = (
            AutoModelForMaskedLM.from_pretrained(self.hf_id, trust_remote_code=True, attn_implementation="eager")
            .to(self.device)
            .eval()
        )
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.d_model = cfg.hidden_size
        self.d_head = self.d_model // self.n_heads
        self.max_nt = cfg.max_position_embeddings - 2  # minus [CLS]/[SEP]

    # -- provenance / firewall --------------------------------------------------------------
    def assert_mechanistic_only(self) -> None:
        """SpliceBERT has no chromosome holdout; generalisation claims raise (config firewall)."""
        assert_holdout_valid("splicebert")  # raises ContaminationError by design

    # -- tokenization -----------------------------------------------------------------------
    def encode(self, seq: str) -> dict:
        """Tokenize a nucleotide string (space-separated internally) and record nt->token alignment.

        Returns {input_ids [1, L], nt_to_token: list mapping sequence position i -> token index}.
        For a plain sequence the layout is [CLS] n0 n1 ... n{k-1} [SEP], so nt i -> token i+1.
        """
        seq = seq.upper()
        if len(seq) > self.max_nt:
            raise ValueError(f"sequence length {len(seq)} exceeds SpliceBERT context {self.max_nt} nt")
        enc = self.tokenizer(" ".join(seq), return_tensors="pt")
        nt_to_token = list(range(1, 1 + len(seq)))  # skip [CLS] at 0
        return {"input_ids": enc["input_ids"].to(self.device), "nt_to_token": nt_to_token}

    # -- forward with cache -----------------------------------------------------------------
    @torch.no_grad()
    def run_with_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass returning (mlm_logits, cache) with per-layer residual/QKV/attn/head/MLP acts."""
        cache: dict = {}
        handles = []

        def cap(name):
            def hook(_m, _inp, out):
                cache[name] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook

        def cap_input(name):
            def hook(_m, inp, _out):
                cache[name] = inp[0].detach()
            return hook

        bert = self.model.bert
        for i, layer in enumerate(bert.encoder.layer):
            handles.append(layer.attention.self.query.register_forward_hook(cap(f"L{i}.q")))
            handles.append(layer.attention.self.key.register_forward_hook(cap(f"L{i}.k")))
            handles.append(layer.attention.self.value.register_forward_hook(cap(f"L{i}.v")))
            # input to the output projection is the concatenated per-head context.
            handles.append(layer.attention.output.dense.register_forward_hook(cap_input(f"L{i}.attn.context")))
            handles.append(layer.intermediate.dense.register_forward_hook(cap(f"L{i}.mlp.intermediate")))
            handles.append(layer.output.dense.register_forward_hook(cap(f"L{i}.mlp.out")))

        out = self.model(input_ids, output_hidden_states=True, output_attentions=True)
        try:
            for i, h in enumerate(out.hidden_states):
                cache[f"resid.{i}"] = h.detach()  # 0 = embeddings, i = after layer i-1
            for i, a in enumerate(out.attentions):
                cache[f"L{i}.attn.probs"] = a.detach()  # [batch, heads, q, k]
        finally:
            for h in handles:
                h.remove()
        return out.logits.detach(), cache

    def head_context(self, cache: dict, layer: int) -> torch.Tensor:
        """Per-head context [batch, seq, heads, d_head] from the cached concat input to output.dense."""
        ctx = cache[f"L{layer}.attn.context"]
        b, s, _ = ctx.shape
        return ctx.view(b, s, self.n_heads, self.d_head)

    # -- interventions (self-restoring) -----------------------------------------------------
    @contextmanager
    def ablate_head(self, layer: int, head: int):
        """Zero one head's contribution (its slice of the context fed to output.dense), restoring on exit."""
        lo, hi = head * self.d_head, (head + 1) * self.d_head
        dense = self.model.bert.encoder.layer[layer].attention.output.dense

        def pre_hook(_m, inp):
            x = inp[0].clone()
            x[..., lo:hi] = 0.0
            return (x,) + inp[1:]

        handle = dense.register_forward_pre_hook(pre_hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def mlm_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits.detach()
