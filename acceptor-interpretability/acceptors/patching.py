"""Causal interventions on a single attention head's output at the masked-AG
query positions - the primitive behind next-steps item 5's intervention
ladder (mean-ablation for necessity, donor-sequence restoration patching for
mediation, and the query/value split from path-patching for routing vs.
content).

`SpliceBertSelfAttention.forward` (verified against `multimolecule`'s
source) already reshapes its output to `[batch, seq_tokens, hidden]` with
each head occupying a contiguous `head_dim`-wide column slice, and separately
returns the attention-probability tensor - exactly what `masked_attentions_at`
already reads in `acceptors.model`. `capture_layers` additionally recomputes
each layer's per-head value tensor (`W_V @ hidden_states`, reshaped) from the
same hooked hidden-state input, confirmed numerically to reconstruct
`attn_output` exactly (`attn_weights @ value_layer` per head).

Because every intervention here only overwrites a layer's *output* (never
anything upstream of it), a replacement tensor can always be built from two
independent, unpatched forward passes (e.g. one on the clean sequence, one
on a corrupted variant) - there is no need to compute anything "live" inside
a patched run. `run_with_replacement` is therefore a single, simple hook:
splice a precomputed tensor into one head's output at the masked-AG token
positions, then let the rest of the forward pass run normally.
"""

import torch


def _self_attention(model, layer: int):
    """`model` is the `acceptors.model.SpliceBert` wrapper; `.model` is the
    underlying `SpliceBertForMaskedLM`, whose own `.model` is the encoder
    stack (`multimolecule`'s naming, not this project's)."""
    return model.model.model.encoder.layer[layer].attention.self


@torch.no_grad()
def capture_layers(model, sequence: str, raw_positions, layers):
    """One masked-AG forward pass (masking `raw_positions` exactly as
    `masked_logits_at` does), hooking every layer in `layers` simultaneously.

    Returns {"logits": [len(raw_positions), vocab], "token_positions": [...],
    layer: {"attn_output": [seq_tokens, hidden], "attn_weights": [heads,
    seq_tokens, seq_tokens], "value_layer": [heads, seq_tokens, head_dim]}
    for each requested layer} (batch dimension dropped throughout).
    """
    token_positions = [p + 1 for p in raw_positions]
    captured = {}
    handles = []
    for layer in layers:
        self_attn = _self_attention(model, layer)

        def hook(module, args, output, _layer=layer, _self_attn=self_attn):
            hidden_states = args[0]
            attn_output, attn_weights = output
            value_layer = _self_attn.transpose_for_scores(_self_attn.value(hidden_states))
            captured[_layer] = {
                "attn_output": attn_output[0],
                "attn_weights": attn_weights[0],
                "value_layer": value_layer[0],
            }

        handles.append(self_attn.register_forward_hook(hook))

    try:
        enc = model.tokenizer(sequence, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"].clone()
        for tp in token_positions:
            input_ids[0, tp] = model.mask_id
        enc["input_ids"] = input_ids
        out = model.model(**enc, output_attentions=True)
    finally:
        for h in handles:
            h.remove()

    return {"logits": out.logits[0, token_positions], "token_positions": token_positions, **captured}


@torch.no_grad()
def run_with_replacement(model, sequence: str, raw_positions, layer: int, head: int, replacement):
    """Masked-AG forward pass on `sequence`, with `layer`'s head `head`
    output at the masked query positions overwritten by the precomputed
    `replacement` tensor (shape `[len(raw_positions), head_dim]`). Returns
    logits at the masked positions after the rest of the forward pass runs
    normally on top of the patched value."""
    token_positions = [p + 1 for p in raw_positions]
    self_attn = _self_attention(model, layer)
    head_dim = self_attn.attention_head_size
    lo, hi = head * head_dim, (head + 1) * head_dim

    def hook(module, args, output):
        attn_output, attn_weights = output
        attn_output = attn_output.clone()
        attn_output[0, token_positions, lo:hi] = replacement
        return attn_output, attn_weights

    handle = self_attn.register_forward_hook(hook)
    try:
        enc = model.tokenizer(sequence, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"].clone()
        for tp in token_positions:
            input_ids[0, tp] = model.mask_id
        enc["input_ids"] = input_ids
        out = model.model(**enc, output_attentions=True)
    finally:
        handle.remove()
    return out.logits[0, token_positions]


def _head_bounds(model, layer: int, head: int):
    head_dim = _self_attention(model, layer).attention_head_size
    return head * head_dim, (head + 1) * head_dim


def full_patch_replacement(model, donor_cap: dict, layer: int, head: int):
    """Restoration-patch replacement: this head's real output at the masked
    positions in `donor_cap` (both routing and content from the donor run) -
    the mediation test's patch."""
    lo, hi = _head_bounds(model, layer, head)
    tp = donor_cap["token_positions"]
    return donor_cap[layer]["attn_output"][tp, lo:hi]


def q_patch_replacement(model, routing_cap: dict, content_cap: dict, layer: int, head: int):
    """Q-patch (routing-only): `routing_cap`'s attention pattern applied to
    `content_cap`'s values - "if this head attended the way it does on
    `routing_cap`, but could only read whatever content is actually in
    `content_cap`, would the readout recover?" Isolates the routing (QK)
    contribution from a fixed, real value tensor."""
    lo, hi = _head_bounds(model, layer, head)
    head_dim = hi - lo
    tp = routing_cap["token_positions"]
    probs = routing_cap[layer]["attn_weights"][head, tp, :]  # [n_pos, seq_tokens]
    values = content_cap[layer]["value_layer"][head]  # [seq_tokens, head_dim]
    return probs @ values


def v_patch_replacement(model, routing_cap: dict, content_cap: dict, layer: int, head: int):
    """V-patch (content-only): `routing_cap`'s real (e.g. degraded)
    attention pattern applied to `content_cap`'s values - "given how little
    (or how differently) this head actually attends under `routing_cap`,
    does reading the correct content in `content_cap` still help?" Isolates
    the content (OV) contribution from a fixed, real attention pattern."""
    return q_patch_replacement(model, routing_cap, content_cap, layer, head)


@torch.no_grad()
def run_with_multi_replacement(model, sequence: str, raw_positions, replacements: dict):
    """Like `run_with_replacement`, but patches several (layer, head) pairs
    at once in a single forward pass - `replacements`: {(layer, head):
    tensor}. One hook per distinct layer, each applying every patch that
    lands in it; the whole-circuit-scrub half of the intervention ladder
    (do several individually-null heads matter jointly?)."""
    token_positions = [p + 1 for p in raw_positions]
    by_layer = {}
    for (layer, head), repl in replacements.items():
        by_layer.setdefault(layer, []).append((head, repl))

    handles = []
    for layer, head_repls in by_layer.items():
        self_attn = _self_attention(model, layer)
        head_dim = self_attn.attention_head_size

        def hook(module, args, output, _head_repls=head_repls, _head_dim=head_dim):
            attn_output, attn_weights = output
            attn_output = attn_output.clone()
            for head, repl in _head_repls:
                lo, hi = head * _head_dim, (head + 1) * _head_dim
                attn_output[0, token_positions, lo:hi] = repl
            return attn_output, attn_weights

        handles.append(self_attn.register_forward_hook(hook))

    try:
        enc = model.tokenizer(sequence, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"].clone()
        for tp in token_positions:
            input_ids[0, tp] = model.mask_id
        enc["input_ids"] = input_ids
        out = model.model(**enc, output_attentions=True)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, token_positions]


@torch.no_grad()
def mean_ablation_vectors(model, sequences, raw_positions, layer_to_heads: dict):
    """Dataset-mean attention output at the masked positions, per (layer,
    head) in `layer_to_heads` ({layer: [head, ...]}), over `sequences` -
    the mean-ablation replacement (the literature-preferred alternative to
    zero-ablation: it keeps the patched component on-distribution instead of
    sending downstream layers to activations they'd never otherwise see).
    One capture pass per sequence regardless of how many heads are requested
    (all requested layers are hooked simultaneously). Returns
    {(layer, head): [len(raw_positions), head_dim] mean tensor}."""
    layers = list(layer_to_heads)
    sums = {(layer, head): None for layer in layers for head in layer_to_heads[layer]}
    n = 0
    for seq in sequences:
        cap = capture_layers(model, seq, raw_positions, layers)
        tp = cap["token_positions"]
        for layer in layers:
            for head in layer_to_heads[layer]:
                lo, hi = _head_bounds(model, layer, head)
                sl = cap[layer]["attn_output"][tp, lo:hi]
                key = (layer, head)
                sums[key] = sl.clone() if sums[key] is None else sums[key] + sl
        n += 1
    return {k: v / n for k, v in sums.items()}
