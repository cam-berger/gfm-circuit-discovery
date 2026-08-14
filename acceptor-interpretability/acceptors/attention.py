"""Per-head attention-mass readout for the head content-control (next-steps
item 4): how much of a query position's attention budget lands on a given
raw-coordinate region, for one head or for every (layer, head) at once.

Raw sequence coordinates map to token indices via +1 (see `acceptors.model`
docstring - <cls> occupies token 0). Both helpers take pre-computed
`attentions` from `SpliceBert.masked_attentions_at` (a tuple of per-layer
[num_heads, seq_tokens, seq_tokens] tensors) so a single forward pass can
feed every head/region combination without re-running the model.
"""

import torch


def region_mass_head(
    attentions, layer: int, head: int, query_tokens, key_start_raw: int, key_end_raw: int
) -> float:
    """Attention mass one (layer, head) sends from `query_tokens` (token
    indices) into raw region [key_start_raw, key_end_raw), averaged over the
    query positions."""
    key_lo, key_hi = key_start_raw + 1, key_end_raw + 1
    layer_attn = attentions[layer][head]  # [seq_tokens, seq_tokens]
    q = torch.as_tensor(query_tokens, device=layer_attn.device)
    return layer_attn[q, key_lo:key_hi].sum(dim=-1).mean().item()


def region_mass_all_heads(attentions, query_tokens, key_start_raw: int, key_end_raw: int) -> torch.Tensor:
    """Same as `region_mass_head`, vectorized over every (layer, head).
    Returns a [num_layers, num_heads] tensor - useful for ranking heads by
    how much they attend to a region without extra forward passes."""
    key_lo, key_hi = key_start_raw + 1, key_end_raw + 1
    per_layer = []
    for layer_attn in attentions:  # [heads, seq_tokens, seq_tokens]
        q = torch.as_tensor(query_tokens, device=layer_attn.device)
        mass = layer_attn[:, q, key_lo:key_hi].sum(dim=-1).mean(dim=-1)  # [heads]
        per_layer.append(mass)
    return torch.stack(per_layer)
