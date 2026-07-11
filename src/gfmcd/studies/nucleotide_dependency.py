"""Substitution-based nucleotide dependency (TO_FIX.md §21.1, §26 step 7).

The zero-shot scalar the partner-choice study traces: how much does perturbing one nucleotide change
the model's prediction at another position? Mask the target, read its nucleotide distribution, then
substitute the query base and re-read — the shift (KL) is the dependency of target on query. A true
splice partner should show higher dependency than a composition-matched decoy; this module is the
instrument, blind to which is which.

Works on any adapter exposing `.encode(seq)`, `.mlm_logits(ids)`, and the SpliceBERT nucleotide vocab.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NT_IDS = [6, 7, 8, 9]  # A, C, G, T in the SpliceBERT vocab
MASK_ID = 4


@torch.no_grad()
def _masked_nt_logprob(adapter, input_ids: torch.Tensor, token_pos: int) -> torch.Tensor:
    """log P(nucleotide | context) at token_pos, with that position masked. Shape [4]."""
    ids = input_ids.clone()
    ids[0, token_pos] = MASK_ID
    logits = adapter.mlm_logits(ids)[0, token_pos, NT_IDS]
    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def substitution_dependency(adapter, seq: str, query_nt: int, target_nt: int) -> float:
    """Mean KL shift in the target's nucleotide distribution when the query base is substituted.

    query_nt / target_nt are SEQUENCE positions (0-based). Returns a non-negative dependency score.
    """
    enc = adapter.encode(seq)
    ids, t2 = enc["input_ids"], enc["nt_to_token"]
    q_tok, p_tok = t2[query_nt], t2[target_nt]

    base = _masked_nt_logprob(adapter, ids, p_tok)
    base_p = base.exp()

    orig = int(ids[0, q_tok])
    shifts = []
    for alt in NT_IDS:
        if alt == orig:
            continue
        mut = ids.clone()
        mut[0, q_tok] = alt
        alt_dist = _masked_nt_logprob(adapter, mut, p_tok)
        shifts.append(float((base_p * (base - alt_dist)).sum()))  # KL(base || alt)
    return sum(shifts) / len(shifts) if shifts else 0.0


@torch.no_grad()
def partner_vs_decoy(adapter, seq: str, anchor_nt: int, true_partner_nt: int, decoy_nt: int) -> dict:
    """Compare the anchor's dependency on its true partner vs a composition-matched decoy.

    A partner-aware model should depend more on the true partner (dependency_true > dependency_decoy).
    Returns both scores and their difference (the study's per-example signal).
    """
    dep_true = substitution_dependency(adapter, seq, query_nt=true_partner_nt, target_nt=anchor_nt)
    dep_decoy = substitution_dependency(adapter, seq, query_nt=decoy_nt, target_nt=anchor_nt)
    return {"dependency_true": dep_true, "dependency_decoy": dep_decoy, "advantage": dep_true - dep_decoy}
