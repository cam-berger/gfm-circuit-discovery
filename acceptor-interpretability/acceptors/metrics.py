"""m_acc: masked-AG logit margin.

For a 2nt window expected to read "AG", mask both positions simultaneously
and score each against the other three nucleotides in {A,C,G,U}: margin =
logit(true base) - max(logit(other 3 bases)). m_acc is the sum of the two
per-position margins. Higher = model is more confident this position looks
like a real acceptor AG, independent of whatever the ground-truth label is
(the same definition is applied to non-splice control windows).
"""

BASES = "ACGU"


def _margin(logits_row, true_base: str, base_ids: dict) -> float:
    true_logit = logits_row[base_ids[true_base]].item()
    other = [base_ids[b] for b in BASES if b != true_base]
    other_max = logits_row[other].max().item()
    return true_logit - other_max


def m_acc(model, sequence: str, ag_start: int) -> float:
    """masked-AG logit margin at raw positions [ag_start, ag_start+2)."""
    positions = [ag_start, ag_start + 1]
    logits = model.masked_logits_at(sequence, positions)
    return m_acc_from_logits(logits, model.base_ids)


def m_acc_from_logits(logits, base_ids: dict) -> float:
    """Same computation as `m_acc`, but starting from an already-computed
    2-row logits tensor (e.g. from a patched forward pass in
    `acceptors.patching`) instead of running the model."""
    m_a = _margin(logits[0], "A", base_ids)
    m_g = _margin(logits[1], "G", base_ids)
    return m_a + m_g
