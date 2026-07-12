# Retracted artifacts — do not cite

## `competing_acceptors.json` — RETRACTED 2026-07-12 (R7)

The experiment SWAPPED L3H1's (key, value) pair between the partner acceptor and a decoy and
concluded that "address and content are separable" from the fact that both-swap restored the
donor margin while attention sat on the decoy.

**It is an algebraic identity.** Attention output is `out_i = Σ_j softmax(q_i·k_j)·v_j`.
Transposing (key, value) PAIRS between two positions permutes the summation index `j`: the
multiset of pairs is unchanged, the softmax denominator is unchanged, and `out' = out`
**exactly** — for any weights, any data. Verified on a randomly-initialised BertForMaskedLM:
`|both-swap − baseline| = 1.2e-06`.

The tells were in this file all along and were rationalised instead of chased:
- baseline margin `4.655215060313543` vs both-swap `4.655215079784393` (Δ = 2e-8)
- key-swap `4.19994298140208` vs value-swap `4.199942835172018` (identical to 7 s.f., because
  they are provably the same intervention)

Replaced by a **one-way copy** (`run_sealed_test.py` arm D), which changes the multiset of
(key, value) pairs and can therefore fail — and one of whose cells does (the partner's value
with no address recovers 0.004).

See `studies/splicebert_partner_selection/CLAIMS.md` R7.
