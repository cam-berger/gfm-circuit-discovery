"""LOCAL dry-run: coordinate-correct scorer (REAL Caduceus tokenizer) + PLANTED mock model.
Validates every non-CUDA assumption so the GPU run is one clean shot. Run: python test_local_dryrun.py
"""
from __future__ import annotations
import sys
import numpy as np
from transformers import AutoTokenizer
from caduceus_scorer import CaduceusScorer
from mock_caduceus import MockCaduceus

MID = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
W = 1024
rng = np.random.default_rng(0)


def sub(seq, pos, alt):
    return seq[:pos] + alt + seq[pos + len(alt):]


def make_seq(donor=400, partner_ag=500):
    s = list("".join(rng.choice(list("ACGT"), size=W)))
    s[donor], s[donor + 1], s[donor + 2] = "G", "T", "A"  # donor GT; d+2=A (no spurious donor at d+1)
    s[partner_ag], s[partner_ag + 1] = "A", "G"           # partner acceptor AG
    if partner_ag - 4 >= 0: s[partner_ag - 4:partner_ag] = list("TGCA")  # partner tag (mock couples to tagged AG)
    return "".join(s)


def main():
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    bid = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
    model = MockCaduceus(base_ids=bid)
    sc = CaduceusScorer(model, tok, device="cpu", batch_size=8, window_length=W)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        ok = ok and cond

    print("== init asserts (prefix=0, trailing [SEP], dynamic ids) passed if we got here ==")
    check("base ids A=7 C=8 G=9 T=10", bid == {"A": 7, "C": 8, "G": 9, "T": 10}, str(bid))

    d, a = 400, 500
    seq = make_seq(d, a)

    # 1. donor recognition + coordinate correctness
    m_don = sc.donor_margin([seq], [d])[0]
    check("margin_donor(clean) > 0 (donor recognised at index d)", m_don > 0, f"= {m_don:.3f}")
    m_off1 = sc.donor_margin([seq], [d + 1])[0]     # masking a non-donor position (no T-next)
    check("coordinate_sensitive: margin at d >> at d+1", m_don > m_off1 + 1.0, f"d={m_don:.3f} d+1={m_off1:.3f}")

    # 2. A_donor: destroying the donor GT collapses the margin
    seq_dead = sub(seq, d, "AA")                    # GT -> AA (no donor)
    m_dead = sc.donor_margin([seq_dead], [d])[0]
    A_donor = m_don - m_dead
    check("A_donor > 0 (donor-destruction anchor)", A_donor > 0.5, f"= {A_donor:.3f}")

    # 3. coupling: corrupting the partner AG (AG->CC) drops the donor margin
    seq_cc = sub(seq, a, "CC")                      # partner AG -> CC
    m_cc = sc.donor_margin([seq_cc], [d])[0]
    C = m_don - m_cc
    check("coupling C > 0 (partner-specific corruption)", C > 0.1, f"= {C:.3f}")

    # 4. distance decay: nearer partner -> larger coupling than farther partner
    seq_near = make_seq(d, d + 40); seq_far = make_seq(d, d + 190)
    C_near = sc.donor_margin([seq_near], [d])[0] - sc.donor_margin([sub(seq_near, d + 40, "CC")], [d])[0]
    C_far = sc.donor_margin([seq_far], [d])[0] - sc.donor_margin([sub(seq_far, d + 190, "CC")], [d])[0]
    check("distance decay: C(near) > C(far)", C_near > C_far, f"near={C_near:.3f} far={C_far:.3f}")

    # 5. long-range transmission curve is monotone-ish decreasing in offset
    curve = []
    for off in (10, 30, 60, 100, 150, 200):
        sq = make_seq(d, d + off)
        curve.append(sc.donor_margin([sq], [d])[0] - sc.donor_margin([sub(sq, d + off, "CC")], [d])[0])
    check("transmission curve decreases with offset", curve[0] > curve[-1], f"{[round(c,2) for c in curve]}")

    # 6. mlm_health runs + finite (PLUMBING only; the real >0 competence gate is Caduceus's job on CUDA,
    #    a random-DNA mock cannot recover a masked base it never sees)
    h = sc.mlm_health([seq], n_probe=30, seed=0)
    check("mlm_health finite (plumbing; real >0 gate is on-CUDA)", np.isfinite(h), f"= {h:.3f}")

    # 7. equal-length batch enforced (SSM padding safety)
    raised = False
    try:
        sc._encode([seq, seq[:-10]])
    except AssertionError:
        raised = True
    check("mixed-length batch REJECTED (no SSM padding)", raised)

    # 8. determinism: two identical calls agree exactly
    check("deterministic readout", sc.donor_margin([seq], [d])[0] == m_don)

    print("\nDRY-RUN:", "ALL PASS ✓" if ok else "FAILURES ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
