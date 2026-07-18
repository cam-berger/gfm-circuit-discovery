"""FULL-PIPELINE dry-run: synthetic planted cohort -> mock margins -> estimands -> stats -> §6 decision.
Validates the whole harness end-to-end (not just the readout): the planted-positive mock MUST land QUALIFIES,
and a partner-permuted negative control must NOT. Run: python test_pipeline_dryrun.py
"""
from __future__ import annotations
import numpy as np
from transformers import AutoTokenizer
from caduceus_scorer import CaduceusScorer
from mock_caduceus import MockCaduceus
import caduceus_t1 as H

MID = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
W = 1024
rng = np.random.default_rng(1)


def sub(s, p, alt): return s[:p] + alt + s[p + len(alt):]


def build_example(gene_seed):
    """One synthetic locus: true donor + partner AG + TWO distance-matched non-partner AGs + a 2nd real donor."""
    r = np.random.default_rng(gene_seed)
    s = list("".join(r.choice(list("ACGT"), size=W)))
    d = 300 + int(r.integers(0, 40))                  # true donor
    a = d + 80 + int(r.integers(0, 40))               # partner acceptor AG
    npg = a + 20 + int(r.integers(0, 20)); npg2 = npg + 20 + int(r.integers(0, 20))  # two non-partner AGs
    d2 = d + 300 + int(r.integers(0, 60))             # a SECOND real donor GT (other_gt), far
    for pos in (d, d2): s[pos], s[pos + 1], s[pos + 2] = "G", "T", "A"
    for pos in (a, npg, npg2): s[pos], s[pos + 1] = "A", "G"
    s[a - 4:a] = list("TGCA")          # PARTNER TAG on the annotated partner only (=> partner-specific)
    for pos in (npg, npg2):
        if pos - 4 >= 0: s[pos - 4:pos] = list("GGGG")   # non-partner AGs: no tag (recognised, not coupling)
    seq = "".join(s)
    return dict(seq=seq, d=d, a=a, npg=npg, npg2=npg2, d2=d2)


def constructs_for(ex, permute_partner=False):
    """Build the tagged construct set for one example. permute_partner => corrupt the NON-partner as 'partner'
    (the FPR negative control: partner-specific structure should vanish)."""
    seq, d, a, npg, npg2, d2 = ex["seq"], ex["d"], ex["a"], ex["npg"], ex["npg2"], ex["d2"]
    part = npg if permute_partner else a              # which AG is treated as 'the partner'
    other = a if permute_partner else npg
    partner_cc = sub(seq, part, "CC")
    return {
        "clean": (seq, d), "partner_cc": (partner_cc, d),
        "donor_dead": (sub(seq, d, "AA"), d),
        "nonpartner_cc": (sub(seq, other, "CC"), d),
        "nonpartner2_cc": (sub(seq, npg2, "CC"), d),  # 2nd non-partner => genuine permuted-null tau
        "clean_other": (seq, d2), "partner_cc_other": (partner_cc, d2),
    }


def run(scorer, permute_partner=False, n_genes=30, per_gene=3):
    tags = ["clean", "partner_cc", "donor_dead", "nonpartner_cc", "nonpartner2_cc", "clean_other", "partner_cc_other"]
    ex_vals = {t: [] for t in tags}; genes = []; splits = []
    for gene in range(n_genes):
        split = "discovery" if gene % 2 == 0 else "confirm"
        for k in range(per_gene):
            ex = build_example(gene * 100 + k + (1000 if permute_partner else 0))
            cons = constructs_for(ex, permute_partner)
            seqs = [cons[t][0] for t in tags]; pos = [cons[t][1] for t in tags]
            m = scorer.donor_margin(seqs, pos)
            for t, v in zip(tags, m):
                ex_vals[t].append(float(v))
            genes.append(f"g{gene}"); splits.append(split)
    # GENUINE permuted-null tau on discovery (two non-partners, no real partner identity)
    disc = [i for i, s in enumerate(splits) if s == "discovery"]
    clean = np.array(ex_vals["clean"]); np1 = np.array(ex_vals["nonpartner_cc"])
    np2 = np.array(ex_vals["nonpartner2_cc"]); dd = np.array(ex_vals["donor_dead"])
    null_ps = ((clean - np1) - (clean - np2))[disc]; A = (clean - dd)[disc]; gd = [genes[i] for i in disc]
    tau = max(0.0, H.paired_gene_bootstrap_ratio(null_ps.tolist(), A.tolist(), gd, seed=0)["ci"][1])
    lr = {10: (0.5, 0.7), 30: (0.4, 0.6), 60: (0.15, 0.25), 100: (0.1, 0.2), 150: (0.08, 0.15)}  # certified curve
    return H.decide(ex_vals, genes, splits, tau=tau, lr_curve=lr, mlm_health=1.0, both_strands_ok=True, seed=0), tau


def main():
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    bid = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
    sc = CaduceusScorer(MockCaduceus(base_ids=bid), tok, device="cpu", batch_size=16, window_length=W)

    print("== PLANTED-POSITIVE (real partner) — must QUALIFY ==")
    pos, tau_pos = run(sc, permute_partner=False)
    print(f"  tau(genuine permuted-null): {tau_pos:.4f}")
    for k in ("C_mean", "partner_spec_mean", "donor_spec_mean", "rho_partner", "joint_power", "long_range_certified", "gates", "VERDICT"):
        print(f"  {k}: {pos[k]}")

    print("\n== PERMUTED-PARTNER negative control — must NOT qualify on partner-specificity ==")
    neg, _ = run(sc, permute_partner=True)
    for k in ("partner_spec_mean", "partner_spec_ci", "VERDICT"):
        print(f"  {k}: {neg[k]}")

    ok = ("QUALIFIES" in pos["VERDICT"]) and ("QUALIFIES" not in neg["VERDICT"])
    print("\nPIPELINE:", "PASS ✓ (positive qualifies, permuted-partner does not)" if ok else "FAIL ✗")
    import sys; sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
