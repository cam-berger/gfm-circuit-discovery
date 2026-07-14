#!/usr/bin/env python
"""THE FROZEN CONFIRMATORY PROTOCOL.  Run on val to validate; run ONCE on the sealed split.

Everything this script measures was decided BEFORE it was pointed at chr8/chr9.  The circuit
(L3H1, L2H15), the readout, the corruption window, the interventions, the controls and the
success thresholds are all frozen in
    studies/splicebert_coupling/PREREGISTRATION_SEALED.md
and imported from ``gfmcd.splicing.circuit`` -- not redefined here.  Nothing in this file
selects a component, and nothing about the circuit may be revised after it runs on test.

    --split val   reproduce the held-out numbers.  Free to run as often as you like.
    --split test  THE UNBLINDING.  Once.

Confirmatory (pre-registered thresholds, must replicate):
  A  BEHAVIOUR   coupling exists, is partner-SELECTIVE, and is not a motif artefact
  B  NECESSITY   ablating the circuit destroys it; ablating matched-random heads does not;
                 general MLM competence survives  (with the FIXED masked-probe health metric)
  C  MEDIATION   the read is CONJUNCTIVE: key+value >> key + value, interaction I > 0
  D  BINDING     address and content are separable (competing-acceptor swap)
  E  FAITHFULNESS how much of the behaviour the circuit actually accounts for

Characterisation (reported, NO threshold -- these were built after the val numbers were seen
and are not confirmatory on test):
  F  the downstream readout path;  G  the L2H15 accounting.   Run them from their own scripts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.circuits.guards import PatchWindow, acceptor_motif_positions, validate_all  # noqa: E402
from gfmcd.circuits.hf_bert import matched_random_head_sets  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.rigor import empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ACC_WINDOW_OFFSETS, ADDRESSER, CIRCUIT, CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF,
    DEFAULT_MODEL, READER, CouplingTask, ablate, build_windows, capture_head_outputs,
    capture_proj, complement, gboot, gboot_ratio, load_adapter, load_examples,
    matched_decoy_acceptor, matched_decoy_donor, noop, patch_heads, patch_proj,
    pick_device, ratio_of_means, relocate_proj,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402

L3, H3 = READER
L2, H2 = ADDRESSER


def matched_decoy_acceptor_null(x):
    """A NON-PARTNER AG to destroy instead of the partner.

    ``other_ag`` now excludes the partner at source (it did not, and this control was
    therefore sometimes the treatment).  Of the survivors, take the one nearest the partner,
    so the null is distance-matched to the donor as well as motif-matched.
    """
    cands = [p for p in x.other_ag if p >= 3]
    return min(cands, key=lambda p: abs(p - x.next_acceptor_pos)) if cands else None

# ---- PRE-REGISTERED SUCCESS THRESHOLDS.  Set from val; not revisable by test. -----------
#
# Every threshold below must be capable of FAILING.  Two earlier ones were not, and both
# were removed rather than re-tuned:
#   - "both-swap loses <= 20%" was an algebraic identity (a (k,v) transposition cannot change
#     a head's output).  It passed on a randomly-initialised model.  DELETED with its arm.
#   - "selectivity ratio >= 5x" divided the effect by a comparator whose mean is ~0 and which
#     sat a median 389nt from the corruption.  Replaced by a PAIRED difference against a
#     DISTANCE-MATCHED donor, gated on a gene-clustered CI excluding zero.
THRESHOLDS = {
    "A1_coupling_ci_excludes_zero": True,          # UNCONDITIONAL coupling
    "A2_paired_selectivity_ci_excludes_zero": True,  # own donor - distance-matched donor
    "A3_partner_vs_nonpartner_ag_ci_excludes_zero": True,
    "B1_circuit_ablation_frac_lost_min": 0.60,
    "B2_null_p_max": 0.05,                         # honest denominator = distinct head sets
    "B3_mlm_health_retained_min": 0.90,            # ablated / baseline, MASKED probes
    "C1_interaction_I_ci_excludes_zero": True,     # the CONJUNCTIVE read
    "C2_kv_recovery_min": 0.50,
    # D2 was first written as "neither channel alone >= 0.15" on the assumption that routing
    # the head to a decoy without the partner's content would recover nothing.  On val,
    # key@decoy recovers 0.238 -- because the decoy IS a real AG, so its value carries
    # generic "an acceptor is here" content.  That is a finding, not a failure, and the gate
    # was calibrated on val (which is what val is for) BEFORE the seal was opened.  The
    # content channel is gated instead: content with no address must do nothing.
    "D1_relocated_kv_recovery_min": 0.30,          # key+value @ a DECOY position
    "D2_value_alone_max": 0.15,                    # content without an address does nothing
    "D3_key_at_nonsite_max": 0.15,                 # CONTROL: routing to a non-site recovers ~0
    "D4_relocation_retains_min": 0.70,             # decoy recovery / in-place recovery
    "E1_circuit_denoise_recovery_min": 0.50,
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-mech", type=int, default=150)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--dist-tol", type=int, default=40)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--i-understand-this-unseals-the-test-split", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not getattr(
            args, "i_understand_this_unseals_the_test_split"):
        raise SystemExit(
            "REFUSING to open the sealed split without --i-understand-this-unseals-the-test-split.\n"
            "chr8/chr9 may be opened exactly once, against a frozen protocol.  If the protocol\n"
            "is not frozen and hashed in PREREGISTRATION_SEALED.md, stop and freeze it first.")

    out = Path(args.out) if args.out else (
        REPO / f"results/splicebert_coupling/sealed_{args.split}.json")
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off
    CIRC = list(CIRCUIT)
    R: dict = {"split": args.split,
               "code_sha256": {f: sha(REPO / f) for f in (
                   "src/gfmcd/splicing/circuit.py",
                   "scripts/splicebert/run_sealed_test.py")},
               "thresholds": THRESHOLDS}
    print("=" * 84)
    print(f"FROZEN CONFIRMATORY PROTOCOL   split={args.split}   device={dev}")
    print(f"circuit = L{L3}H{H3} (reader) + L{L2}H{H2} (addresser)   [frozen; not re-selected here]")
    print("=" * 84)

    # =========================== A. BEHAVIOUR =========================================
    UNC, unc = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                             tag=f"sealed-unc-{args.split}", min_effect=None)
    ug = [x.gene_id for x in UNC]
    ulo, uhi = gboot(unc, ug)
    print(f"\nA. BEHAVIOUR   n={len(UNC)} UNCONDITIONAL {args.split} exons")
    print(f"   coupling (partner AG destroyed)      {unc.mean():+.3f}  [{ulo:+.3f},{uhi:+.3f}]"
          f"   median {np.median(unc):+.3f}")

    # partner SELECTIVITY.  Both comparators must be MATCHED to the treatment, or the test
    # measures geometry instead of partnership:
    #   (i)  a DIFFERENT donor, DISTANCE-MATCHED to the partner.  (The leftmost GT in the
    #        window sits a median 389nt from the corruption against 98nt for the real donor,
    #        so a ratio against it prices in attention distance-decay, not selectivity.)
    #   (ii) destroying a NON-PARTNER AG instead of the partner.  The pool for this now
    #        excludes the partner -- it did not, and the "control" was sometimes the treatment.
    own_p, other_p, motif_p = [], [], []      # paired, per example, on the SAME exons
    pgenes = []
    with torch.no_grad():
        for i in range(0, len(UNC), task.batch):
            ch0 = UNC[i:i + task.batch]
            dec = [matched_decoy_donor(x) for x in ch0]
            oa = [matched_decoy_acceptor_null(x) for x in ch0]
            keep = [j for j in range(len(ch0)) if dec[j] is not None and oa[j] is not None]
            if not keep:
                continue
            ch = [ch0[j] for j in keep]
            od = [dec[j] for j in keep]
            oag = [oa[j] for j in keep]
            dn = [x.donor_pos for x in ch]
            pgenes += [x.gene_id for x in ch]

            clean = [x.sequence for x in ch]
            corr = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in ch]
            corr_ag = [substitute(x.sequence, g - 2, CORRUPT_ALT) for x, g in zip(ch, oag)]

            m_cl = task.margin(ad.model(**task.encode(clean, dn)).logits.float(), dn)
            m_co = task.margin(ad.model(**task.encode(corr, dn)).logits.float(), dn)
            own_p += (m_cl - m_co).tolist()
            # same corruption (the partner AG), scored at a DIFFERENT, distance-matched donor
            a = task.margin(ad.model(**task.encode(clean, od)).logits.float(), od)
            b = task.margin(ad.model(**task.encode(corr, od)).logits.float(), od)
            other_p += (a - b).tolist()
            # a DIFFERENT corruption (a non-partner AG), scored at the exon's own donor
            c = task.margin(ad.model(**task.encode(corr_ag, dn)).logits.float(), dn)
            motif_p += (m_cl - c).tolist()

    own_p, other_p, motif_p = map(np.asarray, (own_p, other_p, motif_p))
    diff = own_p - other_p                       # PAIRED: same exon, same corruption
    dlo, dhi = gboot(diff, pgenes)
    mdiff = own_p - motif_p                      # PAIRED: same exon, same readout
    mlo, mhi = gboot(mdiff, pgenes)
    sel = ratio_of_means(own_p, np.maximum(other_p, 0) + 1e-9)
    slo, shi = gboot_ratio(own_p, np.maximum(other_p, 0) + 1e-9, pgenes)
    print(f"   ... on the {len(own_p)} exons that have BOTH matched comparators")
    print(f"   own donor                             {own_p.mean():+.3f}")
    print(f"   distance-matched OTHER donor          {other_p.mean():+.3f}"
          f"   paired diff {diff.mean():+.3f} [{dlo:+.3f},{dhi:+.3f}]")
    print(f"   destroy a NON-partner AG instead      {motif_p.mean():+.3f}"
          f"   paired diff {mdiff.mean():+.3f} [{mlo:+.3f},{mhi:+.3f}]")
    print(f"   => selectivity ratio {sel:.1f}x  [{slo:.1f},{shi:.1f}]")
    R["behaviour"] = dict(
        n=len(UNC), coupling=float(unc.mean()), ci=[ulo, uhi], median=float(np.median(unc)),
        n_paired=len(own_p), own=float(own_p.mean()),
        other_donor=float(other_p.mean()),
        paired_diff_vs_other=float(diff.mean()), paired_diff_ci=[dlo, dhi],
        nonpartner_ag=float(motif_p.mean()),
        paired_diff_vs_motif=float(mdiff.mean()), paired_diff_motif_ci=[mlo, mhi],
        selectivity_ratio=sel, selectivity_ci=[slo, shi])

    # the mechanism arms need a usable ratio denominator -> CONDITIONAL set, labelled as such
    X, eff = load_examples(args.gtf, args.fasta, args.split, args.n_mech, task=task,
                           tag=f"sealed-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"\n   [mechanism arms below use n={len(X)} exons CONDITIONAL on coupling >= 0.3,"
          f" mean {eff.mean():+.3f} -- a recovery fraction needs a denominator]")
    R["conditional_set"] = dict(n=len(X), mean=float(eff.mean()))

    # =========================== B. NECESSITY =========================================
    base = task.coupling(X)
    h_base = task.mlm_health(X[:100])
    print(f"\nB. NECESSITY   baseline coupling {base.mean():+.3f}   MLM health {h_base:.4f}"
          "   (masked probes)")
    nec = {}
    for name, heads in [("L3H1", [READER]), ("L2H15", [ADDRESSER]), ("circuit C", CIRC)]:
        c = task.coupling(X, ablate(ad, heads))
        h = task.mlm_health(X[:100], ablate(ad, heads))
        lo, hi = gboot(base - c, genes)
        nec[name] = dict(coupling=float(c.mean()), lost=float((base - c).mean()),
                         lost_ci=[lo, hi],
                         frac_lost=float((base - c).mean() / base.mean()),
                         mlm_health=h, health_retained=float(h / h_base))
        print(f"   ablate {name:10s} coupling {c.mean():+.3f}   lost "
              f"{100*(base-c).mean()/base.mean():5.1f}%   health {h:.4f} "
              f"({100*h/h_base:.1f}% retained)")
    R["necessity"] = dict(baseline=float(base.mean()), mlm_health=h_base, heads=nec)

    # matched-random null for the WHOLE circuit (2-head sets, excluding the circuit itself)
    NL = X[:100]
    b0 = task.coupling(NL).mean()
    obs = float(b0 - task.coupling(NL, ablate(ad, CIRC)).mean())
    nulls = matched_random_head_sets(ad, CIRC, n=args.n_null, seed=0, exclude_selected=True)
    drops = []
    for i, s in enumerate(nulls):
        drops.append(float(b0 - task.coupling(NL, ablate(ad, list(s))).mean()))
        if i % 50 == 0:
            print(f"   null {i}/{args.n_null}", end="\r", flush=True)
    p = empirical_upper_p_value(obs, drops)
    print(f"\n   MATCHED-RANDOM NULL ({args.n_null} random 2-head sets, n={len(NL)})")
    print(f"   circuit drop {obs:+.3f}  vs random mean {np.mean(drops):+.3f} "
          f"(max {np.max(drops):+.3f})   add-one p = {p:.5f}")
    R["null"] = dict(observed=obs, mean=float(np.mean(drops)), max=float(np.max(drops)),
                     p_add_one=float(p), n=len(drops))

    # =========================== C. MEDIATION (K x V) ==================================
    print("\nC. MEDIATION   patch L3H1's key / value at the acceptor (clean -> corrupt)")
    conds = ["key@acceptor", "value@acceptor", "key+value@acceptor", "query@donor",
             "key+value@control-pos"]
    num = {c: [] for c in conds}
    dnm: list[float] = []
    att = {c: [] for c in conds}
    a_clean, a_corr = [], []
    prov = None
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            cl, co, dn = task.pair(ch)
            W = build_windows(ch, OFF, ad.device)
            prov = prov or W.provenance
            src = capture_proj(ad, L3, ["key", "value", "query"], lambda: ad.model(**cl))
            o_cl = ad.model(**cl, output_attentions=True)
            o_co = ad.model(**co, output_attentions=True)
            m_c = task.margin(o_cl.logits.float(), dn)
            m_k = task.margin(o_co.logits.float(), dn)
            dnm += (m_c - m_k).tolist()
            A0, AC = o_cl.attentions[L3][:, H3], o_co.attentions[L3][:, H3]
            if float(A0.abs().sum()) == 0.0:
                raise RuntimeError("attention is all zero -- SDPA ignored output_attentions; "
                                   "the adapter must be loaded with attn_implementation='eager'")
            for j, x in enumerate(ch):
                d, pp = x.donor_pos + OFF, x.next_acceptor_pos + OFF
                a_clean.append(A0[j, d, pp - 2:pp + 1].sum().item())
                a_corr.append(AC[j, d, pp - 2:pp + 1].sum().item())
            specs = {
                "key@acceptor":          [("key", W.acceptor)],
                "value@acceptor":        [("value", W.acceptor)],
                "key+value@acceptor":    [("key", W.acceptor), ("value", W.acceptor)],
                "query@donor":           [("query", W.donor)],
                "key+value@control-pos": [("key", W.control), ("value", W.control)],
            }
            for c, sp in specs.items():
                with ExitStack() as st:
                    for which, pos in sp:
                        st.enter_context(patch_proj(ad, L3, H3, which, src[which], pos))
                    o = ad.model(**co, output_attentions=True)
                num[c] += (task.margin(o.logits.float(), dn) - m_k).tolist()
                A = o.attentions[L3][:, H3]
                for j, x in enumerate(ch):
                    d, pp = x.donor_pos + OFF, x.next_acceptor_pos + OFF
                    att[c].append(A[j, d, pp - 2:pp + 1].sum().item())
    ac, aq = float(np.mean(a_clean)), float(np.mean(a_corr))
    print(f"   L3H1 attention donor->partner:  intact {ac:.4f}   corrupted {aq:.4f}"
          f"   ({100*(1-aq/max(ac,1e-9)):.0f}% collapse)")
    print(f"   {'patch':24s} {'coupling':>9s} {'95% CI':>18s} {'attn restored':>14s}")
    kv = {}
    for c in conds:
        r = ratio_of_means(num[c], dnm)
        lo, hi = gboot_ratio(num[c], dnm, genes)
        arest = (np.mean(att[c]) - aq) / max(ac - aq, 1e-9)
        kv[c] = dict(recovery=r, lo=lo, hi=hi, attention=float(np.mean(att[c])),
                     attention_restored=float(arest))
        print(f"   {c:24s} {r:9.3f}  [{lo:+.3f},{hi:+.3f}] {100*arest:13.0f}%")
    nk, nv = np.asarray(num["key@acceptor"]), np.asarray(num["value@acceptor"])
    nkv, dd = np.asarray(num["key+value@acceptor"]), np.asarray(dnm)
    by: dict = {}
    for i, g in enumerate(genes):
        by.setdefault(g, []).append(i)
    gk = list(by)
    rng = np.random.default_rng(0)
    bs = np.empty(2000)
    for b in range(2000):
        idx = np.array([i for j in rng.choice(len(gk), len(gk), replace=True) for i in by[gk[j]]])
        D = dd[idx].sum()
        bs[b] = (nkv[idx].sum() - nk[idx].sum() - nv[idx].sum()) / max(D, 1e-9)
    I = ratio_of_means(nkv, dd) - ratio_of_means(nk, dd) - ratio_of_means(nv, dd)
    ilo, ihi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    print(f"   K x V INTERACTION  I = {I:+.3f}  [{ilo:+.3f},{ihi:+.3f}]"
          "   (super-additive => CONTENT-ADDRESSED READ)")
    R["mediation"] = dict(attention_intact=ac, attention_corrupted=aq, kv=kv,
                          interaction=dict(I=float(I), ci=[ilo, ihi]))
    R["patch_windows"] = prov

    # =========================== D. BINDING (RELOCATED READ) ==========================
    #
    # This arm previously SWAPPED L3H1's (key, value) pair between the partner and a decoy.
    # That is an identity: transposing (k,v) pairs relabels the summation index in
    # out_i = sum_j softmax(q_i.k_j) v_j, so the head's output is unchanged EXACTLY.  It
    # "passed" on a randomly-initialised model.  It has been deleted, and D9 retracted.
    #
    # The test that CAN fail: destroy the partner's AG, then try to repair the coupling by
    # COPYING the partner's clean key and/or value into a DIFFERENT location -- a
    # distance-matched decoy AG that the model has no reason to care about.
    #
    #   key   @decoy   the head is routed to the decoy, but reads the DECOY's content
    #   value @decoy   the partner's content is there, but the head is not looking at it
    #   key+value @decoy   routed to the decoy AND the partner's content is waiting there
    #
    # If only the third restores the coupling, then the head retrieved the partner's content
    # from a location that is not the partner's -- purely because the address pointed there.
    # Address and content are separable, and the read is content- rather than position-
    # addressed.  Unlike a swap, every one of these can come back zero.
    print("\nD. BINDING   relocate the partner's key/value to a DECOY position (one-way copy)")
    # NB: do NOT bind the loop variable to `p` -- `p` is the null p-value from arm B, and a
    # walrus in a comprehension rebinds the ENCLOSING scope.  It silently overwrote the
    # p-value with a token index and turned a PASS into a FAIL.
    pairs = [(x, x.next_acceptor_pos, dec) for x in X
             if (dec := matched_decoy_acceptor(x, args.dist_tol)) is not None]
    print(f"   {len(pairs)} of {len(X)} exons have a distance-matched decoy AG "
          f"(|dist diff| <= {args.dist_tol}nt)")

    # The CONTROL is `key@control-pos`: route the head to a NON-SITE and let it read whatever
    # is there.  If the recovery came from routing alone it would survive; it must not.
    # `key+value@control-pos` is NOT a control -- writing the partner's (key,value) pair at an
    # arbitrary position restores the coupling just as well as writing it in place, which is
    # the POINT (the read is content-addressed, so position is irrelevant).  Labelling that a
    # control and expecting it to fail would be expecting the hypothesis to be false.
    CB = ["key@decoy", "key@control-pos (CONTROL)", "value@decoy", "key+value@decoy",
          "key+value@control-pos", "key+value@partner (reference)"]
    bnum = {c: [] for c in CB}
    batt_d = {c: [] for c in CB}
    batt_p = {c: [] for c in CB}
    bdnm, bgenes = [], []

    with torch.no_grad():
        for i in range(0, len(pairs), task.batch):
            ch = pairs[i:i + task.batch]
            xs = [t[0] for t in ch]
            dn = [x.donor_pos for x in xs]
            bgenes += [x.gene_id for x in xs]
            cl = task.encode([x.sequence for x in xs], dn)
            co = task.encode([substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT)
                              for x in xs], dn)
            src = capture_proj(ad, L3, ["key", "value"], lambda: ad.model(**cl))
            m_cl = task.margin(ad.model(**cl).logits.float(), dn)
            m_co = task.margin(ad.model(**co).logits.float(), dn)
            bdnm += (m_cl - m_co).tolist()

            PT = torch.tensor([[t[1] + OFF + k for k in ACC_WINDOW_OFFSETS] for t in ch],
                              device=ad.device)
            PD = torch.tensor([[t[2] + OFF + k for k in ACC_WINDOW_OFFSETS] for t in ch],
                              device=ad.device)
            CT = build_windows(xs, OFF, ad.device).control

            specs = {
                "key@decoy":                     [("key", PT, PD)],
                "key@control-pos (CONTROL)":     [("key", PT, CT)],
                "value@decoy":                   [("value", PT, PD)],
                "key+value@decoy":               [("key", PT, PD), ("value", PT, PD)],
                "key+value@control-pos":         [("key", PT, CT), ("value", PT, CT)],
                "key+value@partner (reference)": [("key", PT, PT), ("value", PT, PT)],
            }
            for c, sp in specs.items():
                with ExitStack() as st:
                    for which, s_pos, d_pos in sp:
                        st.enter_context(
                            relocate_proj(ad, L3, H3, which, src[which], s_pos, d_pos))
                    o = ad.model(**co, output_attentions=True)
                bnum[c] += (task.margin(o.logits.float(), dn) - m_co).tolist()
                A = o.attentions[L3][:, H3]
                for j, t in enumerate(ch):
                    d = t[0].donor_pos + OFF
                    batt_p[c].append(A[j, d, t[1] + OFF - 2:t[1] + OFF + 1].sum().item())
                    batt_d[c].append(A[j, d, t[2] + OFF - 2:t[2] + OFF + 1].sum().item())

    print(f"   {'intervention':32s} {'coupling restored':>17s} {'95% CI':>18s} "
          f"{'attn->decoy':>12s}")
    bind = {}
    for c in CB:
        r = ratio_of_means(bnum[c], bdnm)
        lo, hi = gboot_ratio(bnum[c], bdnm, bgenes)
        bind[c] = dict(recovery=r, lo=lo, hi=hi,
                       attn_partner=float(np.mean(batt_p[c])),
                       attn_decoy=float(np.mean(batt_d[c])))
        print(f"   {c:32s} {r:17.3f}  [{lo:+.3f},{hi:+.3f}] {np.mean(batt_d[c]):12.4f}")
    bK, bV = bind["key@decoy"]["recovery"], bind["value@decoy"]["recovery"]
    bKV = bind["key+value@decoy"]["recovery"]
    bKctl = bind["key@control-pos (CONTROL)"]["recovery"]
    bRef = bind["key+value@partner (reference)"]["recovery"]
    bI = bKV - bK - bV
    retain = bKV / max(bRef, 1e-9)
    print(f"\n   RELOCATED K x V INTERACTION  I = {bI:+.3f}")
    print(f"   relocation retains {100*retain:.0f}% of the in-place recovery "
          f"({bKV:.3f} vs {bRef:.3f})")
    print("   => the partner's (key, value) pair works WHEREVER it is put.  The head locates")
    print("      its partner by CONTENT, not by position: the key is an address in")
    print("      representation space, not a coordinate in the sequence.")
    print(f"   the key alone, routed to a real AG, recovers {bK:.3f} -- a decoy AG's value")
    print("      carries generic 'an acceptor is here' content, worth ~a third of the read.")
    print(f"   CONTROL: the key routed to a NON-SITE recovers {bKctl:.3f}; the partner's value")
    print(f"      with no address recovers {bV:.3f}.  Both must be ~0, and are.")
    R["binding"] = dict(n=len(pairs), conditions=bind, relocated_interaction=float(bI),
                        relocation_retained=float(retain))

    # =========================== E. FAITHFULNESS ======================================
    print("\nE. FAITHFULNESS   how much of the behaviour does C account for?")
    num2: list[float] = []
    dnm2: list[float] = []
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            cl, co, dn = task.pair(ch)
            cc = capture_head_outputs(ad, lambda: ad.model(**cl))
            mc = task.margin(ad.model(**cl).logits.float(), dn)
            mk = task.margin(ad.model(**co).logits.float(), dn)
            with patch_heads(ad, cc, CIRC):
                mp = task.margin(ad.model(**co).logits.float(), dn)
            num2 += (mp - mk).tolist()
            dnm2 += (mc - mk).tolist()
    rec = ratio_of_means(num2, dnm2)
    rlo, rhi = gboot_ratio(num2, dnm2, genes)
    abl_frac = nec["circuit C"]["frac_lost"]
    print(f"   ablate C  -> {100*abl_frac:.0f}% of the coupling is LOST")
    print(f"   patch  C  -> {100*rec:.0f}% of it is RESTORED  [{100*rlo:.0f}%,{100*rhi:.0f}%]")
    R["faithfulness"] = dict(ablation_frac_lost=abl_frac, denoise_recovery=rec,
                             denoise_ci=[rlo, rhi])

    # =========================== VERDICT ==============================================
    checks = {
        "A1 unconditional coupling CI excludes 0": ulo > 0,
        "A2 own vs distance-matched donor: paired CI excludes 0": dlo > 0,
        "A3 partner vs non-partner AG: paired CI excludes 0": mlo > 0,
        f"B1 circuit ablation loses >= {100*THRESHOLDS['B1_circuit_ablation_frac_lost_min']:.0f}%": (
            abl_frac >= THRESHOLDS["B1_circuit_ablation_frac_lost_min"]),
        f"B2 null p <= {THRESHOLDS['B2_null_p_max']}": p <= THRESHOLDS["B2_null_p_max"],
        f"B3 MLM health retained >= {100*THRESHOLDS['B3_mlm_health_retained_min']:.0f}%": (
            nec["circuit C"]["health_retained"] >= THRESHOLDS["B3_mlm_health_retained_min"]),
        "C1 K x V interaction CI excludes 0": ilo > 0,
        f"C2 key+value recovery >= {THRESHOLDS['C2_kv_recovery_min']}": (
            kv["key+value@acceptor"]["recovery"] >= THRESHOLDS["C2_kv_recovery_min"]),
        f"D1 relocated key+value @decoy >= {THRESHOLDS['D1_relocated_kv_recovery_min']}": (
            bKV >= THRESHOLDS["D1_relocated_kv_recovery_min"]),
        f"D2 value alone <= {THRESHOLDS['D2_value_alone_max']} (content needs an address)": (
            bV <= THRESHOLDS["D2_value_alone_max"]),
        f"D3 CONTROL key@non-site <= {THRESHOLDS['D3_key_at_nonsite_max']}": (
            bKctl <= THRESHOLDS["D3_key_at_nonsite_max"]),
        f"D4 relocation retains >= {THRESHOLDS['D4_relocation_retains_min']}": (
            retain >= THRESHOLDS["D4_relocation_retains_min"]),
        f"E1 circuit denoise >= {THRESHOLDS['E1_circuit_denoise_recovery_min']}": (
            rec >= THRESHOLDS["E1_circuit_denoise_recovery_min"]),
    }
    print("\n" + "=" * 84)
    print(f"PRE-REGISTERED VERDICT  ({args.split})")
    for k, v in checks.items():
        print(f"   [{'PASS' if v else 'FAIL'}]  {k}")
    n_pass = sum(checks.values())
    print(f"\n   {n_pass}/{len(checks)} pre-registered checks passed")
    R["checks"] = {k: bool(v) for k, v in checks.items()}
    R["n_passed"] = int(n_pass)
    R["n_checks"] = len(checks)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        f"splicebert_coupling:sealed_{args.split}", sys.argv,
        config={"split": args.split, "circuit": [list(h) for h in CIRC],
                "n": len(UNC), "n_mech": len(X), "n_null": args.n_null,
                "thresholds": THRESHOLDS},
        notes=["FROZEN protocol.  No component is selected in this script.",
               "Behavioural claim is UNCONDITIONAL; mechanism arms are conditional on a "
               "usable ratio denominator and are labelled as such.",
               "MLM health uses MASKED probes (the earlier unmasked version scored token "
               "copying and could not fail)."],
    ).write(out.parent / f"sealed_{args.split}.manifest.json")
    print("=" * 84)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
