#!/usr/bin/env python
"""Donor-graph experiment 3 -- causal acceptor-address transplantation into L3H1.

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_ACCEPTOR_ADDRESS.md.

Is L3H1's read at the intron-mate acceptor specific to the model's OWN partner over a
nucleotide-, distance-, and reading-phase-matched decoy (differing only in partner identity);
and is the partner's key+value a TRANSPLANTABLE causal carrier of the coupling?

The run writes a frozen preflight artifact FIRST and stops if it fails; only then are the
descriptive preferences (§3) and the causal gates (§4) computed.  Every claim stays at
OV/mediation + Behaviour -- no "position-independent / content-addressed pointer" (R15).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file  # noqa: E402
from gfmcd.rigor import empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.address import (  # noqa: E402
    attn_to_window, build_pairs, match_audit, qk_logit_to_window, windows,
)
from gfmcd.circuits.hf_bert import matched_random_head_sets  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask,
    ablate, build_windows, capture_proj, capture_writes, clamp_write, gboot, gboot_ratio,
    load_adapter, load_examples, pick_device, ratio_of_means, relocate_proj,
)
from gfmcd.splicing.coupling import CouplingConfig, substitute  # noqa: E402

L3, H3 = READER
L2, H2 = ADDRESSER
MLP2 = [(L2, "mlp")]
SPLIT_CHROMS = {"val": ("chr11", "chr16", "chr19"), "test": ("chr8", "chr9")}


# --------------------------------------------------------------------------------------
# Per-batch readout of L3H1's donor-query preference for a+ over a-.
# --------------------------------------------------------------------------------------
def preference(ad, task, enc, donor_tok, wplus, wminus, *, ctx=None):
    """Return per-example (qk_plus, qk_minus, a_plus_mass, a_minus_mass) for L3H1.

    ``enc`` is an already-built encoding (donor masked).  ``ctx`` is an optional context
    factory (e.g. ``ablate([ADDRESSER])``) applied around BOTH forwards.
    """
    Dh = ad.hidden_size // ad.num_heads
    cm = ctx() if ctx is not None else nullcontext()
    with cm:
        proj = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc))
    cm2 = ctx() if ctx is not None else nullcontext()
    with cm2:
        o = ad.model(**enc, output_attentions=True)
    attn = o.attentions[L3][:, H3]                                  # [B,S,S]
    qk_p = qk_logit_to_window(proj["query"], proj["key"], H3, Dh, donor_tok, wplus)
    qk_m = qk_logit_to_window(proj["query"], proj["key"], H3, Dh, donor_tok, wminus)
    a_p = attn_to_window(attn, donor_tok, wplus)
    a_m = attn_to_window(attn, donor_tok, wminus)
    return (qk_p.cpu().numpy(), qk_m.cpu().numpy(),
            a_p.cpu().numpy(), a_m.cpu().numpy())


def sa_mean(ad, task, pairs, off, *, ctx=None, batch=8):
    """Mean over all pairs of S_A = A(d,W+) - A(d,W-) under an optional ctx (for the null)."""
    vals: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            ch = pairs[i:i + batch]
            exs = [p.example for p in ch]
            enc = task.encode([x.sequence for x in exs], [x.donor_pos for x in exs])
            dtok = torch.tensor([x.donor_pos + off for x in exs], device=ad.device)
            wplus = windows([p.a_plus for p in ch], off, ad.device)
            wminus = windows([p.a_minus for p in ch], off, ad.device)
            _, _, ap, am = preference(ad, task, enc, dtok, wplus, wminus, ctx=ctx)
            vals += (ap - am).tolist()
    return float(np.mean(vals))


def corrupt_at(task, exs, pos_list):
    """Encoding with the donor masked and the AG at ``pos_list[i]-2`` destroyed (CC)."""
    seqs = [substitute(x.sequence, p - 2, CORRUPT_ALT) for x, p in zip(exs, pos_list)]
    return task.encode(seqs, [x.donor_pos for x in exs])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=260, help="target phase-matched pairs")
    ap.add_argument("--pool", type=int, default=3000, help="eligible examples screened for decoys")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tol", type=int, default=40, help="generic-AG decoy donor-distance match")
    ap.add_argument("--real-tol", type=int, default=60,
                    help="real-acceptor decoy donor-distance match (sparser -> looser)")
    ap.add_argument("--null-heads", type=int, default=95,
                    help="C-Address matched-random single-head null size (95 = enumerate)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-coding", action="store_true", help="skip the CDS coding covariate")
    ap.add_argument("--force", action="store_true",
                    help="compute §3/§4 even if preflight fails (DIAGNOSTIC ONLY, non-confirmatory)")
    ap.add_argument("--out", default="results/splicebert_coupling/acceptor_address.json")
    args = ap.parse_args()

    out = REPO / args.out
    pf_path = out.parent / ("preflight_" + out.name)

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)          # eager REQUIRED (attention read)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off

    # ----- pairs -----------------------------------------------------------------------
    Xpool, _ = load_examples(args.gtf, args.fasta, args.split, args.pool, task=task,
                             tag=f"addr-{args.split}", min_effect=None)
    cds = None
    if not args.skip_coding:
        try:
            from gfmcd.splicing.phase import load_cds_frames
            cds = load_cds_frames(args.gtf, allowed_chromosomes=SPLIT_CHROMS.get(args.split))
        except Exception as e:                              # coding covariate is optional
            print(f"  (coding covariate unavailable: {e})")
    avail = build_pairs(Xpool, tol=args.tol, real_tol=args.real_tol, cds_lookup=cds)
    pairs = avail[:args.n]
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(avail)} phase-matched pairs from {len(Xpool)} eligible; "
                           "raise --pool or --tol.")
    if len(pairs) < args.n:
        print(f"  WARNING: only {len(avail)} pairs available (< target {args.n}); "
              "raise --pool for a confirmatory run.")
    genes = [p.gene_id for p in pairs]
    strands = {p.strand for p in pairs}
    print(f"{len(pairs)} pairs (of {len(avail)} available, {len(Xpool)} eligible screened)   "
          f"strands={sorted(strands)}   device={dev}")

    # ----- preflight (written FIRST; stop on failure) ----------------------------------
    audit = match_audit(pairs)
    pf = preflight(ad, task, pairs, off, args)
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    if not pf["PASS"]:
        if not args.force:
            print("  *** preflight FAILED -- not computing §3/§4 numbers (prereg §6).")
            return
        print("  *** preflight FAILED but --force set: §3/§4 numbers below are DIAGNOSTIC "
              "ONLY and must NOT be reported as confirmatory.")

    # ----- §3 preferences + §4 transplant, per batch -----------------------------------
    S_QK, S_A, S_Y = [], [], []
    reco_num: dict = {a: [] for a in
                      ["kv@decoy", "k@decoy", "v@decoy", "kv@nonsite", "kv@a+ (injection)"]}
    reco_dnm: list = []
    redirect: list = []                                    # C-Redirect: relocation-induced ΔA at a-
    # C-Address per-example intact S_A (for the null obs drop) accumulates via S_A above.
    with torch.no_grad():
        for i in range(0, len(pairs), args.batch):
            ch = pairs[i:i + args.batch]
            exs = [p.example for p in ch]
            dn = [x.donor_pos for x in exs]
            dtok = torch.tensor([d + off for d in dn], device=ad.device)
            wplus = windows([p.a_plus for p in ch], off, ad.device)
            wminus = windows([p.a_minus for p in ch], off, ad.device)
            W = build_windows(exs, off, ad.device)
            wctl = W.control

            clean = task.encode([x.sequence for x in exs], dn)
            corr = corrupt_at(task, exs, [p.a_plus for p in ch])       # destroy partner a+

            # §3 preference (intact, clean)
            qk_p, qk_m, a_p, a_m = preference(ad, task, clean, dtok, wplus, wminus)
            S_QK += (qk_p - qk_m).tolist()
            S_A += (a_p - a_m).tolist()

            # S_Y: destroy a+ vs destroy a- (behavioural preference)
            m_cl = task.margin(ad.model(**clean).logits.float(), dn).cpu().numpy()
            m_co = task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
            corr_m = corrupt_at(task, exs, [p.a_minus for p in ch])    # destroy decoy a-
            m_com = task.margin(ad.model(**corr_m).logits.float(), dn).cpu().numpy()
            S_Y += ((m_cl - m_co) - (m_cl - m_com)).tolist()           # Y(a+) - Y(a-)

            # §4 C-Transplant: relocate CLEAN a+ key/value into a- on the CORRUPT run
            src = capture_proj(ad, L3, ["key", "value"], lambda: ad.model(**clean))
            reco_dnm += (m_cl - m_co).tolist()

            def reloc(whichs, dst):
                st = ExitStack()
                for w in whichs:
                    st.enter_context(relocate_proj(ad, L3, H3, w, src[w], wplus, dst))
                return st

            arms = {
                "kv@decoy": (["key", "value"], wminus),
                "k@decoy": (["key"], wminus),
                "v@decoy": (["value"], wminus),
                "kv@nonsite": (["key", "value"], wctl),
                "kv@a+ (injection)": (["key", "value"], wplus),
            }
            for name, (whichs, dst) in arms.items():
                with reloc(whichs, dst):
                    m = task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
                reco_num[name] += (m - m_co).tolist()

            # C-Redirect: does relocating the partner KEY into a- RAISE L3H1's donor attention
            # to a-?  Difference-in-differences at the DECOY window isolates the relocated key
            # from the a+-destruction confound (corruption already depresses A(d,W+), so the old
            # W-−W+ contrast could be positive with no relocation at all).
            o0 = ad.model(**corr, output_attentions=True)          # corrupt, NO relocation
            a_m0 = attn_to_window(o0.attentions[L3][:, H3], dtok, wminus).cpu().numpy()
            with relocate_proj(ad, L3, H3, "key", src["key"], wplus, wminus):
                o1 = ad.model(**corr, output_attentions=True)      # corrupt + key relocated
                a_m1 = attn_to_window(o1.attentions[L3][:, H3], dtok, wminus).cpu().numpy()
            redirect += (a_m1 - a_m0).tolist()                     # relocation-induced rise at a-
            print(f"  {min(i+args.batch, len(pairs))}/{len(pairs)}", end="\r", flush=True)
    print()

    # ----- C-Address: L2H15 / MLP2 causal contribution to S_A, with matched-random null --
    sa_intact = float(np.mean(S_A))
    sa_l2abl = sa_mean(ad, task, pairs, off, ctx=ablate(ad, [ADDRESSER]), batch=args.batch)
    # MLP2 clamped to its L2H15-ablated value (isolates MLP2's carriage of L2H15's signal)
    mlp2_drop = mlp2_contribution(ad, task, pairs, off, sa_intact, batch=args.batch)
    null_heads = [hs[0] for hs in matched_random_head_sets(ad, [ADDRESSER], n=args.null_heads,
                                                           exclude_selected=True)]
    # S_A = attentions[3] depends only on the residual feeding layer 3, i.e. layers 0-2.  Heads
    # in layers >= 3 are structurally inert for this statistic (exactly 0 drop) and would
    # zero-inflate the null and deflate the add-one p.  Restrict to the live upstream family.
    null_heads = [h for h in null_heads if h[0] < L3]
    null_drops = [sa_intact - sa_mean(ad, task, pairs, off, ctx=ablate(ad, [h]),
                                      batch=args.batch) for h in null_heads]
    obs_drop = sa_intact - sa_l2abl
    addr_p = empirical_upper_p_value(obs_drop, null_drops)

    # ----- secondary: preference vs a non-partner REAL acceptor (strength-controlled) ---
    real_pairs = [p for p in pairs if p.a_real is not None]
    S_A_real, genes_real = [], []
    if len(real_pairs) >= 8:
        with torch.no_grad():
            for i in range(0, len(real_pairs), args.batch):
                ch = real_pairs[i:i + args.batch]
                exs = [p.example for p in ch]
                dn = [x.donor_pos for x in exs]
                enc = task.encode([x.sequence for x in exs], dn)
                dtok = torch.tensor([d + off for d in dn], device=ad.device)
                wplus = windows([p.a_plus for p in ch], off, ad.device)
                wreal = windows([p.a_real for p in ch], off, ad.device)
                _, _, ap, ar = preference(ad, task, enc, dtok, wplus, wreal)
                S_A_real += (ap - ar).tolist()
                genes_real += [p.gene_id for p in ch]

    # ----- statistics ------------------------------------------------------------------
    def ci(v, g=None):
        return gboot(v, g if g is not None else genes)

    def rec(name):
        v = ratio_of_means(reco_num[name], reco_dnm)
        lo, hi = gboot_ratio(reco_num[name], reco_dnm, genes)
        return dict(recovery=v, lo=lo, hi=hi)

    transplant = {a: rec(a) for a in reco_num}
    kv = transplant["kv@decoy"]
    # R15 position-insensitivity: key+value recovers ~equally at decoy / non-site / a+.
    kv_vals = [transplant["kv@decoy"]["recovery"], transplant["kv@nonsite"]["recovery"],
               transplant["kv@a+ (injection)"]["recovery"]]
    R = {
        "n_pairs": len(pairs), "strands": sorted(strands),
        "match_audit": audit,
        "preference": {
            "S_QK": {"mean": float(np.mean(S_QK)), "ci": ci(S_QK)},
            "S_A": {"mean": float(np.mean(S_A)), "ci": ci(S_A)},
            "S_Y": {"mean": float(np.mean(S_Y)), "ci": ci(S_Y)},
            "S_A_vs_real_acceptor": ({"mean": float(np.mean(S_A_real)),
                                      "ci": ci(S_A_real, genes_real), "n": len(S_A_real),
                                      "n_real_decoys": len(real_pairs),
                                      "distance_confound": audit["real_acceptor_distance"]}
                                     if S_A_real else
                                     {"n": 0, "n_real_decoys": len(real_pairs),
                                      "note": f"{len(real_pairs)} real-acceptor decoys (< 8 needed)"}),
        },
        "C_Redirect": {"S_A_prime": float(np.mean(redirect)), "ci": ci(redirect),
                       "note": "relocation-induced RISE in a- attention (difference-in-differences "
                               "on the corrupt run: A(d,a-)[+key] - A(d,a-)[no reloc])"},
        "C_Dissociate": {
            "kv@decoy": kv, "k@decoy": transplant["k@decoy"], "v@decoy": transplant["v@decoy"],
            "kv_gt_k_R15_expected": bool(kv["recovery"] > transplant["k@decoy"]["recovery"]),
            "note": "GATED on the falsifiable legs (k lo>0, k>v, v<=0.15); kv>k is R15-expected "
                    "content delivery (kv is position-forced, see C_R15) and is reported not gated"},
        "C_R15_position_insensitivity": {
            "kv_recovery_decoy_nonsite_ainjection": kv_vals,
            "spread": float(np.max(kv_vals) - np.min(kv_vals)),
            "interpretation": "key+value recovers ~equally regardless of position: R15-forced, "
                              "NOT content-addressing; no pointer claim licensed."},
        "C_Address": {
            "S_A_intact": sa_intact, "S_A_L2H15_ablated": sa_l2abl, "drop_L2H15": obs_drop,
            "retained_fraction_L2H15": (sa_l2abl / sa_intact) if sa_intact else float("nan"),
            "drop_MLP2": mlp2_drop, "null_head_drops_mean": float(np.mean(null_drops)),
            "null_head_drops_max": float(np.max(null_drops)), "n_null_heads": len(null_heads),
            "add_one_p": addr_p,
            "note": "does L2H15 dominate the matched-decoy preference (distinct from its role "
                    "vs the corrupt partner in l2_accounting)? reported, not gated."},
        "preflight_PASS": pf["PASS"],
        "n_genes": len(set(genes)),
        "meets_power_target": bool(len(pairs) >= 250),
        "confirmatory": bool(pf["PASS"] and len(pairs) >= 250),
        "tolerances": {"generic": args.tol, "real": args.real_tol},
    }

    # ----- verdict (frozen §4; overall = G_Preference and C_Redirect and C_Dissociate; G_Match
    #        is a reported prerequisite and the R15-forced kv transplant is reported, not gated) --
    dissociate = bool(transplant["k@decoy"]["lo"] > 0
                      and transplant["k@decoy"]["recovery"] > transplant["v@decoy"]["recovery"]
                      and transplant["v@decoy"]["recovery"] <= 0.15)
    verdict = {
        "G_Preference": bool(ci(S_QK)[0] > 0 and ci(S_A)[0] > 0 and ci(S_Y)[0] > 0),
        "C_Redirect": bool(R["C_Redirect"]["ci"][0] > 0),
        "C_Dissociate": dissociate,
    }
    verdict["overall"] = all(verdict.values())            # §4: G_Match is NOT part of overall
    verdict["G_Match_reported"] = bool(audit["PASS"])
    verdict["C_Address_reported"] = {"retained_L2H15": R["C_Address"]["retained_fraction_L2H15"],
                                     "p": addr_p}
    R["verdict"] = verdict
    R["REFLECTION"] = {"status": "PENDING -- fill after inspection (prereg §7)"}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))

    # status must not lie: only a passing preflight (and full power) is CONFIRMATORY; a --force
    # diagnostic run with a failed preflight is DIAGNOSTIC.
    status = "CONFIRMATORY" if R["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:acceptor_address", sys.argv, status=status,
        config={"split": args.split, "n_pairs": len(pairs), "n_genes": R["n_genes"],
                "tol": args.tol, "real_tol": args.real_tol, "null_heads": len(null_heads),
                "reader": list(READER), "addresser": list(ADDRESSER),
                "preflight_PASS": pf["PASS"], "meets_power_target": R["meets_power_target"]},
        seeds={"nulls": 0, "bootstrap": 0},
        notes=["Partner-identity preference (distance+phase-matched generic decoy, and a "
               "strength-controlled real-acceptor decoy) + key-only address<->content dissociation.",
               "The key+value transplant is R15-forced (recovers ~equally at decoy/non-site/a+); "
               "reported as the R15 guard, NOT gated -- no content-addressed pointer claim.",
               "Licensed claim stays at QK/attention + OV/mediation + Behaviour.",
               f"status={status} (preflight_PASS={pf['PASS']}, power_target={R['meets_power_target']})."])
    ckpt = Path(args.model) if Path(args.model).is_absolute() else REPO / args.model
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):                    # provenance: hash the data inputs
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    sr = R["preference"]["S_A_vs_real_acceptor"]
    print(f"preference  S_QK {R['preference']['S_QK']['mean']:+.3f}  "
          f"S_A {R['preference']['S_A']['mean']:+.3f}  S_Y {R['preference']['S_Y']['mean']:+.3f}"
          f"  | S_A vs real acceptor {sr.get('mean', float('nan')):+.3f} (n={sr['n']})")
    print(f"C-Redirect (key-only) S_A' {R['C_Redirect']['S_A_prime']:+.3f}  "
          f"| dissociate: kv {kv['recovery']:+.3f} > k {transplant['k@decoy']['recovery']:+.3f} "
          f"> v {transplant['v@decoy']['recovery']:+.3f}")
    print(f"C-R15 kv@[decoy,nonsite,a+] {[round(x,3) for x in kv_vals]}  "
          f"spread {R['C_R15_position_insensitivity']['spread']:.3f} (position-insensitive => "
          f"no pointer claim)")
    print(f"C-Address (reported): L2H15 retains "
          f"{100*R['C_Address']['retained_fraction_L2H15']:.0f}% of S_A ablated (p={addr_p:.3f})")
    print(f"verdict: {verdict}")
    print(f"saved -> {out}")


def mlp2_contribution(ad, task, pairs, off, sa_intact, *, batch=8) -> float:
    """S_A drop when MLP2's write is clamped to its L2H15-ablated value (its carriage of the
    address signal).  Captured per batch inside the ablation, clamped with L2H15 intact."""
    vals: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            ch = pairs[i:i + batch]
            exs = [p.example for p in ch]
            enc = task.encode([x.sequence for x in exs], [x.donor_pos for x in exs])
            with ablate(ad, [ADDRESSER])():
                w_abl = capture_writes(ad, MLP2, lambda: ad.model(**enc))
            dtok = torch.tensor([x.donor_pos + off for x in exs], device=ad.device)
            wplus = windows([p.a_plus for p in ch], off, ad.device)
            wminus = windows([p.a_minus for p in ch], off, ad.device)
            _, _, ap, am = preference(ad, task, enc, dtok, wplus, wminus,
                                      ctx=lambda: clamp_write(ad, MLP2, w_abl))
            vals += (ap - am).tolist()
    return sa_intact - float(np.mean(vals))


# --------------------------------------------------------------------------------------
# Preflight: positive/negative controls, sensitivity, resolution, invariants (prereg §6).
# --------------------------------------------------------------------------------------
def preflight(ad, task, pairs, off, args) -> dict:
    frozen = {
        "attn_row_sum_tol": 1e-3,
        "qk_recon_tol": 1e-3,               # softmax(q.k/sqrt32) must reproduce the model attn
        "injection_recovery_min": 0.30,     # L3H1-only key+value read is a partial mediator
        "value_only_recovery_max": 0.15,    # content at the decoy WITHOUT routing must not recover
        "positive_partner_coupling_min": 0.20,
        "l3h1_ablation_kills_frac_min": 0.50,
        "health_floor": 0.5,
    }
    checks: dict = {}
    sub = pairs[: min(len(pairs), 64)]                     # preflight on a bounded subset
    exs = [p.example for p in sub]
    dn = [x.donor_pos for x in exs]
    genes = [x.gene_id for x in exs]

    with torch.no_grad():
        # probability invariants (G-Attn): eager attention rows sum to 1, head not all-zero
        enc0 = task.encode([x.sequence for x in exs], dn)
        o = ad.model(**enc0, output_attentions=True)
        attn = o.attentions[L3][:, H3]
        row_sums = attn.sum(-1)
        checks["probability_invariants"] = {
            "PASS": bool(float((row_sums - 1).abs().max()) < frozen["attn_row_sum_tol"]
                         and float(attn.abs().max()) > 0),
            "max_row_sum_err": float((row_sums - 1).abs().max()),
            "attn_max": float(attn.abs().max()),
        }
        # G-QK: the analytic readout softmax(q.k/sqrt(d)) must reproduce the model's own L3H1
        # attention (guards the absolute-PE / post-LN assumptions behind S_QK).
        Dh = ad.hidden_size // ad.num_heads
        proj0 = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc0))
        q = proj0["query"][:, :, H3 * Dh:(H3 + 1) * Dh]
        k = proj0["key"][:, :, H3 * Dh:(H3 + 1) * Dh]
        add = (1.0 - enc0["attention_mask"][:, None, :].float()) * -1e9
        recon = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (Dh ** 0.5) + add, dim=-1)
        qk_err = float((recon - attn).abs().max())
        checks["qk_reconstruction"] = {"PASS": bool(qk_err < frozen["qk_recon_tol"]),
                                       "max_abs_err": qk_err}

        # positive control: destroying the real partner collapses the donor margin
        clean = task.encode([x.sequence for x in exs], dn)
        corr = corrupt_at(task, exs, [p.a_plus for p in sub])
        m_cl = task.margin(ad.model(**clean).logits.float(), dn).cpu().numpy()
        m_co = task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
        coupling = float(np.mean(m_cl - m_co))
        checks["positive_control_partner_coupling"] = {
            "PASS": bool(coupling >= frozen["positive_partner_coupling_min"]),
            "coupling": coupling,
        }

        # positive control: L3H1 mean-ablation kills the coupling
        with ablate(ad, [READER])():
            a_cl = task.margin(ad.model(**clean).logits.float(), dn).cpu().numpy()
            a_co = task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
        coupling_abl = float(np.mean(a_cl - a_co))
        killed = 1.0 - (coupling_abl / coupling if coupling else 1.0)
        checks["positive_control_L3H1_ablation"] = {
            "PASS": bool(killed >= frozen["l3h1_ablation_kills_frac_min"]),
            "coupling_L3H1_ablated": coupling_abl, "killed_fraction": killed,
        }

        # transplant sensitivity + specificity on the subset
        dtok = torch.tensor([d + off for d in dn], device=ad.device)
        wplus = windows([p.a_plus for p in sub], off, ad.device)
        wminus = windows([p.a_minus for p in sub], off, ad.device)
        W = build_windows(exs, off, ad.device)
        src = capture_proj(ad, L3, ["key", "value"], lambda: ad.model(**clean))
        dnm = (m_cl - m_co)

        def reloc_recovery(whichs, dst):
            st = ExitStack()
            for w in whichs:
                st.enter_context(relocate_proj(ad, L3, H3, w, src[w], wplus, dst))
            with st:
                m = task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
            return ratio_of_means(m - m_co, dnm)

        inj = reloc_recovery(["key", "value"], wplus)      # put partner (k,v) back at a+ (ceiling)
        vonly = reloc_recovery(["value"], wminus)          # content at decoy WITHOUT routing
        nonsite = reloc_recovery(["key", "value"], W.control)   # reported (R15 position test)
        checks["sensitivity_injection"] = {
            "PASS": bool(inj >= frozen["injection_recovery_min"]), "recovery": inj}
        checks["negative_control_value_only"] = {
            "PASS": bool(vonly <= frozen["value_only_recovery_max"]), "recovery": vonly}
        # kv@non-site is REPORTED, not gated: relocating a high-affinity key is position-
        # forced to recover anywhere (R15), so this cannot be a specificity pass/fail.
        checks["diagnostic_kv_nonsite_recovery"] = {"PASS": True, "recovery": nonsite}

        # swap-is-identity tripwire (R7): a symmetric key+value swap on the CLEAN run must
        # reproduce the un-swapped clean margin to numerical precision.
        swap_diff = swap_identity_margin(ad, task, exs, dn, wplus, wminus)
        checks["swap_is_identity"] = {
            "PASS": bool(np.max(np.abs(swap_diff)) < 1e-4),
            "max_abs_diff": float(np.max(np.abs(swap_diff))),
        }

        # C-Health (R8/R13): the destructive ablations must leave general MLM intact.  The
        # transplant is a 4-token localized copy whose aliveness is shown by the injection
        # sensitivity above (it recovers the true partner signal), not by a health probe.
        health_l3 = task.mlm_health(exs, ctx=ablate(ad, [READER]))
        health_l2 = task.mlm_health(exs, ctx=ablate(ad, [ADDRESSER]))
        checks["health_under_L3H1_ablation"] = {
            "PASS": bool(health_l3 >= frozen["health_floor"]), "health": health_l3}
        checks["health_under_L2H15_ablation"] = {
            "PASS": bool(health_l2 >= frozen["health_floor"]), "health": health_l2}

    # resolution: the pair match must hold
    checks["resolution_pair_match"] = {"PASS": bool(match_audit(pairs)["PASS"]),
                                       **match_audit(pairs)}
    # both strands present
    checks["both_strands_present"] = {"PASS": len({p.strand for p in pairs}) == 2,
                                      "strands": sorted({p.strand for p in pairs})}
    # checkpoint hash recorded (mutation guard baseline)
    ckpt = Path(args.model)
    ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True,
                                     "sha256": sha256_file(ckpt / "pytorch_model.bin")}

    PASS = all(c["PASS"] for c in checks.values())
    return {"frozen": frozen, "commit": git_commit(REPO), "n_pairs": len(pairs),
            "checks": checks, "PASS": bool(PASS)}


def swap_identity_margin(ad, task, exs, dn, wplus, wminus) -> np.ndarray:
    """Per-example (swapped - baseline) margin for a symmetric a+<->a- key+value swap on the
    CLEAN run.  Must be ~0 (R7): swapping complete (k,v) pairs relabels the summation index
    and leaves the head output identical, so this array bounds the arithmetic, not a mechanism.
    """
    clean = task.encode([x.sequence for x in exs], dn)
    src = capture_proj(ad, L3, ["key", "value"], lambda: ad.model(**clean))
    base = task.margin(ad.model(**clean).logits.float(), dn).cpu().numpy()
    st = ExitStack()
    for w in ("key", "value"):
        st.enter_context(relocate_proj(ad, L3, H3, w, src[w], wplus, wminus))
        st.enter_context(relocate_proj(ad, L3, H3, w, src[w], wminus, wplus))
    with st:
        swapped = task.margin(ad.model(**clean).logits.float(), dn).cpu().numpy()
    return swapped - base


if __name__ == "__main__":
    main()
