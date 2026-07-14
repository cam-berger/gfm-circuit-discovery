#!/usr/bin/env python
"""EXPLORATORY -- what is L1H12?

The exhaustive both-strand ablation scan keeps putting it third:

    L3H1   0.830 [0.654, 0.994]     the reader          (in C)
    L2H15  0.515 [0.396, 0.625]     the addresser       (in C)
    L1H12  0.266 [0.178, 0.367]     ???                 (NOT in C)
    L2H4   0.105

0.266 is not noise and it is not small -- it is a third of L3H1.  We froze C = {L3H1, L2H15}
before it, and the honest statement has been "C is the dominant sub-circuit, not the circuit."
This script asks what the third head is doing.

Nothing here is confirmatory.  It is a probe, and it is designed so that "L1H12 is boring"
is a reachable answer.

STAGE 1 -- PROFILE.  Is it necessary, is it specific, does it break the model, and does it
  DENOISE?  L2H15 has high necessity and ~zero denoise recovery -- the signature of machinery
  the reader needs rather than a carrier of the partner signal.  If L1H12 has the same
  signature it is upstream infrastructure; if it denoises, it carries partner information
  independently and C is missing a limb.

STAGE 2 -- ATTENTION.  Where does it look, from the donor and from the acceptor?  Layer-1
  heads are often local/positional.  If its attention is a fixed local window that does not
  move when the partner is destroyed, it is a feature-builder, not a partner-reader.

STAGE 3 -- NESTING.  2x2 against L3H1 and against L2H15.  Is its effect already inside the
  circuit's, or does it add?
      drop(A) + drop(B) ~ drop(A,B)   -> independent, additive: C IS MISSING A COMPONENT
      drop(A,B) ~ max(drop A, drop B) -> nested: it feeds the head we already have
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, CIRCUIT, CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER,
    CouplingTask, ablate, capture_head_outputs, gboot, gboot_ratio, load_adapter,
    load_examples, patch_heads, pick_device, ratio_of_means,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402

L3, H3 = READER
L2, H2 = ADDRESSER
CAND = (1, 12)          # the head under the microscope
L1, H1 = CAND


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/l1h12_probe.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)     # attention is read below
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"l1h12-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} {args.split} exons   device={dev}   probing L{L1}H{H1}\n")
    R: dict = {"split": args.split, "n": len(X), "head": list(CAND)}

    # ================= STAGE 1 -- PROFILE ==========================================
    base = task.coupling(X)
    h_base = task.mlm_health(X[:80])
    print("=" * 78)
    print(f"STAGE 1  PROFILE      baseline coupling {base.mean():+.3f}   "
          f"MLM health {h_base:.4f}")
    prof = {}
    for name, hd in [(f"L{L1}H{H1}", CAND), ("L3H1", READER), ("L2H15", ADDRESSER)]:
        c = task.coupling(X, ablate(ad, [hd]))
        h = task.mlm_health(X[:80], ablate(ad, [hd]))
        lo, hi = gboot(base - c, genes)
        prof[name] = dict(coupling=float(c.mean()), lost=float((base - c).mean()),
                          lost_ci=[lo, hi],
                          frac_lost=float((base - c).mean() / base.mean()),
                          mlm_health=h, health_retained=float(h / h_base))
        print(f"  ablate {name:6s}  coupling {c.mean():+.3f}   lost "
              f"{100*(base-c).mean()/base.mean():5.1f}%  [{lo:+.3f},{hi:+.3f}]   "
              f"health {h:.4f} ({100*h/h_base:.1f}%)")

    # DENOISE: does it CARRY the partner signal, or is it machinery?
    print("\n  DENOISE (patch the clean head output into the corrupt run):")
    den = {}
    for name, hd in [(f"L{L1}H{H1}", CAND), ("L3H1", READER), ("L2H15", ADDRESSER)]:
        num, dnm = [], []
        with torch.no_grad():
            for i in range(0, len(X), task.batch):
                cl, co, dn = task.pair(X[i:i + task.batch])
                cc = capture_head_outputs(ad, lambda: ad.model(**cl))
                mc = task.margin(ad.model(**cl).logits.float(), dn)
                mk = task.margin(ad.model(**co).logits.float(), dn)
                with patch_heads(ad, cc, [hd]):
                    mp = task.margin(ad.model(**co).logits.float(), dn)
                num += (mp - mk).tolist()
                dnm += (mc - mk).tolist()
        r = ratio_of_means(num, dnm)
        lo, hi = gboot_ratio(num, dnm, genes)
        den[name] = dict(recovery=r, lo=lo, hi=hi)
        print(f"    {name:6s} recovery {r:6.3f}  [{lo:+.3f},{hi:+.3f}]")
    print("  A head with HIGH necessity and ~ZERO denoise recovery is MACHINERY the reader")
    print("  needs, not a carrier of the partner signal (that is L2H15's signature).")
    R["profile"] = prof
    R["denoise"] = den

    # ================= STAGE 2 -- ATTENTION =======================================
    print("\n" + "=" * 78)
    print(f"STAGE 2  WHERE DOES L{L1}H{H1} LOOK?")
    # attention from the donor and from the partner acceptor, clean vs corrupted
    loc_d, loc_a, part, self_d = [], [], [], []
    loc_d_c, part_c = [], []
    offs = np.zeros(41)     # mean attention by relative offset, from the donor
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            cl, co, dn = task.pair(ch)
            A0 = ad.model(**cl, output_attentions=True).attentions[L1][:, H1]
            AC = ad.model(**co, output_attentions=True).attentions[L1][:, H1]
            if float(A0.abs().sum()) == 0.0:
                raise RuntimeError("all-zero attention -- adapter must be eager")
            for j, x in enumerate(ch):
                d, p = x.donor_pos + OFF, x.next_acceptor_pos + OFF
                row, rowc = A0[j, d], AC[j, d]
                loc_d.append(row[d - 5:d + 6].sum().item())          # local window at the donor
                loc_d_c.append(rowc[d - 5:d + 6].sum().item())
                self_d.append(row[d].item())
                part.append(row[p - 2:p + 1].sum().item())           # -> the partner AG
                part_c.append(rowc[p - 2:p + 1].sum().item())
                loc_a.append(A0[j, p, p - 5:p + 6].sum().item())     # local window at acceptor
                for k in range(-20, 21):
                    if 0 <= d + k < row.shape[0]:
                        offs[k + 20] += row[d + k].item()
    offs /= len(X)
    print(f"  from the DONOR:")
    print(f"    attention to its own local window [d-5,d+5]   {np.mean(loc_d):.4f}"
          f"   (self alone {np.mean(self_d):.4f})")
    print(f"    attention to the PARTNER AG                   {np.mean(part):.4f}"
          f"   (L3H1 for comparison: ~0.62)")
    print(f"    ... with the partner DESTROYED                {np.mean(part_c):.4f}")
    print(f"    local window with the partner destroyed       {np.mean(loc_d_c):.4f}")
    print(f"  from the ACCEPTOR: attention to its own local window  {np.mean(loc_a):.4f}")
    top = np.argsort(-offs)[:6]
    print("  top relative offsets from the donor (offset: mean attention):")
    print("    " + "  ".join(f"{t-20:+d}:{offs[t]:.3f}" for t in sorted(top, key=lambda t: -offs[t])))
    R["attention"] = dict(
        donor_local=float(np.mean(loc_d)), donor_self=float(np.mean(self_d)),
        donor_to_partner=float(np.mean(part)),
        donor_to_partner_corrupted=float(np.mean(part_c)),
        donor_local_corrupted=float(np.mean(loc_d_c)),
        acceptor_local=float(np.mean(loc_a)),
        offset_profile={str(k - 20): float(offs[k]) for k in range(41)})
    if np.mean(part) < 0.05:
        print("  => it does NOT read the partner.  It is not a second reader.")
    if abs(np.mean(loc_d) - np.mean(loc_d_c)) < 0.02:
        print("  => its attention does not move when the partner is destroyed: a")
        print("     FEATURE-BUILDER, not a partner-dependent router.")

    # ================= STAGE 3 -- NESTING =========================================
    print("\n" + "=" * 78)
    print("STAGE 3  IS ITS EFFECT ALREADY INSIDE THE CIRCUIT'S?")
    d_1 = base - task.coupling(X, ablate(ad, [CAND]))
    nest = {}
    for name, other in [("L3H1", READER), ("L2H15", ADDRESSER)]:
        d_o = base - task.coupling(X, ablate(ad, [other]))
        d_b = base - task.coupling(X, ablate(ad, [CAND, other]))
        extra = d_b - d_o                       # what L1H12 adds ON TOP of `other`
        lo, hi = gboot(extra, genes)
        nest[name] = dict(drop_l1h12=float(d_1.mean()), drop_other=float(d_o.mean()),
                          drop_both=float(d_b.mean()), extra_over_other=float(extra.mean()),
                          extra_ci=[lo, hi],
                          frac_nested=float(1 - extra.mean() / max(d_1.mean(), 1e-9)))
        print(f"  vs {name}:  drop(L1H12) {d_1.mean():+.3f}   drop({name}) {d_o.mean():+.3f}"
              f"   drop(both) {d_b.mean():+.3f}")
        print(f"      L1H12 adds {extra.mean():+.3f} [{lo:+.3f},{hi:+.3f}] on top of {name}"
              f"  => {100*(1-extra.mean()/max(d_1.mean(),1e-9)):.0f}% NESTED")
    dC = base - task.coupling(X, ablate(ad, list(CIRCUIT)))
    dCA = base - task.coupling(X, ablate(ad, list(CIRCUIT) + [CAND]))
    ex = dCA - dC
    lo, hi = gboot(ex, genes)
    nest["circuit C"] = dict(drop_C=float(dC.mean()), drop_C_plus=float(dCA.mean()),
                             extra=float(ex.mean()), extra_ci=[lo, hi])
    print(f"\n  vs the WHOLE circuit C: drop(C) {dC.mean():+.3f}  ->  drop(C + L1H12) "
          f"{dCA.mean():+.3f}")
    print(f"      L1H12 adds {ex.mean():+.3f} [{lo:+.3f},{hi:+.3f}] on top of C")
    if hi < 0.05:
        print("      => FULLY NESTED.  L1H12 is upstream machinery for C, not a missing limb.")
        print("         C stays as it is; L1H12 belongs in the story, not in the head set.")
    else:
        print("      => it ADDS to C.  C is genuinely incomplete and L1H12 is a candidate")
        print("         fourth component.  This would need a re-freeze, not a footnote.")
    R["nesting"] = nest

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:l1h12_probe", sys.argv,
        config={"split": args.split, "n": len(X), "candidate": list(CAND),
                "circuit": [list(h) for h in CIRCUIT]},
        notes=["EXPLORATORY, not confirmatory.  No threshold, no gate.",
               "'L1H12 is boring' is a reachable answer and must stay reachable."],
    ).write(out.parent / "l1h12_probe.manifest.json")
    print("=" * 78)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
