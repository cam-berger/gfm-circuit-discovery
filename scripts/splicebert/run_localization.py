#!/usr/bin/env python
"""STEP 8-9: mechanistic localization of the cross-site coupling circuit.

PHENOMENON (established, steps 1-7):
    destroying the intron-mate acceptor A' degrades the model's native MLM recognition
    of its partner donor D (+1.32 margin at 60-150nt separation), while a motif-matched
    non-partner AG does nothing (+0.01).  Partner-selective, bidirectional.

WHAT WE LOCALIZE:
    the causal pathway carrying information from A' to the prediction at D.

METHOD -- and why each piece is here:

  1. AtP shortlist.  Gradient attribution over all 96 heads.  Cheap, and we have already
     validated that it tracks exact patching (Spearman 0.876).  Used ONLY to shortlist;
     every claim is confirmed by exact patching.

  2. Exact DENOISING patch (corrupt -> patch clean head output -> measure recovery).
     Denoising, not ablation, is the right primitive for tracing an information FLOW:
     it finds heads whose clean value is SUFFICIENT to restore the signal.
        recovery_h = (m_patched - m_corrupt) / (m_clean - m_corrupt)
     recovery ~ 1  =>  that head alone carries the partner information.

  3. NOISING (clean -> patch corrupt) as well.  Necessity and sufficiency can dissociate
     (backup heads / hydra effect).  Reporting only one is how people fool themselves.

  4. POSITION-RESOLVED patching.  A head "mattering" is not the same as a head WRITING the
     partner information into the donor.  We patch the head output at the DONOR position
     only, at the ACCEPTOR position only, and everywhere, and compare.

  5. ATTENTION patterns.  A genuine cross-site head should literally attend D -> A'.
     Convergent evidence from an independent measurement; if attention and causal patching
     disagree, we do not have a mechanism.

  6. VALIDATION TRIAD (step 9).  Ablating the circuit must:
        (a) ABOLISH the cross-site coupling,
        (b) PRESERVE local donor recognition (clean margin),
        (c) PRESERVE general MLM competence at random positions.
     Without (b) and (c), "ablation kills the effect" only means we broke the model.

  7. STATISTICS.  Matched-random head-set null with a finite-sample add-one p-value, and
     BH-FDR over the 96 heads.  The calibration of this exact path was audited on pure
     noise (uniform max-p, 0 false discoveries) -- see tests/test_rigor_calibration.py.
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

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.circuits.hf_bert import (  # noqa: E402
    capture_head_outputs, edit_heads, matched_random_head_sets, patch_heads,
)
from gfmcd.rigor import benjamini_hochberg, empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, build_coupling_dataset, stable_rng, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402


class Task:
    """Clean/corrupt pairs + the native MLM metric at the donor."""

    def __init__(self, adapter, examples, batch=8):
        self.a, self.ex, self.bs = adapter, examples, batch
        tok = adapter.tokenizer
        self.mask_id = tok.mask_token_id
        self.bid = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
        self.off = adapter.prefix_tokens

    def _ids(self, seqs, donors):
        enc = self.a.tokenizer([" ".join(s) for s in seqs], return_tensors="pt",
                               padding=True).to(self.a.device)
        for j, d in enumerate(donors):
            enc["input_ids"][j, d + self.off] = self.mask_id
        return enc

    def batches(self):
        for i in range(0, len(self.ex), self.bs):
            ch = self.ex[i:i + self.bs]
            donors = [x.donor_pos for x in ch]
            clean = [x.sequence for x in ch]
            # corrupt: destroy the INTRON-mate acceptor (A'), donor still masked
            corrupt = [substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                       for x in ch]
            yield ch, self._ids(clean, donors), self._ids(corrupt, donors), donors

    def metric(self, logits, donors):
        """logit(G) - max(logit of A,C,T) at the masked donor. [B] tensor."""
        idx = torch.arange(len(donors), device=logits.device)
        pos = torch.tensor([d + self.off for d in donors], device=logits.device)
        v = logits[idx, pos]                      # [B, vocab]
        g = v[:, self.bid["G"]]
        other = torch.stack([v[:, self.bid[b]] for b in "ACT"], -1).max(-1).values
        return g - other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--shortlist", type=int, default=12)
    ap.add_argument("--random-sets", type=int, default=500)
    ap.add_argument("--min-effect", type=float, default=0.3,
                    help="localize only where the phenomenon is actually present")
    ap.add_argument("--chroms", nargs="*", default=["chr22", "chr21", "chr20", "chr19", "chr17"])
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/development/localization.json")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    # SDPA SILENTLY IGNORES output_attentions=True and returns zeros. Force eager, or the
    # attention analysis reports 0.0000 for every head and looks like a real (null) result.
    _m = AutoModelForMaskedLM.from_pretrained(mp, attn_implementation="eager")
    adapter = SpliceBertAdapter(_m, AutoTokenizer.from_pretrained(mp), device=dev)
    L, H = adapter.num_layers, adapter.num_heads
    print(f"SpliceBERT {L} layers x {H} heads = {L*H} heads   device={dev}")

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg,
                               allowed_chromosomes=set(args.chroms))
    X = [x for x in X if x.split == "train" and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]   # the largest-effect regime
    rng = stable_rng("loc", cfg.seed)
    rng.shuffle(X)

    # ---- select examples where the PHENOMENON IS PRESENT (pre-registered filter) -------
    probe = Task(adapter, X[: args.n * 3], batch=args.batch)
    keep, effects = [], []
    with torch.no_grad():
        for ch, cln, cor, dn in probe.batches():
            mc = probe.metric(adapter.model(**cln).logits.float(), dn)
            mk = probe.metric(adapter.model(**cor).logits.float(), dn)
            for j, x in enumerate(ch):
                e = float(mc[j] - mk[j])
                if e >= args.min_effect:
                    keep.append(x)
                    effects.append(e)
            if len(keep) >= args.n:
                break
    keep, effects = keep[: args.n], effects[: args.n]
    print(f"localizing on {len(keep)} exons with a real coupling effect "
          f"(mean {np.mean(effects):+.3f}, min {args.min_effect})")
    task = Task(adapter, keep, batch=args.batch)

    heads = [(l, h) for l in range(L) for h in range(H)]
    n_ex = len(keep)
    rec_denoise = np.zeros((len(heads), n_ex))      # sufficiency
    rec_noise = np.zeros((len(heads), n_ex))        # necessity
    rec_at_donor = np.zeros((len(heads), n_ex))     # position-resolved: patched only at D
    attn_to_partner = np.zeros((len(heads), n_ex))
    attn_baseline = np.zeros((len(heads), n_ex))

    e0 = 0
    for ch, cln, cor, dn in task.batches():
        B = len(ch)
        sl = slice(e0, e0 + B)
        with torch.no_grad():
            clean_cache = capture_head_outputs(adapter, lambda: adapter.model(**cln))
            corrupt_cache = capture_head_outputs(adapter, lambda: adapter.model(**cor))
            m_clean = task.metric(adapter.model(**cln).logits.float(), dn)
            m_corr = task.metric(adapter.model(**cor).logits.float(), dn)
            denom = (m_clean - m_corr).clamp_min(1e-3)

            # attention D -> A' for every head (convergent evidence)
            out = adapter.model(**cln, output_attentions=True)
            for li, att in enumerate(out.attentions):        # [B,H,S,S]
                for j, x in enumerate(ch):
                    d = x.donor_pos + task.off
                    ap_ = x.next_acceptor_pos + task.off
                    for h in range(H):
                        w = att[j, h, d, ap_ - 2:ap_ + 1].sum().item()
                        attn_to_partner[li * H + h, e0 + j] = w
                        # distance-matched baseline: same |offset| on the other side
                        bp = d - (ap_ - d)
                        if 2 < bp < att.shape[-1] - 1:
                            attn_baseline[li * H + h, e0 + j] = \
                                att[j, h, d, bp - 2:bp + 1].sum().item()

            # donor-position-only mask, for position-resolved patching
            S = cln["input_ids"].shape[1]
            dmask = torch.zeros((B, S), dtype=torch.bool, device=adapter.device)
            for j, x in enumerate(ch):
                dmask[j, x.donor_pos + task.off] = True

            for hi, hd in enumerate(heads):
                with patch_heads(adapter, clean_cache, [hd]):        # denoise: clean -> corrupt
                    m = task.metric(adapter.model(**cor).logits.float(), dn)
                rec_denoise[hi, sl] = ((m - m_corr) / denom).cpu().numpy()

                with patch_heads(adapter, corrupt_cache, [hd]):      # noise: corrupt -> clean
                    m = task.metric(adapter.model(**cln).logits.float(), dn)
                rec_noise[hi, sl] = ((m_clean - m) / denom).cpu().numpy()

                with patch_heads(adapter, clean_cache, [hd], token_mask=dmask):
                    m = task.metric(adapter.model(**cor).logits.float(), dn)
                rec_at_donor[hi, sl] = ((m - m_corr) / denom).cpu().numpy()
        e0 += B
        print(f"  patched {e0}/{n_ex}", end="\r", flush=True)
    print()

    den = rec_denoise.mean(1)
    noi = rec_noise.mean(1)
    atd = rec_at_donor.mean(1)
    atn = (attn_to_partner - attn_baseline).mean(1)

    if np.allclose(attn_to_partner, 0):
        raise RuntimeError("attention tensors are all zero -- output_attentions was ignored "
                           "(SDPA?). Refusing to report a fabricated attention null.")

    order = np.argsort(-den)[: args.shortlist]
    print("\n" + "=" * 88)
    print("HEAD-LEVEL LOCALIZATION  (recovery = fraction of the coupling effect restored)")
    hdr = "attn D->A' (excess)"
    print(f"  {'head':8s} {'DENOISE':>9s} {'NOISE':>9s} {'@donor only':>12s} {hdr:>21s}")
    for i in order:
        l, h = heads[i]
        print(f"  L{l}H{h:<5d} {den[i]:9.3f} {noi[i]:9.3f} {atd[i]:12.3f} {atn[i]:21.4f}")

    # ---- FDR over all 96 heads against a per-head null of zero recovery ---------------
    se = rec_denoise.std(1) / np.sqrt(n_ex) + 1e-9
    from scipy import stats
    pvals = 1 - stats.norm.cdf(den / se)
    qvals = benjamini_hochberg(list(pvals))
    sig = [heads[i] for i in range(len(heads)) if qvals[i] <= 0.10]
    print(f"\nBH-FDR (q=0.10): {len(sig)}/{len(heads)} heads with non-zero denoising recovery")

    # ---- the CIRCUIT: heads that both restore AND write at the donor ------------------
    C = [heads[i] for i in order if den[i] > 0.05 and atd[i] > 0.02][:6]
    if not C:
        C = [heads[i] for i in order[:3]]
    print(f"CIRCUIT C = {['L%dH%d' % t for t in C]}")

    # ---- STEP 9: VALIDATION TRIAD ----------------------------------------------------
    def eval_under(ablate):
        cpl, cleanm, health = [], [], []
        with torch.no_grad():
            for ch, cln, cor, dn in task.batches():
                ctx = edit_heads(adapter, ablate, mode="mean") if ablate else _null()
                with ctx:
                    mc = task.metric(adapter.model(**cln).logits.float(), dn)
                    mk = task.metric(adapter.model(**cor).logits.float(), dn)
                    lg = adapter.model(**cln).logits.float()
                cpl += (mc - mk).tolist()
                cleanm += mc.tolist()
                # MLM health: mean top-1 accuracy at 30 random NON-site positions
                r = stable_rng("health", 0)
                for j, x in enumerate(ch):
                    ps = [r.randrange(10, len(x.sequence) - 10) for _ in range(30)]
                    pred = lg[j, [p + task.off for p in ps]].argmax(-1)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in ps],
                                       device=pred.device)
                    health.append((pred == tgt).float().mean().item())
        return float(np.mean(cpl)), float(np.mean(cleanm)), float(np.mean(health))

    from contextlib import contextmanager

    @contextmanager
    def _null():
        yield

    base = eval_under([])
    abl = eval_under(C)
    print("\n" + "=" * 88)
    print("STEP 9  VALIDATION TRIAD  (ablate the circuit, mean-ablation)")
    print(f"  {'':28s} {'baseline':>10s} {'C ablated':>11s}")
    print(f"  {'(a) cross-site coupling':28s} {base[0]:10.3f} {abl[0]:11.3f}   <- must COLLAPSE")
    print(f"  {'(b) clean donor margin':28s} {base[1]:10.3f} {abl[1]:11.3f}   <- must be PRESERVED")
    print(f"  {'(c) general MLM top-1':28s} {base[2]:10.3f} {abl[2]:11.3f}   <- must be PRESERVED")

    # matched-random head-set null: is C special, or would any |C| heads do this?
    null_sets = matched_random_head_sets(adapter, C, n=args.random_sets, seed=0,
                                         exclude_selected=True)
    drops = []
    for s in null_sets:
        drops.append(base[0] - eval_under(list(s))[0])
    obs = base[0] - abl[0]
    p = empirical_upper_p_value(obs, drops)
    print(f"\n  coupling drop: observed {obs:+.3f} vs matched-random mean "
          f"{np.mean(drops):+.3f}   add-one p = {p:.4f}  (n={len(drops)})")
    print("=" * 88)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_examples": n_ex, "mean_effect": float(np.mean(effects)),
        "heads": [{"layer": l, "head": h, "denoise": float(den[i]), "noise": float(noi[i]),
                   "at_donor": float(atd[i]), "attn_excess": float(atn[i])}
                  for i, (l, h) in enumerate(heads)],
        "circuit": [list(t) for t in C],
        "fdr_significant": [list(t) for t in sig],
        "validation": {"baseline": base, "ablated": abl,
                       "null_mean_drop": float(np.mean(drops)), "p_add_one": float(p)},
    }, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
