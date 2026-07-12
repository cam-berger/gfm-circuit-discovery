#!/usr/bin/env python
"""2x2 FACTORIAL + SUBSPACE DISSECTION of L3H1.

PART 1 -- the 2x2 factorial (measured, not inferred)

    partner   L3H1      donor margin
    intact    intact    M_CI
    corrupt   intact    M_XI
    intact    ablated   M_CA
    corrupt   ablated   M_XA

  partner-dependent role of L3H1 = (M_CI - M_XI) - (M_CA - M_XA)   [difference-in-differences]
  local (partner-independent) role = M_XI - M_XA
      -- the partner is already destroyed in BOTH terms, so this isolates the head's
         partner-independent donor-recognition contribution.

  This replaces the INFERRED 1.31 / 1.01 split with a measured one.

PART 2 -- is there a partner subspace multiplexed inside the head?

  L3H1 is not a dedicated cross-site module (ablating it also costs local donor
  recognition).  But a SUBSPACE of its 32-dim output might be.

    1. delta_i = z_clean[i, donor] - z_corrupt[i, donor]   in L3H1's output space (R^32)
    2. SVD over the FIT split -> top-k partner-sensitive directions U_k
    3. on the EVAL split, project U_k OUT of L3H1's output during the forward pass
    4. does the coupling vanish while the clean donor margin survives?
    5. compare against RANDOM RANK-MATCHED subspaces (the essential control: any rank-k
       projection removes *some* variance, so a random baseline is mandatory)

  U IS FIT ON A DISJOINT SPLIT FROM THE ONE IT IS SCORED ON.  Fitting and evaluating on
  the same exons would make the subspace trivially "work" -- it would have been fit to
  those very deltas.

  Outcomes:
    coupling -> 0 AND margin preserved  => a dedicated partner feature, multiplexed inside
                                           a multifunctional head  (the strong result)
    coupling -> 0 but margin collapses  => local and partner evidence are entangled
    needs many directions               => distributed partner representation
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.circuits.hf_bert import (  # noqa: E402
    _context_from_output, _merge_context, _replace_output, _split_context,
    capture_head_outputs, edit_heads,
)
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, build_coupling_dataset, stable_rng, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

LAYER, HEAD = 3, 1


@contextmanager
def project_out(adapter, layer: int, head: int, U: torch.Tensor):
    """Remove span(U) from head (layer, head)'s output.  U: [Dh, k], orthonormal columns."""
    def hook(_m, _i, output):
        ctx = _context_from_output(output)
        split = _split_context(ctx, adapter.num_heads).clone()   # [B,H,S,Dh]
        z = split[:, head]                                       # [B,S,Dh]
        split[:, head] = z - (z @ U) @ U.transpose(0, 1)
        return _replace_output(output, _merge_context(split))

    h = adapter.attention_self_modules[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def _noop():
    yield


class Task:
    def __init__(self, adapter, batch=8):
        self.a, self.bs = adapter, batch
        self.mask_id = adapter.tokenizer.mask_token_id
        self.bid = {b: adapter.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        self.off = adapter.prefix_tokens

    def ids(self, seqs, donors):
        enc = self.a.tokenizer([" ".join(s) for s in seqs], return_tensors="pt",
                               padding=True).to(self.a.device)
        for j, d in enumerate(donors):
            enc["input_ids"][j, d + self.off] = self.mask_id
        return enc

    def pair(self, ch):
        dn = [x.donor_pos for x in ch]
        cl = [x.sequence for x in ch]
        co = [substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0]) for x in ch]
        return self.ids(cl, dn), self.ids(co, dn), dn

    def metric(self, logits, dn):
        i = torch.arange(len(dn), device=logits.device)
        p = torch.tensor([d + self.off for d in dn], device=logits.device)
        v = logits[i, p]
        return v[:, self.bid["G"]] - torch.stack(
            [v[:, self.bid[b]] for b in "ACT"], -1).max(-1).values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chroms", nargs="*", default=["chr22", "chr21", "chr20", "chr19", "chr17"])
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/subspace.json")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    adapter = SpliceBertAdapter(AutoModelForMaskedLM.from_pretrained(mp),
                                AutoTokenizer.from_pretrained(mp), device=dev)
    task = Task(adapter, batch=args.batch)
    DH = adapter.model.config.hidden_size // adapter.num_heads
    print(f"L{LAYER}H{HEAD}   head_dim={DH}   device={dev}")

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg,
                               allowed_chromosomes=set(args.chroms))
    X = [x for x in X if x.split == "train" and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    rng = stable_rng("sub", cfg.seed)
    rng.shuffle(X)

    # keep only exons where the phenomenon is present
    keep = []
    with torch.no_grad():
        for i in range(0, min(len(X), args.n * 4), args.batch):
            ch = X[i:i + args.batch]
            cl, co, dn = task.pair(ch)
            e = (task.metric(adapter.model(**cl).logits.float(), dn)
                 - task.metric(adapter.model(**co).logits.float(), dn))
            keep += [x for j, x in enumerate(ch) if float(e[j]) >= 0.3]
            if len(keep) >= args.n:
                break
    keep = keep[: args.n]
    half = len(keep) // 2
    FIT, EVAL = keep[:half], keep[half:]          # DISJOINT: fit the subspace, then score it
    print(f"{len(keep)} exons with a real effect -> fit on {len(FIT)}, evaluate on {len(EVAL)}")

    def run(exs, ctx_fn):
        """Return (mean clean margin, mean corrupt margin) under an intervention."""
        C, K = [], []
        with torch.no_grad():
            for i in range(0, len(exs), args.batch):
                cl, co, dn = task.pair(exs[i:i + args.batch])
                with ctx_fn():
                    C += task.metric(adapter.model(**cl).logits.float(), dn).tolist()
                    K += task.metric(adapter.model(**co).logits.float(), dn).tolist()
        return float(np.mean(C)), float(np.mean(K))

    # ---------------- PART 1: the 2x2 factorial ------------------------------------
    M_CI, M_XI = run(EVAL, _noop)
    M_CA, M_XA = run(EVAL, lambda: edit_heads(adapter, [(LAYER, HEAD)], mode="mean"))

    did = (M_CI - M_XI) - (M_CA - M_XA)
    local = M_XI - M_XA
    print("\n" + "=" * 74)
    print("PART 1  2x2 FACTORIAL  (donor margin)")
    print(f"  {'':22s} {'L3H1 intact':>13s} {'L3H1 ablated':>14s}")
    print(f"  {'partner intact':22s} {M_CI:13.3f} {M_CA:14.3f}")
    print(f"  {'partner corrupted':22s} {M_XI:13.3f} {M_XA:14.3f}")
    print(f"\n  partner-dependent role of L3H1 (DiD) = ({M_CI:.3f}-{M_XI:.3f}) - "
          f"({M_CA:.3f}-{M_XA:.3f}) = {did:+.3f}")
    print(f"  local  partner-INdependent role        = {M_XI:.3f} - {M_XA:.3f} = {local:+.3f}")
    tot = did + local
    if abs(tot) > 1e-6:
        print(f"  => L3H1's contribution is {100*did/tot:.0f}% partner-dependent, "
              f"{100*local/tot:.0f}% local")

    # ---------------- PART 2: the partner subspace ---------------------------------
    # delta = clean - corrupt head output at the DONOR, in L3H1's 32-dim output space
    deltas = []
    with torch.no_grad():
        for i in range(0, len(FIT), args.batch):
            ch = FIT[i:i + args.batch]
            cl, co, dn = task.pair(ch)
            cc = capture_head_outputs(adapter, lambda: adapter.model(**cl))
            kk = capture_head_outputs(adapter, lambda: adapter.model(**co))
            for j, x in enumerate(ch):
                p = x.donor_pos + task.off
                deltas.append((cc[LAYER][j, HEAD, p] - kk[LAYER][j, HEAD, p]).cpu().numpy())
    D = np.stack(deltas)                                    # [n_fit, 32]
    Uf, S, _ = np.linalg.svd(D - D.mean(0, keepdims=True), full_matrices=False)
    V = np.linalg.svd(D - D.mean(0, keepdims=True), full_matrices=False)[2]   # [32,32]
    ev = (S ** 2) / (S ** 2).sum()
    print("\n" + "=" * 74)
    print("PART 2  PARTNER SUBSPACE inside L3H1   (fit on FIT, scored on EVAL)")
    print(f"  variance of the clean-vs-corrupt delta explained by top dims: "
          f"{' '.join(f'{e:.2f}' for e in ev[:6])}")

    base_c, base_k = M_CI, M_XI
    base_coupling = base_c - base_k
    print(f"\n  baseline: clean margin {base_c:.3f}   coupling {base_coupling:.3f}")
    print(f"  {'rank':>5s}  {'PARTNER subspace':>28s}   {'RANDOM rank-matched':>28s}")
    print(f"  {'':5s}  {'coupling':>12s} {'clean margin':>15s}   "
          f"{'coupling':>12s} {'clean margin':>15s}")

    rows = []
    grand = stable_rng("rand", 0)
    for k in args.ranks:
        U = torch.tensor(V[:k].T, dtype=torch.float32, device=adapter.device)   # [32,k]
        c, kk_ = run(EVAL, lambda: project_out(adapter, LAYER, HEAD, U))
        cpl = c - kk_

        rc, rm = [], []
        for r in range(args.n_random):
            g = torch.Generator(device="cpu").manual_seed(grand.randrange(1 << 30))
            Q, _ = torch.linalg.qr(torch.randn(DH, k, generator=g))
            Ur = Q.to(adapter.device)
            c2, k2 = run(EVAL, lambda: project_out(adapter, LAYER, HEAD, Ur))
            rc.append(c2 - k2)
            rm.append(c2)
        print(f"  {k:5d}  {cpl:12.3f} {c:15.3f}   {np.mean(rc):12.3f} {np.mean(rm):15.3f}")
        rows.append(dict(rank=k, partner_coupling=cpl, partner_margin=c,
                         random_coupling=float(np.mean(rc)), random_margin=float(np.mean(rm)),
                         random_coupling_sd=float(np.std(rc))))

    print("\n  INTERPRETATION")
    best = min(rows, key=lambda r: abs(r["partner_coupling"]))
    kept = best["partner_margin"] / base_c
    killed = 1 - best["partner_coupling"] / base_coupling
    print(f"  rank {best['rank']}: removes {100*killed:.0f}% of the coupling while keeping "
          f"{100*kept:.0f}% of the clean donor margin")
    print(f"    random rank-{best['rank']} subspace only removes "
          f"{100*(1-best['random_coupling']/base_coupling):.0f}% of the coupling")
    if killed > 0.7 and kept > 0.8:
        print("  => DEDICATED PARTNER FEATURE multiplexed inside a multifunctional head.")
    elif killed > 0.7:
        print("  => partner and local evidence are ENTANGLED in the same computation.")
    else:
        print("  => partner information is DISTRIBUTED; no low-rank direction captures it.")
    print("=" * 74)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "factorial": {"M_CI": M_CI, "M_XI": M_XI, "M_CA": M_CA, "M_XA": M_XA,
                      "partner_dependent_did": did, "local_role": local},
        "explained_variance": ev[:8].tolist(),
        "subspace": rows, "n_fit": len(FIT), "n_eval": len(EVAL),
    }, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
