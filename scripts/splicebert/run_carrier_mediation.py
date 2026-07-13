#!/usr/bin/env python
"""DOES THE CARRIER REACH THE COUPLING *THROUGH L2H15*?   The mediation test that closes the loop.

ESTABLISHED.
  * SpliceBERT's position-embedding matrix carries a PERIOD-3 CARRIER (Fourier bins 339-345,
    847x the median bin power -- the ONLY high-frequency structure in P's spectrum).
  * SIX heads read it out.  R(k) = mean_i (W_Q p_i).(W_K p_{i+k}), computed from the WEIGHTS
    ALONE, is a perfect stride-3 comb for each.  Delete the carrier and comb amplitude falls to
    3.4%; delete an energy-matched band elsewhere and it stays at 99.8%.
  * Deleting the carrier costs 0.159 nats of masked-base log-prob in CODING EXONS vs 0.042 in
    introns (DiD = 0.117).  The energy-matched control band damages exons and introns EQUALLY
    (0.029 / 0.031, DiD = -0.002).  Monotonic dose-response.
  * AND IT DESTROYS THE DONOR-ACCEPTOR COUPLING: 0.946 -> 0.027 (2.8%), monotonically.
    The control band leaves the coupling at 102.5%.

ONE OF THE SIX COMBS IS **L2H15 -- THE ADDRESSER** of the coupling circuit (D12: the L2H15 ->
L3H1 edge is CLOSED, whole-output rescue 0.959 [0.875, 1.052]).  It was found by an ablation
scan on the splicing task that knew nothing about periodicity.

SO THE PROPOSED CHAIN IS:

    period-3 carrier in P  ->  L2H15's comb  ->  the acceptor's ADDRESS  ->  L3H1's read
                                                                        ->  donor-acceptor coupling

THE TEST.  Delete the carrier (the coupling dies).  Then restore ONE head's output to the value
it had in the INTACT model, and see how much coupling comes back.

    rescue(H) = [coupling(carrier deleted, H restored) - coupling(carrier deleted)]
              / [coupling(intact)                     - coupling(carrier deleted)]

  L2H15 rescues a lot   -> the carrier reaches the coupling THROUGH THE ADDRESSER.  The two
                           studies are ONE: the splicing circuit runs on the positional carrier.
  L2H15 rescues ~0      -> the carrier damages the coupling by some other route.  The chain is
                           wrong and must not be drawn.

CONTROLS, every one of which can fail:
  * L3H1 (the READER)          -- downstream of L2H15. Should rescue at least as much.
  * L1H12 (a comb, NOT in C)   -- a comb that is not the addresser. Tests "any comb will do".
  * matched random non-comb heads -- tests "restoring any head helps".
  * ALL 6 COMBS together       -- the ceiling for a comb-mediated account.
  * EVERY head in layer 2      -- if L2H15 alone matches this, the edge is closed (as in D12).
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

from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask,
    gboot_ratio, load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402

BAND = list(range(339, 346))
COMBS = [(0, 0), (0, 11), (1, 10), (1, 12), (2, 2), (2, 15)]
MIN_LEN = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nulls", type=int, default=24)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/phase-lattice/carrier_mediation.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    L, H = ad.num_layers, ad.num_heads

    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P = P0.float().cpu().numpy()
    N, _ = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    G = F.copy()
    for b in BAND:
        G[b] = 0
    P_NO3 = torch.tensor(np.fft.irfft(G, n=N, axis=0) + mu, dtype=P0.dtype, device=P0.device)

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n * 3, task=task,
                         tag=f"mediate-{args.split}", min_effect=None)
    X = [x for x in X if x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    genes = [x.gene_id for x in X]
    print(f"{len(X)} exons, {len(set(genes))} genes   device={dev}\n")

    rng = np.random.default_rng(0)
    noncomb = [(l, h) for l in range(L) for h in range(H) if (l, h) not in set(COMBS)]
    RAND1 = [noncomb[i] for i in rng.choice(len(noncomb), 1, replace=False)]
    RAND6 = [noncomb[i] for i in rng.choice(len(noncomb), 6, replace=False)]
    L2ALL = [(2, h) for h in range(H)]

    # SIZE-MATCHED NULLS, not a single draw.  Restoring ANY head re-injects information the
    # intact position embeddings produced, so restoration is PARTLY A GENERIC UNDO -- six random
    # non-comb heads recover 0.507 on their own.  A single random draw is an alarm, not a null.
    # That is precisely the error that produced R12.  So: a DISTRIBUTION of single random heads
    # for L2H15, and a DISTRIBUTION of random 6-head sets for the six combs.
    NULL1 = [[noncomb[i]] for i in rng.choice(len(noncomb), args.nulls, replace=False)]
    NULL6 = [[noncomb[i] for i in rng.choice(len(noncomb), 6, replace=False)]
             for _ in range(args.nulls)]

    ARMS: list[tuple[str, list]] = [
        ("L2H15  (the ADDRESSER)", [ADDRESSER]),
        ("L3H1   (the READER)", [READER]),
        ("L1H12  (a comb, not in C)", [(1, 12)]),
        ("all 6 COMBS", COMBS),
        ("ALL 16 layer-2 heads", L2ALL),
    ]
    ARMS += [(f"__null1_{i}", hs) for i, hs in enumerate(NULL1)]
    ARMS += [(f"__null6_{i}", hs) for i, hs in enumerate(NULL6)]

    intact, killed = [], []
    num = {n: [] for n, _ in ARMS}

    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            dn = [x.donor_pos for x in ch]
            cl = task.encode([x.sequence for x in ch], dn)
            co = task.encode([substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT)
                              for x in ch], dn)

            # --- INTACT model: coupling, and the head outputs we will restore
            emb.weight.data = P0.clone()
            hc = capture_head_outputs(ad, lambda: ad.model(**cl))
            ho = capture_head_outputs(ad, lambda: ad.model(**co))
            m_cl = task.margin(ad.model(**cl).logits.float(), dn)
            m_co = task.margin(ad.model(**co).logits.float(), dn)
            intact += (m_cl - m_co).tolist()

            # --- CARRIER DELETED
            emb.weight.data = P_NO3.clone()
            k_cl = task.margin(ad.model(**cl).logits.float(), dn)
            k_co = task.margin(ad.model(**co).logits.float(), dn)
            kill = k_cl - k_co
            killed += kill.tolist()

            # --- CARRIER DELETED + one head restored to its INTACT value
            for name, heads in ARMS:
                with patch_heads(ad, hc, heads):
                    r_cl = task.margin(ad.model(**cl).logits.float(), dn)
                with patch_heads(ad, ho, heads):
                    r_co = task.margin(ad.model(**co).logits.float(), dn)
                num[name] += ((r_cl - r_co) - kill).tolist()
            print(f"  {min(i+task.batch, len(X))}/{len(X)}", end="\r", flush=True)
    emb.weight.data = P0.clone()
    print()

    intact = np.asarray(intact)
    killed = np.asarray(killed)
    dnm = intact - killed

    print("=" * 92)
    print("DOES THE PERIOD-3 CARRIER REACH THE COUPLING THROUGH L2H15?")
    print(f"  coupling, intact model        {intact.mean():+.4f}")
    print(f"  coupling, carrier deleted     {killed.mean():+.4f}"
          f"   ({killed.mean()/intact.mean():.1%} of baseline)")
    print(f"  the carrier supplies          {dnm.mean():+.4f}")
    print(f"\n  restore heads' outputs to their INTACT values, carrier still deleted:")
    print(f"  {'':28s} {'rescue':>8s} {'95% CI':>20s}")
    R: dict = {}
    for name, heads in ARMS:
        v = ratio_of_means(num[name], dnm)
        R[name] = dict(rescue=v, heads=[list(h) for h in heads])
        if name.startswith("__"):
            continue
        lo, hi = gboot_ratio(num[name], dnm, genes)
        R[name].update(lo=lo, hi=hi)
        print(f"  {name:28s} {v:8.3f}  [{lo:+.3f}, {hi:+.3f}]")

    n1 = np.array([R[f"__null1_{i}"]["rescue"] for i in range(len(NULL1))])
    n6 = np.array([R[f"__null6_{i}"]["rescue"] for i in range(len(NULL6))])
    a = R["L2H15  (the ADDRESSER)"]["rescue"]
    c6 = R["all 6 COMBS"]["rescue"]
    l2 = R["ALL 16 layer-2 heads"]["rescue"]

    p_a = float((1 + np.sum(n1 >= a)) / (1 + len(n1)))
    p_c = float((1 + np.sum(n6 >= c6)) / (1 + len(n6)))
    print("\n  SIZE-MATCHED NULLS (restoring ANY head is partly a generic undo -- so this, "
          "not a single draw, is the test):")
    print(f"    {len(n1)} random SINGLE non-comb heads:  mean {n1.mean():.3f}  "
          f"max {n1.max():.3f}   |  L2H15 = {a:.3f}   add-one p = {p_a:.4f}")
    print(f"    {len(n6)} random SIX-head sets:          mean {n6.mean():.3f}  "
          f"max {n6.max():.3f}   |  6 COMBS = {c6:.3f}   add-one p = {p_c:.4f}")

    print("\n" + "-" * 92)
    if p_a < 0.05 and p_c < 0.05:
        print("  => THE CARRIER REACHES THE COUPLING THROUGH THE COMBS, AND THROUGH THE ADDRESSER.")
        print(f"     Restoring the six comb heads -- inside a model whose positional carrier has")
        print(f"     been deleted -- brings back {c6:.1%} of the coupling the carrier supplies;")
        print(f"     {len(n6)} size-matched random 6-head sets bring back {n6.mean():.1%} "
              f"(max {n6.max():.1%}), p = {p_c:.4f}.")
        print(f"     L2H15 ALONE brings back {a:.1%}, against {n1.mean():.1%} for a random single")
        print(f"     head (max {n1.max():.1%}), p = {p_a:.4f}.")
        print(f"\n     period-3 carrier in P  ->  L2H15's comb  ->  the acceptor's ADDRESS")
        print(f"                            ->  L3H1's read     ->  donor-acceptor coupling")
    elif p_c < 0.05:
        print(f"  => Mediated by the COMBS as a set (p = {p_c:.4f}), but L2H15 alone is NOT an")
        print(f"     outlier against single-head restoration (p = {p_a:.4f}). Claim the set, not")
        print("     the head.")
    else:
        print(f"  => NOT ESTABLISHED. The combs' rescue ({c6:.3f}) is not an outlier against "
              f"size-matched random 6-head sets (mean {n6.mean():.3f}, p = {p_c:.4f}).")
        print("     Restoring any six heads partly undoes the deletion. DO NOT DRAW THE CHAIN.")

    out = REPO / args.out
    out.write_text(json.dumps({
        "n": len(X), "band": BAND,
        "coupling_intact": float(intact.mean()), "coupling_killed": float(killed.mean()),
        "supplied": float(dnm.mean()), "rescue": R,
    }, indent=1))
    ExperimentManifest.create(
        "phase_lattice:carrier_mediation", sys.argv,
        config={"n": len(X), "band": BAND},
        notes=["Tests whether the period-3 carrier's effect on the donor-acceptor coupling is "
               "MEDIATED by L2H15, the ADDRESSER -- which is one of the six comb heads that read "
               "the carrier out.",
               "Every control can fail: a random head, six random heads, a comb that is not the "
               "addresser, and all 16 layer-2 heads (the ceiling)."],
    ).write(out.parent / "carrier_mediation.manifest.json")
    print("=" * 92)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
