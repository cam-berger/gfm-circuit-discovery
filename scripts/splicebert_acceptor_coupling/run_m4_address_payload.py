#!/usr/bin/env python
"""M4 -- are the class-specific heads ADDRESS-specific or PAYLOAD-specific?

Six heads survive a family-wise asymmetry threshold and all sit in layers 0-1: they matter
far more to the minor GC-AG class than to canonical GT-AG. None of them MEDIATES -- a
donor-token transplant moves nothing example-specific. So the asymmetry is real and it is
not carried as content at the donor. What is it then?

A head's context vector factorises exactly:

    ctx[i] = sum_j A[i,j] * V[j]        ADDRESS = A (where it reads)
                                        PAYLOAD = V (what is there)

Verified exact at every layer (max |A@V - head_output| = 0.000e+00), which is what makes the
two swappable independently. For a matched (GC recipient, GT source) pair with donor
positions d_r and d_s, the source is realigned by delta = d_r - d_s -- A rolled in BOTH
axes, V rolled along sequence -- so its donor-relative geometry lands in the recipient's
frame. Then:

    ADDRESS swap   ctx = A_source_shifted @ V_recipient    source's WHERE, recipient's WHAT
    PAYLOAD swap   ctx = A_recipient      @ V_source_shifted   recipient's WHERE, source's WHAT
    BOTH           ctx = A_source_shifted @ V_source_shifted
    IDENTITY       ctx = A_recipient      @ V_recipient    (memorial: must be exact)

The swap is whole-sequence, matching the mean-ablation that produced the asymmetry in the
first place; a donor-token-only swap would not be the same intervention.

    transport fraction  T = (Y_swapped - Y_GC) / (Y_GT - Y_GC)

Development tier, train split. chr8/9 never read; the validation reserve is not spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.circuits.hf_bert import patch_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs, substitute  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, assert_seal_intact,
    eligible_donor_motifs, flank_and_support_key, match_comparators, train_chromosomes,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402

TAG = "m4-address-payload-v1"
SEED, N_BOOT = 0, 2000

#: The six family-wise survivors, plus two shared load-bearing heads as contrast. Every
#: entry was fixed by the scan audit before this assay existed.
HEADS: list[tuple[int, int, str]] = [
    (0, 4, "class-specific, largest asymmetry"),
    (0, 5, "class-specific"),
    (0, 14, "class-specific, sign-flipped on GT"),
    (0, 3, "class-specific, sign-flipped on GT"),
    (1, 13, "class-specific"),
    (1, 6, "class-specific, weakest survivor"),
    (3, 1, "SHARED transport head, contrast"),
    (1, 12, "SHARED load-bearing, contrast"),
]


def capture_av(ad, enc, layers):
    """(A, V) per requested layer. A is [B,H,S,S]; V is [B,H,S,Dh]."""
    H, Dh = ad.num_heads, ad.head_dim
    store: dict[int, torch.Tensor] = {}
    handles = []

    def mk(layer):
        def hook(_m, _i, out):
            b, s, _ = out.shape
            store[layer] = out.detach().view(b, s, H, Dh).permute(0, 2, 1, 3).clone()
        return hook

    try:
        for L in layers:
            handles.append(ad.attention_self_modules[L].value.register_forward_hook(mk(L)))
        with torch.no_grad():
            out = ad.model(**enc, output_attentions=True)
        A = {L: out.attentions[L].detach().clone() for L in layers}
    finally:
        for h in handles:
            h.remove()
    del out
    return A, store


def shift_A(A, delta):
    """Roll an attention matrix in both axes so donor-relative geometry lands at d_r."""
    return torch.stack([torch.roll(A[b], shifts=(int(delta[b]), int(delta[b])), dims=(1, 2))
                        for b in range(A.shape[0])])


def shift_V(V, delta):
    return torch.stack([torch.roll(V[b], shifts=int(delta[b]), dims=1)
                        for b in range(V.shape[0])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    ad = load_adapter(a.model, pick_device(a.device), eager=True)  # sdpa hides attention
    task = CouplingTask(ad, batch=a.batch)
    off = ad.prefix_tokens

    D = _dataset(a.gtf, a.fasta, allowed_chromosomes=train_chromosomes())
    assert_seal_intact(D)
    pool = lambda m: [x for x in eligible_donor_motifs(D, "train", donor_motifs=(m,))  # noqa: E731
                      if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]
    gc_rows = pool(TREATMENT_MOTIF)
    if a.smoke:
        gc_rows = gc_rows[:4]
    elif a.limit:
        gc_rows = gc_rows[:a.limit]
    coh = match_comparators(gc_rows, pool(COMPARATOR_MOTIF), k=1, key_fn=flank_and_support_key)
    gc, gt = coh.treatment, coh.comparators
    genes = [x.gene_id for x in gc]
    heads = HEADS[:3] if a.smoke else HEADS
    layers = sorted({L for L, _, _ in heads})
    print(f"pairs={len(gc)} genes={len(set(genes))} heads={len(heads)} layers={layers}",
          flush=True)

    conds = ("identity", "address", "payload", "both")
    acc = {f"L{L}H{H}": {c: [] for c in conds} for L, H, _ in heads}
    base_gc, base_gt, ident_err = [], [], 0.0

    for i in range(0, len(gc), a.batch):
        rec, src = gc[i:i + a.batch], gt[i:i + a.batch]
        p_r = [acceptor_readout(x) for x in rec]
        p_s = [acceptor_readout(x) for x in src]
        delta = np.array([r.donor_pos - s.donor_pos for r, s in zip(rec, src)])

        per_cond = {c: [] for c in conds}
        for seq_of in (lambda x: x.sequence,
                       lambda x: substitute(x.sequence, x.donor_pos, MATCHED_OPERATOR)):
            enc_r = task.encode([seq_of(x) for x in rec], p_r)
            enc_s = task.encode([seq_of(x) for x in src], p_s)
            A_r, V_r = capture_av(ad, enc_r, layers)
            A_s, V_s = capture_av(ad, enc_s, layers)
            with torch.no_grad():
                m_r = task.margin(ad.model(**enc_r).logits.float(), p_r).cpu().numpy()
                m_s = task.margin(ad.model(**enc_s).logits.float(), p_s).cpu().numpy()
            per_cond.setdefault("_base_r", []).append(m_r)
            per_cond.setdefault("_base_s", []).append(m_s)

            for L, Hh, _ in heads:
                As, Vs = shift_A(A_s[L], delta), shift_V(V_s[L], delta)
                mats = {
                    "identity": (A_r[L], V_r[L]),
                    "address": (As, V_r[L]),
                    "payload": (A_r[L], Vs),
                    "both": (As, Vs),
                }
                for c, (Am, Vm) in mats.items():
                    ctx = torch.einsum("bhij,bhjd->bhid", Am.float(), Vm.float())
                    mask = torch.ones(ctx.shape[0], ctx.shape[2], dtype=torch.bool)
                    with torch.no_grad(), patch_heads(ad, {L: ctx}, [(L, Hh)],
                                                      token_mask=mask):
                        mm = task.margin(ad.model(**enc_r).logits.float(), p_r).cpu().numpy()
                    acc[f"L{L}H{Hh}"][c].append(mm)
                    if c == "identity":
                        ident_err = max(ident_err, float(np.abs(mm - m_r).max()))
            del A_r, V_r, A_s, V_s

        base_gc.append(per_cond["_base_r"][0] - per_cond["_base_r"][1])
        base_gt.append(per_cond["_base_s"][0] - per_cond["_base_s"][1])
        if (i // a.batch) % 10 == 0:
            print(f"  batch {i//a.batch + 1}/{(len(gc)+a.batch-1)//a.batch}", flush=True)

    y_gc = np.concatenate(base_gc)
    y_gt = np.concatenate(base_gt)

    def boot_T(y_sw):
        by: dict[str, list[int]] = {}
        for j, g in enumerate(genes):
            by.setdefault(str(g), []).append(j)
        keys = list(by)
        rng = np.random.default_rng(SEED)

        def T(idx):
            den = y_gt[idx].mean() - y_gc[idx].mean()
            return (y_sw[idx].mean() - y_gc[idx].mean()) / den if abs(den) > 1e-9 else np.nan

        est = T(np.arange(len(y_gc)))
        dr = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(keys), len(keys))
            dr.append(T(np.array([r for j in pick for r in by[keys[j]]])))
        dr = np.array([d for d in dr if np.isfinite(d)])
        lo, hi = np.quantile(dr, [0.025, 0.975])
        return float(est), float(lo), float(hi)

    print()
    print(f"{'head':>7} {'address T':>22} {'payload T':>22} {'both T':>22}")
    res = {}
    for L, Hh, note in heads:
        k = f"L{L}H{Hh}"
        row = {"note": note}
        for c in conds:
            ys = np.concatenate([np.asarray(v) for v in acc[k][c]][0::2]) - \
                 np.concatenate([np.asarray(v) for v in acc[k][c]][1::2])
            e, lo, hi = boot_T(ys)
            row[c] = {"mean_Y": float(ys.mean()), "T": e, "ci": [lo, hi]}
        res[k] = row
        f = lambda c: f"{row[c]['T']:+.3f} [{row[c]['ci'][0]:+.3f},{row[c]['ci'][1]:+.3f}]"  # noqa: E731
        print(f"{k:>7} {f('address'):>22} {f('payload'):>22} {f('both'):>22}")

    memorials = {
        "identity_swap_is_exact": {
            "PASS": ident_err < 1e-4, "max_abs_err": ident_err,
            "note": "A_rec @ V_rec must reproduce the unpatched margin; otherwise the "
                    "factorisation is approximate and every swap measures its error",
        },
        "arms_paired_one_to_one": {"PASS": len(gc) == len(gt), "n": len(gc)},
    }
    all_pass = all(m["PASS"] for m in memorials.values())

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "gfmcd.m4.address_payload/1",
        "tier": "development", "split": "train",
        "sealed_chr8_chr9_opened": False, "validation_reserve_spent": False,
        "attention_impl": "eager", "n_pairs": len(gc), "n_genes": len(set(genes)),
        "baseline": {"GC_mean_Y": float(y_gc.mean()), "GT_mean_Y": float(y_gt.mean())},
        "transport": res, "memorials": memorials, "all_memorials_pass": all_pass,
        "licenses": ("Development-tier decomposition of each head's contribution into an "
                     "attention (address) and a value (payload) factor. No route, delivery, "
                     "or biological claim follows."),
    }, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m4_address_payload", sys.argv,
        config={"n_pairs": len(gc), "heads": [f"L{L}H{H}" for L, H, _ in heads]},
        seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; chr8/9 not opened; validation reserve not spent."])
    for f_ in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f_)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f_}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"\nidentity memorial max abs err {ident_err:.2e}  pass={all_pass} -> {out}")


if __name__ == "__main__":
    main()
