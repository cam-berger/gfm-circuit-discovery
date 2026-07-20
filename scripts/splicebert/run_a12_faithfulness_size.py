#!/usr/bin/env python
"""Donor-graph experiment 16 -- A12: backward faithfulness-size curve.

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A12_FAITHFULNESS_SIZE.md.

F(K) = A6 behaviour-level faithfulness (keep K clean, scrub complement to CORRUPT).  Rank nodes by
F-loss when removed FROM THE FULL MODEL (well-identified backward, unlike forward at F~0; serial-chain
members all score high since removing any link breaks the chain), on split D; then the keep-top-k curve
on held-out H -> node counts for 50/80/90/95% faithfulness.  Reuses A6's scrub.  No new kernel.
NOTE (R21): F(ALL)=1 / F(empty)=0 are embed-clamp-forced normalization ANCHORS, not memorials; the F(K)
scrub is validated in A6.  --smoke = anchors + a few conditions on tiny-n (val sealed).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file       # noqa: E402
from gfmcd.circuits.hf_bert import _context_from_output, _split_context, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import (                                          # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _block_write, clamp_write,
    gboot_ratio, load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.provenance import clamp_embed, embed_module               # noqa: E402

EMBED = "embed"
CURVE_K = [103, 96, 88, 80, 72, 64, 56, 48, 40, 34, 28, 22, 16, 10, 6, 3, 1]
SEED = 0


def split_nodes(S):
    Sh = [n for n in S if isinstance(n, tuple) and isinstance(n[1], int)]
    Sm = [n for n in S if isinstance(n, tuple) and n[1] == "mlp"]
    return Sh, Sm, (EMBED in S)


def capture(ad, inputs):
    heads, mlps, ebox = {}, {}, {}
    handles = []
    for l, mod in enumerate(ad.attention_self_modules):
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, l=l: heads.__setitem__(
                l, _split_context(_context_from_output(o), ad.num_heads).detach().clone())))
    for l in range(ad.num_layers):
        handles.append(_block_write(ad, l, "mlp").register_forward_hook(
            lambda _m, _i, o, l=l: mlps.__setitem__((l, "mlp"), o.detach().clone())))
    handles.append(embed_module(ad).register_forward_hook(lambda _m, _i, o: ebox.__setitem__("e", o.detach().clone())))
    try:
        out = ad.model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return out, heads, mlps, ebox["e"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a12_faithfulness_size.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out; pf_path = out.parent / "preflight_a12_faithfulness_size.json"
    dev = pick_device(args.device); ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch); off = task.off

    n = min(args.n, 24) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a12-{args.split}", min_effect=None)
    if len(X) < 16:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})

    ALL = set([(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)]
              + [(l, "mlp") for l in range(ad.num_layers)] + [EMBED])
    ALL_L = sorted(ALL, key=str)
    ug = sorted({x.gene_id for x in X}); Dg = set(ug[::2])

    def run_conditions(condsets):
        """condsets: dict name -> node-set K to keep clean. Returns per-example (M-m_co) per name +
        coup, genes, and the two anchors."""
        num = {k: [] for k in condsets}; coup, genes = [], []
        anch = {"full_err": 0.0, "empty_err": 0.0}
        for i in range(0, len(X), args.batch):
            ch = X[i:i + args.batch]; cl, co, dn = task.pair(ch)
            oc, hco, mco, eco = capture(ad, co)   # corrupt caches (scrub source)
            m_cl = task.margin(ad.model(**cl).logits.float(), dn)
            m_co = task.margin(oc.logits.float(), dn)
            B, S = cl["input_ids"].shape
            allpos = torch.arange(S, device=dev)[None].expand(B, S)

            def F_margin(K):
                comp = ALL - set(K)
                Sh, Sm, Se = split_nodes(comp)
                with ExitStack() as st:
                    if Sh: st.enter_context(patch_heads(ad, hco, Sh))
                    if Sm: st.enter_context(clamp_write(ad, Sm, mco))
                    if Se: st.enter_context(clamp_embed(ad, eco, allpos))
                    return task.margin(ad.model(**cl).logits.float(), dn)
            # anchors
            anch["full_err"] = max(anch["full_err"], float((F_margin(ALL) - m_cl).abs().max()))
            anch["empty_err"] = max(anch["empty_err"], float((F_margin(set()) - m_co).abs().max()))
            for name, K in condsets.items():
                num[name] += (F_margin(K) - m_co).cpu().tolist()
            coup += (m_cl - m_co).cpu().tolist(); genes += [x.gene_id for x in ch]
            print(f"  {min(i + args.batch, len(X))}/{len(X)} [{len(condsets)} conds]", end="\r", flush=True)
        print()
        return num, coup, genes, anch

    # ---- Phase 1: rank nodes by F-loss removing from FULL, on D ----
    rank_conds = {f"drop:{nd}": (ALL - {nd}) for nd in ALL_L}
    if args.smoke:
        rank_conds = {k: rank_conds[k] for k in list(rank_conds)[:6]}
    num1, coup1, genes1, anch = run_conditions(rank_conds)
    inD = np.array([g in Dg for g in genes1]); inH = ~inD

    checks = {
        "full_is_clean": {"PASS": bool(anch["full_err"] < 1e-4), "max_abs_err": anch["full_err"]},
        "empty_is_corrupt": {"PASS": bool(anch["empty_err"] < 1e-4), "max_abs_err": anch["empty_err"]},
        "positive_control_partner_coupling": {"PASS": bool(np.mean(coup1) >= 0.20), "coupling": float(np.mean(coup1))},
        "both_genes_split": {"PASS": bool(inD.sum() >= (5 if args.smoke else 40) and inH.sum() >= (5 if args.smoke else 40)),
                             "nD": int(inD.sum()), "nH": int(inH.sum())},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"commit": git_commit(REPO), "n": len(X), "checks": checks}
    pf["PASS"] = bool(all(c["PASS"] for c in pf["checks"].values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True); pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}" + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))
    if args.smoke:
        print("SMOKE: anchors + a few conditions (val sealed)."); return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting (prereg s2)."); return

    def F_of(name, num, coup, mask):
        return ratio_of_means(np.array(num[name])[mask].tolist(), np.array(coup)[mask].tolist())

    loss_D = {}
    for nd in ALL_L:
        loss_D[nd] = 1.0 - F_of(f"drop:{nd}", num1, coup1, inD)   # F(ALL)=1 by construction
    order = sorted(ALL_L, key=lambda nd: -loss_D[nd])              # most-important (highest loss) first

    # ---- Phase 2: keep-top-k curve on held-out H ----
    curve_conds = {f"top:{k}": set(order[:k]) for k in CURVE_K}
    num2, coup2, genes2, _ = run_conditions(curve_conds)
    inH2 = np.array([g not in Dg for g in genes2]); gH = [g for g in genes2 if g not in Dg]
    curve = {}
    for k in CURVE_K:
        vals = (np.array(num2[f"top:{k}"]) )[inH2].tolist(); cc = np.array(coup2)[inH2].tolist()
        curve[k] = {"F_H": ratio_of_means(vals, cc), "ci": list(gboot_ratio(vals, cc, gH))}

    def node_count_for(frac):
        best = None
        for k in sorted(CURVE_K):
            if curve[k]["F_H"] >= frac:
                best = k; break
        return best
    counts = {str(int(f * 100)): node_count_for(f) for f in (0.50, 0.80, 0.90, 0.95)}

    core90 = order[:(node_count_for(0.90) or 103)]
    def compose(nodes):
        heads = [n for n in nodes if isinstance(n, tuple) and isinstance(n[1], int)]
        mlps = [n for n in nodes if isinstance(n, tuple) and n[1] == "mlp"]
        by_layer = {}
        for (l, h) in heads: by_layer.setdefault(l, 0); by_layer[l] += 1
        return {"n_heads": len(heads), "n_mlps": len(mlps), "embed": EMBED in nodes,
                "heads_by_layer": by_layer, "mlps": sorted(l for (l, _) in mlps)}

    result = {
        "n": len(X), "n_genes": len(ug), "nD": int(inD.sum()), "nH": int(inH.sum()),
        "coupling": float(np.mean(coup1)), "curve_K": CURVE_K,
        "curve_H": {str(k): curve[k] for k in CURVE_K},
        "node_counts_for_faithfulness": counts,
        "faithful_core_90pct": {"size": node_count_for(0.90), "composition": compose(core90),
                                "top20_nodes": [str(n) for n in order[:20]]},
        "shape": ("knee" if (counts["90"] and counts["90"] <= 40) else "gradual"),
        "greedy_note": "static backward rank (upper bound on minimal faithful set at each F level)",
        "preflight_PASS": pf["PASS"], "meets_power_target": bool(len(X) >= 150),
        "confirmatory": bool(pf["PASS"] and len(X) >= 150),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s4)"},
    }
    out.write_text(json.dumps(result, indent=1, default=str))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a12_faithfulness_size", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": len(ug), "curve_K": CURVE_K, "preflight_PASS": pf["PASS"]},
        seeds={"gene_split": SEED, "gene_bootstrap": 0},
        notes=["Backward faithfulness-size curve: rank by F-loss from full (D), keep-top-k curve (H). "
               "Reuses A6 scrub. Static-rank upper bound. R21 anchors.", f"status={status}."])
    man.add_input(str(ckpt / "pytorch_model.bin")); man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    print("  F-vs-|K| (held-out): " + "  ".join(f"{k}:{curve[k]['F_H']:.2f}" for k in CURVE_K))
    print(f"  node counts for faithfulness: {counts}  (shape={result['shape']})")
    print(f"  faithful core (90%): size={node_count_for(0.90)}, {compose(core90)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
