#!/usr/bin/env python
"""Donor-graph experiment 10 -- A6: circuit faithfulness / necessity / minimality.

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_A6_CIRCUIT_FAITHFULNESS.md.

Node-level activation patching (post-LN => patch module OUTPUTS, downstream recomputes).  Every
residual writer is patchable: embedding output (clamp_embed), every attention head (patch_heads),
every MLP write (clamp_write on layer.output.dense).  All conditions noise FROM the clean run by
setting nodes to the CORRUPT run's cached activations:

  F(C) faithfulness = complement(C) -> corrupt (circuit stays clean); F=(M-M_co)/(M_cl-M_co)
  N(C) necessity    = C -> corrupt (complement stays clean);          N=(M_cl-M)/(M_cl-M_co)
  minimality        = F(C\\{n}) for each circuit node n
  completeness scan = F(C u {m}) for each non-circuit head/MLP m (residual completeness)
  size-matched null = F(random same-composition node sets)

Two EXACT arithmetic memorials (empty->corrupt == M_co, full->clean == M_cl, both <1e-4) prove the
node cover is complete AND are the health guard (the scrub interpolates two verified-healthy runs).

Reuses patch_heads / clamp_write / clamp_embed / matched_random_head_sets -- NO new kernel.
--smoke = memorials + plumbing on tiny-n (val sealed).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import random as _random
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file  # noqa: E402
from gfmcd.circuits.hf_bert import (  # noqa: E402
    _context_from_output, _split_context, matched_random_head_sets, patch_heads,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _block_write, clamp_write,
    gboot_ratio, load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.provenance import clamp_embed, embed_module  # noqa: E402

# ----- FROZEN circuit C (prereg s1) -------------------------------------------------------
C_HEADS = [(1, 12), (2, 15), (3, 1), (4, 0), (4, 2), (5, 11)]
C_MLPS = [2, 3, 4]
EMBED = "embed"
SEED = 0


def node_reps(n_layers: int, n_heads: int):
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    mlps = [(l, "mlp") for l in range(n_layers)]
    return set(heads) | set(mlps) | {EMBED}


def split_nodes(S):
    Sh = [n for n in S if isinstance(n, tuple) and isinstance(n[1], int)]
    Sm = [n for n in S if isinstance(n, tuple) and n[1] == "mlp"]
    return Sh, Sm, (EMBED in S)


def capture(ad, inputs):
    """One forward; return (output, heads{l:[B,H,S,Dh]}, mlps{(l,'mlp'):[B,S,H]}, embed[B,S,H])."""
    heads, mlps, ebox = {}, {}, {}
    handles = []
    for l, mod in enumerate(ad.attention_self_modules):
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, l=l: heads.__setitem__(
                l, _split_context(_context_from_output(o), ad.num_heads).detach().clone())))
    for l in range(ad.num_layers):
        handles.append(_block_write(ad, l, "mlp").register_forward_hook(
            lambda _m, _i, o, l=l: mlps.__setitem__((l, "mlp"), o.detach().clone())))
    handles.append(embed_module(ad).register_forward_hook(
        lambda _m, _i, o: ebox.__setitem__("e", o.detach().clone())))
    try:
        out = ad.model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return out, heads, mlps, ebox["e"]


@contextmanager
def scrub(ad, S, hsrc, msrc, esrc, allpos):
    """Patch every node in S to its source-run cached value."""
    Sh, Sm, Se = split_nodes(S)
    with ExitStack() as st:
        if Sh:
            st.enter_context(patch_heads(ad, hsrc, Sh))
        if Sm:
            st.enter_context(clamp_write(ad, Sm, msrc))
        if Se:
            st.enter_context(clamp_embed(ad, esrc, allpos))
        yield


def _reconcile(ad, task, X, batch, dev, C, compl_C, outdir):
    """DIAGNOSTIC (post-hoc, added after seeing F(C)=0.005): does injecting the circuit's CLEAN
    OUTPUTS recover (denoising, C16/C17-style) even though letting it RECOMPUTE on a scrubbed
    network (F, complement-noising) does not?  Tests the delivery-vs-source asymmetry.  D keeps the
    corruption at the embedding (C\\embed) so it is a genuine denoising, not un-corruption."""
    conds = {"F_C": (compl_C, "cl", "co"), "N_C": (C, "cl", "co"),
             "N_noembed": (C - {EMBED}, "cl", "co"), "D_noembed": (C - {EMBED}, "co", "cl")}
    raw = {k: [] for k in conds}
    mcl_all, mco_all, genes = [], [], []
    for i in range(0, len(X), batch):
        ch = X[i:i + batch]
        cl, co, dn = task.pair(ch)
        oc, hcl, mclw, ecl = capture(ad, cl)
        od, hco, mcow, eco = capture(ad, co)
        m_cl = task.margin(oc.logits.float(), dn)
        m_co = task.margin(od.logits.float(), dn)
        B, S = cl["input_ids"].shape
        allpos = torch.arange(S, device=dev)[None].expand(B, S)
        for name, (nodes, base, src) in conds.items():
            hs, ms, es = (hco, mcow, eco) if src == "co" else (hcl, mclw, ecl)
            with scrub(ad, nodes, hs, ms, es, allpos):
                lg = ad.model(**(cl if base == "cl" else co)).logits.float()
            raw[name] += task.margin(lg, dn).cpu().tolist()
        mcl_all += m_cl.cpu().tolist(); mco_all += m_co.cpu().tolist()
        genes += [x.gene_id for x in ch]
    coup = [a - b for a, b in zip(mcl_all, mco_all)]
    Fn = [m - c for m, c in zip(raw["F_C"], mco_all)]
    Dn = [m - c for m, c in zip(raw["D_noembed"], mco_all)]
    res = {"n": len(X), "n_genes": len(set(genes)), "coupling": float(np.mean(coup)),
           "F_C": ratio_of_means(Fn, coup),
           "N_C": ratio_of_means([c - m for m, c in zip(raw["N_C"], mcl_all)], coup),
           "N_noembed": ratio_of_means([c - m for m, c in zip(raw["N_noembed"], mcl_all)], coup),
           "D_noembed": ratio_of_means(Dn, coup),
           "D_noembed_ci": list(gboot_ratio(Dn, coup, genes)),
           "F_C_ci": list(gboot_ratio(Fn, coup, genes)),
           "note": ("DIAGNOSTIC reconciliation (post-hoc). F = complement-noising (circuit "
                    "RECOMPUTES on a scrubbed residual). D = denoising injection of the circuit's "
                    "CLEAN OUTPUTS with the corruption kept at the embedding (C\\embed). "
                    "F<<D => delivery-sparse / source-distributed asymmetry (reconciles A6 with "
                    "C16/C17). Not part of the frozen A6 protocol.")}
    (outdir / "a6_reconcile.json").write_text(json.dumps(res, indent=1))
    print(f"RECONCILE: F(C)={res['F_C']:+.3f}{res['F_C_ci']}  N(C)={res['N_C']:+.3f}  "
          f"N(C\\embed)={res['N_noembed']:+.3f}  D(C\\embed)={res['D_noembed']:+.3f}"
          f"{res['D_noembed_ci']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reconcile", action="store_true",
                    help="post-hoc diagnostic: F (noising) vs D (denoising injection) asymmetry")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a6_circuit_faithfulness.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    pf_path = out.parent / "preflight_a6_circuit_faithfulness.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)

    n = min(args.n, 24) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a6-{args.split}",
                         min_effect=None)
    if len(X) < 16:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})

    ALL = node_reps(ad.num_layers, ad.num_heads)
    C = set(C_HEADS) | {(l, "mlp") for l in C_MLPS} | {EMBED}
    compl_C = ALL - C
    non_circuit = sorted([n for n in compl_C if n != EMBED], key=lambda z: (str(z[0]), str(z[1])))

    if args.reconcile:
        _reconcile(ad, task, X, args.batch, dev, C, compl_C, out.parent)
        return

    # size-matched null: 6 random heads + 3 random MLPs + embed (fixed)
    n_null = 3 if args.smoke else args.n_null
    null_heads = matched_random_head_sets(ad, C_HEADS, n=n_null, seed=SEED, exclude_selected=True)
    rr = _random.Random(SEED)
    null_sets = []
    for hs in null_heads:
        ms = tuple(sorted(rr.sample(range(ad.num_layers), len(C_MLPS))))
        null_sets.append(set(hs) | {(l, "mlp") for l in ms} | {EMBED})

    # completeness/minimality scope (trimmed in smoke to exercise the code paths only)
    comp_scan = non_circuit[:4] if args.smoke else non_circuit
    min_scan = sorted(C, key=str)

    # accumulators (per-example, gene-aligned)
    coup_all: list[float] = []
    genes: list[str] = []
    num = {}                                   # key -> list of (M - m_co)
    nec_num: list[float] = []                  # (m_cl - M) for N(C)
    mem_empty_err = 0.0
    mem_full_err = 0.0

    def acc(key, vals):
        num.setdefault(key, []).extend(vals)

    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        out_cl, hcl, mcl_w, ecl = capture(ad, cl)
        out_co, hco, mco_w, eco = capture(ad, co)
        m_cl = task.margin(out_cl.logits.float(), dn)
        m_co = task.margin(out_co.logits.float(), dn)
        coup = (m_cl - m_co)
        B, S = cl["input_ids"].shape
        allpos = torch.arange(S, device=dev)[None].expand(B, S)

        def marg(nodeset, base, src):          # src 'co' or 'cl'
            hsrc, msrc, esrc = (hco, mco_w, eco) if src == "co" else (hcl, mcl_w, ecl)
            with scrub(ad, nodeset, hsrc, msrc, esrc, allpos):
                lg = ad.model(**base).logits.float()
            return task.margin(lg, dn)

        # memorials
        me = marg(ALL, cl, "co")               # empty circuit -> corrupt
        mf = marg(ALL, co, "cl")               # full circuit  -> clean
        mem_empty_err = max(mem_empty_err, float((me - m_co).abs().max()))
        mem_full_err = max(mem_full_err, float((mf - m_cl).abs().max()))

        # F(C), N(C)
        mF = marg(compl_C, cl, "co")
        mN = marg(C, cl, "co")
        acc("F_C", (mF - m_co).cpu().tolist())
        nec_num.extend((m_cl - mN).cpu().tolist())

        # minimality: F(C \ {n})
        for nnode in min_scan:
            mm = marg(compl_C | {nnode}, cl, "co")
            acc(("min", str(nnode)), (mm - m_co).cpu().tolist())

        # completeness: F(C u {m})
        for mnode in comp_scan:
            mm = marg(compl_C - {mnode}, cl, "co")
            acc(("cmp", str(mnode)), (mm - m_co).cpu().tolist())

        # size-matched null
        for k, R in enumerate(null_sets):
            mm = marg(ALL - R, cl, "co")
            acc(("null", k), (mm - m_co).cpu().tolist())

        coup_all.extend(coup.cpu().tolist())
        genes.extend([x.gene_id for x in ch])
        print(f"  batch {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    coupling = float(np.mean(coup_all))
    checks = {
        "empty_circuit_is_corrupt": {"PASS": bool(mem_empty_err < 1e-4), "max_abs_err": mem_empty_err},
        "full_circuit_is_clean": {"PASS": bool(mem_full_err < 1e-4), "max_abs_err": mem_full_err},
        "positive_control_partner_coupling": {"PASS": bool(coupling >= 0.20), "coupling": coupling},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    hb = float(task.mlm_health(X)) if not args.smoke else None
    if hb is not None:
        checks["health_baseline"] = {"PASS": bool(hb >= 0.5), "clean_mlm_top1": hb}
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"commit": git_commit(REPO), "n": len(X), "checks": checks}
    pf["PASS"] = bool(all(c["PASS"] for c in pf["checks"].values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        print("SMOKE: preflight/memorials + plumbing only; no F/N/scan reported (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting faithfulness (prereg s3).")
        return

    def R(key):
        return ratio_of_means(num[key], coup_all)

    F_C = R("F_C")
    N_C = ratio_of_means(nec_num, coup_all)
    F_C_ci = list(gboot_ratio(num["F_C"], coup_all, genes))
    N_C_ci = list(gboot_ratio(nec_num, coup_all, genes))

    # null distribution of F
    null_F = sorted(R(("null", k)) for k in range(len(null_sets)))
    null_p95 = float(np.percentile(null_F, 95))
    # add-one p-value: fraction of null F >= F(C)
    p_faith = (sum(f >= F_C for f in null_F) + 1) / (len(null_F) + 1)

    # minimality: dF_n = F(C) - F(C\{n}), CI on the per-example difference
    minimality = {}
    for nnode in min_scan:
        key = ("min", str(nnode))
        dnum = [a - b for a, b in zip(num["F_C"], num[key])]      # (M_FC - M_drop) per example
        d_ci = list(gboot_ratio(dnum, coup_all, genes))
        minimality[str(nnode)] = {"F_drop": F_C - R(key), "dF_ci": d_ci,
                                  "load_bearing": bool(d_ci[0] > 0)}

    # completeness: dF_m = F(C u {m}) - F(C); addition-null = per-node gains vs their spread
    completeness = {}
    for mnode in comp_scan:
        key = ("cmp", str(mnode))
        completeness[str(mnode)] = {"F_gain": R(key) - F_C}
    add_gains = sorted((v["F_gain"] for v in completeness.values()), reverse=True)
    add_p95 = float(np.percentile(add_gains, 95)) if add_gains else 0.0
    top_omitted = sorted(completeness.items(), key=lambda kv: -kv[1]["F_gain"])[:8]
    incomplete = [k for k, v in completeness.items() if v["F_gain"] > max(add_p95, 0.05)]

    faithful = bool(F_C >= 0.80 and F_C > null_p95 and p_faith < 0.05)
    necessary = bool(N_C >= 0.80 and N_C > null_p95)
    minimal = bool(all(v["load_bearing"] for v in minimality.values()))
    complete = bool(len(incomplete) == 0)

    result = {
        "n": len(X), "n_genes": len(set(genes)), "strands": strands, "coupling": coupling,
        "circuit": {"heads": [list(h) for h in C_HEADS], "mlps": C_MLPS, "embed": True},
        "F_C": F_C, "F_C_ci": F_C_ci, "N_C": N_C, "N_C_ci": N_C_ci,
        "null_F_mean": float(np.mean(null_F)), "null_F_p95": null_p95, "p_faithful": p_faith,
        "residual_1_minus_F": 1.0 - F_C,
        "minimality": minimality,
        "completeness_top_omitted": {k: v for k, v in top_omitted},
        "completeness_incomplete_nodes": incomplete, "addition_null_p95": add_p95,
        "verdict": {"faithful": faithful, "necessary": necessary,
                    "minimal": minimal, "complete": complete},
        "clean_mlm_top1": hb, "preflight_PASS": pf["PASS"],
        "meets_power_target": bool(len(X) >= 150 and len(set(genes)) >= 60),
        "confirmatory": bool(pf["PASS"] and len(X) >= 150),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s5)"},
    }
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a6_circuit_faithfulness", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": result["n_genes"],
                "n_null": len(null_sets), "comp_scan": len(comp_scan), "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"]},
        seeds={"gene_bootstrap": 0, "null_sets": SEED},
        notes=["Node-level circuit faithfulness/necessity/minimality/completeness of the donor "
               "circuit C (C6/C9/C16-C25). Scrub-to-corrupt; two exact arithmetic memorials. "
               "NODE-level (edge scan is T2). U_info/claims NOT redefined.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    print(f"  F(C) {F_C:+.3f} {F_C_ci}   N(C) {N_C:+.3f} {N_C_ci}   "
          f"null F {np.mean(null_F):+.3f} (p95 {null_p95:+.3f}, p={p_faith:.3f})")
    print(f"  verdict: faithful={faithful} necessary={necessary} minimal={minimal} complete={complete}")
    not_lb = [k for k, v in minimality.items() if not v["load_bearing"]]
    print(f"  minimality: {'ALL load-bearing' if not not_lb else 'prunable: ' + ', '.join(not_lb)}")
    print(f"  residual 1-F(C) = {1.0 - F_C:+.3f}; top omitted: "
          + ", ".join(f"{k}:{v['F_gain']:+.3f}" for k, v in top_omitted[:5]))
    if incomplete:
        print(f"  *** INCOMPLETE -- omitted nodes (T2 targets): {incomplete}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
