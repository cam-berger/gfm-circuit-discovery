#!/usr/bin/env python
"""Donor-graph experiment 11 -- A7: position x layer localization of the distributed generation.

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_A7_SOURCE_LOCALIZATION.md.

A6/C27: the 10-node circuit C is necessary but non-faithful (F(C)=0.005); generation is distributed OR
interaction-dependent.  A7 localizes it.  From the A6 complement-noising baseline (scrub every
non-circuit node to CORRUPT; circuit clean), two conditions:

  STATE CLAMP (upper bound): also force the clean pre-L3H1 residual (hidden_states[3]) at the acceptor
    window -- hands the delivery spine the finished acceptor state (should recover; the ceiling).
  WRITERS-OPEN (the question): leave the non-circuit L0-2 writers in layer-block B clean ONLY at
    region R (scrubbed elsewhere) -- can the network LOCALLY construct the payload from R?

Regions {ACC,DON,PPT,REM} x blocks {L0-2,L0,L1,L2}; dR = F(C u R,B) - F(C); interaction
I(R1,R2)=F(CuR1uR2)-F(CuR1)-F(CuR2)+F(C).  Reuses patch_heads + token masks; adds clamp_write_mask
(bool-mask MLP write patch) and clamp_layer_input (residual pre-hook for the state clamp).
--smoke = memorials/plumbing on tiny-n (val sealed).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file  # noqa: E402
from gfmcd.circuits.hf_bert import _context_from_output, _split_context, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _block_write, gboot_ratio,
    load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.provenance import clamp_embed, embed_module  # noqa: E402

C_HEADS = [(1, 12), (2, 15), (3, 1), (4, 0), (4, 2), (5, 11)]
C_MLPS = [2, 3, 4]
REGIONS = ["ACC", "DON", "PPT", "REM"]
BLOCKS = {"L0-2": [0, 1, 2], "L0": [0], "L1": [1], "L2": [2]}
PAIRS = [("ACC", "PPT"), ("ACC", "DON"), ("ACC", "REM"), ("PPT", "DON")]
L3 = 3


def capture(ad, inputs, want_hs=False):
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
        out = ad.model(**inputs, output_hidden_states=want_hs)
    finally:
        for h in handles:
            h.remove()
    return out, heads, mlps, ebox["e"]


@contextmanager
def clamp_write_mask(ad, blocks, source, bmask):
    """Patch each MLP write to `source` where bmask [B,S] is True."""
    handles = []

    def make(key):
        def hook(_m, _i, out):
            src = source[key].to(out.device, out.dtype)
            return torch.where(bmask.to(out.device)[..., None], src, out)
        return hook
    try:
        for lk in blocks:
            handles.append(_block_write(ad, lk[0], lk[1]).register_forward_hook(make(lk)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def clamp_layer_input(ad, layer, source, bmask):
    """Force the residual ENTERING `layer` to `source` where bmask [B,S] is True (state clamp)."""
    mod = ad.encoder_layers[layer]

    def pre(_m, args, kwargs):
        hs = args[0]
        new = torch.where(bmask.to(hs.device)[..., None], source.to(hs.device, hs.dtype), hs)
        return (new,) + tuple(args[1:]), kwargs
    h = mod.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


def region_masks(exs, off, S, dev):
    B = len(exs)
    acc = torch.zeros(B, S, dtype=torch.bool, device=dev)
    don = torch.zeros_like(acc); ppt = torch.zeros_like(acc)
    for i, x in enumerate(exs):
        a = x.next_acceptor_pos + off; d = x.donor_pos + off
        for k in (-3, -2, -1, 0):
            if 0 <= a + k < S: acc[i, a + k] = True
        for k in range(-5, 6):
            if 0 <= d + k < S: don[i, d + k] = True
        for k in range(-30, -4):
            if 0 <= a + k < S: ppt[i, a + k] = True
    return {"ACC": acc, "DON": don, "PPT": ppt, "REM": ~(acc | don | ppt)}


def margin_scrub(ad, task, base, dn, head_specs, mlp_specs, embed=None, layerclamp=None):
    with ExitStack() as st:
        for heads, src, mask in head_specs:
            if heads:
                st.enter_context(patch_heads(ad, src, heads, token_mask=mask))
        for blocks, src, mask in mlp_specs:
            if blocks:
                st.enter_context(clamp_write_mask(ad, blocks, src, mask))
        if embed is not None:
            st.enter_context(clamp_embed(ad, embed[0], embed[1]))
        if layerclamp is not None:
            st.enter_context(clamp_layer_input(ad, layerclamp[0], layerclamp[1], layerclamp[2]))
        lg = ad.model(**base).logits.float()
    return task.margin(lg, dn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a7_source_localization.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    pf_path = out.parent / "preflight_a7_source_localization.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off

    n = min(args.n, 24) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a7-{args.split}",
                         min_effect=None)
    if len(X) < 16:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})

    ALL_HEADS = [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)]
    ALL_MLPS = [(l, "mlp") for l in range(ad.num_layers)]
    nc_heads = [hh for hh in ALL_HEADS if hh not in C_HEADS]
    nc_mlps = [(l, "mlp") for l in range(ad.num_layers) if l not in C_MLPS]

    blocks = BLOCKS if not args.smoke else {"L0-2": [0, 1, 2]}
    pairs = PAIRS if not args.smoke else [("ACC", "PPT")]

    num = {}
    coup_all, genes = [], []
    mem = {"empty_is_corrupt": 0.0, "full_is_clean": 0.0, "maskFalse_is_clean": 0.0,
           "state_identity_is_clean": 0.0}

    def acc(key, vals):
        num.setdefault(key, []).extend(vals)

    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        oc, hcl, mclw, ecl = capture(ad, cl, want_hs=True)
        od, hco, mcow, eco = capture(ad, co)
        Hcl3 = oc.hidden_states[L3]
        m_cl = task.margin(oc.logits.float(), dn)
        m_co = task.margin(od.logits.float(), dn)
        B, S = cl["input_ids"].shape
        allpos = torch.arange(S, device=dev)[None].expand(B, S)
        allT = torch.ones(B, S, dtype=torch.bool, device=dev)
        allF = torch.zeros(B, S, dtype=torch.bool, device=dev)
        Rm = region_masks(ch, off, S, dev)

        # memorials
        me = margin_scrub(ad, task, cl, dn, [(ALL_HEADS, hco, allT)], [(ALL_MLPS, mcow, allT)],
                          embed=(eco, allpos))
        mf = margin_scrub(ad, task, co, dn, [(ALL_HEADS, hcl, allT)], [(ALL_MLPS, mclw, allT)],
                          embed=(ecl, allpos))
        mnf = margin_scrub(ad, task, cl, dn, [(nc_heads, hco, allF)], [(nc_mlps, mcow, allF)])
        msi = margin_scrub(ad, task, cl, dn, [], [], layerclamp=(L3, Hcl3, allT))
        mem["empty_is_corrupt"] = max(mem["empty_is_corrupt"], float((me - m_co).abs().max()))
        mem["full_is_clean"] = max(mem["full_is_clean"], float((mf - m_cl).abs().max()))
        mem["maskFalse_is_clean"] = max(mem["maskFalse_is_clean"], float((mnf - m_cl).abs().max()))
        mem["state_identity_is_clean"] = max(mem["state_identity_is_clean"],
                                             float((msi - m_cl).abs().max()))

        # baseline F(C): scrub all non-circuit everywhere
        base_specs = ([(nc_heads, hco, allT)], [(nc_mlps, mcow, allT)])
        mFC = margin_scrub(ad, task, cl, dn, *base_specs)
        acc("F_C", (mFC - m_co).cpu().tolist())

        # state clamp (ceiling): F(C) + clean pre-L3H1 residual at ACC
        mSt = margin_scrub(ad, task, cl, dn, *base_specs, layerclamp=(L3, Hcl3, Rm["ACC"]))
        acc("state_ACC", (mSt - m_co).cpu().tolist())

        def writers(rmask, bl):
            hb = [hh for hh in nc_heads if hh[0] in bl]
            ha = [hh for hh in nc_heads if hh[0] not in bl]
            mb = [mm for mm in nc_mlps if mm[0] in bl]
            ma = [mm for mm in nc_mlps if mm[0] not in bl]
            notr = ~rmask
            return margin_scrub(ad, task, cl, dn,
                                [(ha, hco, allT), (hb, hco, notr)],
                                [(ma, mcow, allT), (mb, mcow, notr)])

        for Bn, bl in blocks.items():
            for Rn in REGIONS:
                acc((Rn, Bn), (writers(Rm[Rn], bl) - m_co).cpu().tolist())
        for r1, r2 in pairs:
            acc(("pair", r1, r2), (writers(Rm[r1] | Rm[r2], [0, 1, 2]) - m_co).cpu().tolist())

        coup_all.extend((m_cl - m_co).cpu().tolist())
        genes.extend([x.gene_id for x in ch])
        print(f"  batch {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    coupling = float(np.mean(coup_all))
    checks = {
        "empty_circuit_is_corrupt": {"PASS": bool(mem["empty_is_corrupt"] < 1e-4),
                                     "max_abs_err": mem["empty_is_corrupt"]},
        "full_circuit_is_clean": {"PASS": bool(mem["full_is_clean"] < 1e-4),
                                  "max_abs_err": mem["full_is_clean"]},
        "maskFalse_is_clean": {"PASS": bool(mem["maskFalse_is_clean"] < 1e-4),
                               "max_abs_err": mem["maskFalse_is_clean"]},
        "state_clamp_identity": {"PASS": bool(mem["state_identity_is_clean"] < 1e-4),
                                 "max_abs_err": mem["state_identity_is_clean"]},
        "positive_control_partner_coupling": {"PASS": bool(coupling >= 0.20), "coupling": coupling},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
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
        print("SMOKE: memorials/plumbing only; no localization reported (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting (prereg s3).")
        return

    def R(key):
        return ratio_of_means(num[key], coup_all)

    F_C = R("F_C")
    state = R("state_ACC")
    ceiling = state - F_C
    grid = {}
    for Bn in blocks:
        for Rn in REGIONS:
            key = (Rn, Bn)
            dn_ = [a - b for a, b in zip(num[key], num["F_C"])]
            ci = list(gboot_ratio(dn_, coup_all, genes))
            grid[f"{Rn}|{Bn}"] = {"F": R(key), "dR": R(key) - F_C, "dR_ci": ci,
                                  "frac_of_ceiling": (R(key) - F_C) / ceiling if ceiling > 1e-6 else None,
                                  "real": bool(ci[0] > 0)}
    interactions = {}
    for r1, r2 in pairs:
        f12 = R(("pair", r1, r2)); f1 = R((r1, "L0-2")); f2 = R((r2, "L0-2"))
        interactions[f"{r1}x{r2}"] = {"F_joint": f12, "I": f12 - f1 - f2 + F_C,
                                      "dR1": f1 - F_C, "dR2": f2 - F_C}

    l02 = {r: grid[f"{r}|L0-2"] for r in REGIONS}
    best = max(l02.items(), key=lambda kv: kv[1]["dR"])
    acc_hi = l02["ACC"]["frac_of_ceiling"] is not None and l02["ACC"]["frac_of_ceiling"] >= 0.5
    state_hi = ceiling >= 0.20
    any_region = any(v["real"] and (v["frac_of_ceiling"] or 0) >= 0.25 for v in l02.values())
    max_I = max((v["I"] for v in interactions.values()), default=0.0)
    if state_hi and acc_hi:
        verdict = "local_convergence_at_acceptor"
    elif state_hi and not acc_hi and max_I > 0.05:
        verdict = "structured_distributed_interaction"
    elif state_hi and not acc_hi and any_region:
        verdict = "local_when_finished_imports_info"
    elif state_hi and not any_region:
        verdict = "broad_or_wrong_partition"
    else:
        verdict = "inconclusive_low_ceiling"

    result = {
        "n": len(X), "n_genes": len(set(genes)), "strands": strands, "coupling": coupling,
        "F_C": F_C, "state_clamp_ACC": state, "ceiling": ceiling,
        "grid": grid, "interactions": interactions, "verdict": verdict,
        "best_region_L0-2": {"region": best[0], **best[1]},
        "preflight_PASS": pf["PASS"],
        "meets_power_target": bool(len(X) >= 150 and len(set(genes)) >= 60),
        "confirmatory": bool(pf["PASS"] and len(X) >= 150),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s5)"},
    }
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a7_source_localization", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": result["n_genes"],
                "regions": REGIONS, "blocks": list(blocks), "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"]},
        seeds={"gene_bootstrap": 0},
        notes=["Position x layer localization of the distributed generation (A6/C27). State-clamp "
               "(upper bound) vs writers-open (local construction), regions x layer-blocks, dR + "
               "interactions. Reuses A6 scrub + token masks; no claim redefined.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    print(f"  F(C)={F_C:+.3f}  state-clamp(ACC)={state:+.3f}  ceiling={ceiling:+.3f}")
    print("  writers-open dR (L0-2):  " + "  ".join(
        f"{r}={l02[r]['dR']:+.3f}({'*' if l02[r]['real'] else ' '})" for r in REGIONS))
    for Bn in [b for b in blocks if b != "L0-2"]:
        print(f"    {Bn}: " + "  ".join(f"{r}={grid[f'{r}|{Bn}']['dR']:+.3f}" for r in REGIONS))
    print("  interactions: " + "  ".join(f"{k}:I={v['I']:+.3f}" for k, v in interactions.items()))
    print(f"  VERDICT: {verdict}   (best L0-2 region: {best[0]} dR={best[1]['dR']:+.3f})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
