#!/usr/bin/env python
"""Donor-graph experiment 4b -- A1a: token-vs-position decomposition of the embedding-resident
targeting difference.

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_A1_TOKEN_VS_POSITION.md.

A0 (C21) localized L3H1's donor-query targeting DIFFERENCE S_A to the token+positional embeddings.
Is that difference carried by positional GEOMETRY or local sequence CONTENT?  A 2x2 clamps the
word and position embedding components separately at the 8 key-window positions to a SHARED
reference (a+ and decoy made identical on the neutralised dimension), reads L3's S_A, and splits
position_effect vs content_effect.  S_A is a mechanistic intermediate; no identity claim.

Preflight is written FIRST and the run stops if it fails.  ``--smoke`` emits arithmetic/preflight
invariants only (val stays sealed).
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

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file  # noqa: E402
from gfmcd.splicing.address import build_pairs, match_audit, windows  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask,
    capture_proj, gboot, load_adapter, load_examples, pick_device,
)
from gfmcd.splicing.provenance import (  # noqa: E402
    _bcast_gather, capture_embed_components, clamp_embed_component, embed_module, sa_difference,
    shared_window_reference,
)

L3, H3 = READER
OFFS = (-3, -2, -1, 0)
SPLIT_CHROMS = {"val": ("chr11", "chr16", "chr19"), "test": ("chr8", "chr9")}
A0_RESULT = REPO / "results/splicebert_coupling/a0_targeting_provenance.json"


def key_windows(ch, off, other_attr, device):
    wplus = windows([p.a_plus for p in ch], off, device, offsets=OFFS)
    wother = windows([getattr(p, other_attr) for p in ch], off, device, offsets=OFFS)
    donor = torch.tensor([p.example.donor_pos + off for p in ch], device=device)
    return donor, wplus, wother


def build_shared_ref(ad, task, pairs, off, other_attr, batch):
    """Shared grand-mean word/position embedding per offset, over ALL pairs and BOTH roles."""
    H = ad.hidden_size
    K = len(OFFS)
    wsum = torch.zeros(K, H, device=ad.device)
    psum = torch.zeros(K, H, device=ad.device)
    count = 0
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            ch = pairs[i:i + batch]
            exs = [p.example for p in ch]
            enc = task.encode([x.sequence for x in exs], [x.donor_pos for x in exs])
            store = capture_embed_components(ad, lambda: ad.model(**enc))
            _, wplus, wother = key_windows(ch, off, other_attr, ad.device)
            for w in (wplus, wother):
                wsum += _bcast_gather(store["word"], w).sum(0)
                psum += _bcast_gather(store["position"], w).sum(0)
                count += w.shape[0]
    return {"word": wsum / count, "position": psum / count}      # each [K,H]


def sa_cell(ad, enc, donor, wplus, wother, Dh, clamps):
    """S_A per example under a set of (which, source) embedding-component clamps."""
    positions = torch.cat([wplus, wother], dim=1)                # [B, 8]
    with ExitStack() as st:
        for which, src in clamps:
            st.enter_context(clamp_embed_component(ad, which, src, positions))
        proj = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc))
    return sa_difference(proj["query"], proj["key"], donor, wplus, wother, Dh).cpu().numpy()


def sweep_arm(ad, task, pairs, off, other_attr, ref, batch):
    Dh = ad.hidden_size // ad.num_heads
    cells = {c: [] for c in ("full", "word_neutral", "pos_neutral", "both_neutral")}
    genes = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            ch = pairs[i:i + batch]
            exs = [p.example for p in ch]
            enc = task.encode([x.sequence for x in exs], [x.donor_pos for x in exs])
            S = enc["input_ids"].shape[1]
            donor, wplus, wother = key_windows(ch, off, other_attr, ad.device)
            sw = shared_window_reference(ref["word"], wplus, wother, S, ad.device)
            sp = shared_window_reference(ref["position"], wplus, wother, S, ad.device)
            cells["full"] += sa_cell(ad, enc, donor, wplus, wother, Dh, []).tolist()
            cells["word_neutral"] += sa_cell(ad, enc, donor, wplus, wother, Dh,
                                             [("word", sw)]).tolist()
            cells["pos_neutral"] += sa_cell(ad, enc, donor, wplus, wother, Dh,
                                            [("position", sp)]).tolist()
            cells["both_neutral"] += sa_cell(ad, enc, donor, wplus, wother, Dh,
                                             [("word", sw), ("position", sp)]).tolist()
            genes += [p.gene_id for p in ch]
            print(f"  {min(i + batch, len(pairs))}/{len(pairs)}", end="\r", flush=True)
    print()
    return cells, genes


def summarise(cells, genes):
    def ci(v):
        return list(gboot(v, genes))
    full = np.array(cells["full"])
    pos_eff = full - np.array(cells["pos_neutral"])
    con_eff = full - np.array(cells["word_neutral"])
    inter = (full - np.array(cells["word_neutral"]) - np.array(cells["pos_neutral"])
             + np.array(cells["both_neutral"]))
    out = {"n": len(full), "n_genes": len(set(genes)),
           "cells": {c: {"mean": float(np.mean(cells[c])), "ci": ci(cells[c])} for c in cells},
           "position_effect": {"mean": float(np.mean(pos_eff)), "ci": ci(pos_eff.tolist())},
           "content_effect": {"mean": float(np.mean(con_eff)), "ci": ci(con_eff.tolist())},
           "interaction": {"mean": float(np.mean(inter)), "ci": ci(inter.tolist())},
           "floor_over_full": float(np.mean(cells["both_neutral"]) / np.mean(full))
           if np.mean(full) else float("nan")}
    return out


def preflight(ad, task, pairs, off, args, ref_secondary, n_real) -> dict:
    frozen = {"qk_recon_tol": 1e-3, "attn_row_sum_tol": 1e-3, "neutralise_tol": 1e-4,
              "min_real": args.min_real}
    Dh = ad.hidden_size // ad.num_heads
    checks = {}
    sub = pairs[: min(len(pairs), 32)]
    exs = [p.example for p in sub]
    dn = [x.donor_pos for x in exs]
    with torch.no_grad():
        enc = task.encode([x.sequence for x in exs], dn)
        S = enc["input_ids"].shape[1]
        o = ad.model(**enc, output_attentions=True)
        attn = o.attentions[L3][:, H3]
        row = attn.sum(-1)
        checks["probability_invariants"] = {
            "PASS": bool(float((row - 1).abs().max()) < frozen["attn_row_sum_tol"]
                         and float(attn.abs().max()) > 0),
            "max_row_sum_err": float((row - 1).abs().max())}
        proj = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc))
        q = proj["query"][:, :, H3 * Dh:(H3 + 1) * Dh]
        k = proj["key"][:, :, H3 * Dh:(H3 + 1) * Dh]
        add = (1.0 - enc["attention_mask"][:, None, :].float()) * -1e9
        recon = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (Dh ** 0.5) + add, dim=-1)
        qk_err = float((recon - attn).abs().max())
        checks["qk_reconstruction"] = {"PASS": bool(qk_err < frozen["qk_recon_tol"]),
                                       "max_abs_err": qk_err}
        # KEY MEMORIAL: with BOTH components clamped to the shared reference, the a+ and decoy
        # key-window EMBEDDING outputs must be identical at matched offsets (shared ref really
        # removes the discrimination).  Falsifiable: a wrong ref / broadcast bug breaks it.
        donor, wplus, wother = key_windows(sub, off, "a_minus", ad.device)
        sw = shared_window_reference(ref_secondary["word"], wplus, wother, S, ad.device)
        sp = shared_window_reference(ref_secondary["position"], wplus, wother, S, ad.device)
        pos8 = torch.cat([wplus, wother], dim=1)
        store = {}
        with clamp_embed_component(ad, "word", sw, pos8), \
             clamp_embed_component(ad, "position", sp, pos8):
            h = embed_module(ad).register_forward_hook(
                lambda _m, _i, out: store.__setitem__("e", out.detach()))
            ad.model(**enc)
            h.remove()
        e = store["e"]
        b = torch.arange(len(sub), device=ad.device)[:, None].expand(len(sub), len(OFFS))
        ep, eo = e[b, wplus], e[b, wother]
        neu = float((ep - eo).abs().max())
        checks["shared_ref_neutralises"] = {"PASS": bool(neu < frozen["neutralise_tol"]),
                                            "max_abs_err": neu}
        # reported cross-check: S_A(full) on the subset vs A0's S_J full for the secondary arm
        sa_full = float(np.mean(sa_cell(ad, enc, donor, wplus, wother, Dh, [])))
        a0_ref = None
        if A0_RESULT.exists():
            a0 = json.loads(A0_RESULT.read_text())
            a0_ref = a0["secondary_generic_contrast"]["curves"]["S_J"][-1]["mean"]
        checks["full_matches_A0"] = {"PASS": True, "S_A_full_subset": sa_full,
                                     "A0_S_J_full_secondary": a0_ref,
                                     "note": "reported; subset vs full-n A0, expect same sign/scale"}
    checks["resolution_pair_match"] = {"PASS": bool(match_audit(pairs)["PASS"]), **match_audit(pairs)}
    checks["real_acceptor_arm_available"] = {"PASS": bool(n_real >= frozen["min_real"]),
                                             "n_real": n_real, "min_real": frozen["min_real"]}
    checks["both_strands_present"] = {"PASS": len({p.strand for p in pairs}) == 2,
                                      "strands": sorted({p.strand for p in pairs})}
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    PASS = all(c["PASS"] for c in checks.values())
    return {"frozen": frozen, "commit": git_commit(REPO), "n_pairs": len(pairs), "n_real": n_real,
            "checks": checks, "PASS": bool(PASS)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tol", type=int, default=40)
    ap.add_argument("--real-tol", type=int, default=60)
    ap.add_argument("--min-real", type=int, default=30)
    ap.add_argument("--real-cap", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-coding", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a1_token_position.json")
    args = ap.parse_args()

    out = REPO / args.out
    pf_path = out.parent / "preflight_a1_token_position.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off

    Xpool, _ = load_examples(args.gtf, args.fasta, args.split, args.pool, task=task,
                             tag=f"a1tp-{args.split}", min_effect=None)
    cds = None
    if not args.skip_coding:
        try:
            from gfmcd.splicing.phase import load_cds_frames
            cds = load_cds_frames(args.gtf, allowed_chromosomes=SPLIT_CHROMS.get(args.split))
        except Exception as e:
            print(f"  (coding covariate unavailable: {e})")
    avail = build_pairs(Xpool, tol=args.tol, real_tol=args.real_tol, cds_lookup=cds)
    n = min(args.n, 24) if args.smoke else args.n
    pairs = avail[:n]
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(avail)} pairs from {len(Xpool)} eligible; raise --pool/--tol.")
    real_cap = min(args.real_cap, 12) if args.smoke else args.real_cap
    real_pairs = [p for p in avail if p.a_real is not None][:real_cap]
    print(f"{len(pairs)} generic, {len(real_pairs)} real-acceptor pairs (of {len(avail)}); "
          f"strands={sorted({p.strand for p in pairs})}; device={dev}"
          f"{'  [SMOKE]' if args.smoke else ''}")

    ref_secondary = build_shared_ref(ad, task, pairs, off, "a_minus", args.batch)
    pf = preflight(ad, task, pairs, off, args, ref_secondary, n_real=len(real_pairs))
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        Dh = ad.hidden_size // ad.num_heads
        sub = pairs[:min(6, len(pairs))]
        exs = [p.example for p in sub]
        enc = task.encode([x.sequence for x in exs], [x.donor_pos for x in exs])
        S = enc["input_ids"].shape[1]
        donor, wplus, wother = key_windows(sub, off, "a_minus", ad.device)
        sw = shared_window_reference(ref_secondary["word"], wplus, wother, S, ad.device)
        sp = shared_window_reference(ref_secondary["position"], wplus, wother, S, ad.device)
        vals = [sa_cell(ad, enc, donor, wplus, wother, Dh, cl) for cl in
                ([], [("word", sw)], [("position", sp)], [("word", sw), ("position", sp)])]
        finite = all(np.isfinite(v).all() for v in vals)
        print(f"    {'ok ' if finite else 'XX '}selfcheck  four_cells_finite={finite}")
        print("SMOKE: preflight/arithmetic only; no effect directions (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not computing cells (prereg §3).")
        return

    result = {"n_pairs": len(pairs), "n_real": len(real_pairs),
              "strands": sorted({p.strand for p in pairs}),
              "match_audit": match_audit(pairs), "preflight_PASS": pf["PASS"]}
    if len(real_pairs) >= 8:
        print(f"primary arm (a+ vs real acceptor), n={len(real_pairs)}")
        ref_p = build_shared_ref(ad, task, real_pairs, off, "a_real", args.batch)
        cells, gp = sweep_arm(ad, task, real_pairs, off, "a_real", ref_p, args.batch)
        result["primary_real_acceptor_contrast"] = summarise(cells, gp)
        result["primary_real_acceptor_contrast"]["real_distance_confound"] = \
            match_audit(real_pairs)["real_acceptor_distance"]
    else:
        result["primary_real_acceptor_contrast"] = {"n": len(real_pairs), "note": "< 8, skipped"}
    print(f"secondary arm (a+ vs generic AG), n={len(pairs)}")
    cells2, g2 = sweep_arm(ad, task, pairs, off, "a_minus", ref_secondary, args.batch)
    result["secondary_generic_contrast"] = summarise(cells2, g2)

    result["n_genes"] = len(set(p.gene_id for p in pairs))
    result["meets_power_target"] = bool(len(pairs) >= 250 and len(real_pairs) >= args.min_real)
    result["confirmatory"] = bool(pf["PASS"] and result["meets_power_target"])
    result["REFLECTION"] = {"status": "PENDING -- fill after inspection (prereg §6)"}
    out.write_text(json.dumps(result, indent=1))

    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a1_token_position", sys.argv, status=status,
        config={"split": args.split, "n_pairs": len(pairs), "n_real": len(real_pairs),
                "n_genes": result["n_genes"], "reader": list(READER), "addresser": list(ADDRESSER),
                "preflight_PASS": pf["PASS"], "meets_power_target": result["meets_power_target"]},
        seeds={"bootstrap": 0},
        notes=["Token-vs-position 2x2 of the embedding-resident targeting difference (A0/C21): "
               "shared-reference neutralisation of word vs position embedding at the key windows.",
               "S_A is a mechanistic intermediate; primary distance + secondary site-strength "
               "confounds reported; NO identity claim.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    for arm in ("primary_real_acceptor_contrast", "secondary_generic_contrast"):
        r = result[arm]
        if "cells" not in r:
            print(f"{arm}: {r.get('note')}"); continue
        print(f"{arm}: full={r['cells']['full']['mean']:+.2f}  "
              f"position_effect={r['position_effect']['mean']:+.2f}"
              f"{r['position_effect']['ci']}  content_effect={r['content_effect']['mean']:+.2f}"
              f"{r['content_effect']['ci']}  floor/full={r['floor_over_full']:.2f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
