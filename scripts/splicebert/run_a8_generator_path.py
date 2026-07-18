#!/usr/bin/env python
"""Donor-graph experiment 12 -- A8: backward-pruned exact path search through the acceptor generator.

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_A8_GENERATOR_PATH.md.

A7/C28: the payload is built by a serially-interdependent L0->L1->L2 module at the acceptor window.
A8 finds the minimal exact subgraph, scored through L3H1's VALUE path with the ADDRESS held CLEAN:

  For opened S subset of A (acceptor-window L0-2 non-circuit writers), run the S-open scrub, capture
  r3(S)=hidden_states[3] at ACC; with post-LN linearity dV=W_V.dr reconstruct L3H1's donor output using
  the CLEAN attention row A+ and only the S-driven value edit; inject donor-only -> M'(S).  Normalized
  R(S)=(M'(S)-M'(0))/(M'(A)-M'(0)) (M'(0)=scaffold-only baseline).  Backward pruning (leave-group-out
  necessity N_g) + scaffold-aware forward confirmation (P_g), on a discovery/held-out gene split.

Reuses A6/A7 scrub (patch_heads/clamp_write_mask) + A2b/A3 reconstruct-inject (payload.*).  No new
kernel.  --smoke = memorials/plumbing on tiny-n (val sealed).
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

sys.path.insert(0, str(REPO / "scripts" / "splicebert"))
from run_a7_source_localization import capture as a7cap, clamp_write_mask     # noqa: E402

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file       # noqa: E402
from gfmcd.circuits.hf_bert import patch_heads                                 # noqa: E402
from gfmcd.splicing.circuit import (                                          # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot_ratio,
    load_adapter, load_examples, pick_device,
)
from gfmcd.splicing.provenance import clamp_embed, embed_module               # noqa: E402
from gfmcd.splicing.payload import deliver, head_output, inject_margin_donor, value_weight_head  # noqa: E402

L3, H3 = 3, 1
C_HEADS = [(1, 12), (2, 15), (3, 1), (4, 0), (4, 2), (5, 11)]
ACC_OFF = (-3, -2, -1, 0)
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a8_generator_path.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    pf_path = out.parent / "preflight_a8_generator_path.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off
    Dh = ad.hidden_size // ad.num_heads
    Wv = value_weight_head(ad).to(dev)

    n = min(args.n, 24) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a8-{args.split}",
                         min_effect=None)
    if len(X) < 16:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})

    # generator A = acceptor-window L0-2 non-circuit writers; blocks; circuit L0-2 value-check nodes
    A_HEADS = [(l, h) for l in (0, 1, 2) for h in range(ad.num_heads) if (l, h) not in C_HEADS]
    A_MLPS = [(0, "mlp"), (1, "mlp")]
    A_NODES = A_HEADS + A_MLPS
    gL = {l: [(ll, h) for (ll, h) in A_HEADS if ll == l] + ([(l, "mlp")] if l < 2 else [])
          for l in (0, 1, 2)}
    CIRC = {"MLP2": (2, "mlp"), "L1H12": (1, 12), "L2H15": (2, 15)}
    ALL_HEADS = [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)]
    ALL_MLPS = [(l, "mlp") for l in range(ad.num_layers)]

    # conditions: name -> (open_subset, extra_scrub)   extra_scrub: list of (node, "ACC")
    conds = {"empty": ([], None), "full": (A_NODES, None),
             "noL0": ([a for a in A_NODES if a not in gL[0]], None),
             "noL1": ([a for a in A_NODES if a not in gL[1]], None),
             "noL2": ([a for a in A_NODES if a not in gL[2]], None),
             "L0": (gL[0], None), "L1": (gL[1], None), "L2": (gL[2], None),
             "L1L2": (gL[1] + gL[2], None)}
    if not args.smoke:
        for (l, h) in A_HEADS:
            conds[f"no_{l}_{h}"] = ([a for a in A_NODES if a != (l, h)], None)
        for (l, _) in A_MLPS:
            conds[f"no_{l}_mlp"] = ([a for a in A_NODES if a != (l, "mlp")], None)
        for name, node in CIRC.items():
            conds[f"circ_{name}"] = (A_NODES, [(node, "ACC")])

    Mp = {k: [] for k in conds}
    Mp_fullclean = []
    lin_err = 0.0
    mem = {"empty_is_corrupt": 0.0, "full_is_clean": 0.0}
    mcl_all, mco_all, genes = [], [], []

    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        B, S = cl["input_ids"].shape
        dtok = torch.tensor([d + off for d in dn], device=dev)
        accwin = torch.tensor([[x.next_acceptor_pos + off + k for k in ACC_OFF] for x in ch],
                              device=dev)                                  # [B,4]
        bidx = torch.arange(B, device=dev)[:, None].expand(B, 4)
        ACCm = torch.zeros(B, S, dtype=torch.bool, device=dev)
        ACCm[bidx, accwin] = True
        notACC, allT = ~ACCm, torch.ones(B, S, dtype=torch.bool, device=dev)

        def caprun(inp, want_attn):
            vbox = {}
            hh = ad.attention_self_modules[L3].value.register_forward_hook(
                lambda _m, _i, o: vbox.__setitem__("v", o.detach()))
            try:
                o = ad.model(**inp, output_hidden_states=True, output_attentions=want_attn)
            finally:
                hh.remove()
            r3 = o.hidden_states[L3]
            vhead = vbox["v"][:, :, H3 * Dh:(H3 + 1) * Dh]
            ap = o.attentions[L3][torch.arange(B, device=dev), H3, dtok, :] if want_attn else None
            return o, r3, vhead, ap

        o_cl, r3_cl, V_cl, Aplus = caprun(cl, True)
        o_co, r3_co, V_co, _ = caprun(co, False)
        m_cl = task.margin(o_cl.logits.float(), dn); m_co = task.margin(o_co.logits.float(), dn)
        base = torch.einsum("bs,bsd->bd", Aplus, V_co)              # A+_d . V- over all positions
        Aplus_acc = Aplus[bidx, accwin]                            # [B,4]
        r3co_acc = r3_co[bidx, accwin]                             # [B,4,H]

        # value_linearity memorial: (V+ - V-)_acc == W_V (r3cl - r3co)_acc
        dV_true = (V_cl - V_co)[bidx, accwin]
        dV_lin = deliver(Wv, (r3_cl - r3_co)[bidx, accwin])
        lin_err = max(lin_err, float((dV_true - dV_lin).abs().max()))

        def scrub_r3(open_nodes, extra):
            closed = [a for a in A_NODES if a not in open_nodes]
            ho = [n for n in open_nodes if isinstance(n[1], int)]
            mo = [n for n in open_nodes if n[1] == "mlp"]
            hc = [n for n in closed if isinstance(n[1], int)]
            mc = [n for n in closed if n[1] == "mlp"]
            with ExitStack() as st:
                if ho: st.enter_context(patch_heads(ad, hco, ho, token_mask=notACC))
                if hc: st.enter_context(patch_heads(ad, hco, hc, token_mask=allT))
                if mo: st.enter_context(clamp_write_mask(ad, mo, mco_w, notACC))
                if mc: st.enter_context(clamp_write_mask(ad, mc, mco_w, allT))
                for node, _msk in (extra or []):
                    if isinstance(node[1], int):
                        st.enter_context(patch_heads(ad, hco, [node], token_mask=ACCm))
                    else:
                        st.enter_context(clamp_write_mask(ad, [node], mco_w, ACCm))
                return ad.model(**cl, output_hidden_states=True).hidden_states[L3]

        def mprime(r3S):
            dr = r3S[bidx, accwin] - r3co_acc
            od = head_output(base, Aplus_acc, deliver(Wv, dr))
            return inject_margin_donor(ad, task, co, dn, dtok, od)          # numpy [B]

        # need corrupt + clean head/mlp caches for the scrub source
        _, hco, mco_w, eco = a7cap(ad, co)
        _, hcl, mcl_w, ecl = a7cap(ad, cl)

        # scrub-cover memorials (full node cover, plain margin)
        with ExitStack() as st:
            st.enter_context(patch_heads(ad, hco, ALL_HEADS, token_mask=allT))
            st.enter_context(clamp_write_mask(ad, ALL_MLPS, mco_w, allT))
            st.enter_context(clamp_embed(ad, eco, torch.arange(S, device=dev)[None].expand(B, S)))
            me = task.margin(ad.model(**cl).logits.float(), dn)
        with ExitStack() as st:
            st.enter_context(patch_heads(ad, hcl, ALL_HEADS, token_mask=allT))
            st.enter_context(clamp_write_mask(ad, ALL_MLPS, mcl_w, allT))
            st.enter_context(clamp_embed(ad, ecl, torch.arange(S, device=dev)[None].expand(B, S)))
            mf = task.margin(ad.model(**co).logits.float(), dn)
        mem["empty_is_corrupt"] = max(mem["empty_is_corrupt"], float((me - m_co).abs().max()))
        mem["full_is_clean"] = max(mem["full_is_clean"], float((mf - m_cl).abs().max()))

        # full-clean value restore memorial (dr from true clean r3)
        od_fc = head_output(base, Aplus_acc, deliver(Wv, (r3_cl - r3_co)[bidx, accwin]))
        Mp_fullclean.extend(inject_margin_donor(ad, task, co, dn, dtok, od_fc).tolist())

        for name, (open_nodes, extra) in conds.items():
            Mp[name].extend(mprime(scrub_r3(open_nodes, extra)).tolist())

        mcl_all.extend(m_cl.cpu().tolist()); mco_all.extend(m_co.cpu().tolist())
        genes.extend([x.gene_id for x in ch])
        print(f"  batch {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    coup = [a - b for a, b in zip(mcl_all, mco_all)]
    coupling = float(np.mean(coup))
    ug = sorted(set(genes)); Dg = set(ug[::2])
    inD = np.array([g in Dg for g in genes]); inH = ~inD
    mco_mean = float(np.mean(mco_all))
    M0 = float(np.mean(Mp["empty"])) - mco_mean          # scaffold-only recovery vs corrupt
    MA = float(np.mean(Mp["full"])) - mco_mean           # full generator recovery vs corrupt
    fullclean = float(np.mean(Mp_fullclean)) - mco_mean  # C24 clean-acceptor restore vs corrupt
    checks = {
        "value_linearity": {"PASS": bool(lin_err < 1e-4), "max_abs_err": lin_err},
        "empty_circuit_is_corrupt": {"PASS": bool(mem["empty_is_corrupt"] < 1e-4),
                                     "max_abs_err": mem["empty_is_corrupt"]},
        "full_circuit_is_clean": {"PASS": bool(mem["full_is_clean"] < 1e-4),
                                  "max_abs_err": mem["full_is_clean"]},
        "generator_ceiling": {"PASS": bool(MA - M0 >= 0.10), "M_empty_rec": M0, "M_full_rec": MA,
                              "ceiling": MA - M0},
        "full_clean_value_restore": {"PASS": bool(0.28 <= fullclean <= 0.45),
                                     "recovery_vs_corrupt": fullclean},
        "positive_control_partner_coupling": {"PASS": bool(coupling >= 0.20), "coupling": coupling},
        "both_genes_split": {"PASS": bool(inD.sum() >= (5 if args.smoke else 40)
                                          and inH.sum() >= (5 if args.smoke else 40)),
                             "nD": int(inD.sum()), "nH": int(inH.sum())},
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
        extra = c.get("max_abs_err", c.get("value", c.get("ceiling")))
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({extra})" if extra is not None else ""))

    if args.smoke:
        print("SMOKE: memorials/plumbing only; no path search reported (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting (prereg s3).")
        return

    def R(name, mask):
        num = np.array(Mp[name])[mask]; e = np.array(Mp["empty"])[mask]; f = np.array(Mp["full"])[mask]
        return float((num.mean() - e.mean()) / (f.mean() - e.mean()))

    def Rci(name, mask):
        g = [gg for gg, m in zip(genes, mask) if m]
        num = (np.array(Mp[name]) - np.array(Mp["empty"]))[mask].tolist()
        den = (np.array(Mp["full"]) - np.array(Mp["empty"]))[mask].tolist()
        return list(gboot_ratio(num, den, g))

    def necessity(drop_name, mask):                              # N_g = R(full) - R(drop)
        g = [gg for gg, m in zip(genes, mask) if m]
        num = (np.array(Mp["full"]) - np.array(Mp[drop_name]))[mask].tolist()
        den = (np.array(Mp["full"]) - np.array(Mp["empty"]))[mask].tolist()
        return float(np.mean(num) / np.mean(den)), list(gboot_ratio(num, den, g))

    blocks = {}
    for l, drop, single in [(0, "noL0", "L0"), (1, "noL1", "L1"), (2, "noL2", "L2")]:
        Ng, Ng_ci = necessity(drop, inD)
        _, Ng_ci_H = necessity(drop, inH)
        P = (R(single, inD) if l == 2 else R("L1L2", inD) - R("L2", inD) if l == 1
             else R("full", inD) - R("noL0", inD))
        blocks[f"L{l}"] = {"N_g": Ng, "N_g_ci_D": Ng_ci, "N_g_ci_H": Ng_ci_H,
                           "R_single": R(single, inD), "P_g": P,
                           "necessary_D": bool(Ng_ci[0] > 0), "necessary_H": bool(Ng_ci_H[0] > 0)}
    heads = {}
    for (l, h) in A_HEADS + [(0, "mlp"), (1, "mlp")]:
        key = f"no_{l}_{h if isinstance(h, int) else 'mlp'}"
        Nh, Nh_ci = necessity(key, inD)
        heads[f"{l}.{h}"] = {"N_h": Nh, "ci_D": Nh_ci, "load_bearing_D": bool(Nh_ci[0] > 0)}
    circ = {name: {"N": necessity(f"circ_{name}", inD)[0], "ci": necessity(f"circ_{name}", inD)[1]}
            for name in CIRC}

    # frozen G*: blocks necessary on D, heads load-bearing on D; confirm on H
    keep_heads = [k for k, v in heads.items() if v["load_bearing_D"]]
    RH_full = R("full", inH)
    result = {
        "n": len(X), "n_genes": len(ug), "nD": int(inD.sum()), "nH": int(inH.sum()),
        "coupling": coupling, "M_empty_scaffold": M0, "M_full_generator": MA,
        "ceiling_value_path": MA - M0, "full_clean_C24": fullclean,
        "R_full_D": R("full", inD), "R_full_H": RH_full, "R_full_H_ci": Rci("full", inH),
        "blocks": blocks, "heads": heads, "circuit_value_check": circ,
        "load_bearing_heads_D": keep_heads, "n_load_bearing": len(keep_heads),
        "serial_profile": {f"L{l}": {"P_g": blocks[f'L{l}']['P_g'], "N_g": blocks[f'L{l}']['N_g']}
                           for l in (0, 1, 2)},
        "preflight_PASS": pf["PASS"],
        "meets_power_target": bool(len(X) >= 150 and inD.sum() >= 40 and inH.sum() >= 40),
        "confirmatory": bool(pf["PASS"] and len(X) >= 150),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s5)"},
    }
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a8_generator_path", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": len(ug), "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"]},
        seeds={"gene_split": SEED, "gene_bootstrap": 0},
        notes=["Backward-pruned scaffold-aware exact path search through the acceptor generator "
               "(A7/C28). Value-path metric (address held clean, reconstruct-inject). No new kernel.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    print(f"  M'(0)={M0:+.3f} scaffold  M'(A)={MA:+.3f} generator  ceiling={MA - M0:+.3f}  "
          f"(full-clean/C24={fullclean:+.3f})")
    print(f"  R_full: D={R('full', inD):+.3f}  H={RH_full:+.3f}")
    for l in (0, 1, 2):
        b = blocks[f"L{l}"]
        print(f"  L{l}: N_g={b['N_g']:+.3f} {b['N_g_ci_D']} (D)  H-necessary={b['necessary_H']}  "
              f"P_g={b['P_g']:+.3f}  single R={b['R_single']:+.3f}")
    print(f"  circuit value-check: " + "  ".join(f"{k}:N={v['N']:+.3f}" for k, v in circ.items()))
    print(f"  load-bearing heads (D): {len(keep_heads)} -> {keep_heads[:12]}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
