#!/usr/bin/env python
"""Donor-graph experiment 5 -- A2: address-payload factorization of L3H1.

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_A2_ADDRESS_PAYLOAD.md.

A1a/R20: L3H1's attention TARGETING is position-embedding geometry, not identity -- yet the
circuit couples partner-specifically (C19).  Reconciliation: QK attention is a positional ADDRESS
(where L3H1 reads); the OV/value stream carries the partner PAYLOAD (what it retrieves).  Intervene
independently on the donor attention row A and the values V of L3H1, reconstruct its donor output
Sum_j A_dj V_j, inject it (patch_heads, donor only), run downstream, read the native margin.

Preflight/memorials FIRST; the run stops on failure.  --smoke = memorials only (val sealed).
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

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file  # noqa: E402
from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask,
    capture_proj, gboot_ratio, load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402

L3, H3 = READER
SPLIT_CHROMS = {"val": ("chr11", "chr16", "chr19"), "test": ("chr8", "chr9")}
SEED = 0


def corrupt_at(task, exs, pos_list):
    seqs = [substitute(x.sequence, p - 2, CORRUPT_ALT) for x, p in zip(exs, pos_list)]
    return task.encode(seqs, [x.donor_pos for x in exs])


def donor_row_attention(ad, enc, dtok):
    """L3H1 post-softmax donor attention row [B,S] (eager)."""
    o = ad.model(**enc, output_attentions=True)
    A = o.attentions[L3][:, H3]                                   # [B,S,S]
    b = torch.arange(A.shape[0], device=A.device)
    return A[b, dtok]                                             # [B,S]


def value_head(ad, enc, Dh):
    """L3H1 value vectors [B,S,Dh]."""
    v = capture_proj(ad, L3, ["value"], lambda: ad.model(**enc))["value"]
    return v[:, :, H3 * Dh:(H3 + 1) * Dh]


def reconstruct(A_row, V):
    """Sum_j A_dj V_j -> [B,Dh]  (L3H1 donor head output for a chosen address x payload)."""
    return torch.einsum("bs,bsd->bd", A_row, V)


def inject_margin(ad, task, corr, dn, dtok, head_co, recon):
    """Inject ``recon`` as L3H1's donor head output on the corrupt run; return the margin."""
    src = head_co.clone()                                        # [B,H,S,Dh]
    b = torch.arange(src.shape[0], device=src.device)
    src[b, H3, dtok] = recon.to(src.dtype)
    mask = torch.zeros(src.shape[0], src.shape[2], dtype=torch.bool, device=src.device)
    mask[b, dtok] = True
    with patch_heads(ad, {L3: src}, [(L3, H3)], token_mask=mask):
        m = task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
    return m


def rand_projector(Dh, k, device):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    Q, _ = torch.linalg.qr(torch.randn(Dh, k, generator=g))
    return (Q @ Q.T).to(device)                                  # [Dh,Dh] rank-k projector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8, help="random value-subspace control rank")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a2_address_payload.json")
    args = ap.parse_args()

    out = REPO / args.out
    pf_path = out.parent / "preflight_a2_address_payload.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off
    Dh = ad.hidden_size // ad.num_heads
    P_rand = rand_projector(Dh, args.rank, ad.device)

    n = min(args.n, 24) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a2-{args.split}",
                         min_effect=None)
    if len(X) < 8:
        raise RuntimeError(f"only {len(X)} eligible examples; raise pool/split.")
    strands = sorted({x.strand for x in X})
    genes = [x.gene_id for x in X]
    print(f"{len(X)} eligible examples; strands={strands}; device={dev}"
          f"{'  [SMOKE]' if args.smoke else ''}")

    CELLS = ["baseline", "address_only", "payload_only", "joint", "payload_random"]
    num = {c: [] for c in CELLS}
    dnm = []
    recon_err = []                                               # memorial: recon vs captured head
    base_err = []                                                # memorial: baseline == corrupt
    prob_inv = None                                              # G-Attn: eager rows sum to 1

    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            exs = X[i:i + args.batch]
            dn = [x.donor_pos for x in exs]
            dtok = torch.tensor([d + off for d in dn], device=ad.device)
            clean = task.encode([x.sequence for x in exs], dn)
            corr = corrupt_at(task, exs, [x.next_acceptor_pos for x in exs])

            m_cl = task.margin(ad.model(**clean).logits.float(), dn).cpu().numpy()
            m_co = task.margin(ad.model(**corr).logits.float(), dn).cpu().numpy()
            A_cl = donor_row_attention(ad, clean, dtok)          # [B,S]
            A_co = donor_row_attention(ad, corr, dtok)
            V_cl = value_head(ad, clean, Dh)                     # [B,S,Dh]
            V_co = value_head(ad, corr, Dh)
            head_cl = capture_head_outputs(ad, lambda: ad.model(**clean))[L3]   # [B,H,S,Dh]
            head_co = capture_head_outputs(ad, lambda: ad.model(**corr))[L3]

            if prob_inv is None:                                # G-Attn on the first batch's L3H1 rows
                rserr = float((A_cl.sum(-1) - 1).abs().max())
                prob_inv = {"PASS": bool(rserr < 1e-3 and float(A_cl.max()) > 0),
                            "max_row_sum_err": rserr, "attn_max": float(A_cl.max())}
            V_rand = V_co + torch.einsum("bsd,ed->bse", V_cl - V_co, P_rand)
            b = torch.arange(len(exs), device=ad.device)
            # memorial: reconstruction reproduces the model's own head output
            rec_cl = reconstruct(A_cl, V_cl)
            recon_err += (rec_cl - head_cl[b, H3, dtok]).abs().amax(-1).cpu().tolist()

            combos = {
                "baseline": (A_co, V_co), "address_only": (A_cl, V_co),
                "payload_only": (A_co, V_cl), "joint": (A_cl, V_cl),
                "payload_random": (A_cl, V_rand),
            }
            for name, (Arow, V) in combos.items():
                m = inject_margin(ad, task, corr, dn, dtok, head_co, reconstruct(Arow, V))
                num[name] += (m - m_co).tolist()
                if name == "baseline":
                    base_err += np.abs(m - m_co).tolist()
            dnm += (m_cl - m_co).tolist()
            print(f"  {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    coupling = float(np.mean(dnm))
    pf = {
        "frozen": {"recon_tol": 1e-4, "inject_tol": 1e-3, "coupling_min": 0.20,
                   "joint_recovers_min": 0.30},
        "commit": git_commit(REPO), "n": len(X),
        "checks": {
            "reconstruction_matches": {"PASS": bool(max(recon_err) < 1e-4),
                                       "max_abs_err": float(max(recon_err))},
            "inject_identity": {"PASS": bool(max(base_err) < 1e-3),
                                "max_abs_err": float(max(base_err))},
            "probability_invariants": prob_inv,
            "positive_control_partner_coupling": {"PASS": bool(coupling >= 0.20),
                                                  "coupling": coupling},
            "joint_recovers": {"PASS": bool(ratio_of_means(num["joint"], dnm) >= 0.30),
                               "recovery": ratio_of_means(num["joint"], dnm)},
            "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
        },
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    pf["checks"]["checkpoint_recorded"] = {"PASS": True,
                                           "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf["PASS"] = bool(all(c["PASS"] for c in pf["checks"].values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        print("SMOKE: preflight/memorials only; no recovery magnitudes (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting recoveries (prereg §3).")
        return

    def rec(name):
        return {"recovery": ratio_of_means(num[name], dnm),
                "ci": list(gboot_ratio(num[name], dnm, genes))}
    R = {c: rec(c) for c in CELLS}
    # same-row value effect and interaction (in recovery/margin space, per example)
    a = {c: np.array(num[c]) for c in CELLS}
    d = np.array(dnm)
    same_row_cleanA = ratio_of_means((a["joint"] - a["address_only"]).tolist(), dnm)
    same_row_corrA = ratio_of_means((a["payload_only"] - a["baseline"]).tolist(), dnm)
    inter = ratio_of_means((a["joint"] - a["address_only"] - a["payload_only"]
                            + a["baseline"]).tolist(), dnm)
    result = {
        "n": len(X), "n_genes": len(set(genes)), "strands": strands,
        "coupling": coupling, "recoveries": R,
        "same_row_value_effect_cleanA": same_row_cleanA,
        "same_row_value_effect_corruptA": same_row_corrA,
        "interaction_AxV": inter,
        "preflight_PASS": pf["PASS"],
        "meets_power_target": bool(len(X) >= 250),
        "confirmatory": bool(pf["PASS"] and len(X) >= 250),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg §5)"},
    }
    out.write_text(json.dumps(result, indent=1))

    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a2_address_payload", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": result["n_genes"],
                "reader": list(READER), "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"], "rank": args.rank},
        seeds={"bootstrap": 0, "rand_projector": SEED},
        notes=["Address-payload factorial of L3H1 (A0/A1a reconciliation): reconstruct-inject "
               "Sum_j A_dj V_j with A/V from clean/corrupt, read native margin recovery.",
               "Behaviour-level; tests the address-payload split of L3H1's coupling contribution, "
               "NOT attention identity (R20 stands).",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    for c in CELLS:
        print(f"  {c:16s} recovery {R[c]['recovery']:+.3f}  CI[{R[c]['ci'][0]:+.2f},{R[c]['ci'][1]:+.2f}]")
    print(f"  same-row value effect (clean address) {same_row_cleanA:+.3f}  "
          f"(corrupt address) {same_row_corrA:+.3f}  interaction {inter:+.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
