"""AUDIT part 2: (a) how big is the diagonal weight f(0) for the 42 lattice heads at MASKED
positions, and (b) does the VALUE stream carry a mod-3 component (i.e. is C(t), the value
autocovariance, periodic at lag 3)?  Both are premises of the self-term attack.
Also: position-embedding-only mod-3 check (values with content held constant)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, load_examples,
    pick_device,
)

MASK_STRIDE = 7


def main():
    dev = pick_device(None)
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=2)
    OFF = task.off
    inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
    LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
    RPH = {(r["layer"], r["head"]): r["dominant_rel_phase"][0]
           for r in inv["heads"] if r["lattice"]}
    X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 6, task=task, tag="phasefn",
                         min_effect=None)
    X = [x for x in X if x.exon_length >= 60 and x.next_acceptor_pos - x.donor_pos >= 60]

    Dh = ad.hidden_size // ad.num_heads
    vals = {}     # (l) -> value projection [B,S,H*Dh]
    hooks = []
    for l in range(ad.num_layers):
        m = ad.attention_self_modules[l].value
        hooks.append(m.register_forward_hook(
            lambda _m, _i, o, l=l: vals.__setitem__(l, o.detach().float().cpu())))

    diag = {hk: [] for hk in LAT}
    C = {hk: np.zeros(8) for hk in LAT}     # value autocovariance by lag, exon
    Ci = {hk: np.zeros(8) for hk in LAT}    # intron
    nC = {hk: np.zeros(8) for hk in LAT}
    nCi = {hk: np.zeros(8) for hk in LAT}

    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            for o in range(MASK_STRIDE):
                enc = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                                   padding=True).to(ad.device)
                pos = []
                for j, x in enumerate(ch):
                    ps = [p for p in range(len(x.sequence)) if p % MASK_STRIDE == o]
                    for p in ps:
                        enc["input_ids"][j, p + OFF] = task.mask_id
                    pos.append(ps)
                out = ad.model(**enc, output_attentions=True)
                A = out.attentions
                for (l, h) in LAT:
                    Al = A[l][:, h].float().cpu().numpy()      # [B,S,S]
                    for j, x in enumerate(ch):
                        for p in pos[j]:
                            q = p + OFF
                            diag[(l, h)].append(Al[j, q, q])
                # value autocovariance of the head's value subspace, per region
                for (l, h) in LAT:
                    V = vals[l][:, :, h * Dh:(h + 1) * Dh].numpy()   # [B,S,Dh]
                    for j, x in enumerate(ch):
                        for reg, acc, nacc in (
                            ((x.acceptor_pos + 6, x.donor_pos - 6), C, nC),
                            ((x.donor_pos + 8, x.next_acceptor_pos - 8), Ci, nCi)):
                            a, b = reg[0] + OFF, reg[1] + OFF
                            if b - a < 40:
                                continue
                            Z = V[j, a:b]                     # [M,Dh]
                            Z = Z - Z.mean(0, keepdims=True)
                            M = Z.shape[0]
                            for t in range(8):
                                acc[(l, h)][t] += float((Z[:M - t] * Z[t:]).sum())
                                nacc[(l, h)][t] += (M - t) * Dh
    for hh in hooks:
        hh.remove()

    print(f"{len(X)} exons.  DIAGONAL WEIGHT f(0) at MASKED query positions, 42 lattice heads")
    d = np.array([np.mean(diag[hk]) for hk in LAT])
    for grp in (0, 1, 2):
        sel = [k for k in LAT if RPH[k] == grp]
        dd = np.array([np.mean(diag[k]) for k in sel])
        print(f"  rel-phase r={grp}  n={len(sel):2d}   mean f(0) = {dd.mean():.4f}"
              f"   max {dd.max():.4f}   min {dd.min():.4f}")
    print(f"  ALL 42            mean f(0) = {d.mean():.4f}   max {d.max():.4f}")
    print(f"  (uniform baseline over ~1026 columns = {1/1026:.5f})")

    print("\nVALUE AUTOCOVARIANCE  C(t)/C(0)  averaged over the 42 lattice heads")
    print(f"  {'region':7s} " + "  ".join(f"t={t}" for t in range(8)))
    for nm, acc, nacc in (("exon", C, nC), ("intron", Ci, nCi)):
        R = np.stack([acc[k] / np.maximum(nacc[k], 1) for k in LAT])
        R = R / R[:, :1]
        m = R.mean(0)
        print(f"  {nm:7s} " + "  ".join(f"{v:+.3f}" for v in m))
    print("\n  mod-3 signature: is C(3) > C(2) and C(6) > C(5)?  (that is what makes the")
    print("  self term -- and every other term -- sawtooth)")
    for nm, acc, nacc in (("exon", C, nC), ("intron", Ci, nCi)):
        R = np.stack([acc[k] / np.maximum(nacc[k], 1) for k in LAT])
        R = R / R[:, :1]
        m = R.mean(0)
        print(f"  {nm:7s}  C(3)-C(2) = {m[3]-m[2]:+.4f}   C(6)-C(5) = {m[6]-m[5]:+.4f}"
              f"   C(3)-C(4) = {m[3]-m[4]:+.4f}")

    # POSITION EMBEDDINGS: mod-3 structure with content held constant
    P = ad.model.bert.embeddings.position_embeddings.weight.detach().float().cpu().numpy()
    Pw = P[OFF:OFF + 1000]
    Pw = Pw - Pw.mean(0, keepdims=True)
    cc = np.array([float((Pw[:len(Pw) - t] * Pw[t:]).sum() / ((len(Pw) - t) * Pw.shape[1]))
                   for t in range(8)])
    cc = cc / cc[0]
    print("\nPOSITION-EMBEDDING autocovariance C_pos(t)/C_pos(0):")
    print("  " + "  ".join(f"t={t}:{v:+.3f}" for t, v in enumerate(cc)))


if __name__ == "__main__":
    main()
