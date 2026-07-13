"""AUDIT: does the SELF column (j=i) of the roll carry a mod-3 sawtooth of its own?

Three arms, delta = 0..6:
  FULL      : the pre-registered roll (all nucleotide columns)
  SELFONLY  : move ONLY the diagonal mass A[i,i] to column i+delta.  Everything else stays.
  NOSELF    : the pre-registered roll but with the diagonal HELD IN PLACE (attacker's fix)
Fit damage(d) = a*d + b*1[d%3!=0] + c on each, in exon interior and intron interior.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, load_examples,
    pick_device,
)
from contextlib import contextmanager

MASK_STRIDE = 7
MIN_LEN = 60
_ROLL: dict = {}
SELF_W: dict = {}   # (l,h) -> list of diagonal weights at masked positions


def _fwd(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    w = torch.nn.functional.softmax(w, dim=-1)
    spec = _ROLL.get(getattr(module, "layer_idx", -1))
    if spec is not None:
        heads, delta, lo, hi, mode = spec
        w = w.clone()
        S = w.shape[-1]
        idx = torch.arange(S, device=w.device)
        hh = torch.tensor(heads, device=w.device).view(-1, 1)   # [nh,1]
        ii = idx.view(1, -1)                                    # [1,S]
        inrow = ((idx >= lo) & (idx < hi)).view(1, 1, -1)
        if mode in ("selfonly", "noself"):
            diag = w[:, hh, ii, ii].clone() * inrow              # [B,nh,S]
        if mode in ("full", "noself"):
            blk = w[:, heads, :, lo:hi]
            blk = torch.roll(blk, shifts=delta, dims=-1)
            if delta > 0:
                blk[..., :delta] = 0.0
            elif delta < 0:
                blk[..., delta:] = 0.0
            w[:, heads, :, lo:hi] = blk
        if mode in ("selfonly", "noself"):
            src = idx + delta
            ok = ((src >= lo) & (src < hi) & (idx >= lo) & (idx < hi)).view(1, 1, -1)
            si = src.clamp(0, S - 1).view(1, -1)
            if mode == "noself":
                # take the diagonal mass back OUT of where the roll deposited it ...
                w[:, hh, ii, si] = w[:, hh, ii, si] - torch.where(ok, diag, torch.zeros_like(diag))
                # ... and put it back home on the diagonal
                w[:, hh, ii, ii] = w[:, hh, ii, ii] + diag
            else:   # selfonly: move ONLY the diagonal, leave every other column alone
                w[:, hh, ii, ii] = w[:, hh, ii, ii] - diag
                w[:, hh, ii, si] = w[:, hh, ii, si] + torch.where(ok, diag, torch.zeros_like(diag))
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    out = torch.matmul(w, value).transpose(1, 2).contiguous()
    return out, w


ALL_ATTENTION_FUNCTIONS.register("audit_roll", _fwd)


@contextmanager
def roll(ad, heads, delta, seq_len, mode, off=1):
    by = {}
    for l, h in heads:
        by.setdefault(l, []).append(h)
    prev = ad.model.config._attn_implementation
    _ROLL.clear()
    for l, hs in by.items():
        _ROLL[l] = (sorted(hs), int(delta), off, off + seq_len, mode)
    ad.model.config._attn_implementation = "audit_roll"
    try:
        yield
    finally:
        _ROLL.clear()
        ad.model.config._attn_implementation = prev


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    dev = pick_device(None)
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=2)
    OFF = task.off
    inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
    LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
    X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", n, task=task, tag="phasefn",
                         min_effect=None)
    X = [x for x in X if x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
    print(f"{len(X)} exons  device={dev}  lattice={len(LAT)}")

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    def ll(mode, delta):
        """mean log-prob of true base at MASKED positions, per region."""
        acc = {"exon": [], "intron": []}
        with torch.no_grad():
            for i in range(0, len(X), task.batch):
                ch = X[i:i + task.batch]
                lp = [np.full(len(x.sequence), np.nan) for x in ch]
                for o in range(MASK_STRIDE):
                    enc = ad.tokenizer([" ".join(x.sequence) for x in ch],
                                       return_tensors="pt", padding=True).to(ad.device)
                    pos = []
                    for j, x in enumerate(ch):
                        ps = [p for p in range(len(x.sequence)) if p % MASK_STRIDE == o]
                        for p in ps:
                            enc["input_ids"][j, p + OFF] = task.mask_id
                        pos.append(ps)
                    L = enc["input_ids"].shape[1] - 2   # nucleotide columns (CLS..SEP)
                    ctx = (roll(ad, LAT, delta, L, mode, off=OFF)
                           if mode != "none" else _null())
                    with ctx:
                        lg = ad.model(**enc).logits.float()
                    lgp = torch.log_softmax(lg, -1)
                    for j, x in enumerate(ch):
                        idx = torch.tensor([p + OFF for p in pos[j]], device=lgp.device)
                        tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]],
                                           device=lgp.device)
                        v = lgp[j, idx, tgt].cpu().numpy()
                        for p, q in zip(pos[j], v):
                            lp[j][p] = q
                for j, x in enumerate(ch):
                    for nm, (a, b) in regions(x).items():
                        if b - a >= 24:
                            acc[nm].append(np.nanmean(lp[j][a:b]))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    @contextmanager
    def _null():
        yield

    base = ll("none", 0)
    print(f"intact: exon {base['exon']:+.4f}  intron {base['intron']:+.4f} "
          f"(chance {np.log(0.25):+.4f})")

    D = {}
    for mode in ("selfonly", "noself", "full"):
        rows = []
        for d in range(7):
            r = ll(mode, d)
            rows.append(r)
            print(f"  {mode:9s} d={d}  exon LL {r['exon']:+.4f}  intron LL {r['intron']:+.4f}",
                  flush=True)
        D[mode] = rows

    Dl = np.arange(7)
    Xd = np.stack([Dl, (Dl % 3 != 0).astype(float), np.ones(7)], 1)
    print("\n" + "=" * 70)
    print(f"{'arm':10s} {'region':7s} {'a (slope)':>10s} {'b (PHASE)':>10s} {'c':>9s}"
          f" {'dmg(2)':>8s} {'dmg(3)':>8s}")
    for mode in ("selfonly", "noself", "full"):
        for reg in ("exon", "intron"):
            dmg = np.array([D[mode][0][reg] - D[mode][d][reg] for d in range(7)])
            a, b, c = np.linalg.lstsq(Xd, dmg, rcond=None)[0]
            print(f"{mode:10s} {reg:7s} {a:10.5f} {b:10.5f} {c:9.5f} "
                  f"{dmg[2]:8.4f} {dmg[3]:8.4f}")
    json.dump({m: D[m] for m in D}, open("/tmp/audit_selfterm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
