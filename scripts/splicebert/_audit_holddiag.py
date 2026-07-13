"""AUDIT: does the DIAGONAL carry the sawtooth?

Two roll variants, same 42 lattice heads, same delta scan, same scored positions:
  STD   -- roll every nucleotide column, incl. the self column i  (the pre-registered design)
  HOLD  -- roll only columns j != i; the self mass a_i(0) STAYS on column i, then renormalise

If the sawtooth b survives HOLD, the diagonal is not doing the work and the
"[MASK]-self-column information injection" attack is refuted empirically.
If b collapses under HOLD, the attack is real.
"""
import json, sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np, torch

REPO = Path("/Users/cberg/Research/projects/gfm-circuit-discovery")
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import (DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask,
                                    load_adapter, load_examples, pick_device, noop)
from transformers.models.bert.modeling_bert import ALL_ATTENTION_FUNCTIONS

MASK_STRIDE = 7
MIN_LEN = 60
_SPEC = {}


def _fwd(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    w = torch.nn.functional.softmax(w, dim=-1)
    spec = _SPEC.get(getattr(module, "layer_idx", -1))
    if spec is not None:
        heads, delta, lo, hi, hold = spec
        w = w.clone()
        blk = w[:, heads, :, lo:hi].clone()          # [B,nh,S,L]
        L = hi - lo
        if hold:
            sub = blk[:, :, lo:hi, :]                # [B,nh,L,L] square: row lo+t, col t = self
            diag = sub.diagonal(dim1=-2, dim2=-1).clone()
            sub = sub - torch.diag_embed(diag)       # remove self mass before rolling
            sub = torch.roll(sub, shifts=delta, dims=-1)
            if delta > 0:
                sub[..., :delta] = 0.0
            elif delta < 0:
                sub[..., delta:] = 0.0
            sub = sub + torch.diag_embed(diag)       # put self mass back where it was
            blk2 = torch.roll(blk, shifts=delta, dims=-1)
            if delta > 0:
                blk2[..., :delta] = 0.0
            elif delta < 0:
                blk2[..., delta:] = 0.0
            blk2[:, :, lo:hi, :] = sub               # scored rows use the hold-diag version
            blk = blk2
        else:
            blk = torch.roll(blk, shifts=delta, dims=-1)
            if delta > 0:
                blk[..., :delta] = 0.0
            elif delta < 0:
                blk[..., delta:] = 0.0
        w[:, heads, :, lo:hi] = blk
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    out = torch.matmul(w, value).transpose(1, 2).contiguous()
    return out, w


ALL_ATTENTION_FUNCTIONS.register("gfmcd_audit", _fwd)


@contextmanager
def roll2(ad, heads, delta, seq_len, off=1, hold=False):
    by = {}
    for l, h in heads:
        by.setdefault(l, []).append(h)
    prev = ad.model.config._attn_implementation
    _SPEC.clear()
    for l, hs in by.items():
        _SPEC[l] = (sorted(hs), int(delta), off, off + seq_len, hold)
    ad.model.config._attn_implementation = "gfmcd_audit"
    try:
        yield
    finally:
        _SPEC.clear()
        ad.model.config._attn_implementation = prev


dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=4)
OFF = task.off
inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
# WORST CASE FOR THE ATTACK: layer-0 lattice heads. At layer 0 the self column really is the
# bare [MASK] embedding (base-uninformative), so the "information injection" is maximal there.
# At layers>=1 the self column is the residual stream, which is the single most base-informative
# vector available for predicting base i -- rolling it away is a LOSS, not an injection.
_which = sys.argv[1] if len(sys.argv) > 1 else "L0"
if _which == "L0":
    LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"] and r["layer"] == 0]
elif _which == "ALL":
    LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
else:
    l, h = _which.split(",")
    LAT = [(int(l), int(h))]
print("head set:", LAT)

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 40, task=task, tag="audit-hold",
                     min_effect=None)
X = [x for x in X if x.exon_length >= MIN_LEN and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
X = X[:16]
print(f"{len(X)} exons  device={dev}  {len(LAT)} lattice heads")


def ll(ctxf):
    """mean log-prob of true base at MASKED positions, exon interior and intron interior."""
    acc = {"exon": [], "intron": []}
    with torch.no_grad():
        for i in range(0, len(X), 4):
            ch = X[i:i + 4]
            lp = [np.full(len(x.sequence), np.nan) for x in ch]
            for o in range(MASK_STRIDE):
                enc = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                                   padding=True).to(ad.device)
                S = enc["input_ids"].shape[1]
                pos = []
                for j, x in enumerate(ch):
                    ps = [p for p in range(len(x.sequence)) if p % MASK_STRIDE == o]
                    for p in ps:
                        enc["input_ids"][j, p + OFF] = task.mask_id
                    pos.append(ps)
                with ctxf(S - 2 * OFF if OFF else S):
                    lg = ad.model(**enc).logits.float()
                lgp = torch.log_softmax(lg, -1)
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=lgp.device)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]],
                                       device=lgp.device)
                    for p, v in zip(pos[j], lgp[j, idx, tgt].cpu().numpy()):
                        lp[j][p] = v
            for j, x in enumerate(ch):
                a, b = x.acceptor_pos + 6, x.donor_pos - 6
                c, d = x.donor_pos + 8, x.next_acceptor_pos - 8
                if b - a >= 24:
                    acc["exon"].append(np.nanmean(lp[j][a:b]))
                if d - c >= 24:
                    acc["intron"].append(np.nanmean(lp[j][c:d]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def mk(delta, hold):
    def f(seq_len):
        return roll2(ad, LAT, delta, seq_len, off=OFF, hold=hold)
    return f


base = ll(lambda s: noop())
print(f"intact:  exon {base['exon']:+.4f}   intron {base['intron']:+.4f}   "
      f"headroom(exon) {base['exon'] - np.log(0.25):+.4f}")

res = {}
for hold in (False, True):
    tag = "HOLD-DIAG" if hold else "STD"
    res[tag] = {}
    for delta in range(7):
        r = ll(mk(delta, hold))
        dam_e = base["exon"] - r["exon"]
        dam_i = base["intron"] - r["intron"]
        hr = (r["exon"] - np.log(0.25)) / (base["exon"] - np.log(0.25))
        res[tag][delta] = dict(exon=r["exon"], intron=r["intron"], dam_e=dam_e, dam_i=dam_i,
                               headroom_frac=hr)
        print(f"  {tag:10s} d={delta}  LL_exon {r['exon']:+.4f}  damage_exon {dam_e:+.4f}  "
              f"damage_intron {dam_i:+.4f}  headroom {100*hr:5.1f}%")


def fit(d):
    """damage(delta) = a*delta + b*1[delta%3!=0] + c ; return a,b,c"""
    ds = np.arange(7)
    Xm = np.stack([ds, (ds % 3 != 0).astype(float), np.ones(7)], 1)
    y = np.array([d[k] for k in range(7)])
    coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return coef


print("\n" + "=" * 78)
for tag in ("STD", "HOLD-DIAG"):
    ae, be, ce = fit({k: v["dam_e"] for k, v in res[tag].items()})
    ai, bi, ci = fit({k: v["dam_i"] for k, v in res[tag].items()})
    print(f"{tag:10s}  EXON  a={ae:+.5f} b={be:+.5f} c={ce:+.5f} | "
          f"INTRON b={bi:+.5f} | DiD = {be - bi:+.5f}")
d3 = res["STD"][3]["dam_e"]; d2 = res["STD"][2]["dam_e"]
h3 = res["HOLD-DIAG"][3]["dam_e"]; h2 = res["HOLD-DIAG"][2]["dam_e"]
print(f"\nSHAPE gate  damage(3) < damage(2)?   STD: {d3:.4f} < {d2:.4f} = {d3 < d2}   "
      f"HOLD: {h3:.4f} < {h2:.4f} = {h3 < h2}")
json.dump(res, open("/tmp/holddiag.json", "w"), indent=1)
