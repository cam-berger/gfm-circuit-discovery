#!/usr/bin/env python
"""Collect every number the phase-clock figure needs into ONE json, so the render is a pure
function of the data and the figure can never drift from the result it claims to show."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, load_examples,
    pick_device,
)

BAND = list(range(339, 346))
CTRL = list(range(94, 101))                     # energy-matched control band
COMBS = [(0, 0), (0, 11), (1, 10), (1, 12), (2, 2), (2, 15)]
SPAN = 24
OUT = REPO / "results/phase-lattice/fig_phase_data.json"

dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=4)
m = ad.model
emb = m.bert.embeddings.position_embeddings
P0 = emb.weight.detach().clone()
P = P0.float().cpu().numpy()
N, D = P.shape
mu = P.mean(0, keepdims=True)
F = np.fft.rfft(P - mu, axis=0)


def strip(bins):
    G = F.copy()
    for b in bins:
        G[b] = 0
    return torch.tensor(np.fft.irfft(G, n=N, axis=0) + mu, dtype=P0.dtype, device=P0.device)


P3 = strip(BAND)
PC = strip(CTRL)

R: dict = {"band": BAND, "ctrl_band": CTRL, "combs": [list(c) for c in COMBS]}

# ---------------------------------------------------------------- A. the spectrum of P
pw = (np.abs(F) ** 2).sum(1)
pw[0] = 0.0
med = float(np.median(pw[1:]))
R["spectrum"] = dict(
    median=med,
    bins=[dict(k=int(k), period=float(N / k), rel=float(pw[k] / med))
          for k in range(1, len(pw))],
    band_power=float(pw[BAND].sum()), ctrl_power=float(pw[CTRL].sum()),
    band_rel=float(pw[BAND].sum() / med), peak_rel=float(pw[342] / med),
)

# ---------------------------------------------------------------- B. SVD -> the clock is rank 2
G = np.zeros_like(F)
G[BAND] = F[BAND]
CAR = np.fft.irfft(G, n=N, axis=0)
S = np.linalg.svd(CAR, compute_uv=False)
ev = (S ** 2 / (S ** 2).sum())
R["svd"] = [float(x) for x in ev[:10]]

# ---------------------------------------------------------------- C. R(k) AS THE MODEL SEES IT
# The model never sees raw P: layer 0 sees LayerNorm(E_base + P + E_seg); layers 1-2 see the
# residual stream.  Use the empirical position-conditional mean hidden state at each layer.
tok = AutoTokenizer.from_pretrained(str(REPO / DEFAULT_MODEL))
rng = np.random.default_rng(0)
seqs = ["".join(rng.choice(list("ACGT"), 1024)) for _ in range(16)]
enc = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(ad.device)

nH = m.config.num_attention_heads
dh = m.config.hidden_size // nH


def hidden_means(Pw):
    emb.weight.data = Pw.clone()
    with torch.no_grad():
        hs = m.bert(**enc, output_hidden_states=True).hidden_states
    return [h.mean(0).float().cpu().numpy() for h in hs]


def Rk(l, h, Hin, span=12, lo=200, hi=800):
    sa = m.bert.encoder.layer[l].attention.self
    Wq = sa.query.weight.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
    bq = sa.query.bias.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
    Wk = sa.key.weight.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
    bk = sa.key.bias.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
    Q = Hin @ Wq.T + bq
    K = Hin @ Wk.T + bk
    idx = np.arange(lo, hi)
    return [float(np.mean(np.sum(Q[idx] * K[idx + k], axis=1)) / np.sqrt(dh))
            for k in range(-span, span + 1)]


H_int = hidden_means(P0)
H_no3 = hidden_means(P3)
H_ctl = hidden_means(PC)
R["Rk"] = {f"L{l}H{h}": dict(intact=Rk(l, h, H_int[l]), no_carrier=Rk(l, h, H_no3[l]),
                             no_control=Rk(l, h, H_ctl[l]))
           for (l, h) in COMBS}
R["Rk_offsets"] = list(range(-12, 13))

# ---------------------------------------------------------------- D. empirical attention phase mass
X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 8, task=task, tag="fig", min_effect=None)


def phase_mass(Pw):
    emb.weight.data = Pw.clone()
    mass = {c: np.zeros(3) for c in COMBS}
    r = np.random.default_rng(0)
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            e = task.encode([x.sequence for x in ch], [x.donor_pos for x in ch])
            A = m(**e, output_attentions=True).attentions
            Slen = e["input_ids"].shape[1]
            qs = r.integers(SPAN + 4, Slen - SPAN - 4, size=32)
            for (l, h) in mass:
                Al = A[l][:, h].float().cpu().numpy()
                for b in range(len(ch)):
                    for p in qs:
                        for k in range(-SPAN, SPAN + 1):
                            if k:
                                mass[(l, h)][k % 3] += Al[b, p, p + k]
    return {f"L{l}H{h}": float((v / v.sum()).max()) for (l, h), v in mass.items()}


R["phase_mass"] = dict(intact=phase_mass(P0), no_carrier=phase_mass(P3),
                       no_control=phase_mass(PC), chance=1 / 3)
emb.weight.data = P0.clone()

# ---------------------------------------------------------------- E + F. from the run artifacts
dose = json.loads((REPO / "results/phase-lattice/carrier_dose.json").read_text())
prov = json.loads((REPO / "results/phase-lattice/carrier_provenance.json").read_text())
med_ = json.loads((REPO / "results/phase-lattice/carrier_mediation.json").read_text())
CH = float(np.log(0.25))
base = next(r for r in dose["rows"] if r["alpha"] == 1.0)
hr0 = {k: base[k] - CH for k in ("exon", "intron")}
R["dose"] = [dict(alpha=r["alpha"], band=r["band"],
                  exon_kept=(r["exon"] - CH) / hr0["exon"],
                  intron_kept=(r["intron"] - CH) / hr0["intron"],
                  coupling_kept=r["coupling"] / base["coupling"])
             for r in dose["rows"]]
R["provenance"] = {k: dict(enrich=v["enrichment"], mod3=v["mod3_sep"])
                   for k, v in prov["provenance"].items()}
_res = med_["rescue"]
_n1 = [v["rescue"] for k, v in _res.items() if k.startswith("__null1_")]
_n6 = [v["rescue"] for k, v in _res.items() if k.startswith("__null6_")]
R["mediation"] = dict(
    intact=med_["coupling_intact"], killed=med_["coupling_killed"],
    rescue={k: v["rescue"] for k, v in _res.items() if not k.startswith("__")},
    null1=_n1, null6=_n6,
    null1_mean=float(np.mean(_n1)), null1_max=float(np.max(_n1)),
    null6_mean=float(np.mean(_n6)), null6_max=float(np.max(_n6)),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(R, indent=1))
print(f"saved -> {OUT}")
print(f"  spectrum: period-3 band = {R['spectrum']['band_rel']:.0f}x median, "
      f"peak bin = {R['spectrum']['peak_rel']:.0f}x")
print(f"  svd: top-2 = {100*(R['svd'][0]+R['svd'][1]):.1f}% of carrier energy")
print(f"  phase mass: intact {np.mean(list(R['phase_mass']['intact'].values())):.3f} -> "
      f"no-carrier {np.mean(list(R['phase_mass']['no_carrier'].values())):.3f} "
      f"(chance 0.333), control {np.mean(list(R['phase_mass']['no_control'].values())):.3f}")
