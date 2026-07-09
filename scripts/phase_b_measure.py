"""Measure the 2-D signature, the cue-conflict decay curve, and the layer probe. Dump; do not re-derive.

Five models on one plot. Both axes are probe-based, so the mu=0 nucleotide-head LM and the mu=1
amino-acid-head cells are commensurable (a far-field Δloss would not be).

    python scripts/phase_b_measure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from train_cell import pool  # noqa: E402

from gfmcd.data.genome import Split  # noqa: E402
from gfmcd.models.gfm_nano import GFMNano, NanoConfig  # noqa: E402
from gfmcd.signatures import (  # noqa: E402
    anchor_content_by_distance,
    layer_probe,
    peak_anchor_dependence,
    signature,
    start_index,
)

OUT = Path("results/fig_data")
N = 600

# (key, checkpoint, eval pool, mechanism from D33)
MODELS = [
    ("A · uniform, easy detector", "nano_uniform_step8400.pt", "A", "counter"),
    ("B · uniform, hard detector", "cellB_s1_step24000.pt", "B", "mixed"),
    ("C · real CDS, easy detector", "cellC_s0_step8000.pt", "C", "relay"),
    ("D · real CDS, natural", "cellD_s0_step8000.pt", "D", "relay"),
    ("μ=0 · real CDS, LM objective", "nano_ar_internal_step8000.pt", "D", "bag"),
]


def load(name: str, device: str) -> GFMNano:
    blob = torch.load(Path("checkpoints") / name, map_location=device, weights_only=False)
    m = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    m.load_state_dict(blob["state"])
    return m


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    sigs, curves, layers = [], {}, {}
    print(f"{'model':<32}{'x = peak anchor dep':>21}{'at layer':>16}{'y = anchor share':>18}{'content':>10}")

    for key, ckpt, pool_name, mech in MODELS:
        model = load(ckpt, device)
        tok, aa, ph = pool(pool_name, N, model.cfg.max_len, Split.VAL, 4242)
        start = start_index(ph)

        sig, extra, cache = signature(model, tok, ph, device, key)
        peak, best_layer, per_layer = peak_anchor_dependence(model, tok, ph, device)
        sigs.append(dict(model=key, ckpt=ckpt, mechanism=mech, x=peak, best_layer=best_layer,
                         y=sig.anchor_share, content=sig.content_share,
                         x_at_L2=sig.ablation_effect, per_layer=per_layer, n_far=sig.n_far, **extra))
        print(f"{key:<32}{peak:>21.4f}{best_layer.replace('.resid_post',''):>16}"
              f"{sig.anchor_share:>18.4f}{sig.content_share:>10.4f}", flush=True)

        curve, _ = anchor_content_by_distance(model, tok, ph, start, device,
                                             cached=(cache["r_int"], cache["r_sh"]))
        curves[key] = curve

        layers[key] = layer_probe(model, tok, ph, device)

    # untrained control for the layer figure
    torch.manual_seed(1234)
    ref = load(MODELS[0][1], device)
    untrained = GFMNano(ref.cfg).to(device).eval()
    tok, aa, ph = pool("A", N, ref.cfg.max_len, Split.VAL, 4242)
    layers["untrained (random init)"] = layer_probe(untrained, tok, ph, device)

    (OUT / "signature.json").write_text(json.dumps(sigs, indent=2))
    (OUT / "cue_conflict_decay.json").write_text(json.dumps(curves, indent=2))
    (OUT / "layer_probe.json").write_text(json.dumps(layers, indent=2))
    print(f"\nwrote {OUT}/signature.json, cue_conflict_decay.json, layer_probe.json")


if __name__ == "__main__":
    main()
