"""Evaluate the repeat-copy objective sweep and decide the 'unbidden' question (TO_FIX.md §4.6).

For each weight arm (and the legacy weight-8 baseline repeat_copy_s0), report copy accuracy and the
cue-conflict distal-follow share (the causal verdict, not accuracy), then say whether the circuit forms
at copy weight 1. Writes results/repeat-copy/objective_sweep.json (provenance-stamped).

    uv run python scripts/eval_repeat_copy_sweep.py
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd import provenance
from gfmcd.data.repeat_copy import ar_inputs_targets, base_rate_ceiling, make_batch, single_conflict_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig
from gfmcd.paths import project_paths

COPY_ACC_MIN = 0.75
DISTAL_SHARE_MIN = 0.50
# (weight, seed) arms produced by the sweep, plus the legacy weight-8 baseline by explicit stem.
ARMS = [("repeat_copy_w1_s0", 1, 0), ("repeat_copy_w1_s1", 1, 1), ("repeat_copy_w1_s2", 1, 2),
        ("repeat_copy_w2_s0", 2, 0), ("repeat_copy_w4_s0", 4, 0), ("repeat_copy_s0", 8, 0)]


@torch.no_grad()
def _logits(model, tokens, batch, device):
    x = torch.from_numpy(tokens[:, :-1]).long()
    outs = [model(x[i : i + batch].to(device)).cpu() for i in range(0, len(x), batch)]
    return torch.cat(outs).numpy()


def verdict_for(stem: str, weight: int, seed: int, device: str) -> dict | None:
    ckpt = project_paths().checkpoints / f"{stem}_step8000.pt"
    if not ckpt.exists():
        return None
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    data = make_batch(2000, window=256, seed=4242)
    _, targets, copy_mask = ar_inputs_targets(data)
    logits = _logits(model, data.tokens, 64, device)
    copy_acc = float((logits.argmax(-1)[copy_mask] == targets[copy_mask]).mean())

    conflict_tokens, distal, local, score_mask = single_conflict_batch(data, seed=4243)
    cpred = _logits(model, conflict_tokens, 64, device).argmax(-1)
    distal_share = float((cpred[score_mask] == distal[score_mask]).mean())
    local_share = float((cpred[score_mask] == local[score_mask]).mean())
    formed = copy_acc > COPY_ACC_MIN and distal_share > DISTAL_SHARE_MIN and distal_share > local_share
    return {
        "stem": stem, "copy_weight": weight, "seed": seed,
        "base_rate": round(base_rate_ceiling(data), 4), "copy_acc": round(copy_acc, 4),
        "distal_share": round(distal_share, 4), "local_share": round(local_share, 4),
        "circuit_formed": formed,
    }


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = [r for a in ARMS if (r := verdict_for(*a, device)) is not None]

    print(f"{'arm':<20}{'w':>3}{'copy_acc':>10}{'distal':>9}{'local':>8}  verdict")
    for r in rows:
        print(f"{r['stem']:<20}{r['copy_weight']:>3}{r['copy_acc']:>10.4f}{r['distal_share']:>9.4f}"
              f"{r['local_share']:>8.4f}  {'FORMED' if r['circuit_formed'] else 'no circuit'}")

    w1 = [r for r in rows if r["copy_weight"] == 1]
    w1_formed = sum(r["circuit_formed"] for r in w1)
    if w1:
        unbidden = w1_formed >= max(1, (len(w1) + 1) // 2)  # majority of weight-1 seeds
        print(f"\nWeight-1 arms forming the circuit: {w1_formed}/{len(w1)}")
        print("UNBIDDEN:" , "SUPPORTED — the copy circuit forms under an UNWEIGHTED objective (copy weight 1)."
              if unbidden else
              "NOT SUPPORTED — the circuit needs the copy-position weighting; keep the corrected wording.")

    out = project_paths().results / "repeat-copy"
    out.mkdir(parents=True, exist_ok=True)
    blob = provenance.stamp({"arms": rows}, status="REPLICATED" if len(w1) >= 2 else "PROVISIONAL",
                            seed=4242, script="eval_repeat_copy_sweep.py")
    (out / "objective_sweep.json").write_text(json.dumps(blob, indent=2))
    print(f"\nwrote {out / 'objective_sweep.json'}")


if __name__ == "__main__":
    main()
