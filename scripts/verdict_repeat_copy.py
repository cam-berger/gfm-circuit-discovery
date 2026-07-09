"""Causal verdict for the repeat-copy mechanism.

Accuracy alone is not the verdict. The cue conflict edits the first occurrence while leaving the
second occurrence's local continuation unchanged. A copy circuit should follow the edited distal
copy; a local shortcut should not.

    python scripts/verdict_repeat_copy.py --checkpoint checkpoints/repeat_copy_s0_step8000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.repeat_copy import ar_inputs_targets, base_rate_ceiling, conflict_batch, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

COPY_ACC_MIN = 0.75
DISTAL_SHARE_MIN = 0.50


@torch.no_grad()
def logits_for(model: GFMNano, tokens: np.ndarray, batch: int, device: str) -> np.ndarray:
    x = torch.from_numpy(tokens[:, :-1]).long()
    outs = []
    for i in range(0, len(x), batch):
        outs.append(model(x[i : i + batch].to(device)).cpu())
    return torch.cat(outs).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seqs", type=int, default=2000)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    data = make_batch(args.seqs, window=args.window, seed=args.seed)
    _, targets, copy_mask = ar_inputs_targets(data)

    logits = logits_for(model, data.tokens, args.batch, device)
    pred = logits.argmax(axis=-1)
    copy_acc = float((pred[copy_mask] == targets[copy_mask]).mean())
    loss = float(F.cross_entropy(torch.from_numpy(logits).reshape(-1, logits.shape[-1]), torch.from_numpy(targets).reshape(-1)))

    conflict_tokens, distal, local = conflict_batch(data)
    conflict_logits = logits_for(model, conflict_tokens, args.batch, device)
    conflict_pred = conflict_logits.argmax(axis=-1)
    distal_share = float((conflict_pred[copy_mask] == distal[copy_mask]).mean())
    local_share = float((conflict_pred[copy_mask] == local[copy_mask]).mean())
    neither_share = 1.0 - distal_share - local_share

    print(f"{args.checkpoint.name}   step {blob['step']}   seed {blob.get('seed', '?')}\n")
    print(f"{'base-rate copy ceiling':<42}{base_rate_ceiling(data):>10.4f}")
    print(f"{'AR loss, intact':<42}{loss:>10.4f}")
    print(f"{'copy accuracy, intact':<42}{copy_acc:>10.4f}   threshold > {COPY_ACC_MIN}")
    print()
    print("cue conflict on copy positions")
    print(f"{'  follows DISTAL edited copy':<42}{distal_share:>10.4f}   threshold > {DISTAL_SHARE_MIN}")
    print(f"{'  follows LOCAL/original continuation':<42}{local_share:>10.4f}")
    print(f"{'  neither':<42}{neither_share:>10.4f}")

    formed = copy_acc > COPY_ACC_MIN and distal_share > DISTAL_SHARE_MIN and distal_share > local_share
    print(f"\nVERDICT: {'COPY CIRCUIT FORMED' if formed else 'NO COPY CIRCUIT'}")


if __name__ == "__main__":
    main()
