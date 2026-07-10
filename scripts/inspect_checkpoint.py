"""Print a checkpoint's metadata without moving it onto an accelerator (TO_FIX.md §13.2).

    uv run python scripts/inspect_checkpoint.py checkpoints/nano_uniform_step8400.pt

Loads on CPU with weights_only=False and reports whatever provenance the checkpoint recorded
(config, objective/weights, seeds, step, git). Flags fields TO_FIX.md §13.2 asks every checkpoint
to carry but this one is missing, so the gap is visible rather than assumed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Each entry is a tuple of accepted aliases; the field is "present" if any alias is a key.
WANTED = [("config", "cfg"), ("objective",), ("data_seed",), ("model_seed",), ("step",), ("git_sha",), ("versions",)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}")
    if not isinstance(blob, dict):
        print(f"  (not a dict; type={type(blob).__name__})")
        return

    keys = [k for k in blob if k not in ("model", "state_dict", "optimizer")]
    print("  metadata keys:", ", ".join(sorted(keys)) or "(none)")
    for k in keys:
        v = blob[k]
        if isinstance(v, (int, float, str, bool)) or v is None:
            print(f"    {k}: {v}")
        elif isinstance(v, dict):
            print(f"    {k}: {{{', '.join(str(x) for x in list(v)[:8])}{'…' if len(v) > 8 else ''}}}")

    missing = [aliases[0] for aliases in WANTED if not any(a in blob for a in aliases)]
    if missing:
        print("  MISSING provenance fields (TO_FIX.md §13.2):", ", ".join(missing))


if __name__ == "__main__":
    main()
