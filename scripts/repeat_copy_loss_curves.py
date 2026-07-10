"""Evaluate train-vs-validation CE over saved repeat-copy checkpoints.

The training history's `train` column is a weighted minibatch loss, while `loss` is unweighted
validation AR CE. This script recomputes comparable CE on fixed train/val subsets at each saved
checkpoint and plots it alongside the copy-position accuracy history.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from gfmcd.paths import project_paths

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.repeat_copy import ar_inputs_targets, make_batch
from gfmcd.figstyle import BLUE, INK_2, ORANGE, RED, apply_figure_style
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

CKPT_RE = re.compile(r"repeat_copy_s(?P<seed>\d+)_step(?P<step>\d+)\.pt$")


def checkpoints_for(seed: int, root: Path) -> list[Path]:
    paths = []
    for path in root.glob(f"repeat_copy_s{seed}_step*.pt"):
        m = CKPT_RE.match(path.name)
        if m and int(m.group("step")) > 1:
            paths.append(path)
    return sorted(paths, key=lambda p: int(CKPT_RE.match(p.name).group("step")))  # type: ignore[union-attr]


def fixed_subset(seed: int, n: int, window: int, split: str):
    data_seed = seed if split == "train" else seed + 1
    batch = make_batch(n, window=window, seed=data_seed)
    x, y, mask = ar_inputs_targets(batch)
    return torch.from_numpy(x).long(), torch.from_numpy(y).long(), torch.from_numpy(mask)


@torch.no_grad()
def eval_ce(model: GFMNano, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, batch: int, device: str) -> dict:
    total_all = total_copy = copy_n = all_n = copy_corr = all_corr = 0.0
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].to(device)
        yb = y[i : i + batch].to(device)
        mb = mask[i : i + batch].to(device)
        logits = model(xb)
        per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="none").view_as(yb)
        total_all += float(per.sum())
        total_copy += float(per[mb].sum())
        all_n += float(yb.numel())
        copy_n += float(mb.sum())
        pred = logits.argmax(-1)
        all_corr += float((pred == yb).sum())
        copy_corr += float((pred[mb] == yb[mb]).sum())
    return {
        "all_ce": total_all / all_n,
        "copy_ce": total_copy / copy_n,
        "all_acc": all_corr / all_n,
        "copy_acc": copy_corr / copy_n,
    }


def load_history(seed: int, ckpt_root: Path) -> list[dict]:
    path = ckpt_root / f"repeat_copy_s{seed}_history.json"
    rows = json.loads(path.read_text())
    return [{"seed": seed, **row} for row in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,99")
    ap.add_argument("--checkpoint-dir", type=Path, default=project_paths().checkpoints)
    ap.add_argument("--out-dir", type=Path, default=project_paths().results / "repeat-copy")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--subset-seqs", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows: list[dict] = []

    for seed in seeds:
        train = fixed_subset(seed, args.subset_seqs, args.window, "train")
        val = fixed_subset(seed, args.subset_seqs, args.window, "val")
        for ckpt in checkpoints_for(seed, args.checkpoint_dir):
            step = int(CKPT_RE.match(ckpt.name).group("step"))  # type: ignore[union-attr]
            blob = torch.load(ckpt, map_location=device, weights_only=False)
            model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
            model.load_state_dict(blob["state"])
            tr = eval_ce(model, *train, args.batch, device)
            va = eval_ce(model, *val, args.batch, device)
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "train_all_ce": tr["all_ce"],
                    "val_all_ce": va["all_ce"],
                    "train_copy_ce": tr["copy_ce"],
                    "val_copy_ce": va["copy_ce"],
                    "train_copy_acc": tr["copy_acc"],
                    "val_copy_acc": va["copy_acc"],
                    "train_all_acc": tr["all_acc"],
                    "val_all_acc": va["all_acc"],
                }
            )
            print(
                f"s{seed} step {step:5d}  train all/copy CE {tr['all_ce']:.4f}/{tr['copy_ce']:.4f}  "
                f"val all/copy CE {va['all_ce']:.4f}/{va['copy_ce']:.4f}  val copy {va['copy_acc']:.4f}",
                flush=True,
            )

    fig_data = args.out_dir / "fig_data"
    fig_data.mkdir(parents=True, exist_ok=True)
    csv_path = fig_data / "repeat_loss_curves.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    hist_rows = []
    for seed in seeds:
        hist_rows.extend(load_history(seed, args.checkpoint_dir))

    apply_figure_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharex=True)
    for seed in seeds:
        ck = [r for r in rows if r["seed"] == seed]
        hist = [r for r in hist_rows if r["seed"] == seed]
        label = f"s{seed}"
        color = BLUE if seed == seeds[0] else ORANGE
        axes[0].plot([r["step"] for r in ck], [r["train_all_ce"] for r in ck], marker="o", color=color, linestyle="-", label=f"{label} train")
        axes[0].plot([r["step"] for r in ck], [r["val_all_ce"] for r in ck], marker="o", color=color, linestyle="--", label=f"{label} val")
        axes[1].plot([r["step"] for r in ck], [r["train_copy_ce"] for r in ck], marker="o", color=color, linestyle="-", label=f"{label} train")
        axes[1].plot([r["step"] for r in ck], [r["val_copy_ce"] for r in ck], marker="o", color=color, linestyle="--", label=f"{label} val")
        axes[2].plot([r["step"] for r in hist], [r["copy_acc"] for r in hist], color=color, label=label)

    axes[0].set_title("All-token CE")
    axes[0].set_ylabel("cross entropy")
    axes[1].set_title("Copy-position CE")
    axes[2].set_title("Copy accuracy")
    axes[2].set_ylim(0.2, 1.02)
    axes[2].axhline(0.75, color=INK_2, linestyle=":", linewidth=1.1)
    for ax in axes:
        ax.set_xlabel("training step")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    png_path = args.out_dir / "fig_repeat_loss_vs_copy.png"
    fig.savefig(png_path)

    md = args.out_dir / "LOSS_CURVES.md"
    final_lines = []
    for seed in seeds:
        last = max((r for r in rows if r["seed"] == seed), key=lambda r: r["step"])
        final_lines.append(
            f"| s{seed} | {last['train_all_ce']:.4f} | {last['val_all_ce']:.4f} | "
            f"{last['train_copy_ce']:.4f} | {last['val_copy_ce']:.4f} | {last['val_copy_acc']:.4f} |"
        )
    md.write_text(
        "\n".join(
            [
                "# Repeat-Copy Train/Validation Loss Curves",
                "",
                f"Comparable CE recomputed on fixed train/validation subsets of {args.subset_seqs} sequences per seed.",
                "Training-history `train` is not used for this comparison because it is a weighted minibatch loss.",
                "",
                "| Seed | Train All CE | Val All CE | Train Copy CE | Val Copy CE | Val Copy Acc |",
                "|---|---:|---:|---:|---:|---:|",
                *final_lines,
                "",
                f"Raw data: `{csv_path}`.",
                f"Figure: `{png_path}`.",
                "",
            ]
        )
    )
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
