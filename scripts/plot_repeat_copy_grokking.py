"""Plot repeat-copy grokking curves from checkpoint histories.

Writes:
  branches/codex-repeat-induction/fig_data/repeat_grok.csv
  branches/codex-repeat-induction/fig_repeat_grok.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def load_history(path: Path, seed: str) -> list[dict]:
    rows = json.loads(path.read_text())
    out = []
    for row in rows:
        out.append(
            {
                "seed": seed,
                "step": int(row["step"]),
                "train": float(row["train"]),
                "loss": float(row["loss"]),
                "copy_acc": float(row["copy_acc"]),
                "all_acc": float(row["all_acc"]),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("branches/codex-repeat-induction"))
    ap.add_argument("--base-floor", type=float, default=0.2516)
    ap.add_argument("--histories", nargs="+", default=["s0=checkpoints/repeat_copy_s0_history.json", "s99=checkpoints/repeat_copy_s99_history.json"])
    args = ap.parse_args()

    fig_data = args.out_dir / "fig_data"
    fig_data.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in args.histories:
        seed, path = spec.split("=", 1)
        rows.extend(load_history(Path(path), seed))

    csv_path = fig_data / "repeat_grok.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "step", "train", "loss", "copy_acc", "all_acc"])
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = {"s0": "#0b5cad", "s99": "#b03a2e"}
    for seed in sorted({r["seed"] for r in rows}):
        xs = [r["step"] for r in rows if r["seed"] == seed]
        ys = [r["copy_acc"] for r in rows if r["seed"] == seed]
        ax.plot(xs, ys, marker="o", markersize=2.8, linewidth=1.8, label=seed, color=colors.get(seed))

    ax.axhline(args.base_floor, color="#444444", linestyle="--", linewidth=1.2, label=f"base-rate floor {args.base_floor:.3f}")
    ax.axhline(0.75, color="#777777", linestyle=":", linewidth=1.1, label="verdict threshold 0.75")
    ax.set_xlabel("training step")
    ax.set_ylabel("copy-position accuracy")
    ax.set_ylim(0.2, 1.02)
    ax.set_title("Repeat-copy circuit formation")
    ax.legend(frameon=False, loc="lower right")
    ax.annotate("transition", xy=(3750, 0.74), xytext=(2850, 0.88), arrowprops={"arrowstyle": "->", "lw": 1.0})
    fig.tight_layout()
    png_path = args.out_dir / "fig_repeat_grok.png"
    fig.savefig(png_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
