"""Ceiling audit for the repeat-copy branch.

The decisive stop-condition is the local RF-limited CNN: if a model denied access to the first
occurrence reaches the trained transformer's copy accuracy, the branch is demoted.

Writes:
  branches/codex-repeat-induction/fig_data/copy_ceiling.csv
  branches/codex-repeat-induction/fig_copy_ceiling.png
  branches/codex-repeat-induction/COPY_CEILING.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gfmcd.data.repeat_copy import ar_inputs_targets, base_rate_ceiling, make_batch


def apply_style() -> None:
    from gfmcd.figstyle import apply_figure_style

    apply_figure_style()


def local_examples(tokens: np.ndarray, targets: np.ndarray, copy_mask: np.ndarray, crop: int) -> tuple[np.ndarray, np.ndarray]:
    """Return local left-context crops ending at the prediction position.

    The crop is AR-aligned: for position `p`, the model sees `tokens[..., :p+1]` and predicts
    `tokens[p+1]`. The source occurrence is at least ~49 nt behind by construction, so a 33 nt crop
    is denied source access.
    """

    if crop < 1:
        raise ValueError("crop must be positive")
    rows, positions = np.nonzero(copy_mask)
    x = np.zeros((len(rows), crop), dtype=np.int64)
    y = targets[rows, positions].astype(np.int64)
    for i, (r, p) in enumerate(zip(rows, positions)):
        lo = max(0, p - crop + 1)
        chunk = tokens[r, lo : p + 1]
        x[i, -len(chunk) :] = chunk
    return x, y


def subsample(x: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx], y[idx]


def base_rate(y_train: np.ndarray, y_val: np.ndarray) -> float:
    counts = np.bincount(y_train, minlength=4)
    pred = int(counts.argmax())
    return float((y_val == pred).mean())


def local_naive_bayes(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
    """Position-specific multinomial NB over the local crop."""

    crop = x_train.shape[1]
    priors = np.bincount(y_train, minlength=4).astype(float) + 1.0
    log_priors = np.log(priors / priors.sum())
    counts = np.ones((4, crop, 4), dtype=float)
    for c in range(4):
        xc = x_train[y_train == c]
        if len(xc) == 0:
            continue
        for pos in range(crop):
            counts[c, pos] += np.bincount(xc[:, pos], minlength=4)
    probs = counts / counts.sum(axis=-1, keepdims=True)
    log_probs = np.log(probs)

    scores = np.tile(log_priors, (len(x_val), 1))
    for pos in range(crop):
        scores += log_probs[:, pos, x_val[:, pos]].T
    pred = scores.argmax(axis=1)
    return float((pred == y_val).mean())


class LocalCNN(nn.Module):
    def __init__(self, crop: int):
        super().__init__()
        self.embed = nn.Embedding(4, 16)
        self.net = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.readout = nn.Linear(64 * crop, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).transpose(1, 2)
        h = self.net(h).flatten(1)
        return self.readout(h)


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch: int,
    seed: int,
    device: str,
) -> tuple[float, list[dict]]:
    torch.manual_seed(seed)
    model = LocalCNN(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    xt = torch.from_numpy(x_train).long()
    yt = torch.from_numpy(y_train).long()
    xv = torch.from_numpy(x_val).long().to(device)
    yv = torch.from_numpy(y_val).long().to(device)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(xt), generator=gen)
        model.train()
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            logits = model(xt[idx].to(device))
            loss = F.cross_entropy(logits, yt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = []
            for i in range(0, len(xv), batch):
                pred.append(model(xv[i : i + batch]).argmax(-1).cpu())
            pred_t = torch.cat(pred)
            acc = float((pred_t == yv.cpu()).float().mean())
        hist.append({"epoch": epoch, "val_acc": acc})
        print(f"cnn epoch {epoch:02d} val_acc {acc:.4f}", flush=True)
    return hist[-1]["val_acc"], hist


def write_outputs(out_dir: Path, rows: list[dict], cnn_hist: list[dict], crop: int, model_acc: float) -> None:
    fig_data = out_dir / "fig_data"
    fig_data.mkdir(parents=True, exist_ok=True)
    csv_path = fig_data / "copy_ceiling.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "accuracy", "sees_source", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    with (fig_data / "copy_ceiling_cnn_history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "val_acc"])
        writer.writeheader()
        writer.writerows(cnn_hist)

    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = [r["name"] for r in rows]
    vals = [float(r["accuracy"]) for r in rows]
    colors = ["#8c8c8c" if r["sees_source"] == "no" else "#0b5cad" for r in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.axhline(model_acc, color="#b03a2e", linestyle="--", linewidth=1.2, label=f"GFMNano {model_acc:.3f}")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("copy-position accuracy")
    ax.set_title("Repeat-copy ceiling audit")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    png_path = out_dir / "fig_copy_ceiling.png"
    fig.savefig(png_path)

    rf = next(r for r in rows if r["name"] == "RF-limited CNN")
    fired = float(rf["accuracy"]) >= model_acc - 0.02
    md = out_dir / "COPY_CEILING.md"
    md.write_text(
        "\n".join(
            [
                "# Repeat-Copy Ceiling Audit",
                "",
                f"Local crop length: {crop} nt. The source occurrence is outside this crop by construction.",
                "",
                "| Baseline | Sees Source? | Copy Accuracy | Notes |",
                "|---|---:|---:|---|",
                *[
                    f"| {r['name']} | {r['sees_source']} | {float(r['accuracy']):.4f} | {r['notes']} |"
                    for r in rows
                ],
                "",
                f"Trained `GFMNano` copy accuracy: {model_acc:.4f}.",
                "",
                "**Stop condition:** "
                + (
                    "FIRED. The RF-limited CNN approached the trained model without source access; demote the branch."
                    if fired
                    else "not fired. The RF-limited CNN remains far below the trained model without source access."
                ),
                "",
                f"Raw data: `{csv_path}`. Figure: `{png_path}`.",
                "",
            ]
        )
    )
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("branches/codex-repeat-induction"))
    ap.add_argument("--train-seqs", type=int, default=20_000)
    ap.add_argument("--val-seqs", type=int, default=6_000)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--crop", type=int, default=33)
    ap.add_argument("--max-train", type=int, default=240_000)
    ap.add_argument("--max-val", type=int, default=60_000)
    ap.add_argument("--cnn-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--model-acc", type=float, default=0.9608)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = ap.parse_args()

    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"

    train = make_batch(args.train_seqs, window=args.window, seed=args.seed)
    val = make_batch(args.val_seqs, window=args.window, seed=args.seed + 1)
    _, y_train_full, m_train = ar_inputs_targets(train)
    _, y_val_full, m_val = ar_inputs_targets(val)
    x_train, y_train = local_examples(train.tokens, y_train_full, m_train, args.crop)
    x_val, y_val = local_examples(val.tokens, y_val_full, m_val, args.crop)
    x_train, y_train = subsample(x_train, y_train, args.max_train, args.seed + 2)
    x_val, y_val = subsample(x_val, y_val, args.max_val, args.seed + 3)
    print(f"local examples train {x_train.shape} val {x_val.shape} device {device}")

    base = base_rate(y_train, y_val)
    nb = local_naive_bayes(x_train, y_train, x_val, y_val)
    cnn, cnn_hist = train_cnn(x_train, y_train, x_val, y_val, args.cnn_epochs, args.batch, args.seed, device)

    rows = [
        {"name": "base-rate", "accuracy": f"{base:.6f}", "sees_source": "no", "notes": "best constant base on copy positions"},
        {"name": "local naive Bayes", "accuracy": f"{nb:.6f}", "sees_source": "no", "notes": f"position-specific NB over {args.crop} nt local crop"},
        {"name": "RF-limited CNN", "accuracy": f"{cnn:.6f}", "sees_source": "no", "notes": f"CNN over {args.crop} nt local crop; source denied"},
        {"name": "source oracle", "accuracy": "1.000000", "sees_source": "yes", "notes": "upper reference given copied source base"},
        {"name": "GFMNano copy circuit", "accuracy": f"{args.model_acc:.6f}", "sees_source": "yes", "notes": "trained transformer, seed 0 checkpoint"},
    ]
    write_outputs(args.out_dir, rows, cnn_hist, args.crop, args.model_acc)


if __name__ == "__main__":
    main()
