"""Phase D: test the repeat-copy circuit on natural hg38 repeats.

This is an external-validity assay for the synthetic repeat-copy circuit. It samples real 256 nt
hg38 windows, finds naturally occurring exact repeated `(prefix + continuation)` strings, and asks
whether the synthetic-trained model copies from the earlier occurrence under the same cue conflict.

No `pyfaidx` dependency: this script reads `hg38.fa.fai` directly.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from gfmcd.paths import project_paths

import numpy as np
import torch

from gfmcd.data.repeat_copy import ID2NT, IGNORE, NT2ID, RepeatCopyBatch, ar_inputs_targets, base_rate_ceiling, single_conflict_batch
from gfmcd.figstyle import BLUE, ORANGE, RED, apply_figure_style
from gfmcd.models.gfm_nano import GFMNano, NanoConfig


@dataclass(frozen=True)
class FaiEntry:
    length: int
    offset: int
    line_bases: int
    line_width: int


class IndexedFasta:
    def __init__(self, path: Path):
        self.path = path
        self.index: dict[str, FaiEntry] = {}
        for line in path.with_suffix(path.suffix + ".fai").read_text().splitlines():
            chrom, length, offset, line_bases, line_width = line.split("\t")[:5]
            self.index[chrom] = FaiEntry(int(length), int(offset), int(line_bases), int(line_width))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        entry = self.index[chrom]
        if start < 0 or end > entry.length or start >= end:
            raise ValueError((chrom, start, end))
        pieces: list[str] = []
        with self.path.open("rb") as fh:
            pos = start
            while pos < end:
                line_no = pos // entry.line_bases
                in_line = pos % entry.line_bases
                take = min(end - pos, entry.line_bases - in_line)
                fh.seek(entry.offset + line_no * entry.line_width + in_line)
                pieces.append(fh.read(take).decode("ascii"))
                pos += take
        return "".join(pieces).upper()


def encode(seq: str) -> np.ndarray:
    return np.fromiter((NT2ID.get(c, 4) for c in seq), dtype=np.int8, count=len(seq))


def diverse(kmer: str, min_bases: int, max_base_frac: float) -> bool:
    counts = Counter(kmer)
    return len(counts) >= min_bases and max(counts.values()) / len(kmer) <= max_base_frac


def find_repeat(seq: str, prefix: int, min_dist: int, max_dist: int, min_bases: int, max_base_frac: float):
    """Return `(source_continuation, pred_pos, kmer)` for one exact natural repeat."""

    if "N" in seq:
        return None
    seen: dict[str, list[int]] = {}
    for target in range(prefix, len(seq)):
        kmer = seq[target - prefix : target + 1]
        if diverse(kmer, min_bases, max_base_frac):
            for src in seen.get(kmer, []):
                pred_pos = target - 1
                dist = pred_pos - src
                if min_dist <= dist <= max_dist:
                    return src, pred_pos, kmer
        seen.setdefault(kmer, []).append(target)
    return None


def sample_natural_repeats(
    fasta: IndexedFasta,
    n: int,
    chroms: list[str],
    window: int,
    prefix: int,
    min_dist: int,
    max_dist: int,
    min_bases: int,
    max_base_frac: float,
    seed: int,
) -> tuple[RepeatCopyBatch, list[dict]]:
    rng = np.random.default_rng(seed)
    tokens: list[np.ndarray] = []
    first_start: list[int] = []
    second_start: list[int] = []
    unit_len: list[int] = []
    metadata: list[dict] = []

    tries = 0
    max_tries = max(1000, n * 200)
    while len(tokens) < n and tries < max_tries:
        tries += 1
        chrom = str(rng.choice(chroms))
        chrom_len = fasta.index[chrom].length
        start = int(rng.integers(0, chrom_len - window))
        seq = fasta.fetch(chrom, start, start + window)
        found = find_repeat(seq, prefix, min_dist, max_dist, min_bases, max_base_frac)
        if found is None:
            continue
        src, pred_pos, kmer = found
        tok = encode(seq)
        if (tok >= 4).any():
            continue
        tokens.append(tok)
        first_start.append(src - prefix)
        second_start.append(pred_pos - prefix + 1)
        unit_len.append(prefix + 1)
        metadata.append(
            {
                "chrom": chrom,
                "window_start": start,
                "source_continuation": src,
                "prediction_position": pred_pos,
                "distance": pred_pos - src,
                "repeat_kmer": kmer,
                "local_target": seq[pred_pos + 1],
            }
        )

    if len(tokens) < n:
        raise RuntimeError(f"sampled only {len(tokens)}/{n} natural repeats in {tries} tries")

    tok_arr = np.stack(tokens)
    copy_mask = np.zeros((n, window - 1), dtype=bool)
    source_index = np.full((n, window - 1), IGNORE, dtype=np.int64)
    for r, meta in enumerate(metadata):
        p = int(meta["prediction_position"])
        copy_mask[r, p] = True
        source_index[r, p] = int(meta["source_continuation"])

    return (
        RepeatCopyBatch(
            tok_arr,
            copy_mask,
            source_index,
            np.array(first_start, dtype=np.int64),
            np.array(second_start, dtype=np.int64),
            np.array(unit_len, dtype=np.int64),
        ),
        metadata,
    )


@torch.no_grad()
def logits_for(model: GFMNano, tokens: np.ndarray, batch: int, device: str) -> np.ndarray:
    x = torch.from_numpy(tokens[:, :-1]).long()
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i : i + batch].to(device)).cpu())
    return torch.cat(out).numpy()


def shares(pred: np.ndarray, distal: np.ndarray, local: np.ndarray, mask: np.ndarray) -> dict:
    distal_share = float((pred[mask] == distal[mask]).mean())
    local_share = float((pred[mask] == local[mask]).mean())
    return {"distal": distal_share, "local": local_share, "neither": 1.0 - distal_share - local_share}


def prefix_scramble(tokens: np.ndarray, data: RepeatCopyBatch, score_mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = tokens.copy()
    rows, pred_positions = np.nonzero(score_mask)
    for r, p in zip(rows, pred_positions):
        src = int(data.source_index[r, p])
        start = int(data.first_start[r])
        for q in range(start, src):
            out[r, q] = (out[r, q] + int(rng.integers(1, 4))) % 4
    return out


@torch.no_grad()
def attention_to_source(model: GFMNano, tokens: np.ndarray, data: RepeatCopyBatch, heads: list[tuple[int, int]], batch_n: int, device: str) -> dict[str, float]:
    x = torch.from_numpy(tokens[:batch_n, :-1]).long().to(device)
    _, cache = model.run_with_cache(x)
    rows, pred_positions = np.nonzero(data.copy_mask[:batch_n])
    out = {}
    for layer, head in heads:
        pattern = cache[f"L{layer}.attn.pattern"].cpu().numpy()
        vals = [pattern[r, head, p, int(data.source_index[r, p])] for r, p in zip(rows, pred_positions)]
        out[f"L{layer}H{head}"] = float(np.mean(vals))
    return out


def dna(arr: np.ndarray) -> str:
    return "".join(ID2NT[int(x)] for x in arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/repeat_copy_s0_step8000.pt"))
    ap.add_argument("--fasta", type=Path, default=project_paths().hg38())
    ap.add_argument("--out-dir", type=Path, default=project_paths().results / "repeat-copy")
    ap.add_argument("--seqs", type=int, default=1000)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--prefix", type=int, default=12)
    ap.add_argument("--min-dist", type=int, default=48)
    ap.add_argument("--max-dist", type=int, default=160)
    ap.add_argument("--chroms", default="chr17,chr18,chr19")
    ap.add_argument("--min-bases", type=int, default=3)
    ap.add_argument("--max-base-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=9090)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fasta = IndexedFasta(args.fasta)
    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    data, meta = sample_natural_repeats(
        fasta,
        args.seqs,
        chroms,
        args.window,
        args.prefix,
        args.min_dist,
        args.max_dist,
        args.min_bases,
        args.max_base_frac,
        args.seed,
    )

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    _, targets, copy_mask = ar_inputs_targets(data)
    logits = logits_for(model, data.tokens, args.batch, device)
    pred = logits.argmax(axis=-1)
    intact = float((pred[copy_mask] == targets[copy_mask]).mean())

    conflict_tokens, distal, local, score_mask = single_conflict_batch(data, seed=args.seed + 1)
    conflict_pred = logits_for(model, conflict_tokens, args.batch, device).argmax(axis=-1)
    conflict = shares(conflict_pred, distal, local, score_mask)
    scrambled = prefix_scramble(conflict_tokens, data, score_mask, seed=args.seed + 2)
    scrambled_pred = logits_for(model, scrambled, args.batch, device).argmax(axis=-1)
    scrambled_shares = shares(scrambled_pred, distal, local, score_mask)

    heads = [(1, 7), (1, 0), (1, 5), (1, 1), (1, 6)]
    attn = attention_to_source(model, conflict_tokens, data, heads, min(args.seqs, 256), device)
    attn_scr = attention_to_source(model, scrambled, data, heads, min(args.seqs, 256), device)

    out = args.out_dir
    fig_data = out / "fig_data"
    fig_data.mkdir(parents=True, exist_ok=True)
    examples_csv = fig_data / "natural_repeat_examples.csv"
    with examples_csv.open("w", newline="") as fh:
        fieldnames = list(meta[0].keys()) + ["model_pred", "distal_target", "conflict_pred"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        rows, ps = np.nonzero(score_mask)
        for r, p in zip(rows, ps):
            row = dict(meta[r])
            row["model_pred"] = ID2NT[int(pred[r, p])]
            row["distal_target"] = ID2NT[int(distal[r, p])]
            row["conflict_pred"] = ID2NT[int(conflict_pred[r, p])]
            writer.writerow(row)

    summary_csv = fig_data / "natural_repeat_summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in {
            "n_windows": args.seqs,
            "base_rate": base_rate_ceiling(data),
            "intact_copy_acc": intact,
            "conflict_distal": conflict["distal"],
            "conflict_local": conflict["local"],
            "conflict_neither": conflict["neither"],
            "prefix_scramble_distal": scrambled_shares["distal"],
            "prefix_scramble_local": scrambled_shares["local"],
            "prefix_scramble_neither": scrambled_shares["neither"],
        }.items():
            writer.writerow({"metric": k, "value": v})

    attn_csv = fig_data / "natural_repeat_attention.csv"
    with attn_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["head", "attn_conflict", "attn_prefix_scrambled"])
        writer.writeheader()
        for h in attn:
            writer.writerow({"head": h, "attn_conflict": attn[h], "attn_prefix_scrambled": attn_scr[h]})

    import matplotlib.pyplot as plt

    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].bar(["base", "intact", "distal", "local"], [base_rate_ceiling(data), intact, conflict["distal"], conflict["local"]], color=[ORANGE, BLUE, BLUE, RED])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy / share")
    axes[0].set_title("Natural hg38 repeats")
    labels = list(attn)
    axes[1].bar(labels, [attn[h] for h in labels], color=BLUE)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("L1 attention to source")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig_path = out / "fig_natural_repeats.png"
    fig.savefig(fig_path)

    # A representative distal-following example.
    rows, ps = np.nonzero(score_mask)
    chosen = None
    for r, p in zip(rows, ps):
        if conflict_pred[r, p] == distal[r, p]:
            chosen = int(r), int(p)
            break
    if chosen is None:
        chosen = int(rows[0]), int(ps[0])
    r, p = chosen
    src = int(data.source_index[r, p])
    first = int(data.first_start[r])
    second = int(data.second_start[r])
    unit_len = int(data.unit_len[r])

    md = out / "NATURAL_REPEATS.md"
    md.write_text(
        "\n".join(
            [
                "# Natural hg38 Repeat Generalisation",
                "",
                f"Checkpoint: `{args.checkpoint}`. Chromosomes: {', '.join(chroms)}. Windows: {args.seqs}.",
                f"Repeat criterion: exact `{args.prefix + 1}`-mer, source distance {args.min_dist}-{args.max_dist} nt, "
                f"at least {args.min_bases} bases and max base fraction <= {args.max_base_frac}.",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| base-rate ceiling | {base_rate_ceiling(data):.4f} |",
                f"| intact copy accuracy | {intact:.4f} |",
                f"| conflict follows distal edit | {conflict['distal']:.4f} |",
                f"| conflict follows local continuation | {conflict['local']:.4f} |",
                f"| conflict neither | {conflict['neither']:.4f} |",
                f"| prefix-scrambled follows distal edit | {scrambled_shares['distal']:.4f} |",
                f"| prefix-scrambled follows local continuation | {scrambled_shares['local']:.4f} |",
                "",
                "## Source Attention",
                "",
                "| Head | Conflict | Prefix Scrambled |",
                "|---|---:|---:|",
                *[f"| {h} | {attn[h]:.4f} | {attn_scr[h]:.4f} |" for h in attn],
                "",
                "## Example",
                "",
                f"Metadata: `{meta[r]}`.",
                "",
                "```text",
                f"source repeat       {dna(data.tokens[r, first:first + unit_len])}",
                f"second repeat       {dna(data.tokens[r, second:second + unit_len])}",
                f"source conflicted   {dna(conflict_tokens[r, first:first + unit_len])}",
                f"local target        {ID2NT[int(local[r, p])]}",
                f"distal target       {ID2NT[int(distal[r, p])]}",
                f"model intact pred   {ID2NT[int(pred[r, p])]}",
                f"model conflict pred {ID2NT[int(conflict_pred[r, p])]}",
                "```",
                "",
                f"Raw data: `{summary_csv}`, `{examples_csv}`, `{attn_csv}`. Figure: `{fig_path}`.",
                "",
            ]
        )
    )
    print(md.read_text())


if __name__ == "__main__":
    main()
