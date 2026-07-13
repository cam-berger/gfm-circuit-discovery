"""Audit: (1) positional degeneracy of the partner-selection task,
(2) yield + length confound of a within-window cross-pairing set,
(3) yield of the length-matched cross-window 2x2 assignment test.

Run from the repo root:  python scripts/splicebert/audit_partner_feasibility.py
Read-only. Writes nothing.
"""
from __future__ import annotations

import collections
import json
import sys

sys.path.insert(0, "src")

GTF = "data/external/gencode/v50/gencode.v50.basic.annotation.gtf.gz"
VAL = "data/processed/splicebert_partner_selection/v1/val.jsonl"
CONFLICTS = "colab_results/splicebert_run/baselines/val_local_conflicts.jsonl"
EXCLUDED = {"chrX", "chrY", "chrM", "MT"}


def check_positional_leak() -> None:
    """The true acceptor/donor sit at fixed window offsets in every example."""
    n = bad_a = bad_d = 0
    seqlens = collections.Counter()
    for line in open(VAL):
        e = json.loads(line)
        seq_len = len(e["sequence"])
        tp = e["pairs"][e["true_pair_index"]]
        bad_a += tp["acceptor_pos"] != 200
        bad_d += tp["donor_pos"] != seq_len - 200
        seqlens[seq_len] += 1
        n += 1
    print(f"[leak] val examples: {n}")
    print(f"[leak] true acceptor NOT at token 200 : {bad_a}")
    print(f"[leak] true donor NOT at len-200      : {bad_d}")
    print(f"[leak] distinct sequence lengths: {len(seqlens)} (seq_len == 400 + exon_len)")

    # Zero-biology scorer: uses only candidate token coordinates.
    hits = total = 0
    for line in open(CONFLICTS):
        e = json.loads(line)
        seq_len = len(e["sequence"])
        scores = [
            -(abs(p["acceptor_pos"] - 200) + abs(p["donor_pos"] - (seq_len - 200)))
            for p in e["pairs"]
        ]
        best = max(scores)
        arg = [i for i, s in enumerate(scores) if s == best]
        hits += len(arg) == 1 and arg[0] == e["true_pair_index"]
        total += 1
    print(f"[leak] position-only PMR on CONFLICT set: {hits}/{total} = {hits / total:.4f}")


def cross_pair_yield() -> None:
    """Within-window cross-pairing: yield, and the structural length confound."""
    from gfmcd.splicing.gtf import load_transcripts

    rows, seen = [], set()
    for t in load_transcripts(GTF, transcript_types=("protein_coding",)):
        chrom = t.chrom if t.chrom.startswith("chr") else "chr" + t.chrom
        if chrom in EXCLUDED:
            continue
        exons = t.ordered_exons()
        # Both exons must be internal so all four sites are real splice sites.
        for i in range(1, len(exons) - 2):
            e1, e2 = exons[i], exons[i + 1]
            intron = e2.start - e1.end if t.strand == "+" else e1.start - e2.end
            if intron <= 0:
                continue
            key = (e1.chrom, e1.start, e1.end, e2.start, e2.end, e1.strand)
            if key in seen:
                continue
            seen.add(key)
            rows.append((e1.length, intron, e2.length, chrom))

    print(f"\n[cross] consecutive internal-exon pairs (deduped): {len(rows)}")
    for flank in (50, 100, 150, 200):
        span_max = 1022 - 2 * flank  # SpliceBERT max nucleotide length
        ok = [r for r in rows if 20 <= r[0] <= 500 and 20 <= r[2] <= 500 and sum(r[:3]) <= span_max]
        val = sum(r[3] in ("chr17", "chr18") for r in ok)
        test = sum(r[3] in ("chr19", "chr20", "chr21", "chr22") for r in ok)
        print(
            f"[cross] flank={flank:3d} span<={span_max:4d}: n={len(ok):6d} "
            f"(train {len(ok) - val - test} / val {val} / test {test})"
        )

    ok = [r for r in rows if 20 <= r[0] <= 500 and 20 <= r[2] <= 500 and sum(r[:3]) <= 822]
    longer = sum(sum(r[:3]) > max(r[0], r[2]) for r in ok)
    print(
        f"[cross] cross-pair length > BOTH true exon lengths: {longer}/{len(ok)} "
        f"({100 * longer / len(ok):.1f}%)  <-- length prior alone separates them"
    )


def assignment_yield() -> None:
    """Length-matched cross-window 2x2 swap: the recommended decisive test."""
    by_len = collections.defaultdict(list)
    for line in open(VAL):
        e = json.loads(line)
        tp = e["pairs"][e["true_pair_index"]]
        seq = e["sequence"]
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        by_len[tp["donor_pos"] - tp["acceptor_pos"]].append((e["gene_id"], gc))

    def greedy(tol: float) -> int:
        pairs = 0
        for items in by_len.values():
            items = sorted(items, key=lambda x: x[1])
            used = [False] * len(items)
            for i in range(len(items)):
                if used[i]:
                    continue
                for j in range(i + 1, len(items)):
                    if not used[j] and abs(items[i][1] - items[j][1]) <= tol:
                        used[i] = used[j] = True
                        pairs += 1
                        break
        return pairs

    print()
    for tol in (1.0, 0.05, 0.02, 0.01):
        label = "no GC constraint" if tol >= 1 else f"GC within {tol:.2f}"
        print(f"[2x2] {label:18s}: {greedy(tol):5d} disjoint length-matched window pairs (val)")


if __name__ == "__main__":
    check_positional_leak()
    cross_pair_yield()
    assignment_yield()
