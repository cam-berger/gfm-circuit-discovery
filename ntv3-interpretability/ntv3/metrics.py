"""Sequence-level features: composition, k-mers, entropy, and canonical
promoter motifs (TATA/CAAT/GC-box/Initiator). Mirrors the feature set built
in ``dnabert/05_promoter_mlp_feature_characterization.ipynb``. Model-agnostic
-- shared verbatim with the (removed) NTv2 port.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

BASES = "ACGT"


def mononucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    arr = np.array(list(seq))
    rng.shuffle(arr)
    return "".join(arr)


def _try_dinuc_shuffle(seq: str, rng: np.random.Generator) -> str | None:
    """One attempt at a dinucleotide-preserving shuffle via random Eulerian
    walk. Returns None (rather than raising) if the random edge order leaves
    the walk stranded before consuming every edge -- the caller retries with
    a fresh shuffle. A full reachability-guaranteed construction (proper
    Altschul-Erikson 1985) is unnecessary here: these are dense 4-letter
    multigraphs (out-degree ~ n/4 for a ~70nt sequence), so a stranding retry
    succeeds within a handful of attempts in practice, and failure is
    reported loudly rather than silently producing a wrong-composition
    sequence."""

    chars = list(seq)
    edges: dict[str, list[int]] = {b: [] for b in set(chars)}
    for i in range(len(chars) - 1):
        edges[chars[i]].append(i + 1)
    for targets in edges.values():
        rng.shuffle(targets)

    pointers = {node: 0 for node in edges}
    result = [chars[0]]
    current = chars[0]
    for _ in range(len(chars) - 1):
        idx = pointers[current]
        if idx >= len(edges[current]):
            return None  # stranded: this random order has no valid Eulerian walk
        next_pos = edges[current][idx]
        pointers[current] += 1
        result.append(chars[next_pos])
        current = chars[next_pos]
    return "".join(result)


def dinucleotide_shuffle(seq: str, rng: np.random.Generator, *, max_attempts: int = 200) -> str:
    """Dinucleotide-composition-preserving shuffle (Altschul-Erikson 1985
    style random Eulerian walk over the base->base transition multigraph).
    Preserves exact dinucleotide (and hence mononucleotide) composition."""

    if len(seq) < 3:
        return seq
    for _ in range(max_attempts):
        out = _try_dinuc_shuffle(seq, rng)
        if out is not None and len(out) == len(seq):
            return out
    raise RuntimeError(
        f"dinucleotide shuffle failed to find a valid Eulerian walk in "
        f"{max_attempts} attempts for a sequence of length {len(seq)}"
    )


def circular_shift(seq: str, shift: int) -> str:
    shift = shift % len(seq)
    return seq[shift:] + seq[:shift]


def regional_shuffle(seq: str, start: int, end: int, rng: np.random.Generator) -> str:
    region = list(seq[start:end])
    rng.shuffle(region)
    return seq[:start] + "".join(region) + seq[end:]

TATA_MOTIFS = ["TATAAA", "TATAAT"]
CAAT_MOTIFS = ["CCAAT"]
GC_BOX_MOTIFS = ["GGGCGG", "CCGCCC"]
INITIATOR_RE = re.compile(r"[CT][CT]A[ACGT][AT][CT][CT]")


def composition_features(seq: str) -> dict:
    n = len(seq)
    counts = {b: seq.count(b) for b in BASES}
    feats = {f"frac_{b}": counts[b] / n for b in BASES}
    feats["gc_fraction"] = (counts["G"] + counts["C"]) / n
    return feats


def kmer_features(seqs: list[str], k: int) -> tuple[np.ndarray, list[str]]:
    """Exact k-mer frequency features (sliding window, normalized by window
    count), matching the DNABERT-2 notebook's k-mer baseline."""

    kmers = ["".join(p) for p in itertools.product(BASES, repeat=k)]
    index = {kmer: i for i, kmer in enumerate(kmers)}
    out = np.zeros((len(seqs), len(kmers)))
    for row, seq in enumerate(seqs):
        n_windows = len(seq) - k + 1
        if n_windows <= 0:
            continue
        for i in range(n_windows):
            window = seq[i : i + k]
            idx = index.get(window)
            if idx is not None:
                out[row, idx] += 1
        out[row] /= n_windows
    return out, kmers


def shannon_entropy(seq: str) -> float:
    n = len(seq)
    probs = [seq.count(b) / n for b in BASES if seq.count(b) > 0]
    return float(-sum(p * np.log2(p) for p in probs))


def longest_homopolymer(seq: str) -> int:
    best = 1
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def cpg_gpc_features(seq: str) -> dict:
    n = len(seq)
    cpg = seq.count("CG")
    gpc = seq.count("GC")
    g = seq.count("G")
    c = seq.count("C")
    expected_cpg = (g * c) / n if n else 0.0
    return {
        "cpg_count": cpg,
        "cpg_density": cpg / n,
        "gpc_count": gpc,
        "gpc_density": gpc / n,
        "cpg_obs_exp": (cpg / expected_cpg) if expected_cpg > 0 else 0.0,
    }


def _min_hamming(seq: str, motifs: list[str]) -> int:
    L = len(motifs[0])
    best = L
    for i in range(len(seq) - L + 1):
        window = seq[i : i + L]
        for motif in motifs:
            d = sum(a != b for a, b in zip(window, motif))
            best = min(best, d)
    return best


def motif_features(seq: str) -> dict:
    tata_present = any(m in seq for m in TATA_MOTIFS)
    caat_present = any(m in seq for m in CAAT_MOTIFS)
    gc_box_present = any(m in seq for m in GC_BOX_MOTIFS)
    initiator_present = bool(INITIATOR_RE.search(seq))
    return {
        "tata_present": int(tata_present),
        "tata_min_hamming": _min_hamming(seq, TATA_MOTIFS),
        "caat_present": int(caat_present),
        "gc_box_present": int(gc_box_present),
        "gc_box_min_hamming": _min_hamming(seq, GC_BOX_MOTIFS),
        "initiator_present": int(initiator_present),
    }


def full_feature_row(seq: str) -> dict:
    row = {}
    row.update(composition_features(seq))
    row["sequence_entropy"] = shannon_entropy(seq)
    row["homopolymer_run"] = longest_homopolymer(seq)
    row.update(cpg_gpc_features(seq))
    row.update(motif_features(seq))
    return row
