"""Sequence nulls: preserve the statistic, destroy the grammar.

This is the null family LLM interpretability cannot construct. We can synthesise a
sequence with the same dinucleotide composition as a real one, or the same G-density as
a real G-quadruplex, and ask whether the circuit still fires. If it does, the circuit is
reading composition rather than structure, and "no circuit found" becomes a meaningful
result rather than an absence of evidence.

`dinuc_shuffle` uses the Altschul-Erikson algorithm: uniformly sample an Eulerian path
through the dinucleotide graph. A naive within-sequence shuffle destroys dinucleotide
frequencies, which for DNA is a large part of the signal.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict


def _last_edge_tree_is_valid(last_edge: dict[str, str], terminal: str) -> bool:
    for v in last_edge:
        seen, cur = set(), v
        while cur != terminal:
            if cur in seen or cur not in last_edge:
                return False
            seen.add(cur)
            cur = last_edge[cur]
    return True


def dinuc_shuffle(seq: str, rng: random.Random | None = None, max_tries: int = 100) -> str:
    """Shuffle preserving exact dinucleotide counts, first base, and last base."""
    rng = rng or random.Random()
    if len(seq) < 3:
        return seq

    terminal = seq[-1]
    edges: dict[str, list[str]] = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)

    for _ in range(max_tries):
        last_edge = {v: rng.choice(succ) for v, succ in edges.items() if v != terminal}
        if _last_edge_tree_is_valid(last_edge, terminal):
            break
    else:
        raise RuntimeError(f"no Eulerian path found in {max_tries} tries")

    ordered: dict[str, list[str]] = {}
    for v, succ in edges.items():
        rest = list(succ)
        if v in last_edge:
            rest.remove(last_edge[v])
        rng.shuffle(rest)
        ordered[v] = rest + ([last_edge[v]] if v in last_edge else [])

    out, cur, cursor = [seq[0]], seq[0], defaultdict(int)
    for _ in range(len(seq) - 1):
        nxt = ordered[cur][cursor[cur]]
        cursor[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def dinuc_counts(seq: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for a, b in zip(seq, seq[1:]):
        counts[a + b] += 1
    return dict(counts)


G4_MOTIF = re.compile(r"G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}")
G4_MOTIF_RC = re.compile(r"C{3,}\w{1,7}C{3,}\w{1,7}C{3,}\w{1,7}C{3,}")


def has_g4_motif(seq: str) -> bool:
    """Canonical loop-length grammar, checked on both strands."""
    return bool(G4_MOTIF.search(seq) or G4_MOTIF_RC.search(seq))


def g_density_matched(seq: str, rng: random.Random | None = None, max_tries: int = 1000) -> str:
    """A decoy with identical base composition that carries no canonical G4 grammar.

    The Rung 3 null of record: defeats a bag-of-motifs G-counter, not a G4 detector.

    Composition matching alone is insufficient — a plain shuffle preserves G-density but
    can *assemble* a G4 by chance, and such a decoy is a positive, not a null. So we
    reject-sample until the shuffle is G4-negative on both strands.
    """
    rng = rng or random.Random()
    chars = list(seq)
    for _ in range(max_tries):
        rng.shuffle(chars)
        candidate = "".join(chars)
        if not has_g4_motif(candidate):
            return candidate
    raise RuntimeError(f"no G4-negative shuffle found in {max_tries} tries; G-density may be too high")
