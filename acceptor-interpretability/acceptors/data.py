"""GUE `splice_reconstructed` loading + BP/PPT corruption + control construction.

All windows are 400nt with the splice-relevant dinucleotide fixed at index
200 by construction of the upstream dataset (label 0 = acceptor, AG at
[198,200); label 1 = donor, GT at [200,202); label 2 = non-splice negative).
We only use labels 0 and 2 here; label 1 is reserved for the later
shared-transport check (next-steps item 4).
"""

import random

import pandas as pd

BASES = "ACGU"


def load_gue_split(cfg) -> pd.DataFrame:
    url = cfg.data.base_url.format(
        dataset=cfg.data.dataset, config=cfg.data.config, split=cfg.data.split
    )
    df = pd.read_csv(url)
    return df[df["sequence"].str.len() == cfg.data.seq_len].reset_index(drop=True)


def acceptor_rows(df: pd.DataFrame, cfg) -> pd.DataFrame:
    ag_start, ag_len = cfg.coords.ag_start, cfg.coords.ag_len
    mask = df["label"] == cfg.data.acceptor_label
    df = df[mask]
    has_ag = df["sequence"].str[ag_start : ag_start + ag_len] == "AG"
    dropped = (~has_ag).sum()
    if dropped:
        print(f"[data] dropping {dropped}/{len(df)} acceptor rows without AG at "
              f"[{ag_start},{ag_start + ag_len}) (unexpected off-by-one upstream)")
    return df[has_ag].reset_index(drop=True)


def negative_rows(df: pd.DataFrame, cfg) -> pd.DataFrame:
    return df[df["label"] == cfg.data.negative_label].reset_index(drop=True)


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Corruption / control construction
# ---------------------------------------------------------------------------


def _no_ag_choices(prev_base):
    return "ACU" if prev_base == "A" else BASES


def random_no_ag(length: int, rng: random.Random, prev_base=None) -> str:
    """A random ACGU string with no 'AG' substring anywhere, including at the
    seam with `prev_base` (the real base immediately preceding this region)."""
    out = []
    prev = prev_base
    for _ in range(length):
        c = rng.choice(_no_ag_choices(prev))
        out.append(c)
        prev = c
    return "".join(out)


def _fix_right_boundary(region: str, next_base, rng: random.Random) -> str:
    """If region ends in 'A' and `next_base` is 'G', that seam is a decoy AG
    we didn't intend to inject - resample just the last character."""
    if next_base != "G" or region[-1] != "A":
        return region
    prev = region[-2] if len(region) > 1 else None
    choices = [b for b in _no_ag_choices(prev) if b != "A"]
    return region[:-1] + rng.choice(choices)


def corrupt_region(sequence: str, start: int, end: int, rng: random.Random) -> str:
    """Replace sequence[start:end] with content-destroying random ACGU
    (balanced, no injected AG substrings - including at both seams)."""
    prev_base = sequence[start - 1] if start > 0 else None
    next_base = sequence[end] if end < len(sequence) else None
    region = random_no_ag(end - start, rng, prev_base=prev_base)
    region = _fix_right_boundary(region, next_base, rng)
    return sequence[:start] + region + sequence[end:]


def random_ag_variant(sequence: str, ag_start: int, ag_len: int, rng: random.Random) -> str:
    """Whole-sequence content destroyed (no genuine tract/branch-point
    context anywhere) except a bare 'AG' forced back in at the true
    position - the 'context is required' control."""
    body = random_no_ag(len(sequence), rng)
    return body[:ag_start] + "AG" + body[ag_start + ag_len :]


def find_decoy_ag(sequence: str, search_from: int, search_to: int):
    """First 'AG' occurring downstream of the true site, or None."""
    end = min(search_to, len(sequence) - 1)
    for i in range(search_from, end):
        if sequence[i : i + 2] == "AG":
            return i
    return None


def shuffle_region_in_place(sequence: str, start: int, end: int, rng: random.Random) -> str:
    """Permute sequence[start:end] in place - exact same multiset of bases,
    same position and length, order scrambled. Composition-preserving,
    unlike `corrupt_region` (content-destroying). Used to isolate whether an
    effect is driven by base composition or by base order/motif identity."""
    region = list(sequence[start:end])
    rng.shuffle(region)
    return sequence[:start] + "".join(region) + sequence[end:]


# ---------------------------------------------------------------------------
# Branch-point / tract localization from sequence content, not fixed offsets
# ---------------------------------------------------------------------------

# Simplified, hand-set weights approximating the mammalian branch-point
# consensus "yUnAy" (Gao et al.) around the candidate branch-point adenosine
# at relative offset 0: y(-2) U(-1) A(0) n(+1) y(+2). This is a documented
# heuristic stand-in for a trained locator (branchpointer/LaBranchoR need
# either genomic coordinates GUE doesn't carry, or a separate pretrained
# model) - see README "Known approximations".
_PYRIMIDINES = set("CU")


def _bp_score(sequence: str, p: int) -> float:
    score = 0.0
    if p - 2 >= 0 and sequence[p - 2] in _PYRIMIDINES:
        score += 1.0
    if p - 1 >= 0:
        score += 2.0 if sequence[p - 1] == "U" else (0.5 if sequence[p - 1] == "C" else 0.0)
    if p + 2 < len(sequence) and sequence[p + 2] in _PYRIMIDINES:
        score += 1.0
    return score


def locate_branch_point(sequence: str, ag_start: int, search_from: int, search_to: int):
    """Score every 'A' in the branch-point search window [ag_start-search_from,
    ag_start-search_to) against the yUnAy consensus; return the best-scoring
    position (ties broken toward the position closest to the AG, i.e. the
    most-3' candidate - consistent with AG-scanning starting just past the
    tract). Returns None if no 'A' exists in the window (rare)."""
    lo, hi = ag_start - search_from, ag_start - search_to
    candidates = [p for p in range(max(0, lo), hi) if sequence[p] == "A"]
    if not candidates:
        return None
    scored = [(_bp_score(sequence, p), p) for p in candidates]
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[-1][1]


def locate_tract(sequence: str, bp_pos: int, ag_start: int, min_gap: int = 3):
    """The tract is the region between the branch point and the AG, leaving
    a small fixed gap (`min_gap`) immediately after the branch-point A for
    the branch-point-region corruption window to occupy without overlap."""
    start = min(bp_pos + min_gap, ag_start)
    return start, ag_start
