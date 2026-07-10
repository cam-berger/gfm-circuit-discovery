"""Synthetic coding sequence where the frame is the ONLY route to the label.

D24 showed GFM-nano never builds an anchored frame counter on real DNA, and D25 explains why: real
codon usage is biased, so phase is inferable from a ~30 nt bag, and an anchored counter buys 0.4% of
the loss. No self-supervised objective can change that -- any phase-dependent structure repeats every
three bases and is therefore locally detectable.

So we remove the shortcut and supervise a readout that requires the frame:

    sequence   5' region with NO ATG | ATG | codons drawn uniformly from all 64 codons
    label      the amino acid (or stop) of the codon containing each position

All 64 codons, not the 61 sense codons: the three stops are all T-initial, so excluding them depletes
T at codon position 0 and leaves a residual phase cue. With all 64 the sequence is exactly i.i.d.
uniform over {A,C,G,T} and local statistics carry *zero* phase information -- verified in
`tests/test_uniform_cds.py`. Emitting the right amino acid requires knowing which triplet is in frame, and the only
event that fixes the frame is the start codon. Guessing the alignment caps accuracy near 1/3; knowing
it gives 1.0.

The model is never told the frame. It is told the protein. That is Othello-GPT's structure, and it
makes the anchored counter the only solution rather than a marginal one -- the learned analogue of
Rung 0's planted compositional twin.
"""

from __future__ import annotations

import numpy as np

from gfmcd.data.genome import IGNORE
from gfmcd.paths import project_paths

BASES = "TCAG"
AA_TABLE = (
    "FFLLSSSSYY**CC*W"
    "LLLLPPPPHHQQRRRR"
    "IIIMTTTTNNKKSSRR"
    "VVVVAAAADDEEGGGG"
)
NT2ID = {"A": 0, "C": 1, "G": 2, "T": 3}
STOP = "*"

# Codon index in AA_TABLE is 16*i + 4*j + k over TCAG; we store codons as (a, b, c) in ACGT ids.
_CODONS: list[tuple[int, int, int]] = []
_AA: list[str] = []
for i, b1 in enumerate(BASES):
    for j, b2 in enumerate(BASES):
        for k, b3 in enumerate(BASES):
            _CODONS.append((NT2ID[b1], NT2ID[b2], NT2ID[b3]))
            _AA.append(AA_TABLE[16 * i + 4 * j + k])

CODONS = np.array(_CODONS, dtype=np.int8)  # [64, 3] -- all of them, stops included
AMINO_ACIDS = sorted(set(_AA))
AA2ID = {a: i for i, a in enumerate(AMINO_ACIDS)}
CODON_AA = np.array([AA2ID[a] for a in _AA], dtype=np.int64)  # [64]
N_AA = len(AMINO_ACIDS)  # 21: twenty amino acids plus stop

ATG = np.array([0, 3, 2], dtype=np.int8)


def _utr_without_atg(length: int, rng: np.random.Generator) -> np.ndarray:
    """A 5' region containing no ATG, so the start codon is the unique anchor."""
    while True:
        u = rng.integers(0, 4, size=length).astype(np.int8)
        if length < 3:
            return u
        hits = (u[:-2] == 0) & (u[1:-1] == 3) & (u[2:] == 2)
        if not hits.any():
            return u


def make_batch(
    n_seq: int,
    window: int = 256,
    utr_range: tuple[int, int] = (6, 40),
    seed: int = 0,
    codon_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (tokens [n, w], amino-acid labels [n, w], phase labels [n, w]).

    Phase labels are for probing only and never enter any training objective.
    """
    rng = np.random.default_rng(seed)
    tokens = np.zeros((n_seq, window), dtype=np.int8)
    aa = np.full((n_seq, window), IGNORE, dtype=np.int64)
    phase = np.full((n_seq, window), IGNORE, dtype=np.int64)

    lo, hi = utr_range
    atg_codon = int(codon_index(ATG.astype(np.int64)))

    for s in range(n_seq):
        u = int(rng.integers(lo, hi + 1))
        tokens[s, :u] = _utr_without_atg(u, rng)

        n_cod = (window - u) // 3
        if codon_probs is None:
            idx = rng.integers(0, len(CODONS), size=n_cod)
        else:
            idx = rng.choice(len(CODONS), size=n_cod, p=codon_probs)
        idx[0] = atg_codon  # the start codon IS the first codon, not a prefix that gets overwritten
        body = CODONS[idx].reshape(-1)
        end = u + 3 * n_cod
        tokens[s, u:end] = body[: end - u]
        tokens[s, end:] = rng.integers(0, 4, size=window - end)

        pos = np.arange(u, end)
        phase[s, u:end] = (pos - u) % 3
        aa[s, u:end] = np.repeat(CODON_AA[idx], 3)[: end - u]

    return tokens, aa, phase


def codon_index(triplet: np.ndarray) -> np.ndarray:
    """Map (a, b, c) in ACGT ids to the row of CODONS, vectorised over a leading axis."""
    key = CODONS[:, 0] * 16 + CODONS[:, 1] * 4 + CODONS[:, 2]
    lookup = np.zeros(64, dtype=np.int64)
    lookup[key] = np.arange(64)
    return lookup[triplet[..., 0] * 16 + triplet[..., 1] * 4 + triplet[..., 2]]


def alignment_free_ceiling(seed: int = 0, n: int = 40000) -> float:
    """The best a model can do without knowing the frame.

    It sees the local sequence but not the alignment. Optimally it marginalises over the three
    possible alignments and predicts the most likely amino acid. Everything above this number is
    evidence that the model located the start codon and counted.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 64, size=(n, 3))
    stream = CODONS[idx].reshape(n, 9)
    true_aa = CODON_AA[idx[:, 1]]

    votes = np.zeros((n, N_AA))
    for shift in (-1, 0, 1):
        aa = CODON_AA[codon_index(stream[:, 3 + shift : 6 + shift])]
        votes[np.arange(n), aa] += 1
    return float((votes.argmax(1) == true_aa).mean())


# ---------------------------------------------------------------------------
# The lambda sweep: a dial on how much the local shortcut helps.
#
# D26 showed the circuit appears when the objective makes it necessary and not when it is merely
# useful. That is a claim about a threshold, so put a continuous knob on it. Interpolate codon usage
# between uniform (lambda = 0, no local phase cue, circuit mandatory) and real human usage
# (lambda = 1, phase inferable from a ~30 nt bag, circuit worth 0.4%).
#
#     p_lambda = (1 - lambda) * uniform + lambda * p_real
#
# The task never changes -- predict the amino acid of the containing codon. Only the strength of the
# shortcut moves. Then ask, per lambda, whether the anchored counter forms.
# ---------------------------------------------------------------------------

UNIFORM_CODON_PROBS = np.full(64, 1 / 64)


def real_codon_probs(cache: "Path | None" = None) -> np.ndarray:
    """Human codon usage, counted from in-frame codons of GENCODE CDS exons in hg38."""
    from pathlib import Path

    import numpy as _np

    cache = Path(cache) if cache else project_paths().cache / "codon_usage_hg38.npy"
    if cache.exists():
        return _np.load(cache)

    from pyfaidx import Fasta

    from gfmcd.data.genome import SPLIT_CHROMS, Split, parse_gencode_cds, sample_windows

    data = project_paths().references
    fasta = Fasta(str(data / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(data / "gencode.v50.basic.annotation.gtf.gz", chroms=SPLIT_CHROMS[Split.TRAIN], min_len=150)
    tokens, labels = sample_windows(cds, fasta, n=20000, window=256, seed=3)

    counts = _np.ones(64)
    for tok, lab in zip(tokens, labels):
        starts = _np.where(lab == 0)[0]
        starts = starts[starts + 2 < len(tok)]
        tri = _np.stack([tok[starts], tok[starts + 1], tok[starts + 2]], axis=-1).astype(_np.int64)
        good = (tri < 4).all(-1)
        _np.add.at(counts, codon_index(tri[good]), 1)

    probs = counts / counts.sum()
    cache.parent.mkdir(parents=True, exist_ok=True)
    _np.save(cache, probs)
    return probs


def blended_codon_probs(lam: float) -> np.ndarray:
    """A dial on the local shortcut.

    lambda = 0    uniform: the sequence is i.i.d., phase is not locally inferable at all
    lambda = 1    real human codon usage
    lambda > 1    sharpened (p ∝ p_real^lambda): mixture alone reaches only ~64% of real hg38's
                  I(nt;phase), because real DNA also carries inter-codon and amino-acid structure

    Only the shortcut moves. The task -- predict the amino acid of the containing codon -- is fixed.
    """
    real = real_codon_probs()
    p = (1 - lam) * UNIFORM_CODON_PROBS + lam * real if lam <= 1 else np.power(real, lam)
    return p / p.sum()


def nucleotide_phase_information(tokens: np.ndarray, phase: np.ndarray, skip_start: bool = True) -> float:
    """I(nucleotide ; phase) in nats -- the strength of the single-base shortcut."""
    if skip_start:
        body = np.zeros_like(phase, dtype=bool)
        for r, ph in enumerate(phase):
            body[r, int(np.argmax(ph == 0)) + 3 :] = True
        m = (phase >= 0) & body
    else:
        m = phase >= 0

    def H(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    h = H(np.bincount(tokens[m].astype(int), minlength=4))
    h_cond = sum(
        (m & (phase == k)).sum() / m.sum() * H(np.bincount(tokens[m & (phase == k)].astype(int), minlength=4))
        for k in range(3)
    )
    return h - h_cond


# ---------------------------------------------------------------------------
# Cell B of the necessity 2x2: no shortcut, hard detector.
#
# Uniform codons carry no ORF-integrity cue -- stops occur at rate 3/64 in EVERY frame, the true one
# included -- so a planted Kozak-like motif is the only thing distinguishing the true start from a
# decoy. Its strength is calibrated so the Bayes-optimal detector matches real hg38's measured frame
# accuracy of 0.9215 (D28). That makes B a matched-difficulty control rather than an arbitrary one.
#
# Decoy ATGs are injected into the 5' region at real hg38's rate: 42.9% of anchored windows carry at
# least one upstream ATG, mean 0.60 per window.
# ---------------------------------------------------------------------------

PURINES = (0, 2)  # A, G
KOZAK_MINUS3 = PURINES
KOZAK_PLUS3 = 2  # G, the base immediately after ATG
# A uniform 5' region of length L already contains (L-2)/64 ATGs in expectation, so no injection is
# needed: UTR length alone tunes the decoy rate. Injecting on top double-counts (measured 0.90 per
# window against a target of 0.60). Kept as an optional knob, defaulting off.
DECOY_RATE = 0.0
UTR_FOR_REAL_DECOY_RATE = (24, 56)  # mean 40 -> (40-2)/64 = 0.59 decoys, P(>=1) ~ 0.44

# Calibrated so the Bayes-optimal Kozak-only detector reaches real hg38's measured frame accuracy of
# 0.9215 (D28). Bracketed: kozak_p 0.90 -> 0.9045, 0.93 -> 0.9345.
KOZAK_P_MATCHED = 0.92


def _inject_decoys(utr: np.ndarray, n_decoy: int, rng: np.random.Generator) -> np.ndarray:
    """Place `n_decoy` ATGs at random non-overlapping positions in the 5' region."""
    out = utr.copy()
    if len(out) < 3 or n_decoy == 0:
        return out
    slots = list(range(len(out) - 2))
    rng.shuffle(slots)
    placed, used = 0, set()
    for p in slots:
        if placed >= n_decoy:
            break
        if any(q in used for q in (p, p + 1, p + 2)):
            continue
        out[p : p + 3] = ATG
        used.update({p, p + 1, p + 2})
        placed += 1
    return out


def make_batch_hard_detector(
    n_seq: int,
    window: int = 256,
    utr_range: tuple[int, int] = UTR_FOR_REAL_DECOY_RATE,
    seed: int = 0,
    kozak_p: float = KOZAK_P_MATCHED,
    decoy_rate: float = DECOY_RATE,
    codon_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell B. Returns (tokens, amino-acid labels, phase labels).

    Amino-acid labels are derived from the FINAL tokens, after the Kozak edit, so the label is always
    a function of the sequence given the frame -- never of the frame alone.
    """
    from gfmcd.data.genome import amino_acid_labels

    rng = np.random.default_rng(seed)
    tokens = np.zeros((n_seq, window), dtype=np.int8)
    phase = np.full((n_seq, window), IGNORE, dtype=np.int64)
    atg_codon = int(codon_index(ATG.astype(np.int64)))
    lo, hi = utr_range

    for s in range(n_seq):
        u = int(rng.integers(lo, hi + 1))
        utr = rng.integers(0, 4, size=u).astype(np.int8)
        if decoy_rate > 0:
            utr = _inject_decoys(utr, int(rng.poisson(decoy_rate)), rng)
        tokens[s, :u] = utr

        n_cod = (window - u) // 3
        idx = rng.integers(0, len(CODONS), size=n_cod) if codon_probs is None else rng.choice(
            len(CODONS), size=n_cod, p=codon_probs
        )
        idx[0] = atg_codon
        body = CODONS[idx].reshape(-1)
        end = u + 3 * n_cod
        tokens[s, u:end] = body[: end - u]
        tokens[s, end:] = rng.integers(0, 4, size=window - end)

        if rng.random() < kozak_p:
            if u >= 3:
                tokens[s, u - 3] = PURINES[int(rng.integers(0, 2))]
            if u + 3 < window:
                tokens[s, u + 3] = KOZAK_PLUS3

        phase[s, u:end] = (np.arange(u, end) - u) % 3

    return tokens, amino_acid_labels(tokens, phase), phase


def bayes_detector_frame_accuracy(tokens: np.ndarray, phase: np.ndarray) -> tuple[float, float]:
    """(exact, frame) accuracy of the optimal Kozak-only detector on cell-B sequence.

    Scores every ATG by its Kozak match and takes the best; ties broken by earliest. An in-frame decoy
    is harmless, so `frame` is the number that governs the circuit.
    """
    w = tokens.shape[1]
    exact = frame = 0
    for r in range(len(tokens)):
        t = tokens[r]
        true = int(np.argmax(phase[r] == 0))
        cands = [p for p in range(w - 2) if (t[p], t[p + 1], t[p + 2]) == tuple(ATG)]
        if not cands:
            continue
        best, best_score = cands[0], -1
        for p in cands:
            score = 0
            if p - 3 >= 0 and t[p - 3] in KOZAK_MINUS3:
                score += 1
            if p + 3 < w and t[p + 3] == KOZAK_PLUS3:
                score += 1
            if score > best_score:
                best, best_score = p, score
        exact += best == true
        frame += (best - true) % 3 == 0
    return exact / len(tokens), frame / len(tokens)


def calibrate_kozak(target_frame_acc: float = 0.9215, n: int = 1500, seed: int = 0) -> float:
    """Find the Kozak strength that reproduces real hg38's detector difficulty."""
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        _, acc = bayes_detector_frame_accuracy(*[a for a in make_batch_hard_detector(n, seed=seed, kozak_p=mid)[::2]])
        if acc < target_frame_acc:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
