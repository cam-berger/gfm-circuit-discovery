"""Cross-site coupling: is splice-site recognition causally dependent on the PARTNER site?

WHY THIS REPLACES THE PARTNER-SELECTION TASK
--------------------------------------------
Two behavioural designs died before this one, both for the same underlying reason: a
*forced-choice* task can always be won without doing the computation.

  v1 (exon-pair selection).  The window was cut relative to the label, so the true
    acceptor sat at a constant position (200) in 100% of examples and a model reading
    ZERO nucleotides scored PMR 1.000.  Decoys were spurious AG/GT motifs, so an oracle
    site detector also scored 1.000.  The task never required partner information.

  v2 (real-site decoys).  Decoys were real annotated donors the acceptor never pairs
    with -- but they turned out to be *minor* sites (median 1 supporting isoform vs 18
    for true donors).  So "pick the partner" collapsed into "pick the better donor": a
    plain PWM scored 0.687 vs 0.392 chance, and the bilinear readout beat the additive
    control by only +0.008 -- i.e. the acceptor contributed nothing.

The deeper lesson is biological, not technical: in a real genome, a *strong* donor at a
*plausible* exon distance from an acceptor generally IS its partner.  Clean negatives
barely exist, so partner identity is nearly determined by site strength + distance and
there is almost no residual for a behavioural task to expose.

THIS MODULE ASKS A DIFFERENT QUESTION -- one with no task to win, hence nothing to leak:

    When the model recognises a 5' splice site, is that computation READING the 3' site?

Readout is the model's NATIVE masked-LM output (no probe, no readout head, no labels):
mask one nucleotide at the donor and take logit(G) - max(logit of the other bases).
Perturbations SUBSTITUTE real bases -- never ``[MASK]``, which is off-manifold and would
make the model react to the mask token rather than to the loss of the motif.

Evidence ladder (a drop alone proves only NON-LOCAL dependence, never exon definition):
  1. dependence          -- perturbing the partner degrades donor recognition
  2. partner-selectivity -- it degrades ITS OWN donor far more than other donors in the
                            same window, and far more than motif-matched non-partner AGs
  3. localisation        -- a small set of heads carries and causally controls the effect

Pilot (n=200 internal exons, chr-train): partner acceptor -0.436 donor margin vs +0.016
for motif-matched non-partner AGs (~28x); own donor -0.436 vs other donors +0.004 in the
same window (paired Wilcoxon p=2.4e-8).  Effect is heavy-tailed: mean 0.436, median 0.051,
~38% of exons above the 95th pct of the null -- so STRATIFY before claiming anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Sequence

import numpy as np

# Substitutions that are guaranteed to DESTROY the motif and never create the other one.
ACCEPTOR_ALTS = ("CC", "TT", "CA", "TC")      # none is AG
DONOR_ALTS = ("CC", "AA", "TA", "AC")         # none is GT and none is GC
_FLIP = {"A": "C", "C": "A", "G": "T", "T": "G"}


@dataclass(frozen=True)
class CouplingExample:
    """One internal exon in a fixed-length real-genome window."""

    example_id: str
    sequence: str
    acceptor_pos: int              # first exon base; the AG is at [pos-2, pos)
    donor_pos: int                 # first intron base; the GT is at [pos, pos+2)
    chrom: str
    strand: str
    gene_id: str
    split: str
    # annotations for stratification (step 7)
    exon_length: int
    intron_length_down: int | None
    n_isoforms_donor: int          # transcript support -> constitutive vs alternative
    n_isoforms_acceptor: int
    gene_type: str
    other_ag: tuple[int, ...] = ()   # non-partner AG positions (motif-matched null)
    other_gt: tuple[int, ...] = ()   # non-partner GT positions (specificity test)
    real_acc: tuple[int, ...] = ()   # OTHER real annotated acceptors in-window (control)
    real_don: tuple[int, ...] = ()   # OTHER real annotated donors in-window (control)
    next_acceptor_pos: int | None = None
    """Acceptor of the NEXT exon, i.e. this donor's INTRON-mate partner.

    The same donor has TWO partners: its exon-mate acceptor upstream (coupling across the
    EXON) and its intron-mate acceptor downstream (coupling across the INTRON).  Comparing
    the two, at matched distance, is what separates genuine exon definition from generic
    attention distance-decay -- same donor, same model, same readout, only the intervening
    sequence type and the separation differ."""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_constitutive(self) -> bool:
        """A donor used by many isoforms is constitutive; one used by few is alternative."""
        return self.n_isoforms_donor >= 5


def substitute(seq: str, pos: int, alt: str) -> str:
    """Replace len(alt) bases at pos with REAL bases.

    Never uses [MASK]: the mask token is off-manifold and the model has strong priors
    about it, so a masked perturbation measures the model's reaction to the mask rather
    than to the loss of the motif.
    """
    if pos < 0 or pos + len(alt) > len(seq):
        raise ValueError("substitution runs off the sequence")
    return seq[:pos] + alt + seq[pos + len(alt):]


def perturb_random(seq: str, pos: int, k: int = 2) -> str:
    """Flip k bases at pos to a different real base (guaranteed to change)."""
    s = list(seq)
    for i in range(k):
        if 0 <= pos + i < len(s):
            s[pos + i] = _FLIP[s[pos + i]]
    return "".join(s)


def destroy_acceptor(seq: str, acceptor_pos: int) -> list[str]:
    """All variants that destroy the AG at [acceptor_pos-2, acceptor_pos)."""
    return [substitute(seq, acceptor_pos - 2, a) for a in ACCEPTOR_ALTS]


def destroy_donor(seq: str, donor_pos: int) -> list[str]:
    """All variants that destroy the GT at [donor_pos, donor_pos+2)."""
    return [substitute(seq, donor_pos, a) for a in DONOR_ALTS]


def transplant_acceptor(seq: str, acceptor_pos: int, donor_seq: str, donor_acc_pos: int,
                        span: int = 20) -> str:
    """Swap in ANOTHER exon's acceptor context of comparable strength.

    Separates "the model needs AN acceptor here" from "the model needs THIS acceptor".
    If the coupling survives transplantation, it is generic acceptor-presence; if it
    breaks, it is specific to the partner's identity.
    """
    lo = acceptor_pos - span
    hi = acceptor_pos + 2
    src = donor_seq[donor_acc_pos - span: donor_acc_pos + 2]
    if lo < 0 or hi > len(seq) or len(src) != hi - lo:
        raise ValueError("transplant window out of range")
    return seq[:lo] + src + seq[hi:]


class CouplingScorer:
    """Native masked-LM readout of splice-site recognition.

    donor_margin  : mask the G of the donor GT -> logit(G) - max(logit of A,C,T)
    acceptor_margin: mask the G of the acceptor AG -> logit(G) - max(logit of A,C,T)

    Higher = the model more strongly believes a splice site belongs there.
    """

    def __init__(self, model, tokenizer, device: str = "cpu", batch_size: int = 16):
        import torch

        self.m, self.tok, self.dev, self.bs = model, tokenizer, device, batch_size
        self.torch = torch
        self.mask_id = tokenizer.mask_token_id
        self.base_id = {b: tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def _margin(self, seqs: Sequence[str], positions: Sequence[int], base: str) -> np.ndarray:
        torch = self.torch
        out: list[float] = []
        others = [b for b in "ACGT" if b != base]
        with torch.no_grad():
            for i in range(0, len(seqs), self.bs):
                S, P = seqs[i:i + self.bs], positions[i:i + self.bs]
                enc = self.tok([" ".join(s) for s in S], return_tensors="pt",
                               padding=True).to(self.dev)
                for j, p in enumerate(P):
                    enc["input_ids"][j, p + 1] = self.mask_id      # +1 for [CLS]
                logits = self.m(**enc).logits.float()
                for j, p in enumerate(P):
                    v = logits[j, p + 1]
                    hit = v[self.base_id[base]].item()
                    alt = max(v[self.base_id[b]].item() for b in others)
                    out.append(hit - alt)
        return np.asarray(out)

    def donor_margin(self, seqs: Sequence[str], donor_pos: Sequence[int]) -> np.ndarray:
        """Mask the G of the donor GT."""
        return self._margin(seqs, donor_pos, "G")

    def acceptor_margin(self, seqs: Sequence[str], acceptor_pos: Sequence[int]) -> np.ndarray:
        """Mask the G of the acceptor AG (it sits at acceptor_pos - 1)."""
        return self._margin(seqs, [p - 1 for p in acceptor_pos], "G")

    def mlm_health(self, seqs: Sequence[str], n_probe: int = 20, seed: int = 0) -> float:
        """Mean margin at random NON-site positions.

        Used to verify that an intervention (e.g. head ablation) has not simply broken
        the language model. A circuit claim requires the cross-site effect to vanish
        while general MLM competence is PRESERVED.
        """
        rng = random.Random(seed)
        probe_seqs, probe_pos, probe_base = [], [], []
        for s in seqs:
            for _ in range(n_probe):
                p = rng.randrange(10, len(s) - 10)
                probe_seqs.append(s)
                probe_pos.append(p)
                probe_base.append(s[p])
        vals = []
        for b in "ACGT":
            idx = [i for i, x in enumerate(probe_base) if x == b]
            if idx:
                vals.append(self._margin([probe_seqs[i] for i in idx],
                                         [probe_pos[i] for i in idx], b))
        return float(np.concatenate(vals).mean()) if vals else float("nan")


def coupling_effect(scorer: CouplingScorer, ex: CouplingExample, variants: Sequence[str],
                    target_pos: int, base: str = "G") -> float:
    """Mean drop in the target's margin, averaged over motif-destroying variants.

    Positive = the perturbation HURT recognition of the target site.
    """
    clean = scorer._margin([ex.sequence], [target_pos if base == "G" else target_pos], base)[0]
    pert = scorer._margin(list(variants), [target_pos] * len(variants), base)
    return float(clean - pert.mean())


def stable_rng(key: str, seed: int = 0) -> random.Random:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# ---------------------------------------------------------------------------
# Dataset build.  Geometry hygiene is inherited from partner_data (v1 died of a
# positional leak): FIXED window length, acceptor offset randomised over a range
# INDEPENDENT of exon length, real-genome filler, never PAD.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CouplingConfig:
    window_length: int = 1024
    acc_min: int = 100
    acc_max: int = 300
    min_context: int = 100
    min_exon_length: int = 20
    max_exon_length: int = 500
    seed: int = 0
    val_chromosomes: tuple[str, ...] = ("chr11", "chr16", "chr19")
    test_chromosomes: tuple[str, ...] = ("chr8", "chr9")   # SEALED

    def validate(self) -> None:
        if self.acc_min == self.acc_max:
            raise ValueError("acc_min == acc_max reintroduces the v1 positional leak")
        if self.acc_max + self.max_exon_length + self.min_context > self.window_length:
            raise ValueError("acceptor range must be feasible for the longest exon "
                             "independently of exon length, or acceptor_pos leaks it")


def build_coupling_dataset(gtf_path, genome, cfg: CouplingConfig, *,
                           allowed_chromosomes=None, limit=None, **gtf_kwargs):
    """Every internal exon in a fixed-length real-genome window, with stratification keys."""
    from collections import Counter, defaultdict

    from gfmcd.splicing.dataset import reverse_complement
    from gfmcd.splicing.gtf import load_transcripts
    from gfmcd.splicing.partner_data import (
        _acceptor_anchor, _donor_anchor, _to_window, ACCEPTOR_MOTIFS, DONOR_MOTIFS,
    )

    cfg.validate()
    txs = load_transcripts(gtf_path, allowed_chromosomes=allowed_chromosomes, **gtf_kwargs)

    # transcript support -> constitutive vs alternative (step 7)
    don_n: Counter = Counter()
    acc_n: Counter = Counter()
    gene_acc: dict = defaultdict(set)
    gene_don: dict = defaultdict(set)
    for t in txs:
        ex = sorted(t.exons, key=lambda e: e.start)
        for e in ex[1:-1]:
            don_n[(e.gene_id, _donor_anchor(e))] += 1
            acc_n[(e.gene_id, _acceptor_anchor(e))] += 1
            gene_acc[e.gene_id].add(_acceptor_anchor(e))
            gene_don[e.gene_id].add(_donor_anchor(e))

    out, seen = [], set()
    for t in txs:
        exons = sorted(t.exons, key=lambda e: e.start)
        if len(exons) < 3:
            continue
        for i, exon in enumerate(exons[1:-1], start=1):
            length = exon.end - exon.start
            if not (cfg.min_exon_length <= length <= cfg.max_exon_length):
                continue
            key = (exon.gene_id, _acceptor_anchor(exon), _donor_anchor(exon))
            if key in seen:
                continue

            eid = f"{exon.chrom}:{exon.start}-{exon.end}:{exon.strand}:{exon.gene_id}"
            acc_win = stable_rng(eid, cfg.seed).randint(cfg.acc_min, cfg.acc_max)
            N = cfg.window_length
            if N - acc_win - length < cfg.min_context:
                continue
            if exon.strand == "+":
                ws = exon.start - acc_win
                we = ws + N
            else:
                we = exon.end + acc_win
                ws = we - N
            if ws < 0 or we > genome.length(exon.chrom):
                continue
            seq = genome.fetch(exon.chrom, ws, we)
            if exon.strand == "-":
                seq = reverse_complement(seq)
            seq = seq.upper().replace("U", "T")
            if len(seq) != N or set(seq) - set("ACGT"):
                continue

            a = _to_window(_acceptor_anchor(exon), exon.strand, ws, we)
            d = _to_window(_donor_anchor(exon), exon.strand, ws, we)
            if a != acc_win or d != acc_win + length:
                continue
            if seq[a - 2:a] not in ACCEPTOR_MOTIFS or seq[d:d + 2] not in DONOR_MOTIFS:
                continue

            # downstream intron length (for the intron-definition direction, step 6)
            nxt = exons[i + 1] if i + 1 < len(exons) else None
            intron = None
            next_acc = None
            if nxt is not None:
                intron = (nxt.start - exon.end) if exon.strand == "+" else (exon.start - nxt.end)
                if intron is not None and intron < 0:
                    intron = None
                if intron is not None:
                    w = _to_window(_acceptor_anchor(nxt), exon.strand, ws, we)
                    if 4 <= w < N - 4 and seq[w - 2:w] == "AG" and w > d:
                        next_acc = w

            other_ag = tuple(p for p in range(4, N - 4)
                             if seq[p - 2:p] == "AG" and abs(p - a) > 10 and abs(p - d) > 10)
            other_gt = tuple(p for p in range(4, N - 4)
                             if seq[p:p + 2] == "GT" and abs(p - d) > 10 and abs(p - a) > 10)
            # OTHER REAL annotated sites of this gene that fall inside the window
            real_acc = tuple(sorted(
                w for g in gene_acc[exon.gene_id]
                for w in [_to_window(g, exon.strand, ws, we)]
                if 4 <= w < N - 4 and abs(w - a) > 10 and seq[w - 2:w] == "AG"))
            real_don = tuple(sorted(
                w for g in gene_don[exon.gene_id]
                for w in [_to_window(g, exon.strand, ws, we)]
                if 4 <= w < N - 4 and abs(w - d) > 10 and seq[w:w + 2] in ("GT", "GC")))

            c = exon.chrom if exon.chrom.startswith("chr") else f"chr{exon.chrom}"
            split = ("val" if c in cfg.val_chromosomes
                     else "test" if c in cfg.test_chromosomes else "train")

            seen.add(key)
            out.append(CouplingExample(
                example_id=eid, sequence=seq, acceptor_pos=a, donor_pos=d,
                chrom=exon.chrom, strand=exon.strand, gene_id=exon.gene_id, split=split,
                exon_length=length, intron_length_down=intron,
                n_isoforms_donor=don_n[(exon.gene_id, _donor_anchor(exon))],
                n_isoforms_acceptor=acc_n[(exon.gene_id, _acceptor_anchor(exon))],
                gene_type=exon.gene_type,
                other_ag=other_ag, other_gt=other_gt, real_acc=real_acc, real_don=real_don,
                next_acceptor_pos=next_acc,
                metadata={"genomic_window_start": ws, "genomic_window_end": we,
                          "genomic_exon_start": exon.start, "genomic_exon_end": exon.end},
            ))
            if limit and len(out) >= limit:
                return out
    return out
