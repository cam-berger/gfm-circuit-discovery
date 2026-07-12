"""v2 partner-binding dataset: real annotated splice sites as decoys.

WHY THIS MODULE EXISTS
----------------------
The v1 exon-pair dataset (``dataset.py``) is fatally confounded for the *partner*
claim, in two independent ways:

1. **Positional label leak.** ``dataset.py`` sets ``acceptor = cfg.left_flank``, a
   *constant*.  The true acceptor therefore sits at position 200 in 100% of
   examples, and a transformer's hidden states carry absolute position embeddings,
   so the acceptor can be identified by *reading its position*.  A model with no
   neural network at all (position bonus + the same PWM donor score + the same
   length prior) reaches PMR 0.483 where the published baseline is 0.000.

2. **The task is solvable by site detection.** v1's decoys are *spurious* AG / GT
   motif matches, so an oracle site detector scores PMR 1.000.  Decoys differ from
   the true pair in "is this a real splice site", never in "which site pairs with
   which" -- so partner binding is never tested.

v2 fixes both:

* **Flanks are randomised** (``left_flank ~ U[flank_min, flank_max]``), so the
  acceptor's absolute position carries no information.  ``PositionOnlyRival`` must
  fall to chance; ``tests/test_partner_data.py`` enforces this.

* **Decoys are REAL annotated splice sites** -- donors that genuinely exist in the
  same gene, land inside the window at a plausible exon length, and are **never an
  attested partner of this acceptor in any isoform**.  Because every candidate is a
  real donor, site detection is useless *by construction*, and an oracle site
  detector is at chance.

The exclusion of *attested alternative partners* (condition (d) in
``_real_decoy_donors``) is load-bearing: a donor that this acceptor pairs with in
some other isoform is a **positive**, not a decoy.  Counting those as decoys
inflates the yield and mislabels the data.

All sequence is real hg38; all sites are real GENCODE annotations.  Nothing here is
synthetic or chimeric, so the result carries no off-manifold caveat.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from gfmcd.splicing.dataset import GenomeAccessor, reverse_complement
from gfmcd.splicing.gtf import ExonRecord, TranscriptRecord, load_transcripts

DONOR_MOTIFS = ("GT", "GC")
ACCEPTOR_MOTIFS = ("AG",)


@dataclass(frozen=True)
class PartnerConfig:
    """Construction parameters.

    THREE geometry invariants, each killing a distinct leak.  Do not relax any of them.

    1. ``window_length`` is CONSTANT for every example.  If the window were
       ``flank + exon + flank`` (as in v1, and as in this module's own first draft),
       then ``exon_length == len(seq) - 2*flank`` EXACTLY, so a "length-only"
       adversary reading zero nucleotides recovers the true exon length from the
       sequence length alone -- and, since the acceptor is given, the true donor is
       ``acceptor_pos + (len(seq) - 2*flank)``.  That is PMR 1.0 for free.
       Windows are therefore padded to a fixed length with REAL flanking genome.
       Never pad with PAD tokens: the pad count is itself a length channel.

    2. ``acceptor_pos`` is drawn from ``[acc_min, acc_max]``, a range that does NOT
       depend on the exon length.  If the range depended on the exon (e.g. "anywhere
       that leaves room"), then long exons would force small offsets and
       ``acceptor_pos`` would leak the exon length.

    3. ``acc_min != acc_max``, so the acceptor is not at a constant offset (the v1
       leak: acceptor sat at position 200 in 100.0% of examples).
    """

    window_length: int = 1024          # CONSTANT -- carries no information
    acc_min: int = 100                 # acceptor offset range, INDEPENDENT of exon length
    acc_max: int = 300
    min_context: int = 100             # min real genome downstream of the donor
    min_exon_length: int = 20
    max_exon_length: int = 500
    distance_match_bp: int = 30
    seed: int = 0
    val_chromosomes: tuple[str, ...] = ("chr7", "chr13")
    test_chromosomes: tuple[str, ...] = ("chr8", "chr9")  # SEALED

    def validate(self) -> None:
        if self.acc_min < 4 or self.acc_max < self.acc_min:
            raise ValueError("need 4 <= acc_min <= acc_max")
        if self.acc_min == self.acc_max:
            raise ValueError(
                "acc_min == acc_max reintroduces the v1 positional leak: the true "
                "acceptor would sit at a constant position in every example"
            )
        if self.min_exon_length < 1 or self.max_exon_length < self.min_exon_length:
            raise ValueError("bad exon-length bounds")
        # invariant 2: the acceptor range must be feasible for the LONGEST exon, so that
        # the sampling range never has to shrink for long exons (which would leak length).
        need = self.acc_max + self.max_exon_length + self.min_context
        if need > self.window_length:
            raise ValueError(
                f"window_length={self.window_length} too small: the acceptor range must be "
                f"feasible for the longest exon independently of exon length (need >= {need}). "
                "Otherwise acceptor_pos leaks the exon length."
            )


@dataclass(frozen=True)
class PartnerExample:
    """One acceptor, its true donor, and REAL-SITE decoy donors."""

    example_id: str
    sequence: str                           # ALWAYS cfg.window_length long (no length channel)
    acceptor_pos: int                       # in-window; VARIES across examples (leak fix)
    candidate_donor_pos: tuple[int, ...]    # PERMUTED: true donor + real-site decoys, order is
                                            # seeded on the SEQUENCE, never on the label
    true_index: int                         # index of the true donor within candidate_donor_pos
    motif_donor_pos: tuple[int, ...]        # all GT/GC motifs (v1-style easy decoys; for rivals)
    distance_matched: bool                  # a real decoy within distance_match_bp of the true exon length
    split: str
    chrom: str
    strand: str
    gene_id: str
    transcript_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def true_donor_pos(self) -> int:
        return self.candidate_donor_pos[self.true_index]

    @property
    def real_decoy_donor_pos(self) -> tuple[int, ...]:
        return tuple(d for i, d in enumerate(self.candidate_donor_pos) if i != self.true_index)

    @property
    def true_exon_length(self) -> int:
        return self.true_donor_pos - self.acceptor_pos

    @property
    def candidates(self) -> tuple[int, ...]:
        return self.candidate_donor_pos

    def validate(self) -> None:
        seq = self.sequence
        if seq.upper().replace("U", "T") != seq:
            raise ValueError("sequence must be upper-case DNA")
        if not 0 <= self.acceptor_pos < len(seq):
            raise ValueError("acceptor outside sequence")
        if seq[self.acceptor_pos - 2 : self.acceptor_pos] not in ACCEPTOR_MOTIFS:
            raise ValueError(f"{self.example_id}: acceptor not motif-compatible")
        if not 0 <= self.true_index < len(self.candidate_donor_pos):
            raise ValueError("true_index out of range")
        if len(self.candidate_donor_pos) < 2:
            raise ValueError("need >= 2 candidates for a forced choice")
        if len(set(self.candidate_donor_pos)) != len(self.candidate_donor_pos):
            raise ValueError("duplicate candidate donors")
        for d in self.candidate_donor_pos:
            if not 0 <= d < len(seq) - 1:
                raise ValueError("donor outside sequence")
            if seq[d : d + 2] not in DONOR_MOTIFS:
                raise ValueError(f"{self.example_id}: donor {d} not motif-compatible")
            if d <= self.acceptor_pos:
                raise ValueError(f"{self.example_id}: donor {d} is not downstream of the acceptor")


def _acceptor_offset_for(example_id: str, cfg: PartnerConfig) -> int:
    """Deterministic per-example acceptor offset -- reproducible, but not a constant,
    and drawn from a range that does NOT depend on the exon length."""
    h = hashlib.sha256(f"{cfg.seed}:acc:{example_id}".encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    return rng.randint(cfg.acc_min, cfg.acc_max)


def _permute_candidates(donors: Sequence[int], true_donor: int, sequence: str) -> tuple[tuple[int, ...], int]:
    """Shuffle the candidate list with a seed derived from the SEQUENCE, never the label.

    If the true donor always sat at index 0, the candidate INDEX would be a perfect
    label -- a free win for anything that sees list order.  Seeding on the sequence
    keeps the order deterministic and reproducible while making it label-independent.
    """
    h = hashlib.sha256(sequence.encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    cands = list(donors)
    rng.shuffle(cands)
    return tuple(cands), cands.index(true_donor)


def _split_for(chrom: str, cfg: PartnerConfig) -> str:
    c = chrom if chrom.startswith("chr") else f"chr{chrom}"
    if c in cfg.val_chromosomes:
        return "val"
    if c in cfg.test_chromosomes:
        return "test"
    return "train"


def _acceptor_anchor(exon: ExonRecord) -> int:
    """Genomic coordinate of the exon's first base in TRANSCRIPT orientation."""
    return exon.start if exon.strand == "+" else exon.end - 1


def _donor_anchor(exon: ExonRecord) -> int:
    """Genomic coordinate of the first INTRON base after the exon, in transcript orientation."""
    return exon.end if exon.strand == "+" else exon.start - 1


def _to_window(g: int, strand: str, window_start: int, window_end: int) -> int:
    """Genomic coordinate -> index in the transcript-oriented window."""
    if strand == "+":
        return g - window_start
    return window_end - 1 - g


@dataclass
class _GeneIndex:
    """Every annotated site of a gene, plus which donors each acceptor legitimately pairs with."""

    donors: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    attested: dict[tuple[str, int], set[int]] = field(default_factory=lambda: defaultdict(set))


def _build_gene_index(transcripts: Iterable[TranscriptRecord]) -> _GeneIndex:
    idx = _GeneIndex()
    for tx in transcripts:
        exons = sorted(tx.exons, key=lambda e: e.start)
        if len(exons) < 3:
            continue
        for exon in exons[1:-1]:  # internal exons only: both ends are real splice sites
            a = _acceptor_anchor(exon)
            d = _donor_anchor(exon)
            idx.donors[exon.gene_id].add(d)
            idx.attested[(exon.gene_id, a)].add(d)
    return idx


def _real_decoy_donors(
    exon: ExonRecord,
    idx: _GeneIndex,
    *,
    acceptor_win: int,
    true_donor_win: int,
    window_start: int,
    window_end: int,
    seq: str,
    cfg: PartnerConfig,
) -> list[int]:
    """Real annotated donors that this acceptor NEVER pairs with, inside the window.

    Conditions -- every one is load-bearing:
      (a) annotated donor of the SAME gene
      (b) inside the window, motif-compatible
      (c) plausible exon length, downstream of the acceptor in transcript orientation
      (d) NOT an attested partner of this acceptor in ANY isoform  <-- else it is a POSITIVE
    """
    a_anchor = _acceptor_anchor(exon)
    attested = idx.attested[(exon.gene_id, a_anchor)]  # (d)
    out: list[int] = []
    for g in idx.donors[exon.gene_id]:
        if g in attested:
            continue
        w = _to_window(g, exon.strand, window_start, window_end)
        if not 0 <= w < len(seq) - 1:
            continue
        if seq[w : w + 2] not in DONOR_MOTIFS:
            continue
        length = w - acceptor_win
        if not (cfg.min_exon_length <= length <= cfg.max_exon_length):
            continue
        if w == true_donor_win:
            continue
        out.append(w)
    return sorted(set(out))


def _motif_donors(seq: str, acceptor_win: int, cfg: PartnerConfig) -> list[int]:
    """All GT/GC motif positions at a plausible exon length -- the v1-style easy decoys."""
    out = []
    for p in range(len(seq) - 1):
        if seq[p : p + 2] in DONOR_MOTIFS:
            length = p - acceptor_win
            if cfg.min_exon_length <= length <= cfg.max_exon_length:
                out.append(p)
    return out


def build_partner_dataset(
    gtf_path: str | Path,
    genome: GenomeAccessor,
    cfg: PartnerConfig,
    *,
    allowed_chromosomes: set[str] | None = None,
    limit: int | None = None,
    **gtf_kwargs: Any,
) -> list[PartnerExample]:
    """Build the v2 real-site-decoy partner dataset from GENCODE + hg38."""
    cfg.validate()
    transcripts = load_transcripts(
        gtf_path, allowed_chromosomes=allowed_chromosomes, **gtf_kwargs
    )
    idx = _build_gene_index(transcripts)

    out: list[PartnerExample] = []
    seen: set[tuple[str, int, int]] = set()  # dedupe identical exons across isoforms

    for tx in transcripts:
        exons = sorted(tx.exons, key=lambda e: e.start)
        if len(exons) < 3:
            continue
        for exon in exons[1:-1]:
            length = exon.end - exon.start
            if not (cfg.min_exon_length <= length <= cfg.max_exon_length):
                continue
            key = (exon.gene_id, _acceptor_anchor(exon), _donor_anchor(exon))
            if key in seen:
                continue

            example_id = f"{exon.chrom}:{exon.start}-{exon.end}:{exon.strand}:{exon.gene_id}"

            # --- GEOMETRY (see PartnerConfig): fixed total length, acceptor offset drawn
            # --- from a range independent of the exon length, filler is REAL genome.
            acceptor_win = _acceptor_offset_for(example_id, cfg)
            N = cfg.window_length
            right_context = N - acceptor_win - length
            if right_context < cfg.min_context:
                continue  # cannot happen given cfg.validate(), but never trust that silently

            if exon.strand == "+":
                window_start = exon.start - acceptor_win
                window_end = window_start + N
            else:
                window_end = exon.end + acceptor_win
                window_start = window_end - N
            if window_start < 0 or window_end > genome.length(exon.chrom):
                continue

            seq = genome.fetch(exon.chrom, window_start, window_end)
            if exon.strand == "-":
                seq = reverse_complement(seq)
            seq = seq.upper().replace("U", "T")
            if len(seq) != N:
                continue  # ran off the contig -- drop rather than pad (pad-count is a length channel)
            if set(seq) - set("ACGT"):
                continue  # assembly gap

            a_win = _to_window(_acceptor_anchor(exon), exon.strand, window_start, window_end)
            donor_win = _to_window(_donor_anchor(exon), exon.strand, window_start, window_end)
            if a_win != acceptor_win or donor_win != acceptor_win + length:
                continue  # coordinate maths must agree with the construction
            if seq[acceptor_win - 2 : acceptor_win] not in ACCEPTOR_MOTIFS:
                continue
            if seq[donor_win : donor_win + 2] not in DONOR_MOTIFS:
                continue

            decoys = _real_decoy_donors(
                exon, idx,
                acceptor_win=acceptor_win, true_donor_win=donor_win,
                window_start=window_start, window_end=window_end, seq=seq, cfg=cfg,
            )
            if not decoys:
                continue  # no real-site decoy -> cannot test partner binding on this exon

            matched = any(
                abs((d - acceptor_win) - length) <= cfg.distance_match_bp for d in decoys
            )
            cand, true_index = _permute_candidates([donor_win] + decoys, donor_win, seq)
            seen.add(key)
            ex = PartnerExample(
                example_id=example_id,
                sequence=seq,
                acceptor_pos=acceptor_win,
                candidate_donor_pos=cand,
                true_index=true_index,
                motif_donor_pos=tuple(_motif_donors(seq, acceptor_win, cfg)),
                distance_matched=matched,
                split=_split_for(exon.chrom, cfg),
                chrom=exon.chrom,
                strand=exon.strand,
                gene_id=exon.gene_id,
                transcript_id=exon.transcript_id,
                metadata={
                    "genomic_window_start": window_start,
                    "genomic_window_end": window_end,
                    "genomic_exon_start": exon.start,
                    "genomic_exon_end": exon.end,
                    "true_exon_length": length,
                    "n_real_decoys": len(decoys),
                    "decoy_exon_lengths": [d - acceptor_win for d in decoys],
                },
            )
            ex.validate()
            out.append(ex)
            if limit is not None and len(out) >= limit:
                return out
    return out


def split_partner_examples(examples: Iterable[PartnerExample]) -> dict[str, list[PartnerExample]]:
    """Split by the precomputed chromosome assignment.  Genes never span chromosomes,
    so chromosome-disjoint implies gene-disjoint."""
    out: dict[str, list[PartnerExample]] = {"train": [], "val": [], "test": []}
    for ex in examples:
        out[ex.split].append(ex)
    return out


def save_partner_jsonl(examples: Sequence[PartnerExample], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), sort_keys=True) + "\n")
    return p


def load_partner_jsonl(path: str | Path, *, validate: bool = True) -> list[PartnerExample]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["candidate_donor_pos"] = tuple(d["candidate_donor_pos"])
            d["motif_donor_pos"] = tuple(d["motif_donor_pos"])
            ex = PartnerExample(**d)
            if validate:
                ex.validate()
            out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Strata.  Kept OUT of the builder so the build stays label-blind; the length
# prior is fit on TRAIN only and applied to every split.
# ---------------------------------------------------------------------------

class LengthPrior:
    """Marginal exon-length prior, fit on the training split only."""

    def __init__(self, lengths: Sequence[int], bins: int = 60, hi: int = 500):
        import numpy as np

        self.hist, _ = np.histogram(lengths, bins=bins, range=(0, hi), density=True)
        self.bins, self.hi = bins, hi

    def __call__(self, length: int) -> float:
        i = min(int(length / self.hi * self.bins), self.bins - 1)
        return float(self.hist[i]) + 1e-9


def stratify(ex: PartnerExample, prior: LengthPrior) -> str:
    """Which evaluation stratum does this example fall in?

    ``adversarial`` is the headline stratum: BOTH the nearest-downstream rule AND the
    exon-length prior prefer a DECOY over the true donor.  Since every candidate is
    also a real annotated splice site (so site detection is at chance), a model that
    beats chance there cannot be using position, length, distance, or site-hood --
    only partner information remains.  That is what makes the claim identified.
    """
    t = ex.true_exon_length
    decoys = [d - ex.acceptor_pos for d in ex.real_decoy_donor_pos]
    nearer = any(d < t for d in decoys)
    likelier = any(prior(d) > prior(t) for d in decoys)
    if nearer and likelier:
        return "adversarial"
    if nearer or likelier:
        return "conflict"
    return "easy"
