"""Candidate acceptor/donor enumeration for exon-pair selection."""

from __future__ import annotations

from dataclasses import dataclass

from gfmcd.splicing.schema import PairCandidate, SiteCandidate


@dataclass(frozen=True)
class CandidateConfig:
    donor_motifs: tuple[str, ...] = ("GT", "GC")
    acceptor_motifs: tuple[str, ...] = ("AG",)
    min_exon_length: int = 20
    max_exon_length: int = 500
    edge_buffer: int = 4
    max_pairs: int | None = 512

    def validate(self) -> None:
        if self.min_exon_length < 1:
            raise ValueError("min_exon_length must be positive")
        if self.max_exon_length < self.min_exon_length:
            raise ValueError("max_exon_length must be >= min_exon_length")
        if self.edge_buffer < 0:
            raise ValueError("edge_buffer must be non-negative")
        if self.max_pairs is not None and self.max_pairs < 1:
            raise ValueError("max_pairs must be positive or None")


def enumerate_sites(sequence: str, cfg: CandidateConfig) -> tuple[tuple[SiteCandidate, ...], tuple[SiteCandidate, ...]]:
    """Enumerate motif-compatible site anchors in transcript orientation."""
    cfg.validate()
    seq = sequence.upper().replace("U", "T")
    lo = cfg.edge_buffer
    hi = len(seq) - cfg.edge_buffer

    acceptors: list[SiteCandidate] = []
    for pos in range(max(2, lo), hi):
        motif = seq[pos - 2 : pos]
        if motif in cfg.acceptor_motifs:
            acceptors.append(SiteCandidate(position=pos, kind="acceptor", motif=motif))

    donors: list[SiteCandidate] = []
    for pos in range(lo, min(hi, len(seq) - 1)):
        motif = seq[pos : pos + 2]
        if motif in cfg.donor_motifs:
            donors.append(SiteCandidate(position=pos, kind="donor", motif=motif))

    return tuple(acceptors), tuple(donors)


def enumerate_candidate_pairs(
    sequence: str,
    cfg: CandidateConfig,
    *,
    true_acceptor: int | None = None,
    true_donor: int | None = None,
) -> tuple[tuple[SiteCandidate, ...], tuple[SiteCandidate, ...], tuple[PairCandidate, ...], int | None]:
    """Enumerate all plausible exon boundary pairs.

    When ``true_acceptor`` and ``true_donor`` are supplied, the returned final
    element is the index of that pair.  The function raises if an annotated site
    is not motif-compatible; silently inserting it would make candidate
    generation and labels disagree.
    """
    acceptors, donors = enumerate_sites(sequence, cfg)
    acc_index = {s.position: i for i, s in enumerate(acceptors)}
    don_index = {s.position: i for i, s in enumerate(donors)}

    if true_acceptor is not None and true_acceptor not in acc_index:
        motif = sequence[max(0, true_acceptor - 2) : true_acceptor]
        raise ValueError(f"annotated acceptor {true_acceptor} not found (motif={motif!r})")
    if true_donor is not None and true_donor not in don_index:
        motif = sequence[true_donor : true_donor + 2]
        raise ValueError(f"annotated donor {true_donor} not found (motif={motif!r})")

    pairs: list[PairCandidate] = []
    for ai, acceptor in enumerate(acceptors):
        for di, donor in enumerate(donors):
            length = donor.position - acceptor.position
            if cfg.min_exon_length <= length <= cfg.max_exon_length:
                pairs.append(
                    PairCandidate(
                        acceptor_pos=acceptor.position,
                        donor_pos=donor.position,
                        acceptor_index=ai,
                        donor_index=di,
                    )
                )

    # Deterministic ordering is important for stable labels and manifests.
    pairs.sort(key=lambda p: (p.acceptor_pos, p.donor_pos))

    true_index: int | None = None
    if true_acceptor is not None or true_donor is not None:
        if true_acceptor is None or true_donor is None:
            raise ValueError("provide both true_acceptor and true_donor")
        for i, pair in enumerate(pairs):
            if pair.acceptor_pos == true_acceptor and pair.donor_pos == true_donor:
                true_index = i
                break
        if true_index is None:
            raise ValueError(
                "annotated pair excluded by length constraints: "
                f"acceptor={true_acceptor}, donor={true_donor}, "
                f"length={true_donor - true_acceptor}"
            )

    if cfg.max_pairs is not None and len(pairs) > cfg.max_pairs:
        # Keep candidates nearest the annotated pair when present, otherwise the
        # shortest pairs.  Always preserve the true pair.
        if true_index is not None:
            ta, td = true_acceptor, true_donor
            ranked = sorted(
                enumerate(pairs),
                key=lambda x: (
                    abs(x[1].acceptor_pos - ta) + abs(x[1].donor_pos - td),
                    abs(x[1].exon_length - (td - ta)),
                    x[1].acceptor_pos,
                    x[1].donor_pos,
                ),
            )[: cfg.max_pairs]
            kept_old = sorted(i for i, _ in ranked)
            if true_index not in kept_old:
                kept_old[-1] = true_index
                kept_old = sorted(set(kept_old))
            pairs = [pairs[i] for i in kept_old]
            true_index = next(
                i
                for i, p in enumerate(pairs)
                if p.acceptor_pos == true_acceptor and p.donor_pos == true_donor
            )
        else:
            pairs = sorted(pairs, key=lambda p: (p.exon_length, p.acceptor_pos, p.donor_pos))[
                : cfg.max_pairs
            ]
            pairs.sort(key=lambda p: (p.acceptor_pos, p.donor_pos))

    return acceptors, donors, tuple(pairs), true_index
