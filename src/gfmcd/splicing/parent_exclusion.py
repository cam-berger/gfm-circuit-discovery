"""Model-free leakage control for evaluation-parent genes.

WHY THIS EXISTS
----------------
A study continue-pretrains a language model on a modified genomic corpus and then
evaluates it on designed sequence constructs built from authentic "parent" genes.
Training data is permanent in a way evaluation splits are not: a parent gene that
leaks into the training corpus cannot be un-trained, and it silently invalidates
the evaluation. This module answers, before any corpus is built, "which identifiers
must be excluded from training data, and why":

  1. every evaluation parent gene, by version-normalised ID (``normalize_gene_id``);
  2. close homologues that gene-ID exclusion alone would miss -- paralogs and
     near-identical windows, via a seed-based k-mer screen (``kmer_index``,
     ``near_identity_hits``);
  3. whole orthogroups for multi-genome work, so orthologs across species are
     excluded or retained together and never split (``orthogroup_key``);
  4. the full, audited union of the above (``build_exclusion_set``), with a
     reason recorded for every excluded identifier.

Deliberately model-free: no ``torch``, no model adapter, no checkpoint, no
network, no genome file access. Sequences and mappings are plain strings and
mappings so the module is testable in isolation, and importable before a
corpus-build environment exists.

SCOPE, HONESTLY STATED
-----------------------
This module bounds leakage; it does not eliminate it. Three things it does
NOT catch, by construction:

  - **Designed variants** derived from a parent (point mutants, shuffles,
    chimeras) are not identified here. That is a separate, evolving concern
    that lives with the design registry, not with this frozen leakage screen.
  - **Distant homology.** ``near_identity_hits`` is a seed-and-extend-free,
    *exact* k-mer screen. Its sensitivity floor is derived, not assumed --
    see the function docstring. Anything diverged past that floor (distant
    paralogs, most segmental duplications outside their high-identity core,
    most retrogenes after purifying-selection decay) will NOT be flagged by
    this screen and needs a real aligner (e.g. BLAST/minimap2) upstream of
    it, or acceptance of residual risk.
  - **Orthology inference.** ``orthogroup_key`` groups genes using a mapping
    the caller supplies; it never infers orthology itself. Coverage is only
    as good as that mapping (e.g. a species pair or a paralog family the
    mapping omits will not be grouped).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

# `orthogroup_key` prefixes: a gene present in the caller-supplied mapping and a
# gene absent from it are DIFFERENT claims (an asserted family vs. no orthology
# information at all) and must never collide, even for the same gene ID. See
# `orthogroup_key` for the reasoning.
_MAPPED_PREFIX = "orthogroup:"
_ABSENT_PREFIX = "singleton-absent:"

# `build_exclusion_set` reason kinds.
REASON_DIRECT_PARENT = "direct_parent"
REASON_ORTHOGROUP_MEMBER = "orthogroup_member"
REASON_HOMOLOGY_HIT = "homology_hit"


def normalize_gene_id(gene_id: str) -> str:
    """Strip the Ensembl-style version suffix: ``ENSG00000099194.8`` -> ``ENSG00000099194``.

    Matches the convention already used across the repo (e.g.
    ``gfmcd.splicing.pbc_biology``): split on the first ``.`` and keep the
    left half. A gene_id with no ``.`` is returned unchanged.
    """
    return gene_id.split(".", 1)[0]


def orthogroup_key(gene_id: str, mapping: Mapping[str, str]) -> str:
    """Group ``gene_id`` into its orthogroup using a caller-supplied ``mapping``.

    This never infers orthology -- it only looks up ``normalize_gene_id(gene_id)``
    in ``mapping``, which must be keyed by normalized (version-stripped) gene
    IDs. Two genes get the same key iff ``mapping`` says so.

    A gene absent from ``mapping`` is treated as its own (trivial) orthogroup,
    because "no orthology information was supplied" is not the same claim as
    "the mapping asserts this gene has no relatives" -- the former is silence,
    the latter is a (possibly wrong) assertion. To keep those distinguishable,
    and therefore auditable, the two cases use different key prefixes and can
    never compare equal to each other:

      - present in ``mapping``  -> ``"orthogroup:<mapping[normalized_id]>"``
        (a mapping-asserted group; may have any number of members, including
        exactly one -- a "singleton-by-mapping")
      - absent from ``mapping`` -> ``"singleton-absent:<normalized_id>"``
        (no orthology information at all -- a "singleton-by-absence")

    Two absent genes never collide with each other either: each absence key
    embeds its own normalized gene ID, so absence never silently merges
    unrelated genes into one group.
    """
    normalized = normalize_gene_id(gene_id)
    if normalized in mapping:
        return f"{_MAPPED_PREFIX}{mapping[normalized]}"
    return f"{_ABSENT_PREFIX}{normalized}"


def kmer_index(sequences: Mapping[str, str], *, k: int) -> dict[str, set[str]]:
    """Build ``sequence_id -> set of its (uppercased) length-``k`` substrings``.

    A sequence shorter than ``k`` indexes to an empty set (it contributes no
    seeds and can never be hit or hit anything). Case-insensitive: sequences
    are uppercased before k-mers are cut, so mixed-case FASTA does not create
    spurious mismatches.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    index: dict[str, set[str]] = {}
    for seq_id, sequence in sequences.items():
        s = sequence.upper()
        index[seq_id] = {s[i : i + k] for i in range(len(s) - k + 1)} if len(s) >= k else set()
    return index


def near_identity_hits(
    query: str, index: Mapping[str, set[str]], *, k: int, min_shared_kmers: int
) -> dict[str, int]:
    """Seed-based near-identity screen: exact shared k-mers between ``query`` and an index.

    For every ``seq_id`` in ``index`` whose k-mer set shares at least
    ``min_shared_kmers`` exact length-``k`` substrings with ``query``, returns
    ``{seq_id: shared_kmer_count}``. No alignment, no gaps, no external
    aligner -- a match requires a byte-identical window of length ``k``
    somewhere in both sequences.

    SENSITIVITY, DERIVED (read before trusting a "not detected" result)
    ---------------------------------------------------------------------
    A query of length ``L`` has ``L - k + 1`` overlapping k-mers ("seeds").
    A single substitution at position ``p`` corrupts every k-mer window that
    covers ``p`` -- there are at most ``k`` such windows (fewer near the
    query's ends) -- and leaves every other seed untouched. So with ``m``
    substitutions total, the number of surviving (exactly-matching) seeds is
    bounded below by:

        retained >= max(0, (L - k + 1) - m * k)

    This is a WORST-CASE floor (substitutions placed to each destroy a
    disjoint set of k windows); random substitution placement is usually far
    more forgiving because nearby substitutions overlap in which windows they
    destroy. Two consequences follow directly from the inequality:

      - Guaranteed detection is not free of the query's own length: for a
        SINGLE substitution to be survivable, you need ``L - 2k + 1 >= min_shared_kmers``,
        i.e. plenty of seeds outlive it as long as ``L`` is more than a
        couple of k-mer lengths.
      - The floor hits zero, and non-detection becomes achievable even by an
        adversary who must place substitutions to destroy every window, once
        ``m >= (L - k + 1) / k`` -- i.e. roughly one substitution every ``k``
        bases (substitution density ~= ``1/k``). Below that density the
        worst case still leaves at least one exact seed; above it, a
        substitution pattern exists (evenly spaced, one every k bases) that
        leaves NO shared k-mer at all, however small ``min_shared_kmers`` is
        set.

    In short: this screen detects NEAR-identity (sparse edits relative to
    ``k``), not remote homology. A diverged paralog whose substitutions
    average denser than about one per ``k`` bases can share zero seeds with
    its relative and pass through undetected -- see the module docstring for
    what that means for distant paralogs, segmental duplications, and
    retrogenes.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    q = query.upper()
    query_kmers = {q[i : i + k] for i in range(len(q) - k + 1)} if len(q) >= k else set()
    hits: dict[str, int] = {}
    for seq_id, kmers in index.items():
        shared = len(query_kmers & kmers)
        if shared >= min_shared_kmers:
            hits[seq_id] = shared
    return hits


def build_exclusion_set(
    parents: Mapping[str, str],
    *,
    orthogroup_mapping: Mapping[str, str],
    corpus_index: Mapping[str, set[str]],
    k: int,
    min_shared_kmers: int,
) -> dict[str, object]:
    """Build the full, audited set of identifiers to exclude from training data.

    ``parents`` maps evaluation-parent gene ID -> parent sequence (the window
    the homology screen runs against). Three exclusion mechanisms are unioned,
    and every excluded identifier records at least one reason it was excluded:

      1. ``REASON_DIRECT_PARENT``     -- the gene itself is an evaluation parent.
      2. ``REASON_ORTHOGROUP_MEMBER`` -- ``orthogroup_mapping`` places another
         gene in the same orthogroup as a parent (see ``orthogroup_key``); only
         parents that are actually present in ``orthogroup_mapping`` can pull in
         siblings -- a parent absent from the mapping is a singleton-by-absence
         and contributes no orthogroup exclusions (there is nothing to group it
         with).
      3. ``REASON_HOMOLOGY_HIT``      -- a corpus entry in ``corpus_index``
         (built with ``kmer_index``) shares at least ``min_shared_kmers`` exact
         k-mers with a parent sequence (see ``near_identity_hits`` for what this
         screen does and does not catch).

    An identifier can carry more than one reason (e.g. a corpus entry that is
    both an orthogroup member AND a homology hit); all applicable reasons are
    kept, none is dropped in favour of another.

    Returns a JSON-serialisable dict:

        {
          "k": k,
          "min_shared_kmers": min_shared_kmers,
          "parent_ids": [normalized parent gene IDs, sorted],
          "excluded": {
              excluded_id: [
                  {"reason": REASON_*, "detail": "<human-readable why>"},
                  ...
              ],
              ...
          },
        }

    This function does NOT identify designed variants derived from a parent
    (point mutants, shuffles, chimeras) -- that is a separate concern owned by
    the design registry, not by this leakage screen.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    excluded: dict[str, list[dict[str, str]]] = defaultdict(list)
    normalized_parent_ids = sorted({normalize_gene_id(raw_id) for raw_id in parents})

    # 1. Direct parents.
    for raw_id in parents:
        norm_id = normalize_gene_id(raw_id)
        excluded[norm_id].append(
            {
                "reason": REASON_DIRECT_PARENT,
                "detail": f"evaluation parent gene {raw_id!r}",
            }
        )

    # 2. Orthogroup siblings. Invert orthogroup_mapping once: group_key -> members.
    groups: dict[str, list[str]] = defaultdict(list)
    for gene_id in orthogroup_mapping:
        groups[orthogroup_key(gene_id, orthogroup_mapping)].append(gene_id)

    for raw_id in parents:
        norm_id = normalize_gene_id(raw_id)
        key = orthogroup_key(norm_id, orthogroup_mapping)
        if not key.startswith(_MAPPED_PREFIX):
            # Singleton-by-absence: no orthology information, nothing to pull in.
            continue
        for sibling_id in groups[key]:
            sibling_norm = normalize_gene_id(sibling_id)
            if sibling_norm == norm_id:
                continue
            excluded[sibling_norm].append(
                {
                    "reason": REASON_ORTHOGROUP_MEMBER,
                    "detail": f"orthogroup {key!r} shared with parent {raw_id!r}",
                }
            )

    # 3. Near-identity homology hits against the corpus index.
    for raw_id, sequence in parents.items():
        hits = near_identity_hits(sequence, corpus_index, k=k, min_shared_kmers=min_shared_kmers)
        for corpus_id, shared_count in hits.items():
            excluded[corpus_id].append(
                {
                    "reason": REASON_HOMOLOGY_HIT,
                    "detail": (
                        f"shares {shared_count} exact {k}-mers "
                        f"(threshold {min_shared_kmers}) with parent {raw_id!r}"
                    ),
                }
            )

    return {
        "k": k,
        "min_shared_kmers": min_shared_kmers,
        "parent_ids": normalized_parent_ids,
        "excluded": {gene_id: excluded[gene_id] for gene_id in sorted(excluded)},
    }
