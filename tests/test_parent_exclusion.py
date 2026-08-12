"""Tests for the model-free leakage-control screen in gfmcd.splicing.parent_exclusion.

Every numeric choice below (k, window length, thresholds, mutation positions) is
derived from the sensitivity bound documented on `near_identity_hits`, not picked
by feel -- see the comments at each site for the arithmetic.
"""

from __future__ import annotations

import random

from gfmcd.splicing.parent_exclusion import (
    build_exclusion_set,
    kmer_index,
    near_identity_hits,
    normalize_gene_id,
    orthogroup_key,
)

_ALPHABET = "ACGT"
# A fixed substitution cipher: applying it to a base always yields a DIFFERENT
# base, so "mutate at position p" is guaranteed to change the sequence there
# (never coincidentally re-draws the same letter).
_SUBSTITUTE = {"A": "C", "C": "G", "G": "T", "T": "A"}

# k=16 windows of length 200 keep the coincidental-match false-positive rate for
# unrelated random sequences negligible (see test_unrelated_random_window_is_not_detected
# for the derivation) while leaving comfortable room above the min_shared_kmers
# threshold for the near-identity tests.
K = 16
WINDOW_LENGTH = 200
MIN_SHARED_KMERS = 5


def _random_sequence(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def _mutate_at(sequence: str, positions) -> str:
    chars = list(sequence)
    for pos in positions:
        chars[pos] = _SUBSTITUTE[chars[pos]]
    return "".join(chars)


def test_exact_duplicate_window_is_detected():
    rng = random.Random(1)
    seq = _random_sequence(rng, WINDOW_LENGTH)
    index = kmer_index({"corpus_window": seq}, k=K)

    hits = near_identity_hits(seq, index, k=K, min_shared_kmers=MIN_SHARED_KMERS)

    expected_kmers = WINDOW_LENGTH - K + 1  # 185: every window matches itself exactly
    assert hits == {"corpus_window": expected_kmers}


def test_unrelated_random_window_is_not_detected():
    # Two INDEPENDENT random draws over {A,C,G,T}. The k-mer universe has
    # 4**16 ~= 4.295e9 possible 16-mers; each sequence contributes 185 k-mers,
    # so the expected number of COINCIDENTAL shared k-mers between two
    # unrelated sequences is ~= (185 * 185) / 4.295e9 ~= 8e-6. Requiring
    # min_shared_kmers=5 coincidental matches (not just 1) pushes the false
    # positive probability for this single comparison to somewhere around
    # (8e-6)**5 / 5! -- astronomically below anything that could make this
    # test flaky.
    rng_a = random.Random(2)
    rng_b = random.Random(3)
    seq_a = _random_sequence(rng_a, WINDOW_LENGTH)
    seq_b = _random_sequence(rng_b, WINDOW_LENGTH)
    index = kmer_index({"corpus_window": seq_a}, k=K)

    hits = near_identity_hits(seq_b, index, k=K, min_shared_kmers=MIN_SHARED_KMERS)

    assert hits == {}


def test_single_substitution_is_still_detected_per_documented_bound():
    # Documented bound: retained >= (L - k + 1) - m*k. For m=1 substitution,
    # regardless of WHERE it lands, retained >= 185 - 16 = 169, which is far
    # above min_shared_kmers=5 -- so detection is guaranteed by the bound
    # itself, not by luck of the mutation position.
    rng = random.Random(4)
    seq = _random_sequence(rng, WINDOW_LENGTH)
    index = kmer_index({"corpus_window": seq}, k=K)
    mutated = _mutate_at(seq, [WINDOW_LENGTH // 2])

    hits = near_identity_hits(mutated, index, k=K, min_shared_kmers=MIN_SHARED_KMERS)

    guaranteed_floor = max(0, (WINDOW_LENGTH - K + 1) - 1 * K)
    assert guaranteed_floor > MIN_SHARED_KMERS  # sanity: the bound itself licenses detection
    assert "corpus_window" in hits
    assert hits["corpus_window"] >= guaranteed_floor


def test_dense_substitutions_at_the_documented_boundary_are_not_detected():
    # The documented degradation point: non-detection becomes ACHIEVABLE once
    # m >= (L - k + 1) / k, i.e. roughly one substitution every k bases. Placing
    # a substitution at every multiple of k (0, 16, 32, ...) guarantees every
    # k-length window contains exactly one mutated position (k consecutive
    # integers contain exactly one multiple of k), so every original k-mer is
    # destroyed and zero exact seeds can survive.
    rng = random.Random(5)
    seq = _random_sequence(rng, WINDOW_LENGTH)
    index = kmer_index({"corpus_window": seq}, k=K)
    positions = list(range(0, WINDOW_LENGTH, K))
    mutated = _mutate_at(seq, positions)

    # Confirm this construction actually sits at/past the documented boundary
    # before trusting the result below.
    assert len(positions) * K >= (WINDOW_LENGTH - K + 1)

    hits = near_identity_hits(mutated, index, k=K, min_shared_kmers=MIN_SHARED_KMERS)

    assert "corpus_window" not in hits


def test_orthogroup_exclusion_pulls_in_sibling_with_no_shared_id_prefix():
    # Human parent and its mouse ortholog: no shared ID prefix at all, only the
    # caller-supplied orthogroup label connects them.
    mapping = {
        "ENSG00000001": "FAM1",
        "ENSMUSG00000002": "FAM1",
        "ENSG00000099194": "FAM2",
    }

    key_parent = orthogroup_key("ENSG00000001.5", mapping)
    key_sibling = orthogroup_key("ENSMUSG00000002.1", mapping)
    key_unrelated = orthogroup_key("ENSG00000099194.8", mapping)

    assert key_parent == key_sibling
    assert key_parent != key_unrelated
    assert key_parent.startswith("orthogroup:")


def test_gene_absent_from_mapping_is_a_distinguishable_singleton():
    # Singleton-by-absence (no orthology information at all) must never be
    # confusable with singleton-by-mapping (mapping asserts a family of one).
    absent_key = orthogroup_key("ENSG_NOT_IN_ANY_MAPPING", {})
    mapped_singleton_key = orthogroup_key(
        "ENSG00000099194.8", {"ENSG00000099194": "FAM_OF_ONE"}
    )

    assert absent_key.startswith("singleton-absent:")
    assert mapped_singleton_key.startswith("orthogroup:")
    assert absent_key != mapped_singleton_key

    # Two different absent genes must not collide with each other either --
    # absence is not a shared group, it is "no information" per gene.
    assert orthogroup_key("GENE_A", {}) != orthogroup_key("GENE_B", {})


def test_version_normalization_collapses_versions_to_the_same_key():
    assert normalize_gene_id("ENSG1.2") == "ENSG1"
    assert normalize_gene_id("ENSG1.9") == "ENSG1"
    assert normalize_gene_id("ENSG1.2") == normalize_gene_id("ENSG1.9")

    mapping = {"ENSG1": "FAM_X"}
    assert orthogroup_key("ENSG1.2", mapping) == orthogroup_key("ENSG1.9", mapping)


def test_every_excluded_item_carries_a_correct_reason():
    rng = random.Random(6)
    parent_seq = _random_sequence(rng, WINDOW_LENGTH)
    unrelated_seq = _random_sequence(random.Random(7), WINDOW_LENGTH)  # see FP derivation above

    parents = {"PARENT1.2": parent_seq}
    orthogroup_mapping = {"PARENT1": "FAMX", "SIBLING1": "FAMX"}
    corpus_index = kmer_index(
        {
            "corpus_dup": parent_seq,  # exact duplicate of the parent window
            "corpus_unrelated": unrelated_seq,  # should NOT be excluded
        },
        k=K,
    )

    result = build_exclusion_set(
        parents,
        orthogroup_mapping=orthogroup_mapping,
        corpus_index=corpus_index,
        k=K,
        min_shared_kmers=MIN_SHARED_KMERS,
    )

    excluded = result["excluded"]
    assert set(excluded) == {"PARENT1", "SIBLING1", "corpus_dup"}
    assert "corpus_unrelated" not in excluded

    assert {r["reason"] for r in excluded["PARENT1"]} == {"direct_parent"}
    assert {r["reason"] for r in excluded["SIBLING1"]} == {"orthogroup_member"}
    assert {r["reason"] for r in excluded["corpus_dup"]} == {"homology_hit"}

    # Every reason entry is a fully-formed, human-auditable record.
    for reasons in excluded.values():
        assert reasons, "excluded identifier with no reason at all is not auditable"
        for entry in reasons:
            assert entry["reason"]
            assert entry["detail"]

    assert result["parent_ids"] == ["PARENT1"]
    assert result["k"] == K
    assert result["min_shared_kmers"] == MIN_SHARED_KMERS


def test_parent_absent_from_orthogroup_mapping_pulls_in_no_siblings():
    # A parent with no orthology information supplied is a singleton-by-absence
    # (see test_gene_absent_from_mapping_is_a_distinguishable_singleton) and
    # must not spuriously group with anything.
    rng = random.Random(8)
    parent_seq = _random_sequence(rng, WINDOW_LENGTH)
    parents = {"UNMAPPED_PARENT": parent_seq}
    corpus_index = kmer_index({}, k=K)

    result = build_exclusion_set(
        parents,
        orthogroup_mapping={"OTHER_GENE": "SOME_FAMILY"},
        corpus_index=corpus_index,
        k=K,
        min_shared_kmers=MIN_SHARED_KMERS,
    )

    assert set(result["excluded"]) == {"UNMAPPED_PARENT"}
    assert {r["reason"] for r in result["excluded"]["UNMAPPED_PARENT"]} == {"direct_parent"}
