from dataclasses import dataclass

import numpy as np
import pytest

from gfmcd.splicing.biological_features import (
    PsamComponent,
    RbpPsam,
    beam_synonymous_optimize,
    codon_log_preference,
    codon_amino_acid,
    complete_codon_starts,
    feature_delta,
    maximal_synonymous_recode,
    kmer_count_vector,
    parse_rbns_psam,
    psam_region_score,
    synonymous_saturation_variants,
    vienna_unpaired_mean,
)
from gfmcd.splicing.context_localization import RelativeWindow


@dataclass
class Example:
    example_id: str = "example"
    sequence: str = "AAAGAAGAATTT"
    gene_id: str = "gene"


def coding_slots(monkeypatch):
    monkeypatch.setattr(
        "gfmcd.splicing.biological_features.window_codon_slot",
        lambda _example, position, _lookup: position % 3,
    )


def test_synonymous_saturation_preserves_amino_acids(monkeypatch):
    coding_slots(monkeypatch)
    example = Example()
    window = RelativeWindow(0, len(example.sequence))
    variants = synonymous_saturation_variants(
        example,
        window,
        {},
        anchor=0,
    )
    assert complete_codon_starts(example, window, {}, anchor=0) == (0, 3, 6, 9)
    assert variants
    for variant in variants:
        assert codon_amino_acid(variant.original_codon) == codon_amino_acid(
            variant.alternate_codon
        )
        assert variant.sequence != example.sequence
        assert variant.hamming == sum(
            left != right
            for left, right in zip(example.sequence, variant.sequence)
        )


def test_maximal_recode_is_deterministic_and_preserves_translation(monkeypatch):
    coding_slots(monkeypatch)
    example = Example()
    window = RelativeWindow(0, len(example.sequence))
    first, changes_first = maximal_synonymous_recode(
        example,
        window,
        {},
        anchor=0,
        draw=3,
    )
    second, changes_second = maximal_synonymous_recode(
        example,
        window,
        {},
        anchor=0,
        draw=3,
    )
    assert first == second
    assert changes_first == changes_second
    assert first != example.sequence
    for start in range(0, len(first), 3):
        assert codon_amino_acid(first[start : start + 3]) == codon_amino_acid(
            example.sequence[start : start + 3]
        )


def test_maximal_recode_can_forbid_new_splice_dinucleotides(monkeypatch):
    coding_slots(monkeypatch)
    example = Example(sequence="GCTGCTGCTGCT")
    recoded, edits = maximal_synonymous_recode(
        example,
        RelativeWindow(0, 12),
        {},
        anchor=0,
        draw=0,
        forbid_new_splice_dinucleotides=True,
    )
    clean_positions = {
        index
        for index in range(11)
        if example.sequence[index : index + 2] in {"AG", "GT"}
    }
    recoded_positions = {
        index for index in range(11) if recoded[index : index + 2] in {"AG", "GT"}
    }
    assert recoded_positions <= clean_positions
    assert edits


def test_parse_and_score_multicomponent_psam(tmp_path):
    path = tmp_path / "RBP.txt"
    path.write_text(
        "\n".join(
            [
                "# PSAM 0",
                "PSAM  A0=2 n=2 acc_k=1",
                "# A C G U",
                "1 0 0 0 A",
                "0 1 0 0 C",
                "# PSAM 1",
                "PSAM  A0=0.5 n=1 acc_k=1",
                "0 0 1 0 G",
            ]
        )
    )
    model = parse_rbns_psam(path)
    assert model.name == "RBP"
    assert len(model.components) == 2
    # AC contributes 2 at offset 0; G contributes .5 at offset 2.
    assert psam_region_score("ACG", model, start=0, end=3) == pytest.approx(2.5)
    assert psam_region_score("TTT", model, start=0, end=3) == 0.0


def test_psam_validation_and_feature_delta():
    with pytest.raises(ValueError, match="shape"):
        PsamComponent(scale=1.0, matrix=np.ones((2, 3)))
    model = RbpPsam(
        name="toy",
        components=(PsamComponent(scale=1.0, matrix=np.ones((2, 4))),),
        source_path="toy",
    )
    assert psam_region_score("AAAA", model, start=0, end=4) == 3.0
    assert feature_delta({"a": 4.0}, {"a": 1.5}) == {"a": 2.5}


def test_synonymous_beam_optimizes_sequence_only_objective(monkeypatch):
    coding_slots(monkeypatch)
    example = Example(sequence="GAAGAA")
    window = RelativeWindow(0, 6)
    # Lys/Glu synonyms permit the final base of each codon to become G.
    sequence, audit = beam_synonymous_optimize(
        example,
        window,
        {},
        anchor=0,
        objective=lambda value: float(value.count("G")),
        maximize=True,
        max_changed_bases=2,
        beam_size=16,
    )
    assert sequence == "GAGGAG"
    assert audit["changed_bases"] == 2
    assert audit["objective_changed"] > audit["objective_clean"]


def test_vienna_unpaired_mean_requires_a_valid_region():
    with pytest.raises(ValueError, match="invalid"):
        vienna_unpaired_mean("ACGT", start=0, end=3, unpaired_length=7)


def test_kmer_counts_and_codon_preference():
    values, names = kmer_count_vector("ACAC", min_k=1, max_k=2)
    counts = dict(zip(names, values))
    assert counts["A"] == 2
    assert counts["C"] == 2
    assert counts["AC"] == 2
    assert counts["CA"] == 1
    preference = codon_log_preference(
        "GAAGAG",
        (0, 3),
        {"GAA": 9, "GAG": 1},
    )
    assert np.isfinite(preference)
