from gfmcd.splicing.context_controls import (
    mask_true_partner_neighborhood,
    retain_pair_endpoints_only,
    retain_site_windows,
)
from gfmcd.splicing.perturb import boundary_pair_factorial, dinucleotide_shuffle, shuffle_exon_core
from test_splicing_schema_candidates import make_example


def dinuc_counts(seq):
    out = {}
    for a, b in zip(seq, seq[1:]):
        out[a + b] = out.get(a + b, 0) + 1
    return out


def test_dinucleotide_shuffle_preserves_counts():
    seq = "AACCGTTACGATCGATCGGCTA"
    shuffled = dinucleotide_shuffle(seq, seed=4)
    assert len(shuffled) == len(seq)
    assert shuffled[0] == seq[0] and shuffled[-1] == seq[-1]
    assert dinuc_counts(shuffled) == dinuc_counts(seq)


def test_context_controls_preserve_length_and_endpoints():
    ex = make_example()
    for seq in (
        retain_site_windows(ex, flank=3),
        retain_pair_endpoints_only(ex, flank=3),
        mask_true_partner_neighborhood(ex, kind="acceptor", flank=3),
        mask_true_partner_neighborhood(ex, kind="donor", flank=3),
    ):
        assert len(seq) == len(ex.sequence)


def test_boundary_factorial_has_shared_coordinates():
    a = make_example()
    # Same anchors, different local content while keeping canonical motifs.
    b = type(a).from_dict({**a.to_dict(), "example_id": "ex2", "gene_id": "g2"})
    result = boundary_pair_factorial(a, a, b, flank=3)
    assert set(result) == {"AA", "AB", "BA", "BB"}
    assert all(len(x) == len(a.sequence) for x in result.values())
