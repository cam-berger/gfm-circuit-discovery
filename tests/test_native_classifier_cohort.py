import pytest

from gfmcd.splicing.native_classifier import gene_hash_partition


def test_gene_hash_partition_is_deterministic_and_namespaced():
    first = gene_hash_partition(
        "ENSG000001",
        namespace="context-head-census-v1",
    )
    assert first in {"discovery", "confirmation"}
    assert first == gene_hash_partition(
        "ENSG000001",
        namespace="context-head-census-v1",
    )

    assignments = {
        gene_hash_partition(
            f"ENSG{index:06d}",
            namespace="context-head-census-v1",
        )
        for index in range(100)
    }
    assert assignments == {"discovery", "confirmation"}


def test_gene_hash_partition_rejects_invalid_contract():
    with pytest.raises(ValueError):
        gene_hash_partition("gene", namespace="")
    with pytest.raises(ValueError):
        gene_hash_partition("gene", namespace="study", discovery_percent=0)
    with pytest.raises(ValueError):
        gene_hash_partition("gene", namespace="study", discovery_percent=100)
