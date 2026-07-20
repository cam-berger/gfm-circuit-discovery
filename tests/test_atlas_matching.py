from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_head_context_atlas/build_atlas_matched_cohorts.py"
SPEC = importlib.util.spec_from_file_location("atlas_match", SCRIPT)
atlas_match = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(atlas_match)


def row(*, tags: list[str], role: str = "donor", support: int = 2,
        junction_support: int = 1) -> dict:
    return {
        "tags": tags,
        "target_role": role,
        "strand": "+",
        "target_boundary_region": "CDS",
        "distance_mod3": 1,
        "target_partner_distance": 171,
        "n_annotated_transcripts_for_target_site": support,
        "n_annotated_transcripts_for_junction": junction_support,
    }


def test_cluster_categories_are_role_specific() -> None:
    assert atlas_match.CATEGORY_ROLES["donor_cluster"] == ("donor",)
    assert atlas_match.CATEGORY_ROLES["acceptor_cluster"] == ("acceptor",)
    assert atlas_match.CATEGORY_ROLES["single_pair"] == ("donor", "acceptor")


def test_exact_stratum_matches_all_non_focal_context_tags() -> None:
    case = row(tags=["donor_cluster", "cassette_exon", "complex_event"])
    same = row(tags=["cassette_exon", "complex_event"])
    different = row(tags=["complex_event"])
    assert atlas_match.exact_stratum(case, "donor_cluster") == atlas_match.exact_stratum(
        same, "donor_cluster"
    )
    assert atlas_match.exact_stratum(case, "donor_cluster") != atlas_match.exact_stratum(
        different, "donor_cluster"
    )


def test_exact_stratum_preserves_support_and_distance_bands() -> None:
    base = row(tags=["single_pair"], support=2, junction_support=1)
    same_bands = row(tags=[], support=2, junction_support=1)
    other_support = row(tags=[], support=6, junction_support=1)
    other_distance = {**same_bands, "target_partner_distance": 222}
    assert atlas_match.exact_stratum(base, "single_pair") == atlas_match.exact_stratum(
        same_bands, "single_pair"
    )
    assert atlas_match.exact_stratum(base, "single_pair") != atlas_match.exact_stratum(
        other_support, "single_pair"
    )
    assert atlas_match.exact_stratum(base, "single_pair") != atlas_match.exact_stratum(
        other_distance, "single_pair"
    )
