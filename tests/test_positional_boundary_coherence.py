from __future__ import annotations

import copy

import pytest

from gfmcd.splicing import positional_boundary_coherence as pbc


def _checks() -> dict[str, bool]:
    return {"unambiguous_reference": True, "canonical_motif": True}


def _annotation(phase: int) -> dict:
    return {
        "phase_convention": "intron_phase",
        "raw_gtf_frame": str(phase),
        "upstream_cds_end": 1000 + phase,
        "downstream_cds_start": 2000 + phase,
        "basic_phase": phase,
        "comprehensive_phase": phase,
        "basic_support_ids": [f"basic-{phase}"],
        "comprehensive_support_ids": [f"comprehensive-{phase}"],
        "checks": {
            "upstream_downstream_phase_equal": True,
            "basic_transcript_coordinate_agreement": True,
            "comprehensive_no_coding_conflict": True,
            "comprehensive_no_strand_conflict": True,
            "coding_offset_consistent_with_gtf_frame": True,
        },
    }


def _with_motif(length: int, start: int, motif: str, fill: str = "A") -> str:
    sequence = list(fill * length)
    sequence[start : start + len(motif)] = motif
    return "".join(sequence)


def synthetic_candidate_package(n_per_pool: int = 100) -> dict:
    candidates = []
    gene_number = 1

    def source_fields() -> dict:
        nonlocal gene_number
        number = gene_number
        gene_number += 1
        return {
            "gene_id": f"ENSG{number:011d}.7",
            "chrom": "chr1",
            "strand": "+",
            "source_start": number * 1000,
            "source_end": number * 1000 + 200,
            "source_exon_length": 180,
            "transcript_id": f"ENST{number:011d}.2",
            "eligibility_checks": _checks(),
        }

    scaffold_sequence = _with_motif(
        pbc.DEFAULT_CONSTRUCT_LENGTH,
        pbc.DEFAULT_BRIDGE_ACCEPTOR_POS - 2,
        "AG",
        fill="C",
    )
    for index in range(n_per_pool):
        candidates.append(
            {
                "candidate_id": f"scaffold-{index:04d}",
                "role": "scaffold",
                "sequence": scaffold_sequence,
                "donor_window": list(pbc.DEFAULT_DONOR_WINDOW),
                "acceptor_window": list(pbc.DEFAULT_ACCEPTOR_WINDOW),
                "donor_pos": pbc.DEFAULT_DONOR_POS,
                "acceptor_pos": pbc.DEFAULT_ACCEPTOR_POS,
                "bridge_acceptor_pos": pbc.DEFAULT_BRIDGE_ACCEPTOR_POS,
                "geometry": {"bridge_length": 34.0},
                "scores": {},
                **source_fields(),
            }
        )
    donor_sequence = _with_motif(126, 90, "GT")
    acceptor_sequence = _with_motif(140, 48, "AG")
    native_sequence = list("A" * pbc.DEFAULT_CONSTRUCT_LENGTH)
    native_sequence[pbc.DEFAULT_DONOR_POS : pbc.DEFAULT_DONOR_POS + 2] = list("GT")
    native_sequence[
        pbc.DEFAULT_ACCEPTOR_POS - 2 : pbc.DEFAULT_ACCEPTOR_POS
    ] = list("AG")
    native_sequence = "".join(native_sequence)
    for role, sequence, boundary in (
        ("donor", donor_sequence, 90),
        ("acceptor", acceptor_sequence, 50),
    ):
        for phase in (0, 1, 2):
            for index in range(n_per_pool):
                candidates.append(
                    {
                        "candidate_id": f"{role}-p{phase}-{index:04d}",
                        "role": role,
                        "phase": phase,
                        "sequence": sequence,
                        "boundary_index": boundary,
                        "coding_offset": 0,
                        "native_sequence": native_sequence,
                        "native_donor_pos": pbc.DEFAULT_DONOR_POS,
                        "native_acceptor_pos": pbc.DEFAULT_ACCEPTOR_POS,
                        "geometry": {"module_span": float(len(sequence))},
                        "scores": {"maxent": 1.0},
                        "annotation": _annotation(phase),
                        **source_fields(),
                    }
                )
    donor_anchor = pbc.extract_module_covariates(
        next(row for row in candidates if row["role"] == "donor")
    )
    acceptor_anchor = pbc.extract_module_covariates(
        next(row for row in candidates if row["role"] == "acceptor")
    )
    for scaffold in (row for row in candidates if row["role"] == "scaffold"):
        scaffold["donor_anchor_covariates"] = donor_anchor
        scaffold["acceptor_anchor_covariates"] = acceptor_anchor
    return {
        "schema": pbc.CANDIDATE_SCHEMA,
        "candidate_mode": "synthetic_fixture",
        "model_outcomes_read": False,
        "candidates": candidates,
    }


@pytest.fixture(scope="module")
def small_package() -> tuple[dict, dict, dict]:
    candidates = synthetic_candidate_package(n_per_pool=40)
    variance = {"variance": 0.0, "sesoi": 0.05, "alpha": 0.05, "power": 0.8}
    cohort = pbc.build_composition_balanced_cohort(
        candidates,
        blacklisted_gene_ids=set(),
        frozen_variance=variance,
        requested_blocks_by_split={
            "development": 5,
            "confirmation": 5,
            "phase3_reserve": 5,
        },
        minimum_confirmation_blocks=5,
        allowed_chromosomes={"chr1"},
        forbidden_chromosomes={"chr8", "chr9"},
        global_search_width=4,
        inner_folds=2,
        auc_n_boot=4,
        pseudo_family_size=1,
        minimum_stratum_blocks=0,
    )
    return candidates, variance, cohort


def test_candidate_validation_fails_closed_on_phase_provenance_and_support():
    package = synthetic_candidate_package(n_per_pool=1)
    donor = next(row for row in package["candidates"] if row["role"] == "donor")
    missing = copy.deepcopy(donor)
    missing.pop("annotation")
    with pytest.raises(ValueError, match="annotation provenance"):
        pbc.validate_candidate(missing)

    short = copy.deepcopy(donor)
    short["sequence"] = _with_motif(100, 70, "GT")
    short["boundary_index"] = 70
    with pytest.raises(ValueError, match="lacks complete donor exonic support"):
        pbc.validate_candidate(short)


def test_outcome_firewall_rejects_model_quantities():
    package = synthetic_candidate_package(n_per_pool=1)
    package["candidates"][0]["native_margin"] = 1.2
    with pytest.raises(pbc.CohortConstructionError, match="outcome-bearing"):
        pbc.validate_candidate_package(package)


def test_exact_role_specific_feature_support_and_stop_registers():
    package = synthetic_candidate_package(n_per_pool=1)
    donor = pbc.validate_candidate(next(row for row in package["candidates"] if row["role"] == "donor"))
    acceptor = pbc.validate_candidate(
        next(row for row in package["candidates"] if row["role"] == "acceptor")
    )
    scaffold = pbc.validate_candidate(package["candidates"][0])
    donor_features = pbc.extract_module_covariates(donor)
    acceptor_features = pbc.extract_module_covariates(acceptor)
    assert "core.m3.A" in donor_features
    assert "core.p5.A" in donor_features
    assert "core.m20.A" in acceptor_features
    assert "core.p2.A" in acceptor_features
    assert donor_features["exonic.length"] == 90
    assert donor_features["proximal_intronic.length"] == 30
    assert acceptor_features["proximal_intronic.length"] == 30
    assert acceptor_features["exonic.length"] == 90
    assert "coding.tri_pos2.AAA" in donor_features
    cell = pbc.extract_cell_covariates(donor, acceptor, scaffold)
    for register in (0, 1, 2):
        assert f"pair.joined_register{register}_stop_count" in cell


def test_joint_block_roles_are_hashed_once_and_genes_are_globally_disjoint(small_package):
    _candidates, _variance, cohort = small_package
    genes = []
    assert cohort["counts"]["blocks_by_split"] == {
        "confirmation": 5,
        "development": 5,
        "phase3_reserve": 5,
    }
    for block in cohort["blocks"]:
        source_genes = block["source_gene_ids"]
        assert len(source_genes) == 7
        assert pbc.stable_block_split(source_genes) == block["split_role"]
        genes.extend(source_genes)
    assert len(genes) == len(set(genes)) == 7 * len(cohort["blocks"])
    # Individual genes are not required to hash to the block role.
    assert any(
        pbc.stable_split(gene) != block["split_role"]
        for block in cohort["blocks"]
        for gene in block["source_gene_ids"]
    )


def test_cells_pin_geometry_variants_and_complete_three_by_three(small_package):
    _candidates, _variance, cohort = small_package
    for block in cohort["blocks"]:
        cells = [cell for cell in cohort["cells"] if cell["block_id"] == block["block_id"]]
        assert len(cells) == 9
        assert {(cell["donor_phase"], cell["acceptor_phase"]) for cell in cells} == {
            (donor, acceptor) for donor in range(3) for acceptor in range(3)
        }
        for cell in cells:
            assert len(cell["sequence_intact"]) == 1022
            assert cell["donor_pos"] == 392
            assert cell["acceptor_pos"] == 512
            assert cell["sequence_intact"][510:512] == "AG"
            assert cell["sequence_acceptor_ac"][510:512] == "AC"
            assert cell["sequence_acceptor_at"][510:512] == "AT"
            assert cell["sequence_intact"][392:394] == "GT"
            assert cell["delta_class"] in {"M0", "Mplus", "Mminus"}


def test_balance_family_has_all_three_contrasts_and_balanced_outer_folds(small_package):
    _candidates, _variance, cohort = small_package
    audit = cohort["balance_audit_by_split"]["confirmation"]
    assert audit["smd"]["contrasts"] == [
        "M0_minus_Mplus",
        "M0_minus_Mminus",
        "Mplus_minus_Mminus",
    ]
    assert audit["smd"]["n_family"] == 3 * audit["smd"]["n_features"]
    assert set(audit["smd"]["max_abs_smd_by_role"]) >= {"donor", "acceptor", "pair"}
    folds = audit["auc"]["outer_fold_stratum_counts"]
    assert len(folds) == 5
    assert sum(counts.get("chr1|+", 0) for counts in folds) == 5
    assert max(counts.get("chr1|+", 0) for counts in folds) == 1


def test_sample_size_uses_only_frozen_variance():
    plan = pbc.plan_sample_size(
        {"variance": 0.04, "sesoi": 0.05, "alpha": 0.05, "power": 0.8},
        minimum_blocks=10,
    )
    assert plan.calculated_blocks > 10
    assert plan.planned_blocks == plan.calculated_blocks


def test_independent_audit_reconstructs_and_detects_tampering():
    candidates = synthetic_candidate_package(n_per_pool=20)
    variance = {"variance": 0.0, "sesoi": 0.05, "alpha": 0.05, "power": 0.8}
    cohort = pbc.build_composition_balanced_cohort(
        candidates,
        blacklisted_gene_ids=set(),
        frozen_variance=variance,
        requested_blocks_by_split={
            "development": 5,
            "confirmation": 0,
            "phase3_reserve": 0,
        },
        minimum_confirmation_blocks=1,
        allowed_chromosomes={"chr1"},
        forbidden_chromosomes={"chr8", "chr9"},
        global_search_width=4,
        inner_folds=2,
        auc_n_boot=2,
        pseudo_family_size=1,
        minimum_stratum_blocks=0,
    )
    audit = pbc.audit_composition_balanced_cohort(
        cohort,
        candidates,
        blacklisted_gene_ids=set(),
        frozen_variance=variance,
    )
    assert audit["passed"] is True
    assert audit["target_execution_authorized"] is False  # synthetic pseudo family has n=1, not 99

    tampered = copy.deepcopy(cohort)
    tampered["cells"][0]["sequence_intact"] = "A" + tampered["cells"][0]["sequence_intact"][1:]
    audit = pbc.audit_composition_balanced_cohort(
        tampered,
        candidates,
        blacklisted_gene_ids=set(),
        frozen_variance=variance,
    )
    assert audit["passed"] is False
    assert audit["checks"]["sequence_and_covariate_reconstruction"] is False


def test_blacklist_is_version_normalized_and_hard():
    assert pbc.load_blacklist({"gene_ids": ["ENSG000001.12"]}) == {"ENSG000001"}
