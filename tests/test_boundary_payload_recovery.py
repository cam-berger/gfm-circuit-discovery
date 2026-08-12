from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "splicebert"
    / "run_bg2_boundary_payload_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("run_bg2_boundary_payload_recovery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

import audit_bg_boundary as AUDIT  # noqa: E402


def test_edited_attention_recovery_uses_edited_row_exactly():
    attention_edit = torch.tensor([[0.2, 0.3, 0.1, 0.4]])
    residual_delta = torch.tensor(
        [[[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 1.0]]]
    )
    value_weight = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-2.0, 0.25]])
    natural_edit = torch.tensor([[0.25, -0.5, 1.0]])

    delivered = residual_delta @ value_weight.T
    expected_delta = (attention_edit[..., None] * delivered).sum(dim=1)
    expected = natural_edit + expected_delta

    actual_delta = MODULE.edited_attention_delta(
        attention_edit, value_weight, residual_delta
    )
    actual = MODULE.edited_attention_recovery(
        natural_edit, attention_edit, value_weight, residual_delta
    )
    assert torch.allclose(actual_delta, expected_delta)
    assert torch.allclose(actual, expected)


def test_donor_patch_has_only_requested_head_and_token():
    recovered = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source, mask = MODULE.build_donor_patch(
        recovered,
        donor_tokens=torch.tensor([2, 4]),
        num_heads=3,
        sequence_length=6,
        head=1,
    )
    assert source.shape == (2, 3, 6, 2)
    assert mask.shape == (2, 6)
    assert torch.equal(source[0, 1, 2], recovered[0])
    assert torch.equal(source[1, 1, 4], recovered[1])
    assert int(torch.count_nonzero(source)) == 4
    assert torch.equal(mask.nonzero(), torch.tensor([[0, 2], [1, 4]]))


def test_norm_match_is_per_example_and_refuses_zero_energy():
    candidate = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    target = torch.tensor([[0.0, 10.0], [6.0, 8.0]])
    matched, scale = MODULE.norm_match(candidate, target)
    assert torch.allclose(matched.norm(dim=-1), target.norm(dim=-1))
    assert torch.allclose(scale, torch.tensor([2.0, 5.0]))

    with pytest.raises(ValueError, match="zero-energy candidate"):
        MODULE.norm_match(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="zero-energy target"):
        MODULE.norm_match(torch.ones(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="finite"):
        MODULE.norm_match(torch.tensor([[float("nan"), 1.0]]), torch.ones(1, 2))


def test_signed_coordinate_permutation_preserves_spatial_gram():
    delta = torch.arange(2 * 4 * 7, dtype=torch.float32).reshape(2, 4, 7) - 8.0
    sham = MODULE.signed_coordinate_permutation(delta, seed=20260809)
    assert not torch.equal(delta, sham)
    assert torch.allclose(
        delta @ delta.transpose(-1, -2),
        sham @ sham.transpose(-1, -2),
    )
    assert torch.allclose(
        delta.square().sum(dim=-1), sham.square().sum(dim=-1)
    )


def test_parameter_fingerprint_covers_every_parameter_and_detects_mutation():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    before = MODULE.model_parameter_sha256(model)
    assert before == MODULE.model_parameter_sha256(model)
    with torch.no_grad():
        model[1].weight[0, 0].add_(1.0)
    assert MODULE.model_parameter_sha256(model) != before


def test_crossgene_map_stays_within_pair_strand_and_span_stratum():
    records = []
    for pair_id, strand, spans in (
        ("minus3_T_to_A__p1_A", "+", (70, 80)),
        ("plus1_G_to_A__m3_C", "-", (110, 130)),
    ):
        for index, span in enumerate(spans):
            records.append({
                "pair_id": pair_id,
                "strand": strand,
                "partner_span": span,
                "gene_id": f"{pair_id}-gene-{index}",
                "example_id": f"{pair_id}-example-{index}",
            })
    mapping = MODULE.crossgene_source_map(records)
    for target_index, source_index in mapping.items():
        target, source = records[target_index], records[source_index]
        assert source["gene_id"] != target["gene_id"]
        assert source["pair_id"] == target["pair_id"]
        assert source["strand"] == target["strand"]
        assert MODULE.coarse_span_bin(source["partner_span"]) == MODULE.coarse_span_bin(
            target["partner_span"]
        )


def test_target_membership_must_equal_preflight_census():
    examples = [
        SimpleNamespace(example_id="e1", gene_id="g1", strand="+"),
        SimpleNamespace(example_id="e2", gene_id="g2", strand="-"),
    ]

    class FakeBoundaryGrammar:
        @staticmethod
        def cohort_sha256(_examples):
            return "b" * 64

    membership = {
        "ids": ["e1", "e2"],
        "genes": ["g1", "g2"],
        "strands": ["+", "-"],
        "sha256": "b" * 64,
    }
    preflight = {"target_cohorts": {"val": {"target_route": membership}}}
    assert MODULE.validate_target_membership(
        FakeBoundaryGrammar, preflight, examples, split="val"
    ) == membership

    preflight["target_cohorts"]["val"]["target_route"]["ids"] = ["e2", "e1"]
    with pytest.raises(RuntimeError, match="membership"):
        MODULE.validate_target_membership(
            FakeBoundaryGrammar, preflight, examples, split="val"
        )


def _write(path: Path, text: str) -> str:
    path.write_text(text)
    return MODULE.sha256_file(path)


def _passing_preflight(tmp_path: Path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    protocol = tmp_path / "protocol.md"
    amendment = tmp_path / "amendment.md"
    declaration = tmp_path / "boundary_grammar.py"
    gtf = tmp_path / "annotation.gtf.gz"
    fasta = tmp_path / "genome.fa"
    model_hashes = {
        name: _write(checkpoint / name, f"model artifact: {name}")
        for name in MODULE.MODEL_FILES
    }
    checkpoint_hash = model_hashes["pytorch_model.bin"]
    protocol_hash = _write(protocol, "frozen protocol")
    amendment_hash = _write(amendment, "binding pre-execution amendment")
    declaration_hash = _write(declaration, "declaration")
    gtf_hash = _write(gtf, "annotation")
    fasta_hash = _write(fasta, "genome")
    expected_config = {
        "namespace": "bg-v1",
        "role_weights": [["behavior", 3], ["route", 2], ["reserve", 1]],
        "sesoi": {"linear": 0.02},
        "panel": {"positions": [-3, 1]},
        "contrasts": {"minus3": [1, -1]},
    }
    payload = {
        "schema": MODULE.PREFLIGHT_SCHEMA,
        "PASS": True,
        "instrument_PASS": True,
        "target_authorization_PASS": True,
        "smoke": False,
        "checkpoint_sha256": checkpoint_hash,
        "model_file_sha256": model_hashes,
        "gtf_sha256": gtf_hash,
        "fasta_sha256": fasta_hash,
        "protocol_sha256": protocol_hash,
        "amendment_sha256": amendment_hash,
        "declaration_sha256": declaration_hash,
        "config": expected_config,
        "cohort": {
            "split": "train",
            "ids": {"behavior": ["example-b"], "route": ["example-r"]},
            "genes": {"behavior": ["gene-b"], "route": ["gene-r"]},
            "sha256": "a" * 64,
            "counts": {"behavior": 1, "route": 1},
        },
        "checks": {"positive": {"PASS": True}, "negative": {"PASS": True}},
    }
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(payload))
    return (
        preflight,
        checkpoint,
        protocol,
        declaration,
        expected_config,
        gtf,
        fasta,
        amendment,
    )


def test_preflight_is_hash_bound_and_passed(tmp_path):
    args = _passing_preflight(tmp_path)
    loaded = MODULE.validate_preflight(*args)
    assert loaded["PASS"] is True

    args[2].write_text("protocol drift")
    with pytest.raises(RuntimeError, match="protocol_sha256"):
        MODULE.validate_preflight(*args)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [("PASS", False, "PASS"), ("smoke", True, "smoke")],
)
def test_preflight_refuses_failed_or_smoke_artifact(tmp_path, field, value, match):
    (
        preflight, checkpoint, protocol, declaration, config, gtf, fasta, amendment
    ) = _passing_preflight(tmp_path)
    payload = json.loads(preflight.read_text())
    payload[field] = value
    preflight.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match=match):
        MODULE.validate_preflight(
            preflight, checkpoint, protocol, declaration, config, gtf, fasta, amendment
        )


def test_preflight_refuses_declaration_or_config_drift(tmp_path):
    (
        preflight, checkpoint, protocol, declaration, config, gtf, fasta, amendment
    ) = _passing_preflight(tmp_path)
    declaration.write_text("changed declaration")
    with pytest.raises(RuntimeError, match="declaration_sha256"):
        MODULE.validate_preflight(
            preflight, checkpoint, protocol, declaration, config, gtf, fasta, amendment
        )

    declaration.write_text("declaration")
    changed = {**config, "namespace": "bg-v2"}
    with pytest.raises(RuntimeError, match="config"):
        MODULE.validate_preflight(
            preflight, checkpoint, protocol, declaration, changed, gtf, fasta, amendment
        )


def test_bg2_main_contract_binds_frozen_supply_constants_before_model_load():
    fake = SimpleNamespace(
        ANALYSIS_SEED=MODULE.BOOT_SEED,
        BG2_BOOTSTRAP_DRAWS=MODULE.N_BOOT_DEFAULT,
        MIN_BG2_TARGET_GENES=MODULE.MIN_TARGET_ROUTE_GENES,
        MIN_BG2_TARGET_GENES_PER_STRAND=MODULE.MIN_TARGET_ROUTE_PER_STRAND,
        preflight_config=lambda: {
            "analysis": {"signed_coordinate_shams": len(MODULE.SIGNED_SHAM_SEEDS)}
        },
    )
    MODULE.validate_runtime_contract(fake)

    fake.MIN_BG2_TARGET_GENES += 1
    with pytest.raises(RuntimeError, match="supply constants"):
        MODULE.validate_runtime_contract(fake)


def test_crossgene_interval_calls_two_way_target_source_bootstrap(monkeypatch):
    calls = []

    def fake_bootstrap(values, genes_a, genes_b, **kwargs):
        calls.append((list(genes_a), list(genes_b), kwargs["alpha"]))
        return float(sum(values) / len(values)), 0.1 + kwargs["alpha"], 0.9

    monkeypatch.setattr(MODULE, "dyadic_bootstrap", fake_bootstrap)
    summary = MODULE._dyadic_mean_ci(
        MODULE.np.asarray([0.2, 0.4, 0.6]),
        ["target-1", "target-2", "target-3"],
        ["source-2", "source-2", "source-1"],
        n_boot=101,
        seed=17,
    )
    assert calls == [
        (
            ["target-1", "target-2", "target-3"],
            ["source-2", "source-2", "source-1"],
            0.05,
        ),
        (
            ["target-1", "target-2", "target-3"],
            ["source-2", "source-2", "source-1"],
            0.10,
        ),
    ]
    assert summary["lower_95_one_sided"] == pytest.approx(0.2)
    assert summary["n_dyads"] == 3
    assert summary["n_source_genes"] == 2
    assert summary["inference"] == "two_way_target_source_gene_multiplier_bootstrap"


def test_independent_auditor_exactly_mirrors_dyadic_interval(monkeypatch):
    values = MODULE.np.asarray([0.8, -0.1, 0.4, 0.9, 0.2])
    targets = ["t1", "t2", "t3", "t4", "t5"]
    sources = ["s1", "s1", "s2", "s3", "s2"]
    monkeypatch.setattr(AUDIT, "BG2_BOOTSTRAP_DRAWS", 257)
    runner = MODULE._dyadic_mean_ci(
        values, targets, sources, n_boot=257, seed=20260819
    )
    independently_audited = AUDIT._dyadic_mean_ci(
        values, targets, sources, seed=20260819
    )
    assert independently_audited == runner


def _passing_behavior(tmp_path: Path):
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}")
    config = {"namespace": "bg-v1", "sesoi": 0.05}
    primary_ids = ("m3_T_minus_A", "p1_G_minus_A")
    payload = {
        "schema": MODULE.BEHAVIOR_SCHEMA,
        "status": "VERIFICATION",
        "preflight_path": str(preflight),
        "preflight_sha256": MODULE.sha256_file(preflight),
        "config": config,
        "cohort": {
            "split": "val",
            "memberships": {
                "target_behavior": {
                    "ids": ["behavior-example"],
                    "genes": ["behavior-gene"],
                }
            },
        },
        "target": {
            "verdict": "SUPPORTED_VERIFICATION",
            "primary_ids": list(primary_ids),
        },
    }
    behavior = tmp_path / "behavior.json"
    behavior.write_text(json.dumps(payload))
    return behavior, preflight, config, primary_ids


def test_behavior_result_must_be_supported_and_bound_to_same_preflight(tmp_path):
    behavior, preflight, config, primary_ids = _passing_behavior(tmp_path)
    loaded = MODULE.validate_behavior_result(
        behavior,
        preflight_path=preflight,
        expected_config=config,
        expected_primary_ids=primary_ids,
    )
    assert loaded["target"]["verdict"] == "SUPPORTED_VERIFICATION"

    payload = json.loads(behavior.read_text())
    payload["target"]["verdict"] = "INCONCLUSIVE"
    behavior.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="does not authorize"):
        MODULE.validate_behavior_result(
            behavior,
            preflight_path=preflight,
            expected_config=config,
            expected_primary_ids=primary_ids,
        )


def test_behavior_result_refuses_preflight_hash_drift(tmp_path):
    behavior, preflight, config, primary_ids = _passing_behavior(tmp_path)
    preflight.write_text("changed")
    with pytest.raises(RuntimeError, match="sha256"):
        MODULE.validate_behavior_result(
            behavior,
            preflight_path=preflight,
            expected_config=config,
            expected_primary_ids=primary_ids,
        )


def test_behavior_audit_must_equal_fresh_recomputation(tmp_path, monkeypatch):
    result = tmp_path / "behavior.json"
    manifest = tmp_path / "behavior.manifest.json"
    saved = tmp_path / "behavior.audit.json"
    result.write_text("{}")
    manifest.write_text("{}")
    expected = {
        "schema": MODULE.AUDIT_SCHEMA,
        "PASS": True,
        "audited_schema": MODULE.BEHAVIOR_SCHEMA,
        "sources": {},
        "checks": {"target": True},
    }
    saved.write_text(json.dumps(expected))
    monkeypatch.setattr(MODULE, "build_audit", lambda *_args: expected)
    assert MODULE.validate_behavior_audit(result, manifest, saved) == expected

    saved.write_text(json.dumps({**expected, "PASS": False}))
    with pytest.raises(RuntimeError, match="differs"):
        MODULE.validate_behavior_audit(result, manifest, saved)
