"""Mechanical-gate tests for the all-96-head A/B/C/D screen.

Fast tests (no model) cover metric math + destroy/coordinate gates. A guarded model test verifies the
vectorized (batched-examples) and scalar (batch=1) head interventions agree (gate 6).
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "splicebert_head_context_atlas"))

import torch  # noqa: E402
from run_all_head_screen import (  # noqa: E402
    metrics_from_logits,
    gate_destroy_real_bases,
    collateral_health,
    cohort_example_allowed,
    target_base_position,
)
from gfmcd.splicing.competition_seq import WindowedCompetition  # noqa: E402
from run_group_mlp_closure import (  # noqa: E402
    ALL_HEADS,
    layer_matched_random_sets,
    load_rankings,
    parse_head,
    parse_ints,
)
from run_atlas_model_preflight import (  # noqa: E402
    CORE_CATEGORIES,
    health_probes,
    partner_destroyed,
    select_cover_pairs,
)


def test_metrics_from_logits():
    # base_ids A,C,G,T -> 0,1,2,3; native = G (id 2); logits favour G
    logits = torch.tensor([0.0, 1.0, 4.0, 0.5, -9.9])  # index 4 is a non-base token, ignored
    m = metrics_from_logits(logits, 2, [0, 1, 2, 3])
    assert abs(m["margin"] - (4.0 - 1.0)) < 1e-5           # logit(G) - max(A,C,T)=logit(C)=1
    assert m["logprob"] < 0 and m["entropy"] > 0
    # entropy is maximal for uniform logits
    mu = metrics_from_logits(torch.tensor([1.0, 1.0, 1.0, 1.0]), 0, [0, 1, 2, 3])
    assert abs(mu["entropy"] - torch.log(torch.tensor(4.0)).item()) < 1e-5


def test_gate_destroy_real_bases():
    seq = "AAAA" + "GT" + "AAAA"          # donor GT at pos 4
    dseq = "AAAA" + "CC" + "AAAA"          # destroyed, real bases, same length
    ok, msg = gate_destroy_real_bases(seq, dseq, 4, is_donor=True)
    assert ok, msg
    # length change -> fail
    assert not gate_destroy_real_bases(seq, dseq + "A", 4, True)[0]
    # edit outside the motif footprint -> fail
    bad = "TAAA" + "GT" + "AAAA"
    assert not gate_destroy_real_bases(seq, bad, 4, True)[0]
    # non-ACGT introduced -> fail
    assert not gate_destroy_real_bases(seq, "AAAA" + "CN" + "AAAA", 4, True)[0]


def _synthetic_window(eid, seed):
    import random
    rng = random.Random(seed)
    N = 1024
    s = [rng.choice("ACGT") for _ in range(N)]
    # alt5ss_donor: target donor GT@200, competitor donor GT@260, partner acceptor AG@600, decoy donor GT@140
    def put(pos, motif):
        for i, c in enumerate(motif):
            s[pos + i] = c
    put(200, "GT"); put(260, "GT"); put(140, "GT")
    s[598], s[599] = "A", "G"   # AG at [598,599], acceptor anchor 600
    seq = "".join(s)
    return WindowedCompetition(
        example_id=eid, gene_id=f"g{seed}", chrom="chr1", strand="+", split="train",
        site_type="alt5ss_donor", seq=seq, target_win=200, partner_win=600, competitor_win=260,
        target_is_donor=True, competitor_is_donor=True, partner_is_donor=False,
        cluster_size=2, n_usable_competitors=1, target_partner_distance=400, target_competitor_distance=60)


def test_target_base_position_respects_acceptor_coordinate_contract():
    donor = _synthetic_window("donor", 4)
    assert target_base_position(donor) == donor.target_win
    acceptor = WindowedCompetition(
        example_id="acceptor", gene_id="ga", chrom="chr1", strand="+", split="train",
        site_type="alt3ss_acceptor", seq="A" * 198 + "AG" + "A" * 824,
        target_win=200, partner_win=100, competitor_win=260,
        target_is_donor=False, competitor_is_donor=False, partner_is_donor=True,
        cluster_size=2, n_usable_competitors=1,
        target_partner_distance=100, target_competitor_distance=60,
    )
    assert acceptor.seq[acceptor.target_win - 1] == "G"
    assert target_base_position(acceptor) == acceptor.target_win - 1


def test_closure_group_size_and_head_parsers():
    assert parse_ints("1,5,5,96") == (1, 5, 96)
    assert parse_head("L3H1") == (3, 1)
    with pytest.raises(ValueError):
        parse_ints("0,4")
    with pytest.raises(ValueError):
        parse_head("L6H0")


def test_discovery_and_sealed_cohorts_are_disjoint():
    train = SimpleNamespace(chrom="chr1", split="train")
    val = SimpleNamespace(chrom="chr11", split="val")
    sealed = SimpleNamespace(chrom="chr8", split="test")
    assert cohort_example_allowed(train, unlock_sealed=False)
    assert not cohort_example_allowed(val, unlock_sealed=False)
    assert not cohort_example_allowed(sealed, unlock_sealed=False)
    assert cohort_example_allowed(sealed, unlock_sealed=True)
    assert not cohort_example_allowed(train, unlock_sealed=True)


def test_layer_matched_random_sets_preserve_architecture_stratum():
    selected = ((0, 0), (0, 1), (2, 4), (5, 9))
    draws = layer_matched_random_sets(selected, n=12, seed=17)
    assert len(draws) == 12
    assert len(set(draws)) == 12
    assert tuple(sorted(selected)) not in draws
    for draw in draws:
        assert [sum(layer == q for layer, _ in draw) for q in range(6)] == [2, 0, 1, 0, 0, 1]


def test_atlas_preflight_pair_selection_covers_tags_and_roles():
    core = list(CORE_CATEGORIES)
    packets = {
        "d_case": {"tags": core[:5]},
        "d_control": {"tags": core[1:5]},
        "a_case": {"tags": core[1:]},
        "a_control": {"tags": core[1:5]},
    }
    pairs = [
        {
            "pair_id": "p_d", "target_role": "donor", "category": core[0],
            "case_packet_id": "d_case", "control_packet_id": "d_control",
        },
        {
            "pair_id": "p_a", "target_role": "acceptor", "category": core[-1],
            "case_packet_id": "a_case", "control_packet_id": "a_control",
        },
    ]
    selected = select_cover_pairs(pairs, packets)
    assert {pair["target_role"] for pair in selected} == {"donor", "acceptor"}
    covered = set().union(*(set(packets[pair["case_packet_id"]]["tags"]) for pair in selected))
    assert set(CORE_CATEGORIES) <= covered


def test_health_probes_are_deterministic_and_avoid_junction_motifs():
    packet = {
        "packet_id": "p", "donor_win": 200, "acceptor_win": 600,
    }
    sequence = "A" * 1024
    first = health_probes(packet, sequence)
    second = health_probes(packet, sequence)
    assert first == second
    assert len(first) == 4
    for position in first:
        assert min(abs(position - site) for site in (200, 201, 598, 599)) > 12


def test_partner_destroyed_respects_role_specific_motif_coordinates():
    donor_target = _synthetic_window("d", 11)
    donor_packet = {
        "packet_id": "d", "target_role": "donor", "partner_anchor_win": 600,
        "masked_base_win": 200,
    }
    dseq = partner_destroyed(donor_packet, donor_target.seq)
    assert dseq[598:600] != "AG"
    assert dseq[200] == "G"

    acceptor_sequence = list(donor_target.seq)
    acceptor_sequence[260:262] = "GT"
    acceptor_packet = {
        "packet_id": "a", "target_role": "acceptor", "partner_anchor_win": 260,
        "masked_base_win": 599,
    }
    aseq = partner_destroyed(acceptor_packet, "".join(acceptor_sequence))
    assert aseq[260:262] not in {"GT", "GC"}
    assert aseq[599] == "G"


def test_load_rankings_are_full_permutations(tmp_path):
    directional = {}
    contrast = {}
    per_head = {}
    for i, (layer, head) in enumerate(ALL_HEADS):
        label = f"L{layer}H{head}"
        directional[label] = {"donor_mean": float(i)}
        per_head[f"C|{label}"] = {"magnitude_mean": float(96 - i)}
        per_head[f"D|{label}"] = {"magnitude_mean": 0.0}
        if i % 5 == 0:  # mirrors the real artifact: CIs exist only for a candidate subset
            contrast[label] = {"mean": float(96 - i)}
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps({
        "directional_B": directional,
        "C_minus_D_magnitude": contrast,
        "per_head_train_only": per_head,
    }))
    ranks = load_rankings(path)
    assert set(ranks["B"]) == set(ALL_HEADS)
    assert set(ranks["CD"]) == set(ALL_HEADS)
    assert ranks["B"][0] == ALL_HEADS[-1]
    assert ranks["CD"][0] == ALL_HEADS[0]


@pytest.mark.skipif(not (REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt/config.json").exists(),
                    reason="SpliceBERT checkpoint not present")
def test_vectorized_scalar_agreement():
    from run_all_head_screen import ScreenModel, screen_chunk
    from gfmcd.splicing.circuit import load_adapter, pick_device
    dev = pick_device(None)
    adapter = load_adapter(str(REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"),
                           dev, eager=False)
    model = ScreenModel(adapter, batch_size=8)
    w1, w2 = _synthetic_window("ex1", 1), _synthetic_window("ex2", 2)
    d1, d2 = 140, 140  # decoy donor position
    # vectorized: both in one chunk
    recs_v, _ = screen_chunk(model, [w1, w2], [d1, d2])
    # scalar: one at a time
    recs_s1, _ = screen_chunk(model, [w1], [d1])
    recs_s2, _ = screen_chunk(model, [w2], [d2])
    for eid, rs in (("ex1", recs_s1), ("ex2", recs_s2)):
        ev = recs_v[eid]["conditions"]["C"]["head_effects"]
        es = rs[eid]["conditions"]["C"]["head_effects"]
        for a, b in zip(ev, es):
            assert abs(a["effect_margin"] - b["effect_margin"]) < 1e-3, (eid, a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
