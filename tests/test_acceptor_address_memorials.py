"""Memorial / invariant tests for donor-graph experiment 3 (acceptor-address transplant).

A failure here means the science is invalid, not that a refactor broke a unit.  These are
torch/numpy-only (no genome, no model) so they run in well under a second.

Guards:
* the phase-matched decoy really is matched on distance-mod-3 (the new control over
  ``circuit.matched_decoy_acceptor``, which matches distance only);
* the G-Match audit fails when the match is broken;
* the pure QK/attention readout helpers compute exactly what the prereg says.

The R7 swap-is-identity property of ``relocate_proj`` itself is covered by
``tests/test_circuit_memorials.py``; the runner's use of it is checked live in the preflight.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from gfmcd.splicing.address import (
    AddressPair, acceptor_motif_score, attn_to_window, match_audit, phase_matched_decoy,
    qk_logit_to_window, windows,
)


def _example(donor, next_acc, other_ag, seq_len=1024):
    seq = "".join("AG"[(i % 2)] for i in range(seq_len))  # arbitrary; motif not under test here
    return SimpleNamespace(donor_pos=donor, next_acceptor_pos=next_acc,
                           other_ag=list(other_ag), sequence=seq, gene_id="g", strand="+")


def test_phase_matched_decoy_matches_distance_mod_3():
    d, ap = 100, 220                      # partner distance 120, phase 120 % 3 == 0
    # candidates (all >12 from the partner 220): 205 (dist 105, phase 0), 190 (dist 90, phase 0),
    #            200 (dist 100, phase 1 -> wrong phase, excluded). Selection is distance-closest
    #            among phase matches, so 205 (|105-120|=15) beats 190 (|90-120|=30); 200 (wrong
    #            phase) must never be chosen.
    x = _example(d, ap, other_ag=[205, 190, 200])
    am = phase_matched_decoy(x, tol=40)
    assert am is not None
    assert am != 200                              # wrong-phase candidate must be rejected
    assert (am - d) % 3 == (ap - d) % 3           # THE guarantee
    assert abs((am - d) - (ap - d)) <= 40
    assert abs(am - ap) > 12
    assert am == 205                              # distance-closest phase match


def test_phase_matched_decoy_none_when_no_phase_match():
    d, ap = 100, 220                      # phase 0 (needs p % 3 == 1)
    x = _example(d, ap, other_ag=[200, 209, 212])   # phases 1,2,2 -> none match phase 0
    assert phase_matched_decoy(x, tol=40) is None


def test_phase_matched_decoy_excludes_partner_neighbourhood():
    d, ap = 100, 220
    # 217 is phase-0 and within tol, but within 12 of the partner (|217-220|=3) -> rejected;
    # 205 is phase-0 and >12 from the partner -> the valid decoy.
    x = _example(d, ap, other_ag=[217, 205])
    am = phase_matched_decoy(x, tol=40)
    assert am == 205


def _pair(donor, ap, am, mplus=0.5, mminus=0.5):
    ex = SimpleNamespace(donor_pos=donor, gene_id="g", strand="+")
    return AddressPair(example=ex, a_plus=ap, a_minus=am, dist_plus=ap - donor,
                       dist_minus=am - donor, phase=(ap - donor) % 3, motif_plus=mplus,
                       motif_minus=mminus, gc_plus=0.5, gc_minus=0.5, coding=True)


def test_match_audit_passes_when_matched():
    # both strands, matched distance/phase/motif
    pairs = [_pair(100, 200, 203), _pair(100, 200, 203)]
    pairs[1].example.strand = "-"
    audit = match_audit(pairs)
    assert audit["phase_mismatch_rate"] == 0
    assert audit["PASS"] is True


def test_match_audit_fails_on_phase_mismatch():
    # a_minus at 202 breaks the mod-3 match (202-100=102 phase 0 vs partner phase 1)
    pairs = [_pair(100, 200, 202)]
    audit = match_audit(pairs)
    assert audit["phase_mismatch_rate"] == 1
    assert audit["PASS"] is False


def test_match_audit_reports_motif_gap_but_does_not_gate_on_it():
    # The generic-AG decoy cannot be motif-matched to a real acceptor; motif_gap is REPORTED,
    # not gated (the strength-controlled identity claim uses the real-acceptor decoy arm).
    pairs = [_pair(100, 200, 203, mplus=1.0, mminus=0.0) for _ in range(5)]
    audit = match_audit(pairs)
    assert audit["motif_gap_sd"] > 0.15   # a large gap is present and reported
    assert audit["PASS"] is True          # ...but distance+phase are matched, so G-Match passes


def test_match_audit_fails_on_distance_gap():
    # distance is gated: a decoy 30 nt off the partner distance breaks the match
    pairs = [_pair(100, 200, 233) for _ in range(3)]   # dist_minus 133 vs dist_plus 100
    audit = match_audit(pairs)
    assert audit["dist_bias_nt"] > 8.0                 # systematic |mean(a+)-mean(a-)| = 33
    assert audit["PASS"] is False


def test_acceptor_motif_score_rewards_ag_and_pyrimidines():
    seq = "T" * 30 + "AG" + "C" * 30     # AG at [30,32); tract before is all T (pyrimidine)
    s = acceptor_motif_score(seq, pos=32, upstream=20)
    assert 0.99 <= s <= 1.0              # 0.5 (AG) + 0.5 * 1.0 (all-pyrimidine tract)
    s_noag = acceptor_motif_score("A" * 62, pos=32)
    assert s_noag < 0.5                  # no AG, purine tract


def test_qk_logit_to_window_matches_hand_computation():
    torch.manual_seed(0)
    B, S, H, Dh = 2, 12, 4, 3
    query = torch.randn(B, S, H * Dh)
    key = torch.randn(B, S, H * Dh)
    head, q_tok = 1, torch.tensor([5, 7])
    win = torch.tensor([[2, 3], [4, 5]])
    got = qk_logit_to_window(query, key, head, Dh, q_tok, win)
    for b in range(B):
        q = query[b, q_tok[b], head * Dh:(head + 1) * Dh]
        vals = []
        for j in win[b]:
            k = key[b, j, head * Dh:(head + 1) * Dh]
            vals.append(float(q @ k) / (Dh ** 0.5))
        assert abs(float(got[b]) - float(np.mean(vals))) < 1e-5


def test_attn_to_window_sums_selected_columns():
    B, S = 2, 10
    attn = torch.zeros(B, S, S)
    attn[0, 4, 2] = 0.3
    attn[0, 4, 3] = 0.2
    attn[1, 6, 5] = 0.7
    q_tok = torch.tensor([4, 6])
    win = torch.tensor([[2, 3], [5, 9]])
    got = attn_to_window(attn, q_tok, win)
    assert abs(float(got[0]) - 0.5) < 1e-6
    assert abs(float(got[1]) - 0.7) < 1e-6


def test_windows_add_offset_and_offsets():
    w = windows([200], off=1, device="cpu", offsets=(-3, -2, -1, 0))
    assert w.tolist() == [[198, 199, 200, 201]]     # 200 + 1 + {-3,-2,-1,0}
