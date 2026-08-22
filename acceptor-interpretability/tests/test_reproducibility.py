"""Regression tests for the defects found in the 2026-08-22 methods audit.

Model-free by design: everything here runs on sequence strings and the config,
so it executes in a clean checkout with no checkpoint, no network, and no GPU.

    python -m pytest acceptor-interpretability/tests -q

Each test pins one thing that was actually broken and silently produced
unreproducible or confounded numbers.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ARC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARC))

from acceptors.config import load_config  # noqa: E402
from acceptors.data import (  # noqa: E402
    YAG_MINUS3_GUARD,
    _bp_score,
    corrupt_region,
    locate_tract,
    random_no_ag,
    sequence_rng,
    shuffle_region_in_place,
)

SEQ = "ACGTTCTTTCCTTTCAG" * 24  # 408nt, deterministic, DNA alphabet


# ---------------------------------------------------------------------------
# 1. Reproducibility: the corruption draw must not depend on hash randomization
# ---------------------------------------------------------------------------

def test_sequence_rng_is_stable_within_process():
    assert sequence_rng(SEQ, 0).random() == sequence_rng(SEQ, 0).random()


def test_sequence_rng_varies_with_seed_and_sequence():
    assert sequence_rng(SEQ, 0).random() != sequence_rng(SEQ, 1).random()
    assert sequence_rng(SEQ, 0).random() != sequence_rng(SEQ[::-1], 0).random()


@pytest.mark.parametrize("hashseed", ["0", "1", "424242"])
def test_corruption_is_identical_across_processes_and_hashseeds(hashseed):
    """The original code seeded from `hash(seq)`, which CPython salts per
    process, so every run drew different corruptions and no published number
    could be regenerated. This is the test that would have caught it."""
    script = (
        f"import sys; sys.path.insert(0, {str(ARC)!r});"
        "from acceptors.data import sequence_rng, corrupt_region;"
        f"seq={SEQ!r};"
        "print(corrupt_region(seq, 183, 197, sequence_rng(seq, 0))[183:197])"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, check=True,
        env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    expected = corrupt_region(SEQ, 183, 197, sequence_rng(SEQ, 0))[183:197]
    assert out == expected


def test_no_module_uses_hash_for_seeding():
    """Guard against the pattern coming back anywhere in the arc.

    Parsed rather than grepped, so the docstrings that *describe* the
    anti-pattern do not trip it.
    """
    import ast

    offenders = []
    for p in list(ARC.glob("experiments/*.py")) + list(ARC.glob("acceptors/*.py")):
        for node in ast.walk(ast.parse(p.read_text())):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", None)
            if name != "Random":
                continue
            if any(
                isinstance(sub, ast.Call)
                and getattr(sub.func, "id", None) == "hash"
                for arg in node.args
                for sub in ast.walk(arg)
            ):
                offenders.append(f"{p.relative_to(ARC)}:{node.lineno}")
    assert not offenders, f"process-unstable RNG seeding reintroduced at: {offenders}"


# ---------------------------------------------------------------------------
# 2. The -3 pyrimidine of the YAG consensus is not part of the tract
# ---------------------------------------------------------------------------

def test_config_tract_window_spares_the_minus3_position():
    c = load_config()
    assert c.coords.ppt_end == c.coords.ag_start - YAG_MINUS3_GUARD, (
        "the fixed-offset tract window must stop short of the YAG -3 pyrimidine; "
        "including it confounds the tract effect with 3'ss consensus destruction"
    )


def test_located_tract_window_spares_the_minus3_position():
    ag_start = 198
    for bp_pos in (150, 170, 180, 190, 196):
        _, end = locate_tract(SEQ, bp_pos, ag_start, min_gap=3)
        assert end <= ag_start - YAG_MINUS3_GUARD


def test_located_tract_window_is_never_inverted():
    ag_start = 198
    for bp_pos in (150, 195, 197, 198):
        start, end = locate_tract(SEQ, bp_pos, ag_start, min_gap=3)
        assert start <= end


# ---------------------------------------------------------------------------
# 3. The branch-point scorer must read the alphabet the data is actually in
# ---------------------------------------------------------------------------

def test_bp_score_reads_dna_and_rna_identically():
    """GUE sequences are DNA (A/C/G/T). Scoring only "U"/"CU" disabled the
    highest-weight consensus term and reduced the pyrimidine tests to "is C"."""
    # code's stated layout: y(-2) U(-1) A(0) n(+1) y(+2)
    assert _bp_score("CTAGC", 2) == _bp_score("CUAGC", 2) == 4.0
    assert _bp_score("GGAGG", 2) == 0.0


def test_bp_score_counts_thymine_as_a_pyrimidine():
    assert _bp_score("TTAGT", 2) == _bp_score("UUAGU", 2)


# ---------------------------------------------------------------------------
# 4. Corruption invariants the write-up relies on
# ---------------------------------------------------------------------------

def test_corruption_never_injects_ag_including_at_the_seams():
    for seed in range(25):
        out = corrupt_region(SEQ, 183, 197, sequence_rng(SEQ, seed))
        assert "AG" not in out[182:198], "corruption injected a decoy AG at a seam"


def test_corruption_alphabet_matches_the_input_alphabet():
    """Clean and corrupted sequences must be in one alphabet, or the
    sequence-level helpers (branch-point scoring, decoy search) see two."""
    region = random_no_ag(200, sequence_rng(SEQ, 0))
    assert set(region) <= set("ACGT")


def test_shuffle_preserves_composition_exactly():
    out = shuffle_region_in_place(SEQ, 183, 197, sequence_rng(SEQ, 0))
    assert sorted(out[183:197]) == sorted(SEQ[183:197])
    assert out[:183] == SEQ[:183] and out[197:] == SEQ[197:]


# ---------------------------------------------------------------------------
# 5. Provenance: a result is only interpretable with its conditions recorded
# ---------------------------------------------------------------------------

def test_every_model_running_phase_writes_a_manifest():
    runners = [
        "phase0_pilot.py", "phase1_control_fix.py", "phase2_absolute_floor.py",
        "phase3_real_coordinates.py", "phase4_head_content_control.py",
        "phase5_intervention_ladder.py",
    ]
    missing = [
        r for r in runners
        if "write_run_manifest" not in (ARC / "experiments" / r).read_text()
    ]
    assert not missing, f"runners that would emit a result with no manifest: {missing}"


def test_checkpoint_is_pinned_to_a_revision():
    c = load_config()
    revision = getattr(c.model, "revision", None)
    assert revision and len(revision) == 40, (
        "the checkpoint must be pinned to a full revision sha so a re-upload "
        "cannot silently change the model under a rerun"
    )


def test_cohort_digest_is_order_sensitive():
    from acceptors.provenance import cohort_digest

    a = cohort_digest(["ACGT", "TTTT"])
    b = cohort_digest(["TTTT", "ACGT"])
    assert a["n"] == b["n"] == 2
    assert a["sha256"] != b["sha256"]
