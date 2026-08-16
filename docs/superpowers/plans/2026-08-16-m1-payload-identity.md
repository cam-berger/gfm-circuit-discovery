# M1 Payload Identity Implementation Plan

> **Tier: T2 — implementation plan.** Working document for one stage; not a preregistration and not a
> claim. Authorises no run. The binding authority is the spec it argues from.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether SpliceBERT's masked-acceptor readout treats a `GC` donor as a donor, by adding the one missing arm to an existing operator ladder and contrasting it against its two non-motif twins on a common row set.

**Architecture:** A new pure-Python module `acceptor_payload.py` supplies four things the existing machinery lacks — a register-scoped motif-hygiene carve-out, a common-admissible row set, a gene-clustered ρ-difference estimator, and a precision gate. A staged runner then composes those with the existing `margins_at` / `destroy_donor` / `gboot_ratio` primitives. Every unit test runs without the model or a genome; only the runner's smoke and preflight stages need either.

**Tech Stack:** Python 3, numpy, PyTorch (runner only), `uv` for environment management, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-16-acceptor-mechanism-dissection-design.md`](../specs/2026-08-16-acceptor-mechanism-dissection-design.md) §4

## Global Constraints

- Run everything from the repo root with `uv run --extra splicebert --extra biology --extra dev`. A bare `uv run pytest` drops the extras and reports a *failing* suite, not a smaller one.
- Model-dependent work requires the main checkout — model, GTF and FASTA are absent in worktrees. Set `GFMCD_GTF=/Users/cberg/Research/data/genomes/gencode.v50.annotation.gtf.gz` and `GFMCD_FASTA=/Users/cberg/Research/data/genomes/hg38.fa`; without them 13 memorials skip silently and the suite still reports green.
- **Never `git add -A`.** This working directory may be shared with a parallel session. Stage explicit paths only. **Do not push.**
- Do not edit any file under `scripts/splicebert/` — those are donor-study runners cited as evidence by frozen preregistrations, some hash-pinned.
- Do not edit any frozen preregistration, including additively.
- Every result artifact gets a sibling `*.manifest.json`. No runner's `--out` may default to a registered artifact.
- Cohort: **train chromosomes only** (everything outside chr11/16/19 and chr8/9). chr8/9 is sealed and must not be read.
- Frozen constants: `REFERENCE_ALT = "AC"`, `TWIN_ARMS = ("GA", "GG")`, `TREATMENT_ARM = "GC"`, `SESOI = 0.05`, `SEED = 0`, `N_BOOT = 2000`, `MIN_N = 50`.
- Retired vocabulary is forbidden in prose: `10.67`, `10.58`, a `±0.005` carrier precision, "splice-site strength". `[a, b]` denotes a confidence interval and nothing else.

## File Structure

| file | responsibility |
|---|---|
| `src/gfmcd/splicing/acceptor_payload.py` *(create)* | Register-scoped admissibility, common row set, ρ-difference estimator, precision gate, verdict classification. Pure — no torch, no model. |
| `tests/test_acceptor_payload.py` *(create)* | Unit tests for the above. Runs with no model and no genome. |
| `scripts/splicebert_acceptor_coupling/run_m1_payload_identity.py` *(create)* | Staged runner: `smoke` / `preflight` / `target`. Composes the module with existing primitives, emits artifact + manifest. |
| `tests/test_m1_payload_identity.py` *(create)* | Runner-level memorials and CLI contract. Model-dependent cases skip cleanly when the checkpoint is absent. |
| `studies/splicebert_acceptor_coupling/preregistrations/PREREGISTRATION_M1_PAYLOAD_IDENTITY.md` *(create)* | Frozen protocol, pinning the runner's SHA-256. Written after the runner exists, before any target run. |

---

### Task 1: Register-scoped motif-hygiene carve-out

The existing `donor_alt_admissible()` rejects `GC` outright, because `planted_donor_motifs()` flags the `written` register. M1's `GC` arm *needs* that motif — it is the treatment. But `upstream` and `downstream` violations must still disqualify, on every arm.

`planted_donor_motifs()` returns a dict keyed by register name, so the carve-out is a filter on keys, not a waiver.

**Files:**
- Create: `src/gfmcd/splicing/acceptor_payload.py`
- Test: `tests/test_acceptor_payload.py`

**Interfaces:**
- Consumes: `gfmcd.splicing.coupling.planted_donor_motifs(seq, donor_pos, alt) -> dict[str, str]`, `gfmcd.splicing.reverse_coupling.donor_alt_admissible(x, alt) -> bool`, `gfmcd.splicing.coupling.CouplingExample`
- Produces: `TREATMENT_ARM: str`, `REFERENCE_ALT: str`, `TWIN_ARMS: tuple[str, ...]`, `LADDER_ARMS: tuple[str, ...]`, `arm_dispositions(x, alt) -> dict[str, str]`, `arm_admissible(x, alt, *, allow_written_motif: bool = False) -> bool`, `allows_written_motif(alt) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_acceptor_payload.py`:

```python
"""M1 payload-identity primitives. No model or genome required."""

from __future__ import annotations

import pytest

from gfmcd.splicing.acceptor_payload import (
    LADDER_ARMS,
    REFERENCE_ALT,
    TREATMENT_ARM,
    TWIN_ARMS,
    allows_written_motif,
    arm_admissible,
    arm_dispositions,
)
from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.reverse_coupling import donor_alt_admissible

DONOR_POS = 21


def make_example(minus_one: str = "G", plus_three: str = "A", ex_id: str = "e0",
                 gene_id: str = "g0") -> CouplingExample:
    """A window whose donor GT sits at DONOR_POS.

    seq[DONOR_POS-1] is the last exon base; seq[DONOR_POS:DONOR_POS+2] is the GT;
    seq[DONOR_POS+2] is the +3 base. Those three positions drive every straddling register.
    """
    seq = "A" * (DONOR_POS - 1) + minus_one + "GT" + plus_three + "C" * 20
    assert seq[DONOR_POS:DONOR_POS + 2] == "GT"
    return CouplingExample(
        example_id=ex_id, sequence=seq, acceptor_pos=5, donor_pos=DONOR_POS,
        chrom="chr1", strand="+", gene_id=gene_id, split="train",
        exon_length=16, intron_length_down=100, n_isoforms_donor=1,
        n_isoforms_acceptor=1, gene_type="protein_coding", next_acceptor_pos=60,
    )


def test_frozen_arm_constants():
    assert REFERENCE_ALT == "AC"
    assert TREATMENT_ARM == "GC"
    assert TWIN_ARMS == ("GA", "GG")
    assert LADDER_ARMS == ("AC", "AT", "GA", "GG", "GC")
    assert allows_written_motif("GC") is True
    assert allows_written_motif("GG") is False


def test_gc_flags_only_the_written_register_on_a_clean_flank():
    x = make_example(minus_one="G", plus_three="A")
    assert arm_dispositions(x, "GC") == {"written": "GC"}
    assert arm_admissible(x, "GC", allow_written_motif=True) is True
    assert arm_admissible(x, "GC", allow_written_motif=False) is False


def test_gc_upstream_ag_still_disqualifies():
    """A -1 exon base of A makes upstream 'AG' -- a planted acceptor, not the treatment."""
    x = make_example(minus_one="A", plus_three="A")
    assert arm_dispositions(x, "GC")["upstream"] == "AG"
    assert arm_admissible(x, "GC", allow_written_motif=True) is False


def test_gg_downstream_gt_disqualifies():
    """alt='GG' with a +3 base of T plants a canonical GT one position downstream."""
    x = make_example(minus_one="G", plus_three="T")
    assert arm_dispositions(x, "GG")["downstream"] == "GT"
    assert arm_admissible(x, "GG", allow_written_motif=False) is False


def test_reference_arm_is_clean_on_a_clean_flank():
    x = make_example(minus_one="G", plus_three="A")
    assert arm_dispositions(x, REFERENCE_ALT) == {}
    assert arm_admissible(x, REFERENCE_ALT) is True


@pytest.mark.parametrize("minus_one", ["A", "C", "G", "T"])
@pytest.mark.parametrize("plus_three", ["A", "C", "G", "T"])
@pytest.mark.parametrize("alt", ["AC", "AT", "GA", "GG", "GC"])
def test_without_carveout_matches_the_existing_hygiene_rule(minus_one, plus_three, alt):
    """Regression guard: allow_written_motif=False must be exactly donor_alt_admissible."""
    x = make_example(minus_one=minus_one, plus_three=plus_three)
    assert arm_admissible(x, alt, allow_written_motif=False) == donor_alt_admissible(x, alt)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gfmcd.splicing.acceptor_payload'`

- [ ] **Step 3: Write minimal implementation**

Create `src/gfmcd/splicing/acceptor_payload.py`:

```python
"""M1 payload-identity primitives for the masked-acceptor readout.

The ladder asks one question: does the model treat a ``GC`` donor as a donor?  ``GC`` is
the only single-base substitution that keeps the ``G``, destroys the ``T``, and yields a
real (minor) splice donor.  Its non-motif twins ``GA`` and ``GG`` are identical in edit
size, edited position, and keeps-G/destroys-T structure, so the contrast isolates motif
identity with no composition or edit-size confound.

``GC`` is absent from ``coupling.DONOR_ALTS`` by construction ("none is GT and none is
GC") and ``planted_donor_motifs`` flags it.  That rule protects the pseudo-donor control
from silently becoming a real donor; here the planted donor IS the treatment, so the
carve-out is scoped to the ``written`` register of the treatment arm alone.
"""

from __future__ import annotations

from collections.abc import Mapping

from gfmcd.splicing.coupling import CouplingExample, planted_donor_motifs

REFERENCE_ALT = "AC"
TREATMENT_ARM = "GC"
TWIN_ARMS = ("GA", "GG")
LADDER_ARMS = ("AC", "AT", "GA", "GG", "GC")

_WRITTEN_MOTIF_TREATMENTS = frozenset({TREATMENT_ARM})


def allows_written_motif(alt: str) -> bool:
    """True only for arms whose planted ``written`` motif is the treatment itself."""
    return alt in _WRITTEN_MOTIF_TREATMENTS


def arm_dispositions(x: CouplingExample, alt: str) -> Mapping[str, str]:
    """Splice motifs created by writing ``alt`` over this example's donor ``GT``.

    Keys are register names (``upstream`` / ``written`` / ``downstream``); an empty
    mapping means the construct plants nothing.  Serialize this per row -- it is the
    audit record for the carve-out.
    """
    return planted_donor_motifs(x.sequence, x.donor_pos, alt)


def arm_admissible(
    x: CouplingExample, alt: str, *, allow_written_motif: bool = False
) -> bool:
    """Is this construct admissible for this arm?

    With ``allow_written_motif=False`` this is exactly ``donor_alt_admissible``.  With it
    set, a motif in the ``written`` register is accepted -- and only there; ``upstream``
    and ``downstream`` violations disqualify on every arm, because a planted ``AG``
    acceptor or a planted flanking ``GT`` is a confound, never a treatment.
    """
    planted = arm_dispositions(x, alt)
    disqualifying = {
        register: motif
        for register, motif in planted.items()
        if not (allow_written_motif and register == "written")
    }
    return not disqualifying
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -v`
Expected: PASS — 85 tests (5 named + 80 parametrized)

- [ ] **Step 5: Commit**

```bash
git add src/gfmcd/splicing/acceptor_payload.py tests/test_acceptor_payload.py
git commit -m "feat(m1): register-scoped motif-hygiene carve-out for the GC arm"
```

---

### Task 2: Common admissible row set

`run_r2_donor_base_decomposition.py` computes a *separate* admissible subset per arm (`adm = [i for i, x in enumerate(P) if donor_alt_admissible(x, z)]`). Admissibility varies sharply by alt — `GA` is `0.37`, `GG` is `0.842` — so those arms are measured on different row sets and their ρ values are not strictly comparable. M1's contrast must not inherit that.

**Files:**
- Modify: `src/gfmcd/splicing/acceptor_payload.py`
- Test: `tests/test_acceptor_payload.py`

**Interfaces:**
- Consumes: `arm_admissible`, `allows_written_motif`, `LADDER_ARMS` from Task 1
- Produces: `common_admissible_rows(exs, arms) -> list[int]`, `admissible_fractions(exs, arms) -> dict[str, float]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_acceptor_payload.py`:

```python
from gfmcd.splicing.acceptor_payload import admissible_fractions, common_admissible_rows


def test_common_rows_are_admissible_under_every_arm():
    exs = [
        make_example(minus_one="G", plus_three="A", ex_id="clean", gene_id="g1"),
        make_example(minus_one="A", plus_three="A", ex_id="bad_gc", gene_id="g2"),
        make_example(minus_one="G", plus_three="T", ex_id="bad_gg", gene_id="g3"),
    ]
    assert common_admissible_rows(exs, LADDER_ARMS) == [0]


def test_common_rows_empty_arms_returns_all():
    exs = [make_example(ex_id="a"), make_example(minus_one="A", ex_id="b")]
    assert common_admissible_rows(exs, ()) == [0, 1]


def test_admissible_fractions_are_per_arm_and_reported():
    exs = [
        make_example(minus_one="G", plus_three="A", ex_id="clean"),
        make_example(minus_one="A", plus_three="A", ex_id="bad_gc"),
    ]
    fracs = admissible_fractions(exs, ("AC", "GC"))
    assert fracs["AC"] == 1.0
    assert fracs["GC"] == 0.5


def test_admissible_fractions_empty_cohort_is_zero_not_a_crash():
    assert admissible_fractions([], ("AC",)) == {"AC": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -k "common_rows or admissible_fractions" -v`
Expected: FAIL — `ImportError: cannot import name 'common_admissible_rows'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/gfmcd/splicing/acceptor_payload.py`:

```python
from collections.abc import Sequence


def common_admissible_rows(
    exs: Sequence[CouplingExample], arms: Sequence[str]
) -> list[int]:
    """Indices admissible under EVERY arm, so all arms share one row set.

    Measuring each arm on its own admissible subset makes the arms incomparable:
    admissibility runs from 0.37 (``GA``) to 0.842 (``GG``) in the recorded val ladder,
    so per-arm subsets differ in both size and sequence composition.  The contrast is
    only interpretable on rows every arm can construct.
    """
    return [
        i
        for i, x in enumerate(exs)
        if all(
            arm_admissible(x, alt, allow_written_motif=allows_written_motif(alt))
            for alt in arms
        )
    ]


def admissible_fractions(
    exs: Sequence[CouplingExample], arms: Sequence[str]
) -> dict[str, float]:
    """Per-arm admissible fraction of the supplied cohort, for the artifact record.

    Reported rather than assumed: a carve-out that quietly changed ``GC``'s
    admissibility relative to its twins would bias the contrast even on a common row set,
    because the common set would be driven by whichever arm is most restrictive.
    """
    total = len(exs)
    out: dict[str, float] = {}
    for alt in arms:
        allow = allows_written_motif(alt)
        n = sum(1 for x in exs if arm_admissible(x, alt, allow_written_motif=allow))
        out[alt] = (n / total) if total else 0.0
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -v`
Expected: PASS — all tests including the four new ones

- [ ] **Step 5: Commit**

```bash
git add src/gfmcd/splicing/acceptor_payload.py tests/test_acceptor_payload.py
git commit -m "feat(m1): common admissible row set so ladder arms are comparable"
```

---

### Task 3: Gene-clustered ρ-difference estimator

`gboot_ratio` gives a CI for a single ratio. The primary estimand is a *difference of ratios*, `ρ_GC − ρ_twins`, on one common row set. Resampling must be by gene, and both ratios must be recomputed inside each draw so their correlation is preserved — computing two independent CIs and subtracting would overstate the width.

**Files:**
- Modify: `src/gfmcd/splicing/acceptor_payload.py`
- Test: `tests/test_acceptor_payload.py`

**Interfaces:**
- Consumes: numpy
- Produces: `rho(v, ref) -> float`, `pooled_twin_effect(per_arm, twins) -> np.ndarray`, `gboot_rho_difference(v_treatment, v_reference_arm, ref, genes, *, n_draws=2000, seed=0) -> tuple[float, float, float]` returning `(estimate, lo, hi)`, `classify_difference(lo, hi, sesoi) -> str`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_acceptor_payload.py`:

```python
import numpy as np

from gfmcd.splicing.acceptor_payload import (
    classify_difference,
    gboot_rho_difference,
    pooled_twin_effect,
    rho,
)


def test_rho_is_ratio_of_sums_not_mean_of_ratios():
    """Near-zero denominators make per-example ratios explode; ratio-of-means does not."""
    v = np.array([1.0, 1.0])
    ref = np.array([2.0, 1e-12])
    assert rho(v, ref) == pytest.approx(2.0 / (2.0 + 1e-12))


def test_pooled_twin_effect_is_the_rowwise_mean():
    per_arm = {"GA": np.array([1.0, 3.0]), "GG": np.array([3.0, 5.0]), "GC": np.array([0.0, 0.0])}
    assert pooled_twin_effect(per_arm, ("GA", "GG")) == pytest.approx(np.array([2.0, 4.0]))


def test_identical_arms_give_exactly_zero_difference():
    rng = np.random.default_rng(0)
    v = rng.normal(1.0, 0.2, 60)
    ref = rng.normal(1.0, 0.2, 60)
    genes = [f"g{i // 3}" for i in range(60)]
    est, lo, hi = gboot_rho_difference(v, v, ref, genes, n_draws=200, seed=0)
    assert est == pytest.approx(0.0, abs=1e-12)
    assert lo <= 0.0 <= hi


def test_planted_difference_is_recovered():
    rng = np.random.default_rng(1)
    ref = np.abs(rng.normal(1.0, 0.05, 120)) + 0.5
    twin = ref * 0.85
    treat = ref * 0.35
    genes = [f"g{i // 4}" for i in range(120)]
    est, lo, hi = gboot_rho_difference(treat, twin, ref, genes, n_draws=400, seed=0)
    assert est == pytest.approx(0.35 - 0.85, abs=0.02)
    assert hi < 0.0


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(2)
    ref = np.abs(rng.normal(1.0, 0.1, 40)) + 0.5
    a, b = ref * 0.4, ref * 0.8
    genes = [f"g{i // 2}" for i in range(40)]
    first = gboot_rho_difference(a, b, ref, genes, n_draws=100, seed=7)
    second = gboot_rho_difference(a, b, ref, genes, n_draws=100, seed=7)
    assert first == second


def test_resampling_unit_is_the_gene_not_the_row():
    """One gene carrying every row must collapse the interval to a point."""
    ref = np.full(30, 1.0)
    a, b = np.full(30, 0.4), np.full(30, 0.8)
    est, lo, hi = gboot_rho_difference(a, b, ref, ["g0"] * 30, n_draws=50, seed=0)
    assert lo == pytest.approx(est)
    assert hi == pytest.approx(est)


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (-0.50, -0.30, "MATERIAL_NEGATIVE"),
        (-0.04, +0.03, "EQUIVALENT_WITHIN_SESOI"),
        (-0.20, +0.10, "UNRESOLVED"),
        (-0.05, +0.01, "UNRESOLVED"),
        (+0.06, +0.20, "MATERIAL_POSITIVE"),
    ],
)
def test_classification_treats_a_touched_boundary_as_unresolved(lo, hi, expected):
    assert classify_difference(lo, hi, 0.05) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -k "rho or twin or difference or classification" -v`
Expected: FAIL — `ImportError: cannot import name 'classify_difference'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/gfmcd/splicing/acceptor_payload.py`:

```python
import numpy as np


def rho(v: np.ndarray, ref: np.ndarray) -> float:
    """Ratio of means, i.e. ratio of sums on a common row set.

    Never mean-of-ratios: near-zero denominators make the per-example ratio explode.
    """
    return float(np.sum(v) / max(float(np.sum(ref)), 1e-9))


def pooled_twin_effect(
    per_arm: Mapping[str, np.ndarray], twins: Sequence[str]
) -> np.ndarray:
    """Row-wise mean of the non-motif twin arms, on the common row set."""
    if not twins:
        raise ValueError("at least one twin arm is required")
    return np.mean(np.vstack([np.asarray(per_arm[a], dtype=float) for a in twins]), axis=0)


def gboot_rho_difference(
    v_treatment: np.ndarray,
    v_reference_arm: np.ndarray,
    ref: np.ndarray,
    genes: Sequence[str],
    *,
    n_draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(estimate, lo, hi)`` for ``rho(treatment) - rho(reference_arm)``.

    Both ratios are recomputed inside every draw on the same resampled genes, so their
    correlation is preserved; subtracting two independently bootstrapped intervals would
    overstate the width.  The randomisation unit is the gene, never the row.
    """
    t = np.asarray(v_treatment, dtype=float)
    r = np.asarray(v_reference_arm, dtype=float)
    d = np.asarray(ref, dtype=float)
    if not (len(t) == len(r) == len(d) == len(genes)):
        raise ValueError("all arrays and the gene list must be the same length")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")

    estimate = rho(t, d) - rho(r, d)

    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        picked = rng.choice(len(keys), len(keys), replace=True)
        rows = [row for j in picked for row in by[keys[j]]]
        denom = max(float(d[rows].sum()), 1e-9)
        draws[i] = (t[rows].sum() / denom) - (r[rows].sum() / denom)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def classify_difference(lo: float, hi: float, sesoi: float) -> str:
    """Strict classification of a difference interval.  Touching a boundary is unresolved."""
    if lo > sesoi:
        return "MATERIAL_POSITIVE"
    if hi < -sesoi:
        return "MATERIAL_NEGATIVE"
    if lo > -sesoi and hi < sesoi:
        return "EQUIVALENT_WITHIN_SESOI"
    return "UNRESOLVED"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gfmcd/splicing/acceptor_payload.py tests/test_acceptor_payload.py
git commit -m "feat(m1): gene-clustered rho-difference estimator and strict classification"
```

---

### Task 4: Precision gate

The spec requires M1 to stop at `NOT_POWERABLE_ON_THIS_DESIGN` rather than reproduce `C38`'s wall, where raising n from 260 to 512 *widened* the dyadic half-width from `0.201` to `0.221`. The gate must sit on the measurement path — it consumes the same estimator the verdict uses.

**Files:**
- Modify: `src/gfmcd/splicing/acceptor_payload.py`
- Test: `tests/test_acceptor_payload.py`

**Interfaces:**
- Consumes: `gboot_rho_difference` from Task 3
- Produces: `half_width(lo, hi) -> float`, `precision_gate(lo, hi, sesoi) -> str`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_acceptor_payload.py`:

```python
from gfmcd.splicing.acceptor_payload import half_width, precision_gate


def test_half_width_is_half_the_interval():
    assert half_width(-0.10, +0.10) == pytest.approx(0.10)


def test_gate_proceeds_when_the_interval_can_clear_the_sesoi():
    assert precision_gate(-0.04, +0.04, 0.05) == "PROCEED"


def test_gate_stops_when_the_interval_cannot_resolve_the_sesoi():
    assert precision_gate(-0.30, +0.30, 0.05) == "NOT_POWERABLE_ON_THIS_DESIGN"


def test_gate_boundary_case_is_a_stop_not_a_proceed():
    """A half-width exactly at the SESOI cannot distinguish equivalence from material."""
    assert precision_gate(-0.05, +0.05, 0.05) == "NOT_POWERABLE_ON_THIS_DESIGN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -k "half_width or gate" -v`
Expected: FAIL — `ImportError: cannot import name 'half_width'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/gfmcd/splicing/acceptor_payload.py`:

```python
def half_width(lo: float, hi: float) -> float:
    """Half the width of a confidence interval."""
    return (float(hi) - float(lo)) / 2.0


def precision_gate(lo: float, hi: float, sesoi: float) -> str:
    """``PROCEED`` only when the achieved half-width is strictly inside the SESOI.

    A half-width at or above the SESOI cannot separate equivalence from a material
    difference whatever the point estimate turns out to be, so the design is not
    powerable and the stage stops instead of spending the target cohort.
    """
    return "PROCEED" if half_width(lo, hi) < sesoi else "NOT_POWERABLE_ON_THIS_DESIGN"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gfmcd/splicing/acceptor_payload.py tests/test_acceptor_payload.py
git commit -m "feat(m1): precision gate with a NOT_POWERABLE exit"
```

---

### Task 5: Calibration — planted-positive sensitivity and permuted-label FPR

Project convention (CLAUDE.md §3) is that a planted positive and a permuted-label FPR **gate the run** — they have caught decision-layer bugs before any verdict was read. Both must exercise the production decision path, not a parallel reimplementation.

The two arms are measured on the same rows, so the null "GC behaves like a twin" is exchangeability of the arm labels within a gene. Permuting that label is the correct specificity null.

**Files:**
- Modify: `src/gfmcd/splicing/acceptor_payload.py`
- Test: `tests/test_acceptor_payload.py`

**Interfaces:**
- Consumes: `gboot_rho_difference`, `classify_difference` from Task 3
- Produces: `planted_positive_check(ref, genes, *, planted_rho, twin_rho, sesoi, n_draws=400, seed=0) -> dict`, `permuted_label_fpr(v_treatment, v_twin, ref, genes, *, sesoi, n_permutations=200, n_draws=400, seed=0) -> dict`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_acceptor_payload.py`:

```python
from gfmcd.splicing.acceptor_payload import permuted_label_fpr, planted_positive_check


def _ref_and_genes(n=120, per_gene=4, seed=3):
    rng = np.random.default_rng(seed)
    ref = np.abs(rng.normal(1.0, 0.05, n)) + 0.5
    genes = [f"g{i // per_gene}" for i in range(n)]
    return ref, genes


def test_planted_positive_is_recovered_as_material_negative():
    ref, genes = _ref_and_genes()
    out = planted_positive_check(
        ref, genes, planted_rho=0.35, twin_rho=0.85, sesoi=0.05, n_draws=300, seed=0
    )
    assert out["classification"] == "MATERIAL_NEGATIVE"
    assert out["PASS"] is True


def test_planted_positive_check_can_fail_when_nothing_is_planted():
    """Sensitivity must be falsifiable: an unplanted effect must not report PASS."""
    ref, genes = _ref_and_genes()
    out = planted_positive_check(
        ref, genes, planted_rho=0.85, twin_rho=0.85, sesoi=0.05, n_draws=300, seed=0
    )
    assert out["PASS"] is False


def test_permuted_label_fpr_is_near_alpha_under_the_null():
    ref, genes = _ref_and_genes()
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.01, len(ref))
    v_a = ref * 0.85 + noise
    v_b = ref * 0.85 - noise
    out = permuted_label_fpr(
        v_a, v_b, ref, genes, sesoi=0.05, n_permutations=60, n_draws=150, seed=0
    )
    assert out["fpr"] <= 0.15
    assert out["PASS"] is True


def test_permuted_label_fpr_is_deterministic_under_a_fixed_seed():
    ref, genes = _ref_and_genes()
    v_a, v_b = ref * 0.4, ref * 0.85
    first = permuted_label_fpr(
        v_a, v_b, ref, genes, sesoi=0.05, n_permutations=20, n_draws=80, seed=5
    )
    second = permuted_label_fpr(
        v_a, v_b, ref, genes, sesoi=0.05, n_permutations=20, n_draws=80, seed=5
    )
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -k "planted or permuted" -v`
Expected: FAIL — `ImportError: cannot import name 'permuted_label_fpr'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/gfmcd/splicing/acceptor_payload.py`:

```python
def planted_positive_check(
    ref: np.ndarray,
    genes: Sequence[str],
    *,
    planted_rho: float,
    twin_rho: float,
    sesoi: float,
    n_draws: int = 400,
    seed: int = 0,
) -> dict:
    """Sensitivity: a synthetic arm at ``planted_rho`` must classify MATERIAL_NEGATIVE.

    Runs the production estimator and the production classifier, so a decision-layer bug
    fails this check rather than surviving into a verdict.  Falsifiable: with
    ``planted_rho == twin_rho`` there is nothing to detect and PASS must be False.
    """
    d = np.asarray(ref, dtype=float)
    est, lo, hi = gboot_rho_difference(
        d * planted_rho, d * twin_rho, d, genes, n_draws=n_draws, seed=seed
    )
    classification = classify_difference(lo, hi, sesoi)
    return {
        "PASS": classification == "MATERIAL_NEGATIVE",
        "planted_rho": float(planted_rho),
        "twin_rho": float(twin_rho),
        "estimate": est,
        "ci": [lo, hi],
        "classification": classification,
    }


def permuted_label_fpr(
    v_treatment: np.ndarray,
    v_twin: np.ndarray,
    ref: np.ndarray,
    genes: Sequence[str],
    *,
    sesoi: float,
    n_permutations: int = 200,
    n_draws: int = 400,
    seed: int = 0,
    alpha: float = 0.05,
    tolerance: float = 0.05,
) -> dict:
    """Specificity: swap the arm labels with one exchangeable sign per gene.

    The arms are measured on the same rows, so under the null "GC behaves like a twin"
    the two labels are exchangeable within a gene.  A material verdict on permuted labels
    is a false positive; the rate should sit near ``alpha``.
    """
    t = np.asarray(v_treatment, dtype=float)
    w = np.asarray(v_twin, dtype=float)
    d = np.asarray(ref, dtype=float)
    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)

    materials = 0
    for p in range(n_permutations):
        swap = rng.choice([False, True], size=len(keys))
        a, b = t.copy(), w.copy()
        for j, do_swap in enumerate(swap):
            if do_swap:
                rows = by[keys[j]]
                a[rows], b[rows] = w[rows], t[rows]
        _, lo, hi = gboot_rho_difference(a, b, d, genes, n_draws=n_draws, seed=seed + p)
        if classify_difference(lo, hi, sesoi).startswith("MATERIAL_"):
            materials += 1

    fpr = materials / n_permutations
    return {
        "PASS": fpr <= alpha + tolerance,
        "fpr": fpr,
        "alpha": alpha,
        "tolerance": tolerance,
        "n_permutations": n_permutations,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_acceptor_payload.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gfmcd/splicing/acceptor_payload.py tests/test_acceptor_payload.py
git commit -m "feat(m1): planted-positive sensitivity and permuted-label FPR gates"
```

---

### Task 6: The M1 runner

Composes the Task 1–4 primitives with the existing model primitives. Three stages: `smoke` (memorials only, tiny n, target sealed), `preflight` (precision gate on a development slice), `target` (the verdict, authorized only by a passing preflight artifact).

Modelled on `run_r2_donor_base_decomposition.py`, with four differences: train split not val, `GC` added, one common row set instead of per-arm subsets, and stages plus memorials.

**Files:**
- Create: `scripts/splicebert_acceptor_coupling/run_m1_payload_identity.py`
- Test: `tests/test_m1_payload_identity.py` (Task 6)

**Interfaces:**
- Consumes: `acceptor_payload.{LADDER_ARMS, REFERENCE_ALT, TWIN_ARMS, TREATMENT_ARM, common_admissible_rows, admissible_fractions, arm_dispositions, rho, pooled_twin_effect, gboot_rho_difference, classify_difference, precision_gate, planted_positive_check, permuted_label_fpr}`; `gfmcd.splicing.circuit.{CouplingTask, load_adapter, pick_device, load_examples_unscored, gboot_ratio, DEFAULT_GTF, DEFAULT_FASTA, DEFAULT_MODEL}`; `gfmcd.splicing.reverse_coupling.{acceptor_readout, destroy_donor, margins_at, edit_size}`; `gfmcd.provenance.ExperimentManifest`
- Produces: `TRAIN_EXCLUDED_CHROMOSOMES: tuple[str, ...]`, `SESOI: float`, `AC1_VAL_EFFECT: float`, `AC1_WIRING_BAND: tuple[float, float]`, `build_memorials(arm_effects, reference, edit_sizes, noop_coupling, ac_arm_mean=None, calibration=None) -> dict`, `main() -> None`; CLI `--stage {smoke,preflight,target} --out PATH [--n N] [--preflight-artifact PATH]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_m1_payload_identity.py`:

```python
"""M1 runner contract and memorials. Model-dependent cases skip without a checkpoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "splicebert_acceptor_coupling" / "run_m1_payload_identity.py"


def load_runner():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_m1_payload_identity", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_exists():
    assert RUNNER.is_file()


def test_sealed_and_validation_chromosomes_are_excluded_from_train():
    m = load_runner()
    for chrom in ("chr8", "chr9", "chr11", "chr16", "chr19"):
        assert chrom in m.TRAIN_EXCLUDED_CHROMOSOMES


def test_sesoi_is_frozen_at_the_study_convention():
    assert load_runner().SESOI == 0.05


def test_out_is_required_so_no_registered_artifact_is_overwritten(monkeypatch, capsys):
    m = load_runner()
    monkeypatch.setattr(sys, "argv", ["run_m1_payload_identity.py", "--stage", "smoke"])
    with pytest.raises(SystemExit):
        m.main()


def test_target_stage_requires_a_preflight_artifact(monkeypatch):
    """ACR17: refuse before inspecting any caller-supplied path."""
    m = load_runner()
    monkeypatch.setattr(
        sys, "argv",
        ["run_m1_payload_identity.py", "--stage", "target", "--out", "/tmp/x.json"],
    )
    with pytest.raises(SystemExit) as excinfo:
        m.main()
    assert "preflight" in str(excinfo.value).lower()


def test_calibration_is_inherited_at_target_not_recomputed(tmp_path):
    """Target must carry the preflight's calibration forward, per the R3c pattern."""
    src = RUNNER.read_text()
    assert "inherited_from" in src
    assert 'if args.stage == "target":' in src
    assert "target is not authorized" in src


def test_value_linearity_memorial_can_fail():
    """The memorial must detect a broken ladder, not merely assert an identity."""
    m = load_runner()
    good = m.build_memorials(
        arm_effects={"AC": [1.0, 1.0], "GC": [0.4, 0.4]},
        reference=[1.0, 1.0],
        edit_sizes={"AC": 2, "GC": 1},
        noop_coupling=0.0,
    )
    assert good["edit_size_matches_arm"]["PASS"] is True
    assert good["noop_alt_is_exactly_zero"]["PASS"] is True

    bad = m.build_memorials(
        arm_effects={"AC": [1.0, 1.0], "GC": [0.4, 0.4]},
        reference=[1.0, 1.0],
        edit_sizes={"AC": 1, "GC": 1},
        noop_coupling=0.3,
    )
    assert bad["edit_size_matches_arm"]["PASS"] is False
    assert bad["noop_alt_is_exactly_zero"]["PASS"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_m1_payload_identity.py -v`
Expected: FAIL — `assert RUNNER.is_file()` fails; the runner does not exist

- [ ] **Step 3: Write minimal implementation**

Create `scripts/splicebert_acceptor_coupling/run_m1_payload_identity.py`:

```python
#!/usr/bin/env python
"""M1 -- does the masked-acceptor readout treat a ``GC`` donor as a donor?

Adds the one missing cell to the operator ladder measured by R2.  ``GC`` keeps the ``G``,
destroys the ``T``, and is a real minor donor; its twins ``GA`` and ``GG`` are identical
in edit size and structure but are not motifs.  Every arm is measured on ONE common
admissible row set so the ratios are comparable.

Stages: ``smoke`` (memorials only), ``preflight`` (precision gate), ``target`` (verdict,
authorized only by a passing preflight artifact).  Train chromosomes only; chr8/9 sealed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.acceptor_payload import (  # noqa: E402
    LADDER_ARMS, REFERENCE_ALT, TREATMENT_ARM, TWIN_ARMS, admissible_fractions,
    arm_dispositions, classify_difference, common_admissible_rows,
    gboot_rho_difference, permuted_label_fpr, planted_positive_check,
    pooled_twin_effect, precision_gate, rho,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot_ratio,
    load_adapter, load_examples_unscored, pick_device,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    acceptor_readout, destroy_donor, edit_size, margins_at,
)

TRAIN_EXCLUDED_CHROMOSOMES = ("chr8", "chr9", "chr11", "chr16", "chr19")
CURRENT_TAG = "m1-payload-identity-v1"
SESOI = 0.05
SEED = 0
N_BOOT = 2000
MIN_N = 50
SMOKE_N = 8


AC1_VAL_EFFECT = 0.9763          # r1b-d0, val. A coarse wiring anchor, NOT a reproduction.
AC1_WIRING_BAND = (0.5, 1.5)     # train is a different cohort; only gross miswiring fails


def build_memorials(arm_effects, reference, edit_sizes, noop_coupling,
                    ac_arm_mean=None, calibration=None) -> dict:
    """Falsifiable checks on the measurement path.  Each can fail on real data."""
    expected_edit = {"AC": 2, "AT": 1, "GA": 1, "GG": 1, "GC": 1}
    edit_ok = all(edit_sizes.get(a) == e for a, e in expected_edit.items()
                  if a in edit_sizes)
    ref = np.asarray(reference, dtype=float)
    arms_share_rows = all(
        len(np.asarray(v, dtype=float)) == len(ref) for v in arm_effects.values()
    )
    out = {
        "edit_size_matches_arm": {
            "PASS": bool(edit_ok),
            "observed": dict(edit_sizes),
            "note": "AC edits 2 bases; every single-base arm edits 1",
        },
        "noop_alt_is_exactly_zero": {
            "PASS": bool(abs(float(noop_coupling)) < 1e-6),
            "observed": float(noop_coupling),
            "note": "writing GT over GT must produce exactly zero coupling",
        },
        "arms_share_one_row_set": {
            "PASS": bool(arms_share_rows),
            "note": "every arm measured on the common admissible rows",
        },
    }
    if ac_arm_mean is not None:
        lo, hi = AC1_WIRING_BAND
        out["ac_arm_in_ac1_wiring_band"] = {
            "PASS": bool(lo <= float(ac_arm_mean) <= hi),
            "observed": float(ac_arm_mean),
            "band": [lo, hi],
            "ac1_val_reference": AC1_VAL_EFFECT,
            "note": ("cross-artifact wiring anchor only -- train is a different cohort, "
                     "so this is NOT a reproduction of AC1 and must not be reported as one"),
        }
    if calibration is not None:
        out["planted_positive_sensitivity"] = calibration["planted_positive"]
        out["permuted_label_fpr_specificity"] = calibration["permuted_fpr"]
    return out


def _train_examples(args):
    return load_examples_unscored(
        args.gtf, args.fasta, "train", args.n, tag=CURRENT_TAG,
        allowed_chromosomes=None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "preflight", "target"), default="smoke")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--preflight-artifact", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ACR17: refuse every caller-supplied path BEFORE inspecting any of them.
    if args.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")
    if args.stage == "target" and args.preflight_artifact is None:
        raise SystemExit("--preflight-artifact is required for the target stage")
    if args.n is None:
        args.n = SMOKE_N if args.stage == "smoke" else 1000

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=False)
    task = CouplingTask(ad, batch=args.batch)

    X = [x for x in _train_examples(args)
         if x.chrom not in TRAIN_EXCLUDED_CHROMOSOMES and x.next_acceptor_pos is not None]
    rows = common_admissible_rows(X, LADDER_ARMS)
    P = [X[i] for i in rows]
    if args.stage != "smoke" and len(P) < MIN_N:
        raise SystemExit(f"only {len(P)} common-admissible rows; need {MIN_N}")

    genes = [x.gene_id for x in P]
    a_pos = [acceptor_readout(x) for x in P]
    clean = margins_at(task, a_pos, [x.sequence for x in P])

    arm_effects: dict[str, np.ndarray] = {}
    for alt in LADDER_ARMS:
        arm_effects[alt] = clean - margins_at(
            task, a_pos, [destroy_donor(x, alt) for x in P]
        )
    noop = float(np.max(np.abs(
        clean - margins_at(task, a_pos, [destroy_donor(x, "GT") for x in P])
    ))) if P else 0.0

    reference = arm_effects[REFERENCE_ALT]
    twins = pooled_twin_effect(arm_effects, TWIN_ARMS)
    est, lo, hi = gboot_rho_difference(
        arm_effects[TREATMENT_ARM], twins, reference, genes,
        n_draws=N_BOOT, seed=SEED,
    )
    gate = precision_gate(lo, hi, SESOI)

    # Calibration gates the DESIGN, so it runs at smoke and preflight and the target
    # INHERITS it -- the R3c pattern ("target inherits bucket-0 synthetic calibration").
    # It is also the expensive part: 200 permutations x 400 draws is ~80k resamples.
    if args.stage == "target":
        prior = json.loads(Path(args.preflight_artifact).read_text())
        if prior.get("verdict") != "PROCEED":
            raise SystemExit(
                f"preflight verdict is {prior.get('verdict')!r}; target is not authorized"
            )
        calibration = {
            "planted_positive": prior["memorials"]["planted_positive_sensitivity"],
            "permuted_fpr": prior["memorials"]["permuted_label_fpr_specificity"],
            "inherited_from": str(args.preflight_artifact),
        }
    else:
        calibration = {
            "planted_positive": planted_positive_check(
                reference, genes, planted_rho=0.35,
                twin_rho=rho(twins, reference), sesoi=SESOI, n_draws=400, seed=SEED,
            ),
            "permuted_fpr": permuted_label_fpr(
                arm_effects[TREATMENT_ARM], twins, reference, genes,
                sesoi=SESOI, n_permutations=200, n_draws=400, seed=SEED,
            ),
        }
    memorials = build_memorials(
        arm_effects={k: v.tolist() for k, v in arm_effects.items()},
        reference=reference.tolist(),
        edit_sizes={a: edit_size("GT", a) for a in LADDER_ARMS},
        noop_coupling=noop,
        ac_arm_mean=float(reference.mean()),
        calibration=calibration,
    )
    all_pass = all(m["PASS"] for m in memorials.values())

    verdict = "MEMORIALS_FAILED"
    if all_pass:
        if args.stage == "preflight":
            verdict = gate
        elif args.stage == "target":
            verdict = (classify_difference(lo, hi, SESOI)
                       if gate == "PROCEED" else gate)
        else:
            verdict = "SMOKE_OK"

    per_arm = {}
    for alt in LADDER_ARMS:
        arm_lo, arm_hi = gboot_ratio(
            arm_effects[alt], reference, genes, n=N_BOOT, seed=SEED
        )
        per_arm[alt] = {
            "edit_size": edit_size("GT", alt),
            "keeps_G_at_d": alt[0] == "G",
            "keeps_T_at_d_plus_1": alt[1] == "T",
            "is_donor_motif": alt in ("GT", "GC"),
            "mean": float(arm_effects[alt].mean()),
            "rho": rho(arm_effects[alt], reference),
            "rho_ci": [float(arm_lo), float(arm_hi)],
        }

    result = {
        "schema": "gfmcd.m1.payload_identity/1",
        "experiment": "m1_payload_identity",
        "stage": args.stage,
        "tier": "development",
        "split": "train",
        "sealed_chr8_chr9_opened": False,
        "n": len(P),
        "n_genes": len(set(genes)),
        "cohort": {
            "tag": CURRENT_TAG,
            "excluded_chromosomes": list(TRAIN_EXCLUDED_CHROMOSOMES),
            "screened_rows": len(X),
            "common_admissible_rows": len(P),
            "admissible_fractions": admissible_fractions(X, LADDER_ARMS),
            "carve_out": {
                "arm": TREATMENT_ARM,
                "accepted_register": "written",
                "disqualifying_registers": ["upstream", "downstream"],
            },
            "dispositions_sample": {
                x.example_id: dict(arm_dispositions(x, TREATMENT_ARM)) for x in P[:20]
            },
        },
        "arms": per_arm,
        "primary_contrast": {
            "estimand": "rho(GC) - rho(pooled GA,GG) on the common admissible rows",
            "estimate": est,
            "ci": [lo, hi],
            "sesoi": SESOI,
            "classification": classify_difference(lo, hi, SESOI),
            "precision_gate": gate,
        },
        "memorials": memorials,
        "all_memorials_pass": all_pass,
        "verdict": verdict,
        "licenses": (
            "A development-tier statement about whether this checkpoint's masked-acceptor "
            "readout distinguishes a GC donor from non-motif substitutions. No component, "
            "route, mediation, or biological claim."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m1_payload_identity", sys.argv,
        config=result["cohort"], seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; sealed chr8/9 not opened.",
               "Development tier; licenses no component, route, or biological claim."])
    for f in (args.model, args.gtf, args.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"verdict={verdict} -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra splicebert --extra biology --extra dev pytest tests/test_m1_payload_identity.py -v`
Expected: PASS

The manifest API is `ExperimentManifest.create(name, argv, config=, seeds=, status=, notes=)` then `.add_input()` / `.add_output()` / `.write(path)`. There is **no** `for_artifact` classmethod — the available methods are `create`, `add_input`, `add_output`, `to_dict`, `write`.

- [ ] **Step 5: Commit**

```bash
git add scripts/splicebert_acceptor_coupling/run_m1_payload_identity.py tests/test_m1_payload_identity.py
git commit -m "feat(m1): staged payload-identity runner with GC arm and precision gate"
```

---

### Task 7: Full-suite regression and manifest check

The new module and runner must not disturb the existing record. This task adds nothing new; it verifies.

**Files:**
- Test: existing suite

- [ ] **Step 1: Run the full suite with extras and genome env vars**

```bash
GFMCD_GTF=/Users/cberg/Research/data/genomes/gencode.v50.annotation.gtf.gz GFMCD_FASTA=/Users/cberg/Research/data/genomes/hg38.fa uv run --extra splicebert --extra biology --extra dev pytest
```

Expected: the pre-existing pass count plus the new tests; **zero new failures and zero new skips**. Record the exact counts.

- [ ] **Step 2: Verify manifests**

```bash
uv run python scripts/verify_manifest.py
```

Expected: PASS.

- [ ] **Step 3: Run the repo discipline checker**

```bash
python3 .claude/skills/gfm-repo-discipline/scripts/check_repo.py
```

Expected: no *new* FAIL relative to the pre-existing baseline of 7 (all of which are AC5–AC8 naming untracked artifacts). If the count rises, fix before continuing.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u src/gfmcd/splicing/acceptor_payload.py tests/test_acceptor_payload.py tests/test_m1_payload_identity.py
git commit -m "test(m1): green suite, manifests verified, no new discipline findings"
```

---

### Task 8: Freeze the M1 protocol

Written last, because it pins the runner's SHA-256. No model outcome exists yet, so writing it now is prereg-conformant.

**Files:**
- Create: `studies/splicebert_acceptor_coupling/preregistrations/PREREGISTRATION_M1_PAYLOAD_IDENTITY.md`

- [ ] **Step 1: Compute the runner hash**

```bash
shasum -a 256 scripts/splicebert_acceptor_coupling/run_m1_payload_identity.py
```

- [ ] **Step 2: Write the protocol**

Sections, each populated with the frozen values from this plan's Global Constraints:

1. **Status** — `FROZEN FOR TRAIN DEVELOPMENT PREFLIGHT`, with the runner SHA-256 from Step 1 and a statement that chr8/9 is not opened.
2. **Question and evidence boundary** — the `GC`-versus-twins contrast; development tier; no component, route, mediation, or biological claim follows.
3. **Cohort** — train split, `TRAIN_EXCLUDED_CHROMOSOMES`, tag `m1-payload-identity-v1`, common admissible row set, per-arm admissible fractions reported.
4. **Carve-out** — `written` register accepted for `GC` only; `upstream`/`downstream` disqualifying on every arm; dispositions serialized per row. State that a blanket waiver is not permitted.
5. **Estimand and decision rule** — `ρ_GC − ρ_pooled{GA,GG}`, gene-clustered, `SESOI = 0.05`, strict classification with a touched boundary unresolved, `PROCEED` requiring half-width strictly below the SESOI.
6. **Memorials** — the three from `build_memorials`, each stated with its failure mode. Flag any that are architecture-forced as `forced: true`; `noop_alt_is_exactly_zero` is *not* forced because the ladder could be mis-wired.
7. **Interpretation table** — `MATERIAL_NEGATIVE` ⇒ payload carries donor identity; `EQUIVALENT_WITHIN_SESOI` ⇒ payload is canonical-`GT` presence; `UNRESOLVED` or `NOT_POWERABLE_ON_THIS_DESIGN` ⇒ no interpretation, and explicitly not evidence of absence.
8. **Provenance** — runner SHA-256, `uv.lock` hash, GTF `83fba3e9…a453`, FASTA `5be01555…f5e51`, checkpoint `2ad91428…0149d`, `SEED = 0`, `N_BOOT = 2000`.

- [ ] **Step 3: Verify no retired vocabulary**

```bash
grep -n "10\.67\|10\.58\|splice-site strength" studies/splicebert_acceptor_coupling/preregistrations/PREREGISTRATION_M1_PAYLOAD_IDENTITY.md
```

Expected: no matches (exit 1).

- [ ] **Step 4: Commit**

```bash
git add studies/splicebert_acceptor_coupling/preregistrations/PREREGISTRATION_M1_PAYLOAD_IDENTITY.md
git commit -m "docs(m1): freeze the payload-identity protocol"
```

---

## After the plan

The runner exists and is frozen, but **no model run has happened**. The next actions, in order, each requiring the main checkout and the genome env vars:

1. `--stage smoke --n 8 --out <scratch>` — memorials only.
2. `--stage preflight --out results/splicebert_acceptor_coupling/m1_payload_identity_preflight.json` — read the precision gate. A `NOT_POWERABLE_ON_THIS_DESIGN` verdict stops M1 here and is a legitimate outcome.
3. `--stage target --preflight-artifact <the preflight> --out results/splicebert_acceptor_coupling/m1_payload_identity_target.json` — the verdict.

Only after a target verdict does an `AC*` ledger row become appropriate, and it is a **development-tier** row.
