"""Model-free gate battery for designed register-lattice sequence variants.

A "family" here is the set of designed sequence variants built from ONE authentic joined
donor+acceptor construct (``parent``) by editing a shared, localised region (the "seam") to
realise different reading-frame REGISTER conditions -- e.g. a 3x3 lattice of donor-side x
acceptor-side register choices. Every gate in this module is PURELY sequence-level: no model,
no checkpoint, no network. They exist to catch confounds -- an accidentally shifted frame, a
stray stop codon, a composition drift, a cryptic GT/AG created by the edit -- before a designed
family is ever shown to a genomic foundation model.

THE RULE THIS MODULE ENCODES
-----------------------------
A whole family is rejected if ANY member fails ANY gate. There is deliberately no function
anywhere in this module that returns a filtered subset of a family's members: dropping a single
failing cell would be selection on the outcome path and would silently bias the lattice. The only
supported outputs are (a) per-gate, per-member measurements for audit, and (b) one family-level
verdict. ``evaluate_family`` is the sole entry point that produces that verdict.

GATE RESULT SHAPE
------------------
Every ``gate_*`` function returns a dict with at least: the gate name, the measured value(s), the
declared threshold, and a boolean ``passed`` -- never a bare bool, so a gate's outcome can be
audited (and disputed) after the fact. ``gate_site_scores`` is the one exception that can also
carry ``available: False`` (see below).

BOUNDARY CONVENTION (applies to every gate in this module)
------------------------------------------------------------
All threshold comparisons are INCLUSIVE of the boundary: a measured deviation exactly equal to
the declared tolerance PASSES. Gates phrased as a ceiling (composition drift, GC drift, amplitude
spread, edit-distance spread, motif count, stop-codon-count band) use ``<= tolerance``. The one
gate phrased as a floor (similarity to the authentic parent, which must be AT LEAST some fraction
matched) uses ``>= tolerance``.

SITE SCORES AND "NOT EVALUATED"
---------------------------------
``maxentpy`` (github kepbod/maxentpy; not on PyPI) supplies MaxEntScan and may be absent from the
environment. When it is, ``gate_site_scores`` reports ``available: False`` with a stated reason,
and ``evaluate_family`` records the gate under ``gates_not_evaluated`` -- it is NEVER silently
counted as passing. By default an unevaluated gate BLOCKS the family (``family_passes`` is False)
because proceeding on an unevaluated gate is exactly how an unaudited construct reaches a model;
this is a deliberate, documented default and can be overridden via
``evaluate_family(..., block_unevaluated_gates=False)`` for callers that explicitly want to
record a partial verdict while a required dependency is unavailable.

REALISATION SCHEMES (``gate_designed_register`` and ``gate_seam``)
---------------------------------------------------------------------
A designed register can be REALISED two ways, and the two are not interchangeable for auditing:

* ``"rotation"`` (the default, and the PRIMARY genome-anchored operator) -- ``shift_module``
  (``designed_register.py``) circularly rotates the module interior. It is strictly
  LENGTH-PRESERVING, so "register" can never be inferred from a length delta (that inference is
  either vacuously true or vacuously false against this operator -- a gate that cannot fail is
  worse than no gate). Register is instead ``realised_register(shift) == shift % 3`` for the
  DECLARED shift, verified against an exact byte-for-byte RECONSTRUCTION check
  (``variant == shift_module(parent, shift=..., protect_5p=..., protect_3p=...)``), and the two
  checks are independent: reconstruction catches a corrupted variant or a wrong protected-flank
  width even when the declared shift's register happens to match; the register check catches a
  declared shift whose ``% 3`` is simply wrong even when it does reconstruct some (different)
  variant.
* ``"insertion"`` -- a SECONDARY scheme for an operator-robustness panel, not the primary
  mechanism assay. Register is inferred from the net length change relative to ``parent``, modulo
  3. This inference is only meaningful when the register edit really was realised by inserting
  bases; it is retained because other tests exercise it, not because it should audit
  rotation-operator variants.

Every gate that depends on the realisation scheme takes an explicit ``scheme`` argument (default
``"rotation"``) rather than guessing from the data.
"""
from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np

from gfmcd.splicing.designed_register import shift_module

STOP_CODONS = ("TAA", "TAG", "TGA")
GC_BASES = frozenset("GC")
ENCODABLE_BASES = {"A": 0.0, "T": 0.0, "U": 0.0, "G": 1.0, "C": 1.0}
PERIOD3_CYCLES_PER_SAMPLE = 1.0 / 3.0

#: The two register-realisation schemes ``gate_designed_register`` and ``gate_seam`` support. See
#: "REALISATION SCHEMES" in the module docstring.
REALISATION_SCHEMES = ("rotation", "insertion")

#: Default ``gate_seam`` tolerance per scheme (used only when a caller does not pass ``tolerance``
#: explicitly). See ``gate_seam``'s docstring for why these two values differ.
SEAM_TOLERANCE_DEFAULTS: dict[str, float] = {
    "rotation": 0.0,
    "insertion": 2.0,
}

#: Defaults merged under any caller-supplied ``tolerances``; caller keys always win. Keeping these
#: named and centralised means a test that asks "what does the boundary do" can read the exact
#: number it is probing instead of a magic literal.
DEFAULT_TOLERANCES: dict[str, float] = {
    "stop_load": 0.0,               # max |variant_stop_count - parent_stop_count|
    "composition": 0.05,            # max pairwise range of any k-mer frequency, across variants
    "gc_codon_position": 0.05,      # max pairwise range of GC / GC1 / GC2 / GC3, across variants
    "period3_amplitude": 0.10,      # max relative spread of period-3 amplitude, across variants
    "period3_phase_deg": 20.0,      # max circular distance of measured vs. expected phase
    "created_motifs": 0.0,          # max count of new GT/AG dinucleotides introduced by the edit
    "seam_edit_distance": 0.0,      # max pairwise spread of parent->variant edit distance
    "similarity_to_authentic": 0.5, # MINIMUM required fraction of the variant that is inherited
    "site_scores": 2.0,             # max pairwise range of MaxEnt score, in bits, across variants
}


# ---------------------------------------------------------------------------------------------
# Small pure helpers shared across gates
# ---------------------------------------------------------------------------------------------
def _require_nonempty(variants: Mapping[str, str]) -> None:
    if not variants:
        raise ValueError("a family must contain at least one variant")


def _require_registers(variants: Mapping[str, str], intended_registers: Mapping[str, int]) -> None:
    missing = sorted(set(variants) - set(intended_registers))
    if missing:
        raise ValueError(f"intended_registers is missing entries for: {missing}")


def _codons(sequence: str) -> list[str]:
    upper = sequence.upper()
    return [upper[i : i + 3] for i in range(0, len(upper) - 2, 3)]


def _inframe_stop_count(sequence: str) -> int:
    """Count in-frame stop codons, where "in frame" means codon boundaries at 0, 3, 6, ...

    This is a plain positional convention on the joined construct as given (frame anchored at
    its first base) -- it requires no external CDS/ORF metadata, which this module never
    receives. Callers are responsible for providing constructs that are already in the frame
    they intend to audit.
    """
    return sum(1 for codon in _codons(sequence) if codon in STOP_CODONS)


def _kmer_frequencies(sequence: str, k: int) -> dict[str, float]:
    """Frequency of every possible k-mer (order-``k`` nucleotide n-gram) in ``sequence``.

    Denominator is the number of overlapping k-mer windows, ``len(sequence) - k + 1``; zero if
    the sequence is shorter than ``k``.
    """
    upper = sequence.upper()
    keys = ["".join(t) for t in product("ACGT", repeat=k)]
    n_windows = len(upper) - k + 1
    if n_windows <= 0:
        return {key: 0.0 for key in keys}
    counts = dict.fromkeys(keys, 0)
    for i in range(n_windows):
        window = upper[i : i + k]
        if window in counts:
            counts[window] += 1
    return {key: counts[key] / n_windows for key in keys}


def _gc_fraction(bases: Sequence[str]) -> float:
    if not bases:
        return 0.0
    return sum(1 for b in bases if b in GC_BASES) / len(bases)


def numeric_encode(sequence: str) -> np.ndarray:
    """GC-content indicator encoding used for period-3 spectral analysis.

    G/C -> 1.0, A/T/U -> 0.0. This is the classic encoding for reading-frame periodicity in
    coding sequence: codon usage biases GC content by codon position (famously at the wobble /
    3rd position), which produces a genuine period-3 signal whose PHASE tracks which absolute
    offset (mod 3) is "in register" and whose AMPLITUDE is a property of composition, not of
    alignment. Any other symbol (N, ambiguity codes, lowercase already handled by ``.upper()``)
    maps to 0.5 (neutral) so a degenerate base cannot masquerade as periodic signal either way.
    """
    return np.array(
        [ENCODABLE_BASES.get(base, 0.5) for base in sequence.upper()], dtype=float
    )


def _dft_component(values: np.ndarray, cycles_per_sample: float) -> complex:
    """Direct (Goertzel-style) DFT coefficient at an arbitrary frequency, no FFT-bin quantisation.

    Evaluated by direct summation rather than via ``np.fft`` because the frequency of interest
    (1/3 cycles/sample) is only an exact FFT bin when the sequence length is a multiple of 3;
    a direct sum is well-defined for any length.
    """
    n = np.arange(len(values), dtype=float)
    weights = np.exp(-2j * np.pi * cycles_per_sample * n)
    return complex(np.sum(values * weights))


def period3_component(sequence: str) -> tuple[float, float]:
    """Amplitude and phase (radians, ``np.angle`` convention) of the period-3 DFT component.

    Amplitude is normalised by sequence length (``2*|X|/N``) so it is comparable across variants
    of slightly different length; a pure period-3 square wave of unit swing reports amplitude
    close to 1.0 regardless of ``N``. Returns ``(0.0, 0.0)`` for an empty sequence.
    """
    values = numeric_encode(sequence)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    coeff = _dft_component(values, PERIOD3_CYCLES_PER_SAMPLE)
    return float(2.0 * abs(coeff) / n), float(np.angle(coeff))


def _partial_raw_coeff(sequence: str, start: int, end: int) -> complex:
    """Raw (unnormalised) period-3 DFT contribution of ``sequence[start:end]`` AT ITS NATIVE
    (whole-sequence) indices -- i.e. the exponent uses the absolute position ``n``, not an offset
    reset to 0. This is the building block for ``_expected_seam_edit_coeff``: splitting a
    sequence's raw coefficient at position ``p`` gives ``coeff(seq) ==
    _partial_raw_coeff(seq,0,p) + _partial_raw_coeff(seq,p,len(seq))`` exactly, by linearity.
    """
    values = numeric_encode(sequence[start:end])
    n = np.arange(start, end, dtype=float)
    weights = np.exp(-2j * np.pi * PERIOD3_CYCLES_PER_SAMPLE * n)
    return complex(np.sum(values * weights))


def _expected_seam_edit_coeff(parent: str, seam_start: int, seam_end: int, register: int) -> complex:
    """Raw period-3 coefficient predicted for "replace ``parent[seam_start:seam_end]`` with
    ``register`` neutral (A/T-encoded) bases", evaluated EXACTLY (not by naively rotating the
    whole-sequence coefficient).

    Splitting the sum at the seam matters: the part of ``parent`` before the seam sits at
    unchanged native positions and contributes a FIXED term; only the part from ``seam_end``
    onward is displaced by the edit, and picks up a phase factor ``exp(-2*pi*i*register/3)`` for
    that displacement. Naively rotating the parent's WHOLE-sequence coefficient by that same
    factor is only correct in the degenerate case ``seam_start == 0`` (edit at the very front);
    for an internal seam it silently rotates the unchanged upstream flank too, which for a long
    flank swamps the real, local, register-carried signal -- an earlier version of this gate did
    exactly that and produced spurious ~120 degree "errors" on every correctly-realised variant
    whose seam sat away from position 0 (see ``test_gate_period3_clean_family_passes``, which
    pins the fix: measured phase must match this seam-aware prediction EXACTLY, to floating-point
    precision, whenever the edit really is `register` neutral bases inserted at the seam).
    """
    before = _partial_raw_coeff(parent, 0, seam_start)
    after = _partial_raw_coeff(parent, seam_end, len(parent))
    shift = np.exp(-2j * np.pi * PERIOD3_CYCLES_PER_SAMPLE * register)
    return before + shift * after


def _circular_distance_deg(a_rad: float, b_rad: float) -> float:
    """Smallest angular distance between two phases, in degrees, always in [0, 180]."""
    diff = (a_rad - b_rad + np.pi) % (2 * np.pi) - np.pi
    return float(abs(np.degrees(diff)))


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _common_suffix_len(a: str, b: str, max_len: int) -> int:
    n = 0
    while n < max_len and n < len(a) and n < len(b) and a[-(n + 1)] == b[-(n + 1)]:
        n += 1
    return n


def seam_span(variant: str, parent: str) -> dict[str, int]:
    """Locate the shared "seam" -- the edited region -- between ``variant`` and ``parent``.

    Returns the length of the byte-identical common prefix and common suffix (constrained not to
    overlap), and the resulting seam interval endpoints in both sequences' own coordinates. The
    seam is what gate 1 (register), gate 6 (created motifs), and gate 7 (seam match) all key off.
    """
    max_flank = min(len(variant), len(parent))
    prefix = _common_prefix_len(variant, parent)
    prefix = min(prefix, max_flank)
    suffix = _common_suffix_len(variant, parent, max_flank - prefix)
    return {
        "prefix_len": prefix,
        "suffix_len": suffix,
        "variant_seam_start": prefix,
        "variant_seam_end": len(variant) - suffix,
        "parent_seam_start": prefix,
        "parent_seam_end": len(parent) - suffix,
    }


def _levenshtein(a: str, b: str) -> int:
    """Standard O(len(a)*len(b)) edit distance, rolling-row DP."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _longest_common_substring_len(a: str, b: str) -> int:
    """Standard O(len(a)*len(b)) longest-common-substring DP (rolling rows)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr[j] = prev[j - 1] + 1
                best = max(best, curr[j])
        prev = curr
    return best


def _dinucleotide_positions(sequence: str, motif: str) -> list[int]:
    upper = sequence.upper()
    return [i for i in range(len(upper) - 1) if upper[i : i + 2] == motif]


# ---------------------------------------------------------------------------------------------
# Gate 1 -- designed register
# ---------------------------------------------------------------------------------------------
def realised_register(shift: int) -> int:
    """The register realised by a rotation shift: ``shift % 3``.

    This is the single source of truth for "what register does shift ``shift`` realise" against
    the rotation operator (``designed_register.shift_module``). It is NEVER a function of length:
    ``shift_module`` is strictly length-preserving (it circularly rotates the module interior), so
    a length-based inference of register is meaningless against it -- against a length-preserving
    operator, ``(len(variant) - len(parent)) % 3`` is always the same value (0) regardless of the
    real shift, which means a length-based register gate either always passes or always fails, no
    matter what register was actually realised. ``shift % 3`` has no such degeneracy and is
    well-defined for any integer ``shift``, including negative shifts (Python's ``%`` is always
    non-negative for a positive modulus, matching how ``shift_module`` itself reduces ``shift``
    modulo the interior length) and multiples of 3 (register 0).
    """
    return shift % 3


def gate_designed_register(
    variants: Mapping[str, str],
    *,
    parent: str,
    intended_registers: Mapping[str, int],
    scheme: str = "rotation",
    shifts: Mapping[str, int] | None = None,
    protect_5p: int | None = None,
    protect_3p: int | None = None,
) -> dict:
    """The realised register of each variant must equal its intended register, EXACTLY.

    Two mutually exclusive realisation schemes are supported via ``scheme`` (see "REALISATION
    SCHEMES" in the module docstring):

    * ``"rotation"`` (default, PRIMARY): requires ``shifts`` (the DECLARED shift for each
      variant), ``protect_5p``, and ``protect_3p``. A variant passes only if BOTH:

      1. ``realised_register(shifts[name])`` equals the intended register, and
      2. the variant reconstructs EXACTLY: ``variant == shift_module(parent, shift=shifts[name],
         protect_5p=protect_5p, protect_3p=protect_3p)``, byte for byte.

      These two checks are independent and both required -- (1) alone would pass a variant whose
      declared shift has the right register but whose actual bytes were never produced by that
      shift (a corrupted variant, or a wrong protected-flank width); (2) alone would not catch a
      declared shift that reconstructs some variant but is not the one the study intended to
      claim. A variant whose length differs from ``parent`` is rejected immediately with a stated
      reason: ``shift_module`` cannot produce it, so no declared shift could be correct, and there
      is no length delta to check reconstruction against.

    * ``"insertion"`` (SECONDARY -- an operator-robustness panel, not the primary mechanism
      assay): "realised register" is inferred structurally from the net change in length relative
      to ``parent``, modulo 3 -- the frame shift that length change induces at the join. This
      inference is only meaningful when the register edit really was realised by inserting bases;
      retained for callers that exercise that scheme.

    Boundary: exact match required in both schemes -- this gate is deliberately not
    tolerance-based, a register is either right or it is a different experimental condition
    wearing the wrong label.
    """
    _require_nonempty(variants)
    _require_registers(variants, intended_registers)
    if scheme not in REALISATION_SCHEMES:
        raise ValueError(f"scheme must be one of {REALISATION_SCHEMES}, got {scheme!r}")

    measured: dict[str, dict] = {}
    failing_members: list[str] = []

    if scheme == "rotation":
        if shifts is None or protect_5p is None or protect_3p is None:
            raise ValueError(
                "scheme='rotation' requires 'shifts' (the declared shift per variant), "
                "'protect_5p', and 'protect_3p' -- against a length-preserving operator, register "
                "is a property of the declared shift plus reconstruction, never of length"
            )
        missing_shifts = sorted(set(variants) - set(shifts))
        if missing_shifts:
            raise ValueError(f"shifts is missing entries for: {missing_shifts}")

        for name, seq in variants.items():
            intended = int(intended_registers[name]) % 3
            shift = int(shifts[name])
            realised = realised_register(shift)
            register_match = realised == intended

            if len(seq) != len(parent):
                measured[name] = {
                    "shift": shift,
                    "realised": realised,
                    "intended": intended,
                    "register_match": register_match,
                    "reconstructed": False,
                    "match": False,
                    "reason": (
                        f"length changed under scheme='rotation' ({len(seq)} vs parent "
                        f"{len(parent)}) -- shift_module is strictly length-preserving, so no "
                        "declared shift can reconstruct this variant"
                    ),
                }
                failing_members.append(name)
                continue

            expected = shift_module(
                parent, shift=shift, protect_5p=protect_5p, protect_3p=protect_3p
            )
            reconstructed = seq == expected
            match = register_match and reconstructed
            entry = {
                "shift": shift,
                "realised": realised,
                "intended": intended,
                "register_match": register_match,
                "reconstructed": reconstructed,
                "match": match,
            }
            if not reconstructed:
                entry["reason"] = (
                    "variant does not equal shift_module(parent, shift=<declared>, "
                    "protect_5p=..., protect_3p=...) byte for byte -- a wrong declared shift, a "
                    "wrong protected-flank width, or a corrupted variant all fail this check"
                )
            elif not register_match:
                entry["reason"] = "declared shift's register does not match the intended register"
            measured[name] = entry
            if not match:
                failing_members.append(name)
    else:  # scheme == "insertion"
        for name, seq in variants.items():
            realised = (len(seq) - len(parent)) % 3
            intended = int(intended_registers[name]) % 3
            match = realised == intended
            measured[name] = {"realised": realised, "intended": intended, "match": match}
            if not match:
                failing_members.append(name)

    return {
        "gate": "designed_register",
        "scheme": scheme,
        "threshold": "exact match (registers compared modulo 3)",
        "measured": measured,
        "failing_members": failing_members,
        "passed": not failing_members,
    }


# ---------------------------------------------------------------------------------------------
# Gate 2 -- stop-codon load
# ---------------------------------------------------------------------------------------------
def gate_stop_load(variants: Mapping[str, str], *, parent: str, tolerance: float) -> dict:
    """In-frame stop-codon count per variant, banded against the family's reference (``parent``).

    Boundary: a deviation exactly equal to ``tolerance`` PASSES (``<=``).
    """
    _require_nonempty(variants)
    reference = _inframe_stop_count(parent)
    measured: dict[str, dict] = {}
    failing_members: list[str] = []
    for name, seq in variants.items():
        count = _inframe_stop_count(seq)
        deviation = abs(count - reference)
        ok = deviation <= tolerance
        measured[name] = {"count": count, "reference": reference, "deviation": deviation, "passed": ok}
        if not ok:
            failing_members.append(name)
    return {
        "gate": "stop_load",
        "threshold": float(tolerance),
        "reference": reference,
        "measured": measured,
        "failing_members": failing_members,
        "passed": not failing_members,
    }


# ---------------------------------------------------------------------------------------------
# Gate 3 -- composition
# ---------------------------------------------------------------------------------------------
def gate_composition(variants: Mapping[str, str], *, tolerance: float) -> dict:
    """Mono/di/tri-nucleotide frequency vectors, matched across variants (not vs. a reference).

    For each k-mer order the deviation is the maximum pairwise range (max - min) of any single
    frequency component, taken across the family's variants; the gate reports the max of that
    across all three orders. Boundary: a range exactly equal to ``tolerance`` PASSES (``<=``).
    """
    _require_nonempty(variants)
    names = list(variants)
    vectors = {order: {name: _kmer_frequencies(variants[name], order) for name in names} for order in (1, 2, 3)}
    per_order_max_range: dict[int, float] = {}
    worst: dict[int, str] = {}
    for order, by_name in vectors.items():
        keys = next(iter(by_name.values())).keys()
        ranges = {
            key: max(by_name[n][key] for n in names) - min(by_name[n][key] for n in names)
            for key in keys
        }
        worst_key = max(ranges, key=ranges.get)
        per_order_max_range[order] = ranges[worst_key]
        worst[order] = worst_key
    overall_max = max(per_order_max_range.values())
    passed = overall_max <= tolerance
    return {
        "gate": "composition",
        "threshold": float(tolerance),
        "measured": {
            "max_range_by_order": {str(k): v for k, v in per_order_max_range.items()},
            "worst_kmer_by_order": {str(k): v for k, v in worst.items()},
            "overall_max_range": overall_max,
            "vectors": {str(order): vectors[order] for order in vectors},
        },
        "failing_members": [],  # a range is a family-wide property, not attributable to one cell
        "passed": passed,
    }


# ---------------------------------------------------------------------------------------------
# Gate 4 -- GC and codon-position GC
# ---------------------------------------------------------------------------------------------
def gate_gc_and_codon_position(variants: Mapping[str, str], *, tolerance: float) -> dict:
    """Overall GC and GC1/GC2/GC3 (codon-position GC), matched across variants.

    GC1/GC2/GC3 use the 1st/2nd/3rd codon position of each in-frame codon (frame anchored at
    the sequence's first base, matching ``gate_stop_load``'s convention). Boundary: a range
    exactly equal to ``tolerance`` PASSES (``<=``).
    """
    _require_nonempty(variants)
    names = list(variants)
    per_variant: dict[str, dict] = {}
    for name in names:
        upper = variants[name].upper()
        codon_len = (len(upper) // 3) * 3
        trimmed = upper[:codon_len]
        pos = {1: trimmed[0::3], 2: trimmed[1::3], 3: trimmed[2::3]}
        per_variant[name] = {
            "gc": _gc_fraction(upper),
            "gc1": _gc_fraction(pos[1]),
            "gc2": _gc_fraction(pos[2]),
            "gc3": _gc_fraction(pos[3]),
        }
    ranges = {
        metric: max(per_variant[n][metric] for n in names) - min(per_variant[n][metric] for n in names)
        for metric in ("gc", "gc1", "gc2", "gc3")
    }
    overall_max = max(ranges.values())
    passed = overall_max <= tolerance
    return {
        "gate": "gc_and_codon_position",
        "threshold": float(tolerance),
        "measured": {"per_variant": per_variant, "ranges": ranges, "overall_max_range": overall_max},
        "failing_members": [],
        "passed": passed,
    }


# ---------------------------------------------------------------------------------------------
# Gate 5 -- period-3 amplitude and phase
# ---------------------------------------------------------------------------------------------
def gate_period3(
    variants: Mapping[str, str],
    *,
    parent: str,
    intended_registers: Mapping[str, int],
    amplitude_tolerance: float,
    phase_tolerance_deg: float,
    scheme: str = "rotation",
    protect_5p: int | None = None,
    protect_3p: int | None = None,
) -> dict:
    """Amplitude and phase of each variant's period-3 DFT component.

    AMPLITUDE must be matched across variants (a register edit that only touches the seam
    preserves the bulk of the signal): the gate reports the maximum relative spread of amplitude
    across the family (measured directly on each full variant sequence) and compares it to
    ``amplitude_tolerance``.

    PHASE is deliberately DIFFERENT per variant and is checked against each variant's *intended*
    register, not required to match its siblings. The expected phase is computed EXACTLY (not by
    naively rotating the parent's whole-sequence phase, which is only correct when the seam sits
    at position 0): each variant's own seam is located against ``parent`` via ``seam_span``, the
    parent is split at that seam into an unchanged upstream part and a part that would be
    displaced by ``register`` neutral bases, and the two raw DFT contributions are recombined
    with the exact phase factor for that displacement (``_expected_seam_edit_coeff``). A long
    flank on either side of a small seam edit is why the naive whole-sequence rotation fails in
    practice: the flank's own coefficient swamps the small, local, register-carried term, and
    comparing against a formula that assumes the whole sequence rotated produces spurious ~120
    degree "errors" that track nothing but the flank's size. ``phase_tolerance_deg`` bounds the
    circular distance between measured and (correctly) expected phase.

    SCHEME. The expected-phase model (``_expected_seam_edit_coeff``) encodes the INSERTION
    geometry -- "replace the seam span with ``register`` new neutral bases", so everything
    downstream of the seam shifts uniformly by ``register`` positions. A ``"rotation"`` edit does
    something different in kind: nothing downstream of the interior moves at all, because the
    interior is permuted in place at fixed absolute positions. Applying the insertion model to a
    rotation variant produces reliably ~60-120 degrees of spurious error per unit of register,
    far outside the 20 degree default -- it would fail genuinely correct families.

    There is no closed-form predicted phase for a partial-interior circular rotation with fixed
    flanks, so under ``scheme="rotation"`` the phase is MEASURED AND REPORTED BUT NOT GATED, and
    ``phase_gated`` is False with a stated reason. That is deliberately not the same thing as a
    silently skipped gate: the register itself is verified EXACTLY, byte for byte, by
    ``gate_designed_register``'s reconstruction check, which is a stronger test than any phase
    tolerance. Gating here on a formula that does not describe the operator would be a check that
    fails correct work while proving nothing.

    AMPLITUDE, and where it must be measured. A circular rotation multiplies every DFT bin of the
    ROTATED INTERIOR by a phase, so the interior's period-3 magnitude is EXACTLY invariant. The
    WHOLE variant's magnitude is not: the protected flanks do not rotate, so the total is
    |X_flank + X_interior * w^q| and the flank-interior interference makes it vary with the
    register. Measuring amplitude on the whole sequence therefore fails correct rotation families
    for a reason that has nothing to do with a defect -- it is the same interference that makes a
    joined construct's period-3 power depend on register agreement at all.

    So under ``scheme="rotation"``, when ``protect_5p``/``protect_3p`` are supplied, amplitude is
    measured on the interior, where invariance is exact rather than approximate. Without them the
    gate falls back to the whole sequence and says so in its output, because silently applying a
    whole-sequence tolerance to a rotation family is how a correct family gets rejected.

    Boundary: checks PASS when the measured value is exactly at the tolerance (``<=``).
    """
    _require_nonempty(variants)
    _require_registers(variants, intended_registers)
    if scheme not in SEAM_TOLERANCE_DEFAULTS:
        raise ValueError(
            f"unknown scheme {scheme!r}; expected one of {sorted(SEAM_TOLERANCE_DEFAULTS)}"
        )
    phase_gated = scheme == "insertion"
    parent_amplitude, parent_phase = period3_component(parent)
    interior_measured = (
        scheme == "rotation" and protect_5p is not None and protect_3p is not None
    )

    def _amplitude_region(seq: str) -> str:
        if not interior_measured:
            return seq
        return seq[int(protect_5p): len(seq) - int(protect_3p)]

    measured: dict[str, dict] = {}
    amplitudes: list[float] = []
    phase_failures: list[str] = []
    for name, seq in variants.items():
        amplitude, phase = period3_component(seq)
        amplitudes.append(period3_component(_amplitude_region(seq))[0])
        register = int(intended_registers[name]) % 3
        span = seam_span(seq, parent)
        expected_coeff = _expected_seam_edit_coeff(
            parent, span["parent_seam_start"], span["parent_seam_end"], register
        )
        expected_phase = float(np.angle(expected_coeff))
        phase_error_deg = _circular_distance_deg(phase, expected_phase)
        phase_ok = phase_error_deg <= phase_tolerance_deg
        measured[name] = {
            "amplitude": amplitude,
            "phase_rad": phase,
            "phase_deg": float(np.degrees(phase)),
            "intended_register": register,
            "expected_phase_deg": float(np.degrees(expected_phase)) % 360.0,
            "phase_error_deg": phase_error_deg,
            "phase_passed": phase_ok,
        }
        if phase_gated and not phase_ok:
            phase_failures.append(name)
    mean_amplitude = float(np.mean(amplitudes)) if amplitudes else 0.0
    if mean_amplitude > 0:
        amplitude_spread = max(abs(a - mean_amplitude) for a in amplitudes) / mean_amplitude
    else:
        amplitude_spread = 0.0
    amplitude_ok = amplitude_spread <= amplitude_tolerance
    # Amplitude is a family-wide property (a relative spread across variants), so an amplitude
    # failure implicates every member, unioned with whichever members individually failed the
    # (per-member) phase check -- but every member always appears in `measured` for audit either
    # way, per the module's no-drop rule.
    failing_members = sorted(set(phase_failures) | (set(variants) if not amplitude_ok else set()))
    passed = amplitude_ok and not phase_failures
    return {
        "gate": "period3",
        "scheme": scheme,
        "phase_gated": phase_gated,
        "amplitude_measured_on": "interior" if interior_measured else "whole_sequence",
        "amplitude_region_note": (
            None
            if interior_measured
            else (
                "measured on the WHOLE sequence. Under scheme='rotation' the protected flanks do "
                "not rotate, so whole-sequence period-3 magnitude varies with register through "
                "flank-interior interference and is not an invariant. Pass protect_5p/protect_3p "
                "to measure the interior, where rotation invariance is exact."
            )
        ),
        "phase_not_gated_reason": (
            None
            if phase_gated
            else (
                "a circular rotation permutes the interior in place with fixed flanks, so there "
                "is no closed-form predicted phase; the insertion model would produce ~60-120 "
                "degrees of spurious error and fail correct families. The register is verified "
                "EXACTLY, byte for byte, by gate_designed_register's reconstruction check. Phase "
                "is measured and reported here, not used as a pass/fail criterion."
            )
        ),
        "threshold": {
            "amplitude_relative_spread": float(amplitude_tolerance),
            "phase_deg": float(phase_tolerance_deg),
        },
        "parent": {"amplitude": parent_amplitude, "phase_rad": parent_phase, "phase_deg": float(np.degrees(parent_phase)) % 360.0},
        "measured": measured,
        "amplitude_relative_spread": amplitude_spread,
        "amplitude_passed": amplitude_ok,
        "failing_members": failing_members,
        "passed": passed,
    }


# ---------------------------------------------------------------------------------------------
# Gate 6 -- created motifs (GT / AG introduced by the edit)
# ---------------------------------------------------------------------------------------------
def _created_motif_positions(variant: str, parent: str, motif: str, span: dict[str, int]) -> list[int]:
    """Positions of ``motif`` in ``variant`` that have no counterpart at the same aligned position
    in ``parent``. Positions inside the common prefix or common suffix are, by construction,
    byte-identical to parent and therefore inherited (never "created"); positions inside the
    seam (the edited span) have no parent counterpart and count as created whenever they carry
    the motif.
    """
    created = []
    prefix, suffix = span["prefix_len"], span["suffix_len"]
    variant_len = len(variant)
    for pos in _dinucleotide_positions(variant, motif):
        in_prefix = pos + 2 <= prefix
        in_suffix = pos >= variant_len - suffix
        if not (in_prefix or in_suffix):
            created.append(pos)
    return created


def gate_created_motifs(variants: Mapping[str, str], *, parent: str, tolerance: float = 0.0) -> dict:
    """GT and AG dinucleotides present in a variant's edited seam but absent from the parent.

    A motif in the common prefix or common suffix (byte-identical to ``parent``) is inherited,
    never "created". Boundary: a created-motif count exactly equal to ``tolerance`` PASSES
    (``<=``); the default tolerance is 0 -- no new splice-like dinucleotide is expected.
    """
    _require_nonempty(variants)
    measured: dict[str, dict] = {}
    failing_members: list[str] = []
    for name, seq in variants.items():
        span = seam_span(seq, parent)
        created = {
            motif: _created_motif_positions(seq, parent, motif, span) for motif in ("GT", "AG")
        }
        total = sum(len(v) for v in created.values())
        ok = total <= tolerance
        measured[name] = {"created": created, "count": total, "passed": ok}
        if not ok:
            failing_members.append(name)
    return {
        "gate": "created_motifs",
        "threshold": float(tolerance),
        "measured": measured,
        "failing_members": failing_members,
        "passed": not failing_members,
    }


# ---------------------------------------------------------------------------------------------
# Gate 7 -- seam match (edit-matched design)
# ---------------------------------------------------------------------------------------------
def _seam_distance(variant: str, parent: str, *, scheme: str) -> int:
    """The parent->variant distance ``gate_seam`` compares across a family, chosen per ``scheme``.

    ``"rotation"``: plain POSITIONAL Hamming distance (same-index comparison), which is what is
    actually invariant across a ``register_triplet``-built triplet. For an interior with no two
    characters equal at the tested offset, a rotation by ANY nonzero amount (mod interior length)
    changes EVERY interior position -- a fixed point at index i would require ``interior[i] ==
    interior[(i + shift) % L]``, impossible for distinct-enough content unless ``shift % L == 0``
    -- so Hamming distance to ``parent`` equals the interior length for every non-identity arm,
    with EXACTLY zero spread across the family. General Levenshtein distance (``_levenshtein``) is
    NOT this invariant: its optimal alignment can exploit an indel shortcut (delete the wrapped
    span, reinsert it elsewhere) whose cost is ``2 * min(shift, L - shift)``, which differs across
    a triplet's three distinct shifts -- using Levenshtein here would resurrect a
    scheme-mismatched tolerance bug in a new place, the same shape as the length-based register
    bug this module was written to fix. Requires equal-length inputs (callers must check length
    equality themselves; ``gate_seam`` does, per-member, before calling this).

    ``"insertion"``: Levenshtein edit distance (``_levenshtein``), because an insertion-based
    register realisation changes length, for which positional Hamming distance is undefined.
    """
    if scheme == "rotation":
        return sum(1 for a, b in zip(variant, parent) if a != b)
    return _levenshtein(variant, parent)


def gate_seam(
    variants: Mapping[str, str],
    *,
    parent: str,
    scheme: str = "rotation",
    tolerance: float | None = None,
) -> dict:
    """Seam position and edit distance, matched ACROSS VARIANTS -- what makes them edit-matched.

    "Seam position" is the parent-coordinate span edited (``parent_seam_start``,
    ``parent_seam_end``), which must be IDENTICAL across every variant, in both schemes.

    "Edit distance" is SCHEME-DEPENDENT (see ``_seam_distance``), because the two realisation
    schemes are edit-matched by different invariants:

    * ``"rotation"`` (default): positional Hamming distance. A triplet built by
      ``register_triplet`` + ``shift_module`` with a shared ``k`` displaces the same interior with
      three distinct nonzero shifts, and Hamming distance to ``parent`` is EXACTLY equal (the full
      interior length) for every arm -- hence a tolerance of 0. A variant whose length differs
      from ``parent`` cannot be a rotation at all (``shift_module`` is strictly length-preserving);
      it is reported per-member with a stated reason and fails the gate, rather than raising.
    * ``"insertion"``: Levenshtein edit distance. An insertion-scheme triplet realising registers
      0/+1/+2 via ``r in {0, 1, 2}`` extra neutral bases necessarily spans an edit-distance range
      of 2 across the family -- hence the insertion-scheme default tolerance of 2.

    When ``tolerance`` is not given explicitly, it defaults per ``scheme``
    (``SEAM_TOLERANCE_DEFAULTS``); passing ``tolerance`` explicitly always overrides that default.
    Boundary: a spread exactly equal to ``tolerance`` PASSES (``<=``).
    """
    if scheme not in REALISATION_SCHEMES:
        raise ValueError(f"scheme must be one of {REALISATION_SCHEMES}, got {scheme!r}")
    if tolerance is None:
        tolerance = SEAM_TOLERANCE_DEFAULTS[scheme]
    _require_nonempty(variants)
    names = list(variants)
    measured: dict[str, dict] = {}
    for name in names:
        seq = variants[name]
        span = seam_span(seq, parent)
        entry = {
            "parent_seam_start": span["parent_seam_start"],
            "parent_seam_end": span["parent_seam_end"],
            "variant_seam_start": span["variant_seam_start"],
            "variant_seam_end": span["variant_seam_end"],
        }
        if scheme == "rotation" and len(seq) != len(parent):
            entry["edit_distance"] = None
            entry["reason"] = (
                f"length changed under scheme='rotation' ({len(seq)} vs parent {len(parent)}) "
                "-- positional Hamming distance is undefined and shift_module cannot have "
                "produced this variant"
            )
        else:
            entry["edit_distance"] = _seam_distance(seq, parent, scheme=scheme)
        measured[name] = entry

    seam_positions = {(measured[n]["parent_seam_start"], measured[n]["parent_seam_end"]) for n in names}
    positions_match = len(seam_positions) <= 1
    distances = [measured[n]["edit_distance"] for n in names]
    has_undefined = any(d is None for d in distances)
    defined = [d for d in distances if d is not None]
    edit_spread = (max(defined) - min(defined)) if defined else 0.0
    edit_ok = (not has_undefined) and edit_spread <= tolerance
    passed = positions_match and edit_ok
    return {
        "gate": "seam",
        "scheme": scheme,
        "threshold": float(tolerance),
        "measured": measured,
        "seam_positions_match": positions_match,
        "edit_distance_spread": edit_spread,
        "failing_members": [] if passed else list(names),
        "passed": passed,
    }


# ---------------------------------------------------------------------------------------------
# Gate 8 -- similarity to the authentic parent
# ---------------------------------------------------------------------------------------------
def gate_similarity_to_authentic(variants: Mapping[str, str], *, parent: str, tolerance: float) -> dict:
    """Fraction of each variant inherited verbatim from ``parent``, plus the longest common substring.

    "Inherited verbatim" = the union of the common prefix and common suffix found by
    ``seam_span`` (both are, by construction, exact substrings of ``parent``); the "longest
    common substring" is computed independently and may exceed either flank alone if the edited
    region happens to reproduce another stretch of ``parent``. This is a FLOOR, not a ceiling:
    ``tolerance`` is the minimum required fraction. Boundary: a fraction exactly equal to
    ``tolerance`` PASSES (``>=``).
    """
    _require_nonempty(variants)
    measured: dict[str, dict] = {}
    failing_members: list[str] = []
    for name, seq in variants.items():
        span = seam_span(seq, parent)
        inherited = span["prefix_len"] + span["suffix_len"]
        fraction = inherited / len(seq) if len(seq) else 0.0
        lcs_len = _longest_common_substring_len(seq, parent)
        ok = fraction >= tolerance
        measured[name] = {
            "inherited_fraction": fraction,
            "longest_common_substring_len": lcs_len,
            "passed": ok,
        }
        if not ok:
            failing_members.append(name)
    return {
        "gate": "similarity_to_authentic",
        "threshold": float(tolerance),
        "measured": measured,
        "failing_members": failing_members,
        "passed": not failing_members,
    }


# ---------------------------------------------------------------------------------------------
# Gate 9 -- MaxEntScan site scores (optional dependency)
# ---------------------------------------------------------------------------------------------
def _load_maxent():
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix3, load_matrix5

    return maxent, load_matrix5(), load_matrix3()


def _nearest_motif_window(seq: str, search_start: int, search_end: int, motif: str, upstream: int, downstream: int):
    """Best-effort donor/acceptor window: the ``motif`` occurrence closest to the seam whose
    MaxEntScan window (``upstream`` exonic/intronic bases, the motif, ``downstream`` bases) fits
    inside ``seq``. Returns ``None`` if no in-bounds occurrence exists in ``[search_start,
    search_end)``.
    """
    upper = seq.upper()
    lo = max(0, search_start)
    hi = min(len(upper) - 1, search_end)
    candidates = [p for p in _dinucleotide_positions(upper, motif) if lo <= p <= hi]
    candidates = [p for p in candidates if p - upstream >= 0 and p + 2 + downstream <= len(upper)]
    if not candidates:
        return None
    center = (search_start + search_end) // 2
    best = min(candidates, key=lambda p: abs(p - center))
    return upper[best - upstream : best + 2 + downstream]


def gate_site_scores(variants: Mapping[str, str], *, parent: str, tolerance: float) -> dict:
    """MaxEntScan donor (9nt) / acceptor (23nt) scores near the seam, matched across variants.

    Requires the optional ``maxentpy`` dependency (github kepbod/maxentpy, not on PyPI). When it
    is unavailable this returns ``available: False`` with a stated reason and no score
    measurements -- callers (``evaluate_family``) MUST treat this as "not evaluated", never as
    "passed": a gate that silently counts as passing when it did not run is exactly the failure
    mode this module exists to prevent.

    When available, the donor/acceptor windows are located heuristically as the GT/AG occurrence
    nearest the seam whose fixed-width MaxEntScan window fits inside the sequence; a variant
    where no such window exists is reported with a null score and does not by itself fail the
    gate (its deviation is simply not computable), which is also surfaced in ``measured``.
    """
    _require_nonempty(variants)
    try:
        maxent, matrix5, matrix3 = _load_maxent()
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "gate": "site_scores",
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "threshold": float(tolerance),
            "measured": {},
            "failing_members": [],
            "passed": None,
        }

    measured: dict[str, dict] = {}
    donor_scores: list[float] = []
    acceptor_scores: list[float] = []
    for name, seq in variants.items():
        span = seam_span(seq, parent)
        start, end = span["variant_seam_start"], span["variant_seam_end"]
        donor_window = _nearest_motif_window(seq, start, end, "GT", upstream=3, downstream=6)
        acceptor_window = _nearest_motif_window(seq, start, end, "AG", upstream=20, downstream=2)
        donor_score = float(maxent.score5(donor_window, matrix=matrix5)) if donor_window else None
        acceptor_score = float(maxent.score3(acceptor_window, matrix=matrix3)) if acceptor_window else None
        if donor_score is not None:
            donor_scores.append(donor_score)
        if acceptor_score is not None:
            acceptor_scores.append(acceptor_score)
        measured[name] = {
            "donor_window": donor_window,
            "donor_score": donor_score,
            "acceptor_window": acceptor_window,
            "acceptor_score": acceptor_score,
        }
    donor_spread = (max(donor_scores) - min(donor_scores)) if len(donor_scores) > 1 else 0.0
    acceptor_spread = (max(acceptor_scores) - min(acceptor_scores)) if len(acceptor_scores) > 1 else 0.0
    passed = donor_spread <= tolerance and acceptor_spread <= tolerance
    return {
        "gate": "site_scores",
        "available": True,
        "threshold": float(tolerance),
        "measured": measured,
        "donor_score_spread": donor_spread,
        "acceptor_score_spread": acceptor_spread,
        "failing_members": [] if passed else list(variants),
        "passed": passed,
    }


# ---------------------------------------------------------------------------------------------
# The family-level verdict -- the only entry point that may reject a family
# ---------------------------------------------------------------------------------------------
def evaluate_family(
    variants: Mapping[str, str],
    *,
    parent: str,
    intended_registers: Mapping[str, int],
    tolerances: Mapping[str, float],
    scheme: str = "rotation",
    shifts: Mapping[str, int] | None = None,
    protect_5p: int | None = None,
    protect_3p: int | None = None,
    block_unevaluated_gates: bool = True,
) -> dict:
    """Run the full model-free gate battery and return ONE family-level verdict.

    ``scheme`` (default ``"rotation"``, see "REALISATION SCHEMES" in the module docstring) is
    threaded through to the two scheme-aware gates, ``designed_register`` and ``seam``.
    ``scheme="rotation"`` additionally requires ``shifts`` (the declared shift per variant),
    ``protect_5p``, and ``protect_3p`` -- the same arguments ``gate_designed_register`` requires
    for that scheme. ``scheme="insertion"`` ignores all three.

    There is no per-cell "drop" path anywhere in this module: a failing member is never removed
    from the family, only recorded. ``family_passes`` is False if any evaluated gate failed, and
    -- by default, an explicit and documented choice -- also False if any gate could not be
    evaluated (``gates_not_evaluated``), because proceeding past an unevaluated gate is exactly
    how an unaudited construct would reach a model. Pass ``block_unevaluated_gates=False`` only
    if a caller explicitly wants a partial verdict recorded while a dependency is unavailable;
    that caller is then responsible for treating the result as provisional.
    """
    _require_nonempty(variants)
    _require_registers(variants, intended_registers)
    t = {**DEFAULT_TOLERANCES, **dict(tolerances)}

    gates: dict[str, dict] = {
        "designed_register": gate_designed_register(
            variants,
            parent=parent,
            intended_registers=intended_registers,
            scheme=scheme,
            shifts=shifts,
            protect_5p=protect_5p,
            protect_3p=protect_3p,
        ),
        "stop_load": gate_stop_load(variants, parent=parent, tolerance=t["stop_load"]),
        "composition": gate_composition(variants, tolerance=t["composition"]),
        "gc_and_codon_position": gate_gc_and_codon_position(
            variants, tolerance=t["gc_codon_position"]
        ),
        "period3": gate_period3(
            variants,
            parent=parent,
            intended_registers=intended_registers,
            amplitude_tolerance=t["period3_amplitude"],
            phase_tolerance_deg=t["period3_phase_deg"],
            scheme=scheme,
            protect_5p=protect_5p,
            protect_3p=protect_3p,
        ),
        "created_motifs": gate_created_motifs(
            variants, parent=parent, tolerance=t["created_motifs"]
        ),
        "seam": gate_seam(variants, parent=parent, scheme=scheme, tolerance=t["seam_edit_distance"]),
        "similarity_to_authentic": gate_similarity_to_authentic(
            variants, parent=parent, tolerance=t["similarity_to_authentic"]
        ),
        "site_scores": gate_site_scores(variants, parent=parent, tolerance=t["site_scores"]),
    }

    gates_not_evaluated = [name for name, result in gates.items() if result.get("available") is False]
    failing_gates = [
        name
        for name, result in gates.items()
        if name not in gates_not_evaluated and result["passed"] is False
    ]

    family_passes = not failing_gates and not (block_unevaluated_gates and gates_not_evaluated)

    return {
        "gates": gates,
        "failing_gates": failing_gates,
        "gates_not_evaluated": gates_not_evaluated,
        "block_unevaluated_gates": block_unevaluated_gates,
        "family_passes": family_passes,
        "members": list(variants),  # the COMPLETE membership, always -- never filtered
    }


__all__ = [
    "DEFAULT_TOLERANCES",
    "REALISATION_SCHEMES",
    "SEAM_TOLERANCE_DEFAULTS",
    "STOP_CODONS",
    "numeric_encode",
    "period3_component",
    "seam_span",
    "realised_register",
    "gate_designed_register",
    "gate_stop_load",
    "gate_composition",
    "gate_gc_and_codon_position",
    "gate_period3",
    "gate_created_motifs",
    "gate_seam",
    "gate_similarity_to_authentic",
    "gate_site_scores",
    "evaluate_family",
]
