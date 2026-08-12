"""Model-free core for paired-lineage continued pretraining of SpliceBERT.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
----------------------------------------
The positional-proxy-modifiability study set out to train two corpora that differ only in
whether the reading frame continues across a splice junction, then ask whether the trained
checkpoints differ in the annotated-phase preference ``Psi``. Its **original premise is
falsified**: `DECISIONS.md` (2026-08-12, final) records that the learned period-3 carrier is
*necessary* for ``Psi`` but does **not** encode which register is preferred, so a corpus
manipulation cannot re-target the preference *through* the carrier. What remains open, and
what this module supports, is the weaker and still-unanswered question: **does phase-decoupled
continued pretraining change the preference at all, by any route?**

Anything built on this module is therefore a **PILOT**. One training seed per arm is a single
draw from the training-process distribution, and this repository has no repeated-training
variance anywhere in its record (see ``pbc_v2_capacity.power_projection``'s ``cross_lineage``
block, which refuses to project it). A one-seed contrast is an observation about *one lineage
pair*, never a claim about the training process.

Deliberately model-free: no ``torch``, no adapter, no checkpoint, no genome handle, no
network. Sequences are plain strings, plans are plain tuples, so every function here is
testable without the ``splicebert`` extra and without a local genome.

THE OPERATOR (``paired_indel``)
-------------------------------
The two arms differ **only in one integer's residue mod 3**. Both apply the same
length-preserving paired indel to the same source window in the same order:

  * delete ``offset`` bases deep in the UPSTREAM intron, at ``delete_at``;
  * insert ``offset`` bases inside the upstream exon at ``donor_pos - exon_offset``, as a
    tandem duplication of the ``offset`` exonic bases immediately preceding that point
    (a duplication, not random sequence, so composition is very nearly unchanged).

In closed form the result is simply::

    seq[:delete_at] + seq[delete_at + offset : x] + seq[x - offset : x] + seq[x:]

with ``x = donor_pos - exon_offset``. Three invariants follow directly and are asserted:

  1. **length is preserved exactly** -- so the assay window geometry is unchanged;
  2. **``donor_pos`` and ``next_acceptor_pos`` are preserved exactly**, because every base at
     or after ``x`` keeps its index. The donor GT and the partner AG are byte-identical and
     sit at the same coordinates in *both* arms;
  3. the donor->acceptor span is therefore identical across arms too.

Why the deletion goes in the **upstream** intron rather than the downstream one: putting it
downstream (between donor and acceptor, as the exploratory ``indel_register`` probe did) moves
the donor by ``offset`` and shrinks the donor->acceptor span by ``offset``, so the two arms
would differ by 1-2 bases of span as well as by the residue. Deleting upstream removes that
difference entirely. It costs one extra eligibility condition -- the window prefix must be
genuinely intronic, which the caller enforces with ``upstream_prefix_is_intronic``.

WHAT THE OPERATOR DOES AND DOES NOT SHIFT -- state this whenever it is used
---------------------------------------------------------------------------
Relative to the fixed donor coordinate, the content originally in ``[delete_at + offset, x)``
moves ``offset`` bases further from the donor, so **its period-3 register relative to the donor
shifts by ``-offset mod 3``**. That block is the upstream intron's tail plus the bulk of the
upstream exon. The content from ``x`` onward -- the last ``exon_offset`` exonic bases, the
downstream intron, and the whole downstream exon -- keeps its register exactly.

So the manipulation shifts the *bulk* of the upstream exon's frame as a block while leaving the
last ``exon_offset`` exonic bases, and the entire downstream exon, in their native register.
``exon_offset`` is a real limit, not an incidental parameter: a model that reads the upstream
frame only from the final ``exon_offset`` bases would see no manipulation at all. It is kept at
the exploratory probe's value so the insertion never lands in the donor's motif core, and it is
recorded in every artifact.

For the phase-decoupled arm the natural relation ``delta = 0`` (real annotation is frame
continuous across a junction, always) is replaced by ``delta = -offset mod 3``, drawn from
``{1, 2}``; for the phase-preserving arm ``offset`` is a multiple of 3 and ``delta`` stays 0.
The matched configuration is therefore **absent** from the phase-decoupled corpus.

MASKING (``mask_tokens``)
-------------------------
Reproduces the published SpliceBERT recipe (Chen et al. 2024): 15% of eligible positions
selected, of which 80% become ``[MASK]``, 10% a random base, 10% are left unchanged. The 10%
unchanged share is not decoration -- dropping it is a different objective, because the model
would then learn that an unmasked position is always correct.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

# ``indel_register``'s exploratory value: far enough from the donor that the insertion never
# lands in the 5' splice-site core (roughly [donor-3, donor+6]).
DEFAULT_EXON_OFFSET = 30

#: Crop lengths for the variable-length windows of the published recipe, ascending.
DEFAULT_LENGTH_LADDER: tuple[int, ...] = (256, 512, 768, 1024)

#: Published masked-LM recipe.
MASK_RATE = 0.15
MASK_TOKEN_SHARE = 0.80
RANDOM_TOKEN_SHARE = 0.10
# The remaining 0.10 is left unchanged; it is implied rather than named so the three shares
# cannot drift out of summing to one.


class GeometryError(ValueError):
    """The requested edit does not fit the window's geometry. Never silently skipped."""


def frame_offset(k: int, residue: int) -> int:
    """``3k + residue`` -- the single integer that separates the two arms.

    ``residue == 0`` is the phase-preserving (L1) arm; ``residue in (1, 2)`` is the
    phase-decoupled (L2) arm. ``k`` must be positive: ``k == 0`` with ``residue == 0`` is the
    identity edit, which is not edit-matched to the other arm at all -- it carries neither the
    deletion nor the duplication that the decoupled arm pays for, so any difference between the
    arms would be "was the sequence edited" rather than "in what residue".
    """
    if k <= 0:
        raise ValueError(
            f"k must be a positive integer, got {k!r}; k=0 makes the phase-preserving arm an "
            "identity edit rather than an edit-matched control"
        )
    if residue not in (0, 1, 2):
        raise ValueError(f"residue must be 0, 1 or 2, got {residue!r}")
    return 3 * k + residue


def assign_residues(n_windows: int, *, seed: int) -> tuple[int, ...]:
    """Per-window residues for the phase-DECOUPLED arm, drawn deterministically from {1, 2}.

    Fixing every decoupled window at residue 1 would confound "frame does not continue" with
    one specific shift direction, so the residue alternates by a seeded hash rather than a
    counter (a counter would correlate the residue with corpus order, which is also the
    training order). ``seed`` and ``n_windows`` fully determine the result.
    """
    if n_windows < 0:
        raise ValueError(f"n_windows must be non-negative, got {n_windows}")
    out = []
    for index in range(n_windows):
        digest = hashlib.sha256(f"residue:{seed}:{index}".encode()).digest()
        out.append(1 + (digest[0] & 1))
    return tuple(out)


def upstream_prefix_is_intronic(
    *, acceptor_pos: int, intron_length_up: int | None, margin: int = 10
) -> bool:
    """Is the whole window prefix ``[0, acceptor_pos)`` inside the upstream intron?

    The compensating deletion is placed in that prefix, so it must not land in the previous
    exon: deleting exonic bases would change coding content, which is a different manipulation
    from the one under test. ``intron_length_up`` is the annotated upstream intron length from
    ``CouplingExample``; ``None`` means the previous exon was not resolvable and the window is
    rejected rather than assumed safe.
    """
    if intron_length_up is None:
        return False
    return int(intron_length_up) >= int(acceptor_pos) + int(margin)


def indel_geometry(
    *,
    seq_len: int,
    acceptor_pos: int,
    donor_pos: int,
    next_acceptor_pos: int,
    offset: int,
    exon_offset: int = DEFAULT_EXON_OFFSET,
) -> dict[str, int]:
    """Resolve, and fully validate, the coordinates ``paired_indel`` will use.

    Separated from ``paired_indel`` so a corpus builder can test a window for eligibility
    without materialising an edited string, and so both share exactly one definition of what
    "fits" means. ``delete_at`` is placed at ``acceptor_pos // 2`` -- the middle of the intronic
    prefix, hence as far as the window allows from both the previous exon's donor (off-window)
    and this exon's acceptor.

    Raises ``GeometryError`` with the failing condition named. Never returns a partial plan.
    """
    if offset <= 0:
        raise GeometryError(f"offset must be positive, got {offset}")
    if exon_offset < 0:
        raise GeometryError(f"exon_offset must be non-negative, got {exon_offset}")
    if not 0 < acceptor_pos < donor_pos < next_acceptor_pos <= seq_len:
        raise GeometryError(
            "window coordinates must satisfy 0 < acceptor_pos < donor_pos < "
            f"next_acceptor_pos <= seq_len, got acceptor_pos={acceptor_pos}, "
            f"donor_pos={donor_pos}, next_acceptor_pos={next_acceptor_pos}, seq_len={seq_len}"
        )

    insert_at = donor_pos - exon_offset
    delete_at = acceptor_pos // 2

    # The duplicated block must be exonic: it is seq[insert_at - offset : insert_at].
    if insert_at - offset < acceptor_pos:
        raise GeometryError(
            f"the duplicated block [{insert_at - offset}, {insert_at}) would reach outside the "
            f"upstream exon, which starts at {acceptor_pos}; the exon is too short for "
            f"exon_offset={exon_offset} plus offset={offset}"
        )
    # The deleted block must sit strictly inside the intronic prefix, clear of both ends.
    if delete_at < offset:
        raise GeometryError(
            f"delete_at={delete_at} leaves less than offset={offset} bases of window before it"
        )
    if delete_at + offset > acceptor_pos - offset:
        raise GeometryError(
            f"the deleted block [{delete_at}, {delete_at + offset}) is not clear of the "
            f"acceptor at {acceptor_pos}"
        )
    if delete_at + offset > insert_at:
        raise GeometryError("the deleted block must lie strictly before the insertion point")
    return {
        "insert_at": int(insert_at),
        "delete_at": int(delete_at),
        "offset": int(offset),
        "exon_offset": int(exon_offset),
    }


def paired_indel(
    seq: str,
    *,
    acceptor_pos: int,
    donor_pos: int,
    next_acceptor_pos: int,
    offset: int,
    exon_offset: int = DEFAULT_EXON_OFFSET,
) -> str:
    """Length-preserving paired indel; see the module docstring for the full derivation.

    Returns the edited window. ``donor_pos``, ``next_acceptor_pos`` and ``len(seq)`` are
    preserved exactly, which this function asserts on its own output rather than trusting the
    algebra -- a silently displaced donor is precisely the defect that discarded 48 of 54 cells
    in the exploratory probe this operator descends from.
    """
    plan = indel_geometry(
        seq_len=len(seq),
        acceptor_pos=acceptor_pos,
        donor_pos=donor_pos,
        next_acceptor_pos=next_acceptor_pos,
        offset=offset,
        exon_offset=exon_offset,
    )
    x, z, n = plan["insert_at"], plan["delete_at"], plan["offset"]
    edited = seq[:z] + seq[z + n : x] + seq[x - n : x] + seq[x:]

    if len(edited) != len(seq):
        raise GeometryError(
            f"paired indel changed the window length: {len(seq)} -> {len(edited)}"
        )
    if edited[donor_pos:] != seq[donor_pos:]:
        raise GeometryError(
            "paired indel altered the sequence at or after the donor; the donor and the "
            "partner acceptor must be byte-identical and co-located in both arms"
        )
    return edited


def register_shift(offset: int) -> int:
    """The donor-relative period-3 register shift the operator imposes: ``-offset mod 3``.

    0 for the phase-preserving arm, 1 or 2 for the phase-decoupled arm. This is the *only*
    quantity in which the two corpora differ.
    """
    return (-int(offset)) % 3


# ------------------------------------------------------------------------------------------
# Variable-length windows
# ------------------------------------------------------------------------------------------
def crop_interval(
    *,
    seq_len: int,
    keep_lo: int,
    keep_hi: int,
    draw: float,
    ladder: Sequence[int] = DEFAULT_LENGTH_LADDER,
) -> tuple[int, int]:
    """A crop of a ladder length that still contains ``[keep_lo, keep_hi)``.

    The published recipe trains on variable-length windows, so a fixed 1024-nt crop would not
    reproduce it. But a crop is only admissible if the *manipulated* region survives it: a crop
    that cut away the upstream exon or the partner acceptor would train on a window carrying
    none of the intervention, and the arms would silently converge. ``[keep_lo, keep_hi)`` is
    that protected interval, and the shortest ladder length that can hold it wins -- ``draw``
    (a value in [0, 1)) then chooses among the ladder lengths at or above it, and chooses the
    crop's offset within the sequence.

    Returns ``(start, end)`` with ``end - start`` equal to a ladder length (or to ``seq_len``
    if every ladder length exceeds it).
    """
    if not 0 <= keep_lo < keep_hi <= seq_len:
        raise GeometryError(
            f"protected interval [{keep_lo}, {keep_hi}) does not fit in a window of "
            f"{seq_len} bases"
        )
    if not 0.0 <= draw < 1.0:
        raise ValueError(f"draw must lie in [0, 1), got {draw}")
    need = keep_hi - keep_lo
    usable = [int(length) for length in sorted(ladder) if need <= length <= seq_len]
    if not usable:
        # Nothing on the ladder can hold the protected interval; fall back to the whole window
        # rather than cropping the intervention away.
        return (0, seq_len)
    # Pick a length from the admissible tail of the ladder, then an offset for that length.
    length = usable[int(draw * len(usable)) % len(usable)]
    lo_bound = max(0, keep_hi - length)
    hi_bound = min(keep_lo, seq_len - length)
    if hi_bound < lo_bound:                       # pragma: no cover - excluded by construction
        raise GeometryError("no admissible crop offset; ladder selection is inconsistent")
    span = hi_bound - lo_bound
    # A second, decorrelated draw for the offset: reusing `draw` would tie the crop position to
    # the crop length, so short crops would always sit at the same end of the window.
    offset_draw = (draw * len(usable)) % 1.0
    start = lo_bound + int(offset_draw * (span + 1))
    start = min(start, hi_bound)
    return (start, start + length)


def uniform_draw(key: str, *, seed: int) -> float:
    """A deterministic value in [0, 1) from a string key -- reproducible across processes.

    ``random.Random`` is seeded from a hash whose iteration order is not pinned across Python
    builds for some inputs; hashing explicitly makes the corpus plan reproducible from the seed
    and the key alone, which is what the manifest promises.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def batch_plan(
    lengths: Sequence[int], *, batch_size: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Group window indices into equal-length batches, then shuffle the batch order.

    Variable-length training needs equal lengths *within* a batch (padding to the longest
    member would reintroduce a fixed length and waste compute on pad tokens). Bucketing by
    length and shuffling the buckets' batches together keeps the length distribution mixed
    across training while every batch stays rectangular.

    Both arms call this with identical ``lengths`` -- the crop plan depends only on the source
    window's geometry, which the operator preserves -- so the two arms see the same windows in
    the same order in the same batches.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    buckets: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        buckets.setdefault(int(length), []).append(index)
    batches: list[tuple[int, ...]] = []
    for length in sorted(buckets):
        members = buckets[length]
        for start in range(0, len(members), batch_size):
            batches.append(tuple(members[start : start + batch_size]))
    order = sorted(range(len(batches)), key=lambda i: uniform_draw(f"batch:{i}", seed=seed))
    return tuple(batches[i] for i in order)


# ------------------------------------------------------------------------------------------
# Masked-language-model objective
# ------------------------------------------------------------------------------------------
def mask_tokens(
    token_ids: Sequence[int],
    *,
    maskable: Sequence[bool],
    mask_id: int,
    base_ids: Sequence[int],
    draws: Sequence[float],
    rate: float = MASK_RATE,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The published 15% / 80-10-10 masked-LM corruption, as a pure function.

    ``draws`` supplies two independent uniform values per position -- ``draws[2i]`` decides
    whether position ``i`` is selected, ``draws[2i + 1]`` decides which of the three
    corruptions applies and, for the random-base branch, which base. Passing the randomness in
    rather than drawing it here is what lets the two arms be proven identical: the caller
    derives ``draws`` from ``(seed, window_index, epoch)``, never from iteration order.

    Returns ``(input_ids, labels)`` with ``labels[i] == -100`` at every position that is not
    selected, which is the ignore index the loss expects. Note that the 10% "unchanged" branch
    still carries a label: the position is scored even though its input was not corrupted.
    """
    if len(maskable) != len(token_ids):
        raise ValueError("maskable must be the same length as token_ids")
    if len(draws) < 2 * len(token_ids):
        raise ValueError(
            f"need two draws per position: {2 * len(token_ids)} required, {len(draws)} given"
        )
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must lie in [0, 1], got {rate}")
    if not base_ids:
        raise ValueError("base_ids must not be empty")

    inputs = list(int(t) for t in token_ids)
    labels = [-100] * len(inputs)
    for i, token in enumerate(token_ids):
        if not maskable[i] or draws[2 * i] >= rate:
            continue
        labels[i] = int(token)
        choice = draws[2 * i + 1]
        if choice < MASK_TOKEN_SHARE:
            inputs[i] = int(mask_id)
        elif choice < MASK_TOKEN_SHARE + RANDOM_TOKEN_SHARE:
            # Rescale the sub-interval to [0, 1) so every base stays equally likely.
            local = (choice - MASK_TOKEN_SHARE) / RANDOM_TOKEN_SHARE
            inputs[i] = int(base_ids[min(int(local * len(base_ids)), len(base_ids) - 1)])
        # else: left unchanged, but still scored.
    return tuple(inputs), tuple(labels)


def position_draws(*, seed: int, window_index: int, epoch: int, n: int) -> tuple[float, ...]:
    """``2 * n`` deterministic uniforms for one window in one epoch.

    Keyed by ``(seed, window_index, epoch)`` and nothing else, so the masking a window receives
    is independent of which arm is training, of batch composition, and of iteration order.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    out: list[float] = []
    counter = 0
    while len(out) < 2 * n:
        digest = hashlib.sha256(
            f"mask:{seed}:{window_index}:{epoch}:{counter}".encode()
        ).digest()
        for start in range(0, 32, 8):
            out.append(int.from_bytes(digest[start : start + 8], "big") / float(1 << 64))
        counter += 1
    return tuple(out[: 2 * n])


# ------------------------------------------------------------------------------------------
# The evaluation cells
# ------------------------------------------------------------------------------------------
def build_chimeric_cells(
    donors: dict[int, Sequence[dict]], acceptors: dict[int, Sequence[dict]]
) -> tuple[dict, ...]:
    """Cross annotated-phase donor parents with annotated-phase acceptor parents.

    Each parent is ``{"seq", "d", "a", "gene"}`` -- the native window, the donor coordinate and
    the acceptor coordinate as the PBC census records them. Donor parent at annotated phase
    ``p`` is joined to acceptor parent at annotated phase ``q`` at the MIDPOINT of their span,
    giving ``delta = (q - p) mod 3``. Parents are paired by rank within their phase pool, so
    the same donor parent appears once per acceptor phase and the design stays balanced.

    Rejected without substitution, never repaired: a join that changes the window length, one
    whose splice point does not sit strictly between the donor and the acceptor, and one whose
    donor GT or acceptor AG did not survive the join. Repairing any of those would let the cell
    set depend on the sequence content it is supposed to be measuring.

    Cells from a donor and an acceptor of the SAME gene are dropped -- a self-join is not a
    chimera and its coupling is not comparable with a cross-gene one.

    Returns cells carrying the corrupted (``AG -> AC``) counterpart alongside the clean one, so
    the readout never has to re-derive the corruption coordinate.
    """
    cells: list[dict] = []
    phases = (0, 1, 2)
    if not all(p in donors and p in acceptors for p in phases):
        raise ValueError("donors and acceptors must both supply pools for phases 0, 1 and 2")
    depth = min(min(len(donors[p]) for p in phases), min(len(acceptors[p]) for p in phases))
    for rank in range(depth):
        for donor_phase in phases:
            for acceptor_phase in phases:
                parent_d = donors[donor_phase][rank]
                parent_a = acceptors[acceptor_phase][rank]
                if parent_d["gene"] == parent_a["gene"]:
                    continue
                donor_pos, acceptor_pos = int(parent_d["d"]), int(parent_a["a"])
                splice = (donor_pos + acceptor_pos) // 2
                sequence = parent_d["seq"][:splice] + parent_a["seq"][splice:]
                if len(sequence) != len(parent_d["seq"]):
                    continue
                if not donor_pos + 2 <= splice <= acceptor_pos - 2:
                    continue
                if sequence[donor_pos : donor_pos + 2] != "GT":
                    continue
                if sequence[acceptor_pos - 2 : acceptor_pos] != "AG":
                    continue
                cells.append(
                    {
                        "sequence": sequence,
                        "corrupt": sequence[: acceptor_pos - 2] + "AC" + sequence[acceptor_pos:],
                        "donor_pos": donor_pos,
                        "acceptor_pos": acceptor_pos,
                        "splice_at": splice,
                        "delta": (acceptor_phase - donor_phase) % 3,
                        "donor_phase": donor_phase,
                        "acceptor_phase": acceptor_phase,
                        "donor_gene": parent_d["gene"],
                        "acceptor_gene": parent_a["gene"],
                        "cluster": f"{parent_d['gene']}|{parent_a['gene']}",
                    }
                )
    return tuple(cells)


# ------------------------------------------------------------------------------------------
# The endpoint
# ------------------------------------------------------------------------------------------
def psi(coupling: Sequence[float], delta: Sequence[int]) -> float:
    """``mean(coupling | delta == 0) - mean(coupling | delta != 0)``.

    This is the two-group form the base-checkpoint decomposition used, NOT the equal-weight
    three-class form ``mean0 - (mean1 + mean2)/2``. The two agree only when the two mismatched
    classes are equally populated, and the reference value this pilot is compared against
    (``+0.256``) is the two-group form -- mixing them would silently move the comparison.
    """
    import numpy as np

    values = np.asarray(coupling, dtype=np.float64)
    labels = np.asarray(delta, dtype=np.int64)
    if values.shape != labels.shape:
        raise ValueError(f"coupling and delta must align: {values.shape} vs {labels.shape}")
    matched = labels == 0
    if not matched.any() or matched.all():
        raise ValueError("psi needs at least one matched and one mismatched cell")
    return float(values[matched].mean() - values[~matched].mean())


def cluster_bootstrap(
    arms: dict[str, Sequence[float]],
    delta: Sequence[int],
    clusters: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Gene-clustered percentile intervals for each arm's ``Psi`` and for every arm difference.

    All arms are resampled **on the same cluster draw** in each replicate, which is what makes
    the difference interval a paired one. Computing each arm's interval separately and then
    differencing the endpoints would ignore the pairing and overstate the difference's width.

    ``clusters`` is the resampling unit -- for chimeric cells that is the donor-gene/acceptor-
    gene pair, since a cell is not independent of another cell sharing either parent.
    """
    import numpy as np

    labels = np.asarray(delta, dtype=np.int64)
    groups = np.asarray(clusters, dtype=object)
    values = {name: np.asarray(v, dtype=np.float64) for name, v in arms.items()}
    for name, v in values.items():
        if v.shape != labels.shape:
            raise ValueError(f"arm {name!r} does not align with delta: {v.shape} vs {labels.shape}")
    if groups.shape != labels.shape:
        raise ValueError("clusters must align with delta")

    unique = np.unique(groups)
    index_of = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed)
    replicates: dict[str, list[float]] = {name: [] for name in values}
    names = sorted(values)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    for a, b in pairs:
        replicates[f"{a}_minus_{b}"] = []

    for _ in range(int(n_boot)):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[g] for g in picked])
        sub_labels = labels[idx]
        if not (sub_labels == 0).any() or (sub_labels == 0).all():
            continue                      # degenerate draw: no contrast to compute, skip it
        per_arm = {name: psi(v[idx], sub_labels) for name, v in values.items()}
        for name, value in per_arm.items():
            replicates[name].append(value)
        for a, b in pairs:
            replicates[f"{a}_minus_{b}"].append(per_arm[a] - per_arm[b])

    point = {name: psi(v, labels) for name, v in values.items()}
    for a, b in pairs:
        point[f"{a}_minus_{b}"] = point[a] - point[b]

    out: dict[str, dict[str, float]] = {}
    for name, draws in replicates.items():
        arr = np.asarray(draws, dtype=np.float64)
        out[name] = {
            "estimate": float(point[name]),
            "ci_low": float(np.quantile(arr, 0.025)) if arr.size else float("nan"),
            "ci_high": float(np.quantile(arr, 0.975)) if arr.size else float("nan"),
            "n_boot_used": int(arr.size),
        }
    out["_meta"] = {
        "n_cells": float(labels.size),
        "n_clusters": float(len(unique)),
        "n_boot_requested": float(n_boot),
        "seed": float(seed),
    }
    return out


# ------------------------------------------------------------------------------------------
# Corpus identity
# ------------------------------------------------------------------------------------------
def corpus_digest(sequences: Iterable[str]) -> str:
    """sha256 over the corpus in order, with an explicit separator.

    Order is part of the corpus's identity here -- the two arms are required to present the
    same source windows in the same order -- so this is deliberately not order-insensitive.
    The separator stops two different corpora that concatenate to the same string from
    colliding.
    """
    hasher = hashlib.sha256()
    for sequence in sequences:
        hasher.update(sequence.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def period3_band_power(table, bins: Sequence[int]) -> float:
    """Total power in ``bins`` of the mean-centred position table's row-wise rFFT.

    Reuses ``positional_spectrum.power_spectrum`` rather than recomputing an FFT, so a change
    to the study's spectral convention cannot leave this function reporting a stale quantity.
    ``bins`` for the 1026-row SpliceBERT table is ``p3_phase_surgery.P3_BAND_1026``.
    """
    import numpy as np

    from gfmcd.splicing.positional_spectrum import power_spectrum

    spectrum = power_spectrum(np.asarray(table))
    selected = np.asarray(sorted({int(b) for b in bins}), dtype=int)
    if selected.size == 0:
        raise ValueError("bins must not be empty")
    if selected.min() < 0 or selected.max() >= spectrum.shape[0]:
        raise ValueError(
            f"bins {selected.min()}..{selected.max()} fall outside the spectrum's "
            f"{spectrum.shape[0]} rFFT bins"
        )
    return float(spectrum[selected].sum())
