"""ENGINEERING PILOT: is the trained annotated-phase preference carried by the period-3 carrier?

WHAT THIS IS, STATED BEFORE ANY NUMBER IS READ
-----------------------------------------------
An **engineering pilot**, deliberately narrower than the confirmatory study it would precede,
and it is not the earlier ``run_lineage_pilot.py`` re-run. It differs from that pilot in three
ways that each exist to close a hole the pilot left open:

1. **A third arm, L0** -- native continued pretraining on the *unedited* source windows, same
   windows, same crop plan, same masking draws, same schedule, same step count. Without L0 a rise
   in ``Psi`` under L1 cannot be separated from GENERIC CONTINUATION: any further pretraining on
   any genomic corpus might raise the preference. L0 is what makes "phase-preserving training
   raised it" a different statement from "training raised it".
2. **``k = 10`` rather than ``k = 1``.** The arms differ by ``3k`` against ``3k + 1`` or
   ``3k + 2``, so they differ in edit SIZE as well as in residue mod 3. At ``k = 1`` that
   mismatch is 1-2 bases against 3 -- 33-67% of the edit. At ``k = 10`` it is 1-2 bases against
   30 -- 3-7%. This REDUCES the size confound; it does NOT REMOVE it, and no arithmetic here can,
   because the residue cannot change without changing the integer.
3. **A phase-sensitive carrier readout, and a difference-of-differences causal endpoint.** The
   pilot reported period-3 band POWER, which sums squared moduli and is blind both to a phase
   rotation and to redistribution across bins or hidden dimensions. This runner reports the
   complex band coefficients, and it asks the causal question with the contrast that can answer
   it (below), not with a per-arm dose ladder alone.

WHAT IS MEASURED, AND THE ONE ENDPOINT THAT ANSWERS THE QUESTION
-----------------------------------------------------------------
Per-arm dose ladders answer "does this checkpoint's ``Psi`` need the carrier?". They do not
answer "does the TRAINING EFFECT need the carrier?". That question is a difference of
differences::

    DoD(band, dose) = [Psi_X - Psi_Y]_intact  -  [Psi_X - Psi_Y]_{band attenuated to dose}

computed for the period-3 band and, at the same doses, for power-matched CONTROL bands. A large
``DoD`` for period-3 and a small one for the controls says the training contrast lives on the
carrier. Both the ladders and the ``DoD`` are reported; the ``DoD`` is the endpoint.

R12/R13: NEVER READ A MECHANISM FROM A DESTROYED BEHAVIOUR
-----------------------------------------------------------
Every dose reports its own health block beside its ``Psi``: the full coupling DISTRIBUTION (not
its mean alone), masked MLM top-1, and two splice-motif control behaviours -- masked donor-G
top-1 and masked acceptor-G top-1. This matters more here than in the pilot because continued
pretraining moved the overall coupling mean substantially (the pilot's L1 reached -0.134 from a
base of +0.049), so a signed "coupling retention" ratio against a near-zero or negative baseline
is NOT a retention. The artifact reports the raw baseline beside every ratio and flags the ones
that are not interpretable.

WHAT THIS RUN CANNOT SUPPORT, WHATEVER IT SHOWS
------------------------------------------------
* **Two training seeds is not a training-process claim.** The replicate for a corpus intervention
  is the TRAINING SEED. Every bootstrap interval here covers EVALUATION-PARENT resampling only.
  Across seeds this reports a RANGE over two draws, never an interval, and two draws bound
  nothing.
* **A non-detection is not an equivalence.** An arm difference whose interval includes zero but
  also includes effects above the 0.05 SESOI is UNRESOLVED, and is reported with that word.
* **L1 vs L0 is not edit-matched.** L0 carries no indel at all, so L1-minus-L0 conflates
  "phase-preserving edit" with "any edit". L1 vs L2 is the edit-matched contrast; L1 vs L0 is the
  generic-continuation control. Neither substitutes for the other.
* **``Omega3`` is not run here.** It tests whether an IMPOSED carrier relocation moves behaviour,
  which is a different question from whether TRAINING moved the learned Fourier phase. See the
  ``omega3`` block of the artifact.

    uv run python scripts/splicebert_positional_proxy_modifiability/run_lineage_carrier_dependence.py \
        --out results/splicebert_positional_proxy_modifiability/lineage_carrier_dependence_v1.json \
        --weights-dir /path/outside/this/repository
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_lineage_pilot as pilot  # noqa: E402  -- corpus and training primitives, reused verbatim

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.carrier_phase import (  # noqa: E402
    P3_BINS_1026,
    PROTECTED_PERIODS,
    band_coefficients,
    band_phase_profile,
    band_rotation,
    bin_energy,
    dimension_energy,
    dominant_dimensions,
    protected_period_bins,
    redistribution,
    retention_ratio,
)
from gfmcd.splicing.lineage_pretraining import (  # noqa: E402
    DEFAULT_EXON_OFFSET,
    DEFAULT_LENGTH_LADDER,
    GeometryError,
    assign_residues,
    build_chimeric_cells,
    cluster_draw_indices,
    corpus_digest,
    crop_interval,
    edit_budget,
    frame_offset,
    kmer_profile,
    paired_indel,
    percentile_interval,
    period3_band_power,
    profile_distance,
    psi,
    psi_replicates,
    register_shift,
    sesoi_verdict,
    uniform_draw,
    upstream_prefix_is_intronic,
)
from gfmcd.splicing.parent_exclusion import (  # noqa: E402
    REASON_DIRECT_PARENT,
    REASON_HOMOLOGY_HIT,
    kmer_index,
    near_identity_hits,
    normalize_gene_id,
)
from gfmcd.splicing.positional_spectrum import (  # noqa: E402
    power_spectrum,
    scale_band,
    select_power_matched_controls,
)

EXPERIMENT_ID = "splicebert-positional-proxy-modifiability:lineage-carrier-dependence-v1"

BASE_CHECKPOINT_SHA256 = pilot.BASE_CHECKPOINT_SHA256
DEFAULT_CHECKPOINT = pilot.DEFAULT_CHECKPOINT
DEFAULT_CENSUS = pilot.DEFAULT_CENSUS
SEALED_CHROMOSOMES = pilot.SEALED_CHROMOSOMES

#: The base-checkpoint reading this run must reproduce before it is allowed to train anything.
#: Source: studies/splicebert_positional_proxy_modifiability/DECISIONS.md, 2026-08-12.
REFERENCE_PSI = pilot.REFERENCE_PSI
REFERENCE_COUPLING = pilot.REFERENCE_COUPLING
REFERENCE_CELLS = pilot.REFERENCE_CELLS
PSI_TOLERANCE = pilot.PSI_TOLERANCE
CELL_COUNT_TOLERANCE = pilot.CELL_COUNT_TOLERANCE

P3_BINS = P3_BINS_1026

#: The k=1 pilot's readings. Recorded for CONTEXT ONLY. This run changes k, adds an arm, and
#: therefore draws a different corpus, so these are NOT reproduction gates and must never be
#: reported as though this run had reproduced or failed to reproduce them.
PILOT_REFERENCE = {
    "artifact": "results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.json",
    "k": 1,
    "arms": {
        "L1": {"psi": 0.6420838275379966, "ci": [0.5212034474650911, 0.7714749932017428]},
        "L2": {"psi": 0.29748270489713724, "ci": [0.21371508257221652, 0.3947694231572315]},
    },
    "corpus_sha256": {
        "L1": "99a2f78a6f0af492a1ef1ae1fdffb32d09d0c05753162d9e80c730744f2fd17f",
        "L2": "fa57a1b72551f18843539cb7c7ce74ce8ef8a5520cbf3a82ff534f20a46f7e6d",
    },
    "comparability": (
        "NOT COMPARABLE AS A GATE. This run uses k=10 (the pilot used k=1) and adds an unedited "
        "L0 arm, so the eligible-window set, the corpus bytes and the edit sizes all differ. "
        "The pilot values are printed for orientation only."
    ),
}

#: The smallest effect the programme calls material, frozen upstream in PREREGISTRATION.md.
SESOI = 0.05

ARMS: tuple[str, ...] = ("L0", "L1", "L2")
ARM_LABEL = {
    "L0": "native (no edit) -- generic-continuation control",
    "L1": "phase-preserving (offset 3k)",
    "L2": "phase-decoupled (offset 3k+1 / 3k+2)",
}


# ------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def guard_weights_dir(path: Path) -> Path:
    """Refuse a weights directory inside the repository. Binaries are never committed here."""
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    raise SystemExit(
        f"FATAL: --weights-dir {resolved} lies inside the repository at {REPO_ROOT}. "
        "Checkpoints and per-cell arrays are binaries and are never written into this tree."
    )


def json_default(value):
    """Serialise NumPy scalars and arrays without silently turning integers into floats."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value)!r} is not JSON serialisable")


def summarise(values: np.ndarray) -> dict:
    """Distribution summary. A mean alone hides a bimodal or sign-flipped coupling population."""
    arr = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "abs_mean": float(np.abs(arr).mean()),
        "rms": float(np.sqrt(np.mean(np.square(arr)))),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "fraction_positive": float((arr > 0).mean()),
        "n": int(arr.size),
    }


def spread(values: list[float]) -> dict:
    """Across-seed summary. A RANGE, never an interval -- two draws bound nothing."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n_seeds": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n_seeds": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range_note": (
            "RANGE across training seeds, not a confidence interval. With this many seeds it "
            "bounds nothing; it is reported so the spread is visible rather than hidden."
        ),
    }


# ------------------------------------------------------------------------------------------
# Corpus: three arms from one staged window set
# ------------------------------------------------------------------------------------------
def build_three_arm_corpus(
    *,
    gtf: str,
    fasta: str,
    chromosomes: tuple[str, ...],
    excluded_genes: set[str],
    parent_sequences: dict[str, str],
    kmer_k: int,
    min_shared_kmers: int,
    max_windows: int,
    seed: int,
    k: int,
    exon_offset: int,
    composition_k: tuple[int, ...],
) -> dict:
    """Stage one window set, then emit L0 (unedited), L1 (3k) and L2 (3k+residue) from it.

    Follows ``run_lineage_pilot.build_training_corpus`` exactly -- same eligibility, same
    two-stage leakage screen, same deterministic ordering and cap -- and adds the L0 arm plus the
    edit and composition accounting the three arms must be checked against. The geometry gate is
    still applied for every residue, so all three arms are built from an identical window set:
    dropping a window for one arm only would make the arms differ in their SOURCE as well as in
    their edit, which is the confound the whole design exists to avoid.
    """
    from gfmcd.splicing.coupling import CouplingConfig, build_coupling_dataset
    from gfmcd.splicing.dataset import PyfaidxGenome

    for sealed in SEALED_CHROMOSOMES:
        if sealed in chromosomes:
            raise SystemExit(f"FATAL: {sealed} is globally sealed and may not supply training data")

    raw = build_coupling_dataset(
        gtf, PyfaidxGenome(fasta), CouplingConfig(), allowed_chromosomes=chromosomes
    )

    tally = {
        "windows_from_gtf": len(raw),
        "rejected_not_eligible": 0,
        "rejected_sealed_chromosome": 0,
        "rejected_gene_excluded": 0,
        "rejected_upstream_prefix_not_intronic": 0,
        "rejected_indel_geometry": 0,
        "rejected_homology_hit": 0,
    }

    staged: list = []
    for example in raw:
        chrom = example.chrom if example.chrom.startswith("chr") else f"chr{example.chrom}"
        if chrom in SEALED_CHROMOSOMES:
            tally["rejected_sealed_chromosome"] += 1
            continue
        gap = (
            (example.next_acceptor_pos - example.donor_pos)
            if example.next_acceptor_pos is not None
            else None
        )
        if (
            example.next_acceptor_pos is None
            or example.sequence[example.donor_pos : example.donor_pos + 2] != "GT"
            or gap is None
            or not 60 <= gap <= 150
        ):
            tally["rejected_not_eligible"] += 1
            continue
        if normalize_gene_id(example.gene_id) in excluded_genes:
            tally["rejected_gene_excluded"] += 1
            continue
        if not upstream_prefix_is_intronic(
            acceptor_pos=example.acceptor_pos, intron_length_up=example.intron_length_up
        ):
            tally["rejected_upstream_prefix_not_intronic"] += 1
            continue
        try:
            for residue in (0, 1, 2):
                paired_indel(
                    example.sequence,
                    acceptor_pos=example.acceptor_pos,
                    donor_pos=example.donor_pos,
                    next_acceptor_pos=example.next_acceptor_pos,
                    offset=frame_offset(k, residue),
                    exon_offset=exon_offset,
                )
        except GeometryError:
            tally["rejected_indel_geometry"] += 1
            continue
        staged.append(example)

    staged.sort(key=lambda ex: uniform_draw(ex.example_id, seed=seed))
    staged = staged[: int(max_windows)]

    index = kmer_index({ex.example_id: ex.sequence for ex in staged}, k=kmer_k)
    homology_hits: dict[str, int] = {}
    for parent_id, parent_seq in parent_sequences.items():
        for window_id, shared in near_identity_hits(
            parent_seq, index, k=kmer_k, min_shared_kmers=min_shared_kmers
        ).items():
            homology_hits[window_id] = max(homology_hits.get(window_id, 0), shared)
    kept = [ex for ex in staged if ex.example_id not in homology_hits]
    tally["rejected_homology_hit"] = len(staged) - len(kept)

    residues = assign_residues(len(kept), seed=seed)
    arms: dict[str, list[str]] = {arm: [] for arm in ARMS}
    offsets_by_arm: dict[str, list[int]] = {arm: [] for arm in ARMS}
    plans: list[dict] = []
    for position, example in enumerate(kept):
        offsets = {
            "L0": 0,
            "L1": frame_offset(k, 0),
            "L2": frame_offset(k, residues[position]),
        }
        for arm, offset in offsets.items():
            arms[arm].append(
                example.sequence
                if offset == 0
                else paired_indel(
                    example.sequence,
                    acceptor_pos=example.acceptor_pos,
                    donor_pos=example.donor_pos,
                    next_acceptor_pos=example.next_acceptor_pos,
                    offset=offset,
                    exon_offset=exon_offset,
                )
            )
            offsets_by_arm[arm].append(offset)
        keep_lo = max(0, example.acceptor_pos - 20)
        keep_hi = min(len(example.sequence), example.next_acceptor_pos + 20)
        start, end = crop_interval(
            seq_len=len(example.sequence),
            keep_lo=keep_lo,
            keep_hi=keep_hi,
            draw=uniform_draw(f"crop:{example.example_id}", seed=seed),
        )
        plans.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "crop": (start, end),
                "offsets": offsets,
                "register_shift": {a: register_shift(o) if o else 0 for a, o in offsets.items()},
            }
        )

    cropped = {
        arm: [seq[p["crop"][0] : p["crop"][1]] for seq, p in zip(arms[arm], plans)]
        for arm in ARMS
    }
    profiles = {
        width: {arm: kmer_profile(cropped[arm], width) for arm in ARMS}
        for width in composition_k
    }
    composition = {
        str(width): {
            f"{arm}_vs_L0": profile_distance(profiles[width]["L0"], profiles[width][arm])
            for arm in ARMS
        }
        | {"L1_vs_L2": profile_distance(profiles[width]["L1"], profiles[width]["L2"])}
        for width in composition_k
    }

    return {
        "arms": arms,
        "cropped": cropped,
        "plans": plans,
        "tally": tally,
        "residues": residues,
        "n_windows": len(kept),
        "genes": sorted({normalize_gene_id(p["gene_id"]) for p in plans}),
        "edit_budget": {arm: edit_budget(offsets_by_arm[arm]) for arm in ARMS},
        "composition_delta": composition,
        "composition_k": list(composition_k),
        "cropped_bases": {arm: int(sum(len(s) for s in cropped[arm])) for arm in ARMS},
    }


# ------------------------------------------------------------------------------------------
# Model-side measurement
# ------------------------------------------------------------------------------------------
def install_table(model, table: np.ndarray) -> None:
    """Copy a position table into the live model. Shape drift is fatal, never coerced."""
    import torch

    from gfmcd.splicing.pbc_register_runner import position_embedding_module

    embedding = position_embedding_module(model)
    tensor = torch.as_tensor(table, dtype=embedding.weight.dtype, device=embedding.weight.device)
    if tensor.shape != embedding.weight.shape:
        raise ValueError(
            f"table {tuple(tensor.shape)} does not match the position embedding "
            f"{tuple(embedding.weight.shape)}"
        )
    with torch.no_grad():
        embedding.weight.copy_(tensor)


def read_table(model) -> np.ndarray:
    from gfmcd.splicing.pbc_register_runner import position_embedding_module

    return (
        position_embedding_module(model).weight.detach().float().cpu().numpy().astype(np.float64)
    )


def motif_top1(adapter, task, cells, *, offset: int, expected: str, batch: int) -> float:
    """Masked top-1 accuracy at a nominated splice-motif base, ``offset`` from the acceptor.

    A biological control behaviour rather than a generic one: it asks whether the model still
    recognises the 3' AG it is looking at, which a model that has merely been shifted off its
    coupling baseline should still do. Masked, always -- an unmasked probe reads the input back
    and cannot fail (``CouplingTask.mlm_health``'s docstring records what that cost once).
    """
    import torch

    correct = 0
    total = 0
    expected_id = adapter.tokenizer.convert_tokens_to_ids(expected)
    with torch.no_grad():
        for start in range(0, len(cells), batch):
            chunk = cells[start : start + batch]
            encoded = adapter.tokenizer(
                [" ".join(cell["sequence"]) for cell in chunk], return_tensors="pt", padding=True
            ).to(adapter.device)
            probes = [cell["acceptor_pos"] + offset for cell in chunk]
            for row, probe in enumerate(probes):
                encoded["input_ids"][row, probe + task.off] = task.mask_id
            logits = adapter.model(**encoded).logits.float()
            for row, (cell, probe) in enumerate(zip(chunk, probes)):
                if cell["sequence"][probe] != expected:
                    continue                      # not the motif base: not this control's subject
                total += 1
                correct += int(int(logits[row, probe + task.off].argmax()) == expected_id)
    return float(correct / total) if total else float("nan")


def score_condition(model, tokenizer, cells, *, device: str, batch: int) -> dict:
    """One condition: per-cell coupling, the margin halves, MLM health and two motif controls."""
    import torch

    from gfmcd.adapters.splicebert import SpliceBertAdapter
    from gfmcd.splicing.circuit import CouplingTask
    from gfmcd.splicing.coupling import CouplingExample

    adapter = SpliceBertAdapter(model, tokenizer, device=device)
    task = CouplingTask(adapter, batch=batch)
    donor_positions = [cell["donor_pos"] for cell in cells]

    def margins(key: str) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(cells), batch):
                chunk = list(range(start, min(start + batch, len(cells))))
                encoded = task.encode(
                    [cells[i][key] for i in chunk], [donor_positions[i] for i in chunk]
                )
                logits = adapter.model(**encoded).logits.float()
                out.append(task.margin(logits, [donor_positions[i] for i in chunk]).cpu().numpy())
        return np.concatenate(out)

    clean = margins("sequence")
    corrupt = margins("corrupt")

    examples = [
        CouplingExample(
            example_id=f"chimera-{i}",
            sequence=cell["sequence"],
            acceptor_pos=cell["acceptor_pos"],
            donor_pos=cell["donor_pos"],
            chrom="chimera",
            strand="+",
            gene_id=cell["donor_gene"],
            split="eval",
            exon_length=0,
            intron_length_down=None,
            n_isoforms_donor=0,
            n_isoforms_acceptor=0,
            gene_type="chimera",
            next_acceptor_pos=cell["acceptor_pos"],
        )
        for i, cell in enumerate(cells)
    ]
    return {
        "coupling": clean - corrupt,
        "clean_margin": clean,
        "corrupt_margin": corrupt,
        # `CouplingTask.mlm_health` is the MASKED probe. `CouplingScorer.mlm_health` does not mask
        # and once returned 0.998 for a model with 94 of 96 heads ablated -- it cannot fail.
        "mlm_health_masked_top1": float(task.mlm_health(examples)),
        # Free from the clean margin: margin = logit(G) - max(logit A, C, T), so margin > 0 is
        # exactly "G is the top-1 base at the masked donor".
        "donor_g_top1_rate": float((clean > 0).mean()),
        "acceptor_g_top1_rate": motif_top1(
            adapter, task, cells, offset=-1, expected="G", batch=batch
        ),
    }


# ------------------------------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output artifact path (no default, by design)")
    parser.add_argument(
        "--weights-dir",
        required=True,
        help="directory for trained weights and per-cell arrays; MUST lie outside the repository",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--census", default=DEFAULT_CENSUS)
    parser.add_argument("--gtf", default=os.environ.get("GFMCD_GTF"))
    parser.add_argument("--fasta", default=os.environ.get("GFMCD_FASTA"))
    parser.add_argument("--chroms", default="chr1,chr2,chr3,chr4,chr10")
    parser.add_argument("--windows", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=10, help="the k in offset = 3k + residue")
    parser.add_argument("--exon-offset", type=int, default=DEFAULT_EXON_OFFSET)
    parser.add_argument("--per-phase", type=int, default=60)
    parser.add_argument("--kmer-k", type=int, default=31)
    parser.add_argument("--min-shared-kmers", type=int, default=1)
    parser.add_argument(
        "--seeds",
        default="20260812,20260813",
        help="comma-separated training seeds; one lineage triple (L0/L1/L2) per seed",
    )
    parser.add_argument("--composition-k", default="1,3")
    parser.add_argument("--eval-batch", type=int, default=9)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--doses", default="1.0,0.75,0.5,0.25,0.0")
    parser.add_argument("--n-controls", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--reuse-weights",
        action="store_true",
        help="load persisted arm weights instead of training, when they are already present",
    )
    parser.add_argument("--smoke", action="store_true", help="mechanics only; base gate NOT enforced")
    parser.add_argument("--overwrite-record", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.gtf or not args.fasta:
        raise SystemExit("FATAL: --gtf/--fasta (or GFMCD_GTF/GFMCD_FASTA) are required")

    out = guard_output_path(Path(args.out).expanduser().resolve(), overwrite=args.overwrite_record)
    manifest_path = guard_output_path(
        out.with_suffix(".manifest.json"), overwrite=args.overwrite_record
    )
    weights_dir = guard_weights_dir(Path(args.weights_dir))

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from gfmcd.splicing.circuit import pick_device

    device = args.device or pick_device()
    started = time.time()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    doses = [float(d) for d in str(args.doses).split(",") if d.strip()]
    composition_k = tuple(int(v) for v in str(args.composition_k).split(",") if v.strip())
    if 1.0 not in doses:
        raise SystemExit("FATAL: the dose grid must contain 1.0 -- it is the intact reference")
    steps = 12 if args.smoke else args.steps
    per_phase = 6 if args.smoke else args.per_phase
    n_boot = 200 if args.smoke else args.n_boot

    print(f"[1/8] verifying the base checkpoint  (device {device})", flush=True)
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_sha = pilot.verify_checkpoint(checkpoint_dir)
    print(f"      sha256 {checkpoint_sha} OK", flush=True)

    print("[2/8] streaming evaluation parents from the PBC census", flush=True)
    donors, acceptors = pilot.load_eval_parents(Path(args.census), per_phase)
    cells = build_chimeric_cells(donors, acceptors)
    if not cells:
        raise SystemExit("FATAL: no evaluation cells were built; geometry has drifted")
    eval_gene_chrom = {
        normalize_gene_id(parent["gene"]): parent["chrom"]
        for pool in (donors, acceptors)
        for parents in pool.values()
        for parent in parents
    }
    eval_genes = set(eval_gene_chrom)
    parent_sequences = {
        parent["gene"]: parent["seq"]
        for pool in (donors, acceptors)
        for parents in pool.values()
        for parent in parents
    }
    delta = np.asarray([cell["delta"] for cell in cells])
    donor_clusters = np.asarray(
        [normalize_gene_id(cell["donor_gene"]) for cell in cells], dtype=object
    )
    acceptor_clusters = np.asarray(
        [normalize_gene_id(cell["acceptor_gene"]) for cell in cells], dtype=object
    )
    print(
        f"      cells {len(cells)}  delta sizes "
        f"{[int((delta == d).sum()) for d in (0, 1, 2)]}  eval genes {len(eval_genes)}",
        flush=True,
    )

    # One shared draw sequence for EVERY condition and arm, so all differences are paired.
    draws = {
        "donor_gene": cluster_draw_indices(donor_clusters, delta, n_boot=n_boot, seed=seeds[0]),
        "acceptor_gene": cluster_draw_indices(
            acceptor_clusters, delta, n_boot=n_boot, seed=seeds[0]
        ),
    }

    def interval(values: np.ndarray, scheme: str = "donor_gene") -> dict:
        return percentile_interval(psi_replicates(values, draws[scheme]), psi(values, delta))

    def contrast(left: np.ndarray, right: np.ndarray, scheme: str = "donor_gene") -> dict:
        replicates = psi_replicates(left, draws[scheme]) - psi_replicates(right, draws[scheme])
        return percentile_interval(replicates, psi(left, delta) - psi(right, delta))

    print("[3/8] BASE checkpoint: the stop rule, before anything is trained", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    base_model = AutoModelForMaskedLM.from_pretrained(str(checkpoint_dir)).to(device).eval()
    base_table = read_table(base_model)
    base_scores = score_condition(base_model, tokenizer, cells, device=device, batch=args.eval_batch)
    psi_base = psi(base_scores["coupling"], delta)
    reproduced = (
        abs(psi_base - REFERENCE_PSI) <= PSI_TOLERANCE
        and abs(len(cells) - REFERENCE_CELLS) <= CELL_COUNT_TOLERANCE
    )
    print(
        f"      base Psi {psi_base:+.5f} (reference {REFERENCE_PSI:+.3f})  "
        f"coupling {base_scores['coupling'].mean():+.5f} (reference {REFERENCE_COUPLING:+.3f})  "
        f"reproduced={reproduced}",
        flush=True,
    )
    if not reproduced and not args.smoke:
        payload = {
            "status": "base_reproduction_failed",
            "artifact_class": "ENGINEERING PILOT",
            "stop_reason": (
                "the base checkpoint did not reproduce the recorded annotated-phase reading on "
                "these cells, so the evaluation instrument has drifted; NO arm was trained and "
                "no carrier-dependence quantity is reported"
            ),
            "base": {
                "psi": psi_base,
                "coupling": summarise(base_scores["coupling"]),
                "n_cells": len(cells),
                "mlm_health_masked_top1": base_scores["mlm_health_masked_top1"],
            },
            "reference": {
                "psi": REFERENCE_PSI,
                "coupling_mean": REFERENCE_COUPLING,
                "n_cells": REFERENCE_CELLS,
                "psi_tolerance": PSI_TOLERANCE,
                "cell_count_tolerance": CELL_COUNT_TOLERANCE,
            },
            "checkpoint_sha256": checkpoint_sha,
            "wall_clock_seconds": time.time() - started,
        }
        _write(out, manifest_path, payload, args, checkpoint_sha)
        print("\nSTOPPED: base Psi did not reproduce. Nothing was trained.", flush=True)
        return

    print("[4/8] freezing the intervention bands on the BASE table", flush=True)
    protected = protected_period_bins(base_table.shape[0], PROTECTED_PERIODS)
    control_bands = select_power_matched_controls(
        base_table,
        [{"name": "period3", "bins": list(P3_BINS)}],
        protected_bands=[band["bins"] for band in protected["bands"]],
        n_controls=args.n_controls,
    )["period3"]
    if not control_bands:
        raise SystemExit(
            f"FATAL: no {args.n_controls} power-matched control bands exist outside the protected "
            "periods; the dose ladder has no specificity control and must not be read"
        )
    base_power = power_spectrum(base_table)
    bands: dict[str, tuple[int, ...]] = {"period3": tuple(P3_BINS)}
    for position, band in enumerate(control_bands, start=1):
        bands[f"control{position}"] = tuple(int(b) for b in band)
    band_report = {
        name: {
            "bins": list(bins),
            "periods": [float(base_table.shape[0]) / b for b in bins],
            "base_power": float(base_power[list(bins)].sum()),
            "base_power_ratio_to_period3": float(
                base_power[list(bins)].sum() / base_power[list(P3_BINS)].sum()
            ),
            "role": "carrier" if name == "period3" else "power-matched control",
        }
        for name, bins in bands.items()
    }
    print(f"      bands {[(n, b['bins'][0], b['bins'][-1]) for n, b in band_report.items()]}",
          flush=True)

    conditions = [("intact", "none", 1.0)]
    for name in bands:
        for dose in doses:
            if dose == 1.0:
                continue           # identical to `intact`: scale_band(t, bins, 1.0) is t exactly
            conditions.append((f"{name}@{dose:g}", name, dose))

    coupling_store: dict[tuple, np.ndarray] = {}
    condition_report: dict[str, dict] = {}
    primitive_files: dict[str, dict] = {}

    def run_ladder(model, table: np.ndarray, tag: str, first_scores: dict | None) -> dict:
        """Every band x dose on one checkpoint. ``first_scores`` supplies the intact rung."""
        rows: dict[str, dict] = {}
        arrays: dict[str, np.ndarray] = {}
        for label, band_name, dose in conditions:
            if label == "intact" and first_scores is not None:
                scores = first_scores
            else:
                install_table(
                    model, table if band_name == "none" else scale_band(table, bands[band_name], dose)
                )
                scores = score_condition(
                    model, tokenizer, cells, device=device, batch=args.eval_batch
                )
            coupling = scores["coupling"]
            coupling_store[(tag, label)] = coupling
            arrays[f"coupling__{label}"] = coupling
            arrays[f"clean_margin__{label}"] = scores["clean_margin"]
            arrays[f"corrupt_margin__{label}"] = scores["corrupt_margin"]
            rows[label] = {
                "band": band_name,
                "dose": dose,
                "psi": interval(coupling),
                "psi_acceptor_clustered": interval(coupling, "acceptor_gene"),
                "coupling": summarise(coupling),
                "clean_margin": summarise(scores["clean_margin"]),
                "corrupt_margin": summarise(scores["corrupt_margin"]),
                "health": {
                    "mlm_health_masked_top1": scores["mlm_health_masked_top1"],
                    "donor_g_top1_rate": scores["donor_g_top1_rate"],
                    "acceptor_g_top1_rate": scores["acceptor_g_top1_rate"],
                },
            }
            print(
                f"        {tag:16s} {label:14s} Psi {rows[label]['psi']['estimate']:+.5f}  "
                f"coupling {coupling.mean():+.5f}  |coupling| {np.abs(coupling).mean():.5f}  "
                f"mlm {scores['mlm_health_masked_top1']:.4f}  "
                f"AG {scores['acceptor_g_top1_rate']:.4f}",
                flush=True,
            )
        install_table(model, table)                     # always leave the model as we found it
        path = weights_dir / f"{tag}_primitives.npz"
        np.savez_compressed(path, **arrays)
        primitive_files[tag] = {"path": str(path), "sha256": sha256_file(path)}

        intact = rows["intact"]
        for label, row in rows.items():
            row["psi_retention"] = retention_ratio(
                row["psi"]["estimate"], intact["psi"]["estimate"]
            )
            row["coupling_retention_signed"] = retention_ratio(
                row["coupling"]["mean"], intact["coupling"]["mean"]
            )
            row["coupling_retention_magnitude"] = retention_ratio(
                row["coupling"]["abs_mean"], intact["coupling"]["abs_mean"]
            )
            row["mlm_health_retention"] = retention_ratio(
                row["health"]["mlm_health_masked_top1"],
                intact["health"]["mlm_health_masked_top1"],
            )
            # A signed retention ratio is only a retention when its baseline is reliably positive.
            # Two standard errors is the bar, stated here rather than left for the reader to
            # guess: below it the sign of the denominator is not established and the ratio can
            # invert on noise. The magnitude retention and the health probes carry R12/R13 then.
            standard_error = intact["coupling"]["sd"] / math.sqrt(len(cells))
            row["signed_retention_interpretable"] = bool(
                intact["coupling"]["mean"] > 2.0 * standard_error
            )
            row["signed_retention_baseline"] = intact["coupling"]["mean"]
            row["psi_sesoi_verdict"] = sesoi_verdict(row["psi"], SESOI)
        return rows

    print("[5/8] base dose ladder", flush=True)
    condition_report["base"] = run_ladder(base_model, base_table, "base", base_scores)
    del base_model
    gc.collect()

    print("[6/8] lineage triples", flush=True)
    corpora: dict[int, dict] = {}
    arm_records: dict[str, dict] = {}
    tables: dict[str, np.ndarray] = {"base": base_table}
    weight_files: dict[str, dict] = {}
    for seed in seeds:
        print(f"      corpus for seed {seed} (k={args.k})", flush=True)
        corpus = build_three_arm_corpus(
            gtf=args.gtf,
            fasta=args.fasta,
            chromosomes=tuple(c.strip() for c in args.chroms.split(",") if c.strip()),
            excluded_genes=eval_genes,
            parent_sequences=parent_sequences,
            kmer_k=args.kmer_k,
            min_shared_kmers=args.min_shared_kmers,
            max_windows=40 if args.smoke else args.windows,
            seed=seed,
            k=args.k,
            exon_offset=args.exon_offset,
            composition_k=composition_k,
        )
        digests = {arm: corpus_digest(seqs) for arm, seqs in corpus["arms"].items()}
        if len({digests[arm] for arm in ARMS}) != len(ARMS):
            raise SystemExit(
                f"FATAL: two of the three corpora are byte-identical at seed {seed}; "
                "the operator did nothing"
            )
        corpora[seed] = {**corpus, "sha256": digests}
        print(f"      windows {corpus['n_windows']}  tally {corpus['tally']}", flush=True)

        for arm in ARMS:
            tag = f"seed{seed}_{arm}"
            state_path = weights_dir / f"{tag}_state_dict.pt"
            table_path = weights_dir / f"{tag}_position_table.npy"
            if args.reuse_weights and state_path.exists():
                print(f"      [{tag}] reusing persisted weights", flush=True)
                model = AutoModelForMaskedLM.from_pretrained(str(checkpoint_dir))
                model.load_state_dict(torch.load(state_path, map_location="cpu"))
                model = model.to(device).eval()
                record = {"reused_persisted_weights": True, "steps_trained": None}
            else:
                print(f"      [{tag}] continued pretraining, {steps} steps", flush=True)
                model, record = pilot.train_arm(
                    checkpoint=str(checkpoint_dir),
                    sequences=corpus["arms"][arm],
                    plans=corpus["plans"],
                    device=device,
                    steps=steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=seed,
                )
                record["reused_persisted_weights"] = False
                print(
                    f"      [{tag}] {record['steps_trained']} steps in "
                    f"{record['wall_clock_seconds']:.0f}s  loss "
                    f"{record['loss_first_10_mean']:.4f} -> {record['loss_last_10_mean']:.4f}",
                    flush=True,
                )
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, state_path)
            table = read_table(model)
            np.save(table_path, table)
            tables[tag] = table
            weight_files[tag] = {
                "state_dict": {"path": str(state_path), "sha256": sha256_file(state_path)},
                "position_table": {"path": str(table_path), "sha256": sha256_file(table_path)},
            }
            arm_records[tag] = record
            condition_report[tag] = run_ladder(model, table, tag, None)
            del model
            gc.collect()

    print("[7/8] complex period-3 carrier: amplitude, phase and redistribution", flush=True)
    base_table_path = weights_dir / "base_position_table.npy"
    np.save(base_table_path, base_table)
    primitive_files["base_position_table"] = {
        "path": str(base_table_path),
        "sha256": sha256_file(base_table_path),
    }
    reference_dimensions = dominant_dimensions(base_table, P3_BINS)
    base_bin_energy = bin_energy(base_table, P3_BINS)
    base_dimension_energy = dimension_energy(base_table, P3_BINS)
    carrier: dict[str, dict] = {}
    for tag, table in tables.items():
        coefficients = band_coefficients(table, P3_BINS)
        path = weights_dir / f"{tag}_p3_band_coefficients.npy"
        np.save(path, coefficients)
        primitive_files[f"{tag}_p3_coefficients"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        carrier[tag] = {
            "band_power": period3_band_power(table, P3_BINS),
            "per_bin": band_phase_profile(
                table, P3_BINS, reference_dimensions=reference_dimensions
            ),
            "vs_base": {
                "per_bin_rotation": band_rotation(base_table, table, P3_BINS),
                "bin_redistribution": redistribution(base_bin_energy, bin_energy(table, P3_BINS)),
                "dimension_redistribution": redistribution(
                    base_dimension_energy, dimension_energy(table, P3_BINS)
                ),
            },
        }
    for seed in seeds:
        left, right = f"seed{seed}_L1", f"seed{seed}_L2"
        if left in tables and right in tables:
            carrier[f"{left}_vs_{right}"] = {
                "per_bin_rotation": band_rotation(tables[left], tables[right], P3_BINS),
                "bin_redistribution": redistribution(
                    bin_energy(tables[left], P3_BINS), bin_energy(tables[right], P3_BINS)
                ),
                "dimension_redistribution": redistribution(
                    dimension_energy(tables[left], P3_BINS),
                    dimension_energy(tables[right], P3_BINS),
                ),
            }

    print("[8/8] difference-of-differences: does the TRAINING contrast need the carrier?",
          flush=True)
    pairs = (("L1", "L2"), ("L1", "L0"), ("L2", "L0"))
    dod: dict[str, dict] = {}
    for seed in seeds:
        for left_arm, right_arm in pairs:
            left_tag, right_tag = f"seed{seed}_{left_arm}", f"seed{seed}_{right_arm}"
            if (left_tag, "intact") not in coupling_store:
                continue
            intact_gap = contrast(
                coupling_store[(left_tag, "intact")], coupling_store[(right_tag, "intact")]
            )
            entries: dict[str, dict] = {}
            for label, band_name, dose in conditions:
                if label == "intact":
                    continue
                dosed_left = coupling_store[(left_tag, label)]
                dosed_right = coupling_store[(right_tag, label)]
                replicates = (
                    psi_replicates(coupling_store[(left_tag, "intact")], draws["donor_gene"])
                    - psi_replicates(coupling_store[(right_tag, "intact")], draws["donor_gene"])
                    - psi_replicates(dosed_left, draws["donor_gene"])
                    + psi_replicates(dosed_right, draws["donor_gene"])
                )
                point = (
                    psi(coupling_store[(left_tag, "intact")], delta)
                    - psi(coupling_store[(right_tag, "intact")], delta)
                    - psi(dosed_left, delta)
                    + psi(dosed_right, delta)
                )
                dosed_gap = contrast(dosed_left, dosed_right)
                dod_block = percentile_interval(replicates, point)
                entries[label] = {
                    "band": band_name,
                    "dose": dose,
                    "dosed_gap": dosed_gap,
                    "dosed_gap_sesoi_verdict": sesoi_verdict(dosed_gap, SESOI),
                    "difference_of_differences": dod_block,
                    "difference_of_differences_sesoi_verdict": sesoi_verdict(dod_block, SESOI),
                    "gap_retention": retention_ratio(
                        dosed_gap["estimate"], intact_gap["estimate"]
                    ),
                    "health_floor": {
                        "mlm_health_masked_top1": min(
                            condition_report[left_tag][label]["health"]["mlm_health_masked_top1"],
                            condition_report[right_tag][label]["health"]["mlm_health_masked_top1"],
                        ),
                        "acceptor_g_top1_rate": min(
                            condition_report[left_tag][label]["health"]["acceptor_g_top1_rate"],
                            condition_report[right_tag][label]["health"]["acceptor_g_top1_rate"],
                        ),
                        "coupling_magnitude_retention": min(
                            condition_report[left_tag][label]["coupling_retention_magnitude"],
                            condition_report[right_tag][label]["coupling_retention_magnitude"],
                        ),
                    },
                }
            dod[f"seed{seed}_{left_arm}_minus_{right_arm}"] = {
                "intact_gap": intact_gap,
                "intact_gap_sesoi_verdict": sesoi_verdict(intact_gap, SESOI),
                "by_condition": entries,
            }

    arm_contrasts: dict[str, dict] = {}
    for seed in seeds:
        for arm in ARMS:
            tag = f"seed{seed}_{arm}"
            if (tag, "intact") not in coupling_store:
                continue
            block = contrast(coupling_store[(tag, "intact")], coupling_store[("base", "intact")])
            arm_contrasts[f"{tag}_minus_base"] = {
                **block,
                "sesoi_verdict": sesoi_verdict(block, SESOI),
                "caveat": (
                    "cross-checkpoint: the trained arm is not the instrument the base reading was "
                    "calibrated on, and its overall coupling mean has moved under training"
                ),
            }

    # Across-seed spread of the headline quantities. A RANGE over seeds, never an interval.
    def arm_psi(seed: int, arm: str) -> float:
        return condition_report[f"seed{seed}_{arm}"]["intact"]["psi"]["estimate"]

    across_seeds = {
        "psi_by_arm": {
            arm: spread([arm_psi(seed, arm) for seed in seeds if f"seed{seed}_{arm}" in condition_report])
            for arm in ARMS
        },
        "gap_by_pair": {
            f"{left}_minus_{right}": spread(
                [
                    dod[f"seed{seed}_{left}_minus_{right}"]["intact_gap"]["estimate"]
                    for seed in seeds
                    if f"seed{seed}_{left}_minus_{right}" in dod
                ]
            )
            for left, right in pairs
        },
    }

    payload = {
        "status": "completed",
        "artifact_class": "ENGINEERING PILOT",
        "question": (
            "Does the annotated-phase preference produced by phase-structured continued "
            "pretraining depend on the learned period-3 positional carrier?"
        ),
        "scope_caveats": {
            "seeds": (
                f"{len(seeds)} training seeds, one lineage triple each. Observed in "
                f"{len(seeds)} exploratory lineage triples -- NOT demonstrated. The replicate "
                "for a corpus intervention is the TRAINING SEED; every bootstrap interval here "
                "covers EVALUATION-PARENT resampling and nothing else, and the across-seed "
                "numbers are a RANGE over a handful of draws, never an interval."
            ),
            "non_detection": (
                "An arm contrast whose interval includes zero but also includes effects above "
                f"the {SESOI} SESOI is UNRESOLVED, not equivalent. Read `sesoi_verdict` on each "
                "contrast; this artifact never writes 'left it at baseline'."
            ),
            "edit_size": (
                f"k={args.k}: the arms differ by 1-2 bases of edit SIZE (offset {3 * args.k} "
                f"against {3 * args.k + 1}/{3 * args.k + 2}) as well as by residue mod 3. This "
                "REDUCES the size confound relative to the k=1 pilot; it does NOT REMOVE it."
            ),
            "L0_is_not_edit_matched": (
                "L0 carries no indel, so L1-minus-L0 conflates 'phase-preserving edit' with 'any "
                "edit'. L1 vs L2 is the edit-matched contrast; L0 is the generic-continuation "
                "control. Neither substitutes for the other."
            ),
            "band_power_wording": (
                "A scalar period-3 band power licenses only 'aggregate period-3 band power was "
                "similar'. It sums squared moduli, so it is blind to a phase rotation, and it "
                "sums across bins and hidden dimensions, so it is blind to redistribution. See "
                "`carrier_fourier`, which reports all three."
            ),
            "cross_checkpoint": (
                "The trained arms are no longer the instrument the base reading was calibrated "
                "on; their overall coupling means move substantially under training, so signed "
                "coupling retention ratios against a near-zero or negative baseline are not "
                "retentions. Every ratio is reported beside its raw baseline and a "
                "`signed_retention_interpretable` flag."
            ),
        },
        "primary_endpoint": {
            "name": "DoD(band, dose) = [Psi_X - Psi_Y]_intact - [Psi_X - Psi_Y]_band-attenuated",
            "why": (
                "A per-arm dose ladder asks whether a checkpoint's Psi needs the carrier. Only "
                "the difference of differences asks whether the TRAINING CONTRAST needs it, and "
                "only its comparison against power-matched control bands makes that carrier-"
                "specific."
            ),
            "clustering": "donor_gene, shared draws across every arm and condition (paired)",
            "values": dod,
        },
        "dose_ladder": condition_report,
        "carrier_fourier": carrier,
        "arm_contrasts_vs_base": arm_contrasts,
        "across_seeds": across_seeds,
        "sesoi": SESOI,
        "bands": {
            "definitions": band_report,
            "protected_periods": PROTECTED_PERIODS,
            "protected_bins": protected,
            "selection": (
                "positional_spectrum.select_power_matched_controls on the BASE table, with the "
                "period-3 band and its representable harmonic protected. The same bands are used "
                "on every arm so the intervention is identical across arms; per-arm power ratios "
                "are recorded in `carrier_fourier`."
            ),
            "doses": doses,
        },
        "base_reproduction": {
            "reproduced": bool(reproduced),
            "psi_observed": psi_base,
            "psi_reference": REFERENCE_PSI,
            "coupling_observed": float(base_scores["coupling"].mean()),
            "coupling_reference": REFERENCE_COUPLING,
            "n_cells_observed": len(cells),
            "n_cells_reference": REFERENCE_CELLS,
            "psi_tolerance": PSI_TOLERANCE,
            "source": (
                "studies/splicebert_positional_proxy_modifiability/DECISIONS.md, 2026-08-12"
            ),
        },
        "pilot_reference": PILOT_REFERENCE,
        "design": {
            "arms": {arm: {"label": ARM_LABEL[arm]} for arm in ARMS},
            "k": args.k,
            "offsets": {
                "L0": 0,
                "L1": frame_offset(args.k, 0),
                "L2": [frame_offset(args.k, 1), frame_offset(args.k, 2)],
            },
            "register_shifts": {
                "L0": 0,
                "L1": register_shift(frame_offset(args.k, 0)),
                "L2": [
                    register_shift(frame_offset(args.k, 1)),
                    register_shift(frame_offset(args.k, 2)),
                ],
            },
            "operator": (
                "length-preserving paired indel: delete `offset` bases deep in the UPSTREAM "
                "intron, insert a tandem duplication of `offset` exonic bases at "
                "donor_pos - exon_offset. donor_pos, next_acceptor_pos, the donor->acceptor span "
                "and the window length are byte-preserved and IDENTICAL across all three arms."
            ),
            "operator_limit": (
                f"the final {args.exon_offset} exonic bases before the donor, and the whole "
                "downstream exon, keep their native register; only the bulk of the upstream exon "
                "is shifted as a block"
            ),
            "exon_offset": args.exon_offset,
            "held_identical_across_arms": [
                "source windows",
                "window order",
                "crop plan (lengths and offsets)",
                "masking draws (keyed by seed, window index, epoch)",
                "optimizer, schedule, learning rate",
                "step count",
                "batch composition and order",
            ],
            "only_intended_difference": "the paired indel's offset (0 for L0; residue mod 3 for L1 vs L2)",
            "mlm_recipe": {
                "mask_rate": 0.15,
                "mask_token_share": 0.80,
                "random_token_share": 0.10,
                "unchanged_share": 0.10,
                "source": "Chen et al. 2024, reimplemented -- upstream ships architecture only",
                "length_ladder": list(DEFAULT_LENGTH_LADDER),
            },
        },
        "corpus": {
            str(seed): {
                "chromosomes": [c.strip() for c in args.chroms.split(",") if c.strip()],
                "sealed_chromosomes_never_drawn": list(SEALED_CHROMOSOMES),
                "n_windows": block["n_windows"],
                "n_source_genes": len(block["genes"]),
                "exclusion_tally": block["tally"],
                "sha256": block["sha256"],
                "edit_budget": block["edit_budget"],
                "composition_delta_vs": block["composition_delta"],
                "composition_k": block["composition_k"],
                "cropped_bases": block["cropped_bases"],
                "cropped_bases_identical_across_arms": len(set(block["cropped_bases"].values())) == 1,
                "residue_counts": {
                    str(r): int(list(block["residues"]).count(r))
                    for r in sorted(set(block["residues"]))
                },
            }
            for seed, block in corpora.items()
        },
        "training": arm_records,
        "leakage_control": {
            "evaluation_parent_genes_excluded": len(eval_genes),
            "excluded_by": REASON_DIRECT_PARENT,
            "evaluation_parent_chromosomes": {
                chrom: sum(1 for c in eval_gene_chrom.values() if c == chrom)
                for chrom in sorted(set(eval_gene_chrom.values()))
            },
            "evaluation_genes_at_risk_on_training_chromosomes": sum(
                1
                for chrom in eval_gene_chrom.values()
                if chrom in [c.strip() for c in args.chroms.split(",")]
            ),
            "near_identity_screen": {
                "reason": REASON_HOMOLOGY_HIT,
                "k": args.kmer_k,
                "min_shared_kmers": args.min_shared_kmers,
                "module": "gfmcd.splicing.parent_exclusion",
            },
            "residual_risk": (
                "the k-mer screen detects NEAR-identity only; distant paralogs diverged past "
                "about one substitution per k bases are not flagged"
            ),
        },
        "evaluation": {
            "n_cells": len(cells),
            "delta_sizes": {str(d): int((delta == d).sum()) for d in (0, 1, 2)},
            "parents_per_phase": per_phase,
            "readout": "CouplingTask.margin at the masked donor; corruption AG->AC at the acceptor",
            "psi_definition": "mean(coupling | delta == 0) - mean(coupling | delta != 0)",
            "n_boot": n_boot,
            "clustering_schemes": ["donor_gene", "acceptor_gene"],
            "shared_draws": (
                "one draw sequence is used for EVERY arm and EVERY dose, so all contrasts and the "
                "difference of differences are paired rather than independently resampled"
            ),
            "health_probes": {
                "mlm_health_masked_top1": (
                    "CouplingTask.mlm_health -- MASKED top-1 at non-site positions. NOT "
                    "CouplingScorer.mlm_health, which does not mask and cannot fail."
                ),
                "donor_g_top1_rate": "fraction of cells whose masked donor G is the top-1 base",
                "acceptor_g_top1_rate": "masked top-1 at the 3' AG's G, with the donor unmasked",
            },
        },
        "omega3": {
            "run": False,
            "reason": (
                "Omega3 asks whether an IMPOSED carrier relocation moves behaviour. This run asks "
                "whether TRAINING moved the learned Fourier phase, and whether the trained "
                "contrast depends on the carrier. They are different questions and one may not "
                "stand in for the other; no Omega3 capability preflight was run here, so no Omega3 "
                "number is reported."
            ),
        },
        "persistence": {
            "weights_dir": str(weights_dir),
            "weights": weight_files,
            "primitives": primitive_files,
            "note": (
                "Binaries are never committed to this repository. Every reported statistic is "
                "rederivable from the per-cell arrays and position tables persisted here, without "
                "re-running the model; the sha256 of each file is recorded above."
            ),
        },
        "checkpoint_sha256": checkpoint_sha,
        "seeds": {"training_and_corpus": seeds, "bootstrap": seeds[0]},
        "device": device,
        "wall_clock_seconds": time.time() - started,
    }
    _write(out, manifest_path, payload, args, checkpoint_sha, weight_files=weight_files,
           primitive_files=primitive_files, corpora=corpora)
    print(f"\nwrote {out}\nwrote {manifest_path}", flush=True)
    _summarise_to_stdout(payload, seeds)


def _summarise_to_stdout(payload: dict, seeds: list[int]) -> None:
    print("\n--- Psi at intact ---", flush=True)
    for tag, rows in payload["dose_ladder"].items():
        block = rows["intact"]["psi"]
        print(
            f"{tag:16s} Psi {block['estimate']:+.5f} [{block['ci_low']:+.5f}, "
            f"{block['ci_high']:+.5f}]  coupling {rows['intact']['coupling']['mean']:+.5f}  "
            f"mlm {rows['intact']['health']['mlm_health_masked_top1']:.4f}",
            flush=True,
        )
    print("\n--- difference of differences (period-3 vs controls) ---", flush=True)
    for name, block in payload["primary_endpoint"]["values"].items():
        print(f"{name}: intact gap {block['intact_gap']['estimate']:+.5f}", flush=True)
        for label, entry in block["by_condition"].items():
            interval = entry["difference_of_differences"]
            print(
                f"    {label:14s} DoD {interval['estimate']:+.5f} "
                f"[{interval['ci_low']:+.5f}, {interval['ci_high']:+.5f}]  "
                f"gap retention {entry['gap_retention']:+.3f}",
                flush=True,
            )


def _write(out: Path, manifest_path: Path, payload: dict, args, checkpoint_sha: str,
           *, weight_files: dict | None = None, primitive_files: dict | None = None,
           corpora: dict | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")

    manifest = ExperimentManifest.create(
        EXPERIMENT_ID,
        [
            str(Path(sys.argv[0]).relative_to(REPO_ROOT))
            if str(REPO_ROOT) in sys.argv[0]
            else sys.argv[0],
            *sys.argv[1:],
        ],
        repo=REPO_ROOT,
        status="PILOT",
        config={
            "artifact_class": "ENGINEERING PILOT",
            "stage": "engineering pilot",
            "arms": list(ARMS),
            "k": args.k,
            "seeds": args.seeds,
            "chromosomes": args.chroms,
            "steps_requested": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "exon_offset": args.exon_offset,
            "per_phase": args.per_phase,
            "kmer_k": args.kmer_k,
            "min_shared_kmers": args.min_shared_kmers,
            "doses": args.doses,
            "n_controls": args.n_controls,
            "device": payload.get("device"),
            "smoke": bool(args.smoke),
            "reuse_weights": bool(args.reuse_weights),
            "status_of_run": payload["status"],
        },
        seeds={"training_and_corpus": args.seeds, "bootstrap": args.seeds.split(",")[0]},
        notes=[
            "ENGINEERING PILOT -- a handful of training seeds, three arms each. Not the "
            "preregistered confirmatory study and not a training-process claim.",
            "L0 (native, no edit) is the generic-continuation control: without it a rise in Psi "
            "under L1 cannot be separated from any further pretraining raising it.",
            f"k={args.k} reduces but does not remove the edit-size confound between L1 and L2.",
            "Period-3 band POWER is blind to phase and to redistribution; the complex band "
            "coefficients, per-bin phases and per-dimension energies are reported instead.",
            "Omega3 was NOT run: imposed relocation is a different question from trained phase "
            "movement.",
            "Trained weights and per-cell arrays are persisted OUTSIDE this repository; their "
            "sha256 values are recorded in the artifact.",
            f"base checkpoint pytorch_model.bin sha256 = {checkpoint_sha}",
        ],
    )
    manifest.add_input(Path(args.checkpoint) / "pytorch_model.bin")
    manifest.add_input(Path(args.census))
    manifest.add_input(REPO_ROOT / "src/gfmcd/splicing/lineage_pretraining.py")
    manifest.add_input(REPO_ROOT / "src/gfmcd/splicing/carrier_phase.py")
    manifest.add_input(REPO_ROOT / "scripts/splicebert_positional_proxy_modifiability/run_lineage_pilot.py")
    manifest.add_input(Path(__file__))
    for label, path in (("gtf", args.gtf), ("fasta", args.fasta)):
        try:
            manifest.inputs.append({**file_record(path), "role": label})
        except (OSError, ValueError):
            manifest.inputs.append({"path": str(path), "role": label, "sha256": None})
    if corpora:
        manifest.config["corpus_sha256"] = {
            str(seed): block["sha256"] for seed, block in corpora.items()
        }
        manifest.config["corpus_windows"] = {
            str(seed): block["n_windows"] for seed, block in corpora.items()
        }
    if weight_files:
        manifest.config["persisted_weight_sha256"] = {
            tag: {kind: entry["sha256"] for kind, entry in block.items()}
            for tag, block in weight_files.items()
        }
    if primitive_files:
        manifest.config["persisted_primitive_sha256"] = {
            tag: entry["sha256"] for tag, entry in primitive_files.items()
        }
    manifest.add_output(out)
    manifest.write(manifest_path)


if __name__ == "__main__":
    main()
