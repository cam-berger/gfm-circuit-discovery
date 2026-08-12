"""PILOT: does phase-decoupled continued pretraining move the annotated-phase preference at all?

WHAT THIS IS, STATED BEFORE ANY NUMBER IS READ
-----------------------------------------------
This is a **PILOT**, not the preregistered confirmatory study, and it is deliberately narrower
than the programme it descends from.

`studies/splicebert_positional_proxy_modifiability/DECISIONS.md` (2026-08-12, final) records
that the original premise is **falsified**: the learned period-3 carrier is *necessary* for the
annotated-phase preference ``Psi`` but does **not** encode which register is preferred.
Relocating the carrier does not relocate the preference. So a corpus manipulation cannot
re-target ``Psi`` *through* the carrier, and nothing this runner emits may be read as evidence
for the retired "positional surrogate" claim.

What is still open, and all this runner asks: **does phase-decoupled continued pretraining
change the preference AT ALL, by any route?**

**ONE SEED PER ARM IS NOT A TRAINING-PROCESS CLAIM.** The replicate for a corpus intervention
is the training seed, not the evaluation gene; this repository holds no repeated-training
variance anywhere (see ``pbc_v2_capacity.power_projection``'s ``cross_lineage`` block, which
refuses to project it). The gene-clustered interval on ``Delta_Psi`` is an interval for THIS
LINEAGE PAIR under evaluation-cell resampling, and for nothing else. It does not cover
training-seed variability, and it must never be quoted as though it did.

WHAT IT RUNS
------------
Continued pretraining from the base SpliceBERT.1024nt checkpoint -- a native third stage, not a
run from scratch -- on two corpora that differ in exactly one integer's residue mod 3:

  L1  phase-preserving   paired indel with offset 3k      -> donor-relative register shift 0
  L2  phase-decoupled    paired indel with offset 3k + q  -> register shift 1 or 2, q in {1,2}

Same source windows, same window order, same crop plan, same masking draws, same optimizer
schedule, same step count, same token budget. See ``gfmcd.splicing.lineage_pretraining`` for the
operator's derivation, its three preserved invariants, and -- important -- the block of sequence
it does NOT shift.

Objective is the published SpliceBERT recipe (Chen et al. 2024): 15% masking with 80% [MASK] /
10% random / 10% unchanged, AdamW at lr 1e-4, variable-length windows. The official repository
ships architecture only, so the trainer here is a reimplementation of the published recipe, not
upstream code.

LEAKAGE CONTROL
---------------
Every gene contributing a donor or acceptor parent to the evaluation cells is excluded from
BOTH training corpora by version-normalised ID, together with near-identity k-mer hits against
the parent windows (``gfmcd.splicing.parent_exclusion``). chr8 and chr9 are globally sealed and
are never drawn from. Training data is permanent in a way an evaluation split is not: a leaked
parent cannot be un-trained.

THE STOP RULE, WHICH FIRES BEFORE ANY TRAINING
-----------------------------------------------
The base checkpoint is evaluated FIRST on the same cells. If its ``Psi`` does not reproduce the
recorded base-checkpoint reading (+0.256 on 479 cells, coupling +0.049), the evaluation has
drifted and the run STOPS with a ``base_reproduction_failed`` artifact rather than reporting a
``Delta_Psi`` computed on an instrument that does not agree with the record.

    uv run python scripts/splicebert_positional_proxy_modifiability/run_lineage_pilot.py \
        --out results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.lineage_pretraining import (  # noqa: E402
    DEFAULT_EXON_OFFSET,
    DEFAULT_LENGTH_LADDER,
    GeometryError,
    assign_residues,
    batch_plan,
    build_chimeric_cells,
    cluster_bootstrap,
    corpus_digest,
    crop_interval,
    frame_offset,
    mask_tokens,
    paired_indel,
    period3_band_power,
    position_draws,
    psi,
    register_shift,
    uniform_draw,
    upstream_prefix_is_intronic,
)
from gfmcd.splicing.parent_exclusion import (  # noqa: E402
    REASON_DIRECT_PARENT,
    REASON_HOMOLOGY_HIT,
    kmer_index,
    normalize_gene_id,
    near_identity_hits,
)

EXPERIMENT_ID = "splicebert-positional-proxy-modifiability:lineage-pilot-v1"

BASE_CHECKPOINT_SHA256 = (
    "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d"
)
DEFAULT_CHECKPOINT = (
    "/Users/cberg/Research/projects/gfm-circuit-discovery/data/models/"
    "splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
)
DEFAULT_CENSUS = (
    "/Users/cberg/Research/projects/gfm-circuit-discovery-pbc-successor/results/"
    "splicebert_positional_boundary_coherence/development/candidate_census_v1.json"
)

#: Globally sealed. Never opened, in any study, for any reason.
SEALED_CHROMOSOMES = ("chr8", "chr9")

#: The base-checkpoint reading this run must reproduce before it is allowed to train.
#: Source: studies/splicebert_positional_proxy_modifiability/DECISIONS.md, 2026-08-12 (later).
REFERENCE_PSI = 0.256
REFERENCE_COUPLING = 0.049
REFERENCE_CELLS = 479
PSI_TOLERANCE = 0.06
CELL_COUNT_TOLERANCE = 80

#: rFFT bins of the period-3 carrier in the 1026-row position table (C8's operator band).
P3_BINS = tuple(range(339, 346))


# ------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint(checkpoint_dir: Path) -> str:
    """Hard-fail unless the base weights are byte-identical to the pinned checkpoint."""
    weights = checkpoint_dir / "pytorch_model.bin"
    if not weights.exists():
        raise SystemExit(f"FATAL: no pytorch_model.bin under {checkpoint_dir}")
    digest = sha256_file(weights)
    if digest != BASE_CHECKPOINT_SHA256:
        raise SystemExit(
            f"FATAL: base checkpoint sha256 mismatch at {weights}\n"
            f"  expected {BASE_CHECKPOINT_SHA256}\n"
            f"  found    {digest}\n"
            "Refusing to continue-pretrain from a checkpoint that is not the pinned base."
        )
    return digest


def load_eval_parents(census: Path, per_phase: int) -> tuple[dict, dict]:
    """Stream the PBC census into donor and acceptor parent pools, `per_phase` per phase.

    The census is a 2.34 GB hash-bound input; ``stream_candidates`` verifies it and yields one
    candidate at a time. One parent per (role, gene) so a single gene never supplies two parents
    in the same role. ``native_sequence`` / ``native_donor_pos`` / ``native_acceptor_pos`` are
    the FULL window and its coordinates -- ``sequence`` is only the 60-nt module and using its
    positions silently rejects every candidate.
    """
    from gfmcd.splicing.pbc_v2_capacity import stream_candidates

    donors: dict[int, list[dict]] = defaultdict(list)
    acceptors: dict[int, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for candidate in stream_candidates(census):
        role, phase, gene = candidate.get("role"), candidate.get("phase"), candidate.get("gene_id")
        if role not in ("donor", "acceptor") or phase is None or (role, gene) in seen:
            continue
        pool = donors if role == "donor" else acceptors
        if len(pool[int(phase)]) >= per_phase:
            continue
        sequence = candidate.get("native_sequence")
        donor_pos = candidate.get("native_donor_pos")
        acceptor_pos = candidate.get("native_acceptor_pos")
        if not sequence or donor_pos is None or acceptor_pos is None:
            continue
        seen.add((role, gene))
        pool[int(phase)].append(
            {
                "seq": sequence,
                "d": int(donor_pos),
                "a": int(acceptor_pos),
                "gene": gene,
                "chrom": candidate.get("chrom"),
            }
        )
        if all(len(donors[p]) >= per_phase for p in (0, 1, 2)) and all(
            len(acceptors[p]) >= per_phase for p in (0, 1, 2)
        ):
            break
    return dict(donors), dict(acceptors)


def build_training_corpus(
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
) -> dict:
    """Source windows, leakage screen, and both arms' corpora -- built once, in one order.

    Returns the two arms plus a full exclusion tally. The screen runs in two stages, and both
    are counted separately, because "excluded by gene ID" and "excluded by near-identity" are
    different assurances and collapsing them hides which one is doing the work.
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

    # Stage 1: assay eligibility + the operator's own geometry + gene-ID exclusion.
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
        # The largest offset either arm can use must fit, or the two arms would draw from
        # different window sets -- which is exactly the confound the design forbids.
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

    # Deterministic order, independent of GTF order, then cap.
    staged.sort(key=lambda ex: uniform_draw(ex.example_id, seed=seed))
    staged = staged[: int(max_windows)]

    # Stage 2: near-identity screen of the staged corpus against the evaluation parents.
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
    arms: dict[str, list[str]] = {"L1": [], "L2": []}
    plans: list[dict] = []
    for index_in_corpus, example in enumerate(kept):
        offsets = {"L1": frame_offset(k, 0), "L2": frame_offset(k, residues[index_in_corpus])}
        for arm, offset in offsets.items():
            arms[arm].append(
                paired_indel(
                    example.sequence,
                    acceptor_pos=example.acceptor_pos,
                    donor_pos=example.donor_pos,
                    next_acceptor_pos=example.next_acceptor_pos,
                    offset=offset,
                    exon_offset=exon_offset,
                )
            )
        # The crop plan depends only on preserved geometry, so it is shared by both arms.
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
                "register_shift": {a: register_shift(o) for a, o in offsets.items()},
            }
        )

    return {
        "arms": arms,
        "plans": plans,
        "tally": tally,
        "residues": residues,
        "homology_hits": homology_hits,
        "n_windows": len(kept),
        "genes": sorted({normalize_gene_id(p["gene_id"]) for p in plans}),
    }


# ------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------
def train_arm(
    *,
    checkpoint: str,
    sequences: list[str],
    plans: list[dict],
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    warmup_fraction: float = 0.1,
) -> tuple[object, dict]:
    """Continue pretraining one arm. Returns (model, record).

    Every source of randomness is derived from ``seed`` plus the window's index, never from
    iteration order or from the arm, so the two arms differ only in the bytes of their corpora.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model.train()
    torch.manual_seed(seed)

    mask_id = tokenizer.mask_token_id
    base_ids = [tokenizer.convert_tokens_to_ids(b) for b in "ACGT"]
    special_ids = set(tokenizer.all_special_ids)

    cropped = [seq[p["crop"][0] : p["crop"][1]] for seq, p in zip(sequences, plans)]
    lengths = [len(seq) for seq in cropped]
    batches = batch_plan(lengths, batch_size=batch_size, seed=seed)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, int(steps * warmup_fraction))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)

    losses: list[float] = []
    tokens_seen = 0
    masked_positions = 0
    started = time.time()
    step = 0
    epoch = 0
    while step < steps:
        for batch in batches:
            if step >= steps:
                break
            texts = [" ".join(cropped[i]) for i in batch]
            encoded = tokenizer(texts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"]
            labels = torch.full_like(input_ids, -100)
            for row, window_index in enumerate(batch):
                ids = input_ids[row].tolist()
                maskable = [token not in special_ids for token in ids]
                draws = position_draws(
                    seed=seed, window_index=window_index, epoch=epoch, n=len(ids)
                )
                new_ids, new_labels = mask_tokens(
                    ids,
                    maskable=maskable,
                    mask_id=mask_id,
                    base_ids=base_ids,
                    draws=draws,
                )
                input_ids[row] = torch.tensor(new_ids, dtype=input_ids.dtype)
                labels[row] = torch.tensor(new_labels, dtype=labels.dtype)
                masked_positions += sum(1 for value in new_labels if value != -100)
            encoded["input_ids"] = input_ids
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)

            output = model(**encoded, labels=labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)

            losses.append(float(output.loss.detach().cpu()))
            tokens_seen += int(encoded["input_ids"].numel())
            step += 1
        epoch += 1

    elapsed = time.time() - started
    model.eval()
    record = {
        "steps_trained": step,
        "steps_requested": int(steps),
        "epochs_touched": epoch,
        "batch_size": int(batch_size),
        "batches_per_epoch": len(batches),
        "learning_rate": float(lr),
        "warmup_steps": warmup,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "optimizer": "AdamW",
        "schedule": "linear warmup then linear decay",
        "tokens_seen": int(tokens_seen),
        "masked_positions": int(masked_positions),
        "crop_length_counts": {
            str(length): int(lengths.count(length)) for length in sorted(set(lengths))
        },
        "loss_first_10_mean": float(np.mean(losses[:10])) if losses else float("nan"),
        "loss_last_10_mean": float(np.mean(losses[-10:])) if losses else float("nan"),
        "loss_curve_every_25": [float(v) for v in losses[::25]],
        "wall_clock_seconds": float(elapsed),
        "device": device,
    }
    return model, record


# ------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------
def evaluate(model, tokenizer, cells: tuple[dict, ...], *, device: str, batch: int) -> dict:
    """Coupling per cell, general MLM health, and the position table's period-3 band power."""
    import torch

    from gfmcd.adapters.splicebert import SpliceBertAdapter
    from gfmcd.splicing.circuit import CouplingTask
    from gfmcd.splicing.coupling import CouplingExample
    from gfmcd.splicing.pbc_register_runner import position_embedding_module

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
                out.append(
                    task.margin(logits, [donor_positions[i] for i in chunk]).cpu().numpy()
                )
        return np.concatenate(out)

    coupling = margins("sequence") - margins("corrupt")

    # `CouplingTask.mlm_health` is the MASKED probe. `CouplingScorer.mlm_health` does not mask
    # and once returned 0.998 for a model with 94 of 96 heads ablated -- it cannot fail, so it
    # licenses nothing. Wrap the chimeric cells so the masked probe can address them.
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
    health = task.mlm_health(examples)

    table = position_embedding_module(model).weight.detach().float().cpu().numpy()
    return {
        "coupling": coupling,
        "coupling_mean": float(coupling.mean()),
        "coupling_sd": float(coupling.std()),
        "mlm_health_masked_top1": float(health),
        "position_table_shape": list(table.shape),
        "period3_band_power": period3_band_power(table, P3_BINS),
        "period3_band_bins": list(P3_BINS),
    }


# ------------------------------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output artifact path (no default, by design)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--census", default=DEFAULT_CENSUS)
    parser.add_argument("--gtf", default=os.environ.get("GFMCD_GTF"))
    parser.add_argument("--fasta", default=os.environ.get("GFMCD_FASTA"))
    parser.add_argument(
        "--chroms",
        # chr10 is deliberately included: every evaluation parent lives on chr10 or chr11, so a
        # training set disjoint from both would make the gene-ID exclusion vacuous -- it would
        # report "0 dropped" while never having been tested. chr11 is left out because it is the
        # coupling study's validation split; chr8/chr9 are globally sealed and are refused.
        default="chr1,chr2,chr3,chr4,chr10",
        help="comma-separated training chromosomes; chr8/chr9 are refused",
    )
    parser.add_argument("--windows", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=1, help="the k in offset = 3k + residue")
    parser.add_argument("--exon-offset", type=int, default=DEFAULT_EXON_OFFSET)
    parser.add_argument("--per-phase", type=int, default=60, help="evaluation parents per phase")
    parser.add_argument("--kmer-k", type=int, default=31)
    parser.add_argument("--min-shared-kmers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--eval-batch", type=int, default=9)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="mechanics only: tiny corpus, few steps, base gate NOT enforced",
    )
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

    from gfmcd.splicing.circuit import pick_device

    device = args.device or pick_device()
    started = time.time()

    print("[1/7] verifying the base checkpoint")
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_sha = verify_checkpoint(checkpoint_dir)
    print(f"      sha256 {checkpoint_sha} OK")

    print("[2/7] streaming evaluation parents from the PBC census")
    per_phase = 6 if args.smoke else args.per_phase
    donors, acceptors = load_eval_parents(Path(args.census), per_phase)
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
    clusters = np.asarray([cell["cluster"] for cell in cells], dtype=object)
    print(
        f"      cells {len(cells)}  clusters {len(set(clusters))}  "
        f"delta sizes {[int((delta == d).sum()) for d in (0, 1, 2)]}  "
        f"eval genes {len(eval_genes)}"
    )

    print("[3/7] building the two corpora with the leakage screen")
    chromosomes = tuple(c.strip() for c in args.chroms.split(",") if c.strip())
    corpus = build_training_corpus(
        gtf=args.gtf,
        fasta=args.fasta,
        chromosomes=chromosomes,
        excluded_genes=eval_genes,
        parent_sequences=parent_sequences,
        kmer_k=args.kmer_k,
        min_shared_kmers=args.min_shared_kmers,
        max_windows=40 if args.smoke else args.windows,
        seed=args.seed,
        k=args.k,
        exon_offset=args.exon_offset,
    )
    print(f"      windows {corpus['n_windows']}  tally {corpus['tally']}")
    digests = {arm: corpus_digest(seqs) for arm, seqs in corpus["arms"].items()}
    if digests["L1"] == digests["L2"]:
        raise SystemExit("FATAL: the two corpora are byte-identical; the operator did nothing")

    print("[4/7] evaluating the BASE checkpoint (stop rule)")
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    base_model = AutoModelForMaskedLM.from_pretrained(str(checkpoint_dir)).to(device).eval()
    base = evaluate(base_model, tokenizer, cells, device=device, batch=args.eval_batch)
    psi_base = psi(base["coupling"], delta)
    print(
        f"      base Psi {psi_base:+.5f}  (reference {REFERENCE_PSI:+.3f})   "
        f"coupling {base['coupling_mean']:+.5f}  (reference {REFERENCE_COUPLING:+.3f})"
    )
    reproduced = (
        abs(psi_base - REFERENCE_PSI) <= PSI_TOLERANCE
        and abs(len(cells) - REFERENCE_CELLS) <= CELL_COUNT_TOLERANCE
    )
    del base_model

    steps = 12 if args.smoke else args.steps
    if not reproduced and not args.smoke:
        payload = {
            "status": "base_reproduction_failed",
            "artifact_class": "PILOT",
            "stop_reason": (
                "the base checkpoint did not reproduce the recorded annotated-phase reading on "
                "these cells, so the evaluation has drifted; no lineage was trained and NO "
                "Delta_Psi is reported"
            ),
            "base": {
                "psi": psi_base,
                "coupling_mean": base["coupling_mean"],
                "n_cells": len(cells),
                "mlm_health_masked_top1": base["mlm_health_masked_top1"],
                "period3_band_power": base["period3_band_power"],
            },
            "reference": {
                "psi": REFERENCE_PSI,
                "coupling_mean": REFERENCE_COUPLING,
                "n_cells": REFERENCE_CELLS,
                "psi_tolerance": PSI_TOLERANCE,
                "cell_count_tolerance": CELL_COUNT_TOLERANCE,
                "source": (
                    "studies/splicebert_positional_proxy_modifiability/DECISIONS.md, "
                    "2026-08-12 (later) and (final)"
                ),
            },
            "checkpoint_sha256": checkpoint_sha,
            "wall_clock_seconds": time.time() - started,
        }
        _write(out, manifest_path, payload, args, checkpoint_sha, corpus=None, digests=None)
        print("\nSTOPPED: base Psi did not reproduce. No arm was trained.")
        return

    trained: dict[str, dict] = {}
    arm_records: dict[str, dict] = {}
    for step_no, arm in (("5/7", "L1"), ("6/7", "L2")):
        print(f"[{step_no}] continued pretraining arm {arm} ({steps} steps)")
        model, record = train_arm(
            checkpoint=str(checkpoint_dir),
            sequences=corpus["arms"][arm],
            plans=corpus["plans"],
            device=device,
            steps=steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        print(
            f"      {record['steps_trained']} steps in {record['wall_clock_seconds']:.0f}s   "
            f"loss {record['loss_first_10_mean']:.4f} -> {record['loss_last_10_mean']:.4f}"
        )
        arm_records[arm] = record
        trained[arm] = evaluate(model, tokenizer, cells, device=device, batch=args.eval_batch)
        print(
            f"      Psi {psi(trained[arm]['coupling'], delta):+.5f}   "
            f"coupling {trained[arm]['coupling_mean']:+.5f}   "
            f"mlm {trained[arm]['mlm_health_masked_top1']:.4f}"
        )
        del model

    print("[7/7] clustered intervals under three resampling schemes")
    arm_coupling = {
        "base": base["coupling"],
        "L1": trained["L1"]["coupling"],
        "L2": trained["L2"]["coupling"],
    }
    # THE PAIR KEY IS DEGENERATE, AND THAT MATTERS. Every donor parent is crossed with every
    # acceptor parent of the same rank exactly once, so `donor_gene|acceptor_gene` is unique to
    # a single cell and resampling it is an ORDINARY CELL BOOTSTRAP, not a gene-clustered one.
    # It is computed anyway, and first, because the reference reading this run is gated against
    # (+0.256 [+0.191, +0.320]) used exactly this key -- changing it would silently move the
    # comparison. The two genuine one-way gene clusterings are computed alongside it and are the
    # ones to quote for effect existence: a donor parent contributes three cells and an acceptor
    # parent three, so those keys really do group.
    scheme_clusters = {
        "pair_reference_comparable": clusters,
        "donor_gene": np.asarray(
            [normalize_gene_id(cell["donor_gene"]) for cell in cells], dtype=object
        ),
        "acceptor_gene": np.asarray(
            [normalize_gene_id(cell["acceptor_gene"]) for cell in cells], dtype=object
        ),
    }
    schemes = {
        name: cluster_bootstrap(
            arm_coupling, delta, keys, n_boot=args.n_boot, seed=args.seed
        )
        for name, keys in scheme_clusters.items()
    }
    intervals = schemes["pair_reference_comparable"]
    for name, block in schemes.items():
        print(
            f"      {name:26s} clusters {int(block['_meta']['n_clusters']):4d}  "
            f"Delta_Psi {block['L1_minus_L2']['estimate']:+.5f} "
            f"[{block['L1_minus_L2']['ci_low']:+.5f}, {block['L1_minus_L2']['ci_high']:+.5f}]"
        )

    def arm_block(name: str, block: dict) -> dict:
        return {
            "psi": intervals[name],
            "coupling_mean": block["coupling_mean"],
            "coupling_sd": block["coupling_sd"],
            "coupling_by_delta": {
                str(d): {
                    "mean": float(block["coupling"][delta == d].mean()),
                    "n": int((delta == d).sum()),
                }
                for d in (0, 1, 2)
            },
            "mlm_health_masked_top1": block["mlm_health_masked_top1"],
            "mlm_health_probe": (
                "CouplingTask.mlm_health -- MASKED top-1 at non-site positions. NOT "
                "CouplingScorer.mlm_health, which does not mask and cannot fail."
            ),
            "period3_band_power": block["period3_band_power"],
            "period3_band_bins": block["period3_band_bins"],
            "position_table_shape": block["position_table_shape"],
        }

    payload = {
        "status": "completed",
        "artifact_class": "PILOT",
        "pilot_caveat": (
            "ONE SEED PER ARM. The replicate for a corpus intervention is the TRAINING SEED, "
            "not the evaluation gene. Every interval below is a gene-clustered interval over "
            "evaluation-cell resampling for THIS ONE LINEAGE PAIR; none of them covers "
            "training-seed variability, and none may be quoted as a training-process claim."
        ),
        "premise_note": (
            "The original 'positional surrogate' premise is FALSIFIED (DECISIONS.md, "
            "2026-08-12 final): the period-3 carrier is necessary for Psi but does not encode "
            "which register is preferred, so a corpus manipulation cannot re-target the "
            "preference through the carrier. This run asks only whether phase-decoupled "
            "continued pretraining changes the preference AT ALL, by any route."
        ),
        "primary_endpoint": {
            "name": "Delta_Psi = Psi_L1 - Psi_L2",
            **intervals["L1_minus_L2"],
            "clustering": "pair_reference_comparable",
            "interval_covers": "evaluation-cell resampling only",
            "interval_does_not_cover": "training-seed variability (one seed per arm)",
            "clustering_caveat": (
                "the pair key donor_gene|acceptor_gene is unique to one cell in this design, so "
                "this interval is an ordinary CELL bootstrap. It is reported first only because "
                "the base reading this run is gated against used the same key. For effect "
                "existence, quote the donor_gene / acceptor_gene schemes in "
                "`clustering_schemes`, which genuinely group cells sharing a parent."
            ),
            "widest_gene_clustered": {
                scheme: schemes[scheme]["L1_minus_L2"]
                for scheme in ("donor_gene", "acceptor_gene")
            },
        },
        "clustering_schemes": {
            name: {
                "n_clusters": int(block["_meta"]["n_clusters"]),
                "psi": {arm: block[arm] for arm in ("base", "L1", "L2")},
                "L1_minus_L2": block["L1_minus_L2"],
                "L1_minus_base": block["L1_minus_base"],
                "L2_minus_base": block["L2_minus_base"],
            }
            for name, block in schemes.items()
        },
        "arms": {
            "base": arm_block("base", base),
            "L1": {**arm_block("L1", trained["L1"]), "training": arm_records["L1"]},
            "L2": {**arm_block("L2", trained["L2"]), "training": arm_records["L2"]},
        },
        "secondary_contrasts": {
            "L1_minus_base": intervals["L1_minus_base"],
            "L2_minus_base": intervals["L2_minus_base"],
        },
        "base_reproduction": {
            "reproduced": bool(reproduced),
            "psi_observed": psi_base,
            "psi_reference": REFERENCE_PSI,
            "coupling_observed": base["coupling_mean"],
            "coupling_reference": REFERENCE_COUPLING,
            "n_cells_observed": len(cells),
            "n_cells_reference": REFERENCE_CELLS,
            "psi_tolerance": PSI_TOLERANCE,
            "source": (
                "studies/splicebert_positional_proxy_modifiability/DECISIONS.md, "
                "2026-08-12 (later) and (final)"
            ),
        },
        "design": {
            "arms": {
                "L1": {
                    "label": "phase-preserving",
                    "offset": frame_offset(args.k, 0),
                    "register_shift": register_shift(frame_offset(args.k, 0)),
                },
                "L2": {
                    "label": "phase-decoupled",
                    "offsets": sorted({frame_offset(args.k, r) for r in set(corpus["residues"])}),
                    "register_shifts": sorted(
                        {register_shift(frame_offset(args.k, r)) for r in set(corpus["residues"])}
                    ),
                    "residue_counts": {
                        str(r): int(corpus["residues"].count(r)) for r in sorted(set(corpus["residues"]))
                    },
                },
            },
            "operator": (
                "length-preserving paired indel: delete `offset` bases deep in the UPSTREAM "
                "intron, insert a tandem duplication of `offset` exonic bases at "
                "donor_pos - exon_offset. donor_pos, next_acceptor_pos, the donor->acceptor "
                "span and the window length are byte-preserved and IDENTICAL across arms."
            ),
            "operator_limit": (
                f"the final {args.exon_offset} exonic bases before the donor, and the whole "
                "downstream exon, keep their native register; only the bulk of the upstream "
                "exon is shifted as a block"
            ),
            "exon_offset": args.exon_offset,
            "k": args.k,
            "held_identical_across_arms": [
                "source windows",
                "window order",
                "crop plan (lengths and offsets)",
                "masking draws (keyed by seed, window index, epoch)",
                "optimizer, schedule, learning rate",
                "step count",
                "batch composition and order",
            ],
            "only_difference": "the paired indel's offset residue mod 3",
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
            "chromosomes": list(chromosomes),
            "sealed_chromosomes_never_drawn": list(SEALED_CHROMOSOMES),
            "n_windows": corpus["n_windows"],
            "n_source_genes": len(corpus["genes"]),
            "exclusion_tally": corpus["tally"],
            "sha256": digests,
            "token_budget_per_arm": arm_records["L1"]["tokens_seen"],
            "token_budget_matched": (
                arm_records["L1"]["tokens_seen"] == arm_records["L2"]["tokens_seen"]
            ),
        },
        "leakage_control": {
            "evaluation_parent_genes_excluded": len(eval_genes),
            "excluded_by": REASON_DIRECT_PARENT,
            "windows_dropped_by_gene_id": corpus["tally"]["rejected_gene_excluded"],
            "windows_dropped_by_near_identity": corpus["tally"]["rejected_homology_hit"],
            # Without these two fields a `windows_dropped_by_gene_id` of 0 is uninterpretable:
            # it could mean the screen worked on a real risk, or that no evaluation gene was ever
            # on a training chromosome so there was nothing to remove. Say which.
            "evaluation_parent_chromosomes": {
                chrom: sum(1 for c in eval_gene_chrom.values() if c == chrom)
                for chrom in sorted(set(eval_gene_chrom.values()))
            },
            "evaluation_genes_at_risk_on_training_chromosomes": sum(
                1 for chrom in eval_gene_chrom.values() if chrom in chromosomes
            ),
            "at_risk_note": (
                "an evaluation gene can only leak if its chromosome is in the training set; "
                "`windows_dropped_by_gene_id` should be read against "
                "`evaluation_genes_at_risk_on_training_chromosomes`, never on its own"
            ),
            "near_identity_screen": {
                "reason": REASON_HOMOLOGY_HIT,
                "k": args.kmer_k,
                "min_shared_kmers": args.min_shared_kmers,
                "module": "gfmcd.splicing.parent_exclusion",
            },
            "residual_risk": (
                "the k-mer screen detects NEAR-identity only; distant paralogs diverged past "
                "about one substitution per k bases are not flagged -- see "
                "parent_exclusion.near_identity_hits' derived sensitivity floor"
            ),
        },
        "evaluation": {
            "n_cells": len(cells),
            "n_clusters": int(len(set(clusters))),
            "cluster_unit": "donor_gene|acceptor_gene (degenerate: one cell per key -- see "
                            "primary_endpoint.clustering_caveat)",
            "delta_sizes": {str(d): int((delta == d).sum()) for d in (0, 1, 2)},
            "parents_per_phase": per_phase,
            "readout": "CouplingTask.margin at the masked donor; corruption AG->AC at the acceptor",
            "psi_definition": "mean(coupling | delta == 0) - mean(coupling | delta != 0)",
            "n_boot": args.n_boot,
        },
        "checkpoint_sha256": checkpoint_sha,
        "seeds": {"corpus_and_training": args.seed, "bootstrap": args.seed},
        "device": device,
        "wall_clock_seconds": time.time() - started,
        "trained_weights_persisted": False,
        "trained_weights_note": (
            "binaries are never committed to this repository; the arms are reproducible from "
            "the base checkpoint hash, the corpus hashes and the seed recorded here"
        ),
    }
    _write(out, manifest_path, payload, args, checkpoint_sha, corpus=corpus, digests=digests)
    print(f"\nwrote {out}\nwrote {manifest_path}")
    for arm in ("base", "L1", "L2"):
        block = payload["arms"][arm]["psi"]
        print(
            f"\nPsi {arm:4s} {block['estimate']:+.5f} "
            f"[{block['ci_low']:+.5f}, {block['ci_high']:+.5f}]  "
            f"coupling {payload['arms'][arm]['coupling_mean']:+.5f}  "
            f"mlm {payload['arms'][arm]['mlm_health_masked_top1']:.4f}  "
            f"p3band {payload['arms'][arm]['period3_band_power']:.1f}",
            end="",
        )
    print("\n")
    for name, block in schemes.items():
        entry = block["L1_minus_L2"]
        print(
            f"Delta_Psi ({name:26s}) {entry['estimate']:+.5f} "
            f"[{entry['ci_low']:+.5f}, {entry['ci_high']:+.5f}]"
        )


def _write(out: Path, manifest_path: Path, payload: dict, args, checkpoint_sha: str,
           *, corpus: dict | None, digests: dict | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        EXPERIMENT_ID,
        [str(Path(sys.argv[0]).relative_to(REPO_ROOT)) if str(REPO_ROOT) in sys.argv[0]
         else sys.argv[0], *sys.argv[1:]],
        repo=REPO_ROOT,
        status="PILOT",
        config={
            "artifact_class": "PILOT",
            "seeds_per_arm": 1,
            "stage": "pilot",
            "premise": "original positional-surrogate premise falsified; tests change-at-all only",
            "chromosomes": args.chroms,
            "steps_requested": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "k": args.k,
            "exon_offset": args.exon_offset,
            "per_phase": args.per_phase,
            "kmer_k": args.kmer_k,
            "min_shared_kmers": args.min_shared_kmers,
            "device": payload.get("device"),
            "smoke": bool(args.smoke),
            "status_of_run": payload["status"],
        },
        seeds={"corpus_and_training": args.seed, "bootstrap": args.seed},
        notes=[
            "PILOT -- one training seed per arm. Not the preregistered confirmatory study and "
            "not a training-process claim.",
            "The original 'positional surrogate' premise is falsified; this run tests only "
            "whether phase-decoupled continued pretraining changes Psi at all, by any route.",
            "The trainer is a reimplementation of the published SpliceBERT recipe; the official "
            "repository ships architecture only.",
            "Trained weights are NOT persisted into this repository (no binaries).",
            f"base checkpoint pytorch_model.bin sha256 = {checkpoint_sha}",
        ],
    )
    manifest.add_input(Path(args.checkpoint) / "pytorch_model.bin")
    manifest.add_input(Path(args.census))
    manifest.add_input(REPO_ROOT / "src/gfmcd/splicing/lineage_pretraining.py")
    manifest.add_input(Path(__file__))
    for label, path in (("gtf", args.gtf), ("fasta", args.fasta)):
        try:
            manifest.inputs.append({**file_record(path), "role": label})
        except (OSError, ValueError):
            manifest.inputs.append({"path": str(path), "role": label, "sha256": None})
    if digests:
        manifest.config["corpus_sha256"] = digests
    if corpus:
        manifest.config["corpus_windows"] = corpus["n_windows"]
        manifest.config["exclusion_tally"] = corpus["tally"]
    manifest.add_output(out)
    manifest.write(manifest_path)


if __name__ == "__main__":
    main()
