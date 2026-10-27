#!/usr/bin/env python
"""Export one short-intron SpliceBERT attention example for visualization.

This is a post-softmax attention artifact from one forward pass.  It is descriptive
visualization input, not cohort evidence and not an OV, mediation, or behavior result.

``--source constructed`` (default) rebuilds the original 198-nt constructed sequence,
unmasked.  ``--source val`` takes the first eligible validation example (chr11/16/19,
60-150-nt donor-acceptor intron) in the stable order ``--tag`` defines, masks the donor G
exactly as the coupling task does, and exports a crop spanning donor to paired acceptor
in both absolute (A[q, k]) and relative (A[q, q+delta]) frames.  The example is chosen by
that rule, not by inspecting its attention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    load_adapter,
    load_examples,
)


TOTAL_NT = 198
WIN_LO = 21
WIN_HI = 177
ROUND_DECIMALS = 4

EXON_5 = "CAGGTGAGCACCTGCACAGCTGACCTGAAGATCACCGTGGACAAG"
EXON_3 = "AGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTG"
ORIGINAL_INTRON_BODY = (
    "ACGTACGTACGTACGGCTAACGTTGCACGATCGATCGTACGATCGGCATCGATCGTACGT"
    "AGCTAGCTAGGCATCGTACGATCGATCGTTAGCACGATCGAT"
)

# Preserve the ordering used by the original visualization export.
CIRCUIT_HEADS = (
    (3, 1),
    (0, 7),
    (0, 12),
    (0, 10),
    (0, 8),
    (0, 13),
    (0, 11),
    (2, 15),
)
CIRCUIT_HEAD_SET = frozenset(CIRCUIT_HEADS)

# --source val: the rule that picks the example, and what is exported from it.
VAL_TAG = "fig2-attention-val-window"
VAL_MARGIN = 12
VAL_HEADS = ((2, 15), (3, 1))
VAL_DECIMALS = 6  # a log colour scale needs resolution near its 1e-3 floor


def _tile_left(sequence: str, length: int) -> str:
    """Tile while preserving the splice-adjacent right edge."""
    copies = (length + len(sequence) - 1) // len(sequence)
    return (sequence * copies)[-length:]


def _tile_right(sequence: str, length: int) -> str:
    """Tile while preserving the splice-adjacent left edge."""
    copies = (length + len(sequence) - 1) // len(sequence)
    return (sequence * copies)[:length]


def build_sequence(intron_length: int) -> tuple[str, int, int]:
    """Build a 198-nt sequence with equal tiled exonic flanks."""
    if intron_length < 4:
        raise ValueError("intron length must leave room for GT, filler, and AG")
    flank_total = TOTAL_NT - intron_length
    if flank_total <= 0 or flank_total % 2:
        raise ValueError(
            f"intron length must be even and less than {TOTAL_NT} so the flanks are equal"
        )
    flank_length = flank_total // 2
    body_length = intron_length - 4
    intron_body = _tile_right(ORIGINAL_INTRON_BODY, body_length)
    intron = "GT" + intron_body + "AG"
    sequence = _tile_left(EXON_5, flank_length) + intron + _tile_right(EXON_3, flank_length)
    donor_nt = flank_length
    acceptor_nt = donor_nt + intron_length

    assert len(sequence) == TOTAL_NT
    assert sequence[donor_nt : donor_nt + 2] == "GT"
    assert sequence[acceptor_nt - 2 : acceptor_nt] == "AG"
    return sequence, donor_nt, acceptor_nt


def _rounded(value: torch.Tensor | float) -> float:
    return round(float(value), ROUND_DECIMALS)


def _artifact(
    attentions: tuple[torch.Tensor, ...],
    *,
    sequence: str,
    donor_nt: int,
    acceptor_nt: int,
    prefix_tokens: int,
    model_label: str,
) -> dict[str, Any]:
    n_layers = len(attentions)
    if not n_layers:
        raise RuntimeError("model returned no attention tensors")
    n_heads = int(attentions[0].shape[1])
    donor_token = donor_nt + prefix_tokens
    acceptor_token = acceptor_nt + prefix_tokens

    maps: dict[str, list[list[float]]] = {}
    for layer, head in CIRCUIT_HEADS:
        cropped = attentions[layer][0, head, WIN_LO:WIN_HI, WIN_LO:WIN_HI]
        # Cast before rounding: float32.tolist() re-exposes binary tails such as
        # 0.002400000113993883 and more than doubles the supposedly compact JSON.
        maps[f"L{layer}H{head}"] = np.round(
            cropped.float().cpu().numpy().astype(np.float64), ROUND_DECIMALS
        ).tolist()

    summaries = []
    for layer in range(n_layers):
        for head in range(n_heads):
            attention = attentions[layer][0, head]
            summaries.append(
                {
                    "layer": layer,
                    "head": head,
                    "donor_to_acceptor": _rounded(attention[donor_token, acceptor_token]),
                    "acceptor_to_donor": _rounded(attention[acceptor_token, donor_token]),
                    "is_circuit": (layer, head) in CIRCUIT_HEAD_SET,
                }
            )

    return {
        "_meta": {
            "assay": "live_attention_extraction_for_viz",
            "model": model_label,
            "note": (
                "Real post-softmax attention for one constructed canonical short-intron "
                "donor-acceptor example. Visualization only; not a ledger result."
            ),
        },
        "sequence": sequence,
        "donor_nt": donor_nt,
        "acceptor_nt": acceptor_nt,
        "intron_len": acceptor_nt - donor_nt,
        "donor_token": donor_token,
        "acceptor_token": acceptor_token,
        "win_lo": WIN_LO,
        "win_hi": WIN_HI,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "circuit_head_maps": maps,
        "per_head_summary": summaries,
    }


def _val_artifact(
    attentions: tuple[torch.Tensor, ...],
    *,
    example: Any,
    coupling: float,
    prefix_tokens: int,
    model_label: str,
) -> dict[str, Any]:
    """Absolute and relative crops spanning the masked donor and its paired acceptor."""
    sequence = example.sequence
    donor_nt, acceptor_nt = example.donor_pos, example.next_acceptor_pos
    lo, hi = donor_nt - VAL_MARGIN, acceptor_nt + VAL_MARGIN
    size = hi - lo
    deltas = np.arange(size) - size // 2
    if lo + deltas[0] < 0 or hi - 1 + deltas[-1] >= len(sequence):
        raise ValueError("relative frame would read keys outside the model window")

    queries = np.arange(lo, hi) + prefix_tokens
    maps: dict[str, dict[str, list[list[float]]]] = {}
    for layer, head in VAL_HEADS:
        attention = attentions[layer][0, head].float().cpu().numpy().astype(np.float64)
        absolute = attention[lo + prefix_tokens:hi + prefix_tokens, lo + prefix_tokens:hi + prefix_tokens]
        relative = attention[queries[:, None], queries[:, None] + deltas[None, :]]
        maps[f"L{layer}H{head}"] = {
            "absolute": np.round(absolute, VAL_DECIMALS).tolist(),
            "relative": np.round(relative, VAL_DECIMALS).tolist(),
        }

    donor_token, acceptor_token = donor_nt + prefix_tokens, acceptor_nt + prefix_tokens
    summaries = [
        {
            "layer": layer,
            "head": head,
            "donor_to_acceptor": _rounded(attentions[layer][0, head, donor_token, acceptor_token]),
            "acceptor_to_donor": _rounded(attentions[layer][0, head, acceptor_token, donor_token]),
            "is_circuit": (layer, head) in CIRCUIT_HEAD_SET,
        }
        for layer in range(len(attentions))
        for head in range(int(attentions[0].shape[1]))
    ]
    return {
        "_meta": {
            "assay": "live_attention_extraction_for_viz",
            "model": model_label,
            "note": (
                "Real post-softmax attention for one validation example with the donor G "
                "masked as in the coupling task. Visualization only; not a ledger result."
            ),
        },
        "example_id": example.example_id,
        "gene_id": example.gene_id,
        "chrom": example.chrom,
        "strand": example.strand,
        "split": example.split,
        "selection": {"tag": VAL_TAG, "rule": "first eligible val example in stable order"},
        "example_coupling": round(coupling, ROUND_DECIMALS),
        "sequence": sequence,
        "donor_nt": donor_nt,
        "acceptor_nt": acceptor_nt,
        "intron_len": acceptor_nt - donor_nt,
        "donor_masked": True,
        "crop": [lo, hi],
        "deltas": [int(deltas[0]), int(deltas[-1])],
        "frames": {"absolute": "A[q, k], q and k in crop", "relative": "A[q, q+delta], q in crop"},
        "n_layers": len(attentions),
        "n_heads": int(attentions[0].shape[1]),
        "head_maps": maps,
        "per_head_summary": summaries,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return revision + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", choices=("constructed", "val"), default="constructed")
    parser.add_argument("--intron", type=int, default=54, help="constructed source only")
    parser.add_argument("--gtf", default=DEFAULT_GTF, help="val source only")
    parser.add_argument("--fasta", default=DEFAULT_FASTA, help="val source only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = ("figures/splicebert/viz/live_attention_short_intron.json"
                    if args.source == "constructed"
                    else "figures/splicebert/viz/live_attention_val_window.json")

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = REPO / output_path
    manifest_path = _manifest_path(output_path)
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite immutable output: {names}")

    adapter = load_adapter(args.model, args.device, eager=True)
    if args.source == "constructed":
        sequence, donor_nt, acceptor_nt = build_sequence(args.intron)
        batch = adapter.tokenize(sequence)
        inputs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
    else:
        task = CouplingTask(adapter, batch=1)
        examples, couplings = load_examples(
            args.gtf, args.fasta, "val", 1, task=task, tag=VAL_TAG, min_effect=None
        )
        example, coupling = examples[0], float(couplings[0])
        sequence, donor_nt, acceptor_nt = example.sequence, example.donor_pos, example.next_acceptor_pos
        assert sequence[donor_nt:donor_nt + 2] == "GT"
        assert sequence[acceptor_nt - 2:acceptor_nt] == "AG"
        inputs = dict(task.encode([sequence], [donor_nt]))
        if int(inputs["input_ids"][0, donor_nt + adapter.prefix_tokens]) != task.mask_id:
            raise RuntimeError("donor G is not masked at the expected token")
    with torch.no_grad():
        raw = adapter.model(**inputs, output_attentions=True, return_dict=True)
    attentions = tuple(raw.attentions or ())

    if not attentions or any(torch.count_nonzero(attention).item() == 0 for attention in attentions):
        raise RuntimeError(
            "attention tensors are all zero; output_attentions was ignored. "
            "Refusing to serialize a fabricated attention null."
        )
    expected_tokens = len(sequence) + adapter.tokenizer.num_special_tokens_to_add(pair=False)
    expected_shape = (1, adapter.num_heads, expected_tokens, expected_tokens)
    if len(attentions) != adapter.num_layers or any(tuple(x.shape) != expected_shape for x in attentions):
        shapes = [tuple(x.shape) for x in attentions]
        raise RuntimeError(f"attention shape invariant failed: expected {expected_shape}, got {shapes}")
    if any(not torch.isfinite(x).all().item() for x in attentions):
        raise RuntimeError("attention tensors contain non-finite values")
    row_error = max(float((x.sum(dim=-1) - 1).abs().max()) for x in attentions)
    if row_error > 1e-4:
        raise RuntimeError(f"post-softmax row-sum invariant failed: max error {row_error:.3g}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO / model_path
    model_label = f"{model_path.name} (local {args.device.upper()} forward, eager attn)"
    if args.source == "constructed":
        artifact = _artifact(
            attentions,
            sequence=sequence,
            donor_nt=donor_nt,
            acceptor_nt=acceptor_nt,
            prefix_tokens=adapter.prefix_tokens,
            model_label=model_label,
        )
    else:
        artifact = _val_artifact(
            attentions,
            example=example,
            coupling=coupling,
            prefix_tokens=adapter.prefix_tokens,
            model_label=model_label,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")

    if args.source == "constructed":
        data = {
            "construction": "tiled real exonic flanks plus GT/body-prefix/AG constructed intron",
            "sequence_count": 1,
            "source_5_prime_exon": EXON_5,
            "source_3_prime_exon": EXON_3,
            "source_intron_body": ORIGINAL_INTRON_BODY,
        }
        parameters = {
            "intron_nt": args.intron,
            "total_nt": TOTAL_NT,
            "donor_nt": donor_nt,
            "acceptor_nt": acceptor_nt,
            "window": [WIN_LO, WIN_HI],
            "attention_implementation": "eager",
            "device": args.device,
            "round_decimals": ROUND_DECIMALS,
        }
        claim_boundary = (
            "One constructed sequence and one post-softmax forward pass; descriptive visualization "
            "only. No OV output, mediation, behavioral necessity, or cohort claim."
        )
    else:
        fai = Path(str(args.fasta) + ".fai")
        data = {
            "construction": "real genomic window from the coupling dataset; donor G masked",
            "sequence_count": 1,
            "split": "val",
            "example_id": example.example_id,
            "gene_id": example.gene_id,
            "selection": f"first eligible val example after stable shuffle with tag {VAL_TAG!r}",
            "gtf": {"path": str(args.gtf), "sha256": _sha256(Path(args.gtf))},
            "fasta": {"path": str(args.fasta),
                      "fai_sha256": _sha256(fai) if fai.exists() else None},
        }
        parameters = {
            "total_nt": len(sequence),
            "donor_nt": donor_nt,
            "acceptor_nt": acceptor_nt,
            "intron_nt": acceptor_nt - donor_nt,
            "crop": artifact["crop"],
            "deltas": artifact["deltas"],
            "margin_nt": VAL_MARGIN,
            "heads": [f"L{layer}H{head}" for layer, head in VAL_HEADS],
            "attention_implementation": "eager",
            "device": args.device,
            "round_decimals": VAL_DECIMALS,
        }
        claim_boundary = (
            "One validation example and one post-softmax forward pass; descriptive visualization "
            "only. No OV output, mediation, behavioral necessity, or cohort claim."
        )

    revision_path = model_path / ".hf_revision"
    manifest = {
        "assay": "live_attention_extraction_for_viz",
        "artifact": {
            "path": str(output_path.relative_to(REPO)),
            "sha256": _sha256(output_path),
        },
        "checkpoint": {
            "path": str(model_path.relative_to(REPO)) if model_path.is_relative_to(REPO) else str(model_path),
            "hf_revision": revision_path.read_text().strip() if revision_path.exists() else None,
            "config_sha256": adapter.metadata().config_sha256,
        },
        "data": data,
        "parameters": parameters,
        "seed": None,
        "code_revision": _git_revision(),
        "command": shlex.join(sys.argv),
        "claim_boundary": claim_boundary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"WROTE {output_path.relative_to(REPO)}")
    print(f"MANIFEST {manifest_path.relative_to(REPO)}")
    print(f"SEQUENCE_NT {len(sequence)}")
    print(f"DONOR_NT {donor_nt}")
    print(f"ACCEPTOR_NT {acceptor_nt}")
    print(f"INTRON_NT {acceptor_nt - donor_nt}")
    if args.source == "constructed":
        print(f"WIN_LO {WIN_LO}")
        print(f"WIN_HI {WIN_HI}")
        print("CROP_LO 42")
        print("CROP_N 72")
    else:
        print(f"EXAMPLE {example.example_id} {example.gene_id} coupling={coupling:.4f}")
        print(f"CROP {artifact['crop']} DELTAS {artifact['deltas']}")
    print(f"MAX_ATTENTION {max(float(x.max()) for x in attentions):.6g}")
    print(f"MAX_ROW_SUM_ERROR {row_error:.3g}")


if __name__ == "__main__":
    main()
