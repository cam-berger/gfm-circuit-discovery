#!/usr/bin/env python
"""BG2 -- route mediation of the AG-preserving 3' boundary-grammar effects.

This assay follows a passed BG1 instrument preflight.  For the two preregistered
directed contrasts (T->A at -3 and G->A at +1), it asks whether the behavioral
effect can be restored through the already established L3H1 acceptor-payload
route while leaving the edited example's routing fixed::

    delta_r = r3(reference) - r3(edit)                     at ACC4
    delta_V = W_V delta_r
    recovered_o = o_edit + sum_j A_edit[donor,j] delta_V[j]

``recovered_o`` replaces only L3H1's edited donor-token output.  Thus the
primary raw linear-margin lift is a route-transmission sufficiency estimand, not an
attention or representation association.  Same-transition cross-gene payloads
and signed hidden-coordinate permutations are independently norm-matched at
the final donor-head output; those null families are never pooled.

The biological coordinates are -3, -2, -1, +1 around the 3' splice boundary.
The invariant AG bases (-2/-1) are never edited.  A positive result supports a
sequence-structured boundary grammar carried by this model route; it does not
identify U2AF1, spliceosomal binding, or biochemical mechanism.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from audit_bg_boundary import AUDIT_SCHEMA, build_audit  # noqa: E402
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.hf_bert import patch_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    eligible,
    gboot_ratio,
    load_adapter,
    pick_device,
    ratio_of_means,
)
from gfmcd.splicing.payload import value_weight_head  # noqa: E402


PREFLIGHT_SCHEMA = "gfmcd.splicebert.boundary_grammar.preflight.v1"
BEHAVIOR_SCHEMA = "gfmcd.splicebert.boundary_grammar.behavior.v1"
RESULT_SCHEMA = "gfmcd.splicebert.boundary_payload_recovery.v1"
LAYER, HEAD = 3, 1
ACC4_BIOLOGICAL_POSITIONS = (-3, -2, -1, 1)
N_BOOT_DEFAULT = 2000
BOOT_SEED = 20260809
SIGNED_SHAM_SEEDS = tuple(2026080900 + index for index in range(19))
MODEL_FILES = (
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

# The six directed cell pairs are frozen before looking at BG2 outcomes.  The
# first group averages over +1=A/G; the second averages over -3=A/C/G/T.
DIRECTED_PAIRS = (
    {
        "pair_id": "minus3_T_to_A__p1_A",
        "transition": "minus3_T_to_A",
        "background": "p1_A",
        "reference_cell": "m3_T__p1_A",
        "edit_cell": "m3_A__p1_A",
    },
    {
        "pair_id": "minus3_T_to_A__p1_G",
        "transition": "minus3_T_to_A",
        "background": "p1_G",
        "reference_cell": "m3_T__p1_G",
        "edit_cell": "m3_A__p1_G",
    },
    {
        "pair_id": "plus1_G_to_A__m3_A",
        "transition": "plus1_G_to_A",
        "background": "m3_A",
        "reference_cell": "m3_A__p1_G",
        "edit_cell": "m3_A__p1_A",
    },
    {
        "pair_id": "plus1_G_to_A__m3_C",
        "transition": "plus1_G_to_A",
        "background": "m3_C",
        "reference_cell": "m3_C__p1_G",
        "edit_cell": "m3_C__p1_A",
    },
    {
        "pair_id": "plus1_G_to_A__m3_G",
        "transition": "plus1_G_to_A",
        "background": "m3_G",
        "reference_cell": "m3_G__p1_G",
        "edit_cell": "m3_G__p1_A",
    },
    {
        "pair_id": "plus1_G_to_A__m3_T",
        "transition": "plus1_G_to_A",
        "background": "m3_T",
        "reference_cell": "m3_T__p1_G",
        "edit_cell": "m3_T__p1_A",
    },
)


def _canonical(value: Any) -> Any:
    """Convert tuples/dataclasses/numpy scalars to their canonical JSON form."""
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolved(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def checkpoint_weights_path(path: str | Path) -> Path:
    """Resolve the single checkpoint weight file used by this repository."""
    path = _resolved(path)
    if path.is_file():
        return path
    candidate = path / "pytorch_model.bin"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"checkpoint weights not found at {candidate}")


def model_file_hashes(path: str | Path) -> dict[str, str]:
    path = _resolved(path)
    if path.is_file():
        raise RuntimeError(
            "canonical BG runs require the complete model/tokenizer directory, not weights alone"
        )
    hashes = {}
    for name in MODEL_FILES:
        source = path / name
        if not source.is_file():
            raise RuntimeError(f"frozen model artifact is missing: {source}")
        hashes[name] = sha256_file(source)
    return hashes


def validate_preflight(
    path: str | Path,
    checkpoint_path: str | Path,
    protocol_path: str | Path,
    declaration_path: str | Path,
    expected_config: Mapping[str, Any],
    gtf_path: str | Path | None = None,
    fasta_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the BG1 authorization artifact and fail closed on any drift."""
    path = Path(path)
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read BG1 preflight {path}") from exc
    if result.get("schema") != PREFLIGHT_SCHEMA:
        raise RuntimeError(f"wrong preflight schema: expected {PREFLIGHT_SCHEMA}")
    if result.get("PASS") is not True:
        raise RuntimeError("BG1 preflight PASS is not true; target execution is forbidden")
    if result.get("smoke") is not False:
        raise RuntimeError("a smoke preflight cannot authorize BG2 target execution")

    bound = {
        "checkpoint_sha256": checkpoint_weights_path(checkpoint_path),
        "protocol_sha256": _resolved(protocol_path),
        "declaration_sha256": _resolved(declaration_path),
    }
    for field, source in bound.items():
        if not source.is_file():
            raise RuntimeError(f"preflight-bound input is missing: {source}")
        observed = sha256_file(source)
        if result.get(field) != observed:
            raise RuntimeError(
                f"preflight {field} mismatch: recorded {result.get(field)!r}, current {observed}"
            )
    if (
        result.get("model_file_sha256") != model_file_hashes(checkpoint_path)
        or (
            expected_config.get("model_file_sha256") is not None
            and result.get("model_file_sha256") != expected_config.get("model_file_sha256")
        )
    ):
        raise RuntimeError("preflight model/tokenizer file hashes drifted")
    for field, source in (("gtf_sha256", gtf_path), ("fasta_sha256", fasta_path)):
        if source is None:
            raise RuntimeError(f"{field} validation requires its current input path")
        source = _resolved(source)
        observed = sha256_file(source) if source.is_file() else None
        if (
            observed is None
            or result.get(field) != observed
            or (
                expected_config.get(field) is not None
                and observed != expected_config.get(field)
            )
        ):
            raise RuntimeError(f"preflight {field} mismatch for {source}")
    if _canonical(result.get("config")) != _canonical(expected_config):
        raise RuntimeError("preflight config does not equal the current frozen declaration")

    cohort = result.get("cohort")
    if not isinstance(cohort, dict) or cohort.get("split") != "train":
        raise RuntimeError("preflight cohort must be the train split")
    ids = cohort.get("ids")
    genes = cohort.get("genes")
    if not isinstance(ids, dict) or not isinstance(genes, dict):
        raise RuntimeError("preflight cohort ids/genes must be mappings by stable role")
    for role in ("behavior", "route"):
        if not isinstance(ids.get(role), list) or not ids[role]:
            raise RuntimeError(f"preflight cohort has no ordered {role} IDs")
        if not isinstance(genes.get(role), list) or len(genes[role]) != len(ids[role]):
            raise RuntimeError(f"preflight cohort {role} genes do not align with IDs")
        if len(set(ids[role])) != len(ids[role]) or len(set(genes[role])) != len(genes[role]):
            raise RuntimeError(f"preflight cohort {role} is not one-example-per-gene")
    counts = cohort.get("counts")
    if not isinstance(counts, dict):
        raise RuntimeError("preflight cohort counts are missing")
    for role, role_ids in ids.items():
        if int(counts.get(role, -1)) != len(role_ids):
            raise RuntimeError(f"preflight cohort count mismatch for {role}")
    digest = cohort.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("preflight cohort sha256 is missing or malformed")

    checks = result.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise RuntimeError("preflight checks are missing")
    failed = [name for name, check in checks.items()
              if not isinstance(check, dict) or check.get("PASS") is not True]
    if failed:
        raise RuntimeError(f"preflight contains failed checks: {failed}")
    return result


def validate_behavior_result(
    path: str | Path,
    *,
    preflight_path: str | Path,
    expected_config: Mapping[str, Any],
    expected_primary_ids: Sequence[str],
) -> dict[str, Any]:
    """Require a supported BG1 val result bound to this exact preflight/design."""
    path = _resolved(path)
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read BG1 behavior result {path}") from exc
    if result.get("schema") != BEHAVIOR_SCHEMA:
        raise RuntimeError(f"wrong BG1 behavior schema: {result.get('schema')!r}")
    if result.get("status") != "VERIFICATION":
        raise RuntimeError("BG1 behavior result must have VERIFICATION status")
    cohort = result.get("cohort", {})
    if cohort.get("split") != "val":
        raise RuntimeError("BG1 behavior result must use the val split")
    membership = cohort.get("memberships", {}).get("target_behavior", {})
    behavior_ids = membership.get("ids")
    behavior_genes = membership.get("genes")
    if (
        not isinstance(behavior_ids, list)
        or not behavior_ids
        or not isinstance(behavior_genes, list)
        or len(behavior_ids) != len(behavior_genes)
        or len(set(behavior_genes)) != len(behavior_genes)
    ):
        raise RuntimeError("BG1 behavior target membership is missing or not gene-unique")
    target = result.get("target", {})
    if target.get("verdict") != "SUPPORTED_VERIFICATION":
        raise RuntimeError("BG1 behavior target verdict does not authorize BG2")
    if list(target.get("primary_ids", ())) != list(expected_primary_ids):
        raise RuntimeError("BG1 behavior primary IDs differ from the frozen declaration")
    if _canonical(result.get("config")) != _canonical(expected_config):
        raise RuntimeError("BG1 behavior config differs from the frozen declaration")
    preflight_path = _resolved(preflight_path)
    if result.get("preflight_sha256") != sha256_file(preflight_path):
        raise RuntimeError("BG1 behavior/preflight sha256 mismatch")
    recorded_path = result.get("preflight_path")
    if not isinstance(recorded_path, str) or _resolved(recorded_path).resolve() != preflight_path.resolve():
        raise RuntimeError("BG1 behavior result names a different preflight path")
    return result


def validate_behavior_audit(
    result_path: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    """Require the saved BG1 audit to equal a fresh independent recomputation."""

    result_path = _resolved(result_path)
    manifest_path = _resolved(manifest_path)
    audit_path = _resolved(audit_path)
    try:
        recorded = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read BG1 audit {audit_path}") from exc
    recomputed = build_audit(result_path, manifest_path)
    if recomputed.get("PASS") is not True:
        raise RuntimeError(
            "fresh BG1 independent audit failed: "
            + "; ".join(map(str, recomputed.get("failures", ())))
        )
    if recorded != recomputed:
        raise RuntimeError("saved BG1 audit differs from fresh independent recomputation")
    if (
        recorded.get("schema") != AUDIT_SCHEMA
        or recorded.get("audited_schema") != BEHAVIOR_SCHEMA
    ):
        raise RuntimeError("BG1 audit schema does not authorize BG2")
    return recorded


def edited_attention_delta(
    attention_edit: torch.Tensor,
    value_weight: torch.Tensor,
    residual_delta: torch.Tensor,
) -> torch.Tensor:
    """``sum_j A_edit[j] W_V (r_ref[j]-r_edit[j])`` at the donor."""
    if attention_edit.ndim != 2 or residual_delta.ndim != 3 or value_weight.ndim != 2:
        raise ValueError("expected A [B,K], W_V [Dh,D], and residual delta [B,K,D]")
    if attention_edit.shape != residual_delta.shape[:2]:
        raise ValueError("attention and residual-window shapes disagree")
    if residual_delta.shape[-1] != value_weight.shape[-1]:
        raise ValueError("residual width and value projection disagree")
    delivered = residual_delta @ value_weight.T
    return torch.einsum("bk,bkd->bd", attention_edit, delivered)


def edited_attention_recovery(
    natural_edit: torch.Tensor,
    attention_edit: torch.Tensor,
    value_weight: torch.Tensor,
    residual_delta: torch.Tensor,
) -> torch.Tensor:
    """Natural edited L3H1 donor output plus the edited-address payload delta."""
    delta = edited_attention_delta(attention_edit, value_weight, residual_delta)
    if natural_edit.shape != delta.shape:
        raise ValueError("natural edited output and recovered delta shapes disagree")
    return natural_edit + delta


def build_donor_patch(
    recovered: torch.Tensor,
    *,
    donor_tokens: torch.Tensor | Sequence[int],
    num_heads: int,
    sequence_length: int,
    head: int = HEAD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a source cache whose only populated slots are L3H1 donor outputs."""
    if recovered.ndim != 2:
        raise ValueError("recovered head output must have shape [B,Dh]")
    if not 0 <= head < num_heads:
        raise ValueError("head index is outside num_heads")
    donor_tokens = torch.as_tensor(donor_tokens, device=recovered.device, dtype=torch.long)
    if donor_tokens.shape != (recovered.shape[0],):
        raise ValueError("donor_tokens must have one index per example")
    if bool(((donor_tokens < 0) | (donor_tokens >= sequence_length)).any()):
        raise ValueError("donor token is outside the encoded sequence")
    source = recovered.new_zeros(
        recovered.shape[0], num_heads, sequence_length, recovered.shape[1]
    )
    batch = torch.arange(recovered.shape[0], device=recovered.device)
    source[batch, head, donor_tokens] = recovered
    mask = torch.zeros(
        recovered.shape[0], sequence_length, dtype=torch.bool, device=recovered.device
    )
    mask[batch, donor_tokens] = True
    return source, mask


def norm_match(
    candidate: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match each candidate donor-output delta to the target norm, never zero-fill."""
    if candidate.shape != target.shape or candidate.ndim != 2:
        raise ValueError("candidate and target must share shape [B,Dh]")
    if not bool(torch.isfinite(candidate).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("candidate and target energies must be finite")
    candidate_norm = candidate.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    if bool((candidate_norm <= eps).any()):
        raise ValueError("zero-energy candidate cannot be norm matched")
    if bool((target_norm <= eps).any()):
        raise ValueError("zero-energy target cannot define a matched intervention")
    scale = target_norm / candidate_norm
    return candidate * scale[:, None], scale


def signed_coordinate_permutation(delta: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Apply one signed hidden-coordinate permutation to every ACC4 row.

    The transform is orthogonal and shared across rows/examples, preserving every
    spatial row Gram matrix and per-position energy exactly.
    """
    if delta.ndim < 2 or delta.shape[-1] < 2:
        raise ValueError("payload delta needs a nontrivial hidden coordinate axis")
    rng = np.random.default_rng(seed)
    permutation = torch.as_tensor(
        rng.permutation(delta.shape[-1]), dtype=torch.long, device=delta.device
    )
    signs = torch.as_tensor(
        rng.choice((-1.0, 1.0), size=delta.shape[-1]),
        dtype=delta.dtype,
        device=delta.device,
    )
    return delta.index_select(-1, permutation) * signs


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def model_parameter_sha256(model: torch.nn.Module) -> str:
    """Hash every named parameter so intervention hooks cannot silently mutate the model."""

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(raw.tobytes(order="C"))
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def stable_half_roll(records: Sequence[dict[str, Any]]) -> dict[int, int]:
    """Within one matched stratum, map every target to a different source gene."""
    order = sorted(
        range(len(records)),
        key=lambda index: hashlib.sha256(
            f"bg2-crossgene|{records[index]['gene_id']}|{records[index]['example_id']}".encode()
        ).hexdigest(),
    )
    if len(order) < 2:
        raise RuntimeError("cross-gene control requires at least two genes per directed pair")
    shift = max(1, len(order) // 2)
    mapping = {target: order[(position + shift) % len(order)]
               for position, target in enumerate(order)}
    if any(records[target]["gene_id"] == records[source]["gene_id"]
           for target, source in mapping.items()):
        raise RuntimeError("cross-gene matching reused a target gene")
    return mapping


def coarse_span_bin(span: int) -> str:
    """Two fixed bins over the frozen eligible 60--150 nt partner span."""
    if 60 <= span <= 104:
        return "060_104"
    if 105 <= span <= 150:
        return "105_150"
    raise ValueError(f"partner span {span} is outside the frozen eligible range")


def crossgene_source_map(records: Sequence[dict[str, Any]]) -> dict[int, int]:
    """Match by directed pair, strand, and coarse span before deterministic rotation."""
    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (
            str(record["pair_id"]),
            str(record["strand"]),
            coarse_span_bin(int(record["partner_span"])),
        )
        strata[key].append(index)
    output: dict[int, int] = {}
    for key, indices in sorted(strata.items()):
        if len(indices) < 2:
            raise RuntimeError(
                f"cross-gene stratum {key} has {len(indices)} example(s); refusing an unmatched null"
            )
        local = [records[index] for index in indices]
        mapping = stable_half_roll(local)
        output.update({indices[target]: indices[source] for target, source in mapping.items()})
    if set(output) != set(range(len(records))):
        raise AssertionError("cross-gene mapping is incomplete")
    return output


def _context_tensor(output: Any) -> torch.Tensor:
    context = output if torch.is_tensor(output) else output[0]
    if not torch.is_tensor(context) or context.ndim != 3:
        raise TypeError("attention-self output is not a [B,S,D] tensor")
    return context


def capture_forward(ad, encoded: Mapping[str, torch.Tensor], donor_tokens: torch.Tensor):
    """Capture logits, L3 input, edited attention, and natural L3H1 donor output."""
    box: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        box["context"] = _context_tensor(output).detach().clone()

    handle = ad.attention_self_modules[LAYER].register_forward_hook(hook)
    try:
        output = ad.model(
            **encoded,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        handle.remove()
    if "context" not in box or output.attentions is None or output.hidden_states is None:
        raise RuntimeError("failed to capture the BG2 route state")
    context = box["context"]
    batch, sequence_length, width = context.shape
    if width % ad.num_heads:
        raise RuntimeError("attention context width is not divisible by the head count")
    heads = context.view(
        batch, sequence_length, ad.num_heads, width // ad.num_heads
    ).permute(0, 2, 1, 3)
    batch_index = torch.arange(batch, device=donor_tokens.device)
    natural_head = heads[batch_index, HEAD, donor_tokens]
    attention = output.attentions[LAYER][batch_index, HEAD, donor_tokens, :]
    if float(attention.abs().max()) == 0.0:
        raise RuntimeError("edited attention is all zero; eager attention was not active")
    return output, output.hidden_states[LAYER], attention, natural_head


def _value_heads(ad, residual: torch.Tensor) -> torch.Tensor:
    """Project an L3-input residual and split its exact value vectors by head."""
    value = ad.attention_self_modules[LAYER].value(residual)
    batch, sequence_length, width = value.shape
    head_width = width // ad.num_heads
    return value.view(batch, sequence_length, ad.num_heads, head_width).permute(0, 2, 1, 3)


def _margin_arrays(bg, logits: torch.Tensor, donors: Sequence[int], task: CouplingTask):
    linear, maximum = bg.margins_from_logits(
        logits,
        donors,
        base_token_ids=task.bid,
        token_offset=task.off,
    )
    selected = logits.detach().cpu().numpy()[
        np.arange(len(donors)), np.asarray(donors, dtype=int) + task.off
    ]
    base_logits = np.column_stack([
        selected[:, task.bid[base]] for base in ("A", "C", "G", "T")
    ]).astype(float)
    return (
        np.asarray(linear, dtype=float),
        np.asarray(maximum, dtype=float),
        base_logits,
    )


def capture_records(ad, task: CouplingTask, examples: Sequence[Any], bg):
    """Capture all six preregistered reference/edit pairs, retaining only reinjection state."""
    value_weight = value_weight_head(ad).to(ad.device)
    records: list[dict[str, Any]] = []
    diagnostics = {
        "value_linearity_max_abs": 0.0,
        "edited_head_reconstruction_max_abs": 0.0,
        "edited_attention_row_sum_max_abs": 0.0,
    }
    panels = {}
    construct_hashes = {}
    for example in examples:
        cells = bg.build_target_panel(example)
        bg.validate_panel(example, cells, bg.TARGET_PANEL).require_passed()
        panels[example.example_id] = cells
        construct_hashes[example.example_id] = bg.construct_sha256(
            cells, panel_id=bg.TARGET_PANEL.panel_id
        )

    for pair in DIRECTED_PAIRS:
        for start in range(0, len(examples), task.batch):
            chunk = list(examples[start:start + task.batch])
            reference_sequences = [panels[item.example_id][pair["reference_cell"]] for item in chunk]
            edit_sequences = [panels[item.example_id][pair["edit_cell"]] for item in chunk]
            donors = [int(item.donor_pos) for item in chunk]
            reference_input = task.encode(reference_sequences, donors)
            edit_input = task.encode(edit_sequences, donors)
            donor_tokens = torch.as_tensor(
                [position + task.off for position in donors],
                dtype=torch.long,
                device=ad.device,
            )
            reference_output, reference_r3, _, _ = capture_forward(
                ad, reference_input, donor_tokens
            )
            edit_output, edit_r3, attention_edit, natural_edit = capture_forward(
                ad, edit_input, donor_tokens
            )
            batch_index = torch.arange(len(chunk), device=ad.device)[:, None].expand(-1, 4)
            acceptor_tokens = torch.as_tensor(
                [
                    [
                        bg.biological_to_array_index(
                            int(item.next_acceptor_pos),
                            position,
                            sequence_length=len(item.sequence),
                            strand=item.strand,
                        )
                        + task.off
                        for position in ACC4_BIOLOGICAL_POSITIONS
                    ]
                    for item in chunk
                ],
                dtype=torch.long,
                device=ad.device,
            )
            residual_delta = (reference_r3 - edit_r3)[batch_index, acceptor_tokens]
            attention_acc4 = attention_edit[batch_index, acceptor_tokens]
            recovered_delta = edited_attention_delta(
                attention_acc4, value_weight, residual_delta
            )

            reference_values = _value_heads(ad, reference_r3)[:, HEAD]
            edit_values = _value_heads(ad, edit_r3)[:, HEAD]
            true_value_delta = (reference_values - edit_values)[batch_index, acceptor_tokens]
            linear_value_delta = residual_delta @ value_weight.T
            diagnostics["value_linearity_max_abs"] = max(
                diagnostics["value_linearity_max_abs"],
                float((true_value_delta - linear_value_delta).abs().max()),
            )
            reconstructed_edit = torch.einsum("bs,bsd->bd", attention_edit, edit_values)
            per_example_linearity_error = (
                true_value_delta - linear_value_delta
            ).abs().flatten(1).max(dim=1).values
            per_example_reconstruction_error = (
                reconstructed_edit - natural_edit
            ).abs().max(dim=1).values
            per_example_attention_error = (attention_edit.sum(dim=-1) - 1.0).abs()
            diagnostics["edited_head_reconstruction_max_abs"] = max(
                diagnostics["edited_head_reconstruction_max_abs"],
                float((reconstructed_edit - natural_edit).abs().max()),
            )
            diagnostics["edited_attention_row_sum_max_abs"] = max(
                diagnostics["edited_attention_row_sum_max_abs"],
                float((attention_edit.sum(dim=-1) - 1.0).abs().max()),
            )
            reference_linear, reference_max, reference_base_logits = _margin_arrays(
                bg, reference_output.logits, donors, task
            )
            edit_linear, edit_max, edit_base_logits = _margin_arrays(
                bg, edit_output.logits, donors, task
            )

            for local_index, example in enumerate(chunk):
                record = {
                    **pair,
                    "example_id": example.example_id,
                    "gene_id": example.gene_id,
                    "chrom": example.chrom,
                    "strand": example.strand,
                    "donor_pos": int(example.donor_pos),
                    "acceptor_pos": int(example.next_acceptor_pos),
                    "partner_span": int(example.next_acceptor_pos - example.donor_pos),
                    "reference_sequence_sha256": sequence_sha256(reference_sequences[local_index]),
                    "edit_sequence_sha256": sequence_sha256(edit_sequences[local_index]),
                    "panel_construct_sha256": construct_hashes[example.example_id],
                    "residual_delta_sha256": tensor_sha256(residual_delta[local_index]),
                    "input_ids": edit_input["input_ids"][local_index].detach().cpu(),
                    "attention_mask": (
                        edit_input["attention_mask"][local_index].detach().cpu()
                        if "attention_mask" in edit_input else None
                    ),
                    "donor_token": int(donor_tokens[local_index]),
                    "acceptor_tokens": acceptor_tokens[local_index].detach().cpu(),
                    "residual_delta": residual_delta[local_index].detach().cpu(),
                    "attention_edit_acc4": attention_acc4[local_index].detach().cpu(),
                    "natural_edit_head": natural_edit[local_index].detach().cpu(),
                    "true_head_delta": recovered_delta[local_index].detach().cpu(),
                    "value_linearity_max_abs": float(
                        per_example_linearity_error[local_index]
                    ),
                    "edited_head_reconstruction_max_abs": float(
                        per_example_reconstruction_error[local_index]
                    ),
                    "edited_attention_row_sum_abs_error": float(
                        per_example_attention_error[local_index]
                    ),
                    "reference_linear_margin": float(reference_linear[local_index]),
                    "edit_linear_margin": float(edit_linear[local_index]),
                    "reference_max_margin": float(reference_max[local_index]),
                    "edit_max_margin": float(edit_max[local_index]),
                    "reference_base_logits": reference_base_logits[local_index].tolist(),
                    "edit_base_logits": edit_base_logits[local_index].tolist(),
                }
                records.append(record)
        print(f"captured {pair['pair_id']}: {len(examples)} genes", flush=True)
    return records, diagnostics, value_weight.detach().cpu()


def injected_margins(
    ad,
    task: CouplingTask,
    records: Sequence[dict[str, Any]],
    head_deltas: Sequence[torch.Tensor],
    bg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject ``o_edit + head_delta`` donor-only into each edited run."""
    if len(records) != len(head_deltas):
        raise ValueError("records and head deltas must align")
    linear_values: list[float] = []
    max_values: list[float] = []
    base_values: list[list[float]] = []
    for start in range(0, len(records), task.batch):
        chunk = records[start:start + task.batch]
        deltas = head_deltas[start:start + task.batch]
        lengths = {int(record["input_ids"].shape[0]) for record in chunk}
        if len(lengths) != 1:
            raise RuntimeError("cannot batch encoded examples of different lengths")
        encoded = {
            "input_ids": torch.stack([record["input_ids"] for record in chunk]).to(ad.device)
        }
        if chunk[0]["attention_mask"] is not None:
            encoded["attention_mask"] = torch.stack(
                [record["attention_mask"] for record in chunk]
            ).to(ad.device)
        donor_tokens = torch.as_tensor(
            [record["donor_token"] for record in chunk], dtype=torch.long, device=ad.device
        )
        recovered = torch.stack(
            [record["natural_edit_head"] + delta for record, delta in zip(chunk, deltas)]
        ).to(ad.device)
        source, mask = build_donor_patch(
            recovered,
            donor_tokens=donor_tokens,
            num_heads=ad.num_heads,
            sequence_length=next(iter(lengths)),
        )
        with patch_heads(ad, {LAYER: source}, [(LAYER, HEAD)], token_mask=mask):
            logits = ad.model(**encoded).logits.float()
        donors = [int(record["donor_pos"]) for record in chunk]
        linear, maximum, base_logits = _margin_arrays(bg, logits, donors, task)
        linear_values.extend(linear.tolist())
        max_values.extend(maximum.tolist())
        base_values.extend(base_logits.tolist())
    return (
        np.asarray(linear_values),
        np.asarray(max_values),
        np.asarray(base_values),
    )


def _matched_control_deltas(
    records: Sequence[dict[str, Any]], value_weight: torch.Tensor
) -> tuple[list[torch.Tensor], list[float], list[int]]:
    sources = crossgene_source_map(records)
    output, scales = [], []
    for target_index in range(len(records)):
        target = records[target_index]
        source = records[sources[target_index]]
        raw = edited_attention_delta(
            target["attention_edit_acc4"][None],
            value_weight,
            source["residual_delta"][None],
        )
        matched, scale = norm_match(raw, target["true_head_delta"][None])
        output.append(matched[0])
        scales.append(float(scale[0]))
    return output, scales, [sources[index] for index in range(len(records))]


def _signed_control_deltas(
    records: Sequence[dict[str, Any]], value_weight: torch.Tensor, seed: int
) -> tuple[list[torch.Tensor], list[float], list[str]]:
    output, scales, residual_hashes = [], [], []
    for record in records:
        sham_residual = signed_coordinate_permutation(
            record["residual_delta"], seed=seed
        )
        raw = edited_attention_delta(
            record["attention_edit_acc4"][None], value_weight, sham_residual[None]
        )
        matched, scale = norm_match(raw, record["true_head_delta"][None])
        output.append(matched[0])
        scales.append(float(scale[0]))
        residual_hashes.append(tensor_sha256(sham_residual))
    return output, scales, residual_hashes


def target_panel_eligible(examples: Iterable[Any], bg) -> tuple[Any, ...]:
    """Apply only the model-free target-panel common-support filter."""
    output = []
    for example in examples:
        cells = bg.build_panel(
            example.sequence,
            int(example.next_acceptor_pos),
            bg.TARGET_PANEL,
            strand=example.strand,
            require_valid=False,
        )
        if bg.validate_panel(example, cells, bg.TARGET_PANEL).passed:
            output.append(example)
    return tuple(output)


def validate_train_cohort(bg, preflight: Mapping[str, Any], examples: Sequence[Any]):
    """Rebuild the outcome-blind train memberships recorded by the preflight."""
    cohort = preflight["cohort"]
    train_pool = target_panel_eligible(eligible(examples, "train"), bg)
    roles = tuple(cohort["ids"])
    rebuilt = {
        role: bg.select_gene_cohort(
            train_pool,
            role=role,
            namespace=bg.ROLE_NAMESPACE,
            role_weights=bg.ROLE_WEIGHTS,
            n=int(cohort["counts"][role]),
        )
        for role in roles
    }
    declaration = bg.cohort_declaration(rebuilt)
    for field in ("ids", "genes", "sha256"):
        if _canonical(declaration[field]) != _canonical(cohort[field]):
            raise RuntimeError(f"current data do not reconstruct preflight cohort {field}")
    route = rebuilt["route"]
    if (
        len(route) < bg.MIN_TRAIN_ROUTE_GENES
        or {example.strand for example in route} != {"+", "-"}
    ):
        raise RuntimeError(
            "preflight route cohort fails the frozen train supply/both-strands gate"
        )
    return rebuilt


def select_target_cohort(
    bg,
    examples: Sequence[Any],
    *,
    split: str,
    n: int | None,
    minimum_genes: int,
    minimum_per_strand: int,
):
    if split != "val":
        raise RuntimeError(
            "BG2 is frozen to val; train is preflight, while test is outside BG2 and "
            "requires an explicit hypothesis-specific held-out replication"
        )
    pool = target_panel_eligible(eligible(examples, split), bg)
    selected = bg.select_gene_cohort(
        pool,
        role="route",
        namespace=bg.ROLE_NAMESPACE,
        role_weights=bg.ROLE_WEIGHTS,
        n=n,
    )
    if len(selected) < minimum_genes:
        raise RuntimeError(
            f"BG2 route cohort has {len(selected)} genes; frozen minimum is {minimum_genes}"
        )
    strand_counts = {
        strand: sum(example.strand == strand for example in selected)
        for strand in ("+", "-")
    }
    if min(strand_counts.values()) < minimum_per_strand:
        raise RuntimeError(
            f"BG2 route cohort fails the frozen >={minimum_per_strand} genes per strand "
            "supply gate: "
            f"{strand_counts}"
        )
    if len({example.gene_id for example in selected}) != len(selected):
        raise AssertionError("target cohort is not one-example-per-gene")
    if any(bg.stable_gene_role(
        example.gene_id,
        namespace=bg.ROLE_NAMESPACE,
        role_weights=bg.ROLE_WEIGHTS,
    ) != "route" for example in selected):
        raise AssertionError("target cohort crossed the stable role boundary")
    return selected


def validate_target_membership(
    bg,
    preflight: Mapping[str, Any],
    examples: Sequence[Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Require the target cohort to equal the model-free membership frozen at preflight."""
    current = {
        "ids": [example.example_id for example in examples],
        "genes": [example.gene_id for example in examples],
        "strands": [example.strand for example in examples],
        "sha256": bg.cohort_sha256(examples),
    }
    frozen = preflight.get("target_cohorts", {}).get(split, {}).get("target_route")
    if _canonical(frozen) != _canonical(current):
        raise RuntimeError("BG2 target route membership differs from the preflight census")
    return current


def _transition_gene_values(cells: Sequence[dict[str, Any]], value_fn):
    expected = {
        "minus3_T_to_A": {"p1_A", "p1_G"},
        "plus1_G_to_A": {"m3_A", "m3_C", "m3_G", "m3_T"},
    }
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["transition"], cell["gene_id"])].append(
            (cell["background"], float(value_fn(cell)))
        )
    output: dict[str, tuple[np.ndarray, list[str]]] = {}
    for transition, backgrounds in expected.items():
        values, genes = [], []
        for (item_transition, gene), rows in grouped.items():
            if item_transition != transition:
                continue
            if {background for background, _value in rows} != backgrounds:
                raise RuntimeError(
                    f"gene {gene} lacks the complete {transition} background average"
                )
            values.append(float(np.mean([value for _background, value in rows])))
            genes.append(gene)
        if not values:
            raise RuntimeError(f"no cells for transition {transition}")
        output[transition] = (np.asarray(values, dtype=float), genes)
    return output


def _mean_ci(values: np.ndarray, genes: Sequence[str], *, n_boot: int, seed: int):
    by_gene: dict[str, list[float]] = defaultdict(list)
    for value, gene in zip(values, genes):
        if np.isfinite(value):
            by_gene[gene].append(float(value))
    ordered_genes = list(by_gene)
    if not ordered_genes:
        raise RuntimeError("gene bootstrap has no finite values")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.integers(0, len(ordered_genes), size=len(ordered_genes))
        draws[index] = np.mean([
            value
            for gene_index in sampled
            for value in by_gene[ordered_genes[gene_index]]
        ])
    return {
        "mean": float(np.mean(values)),
        "lower_95_one_sided": float(np.quantile(draws, 0.05)),
        "gene_bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n_genes": len(set(genes)),
        "n_bootstrap": n_boot,
        "seed": seed,
    }


def summarize_cells(cells: Sequence[dict[str, Any]], bg, *, n_boot: int):
    behavior_linear = _transition_gene_values(
        cells, lambda cell: cell["reference_linear_margin"] - cell["edit_linear_margin"]
    )
    behavior_max = _transition_gene_values(
        cells, lambda cell: cell["reference_max_margin"] - cell["edit_max_margin"]
    )
    true_linear = _transition_gene_values(cells, lambda cell: cell["true"]["linear_lift"])
    true_max = _transition_gene_values(cells, lambda cell: cell["true"]["max_lift"])
    cross_linear = _transition_gene_values(
        cells, lambda cell: cell["cross_gene"]["linear_lift"]
    )
    true_minus_cross = _transition_gene_values(
        cells,
        lambda cell: cell["true"]["linear_lift"] - cell["cross_gene"]["linear_lift"],
    )

    transitions = tuple(behavior_linear)
    observed = np.asarray([np.mean(true_linear[name][0]) for name in transitions])
    null = np.empty((len(SIGNED_SHAM_SEEDS), len(transitions)), dtype=float)
    signed_by_transition: dict[str, list[dict[str, float]]] = {
        name: [] for name in transitions
    }
    for seed_index, seed in enumerate(SIGNED_SHAM_SEEDS):
        aggregate = _transition_gene_values(
            cells,
            lambda cell, index=seed_index: cell["signed_coordinate_shams"][index][
                "linear_lift"
            ],
        )
        for transition_index, transition in enumerate(transitions):
            statistic = float(np.mean(aggregate[transition][0]))
            null[seed_index, transition_index] = statistic
            signed_by_transition[transition].append({"seed": seed, "mean": statistic})
    adjusted_p = bg.max_family_add_one_pvalues(observed, null, two_sided=False)

    output = {}
    for transition_index, transition in enumerate(transitions):
        linear_values, genes = true_linear[transition]
        max_values, max_genes = true_max[transition]
        denominator, denominator_genes = behavior_max[transition]
        if genes != max_genes or genes != denominator_genes:
            raise RuntimeError("transition-level gene order drifted across endpoints")
        denominator_sum = float(np.sum(denominator))
        if denominator_sum <= 1e-9:
            recovery_ratio = None
            recovery_ci = None
        else:
            recovery_ratio = ratio_of_means(max_values, denominator)
            recovery_ci = list(
                gboot_ratio(
                    max_values,
                    denominator,
                    genes,
                    n=n_boot,
                    seed=BOOT_SEED + 20 + transition_index,
                )
            )
        cross_values, cross_genes = cross_linear[transition]
        difference, difference_genes = true_minus_cross[transition]
        if genes != cross_genes or genes != difference_genes:
            raise RuntimeError("cross-gene control is not target-paired")
        primary = _mean_ci(
            linear_values,
            genes,
            n_boot=n_boot,
            seed=BOOT_SEED + transition_index,
        )
        cross_difference = _mean_ci(
            difference,
            genes,
            n_boot=n_boot,
            seed=BOOT_SEED + 10 + transition_index,
        )
        output[transition] = {
            "behavior_linear_reference_minus_edit": _mean_ci(
                behavior_linear[transition][0],
                behavior_linear[transition][1],
                n_boot=n_boot,
                seed=BOOT_SEED + 30 + transition_index,
            ),
            "primary_linear_raw_lift": primary,
            "secondary_max_raw_lift": _mean_ci(
                max_values,
                genes,
                n_boot=n_boot,
                seed=BOOT_SEED + 40 + transition_index,
            ),
            "secondary_max_denominator_reference_minus_edit": {
                **_mean_ci(
                    denominator,
                    genes,
                    n_boot=n_boot,
                    seed=BOOT_SEED + 60 + transition_index,
                ),
                "sum": denominator_sum,
                "positive": bool(denominator_sum > 0.0),
                "near_zero": bool(abs(denominator_sum) <= 1e-9),
            },
            "secondary_max_recovery_ratio_of_means": recovery_ratio,
            "secondary_max_recovery_ratio_ci95": recovery_ci,
            "cross_gene_same_transition": {
                **_mean_ci(
                    cross_values,
                    genes,
                    n_boot=n_boot,
                    seed=BOOT_SEED + 50 + transition_index,
                ),
                "true_minus_cross_gene": cross_difference,
                "family": "same-output norm-matched cross-gene; not pooled with permutations",
            },
            "signed_coordinate_permutation": {
                "controls": signed_by_transition[transition],
                "p_add_one_one_sided_fwer_max_family": float(adjusted_p[transition_index]),
                "family_size": len(SIGNED_SHAM_SEEDS),
                "scanned_family": list(transitions),
                "family": "same-output norm-matched signed-coordinate permutations",
            },
        }
        output[transition]["route_supported"] = bool(
            primary["lower_95_one_sided"] > 0.0
            and cross_difference["lower_95_one_sided"] > 0.0
            and adjusted_p[transition_index] <= 0.05
        )
    return output


def _public_cells(
    records: Sequence[dict[str, Any]],
    true_linear: np.ndarray,
    true_max: np.ndarray,
    true_base_logits: np.ndarray,
    cross_linear: np.ndarray,
    cross_max: np.ndarray,
    cross_base_logits: np.ndarray,
    cross_scales: Sequence[float],
    cross_norms: Sequence[float],
    cross_sources: Sequence[int],
    signed_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for index, record in enumerate(records):
        signed = []
        for result in signed_results:
            signed.append({
                "seed": result["seed"],
                "linear_margin": float(result["linear"][index]),
                "max_margin": float(result["maximum"][index]),
                "base_logits_ACGT": result["base_logits"][index].tolist(),
                "linear_lift": float(result["linear"][index] - record["edit_linear_margin"]),
                "max_lift": float(result["maximum"][index] - record["edit_max_margin"]),
                "norm_scale": float(result["scales"][index]),
                "matched_head_delta_norm": float(
                    result["norms"][index]
                ),
                "norm_relative_error": abs(
                    float(result["norms"][index])
                    - float(record["true_head_delta"].norm())
                ) / float(record["true_head_delta"].norm()),
                "transformed_residual_delta_sha256": result["residual_hashes"][index],
            })
        source = records[cross_sources[index]]
        output.append({
            "example_id": record["example_id"],
            "gene_id": record["gene_id"],
            "chrom": record["chrom"],
            "strand": record["strand"],
            "donor_pos": record["donor_pos"],
            "acceptor_pos": record["acceptor_pos"],
            "partner_span": record["partner_span"],
            "span_bin": coarse_span_bin(record["partner_span"]),
            "pair_id": record["pair_id"],
            "transition": record["transition"],
            "background": record["background"],
            "reference_cell": record["reference_cell"],
            "edit_cell": record["edit_cell"],
            "reference_sequence_sha256": record["reference_sequence_sha256"],
            "edit_sequence_sha256": record["edit_sequence_sha256"],
            "panel_construct_sha256": record["panel_construct_sha256"],
            "residual_delta_sha256": record["residual_delta_sha256"],
            "reference_linear_margin": record["reference_linear_margin"],
            "edit_linear_margin": record["edit_linear_margin"],
            "reference_max_margin": record["reference_max_margin"],
            "edit_max_margin": record["edit_max_margin"],
            "reference_base_logits_ACGT": record["reference_base_logits"],
            "edit_base_logits_ACGT": record["edit_base_logits"],
            "base_logit_order": ["A", "C", "G", "T"],
            "acceptor_tokens": record["acceptor_tokens"].tolist(),
            "edited_attention_ACC4": record["attention_edit_acc4"].tolist(),
            "value_linearity_max_abs": record["value_linearity_max_abs"],
            "edited_head_reconstruction_max_abs": record[
                "edited_head_reconstruction_max_abs"
            ],
            "edited_attention_row_sum_abs_error": record[
                "edited_attention_row_sum_abs_error"
            ],
            "true_head_delta_norm": float(record["true_head_delta"].norm()),
            "true": {
                "linear_margin": float(true_linear[index]),
                "max_margin": float(true_max[index]),
                "base_logits_ACGT": true_base_logits[index].tolist(),
                "linear_lift": float(true_linear[index] - record["edit_linear_margin"]),
                "max_lift": float(true_max[index] - record["edit_max_margin"]),
            },
            "cross_gene": {
                "source_example_id": source["example_id"],
                "source_gene_id": source["gene_id"],
                "source_strand": source["strand"],
                "source_span_bin": coarse_span_bin(source["partner_span"]),
                "source_residual_delta_sha256": source["residual_delta_sha256"],
                "linear_margin": float(cross_linear[index]),
                "max_margin": float(cross_max[index]),
                "base_logits_ACGT": cross_base_logits[index].tolist(),
                "linear_lift": float(cross_linear[index] - record["edit_linear_margin"]),
                "max_lift": float(cross_max[index] - record["edit_max_margin"]),
                "norm_scale": float(cross_scales[index]),
                "matched_head_delta_norm": float(cross_norms[index]),
                "norm_relative_error": abs(
                    float(cross_norms[index])
                    - float(record["true_head_delta"].norm())
                ) / float(record["true_head_delta"].norm()),
            },
            "signed_coordinate_shams": signed,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", default=(
        "results/splicebert_coupling/bg1_boundary_preflight_v1.json"
    ))
    parser.add_argument("--behavior-result", default=(
        "results/splicebert_coupling/bg1_boundary_grammar_val_v1.json"
    ))
    parser.add_argument("--behavior-manifest", default=(
        "results/splicebert_coupling/bg1_boundary_grammar_val_v1.manifest.json"
    ))
    parser.add_argument("--behavior-audit", default=(
        "results/splicebert_coupling/bg1_boundary_grammar_val_v1.audit.json"
    ))
    parser.add_argument("--protocol", default=(
        "studies/splicebert_partner_selection/preregistrations/"
        "PREREGISTRATION_BG1_ACCEPTOR_BOUNDARY_GRAMMAR.md"
    ))
    parser.add_argument(
        "--declaration", default="src/gfmcd/splicing/boundary_grammar.py"
    )
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out",
        default="results/splicebert_coupling/bg2_boundary_payload_recovery_val_v1.json",
    )
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("batch must be positive")

    # No --force escape hatch: both artifacts are resolved and guarded before model load.
    out = guard_output_path(_resolved(args.out))
    manifest_path = guard_output_path(out.with_suffix(".manifest.json"))

    from gfmcd.splicing import boundary_grammar as bg

    if (
        bg.ANALYSIS_SEED != BOOT_SEED
        or bg.BG2_BOOTSTRAP_DRAWS != N_BOOT_DEFAULT
        or bg.preflight_config()["analysis"]["signed_coordinate_shams"]
        != len(SIGNED_SHAM_SEEDS)
    ):
        raise RuntimeError("BG2 analysis seeds/count drifted from the frozen declaration")

    preflight_path = _resolved(args.preflight)
    protocol_path = _resolved(args.protocol)
    declaration_path = _resolved(args.declaration)
    gtf_path = _resolved(args.gtf)
    fasta_path = _resolved(args.fasta)
    expected_config = bg.preflight_config()
    if (
        MIN_TARGET_ROUTE_GENES != bg.MIN_BG2_TARGET_GENES
        or MIN_TARGET_ROUTE_PER_STRAND != bg.MIN_BG2_TARGET_GENES_PER_STRAND
    ):
        raise RuntimeError("BG2 runner supply constants drifted from the frozen declaration")
    preflight = validate_preflight(
        preflight_path,
        args.model,
        protocol_path,
        declaration_path,
        expected_config,
        gtf_path=gtf_path,
        fasta_path=fasta_path,
    )
    behavior_path = _resolved(args.behavior_result)
    behavior = validate_behavior_result(
        behavior_path,
        preflight_path=preflight_path,
        expected_config=expected_config,
        expected_primary_ids=(
            bg.TARGET_MINUS3_T_MINUS_A_ID,
            bg.TARGET_PLUS1_G_MINUS_A_ID,
        ),
    )
    behavior_manifest_path = _resolved(args.behavior_manifest)
    behavior_audit_path = _resolved(args.behavior_audit)
    behavior_audit = validate_behavior_audit(
        behavior_path, behavior_manifest_path, behavior_audit_path
    )

    all_examples = bg.load_boundary_examples(str(gtf_path), str(fasta_path))
    rebuilt_preflight = validate_train_cohort(bg, preflight, all_examples)
    target_examples = select_target_cohort(
        bg,
        all_examples,
        split=args.split,
        n=None,
        minimum_genes=bg.MIN_BG2_TARGET_GENES,
        minimum_per_strand=bg.MIN_BG2_TARGET_GENES_PER_STRAND,
    )
    frozen_target_membership = validate_target_membership(
        bg, preflight, target_examples, split=args.split
    )
    preflight_genes = {
        example.gene_id
        for examples in rebuilt_preflight.values()
        for example in examples
    }
    target_genes = {example.gene_id for example in target_examples}
    if preflight_genes.intersection(target_genes):
        raise RuntimeError("BG2 target genes overlap the BG1 preflight cohort")
    behavior_genes = set(
        behavior["cohort"]["memberships"]["target_behavior"]["genes"]
    )
    if behavior_genes.intersection(target_genes):
        raise RuntimeError("BG2 route genes overlap the BG1 val behavior cohort")

    torch.set_grad_enabled(False)
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    task = CouplingTask(adapter, batch=args.batch)
    checkpoint_before = model_parameter_sha256(adapter.model)

    records, diagnostics, value_weight = capture_records(
        adapter, task, target_examples, bg
    )
    diagnostics["n_records"] = len(records)
    diagnostics["n_genes"] = len(target_examples)
    diagnostics["all_true_head_deltas_nonzero"] = bool(
        all(float(record["true_head_delta"].norm()) > 1e-12 for record in records)
    )
    capture_pass = bool(
        diagnostics["value_linearity_max_abs"] <= 1e-4
        and diagnostics["edited_head_reconstruction_max_abs"] <= 1e-4
        and diagnostics["edited_attention_row_sum_max_abs"] <= 1e-4
        and diagnostics["all_true_head_deltas_nonzero"]
    )
    diagnostics["capture_PASS"] = capture_pass
    if not capture_pass:
        raise RuntimeError(
            "BG2 capture invariant failed; freeze interpretation and diagnose the instrument"
        )

    zero_deltas = [torch.zeros_like(record["true_head_delta"]) for record in records]
    identity_linear, identity_max, _identity_base_logits = injected_margins(
        adapter, task, records, zero_deltas, bg
    )
    diagnostics["edited_output_replay_linear_max_abs"] = float(max(
        abs(identity_linear[index] - record["edit_linear_margin"])
        for index, record in enumerate(records)
    ))
    diagnostics["edited_output_replay_max_margin_max_abs"] = float(max(
        abs(identity_max[index] - record["edit_max_margin"])
        for index, record in enumerate(records)
    ))
    diagnostics["edited_output_replay_PASS"] = bool(
        diagnostics["edited_output_replay_linear_max_abs"] <= 1e-4
        and diagnostics["edited_output_replay_max_margin_max_abs"] <= 1e-4
    )
    if not diagnostics["edited_output_replay_PASS"]:
        raise RuntimeError(
            "natural edited-output replay failed; freeze interpretation before BG2 interventions"
        )

    true_deltas = [record["true_head_delta"] for record in records]
    true_linear, true_max, true_base_logits = injected_margins(
        adapter, task, records, true_deltas, bg
    )

    cross_deltas, cross_scales, cross_sources = _matched_control_deltas(
        records, value_weight
    )
    cross_linear, cross_max, cross_base_logits = injected_margins(
        adapter, task, records, cross_deltas, bg
    )
    cross_norms = [float(delta.norm()) for delta in cross_deltas]
    diagnostics["cross_gene_norm_match_max_relative"] = float(max(
        abs(norm - float(record["true_head_delta"].norm()))
        / float(record["true_head_delta"].norm())
        for norm, record in zip(cross_norms, records)
    ))

    signed_results = []
    signed_norm_error = 0.0
    for control_index, seed in enumerate(SIGNED_SHAM_SEEDS):
        deltas, scales, residual_hashes = _signed_control_deltas(
            records, value_weight, seed
        )
        linear, maximum, base_logits = injected_margins(
            adapter, task, records, deltas, bg
        )
        norms = [float(delta.norm()) for delta in deltas]
        signed_norm_error = max(
            signed_norm_error,
            max(
                abs(norm - float(record["true_head_delta"].norm()))
                / float(record["true_head_delta"].norm())
                for norm, record in zip(norms, records)
            ),
        )
        signed_results.append({
            "seed": seed,
            "linear": linear,
            "maximum": maximum,
            "base_logits": base_logits,
            "scales": scales,
            "norms": norms,
            "residual_hashes": residual_hashes,
        })
        print(
            f"signed-coordinate control {control_index + 1}/{len(SIGNED_SHAM_SEEDS)}",
            flush=True,
        )
    diagnostics["signed_coordinate_norm_match_max_relative"] = float(signed_norm_error)
    diagnostics["norm_match_PASS"] = bool(
        diagnostics["cross_gene_norm_match_max_relative"] <= 1e-3
        and diagnostics["signed_coordinate_norm_match_max_relative"] <= 1e-3
    )
    if not diagnostics["norm_match_PASS"]:
        raise RuntimeError("same-output norm matching drifted beyond tolerance")

    checkpoint_after = model_parameter_sha256(adapter.model)
    diagnostics["checkpoint_parameter_sha256_before"] = checkpoint_before
    diagnostics["checkpoint_parameter_sha256_after"] = checkpoint_after
    diagnostics["checkpoint_unmutated_PASS"] = checkpoint_before == checkpoint_after
    if not diagnostics["checkpoint_unmutated_PASS"]:
        raise RuntimeError("BG2 unexpectedly mutated checkpoint parameters")

    cells = _public_cells(
        records,
        true_linear,
        true_max,
        true_base_logits,
        cross_linear,
        cross_max,
        cross_base_logits,
        cross_scales,
        cross_norms,
        cross_sources,
        signed_results,
    )
    summaries = summarize_cells(cells, bg, n_boot=N_BOOT_DEFAULT)
    all_supported = all(item["route_supported"] for item in summaries.values())
    target_cohort = bg.cohort_declaration({"route": target_examples})
    constructs = [
        bg.panel_construct_declaration(example, bg.TARGET_PANEL)
        for example in target_examples
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "status": "VERIFICATION",
        "verdict": (
            "BG2_ROUTE_SUPPORTED" if all_supported else "BG2_INCONCLUSIVE"
        ),
        "claim_boundary": (
            "A positive verdict establishes causal transmission/sufficiency through the declared "
            "model route, not complete mediation or a mediated fraction. It does not identify "
            "U2AF1, binding, or spliceosomal biochemistry."
        ),
        "wording_if_independent_audit_passes": (
            "On gene-disjoint route examples, edited-A acceptor-window value changes for the "
            "-3 T-over-A and +1 G-over-A effects produced donor-side lift through L3H1 above "
            "same-transition cross-gene and final-output-norm-matched direction controls."
            if all_supported else
            "No joint L3H1 boundary-payload route wording is licensed by this result."
        ),
        "estimand": {
            "primary": "linear G-vs-mean(A,C,T) raw lift after edited-A L3H1 payload recovery",
            "secondary": "max-competitor raw lift and ratio-of-means recovery",
            "formula": "o_edit + sum_ACC4 A_edit(donor,j) W_V [r3_ref(j)-r3_edit(j)]",
            "held_fixed": "edited post-softmax L3H1 attention row",
            "native_attention_sensitivity_run": False,
        },
        "config": {
            "design": expected_config,
            "split": args.split,
            "device": device,
            "n_genes": len(target_examples),
            "minimum_route_genes": bg.MIN_BG2_TARGET_GENES,
            "minimum_route_genes_per_strand": bg.MIN_BG2_TARGET_GENES_PER_STRAND,
            "batch": args.batch,
            "n_boot": N_BOOT_DEFAULT,
            "acc4_biological_positions": list(ACC4_BIOLOGICAL_POSITIONS),
            "acc4_array_offsets": [-3, -2, -1, 0],
            "directed_pairs": list(DIRECTED_PAIRS),
            "signed_sham_seeds": list(SIGNED_SHAM_SEEDS),
            "cross_gene_strata": ["directed_pair", "strand", "partner_span_060_104_or_105_150"],
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "schema": preflight["schema"],
            "PASS": preflight["PASS"],
            "checkpoint_sha256": preflight["checkpoint_sha256"],
            "model_file_sha256": preflight["model_file_sha256"],
            "gtf_sha256": preflight["gtf_sha256"],
            "fasta_sha256": preflight["fasta_sha256"],
            "protocol_sha256": preflight["protocol_sha256"],
            "declaration_file_sha256": preflight["declaration_sha256"],
            "declaration_payload_sha256": bg.declaration_sha256(),
            "train_cohort_sha256": preflight["cohort"]["sha256"],
        },
        "behavior_authorization": {
            "path": str(behavior_path),
            "sha256": sha256_file(behavior_path),
            "manifest_path": str(behavior_manifest_path),
            "manifest_sha256": sha256_file(behavior_manifest_path),
            "audit_path": str(behavior_audit_path),
            "audit_sha256": sha256_file(behavior_audit_path),
            "audit_schema": behavior_audit["schema"],
            "audit_PASS": behavior_audit["PASS"],
            "schema": behavior["schema"],
            "status": behavior["status"],
            "verdict": behavior["target"]["verdict"],
            "primary_ids": behavior["target"]["primary_ids"],
            "preflight_sha256": behavior["preflight_sha256"],
        },
        "cohort": {
            "split": args.split,
            **target_cohort,
            "preflight_membership": frozen_target_membership,
            "data_sha256": frozen_target_membership["sha256"],
            "strands": frozen_target_membership["strands"],
            "counts": {
                "route": len(target_examples),
                "route_plus_strand": sum(item.strand == "+" for item in target_examples),
                "route_minus_strand": sum(item.strand == "-" for item in target_examples),
            },
        },
        "diagnostics": diagnostics,
        "summaries": summaries,
        "constructs": constructs,
        "cells": cells,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_coupling:bg2_boundary_payload_recovery_v1",
        sys.argv,
        repo=REPO,
        status="VERIFICATION",
        config={
            **result["config"],
            "target_cohort_sha256": result["cohort"]["sha256"],
            "target_cohort_data_sha256": result["cohort"]["data_sha256"],
            "preflight_sha256": result["preflight"]["sha256"],
            "behavior_result_sha256": result["behavior_authorization"]["sha256"],
            "behavior_manifest_sha256": result["behavior_authorization"]["manifest_sha256"],
            "behavior_audit_sha256": result["behavior_authorization"]["audit_sha256"],
        },
        seeds={
            "bootstrap": BOOT_SEED,
            "signed_coordinate_first": SIGNED_SHAM_SEEDS[0],
            "signed_coordinate_count": len(SIGNED_SHAM_SEEDS),
        },
        notes=[
            "BG1 passed non-smoke preflight was hash-validated before target execution.",
            "Cross-gene and signed-coordinate null families are separately recorded and never pooled.",
            "After an independent BG2 audit, the strongest positive claim is model-functional route sufficiency, not U2AF1 identity.",
        ],
    )
    model_path = _resolved(args.model)
    model_inputs = (
        [model_path] if model_path.is_file()
        else [model_path / name for name in MODEL_FILES]
    )
    for path in (
        *model_inputs,
        gtf_path,
        fasta_path,
        preflight_path,
        behavior_path,
        behavior_manifest_path,
        behavior_audit_path,
        protocol_path,
        declaration_path,
        Path(__file__),
        REPO / "scripts/splicebert/audit_bg_boundary.py",
        REPO / "src/gfmcd/splicing/coupling.py",
        REPO / "src/gfmcd/splicing/circuit.py",
        REPO / "src/gfmcd/splicing/payload.py",
        REPO / "src/gfmcd/circuits/hf_bert.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps({
        "verdict": result["verdict"],
        "summaries": summaries,
        "result": str(out),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
