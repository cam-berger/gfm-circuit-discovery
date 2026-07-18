#!/usr/bin/env python3
"""Export measured SpliceBERT screening graphs for jointly masked nucleotide segments.

This is deliberately a component-level adaptation of attribution graphs, not a
feature-level reproduction of the cross-layer-transcoder method.  It uses one
shared MLM forward per masked segment, local activation×Jacobian scores to rank
attention OV routes and MLP writes, and explicit remainder nodes.  Every edge is
labelled ``screening``; the output must not be read as a validated causal circuit.

The default input is one deterministic, unconditional GENCODE validation example.
It exports the donor GT and its intron-mate acceptor AG as two separate joint-mask
conditions, with a focused graph for each nucleotide in each segment.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from gfmcd.circuits.hf_bert import _split_context
from gfmcd.provenance import sha256_file
from gfmcd.splicing.circuit import (
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    _dataset,
    eligible,
    load_adapter,
    pick_device,
)
from gfmcd.splicing.coupling import stable_rng


BASES = "ACGT"


class ForwardCapture(AbstractContextManager):
    """Capture differentiable component activations from one eager-attention pass."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.word_embedding: torch.Tensor | None = None
        self.attention_inputs: dict[int, torch.Tensor] = {}
        self.contexts: dict[int, torch.Tensor] = {}
        self.attentions: dict[int, torch.Tensor] = {}
        self.mlp_outputs: dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        embeddings = self.adapter.model.bert.embeddings.word_embeddings

        def embedding_hook(_module, _inputs, output):
            if not output.requires_grad:
                output.requires_grad_(True)
            self.word_embedding = output
            return output

        self.handles.append(embeddings.register_forward_hook(embedding_hook))
        for layer, block in enumerate(self.adapter.encoder_layers):
            self.handles.append(
                self.adapter.attention_self_modules[layer].register_forward_hook(
                    self._attention_hook(layer)
                )
            )
            self.handles.append(
                block.output.dense.register_forward_hook(self._mlp_hook(layer))
            )
        return self

    def _attention_hook(self, layer: int):
        def hook(_module, inputs, output):
            context = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("eager attention must return post-softmax weights")
            self.attention_inputs[layer] = inputs[0]
            self.contexts[layer] = context
            self.attentions[layer] = output[1]
            return output

        return hook

    def _mlp_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.mlp_outputs[layer] = output
            return output

        return hook

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--split", default="val")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--heads-per-position", type=int, default=2)
    parser.add_argument("--sources-per-head", type=int, default=3)
    parser.add_argument("--max-frontier", type=int, default=6)
    parser.add_argument(
        "--output",
        default="results/splicebert_visualizer/segment_attribution_bundle.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.heads_per_position, args.sources_per_head, args.max_frontier) < 1:
        raise ValueError("graph expansion limits must all be positive")
    torch.manual_seed(0)
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)

    examples = eligible(_dataset(args.gtf, args.fasta), args.split)
    stable_rng("segment-attribution-v1", 0).shuffle(examples)
    if not 0 <= args.example_index < len(examples):
        raise IndexError(f"example-index {args.example_index} outside {len(examples)} examples")
    example = examples[args.example_index]
    if example.next_acceptor_pos is None:
        raise RuntimeError("selected example has no intron-mate acceptor")

    segments = [
        ("donor-gt", example.donor_pos, example.donor_pos + 2),
        ("partner-acceptor-ag", example.next_acceptor_pos - 2, example.next_acceptor_pos),
    ]
    graphs: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    for segment_id, start, end in segments:
        segment = run_segment(
            adapter,
            example.sequence,
            segment_id,
            start,
            end,
            heads_per_position=args.heads_per_position,
            sources_per_head=args.sources_per_head,
            max_frontier=args.max_frontier,
        )
        graphs.extend(segment["graphs"])
        traces.extend(segment["traces"])
        gate_records.append(segment["gates"])

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path
    script_sha256 = sha256_file(__file__)
    gtf_sha256 = sha256_file(args.gtf)
    fasta_fai = Path(f"{args.fasta}.fai")
    fasta_fai_sha256 = sha256_file(fasta_fai) if fasta_fai.exists() else "missing"
    bundle = {
        "schemaVersion": "0.1.0",
        "model": {
            "id": "SpliceBERT.1024nt",
            "checkpoint": sha256_file(model_path / "pytorch_model.bin"),
            "tokenizer": f"single nucleotide · {adapter.tokenizer_mode}",
            "contextLength": adapter.max_nucleotides,
        },
        "title": "Measured joint-mask splice-site attribution",
        "description": (
            "Real SpliceBERT forward/gradient measurements on one deterministic GENCODE "
            "validation example. Edges are component-level screening attributions, not a "
            "cross-layer-transcoder feature graph and not a validated causal circuit."
        ),
        "sequence": {
            "id": example.example_id,
            "label": f"{example.gene_id} · {example.chrom}:{example.metadata.get('window_start', 'window')}",
            "bases": example.sequence,
            "coordinateSystem": {
                "chromosome": example.chrom,
                "strand": example.strand,
                "modelOffset": 0,
                "indexConvention": "zero-based-half-open",
            },
        },
        "annotations": annotations_for(example),
        "carriers": [
            {
                "id": "p3",
                "label": "Period 3",
                "period": 3,
                "phaseOffset": 0,
                "amplitude": 1,
                "evidence": "Canonical geometry only; no carrier-specific QK decomposition in this export.",
            },
            {
                "id": "p1063",
                "label": "Period 10.63",
                "period": 10.63,
                "phaseOffset": 0.7,
                "amplitude": 1,
                "evidence": "Canonical geometry only; no carrier-specific QK decomposition in this export.",
            },
        ],
        "conditions": [
            {
                "id": "joint-segment-mask",
                "label": "Joint segment mask",
                "description": "Every nucleotide in the declared segment is [MASK] in one shared forward.",
            }
        ],
        "traces": traces,
        "attributionGraphs": graphs,
        "provenance": {
            "source": "scripts/splicebert/export_segment_attribution_bundle.py",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": git_revision(),
            "git_dirty": str(git_dirty()),
            "exporter_sha256": script_sha256,
            "gtf": str(args.gtf),
            "gtf_sha256": gtf_sha256,
            "fasta": str(args.fasta),
            "fasta_fai_sha256": fasta_fai_sha256,
            "example_selection": (
                f"eligible({args.split}); deterministic shuffle tag segment-attribution-v1; "
                f"index {args.example_index}; no outcome filter"
            ),
            "attribution_semantics": (
                "activation×local-Jacobian screening; attention edges are fixed-pattern OV routes; "
                "MLP nodes are whole-block outputs; all edges require exact patch validation"
            ),
            "correctness_gates": json.dumps(gate_records, sort_keys=True),
        },
    }
    bundle = round_floats(bundle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.write_text(json.dumps(bundle, indent=2) + "\n")
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": subprocess.list2cmdline([sys.executable, __file__, *sys.argv[1:]]),
                "output": str(output),
                "output_sha256": sha256_file(output),
                "checkpoint_sha256": sha256_file(model_path / "pytorch_model.bin"),
                "git_revision": git_revision(),
                "git_dirty": git_dirty(),
                "exporter_sha256": script_sha256,
                "device": device,
                "data": {
                    "gtf": str(args.gtf),
                    "gtf_sha256": gtf_sha256,
                    "fasta": str(args.fasta),
                    "fasta_fai_sha256": fasta_fai_sha256,
                    "example_id": example.example_id,
                    "selection_seed_tag": "segment-attribution-v1",
                    "example_index": args.example_index,
                },
                "gates": gate_records,
                "claim_boundary": "Exploratory measured screening graph; no causal or biological claim.",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), "graphs": len(graphs), "traces": len(traces), "gates": gate_records}, indent=2))


def run_segment(
    adapter,
    sequence: str,
    segment_id: str,
    start: int,
    end: int,
    *,
    heads_per_position: int,
    sources_per_head: int,
    max_frontier: int,
) -> dict[str, Any]:
    if not (0 <= start < end <= len(sequence)):
        raise ValueError(f"invalid segment {start}:{end}")
    batch = adapter.tokenize(sequence)
    ids = batch.input_ids.clone()
    token_start = adapter.nucleotide_to_token(start)
    token_end = adapter.nucleotide_to_token(end)
    ids[:, token_start:token_end] = adapter.tokenizer.mask_token_id

    with ForwardCapture(adapter) as capture, torch.enable_grad():
        raw = adapter.model(
            input_ids=ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        if capture.word_embedding is None:
            raise RuntimeError("word embedding hook did not fire")
        hidden = raw.hidden_states[-1]
        base_ids = torch.tensor(
            [adapter.tokenizer.convert_tokens_to_ids(base) for base in BASES],
            device=hidden.device,
        )
        predicted = []
        for position in range(start, end):
            token = adapter.nucleotide_to_token(position)
            predicted.append(BASES[int(raw.logits[0, token, base_ids].argmax())])

        gates = correctness_gates(adapter, raw, capture)
        graphs = []
        traces = []
        for focus in range(start, end):
            token = adapter.nucleotide_to_token(focus)
            base_logits = raw.logits[0, token, base_ids].float()
            log_probs = base_logits.log_softmax(-1)
            probabilities = base_logits.softmax(-1)
            native_index = BASES.index(sequence[focus])
            predicted_index = int(probabilities.argmax())
            alternative_index = max(
                (index for index in range(len(BASES)) if index != native_index),
                key=lambda index: float(probabilities[index].detach()),
            )
            metric_tensors = {
                "target-log-probability": log_probs[native_index],
                "output-entropy": -(probabilities * log_probs).sum(),
            }
            activations = [
                capture.word_embedding,
                *[capture.contexts[layer] for layer in range(adapter.num_layers)],
                *[capture.mlp_outputs[layer] for layer in range(adapter.num_layers)],
                hidden,
            ]
            gradients = {}
            for metric_id, metric in metric_tensors.items():
                values = torch.autograd.grad(
                    metric,
                    activations,
                    retain_graph=True,
                    allow_unused=False,
                )
                gradients[metric_id] = {
                    "word": values[0],
                    "contexts": {layer: values[1 + layer] for layer in range(adapter.num_layers)},
                    "mlps": {
                        layer: values[1 + adapter.num_layers + layer]
                        for layer in range(adapter.num_layers)
                    },
                    "hidden": values[-1],
                }
            metrics = [
                {
                    "id": "target-log-probability",
                    "label": "Native-base log-probability",
                    "definition": f"log p({sequence[focus]} at {focus} | x[{start}:{end}]=[MASK]) over A/C/G/T",
                    "value": float(metric_tensors["target-log-probability"].detach()),
                    "unit": "nats",
                    "category": "token-evidence",
                    "note": "Native-base reconstruction evidence under one joint segment mask.",
                },
                {
                    "id": "output-entropy",
                    "label": "Output entropy",
                    "definition": f"−Σ p(base) log p(base) at {focus} under the joint segment mask",
                    "value": float(metric_tensors["output-entropy"].detach()),
                    "unit": "nats",
                    "category": "uncertainty",
                    "note": "Positive effects increase uncertainty; negative effects sharpen the four-base distribution.",
                },
            ]
            graphs.append(
                build_graph(
                    adapter,
                    sequence,
                    segment_id,
                    start,
                    end,
                    focus,
                    "".join(predicted),
                    sequence[focus],
                    BASES[predicted_index],
                    BASES[alternative_index],
                    metrics,
                    capture,
                    gradients,
                    hidden,
                    heads_per_position=heads_per_position,
                    sources_per_head=sources_per_head,
                    max_frontier=max_frontier,
                )
            )
            traces.extend(attention_traces(adapter, sequence, segment_id, start, end, focus, capture))
    return {"graphs": graphs, "traces": traces, "gates": {"segment": segment_id, **gates}}


def build_graph(
    adapter,
    sequence: str,
    segment_id: str,
    start: int,
    end: int,
    focus: int,
    predicted_bases: str,
    native_base: str,
    predicted_base: str,
    alternative_base: str,
    metrics: list[dict[str, Any]],
    capture: ForwardCapture,
    gradients: dict[str, dict[str, Any]],
    final_hidden: torch.Tensor,
    *,
    heads_per_position: int,
    sources_per_head: int,
    max_frontier: int,
) -> dict[str, Any]:
    prefix = f"{segment_id}-{start}-{end}-q{focus}"
    root = f"{prefix}-metric"
    nodes: dict[str, dict[str, Any]] = {
        root: {
            "id": root,
            "kind": "metric",
            "label": f"{native_base}@{focus}",
            "detail": f"Native-base evidence; model predicts {predicted_base} under the shared joint mask.",
            "position": focus,
        }
    }
    edges: list[dict[str, Any]] = []
    metric_ids = [metric["id"] for metric in metrics]

    def effects(values: dict[str, float]) -> dict[str, dict[str, float]]:
        return {metric_id: {"screeningAttribution": float(values[metric_id])} for metric_id in metric_ids}

    def add_edge(edge_id, source, target, values, route, *, status="screening", attention=None, ov_norm=None):
        edge = {
            "id": f"{prefix}-{edge_id}",
            "sourceNodeId": source,
            "targetNodeId": target,
            "routeKind": route,
            "screeningAttribution": float(values["target-log-probability"]),
            "metricEffects": effects(values),
            "validationStatus": status,
        }
        if attention is not None:
            edge["attentionWeight"] = attention
        if ov_norm is not None:
            edge["ovWriteNorm"] = ov_norm
        edges.append(edge)

    state = lambda layer, position: f"{prefix}-r{layer}-p{position}"
    nodes[state(adapter.num_layers, focus)] = residual_node(state(adapter.num_layers, focus), adapter.num_layers, focus)
    root_values = {
        metric_id: dot(
            final_hidden[0, adapter.nucleotide_to_token(focus)],
            gradients[metric_id]["hidden"][0, adapter.nucleotide_to_token(focus)],
        )
        for metric_id in metric_ids
    }
    add_edge("readout", state(adapter.num_layers, focus), root, root_values, "projects-through")

    frontier = [focus]
    for layer in reversed(range(adapter.num_layers)):
        next_scores = {position: float("inf") for position in frontier}
        planned: list[tuple[int, int, list[int]]] = []
        for query_position in frontier:
            head_scores = [
                abs(head_effect(adapter, capture, gradients, "target-log-probability", layer, head, query_position))
                for head in range(adapter.num_heads)
            ]
            selected_heads = sorted(range(adapter.num_heads), key=lambda head: (-head_scores[head], head))[:heads_per_position]
            for head in selected_heads:
                source_scores = [
                    abs(route_effect(adapter, capture, gradients, "target-log-probability", layer, head, query_position, key))
                    for key in range(len(sequence))
                ]
                selected_sources = sorted(range(len(sequence)), key=lambda key: (-source_scores[key], key))[:sources_per_head]
                planned.append((query_position, head, selected_sources))
                for key in selected_sources:
                    next_scores[key] = max(next_scores.get(key, 0.0), source_scores[key])
        required = list(dict.fromkeys(frontier))
        candidates = [position for position in sorted(next_scores, key=lambda p: (-next_scores[p], p)) if position not in required]
        next_frontier = (required + candidates)[:max_frontier]

        for position in next_frontier:
            node_id = state(layer, position)
            nodes[node_id] = residual_node(node_id, layer, position)

        for query_position, head, sources in planned:
            writer = f"{prefix}-l{layer}h{head}-q{query_position}"
            nodes[writer] = {
                "id": writer,
                "kind": "attention-head",
                "label": f"L{layer}H{head}",
                "detail": f"Measured attention context writer at query {query_position}; source edges are fixed-pattern OV attributions.",
                "layer": layer,
                "head": head,
                "position": query_position,
            }
            writer_values = {
                metric_id: head_effect(adapter, capture, gradients, metric_id, layer, head, query_position)
                for metric_id in metric_ids
            }
            add_edge(f"l{layer}h{head}-q{query_position}-write", writer, state(layer + 1, query_position), writer_values, "writes-to")
            retained = {metric_id: 0.0 for metric_id in metric_ids}
            kept_sources = [key for key in sources if key in next_frontier]
            for key in kept_sources:
                route_values = {
                    metric_id: route_effect(adapter, capture, gradients, metric_id, layer, head, query_position, key)
                    for metric_id in metric_ids
                }
                for metric_id in metric_ids:
                    retained[metric_id] += route_values[metric_id]
                attention = float(capture.attentions[layer][0, head, adapter.nucleotide_to_token(query_position), adapter.nucleotide_to_token(key)].detach())
                ov_norm = route_vector(adapter, capture, layer, head, query_position, key).norm().item()
                add_edge(
                    f"l{layer}h{head}-q{query_position}-k{key}",
                    state(layer, key),
                    writer,
                    route_values,
                    "attends-to",
                    attention=attention,
                    ov_norm=ov_norm,
                )
            remainder_values = {metric_id: writer_values[metric_id] - retained[metric_id] for metric_id in metric_ids}
            remainder = f"{writer}-remainder"
            nodes[remainder] = {
                "id": remainder,
                "kind": "remainder",
                "label": f"Other L{layer}H{head} sources",
                "detail": "Signed per-key OV attribution omitted by adaptive source pruning.",
                "layer": layer,
                "position": query_position,
            }
            add_edge(f"{writer}-remainder", remainder, writer, remainder_values, "attends-to", status="remainder")

        for query_position in frontier:
            mlp = f"{prefix}-mlp{layer}-q{query_position}"
            nodes[mlp] = {
                "id": mlp,
                "kind": "mlp",
                "label": f"MLP{layer}",
                "detail": f"Whole MLP output at position {query_position}; not a sparse feature decomposition.",
                "layer": layer,
                "position": query_position,
            }
            mlp_values = {
                metric_id: mlp_effect(adapter, capture, gradients, metric_id, layer, query_position)
                for metric_id in metric_ids
            }
            add_edge(f"mlp{layer}-q{query_position}-in", state(layer, query_position), mlp, mlp_values, "transforms")
            add_edge(f"mlp{layer}-q{query_position}-out", mlp, state(layer + 1, query_position), mlp_values, "writes-to")
        frontier = next_frontier

    word = capture.word_embedding
    assert word is not None
    for position in frontier:
        token = f"{prefix}-input-p{position}"
        masked = start <= position < end
        nodes[token] = {
            "id": token,
            "kind": "input-token",
            "label": f"{'[MASK]' if masked else sequence[position]}@{position}",
            "detail": (
                f"[MASK] embedding at model index {position}; native {sequence[position]} is withheld."
                if masked
                else f"Native {sequence[position]} embedding at model index {position}."
            ),
            "position": position,
        }
        token_values = {
            metric_id: dot(
                word[0, adapter.nucleotide_to_token(position)],
                gradients[metric_id]["word"][0, adapter.nucleotide_to_token(position)],
            )
            for metric_id in metric_ids
        }
        add_edge(f"input-p{position}", token, state(0, position), token_values, "residual")

    return {
        "id": f"{prefix}-residual-writers",
        "label": f"{segment_id} · focus {focus} · measured screening graph",
        "lens": "residual-writers",
        "targetPosition": focus,
        "targetBase": native_base,
        "alternativeBase": alternative_base,
        "metricDefinition": metrics[0]["definition"],
        "metricValue": metrics[0]["value"],
        "metricUnit": metrics[0]["unit"],
        "metrics": metrics,
        "conditionId": "joint-segment-mask",
        "referenceDescription": (
            "Local zero-activation screening within the measured joint-mask forward; attention source edges freeze the observed attention pattern and trace OV only."
        ),
        "maskedSegment": {
            "start": start,
            "end": end,
            "maskingMode": "joint-segment",
            "nativeBases": sequence[start:end],
            "predictedBases": predicted_bases,
            "aggregation": "focused-position",
            "source": "measured",
            "note": "All bases in the span were masked simultaneously; predictions are per-position marginals under that shared context.",
        },
        "rootNodeId": root,
        "nodes": list(nodes.values()),
        "edges": edges,
        "source": "measured",
        "note": (
            "Real activations and gradients, but component-level and approximate. Remainders are explicit. "
            "No edge is causally licensed until exact activation/path patching validates it."
        ),
    }


def attention_traces(adapter, sequence, segment_id, start, end, focus, capture):
    traces = []
    offset = adapter.prefix_tokens
    for layer in range(adapter.num_layers):
        module = adapter.attention_self_modules[layer]
        hidden = capture.attention_inputs[layer]
        query = module.transpose_for_scores(module.query(hidden))
        key = module.transpose_for_scores(module.key(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(adapter.head_dim)
        q = adapter.nucleotide_to_token(focus)
        for head in range(adapter.num_heads):
            traces.append(
                {
                    "id": f"{segment_id}-{start}-{end}-q{focus}-l{layer}h{head}",
                    "layer": layer,
                    "head": head,
                    "headLabel": f"L{layer}H{head}",
                    "queryIndex": focus,
                    "queryLabel": f"joint-mask focus {focus}",
                    "conditionId": "joint-segment-mask",
                    "attention": capture.attentions[layer][0, head, q, offset:offset + len(sequence)].detach().float().cpu().tolist(),
                    "totalQkLogit": scores[0, head, q, offset:offset + len(sequence)].detach().float().cpu().tolist(),
                    "carrierQkLogit": {},
                }
            )
    return traces


def correctness_gates(adapter, raw, capture):
    reconstructed_logits = adapter.model.cls(raw.hidden_states[-1])
    logit_error = float((reconstructed_logits - raw.logits).abs().max().detach())
    context_errors = []
    for layer in range(adapter.num_layers):
        module = adapter.attention_self_modules[layer]
        values = module.transpose_for_scores(module.value(capture.attention_inputs[layer]))
        reconstructed = torch.matmul(capture.attentions[layer], values)
        observed = _split_context(capture.contexts[layer], adapter.num_heads)
        context_errors.append(float((reconstructed - observed).abs().max().detach()))
    max_context_error = max(context_errors)
    attention_max = max(float(value.detach().abs().max()) for value in capture.attentions.values())
    if logit_error > 1e-5:
        raise RuntimeError(f"MLM-head reconstruction failed: {logit_error}")
    if max_context_error > 1e-5:
        raise RuntimeError(f"per-key attention context reconstruction failed: {max_context_error}")
    if attention_max <= 0:
        raise RuntimeError("attention tensors are all zero; eager attention plumbing failed")
    return {
        "mlm_head_max_abs_error": logit_error,
        "per_key_context_max_abs_error": max_context_error,
        "attention_nonzero_max": attention_max,
        "pass": True,
    }


def route_vector(adapter, capture, layer, head, query_position, key_position):
    module = adapter.attention_self_modules[layer]
    values = module.transpose_for_scores(module.value(capture.attention_inputs[layer]))
    q = adapter.nucleotide_to_token(query_position)
    k = adapter.nucleotide_to_token(key_position)
    return capture.attentions[layer][0, head, q, k] * values[0, head, k]


def head_effect(adapter, capture, gradients, metric_id, layer, head, position):
    context = _split_context(capture.contexts[layer], adapter.num_heads)
    grad = _split_context(gradients[metric_id]["contexts"][layer], adapter.num_heads)
    token = adapter.nucleotide_to_token(position)
    return dot(context[0, head, token], grad[0, head, token])


def route_effect(adapter, capture, gradients, metric_id, layer, head, query_position, key_position):
    grad = _split_context(gradients[metric_id]["contexts"][layer], adapter.num_heads)
    q = adapter.nucleotide_to_token(query_position)
    return dot(route_vector(adapter, capture, layer, head, query_position, key_position), grad[0, head, q])


def mlp_effect(adapter, capture, gradients, metric_id, layer, position):
    token = adapter.nucleotide_to_token(position)
    return dot(capture.mlp_outputs[layer][0, token], gradients[metric_id]["mlps"][layer][0, token])


def dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left * right).sum().detach())


def residual_node(node_id: str, layer: int, position: int):
    return {
        "id": node_id,
        "kind": "residual-state",
        "label": f"r{layer}@{position}",
        "detail": f"Residual state at network depth {layer}, model index {position}.",
        "layer": layer,
        "position": position,
    }


def annotations_for(example):
    annotations = [
        {
            "id": "exon",
            "type": "exon",
            "label": "Internal exon",
            "start": example.acceptor_pos,
            "end": example.donor_pos,
            "strand": example.strand,
        },
        {
            "id": "acceptor",
            "type": "acceptor",
            "label": "Exon-mate acceptor AG",
            "start": example.acceptor_pos - 2,
            "end": example.acceptor_pos,
            "strand": example.strand,
        },
        {
            "id": "donor",
            "type": "donor",
            "label": "Donor GT",
            "start": example.donor_pos,
            "end": example.donor_pos + 2,
            "strand": example.strand,
        },
        {
            "id": "partner-acceptor",
            "type": "acceptor",
            "label": "Intron-mate acceptor AG",
            "start": example.next_acceptor_pos - 2,
            "end": example.next_acceptor_pos,
            "strand": example.strand,
        },
        {
            "id": "intron",
            "type": "intron",
            "label": "Downstream intron",
            "start": example.donor_pos + 2,
            "end": example.next_acceptor_pos - 2,
            "strand": example.strand,
        },
    ]
    return [item for item in annotations if 0 <= item["start"] < item["end"] <= len(example.sequence)]


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return True


def round_floats(value: Any, digits: int = 7) -> Any:
    """Keep JSON bundles practical without changing graph ordering or decisions."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
