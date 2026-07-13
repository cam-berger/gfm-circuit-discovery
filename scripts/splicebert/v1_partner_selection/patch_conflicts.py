#!/usr/bin/env python
"""Activation-patch clean SpliceBERT head outputs into aligned corrupt runs.

This script establishes mediation for a *registered corruption*.  Endpoint-mask
and exon-core controls are useful instrumentation and context-dependence tests,
but they are not by themselves proof of partner-specific computation.  The
report carries that limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.hf_bert import (
    capture_head_outputs,
    head_patch_scan,
    patch_heads,
)
from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.context_controls import (
    mask_true_partner_neighborhood,
    retain_site_windows,
)
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.perturb import shuffle_exon_core
from gfmcd.splicing.readout import collate_pair_indices, true_pair_margin
from gfmcd.splicing.training import load_pair_checkpoint


def parse_heads(path: Path | None) -> list[tuple[int, int]]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    rows = data.get("selected_heads", data)
    return [(int(x["layer"]), int(x["head"])) for x in rows]


def corrupt_sequences(examples, kind: str, flank: int, seed: int) -> list[str]:
    out = []
    for i, example in enumerate(examples):
        if kind == "acceptor-mask":
            out.append(mask_true_partner_neighborhood(example, kind="acceptor", flank=flank))
        elif kind == "donor-mask":
            out.append(mask_true_partner_neighborhood(example, kind="donor", flank=flank))
        elif kind == "site-windows-only":
            out.append(retain_site_windows(example, flank=flank, include_all_candidates=True))
        elif kind == "exon-core-shuffle":
            sequence, _ = shuffle_exon_core(example, protect=flank, seed=seed + i)
            out.append(sequence)
        else:  # pragma: no cover
            raise ValueError(kind)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--examples",
        default="results/splicebert_partner_selection/baselines/val_local_conflicts.jsonl",
    )
    p.add_argument(
        "--corruption",
        choices=("acceptor-mask", "donor-mask", "site-windows-only", "exon-core-shuffle"),
        required=True,
    )
    p.add_argument("--selected-head-report", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-examples", type=int, default=16)
    p.add_argument("--flank", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-gap", type=float, default=0.05)
    p.add_argument("--token-patch", choices=("all", "endpoints"), default="all")
    p.add_argument("--allow-secondary-regime", action="store_true")
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/circuits/patching")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    examples_path = resolve_repo_path(repo, args.examples)
    examples = load_jsonl(examples_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if len(examples) < 2 or any(x.split != "val" for x in examples):
        raise ValueError("patch discovery requires at least two validation examples")

    model_cfg = cfg["model"]
    adapter = SpliceBertAdapter.from_pretrained(
        resolve_repo_path(repo, model_cfg["path"]),
        tokenizer_mode=model_cfg.get("tokenizer_mode", "auto"),
        device=args.device,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        model_class=model_cfg.get("model_class", "masked_lm"),
    )
    checkpoint_path = resolve_repo_path(repo, args.checkpoint)
    readout, checkpoint = load_pair_checkpoint(checkpoint_path, adapter=adapter)
    regime = checkpoint.get("extra_metadata", {}).get("regime", "unknown")
    if regime != "additive-frozen" and not args.allow_secondary_regime:
        raise SystemExit(f"refusing secondary trace target {regime!r} without explicit override")

    clean_sequences = [x.sequence for x in examples]
    corrupt = corrupt_sequences(examples, args.corruption, args.flank, args.seed)
    clean_tokens = adapter.tokenize(clean_sequences)
    corrupt_tokens = adapter.tokenize(corrupt)
    if clean_tokens.input_ids.shape != corrupt_tokens.input_ids.shape:
        raise RuntimeError("clean/corrupt token shapes are not aligned")
    pair_batch = collate_pair_indices(
        examples,
        token_offset=clean_tokens.prefix_tokens,
        device=adapter.device,
    )
    representation_layer = int(cfg["readout"]["representation_layer"])

    @torch.no_grad()
    def score(tokenized) -> float:
        outputs = adapter.forward(
            tokenized,
            output_hidden_states=True,
            output_attentions=False,
            grad=False,
        )
        logits = readout(outputs.hidden_states[representation_layer], pair_batch)
        return float(true_pair_margin(logits, pair_batch).mean())

    clean_score = score(clean_tokens)
    corrupt_score = score(corrupt_tokens)
    gap = clean_score - corrupt_score
    if gap < args.min_gap:
        raise RuntimeError(
            f"clean-corrupt gap {gap:.6f} is below --min-gap {args.min_gap}; "
            "do not interpret patch recovery on a behaviorally ineffective corruption"
        )

    source_cache = capture_head_outputs(adapter, lambda: score(clean_tokens))

    token_mask = None
    if args.token_patch == "endpoints":
        token_mask = torch.zeros_like(corrupt_tokens.attention_mask, dtype=torch.bool)
        for b, example in enumerate(examples):
            for pos in (example.true_pair.acceptor_pos, example.true_pair.donor_pos):
                lo = max(0, pos - args.flank) + corrupt_tokens.prefix_tokens
                hi = min(len(example.sequence), pos + args.flank) + corrupt_tokens.prefix_tokens
                token_mask[b, lo:hi] = True

    patch_results = head_patch_scan(
        adapter,
        clean_score=clean_score,
        corrupt_score=corrupt_score,
        source_cache=source_cache,
        corrupt_score_fn=lambda: score(corrupt_tokens),
        token_mask=token_mask,
    )
    selected_report = (
        resolve_repo_path(repo, args.selected_head_report) if args.selected_head_report else None
    )
    selected = parse_heads(selected_report)
    selected_result = None
    if selected:
        with patch_heads(adapter, source_cache, selected, token_mask=token_mask):
            patched = score(corrupt_tokens)
        selected_result = {
            "heads": selected,
            "patched_score": patched,
            "recovery_fraction": (patched - corrupt_score) / gap,
        }

    run_name = checkpoint_path.parent.parent.name + "_" + checkpoint_path.parent.name
    out_dir = resolve_repo_path(repo, args.out_dir) / run_name / args.corruption
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "regime": regime,
        "corruption": args.corruption,
        "token_patch": args.token_patch,
        "n_examples": len(examples),
        "clean_score": clean_score,
        "corrupt_score": corrupt_score,
        "clean_corrupt_gap": gap,
        "single_head_patch_results": [x.to_dict() for x in patch_results],
        "selected_set_patch": selected_result,
        "claim_scope": (
            "MEDIATION_OF_REGISTERED_CONTEXT/SITE_CORRUPTION; NOT YET PARTNER-SPECIFIC CIRCUIT PROOF"
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "corrupt_sequences.jsonl").write_text(
        "".join(
            json.dumps({"example_id": x.example_id, "sequence": sequence}) + "\n"
            for x, sequence in zip(examples, corrupt)
        )
    )

    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:patch:{run_name}:{args.corruption}",
        sys.argv,
        repo=repo,
        config=cfg,
        seeds={"corruption": args.seed},
        notes=[
            "Clean and corrupt token coordinates are aligned.",
            "The script refuses to patch when the behavioral gap is below the registered minimum.",
            "Endpoint/context corruptions are instrumentation, not sufficient partner-specific evidence.",
        ],
    )
    manifest.add_input(examples_path)
    manifest.add_input(checkpoint_path)
    if selected_report:
        manifest.add_input(selected_report)
    for path in out_dir.iterdir():
        if path.is_file():
            manifest.add_output(path)
    manifest.write(out_dir / "run_manifest.json")
    print(json.dumps({k: v for k, v in report.items() if k != "single_head_patch_results"}, indent=2))


if __name__ == "__main__":
    main()
