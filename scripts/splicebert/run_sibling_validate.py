#!/usr/bin/env python
"""ORIGIN-SIBLINGS STEP 1 — validate the three checkpoints load and are parity-matched.

Three checkpoints, on disk (no download):
  SpliceBERT.1024nt        FINAL   (72-species, stage-2), P=(1026,512)
  SpliceBERT.510nt         stage-1 (72-species),          P=(512,512)
  SpliceBERT-human.510nt   stage-1 HUMAN-ONLY,            P=(512,512)

This script asserts architecture parity (6L / 16H / hidden 512 / vocab 10 / absolute
learned P present), single-nucleotide tokenisation (one token per nt), and records the
config SHA-256 + weight SHA-256 (full + wtsha=first16) for each. The only INTENDED
differences are corpus (72-species vs human) and position cap (1026 vs 512).

Writes an immutable JSON + sibling manifest. No model surgery, no spectrum here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import load_adapter  # noqa: E402

ARCHIVE = REPO / "data/models/splicebert_official_archive/extracted/models"
CKPTS = {
    "final_1024": {"path": ARCHIVE / "SpliceBERT.1024nt",
                   "role": "FINAL 72-species stage-2", "exp_wtsha": "2ad91428c318e6c4",
                   "exp_maxpos": 1026},
    "stage1_510": {"path": ARCHIVE / "SpliceBERT.510nt",
                   "role": "stage-1 72-species", "exp_wtsha": "b3a48c768a029dc6",
                   "exp_maxpos": 512},
    "human_510": {"path": ARCHIVE / "SpliceBERT-human.510nt",
                  "role": "stage-1 HUMAN-ONLY", "exp_wtsha": "7d9e7b6d53f2e42e",
                  "exp_maxpos": 512},
}
OUT = REPO / "results/origin-siblings/canonical/sibling_validate.json"


def validate_one(key: str, spec: dict) -> dict:
    path = spec["path"]
    cfg_sha = sha256_file(path / "config.json")
    wt_full = sha256_file(path / "pytorch_model.bin")
    wtsha = wt_full[:16]

    ad = load_adapter(str(path), "cpu")
    m = ad.model
    cfg = m.config

    # absolute learned position table
    emb = m.bert.embeddings.position_embeddings
    P = emb.weight.detach().float().cpu()
    Pshape = list(P.shape)

    # single-nt tokenisation: one token per nucleotide. Encode a known string.
    probe = "ACGTACGTAC"  # 10 nt
    tok = ad.tokenize(probe)
    # count non-special tokens == 10 ; adapter reports prefix_tokens (CLS etc.)
    n_ids = int(tok.input_ids.shape[-1])
    prefix = int(ad.prefix_tokens)
    n_core = n_ids - prefix - 1  # minus CLS-side prefix and trailing SEP
    single_nt = ad._is_single_nt_mode(ad.tokenizer_mode)

    rec = {
        "role": spec["role"],
        "path": str(path.relative_to(REPO)),
        "config_sha256": cfg_sha,
        "weight_sha256": wt_full,
        "wtsha": wtsha,
        "wtsha_expected": spec["exp_wtsha"],
        "wtsha_match": wtsha == spec["exp_wtsha"],
        "architectures": list(cfg.architectures),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "hidden_size": int(cfg.hidden_size),
        "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": int(cfg.max_position_embeddings),
        "max_position_expected": spec["exp_maxpos"],
        "position_embedding_type": str(cfg.position_embedding_type),
        "P_shape": Pshape,
        "adapter_num_layers": int(ad.num_layers),
        "adapter_num_heads": int(ad.num_heads),
        "adapter_hidden_size": int(ad.hidden_size),
        "tokenizer_mode": ad.tokenizer_mode,
        "single_nt": bool(single_nt),
        "probe_len_nt": len(probe),
        "probe_core_tokens": n_core,
        "prefix_tokens": prefix,
    }
    # hard assertions — parity with the final checkpoint's architecture
    assert rec["wtsha_match"], f"{key}: wtsha mismatch {wtsha} != {spec['exp_wtsha']}"
    assert rec["num_hidden_layers"] == 6, f"{key}: layers {rec['num_hidden_layers']}"
    assert rec["num_attention_heads"] == 16, f"{key}: heads {rec['num_attention_heads']}"
    assert rec["hidden_size"] == 512, f"{key}: hidden {rec['hidden_size']}"
    assert rec["vocab_size"] == 10, f"{key}: vocab {rec['vocab_size']}"
    assert rec["position_embedding_type"] == "absolute", f"{key}: not absolute P"
    assert rec["max_position_embeddings"] == spec["exp_maxpos"], f"{key}: maxpos"
    assert Pshape == [spec["exp_maxpos"], 512], f"{key}: P shape {Pshape}"
    assert single_nt, f"{key}: tokenizer not single-nt"
    assert n_core == len(probe), f"{key}: {n_core} core tokens for {len(probe)} nt"
    return rec


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, spec in CKPTS.items():
        print(f"validating {key} ({spec['role']}) ...")
        results[key] = validate_one(key, spec)
        r = results[key]
        print(f"  wtsha {r['wtsha']} match={r['wtsha_match']}  "
              f"P{r['P_shape']}  {r['num_hidden_layers']}L/{r['num_attention_heads']}H/"
              f"{r['hidden_size']}  single_nt={r['single_nt']}")

    # parity cross-check: siblings share architecture with final; differ only in maxpos
    arch_keys = ("num_hidden_layers", "num_attention_heads", "hidden_size",
                 "vocab_size", "position_embedding_type")
    fin = results["final_1024"]
    parity = {k: all(results[c][k] == fin[k] for c in results) for k in arch_keys}
    corpus_diff = {"stage1_510": "72-species", "human_510": "human-only",
                   "final_1024": "72-species"}
    maxpos = {k: results[k]["max_position_embeddings"] for k in results}

    record = {
        "experiment": "origin-siblings-validate-v1",
        "checkpoints": results,
        "architecture_parity_all_three": parity,
        "architecture_parity_holds": all(parity.values()),
        "corpus_by_checkpoint": corpus_diff,
        "maxpos_by_checkpoint": maxpos,
        "intended_differences": {
            "corpus": "stage1_510(72sp) vs human_510(human-only) — CORPUS contrast",
            "maxpos_and_stage": "stage1_510(512,stage1) vs final_1024(1026,stage2) — "
                                "STAGE contrast (length+stage confounded)",
        },
    }
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nwrote {OUT.relative_to(REPO)}")

    man = ExperimentManifest.create(
        "origin-siblings-validate-v1",
        ["scripts/splicebert/run_sibling_validate.py"],
        repo=REPO,
        status="CANONICAL",
        config={"checkpoints": {k: str(v["path"].relative_to(REPO))
                                for k, v in CKPTS.items()}},
    )
    for spec in CKPTS.values():
        man.add_input(spec["path"] / "pytorch_model.bin")
        man.add_input(spec["path"] / "config.json")
    man.add_output(OUT)
    (OUT.with_suffix(".manifest.json")).write_text(
        json.dumps(man.to_dict(), indent=2, sort_keys=True))
    print(f"wrote {OUT.with_suffix('.manifest.json').relative_to(REPO)}")
    print(f"\nARCH PARITY (all three): {record['architecture_parity_holds']}")


if __name__ == "__main__":
    main()
