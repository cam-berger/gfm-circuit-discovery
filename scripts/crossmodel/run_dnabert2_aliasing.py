#!/usr/bin/env python
"""DNABERT-2 TOKENIZER-ALIASING CONTRAST (Phase 2, step 5).

This is a DELIBERATE METHODOLOGICAL CONTRAST, not a convergence test.  The point is to show
WHY single-nucleotide tokenization is necessary to express (and therefore to see) a
codon-scale positional code -- a null here is a TOKENIZER artifact, never evidence that the
code is absent from the sequence.

TWO FINDINGS, BOTH REPORTED.

  (1) DNABERT-2 (zhihan1996/DNABERT-2-117M) HAS NO LEARNED POSITION TABLE.  Its MosaicBERT
      backbone uses ALiBi (attention linear biases): position enters as a FIXED, non-learned
      slope added to attention scores, not as a trainable P matrix.  The config's
      ``position_embedding_type="absolute"`` is vestigial -- the state dict contains
      word_embeddings and token_type_embeddings but NO ``position_embeddings`` key.  So the
      carrier's first defining object (a learned P) does not exist here to FFT.

  (2) EVEN IF IT DID, the BPE tokenizer makes a nucleotide-wavelength period unrepresentable
      in token-position space.  A period-T (nt) carrier would need to appear at period T/(nt
      per token) in TOKEN index.  DNABERT-2's BPE tokens span a mean of ~4.6 nt with high
      variance (1-8 nt) and content-dependent boundaries, so:
        * a codon (3 nt) is < 1 token -> below the token-index Nyquist; not representable;
        * the token->nt map is non-uniform and sequence-dependent, so any fixed token-index
          period aliases to a smeared distribution of nt periods, not a line.
      We quantify the nt-per-token distribution and show the codon period falls below the
      representable band.

CONCLUSION FRAME.  The absence of a codon-scale positional code in DNABERT-2 is fully
explained by tokenization (no single-nt grid, and in fact no learned P at all).  It says
nothing about whether the genome carries such a code -- only that this tokenizer cannot
express one.  Single-nt models (SpliceBERT, RNA-FM, RNA-BERT) are the only place the
question is even well posed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HF_ID = "zhihan1996/DNABERT-2-117M"
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/origin-crossmodel/canonical"))
    ap.add_argument("--nseq", type=int, default=200)
    ap.add_argument("--seqlen", type=int, default=300)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()

    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(HF_ID, trust_remote_code=True)
    wpath = hf_hub_download(HF_ID, "pytorch_model.bin")
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    keys = list(sd.keys())
    pos_keys = [k for k in keys if k.endswith("position_embeddings.weight")]
    has_learned_P = len(pos_keys) > 0

    # (2) tokenizer aliasing quantification
    tok = AutoTokenizer.from_pretrained(HF_ID, trust_remote_code=True)
    rng = np.random.default_rng(SEED)
    seqs = ["".join(rng.choice(list("ACGT"), args.seqlen)) for _ in range(args.nseq)]
    tok_lens = []
    tokens_per_seq = []
    for s in seqs:
        enc = tok(s, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(enc["input_ids"])
        tok_lens.extend(len(t) for t in toks)
        tokens_per_seq.append(len(toks))
    tok_lens = np.array(tok_lens)
    mean_nt = float(tok_lens.mean())

    vals, cnts = np.unique(tok_lens, return_counts=True)
    dist = {int(v): float(c / len(tok_lens)) for v, c in zip(vals, cnts)}

    # codon period in token-index units, and the token-index Nyquist
    codon_tokens = 3.0 / mean_nt              # < 1 -> below Nyquist (0.5 cycles/token max)
    p1063_tokens = (1026.0 / 96.0) / mean_nt  # =10.63 nt period in token index

    payload = {
        "assay": "dnabert2_tokenizer_aliasing",
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Methodological contrast: why single-nt tokenization is required to express a "
                "codon-scale positional code. A null here is a tokenizer artifact.",
        "model": {
            "hf_id": HF_ID,
            "config_position_embedding_type": getattr(cfg, "position_embedding_type", None),
            "config_alibi_starting_size": getattr(cfg, "alibi_starting_size", None),
            "state_dict_n_keys": len(keys),
            "position_embeddings_keys": pos_keys,
            "has_learned_position_table": bool(has_learned_P),
            "embedding_keys_present": [k for k in keys if "embeddings" in k and "weight" in k],
            "verdict": ("NO learned position table (ALiBi backbone) -- the carrier's first "
                        "defining object does not exist here" if not has_learned_P
                        else "learned position table present"),
        },
        "tokenizer": {
            "class": type(tok).__name__,
            "vocab_size": int(getattr(cfg, "vocab_size", tok.vocab_size)),
            "type": "BPE (~4k), multi-nt tokens",
            "nt_per_token_mean": mean_nt,
            "nt_per_token_median": float(np.median(tok_lens)),
            "nt_per_token_std": float(tok_lens.std()),
            "nt_per_token_min": int(tok_lens.min()),
            "nt_per_token_max": int(tok_lens.max()),
            "nt_per_token_distribution": dist,
            "n_tokens_sampled": int(len(tok_lens)),
        },
        "aliasing": {
            "codon_period_nt": 3.0,
            "codon_period_in_token_index": codon_tokens,
            "token_index_nyquist_period": 2.0,   # need period >= 2 token-steps to resolve
            "codon_representable_in_token_index": bool(codon_tokens >= 2.0),
            "carrier_10p63_period_nt": 1026.0 / 96.0,
            "carrier_10p63_period_in_token_index": p1063_tokens,
            "carrier_10p63_representable": bool(p1063_tokens >= 2.0),
            "explanation": ("A period-T(nt) carrier needs period T/(nt-per-token) in TOKEN "
                            "index. Codon (3nt) -> %.2f token-steps, below the Nyquist period "
                            "of 2 token-steps: NOT representable. Non-uniform, content-dependent "
                            "token boundaries further smear any fixed token-period across nt "
                            "periods." % codon_tokens),
        },
    }
    out = outdir / "dnabert2_aliasing.json"
    out.write_text(json.dumps(payload, indent=1))
    (outdir / "dnabert2_aliasing.manifest.json").write_text(json.dumps({
        "artifact": "dnabert2_aliasing.json", "assay": "dnabert2_tokenizer_aliasing",
        "produced_utc": payload["produced_utc"], "git_rev": git_rev,
        "environment": "gfmcd-mm (conda): python3.11 torch transformers einops",
        "model": {"hf_id": HF_ID,
                  "weights_sha256": hashlib.sha256(open(wpath, "rb").read()).hexdigest()},
        "params": {"nseq": args.nseq, "seqlen": args.seqlen}, "seed": SEED,
        "command": "python scripts/crossmodel/run_dnabert2_aliasing.py",
    }, indent=1))
    print(json.dumps(payload, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
