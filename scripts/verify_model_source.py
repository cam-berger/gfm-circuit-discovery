"""Verify a pretrained substrate's weights against its published architecture (TO_FIX.md §10.2).

Loads the model + tokenizer named in `gfmcd.config.SUBSTRATES` and checks the facts we rely on, so a
community mirror is never silently trusted. Prints metadata and PASS/FAIL per field.

    uv run --extra models python scripts/verify_model_source.py splicebert
"""

from __future__ import annotations

import argparse

from gfmcd.config import SUBSTRATES

# Published SpliceBERT facts (Chen et al.). A mirror must match these.
# NB: SpliceBERT ships two context lengths — 510nt (max_position 512) and 1024nt (max_position 1026).
# The yangheng mirror is the 510nt variant. The tokenizer is single-nucleotide but expects
# SPACE-SEPARATED input ("A C G T"); an unspaced string collapses to a single [UNK] (verified).
EXPECTED = {
    "splicebert": {
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "num_attention_heads": 16,
        "max_position_embeddings": 512,  # 510 nt + [CLS] + [SEP]
        "single_nt": True,
    }
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("substrate", choices=sorted(SUBSTRATES))
    args = ap.parse_args()

    sub = SUBSTRATES[args.substrate]
    print(f"substrate: {sub.name}  hf_id: {sub.hf_id}")
    if sub.source:
        print(f"source: {sub.source.source_type} {sub.source.source_uri}  is_official={sub.source.is_official}")
        if not sub.source.is_official:
            print("  WARNING: community mirror — verify against the official release before any published claim.")

    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(sub.hf_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(sub.hf_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(sub.hf_id, trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())

    exp = EXPECTED.get(args.substrate, {})
    checks = []
    for field in ("num_hidden_layers", "hidden_size", "num_attention_heads", "max_position_embeddings"):
        got = getattr(cfg, field, None)
        want = exp.get(field)
        ok = want is None or got == want
        checks.append(ok)
        print(f"  {field}: {got}" + (f"  [want {want}: {'PASS' if ok else 'FAIL'}]" if want is not None else ""))
    print(f"  position_embedding_type: {getattr(cfg, 'position_embedding_type', 'absolute')}")
    print(f"  parameters: {n_params/1e6:.2f}M")

    # Empirical single-nucleotide tokenization check (do not trust the flag). SpliceBERT expects
    # SPACE-SEPARATED nucleotides; the unspaced form collapses to [UNK], which is exactly the trap
    # the empirical check exists to catch.
    unspaced = tok("ACGTACGT", add_special_tokens=False)["input_ids"]
    spaced = tok(" ".join("ACGTACGT"), add_special_tokens=False)["input_ids"]
    single_nt = len(spaced) == 8
    checks.append(single_nt or not exp.get("single_nt"))
    print(f"  tokenize unspaced 'ACGTACGT' -> {len(unspaced)} tokens (expect 1 [UNK]-like: needs spacing)")
    print(f"  tokenize spaced   'A C G T…' -> {len(spaced)} tokens  [single-nt: {'PASS' if single_nt else 'FAIL'}]")
    print("  REQUIRED INPUT FORMAT: space-separated nucleotides, e.g. ' '.join(seq)")

    print("VERIFY:", "PASS" if all(checks) else "FAIL — mirror does not match the published architecture")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
