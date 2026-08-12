# DNABERT-2 Checkpoint Provenance

> **Tier: T2 — provenance.** Instrument identity record for the `DBP*` study. Supports `DBP1`; asserts no
> mechanism. Claims are in [`studies/dnabert_promoter/CLAIMS.md`](../../../studies/dnabert_promoter/CLAIMS.md).

## Checkpoints

- Original: `zhihan1996/DNABERT-2-117M`
- Original remote-code revision observed: `7bce263b15377fc15361f52cfab88f8b586abda0`
- Experimental checkpoint: `quietflamingo/dnabert2-no-flashattention`
- Pinned experimental revision: `813031b2bf86d9e960a027e2734c908009f31601`

## Verified equivalence

The two repositories matched on architecture configuration, tokenizer metadata,
special-token IDs, and representative tokenization outputs.

After excluding the unused, randomly initialized pooler tensors, all pretrained
encoder weights matched exactly:

- encoder tensors compared: 136
- encoder parameters compared: 116,477,952
- missing or extra keys: 0
- shape mismatches: 0
- value mismatches: 0
- maximum absolute parameter difference: 0.0

## Runtime limitation

The original repository's historical Triton FlashAttention code could not run
in the current environment because it calls `tl.dot(..., trans_b=True)`, while
the installed Triton version no longer accepts `trans_b`.

This is a runtime compatibility issue, not a checkpoint discrepancy.

## Conclusion

The frozen experiments use the original DNABERT-2 tokenizer and exact pretrained
encoder weights through the compatible no-flash implementation. The same pinned
checkpoint will be used as the starting point for self-fine-tuning.

Direct hidden-state equality with the historical FlashAttention implementation
was not measured because that implementation cannot compile in this environment.
