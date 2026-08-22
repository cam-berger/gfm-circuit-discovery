"""SpliceBERT loading, device selection, and the masked-AG readout.

Tokenization is single-nucleotide (multimolecule's RnaTokenizer), so a raw
sequence of length L tokenizes to L+2 tokens (<cls> + L bases + <eos>), and
raw index i maps 1:1 to token index i+1. Everything below relies on that.
"""

import torch
from multimolecule import RnaTokenizer, SpliceBertForMaskedLM


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


class SpliceBert:
    """Thin wrapper: load once, reuse for every masked-margin query."""

    def __init__(self, checkpoint: str, device_cfg: str = "auto", revision: str = None):
        self.device = resolve_device(device_cfg)
        self.checkpoint = checkpoint
        self.revision = revision
        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = RnaTokenizer.from_pretrained(checkpoint, **kwargs)
        self.model = SpliceBertForMaskedLM.from_pretrained(checkpoint, **kwargs)
        self.model.to(self.device)
        self.model.eval()

        # The corruption alphabet in `acceptors.data` is DNA; this tokenizer is
        # expected to fold T onto U so clean and corrupted bases are comparable.
        # Assert it rather than relying on it silently.
        assert getattr(self.tokenizer, "replace_T_with_U", False), (
            f"{checkpoint} tokenizer does not declare replace_T_with_U; the DNA "
            "corruption alphabet in acceptors.data would tokenize to <unk>."
        )

        self.mask_id = self.tokenizer.mask_token_id
        self.base_ids = {
            b: self.tokenizer.convert_tokens_to_ids(b) for b in "ACGU"
        }

    @torch.no_grad()
    def logits_for(self, sequence: str) -> torch.Tensor:
        """Return per-position vocab logits, shape [seq_len_tokens, vocab]."""
        enc = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        return out.logits[0]

    @torch.no_grad()
    def masked_logits_at(self, sequence: str, raw_positions) -> torch.Tensor:
        """Mask every position in `raw_positions` (0-indexed into `sequence`)
        simultaneously and return logits at those (token-index) positions,
        shape [len(raw_positions), vocab]."""
        enc = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"].clone()
        token_positions = [p + 1 for p in raw_positions]
        for tp in token_positions:
            input_ids[0, tp] = self.mask_id
        enc["input_ids"] = input_ids
        out = self.model(**enc)
        return out.logits[0, token_positions]

    @torch.no_grad()
    def masked_attentions_at(self, sequence: str, raw_positions):
        """Mask every position in `raw_positions` simultaneously (same
        convention as `masked_logits_at`) and return the full attention
        stack from that forward pass, plus the token-index positions that
        were masked (for use as attention query indices downstream).

        Returns (attentions, token_positions): attentions is a tuple of
        per-layer tensors [num_heads, seq_tokens, seq_tokens] (batch
        dimension dropped, one entry per transformer layer)."""
        enc = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"].clone()
        token_positions = [p + 1 for p in raw_positions]
        for tp in token_positions:
            input_ids[0, tp] = self.mask_id
        enc["input_ids"] = input_ids
        out = self.model(**enc, output_attentions=True)
        attentions = tuple(a[0] for a in out.attentions)
        return attentions, token_positions
