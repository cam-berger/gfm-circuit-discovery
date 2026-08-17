"""Calibration of ``roll_attention``: the delta = 0 identity, ON A PADDED BATCH TOO.

``roll_attention`` replaces the model's attention implementation with a registered custom
function, and ``delta = 0`` deliberately runs the whole code path (roll by zero, kill the
wrap, restore the diagonal, renormalise) so that the calibration tests the machinery rather
than testing that doing nothing does nothing.  Its docstring promises the delta = 0 arm
"must reproduce the unhooked model to numerical precision".

AN EQUAL-LENGTH BATCH CANNOT TEST THAT PROMISE.  transformers builds the additive attention
mask from a registry keyed on ``config._attn_implementation``; for an unrecognised name it
returns no mask at all, so the custom function was handed ``attention_mask=None`` and every
PAD column was fully visible to the softmax.  On a batch with nothing to pad the withheld
mask is the mask transformers would have skipped anyway, so the identity held to 3e-05 and
the bug was invisible.  Pad one sequence and the same identity broke by ~10 logits.

Hence: every arm below runs BOTH an equal-length and a padded batch.

The float32 floor here is ~4e-05 on logits whose scale is ~40 (measured 98 -> 1024 tokens,
and it is the sdpa/eager difference, not the roll: swapping the implementation with no head
selected moves the logits by the same amount).  ``TOL`` sits 25x above that floor and four
orders of magnitude below the defect it exists to catch.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gfmcd.splicing.circuit import load_adapter, roll_attention
from gfmcd.splicing.config import load_config

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / load_config(REPO / "configs/splicebert_partner_selection/base.json")["model"]["path"]

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="SpliceBERT weights not present"
)

TOL = 1e-3
HEAD = (3, 1)          # the partner-selection head; any head exercises the same code path
LONG, SHORT = 96, 32


@pytest.fixture(scope="module")
def adapter():
    # eager: the unhooked baseline must use the same attention math as the custom function,
    # or the identity is measured against sdpa and carries its rounding as well as its own.
    return load_adapter(str(MODEL_PATH), "cpu", eager=True)


def _seqs(lengths: list[int], seed: int = 1) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    return ["".join("ACGT"[i] for i in torch.randint(0, 4, (n,), generator=g).tolist())
            for n in lengths]


def _logits(ad, seqs, *, delta=None, seq_len=None):
    batch = ad.tokenize(seqs)
    kw = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
    if delta is None:
        with torch.no_grad():
            return ad.model(**kw).logits
    with torch.no_grad(), roll_attention(ad, [HEAD], delta, seq_len=seq_len, off=1):
        return ad.model(**kw).logits


def _nuc_width(ad, seqs) -> int:
    """The nucleotide columns of the (possibly padded) batch -- what the runner passes."""
    return ad.tokenize(seqs).input_ids.shape[1] - 2


@pytest.mark.parametrize("lengths, label", [
    ([LONG, LONG], "equal-length"),
    ([LONG, SHORT], "padded"),
    ([LONG, SHORT, 90, 12], "padded-4"),
])
def test_delta_zero_reproduces_the_unhooked_model(adapter, lengths, label):
    """The promise in the docstring, held to on a padded batch as well as an equal one."""
    seqs = _seqs(lengths)
    base = _logits(adapter, seqs)
    rolled = _logits(adapter, seqs, delta=0, seq_len=_nuc_width(adapter, seqs))
    assert (base - rolled).abs().max().item() < TOL, f"{label}: delta=0 is not the identity"


def test_padded_batch_agrees_with_the_unpadded_run(adapter):
    """A PAD column must not reach a real query.

    Independent of the delta = 0 identity: that arm compares the rolled model against the
    unhooked model on the SAME batch, so a mask both runs are missing cancels.  This arm
    compares a sequence rolled inside a padded batch against the same sequence rolled on
    its own, and nothing cancels.
    """
    seqs = _seqs([LONG, SHORT])
    short = [seqs[1]]
    batched = _logits(adapter, seqs, delta=0, seq_len=_nuc_width(adapter, seqs))
    alone = _logits(adapter, short, delta=0, seq_len=_nuc_width(adapter, short))
    real = SHORT + 1                                    # CLS + nucleotides, before SEP
    assert (batched[1, :real] - alone[0, :real]).abs().max().item() < TOL


@pytest.mark.parametrize("lengths", [[LONG, LONG], [LONG, SHORT]])
def test_a_nonzero_roll_still_moves_the_logits(adapter, lengths):
    """The other direction: a fix that restored the identity by neutering the roll fails here.

    On the padded batch this asserts only that the call is not silently inert -- the roll
    itself is refused there (see below), so what must not happen is a quiet no-op.
    """
    seqs = _seqs(lengths)
    base = _logits(adapter, seqs)
    width = _nuc_width(adapter, seqs)
    try:
        rolled = _logits(adapter, seqs, delta=3, seq_len=width)
    except ValueError:
        return                                          # refused, not silently inert
    assert (base - rolled).abs().max().item() > 1e-2


def test_a_nonzero_roll_refuses_a_padded_batch(adapter):
    """delta != 0 over a padded batch is not a well-posed intervention -- it must raise.

    The roll holds CLS/SEP/PAD in place by keeping them OUTSIDE the rolled column block.
    That works only when every sequence ends where the block ends.  Pad one and the short
    sequence's SEP and PAD columns sit INSIDE the block, so the same nominal delta is a
    different intervention per row.  The single caller
    (``scripts/splicebert/run_comb_shift.py``) feeds fixed 1024nt windows and never trips
    this; a future one must not discover it as a quiet bias.
    """
    seqs = _seqs([LONG, SHORT])
    with pytest.raises(ValueError, match="padded"):
        _logits(adapter, seqs, delta=3, seq_len=_nuc_width(adapter, seqs))
