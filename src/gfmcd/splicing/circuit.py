"""THE FROZEN CIRCUIT -- one definition, imported by every script that measures it.

WHY THIS MODULE EXISTS
----------------------
Every ``run_*.py`` in this study re-implemented its own ``enc_of``, ``metric``, example
filter and bootstrap.  That duplication is not a style problem, it is the *mechanism* by
which the two-base patch-window bug survived review: the correct window existed in one
file and the wrong one in another, and nothing forced them to agree.

From here on the circuit, the readout, the corruption, the eligibility filter and every
intervention live HERE.  The exploratory scripts and the sealed-test script import the
same functions, so a sealed-test number cannot differ from a val number for any reason
other than the data.

TWO SUBTLETIES THAT CHANGE WHAT THE NUMBERS MEAN
------------------------------------------------
1. ``min_effect``.  Recovery fractions are ratios, and a ratio needs a denominator that is
   not ~0, so the mediation experiments select examples whose baseline coupling is >= 0.3.
   That filter CONDITIONS ON THE OUTCOME.  It is legitimate for a *ratio* and illegitimate
   for a *behavioural mean*: the same held-out data gives coupling +1.359 with the filter
   and a much smaller number without it.  ``load_examples(min_effect=None)`` returns the
   unconditional set; the behavioural claim must use it.  Any conditional number must be
   labelled conditional.

2. The corruption sits at the AG, i.e. ``[a-2, a-1]`` -- never at ``a`` (the first exon
   base).  Every patch window is validated against the corruption mask by
   ``gfmcd.circuits.guards``.  A patch that does not overlap the corruption cannot repair
   it, and the resulting near-zero "recovery" reads as a scientific null.  It is not one.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
try:
    # transformers >= 4.48 exposes the custom-attention registry used by ``roll_attention``
    # (the period-3 clock study).  The project pins ``transformers>=4.40``, so guard it: the
    # registry is unrelated to every other assay and its absence must not brick module import.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:                                     # pragma: no cover - version-dependent
    ALL_ATTENTION_FUNCTIONS = None

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.guards import (
    PatchWindow, acceptor_motif_positions, adjacent_window_controls, validate_all,
)
from gfmcd.circuits.hf_bert import capture_head_outputs, edit_heads, patch_heads
from gfmcd.splicing.coupling import (
    ACCEPTOR_ALTS, CouplingConfig, CouplingExample, build_coupling_dataset, stable_rng,
    substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome

# --------------------------------------------------------------------------------------
# THE FROZEN CIRCUIT.  Selected on train, confirmed on val.  Not revisable by test data.
# --------------------------------------------------------------------------------------
READER: tuple[int, int] = (3, 1)       # L3H1  -- the content-addressed read
ADDRESSER: tuple[int, int] = (2, 15)   # L2H15 -- builds the acceptor's address (key)
CIRCUIT: tuple[tuple[int, int], ...] = (READER, ADDRESSER)

CORRUPT_ALT: str = ACCEPTOR_ALTS[0]    # "CC" -- destroys AG, cannot create AG

# Eligibility: an internal exon whose intron-mate acceptor is in-window and whose
# donor is a canonical GT, at a separation the 1024nt context can actually hold.
GAP_MIN, GAP_MAX = 60, 150
MIN_EFFECT = 0.3                       # ONLY for ratio denominators.  See module docstring.

# Supply via --gtf/--fasta or the GFMCD_GTF/GFMCD_FASTA environment variables (REPRODUCE.md §2).
# These resolve to a non-existent path rather than None so that `Path(DEFAULT_GTF)` still type-checks:
# the data-gated memorials import these constants directly and pass them to `_dataset(...)`, and a None
# here raises TypeError inside pathlib instead of letting the caller's `skipif`/existence check decide.
DEFAULT_GTF: str = os.environ.get("GFMCD_GTF", "/nonexistent/gencode.gtf.gz")
DEFAULT_FASTA: str = os.environ.get("GFMCD_FASTA", "/nonexistent/hg38.fa")
DEFAULT_MODEL = "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_adapter(model_path: str, device: str, *, eager: bool = False) -> SpliceBertAdapter:
    """``eager=True`` is REQUIRED whenever attention weights are read.

    SDPA silently ignores ``output_attentions=True`` and returns all-zero attention, which
    is indistinguishable from a real null until you notice every one of 96 heads reports
    exactly 0.0000.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    repo = Path(__file__).resolve().parents[3]
    mp = model_path if Path(model_path).is_absolute() else str(repo / model_path)
    kw = {"attn_implementation": "eager"} if eager else {}
    return SpliceBertAdapter(
        AutoModelForMaskedLM.from_pretrained(mp, **kw),
        AutoTokenizer.from_pretrained(mp),
        device=device,
    )


# --------------------------------------------------------------------------------------
# The behaviour: native masked-LM readout of the donor.
# --------------------------------------------------------------------------------------
class CouplingTask:
    """Mask the G of the donor GT; score logit(G) - max(logit A,C,T).

    ``coupling`` = margin(clean) - margin(partner-acceptor destroyed).  No probe, no
    readout head, no labels -- so there is no task to win and nothing to leak.
    """

    def __init__(self, adapter: SpliceBertAdapter, batch: int = 8):
        self.ad = adapter
        self.batch = batch
        self.mask_id = adapter.tokenizer.mask_token_id
        self.bid = {b: adapter.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        self.off = adapter.prefix_tokens

    def encode(self, seqs: Sequence[str], donors: Sequence[int]):
        enc = self.ad.tokenizer([" ".join(s) for s in seqs], return_tensors="pt",
                                padding=True).to(self.ad.device)
        for j, d in enumerate(donors):
            enc["input_ids"][j, d + self.off] = self.mask_id
        return enc

    def margin(self, logits: torch.Tensor, donors: Sequence[int]) -> torch.Tensor:
        i = torch.arange(len(donors), device=logits.device)
        p = torch.tensor([d + self.off for d in donors], device=logits.device)
        v = logits[i, p]
        alt = torch.stack([v[:, self.bid[b]] for b in "ACT"], -1).max(-1).values
        return v[:, self.bid["G"]] - alt

    def pair(self, exs: Sequence[CouplingExample]):
        """(clean_enc, corrupt_enc, donor_positions) for one batch."""
        dn = [x.donor_pos for x in exs]
        clean = self.encode([x.sequence for x in exs], dn)
        corrupt = self.encode(
            [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in exs], dn)
        return clean, corrupt, dn

    def coupling(self, exs: Sequence[CouplingExample],
                 ctx: Callable[[], Iterator] = None) -> np.ndarray:
        """Per-example coupling under an optional intervention context factory."""
        ctx = ctx or noop
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(exs), self.batch):
                cl, co, dn = self.pair(exs[i:i + self.batch])
                with ctx():
                    mc = self.margin(self.ad.model(**cl).logits.float(), dn)
                    mk = self.margin(self.ad.model(**co).logits.float(), dn)
                out += (mc - mk).tolist()
        return np.asarray(out)

    def margins(self, exs: Sequence[CouplingExample],
                ctx: Callable[[], Iterator] = None) -> tuple[np.ndarray, np.ndarray]:
        """(clean_margin, corrupt_margin) per example -- for the 2x2 factorial."""
        ctx = ctx or noop
        C: list[float] = []
        K: list[float] = []
        with torch.no_grad():
            for i in range(0, len(exs), self.batch):
                cl, co, dn = self.pair(exs[i:i + self.batch])
                with ctx():
                    C += self.margin(self.ad.model(**cl).logits.float(), dn).tolist()
                    K += self.margin(self.ad.model(**co).logits.float(), dn).tolist()
        return np.asarray(C), np.asarray(K)

    def mlm_health(self, exs: Sequence[CouplingExample], ctx: Callable[[], Iterator] = None,
                   n_probe: int = 40, seed: int = 0) -> float:
        """Top-1 masked-prediction accuracy at random NON-site positions.

        A circuit claim requires the coupling to die while general MLM competence SURVIVES.
        Without this control, "ablation killed the effect" is indistinguishable from
        "ablation killed the model".

        THE PROBE POSITIONS MUST BE MASKED.  An earlier version of this control read the
        argmax at random *unmasked* positions and compared it to the base sitting right
        there in the input -- so it scored the model's ability to COPY its own input.  It
        returned ~0.99 for everything, including a model with 94 of 96 heads ablated (0.998,
        *above* baseline).  It could not have failed, so it never licensed the claim it was
        cited for.  Masking the probes is what makes this a measurement.
        """
        ctx = ctx or noop
        r = stable_rng(f"health-{seed}", 0)
        acc: list[float] = []
        with torch.no_grad():
            for i in range(0, len(exs), self.batch):
                ch = exs[i:i + self.batch]
                dn = [x.donor_pos for x in ch]
                enc = self.encode([x.sequence for x in ch], dn)
                probes = []
                for j, x in enumerate(ch):
                    ps = sorted(r.sample(range(10, len(x.sequence) - 10), n_probe))
                    # never probe the donor (already masked) or the partner AG
                    ps = [p for p in ps if abs(p - x.donor_pos) > 2
                          and abs(p - (x.next_acceptor_pos or -99)) > 3]
                    for p in ps:
                        enc["input_ids"][j, p + self.off] = self.mask_id
                    probes.append(ps)
                with ctx():
                    lg = self.ad.model(**enc).logits.float()
                for j, x in enumerate(ch):
                    if not probes[j]:
                        continue
                    pred = lg[j, [p + self.off for p in probes[j]]].argmax(-1)
                    tgt = torch.tensor([self.bid[x.sequence[p]] for p in probes[j]],
                                       device=pred.device)
                    acc.append((pred == tgt).float().mean().item())
        return float(np.mean(acc))


# --------------------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------------------
def eligible(examples: Iterable[CouplingExample], split: str) -> list[CouplingExample]:
    return [x for x in examples
            if x.split == split
            and x.next_acceptor_pos is not None
            and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
            and GAP_MIN <= (x.next_acceptor_pos - x.donor_pos) <= GAP_MAX]


_DATASET_CACHE: dict = {}


def _dataset(gtf: str, fasta: str) -> list[CouplingExample]:
    """Memoised.  Parsing the 80MB GENCODE GTF takes ~40s and every script did it twice."""
    key = (gtf, fasta)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = build_coupling_dataset(gtf, PyfaidxGenome(fasta), CouplingConfig())
    return _DATASET_CACHE[key]


def load_examples(
    gtf: str, fasta: str, split: str, n: int, *, task: CouplingTask, tag: str,
    min_effect: float | None = MIN_EFFECT, pool: int = 6,
) -> tuple[list[CouplingExample], np.ndarray]:
    """Load ``n`` eligible examples for ``split``.

    ``min_effect=None``  -> UNCONDITIONAL.  Use this for the behavioural claim.
    ``min_effect=0.3``   -> conditional on a usable ratio denominator.  Use this, and say
                            so, for recovery/rescue fractions.

    ``pool`` bounds how many candidates are screened, so the conditional set is drawn
    from a defined superset rather than "however far we had to search".
    """
    cfg = CouplingConfig()
    X = eligible(_dataset(gtf, fasta), split)
    stable_rng(tag, cfg.seed).shuffle(X)
    if min_effect is None:
        keep = X[:n]
        return keep, task.coupling(keep)

    keep: list[CouplingExample] = []
    eff: list[float] = []
    screened = 0
    with torch.no_grad():
        for i in range(0, min(len(X), n * pool), task.batch):
            ch = X[i:i + task.batch]
            cl, co, dn = task.pair(ch)
            e = (task.margin(task.ad.model(**cl).logits.float(), dn)
                 - task.margin(task.ad.model(**co).logits.float(), dn))
            screened += len(ch)
            for j, x in enumerate(ch):
                if float(e[j]) >= min_effect:
                    keep.append(x)
                    eff.append(float(e[j]))
            if len(keep) >= n:
                break
    if len(keep) < n:
        raise RuntimeError(
            f"only {len(keep)} of {screened} screened {split} examples clear "
            f"min_effect={min_effect}; asked for {n}.  Raise --pool or lower n rather "
            "than silently analysing a smaller set.")
    return keep[:n], np.asarray(eff[:n])


# --------------------------------------------------------------------------------------
# Statistics.  Gene-grouped throughout: exons of one gene are not independent.
# --------------------------------------------------------------------------------------
def gboot(vals, genes, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    by: dict = {}
    for v, g in zip(vals, genes):
        if not np.isnan(v):
            by.setdefault(g, []).append(v)
    k = list(by)
    if not k:
        return (float("nan"), float("nan"))
    m = np.empty(n)
    for i in range(n):
        p = rng.choice(len(k), len(k), replace=True)
        m[i] = np.mean([v for j in p for v in by[k[j]]])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def gboot_ratio(num, dnm, genes, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """CI for a RATIO OF MEANS.  Never mean-of-ratios: near-zero denominators make the
    per-example ratio explode (this produced a 'recovery' of -14.079 once)."""
    rng = np.random.default_rng(seed)
    by: dict = {}
    for a, b, g in zip(num, dnm, genes):
        by.setdefault(g, []).append((a, b))
    k = list(by)
    out = np.empty(n)
    for i in range(n):
        p = rng.choice(len(k), len(k), replace=True)
        pr = [t for j in p for t in by[k[j]]]
        out[i] = sum(t[0] for t in pr) / max(sum(t[1] for t in pr), 1e-9)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def ratio_of_means(num, dnm) -> float:
    return float(np.sum(num) / max(np.sum(dnm), 1e-9))


# --------------------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------------------
@contextmanager
def noop() -> Iterator[None]:
    yield


def ablate(ad: SpliceBertAdapter, heads: Iterable[tuple[int, int]], mode: str = "mean"):
    """Factory (not a context) so it can be passed where a ``ctx()`` is expected."""
    hs = list(heads)
    return lambda: edit_heads(ad, hs, mode=mode)


def complement(ad: SpliceBertAdapter,
               keep: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    k = set(keep)
    return [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)
            if (l, h) not in k]


# --------------------------------------------------------------------------------------
# ROLL THE ROUTING, KEEP THE VALUES.  (the codon-phase-lattice intervention)
# --------------------------------------------------------------------------------------
# Ablation could not answer whether the lattice is USED: deleting 42 of 96 heads drives
# SpliceBERT to chance-level nucleotide prediction (log-prob -1.377 vs ln(1/4) = -1.386),
# so the behaviour is gone along with the mechanism and the metric measures nothing (R13).
#
# This intervention leaves the model intact.  For a chosen head it rolls the POST-SOFTMAX
# attention probabilities along the KEY axis by ``delta`` and lets the head collect the
# very same value vectors it always collected -- just from ``delta`` nt off target.
# W_Q, W_K, W_V, W_O, every other head and every MLP are untouched.  The head still emits a
# convex combination of REAL value vectors with ITS OWN weight profile; it is simply aimed
# off.  That is the cleanest available dissociation of *where a head looks* (QK) from *what
# it collects* (OV) -- and the shuffle test only ever constrained the former.
#
# READ THIS BEFORE USING IT FOR A CLAIM.  The perturbation SIZE is itself phase-dependent,
# by construction, and not for any interesting reason:
#
#     out_i^d - out_i^0  =  sum_k  a_k ( v_{i+k+d} - v_{i+k} )
#     E|| . ||^2         =  sum_t  w(t) [ 2C(t) - C(t+d) - C(t-d) ]
#
# with C = autocovariance of the value vectors along the sequence and w(t) = sum_k a_k
# a_{k+t} the autocorrelation of the head's OWN attention profile.  For a phase-pure head w
# is supported on t = 0 (mod 3) -- exactly the lags at which C is periodic in coding
# sequence.  So a phase-pure head, rolled off-phase, takes a BIGGER hit than the same head
# rolled by 3, *whether or not anything downstream uses phase at all*.  The lattice heads
# were SELECTED for phase purity, so raw damage will sawtooth for them, in exons, and not
# for random heads or in introns -- reproducing every naive gate with no codon computation
# anywhere in the model.
#
# Therefore: NEVER read raw damage(delta) as evidence of function.  Divide it by the
# perturbation actually delivered (``capture_head_outputs`` before and after gives
# ||delta-out||) and ask whether the model is hurt MORE THAN THE SIZE OF THE PERTURBATION
# EXPLAINS.  ``run_phase_shift.py`` does this and says so in its own docstring.

_ROLL: dict[int, tuple[list[int], int, int, int]] = {}


def _rolled_attention_forward(module, query, key, value, attention_mask,
                              scaling=None, dropout=0.0, **kw):
    """``eager_attention_forward``, plus a roll of the probabilities for selected heads."""
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    w = torch.nn.functional.softmax(w, dim=-1)
    w = torch.nn.functional.dropout(w, p=dropout, training=module.training)

    spec = _ROLL.get(getattr(module, "layer_idx", -1))
    if spec is not None:
        heads, delta, lo, hi = spec
        w = w.clone()
        orig = w[:, heads, :, lo:hi]                     # nucleotide columns only
        blk = torch.roll(orig, shifts=delta, dims=-1)
        if delta > 0:                                    # kill the wrap-around
            blk[..., :delta] = 0.0
        elif delta < 0:
            blk[..., delta:] = 0.0

        # KEEP EACH QUERY'S READ OF ITSELF EXACTLY AS IT WAS.
        # Without this the roll silently does two things at once.  (1) It slides a comb tooth
        # onto the query's OWN column -- which, at a masked position, is the [MASK] token.  For
        # a comb whose teeth are all ≡ r (mod 3) that happens at delta ≡ -r (mod 3), i.e. at ONE
        # phase class -- a mod-3-correlated information loss that inflates the sawtooth for free
        # (L1H10's comb {-7,-4,-1,2,5} lands a tooth on 0 at delta = 1, 4, 7).  (2) It rolls the
        # query's self-attention AWAY, a loss the design wrongly assumed was delta-independent.
        # Restoring the diagonal removes both.  The intervention is now exactly:
        #     "the head keeps reading ITSELF as before; only its reads of OTHER positions shift."
        q = torch.arange(lo, hi, device=w.device)
        blk[:, :, q, q - lo] = orig[:, :, q, q - lo]

        w[:, heads, :, lo:hi] = blk
        # CLS/SEP/PAD columns keep their original mass; renormalise the row to a convex
        # combination again.  At delta = 0 this whole path runs and must be the identity.
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)

    out = torch.matmul(w, value).transpose(1, 2).contiguous()
    return out, w


if ALL_ATTENTION_FUNCTIONS is not None:
    ALL_ATTENTION_FUNCTIONS.register("gfmcd_roll", _rolled_attention_forward)


@contextmanager
def roll_attention(ad: SpliceBertAdapter, heads: Iterable[tuple[int, int]], delta: int,
                   *, seq_len: int, off: int = 1) -> Iterator[None]:
    """Roll ``heads``' attention along the key axis by ``delta``.  Values untouched.

    ``seq_len`` is the number of NUCLEOTIDE columns; ``off`` where they start (1, past CLS).
    Special-token columns are held in place and the row is renormalised.

    ``delta = 0`` deliberately runs the FULL code path (roll by zero, zero the wrap,
    renormalise) rather than short-circuiting, so the calibration tests the machinery
    instead of testing that doing nothing does nothing.  It must reproduce the unhooked
    model to numerical precision; ``tests/test_roll_identity.py`` asserts it.
    """
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError(
            "roll_attention needs the custom-attention registry (transformers>=4.48); "
            f"installed transformers lacks ALL_ATTENTION_FUNCTIONS. Upgrade to run the period-3 "
            "clock study; every other assay works on the pinned transformers>=4.40.")
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    prev_impl = ad.model.config._attn_implementation
    _ROLL.clear()
    for l, hs in by_layer.items():
        _ROLL[l] = (sorted(hs), int(delta), off, off + seq_len)
    ad.model.config._attn_implementation = "gfmcd_roll"
    try:
        yield
    finally:
        _ROLL.clear()
        ad.model.config._attn_implementation = prev_impl


@contextmanager
def patch_proj(ad: SpliceBertAdapter, layer: int, head: int, which: str,
               source: torch.Tensor, positions: torch.Tensor) -> Iterator[None]:
    """Patch one head's query/key/value at token window ``positions`` [B,K].

    BERT packs heads as contiguous slices of the projection output, so head ``h`` owns
    ``[h*Dh : (h+1)*Dh]``.
    """
    mod = getattr(ad.attention_self_modules[layer], which)
    Dh = ad.hidden_size // ad.num_heads
    lo, hi = head * Dh, (head + 1) * Dh

    def hook(_m, _i, out):
        out = out.clone()
        pos = positions if positions.dim() == 2 else positions[:, None]
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(pos)
        out[b, pos, lo:hi] = source[b, pos, lo:hi].to(out.dtype)
        return out

    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def relocate_proj(ad: SpliceBertAdapter, layer: int, head: int, which: str,
                  source: torch.Tensor, src_pos: torch.Tensor,
                  dst_pos: torch.Tensor) -> Iterator[None]:
    """COPY one head's key/value FROM ``src_pos`` in ``source`` INTO ``dst_pos`` of this run.

    NEVER SWAP.  Transposing the (key, value) PAIR between two positions is a relabelling of
    the summation index in ``out_i = sum_j softmax(q_i . k_j) v_j``: the multiset of (k, v)
    pairs is unchanged, the softmax denominator is unchanged, and the head's output is
    therefore IDENTICAL -- exactly, for any weights.  We shipped such a swap as a
    "competing-acceptor double dissociation" and it reproduced on a RANDOMLY INITIALISED
    model (|both-swap - baseline| = 1.2e-06).  It measured nothing.

    A one-way copy, sourced from a DIFFERENT forward pass (clean -> corrupt), genuinely
    changes the multiset.  It can fail, which is what makes it an experiment:

        key  only @ decoy  -> the head is routed to the decoy but reads the DECOY's content
        value only @ decoy -> the partner's content sits somewhere the head does not look
        key + value @ decoy -> the head is routed to the decoy AND finds the PARTNER there

    If the third restores the coupling and neither of the first two does, the address and the
    content are separable and the read is content-addressed rather than position-addressed.
    """
    mod = getattr(ad.attention_self_modules[layer], which)
    Dh = ad.hidden_size // ad.num_heads
    lo, hi = head * Dh, (head + 1) * Dh
    if src_pos.shape != dst_pos.shape:
        raise ValueError("source and destination windows must have the same width")

    def hook(_m, _i, out):
        out = out.clone()
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(src_pos)
        out[b, dst_pos, lo:hi] = source[b, src_pos, lo:hi].to(out.dtype)
        return out

    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def matched_decoy_donor(x) -> int | None:
    """A DIFFERENT GT at a separation from the partner comparable to the true donor's.

    ``other_gt[0]`` is the LEFTMOST GT in the 1024nt window -- median 389nt from the
    corrupted AG against 98nt for the real donor.  Using it as the specificity comparator
    confounds "the model tracks ITS OWN partner" with plain attention distance-decay, and
    the effect it measures is ~0, so a ratio against it divides by noise.
    """
    pt, d = x.next_acceptor_pos, x.donor_pos
    target = pt - d
    cands = [g for g in x.other_gt if g < pt - 10 and abs(g - d) > 10]
    return min(cands, key=lambda g: abs((pt - g) - target)) if cands else None


def matched_decoy_acceptor(x, tol: int = 40) -> int | None:
    """A non-partner AG at a matched distance from the donor (the binding-arm decoy)."""
    d, pt = x.donor_pos, x.next_acceptor_pos
    dist = pt - d
    cands = [p for p in x.other_ag
             if p > d + 20 and abs((p - d) - dist) <= tol and abs(p - pt) > 12
             and p + 1 < len(x.sequence) and p - 3 > 0]
    return min(cands, key=lambda p: abs((p - d) - dist)) if cands else None


def _block_write(ad: SpliceBertAdapter, layer: int, kind: str):
    """The module whose OUTPUT is a block's write into the residual, BEFORE add+LayerNorm.

    HF BERT:  attention write = layer.attention.output.dense(context)
              MLP write       = layer.output.dense(intermediate)
    Clamping one of these freezes that block's *contribution* while leaving the residual
    stream (which carries an upstream patch) flowing through the add.
    """
    L = ad.encoder_layers[layer]
    if kind == "attn":
        return L.attention.output.dense
    if kind == "mlp":
        return L.output.dense
    raise ValueError(f"unknown block kind: {kind!r}")


@contextmanager
def clamp_write(ad: SpliceBertAdapter, blocks: Iterable[tuple[int, str]],
                source: dict, positions: torch.Tensor | None = None) -> Iterator[None]:
    """Force each block's residual write to a cached value (usually the CORRUPT run's).

    This is the freeze in path patching: with every downstream block clamped, whatever
    recovery survives an upstream patch reached the readout through the residual stream
    alone -- i.e. it is a DIRECT effect.
    """
    handles = []

    def make(key):
        def hook(_m, _i, out):
            src = source[key].to(device=out.device, dtype=out.dtype)
            if positions is None:
                return src.clone()
            out = out.clone()
            pos = positions if positions.dim() == 2 else positions[:, None]
            b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(pos)
            out[b, pos] = src[b, pos]
            return out
        return hook

    try:
        for layer, kind in blocks:
            mod = _block_write(ad, layer, kind)
            handles.append(mod.register_forward_hook(make((layer, kind))))
        yield
    finally:
        for h in handles:
            h.remove()


def capture_writes(ad: SpliceBertAdapter, blocks: Iterable[tuple[int, str]],
                   run: Callable[[], object]) -> dict:
    """Cache each block's residual write from one forward pass."""
    store: dict = {}
    handles = []

    def make(key):
        def hook(_m, _i, out):
            store[key] = out.detach().clone()
        return hook

    try:
        for layer, kind in blocks:
            handles.append(_block_write(ad, layer, kind).register_forward_hook(make((layer, kind))))
        run()
    finally:
        for h in handles:
            h.remove()
    return store


def capture_proj(ad: SpliceBertAdapter, layer: int, which: Iterable[str],
                 run: Callable[[], object]) -> dict:
    """Cache a layer's query/key/value projection outputs from one forward pass."""
    store: dict = {}
    handles = []
    for w in which:
        mod = getattr(ad.attention_self_modules[layer], w)
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, w=w: store.__setitem__(w, o.detach().clone())))
    try:
        run()
    finally:
        for h in handles:
            h.remove()
    return store


def downstream_blocks(ad: SpliceBertAdapter, after_layer: int) -> list[tuple[int, str]]:
    """Every residual write strictly AFTER the attention block of ``after_layer``.

    Starts with that layer's own MLP -- clamping ``after_layer``'s ATTENTION write would
    overwrite the very patch whose effect we are tracing.
    """
    out: list[tuple[int, str]] = [(after_layer, "mlp")]
    for l in range(after_layer + 1, ad.num_layers):
        out += [(l, "attn"), (l, "mlp")]
    return out


# --------------------------------------------------------------------------------------
# Patch windows.  Always validated against the corruption; the indices go into provenance.
# --------------------------------------------------------------------------------------
ACC_WINDOW_OFFSETS = (-3, -2, -1, 0)   # covers the AG at [a-2, a-1], plus one base either side


@dataclass(frozen=True)
class Windows:
    acceptor: torch.Tensor      # [B,4] the AG window -- OVERLAPS the corruption
    donor: torch.Tensor         # [B,1] the masked donor
    control: torch.Tensor       # [B,4] distance-matched non-site -- MISSES the corruption
    provenance: list[dict]


def build_windows(exs: Sequence[CouplingExample], off: int, device) -> Windows:
    acc = torch.tensor([[x.next_acceptor_pos + off + k for k in ACC_WINDOW_OFFSETS]
                        for x in exs], device=device)
    don = torch.tensor([[x.donor_pos + off] for x in exs], device=device)
    ctl = torch.tensor([[max(4, x.donor_pos + off - (x.next_acceptor_pos - x.donor_pos)) + k
                         for k in ACC_WINDOW_OFFSETS] for x in exs], device=device)
    corr = acceptor_motif_positions(exs[0].next_acceptor_pos, off)
    wins = [
        PatchWindow("acceptor", tuple(int(v) for v in acc[0]), corr),
        PatchWindow("control-pos", tuple(int(v) for v in ctl[0]), corr, is_control=True),
    ]
    wins += adjacent_window_controls(tuple(int(v) for v in acc[0]), corr)
    return Windows(acc, don, ctl, validate_all(wins))


__all__ = [
    "ACC_WINDOW_OFFSETS", "ADDRESSER", "CIRCUIT", "CORRUPT_ALT", "DEFAULT_FASTA",
    "DEFAULT_GTF", "DEFAULT_MODEL", "GAP_MAX", "GAP_MIN", "MIN_EFFECT", "READER",
    "CouplingTask", "Windows", "ablate", "build_windows", "capture_head_outputs",
    "capture_proj", "capture_writes", "clamp_write", "complement", "downstream_blocks",
    "eligible", "gboot", "gboot_ratio", "load_adapter", "load_examples",
    "matched_decoy_acceptor", "matched_decoy_donor", "noop", "patch_heads", "patch_proj",
    "pick_device", "ratio_of_means", "relocate_proj", "roll_attention",
]
