"""The five pillars, as importable functions that return data instead of printing it.

Every measurement the scripts make lives here as a function you can call in a notebook and get an
array or a dataclass back. The heavy logic is not re-implemented -- these wrap `gfmcd.signatures`
and `gfmcd.oracle`, which already carry the corrected, house-ruled versions of each assay. What this
module adds is (a) one calling convention, (b) loss-based deltas with the decoy-Met control attached
by default, and (c) the pillar structure the study template walks through:

    1 DISCOVERY   which components look responsible          -> attention_survey, ig_head_ranking
    2 TESTING     is it a real mechanism or a shortcut?      -> shortcut_ceiling, cue_conflict_curve
    3 ABLATION    what breaks when you remove a component    -> head_ablation_table, redundancy_test
    4 BASELINES   what could a bounded-window statistic do   -> pooled_ceiling, ceiling_report
    5 CONFLICT    which cue wins when they disagree          -> mechanism_signature, single_conflict

House rules baked in (from D24 / D27, the repo's hard-won lessons):
  * ablation results carry Δloss, never accuracy alone -- accuracy saturates and lies (D27, D31);
  * start-codon ablation always reports the paired decoy-Met control (D24);
  * cue conflict, not ablation, is what separates a counter from a bag (D24);
  * attribution *ranks*; only loss-based intervention *decides* (the L1H2 lesson, D27).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd import signatures as sig
from gfmcd.data.genome import IGNORE
from gfmcd.lab.intervene import ablate_heads
from gfmcd.lab.load import LoadedModel


# ----------------------------------------------------------------------------
# shared forward helpers
# ----------------------------------------------------------------------------


@torch.no_grad()
def _run_logits(lm: LoadedModel, tokens: np.ndarray, batch: int = 64) -> torch.Tensor:
    """Full logits over a token array, batched. [n, w, out_dim]."""
    t = torch.from_numpy(tokens).long()
    out = []
    for i in range(0, len(t), batch):
        out.append(lm.model(t[i : i + batch].to(lm.device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def aa_loss_acc(lm: LoadedModel, tokens: np.ndarray, aa: np.ndarray) -> tuple[float, float]:
    """Cross-entropy and accuracy on amino-acid labels (IGNORE-masked). For the aa-head models."""
    logits = _run_logits(lm, tokens)
    y = torch.from_numpy(aa)
    m = y != IGNORE
    loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE))
    acc = float((logits.argmax(-1)[m] == y[m]).float().mean())
    return loss, acc


# ----------------------------------------------------------------------------
# PILLAR 1 -- DISCOVERY
# ----------------------------------------------------------------------------


@torch.no_grad()
def attention_survey(
    lm: LoadedModel, tokens: np.ndarray, target_pos: np.ndarray, min_dist: int = 10, batch: int = 32
) -> np.ndarray:
    """Mean attention paid by DOWNSTREAM positions to a per-row target column, per (layer, head).

    Returns [n_layers, n_heads]. `min_dist` restricts to query positions at least that far past the
    target -- the marker-following signal lives in the far field, and averaging over all queries
    dilutes it with positional heads (this is the convention of scripts/trace_uniform.py::attention_to).

    The first-pass "who looks responsible" screen. Attribution ranks; it does not decide -- a head can
    top this and be causally inert (the L1H2 lesson). Always follow with `head_ablation_table`.
    """
    model = lm.model
    L, H = lm.cfg.n_layers, lm.cfg.n_heads
    t = torch.from_numpy(tokens).long()
    acc = np.zeros((L, H))
    n_seen = 0
    for i in range(0, len(t), batch):
        chunk = t[i : i + batch].to(lm.device)
        _, cache = model.run_with_cache(chunk)
        tgt = target_pos[i : i + batch]
        for b in range(len(chunk)):
            if tgt[b] < 0:
                continue
            lo = int(tgt[b]) + min_dist
            if lo >= chunk.shape[1] - 1:
                continue
            for l in range(L):
                pat = cache[f"L{l}.attn.pattern"][b, :, lo:, int(tgt[b])]  # [head, q_downstream]
                acc[l] += pat.mean(dim=-1).cpu().numpy()
            n_seen += 1
    return acc / max(n_seen, 1)


def ig_head_ranking(lm: LoadedModel, tokens: np.ndarray, phase: np.ndarray) -> dict:
    """Integrated-gradient QK attribution for the reading-frame counter (aa-head, uniform-like data).

    This is the instrument that recovers the marker edge a naive gradient reports as 0 (softmax
    saturation, D20). It is defined for the planted TinyTransformer tracer path; for GFMNano the
    supported discovery screen is `attention_survey` + `head_ablation_table`. Raises with a pointer
    if called on an incompatible model, rather than returning a misleading number.
    """
    raise NotImplementedError(
        "QK-IG attribution (gfmcd.tracer.qk) targets the planted TinyTransformer, where the edge set "
        "is known. On a trained GFMNano, rank heads with attention_survey() then decide with "
        "head_ablation_table() -- attribution ranks, intervention decides (D27)."
    )


# ----------------------------------------------------------------------------
# PILLAR 3 -- ABLATION  (kept adjacent to discovery; pillar 2 is the ceiling below)
# ----------------------------------------------------------------------------


@dataclass
class HeadEffect:
    layer: int
    head: int
    attn_to_target: float
    delta_loss: float
    delta_acc: float


def head_ablation_table(
    lm: LoadedModel,
    tokens: np.ndarray,
    labels: np.ndarray,
    target_pos: np.ndarray | None = None,
    heads: list[tuple[int, int]] | None = None,
) -> list[HeadEffect]:
    """Ablate each head in turn; report Δloss (the arbiter) and Δaccuracy (which saturates).

    `labels` are amino-acid labels for the aa-head models. Δloss is the number to trust: the L1H2
    lesson is that a head can carry huge attention specificity and move the loss by 2e-6 while
    accuracy (pinned at 1.0) shows nothing. Sorted by Δloss, descending.
    """
    if not lm.is_aa_head:
        raise ValueError("head_ablation_table scores amino-acid loss; use copy_head_ablation for copy models")
    base_loss, base_acc = aa_loss_acc(lm, tokens, labels)
    attn = (
        attention_survey(lm, tokens, target_pos)
        if target_pos is not None
        else np.zeros((lm.cfg.n_layers, lm.cfg.n_heads))
    )
    heads = heads or [(l, h) for l in range(lm.cfg.n_layers) for h in range(lm.cfg.n_heads)]
    out = []
    for l, h in heads:
        with ablate_heads(lm.model, (l, h)):
            loss, acc = aa_loss_acc(lm, tokens, labels)
        out.append(HeadEffect(l, h, float(attn[l, h]), loss - base_loss, base_acc - acc))
    out.sort(key=lambda e: e.delta_loss, reverse=True)
    return out


@dataclass
class RedundancyResult:
    singles: dict[tuple[int, int], float]  # per-head Δloss alone
    sum_of_singles: float
    subset_delta: float  # Δloss ablating the whole set at once
    interpretation: str


def redundancy_test(
    lm: LoadedModel, tokens: np.ndarray, labels: np.ndarray, heads: list[tuple[int, int]]
) -> RedundancyResult:
    """Ablate each head alone vs all together. Sum >> joint => partial redundancy (the copy-circuit case).

    A distributed circuit where each of N heads costs a lot alone is anomalous unless the heads are
    non-independent. This measures it directly (COPY_TRACE.md's finding: Σ singles 2.91 vs joint 1.32).
    """
    base_loss, _ = aa_loss_acc(lm, tokens, labels)
    singles = {}
    for l, h in heads:
        with ablate_heads(lm.model, (l, h)):
            loss, _ = aa_loss_acc(lm, tokens, labels)
        singles[(l, h)] = loss - base_loss
    with ablate_heads(lm.model, heads):
        joint_loss, _ = aa_loss_acc(lm, tokens, labels)
    subset = joint_loss - base_loss
    s = sum(singles.values())
    if subset < 0.7 * s:
        interp = "partial redundancy: joint << sum of singles (heads are non-independent)"
    elif subset > 1.3 * s:
        interp = "super-additive: joint > sum of singles (heads cooperate)"
    else:
        interp = "additive: joint ~ sum of singles (independent contributions)"
    return RedundancyResult(singles, s, subset, interp)


# ----------------------------------------------------------------------------
# PILLAR 4 -- BASELINES  (the pooled-statistic ceiling: the decisive control)
# ----------------------------------------------------------------------------


@dataclass
class CeilingReport:
    pooled_oracle_acc: float  # anchor-free codon-level naive Bayes, per D30 (~0.9509 on real CDS)
    model_phase_acc: float  # the model's own phase-probe accuracy, far field
    demanded: float  # what the objective demands (1.0 for the forcing task)
    circuit_predicted: bool
    note: str = ""


def pooled_ceiling(
    lm: LoadedModel, tokens: np.ndarray, phase: np.ndarray, ctx: int = 120
) -> float:
    """Anchor-free pooled ceiling: can a bounded-window codon statistic recover the frame with no ATG?

    This is the shortcut-proofness test (pillar 2/4). Reproduces D30's 0.9509 on real CDS. If this
    meets what the objective demands, there is no circuit to trace -- the shortcut suffices.
    """
    from gfmcd.oracle import oracle_correct

    ok = oracle_correct(tokens.astype(np.int64), phase, ctx=ctx)
    coding = phase >= 0
    return float(ok[coding].mean())


def ceiling_report(lm: LoadedModel, tokens: np.ndarray, phase: np.ndarray, demanded: float = 1.0) -> CeilingReport:
    """Pillar 2, in one call: pooled ceiling vs the model, with the circuit-predicted verdict."""
    pooled = pooled_ceiling(lm, tokens, phase)
    layers = sig.layer_probe(lm.model, tokens, phase, lm.device)
    model_phase = max(layers.values())
    predicted = demanded > pooled + 1e-6
    note = (
        "pooled route is imperfect -> a circuit can pay off"
        if predicted
        else "pooled route suffices -> expect NO circuit (nothing to trace)"
    )
    return CeilingReport(pooled, model_phase, demanded, predicted, note)


# ----------------------------------------------------------------------------
# PILLAR 5 -- CONFLICT  (the mechanism verdict: counter / relay / bag)
# ----------------------------------------------------------------------------


def cue_conflict_curve(
    lm: LoadedModel,
    tokens: np.ndarray,
    phase: np.ndarray,
    bins: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Insert one nt after the ATG; measure anchor-share vs content-share by distance from the edit.

    The load-bearing intervention (D24): ablation cannot see a redundant circuit, cue conflict can.
    A counter follows the anchor at every distance; a bag follows content beyond ~30 nt. Read at the
    phase probe. Returns [{distance, anchor_share, content_share, n}].

    NB: this reads at the L2 phase probe, which D35 showed is the WRONG readout for the mechanism
    verdict -- on real DNA the anchor does not reach L2. `verdict_2x2` uses `output_conflict_share`
    below. This function is kept for the by-distance probe view.
    """
    start = sig.start_index(phase)
    curve, _ = sig.anchor_content_by_distance(lm.model, tokens, phase, start, lm.device, bins=bins)
    return [
        {"distance": d, "anchor_share": a, "content_share": c, "n": n} for (d, a, c, n) in curve
    ]


@torch.no_grad()
def output_conflict_share(lm: LoadedModel, tokens: np.ndarray, phase: np.ndarray, far: int = 64) -> dict:
    """Cue conflict read at the model's OUTPUT, not a probe (D35). Returns {anchor, content, n}.

    Insert one nt after the ATG: the local codon structure slides one base while the anchor stays put,
    so anchor and content disagree about the frame by exactly one. Score the model's own far-field
    amino-acid predictions against the frame the anchor implies vs the frame the content implies.
    This is the y-axis the corrected 2x2 uses; a fixed-layer probe measures whichever variable that
    layer happens to carry, and on real DNA the anchor is not at L2 (D34/D35).
    """
    from gfmcd.data.genome import amino_acid_labels

    start = sig.start_index(phase)
    body = phase >= 0
    ins = start + 3
    sh = sig.frameshift(tokens, ins)

    # slide the coding mask with the inserted base
    body_sh = np.zeros_like(body)
    for r, i in enumerate(ins):
        body_sh[r, :i] = body[r, :i]
        body_sh[r, i + 1 :] = body[r, i:-1]

    pos = np.arange(tokens.shape[1])[None, :]
    aa_anchor = amino_acid_labels(sh, np.where(body_sh, (pos - start[:, None]) % 3, IGNORE))
    aa_content = amino_acid_labels(sh, np.where(body_sh, (pos - (start[:, None] + 1)) % 3, IGNORE))

    pred = _run_logits(lm, sh).argmax(-1).numpy()
    m = body_sh & (pos - start[:, None] >= far) & (aa_anchor != IGNORE) & (aa_content != IGNORE)
    return {
        "anchor": float((pred[m] == aa_anchor[m]).mean()),
        "content": float((pred[m] == aa_content[m]).mean()),
        "n": int(m.sum()),
    }


@dataclass
class MechanismSignature:
    name: str
    anchor_dependence: float  # x: peak-over-depth far-field ATG-ablation effect minus decoy control
    anchor_share: float  # y: far-field cue-conflict anchor share (read at the probe)
    ablation_raw: float
    control_raw: float
    verdict: str


def mechanism_signature(lm: LoadedModel, tokens: np.ndarray, phase: np.ndarray, name: str | None = None) -> MechanismSignature:
    """The 2-D mechanism signature: (anchor dependence, anchor share) -> counter / relay / bag.

    x uses peak_anchor_dependence (max over depth) because a fixed-layer probe measures whichever
    variable that layer carries -- the mistake that produced the retracted relay (D34/D35). y is the
    far-field cue-conflict anchor share. The verdict follows the frozen thresholds (PREREGISTRATION):
        counter  x large, y > 0.5
        relay    x large, y < 0.5
        bag      x ~ 0,   y ~ 0
    """
    name = name or lm.name
    peak, best_layer, _ = sig.peak_anchor_dependence(lm.model, tokens, phase, lm.device)
    s, _extra, _cache = sig.signature(lm.model, tokens, phase, lm.device, name)
    y = s.anchor_share_l2probe
    if peak < 0.10:
        verdict = "bag (anchor not used at range)"
    elif y > 0.50:
        verdict = "counter (frame counted from the anchor)"
    else:
        verdict = "relay (anchor load-bearing but not counted from)"
    return MechanismSignature(name, float(peak), float(y), s.ablation_raw, s.control_raw, verdict)


@torch.no_grad()
def _cache_resid(lm: LoadedModel, tokens: np.ndarray, batch: int = 64) -> dict:
    """resid_post for every layer, concatenated over the batch. {layer_idx: tensor[n, w, d]}."""
    t = torch.from_numpy(tokens).long()
    outs = {L: [] for L in range(lm.cfg.n_layers)}
    for i in range(0, len(t), batch):
        _, cache = lm.model.run_with_cache(t[i : i + batch].to(lm.device))
        for L in range(lm.cfg.n_layers):
            outs[L].append(cache[f"L{L}.resid_post"].cpu())
    return {L: torch.cat(v) for L, v in outs.items()}


def phase_decodability_by_layer(lm: LoadedModel, tokens: np.ndarray, phase: np.ndarray,
                                steps: int = 200) -> list[float]:
    """Linear-probe accuracy for the reading FRAME at each layer's residual stream (coding positions).

    Where the frame becomes decodable is where the anchor has been located and the count established --
    the spatial signature of the circuit forming across depth. A control (untrained model) sits at ~1/3.
    """
    from gfmcd.signatures import fit_probe

    resid = _cache_resid(lm, tokens, batch=64)
    mask = phase >= 0
    y = torch.from_numpy(phase[mask]).long()
    accs = []
    for L in range(lm.cfg.n_layers):
        x = resid[L][torch.from_numpy(mask)]
        probe, mu, sd = fit_probe(x, y, steps=steps)
        accs.append(float((probe((x - mu) / sd).argmax(-1) == y).float().mean()))
    return accs


@dataclass
class Verdict2x2:
    """The frozen 2x2 verdict (PREREGISTRATION): circuit forms iff anchor share > 0.50 AND the
    start-codon ablation effect exceeds the decoy-Met control by > 0.10. Both are causal, neither is
    training accuracy (which saturates, D27). Continuous values reported regardless of side."""

    cell: str
    name: str
    intact_acc: float
    ablated_acc: float
    control_acc: float
    ablation_effect: float  # (intact - ablated) - (intact - control), the anchor-specific drop
    anchor_share: float  # far-field cue-conflict anchor share (>=64 nt), read at the OUTPUT (D35)
    content_share: float
    circuit_formed: bool
    anchor_share_l2probe: float = 0.0  # the old L2-probe reading, kept for transparency (retracted D35)


ANCHOR_SHARE_MIN = 0.50
ABLATION_MIN = 0.10


def verdict_2x2(lm: LoadedModel, tokens: np.ndarray, aa: np.ndarray, phase: np.ndarray, cell: str | None = None) -> Verdict2x2:
    """Apply the frozen 2x2 verdict to a cell checkpoint. Wraps scripts/verdict_2x2.py as one call.

    The 2x2 is (shortcut absent/present) x (detector easy/hard); the relay survives in cell D alone
    (D34/D35). This measures both frozen criteria and returns the continuous values plus the boolean.
    """
    from gfmcd.data.uniform_cds import ATG
    from gfmcd.signatures import downstream_met, fit_probe

    NON_START = np.array([1, 3, 1], dtype=np.int8)  # CTC
    cell = cell or (lm.meta or {}).get("cell") or {"uniform": "A"}.get(lm.pool, lm.pool)
    start = sig.start_index(phase)

    intact = aa_loss_acc(lm, tokens, aa)[1]

    abl = tokens.copy()
    for r, s in enumerate(start):
        abl[r, s : s + 3] = NON_START
    ablated = aa_loss_acc(lm, abl, aa)[1]

    met = downstream_met(tokens, start)
    have = met >= 0
    ctl = tokens[have].copy()
    for i, r in enumerate(np.where(have)[0]):
        ctl[i, met[r] : met[r] + 3] = NON_START
    control = aa_loss_acc(lm, ctl, aa[have])[1]
    intact_have = aa_loss_acc(lm, tokens[have], aa[have])[1]

    ablation_effect = (intact - ablated) - (intact_have - control)

    # Far-field cue conflict read at the model's OUTPUT (D35). The probe view (L2) is kept alongside
    # for transparency, but the verdict follows the output: on real DNA the anchor is not at L2, and
    # reading there labelled the C/D counters "no circuit" (the error D35 corrected).
    out = output_conflict_share(lm, tokens, phase, far=64)
    anchor, content = out["anchor"], out["content"]
    probe = cue_conflict_curve(lm, tokens, phase, bins=[(64, 200)])
    anchor_l2probe = probe[0]["anchor_share"] if probe else 0.0

    formed = anchor > ANCHOR_SHARE_MIN and ablation_effect > ABLATION_MIN
    return Verdict2x2(str(cell), lm.name, intact, ablated, control, ablation_effect, anchor, content,
                      formed, anchor_l2probe)


@dataclass
class LambdaPoint:
    name: str
    lam: float
    shortcut_nats: float  # I(nt; phase) mutual information -- the shortcut's information content
    aa_acc: float
    anchor_share: float
    verdict: str


def lambda_sweep(checkpoints: list[str], n: int = 384, seed: int = 555) -> list[LambdaPoint]:
    """Sweep a set of checkpoints (varying shortcut strength) and return mechanism vs information content.

    Each checkpoint's data pool has a different I(nt;phase): as the shortcut carries more of the frame,
    the mechanism should shift from anchored counter to anchor-free bag -- all at aa_acc ~ 1.0, so only
    the mechanism axis (anchor share) separates them. Reads `lam` and `shortcut_nats` from each blob.
    """
    from gfmcd.lab.load import load_checkpoint, load_uniform_windows

    out = []
    for stem in checkpoints:
        m = load_checkpoint(stem)
        meta = m.meta or {}
        lam = meta.get("lam", np.nan)
        nats = meta.get("shortcut_nats", np.nan)
        tok, aa, phase = load_uniform_windows(n=n, seed=seed)
        acc = aa_loss_acc(m, tok, aa)[1]
        mech = mechanism_signature(m, tok, phase, name=stem)
        out.append(LambdaPoint(stem, float(lam), float(nats), acc, mech.anchor_share, mech.verdict))
    out.sort(key=lambda p: (np.isnan(p.shortcut_nats), p.shortcut_nats))
    return out


def start_codon_ablation(lm: LoadedModel, tokens: np.ndarray, phase: np.ndarray) -> dict:
    """Far-field start-codon ablation with the paired decoy-Met control (D24). Δloss + Δphase-probe.

    The control is not optional: destroying an in-frame downstream ATG (an ordinary Met) tests that
    the effect is the *position* of the anchor, not the edit type.
    """
    s, extra, _ = sig.signature(lm.model, tokens, phase, lm.device, lm.name)
    return {
        "far_intact_phase": extra["intact_far"],
        "far_ablated_phase": extra["ablated_far"],
        "ablation_drop": s.ablation_raw,
        "decoy_control_drop": s.control_raw,
        "anchor_specific_drop": s.ablation_effect,  # ablation minus control
        "n_far": s.n_far,
    }


# ----------------------------------------------------------------------------
# REPEAT-COPY variants (the causal, n_out=None induction models)
# ----------------------------------------------------------------------------


@torch.no_grad()
def copy_metrics(lm: LoadedModel, batch, seed: int = 0) -> dict:
    """Copy accuracy, base-rate ceiling, and single-edit cue-conflict shares for a repeat-copy model.

    `batch` is a RepeatCopyBatch. Reports the shortcut-proof positive: model >> base rate, and under
    a single distal edit the model follows the distal source, not the local continuation.
    """
    from gfmcd.data.repeat_copy import (
        ar_inputs_targets,
        base_rate_ceiling,
        copy_accuracy,
        single_conflict_batch,
    )

    inp, tgt, cmask = ar_inputs_targets(batch)
    logits = _run_logits(lm, inp).numpy()
    acc = copy_accuracy(logits, tgt, cmask)
    base = base_rate_ceiling(batch)

    edited, distal, local, smask = single_conflict_batch(batch, seed=seed)
    clog = _run_logits(lm, edited[:, :-1]).numpy().argmax(-1)
    follow_distal = float((clog[smask] == distal[smask]).mean())
    follow_local = float((clog[smask] == local[smask]).mean())
    return {
        "copy_accuracy": acc,
        "base_rate_ceiling": base,
        "conflict_follows_distal": follow_distal,
        "conflict_follows_local": follow_local,
        "conflict_neither": 1.0 - follow_distal - follow_local,
    }


@torch.no_grad()
def copy_loss(lm: LoadedModel, inp: np.ndarray, tgt: np.ndarray, cmask: np.ndarray) -> float:
    """Cross-entropy restricted to copy positions. The arbiter for copy-head ablation."""
    logits = _run_logits(lm, inp)
    y = torch.from_numpy(tgt).long()
    m = torch.from_numpy(cmask)
    return float(F.cross_entropy(logits[m], y[m]))


@torch.no_grad()
def copy_source_attention(lm: LoadedModel, batch, batch_size: int = 32) -> np.ndarray:
    """Mean attention from each copy position to its source position, per (layer, head). [L, H].

    The induction analogue of attention_survey: a source-following head attends from the second
    occurrence back to the matching position in the first. Ranks candidate copy heads; the ablation
    table decides.
    """
    model = lm.model
    L, H = lm.cfg.n_layers, lm.cfg.n_heads
    inp = batch.tokens[:, :-1]
    cmask = batch.copy_mask[:, :-1] if batch.copy_mask.shape[1] == batch.tokens.shape[1] else batch.copy_mask
    src = batch.source_index
    t = torch.from_numpy(inp).long()
    acc = np.zeros((L, H))
    n_seen = 0
    for i in range(0, len(t), batch_size):
        chunk = t[i : i + batch_size].to(lm.device)
        _, cache = model.run_with_cache(chunk)
        for b in range(len(chunk)):
            qs = np.where(cmask[i + b])[0]
            qs = qs[qs < chunk.shape[1]]
            if len(qs) == 0:
                continue
            ks = src[i + b, qs]
            good = (ks >= 0) & (ks < chunk.shape[1])
            qs, ks = qs[good], ks[good]
            if len(qs) == 0:
                continue
            for l in range(L):
                pat = cache[f"L{l}.attn.pattern"][b]  # [head, q, k]
                acc[l] += pat[:, qs, ks].mean(dim=-1).cpu().numpy()
            n_seen += 1
    return acc / max(n_seen, 1)


def copy_head_ablation(lm: LoadedModel, batch, heads: list[tuple[int, int]] | None = None) -> list[HeadEffect]:
    """Ablate each head; report Δ(copy-position loss). The copy-circuit analogue of head_ablation_table."""
    from gfmcd.data.repeat_copy import ar_inputs_targets

    inp, tgt, cmask = ar_inputs_targets(batch)
    base = copy_loss(lm, inp, tgt, cmask)
    attn = copy_source_attention(lm, batch)
    heads = heads or [(l, h) for l in range(lm.cfg.n_layers) for h in range(lm.cfg.n_heads)]
    out = []
    for l, h in heads:
        with ablate_heads(lm.model, (l, h)):
            loss = copy_loss(lm, inp, tgt, cmask)
        out.append(HeadEffect(l, h, float(attn[l, h]), loss - base, 0.0))
    out.sort(key=lambda e: e.delta_loss, reverse=True)
    return out


def copy_redundancy(lm: LoadedModel, batch, heads: list[tuple[int, int]]) -> RedundancyResult:
    """Single vs joint copy-loss ablation for a set of copy heads. sum >> joint => partial redundancy.

    The induction analogue of redundancy_test. COPY_TRACE.md found the 5 copy heads partially
    redundant (Σ singles ~2.9 vs joint ~1.3): no single head is necessary, so ablation alone
    understates the circuit -- the reason cue conflict (copy_metrics) is the decisive test.
    """
    from gfmcd.data.repeat_copy import ar_inputs_targets

    inp, tgt, cmask = ar_inputs_targets(batch)
    base = copy_loss(lm, inp, tgt, cmask)
    singles = {}
    for l, h in heads:
        with ablate_heads(lm.model, (l, h)):
            singles[(l, h)] = copy_loss(lm, inp, tgt, cmask) - base
    with ablate_heads(lm.model, heads):
        joint = copy_loss(lm, inp, tgt, cmask) - base
    s = sum(singles.values())
    if joint < 0.7 * s:
        interp = "partial redundancy: joint << sum of singles (heads are non-independent)"
    elif joint > 1.3 * s:
        interp = "super-additive: joint > sum of singles (heads cooperate)"
    else:
        interp = "additive: joint ~ sum of singles (independent contributions)"
    return RedundancyResult(singles, s, joint, interp)
