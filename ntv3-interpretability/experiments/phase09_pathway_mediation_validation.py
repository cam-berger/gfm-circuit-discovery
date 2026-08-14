"""Phase 09 -- Pathway and mediation validation.

Port of dnabert/09_promoter_pathway_mediation_validation.ipynb to
Nucleotide Transformer v3 (8M): the consolidating phase. Phases 01-08
established that individual components carry necessary, sometimes-
transferable, GC-aligned promoter signal, mostly examined one component
(or one subspace) at a time. This phase asks the question that ties them
together: when you perturb signal *early* in the stack, how does that
perturbation actually get to the final readout -- does it flow through a
specific downstream pathway, or is it carried redundantly by several
components such that restoring any one of them barely matters?

Method, adapted from the original's Layer-1-to-Layer-12 propagation +
downstream branch mediation (attention/MLP restoration) structure to
NTv3's conv/deconv stack:

  1. **Intervention point**: ``conv2_main``, phase03's single most
     necessary component (mean necessity effect +0.63). A promoter
     direction is estimated there (promoter-mean minus non-promoter-mean
     pooled activation, the same construction as phase06's, computed fresh
     for this component) and injected into non-promoter recipients --
     sufficiency-style, like phase08, rather than the original's
     bidirectional promoter-and-non-promoter design (simplified and
     documented, not silently narrowed).
  2. **Stage-wise propagation**: using phase01's saved *per-depth* probes
     (not just the final one), score every downstream ``hidden_states``
     depth's promoter-evidence for the intervened vs. baseline run --
     tracing concretely how much the injected signal has moved the
     probe-visible promoter evidence by each depth, not just an abstract
     activation-space alignment.
  3. **Individual mediation**: for each of a handful of downstream
     components spread across the remaining conv/transformer/deconv stack,
     test whether restoring that component's *clean* (non-intervened)
     activation -- while the conv2_main intervention is still active --
     blocks the propagated effect on the final-depth readout.
     ``blocked_effect = total_effect - remaining_effect``.
  4. **Joint mediation**: restore the top-2 and top-3 individually-blocking
     components together; compare the joint blocked effect to the sum of
     individual blocked effects (redundant if joint < sum, synergistic if
     joint > sum, additive if joint ~= sum).

**Read the joint-mediation numbers with one structural caveat in mind.**
All five mediation candidates sit on a single serial chain (conv tower ->
transformer -> deconv tower), not on parallel branches the way DNABERT-2's
per-layer attention/MLP branches loosely are. ``inject_activation``
restores a component by wholesale-replacing its output tensor with a
fixed, previously-captured clean value -- so once the most-downstream
restored component in a joint set is forced to that fixed value, nothing
any more-upstream restored component did to reach it can still matter
through that computational path (only an unblocked skip connection can
carry signal around it). In this run, restoring ``deconv4_main`` alone
produced the *identical* blocked effect as restoring it together with
``conv4_main`` and/or ``conv7_main`` -- consistent with that screening
property, since conv4/conv7 are strictly upstream of deconv4 on the
non-skip path. That is a real property of *these specific hook points'
position in the graph*, not necessarily evidence that the model's
pathway is non-redundant in the stronger sense the original notebook's
same-layer branch comparisons intended. Treat "redundant" conclusions
below as bounded by this, not as a clean answer.

This phase does not introduce new causal machinery beyond nesting two of
phase08's/patching.py's hooks in one forward pass (inject at conv2_main,
restore-via-injection at a downstream component) -- see ``ntv3/patching.py``
for both primitives.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ntv3 import config, data, viz
from ntv3.model import Ntv3Adapter
from ntv3.patching import capture_activation, inject_activation, inject_subspace_direction
from phase03_component_localization import build_component_registry
from phase04_causal_interventions import score_sequences

INTERVENTION_COMPONENT = "conv2_main"
FINAL_READOUT_DEPTH_LABEL = "deconv7"
MEDIATION_CANDIDATES = ["conv4_main", "conv7_main", "xformer1_fc2", "deconv1_main", "deconv4_main"]
DIRECTION_COHORT_PER_CLASS = 60

RESULTS_PATH = config.RESULTS_DIR / "phase09_results.json"


def load_all_depth_probes():
    saved = joblib.load(config.RESULTS_DIR / "ntv3_all_depth_probes.joblib")
    return saved["probes"], saved["depth_labels"]


def pooled_activation_full_res(adapter: Ntv3Adapter, module, seqs, batch_size: int = 32) -> np.ndarray:
    """Pooled activation of a full-nucleotide-resolution (B, C, L) conv/deconv
    module output. See phase06/07's identical helper."""
    pieces = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        act = capture_activation(module, lambda b=batch: adapter.forward(b, output_hidden_states=False))
        act = act.permute(0, 2, 1)
        pieces.append(adapter.masked_mean_at_depth(act, batch).cpu().numpy())
    return np.concatenate(pieces, axis=0)


def score_all_depths(adapter, probes, seqs, batch_size=32) -> np.ndarray:
    """[n_seq, n_depths] P(promoter), one column per hidden_states depth."""
    n_depths = len(probes)
    all_scores = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        out = adapter.forward(batch, output_hidden_states=True)
        row = np.zeros((len(chunk), n_depths))
        for d in range(n_depths):
            pooled = adapter.masked_mean_at_depth(out.hidden_states[d], batch).cpu().numpy()
            row[:, d] = probes[d].predict_proba(pooled)[:, 1]
        all_scores.append(row)
    return np.concatenate(all_scores, axis=0)


def score_all_depths_with_intervention(
    adapter, probes, seqs, intervention_module, direction_t, scale, batch_size=32,
    restore_module=None, restore_activation=None,
) -> np.ndarray:
    n_depths = len(probes)
    all_scores = []
    offset = 0
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        with inject_subspace_direction(intervention_module, batch, direction_t, scale):
            if restore_module is not None:
                donor_chunk = restore_activation[offset : offset + len(chunk)]
                with inject_activation(restore_module, donor_chunk):
                    out = adapter.forward(batch, output_hidden_states=True)
            else:
                out = adapter.forward(batch, output_hidden_states=True)
        row = np.zeros((len(chunk), n_depths))
        for d in range(n_depths):
            pooled = adapter.masked_mean_at_depth(out.hidden_states[d], batch).cpu().numpy()
            row[:, d] = probes[d].predict_proba(pooled)[:, 1]
        all_scores.append(row)
        offset += len(chunk)
    return np.concatenate(all_scores, axis=0)


def main():
    t0 = time.time()
    notes = [
        f"intervention component: {INTERVENTION_COMPONENT} (phase03's top-necessity component)",
        "unidirectional (non-promoter -> promoter push) simplification of the original's "
        "bidirectional promoter/non-promoter design",
    ]

    print("Loading all per-depth probes from phase01...")
    probes, depth_labels = load_all_depth_probes()
    final_depth_index = len(depth_labels) - 1
    print(f"  {len(probes)} depths; final readout = {depth_labels[final_depth_index]}")

    print(f"Sampling promoter/non-promoter direction cohort (n={DIRECTION_COHORT_PER_CLASS} per class)...")
    test_df = data.load_split("test")
    direction_df = data.balanced_pilot(test_df, DIRECTION_COHORT_PER_CLASS, seed=config.SEED)
    direction_seqs = list(direction_df["sequence"])
    direction_labels = direction_df["label"].to_numpy()

    print("Loading NTv3 adapter...")
    adapter = Ntv3Adapter.from_pretrained()
    registry = {e["name"]: e for e in build_component_registry(adapter)}
    intervention_module = registry[INTERVENTION_COMPONENT]["module"]

    print(f"Estimating promoter direction at {INTERVENTION_COMPONENT}...")
    direction_act = pooled_activation_full_res(adapter, intervention_module, direction_seqs)
    promoter_dir = direction_act[direction_labels == 1].mean(axis=0) - direction_act[direction_labels == 0].mean(axis=0)
    scale = float(np.linalg.norm(promoter_dir))
    promoter_dir_unit = promoter_dir / (scale + 1e-12)
    direction_t = torch.tensor(promoter_dir_unit, dtype=torch.float32)
    print(f"  direction norm (used as injection scale): {scale:.4f}")

    print("\nBuilding recipient (non-promoter) cohort for propagation + mediation...")
    negatives = test_df[test_df["label"] == 0]
    n_recipients = min(config.CAUSAL_COHORT_PER_CLASS, len(negatives))
    recipients_df = negatives.sample(n=n_recipients, replace=False, random_state=config.SEED + 5).reset_index(drop=True)
    recipient_seqs = list(recipients_df["sequence"])
    print(f"  {n_recipients} non-promoter recipients")

    baseline_all_depths = score_all_depths(adapter, probes, recipient_seqs)
    intervened_all_depths = score_all_depths_with_intervention(
        adapter, probes, recipient_seqs, intervention_module, direction_t, scale,
    )
    propagation = (intervened_all_depths - baseline_all_depths).mean(axis=0)  # [n_depths]
    print("\nStage-wise propagation of the conv2_main intervention (mean P(promoter) change by depth):")
    for lbl, val in zip(depth_labels, propagation):
        print(f"  {lbl:10s} {val:+.4f}")

    total_effect = intervened_all_depths[:, final_depth_index] - baseline_all_depths[:, final_depth_index]
    print(f"\nTotal effect at final readout ({depth_labels[final_depth_index]}): mean={total_effect.mean():+.4f}")

    # -- individual mediation ---------------------------------------------------------
    print(f"\nIndividual mediation: does restoring each candidate block the effect?")
    print("  capturing each candidate's CLEAN (non-intervened) activation on the recipient cohort...")
    clean_candidate_activations = {}
    for name in MEDIATION_CANDIDATES:
        module = registry[name]["module"]
        pieces = []
        for start in range(0, len(recipient_seqs), 32):
            chunk = recipient_seqs[start : start + 32]
            batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
            pieces.append(capture_activation(module, lambda b=batch: adapter.forward(b, output_hidden_states=False)))
        clean_candidate_activations[name] = torch.cat(pieces, dim=0)

    individual_results = []
    for name in MEDIATION_CANDIDATES:
        module = registry[name]["module"]
        restored_all_depths = score_all_depths_with_intervention(
            adapter, probes, recipient_seqs, intervention_module, direction_t, scale,
            restore_module=module, restore_activation=clean_candidate_activations[name],
        )
        remaining_effect = restored_all_depths[:, final_depth_index] - baseline_all_depths[:, final_depth_index]
        blocked_effect = total_effect - remaining_effect
        individual_results.append({
            "component": name,
            "mean_total_effect": float(total_effect.mean()),
            "mean_remaining_effect": float(remaining_effect.mean()),
            "mean_blocked_effect": float(blocked_effect.mean()),
            "fraction_blocked": float(blocked_effect.mean() / total_effect.mean()) if total_effect.mean() != 0 else float("nan"),
        })
        print(f"  restore {name:16s}  remaining={remaining_effect.mean():+.4f}  "
              f"blocked={blocked_effect.mean():+.4f}  ({blocked_effect.mean()/total_effect.mean()*100:+.0f}% of total)")

    # -- joint mediation ----------------------------------------------------------------
    ranked = sorted(individual_results, key=lambda r: -r["mean_blocked_effect"])
    top2 = [r["component"] for r in ranked[:2]]
    top3 = [r["component"] for r in ranked[:3]]
    print(f"\nJoint mediation: top-2 {top2}, top-3 {top3}...")

    def score_with_joint_restore(names: list[str]) -> np.ndarray:
        scores = []
        offset = 0
        for start in range(0, len(recipient_seqs), 32):
            chunk = recipient_seqs[start : start + 32]
            batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
            with inject_subspace_direction(intervention_module, batch, direction_t, scale):
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for name in names:
                        donor = clean_candidate_activations[name][offset : offset + len(chunk)]
                        stack.enter_context(inject_activation(registry[name]["module"], donor))
                    out = adapter.forward(batch, output_hidden_states=True)
            pooled = adapter.masked_mean_at_depth(out.hidden_states[final_depth_index], batch).cpu().numpy()
            scores.append(probes[final_depth_index].predict_proba(pooled)[:, 1])
            offset += len(chunk)
        return np.concatenate(scores)

    joint_results = {}
    for label, names in [("top2", top2), ("top3", top3), ("all_candidates", MEDIATION_CANDIDATES)]:
        joint_restored = score_with_joint_restore(names)
        joint_remaining = joint_restored - baseline_all_depths[:, final_depth_index]
        joint_blocked = total_effect - joint_remaining
        sum_individual = sum(r["mean_blocked_effect"] for r in individual_results if r["component"] in names)
        joint_results[label] = {
            "components": names,
            "mean_joint_blocked_effect": float(joint_blocked.mean()),
            "sum_individual_blocked_effects": float(sum_individual),
            "additivity_residual": float(joint_blocked.mean() - sum_individual),
        }
        print(f"  {label:14s} joint_blocked={joint_blocked.mean():+.4f}  sum_individual={sum_individual:+.4f}  "
              f"residual={joint_blocked.mean()-sum_individual:+.4f} "
              f"({'redundant' if joint_blocked.mean() < sum_individual else 'synergistic/additive'})")

    # -- results + evidence summary ----------------------------------------------------
    joint_all_equal_individual_best = np.allclose(
        [joint_results[k]["mean_joint_blocked_effect"] for k in ("top2", "top3", "all_candidates")],
        ranked[0]["mean_blocked_effect"], atol=1e-6,
    )
    if abs(total_effect.mean()) < 1e-6:
        interpretation = "No detectable propagated effect at the final readout; mediation results are not interpretable."
    else:
        best = ranked[0]
        frac = best["mean_blocked_effect"] / total_effect.mean()
        caveat = (
            " Caveat: every joint-restoration condition tested here produced exactly the same "
            f"blocked effect as restoring {best['component']} alone -- consistent with the "
            "downstream-overwrite screening property described in this module's docstring "
            "(all five candidates sit on one serial chain), not necessarily genuine redundancy "
            "in the deeper sense. This scan cannot distinguish the two without a mediation "
            "method that preserves upstream contributions through skip connections."
        ) if joint_all_equal_individual_best else ""
        if frac > 0.5:
            interpretation = (
                f"{best['component']} individually blocks the majority ({frac*100:.0f}%) of the "
                f"propagated effect at the final readout.{caveat}"
            )
        else:
            interpretation = (
                f"No single tested component blocks a majority of the propagated effect "
                f"(best: {best['component']} at {frac*100:.0f}%) -- evidence leans toward a "
                f"distributed causal contribution across the downstream stack, consistent with "
                f"this repo's broader finding (MECHANISM.md) that faithfulness tends to be "
                f"irreducibly distributed rather than a sparse circuit. This is a five-candidate "
                f"scan, not an exhaustive one -- it bounds but does not close the question.{caveat}"
            )
    print(f"\nInterpretation: {interpretation}")

    results = {
        "meta": {
            "phase": "09_pathway_mediation_validation",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "intervention_component": INTERVENTION_COMPONENT,
            "final_readout_depth_index": final_depth_index,
            "final_readout_depth_label": depth_labels[final_depth_index],
            "recipient_cohort_size": n_recipients,
            "mediation_candidates": MEDIATION_CANDIDATES,
        },
        "stage_propagation": {lbl: float(v) for lbl, v in zip(depth_labels, propagation)},
        "total_effect_mean": float(total_effect.mean()),
        "individual_mediation": individual_results,
        "joint_mediation": joint_results,
        "joint_mediation_matches_top_individual": bool(joint_all_equal_individual_best),
        "interpretation": interpretation,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    fig, ax = viz.line_plot(
        depth_labels, {"mean P(promoter) change vs. baseline": propagation.tolist()},
        title=f"NTv3 8M: propagation of a {INTERVENTION_COMPONENT} promoter-direction push",
        xlabel="hidden_states depth", ylabel="mean P(promoter) change",
        figsize=(11, 5),
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    ax.set_xticklabels(depth_labels, rotation=60, ha="right", fontsize=7.5)
    viz.save(fig, config.FIGURES_DIR / "phase09_stage_propagation.png")

    names = [r["component"] for r in individual_results]
    blocked = [r["mean_blocked_effect"] for r in individual_results]
    fig, ax = viz.bar_plot(
        names, blocked,
        title=f"NTv3 8M: individual mediation -- effect blocked by restoring each component",
        ylabel="mean blocked effect (P(promoter) units)",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    ax.axhline(total_effect.mean(), color=viz.CATEGORICAL[1], linewidth=1.2, linestyle=":", label="total effect")
    ax.legend(frameon=False, fontsize=9, labelcolor=viz.INK_SECONDARY)
    viz.save(fig, config.FIGURES_DIR / "phase09_individual_mediation.png")

    print(f"\nPhase 09 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
