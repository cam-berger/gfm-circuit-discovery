"""Phase 04 -- Causal interventions (reconstruct-inject).

Port of dnabert/04_promoter_causal_interventions.ipynb to Nucleotide
Transformer v3 (8M), reframed like phase03 around conv/deconv/transformer
components rather than attention heads (see phase03's module docstring for
why -- the Transformer bottleneck is a single token at this study's 70nt
promoter length).

Where phase03 asks a **necessity** question (does removing a component's
signal hurt the promoter readout), this phase asks a **sufficiency**
question via reconstruct-inject: if a component's activation is captured
from a clean promoter and transplanted into a dinucleotide-shuffled
("corrupted") version of some promoter's forward pass, does it restore the
promoter-associated readout that the corruption destroyed?

Main metric, unchanged from the original:

    recovery_fraction = (patched_score - corrupt_score) / (clean_score - corrupt_score)

    0   = no rescue
    1   = full rescue
    0-1 = partial rescue
    >1  = over-restoration
    <0  = patching worsens the corrupted representation

Three donor conditions per component, mirroring the original's structure:
  - **matched**: the sequence's own clean (pre-corruption) activation --
    tests whether the component carries this sequence's information.
  - **unrelated**: a different clean promoter's activation (fixed
    derangement, no sequence patched with its own donor) -- if this rescues
    about as well as matched, the component carries a transferable
    promoter-class signal rather than sequence-specific information.
  - **non-promoter**: a real negative (label==0) sequence's activation --
    tests whether *any* activation rescues (a null result here would mean
    the readout is fooled by generic activity, not promoter content).

A **self-patch control** (inject a corrupted sequence's own captured
activation back into itself) is run first for every component as an
implementation check: it must reproduce the unpatched corrupt score
exactly, or the injection hook is wrong. See ``ntv3/patching.py`` for why
NTv3 needs no nucleotide-span projection matrix here, unlike the DNABERT-2
original (single-nucleotide tokenization, length-preserving corruption).

Target components: the top ``N_TARGET_COMPONENTS`` hookable components from
phase03's necessity ranking (by |mean effect|) -- skip-connection
sufficiency is not tested here (their ablation used a parallel
``Core.forward`` reimplementation in phase03; injecting into them would
need the analogous reimplementation, deferred rather than rushed).
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

from ntv3 import config, data, metrics, stats, viz
from ntv3.model import Ntv3Adapter
from ntv3.patching import capture_activation, inject_activation
from phase03_component_localization import build_component_registry

RESULTS_PATH = config.RESULTS_DIR / "phase04_results.json"
N_TARGET_COMPONENTS = 4
MINIMUM_RESCUE_GAP = 0.10  # probability units; see printed gap distribution
SELF_PATCH_TOLERANCE = 1e-4


def load_final_depth_probe():
    saved = joblib.load(config.RESULTS_DIR / "ntv3_all_depth_probes.joblib")
    final_depth = len(saved["depth_labels"]) - 1
    return saved["probes"][final_depth], final_depth, saved["depth_labels"][final_depth]


def score_batch(adapter: Ntv3Adapter, probe, depth_index: int, batch) -> np.ndarray:
    out = adapter.forward(batch, output_hidden_states=True)
    pooled = adapter.masked_mean_at_depth(out.hidden_states[depth_index], batch).cpu().numpy()
    return probe.predict_proba(pooled)[:, 1]


def score_sequences(adapter: Ntv3Adapter, probe, depth_index: int, seqs: list[str], batch_size: int = 32) -> np.ndarray:
    scores = []
    for start in range(0, len(seqs), batch_size):
        batch = adapter.tokenize(seqs[start : start + batch_size], pad_length=config.PAD_LENGTH)
        scores.append(score_batch(adapter, probe, depth_index, batch))
    return np.concatenate(scores)


def capture_component_for_cohort(adapter: Ntv3Adapter, module, seqs: list[str], batch_size: int = 32) -> torch.Tensor:
    """Batched capture of one module's output across a whole cohort, in
    cohort order, as a single stacked tensor."""
    pieces = []
    for start in range(0, len(seqs), batch_size):
        batch = adapter.tokenize(seqs[start : start + batch_size], pad_length=config.PAD_LENGTH)
        pieces.append(capture_activation(module, lambda b=batch: adapter.forward(b, output_hidden_states=False)))
    return torch.cat(pieces, dim=0)


def score_with_injected_cohort(
    adapter: Ntv3Adapter, probe, depth_index: int, module, recipient_seqs: list[str],
    donor_activations: torch.Tensor, batch_size: int = 32,
) -> np.ndarray:
    """Score ``recipient_seqs`` with ``module``'s output replaced, per
    sequence, by the matching row of ``donor_activations`` (same cohort
    order)."""
    scores = []
    for start in range(0, len(recipient_seqs), batch_size):
        chunk = recipient_seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        donor_chunk = donor_activations[start : start + len(chunk)]
        with inject_activation(module, donor_chunk):
            scores.append(score_batch(adapter, probe, depth_index, batch))
    return np.concatenate(scores)


def no_fixed_point_derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    while np.any(perm == np.arange(n)) or n <= 1:
        perm = rng.permutation(n)
        if n <= 1:
            break
    return perm


def main():
    t0 = time.time()
    notes = []

    print("Loading final-depth probe from phase01...")
    probe, depth_index, depth_label = load_final_depth_probe()
    print(f"  depth={depth_index} ({depth_label})")

    print("Loading phase03 component necessity ranking...")
    phase03_results = json.loads((config.RESULTS_DIR / "phase03_results.json").read_text())
    ranked = sorted(phase03_results["component_necessity"], key=lambda r: -abs(r["mean"]))
    target_names = [r["name"] for r in ranked[:N_TARGET_COMPONENTS]]
    print(f"  target components (top {N_TARGET_COMPONENTS} by |necessity effect|): {target_names}")
    notes.append(f"target components chosen from phase03 necessity ranking: {target_names}")

    print("Building clean/corrupt cohort (dinucleotide shuffle, same cohort as phase02/03)...")
    test_df = data.load_split("test")
    positives = test_df[test_df["label"] == 1]
    n_cohort = min(config.CAUSAL_COHORT_PER_CLASS, len(positives))
    cohort = positives.sample(n=n_cohort, replace=False, random_state=config.SEED).reset_index(drop=True)
    clean_seqs_all = list(cohort["sequence"])

    rng = np.random.default_rng(config.SEED)
    corrupt_seqs_all = [metrics.dinucleotide_shuffle(s, rng) for s in clean_seqs_all]

    print("Loading NTv3 adapter...")
    adapter = Ntv3Adapter.from_pretrained()

    clean_scores_all = score_sequences(adapter, probe, depth_index, clean_seqs_all)
    corrupt_scores_all = score_sequences(adapter, probe, depth_index, corrupt_seqs_all)
    gaps_all = clean_scores_all - corrupt_scores_all
    print(f"  clean-corrupt gap: mean={gaps_all.mean():.3f} median={np.median(gaps_all):.3f} "
          f"p25={np.percentile(gaps_all, 25):.3f} p75={np.percentile(gaps_all, 75):.3f}")

    rescue_mask = gaps_all >= MINIMUM_RESCUE_GAP
    clean_seqs = [s for s, m in zip(clean_seqs_all, rescue_mask) if m]
    corrupt_seqs = [s for s, m in zip(corrupt_seqs_all, rescue_mask) if m]
    clean_scores = clean_scores_all[rescue_mask]
    corrupt_scores = corrupt_scores_all[rescue_mask]
    gaps = gaps_all[rescue_mask]
    n = len(clean_seqs)
    print(f"  rescue cohort (gap >= {MINIMUM_RESCUE_GAP}): {n} / {n_cohort}")
    notes.append(f"rescue cohort n={n}/{n_cohort} (gap >= {MINIMUM_RESCUE_GAP}); "
                 f"full-cohort gap mean={gaps_all.mean():.4f}")
    if n < 10:
        raise RuntimeError(f"rescue cohort too small (n={n}) to draw any conclusion; lower MINIMUM_RESCUE_GAP")

    unrelated_idx = no_fixed_point_derangement(n, config.SEED + 1)
    unrelated_donor_seqs = [clean_seqs[i] for i in unrelated_idx]

    train_df = data.load_split("train")
    negatives = train_df[train_df["label"] == 0]
    nonpromoter_donors = negatives.sample(n=n, replace=False, random_state=config.SEED + 2).reset_index(drop=True)
    nonpromoter_seqs = list(nonpromoter_donors["sequence"])

    registry = {e["name"]: e for e in build_component_registry(adapter)}

    component_results = []
    for name in target_names:
        entry = registry[name]
        module = entry["module"]
        print(f"\nComponent {name}...")

        clean_act = capture_component_for_cohort(adapter, module, clean_seqs)
        corrupt_act = capture_component_for_cohort(adapter, module, corrupt_seqs)
        unrelated_act = clean_act[unrelated_idx]  # same clean captures, reindexed -- no recompute needed
        nonpromoter_act = capture_component_for_cohort(adapter, module, nonpromoter_seqs)

        # -- self-patch control: corrupt's own activation injected into itself --
        self_patched_scores = score_with_injected_cohort(adapter, probe, depth_index, module, corrupt_seqs, corrupt_act)
        self_patch_max_diff = float(np.max(np.abs(self_patched_scores - corrupt_scores)))
        self_patch_ok = self_patch_max_diff < SELF_PATCH_TOLERANCE
        status = "OK" if self_patch_ok else "FAILED -- do not trust this component's results"
        print(f"  self-patch control: max|diff|={self_patch_max_diff:.2e} ({status})")

        def recovery(patched_scores: np.ndarray) -> np.ndarray:
            return (patched_scores - corrupt_scores) / gaps

        matched_scores = score_with_injected_cohort(adapter, probe, depth_index, module, corrupt_seqs, clean_act)
        unrelated_scores = score_with_injected_cohort(adapter, probe, depth_index, module, corrupt_seqs, unrelated_act)
        nonpromoter_scores = score_with_injected_cohort(adapter, probe, depth_index, module, corrupt_seqs, nonpromoter_act)

        matched_recovery = recovery(matched_scores)
        unrelated_recovery = recovery(unrelated_scores)
        nonpromoter_recovery = recovery(nonpromoter_scores)

        matched_summary = stats.bootstrap_ci(matched_recovery, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        unrelated_summary = stats.bootstrap_ci(unrelated_recovery, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        nonpromoter_summary = stats.bootstrap_ci(nonpromoter_recovery, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)

        matched_vs_unrelated = stats.paired_bootstrap_diff(matched_recovery, unrelated_recovery, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        matched_vs_nonpromoter = stats.paired_bootstrap_diff(matched_recovery, nonpromoter_recovery, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        unrelated_vs_nonpromoter = stats.paired_bootstrap_diff(unrelated_recovery, nonpromoter_recovery, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)

        print(f"  matched recovery:      mean={matched_summary['mean']:+.3f} CI=({matched_summary['ci_low']:+.3f}, {matched_summary['ci_high']:+.3f})")
        print(f"  unrelated recovery:    mean={unrelated_summary['mean']:+.3f} CI=({unrelated_summary['ci_low']:+.3f}, {unrelated_summary['ci_high']:+.3f})")
        print(f"  non-promoter recovery: mean={nonpromoter_summary['mean']:+.3f} CI=({nonpromoter_summary['ci_low']:+.3f}, {nonpromoter_summary['ci_high']:+.3f})")
        print(f"  matched - unrelated:   mean={matched_vs_unrelated['mean']:+.3f} CI=({matched_vs_unrelated['ci_low']:+.3f}, {matched_vs_unrelated['ci_high']:+.3f})")

        component_results.append({
            "name": name,
            "self_patch_max_diff": self_patch_max_diff,
            "self_patch_ok": self_patch_ok,
            "matched_recovery": matched_summary,
            "unrelated_recovery": unrelated_summary,
            "nonpromoter_recovery": nonpromoter_summary,
            "matched_vs_unrelated_diff": matched_vs_unrelated,
            "matched_vs_nonpromoter_diff": matched_vs_nonpromoter,
            "unrelated_vs_nonpromoter_diff": unrelated_vs_nonpromoter,
        })

    results = {
        "meta": {
            "phase": "04_causal_interventions",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "readout_depth_index": depth_index,
            "readout_depth_label": depth_label,
            "cohort_size_full": n_cohort,
            "cohort_size_rescue": n,
            "minimum_rescue_gap": MINIMUM_RESCUE_GAP,
            "target_components": target_names,
        },
        "component_results": component_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    conditions = ["matched_recovery", "unrelated_recovery", "nonpromoter_recovery"]
    condition_labels = ["matched\ndonor", "unrelated\npromoter donor", "non-promoter\ndonor"]
    fig, ax = viz.new_figure(figsize=(11, 5.5))
    x = np.arange(len(target_names))
    width = 0.25
    for ci, (cond, clabel) in enumerate(zip(conditions, condition_labels)):
        means = [r[cond]["mean"] for r in component_results]
        los = [r[cond]["mean"] - r[cond]["ci_low"] for r in component_results]
        his = [r[cond]["ci_high"] - r[cond]["mean"] for r in component_results]
        ax.bar(x + (ci - 1) * width, means, width=width, yerr=[los, his],
               color=viz.CATEGORICAL[ci], label=clabel, error_kw=dict(ecolor=viz.INK_SECONDARY, capsize=3), zorder=2)
    ax.axhline(0, color=viz.AXIS, linewidth=1, zorder=1)
    ax.axhline(1, color=viz.AXIS, linewidth=1, linestyle=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=9, labelcolor=viz.INK_SECONDARY)
    viz._base_style(ax, title="NTv3 8M: reconstruct-inject recovery fraction by donor type",
                     xlabel="component", ylabel="recovery fraction")
    viz.save(fig, config.FIGURES_DIR / "phase04_recovery_by_donor_type.png")

    print(f"Phase 04 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
