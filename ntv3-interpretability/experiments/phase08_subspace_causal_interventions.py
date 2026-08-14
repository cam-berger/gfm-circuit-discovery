"""Phase 08 -- Causal interventions on the low-rank subspace.

Port of dnabert/08_promoter_mlp_subspace_causal_interventions.ipynb to
Nucleotide Transformer v3 (8M). Phase07 found the top singular directions
of ``deconv7_main``'s clean-minus-corrupt delta vectors (an unsupervised
SVD, not aimed at "promoter" specifically). This phase tests whether that
subspace is *causally* involved, the way phase04 tested individual
components:

  1. **Necessity**: project the top-r subspace out of clean promoter
     sequences' activation (``ntv3.patching.project_out_subspace``) and
     measure whether the promoter-probe readout drops.
  2. **Sufficiency**: inject the (sign-oriented, magnitude-scaled) subspace
     direction into non-promoter sequences' activation
     (``inject_subspace_direction``) and measure whether the readout rises.
  3. **Specificity**: compare both effects against random directions of the
     same rank and norm, drawn fresh per trial -- if the learned subspace
     is doing something the direction's *identity* explains (not just "any
     perturbation of this size"), its effect should sit outside the random
     directions' distribution.
  4. **Rank**: repeat for rank 1, 2, and 4, matching phase07's finding that
     a single direction was the weakest predictor there too.

Each singular vector's sign is arbitrary from SVD alone, so before using it
as an injection direction this phase orients it by its correlation with the
clean-minus-corrupt score gap from phase07's own cohort (a vector that
*reduces* the gap when projected onto is flipped) -- oriented once here,
not assumed from phase07.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ntv3 import config, data, metrics, stats, viz
from ntv3.model import Ntv3Adapter
from ntv3.patching import inject_subspace_direction, project_out_subspace
from phase03_component_localization import build_component_registry
from phase04_causal_interventions import load_final_depth_probe, score_sequences

RESULTS_PATH = config.RESULTS_DIR / "phase08_results.json"
TARGET_COMPONENT = "deconv7_main"
RANKS = [1, 2, 4]
N_RANDOM_TRIALS = 20


def score_with_subspace_hook(adapter, probe, depth_index, seqs, module, hook_maker, batch_size=32):
    scores = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        with hook_maker(batch):
            out = adapter.forward(batch, output_hidden_states=True)
        pooled = adapter.masked_mean_at_depth(out.hidden_states[depth_index], batch).cpu().numpy()
        scores.append(probe.predict_proba(pooled)[:, 1])
    return np.concatenate(scores)


def random_orthonormal_directions(r: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.standard_normal((r, dim))
    q, _ = np.linalg.qr(mat.T)
    return q[:, :r].T  # [r, dim], orthonormal rows


def main():
    t0 = time.time()
    notes = [f"target component: {TARGET_COMPONENT}"]

    print("Loading final-depth probe and phase07 SVD basis...")
    probe, depth_index, depth_label = load_final_depth_probe()
    svd = np.load(config.RESULTS_DIR / "phase07_svd_basis.npz")
    Vt = svd["Vt"]  # [n_components, hidden]
    hidden_dim = Vt.shape[1]

    print("Reconstructing phase04/07's clean/corrupt cohort (same seeds) to orient basis signs...")
    test_df = data.load_split("test")
    positives = test_df[test_df["label"] == 1]
    n_cohort = min(config.CAUSAL_COHORT_PER_CLASS, len(positives))
    cohort = positives.sample(n=n_cohort, replace=False, random_state=config.SEED).reset_index(drop=True)
    clean_seqs = list(cohort["sequence"])
    rng = np.random.default_rng(config.SEED)
    corrupt_seqs = [metrics.dinucleotide_shuffle(s, rng) for s in clean_seqs]

    print("Loading NTv3 adapter...")
    adapter = Ntv3Adapter.from_pretrained()
    registry = {e["name"]: e for e in build_component_registry(adapter)}
    module = registry[TARGET_COMPONENT]["module"]

    from phase07_low_rank_subspace_analysis import pooled_activation_batched
    clean_act = pooled_activation_batched(adapter, module, clean_seqs)
    corrupt_act = pooled_activation_batched(adapter, module, corrupt_seqs)
    delta = clean_act - corrupt_act
    delta_centered = delta - delta.mean(axis=0, keepdims=True)
    clean_scores_cohort = score_sequences(adapter, probe, depth_index, clean_seqs)
    corrupt_scores_cohort = score_sequences(adapter, probe, depth_index, corrupt_seqs)
    gaps_cohort = clean_scores_cohort - corrupt_scores_cohort

    max_rank = max(RANKS)
    oriented = np.zeros((max_rank, hidden_dim))
    scales = np.zeros(max_rank)
    for k in range(max_rank):
        proj_k = delta_centered @ Vt[k]
        corr = np.corrcoef(proj_k, gaps_cohort)[0, 1]
        sign = 1.0 if corr >= 0 else -1.0
        oriented[k] = sign * Vt[k]
        scales[k] = proj_k.std()
    print(f"  oriented top-{max_rank} directions; per-direction scale (proj std): {np.round(scales, 3).tolist()}")

    print("\nBuilding necessity cohort (promoter positives) and sufficiency cohort (non-promoters)...")
    necessity_seqs = clean_seqs  # same clean promoters used throughout
    necessity_baseline = clean_scores_cohort

    negatives = test_df[test_df["label"] == 0]
    n_suff = min(config.CAUSAL_COHORT_PER_CLASS, len(negatives))
    suff_cohort = negatives.sample(n=n_suff, replace=False, random_state=config.SEED).reset_index(drop=True)
    sufficiency_seqs = list(suff_cohort["sequence"])
    sufficiency_baseline = score_sequences(adapter, probe, depth_index, sufficiency_seqs)
    print(f"  necessity cohort n={len(necessity_seqs)}, sufficiency cohort n={len(sufficiency_seqs)}")

    rank_results = []
    for r in RANKS:
        print(f"\n=== rank {r} ===")
        basis_r = oriented[:r]  # [r, hidden], real basis
        combined_direction = basis_r.sum(axis=0)
        combined_direction = combined_direction / (np.linalg.norm(combined_direction) + 1e-12)
        mean_scale = float(scales[:r].mean())

        import torch
        basis_t = torch.tensor(basis_r, dtype=torch.float32)
        direction_t = torch.tensor(combined_direction, dtype=torch.float32)

        # -- necessity: real subspace --
        necessity_ablated = score_with_subspace_hook(
            adapter, probe, depth_index, necessity_seqs, module,
            lambda b: project_out_subspace(module, b, basis_t),
        )
        necessity_effect = necessity_baseline - necessity_ablated
        necessity_summary = stats.bootstrap_ci(necessity_effect, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)

        # -- sufficiency: real direction --
        sufficiency_injected = score_with_subspace_hook(
            adapter, probe, depth_index, sufficiency_seqs, module,
            lambda b: inject_subspace_direction(module, b, direction_t, mean_scale),
        )
        sufficiency_effect = sufficiency_injected - sufficiency_baseline
        sufficiency_summary = stats.bootstrap_ci(sufficiency_effect, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)

        print(f"  necessity (real subspace):   mean drop={necessity_summary['mean']:+.4f} "
              f"CI=({necessity_summary['ci_low']:+.4f}, {necessity_summary['ci_high']:+.4f})")
        print(f"  sufficiency (real direction): mean rise={sufficiency_summary['mean']:+.4f} "
              f"CI=({sufficiency_summary['ci_low']:+.4f}, {sufficiency_summary['ci_high']:+.4f})")

        # -- specificity: random norm-matched directions --
        rand_rng = np.random.default_rng(config.SEED + r)
        random_necessity_means = []
        random_sufficiency_means = []
        for trial in range(N_RANDOM_TRIALS):
            rand_basis = random_orthonormal_directions(r, hidden_dim, rand_rng)
            rand_combined = rand_basis.sum(axis=0)
            rand_combined = rand_combined / (np.linalg.norm(rand_combined) + 1e-12)
            rand_basis_t = torch.tensor(rand_basis, dtype=torch.float32)
            rand_direction_t = torch.tensor(rand_combined, dtype=torch.float32)

            rand_ablated = score_with_subspace_hook(
                adapter, probe, depth_index, necessity_seqs, module,
                lambda b: project_out_subspace(module, b, rand_basis_t),
            )
            random_necessity_means.append(float((necessity_baseline - rand_ablated).mean()))

            rand_injected = score_with_subspace_hook(
                adapter, probe, depth_index, sufficiency_seqs, module,
                lambda b: inject_subspace_direction(module, b, rand_direction_t, mean_scale),
            )
            random_sufficiency_means.append(float((rand_injected - sufficiency_baseline).mean()))

        necessity_percentile = float(np.mean(np.array(random_necessity_means) < necessity_summary["mean"]) * 100)
        sufficiency_percentile = float(np.mean(np.array(random_sufficiency_means) < sufficiency_summary["mean"]) * 100)
        print(f"  necessity effect percentile vs. {N_RANDOM_TRIALS} random rank-{r} directions:   {necessity_percentile:.0f}th")
        print(f"  sufficiency effect percentile vs. {N_RANDOM_TRIALS} random rank-{r} directions: {sufficiency_percentile:.0f}th")

        rank_results.append({
            "rank": r,
            "necessity": necessity_summary,
            "sufficiency": sufficiency_summary,
            "random_necessity_means": random_necessity_means,
            "random_sufficiency_means": random_sufficiency_means,
            "necessity_percentile_vs_random": necessity_percentile,
            "sufficiency_percentile_vs_random": sufficiency_percentile,
        })

    # -- results ----------------------------------------------------------------------
    results = {
        "meta": {
            "phase": "08_subspace_causal_interventions",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "target_component": TARGET_COMPONENT,
            "readout_depth_index": depth_index,
            "readout_depth_label": depth_label,
            "necessity_cohort_size": len(necessity_seqs),
            "sufficiency_cohort_size": len(sufficiency_seqs),
            "n_random_trials": N_RANDOM_TRIALS,
        },
        "rank_results": rank_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    ranks = [r["rank"] for r in rank_results]
    nec_means = [r["necessity"]["mean"] for r in rank_results]
    nec_los = [r["necessity"]["ci_low"] for r in rank_results]
    nec_his = [r["necessity"]["ci_high"] for r in rank_results]
    suf_means = [r["sufficiency"]["mean"] for r in rank_results]
    suf_los = [r["sufficiency"]["ci_low"] for r in rank_results]
    suf_his = [r["sufficiency"]["ci_high"] for r in rank_results]

    fig, ax = viz.new_figure(figsize=(8, 5))
    x = np.arange(len(ranks))
    width = 0.35
    ax.bar(x - width / 2, nec_means, width=width,
           yerr=[np.array(nec_means) - np.array(nec_los), np.array(nec_his) - np.array(nec_means)],
           color=viz.CATEGORICAL[0], label="necessity (score drop when removed)", error_kw=dict(capsize=3), zorder=2)
    ax.bar(x + width / 2, suf_means, width=width,
           yerr=[np.array(suf_means) - np.array(suf_los), np.array(suf_his) - np.array(suf_means)],
           color=viz.CATEGORICAL[1], label="sufficiency (score rise when injected)", error_kw=dict(capsize=3), zorder=2)
    ax.axhline(0, color=viz.AXIS, linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"rank {r}" for r in ranks])
    ax.legend(frameon=False, fontsize=9, labelcolor=viz.INK_SECONDARY)
    viz._base_style(ax, title=f"NTv3 8M: {TARGET_COMPONENT} subspace necessity & sufficiency by rank",
                     ylabel="mean P(promoter) change")
    viz.save(fig, config.FIGURES_DIR / "phase08_necessity_sufficiency_by_rank.png")

    fig, ax = viz.bar_plot(
        [f"rank {r}" for r in ranks],
        [r["necessity_percentile_vs_random"] for r in rank_results],
        title=f"NTv3 8M: real-subspace necessity effect, percentile vs. {N_RANDOM_TRIALS} random directions",
        ylabel="percentile (100 = stronger than every random draw)",
    )
    ax.axhline(50, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase08_specificity_percentile.png")

    print(f"Phase 08 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
