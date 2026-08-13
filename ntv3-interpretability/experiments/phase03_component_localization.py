"""Phase 03 -- Component localization.

Port of dnabert/03_promoter_component_localization.ipynb to Nucleotide
Transformer v3 (8M), **reframed** rather than mechanically translated: the
DNABERT-2/NTv2 original tests necessity of attention heads and grouped
heads. NTv3's Transformer bottleneck collapses to a single token at the
70nt promoter length this study uses (see README.md's architecture note),
so "which head" is close to meaningless there. This phase instead asks the
analogous question of NTv3's real position-resolved machinery: which
conv/deconv tower stage, which channel group within one, and which
skip connection does the frozen model's promoter signal depend on?

Method, mirrored from the original's necessity-testing structure:
  1. Component necessity sweep -- mean-ablate (ntv3.patching.mean_ablate)
     each of the 32 hookable conv/deconv/transformer-FFN components one at
     a time; score the promoter-positive cohort (same cohort and readout
     probe as phase02) before and after; rank by mean confidence drop.
  2. Skip-connection necessity sweep -- mean-ablate each of the 7
     conv-to-deconv skip connections (ntv3.patching.forward_with_ablated_skip,
     since the residual add isn't a hookable module boundary; see that
     module's docstring for why).
  3. Channel-group necessity -- for the single most-necessary component
     from step 1, split its channels into 8 groups of 32 and ablate each
     group individually, to see whether necessity concentrates in a subset
     of channels or is distributed across the whole component.

Ablation is mean-ablation throughout: each component's replacement value is
its own per-channel mean over a fixed reference cohort (REFERENCE_COHORT_SIZE
sequences, balanced by label, drawn from the train split -- disjoint from
the promoter-positive test-split cohort being scored), not zero. See
ntv3/patching.py's module docstring for why.

**Readout depth: the final hidden_states entry (deconv7, full nucleotide
resolution), not phase01's best-by-accuracy depth.** hidden_states are
produced in forward-pass order (conv1..conv7, xformer1-2, deconv1..deconv7);
a component can only change a readout that runs *after* it. phase01's best
depth turned out to be conv7 -- captured mid-forward, before the Transformer
bottleneck, deconv tower, and every skip connection even execute. An earlier
version of this script used that depth and got an exact 0.0000 effect for
every one of those components: not a bug, a direct consequence of asking
whether ablating something changes a value computed before it runs. The
final depth is the only one causally downstream of every component tested
here, so it's the one this phase uses -- see ``ntv3_all_depth_probes.joblib``
(saved by phase01 for exactly this reason).
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

from ntv3 import config, data, stats, viz
from ntv3.model import Ntv3Adapter
from ntv3.patching import ChannelMeans, collect_channel_means, forward_with_ablated_skip, mean_ablate

RESULTS_PATH = config.RESULTS_DIR / "phase03_results.json"
REFERENCE_COHORT_SIZE = 50  # per class; means computed over this cohort
N_CHANNEL_GROUPS = 8


def load_final_depth_probe():
    """The last hidden_states depth (deconv7, full nucleotide resolution) --
    the only depth causally downstream of every component this phase
    ablates. See this module's docstring."""
    saved = joblib.load(config.RESULTS_DIR / "ntv3_all_depth_probes.joblib")
    final_depth = len(saved["depth_labels"]) - 1
    return saved["probes"][final_depth], final_depth, saved["depth_labels"][final_depth]


def score_sequences(adapter: Ntv3Adapter, probe, depth_index: int, seqs: list[str], batch_size: int = 32) -> np.ndarray:
    scores = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        out = adapter.forward(batch, output_hidden_states=True)
        pooled = adapter.masked_mean_at_depth(out.hidden_states[depth_index], batch).cpu().numpy()
        scores.append(probe.predict_proba(pooled)[:, 1])
    return np.concatenate(scores)


def score_sequences_ablated_skip(
    adapter: Ntv3Adapter, probe, depth_index: int, seqs: list[str], *,
    ablate_conv_stage: int, replacement: torch.Tensor, batch_size: int = 32,
) -> np.ndarray:
    scores = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        with torch.no_grad():
            out = forward_with_ablated_skip(
                adapter.model.core, batch.input_ids,
                ablate_conv_stage=ablate_conv_stage, replacement=replacement,
                output_hidden_states=True,
            )
        pooled = adapter.masked_mean_at_depth(out["hidden_states"][depth_index], batch).cpu().numpy()
        scores.append(probe.predict_proba(pooled)[:, 1])
    return np.concatenate(scores)


def build_component_registry(adapter: Ntv3Adapter) -> list[dict]:
    registry = []
    for i, block in enumerate(adapter.conv_tower_blocks):
        registry.append({"name": f"conv{i + 1}_main", "module": block.conv.conv, "channel_dim": 1})
        registry.append({"name": f"conv{i + 1}_res", "module": block.res_conv.conv_block.conv, "channel_dim": 1})
    for j, layer in enumerate(adapter.transformer_blocks):
        registry.append({"name": f"xformer{j + 1}_fc1", "module": layer.fc1, "channel_dim": 2})
        registry.append({"name": f"xformer{j + 1}_fc2", "module": layer.fc2, "channel_dim": 2})
    for k, block in enumerate(adapter.deconv_tower_blocks):
        registry.append({"name": f"deconv{k + 1}_main", "module": block.conv.conv, "channel_dim": 1})
        registry.append({"name": f"deconv{k + 1}_res", "module": block.res_conv.conv_block.conv, "channel_dim": 1})
    return registry


def main():
    t0 = time.time()
    notes = []

    print("Loading final-depth probe from phase01 (causally downstream of every component)...")
    probe, depth_index, depth_label = load_final_depth_probe()
    print(f"  depth={depth_index} ({depth_label})")

    print("Sampling cohorts...")
    test_df = data.load_split("test")
    positives = test_df[test_df["label"] == 1]
    n_cohort = min(config.CAUSAL_COHORT_PER_CLASS, len(positives))
    cohort = positives.sample(n=n_cohort, replace=False, random_state=config.SEED).reset_index(drop=True)
    seqs = list(cohort["sequence"])
    print(f"  scored cohort: {n_cohort} label==1 sequences from test split (matches phase02)")

    train_df = data.load_split("train")
    reference = data.balanced_pilot(train_df, REFERENCE_COHORT_SIZE, seed=config.SEED + 1)
    reference_seqs = list(reference["sequence"])
    print(f"  reference cohort (for channel means): {len(reference_seqs)} sequences, train split, disjoint seed")
    notes.append(
        f"scored cohort n={n_cohort} (label==1, test split, same as phase02); "
        f"reference cohort n={len(reference_seqs)} (balanced, train split) for mean-ablation baselines"
    )

    print("Loading NTv3 adapter...")
    adapter = Ntv3Adapter.from_pretrained()

    baseline_scores = score_sequences(adapter, probe, depth_index, seqs)
    print(f"  baseline mean P(promoter) = {baseline_scores.mean():.3f}")

    def reference_forward():
        for start in range(0, len(reference_seqs), 32):
            chunk = reference_seqs[start : start + 32]
            batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
            adapter.forward(batch, output_hidden_states=False)

    # -- 1. component necessity sweep ------------------------------------------------
    registry = build_component_registry(adapter)
    print(f"Component necessity sweep ({len(registry)} components x {n_cohort} sequences)...")
    component_results = []
    component_means: dict[str, ChannelMeans] = {}
    for entry in registry:
        means = collect_channel_means(entry["module"], reference_forward, channel_dim=entry["channel_dim"])
        component_means[entry["name"]] = means
        with mean_ablate(entry["module"], means):
            ablated_scores = score_sequences(adapter, probe, depth_index, seqs)
        effect = baseline_scores - ablated_scores
        summary = stats.bootstrap_ci(effect, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        summary["name"] = entry["name"]
        summary["n_channels"] = int(means.mean.shape[0])
        component_results.append(summary)
        print(f"  {entry['name']:16s} n_ch={summary['n_channels']:4d} "
              f"mean_effect={summary['mean']:+.4f} CI=({summary['ci_low']:+.4f}, {summary['ci_high']:+.4f})")

    component_results.sort(key=lambda r: -r["mean"])
    top_component = component_results[0]
    print(f"Most necessary component: {top_component['name']} (mean effect {top_component['mean']:+.4f})")

    # -- 2. skip-connection necessity sweep -------------------------------------------
    print(f"Skip-connection necessity sweep ({adapter.num_downsamples} stages x {n_cohort} sequences)...")
    skip_results = []
    for stage in range(adapter.num_downsamples):
        stage_module = adapter.model.core.stem if stage == 0 else adapter.conv_tower_blocks[stage - 1].avg_pool
        means = collect_channel_means(stage_module, reference_forward, channel_dim=1)
        replacement = means.mean.view(1, -1, 1)
        ablated_scores = score_sequences_ablated_skip(
            adapter, probe, depth_index, seqs, ablate_conv_stage=stage, replacement=replacement,
        )
        effect = baseline_scores - ablated_scores
        summary = stats.bootstrap_ci(effect, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        summary["stage"] = stage
        skip_results.append(summary)
        print(f"  skip{stage + 1:<2d} mean_effect={summary['mean']:+.4f} "
              f"CI=({summary['ci_low']:+.4f}, {summary['ci_high']:+.4f})")

    # -- 3. channel-group necessity on the top component ------------------------------
    top_name = top_component["name"]
    top_entry = next(e for e in registry if e["name"] == top_name)
    top_means = component_means[top_name]
    n_ch = int(top_means.mean.shape[0])
    group_size = max(1, n_ch // N_CHANNEL_GROUPS)
    groups = [list(range(g * group_size, min((g + 1) * group_size, n_ch))) for g in range(N_CHANNEL_GROUPS)]
    groups = [g for g in groups if g]
    print(f"Channel-group necessity on {top_name} ({len(groups)} groups of ~{group_size} channels)...")
    group_results = []
    for gi, group in enumerate(groups):
        with mean_ablate(top_entry["module"], top_means, channels=group):
            ablated_scores = score_sequences(adapter, probe, depth_index, seqs)
        effect = baseline_scores - ablated_scores
        summary = stats.bootstrap_ci(effect, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        summary["group"] = gi
        summary["channels"] = [group[0], group[-1]]
        group_results.append(summary)
        print(f"  group {gi} ch[{group[0]}:{group[-1]}] mean_effect={summary['mean']:+.4f}")

    # -- results ----------------------------------------------------------------------
    results = {
        "meta": {
            "phase": "03_component_localization",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "readout_depth_index": depth_index,
            "readout_depth_label": depth_label,
            "cohort_size": n_cohort,
            "reference_cohort_size": len(reference_seqs),
            "n_components_tested": len(registry),
        },
        "component_necessity": component_results,
        "skip_connection_necessity": skip_results,
        "top_component": top_name,
        "channel_group_necessity": group_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    names = [r["name"] for r in component_results]
    means_v = [r["mean"] for r in component_results]
    los = [r["mean"] - r["ci_low"] for r in component_results]
    his = [r["ci_high"] - r["mean"] for r in component_results]
    fig, ax = viz.bar_plot(
        names, means_v,
        title="NTv3 8M: component necessity (mean-ablation, promoter cohort)",
        ylabel="mean P(promoter) drop when ablated",
        errors=[los, his],
        figsize=(13, 5.5),
    )
    for label in ax.get_xticklabels():
        label.set_fontsize(6.5)
    viz.save(fig, config.FIGURES_DIR / "phase03_component_necessity.png")

    skip_labels = [f"skip{r['stage'] + 1}" for r in skip_results]
    skip_means = [r["mean"] for r in skip_results]
    skip_los = [r["ci_low"] for r in skip_results]
    skip_his = [r["ci_high"] for r in skip_results]
    fig, ax = viz.errorbar_plot(
        skip_labels, skip_means, skip_los, skip_his,
        title="NTv3 8M: skip-connection necessity (conv-tower stage -> matching deconv stage)",
        ylabel="mean P(promoter) drop when skip ablated",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase03_skip_connection_necessity.png")

    group_labels = [f"g{r['group']}\n[{r['channels'][0]}:{r['channels'][1]}]" for r in group_results]
    group_means = [r["mean"] for r in group_results]
    group_los = [r["ci_low"] for r in group_results]
    group_his = [r["ci_high"] for r in group_results]
    fig, ax = viz.errorbar_plot(
        group_labels, group_means, group_los, group_his,
        title=f"NTv3 8M: channel-group necessity within {top_name}",
        ylabel="mean P(promoter) drop when group ablated",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase03_channel_group_necessity.png")

    print(f"Phase 03 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
