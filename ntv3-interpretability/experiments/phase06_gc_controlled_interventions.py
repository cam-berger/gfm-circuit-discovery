"""Phase 06 -- Controlled GC interventions.

Port of dnabert/06_gc_controlled_interventions.ipynb to Nucleotide
Transformer v3 (8M). Phase05 found GC fraction was the strongest simple
biological feature correlated with ``deconv7_main``'s transferable rescue
signal (Spearman rho +0.35) after the dominant probe_score confound. This
phase tests it causally: construct sequence variants with a controlled GC
shift (length-preserving, distributed substitutions, not one contiguous
edit block) and ask whether the promoter-probe readout moves accordingly,
with a **matched neutral control** -- same number of edited positions, but
substitutions that stay within their base class (A<->T, G<->C) and so
leave GC fraction unchanged -- to separate a GC-specific effect from
generic sequence disruption.

Main questions (unchanged from the original):
  1. Does increasing GC content increase the promoter-probe readout?
  2. Does decreasing GC content reduce it?
  3. Does the change follow GC magnitude monotonically (dose-response)?
  4. Can the GC effect be separated from generic disruption (matched
     neutral control)?
  5. Does ``deconv7_main``'s activation move along a "promoter direction"
     (promoter-mean minus non-promoter-mean pooled activation) in
     proportion to the GC shift?

Question 5's direction is estimated on one random half of a promoter/
non-promoter pool and tested on the other half, so the projection-vs-GC
correlation isn't evaluated on the same data that defined the direction --
a single-split simplification of the original notebook's cross-validation,
documented rather than silently matched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ntv3 import config, data, metrics, stats, viz
from ntv3.model import Ntv3Adapter
from ntv3.patching import capture_activation
from phase03_component_localization import build_component_registry
from phase04_causal_interventions import load_final_depth_probe, score_sequences

RESULTS_PATH = config.RESULTS_DIR / "phase06_results.json"
TARGET_COMPONENT = "deconv7_main"
GC_DELTAS = [-0.20, -0.10, 0.10, 0.20]
N_REPEATS = 3
N_SOURCE = 60  # per-class-balanced source cohort size


def shift_gc(seq: str, delta_fraction: float, rng: np.random.Generator) -> tuple[str, int]:
    """Substitute the minimum number of positions needed to shift GC
    content by ``delta_fraction`` of the sequence length: AT->GC for a
    positive delta, GC->AT for a negative one, positions chosen at random
    (distributed, not a contiguous block). Returns ``(variant, n_edits)``;
    ``n_edits`` may be less than the target if too few eligible bases exist."""
    chars = list(seq)
    n_target = round(abs(delta_fraction) * len(seq))
    if delta_fraction > 0:
        eligible = [i for i, c in enumerate(chars) if c in "AT"]
        pool = "GC"
    else:
        eligible = [i for i, c in enumerate(chars) if c in "GC"]
        pool = "AT"
    n_edits = min(n_target, len(eligible))
    chosen = rng.choice(eligible, size=n_edits, replace=False) if n_edits > 0 else np.array([], dtype=int)
    for i in chosen:
        chars[int(i)] = rng.choice(list(pool))
    return "".join(chars), n_edits


def matched_neutral_substitution(seq: str, n_edits: int, rng: np.random.Generator) -> str:
    """Same number of base substitutions as a GC variant, but within-class
    (A<->T or G<->C), so GC fraction is unchanged -- isolates a GC-specific
    effect from generic sequence disruption."""
    chars = list(seq)
    n_edits = min(n_edits, len(chars))
    positions = rng.choice(len(chars), size=n_edits, replace=False) if n_edits > 0 else np.array([], dtype=int)
    swap = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    for i in positions:
        chars[int(i)] = swap[chars[int(i)]]
    return "".join(chars)


def main():
    t0 = time.time()
    notes = [f"target component: {TARGET_COMPONENT}"]

    print("Loading final-depth probe...")
    probe, depth_index, depth_label = load_final_depth_probe()

    print(f"Sampling source cohort (n={N_SOURCE}, balanced by label)...")
    test_df = data.load_split("test")
    source_df = data.balanced_pilot(test_df, N_SOURCE // 2, seed=config.SEED)
    source_seqs = list(source_df["sequence"])
    source_labels = source_df["label"].to_numpy()
    print(f"  {len(source_seqs)} source sequences")

    print("Loading NTv3 adapter...")
    adapter = Ntv3Adapter.from_pretrained()

    baseline_scores = score_sequences(adapter, probe, depth_index, source_seqs)

    # -- 1. generate GC variants + matched neutral controls ---------------------------
    print(f"Generating GC variants (deltas={GC_DELTAS}, {N_REPEATS} repeats)...")
    rng = np.random.default_rng(config.SEED)
    variant_rows = []  # source_idx, delta, repeat, kind ("gc"/"neutral"), sequence, n_edits
    for si, seq in enumerate(source_seqs):
        for delta in GC_DELTAS:
            for rep in range(N_REPEATS):
                gc_variant, n_edits = shift_gc(seq, delta, rng)
                neutral_variant = matched_neutral_substitution(seq, n_edits, rng)
                variant_rows.append({"source_idx": si, "delta": delta, "repeat": rep, "kind": "gc",
                                      "sequence": gc_variant, "n_edits": n_edits})
                variant_rows.append({"source_idx": si, "delta": delta, "repeat": rep, "kind": "neutral",
                                      "sequence": neutral_variant, "n_edits": n_edits})
    variant_df = pd.DataFrame(variant_rows)
    print(f"  {len(variant_df)} variant sequences ({len(variant_df) // 2} GC + {len(variant_df) // 2} neutral)")

    # -- 2. score all variants ---------------------------------------------------------
    print("Scoring all variants...")
    variant_scores = score_sequences(adapter, probe, depth_index, list(variant_df["sequence"]))
    variant_df["score"] = variant_scores
    variant_df["baseline_score"] = variant_df["source_idx"].map(lambda i: baseline_scores[i])
    variant_df["delta_score"] = variant_df["score"] - variant_df["baseline_score"]
    variant_df["achieved_gc_shift"] = variant_df.apply(
        lambda r: (metrics.composition_features(r["sequence"])["gc_fraction"]
                   - metrics.composition_features(source_seqs[r["source_idx"]])["gc_fraction"]),
        axis=1,
    )

    # average over repeats per (source, delta, kind)
    per_source = (
        variant_df.groupby(["source_idx", "delta", "kind"])["delta_score"].mean().reset_index()
    )
    gc_specific = []
    for delta in GC_DELTAS:
        gc_vals = per_source[(per_source["delta"] == delta) & (per_source["kind"] == "gc")]["delta_score"].to_numpy()
        neutral_vals = per_source[(per_source["delta"] == delta) & (per_source["kind"] == "neutral")]["delta_score"].to_numpy()
        gc_summary = stats.bootstrap_ci(gc_vals, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        neutral_summary = stats.bootstrap_ci(neutral_vals, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        specific_diff = stats.paired_bootstrap_diff(gc_vals, neutral_vals, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        gc_specific.append({
            "delta": delta, "gc_effect": gc_summary, "neutral_effect": neutral_summary,
            "gc_minus_neutral": specific_diff,
        })
        print(f"  delta={delta:+.2f}: GC effect mean={gc_summary['mean']:+.4f}  "
              f"neutral effect mean={neutral_summary['mean']:+.4f}  "
              f"GC-specific={specific_diff['mean']:+.4f} CI=({specific_diff['ci_low']:+.4f}, {specific_diff['ci_high']:+.4f})")

    # -- 3. promoter direction in target-component activation space -------------------
    print(f"\nBuilding promoter direction in {TARGET_COMPONENT} activation space (single train/test split)...")
    registry = {e["name"]: e for e in build_component_registry(adapter)}
    module = registry[TARGET_COMPONENT]["module"]

    all_idx = np.arange(len(source_seqs))
    rng2 = np.random.default_rng(config.SEED + 3)
    perm = rng2.permutation(all_idx)
    half = len(perm) // 2
    direction_idx, test_idx = perm[:half], perm[half:]

    def pooled_activation(seqs: list[str], batch_size: int = 32) -> np.ndarray:
        # Raw conv/deconv module output is channel-first (B, C, L); masked_mean_at_depth
        # expects sequence-first (B, L, H) like the adapter's own hidden_states -- permute
        # before pooling, or content_span_at_depth divides by the wrong (channel-count) axis.
        pieces = []
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start : start + batch_size]
            batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
            act = capture_activation(module, lambda b=batch: adapter.forward(b, output_hidden_states=False))
            act = act.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
            pieces.append(adapter.masked_mean_at_depth(act, batch).cpu().numpy())
        return np.concatenate(pieces, axis=0)

    source_activation = pooled_activation(source_seqs)
    direction_labels = source_labels[direction_idx]
    direction_vec = (
        source_activation[direction_idx][direction_labels == 1].mean(axis=0)
        - source_activation[direction_idx][direction_labels == 0].mean(axis=0)
    )
    direction_vec = direction_vec / (np.linalg.norm(direction_vec) + 1e-12)

    test_source_set = set(test_idx.tolist())
    variant_test_mask = variant_df["source_idx"].isin(test_source_set) & (variant_df["kind"] == "gc")
    test_variants = variant_df[variant_test_mask].reset_index(drop=True)
    print(f"  scoring {len(test_variants)} held-out GC variants' component activations...")
    test_variant_activation = pooled_activation(list(test_variants["sequence"]))
    test_source_activation = source_activation[test_variants["source_idx"].to_numpy()]
    delta_activation = test_variant_activation - test_source_activation
    projection = delta_activation @ direction_vec

    proj_vs_gcshift = stats.spearman(projection, test_variants["achieved_gc_shift"].to_numpy())
    proj_vs_deltascore = stats.spearman(projection, test_variants["delta_score"].to_numpy())
    print(f"  held-out projection vs. achieved GC shift: rho={proj_vs_gcshift['rho']:+.3f} p={proj_vs_gcshift['p_value']:.2e}")
    print(f"  held-out projection vs. delta_score:        rho={proj_vs_deltascore['rho']:+.3f} p={proj_vs_deltascore['p_value']:.2e}")

    # -- results ----------------------------------------------------------------------
    results = {
        "meta": {
            "phase": "06_gc_controlled_interventions",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "target_component": TARGET_COMPONENT,
            "readout_depth_index": depth_index,
            "readout_depth_label": depth_label,
            "n_source": len(source_seqs),
            "gc_deltas": GC_DELTAS,
            "n_repeats": N_REPEATS,
        },
        "gc_specific_effects": gc_specific,
        "promoter_direction": {
            "n_direction_sequences": int(len(direction_idx)),
            "n_test_sequences": int(len(test_idx)),
            "projection_vs_gc_shift_spearman": proj_vs_gcshift,
            "projection_vs_delta_score_spearman": proj_vs_deltascore,
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    variant_df.to_csv(config.RESULTS_DIR / "phase06_variants.csv", index=False)
    print(f"\nWrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    deltas = [r["delta"] for r in gc_specific]
    gc_means = [r["gc_effect"]["mean"] for r in gc_specific]
    gc_los = [r["gc_effect"]["ci_low"] for r in gc_specific]
    gc_his = [r["gc_effect"]["ci_high"] for r in gc_specific]
    neutral_means = [r["neutral_effect"]["mean"] for r in gc_specific]
    neutral_los = [r["neutral_effect"]["ci_low"] for r in gc_specific]
    neutral_his = [r["neutral_effect"]["ci_high"] for r in gc_specific]
    fig, ax = viz.new_figure(figsize=(7.5, 5))
    ax.errorbar(deltas, gc_means, yerr=[np.array(gc_means) - np.array(gc_los), np.array(gc_his) - np.array(gc_means)],
                fmt="o-", color=viz.CATEGORICAL[0], label="GC-shifted", linewidth=2, capsize=3)
    ax.errorbar(deltas, neutral_means, yerr=[np.array(neutral_means) - np.array(neutral_los), np.array(neutral_his) - np.array(neutral_means)],
                fmt="o--", color=viz.CATEGORICAL[2], label="matched neutral", linewidth=2, capsize=3)
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle=":")
    ax.axvline(0, color=viz.AXIS, linewidth=1, linestyle=":")
    viz._base_style(ax, title="NTv3 8M: GC-shift dose-response vs. matched neutral control",
                     xlabel="target GC delta", ylabel="mean P(promoter) change")
    ax.legend(frameon=False, fontsize=9, labelcolor=viz.INK_SECONDARY)
    viz.save(fig, config.FIGURES_DIR / "phase06_gc_dose_response.png")

    fig, ax = viz.scatter_plot(
        projection, test_variants["delta_score"].to_numpy(),
        title=f"NTv3 8M: {TARGET_COMPONENT} promoter-direction projection vs. score change (held-out)",
        xlabel="projection onto promoter direction",
        ylabel="delta_score (GC variant - source)",
    )
    viz.save(fig, config.FIGURES_DIR / "phase06_direction_projection.png")

    print(f"Phase 06 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
