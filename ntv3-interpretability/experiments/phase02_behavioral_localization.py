"""Phase 02 -- Behavioral localization.

Port of dnabert/02_promoter_behavioral_localization.ipynb to Nucleotide
Transformer v3 (8M). Purely behavioral: it corrupts the input sequence in
controlled ways and measures how the frozen model's promoter-confidence
readout changes, without touching model internals. That makes it safe with
respect to the U-Net bottleneck-degeneracy issue tracked in
``ntv3/config.py`` (phase03+ hooks specific components; this phase only
ever calls ``adapter.forward`` on whole corrupted sequences).

Readout: the phase01 best-depth probe (``results/ntv3_best_depth_probe.joblib``)
applied to the corresponding ``hidden_states`` depth, scoring P(promoter) for
a sequence. "Effect" of a corruption = baseline_score - corrupted_score:
positive means removing/scrambling that structure reduced the model's
promoter confidence, i.e. the model was using it.

Three corruption families, run on a held-out cohort of confirmed positives
(label==1) from the test split -- corrupting a true promoter's structure and
watching confidence drop is informative; corrupting a negative near zero
confidence mostly isn't (floor effect):

  1. Circular shift -- shift the sequence cyclically by varying offsets.
     Tests whether the model relies on *absolute position* (any shift hurts)
     or only *local motif structure* (only large shifts hurt).
  2. Mono- vs di-nucleotide-preserving shuffle -- full shuffle destroys all
     order; dinucleotide shuffle preserves local (adjacent-base) statistics
     but destroys longer-range order. Comparing their effects asks how much
     of the promoter signal is order-dependent beyond simple composition.
  3. Regional (windowed) shuffle -- shuffle a sliding window and sweep its
     position, localizing *where* in the 70nt window the model-relevant
     structure sits.

Deviation from the DNABERT-2 original (documented, not hidden): sample sizes
are reduced for CPU-only inference (``config.CAUSAL_COHORT_PER_CLASS``), and
window/shift/repeat grids are coarser than the original notebook's.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ntv3 import config, data, metrics, stats, viz
from ntv3.model import Ntv3Adapter

RESULTS_PATH = config.RESULTS_DIR / "phase02_results.json"

CIRCULAR_SHIFTS = list(range(0, config.MAX_SEQ_LEN, 5))  # 0, 5, ..., 65
REGIONAL_WINDOW = 15
REGIONAL_STRIDE = 5
REGIONAL_SHUFFLE_REPEATS = 3


def load_best_depth_probe():
    saved = joblib.load(config.RESULTS_DIR / "ntv3_best_depth_probe.joblib")
    return saved["probe"], saved["depth"], saved["depth_label"]


def score_sequences(adapter: Ntv3Adapter, probe, depth_index: int, seqs: list[str], batch_size: int = 32) -> np.ndarray:
    """P(promoter) for each sequence, read out at a fixed hidden_states depth."""
    scores = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        out = adapter.forward(batch, output_hidden_states=True)
        pooled = adapter.masked_mean_at_depth(out.hidden_states[depth_index], batch).cpu().numpy()
        scores.append(probe.predict_proba(pooled)[:, 1])
    return np.concatenate(scores)


def main():
    t0 = time.time()
    notes = []

    print("Loading best-depth probe from phase01...")
    probe, depth_index, depth_label = load_best_depth_probe()
    print(f"  depth={depth_index} ({depth_label})")

    print("Loading GUE prom_core_all test split, sampling promoter-positive cohort...")
    test_df = data.load_split("test")
    positives = test_df[test_df["label"] == 1]
    n = min(config.CAUSAL_COHORT_PER_CLASS, len(positives))
    cohort = positives.sample(n=n, replace=False, random_state=config.SEED).reset_index(drop=True)
    seqs = list(cohort["sequence"])
    notes.append(f"cohort: {n} label==1 sequences from test split (config.CAUSAL_COHORT_PER_CLASS={config.CAUSAL_COHORT_PER_CLASS})")
    print(f"  cohort size: {n}")

    print("Loading NTv3 adapter...")
    adapter = Ntv3Adapter.from_pretrained()

    print("Scoring baseline (uncorrupted) sequences...")
    baseline_scores = score_sequences(adapter, probe, depth_index, seqs)
    print(f"  mean baseline P(promoter) = {baseline_scores.mean():.3f}")

    rng = np.random.default_rng(config.SEED)

    # -- 1. circular shift ---------------------------------------------------------
    print(f"Circular shift ({len(CIRCULAR_SHIFTS)} offsets x {n} sequences)...")
    circular_rows = []
    shift_effects: dict[int, list[float]] = {s: [] for s in CIRCULAR_SHIFTS}
    for shift in CIRCULAR_SHIFTS:
        shifted = [metrics.circular_shift(s, shift) for s in seqs]
        shifted_scores = score_sequences(adapter, probe, depth_index, shifted)
        effect = baseline_scores - shifted_scores
        shift_effects[shift] = effect.tolist()
        for seq_idx, (b, sh, e) in enumerate(zip(baseline_scores, shifted_scores, effect)):
            circular_rows.append({"seq_idx": seq_idx, "shift": shift, "baseline_score": float(b),
                                   "shifted_score": float(sh), "effect": float(e)})
    shift_summary = {
        str(shift): stats.bootstrap_ci(np.array(vals), n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        for shift, vals in shift_effects.items()
    }
    print("  done")

    # -- 2. mono vs dinucleotide shuffle --------------------------------------------
    print(f"Mono vs dinucleotide shuffle ({config.N_SHUFFLE_REPEATS} repeats x {n} sequences)...")
    mono_effects = np.zeros((config.N_SHUFFLE_REPEATS, n))
    dinuc_effects = np.zeros((config.N_SHUFFLE_REPEATS, n))
    for r in range(config.N_SHUFFLE_REPEATS):
        mono_seqs = [metrics.mononucleotide_shuffle(s, rng) for s in seqs]
        dinuc_seqs = [metrics.dinucleotide_shuffle(s, rng) for s in seqs]
        mono_effects[r] = baseline_scores - score_sequences(adapter, probe, depth_index, mono_seqs)
        dinuc_effects[r] = baseline_scores - score_sequences(adapter, probe, depth_index, dinuc_seqs)
    mono_per_seq = mono_effects.mean(axis=0)
    dinuc_per_seq = dinuc_effects.mean(axis=0)
    mono_summary = stats.bootstrap_ci(mono_per_seq, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
    dinuc_summary = stats.bootstrap_ci(dinuc_per_seq, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
    mono_vs_dinuc_diff = stats.paired_bootstrap_diff(mono_per_seq, dinuc_per_seq, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
    mono_vs_dinuc_test = stats.wilcoxon_one_sided(mono_per_seq, dinuc_per_seq)
    print(f"  mono effect mean={mono_summary['mean']:.4f}  dinuc effect mean={dinuc_summary['mean']:.4f}")
    print(f"  paired diff (mono - dinuc): mean={mono_vs_dinuc_diff['mean']:.4f} "
          f"CI=({mono_vs_dinuc_diff['ci_low']:.4f}, {mono_vs_dinuc_diff['ci_high']:.4f})")

    # -- 3. regional shuffle ---------------------------------------------------------
    window_starts = list(range(0, config.MAX_SEQ_LEN - REGIONAL_WINDOW + 1, REGIONAL_STRIDE))
    print(f"Regional shuffle ({len(window_starts)} windows x {REGIONAL_SHUFFLE_REPEATS} repeats x {n} sequences)...")
    regional_effects: dict[int, list[float]] = {ws: [] for ws in window_starts}
    for ws in window_starts:
        we = ws + REGIONAL_WINDOW
        per_repeat = np.zeros((REGIONAL_SHUFFLE_REPEATS, n))
        for r in range(REGIONAL_SHUFFLE_REPEATS):
            shuffled = [metrics.regional_shuffle(s, ws, we, rng) for s in seqs]
            per_repeat[r] = baseline_scores - score_sequences(adapter, probe, depth_index, shuffled)
        regional_effects[ws] = per_repeat.mean(axis=0).tolist()
    regional_summary = {
        str(ws): stats.bootstrap_ci(np.array(vals), n_resamples=config.N_BOOTSTRAP, seed=config.SEED)
        for ws, vals in regional_effects.items()
    }
    print("  done")

    # -- results ----------------------------------------------------------------------
    results = {
        "meta": {
            "phase": "02_behavioral_localization",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "readout_depth_index": depth_index,
            "readout_depth_label": depth_label,
            "cohort_size": n,
        },
        "circular_shift_summary": shift_summary,
        "mono_shuffle_summary": mono_summary,
        "dinucleotide_shuffle_summary": dinuc_summary,
        "mono_vs_dinucleotide_diff": mono_vs_dinuc_diff,
        "mono_vs_dinucleotide_wilcoxon": mono_vs_dinuc_test,
        "regional_shuffle_window_starts": window_starts,
        "regional_shuffle_window_width": REGIONAL_WINDOW,
        "regional_shuffle_summary": regional_summary,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_PATH}")

    np.savez_compressed(
        config.RESULTS_DIR / "phase02_circular_shift_raw.npz",
        **{f"shift_{s}": np.array(v) for s, v in shift_effects.items()},
    )

    # -- visualization ------------------------------------------------------------
    means = [shift_summary[str(s)]["mean"] for s in CIRCULAR_SHIFTS]
    los = [shift_summary[str(s)]["ci_low"] for s in CIRCULAR_SHIFTS]
    his = [shift_summary[str(s)]["ci_high"] for s in CIRCULAR_SHIFTS]
    fig, ax = viz.line_plot(
        CIRCULAR_SHIFTS,
        {"mean confidence drop": means},
        bands={"mean confidence drop": (los, his)},
        title="NTv3 8M: circular-shift positional sensitivity",
        xlabel="shift offset (nt)",
        ylabel="P(promoter) drop vs. unshifted",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase02_circular_shift.png")

    fig, ax = viz.errorbar_plot(
        ["mononucleotide\n(full) shuffle", "dinucleotide-preserving\nshuffle"],
        [mono_summary["mean"], dinuc_summary["mean"]],
        [mono_summary["ci_low"], dinuc_summary["ci_low"]],
        [mono_summary["ci_high"], dinuc_summary["ci_high"]],
        title="NTv3 8M: corruption effect by shuffle order",
        ylabel="mean P(promoter) drop vs. baseline",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase02_corruption_comparison.png")

    r_means = [regional_summary[str(ws)]["mean"] for ws in window_starts]
    r_los = [regional_summary[str(ws)]["ci_low"] for ws in window_starts]
    r_his = [regional_summary[str(ws)]["ci_high"] for ws in window_starts]
    fig, ax = viz.line_plot(
        window_starts,
        {"mean confidence drop": r_means},
        bands={"mean confidence drop": (r_los, r_his)},
        title=f"NTv3 8M: regional shuffle sensitivity ({REGIONAL_WINDOW}nt window)",
        xlabel="window start position (nt)",
        ylabel="P(promoter) drop vs. baseline",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase02_regional_shuffle.png")

    print(f"Phase 02 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
