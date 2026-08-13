"""Phase 01 -- Representation baseline.

Port of dnabert/01_Representation_baseline.ipynb to Nucleotide Transformer v3
(8M, ``InstaDeepAI/NTv3_8M_pre``). Replaces the earlier NTv2 phase01
(``ntv2-interpretability/``, removed) -- rewritten, not renamed, because
NTv3's U-Net has no single stack of "encoder layers" to sweep: it has a
7-stage conv downsampling tower, a 2-layer Transformer bottleneck, and a
7-stage deconv upsampling tower (16 ``hidden_states`` entries total; see
``ntv3/model.py``'s docstring for how this was verified against the real
checkpoint).

Question: does NTv3 linearly encode promoter identity at any depth of the
U-Net, and how much of that signal is explainable by trivial
composition/k-mer statistics?

Method (unchanged from the DNABERT-2 original):
  1. composition baseline (A/C/G/T/GC fractions) -> LogisticRegression
  2. k-mer baselines (k=1,2,3) -> StandardScaler + LogisticRegression
  3. NTv3 forward pass, output_hidden_states=True, mean-pool each depth's
     hidden state over the real (unpadded) nucleotide span, mapped down to
     that depth's resolution (``Ntv3Adapter.masked_mean_at_depth``)
  4. one linear probe per U-Net depth (16 probes: conv1..conv7, xformer1-2,
     deconv1..deconv7)
  5. shuffled-label control at the best depth

Deviations from the DNABERT-2 original (documented, not hidden):
  - pilot cohort sizes are smaller (ntv3.config.PILOT_PER_CLASS_*) because
    this runs on CPU-only inference.
  - GUE prom_core_all rows are a fixed 70nt; NTv3 requires input lengths
    that are multiples of 128 (2**num_downsamples), so every sequence is
    center-padded with 'N' to 128nt before tokenizing (ntv3.config.PAD_LENGTH).
    At 128nt the Transformer bottleneck is a single token -- fine for this
    phase (pooling, not attention structure) but tracked as an open question
    for later phases; see ntv3/config.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ntv3 import config, data, metrics, viz
from ntv3.model import Ntv3Adapter

RESULTS_PATH = config.RESULTS_DIR / "phase01_results.json"
PROBE_PATH = config.RESULTS_DIR / "ntv3_best_depth_probe.joblib"
REPR_CACHE = config.RESULTS_DIR / "phase01_representations.npz"


def extract_representations(adapter: Ntv3Adapter, sequences: list[str], batch_size: int = 16):
    """Return array [n_seq, n_depths, hidden] of mean-pooled reps, one column
    per ``hidden_states`` depth (conv tower, Transformer bottleneck, deconv
    tower, in forward order -- see ``ntv3.model.stage_labels``)."""

    all_reps = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        out = adapter.forward(batch, output_hidden_states=True)
        depth_reps = [
            adapter.masked_mean_at_depth(h, batch).cpu().numpy() for h in out.hidden_states
        ]
        all_reps.append(np.stack(depth_reps, axis=1))  # [chunk, n_depths, hidden]
    return np.concatenate(all_reps, axis=0)


def eval_probe(probe, X, y) -> dict:
    pred = probe.predict(X)
    proba = probe.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def main():
    t0 = time.time()
    notes = []

    print("Loading GUE prom_core_all splits...")
    splits = data.load_all_splits()
    pilot = {
        "train": data.balanced_pilot(splits["train"], config.PILOT_PER_CLASS_TRAIN, seed=config.SEED),
        "dev": data.balanced_pilot(splits["dev"], config.PILOT_PER_CLASS_DEV, seed=config.SEED),
        "test": data.balanced_pilot(splits["test"], config.PILOT_PER_CLASS_TEST, seed=config.SEED),
    }
    for name, df in pilot.items():
        print(f"  {name}: {len(df)} rows, class balance {df['label'].value_counts().to_dict()}")
    notes.append(
        f"pilot sizes train={len(pilot['train'])} dev={len(pilot['dev'])} test={len(pilot['test'])} "
        f"(DNABERT-2 original: 2000/800/800; reduced for CPU-only inference)"
    )
    notes.append(
        f"sequences center-padded from 70nt to {config.PAD_LENGTH}nt with 'N' "
        f"(NTv3 requires multiples of 128); Transformer bottleneck is 1 token at this length"
    )

    # -- 1. composition baseline ------------------------------------------------
    print("Composition baseline...")
    comp_train = np.array([
        list(metrics.composition_features(s).values()) for s in pilot["train"]["sequence"]
    ])
    comp_test = np.array([
        list(metrics.composition_features(s).values()) for s in pilot["test"]["sequence"]
    ])
    comp_clf = LogisticRegression(max_iter=1000)
    comp_clf.fit(comp_train, pilot["train"]["label"])
    composition_metrics = eval_probe(comp_clf, comp_test, pilot["test"]["label"])
    print("  composition:", composition_metrics)

    # -- 2. k-mer baselines -------------------------------------------------------
    kmer_results = {}
    for k in (1, 2, 3):
        print(f"k-mer baseline k={k}...")
        X_train, _ = metrics.kmer_features(list(pilot["train"]["sequence"]), k)
        X_test, _ = metrics.kmer_features(list(pilot["test"]["sequence"]), k)
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", max_iter=3000))
        clf.fit(X_train, pilot["train"]["label"])
        kmer_results[f"k{k}"] = eval_probe(clf, X_test, pilot["test"]["label"])
        print(f"  k={k}:", kmer_results[f"k{k}"])

    # -- 3. NTv3 representations --------------------------------------------------
    print("Loading NTv3 adapter (gated model -- requires HF auth, see README.md)...")
    adapter = Ntv3Adapter.from_pretrained()
    meta = adapter.metadata()
    depth_labels = adapter.stage_labels()
    depth_resolutions = adapter.resolution_schedule(config.PAD_LENGTH)
    print(f"  {meta.num_layers} transformer layers, {meta.num_downsamples} downsamples, "
          f"hidden={meta.hidden_size}, heads={meta.num_heads}, params={meta.parameter_count:,}")
    print(f"  depths: {list(zip(depth_labels, depth_resolutions))}")

    reps = {}
    for name in ("train", "dev", "test"):
        print(f"Extracting representations: {name} ({len(pilot[name])} sequences)...")
        reps[name] = extract_representations(adapter, list(pilot[name]["sequence"]))
        print(f"  shape {reps[name].shape}")

    np.savez_compressed(
        REPR_CACHE,
        **{f"{name}_reps": reps[name] for name in reps},
        **{f"{name}_labels": pilot[name]["label"].to_numpy() for name in pilot},
    )

    # -- 4. per-depth probes --------------------------------------------------------
    n_depths = reps["train"].shape[1]
    depth_metrics_dev = []
    depth_metrics_test = []
    probes = {}
    for depth in range(n_depths):
        X_train = reps["train"][:, depth, :]
        X_dev = reps["dev"][:, depth, :]
        X_test = reps["test"][:, depth, :]
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(solver="liblinear", max_iter=3000, random_state=config.SEED)
        )
        clf.fit(X_train, pilot["train"]["label"])
        m_dev = eval_probe(clf, X_dev, pilot["dev"]["label"])
        m_test = eval_probe(clf, X_test, pilot["test"]["label"])
        depth_metrics_dev.append(m_dev)
        depth_metrics_test.append(m_test)
        probes[depth] = clf
        label = f"{depth_labels[depth]}(L={depth_resolutions[depth]})"
        print(f"  depth {label}: dev balanced_acc={m_dev['balanced_accuracy']:.3f} "
              f"roc_auc={m_dev['roc_auc']:.3f} | test balanced_acc={m_test['balanced_accuracy']:.3f} "
              f"roc_auc={m_test['roc_auc']:.3f}")

    best_depth = int(np.argmax([m["roc_auc"] for m in depth_metrics_dev]))
    best_probe = probes[best_depth]
    joblib.dump({"probe": best_probe, "depth": best_depth, "depth_label": depth_labels[best_depth]}, PROBE_PATH)
    # All 16 per-depth probes, not just the best one: phase02's behavioral corruption
    # experiments are readout-position-agnostic (input corruption affects every depth),
    # so the single best-accuracy probe is fine there. Phase03+'s *causal component*
    # ablation experiments are not -- ablating a component can only change a readout
    # that is causally downstream of it in the forward pass, so those phases need the
    # probe at the *final* depth (deconv7, index -1), not whichever depth happened to
    # probe best. Saving every depth's probe lets each phase pick correctly instead of
    # everyone defaulting to `best_depth` out of convenience.
    joblib.dump(
        {"probes": probes, "depth_labels": depth_labels, "depth_resolutions": depth_resolutions},
        config.RESULTS_DIR / "ntv3_all_depth_probes.joblib",
    )
    print(f"Best depth (by dev ROC-AUC): {best_depth} ({depth_labels[best_depth]})")

    # -- 5. shuffled-label control at best depth ------------------------------------
    print(f"Shuffled-label control ({config.N_LABEL_SHUFFLES} repeats) at depth {depth_labels[best_depth]}...")
    rng = np.random.default_rng(config.SEED)
    shuffle_metrics = []
    X_train_best = reps["train"][:, best_depth, :]
    X_test_best = reps["test"][:, best_depth, :]
    y_train = pilot["train"]["label"].to_numpy()
    for i in range(config.N_LABEL_SHUFFLES):
        y_shuffled = rng.permutation(y_train)
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(solver="liblinear", max_iter=3000, random_state=config.SEED + i)
        )
        clf.fit(X_train_best, y_shuffled)
        shuffle_metrics.append(eval_probe(clf, X_test_best, pilot["test"]["label"]))
    shuffle_mean = {
        k: float(np.mean([m[k] for m in shuffle_metrics])) for k in ("accuracy", "balanced_accuracy", "roc_auc")
    }
    print("  shuffled-label mean:", shuffle_mean)

    # -- results ----------------------------------------------------------------------
    results = {
        "meta": {
            "phase": "01_representation_baseline",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "pad_length": config.PAD_LENGTH,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "num_depths": n_depths,
            "depth_labels": depth_labels,
            "depth_resolutions": depth_resolutions,
            "num_transformer_layers": meta.num_layers,
            "num_downsamples": meta.num_downsamples,
            "num_heads": meta.num_heads,
            "hidden_size": meta.hidden_size,
            "parameter_count": meta.parameter_count,
        },
        "composition_baseline": composition_metrics,
        "kmer_baselines": kmer_results,
        "depthwise_dev": depth_metrics_dev,
        "depthwise_test": depth_metrics_test,
        "best_depth": best_depth,
        "best_depth_label": depth_labels[best_depth],
        "shuffled_label_control": shuffle_mean,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    depth_tick_labels = [f"{lbl} (L={res})" for lbl, res in zip(depth_labels, depth_resolutions)]
    fig, ax = viz.line_plot(
        depth_tick_labels,
        {
            "dev balanced accuracy": [m["balanced_accuracy"] for m in depth_metrics_dev],
            "test balanced accuracy": [m["balanced_accuracy"] for m in depth_metrics_test],
        },
        title="NTv3 8M: per-depth promoter probe accuracy (U-Net: conv -> transformer -> deconv)",
        xlabel="hidden_states depth",
        ylabel="balanced accuracy",
        figsize=(11, 5),
    )
    ax.axhline(composition_metrics["balanced_accuracy"], color=viz.INK_MUTED, linestyle=":",
               linewidth=1.5, label="composition baseline")
    bottleneck_idx = depth_resolutions.index(min(depth_resolutions))
    ax.axvline(bottleneck_idx, color=viz.AXIS, linewidth=0.8, linestyle="--")
    ax.legend(frameon=False, fontsize=9, labelcolor=viz.INK_SECONDARY)
    ax.set_xticklabels(depth_tick_labels, rotation=60, ha="right", fontsize=7.5)
    viz.save(fig, config.FIGURES_DIR / "phase01_depthwise_accuracy.png")

    fig, ax = viz.bar_plot(
        ["composition", "1-mer", "2-mer", "3-mer", f"NTv3 best ({depth_labels[best_depth]})"],
        [
            composition_metrics["balanced_accuracy"],
            kmer_results["k1"]["balanced_accuracy"],
            kmer_results["k2"]["balanced_accuracy"],
            kmer_results["k3"]["balanced_accuracy"],
            depth_metrics_test[best_depth]["balanced_accuracy"],
        ],
        title="NTv3 8M best depth vs. composition / k-mer baselines",
        ylabel="test balanced accuracy",
    )
    viz.save(fig, config.FIGURES_DIR / "phase01_baseline_comparison.png")

    print(f"Phase 01 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
