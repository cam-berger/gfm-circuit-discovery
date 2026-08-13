"""Phase 07 -- Low-rank subspace analysis.

Port of dnabert/07_low_rank_subspace_analysis.ipynb to Nucleotide
Transformer v3 (8M). Phase06 found a single rank-1 direction (promoter-mean
minus non-promoter-mean pooled activation) in ``deconv7_main`` whose
projection generalized to held-out sequences (rho=0.83 with achieved GC
shift). This phase asks whether that's the right amount of structure, or
whether the causal signal actually needs more (or less) than one dimension:

  1. **Matched delta vectors**: for each clean/dinucleotide-corrupted
     promoter pair (same cohort and corruption as phase04),
     ``delta = pooled_activation(clean) - pooled_activation(corrupt)`` at
     ``deconv7_main``.
  2. **Source-averaged SVD and effective rank**: SVD of the delta matrix;
     effective rank via the participation ratio
     ``(sum(s)**2) / sum(s**2)`` (a soft rank that penalizes spread-out
     singular values, not just a hard variance-explained cutoff).
  3. **Cross-validated held-out subspace reconstruction**: fit the SVD
     basis on a train fold only, reconstruct held-out delta vectors from
     their projection onto the top-r train-fold components, and measure
     reconstruction quality (R^2) as a function of r.
  4. **Predict downstream probe effects from held-out low-rank
     coordinates**: regress each held-out pair's clean-corrupt score gap
     on its top-r projection coordinates (fit on train folds only), report
     cross-validated R^2 vs. r -- this is phase06's rank-1 direction
     result, generalized to rank r and evaluated honestly out-of-fold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ntv3 import config, data, metrics, viz
from ntv3.model import Ntv3Adapter
from ntv3.patching import capture_activation
from phase03_component_localization import build_component_registry
from phase04_causal_interventions import load_final_depth_probe, score_sequences

RESULTS_PATH = config.RESULTS_DIR / "phase07_results.json"
TARGET_COMPONENT = "deconv7_main"
N_CV_FOLDS = 5
RANKS_TO_TEST = [1, 2, 4, 8, 16, 32]


def pooled_activation_batched(adapter: Ntv3Adapter, module, seqs: list[str], batch_size: int = 32) -> np.ndarray:
    pieces = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        batch = adapter.tokenize(chunk, pad_length=config.PAD_LENGTH)
        act = capture_activation(module, lambda b=batch: adapter.forward(b, output_hidden_states=False))
        act = act.permute(0, 2, 1)  # (B, C, L) -> (B, L, C), matches phase06
        pieces.append(adapter.masked_mean_at_depth(act, batch).cpu().numpy())
    return np.concatenate(pieces, axis=0)


def main():
    t0 = time.time()
    notes = [f"target component: {TARGET_COMPONENT}"]

    print("Loading final-depth probe...")
    probe, depth_index, depth_label = load_final_depth_probe()

    print("Reconstructing phase04's clean/corrupt cohort (same seeds)...")
    test_df = data.load_split("test")
    positives = test_df[test_df["label"] == 1]
    n_cohort = min(config.CAUSAL_COHORT_PER_CLASS, len(positives))
    cohort = positives.sample(n=n_cohort, replace=False, random_state=config.SEED).reset_index(drop=True)
    clean_seqs = list(cohort["sequence"])

    rng = np.random.default_rng(config.SEED)
    corrupt_seqs = [metrics.dinucleotide_shuffle(s, rng) for s in clean_seqs]
    n = len(clean_seqs)
    print(f"  {n} clean/corrupt pairs (full causal cohort, not gap-filtered -- this phase doesn't divide by the gap)")

    print("Loading NTv3 adapter...")
    adapter = Ntv3Adapter.from_pretrained()

    clean_scores = score_sequences(adapter, probe, depth_index, clean_seqs)
    corrupt_scores = score_sequences(adapter, probe, depth_index, corrupt_seqs)
    gaps = clean_scores - corrupt_scores

    registry = {e["name"]: e for e in build_component_registry(adapter)}
    module = registry[TARGET_COMPONENT]["module"]

    print(f"Capturing pooled {TARGET_COMPONENT} activations for clean and corrupt runs...")
    clean_act = pooled_activation_batched(adapter, module, clean_seqs)
    corrupt_act = pooled_activation_batched(adapter, module, corrupt_seqs)
    delta = clean_act - corrupt_act  # [n, hidden]
    print(f"  delta matrix shape: {delta.shape}")

    # -- 1. source-averaged SVD and effective rank ------------------------------------
    print("\nSVD of the delta matrix...")
    delta_centered = delta - delta.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(delta_centered, full_matrices=False)
    participation_ratio = float((S.sum() ** 2) / (S ** 2).sum())
    cumulative_var = np.cumsum(S ** 2) / np.sum(S ** 2)
    rank_for_90pct = int(np.searchsorted(cumulative_var, 0.90) + 1)
    print(f"  {len(S)} singular values; participation-ratio effective rank = {participation_ratio:.2f}")
    print(f"  rank needed for 90% variance explained = {rank_for_90pct}")
    print(f"  top-5 singular values: {np.round(S[:5], 3).tolist()}")

    # -- 2 & 3. cross-validated held-out reconstruction + gap prediction, by rank -----
    print(f"\nCross-validated ({N_CV_FOLDS}-fold) held-out reconstruction and gap prediction by rank...")
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=config.SEED)
    rank_results = []
    for r in RANKS_TO_TEST:
        if r > n - (n // N_CV_FOLDS) - 1:
            print(f"  rank {r}: skipped (exceeds train-fold sample size for n={n})")
            continue
        recon_r2_folds = []
        gap_r2_folds = []
        for train_idx, test_idx in kf.split(delta):
            train_delta = delta[train_idx]
            train_mean = train_delta.mean(axis=0, keepdims=True)
            Ut, St, Vtt = np.linalg.svd(train_delta - train_mean, full_matrices=False)
            basis = Vtt[:r]  # [r, hidden]

            test_delta = delta[test_idx]
            test_centered = test_delta - train_mean
            coords = test_centered @ basis.T  # [n_test, r]
            reconstructed = coords @ basis + train_mean
            ss_res = ((test_delta - reconstructed) ** 2).sum()
            ss_tot = ((test_delta - test_delta.mean(axis=0, keepdims=True)) ** 2).sum()
            recon_r2_folds.append(float(1 - ss_res / ss_tot))

            pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=config.SEED))
            train_coords = (train_delta - train_mean) @ basis.T
            pipe.fit(train_coords, gaps[train_idx])
            gap_pred = pipe.predict(coords)
            gap_ss_res = ((gaps[test_idx] - gap_pred) ** 2).sum()
            gap_ss_tot = ((gaps[test_idx] - gaps[train_idx].mean()) ** 2).sum()
            gap_r2_folds.append(float(1 - gap_ss_res / gap_ss_tot))

        rank_results.append({
            "rank": r,
            "mean_reconstruction_r2": float(np.mean(recon_r2_folds)),
            "mean_gap_prediction_r2": float(np.mean(gap_r2_folds)),
            "reconstruction_r2_folds": recon_r2_folds,
            "gap_prediction_r2_folds": gap_r2_folds,
        })
        print(f"  rank {r:3d}: held-out reconstruction R^2={np.mean(recon_r2_folds):+.3f}  "
              f"held-out gap-prediction R^2={np.mean(gap_r2_folds):+.3f}")

    # -- results ----------------------------------------------------------------------
    results = {
        "meta": {
            "phase": "07_low_rank_subspace_analysis",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "target_component": TARGET_COMPONENT,
            "readout_depth_index": depth_index,
            "readout_depth_label": depth_label,
            "n_pairs": n,
            "hidden_dim": int(delta.shape[1]),
        },
        "singular_values": S.tolist(),
        "participation_ratio_effective_rank": participation_ratio,
        "rank_for_90pct_variance": rank_for_90pct,
        "rank_cv_results": rank_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    np.savez_compressed(config.RESULTS_DIR / "phase07_svd_basis.npz", U=U, S=S, Vt=Vt, delta_mean=delta.mean(axis=0))
    print(f"\nWrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    fig, ax = viz.line_plot(
        list(range(1, len(S) + 1)), {"cumulative variance explained": cumulative_var.tolist()},
        title=f"NTv3 8M: {TARGET_COMPONENT} delta-vector SVD spectrum",
        xlabel="rank", ylabel="cumulative variance explained",
    )
    ax.axhline(0.90, color=viz.AXIS, linewidth=1, linestyle="--")
    ax.axvline(participation_ratio, color=viz.CATEGORICAL[1], linewidth=1.5, linestyle=":",
               label=f"participation-ratio rank ({participation_ratio:.1f})")
    ax.legend(frameon=False, fontsize=9, labelcolor=viz.INK_SECONDARY)
    viz.save(fig, config.FIGURES_DIR / "phase07_svd_spectrum.png")

    ranks = [r["rank"] for r in rank_results]
    recon_r2 = [r["mean_reconstruction_r2"] for r in rank_results]
    gap_r2 = [r["mean_gap_prediction_r2"] for r in rank_results]
    fig, ax = viz.line_plot(
        ranks, {"held-out reconstruction R^2": recon_r2, "held-out gap-prediction R^2": gap_r2},
        title=f"NTv3 8M: {N_CV_FOLDS}-fold CV subspace quality vs. rank",
        xlabel="subspace rank", ylabel="R^2",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase07_rank_cv_curves.png")

    print(f"Phase 07 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
