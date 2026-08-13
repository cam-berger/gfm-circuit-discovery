"""Phase 05 -- Biological feature characterization of the transferable
causal signal.

Port of dnabert/05_promoter_mlp_feature_characterization.ipynb to
Nucleotide Transformer v3 (8M). The DNABERT-2 original follows up on
Notebook 4's headline finding (the Layer-1 MLP carries a *transferable
promoter-class* signal: unrelated promoter donors rescue about as well as
matched donors, non-promoter donors don't rescue at all) by asking what
sequence features actually drive that transfer. This phase asks the same
question about phase04's analogous finding: ``deconv7_main`` -- the final
deconv-tower stage, right before the output head -- showed matched and
unrelated promoter-donor recovery within noise of each other and well above
non-promoter donor recovery (see phase04's results and README.md).

Main questions (unchanged):
  1. Which sequence features distinguish promoter donors from non-promoter
     donors?
  2. Which features predict this component's rescue magnitude?
  3. Does a donor's biological feature profile predict rescue *beyond* its
     own final-depth probe score (the obvious confound: promoter donors
     already score high on the readout probe before any patching happens)?

This phase reconstructs phase04's exact cohort, corruption, and donor pool
from the same seeds (no new artifacts saved by phase04 beyond its results
JSON) rather than depending on intermediate per-sequence data it didn't
persist -- consistent with phase02/03/04 each independently regenerating
their cohort from ``config.SEED``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ntv3 import config, data, metrics, stats, viz
from ntv3.model import Ntv3Adapter
from ntv3.patching import capture_activation, inject_activation
from phase03_component_localization import build_component_registry
from phase04_causal_interventions import (
    MINIMUM_RESCUE_GAP,
    capture_component_for_cohort,
    load_final_depth_probe,
    no_fixed_point_derangement,
    score_sequences,
    score_with_injected_cohort,
)

RESULTS_PATH = config.RESULTS_DIR / "phase05_results.json"
TARGET_COMPONENT = "deconv7_main"  # phase04's clearest transferable-signal carrier
N_CV_FOLDS = 5


def feature_row(seq: str, probe_score: float) -> dict:
    row = metrics.full_feature_row(seq)
    row["probe_score"] = probe_score
    return row


def main():
    t0 = time.time()
    notes = [f"target component: {TARGET_COMPONENT} (phase04's clearest transferable-signal carrier)"]

    print("Loading final-depth probe...")
    probe, depth_index, depth_label = load_final_depth_probe()

    print("Reconstructing phase04's cohort, corruption, and donor pool (same seeds)...")
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

    rescue_mask = gaps_all >= MINIMUM_RESCUE_GAP
    clean_seqs = [s for s, m in zip(clean_seqs_all, rescue_mask) if m]
    corrupt_seqs = [s for s, m in zip(corrupt_seqs_all, rescue_mask) if m]
    corrupt_scores = corrupt_scores_all[rescue_mask]
    gaps = gaps_all[rescue_mask]
    n = len(clean_seqs)
    print(f"  rescue cohort: {n} (matches phase04)")

    unrelated_idx = no_fixed_point_derangement(n, config.SEED + 1)
    unrelated_donor_seqs = [clean_seqs[i] for i in unrelated_idx]
    unrelated_donor_scores = clean_scores_all[rescue_mask][unrelated_idx]

    train_df = data.load_split("train")
    negatives = train_df[train_df["label"] == 0]
    nonpromoter_donors = negatives.sample(n=n, replace=False, random_state=config.SEED + 2).reset_index(drop=True)
    nonpromoter_seqs = list(nonpromoter_donors["sequence"])
    nonpromoter_scores = score_sequences(adapter, probe, depth_index, nonpromoter_seqs)

    registry = {e["name"]: e for e in build_component_registry(adapter)}
    module = registry[TARGET_COMPONENT]["module"]

    clean_act = capture_component_for_cohort(adapter, module, clean_seqs)
    unrelated_act = clean_act[unrelated_idx]
    nonpromoter_act = capture_component_for_cohort(adapter, module, nonpromoter_seqs)

    unrelated_patched = score_with_injected_cohort(adapter, probe, depth_index, module, corrupt_seqs, unrelated_act)
    nonpromoter_patched = score_with_injected_cohort(adapter, probe, depth_index, module, corrupt_seqs, nonpromoter_act)

    unrelated_recovery = (unrelated_patched - corrupt_scores) / gaps
    nonpromoter_recovery = (nonpromoter_patched - corrupt_scores) / gaps

    # -- combined donor pool: every donor that was ever patched into a recipient --
    donor_rows = []
    for seq, score, rec in zip(unrelated_donor_seqs, unrelated_donor_scores, unrelated_recovery):
        row = feature_row(seq, float(score))
        row["donor_class"] = "promoter"
        row["recovery_fraction"] = float(rec)
        donor_rows.append(row)
    for seq, score, rec in zip(nonpromoter_seqs, nonpromoter_scores, nonpromoter_recovery):
        row = feature_row(seq, float(score))
        row["donor_class"] = "non-promoter"
        row["recovery_fraction"] = float(rec)
        donor_rows.append(row)
    donor_df = pd.DataFrame(donor_rows)
    print(f"  combined donor pool: {len(donor_df)} donors ({(donor_df['donor_class']=='promoter').sum()} promoter, "
          f"{(donor_df['donor_class']=='non-promoter').sum()} non-promoter)")
    notes.append(f"donor pool n={len(donor_df)} (unrelated-promoter + non-promoter donors from phase04's reconstruction)")

    feature_cols = [c for c in donor_df.columns if c not in ("donor_class", "recovery_fraction")]

    # -- 1. which features distinguish promoter vs non-promoter donors? --------------
    print("\nFeature differences: promoter vs non-promoter donors...")
    class_diff_rows = []
    promoter_mask = donor_df["donor_class"] == "promoter"
    for feat in feature_cols:
        a = donor_df.loc[promoter_mask, feat].to_numpy(dtype=float)
        b = donor_df.loc[~promoter_mask, feat].to_numpy(dtype=float)
        diff = stats.bootstrap_ci(a, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)["mean"] - \
               stats.bootstrap_ci(b, n_resamples=config.N_BOOTSTRAP, seed=config.SEED)["mean"]
        class_diff_rows.append({"feature": feat, "promoter_mean": float(a.mean()),
                                 "nonpromoter_mean": float(b.mean()), "diff": float(diff)})
    class_diff_df = pd.DataFrame(class_diff_rows).sort_values("diff", key=abs, ascending=False)
    print(class_diff_df.head(8).to_string(index=False))

    # -- 2. which features correlate with rescue magnitude? --------------------------
    print("\nFeature correlation with recovery_fraction (combined donor pool)...")
    corr_rows = []
    y = donor_df["recovery_fraction"].to_numpy()
    for feat in feature_cols:
        x = donor_df[feat].to_numpy(dtype=float)
        if np.std(x) == 0:
            continue
        sp = stats.spearman(x, y)
        corr_rows.append({"feature": feat, "spearman_rho": sp["rho"], "spearman_p": sp["p_value"]})
    corr_df = pd.DataFrame(corr_rows).sort_values("spearman_rho", key=abs, ascending=False)
    print(corr_df.head(8).to_string(index=False))

    # -- 3. cross-validated: do biological features predict rescue beyond probe_score? --
    print(f"\nCross-validated ({N_CV_FOLDS}-fold) rescue prediction: probe-score-only vs +GC vs full features...")
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=config.SEED)
    y = donor_df["recovery_fraction"].to_numpy()

    models = {
        "probe_score_only": ["probe_score"],
        "probe_score_plus_gc": ["probe_score", "gc_fraction"],
        "full_biological_features": feature_cols,
    }
    cv_results = {}
    for model_name, cols in models.items():
        X = donor_df[cols].to_numpy(dtype=float)
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=config.SEED))
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        cv_results[model_name] = {"mean_r2": float(scores.mean()), "fold_r2": scores.tolist()}
        print(f"  {model_name:28s} mean R^2 = {scores.mean():+.3f}  folds={np.round(scores, 3).tolist()}")

    # -- results ----------------------------------------------------------------------
    results = {
        "meta": {
            "phase": "05_component_feature_characterization",
            "model": config.MODEL_NAME,
            "seed": config.SEED,
            "runtime_seconds": time.time() - t0,
            "notes": notes,
            "target_component": TARGET_COMPONENT,
            "readout_depth_index": depth_index,
            "readout_depth_label": depth_label,
            "donor_pool_size": len(donor_df),
            "n_cv_folds": N_CV_FOLDS,
        },
        "class_feature_differences": class_diff_df.to_dict(orient="records"),
        "rescue_correlations": corr_df.to_dict(orient="records"),
        "cv_model_comparison": cv_results,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    donor_df.to_csv(config.RESULTS_DIR / "phase05_donor_pool.csv", index=False)
    print(f"\nWrote {RESULTS_PATH}")

    # -- visualization ------------------------------------------------------------
    top_corr = corr_df.head(10)
    fig, ax = viz.bar_plot(
        top_corr["feature"].tolist(), top_corr["spearman_rho"].tolist(),
        title=f"NTv3 8M: features most correlated with {TARGET_COMPONENT} rescue",
        ylabel="Spearman rho vs. recovery_fraction",
        figsize=(9, 5),
    )
    viz.save(fig, config.FIGURES_DIR / "phase05_rescue_correlations.png")

    model_names = list(cv_results.keys())
    model_r2 = [cv_results[m]["mean_r2"] for m in model_names]
    fig, ax = viz.bar_plot(
        [m.replace("_", "\n") for m in model_names], model_r2,
        title=f"NTv3 8M: {N_CV_FOLDS}-fold CV rescue prediction by feature set",
        ylabel="mean cross-validated R^2",
    )
    ax.axhline(0, color=viz.AXIS, linewidth=1, linestyle="--")
    viz.save(fig, config.FIGURES_DIR / "phase05_cv_model_comparison.png")

    print(f"Phase 05 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
