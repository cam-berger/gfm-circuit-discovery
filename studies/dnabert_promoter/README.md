# DNABERT-2 promoter mechanism — study README

The second real GFM in this repository, and the first test of whether the method travels. The question
is **not** whether DNABERT-2 can classify promoters — it is whether the discipline built on SpliceBERT
(prereg → intervention → bounded claim, with evidence that never promotes itself) recovers a causally
supported account in a different model, on a different behaviour, with a different tokenizer.

**Start with [`CLAIMS.md`](CLAIMS.md), then [`CORRECTIONS.md`](CORRECTIONS.md).** Six of this arc's
original headline sentences are bounded or withdrawn; reading the prose without the corrections will
give you a stronger impression than the artifacts support. `CORRECTIONS.md` also records two findings
this review raised and then refuted — the arc's data hygiene and its direction handling are both sound.

## Status

| Aspect | Where it stands |
|---|---|
| Data hygiene | Leakage-controlled splits, disjointness asserted; near-duplicates unscreened (DBP2) |
| Representation | Established on the cleaned splits (DBP3) |
| Endpoint choice | Working choice, `0.0019` single-seed margin — not a localization result (DBP4) |
| Component necessity | Candidate ranking only; no calibration pair, one seed (DBP5) |
| Conditional causal core | Established and test-confirmed, conditional and non-additive (DBP6) |
| Compactness | **Refuted at the sizes tested** — no knee, MLP-dominated (DBP7) |
| Example-specific delivery | **Null** — transfer is class-level (DBP8) |
| Biological mechanism | Positional enrichment only; causal test is a null at n=71 (DBP9, DBP10) |
| Preregistration | **None frozen.** Validation-selected and test-confirmed, but not preregistered (DBPR8) |
| Fine-tuned analysis | Completed artifact package: predictive evaluation, layer probing, early-component ablation/restoration, activation-source compatibility, and TATA perturbation (DBP11–DBP16). Cross-state interpretation remains descriptive because the frozen and fine-tuned assays are not matched (DBP17, DBPR10). |

## The one-paragraph account

A frozen DNABERT-2 encoder carries promoter-discriminative information that a masked-mean linear probe
recovers at an early encoder layer without any fine-tuning, comfortably above a k-mer baseline, on splits
from which every exact cross-split duplicate has been removed. Zeroing and patching early components show that this information is not spread uniformly: a five-node set —
heads `L0H8`, `L1H8`, `L2H9`, `L2H3` and `MLP0` — destroys a majority of the corruption effect when
knocked out and restores most of it when patched clean, on an untouched test cohort. But that set is
**not a compact circuit**: standing alone it reproduces under a fifth of full-model behaviour, and the
faithfulness curve reaches 1.0 only once all three early MLPs are included, with no knee anywhere. What
the transplant moves is a **class label's worth of evidence, not a sequence-specific payload** — an
unrelated promoter's activations work as well as the matched one's. And the biological bridge is
unbuilt: TATA-like motifs are positionally enriched in the data, but disrupting them does not measurably
damage the readout beyond a matched control.

So: a real representation, a real conditional causal core, a distributed and MLP-dominated computation,
an unresolved payload semantics, and no biological mechanism. That shape will be familiar — it is what
the SpliceBERT donor arc converged on at **C33**, reached independently in a second model. Reproducing
the *shape of the limit* is the transferable result here; none of the SpliceBERT numbers transfer, and
no `C*`, `HCA*` or `GFS*` claim depends on anything in this study.

## What this study may not be quoted for

- "Layer 2 is the promoter layer" — layers 1–4 are not separated by the data (DBP4).
- "Compact", "sparse", or bare "the circuit" — refuted at the sizes tested (DBP7, DBPR4).
- "Transferable", "sequence-specific delivery" — null contrast (DBP8, DBPR5).
- A TATA-box mechanism — the causal test is a null (DBP10).
- The old seed-42 self-fine-tuning smoke metrics withdrawn under DBPR6 — they remain withdrawn and are not revived by the later fine-tuned package.
- A causal statement that fine-tuning *caused* an L2→L10 relocation or a +0.125 BA gain — the frozen and fine-tuned studies use different GUE promoter configurations, sequence lengths, readouts, and cohort policies (DBP17, DBPR10).
- Anything from `results/dnabert_promoter/causal/layer3_*` — different endpoint, superseded (DBPR7).

## Layout

| Path | What is there |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | The `DBP1`–`DBP17` ledger. The point of contact for every claim. |
| [`CORRECTIONS.md`](CORRECTIONS.md) | `DBPR1`–`DBPR11`. What was withdrawn or bounded, why — including the fine-tuning artifact update, the unmatched-comparison boundary, and the post-merge remediation that bounded the Layer-10 endpoint to a band. |
| [`DECISIONS.md`](DECISIONS.md) | Why the assays are shaped the way they are, including known departures. |
| [`FROZEN_EXPERIMENT_REGISTRY.md`](FROZEN_EXPERIMENT_REGISTRY.md) | The frozen arc's working registry of assay definitions, metric formulas, and per-block status. Not a claim record. |
| [`FINETUNED_VS_FROZEN.md`](FINETUNED_VS_FROZEN.md) | Concise record of the completed fine-tuned assays and the licensed descriptive comparison to the frozen study. |
| [`results/dnabert_promoter/`](../../results/dnabert_promoter/RESULTS_INDEX.md) | Artifacts, with the family index naming which control claims. |
| `scripts/dnabert_promoter/` | 36 runners: cohort building, probing, ablation, restoration, faithfulness, motif tests, and `analyze_finetuned_layer_selection.py`. |
| `notebooks/dnabert_finetuned_experiments.ipynb` | The producer of the 52 fine-tuned artifacts. **Not a runner** — it has no manifests and is not re-runnable from a clean checkout. Porting it to `scripts/dnabert_promoter/` is the gating item for the fine-tuned rows (`DBPR11`). |
| `src/gfmcd/adapters/dnabert.py`, `src/gfmcd/dnabert/` | Model access, dataset, perturbation utilities. |
| `configs/dnabert_promoter/base.json` | The frozen assay configuration. |
| `figures/dnabert_promoter/` | Published figures with captions. Figures 1–3 and 9–12 are canonical; 4–8 depict the superseded Layer-3 replicate. |
| `tests/test_dnabert_{adapter,dataset,perturb,manifest}.py` | Unit coverage for the adapter, dataset, perturbations, and the sibling-manifest contract. |

## Fine-tuned extension

A completed fine-tuned DNABERT-2 analysis is now part of this study family. It evaluates
`magicslabnu/DNABERT-2-finetuned-prom_300_all` (revision
`7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12`) using the compatible
`quietflamingo/dnabert2-no-flashattention` implementation at revision
`813031b2bf86d9e960a027e2734c908009f31601`.

On the official GUE `prom_300_all` test split, the fine-tuned classifier reaches balanced accuracy
`0.935205`, MCC `0.870487`, and ROC-AUC `0.984785`; excluding exact test sequences found in train or
development leaves `n=5,900` and balanced accuracy `0.935136`. A masked-mean linear probe sweep returns
Layer 10 under the stated development rule (`dev BA = 0.942919`; descriptive test BA `0.932418`), but that
argmax leads its runner-up by `0.33` standard errors: layers **7–11** are within one standard error and
**5–11** within two, and the test argmax is Layer 8. Quote the band, not Layer 10 — and read it as
representational accessibility, not causal localization (`DBP13`, `DBPR11`).

On a locked balanced 256-example test cohort, zeroing `L0H8` lowers balanced accuracy
`0.957031 → 0.511719`; zeroing `L0_MLP` gives `0.816406`; ablating both gives `0.503906`.
Restoring both clean activations after joint ablation reproduces the baseline logits exactly. Activation-source
swaps are much more disruptive across classes than within class, especially for `L0_MLP` and the combined
pathway. A paired TATA-like motif perturbation on 354 eligible promoters gives mean excess motif damage
`-0.021874` (`p=0.162760`), so no positive TATA-specific dependence is licensed.

**Comparison boundary.** These results do **not** form a controlled before/after fine-tuning experiment
with the frozen arc: the frozen canonical study uses GUE `prom_core_all` at 70 nt on leakage-cleaned
splits and a fixed Layer-2 probe, whereas the fine-tuned package uses GUE `prom_300_all` at 300 nt and
the checkpoint's classifier for the primary behavioural evaluation. The two states may be compared
descriptively, but differences may not be attributed uniquely to fine-tuning until a matched assay is run
(DBP17, DBPR10). See [`FINETUNED_VS_FROZEN.md`](FINETUNED_VS_FROZEN.md).

## The instrument, precisely

- **Model** `quietflamingo/dnabert2-no-flashattention`, revision `813031b2bf86d9e960a027e2734c908009f31601`.
- **Hidden-state contract** the adapter returns exactly the **12 encoder-layer outputs and no embedding
  output**, so `hidden_states[i]` *is* encoder layer `i` with no offset. This is asserted in
  `src/gfmcd/adapters/dnabert.py` and is why layer indices here need no `+1` correction — do not carry
  that assumption to another adapter.
- **Readout** masked mean over biological tokens at encoder Layer 2 → fixed linear probe
  `selected_layer_2_probe.joblib`, sha256 `bd0d9e14e1935e3ebd8e2ce638ccd5a721feb9dc9a0895447d9b62080ce98b3b`.
  The same artifact is used for validation and test.
- **Data** `leannmlindsey/GUE`, config `prom_core_all`; 70-nt sequences. Raw split 47,356 / 5,920 / 5,920
  with hashes in `results/dnabert_promoter/shared/split_manifest.json`; every result is computed on the
  **leakage-controlled** splits of 47,149 / 5,856 / 5,865 produced by `clean_promoter_cohort.py`, which
  asserts the retained hash sets are pairwise disjoint (DBP2).
- **Seed** 42 throughout. One seed. Every effect in this study is single-seed.
- **Identity controls** exactly zero in every committed causal family.

## Running it

Canonical environment is `uv`, from the repository root, as everywhere else in this repo —
[`REPRODUCE.md`](../../REPRODUCE.md) has the full contract. This study has its own extra, `dnabert`; pass
it rather than relying on `splicebert`, which is only where `scikit-learn` happened to live before the
extra existed.

```bash
uv run --extra dnabert python scripts/dnabert_promoter/run_environment_preflight.py
uv run --extra dnabert python scripts/dnabert_promoter/build_promoter_cohort.py      # writes shared/ + leakage_report.json
uv run --extra dnabert python scripts/dnabert_promoter/extract_frozen_representations.py
uv run --extra dnabert python scripts/dnabert_promoter/run_frozen_layer_probes.py    # selects the endpoint
uv run --extra dnabert python scripts/dnabert_promoter/run_sequence_baselines.py
uv run --extra dnabert python scripts/dnabert_promoter/run_frozen_circuit_accounting.py
uv run --extra dnabert python scripts/dnabert_promoter/run_frozen_faithfulness_curve.py
uv run --extra dnabert python scripts/dnabert_promoter/run_component_donor_matrix.py
uv run --extra dnabert python scripts/dnabert_promoter/run_tata_perturbation_probe.py
uv run --extra dnabert python scripts/dnabert_promoter/plot_frozen_figures.py
```

Fix these within a claim: checkpoint, tokenizer, dataset version and split hashes, sequence length,
pooling and readout definition, intervention definition, seed, metric.

**Known gap:** only 2 of the 35 runners currently write a sibling `*.manifest.json`, against the
repository's invariant that every result carries one. Emitting manifests from the remaining runners —
using the existing helper in `src/gfmcd/splicing/io.py` — is a prerequisite for promoting any `DBP` row,
and is tracked in [`docs/ROADMAP.md`](../../docs/ROADMAP.md) and `DECISIONS.md`.

## What would move this study forward

In dependency order, each step gated on the one before:

1. **Commit the cleaning stage's provenance** — `shared/clean/cleaning_report.json` and the cleaned split
   hashes — so a reader can verify the leakage control without re-running it, and add a near-duplicate
   screen, which exact-hash matching does not provide at 70 nt.
2. **Multiple seeds.** Every effect here is single-seed; the endpoint margin is `0.0019`. Effect-existence
   needs seeded draws, not a within-split held-out set.
3. **Calibration controls** — planted-positive sensitivity and permuted-label FPR — for the head ranking,
   the group selection, and the transplant matrix, which is what would turn DBP5 from a candidate ranking
   into a necessity claim.
4. **Freeze a preregistration** before the next confirmatory run, with memorials that can fail and sit on
   the measurement path.
5. **Extend manifest emission** to the remaining runners. The 15 canonical ones now write a sibling
   `*.manifest.json` via `gfmcd.dnabert.write_run_manifest`; the development and superseded runners do not,
   and the ~104 pre-existing artifacts fill in only when their assay is re-run.
6. **Power the `TATAAA` cohort** (n=48 is the only stratum worth the attempt) with held-out confirmation
   and motif-specific component localization.
7. **Matched frozen-versus-fine-tuned comparison.** The fine-tuned artifact package now exists, but it uses GUE `prom_300_all` at 300 nt while the canonical frozen arc uses `prom_core_all` at 70 nt with a different readout and cleaning policy. A matched assay is required before attributing performance or layer-profile differences to fine-tuning itself.
