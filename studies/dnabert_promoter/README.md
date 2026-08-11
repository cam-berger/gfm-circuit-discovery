# DNABERT-2 promoter mechanism — study README

> **Tier: T1 — canonical.** Current, authoritative. Read this.

The second real GFM in this repository, and the first test of whether the method travels. The question
is **not** whether DNABERT-2 can classify promoters — it is whether the discipline built on SpliceBERT
(prereg → intervention → bounded claim, with evidence that never promotes itself) recovers a causally
supported account in a different model, on a different behaviour, with a different tokenizer.

**Start with [`CLAIMS.md`](CLAIMS.md), then [`CORRECTIONS.md`](CORRECTIONS.md).** Eight of this arc's
original headline sentences are bounded or withdrawn; reading the prose without the corrections will
give you a stronger impression than the artifacts support.

## Status

| Aspect | Where it stands |
|---|---|
| Representation | Established, but bounded by a contaminated split (DBP2, DBP3) |
| Endpoint choice | Working choice, `0.0019` single-seed margin — not a localization result (DBP4) |
| Component necessity | Candidate only; the ranking statistic is direction-blind (DBP5) |
| Conditional causal core | Established and test-confirmed, conditional and non-additive (DBP6) |
| Compactness | **Refuted at the sizes tested** — no knee, MLP-dominated (DBP7) |
| Example-specific delivery | **Null** — transfer is class-level (DBP8) |
| Biological mechanism | Positional enrichment only; causal test is a null at n=71 (DBP9, DBP10) |
| Preregistration | **None frozen.** Validation-selected and test-confirmed, but not preregistered (DBPR8) |
| Fine-tuning comparison | Not started. No code or artifacts exist (DBPR6) |

## The one-paragraph account

A frozen DNABERT-2 encoder carries promoter-discriminative information that a masked-mean linear probe
recovers at an early encoder layer without any fine-tuning, comfortably above a k-mer baseline. Zeroing
and patching early components shows that this information is not spread uniformly: a five-node set —
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

- Any generalization statement — the split carries 104 cross-split duplicate groups (DBP2).
- "Layer 2 is the promoter layer" — layers 1–4 are not separated by the data (DBP4).
- "Compact", "sparse", or bare "the circuit" — refuted at the sizes tested (DBP7, DBPR4).
- "Transferable", "sequence-specific delivery" — null contrast (DBP8, DBPR5).
- A TATA-box mechanism — the causal test is a null (DBP10).
- Anything about a fine-tuned DNABERT-2 — no such artifact exists (DBPR6).
- Anything from `results/dnabert_promoter/causal/layer3_*` — different endpoint, superseded (DBPR7).

## Layout

| Path | What is there |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | The `DBP1`–`DBP10` ledger. The point of contact for every claim. |
| [`CORRECTIONS.md`](CORRECTIONS.md) | `DBPR1`–`DBPR8`. What was withdrawn or bounded, and why. |
| [`DECISIONS.md`](DECISIONS.md) | Why the assays are shaped the way they are, including known departures. |
| [`FROZEN_EXPERIMENT_REGISTRY.md`](FROZEN_EXPERIMENT_REGISTRY.md) | T2. The arc's own working registry of assay definitions, metric formulas, and per-block status. |
| [`results/dnabert_promoter/`](../../results/dnabert_promoter/RESULTS_INDEX.md) | Artifacts, with the family index naming which control claims. |
| `scripts/dnabert_promoter/` | 35 runners: cohort building, probing, ablation, restoration, faithfulness, motif tests. |
| `src/gfmcd/adapters/dnabert.py`, `src/gfmcd/dnabert/` | Model access, dataset, perturbation utilities. |
| `configs/dnabert_promoter/base.json` | The frozen assay configuration. |
| `figures/dnabert_promoter/` | Published figures with captions. |
| `tests/test_dnabert_{adapter,dataset,perturb}.py` | Unit coverage for the new modules. |

## The instrument, precisely

- **Model** `quietflamingo/dnabert2-no-flashattention`, revision `813031b2bf86d9e960a027e2734c908009f31601`.
- **Hidden-state contract** the adapter returns exactly the **12 encoder-layer outputs and no embedding
  output**, so `hidden_states[i]` *is* encoder layer `i` with no offset. This is asserted in
  `src/gfmcd/adapters/dnabert.py` and is why layer indices here need no `+1` correction — do not carry
  that assumption to another adapter.
- **Readout** masked mean over biological tokens at encoder Layer 2 → fixed linear probe
  `selected_layer_2_probe.joblib`, sha256 `bd0d9e14e1935e3ebd8e2ce638ccd5a721feb9dc9a0895447d9b62080ce98b3b`.
  The same artifact is used for validation and test.
- **Data** `leannmlindsey/GUE`, config `prom_core_all`; 70-nt sequences; 47,356 train / 5,920 validation
  / 5,920 test, split hashes in `results/dnabert_promoter/shared/split_manifest.json`. **Not
  duplicate-clean** — see DBP2 before using any metric.
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

1. **De-duplicate the split** and re-run the probe. Until this happens, DBP3/DBP4 are upper bounds and
   nothing downstream can be called generalization.
2. **Multiple seeds.** Every effect here is single-seed; the endpoint margin is `0.0019`. Effect-existence
   needs seeded draws, not a within-split held-out set.
3. **Signed, controlled necessity** for the head ranking, replacing the direction-blind statistic.
4. **Freeze a preregistration** before the next confirmatory run, with memorials that can fail and sit on
   the measurement path.
5. **Emit manifests** from every runner.
6. **Power the `TATAAA` cohort** (n=48 is the only stratum worth the attempt) with held-out confirmation
   and motif-specific component localization.
7. Only then: the fine-tuning arc, and only then a cross-model comparison with the SpliceBERT record.
