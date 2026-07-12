# Claims registry — splicebert_partner_selection

One row per empirical claim. **A passing unit test verifies software, never a
scientific claim** — it can advance a claim to `unit-tested`, never to `supported`.
Only a fresh, manifest-stamped artifact advances a claim to `supported`. Downgrades
and retractions are permanent rows; later analyses refine, they do not overwrite.

Status vocabulary: `open` · `implemented` · `unit-tested` · `artifact-supported` ·
`pending-fresh-run` · `retired` · `retracted`.

## Scientific claims

| ID | Claim | Status | Evidence / what would advance it |
|----|-------|--------|----------------------------------|
| C1 | On natural cue conflicts, `additive-frozen` selects the true partner far above the local PWM+length baseline (~0 on the conflict set by construction). | **retracted** | **RETRACTED (R2)** — rests on a positional label leak + site detection, not partner selection. |
| C2 | The partner information is **linearly/additively accessible** at the site endpoints (an interaction readout does not beat the additive one). | **retracted** | **RETRACTED (R2)** — 'interaction ≈ additive ⇒ partner info is linearly accessible' is a fallacy: the task never required an interaction. |
| C3 | The effect is **attributable to pretraining**, not to readout capacity/training. | **retracted** | **RETRACTED (R2)** — the randomized-backbone control does not control for the positional leak (a random net does not expose position linearly), so it cannot attribute the effect to pretraining. |
| C4 | A **compact, localized head circuit** is necessary AND mediates a partner-specific corruption AND is conflict-specific. | **retracted** | **RETRACTED (R3)** — superseded. The forced-choice partner task is intrinsically degenerate; replaced by the cross-site coupling study (D1–D4). |
| C5 | The model **uses** this circuit in its native masked-LM computation (not just probe-decodable). | **open** | Out of the registered scope; needs an MLM-native readout arm. Explicitly NOT claimed by C1–C4. |
| C6 | The frozen circuit score is a **biological predictor** — adds value beyond local controls on independent splicing data. | **open** | `EXTERNAL_VALIDATION.md` phase; not started. Requires held-out external data never used for component selection. |
| C7 | The behavioral advantage survives a **receptive-field-limited CNN / composition** rival, not only the PWM+length baseline. | **open** | Rival not implemented (RRR finding 7). PWM+length is the current cheap rival; a composition/RF-CNN baseline would harden C1. |

## Cross-site coupling study (the replacement design)

No task to win, so nothing to leak: native masked-LM readout, perturbations substitute
real bases (never `[MASK]`), and every claim carries a matched control.

**Provenance.** Discovery on train (chr20/21/22) — `results/splicebert_coupling/per_example.json`,
`exon_vs_intron.json`, `localization.json`, `subspace.json`. **Confirmation on gene-disjoint held-out
chromosomes (chr11/16/19, n = 400, 1000 null sets)** — `scripts/splicebert/run_confirm.py` →
`results/splicebert_coupling/confirmation.json`. Test split (chr8, chr9) **SEALED**.
D4 and D5 were promoted from `pending-fresh-run` on that run (2026-07-12); the same run retracted D4′
(see R4). Overview: `docs/STATUS.md` §3b.

| ID | Claim | Status | Evidence / what would advance it |
|----|-------|--------|----------------------------------|
| D1 | Splice-site recognition is **non-locally dependent** on the partner site: destroying the partner degrades the model's native MLM recognition of the site. | **artifact-supported (held-out)** | `results/splicebert_coupling/per_example.json` (train, n=250): partner acceptor **+0.380** vs motif-matched non-partner AG **−0.000** and distance-matched random position **+0.010**. **The coupling itself replicates on gene-disjoint held-out chromosomes**: mean **1.359** [1.266, 1.450], n=400 (`confirmation.json::val_coupling`). |
| D2 | The dependence is **partner-selective**: destroying acceptor A degrades ITS OWN donor far more than other donors in the same window. | **artifact-supported (train)** | own donor **+0.380** vs other donors **+0.003**; paired Wilcoxon **p = 9.3e-18** (n=250). Bidirectional (donor→acceptor +0.462 vs +0.006). |
| D3 | The coupling is **stronger across the intron than across the exon** at matched separation — the model represents the *intron* (the spliced-out unit), not the exon. | **artifact-supported (train)** | exon +0.288 [0.234,0.348] vs intron +0.490 [0.428,0.552]; at matched distance the ratio is **0.24×**. **Retracts our own earlier "exon definition" reading**, which was distance decay. LIMITATION: only *short* introns fit the 1024nt context, so this cannot generalise to typical (long) human introns. |
| D4 | A **single head (L3, H1)** carries the cross-site coupling: it ranks 1st of 96 on held-out data and one head alone recovers most of the effect. | **artifact-supported (held-out)** | **REPLICATED.** `results/splicebert_coupling/confirmation.json` — gene-disjoint held-out chromosomes (chr11/16/19, n=400, 1000 null sets): L3H1 **rank 1 of 96**; denoise recovery **0.808** (next head L4H2 = 0.186); matched-random null observed **1.193** vs null mean 0.012 / max 0.794 → **p_add_one = 0.000999** (the floor); partner-dependent DiD **1.110** [1.018, 1.205]; MLM health 0.9958 → 0.9953 under ablation, so the head is not simply breaking the model. Scope: **selected on train, confirmed on held-out.** |
| D4′ | ~~L3H1 is **dedicated** to the coupling: a **measured** partner-independent role of ≈0.~~ | **retracted** | **RETRACTED (R4)** — the same held-out run that confirmed D4 falsifies this sub-claim. The partner-independent local role is **−0.200 [−0.272, −0.128]**, a CI that excludes zero. L3H1 carries the coupling, but it is not *only* the coupling. |
| D5 | The partner feature occupies a **~2-dimensional subspace** of L3H1's 32-dim output. | **artifact-supported (held-out)** | **REPLICATED.** Direction fit on train, scored on held-out (`confirmation.json`): a rank-2 projection removes coupling **1.359 → 0.271** (80% of the effect), while a **random** rank-2 projection removes ~2% (1.359 → 1.299). Rank-1 leaves 0.743 and rank-4 adds nothing (0.294), so 2 is the knee. The top-2 directions carry 68% of the output variance. |

## Statistical-machinery claims

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| SM1 | The matched-random-null p-value + BH-FDR path is **calibrated** (uniform max-p, ~0 false discoveries on pure noise). | **artifact-supported** | `results/.../circuits/calibration_audit.json` (KS-uniform maxima, 0 FDR discoveries); `tests/test_rigor_calibration.py`. |
| SM2 | The head-intervention instrument is correct (round-trip patch identity; AtP tracks exact patching). | **unit-tested** | `tests/test_circuits_hf_bert.py` (9), `tests/test_attribution.py` (Spearman 0.876 vs exact). Advances to supported only via a manifest-backed run. |

## Where the coupling study sits on the claim ladder

Per `docs/REAL_GFM_CIRCUIT_DISCOVERY_PROTOCOL.md` §7. **Do not promote past this line.**

| Rung | Status |
|---|---|
| `REPRESENTATION_EVIDENCE` | ✅ |
| `BEHAVIORAL_RELATION_EVIDENCE` | ✅ — native MLM readout, matched motif / distance / non-partner controls (D1, D2) |
| `COMPONENT_LOCALIZATION` | ✅ — L3H1 rank 1/96, held-out, against 1000 matched-random single-head sets |
| `CAUSAL_MEDIATION` | ✅ — denoise recovery 0.808 held-out; grouped ablation drops the effect 1.445 → 0.139 |
| **`CIRCUIT_CANDIDATE`** | ✅ ← **this is the current label** |
| `CAUSALLY_VALIDATED_CIRCUIT` | ❌ **not yet.** Missing: (a) a **path account** — QK/OV decomposition and edge patching showing *how* the coupling is routed and written, not just which head carries it; (b) replication under a **second ablation null** (only denoise is reported); (c) the residual **19%** is measured but unexplained. |
| `BIOLOGICAL_PREDICTOR` | ❌ not started (C6 / `EXTERNAL_VALIDATION.md`) |

**Two limits that must travel with every statement of this result:**
1. **It is coupling across the *intron*, not the exon** (D3). At matched separation, exon +0.288 vs intron +0.490. This retracted our own earlier "exon definition" reading. And only *short* introns fit the 1024-nt context, so it does **not** generalise to typical human introns.
2. **The test split (chr8, chr9) is still SEALED.** Everything above is train + held-out *validation*. One unblinding, after the freeze.

---

## Retractions (permanent)

| ID | Retracted claim | Date | Reason |
|----|-----------------|------|--------|
| R6 | "L2H15 refines the acceptor's representation — **both** the key that makes it findable **and** the value that makes it readable." | 2026-07-12 | Causally false. The rescue shows **value-only rescue = −0.019** [−0.030, −0.011] (zero) while **key-only = 0.423**, and K+V adds nothing over K alone. The claim was inferred from **cosine similarity**, which ranked the *value* as more perturbed (0.929 vs 0.950) — the exact opposite of the causal result. **Cosine magnitude is not causal relevance.** Superseded by D7. |
| R4 | "L3H1 is **dedicated** to the partner coupling — its measured partner-independent role is ≈ 0." (the second half of D4, as first written) | 2026-07-12 | **Falsified by the same held-out run that confirmed the rest of D4.** The 2×2 factorial on gene-disjoint chromosomes gives a partner-independent local role of **−0.200 [−0.272, −0.128]** — a confidence interval that excludes zero. The "≈ 0" came from the train-split factorial (local −0.005, n_eval=50), where the effect was not resolvable at that sample size. **The head carries the coupling; it does not carry *only* the coupling.** D4's surviving content (rank 1 of 96, recovery 0.808, p = 0.000999, DiD 1.110) is unaffected — this retracts the dedication, not the localization. Recorded rather than quietly reworded, because the discovery-split number was the one that flattered us. |
| R6 | "L2H15 refines the acceptor's representation — **both** the key that makes it findable **and** the value that makes it readable." | 2026-07-12 | Causally false. The rescue shows **value-only rescue = −0.019** [−0.030, −0.011] (zero) while **key-only = 0.423**, and K+V adds nothing over K alone. The claim was inferred from **cosine similarity**, which ranked the *value* as more perturbed (0.929 vs 0.950) — the exact opposite of the causal result. **Cosine magnitude is not causal relevance.** Superseded by D7. |
| R4 | "L3H1 is a **single dedicated head** carrying the coupling; its partner-independent role is ≈0 (−0.005), so it is 100% partner-dependent." | 2026-07-12 | Both halves fail on scaled held-out data. (i) The −0.005 local estimate was **noise at n=50**; at n=400 with gene-grouped CIs the partner-absent contribution is **−0.200 [−0.272, −0.128]**, CI excluding zero — the head supplies *signed* evidence, opposing a donor whose partner is absent. (ii) The **exhaustive** 96-head ABLATION scan (which the denoise ranking had missed entirely) found **L2H15 at 0.542**, invisible to denoising. L3H1 is the largest contributor, not the circuit. Superseded by D4/D4b/D7. |
| R5 | "L3H1 accounts for **33% of the total causal mass**." | 2026-07-12 | Statistically invalid framing. That was 33% of the *summed positive marginal ablation effects*, and marginal effects **double-count serial or redundant components**. Directly falsified by the data: 0.944 + 0.542 = **1.486 > the 1.121 baseline coupling**, so the two heads cannot be independent additive shares. The over-additivity is itself evidence of overlap — and the 2×2 factorial confirms it (92% of L2H15 nested inside L3H1). Never report summed marginals as shares. |
| R2 | "SpliceBERT's frozen representation selects the true splice partner on cue conflicts (PMR 0.807 vs 0.056 control)." | 2026-07-12 | **Two independent fatal confounds, both verified on the real data.** (i) POSITIONAL LEAK: `dataset.py:135` set `acceptor = cfg.left_flank`, a CONSTANT — the true acceptor sat at position 200 in **100.0%** of examples (n=3000, one distinct value). Both endpoints were pinned to the window edges, so a **geometry oracle reading zero nucleotides scores PMR 1.000**; a leak-only model with no neural network at all scores 0.483 where the published baseline is 0.000. `len(seq) = 400 + exon_len` exactly, so a length-only adversary scores 0.495. (ii) SITE DETECTION SUFFICES: decoys were spurious AG/GT motifs, so an **oracle site detector scores PMR 1.000** — the task never required partner information. The randomized-backbone control does not catch either. Supersedes C1–C3. |
| R3 | "Real-site decoys (v2) make the partner task identified." | 2026-07-12 | The decoys were real annotated donors, but **minor** ones (median **1** supporting isoform vs **18** for true donors; 64% single-isoform). "Pick the partner" collapsed into "pick the better donor": a plain PWM scores **0.687** vs 0.392 chance, and the bilinear readout beat the additive control by **+0.008** — the acceptor contributed nothing. Root cause is biological, not technical: in a real genome a *strong* donor at a *plausible* exon distance from an acceptor generally **is** its partner, so clean negatives barely exist and the forced-choice task is intrinsically degenerate. Superseded by the cross-site coupling design (D1–D4). |
| R1 | "additive-random-frozen control ≈ additive-frozen, so the behavioral advantage is NOT pretraining-specific (NO-GO)." | 2026-07-11 | The control was a **no-op**: `init_weights()`/`apply(_init_weights)` do nothing after `from_pretrained` on transformers ≥5, so the "random" backbone was the pretrained one (byte-identical results). Fixed via `AutoModelForMaskedLM.from_config`; the corrected control collapses to 0.056. Superseded by C3. |
