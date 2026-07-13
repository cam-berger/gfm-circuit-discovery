> # ⛔ ARCHIVED — THE STUDY THIS PLANS HAS BEEN RETRACTED IN FULL
> **[ARCHIVED]** · **Do not cite this document, and do not build on this design.**
>
> The `splicebert_partner_selection_v1` study was retracted (R1/R2/R3, 2026-07-12): the task carried a
> **positional leak** — a geometry oracle reading *zero nucleotides* scored a perfect 1.000 — and,
> independently, **site detection alone sufficed**, so the task never required partner information. The
> v2 fix was found to be biologically unfixable: clean negatives barely exist.
>
> This file is also stale in the *opposite* direction. §15 says "no SpliceBERT scientific metric or
> circuit result has been produced". That was written before any of it ran. Since then the model *has*
> been run, the study above was retracted, and a **replacement design (cross-site coupling) produced the
> program's first held-out-confirmed circuit in a real GFM** — of which this document knows nothing.
>
> Its reusable content (gate ladder, claim ladder, stop rules, model-with-readout rule) has been lifted
> into the protocol, together with the degeneracy audit that would have caught the retraction.
>
> **Read instead:** `docs/REAL_GFM_CIRCUIT_DISCOVERY_PROTOCOL.md` · `docs/STATUS.md` §3 ·
> `studies/splicebert_partner_selection/CLAIMS.md`

---

# Partner-Aware Splice-Site Circuit Discovery in SpliceBERT

## Implementation plan, methodology, runbook, claim rules, and repository map

**Study identifier:** `splicebert_partner_selection_v1`  
**Primary model:** the official pretrained SpliceBERT masked-language model  
**Primary biological relation:** selection of a compatible acceptor–donor pair for an internal exon  
**Primary mechanistic target:** a causal computation in the pretrained backbone that makes endpoint representations depend on their splice partner, rather than only on independent local motif strengths  
**Status:** implementation scaffold complete; no scientific result is claimed until the registered experiments are run

---

## 1. The paper-sized objective

The first real-GFM paper should make one defensible transition:

> **From ground-truth-calibrated synthetic circuits to a causally validated partner-aware splice-site circuit in pretrained SpliceBERT, with an external test of whether circuit activity predicts real splice-disrupting variants beyond local motif baselines.**

The intended contributions are:

1. **A calibrated genomic circuit-discovery protocol.** The protocol separates behavioral competence, shortcut exclusion, component necessity, mediation, sufficiency, and biological validity. It is first checked on the repository's planted and controlled models.
2. **A splice-pair cue-conflict benchmark.** Natural internal exons are presented with multiple motif-compatible acceptors and donors. A frozen local model supplies a strong shortcut baseline and identifies examples where local evidence favors the wrong pair.
3. **A causal circuit in a pretrained genomic model.** The primary target is a compact, replicated set of SpliceBERT components whose intervention selectively changes partner-aware splice-pair behavior and whose clean activations mediate the behavior under aligned counterfactuals.
4. **A pretraining-versus-adaptation result.** Frozen-pretrained, randomly initialized, and fine-tuned conditions determine whether the mechanism was already present in pretraining, merely readable after pretraining, or installed by task supervision.
5. **A biological validation.** Circuit activity is tested on independent splice-disrupting variants or experimentally measured alternative-splicing events and compared with local sequence scores and whole-model predictions.
6. **A calibrated shortcut-gap hypothesis.** Across the controlled tasks and splice study, mechanism formation is related to the performance gap between registered shortcut classes and relational solutions. This is a hypothesis about probability and optimization, not an “if and only if” law.

A negative result remains useful. If SpliceBERT does not beat the registered local/additive controls, the correct conclusion is that the model does not provide evidence for a partner-aware mechanism on this assay. If fine-tuning creates the behavior but the frozen model lacks it, the result is an **adaptation-installed circuit**, not a pretrained circuit.

---

## 2. What “partner-aware” means operationally

A splice-pair score is **strictly endpoint-additive** when it can be written as

\[
S(a,d,x) = f_a(a, x_{\text{local around }a}) + f_d(d, x_{\text{local around }d}) + g(d-a),
\]

where the acceptor and donor terms are evaluated independently and the only pair-level feature is a registered exon-length prior.

A model is **behaviorally partner-aware** on this study only when all of the following are true:

1. It outperforms the frozen local/additive baselines on held-out genes or chromosomes.
2. It succeeds on a frozen natural conflict set where the local model selects a decoy pair.
3. Its behavior degrades when distal context or the partner endpoint is removed while local endpoint windows are preserved.
4. It shows a non-additive response in the registered AA/AB/BA/BB endpoint-swap assay, with matched local controls near zero.

A **partner-aware circuit** requires more than behavior:

5. A compact component set is necessary under multiple ablation nulls.
6. Clean component activations restore a meaningful fraction of the clean–corrupt behavior gap when patched into aligned corrupt inputs.
7. The selected set beats size-matched random component sets.
8. The effect replicates across seeds or independently trained readouts and is not an artifact of one head numbering.
9. Reverse-engineering of QK/OV or MLP paths explains how information is routed and written, rather than merely identifying high-effect components.

A **biological discovery** additionally requires that the circuit-derived quantity predicts an independent experimental outcome beyond established local baselines.

---

## 3. Why SpliceBERT is the first target

SpliceBERT is the cleanest bridge from the current repository because it is a compact BERT-style masked-language model trained on pre-mRNA, uses single-nucleotide tokenization, and exposes ordinary Transformer layers that the existing intervention machinery can hook. It is small enough for exhaustive head screens and repeated causal runs, while its pretraining domain makes splice relationships a plausible native computation.

The choice is methodological rather than a claim that SpliceBERT already contains the desired circuit. The study is designed to answer that question.

The official checkpoint must be used as the primary model. Compatible ports may be used for engineering smoke tests only and must be labeled with a representation-equivalence check before scientific numbers are compared.

---

## 4. The central experimental distinction

SpliceBERT has no native “choose this acceptor–donor pair” output. A downstream readout is therefore necessary, but the readout can accidentally manufacture the very interaction the study is trying to find.

The repository implements several regimes with different claim scope:

| Regime | Backbone | Readout | What a positive result means |
|---|---|---|---|
| `additive-frozen` | pretrained, frozen | independent acceptor + donor + length | **Primary condition.** Partner information used by the readout must already be exposed in contextual endpoint states. |
| `additive-random-frozen` | randomly reinitialized, frozen | same additive readout | Controls for architecture, token identity, and readout capacity without pretraining. |
| `interaction-frozen` | pretrained, frozen | additive terms plus bilinear endpoint interaction | Diagnostic upper bound. The readout can synthesize a relation; this cannot alone prove a pretrained circuit. |
| `additive-finetuned` | pretrained, trainable | additive readout | Tests whether supervision installs partner information into the backbone while keeping the final readout constrained. |
| `interaction-finetuned` | pretrained, trainable | bilinear interaction | Maximum task-adaptation condition; useful for mechanism-emergence comparison, not evidence of a native circuit. |
| native MLM dependency | pretrained, frozen | no trained pair readout | Zero-shot pilot: does native masked-token prediction at one site depend on the other site? |

The primary mechanistic trace is run on `additive-frozen` only. Other regimes are explicitly secondary unless the study is reframed as circuit formation during fine-tuning.

---

## 5. Data unit and coordinate conventions

Each example is one internal exon with enough intronic context on both sides. The sequence is always stored in **transcript orientation**, including reverse-complementing minus-strand genomic windows.

Coordinates are zero-based within the oriented sequence:

- `acceptor_pos` is the first exonic nucleotide; canonical `AG` is at `sequence[pos-2:pos]`.
- `donor_pos` is the first intronic nucleotide after the exon; canonical `GT` or `GC` is at `sequence[pos:pos+2]`.
- The true exon is `[acceptor_pos, donor_pos)`.

Candidate acceptors and donors are motif-compatible positions in the same window. Candidate pairs must satisfy the registered exon-length range. Every example must contain the annotated pair and at least one decoy pair.

The data builder:

- parses GENCODE GTF without requiring pandas;
- fetches the reference genome with `pyfaidx`;
- handles both strands;
- rejects noncanonical annotated boundaries when `canonical_only=true`;
- deduplicates shared exons across transcripts by default;
- creates chromosome-held-out train/validation/test partitions;
- writes JSONL plus source hashes and build manifests.

The default split reserves chromosomes 17–18 for validation and 19–22 for the final test. These values are configurable but must be frozen before model development. No test file should be loaded by training or discovery scripts.

---

## 6. Shortcut classes and cue conflict

The study does not define “shortcut” as anything the researchers dislike. It registers concrete model classes:

1. **Local PWM + length prior:** acceptor PWM, donor PWM, and empirical exon-length distribution.
2. **Additive contextual readout:** independent scores from full-context acceptor and donor hidden states plus length.
3. **Random-backbone additive readout:** same architecture without pretrained representations.
4. **Context-restricted controls:** the same trained readout after all nonlocal sequence content is replaced by `N`.
5. **Optional receptive-field-limited CNN:** add before final preregistration if compute permits; it should be trained on the identical candidate task and split.

The primary natural cue-conflict set is frozen on validation examples where the local PWM+length model ranks a decoy above the annotated pair. This is more informative than ordinary accuracy because local and relational rules disagree.

The final test conflict set is generated only after preregistration and test unblinding. The local baseline itself is fit once on training data and never changed in response to circuit results.

---

## 7. Counterfactual assays

### 7.1 Context-removal controls

These length-preserving controls keep candidate coordinates fixed:

- `all_candidate_site_windows_only`: preserve local windows around every candidate site; replace everything else with `N`.
- `true_pair_endpoints_only`: preserve only the true acceptor and donor windows.
- `true_acceptor_masked` and `true_donor_masked`: replace one true endpoint neighborhood with `N`.
- `exon-core-shuffle`: preserve boundary neighborhoods while dinucleotide-shuffling the exon interior.

These tests establish context or endpoint dependence. They do **not** alone prove partner-specific computation.

### 7.2 Natural local-versus-annotation conflict

The frozen local model selects a decoy while the annotation identifies the true pair. The model follows either:

- the local shortcut, or
- information not captured by the registered local model.

Success does not prove a circuit, but it is the primary behavioral gate for tracing.

### 7.3 AA/AB/BA/BB boundary factorial

Two aligned examples supply acceptor and donor boundary windows. On one shared background, construct:

- `AA`: acceptor A + donor A
- `AB`: acceptor A + donor B
- `BA`: acceptor B + donor A
- `BB`: acceptor B + donor B

For scalar score `f`, calculate

\[
I = f(AA) - f(AB) - f(BA) + f(BB).
\]

A strictly endpoint-additive local score has `I≈0`. A non-zero model contrast rejects strict additivity for the scored function. Because the sequences are chimeric, this is a secondary mechanistic assay and must not replace the natural conflict endpoint.

### 7.4 Native masked-LM dependency

For a candidate pair, mask each acceptor motif token one at a time and measure its log probability with the donor intact versus the donor neighborhood replaced by `N`; then repeat in the opposite direction. The symmetric score is

\[
D(a,d) = [PLL(a\mid d)-PLL(a\mid \cancel d)] + [PLL(d\mid a)-PLL(d\mid \cancel a)].
\]

This is a zero-shot dependency pilot. It is sensitive to replacement choice and candidate subsampling, so motif-only and wider-neighborhood variants should be reported separately.

---

## 8. Causal discovery pipeline

### Gate 0 — Instrument calibration

Before tracing SpliceBERT, the existing planted compositional and shortcut twins must continue to pass. Add a LayerNorm-bearing planted model if the SpliceBERT tracer uses approximations not exercised by the current twins.

**Pass condition:** the tracer recovers the known path in the compositional twin, rejects it in the behaviorally similar shortcut twin, and documents false positives/negatives.

### Gate 1 — Dataset audit

Run smoke construction, manually inspect plus- and minus-strand examples, verify canonical motifs and exon intervals, then build the full dataset.

**Pass condition:** zero coordinate errors in tests and a manually reviewed stratified sample; no gene or chromosome leakage; every retained example has at least one decoy.

### Gate 2 — Freeze shortcut baselines and conflicts

Fit the local PWM+length model on training only. Save it, record hashes, and freeze validation conflict examples.

**Pass condition:** baseline metrics and conflict-set size are stable and the baseline file is immutable for later stages.

### Gate 3 — Establish behavior before tracing

Run native MLM dependency and all readout regimes over at least five seeds. Evaluate full validation, natural conflict validation, and context controls.

The primary frozen-pretrained claim requires:

- `additive-frozen` exceeds the registered local baseline with a gene-grouped confidence interval above zero;
- the advantage persists on the natural conflict set;
- it exceeds `additive-random-frozen`;
- removal of distal/partner context causes a registered margin reduction;
- the result replicates in at least three seeds.

If only fine-tuned conditions pass, change the paper question to **how task supervision installs a splice-pair circuit**.

### Gate 4 — Component localization

Use loss- or margin-based head intervention, not attention rank alone. Screen zero, mean, and resample ablations on a frozen validation conflict batch.

Promote a head only when:

- its effect is positive under at least two registered ablation nulls;
- the grouped selected set has a larger effect than 95% of size-matched random sets;
- the effect is not explained by global degradation on matched non-conflict examples.

The supplied `trace_heads.py` produces a **component-localization** result, not a complete circuit claim.

### Gate 5 — Mediation by activation patching

Construct aligned clean/corrupt inputs with a real behavior gap. Cache clean head outputs and patch them into the corrupt run.

Require:

- a nontrivial clean–corrupt score gap before patching;
- positive recovery for the registered candidate set;
- recovery above matched random component sets;
- localization to biologically relevant positions or routes when token-restricted patching is used.

The supplied patch script supports endpoint/context corruptions as instrumentation. A partner-specific paper claim additionally requires patching the AA/AB/BA/BB interaction contrast or another counterfactual that preserves local endpoint evidence while changing the relation.

### Gate 6 — Reverse-engineer the path

Only after causal localization:

1. inspect attention routing from candidate endpoint/query positions;
2. decompose QK selection and OV residual writing;
3. test whether candidate heads pass endpoint identity, distance, strand, or compatibility information;
4. use edge/path patching to distinguish parallel heads from a serial circuit;
5. inspect MLPs or features between heads when head-only mediation is incomplete.

A circuit diagram must specify nodes, directed edges, represented variables, intervention effect, and evidence status.

### Gate 7 — Necessity, sufficiency, and completeness

- **Necessity:** grouped ablation removes a substantial fraction of partner-aware behavior while matched random groups do not.
- **Sufficiency/mediation:** preserving or patching the proposed circuit restores a registered fraction of the clean–corrupt gap.
- **Completeness:** the residual unexplained effect is measured; “compact circuit” does not mean “all computation is captured.”
- **Specificity:** effects are larger on partner conflicts than matched ordinary examples and do not simply destroy all token representations.

### Gate 8 — Replication and architecture controls

Replicate at the level of computation, not head number. Independent readout seeds may realize the same routing through different individual heads.

Required controls:

- at least three successful seeds;
- multiple model/readout checkpoints where feasible;
- shuffled labels or random backbone;
- matched random component sets;
- plus/minus strand stratification;
- exon-length, candidate-count, chromosome, and gene-family stratification;
- equivalent conclusions under more than one ablation null.

### Gate 9 — External biological validation

Freeze the circuit and its scalar activation metric before opening the external dataset. Candidate validation targets include experimentally measured splice-disrupting variants, cryptic splice-site activation, or independent alternative-splicing assays.

Compare nested models:

1. local acceptor/donor scores and distance;
2. whole-model readout score;
3. circuit activity alone;
4. local features + circuit activity;
5. local features + whole-model score;
6. all features.

The strongest result is incremental predictive value from the circuit metric on held-out genes or experiments, followed by targeted perturbation validation. A whole-model correlation without circuit increment is not a biological discovery.

---

## 9. Registered claim ladder

Use these exact labels in result files and prose:

| Label | Minimum evidence |
|---|---|
| `REPRESENTATION_EVIDENCE` | Probe/readout can recover the variable. |
| `BEHAVIORAL_RELATION_EVIDENCE` | Model beats registered shortcuts on conflicts and context controls. |
| `COMPONENT_LOCALIZATION` | Individual/group interventions alter the target metric. |
| `CAUSAL_MEDIATION` | Clean activations patch the behavior into aligned corrupt inputs. |
| `CIRCUIT_CANDIDATE` | Replicated directed component/path account with necessity and mediation. |
| `CAUSALLY_VALIDATED_CIRCUIT` | Necessity, sufficiency/mediation, specificity, random controls, replication, and measured incompleteness. |
| `BIOLOGICAL_PREDICTOR` | Circuit metric predicts an independent biological outcome beyond local controls. |
| `BIOLOGICAL_DISCOVERY_CANDIDATE` | Novel rule generated from the circuit survives independent experimental or cohort validation. |

Never use a higher label because a visualization “looks right.”

---

## 10. Repository implementation

### New source modules

```text
src/gfmcd/
├── adapters/
│   ├── base.py                 # common model output/token contracts
│   └── splicebert.py           # official/compatible HF checkpoint adapter
├── circuits/
│   ├── hf_bert.py              # head capture, ablation, patching, random sets
│   └── qkov.py                 # QK logits and OV write decomposition
├── provenance.py               # hashes and experiment manifests
└── splicing/
    ├── schema.py               # audited coordinate/data contracts
    ├── gtf.py                  # dependency-free GTF parser
    ├── dataset.py              # transcript-oriented exon examples
    ├── candidates.py           # acceptor/donor/pair enumeration
    ├── io.py                   # JSONL and dataset manifests
    ├── baselines.py            # local PWM + length shortcut
    ├── context_controls.py     # distal/partner removal controls
    ├── perturb.py              # aligned edits and endpoint factorials
    ├── readout.py              # additive and bilinear pair heads
    ├── training.py             # explicit training/checkpoint loop
    ├── scoring.py              # clean/counterfactual fixed-candidate scoring
    ├── metrics.py              # grouped bootstrap, margins, recovery
    ├── zero_shot.py            # native masked-LM dependency
    └── config.py               # configuration loading
```

### New executable scripts

```text
scripts/splicebert/
├── fetch_resources.sh
├── build_dataset.py
├── evaluate_local_baselines.py
├── evaluate_native_dependency.py
├── train_pair_readout.py
├── evaluate_behavior.py
├── pair_swap_factorial.py
├── trace_heads.py
├── patch_conflicts.py
├── run_smoke.sh
└── run_validation_pipeline.sh
```

### Study records

```text
studies/splicebert_partner_selection/
├── README.md
├── METHOD.md
├── RUNBOOK.md
├── PREREGISTRATION.md
├── EXTERNAL_VALIDATION.md
├── DECISIONS.md
└── RESULTS_TEMPLATE.md
```

The generic template in `studies/TEMPLATE_REAL_GFM_STUDY/` should be copied for future GFM targets.

---

## 11. Exact startup sequence

Run from the repository root.

### 11.1 Environment

```bash
uv sync --extra splicebert --extra dev
uv run pytest -q tests/test_splicing_*.py
```

### 11.2 Download resources

```bash
bash scripts/splicebert/fetch_resources.sh
```

The script downloads GENCODE resources and the official SpliceBERT archive, writes checksums, and deliberately refuses to guess which extracted checkpoint directory is the intended model. Inspect the extracted directory and set `model.path` in `configs/splicebert_partner_selection/base.json`.

### 11.3 Smoke dataset

```bash
uv run python scripts/splicebert/build_dataset.py \
  --config configs/splicebert_partner_selection/base.json \
  --max-examples 200
```

Inspect at least 20 examples, including both strands. Then delete the smoke output and rebuild without `--max-examples`.

### 11.4 Freeze local baseline and validation conflicts

```bash
uv run python scripts/splicebert/evaluate_local_baselines.py \
  --config configs/splicebert_partner_selection/base.json
```

Do not use `--include-test` during development.

### 11.5 Native dependency pilot

```bash
uv run python scripts/splicebert/evaluate_native_dependency.py \
  --config configs/splicebert_partner_selection/base.json \
  --local-baseline results/splicebert_partner_selection/baselines/local_pair_baseline.pkl
```

### 11.6 Train registered regimes

```bash
for seed in 0 1 2 3 4; do
  for regime in additive-frozen additive-random-frozen interaction-frozen additive-finetuned interaction-finetuned; do
    uv run python scripts/splicebert/train_pair_readout.py \
      --config configs/splicebert_partner_selection/base.json \
      --regime "$regime" --seed "$seed"
  done
done
```

Start with smoke limits on one seed before launching the full matrix.

### 11.7 Evaluate validation behavior

```bash
uv run python scripts/splicebert/evaluate_behavior.py \
  --checkpoint artifacts/splicebert_partner_selection/readouts/additive-frozen/seed0/checkpoint.pt

uv run python scripts/splicebert/evaluate_behavior.py \
  --checkpoint artifacts/splicebert_partner_selection/readouts/additive-frozen/seed0/checkpoint.pt \
  --subset-file results/splicebert_partner_selection/baselines/val_local_conflicts.jsonl
```

### 11.8 Non-additivity assay

```bash
uv run python scripts/splicebert/pair_swap_factorial.py \
  --checkpoint artifacts/splicebert_partner_selection/readouts/additive-frozen/seed0/checkpoint.pt \
  --local-baseline results/splicebert_partner_selection/baselines/local_pair_baseline.pkl
```

### 11.9 Trace heads only after behavior passes

```bash
uv run python scripts/splicebert/trace_heads.py \
  --checkpoint artifacts/splicebert_partner_selection/readouts/additive-frozen/seed0/checkpoint.pt
```

### 11.10 Patch registered corruptions

```bash
uv run python scripts/splicebert/patch_conflicts.py \
  --checkpoint artifacts/splicebert_partner_selection/readouts/additive-frozen/seed0/checkpoint.pt \
  --corruption donor-mask \
  --selected-head-report results/splicebert_partner_selection/circuits/head_screen/additive-frozen_seed0/report.json
```

The endpoint/context patch result is an instrumentation milestone. Add a partner-specific patching assay before writing the final circuit claim.

### 11.11 Final test unblinding

Only after the preregistration, candidate circuit, analysis code, thresholds, and external-validation plan are frozen:

```bash
uv run python scripts/splicebert/evaluate_local_baselines.py --include-test

uv run python scripts/splicebert/evaluate_behavior.py \
  --checkpoint <frozen-checkpoint> \
  --split test --unblind-test
```

No hyperparameter or circuit-component selection may be changed after this step without labeling the new analysis exploratory.

---

## 12. Immediate work queue

### Week 1: engineering and audit

- Merge the repository cleanup work first.
- Run new unit tests and the old planted-tracer suite.
- Fetch official SpliceBERT and verify one-token-per-nucleotide mapping.
- Build a 200-example smoke dataset.
- Manually audit coordinates and reverse-complement logic.
- Add a tiny end-to-end CI test using a fake BERT model; never download model weights in CI.

### Week 2: full dataset and baselines

- Build full GENCODE examples and source manifests.
- Generate descriptive statistics by split, strand, exon length, and candidate count.
- Fit and freeze local baseline.
- Freeze validation conflict set.
- Decide whether a receptive-field-limited CNN is required before preregistration.

### Week 3: behavioral regimes

- Run the native dependency pilot.
- Train all regimes for five seeds.
- Compare full validation, natural conflicts, and context controls.
- Hold a stop/go review. Do not trace a circuit if the primary condition does not pass behavioral gates.

### Week 4: causal localization

- Freeze 16–32 validation conflict examples for discovery.
- Run cross-null head screens.
- Test grouped candidate sets against matched random sets.
- Add matched non-conflict degradation controls.
- Select a small candidate set without opening the test split.

### Week 5: mediation and path discovery

- Run aligned activation patching.
- Implement and preregister partner-specific factorial patching.
- Decompose candidate heads into QK selection and OV write roles.
- Add edge/path patching and MLP mediation as required.

### Week 6: replication and freeze

- Repeat the mechanism trace across successful seeds.
- Produce one claim-status table and one circuit graph with evidence per edge.
- Freeze test analysis and external-validation metric.
- Unblind test once.

### Following phase: external biology

- Normalize an independent variant/event dataset into the external-validation schema.
- Compute frozen local, whole-model, and circuit scores.
- Test incremental value with gene-grouped or experiment-grouped splits.
- Select any proposed novel biological rule before targeted validation.

---

## 13. Stop rules

Stop or reframe rather than forcing a circuit narrative:

1. **No behavioral advantage:** if `additive-frozen` does not beat registered baselines on conflicts, do not trace a pretrained partner circuit.
2. **Readout-only interaction:** if only `interaction-frozen` succeeds, conclude that SpliceBERT representations permit a downstream interaction but do not show that the backbone computes it.
3. **Fine-tuning-only effect:** if only fine-tuned regimes succeed, study circuit installation during adaptation.
4. **Global damage:** if candidate-head ablation harms ordinary examples as much as conflicts, the heads are not specific enough for the proposed circuit.
5. **No mediation:** necessity without patch recovery supports involvement but not a sufficient path account.
6. **Seed instability:** report a fragile or multiply realizable mechanism; do not hide failed seeds.
7. **No external increment:** a valid model circuit can still fail to provide a new biological predictor. Keep the mechanistic claim and drop the discovery claim.

---

## 14. Future-study template

Every future real-GFM target should instantiate the same fields:

- biological relation and exact model output;
- registered shortcut classes;
- natural and synthetic cue conflicts;
- zero-shot, frozen-readout, and adapted conditions;
- causal score with sign convention;
- component and edge interventions;
- clean/corrupt alignment rules;
- necessity, mediation, specificity, completeness, and replication thresholds;
- external biological validation;
- negative-outcome interpretation;
- immutable manifests and test-unblinding rule.

This makes SpliceBERT the first implementation of a reusable program rather than a one-off notebook.

---

## 15. Implementation status at handoff

### Implemented and unit-tested

- transcript-oriented GTF/FASTA data contracts and candidate enumeration;
- local PWM + exon-length shortcut baseline and frozen conflict-set generation;
- official/compatible Hugging Face SpliceBERT adapter with token-alignment checks;
- additive, randomized-backbone, bilinear, and fine-tuning training regimes;
- validation-only behavioral evaluation and explicit test-unblinding guard;
- context-removal and endpoint-removal controls;
- native masked-LM partner-dependency pilot;
- AA/AB/BA/BB non-additivity assay;
- attention-head zero/mean/resample intervention;
- matched-random component-set control;
- clean-to-corrupt head activation patching;
- residual-layer patching;
- initial QK/OV decomposition helpers;
- hashes and experiment manifests;
- study/preregistration/runbook/future-study templates;
- 11 new focused tests, plus 27 selected legacy tests passing in the handoff environment.

### Intentionally not claimed or not yet completed

- official model weights and GENCODE data were not downloaded or executed in this environment;
- no SpliceBERT scientific metric or circuit result has been produced;
- the exact extracted official checkpoint directory must be selected and recorded;
- a receptive-field-limited CNN baseline is optional but should be decided before preregistration;
- endpoint/context patching is implemented, but a **partner-specific** activation-patching assay for the factorial interaction contrast still needs to be finalized after observing whether the behavioral factorial has adequate signal;
- full directed edge/path patching and MLP-feature tracing remain follow-on work after the first causal localization;
- the independent splice-variant/event dataset adapter and statistical model remain to be implemented once the external dataset is chosen;
- the full legacy test suite was not run to completion during packaging; the new focused suite and selected legacy suites passed.

The code is therefore an executable research scaffold, not a completed experiment and not evidence that the proposed circuit exists.
