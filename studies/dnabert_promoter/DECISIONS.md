# DNABERT-2 promoter study — design decisions

> **Tier: T2 — provenance.** Why the assays are shaped this way. Not a claim record; claims are in
> [`CLAIMS.md`](CLAIMS.md).

Recorded on 2026-08-11 from the committed scripts, configs, and artifacts. Where a decision was made
implicitly by the code rather than deliberately, that is said plainly — several of the entries below are
reconstructions of what the runners do, not minutes of a choice that was written down at the time.

## D1 — Promoter classification on GUE `prom_core_all` as the scoped behaviour

70-nt core-promoter sequences, balanced two-class, from `leannmlindsey/GUE`. Chosen because it is a
standard genomic-LM benchmark with a fixed split, which makes the *method* comparison the interesting
variable rather than the dataset. The cost is that a benchmark split is not curated for our purposes: the
raw cohort turns out to carry 259 exact-duplicate groups, 104 of them spanning splits. A separate cleaning
stage handles that (**D13**, **DBP2**) — a published split being standard is not the same thing as its being
a clean held-out set, and this study is the demonstration of that.

## D2 — A frozen encoder plus a fixed linear probe, not a fine-tuned classifier

The behaviour under study has to exist in the pretrained model for the mechanistic question to be about
pretraining rather than about task adaptation. Freezing the encoder and putting a single linear probe on
a masked-mean pooled layer keeps the readout simple enough that interventions are interpretable: the
promoter margin is a linear functional of one hidden state, so patching a component has a traceable path
to the score. The consequence is that everything here is scoped to *that* readout — a different pooling
rule or a fine-tuned head is a different behaviour, not the same one measured better.

## D3 — Masked mean over biological tokens only

DNABERT-2 uses BPE over nucleotides, so token count varies per sequence and special tokens carry no
sequence content. Averaging over the biological-token mask makes the readout length-invariant and keeps
`[CLS]`/`[SEP]` out of the pooled state. This matters more than it looks: it is also what makes
token-span localization hard, because BPE token boundaries do not align to nucleotide spans — which is
why that assay is blocked with no artifact and no claim.

## D4 — Layer selection on validation, test used descriptively

`run_frozen_layer_probes.py` fits one probe per encoder layer, ranks on validation balanced accuracy
with validation ROC-AUC as tie-breaker, and records `test_usage: "descriptive evaluation only; not used
for layer selection"`. The discipline is right and was honoured. What the discipline cannot fix is that
the margin is `0.0019` balanced accuracy across layers 1–4 on a single seed, so the *selection* is not
resolved by the data even though the *protocol* was clean (**DBP4**, **DBPR2**). Held-out is not the
same as stable, exactly as repo convention 4 warns.

## D5 — The 12-not-13 hidden-state contract

DNABERT-2's custom implementation ignores `output_hidden_states` and is driven with
`output_all_encoded_layers=True`, returning the 12 encoder-layer outputs with **no embedding output**.
The adapter asserts `len(hidden_states) == num_layers` and fails loudly otherwise. This removes the usual
off-by-one hazard — here `hidden_states[i]` is encoder layer `i` — but it is a property of *this*
adapter, so the assertion is the load-bearing part and must not be relaxed. Any port to another model
has to re-establish which convention holds before reusing a layer index.

## D6 — Two readout layers exist in the tree, and only one is canonical

An earlier pass ran the causal suite at encoder Layer 3 against `layer_3_probe.joblib`; the canonical
suite runs at Layer 2 against `selected_layer_2_probe.joblib`. Both survive in
`results/dnabert_promoter/causal/`, distinguished only by a `layer3_` filename prefix and a
`selected_layer` field. Results are immutable here, so the Layer-3 family was **not deleted** — it is
classified `superseded` in `results/MANIFEST.csv` and marked in the family index (**DBPR7**). If you are
adding a causal runner, read `SELECTED_LAYER` and the probe path together before comparing anything to a
committed number; the filename prefix is the only cue in the directory listing.

## D7 — Corruption as the behavioural handle

Mononucleotide-shuffle corruption of the promoter sequence, with the clean/corrupt margin difference as
the total effect, and recovery fraction defined against that difference. This gives every intervention a
common denominator and makes identity controls checkable — they are exactly zero throughout, which is
the memorial that the patching machinery is wired correctly. Matched-edit-count and GC-preserving
corruption variants are defined in the registry but only partly run.

## D8 — Necessity and delivery reported as conditional effects, never added

`necessity_fraction = (clean − knockout) / total_effect` and
`delivery_fraction = (delivery − corrupt) / total_effect` are both normalized by the same total effect,
which makes them comparable but **not complementary**: they sum to `1.51` on the test cohort and delivery
maxima exceed `1.0`, because residual and LayerNorm paths stay nonlinear under patching. Repo convention
5 forbids reporting these as additive fractions and the ledger obeys that (**DBP6**). Anyone tempted to
write "the circuit accounts for X% of the behaviour" should read the two numbers again.

## D9 — Faithfulness measured standalone, against a growing circuit

The faithfulness curve runs the candidate set *alone* — everything outside it ablated — and grows it by
ranked node order, which is the only way to distinguish a compact mechanism from a well-chosen slice of a
distributed one. It is the decision that produced this study's most important negative result
(**DBP7**): conditional delivery of `0.882` and standalone faithfulness of `0.194` are the same set of
nodes measured two ways, and only the second speaks to compactness. Keeping both assays is what stopped
the arc's own "compact confirmed component set" language from surviving review.

## D10 — Donor-matrix design: matched, same-class, opposite-class, zero

Four source conditions per component, because the interesting contrast is not "does patching help" but
"does patching from *this* example help more than from an unrelated example of the same class". Matched
versus same-class isolates example specificity; same-class versus opposite-class isolates class signal;
zero is the floor. This design is what makes **DBP8** a bounded claim rather than an overstatement — it
had the control that could falsify example specificity, and that control came back null. The design was
right; the arc's reading of it was not (**DBPR5**).

**Terminology warning.** "Donor" in this study means the *source example whose activations are
transplanted*. Everywhere else in this repository "donor" means a **splice donor site**. The collision is
unfortunate and is recorded in [`docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md); prefer
"source example" / "activation source" in new prose.

## D11 — TATA-like motifs as the first biological target, with a matched non-motif control

`TATA[AT][AT]` in an early positional window, disrupted two bases at a time, against a control that makes
the *identical* substitutions (T→G, A→C) at a non-motif site in the same sequence. Matching the edit
operation and count is what makes the contrast about the motif rather than about local damage. The
early-window restriction was chosen after full-window enrichment came back null — which makes the window
a **post-hoc, dataset-aligned** choice, not a TSS-anchored one, and the ledger says so (**DBP9**).

## D12 — The study is `dnabert_promoter`; two recorded labels still say `dnabert_promoter_mechanism`

The arc arrived split across two directories — `studies/dnabert_promoter/` holding the registry and
`studies/dnabert_promoter_mechanism/` holding five 0-byte files — and was consolidated into
`studies/dnabert_promoter/` on 2026-08-11, matching `results/`, `scripts/` and `configs/`, which all
already used that name.

Two string labels were **deliberately not renamed**: `"study": "frozen_dnabert_promoter_mechanism"` in
`results/dnabert_promoter/frozen_claim_tests/registry/frozen_config.json`, because it is inside an
immutable result artifact; and `"project": "dnabert_promoter_mechanism"` in
`configs/dnabert_promoter/base.json`, so that the config keeps agreeing with the artifacts recorded under
it. Both are labels, not paths — nothing resolves through them. Read either as naming this study.

## D13 — Leakage control is a separate stage, and its report is not committed

`build_promoter_cohort.py` writes `shared/leakage_report.json` over the **raw** GUE splits;
`clean_promoter_cohort.py` then removes conflicting-label hashes globally, de-duplicates within split, drops
cross-split repeats under `train > validation > test` precedence, and asserts the retained hash sets are
pairwise disjoint. `extract_frozen_representations.py` reads the cleaned directory, so the cleaned cohort is
what every number in the study is measured on (`DBP2`).

Splitting diagnosis from treatment is the right shape — the raw report stays as evidence of what was wrong
— but it has one sharp edge, which this review walked straight into: **the diagnosis is committed and the
treatment's output is not.** A reader who finds `leakage_report.json` and stops there will conclude the
study is contaminated. The fix is to commit `shared/clean/cleaning_report.json` and the cleaned split
hashes; until then the cleaned row counts are recoverable only indirectly, from `train_rows` /
`validation_rows` / `test_rows` in `frozen/layer_probes/all_layer_probe_results.json`. The manifest helper
records which cohort a run read, so future artifacts state it on their own.


## Fine-tuned extension decisions

These entries record the design of the completed fine-tuned checkpoint analysis. They extend the frozen
study record; they do not retroactively change D1–D13.

## D14 — Treat the fine-tuned checkpoint as a distinct analyzed model state

The analyzed task-specific checkpoint is `magicslabnu/DNABERT-2-finetuned-prom_300_all` at revision
`7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12`. It is instantiated through
`quietflamingo/dnabert2-no-flashattention` at revision
`813031b2bf86d9e960a027e2734c908009f31601` because that implementation is compatible with the current
environment. State-dict loading is exact (zero missing/unexpected keys) and the classifier head is checked
against the checkpoint. The model state is therefore pinned for analysis, but this does not by itself prove
that it is the exact supervised descendant of the frozen assay state under a controlled training run.

## D15 — Fine-tuned behaviour is GUE `prom_300_all`, 300 nt

The fine-tuned checkpoint is evaluated on `leannmlindsey/GUE`, task `prom_300_all`, with 300-nt sequences
and official split sizes `47,356 / 5,920 / 5,920`. This is deliberately recorded as a different scoped
behaviour from D1's frozen `prom_core_all`, 70-nt assay. Exact cross-split overlaps are audited and a
strict-unseen test sensitivity analysis is reported, but the frozen cleaner's policy is not silently
ported to this separate package.

## D16 — Layer probing is a representation assay, selected only on development data

Masked-mean representations are extracted from all 12 encoder layers and each layer receives an independent
`StandardScaler + LogisticRegression` probe. The endpoint is selected by development balanced accuracy
with development ROC-AUC as tie-breaker. Test metrics remain descriptive. Layer 10 wins by the stated
development rule; Layer 8 has slightly higher test BA and is not substituted post hoc. The probe is
interpreted as linear accessibility, never as causal localization.

## D17 — Lock one balanced 256-example test cohort for targeted causal assays

The targeted ablation and activation-restoration assays use a deterministic seed-42 cohort of 256 test
examples, balanced `128/128` across non-promoter/promoter labels. Reusing one locked cohort makes the
ablation and restoration conditions directly comparable. It does not substitute for multi-seed or
multi-cohort robustness.

## D18 — Intervene on `L0H8` before the attention output projection and on the Layer-0 MLP output

For the 12-head, 768-dimensional encoder, head 8 occupies concatenated head slice `[512:576]`. The head
intervention zeros that slice immediately before `attention.output.dense`. The MLP intervention zeros the
Layer-0 `mlp.wo` output before residual addition and LayerNorm. These sites are chosen so the interventions
target the branch outputs directly and are reproducible.

## D19 — Restoration is evaluated in the joint-ablation background

Clean `L0H8` and `L0_MLP` activations are cached, both branches are ablated, and then H8 alone, MLP alone,
or both are restored. This design asks how much of the jointly corrupted computation each clean branch can
recover. It is intentionally not used to rank branches as context-free causes: independent ablation and
joint-background restoration answer different questions.

## D20 — Align activation sources by nucleotide overlap, not raw BPE token index

DNABERT-2 token boundaries vary across sequences, so transplanting activations by token index would mix
different nucleotide spans. For each recipient token, the fine-tuned donor-matrix assay uses tokenizer
offsets to find overlapping source-token nucleotide intervals and inserts an overlap-weighted source
activation; `[CLS]` and `[SEP]` map to their corresponding special tokens. The primary contrast is
same-class versus opposite-class source compatibility.

## D21 — Use a paired motif-versus-mirrored-control TATA assay

For each eligible `TATA[AT][AT]` promoter, the earliest occurrence is selected and replaced with `CGCGCG`.
A mirrored 6-nt control window in the same sequence receives the same replacement, shifted deterministically
if it overlaps the motif. The primary statistic is excess motif damage:
`(baseline - motif_perturbed) - (baseline - control_perturbed)`. This controls local edit magnitude but is
not the same perturbation design as D11's frozen 70-nt assay.

## D22 — Cross-state comparison is descriptive until task and readout are matched

The completed fine-tuned package does not make D1/D2 and D15 the same experiment. Frozen and fine-tuned
states differ in GUE task configuration, sequence length, cohort treatment, and primary readout, and the
frozen Layer-2 working choice is not a stable localization result. Therefore the current comparison is
allowed to ask what each analyzed state looks like internally, but not to identify fine-tuning as the
unique cause of their difference. A causal fine-tuning comparison requires a matched assay, ideally
evaluating the exact pre-fine-tune parent and fine-tuned checkpoint on the same examples under the same
representation/intervention definitions.


## Known departures from repository convention

Recorded rather than quietly tolerated. Each blocks promotion of the affected rows.

| Convention | Departure | Consequence |
|---|---|---|
| 1 — prereg → smoke → confirmatory → claim | **No preregistration was ever frozen** for this arc, though its prose said "preregistered". | `DBPR8`. No `DBP` row may be cited as preregistered; the family is unaudited in `docs/PREREG_CONFORMANCE.md`. A protocol must be frozen before the next confirmatory run. |
| 2 — memorials that can fail, on the measurement path | Identity controls are genuine and exact, but there are no arithmetic anchors of the `value_linearity` kind. | Identity-zero is necessary and was met; it is not sufficient as a memorial set. |
| 3 — calibration for identification claims | **No planted-positive sensitivity and no permuted-label FPR** for the head ranking, group selection, or transplant matrix. | The identification claims (DBP5 especially) stay candidates. Calibration has caught decision-layer bugs in this repo before, and it needs GPU runs, so it is a gate rather than something this review could close. |
| 4 — held-out plus multiple seeded draws | Validation→test discipline honoured; **single seed (42) throughout**. | Every effect is single-seed. Effect-existence is not established for any of them, and the endpoint margin is inside noise. |
| 7 — every result writes a manifest | **Fixed forward, gap preserved backward.** `src/gfmcd/dnabert/manifest.py` is now wired into all 15 canonical runners, so a rerun writes hashed inputs/outputs, the probe sha256, split provenance, seed, command and git state; `tests/test_dnabert_manifest.py` pins the contract. The ~104 *existing* artifacts still have none. | Historical rows carry a blank `manifest` column rather than a back-filled record — the run conditions were never captured, and inventing them would be worse than an honest blank. They fill in when the affected assays are next re-run. |
