# DNABERT-2 promoter study — design decisions

> Why the assays are shaped this way. Not a claim record; claims are in
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


## Known departures from repository convention

Recorded rather than quietly tolerated. Each blocks promotion of the affected rows.

| Convention | Departure | Consequence |
|---|---|---|
| 1 — prereg → smoke → confirmatory → claim | **No preregistration was ever frozen** for this arc, though its prose said "preregistered". | `DBPR8`. No `DBP` row may be cited as preregistered; the family is unaudited in `docs/PREREG_CONFORMANCE.md`. A protocol must be frozen before the next confirmatory run. |
| 2 — memorials that can fail, on the measurement path | Identity controls are genuine and exact, but there are no arithmetic anchors of the `value_linearity` kind. | Identity-zero is necessary and was met; it is not sufficient as a memorial set. |
| 3 — calibration for identification claims | **No planted-positive sensitivity and no permuted-label FPR** for the head ranking, group selection, or transplant matrix. | The identification claims (DBP5 especially) stay candidates. Calibration has caught decision-layer bugs in this repo before, and it needs GPU runs, so it is a gate rather than something this review could close. |
| 4 — held-out plus multiple seeded draws | Validation→test discipline honoured; **single seed (42) throughout**. | Every effect is single-seed. Effect-existence is not established for any of them, and the endpoint margin is inside noise. |
| 7 — every result writes a manifest | **Fixed forward for runners.** `src/gfmcd/dnabert/manifest.py` is wired into the canonical frozen runners, so a rerun records hashed inputs/outputs, probe sha256, split provenance, seed, command, and git state. Most historical frozen artifacts still predate this contract. | Historical rows keep blank manifest fields rather than invented provenance. Closing this gap requires rerunning the affected assays; until then every DBP claim carries the provenance limitation. |
