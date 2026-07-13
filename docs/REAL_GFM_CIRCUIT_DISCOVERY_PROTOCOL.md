# Real-GFM circuit discovery protocol

**[METHOD]** — the repository-wide protocol for taking the instruments validated on toy models and
pointing them at a pretrained genomic foundation model.
**Read this if** you are starting a study on a new GFM. This is the asset the next study springs from.
**Status:** current. Rewritten 2026-07-12 to absorb the gate ladder and claim ladder from the (now
archived) SpliceBERT v1 plan, and to add §2 — the task-degeneracy audit, which exists because its
absence cost us an entire study.

Worked instance: `studies/splicebert_partner_selection/`. Read its `CLAIMS.md` — especially the
retraction rows — before you design anything.

---

## 0. The one rule

> **The model will win your task by the cheapest route that exists. Your job, before you measure
> anything, is to enumerate the routes.**

Everything below is machinery for doing that enumeration honestly.

---

## 1. The seven questions, in order

Do not skip forward. Question *n* is not interpretable until question *n−1* has passed.

1. **Does the model perform the behavior?** Use the model's native output where possible; otherwise the
   smallest registered readout.
2. **Does it beat explicit shortcuts?** Name and fit the shortcut class. Never infer its ceiling from
   intuition — fit it.
3. **Does it follow the relational cue under conflict?** Ordinary in-distribution accuracy is worthless
   when the local and relational rules agree. Make them disagree.
4. **Which components are necessary?** Multiple intervention nulls, matched random component sets.
5. **Which activations mediate the behavior?** Patch clean activations into aligned corrupt inputs with a
   verified behavior gap.
6. **How does information flow?** Edge/path interventions and QK/OV decomposition.
7. **Does the mechanism predict biology outside the model?** Freeze the circuit metric *before* opening
   the external data.

---

## 2. The task-degeneracy audit — run this FIRST

**This section is written in blood.** The SpliceBERT partner-selection study passed a behavioral gate
with a result of 0.807 vs 0.056 against a randomised control, was declared GO, and was **retracted in
full** three weeks later. Two independent confounds, either one fatal:

- a **positional leak** — the answer sat at a constant index, so a *geometry oracle reading zero
  nucleotides* scored a perfect 1.000; and
- **site detection sufficed** — the decoys were spurious motifs, so an *oracle site detector* also
  scored 1.000. The task never required the relation it was named after.

Neither was caught by the registered controls, including the randomised-backbone control, because a
random network does not expose position linearly and so cannot detect a positional leak.

Before you fit a single model, build these adversaries and try to **beat your own task without doing the
science**. Each one must score at chance. If any of them scores well, the task is degenerate — fix it or
kill it.

| Adversary | What it reads | Catches |
|---|---|---|
| **Geometry oracle** | positions only — **zero nucleotides** | the answer is pinned to a constant index or window edge |
| **Length oracle** | `len(sequence)` and nothing else | the label is recoverable from sequence length |
| **Site-detection oracle** | the true motif positions, no partner information | the task collapses to "find the site", not "find the relation" |
| **Local-only model** | the target's own neighbourhood, distal context replaced with `N` | the relation is not needed |
| **Label-shuffled model** | everything, with permuted labels | the pipeline leaks the label |
| **Class-prior model** | the marginal label distribution | the negatives are not matched |

Then ask the question no adversary can ask for you:

> **Do clean negatives exist in the biology at all?**

This is what killed the SpliceBERT v2 fix. Its decoys were real annotated donors — but *minor* ones
(median 1 supporting isoform vs 18 for true donors). "Pick the partner" quietly became "pick the better
donor", which a plain PWM does at 0.687 against 0.392 chance. In a real genome a strong donor at a
plausible distance from an acceptor generally **is** its partner. The negatives you need do not exist,
and no amount of engineering conjures them.

**When a forced-choice task is degenerate, do not patch it. Remove the task.** The design that worked
has no task, no label, and no decoys: read the model's **native** output, perturb the thing you care
about, substitute **real bases** (never `[MASK]`, which is off-distribution), and score every effect
against a matched control. There is nothing to win, so there is nothing to leak.

---

## 3. Non-negotiable controls

- Planted positive **and negative** mechanism controls. A positive control alone cannot falsify a tracer.
- An untrained/randomised-backbone control — **and verify it is not a no-op.** Ours was. `init_weights()`
  does nothing after `from_pretrained` on transformers ≥ 5; the "random" backbone was byte-identical to
  the pretrained one. Assert that the control's outputs actually differ.
- A registered local/additive baseline, fit once on train and never touched again.
- A natural cue-conflict set, frozen before tracing.
- Matched ordinary examples, to detect global degradation masquerading as specificity.
- At least two ablation nulls (zero / mean / resample).
- Matched random component sets, size-matched.
- Multiple seeds.
- Grouped confidence intervals, by gene or transcript — never by example.
- Explicit, single test-set unblinding.

---

## 4. The model-with-readout rule

When a pretrained model has no native output for your task, **the readout is part of the experimental
system, not a neutral window onto it.**

| Regime | What a positive result licenses |
|---|---|
| frozen backbone + **constrained additive** readout | the information is already exposed in the contextual representation |
| frozen backbone + **flexible interaction** readout | a diagnostic upper bound. The readout can *synthesise* the relation; this alone proves nothing about the backbone |
| **fine-tuned** backbone | supervision may have *installed* the computation. This is a circuit-formation result, not a pretrained-circuit result |
| **randomised** backbone, same readout | controls for architecture and readout capacity |

Report these separately, and trace only the regime that matches the claim you intend to make.

**Prefer a native readout to any of them.** A native masked-LM readout has no capacity of its own to
manufacture your effect. It is why the coupling design survived where the partner-selection design died.

---

## 5. The counterfactual design rule

A counterfactual is valid only when **all** of these hold:

- the intended cue changes;
- protected nuisance features are measured and matched (length, composition, motif strength, distance);
- token coordinates stay aligned for patching, or a registered alignment map is supplied;
- the clean–corrupt behavior gap is **verified nonzero before you patch** — patching into a gap that
  does not exist recovers nothing and means nothing;
- the edit recipe is stored per example;
- substitutions use **real bases**, not `[MASK]` or `N`, wherever the model's training distribution cares;
- negative and unintended effects are reported.

---

## 6. The gate ladder

A gate that fails stops the ladder. It does not get worked around.

| Gate | What it establishes | Pass condition |
|---|---|---|
| **0 · Instrument calibration** | The tracer works, and its error rate is known | Recovers the known path in the planted compositional twin, **rejects it** in the behaviorally-identical shortcut twin, and documents false positives/negatives. Add a LayerNorm-bearing planted model if the target has LayerNorm — **ours does not yet, and this is the repo's last open Gate-0 hole** |
| **1 · Dataset audit** | The data says what you think | Zero coordinate errors in tests *and* in a manually reviewed stratified sample, both strands; no gene or chromosome leakage; **§2's degeneracy audit passes** |
| **2 · Freeze baselines** | The shortcut class is fixed before you can be tempted | Baseline fit on train only, hashed, immutable; conflict set frozen |
| **3 · Behavior** | There is something to trace | Beats the registered baseline with a **gene-grouped** CI above zero; advantage persists on the conflict set; exceeds the randomised backbone; ≥ 3 seeds |
| **4 · Component localization** | Which parts matter | Effect positive under ≥ 2 ablation nulls; the grouped set beats 95% of size-matched random sets; not explained by global degradation on matched ordinary examples |
| **5 · Mediation** | The parts carry the behavior | Verified clean–corrupt gap; positive recovery; recovery above matched random sets |
| **6 · Path** | How information flows | QK selection and OV writes decomposed; edge/path patching separates parallel heads from a serial circuit |
| **7 · Necessity, sufficiency, completeness** | The circuit account is honest | Necessity + mediation + **measured** residual. "Compact circuit" never means "all computation captured" |
| **8 · Replication** | It is not one seed's accident | ≥ 3 seeds; replicate at the level of *computation*, not head number — different seeds route the same computation through different heads |
| **9 · External validation** | It is biology, not just mechanism | Circuit metric frozen **before** the external data is opened; incremental predictive value over local baselines on held-out genes |

**Attribution, attention maps, probes and SAE features are discovery aids. None is causal evidence on
its own.** Attribution ranks; only intervention decides.

---

## 7. The claim ladder

Use these exact labels in result files and in prose. Never promote a claim because a visualisation
"looks right".

| Label | Minimum evidence |
|---|---|
| `REPRESENTATION_EVIDENCE` | A probe or readout can recover the variable. |
| `BEHAVIORAL_RELATION_EVIDENCE` | The model beats registered shortcuts on conflicts and context controls. |
| `COMPONENT_LOCALIZATION` | Individual or grouped interventions move the target metric. |
| `CAUSAL_MEDIATION` | Clean activations patch the behavior into aligned corrupt inputs. |
| `CIRCUIT_CANDIDATE` | A replicated directed component/path account, with necessity and mediation. |
| `CAUSALLY_VALIDATED_CIRCUIT` | Necessity, sufficiency/mediation, specificity, random controls, replication, and **measured incompleteness**. |
| `BIOLOGICAL_PREDICTOR` | The circuit metric predicts an independent biological outcome beyond local controls. |
| `BIOLOGICAL_DISCOVERY_CANDIDATE` | A novel rule generated from the circuit survives independent experimental validation. |

A passing unit test verifies **software**, never a scientific claim. It can advance a claim to
`unit-tested`; it can never advance one to `supported`. Only a fresh, manifest-stamped artifact does that.

---

## 8. Stop rules

Stop or reframe rather than forcing a circuit narrative. Each of these is a legitimate result.

1. **No behavioral advantage** → do not trace a pretrained circuit. Say the model does not provide
   evidence for the mechanism on this assay.
2. **Readout-only interaction** → the representation *permits* a downstream interaction; the backbone is
   not shown to compute it.
3. **Fine-tuning-only effect** → the paper is now about circuit *installation during adaptation*.
4. **Global damage** → if ablating the candidate heads harms ordinary examples as much as conflicts, the
   heads are not specific and this is not your circuit.
5. **No mediation** → necessity without patch recovery is involvement, not a path account.
6. **Seed instability** → report a fragile or multiply-realisable mechanism. Do not hide failed seeds.
7. **No external increment** → keep the mechanistic claim, drop the biological-discovery claim.
8. **The task is degenerate** (§2) → remove the task. Do not patch it.

---

## 9. The result registry

Every settled result is generated from a run manifest carrying: commit hash, dirty status, config, input
hashes, checkpoint hash, seeds, command, output hashes, and claim status.

> **Markdown summaries are *views* of artifacts under `results/`. They are never an alternative source
> of truth.** If a markdown file and its artifact disagree, the artifact wins and the markdown is stale.

---

## 10. Checklist — adding a new real-GFM study

1. Copy `studies/TEMPLATE_REAL_GFM_STUDY/` to `studies/<study_id>/`.
2. Define **one** behavior and **one** biological relation. A list of loosely-related targets is how a
   study becomes unfalsifiable.
3. **Run the §2 degeneracy audit before anything else.** Write down the strongest local/additive shortcut
   class, and the adversaries, and beat your own task with them.
4. Prefer a **native** model output. If you must add a readout, register every regime in §4 separately.
5. Specify natural *and* controlled cue conflicts.
6. Implement an adapter under `src/gfmcd/adapters/` that validates token alignment, layer access, and
   model metadata.
7. Put domain-specific schemas, datasets, baselines, perturbations, and metrics in a dedicated source
   package.
8. Add tests for coordinates, strand/orientation, token mapping, and intervention hooks — and add a
   **memorial test** for every bug that ever produced a retracted claim. *A guard that does not run is
   not a guard: `tests/test_partner_data.py` is currently failing, and it is the guard on the leak that
   cost us the study.*
9. Freeze validation conflict examples before circuit tracing.
10. Promote claims only through the §7 ladder: behavior → necessity → mediation → path → replication →
    external validation.
11. Keep a `CLAIMS.md` with permanent retraction rows. Later analyses refine; they never overwrite.
