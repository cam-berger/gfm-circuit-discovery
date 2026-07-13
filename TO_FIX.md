# TO_FIX — Scientific Corrections, Reproducibility, and the SpliceBERT Transition

**Repository:** `gfm-circuit-discovery`  
**Purpose:** implementation specification for a terminal coding agent  
**Priority:** correctness first, reproducibility second, expansion third  
**Rule:** do not make a claim stronger because the prose sounds better. A result is only promoted after the corresponding command, artifact, and test exist.

---

## 0. What this cleanup must accomplish

This pass has two goals:

1. **Make the current repository scientifically honest and reproducible.** Correct the places where the prose overstates what the code or experiments establish.
2. **Turn the repository into a reusable circuit-discovery framework.** Add a study template and begin a first real pretrained-model study: partner-aware splice-site computation in **SpliceBERT**.

The cleaned repository should tell this story:

> We have a calibrated causal instrument, controlled genomic tasks, and cue-conflict assays that distinguish relational computation from local or pooled shortcuts. We have not yet established a universal “sufficiency law,” a sequential reading-frame counter, or an unforced repeat-copy circuit. The next paper-sized objective is to use the calibrated framework to discover and causally validate a partner-aware splicing circuit in pretrained SpliceBERT, then test whether that circuit predicts real variant-induced splicing effects.

---

## 1. Non-negotiable working rules

The agent must follow these rules throughout the cleanup.

### 1.1 Never invent or silently update a result

- Do not change a number unless the generating experiment was run.
- Do not mark a result `SETTLED` because a script exists.
- Do not infer that an old checkpoint still reproduces a result without actually running the current code against it.
- If required data or checkpoints are missing, mark the result `BLOCKED` or `PROVISIONAL`, state what is missing, and leave the old number clearly labeled as historical.

### 1.2 Preserve the correction history

- Do not delete withdrawn or retracted claims from `docs/RESULTS.md` or `docs/DECISIONS.md`.
- Old proposal files may be moved to `docs/archive/`, but must not be rewritten to make the historical record look cleaner.
- Add a banner to archived proposals: `SUPERSEDED — retained for provenance; do not cite as current project claims.`

### 1.3 One logical change per commit

Recommended branch:

```bash
git checkout -b cleanup/scientific-claims-and-splicebert
```

Recommended commit sequence appears in Section 12.

### 1.4 Every result needs an artifact trail

A result is eligible for active prose only when it has:

- an experiment configuration;
- the exact command used;
- a seed;
- data and model provenance;
- a metrics file;
- the current git commit hash;
- and a status: `EXPLORATORY`, `PROVISIONAL`, `REPLICATED`, `SETTLED`, `WITHDRAWN`, or `RETRACTED`.

### 1.5 Active documentation must have one source of truth

Use `docs/RESULTS.md` as the human-readable source of truth and add a machine-readable registry under `results/registry/`. The README and proposal must summarize results from that source rather than independently maintaining different versions of the story.

---

## 2. Establish and record the baseline before editing

Run these commands from the repository root:

```bash
git status --short
git rev-parse HEAD
python --version
uv --version
uv sync --extra lab --extra dev --extra models --extra genome
uv run pytest --collect-only -q
uv run pytest -q
```

Create `artifacts/audit/baseline.md` containing:

- current commit hash;
- Python and `uv` versions;
- the number of tests collected;
- the complete test command;
- pass/fail/skip totals;
- missing data/checkpoint warnings;
- operating system and accelerator.

Do **not** hard-code the test count into README prose. The uploaded repository currently collects 131 tests, while active documents mention 102, 127, and approximately 130. Replace those stale claims with the command a user should run.

Acceptance check:

```bash
rg -n "102 tests|127 tests|130 tests|~130 tests" README.md docs
```

The command should return no active-document claims that are intended to stay current.

---

# Part I — Correct the current scientific claims

## 3. Replace the “sufficiency law” with the shortcut-gap hypothesis

### 3.1 Problem

The repository currently states an if-and-only-if law:

> A relational circuit forms iff no pooled statistic is sufficient.

The experiments do not establish an iff law. Removing a known shortcut does not guarantee that optimization will find the intended mechanism; a model may underfit, memorize, learn another heuristic, or lack the necessary capacity. A circuit may also form even when a shortcut is available because of pretraining, regularization, architecture, or optimization history.

### 3.2 Required terminology

Use **shortcut-gap hypothesis** in active documentation.

Recommended definition:

> The probability that a model learns a relational mechanism should increase with the objective advantage of that mechanism over a pre-specified shortcut class, while also depending on example frequency, architecture, capacity, regularization, initialization, and training budget.

Define the measured quantity as:

```text
shortcut_gap = loss(best registered shortcut) - loss(best relational solution)
```

This is a hypothesis to test, not a settled law.

### 3.3 Files to inspect and edit

At minimum:

- `README.md`
- `docs/RESULTS.md`
- `docs/REC.md`
- `docs/FOUNDATIONS_circuit_discovery.md`
- `docs/Proposal_Sufficiency_Circuit_Discovery.md`
- `docs/EXPERIMENTS.md`
- `docs/PLAN.md`
- `docs/PLAN_reading_frame.md`
- `docs/PLAN_repeat_copy.md`
- `docs/BIOLOGY_genomic_computations.md`
- generated notebook source in `scripts/build_notebooks.py`
- executed notebooks if they contain the claim

Search:

```bash
rg -n -i "sufficiency law|if and only if|\biff\b|knife edge|only when no|must mechanize" \
  README.md docs scripts/build_notebooks.py notebooks
```

Do not blindly replace every `iff`; inspect the mathematical context. Hand-built circuit definitions may legitimately use iff conditions.

### 3.4 Required status change in `docs/RESULTS.md`

Replace the current one-line `SETTLED` law with a section similar to:

```markdown
## 0. Current organizing hypothesis

**HYPOTHESIS, NOT YET A LAW.** Across the controlled tasks completed so far, relational
mechanisms become more useful as the best registered shortcut leaves more objective value
unclaimed. We call this the shortcut-gap hypothesis. Current experiments support the direction
of the hypothesis but do not establish necessity, sufficiency, or an if-and-only-if threshold.
```

### 3.5 Acceptance criteria

- No active file calls the idea a settled law.
- Every active summary calls it a hypothesis.
- The shortcut class is explicitly registered per experiment rather than described as “all pooled statistics.”
- The results document separates observed results from the general hypothesis.

---

## 4. Correct the repeat-copy objective claim

### 4.1 Problem

`src/gfmcd/data/repeat_copy.py` says the model uses an ordinary causal next-token objective and the proposal says the circuit formed “unbidden,” “without coercion,” or “with no forcing objective.”

The actual training script uses:

- `weighted_ar_loss`;
- default `--copy-weight 8.0`;
- an 8× weight on copy positions.

This is a legitimate forcing signal. It does not invalidate the circuit trace, but it changes the claim.

### 4.2 Correct wording

Use:

> A causal nucleotide transformer learned a distributed repeat-copy circuit under a copy-position-weighted autoregressive objective. The current default gives copy positions 8× the per-token weight of background positions. Whether the same mechanism emerges under an unweighted causal language-model objective remains untested.

Do not use:

- “ordinary causal objective”;
- “unbidden”;
- “no forcing objective”;
- “without coercion”;
- “forms on its own,” unless the copy weight is 1 and the result is independently reproduced.

### 4.3 Required code and documentation edits

1. Fix the module docstring in `src/gfmcd/data/repeat_copy.py`.
2. Keep the accurate training-script docstring in `scripts/train_repeat_copy.py`.
3. Update all active prose and figure captions.
4. Update notebook text through `scripts/build_notebooks.py`, then regenerate notebooks.
5. Update the repeat-copy branch documents:
   - `branches/codex-repeat-induction/FINDING.md`
   - `COPY_CEILING.md`
   - `LOSS_CURVES.md`
   - `IMPLEMENTATION.md`
   - `NATURAL_REPEATS.md`
6. Rename figure headings that say “forms unbidden.”

Search:

```bash
rg -n -i "unbidden|no forcing|without coercion|ordinary causal|forms on its own|not forced" . \
  --glob '!docs/archive/**' --glob '!*.png' --glob '!uv.lock'
```

### 4.4 Add objective-accounting metrics

Add a helper such as:

```python
def effective_copy_loss_share(copy_mask: torch.Tensor, copy_weight: float) -> float:
    copy_tokens = copy_mask.sum().item()
    background_tokens = copy_mask.numel() - copy_tokens
    return (copy_weight * copy_tokens) / (copy_weight * copy_tokens + background_tokens)
```

Log and save:

- raw copy-token fraction;
- copy weight;
- effective copy-loss share;
- copy-position CE;
- background-position CE;
- all-token CE;
- copy accuracy;
- cue-conflict distal and local follow rates.

### 4.5 Add tests

At minimum:

- `copy_weight=1` exactly matches ordinary mean cross-entropy within numerical tolerance;
- effective share is monotonic in `copy_weight`;
- checkpoint metadata records weight, repeat frequency, data seed, model seed, and token fraction.

### 4.6 Run the missing sweep

Add `configs/experiments/repeat_copy_objective_sweep.toml` and a runner.

Minimum grid:

- copy weight: `1, 2, 4, 8`;
- at least 5 model seeds;
- at least 2 repeat densities or copy-token fractions;
- matched model, optimizer, token budget, and evaluation set.

Primary outputs:

- probability of achieving copy accuracy above the registered threshold;
- time/steps to transition;
- cue-conflict behavior;
- causal circuit effect;
- all-token and copy-token CE.

Do not update the headline claim until the weight-1 condition has been run.

---

## 5. Rename the reading-frame result until a sequential counter is demonstrated

### 5.1 Problem

GFM-nano uses learned absolute position embeddings. The current evidence shows an anchor-relative modulo-3 computation, but it does not prove that the model propagates a sequential count from one position to the next. It may combine the anchor position and query position through a compact residue-class lookup.

### 5.2 Required terminology

For trained GFM-nano results, use:

- **anchor-relative modulo-3 circuit**;
- **anchor-relative frame circuit**;
- **phase-maintenance mechanism**.

Reserve **sequential counter** for a mechanism that passes the tests in Section 5.4.

The hand-built planted model may still be called a counter if it is explicitly constructed to implement one.

### 5.3 Correct overstatements

Replace statements like:

- “MLPs count”;
- “finds the start codon and counts position from it”;
- “counter circuit emerges”;

with wording such as:

> The model uses the start anchor and develops a linearly recoverable relative phase representation across depth. The present evidence does not distinguish iterative counting from an anchor/position residue computation.

Search:

```bash
rg -n -i "MLPs count|counts position|counter circuit|reading-frame counter|sequential counter" \
  README.md docs branches scripts/build_notebooks.py notebooks
```

Inspect every hit. Do not rename files immediately if doing so would break imports or links; terminology in prose is the first priority.

### 5.4 Add counter-versus-lookup experiments

Create `docs/studies/reading_frame/COUNTER_VS_LOOKUP.md` and corresponding configs.

Required tests:

1. **Held-out anchor positions**  
   Train with a subset of absolute anchor positions and test on unseen positions.
2. **Joint translation**  
   Translate anchor and query together while keeping relative distance fixed.
3. **Length extrapolation**  
   Test beyond trained sequence lengths or absolute-position ranges when architecture permits.
4. **Position-encoding intervention**  
   Compare learned absolute embeddings to relative-position or rotary variants while holding the task fixed.
5. **Transition consistency**  
   Test whether the phase representation at position `i+1` is produced by a stable transformation of the representation at `i`.
6. **Composition test**  
   Apply the inferred transition repeatedly and test whether it composes modulo 3.
7. **Patch-forward test**  
   Patch the putative phase state from one position into a downstream position and test the predicted phase shift.

Pre-register the distinction:

- **Sequential counter:** passes transition, composition, held-out-position, and extrapolation tests.
- **Anchor/position lookup:** solves cue conflict but fails transition or extrapolation tests.
- **Unresolved anchor-relative circuit:** behavior is anchored but internal implementation is not separable with current evidence.

---

## 6. Mark the reading-frame 2×2 as provisional until it is reproducible

### 6.1 Problem

The active proposal calls the 2×2 complete, while `docs/RESULTS.md` calls it in flight and `docs/TODO.md` records that cells C/D were read at the wrong layer. Required genome data and checkpoints are not included, so the claims cannot be independently regenerated from a clean checkout.

### 6.2 Required status

Until the data, checkpoints, and output-level analysis are restored:

- mark the 2×2 **PROVISIONAL / REPRODUCTION BLOCKED**;
- do not use it as a headline contribution;
- do not state that three of four cells contain a counter as a settled result;
- keep the historical values in `docs/RESULTS.md` with the caveat and exact missing artifacts.

### 6.3 Repair the measurement

Implement the existing TODO correctly:

- real-CDS cells must use output-level cue-conflict behavior;
- use peak anchor dependence rather than a fixed `L2.resid_post` readout;
- add a guard that refuses to classify a non-uniform real-CDS model from the fixed L2 probe;
- regenerate the table from current code and pinned artifacts;
- run multiple seeds or state clearly when only one seed exists.

### 6.4 Acceptance criteria

The 2×2 may return to `REPLICATED` only when:

- hg38 and GENCODE can be fetched from a clean checkout;
- every cell has a pinned config and checkpoint hash;
- the output-level metric is used;
- the same verdict is obtained from at least two independent causal measurements;
- the result is reproduced across the pre-registered number of seeds.

---

## 7. Correct the Caduceus prediction

### 7.1 Problem

The proposal says reverse-complement equivariance makes Caduceus structurally blind to CTCF orientation. This confuses **equivariance** with **invariance**.

An RC-equivariant model may preserve orientation in a representation that transforms predictably under reverse complementation. A convergent motif pair remains convergent after the entire sequence is reverse-complemented.

### 7.2 Required edit

Remove the “orientation blindness” prediction from all active documents.

Replace it with:

> Caduceus is a valuable later architecture comparison because it tests whether an orientation-sensitive relational computation is represented in an RC-consistent form. RC equivariance does not imply loss of relative orientation. The experiment should test representation and causal use of relative orientation, not assume blindness.

Search:

```bash
rg -n -i "Caduceus.*blind|orientation.*outside.*hypothesis|identified with.*reverse complement|RC-equivariance blindness" \
  README.md docs branches
```

### 7.3 Future experiment

Keep CTCF orientation as a later study only after the SpliceBERT pipeline is working. The correct questions are:

- Does the model distinguish convergent, divergent, and tandem motif pairs?
- Does the representation transform consistently under whole-sequence reverse complementation?
- Does a downstream readout preserve or erase orientation?
- Is any orientation-sensitive mechanism causally used in a loop/contact prediction task?

---

## 8. Distinguish controlled genomic transformers from pretrained foundation models

### 8.1 Required terminology

- **GFM-nano** may remain the code name.
- In scientific prose call it a **controlled genomic transformer trained from scratch**.
- Use **pretrained genomic foundation model** only for models such as SpliceBERT, Nucleotide Transformer, Caduceus, HyenaDNA, Evo, and similar broad pretraining systems.

### 8.2 Correct current claims

Do not say that the current toy-model experiments already discover circuits “inside GFMs.” Instead say they calibrate methods and establish controlled mechanisms that will be carried into pretrained GFMs.

### 8.3 README story

The README opening should follow this order:

1. calibrated planted twins;
2. controlled genomic transformers;
3. causal cue-conflict protocol;
4. first pretrained-model target: SpliceBERT;
5. long-term benchmark and cross-architecture program.

---

## 9. Interpret DART-Eval as motivation, not mechanistic proof

### 9.1 Correct claim

DART-Eval reports that current DNA language models are inconsistent, often matched or exceeded by simpler ab initio models, and especially weak on counterfactual tasks. It does **not** establish that a model is pooling, that a circuit is absent, or why a model fails.

Use:

> DART-Eval motivates a mechanistic audit: when a foundation model fails to beat a strong local or ab initio baseline, is it relying on the same shortcut, failing to represent the relevant relation, or representing it without causally using it?

Do not use:

> DART-Eval proves the models are pooling because pooling suffices.

Update the literature-gap table and all related prose.

---

## 10. Fix SpliceBERT provenance before adding experiments

### 10.1 Model facts to encode

SpliceBERT is a BERT-style primary-RNA language model with:

- six Transformer encoder layers;
- hidden size 512;
- 16 attention heads;
- learned absolute position embeddings;
- single-nucleotide tokenization;
- maximum input length 1024 nt;
- approximately 19.4 million parameters;
- masked-language-model pretraining on over two million primary-RNA sequences from 72 vertebrates.

### 10.2 Do not silently trust a community mirror

`src/gfmcd/config.py` currently points at `yangheng/SpliceBERT-510nt` and comments that official weights are elsewhere. Replace this ambiguous state with explicit provenance.

Add fields to `Substrate` or a new `ModelSource` structure:

```python
source_type: Literal["official_repo", "zenodo", "huggingface", "local"]
source_uri: str
revision: str | None
sha256: str | None
is_official: bool
```

Preferred order:

1. official paper repository or Zenodo release;
2. verified Hugging Face conversion with a documented conversion script and checksum;
3. community mirror only as a fallback, clearly labeled.

Add `scripts/verify_model_source.py` that prints model metadata and verifies the expected architecture.

### 10.3 Contamination language

Keep the useful firewall, but make the wording precise:

- no chromosome is a pretraining holdout for a model trained on human transcripts;
- chromosome splits can still prevent leakage between a supervised development and test set;
- claims about unseen human sequence generalization must be qualified;
- mechanistic claims concern the computation implemented by the released weights;
- external assay data may test biological association, but not erase pretraining exposure to the underlying reference sequence.

---

# Part II — Make the repository reproducible

## 11. Remove machine-specific paths

### 11.1 Add a central path module

Create `src/gfmcd/paths.py`:

```python
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    references: Path
    cache: Path
    checkpoints: Path
    results: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(os.environ.get("GFMCD_ROOT", Path(__file__).resolve().parents[2]))
        data = Path(os.environ.get("GFMCD_DATA_DIR", root / "data"))
        return cls(
            root=root,
            data=data,
            references=Path(os.environ.get("GFMCD_REFERENCE_DIR", data / "references")),
            cache=Path(os.environ.get("GFMCD_CACHE_DIR", data / "cache")),
            checkpoints=Path(os.environ.get("GFMCD_CHECKPOINT_DIR", root / "checkpoints")),
            results=Path(os.environ.get("GFMCD_RESULTS_DIR", root / "results")),
        )
```

Use it everywhere instead of `Path.home() / "Research/data/genomes"` or `cd ~/Research/...`.

Search after editing:

```bash
rg -n "~/Research|/Users/|Path\.home\(\).*Research" . \
  --glob '!logs/archive/**' --glob '!docs/archive/**' --glob '!uv.lock'
```

Only historical logs/archive entries may remain.

### 11.2 Shell scripts

Every shell script must begin with:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
```

---

## 12. Add reference-data fetching and verification

Create:

- `data/references/manifest.json`;
- `scripts/fetch_reference.py`;
- `docs/DATA_PROVENANCE.md`.

The manifest must include:

- assembly name and release;
- official download URL;
- expected filename;
- SHA-256 checksum;
- compression type;
- license/usage note;
- date retrieved.

The fetcher must:

1. download to a temporary file;
2. verify the checksum;
3. atomically move the file into place;
4. refuse to continue on mismatch;
5. support `--verify-only`;
6. print the exact installed versions.

Do not fabricate checksums. Obtain them from the official source when available or compute and pin them after the first verified download, recording how they were obtained.

Add a test that verifies manifest schema without downloading large files.

---

## 13. Add cache and checkpoint manifests

### 13.1 Cache sidecars

Every generated `.npz` cache must have a JSON sidecar containing:

- generator function and version;
- source reference checksums;
- config;
- seed;
- train/validation/test split definition;
- array shapes and dtypes;
- creation timestamp;
- git commit.

The loader must compare requested parameters against the sidecar and refuse stale or mismatched caches.

### 13.2 Checkpoint metadata

Every checkpoint must record:

- model configuration;
- optimizer configuration;
- objective and weights;
- task configuration;
- data seed and model seed separately;
- training step and token count;
- commit hash;
- package versions where practical.

Add a command:

```bash
uv run python scripts/inspect_checkpoint.py path/to/checkpoint.pt
```

It should print metadata without loading the model onto an accelerator.

---

## 14. Add an experiment registry

Use standard-library TOML for configs and JSON for outputs to avoid another dependency.

Recommended layout:

```text
configs/
  experiments/
    reading_frame_uniform.toml
    reading_frame_counter_vs_lookup.toml
    repeat_copy_objective_sweep.toml
    splicebert_partner_choice_pilot.toml

results/
  registry/
    <run_id>/
      manifest.json
      metrics.json
      command.txt
      stdout.log
      figures/
```

Create `src/gfmcd/experiments/manifest.py` and a helper that writes:

```json
{
  "run_id": "...",
  "study": "repeat_copy",
  "status": "EXPLORATORY",
  "git_commit": "...",
  "config_path": "...",
  "model_source": "...",
  "data_sources": [],
  "seeds": {"model": 0, "data": 1},
  "command": "...",
  "started_at": "...",
  "completed_at": "..."
}
```

Add `scripts/list_results.py` and `scripts/render_results_summary.py` so `docs/RESULTS.md` can include generated tables or verified snippets rather than manually copied numbers.

---

## 15. Add CI and test categories

Create `.github/workflows/ci.yml`.

Pull-request CI should run:

```bash
uv sync --extra dev --extra lab
uv run pytest -q -m "not genome and not checkpoint and not slow"
```

Add pytest markers in `pyproject.toml`:

- `genome` — requires downloaded reference genome;
- `checkpoint` — requires external model checkpoints;
- `slow` — long optimization loop;
- `network` — downloads model/data;
- `gpu` — requires CUDA/MPS.

Add a scheduled/manual workflow for the broader suite where resources are available.

The CI must not download large models or genomes during ordinary pull requests.

---

## 16. Add the missing LayerNorm calibration

The tracer was calibrated on LayerNorm-free planted models while GFM-nano and SpliceBERT use LayerNorm.

Create a planted LayerNorm control that preserves a known circuit and known null pathway. Measure:

- direct-path attribution conservation error;
- edge recall and precision;
- causal effect predicted versus actual;
- sensitivity to frozen versus recomputed LayerNorm statistics;
- error as intervention magnitude increases.

Do not call Gate 0 complete until this test is in `docs/RESULTS.md` with a status and reproducible artifact.

---

# Part III — Organize the repository for repeatable studies

## 17. Add a study template instead of adding one-off scripts forever

Create:

```text
docs/studies/
  INDEX.md
  TEMPLATE/
    QUESTION.md
    PREREGISTRATION.md
    DATA_CARD.md
    MODEL_CARD.md
    BASELINES.md
    CUE_CONFLICTS.md
    INTERVENTIONS.md
    RESULTS.md
    LIMITATIONS.md
  reading_frame/
  repeat_copy/
  splicebert_partner_choice/
```

Do not immediately delete legacy documents. `docs/studies/INDEX.md` should map old paths to the new organization. Move files with `git mv` only after links and notebooks are updated.

Every future study must specify:

1. target computation;
2. model behavior that demonstrates competence;
3. registered shortcut class;
4. shortcut ceilings;
5. cue-conflict generator;
6. scalar causal metric;
7. intervention targets;
8. necessity test;
9. sufficiency/mediation test;
10. replication plan;
11. external biological validation;
12. explicit failure interpretations.

### 17.1 Code interfaces

Add lightweight protocols or abstract base classes under `src/gfmcd/studies/`:

```python
class BehaviorAssay(Protocol):
    def score(self, model, batch) -> dict[str, float]: ...

class CueConflict(Protocol):
    def build(self, batch, seed: int): ...

class ShortcutBaseline(Protocol):
    def fit(self, train): ...
    def score(self, test) -> dict[str, float]: ...

class InterventionAssay(Protocol):
    def necessity(self, model, batch, components): ...
    def sufficiency(self, model, clean, corrupt, components): ...
```

Keep these interfaces small. Do not refactor all existing code before a SpliceBERT pilot proves what abstractions are actually needed.

---

# Part IV — First pretrained-model path: SpliceBERT

## 18. Paper-sized objective

The first real-model paper should target one claim:

> **From calibrated synthetic mechanisms to a causally validated partner-aware splice-site circuit in pretrained SpliceBERT.**

The project should determine whether SpliceBERT merely encodes local splice-site motifs, or causally uses a relation between splice partners when local evidence and pairing evidence disagree.

This is a better first target than CTCF or a state-space model because SpliceBERT is small, single-nucleotide, BERT-based, biologically focused, and close enough to GFM-nano that the current hook and intervention infrastructure can be adapted.

---

## 19. Add the SpliceBERT adapter

Recommended layout:

```text
src/gfmcd/models/adapters/
  __init__.py
  base.py
  splicebert.py

src/gfmcd/interventions/
  __init__.py
  hooks.py
  bert.py
```

The adapter must expose:

- token IDs and nucleotide-to-token alignment;
- embeddings and positional embeddings;
- per-layer residual input/output;
- Q, K, V projections;
- attention logits and probabilities;
- per-head output before and after output projection;
- MLP intermediate and output;
- final MLM logits;
- safe, self-restoring activation and head interventions.

Smoke tests:

- exact output agreement before and after installing/removing hooks;
- token alignment for A/C/G/T/N and special tokens;
- shape checks: 6 layers, 16 heads, hidden size 512;
- sequence length guard at 1024 tokens including special tokens;
- head ablation changes only the intended head write;
- repeated runs are deterministic in evaluation mode.

Do not begin circuit claims until the hook layer passes these tests.

---

## 20. Build the splice-partner dataset

Create:

```text
src/gfmcd/data/splicing.py
scripts/build_splice_partner_dataset.py
configs/experiments/splicebert_partner_choice_pilot.toml
docs/studies/splicebert_partner_choice/DATA_CARD.md
```

### 20.1 Initial dataset

Start with GENCODE canonical introns that fit within SpliceBERT’s context:

- canonical GT donor and AG acceptor;
- intron plus flanking exonic sequence within the 1024-token limit;
- record donor, acceptor, branch-region, exon boundaries, transcript, gene, strand, and sequence source;
- exclude ambiguous bases for the first pilot;
- group splits by gene to prevent related transcripts from crossing supervised train/test splits.

A chromosome split is useful for supervised evaluation hygiene, but must not be described as a pretraining holdout.

### 20.2 Matched decoys

For every true partner, construct decoys matched on as many local cues as possible:

- canonical dinucleotide;
- local motif score;
- GC content;
- distance distribution;
- candidate count;
- repeat content;
- local k-mer composition.

Include at least:

- donor with intronic decoy AG candidates;
- acceptor with decoy GT candidates where applicable;
- matched-distance random sites;
- local-motif-preserving but partner-swapped examples.

Save the exact matching diagnostics. If matching fails, do not treat the set as a cue conflict.

---

## 21. Define three behavior levels

### 21.1 Level A — zero-shot dependency

Implement substitution-based nucleotide dependency:

- perturb a query nucleotide or splice-site dinucleotide;
- measure the change in log probability or log odds at a candidate partner;
- score true-partner dependency against matched decoy dependency.

This reproduces the type of phenomenon already reported in nucleotide-dependency work, then gives the repository a scalar quantity to trace causally.

### 21.2 Level B — frozen-backbone readout

Train a small registered readout on frozen SpliceBERT representations to choose the true partner among matched candidates.

Use a deliberately limited readout first:

- linear or bilinear pair score;
- no deep MLP that could install a new relational computation;
- report whether local-only, additive, and interaction terms are needed.

### 21.3 Level C — fine-tuned model

Only after Levels A and B are characterized, fine-tune the backbone for partner choice or variant-effect prediction.

This three-level design answers:

- Was the relation present during pretraining?
- Was it merely exposed by a readout?
- Was it installed or reorganized by fine-tuning?

---

## 22. Register the shortcut baselines

Create `src/gfmcd/baselines/splicing.py`.

Minimum baseline family:

1. donor-only local score;
2. acceptor-only local score;
3. additive donor + acceptor score with no interaction;
4. distance-only model;
5. local k-mer logistic regression;
6. bag/DeepSets model over local candidate features;
7. receptive-field-limited CNN;
8. explicit donor–acceptor interaction model.

The shortcut-gap is measured relative to this registered family. Do not claim to rule out every possible shortcut.

Report:

- ordinary test performance;
- cue-conflict performance;
- calibration;
- performance as matching strictness increases.

---

## 23. Build cue conflicts for partner-aware splicing

Create `docs/studies/splicebert_partner_choice/CUE_CONFLICTS.md` before running the full experiment.

Minimum cue conflicts:

1. **True versus matched decoy**  
   Local motif strength is matched; only annotated pairing differs.
2. **Partner swap**  
   Preserve candidate local sequences while changing which donor and acceptor contexts are paired.
3. **One-sided weakening**  
   Weaken the donor while preserving acceptor local sequence, and vice versa.
4. **Spacing perturbation**  
   Preserve local motifs while altering separation within the model’s context.
5. **Canonical-to-cryptic competition**  
   Introduce or reveal a strong local decoy and test whether the model follows the new partner relation.
6. **Matched irrelevant edit**  
   Control for the number and distance of edits.

Every conflict must return:

- clean sequence;
- corrupt/conflict sequence;
- intended relational target;
- shortcut target;
- edit locations;
- matching diagnostics;
- score mask.

---

## 24. Define what counts as a circuit

A component set is not a circuit because it attends between splice sites. Require all of the following:

### 24.1 Competence

The model must exhibit partner-aware behavior beyond the registered local/additive baselines on held-out cue conflicts.

### 24.2 Localization

A compact set of heads, MLP features, or residual directions must carry a substantial fraction of the clean-versus-conflict effect.

### 24.3 Necessity

Ablating the proposed set must reduce the relational score more than:

- matched random component sets;
- same-size high-attention but low-causal sets;
- edits at irrelevant positions.

### 24.4 Sufficiency or mediation

Clean-to-corrupt activation patching through the proposed path must restore the true-partner score. Report recovery as:

```text
recovery = (patched_score - corrupt_score) / (clean_score - corrupt_score)
```

Report a bootstrap confidence interval and compare to matched random paths.

### 24.5 Specificity

The circuit should respond to cognate partner relations more than motif-matched decoys and matched-distance random pairs.

### 24.6 Replication

The high-level computation must recur across:

- multiple dataset samples;
- multiple readout seeds;
- preferably more than one model checkpoint or conversion.

Individual head identities need not match exactly if the causal computation does.

### 24.7 External biological validity

Circuit-derived scores must predict an external splicing outcome better than local motif scores alone.

Do not call a result a biological discovery until the inferred relation predicts an independent assay or observed transcript change.

---

## 25. External validation path

Use a staged plan.

### Stage 1 — immediately accessible

- GENCODE canonical pairs and matched decoys;
- MFASS and Vex-seq variant-effect datasets used in the SpliceBERT study, if licensing and access permit;
- public splice-site usage or cryptic-site datasets with clear provenance.

### Stage 2 — stronger partner-change validation

Reproduce or adapt the rare-variant/outlier-junction analysis reported in nucleotide-dependency work if the event list and required data are accessible. Keep controlled-access GTEx requirements separate from open-data tasks.

### Stage 3 — biological discovery

Use the traced circuit to generate a specific, testable interaction hypothesis not already encoded in the baseline. Validate on an independent dataset and, if feasible, nominate a small set of minigene perturbations for experimental follow-up.

---

## 26. Immediate tasks that can begin now

Execute in this order:

1. Correct active claims and archive the superseded proposal.
2. Add central paths, data manifests, and CI.
3. Add the repeat-copy weight-1 control and objective sweep runner.
4. Add counter-versus-lookup preregistration and data generators.
5. Verify an official SpliceBERT checkpoint and create the adapter.
6. Reproduce basic MLM output and the paper’s model dimensions.
7. Implement substitution-based nucleotide dependency on a tiny synthetic sequence.
8. Build a 200–500 intron pilot dataset with true and matched decoy partners.
9. Run local/additive baselines before tracing anything.
10. Run the zero-shot true-partner versus decoy dependency assay.
11. Only if a behavioral effect exists, perform a head/MLP intervention screen.
12. Freeze the full preregistration before scaling the dataset or selecting a final circuit.

A negative pilot is useful. If SpliceBERT does not exhibit partner-aware behavior after strict matching, do not force a circuit trace. Report that the relation is absent, too weak, or inaccessible to the chosen readout, then move to a supervised/fine-tuned condition with the distinction preserved.

---

# Part V — Paper plan and repository outputs

## 27. Target paper contribution set

The first paper should aim for these contributions:

1. **A ground-truth-calibrated genomic circuit-discovery protocol** that measures known blind spots and includes cue conflict, necessity, and mediation.
2. **A quantitative shortcut-gap study** showing how objective weight and data frequency affect the probability and timing of a relational repeat-copy mechanism.
3. **A causal partner-aware splicing circuit in pretrained SpliceBERT**, or a well-controlled negative result demonstrating that represented splice-site dependencies are not causally used.
4. **A pretraining-versus-readout-versus-fine-tuning comparison** identifying when the relational mechanism is present or installed.
5. **External biological validation** showing whether circuit activity predicts variant-induced splicing effects beyond local motif baselines.

Do not include CTCF, NMD, RNA folding, and Evo 2 as simultaneous primary aims. List them as future benchmark studies after the SpliceBERT template is proven.

---

## 28. Planned figures

1. **Calibration ladder** — planted twins → controlled transformer → pretrained SpliceBERT.
2. **Shortcut-gap sweep** — objective advantage, emergence probability, grokking time, and causal circuit strength.
3. **SpliceBERT behavioral assay** — true partner versus matched decoys under cue conflict.
4. **Causal circuit** — component/path localization, necessity, mediation, and random-set controls.
5. **External validation** — variant or splice-junction outcomes predicted by circuit score versus local baselines.

Every figure must have a tidy CSV/JSON source in the run registry.

---

## 29. Claim-status table to add to the active proposal

| Claim | Correct current status |
|---|---|
| Planted twins discriminate compositional and shortcut mechanisms | Settled within the planted architecture |
| Frozen-attention tracer misses QK selection edges | Settled |
| Integrated-gradient QK method recovers the planted marker edge | Settled within planted models |
| LayerNorm faithfulness is calibrated | Not yet; required |
| Real-DNA next-token GFM-nano uses an anchored frame circuit | Current evidence says no at long range for the tested condition |
| Forced amino-acid model learns an anchor-relative modulo-3 circuit | Supported; sequential counting unresolved |
| Inert marker head demonstrates attention is not causality | Supported for the analyzed checkpoint |
| Repeat-copy circuit emerges without forcing | False as currently written; objective uses 8× copy weighting |
| Repeat-copy circuit exists under weighted objective | Supported on the reported checkpoints; artifacts must be restored/pinned |
| Sufficiency law is an iff law | Unsupported; replace with shortcut-gap hypothesis |
| Reading-frame 2×2 is settled | Provisional/reproduction blocked |
| RC equivariance makes Caduceus orientation blind | Incorrect |
| SpliceBERT partner-aware circuit exists | Open research question |

---

## 30. Final acceptance checklist

The cleanup is complete only when all boxes are satisfied.

### Claims

- [ ] No active file calls the shortcut-gap idea a settled iff law.
- [ ] No active file says repeat-copy was unforced.
- [ ] Trained reading-frame results are called anchor-relative modulo-3 unless counter tests pass.
- [ ] The 2×2 is not presented as settled without reproducible artifacts.
- [ ] Caduceus orientation blindness language is removed.
- [ ] DART-Eval is described as behavioral motivation, not mechanistic proof.
- [ ] GFM-nano is distinguished from a pretrained foundation model.

### Reproducibility

- [ ] Clean setup instructions use `uv`.
- [ ] No active code depends on `~/Research/...`.
- [ ] Reference data have a versioned checksum manifest and fetcher.
- [ ] Cache and checkpoint metadata are recorded.
- [ ] CI runs the synthetic test suite.
- [ ] Result-producing commands write manifests and metrics.
- [ ] LayerNorm calibration is either complete or visibly open.

### Scientific controls

- [ ] Repeat-copy weight-1 control exists.
- [ ] Repeat-copy objective sweep is configured.
- [ ] Counter-versus-lookup tests are preregistered.
- [ ] Real-CDS mechanism classification uses output-level measures.

### SpliceBERT

- [ ] Official or verified model source is pinned.
- [ ] Adapter and hook smoke tests pass.
- [ ] Pilot splice-partner dataset and data card exist.
- [ ] Local/additive baselines run before circuit tracing.
- [ ] Cue conflicts are frozen before full evaluation.
- [ ] Necessity, mediation, specificity, and replication criteria are defined.

### Documentation

- [ ] `docs/studies/TEMPLATE/` exists.
- [ ] `docs/studies/splicebert_partner_choice/` contains a preregistration.
- [ ] The superseded proposal is archived.
- [ ] The new proposal matches the statuses in `docs/RESULTS.md`.

---

## 31. Recommended commit sequence

1. `docs: correct scientific claims and archive superseded proposal`
2. `refactor: centralize project and data paths`
3. `data: add reference and cache provenance manifests`
4. `ci: add pytest markers and synthetic pull-request workflow`
5. `results: add run manifests and machine-readable metrics`
6. `test: calibrate tracer under LayerNorm`
7. `experiment: add repeat-copy objective sweep and weight-1 control`
8. `experiment: preregister counter-versus-lookup tests`
9. `docs: add reusable genomic circuit study template`
10. `model: add verified SpliceBERT adapter and hook tests`
11. `data: add splice-partner pilot builder and baseline suite`
12. `experiment: add SpliceBERT dependency and cue-conflict pilot`

At the end of each commit:

```bash
uv run pytest -q -m "not genome and not checkpoint and not slow"
git diff --check
```

Before merging:

```bash
uv run pytest -q
uv run python scripts/audit_claims.py
uv run python scripts/list_results.py
```

The purpose of this cleanup is not to make the project sound smaller. It is to make the central advance defensible: a calibrated route from synthetic ground-truth mechanisms to a real, causally validated computation in a pretrained genomic model.
