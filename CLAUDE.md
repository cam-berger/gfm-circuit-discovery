# gfm-circuit-discovery — the canonical repo

**This repository is the single source of truth.** It is version-controlled, runnable, and
verified file-by-file (sha256) against the old 18 GB working copy. Everything of record lives
here.

> ## ⛔ Do not use `~/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/`
>
> That tree is a **retired working copy**. It is not a git repo, its docs still assert claims
> that have since been **retracted**, and anything edited there is invisible to the record.
> Everything of record from it has been rescued into this repo (commit `70f53e3`); the ~18 GB
> that remains there is duplicate genome, the retracted v1 dataset, and raw dumps that **no
> claim cites**.

---

## ⛔ THE PREFLIGHT GATE — read `studies/splicebert_partner_selection/METHODS_AUDIT.md`

Three retractions (R16, R18, and the D26 clobber) all have ONE root cause:

> **My error-detection fires on SURPRISE, not on PROCEDURE — so it only fires when I dislike the
> answer. My nulls are therefore more trustworthy than my positives. That is backwards.**

"Validate the instrument" is written in this file three times already and was ignored three times.
So it is now a GATE, not advice. **No estimator touches real data until `preflight()` passes:**

1. **POSITIVE CONTROL** — recover a KNOWN answer, to a tolerance declared in advance.
2. **NEGATIVE CONTROL** — with the signal destroyed, return the null. **Report the FPR.**
   *(This is the arm I skip every single time. Injecting a signal and recovering it measures
   SENSITIVITY, never SPECIFICITY.)*
3. **CONTROL VALIDITY** — anything CALLED a control must be MEASURED to have the property that
   makes it one. A band asserted inert must report its max reader amplitude and kept-fraction.
   *(The "energy-matched INERT band" was read at amplitude 2.2 — LARGER than the carrier it
   controlled for. It out-damaged the carrier, and I explained that away as "the carrier is weak".)*
4. **BOUNDS & RESOLUTION** — declare every output's physical bounds and the instrument's
   resolution/power BEFORE measuring. Out-of-bounds = HARD FAILURE, never a surprising result.
   *(Three post-softmax attempts returned >100% modulation of a PROBABILITY.)*

**RULE 5 — THE ANOMALOUS-CONTROL TRIPWIRE.** If a control behaves strangely, the FIRST hypothesis
is that the instrument is broken. **You may NOT construct a scientific story to accommodate an
anomalous control.** Resolve it at the instrument level, or the run is VOID.

**RULE 6 — THE SYMMETRY RULE.** A result that CONFIRMS what you expected gets the SAME scrutiny as
one that disconfirms it. **For every positive, state what would have made it an artifact — and go
check that thing.**

Also: **never hard-code a vocab offset** (A=6, not 5), and **never feed bare nucleotides** — the
model wants `[CLS]` + 510 + `[SEP]` (0.0144 nats, 100x the effects being measured).

---

## Running it

```bash
.venv/bin/python -m pytest -q          # 192 passed, 1 skipped
.venv/bin/python scripts/splicebert/<script>.py --help
```

- The venv is at `.venv` (built with `uv sync --extra splicebert --extra dev`). **Use
  `.venv/bin/python`, not `python`.** `pip` is not installed in it — use `uv pip install
  --python .venv/bin/python`.
- Device is **MPS**. It is a *single* GPU. **Never run more than one model process at a time**
  — parallel agents each loading SpliceBERT will thrash it and nothing finishes.
- The model ships in-repo: `data/models/splicebert_official_archive/extracted/models/`
  (`SpliceBERT.1024nt`, `SpliceBERT.510nt`, `SpliceBERT-human.510nt`).
- The **genome and GTF live outside the repo** at `~/Research/data/genomes/` (with `fetch.sh` +
  `CHECKSUMS.md5`). This is deliberate — 3 GB of public reference data does not belong in git.
  `DEFAULT_FASTA` / `DEFAULT_GTF` in `src/gfmcd/splicing/circuit.py` point there.

## Layout

| path | what |
|---|---|
| `src/gfmcd/splicing/circuit.py` | **the frozen harness.** Every script imports the circuit, the readout, the corruption, the eligibility filter and the interventions from here, so a sealed-test number cannot differ from a val number for any reason but the data. |
| `scripts/splicebert/` | the live experiments |
| `scripts/splicebert/v1_partner_selection/` | the **retracted** v1 pipeline — quarantined, kept because CLAIMS cites its outputs |
| `studies/splicebert_partner_selection/` | `CLAIMS.md` (**the registry**), `FINDINGS.md`, `DECISIONS.md` (append-only) |
| `studies/codon_phase_lattice/` | the period-3 clock study + its pre-registration |
| `tests/test_*_memorials.py` | one tripwire per retraction. **Do not delete these.** |
| `.archive/` | the adversarial scratch workbench (tarball) and the codex-repeat branch |

## Before you write a number down

This project has retracted **fifteen** claims. Every one died the same way: an experiment that
**could not have failed**. The two rules that came out of it are load-bearing:

> **A control that cannot fail is not a control.**
>
> **An intervention that destroys the behaviour cannot measure the mechanism.**

The specific traps, all of which have already bitten:

- **R7 / R15 — the algebra check.** Swapping a (key, value) *pair* between two positions is an
  identity: `out_i = Σ_j softmax(q_i·k_j)·v_j` is invariant under permuting `j`. And *copying* a
  pair to a new slot is barely better — BERT's positions are **absolute**, so `q_i·k_j` has no
  term in the slot index, and a relocated pair attracts *exactly* the attention it attracted at
  home (measured 9.9e-07). Both "results" passed their pre-registered gates on **untrained
  weights**. Before running an intervention, ask whether it is algebraically forced.
- **R8 / R13 — the health gate.** An "MLM health" check that read the argmax at *unmasked*
  positions scored the model copying its own input: 0.99 for everything, including a model with
  94 of 96 heads ablated. And the lattice-ablation "function test" read its metric off a model
  ablation had driven to **chance** (−1.377 vs ln¼ = −1.386). **Report MLM health next to every
  number, and gate on a behaviour that is NOT the outcome** — gating on the outcome is how
  `run_carrier_surgery.py` printed "VOID" on its own true positive.
- **R12 — the size-matched null.** "Layers 4–5 are specifically required" died to a random
  ablation of the same size (p = 0.42). **One random draw is an alarm, not a null.** Use a
  distribution.
- **R10 — conditioning on the outcome.** `min_effect=0.3` is legitimate for a *ratio* (it needs
  a denominator) and illegitimate for a *magnitude*. Label every conditional number conditional.
- **R14 — the metric that measures the wrong thing.** "Phase purity" (fraction of local
  attention mass on one residue class mod 3) is **maximised by a head attending to exactly one
  position** — one offset has one remainder. It was ranking heads by *sharpness*:
  `corr(purity, effective support) = −0.767`. A whole 42-head "lattice" evaporated.
- **A NUMBER THAT LOOKS LIKE BIOLOGY IS A HYPOTHESIS, NOT AN INTERPRETATION.** The position table
  holds a second carrier at period **10.69** — and 10.5 is B-DNA's helical repeat, 10.8 is an
  α-helix in codons, 11 is A-form RNA. Every one of those is a story you can tell in a sentence,
  and none of them was true. The rule that came out of it:

  > **Subspace identity establishes the carrier. Causal function establishes its meaning.**

  Concretely: the α-helix reading died to a *powered* translation test (the codon clock, run
  through the same assay, shows DiD +0.133 [+0.112, +0.156]; the carrier shows +0.012 [−0.002,
  +0.026]). The chromatin reading died to a helical-phasing test built from the sequence alone.
  And the "two clocks sharpen the address" reading died to **algebra** — for teeth
  `p ∝ (1 + a·cos)`, `H ≈ log N − a²/4`, so one added cosine can concentrate a comb by at most
  ~8% at the measured depth. *Before* running the experiment, ask what the effect size **can** be.
- **R16 — VALIDATE THE INSTRUMENT'S DETECTION FLOOR *BEFORE* TRUSTING ITS NULL.** A tight
  confidence interval around zero is worthless if the measuring device cannot see the effect
  size in question. The WW/SS helical-phasing covariate returned a beautiful null
  (−0.0006 [−0.0085, +0.0072]) — and its estimator turned out to have a **5–10% detection
  floor** against a literature effect of **a few percent**. Inject a signal of known size and
  check you can find it. Four estimator versions each produced a *different* artefact before
  this was caught (scan-edge peaks, trend leakage, filter shaping, a too-weak shuffle null), so:
  **when a spectral estimate keeps peaking at the edge of your scan, the estimator is talking,
  not the data.**
- **KNOW WHETHER YOUR DETECTOR IS LINEAR OR QUADRATIC IN THE EFFECT SIZE.** Pooling *power*
  across unaligned windows gives a statistic that grows as **depth²**; pooling *complex
  coefficients* across aligned landmarks gives one that grows as **depth**. At a few percent
  that is the difference between needing 625× more data and needing 25×. Before scaling up a
  null, ask which detector you built — more data cannot rescue the wrong one. And **pool the
  sufficient statistics BEFORE normalising**: per-window normalise-then-average (`C += v/v[0]`)
  manufactures precise-looking noise when no single window carries a usable estimate.
- **The selection effect.** If you select components by a property and then intervene on that
  property, the result is guaranteed. A stride-3 comb rolled by 3 slides onto itself, so its
  perturbation is small in phase **by set arithmetic** — verified on white noise, with no model
  at all (`tests/test_phase_shift_memorials.py`).

**Statistics:** gene-grouped bootstrap; ratio-of-means, never mean-of-ratios; add-one empirical
p-values with a distinct denominator. Manifests (`ExperimentManifest`) next to every result.

## Where the science stands

- **The coupling circuit is real and sealed** (13/13 on held-out chr8/9): L3H1 reads the
  partner acceptor; L2H15 builds its address; the readout is a distributed chain.
- **The address is built from a learned period-3 positional clock** (D21) — a rank-2 rotary
  phase code in the position table (Fourier bins 339–345, 802× the median bin). Delete it and
  the combs' attention collapses to chance, coding prediction loses 0.159 nats (introns: 0.042;
  energy-matched controls: equal), and the coupling dies 0.946 → 0.027. Restoring **L2H15
  alone** recovers 36%. It is learned in stage-1 pretraining and **only with multi-species
  data** (477.7× vs 6.3× human-only).
- **Open:** whether the multi-species effect is species diversity or just data volume — the two
  released checkpoints cannot separate them.

**Read `studies/splicebert_partner_selection/CLAIMS.md` before quoting any number.** It carries
the live claims *and* the retractions, and the retractions are permanent.
