# Mech Interp Gap Analysis

> Tracks method coverage and open opportunities. Not a claims ledger.
>
> Method inventory current as of 2026-08-04; the paper-location note reflects the current in-tree draft.
> Re-read CLAIMS.md and ROADMAP.md before acting on any item here.
>
> **Provenance.** Authored on branch `jeyashree-ajay` (Jeyashree Krishnan, Ajay — Algoverse CSAP cohort
> 2026) and ported here on 2026-08-04. Five module paths were corrected against the tree during the port
> and the Priority-A2 recommendation was rewritten because the module it proposes creating already
> exists; see `studies/splicebert_partner_selection/CORRECTIONS.md` R46. This file is **advisory**: it
> ranks method coverage and proposes work. It licenses nothing and is not evidence for any claim.

This document maps the standard mechanistic interpretability toolkit against what is implemented in this
project, identifies gaps, and prioritizes which gaps are worth closing given the project's scientific
open problems.

---

## 1. What is already implemented

| Category | Method | Implementation |
|---|---|---|
| **Localization** | Head ablation (zero, mean, resample) | `src/gfmcd/circuits/hf_bert.py` — `head_ablation_scan`, `patch_heads` |
| **Localization** | MLP ablation | `src/gfmcd/splicing/mlp_units.py` |
| **Localization** | Positional Fourier spectrum | `src/gfmcd/splicing/positional_spectrum.py` — period-3 and ≈10.63 carrier detection |
| **Localization** | SVD / low-rank subspace extraction | `src/gfmcd/splicing/subspace.py` — `U_info` rank-8 readout direction |
| **Localization** | QK logit decomposition | `src/gfmcd/splicing/address.py` — analytic QK logit-to-window, relative vs. absolute geometry |
| **Causal intervention** | Activation patching (reconstruct-inject) | `src/gfmcd/splicing/circuit.py` — exact post-LN transplant; all `run_a*_*.py` runners |
| **Causal intervention** | Path patching (residual stream) | `src/gfmcd/circuits/hf_bert_residual.py` |
| **Causal intervention** | Interchange intervention (unit-level) | `src/gfmcd/splicing/mlp_units.py`, `address.py` synthetic swaps |
| **Causal intervention** | Nucleotide-invariant phase surgery | `src/gfmcd/splicing/p3_phase_surgery.py` — C36/Ψ_C confirmation |
| **Causal intervention** | Carrier dose-response injection | `scripts/splicebert/run_carrier_dose.py` |
| **Mediation** | OV decomposition | `src/gfmcd/splicing/payload.py` — value-projection linearity and delivered-state injection |
| **Mediation** | Rescue / restoration test | `run_l2_rescue.py`, `run_ablation_scan.py` — C9 L2H15/L3H1 restoration |
| **Mediation** | Cross-example subspace transplant | `src/gfmcd/splicing/subspace.py` `patch_component` — C38/C41 |
| **Faithfulness** | Node-removal faithfulness curve | `run_ablation_scan.py`, `run_a12_faithfulness_size.py` — C33 no-knee finding |
| **Faithfulness** | Matched null controls throughout | Every major assay; permuted-label FPR gates on all preregistrations |
| **Controls** | Biological counterfactuals | `src/gfmcd/splicing/coupling.py` — AG→CC, GT→CC corruption; phase-matched decoys |
| **Controls** | MLM health probes | `circuit.py` `CouplingTask`; health-shard monitoring in Atlas |
| **Controls** | Planted-positive / sensitivity gates | All preregistrations; `test_circuit_memorials.py` |

**Verdict:** causal interventions, mediation tests, spectrum analysis, and held-out power gating are all
well-developed. The project already exceeds most published GFM interpretability work on intervention rigor.

---

## 2. What is absent and why it matters

### Priority A — closes an active open problem

#### A1. Automated edge / circuit discovery (ACDC, EAP)
**What it is:** ACDC (Conmy et al. 2023) and edge attribution patching (EAP) enumerate edges in the
computational graph automatically, returning a minimal circuit as a sparse subgraph, not just a node ranking.

**What it would add here:**
- Independent validation of the hand-built A0→A12 donor circuit against an unsupervised search. The
  current structured core is identified by expert-driven ablation; ACDC would confirm or refute the
  claim that no shorter path explains 90% faithfulness.
- Might locate structural motifs the manual walk missed — particularly whether any MLP→head or
  head→MLP edges were overlooked in the address-bind-deliver path.
- Directly addresses the "adaptive minimality" open problem (ROADMAP §open #2).

**Feasibility:** SpliceBERT is 6L×16H = 103 attention heads plus MLP layers. ACDC on a 1024-token sequence
will be expensive (O(nodes²) patching passes) but tractable on a GPU with batch accumulation. Requires a
behavioral metric that can be evaluated per-patch; `CouplingTask.margin` is the right metric.

**Recommendation:** implement a lightweight EAP run on a held-out cohort of 50–100 pairs. Use the existing
`patch_heads` hook infrastructure. Compare the recovered edge set against the manual circuit diagram.

---

#### A2. Gradient-based attribution (integrated gradients, attention-gradient, Jacobian)
**What it is:** input-gradient methods compute the sensitivity of the output to each input token or
internal activation. Integrated gradients (Sundararajan et al. 2017) is the standard axiomatic choice.

**What it would add here:**
- Fast, cheap candidate generation for the acceptor payload (open problem C30 / ROADMAP Track 2). The
  current approach to payload semantics is subspace-based; gradient attribution would add a second,
  complementary localization lens without requiring additional patching runs.
- Would flag whether the donor-side readout is sensitive to nucleotide content vs. position at each token
  — a useful complementary signal to the existing QK analysis.
- METHODOLOGY.md already states: "Attribution ranks; intervention decides." IG fits exactly into that
  slot — use it to generate candidates, then test by patching.

**Feasibility:** trivial to add; `torch.autograd` + a baseline (mask token or zero embedding). No new
infrastructure beyond a gradient accumulation loop. Can run on CPU for small cohorts.

**Recommendation (corrected on port).** `src/gfmcd/circuits/attribution.py` **already exists** and
provides `head_attribution_scan` / `AttributionEffect`. Do **not** create it — and do not overwrite it:
it was restored in `ab670640` precisely because a frozen preregistration cites it *as evidence*, so it
cannot be replaced to suit a new API. The real gap is narrower than this section claims: add
*integrated-gradients* and *attention-gradient* variants **alongside** the existing scan, and use them in
the Track 2 payload study as the candidate-generation step.

---

#### A3. LogitLens / tuned-lens (residual stream reading at each layer)
**What it is:** project the residual stream at each layer through the unembedding head to see the model's
"running prediction" as information builds up layer by layer.

**What it would add here:**
- Directly traces when and where donor-acceptor coupling information enters the residual stream. The
  current account (MLP0 → MLP1 → MLP2 → transporter heads → donor readout) is established by patching;
  LogitLens would corroborate the layer ordering and detect early-layer signals that the manual walk may
  not have probed.
- Useful for the ≈10.63 carrier OV open problem (ROADMAP §open #5): if the carrier write is
  behaviourally redundant, it should still appear as a residual-stream component at the donor position
  before the transporter heads — LogitLens can test this without needing an L0 reader.

**Feasibility:** SpliceBERT is an MLM so the unembedding head is the masked-language-model head; the
token-space projection is well-defined. No training required. Takes one forward pass per example with
hooks on every layer norm output.

**Recommendation:** implement as a one-shot diagnostic script. Use on 10–20 examples to profile
layer-by-layer coupling information and cite the result as a supplementary diagnostic.

---

### Priority B — methodologically valuable, not yet blocking

#### B1. Sparse autoencoders (SAEs) / dictionary learning
**What it is:** train a sparse autoencoder on residual-stream or MLP activations to decompose the
hidden state into interpretable features (Anthropic "Towards Monosemanticity" approach).

**What it would add here:**
- Might resolve the acceptor payload semantic identity (C30): if the acceptor-local state projects onto
  a learnable feature dictionary, a feature could be associated with biological variables (MaxEnt score,
  branch-point distance, exon length) more precisely than a linear probe.
- Could identify whether the period-3 carrier is mono-feature or multi-feature at the token level.

**Caveats:** SAEs require a reasonably large corpus of activations to train reliably; the existing cohort
(~500–2000 examples) is on the small side. An SAE trained on too few examples will overfit and its
features will not be interpretable. This is a substantial infrastructure commitment relative to the
current experiment style, and the activation capture would need to be designed carefully to not corrupt
the immutable result discipline.

**Recommendation:** defer until the acceptor payload semantics question is well-powered (requires larger
gene-disjoint cohorts). If the project gets a second GPU run, budget one run for SAE training on MLP0
activations at the acceptor position (the payload bottleneck layer).

---

#### B2. Causal scrubbing (proper)
**What it is:** the Lindsey et al. 2023 framework that formalizes a hypothesis about the circuit as a
computational graph and resamples activations in a hypothesis-consistent way to measure whether the
hypothesis explains the behavior at every internal node.

**What it would add here:**
- Provides a single unified faithfulness score for the entire A0→A12 hypothesis as a graph, not just
  individual head/MLP ablations. The current faithfulness curve (C33) measures marginal contribution of
  node sets; causal scrubbing measures the full hypothesis jointly.
- Would produce a precise "how much of the computation is explained by the proposed circuit" number
  that is rigorously comparable across papers.

**Caveats:** requires a formalized hypothesis graph, which must be specified before running (preregistered).
The existing account is complex enough (address × payload × delivery, three sub-circuits) that writing
the hypothesis graph is a non-trivial scientific task. This is the right tool to use *after* the manual
account is stable, not during exploration.

**Recommendation:** treat as a confirmatory closing step after Track 1 and Track 2 are resolved. Write
the hypothesis graph as a preregistration, implement the resampler on top of the existing patching hooks,
and run it as a final "does the whole story hang together" gate.

---

#### B3. Steering vectors
**What it is:** add a direction to the residual stream at inference time and observe behavioral change,
without zeroing out other components (contrast with patching which transplants specific content).

**What it would add here:**
- Direct test of payload semantics: if adding the U_info direction at the acceptor position (with random
  content otherwise) moves the donor margin, the direction is causally sufficient for the behavioral
  effect. This complements the transplant approach but doesn't require a matched source example.
- Useful for Track 2 (site-choice steering): steer the acceptor's latent toward a high-MaxEnt vs.
  low-MaxEnt phenotype and measure which donor site wins.

**Caveats:** the project already does dose-response injection (carrier_dose.py) which is structurally
similar. The main gap is steering *content* directions (e.g. the payload) rather than positional
directions (the carrier). The transplant approach (C38/C41) is already the mechanistically stronger test
because it uses real biological content; steering adds a null-content baseline.

**Recommendation:** implement as a variant of the existing `patch_component` function — subtract the
current component and add a scaled target direction. Use it in Track 2 as the null-content baseline
against the full transplant.

---

### Priority C — out of scope or not applicable here

| Method | Reason not applicable |
|---|---|
| Influence functions / training-data attribution | Requires access to the training data (72-species RNA) and the training optimizer states, neither of which is available for SpliceBERT. Origin is attributed comparatively (sibling contrast C39/C40) instead. |
| Probing classifiers (learned linear heads) | The project policy is: all readouts are native model outputs, not learned heads. Probes are a localization tool, not intervention evidence; the existing QK/OV decomposition is a better-controlled substitute. This is a deliberate methodological choice documented in METHODOLOGY.md. |
| Mechanistic anomaly detection / representation engineering | No misalignment or deceptiveness hypothesis to test. This is a biological GFM, not an RLHF-trained model. |
| KL-divergence faithful attribution (APA/Nanda 2024) | Could substitute for the current node-removal curve; not wrong, but the current faithfulness metric is already preregistered and any change would need a new preregistration. |

---

## 3. Relation to the current paper draft

The current anonymous NeurIPS 2026 main-track draft now lives at
[`../tex/splicebert_neurips_2026.tex`](../tex/splicebert_neurips_2026.tex). Its purpose is not to claim that
the repository has exhausted the mechanistic-interpretability toolkit. It uses the interventions already in
the scientific record to show an evidence-gated transition from model behavior, through causal organization
and measured boundaries, to a testable biological hypothesis.

This advisory analysis predates that manuscript and does not control its scope. The ACDC/EAP, gradient,
lens, steering, and SAE suggestions above are optional follow-up methods, not missing evidence silently
required to license the draft's current claims. If one becomes part of a scientific claim, it must enter the
normal preflight, intervention, null-comparison, immutable-artifact, and claim-ledger workflow before the
paper can cite it as a result.

---

## 4. Implementation priorities (post-submission)

| Priority | Method | Estimated effort | Unlocks |
|---|---|---|---|
| 1 | Integrated gradients (`src/gfmcd/circuits/attribution.py`) | 1 day | Track 2 payload candidate generation |
| 2 | LogitLens diagnostic script | 1 day | ≈10.63 OV open problem; layer-ordering corroboration |
| 3 | EAP / lightweight ACDC run on 50-pair cohort | 3–5 days | Adaptive minimality; circuit validation |
| 4 | Steering vector variant of `patch_component` | 1 day | Track 2 null-content baseline |
| 5 | Causal scrubbing (full hypothesis graph) | 1 week | Final faithfulness statement for journal version |
| 6 | SAEs on MLP0 acceptor activations | 2 weeks | Payload semantics (requires larger cohort first) |
