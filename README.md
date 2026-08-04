# Mechanistic Discovery in Genomic Foundation Models

Branch: `jeyashree-ajay`
Team: Jeyashree Krishnan, Ajay (Algoverse CSAP cohort, 2026)

This branch extends the main `gfm-circuit-discovery` repository with workshop paper preparation,
repository cleanup, and a gap analysis of the interpretability methods used in this project.
The scientific record on the main branch is preserved and extended here, not replaced.

---

## What we found and fixed

### Code issues corrected

- About 34 scripts had absolute file paths hardcoded to one developer's local machine.
  Every script now reads genome paths from environment variables or command line arguments,
  so the code runs on any machine.
- One script had an unclosed file handle inside a SHA-256 checksum call. This was replaced
  with a safe read that closes the file automatically.
- The provenance audit script referenced a path three directories above the repository root,
  which would silently read the wrong file on any other machine. Fixed to a repository-relative
  path.

### Scientific issues corrected (inherited from main branch, documented here for awareness)

- The payload was previously labeled as splice-site strength. This label was withdrawn after
  it failed independently seeded draws and two-way gene-dyadic inference. The payload is now
  described only as acceptor-local and of unresolved semantic identity (correction R22).
- An earlier claim about the 10.63-nucleotide carrier's write to the donor readout was
  preregistered but could not be evaluated as written because the required L0 reader heads
  were not available. This is recorded as a bounded limitation, not a finding.
- Several preregistration conformance gaps were identified: some runners departed from their
  frozen protocol. These are documented in `docs/PREREG_CONFORMANCE.md` and the affected
  results are labeled accordingly in the claim ledger.

### What we have scientifically

We have a causally supported circuit account of how SpliceBERT links a splice donor to its
paired acceptor: a period-3 positional carrier routes addressing information through one
keystone attention head, which combines it with an acceptor-local payload and writes a
compact low-rank representation that downstream heads carry to the masked-donor output.
Every step is supported by activation patching and controlled interventions. The carrier's
origin is attributed to multi-species pretraining by a natural experiment using matched
sibling checkpoints.

### What we have scientifically and what we still need

The acceptor payload's semantic identity is unresolved. We know it is local to the
acceptor window and is not explained by bulk sequence composition, but we cannot yet name
what biological information it encodes. The cross-context generalization of the mechanism
beyond what is shown has not been demonstrated at the required specificity level. Automated
circuit discovery (edge attribution patching) has not been run, so we cannot yet claim the
identified circuit is the minimal sufficient subgraph. These are the three open problems
the paper is honest about.

Parallel work on a separate branch (`sri-algoverse-acceptor-interpretability`) studied the
reciprocal question: how does SpliceBERT recognize the 3-prime acceptor from its branch point
and polypyrimidine tract? That work ran five phases of behavioral and causal analysis. The
key finding is that the polypyrimidine tract is the dominant, close-to-sufficient driver of
acceptor confidence, and the signal is order-driven and identity-driven, not explained by
bulk pyrimidine composition (a shuffle that preserves the same bases already destroys almost
the full signal). Five attention heads (L5H8, L4H12, L3H5, L0H13, L0H14) show genuine
content-sensitive attention into the tract. However, causal intervention found that none of
these five heads individually or jointly explain more than about 2 percent of the real
acceptor coupling effect. The computation is more distributed than a small head list can
capture, and single-head or small-group causal necessity is a clean null in that study.

This means the acceptor side of the circuit is open at the causal level. The next step
there is to path-patch backward from the acceptor margin itself (the way the IOI methodology
does) rather than filtering through attention patterns, and to include MLP outputs and
non-AG-query token positions as candidate mediators.

Whether the acceptor circuit shares its transport heads with the donor spine (L4H0, L4H2,
L5H11) is an unanswered question. If the two circuits converge on the same delivery mechanism,
that would considerably simplify the full account of how SpliceBERT handles splice-site
recognition. If they do not, it suggests two independent communication pathways.

Real branch-point annotations using branchpointer or LaBranchoR, instead of the PWM-based
stand-in used in the acceptor study, may still reveal a legible branch-point signal that the
heuristic locator missed. The branch-point-alone effect did not survive four independent
attempts to confirm it, but those used an approximate locator and a small sample.

Separately, a DNABERT-2 promoter circuit study (on the `codex/frozen-dnabert-initial` branch)
completed nine phases of analysis including MLP subspace decomposition and pathway mediation
validation for a promoter recognition task. That work is on a different model and a different
biological task, but it demonstrated that a similar reconstruct-inject and low-rank-subspace
methodology applies to DNABERT-2. Integrating those findings as a cross-model transfer
example would require re-running the core assays with the same controlled vocabulary and
evidence standards used here, and is not a short-term task.

---

## What this branch does

### 1. Workshop paper preparation

The repository now contains a LaTeX paper in `tex/splicebert_neurips_2026.tex`.
This paper describes the mechanistic circuit we discovered inside SpliceBERT and is
being adapted for two workshop submissions (see below).

### 2. Repository cleanup

All hardcoded file paths belonging to a specific developer were removed from approximately
34 scripts across the `scripts/` tree and three test files. Every script now reads genome
paths from environment variables (`GFMCD_GTF`, `GFMCD_FASTA`) or from command line arguments
with no default, so the code runs on any machine without modification.

A resource leak in `scripts/crossmodel/run_dnabert2_aliasing.py` was also fixed.
An unclosed file handle inside a SHA-256 checksum call was replaced with a safe
`Path.read_bytes()` call.

### 3. Mechanistic interpretability gap analysis

`docs/MECH_INTERP_GAP_ANALYSIS.md` maps the standard mechanistic interpretability toolkit
against what is implemented here, identifies what is missing, and ranks the missing methods
by scientific priority. The three highest-priority gaps are integrated gradients attribution,
a LogitLens diagnostic script, and a lightweight edge attribution patching run.

### 4. Repository structure alignment

A `slides/` directory was added for presentation materials. The `.gitignore` was updated
to exclude runtime logs and scratch outputs. The repository map in the original README
was updated to reflect the current working tree after the 2026-07-31 cleanup.

---

## Workshop submission targets

This project is submitting the same paper to two workshops at NeurIPS 2026, Sydney.
The paper is adapted to each venue's format and length limit.

### Workshop 1: Interpretability for Discovery

Venue: NeurIPS 2026 Workshop on Mechanistic Interpretability for Scientific Discovery
Deadline: August 29, 2026
Page limit: 5 pages, plus references
Format: short paper

### Workshop 2: InterpScience

Venue: NeurIPS 2026 Workshop on Interpretability in Science (interpscience.github.io)
Deadline: August 28, 2026
Page limit: 9 pages for long papers, plus references
Format: long paper

Note on dual submission: the dual submission policy of each workshop must be verified before
the final submission. If the policy prohibits submitting the same paper to two workshops at
the same conference, the team must choose one. Our current recommendation is InterpScience
because the longer page limit allows the control discipline and three-part circuit structure
to be described properly. The Interpretability for Discovery deadline is one day later, so
switching targets at the last moment is possible if needed.

The longer InterpScience version should expand the methods section to describe the
reconstruct-inject design in more detail, include the cross-model carrier comparison as a
full subsection rather than a paragraph, and add a figure for the QK and OV decomposition
showing how address and payload were separated.

---

## Claims in the paper

The following claims are what the paper asserts. Each one is supported by a specific
experiment with controls. The claim ledger in
`studies/splicebert_partner_selection/CLAIMS.md` is the authoritative record.
This list is a plain-language summary for orientation only.

**Claim 1. Donor-acceptor coupling is real and specific.**
A masked splice donor's MLM margin depends on its own paired acceptor when the acceptor is
corrupted. The effect is partner-specific (the partner's corruption causes the change, not
an arbitrary site's) and donor-specific (only the masked donor's margin changes, not a
nearby donor's). The coupling margin is +0.872 logit units with a 95 percent interval of
[0.771, 0.969]. Magnitude controls at non-partner and non-donor positions are +0.007 and
+0.005 respectively.

**Claim 2. L3H1 is the keystone head.**
Ablating L3H1 and L2H15 together removes 81.6 percent of coupling while retaining 99.2
percent of general masked language model health. L3H1's contribution is conjunctive: the
key alone and the value alone have small effects, but key and value together give +0.663
logit units with an interaction term of +0.531 [0.477, 0.583].

**Claim 3. The circuit has an address-payload-deliver structure.**
The address is derived from position-embedding geometry, not from acceptor sequence content.
The payload is assembled from the acceptor-local residual stream by a serial generator
(early attention heads, then MLP layers 0 through 2). L3H1 combines these two streams and
writes a rank-8 low-rank representation at the donor position. Downstream transporter heads
carry this representation to the masked-donor readout.

**Claim 4. The period-3 carrier is causally required.**
Deleting the rank-2 period-3 Fourier component of the position-embedding matrix collapses
donor-acceptor coupling from 0.946 to 0.027. The same deletion inflicts disproportionate
damage on exon predictions relative to intron predictions (+0.159 versus +0.042 nats).
Restoring a single head after deletion recovers more than 34 percent of coupling, well
above the best single-head random null of 4.2 percent.

**Claim 5. Phase surgery confirms causal use without nucleotide confounds.**
Shifting the period-3 phase by off-register offsets reduces coupling selectively inside
coding sequence blocks while in-register shifts preserve it. The preregistered estimand
is Psi-C = +0.141 logit units [0.110, 0.172] across 943 held-out genes, exceeding the
frozen smallest effect size of interest (0.05 logit units). The learned period-3 direction
exceeds all 19 same-geometry hidden-axis controls with a paired excess of +0.136 [0.051, 0.222].

**Claim 6. The carrier is a multi-species pretraining artifact.**
The carrier is strong in the 72-species stage-1 checkpoint (435 times median band power) and
undetected in the human-only sibling checkpoint (6.2 times; permutation p = 0.071; ratio 70x).
Both carriers are absent from RNA-FM and RNA-BERT, which were trained on ORF-free non-coding
RNA. A model-free spectral scan finds no period-3 periodicity in 60 megabases of human genome.

**Claim 7. The circuit is distributed, not sparse.**
A complement faithfulness size curve rises gradually and crosses 90 percent faithfulness only
near 72 of 103 measured nodes, with no sparse knee. The named components are causally
well-supported and top-ranked, but the broader distributed computation is load-bearing.
Minimality is not claimed.

**What is not claimed (important).**
The payload's semantic identity is unresolved. The earlier label of "splice-site strength"
was withdrawn after it failed independently seeded draws and two-way gene-dyadic inference
(correction R22). The 10.63-nucleotide carrier's downstream write and behavioral function
are open. Transfer of the mechanism across genes and contexts beyond what is shown has not
been demonstrated at the required specificity level.

**Claims to develop in future work (from other branches).**
The following results exist in separate branches and are not yet part of the paper claims.
They are recorded here so they are not lost when planning the next phase of the project.

From `sri-algoverse-acceptor-interpretability`:
- The polypyrimidine tract is the dominant, close-to-sufficient driver of SpliceBERT's
  acceptor confidence signal (masked-AG margin). Effect size approximately +1.245 logit
  units [0.875, 1.624] for PPT disruption, acceptor-specific against a far-intronic floor
  of +0.046 [0.011, 0.085]. PPT disruption alone reaches the same absolute margin floor as
  having no real context at all (random-AG floor), whereas branch-point disruption alone
  does not.
- The PPT signal is order-driven and identity-driven: a shuffle that preserves the same
  bases but reorders them already destroys approximately 97 percent of the effect, so the
  model is reading a specific positional pattern, not aggregate composition.
- Five attention heads (L5H8, L4H12, L3H5, L0H13, L0H14) show genuine content-sensitive
  attention into the tract, with L5H8 and L4H12 being order-sensitive. None of the original
  six donor-circuit candidate heads pass this content-sensitivity test at either region.
- These five heads do not individually or jointly explain more than approximately 2 percent
  of the acceptor coupling effect, as measured by mean-ablation and restoration-patching.
  The acceptor-tract computation is more distributed or more indirectly routed than a
  five-head account can capture.
- The branch-point-alone effect did not survive four independent attempts to establish it
  (matched floor, absolute floor, per-sequence relocation, and attention content-control).
  Treat it as unresolved and likely weak in this model pending real branch-point annotations.

These results need to be incorporated into this branch with proper preregistration, gene-
disjoint cohorts, and the controlled intervention methodology used in the donor-circuit arc
before they can be claimed in the paper.

---

## Figures needed

The paper needs at least seven figures. All must be final-quality PDFs placed in
`figures/splicebert/workshop_edits/`. The short five-page version should include Figures 1
through 5 and select the most important panels from Figures 6 and 7 as space permits. The
long nine-page InterpScience version should include all seven. Detailed descriptions of what
each panel must show are given below.

### Figure 1: Paper idea overview

Suggested filename: `fig_overview_idea.pdf`

This is the first figure a reader encounters and must explain the scientific question and
experimental strategy at a glance, without requiring the reader to know how transformers
work internally. It should be a schematic illustration, not a data figure.

**Panel A: The biological problem.**
Show a short intron in a pre-mRNA transcript. Label the splice donor site (GU dinucleotide
at the start of the intron) and the splice acceptor site (AG dinucleotide at the end of the
intron). Show that these two sites must be recognized as a matched pair for correct splicing.
The distance between them (hundreds to thousands of nucleotides) should be visually
represented, making clear that the model must bridge a large span to link them.

**Panel B: The modeling setup.**
Show the same transcript fed into SpliceBERT as a sequence of tokens. Indicate the masked
donor token. Show the output as a probability distribution over nucleotide identities at the
masked position, labeled as the masked-donor margin. The illustration should convey that the
model predicts what nucleotide belongs at the donor site, and that we examine how its internal
computations achieve this.

**Panel C: The experimental question and approach.**
Show two sequences side by side: one with the acceptor intact and one with the acceptor
corrupted (AG replaced by CC). Show that corrupting the acceptor changes the masked-donor
margin. Then indicate that the paper uses mechanistic interpretability (activation patching,
subspace analysis, Fourier analysis) to trace which internal components are responsible for
this effect. This panel should convey the core logic: we perturb the acceptor, measure the
effect at the donor, and trace the pathway carrying the information.

### Figure 2: Donor-acceptor coupling is behaviorally real and specific

Suggested filename: `fig_coupling_assay.pdf`

This figure establishes the phenomenon being explained. It is a data figure based on the
coupling assay results and should stand on its own as the answer to the question "does the
model actually encode a relationship between donor and acceptor?"

**Panel A: Coupling effect sizes by control type.**
A dot plot or bar chart with 95 percent confidence intervals showing the change in
masked-donor margin under four conditions: (1) partner acceptor corrupted; (2) non-partner
acceptor corrupted at the same intron distance from a different gene; (3) non-donor masked
position, meaning the same corruption but readout at a nearby donor that is not the
paired donor; and (4) permuted-label null. The partner condition shows +0.872 logit units
[0.771, 0.969]. The non-partner and non-donor controls show approximately +0.007 and +0.005
respectively. The permuted null should be near zero. This panel establishes partner
specificity and donor specificity, ruling out bulk sequence effects.

**Panel B: Corruption operator validation.**
A table or bar chart showing that the coupling effect is stable across the six
motif-hygienic dinucleotide substitutions tested as corruption operators. Each operator
should show an effect within 25 percent of the main estimate. This establishes that the
result does not depend on the specific bases chosen for corruption, and that the assay
design is robust before any circuit work begins.

### Figure 3: The address-payload-deliver circuit

Suggested filename: `fig_circuit_structure.pdf`

This is the main mechanistic figure. It shows the full information flow from inputs to the
masked-donor output through the identified circuit components.

**Panel A: Circuit diagram.**
A schematic diagram showing the address stream and the payload stream converging at L3H1.

The payload stream runs from the acceptor-local window (four token positions around the
acceptor site) through three early attention heads (L0H7, L0H12, L0H10), then through MLP
layer 0, MLP layer 1, and MLP layer 2. Each step should be shown as a labeled node with an
arrow to the next. Use one consistent color for this stream throughout, for example blue.

The address stream runs from the position-embedding matrix through the period-3 Fourier
component to head L2H15, which amplifies the positional attraction through L3H1's address
channel. Use a second consistent color for this stream throughout, for example orange.

The two streams converge at L3H1. L3H1 writes the rank-8 U-info representation at the
masked-donor position, shown as a bottleneck node. Downstream transporter heads (L4H0, L4H2,
L5H11) carry U-info to the masked-donor MLM margin.

The diagram should include a label or annotation making clear that this is a structured
causal core within an irreducibly distributed computation, not a minimal circuit.

**Panel B: Key intervention results supporting the circuit.**
A compact table or set of labeled bars showing the four most important intervention effects
that support the circuit structure: (1) ablating L3H1 and L2H15 together removes 81.6
percent of coupling while retaining 99.2 percent MLM health; (2) the key-only versus
value-only versus key-and-value factorial at L3H1, showing an interaction of +0.531 [0.477,
0.583]; (3) restoring L3H1 alone after full ablation recovers 0.83 of coupling; (4) U-info
rank-8 recovery at 0.92 versus random-subspace controls at or below 0.05. Numbers should
include confidence intervals or control comparisons where available.

### Figure 4: Address and payload are separable streams

Suggested filename: `fig_qk_ov_decomposition.pdf`

This figure shows the evidence that the address and payload are distinct streams that can be
independently manipulated. This supports the address-payload-deliver framing rather than a
simpler single-stream account.

**Panel A: Factorial intervention at L3H1.**
A grouped bar chart showing coupling recovery under four conditions at L3H1: key only
transplanted, value only transplanted, both transplanted, and neither (baseline). Key only
and value only show small effects. Both together show approximately +0.663 logit units with
an interaction of approximately +0.531. This is the main evidence that the two streams are
conjunctive, not additive, at L3H1. Error bars should be shown.

**Panel B: U-info subspace generalization across gene-disjoint sets.**
A scatter plot or bar chart showing the rank-1 overlap of the U-info direction across
independently derived gene-disjoint sets, with a value of approximately 0.970. The held-out
sufficiency of approximately +0.875 logit units should also be shown. Random-subspace
controls at each rank should appear for comparison. This establishes that U-info is a stable
gene-generic readout direction and not an artifact of fitting to a specific gene set.

**Panel C: Payload is localized to the acceptor window.**
A heatmap or bar chart showing coupling recovery when the payload transplant is restricted
to different token positions around the acceptor site. The four-token acceptor window should
show strong recovery. Positions outside this window should show near-zero recovery. This
localizes the payload to a specific position range and distinguishes it from a bulk-sequence
or global effect.

### Figure 5: Nucleotide-invariant phase surgery confirms causal carrier use

Suggested filename: `fig_phase_surgery.pdf`

This is the confirmatory figure for the period-3 carrier. It communicates that the carrier
is causally required for coupling, that this is not driven by nucleotide content, and that
the effect size exceeds the preregistered threshold.

**Panel A: Mean coupling under period-3 phase shifts by genomic context.**
Mean donor-acceptor coupling (y axis) against phase shift value 0 through 9 (x axis), shown
separately for CDS blocks and upstream intron controls. Off-register shifts (1, 2, 4, 5, 7,
8) reduce coupling in CDS but not upstream introns. In-register shifts (3, 6, 9) preserve
coupling in both conditions close to the baseline. Ribbons of plus or minus one gene-level
standard error are appropriate as a descriptive measure. The modulo-3 pattern should be
immediately visible.

**Panel B: The preregistered estimand Psi-C.**
A single point estimate with a 95 percent confidence interval: Psi-C = +0.141 logit units
[0.110, 0.172], across n = 943 held-out genes. A horizontal reference line at the frozen
smallest effect size of interest (0.05 logit units) should be shown. The estimate clearly
exceeds the threshold.

**Panel C: Learned direction versus same-geometry hidden-axis controls.**
A strip plot or dot plot showing the coupling effect of the learned period-3 direction as
a distinct colored point, with the 19 same-geometry hidden-axis control directions shown as
gray points. The learned direction should be an outlier above all 19 controls. The paired
excess is +0.136 [0.051, 0.222]. This shows that alignment to the specific learned hidden
features matters, not just the Fourier support or perturbation magnitude.

**Panel D: Coding sequence versus non-coding exon contrast.**
The contrast between CDS phase effects and non-CDS exon phase effects: +0.006 logit units
[negative 0.080, +0.093]. Show this as a point estimate with its interval crossing zero.
The honest conclusion is that coding enrichment is not established and this should be stated
in the caption. Do not soften this result; the inconclusive estimate is part of the
scientific record.

### Figure 6: The period-3 carrier is built by multi-species pretraining

Suggested filename: `fig_carrier_origin.pdf`

This figure provides the corpus-attribution evidence for the period-3 carrier and shows its
Fourier structure in the position-embedding matrix.

**Panel A: Fourier power spectrum of the position-embedding matrix.**
The power spectrum of the 1026 position row vectors, plotted as power (y axis, log scale)
against period in nucleotides (x axis). The period-3 peak should be clearly visible and
labeled, showing 289 times median band power. The approximately 10.63-nucleotide peak
should also be visible and labeled. All other periods should appear near the median baseline.
This panel introduces the reader to what a carrier looks like before the cross-model
comparison.

**Panel B: Period-3 carrier power across four models.**
A bar chart or strip chart showing the Fourier power of the period-3 band relative to the
median band in four models: SpliceBERT.510nt (72 species), SpliceBERT-human.510nt
(human-only sibling), RNA-FM, and RNA-BERT. The 72-species model shows 435 times median
power. The human-only sibling shows 6.2 times. RNA-FM and RNA-BERT show values near 1.
The logarithmic y scale is appropriate. The sibling pair shares architecture, training stage,
and sequence length; only the training corpus differs. This is the natural experiment
attributing the carrier to multi-species pretraining.

**Panel C: Period-10.63 carrier power across the same four models.**
The same design as Panel B but for the approximately 10.63-nucleotide carrier. The contrast
is larger: 369 times versus 2.4 times for the sibling pair. Multi-species pretraining
independently builds two distinct positional carriers, and a human-only corpus builds
neither.

### Figure 7: The named circuit is a structured core within a distributed computation

Suggested filename: `fig_faithfulness_curve.pdf`

This figure presents the faithfulness analysis. It stands alone from the circuit diagram so
that the distributed structure argument gets its own dedicated visual.

**Panel A: Complement faithfulness size curve.**
The fraction of coupling preserved (y axis, 0 to 1) when only nodes outside the named
circuit are active, plotted against the number of nodes included in the complement set
(x axis, 0 to 103 nodes total). The curve rises gradually with no visible knee or elbow,
crossing 90 percent only near 72 of 103 nodes. A horizontal dashed line at 0.90 should be
drawn for reference. A vertical shaded region indicating the approximate 90 percent crossing
range (between 64 and 72 nodes) is appropriate. The gradual rise is the main message:
there is no sparse region where a small number of nodes accounts for nearly all the
computation.

**Panel B: Named spine necessity, sufficiency, and standalone generation.**
Three point estimates on a common scale shown as labeled bars or a small table: necessity
N = 0.989 (ablating all non-spine nodes leaves nearly zero coupling), sufficiency D = 0.986
(injecting the spine into a corrupted stream recovers nearly full coupling), and standalone
faithfulness F = 0.005 (running the spine alone without broader context generates almost no
coupling). The contrast between D and F is the key result: the named components are causally
sufficient when injected into a functioning context but they are not a self-contained
generator. This panel should prevent the reader from concluding that the circuit is a
minimal sparse generator.

### Figure 8: The acceptor circuit is a future extension (preliminary results from separate branch)

Suggested filename: `fig_acceptor_circuit_prelim.pdf`

This figure can be included in the long InterpScience version as a forward-looking section
showing where the account is headed and what a reciprocal acceptor-circuit study would
need to establish. It is based on preliminary results from the `sri-algoverse-acceptor-
interpretability` branch and must be clearly labeled as preliminary and not yet part of the
main paper claims. Do not include it in the five-page version.

**Panel A: PPT disruption versus branch-point disruption of acceptor confidence.**
A forest plot showing the change in masked-AG margin under four conditions: branch-point
region disrupted alone, polypyrimidine tract disrupted alone, both disrupted together, and
context required (real acceptor window minus random-AG baseline). Show the effect size and
95 percent bootstrap confidence interval for each. The key result: PPT-alone reaches the
same absolute floor as having no real context, while BP-alone stays above the floor. Acceptor-
specific controls (matched far-intronic region) should appear at approximately +0.046 for
reference. All numbers from the sri branch study, n = 200 acceptors, 5000-resample bootstrap.

**Panel B: Content-sensitive attention heads versus causal null.**
A two-panel comparison: on the left, a dot plot showing the five attention heads
(L5H8, L4H12, L3H5, L0H13, L0H14) ranked by content-sensitive attention into the PPT
region (the score that passed the content-control filter in Phase 4 of the sri study). On
the right, the same five heads' joint mean-ablation effect on acceptor confidence as a bar,
shown against the full PPT-disruption effect for scale. The small bar on the right illustrates
the main finding: content-sensitive attention does not imply individual or small-group causal
necessity in this model. Label the right panel as approximately 2 percent of the disruption
effect.

**Panel C: Shared-transport question (schematic, not yet data).**
A schematic showing the open question: do the acceptor-reading heads write to the same
transport heads (L4H0, L4H2, L5H11) that carry the donor payload to the donor readout? Use
a split diagram, left side showing the established donor spine ending at the transport heads,
right side showing the acceptor-reading circuit with an open arrow to the same transport
heads labeled "shared?" and a separate arrow to an unlabeled output labeled "independent?"
This communicates the shared-transport hypothesis as an open problem without claiming a result.

---

## Results to generate as soon as possible

The following experiments are not yet run. They are needed to strengthen the paper
before submission or to support the open questions described in the Discussion section.

### Highest priority before August 28

**Integrated gradients attribution.**
This adds a second localization method that complements the existing patching approach.
The practical use is to generate candidate directions for the payload semantic identity
question (Track 2 in the roadmap). The implementation should be a single utility
function in `src/gfmcd/circuits/attribution.py` with integrated gradients and an
attention-gradient variant. The baseline should be the mask token or the zero embedding.
This runs on CPU for small cohorts and requires no new infrastructure.

**LogitLens diagnostic.**
Project the residual stream at each layer through the unembedding head to show the model's
running prediction layer by layer. This traces when donor-acceptor coupling information
enters the residual stream and can corroborate the manual layer-ordering account (MLP0
through MLP2 to transporter heads). SpliceBERT's unembedding head is the masked language
model head, so no training is required. One forward pass per example with hooks on every
layer-norm output is sufficient. Run on 10 to 20 examples and add to the supplementary
material of the long paper.

### Priority after submission (post-August 28)

**Edge attribution patching run.**
Run a lightweight edge attribution patching pass on 50 to 100 held-out example pairs
to independently validate the hand-assembled A0 to A12 circuit as the minimal sufficient
subgraph. Compare the recovered edge set against the manual circuit diagram. The
`CouplingTask.margin` is the right behavioral metric. Use the existing `patch_heads`
hook infrastructure in `src/gfmcd/circuits/hf_bert.py`.

**Steering vector baseline for payload semantics.**
Implement a variant of the existing `patch_component` function that subtracts the current
component and adds a scaled target direction. Use this in Track 2 as a null-content
baseline against the full transplant. This tests whether the U-info direction is
causally sufficient without real biological content from a source example.

**Larger cohort for payload semantics (Track 2).**
The acceptor payload semantic identity question requires more power than the current
cohort provides. A larger gene-disjoint cohort with paired site-choice contrasts is
needed before the MaxEnt or splice-signal candidate directions can be tested properly.

### Acceptor circuit work (separate branch, not yet integrated)

The following experiments are needed to convert the preliminary acceptor-circuit findings
from the `sri-algoverse-acceptor-interpretability` branch into proper paper claims. None
of these is ready for the August 28 deadline; they are recorded here to plan the post-
submission phase.

**Path-patching backward from the acceptor margin.**
Run path-patching backward from the masked-AG margin (the acceptor confidence metric) rather
than forward from the PPT-attentive heads, the way the IOI methodology nominates components.
This lets the model tell us which components matter rather than filtering by attention pattern
first. Include MLP outputs at each layer and token positions other than the AG query, since
a component could write tract information somewhere other than the AG row and a later layer
could read it there.

**Shared-transport test.**
Once a mediating component in the acceptor circuit is found, test whether it routes
information through the same transport heads (L4H0, L4H2, L5H11) that carry the donor
payload. A full-spine ablation removing those three heads should degrade both donor coupling
and acceptor confidence if the circuits share a delivery mechanism. An interchange
intervention transplanting the acceptor-local state into the donor-spine framework would
test whether the U-info readout direction is relevant to both.

**Real branch-point annotations.**
Replace the PWM-based branch-point locator used in Phases 1 to 3 of the sri study with
branchpointer or LaBranchoR coordinates matched to the GUE acceptor sequences. Re-run
the BP-only disruption analysis on a larger sample (n above 400) using real annotations
before concluding the branch-point effect is absent. CIs in the existing study are wide
enough that a larger sample with a correct locator could still show a small real signal.

---

## Code quality notes

These rules apply to all scripts and source files in this repository. They are not
suggestions; they reflect past incidents that caused wasted work or invalidated results.

### Paths and portability

Never hardcode absolute paths to genome files, annotation files, or model checkpoints
in any script or source file. Every path must be either:
- A command-line argument with no default value (the script fails clearly if not provided), or
- Read from an environment variable with a clearly nonexistent fallback path so the
  script fails immediately rather than silently using the wrong file.

The canonical environment variable names are `GFMCD_GTF` for the annotation file,
`GFMCD_FASTA` for the genome sequence, and `GFMCD_FASTA_FAI` for the genome index.
For other inputs, invent a clearly named `GFMCD_*` variable and document it in `REPRODUCE.md`.

### Result integrity

Every script that writes a result artifact must also write a manifest file beside the
artifact. The manifest records the git commit, the input file hashes, and the key
configuration values. Never silently overwrite an existing result. Use `guard_output_path`
from `gfmcd.artifact_guard` to enforce this. If an experiment must be rerun, pass the
`--overwrite-record` flag explicitly.

### Preregistration discipline

No confirmatory experiment may be run without a frozen preregistration that specifies
the behavioral metric, the falsifiable memorials, the null controls, and the smallest
effect size of interest. The smoke gate (memorials only, sealed data excluded) must
pass before the confirmatory run. This rule exists because the memorials have caught
real bugs before any result was read. It is not optional for new experiments.

### Evidence ladder

Do not promote evidence across levels. A feature visible in the position table is not
proven to be used by attention. Attention patterns are not proven to be causal. A
causal model mechanism is not proven to be a biological mechanism. Label each result
with the highest level of the evidence ladder it reaches, not the most interesting
interpretation it is consistent with.

The evidence ladder levels are: Position, QK, Attention, OV or mediation, Behaviour,
Biology. Attribution ranks; intervention decides.

### Claim ledger discipline

New results go into the CLAIMS.md ledger with a numbered entry. Corrections to existing
claims go into CORRECTIONS.md with an R-number. Never edit a past claim silently.
Never quote a number from memory; always quote the claim ledger entry that controls it.
Several numbers that appear natural have been withdrawn or bounded and must not reappear
in prose without their correction context. Read `docs/NOMENCLATURE.md` before writing
any result sentence.

### Running the code

The canonical package manager is `uv`. Run all experiments from the repository root:

```bash
uv run python scripts/splicebert/run_<experiment>.py --gtf $GFMCD_GTF --fasta $GFMCD_FASTA
```

The test suite requires the splicebert, biology, and dev extras:

```bash
uv run --extra splicebert --extra biology --extra dev pytest
```

Many tests skip without a local genome because they are data-gated memorials. This is
by design. A test that skips without the genome is not a test failure.

The `scripts/crossmodel/` directory requires a separate environment by design. Do not
install its dependencies into the main environment.

---

## Scientific record

The authoritative scientific record is on the `master` branch. This branch adds the
following documents to that record:

- `tex/splicebert_neurips_2026.tex`: the workshop paper draft
- `docs/MECH_INTERP_GAP_ANALYSIS.md`: mechanistic interpretability gap analysis
- This README

The claim ledgers, corrections, preregistrations, and result artifacts on master are
unchanged. If you find a discrepancy between what this README says and what
`studies/splicebert_partner_selection/CLAIMS.md` says, the claims ledger is correct.

For the current mechanism statement, limitations, and checkpoint caveats, read
`MECHANISM.md` on the master branch. Do not quote mechanism descriptions from this
README or from the paper draft; they may be slightly out of date or simplified.

---

## Current mechanism summary

A structured causal core binds a predominantly relative positional address to an
acceptor-local payload and delivers the resulting information to the donor readout
within an irreducibly distributed computation.

The key limitation is equally important: the payload is acceptor-local and is not
explained by preregistered bulk-composition variables, but its semantic identity
remains unresolved. The earlier label of splice-site strength was withdrawn.

---

## Start here

1. Read `docs/METHODOLOGY.md` and `docs/EVIDENCE_LADDER.md` for the method rules.
2. Read `MECHANISM.md` for the current mechanism statement and its caveats.
3. Use `studies/splicebert_partner_selection/CLAIMS.md` for the live claim ledger.
4. Read `studies/splicebert_partner_selection/CORRECTIONS.md` before quoting any number.
5. Read `docs/ROADMAP.md` for open problems and the next experiments to run.
6. Read `docs/MECH_INTERP_GAP_ANALYSIS.md` to understand which interpretability methods
   are missing and which to implement next.
7. Follow `REPRODUCE.md` for the environment setup and the claim-to-artifact map.
