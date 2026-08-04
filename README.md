# Mechanistic Discovery in Genomic Foundation Models

Branch: `jeyashree-ajay`
Team: Jeyashree Krishnan, Ajay (Algoverse CSAP cohort, 2026)
Supervised by: Cameron Berger

This branch extends the main `gfm-circuit-discovery` repository with workshop paper preparation,
repository cleanup, and a gap analysis of the interpretability methods used in this project.
The scientific record on the main branch is preserved and extended here, not replaced.

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
