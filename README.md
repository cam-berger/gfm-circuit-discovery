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

The paper currently references two figures. Both must be final-quality PDFs placed in
`figures/splicebert/workshop_edits/` for the short version and adapted as needed for the
long version. Detailed descriptions of what each panel must show are given below.

### Figure 1: Circuit structure and faithfulness boundary

Filename referenced in paper: `fig_full_circuit_edit.pdf`

**Panel A: Address-payload-deliver circuit diagram.**
This panel should show the full information flow from the acceptor-local residual stream
to the masked-donor margin. The diagram should make the following structure visible:

Acceptor local window (four token positions) feeds into three early attention heads
(L0H7, L0H12, L0H10). These heads feed into MLP layer 0, which mediates the payload
through MLP layer 1 and MLP layer 2. In parallel, the period-3 positional carrier
(a rank-2 Fourier component of the position-embedding matrix) provides the address
through head L2H15. These two streams (payload and address) converge at L3H1. L3H1
writes the rank-8 U-info representation at the masked-donor position. Downstream
transporter heads (L4H0, L4H2, L5H11) carry U-info to the masked-donor MLM margin.

The diagram should label each node with its layer and head number. It should indicate
with arrows which stream carries the address and which carries the payload. It should
show U-info as a bottleneck node between L3H1 and the transporter heads. It should
not imply that this is a minimal circuit or that these components are the only ones
contributing to the computation.

**Panel B: Complement faithfulness size curve.**
This panel should show the fraction of coupling preserved when only nodes outside the
named circuit are active, plotted against the number of nodes in the complement set.
The x axis should run from 0 to 103 (the total number of measured nodes). The y axis
should run from 0 to 1 (fraction of coupling preserved). The curve should show that
faithfulness rises gradually with no clear knee or elbow, crossing 90 percent only
near 72 nodes. A horizontal dashed reference line at 0.9 is appropriate. A vertical
marker indicating where the named circuit nodes end and the complement begins would
help the reader understand the setup. The main message is that there is no sparse
knee, which supports the irreducibly distributed conclusion.

### Figure 2: Phase surgery confirmation

Filename referenced in paper: `fig5_p3_phase_surgery_confirmation_edit.pdf`

**Panel A: Mean coupling under period-3 phase shifts.**
This panel should show the mean donor-acceptor coupling score on the y axis and the
phase shift value (0 through 9) on the x axis, for two conditions: CDS blocks (coding
sequence) and upstream intron controls. Off-register shifts (1, 2, 4, 5, 7, 8) should
show reduced coupling in CDS but not in the upstream intron condition. In-register
shifts (3, 6, 9) should show coupling preserved close to the shift-0 baseline in both
conditions. Ribbons showing plus or minus one gene-level standard error should be
included as a descriptive measure (not a confidence interval for inference). The
panel should make the modulo-3 pattern immediately visually apparent.

**Panel B: The preregistered estimand Psi-C.**
This panel should show a single point estimate with a 95 percent confidence interval:
Psi-C = +0.141 logit units [0.110, 0.172], across n = 943 genes. A horizontal line at
the frozen smallest effect size of interest (0.05 logit units) should be shown for
reference. The panel should make clear that the estimate comfortably exceeds the
preregistered threshold.

**Panel C: Learned direction versus same-geometry hidden-axis controls.**
This panel should show the coupling effect of the learned period-3 direction compared
to the 19 same-geometry hidden-axis control directions. One approach is a strip plot
or dot plot with the 19 control effect sizes shown as gray points and the learned
direction shown as a distinct colored point above all of them. The panel should
communicate that the learned direction is an outlier relative to the same-geometry
null distribution, supporting the interpretation that phase alignment to learned
hidden features matters, not just Fourier support or perturbation magnitude.

**Panel D: CDS versus non-CDS exon contrast.**
This panel shows the contrast between CDS phase effects and non-CDS exon phase effects.
The estimate is +0.006 logit units [negative 0.080, +0.093], which is inconclusive.
This panel should be presented honestly: the interval crosses zero and coding enrichment
is not established. A forest plot or interval plot centered on zero with the estimate
shown is appropriate. The honest conclusion (coding enrichment not established) should
be stated in the caption and reflected in the visual.

### Figure 3 (long version only): Cross-model carrier comparison

This figure is needed for the InterpScience long paper and can be added as a supplementary
figure in the short version if space permits.

**Panel A: Period-3 carrier power in sibling checkpoints.**
A bar chart or strip chart showing the Fourier power of the period-3 band (relative to
the median band across all frequencies) in four models: SpliceBERT.510nt (72 species,
the multi-species sibling), SpliceBERT-human.510nt (human-only sibling), RNA-FM, and
RNA-BERT. SpliceBERT.510nt should show 435 times median power. SpliceBERT-human.510nt
should show 6.2 times. RNA-FM and RNA-BERT should show values near 1 (absent). The
logarithmic scale is appropriate given the range.

**Panel B: Period-10.63 carrier power in sibling checkpoints.**
The same design as Panel A but for the approximately 10.63-nucleotide carrier. The
contrast should be even larger: 369 times versus 2.4 times for the sibling pair.
The message is that multi-species pretraining builds both carriers, and a human-only
corpus does not.

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
