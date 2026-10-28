# The project so far

This repository asks how a genomic foundation model carries out a piece of biology, and tries to answer
with interventions rather than with pictures of attention. SpliceBERT, a small masked-language model of
RNA, is the worked example. The aim is a method that transfers to other models, so each finding below is
stated at the level of evidence it reached and no further. Exact wording lives in each study's claim
ledger ([study index](../studies/README.md)); this page tells the story in order.

## A behavior worth explaining (July 2026)

When SpliceBERT is asked to fill in the masked `G` of a splice donor, it does better when the donor's own
intron partner, the downstream acceptor, is intact. Replacing that acceptor's `AG` with `CC` lowers the
donor's margin by 0.872 logits (95% CI 0.771 to 0.969) on 400 short introns from the sealed chromosome 8/9
test split, while matched non-partner corruptions do almost nothing (C1, C2). The model behaves as if the donor
knows its mate. The sealed test was run once, on 2026-07-12, and that split has been spent.

## Taking it apart

A series of experiments named A0 through A12 then dissected the behavior. Ablating
two attention heads, L3H1 and L2H15, removes 81.6% of the coupling while leaving the model's ordinary
masked-token accuracy almost untouched (C3). The account that came out of that work has four parts.
L3H1 finds the partner mostly by position, using a relative-offset signal in the position embeddings
with absolute-position and edge modulation (C22, C32). Early heads and MLPs build an acceptor-local value that L3H1 reads (C29, C31). L3H1 writes
the combination into a low-rank state at the donor, and later heads carry it to the output.

Two results limited how far this could be pushed. The acceptor-local value was at first described as
encoding splice-site strength. That association did not hold up across independently seeded samples, and
the label was withdrawn on 2026-07-17 (correction R22). The value is acceptor-local and is not explained
by bulk sequence composition, but what it encodes could not be identified on this checkpoint (C30).
Then a held-out faithfulness curve showed no sparse knee: linear interpolation puts 90% recovery at about
64 of 103 ranked nodes (C33). The mechanism is a structured causal core within distributed computation
under the tested interventions. The hand-built core cannot regenerate its own outputs (C27), and
minimality is not claimed (C33). [MECHANISM.md](MECHANISM.md) states the mechanism and its limits in one
place.

## The positional carriers

The address step depends on position, so the next question was what positional signal the model uses.
A learned period-3 direction in the position embeddings contributes across the donor-acceptor span
(C34). Within carrier patching, its coupling contribution is mediated mainly through L3H1's attention
pattern, not carrier-dependent changes in L3H1's values; upstream effects on payload construction are not
excluded (C35). Surgery that changes only this direction, and
no nucleotides, shows the model still uses it: in coding sequence against upstream intron the effect is
0.141 logits (CI 0.110 to 0.172, n=943), larger than all 19 matched control directions (C36). The matched
contrast against non-coding exons is inconclusive, so this says nothing about coding specificity or
reading frame. A second positional carrier, with a period of roughly 10.63 nucleotides, reaches
post-softmax attention in eight heads; that attention circuit is measured on the stage-1
`SpliceBERT.510nt` checkpoint (C11). On `SpliceBERT.1024nt`, deleting it has a small suppressive effect on
coupling (−0.033, CI −0.061 to −0.005), below the smallest effect of interest and with equivalence
underpowered (C12, C37, R31). Two RNA language models
trained on non-coding RNA show neither periodicity at the assay's detection threshold (C40).

## Testing how general it is

Several studies then asked whether the account holds outside the original setup.

- The [Head–Context Atlas](../studies/splicebert_head_context_atlas/README.md) screened every component
  across sequence contexts. Four general-use components (L2H15, L3H1, L1H12 and the layer-2 MLP)
  validated for both donor and acceptor reconstruction; no context-specific use was established, and the
  Atlas does not show that the donor mechanism transfers (HCA18, HCA19, HCA23).
- On the Atlas's held-out genes, the donor-side state moved the margin in unrelated genes, but as a
  generic readout axis rather than gene-specific information (C41, a verification result, not a
  confirmatory one).
- [Acceptor coupling](../studies/splicebert_acceptor_coupling/README.md) ran the experiment from the other
  end, masking the acceptor and corrupting the donor. A validation gate recorded a large pair-specific
  response (verification tier) and some components have bounded support, but the planned mechanism
  handoff failed and that line is closed.
- [GTEx fine-tuning shift](../studies/splicebert_gtex_finetune_shift/README.md) compared the original model
  with published fine-tuned classifiers.
- [DNABERT-2 promoter](../studies/dnabert_promoter/README.md) applied the method to a second model and a
  different task. It found the same overall shape, a structured causal core within a distributed,
  MLP-dominated computation. It used a single seed and froze no preregistration (DBPR8), and it says
  nothing about the SpliceBERT claims.

## The coding-boundary thread (August to October 2026)

A separate line asked whether coupling cares about coding phase at the exon edge. In one fixed-span
lattice of chimeras that join a donor module from one gene to an acceptor module from another, joins whose
annotated boundary phases match couple more strongly (C43). Whether
the period-3 carrier produces that preference is unresolved, because the instrument built to test it
failed its own checks before the target data were opened (C44). The
[positional boundary coherence](../studies/splicebert_positional_boundary_coherence/README.md) successor
stopped on a cohort-capacity limit before model access. The
[CC/max crossed-register verification](../studies/splicebert_register_operator_replication/README.md) first
stopped at null calibration, before any model forward pass (C45). Its amended run measured Ψ=+0.430 logit
units but failed its frozen resolution gate (MDE80 0.0647 against 0.05), so the verdict is
`INSTRUMENT_UNRESOLVED` and the measurement supports no behavior conclusion (C46).

## The paper

A paper built on the donor mechanism was accepted to the NeurIPS 2026 XAI4Science workshop. Reviewer
comments led to a round of record fixes in October 2026. C43's crossed-register endpoint and its 99
pseudo-register controls now re-execute bit for bit from committed raw rows (R49), two summary statements
were corrected (R53, R54), and the CC/max results were committed (R56).
[RESULTS.md](RESULTS.md) holds the quantitative summary and the audit of the manuscript draft.

## Where things stand

On this checkpoint, the donor mechanism's address, payload route and delivery are causally supported.
The payload's meaning is unknown, and minimality is not claimed. The open questions are what the
payload encodes, whether any of this transfers to other models under a preregistered test, and whether
the coding-boundary preference has a mechanism. They are laid
out in the [roadmap](ROADMAP.md).

## What else is in this folder

| File | Use it to |
|---|---|
| [MECHANISM.md](MECHANISM.md) | Read the mechanism statement and its limits |
| [RESULTS.md](RESULTS.md) | Look up a number, or check the manuscript audit |
| [METHODOLOGY.md](METHODOLOGY.md) | Design an experiment; includes the evidence ladder and the rules for biological claims |
| [TRANSFER_GUIDE.md](TRANSFER_GUIDE.md) | Apply the method to another model |
| [NOMENCLATURE.md](NOMENCLATURE.md) | Check which terms are retired and which numbers name which objects |
| [PREREG_CONFORMANCE.md](PREREG_CONFORMANCE.md) | Check whether a preregistered gate was actually enforced |
| [ROADMAP.md](ROADMAP.md) | See next steps, deferred work, and open owner decisions |
| [REPRODUCE.md](REPRODUCE.md) | Set up the environment, get the data, rerun a result; third-party licenses |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Add a study or change the record; how to audit a branch |
| [REFERENCES.md](REFERENCES.md) | External literature |
| [history/](history/) | Two dated reviews kept because code cites them |

Claims and corrections are not kept here. Each study owns its `CLAIMS.md`, `CORRECTIONS.md` and
`DECISIONS*.md`, and the [results index](../results/RESULTS_INDEX.md) maps every artifact to its claim.
