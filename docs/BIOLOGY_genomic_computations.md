# Genomic Computations for Circuit Hunters

This document is a biology primer for the ML researchers on this project. It exists to answer one
question for each phenomenon we study: *what is the DNA model actually being asked to compute?*
Mechanistic interpretability needs a precise target — "predict the next nucleotide" is not
precise enough, because the correct prediction at a given position is the output of a different
underlying computation depending on genomic context (reading frame, repeat structure, splice
site, etc.). This doc gives the minimum molecular biology needed to state each target precisely,
and for every computation we distinguish two hypotheses an interpreter should hold in tension:
a **pooled / shortcut** solution (some cheap, non-positional statistic that gets most of the
loss for free) and a **circuit** solution (a mechanism that tracks an actual biological anchor —
a position, a strand, an orientation). Standard textbook molecular biology is stated below without
citation; specific quantitative claims and every non-textbook fact are sourced.

---

## 1. Minimum molecular biology

**DNA alphabet and strands.** DNA is a linear polymer of four nucleotide bases — adenine (A),
cytosine (C), guanine (G), thymine (T) — read as a sequence over the alphabet `{A, C, G, T}`
(the `N` token in GFM-nano's vocabulary denotes an unknown/masked base, not a fifth biological
nucleotide). DNA is double-stranded: the two strands run antiparallel and are complementary under
the pairing rule A–T, C–G, so the sequence on one strand determines the sequence on the other by
reverse-complementation. A gene can be encoded on either strand; "reverse complement" is not an
edge case, it is half of the genome. Any circuit that is supposed to generalize across the genome,
rather than fit one strand's idiosyncrasies, should be checked for behavior under
reverse-complementation of its input.

**Central dogma.** The classical flow of information is DNA → RNA → protein. A gene's DNA is
*transcribed* into a messenger RNA (mRNA) copy (using U in place of T), and the mRNA is
*translated* into a protein — a chain of amino acids. Not all DNA sequence is protein-coding: most
of the genome is transcribed rarely or never, or is transcribed but not translated. The
reading-frame and NMD computations described below apply specifically to the coding portion.

**Gene, CDS, exon, intron.** A *gene* is a stretch of genomic DNA that gets transcribed into RNA.
Within a gene, the *coding sequence* (CDS) is the subset of that RNA that is actually translated
into protein — it excludes untranslated flanking regions. Genes are broken into *exons* (segments
that are retained in the mature, spliced mRNA) separated by *introns* (segments that are cut out
during processing). The CDS is the concatenation of the coding parts of the exons after splicing;
the intron sequence never appears in the mature mRNA and is never translated. Splicing is what
makes "the CDS" a discontiguous subset of the raw genomic sequence — a fact that matters directly
for the NMD rule in Section 4, which is defined relative to exon-exon junctions, not raw genomic
position.

**Transcription.** An enzyme (RNA polymerase) reads one strand of the DNA template and synthesizes
a complementary RNA strand, base by base, 5′ to 3′. The immediate product (pre-mRNA) contains both
exons and introns; splicing then removes the introns and ligates the exons together into the
mature mRNA that is exported for translation.

**Translation.** Ribosomes read the mature mRNA in triplets and, guided by transfer RNAs, assemble
the corresponding chain of amino acids. Translation begins at a *start codon* and proceeds
codon-by-codon until a *stop codon* is reached; the ribosome does not re-read or skip — it
consumes the mRNA in a strict, non-overlapping, triplet-at-a-time scan. That scan is the source of
"reading frame" as a computational quantity (Section 2).

**Codon and the genetic code.** A *codon* is a group of three consecutive nucleotides. There are
4³ = 64 possible codons. The *genetic code* is the fixed, (nearly) universal mapping from each of
the 64 codons to one of 20 amino acids, or to a translation-stop signal. Because there are only 20
amino acids and 61 non-stop codons, the code is *degenerate*: most amino acids are specified by
more than one codon (synonymous codons). This redundancy is the mechanistic basis of codon usage
bias, discussed in Section 2.

**Start codon.** Translation (in the standard case) begins at an ATG codon, which also codes for
the amino acid methionine. Not every ATG in a transcript is a start codon — translation initiation
depends on surrounding sequence context (e.g., a favorable Kozak-like context in the classical
eukaryotic case) — but every CDS begins with an ATG, and that ATG defines position zero of the
reading frame for everything downstream of it.

**Stop codons.** Three of the 64 codons — TAA, TAG, TGA — do not code for an amino acid; instead
they signal the ribosome to terminate translation. A CDS therefore looks like: ATG, then a run of
sense codons, then one of {TAA, TAG, TGA}, with no stop codon occurring in-frame anywhere in
between (Section 2 explains why this last property is close to definitional, not incidental).

---

## 2. Reading frame as a computation

**Definition.** Fix a start codon ATG at some genomic position `p`. For any downstream position
`p + k`, its *reading frame* (or *phase*) is `k mod 3` — equivalently, which of the three
nucleotides within its codon it occupies (first, second, or third codon position). The frame of a
position is not a property of the DNA sequence alone; it is a property of the sequence *relative
to* a chosen ATG anchor. The same nucleotide, read from a different start point, sits in a
different frame and — if translated — would specify a different codon and a different amino acid.

**Why three forward frames.** A double-stranded, linear DNA sequence read left-to-right can be
partitioned into codons starting at position 0, 1, or 2 — three possible offsets, hence three
forward reading frames (and, symmetrically, three more on the reverse-complement strand, for six
total, though this project's tasks are stated on a single designated strand). Only one of the
three forward frames is the one the ribosome actually uses for a given CDS, because only one of
them is anchored at the real ATG.

**Open reading frame (ORF).** An ORF is a run of sequence, starting at ATG and read in one
consistent frame, that contains no in-frame stop codon until its own terminal stop codon. "In
CDS-frame reading is an open reading frame" is close to a tautology: a CDS is *defined* as the
region that gets translated start-to-stop without interruption, so by construction it cannot
contain an in-frame stop codon before its true terminus — if it did, translation would halt there
and the region past that point would not, in fact, be part of the same CDS. This is why "no
in-frame stop codons" is a near-universal, checkable property of real coding sequence in the
correct frame, and a useful planted invariant for synthetic data (Section 6).

**Why this is the right first circuit target.** Reading frame is attractive as a first target
precisely because it admits two competing solutions with very different mechanistic footprints,
and the gap between them is small enough to be interesting but not zero:

- *The pooled shortcut.* Because the genetic code is degenerate, organisms do not use synonymous
  codons with equal frequency — coding sequence has a biased, non-uniform codon usage spectrum
  (some synonymous codons are used far more often than others, for reasons including tRNA
  abundance and other translational-efficiency pressures). A consequence is that the local
  *composition* of nucleotides — which triplets appear, ignoring where the reading actually
  started — is itself weakly informative about which frame is "real," without ever locating the
  ATG. A model could in principle look at a pooled bag of nearby triplet counts (a
  position-agnostic statistic) and get most of the way to the right answer, because on real human
  CDS this compositional signal is genuinely present, if weak: this project's working figure is
  that local codon composition carries roughly 0.024 nats of mutual information with phase — a
  small but nonzero signal, and one a network with enough capacity can exploit without ever
  representing "where is the ATG."
- *The true circuit.* The mechanistically correct computation is to locate the actual ATG (or
  otherwise inherit its position from context) and then track position mod 3 forward from it —
  an exact counting operation anchored at a specific token, not a statistic pooled over a window.

The interpretability bet is that these two solutions are behaviorally close (the composition
shortcut buys a lot of accuracy) but mechanistically completely different (one needs no anchor and
no counter; the other needs both), and that **only near-perfect accuracy forces the network to
build the counting circuit** — a model satisfied with the accuracy the shortcut alone provides has
no gradient pressure to build anything more expensive. This makes reading frame a clean case study
in shortcut learning under capacity/accuracy pressure, the general phenomenon documented for deep
networks more broadly in Geirhos et al. (2020). Concretely, an
interpreter chasing the circuit should look for a component whose activation depends on *distance
since the last ATG* (a counter, likely realized as some form of positional/modular tracking), as
opposed to a component whose activation depends only on *which triplets are locally present*
regardless of offset (a compositional detector). The two are dissociable by construction: you can
build synthetic sequences that fix local codon composition while moving the ATG, or fix the ATG
while permuting codon identities subject to the frame constraint, and see which one moves the
network's prediction (see Section 6).

---

## 3. Repeats and copying as a computation

Real genomes are full of repeated sequence, of several structurally distinct kinds:

- **Tandem repeats.** A short motif repeated back-to-back many times in a row (e.g., a
  microsatellite of a few bases repeated dozens of times consecutively). The repeat unit and copy
  number vary; the defining property is immediate adjacency of copies.
- **Transposable elements.** Mobile genetic sequences that have copied themselves into many
  locations across the genome over evolutionary time. Copies at different genomic loci are
  similar to each other (often not identical, since each copy accumulates mutations
  independently) but are not adjacent to one another.
- **Segmental duplications.** Larger blocks of sequence (up to genes or multi-gene regions) that
  exist in more than one copy in the genome, again generally at separate loci rather than
  back-to-back.

**Why prediction inside a repeat is a copy operation.** Whatever the repeat's structural class,
the common computational fact is this: once a model has seen enough of a repeated segment to
recognize *that it is inside a repeat* and *which earlier stretch of sequence it is a repeat of*,
the correct next-token prediction is simply to retrieve whatever came next after the matching
earlier occurrence and emit that. This is a lookup-and-copy operation, not a "generate from
context statistics" operation — the content being predicted was never learnable from composition
or local statistics alone, because a competent predictor needs to reproduce the *specific*
sequence that occurred earlier, not merely something typical of that class of repeat.

**The induction-head analogy.** This is structurally identical to the induction-head mechanism
described in the language-model interpretability literature: a two-step circuit in which one
component finds a previous occurrence of the current token (or a short local context match) and a
second component copies whatever token followed that previous occurrence, producing the
"complete-the-pattern" behavior characteristic of in-context copying, as described in
Olsson et al. (2022). The genomic analogue is a natural
target for exactly the same circuit-discovery techniques used to find induction heads in language
transformers: we expect a "previous-token"/matching component followed by a copy/read-out
component, realized over nucleotide tokens instead of word or byte tokens. Where the reading-frame
task probes a network's ability to track *position relative to an anchor*, the repeat/copy task
probes its ability to track *content relative to a remembered occurrence* — a different, and in
some ways more classically "attention-shaped," retrieval computation.

---

## 4. A tour of upcoming genomic computations

Each of the following is a candidate future target. For each, we give: the biology, the
computation a model must perform to solve it exactly, and what a pooled shortcut versus a genuine
circuit would look like.

### Splice-site recognition and exon definition

*Biology.* Introns are removed from pre-mRNA by the splicing machinery, which recognizes short,
partially conserved motifs at intron boundaries: the intron overwhelmingly begins with GT (the
*donor* site, at the exon-intron boundary) and ends with AG (the *acceptor* site, at the
intron-exon boundary) — the canonical "GT...AG" rule. But GT and AG dinucleotides are extremely
common in the genome by chance; the great majority of GT/AG dinucleotides are not real splice
sites. Correct splice-site recognition also depends on additional weak signals in the surrounding
sequence and on exon length and definition — real splicing machinery is understood to often
recognize an exon as a *unit* bounded by an acceptor and a donor site, rather than scoring each
splice site in isolation.

*The computation.* A model must combine a weak local dinucleotide cue (GT or AG) with enough
surrounding context to distinguish a functional splice site from the much larger population of
coincidental GT/AG occurrences.

*Shortcut vs. circuit.* A pooled shortcut would score a fixed local window around each GT or AG
independently — treating "is this a splice site" as a per-position classification driven mostly
by adjacent sequence, with no reference to any paired site. This underperforms because it throws
away the constraint that a real intron needs *both* a donor and an acceptor at a plausible
distance, bounding a plausible exon. The circuit-level solution is closer to summed
exon-definition scoring: evaluate a donor-acceptor pair jointly (or an exon as bounded by its
two flanking sites) rather than each motif alone, which is expected to outperform single-site
scoring precisely because it uses the pairing constraint the single-site version discards. An
interpreter should look for attention or induction-like structure that links a candidate donor to
a candidate acceptor at long range, not merely local motif detectors.

### Nonsense-mediated decay (NMD) and the 50–55 nt rule

*Biology.* Nonsense-mediated decay is a quality-control pathway that degrades mRNAs bearing a
*premature* stop codon — a stop codon that terminates translation earlier than the transcript's
authentic stop. Cells distinguish a premature from a normal stop largely by position relative to
splicing: after splicing, a protein complex is deposited at each exon-exon junction, and — in the
classical rule — a stop codon located more than roughly 50–55 nucleotides upstream of the *last*
exon-exon junction is treated as premature and triggers degradation of the transcript, whereas a
stop codon in the last exon, or close enough to the final junction, is treated as normal.

*The computation.* This is fundamentally a positional/counting computation relative to an anchor
that is itself derived from splicing: the model must know where the last exon-exon junction is
(a fact not visible from local sequence around the stop codon alone) and count the distance from
the stop codon to that junction, then threshold at roughly 50–55 nt.

*Shortcut vs. circuit.* A pooled shortcut might learn "stop codons near the end of a long
transcript are usually fine, stop codons in the middle are usually flagged" from gross statistical
regularities of transcript length and stop-codon position, without ever representing "junction" as
a distinct entity. The circuit solution requires explicitly locating the last exon-exon junction
(itself downstream of solving splicing) and performing a genuine distance count from the stop
codon to that junction, then comparing to a threshold — an anchored counting computation
one step more complex than reading frame, because the anchor itself is a derived, non-local
quantity rather than a fixed sequence motif like ATG.

### CTCF motif orientation

*Biology.* CTCF is a DNA-binding protein whose binding sites contain a recognizable sequence
motif; a distinguishing feature of CTCF sites relevant to genome folding is that they are oriented
— the motif has a direction on the DNA (which strand, which way it reads), and pairs of CTCF sites
tend to organize chromatin loops preferentially when they face each other in a "convergent"
orientation, rather than in a "same-direction" or "divergent" orientation.

*The computation.* Simply detecting the motif's presence is not sufficient to predict its
functional role; the direction the motif faces relative to a nearby partner site is itself the
functionally relevant variable. This makes CTCF orientation a *relational* feature: the same motif
sequence, read on the two strands, functions differently depending on orientation and on the
orientation of whatever site it might pair with downstream.

*Shortcut vs. circuit.* A pooled shortcut would detect "CTCF motif present nearby" without
representing strand/direction at all — sufficient for some coarse tasks but blind to any task that
depends on orientation. A circuit-level solution must represent the motif together with its
orientation as a joint feature, and, for loop-relevant tasks, relate the orientation of one site to
that of another site at long genomic distance — structurally analogous to needing a signed,
directional feature rather than an unsigned presence/absence detector.

### G-quadruplexes and RNA secondary structure

*Biology.* Some DNA and RNA sequences, particularly those rich in guanine, can fold back on
themselves and form structures stabilized by non-standard base pairing — G-quadruplexes in
G-rich DNA/RNA, and more general secondary structure (hairpins, stems, loops) formed by
Watson-Crick base pairing between distant parts of a single RNA molecule. Both phenomena mean
that a nucleotide's biological role can depend on which other, potentially far-away, nucleotide it
pairs with in three-dimensional folded structure, not merely on the nucleotides immediately next
to it in the linear sequence.

*The computation.* This is a long-range relational computation: predicting folding behavior (or
downstream effects of folding, such as effects on replication or translation) requires identifying
compatible pairing partners at long linear distance and evaluating the folded structure that
results, rather than reading local sequence context.

*Shortcut vs. circuit.* A pooled shortcut would use local compositional cues (e.g., raw G-richness
as a proxy for quadruplex propensity) without representing any actual pairing partner — a coarse
signal that correlates with the true structural computation without performing it. A genuine
circuit would need something closer to a matching/attention computation over long-range pairs of
positions, similar in shape to a base-pairing "contact prediction" problem: identify candidate
partner positions and evaluate their compatibility, rather than scoring each position in isolation.

---

## 5. How GFMs "see" DNA

**Tokenization.** DNA foundation models diverge sharply from natural-language models in how they
tokenize, and the choice has direct consequences for what a reading-frame or repeat circuit could
even look like mechanistically. GFM-nano tokenizes at single-nucleotide resolution (vocabulary
`{A, C, G, T, N}`), so one token is exactly one base — a codon spans exactly three consecutive
token positions, and reading frame is literally "position mod 3 in token-index space," with no
ambiguity introduced by the tokenizer itself. Other published DNA models make different choices:
DNABERT (Ji et al., 2021) tokenizes DNA sequence into
overlapping k-mers, and the
Nucleotide Transformer (Dalla-Torre et al., 2024) uses
non-overlapping k-mer tokenization; both choices mean a single token can straddle codon or frame
boundaries in a way that a single-nucleotide tokenizer does not, which is one reason GFM-nano's
single-nucleotide vocabulary is deliberately convenient for the reading-frame study — it removes
tokenization as a confound when asking whether the network tracks phase.

**Masked-LM vs. causal objectives.** DNABERT and the Nucleotide Transformer are trained with
masked-language-modeling objectives (predict a masked-out span from bidirectional context), which
is a natural fit for DNA, where "upstream" and "downstream" context are both biologically
available to the actual cellular machinery reading the sequence (a ribosome, a splicing complex,
etc., all act on a complete, already-existing molecule). Enformer
(Avsec et al., 2021) takes a long genomic window as input and
predicts many genomic tracks (e.g., measures of gene expression and regulatory activity) as
output, rather than doing next-token or masked-token prediction. GFM-nano, by contrast, is a
causal (autoregressive) small transformer — a choice that keeps it architecturally close to the
language-model interpretability toolkit but means that, unlike a real ribosome or spliceosome, it
never sees sequence downstream of its current position at inference time. This is a real
mismatch with biology worth keeping in mind: some biological decisions (e.g., recognizing an
acceptor site, or the last exon-exon junction for NMD) genuinely depend on sequence that a causal
model can only have inferred from earlier context, never observed directly.

**Caveats an interpreter must respect.**

- *Strand symmetry.* Because a gene can sit on either DNA strand, and the reverse complement of a
  functional motif is not generally the same string, a circuit discovered on forward-strand
  examples should be checked for whether it is strand-specific (correctly, biologically) or
  merely an artifact of how training data was generated (incorrectly, a confound).
- *Coding vs. non-coding context.* Reading frame, ORFs, and codon usage are properties of coding
  sequence; splice motifs live at intron/exon boundaries; NMD depends on exon-junction structure.
  A window of raw genomic sequence handed to the model may be entirely non-coding, entirely
  intronic, or span a boundary — the "correct" computation to expect from the network is not the
  same in every window, and an interpreter should always ask which regime a given input actually
  falls into before judging whether a circuit fired correctly.
- *Partial genes in a window.* A fixed-length training window is not guaranteed to contain a
  whole gene, a whole exon, or even a whole codon at its edges. A model may need to represent "I
  do not know the frame/exon structure here because the anchor (ATG, junction, etc.) is outside my
  window" as a distinct state from confidently computing frame from a visible anchor — worth
  checking for directly, rather than assuming the model always has enough context to solve the
  task as posed.

---

## 6. Reverse-engineering checklist for the biologist-skeptic

For each computation studied, the corresponding biological ground truth should be plantable in
synthetic data and checkable in real data, so that a circuit claim is falsifiable rather than
asserted from behavioral accuracy alone.

- **Reading frame.** Ground truth: the true ATG position and the resulting frame of every
  downstream position are known exactly by construction in synthetic CDS, and the in-frame-stop
  absence property is directly checkable (a correct CDS in the claimed frame must have zero
  in-frame stop codons before its terminus). Falsification test: construct paired sequences that
  hold local codon composition fixed while moving the ATG (or vice versa) and check whether the
  claimed "counting" component's activation follows the ATG or follows the composition.
- **Repeat/copy.** Ground truth: plant repeats of known type (tandem, dispersed) at known
  positions with known source spans, so the "which earlier occurrence" answer is known exactly.
  Falsification test: check whether an induction-like component's output changes when the *earlier
  occurrence* is edited (should change the prediction) versus when unrelated flanking sequence is
  edited (should not).
- **Splice sites.** Ground truth: canonical GT/AG dinucleotides at planted, known intron
  boundaries, with distractor GT/AG dinucleotides planted elsewhere that are not real boundaries.
  Falsification test: check whether the model's score depends on the presence of *both* donor and
  acceptor at a plausible distance (exon-definition behavior) versus firing on an isolated
  motif with no valid partner.
- **NMD 50–55 nt rule.** Ground truth: constructs with a premature stop codon placed at a known,
  controlled distance upstream of a known, controlled exon-exon junction position, spanning both
  sides of the 50–55 nt threshold. Falsification test: check whether the predicted outcome flips
  at the correct distance threshold when the junction is moved, as opposed to depending only on
  gross transcript length or stop-codon position in isolation.
- **CTCF orientation.** Ground truth: planted motif instances in both orientations, alone and in
  oriented pairs (convergent vs. non-convergent), with known expected functional readout.
  Falsification test: check whether the model's representation of the motif changes under
  reverse-complementation of just that motif (holding everything else fixed) — a true
  orientation-sensitive circuit must react to this; a presence-only detector will not.
- **G-quadruplex / secondary structure.** Ground truth: planted sequences with known folding
  partners (designed complementary stems) versus scrambled controls with matched local composition
  but no valid long-range pairing partner. Falsification test: check whether the model's behavior
  depends on the specific identity of the distal partner sequence (true pairing computation) or
  only on local G-richness / composition (shortcut).

---

## Glossary

- **ATG / start codon** — the codon that (in the standard case) initiates translation; also codes
  for methionine.
- **CDS (coding sequence)** — the subset of a transcript's exonic sequence that is translated into
  protein.
- **Codon** — a group of three consecutive nucleotides; the unit read by the ribosome during
  translation.
- **Codon usage bias** — the phenomenon that synonymous codons (codons specifying the same amino
  acid) are not used with equal frequency in real coding sequence.
- **CTCF** — a DNA-binding protein whose oriented binding sites help organize chromatin loop
  structure.
- **Exon / intron** — retained vs. excised segments of a gene's transcript, respectively, relative
  to splicing.
- **G-quadruplex** — a four-stranded DNA/RNA secondary structure formed by G-rich sequence via
  non-Watson-Crick base pairing.
- **Genetic code** — the fixed mapping from each of the 64 codons to one of 20 amino acids or to a
  stop signal.
- **NMD (nonsense-mediated decay)** — an mRNA quality-control pathway that degrades transcripts
  bearing a premature stop codon.
- **ORF (open reading frame)** — a run of sequence starting at a start codon and read in a fixed
  frame, containing no in-frame stop codon until its own terminal stop.
- **Reading frame / phase** — the offset (mod 3) of a position relative to a chosen start-codon
  anchor.
- **Reverse complement** — the sequence obtained by pairing-complementing and reversing the order
  of a DNA strand, equal to the sequence read on the opposite strand.
- **Splice donor / acceptor site** — the GT (donor) / AG (acceptor) dinucleotides marking the
  start/end of an intron.
- **Stop codon** — one of TAA, TAG, TGA; signals translation termination.
- **Tandem repeat / transposable element / segmental duplication** — three structurally distinct
  classes of repeated genomic sequence, distinguished by adjacency and copy mechanism.
- **Transcription / translation** — DNA-to-RNA copying, and RNA-to-protein decoding, respectively.

## References

- Avsec, Ž., et al. (2021). *Enformer: effective gene expression prediction from sequence by
  integrating long-range interactions*. Nature Methods.
  https://doi.org/10.1038/s41592-021-01252-x
- Dalla-Torre, H., et al. (2024). *Nucleotide Transformer*. Nature Methods.
  https://doi.org/10.1038/s41592-024-02523-z
- Geirhos, R., et al. (2020). *Shortcut learning in deep neural networks*. Nature Machine
  Intelligence. https://doi.org/10.1038/s42256-020-00257-z
- Ji, Y., et al. (2021). *DNABERT: pre-trained Bidirectional Encoder Representations from
  Transformers model for DNA-language in genome*. Bioinformatics.
  https://doi.org/10.1093/bioinformatics/btab083
- Olsson, C., et al. (2022). *In-context Learning and Induction Heads*. Anthropic / arXiv.
  https://doi.org/10.48550/arxiv.2209.11895
