# Biological hypothesis register — SpliceBERT positional boundary coherence

> Candidate biological narrative; it cannot promote a claim or supersede the
> frozen design and claim ledger.

**Status: candidate register.** This document is not a claim ledger, a preregistration, or evidence. The
successor study's authoritative record is [`CLAIMS.md`](CLAIMS.md); prior evidence remains governed by
the partner-selection [`CLAIMS.md`](../splicebert_partner_selection/CLAIMS.md), the GTEx fine-tuning
[`CLAIMS.md`](../splicebert_gtex_finetune_shift/CLAIMS.md), and the result manifests that those ledgers
name. If this register and a ledger disagree, the ledger wins.

This is a focused replacement for scratchpad entries HYP-08 through HYP-12. It does not adopt the
scratchpad's untracked `[computed]` quantities as evidence. The first phase below either reproduces the
needed sequence-side measurements in an immutable artifact or leaves them out of the paper.

Read [`docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md) before moving text from this register into a
manuscript. In particular, **annotated coding-boundary phase**, **carrier register**, and **reading frame**
are different objects.

---

## 1. The decision

The current record does **not** establish a chain

> period-3 → ≈10.63 → annotated boundary phase.

It establishes three checkpoint-scoped observations:

1. In base `SpliceBERT.1024nt`, the clean nucleotide-invariant `C36` surgery establishes causal,
   phase-aligned use of a period-3 hidden direction for donor–acceptor coupling. `C34` supplies
   discovery-established evidence that the contribution is span-distributed, with only gate-qualified
   confirmation. Within carrier patching, `C35` recovery is mediated mainly through L3H1's attention
   pattern rather than carrier-dependent changes in that head's own values; upstream effects on payload
   construction remain possible. Coding enrichment and reading-frame semantics are not established.
2. A distinct ≈10.63 carrier reaches controlled post-softmax attention. Nine readers are present on the
   official 1024-nt checkpoint. Their conditional simple value-main effects on the donor coupling margin
   are sub-SESOI; the ordinary coupling change is small and suppressive, with equivalence underpowered;
   and the preregistered donor-readout `U_info` write was not computed and is not evaluable as scoped
   (`C11`, `C12`, `C37`). Fine-tuned descendants retain spectral power and peaks at the named band and
   causally use it for **local** donor and acceptor classification, with no detected paired-site coupling
   (`GFS23`, `GFS25`).
3. In base `SpliceBERT.1024nt`, annotated phase-matched chimeric boundaries have a larger masked-donor
   coupling response than mismatched boundaries at one 120-nt span (`C43`). The existing pseudo-phase null
   does not match phase-correlated sequence composition. The attempted period-3 bridge failed its
   instrument gates before the target was opened, so it is unresolved rather than negative (`C44`).

The leading hypothesis is therefore asymmetric:

> **Coding-sequence statistics may help select useful positional frequencies. SpliceBERT uses period-3 as a
> relative routing coordinate and reuses both period-3 and ≈10.63 for local splice-site evidence after
> fine-tuning. Its apparent preference for matched coding-boundary phase most likely reflects
> phase-correlated boundary sequence structure. Whether either carrier causally supports that preference is
> an open, testable bridge.**

This account is biologically grounded without giving the model a frame sensor. Period-3 sequence structure
can arise from unequal codon use, and published intron-phase associations can arise from splice-site
nucleotide bias. A 10–11-nt coding signal also has a plausible protein-sequence source, but that source has
not been demonstrated in the SpliceBERT training distribution.

The learned relationship need not be perfectly symmetric across all three annotated orientations to be
scientifically interesting. SpliceBERT saw a finite, potentially uneven training corpus, so partial register
concordance could be compatible with uneven support for the relevant boundary patterns. This is a candidate
limitation, not an explanation established by the present model outcomes. It cannot excuse failed controls
or change a decision threshold; a causal training-data account would require an independent corpus analysis
and controlled training intervention.

### Checkpoint map

| Observation | Checkpoint | Highest licensed level | Not licensed |
|---|---|---|---|
| period-3 donor circuit | base `SpliceBERT.1024nt` | causal behavior and conditional attention mediation | codon reading or coding specificity |
| ≈10.63 attention readers | stage-1 `SpliceBERT.510nt`; replicated readers on base `SpliceBERT.1024nt` | controlled post-softmax attention; sub-SESOI simple value-main on the donor coupling margin | `U_info` write or general biological function |
| retained P3 and ≈10.63 local use | five GTEx-fine-tuned 1024-nt classifiers | causal local donor/acceptor margins | base-model partner-selection mechanism |
| annotated boundary-phase preference | base `SpliceBERT.1024nt` | behavior in fresh chimeric constructs | circuit, frame sensor, or spliceosomal mechanism |

No analysis may pool these checkpoints. A cross-checkpoint comparison must report each checkpoint's result
separately.

**Audit caveat.** `C43` has an independent result audit, but its raw rows are absent, the audit is not
clean-checkout re-executable, gates 2–9 survive as attestations, and its resolution pass sits at the
Monte-Carlo noise floor. `C44` reused a development lattice whose behavior was already observed and was not
outcome-blind for that preflight. Neither the exon-edge study nor the GTEx gates have an independent
repository-wide preregistration-conformance audit in
[`docs/PREREG_CONFORMANCE.md`](../../docs/PREREG_CONFORMANCE.md). The GTEx ledger calls `GFS23`/`GFS25`
established while the global manifest marks their controlling artifacts `development`; reconcile that
status before paper prose uses “established.” `GFS25`'s V2 genes were named-carrier-outcome-fresh but had
appeared in the earlier candidate scan.

---

## 2. The three live hypotheses

The IDs below belong only to this register. They are not claim IDs.

### PBC-01 — Is the boundary-phase preference a sequence-encoded coding-boundary prior?

**Priority: 1. This is the modal explanation.**

**Candidate.** Annotated intron phase is correlated with codon-position composition, terminal codons, and
splice-site boundary bases. A masked-language model can therefore prefer a phase-matched join by reading
ordinary nucleotide structure. It need not represent translation, count codons, or infer frame.

**Why it fits.** `C43` is a large and freshly replicated behavioral contrast, but its 99 pseudo-phase
controls permute labels while leaving sequences fixed. They reject an arbitrary label effect; they do not
separate annotated phase from sequence features that covary with phase. The prespecified stop-load analysis
weakens one sparse rival, but it does not control dense mono-, di-, tri-, codon-position, or splice-boundary
composition. Published work likewise reports that apparent intron-phase preferences around amino acids and
protein secondary structure can be explained largely by splice-site nucleotide bias.

**Strong rival.** A phase-linked behavior remains after composition balance because the model uses a
relative carrier register to compare the two boundaries. That rival is PBC-02.

**Decisive observation.** On a fresh, outcome-blind crossed lattice, the phase-match contrast remains
material after balance on the frozen composition family and after a classifier can no longer recover the
match label from that family. Failure to achieve balance is an instrument failure. A capable equivalence to
zero after balance supports a composition-associated explanation within the measured feature family and
ends this cohort × feature-family × carrier-intervention bridge program.

**FORBIDDEN:** “SpliceBERT reads frame”; “the spliceosome checks frame”; “only phase-matched joins couple”;
“period-3 explains `C43`.”

**PERMITTED NOW:** *At one fixed span, phase-matched chimeric coding boundaries produce a larger
masked-donor coupling response than mismatched boundaries. Existing controls do not distinguish annotated
phase from phase-correlated sequence composition.*

### PBC-02 — Does period-3 causally implement the residual boundary preference?

**Priority: 2 and the primary paper closure path. A corrected directional Phase-II P3 result supports a
rough scoped link; a material result is stronger and alone opens Phase III. Register and route tests are
optional strengthening. ≈10.63 is optional.**

**Candidate.** Period-3 supplies a short relative register. The ≈10.63 carrier supplies a distinct,
coarser positional coordinate. The two features can be used in parallel. They do not need to transform into
one another, and they need not share a head or downstream route.

**Why it fits.** Period-3 already supports base-model partner coupling and acts mainly through attention.
The contribution is span-distributed, which makes it a plausible bridge to a relational boundary effect.
Both named bands also retain spectral power and peak frequency after fine-tuning, and their controlled
attenuation independently changes local donor and acceptor margins. This makes reuse plausible. It does not
establish reuse for the `C43` estimand; the preregistered full-rank coordinate-retention test did not pass,
and the near-perfect dominant rank-2 overlaps are post hoc.

**Strong rivals.** The phase preference is fully explained by PBC-01; period-3 supports ordinary coupling
but not the phase contrast; ≈10.63 supports local classification only; or an untested distributed route
supports the contrast without either named band.

**Predictions.** On identical, composition-balanced sequences:

- health-preserving period-3 attenuation reduces the phase-match contrast;
- health-preserving ≈10.63 attenuation may reduce the contrast independently;
- joint attenuation has a prespecified interaction sign rather than a post-hoc label;
- route-specific restoration after attenuation rescues the effect in Phase III; and
- same-band equal-RMS phase controls and intervention-size-matched null bands do not reproduce it.

A positive period-3 result does not promote ≈10.63. Each band has its own hypothesis family. A positive
joint interaction is interpretable only when both single-band interventions pass their capability and
health gates.

**FORBIDDEN:** a linear “3 → 10.63 → phase” diagram; “two-frequency frame code”; “the model counts codons”;
use of `GFS25` as if it were a base-model coupling result; interpreting a failed preflight as a null.

**PERMITTED NOW:** *Period-3 is a causal relative coordinate for base-model partner coupling. Period-3 and
≈10.63 each support local splice-site evidence after fine-tuning. Their role in annotated
boundary-phase preference has not been tested with a working instrument.*

### PBC-03 — Does ≈10.63 have a coding-sequence referent, or is it an optimization-selected ruler?

**Priority: 3. This is an origin fork, not a prerequisite for PBC-01 or PBC-02.**

**Biological candidate.** Amphipathic helices can impose approximately 3.6-residue compositional
correlations that may be reflected as approximately 10.8-nt correlations in coding sequence. A conserved
CDS-enriched signal in the actual multispecies training distribution could help select the ≈10.63 carrier.

**Model-native rival.** The optimizer manufactures a second frequency that complements period-3 or fits the
finite learned position table. The measured carrier is not precise enough to reject `32/3 = 10.6667`, which
corresponds to 48 cycles across a 512-row table. The existing checkpoint contrast associates the carrier
with the multispecies training regime but does not isolate corpus size, composition, sampling, training
duration, or optimization history (`C39`; `R37`).

**Why this remains open.** The ≈10.298 mouse ES-cell dyad period is distinguishable from ≈10.63, rejecting
numerical identity to that measured signal (`C14`). Broader DNA-folding or chromatin-associated sources are
not adjudicated, but they are not an active candidate in this closure. There is an unresolved source
conflict that must be corrected in the canonical record before broader DNA-helical prose is used:
[`docs/NOMENCLATURE.md`](../../docs/NOMENCLATURE.md) gives a 10.2–10.5 literature range, while Rhodes and
Klug report 10.6 ± 0.1 bp for random-sequence DNA. The existing whole-human-genome and ORF-free-model
results are sensitivity-bounded and do not assay matched multispecies CDS (`C40`).

**Decisive observations.** The biological candidate requires a reproducible 10.5–10.8-nt signal in matched
multispecies CDS, enrichment in α-helical versus β/coil coding regions, survival in held-out species, and
loss under sequence controls that preserve simpler composition. The ruler candidate requires an explicit
`λ = 32/3` test and, ultimately, a controlled position-table-length or context-length intervention. A null
CDS scan with passing sensitivity and resolution demotes the tested biological referent; it leaves the ruler
candidate open but does not positively support it.

**FORBIDDEN:** “≈10.63 is nucleosomal”; “≈10.63 is α-helical”; “corpus-derived”; “10.630 ± 0.005”; absence
claims from sensitivity-bounded scans.

**PERMITTED NOW:** *≈10.63 is a model-native, learned, training-regime-associated positional carrier with a
demonstrated local-classification function after fine-tuning. Its biological referent, if any, is unknown.*

---

## 3. Three-phase sequence with Phase-II minimum closure

The phases are gates. The clock does not authorize a later phase. Every result-producing run must write an
immutable machine-readable artifact and manifest, and an independent audit must recompute the decision
statistics from recorded endpoints.

### Phase I — separate sequence biology from model geometry

This phase reads no new SpliceBERT outcomes. Sample planning may use the frozen, sign-free `C43` variance.

#### I-A. Freeze a composition-balanced, outcome-blind boundary cohort

Build a new gene-disjoint 3 × 3 crossed lattice. Blacklist every gene used by `C43`, `C44`, or `GFS25`.
Do not spend the globally sealed chromosome pool by default. It requires an explicit owner decision and a
provenance update; otherwise name and hash a fresh split from permitted unsealed chromosomes before model
inference. Do not use the known `C43` target as confirmatory evidence.

Freeze the matching family before cohort selection:

- donor and acceptor core boundary one-hot features, scored separately from flanking composition;
- exonic mono-, di-, and trinucleotide features on both sides;
- codon-position-specific base and trinucleotide summaries;
- GC, CpG, source-exon length, strand, chromosome, and donor–acceptor span;
- donor and acceptor sequence scores, with their exact support declared;
- in-frame stop count and stop excess; and
- source-gene and phase-cell representation.

Use only sequence and annotation to construct weights or matched blocks. Freeze a 120-nt primary stratum
for direct continuity with `C43`. If resources permit, reserve 90-nt and 150-nt strata as a corrected
secondary geometry family; do not pool them silently with the primary endpoint.

**Authorization gates:**

1. every prespecified scalar has maximum absolute standardized mean difference `≤ 0.10`;
2. a nested, cross-fitted classifier trained on the complete frozen feature family has phase-match AUC with
   a 95% upper bound `≤ 0.55` after weighting or matching;
3. every 3 × 3 cell and every source stratum meets the frozen minimum count;
4. a planted imbalance is detected and a permuted label is not;
5. sample size is frozen from the sign-free upper-variance projection in `C43`; target about 600 blocks so
   the projected `MDE80` is `≤ 0.04`, but do not call that projection an observed capability gate; and
6. identifiers, sequences, annotations, matching weights, and split hashes are sealed before model output.

If gates 1–4 fail, there is no composition-controlled test. If gate 5 fails, increase the planned cohort
before opening outcomes. The observed development-only `MDE80` is computed in Phase II. A target run that
misses `0.05` cannot license a material bounded null; the stricter `0.04` planning target prevents another
decision at the Monte-Carlo boundary. Never relabel either failure as evidence against PBC-01.

#### I-B. Test the ≈10.63 origin fork in sequence

Use the actual 72-species training transcripts if they can be recovered and hashed. Otherwise label the
result as a proxy-corpus assay; it cannot establish training-data provenance.

Analyze windows independently and aggregate by transcript, gene, and species. Do not concatenate sequences
and create artificial phase resets. Primary helix labels must come from structure annotations independent
of the nucleotide spectrum; sequence-predicted labels are sensitivity analyses only. Compare:

- CDS against matched UTR, intron, and noncoding transcript sequence;
- structure-annotated α-helical coding regions against matched β-sheet and coil regions, with
  sequence-predicted labels used only for sensitivity;
- multispecies against token-count-, length-, GC-, and transcript-class-matched human sequence;
- native CDS against amino-acid-order shuffles, codon-order shuffles, and synonymous recodings; and
- frozen tests at period 3, ≈10.63, 10.8, and `32/3`, plus a corrected 2.8–3.2 and 9.5–11.5 scan.

Calibrate each sequence statistic with planted tones spanning the target effect scale. Correct across the
full wavelength, indicator, sequence-class, and control family. A signal near 10.6 is not called
α-helical unless it is selectively enriched in the helix stratum and survives held-out species. Freeze the
expected result of each control: amino-acid- and codon-order shuffles disrupt protein order, whereas
synonymous recoding preserves amino-acid order and tests whether the nucleotide realization is necessary.
Before distinguishing ≈10.63 from `32/3`, simulation at the observed sequence lengths and covariance must
show at least 80% power at the corrected alpha. Otherwise report an unresolved 10.6-family wavelength.

**Phase-I decisions:**

| Result | Decision |
|---|---|
| balanced cohort cannot be built | instrument failure; redesign and stop |
| hashed original-corpus signal passes helix, held-out-species, sensitivity, and wavelength-compatibility gates | advance PBC-03 as a candidate training-data referent |
| proxy-corpus signal passes | report a conserved biological analogue; do not claim training provenance |
| calibrated original-corpus CDS equivalence/null | demote the tested CDS/helix referent; the ruler remains open but unsupported |
| uncalibrated or proxy-only null | unresolved; do not change PBC-03 |

Only I-A authorizes Phase II. I-B is an independent origin branch; its failure does not prevent a
well-powered carrier-behavior test.

### Phase II — fresh behavior and the period-3-first P3 × P10 factorial

Run this phase on base `SpliceBERT.1024nt`, because `C43` is a base-checkpoint behavior. `C37` supplies a
same-checkpoint ≈10.63 capability prior; do not import the stage-1 head list or the GTEx intervention dose.

Freeze the attenuation grid `{0.25, 0.50, 0.75, 1.00}` separately for each band, where the value is the
fraction of nominated carrier coefficients removed. On development genes, choose the **largest** dose that
passes every carrier-capability and health threshold below; never use `Ψ`, annotated-phase labels, or a
target-direction statistic for dose selection. Seal the doses before the fresh target. P3 and ≈10.63 may
select different doses, so their effect magnitudes cannot be compared directly. The primary target has four
atomic arms:

| Arm | period-3 | ≈10.63 |
|---|---|---|
| `Y00` | intact | intact |
| `Y10` | attenuated | intact |
| `Y01` | intact | attenuated |
| `Y11` | attenuated | attenuated |

Every arm uses identical nucleotide sequences, and each table intervention must be byte-identical in the
intact and `AG→AC` states. Add exact original-band replay as an identity control, not as behavioral rescue.
Use separate null families for period-3, ≈10.63, and the joint edit. Nulls must match band width, changed RMS
energy, and dose; never pool nulls of different intervention size. For each single-band intervention use
exactly 31 same-band equal-RMS phase controls and exactly 31 hidden-axis controls. Use exactly 31 joint
control pairs matched to the combined support and per-band doses.

Retain the exact `C43` readout and primary operator:

`f = logit(G) − mean(logit(A), logit(C), logit(T))`, with coupling defined by true-acceptor `AG→AC`
corruption, and `Ψ = M0 − (M+ + M−)/2`. Keep `AG→AT` as the frozen operator-sensitivity arm.

**Preflight gates, before target exposure:**

- identity and reconstructability pass at frozen numerical tolerance;
- raw-table replay is exact, off-band leakage is `≤ 1e−10`, and target-band dose passes;
- the period-3 edit preserves ≈10.63, the ≈10.63 edit preserves period-3, and the dual edit is identical in
  both composition orders;
- period-3 attenuation changes the established period-3 coupling positive control by at least `0.05`, its
  95% interval is above zero, it exceeds all 31 period-3 capability controls, and the ≈10.63 reader score
  retains at least 80% of intact magnitude;
- ≈10.63 attenuation reduces the frozen aggregate carrier-phase attention score across the nine `C37`
  readers by at least 50%, its 95% lower bound exceeds 25%, and it exceeds all 31 ≈10.63 capability
  controls, while the period-3 coupling positive retains at least 80% of intact magnitude;
- the native-source positive, phase-blind crossed positive, bridge negative, `AC/AT` invariance,
  protected-base health, determinism, and construction audit reproduce independently of target `Ψ`;
- native-source `AG→AC` coupling lower bound is above `+0.05`, phase-blind crossed coupling lower bound is
  above `+0.02`, and the bridge interval is inside `[-0.05, +0.05]`;
- phase-blind `AC−AT` lies inside `[-0.05, +0.05]`, protected true-base log-probability lower bound is above
  `−0.10` nat, masked-base accuracy retention is at least `0.80`, and repeat-forward error is `≤ 1e−8`;
- intact donor and acceptor masked-margin lower bounds exceed floors frozen from fresh native-source windows;
- same-band controls pass the same mechanical, carrier-specificity, and health thresholds; and
- observed development-only `MDE80 ≤ 0.04` for clean `Ψ` and `N3`, and `MDE80 ≤ 0.05` for the
  optional `N10` and `J` endpoints.

A failed capability or health gate returns **INSTRUMENT_UNRESOLVED**. It is not a carrier null.

Let `Yab` be the composition-balanced phase preference `Ψ` in each arm. Freeze:

- `N3 = Y00 − Y10`;
- `N10 = Y00 − Y01`;
- `Nboth = Y00 − Y11`; and
- `J = Y11 − Y10 − Y01 + Y00`.

The subscripts follow the table: `1` means attenuation and `0` means intact.

The materiality SESOI is `0.05` raw-logit units for `N3`, `N10`, and `J`. A supportive single-band effect
has a 95% lower bound above `+0.05`; a bounded single-band null has its full interval inside
`[-0.05, +0.05]`. For this arm coding, `J < −0.05` is super-additive attenuation damage and `J > +0.05`
is sub-additive damage. Freeze which sign advances the biological hypothesis before target exposure;
otherwise report the sign without naming it synergy, complementarity, or redundancy. `Nboth` is descriptive
unless it is explicitly added to the full confirmatory correction family before the target opens.

`J = 0` is additivity on the raw-logit `Ψ` scale. Interpret redundancy or synergy only after both single
effects pass and only in the preregistered direction. Keep `Ψ` as the primary behavior; do not replace it
with a ratio. Use exactly 20,000 chromosome-by-strand whole-block bootstraps, simultaneous max-T bounds
for the directional endpoints, a separate simultaneous mismatch-equivalence family, and exactly 999
balanced pseudo-phase controls. Use exact add-one permutation p-values and correct over the complete target
and structured-null family. Simultaneous target inference may span `N3`, `N10`, and `J`, but specificity
tests remain three separate P3, P10, and joint structured-null families because their intervention sizes
differ. Two positive main effects do not establish carrier coupling; that requires a material corrected
`J`.

Phase I cohort construction and Phase II outcomes can be sealed in one atomic package, but then they are one
experiment. Do not call the clean `Y00` result an independent replication of the factorial effects.

**Phase-II decisions:**

| Clean balanced `Ψ` | `N3` | `N10` | `J` | Licensed result |
|---|---|---|---|---|
| capable equivalence/null | any | any | any | composition-associated account supported within the matched family; there is no positive clean preference to mediate, and edit-induced contrasts are reported separately |
| material positive | bounded/null | bounded/null | equivalent/null | no material dependence on the two tested bands is detected under these edits and doses |
| directional corrected positive | directional corrected positive | any | any | if one or both primary intervals do not establish `>0.05` materiality: rough, assay-scoped P3–boundary-preference link; no materiality language and no Phase III |
| material positive | material positive | bounded/null | equivalent/null | period-3 bridge advances; keep ≈10.63 separate |
| material positive | bounded/null | material positive | equivalent/null | ≈10.63 bridge advances; do not infer a three-state register |
| material positive | material positive | material positive | equivalent/null | two parallel supports; do not call them coupled |
| material positive | material positive | material positive | material, corrected | dual-scale interaction; authorize joint mediation |

The first row is scientifically useful closure. It says that `C43` detected a real annotated-phase
preference in its original construct family, while the new measured feature family accounts for that
contrast. It does not retract `C43`.
Negative `N3` or `N10` means attenuation increases the preference and is a suppressive carrier result, not
supportive use. A material `J` with one or both single-band effects bounded/null is a context-dependent
interaction, not joint carrier support. The sign of `J` follows the frozen damage map above.

### Phase III — optional period-3 register strengthening and route-specific rescue

Reserve a third, outcome-fresh gene split at Phase-I freeze. Audited directional Phase-II support closes the
minimum rough link, but **only material `Ψ00` and material corrected `N3` open the optional P3-only
Phase-III path.** Opening does not depend on `N10`, `J`, or resolution of ≈10.63's biological origin.
Discovery may use development data. All component identities, signs, windows, patch sites, controls, and
output paths are frozen before the reserve opens.

Use a new full-span analytic intervention on the complete learned P3 band, never the stopped `C44` local
operator. Across nucleotide span `[392,512)`, impose register `r∈{-1,0,+1}` with a phase ramp from zero at
the donor to `−2πr/3` at the acceptor. This sign convention predicts annotated phase difference
`δ=(p_A-p_D) mod 3 = -r mod 3`. The independent truth table is `r=-1→δ=+1`, `r=0→δ=0`, and
`r=+1→δ=-1`; an opposite-handed material ridge is a falsification, not a label swap.

For every register execute the learned-P3 target, 31 same-RMS signed hidden-axis controls, a same-RMS
common-mode rotation, and a same-RMS upstream decoy-span ramp. Require exact `r=0` replay, unchanged
nucleotides and non-P3 residual, P10 retention, global health, target superiority to the full hidden-axis
family, positive paired target-minus-common and target-minus-decoy intervals, and failure of both named
controls to reproduce all three register contrasts.

For biological phase difference `δ`, imposed period-3 register `r`, ≈10.63 state `z`, and frozen phase-grid
coupling endpoint `G`, preregister:

- `S_r(z) = G(δ = −r, r, z) − mean_{δ ≠ −r} G(δ, r, z)`;
- `Ω3(z) = mean_r S_r(z)`; and
- `Jreg = Ω3(P10 intact) − Ω3(P10 attenuated)`.

The full relocation claim requires every `S_r` simultaneous lower bound above zero; a positive mean is
insufficient for that claim. A weaker preregistered strengthening tier is available when `Ω3` is materially
positive, all three `S_r` point estimates have the predicted sign, and at least two of the three simultaneous
lower bounds exceed zero. It additionally requires full health/audit passage, superiority to the 31
hidden-axis controls, a corrected partial-only family jointly separating target from common-mode and decoy
while resolving handedness, and neither named control reproducing the partial rule. Define `Ωopp` by
replacing the frozen preferred relation `δ=−r` with `δ=+r` in each register contrast, and
`H=Ω3−Ωopp`; the simultaneous lower bound for `H` and the two target-minus-control bounds must exceed zero
in that family, and the opposite orientation must not satisfy the same partial rule. That outcome,
`P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED`, is partial register concordance, not complete cyclic relocation
and not authority for route mediation. An opposite-only raw partial pattern is
`P3_REGISTER_PARTIAL_HANDEDNESS_FALSIFIED` only if its own hidden-axis and paired-control specificity pass
and a separately corrected family jointly resolves opposite-minus-expected handedness plus separation from
the common-mode and decoy controls. Expected and opposite raw partial patterns together, or either one with
unresolved handedness, are
`P3_REGISTER_PARTIAL_HANDEDNESS_AMBIGUOUS`. `Jreg` tests whether ≈10.63 changes the period-3 register effect.
It does not by itself establish a directed arrow.

On development, screen all 96 heads and all six MLP outputs with same-corruption-state, full-span output
scrubs. Freeze the smallest jointly remeasured descending prefix reaching 80% of development `Ω3`, capped
at eight components; if eight do not reach 80%, name it a distributed candidate set. Match 31 null sets on
component count, layer counts, head/MLP type, patched-token count, and output-delta norm.

On reserve, measure raw relocation, same-state path scrub, restoration into the P3-attenuated host, and a
separate sufficiency injection. Require simultaneous material lower bounds for absolute path necessity,
restoration, and sufficiency, plus exact add-one superiority and paired separation from all 31 matched
component sets. QK and attention measurements may localize a selector or router, but only controlled output
scrub/restoration licenses mediation. Report absolute logit effects, not a mediated percentage.

If ≈10.63 also survives Phase II with a corrected material `J`, an optional P10-attenuated register arm may
estimate `Jreg` and its path effects in a separately corrected family. That arm may remain unresolved
without weakening P3 relocation or mediation. Do not assume L3H1, and do not use GTEx local-margin effects
as mediation for the base checkpoint.

A directed `period-3 → ≈10.63` statement requires asymmetric evidence: period-3 attenuation changes a later
≈10.63-specific reader state, restoring that state rescues `J`, and the reciprocal route is bounded. Without
all three conditions, write **joint use**, never an arrow.

---

## 4. Execution order

| Gate | Work | Stop condition |
|---|---|---|
| Freeze | seal hypotheses, splits, feature family, test family, thresholds, commands, and manifests | any data or split provenance ambiguity |
| Phase I | construct and independently audit the fresh composition-balanced cohort | balance, sensitivity, provenance, or calibration failure; the optional ≈10.63 origin branch never blocks P3 |
| Phase II development | select P3 dose and pass capability, health, specificity, power, and false-positive gates | any mandatory P3 gate fails |
| Phase II target | open the fresh target once and audit the clean preference plus `N3`; corrected directional support closes the minimum rough link, while material support alone opens Phase III | no Phase III unless clean preference and P3 necessity are material |
| Phase III development | validate handedness, relocation controls, route discovery, matched component sets, and reserve power | any register or path instrument gate fails |
| Phase III reserve | optionally test partial/full P3 register concordance and, after full relocation, necessity, restoration, and separate sufficiency | no full relocation without three resolved orientations; no mediation wording without all frozen route gates |

The minimum positive paper tier is a corrected directional Phase-II P3 attenuation link. A material
Phase-II bridge, partial register concordance, complete three-register relocation, and path mediation are
successively stronger rungs. The ≈10.63 origin and optional modifier arm may remain unresolved.

Implementation belongs in a new study and script tree, created from
[`studies/TEMPLATE_REAL_GFM_STUDY`](../TEMPLATE_REAL_GFM_STUDY/README.md). The archived exon-edge runners and
the unexecuted superseded codon/phase program must not be restored. The new study may cite `C43` and `C44`
as provenance and consume their compact recorded artifacts, but it must own its runners, preregistration,
manifests, audits, and claim rows.

---

## 5. Paper narrative by terminal outcome

### Most probable clean narratives

If composition balance removes the `C43` contrast:

> SpliceBERT contains two training-regime-associated positional bands that are reused differently.
> Period-3 is a causal relative-routing coordinate in base-model partner selection, and period-3 and
> ≈10.63 support local splice-site evidence after fine-tuning. Separately, the original chimeric
> boundary-phase preference is consistent with, and accounted for within, the prespecified measured family
> of phase-correlated sequence features. This does not imply a frame sensor.

If the balanced contrast persists but both carrier effects are bounded:

> The base model retains an annotated phase-linked preference after prespecified sequence balance, but no
> material dependence on the period-3 or ≈10.63 bands is detected under the tested edits and doses. The
> responsible sequence route remains unlocalized.

### Primary paper claim at the directional Phase-II floor

> **In a fresh composition-balanced fixed-span chimeric assay at `SpliceBERT.1024nt`, health-preserving
> attenuation of the learned period-3 band shifts the model's annotated boundary-phase preference in the
> prespecified direction relative to both intervention-matched control families. The effect is statistically
> resolved but is not shown to exceed the `0.05` materiality threshold; this is a rough, assay-scoped causal
> link, not material mechanism closure.**

### Stronger paper claim if Phase II is material

> **In a fresh composition-balanced fixed-span chimeric assay at `SpliceBERT.1024nt`, health-preserving
> attenuation of the learned period-3 band materially reduces the model's annotated boundary-phase
> preference relative to intervention-matched controls. Period-3 therefore causally supports this scoped
> sequence-encoded boundary-coherence behavior.**

If Phase III additionally finds a material aggregate response, the predicted sign in all three registers,
simultaneous support in at least two, full specificity/health/audit passage, no reproduction by common-mode
or decoy controls, a positive expected-minus-opposite handedness bound, and no opposite partial pattern, add:

> *A full-span analytic period-3 register intervention is directionally concordant across all three
> prespecified register relations, with at least two individually resolved. This partial concordance
> strengthens the rough, assay-scoped P3–boundary-coherence link, while complete cyclic relocation and its
> internal mediation remain unresolved.*

### Strongest claim if full relocation and mediation also pass

> **In a fresh composition-balanced fixed-span chimeric assay at `SpliceBERT.1024nt`, health-preserving
> attenuation of the learned period-3 band materially reduces the model's annotated boundary-phase
> preference. A full-span analytic period-3 register intervention then cyclically relocates the preferred
> boundary relation, and a frozen internal path is necessary, rescuable, and separately sufficient for a
> scoped portion of that relocation. SpliceBERT therefore uses a learned relative period-3 coordinate to
> evaluate a sequence-encoded coding-boundary coherence prior in this assay.**

If the optional P10 factorial, `Jreg`, and joint path family also pass, add only:

> *Attenuating the separate ≈10.63 carrier changes the period-3 register effect through a corrected
> non-additive interaction in this assay.*

The primary claim does not require a causal P10 effect or a biological explanation for ≈10.63.

If Phase I-B also identifies a wavelength-compatible, held-out-species, helix-enriched ≈10.6–10.8-nt CDS
signal in the hashed original 72-species training corpus, add one separate sentence:

> *The ≈10.63 carrier has a candidate training-data referent in conserved coding-sequence periodicity
> associated with α-helical protein structure.*

A proxy-corpus positive licenses only: *A conserved coding-sequence analogue is a candidate biological
source for ≈10.63.*

Even this maximal outcome does **not** license “SpliceBERT reads the reading frame,” “the model implements
translation,” “the model implements U2AF or the spliceosome,” or “the spliceosome selects sites by phase.”
The endpoint remains a masked-donor logit response under acceptor corruption in synthetic chimeric
constructs.

### Valuable asymmetric outcomes

- **Period-3 only:** the strongest mechanistic result. Write that a causal relative carrier supports the
  composition-controlled phase-linked preference. Keep ≈10.63 as a separate local-classification carrier.
- **≈10.63 only:** report causal support for this base-checkpoint phase-linked behavior, but do not call it
  a phase register or generalize it to ordinary coupling.
- **Both, `J` bounded inside ±0.05:** two separable supports. Do not write “integrated code.”
- **Both, material corrected `J`:** a dual-scale interaction is supported. Report its sign with the frozen
  damage map; do not choose “complementary” or “redundant” after seeing the data.
- **Instrument failure:** preserve it as unresolved. Do not spend the target or simplify the narrative.

---

## 6. Primary biological anchors

- Eskesen et al., “Periodicity of DNA in exons,” *BMC Molecular Biology* 5, 12 (2004),
  <https://doi.org/10.1186/1471-2199-5-12>. Unequal codon usage can reproduce exon period-3; equal codon
  frequencies remove it.
- Herzel, Weiss, and Trifonov, “10–11 bp periodicities in complete genomes reflect protein structure and
  DNA folding,” *Bioinformatics* 15, 187–193 (1999),
  <https://doi.org/10.1093/bioinformatics/15.3.187>. The authors attribute short-range correlations mainly
  to protein/amphipathic-helix structure and longer-range correlations mainly to DNA folding.
- Whamond and Thornton, “An analysis of intron positions in relation to nucleotides, amino acids, and
  protein secondary structure,” *Journal of Molecular Biology* 359, 238–247 (2006),
  <https://doi.org/10.1016/j.jmb.2006.03.029>. It supplies the main biological reason to treat local
  splice-boundary nucleotide bias as the leading phase rival.
- Rhodes and Klug, “Sequence-dependent helical periodicity of DNA,” *Nature* 292, 378–380 (1981),
  <https://doi.org/10.1038/292378a0>. Their random-sequence DNA estimate of 10.6 ± 0.1 bp exposes a range
  conflict in the canonical prose. This register does not use it to reopen numerical identity to the
  distinguishable ≈10.298 mouse ES-cell dyad signal.

---

## 7. Manuscript language guard

Until the sequence completes, the one safe synthesis is:

> **SpliceBERT contains a causal period-3 relative-position circuit, a distinct ≈10.63 positional carrier
> with demonstrated local-classification use after fine-tuning, and a behavioral preference for matched
> annotated coding-boundary phase in chimeric constructs. The causal relationships among those observations
> remain open.**
