# DRAFT — sourced branchpoint × operational pyrimidine-rich interval factorial

> **Tier: T2 — draft.** Prospective instrument specification; not a claim or result.
>
> **Status: DRAFT — NOT FROZEN.** Smoke and discovery preflight only. Target execution requires an
> annotation/cohort audit and a reviewed frozen successor. No conforming builder, runner, registered
> source audit, or model artifact is included in this tree. All implementation language below is
> normative: a conforming successor **must** meet it before target execution is possible.

## Question

Does the frozen SpliceBERT single-`G` acceptor reconstruction depend causally on sequence around a
sourced branchpoint (BP) and in a prespecified sequence-derived pyrimidine-rich interval, relative to
transition-matched controls, and is their joint effect additive, non-additive, or unresolved?

This is a model-behavior question motivated by biological annotations. It is not a claim about molecular
splicing causation.

## Population and endpoint

- Official `SpliceBERT.1024nt`; exact checkpoint/config/tokenizer hashes recorded.
- Native GENCODE-v50-**basic**/GRCh38 windows with an annotated acceptor `AG` and one source-bound branchpoint `A`.
- Primary BP cohort: Mercer Data S3 high-confidence BP-to-GENCODE-v12-intron associations. The source
  preparation retains exactly one source-declared `A` per genomic intron, uniquely lifts both intron
  endpoints and the BP hg19→GRCh38, and exact-joins the same-strand intron to one GENCODE-v50 gene.
  Multi-transcript introns retain all compatible protein-coding transcripts represented in the
  GENCODE v50 **basic** GTF rather than an arbitrary representative; transcripts present only in the
  comprehensive GTF are outside the estimand. The downstream builder independently revalidates the
  GRCh38 reference `A` and uniquely associates the row to its acceptor. Pineda and predicted
  Branchpointer cohorts are separate sensitivity families and cannot promote the primary source claim.
- A `Mercer` label is not sufficient provenance. The cohort builder requires the prepared BED's exact
  sibling `gfmcd.mercer_hc_branchpoint_source_audit/1` and preparation manifest, rechecks their
  outcome-blind status, source/assembly/GTF contract and invariant set, and binds both sidecars as
  cohort-manifest inputs. The preparation manifest must in turn bind the exact Mercer Data S3,
  liftOver-chain, and current-GTF input records. The production runner reopens the prepared BED/audit/
  manifest byte for byte and verifies that their recorded raw-input chain agrees. Source preparation
  and cohort manifests also bind the transformation scripts, shared source-contract/core code, GTF
  parser, FASTA adapter, provenance and overwrite guard, `pyproject.toml`, `uv.lock`, and recorded
  Python/platform/package runtime identity. Each constructor snapshots these bytes before parsing,
  serializes its complete data/audit/manifest sibling package, rehashes the snapshot immediately
  before publication, and publishes only with create-only atomic writes. Manifest output records are
  computed from the serialized bytes, not from later live-file reads.
- The primary production bytes are fixed, not operator-supplied labels: decompressed Mercer Data S3
  SHA-256 `ad2dfae5a099a671810273a06246b394de461020bc6e0828515dad500723adf0`
  (13,130,260 bytes; 177,113 rows); UCSC `hg19ToHg38.over.chain.gz` SHA-256
  `5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d`; compressed GENCODE v50
  basic GTF SHA-256 `5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7`;
  and the repository's local GRCh38 `hg38.fa` identity SHA-256
  `5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51`.
  The proposed source-preparation environment pins `pyliftover==0.4.1`. A byte, annotation-set, transcript-type,
  dependency, or runtime-version mismatch stops construction.
- No curated PPT annotation is asserted. Primary `opPPT80_v1` chooses the longest interval of at least
  10 nt between BP and terminal `AG` with C/T at both ends and C/T fraction at least 0.80; ties resolve by
  C/T fraction, then 3'-most end, then 3'-most start. Missing intervals remain missing.
- Gene-disjoint `discovery` and `evaluation` splits fixed in the cohort file.
- The primary production family accepts exactly one Mercer-labeled experimental BP source and
  `opPPT80_v1` operational intervals. Pineda, predicted BP, input-PPT, and Branchpointer-PPT rows
  require separate sensitivity runs and cannot be mixed into or promote the Mercer primary family.
- `chr8` and `chr9` are excluded before sequence construction and cannot enter either split; this
  study does not spend the donor study's sealed chromosomes.
- Smoke uses 16 rows. Production preflight and target are fixed at 200 discovery rows and 500
  evaluation rows, respectively, with edit count 3, batch size 32, seed 0, 2,000 inference draws,
  200 calibration trials, and four health probes. These are not command-line degrees of freedom.
- Primary: acceptor `G` masked alone, `G` versus best `A/C/T` logit margin.
- Secondary bridge family: acceptor `A` masked alone; sum of separately masked A and G margins; Sri-style
  simultaneous `[MASK][MASK]` A+G margin. Secondary endpoints cannot promote the primary claim.

## Constructs

For edit dose `k`:

1. clean;
2. BP disrupted with exactly `k` substitutions, including the annotated branchpoint `A`;
3. operational pyrimidine-rich interval disrupted with exactly `k` `C/T→A/G` substitutions;
4. both factors disrupted, exactly `2k` substitutions;
5. BP control, exactly the BP arm's transition signature in a distinct equal-width interval;
6. PPT control, exactly the PPT arm's transition signature in a distinct equal-width interval;
7. combined control, exactly `2k` substitutions from those two controls;
8. instrument positive control, destruction of the terminal acceptor `A` while the `G` is masked.

Every PPT edit is `C/T→A/G`; “prioritize” is insufficient. All lengths and coordinates are fixed. A
construct fails if its factor/control widths or transition signatures differ, it changes an outside
base, or it creates a new `AG`, `GT`, or `GC`. Failure stops the run; examples are not silently dropped.
The positive control calibrates the model/construct/metric path and never enters the target interpretation.

Control locations are selected outcome-blind under
`joint_transition_capable_bounded_nearest_v3`. Each candidate has factor-matched width, enough source
bases to realize the exact edit-transition signature, no overlap with either factor or the other
control, full-window A/C/G/T composition L1 distance at most 6, and absolute factor-versus-control
acceptor-distance difference at most 100 nt. The deterministic joint objective minimizes combined
composition mismatch, then C/T mismatch, then distance mismatch, then coordinates. Rows matching
full A/C/G/T counts exactly form a prespecified sensitivity subset. Primary controls are therefore
**bounded nearest transition-matched controls**, not claimed exchangeable sequence windows; their
measured damage must pass the equivalence gate below.

## Estimands

For margin cells `M_clean`, `M_BP`, `M_PPT`, `M_both` and their matched controls:

- BP damage: `M_clean - M_BP`;
- PPT damage: `M_clean - M_PPT`;
- joint damage: `M_clean - M_both`;
- damage interaction: `joint - BP - PPT`;
- BP/PPT/joint control damage and the corresponding control interaction, defined analogously;
- primary BP/PPT/joint specificity: factor damage minus its matched-control damage;
- interaction specificity: factor interaction minus the combined-control interaction.

The restoration contrasts (`M_PPT-M_both`, `M_BP-M_both`, `M_clean-M_both`) are reported as the same
factorial viewed from the doubly disrupted state. They do not count as independent sufficiency evidence.
A later cross-gene or synthetic-background transplant is required for that claim.

## Statistics

- Unit: gene; examples from one gene are never independently permuted or bootstrapped.
- Production inference requires at least 30 independent genes.
- One simultaneous 95% whole-gene cluster-bootstrap endpoint-scale-normalized
  maximum-absolute-deviation family over the full primary single-G contrasts. For endpoint `j`, the
  scale is the standard deviation of its gene-cluster bootstrap estimates; the family critical value
  is the 95th percentile of the maximum absolute scale-normalized bootstrap deviation and the interval
  is `estimate_j ± critical × scale_j`. The implementation retains the historical `studentized=True`
  flag for this calculation, but this is not a nested bootstrap-`t` procedure.
- Exact add-one Monte-Carlo p-values use one sign per gene on raw cluster sums and the full-family
  maximum absolute statistic.
- SESOI: `0.05` logit-margin units.
- Secondary bridge endpoints form one separately corrected family and cannot alter the primary decision.
- Positive material effect: simultaneous lower edge `>= +0.05`; negative material effect: upper edge
  `<= -0.05`; equivalence to zero: simultaneous interval inside `[-0.05,+0.05]`; otherwise unresolved.

## Preflight gates on discovery genes

1. sidecar-bound schema, assembly, canonical source-byte pins, basic-GTF estimand, closed transformation
   input manifests, runtime identity, genomic→transcript coordinates, gene splits, absence of chr8/9,
   recorded control calipers, and every construct invariant pass;
2. clean acceptor-G mean margin `> +0.50`;
3. the terminal-acceptor positive control is materially detected through the production family;
4. non-site top-1 health is at least 0.50 under every construct and every construct's health damage is
   equivalent within `±0.10` under simultaneous inference;
5. BP, PPT, combined matched-control damage, and the matched-control interaction are each equivalent
   within `±0.05`;
6. row-level gene-sign-flipped nulls are passed through the exact endpoint-scale-normalized production
   classifier.
   A false positive requires one complete declared factor path: raw BP, PPT, or joint damage and its
   matched-control specificity are both material in the same direction, while BP/PPT/joint control
   damages and the control interaction are equivalent. Positive and negative paths are both counted.
   The two interaction-only declarations are also counted: raw and control-adjusted interactions must
   both be material in the same positive or negative direction while all four control terms are
   equivalent. There are therefore eight declared null paths in total.
   The live acceptor anchor remains a separate observed gate; it is not sign-flipped into a null and
   then required to remain positive;
7. the 13-row structured plant is algebraically additive: raw BP/PPT damages and their specificity
   terms receive `+3×SESOI`; raw joint damage and joint specificity receive `+6×SESOI`; raw and
   control-adjusted interactions plus all four control terms remain zero; and the acceptor anchor
   receives `+3×SESOI`. All 13 required material/equivalence labels must be recovered in one trial.
   The one-sided 95% Clopper--Pearson FPR upper bound is `<=0.05` and the conjunctive power lower bound
   is `>=0.90`;
8. full checkpoint bundle, cohort/audit/manifest, protocol, seal-normalized runner, construct and
   statistics cores, imported measurement dependencies, runtime/library identity, family, fixed
   production parameters, and preflight artifact/manifest hashes are bound. Target additionally
   requires a literal reviewed instrument-fingerprint pin, a preflight produced only after both
   protocol and instrument pins were installed, and a clean worktree. It validates the exact gate
   key set, all gate values, `PREFLIGHT_GO`, family sizes, fixed draws/trials, scale-normalized classifier,
   structured plant, null declaration paths, and recorded Clopper--Pearson thresholds rather than
   trusting summary booleans.

Before any production row is scored, the runner rejects unknown JSON fields and independently validates
every runner-facing coordinate transform, genomic window, intron-containment relation, `opPPT80_v1`
interval, stable gene split, globally optimal bounded-control pair, frozen-seed construct, and construct
audit from the exact cohort row and audited construction contract. It reopens and byte/semantically
validates the complete Mercer preparation package, but does not claim to rerun the full liftOver,
GENCODE join, or FASTA extraction inside the outcome runner. It accepts neither a self-reported row
audit nor a locally plausible control pair as sufficient. The complete cohort-manifest input and output
sets must equal the independently reconstructed source/transformation closure; missing, duplicate, or
extra records fail.

All run-defining inputs are byte-snapshotted before semantic use and rechecked before publication.
The checkpoint is copied from the verified complete directory inventory into a private staged bundle,
and only that bundle is loaded. For target execution, an output-name-independent, immutable attempt
record keyed by the stable experiment holdout namespace and cohort identity is durably published after
authorization and byte-only input verification but before evaluation rows are semantically parsed or the
model is loaded. The record binds the authorized protocol and instrument inside its payload, while its
path deliberately remains independent of those revision identities. It conservatively marks the target
outcomes as possibly spent even if execution later fails. Result and manifest publication use
create-only atomic writes.

The maximum simultaneous half-width is recorded only as a diagnostic. There is no global width gate:
each scientific conclusion is determined by its own simultaneous classification, while the exact
null/plant trials calibrate the complete decision path. Preflight stores its primary, secondary,
accounting, health, and per-row sufficient values so every gate remains auditable.

An anomalous matched control or health arm freezes interpretation. It is not removed or replaced after
outcome inspection. The draft hard-rejects chr8/9; no donor-sealed row may be touched before a frozen
successor. Discovery preflight parses the complete outcome-blind cohort file, including evaluation-row
annotations, to verify provenance and splits. It does **not** run the model on evaluation rows. Thus the
evaluation annotations are not claimed unopened; only evaluation model outcomes remain held out.

## Freeze and target authorization

The exact acknowledgement required after protocol review and both hash pins are installed is:
`I_ACKNOWLEDGE_BP_PPT_WILL_OPEN_HELD_OUT_EVALUATION_OUTCOMES`. The target remains unreachable without
that exact string, the bound passing post-freeze preflight, fresh output paths, a clean worktree, and an
unused canonical target-attempt key. Changing `--out`, protocol bytes, or instrument revision cannot
create a second allegedly unseen attempt on the same cohort.

## Prespecified robustness envelope

The exact-full-composition control subset, Pineda experimental BP source, predicted Branchpointer BP
source, input-PPT coordinates, and pinned Branchpointer-`getPPT` operational intervals are distinct
non-promoting sensitivity families. None has been run by this correction. A future runner/artifact must
freeze each cohort, family, and multiplicity policy before reading its model outcomes. Until then the
primary scope is only Mercer-prepared BP rows with `opPPT80_v1` and bounded-nearest controls.

## Outcome language

- `BOUNDED_FACTOR_SENSITIVITY` requires at least one BP, PPT, or joint **raw damage and matched-control
  specificity pair** to be materially positive after every control, health, anchor, construction, and
  bound-preflight gate. A materially negative raw+specific pair is reported as reversed sensitivity;
  resolved factor pairs pointing in opposite directions receive the mixed-direction label. A raw and
  specificity term that disagree within the same factor do not form a resolved pair and remain
  unresolved.
- `ADDITIVE_BP_AND_OPERATIONAL_PPT_SENSITIVITY` requires all three positive raw+specific pairs and both
  the raw and control-adjusted interaction to be equivalent. `NONADDITIVE...` requires those three
  positive paths plus raw and adjusted interactions material in the same direction. Interaction-only
  materiality is reported separately and does not masquerade as resolved main effects.
- A bounded null requires raw BP, PPT, and joint damage, all three corresponding specificity terms,
  and both interaction terms to be equivalent within the SESOI; it is a null for this
  dose/window/endpoint, not biological absence.
- Secondary-only result: double-mask-policy dependence; no native single-token behavior claim.
- No result licenses a curated-PPT claim, molecular mechanism, splice usage, or internal model mediator.
