# FROZEN R1c successor — acceptor-only paired-donor behavior confirmation

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

> **Status: FROZEN FOR PREFLIGHT ONLY (2026-08-17); CHR8/9 TARGET SEALED.** The reviewed protocol,
> normalized instrument, and complete downstream-R5-stack identities are pinned in source. The
> historical `50cda…` preflight remains engineering history and cannot authorize this successor.
> `FROZEN_PREFLIGHT_ARTIFACT_SHA256` remains `None`; target execution therefore remains impossible
> until a new passing successor preflight is independently reviewed and pinned and the explicit
> chr8/9 acknowledgement is supplied.

## 0. Question and evidence boundary

In the frozen `SpliceBERT.1024nt` masked language model, does destroying an intron's paired donor
reduce the native single-`G` margin at the downstream intron-mate acceptor more than destroying a
distance-matched non-partner donor? On outcome-blind distance-matched exon/intron dyads, is that
paired-donor selectivity larger downstream than at the upstream exon-mate acceptor?

R1c is a **behavior-rung** successor to R1/R1b. It corrects their row-wise permutation defect and is
acceptor-only: both readout levels are masked acceptor `G`s, and both edit levels are the same
`GT -> AC` donor intervention at either the real paired donor (`D`) or a matched non-partner donor
(`N`). There is no masked-donor forward anchor. It does not identify a component, address, payload,
write, delivery path, transported
semantic, generative mechanism, or biological mechanism. A successful endpoint-outcome-held-out result would license
only a bounded statement about a native masked-token behavior in this frozen checkpoint and would
motivate, but not establish, the separate mechanistic follow-ups.

**Holdout scope disclosure.** Earlier unrelated development tooling materialized chr8/9 sequence
records before filtering them away. No known earlier run scored the masked-acceptor endpoint on the
exact R1c target rows, but the chromosome split is not globally sequence-unseen. Throughout this
protocol, “sealed” means the exact target's masked-acceptor endpoint outcomes remain unopened until
the create-only attempt receipt; it must never be expanded to “globally unseen,” “sequence unseen,” or
“virgin chromosome split.” Every artifact serializes this distinction as part of the frozen method.

The earlier v3 construction / `gfmcd.r1c.reverse_behavior/2` development artifact remains immutable
historical development evidence. This v4 construction / schema 3 instrument is **unrun** at this
source/method freeze and cannot reinterpret, overwrite, or use the v3 bytes as target-binding preflight
evidence.

## 1. Stages, fixed sizes, and seal

1. **Smoke/development:** validation chromosomes only. Its size may be reduced for mechanical tests;
   it cannot alter a claim or authorise the target.
2. **Preflight:** validation chromosomes 11/16/19, exactly 400 outcome-blind retained examples, using
   the complete fixed production path in §7. It writes an immutable artifact and sibling manifest and
   may not load chr8/9. The chromosome allowlist is applied while loading the GTF and before any FASTA
   fetch; filtering an already materialized all-chromosome dataset is forbidden.
3. **Target:** sealed test chromosomes 8/9, exactly 400 outcome-blind retained examples. It remains
   unavailable because `FROZEN_PREFLIGHT_ARTIFACT_SHA256` is `None`; only an independently reviewed,
   passing successor preflight can supply that later seal.

A production preflight or target requires a clean worktree, strict Apple MPS execution, and
`PYTORCH_ENABLE_MPS_FALLBACK` absent. A future target additionally requires all of the following
before any chr8/9 cohort construction, target path calculation, model/tokenizer load, or output-path
inspection (reading GTF/FASTA/FAI bytes only to establish the pre-read source identity is permitted):

- this protocol is reviewed and its header is marked `FROZEN`;
- the exact protocol SHA-256, full reviewed instrument fingerprint, and one normalized downstream R5
  stack fingerprint are pinned in the runner; the full stack payload covers R1c plus every R5 method,
  protocol, and transitive runtime source and must validate against the current clean checkout;
- the operator supplies the exact acknowledgement
  `I_ACKNOWLEDGE_R1C_WILL_OPEN_SEALED_CHR8_CHR9`;
- a fresh passing 400-example preflight is run **after the reviewed R1c protocol and instrument pins,
  plus the normalized full R1c→R5 source/method stack pin, are installed**. This preflight cannot
  depend on output-derived package pins whose bytes do not yet exist: its own R1c package pins and
  the later R5a/R5b/R5c package pins are installed only after their producing stages. The preflight
  artifact records all six freeze-authorization booleans true, and its sibling manifest agrees
  byte-for-byte and binds the exact production instrument described in §8;
- after that run, the exact immutable preflight artifact SHA-256 is reviewed and installed as the
  runner's third seal literal; target authorization requires literal equality to that hash before
  parsing the artifact or loading a model; and
- the requested output and manifest paths are fresh, and the canonical R1c chr8/9 target-attempt key
  has never been spent.

The target-attempt key is independent of `--out`, endpoint labels, construction revisions, protocol/code
revisions, and the reviewed method fingerprint. Before any chr8/9 census exists, it is derived only from
schema/program/stage, the declared holdout and `{chr8,chr9,n=400}`, and the exact checkpoint, GTF, FASTA,
and FASTA-FAI identities. Observed row, gene, construction, and partner-margin digests cannot enter this
pre-read key. Method, construction, stack, preflight, Git, and requested output identities are context
inside the receipt. A later implementation or renamed endpoint cannot mint a second attempt for the same
source/checkpoint/holdout. The receipt is create-only under the Git common directory and conservatively
marks the target as potentially spent even if the process crashes before a result is written.

After immutable artifact and manifest publication, a separate create-only completion authenticates their
exact relocation-safe identities, the receipt bytes and IDs, method and R5-stack fingerprints, observed
ordered cohort/construction/partner-margin digests, every gate, and the official verdict. Failure before
completion remains spent. The artifact embeds the exact receipt identity and only the declared completion
logical name, avoiding a circular hash over completion bytes.

There is no `--allow-draft-target` escape hatch. The sealed split is not inspected during instrument
development.

## 2. Outcome-blind cohort and construction

- Source population: the existing canonical internal-short-intron cohort, both strands, with an
  in-window downstream intron-mate acceptor and donor-to-acceptor span 60--150 nt.
- Sampling uses `load_examples_unscored`, which performs no task forward and reads no masked-token
  margin, corruption effect, attention value, or other model output. Structural filtering and the
  first-400 selection therefore finish before the adapter and task are loaded.
- On target, the durable attempt is written after authorization, exact fingerprint verification, and
  preflight-package validation, but before `load_examples_unscored` is allowed to construct chr8/9 rows
  and before any model or tokenizer load.
- Primary eligibility requires the annotated paired donor, downstream intron-mate acceptor (`P`),
  upstream exon-mate acceptor (`X`), and matched non-partner donor (`N`) used by the 2x2.
- At least 30 distinct genes are required for every stored stage and every production-classifier call.
- The paired donor is changed `GT -> AC`. The matched non-partner donor receives the same edit. Both
  edited sequences are checked with `planted_donor_motifs`; any planted `GT` or `GC` aborts.
- The matched donor is the eligible non-partner `GT` whose span to the downstream acceptor most nearly
  matches the true donor's span. The absolute span mismatch, equivalently the absolute difference
  between the two donor coordinates, must be at most 40 nt. Rows outside this caliper are excluded
  with an explicit reason.
- The interaction is evaluated only on outcome-blind distance-matched dyads satisfying
  `abs(exon_length - intron_length_down) <= 20 nt`. This same threshold applies to validation and
  target. It is applied before the first 400 rows are selected and before any model score. If fewer
  than 400 eligible rows exist, the run stops; the caliper is not relaxed.
- Within the stage's already-authorized chromosome allowlist, construction scans the exhaustive
  deterministically shuffled eligible split (`load_examples_unscored(..., n=None)`), applies all
  comparator and dyad filters, and then takes the first 400. This avoids making capacity depend on an
  arbitrary finite prescreen while remaining outcome-blind. The artifact binds ordered-ID hashes for
  the screened, comparator-complete, dyad-eligible, and selected populations.
- The deterministic cohort-order tag remains pinned to the pre-redesign v2 ordering
  (`r1c-single-g-primary-family-v2`) while the scientific construction/schema advance to v4/v3. This
  prevents the redesign itself from silently drawing a new validation cohort after `Y_PD`, `Y_PN`, and
  `Y_XD` were observed. The tag and exhaustive-scan rule are both fingerprint-bound.
- Each acceptor endpoint masks only the `G` of an `AG` and measures the native
  `logit(G) - max(logit(A), logit(C), logit(T))` margin. There is no joint `AG` mask and no probe-trained
  endpoint.

Comparator and dyad completeness are outcome-blind restrictions. Every artifact reports screened,
comparator-complete, dyad-eligible, retained, and excluded counts and ordered-ID hashes; stable
exclusion reasons; example, gene, chromosome, and strand counts; donor-decoy mismatch; and dyad
distance mismatch.

## 3. Sole decision-bearing family

The four raw 2x2 cells are clean-minus-edit margin drops, never ratios:

| Cell | Masked acceptor readout | Donor edit |
|---|---|---|
| `Y_PD` | downstream intron-mate acceptor `P = next_acceptor_pos-1` | real paired donor `D`, `GT -> AC` |
| `Y_PN` | the same `P` readout | matched non-partner donor `N`, the same `GT -> AC` edit |
| `Y_XD` | upstream exon-mate acceptor `X = acceptor_pos-1` | the same real donor edit `D` |
| `Y_XN` | the same `X` readout | the same matched non-partner edit `N` |

The sole decision-bearing family is exactly, in order:

1. `Y_PD`;
2. `Y_PN`;
3. `S_P = Y_PD - Y_PN`; and
4. `I = (Y_PD - Y_PN) - (Y_XD - Y_XN)`.

`SESOI = 0.05` margin units. All four cells and every derived contrast use the same retained dyad
cohort. `Y_XD`, `Y_XN`, and `S_X = Y_XD - Y_XN` are stored as descriptive values but are not required
to be equivalent. In particular, a positive `Y_XD` does not invalidate the core paired-donor claim;
the interaction asks whether paired-donor selectivity is larger at `P` than at `X`.

There are two nested declaration paths. Each path requires **both** its Holm-adjusted boundary-test
IUT and its strict simultaneous-CI conjunction to pass; either gate alone is insufficient:

- **core paired-donor behavior:** one-sided tests of `theta > +0.05` for `Y_PD` and `S_P`, TOST of
  `-0.05 < theta < +0.05` for `Y_PN`, and strict simultaneous intervals classifying `Y_PD` and `S_P`
  material-positive and `Y_PN` equivalent;
- **stronger downstream preference:** every core condition plus a one-sided test of `theta_I > +0.05`
  and a strict simultaneous interval classifying `I` material-positive.

The raw p-value for each named path is the maximum of its constituent one-sided p-values (an
intersection-union test). Holm correction is then applied across **exactly two hypotheses**, the named
core and stronger paths. A p-value/CI disagreement always returns `UNRESOLVED`; it can never be broken
in the desired direction.

Both masked readouts must have mean clean margin at least `0.50`. Those clean margins and the non-site
MLM health assay are positive controls. No masked-donor endpoint is measured or invented as an anchor.

## 4. Descriptive 2x2 row

`Y_XD`, `Y_XN`, and `S_X` are reported per example and as simple means, with no separate interval,
p-value, equivalence requirement, calibration gate, or verdict. Their information enters inference
only through the preregistered `I` row in the four-member simultaneous family. This keeps the core path
from treating a biologically plausible exon-mate response as an invalid control while still allowing
the stronger downstream-preference claim to fail cleanly.

## 5. Whole-gene boundary inference and strict simultaneous intervals

The production classifier is one indivisible pipeline. Its point estimate for each endpoint is the raw
example-level mean. All randomisation and bootstrap operations preserve whole genes.

First, the ordered positive endpoints `Y_PD`, `S_P`, and `I` receive one-sided randomisation tests of
`H0: theta <= +0.05` against `H1: theta > +0.05`. `Y_PN` receives two one-sided tests:
`H0L: theta <= -0.05` against `theta > -0.05`, and `H0U: theta >= +0.05` against
`theta < +0.05`. Its TOST p-value is the maximum of those edge p-values. Every test centers its row at
the stated null boundary before whole-gene sign flipping. The same ordered gene labels, the same draw
index, and the same Rademacher sign matrix are reused across all five boundary tests, including both
TOST edges. Each p-value is exact add-one Monte Carlo, `(1 + exceedances)/(B + 1)`, with ties included.
The observed statistic is formed from per-example boundary-centered values rather than subtracting the
boundary after averaging. A conservative float64 comparison tolerance,
`eps * max(128,n_examples) * max(1,max|centered values|,max|oriented null|,|statistic|)`, snaps only
machine-scale zero statistics to zero and counts `oriented_null >= statistic - tolerance`. This
prevents an exact closed-null boundary such as 128 copies of `0.05` from becoming a false alternative
because their floating mean is `0.05000000000000001`; the tolerance can only add null-tail ties.
At the fixed production `B = 4,000`, the minimum attainable p-value is `1/4,001`.
The artifact serializes every boundary, alternative, oriented statistic, exceedance count, `B`, seed,
namespace, per-test null-draw digest, shared-null digest, shared sign-matrix digest, and ordered-gene-label
digest.

Second, constituent p-values are combined into path p-values by intersection-union testing:

- `p_core = max(p_Y_PD, p_Y_PN_TOST, p_S_P)`;
- `p_stronger = max(p_Y_PD, p_Y_PN_TOST, p_S_P, p_I)`.

Holm step-down correction is applied once across exactly `{core, stronger}`. Both raw IUT p-values and
both adjusted p-values are stored. No endpoint-wise zero-boundary max-absolute p-value enters a verdict.
The shared helper still emits that historical diagnostic while constructing intervals; R1c renames it
as non-decision-bearing in the artifact.

Third, the gene-cluster bootstrap resamples whole genes, preserves all rows from a gene, scales each
endpoint by its own bootstrap standard deviation, and uses the 95th percentile of the maximum absolute
standardized deviation across all four endpoints. Multiplying that critical value by each endpoint's
scale yields simultaneous 95% interval half-widths. Interval boundaries are **strict**:

- `MATERIAL_POSITIVE` only when the lower bound is `> +0.05`;
- `MATERIAL_NEGATIVE` only when the upper bound is `< -0.05`;
- `EQUIVALENT_WITHIN_SESOI` only when the lower bound is `> -0.05` and the upper bound is `< +0.05`;
- a bound that touches either SESOI edge is `UNRESOLVED`.

A path is declared only when its Holm-adjusted IUT p-value is at most `0.05` **and** every required
strict interval classification passes. If exactly one of those gates passes, the artifact explicitly
records `BOUNDARY_P_STRICT_CI_DISAGREEMENT` and the path is unresolved. If neither passes, the path is
`NOT_DECLARED`: that is a valid scientific nonconfirmation, not an instrument failure. There is no
global maximum-half-width gate; endpoint widths and their maximum remain disclosed diagnostics.

### 5.1 Necessary equal-gene robustness conjunction

The exact sign-flip/IUT/Holm tests plus strict simultaneous intervals above remain the inferential
authority. Alongside them, not in place of them, production runs apply a preregistered tail-sensitivity
gate with at least 10,000 deterministic whole-gene bootstrap draws. Each gene has equal weight. For
`Y_PD`, `S_P`, and `I`, the gate tests the same positive `+0.05` boundary; for `Y_PN`, it tests the same
`[-0.05,+0.05]` equivalence region. Every endpoint must pass under both the raw equal-gene estimator and
the 10%-trimmed equal-gene estimator. Quantiles and all sufficient replay metadata are emitted in
JSON-native form.

The core robustness path requires raw and trimmed support for `Y_PD`, `Y_PN`, and `S_P`; the stronger
path additionally requires `I`. Exact stronger support with a robustness-sensitive `I` may license only
the core wording. Exact core support that fails this raw+trimmed conjunction returns
`PAIRED_DONOR_PATH_UNRESOLVED_ROBUSTNESS`. When construction, MPS runtime, provenance, calibration,
margin, health, and replay checks are healthy, such a tail-driven failure is a valid **scientific
unresolved** outcome, never an instrument failure and never permission to rerun the sealed target.

## 6. Exact production-decision-path preflight

Preflight calls the R1c-local calibrator on the four primitive 2x2 cells, whole-gene labels, and the
**exact §5 classifier**: SESOI-boundary tests, `Y_PN` TOST, path IUT, Holm over exactly two paths, and
the strict simultaneous-CI second gate. It uses the target's `SESOI = 0.05`, `alpha = 0.05`,
minimum-gene rule, randomisation count, seed, and endpoint-scale-normalized interval construction
(`studentized=True` in the shared helper, not a nested bootstrap-`t`). Every least-favourable case and
plant trial receives a fresh exact classifier call and a unique auditable namespace.

The outer trials do **not** sign-flip effect-bearing raw outcomes. First, each of the four cell rows is
centered by subtracting its observed example-level mean. One Rademacher sign is then assigned per gene
and shared across every example and all four centered cells from that gene. `S_P`, `S_X`, and `I` are
derived only after this whole-gene flip. This construction is translation-invariant and preserves the
2x2 algebra in every trial.

Five algebraically coherent least-favourable composite-null cases are generated as primitive-cell
translations, never as independently invented derived rows: `Y_PD = +0.05`, both `Y_PN` equivalence
edges (`-0.05` and `+0.05`), `S_P = +0.05`, and `I = +0.05`. The first four are null surfaces for both
named paths; the `I` surface applies only to the stronger path. Non-boundary constituents are placed in
their alternatives subject to the exact 2x2 algebra. To prevent floating-point summation from turning an
exact mathematical boundary into an apparent strict inequality, each synthetic boundary is shifted
`1e-12` toward its composite null. This numerical guard is fingerprint-bound and disclosed beside the
nominal boundary; it is negligible relative to the `0.05` SESOI and may only make the calibration more
conservative.

The `Y_PD` and `S_P` faces contain coupled nuisance parameters because
`S_P = Y_PD - Y_PN`; a single convenient point on either face is not asserted to maximize rejection
over that continuum. Instead, the instrument binds a uniform event-inclusion argument. For any named
path `h` and any null constituent `j`,

`{declare h} subseteq {Holm_h(max_k p_hk) <= alpha} subseteq {p_hj <= alpha}`.

The second inclusion follows because a raw IUT p-value is the maximum constituent p-value and Holm's
adjusted p-value is never smaller than that raw value. The strict simultaneous-CI conjunction can only
remove declarations. Because endpoint `j` is centered at its own null boundary and tested by the
whole-gene sign-flip/add-one rule, this supplies a level-`alpha` upper bound for every value of all
other constituents; no nuisance monotonicity or interpolation assumption is required. The artifact
also binds three coherent primitive-cell representatives on each of the `Y_PD` and `S_P` faces,
spanning the admissible negative and positive interiors of `Y_PN` and multiple positive `I` values.
Their middle family vectors are exactly the classifier surfaces executed in all 1,000 outer trials.
The representatives audit the 2x2 geometry; the event inclusion, not the finite grid, controls the
full nuisance continuum.

For each named path, calibration records declarations separately for every applicable boundary case;
its false-positive count is the maximum across those cases. With 1,000 trials, **every** case must have
at most 38 declarations, equivalently the worst-case one-sided 95% Clopper--Pearson upper bound must be
at most `0.05`. The larger audit was chosen before freezing because the earlier 200-trial design gave an
unstable pass/fail decision near the acceptance edge: 5/200 had an upper bound of `0.05184`, whereas
extending the same deterministic namespace to 1,000 trials gave 10/1,000 and an upper bound of
`0.01690`. This changes audit precision, not the scientific alpha, SESOI, classifier, seed, or
observed-data decision.

The plant was frozen before `Y_XN` was scored. It adds `delta = 0.40` only to raw cell `Y_PD`, with raw
cell vector `[0.40, 0, 0, 0]`. Deriving the family after planting gives exactly
`[Y_PD, Y_PN, S_P, I] = [0.40, 0, 0.40, 0.40]`. A plant trial separately records whether the exact core
path succeeds and whether the nested stronger path succeeds. Both paths require at least 916 successes;
equivalently each one-sided 95% Clopper--Pearson lower bound must be at least `0.90`. Thus at 1,000 trials,
38 boundary false positives pass and 39 fail, while 916 plant successes pass and 915 fail.

Every calibration trial serializes its outer sign namespace and, for all five boundary cases plus the
plant, every boundary-test replay field and adjusted path p-value from §5. The ordered trial record has
its own canonical digest. Preflight also requires observed `Y_PN` to pass both its TOST and its strict
simultaneous equivalence interval. Calibration failures block target execution; there is no additional
global half-width criterion.

## 7. Non-site MLM health and fixed production contract

For every retained example, eight deterministic non-site probe positions are scored on both the clean
sequence and its paired-donor-corrupt counterpart. The downstream acceptor `G` remains masked in both
forwards, and each probe is also masked; probes exclude both annotated boundaries and the edited donor.
The artifact stores per-example clean and corrupt top-1 nucleotide accuracy and their paired damage.

The fixed health gates are mean clean accuracy at least `0.40`, mean corrupt accuracy at least `0.40`,
and absolute mean damage at most `0.05`. Health is a collateral-damage gate, not evidence for the
paired-donor behavior. Mean clean margins at both `P` and `X` must be at least `0.50`, planted-motif
checks must pass, the fixed dyad caliper must supply 400 rows, and at least 30 genes must remain. No
masked-donor readout is part of health, eligibility, calibration, or verdict.

Both preflight and target use exactly:

| Setting | Fixed value |
|---|---:|
| retained examples | 400 |
| batch size | 16 |
| seed | 0 |
| bootstrap/sign-flip draws per classifier call | 4,000 |
| raw + 10%-trimmed equal-gene robustness draws | 10,000 |
| trials per least-favourable boundary case and coherent plant | 1,000 |
| health probes per example per construct | 8 |
| matched-donor distance caliper | 40 nt |
| exon/intron dyad distance caliper | 20 nt |
| coherent plant delta | 0.40 margin units |

Changing any production value requires a new reviewed protocol and a fresh preflight. Smoke runs may
use smaller values but cannot be supplied as target authorisation evidence.

The earlier 2,000-draw development diagnostic is not production calibration for this instrument and
cannot bind a target. At protocol freeze the v4/schema-3 construction remains unrun; its first eligible
production preflight must use all 4,000 draws per classifier call and all 1,000 outer trials.

## 8. Frozen instrument identity, provenance, and outputs

Before scoring, the runner constructs one canonical instrument fingerprint covering:

- every regular file in the checkpoint bundle;
- the exact GTF, FASTA, pre-existing FASTA `.fai`, this protocol, seal-normalized runner, and statistics module (only the four
  literal `FROZEN_*_SHA256` assignment values are normalized, avoiding self-referential protocol,
  instrument, downstream-stack, and preflight-artifact hash cycles;
  all runner logic remains covered);
- the imported measurement and construction modules, adapter, dataset/GTF loaders, provenance helper,
  `pyproject.toml`, and `uv.lock`;
- construction version, ordered raw-cell, primary-family, and descriptive names; boundary hypotheses;
  both nested IUT paths and their exact Holm family; strict-CI conjunction; common-sign audit; all five
  least-favourable primitive-cell cases, the nuisance-continuum event-inclusion proof and representative
  surfaces, and their numerical guard; residual-centering/sign-flip policy;
  plant; raw and 10%-trimmed equal-gene robustness; thresholds; both calipers; exhaustive scan rule;
  fixed sample size; and every production count;
  and
- Python and platform identity, relevant package versions, selected device, and the full runtime
  contract; and
- the full normalized downstream R5 stack payload and its fingerprint, validated against current
  sources before any production model access.

After the authorization-only read of the pinned repository protocol, every production input is
content-snapshotted before user-supplied data are semantically used. Repository sources are serialized by
repository-relative logical name; external inputs use bare logical names plus exact size and SHA-256, so
artifact/manifest matching is relocation-safe. The snapshotted protocol hash must
equal the authorization hash. The snapshot also includes the complete checkpoint directory inventory,
GTF, FASTA, FASTA FAI, seal-normalized runner, statistics and construction sources,
environment locks, and (for target) the preflight artifact and its manifest. Additions, removals, or byte
changes in the checkpoint directory abort. The complete checkpoint inventory is copied into a private
temporary directory, the copy is rehashed against the snapshot, and only that absolute staged path is
passed to the model/tokenizer loader. The original checkpoint path is never loaded on the production path.
All bound inputs are reverified before publication; target drift leaves the conservative attempt spent but
cannot publish a result.

Freezing requires review and pinning of that complete fingerprint in addition to the protocol hash,
followed by a fresh production preflight and review/pinning of that artifact's exact SHA-256.
The target fingerprint must exactly equal both the reviewed pin and the passing preflight's bound
fingerprint. The preflight artifact's sibling manifest must identify and hash the exact artifact bytes.
An otherwise identical draft or partially pinned preflight is non-binding; normalization of seal
literals cannot make it target-eligible.
Before target loading, the **artifact** hash must equal the reviewed preflight seal pin and its sibling
manifest must bind those exact artifact bytes. The literal artifact pin covers the complete 1,000-trial calibration registry without rerunning thousands of nested
4,000-draw classifiers on the authorization path. Hash parity is necessary but not the only check: the
runner independently rebuilds all four primitive cells and the ordered family from serialized margin
levels, recomputes the observed boundary classifier from row-level genes under the frozen seed and
namespaces, rederives health, clean-margin, gene-count, observed-equivalence, and selected-cohort gates,
and requires the selected cohort hash and raw+trimmed equal-gene robustness record to reproduce. It also rederives every compact calibration audit's
strict-CI/IUT/Holm decision and recomputes all false-positive and plant-success counters from the 1,000
stored trial decisions. It then revalidates the artifact's frozen
status, stage, split, sample size, construction version, exact thirteen-gate set, ordered four-row family,
scale-normalized `alpha = 0.05`/`SESOI = 0.05` inference, every add-one boundary test and shared-sign
digest, both TOST edges, both raw IUT and Holm-adjusted path p-values, strict intervals and disagreement
policy, endpoint-centered primitive 2x2 outer trials, every least-favourable boundary case, the uniform
nuisance-continuum contract, structured
raw-cell plant, exact trial counts and Clopper--Pearson bounds, production constants, runtime contract,
and sealed-split attestation against the current target instrument.

Every non-smoke artifact and sibling manifest is published with create-only atomic writes. Manifest input
records come from the pre-use snapshot rather than post-outcome live rehashing, and the manifest preserves
the pre-run Git commit and dirty state (the target must record clean even though its own attempt ledger entry
is subsequently created). Per-example sufficient rows contain IDs, genes, chromosomes, strands, comparator
coordinates, both caliper mismatches, all four raw cells, the four primary values, descriptive `X`
cells/contrast, all clean/edit margin levels, and clean/corrupt health. In particular, every row preserves
`margin_levels.P_clean`, `P_under_D`, and `P_under_N` unchanged for same-row downstream common-leg replay.
The artifact emits ordered hashes of the source cohort, example IDs, sorted-unique genes, ordered genes,
exact six-position rows, partner-margin levels, and construction. The manifest records the command, seed, all
input and output hashes, Git revision, dirty state, input-snapshot digest, and target-attempt identity when
applicable. The preflight artifact and manifest therefore form a relocation-safe, outcome-blindly
constructed 400-row chr11/16/19 validation handoff for later R5 preflights; they have no target receipt or
completion. This permits gene-level reanalysis without reopening the checkpoint.

The model-free preflight-package verifier authenticates the exact artifact/manifest output and source
identities, current R5 stack, fingerprint, all row-derived sufficient statistics, calibration, robustness,
gates, and `PREFLIGHT_GO`, and returns a compact attestation. For a published target, the target-package
verifier additionally loads the exact common-dir attempt and completion. From stored margin levels it
reconstructs all four primitive cells, family rows, exact boundary inference, raw+trimmed robustness,
health and cohort identities, gates, and verdict, then replays canonical receipt/completion IDs,
source/runtime/stack bindings, input snapshots, manifest outputs, and every crosslink. Any stale top-level
decision or repackaged row, robustness, preflight, receipt, or completion is rejected before R5 entry.

## 9. Target decision and claim language

The sealed target must first pass its bound preflight and every construction, capacity, gene-count,
clean-margin, and health gate. The verdict then follows the exact nested paths:

- `CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE` when the stronger Holm-adjusted IUT, complete strict-CI
  conjunction, and stronger raw+trimmed equal-gene robustness conjunction pass;
- `CONFIRMED_PAIRED_DONOR_ACCEPTOR_BEHAVIOR` when the exact core path and its raw+trimmed robustness
  conjunction pass but the stronger exact/robust conjunction is not declared;
- `PAIRED_DONOR_PATH_UNRESOLVED_ROBUSTNESS` when the exact core path is declared but its necessary
  raw+trimmed equal-gene conjunction fails;
- `PAIRED_DONOR_PATH_UNRESOLVED` when the core path's Holm and strict-CI gates disagree; and
- `NO_CONFIRMED_PAIRED_DONOR_PATH` when both necessary core gates agree in failing.

The final case is a valid nonconfirmation (`NO_PATH`), not `INSTRUMENT_UNRESOLVED`: once construction,
capacity, calibration, observed control, gene-count, clean-margin, and health gates have passed, failure
of the candidate scientific path does not retroactively invalidate the assay. A boundary touch is handled
by the same rule: it is unresolved only when the p-value gate passes; when both gates fail, the path is
simply not declared.

The core paired-donor claim does not require `I` and does not require `Y_XD`, `Y_XN`, or `S_X` to be
equivalent. The stronger wording is unavailable unless the complete core path and positive interaction
both hold in the same four-row simultaneous family.

Even a successful core R1c target licenses only this statement: in this checkpoint, dyad-matched cohort,
corruption, single-nucleotide endpoint, and robustness envelope, the native downstream-acceptor
masked-token margin causally depends on the paired-donor input and is selective over the matched
non-partner donor edit. The stronger path additionally licenses that the paired-donor selectivity is
larger at the downstream intron-mate acceptor than at the upstream exon-mate acceptor. It
does not license an internal role, a shared donor/acceptor circuit, mediation, semantic transfer,
generative faithfulness, RNA processing in cells, spliceosome chemistry, or a biological mechanism.
Those bridges require the authenticated R3d component prior and the conditional R5
acceptor-mechanism program with its own successful causal interventions. R3c remains provenance for
prior component exposure and does not substitute for the R3d scientific entry gate.
