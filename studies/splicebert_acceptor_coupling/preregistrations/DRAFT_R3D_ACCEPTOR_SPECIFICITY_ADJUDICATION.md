# DRAFT R3d — prospective acceptor-specificity adjudication on train chromosomes

> **Status: `DRAFT_NOT_FROZEN`.** The runner exists, but every literal freeze pin is
> `None`; no R3d target is authorised. A smoke run, an immutable exposure registry, a
> clean 256-gene preflight, review, and literal protocol/instrument/preflight pins are
> required before target execution. Development smoke runs cannot authorise a target;
> no R3d production preflight or target model/data run is recorded by this draft.

R3d adjudicates the two R3c v2.1 paths that ended
`INCONCLUSIVE_METHOD_DISAGREEMENT`: reuse of `L3H1` and the original distinct-route
conjunction. It changes the inferential construction, not the masked-acceptor assay or
the estimands. Every reported interval is the inversion of its decision-bearing exact
test, so a p-value/interval disagreement is not a possible R3d disposition.

R3d is a **prospective, endpoint-specific verification on registry-disjoint train
genes**. It is
not a sealed-split replication or an independent confirmation: the hypotheses and
thresholds were informed by opened R3/R3c results on the same checkpoint, and earlier
unrestricted exploratory loaders have already read chr8/9. R3d's explicit chromosome
restriction prevents further forbidden-chromosome example construction and model
scoring; it does not restore a historical seal or mean those FASTA bytes were unread.

## 1. Scoped claim and exclusions

The only positive statement R3d can license is that a frozen component-level pattern
reappears for the masked-acceptor-G endpoint on a registry-disjoint train cohort under
the two declared interventions.

R3d cannot establish attention selection, an address, a donor payload, an acceptor-side
write, downstream delivery, mediation, sufficiency, faithfulness, a complete circuit,
or a biological mechanism. It never constructs a masked-donor endpoint. A healthy
scientific failure is `NOT_SUPPORTED`; an ineligible path is `HEALTH_FAILED`. Neither is
evidence that the component or route is absent.

## 2. Bound predecessor instrument

R3d reuses the measurement helpers from the exact R3c v2.1 runner and binds both that
runner's bytes and its clean preflight artifact. The R3c artifact must replay as
`gfmcd.r3c.acceptor_head_confirmation/2`, stage `preflight`, verdict
`TARGET_BINDABLE_PREFLIGHT_GO`, with all global gates passed. Its recorded Path-A health
must authorise `L3H1_reused_component`, because otherwise every R3d path is unreachable.
Its Path-B health is bound as a monotone, path-local ceiling: suppression carries forward
but does not erase or block a healthy Path A.

R3d calls R3c v2.1's own semantic preflight validator on that artifact and its recorded
instrument binding; a top-level GO label or artifact byte pin alone is insufficient.

The following runner literals remain `None` until freeze:

- `FROZEN_R3C_V2_1_RUNNER_SHA256`
- `FROZEN_R3C_V2_1_PREFLIGHT_SHA256`
- `FROZEN_EXPOSED_GENE_REGISTRY_SHA256`
- `FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256`
- `FROZEN_PROTOCOL_SHA256`
- `FROZEN_INSTRUMENT_FINGERPRINT_SHA256`
- `FROZEN_PREFLIGHT_ARTIFACT_SHA256`
- `FROZEN_PREFLIGHT_MANIFEST_SHA256`

Changing the R3c runner, its preflight, the registry, the statistical core, this
protocol, or any executable R3d byte changes the instrument binding.

## 3. Prospective cohort

### 3.1 Restriction before sequence materialisation

The GTF loader receives an explicit allow-list before any sequence-window FASTA fetch.
The only allowed chromosomes are:

`chr1–chr7, chr10, chr12–chr15, chr17, chr18, chr20–chr22, chrX, chrY`.

Validation chromosomes `chr11/chr16/chr19` and test chromosomes `chr8/chr9` are absent
from the allow-list. Filtering an already materialised all-chromosome dataset is
forbidden. The runner records the allow-list and fails if a returned row lies outside it.
The provenance snapshot separately hashes the entire FASTA file before execution; this
does read all file bytes, including forbidden chromosomes. The audit therefore asserts
only `forbidden_chromosomes_materialized_as_examples=false` and
`forbidden_chromosomes_model_scored=false`, never that forbidden FASTA bytes were unread.

Eligibility is model-outcome-blind and inherited from the acceptor-only R3c assay:

1. the dataset row is train-split eligible under `CouplingConfig`;
2. the paired upstream donor and a matched pseudo donor are distinct canonical `GT`s;
3. their positional mismatch is at most 40 tokens;
4. paired `GT→AC` is admissible; and
5. pseudo `GT→AC` plants no donor motif.

No acceptor pseudo is required. No model output enters eligibility.

### 3.2 Immutable exposure registry

Before freeze, a separate immutable JSON file with schema
`gfmcd.r3d.exposed_gene_registry/1` must enumerate sorted, unique gene IDs and carry the
extensional frozen `scope_id`
`claude_inverse_acceptor_and_clean_alt_head_screen_direct_exposures_v1`. It contains
exactly two registered direct-exposure classes: (A) the reconstructed Claude M1/M2/M3
inverse-acceptor scratch cohorts, which did **not** all have the R3d paired-versus-pseudo
endpoint; and (B) all 400 clean-alt head-screen shards. Source B includes both its donor
and acceptor target strata because their joint donor/acceptor ratios selected the reverse
heads. The registry does not claim to enumerate every historical candidate-selection
exposure.

Its `gene_ids_sha256` is the canonical JSON hash of that exact list. The runner separately
pins the registry bytes and its dedicated sibling manifest, then validates the manifest's
builder, source bindings, 400-shard/both-strata contract, output hash/size, and scope.
Adding a newly registered gene requires a new registry version and a new instrument
freeze; a target cannot silently continue against a stale registry.

The registry's structured `scope` explicitly records `global_freshness_claimed: false`.
HCA18/broad Atlas scoring, donor-circuit work, earlier R3 train candidate generation,
and other train-corpus model scoring are explicitly outside this two-class blacklist.
Therefore “registry-disjoint” and “endpoint-specific prospective verification” are the
only allowed cohort terms; “candidate-selection-disjoint,” “unexposed genes,”
“gene-fresh,” and “independent train confirmation” are forbidden.

The registry is an exclusion boundary, not evidence. It contains no fitted value,
activation, effect estimate, or outcome-based rank.

### 3.3 One row per gene and stage allocation

Rows within each gene are ranked by SHA-256 of
`r3d-train-one-row-per-gene-v1 | gene_id | example_id | pseudo_donor`; the minimum is
selected. Genes are then ordered by SHA-256 of
`r3d-train-gene-order-v1 | gene_id`, with gene ID as a deterministic tie-breaker.
Exactly one row is analysed per gene, so the independent unit and the measured row are
identical.

- **Smoke:** the first 32 train-eligible registry genes (caller may raise this, never
  below 30).
- **Preflight:** exactly the first 256 train-eligible registry genes.
- **Target:** every eligible gene absent from the frozen two-class direct-exposure
  registry, with no arbitrary `n` cap. Target execution fails if fewer than 2,500 genes
  remain.

Preflight and target are therefore gene-disjoint by construction. The target is still
not an independent confirmation for the reasons in the opening scope statement.

## 4. Assay and intervention family

For every example, the sole readout masks exactly the `G` at
`next_acceptor_pos - 1`. Native coupling is clean masked-acceptor margin minus the same
margin after the paired upstream donor changes `GT→AC`; the negative control applies the
same edit to the matched pseudo donor. No masked donor is encoded or scored.

Two R3c v2.1 interventions are retained:

- `within_example_token_mean_ablation`; and
- same-example paired/pseudo corrupt-to-clean `activation_noising`.

The fixed component/mode registry is:

- `L3H1` under mean ablation and activation noising;
- `L4H11`, `L4H9`, `L4H0`, `L4H2`, and `L5H11` under activation noising only.

For each component/mode, partner and pseudo drops are primitive; specificity is derived
as `partner − pseudo`. The delivery gap is derived, separately for partner and
specificity, as

`mean(L4H11,L4H9) − mean(L4H0,L4H2,L5H11)`.

Specificity and gap rows are always derived after an outer randomisation or plant.
They are never independently edited.

This gives exactly **14 primitive rows** and **23 derived family rows**.

## 5. Frozen paths

The SESOI is `0.05` raw masked-margin logit units. Boundaries are strict: touching
`±0.05` is unresolved.

### Path A — `L3H1_reused_component` (6 rows)

Under both intervention modes:

- partner: `MATERIAL_POSITIVE` (`lower bound > +0.05`);
- pseudo: `EQUIVALENT_WITHIN_SESOI` (`lower > −0.05` and `upper < +0.05`);
- specificity: `MATERIAL_POSITIVE`.

Path A contains 8 decision edges: four positive lower-bound edges and two TOST edges
for each of two pseudo rows.

### Path B — `distinct_acceptor_route_candidate` (17 rows)

Under activation noising:

- `L4H11` and `L4H9`: partner positive, pseudo equivalent, specificity positive;
- `L4H0`, `L4H2`, and `L5H11`: partner, pseudo, and specificity all equivalent; and
- partner and specificity delivery gaps: positive.

Path B contains 28 decision edges. This is the original R3c route path. **`L4H7` is not
in it.** R3d does not expand the route using a previously supported secondary head.

`L2H15` is excluded because it is health-failed and may not be revived. No unregistered
head or MLP may enter a disposition.

## 6. Exact test inversion and fixed-sequence control

The randomisation unit is the whole gene. Production uses `B=4,000`, seed 0, and exact
add-one Monte Carlo p-values. Within one path, every decision edge shares the same
whole-gene Rademacher sign matrix and ordered gene vector. Each edge records its sign,
gene, endpoint, crossing, adjusted-crossing, exceedance, and adjacent-transition hashes.

The randomisation test assumes independent gene endpoint vectors that, in a location-shift
model, are jointly centrally symmetric and invariant under arbitrary independent
gene-wise sign flips about the tested boundary. The test is exact at the boundary and
conservative inside a closed one-sided null only under the corresponding monotone
location-shift assumption. Marginal symmetry, global symmetry alone, or merely a mean on
the null side is insufficient. One row per gene removes unequal cluster weighting but
does not make these assumptions automatic. The bounds below are
**pointwise one-sided decision bounds** for their respective edges, not a simultaneous
23-row confidence region. Multiplicity is controlled at the conjunctive path/study level
by the IUT plus fixed sequence, not by relabelling these pointwise bounds simultaneous.

For a one-sided `greater` edge, inversion returns the first float64 boundary whose
strict test no longer rejects; it is the lower confidence bound. For `less`, inversion
returns the last non-rejecting boundary; it is the upper bound. The transition itself,
one representable rejecting neighbour, and one non-rejecting neighbour are all evaluated
and serialized. For every reference boundary `b`:

- `lower > b` iff the exact `greater` test at `b` has `p < α`; and
- `upper < b` iff the exact `less` test at `b` has `p < α`.

Thus each reported pointwise confidence statement and decision are the same test. `p = α` does
not reject.

Studywise error is controlled by a **fixed sequence**:

1. test Path A at `α=.05` as an intersection-union conjunction;
2. only if every Path-A edge rejects, every interval has its required strict class, and
   health passes, test Path B at `α=.05`;
3. otherwise Path B is `NOT_TESTED_GATE_CLOSED`.

Under any configuration, the first true null in this ordered family is tested at most at
`.05`, giving strong studywise FWER control without an outcome-informed unequal alpha
split. The IUT and fixed sequence do not assume independence among edges or paths. A path
is `SUPPORTED` only when all its constituent edges and health gates pass.

## 7. Three-way monotone health

The bounded identical-policy token-reconstruction check is run for every registered
component/mode. The maximum interpretable set can only shrink:

1. the bound R3c v2.1 preflight health must authorise the path;
2. the R3d 256-gene preflight health must authorise it; and
3. target health must also pass.

Failure at any earlier level is permanent for this instrument. Target measurements may
not revive a preflight-suppressed path. Calibration enables both synthetic paths only to
test the classifier; synthetic health does not override observed health.

Because Path A is the fixed-sequence entry gate, an R3d preflight with Path-A health
suppressed cannot authorise or spend the target: every scientific target path would be
unreachable. Path-B health is local and may be suppressed while a healthy Path A
proceeds. On target, a new Path-A health failure yields `HEALTH_FAILED` and closes B;
a Path-B health failure after supported A yields B=`HEALTH_FAILED` without erasing A.

## 8. Preflight

The 256-gene preflight is run before target and must pass all assay, sensitivity,
resolution, and invariant gates:

1. paired-donor native coupling mean `> +0.20`;
2. matched-pseudo native coupling absolute mean `< 0.05`;
3. exactly one masked acceptor `G`, both donor targets are `GT`, both edits are `GT→AC`,
   and no donor readout exists;
4. no edit plants a donor motif;
5. explicit preconstruction train allow-list and one-row-per-gene selection replay;
   the complete eligible/exposed/registry-disjoint capacity record must match the frozen
   registry byte-for-byte at the semantic JSON level;
6. exact 14-primitive/23-derived algebra;
7. bound R3c health replay plus execution of current R3d health, with eligibility applied
   locally to dependent paths rather than as an all-path global success requirement;
8. exact test/inversion duality and adjacent-float transition checks;
9. all 36 boundary-face engineering diagnostics under the frozen familywise max-count
   gate; and
10. the 500-trial coherent planted-sensitivity gate.

### 8.1 Compact 36-face calibration

The suite contains exactly 8 Path-A faces and 28 Path-B faces. Faces are not packed
across the gate levels. Each separately puts one decision edge of its assigned path
at the relevant `+0.05`, `−0.05`, or `+0.05` TOST boundary while every other conjunct
remains strictly inside its alternative.

For each face, 200 trials center all 14 primitive rows, apply one shared whole-gene sign
flip, add the coherent primitive-space face offset, derive the 23-row family, and run the
production path classifier. A face passes when erroneous path support occurs in at most
20 of 200 trials. This is a frozen familywise maximum-count engineering test, not a claim
that 10% is the nominal inferential size. At a per-face Binomial(200,.05) point null,
`P(K>20)=0.001159908250747922`; the union bound over 36 faces is
`0.041756697026925194`, so suite-pass probability is at least
`0.9582433029730748` without assuming cross-face independence. The one-sided 95%
Clopper–Pearson upper bound at `K=20` is `0.14198925159094505` and is serialized as a
descriptive diagnostic only. The face suite remains an implementation diagnostic, not a
replacement for the analytic fixed-sequence FWER claim.

A separate coherent `+0.40` partner-only primitive plant leaves pseudo rows at zero and
derives all specificity/gap effects. It uses 500 trials; each path must be supported in
at least 462. The one-sided 95% Clopper–Pearson lower bound at 462/500 is
`0.9015717845992148`. Because B is tested only after A support, B successes must not
exceed A successes and the operative sensitivity event is the joint fixed-sequence B
count. At true joint success probability .95, `P(K≥462)=0.9953942562690719`. No
production threshold may be weakened after observing preflight output.

Smoke runs exercise the same 36 faces with reduced draws/trials and gate on a
**zero-residual coherent classifier/algebra calibration** in which every smoke plant
must succeed. This is mechanics-only: it checks the primitive-to-family algebra,
fixed-sequence classifier, exact test/inversion path, and compact audit machinery without
mistaking a 32-gene equivalence-power draw for an instrument failure. The same smoke
invocation also computes and prints the **observed-centered-residual plant outcome** with
its path counts and support patterns as an **explicit non-gating diagnostic**. That
diagnostic may fail, cannot change the smoke verdict, cannot weaken any production gate,
and cannot authorise a target.

The 256-gene production preflight remains unchanged: it centers the observed primitive
rows, uses those observed residuals for all 36×200 face trials and all 500 coherent-plant
trials, and requires at least 462 successes under 4,000 classifier draws. It serializes
compact per-face and per-path support patterns, counts, namespaces, and ordered
classifier/outer-null digests. Before spending target, the target runner reconstructs the
ordered primitive matrix from exact per-gene sufficient statistics, reruns all 36×200
face trials and 500 fixed-sequence plant trials through the production classifier, and
requires exact artifact equality.
A bindable preflight also runs that same semantic validator on its fully assembled result,
including predecessor-package/snapshot authentication, before result serialization and
publication; a producer cannot publish a bindable GO solely from its own top-level flags.

## 9. Target authorisation and immutable publication

Target invocation checks all eight literal pins and frozen protocol status before
inspecting any caller-supplied model, GTF, FASTA, registry, preflight, or output path.
After freeze it also requires:

- the separately pinned R3d preflight artifact and sibling manifest;
- the exact pinned registry and registry manifest, R3c runner, and R3c preflight;
- an instrument fingerprint match; and
- a clean git worktree with a full commit identity.

The target does not trust the preflight's PASS labels. Before any target row is
constructed or any model adapter is loaded, it revalidates the exact binding and clean
manifest, cohort/registry allocation, 14→23 algebra, baseline and construction records,
three-way health, publication-facing component summaries, fixed-sequence inference from
per-gene sufficient statistics, every compact calibration count/CP/operating-characteristic
field, and the full production calibration replay described in §8.1.

Before target JSON serialization or either create-only publication write, the runner
also calls the model-free `validate_r3d_target` replay. It first reruns the bound R3d
preflight validator, authenticates every predecessor package against the target input
snapshot, and restores compact per-gene statistics from their lexicographic storage
order to the frozen hash-ranked cohort order. It then rederives 14→23 algebra,
independent partner/pseudo/clean-margin baselines, construction and component summaries,
three-way health, exact fixed-sequence inference, global gates, official fail-closed path
dispositions, verdict, and the dynamic scientific licence. Thus publication-facing
fields cannot be promoted by internally consistent PASS labels alone. The same function
is the required model-free post-publication scientific replay after the target artifact
and sibling manifest have passed immutable package validation.
For that replay, predecessor packages are identified by stable `results/...` logical
path, byte size, and SHA-256 rather than by a worktree-specific absolute prefix. The
recorded snapshot must still authenticate the original recorded paths and bytes, but an
exact copy in another clean worktree is therefore auditable without weakening any byte
binding.
Malformed or non-replayable target records abort and leave only the spent-attempt/error
receipt. In contrast, a well-formed measured `false` assay/construction invariant or a
well-formed zero-count required-health execution record is replayed as a failed global
gate and may publish only `INCONCLUSIVE` with every official path fail-closed. The
validator must not turn those declared §10 dispositions into an execution exception.

All checkpoint files, GTF, FASTA, the exact `.fai` sidecar consumed by Pyfaidx,
registry, predecessor artifacts, runner, protocol, and statistical-core inputs are
snapshotted by size and SHA-256 before execution and checked
again before publication. Result and sibling manifest paths are create-only and may not
overwrite an existing file. The result records exact per-gene sufficient statistics,
cohort hashes, exact argv, seeds, draw counts, namespaces, input hashes, and the instrument
binding.

After all read-only authorization, semantic replay, and clean-git checks, but before
cohort construction or model loading, the runner writes a create-only target-attempt
receipt under the repository's shared Git common directory at
`gfmcd_target_attempts/r3d/`. Its output-independent key binds R3d target version,
checkpoint, GTF, FASTA and `.fai`, masked endpoint/operator, registry scope and gene-list
digest, and the stable target gene/ordered-row digests. Registry packaging, code revision,
and output pathname cannot mint a new biological target. The receipt records invocation,
start time, authorised commit, all literal pins, fingerprint, and the pre-attempt input
snapshot. A crash after reservation permanently spends the target; silent retry is
forbidden. A separate create-only completion receipt records either published
result/manifest hashes or a best-effort error closure, while the start receipt remains
authoritative after a hard crash.

The target uses every eligible registry-disjoint gene. It may not be rerun on a favourable
subset, repackaged to evade the attempt key, reported under a wider SESOI, or converted
into a sealed-split claim.

## 10. Frozen consequences

| result | licensed consequence |
|---|---|
| A and B supported | Prospective registry-disjoint train-endpoint support for `L3H1` reuse and the original distinct-route component pattern. The separately frozen R5 chain may be considered on its own gates. No address/write/delivery/mediation language is licensed by R3d itself. |
| A supported; B healthy but effect not supported | Prospective registry-disjoint train-endpoint support for `L3H1` reuse only; B=`NOT_SUPPORTED`. No distinct-route claim. |
| A supported; B-local health fails | Prospective registry-disjoint train-endpoint support for `L3H1` reuse only; B=`HEALTH_FAILED`. A B-local health failure does not erase A. |
| A healthy but effect not supported | A=`NOT_SUPPORTED`; Path B is not tested. R3d supplies no positive reuse or route claim. |
| target A-local health fails | A=`HEALTH_FAILED`; Path B is not tested. R3d supplies no positive reuse or route claim. (Such an A failure at preflight would have blocked target authorisation.) |
| an assay invariant, required health execution/replay, provenance binding, inversion-coherence check, or calibration gate fails | `INCONCLUSIVE`; no path disposition is promoted. Scientific component-health eligibility itself is path-local as specified above. |

Raw path calculations remain available as diagnostics, but if any global assay,
provenance, construction, semantic-replay, or calibration gate fails, every official
top-level path disposition is `INCONCLUSIVE_GLOBAL_GATE_FAILED` and the result licenses
no positive scientific claim. A diagnostic `SUPPORTED` label can never survive as the
official disposition through a failed global gate.

No R3d result silently changes the R3c ledger. Promotion requires an immutable target
artifact, audit/replay, and an explicit claim-ledger update that retains the prior
inconclusive R3c record.
