# FROZEN R3c v2.1 successor — acceptor-only focused head confirmation

> **Status: FROZEN V2.1 FOR CLEAN BUCKET-0 PREFLIGHT — BUCKET 1 REMAINS SEALED.** R3c v1 produced a development-only
> bucket-0 preflight, artifact SHA-256
> `b4d9172b911588f7ba83e7441564851b0111763e3c0ffea5931403dc27d3f913`. Independent review found
> that `L2H15_mean_component` failed bucket-0 health and could nevertheless be revived by bucket-1
> health even though its v1 plant had been suppressed. No bucket-1 outcome was opened. V2 carries
> that reviewed L2H15 suppression forward, makes later live-health authorization monotone, and
> calibrates all seven paths synthetically. Target execution remains hard-blocked until this
> successor protocol and instrument are frozen, a fresh clean-worktree bucket-0 preflight passes,
> and its exact artifact-byte hash is reviewed and installed. There is no command-line bypass.

The first durable v2 preflight (artifact SHA-256
`bc5f0797201a85fa2584a644212454aeb6fb7f4f1fd5c33497f9a552a981c031`) passed every scientific,
calibration, and capability gate. Activating its literal pin exposed an authorization-order defect:
a target invocation missing `--preflight-artifact` inspected the supplied model path before failing.
No bucket-1 outcome was opened. V2.1 adds an early mandatory-argument stop before model, GTF, FASTA,
output, or cohort paths are inspected. Because runner bytes are bound, v2 cannot authorize v2.1;
the corrected guard requires a fresh clean bucket-0 artifact even though the estimands are unchanged.

## 1. Question and evidence boundary

At the model-native endpoint where SpliceBERT reconstructs the masked `G` of the downstream
intron-mate acceptor, do fixed donor-reader candidates and fixed later-head candidates make selective,
causal contributions to the margin induced by the paired upstream donor?

R3c is a component-confirmation experiment. It may support intervention-specific component and route
**candidates**. It does not establish address, payload, an acceptor-side representation, write
location, edge/path mediation, sufficiency, faithfulness, a complete circuit, or a biological
mechanism. In particular, a supported `distinct_acceptor_route_candidate` is a prespecified
route-contrast/component pattern relative to historically established donor-route comparator
identities. With no reciprocal donor endpoint in this protocol, it is not a same-protocol double
dissociation and is not itself delivery mediation.

There is exactly one readout in this experiment:

`q_A = next_acceptor_pos - 1`, the `G` of the downstream intron-mate `AG`.

The paired perturbation overwrites the upstream paired donor `GT -> AC`. The specificity control
overwrites one distance-calipered matched non-partner donor `GT -> AC` in the same example. No donor
base is masked, no donor-margin endpoint is constructed, and no forward masked-donor result enters a
gate or verdict.

## 2. Outcome-blind cohort

Before loading a model or reading any model output, restrict annotation construction to validation
chromosomes `chr11`, `chr16`, and `chr19`. Sealed chr8/9 is never fetched or opened.

Reconstruct historical R3 using validation tag `r1b-d0`, the first 1,000 structurally loaded rows,
and both legacy matched-decoy constructors. Only rows with both legacy decoys belong to historical
R3. Abort unless this reconstruction is exactly:

- 974 rows;
- 645 genes; and
- ordered example-ID SHA-256
  `161dff038d747dbaa7b91a96d24b34d45f55de5482b5140410641354e46d2a8e`.

Exclude every gene represented in that immutable historical cohort. Then exhaustively load the full
validation census under tag `r3c-acceptor-head-confirmation-v1`. Current R3c eligibility requires only
the donor-side pseudo comparator; it must not require R3b's matched acceptor pseudo. Retain a row only
when:

1. its gene is absent from historical R3;
2. `matched_decoy_donor` returns a donor distinct from the paired donor;
3. absolute paired-versus-pseudo donor-position mismatch is at most 40 nt;
4. the paired `GT -> AC` edit is admissible; and
5. the pseudo `GT -> AC` edit plants no donor or acceptor motif in its flanking context.

Partition genes with the preserved R3b salt:

`sha256("r3b-gene|" + gene_id) mod 2`.

Bucket 0 supplies the first 500 rows for preflight. Bucket 1 supplies the first 500 rows for a future
target. Both must contain at least 30 genes. The runner recomputes and aborts on drift from the
independently audited full-GTF census identity: 6,064 screened current rows; historical excluded-gene
hash `0964c06136fd9a8fa9251ad13a2ca4d45e3f0c8cea45323dbdc1fe2092b8fda0`; bucket capacities
1,010 rows/417 genes and 1,058 rows/443 genes; bucket-0 first-500 identity 284 genes and hash
`c4b7d17a14dc73b1326a2135cf3a8c49c46d5d78fa894f4879b3c574d4df07e2`; bucket-1 first-500 identity
304 genes and hash `02d4c776d911f261bdd513e3c993a18cab8373e9d242b22ccbdf2fd90f448f8f`.
The exact sorted selected-gene-list hashes are
`8745477f1fe458173774a1a905d569574ed917169621cfbfacbd7b8a65746d6d` for bucket 0 and
`484b2c4a11f47c608ea51bcaf2a4234d3197a56f6a10ffb38be9706e5fbd2c5e` for bucket 1.

Selection reads no logits, activations, attention, or prior R3c outcomes. The artifact records the
complete census, exclusions, bucket capacities, realized gene counts, ordered row hashes, and the
historical excluded-gene hash.

## 3. Fixed registry and interventions

The exact ordered registry is:

1. `L3H1`
2. `L2H15`
3. `L1H12`
4. `L4H11`
5. `L4H9`
6. `L4H7`
7. `L4H0`
8. `L4H2`
9. `L5H11`
10. `L0H2`

There are no MLPs. `L0H2` is an inherited negative/instrument comparator, not an eighth claim.

Each head receives two estimands:

1. **Within-example token-mean ablation.** Replace that head's output at every token by the same
   example's mean over tokens. Another example is never a reference.
2. **Same-example corrupt-to-clean activation noising.** Capture the full head output from the paired-
   donor-corrupt or pseudo-donor-corrupt version of one example and patch it into the aligned clean
   version of that same example. Partner and pseudo sources are separate. A minibatch roll or other
   example as source is forbidden.

## 4. Primitive and derived endpoints

Let `Y = margin(clean) - margin(paired donor corrupt)` and
`P = margin(clean) - margin(pseudo donor corrupt)` at the masked acceptor `G`.

For head `h` and mode `m`, primitive row-level component drops are:

- `partner(h,m) = Y(native) - Y(h,m)` for mean ablation, or native clean margin minus the clean
  margin after paired-corrupt head noising;
- `pseudo(h,m) = P(native) - P(h,m)` for mean ablation, or native clean margin minus the clean margin
  after pseudo-corrupt head noising.

Derive, never independently measure or plant:

`specific(h,m) = partner(h,m) - pseudo(h,m)`.

The primary head family therefore contains 10 heads × 2 modes × 3 endpoints = exactly 60 ordered
rows. Append exactly two activation-noising route rows:

`gap_e = mean(e(L4H11), e(L4H9)) - mean(e(L4H0), e(L4H2), e(L5H11))`

for `e in {partner, specific}`. The complete family has exactly 62 rows. Both gaps are derived from
the same primitive rows after any resampling or plant; they are never independently edited.

All endpoints are raw masked-logit-margin units. Ratios do not enter inference.

## 5. Estimation and strict testing

The independent unit is the gene. For all 62 rows together, construct an endpoint-scale-normalized
simultaneous whole-gene bootstrap envelope at studywise `alpha = 0.05`. The shared helper's historical
inclusive interval label and zero-boundary max statistic are retained only as diagnostics. R3c
reclassifies each interval strictly:

- material positive only when `lo > +0.05`;
- material negative only when `hi < -0.05`;
- equivalent only when `-0.05 < lo <= hi < +0.05`;
- otherwise unresolved.

Touching a SESOI boundary is unresolved.

Decision-bearing p-values use exact add-one whole-gene boundary sign flips:

- a positive constituent tests `H0: mean <= +0.05` against `H1: mean > +0.05`;
- an equivalence constituent uses TOST at `-0.05` and `+0.05`, with IUT p-value equal to the maximum
  of the two one-sided p-values.

Every decision-bearing positive edge and both edges of every TOST reuse one identical 4,000-draw
Rademacher matrix: one sign per ordered gene, shared across every endpoint row. The artifact records
the namespace, sign-matrix SHA-256, ordered-gene-label SHA-256, edge order, edge-specific null-draw
hashes, exceedance counts, add-one floor, and `B=4000`.

The observed statistic is computed from per-example boundary-centered values. A conservative
float64 tolerance,
`eps * max(128,n_examples) * max(1,max|centered values|,max|oriented null|,|statistic|)`, snaps only
machine-scale zero statistics to zero and counts `oriented_null >= statistic - tolerance`. Thus an
exact closed-null boundary cannot become a false alternative because of summation rounding; this
rule can only include additional numerical ties in the null tail.

For one named path, the IUT p-value is the maximum constituent p-value. Apply Holm across exactly the
following seven ordered path claims:

1. `L3H1_reused_component`: full partner-positive, pseudo-equivalent, specific-positive conjunction
   under both modes;
2. `L2H15_mean_component`: that full conjunction under mean ablation;
3. `L1H12_mean_component`: same-form secondary claim;
4. `L4H11_noising_component`: full conjunction under activation noising;
5. `L4H9_noising_component`: full conjunction under activation noising;
6. `L4H7_noising_component`: same-form secondary claim; and
7. `distinct_acceptor_route_candidate`: full H11 and H9 noising conjunctions, both positive route-gap
   rows, and partner/pseudo/specific equivalence under noising for each of `L4H0`, `L4H2`, and
   `L5H11`.

A path is `SUPPORTED` only when its Holm-adjusted IUT p-value is strictly less than 0.05, every
constituent has the matching strict simultaneous-CI classification, and all involved component/mode
health checks pass. A p/CI disagreement is `INCONCLUSIVE_METHOD_DISAGREEMENT`. Health failure
suppresses only that component/mode and paths that depend on it; it cannot suppress unrelated paths.

## 6. Exact production-path preflight

Before any outer trial, subtract each observed mean from the 40 primitive partner/pseudo rows. Apply
one sign per gene, shared across every primitive row and all examples from that gene. Derive all 20
specific rows and both gaps only after the outer flip. This makes calibration invariant to observed
endpoint translations and preserves every algebraic relation.

Every outer case invokes the exact production classifier: 62-row endpoint-scaled simultaneous
envelope, strict boundary/TOST tests using one shared Rademacher matrix, constituent IUT, Holm across
all seven paths, strict CI concordance, and fixed local health suppression.

Global path-family validity is analytic, not an empirical claim that one chosen nuisance midpoint is
the global least-favourable configuration. Under any true named-path null, at least one constituent
positive-boundary null or TOST union-null is true. Its boundary p-value is super-uniform; taking the
maximum constituent p-value preserves validity for the path IUT; and Holm strongly controls FWER
across the seven named paths under arbitrary dependence. Requiring strict simultaneous-CI
concordance and health can only remove a declaration, never create one.

The outer simulations are therefore called **boundary-face end-to-end diagnostics**. They exercise
the complete implementation on prespecified coherent faces but do not claim that a single nuisance
midpoint maximizes declaration probability over the full coherent alternative continuum. Every
assigned path has exactly one constituent or one TOST edge on its null; every other conjunct of that
path is strictly inside its alternative. The isolated component faces use primitive `(partner,
pseudo)` values:

- partner boundary: `(.05, -.025)`, giving specific `.075`;
- specific boundary: `(.075, .025)`, giving specific `.05`;
- pseudo lower edge: `(.40, -.05)`, giving specific `.45`; and
- pseudo upper edge: `(.40, .05)`, giving specific `.35`.

L3H1 cycles all four faces separately in each mode. L2H15 mean, secondary L1H12 mean, H11 noising,
H9 noising, and secondary H7 noising each cycle all four. The two isolated gap faces are:

- partner gap: candidates `(.075, -.025)`, comparators `(.025, 0)`; and
- specific gap: candidates `(.10, .025)`, comparators `(.025, 0)`.

For each of L4H0, L4H2, and L5H11, all six comparator TOST edges are isolated with `(partner,
pseudo)` equal to `(-.05,-.025)`, `(.05,.025)`, `(-.025,-.05)`, `(.025,.05)`,
`(-.025,.025)`, and `(.025,-.025)` for partner-lower, partner-upper, pseudo-lower,
pseudo-upper, specific-lower, and specific-upper respectively.

The distinct route itself therefore requires 28 algebraically distinct faces: eight H11/H9 faces,
two gap faces, and eighteen comparator edges. Twelve path-disjoint L3/L2/L1/H7 faces are packed into
the first twelve route faces. Packing is permitted only because the runner asserts that each
assigned named path still has exactly one null constituent/edge and all its remaining conjuncts are
strict alternatives. Thus the frozen suite remains exactly 28 surfaces rather than 40 ungrouped
surfaces, with no distributional symmetry assumption.

Each surface has 200 independently namespaced trials. Within one surface, a false positive is any
assigned null path declared `SUPPORTED` by the exact seven-path classifier. Surfaces are alternative
DGP stress tests, not one simultaneous inferential family, so they are not unioned across
counterfactual surfaces. Every surface independently requires its one-sided 95% Clopper--Pearson
upper bound to be at most .05 (at 200 trials, at most four false declarations). The artifact records
all 200 compact classifier audits per surface, shared-sign and ordered-gene digests, per-path counts,
offset and record hashes, namespace templates, and the deterministic worst-surface summary.

The coherent sensitivity plant is frozen at `delta = 0.40`. In primitive space it adds `+0.40` only
to the partner row for each positive component/mode and leaves every pseudo and route comparator at
zero. Thus every positive specificity and both route gaps derive to exactly `+0.40`. Per-path planted
power is recorded with all seven synthetic paths health-enabled, regardless of observed live health.
This makes the engineering false-positive and power preflight no easier when a live path happens to
fail health. Every path must have a one-sided 95% Clopper--Pearson power lower bound of at least 0.90.
Live health is applied separately and can only suppress a scientific declaration.

With 200 plant trials, every path needs at least 188 successes for its one-sided 95%
Clopper--Pearson lower bound to reach 0.90. Across 28 boundary surfaces plus one coherent plant, the
production preflight invokes the exact production classifier 5,800 times. Diagnostic or plant
failures cannot be overridden.

## 7. Health and assay controls

For at most the first 16 rows, choose four deterministic non-site probes per row, excluding the
acceptor readout and both donor-edit neighborhoods. Each probe is masked on byte-aligned source and
target encodings.

- Mean-ablation health uses the clean source policy.
- Noising health separately uses paired-donor-corrupt and pseudo-donor-corrupt sources.
- Top-1 is restricted to A/C/G/T token IDs.
- Every relevant source policy must reach accuracy 0.50.

The candidate heads cannot serve as instrument controls. Bucket 0 instead runs an independent live-
hook battery: clean-to-identical-clean patching of every head reproduces full logits and the acceptor
margin within `1e-5`; split then merge of captured post-attention head contexts is exact at every
layer; clean/corrupt encodings preserve same-example row order and differ only at the declared donor
tokens; the control outputs agree between batch size 1 and the frozen production grouping within
`1e-5`; and the captured final hidden state passed through the checkpoint's complete `model.cls`
reproduces full MLM logits and the acceptor-G margin within `1e-5`.

For live sensitivity, all 16 layer-3 heads are patched together from clean into the aligned paired-
donor-corrupt and pseudo-donor-corrupt target. This aggregate hook—not L3H1 or any other candidate—
must show partner restoration strictly above +0.20 and pseudo restoration strictly within ±0.05 by
the same shared-sign boundary/TOST and strict simultaneous-CI conjunction. Explicit value-product
`A@V` reconstruction is deferred to R5a; R3c memorializes exact post-attention head-context
split/merge only.

Preflight also requires at least 30 genes, native paired-donor coupling above +0.20, absolute native
pseudo coupling below 0.05, exact acceptor-G masking, paired and pseudo `GT -> AC` targets, no planted
motifs, nonzero health trials, the exact 40/62 row registries, passing boundary-face diagnostic and
planted-power gates, and the independent live-hook battery. A target inherits the exact pinned
bucket-0 capability record. L3H1, L2H15, H11, H9, and every other candidate may fail to confirm on
bucket 1 without turning that scientific negative into instrument failure.

Bucket-0 live health freezes the maximum set of paths that may be interpreted on target. Effective
target health is the conjunction of reviewed bucket-0 path health and current bucket-1 path health.
A bucket-1 health failure may remove a path, but a bucket-1 pass can never revive a path that failed
bucket-0 health. The preflight and target artifacts serialize the authorized and suppressed path
lists and the exact conjunction rule. Diagnosing and reviving a preflight-suppressed path requires
another prospective successor; it cannot happen inside the held-out run.

Because v2 was designed after observing the v1 `L2H15_mean_component` health failure (31/64 probes),
the v2 ceiling conservatively keeps that path suppressed even if the repeated deterministic v2
health sample happens to cross 32/64. The repeated measurement is a reproducibility diagnostic, not
a retry that can restore authorization. `L4H2` mean health also failed 31/64 in v1, but no named path
depends on that mode; the distinct-route path depends on L4H2 noising, which passed.

The preflight gate map is exact and ordered in the runner. There is no global maximum-width gate;
strict constituent classifications and exact path calibration are the resolution standard.

## 8. Fixed production parameters

| Setting | Value |
|---|---:|
| rows per bucket | 500 |
| batch | 32 |
| seed | 0 |
| boundary/bootstrap draws per classifier | 4,000 |
| outer null/plant meta-trials | 200 |
| coherent boundary-face surfaces | exactly 28 |
| production classifier calls | exactly 5,800 |
| SESOI | 0.05 margin units |
| Holm family | exactly 7 paths |
| coherent plant delta | 0.40 margin units |
| donor-pseudo distance caliper | 40 nt |
| minimum independent genes | 30 |
| bounded health rows/probes | 16 / 4 each |

Smoke may use smaller mechanics-only counts and cannot bind a target.

A pre-bucket-0 synthetic benchmark used the exact 40-row/500-example/284-gene algebra and all 28
surfaces, with 400 draws and two trials (58 exact classifier calls). Three repetitions took a median
0.4005 seconds on the current bound runtime. Linear scaling in draws and trials estimates about 400.5
seconds (6.67 minutes) for the 5,800-call production calibration. Serialization at one and two trials
was 213,173 and 357,697 bytes; the fitted fixed-plus-per-trial estimate is 28,973,449 bytes (27.6 MiB)
for the production calibration record. This is a planning estimate, not a gate, and excludes model
execution. The runner's exact accounting predicts 1,110 model forward calls for the bucket-0
measurement, health, and independent capability battery before calibration.

## 9. Provenance, immutability, and target seal

The instrument fingerprint binds the complete checkpoint inventory; exact GTF and FASTA; this
protocol; seal-normalized runner; statistics, hook, adapter, construction, dataset, and provenance
sources; ordered head/primitive/family/path registries; strict test and interval policies; the
complete ordered boundary-face offset/hash suite and coherent plant; capability policy; health
policy; fixed counts; historical and current census
identities; `r3b-gene` partition salt; and runtime/library identity.

Production stages snapshot all bound inputs by resolved path, byte count, and SHA-256, stage a private
byte-identical checkpoint copy, verify inputs again before publication, and write a new artifact plus
sibling manifest with exclusive-create semantics. Existing output or manifest paths are never
overwritten. Compact sorted gene-level row counts and raw component sums reproduce every row-weighted
mean and whole-gene statistic without committing activation captures. Exact sorted gene lists and
hashes are exposed for the historical exclusion and both fixed R3c buckets so later studies can
enforce cohort exclusion directly.

A development preflight run while the protocol or instrument pin is absent has status
`DRAFT_NOT_FROZEN` and can never authorise bucket 1. Under this reviewed freeze, target requires a
fresh clean-worktree bucket-0 preflight whose exact binding payload, fingerprint, gate family,
component order, 62 rows, seven paths, and capability record match the target instrument. The
reviewed preflight artifact bytes then receive a third literal SHA-256 pin, excluded from runner-hash
normalization exactly like the other seal literals. An editable artifact plus an editable sibling
manifest is never sufficient.

On target, the runner validates the protocol and all three literal seals and requires the
`--preflight-artifact` argument before inspecting any caller-supplied model, GTF, FASTA, output,
cohort, or preflight path. There is no later-path fallback for a missing authorization input.

At binding, observed 62-row and two-row capability inference are replayed from exact gene counts and
raw cluster sums through the frozen production classifier, including CI, boundary p-values, Holm,
and shared sign digests. Calibration records are checked trial by trial for canonical namespaces,
shared-sign metadata, Holm algebra, declarations, per-surface counts and Clopper--Pearson bounds,
plant successes, and aggregate hashes; the literal artifact hash pins the residual simulation bytes.
Target inherits bucket-0 synthetic calibration and capability, then reruns the current-bucket model
and construction invariants. It applies the serialized bucket-0 live-health authorization ceiling
before current-bucket inference. At protocol freeze no target run is authorised; bucket 1 remains
sealed until the reviewed preflight artifact receives its literal pin.

## 10. Interpretation table

| Frozen outcome | Licensed wording |
|---|---|
| L3H1 path supported | L3H1 is a confirmed acceptor-margin component under both declared estimands. |
| L2H15 mean path supported | L2H15 is a confirmed mean-ablation component; an address role still requires QK intervention. |
| H11/H9 paths and distinct path supported | Within the tested registry, the fixed noising candidates show a prespecified route-contrast/component pattern relative to the historically established donor-route comparator identities; reciprocal double dissociation and delivery mediation remain open. |
| Any p/CI disagreement | That path is `INCONCLUSIVE_METHOD_DISAGREEMENT`: scientifically unresolved under the frozen decision rule, not an instrument failure. |
| Health failure | Only the affected component/mode and dependent paths are unavailable. |
| Calibration failure | No positive, negative, or equivalence component conclusion is licensed. |

Even the strongest joint outcome remains a candidate map for the next causal stages: acceptor-query
QK/address tests, donor-value-to-acceptor write reconstruction, acceptor-side subspace necessity and
sufficiency, and downstream path mediation with matched null heads.
