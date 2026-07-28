# DNABERT-2 Reconstruction Atlas results index

> **Tier: T1 — canonical for result status.**

## Scientific results

| Run | Question | Status | Artifact | Manifest | Claim IDs |
|---|---|---|---|---|---|
| GENCODE behavior development v1 | Does any admitted class exceed all empirical and low-order controls? | development gate failed for every class; confirmation sealed | `../../results/dnabert2_reconstruction_atlas/behavior_development_v1.json` | `../../results/dnabert2_reconstruction_atlas/behavior_development_v1.manifest.json` | D1 |
| Multisource behavior development v4 | Do promoter, poly(A), TFBS, branch-point, PPT, enhancer, splice, or translation classes exceed every control? | development gate failed for every class; five powered, eight underpowered; confirmation sealed | `../../results/dnabert2_reconstruction_atlas/behavior_development_multisource_v4.json` | `../../results/dnabert2_reconstruction_atlas/behavior_development_multisource_v4.manifest.json` | D2 |
| Context-margin atlas v1 | Does a class-specific, target-preserving proximal-context corruption reduce DNABERT-2's geometry-qualified alternative margin more than an equal-dose distal corruption? | nine classes replicated; promoter did not replicate; stop failed development; acceptor/donor preflight-unresolved | `../../results/dnabert2_reconstruction_atlas/context_margin_v1_replication.json` | `../../results/dnabert2_reconstruction_atlas/context_margin_v1_replication.manifest.json` | D3 |

### Replicated context-margin atlas v1

The frozen assay scored 2,059 development targets across 11 preflight-admitted classes, then
opened only ten development-qualified classes on 1,781 untouched confirmation targets.

| Site class | Dev n | Dev effect | Confirmation n | Confirmation effect (95% CI) | Holm p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| start codon | 219 | +0.0468 | 222 | +0.0479 (+0.0306, +0.0673) | 0.0010 | replicated |
| distal enhancer anchor | 85 | +0.0395 | 98 | +0.0384 (+0.0101, +0.0730) | 0.0144 | replicated |
| poly(A) cleavage | 112 | +0.0298 | 108 | +0.0348 (+0.0125, +0.0601) | 0.0068 | replicated |
| branch point | 145 | +0.0412 | 138 | +0.0304 (+0.00443, +0.0585) | 0.0270 | replicated |
| REST peak anchor | 208 | +0.0136 | 230 | +0.0213 (+0.0126, +0.0302) | 0.0010 | replicated |
| CEBPB peak anchor | 216 | +0.0215 | 213 | +0.0183 (+0.00795, +0.0295) | 0.0035 | replicated |
| CTCF peak anchor | 217 | +0.0141 | 198 | +0.0159 (+0.00643, +0.0255) | 0.0036 | replicated |
| poly(A) signal | 241 | +0.0127 | 245 | +0.0127 (+0.00555, +0.0208) | 0.0036 | replicated |
| polypyrimidine tract | 250 | +0.0109 | 228 | +0.0125 (+0.00830, +0.0172) | 0.0010 | replicated |
| promoter TSS anchor | 85 | +0.0247 | 101 | +0.00066 (−0.0117, +0.0126) | 0.4548 | not replicated |
| stop codon | 281 | +0.00330 | — | — | — | development failed |

Effects are nats per complete BPE-covered nucleotide:
`margin(distal equal-dose corruption) − margin(class-specific proximal corruption)`.
Splice acceptor and donor were not target-scored because their margin positive controls failed
before outcomes. The nine replicated rows license a DNABERT-2 proximal-context behavior, not a
cellular functional claim or an internal mechanism. For TF/enhancer rows the five-nucleotide
target remains a derived peak/cCRE anchor, not a resolved motif. The distal control is exactly
matched for width, composition, and changed-base count but not target distance; this bounds the
claim to proximal-context dependence.

### Development behavior census v1

The frozen, qualified development screen scored 1,725 matched targets across 24
class × contrast tests with add-one permutation values and full-family Holm correction.

| Site class | Clusters | Same motif | Composition/token | Same sequence | N-gram range | Development verdict |
|---|---:|---:|---:|---:|---:|---|
| splice acceptor | 501 | −0.110 | −0.106 | −0.099 | +0.089 to +0.156 | not qualified |
| splice donor | 451 | −0.032 | +0.023 | −0.023 | +0.186 to +0.225 | not qualified |
| start codon | 354 | −0.068 | −0.080 | −0.042 | +0.106 to +0.126 | not qualified |
| stop codon | 419 | −0.130 | −0.117 | −0.123 | +0.079 to +0.110 | not qualified |

Effects are nats per complete BPE-covered nucleotide. Every n-gram contrast passes the
0.02 SESOI and full-family correction; no class passes its empirical families. D1 is therefore
a bounded development result. Confirmation remains sealed, no `NULL` verdict is issued, and
P4–P6 are stopped.

### Multisource development behavior census v4

The frozen, qualified development screen scored 3,189 complete matched targets across 78
class × contrast tests with 9,999 add-one permutations and Holm correction across the full
family.

| Site class | Clusters | Power | Same motif | Composition/token | Same sequence | N-gram range | Verdict |
|---|---:|---|---:|---:|---:|---:|---|
| distal enhancer core | 210 | underpowered | −0.012 | −0.070 | −0.020 | +0.191 to +0.241 | not qualified |
| poly(A) cleavage region | 383 | powered | −0.068 | −0.080 | −0.069 | +0.144 to +0.174 | not qualified |
| poly(A) signal | 198 | underpowered | −0.044 | −0.037 | −0.015 | +0.138 to +0.197 | not qualified |
| EPD promoter TSS core | 222 | underpowered | −0.021 | −0.023 | −0.043 | +0.133 to +0.170 | not qualified |
| splice acceptor | 373 | powered | −0.048 | −0.078 | −0.066 | +0.118 to +0.177 | not qualified |
| branch point | 384 | powered | −0.092 | −0.123 | −0.103 | +0.131 to +0.156 | not qualified |
| splice donor | 316 | powered | −0.073 | −0.080 | −0.057 | +0.156 to +0.176 | not qualified |
| polypyrimidine tract | 87 | underpowered | −0.004 | +0.004 | +0.020 | +0.124 to +0.246 | not qualified |
| CEBPB peak core | 196 | underpowered | −0.012 | −0.087 | −0.087 | +0.231 to +0.251 | not qualified |
| CTCF peak core | 134 | underpowered | +0.001 | −0.066 | −0.037 | +0.178 to +0.212 | not qualified |
| REST peak core | 146 | underpowered | −0.082 | −0.014 | −0.066 | +0.175 to +0.223 | not qualified |
| start codon | 244 | underpowered | −0.099 | −0.087 | −0.098 | +0.060 to +0.089 | not qualified |
| stop codon | 296 | powered | −0.150 | −0.127 | −0.117 | +0.077 to +0.100 | not qualified |

Effects are nats per complete BPE-covered nucleotide. Every n-gram contrast passes the +0.02
SESOI and full-family correction. No empirical contrast passes both gates. The largest positive
empirical estimate, PPT versus same-sequence non-site (+0.020), has add-one `p=0.2445` and
Holm `p=1.0`. D2 is therefore a bounded development result, not a confirmatory null. The
confirmation runner refuses to open because no class development-qualified; P4–P6 are stopped.

## Infrastructure verification

The test suite verifies coordinate schemas, BPE cover selection, masking, scoring normalization,
alternative eligibility, empirical-family separation, context invariants, gate failure behavior,
add-one inference, matched-size causal nulls, adapter unpadding maps, and immutable writes.
Infrastructure tests license only the IF rows in `CLAIMS.md`.

### Phase-A bounded collection qualification

`results/dnabert2_reconstruction_atlas/collection_qualification_v1.json` records 5,120 target
examples and 2,560 baseline-only examples with zero cluster overlap. Under the frozen
one-control-per-family supply design, splice acceptor (501/503 development/confirmation
clusters), splice donor (451/445), start codon (354/350), and stop codon (419/393) enter
preflight calibration. Kozak (5/6) is quarantined before model outcomes. The rejected
three-control supply audit is preserved as a bound comparison in the same artifact.

This is a collection/instrument result only. It licenses IF6 and no biological behavior claim.

### Preflight calibration v1 — instrument unresolved

`results/dnabert2_reconstruction_atlas/preflight_calibration_v1.json` used 64 development
clusters per supply-qualified class and no empirical target-control contrast. Geometry, matching
balance, split invariants, finite scores, deterministic replay, and batch invariance passed.

The positive control did not qualify. Four-copy effects were +0.0051 acceptor, −0.0043 donor,
+0.0089 start codon, and +0.0088 stop codon nats per covered nucleotide; only stop codon was
strictly dose-monotone. The v1 plant was randomly located anywhere downstream after a ≥32-nt
buffer, so it mixed sensitivity with arbitrary-distance context use. Status is
`INSTRUMENT_UNRESOLVED`; interpretation is frozen while placement is diagnosed. This licenses
IF7 and no behavior claim.

### Preflight calibration v2 — placement validated, gate implementation amended

`results/dnabert2_reconstruction_atlas/preflight_calibration_v2.json` used the preregistered
nearest eligible placement. Four-copy effects were +0.090 acceptor, +0.161 donor, +0.113 start
codon, and +0.206 stop codon nats per covered nucleotide. Every dose curve increased from one
to two to four copies; donor was labeled non-monotone only because the implementation also
required the one-copy mean to be positive, a requirement not present in the draft.

The same review found that sensitivity/resolution were computed from heterogeneous positive-effect
variance. Before freezing, calibration v3 uses the irrelevant-exchange null-noise MDE and CI
half-width for those gates, while retaining the positive effect and dose ordering for the
sensitivity positive control. V2 remains development-only diagnosis and licenses IF8, not behavior.

### Preflight calibration v3 — lock calibration complete

`results/dnabert2_reconstruction_atlas/preflight_calibration_v3.json` corrects the two v2 gate
definitions without changing any sequence or score. All four classes have ordered copy doses and
four-copy effects above +0.09 nats per covered nucleotide. Irrelevant-exchange null-noise MDEs
range 0.0077–0.0139 and CI half-widths 0.0051–0.0096.

These controls freeze `preflight_v1.frozen.json`: positive floor 0.05, negative equivalence
0.02, MDE ceiling 0.015, CI-half-width ceiling 0.01, behavior SESOI 0.02, and 350 clusters
per split. This licenses IF9. It still does not authorize target execution until the same
instrument reruns against the frozen config and emits `QUALIFIED`.

### Frozen preflight v1 — qualified

`results/dnabert2_reconstruction_atlas/preflight_qualified_v1.json` reruns calibration v3
against the frozen config hash and exact collection, match, matching-audit, and baseline hashes.
All eight mandatory gate families pass and the report decision is `QUALIFIED`.

This licenses IF10 and authorizes the development behavior screen. It is still an instrument
result, not evidence for H1.

### Multisource collection qualification v2–v4

V2 adds 13 frozen classes from GENCODE v50, EPDnew human v006, PolyASite 2.0 GRCh38.96,
UCSC’s multi-experiment ENCODE TF ChIP clusters, Mercer high-confidence lariat-seq branch points,
and ENCODE Registry v4 GRCh38 cCREs. Its seven-nucleotide promoter, enhancer, and TF centers fail
outcome-blind exact-control supply and remain preserved in
`collection_qualification_v2.json`.

V3 changes only those broad-interval assay anchors to symmetric five-nucleotide centers and
prioritizes local control candidates by composition, making all 13 classes preflight-eligible.
V4 expands that same candidate search from 128 to 512 positions without changing targets,
controls, calipers, or statistical gates. It retains the same 6,448 complete matched targets,
showing that exact BPE geometry—not candidate depth—limits the remaining supply. Five classes meet
the 250-locus behavior floor on both splits; eight are frozen as underpowered. These are
collection facts (IF12–IF14), not biological outcomes.

### Frozen multisource preflight v4 — qualified

`preflight_qualified_multisource_v4.json` binds the exact v4 collection, matches, matching audit,
baseline corpus, frozen config, and pinned checkpoint. Every class has 64 preflight-eligible
examples. All graded-copy curves are monotone and their four-copy means range from +0.082 to
+0.226 nats per covered nucleotide. All negative-equivalence, sensitivity, resolution, balance,
split, coordinate, geometry, replay, batch, finiteness, and model-health gates pass.

This licenses IF15 and authorized the multisource development screen. It is not biological
evidence by itself.

Bulk runtime collections and per-site shards will remain ignored. Every settled compact result must
be immutable, indexed here, and accompanied by a manifest containing checkpoint, data inputs,
parameters, seeds, code revision, and command.
