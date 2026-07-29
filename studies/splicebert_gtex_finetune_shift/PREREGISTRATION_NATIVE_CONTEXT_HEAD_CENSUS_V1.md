# Preregistration: complete layer-5 head census for confirmed context effects

> **Tier: T2 — provenance.** Frozen before evaluating any new head outcome.
> The prior validation bridge has already exposed `L5H6`, `L5H7`, `L5H9`, and
> `L5H11` on the historical validation cohort. That cohort is therefore excluded
> from confirmation here; prior observations motivate the census but do not select
> a head.

## 1. Question and evidence boundary

GFS14–GFS16 establish two higher-than-dinucleotide sequence-order effects, localize
their integration to the final encoder block, and show that both converge on the final
MLP. The frozen coupling head `L5H6` does not explain either context effect. This study
asks:

> Which, if any, of the 16 layer-5 attention heads causally transport each confirmed
> context effect into the classified target token?

The two endpoints and corruptions remain unchanged:

- donor margin at `donor_pos - 1`, with `[-80,-48)` relative to `donor_pos`
  shuffled by exact directed-dinucleotide-preserving draw 0; and
- acceptor margin at `next_acceptor_pos`, with `[+16,+48)` relative to that
  acceptor shuffled by the same frozen algorithm and draw.

A positive result identifies a causal carrier under a one-head target-token
intervention. It does not by itself identify the source motif, establish a direct
head-to-MLP edge, or show that the selected heads form a complete/minimal circuit.

## 2. Prior exposure and fresh split

The earlier bridge measured only `{L5H6,L5H7,L5H11,L5H9}` on chromosomes
11/16/19. Those values remain part of the record but are not confirmation evidence
for this census.

The new census is restricted to the annotation-eligible chromosome-training pool.
Before assignment, every gene exposed in any of these train-stage artifacts is
excluded:

- `native_mechanism_preflight_v1.json`;
- `native_mechanism_discovery_v1.json`;
- `native_context_preflight_v1.json`; and
- `native_context_discovery_v1.json`.

Remaining genes are assigned outcome-independently using

`SHA256("gtex-native-context-head-census-v1:" + gene_id) mod 100`.

- buckets `0–69`: discovery;
- buckets `70–99`: fresh internal confirmation.

Every exon of a gene receives the same assignment. Each partition is shuffled with
its own stable tag before taking the frozen sample size:

- discovery: seed-2020 checkpoint, `n=128`;
- confirmation: all five checkpoints 2020–2024, `n=200`.

The two cohorts must have zero gene overlap, zero prior-exposed-gene overlap, and no
chromosome in `{chr8, chr9, chr11, chr16, chr19}`. The chr8/9 test and historically
touched validation chromosomes remain unopened. “Fresh confirmation” here means
unseen by this analysis program, not necessarily absent from the checkpoint's
supervised training corpus.

## 3. Instrument gate

No new head outcome opens unless all of the following recorded controls pass:

1. `native_mechanism_preflight_v1.json` is `PASS`, including exact direct-layer
   replay, attention-context reconstruction, head identity/no-op, projection
   reconstruction, and QKV-to-whole-head replay.
2. `native_context_preflight_v1.json` is `PASS`, including perturbation invariants,
   local positive controls, negative controls, source recovery, and target recovery.
3. `native_context_confirmation_audit_v1.json` licenses both complete 32-nt regions.
4. `native_context_component_bridge_v1.json` is complete and replays the preceding
   context confirmation within `2e-5` for every endpoint and seed.
5. The loaded checkpoint has exactly six layers and 16 heads, uses eager attention,
   and the clean-minus-corrupt behavioral effect on the discovery cohort has a
   gene-bootstrap interval above zero and mean at least `0.05` for an endpoint.

A failed prerequisite or behavior gate stops that endpoint. Thresholds are not
relaxed.

## 4. Complete discovery census

For every head `h ∈ {0,…,15}`, only its layer-5 context vector at the classified
target token is patched:

- **noising damage:** corrupt head into clean run,
  `clean margin - patched margin`;
- **denoising recovery:** clean head into corrupt run,
  `patched margin - corrupt margin`.

All 64 endpoint × head × direction tests form one discovery family. Uncertainty uses
5,000 gene-grouped bootstrap draws. Zero-effect tests use 9,999 gene-level sign flips
with exact add-one p-values and BH correction over all 64 tests.

Separately for donor and acceptor, a head is eligible only when:

1. damage and recovery are each at least `0.05`;
2. both 95% interval lower bounds exceed zero; and
3. both family-adjusted q-values are at most `0.10`.

Eligible heads are ranked by
`min(mean damage, mean recovery)`, with head index as tie-break, and capped at three
per endpoint. If none qualifies, that endpoint records a bounded failure of
single-head resolution and does not proceed to confirmation.

For each endpoint, three equal-size control heads per candidate are drawn
deterministically without replacement from the non-candidate heads using tag
`gtex-native-context-head-census-controls-v1:{endpoint}`. Candidate and control
identities are written to the immutable discovery artifact before confirmation.

## 5. Five-seed confirmation

Only frozen candidates and equal-size controls are evaluated on the fresh internal
confirmation cohort. The per-example five-seed mean is formed before gene-grouped
inference.

For each endpoint:

- candidate damage/recovery tests form the primary candidate family;
- candidate-minus-mean-control damage/recovery tests form a separate control-contrast
  family; and
- the two endpoint behavioral effects form a separate two-test family.

Every family uses the same 5,000 grouped bootstraps, 9,999 exact add-one gene sign
flips, and BH correction over its complete frozen membership. A candidate confirms
only when, in both directions:

1. its mean is at least `0.05`;
2. its 95% interval is above zero;
3. its family-adjusted q-value is at most `0.05`;
4. all five seed estimates are positive; and
5. candidate-minus-mean-control has an interval above zero and adjusted q-value at
   most `0.05`.

The candidate-control contrast has no post hoc minimum effect threshold; its estimate
and interval are reported in full. All raw paired values and all tested controls
remain in the artifact.

## 6. Licensed outcomes and next bridge

| Outcome | Licensed interpretation |
|---|---|
| no discovery candidate | no one-head carrier clears the frozen resolution floor |
| discovery candidate fails fresh confirmation | exploratory head effect did not generalize |
| candidate confirms | seed-reproducible causal L5 carrier of the specified context effect |
| several candidates confirm | multiple late attention routes carry that context effect |
| different donor/acceptor candidates | endpoint-specialized routes converge on the shared MLP |
| overlapping candidates | a shared head participates in both context computations |

This census does not establish circuit completeness. The immediate contingent next
experiment is joint candidate noising/denoising with matched-size random head sets,
followed by path-specific candidate-head-to-final-MLP interventions. Those tests must
not sum single-head effects as additive mediation percentages.
