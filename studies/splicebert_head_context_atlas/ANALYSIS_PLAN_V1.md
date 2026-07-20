# Head–Context Atlas analysis plan v1

Status: **FROZEN BEFORE THE FULL DISCOVERY MATRIX WAS READ** (2026-07-19).

This plan turns the 102-component discovery matrix into two bounded products:

1. an atlas of **general-use** versus **context-associated** component behavior; and
2. a small, mechanically testable shortlist for QK/OV/MLP/group/DAG validation.

It does not assign biological functions from screen ranks. Competition is not a primary category; the
closed competition experiment remains a secondary historical lens.

## Inputs and split boundary

- Target matrix: one row per unique discovery packet × component, carrying A and B effects.
- Health matrix: a deterministic, role-balanced 512-packet panel with genuinely masked non-site probes.
- Pair table: 6,487 matched discovery case/control pairs across ten role-valid contrasts.
- Components: all 96 attention heads and six MLP residual writes.
- Internal validation is physically separate and **must not be loaded** until the discovery shortlist and
  its exact intervention package are frozen. chr8/9 remains sealed.

Every packet enters the general-use census once, even if it belongs to several overlapping context
contrasts. Context estimates use the pair table; labels are never made mutually exclusive.

## Model behaviors

For packet `i` and component `h`:

- `A_ih = margin(native) − margin(native with h zeroed)` — causal necessity for native masked-site
  reconstruction.
- `B_ih = margin(native) − margin(native with h scrubbed to its partner-destroyed state)` — component
  mediation of donor–acceptor partner dependence.

Positive values support the native readout; negative values suppress it. A and B are separate causal
cuts and are never added. Single-component effects are not called circuit faithfulness.

The frozen practical-effect floor is `τ = 0.05 nats`. This is not imported from a prior head: it is the
model-native log-odds shift `log(1.05)=0.0488`, rounded to 0.05—approximately a 5% multiplicative change
in native-versus-best-alternative odds. A different floor may be shown only as a labeled sensitivity
curve; it cannot replace the primary classification after outcomes are read.

## 1. General-use analysis

Analyze the unique packet union, separately for `{A,B} × {donor,acceptor}`. For each component:

- point estimate = unweighted packet mean within role;
- uncertainty = one-way gene-clustered sandwich interval (all packets from one gene form a cluster);
- discovery family = 102 components × two target roles, BH-FDR separately for A and B;
- MDE = `(z_0.975 + z_0.80) × clustered SE`.

Role-level labels:

- `material_support`: CI lower bound `> +τ` and BH `q≤0.05`;
- `material_suppress`: CI upper bound `< −τ` and `q≤0.05`;
- `detected_subfloor`: CI excludes zero / `q≤0.05`, but not the practical floor;
- `bounded_small`: the entire CI lies inside `[−τ,+τ]`;
- otherwise `inconclusive`.

Cross-role component profile:

- same material label in both roles → `cross_role_general`;
- material in only one role → `role_associated`;
- material with opposite signs → `role_opposed`;
- otherwise unresolved.

Masked-health A/B effects are summarized with the same gene-clustered estimator. They annotate whether a
component also affects generic nucleotide reconstruction (`broad_MLM_flag`); they do not post-treatment
adjust or silently delete a target effect. Target and health effects remain visible side by side.

## 2. Context-association analysis

For focal context `k`, target role `r`, component `h`, and matched pair `p`:

`D_pkrh = E_h(case_p) − E_h(control_p)`, for `E∈{A,B}`.

Cases carry `k`; controls lack `k`; target role, strand, region, distance/phase/support strata, and all
non-focal core tags are exact. Continuous strength, distance, composition, motif, packet-position, and
≈10.63 geometry covariates passed maximum absolute SMD ≤0.10 before model inference.

- Controls used twice receive inverse-reuse weights within their contrast.
- Point estimate is the weighted mean pair difference.
- Standard errors use two-endpoint gene-dyadic clustering. With weighted residual score `s_p` and
  incident-gene score `S_g=Σ_{p incident on g}s_p`, the sandwich meat is
  `Σ_g S_g² − Σ_p s_p²`; this retains one observation-level diagonal while allowing dependence through
  either gene endpoint.
- Discovery family = all ten contrasts × 102 components, BH-FDR separately for A and B.
- Report effective pair count, dyadic SE/CI, MDE, raw case/control means, and match/reuse diagnostics.

Stability is a distinct gate, not another seed called a “replication.” A frozen hash partitions genes
into two disjoint halves; a pair enters a half only when **both** endpoint genes are in it. A shortlist
association requires at least 40 within-half pairs and the same point-estimate direction in both halves.
Cross-half edges are dropped. If support is insufficient, status is `stability_instrument_limited`, not a
negative result.

Context labels use the same material/subfloor/bounded/inconclusive rules as the general analysis. A
`context_associated` discovery candidate additionally requires full-cohort materiality, BH `q≤0.05`, and
the two-half direction gate.

## 3. Discovery shortlist (frozen rule)

Nothing is shortlisted by raw rank alone.

### General candidates — maximum four

Eligible components are `cross_role_general` for A or B, have MDE≤τ in both roles, and do not fail the
masked-health instrument. Rank by the smaller role-specific amount by which the confidence bound clears
`τ`. Retain at most four unique components. Do not force an MLP/head quota when no component qualifies.

### Context candidates — maximum eight

Eligible associations are `context_associated`, have effective pair count ≥80 and MDE≤τ. Rank by the
signed confidence-bound excess beyond `τ`. Greedily retain at most eight unique component–context
associations, with at most two per contrast and at most three associations for one component. Preserve
donor and acceptor coverage when eligible candidates exist in both roles; never invent coverage.

L3H1/B is a pre-existing donor-partner positive control and is reported independently. It neither gains
automatic atlas status nor consumes a shortlist slot unless it meets the same frozen rule.

The shortlist JSON must include input hashes, this plan's hash, every eligible candidate, the deterministic
tie-break, exclusions, and selected order before internal validation is made accessible.

## 4. Mechanistic validation ladder

For each shortlisted attention head:

1. exact single and ranked-group patches (never sum single-head effects);
2. QK routing and OV writing **main effects**, with source-identity and importance-matched controls;
3. downstream MLP/path clamps where the effect requires later residual writes;
4. representative attribution DAGs for a deterministic median-effect matched case/control pair, plus one
   negative-control pair—not the most visually dramatic example.

For MLP candidates: exact write patch, upstream attention dependence, downstream path clamp, and a
matched head-group control. Group validation measures joint behavior and recovery relative to a
non-negligible behavioral denominator.

Internal validation repeats only the frozen shortlist family with gene-dyadic inference. Results may be
material, detected-subfloor, bounded-small (only if the CI fits inside ±τ), or inconclusive. A discovery
association that fails validation is retired, not renamed.

## 5. Visualization boundary

`gfm-visualizer` receives only provenance-pinned, validated component profiles, uncertainty, matched
examples, interventions, and DAGs. Discovery-only ranks remain available in the experiment repository but
are visibly labeled exploratory and are not exported as head-function glossary entries.
